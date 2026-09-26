"""One wake-up call attempt, end to end:

    CALLING → RINGING → ANSWERED → GREETING → CONVERSING ⇄ WAKE_VERIFICATION
            → AWAKE_CONFIRMED → GOODBYE → (hangup) → attempt result

Owns the safety rails: ring timeout, max call duration, silence nudges, dead-audio detection,
LLM/TTS failure fallbacks, and it ALWAYS hangs up in a finally block.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from agents.prompts import SUMMARY_PROMPT, VOICE_TOOLS, custom_instructions, voice_instructions
from agents.state_machine import S, WakeStateMachine
from services.audio.bridge import AudioBridge
from services.config import WakeConfig, mask_phone
from services.observability import event
from services.store import Store, now_iso
from services.voice.base import (
    AgentTurnDone,
    AgentTurnStarted,
    ProviderError,
    RealtimeVoiceProvider,
    ToolCall,
    UserSpeechStarted,
    UserTranscript,
)
from services.whatsapp.base import CallState, NotAuthenticated, WhatsAppError, WhatsAppProvider

log = logging.getLogger("wake.call")

RING_TIMEOUT_S = 70             # WaCalls itself gives up at 60 s; this is our backstop
FIRST_AUDIO_TIMEOUT_S = 10      # answered but no inbound PCM at all → audio leg is dead
AUDIO_STALL_S = 15              # inbound PCM stopped mid-call
MAX_CONSECUTIVE_AGENT_ERRORS = 4

# Deterministic safety net for "stop": honoured even if the LLM never emits end_call (sim:
# the agent said "Okay, I'll stop", was cut off before its marker, and a retry would have followed).
STOP_RE = re.compile(
    r"\b(stop (calling|it|this)|don'?t call|do not call|stop the call|never call|quit calling|"
    r"hang up|go away)\b|बंद कर|मत कर|कॉल मत|फ़ोन मत|फोन मत|call mat|band karo|mat karo",
    re.I,
)

FALLBACK_LINES = [
    "Hey, wake up! Come on, sit up for me.",
    "Rise and shine! Are you sitting up yet?",
    "Still there? Say something so I know you're up.",
    "Hello! Open your eyes, it's morning.",
]


@dataclass
class AttemptResult:
    state: S                         # COMPLETED or a failure state
    call_id: str | None = None
    end_reason: str | None = None    # WaCalls end reason
    answered: bool = False
    wake_confirmed: bool = False
    ended_by: str | None = None      # agent | her | timeout | error
    stop_requested: bool = False     # she asked us to stop: never retry
    duration_s: float = 0.0
    user_turns: int = 0
    agent_turns: int = 0
    barge_ins: int = 0
    evidence: dict = field(default_factory=dict)
    summary: str = ""
    detail: str = ""

    def brief(self) -> dict:
        return {
            "state": self.state.value, "answered": self.answered, "wake_confirmed": self.wake_confirmed,
            "ended_by": self.ended_by, "end_reason": self.end_reason, "stop_requested": self.stop_requested,
            "duration_s": round(self.duration_s), "user_turns": self.user_turns, "agent_turns": self.agent_turns,
            "evidence": self.evidence, "summary": self.summary, "detail": self.detail,
        }


class WakeCall:
    def __init__(
        self,
        *,
        wa: WhatsAppProvider,
        voice_factory: Callable[[], RealtimeVoiceProvider],
        cfg: WakeConfig,
        store: Store,
        run_id: int,
        attempt: int,
        sm: WakeStateMachine,
        phone: str | None = None,
        test_mode: bool = False,
        opening_override: str | None = None,
        custom_objective: str | None = None,
        show_transcript: bool = False,
        language: str | None = None,
        max_minutes: float | None = None,
    ):
        self.wa, self.voice_factory, self.cfg, self.store = wa, voice_factory, cfg, store
        self.run_id, self.attempt, self.sm = run_id, attempt, sm
        self.phone = phone or cfg.contact_phone
        self.test_mode = test_mode or bool(custom_objective)
        self.opening_override = opening_override
        self.custom_objective = custom_objective
        self.show_transcript = show_transcript  # console only; never written to logs or the DB
        self.language = language
        self.max_minutes = max_minutes
        self.call_id: str | None = None
        self.voice: RealtimeVoiceProvider | None = None
        self.bridge: AudioBridge | None = None
        self.r = AttemptResult(state=S.CALLING)
        self._done = asyncio.Event()
        self._answered_at: float | None = None
        self._last_user_final = 0.0
        self._last_agent_done = 0.0
        self._agent_speaking = False
        self._agent_errors = 0
        self._silence_nudges = 0
        self._awake = False
        self._goodbye_started = False
        self._heard_this_turn = False
        self._fallback_i = random.randrange(len(FALLBACK_LINES))
        self._bg: set[asyncio.Task] = set()
        self._audio_stats: dict | None = None
        self._last_partial = ""   # unfinished words when the call ends (console only)

    # ================================================================ main
    async def run(self) -> AttemptResult:
        row = self.store.create_call(self.run_id, self.attempt)
        started = time.monotonic()
        try:
            await self._run()
        except asyncio.CancelledError:
            self._fail(S.AGENT_ERROR, "cancelled")
            raise
        except NotAuthenticated as e:
            self._fail(S.CALL_FAILED, f"whatsapp not authenticated: {e}")
        except WhatsAppError as e:
            self._fail(S.CALL_FAILED, str(e))
        except Exception as e:
            log.exception("attempt crashed")
            self._fail(S.AGENT_ERROR, f"{type(e).__name__}: {e}")
        finally:
            await self._cleanup()
            if self._answered_at:
                self.r.duration_s = time.monotonic() - self._answered_at
            if self.voice is not None:
                self.r.user_turns = getattr(self.voice, "user_turns", lambda: 0)()
            self.r.summary = await self._summarize()
            self.store.update_call(
                row,
                call_id=self.call_id,
                answered_at=self._iso_answered(started),
                call_end=now_iso(),
                call_status=self.r.state.value,
                end_reason=self.r.end_reason,
                wake_confirmed=int(self.r.wake_confirmed),
                user_turns=self.r.user_turns,
                agent_turns=self.r.agent_turns,
                conversation_summary=self.r.summary or None,
            )
            if self.cfg.privacy.get("store_transcript") and self.voice is not None:
                self.store.save_transcript(row, getattr(self.voice, "transcript", lambda: [])())
            event("CALL_ENDED", run=self.run_id, attempt=self.attempt, state=self.r.state.value,
                  reason=self.r.end_reason, awake=self.r.wake_confirmed, secs=round(self.r.duration_s))
        return self.r

    def _iso_answered(self, started: float) -> str | None:
        if not self._answered_at:
            return None
        from datetime import datetime, timedelta, timezone

        return (datetime.now(timezone.utc) - timedelta(seconds=time.monotonic() - self._answered_at)).isoformat(
            timespec="seconds")

    async def _run(self) -> None:
        # ---------------------------------------------------------- CALLING
        self.sm.to(S.CALLING, attempt=self.attempt)
        await self.wa.connect()
        await self.wa.authenticate()
        # prepare the voice session while it rings, so the greeting starts instantly on answer
        self.voice = self.voice_factory()
        instructions = (custom_instructions(self.cfg, self.custom_objective, self.language) if self.custom_objective
                        else voice_instructions(self.cfg, self.attempt))
        voice_ready = asyncio.create_task(self.voice.connect(instructions, VOICE_TOOLS))
        self.call_id = await self.wa.call(self.phone)

        # ---------------------------------------------------------- RINGING
        self.sm.to(S.RINGING, call=self.call_id[:10])
        deadline = time.monotonic() + RING_TIMEOUT_S
        st = await self.wa.get_call_status(self.call_id)
        while st.state not in (CallState.CONNECTED, CallState.ENDED):
            left = deadline - time.monotonic()
            if left <= 0:
                await self.wa.hangup(self.call_id)
                self._fail(S.NO_ANSWER, "ring timeout", reason="timeout")
                return
            st = await self.wa.wait_status_change(self.call_id, min(left, 5))
        if st.state == CallState.ENDED:
            reason = st.end_reason or "unknown"
            # WaCalls reports her pressing "decline" while it rings as user_ended, not declined
            # (live, 2026-09-22: she declined 3x and the retry loop kept calling). Normalise it.
            if reason == "user_ended":
                reason = "declined"
                self.r.stop_requested = True  # an explicit "no": never retry this run
                event("CALL_DECLINED", run=self.run_id)
            # declined/busy/timeout/do_not_disturb: she didn't pick up
            self._fail(S.NO_ANSWER if reason != "failed" else S.CALL_FAILED, f"ended while ringing: {reason}",
                       reason=reason)
            return

        # --------------------------------------------------------- ANSWERED
        self._answered_at = time.monotonic()
        self.r.answered = True
        self.sm.to(S.ANSWERED)
        event("CALL_ANSWERED", run=self.run_id, call=self.call_id[:10])
        try:
            await asyncio.wait_for(voice_ready, 10)
        except Exception as e:
            await self._hangup_now()
            self._fail(S.AGENT_ERROR, f"voice provider failed to start: {type(e).__name__}")
            return

        self.bridge = AudioBridge(self.wa, self.call_id, self.voice, self._on_voice_event)
        self.bridge.start()

        # --------------------------------------------------------- GREETING
        self.sm.to(S.GREETING)
        event("AGENT_GREETING", run=self.run_id)
        opening = self.cfg.raw.get("opening_line") or "Good morning! Wake up, it's time."
        if self.opening_override:
            opening = self.opening_override
        elif self.test_mode:
            opening = "Hello, can you hear me? This is a test call from the wake-up assistant."
        # WhatsApp's media path is still settling at the moment of "connected": audio sent in the
        # first ~1.5 s is often not heard (live test: she said "hello?" repeatedly). Brief pause.
        await asyncio.sleep(1.5)
        await self.voice.say(opening)
        self.sm.to(S.CONVERSING)

        # ------------------------------------------------------- CONVERSING
        watchdog = asyncio.create_task(self._watchdog())
        try:
            await self._done.wait()
        finally:
            watchdog.cancel()

    # ============================================================== events
    async def _on_voice_event(self, ev) -> None:
        if isinstance(ev, AgentTurnStarted):
            self._agent_speaking = True
            self._heard_this_turn = False
        elif isinstance(ev, AgentTurnDone):
            if self.bridge is not None and not self.bridge.barge_in_enabled:
                # opening line finished: wait for it to actually play out, then allow interruptions
                self._spawn(self._enable_barge_in_after_playout())
            if self.show_transcript and ev.text:
                print(f"    AI : {ev.text}{' [cut off]' if ev.interrupted else ''}", flush=True)
            self._agent_speaking = False
            self._last_agent_done = time.monotonic()
            self.r.agent_turns += 1
            self._agent_errors = 0
        elif isinstance(ev, UserSpeechStarted):
            if not self._heard_this_turn:
                self._heard_this_turn = True
                event("USER_SPEECH_DETECTED", run=self.run_id)
        elif isinstance(ev, UserTranscript):
            self._last_partial = "" if ev.final else ev.text
            if ev.final:
                if self.show_transcript:
                    print(f"    HER: {ev.text}", flush=True)
                if STOP_RE.search(ev.text) and not self.r.stop_requested:
                    self.r.stop_requested = True
                    event("STOP_REQUESTED", run=self.run_id, source="transcript")
                self._last_user_final = time.monotonic()
                self._silence_nudges = 0
                self._heard_this_turn = False
                event("USER_UTTERANCE", run=self.run_id, chars=len(ev.text))  # length only, never content
        elif isinstance(ev, ToolCall):
            await self._on_tool(ev)
        elif isinstance(ev, ProviderError):
            await self._on_provider_error(ev)

    async def _on_tool(self, ev: ToolCall) -> None:
        if ev.name == "report_wake_evidence":
            self.r.evidence.update({k: bool(v) for k, v in ev.arguments.items() if k in
                                    ("says_awake", "sitting_up", "feet_on_floor", "sounds_alert")})
            if self.sm.state == S.CONVERSING:
                self.sm.to(S.WAKE_VERIFICATION)
            event("WAKE_VERIFICATION", run=self.run_id, **self.r.evidence)
        elif ev.name == "awake_confirmed":
            turns = getattr(self.voice, "user_turns", lambda: 99)()
            need = self.cfg.verification["min_user_turns"]
            if turns < need and not self.test_mode:
                event("AWAKE_CLAIM_REJECTED", run=self.run_id, user_turns=turns, need=need)
                await self.voice.respond(
                    hint="System: too early to accept that she is awake. Keep verifying with another question.")
                return
            if not self._awake:
                self._awake = True
                self.r.wake_confirmed = True
                if self.sm.state in (S.CONVERSING, S.WAKE_VERIFICATION):
                    self.sm.to(S.AWAKE_CONFIRMED)
                event("AWAKE_CONFIRMED", run=self.run_id, user_turns=turns)
        elif ev.name == "end_call":
            reason = str(ev.arguments.get("reason", ""))
            # Spawned, not awaited: this handler runs inside the bridge's event pump, which is the
            # thing that forwards the goodbye audio. Awaiting here deadlocked it: the call hung up
            # with the goodbye never sent (live call, 2026-09-22: goodbye out_ms=0).
            if reason == "she_asked_to_stop":
                self.r.stop_requested = True
                event("STOP_REQUESTED", run=self.run_id)
                self._spawn(self._goodbye_and_hangup(ended_by="agent"))
            elif self.custom_objective and self._they_are_talking():
                event("END_CALL_REJECTED", run=self.run_id, reason="they are still talking")
                await self.voice.respond(hint="System: they are still talking; listen, don't end the call.")
            elif self.custom_objective and self._silence_ending(reason):
                event("END_CALL_REJECTED", run=self.run_id, reason="silence is not a reason to hang up")
            elif self._awake or self.test_mode:
                self._spawn(self._goodbye_and_hangup(ended_by="agent"))
            else:
                event("END_CALL_REJECTED", run=self.run_id, reason=reason or "none")
                await self.voice.respond(
                    hint="System: you cannot end the call yet, she has not confirmed she is awake. Keep going.")

    async def _on_provider_error(self, ev: ProviderError) -> None:
        event("AGENT_PROVIDER_ERROR", run=self.run_id, where=ev.where, err=ev.message, fatal=ev.fatal)
        if ev.fatal:
            await self._hangup_now()
            self._fail(S.AUDIO_FAILED if ev.where == "asr" else S.AGENT_ERROR, f"{ev.where}: {ev.message}")
            return
        if ev.where in ("llm", "tts"):
            self._agent_errors += 1
            self._agent_speaking = False
            if self._agent_errors >= MAX_CONSECUTIVE_AGENT_ERRORS:
                await self._hangup_now()
                self._fail(S.AGENT_ERROR, f"{self._agent_errors} consecutive {ev.where} failures")
                return
            if ev.where == "llm":
                if self.custom_objective:
                    # conversation: NEVER wake-up lines (live: "Rise and shine!" mid-pitch, 6 times).
                    # First retry silently; if it fails again, a neutral line kept out of context.
                    if self._agent_errors == 1:
                        await self.voice.respond()
                    else:
                        await self.voice.say("Sorry, one second.", remember=False)
                        await asyncio.sleep(0.5)
                        await self.voice.respond()
                    return
                # wake-up: the model is down/slow: keep her engaged with a canned line, not dead air
                self._fallback_i = (self._fallback_i + 1) % len(FALLBACK_LINES)
                await self.voice.say(FALLBACK_LINES[self._fallback_i])

    def _spawn(self, coro) -> None:
        t = asyncio.get_running_loop().create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def _enable_barge_in_after_playout(self) -> None:
        end = time.monotonic() + 15
        while self.wa.audio_backlog_ms(self.call_id) > 0 and time.monotonic() < end:
            await asyncio.sleep(0.1)
        if self.bridge is not None:
            self.bridge.barge_in_enabled = True

    # ============================================================ watchdog
    async def _watchdog(self) -> None:
        if self.max_minutes:
            max_s = int(min(self.max_minutes, 15) * 60)
        else:
            max_s = self.cfg.max_call_seconds if not self.test_mode else 180
        # A sleeping person needs nudging after ~12 s; someone listening to a pitch is thinking, not
        # asleep (live call 2026-09-22: "are you there?" every 13-15 s wrecked the conversation).
        nudge_s = self.cfg.verification["silence_nudge_seconds"] if not self.custom_objective else 25.0
        while not self._done.is_set():
            await asyncio.sleep(0.5)
            now = time.monotonic()

            st = await self.wa.get_call_status(self.call_id)
            if st.state == CallState.ENDED:
                self.r.end_reason = st.end_reason
                if self._goodbye_started or self._awake:
                    self._finish(S.COMPLETED, ended_by="agent" if self._goodbye_started else "her")
                elif self.test_mode and self.r.agent_turns > 0:
                    self._finish(S.COMPLETED, ended_by="her")
                else:
                    # she hung up before confirming. COMPLETED (the call itself worked); the supervisor
                    # decides from wake_confirmed=False whether to call back.
                    self._finish(S.COMPLETED, ended_by="her")
                return

            # dead audio leg
            b = self.bridge
            if b.first_inbound_at is None and now - self._answered_at > FIRST_AUDIO_TIMEOUT_S:
                await self._hangup_now()
                self._fail(S.AUDIO_FAILED, "no inbound audio after answer")
                return
            if b.last_inbound_at and now - b.last_inbound_at > AUDIO_STALL_S:
                await self._hangup_now()
                self._fail(S.AUDIO_FAILED, f"inbound audio stalled {AUDIO_STALL_S}s")
                return

            # hard cap on call length
            if now - self._answered_at > max_s:
                if not self._goodbye_started:
                    event("MAX_DURATION_REACHED", run=self.run_id, secs=max_s)
                    self._goodbye_started = True
                    await self.voice.say("Okay, I have to go now. Please get up, seriously! Bye!")
                    await self._drain_then_hangup(timeout=8)
                    self._finish(S.TIMEOUT if not self._awake else S.COMPLETED, ended_by="timeout")
                    return

            # silence → nudge (she may have fallen back asleep)
            busy = getattr(self.voice, "is_busy", lambda: False)() or self.wa.audio_backlog_ms(self.call_id) > 0
            if not busy and not self._goodbye_started:
                quiet_since = max(self._last_user_final, self._last_agent_done, self._answered_at)
                heard_recently = b.last_loud_at and now - b.last_loud_at < 2.0
                if now - quiet_since > nudge_s and not heard_recently:
                    self._silence_nudges += 1
                    secs = int(now - max(self._last_user_final, self._answered_at))
                    event("SILENCE_NUDGE", run=self.run_id, n=self._silence_nudges, silent_s=secs)
                    self._last_agent_done = now  # debounce until the nudge finishes
                    if self.custom_objective:
                        hint = (f"No reply for {secs} seconds. They are probably listening or thinking. Do NOT ask if "
                                "they are there. Continue with your next useful point, or ask one specific question.")
                    else:
                        hint = f"She has been silent for {secs} seconds"
                    await self.voice.respond(hint=hint)

    # ============================================================= endings
    def _they_are_talking(self) -> bool:
        """Mid-utterance right now (an unfinished transcript). Loudness is NOT used: on a noisy line
        (TV, traffic) it is always 'loud' and would block every legitimate goodbye (sim-caught).
        Speech that starts during the goodbye is handled by _goodbye_and_hangup, which stays on."""
        return bool(self._last_partial)

    def _silence_ending(self, reason: str) -> bool:
        """The model wants to hang up only because nobody replied (live call: it 'wrapped up' after
        two nudges and hung up on the person mid-sentence). Not a valid reason in a conversation."""
        return self._silence_nudges >= 1 and reason not in ("she_asked_to_stop",) and \
            time.monotonic() - self._last_user_final > 20

    async def _goodbye_and_hangup(self, ended_by: str) -> None:
        if self._goodbye_started and self.sm.state == S.GOODBYE:
            return
        self._goodbye_started = True
        if self.sm.state in (S.CONVERSING, S.WAKE_VERIFICATION, S.AWAKE_CONFIRMED):
            self.sm.to(S.GOODBYE)
        heard_before = self._last_user_final
        # the model's goodbye sentence precedes the [[end_call]] marker; let it finish playing
        interrupted = await self._drain_then_hangup(timeout=12, abort_if_they_speak=bool(self.custom_objective),
                                                    heard_before=heard_before)
        if interrupted:
            # they started talking during our goodbye: stay on the line and listen
            self._goodbye_started = False
            if self.sm.state == S.GOODBYE:
                self.sm.to(S.CONVERSING, reason="spoke during goodbye")
            event("GOODBYE_ABORTED", run=self.run_id)
            return
        self._finish(S.COMPLETED, ended_by=ended_by)

    async def _drain_then_hangup(self, timeout: float, abort_if_they_speak: bool = False,
                                 heard_before: float = 0.0) -> bool:
        """Returns True (and does NOT hang up) if they spoke during the goodbye."""
        end = time.monotonic() + timeout
        busy = getattr(self.voice, "is_busy", lambda: False)

        def spoke() -> bool:
            return abort_if_they_speak and (self._last_user_final > heard_before or bool(self._last_partial))

        while time.monotonic() < end and (busy() or self.wa.audio_backlog_ms(self.call_id) > 0):
            if spoke():
                return True
            await asyncio.sleep(0.1)
        tail_end = time.monotonic() + (1.5 if abort_if_they_speak else 0.8)  # jitter tail / last chance
        while time.monotonic() < tail_end:
            if spoke():
                return True
            await asyncio.sleep(0.1)
        await self._hangup_now()
        return False

    async def _hangup_now(self) -> None:
        if self.call_id and self._audio_stats is None:
            self._audio_stats = getattr(self.wa, "audio_stats", lambda c: {})(self.call_id)
        if self.call_id:
            try:
                await self.wa.hangup(self.call_id)
            except Exception:
                log.warning("hangup failed", exc_info=True)

    def _finish(self, state: S, ended_by: str) -> None:
        if self._done.is_set():
            return
        self.r.ended_by = ended_by
        self.r.state = state
        if state == S.COMPLETED and self.sm.state != S.COMPLETED:
            if self.sm.state in (S.CONVERSING, S.WAKE_VERIFICATION, S.AWAKE_CONFIRMED, S.GOODBYE, S.ANSWERED,
                                 S.GREETING):
                self.sm.to(S.COMPLETED, awake=self._awake, ended_by=ended_by)
        elif state != S.COMPLETED:
            self.sm.to(state)
        self._done.set()

    def _fail(self, state: S, detail: str, reason: str | None = None) -> None:
        if self._done.is_set():
            return
        self.r.detail = detail
        self.r.end_reason = reason or self.r.end_reason
        self.r.ended_by = "error" if state != S.NO_ANSWER else None
        self.r.state = state
        try:
            self.sm.to(state, detail=detail[:120])
        except Exception:
            self.sm.state = state
        self._done.set()

    async def _cleanup(self) -> None:
        if self.show_transcript and self._last_partial:
            print(f"    HER (partial, call ended mid-sentence): {self._last_partial}", flush=True)
        if self.call_id:
            st = None
            try:
                st = await self.wa.get_call_status(self.call_id)
            except Exception:
                pass
            if not st or st.state != CallState.ENDED:
                await self._hangup_now()  # never leave a call open
        if self.bridge:
            await self.bridge.stop()
            self.r.barge_ins = self.bridge.barge_ins
            stats = self._audio_stats or getattr(self.wa, "audio_stats", lambda c: {})(self.call_id)
            event("AUDIO_STATS", run=self.run_id, **stats)
        if self.voice:
            try:
                await self.voice.close()
            except Exception:
                pass
        release = getattr(self.wa, "release", None)
        if release and self.call_id:
            try:
                await release(self.call_id)
            except Exception:
                log.debug("release failed", exc_info=True)

    async def _summarize(self) -> str:
        if not self.voice or not self.r.answered:
            return ""
        try:
            return await asyncio.wait_for(self.voice.summarize(SUMMARY_PROMPT), 20)
        except Exception:
            return ""


def describe_target(phone: str) -> str:
    return mask_phone(phone)
