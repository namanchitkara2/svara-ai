"""The ONLY capabilities the wake agent has. No shell, no filesystem, no network beyond these.

    whatsapp.call          place the wake-up call and run the voice conversation to completion
    whatsapp.call_status   status of the current/last call
    whatsapp.hangup        hang up the current call
    whatsapp.get_contact   who we are waking up (number masked)
    wake.get_context       objective, schedule, attempts used/left, previous attempt outcomes
    wake.update_state      record a supervisor note / state
    wake.finish            end the run with a verdict and summary

Guardrails live HERE, not in the prompt: attempt ceiling, minimum retry delay, no calls after
she asked us to stop, and only the configured contact can ever be called.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable

from agents.state_machine import S, WakeStateMachine
from agents.wake_call import AttemptResult, WakeCall
from services.config import WakeConfig, mask_phone
from services.observability import event
from services.store import Store
from services.voice.base import RealtimeVoiceProvider
from services.whatsapp.base import WhatsAppProvider

TOOL_SPECS = [
    ("whatsapp_call", "Place the WhatsApp wake-up call to the configured contact and run the whole voice "
                      "conversation. Blocks until the call ends and returns the outcome. Enforces the retry "
                      "delay by itself.", {}),
    ("whatsapp_call_status", "Status of the current or last call.", {}),
    ("whatsapp_hangup", "Hang up the current call if one is active.", {}),
    ("whatsapp_get_contact", "Who is being woken up.", {}),
    ("wake_get_context", "Objective, schedule, attempts used and left, and every previous attempt outcome.", {}),
    ("wake_update_state", "Record a short supervisor note about your reasoning.",
     {"note": {"type": "string"}}),
    ("wake_finish", "Finish the wake-up run. Call exactly once, at the end.",
     {"wake_confirmed": {"type": "boolean"}, "summary": {"type": "string"}}),
]


def openai_tool_schema() -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props)},
        }}
        for name, desc, props in TOOL_SPECS
    ]


@dataclass
class RunContext:
    run_id: int
    cfg: WakeConfig
    trigger: str
    phone: str
    test_mode: bool = False
    opening_override: str | None = None
    custom_objective: str | None = None
    show_transcript: bool = False
    language: str | None = None
    max_minutes: float | None = None
    attempts: list[AttemptResult] = field(default_factory=list)
    finished: bool = False
    final_confirmed: bool = False
    final_summary: str = ""
    last_attempt_end: float | None = None
    current_call: WakeCall | None = None
    notes: list[str] = field(default_factory=list)


class WakeTools:
    def __init__(self, ctx: RunContext, wa: WhatsAppProvider, voice_factory: Callable[[], RealtimeVoiceProvider],
                 store: Store):
        self.ctx, self.wa, self.voice_factory, self.store = ctx, wa, voice_factory, store

    # ---------------------------------------------------------------- policy
    def max_attempts(self) -> int:
        r = self.ctx.cfg.retry
        return 1 if (self.ctx.test_mode or not r["enabled"]) else r["max_attempts"]

    def can_call(self) -> tuple[bool, str]:
        if self.ctx.finished:
            return False, "run already finished"
        if len(self.ctx.attempts) >= self.max_attempts():
            return False, f"attempt limit reached ({self.max_attempts()})"
        if any(a.stop_requested for a in self.ctx.attempts):
            return False, "she asked us to stop or declined the call; not calling again"
        if any(a.wake_confirmed for a in self.ctx.attempts):
            return False, "already confirmed awake"
        return True, ""

    # ------------------------------------------------------------------ tools
    async def whatsapp_call(self) -> dict:
        ok, why = self.can_call()
        if not ok:
            return {"error": why}
        delay = self.ctx.cfg.retry["delay_seconds"]
        if self.ctx.last_attempt_end is not None:
            wait = delay - (time.monotonic() - self.ctx.last_attempt_end)
            if wait > 0:
                event("RETRY_WAIT", run=self.ctx.run_id, seconds=int(wait))
                await asyncio.sleep(wait)
        attempt = len(self.ctx.attempts) + 1
        self.store.update_run(self.ctx.run_id, attempts=attempt, state=S.CALLING.value)
        sm = WakeStateMachine(self.ctx.run_id,
                              on_change=lambda s: self.store.update_run(self.ctx.run_id, state=s.value))
        call = WakeCall(wa=self.wa, voice_factory=self.voice_factory, cfg=self.ctx.cfg, store=self.store,
                        run_id=self.ctx.run_id, attempt=attempt, sm=sm, phone=self.ctx.phone,
                        test_mode=self.ctx.test_mode, opening_override=self.ctx.opening_override,
                        custom_objective=self.ctx.custom_objective, show_transcript=self.ctx.show_transcript,
                        language=self.ctx.language, max_minutes=self.ctx.max_minutes)
        self.ctx.current_call = call
        try:
            res = await call.run()
        finally:
            self.ctx.current_call = None
            self.ctx.last_attempt_end = time.monotonic()
        self.ctx.attempts.append(res)
        out = res.brief()
        out["attempt"] = attempt
        out["attempts_left"] = self.attempts_left()
        return out

    async def followup_call(self, opening: str, objective: str) -> dict:
        """Call back after a confirmed wake-up to check she's still up. Not a retry attempt."""
        if any(a.stop_requested for a in self.ctx.attempts):
            return {"skipped": "she asked us to stop"}
        n = len(self.ctx.attempts) + 1
        sm = WakeStateMachine(self.ctx.run_id,
                              on_change=lambda s: self.store.update_run(self.ctx.run_id, state=s.value))
        call = WakeCall(wa=self.wa, voice_factory=self.voice_factory, cfg=self.ctx.cfg, store=self.store,
                        run_id=self.ctx.run_id, attempt=n, sm=sm, phone=self.ctx.phone, test_mode=True,
                        opening_override=opening, custom_objective=objective,
                        show_transcript=self.ctx.show_transcript, language=self.ctx.language,
                        max_minutes=self.ctx.cfg.max_call_seconds / 60)
        self.ctx.current_call = call
        try:
            res = await call.run()
        finally:
            self.ctx.current_call = None
        self.ctx.attempts.append(res)
        return res.brief()

    async def whatsapp_call_status(self) -> dict:
        c = self.ctx.current_call
        if c and c.call_id:
            st = await self.wa.get_call_status(c.call_id)
            return {"call_id": c.call_id[:10], "state": st.state.value, "wake_state": c.sm.state.value}
        if self.ctx.attempts:
            return {"last_attempt": self.ctx.attempts[-1].brief()}
        return {"state": "no call yet"}

    async def whatsapp_hangup(self) -> dict:
        c = self.ctx.current_call
        if c and c.call_id:
            await self.wa.hangup(c.call_id)
            return {"ok": True}
        return {"ok": False, "error": "no active call"}

    async def whatsapp_get_contact(self) -> dict:
        return {"name": self.ctx.cfg.contact_name, "phone": mask_phone(self.ctx.phone),
                "timezone": self.ctx.cfg.raw["contact"].get("timezone")}

    async def wake_get_context(self) -> dict:
        ok, why = self.can_call()
        return {
            "objective": self.ctx.cfg.raw["objective"],
            "trigger": self.ctx.trigger,
            "schedule": f"{self.ctx.cfg.schedule['time']} {self.ctx.cfg.schedule['timezone']}",
            "retry_delay_seconds": self.ctx.cfg.retry["delay_seconds"],
            "max_attempts": self.max_attempts(),
            "attempts_used": len(self.ctx.attempts),
            "attempts_left": self.attempts_left(),
            "attempts": [a.brief() for a in self.ctx.attempts],
            "can_call": ok,
            "why_not": why or None,
        }

    async def wake_update_state(self, note: str = "") -> dict:
        self.ctx.notes.append(note[:300])
        event("SUPERVISOR_NOTE", run=self.ctx.run_id, chars=len(note))
        return {"ok": True}

    async def wake_finish(self, wake_confirmed: bool = False, summary: str = "") -> dict:
        # the model cannot claim a success the calls didn't produce
        actual = any(a.wake_confirmed for a in self.ctx.attempts)
        self.ctx.finished = True
        self.ctx.final_confirmed = bool(wake_confirmed) and actual
        self.ctx.final_summary = (summary or "")[:600]
        return {"ok": True, "wake_confirmed_recorded": self.ctx.final_confirmed}

    def attempts_left(self) -> int:
        return max(0, self.max_attempts() - len(self.ctx.attempts))

    async def dispatch(self, name: str, args: dict) -> dict:
        fn = {
            "whatsapp_call": self.whatsapp_call,
            "whatsapp_call_status": self.whatsapp_call_status,
            "whatsapp_hangup": self.whatsapp_hangup,
            "whatsapp_get_contact": self.whatsapp_get_contact,
            "wake_get_context": self.wake_get_context,
            "wake_update_state": self.wake_update_state,
            "wake_finish": self.wake_finish,
        }.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        allowed = _params(name)
        try:
            return await fn(**{k: v for k, v in (args or {}).items() if k in allowed})
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}


def _params(name: str) -> set[str]:
    for n, _, props in TOOL_SPECS:
        if n == name:
            return set(props)
    return set()


def deterministic_should_retry(res: AttemptResult) -> bool:
    """Fallback policy used when the LLM supervisor is unavailable."""
    if res.wake_confirmed or res.stop_requested:
        return False
    if res.end_reason in ("declined", "busy", "do_not_disturb"):
        return False  # she actively rejected the call
    if res.state in (S.NO_ANSWER, S.CALL_FAILED, S.AUDIO_FAILED, S.AGENT_ERROR, S.TIMEOUT):
        return True
    # she hung up without confirming: retry unless the conversation clearly showed she was up
    ev = res.evidence
    return not (ev.get("sitting_up") and ev.get("says_awake"))
