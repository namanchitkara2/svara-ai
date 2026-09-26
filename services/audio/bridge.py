"""Audio bridge between a WhatsAppProvider call and a RealtimeVoiceProvider.

    WhatsApp PCM (16k s16le) ──► [level meter, stall watchdog] ──► voice.send_audio
    voice AudioOut (16k s16le) ──► [pacer inside the provider] ──► WhatsApp
    voice UserSpeechStarted while the agent is audible ──► flush WhatsApp queue + voice.interrupt

Both providers speak 16 kHz here, so no resampling is needed on the NVIDIA path. If a provider
uses another rate it must resample internally (services/audio/resample.py has the helper).
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from services.observability import event
from services.voice.base import AudioOut, RealtimeVoiceProvider, UserSpeechStarted, VoiceEvent
from services.whatsapp.base import WhatsAppProvider

log = logging.getLogger("wake.bridge")

SPEECH_RMS = 700          # int16 RMS above which a frame counts as "she's making sound"
# Fast barge-in (live call: "you are not stopping in between"). Waiting for ASR text took 1 s+.
# Now: sustained voice energy well above the line's noise floor pauses our audio in ~0.3 s;
# transcribed words then make it a real interruption, otherwise playback resumes (no loss).
VAD_MIN_RMS = 800
VAD_FLOOR_MULT = 3.0
VAD_FRAMES = 5            # 5 x 60 ms = 300 ms of sustained voice
PAUSE_CONFIRM_S = 1.6     # no words within this after a pause => false alarm, resume


def rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    a = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2").astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


class AudioBridge:
    def __init__(self, wa: WhatsAppProvider, call_id: str, voice: RealtimeVoiceProvider, on_event):
        """on_event(ev) is awaited for every non-audio voice event (transcripts, tools, errors)."""
        self.wa = wa
        self.call_id = call_id
        self.voice = voice
        self._on_event = on_event
        self._tasks: list[asyncio.Task] = []
        self.first_inbound_at: float | None = None
        self.last_inbound_at: float | None = None
        self.last_loud_at: float | None = None
        self.inbound_frames = 0
        self.barge_ins = 0
        self._agent_audio_turn: int | None = None
        # The opening line carries the whole point of the call and people always say "hello?"
        # the instant they pick up. Live calls: opening cut after 0.1-0.5 s every time. So the
        # first agent turn cannot be barged; WakeCall enables barge-in once it has finished.
        self.barge_in_enabled = False
        self._noise_floor = 300.0
        self._voiced_frames = 0
        self._paused_at: float | None = None
        self._turn_sent_start_ms = 0

    def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._pump_in(), name="bridge-in"),
            asyncio.create_task(self._pump_out(), name="bridge-out"),
        ]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    @property
    def agent_audible(self) -> bool:
        """Is agent audio actually queued/playing to her right now?

        Only audible speech is barge-in-able. Live test 2026-09-22: counting "LLM still thinking"
        as audible made every one of her short replies cancel the agent's pending answer
        (5 utterances, 2 agent turns). A reply she hasn't heard yet is superseded by the
        provider itself when her new final transcript arrives.
        """
        return self.wa.audio_backlog_ms(self.call_id) > 0

    # --------------------------------------------------------------- inbound
    async def _pump_in(self) -> None:
        async for pcm in self.wa.receive_audio(self.call_id):
            now = time.monotonic()
            if self.first_inbound_at is None:
                self.first_inbound_at = now
                event("AUDIO_STREAM_CONNECTED", call=self.call_id[:10], frame_bytes=len(pcm))
            self.last_inbound_at = now
            self.inbound_frames += 1
            level = rms(pcm)
            if level > SPEECH_RMS:
                self.last_loud_at = now
            # track the line's noise floor (slow up, fast down)
            self._noise_floor = min(level, self._noise_floor * 1.02 + 2) if level > 0 else self._noise_floor
            voiced = level > max(VAD_MIN_RMS, self._noise_floor * VAD_FLOOR_MULT)
            self._voiced_frames = self._voiced_frames + 1 if voiced else 0
            if (self._voiced_frames >= VAD_FRAMES and self._paused_at is None and self.barge_in_enabled
                    and self.wa.audio_backlog_ms(self.call_id) > 0):
                self._paused_at = now
                self.wa.pause_audio(self.call_id)
                event("BARGE_PAUSE", call=self.call_id[:10])
            if self._paused_at is not None and now - self._paused_at > PAUSE_CONFIRM_S and self._voiced_frames == 0:
                self._paused_at = None
                self.wa.resume_audio(self.call_id)
                event("BARGE_RESUME", call=self.call_id[:10])   # it was noise, carry on
            await self.voice.send_audio(pcm)

    # -------------------------------------------------------------- outbound
    async def _pump_out(self) -> None:
        async for ev in self.voice.receive():
            try:
                await self._handle(ev)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("bridge event handler failed")

    async def _handle(self, ev: VoiceEvent) -> None:
        if isinstance(ev, AudioOut):
            if ev.turn_id != self._agent_audio_turn:
                self._agent_audio_turn = ev.turn_id
                self._turn_sent_start_ms = self.wa.audio_sent_ms(self.call_id)
            await self.wa.send_audio(self.call_id, ev.pcm)
            return
        if isinstance(ev, UserSpeechStarted):
            if (self.agent_audible or self._paused_at is not None) and self.barge_in_enabled:
                self._paused_at = None
                dropped = await self.wa.flush_audio(self.call_id)
                played = self.wa.audio_sent_ms(self.call_id) - self._turn_sent_start_ms
                await self.voice.interrupt(audio_played_ms=played)
                self.barge_ins += 1
                event("BARGE_IN", call=self.call_id[:10], played_ms=played, dropped_ms=dropped)
            await self._on_event(ev)
            return
        await self._on_event(ev)
