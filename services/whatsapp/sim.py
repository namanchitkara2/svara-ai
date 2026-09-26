"""Simulated WhatsApp call with a simulated person on the other end. No real phone is dialled.

The WakeCall/voice stack under test is the REAL one; only the WhatsApp transport is replaced:

    agent audio ──► OutboundPacer (same real-time pacing as WaCalls) ──► callee "ears"
                                                                            │ Riva ASR
                                                                            ▼
                                                                   persona LLM (Nemotron)
                                                                            │ Riva TTS (+noise)
    agent ASR ◄── receive_audio() 60 ms frames ◄────────────── callee "mouth" ◄┘

So a simulation exercises the true audio path: if the agent's TTS is unintelligible, the
simulated person can't understand it either. Everything is recorded to a stereo WAV.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import riva.client
from openai import AsyncOpenAI

from services.audio.pacer import OutboundPacer
from services.whatsapp.base import AuthStatus, CallState, CallStatus, WhatsAppProvider

log = logging.getLogger("wake.sim")
FRAME = 1920  # 60 ms @ 16 kHz s16le


@dataclass
class Persona:
    name: str
    description: str                   # who they are, mood, what they want
    language: str = "en"               # en | hi
    voice: str = "Magpie-Multilingual.EN-US.Sofia"
    says_hello_on_answer: bool = True  # the thing that cut every real opening line
    noise_rms: float = 0.0             # background noise level (0 = clean line)
    ring_s: float = 2.0
    answers: bool = True
    max_turns: int = 10
    interrupt_after_s: float | None = None   # talk over the agent once it has spoken this long


@dataclass
class SimTurn:
    who: str
    text: str
    t: float


@dataclass
class SimReport:
    turns: list[SimTurn] = field(default_factory=list)
    reply_latencies: list[float] = field(default_factory=list)   # callee stops -> first agent audio
    stop_latencies: list[float] = field(default_factory=list)    # callee starts talking over agent -> agent silent
    agent_audio_after_hangup_request: float = 0.0
    hung_up_by: str | None = None
    wav: str | None = None


class SimWhatsApp(WhatsAppProvider):
    def __init__(self, persona: Persona, *, api_key: str, riva_server: str, tts_fid: str, asr_fid: str,
                 asr_multi_fid: str, llm_base_url: str, llm_model: str, record_to: Path | None = None):
        self.p = persona
        self.report = SimReport()
        self._key, self._riva, self._tts_fid = api_key, riva_server, tts_fid
        self._asr_fid = asr_multi_fid if persona.language == "hi" else asr_fid
        self._asr_lang = "multi" if persona.language == "hi" else "en-US"
        self._tts_lang = "hi-IN" if persona.language == "hi" else "en-US"
        self._llm = AsyncOpenAI(base_url=llm_base_url, api_key=api_key, timeout=30, max_retries=2)
        self._llm_model = llm_model
        self.record_to = record_to
        self.status = CallStatus("sim-call", CallState.RINGING)
        self._changed = asyncio.Event()
        self._mouth = bytearray()
        self._mouth_lock = threading.Lock()
        self._agent_track = bytearray()
        self._callee_track = bytearray()
        self._pacer: OutboundPacer | None = None
        self._ears: queue.Queue[bytes | None] = queue.Queue()
        self._heard: list[str] = []
        self._last_agent_audio = 0.0
        self._callee_done_speaking: float | None = None
        self._t0 = time.monotonic()
        self._tasks: list[asyncio.Task] = []
        self._rng = np.random.default_rng(7)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connected_at: float | None = None
        self._agent_talk_start: float | None = None
        self._interrupt_at: float | None = None

    # ------------------------------------------------------- provider API
    async def connect(self) -> None: ...

    async def authenticate(self) -> AuthStatus:
        return AuthStatus("sim", True, "open")

    async def call(self, phone: str) -> str:
        self._loop = asyncio.get_running_loop()
        self._pacer = OutboundPacer(self._on_agent_pcm)
        self._pacer.start()
        self._tasks.append(asyncio.create_task(self._lifecycle()))
        return "sim-call"

    async def get_call_status(self, call_id: str) -> CallStatus:
        return self.status

    async def wait_status_change(self, call_id: str, timeout: float) -> CallStatus:
        self._changed.clear()
        try:
            await asyncio.wait_for(self._changed.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        return self.status

    async def receive_audio(self, call_id: str) -> AsyncIterator[bytes]:
        nxt = time.monotonic()
        while self.status.state != CallState.ENDED:
            if self.status.state == CallState.CONNECTED:
                with self._mouth_lock:
                    chunk = bytes(self._mouth[:FRAME])
                    del self._mouth[:FRAME]
                if len(chunk) < FRAME:
                    chunk = chunk + b"\x00" * (FRAME - len(chunk))
                frame = self._add_noise(chunk)
                self._callee_track += frame
                yield frame
            nxt += 0.06
            await asyncio.sleep(max(0, nxt - time.monotonic()))

    async def send_audio(self, call_id: str, pcm: bytes) -> None:
        if self._pacer and self.status.state == CallState.CONNECTED:
            self._pacer.push(pcm)

    async def flush_audio(self, call_id: str) -> int:
        return self._pacer.flush() if self._pacer else 0

    def pause_audio(self, call_id: str) -> None:
        if self._pacer:
            self._pacer.pause()

    def resume_audio(self, call_id: str) -> None:
        if self._pacer:
            self._pacer.resume()

    def audio_backlog_ms(self, call_id: str) -> int:
        return self._pacer.queued_ms if self._pacer else 0

    def audio_sent_ms(self, call_id: str) -> int:
        return self._pacer.sent_ms if self._pacer else 0

    def audio_stats(self, call_id: str) -> dict:
        return {"out_ms": self.audio_sent_ms(call_id)}

    async def hangup(self, call_id: str) -> None:
        self._end("agent")

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        self._ears.put(None)
        if self._pacer:
            await self._pacer.close()
        self._write_wav()

    # --------------------------------------------------------- the callee
    def _on_agent_pcm(self, chunk: bytes) -> None:
        # keep the agent track time-aligned with the callee track (silence between utterances)
        if self._connected_at is not None:
            pos = int((time.monotonic() - self._connected_at) * 16000) * 2
            if len(self._agent_track) < pos:
                self._agent_track += b"\x00" * (pos - len(self._agent_track))
        self._agent_track += chunk
        if any(chunk):
            if time.monotonic() - self._last_agent_audio > 0.5:
                self._agent_talk_start = time.monotonic()
            self._last_agent_audio = time.monotonic()
            if self._callee_done_speaking is not None:
                self.report.reply_latencies.append(time.monotonic() - self._callee_done_speaking)
                self._callee_done_speaking = None
        self._ears.put(chunk)

    def _end(self, by: str) -> None:
        if self.status.state != CallState.ENDED:
            self.status.state = CallState.ENDED
            self.status.end_reason = "user_ended"
            self.report.hung_up_by = self.report.hung_up_by or by
            self._changed.set()

    async def _lifecycle(self) -> None:
        await asyncio.sleep(self.p.ring_s)
        if not self.p.answers:
            self.status.state, self.status.end_reason = CallState.ENDED, "timeout"
            self._changed.set()
            return
        self.status.state, self.status.ever_connected = CallState.CONNECTED, True
        self._connected_at = time.monotonic()
        self._changed.set()
        threading.Thread(target=self._ears_worker, daemon=True).start()
        if self.p.says_hello_on_answer:
            await asyncio.sleep(0.4)
            await self._speak("Hello?" if self.p.language == "en" else "हेलो?")
        await self._converse()

    def _ears_worker(self) -> None:
        """Streaming ASR over the agent's audio, as the callee would hear it."""
        auth = riva.client.Auth(use_ssl=True, uri=self._riva, metadata_args=[
            ["function-id", self._asr_fid], ["authorization", f"Bearer {self._key}"]])
        asr = riva.client.ASRService(auth)
        cfg = riva.client.StreamingRecognitionConfig(config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM, sample_rate_hertz=16000, language_code=self._asr_lang,
            max_alternatives=1, enable_automatic_punctuation=True, audio_channel_count=1), interim_results=False)

        def audio():
            while True:
                try:
                    c = self._ears.get(timeout=0.1)
                except queue.Empty:
                    c = b"\x00" * 3200
                if c is None:
                    return
                yield c

        try:
            for resp in asr.streaming_response_generator(audio_chunks=audio(), streaming_config=cfg):
                for r in resp.results:
                    if r.is_final and r.alternatives and r.alternatives[0].transcript.strip():
                        self._heard.append(r.alternatives[0].transcript.strip())
        except Exception as e:
            log.warning("sim ears ended: %s", type(e).__name__)

    async def _converse(self) -> None:
        history: list[dict] = []
        for _ in range(self.p.max_turns):
            # wait until the agent has spoken and then gone quiet for ~1 s (a natural turn end)
            waited = 0.0
            while self.status.state == CallState.CONNECTED:
                now = time.monotonic()
                if (self.p.interrupt_after_s and self._agent_talk_start and now - self._last_agent_audio < 0.2
                        and now - self._agent_talk_start > self.p.interrupt_after_s):
                    self._interrupt_at = now
                    self._agent_talk_start = None
                    asyncio.create_task(self._measure_stop(now))
                    await self._speak("Wait, wait, stop. Let me say something." if self.p.language == "en"
                                      else "रुको रुको, मेरी बात सुनो।")
                    self._heard.clear()
                    break
                quiet = now - self._last_agent_audio
                if self._heard and quiet > 1.0 and (self._pacer.queued_ms == 0):
                    break
                await asyncio.sleep(0.1)
                waited += 0.1
                if waited > 25:  # agent said nothing for 25 s
                    self._heard.append("(silence)")
            if self.status.state != CallState.CONNECTED:
                return
            await asyncio.sleep(0.6)   # let the ASR flush the tail of the agent's sentence
            agent_said = " ".join(self._heard)
            self._heard.clear()
            self.report.turns.append(SimTurn("AGENT(heard)", agent_said, time.monotonic() - self._t0))
            history.append({"role": "user", "content": agent_said})
            reply = await self._persona_reply(history)
            history.append({"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)})
            action, say = reply.get("action", "speak"), (reply.get("say") or "").strip()
            if say and action in ("speak", "hangup"):
                await self._speak(say)
            if action == "hangup":
                await asyncio.sleep(1.0)
                self._end("callee")
                return
            if action == "silent":
                await asyncio.sleep(8)

    async def _measure_stop(self, started: float) -> None:
        # agent counts as stopped once no agent audio has been sent for 300 ms
        while self.status.state == CallState.CONNECTED and time.monotonic() - started < 8:
            if time.monotonic() - self._last_agent_audio > 0.3 and self._last_agent_audio > started - 5:
                self.report.stop_latencies.append(max(0.0, self._last_agent_audio - started))
                return
            await asyncio.sleep(0.05)
        self.report.stop_latencies.append(8.0)

    async def _persona_reply(self, history: list[dict]) -> dict:
        lang = "Hindi (Devanagari script)" if self.p.language == "hi" else "English"
        sys = (f"You are role-playing {self.p.name} answering a phone call. {self.p.description}\n"
               f"You only hear the caller's words (transcribed, may contain errors). Reply in {lang}, like a real "
               "person on the phone: short, natural. Respond ONLY with JSON: "
               '{"action": "speak"|"silent"|"hangup", "say": "..."}. Use hangup (with a short bye in say) when '
               "the conversation is naturally over or you'd realistically hang up.")
        try:
            r = await self._llm.chat.completions.create(
                model=self._llm_model, messages=[{"role": "system", "content": sys}] + history[-12:],
                max_tokens=120, temperature=0.8, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            txt = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0)) if m else {"action": "speak", "say": txt[:120]}
        except Exception as e:
            log.warning("persona llm failed: %s", type(e).__name__)
            return {"action": "speak", "say": "Hmm?"}

    async def _speak(self, text: str) -> None:
        from services.voice.text import clean_for_speech
        text = re.sub(r"\*[^*]{1,40}\*", " ", text)  # persona stage directions like *sigh*
        text = clean_for_speech(text) or "Hmm?"
        self.report.turns.append(SimTurn(self.p.name, text, time.monotonic() - self._t0))
        pcm = await asyncio.to_thread(self._tts, text)
        with self._mouth_lock:
            self._mouth += pcm
        # done speaking when the mouth buffer drains
        while True:
            with self._mouth_lock:
                left = len(self._mouth)
            if not left or self.status.state == CallState.ENDED:
                break
            await asyncio.sleep(0.06)
        self._callee_done_speaking = time.monotonic()

    def _tts(self, text: str) -> bytes:
        auth = riva.client.Auth(use_ssl=True, uri=self._riva, metadata_args=[
            ["function-id", self._tts_fid], ["authorization", f"Bearer {self._key}"]])
        tts = riva.client.SpeechSynthesisService(auth)
        return b"".join(r.audio for r in tts.synthesize_online(
            text, voice_name=self.p.voice, language_code=self._tts_lang, sample_rate_hz=16000,
            encoding=riva.client.AudioEncoding.LINEAR_PCM))

    def _add_noise(self, frame: bytes) -> bytes:
        if self.p.noise_rms <= 0:
            return frame
        x = np.frombuffer(frame, dtype="<i2").astype(np.float32)
        n = np.arange(x.size) + len(self._callee_track) // 2
        x = x + self._rng.normal(0, self.p.noise_rms, x.size) + self.p.noise_rms * 1.5 * np.sin(n / 16000 * 2 * np.pi * 120)
        return np.clip(x, -32768, 32767).astype("<i2").tobytes()

    def _write_wav(self) -> None:
        if not self.record_to:
            return
        a = np.frombuffer(bytes(self._agent_track), dtype="<i2")
        c = np.frombuffer(bytes(self._callee_track), dtype="<i2")
        n = max(a.size, c.size)
        st = np.zeros((n, 2), dtype="<i2")
        st[: a.size, 0], st[: c.size, 1] = a, c
        self.record_to.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.record_to), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(st.tobytes())
        self.report.wav = str(self.record_to)
