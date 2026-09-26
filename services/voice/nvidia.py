"""RealtimeVoiceProvider built on NVIDIA hosted models (one NVIDIA_API_KEY):

    her audio ──► Riva streaming ASR (Parakeet, 16 kHz, grpc.nvcf.nvidia.com)
                    │ interim text  ─► UserSpeechStarted (barge-in)
                    │ final text    ─► turn
                    ▼
               Nemotron chat (integrate.api.nvidia.com, OpenAI-compatible, streamed)
                    │ sentences + [[tool]] markers
                    ▼
               Riva TTS (Magpie, 16 kHz LINEAR_PCM, streamed)  ─► AudioOut

Why a cascade: NVIDIA exposes no speech-to-speech session like OpenAI Realtime, and the
user asked for NVIDIA-only. All three stages run at 16 kHz, which is WhatsApp's native rate,
so there is no resampling anywhere on this path.

Latency budget (measured 2026-09-22 from India): ASR endpoint ~0.5-0.8 s after she stops,
Nemotron-3-super TTFT ~0.56 s, Magpie first audio ~0.73 s. First sentence is synthesised
while the model is still generating the rest.

The gRPC clients are synchronous, so ASR and each TTS request run in worker threads and hand
results to the event loop with call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import re
import threading
import time
from typing import AsyncIterator

import riva.client
from openai import AsyncOpenAI

from services.voice.base import (
    AgentTurnDone,
    AgentTurnStarted,
    AudioOut,
    ProviderError,
    RealtimeVoiceProvider,
    Tool,
    ToolCall,
    UserSpeechStarted,
    UserTranscript,
    VoiceEvent,
)
from services.voice.text import Marker, SentenceStreamer, clean_for_speech

log = logging.getLogger("wake.voice.nvidia")

_SILENCE_100MS = b"\x00" * 3200
_INTERIM_STALL_S = 2.0     # partial unchanged this long => finished (noisy-line fallback; 1.2 s cut Naman off)
_INCOMPLETE_HOLD_S = 1.2   # extra wait when an utterance ends mid-thought ("and...", "matlab...")
# endings that mean "I'm not done yet" (English + Hinglish/Hindi)
_INCOMPLETE_RE = re.compile(
    r"(\b(and|but|so|or|because|like|um+|uh+|the|a|to|of|with|if|then|actually|basically|i mean|you know|"
    r"ki|aur|toh|to|matlab|lekin|par|ya|jaise|ki agar)|और|तो|कि|मतलब|लेकिन|या)\s*[,.…-]*\s*$",
    re.I,
)
_BARGE_IN_MIN_CHARS = 4       # a real word, not a breath/"hm"
# words that must stop the agent instantly, however short (live: "stop stop stop" was ignored)
_STOP_WORDS_RE = re.compile(r"\b(stop|wait|hold on|hang on|one sec|listen|ruko|ruk|bas|suno|ek minute)\b", re.I)
_MAX_WORDS_PER_TURN = 40      # live: 20 s monologues; people interrupt, so keep turns short
_BARGE_IN_GRACE_S = 0.3       # speech must start this long after the agent's turn to count as barge-in
_MAX_SENTENCES_PER_TURN = 3   # phone turns stay short no matter what the model writes
_HEDGE_AFTER_S = 1.2          # no first token by then -> race a duplicate LLM request (measured TTFT p50 ~0.6 s)
_BILINGUAL_SPEAKERS = {"Sofia", "Leo"}   # speakers that exist as both EN-US and HI-IN Magpie voices
_SPECULATE_AFTER_S = 0.35     # partial transcript stable this long -> start the reply speculatively


class NvidiaVoiceProvider(RealtimeVoiceProvider):
    def __init__(
        self,
        api_key: str,
        *,
        llm_model: str = "nvidia/nemotron-3-super-120b-a12b",
        llm_fallback_models: list[str] | None = None,
        llm_base_url: str = "https://integrate.api.nvidia.com/v1",
        riva_server: str = "grpc.nvcf.nvidia.com:443",
        asr_function_id: str = "1598d209-5e27-4d3c-8079-4751568b1081",
        tts_function_id: str = "877104f7-e885-42b9-8de8-f6e4c6303969",
        tts_voice: str = "Magpie-Multilingual.EN-US.Aria",
        language_code: str = "en-US",
        asr_language_code: str | None = None,
        pronunciations: dict[str, str] | None = None,
        extra_llms: dict[str, tuple[str, str]] | None = None,
        llm_first_token_timeout: float = 6.0,
    ):
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is not set")
        self._key = api_key
        self._models = [llm_model] + list(llm_fallback_models or [])
        # A model may be prefixed with its provider: "gemini:gemini-3.5-flash-lite". Unprefixed =
        # NVIDIA NIM. Gemini is the fallback because NIM errored on 14% of turns in live calls.
        self._clients = {"nvidia": AsyncOpenAI(base_url=llm_base_url, api_key=api_key, timeout=20, max_retries=0)}
        if extra_llms:
            for name, (url, k) in extra_llms.items():
                if k:
                    self._clients[name] = AsyncOpenAI(base_url=url, api_key=k, timeout=20, max_retries=0)
        self._llm = self._clients["nvidia"]
        self._riva_server = riva_server
        self._asr_fid = asr_function_id
        self._tts_fid = tts_function_id
        self._voice = tts_voice
        self._lang = language_code                       # TTS language
        self._asr_lang = asr_language_code or language_code
        self._ttft_timeout = llm_first_token_timeout
        # spoken-only respellings, e.g. {"Asha": "Aasha"}: a mispronounced name read back as a
        # different (and sometimes rude) word, and the text shown/logged is unchanged
        self._pron = {k: v for k, v in (pronunciations or {}).items() if k and v}
        self._said: set[str] = set()
        # per-turn latency marks (monotonic seconds); emitted as a TURN_LATENCY event
        self._t: dict[str, float] = {}
        self._last_speech_at = 0.0     # last time her partial transcript changed (~ end of speech)
        # Speculative reply: started on a stable partial, output HELD until the final transcript
        # confirms the same words (then released) or differs (then rolled back without a trace).
        self._spec: dict | None = None
        self._holding = False
        self._held: list[VoiceEvent] = []
        self._first_audible_reported = False   # normalised sentences already spoken this call (no repeats)

        self._events: asyncio.Queue[VoiceEvent] = asyncio.Queue()
        self._history: list[dict] = []
        self._instructions = ""
        self._loop: asyncio.AbstractEventLoop | None = None

        self._asr_in: queue.Queue[bytes | None] = queue.Queue(maxsize=200)
        self._asr_thread: threading.Thread | None = None
        self._closed = False
        self._asr_failures = 0

        self._turn_task: asyncio.Task | None = None
        self._turn_id = 0
        self._speaking = False            # agent audio being produced/played for current turn
        self._spoken_this_turn: list[str] = []
        self._pending_user: list[str] = []
        self._barged = False
        self._interim_text = ""
        self._interim_at = 0.0
        self._synth_final = ""          # text we already finalised ourselves (dedupe the late server final)
        self._stall_task: asyncio.Task | None = None
        self._utt_started_at: float | None = None   # when the current (unfinished) utterance began
        self._agent_turn_started_at = 0.0

    # ------------------------------------------------------------ lifecycle
    def _auth(self, function_id: str) -> riva.client.Auth:
        return riva.client.Auth(
            use_ssl=True,
            uri=self._riva_server,
            metadata_args=[["function-id", function_id], ["authorization", f"Bearer {self._key}"]],
        )

    async def connect(self, instructions: str, tools: list[Tool]) -> None:
        self._loop = asyncio.get_running_loop()
        self._stall_task = asyncio.create_task(self._interim_stall_watch(), name="asr-stall")
        self._instructions = instructions + _tool_protocol(tools)
        self._tts = riva.client.SpeechSynthesisService(self._auth(self._tts_fid))
        self._asr = riva.client.ASRService(self._auth(self._asr_fid))
        self._asr_thread = threading.Thread(target=self._asr_worker, name="riva-asr", daemon=True)
        self._asr_thread.start()

    async def close(self) -> None:
        self._closed = True
        if self._stall_task:
            self._stall_task.cancel()
        await self._cancel_turn()
        try:
            self._asr_in.put_nowait(None)
        except queue.Full:
            pass

    # ------------------------------------------------------------ audio in
    async def send_audio(self, pcm: bytes) -> None:
        try:
            self._asr_in.put_nowait(pcm)
        except queue.Full:  # ASR is behind: drop the oldest audio rather than stall the call
            try:
                self._asr_in.get_nowait()
                self._asr_in.put_nowait(pcm)
            except (queue.Empty, queue.Full):
                pass

    def _asr_audio(self):
        """Blocking generator for the gRPC stream. Keeps the stream alive with silence."""
        while not self._closed:
            try:
                chunk = self._asr_in.get(timeout=0.1)
            except queue.Empty:
                yield _SILENCE_100MS
                continue
            if chunk is None:
                return
            yield chunk

    def _asr_worker(self) -> None:
        cfg = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=16000,
                language_code=self._asr_lang,
                max_alternatives=1,
                enable_automatic_punctuation=True,
                audio_channel_count=1,
            ),
            interim_results=True,
        )
        if self._asr_lang == "en-US":
            # Faster end-of-utterance than the server default: 500 ms of silence ends the turn.
            # NOT for the multilingual model: there it splits words ("प / ीओप"). On a noisy line it
            # detected speech three times and returned zero finals, so the agent thought the person
            # was silent. 1000 ms: sleepy speech has "Ugh... five more minutes" pauses, and people
            # thinking aloud pause mid-sentence (at 800 ms it answered before they had finished).
            riva.client.add_endpoint_parameters_to_config(cfg, 0, 0.0, 1000, 0, 0.0, 0.0)
        while not self._closed:
            started = time.monotonic()
            try:
                for resp in self._asr.streaming_response_generator(audio_chunks=self._asr_audio(), streaming_config=cfg):
                    for res in resp.results:
                        if not res.alternatives:
                            continue
                        text = res.alternatives[0].transcript.strip()
                        if text:
                            self._post(self._on_asr, text, bool(res.is_final))
                self._asr_failures = 0
            except Exception as e:  # stream dropped / NVCF hiccup: reconnect
                if self._closed:
                    return
                self._asr_failures = 0 if time.monotonic() - started > 30 else self._asr_failures + 1
                log.warning("ASR stream ended (%s), reconnect #%d", type(e).__name__, self._asr_failures)
                fatal = self._asr_failures >= 5
                self._post(self._emit, ProviderError("asr", type(e).__name__, fatal=fatal))
                if fatal:
                    return
                time.sleep(min(0.5 * self._asr_failures, 3))

    def _report_latency(self) -> None:
        t = self._t
        if "speech_end" not in t or "audible" not in t:
            return
        ms = lambda a, b: int((t[b] - t[a]) * 1000) if a in t and b in t else None  # noqa: E731
        from services.observability import event
        event("TURN_LATENCY",
              endpoint_ms=ms("speech_end", "final"),
              queue_ms=ms("final", "llm_start") if "spec_start" not in t else None,
              llm_ttft_ms=ms("llm_start", "llm_first_token"),
              first_sentence_ms=ms("llm_first_token", "first_sentence"),
              tts_ms=ms("first_sentence", "tts_first_audio"),
              total_ms=ms("speech_end", "audible"),
              speculative="hit" if t.get("spec_hit") else ("miss" if t.get("spec_miss") else None),
              synthetic_final=t.get("synthetic") or None)

    def _post(self, fn, *args) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(fn, *args)

    def _emit(self, ev: VoiceEvent) -> None:
        self._events.put_nowait(ev)

    def _emit_turn(self, ev: VoiceEvent) -> None:
        """Events produced by a reply turn: held while that turn is still speculative."""
        if self._holding:
            self._held.append(ev)
            return
        if isinstance(ev, AudioOut) and not self._first_audible_reported and "speech_end" in self._t:
            self._first_audible_reported = True
            self._t["audible"] = time.monotonic()
            self._report_latency()
        self._emit(ev)

    async def _interim_stall_watch(self) -> None:
        """Client-side endpointing fallback: on a noisy line the server may never finalise."""
        while not self._closed:
            await asyncio.sleep(0.05)
            stable = time.monotonic() - self._interim_at if self._interim_text else 0.0
            if (self._interim_text and stable > _SPECULATE_AFTER_S and self._spec is None
                    and not _INCOMPLETE_RE.search(self._interim_text)
                    and (not self._turn_task or self._turn_task.done()) and len(_norm(self._interim_text)) >= 2):
                self._start_speculative(self._interim_text)
            if self._interim_text and stable > _INTERIM_STALL_S:
                text, self._interim_text = self._interim_text, ""
                self._synth_final = text
                self._on_asr(text, True, synthetic=True)

    def _on_asr(self, text: str, final: bool, synthetic: bool = False) -> None:
        if len(_norm(text)) < 2:
            return  # noise transcribed as punctuation only ("।", ".") -- not speech
        if not synthetic:
            if final:
                self._interim_text = ""
                done, self._synth_final = self._synth_final, ""
                if done and _norm(text).startswith(_norm(done)[: max(4, len(_norm(done)) - 3)]):
                    rest = text[len(done):].strip() if text.startswith(done) else ""
                    if not rest:
                        return  # server caught up with what we already finalised
                    text = rest
            else:
                if not self._interim_text:
                    self._utt_started_at = time.monotonic()
                if text != self._interim_text:
                    self._interim_text, self._interim_at = text, time.monotonic()
                    self._last_speech_at = self._interim_at
                    # she kept talking past a speculative reply: drop it now so a fresh one can start
                    # on her latest pause (sim: stale "ugh" speculation blocked the useful one)
                    if self._spec is not None and not self._spec.get("aborting") and not self._spec_matches(text):
                        self._spec["aborting"] = True
                        asyncio.get_running_loop().create_task(self._abort_speculative())
        # Barge-in = NEW speech that started after the agent began talking. A late final (or
        # interim) for something she said *before* the agent's reply is the tail of her previous
        # turn, not an interruption. Sim run: that mistake caused 10 barge-ins in 70 s.
        # Barge-in = she is producing NEW words after the agent started talking. Judge by when the
        # partial last changed, not when her utterance began: if the agent started while she was
        # mid-sentence, her continuing is exactly an interruption (sim "interrupter": the old
        # utterance-start rule let the agent talk over her). Late FINALS never barge (tail of her
        # previous turn).
        changed_at = self._interim_at if self._interim_text == text else time.monotonic()
        if (not final and (len(text) >= _BARGE_IN_MIN_CHARS or _STOP_WORDS_RE.search(text))
                and changed_at > self._agent_turn_started_at + _BARGE_IN_GRACE_S):
            self._emit(UserSpeechStarted())
        self._emit(UserTranscript(text=text, final=final))
        if final:
            self._utt_started_at = None
            if self._spec is not None and not self._spec.get("aborting"):
                if self._spec_matches(text):
                    self._confirm_speculative(text)
                    return
                if os.environ.get("WAKE_DEBUG_SPEC"):  # console-only diagnostics, never logged
                    print(f"    [spec miss] partial={self._spec['text']!r} final={text!r}", flush=True)
                asyncio.get_running_loop().create_task(self._restart_after_abort(text, synthetic))
                return
            if self._spec is not None:  # abort in flight
                asyncio.get_running_loop().create_task(self._restart_after_abort(text, synthetic))
                return
            self._t = {"speech_end": self._last_speech_at or time.monotonic(), "final": time.monotonic(),
                       "synthetic": synthetic}
            self._first_audible_reported = False
            self._pending_user.append(text)
            if _INCOMPLETE_RE.search(text) and (not self._turn_task or self._turn_task.done()):
                # sounds unfinished: give them a moment; any new speech merges into this turn
                self._hold_gen = getattr(self, "_hold_gen", 0) + 1
                asyncio.get_running_loop().create_task(self._start_after_hold(self._hold_gen))
                return
            if self._turn_task and not self._turn_task.done():
                # still generating a reply she hasn't heard: restart it with her new words included
                # (_after_turn starts the new turn). If it is already speaking, it finishes/gets
                # barged, then _after_turn answers the pending text.
                if not self._spoken_this_turn:
                    self._turn_task.cancel()
                return
            self._start_turn(None)

    async def _start_after_hold(self, gen: int) -> None:
        await asyncio.sleep(_INCOMPLETE_HOLD_S)
        # still the latest hold, nothing new being said, nobody else started a turn
        if gen != getattr(self, "_hold_gen", 0) or self._interim_text or self._closed:
            return
        if self._pending_user and (not self._turn_task or self._turn_task.done()):
            self._start_turn(None)

    async def _restart_after_abort(self, text: str, synthetic: bool) -> None:
        if self._spec is not None:
            await self._abort_speculative()
        self._t = {"speech_end": self._last_speech_at or time.monotonic(), "final": time.monotonic(),
                   "synthetic": synthetic, "spec_miss": True}
        self._first_audible_reported = False
        self._pending_user.append(text)
        if not self._turn_task or self._turn_task.done():
            self._start_turn(None)

    # ---------------------------------------------------------- events out
    async def receive(self) -> AsyncIterator[VoiceEvent]:
        while not (self._closed and self._events.empty()):
            try:
                yield await asyncio.wait_for(self._events.get(), 0.5)
            except asyncio.TimeoutError:
                continue

    # ---------------------------------------------------------------- turns
    async def say(self, text: str, remember: bool = True) -> None:
        await self._cancel_turn()
        if remember:
            self._history.append({"role": "assistant", "content": text})
        self._turn_task = asyncio.create_task(self._speak_fixed(text))
        self._turn_task.add_done_callback(self._after_turn)

    async def respond(self, hint: str | None = None) -> None:
        if self._turn_task and not self._turn_task.done():
            return
        self._start_turn(hint)

    async def interrupt(self, audio_played_ms: int | None = None) -> None:
        if self._spec is not None:
            await self._abort_speculative()
        self._barged = True
        await self._cancel_turn(interrupted=True)

    def is_busy(self) -> bool:
        return bool(self._turn_task and not self._turn_task.done())

    def _start_speculative(self, text: str) -> None:
        user_text = " ".join(self._pending_user + [text])
        self._spec = {"text": text, "hist_len": len(self._history), "said": set(self._said),
                      "pending": list(self._pending_user)}
        self._pending_user.clear()
        self._holding, self._held = True, []
        self._first_audible_reported = False
        self._t = {"speech_end": self._last_speech_at or time.monotonic(), "spec_start": time.monotonic()}
        self._history.append({"role": "user", "content": user_text})
        self._turn_task = asyncio.create_task(self._llm_turn())
        self._turn_task.add_done_callback(self._after_turn)
        self._spec["task"] = self._turn_task

    def _spec_matches(self, final_text: str) -> bool:
        a, b = _norm(self._spec["text"]), _norm(final_text)
        return a == b or (b.startswith(a) and len(b) - len(a) <= 2)

    def _confirm_speculative(self, final_text: str) -> None:
        spec, self._spec = self._spec, None
        self._history[spec["hist_len"]]["content"] = " ".join(spec["pending"] + [final_text])
        self._holding = False
        held, self._held = self._held, []
        self._t["final"] = time.monotonic()
        self._t["spec_hit"] = True
        for ev in held:
            self._emit_turn(ev)

    async def _abort_speculative(self) -> None:
        spec, self._spec = self._spec, None
        if spec is None:
            return
        t = spec["task"]
        if not t.done():
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        # roll back: nothing she didn't hear may leak into context, dedupe set or event stream
        del self._history[spec["hist_len"]:]
        self._said = spec["said"]
        self._pending_user[:0] = spec["pending"]
        self._holding, self._held = False, []

    def _start_turn(self, hint: str | None) -> None:
        if self._pending_user:
            self._history.append({"role": "user", "content": " ".join(self._pending_user)})
            self._pending_user.clear()
        if hint:
            self._history.append({"role": "user", "content": f"[{hint}]"})
        self._turn_task = asyncio.create_task(self._llm_turn())
        self._turn_task.add_done_callback(self._after_turn)

    def _after_turn(self, task: asyncio.Task) -> None:
        # she said something while the previous reply was already being spoken: answer it now
        if self._pending_user and not self._closed and (self._turn_task is task or self._turn_task.done()):
            self._start_turn(None)

    async def _cancel_turn(self, interrupted: bool = False) -> None:
        t = self._turn_task
        if t and not t.done():
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    async def _speak_fixed(self, text: str) -> None:
        self._turn_id += 1
        tid = self._turn_id
        self._agent_turn_started_at = time.monotonic()
        self._emit(AgentTurnStarted(tid))
        spoken = clean_for_speech(text)
        # mark as "already speaking" so her "hello?" queues a reply instead of cancelling this line
        self._spoken_this_turn = [spoken]
        for sent in _split_all(text):
            self._said.add(_norm(sent))
        try:
            await self._tts_stream(spoken, tid)
            self._emit(AgentTurnDone(tid, spoken))
        except asyncio.CancelledError:
            self._emit(AgentTurnDone(tid, spoken, interrupted=True))
            raise
        except Exception as e:
            self._emit(ProviderError("tts", type(e).__name__))

    async def _llm_turn(self) -> None:
        self._turn_id += 1
        tid = self._turn_id
        self._spoken_this_turn = []
        self._queued_sentences = 0
        self._queued_words = 0
        self._barged = False
        full_text = ""
        started = False
        try:
            sentences: asyncio.Queue[str | None] = asyncio.Queue()
            tts_task = asyncio.create_task(self._tts_worker(sentences, tid))
            streamer = SentenceStreamer()
            try:
                async for delta in self._llm_stream():
                    full_text += delta
                    for item in streamer.feed(delta):
                        started = self._route(item, sentences, tid, started)
                for item in streamer.finish():
                    started = self._route(item, sentences, tid, started)
            finally:
                await sentences.put(None)
            await tts_task
            self._history.append({"role": "assistant", "content": full_text.strip() or "..."})
            if started:
                self._emit_turn(AgentTurnDone(tid, " ".join(self._spoken_this_turn)))
        except asyncio.CancelledError:
            said = " ".join(self._spoken_this_turn)
            if said:
                # she only heard part of it: keep the context honest
                self._history.append({"role": "assistant", "content": said + " —"})
            if started:
                self._emit_turn(AgentTurnDone(tid, said, interrupted=True))
            raise
        except _LLMUnavailable as e:
            self._emit_turn(ProviderError("llm", str(e)))
        except Exception as e:
            log.exception("turn failed")
            self._emit_turn(ProviderError("llm", type(e).__name__))

    def _route(self, item: str | Marker, sentences: asyncio.Queue, tid: int, started: bool) -> bool:
        if isinstance(item, Marker):
            self._emit_turn(ToolCall(item.name, item.args))
            return started
        key = _norm(item)
        if key in self._said:
            return started  # the model repeated itself (sim: re-said the whole opening after "Hello")
        self._said.add(key)
        self._queued_sentences = getattr(self, "_queued_sentences", 0) + 1
        self._queued_words = getattr(self, "_queued_words", 0) + len(item.split())
        if self._queued_sentences > _MAX_SENTENCES_PER_TURN or (
                self._queued_words > _MAX_WORDS_PER_TURN and self._queued_sentences > 1):
            return started  # phone turns must stay short; the rest is not spoken
        if not started:
            self._t.setdefault("first_sentence", time.monotonic())
            self._agent_turn_started_at = time.monotonic()
            self._emit_turn(AgentTurnStarted(tid))
        sentences.put_nowait(item)
        return True

    async def _tts_worker(self, sentences: asyncio.Queue, tid: int) -> None:
        while (s := await sentences.get()) is not None:
            await self._tts_stream(s, tid)
            self._spoken_this_turn.append(s)

    def _voice_for(self, text: str) -> tuple[str, str]:
        """Pick the voice by the sentence's script. A Hindi voice reading English mangles it (heard
        on a live call), so Latin-script sentences go to the same speaker's EN-US voice and
        Devanagari ones to its HI-IN voice."""
        dev = len(re.findall(r"[\u0900-\u097F]", text))
        lat = len(re.findall(r"[A-Za-z]", text))
        name = self._voice.rsplit(".", 1)[-1]           # Sofia / Leo / Aria ...
        if self._voice.startswith("Magpie-Multilingual.HI-IN.") and lat > dev and name in _BILINGUAL_SPEAKERS:
            return f"Magpie-Multilingual.EN-US.{name}", "en-US"
        if self._voice.startswith("Magpie-Multilingual.EN-US.") and dev > lat and name in _BILINGUAL_SPEAKERS:
            return f"Magpie-Multilingual.HI-IN.{name}", "hi-IN"
        return self._voice, self._lang

    async def _tts_stream(self, text: str, tid: int) -> None:
        """Synthesize one sentence; AudioOut chunks are emitted as they stream back."""
        if not text:
            return
        for word, spoken in self._pron.items():
            text = re.sub(rf"(?<![\w-]){re.escape(word)}(?![\w-])", spoken, text)
        voice, lang = self._voice_for(text)
        done = asyncio.Event()
        err: list[Exception] = []
        cancelled = threading.Event()

        def work() -> None:
            try:
                for resp in self._tts.synthesize_online(
                    text,
                    voice_name=voice,
                    language_code=lang,
                    sample_rate_hz=16000,
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                ):
                    if cancelled.is_set():
                        return
                    if resp.audio:
                        if "tts_first_audio" not in self._t:
                            self._t["tts_first_audio"] = time.monotonic()
                        self._post(self._emit_turn, AudioOut(resp.audio, tid))
            except Exception as e:
                err.append(e)
            finally:
                self._post(done.set)

        threading.Thread(target=work, name="riva-tts", daemon=True).start()
        try:
            await asyncio.wait_for(done.wait(), 20)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        if err:
            raise err[0]

    def _client_for(self, model: str) -> tuple[object, str]:
        provider, _, name = model.partition(":")
        if name and provider in self._clients:
            return self._clients[provider], name
        return self._clients["nvidia"], model

    async def _open_stream(self, model: str, messages: list[dict]):
        client, name = self._client_for(model)
        stream = await client.chat.completions.create(
            model=name,
            messages=messages,
            stream=True,
            max_tokens=120,
            temperature=0.8,
            extra_body=({"chat_template_kwargs": {"enable_thinking": False}} if client is self._clients["nvidia"] else {}),
        )
        it = stream.__aiter__()
        first = await it.__anext__()
        return first, it

    async def _hedged_first(self, model: str, messages: list[dict]):
        """Start one request; if no first token within _HEDGE_AFTER_S, race a second identical one.

        NIM's tail latency is the problem, not its median (sim: median 1.8 s, one turn 7.25 s).
        Whichever request produces a first token first wins; the loser is cancelled.
        """
        tasks = [asyncio.create_task(self._open_stream(model, messages))]
        deadline = time.monotonic() + self._ttft_timeout
        hedged = False
        errors: list[BaseException] = []
        try:
            while tasks:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    raise asyncio.TimeoutError()
                wait_for = min(timeout, _HEDGE_AFTER_S) if not hedged else timeout
                done, _ = await asyncio.wait(tasks, timeout=wait_for, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    tasks.remove(t)
                    if t.exception() is None:
                        return t.result()
                    errors.append(t.exception())
                if not hedged and (not done or errors):
                    hedged = True
                    tasks.append(asyncio.create_task(self._open_stream(model, messages)))
                elif not tasks:
                    break
            raise errors[-1] if errors else asyncio.TimeoutError()
        finally:
            for t in tasks:
                t.cancel()

    async def _llm_stream(self) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": self._instructions}] + self._history[-30:]
        last_err = "no model"
        for model in self._models:
            try:
                self._t.setdefault("llm_start", time.monotonic())
                first, it = await self._hedged_first(model, messages)
                self._t.setdefault("llm_first_token", time.monotonic())
                if first.choices and first.choices[0].delta.content:
                    yield first.choices[0].delta.content
                async for chunk in it:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except asyncio.CancelledError:
                raise
            except BaseException as e:  # TimeoutError / StopAsyncIteration / APIError
                last_err = f"{model}: {type(e).__name__}"
                log.warning("LLM attempt failed: %s", last_err)
        raise _LLMUnavailable(last_err)

    # ------------------------------------------------------------- summary
    async def summarize(self, prompt: str) -> str:
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in self._history)
        for model in [m for m in self._models for _ in range(3)]:
            client, name = self._client_for(model)
            try:
                r = await client.chat.completions.create(
                    model=name,
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": convo or "(no conversation)"}],
                    max_tokens=120,
                    temperature=0.2,
                    extra_body=({"chat_template_kwargs": {"enable_thinking": False}}
                                if client is self._clients["nvidia"] else {}),
                )
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                log.warning("summary failed on %s: %s", model, type(e).__name__)
        return ""

    def transcript(self) -> list[tuple[str, str]]:
        return [(m["role"], m["content"]) for m in self._history]

    def user_turns(self) -> int:
        return sum(1 for m in self._history if m["role"] == "user" and not m["content"].startswith("["))


def _split_all(text: str) -> list[str]:
    ss = SentenceStreamer()
    return [x for x in ss.feed(text) + ss.finish() if isinstance(x, str)]


def _norm(t: str) -> str:
    return "".join(ch for ch in t.lower() if ch.isalnum())


class _LLMUnavailable(Exception):
    pass


def _tool_protocol(tools: list[Tool]) -> str:
    if not tools:
        return ""
    lines = [
        "",
        "",
        "ACTIONS: you can take actions by writing a marker on its own, AFTER your spoken words, exactly like",
        '[[tool_name {"arg": "value"}]]. Markers are never read aloud. Available actions:',
    ]
    for t in tools:
        props = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in t.parameters.get("properties", {}).items())
        lines.append(f"- {t.name}({props}): {t.description}")
    return "\n".join(lines)
