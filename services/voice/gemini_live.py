"""RealtimeVoiceProvider on Gemini Live (native speech-to-speech).

    her 16 kHz PCM ──► Live API websocket ──► model hears the audio directly
                                          ◄── model speaks (natural prosody, its own VAD)

Why: the NVIDIA cascade (Parakeet ASR -> Nemotron -> Magpie TTS) sounds synthetic and adds
three hops of latency. A native-audio model hears tone and pauses, handles turn-taking itself,
and speaks like a person. The verdict on the cascade from everyone who heard it: "it sounds robotic".

The call stack is unchanged: this class emits the same events (AudioOut, UserTranscript,
AgentTurnStarted/Done, ToolCall, ProviderError) at the same 16 kHz s16le mono boundary, so
WakeCall/AudioBridge don't know which provider is in use. Output is resampled when the model
returns another rate (the mimeType carries it).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import ssl
import time
from typing import AsyncIterator

import certifi
import websockets

from services.audio.resample import StreamResampler
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
from services.voice.text import clean_for_speech

log = logging.getLogger("wake.voice.gemini")

WS_URL = ("wss://generativelanguage.googleapis.com/ws/"
          "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
IN_RATE = 16000
_RATE_RE = re.compile(r"rate=(\d+)")


class GeminiLiveVoiceProvider(RealtimeVoiceProvider):
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.5-flash-native-audio-latest",
        voice: str = "Aoede",
        language_code: str = "en-US",
        silence_ms: int = 900,          # how long a pause must be before the model answers
        languages: list[str] | None = None,   # restrict speech recognition to these
        pronunciations: dict[str, str] | None = None,
        ws_url: str = WS_URL,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self._key = api_key
        self._model = model if model.startswith("models/") else f"models/{model}"
        self._voice = voice
        self._lang = language_code
        self._silence_ms = silence_ms
        # Without this the model auto-detects ANY language from short noisy utterances: on a live call
        # a callee's replies came back as French/Italian/Spanish and it answered him in Italian.
        self._languages = languages or ["en-IN", "hi-IN", "pa-IN"]
        self._pron = {k: v for k, v in (pronunciations or {}).items() if k and v}
        self._ws_url = ws_url

        self._ws = None
        self._events: asyncio.Queue[VoiceEvent] = asyncio.Queue()
        self._rx_task: asyncio.Task | None = None
        self._closed = False
        self._turn_id = 0
        self._speaking = False
        self._out_rs: StreamResampler | None = None
        self._out_rate = 0
        self._user_text: list[str] = []
        self._agent_text: list[str] = []
        self._transcript: list[tuple[str, str]] = []
        self._user_turns = 0
        self._tool_ids: dict[str, str] = {}

    # ------------------------------------------------------------ lifecycle
    async def connect(self, instructions: str, tools: list[Tool]) -> None:
        # websockets does not use httpx's CA bundle; without this the handshake fails with
        # CERTIFICATE_VERIFY_FAILED on macOS
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self._ws = await websockets.connect(f"{self._ws_url}?key={self._key}", max_size=None,
                                            ping_interval=20, ping_timeout=20, ssl=ssl_ctx)
        decls = [{"name": t.name, "description": t.description, "parameters": _openapi(t.parameters)}
                 for t in tools]
        setup = {
            "setup": {
                "model": self._model,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice}}},
                },
                "systemInstruction": {"parts": [{"text": instructions + _SPEECH_RULES + self._lang_rule()}]},
                "inputAudioTranscription": {"languageCodes": self._languages},
                "outputAudioTranscription": {"languageCodes": self._languages},
                "realtimeInputConfig": {
                    "automaticActivityDetection": {
                        "disabled": False,
                        # people trail off mid-sentence; wait for a real pause before answering
                        "silenceDurationMs": self._silence_ms,
                        "prefixPaddingMs": 300,
                    }
                },
            }
        }
        if decls:
            setup["setup"]["tools"] = [{"functionDeclarations": decls}]
        await self._ws.send(json.dumps(setup))
        raw = await asyncio.wait_for(self._ws.recv(), 20)
        msg = _loads(raw)
        if "setupComplete" not in msg:
            raise RuntimeError(f"Live API setup failed: {str(msg)[:200]}")
        self._rx_task = asyncio.create_task(self._receiver(), name="gemini-live-rx")

    def _lang_rule(self) -> str:
        names = {"en-IN": "English", "en-US": "English", "hi-IN": "Hindi", "pa-IN": "Punjabi",
                 "ur-IN": "Urdu", "bn-IN": "Bengali"}
        allowed = ", ".join(dict.fromkeys(names.get(c, c) for c in self._languages))
        return (f"\n- You speak ONLY these languages: {allowed}. Never reply in any other language, "
                "whatever the transcription seems to say. If a reply sounds like another language or "
                "you cannot make it out, assume it was mis-heard and simply ask them to repeat.")

    async def close(self) -> None:
        self._closed = True
        if self._rx_task:
            self._rx_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------- audio in
    async def send_audio(self, pcm: bytes) -> None:
        if self._closed or self._ws is None or not pcm:
            return
        try:
            await self._ws.send(json.dumps({"realtimeInput": {
                "audio": {"mimeType": f"audio/pcm;rate={IN_RATE}", "data": base64.b64encode(pcm).decode()}}}))
        except Exception as e:
            if not self._closed:
                log.warning("send_audio failed: %s", type(e).__name__)

    # ----------------------------------------------------------- events out
    async def receive(self) -> AsyncIterator[VoiceEvent]:
        while not (self._closed and self._events.empty()):
            try:
                yield await asyncio.wait_for(self._events.get(), 0.5)
            except asyncio.TimeoutError:
                continue

    async def _receiver(self) -> None:
        try:
            async for raw in self._ws:
                self._handle(_loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._closed:
                log.warning("live socket closed: %s", type(e).__name__)
                self._events.put_nowait(ProviderError("llm", type(e).__name__, fatal=True))

    def _handle(self, msg: dict) -> None:
        if "toolCall" in msg:
            for fc in msg["toolCall"].get("functionCalls", []):
                if fc.get("id"):
                    self._tool_ids[fc["name"]] = fc["id"]
                self._events.put_nowait(ToolCall(fc.get("name", ""), fc.get("args") or {}))
                asyncio.create_task(self._ack_tool(fc))
            return
        sc = msg.get("serverContent")
        if not sc:
            return
        if sc.get("interrupted"):
            self._end_turn(interrupted=True)
        it = sc.get("inputTranscription")
        if it and it.get("text"):
            self._user_text.append(it["text"])
            self._events.put_nowait(UserSpeechStarted())
            self._events.put_nowait(UserTranscript(text=it["text"], final=False))
        ot = sc.get("outputTranscription")
        if ot and ot.get("text"):
            self._agent_text.append(ot["text"])
        for part in (sc.get("modelTurn") or {}).get("parts", []):
            data = (part.get("inlineData") or {}).get("data")
            if data:
                self._emit_audio((part["inlineData"].get("mimeType") or ""), base64.b64decode(data))
            elif part.get("text"):
                self._agent_text.append(part["text"])
        if sc.get("turnComplete") or sc.get("generationComplete"):
            # her turn is over as soon as the model answers it
            if self._user_text:
                text = " ".join(self._user_text).strip()
                self._user_text = []
                if text:
                    self._user_turns += 1
                    self._transcript.append(("user", text))
                    self._events.put_nowait(UserTranscript(text=text, final=True))
            self._end_turn()

    def _emit_audio(self, mime: str, pcm: bytes) -> None:
        m = _RATE_RE.search(mime or "")
        rate = int(m.group(1)) if m else 24000
        if rate != self._out_rate:
            self._out_rate = rate
            self._out_rs = StreamResampler(rate, IN_RATE)
        if not self._speaking:
            self._speaking = True
            self._turn_id += 1
            self._events.put_nowait(AgentTurnStarted(self._turn_id))
        out = self._out_rs.process(pcm) if self._out_rs else pcm
        if out:
            self._events.put_nowait(AudioOut(out, self._turn_id))

    def _end_turn(self, interrupted: bool = False) -> None:
        if not self._speaking:
            return
        self._speaking = False
        said = clean_for_speech(" ".join(self._agent_text))
        self._agent_text = []
        if said:
            self._transcript.append(("assistant", said))
        self._events.put_nowait(AgentTurnDone(self._turn_id, said, interrupted=interrupted))

    async def _ack_tool(self, fc: dict) -> None:
        try:
            await self._ws.send(json.dumps({"toolResponse": {"functionResponses": [
                {"id": fc.get("id"), "name": fc.get("name"), "response": {"ok": True}}]}}))
        except Exception:
            pass

    # ---------------------------------------------------------------- turns
    async def say(self, text: str, remember: bool = True) -> None:
        spoken = clean_for_speech(text)
        for word, sp in self._pron.items():
            spoken = re.sub(rf"(?<![\w-]){re.escape(word)}(?![\w-])", sp, spoken)
        # one clean turn: the model must not add a greeting of its own before/after
        await self._client_text(f'Speak this now, word for word, and then stop and listen: "{spoken}"')
        if remember:
            self._transcript.append(("assistant", spoken))

    async def respond(self, hint: str | None = None) -> None:
        await self._client_text(f"[{hint}]" if hint else "[continue]")

    async def interrupt(self, audio_played_ms: int | None = None) -> None:
        # the model stops on its own when it hears her; this just closes our turn bookkeeping
        self._end_turn(interrupted=True)

    async def _client_text(self, text: str) -> None:
        if self._closed or self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"clientContent": {
                "turns": [{"role": "user", "parts": [{"text": text}]}], "turnComplete": True}}))
        except Exception as e:
            self._events.put_nowait(ProviderError("llm", type(e).__name__))

    def is_busy(self) -> bool:
        return self._speaking

    def user_turns(self) -> int:
        return self._user_turns

    def transcript(self) -> list[tuple[str, str]]:
        return list(self._transcript)

    async def summarize(self, prompt: str) -> str:
        """Summaries run on the text model; the live socket is audio-only."""
        from openai import AsyncOpenAI

        convo = "\n".join(f"{r}: {t}" for r, t in self._transcript)
        try:
            c = AsyncOpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                            api_key=self._key, timeout=20, max_retries=1)
            r = await c.chat.completions.create(
                model="gemini-3.5-flash-lite", max_tokens=120, temperature=0.2,
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": convo or "(no conversation)"}])
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            log.warning("summary failed: %s", type(e).__name__)
            return ""


_SPEECH_RULES = """

HOW YOU SPEAK (voice call)
- You are speaking out loud on a phone call. Sound like a real person: natural rhythm, contractions,
  short turns of one or two sentences, and let them finish before you answer.
- Never read out lists, bullets, markdown or emoji.
- When you use one of your tools, keep talking naturally; never mention the tool, never narrate
  what you are doing or why, and never read your own reasoning out loud. Speak only the words you
  would actually say to the person on the phone.
- Do not greet or speak on your own before you are given your first line. When you are asked to
  speak a line word for word, say exactly that and nothing else, then stop and listen."""


def _openapi(schema: dict) -> dict:
    """Gemini function declarations use the OpenAPI subset with upper-case type names."""
    out = {}
    for k, v in (schema or {}).items():
        if k == "type" and isinstance(v, str):
            out[k] = v.upper()
        elif isinstance(v, dict):
            out[k] = _openapi(v)
        else:
            out[k] = v
    return out


def _loads(raw) -> dict:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


_ = time
