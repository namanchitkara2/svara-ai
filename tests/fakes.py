"""In-memory WhatsApp + voice providers for exercising the wake logic without a phone."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from services.voice.base import (
    AgentTurnDone,
    AgentTurnStarted,
    AudioOut,
    RealtimeVoiceProvider,
    Tool,
    ToolCall,
    UserTranscript,
)
from services.whatsapp.base import AuthStatus, CallState, CallStatus, WhatsAppProvider


class FakeWhatsApp(WhatsAppProvider):
    def __init__(self, answer: bool = True, answer_after: float = 0.05, hangup_after: float | None = None,
                 audio: bool = True, end_reason_if_no_answer: str = "timeout"):
        self.answer, self.answer_after, self.hangup_after = answer, answer_after, hangup_after
        self.audio, self.no_answer_reason = audio, end_reason_if_no_answer
        self.status: dict[str, CallStatus] = {}
        self.sent = bytearray()
        self.calls = 0
        self.hangups = 0
        self._changed = asyncio.Event()

    async def connect(self): ...
    async def close(self): ...

    async def authenticate(self):
        return AuthStatus("s1", True, "open")

    async def call(self, phone: str) -> str:
        self.calls += 1
        cid = f"call{self.calls}"
        self.status[cid] = CallStatus(cid, CallState.RINGING)
        asyncio.get_running_loop().create_task(self._lifecycle(cid))
        return cid

    async def _lifecycle(self, cid):
        await asyncio.sleep(self.answer_after)
        if not self.answer:
            self._set(cid, CallState.ENDED, self.no_answer_reason)
            return
        self._set(cid, CallState.CONNECTED)
        if self.hangup_after is not None:
            await asyncio.sleep(self.hangup_after)
            self._set(cid, CallState.ENDED, "user_ended")

    def _set(self, cid, state, reason=None):
        st = self.status[cid]
        if st.state == CallState.ENDED:
            return
        st.state = state
        st.end_reason = reason
        if state == CallState.CONNECTED:
            st.ever_connected = True
        self._changed.set()

    async def get_call_status(self, call_id):
        return self.status[call_id]

    async def wait_status_change(self, call_id, timeout):
        self._changed.clear()
        try:
            await asyncio.wait_for(self._changed.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        return self.status[call_id]

    async def receive_audio(self, call_id) -> AsyncIterator[bytes]:
        while self.status[call_id].state != CallState.ENDED:
            if self.audio and self.status[call_id].state == CallState.CONNECTED:
                yield b"\x00\x00" * 960
            await asyncio.sleep(0.06)

    async def send_audio(self, call_id, pcm):
        self.sent += pcm

    async def flush_audio(self, call_id):
        return 0

    def audio_backlog_ms(self, call_id):
        return 0

    def audio_sent_ms(self, call_id):
        return len(self.sent) // 32

    async def hangup(self, call_id):
        self.hangups += 1
        self._set(call_id, CallState.ENDED, "user_ended")


class ScriptedVoice(RealtimeVoiceProvider):
    """Plays a scripted conversation: after each agent turn, she 'says' the next line.
    Lines starting with '@' are tool markers the model emits instead."""

    def __init__(self, script: list[str | tuple]):
        self.script = list(script)
        self.q: asyncio.Queue = asyncio.Queue()
        self.tid = 0
        self.users = 0
        self.closed = False
        self.hints: list[str] = []

    async def connect(self, instructions: str, tools: list[Tool]):
        self.instructions = instructions

    async def send_audio(self, pcm): ...

    async def receive(self):
        while not self.closed:
            try:
                yield await asyncio.wait_for(self.q.get(), 0.2)
            except asyncio.TimeoutError:
                continue

    async def _turn(self, text):
        self.tid += 1
        await self.q.put(AgentTurnStarted(self.tid))
        await self.q.put(AudioOut(b"\x01\x00" * 1600, self.tid))
        await self.q.put(AgentTurnDone(self.tid, text))
        await asyncio.sleep(0.01)
        await self._next()

    async def _next(self):
        while self.script:
            item = self.script.pop(0)
            if isinstance(item, tuple):  # (tool_name, args)
                await self.q.put(ToolCall(item[0], item[1]))
                continue
            if item == "<silence>":
                return
            self.users += 1
            await self.q.put(UserTranscript(item, final=True))
            await asyncio.sleep(0.01)
            await self._turn("ok")
            return

    async def say(self, text, remember=True):
        asyncio.get_running_loop().create_task(self._turn(text))

    async def respond(self, hint=None):
        if hint:
            self.hints.append(hint)
        asyncio.get_running_loop().create_task(self._turn("nudge"))

    async def interrupt(self, audio_played_ms=None): ...

    async def close(self):
        self.closed = True

    def user_turns(self):
        return self.users

    async def summarize(self, prompt):
        return "scripted summary"
