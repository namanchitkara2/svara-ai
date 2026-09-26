"""Realtime voice provider contract, independent of the wake-up logic.

Shaped like a speech-to-speech session (OpenAI Realtime / Gemini Live) so providers are swappable:
you configure it with instructions + tools, stream her audio in, and consume events out.
Audio at this boundary is 16 kHz s16le mono. A provider that works at another rate
resamples internally (services/audio/resample.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

SAMPLE_RATE = 16000


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


# ---------------------------------------------------------------- events out
@dataclass
class AudioOut:
    pcm: bytes                    # 16 kHz s16le mono, agent speech
    turn_id: int


@dataclass
class AgentTurnStarted:
    turn_id: int


@dataclass
class AgentTurnDone:
    turn_id: int
    text: str                     # what was actually synthesised (for the in-memory context only)
    interrupted: bool = False


@dataclass
class UserSpeechStarted:
    """Her voice was detected (barge-in trigger)."""


@dataclass
class UserTranscript:
    text: str
    final: bool


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderError:
    where: str                    # asr | llm | tts
    message: str
    fatal: bool = False


VoiceEvent = AudioOut | AgentTurnStarted | AgentTurnDone | UserSpeechStarted | UserTranscript | ToolCall | ProviderError


class RealtimeVoiceProvider(ABC):
    @abstractmethod
    async def connect(self, instructions: str, tools: list[Tool]) -> None: ...

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None:
        """Her audio, 16 kHz s16le mono."""

    @abstractmethod
    def receive(self) -> AsyncIterator[VoiceEvent]:
        """Events: agent audio, transcripts, barge-in, tool calls, errors."""

    @abstractmethod
    async def say(self, text: str, remember: bool = True) -> None:
        """Speak a fixed line (the opening line / a goodbye); remember=False keeps it out of context."""

    @abstractmethod
    async def respond(self, hint: str | None = None) -> None:
        """Ask the model to take a turn now (e.g. she has been silent). `hint` is a private system note."""

    @abstractmethod
    async def interrupt(self, audio_played_ms: int | None = None) -> None:
        """Barge-in: stop generating, and trim the context to what she actually heard."""

    @abstractmethod
    async def close(self) -> None: ...

    # optional
    async def summarize(self, prompt: str) -> str:
        return ""


ToolHandler = Callable[[ToolCall], Awaitable[str | None]]
