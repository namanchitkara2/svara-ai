"""The only WhatsApp surface the rest of the app sees. Everything WaCalls-specific lives in wacalls.py."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

# Audio contract at this boundary: raw PCM, signed 16-bit little-endian, mono, 16 kHz.
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2


class CallState(str, Enum):
    STARTING = "starting"
    RINGING = "ringing"
    CONNECTED = "connected"        # answered and media is flowing
    RECONNECTING = "reconnecting"
    ENDED = "ended"
    UNKNOWN = "unknown"


@dataclass
class CallStatus:
    call_id: str
    state: CallState
    end_reason: str | None = None  # declined | timeout | user_ended | busy | failed | ...
    ever_connected: bool = False


@dataclass
class AuthStatus:
    session_id: str | None
    paired: bool
    state: str                      # open | qr | connecting | logged_out | missing
    jid_masked: str | None = None


class WhatsAppError(Exception):
    pass


class NotAuthenticated(WhatsAppError):
    pass


class WhatsAppProvider(ABC):
    @abstractmethod
    async def connect(self) -> None:
        """Reach the transport and start listening for call events."""

    @abstractmethod
    async def authenticate(self) -> AuthStatus:
        """Verify a paired WhatsApp session exists. Raises NotAuthenticated if not."""

    @abstractmethod
    async def call(self, phone: str) -> str:
        """Place an outgoing voice call. Audio is attached before this returns. Returns call id."""

    @abstractmethod
    async def get_call_status(self, call_id: str) -> CallStatus: ...

    @abstractmethod
    async def wait_status_change(self, call_id: str, timeout: float) -> CallStatus:
        """Block until the call's state changes (or timeout) and return the current status."""

    @abstractmethod
    def receive_audio(self, call_id: str) -> AsyncIterator[bytes]:
        """Her audio: 16 kHz s16le mono chunks, as they arrive."""

    @abstractmethod
    async def send_audio(self, call_id: str, pcm: bytes) -> None:
        """Queue 16 kHz s16le mono audio for her. Paced to real time by the provider."""

    @abstractmethod
    async def flush_audio(self, call_id: str) -> int:
        """Drop queued-but-unsent audio (barge-in). Returns milliseconds discarded."""

    def pause_audio(self, call_id: str) -> None:
        """Hold outbound audio without discarding it (fast barge-in). Optional."""

    def resume_audio(self, call_id: str) -> None:
        """Continue held outbound audio. Optional."""

    @abstractmethod
    def audio_backlog_ms(self, call_id: str) -> int:
        """Milliseconds of audio queued for her but not yet sent."""

    @abstractmethod
    def audio_sent_ms(self, call_id: str) -> int:
        """Total milliseconds of audio actually sent to her on this call."""

    @abstractmethod
    async def hangup(self, call_id: str) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...
