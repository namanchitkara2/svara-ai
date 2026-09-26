"""Real-time pacer for outbound PCM.

WaCalls' CallManager.FeedCapturedPCM encodes and transmits the moment it has a 60 ms frame;
it does NOT pace. TTS returns seconds of audio in a burst, so without pacing a whole sentence
would hit WhatsApp's jitter buffer at once and most of it would be discarded. This pacer
releases fixed 20 ms chunks on a monotonic clock, supports instant flush for barge-in, pads the
tail of an utterance to a whole codec frame, and backs off if the transport's send buffer grows.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

SAMPLE_RATE = 16000
BYTES_PER_MS = SAMPLE_RATE * 2 // 1000        # 32 bytes/ms at 16 kHz s16le
CODEC_FRAME_BYTES = 960 * 2                   # MLow frame = 960 samples = 60 ms


class OutboundPacer:
    def __init__(
        self,
        send: Callable[[bytes], None],
        backlog_bytes: Callable[[], int] = lambda: 0,
        chunk_ms: int = 20,
        max_transport_backlog: int = 64 * 1024,
        max_queue_ms: int = 60_000,
    ):
        self._send = send
        self._backlog = backlog_bytes
        self._chunk = chunk_ms * BYTES_PER_MS
        self._chunk_s = chunk_ms / 1000
        self._max_transport_backlog = max_transport_backlog
        self._max_queue = max_queue_ms * BYTES_PER_MS
        self._buf = bytearray()
        self._since_idle = 0            # bytes sent since the queue last ran dry
        self.sent_bytes = 0
        self.dropped_bytes = 0
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._task: asyncio.Task | None = None
        self._closed = False
        self.paused = False

    # ------------------------------------------------------------------ API
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="outbound-pacer")

    def push(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        if len(pcm) % 2:
            pcm = pcm[:-1]
        self._buf += pcm
        overflow = len(self._buf) - self._max_queue
        if overflow > 0:  # never let the queue grow without bound
            del self._buf[: overflow + (overflow % 2)]
            self.dropped_bytes += overflow
        self._idle.clear()
        self._wake.set()

    def pause(self) -> None:
        """Stop releasing audio immediately but keep the queue (might be a false alarm)."""
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        self._wake.set()

    def flush(self) -> int:
        """Barge-in: discard everything not yet sent. Returns ms discarded."""
        n = len(self._buf)
        self._buf.clear()
        self._since_idle = 0
        self.paused = False
        self._idle.set()
        return n // BYTES_PER_MS

    @property
    def queued_ms(self) -> int:
        return len(self._buf) // BYTES_PER_MS

    @property
    def sent_ms(self) -> int:
        return self.sent_bytes // BYTES_PER_MS

    async def wait_idle(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def close(self) -> None:
        self._closed = True
        self._buf.clear()
        self._wake.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    # ------------------------------------------------------------- internals
    async def _run(self) -> None:
        next_t = time.monotonic()
        while not self._closed:
            if not self._buf:
                if self._since_idle % CODEC_FRAME_BYTES:
                    # pad the tail so the utterance's last codec frame is flushed now,
                    # not glued to the start of the next utterance
                    pad = CODEC_FRAME_BYTES - (self._since_idle % CODEC_FRAME_BYTES)
                    self._emit(b"\x00" * pad)
                self._since_idle = 0
                self._idle.set()
                self._wake.clear()
                await self._wake.wait()
                next_t = time.monotonic()
                continue

            if self.paused:
                await asyncio.sleep(self._chunk_s)
                next_t = time.monotonic()
                continue

            if self._backlog() > self._max_transport_backlog:
                # transport is not draining: hold off one tick instead of piling on
                await asyncio.sleep(self._chunk_s)
                next_t = time.monotonic()
                continue

            chunk = bytes(self._buf[: self._chunk])
            del self._buf[: self._chunk]
            self._emit(chunk)

            next_t += self._chunk_s
            delay = next_t - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.25:
                next_t = time.monotonic()  # we fell far behind (GC/CPU stall): resync, don't burst

    def _emit(self, chunk: bytes) -> None:
        try:
            self._send(chunk)
        except Exception:
            self.dropped_bytes += len(chunk)
            return
        self.sent_bytes += len(chunk)
        self._since_idle += len(chunk)
