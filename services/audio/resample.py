"""Streaming resampler for providers that don't run at WhatsApp's 16 kHz.

The NVIDIA path is 16 kHz end to end and does not use this. It exists so a 24 kHz provider
(OpenAI Realtime: in 24k/out 24k; Gemini Live: in 16k/out 24k) can be plugged in behind
RealtimeVoiceProvider: 16k→24k is ×3/2 and 24k→16k is ×2/3, and soxr's streaming mode keeps
filter state across chunks so there are no clicks at chunk boundaries.
"""

from __future__ import annotations

import numpy as np
import soxr


class StreamResampler:
    def __init__(self, in_rate: int, out_rate: int):
        self.passthrough = in_rate == out_rate
        self._rs = None if self.passthrough else soxr.ResampleStream(in_rate, out_rate, 1, dtype="int16", quality="HQ")
        self._odd = b""

    def process(self, pcm: bytes, last: bool = False) -> bytes:
        if self.passthrough:
            return pcm
        pcm = self._odd + pcm
        cut = len(pcm) - (len(pcm) % 2)
        self._odd = pcm[cut:]
        x = np.frombuffer(pcm[:cut], dtype="<i2")
        return self._rs.resample_chunk(x, last=last).astype("<i2").tobytes()
