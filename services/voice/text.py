"""Streaming text helpers for the cascaded (ASR → LLM → TTS) voice provider.

- The LLM streams tokens; we cut them into speakable sentences as early as possible so TTS
  can start while the model is still generating.
- Tool calls are emitted in-band as markers on their own:  [[tool_name {"json": "args"}]]
  (a text protocol keeps latency at one streamed completion instead of a tool round-trip).
- Emoji / markdown are stripped before synthesis: Nemotron likes "😄", TTS reads it badly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, transport, symbols & pictographs ext
    "\U00002600-\U000027bf"  # misc symbols, dingbats (❤ ✨ ☀)
    "\U0000fe0f\U0000200d"   # variation selector, ZWJ
    "\U00002190-\U000021ff"  # arrows
    "\U00002b00-\U00002bff"
    "]+"
)
_MARKDOWN_RE = re.compile(r"[*_`#>~]+")
# Nemotron occasionally leaks CJK/kana into Hindi ("नマン"); never speak those scripts
_FOREIGN_SCRIPT_RE = re.compile("[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_STAGE_RE = re.compile(r"\((?:[^()]*?(?:laugh|giggl|sigh|whisper|pause|chuckl|smil)[^()]*)\)", re.I)
# includes the Devanagari danda (।, ॥): without it a Hindi reply was one "sentence", so TTS could
# not start until the whole reply was generated (sim: Hindi median reply latency 4.3 s).
_SENTENCE_END = re.compile(r"([.!?…।॥]+[\"')\]]*)(\s+|$)")
MARKER_RE = re.compile(r"\[\[\s*([a-z_]+)\s*(\{.*?\})?\s*\]\]", re.S)


_LIST_RE = re.compile(r"(^|\s)(\d{1,2}[.)]|[-•])\s+")   # "1. ", "2) ", "- ", "• " read aloud as numbers/dashes


def clean_for_speech(text: str) -> str:
    text = MARKER_RE.sub(" ", text)
    text = _LIST_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _FOREIGN_SCRIPT_RE.sub("", text)
    text = _STAGE_RE.sub(" ", text)
    text = _MARKDOWN_RE.sub("", text)
    text = text.replace("—", ", ").replace("–", ", ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?…])", r"\1", text)
    text = re.sub(r",\s*([.!?])", r"\1", text)
    return text.lstrip(" ,").rstrip()


@dataclass
class Marker:
    name: str
    args: dict


def parse_marker(raw: str) -> Marker | None:
    m = MARKER_RE.fullmatch(raw.strip())
    if not m:
        return None
    args: dict = {}
    if m.group(2):
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
    return Marker(m.group(1), args if isinstance(args, dict) else {})


class SentenceStreamer:
    """Feed LLM deltas; get back speakable sentences and tool markers, in order."""

    def __init__(self, min_chars: int = 12, first_clause_chars: int = 16):
        self._buf = ""
        self._min = min_chars
        # the FIRST chunk of a reply may be a clause ("Nope, not happening,") so audio starts
        # sooner; later chunks stay whole sentences for natural prosody
        self._first_clause = first_clause_chars
        self._emitted = False

    def feed(self, delta: str) -> list[str | Marker]:
        self._buf += delta
        return self._drain(final=False)

    def finish(self) -> list[str | Marker]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[str | Marker]:
        out: list[str | Marker] = []
        while True:
            start = self._buf.find("[[")
            if start != -1:
                end = self._buf.find("]]", start)
                if end == -1:
                    # speak what's before an unfinished marker, hold the marker itself
                    head = self._buf[:start]
                    out += self._sentences(head, final=True)
                    self._buf = self._buf[start:]
                    if final:
                        self._buf = ""
                    return out
                head, raw, self._buf = self._buf[:start], self._buf[start : end + 2], self._buf[end + 2 :]
                out += self._sentences(head, final=True)
                mk = parse_marker(raw)
                if mk:
                    out.append(mk)
                continue
            out += self._sentences_keep_tail(final)
            return out

    def _sentences(self, text: str, final: bool) -> list[str]:
        saved, self._buf = self._buf, text
        res = self._sentences_keep_tail(final)
        self._buf = saved
        return res

    def _sentences_keep_tail(self, final: bool) -> list[str]:
        out: list[str] = []
        pos = 0
        for m in _SENTENCE_END.finditer(self._buf):
            cand = self._buf[pos : m.end()]
            if len(cand.strip()) >= self._min or not out:
                if len(cand.strip()) >= self._min:
                    out.append(cand)
                    pos = m.end()
        tail = self._buf[pos:]
        if not out and not self._emitted and not final and self._first_clause:
            m = re.search(r"^(.{%d,}?[,;:])\s" % self._first_clause, tail)
            if m:
                out.append(m.group(1))
                pos += m.end()
                tail = self._buf[pos:]
        if final:
            if tail.strip():
                out.append(tail)
            self._buf = ""
        else:
            self._buf = tail
        res = [s for s in (clean_for_speech(x) for x in out) if s]
        if res:
            self._emitted = True
        return res
