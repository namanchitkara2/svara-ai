import asyncio
import time

import numpy as np

from services.audio.pacer import CODEC_FRAME_BYTES, OutboundPacer
from services.audio.resample import StreamResampler
from services.config import mask_phone, normalize_phone
from services.observability import scrub
from services.voice.text import Marker, SentenceStreamer, clean_for_speech


async def test_pacer_is_real_time_and_frame_aligned():
    got: list[tuple[float, int]] = []
    p = OutboundPacer(lambda b: got.append((time.monotonic(), len(b))))
    p.start()
    p.push(b"\x01\x00" * 16000)          # 1.0 s of audio in one burst
    t0 = time.monotonic()
    await p.wait_idle(5)
    await asyncio.sleep(0.05)
    total = sum(n for _, n in got)
    assert 0.9 <= got[-1][0] - t0 <= 1.2    # released over ~1 s, not instantly
    assert total % CODEC_FRAME_BYTES == 0   # tail padded to a whole 60 ms MLow frame
    await p.close()


async def test_pacer_flush_for_barge_in():
    p = OutboundPacer(lambda b: None)
    p.start()
    p.push(b"\x00\x00" * 32000)           # 2 s
    await asyncio.sleep(0.1)
    dropped = p.flush()
    assert 1700 <= dropped <= 2000
    assert p.queued_ms == 0
    await p.close()


def test_resampler_16k_24k_roundtrip_length():
    up, down = StreamResampler(16000, 24000), StreamResampler(24000, 16000)
    x = (np.sin(np.arange(16000) / 16000 * 2 * np.pi * 440) * 8000).astype("<i2").tobytes()
    y = b"".join(up.process(x[i:i + 1920]) for i in range(0, len(x), 1920)) + up.process(b"", last=True)
    assert abs(len(y) / 2 - 24000) < 200
    z = down.process(y, last=True)
    assert abs(len(z) / 2 - 16000) < 200
    assert StreamResampler(16000, 16000).process(x) is x


def test_sentence_streamer_splits_and_extracts_markers():
    s = SentenceStreamer()
    out = []
    for d in ["Nope 😄. Sit up first, I'm staying right here! Are you ", "sitting up? [[report_wake",
              '_evidence {"sitting_up": false}]] Okay.']:
        out += s.feed(d)
    out += s.finish()
    assert Marker("report_wake_evidence", {"sitting_up": False}) in out
    texts = [o for o in out if isinstance(o, str)]
    assert all("😄" not in t and "[[" not in t for t in texts)
    assert texts[-1] == "Okay."


def test_clean_for_speech():
    assert clean_for_speech("Good morning ❤️. Wake up — *now* (laughs)") == "Good morning. Wake up, now"


def test_phone_helpers_and_scrub():
    assert normalize_phone("+91 99999-99999") == "+919999999999"
    assert mask_phone("+919999999999") == "+91*******999"
    assert "nvapi" not in scrub("key=nvapi-abcdefghijklmnop")
    assert "abcdefgh" not in scrub("Authorization: Bearer abcdefghijkl")


def test_schedule_time_survives_yaml_roundtrip(tmp_path):
    import yaml
    from services.config import EXAMPLE_CONFIG, load_config, save_config
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["schedule"]["time"] = "10:30"
    p = tmp_path / "w.yaml"
    save_config(raw, p)
    assert "'10:30'" in p.read_text()
    assert load_config(p).schedule["time"] == "10:30"
    p.write_text(p.read_text().replace("'10:30'", "10:30"))   # hand-edited, unquoted
    assert load_config(p).schedule["time"] == "10:30"


def test_hindi_danda_splits_sentences():
    s = SentenceStreamer()
    out = s.feed("अंकल, POP ठेकेदार हायर हो गया है। बाथरूम का पानी ") + s.feed("चेक हो गया है। ठीक?") + s.finish()
    assert out[0] == "अंकल, POP ठेकेदार हायर हो गया है।" and len(out) >= 2


def test_first_chunk_can_be_a_clause():
    import re as _re
    s = SentenceStreamer()
    out = []
    for tok in _re.findall(r"\S+\s*", "Nope, not happening, you said that ten minutes ago. Sit up for me."):
        out += s.feed(tok)
    out += s.finish()
    assert out[0] == "Nope, not happening," and out[-1] == "Sit up for me."


def test_voice_follows_script_and_respells():
    from services.voice.nvidia import NvidiaVoiceProvider, _INCOMPLETE_RE
    v = NvidiaVoiceProvider("k", tts_voice="Magpie-Multilingual.HI-IN.Sofia", language_code="hi-IN",
                            pronunciations={"GTM-OS": "G T M O S"})
    assert v._voice_for("Quick pitch for GTM-OS, Naman.") == ("Magpie-Multilingual.EN-US.Sofia", "en-US")
    assert v._voice_for("नमस्ते अंकल, आप कैसे हैं?") == ("Magpie-Multilingual.HI-IN.Sofia", "hi-IN")
    assert _INCOMPLETE_RE.search("I was thinking that and")
    assert _INCOMPLETE_RE.search("haan matlab")
    assert not _INCOMPLETE_RE.search("Yes, I'm up.")


async def test_pacer_pause_keeps_audio_and_resume_plays_it():
    got = []
    p = OutboundPacer(lambda b: got.append(len(b)))
    p.start()
    p.push(b"\x00\x01" * 16000)   # 1 s
    await asyncio.sleep(0.1)
    p.pause()
    sent = sum(got)
    await asyncio.sleep(0.3)
    assert sum(got) == sent and p.queued_ms > 500     # paused: nothing sent, nothing lost
    p.resume()
    await p.wait_idle(3)
    assert sum(got) >= 32000
    await p.close()


def test_lists_are_not_read_aloud():
    assert clean_for_speech("GTM-OS does: 1. build an ICP 2. read profiles - draft notes") == \
        "GTM-OS does: build an ICP read profiles draft notes"
