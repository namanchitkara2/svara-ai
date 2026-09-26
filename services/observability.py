"""Structured event log.

Every lifecycle step is one line:  `08:00:02 WHATSAPP_CALL_STARTED run=12 attempt=1 call=AB12..`
mirrored as JSON to ~/.wake-agent/logs/wake.jsonl.

Rules:
- phone numbers go through mask_phone() before they get here
- conversation text is never logged (only lengths / counts)
- anything that looks like a secret is scrubbed as a last line of defence
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_SECRET_RE = re.compile(r"(nvapi-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,}|sk-[A-Za-z0-9]{16,})")
_lock = threading.Lock()
_jsonl: Path | None = None
_tz = ZoneInfo("Asia/Kolkata")
_listeners: list = []

log = logging.getLogger("wake")


def scrub(value: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", value)


def setup(log_dir: Path, tz: str = "Asia/Kolkata", verbose: bool = False) -> None:
    global _jsonl, _tz
    _jsonl = log_dir / "wake.jsonl"
    _tz = ZoneInfo(tz)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # third-party loggers that could echo request headers
    for noisy in ("httpx", "httpcore", "aioice", "aiortc", "grpc", "openai", "apscheduler.executors"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def add_listener(fn) -> None:
    """fn(record: dict) is called for every event (used by the API's live view)."""
    _listeners.append(fn)


def event(name: str, **fields) -> dict:
    now = datetime.now(_tz)
    rec = {"ts": now.isoformat(timespec="milliseconds"), "event": name}
    rec.update({k: v for k, v in fields.items() if v is not None})
    line = " ".join([now.strftime("%H:%M:%S"), name] + [f"{k}={v}" for k, v in fields.items() if v is not None])
    line = scrub(line)
    print(line, file=sys.stderr, flush=True)
    if _jsonl is not None:
        with _lock, _jsonl.open("a") as f:
            f.write(scrub(json.dumps(rec, default=str)) + "\n")
    for fn in list(_listeners):
        try:
            fn(rec)
        except Exception:  # listeners must never break the call path
            pass
    return rec
