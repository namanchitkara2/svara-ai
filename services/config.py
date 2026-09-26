"""Configuration: wake.yaml (behaviour, editable) + .env (secrets, never logged)."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "wake.yaml"            # local, gitignored (real numbers)
EXAMPLE_CONFIG = ROOT / "config" / "wake.example.yaml"    # committed template

load_dotenv(ROOT / ".env")

PHONE_RE = re.compile(r"^\+?[0-9][0-9 \-]{6,18}$")


def normalize_phone(phone: str) -> str:
    """'+91 99999-99999' -> '+919999999999'. Raises ValueError on garbage."""
    if not PHONE_RE.match(phone.strip()):
        raise ValueError(f"invalid phone number: {phone!r}")
    digits = re.sub(r"\D", "", phone)
    return "+" + digits


def mask_phone(phone: str) -> str:
    """Logs never carry the full number: '+919999999999' -> '+91******999'."""
    d = re.sub(r"\D", "", phone)
    if len(d) < 6:
        return "***"
    return f"+{d[:2]}{'*' * (len(d) - 5)}{d[-3:]}"


@dataclass(frozen=True)
class Settings:
    """Process-level settings from the environment. Secret values are never repr'd."""

    home: Path
    wacalls_url: str
    wacalls_api_token: str = field(repr=False)
    wacalls_session_id: str
    nvidia_api_key: str = field(repr=False)
    nvidia_llm_base_url: str
    riva_server: str
    riva_asr_function_id: str
    riva_tts_function_id: str
    riva_asr_multilingual_function_id: str
    api_host: str
    api_port: int
    api_token: str = field(repr=False)
    config_path: Path
    gemini_api_key: str = field(repr=False, default="")
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    @property
    def db_path(self) -> Path:
        return self.home / "data" / "wake.db"

    @property
    def log_dir(self) -> Path:
        return self.home / "logs"


def load_settings() -> Settings:
    home = Path(os.environ.get("WAKE_HOME", Path.home() / ".wake-agent")).expanduser()
    (home / "data").mkdir(parents=True, exist_ok=True)
    (home / "logs").mkdir(parents=True, exist_ok=True)
    return Settings(
        home=home,
        wacalls_url=os.environ.get("WACALLS_URL", "http://127.0.0.1:8787").rstrip("/"),
        wacalls_api_token=os.environ.get("WACALLS_API_TOKEN", ""),
        wacalls_session_id=os.environ.get("WACALLS_SESSION_ID", ""),
        nvidia_api_key=os.environ.get("NVIDIA_API_KEY", ""),
        nvidia_llm_base_url=os.environ.get("NVIDIA_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        gemini_base_url=os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        riva_server=os.environ.get("RIVA_SERVER", "grpc.nvcf.nvidia.com:443"),
        riva_asr_function_id=os.environ.get("RIVA_ASR_FUNCTION_ID", "1598d209-5e27-4d3c-8079-4751568b1081"),
        riva_tts_function_id=os.environ.get("RIVA_TTS_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969"),
        # parakeet-1.1b-rnnt-multilingual: hi-IN + 24 other languages + "multi" (code-mixed Hinglish)
        riva_asr_multilingual_function_id=os.environ.get("RIVA_ASR_MULTILINGUAL_FUNCTION_ID",
                                                         "71203149-d3b7-4460-8231-1be2543a1fca"),
        api_host=os.environ.get("WAKE_API_HOST", "127.0.0.1"),
        api_port=int(os.environ.get("WAKE_API_PORT", "8790")),
        api_token=os.environ.get("WAKE_API_TOKEN", ""),
        config_path=Path(os.environ.get("WAKE_CONFIG", DEFAULT_CONFIG)),
    )


# ---------------------------------------------------------------- wake.yaml


@dataclass
class WakeConfig:
    raw: dict[str, Any]

    # convenience accessors -------------------------------------------------
    @property
    def caller_name(self) -> str:
        return self.raw.get("caller_name", "your partner")

    @property
    def contact_name(self) -> str:
        return self.raw["contact"]["name"]

    @property
    def contact_phone(self) -> str:
        return normalize_phone(self.raw["contact"]["phone"])

    @property
    def schedule(self) -> dict[str, Any]:
        return self.raw["schedule"]

    @property
    def retry(self) -> dict[str, Any]:
        r = self.raw.get("retry", {})
        return {
            "enabled": bool(r.get("enabled", True)),
            # hard ceiling: never retry forever, whatever the config says
            "max_attempts": max(1, min(int(r.get("max_attempts", 3)), 5)),
            "delay_seconds": max(30, int(r.get("delay_seconds", 300))),
        }

    @property
    def max_call_seconds(self) -> int:
        return max(60, min(int(self.raw.get("maximum_call_duration_minutes", 15)), 30)) * 60

    @property
    def voice(self) -> dict[str, Any]:
        return self.raw.get("voice", {})

    @property
    def verification(self) -> dict[str, Any]:
        v = self.raw.get("verification", {})
        return {
            "min_user_turns": int(v.get("min_user_turns", 3)),
            "silence_nudge_seconds": float(v.get("silence_nudge_seconds", 12)),
        }

    @property
    def followup(self) -> dict[str, Any]:
        f = self.raw.get("followup", {})
        return {
            "enabled": bool(f.get("enabled", False)),
            "after_minutes": max(2, min(int(f.get("after_minutes", 10)), 60)),
            "max_checks": max(1, min(int(f.get("max_checks", 2)), 3)),
        }

    @property
    def privacy(self) -> dict[str, Any]:
        return self.raw.get("privacy", {})


def _fix_time(raw: dict[str, Any]) -> None:
    """YAML 1.1 reads unquoted 10:30 as the base-60 integer 630. Normalise back to 'HH:MM'."""
    t = raw.get("schedule", {}).get("time")
    if isinstance(t, int):
        raw["schedule"]["time"] = f"{t // 60:02d}:{t % 60:02d}"
    elif t is not None:
        raw["schedule"]["time"] = str(t)


def validate_config(raw: dict[str, Any]) -> None:
    _fix_time(raw)
    for key in ("contact", "schedule", "objective", "personality"):
        if key not in raw:
            raise ValueError(f"config missing '{key}'")
    normalize_phone(raw["contact"]["phone"])
    hh, mm = str(raw["schedule"]["time"]).split(":")
    if not (0 <= int(hh) < 24 and 0 <= int(mm) < 60):
        raise ValueError("schedule.time must be HH:MM")
    from zoneinfo import ZoneInfo

    ZoneInfo(raw["schedule"]["timezone"])  # raises on unknown tz


def load_config(path: Path | None = None) -> WakeConfig:
    path = path or load_settings().config_path
    raw = yaml.safe_load(path.read_text())
    validate_config(raw)
    return WakeConfig(raw)


def save_config(raw: dict[str, Any], path: Path | None = None) -> WakeConfig:
    """Validate, then atomically replace wake.yaml."""
    validate_config(raw)
    path = path or load_settings().config_path
    class _Q(str):
        pass

    yaml.SafeDumper.add_representer(_Q, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v, style="'"))
    raw = {**raw, "schedule": {**raw["schedule"], "time": _Q(raw["schedule"]["time"])}}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True, default_style=None)
    os.replace(tmp, path)
    return load_config(path)
