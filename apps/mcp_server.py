"""MCP server exposing the wake agent's tools to Claude Code (stdio).

Exactly the restricted tool surface from agents/tools.py plus scheduling. No shell, no files.
Register:  claude mcp add wake-agent -- uv --directory ~/wake-agent run wake-agent-mcp
"""

from __future__ import annotations

import asyncio

import httpx
from mcp.server.mcpserver import MCPServer

from apps.runtime import run_wake
from services.config import load_config, load_settings, mask_phone
from services.store import Store
from services.whatsapp.wacalls import WaCallsProvider

mcp = MCPServer("wake-agent")
_settings = load_settings()
_bg: set[asyncio.Task] = set()


def _daemon(method: str, path: str, **kw) -> dict:
    r = httpx.request(method, f"http://{_settings.api_host}:{_settings.api_port}{path}",
                      headers={"Authorization": f"Bearer {_settings.api_token}"}, timeout=10, **kw)
    r.raise_for_status()
    return r.json()


@mcp.tool(name="whatsapp_get_contact")
def whatsapp_get_contact() -> dict:
    """The configured wake-up contact (number masked)."""
    cfg = load_config(_settings.config_path)
    return {"name": cfg.contact_name, "phone": mask_phone(cfg.contact_phone), "timezone": cfg.schedule["timezone"]}


@mcp.tool(name="whatsapp_call")
async def whatsapp_call(test: bool = True) -> dict:
    """Start a wake-up call to the configured contact now, in the background.
    test=True runs the short 'can you hear me?' test; test=False runs the full wake-up with retries."""
    t = asyncio.create_task(run_wake(trigger="test" if test else "manual", test_mode=test))
    _bg.add(t)
    t.add_done_callback(_bg.discard)
    return {"started": True, "test": test, "hint": "poll wake_get_context for the outcome"}


@mcp.tool(name="whatsapp_call_status")
async def whatsapp_call_status() -> dict:
    """WhatsApp session state and the latest run."""
    wa = WaCallsProvider(_settings.wacalls_url, _settings.wacalls_api_token, _settings.wacalls_session_id)
    try:
        await wa.connect()
        sessions = [{"state": s.get("state"), "paired": s.get("paired")} for s in await wa.list_sessions()]
    except Exception as e:
        sessions = [{"error": type(e).__name__}]
    finally:
        await wa.close()
    runs = Store(_settings.db_path).recent_runs(1)
    return {"whatsapp_sessions": sessions, "latest_run": runs[0] if runs else None}


@mcp.tool(name="whatsapp_hangup")
async def whatsapp_hangup() -> dict:
    """Hang up every live call on the paired session (emergency stop)."""
    wa = WaCallsProvider(_settings.wacalls_url, _settings.wacalls_api_token, _settings.wacalls_session_id)
    try:
        await wa.connect()
        st = await wa.authenticate()
        r = await wa._req("GET", f"/api/sessions/{st.session_id}/calls")
        calls = r.json().get("calls", [])
        for c in calls:
            await wa._req("DELETE", f"/api/sessions/{st.session_id}/calls/{c['callId']}", ok=(200, 204, 404))
        return {"hung_up": len(calls)}
    finally:
        await wa.close()


@mcp.tool(name="wake_get_context")
def wake_get_context() -> dict:
    """Schedule, next run, and recent runs with their outcomes."""
    try:
        return _daemon("GET", "/api/status")
    except Exception:
        cfg = load_config(_settings.config_path)
        return {"daemon": "not running", "schedule": cfg.schedule,
                "recent_runs": Store(_settings.db_path).recent_runs(5)}


@mcp.tool(name="wake_update_state")
def wake_update_state(time: str | None = None, timezone: str | None = None, enabled: bool | None = None,
                      personality: str | None = None, opening_line: str | None = None) -> dict:
    """Change the wake-up schedule or persona. Validated; the daemon reschedules immediately."""
    cfg = load_config(_settings.config_path)
    raw = cfg.raw
    if time:
        raw["schedule"]["time"] = time
    if timezone:
        raw["schedule"]["timezone"] = timezone
    if enabled is not None:
        raw["schedule"]["enabled"] = enabled
    if personality:
        raw["personality"] = personality
    if opening_line:
        raw["opening_line"] = opening_line
    return _daemon("PUT", "/api/config", json=raw)


@mcp.tool(name="wake_schedule_oneoff")
def wake_schedule_oneoff(in_minutes: float = 3, test: bool = True) -> dict:
    """Schedule a one-off run a few minutes from now (to verify the unattended path end to end)."""
    return _daemon("POST", "/api/schedule/oneoff", json={"in_minutes": in_minutes, "test": test})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
