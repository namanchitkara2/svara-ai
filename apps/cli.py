"""wake-agent CLI.

  wake-agent serve                                   run the unattended daemon (scheduler + API)
  wake-agent pair                                    show WhatsApp pairing status / how to pair
  wake-agent test     --contact "+919999999999"      short test call: "Hello, can you hear me?"
  wake-agent call-now --contact "+919999999999"      full wake-up run right now (with retries)
  wake-agent schedule --time 08:00 --timezone Asia/Kolkata --contact "+919999999999"
  wake-agent schedule --in-minutes 3 [--test]        one-off near-future run (needs `serve` running)
  wake-agent status                                  schedule, next run, recent runs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

from services.config import load_config, load_settings, mask_phone, normalize_phone, save_config
from services.observability import setup


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_serve(_a) -> None:
    from apps.daemon import serve

    asyncio.run(serve())


def cmd_call(a, test: bool) -> None:
    from apps.runtime import run_wake

    s = load_settings()
    cfg = load_config(s.config_path)
    setup(s.log_dir, cfg.schedule["timezone"], verbose=a.verbose)
    phone = normalize_phone(a.contact) if a.contact else cfg.contact_phone
    if phone != cfg.contact_phone and not a.allow_other:
        sys.exit(f"refusing to call {mask_phone(phone)}: not the configured contact (pass --allow-other)")
    if a.name:
        cfg.raw["contact"]["name"] = a.name  # this call only; wake.yaml is not modified
    if a.voice:
        cfg.raw.setdefault("voice", {})["tts_voice_override"] = a.voice
    if getattr(a, "live", False):
        cfg.raw.setdefault("voice", {}).update({"provider": "gemini_live"})
    if getattr(a, "live_model", None):
        cfg.raw.setdefault("voice", {})["live_model"] = a.live_model
    if getattr(a, "live_voice", None):
        cfg.raw.setdefault("voice", {})["live_voice"] = a.live_voice
    if getattr(a, "live_languages", None):
        cfg.raw.setdefault("voice", {})["live_languages"] = [x.strip() for x in a.live_languages.split(",")]
    res = asyncio.run(run_wake(cfg=cfg, trigger="custom" if a.objective else ("test" if test else "manual"), phone=phone,
                               test_mode=test, use_supervisor=not a.no_supervisor,
                               opening=a.opening, objective=a.objective, show_transcript=a.show_transcript,
                               language=a.language, max_minutes=a.max_minutes))
    _print(res)


def cmd_pair(a) -> None:
    s = load_settings()

    async def go():
        from services.whatsapp.wacalls import WaCallsProvider

        wa = WaCallsProvider(s.wacalls_url, s.wacalls_api_token, s.wacalls_session_id)
        try:
            await wa.connect()
            sessions = await wa.list_sessions()
            if not sessions:
                sid = await wa.create_session("wake-agent")
                sessions = await wa.list_sessions()
                print(f"created session {sid[:8]}")
            for x in sessions:
                print(f"session {x['id'][:8]}  state={x.get('state')}  paired={x.get('paired')}")
            if any(x.get("paired") and x.get("state") == "open" for x in sessions):
                print("\nWhatsApp is paired and connected.")
                return
            if getattr(a, "phone", None):
                await _pair_by_phone(wa, sessions[0]["id"], a.phone)
            else:
                await _show_qr_until_paired(wa, s, sessions[0]["id"])
        finally:
            await wa.close()

    asyncio.run(go())


async def _pair_by_phone(wa, sid: str, phone: str) -> None:
    """Link via an 8-character code instead of a QR (uses our WaCalls pair-phone patch)."""
    import subprocess
    import sys as _sys

    digits = normalize_phone(phone).lstrip("+")
    r = await wa._req("POST", f"/api/sessions/{sid}/pair-phone", json={"phone": digits})
    print(f"\nLinking code: {r.json()['code']}\n"
          "Enter it on the phone: WhatsApp > Linked devices > Link a device > Link with phone number instead\n")
    for _ in range(60):
        await asyncio.sleep(3)
        ses = next((x for x in await wa.list_sessions() if x["id"] == sid), {})
        if ses.get("paired"):
            print("Linked.")
            if _sys.platform == "darwin":
                # whatsmeow doesn't reconnect after phone-code pairing; a restart restores the session
                import os
                subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/com.wakeagent.wacalls"],
                               check=False, capture_output=True)
                print("Restarted WaCalls so the new session connects.")
            return
    print("Not linked yet (code expired?). Run the command again for a new code.")


async def _show_qr_until_paired(wa, s, sid: str) -> None:
    """Render WaCalls' rotating pairing QR to a local PNG window until the phone scans it."""
    import json as _json
    import subprocess

    import segno

    page = s.home / "pair-qr.html"   # short-lived; deleted once paired
    page.write_text('<meta http-equiv="refresh" content="2"><body style="font:16px system-ui;text-align:center">'
                    '<p>Waiting for QR...</p></body>')
    page.chmod(0o600)
    print("\nScan with your phone: WhatsApp → Settings → Linked devices → Link a device")
    await wa._req("POST", f"/api/sessions/{sid}/pair")
    opened = False
    try:
        async with wa._http.stream("GET", "/api/events", timeout=httpx.Timeout(15, read=90)) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                ev = _json.loads(line[6:])
                if ev.get("sessionId") != sid:
                    continue
                qr = ev.get("qr")
                if ev.get("type") in ("session-qr", "auth-state") and qr:
                    svg = segno.make(qr, error="l").svg_inline(scale=8, border=4)
                    page.write_text('<meta http-equiv="refresh" content="2"><body style="font:16px system-ui;'
                                    'text-align:center;background:#fff"><h3>WhatsApp → Linked devices → Link a device'
                                    f'</h3>{svg}<p>Refreshes automatically.</p></body>')
                    if not opened:
                        subprocess.run(["open", str(page)], check=False)
                        opened = True
                    print("  QR ready in your browser (auto-refreshes)")
                if ev.get("type") == "auth-state" and ev.get("paired") and ev.get("state") == "open":
                    page.write_text('<body style="font:20px system-ui;text-align:center"><h2>Paired. You can close this tab.</h2></body>')
                    print("\nPaired and connected.")
                    await asyncio.sleep(3)
                    return
                if ev.get("type") == "auth-state" and ev.get("state") == "logged_out":
                    print("  QR expired, requesting a new one...")
                    await wa._req("POST", f"/api/sessions/{sid}/pair")
    finally:
        page.unlink(missing_ok=True)


def cmd_schedule(a) -> None:
    s = load_settings()
    if a.in_minutes is not None:
        r = httpx.post(f"http://{s.api_host}:{s.api_port}/api/schedule/oneoff",
                       headers={"Authorization": f"Bearer {s.api_token}"},
                       json={"in_minutes": a.in_minutes, "test": a.test,
                             "phone": normalize_phone(a.contact) if a.contact else None}, timeout=10)
        _print(r.json())
        return
    cfg = load_config(s.config_path)
    raw = cfg.raw
    if a.time:
        raw["schedule"]["time"] = a.time
    if a.timezone:
        raw["schedule"]["timezone"] = a.timezone
        raw["contact"]["timezone"] = a.timezone
    if a.contact:
        raw["contact"]["phone"] = normalize_phone(a.contact)
    raw["schedule"]["enabled"] = not a.disable
    save_config(raw, s.config_path)
    print(f"saved: {raw['schedule']['time']} {raw['schedule']['timezone']} -> {mask_phone(raw['contact']['phone'])}"
          f" ({'enabled' if not a.disable else 'disabled'})")
    try:
        r = httpx.put(f"http://{s.api_host}:{s.api_port}/api/config",
                      headers={"Authorization": f"Bearer {s.api_token}"}, json=raw, timeout=10)
        print("daemon rescheduled, next run:", r.json().get("next_run"))
    except httpx.HTTPError:
        print("daemon not running; it will pick this up on start (wake-agent serve)")


def cmd_status(_a) -> None:
    s = load_settings()
    try:
        r = httpx.get(f"http://{s.api_host}:{s.api_port}/api/status",
                      headers={"Authorization": f"Bearer {s.api_token}"}, timeout=5)
        _print(r.json())
    except httpx.HTTPError:
        from services.store import Store

        print("daemon NOT running. Recent runs from the database:")
        _print(Store(s.db_path).recent_runs(5))


def main() -> None:
    p = argparse.ArgumentParser(prog=Path(sys.argv[0]).name or "svara")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    pp = sub.add_parser("pair")
    pp.add_argument("--phone", help="link by phone number (8-char code) instead of QR, e.g. +91XXXXXXXXXX")
    sub.add_parser("status")
    for name in ("test", "call-now"):
        sp = sub.add_parser(name)
        sp.add_argument("--contact")
        sp.add_argument("--allow-other", action="store_true", help="allow calling a number other than the config")
        sp.add_argument("--no-supervisor", action="store_true", help="skip the LLM supervisor (single attempt)")
        sp.add_argument("--opening", help="override the first line spoken (this call only)")
        sp.add_argument("--objective", help="custom purpose for a one-off non-wake call (this call only)")
        sp.add_argument("--show-transcript", action="store_true",
                        help="print both sides of the conversation to this console (never logged/stored)")
        sp.add_argument("--language", choices=["en", "hi"], help="conversation language (hi = Hindi/Hinglish)")
        sp.add_argument("--live", action="store_true",
                        help="use Gemini Live native speech-to-speech instead of the NVIDIA cascade")
        sp.add_argument("--live-model", help="Gemini Live model (default gemini-2.5-flash-native-audio-latest)")
        sp.add_argument("--live-languages", help="comma-separated, e.g. pa-IN,hi-IN,en-IN")
        sp.add_argument("--live-voice", help="Gemini Live voice, e.g. Aoede, Puck, Charon, Kore")
        sp.add_argument("--voice", help="TTS voice for this call, e.g. Magpie-Multilingual.HI-IN.Sofia")
        sp.add_argument("--name", help="who is being called (this call only)")
        sp.add_argument("--max-minutes", type=float, help="cap for this call (max 15)")
        sp.add_argument("-v", "--verbose", action="store_true")
    sc = sub.add_parser("schedule")
    sc.add_argument("--time")
    sc.add_argument("--timezone")
    sc.add_argument("--contact")
    sc.add_argument("--disable", action="store_true")
    sc.add_argument("--in-minutes", type=float)
    sc.add_argument("--test", action="store_true")
    a = p.parse_args()
    {
        "serve": cmd_serve,
        "pair": cmd_pair,
        "status": cmd_status,
        "test": lambda a: cmd_call(a, test=True),
        "call-now": lambda a: cmd_call(a, test=False),
        "schedule": cmd_schedule,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
