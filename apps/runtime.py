"""Composition root: builds providers from config and runs one wake-up objective."""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from agents.supervisor import WakeSupervisor
from agents.tools import RunContext, WakeTools
from services.config import Settings, WakeConfig, load_config, load_settings, mask_phone, normalize_phone
from services.observability import event
from services.store import DuplicateRun, Store, now_iso
from services.voice.gemini_live import GeminiLiveVoiceProvider
from services.voice.nvidia import NvidiaVoiceProvider
from services.whatsapp.wacalls import WaCallsProvider

# one wake run at a time per process (the DB fire_key covers cross-process duplicates)
_run_lock = asyncio.Lock()


def make_whatsapp(settings: Settings) -> WaCallsProvider:
    return WaCallsProvider(settings.wacalls_url, settings.wacalls_api_token, settings.wacalls_session_id)


LANGUAGE_PRESETS = {
    # English: fast en-US streaming Parakeet + Aria
    "en": {},
    # Hindi / Hinglish: multilingual Parakeet in code-mixed mode + Hindi Magpie voice
    "hi": {"asr": "multilingual", "asr_language_code": "multi", "language_code": "hi-IN",
           "tts_voice": "Magpie-Multilingual.HI-IN.Leo"},
}


def make_voice_factory(settings: Settings, cfg: WakeConfig, language: str | None = None):
    v = dict(cfg.voice)
    preset = LANGUAGE_PRESETS.get(language or v.get("language", "en"), {})
    v.update(preset)
    if v.get("tts_voice_override"):
        v["tts_voice"] = v["tts_voice_override"]
    asr_fid = settings.riva_asr_multilingual_function_id if v.get("asr") == "multilingual" else settings.riva_asr_function_id

    def factory():
        if v.get("provider") == "gemini_live":
            # native speech-to-speech: the model hears her voice and speaks back (no TTS chain)
            return GeminiLiveVoiceProvider(
                settings.gemini_api_key,
                model=v.get("live_model", "gemini-2.5-flash-native-audio-latest"),
                voice=v.get("live_voice", "Aoede"),
                language_code=v.get("language_code", "en-US"),
                silence_ms=int(v.get("live_silence_ms", 900)),
                languages=v.get("live_languages") or ["en-IN", "hi-IN", "pa-IN"],
                pronunciations=v.get("pronunciations") or {},
            )
        return NvidiaVoiceProvider(
            settings.nvidia_api_key,
            llm_model=v.get("llm_model", "nvidia/nemotron-3-super-120b-a12b"),
            llm_fallback_models=v.get("llm_fallback_models") or [],
            llm_base_url=settings.nvidia_llm_base_url,
            riva_server=settings.riva_server,
            asr_function_id=asr_fid,
            tts_function_id=settings.riva_tts_function_id,
            tts_voice=v.get("tts_voice", "Magpie-Multilingual.EN-US.Aria"),
            language_code=v.get("language_code", "en-US"),
            asr_language_code=v.get("asr_language_code"),
            pronunciations=v.get("pronunciations") or {},
            extra_llms={"gemini": (settings.gemini_base_url, settings.gemini_api_key)},
        )

    return factory


def scheduled_fire_key(cfg: WakeConfig, when: datetime | None = None) -> str:
    tz = ZoneInfo(cfg.schedule["timezone"])
    d = (when or datetime.now(tz)).astimezone(tz).date().isoformat()
    return f"schedule:{d}"


async def run_wake(
    *,
    trigger: str,
    fire_key: str | None = None,
    phone: str | None = None,
    test_mode: bool = False,
    settings: Settings | None = None,
    cfg: WakeConfig | None = None,
    use_supervisor: bool = True,
    opening: str | None = None,
    objective: str | None = None,
    show_transcript: bool = False,
    language: str | None = None,
    max_minutes: float | None = None,
) -> dict:
    settings = settings or load_settings()
    cfg = cfg or load_config(settings.config_path)
    store = Store(settings.db_path)
    target = normalize_phone(phone) if phone else cfg.contact_phone
    fire_key = fire_key or f"{trigger}:{now_iso()}"

    try:
        run_id = store.create_run(
            fire_key=fire_key, trigger=trigger, contact_name=cfg.contact_name, contact_phone=target,
            wake_schedule=f"{cfg.schedule['time']} {cfg.schedule['timezone']}",
        )
    except DuplicateRun:
        event("DUPLICATE_RUN_SKIPPED", fire_key=fire_key)
        return {"skipped": True, "reason": "duplicate", "fire_key": fire_key}

    if _run_lock.locked():
        store.update_run(run_id, state="CALL_FAILED", error="another wake run is in progress", finished_at=now_iso())
        event("RUN_REJECTED_BUSY", run=run_id)
        return {"run_id": run_id, "skipped": True, "reason": "busy"}

    async with _run_lock:
        event("WAKE_RUN_STARTED", run=run_id, trigger=trigger, to=mask_phone(target), test=test_mode or None)
        wa = make_whatsapp(settings)
        ctx = RunContext(run_id=run_id, cfg=cfg, trigger=trigger, phone=target, test_mode=test_mode or bool(objective),
                         opening_override=opening, custom_objective=objective,
                         show_transcript=show_transcript, language=language, max_minutes=max_minutes)
        tools = WakeTools(ctx, wa, make_voice_factory(settings, cfg, language), store)
        try:
            if use_supervisor and not ctx.test_mode:
                sup = WakeSupervisor(tools, settings.nvidia_api_key, settings.nvidia_llm_base_url,
                                     cfg.voice.get("llm_model", "nvidia/nemotron-3-super-120b-a12b"))
                await sup.run()
            else:
                await tools.whatsapp_call()
                confirmed = any(a.wake_confirmed for a in ctx.attempts)
                await tools.wake_finish(confirmed, ctx.attempts[-1].summary if ctx.attempts else "")
        except Exception as e:
            store.update_run(run_id, error=f"{type(e).__name__}: {e}"[:300])
            event("WAKE_RUN_ERROR", run=run_id, err=type(e).__name__)
        finally:
            await wa.close()

        if ctx.final_confirmed and cfg.followup["enabled"] and trigger != "test":
            await _followup_checks(tools, cfg, run_id)

        last = ctx.attempts[-1] if ctx.attempts else None
        final_state = "COMPLETED" if ctx.final_confirmed else (last.state.value if last else "AGENT_ERROR")
        if last and last.state.value == "COMPLETED" and not ctx.final_confirmed:
            final_state = "COMPLETED"  # call worked, but not confirmed (e.g. she hung up / asked to stop)
        store.update_run(run_id, state=final_state, wake_confirmed=int(ctx.final_confirmed),
                         summary=ctx.final_summary or None, finished_at=now_iso())
        event("WAKE_RUN_FINISHED", run=run_id, state=final_state, awake=ctx.final_confirmed,
              attempts=len(ctx.attempts))
        return {
            "run_id": run_id, "state": final_state, "wake_confirmed": ctx.final_confirmed,
            "attempts": [a.brief() for a in ctx.attempts], "summary": ctx.final_summary,
        }


async def run_wake_oneoff(phone: str | None = None, test_mode: bool = False, fire_key: str | None = None,
                          name: str | None = None, opening: str | None = None, objective: str | None = None,
                          live: bool = True) -> dict:
    """Target of one-off scheduled jobs (serialised by reference in the job store)."""
    event("WAKE_SCHEDULE_TRIGGERED", kind="one-off")
    cfg = load_config()
    if name:
        cfg.raw["contact"]["name"] = name
    if opening:
        cfg.raw["opening_line"] = opening
    if live:
        cfg.raw.setdefault("voice", {})["provider"] = "gemini_live"
    return await run_wake(trigger="oneoff", fire_key=fire_key, phone=phone, test_mode=test_mode, cfg=cfg,
                          objective=objective)


async def _followup_checks(tools, cfg: WakeConfig, run_id: int) -> None:
    """They said they were awake. Call back a few minutes later to make sure they didn't fall asleep."""
    f = cfg.followup
    name = cfg.contact_name
    for i in range(1, f["max_checks"] + 1):
        if any(a.stop_requested for a in tools.ctx.attempts):
            return
        event("FOLLOWUP_WAIT", run=run_id, minutes=f["after_minutes"], check=i)
        await asyncio.sleep(f["after_minutes"] * 60)
        opening = (f"Hey {name}, it's {cfg.caller_name}'s assistant again, just checking in. "
                   "You're still up, right? Not back under the blanket?")
        objective = (
            f"This is a check-back call {f['after_minutes']} minutes after {name} confirmed they were awake. "
            "Find out whether they are still up and out of bed, or fell asleep again. If they are up and doing "
            "something (getting ready, coffee, work), be warm, say well done, and end the call quickly. "
            "If they sound asleep or admit they went back to bed, wake them up properly: be kind but "
            "persistent, get them to sit up and put their feet on the floor, and only then let them go. "
            "If they are annoyed or ask you to stop, apologise, say goodbye and end the call."
        )
        res = await tools.followup_call(opening, objective)
        event("FOLLOWUP_DONE", run=run_id, check=i, state=res.get("state"), answered=res.get("answered"))
        if not res.get("answered"):
            return  # she is probably up and busy; don't keep calling


def mark_orphans(store: Store) -> int:
    """A crash/restart mid-run leaves rows without finished_at. Close them honestly."""
    n = 0
    for r in store.orphaned_runs():
        store.update_run(r["id"], state="AGENT_ERROR", error="process restarted mid-run", finished_at=now_iso())
        event("ORPHAN_RUN_CLOSED", run=r["id"], state=r["state"])
        n += 1
    return n
