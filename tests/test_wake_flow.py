"""End-to-end wake logic with fake providers: state machine, verification gate, retries, failures."""

import asyncio
import copy

import pytest
import yaml

from agents.state_machine import S, WakeStateMachine, IllegalTransition
from agents.tools import RunContext, WakeTools
from agents.wake_call import WakeCall
from services.config import EXAMPLE_CONFIG, WakeConfig
from services.store import DuplicateRun, Store
from tests.fakes import FakeWhatsApp, ScriptedVoice


@pytest.fixture
def cfg():
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["verification"] = {"min_user_turns": 2, "silence_nudge_seconds": 0.6}
    raw["retry"] = {"enabled": True, "max_attempts": 3, "delay_seconds": 30}
    return WakeConfig(raw)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "w.db")


def _run(store, cfg):
    return store.create_run(fire_key=f"t:{id(cfg)}:{asyncio.get_event_loop().time()}", trigger="test",
                            contact_name="GF", contact_phone=cfg.contact_phone, wake_schedule="08:00")


async def _attempt(store, cfg, wa, voice, attempt=1):
    run_id = _run(store, cfg)
    sm = WakeStateMachine(run_id)
    call = WakeCall(wa=wa, voice_factory=lambda: voice, cfg=cfg, store=store, run_id=run_id, attempt=attempt, sm=sm)
    res = await asyncio.wait_for(call.run(), 15)
    return res, sm


async def test_happy_path_confirms_and_hangs_up(store, cfg):
    wa = FakeWhatsApp()
    voice = ScriptedVoice([
        "five more minutes",
        ("report_wake_evidence", {"says_awake": True}),
        "I'm sitting up",
        ("report_wake_evidence", {"sitting_up": True, "feet_on_floor": True}),
        ("awake_confirmed", {}),
        ("end_call", {"reason": "awake_confirmed"}),
    ])
    res, sm = await _attempt(store, cfg, wa, voice)
    assert res.state == S.COMPLETED and res.wake_confirmed and res.ended_by == "agent"
    assert wa.hangups >= 1
    for s in (S.CALLING, S.RINGING, S.ANSWERED, S.GREETING, S.CONVERSING, S.WAKE_VERIFICATION,
              S.AWAKE_CONFIRMED, S.GOODBYE, S.COMPLETED):
        assert s in sm.history


async def test_early_awake_claim_is_rejected(store, cfg):
    wa = FakeWhatsApp(hangup_after=2.0)
    voice = ScriptedVoice([("awake_confirmed", {}), ("end_call", {"reason": "awake_confirmed"})])
    res, _ = await _attempt(store, cfg, wa, voice)
    assert not res.wake_confirmed          # 0 user turns < min_user_turns
    assert any("too early" in h for h in voice.hints) or any("cannot end" in h for h in voice.hints)


async def test_no_answer(store, cfg):
    res, sm = await _attempt(store, cfg, FakeWhatsApp(answer=False, end_reason_if_no_answer="timeout"),
                             ScriptedVoice([]))
    assert res.state == S.NO_ANSWER and res.end_reason == "timeout" and not res.stop_requested


async def test_decline_while_ringing_blocks_retry(store, cfg):
    # WaCalls reports a decline during ringing as user_ended
    res, _ = await _attempt(store, cfg, FakeWhatsApp(answer=False, end_reason_if_no_answer="user_ended"),
                            ScriptedVoice([]))
    assert res.end_reason == "declined" and res.stop_requested
    from agents.tools import deterministic_should_retry
    assert not deterministic_should_retry(res)


async def test_she_hangs_up(store, cfg):
    res, _ = await _attempt(store, cfg, FakeWhatsApp(hangup_after=0.5), ScriptedVoice(["<silence>"]))
    assert res.state == S.COMPLETED and res.ended_by == "her" and not res.wake_confirmed


async def test_dead_audio_fails_safe(store, cfg, monkeypatch):
    import agents.wake_call as wc
    monkeypatch.setattr(wc, "FIRST_AUDIO_TIMEOUT_S", 0.5)
    wa = FakeWhatsApp(audio=False)
    res, _ = await _attempt(store, cfg, wa, ScriptedVoice(["<silence>"]))
    assert res.state == S.AUDIO_FAILED and wa.hangups >= 1


async def test_silence_triggers_nudge(store, cfg):
    wa = FakeWhatsApp(hangup_after=4.5)
    voice = ScriptedVoice(["<silence>"])
    await _attempt(store, cfg, wa, voice)
    assert any("silent" in h for h in voice.hints)


async def test_stop_request_ends_and_blocks_retry(store, cfg):
    wa = FakeWhatsApp()
    voice = ScriptedVoice(["please stop calling me", ("end_call", {"reason": "she_asked_to_stop"})])
    res, _ = await _attempt(store, cfg, wa, voice)
    assert res.stop_requested and res.state == S.COMPLETED
    ctx = RunContext(run_id=1, cfg=cfg, trigger="t", phone=cfg.contact_phone, attempts=[res])
    assert WakeTools(ctx, wa, lambda: voice, store).can_call()[0] is False


async def test_attempt_ceiling(store, cfg):
    c = copy.deepcopy(cfg.raw)
    c["retry"] = {"enabled": True, "max_attempts": 99, "delay_seconds": 0}
    capped = WakeConfig(c).retry
    assert capped["max_attempts"] == 5 and capped["delay_seconds"] >= 30   # never infinite / never hammering


def test_duplicate_fire_key_is_rejected(store, cfg):
    store.create_run(fire_key="schedule:2026-09-23", trigger="schedule", contact_name="GF",
                     contact_phone="+911", wake_schedule="08:00")
    with pytest.raises(DuplicateRun):
        store.create_run(fire_key="schedule:2026-09-23", trigger="schedule", contact_name="GF",
                         contact_phone="+911", wake_schedule="08:00")


def test_illegal_transition():
    sm = WakeStateMachine(1)
    with pytest.raises(IllegalTransition):
        sm.to(S.AWAKE_CONFIRMED)


async def test_stop_request_honoured_even_without_llm_marker(store, cfg):
    wa = FakeWhatsApp(hangup_after=1.5)
    voice = ScriptedVoice(["I said I'm up. Just stop calling me."])   # no end_call marker at all
    res, _ = await _attempt(store, cfg, wa, voice)
    assert res.stop_requested


def test_stop_patterns():
    from agents.wake_call import STOP_RE
    for t in ["Stop calling me", "don't call again", "कॉल मत करो", "बंद करो ये", "band karo yaar"]:
        assert STOP_RE.search(t), t
    for t in ["five more minutes", "I'm up", "stop it's too early"]:
        assert not STOP_RE.search(t) or t.startswith("stop it"), t


async def test_followup_checks_call_back_and_stop_rules(cfg, monkeypatch):
    """After a confirmed wake-up we call back; we stop if she doesn't answer or asked us to stop."""
    import apps.runtime as rt

    cfg.raw["followup"] = {"enabled": True, "after_minutes": 10, "max_checks": 2}
    slept: list[float] = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(rt.asyncio, "sleep", fake_sleep)

    class Tools:
        def __init__(self, answered):
            self.ctx = type("C", (), {"attempts": []})()
            self.calls = []
            self._answered = answered

        async def followup_call(self, opening, objective):
            self.calls.append((opening, objective))
            return {"answered": self._answered, "state": "COMPLETED"}

    t = Tools(answered=True)
    await rt._followup_checks(t, cfg, 1)
    assert len(t.calls) == 2 and slept == [600, 600]          # two checks, 10 minutes apart
    assert "still up" in t.calls[0][0]

    t2 = Tools(answered=False)
    await rt._followup_checks(t2, cfg, 1)
    assert len(t2.calls) == 1                                  # no answer -> she's up and busy, stop

    t3 = Tools(answered=True)
    t3.ctx.attempts = [type("R", (), {"stop_requested": True})()]
    await rt._followup_checks(t3, cfg, 1)
    assert t3.calls == []                                      # she asked us to stop
