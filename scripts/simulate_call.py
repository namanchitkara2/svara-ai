"""Run the REAL wake/call stack against a simulated person (no phone is dialled).

  uv run python scripts/simulate_call.py wake_sleepy
  uv run python scripts/simulate_call.py sales_discovery
  uv run python scripts/simulate_call.py party_invite_hindi_noisy
  uv run python scripts/simulate_call.py all

Everyone in SCENARIOS is invented: a sleepy housemate, a sales prospect, a marketing
feedback call and three friends being invited to the same party.

Prints both sides of the conversation, reply latencies, and saves a stereo WAV
(left = agent, right = simulated person) under ~/.wake-agent/sim/.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time

from agents.state_machine import WakeStateMachine
from agents.wake_call import WakeCall
from apps.runtime import make_voice_factory
from services.config import load_config, load_settings
from services.observability import setup
from services.store import Store
from services.whatsapp.sim import Persona, SimWhatsApp

SCENARIOS = {
    # Every person below is invented. Each persona exercises one hard part of the stack: the
    # wake-verification gate, silence, an early hangup, barge-in, a long open-ended answer, and a
    # noisy Hindi line. A scenario with an `objective` runs as a custom call (no wake gate); the
    # three wake_* ones run the wake flow.
    "wake_sleepy": dict(
        persona=Persona(
            "Asha", "You were fast asleep and the call woke you. You're groggy and want five more minutes. "
            "You resist at first, then slowly wake up if the caller is persistent and warm. You sit up only "
            "after being asked twice. Eventually you admit you're up and have work at 10.",
            language="en", voice="Magpie-Multilingual.EN-US.Sofia"),
    ),
    "wake_silent": dict(
        persona=Persona(
            "Asha", "You are very deeply asleep. The first two times the caller talks you don't respond at all "
            "(action silent). After that you mumble, then wake up slowly.", language="en",
            voice="Magpie-Multilingual.EN-US.Sofia"),
    ),
    "wake_annoyed_hangup": dict(
        persona=Persona("Asha", "You are annoyed at being woken by a bot. You say you're awake just to get rid of "
                        "it and hang up after the second exchange.", language="en",
                        voice="Magpie-Multilingual.EN-US.Sofia"),
    ),
    # Sales: a qualification call that has to survive being interrupted.
    "sales_discovery": dict(
        name="Dev Malhotra", voice="Magpie-Multilingual.EN-US.Sofia",
        opening="Hi Dev, this is Emely, an AI assistant calling for Northwind Studio. Two minutes about your "
                "delivery tracking, is now alright?",
        objective="You are Emely, an AI assistant for Northwind Studio, a fictional logistics software company. "
                  "Qualify Dev in three questions: how many deliveries a day his team handles, what they use to "
                  "track them today, and who signs off on new tools. Keep every answer to two sentences. If he is "
                  "interested, offer a 20-minute demo on Thursday. Never invent pricing: if he asks, say you will "
                  "have someone send the current rate card.",
        persona=Persona("Dev Malhotra", "You are an impatient operations lead. You interrupt long answers, ask sharp "
                        "questions ('how is this different from what we have?', 'what does it cost?'), and want "
                        "short answers. You end up mildly interested but will not commit on the call.",
                        language="en", voice="Magpie-Multilingual.EN-US.Leo", interrupt_after_s=3.0, max_turns=6),
    ),
    # Marketing: an open-ended feedback call, where the person talks for a long time.
    "marketing_feedback": dict(
        name="Riya Nair", voice="Magpie-Multilingual.EN-US.Aria",
        opening="Hi Riya, this is Emely, an AI assistant from Northwind Studio. You finished the free trial last "
                "week, and I would love two minutes of honest feedback.",
        objective="You are Emely, collecting trial feedback for the fictional Northwind Studio. Ask what she was "
                  "trying to get done, what got in the way, and how likely she is to recommend it out of ten. Let "
                  "her talk without cutting in, reflect back what she says in a few words, do not defend the "
                  "product, and do not offer any discount. Thank her and end the call.",
        persona=Persona("Riya Nair", "You are a friendly marketing manager who gives long, rambling answers with "
                        "real detail: onboarding was confusing, the reporting view was the best part, and your team "
                        "never finished the import. You rate it seven out of ten.",
                        language="en", voice="Magpie-Multilingual.EN-US.Sofia"),
    ),
    # Inviting several friends to the same party: the same objective, three very different people.
    "party_invite_keen": dict(
        name="Kabir", voice="Magpie-Multilingual.EN-US.Sofia",
        opening="Hey Kabir, it's Emely, Sam's AI assistant. Sam is throwing a small house party on Saturday at "
                "eight, and he asked me to check if you are in.",
        objective="You are Emely, Sam's AI assistant, inviting a friend to Sam's house party: Saturday, 8 pm, at "
                  "Sam's place, about fifteen people, dinner is covered. Get a clear yes, no or maybe, ask whether "
                  "he is bringing anyone, and mention that Sam will send the address on WhatsApp. Facts you do not "
                  "have (parking, who else is coming) you say you will check with Sam.",
        persona=Persona("Kabir", "You are delighted to be invited. You say yes straight away, ask who else is "
                        "coming and whether you should bring something, and joke about being late as usual.",
                        language="en", voice="Magpie-Multilingual.EN-US.Leo"),
    ),
    "party_invite_busy": dict(
        name="Meera", voice="Magpie-Multilingual.EN-US.Sofia",
        opening="Hi Meera, this is Emely, Sam's AI assistant. Sam is having people over on Saturday at eight and "
                "he wanted me to ask you directly.",
        objective="You are Emely, Sam's AI assistant, inviting a friend to Sam's house party: Saturday, 8 pm, at "
                  "Sam's place, dinner covered. She has a conflict, so find out whether a late arrival works, take "
                  "a clear maybe if that is the honest answer, and tell her Sam will text the address. Do not "
                  "pressure her and do not invent a second date.",
        persona=Persona("Meera", "You have a cousin's engagement dinner that evening. You are warm but torn, ask "
                        "how late people will still be there, and land on a maybe. You are slightly thrown that an "
                        "AI is calling and ask about it once.",
                        language="en", voice="Magpie-Multilingual.EN-US.Aria"),
    ),
    "party_invite_hindi_noisy": dict(
        language="hi", name="Nikhil", voice="Magpie-Multilingual.HI-IN.Sofia",
        opening="नमस्ते निखिल! मैं एमली हूँ, सैम की AI असिस्टेंट। सैम शनिवार रात आठ बजे घर पर छोटी सी पार्टी रख रहे हैं, "
                "और उन्होंने आपको बुलाने के लिए कहा है।",
        objective="You are Emely, Sam's AI assistant, inviting a friend in Hindi (respectful, natural Hinglish is "
                  "fine) to Sam's house party: Saturday, 8 pm, at Sam's place, dinner covered. Get a yes, no or "
                  "maybe, ask if he is bringing his wife, and say Sam will send the address. Anything you were not "
                  "told, say you will check with Sam.",
        persona=Persona("Nikhil", "You are at home with the TV on loud. You speak Hindi. You are happy to be "
                        "invited but distracted: you ask who else is coming, whether you can bring your wife, and "
                        "whether there is parking. You say yes in the end.",
                        language="hi", voice="Magpie-Multilingual.HI-IN.Leo", noise_rms=700),
    ),
}


async def run(name: str) -> dict:
    sc = SCENARIOS[name]
    s = load_settings()
    cfg = load_config()
    if sc.get("name"):
        cfg.raw["contact"]["name"] = sc["name"]
    if sc.get("voice"):
        cfg.raw["voice"]["tts_voice_override"] = sc["voice"]
    cfg.raw["maximum_call_duration_minutes"] = 3
    wav = s.home / "sim" / f"{name}-{int(time.time())}.wav"
    wa = SimWhatsApp(sc["persona"], api_key=s.nvidia_api_key, riva_server=s.riva_server,
                     tts_fid=s.riva_tts_function_id, asr_fid=s.riva_asr_function_id,
                     asr_multi_fid=s.riva_asr_multilingual_function_id, llm_base_url=s.nvidia_llm_base_url,
                     llm_model=cfg.voice.get("llm_model"), record_to=wav)
    store = Store(s.home / "data" / "sim.db")
    run_id = store.create_run(fire_key=f"sim:{name}:{time.time()}", trigger="sim", contact_name=cfg.contact_name,
                              contact_phone=cfg.contact_phone, wake_schedule=None)
    call = WakeCall(wa=wa, voice_factory=make_voice_factory(s, cfg, sc.get("language")), cfg=cfg, store=store,
                    run_id=run_id, attempt=1, sm=WakeStateMachine(run_id), show_transcript=True,
                    opening_override=sc.get("opening"), custom_objective=sc.get("objective"),
                    language=sc.get("language"), max_minutes=3)
    print(f"\n==================== {name} ====================")
    res = await call.run()
    await wa.close()
    rep = wa.report
    print("\n--- simulated person's side ---")
    for t in rep.turns:
        print(f"  {t.t:5.1f}s {t.who}: {t.text}")
    lat = rep.reply_latencies
    out = {
        "scenario": name, "state": res.state.value, "wake_confirmed": res.wake_confirmed,
        "ended_by": res.ended_by, "secs": round(res.duration_s), "agent_turns": res.agent_turns,
        "user_turns": res.user_turns, "barge_ins": res.barge_ins,
        "reply_latency_median": round(statistics.median(lat), 2) if lat else None,
        "reply_latency_max": round(max(lat), 2) if lat else None,
        "stop_after_interrupt_s": [round(x, 2) for x in rep.stop_latencies] or None,
        "wav": rep.wav,
    }
    print("\n--- result ---")
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


async def main() -> None:
    setup(load_settings().log_dir)
    names = list(SCENARIOS) if sys.argv[1:] == ["all"] else sys.argv[1:] or ["wake_sleepy"]
    results = [await run(n) for n in names]
    if len(results) > 1:
        print("\n==================== summary ====================")
        for r in results:
            print(f"  {r['scenario']:18} {r['state']:10} awake={r['wake_confirmed']!s:5} "
                  f"lat_med={r['reply_latency_median']} barge={r['barge_ins']} secs={r['secs']} "
                  f"stop_after_interrupt={r['stop_after_interrupt_s']}")


if __name__ == "__main__":
    asyncio.run(main())
