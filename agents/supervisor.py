"""The high-level wake agent.

An LLM (Nemotron on NVIDIA NIM, OpenAI-compatible function calling) owns the wake-up objective
and drives it ONLY through agents/tools.py. It decides whether to call again after a failed or
inconclusive attempt, and writes the final summary. It never has shell/filesystem access.

Unattended safety: if the LLM is unreachable, slow, or misbehaves, a deterministic policy takes
over (call → retry per config → finish). The 08:00 call does not depend on the LLM being up.
"""

from __future__ import annotations

import asyncio
import json
import logging

from openai import AsyncOpenAI

from agents.tools import RunContext, WakeTools, deterministic_should_retry, openai_tool_schema
from services.observability import event

log = logging.getLogger("wake.supervisor")

MAX_STEPS = 14

SYSTEM = """You are the supervisor of an automated wake-up call service.
Your objective: {objective}
Contact: {contact}. You act only through your tools.

Procedure:
1. Call whatsapp_call to place the wake-up call (it runs the whole conversation and returns the outcome).
2. Read the outcome. If she was not confirmed awake, decide whether to call again:
   - she DECLINED the call (end_reason declined/busy/do_not_disturb) -> never call again
   - no answer (rang out) / call failed / audio or agent error / timeout -> call again if attempts remain
   - she hung up early without confirming (short call, no evidence of being up) -> call again
   - she clearly asked us to stop, or evidence shows she is up -> do NOT call again
3. Finish with wake_finish(wake_confirmed, summary). The summary is 1-2 sentences for the person who
   set this up. Never invent outcomes; only report what the tool results say.
Never call more times than attempts_left allows. Be brief; don't narrate."""


class WakeSupervisor:
    def __init__(self, tools: WakeTools, api_key: str, base_url: str, model: str):
        self.tools = tools
        self.ctx: RunContext = tools.ctx
        self.model = model
        self.llm = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=30, max_retries=2) if api_key else None

    async def run(self) -> RunContext:
        if self.llm is not None:
            try:
                await self._llm_loop()
            except Exception as e:
                event("SUPERVISOR_FALLBACK", run=self.ctx.run_id, reason=type(e).__name__)
        if not self.ctx.finished:
            await self._deterministic()
        return self.ctx

    # ------------------------------------------------------------ LLM loop
    async def _llm_loop(self) -> None:
        cfg = self.ctx.cfg
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM.format(objective=cfg.raw["objective"], contact=cfg.contact_name)},
            {"role": "user", "content": json.dumps({"event": "wake run started", **await self.tools.wake_get_context()})},
        ]
        tools = openai_tool_schema()
        idle_turns = 0
        for _ in range(MAX_STEPS):
            if self.ctx.finished:
                return
            resp = await self._complete(messages, tools)
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                **({"tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                                   for c in calls]} if calls else {}),
            })
            if not calls:
                idle_turns += 1
                if idle_turns >= 2:
                    raise RuntimeError("supervisor stopped calling tools")
                messages.append({"role": "user", "content": "Use a tool. If you are done, call wake_finish."})
                continue
            idle_turns = 0
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                event("SUPERVISOR_TOOL", run=self.ctx.run_id, tool=c.function.name)
                result = await self.tools.dispatch(c.function.name, args)
                messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result, default=str)})
                # the LLM should not be able to stall a live wake-up: if it hasn't dialled yet, we do
            if not self.ctx.attempts and not any(c.function.name == "whatsapp_call" for c in calls) and len(messages) > 8:
                raise RuntimeError("supervisor did not place the call")
        raise RuntimeError("supervisor exceeded step budget")

    async def _complete(self, messages, tools):
        last = None
        for attempt in range(3):
            try:
                return await asyncio.wait_for(
                    self.llm.chat.completions.create(
                        model=self.model, messages=messages, tools=tools, tool_choice="auto",
                        temperature=0.2, max_tokens=400,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    ),
                    40,
                )
            except Exception as e:  # NIM returns sporadic 500s; retry briefly
                last = e
                await asyncio.sleep(1.5 * (attempt + 1))
        raise last

    # ------------------------------------------------ deterministic fallback
    async def _deterministic(self) -> None:
        while True:
            ok, _ = self.tools.can_call()
            if not ok:
                break
            if self.ctx.attempts and not deterministic_should_retry(self.ctx.attempts[-1]):
                break
            await self.tools.whatsapp_call()
        confirmed = any(a.wake_confirmed for a in self.ctx.attempts)
        summaries = [a.summary for a in self.ctx.attempts if a.summary]
        summary = summaries[-1] if summaries else (
            "She was confirmed awake." if confirmed else
            f"Not confirmed awake after {len(self.ctx.attempts)} attempt(s); last: "
            f"{self.ctx.attempts[-1].state.value if self.ctx.attempts else 'no call placed'}.")
        await self.tools.wake_finish(wake_confirmed=confirmed, summary=summary)
