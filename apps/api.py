"""Local control API + tiny config UI. Bound to 127.0.0.1, every /api route needs WAKE_API_TOKEN."""

from __future__ import annotations

import asyncio
import hmac
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from apps.runtime import run_wake
from services.config import load_config, mask_phone, save_config
from services.store import Store


class OneOff(BaseModel):
    in_minutes: float = 3
    phone: str | None = None
    test: bool = False
    name: str | None = None
    opening: str | None = None
    objective: str | None = None
    live: bool = True          # native speech-to-speech by default


class CallNow(BaseModel):
    phone: str | None = None
    test: bool = False


def build_app(settings, scheduler) -> FastAPI:
    app = FastAPI(title="WakeAgent", docs_url=None, redoc_url=None, openapi_url=None)
    store = Store(settings.db_path)
    running: set[asyncio.Task] = set()

    def auth(authorization: str = Header(default="")) -> None:
        tok = settings.api_token
        if not tok:
            raise HTTPException(503, "WAKE_API_TOKEN not configured")
        if not hmac.compare_digest(authorization.removeprefix("Bearer ").strip(), tok):
            raise HTTPException(401, "unauthorized")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/api/status", dependencies=[Depends(auth)])
    def status():
        cfg = load_config(settings.config_path)
        nxt = scheduler.next_run()
        return {
            "contact": {"name": cfg.contact_name, "phone": mask_phone(cfg.contact_phone)},
            "schedule": cfg.schedule,
            "next_run": nxt.isoformat() if nxt else None,
            "jobs": scheduler.jobs(),
            "recent_runs": [{k: r[k] for k in ("id", "trigger", "state", "attempts", "wake_confirmed", "summary",
                                               "created_at", "finished_at")} for r in store.recent_runs(10)],
        }

    @app.get("/api/config", dependencies=[Depends(auth)])
    def get_config():
        return load_config(settings.config_path).raw

    @app.put("/api/config", dependencies=[Depends(auth)])
    def put_config(raw: dict):
        try:
            cfg = save_config(raw, settings.config_path)
        except Exception as e:
            raise HTTPException(400, str(e))
        nxt = scheduler.apply(cfg)
        return {"ok": True, "next_run": nxt.isoformat() if nxt else None}

    @app.post("/api/call-now", dependencies=[Depends(auth)])
    async def call_now(body: CallNow):
        t = asyncio.create_task(run_wake(trigger="test" if body.test else "manual", phone=body.phone,
                                         test_mode=body.test))
        running.add(t)
        t.add_done_callback(running.discard)
        return {"started": True}

    @app.post("/api/schedule/oneoff", dependencies=[Depends(auth)])
    def oneoff(body: OneOff):
        cfg = load_config(settings.config_path)
        when = datetime.now(ZoneInfo(cfg.schedule["timezone"])) + timedelta(minutes=body.in_minutes)
        job = scheduler.add_one_off(when, phone=body.phone, test_mode=body.test, name=body.name,
                                    opening=body.opening, objective=body.objective, live=body.live)
        return {"job": job, "at": when.isoformat()}

    @app.get("/api/runs/{run_id}", dependencies=[Depends(auth)])
    def run_detail(run_id: int):
        r = store.get_run(run_id)
        if not r:
            raise HTTPException(404)
        return {"run": r, "calls": store.calls_for_run(run_id)}

    @app.get("/", response_class=HTMLResponse)
    def ui():
        return _UI

    return app


_UI = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WakeAgent</title><style>
:root{--bg:#fbfaf7;--fg:#1d1c1a;--mut:#6b675f;--card:#fff;--line:#e6e2da;--acc:#b4532a}
@media (prefers-color-scheme:dark){:root{--bg:#161513;--fg:#ece9e3;--mut:#9c978d;--card:#1f1e1b;--line:#33312c;--acc:#e0895c}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:760px;margin:auto}
h1{font-size:20px;margin:0 0 4px}p{color:var(--mut);margin:0 0 16px}section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
textarea{width:100%;min-height:360px;font:13px ui-monospace,monospace;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px;box-sizing:border-box}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:8px 14px;font:inherit;cursor:pointer;margin-right:8px}
input{font:inherit;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);width:100%;box-sizing:border-box}
pre{white-space:pre-wrap;font:12px ui-monospace,monospace;color:var(--mut)}</style></head><body>
<h1>WakeAgent</h1><p>Local control panel. Token is kept only in this tab.</p>
<section><input id="tok" type="password" placeholder="WAKE_API_TOKEN"><br><br><button onclick="load()">Load</button></section>
<section><b>Status</b><pre id="st">not loaded</pre></section>
<section><b>Configuration (wake.yaml as JSON)</b><textarea id="cfg"></textarea><br><br>
<button onclick="save()">Save &amp; reschedule</button><button onclick="post('/api/schedule/oneoff',{in_minutes:3,test:true})">Test in 3 min</button></section>
<script>
const H=()=>({'Authorization':'Bearer '+document.getElementById('tok').value,'Content-Type':'application/json'});
async function load(){const s=await fetch('/api/status',{headers:H()});document.getElementById('st').textContent=JSON.stringify(await s.json(),null,2);
const c=await fetch('/api/config',{headers:H()});document.getElementById('cfg').value=JSON.stringify(await c.json(),null,2)}
async function save(){let b;try{b=JSON.parse(document.getElementById('cfg').value)}catch(e){alert('Invalid JSON');return}
const r=await fetch('/api/config',{method:'PUT',headers:H(),body:JSON.stringify(b)});alert(JSON.stringify(await r.json()));load()}
async function post(p,b){const r=await fetch(p,{method:'POST',headers:H(),body:JSON.stringify(b)});alert(JSON.stringify(await r.json()))}
</script></body></html>"""
