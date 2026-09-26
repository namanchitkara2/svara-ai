"""`wake-agent serve`: the unattended runtime (scheduler + local API + config watcher).

Runs under launchd (deploy/macos) or Docker (docker-compose.yml). Claude Code is not needed.
"""

from __future__ import annotations

import asyncio
import os
import signal

import uvicorn

from apps.api import build_app
from apps.runtime import mark_orphans
from services.config import load_config, load_settings
from services.observability import event, setup
from services.scheduler import WakeScheduler
from services.store import Store


async def serve() -> None:
    settings = load_settings()
    cfg = load_config(settings.config_path)
    setup(settings.log_dir, cfg.schedule["timezone"])
    store = Store(settings.db_path)
    mark_orphans(store)

    sched = WakeScheduler(f"sqlite:///{settings.home / 'data' / 'scheduler.db'}")
    sched.start()
    sched.apply(cfg)
    event("DAEMON_STARTED", api=f"{settings.api_host}:{settings.api_port}", pid=os.getpid())

    app = build_app(settings, sched)
    server = uvicorn.Server(uvicorn.Config(app, host=settings.api_host, port=settings.api_port,
                                           log_level="warning", access_log=False))

    async def watch_config() -> None:
        last = settings.config_path.stat().st_mtime
        while True:
            await asyncio.sleep(15)
            try:
                m = settings.config_path.stat().st_mtime
                if m != last:
                    last = m
                    sched.apply(load_config(settings.config_path))
                    event("CONFIG_RELOADED")
            except Exception as e:
                event("CONFIG_RELOAD_FAILED", err=str(e)[:120])

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    tasks = [asyncio.create_task(server.serve()), asyncio.create_task(watch_config())]
    await stop.wait()
    event("DAEMON_STOPPING")
    server.should_exit = True
    sched.shutdown()
    for t in tasks:
        t.cancel()
