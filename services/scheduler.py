"""Persistent 08:00 scheduler.

- APScheduler 3.11 AsyncIOScheduler, jobs persisted in ~/.wake-agent/data/scheduler.db (SQLAlchemy),
  so a restart keeps the schedule and knows when the job last/next fires.
- CronTrigger in the configured timezone (Asia/Kolkata), independent of the laptop's own timezone.
- misfire_grace_time: if the process/machine was down or asleep at 08:00 and comes back within the
  grace window, the job still fires once (coalesce=True collapses a backlog into one run).
- Idempotency: the job's run is keyed 'schedule:YYYY-MM-DD' with a UNIQUE constraint in wake.db,
  so a second process, a double fire, or a misfire replay can never call her twice on one day.
"""

from __future__ import annotations

import logging
from datetime import datetime

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from services.config import WakeConfig
from services.observability import event

log = logging.getLogger("wake.scheduler")

JOB_ID = "daily-wake"
MISFIRE_GRACE_S = 20 * 60


async def _fire() -> None:
    # imported lazily: the job is serialised by reference ("services.scheduler:_fire")
    from apps.runtime import run_wake, scheduled_fire_key
    from services.config import load_config

    cfg = load_config()
    event("WAKE_SCHEDULE_TRIGGERED", schedule=f"{cfg.schedule['time']} {cfg.schedule['timezone']}")
    await run_wake(trigger="schedule", fire_key=scheduled_fire_key(cfg))


class WakeScheduler:
    def __init__(self, db_url: str):
        self.sched = AsyncIOScheduler(
            jobstores={"default": SQLAlchemyJobStore(url=db_url)},
            job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": MISFIRE_GRACE_S},
        )

    def start(self) -> None:
        self.sched.start()

    def shutdown(self) -> None:
        self.sched.shutdown(wait=False)

    def apply(self, cfg: WakeConfig) -> datetime | None:
        """(Re)install the daily job from config. Returns the next fire time, or None if disabled."""
        s = cfg.schedule
        if not s.get("enabled", True):
            if self.sched.get_job(JOB_ID):
                self.sched.remove_job(JOB_ID)
            event("SCHEDULE_DISABLED")
            return None
        hh, mm = (int(x) for x in str(s["time"]).split(":"))
        days = ",".join(d[:3].lower() for d in (s.get("days") or [])) or None
        trigger = CronTrigger(hour=hh, minute=mm, day_of_week=days, timezone=s["timezone"])
        job = self.sched.add_job(_fire, trigger, id=JOB_ID, replace_existing=True, name="daily wake-up call")
        nxt = job.next_run_time
        event("SCHEDULE_SET", time=s["time"], tz=s["timezone"], days=days or "daily",
              next=nxt.isoformat() if nxt else None)
        return nxt

    def add_one_off(self, when: datetime, phone: str | None = None, test_mode: bool = False, **kw) -> str:
        """A one-off run (a near-future test, or 'wake me in 4 hours')."""
        job = self.sched.add_job(
            "apps.runtime:run_wake_oneoff", "date", run_date=when,
            kwargs={"phone": phone, "test_mode": test_mode, "fire_key": f"oneoff:{when.isoformat()}", **kw},
            misfire_grace_time=MISFIRE_GRACE_S,
        )
        event("ONE_OFF_SCHEDULED", at=when.isoformat(), job=job.id)
        return job.id

    def next_run(self) -> datetime | None:
        j = self.sched.get_job(JOB_ID)
        return j.next_run_time if j else None

    def jobs(self) -> list[dict]:
        return [{"id": j.id, "name": j.name, "next": j.next_run_time.isoformat() if j.next_run_time else None}
                for j in self.sched.get_jobs()]
