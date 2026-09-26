"""SQLite persistence for runs and call attempts (stdlib sqlite3, one short connection per op)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DuplicateRun(Exception):
    """Another process already owns this fire_key (e.g. today's 08:00)."""


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA.read_text())

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 15000")
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ runs
    def create_run(self, *, fire_key: str, trigger: str, contact_name: str, contact_phone: str,
                   wake_schedule: str | None) -> int:
        try:
            with self._conn() as c:
                cur = c.execute(
                    "INSERT INTO runs (fire_key, trigger, contact_name, contact_phone, wake_schedule, state,"
                    " created_at, heartbeat_at) VALUES (?,?,?,?,?,?,?,?)",
                    (fire_key, trigger, contact_name, contact_phone, wake_schedule, "SCHEDULED", now_iso(), now_iso()),
                )
                return int(cur.lastrowid)
        except sqlite3.IntegrityError as e:
            raise DuplicateRun(fire_key) from e

    def update_run(self, run_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields["heartbeat_at"] = now_iso()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE runs SET {cols} WHERE id = ?", (*fields.values(), run_id))

    def get_run(self, run_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def recent_runs(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))]

    def orphaned_runs(self) -> list[dict]:
        """Runs that never reached a terminal state (process died mid-run)."""
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM runs WHERE finished_at IS NULL")]

    # ----------------------------------------------------------------- calls
    def create_call(self, run_id: int, attempt: int) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO calls (run_id, attempt, call_start, call_status) VALUES (?,?,?,?)",
                (run_id, attempt, now_iso(), "CALLING"),
            )
            return int(cur.lastrowid)

    def update_call(self, call_row_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE calls SET {cols} WHERE id = ?", (*fields.values(), call_row_id))

    def calls_for_run(self, run_id: int) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM calls WHERE run_id = ? ORDER BY attempt", (run_id,))]

    def save_transcript(self, call_row_id: int, turns: list[tuple[str, str]]) -> None:
        with self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO transcripts (call_row_id, seq, role, text) VALUES (?,?,?,?)",
                [(call_row_id, i, role, text) for i, (role, text) in enumerate(turns)],
            )
