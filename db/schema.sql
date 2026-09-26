-- WakeAgent persistent state. Deliberately minimal: no raw audio, no transcript by default.
-- Lives at ~/.wake-agent/data/wake.db (outside the repo).

PRAGMA journal_mode = WAL;

-- One row per wake-up objective (a scheduled 08:00 run, or a manual call-now/test).
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Idempotency key. 'schedule:2026-09-23' for the 08:00 job; the UNIQUE constraint is what
    -- stops two processes (or a restart + misfire replay) from calling her twice.
    fire_key        TEXT    NOT NULL UNIQUE,
    trigger         TEXT    NOT NULL,              -- schedule | manual | test
    contact_name    TEXT    NOT NULL,
    contact_phone   TEXT    NOT NULL,
    wake_schedule   TEXT,                          -- "08:00 Asia/Kolkata"
    state           TEXT    NOT NULL,              -- see agents/state_machine.py
    attempts        INTEGER NOT NULL DEFAULT 0,
    wake_confirmed  INTEGER NOT NULL DEFAULT 0,
    summary         TEXT,                          -- short LLM summary of the whole run
    error           TEXT,
    created_at      TEXT    NOT NULL,
    finished_at     TEXT,
    heartbeat_at    TEXT                           -- stale heartbeat => run was orphaned by a crash
);

-- One row per WhatsApp call attempt.
CREATE TABLE IF NOT EXISTS calls (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                INTEGER NOT NULL REFERENCES runs(id),
    attempt               INTEGER NOT NULL,
    call_id               TEXT,                    -- WaCalls call id
    call_start            TEXT    NOT NULL,
    answered_at           TEXT,
    call_end              TEXT,
    call_status           TEXT    NOT NULL,        -- final state of the attempt
    end_reason            TEXT,                    -- WaCalls endReason (user_ended/declined/timeout/...)
    wake_confirmed        INTEGER NOT NULL DEFAULT 0,
    user_turns            INTEGER NOT NULL DEFAULT 0,
    agent_turns           INTEGER NOT NULL DEFAULT 0,
    conversation_summary  TEXT
);

-- Only written when privacy.store_transcript = true in wake.yaml.
CREATE TABLE IF NOT EXISTS transcripts (
    call_row_id  INTEGER NOT NULL REFERENCES calls(id),
    seq          INTEGER NOT NULL,
    role         TEXT    NOT NULL,
    text         TEXT    NOT NULL,
    PRIMARY KEY (call_row_id, seq)
);

CREATE INDEX IF NOT EXISTS calls_run ON calls(run_id);
