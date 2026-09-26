"""Wake run state machine.

    SCHEDULED → CALLING → RINGING → ANSWERED → GREETING → CONVERSING
        → WAKE_VERIFICATION → AWAKE_CONFIRMED → GOODBYE → COMPLETED

Failure states: NO_ANSWER, CALL_FAILED, AUDIO_FAILED, AGENT_ERROR, TIMEOUT.
A failed attempt may go back to CALLING (retry) until retry.max_attempts is spent.
Every transition is validated and logged as a structured event.
"""

from __future__ import annotations

from enum import Enum

from services.observability import event


class S(str, Enum):
    SCHEDULED = "SCHEDULED"
    CALLING = "CALLING"
    RINGING = "RINGING"
    ANSWERED = "ANSWERED"
    GREETING = "GREETING"
    CONVERSING = "CONVERSING"
    WAKE_VERIFICATION = "WAKE_VERIFICATION"
    AWAKE_CONFIRMED = "AWAKE_CONFIRMED"
    GOODBYE = "GOODBYE"
    COMPLETED = "COMPLETED"
    # failures
    NO_ANSWER = "NO_ANSWER"
    CALL_FAILED = "CALL_FAILED"
    AUDIO_FAILED = "AUDIO_FAILED"
    AGENT_ERROR = "AGENT_ERROR"
    TIMEOUT = "TIMEOUT"


FAILURES = {S.NO_ANSWER, S.CALL_FAILED, S.AUDIO_FAILED, S.AGENT_ERROR, S.TIMEOUT}
IN_CALL = {S.ANSWERED, S.GREETING, S.CONVERSING, S.WAKE_VERIFICATION, S.AWAKE_CONFIRMED, S.GOODBYE}

_ALLOWED: dict[S, set[S]] = {
    S.SCHEDULED: {S.CALLING, S.AGENT_ERROR, S.CALL_FAILED},
    S.CALLING: {S.RINGING, S.CALL_FAILED, S.AGENT_ERROR},
    S.RINGING: {S.ANSWERED, S.NO_ANSWER, S.CALL_FAILED, S.AGENT_ERROR},
    S.ANSWERED: {S.GREETING, S.AUDIO_FAILED, S.CALL_FAILED, S.AGENT_ERROR, S.COMPLETED},
    S.GREETING: {S.CONVERSING, S.AUDIO_FAILED, S.AGENT_ERROR, S.TIMEOUT, S.COMPLETED, S.CALL_FAILED},
    S.CONVERSING: {S.WAKE_VERIFICATION, S.AWAKE_CONFIRMED, S.GOODBYE, S.AUDIO_FAILED, S.AGENT_ERROR, S.TIMEOUT,
                   S.COMPLETED, S.CALL_FAILED},
    S.WAKE_VERIFICATION: {S.CONVERSING, S.AWAKE_CONFIRMED, S.GOODBYE, S.AUDIO_FAILED, S.AGENT_ERROR, S.TIMEOUT,
                          S.COMPLETED, S.CALL_FAILED},
    S.AWAKE_CONFIRMED: {S.GOODBYE, S.COMPLETED},
    S.GOODBYE: {S.COMPLETED, S.CONVERSING},   # CONVERSING: she spoke during our goodbye, stay on
    S.COMPLETED: set(),
}
# any failure can be retried (→ CALLING) or be final
for _f in FAILURES:
    _ALLOWED[_f] = {S.CALLING}


class IllegalTransition(Exception):
    pass


class WakeStateMachine:
    def __init__(self, run_id: int, on_change=None, initial: S = S.SCHEDULED):
        self.run_id = run_id
        self.state = initial
        self.history: list[S] = [initial]
        self._on_change = on_change

    def to(self, new: S, **fields) -> None:
        if new == self.state:
            return
        if new not in _ALLOWED[self.state]:
            raise IllegalTransition(f"{self.state.value} -> {new.value}")
        old, self.state = self.state, new
        self.history.append(new)
        event("STATE", run=self.run_id, frm=old.value, to=new.value, **fields)
        if self._on_change:
            self._on_change(new)

    @property
    def terminal(self) -> bool:
        return self.state == S.COMPLETED or self.state in FAILURES

    @property
    def failed(self) -> bool:
        return self.state in FAILURES
