# Troubleshooting

Every entry below happened for real during development (2026-09-22).

| Symptom | Cause | Fix |
|---|---|---|
| `POST …/calls -> 500: usync devices … websocket not connected` right after linking | After phone-code pairing the whatsmeow socket doesn't reconnect | `launchctl kickstart -k gui/$(id -u)/com.wakeagent.wacalls` (restores the session from `wacalls.db`) |
| "The call didn't go", but WaCalls history shows it answered | Calls placed by a **linked device** don't appear in the phone's WhatsApp Recents | Check `GET /api/sessions/<sid>/history` (it shows the contact name) |
| Audio leg never opens; aiortc ICE stuck in `checking` | macOS Application Firewall drops UDP between two unsigned local processes on en0/IPv6 candidates | Already handled: ICE is restricted to `127.0.0.1` (`restrict_ice_hosts`) |
| Opening line cut off after 0.1–0.5 s | Callee says "hello?" while answering, which triggered barge-in | Already handled: the first agent turn is not barge-able |
| AI talks to itself / "are you there?" while the other person *is* talking (Hindi) | Multilingual ASR never finalises on a noisy line; fast-endpoint params split words | Already handled: default endpointing for the multilingual model, plus a 1.2 s stalled-partial fallback |
| AI keeps cancelling its own replies | Late ASR finals for her *previous* sentence counted as barge-in | Already handled: only speech that starts after the agent's turn can barge in |
| Goodbye never heard, call just drops | `end_call` handler awaited inside the audio event pump (deadlock) | Already handled: goodbye runs as its own task |
| Retry loop kept calling after she declined | WaCalls reports a decline while ringing as `user_ended` | Already handled: normalised to `declined`, never retried |
| AI states facts nobody gave it | LLM filling gaps | Already handled: strict "facts" section in prompts; verify with `scripts/simulate_call.py party_invite_hindi_noisy` |
| `LLM attempt failed: … APIError / 500` | NVIDIA NIM transient errors | Automatic: hedged second request, retry, canned fallback line |
| Scheduled call didn't fire | Mac asleep / lid closed / on battery | Plug in; `sudo pmset -a disablesleep 1`; check `launchctl list \| grep wakeagent` |
| Schedule time became a number | YAML reads unquoted `10:30` as 630 | Already handled: normalised on load, quoted on save |

**Logs**
- `~/.wake-agent/logs/wake.jsonl`: structured events.
- `~/.wake-agent/logs/com.wakeagent.*.log`: service output.
- `WACALLS_DIAG_DIR=…` enables WaCalls per-call JSONL.

**Health**
- `curl 127.0.0.1:8787/healthz`
- `curl 127.0.0.1:8790/healthz`
- `uv run svara pair`: shows the session state.
