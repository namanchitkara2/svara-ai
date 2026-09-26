# Svara.ai: Research (Phase 1)

Date: 2026-09-22. Every claim below was checked against the source code at the commit named, not only the README.

## TL;DR

| Decision | Choice | Why |
|---|---|---|
| WhatsApp voice transport | **jobasfernandes/WaCallsNative @ `develop` (`026b889`, 2026-08-31)**, run as an unmodified sidecar | Maintained fork of WaCalls with auth, call timeouts, webhooks, persisted history, and the periodic relay re-registration that plausibly fixes the "~20 s hang-up" bug still open against upstream `main` |
| Audio access | Headless WebRTC peer (Python `aiortc`) acting as the "browser"; it opens a data channel labelled `pcm` | Raw 16 kHz mono s16le PCM both ways. No browser, no virtual audio device, no fork |
| Realtime voice | OpenAI Realtime (`gpt-realtime` family) over WebSocket, `audio/pcm` @ 24 kHz, `server_vad` + `interrupt_response` + `idle_timeout_ms` | Speech-to-speech, barge-in and turn detection built in. `idle_timeout_ms` fires when she goes quiet, which is exactly the "fell back asleep" case |
| Resampling | 16 k → 24 k (×3/2) in, 24 k → 16 k (×2/3) out, polyphase FIR | Clean rational ratio, cheap, stateful per stream |
| Scheduler | APScheduler 3.11 (stable) + SQLite job store + a UNIQUE `(schedule_id, fire_date)` run row for idempotency | Survives restarts, misfire grace, no double calls |
| High-level agent | Claude (Anthropic Messages API tool-use loop) with only the `whatsapp.*` / `wake.*` tools, plus an MCP server exposing the same tools to Claude Code | Claude supervises the attempt/retry/finish decisions and writes the summary. A deterministic fallback keeps the wake-up working if Claude is unreachable |

## 1. WaCalls (JotaDev66/WaCalls)

- `main` = `edeb31f` (v1.0.0, 2026-06-25). Go 1.26, whatsmeow, pion/webrtc v4, a pure-Go port of Meta's **MLow** codec (no cgo).
- **How a call is placed:** `POST /api/sessions/{sid}/calls {phone}` → `CallManager.StartCall` builds a `<call><offer>` stanza and sends it through whatsmeow (`internal/wa` VoipSocket). When the peer accepts, `events.CallAccept` arrives with `<relay>` endpoints and hop-by-hop keys. Then STUN binding/allocate runs on WhatsApp's relays, followed by ICE, DTLS and **SCTP DataChannel** to the relay (pion). SRTP media (`PT=120`, 16 kHz clock) flows inside that data channel.
- **How audio is exposed:** `cmd/server/bridge.go`. The *browser* POSTs an SDP **offer** to `/api/sessions/{sid}/calls/{id}/webrtc` (`{"sdp_offer"}`) and the server **answers** (`{"sdp_answer"}`). Audio does **not** use a media track. It uses a WebRTC **data channel labelled `pcm`**, carrying **raw 16 kHz mono Int16 little-endian PCM** in both directions (`media.PCMInt16LEToFloat32` / `PCMFloat32ToInt16LE`). Any other label is ignored.
- **Codec framing:** `mlowSampleRate = 16000`, `mlowFrameSize = 960` → **60 ms frames**. `FeedCapturedPCM` accumulates to 960 samples, encodes, and sends immediately. **It does not pace**, so the sender must feed at real-time rate. `startSilenceKeepaliveLocked` injects encoded silence when no capture has arrived for >120 ms, so the client does not need to stream silence.
- **Downlink:** `cm.OnPeerAudio(pcm16 []float32)` writes decoded frames to the bridge. **If no bridge is attached yet, the audio is dropped**, so the audio client must attach during RINGING.
- **Headless:** yes. The server is a single Go binary. The React client is optional.
- **Auth/session persistence:** QR pairing as a WhatsApp *linked device*. whatsmeow device keys are stored in `wacalls.db` (SQLite), and `SessionManager.Restore` reconnects on boot. **This file is a credential.**
- **Multiple calls:** yes (`-max-calls-per-session`, default 8, routed by call ID).
- **Restart:** sessions restore. Calls in progress are lost.
- **Security:** `main` has **no authentication** and `Access-Control-Allow-Origin: *`. Any web page open in a browser on the same machine could place calls through `localhost`. Not acceptable as-is.
- **Open issues that matter:**
  - **#38**: "WhatsApp terminates the call after ~20–22 s, every time" (v1.0.0). The reporter's hypothesis is missing periodic STUN consent freshness on the relay. **This would kill a 15-minute wake-up call.**
  - **#31**: whatsmeow pinned before the passkey pairing fix (whatsmeow PR #1186, 2026-07-01). New device pairing may fail.
- Upstream `develop` (`d16a076`, 2026-07-27) periodically syncs from the fork below.

## 2. WaCallsNative (jobasfernandes/WaCallsNative): the one we use

The fork by a WaCalls maintainer (commits flow back upstream through PRs #42 and #44). `develop` = `026b889` (2026-08-31), 463 commits.

Relevant deltas vs upstream `main`, all verified in source:

| Area | What changed | Source |
|---|---|---|
| Relay keep-alive | `52970aa` *"refresh relay registration periodically over keepalive"*: STUN binding and allocate re-sent every 5 keep-alive ticks (~5.5 s) instead of only at setup. This is the fix hypothesised in #38 | `internal/voip/transport/sctprelay.go` |
| Auth | Admin login required to boot (`WACALLS_ADMIN_USER`/`_PASSWORD`). Optional `WACALLS_API_TOKEN` bearer for automation. CORS same-origin by default. Per-IP rate limit | `internal/app/auth*.go`, README §Security |
| Timeouts | 60 s ring/answer, 30 s media connect, 4 h cap. Ends with reason `timeout` | `2e39cc1`, `e3da757` |
| Browser leg | 30 s grace window: if the PCM peer disconnects, the call is held for 30 s, then ended | `internal/app/session/grace.go` |
| Webhooks | HMAC-SHA256 signed `call.ringing` / `call.active` / `call.ended` (`X-Wacalls-Signature: v1=hex(HMAC(ts + "." + body))`) | README §Webhooks |
| History | Ended calls persisted (survive restart) | `bf5406f` |
| whatsmeow | `v0.0.0-20260713…`, which includes the passkey pairing fix (#31) | `go.mod` |
| Stale offers | Offers replayed after offline reconnect are dropped (`1b78a46`) | `offer_freshness.go` |
| Diagnostics | `WACALLS_DIAG_DIR` per-call JSONL (metadata only, no media or keys). `-doctor` self-check | README |
| Ops | `Dockerfile`, `docker-compose.yml`, `/healthz`, OpenAPI at `/api/openapi.yaml`. Images at `ghcr.io/jotadev66/wacalls:{latest,develop}` | repo root |

The data channel contract is unchanged: label `pcm`, 16 kHz mono s16le, client = SDP offerer (`internal/app/session/bridge.go`).

**Status of #38 on this fork: unverified.** The fix is plausible, but nobody has closed the issue. **Phase 2 must prove a call survives more than 3 minutes before anything else is trusted.**

## 3. WPPConnect WA-JS (wppconnect-team/wa-js)

- Injects into a real WhatsApp Web page and calls WhatsApp Web's *own* VoIP stack (`WPP.call.offer` → `startWAWebVoipCall`, `src/call/functions/offer.ts`).
- Audio therefore flows through the browser's `getUserMedia` and speakers. Getting PCM in and out means fake media devices, virtual audio drivers, or `--use-file-for-fake-audio-capture` hacks in headless Chrome. The brief explicitly asks us to avoid this.
- **Verdict:** a fallback only, if both WaCalls lines die.

## 4. whatsapp-mcp (ekaksher/whatsapp-mcp)

- A Python MCP server that drives WhatsApp Web through Playwright. Tools: auth status, list chats, **send/read text messages**. **No calling.**
- **Verdict:** not used for voice. The *pattern* (a small stdio MCP with narrow tools) is what we copy for `wake-agent-mcp`. Could later be used to send her a text if she doesn't answer. Not in scope.

## 5. WhatsApp Calls Research Group (WhiskeySockets/wacrg)

- A provenance-tracked **spec** of the 1:1 call protocol (signalling, crypto, relay, encodings), from the Baileys maintainers. Its own README says: "No real captures exist in this repository yet… most facts are `probable` or `speculative`."
- **Verdict:** reference material for debugging relay and termination issues (e.g. #38). No runnable code to integrate.

## 6. OpenAI Realtime (openai/openai-node, openai/openai-realtime-agents)

From `openai-node/src/resources/realtime/realtime.ts` (the SDK's typed schema):
- Models include `gpt-realtime`, `gpt-realtime-1.5`, `gpt-realtime-2`, `gpt-realtime-2.1`, `gpt-realtime-2.1-mini`, `gpt-realtime-mini`. **Configurable. The default is `gpt-realtime`.**
- Audio formats: `audio/pcm` (*"Only a 24kHz sample rate is supported"*), `audio/pcmu`, `audio/pcma` (G.711, 8 kHz). **We use `audio/pcm` @ 24 kHz** to avoid throwing away half the bandwidth.
- Turn detection: `server_vad` (`threshold`, `silence_duration_ms`, `prefix_padding_ms`, `create_response`, `interrupt_response`, **`idle_timeout_ms`**) or `semantic_vad` (`eagerness`). `idle_timeout_ms` (server_vad only) auto-triggers a response and emits `input_audio_buffer.timeout_triggered`. That is our "she went silent" nudge.
- Barge-in: `input_audio_buffer.speech_started` → we flush the unsent outbound PCM queue immediately, then send `conversation.item.truncate` with the audio actually played, so the model's context matches what she heard.
- Function tools are supported inside the realtime session. The voice model reports wake evidence (`report_wake_evidence`) and requests `end_call`. Transcription of her speech (`input_audio_transcription`) is used only in memory, for the summary.
- `openai-realtime-agents` is a Next.js browser demo (agent handoffs over WebRTC in the browser). Useful for prompt patterns only. Wrong runtime for a headless phone bridge.

## 7. Alternatives considered for the realtime layer

| Option | Audio I/O | Notes |
|---|---|---|
| **OpenAI Realtime** (chosen) | 24 k in/out | Mature server VAD, idle timeout, truncate semantics, function calling |
| Gemini Live | **16 k in** / 24 k out | Input matches WhatsApp natively (one resample instead of two). Barge-in supported. A good second provider. `RealtimeVoiceProvider` is kept genuinely swappable for this |
| STT → LLM → TTS (e.g. Deepgram + Claude + Cartesia) | any | More moving parts, higher latency (~1–2 s), barge-in must be hand-built. Only worth it if a speech-to-speech provider is unavailable |

## 8. Silero VAD (Sahl-AI/silero-vad)

- A non-fork copy of Silero VAD (last push 2024-10). Silero is the standard local VAD (ONNX, 8/16 kHz, 30–100 ms windows).
- **Verdict:** not needed on the main path, because the realtime provider's `server_vad` does turn detection and barge-in. We use a cheap local RMS energy meter only for observability (`USER_SPEECH_DETECTED`) and the "AI doesn't hear anything" watchdog. Silero is documented as the upgrade path if a provider without server VAD is swapped in.

## 9. APScheduler (agronholm/apscheduler)

- Latest stable is **3.11.3**. 4.x is still pre-release, so we use 3.x.
- `BackgroundScheduler`/`AsyncIOScheduler` + `SQLAlchemyJobStore` persists jobs. `CronTrigger(hour, minute, timezone="Asia/Kolkata")`, `misfire_grace_time` and `coalesce=True` handle "the machine was asleep at 08:00".
- APScheduler alone does not stop two processes firing the same job. We add a DB row with `UNIQUE(schedule_id, fire_date)` inserted before dialing. The second inserter loses and exits.

## 10. Newer or better implementations searched

- The WaCalls line (WaCalls → WaCallsNative) is the only open-source project found with **server-side, headless, bidirectional PCM** for WhatsApp 1:1 calls. Upstream acknowledges `whatsapp-rust` (MLow reference) and `zapo` (VoIP media reference) as the lineage. Neither exposes a ready call server.
- WhatsApp Business **Cloud API Calling** is the official alternative. It needs a Business account and number, templates and opt-in, and the recipient sees a business profile. It is a legitimate fallback if the unofficial route gets banned.

## 11. Risks (to be re-checked in Phase 2 and Phase 7)

1. **Call longevity (#38).** The deciding test.
2. **Ban risk.** whatsmeow is unofficial. Whatever number is paired as the linked device carries it. If it's his personal number, she sees a call from him, which is the only realistic way she answers at 8 am. If it's a spare number, the ban risk is isolated but she may ignore an unknown caller.
3. **The host must be awake at 08:00.** A lidded MacBook sleeps, so no call. This needs `pmset` wake plus `caffeinate` under launchd, or a VPS.
4. **Consent and honesty.** The bot calls from his number with a synthetic voice. It must never claim to be him or a human, and she should know in advance that a bot will call.
