# Setup

Tested on macOS 26 (Apple Silicon), Python 3.12 via `uv`, Go 1.27, Node 22.

## 1. Install

```bash
brew install go node uv
git clone https://github.com/namanchitkara2/svara-ai ~/svara-ai && cd ~/svara-ai
uv sync
./services/wacalls/build.sh                # pinned WaCallsNative + pair-phone patch -> ~/.wake-agent/bin/wacalls
cp .env.example .env && chmod 600 .env     # your own keys: NVIDIA_API_KEY and/or GEMINI_API_KEY, plus the three tokens
cp config/wake.example.yaml config/wake.yaml   # real contact, time, persona (gitignored)
```

The CLI is `svara`; `wake-agent` remains a working alias for it. Runtime state (paired WhatsApp
credentials, databases, logs) lives outside the repo in `~/.wake-agent`, overridable with `WAKE_HOME`.
No API key is ever read from anywhere but your local `.env`.

## 2. Run as background services (macOS)

```bash
./deploy/macos/install.sh                  # LaunchAgents: wacalls + daemon (caffeinate-wrapped)
sudo pmset -a disablesleep 1               # only if the lid will be closed overnight
```

The Mac must be **plugged in** (`caffeinate -s` only holds on AC power).

## 3. Link WhatsApp (once)

```bash
uv run svara pair                     # QR in the browser, auto-refreshing
```

Or by phone number, with no scan needed:

```bash
uv run svara pair --phone "+91XXXXXXXXXX"   # prints an 8-character code; enter it on the phone:
                                                 # Linked devices > Link with phone number instead

```

## 4. Try it without calling anyone

```bash
uv run pytest
uv run python scripts/simulate_call.py all
```

## 5. Real calls

```bash
uv run svara test --contact "+91XXXXXXXXXX" --show-transcript        # short test call
uv run svara call-now --contact "+91XXXXXXXXXX"                      # full wake-up with retries
uv run svara schedule --time 08:00 --timezone Asia/Kolkata --contact "+91XXXXXXXXXX"
uv run svara schedule --in-minutes 3 --test                          # near-future one-off (daemon)
uv run svara schedule --disable
uv run svara status
```

One-off calls with a custom purpose (not a wake-up):

```bash
uv run svara test --allow-other --contact "+91..." --name "Nikhil" --language hi \
  --voice Magpie-Multilingual.HI-IN.Sofia --max-minutes 5 --show-transcript \
  --opening "नमस्ते…" --objective "What the call is for, and the only facts it may state."
```

## 6. Claude Code (MCP)

```bash
claude mcp add svara -- uv --directory ~/svara-ai run svara-mcp
```

## 7. Docker / VPS

```bash
docker compose up -d --build
docker compose exec wake-agent uv run svara pair
```
