#!/bin/bash
# Installs two LaunchAgents so Svara.ai runs unattended (no terminal needed):
#   com.wakeagent.wacalls  - WaCalls WhatsApp server on 127.0.0.1:8787 (KeepAlive)
#   com.wakeagent.daemon   - scheduler + API, wrapped in caffeinate -i -s so the Mac never idle-sleeps
# Lid-closed sleep is a hardware policy caffeinate can't override: run  sudo pmset -a disablesleep 1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LA="$HOME/Library/LaunchAgents"; LOGS="$HOME/.wake-agent/logs"; mkdir -p "$LA" "$LOGS"
# launchd has a minimal PATH, so the plists need an absolute path to uv
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[ -x "$UV" ] || { echo "uv not found; install it from https://docs.astral.sh/uv/"; exit 1; }
write() { # label, program-args-xml
cat > "$LA/$1.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$1</string>
  <key>ProgramArguments</key><array>$2</array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin</string><key>PYTHONUNBUFFERED</key><string>1</string></dict>
  <key>StandardOutPath</key><string>$LOGS/$1.log</string>
  <key>StandardErrorPath</key><string>$LOGS/$1.log</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$1" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$1.plist"
echo "installed $1"
}
write com.wakeagent.wacalls "<string>$ROOT/scripts/run_wacalls.sh</string>"
write com.wakeagent.daemon "<string>/usr/bin/caffeinate</string><string>-i</string><string>-s</string><string>$UV</string><string>run</string><string>--directory</string><string>$ROOT</string><string>wake-agent</string><string>serve</string>"
