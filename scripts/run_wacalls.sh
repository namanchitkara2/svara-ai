#!/bin/bash
# Runs the pinned WaCallsNative server on loopback only, with secrets from wake-agent/.env.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WAKE_HOME="${WAKE_HOME:-$HOME/.wake-agent}"
set -a; source "$ROOT/.env"; set +a
export WACALLS_ADMIN_USER WACALLS_ADMIN_PASSWORD WACALLS_API_TOKEN
exec "$WAKE_HOME/bin/wacalls" -addr 127.0.0.1:8787 -db "$WAKE_HOME/data/wacalls.db" -static ""
