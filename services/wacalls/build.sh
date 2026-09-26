#!/bin/bash
# Reproducible WaCalls build: pinned WaCallsNative commit + our pair-phone patch.
set -euo pipefail
PIN=026b889bef9e649600ced1992fb2228fe7954b77
HERE="$(cd "$(dirname "$0")" && pwd)"
WAKE_HOME="${WAKE_HOME:-$HOME/.wake-agent}"
SRC="$WAKE_HOME/src/wacalls"
mkdir -p "$WAKE_HOME/bin" "$WAKE_HOME/src"
if [ ! -d "$SRC/.git" ]; then git clone -q https://github.com/jobasfernandes/WaCallsNative "$SRC"; fi
cd "$SRC" && git fetch -q origin && git checkout -q -f "$PIN" && git clean -qfd -e internal/app/webui/dist
for p in "$HERE"/patches/*.patch; do git apply "$p"; done
(cd client && npm ci --silent && npm run build --silent)
rm -rf internal/app/webui/dist && cp -r client/dist internal/app/webui/dist
go test ./internal/app/... >/dev/null
CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=wake-agent-${PIN:0:7}+patches" -o "$WAKE_HOME/bin/wacalls" ./cmd/server
echo "built $WAKE_HOME/bin/wacalls"
