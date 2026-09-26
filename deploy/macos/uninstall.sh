#!/bin/bash
for l in com.wakeagent.daemon com.wakeagent.wacalls; do
  launchctl bootout "gui/$(id -u)/$l" 2>/dev/null; rm -f "$HOME/Library/LaunchAgents/$l.plist"; echo "removed $l"
done
