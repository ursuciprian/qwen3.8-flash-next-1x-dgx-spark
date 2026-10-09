#!/usr/bin/env bash
# k74 queue (2026-10-08): starts the backlog runner for k74 (JOBS=k74 GRACE0=60) only after k73 is over: the k73 waiter
# (after_k56c.sh, pid $WP) is gone, the runner it started for k73 (pid from k73/queue.log) has exited, and k73 has a
# final STATE. A k73 waiter that exits without starting its runner, a k73 without a final STATE, or a STOPPED runner
# needs a person: then this exits without starting anything. Never touches the gpu-lock itself; the runner does.
# Start: WP=<k73 waiter pid> setsid nohup bash ~/GEN-AI/backlog/k74/after_k73.sh > ~/GEN-AI/backlog/k74/after_k73.nohup 2>&1 < /dev/null &
# Stop:  kill -TERM <pid in k74/queue.log> while it waits.
set -u
BL=$HOME/GEN-AI/backlog; WP=${WP:?k73 waiter pid}
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" >> "$BL/k74/queue.log"; }
final() { head -1 "$BL/$1/STATE" 2>/dev/null | grep -qE '^(DONE|FAILED|SKIPPED|STOPPED)'; }
waitpid() { while kill -0 "$1" 2>/dev/null; do [ $(( $(date +%s) - t0 )) -gt 345600 ] && { log "gave up after 96 h"; exit 1; }; sleep 60; done; }
log "k74 queue pid $$ waiting for the k73 waiter (pid $WP), then k73's runner"
echo "queued: after_k73.sh pid $$ waits for k73 (waiter $WP, then its runner), then runner.sh JOBS=k74 GRACE0=60" > "$BL/k74/STATE"
t0=$(date +%s)
waitpid "$WP"
RP=$(grep -oE 'runner pid [0-9]+ \(JOBS=k73' "$BL/k73/queue.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
[ -n "$RP" ] || { log "k73 waiter $WP gone without starting a runner (k73 STATE '$(head -1 "$BL/k73/STATE" 2>/dev/null)'): not starting k74"; exit 1; }
log "k73 runner pid $RP"; waitpid "$RP"
final k73 || { log "k73 runner $RP gone, k73 STATE '$(head -1 "$BL/k73/STATE" 2>/dev/null)': not starting k74"; exit 1; }
head -1 "$BL/STATE" | grep -q '^STOPPED' && { log "runner STATE '$(head -1 "$BL/STATE")': not starting k74"; exit 1; }
log "k73: $(head -1 "$BL/k73/STATE" | cut -c1-160); starting the runner"
JOBS=k74 GRACE0=60 setsid nohup bash "$BL/runner.sh" > "$BL/runner-k74.nohup" 2>&1 < /dev/null &
log "runner pid $! (JOBS=k74 GRACE0=60)"
