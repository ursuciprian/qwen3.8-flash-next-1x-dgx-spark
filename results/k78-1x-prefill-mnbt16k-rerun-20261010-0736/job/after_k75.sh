#!/usr/bin/env bash
# k78 queue (2026-10-09, k78-1x-prefill-mnbt16k-rerun, last in the chain): starts the backlog runner for k78
# (JOBS=k78 GRACE0=60) after k75, modeled on k75/after_k76.sh: wait for the k75 waiter (pid $WP), then the runner it
# started (pid from k75/queue.log), then require a final k75 STATE (DONE/FAILED/SKIPPED) and a runner STATE that is not
# STOPPED. If the chain dies (the k75 waiter exits without starting a runner, or its runner ends without a final k75
# STATE): exit without starting. Never touches the gpu-lock itself; the runner does.
# Start: WP=<k75 waiter pid> setsid nohup bash ~/GEN-AI/backlog/k78/after_k75.sh > ~/GEN-AI/backlog/k78/after_k75.nohup 2>&1 < /dev/null &
# Stop:  kill -TERM <pid in k78/queue.log> while it waits.
set -u
umask 022   # STATE and the runner it starts (runner files for k78) come out 0644
BL=$HOME/GEN-AI/backlog; WP=${WP:?k75 waiter pid (k75/after_k76.sh)}
log() { echo "[$(TZ=Europe/Bucharest date '+%F %T %Z')] $*" >> "$BL/k78/queue.log"; }
final() { head -1 "$BL/$1/STATE" 2>/dev/null | grep -qE '^(DONE|FAILED|SKIPPED)'; }
waitpid() { while kill -0 "$1" 2>/dev/null; do [ $(( $(date +%s) - t0 )) -gt 345600 ] && { log "gave up after 96 h"; exit 1; }; sleep 60; done; }
log "k78 queue pid $$ waiting for the k75 waiter (pid $WP), then k75's runner"
echo "queued: after_k75.sh pid $$ waits for k75 (waiter $WP, then its runner), then runner.sh JOBS=k78 GRACE0=60" > "$BL/k78/STATE"
t0=$(date +%s)
waitpid "$WP"
RP=$(grep -oE 'runner pid [0-9]+ \(JOBS=k75' "$BL/k75/queue.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
[ -n "$RP" ] || { log "k75 waiter $WP gone without starting a runner (k75 STATE '$(head -1 "$BL/k75/STATE" 2>/dev/null)'): not starting k78"; exit 1; }
log "k75 runner pid $RP"; waitpid "$RP"
final k75 || { log "k75 runner $RP gone, k75 STATE '$(head -1 "$BL/k75/STATE" 2>/dev/null)': not starting k78"; exit 1; }
log "k75: $(head -1 "$BL/k75/STATE" | cut -c1-160)"
head -1 "$BL/STATE" | grep -q '^STOPPED' && { log "runner STATE '$(head -1 "$BL/STATE")': not starting k78"; exit 1; }
log "starting the runner"
JOBS=k78 GRACE0=60 setsid nohup bash "$BL/runner.sh" > "$BL/runner-k78.nohup" 2>&1 < /dev/null &
log "runner pid $! (JOBS=k78 GRACE0=60)"
