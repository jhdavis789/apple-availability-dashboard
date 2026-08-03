#!/bin/bash
# Pull Apple availability + eBay prices, rebuild dashboard, deploy.
#
# Scheduled by launchd (com.jhdavis.apple-availability.plist, :00 and :30), NOT
# crontab. cron silently drops every tick the laptop sleeps through and never
# makes them up, which is why collection uptime was 53.8% — 774 of 1,440
# expected ticks over 30 days. launchd's StartCalendarInterval replays one
# missed run the moment the Mac wakes. That replay lands *before* Wi-Fi has
# reassociated, hence the connectivity gate in step 0.
#
# The name is kept for the paths and log that reference it.

set -e

REAP_AFTER=1500   # seconds; a healthy run is ~4 min, so 25 min means wedged

# launchd runs this single-instance: a run hung on a stalled TLS connection
# blocks every later tick indefinitely. Kill any prior instance that has
# outlived a plausible run.
#
# Exclude by process GROUP, not by $$ — launchd wraps bash so the shell's $$
# differs from the PID pgrep reports for it, and excluding "self" by $$ fails,
# leaving the reaper poking its own run. Every `ps ... | tr` needs `|| true`:
# a pid can vanish mid-loop and `set -e` + a failing command substitution in an
# assignment aborts the whole script right after the header.
MYPGID="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ' || true)"
for pid in $(pgrep -f "cron_update.sh" 2>/dev/null || true); do
  [ "$pid" = "$$" ] && continue
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  [ -n "$MYPGID" ] && [ "$pgid" = "$MYPGID" ] && continue
  # macOS ps has no `etimes` keyword — asking for it dumps the keyword list and
  # the reaper silently never reaps. Parse `etime` ([[dd-]hh:]mm:ss) by hand;
  # 10# guards leading-zero octal.
  et="$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  [ -n "$et" ] || continue
  case "$et" in *[!0-9:-]*) continue;; esac
  days=0; case "$et" in *-*) days="${et%%-*}"; et="${et#*-}";; esac
  IFS=: read -r f1 f2 f3 <<< "$et" || true
  if [ -n "${f3:-}" ]; then esec=$(( (10#$days*24 + 10#$f1)*3600 + 10#$f2*60 + 10#$f3 ))
  else esec=$(( 10#$days*86400 + 10#$f1*60 + 10#$f2 )); fi
  [ -n "${esec:-}" ] || continue
  if [ "$esec" -gt "$REAP_AFTER" ] && [ -n "$pgid" ]; then
    kill -KILL -"$pgid" 2>/dev/null || true
  fi
done

# Bound any single network op. macOS ships no timeout(1); a self-exiting poller
# is used rather than `sleep N; kill` because a signal-killed sleep prints
# job-control noise into the log and can be orphaned into launchd's process
# group, keeping the job flagged as still running.
with_timeout() {              # with_timeout SECS cmd args...
  local secs="$1"; shift
  "$@" & local cmd_pid=$!
  ( local w=0
    while [ "$w" -lt "$secs" ]; do
      kill -0 "$cmd_pid" 2>/dev/null || exit 0
      sleep 1; w=$((w + 1))
    done
    kill -TERM "$cmd_pid" 2>/dev/null || true; sleep 3
    kill -KILL "$cmd_pid" 2>/dev/null || true ) &
  local rc=0
  wait "$cmd_pid" 2>/dev/null || rc=$?
  return "$rc"
}

# Export GH_TOKEN so gh credential helper works without keychain access.
# '|| true' so a transient read failure can't trip `set -e` before logging starts.
export GH_TOKEN="$(cat ~/.github_deploy_token 2>/dev/null || true)"

# Abort a transfer that stalls mid-flight rather than wedging the next tick.
export GIT_HTTP_LOW_SPEED_LIMIT=1000
export GIT_HTTP_LOW_SPEED_TIME=30
export GIT_TERMINAL_PROMPT=0

PYTHON=/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
BASE_DIR="/Users/Jackson/.openclaw/workspace/research/CG Side Projects/apple-availability"
DASH_DIR="$BASE_DIR/dashboard copy"
EBAY_DIR="$BASE_DIR/Ebay Scrape"
LOG="$DASH_DIR/cron_update.log"
RAW_DIR="$BASE_DIR/raw_api_responses_v3"
RAW_RETENTION_DAYS=45  # keep raw snapshots this long; downsampled data.json/store_map.json hold the history

# Rotate log if over 1MB
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG" 2>/dev/null || echo 0)" -gt 1048576 ]; then
  mv "$LOG" "$LOG.old"
fi

exec >> "$LOG" 2>&1
echo ""
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="

# 0. Don't spend a tick on a run that cannot succeed. cron fires on a laptop
# that is often asleep or mid-reassociation; an offline tick is not a failure,
# so skip with exit 0 rather than filling the log with tracebacks that hide the
# real ones.
online() { curl -sf --max-time 8 -o /dev/null https://www.google.com/generate_204 2>/dev/null; }
net_waited=0
until online; do
  if [ "$net_waited" -ge 120 ]; then
    echo "OFFLINE after 120s — skipping this tick"
    exit 0
  fi
  sleep 10
  net_waited=$((net_waited + 10))
done

# 1. Pull Apple availability data
echo "[1/5] Pulling Apple availability..."
cd "$BASE_DIR"
with_timeout 900 $PYTHON "$DASH_DIR/availability_matrix_csv_rest.py" || echo "WARNING: Apple pull failed"

# 2. Pull eBay pricing data (every 2 hours only — check minute=00 and even hour)
HOUR=$(date '+%H')
MINUTE=$(date '+%M')
if [ "$MINUTE" -lt 15 ] && [ $(( HOUR % 2 )) -eq 0 ]; then
  echo "[2/5] Pulling eBay pricing..."
  cd "$EBAY_DIR"
  with_timeout 600 $PYTHON "$EBAY_DIR/ebay_scraper.py" || echo "WARNING: eBay pull failed"
else
  echo "[2/5] Skipping eBay (runs every 2h at :00)"
fi

# 3. Rebuild data.json + store_map.json
echo "[3/5] Rebuilding data..."
cd "$DASH_DIR"
with_timeout 600 $PYTHON "$DASH_DIR/build_data.py" || { echo "ERROR: build_data.py failed"; exit 1; }

# 3b. Retention: prune raw snapshots older than the window + wave2 probe junk.
# Raw is only an intermediate; data.json/store_map.json carry the downsampled history.
echo "[4/5] Pruning raw snapshots older than ${RAW_RETENTION_DAYS}d..."
if [ -d "$RAW_DIR" ]; then
  before=$(find "$RAW_DIR" -name 'raw_responses_*.json.gz' | wc -l | tr -d ' ')
  find "$RAW_DIR" -name 'raw_responses_*.json.gz' -mtime +${RAW_RETENTION_DAYS} -delete 2>/dev/null || true
  find "$RAW_DIR" -name '*_wave2.json.gz' -delete 2>/dev/null || true
  after=$(find "$RAW_DIR" -name 'raw_responses_*.json.gz' | wc -l | tr -d ' ')
  echo "  raw files: $before -> $after ($(du -sh "$RAW_DIR" 2>/dev/null | cut -f1) on disk)"
fi

# 4. Deploy to GitHub Pages (push directly from source repo)
#
# The push used to be `git push || echo WARNING`. On 2026-07-02 a commit was
# made on GitHub directly (the Actions Pages workflow), so every subsequent push
# was rejected as non-fast-forward — and the warning was swallowed. The site sat
# frozen for 32 days while 986 local commits piled up and the log said
# "WARNING: git push failed" 1,500 times. Two fixes: rebase onto the remote
# first so a server-side commit self-heals, and make a failed deploy a non-zero
# exit so it is visibly a failure rather than a line in a log nobody reads.
echo "[5/5] Deploying to GitHub Pages..."
cd "$DASH_DIR"
cp dashboard.html index.html
git add data.json store_map.json index.html
git commit -m "Auto-update $(date '+%Y-%m-%d %H:%M')" || echo "No changes to commit"

# --autostash: any tracked file left modified (an edit in progress, a partial
# build) makes a plain `git pull --rebase` refuse outright, which silently
# disables the very self-healing this line exists to provide.
if ! with_timeout 120 git pull --rebase --autostash --quiet origin main; then
  echo "WARNING: rebase onto origin/main failed — aborting and leaving the tree clean"
  git rebase --abort 2>/dev/null || true
fi

if with_timeout 120 git push origin main; then
  echo "Deployed."
else
  echo "ERROR: git push failed — the live site is NOT updated"
  echo "  local vs origin: $(git rev-list --left-right --count origin/main...main 2>/dev/null || echo '?')"
  exit 1
fi

echo "=== Done $(date '+%Y-%m-%d %H:%M:%S') ==="
