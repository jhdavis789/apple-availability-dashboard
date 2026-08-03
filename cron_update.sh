#!/bin/bash
# Cron job: pull Apple availability + eBay prices, rebuild dashboard, deploy
# Runs every 30 minutes via crontab

set -e

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
$PYTHON "$DASH_DIR/availability_matrix_csv_rest.py" || echo "WARNING: Apple pull failed"

# 2. Pull eBay pricing data (every 2 hours only — check minute=00 and even hour)
HOUR=$(date '+%H')
MINUTE=$(date '+%M')
if [ "$MINUTE" -lt 15 ] && [ $(( HOUR % 2 )) -eq 0 ]; then
  echo "[2/5] Pulling eBay pricing..."
  cd "$EBAY_DIR"
  $PYTHON "$EBAY_DIR/ebay_scraper.py" || echo "WARNING: eBay pull failed"
else
  echo "[2/5] Skipping eBay (runs every 2h at :00)"
fi

# 3. Rebuild data.json + store_map.json
echo "[3/5] Rebuilding data..."
cd "$DASH_DIR"
$PYTHON "$DASH_DIR/build_data.py" || { echo "ERROR: build_data.py failed"; exit 1; }

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

if ! git pull --rebase --quiet origin main; then
  echo "WARNING: rebase onto origin/main failed — aborting and leaving the tree clean"
  git rebase --abort 2>/dev/null || true
fi

if git push origin main; then
  echo "Deployed."
else
  echo "ERROR: git push failed — the live site is NOT updated"
  echo "  local vs origin: $(git rev-list --left-right --count origin/main...main 2>/dev/null || echo '?')"
  exit 1
fi

echo "=== Done $(date '+%Y-%m-%d %H:%M:%S') ==="
