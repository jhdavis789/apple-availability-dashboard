#!/bin/bash
# Cron job: pull Apple availability + eBay prices, rebuild dashboard, deploy
# Runs every 30 minutes via crontab

set -e

# Export GH_TOKEN so gh credential helper works without keychain access.
# '|| true' so a transient read failure can't trip `set -e` before logging starts.
export GH_TOKEN="$(cat ~/.github_deploy_token 2>/dev/null || true)"

PYTHON=/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
BASE_DIR="/Users/Jackson/.openclaw/workspace/research/CG Side Projects/apple-availability"
DASH_DIR="$BASE_DIR/dashboard copy"
EBAY_DIR="$BASE_DIR/Ebay Scrape"
LOG="$DASH_DIR/cron_update.log"
RAW_DIR="$BASE_DIR/raw_api_responses"
RAW_RETENTION_DAYS=45  # keep raw snapshots this long; downsampled data.json/store_map.json hold the history

# Rotate log if over 1MB
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG" 2>/dev/null || echo 0)" -gt 1048576 ]; then
  mv "$LOG" "$LOG.old"
fi

exec >> "$LOG" 2>&1
echo ""
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="

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
  before=$(find "$RAW_DIR" -name 'raw_responses_*.json' | wc -l | tr -d ' ')
  find "$RAW_DIR" -name 'raw_responses_*.json' -mtime +${RAW_RETENTION_DAYS} -delete 2>/dev/null || true
  find "$RAW_DIR" -name '*_wave2.json' -delete 2>/dev/null || true
  after=$(find "$RAW_DIR" -name 'raw_responses_*.json' | wc -l | tr -d ' ')
  echo "  raw files: $before -> $after ($(du -sh "$RAW_DIR" 2>/dev/null | cut -f1) on disk)"
fi

# 4. Deploy to GitHub Pages (push directly from source repo)
echo "[5/5] Deploying to GitHub Pages..."
cd "$DASH_DIR"
cp dashboard.html index.html
git add data.json store_map.json index.html
git commit -m "Auto-update $(date '+%Y-%m-%d %H:%M')" || echo "No changes to commit"
git push origin main || echo "WARNING: git push failed"

echo "=== Done $(date '+%Y-%m-%d %H:%M:%S') ==="
