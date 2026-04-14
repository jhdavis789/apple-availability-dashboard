#!/bin/bash
# Cron job: pull Apple availability + eBay prices, rebuild dashboard, deploy
# Runs every 30 minutes via crontab

set -e

# Export GH_TOKEN so gh credential helper works without keychain access
export GH_TOKEN=$(cat ~/.github_deploy_token 2>/dev/null)

PYTHON=/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
BASE_DIR="/Users/Jackson/.openclaw/workspace/research/apple-availability"
DASH_DIR="$BASE_DIR/dashboard copy"
EBAY_DIR="$BASE_DIR/Ebay Scrape"
LOG="$DASH_DIR/cron_update.log"

# Rotate log if over 1MB
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG" 2>/dev/null || echo 0)" -gt 1048576 ]; then
  mv "$LOG" "$LOG.old"
fi

exec >> "$LOG" 2>&1
echo ""
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="

# 1. Pull Apple availability data
echo "[1/4] Pulling Apple availability..."
cd "$BASE_DIR"
$PYTHON "$DASH_DIR/availability_matrix_csv_rest.py" || echo "WARNING: Apple pull failed"

# 2. Pull eBay pricing data (every 2 hours only — check minute=00 and even hour)
HOUR=$(date '+%H')
MINUTE=$(date '+%M')
if [ "$MINUTE" -lt 15 ] && [ $(( HOUR % 2 )) -eq 0 ]; then
  echo "[2/4] Pulling eBay pricing..."
  cd "$EBAY_DIR"
  $PYTHON "$EBAY_DIR/ebay_scraper.py" || echo "WARNING: eBay pull failed"
else
  echo "[2/4] Skipping eBay (runs every 2h at :00)"
fi

# 3. Rebuild data.json
echo "[3/4] Rebuilding data.json..."
cd "$DASH_DIR"
$PYTHON "$DASH_DIR/build_data.py" || { echo "ERROR: build_data.py failed"; exit 1; }

# 4. Deploy to GitHub Pages
echo "[4/4] Deploying to GitHub Pages..."
cp "$DASH_DIR/dashboard.html" "$DASH_DIR/deploy/index.html"
cp "$DASH_DIR/data.json" "$DASH_DIR/deploy/data.json"
cp "$DASH_DIR/store_map.json" "$DASH_DIR/deploy/store_map.json" 2>/dev/null || true
cd "$DASH_DIR/deploy"
git add -A
git commit -m "Auto-update $(date '+%Y-%m-%d %H:%M')" || echo "No changes to commit"
git push origin main || echo "WARNING: git push failed"

echo "=== Done $(date '+%Y-%m-%d %H:%M:%S') ==="
