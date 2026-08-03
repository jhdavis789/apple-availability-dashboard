#!/usr/bin/env python3
"""
Preprocess all availability CSV files into a single JSON file for the dashboard.
Parses timestamps from filenames, extracts availability percentages,
and flags erroneous snapshots (store count = 0, ERR values, etc.).

CSV files are read from the PARENT directory (../).
data.json is written into this script's own directory (dashboard/).
"""

import csv
import glob
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
CSV_DIR = os.path.join(BASE_DIR, "csvs")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "data.json")
EBAY_DB_PATH = os.path.join(BASE_DIR, "Ebay Scrape", "ebay_data.db")

# Raw snapshots live in the compact v3 archive and must be read through the
# codec — they are gzipped and share a content-addressed blob pool, so a plain
# glob+json.load finds nothing. That is exactly what happened: the scraper moved
# to v3 on 2026-07-20, this file kept globbing `../raw_api_responses/*.json`,
# and the heatmap silently froze for two weeks because "no files found" was a
# print statement rather than an error. See DATA_SOURCES.md, Apple Store Retail
# API: "Read/write ONLY via raw_codec.py".
sys.path.insert(0, BASE_DIR)
import raw_codec  # noqa: E402

RAW_API_DIR = raw_codec.DEFAULT_DIR


def raw_paths():
    """Every raw snapshot path, oldest first, excluding the wave2 probe junk.

    Raises if the archive is empty. Every scraper run writes a raw snapshot, so
    "no raw files" always means the reader is looking in the wrong place — the
    condition that froze the heatmap for two weeks while build_data.py printed
    "No raw API response files found" and exited 0. It must be loud.
    """
    paths = [p for p in raw_codec.iter_raw_paths()
             if "wave2" not in os.path.basename(p)]
    if not paths:
        raise RuntimeError(
            f"No raw snapshots under {raw_codec.DEFAULT_DIR}. The scraper writes "
            f"one per run, so this means the archive moved or the codec is "
            f"pointed at the wrong directory — not that there is no data."
        )
    return paths


def read_raw(path):
    """One raw snapshot in the legacy v2 shape, or None if unreadable."""
    try:
        return raw_codec.load_raw(path)
    except Exception as e:  # noqa: BLE001 - report and skip the one file
        print(f"  Skipping {os.path.basename(path)}: {e}")
        return None

# Models to exclude from dashboard output (easy to add more later)
EXCLUDED_MODELS = {
    'MacBook Pro 14" M4 Max ($3,499)',
}

# Historical model name -> current name.
#
# The series key IS the display name, and the display name contains the price.
# So when Apple reprices a SKU and we correct the label, the same physical
# product becomes two disconnected series with a break at the rename date.
# This map stitches them back together. Both sides are the SAME part number —
# only the price text changed. Never alias two different parts together.
#
# 2026-08-03: all ten labels were repriced after `discover_skus.py --prices`
# found every one of them stale (Macs ~+$200, Mac Studio +$500/+$1,300, iPhone
# 16 -$70 once carrier-activation pricing was separated from unlocked).
MODEL_ALIASES = {
    'Mac Mini M4 16/512 ($799)': 'Mac Mini M4 16/512 ($999)',
    'Mac Mini M4 24/512 ($999)': 'Mac Mini M4 24/512 ($1,199)',
    'Mac Mini M4 Pro ($1,399)': 'Mac Mini M4 Pro ($1,599)',
    'Mac Studio M4 Max ($1,999)': 'Mac Studio M4 Max ($2,499)',
    'Mac Studio M3 Ultra ($3,999)': 'Mac Studio M3 Ultra ($5,299)',
    'iMac 24" M4 8-core ($1,299)': 'iMac 24" M4 8-core ($1,499)',
    'iMac 24" M4 10-core ($1,499)': 'iMac 24" M4 10-core ($1,699)',
    'MacBook Pro 14" M5 ($1,699)': 'MacBook Pro 14" M5 ($1,999)',
    'MacBook Pro 14" M5 Pro ($2,199)': 'MacBook Pro 14" M5 Pro ($2,499)',
    'iPhone 16 128GB ($799)': 'iPhone 16 128GB ($729)',
    # The $599 base was tracked before Apple repriced it to $799; same part
    # (MU9D3LL/A), which is back in the tracked set as the 16/256 config.
    'Mac Mini M4 ($599)': 'Mac Mini M4 16/256 ($799)',
}

# An alias whose target is itself an alias key would silently half-migrate.
assert not (set(MODEL_ALIASES.values()) & set(MODEL_ALIASES)), \
    "MODEL_ALIASES must not chain: a target is also a key"


def canonical_model(name):
    """Map a historical model name onto its current name."""
    return MODEL_ALIASES.get(name, name)


def parse_timestamp(filename):
    """Extract datetime from filename like availability_matrix_20260207_130915.csv"""
    base = os.path.basename(filename)
    match = re.search(r"(\d{8})_(\d{6})", base)
    if not match:
        return None
    date_str, time_str = match.groups()
    return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S").isoformat()


def parse_csv_file(filepath):
    """Parse a single CSV file and return structured data."""
    timestamp = parse_timestamp(filepath)
    if not timestamp:
        return None

    filename = os.path.basename(filepath)
    file_type = "standard"
    if "3_item" in filename:
        file_type = "3_item"
    elif "availability_qty" in filename:
        file_type = "qty"

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return None

    rows = []
    # Use csv reader to handle quoted fields with commas
    reader = csv.reader(content.splitlines())
    for row in reader:
        rows.append(row)

    if len(rows) < 2:
        return None

    header = rows[0]
    # Parse city names and store counts from header
    cities = []
    for col in header[1:]:
        col = col.strip()
        match = re.match(r"(.+?)\s*\((\d+|\?)\)", col)
        if match:
            city_name = match.group(1).strip()
            store_count_str = match.group(2)
            store_count = int(store_count_str) if store_count_str != "?" else -1
            cities.append({"name": city_name, "store_count": store_count})
        else:
            cities.append({"name": col, "store_count": -1})

    # Parse data rows
    products = []
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        model = canonical_model(row[0].strip().strip('"'))
        values = {}
        for i, city_info in enumerate(cities):
            if i + 1 < len(row):
                raw = row[i + 1].strip()
                if raw in ("ERR", "N/A", ""):
                    values[city_info["name"]] = None
                else:
                    # Parse percentage
                    pct = raw.replace("%", "").strip()
                    try:
                        values[city_info["name"]] = int(pct)
                    except ValueError:
                        values[city_info["name"]] = None
            else:
                values[city_info["name"]] = None
        products.append({"model": model, "values": values})

    return {
        "timestamp": timestamp,
        "file": filename,
        "file_type": file_type,
        "cities": cities,
        "products": products,
    }


def load_ebay_prices():
    """Load eBay price history from the SQLite database."""
    if not os.path.exists(EBAY_DB_PATH):
        print(f"eBay database not found at {EBAY_DB_PATH}, skipping")
        return None

    try:
        conn = sqlite3.connect(EBAY_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all products
        cursor.execute("SELECT id, product_name, model_number FROM products ORDER BY id")
        products = cursor.fetchall()

        # Get all availability summaries
        cursor.execute("""
            SELECT
                p.product_name,
                a.scraped_at,
                a.avg_price,
                a.median_price,
                a.min_price,
                a.max_price,
                a.total_listings
            FROM availability_summary a
            JOIN products p ON a.product_id = p.id
            ORDER BY p.product_name, a.scraped_at
        """)
        rows = cursor.fetchall()
        conn.close()

        # Build structured output: { products: [...], data: { product_name: [...] } }
        # Canonicalise: the eBay DB stores whatever display name a product had
        # when it was first scraped, and the dashboard joins these keys to the
        # availability model names. A relabelled SKU therefore silently orphans
        # its whole price series — which is exactly what happened on 2026-08-03.
        product_names = [canonical_model(p["product_name"]) for p in products]
        data = {}
        for row in rows:
            name = canonical_model(row["product_name"])
            if name not in data:
                data[name] = []
            # Convert scraped_at from "YYYY-MM-DD HH:MM:SS" to ISO format
            # Append Z since SQLite CURRENT_TIMESTAMP is UTC
            ts = row["scraped_at"].replace(" ", "T") + "Z"
            data[name].append({
                "timestamp": ts,
                "avg_price": round(row["avg_price"], 2) if row["avg_price"] else None,
                "median_price": round(row["median_price"], 2) if row["median_price"] else None,
                "min_price": round(row["min_price"], 2) if row["min_price"] else None,
                "max_price": round(row["max_price"], 2) if row["max_price"] else None,
                "total_listings": row["total_listings"],
            })

        total_points = sum(len(v) for v in data.values())
        print(f"Loaded eBay price data: {len(product_names)} products, {total_points} data points")
        return {"products": product_names, "data": data}

    except Exception as e:
        print(f"Error loading eBay data: {e}")
        return None


def load_carried_values():
    """Availability measured against the stores that actually stock each SKU.

    Apple's pickup API returns three states, and `ineligible` is overloaded:

      * ~38% of (store, SKU) pairs are `ineligible` for the entire window —
        that store simply does not carry that configuration.
      * but `ineligible` is ALSO the state a carried SKU falls into when it
        sells out. Measured over 2.5 days, `available <-> ineligible`
        transitions outnumber `available <-> unavailable` 356 to 44.

    So dropping every `ineligible` reading (the obvious reading of "exclude it
    from both camps") deletes the primary out-of-stock signal and reports
    MacBook Pro, iMac and the Mac mini 16/512 at 100% available. Splitting on
    the letter is the wrong cut.

    The right cut is per (store, SKU) PERSISTENCE: a store that has ever shown
    the SKU as available demonstrably stocks it, so its `ineligible` readings
    mean sold out and belong in the denominator. A store that has never shown it
    available in the whole archive does not stock it and belongs in neither
    camp — it is dropped from the denominator entirely.

    Returns ({timestamp: {model: {city: pct}}}, {model: {city: carried_count}}).
    Only covers the raw retention window; older snapshots keep the all-stores
    percentage the scraper wrote into the CSVs, which cannot be recomputed.
    """
    paths = raw_paths()
    print(f"Computing carried-store availability from {len(paths)} raw files...")

    # Pass 1: which stores have ever had this SKU available.
    ever = defaultdict(set)          # sku -> {store}
    for p in paths:
        raw = read_raw(p)
        if raw is None:
            continue
        for snap in raw.get("snapshots", []):
            for store, per_sku in (snap.get("avail") or {}).items():
                for sku, letter in per_sku.items():
                    if letter == "a":
                        ever[sku].add(store)

    # Pass 2: per snapshot, per city, availability over carried stores only.
    by_ts = {}
    carried_counts = defaultdict(lambda: defaultdict(int))
    for p in paths:
        raw = read_raw(p)
        if raw is None:
            continue
        # Key on the FILENAME stamp, not raw["timestamp"]. The internal field
        # records when the payload was serialised (2026-08-03T12:01:55.445523)
        # while the filename carries the scheduled tick (…_120000) — which is
        # what the CSV snapshots are keyed on. Using the internal one joins
        # nothing, silently.
        ts = parse_timestamp(p)
        if not ts:
            continue
        city_of = {}
        for city, stores in (raw.get("city_assignments") or {}).items():
            for sn in stores:
                city_of[sn] = city
        part_to_name = {v: k for k, v in (raw.get("products") or {}).items()}

        # A store can appear under several zip batches; 'a' must win over 'i'.
        state = defaultdict(dict)     # sku -> store -> letter
        for snap in raw.get("snapshots", []):
            part_to_name.update({v: k for k, v in (snap.get("parts") or {}).items()})
            for store, per_sku in (snap.get("avail") or {}).items():
                for sku, letter in per_sku.items():
                    prev = state[sku].get(store)
                    if prev is None or (prev != "a" and letter == "a"):
                        state[sku][store] = letter

        out = {}
        for sku, per_store in state.items():
            name = canonical_model(part_to_name.get(sku, sku))
            if name in EXCLUDED_MODELS:
                continue
            num = defaultdict(int)
            den = defaultdict(int)
            for store, letter in per_store.items():
                city = city_of.get(store)
                if city is None or store not in ever.get(sku, ()):
                    continue          # unassigned, or this store never stocks it
                den[city] += 1
                if letter == "a":
                    num[city] += 1
            vals = {}
            for city, d in den.items():
                vals[city] = round(100 * num[city] / d)
                carried_counts[name][city] = max(carried_counts[name][city], d)
            if vals:
                out[name] = vals
        if out:
            by_ts[ts] = out

    print(f"  carried-store basis computed for {len(by_ts)} snapshots")
    return by_ts, {k: dict(v) for k, v in carried_counts.items()}


def _parse_one_raw_file(filepath):
    """Parse a single raw API response file and return (timestamp, store_avail_dict).

    store_avail_dict is keyed by storeNumber, value is {product_name: 0|1}.
    Also captures store metadata (name, lat, lng, city, state) on first encounter.
    Handles both single-product and batch response formats.
    """
    raw = read_raw(filepath)
    if raw is None:
        return None, None, None

    timestamp = raw.get("timestamp", "")
    store_meta = {}  # storeNumber -> {id, name, lat, lng, city, state}
    store_avail = {}  # storeNumber -> {product_name: 0|1}

    # Build part->product_name map from the top-level products dict
    products_map = raw.get("products", {})  # {product_name: part_number}
    part_to_name = {v: k for k, v in products_map.items()}

    # --- v2/v3 shape (current) --------------------------------------------
    # {"stores": {storeNumber: {name, lat, lon, city, state, ...}},
    #  "snapshots": [{"parts": {name: sku}, "avail": {store: {sku: "a"|"i"|"u"}}}]}
    if "snapshots" in raw:
        for sn, meta in (raw.get("stores") or {}).items():
            store_meta[sn] = {
                "id": sn,
                "name": meta.get("name", ""),
                # v2 renamed storelatitude/storelongitude to lat/lon.
                "lat": meta.get("lat"),
                "lng": meta.get("lon"),
                "city": meta.get("city", ""),
                "state": meta.get("state", ""),
            }
            store_avail.setdefault(sn, {})

        for snap in raw.get("snapshots", []):
            snap_part_to_name = {v: k for k, v in (snap.get("parts") or {}).items()}
            for sn, per_sku in (snap.get("avail") or {}).items():
                if sn not in store_avail:
                    # A store that appears in a snapshot but not the directory
                    # still deserves a row; coords just stay unknown.
                    store_meta.setdefault(sn, {"id": sn, "name": "", "lat": None,
                                               "lng": None, "city": "", "state": ""})
                    store_avail[sn] = {}
                for sku, letter in per_sku.items():
                    name = snap_part_to_name.get(sku) or part_to_name.get(sku)
                    if name and name not in EXCLUDED_MODELS:
                        store_avail[sn][canonical_model(name)] = 1 if letter == "a" else 0
        return timestamp, store_meta, store_avail

    # --- v1 shape (deprecated, kept so old archives still parse) -----------
    for resp in raw.get("responses", []):
        product = resp.get("product", "")
        # For batch responses, build part->name from the resp-level parts dict
        resp_parts = resp.get("parts", {})  # {product_name: part_number}
        resp_part_to_name = {v: k for k, v in resp_parts.items()}

        body = resp.get("response", {}).get("body", {})
        for s in body.get("stores", []):
            sn = s.get("storeNumber", "")
            if not sn:
                continue

            if sn not in store_meta:
                store_meta[sn] = {
                    "id": sn,
                    "name": s.get("storeName", ""),
                    "lat": s.get("storelatitude"),
                    "lng": s.get("storelongitude"),
                    "city": s.get("city", ""),
                    "state": s.get("state", ""),
                }
                store_avail[sn] = {}

            pa = s.get("partsAvailability", {})
            for pn, info in pa.items():
                pickup = info.get("pickupDisplay", "")
                avail = 1 if pickup == "available" else 0
                # Resolve product name: try resp-level parts, then top-level, then use raw product field
                name = resp_part_to_name.get(pn) or part_to_name.get(pn) or product
                if name and name != "batch" and name not in EXCLUDED_MODELS:
                    store_avail[sn][name] = avail

    return timestamp, store_meta, store_avail


def load_store_map():
    """Load per-store availability data from ALL raw API response files.

    Builds a time series of per-store, per-product availability for the
    time slider and delta features. Also includes the latest snapshot for
    default display.
    """
    candidates = raw_paths()
    print(f"Loading store map from {len(candidates)} raw API files...")

    # Collect store metadata from all files (latest wins for coords etc.)
    all_store_meta = {}
    # Time series: list of {timestamp, avail: {storeNumber: {product: 0|1}}}
    snapshots = []

    for filepath in candidates:
        timestamp, store_meta, store_avail = _parse_one_raw_file(filepath)
        if timestamp is None:
            continue

        # Merge metadata (latest file updates coords/names)
        all_store_meta.update(store_meta)

        # Build compact availability snapshot
        # Only store per-store avail as {product_name: 0|1}
        snapshots.append({
            "t": timestamp,
            "a": store_avail,  # {storeNumber: {product: 0|1}}
        })

    # Filter to stores with coordinates
    valid_stores = {
        sn: meta for sn, meta in all_store_meta.items()
        if meta["lat"] is not None and meta["lng"] is not None
    }

    store_list = list(valid_stores.values())

    # Build compact time series: for each snapshot, only include store IDs present
    # Convert to smaller format: snapshots[].a = {storeId: {product: 0|1}}
    # Filter snapshot avail to only valid stores
    compact_snapshots = []
    for snap in snapshots:
        filtered_avail = {}
        for sn, avail in snap["a"].items():
            if sn in valid_stores:
                filtered_avail[sn] = avail
        compact_snapshots.append({
            "t": snap["t"],
            "a": filtered_avail,
        })

    print(f"Loaded store map: {len(store_list)} stores, {len(compact_snapshots)} snapshots")
    return {
        "stores": store_list,
        "snapshots": compact_snapshots,
    }


def _parse_lead_days(quote):
    """Map pickupSearchQuote to numeric lead days."""
    if not quote:
        return None
    q = quote.lower()
    if "today" in q:
        return 0
    if "tomorrow" in q:
        return 1
    return None


def load_lead_times():
    """Extract pickup lead time data from raw API response files.

    Parses pickupSearchQuote ("Available Today" = 0 days, "Available Tomorrow" = 1 day)
    for each store+product, then aggregates to per-city averages.
    """
    # Load store-to-region mapping for city aggregation
    sa_path = os.path.join(BASE_DIR, "store_assignments.json")
    store_region = {}
    if os.path.exists(sa_path):
        with open(sa_path) as f:
            sa = json.load(f)
        for region, store_nums in sa.get("city_assignments", {}).items():
            for sn in store_nums:
                store_region[sn] = region

    candidates = raw_paths()

    print(f"Extracting lead times from {len(candidates)} raw API files...")

    snapshots = []
    for filepath in candidates:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            continue

        timestamp = raw.get("timestamp", "")
        if not timestamp:
            continue

        products_map = raw.get("products", {})
        part_to_name = {v: k for k, v in products_map.items()}

        # {city: {product: [lead_days, ...]}} for this snapshot
        city_product_leads = defaultdict(lambda: defaultdict(list))

        for resp in raw.get("responses", []):
            resp_parts = resp.get("parts", {})
            resp_part_to_name = {v: k for k, v in resp_parts.items()}
            product = resp.get("product", "")

            stores = resp.get("response", {}).get("body", {}).get("stores", [])
            for store in stores:
                sn = store.get("storeNumber", "")
                if not sn:
                    continue
                region = store_region.get(sn)
                if not region:
                    continue

                for pn, pv in store.get("partsAvailability", {}).items():
                    name = resp_part_to_name.get(pn) or part_to_name.get(pn) or product
                    if not name or name == "batch":
                        continue
                    if name in EXCLUDED_MODELS:
                        continue

                    quote = pv.get("pickupSearchQuote", "")
                    lead = _parse_lead_days(quote)
                    if lead is not None:
                        city_product_leads[region][name].append(lead)

        # Compute averages for this snapshot
        if city_product_leads:
            data = {}
            for city, products in city_product_leads.items():
                data[city] = {}
                for prod, leads in products.items():
                    if leads:
                        data[city][prod] = round(sum(leads) / len(leads), 2)
            snapshots.append({"t": timestamp, "data": data})

    print(f"  Extracted lead times from {len(snapshots)} snapshots")
    return {"snapshots": snapshots} if snapshots else None


def load_delivery_times():
    """Extract delivery time data from raw API response files.

    Reads the delivery_times key directly (already per-city, per-product).
    Returns {"snapshots": [{"t": timestamp, "data": {city: {product: days}}}]}.
    """
    candidates = raw_paths()
    print(f"Extracting delivery times from {len(candidates)} raw API files...")

    snapshots = []
    for filepath in candidates:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            continue

        timestamp = raw.get("timestamp", "")
        if not timestamp:
            continue

        dt = raw.get("delivery_times")
        if not dt:
            continue

        # Reshape from {product: {city: {days, ...}}} to {city: {product: days}}
        data = {}
        for product, cities in dt.items():
            if product in EXCLUDED_MODELS:
                continue
            for city, info in cities.items():
                days = info.get("days") if isinstance(info, dict) else None
                if days is not None:
                    if city not in data:
                        data[city] = {}
                    data[city][product] = days

        if data:
            snapshots.append({"t": timestamp, "data": data})

    print(f"  Extracted delivery times from {len(snapshots)} snapshots")
    return {"snapshots": snapshots} if snapshots else None


STORE_ASSIGNMENTS_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "store_assignments.json"))
GLITCH_INELIGIBLE_THRESHOLD = 0.80


def _is_glitch_snapshot(raw_data):
    """Detect glitch snapshots where nearly all stores show ineligible."""
    total = 0
    ineligible = 0
    for resp in raw_data.get("responses", []):
        stores = resp.get("response", {}).get("body", {}).get("stores", [])
        for store in stores:
            for pn, pv in store.get("partsAvailability", {}).items():
                total += 1
                if pv.get("pickupDisplay") == "ineligible":
                    ineligible += 1
    if total == 0:
        return True
    return (ineligible / total) >= GLITCH_INELIGIBLE_THRESHOLD


def _extract_store_hours(raw_api_dir):
    """Extract store hours and timezone from the latest raw API file.

    Returns dict: storeNumber -> {
        'timezone': str,
        'schedule': {0: (open_h, close_h), 1: ..., 6: ...}  # 0=Mon, 6=Sun
    }
    where open_h/close_h are fractional hours (e.g. 9.0 for 9 AM, 21.0 for 9 PM).
    """
    import re as _re

    candidates = raw_paths()
    # Use latest file for store hours
    with open(candidates[-1], "r", encoding="utf-8") as f:
        raw = json.load(f)

    def parse_time(t_str):
        """Parse '9:00 AM' -> 9.0, '9:30 PM' -> 21.5"""
        m = _re.match(r'(\d+):(\d+)\s*(AM|PM)', t_str.strip(), _re.IGNORECASE)
        if not m:
            return None
        h, mn, ap = int(m.group(1)), int(m.group(2)), m.group(3).upper()
        if ap == 'PM' and h != 12:
            h += 12
        elif ap == 'AM' and h == 12:
            h = 0
        return h + mn / 60.0

    DAY_MAP = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6}

    def expand_days(days_str):
        """Expand 'Mon-Sat:' -> [0,1,2,3,4,5], 'Sun:' -> [6]"""
        days_str = days_str.strip().rstrip(':').strip()
        result = []
        for part in days_str.split(','):
            part = part.strip()
            if '-' in part:
                start_s, end_s = part.split('-', 1)
                start_d = DAY_MAP.get(start_s.strip()[:3].lower())
                end_d = DAY_MAP.get(end_s.strip()[:3].lower())
                if start_d is not None and end_d is not None:
                    if start_d <= end_d:
                        result.extend(range(start_d, end_d + 1))
                    else:
                        result.extend(range(start_d, 7))
                        result.extend(range(0, end_d + 1))
            else:
                d = DAY_MAP.get(part[:3].lower())
                if d is not None:
                    result.append(d)
        return result

    store_hours = {}
    for resp in raw.get("responses", []):
        stores = resp.get("response", {}).get("body", {}).get("stores", [])
        for store in stores:
            sn = store.get("storeNumber", "")
            if not sn or sn in store_hours:
                continue
            tz = store.get("retailStore", {}).get("timezone", "")
            sh = store.get("storeHours", {})
            hours_list = sh.get("hours", [])
            schedule = {}
            for entry in hours_list:
                timings = entry.get("storeTimings", "")
                days_str = entry.get("storeDays", "")
                if '-' not in timings:
                    continue
                open_s, close_s = timings.split('-', 1)
                open_h = parse_time(open_s)
                close_h = parse_time(close_s)
                if open_h is None or close_h is None:
                    continue
                for d in expand_days(days_str):
                    schedule[d] = (open_h, close_h)
            if schedule and tz:
                store_hours[sn] = {"timezone": tz, "schedule": schedule}

    return store_hours


def _compute_open_hours(restock_str, sellout_str, schedule, tz_name):
    """Compute total open hours between restock and sellout times.

    Walks day by day, summing only the hours the store is open.
    Falls back to wall-clock hours if timezone handling fails.
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        # Python < 3.9 fallback
        return None

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return None

    t0 = datetime.fromisoformat(restock_str.replace("Z", "+00:00").replace("+00:00", ""))
    t1 = datetime.fromisoformat(sellout_str.replace("Z", "+00:00").replace("+00:00", ""))

    # Timestamps are in the machine's local timezone (America/Los_Angeles)
    machine_tz = ZoneInfo("America/Los_Angeles")
    t0_aware = t0.replace(tzinfo=machine_tz) if t0.tzinfo is None else t0
    t1_aware = t1.replace(tzinfo=machine_tz) if t1.tzinfo is None else t1

    # Convert to store local time
    t0_local = t0_aware.astimezone(tz)
    t1_local = t1_aware.astimezone(tz)

    total_open = 0.0
    current = t0_local

    while current < t1_local:
        dow = current.weekday()  # 0=Mon, 6=Sun
        open_h, close_h = schedule.get(dow, (0, 24))

        current_h = current.hour + current.minute / 60.0
        # End of this day's window
        end_of_day = current.replace(hour=0, minute=0, second=0, microsecond=0)
        from datetime import timedelta
        end_of_day = end_of_day + timedelta(hours=close_h)

        if current_h < open_h:
            # Before store opens — skip to open time
            current = current.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=open_h)
            if current >= t1_local:
                break
            current_h = open_h

        if current_h >= close_h:
            # Past closing — skip to next day's opening
            current = current.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            continue

        # We're within open hours
        effective_end = min(t1_local, end_of_day)
        if effective_end > current:
            total_open += (effective_end - current).total_seconds() / 3600.0
        current = end_of_day

    return round(total_open, 2)


def load_cycles():
    """Detect restock-to-sellout cycles from raw API response files.

    Returns a dict with:
      - cycles: list of individual cycle events
      - store_rankings: stores ranked by demand (fastest sellout)
      - summary: aggregate stats
    """
    candidates = raw_paths()
    print(f"Computing restock/sellout cycles from {len(candidates)} files...")

    # Load store-to-region mapping
    store_region = {}  # storeNumber -> region (e.g. "NYC")
    store_extra = {}   # storeNumber -> {city, state}
    if os.path.exists(STORE_ASSIGNMENTS_PATH):
        with open(STORE_ASSIGNMENTS_PATH) as f:
            sa = json.load(f)
        for region, store_nums in sa.get("city_assignments", {}).items():
            for sn in store_nums:
                store_region[sn] = region
        for sn, meta in sa.get("stores", {}).items():
            store_extra[sn] = {"city": meta.get("city", ""), "state": meta.get("state", "")}
        print(f"  Loaded region mapping for {len(store_region)} stores")

    # Load store hours for open-hours cycle calculation
    store_hours = _extract_store_hours(RAW_API_DIR)
    if store_hours:
        print(f"  Loaded store hours for {len(store_hours)} stores")

    # Build per-(store, product) timelines: [(timestamp_str, is_available)]
    timeline_raw = defaultdict(lambda: defaultdict(list))
    store_info = {}  # storeNumber -> {name, timezone}
    skipped = 0

    for fpath in candidates:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            continue

        ts_str = raw.get("timestamp", "")
        if not ts_str:
            continue

        if _is_glitch_snapshot(raw):
            skipped += 1
            continue

        products_map = raw.get("products", {})
        part_to_name = {v: k for k, v in products_map.items()}

        for resp in raw.get("responses", []):
            product = resp.get("product", "")
            resp_parts = resp.get("parts", {})
            resp_part_to_name = {v: k for k, v in resp_parts.items()}

            stores = resp.get("response", {}).get("body", {}).get("stores", [])
            for store in stores:
                sn = store.get("storeNumber", "")
                if not sn:
                    continue

                if sn not in store_info:
                    store_info[sn] = {
                        "name": store.get("storeName", ""),
                        "timezone": store.get("retailStore", {}).get("timezone", ""),
                    }

                for pn, pv in store.get("partsAvailability", {}).items():
                    display = pv.get("pickupDisplay", "")
                    name = resp_part_to_name.get(pn) or part_to_name.get(pn) or product
                    if name and name != "batch" and name not in EXCLUDED_MODELS:
                        timeline_raw[sn][name].append((ts_str, display))

    if skipped:
        print(f"  Skipped {skipped} glitch snapshots")

    # Sort and deduplicate timelines (only keep state changes)
    timelines = {}
    for sn in timeline_raw:
        for prod in timeline_raw[sn]:
            entries = sorted(timeline_raw[sn][prod], key=lambda x: x[0])
            deduped = []
            prev = None
            for ts, display in entries:
                if display != prev:
                    deduped.append((ts, display))
                    prev = display
            timelines[(sn, prod)] = deduped

    # Find the earliest snapshot timestamp (first observation — not a real restock)
    all_timestamps = set()
    for (sn, prod), entries in timelines.items():
        if entries:
            all_timestamps.add(entries[0][0])
    first_snapshot_ts = min(all_timestamps) if all_timestamps else None

    # Detect cycles
    cycles = []
    for (sn, prod), entries in timelines.items():
        restock_ts = None
        for i, (ts_str, display) in enumerate(entries):
            if display == "available":
                if i == 0 or entries[i - 1][1] in ("ineligible", "unavailable"):
                    # Skip if this is the very first observation (left-censored)
                    if i == 0 and ts_str == first_snapshot_ts:
                        continue
                    restock_ts = ts_str
            elif display in ("ineligible", "unavailable"):
                if restock_ts is not None:
                    # Compute duration
                    try:
                        t0 = datetime.fromisoformat(restock_ts.replace("Z", "+00:00").replace("+00:00", ""))
                        t1 = datetime.fromisoformat(ts_str.replace("Z", "+00:00").replace("+00:00", ""))
                        duration_h = (t1 - t0).total_seconds() / 3600.0
                    except Exception:
                        restock_ts = None
                        continue
                    if duration_h < 5.0 / 60.0:  # skip <5 min noise
                        restock_ts = None
                        continue
                    extra = store_extra.get(sn, {})
                    # Compute open hours (subtracting closed hours)
                    open_h = duration_h
                    sh = store_hours.get(sn)
                    if sh and sh.get("schedule"):
                        oh = _compute_open_hours(restock_ts, ts_str, sh["schedule"], sh["timezone"])
                        if oh is not None:
                            open_h = oh
                    cycles.append({
                        "store": sn,
                        "store_name": store_info.get(sn, {}).get("name", sn),
                        "city": extra.get("city", ""),
                        "state": extra.get("state", ""),
                        "region": store_region.get(sn, ""),
                        "product": prod,
                        "restock": restock_ts,
                        "sellout": ts_str,
                        "hours": round(open_h, 2),
                        "wall_hours": round(duration_h, 2),
                    })
                    restock_ts = None

    if not cycles:
        print("  No complete cycles found")
        return None

    # Build store rankings
    store_cycles = defaultdict(list)
    for c in cycles:
        store_cycles[c["store"]].append(c)

    rankings = []
    for sn, sc in store_cycles.items():
        avg_h = sum(c["hours"] for c in sc) / len(sc)
        extra = store_extra.get(sn, {})
        rankings.append({
            "store": sn,
            "name": store_info.get(sn, {}).get("name", sn),
            "region": store_region.get(sn, ""),
            "city": extra.get("city", ""),
            "state": extra.get("state", ""),
            "cycles": len(sc),
            "avg_hours": round(avg_h, 2),
            "fastest": round(min(c["hours"] for c in sc), 2),
            "products": len(set(c["product"] for c in sc)),
        })
    rankings.sort(key=lambda r: r["avg_hours"])

    # Summary
    all_hours = [c["hours"] for c in cycles]
    same_day = sum(1 for c in cycles if c["restock"][:10] == c["sellout"][:10])
    summary = {
        "total_cycles": len(cycles),
        "avg_hours": round(sum(all_hours) / len(all_hours), 2),
        "fastest_hours": round(min(all_hours), 2),
        "same_day_count": same_day,
        "same_day_pct": round(100 * same_day / len(cycles), 1),
        "stores_with_cycles": len(rankings),
        "products_with_cycles": len(set(c["product"] for c in cycles)),
    }

    print(f"  Found {len(cycles)} cycles across {len(rankings)} stores")
    return {
        "cycles": sorted(cycles, key=lambda c: c["restock"]),
        "store_rankings": rankings,
        "summary": summary,
    }


def _trim_discontinued_tails(snapshots, recent_days=2):
    """End discontinued product lines at their last real (>0%) reading.

    "Currently tracked" = any model present in a snapshot within `recent_days`
    of the newest snapshot (time-based, so a collection outage gap can't pull
    stale products into the current set). For every other model, find the last
    timestamp it had a value > 0 in any city, then remove that model from all
    later snapshots so its chart line ends at discontinuation instead of
    flat-lining at 0% to the present. Models that never had a >0 reading are
    dropped entirely. Mutates `snapshots` in place.
    """
    if not snapshots:
        return
    snapshots.sort(key=lambda s: s["timestamp"])

    latest_dt = datetime.fromisoformat(snapshots[-1]["timestamp"])
    cutoff = latest_dt - timedelta(days=recent_days)
    current = set()
    for snap in snapshots:
        try:
            if datetime.fromisoformat(snap["timestamp"]) >= cutoff:
                for p in snap["products"]:
                    current.add(p["model"])
        except (ValueError, TypeError):
            continue

    # Last timestamp each discontinued model had real availability.
    last_alive = {}
    for snap in snapshots:
        ts = snap["timestamp"]
        for p in snap["products"]:
            m = p["model"]
            if m in current:
                continue
            if any((v is not None and v > 0) for v in p["values"].values()):
                last_alive[m] = ts

    trimmed = {}
    for snap in snapshots:
        ts = snap["timestamp"]
        kept = []
        for p in snap["products"]:
            m = p["model"]
            if m in current:
                kept.append(p)
                continue
            alive_ts = last_alive.get(m)
            if alive_ts is not None and ts <= alive_ts:
                kept.append(p)
            else:
                trimmed[m] = trimmed.get(m, 0) + 1
        snap["products"] = kept

    if trimmed:
        ended = {m: last_alive.get(m, "never") for m in trimmed}
        print(f"Ended {len(trimmed)} discontinued product line(s) at last real reading:")
        for m, ts in sorted(ended.items()):
            print(f"  {m} -> {str(ts)[:16]} ({trimmed[m]} trailing 0% points dropped)")


def main():
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "availability_matrix_*.csv")))
    print(f"Looking for CSVs in: {CSV_DIR}")
    print(f"Found {len(csv_files)} CSV files")

    snapshots = []
    all_models = set()
    all_cities = set()

    for f in csv_files:
        result = parse_csv_file(f)
        if result:
            snapshots.append(result)
            for p in result["products"]:
                all_models.add(p["model"])
            for c in result["cities"]:
                all_cities.add(c["name"])

    snapshots.sort(key=lambda x: x["timestamp"])

    # Filter excluded models from snapshots and all_models
    all_models -= EXCLUDED_MODELS
    for snap in snapshots:
        snap["products"] = [p for p in snap["products"] if p["model"] not in EXCLUDED_MODELS]

    # --- Downsample old snapshots to reduce data size ---
    # Last 48h: full resolution, 2-7d: hourly, 7-30d: 4h, 30d+: 8h
    now = datetime.now()
    downsampled = []
    last_kept_hour = None
    for snap in snapshots:
        try:
            ts = datetime.fromisoformat(snap["timestamp"])
        except Exception:
            downsampled.append(snap)
            continue
        age_hours = (now - ts).total_seconds() / 3600
        if age_hours <= 48:
            bucket = None  # keep all
        elif age_hours <= 168:  # 7 days
            bucket = ts.strftime("%Y%m%d%H")  # hourly
        elif age_hours <= 720:  # 30 days
            bucket = ts.strftime("%Y%m%d") + str(ts.hour // 4)  # 4h
        else:
            bucket = ts.strftime("%Y%m%d") + str(ts.hour // 8)  # 8h

        if bucket is None or bucket != last_kept_hour:
            # Strip unnecessary fields
            snap.pop("file", None)
            snap.pop("file_type", None)
            downsampled.append(snap)
            last_kept_hour = bucket

    raw_snapshot_count = len(snapshots)
    print(f"Downsampled {raw_snapshot_count} → {len(downsampled)} snapshots")
    snapshots = downsampled

    # --- End discontinued product lines at their last real data point ---
    # A product no longer tracked (absent from the recent snapshots) keeps recording
    # 0% while its Apple SKU is invalid, which draws a flat 0% tail to the present.
    # Drop each discontinued product's entries after its last >0% reading so its line
    # simply ends where the product was actually discontinued.
    _trim_discontinued_tails(snapshots)

    # --- Attach the carried-store denominator where raw data exists ---
    # Raw and CSV files are written by the same run and share a timestamp
    # exactly, so this is a direct key match, not a nearest-neighbour join.
    carried_by_ts, carried_counts = load_carried_values()
    attached = 0
    for snap in snapshots:
        per_model = carried_by_ts.get(snap["timestamp"])
        if not per_model:
            continue
        for prod in snap["products"]:
            vals = per_model.get(prod["model"])
            if vals:
                prod["cvalues"] = vals
                attached += 1
    print(f"  attached carried-basis values to {attached} product rows "
          f"across {sum(1 for s in snapshots if any('cvalues' in p for p in s['products']))} snapshots")

    # Load eBay price data and downsample older points
    ebay_prices = load_ebay_prices()

    # The dashboard joins eBay series to availability model names by string.
    # A relabelled SKU orphans its price series and the overlay just renders
    # nothing — no error, no empty state. Say so.
    if ebay_prices:
        sku_keyed = [n for n in ebay_prices["products"] if "(keyword)" not in n]
        orphans = [n for n in sku_keyed if n not in all_models]
        if orphans:
            print(f"WARNING: {len(orphans)}/{len(sku_keyed)} eBay series do not match "
                  f"any tracked model — their price overlay will not render:")
            for n in orphans:
                print(f"    {n}")
            print("  Add a MODEL_ALIASES entry, or fix Ebay Scrape/config.json.")

    if ebay_prices and "data" in ebay_prices:
        for prod, points in ebay_prices["data"].items():
            if not points:
                continue
            recent = []
            last_day = None
            for pt in points:
                ts_str = pt.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00").replace("+00:00", ""))
                    age_days = (now - ts).total_seconds() / 86400
                except Exception:
                    age_days = 0
                if age_days <= 14:
                    recent.append(pt)
                else:
                    day = ts_str[:10]
                    if day != last_day:
                        recent.append(pt)
                        last_day = day
            ebay_prices["data"][prod] = recent
        total_pts = sum(len(v) for v in ebay_prices["data"].values())
        print(f"eBay prices downsampled to {total_pts} points")

    # Load per-store map data → separate file with compact encoding
    store_map = load_store_map()
    STORE_MAP_FILE = os.path.join(SCRIPT_DIR, "store_map.json")
    if store_map:
        sm_snaps = store_map.get("snapshots", [])
        # Drop empty snapshots, keep every 2nd
        sm_snaps = [s for s in sm_snaps if s.get("a") and any(len(v) > 0 for v in s["a"].values())]
        sm_snaps = sm_snaps[::2]

        # Build product index for compact encoding
        all_sm_products = set()
        for snap in sm_snaps:
            for store_avail in snap["a"].values():
                all_sm_products.update(store_avail.keys())
        product_index = sorted(all_sm_products)

        # Convert {product_name: 0|1} → [0|1] array by index
        for snap in sm_snaps:
            compact_a = {}
            for store_id, avail_dict in snap["a"].items():
                compact_a[store_id] = [avail_dict.get(p, 0) for p in product_index]
            snap["a"] = compact_a

        compact_store_map = {
            "stores": store_map["stores"],
            "products": product_index,
            "snapshots": sm_snaps,
        }
        with open(STORE_MAP_FILE, "w") as f:
            json.dump(compact_store_map, f, separators=(",", ":"))
        sm_size = os.path.getsize(STORE_MAP_FILE)
        print(f"Store map: {len(sm_snaps)} snapshots, {len(product_index)} products → {sm_size/1024/1024:.1f} MB")

    output = {
        "generated_at": datetime.now().isoformat(),
        "total_snapshots": raw_snapshot_count,
        "all_models": sorted(all_models),
        "all_cities": sorted(all_cities),
        "snapshots": snapshots,
    }
    if ebay_prices:
        output["ebay_prices"] = ebay_prices
    # How many stores stock each SKU in each city — the denominator behind
    # `cvalues`, so the UI can show n and refuse to print a percentage off a
    # handful of stores.
    output["carried_counts"] = carried_counts
    output["carried_from"] = min(carried_by_ts) if carried_by_ts else None

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    core_size = os.path.getsize(OUTPUT_FILE)
    print(f"Core data: {len(snapshots)} snapshots → {core_size/1024/1024:.1f} MB")
    print(f"Models: {sorted(all_models)}")
    print(f"Cities: {sorted(all_cities)}")
    if EXCLUDED_MODELS:
        print(f"Excluded: {sorted(EXCLUDED_MODELS)}")


if __name__ == "__main__":
    main()
