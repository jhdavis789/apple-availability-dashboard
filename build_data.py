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
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "data.json")
EBAY_DB_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "Ebay Scrape", "ebay_data.db"))
RAW_API_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "raw_api_responses"))


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
        model = row[0].strip().strip('"')
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
        product_names = [p["product_name"] for p in products]
        data = {}
        for row in rows:
            name = row["product_name"]
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


def _parse_one_raw_file(filepath):
    """Parse a single raw API response file and return (timestamp, store_avail_dict).

    store_avail_dict is keyed by storeNumber, value is {product_name: 0|1}.
    Also captures store metadata (name, lat, lng, city, state) on first encounter.
    Handles both single-product and batch response formats.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"  Skipping {os.path.basename(filepath)}: {e}")
        return None, None, None

    timestamp = raw.get("timestamp", "")
    store_meta = {}  # storeNumber -> {id, name, lat, lng, city, state}
    store_avail = {}  # storeNumber -> {product_name: 0|1}

    # Build part->product_name map from the top-level products dict
    products_map = raw.get("products", {})  # {product_name: part_number}
    part_to_name = {v: k for k, v in products_map.items()}

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
                if name and name != "batch":
                    store_avail[sn][name] = avail

    return timestamp, store_meta, store_avail


def load_store_map():
    """Load per-store availability data from ALL raw API response files.

    Builds a time series of per-store, per-product availability for the
    time slider and delta features. Also includes the latest snapshot for
    default display.
    """
    if not os.path.isdir(RAW_API_DIR):
        print(f"Raw API directory not found at {RAW_API_DIR}, skipping store map")
        return None

    files = sorted(glob.glob(os.path.join(RAW_API_DIR, "raw_responses_*.json")))
    candidates = [f for f in files if "wave2" not in os.path.basename(f)]
    if not candidates:
        print("No raw API response files found, skipping store map")
        return None

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


def load_cycles():
    """Detect restock-to-sellout cycles from raw API response files.

    Returns a dict with:
      - cycles: list of individual cycle events
      - store_rankings: stores ranked by demand (fastest sellout)
      - summary: aggregate stats
    """
    if not os.path.isdir(RAW_API_DIR):
        print("Raw API directory not found, skipping cycles")
        return None

    files = sorted(glob.glob(os.path.join(RAW_API_DIR, "raw_responses_*.json")))
    candidates = [f for f in files if "wave2" not in os.path.basename(f)]
    if not candidates:
        print("No raw API response files found, skipping cycles")
        return None

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
                    if name and name != "batch":
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

    # Detect cycles
    cycles = []
    for (sn, prod), entries in timelines.items():
        restock_ts = None
        for i, (ts_str, display) in enumerate(entries):
            if display == "available":
                if i == 0 or entries[i - 1][1] in ("ineligible", "unavailable"):
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
                    cycles.append({
                        "store": sn,
                        "store_name": store_info.get(sn, {}).get("name", sn),
                        "city": extra.get("city", ""),
                        "state": extra.get("state", ""),
                        "region": store_region.get(sn, ""),
                        "product": prod,
                        "restock": restock_ts,
                        "sellout": ts_str,
                        "hours": round(duration_h, 2),
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

    # Load eBay price data
    ebay_prices = load_ebay_prices()

    # Load per-store map data
    store_map = load_store_map()

    # Load restock/sellout cycle data
    cycle_data = load_cycles()

    output = {
        "generated_at": datetime.now().isoformat(),
        "total_snapshots": len(snapshots),
        "all_models": sorted(all_models),
        "all_cities": sorted(all_cities),
        "snapshots": snapshots,
    }
    if ebay_prices:
        output["ebay_prices"] = ebay_prices
    if store_map:
        output["store_map"] = store_map
    if cycle_data:
        output["cycles"] = cycle_data

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Written {len(snapshots)} snapshots to {OUTPUT_FILE}")
    print(f"Models: {sorted(all_models)}")
    print(f"Cities: {sorted(all_cities)}")


if __name__ == "__main__":
    main()
