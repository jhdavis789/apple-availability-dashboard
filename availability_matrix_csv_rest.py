#!/usr/bin/env python3
"""Apple Product Availability Matrix - CSV Output with Diff (Parallel API calls)
Includes store deduplication (nearest-city assignment) and overflow zip support."""
import requests
import csv
import json
import time
import glob
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def haversine(lat1, lon1, lat2, lon2):
    """Return distance in miles between two lat/lng points."""
    R = 3959  # Earth radius in miles
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

PRODUCTS = {
    "Mac Mini M4 ($599)": "MU9D3LL/A",
    "Mac Mini M4 Pro ($1,399)": "MCX44LL/A",
    "Mac Mini M4 16/512 ($799)": "MU9E3LL/A",
    "Mac Mini M4 24/512 ($999)": "MCYT4LL/A",
    "iMac 24\" M4 8-core ($1,299)": "MWUF3LL/A",
    "iMac 24\" M4 10-core ($1,499)": "MWV13LL/A",
    "MacBook Pro 14\" M5 ($1,599)": "MDE04LL/A",
    "MacBook Pro 14\" M4 Pro ($1,999)": "MX2H3LL/A",
    "MacBook Pro 14\" M4 Max ($3,499)": "MX2K3LL/A",
}

CITIES = {
    "NYC": "10001", "LA": "90001", "SF": "94102", "Austin": "78701",
    "Boston": "02139", "Chicago": "60601", "Houston": "77001", "Phoenix": "85001",
    "Seattle": "98101", "Miami": "33101", "Denver": "80201", "Atlanta": "30301",
}

# City center coordinates for nearest-city assignment via haversine
CITY_COORDS = {
    "NYC":     (40.7128, -74.0060),
    "LA":      (34.0522, -118.2437),
    "SF":      (37.7749, -122.4194),
    "Austin":  (30.2672, -97.7431),
    "Boston":  (42.3601, -71.0589),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "Seattle": (47.6062, -122.3321),
    "Miami":   (25.7617, -80.1918),
    "Denver":  (39.7392, -104.9903),
    "Atlanta": (33.7490, -84.3880),
}

# Overflow zip codes for metros where the 12-store API cap misses stores.
# Discovered stores are assigned to their nearest city, not necessarily the parent.
OVERFLOW_ZIPS = {
    "NYC": ["11201", "10314", "10801", "11501"],  # Brooklyn, Staten Island, Westchester, LI
    "LA":  ["92602"],                              # Irvine/OC (+7 stores: South Coast Plaza, Fashion Island, etc.)
    "SF":  ["95110"],                              # San Jose (+4 stores: Valley Fair, Apple Park, etc.)
}

# Stores farther than this from any tracked city are excluded
MAX_ASSIGNMENT_DISTANCE = 75  # miles

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
OUT_DIR = Path("/Users/Jackson/.openclaw/workspace/research/apple-availability")
ASSIGNMENTS_CACHE = OUT_DIR / "store_assignments.json"
MAX_WORKERS = 1  # Serialized to respect rate limits (~15 req burst, 541 after)

def check_batch(parts: list, zip_code: str) -> dict:
    """Check availability for multiple parts in one API call (batch).
    Returns dict with per-part store-level data for deduplication."""
    url = "https://www.apple.com/shop/retail/pickup-message"
    params = {"pl": "true", "location": zip_code}
    for i, part in enumerate(parts):
        params[f"parts.{i}"] = part

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)

            if resp.status_code == 541:
                wait = 15 * (attempt + 1)  # 15s, 30s, 45s backoff
                print(f"    Rate limited on {zip_code}, waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200 or not resp.text.startswith('{'):
                raise Exception(f"HTTP {resp.status_code}")

            data = resp.json()
            api_stores = data.get("body", {}).get("stores", [])

            # {part: {storeNumber: {available, pickupDisplay}}}
            part_store_avail = {p: {} for p in parts}
            for s in api_stores:
                sn = s.get("storeNumber", "")
                if not sn:
                    continue
                pa = s.get("partsAvailability", {})
                for part in parts:
                    info = pa.get(part, {})
                    display = info.get("pickupDisplay", "ineligible")
                    part_store_avail[part][sn] = {
                        "available": display == "available",
                        "pickupDisplay": display,
                    }

            return {"part_stores": part_store_avail, "total": len(api_stores),
                    "raw_response": data, "error": False}

        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"    ERR: {zip_code} ({e})")
                return {"part_stores": {p: {} for p in parts}, "total": 0,
                        "raw_response": None, "error": True}

    return {"part_stores": {p: {} for p in parts}, "total": 0,
            "raw_response": None, "error": True}


def check(part: str, zip_code: str) -> dict:
    """Single-part check (used for store discovery). Wraps check_batch."""
    result = check_batch([part], zip_code)
    store_avail = result["part_stores"].get(part, {})
    return {"stores": store_avail, "total": result["total"],
            "raw_response": result["raw_response"], "error": result["error"]}

def get_previous_csv():
    """Get the most recent CSV file before this run."""
    files = sorted(glob.glob(str(OUT_DIR / "availability_matrix_*.csv")))
    return files[-1] if files else None

def load_csv(path):
    """Load CSV into dict: {model: {city: pct}}"""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["Model"]
            data[model] = {k: v for k, v in row.items() if k != "Model"}
    return data

def print_diff(old_path, new_path):
    """Print diff matrix between old and new CSV."""
    old = load_csv(old_path)
    new = load_csv(new_path)

    print(f"\n{'='*100}")
    print(f"DIFF: Changes from previous run")
    print(f"{'='*100}")

    # Get cities from first row
    cities = list(list(new.values())[0].keys()) if new else []

    # Header
    print(f"{'Model':<40}", end="")
    for c in cities:
        city_name = c.split(" (")[0]
        print(f"{city_name:>8}", end="")
    print()
    print("-" * 100)

    changes = 0
    for model in new:
        row = f"{model:<40}"
        has_change = False
        for city in cities:
            old_val = old.get(model, {}).get(city, "N/A")
            new_val = new[model].get(city, "N/A")

            if old_val != new_val:
                # Show change as old→new
                old_num = old_val.replace("%", "").replace("ERR", "?")
                new_num = new_val.replace("%", "").replace("ERR", "?")
                row += f"{old_num}→{new_num}".rjust(8)
                has_change = True
                changes += 1
            else:
                row += f"{'·':>8}"

        if has_change:
            print(row)

    if changes == 0:
        print("  No changes detected.")
    else:
        print(f"\n  {changes} cell(s) changed.")

def print_matrix(rows, counts):
    """Print the raw availability matrix."""
    print(f"\n{'='*100}")
    print(f"CURRENT AVAILABILITY")
    print(f"{'='*100}")

    # Header
    print(f"{'Model':<40}", end="")
    for city in CITIES:
        print(f"{city:>8}", end="")
    print()
    print("-" * 100)

    # Rows
    for row in rows:
        print(f"{row['Model']:<40}", end="")
        for city in CITIES:
            col = f"{city} ({counts[city]})"
            print(f"{row[col]:>8}", end="")
        print()

def _all_zip_codes():
    """Return deduplicated list of all primary + overflow zip codes."""
    zips = list(CITIES.values())
    for overflow_list in OVERFLOW_ZIPS.values():
        zips.extend(overflow_list)
    return list(dict.fromkeys(zips))


def _zip_to_label(zip_code: str) -> str:
    """Map a zip code back to its city label for raw response tagging."""
    for city, z in CITIES.items():
        if z == zip_code:
            return city
    for parent_city, overflow_list in OVERFLOW_ZIPS.items():
        if zip_code in overflow_list:
            return f"{parent_city}_overflow"
    return "unknown"


def discover_stores(part: str, product_name: str) -> tuple:
    """Query all primary + overflow zips to discover the full store universe.
    Returns (all_stores dict, raw_responses list)."""
    all_stores = {}  # storeNumber -> metadata
    raw_responses = []

    all_zips = []
    for city, z in CITIES.items():
        all_zips.append((city, z))
    for parent_city, overflow_list in OVERFLOW_ZIPS.items():
        for z in overflow_list:
            all_zips.append((f"{parent_city}_overflow", z))

    print(f"Discovering stores across {len(all_zips)} zip codes...")

    def _discover_one(label, z):
        time.sleep(3)
        result = check(part, z)
        return label, z, result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_discover_one, label, z) for label, z in all_zips]
        for future in as_completed(futures):
            label, z, result = future.result()
            stores_found = 0
            if result["raw_response"]:
                body_stores = result["raw_response"].get("body", {}).get("stores", [])
                stores_found = len(body_stores)
                for s in body_stores:
                    sn = s.get("storeNumber", "")
                    if sn and sn not in all_stores:
                        all_stores[sn] = {
                            "storeNumber": sn,
                            "storeName": s.get("storeName", ""),
                            "city": s.get("city", ""),
                            "state": s.get("state", ""),
                            "lat": s.get("storelatitude"),
                            "lng": s.get("storelongitude"),
                            "assigned_city": None,
                        }
                raw_responses.append({
                    "product": product_name, "part": part,
                    "city": label, "zip": z,
                    "response": result["raw_response"]
                })
            print(f"  {label} ({z}): {stores_found} returned, {len(all_stores)} unique total")

    return all_stores, raw_responses


def assign_stores(all_stores: dict) -> dict:
    """Assign each store to its nearest city using haversine.
    Returns {city: [storeNumber, ...]}."""
    city_assignments = {city: [] for city in CITIES}
    excluded = 0

    for sn, store in all_stores.items():
        lat, lng = store.get("lat"), store.get("lng")
        if lat is None or lng is None:
            print(f"  Warning: {sn} ({store['storeName']}) has no coordinates, skipping")
            continue

        best_city, best_dist = None, float("inf")
        for city, (clat, clng) in CITY_COORDS.items():
            d = haversine(lat, lng, clat, clng)
            if d < best_dist:
                best_dist = d
                best_city = city

        if best_dist > MAX_ASSIGNMENT_DISTANCE:
            excluded += 1
            continue

        store["assigned_city"] = best_city
        city_assignments[best_city].append(sn)

    print("\nStore assignments:")
    for city in CITIES:
        print(f"  {city}: {len(city_assignments[city])} stores")
    if excluded:
        print(f"  ({excluded} stores excluded, >{MAX_ASSIGNMENT_DISTANCE}mi from any city)")

    return city_assignments


def save_assignments(all_stores: dict, city_assignments: dict):
    """Save store assignments to disk so we don't re-discover every run."""
    data = {
        "saved_at": datetime.now().isoformat(),
        "stores": {sn: meta for sn, meta in all_stores.items() if meta.get("assigned_city")},
        "city_assignments": {city: list(sns) for city, sns in city_assignments.items()},
    }
    with open(ASSIGNMENTS_CACHE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved store assignments to {ASSIGNMENTS_CACHE.name}")


def load_assignments() -> tuple:
    """Load cached store assignments. Returns (all_stores, city_assignments) or (None, None)."""
    if not ASSIGNMENTS_CACHE.exists():
        return None, None
    try:
        with open(ASSIGNMENTS_CACHE) as f:
            data = json.load(f)
        all_stores = data["stores"]
        city_assignments = data["city_assignments"]
        print(f"Loaded cached store assignments from {ASSIGNMENTS_CACHE.name} (saved {data['saved_at']})")
        for city in CITIES:
            print(f"  {city}: {len(city_assignments.get(city, []))} stores")
        return all_stores, city_assignments
    except Exception as e:
        print(f"Failed to load assignments cache: {e}")
        return None, None


def collect_availability(products: dict, city_assignments: dict) -> tuple:
    """Query availability for all products across all zip codes using batch API.
    One request per zip (all products batched). Returns (rows, counts, raw_responses)."""
    raw_responses = []
    all_zips = _all_zip_codes()
    counts = {city: len(stores) for city, stores in city_assignments.items()}
    parts_list = list(products.values())
    name_by_part = {v: k for k, v in products.items()}

    print(f"Checking availability ({len(all_zips)} batch requests, {len(products)} products each)...")

    # {product_name: {storeNumber: bool}}
    product_store_avail = {name: {} for name in products}

    def _check_zip(z):
        time.sleep(3)
        result = check_batch(parts_list, z)
        return z, result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_check_zip, z) for z in all_zips]
        for future in as_completed(futures):
            z, result = future.result()
            if result["raw_response"]:
                raw_responses.append({
                    "product": "batch",
                    "parts": {name_by_part[p]: p for p in parts_list},
                    "city": _zip_to_label(z), "zip": z,
                    "response": result["raw_response"]
                })
            # Aggregate per-product store availability
            for part, store_avail in result["part_stores"].items():
                name = name_by_part.get(part)
                if not name:
                    continue
                for sn, avail in store_avail.items():
                    # OR logic: available if ANY query says so
                    product_store_avail[name][sn] = (
                        product_store_avail[name].get(sn, False) or avail["available"]
                    )

    # Compute per-city percentages from assigned stores
    rows = []
    for name in products:
        print(f"  {name}")
        store_availability = product_store_avail.get(name, {})
        row = {"Model": name}
        for city in CITIES:
            assigned = city_assignments[city]
            if not assigned:
                row[f"{city} ({counts[city]})"] = "0%"
                continue
            avail_count = sum(1 for sn in assigned if store_availability.get(sn, False))
            pct = 100 * avail_count // len(assigned)
            row[f"{city} ({counts[city]})"] = f"{pct}%"
        rows.append(row)

    print_matrix(rows, counts)
    return rows, counts, raw_responses


def main():
    prev_csv = get_previous_csv()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # === Load cached store assignments (or discover if no cache) ===
    all_stores, city_assignments = load_assignments()
    discovery_raw = []
    if all_stores is None:
        print(f"\n{'='*100}")
        print(f"STORE DISCOVERY (no cache found, running discovery...)")
        print(f"{'='*100}")
        first_part = list(PRODUCTS.values())[0]
        first_name = list(PRODUCTS.keys())[0]
        all_stores, discovery_raw = discover_stores(first_part, first_name)
        city_assignments = assign_stores(all_stores)
        save_assignments(all_stores, city_assignments)

    # # --- To force re-discovery, uncomment below and comment out load_assignments above ---
    # print(f"\n{'='*100}")
    # print(f"STORE DISCOVERY")
    # print(f"{'='*100}")
    # first_part = list(PRODUCTS.values())[0]
    # first_name = list(PRODUCTS.keys())[0]
    # all_stores, discovery_raw = discover_stores(first_part, first_name)
    # city_assignments = assign_stores(all_stores)
    # save_assignments(all_stores, city_assignments)

    counts = {city: len(stores) for city, stores in city_assignments.items()}

    # === Batch all products in one pass ===
    print(f"\n{'='*100}")
    print(f"PRODUCTS ({len(PRODUCTS)}): {', '.join(PRODUCTS.keys())}")
    print(f"{'='*100}")
    all_rows, _, raw1 = collect_availability(PRODUCTS, city_assignments)
    all_raw = discovery_raw + raw1

    # Write CSV
    new_csv = OUT_DIR / f"availability_matrix_{ts}.csv"
    cols = ["Model"] + [f"{c} ({counts[c]})" for c in CITIES]
    with open(new_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n✅ Saved: {new_csv.name}")

    # Write raw JSON with assignment metadata
    raw_dir = OUT_DIR / "raw_api_responses"
    raw_dir.mkdir(exist_ok=True)
    raw_file = raw_dir / f"raw_responses_{ts}.json"
    with open(raw_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "products": dict(PRODUCTS),
            "cities": dict(CITIES),
            "overflow_zips": dict(OVERFLOW_ZIPS),
            "city_assignments": {city: list(stores) for city, stores in city_assignments.items()},
            "store_metadata": {sn: meta for sn, meta in all_stores.items() if meta.get("assigned_city")},
            "responses": all_raw
        }, f, indent=2)

    print(f"📦 Raw API data: {raw_file.name}")

    # Print combined matrix
    print_matrix(all_rows, counts)

    # Diff
    if prev_csv:
        print_diff(prev_csv, str(new_csv))
    else:
        print("\n  No previous CSV to compare.")


if __name__ == "__main__":
    main()
