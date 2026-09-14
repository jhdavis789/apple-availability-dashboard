"""Preserve published history when some original iCloud observations are offline."""
import copy
from datetime import datetime
from availability_matrix_csv_rest import PRODUCTS, RETAIL_START, active_products


def merge_data(previous, current):
    result = copy.deepcopy(previous)
    by_time = {s['timestamp']: s for s in previous['snapshots']}
    for snap in current['snapshots']:
        by_time.setdefault(snap['timestamp'], snap)
    result.update({k: v for k, v in current.items() if k not in {'snapshots', 'ebay_prices'}})
    result['snapshots'] = sorted(by_time.values(), key=lambda s: s['timestamp'])
    starts = {name: RETAIL_START[sku] for name, sku in PRODUCTS.items() if sku in RETAIL_START}
    # Announced products have no pickup observations before their retail date.
    for snap in result['snapshots']:
        snap['products'] = [p for p in snap['products']
                            if snap['timestamp'][:10] >= starts.get(p['model'], '0000')]
    result['all_models'] = sorted(set(previous['all_models']) | set(current['all_models']) | set(PRODUCTS))
    result['total_snapshots'] = max(previous.get('total_snapshots', 0), current.get('total_snapshots', 0), len(by_time))
    if current.get('ebay_prices') or previous.get('ebay_prices'):
        series={}
        for dataset in (previous, current):
            for name,points in dataset.get('ebay_prices',{}).get('data',{}).items():
                target=series.setdefault(name,{})
                for point in points:target.setdefault(point['timestamp'],point)
        result['ebay_prices']={'products':sorted(series),'data':{name:[pts[t] for t in sorted(pts)] for name,pts in series.items()}}
    active = active_products()
    today = datetime.now().date().isoformat()
    result['model_lifecycle'] = {}
    for name in result['all_models']:
        dates = [s['timestamp'] for s in result['snapshots'] if any(p['model'] == name for p in s['products'])]
        launch = starts.get(name)
        status = 'upcoming' if launch and launch > today else 'active' if name in active else 'historical'
        result['model_lifecycle'][name] = dict(status=status, retail_start=launch,
            first_observed=min(dates) if dates else None, last_observed=max(dates) if dates else None)
    return result


def merge_map(previous, current):
    products = sorted(set(previous.get('products', [])) | set(current.get('products', [])))
    by_time = {}
    starts = {name: RETAIL_START[sku] for name, sku in PRODUCTS.items() if sku in RETAIL_START}
    for dataset in (previous, current):
        for original in dataset.get('snapshots', []):
            snap = copy.deepcopy(original)
            # The public map uses 't', while the input codec can use timestamp.
            stamp = snap.get('t', snap.get('timestamp'))
            if stamp in by_time:
                continue
            for store, values in snap['a'].items():
                mapping = dict(zip(dataset['products'], values))
                snap['a'][store] = [mapping.get(p) if str(stamp)[:10] >= starts.get(p, '0000') else None for p in products]
            by_time[stamp] = snap
    stores = {s['id']: s for s in previous.get('stores', []) + current.get('stores', [])}
    return dict(stores=list(stores.values()),
                products=products, snapshots=[by_time[t] for t in sorted(by_time)])
