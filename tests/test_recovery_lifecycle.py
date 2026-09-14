import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from availability_matrix_csv_rest import active_products
from recovery_merge import merge_data, merge_map

class RecoveryTests(unittest.TestCase):
    def test_launch_boundary(self):
        self.assertNotIn('MJQ34LL/A', active_products('2026-09-17').values())
        self.assertIn('MJQ34LL/A', active_products('2026-09-18').values())
        self.assertNotIn('MK1E4LL/A', active_products('2026-10-22').values())
        self.assertIn('MK1E4LL/A', active_products('2026-10-23').values())
        self.assertNotIn('MG7K4LL/A', active_products('2026-09-18').values())

    def test_history_and_real_zero_preserved(self):
        old = {'all_models': ['Old'], 'snapshots': [dict(timestamp='2026-09-01T00:00:00', products=[dict(model='Old', values={'NYC':0})])]}
        new = {'all_models': ['New'], 'snapshots': [dict(timestamp='2026-09-14T00:00:00', products=[dict(model='New', values={'NYC':50})])]}
        merged = merge_data(old,new)
        self.assertEqual(merged['snapshots'][0],old['snapshots'][0])
        self.assertEqual(len(merged['snapshots']),2)
        self.assertNotIn('Old',[p['model'] for p in merged['snapshots'][-1]['products']])

    def test_store_map_preserves_indices_and_missing(self):
        old=dict(stores=[dict(id='R1')],products=['Old'],snapshots=[dict(t='2026-09-01',a={'R1':[0]})])
        new=dict(stores=[dict(id='R1')],products=['New'],snapshots=[dict(t='2026-09-14',a={'R1':[1]})])
        merged=merge_map(old,new)
        self.assertEqual(merged['products'],['New','Old'])
        self.assertEqual(merged['snapshots'][0]['a']['R1'],[None,0])
        self.assertEqual(merged['snapshots'][1]['a']['R1'],[1,None])
