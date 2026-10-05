import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from pipeline import norm, keys, features, score, build_index, Index, batch_pairs, cheap_filter, candidate_keep, FEATURE_NAMES

class PipelineTests(unittest.TestCase):
    def test_metric_and_singletons(self):
        truth = [{'a', 'b'}, set(), {'c'}, set()]
        result = score(np.array([0, 0, 0, 3]), np.array(['a', 'b', 'x', 'z']), np.ones(4), .5, range(4), truth)
        self.assertAlmostEqual(result['macro_f0.5'], (5/7 + 1 + 0 + 0)/4)
        self.assertEqual(result['correct_singletons'], 1)

    def test_unicode_and_empty_features(self):
        self.assertEqual(norm('École & Co.'), 'ecole and co')
        x = features(('S1-x', '', '', 'Unseen'), ('S2-x', '', '', 'Unseen'))
        self.assertEqual(len(x), 50)
        self.assertEqual(len(FEATURE_NAMES),50)
        self.assertTrue(np.isfinite(x).all())

    def test_filter_consistent_with_training_features(self):
        left=('S1-a','Acme Corporation','123 Main Street','NovelCountry')
        for right in [('S2-a','Entirely different name','123 Main St','NovelCountry'),
                      ('S2-b','Acme Corp','','NovelCountry'),
                      ('S2-c','Unrelated','987 Foreign Road','NovelCountry')]:
            x=np.array(features(left,right),dtype=np.float32)
            self.assertEqual(bool(cheap_filter(left,right)),bool(candidate_keep(x[10],x[17],bool(x[44]+x[45]))))
        self.assertTrue(cheap_filter(left,('S2-d','Different script or alias','123 Main Street','NovelCountry')))

    def test_candidate_inference_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'test').mkdir()
            rows = [('S2-a', 'École Alpha SARL', '123 Rue Victor', 'France'), ('S2-b', 'Ecole Alpha', '123 Rue Victor', 'France')]
            for s, data in [(2, rows), (3, [('S3-c', 'Ecole Alpha', '123 Rue Victor', 'France'), ('S3-d', 'Unrelated', '999 Elsewhere', 'Other')])]:
                with (root/'test'/f'test_source{s}.tsv').open('w', encoding='utf8', newline='') as f:
                    w=csv.writer(f,delimiter='\t'); w.writerow(['entity_id','business_name','business_address','country']); w.writerows(data)
            idx=Index(build_index(root,'test',root/'work'))
            x,q,targets=batch_pairs([('S1-a','Ecole Alpha','123 Rue Victor','France')],idx)
            self.assertEqual(set(targets), {'S2-a','S2-b','S3-c'})
            self.assertEqual(len(targets), len(x))
            self.assertTrue((q==0).all())
            idx.store.close(); idx.file.close()
            for shard in idx.shards+idx.extra_shards:
                if isinstance(shard,np.memmap): shard._mmap.close()

if __name__=='__main__': unittest.main()
