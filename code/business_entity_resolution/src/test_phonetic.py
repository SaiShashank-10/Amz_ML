import unittest
import csv,tempfile
from pathlib import Path
import numpy as np
import phonetic_features as p
import phonetic_pipeline as v4
import phonetic_index as pi
from test_enhanced import ContextStub

class PhoneticTests(unittest.TestCase):
    def test_cross_script_and_legal_suffix(self):
        left=('S1-x','United Consultants Private Limited','12 Park Road','India')
        right=('S2-y','\u092f\u0942\u0928\u093e\u0907\u091f\u0947\u0921 \u0915\u0902\u0938\u0932\u094d\u091f\u0947\u0902\u091f\u094d\u0938 \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921','12 Park Rd','India')
        self.assertEqual(p.forms(left)[2],p.forms(right)[2])
        self.assertTrue(p.keys(left)&p.keys(right))
        f=v4.features(left,right,[],ContextStub())
        self.assertEqual(len(f),154);self.assertTrue(np.isfinite(f).all())

    def test_preserve_name_and_numbers(self):
        left=('S1-x','Royal Energy','1040 Grove Court Unit 104','Unknown')
        right=('S2-x','Royal Energy','1042 Grove Court Unit 104','Unknown')
        self.assertIn('rl',p.forms(left)[1].split())
        f=p.features(left,right)
        self.assertEqual(f[20],0);self.assertEqual(f[21],1)

    def test_unicode_accent_and_candra(self):
        self.assertEqual(p.romanize('Soci\u00e9t\u00e9'), 'societe')
        self.assertNotIn('candra',p.romanize('\u092b\u0949\u0930\u094d\u091a\u0942\u0928'))

    def test_gate_uses_predictions_not_labels(self):
        class IndexStub:
            def extra_candidates(self,rows):return [[1] for _ in rows]
        gate=pi.Gate.__new__(pi.Gate);gate.index=IndexStub()
        gate.by_offset={1:('S2-native','ntd knsltnts')};gate.by_id={'S2-native':'ntd knsltnts'}
        row=('S1-test','United Consultants','','AnyCountry')
        self.assertEqual(gate.select([row],[set()]),[True])
        self.assertEqual(gate.select([row],[{'S2-native'}]),[False])

    def test_real_phonetic_index_retrieval_and_country_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'test').mkdir()
            native='\u092f\u0942\u0928\u093e\u0907\u091f\u0947\u0921 \u0915\u0902\u0938\u0932\u094d\u091f\u0947\u0902\u091f\u094d\u0938'
            for source,country in [(2,'Novel'),(3,'Different')]:
                with (root/'test'/f'test_source{source}.tsv').open('w',encoding='utf8',newline='') as f:
                    w=csv.writer(f,delimiter='\t');w.writerow(['entity_id','business_name','business_address','country'])
                    w.writerow([f'S{source}-native',native,'',country])
            path=p.base.build_index(root,'test',root/'work');pi.build(path);pi.build_native(path)
            idx=pi.Index(path);row=('S1-a','United Consultants','','Novel')
            found={idx.get(off)[0] for off in idx.candidates([row])[0]}
            self.assertEqual(found,{'S2-native'})
            idx.store.close();idx.file.close()
            for shard in idx.shards+idx.extra_shards+idx.phon_shards:
                if isinstance(shard,np.memmap):shard._mmap.close()

if __name__=='__main__':unittest.main()
