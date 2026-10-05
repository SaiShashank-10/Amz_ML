import csv,json,os,tempfile,unittest
from pathlib import Path
from finish_empty import merge

class EmptyRecoveryTests(unittest.TestCase):
    def test_merge_preserves_nonselected_and_exports_scored_candidates(self):
        original=Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp);root=Path(temp);(root/'output_v4').mkdir();work=root/'artifacts_v5';(work/'prediction_chunks').mkdir(parents=True)
                (root/'output_v4/matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a\nS1-b\t\nS1-c\t\n')
                (root/'output_v4/candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-a,S3-a\nS1-b\tS2-b\nS1-c\t\n')
                (work/'prediction_complete.json').write_text(json.dumps({'chunks':1,'entities':1}))
                (work/'prediction_chunks/000000.json').write_text(json.dumps([['S1-b',['S2-b','S3-b'],['S3-b']]]))
                merge(work)
                with (root/'output_v5/matching_results.tsv').open() as f:rows=list(csv.reader(f,delimiter='\t'))
                self.assertEqual(rows[1:],[['S1-a','S2-a'],['S1-b','S3-b'],['S1-c','']])
                self.assertIn('S1-b\tS2-b,S3-b',(root/'output_v5/candidate_pairs.tsv').read_text())
            finally:os.chdir(original)

if __name__=='__main__':unittest.main()
