"""Strict bounded-memory validation, including every candidate ID's existence.

Requires output order to match Source 1 input order (our pipeline guarantees this).
The challenge permits other orders; this stricter condition simplifies streaming.
"""
import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path
import numpy as np

def read(path):
    with open(path, encoding='utf8', newline='') as f:
        yield from csv.reader(f, delimiter='\t')

def encode(s):
    b=s.encode('ascii')
    assert len(b)<=32, 'ID exceeds validator fixed-width storage'
    return b

def validate(data, output):
    def targets():
        for source in (2,3):
            it=read(data/f'test_source{source}.tsv'); next(it)
            for row in it:
                assert row[0].startswith(f'S{source}-')
                yield encode(row[0])
    ids=np.fromiter(targets(), dtype='S32'); ids.sort()
    refs=read(data/'test_source1.tsv'); next(refs)
    matches=read(output/'matching_results.tsv'); candidates=read(output/'candidate_pairs.tsv')
    assert next(matches)==['source1_entity_id','matched_entity_ids']
    assert next(candidates)==['source1_entity_id','candidate_entity_ids']
    count=0; pairs=0; links=0; countries=Counter(); pending=[]; s1=[]
    def check_pending():
        if not pending: return
        values=np.array(pending,dtype='S32'); pos=np.searchsorted(ids,values)
        assert np.all(pos<len(ids)), 'Unknown target ID'
        assert np.all(ids[pos]==values), 'Unknown target ID'
        pending.clear()
    for ref, match, cand in itertools.zip_longest(refs,matches,candidates):
        assert ref is not None and match is not None and cand is not None, 'Row count mismatch'
        assert len(match)==len(cand)==2, 'Incorrect number of output columns'
        assert ref[0]==match[0]==cand[0], 'Missing, duplicate, extra or reordered reference entity'
        m=match[1].split(',') if match[1] else []
        c=cand[1].split(',') if cand[1] else []
        assert len(set(m))==len(m) and len(set(c))==len(c), 'Duplicate target within row'
        assert set(m)<=set(c), 'Match was not scored as a candidate'
        for tid in c:
            assert tid.startswith(('S2-','S3-')) and tid.strip()==tid
            pending.append(encode(tid))
        s1.append(encode(ref[0])); count+=1; pairs+=len(c); links+=len(m); countries[ref[3]]+=1
        if count%5000==0: check_pending()
    check_pending()
    assert len(np.unique(np.array(s1,dtype='S32')))==count, 'Duplicated Source 1 IDs'
    result={'status':'PASS','entities':count,'candidate_pairs':pairs,'matched_pairs':links,'countries':dict(countries),'all_target_ids_exist':True,'all_matches_are_candidates':True}
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--test-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(); validate(a.test_dir,a.output_dir)
