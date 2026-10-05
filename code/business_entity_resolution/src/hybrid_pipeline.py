"""Deadline-aware v4 correction cascade, preserving the completed v3 submission."""
import argparse
import csv
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import pipeline as base
import phonetic_index
import phonetic_pipeline as v4

def select(data,work):
    path=work/'gate.json';index=Path('artifacts/test_index')
    if path.exists():return json.loads(path.read_text())
    gate=phonetic_index.Gate(index);refs=base.records(data/'test/test_source1.tsv')
    selected=[];count=0
    number=0
    while rows:=list(itertools.islice(refs,200)):
        prior=json.loads((Path('artifacts_v3/prediction_chunks')/f'{number:06d}.json').read_text())
        assert [r[0] for r in rows]==[r[0] for r in prior]
        flags=gate.select(rows,[set(r[2]) for r in prior])
        selected.extend(count+i for i,flag in enumerate(flags) if flag);count+=len(rows)
        if number%100==0:base.log(f'Correction gate: {count:,} checked; {len(selected):,} selected')
        number+=1
    assert count==json.loads(Path('artifacts_v3/prediction_stats.json').read_text())['entities']
    result={'entities':count,'selected':selected,'selected_count':len(selected),'rule_version':1,
        'baseline_model_sha256':hashlib.sha256(Path('artifacts_v3/model.txt').read_bytes()).hexdigest()}
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(path)
    base.log(f'Correction gate complete: {len(selected):,}/{count:,} references')
    return result

def predict(data,work,workers):
    metrics=json.loads((work/'metrics.json').read_text())
    assert metrics['hybrid_held_out']['macro_f0.5']>metrics['original_on_same_holdout']['macro_f0.5'], 'Cascade did not improve offline; keep v3 submission'
    assert metrics['hybrid_paired_improvement']['approximate_95pct_ci'][0]>0, 'Offline gain is inconclusive; review before spending an upload'
    assert (work/'model.txt').exists() and 'Enhanced final model saved' in (work/'train.log').read_text(encoding='utf-8-sig',errors='replace')
    gate=select(data,work);selected=set(gate['selected']);threshold=metrics['threshold']
    chunks=work/'hybrid_chunks';chunks.mkdir(exist_ok=True)
    manifest=chunks/'manifest.json';signature={'model':hashlib.sha256((work/'model.txt').read_bytes()).hexdigest(),'gate':hashlib.sha256((work/'gate.json').read_bytes()).hexdigest(),'threshold':threshold}
    if manifest.exists():assert json.loads(manifest.read_text())==signature
    else:manifest.write_text(json.dumps(signature))
    rows=(row for i,row in enumerate(base.records(data/'test/test_source1.tsv')) if i in selected)
    total=0;number=0
    with ProcessPoolExecutor(max_workers=workers,initializer=v4.init_predict,initargs=('artifacts/test_index','artifacts_v3/test_reference',str(work/'model.txt'),threshold)) as pool:
        pending=set()
        while batch:=list(itertools.islice(rows,200)):
            path=chunks/f'{number:06d}.json';number+=1
            if path.exists():total+=len(batch)
            else:pending.add(pool.submit(v4.predict_chunk,(str(path),batch)))
            if len(pending)>=workers*2:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:total+=future.result()
                if total%1000==0:base.log(f'Correction inference: {total:,}/{len(selected):,} complete')
        for future in pending:total+=future.result()
    assert total==len(selected)
    (work/'hybrid_complete.json').write_text(json.dumps({'chunks':number,'entities':total}))

def merge(work,output):
    done=json.loads((work/'hybrid_complete.json').read_text())
    def changes():
        for i in range(done['chunks']):yield from json.loads((work/'hybrid_chunks'/f'{i:06d}.json').read_text())
    iterator=iter(changes());next_change=next(iterator,None);output.mkdir(exist_ok=True)
    stats={'entities':0,'candidates':0,'matches':0,'singletons':0,'corrected_entities':0}
    with (output/'matching_results.tsv').open('w',encoding='utf8',newline='') as fm,(output/'candidate_pairs.tsv').open('w',encoding='utf8',newline='') as fc:
        wm=csv.writer(fm,delimiter='\t',lineterminator='\n');wc=csv.writer(fc,delimiter='\t',lineterminator='\n')
        wm.writerow(['source1_entity_id','matched_entity_ids']);wc.writerow(['source1_entity_id','candidate_entity_ids'])
        for path in sorted(Path('artifacts_v3/prediction_chunks').glob('[0-9][0-9][0-9][0-9][0-9][0-9].json')):
            for sid,c,m in json.loads(path.read_text()):
                if next_change is not None and sid==next_change[0]:
                    sid,c,m=next_change;next_change=next(iterator,None);stats['corrected_entities']+=1
                assert set(m)<=set(c)
                wm.writerow([sid,','.join(m)]);wc.writerow([sid,','.join(c)])
                stats['entities']+=1;stats['candidates']+=len(c);stats['matches']+=len(m);stats['singletons']+=not m
    assert next_change is None and stats['corrected_entities']==done['entities']
    (work/'prediction_stats.json').write_text(json.dumps(stats,indent=2));base.log('Hybrid output complete '+json.dumps(stats))

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--stage',choices=['select','predict','merge','all'],default='all');a=p.parse_args();work=Path('artifacts_v4')
    if a.stage in ['select','all']:select(a.data_dir,work)
    if a.stage in ['predict','all']:predict(a.data_dir,work,a.workers)
    if a.stage in ['merge','all']:merge(work,Path('output_v4'))

if __name__=='__main__':main()
