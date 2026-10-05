"""Apply the prevalidated empty-recovery branch, then validate/package v5."""
import argparse,csv,hashlib,itertools,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import pipeline as base
import phonetic_pipeline as v4

def gate(work):
    path=work/'gate.json'
    if path.exists():return json.loads(path.read_text())
    prior=set(json.loads(Path('artifacts_v4/gate.json').read_text())['selected']);selected=[]
    with Path('output_v4/matching_results.tsv').open(encoding='utf8',newline='') as f:
        for i,row in enumerate(csv.DictReader(f,delimiter='\t')):
            if not row['matched_entity_ids'] and i not in prior:selected.append(i)
    result={'selected':selected,'selected_count':len(selected),'entities':i+1};path.write_text(json.dumps(result));return result

def predict(data,work,workers):
    metrics=json.loads((work/'metrics.json').read_text())
    assert metrics['paired_improvement']['approximate_95pct_ci'][0]>0,'STOP: fresh evidence does not support an improvement; retain the 0.962 v4 file'
    choice=gate(work);selected=set(choice['selected']);threshold=json.loads(Path('artifacts_v4/metrics.json').read_text())['threshold']
    chunks=work/'prediction_chunks';chunks.mkdir(exist_ok=True)
    signature={'model':hashlib.sha256(Path('artifacts_v4/model.txt').read_bytes()).hexdigest(),'gate':hashlib.sha256((work/'gate.json').read_bytes()).hexdigest(),'threshold':threshold}
    manifest=chunks/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==signature
    else:manifest.write_text(json.dumps(signature))
    rows=(r for i,r in enumerate(base.records(data/'test/test_source1.tsv')) if i in selected);number=0;total=0
    with ProcessPoolExecutor(max_workers=workers,initializer=v4.init_predict,initargs=('artifacts/test_index','artifacts_v3/test_reference','artifacts_v4/model.txt',threshold)) as pool:
        pending=set()
        while batch:=list(itertools.islice(rows,200)):
            path=chunks/f'{number:06d}.json';number+=1
            if path.exists():total+=len(batch)
            else:pending.add(pool.submit(v4.predict_chunk,(str(path),batch)))
            if len(pending)>=workers*2:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for f in done:total+=f.result()
                if total%1000==0:base.log(f'Empty recovery: {total:,}/{len(selected):,} complete')
        for f in pending:total+=f.result()
    assert total==len(selected)
    (work/'prediction_complete.json').write_text(json.dumps({'entities':total,'chunks':number}))

def merge(work):
    done=json.loads((work/'prediction_complete.json').read_text());output=Path('output_v5');output.mkdir(exist_ok=True)
    def changes():
        for i in range(done['chunks']):yield from json.loads((work/'prediction_chunks'/f'{i:06d}.json').read_text())
    it=iter(changes());changed=next(it,None);stats={'entities':0,'candidates':0,'matches':0,'singletons':0,'reprocessed_entities':0,'new_nonempty_entities':0}
    with Path('output_v4/matching_results.tsv').open(encoding='utf8',newline='') as oldm,Path('output_v4/candidate_pairs.tsv').open(encoding='utf8',newline='') as oldc,(output/'matching_results.tsv').open('w',encoding='utf8',newline='') as fm,(output/'candidate_pairs.tsv').open('w',encoding='utf8',newline='') as fc:
        mr=csv.reader(oldm,delimiter='\t');cr=csv.reader(oldc,delimiter='\t');wm=csv.writer(fm,delimiter='\t',lineterminator='\n');wc=csv.writer(fc,delimiter='\t',lineterminator='\n')
        wm.writerow(next(mr));wc.writerow(next(cr))
        for mrow,crow in itertools.zip_longest(mr,cr):
            assert mrow is not None and crow is not None and mrow[0]==crow[0]
            sid=mrow[0];m=mrow[1].split(',') if mrow[1] else [];c=crow[1].split(',') if crow[1] else []
            if changed is not None and changed[0]==sid:
                assert not m
                _,c,m=changed;changed=next(it,None);stats['reprocessed_entities']+=1;stats['new_nonempty_entities']+=bool(m)
            assert set(m)<=set(c)
            wm.writerow([sid,','.join(m)]);wc.writerow([sid,','.join(c)])
            stats['entities']+=1;stats['candidates']+=len(c);stats['matches']+=len(m);stats['singletons']+=not m
    assert changed is None and stats['reprocessed_entities']==done['entities']
    (work/'prediction_stats.json').write_text(json.dumps(stats,indent=2));base.log(json.dumps(stats))

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    work=Path('artifacts_v5');data=a.data_dir.resolve();root=Path.cwd();src=root/'code/business_entity_resolution/src'
    predict(data,work,a.workers);merge(work)
    def run(script,args,log,cwd=root):
        print(f'Running {script}; log: {log}',flush=True)
        with (work/log).open('w',encoding='utf8') as f:subprocess.run([sys.executable,'-u',str(src/script),*map(str,args)],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
    run('validate_streaming.py',['--test-dir',data/'test','--output-dir',root/'output_v5'],'validation_streaming.log')
    run('validate_submission.py',['--test-dir',data/'test','--matching',root/'output_v5/matching_results.tsv','--check-ids'],'validation_official.log',src.parent)
    run('package_empty.py',[],'package.log')
    print('COMPLETE: v5 validated and packaged.',flush=True)

if __name__=='__main__':main()
