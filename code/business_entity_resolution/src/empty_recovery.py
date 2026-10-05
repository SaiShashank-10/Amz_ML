"""Fresh assessment of an empty-prediction recovery branch using frozen models."""
import argparse,json,itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import lightgbm as lgb
import pipeline as base
import enhanced_pipeline as v3
import phonetic_pipeline as v4
import phonetic_index as pi

def init():
    global OLD,NEW,GATE,OLD_TH,NEW_TH
    v3.init_worker('artifacts/train_index','artifacts_v3/train_reference')
    v4.init_worker('artifacts/train_index','artifacts_v3/train_reference')
    OLD=lgb.Booster(model_file='artifacts_v3/model.txt');NEW=lgb.Booster(model_file='artifacts_v4/model.txt')
    OLD_TH=json.loads(Path('artifacts_v3/metrics.json').read_text())['threshold']
    NEW_TH=json.loads(Path('artifacts_v4/metrics.json').read_text())['threshold']
    GATE=pi.Gate(Path('artifacts/train_index'))

def predict(rows):
    x,g,t=v3.make_pairs(rows);p=OLD.predict(x,num_threads=1) if len(x) else []
    old=[set() for _ in rows]
    for gi,ti,prob in zip(g,t,p):
        if prob>=OLD_TH:old[int(gi)].add(str(ti))
    flags=GATE.select(rows,old);selected=[i for i in range(len(rows)) if flags[i] or not old[i]]
    new={i:set() for i in selected}
    if selected:
        x,g,t=v4.make_pairs([rows[i] for i in selected]);p=NEW.predict(x,num_threads=1) if len(x) else []
        for gi,ti,prob in zip(g,t,p):
            if prob>=NEW_TH:new[selected[int(gi)]].add(str(ti))
    return [(row[0],sorted(new[i] if flags[i] else old[i]),sorted(new[i] if i in new else old[i]),bool(not flags[i] and not old[i])) for i,row in enumerate(rows)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--sample-size',type=int,default=30000)
    p.add_argument('--work-dir',type=Path,default=Path('artifacts_v5'));p.add_argument('--exclude-work',type=Path,nargs='*',default=[]);a=p.parse_args()
    work=a.work_dir;work.mkdir(exist_ok=True,parents=True)
    sample_path=work/'sample.json'
    sample=[tuple(r) for r in json.loads(sample_path.read_text())] if sample_path.exists() else []
    if len(sample)<a.sample_size:
        excluded={r[0] for d in ['artifacts','artifacts_v3','artifacts_v4'] for r in json.loads((Path(d)/'sample.json').read_text())}
        excluded.update(r[0] for d in a.exclude_work for r in json.loads((d/'sample.json').read_text()))
        excluded.update(r[0] for r in sample)
        additional=[];seen=0;needed=a.sample_size-len(sample)
        rng=np.random.default_rng(20260929 if not sample else 20260930)
        for row in base.records(a.data_dir/'train/train_source1.tsv'):
            if row[0] in excluded:continue
            if seen<needed:additional.append(row)
            else:
                j=int(rng.integers(seen+1))
                if j<needed:additional[j]=row
            seen+=1
        assert not ({r[0] for r in additional}&excluded)
        sample.extend(additional)
        sample_path.write_text(json.dumps(sample));base.log(f'Fresh {len(sample):,}-reference assessment sample saved; excludes all previous supervised samples')
    truth=base.load_truth(a.data_dir,{r[0] for r in sample})
    chunks=work/'assessment_chunks';chunks.mkdir(exist_ok=True)
    tasks=[(i,sample[i:i+100]) for i in range(0,len(sample),100) if not (chunks/f'{i:06d}.json').exists()]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init) as pool:
        for (i,_),result in zip(tasks,pool.map(predict,[rows for i,rows in tasks])):
            (chunks/f'{i:06d}.json').write_text(json.dumps(result))
            if i%1000==0:base.log(f'Fresh paired assessment {i+100:,}/{len(sample):,} complete')
    results=[r for f in sorted(chunks.glob('*.json')) for r in json.loads(f.read_text())]
    def score(pred,true):return 1.25*len(pred&true)/(.25*len(true)+len(pred)) if true else float(not pred)
    old=[];new=[];old_groups=[];new_groups=[];old_targets=[];new_targets=[]
    for i,(sid,o,n,flag) in enumerate(results):
        old.append(score(set(o),truth[sid]));new.append(score(set(n),truth[sid]));old_groups.extend([i]*len(o));new_groups.extend([i]*len(n));old_targets.extend(o);new_targets.extend(n)
    truths=[truth[r[0]] for r in results];d=np.asarray(new)-np.asarray(old);mean=float(d.mean());margin=float(1.96*d.std(ddof=1)/np.sqrt(len(d)))
    metrics={'version':5,'seed':20260929,'extension_seed':20260930,'entities':len(results),'prior_sample_overlap':0,'newly_processed_entities':sum(r[3] for r in results),
        'v4_same_fresh_holdout':base.score(np.array(old_groups),np.array(old_targets),np.ones(len(old_targets)),.5,range(len(results)),truths),
        'proposed_same_fresh_holdout':base.score(np.array(new_groups),np.array(new_targets),np.ones(len(new_targets)),.5,range(len(results)),truths),
        'paired_improvement':{'mean':mean,'approximate_95pct_ci':[mean-margin,mean+margin]},'model_policy':'Frozen v3/v4 models and thresholds; v4 additional branch for previously ungated empty v3 predictions; no new fitting or threshold tuning.'}
    (work/'metrics.json').write_text(json.dumps(metrics,indent=2));base.log(json.dumps(metrics))

if __name__=='__main__':main()
