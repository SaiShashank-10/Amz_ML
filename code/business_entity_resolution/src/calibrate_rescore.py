"""Develop source-specific conservative thresholds on existing v4 positives."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pipeline as base
import match_rescore as rescore

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--assessment-work',type=Path,default=Path('artifacts_v5'));p.add_argument('--evaluate-only',action='store_true');a=p.parse_args()
    work=Path('artifacts_v7');work.mkdir(exist_ok=True);origin=a.assessment_work
    sample=[tuple(r) for r in json.loads((origin/'sample.json').read_text())];byid={r[0]:r for r in sample}
    old=[r for f in sorted((origin/'assessment_chunks').glob('*.json')) for r in json.loads(f.read_text())]
    assert len(old)==len(sample)
    dest=work/('fresh_rescore' if a.evaluate_only else 'development_rescore');dest.mkdir(exist_ok=True)
    tasks=[]
    for i in range(0,len(old),500):
        chunk=old[i:i+500]
        if not (dest/f'{i:06d}.json').exists():tasks.append((i,[byid[r[0]] for r in chunk],[r[1] for r in chunk]))
    with ProcessPoolExecutor(max_workers=a.workers,initializer=rescore.init,initargs=('artifacts/train_index','artifacts_v3/train_reference')) as pool:
        for i,result in pool.map(rescore.batch,tasks):
            (dest/f'{i:06d}.json').write_text(json.dumps(result));base.log(f'Positive-link assessment {i+len(result):,}/{len(old):,}')
    scored=[r for f in sorted(dest.glob('*.json')) for r in json.loads(f.read_text())]
    assert [r[0] for r in scored]==[r[0] for r in old]
    truth=base.load_truth(a.data_dir,set(byid));truths=[truth[r[0]] for r in scored]
    g=[];t=[];prob=[]
    for i,(sid,pairs) in enumerate(scored):
        for tid,pr in pairs:g.append(i);t.append(tid);prob.append(pr)
    g=np.array(g);t=np.array(t);prob=np.array(prob);source2=np.char.startswith(t,'S2-')
    def metric(keep):return base.score(g,t,keep.astype(float),.5,range(len(scored)),truths)
    baseline=metric(np.ones(len(g),dtype=bool))
    if a.evaluate_only:
        policy=json.loads((work/'policy.json').read_text());trials=[]
    else:
        choices=[0.,.1,.2,.3,.4,.5,.575,.65,.75,.85];trials=[]
        for s2 in choices:
            for s3 in choices:
                m=metric(prob>=np.where(source2,s2,s3));trials.append({'source2_threshold':s2,'source3_threshold':s3,'macro_f0.5':m['macro_f0.5']})
        policy=max(trials,key=lambda r:r['macro_f0.5']);(work/'policy.json').write_text(json.dumps(policy,indent=2))
    keep=prob>=np.where(source2,policy['source2_threshold'],policy['source3_threshold']);proposed=metric(keep)
    differences=[]
    for (sid,pairs),true in zip(scored,truths):
        o={tid for tid,p in pairs};n={tid for tid,p in pairs if p>=(policy['source2_threshold'] if tid.startswith('S2-') else policy['source3_threshold'])}
        def f(pred):return 1.25*len(pred&true)/(.25*len(true)+len(pred)) if true else float(not pred)
        differences.append(f(n)-f(o))
    d=np.array(differences);mean=float(d.mean());margin=float(1.96*d.std(ddof=1)/np.sqrt(len(d)))
    report={'baseline':baseline,'proposed':proposed,'policy':policy,'paired_improvement':{'mean':mean,'approximate_95pct_ci':[mean-margin,mean+margin]},'trials':trials,'note':'Fresh validation with frozen thresholds.' if a.evaluate_only else 'Development thresholds; not independent assessment.'}
    name='metrics.json' if a.evaluate_only else 'development.json';(work/name).write_text(json.dumps(report,indent=2));base.log(json.dumps({k:v for k,v in report.items() if k!='trials'}))

if __name__=='__main__':main()
