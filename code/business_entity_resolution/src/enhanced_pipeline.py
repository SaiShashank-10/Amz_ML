"""Offline ER v3: broader retrieval, rarity, ambiguity and numeric context.

The original pipeline remains unchanged. No business data is downloaded.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import mmap
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
from pathlib import Path
import lightgbm as lgb
import numpy as np
import xxhash
from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split
import pipeline as base

VERSION=3
SEED=20260927
FREQ_DT=np.dtype([('key','<u8'),('count','<u4')])

def hkey(country,kind,text):
    return xxhash.xxh3_64_intdigest(country+'|'+kind+'|'+text)

@lru_cache(maxsize=30000)
def extended(row):
    n,a,nt,at,nums,c=base.prep(row)
    name=re.sub(r'\b(?:l\W*l\W*c|l\W*l\W*p|p\W*v\W*t)\b',' ',row[1].casefold())
    name=re.sub(r'https?://|www\.|\.(?:com|in|org|net)\b',' ',name)
    tokens=[t for t in base.norm(name).split() if t not in base.LEGAL and t not in {'mr','mrs','ms','shri','sri','smt'} and not (t.isdigit() and len(t)>=7)]
    core=' '.join(tokens)
    compact=''.join(tokens)
    acronym=''.join(t[0] for t in tokens if t)
    words=frozenset(t for t in at if not t.isdigit())
    alpha=any(ord(ch)>127 for ch in n)
    return core,compact,acronym,words,alpha

def build_reference(data,split,work):
    dest=work/f'{split}_reference'; dest.mkdir(exist_ok=True,parents=True)
    input_path=data/split/f'{split}_source1.tsv'
    fp=base.fingerprint([input_path]); done=dest/'complete.json'
    if done.exists():
        assert json.loads(done.read_text())['input']==fp
        return dest
    extra=dest/'extra';extra.mkdir(exist_ok=True)
    handles=[open((dest if i<256 else extra)/f'{i%256:02x}.bin','wb') for i in range(512)]
    buffers=[[] for _ in handles]; freq=Counter(); countries=Counter()
    def flush():
        for f,b in zip(handles,buffers):
            if b:np.asarray(b,dtype=base.DT).tofile(f);b.clear()
    with open(dest/'records.jsonl','wb') as store:
        for count,row in enumerate(base.records(input_path),1):
            off=store.tell();store.write((json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode())
            for k in base.keys(row):buffers[k&255].append((k,off))
            for k in base.extra_keys(row):buffers[256+(k&255)].append((k,off))
            n,a,nt,at,nums,c=base.prep(row); core=extended(row)[0]
            countries[c]+=1
            for kind,value in [('N',n),('C',' '.join(nt)),('E',core),('A',' '.join(sorted(at)))]:
                if value:freq[hkey(c,kind,value)]+=1
            for kind,tokens in [('n',set(nt)),('a',at)]:
                freq.update(hkey(c,kind,t) for t in tokens)
            if count%10000==0:flush()
            if count%250000==0:base.log(f'{split}: reference context {count:,}')
    flush()
    for f in handles:f.close()
    for i in range(512):
        p=(dest if i<256 else extra)/f'{i%256:02x}.bin'
        arr=np.fromfile(p,dtype=base.DT);arr.sort(order=['key','offset']);arr.tofile(p)
    arr=np.array(list(freq.items()),dtype=FREQ_DT);arr.sort(order='key');arr.tofile(dest/'frequency.bin')
    done.write_text(json.dumps({'input':fp,'version':VERSION,'countries':countries,'records':count}))
    base.log(f'{split}: reference context ready')
    return dest

class Context:
    def __init__(self,path):
        self.index=base.Index(path,max_block=500,max_candidates=120)
        self.freq=np.memmap(path/'frequency.bin',dtype=FREQ_DT,mode='r')
        # The packed 12-byte structure has an unaligned 8-byte key field.
        # Materialize aligned keys once rather than letting searchsorted copy
        # the entire key column for every single frequency lookup.
        self.frequency_keys=np.ascontiguousarray(self.freq['key'])
        self.pop=json.loads((path/'complete.json').read_text())['countries']

    @lru_cache(maxsize=100000)
    def count(self,c,kind,text):
        if not text:return 0
        key=np.uint64(hkey(c,kind,text));i=int(np.searchsorted(self.frequency_keys,key))
        return int(self.freq[i]['count']) if i<len(self.freq) and self.freq[i]['key']==key else 0

    def weight(self,c,kind,t):
        return math.log1p(self.pop.get(c,1)/(1+self.count(c,kind,t)))

def build_prepared_cache(base_work,work):
    """Optional acceleration only: precompute deterministic target text transforms."""
    source=base_work/'test_index/records.jsonl';dest=work/'prepared_test';dest.mkdir(exist_ok=True)
    manifest=dest/'complete.json';fp=base.fingerprint([source])
    if manifest.exists():
        assert json.loads(manifest.read_text())['input']==fp
        return
    offsets=[];positions=[];count=0
    with source.open('rb') as src,(dest/'records.jsonl').open('wb') as out,(dest/'offsets.bin').open('wb') as fo,(dest/'positions.bin').open('wb') as fpout:
        while True:
            offset=src.tell();line=src.readline()
            if not line:break
            row=tuple(json.loads(line));p=base.prep(row);e=extended(row)
            serial=[row[0],[p[0],p[1],p[2],sorted(p[3]),p[4],p[5]],[e[0],e[1],e[2],sorted(e[3]),e[4]]]
            offsets.append(offset);positions.append(out.tell());out.write((json.dumps(serial,ensure_ascii=False,separators=(',',':'))+'\n').encode())
            count+=1
            if count%10000==0:
                np.asarray(offsets,dtype='<u8').tofile(fo);np.asarray(positions,dtype='<u8').tofile(fpout);offsets.clear();positions.clear()
            if count%500000==0:base.log(f'Precomputed test text {count:,} records')
        np.asarray(offsets,dtype='<u8').tofile(fo);np.asarray(positions,dtype='<u8').tofile(fpout)
    manifest.write_text(json.dumps({'input':fp,'records':count,'version':VERSION}))

class PreparedRow(tuple):
    def __new__(cls,ident,p,e):
        obj=super().__new__(cls,(ident,p[0],p[1],p[5]))
        obj.prepared=(p[0],p[1],tuple(p[2]),frozenset(p[3]),tuple(p[4]),p[5])
        obj.extended=(e[0],e[1],e[2],frozenset(e[3]),e[4])
        return obj

class PreparedIndex(base.Index):
    def __init__(self,index,cache):
        super().__init__(index,max_block=350,max_candidates=500)
        self.offsets=np.memmap(cache/'offsets.bin',dtype='<u8',mode='r')
        self.positions=np.memmap(cache/'positions.bin',dtype='<u8',mode='r')
        self.cache_file=(cache/'records.jsonl').open('rb')
        self.cache=mmap.mmap(self.cache_file.fileno(),0,access=mmap.ACCESS_READ)
    def get(self,offset):
        i=int(np.searchsorted(self.offsets,np.uint64(offset)))
        assert self.offsets[i]==offset
        start=int(self.positions[i]);end=self.cache.find(b'\n',start)
        ident,p,e=json.loads(self.cache[start:end]);return PreparedRow(ident,p,e)

def use_prepared_cache(index,cache):
    global IDX,extended
    IDX=PreparedIndex(Path(index),Path(cache))
    original_prep=base.prep;original_extended=extended
    base.prep=lambda row:row.prepared if isinstance(row,PreparedRow) else original_prep(row)
    extended=lambda row:row.extended if isinstance(row,PreparedRow) else original_extended(row)

def weighted(left,right,c,kind,ctx):
    left,right=set(left),set(right);union=left|right;common=left&right
    weights={t:ctx.weight(c,kind,t) for t in union}
    sl=sum(weights[t] for t in left);sr=sum(weights[t] for t in right);si=sum(weights[t] for t in common)
    return [si/max(1e-6,sl+sr-si),si/max(1e-6,sl),si/max(1e-6,sr),
        max((weights[t] for t in left-right),default=0),max((weights[t] for t in right-left),default=0),
        max((weights[t] for t in common),default=0)]

def similarity(left,right):
    _,a,nt,_,nums,_=base.prep(left);_,b,mt,_,digs,_=base.prep(right)
    n=fuzz.ratio(extended(left)[1],extended(right)[1])/100
    ad=fuzz.token_set_ratio(a,b)/100
    num=len(set(nums)&set(digs))/max(1,len(set(nums)|set(digs)))
    return n,ad,.45*n+.45*ad+.10*num

def features(left,right,neighbors,ctx):
    vals=base.features(left,right)
    n,a,nt,at,nums,c=base.prep(left);m,b,mt,bt,digs,_=base.prep(right)
    nc,ncompact,na,nwords,nscr=extended(left);mc,mcompact,ma,mwords,mscr=extended(right)
    for x,y in [(nc,mc),(ncompact,mcompact),(' '.join(sorted(nwords)),' '.join(sorted(mwords)))]:
        vals.extend(f(x,y)/100 for f in (fuzz.ratio,fuzz.partial_ratio,fuzz.token_sort_ratio,fuzz.token_set_ratio))
    vals.extend([float(bool(ncompact) and ncompact==mcompact),float(bool(na) and na==mcompact),
        float(bool(ma) and ma==ncompact),float(nscr),float(mscr),float(nscr!=mscr)])
    vals+=weighted(nt,mt,c,'n',ctx)+weighted(at,bt,c,'a',ctx)
    for kind,l,r in [('N',n,m),('C',' '.join(nt),' '.join(mt)),('E',nc,mc),('A',' '.join(sorted(at)),' '.join(sorted(bt)))]:
        lc,rc=ctx.count(c,kind,l),ctx.count(c,kind,r)
        vals.extend([math.log1p(lc),math.log1p(rc),float(lc==1),float(rc==1)])
    for k in [1,2,3,4,5]:
        ls={v for v in nums if len(v)>=k};rs={v for v in digs if len(v)>=k}
        vals.extend([len(ls-rs),len(rs-ls),len(ls&rs)])
    numeric=[max((fuzz.ratio(t,u)/100 for u in digs),default=0) for t in nums]
    vals.extend([min(numeric,default=0),sum(numeric)/max(1,len(numeric))])
    comp=[similarity(other,right) for other in neighbors]
    best=np.max(comp,axis=0).tolist() if comp else [0.,0.,0.]
    own=similarity(left,right)
    vals+=best+[own[i]-best[i] for i in range(3)]+[len(neighbors)]
    return vals

def init_worker(index_path,reference_path):
    global IDX,CTX
    base.norm=lru_cache(maxsize=30000)(base.norm.__wrapped__)
    base.prep=lru_cache(maxsize=20000)(base.prep.__wrapped__)
    IDX=base.Index(Path(index_path),max_block=350,max_candidates=500)
    CTX=Context(Path(reference_path))

def make_pairs(rows):
    initial=IDX.candidates(rows); near=CTX.index.candidates(rows)
    pools=[];neighbors=[];seed_rows=[];seed_owner=[]
    for i,(row,offsets,refs) in enumerate(zip(rows,initial,near)):
        pool={off:IDX.get(off) for off in offsets}
        others=[CTX.index.get(off) for off in refs]
        others=[r for r in others if r[0]!=row[0]]
        others.sort(key=lambda r:similarity(row,r)[2],reverse=True)
        neighbors.append(others[:8]);pools.append(pool)
        seeds=[]
        original=base.prep(row);original_signature=(original[2],tuple(sorted(original[3])))
        for off,target in pool.items():
            n,a,combined=similarity(row,target)
            unique_address=(a>=.99 and CTX.count(original[5],'A',' '.join(sorted(original[3])))==1)
            if (n>=.88 and a>=.82) or (n>=.65 and a>=.98) or unique_address:seeds.append((combined,off,target))
        seen={original_signature};added=0
        for _,_,target in sorted(seeds,reverse=True):
            pp=base.prep(target);signature=(pp[2],tuple(sorted(pp[3])))
            if signature in seen:continue
            seen.add(signature);seed_rows.append(target);seed_owner.append(i);added+=1
            if added==3:break
    if seed_rows:
        for owner,offsets in zip(seed_owner,IDX.candidates(seed_rows)):
            pool=pools[owner]
            for off in offsets:
                if off not in pool:pool[off]=IDX.get(off)
    xs=[];groups=[];targets=[]
    for i,(row,pool) in enumerate(zip(rows,pools)):
        eligible=[]
        for off,target in pool.items():
            _,a,nt,*_=base.prep(row);_,b,mt,*_=base.prep(target)
            ns=fuzz.token_set_ratio(' '.join(nt),' '.join(mt))/100
            ads=fuzz.token_set_ratio(a,b)/100
            compact=fuzz.ratio(extended(row)[1],extended(target)[1])/100
            if base.candidate_keep(ns,ads,not a or not b) or compact>=.85:
                eligible.append((off,target,compact,ads,.45*compact+.55*ads))
        # Union of rankings protects different kinds of noisy matches.
        selected=set()
        for col in (2,3,4):
            selected.update(t[0] for t in sorted(eligible,key=lambda t:(-t[col],t[0]))[:45])
        for off,target,*_ in sorted(eligible):
            if off not in selected:continue
            xs.append(features(row,target,neighbors[i],CTX));groups.append(i);targets.append(target[0])
    width=120
    return np.asarray(xs,dtype=np.float32).reshape(-1,width),np.asarray(groups,dtype=np.int32),targets

def extract(task):
    start,rows,truth=task;x,g,t=make_pairs(rows)
    y=np.array([tid in truth[rows[int(i)][0]] for i,tid in zip(g,t)],dtype=np.uint8)
    return start,x,y,g+start,t

def prepare(data,base_work,work,size,workers):
    work.mkdir(exist_ok=True,parents=True)
    if (work/'features_complete.json').exists():return
    sample_path=work/'sample.json'
    if sample_path.exists():sample=[tuple(r) for r in json.loads(sample_path.read_text())]
    else:
        excluded={r[0] for r in json.loads((base_work/'sample.json').read_text())}
        rng=np.random.default_rng(SEED);sample=[];seen=0
        for row in base.records(data/'train/train_source1.tsv'):
            if row[0] in excluded:continue
            if seen<size:sample.append(row)
            else:
                j=int(rng.integers(seen+1))
                if j<size:sample[j]=row
            seen+=1
        sample_path.write_text(json.dumps(sample))
    truth=base.load_truth(data,{r[0] for r in sample});(work/'truth.json').write_text(json.dumps({k:sorted(v) for k,v in truth.items()}))
    # Fresh holdout never overlaps any reference inspected in the original run.
    strata=[r[3]+('|single' if not truth[r[0]] else '|matched') for r in sample]
    fit,rest=train_test_split(np.arange(len(sample)),test_size=1/3,random_state=SEED,stratify=strata)
    tune,audit=train_test_split(rest,test_size=.5,random_state=SEED,stratify=np.array(strata)[rest])
    (work/'split_indices.json').write_text(json.dumps({k:v.tolist() for k,v in [('fit',fit),('tune',tune),('audit',audit)]}))
    ref=build_reference(data,'train',work)
    chunks=work/'feature_chunks';chunks.mkdir(exist_ok=True)
    tasks=[]
    for start in range(0,len(sample),100):
        if not (chunks/f'{start:06d}.npz').exists():
            rows=sample[start:start+100];tasks.append((start,rows,{r[0]:truth[r[0]] for r in rows}))
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(str(base_work/'train_index'),str(ref))) as pool:
        pending=set();it=iter(tasks);finished=False;done_count=0
        while pending or not finished:
            while len(pending)<workers*2 and not finished:
                task=next(it,None)
                if task is None:finished=True
                else:pending.add(pool.submit(extract,task))
            if not pending:break
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                start,x,y,g,t=future.result();p=chunks/f'{start:06d}.npz';tmp=p.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez(f,X=x,y=y,group=g,target=np.array(t))
                tmp.replace(p);done_count+=len(set(g))
                if start%1000==0:base.log(f'Enhanced feature chunk {start:,}; {len(y):,} pairs')
    files=sorted(chunks.glob('*.npz'));total=0
    for p in files:
        with np.load(p) as z:total+=len(z['y'])
    maps={k:np.lib.format.open_memmap(work/f'{k}.npy',mode='w+',dtype=d,shape=shape) for k,d,shape in
        [('X',np.float32,(total,120)),('y',np.uint8,(total,)),('group',np.int32,(total,)),('target','U16',(total,))]}
    cursor=0
    for p in files:
        with np.load(p) as z:
            n=len(z['y'])
            for k in maps:maps[k][cursor:cursor+n]=z[k]
            cursor+=n
    for arr in maps.values():arr.flush()
    (work/'features_complete.json').write_text(json.dumps({'pairs':total,'entities':len(sample),'features':120,'version':VERSION}))
    base.log(f'Enhanced features complete: {total:,} pairs')

def model(trees,leaves):
    return lgb.LGBMClassifier(n_estimators=trees,num_leaves=leaves,learning_rate=.035,
        min_child_samples=50,colsample_bytree=.9,reg_lambda=5,n_jobs=8,random_state=SEED,
        deterministic=True,force_col_wise=True,verbosity=-1)

def init_baseline(index,model_path):
    global LEGACY_INDEX,LEGACY_MODEL
    LEGACY_INDEX=base.Index(Path(index));LEGACY_MODEL=lgb.Booster(model_file=model_path)

def baseline_chunk(task):
    indices,rows=task;x,g,t=base.batch_pairs(rows,LEGACY_INDEX)
    p=LEGACY_MODEL.predict(x,num_threads=1) if len(x) else np.array([])
    return np.asarray(indices)[g],t,p

def train(work,base_work):
    x=np.load(work/'X.npy',mmap_mode='r');y=np.load(work/'y.npy');g=np.load(work/'group.npy');t=np.load(work/'target.npy')
    splits=json.loads((work/'split_indices.json').read_text());sample=json.loads((work/'sample.json').read_text())
    tr=json.loads((work/'truth.json').read_text());truth=[set(tr[r[0]]) for r in sample]
    masks={k:np.isin(g,v) for k,v in splits.items()}
    best=None;trials=[]
    # Only tuning labels choose complexity/threshold. Audit is evaluated once.
    for trees,leaves in [(600,31),(800,63)]:
        clf=model(trees,leaves);base.log(f'Training enhanced model: {trees} trees, {leaves} leaves')
        clf.fit(x[masks['fit']],y[masks['fit']],eval_set=[(x[masks['tune']],y[masks['tune']])],callbacks=[lgb.log_evaluation(100)])
        p=clf.booster_.predict(x[masks['tune']],num_threads=8)
        choice=max([(base.score(g[masks['tune']],t[masks['tune']],p,th,splits['tune'],truth)['macro_f0.5'],float(th)) for th in np.arange(.25,.951,.025)])
        trials.append({'trees':trees,'leaves':leaves,'macro_f0.5':choice[0],'threshold':choice[1]})
        if best is None or choice[0]>best[0]:best=(choice[0],choice[1],trees,leaves);clf.booster_.save_model(str(work/'assessment_model.txt'))
    _,threshold,trees,leaves=best
    selected=lgb.Booster(model_file=str(work/'assessment_model.txt'))
    p=selected.predict(x[masks['audit']],num_threads=8)
    audit=base.score(g[masks['audit']],t[masks['audit']],p,threshold,splits['audit'],truth)
    errors=[]
    for gi,ti,yi,pi in zip(g[masks['audit']],t[masks['audit']],y[masks['audit']],p):
        if bool(yi)!=(pi>=threshold):errors.append({'source1_entity_id':sample[int(gi)][0], 'target':str(ti),'label':bool(yi),'probability':float(pi)})
    (work/'validation_errors.json').write_text(json.dumps(errors,indent=2))
    metrics={'version':VERSION,'seed':SEED,'threshold':threshold,'trees':trees,'leaves':leaves,'features':120,
        'sample_entities':len(sample),'training_pairs':len(y),'trials':trials,'held_out':audit,
        'blocking_ceiling':base.score(g[masks['audit']],t[masks['audit']],y[masks['audit']],.5,splits['audit'],truth),
        'candidate_recall':int(y.sum())/sum(map(len,truth))}
    metrics['held_out_by_country']={c:base.score(g[masks['audit']],t[masks['audit']],p,threshold,[i for i in splits['audit'] if sample[i][3]==c],truth) for c in sorted({r[3] for r in sample})}
    # Apples-to-apples comparison: original blocker/model on these SAME fresh entities.
    base.log('Evaluating original pipeline on the same fresh holdout')
    bg=[];bt=[];bp=[];audit_ids=splits['audit']
    tasks=[(audit_ids[s:s+100],[tuple(sample[i]) for i in audit_ids[s:s+100]]) for s in range(0,len(audit_ids),100)]
    with ProcessPoolExecutor(max_workers=3,initializer=init_baseline,initargs=(str(base_work/'train_index'),str(base_work/'model.txt'))) as pool:
        for gg,tt,pp in pool.map(baseline_chunk,tasks):bg.append(gg);bt.extend(tt);bp.append(pp)
    old_threshold=json.loads((base_work/'metrics.json').read_text())['threshold']
    metrics['original_on_same_holdout']=base.score(np.concatenate(bg),np.asarray(bt),np.concatenate(bp),old_threshold,audit_ids,truth)
    (work/'metrics.json').write_text(json.dumps(metrics,indent=2));base.log('Fresh held-out assessment: '+json.dumps(audit))
    clf=model(trees,leaves);base.log('Refitting enhanced model on full supervised sample')
    clf.fit(x,y);clf.booster_.save_model(str(work/'model.txt'))
    base.log('Enhanced final model saved')

def init_predict(index,ref,model_path,threshold):
    global MODEL,THRESHOLD
    init_worker(index,ref);MODEL=lgb.Booster(model_file=model_path);THRESHOLD=threshold
    cache=Path(ref).parent/'prepared_test'
    if (cache/'complete.json').exists():use_prepared_cache(index,cache)

def predict_chunk(task):
    filename,rows=task;x,g,t=make_pairs(rows);p=MODEL.predict(x,num_threads=1) if len(x) else []
    candidates=[[] for _ in rows];matches=[[] for _ in rows]
    for i,tid,prob in zip(g,t,p):
        candidates[int(i)].append(tid)
        if prob>=THRESHOLD:matches[int(i)].append(tid)
    result=[(r[0],sorted(set(c)),sorted(set(m))) for r,c,m in zip(rows,candidates,matches)]
    path=Path(filename);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(path)
    return len(rows)

def predict(data,base_work,work,output,workers):
    base.build_index(data,'test',base_work)
    build_prepared_cache(base_work,work)
    ref=build_reference(data,'test',work);output.mkdir(exist_ok=True,parents=True)
    threshold=json.loads((work/'metrics.json').read_text())['threshold'];chunks=work/'prediction_chunks';chunks.mkdir(exist_ok=True)
    signature={'version':VERSION,'model':hashlib.sha256((work/'model.txt').read_bytes()).hexdigest(),'threshold':threshold,'data':base.fingerprint([data/'test'/f'test_source{s}.tsv' for s in (1,2,3)])}
    manifest=chunks/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==signature
    else:manifest.write_text(json.dumps(signature))
    iterator=base.records(data/'test/test_source1.tsv');number=0;total=0
    with ProcessPoolExecutor(max_workers=workers,initializer=init_predict,initargs=(str(base_work/'test_index'),str(ref),str(work/'model.txt'),threshold)) as pool:
        pending=set()
        while rows:=list(itertools.islice(iterator,200)):
            file=chunks/f'{number:06d}.json';number+=1
            if file.exists():total+=len(rows)
            else:pending.add(pool.submit(predict_chunk,(str(file),rows)))
            if len(pending)>=workers*2:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:total+=future.result()
                if total%5000==0:base.log(f'Enhanced test inference {total:,} entities complete')
        for future in pending:total+=future.result()
    stats={'entities':0,'candidates':0,'matches':0,'singletons':0}
    with (output/'matching_results.tsv').open('w',encoding='utf8',newline='') as fm,(output/'candidate_pairs.tsv').open('w',encoding='utf8',newline='') as fc:
        wm=csv.writer(fm,delimiter='\t',lineterminator='\n');wc=csv.writer(fc,delimiter='\t',lineterminator='\n')
        wm.writerow(['source1_entity_id','matched_entity_ids']);wc.writerow(['source1_entity_id','candidate_entity_ids'])
        for i in range(number):
            for sid,c,m in json.loads((chunks/f'{i:06d}.json').read_text()):
                assert set(m)<=set(c);wm.writerow([sid,','.join(m)]);wc.writerow([sid,','.join(c)])
                stats['entities']+=1;stats['candidates']+=len(c);stats['matches']+=len(m);stats['singletons']+=not m
    (work/'prediction_stats.json').write_text(json.dumps(stats,indent=2));base.log('Enhanced inference complete '+json.dumps(stats))

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--base-work',type=Path,default=Path('artifacts'))
    p.add_argument('--work-dir',type=Path,default=Path('artifacts_v3'));p.add_argument('--output-dir',type=Path,default=Path('output_v3'))
    p.add_argument('--sample-size',type=int,default=60000);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--stage',choices=['reference-train','reference-test','prepare-cache','prepare','train','predict','all'],default='all')
    a=p.parse_args();a.work_dir.mkdir(exist_ok=True,parents=True);a.base_work.mkdir(exist_ok=True,parents=True)
    if a.stage in ['prepare','train','all']:
        if not (a.base_work/'sample.json').exists():base.prepare_training(a.data_dir,a.base_work,30000)
        if not (a.base_work/'model.txt').exists():base.train(a.base_work)
        base.build_index(a.data_dir,'train',a.base_work)
    if a.stage=='prepare-cache':base.build_index(a.data_dir,'test',a.base_work)
    if a.stage.startswith('reference-'):build_reference(a.data_dir,a.stage.split('-')[1],a.work_dir)
    if a.stage=='prepare-cache':build_prepared_cache(a.base_work,a.work_dir)
    if a.stage in ['prepare','all']:prepare(a.data_dir,a.base_work,a.work_dir,a.sample_size,a.workers)
    if a.stage in ['train','all']:train(a.work_dir,a.base_work)
    if a.stage in ['predict','all']:predict(a.data_dir,a.base_work,a.work_dir,a.output_dir,a.workers)

if __name__=='__main__':main()
