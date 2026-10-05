"""Score only existing positive links with the frozen v4 classifier."""
import argparse,csv,hashlib,itertools,json,mmap
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import lightgbm as lgb
import xxhash
import pipeline as base
import phonetic_pipeline as v4

def build_lookup(index):
    dest=index/'id_lookup';dest.mkdir(exist_ok=True);done=dest/'complete.json'
    if done.exists():return
    buf=[]
    with (index/'records.jsonl').open('rb') as src,(dest/'pairs.bin').open('wb') as out:
        count=0
        while True:
            offset=src.tell();line=src.readline()
            if not line:break
            ident=json.loads(line)[0];buf.append((xxhash.xxh3_64_intdigest(ident),offset));count+=1
            if len(buf)==10000:np.asarray(buf,dtype=base.DT).tofile(out);buf.clear()
        if buf:np.asarray(buf,dtype=base.DT).tofile(out)
    arr=np.fromfile(dest/'pairs.bin',dtype=base.DT);arr.sort(order=['key','offset'])
    arr['key'].tofile(dest/'keys.bin');arr['offset'].tofile(dest/'offsets.bin')
    done.write_text(json.dumps({'records':count,'input':base.fingerprint([index/'records.jsonl'])}))
    base.log(f'ID lookup complete: {index}, {count:,} records')

class Lookup:
    def __init__(self,index):
        self.keys=np.memmap(index/'id_lookup/keys.bin',dtype='<u8',mode='r');self.offsets=np.memmap(index/'id_lookup/offsets.bin',dtype='<u8',mode='r')
        self.file=(index/'records.jsonl').open('rb');self.store=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
    def get(self,ident):
        key=np.uint64(xxhash.xxh3_64_intdigest(ident));lo=int(np.searchsorted(self.keys,key));hi=int(np.searchsorted(self.keys,key,'right'))
        for i in range(lo,hi):
            off=int(self.offsets[i]);end=self.store.find(b'\n',off);row=tuple(json.loads(self.store[off:end]))
            if row[0]==ident:return row
        raise KeyError(ident)

def init(index,ref):
    global LOOKUP,CTX,MODEL
    base.prep=lru_cache(maxsize=20000)(base.prep.__wrapped__)
    base.norm=lru_cache(maxsize=30000)(base.norm.__wrapped__)
    LOOKUP=Lookup(Path(index));CTX=v4.Context(Path(ref));MODEL=lgb.Booster(model_file='artifacts_v4/model.txt')

def batch(task):
    number,rows,matches=task;neighbors=CTX.index.candidates(rows);x=[];groups=[];ids=[]
    for i,(row,targets,offsets) in enumerate(zip(rows,matches,neighbors)):
        if not targets:continue
        near=[CTX.index.get(off) for off in offsets];near=[r for r in near if r[0]!=row[0]]
        near.sort(key=lambda r:v4.similarity(row,r)[2],reverse=True);near=near[:8]
        for tid in targets:x.append(v4.features(row,LOOKUP.get(tid),near,CTX));groups.append(i);ids.append(tid)
    probabilities=MODEL.predict(np.asarray(x,dtype=np.float32).reshape(-1,154),num_threads=1) if x else []
    result=[[r[0],[]] for r in rows]
    for i,tid,p in zip(groups,ids,probabilities):result[i][1].append([tid,float(p)])
    return number,result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=0);p.add_argument('--split',choices=['train','test'],default='test');a=p.parse_args()
    work=Path('artifacts_v7');work.mkdir(exist_ok=True);index=Path('artifacts')/(a.split+'_index');build_lookup(index)
    if a.split=='train':return
    dest=work/'test_rescore';dest.mkdir(exist_ok=True)
    manifest=dest/'manifest.json';signature={'model':hashlib.sha256(Path('artifacts_v4/model.txt').read_bytes()).hexdigest(),'matching':base.fingerprint([Path('output_v4/matching_results.tsv')])}
    if manifest.exists():assert json.loads(manifest.read_text())==signature
    else:manifest.write_text(json.dumps(signature))
    refs=base.records(a.data_dir/'test/test_source1.tsv');total=0;number=0
    with Path('output_v4/matching_results.tsv').open(encoding='utf8',newline='') as f,ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(index),'artifacts_v3/test_reference')) as pool:
        reader=csv.reader(f,delimiter='\t');next(reader);pending=set()
        while rows:=list(itertools.islice(refs,500)):
            previous=list(itertools.islice(reader,len(rows)));assert [r[0] for r in rows]==[r[0] for r in previous]
            filename=dest/f'{number:06d}.json';n=number;number+=1
            if filename.exists():total+=len(rows)
            else:pending.add(pool.submit(batch,(n,rows,[r[1].split(',') if r[1] else [] for r in previous])))
            if len(pending)>=a.workers*2:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    i,result=future.result();path=dest/f'{i:06d}.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(path);total+=len(result)
                if total%5000==0:base.log(f'Matched-link rescore: {total:,} references complete')
            if a.limit and number*500>=a.limit:break
        for future in pending:
            i,result=future.result();path=dest/f'{i:06d}.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(path);total+=len(result)
    base.log(f'Rescore pass complete: {total:,} references, {number:,} chunks')
    if not a.limit:(work/'rescore_complete.json').write_text(json.dumps({'entities':total,'chunks':number}))

if __name__=='__main__':main()
