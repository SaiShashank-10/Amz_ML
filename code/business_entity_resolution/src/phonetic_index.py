"""Additional bounded phonetic blocking over supplied target records only."""
import json
import re
from pathlib import Path
from collections import Counter
import numpy as np
import pipeline as base
import phonetic_features as phon

SCRIPT=re.compile('[\u0900-\u0dff]')

def build_native(index_path):
    dest=index_path/'phonetic_v4';path=dest/'native.jsonl'
    if path.exists():return
    temp=path.with_suffix('.tmp')
    with (index_path/'records.jsonl').open('rb') as stream,temp.open('w',encoding='utf8') as out:
        while True:
            offset=stream.tell();line=stream.readline()
            if not line:break
            row=tuple(json.loads(line))
            if SCRIPT.search(row[1]):out.write(json.dumps([offset,row[0],phon.forms(row)[1]])+'\n')
    temp.replace(path)

class Gate:
    def __init__(self,path):
        self.index=Index(path)
        self.by_offset={}
        with (path/'phonetic_v4/native.jsonl').open() as f:
            for line in f:
                offset,ident,name=json.loads(line);self.by_offset[offset]=(ident,name)
        self.by_id={ident:name for ident,name in self.by_offset.values()}
    def select(self,rows,matches):
        from rapidfuzz import fuzz
        candidates=self.index.extra_candidates(rows);selected=[]
        for row,matched,offsets in zip(rows,matches,candidates):
            name=phon.forms(row)[1]
            suspicious=any(ident in self.by_id and fuzz.token_sort_ratio(name,self.by_id[ident])<50 for ident in matched)
            recoverable=any(ident not in matched and fuzz.token_sort_ratio(name,pname)>=80
                for off in offsets if off in self.by_offset for ident,pname in [self.by_offset[off]])
            selected.append(suspicious or recoverable)
        return selected

def build(index_path):
    dest=index_path/'phonetic_v4';dest.mkdir(exist_ok=True)
    fp=base.fingerprint([index_path/'records.jsonl']);done=dest/'complete.json'
    if done.exists():
        assert json.loads(done.read_text())['input']==fp
        return
    handles=[(dest/f'{i:02x}.bin').open('wb') for i in range(256)]
    buffers=[[] for _ in handles]
    def flush():
        for f,b in zip(handles,buffers):
            if b:np.asarray(b,dtype=base.DT).tofile(f);b.clear()
    with (index_path/'records.jsonl').open('rb') as stream:
        count=0
        while True:
            offset=stream.tell();line=stream.readline()
            if not line:break
            row=tuple(json.loads(line))
            for key in phon.keys(row):buffers[key&255].append((key,offset))
            count+=1
            if count%10000==0:flush()
            if count%250000==0:base.log(f'Phonetic index {index_path.name}: {count:,} records')
    flush()
    for f in handles:f.close()
    for i in range(256):
        path=dest/f'{i:02x}.bin';arr=np.fromfile(path,dtype=base.DT);arr.sort(order=['key','offset']);arr.tofile(path)
    done.write_text(json.dumps({'input':fp,'records':count,'version':4}))

class Index(base.Index):
    def __init__(self,path,max_block=350,max_candidates=500):
        super().__init__(path,max_block,max_candidates)
        self.phon_shards=[np.memmap(p,dtype=base.DT,mode='r') if p.stat().st_size else np.empty(0,dtype=base.DT) for i in range(256) for p in [path/'phonetic_v4'/f'{i:02x}.bin']]
    def extra_candidates(self,rows):
        requests=[[] for _ in range(256)];counts=[Counter() for _ in rows]
        for i,row in enumerate(rows):
            for k in phon.keys(row):requests[k&255].append((k,i))
        for shard,req in zip(self.phon_shards,requests):
            if not req:continue
            ks=np.array([k for k,i in req],dtype=np.uint64)
            lo=np.searchsorted(shard['key'],ks,'left');hi=np.searchsorted(shard['key'],ks,'right')
            for (_,i),start,end in zip(req,lo,hi):
                if 0<end-start<=500:counts[i].update(int(x) for x in shard['offset'][start:end])
        return [sorted(c,key=lambda k:(-c[k],k))[:200] for c in counts]
    def candidates(self,rows):
        return [sorted(set(old)|set(extra)) for old,extra in zip(super().candidates(rows),self.extra_candidates(rows))]
