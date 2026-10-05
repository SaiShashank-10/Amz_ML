"""Apply independently assessed source thresholds to final matched-link scores."""
import argparse,csv,itertools,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--skip-package',action='store_true');a=p.parse_args();root=Path.cwd();work=root/'artifacts_v7';src=Path(__file__).resolve().parent;data=a.data_dir.resolve()
    metrics=json.loads((work/'metrics.json').read_text());assert metrics['paired_improvement']['approximate_95pct_ci'][0]>0,'STOP: no reliable fresh improvement; retain v4'
    def run(script,args,log,cwd=root):
        print(f'Running {script}; log: {log}',flush=True)
        with (work/log).open('w',encoding='utf8') as f:subprocess.run([sys.executable,'-u',str(src/script),*map(str,args)],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
    run('match_rescore.py',['--data-dir',data,'--workers',a.workers],'predict.log')
    done=json.loads((work/'rescore_complete.json').read_text());policy=json.loads((work/'policy.json').read_text());output=root/'output_v7';output.mkdir(exist_ok=True)
    stats={'entities':0,'candidates':0,'matches':0,'singletons':0,'removed_matches':0}
    def scored():
        for i in range(done['chunks']):yield from json.loads((work/'test_rescore'/f'{i:06d}.json').read_text())
    with (root/'output_v4/matching_results.tsv').open(encoding='utf8',newline='') as old,(output/'matching_results.tsv').open('w',encoding='utf8',newline='') as fm,(output/'candidate_pairs.tsv').open('w',encoding='utf8',newline='') as fc:
        reader=csv.reader(old,delimiter='\t');next(reader);wm=csv.writer(fm,delimiter='\t',lineterminator='\n');wc=csv.writer(fc,delimiter='\t',lineterminator='\n')
        wm.writerow(['source1_entity_id','matched_entity_ids']);wc.writerow(['source1_entity_id','candidate_entity_ids'])
        for row,result in itertools.zip_longest(reader,scored()):
            assert row is not None and result is not None and row[0]==result[0]
            sid,pairs=result;c=[tid for tid,pr in pairs];expected=row[1].split(',') if row[1] else [];assert c==expected
            m=[tid for tid,pr in pairs if pr>=(policy['source2_threshold'] if tid.startswith('S2-') else policy['source3_threshold'])]
            wm.writerow([sid,','.join(m)]);wc.writerow([sid,','.join(c)])
            stats['entities']+=1;stats['candidates']+=len(c);stats['matches']+=len(m);stats['singletons']+=not m;stats['removed_matches']+=len(c)-len(m)
    assert stats['entities']==done['entities'];(work/'prediction_stats.json').write_text(json.dumps(stats,indent=2))
    run('validate_streaming.py',['--test-dir',data/'test','--output-dir',output],'validation_streaming.log')
    run('validate_submission.py',['--test-dir',data/'test','--matching',output/'matching_results.tsv','--check-ids'],'validation_official.log',src.parent)
    if not a.skip_package:run('package_rescore.py',[],'package.log')
    print('COMPLETE: v7 predictions validated'+(' and packaged.' if not a.skip_package else '.'),flush=True)

if __name__=='__main__':main()
