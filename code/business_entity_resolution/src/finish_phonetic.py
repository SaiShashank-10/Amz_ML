"""Run the measured correction cascade, validate outputs, and package v4."""
import argparse,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);a=p.parse_args()
    root=Path.cwd();src=root/'code/business_entity_resolution/src';work=root/'artifacts_v4';data=a.data_dir.resolve()
    def run(script,args,log,cwd=root):
        print(f'Running {script}; log: {log}',flush=True)
        with (work/log).open('w',encoding='utf8') as f:
            subprocess.run([sys.executable,'-u',str(src/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,cwd=cwd,check=True)
    run('hybrid_pipeline.py',['--data-dir',data,'--stage','predict'],'predict.log')
    run('hybrid_pipeline.py',['--data-dir',data,'--stage','merge'],'merge.log')
    run('validate_streaming.py',['--test-dir',data/'test','--output-dir',root/'output_v4'],'validation_streaming.log')
    run('validate_submission.py',['--test-dir',data/'test','--matching',root/'output_v4/matching_results.tsv','--check-ids'],'validation_official.log',src.parent)
    run('package_hybrid.py',[],'package.log')
    print('COMPLETE: v4 cascade predictions validated and packaged.',flush=True)

if __name__=='__main__':main()
