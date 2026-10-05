"""Resume v3 inference, validate both outputs, then create the submission ZIP."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    root = a.root.resolve()
    data = a.data_dir.resolve()
    work = root / 'artifacts_v3'
    code = root / 'code/business_entity_resolution'
    src = code / 'src'
    metrics = json.loads((work / 'metrics.json').read_text())
    assert (work / 'model.txt').is_file(), 'Complete training first'
    assert 'Enhanced final model saved' in (work / 'train.log').read_text(encoding='utf-8-sig', errors='replace'), 'Training has not finished'
    assert 'PASS:' in (work / 'cache_equivalence.log').read_text(encoding='utf-8-sig'), 'Verify inference cache first'
    assert metrics['held_out']['macro_f0.5'] > metrics['original_on_same_holdout']['macro_f0.5'], 'Revision did not improve the held-out comparison; review before generating submission'

    def run(script, args, log, cwd=root):
        print(f'Running {script}; log: {log}', flush=True)
        with (work / log).open('w', encoding='utf8') as stream:
            subprocess.run([sys.executable, '-u', str(src / script), *map(str, args)], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)

    run('enhanced_pipeline.py', ['--data-dir', data, '--stage', 'predict', '--workers', a.workers], 'predict.log')
    run('validate_streaming.py', ['--test-dir', data / 'test', '--output-dir', root / 'output_v3'], 'validation_streaming.log')
    # The official utility auto-discovers candidates relative to cwd. Use the
    # code directory for its matching-only check; streaming checks both files.
    run('validate_submission.py', ['--matching', root / 'output_v3/matching_results.tsv', '--test-dir', data / 'test', '--check-ids'], 'validation_official.log', code)
    run('package_enhanced.py', ['--root', root], 'package.log')
    print('COMPLETE: revised outputs validated and submission ZIP created.', flush=True)


if __name__ == '__main__':
    main()
