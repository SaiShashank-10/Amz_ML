"""Rebuild the submitted v7 cascade from the supplied data in an isolated directory.

No training, candidate, feature, threshold or inference definitions are changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    data, work = args.data_dir.resolve(), args.work_root.resolve()
    src = Path(__file__).resolve().parent
    common = ['--data-dir', str(data), '--workers', str(args.workers)]
    # The two development sampling calls are intentional: their seeds differ.
    steps = [
        ('enhanced_pipeline.py', common + ['--stage', 'prepare'], 'artifacts_v3/prepare.log'),
        ('enhanced_pipeline.py', common + ['--stage', 'train'], 'artifacts_v3/train.log'),
        ('enhanced_pipeline.py', common + ['--stage', 'predict'], 'artifacts_v3/predict.log'),
        ('phonetic_pipeline.py', common + ['--stage', 'prepare'], 'artifacts_v4/prepare.log'),
        ('phonetic_pipeline.py', common + ['--stage', 'train'], 'artifacts_v4/train.log'),
        ('build_test_phonetic.py', [], 'artifacts_v4/test_index.log'),
        ('hybrid_pipeline.py', common + ['--stage', 'all'], 'artifacts_v4/cascade.log'),
        ('empty_recovery.py', common + ['--sample-size', '10000'], 'artifacts_v5/initial.log'),
        ('empty_recovery.py', common + ['--sample-size', '30000'], 'artifacts_v5/extension.log'),
        ('match_rescore.py', common + ['--split', 'train'], 'artifacts_v7/train_lookup.log'),
        ('calibrate_rescore.py', common, 'artifacts_v7/development.log'),
        ('empty_recovery.py', common + ['--sample-size', '10000', '--work-dir', 'artifacts_v7/fresh', '--exclude-work', 'artifacts_v5'], 'artifacts_v7/fresh_baseline.log'),
        ('calibrate_rescore.py', common + ['--assessment-work', 'artifacts_v7/fresh', '--evaluate-only'], 'artifacts_v7/fresh_evaluation.log'),
        ('finish_rescore.py', common + ['--skip-package'], 'artifacts_v7/finish.log'),
    ]
    for script, flags, logfile in steps:
        print(subprocess.list2cmdline([sys.executable, '-u', str(src / script), *flags]), flush=True)
        print('  working directory:', work, '; log:', logfile, flush=True)
    if args.dry_run:
        return
    required = [data / split / f'{split}_source{i}.tsv' for split in ('train', 'test') for i in (1, 2, 3)]
    required.append(data / 'train/train_ground_truth.tsv')
    for path in required:
        if not path.is_file():
            parser.error(f'Missing supplied input: {path}')
    # A completed training log is a cascade precondition. Do not truncate a
    # historical completed log or mix a new orchestration run with old caches.
    if work.exists() and any(work.iterdir()):
        parser.error('Use a new or empty --work-root directory to avoid mixing cached runs.')
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for number, (script, flags, logfile) in enumerate(steps, 1):
        log = work / logfile
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f'Stage {number}/{len(steps)}: {script}; log {log}', flush=True)
        with log.open('w', encoding='utf8') as stream:
            subprocess.run([sys.executable, '-u', str(src / script), *flags], cwd=work,
                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    output = work / 'output'
    output.mkdir(exist_ok=True)
    hashes = {}
    for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
        shutil.copy2(work / 'output_v7' / name, output / name)
        with (output / name).open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (work / 'reproduction_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf8')
    print(f'COMPLETE: validated regenerated files in {output}', flush=True)


if __name__ == '__main__':
    main()
