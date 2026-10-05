"""Package the validated positive-link rescoring cascade and its evidence."""
import hashlib,json,shutil,zipfile
from pathlib import Path

def main():
    root=Path.cwd();work=root/'artifacts_v7';code=root/'code/business_entity_resolution';output=root/'output_v7'
    metrics=json.loads((work/'metrics.json').read_text());policy=json.loads((work/'policy.json').read_text());stats=json.loads((work/'prediction_stats.json').read_text());validation=json.loads((work/'validation_streaming.log').read_text(encoding='utf-8-sig'))
    assert validation['status']=='PASS' and b'PASS' in (work/'validation_official.log').read_bytes()
    reports=code/'reports_v7';reports.mkdir(exist_ok=True)
    for name in ['metrics.json','policy.json','development.json','rescore_consistency.json','prediction_stats.json','fresh_evaluation.log','fresh_baseline.log','predict.log','validation_streaming.log','validation_official.log']:
        shutil.copy2(work/name,reports/name)
    shutil.copy2(work/'fresh/sample.json',reports/'fresh_sample.json')
    doc=root/'Documentation_v7.md';previous=(root/'Documentation_v4.md').read_text(encoding='utf8')
    doc.write_text(f'''# Business Entity Resolution: source-calibrated positive-link cascade (v7)

Team: BusinessEntityResolution (replace with registered team name). Members: not supplied.

## Final method

The v4 submission achieved a participant-reported public score of 0.962. This version retains its complete preliminary prediction pipeline and then applies the frozen v4 LightGBM classifier to EVERY previously accepted pair. It does not add links. Source 2 and Source 3 use separately calibrated acceptance thresholds: {policy['source2_threshold']} and {policy['source3_threshold']}, respectively. Source comes from the allowed ID prefix; numerical ID contents are never features. The country remains an open normalized string. All countries and every reference, including France, are exported.

This is model agreement with a calibrated final classifier: prior v3/v4 acceptance is required, and the final frozen v4 classifier must also accept the pair. The model uses the same 154 string, rarity, numeric-order and competing-reference features as v4, with the same neighbor selection and label-free population statistics. The final model is not retrained for this stage. The input for this last ML stage is the set of v4-positive pairs, so `candidate_pairs.tsv` now contains EXACTLY those pairs, including links rejected by the last classifier. Earlier larger candidate pools belong to preliminary stages and are not exported as final-stage candidates. Empty preliminary predictions have an empty final candidate list. This deliberately trades some recall for precision; all test outputs remain many-to-one from targets to the deduplicated reference framework, without imposing a forced target assignment.

An ID-to-offset lookup accelerates retrieval of the supplied records. A 64-bit hash locates a range, but the complete original ID is always checked, including collision handling. This is only a join/index mechanism, never a predictive ID feature. No external identity lookup, API, geocoder, business dataset or internet augmentation is used.

## Development and independent assessment

Thresholds are selected on 30,000 references previously held out from all model fitting. These are explicitly development data for v7, not a new independent score. The grid for each target source is 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.575, 0.65, 0.75 and 0.85; the pair maximizing macro-F0.5 is frozen. A separate new 10,000-reference sample excludes every reference from the original 30,000 sample, v3 60,000 sample, v4 60,000 sample and the 30,000 development sample. It is evaluated once with the frozen models and thresholds. No new fitting or tuning occurs on this final sample.

Both the existing v4 cascade and this exact rescoring cascade are evaluated on the same fresh references. Automatic production requires a positive lower endpoint of the approximate 95% paired improvement interval.

```json
{json.dumps(metrics,indent=2)}
```

The score is per-reference macro-F0.5, including singleton credit. Pair precision and recall are diagnostics. France has no provided labels, so its performance is not claimed from this local evaluation. The public/private score for v7 is unknown until organizer evaluation. A score of 0.992 or top-50 placement is not promised. The previous unsuccessful empty-recovery and simple ambiguity-rule experiments are not applied to this submitted file.

## Output verification

```json
{json.dumps({'prediction_stats':stats,'validation':validation},indent=2)}
```

A benchmark also rescored 6,088 already-v4-accepted links from the first 10,000 test references without any unexpected rejection at the original threshold; its report is included. Both full output files passed streaming validation of IDs, source prefixes, reference coverage, headers, uniqueness and match inclusion in candidates. The official matching-file validator passed with ID existence checks. All final candidates are verified to equal the actual final classifier's input lists. The ZIP includes source, pinned dependencies, previously trained MIT-licensed models, methodology and reports. All models are far below 8 billion parameters. A SHA-256 manifest and ZIP CRC check verify package integrity.

## Exact reproduction

First reproduce v3 and v4 using the supplementary method and README, retaining `artifacts`, `artifacts_v3`, `artifacts_v4` and `output_v4`. Recreate the development sample/predictions using `empty_recovery.py --sample-size 10000`, then `--sample-size 30000`, in the default `artifacts_v5` directory. Those commands produce assessment caches only; do not run `finish_empty.py`, since that rejected experiment is not part of this model.

Build training ID lookup with `match_rescore.py --split train --data-dir DATASET`. Run `calibrate_rescore.py --data-dir DATASET` to generate the development policy. Generate a new baseline assessment with `empty_recovery.py --data-dir DATASET --sample-size 10000 --work-dir artifacts_v7/fresh --exclude-work artifacts_v5`. Then run `calibrate_rescore.py --data-dir DATASET --assessment-work artifacts_v7/fresh --evaluate-only`. Finally run `finish_rescore.py --data-dir DATASET`; it regenerates test scores, writes both TSVs under `output_v7`, validates them and packages this archive. All commands use source scripts under `code/business_entity_resolution/src/` and run from the workspace root. Intermediate indexes are regenerated exclusively from supplied data. An existing completed score chunk is reused only under a matching model/input manifest.

## Supplementary baseline architecture

The following describes the underlying v4 baseline. Its earlier test counts and assessment scores belong to v4; the final v7 counts and independent comparison appear above.

{previous}
''',encoding='utf8')
    files=[(doc,'Documentation_template.md')]+[(p,'output/'+p.name) for p in output.glob('*.tsv')]
    files += [(p,'code/business_entity_resolution/'+p.relative_to(code).as_posix()) for p in code.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.venv' not in p.parts]
    manifest={}
    for p,arc in files:
        with p.open('rb') as f:manifest[arc]=hashlib.file_digest(f,'sha256').hexdigest()
    mf=root/'SHA256SUMS_v7.json';mf.write_text(json.dumps(manifest,indent=2));files.append((mf,'SHA256SUMS.json'))
    dest=root/'BusinessEntityResolution_v7_submission.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,arc in files:z.write(p,arc)
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    print(dest)

if __name__=='__main__':main()
