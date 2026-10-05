"""Package v5 with fresh assessment, complete outputs, and reproducible code."""
import hashlib,json,shutil,zipfile
from pathlib import Path

def main():
    root=Path.cwd();work=root/'artifacts_v5';code=root/'code/business_entity_resolution';output=root/'output_v5'
    m=json.loads((work/'metrics.json').read_text());v=json.loads((work/'validation_streaming.log').read_text(encoding='utf-8-sig'));stats=json.loads((work/'prediction_stats.json').read_text())
    assert v['status']=='PASS' and b'PASS' in (work/'validation_official.log').read_bytes()
    dest=code/'reports_v5';dest.mkdir(exist_ok=True)
    for name in ['metrics.json','sample.json','prediction_stats.json','gate.json','assessment.log','assessment_extended.log','finish.log','validation_streaming.log','validation_official.log']:
        shutil.copy2(work/name,dest/name)
    doc=root/'Documentation_v5.md';previous=(root/'Documentation_v4.md').read_text(encoding='utf8')
    doc.write_text(f'''# Business Entity Resolution: final empty-prediction recovery cascade (v5)

Team: BusinessEntityResolution (replace with registered name). Members: not supplied.

## Final submitted method

The participant reported public leaderboard scores of 0.954 for v3 and 0.962 for v4. This submission extends v4 only for references that were not selected by its cross-script gate and still have empty predictions. These references are scored using the existing trained v4 model and v4 candidate generation. All other v4 matches and candidates are preserved. No new model fitting, threshold adjustment or leaderboard-label reconstruction is performed. The v4 threshold remains 0.575. An empty prediction is not automatically turned into a match: the trained classifier must accept an actual candidate.

Every reference's exported candidate list is exactly the set scored by its final classifier branch. The previous gate and earlier classifier stages are preliminary stages for references whose final classifier is replaced. Both source files and every country are retained. There is no external data lookup, API, geocoder, downloaded business dataset or manual identity assignment. All original v4/v3 model licenses, architecture, features and blocking details are retained in the supplementary methodology below.

## Fresh offline assessment

A fresh sample of {m['entities']:,} supplied training references is drawn with seed 20260929, with an additional disjoint extension using seed 20260930. The total assessment size was increased before inspecting the initial assessment score. It excludes all 150,000 references used in the three previous supervised samples. Models and thresholds are frozen before this assessment; this sample is not subsequently used for fitting. The exact v4 baseline and proposed v5 cascade are scored on the same entities. The rule was fixed before examining these labels. Automatic generation requires a positive lower endpoint of the approximate 95% paired improvement interval. This measures local evidence, not certainty about the hidden test set.

```json
{json.dumps(m,indent=2)}
```

Macro-F0.5 is calculated per reference, with correct empty singletons scoring 1 and false merges on singletons scoring 0. Pair precision/recall are separate diagnostics. The final leaderboard score is unknown until the organizer evaluates this file. No 0.992 score or top-50 ranking is claimed. Missing addresses, ambiguous identities, missed candidates and unlabeled-country distribution shift remain limitations.

## Final outputs and validation

```json
{json.dumps({'output':stats,'validation':v},indent=2)}
```

Both full TSVs passed the bounded-memory validator, including every target ID, full unique reference coverage, duplicate-free ID lists, exact headers and matches being a subset of final candidates. The official validator additionally passed the matching file with ID existence checks enabled; its optional large candidate loading was replaced by the complete streaming check. The ZIP includes both TSVs, all pipeline source, pinned dependencies, v3/v4 trained models, fresh assessment evidence and a SHA-256 manifest. It is CRC-tested.

## Reproduction

Reproduce v3 and v4 as described in the supplementary methodology and README, retaining their artifacts. From the workspace root, run `python code/business_entity_resolution/src/empty_recovery.py --data-dir DATASET --sample-size 10000`, then repeat with `--sample-size 30000` in the same work directory to reproduce the exact two-stage fresh sample. The final decision uses the combined 30,000-reference assessment. Then run `python code/business_entity_resolution/src/finish_empty.py --data-dir DATASET` to select empty references, run the frozen v4 model, merge full outputs, validate and package. Test outputs are in `output_v5/` during reproduction and at `output/` inside this ZIP. Intermediate caches are rebuilt from the supplied data and are not external dependencies. The complete workflow requires the previous model-training stages, whose code is included; no hidden service is required.

## Supplementary v4 architecture and baseline evidence

The following section describes the v4 baseline underlying the final v5 cascade. Its test counts and local scores refer to that earlier baseline; the final submitted file and fresh comparison are documented above.

{previous}
''',encoding='utf8')
    files=[(doc,'Documentation_template.md')]+[(p,'output/'+p.name) for p in output.glob('*.tsv')]
    files += [(p,'code/business_entity_resolution/'+p.relative_to(code).as_posix()) for p in code.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.venv' not in p.parts]
    manifest={}
    for p,arc in files:
        with p.open('rb') as f:manifest[arc]=hashlib.file_digest(f,'sha256').hexdigest()
    mf=root/'SHA256SUMS_v5.json';mf.write_text(json.dumps(manifest,indent=2));files.append((mf,'SHA256SUMS.json'))
    archive=root/'BusinessEntityResolution_v5_submission.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,arc in files:z.write(p,arc)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(archive)

if __name__=='__main__':main()
