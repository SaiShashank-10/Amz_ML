"""Package validated v4 cascade outputs with its actual measured methodology."""
import hashlib,json,shutil,zipfile
from pathlib import Path

def main():
    root=Path.cwd();work=root/'artifacts_v4';code=root/'code/business_entity_resolution';output=root/'output_v4'
    m=json.loads((work/'metrics.json').read_text());v=json.loads((work/'validation_streaming.log').read_text(encoding='utf-8-sig'))
    assert v['status']=='PASS' and b'PASS' in (work/'validation_official.log').read_bytes()
    gate=json.loads((work/'gate.json').read_text());stats=json.loads((work/'prediction_stats.json').read_text())
    assert stats['corrected_entities']==gate['selected_count']
    for folder,names in [('models_v4',['model.txt','assessment_model.txt','metrics.json']),('reports_v4',['metrics.json','prediction_stats.json','split_indices.json','split_integrity.json','tests.log','validation_streaming.log','validation_official.log','train.log','gate.log','predict.log'])]:
        dest=code/folder;dest.mkdir(exist_ok=True)
        for name in names:shutil.copy2(work/name,dest/name)
    doc=root/'Documentation_v4.md'
    doc.write_text(f'''# Business Entity Resolution: v4 targeted correction cascade

Team: BusinessEntityResolution (replace with registered team name). Members: not supplied.

## Methodology and data

The previous v3 submission scored 0.954 on the public leaderboard according to the participant. The new approach preserves that complete prediction set and replaces the final classifier for references selected by a deterministic cross-script gate. There are no internet business lookups, geocoders, registration services or external identity datasets. All candidate records, population statistics and labels come from the supplied challenge files.

The new supervised sample contains {m['sample_entities']:,} Source 1 references, excluding every reference from the original 30,000 sample and the v3 60,000 sample. Seed {m['seed']} determines a country/singleton-stratified split: two thirds fitting, one sixth threshold/model selection, one sixth fresh assessment. All target records remain available for retrieval. Candidate misses count as false negatives. The final new model is refitted on the complete sample only after held-out assessment. Reusing label-free v3 reference-frequency indexes does not introduce held-out matching labels.

## Candidate retrieval and normalization

The original exact, token-combination, address and numeric hash indexes are supplemented with phonetic-name, name-token, name-prefix/suffix and romanized-address-pair indexes. Indic scripts are converted mechanically using Python Unicode character names, vowel-sign and virama handling. Phonetic folding reduces vowel and consonant-spelling variations. Latin accents are normalized too. This is a deterministic text transform, not a pretrained translation model or business-name dictionary. Transliteration is approximate and is only evidence for supervised matching, never an automatic merge rule.

Supplementary blocks larger than 500 records are skipped; at most 200 supplementary offsets are retained by shared-key count. They are unioned with the v3 initial candidates. Strong variants can expand retrieval. A cheap filter then retains plausible name/address or phonetic similarity, followed by the union of the top 60 candidates under four rankings: original compact name, address, combined similarity and phonetic name. At most 240 candidate pairs per corrected reference reach the new final classifier.

## Features and model

The new classifier uses 154 features: the previous 120 name/address, rarity, uniqueness, numeric and competing-reference features plus 34 romanized/phonetic fuzzy-similarity and ordered numeric features. Numeric evidence preserves first street/building numbers and alphanumeric units instead of relying only on an unordered number set. Two prespecified LightGBM configurations are compared on tuning data: 800 trees/63 leaves and 1,200 trees/127 leaves. Learning rate 0.035, minimum leaf size 50, L2 regularization 5 and feature fraction 0.9 are fixed. The threshold grid is 0.25 through 0.95 in steps of 0.025.

Selected new model: {m['trees']} trees, maximum {m['leaves']} leaves, threshold {m['threshold']}. Supervised matrix: {m['training_pairs']:,} pairs. The previous v3 model is also required for the cascade. Both models and project code are MIT licensed and far below 8 billion parameters; no pretrained model is used.

## Exact cascade rule and candidate audit

The gate selects a reference if supplementary retrieval finds an Indic-name candidate absent from its v3 predicted matches with phonetic token-sort similarity at least 80/100, or if a v3 predicted Indic-name match has phonetic token-sort similarity below 50/100. It uses records and v3 predictions, never labels. All other references retain their v3 final predictions and candidate sets. Selected references use the new model and its final scored candidate set. The candidate TSV therefore records the actual inputs to the final classifier branch used for each reference. Earlier gate and preliminary v3 computations are not claimed as final-model candidates for corrected references.

Test gate selection: {gate['selected_count']:,} out of {gate['entities']:,} references. Both source-2 and source-3 matches are many-to-one; no forced one-to-one assignment is used. Countries remain arbitrary strings. Every test reference, including France, is exported; unselected references are preserved.

## Offline evidence for this exact submitted cascade

The v3 model and the new cascade are evaluated on the SAME fresh references. The gate is applied using each assessment reference's actual v3 predictions and supplementary candidates.

```json
{json.dumps({'v3_same_holdout':m['original_on_same_holdout'],'submitted_cascade_holdout':m['hybrid_held_out'],'gated_assessment_entities':m['hybrid_gate_entities'],'paired_improvement':m['hybrid_paired_improvement'],'new_model_on_all_assessment_entities_not_the_submitted_cascade':m['held_out'],'new_blocking_ceiling':m['blocking_ceiling']},indent=2)}
```

Macro-F0.5 averages 1.25*TP/(0.25*true_count+predicted_count) across reference entities, with 1 for a correctly empty singleton and 0 for a false merge on a singleton. Micro pair precision and recall are separate diagnostics. Local scores are not public/private leaderboard scores. The revised leaderboard score is unknown until submission. Neither 0.99 nor a rank is guaranteed. Remaining limitations include ambiguous businesses with missing addresses, approximate transliteration, blocked large candidate groups and unseen-country distribution shifts.

## Outputs and verification

```json
{json.dumps(v,indent=2)}
```

The bounded-memory validator checks both complete TSVs, all target-ID existence, unique reference coverage, duplicate-free lists, exact headers and match inclusion in final candidates. The official validator additionally checks the matching file; its optional candidate loading is omitted to control memory. The ZIP includes both TSVs, source, pinned dependencies, trained models, reports and this methodology. SHA-256 hashes cover every packaged input file; the archive is CRC-checked.

## Reproduction

Run from a workspace root with the supplied dataset. First reproduce v3 using `enhanced_pipeline.py --stage all` as documented in README.md; keep its `artifacts_v3` directory, models and prediction chunks. Then run `phonetic_pipeline.py --stage prepare`, followed by `--stage train`, using the same data directory and default artifact paths. Build test supplementary retrieval with `phonetic_index.build(Path('artifacts/test_index'))` and `phonetic_index.build_native(...)`. Finally run `hybrid_pipeline.py --stage all --data-dir DATASET`, validate `output_v4`, and run `package_hybrid.py`. Source for every component is included. Intermediate caches and full data are not packaged. A resumable deadline-aware runner is provided as `finish_phonetic.py`.
''',encoding='utf8')
    files=[(doc,'Documentation_template.md')]+[(p,'output/'+p.name) for p in output.glob('*.tsv')]
    files += [(p,'code/business_entity_resolution/'+p.relative_to(code).as_posix()) for p in code.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.venv' not in p.parts]
    manifest={}
    for p,arc in files:
        with p.open('rb') as f:manifest[arc]=hashlib.file_digest(f,'sha256').hexdigest()
    mf=root/'SHA256SUMS_v4.json';mf.write_text(json.dumps(manifest,indent=2));files.append((mf,'SHA256SUMS.json'))
    dest=root/'BusinessEntityResolution_v4_submission.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,arc in files:z.write(p,arc)
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    print(dest)

if __name__=='__main__':main()
