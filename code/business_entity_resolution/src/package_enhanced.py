"""Package only completed, validated v3 predictions. Original ZIP stays intact."""
import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import date
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--team',default='BusinessEntityResolution');p.add_argument('--members',default='Not supplied')
    a=p.parse_args();root=a.root.resolve();w=root/'artifacts_v3';output=root/'output_v3';code=root/'code/business_entity_resolution'
    m=json.loads((w/'metrics.json').read_text());stats=json.loads((w/'prediction_stats.json').read_text())
    validation=json.loads((w/'validation_streaming.log').read_text(encoding='utf-8-sig'))
    assert validation['status']=='PASS'
    official=(w/'validation_official.log').read_bytes();assert b'PASS' in official and b'FAIL' not in official
    model_dir=code/'models_v3';report_dir=code/'reports_v3';model_dir.mkdir(exist_ok=True);report_dir.mkdir(exist_ok=True)
    for name in ['model.txt','assessment_model.txt','metrics.json']:shutil.copy2(w/name,model_dir/name)
    for name in ['metrics.json','prediction_stats.json','split_indices.json','split_integrity.json','validation_errors.json','validation_streaming.log','validation_official.log','train.log','predict.log','cache_equivalence.log']:
        shutil.copy2(w/name,report_dir/name)
    fresh=m['held_out'];old=m['original_on_same_holdout'];ceil=m['blocking_ceiling']
    text=f'''# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** {a.team}  
**Team Members:** {a.members}  
**Submission Date:** {date.today().isoformat()}  
**Version:** Revised offline pipeline v3

## 1. Executive Summary

This solution combines disk-backed candidate retrieval with a supervised LightGBM classifier. Relative to the original solution, it adds reference-population rarity, name/address uniqueness, competing-reference similarities, detailed numeric and abbreviation features, and candidate expansion through plausible record variants. All processing uses only the supplied records and labels.

## 2. Methodology

### 2.1 Problem Analysis

The original run achieved local macro-F0.5 0.927462 on 6,000 held-out training references; the user reported a separate Unstop score of 0.898. These are different evaluations. Its weaknesses included unique-name records with missing target addresses, numeric truncation/typos, similar businesses sharing an address, and missed candidates. A repeat of the original pipeline produced identical trained-model bytes and matching-file SHA-256 hashes.

### 2.2 Solution Strategy

The revised supervised sample contains {m['sample_entities']:,} references, drawn with seed {m['seed']} after excluding all 30,000 references from the original run. Splits are by complete Source 1 entity, stratified by country and singleton status: 40,000 fitting, 10,000 tuning, 10,000 final assessment. The split-integrity report verifies disjointness and records a hash of the assessment IDs. Model complexity and threshold are selected only on tuning data. The final model is refitted on the full new sample after assessment; its own training labels are not reported as a held-out score.

Reference frequency and competition statistics use the complete supplied Source 1 input population, without ground-truth labels. This is an explicitly transductive, label-free context computation, performed separately for train and test. France is included in the test reference population. Country values are arbitrary normalized strings, never a hard-coded country vocabulary or country one-hot features. ID values are used only for joins, exclusions and output, never as predictive features.

## 3. Candidate Generation (Blocking)

The original sorted hash indexes cover every Source 2 and Source 3 record. Keys use normalized names, legal-word-reduced names, name prefixes and token pairs, compact names, address tokens and name/address numeric combinations. Text normalization handles Unicode accents, case, punctuation and a fixed small abbreviation map. Keys are scoped by the record's country string, so erroneous country labels can cause misses.

The revised lookup allows blocks of at most 350 records and initially retains at most 500 targets per reference, ordered by shared-key count. Up to three distinct plausible target variants are used as additional retrieval queries. Variant eligibility uses strong compact-name/address similarity, or near-exact address similarity when that reference address is unique. Duplicate normalized query signatures are skipped. No ground-truth label is consulted during retrieval.

The final cheap filter preserves strong names, strong addresses, suitable combined similarities and name-only evidence for missing addresses. It also admits compact-name similarity >= 0.85. The final candidate set is the union of the top 45 candidates by compact-name similarity, address similarity, and combined similarity. Thus at most 135 pairs per reference reach the classifier. The candidate TSV contains exactly these scored pairs, including rejected predictions. Missing retrieval links count against validation recall and are not inserted using labels.

- Sample candidate recall: {m['candidate_recall']:.6f}.
- Fresh assessment candidate recall: {ceil['micro_recall']:.6f}.
- Fresh assessment oracle macro-F0.5 ceiling: {ceil['macro_f0.5']:.6f}.
- Test candidate pairs actually scored: {stats['candidates']:,}.

## 4. Matching Model

The matcher has 120 numeric pair features: the original 50 fuzzy name/address, overlap, numeric and missingness features; compact-name, acronym and script indicators; IDF-weighted overlaps and unmatched-token importance; reference-population exact-name/core-name/address counts and uniqueness flags; numeric agreement grouped by digit length and approximate number alignment; and similarities/margins relative to up to eight competing reference records.

Token rarity uses `log(1 + country_reference_count / (1 + token_document_frequency))`, derived solely from supplied reference records. Competing references are retrieved without labels. Their features help distinguish a plausible pair from an even closer alternative. There is no hard one-to-one matching constraint; each reference may match zero, one or many target records.

Two LightGBM configurations were prespecified for tuning: 600 trees with 31 leaves, and 800 trees with 63 leaves. Both use learning rate 0.035, minimum 50 observations per leaf, L2 regularization 5, feature fraction 0.9 and deterministic seed {m['seed']}. The threshold grid is 0.25 through 0.95 in steps of 0.025, optimized for singleton-aware macro-F0.5.

Selected model: {m['trees']} trees, maximum {m['leaves']} leaves per tree; threshold {m['threshold']:.6f}. Training matrix: {m['training_pairs']:,} candidate pairs. LightGBM and the distributed model/project are MIT licensed. The tree model is far below the 8-billion-parameter limit. No pretrained model, external business dataset, API, registration lookup or geocoder is used.

The optional prepared-text cache stores deterministic text transformations, not predictions or labels. Cached and uncached feature/candidate equivalence is checked and logged. It affects runtime only.

## 5. Results & Error Analysis

Both pipelines below are assessed on the SAME fresh references, none of which appeared in the original supervised sample:

| Measure | Original pipeline | Revised pipeline |
|---|---:|---:|
| Assessment entities | {old['entities']:,} | {fresh['entities']:,} |
| Macro-F0.5 | {old['macro_f0.5']:.6f} | {fresh['macro_f0.5']:.6f} |
| Micro pair precision | {old['micro_precision']:.6f} | {fresh['micro_precision']:.6f} |
| Micro pair recall | {old['micro_recall']:.6f} | {fresh['micro_recall']:.6f} |
| Correct singletons | {old['correct_singletons']}/{old['singletons']} | {fresh['correct_singletons']}/{fresh['singletons']} |

Revised false positives: {fresh['fp']:,}; false negatives including blocking misses: {fresh['fn']:,}. The approximate 95% entity-level normal confidence interval for local macro-F0.5 is {fresh['approximate_95pct_ci']}; it does not account for all distribution shift. Pair precision/recall are diagnostics, not the competition's macro score.

Country-specific fresh assessment results:

```json
{json.dumps(m['held_out_by_country'],indent=2)}
```

Scored pair errors are included in `reports_v3/validation_errors.json`; blocking misses are included in aggregate recall but have no scored pair entry. Remaining limitations include severe native-script/name changes with cropped addresses, ambiguous common names with no address, country-label errors, and bounded candidate retrieval. France has no supplied labeled examples, so no local France accuracy is claimed.

The public/private leaderboard scores for this revised file are UNKNOWN until the organizer evaluates it. A 0.99 score or top-50 placement is not promised or claimed from a local result.

## 6. Conclusion

This package contains an actually trained, independently assessed revised model and predictions for the full test population. The local paired comparison above is the evidence of its measured effect; hidden-test performance remains an organizer evaluation. The original submission has been preserved separately.

## Appendix

### A. Code Artefacts and Reproduction

All source is under `code/business_entity_resolution/src/`. Run `enhanced_pipeline.py --stage all` with the data path, base-work directory, enhanced-work directory and output directory as shown in README.md. It bootstraps the original target indexes/model when needed and then regenerates the revised pipeline. `requirements.txt` pins dependencies. Temporary multi-GB indexes and feature caches are rebuilt and are not shipped.

### B. Output and Verification

- Reference rows: {stats['entities']:,}.
- Matched links: {stats['matches']:,}.
- Empty predictions: {stats['singletons']:,}.
- Candidate pairs: {stats['candidates']:,}.
- Country coverage: {validation['countries']}.

The official validator passed the matching file with ID checks enabled. The bounded-memory validator passed both full files, checking every target ID, exact headers, full and unique reference coverage, duplicate-free lists, allowed source prefixes, and match inclusion in candidates. Validation logs, trained/assessment models, local metrics and split integrity are included. The ZIP is CRC-tested and contains a SHA-256 manifest.
'''
    doc=root/'Documentation_v3.md';doc.write_text(text,encoding='utf8')
    files=[(doc,'Documentation_template.md')]+[(f,'output/'+f.name) for f in sorted(output.glob('*.tsv'))]
    files += [(f,'code/business_entity_resolution/'+str(f.relative_to(code)).replace('\\','/')) for f in sorted(code.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and '.venv' not in f.parts]
    manifest={}
    for file,arc in files:
        with file.open('rb') as stream:manifest[arc]=hashlib.file_digest(stream,'sha256').hexdigest()
    mf=root/'SHA256SUMS_v3.json';mf.write_text(json.dumps(manifest,indent=2));files.append((mf,'SHA256SUMS.json'))
    slug=''.join(c if c.isalnum() or c in '-_' else '_' for c in a.team).strip('_')[:80] or 'BusinessEntityResolution'
    archive=root/f'{slug}_v3_submission.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file,arc in files:z.write(file,arc)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(archive)

if __name__=='__main__':main()
