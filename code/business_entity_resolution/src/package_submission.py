"""Create the final methodology document and reproducible submission archive."""
import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import date
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--team',default='BusinessEntityResolution'); p.add_argument('--members',default='Not supplied')
    a=p.parse_args(); root=a.root.resolve(); work=root/'artifacts'; code=root/'code/business_entity_resolution'
    validation=json.loads((work/'validation_streaming.log').read_text(encoding='utf-8-sig'))
    assert validation['status']=='PASS'
    official=(work/'validation_official.log').read_bytes()
    assert b'PASS' in official and b'FAIL' not in official, 'Official validation must pass before packaging'
    metrics=json.loads((work/'metrics.json').read_text()); stats=json.loads((work/'prediction_stats.json').read_text())
    inventory=json.loads((work/'data_inventory.json').read_text())
    target_count=sum(v['rows'] for k,v in inventory.items() if k.replace('\\','/').startswith('test/') and k.endswith(('source2.tsv','source3.tsv')))
    reduction=1-stats['candidates']/(stats['entities']*target_count)
    models=code/'models'; models.mkdir(exist_ok=True)
    reports=code/'reports'; reports.mkdir(exist_ok=True)
    for name in ('model.txt','metrics.json'): shutil.copy2(work/name,models/name)
    for name in ('metrics.json','data_inventory.json','prediction_stats.json','validation_errors.json','error_analysis.json','splits.json','validation_streaming.log','validation_official.log','run.log','predict.log'):
        shutil.copy2(work/name,reports/name)
    m=metrics['held_out']; tune=metrics['tuning']; ceiling=metrics['held_out_blocking_ceiling']
    analysis=json.loads((work/'error_analysis.json').read_text())['counts']
    sample=json.loads((work/'sample.json').read_text()); truth=json.loads((work/'truth.json').read_text())
    doc=f'''# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** {a.team}  
**Team Members:** {a.members}  
**Submission Date:** {date.today().isoformat()}

## 1. Executive Summary

This solution uses a disk-backed inverted blocking index followed by a supervised LightGBM binary classifier trained from scratch on the supplied labeled data. The threshold optimizes entity-level macro-F0.5 including singletons. All test reference entities receive exactly one output row, including France.

## 2. Methodology

### 2.1 Problem Analysis

The supplied data contains millions of records, noisy names, reordered addresses and missing target addresses. Reference entities are deduplicated and target entities may match zero, one or many records. Dataset counts, country counts, missing fields, file sizes and SHA-256 hashes are recorded in `reports/data_inventory.json`.

No external business data, lookup service, geocoder, API, pretrained embedding or external identity augmentation was used. No ID numerical patterns were used as features. The original data was read as UTF-8 tab-separated files without modification.

### 2.2 Solution Strategy

**Approach Type:** Blocking + supervised pair classifier.

The system makes the full target population searchable using on-disk sorted hash-key shards. A uniformly random reservoir sample of {metrics['sample_entities']:,} training reference entities is used for supervised learning; all training Source 2 and Source 3 records remain available for candidate retrieval. The sample is partitioned by reference entity into 60% fitting, 20% threshold tuning and 20% held-out assessment, stratified by country and singleton status, with seed {metrics['seed']}. Pairs from a reference never span these partitions. Indexing is label-free. True pairs missed by blocking are not inserted into training or validation candidates, and count against recall.

The final model is refitted on all {metrics['sample_entities']:,} sampled training entities after selecting the threshold and assessing the holdout. This is genuine sampled supervised training, not full supervised fitting on every training reference. The held-out assessment describes the pre-refit model, not an evaluation of the refitted model on its own training labels.

## 3. Candidate Generation (Blocking)

Text normalization applies Unicode NFKD decomposition, accent removal, case folding, punctuation tokenization, ampersand handling and a fixed small abbreviation map. A generic legal-word set is removed from a secondary name representation. Original normalized names are also retained. Country is an open string value; no fixed country vocabulary or one-hot country encoding is used.

Blocking keys include full normalized name, sorted core name, sorted four-character core token prefixes, sorted address tokens, pairs of up to five longest name-token prefixes, and name-token prefixes paired with address numbers or long address tokens. A supplementary index adds compact core names, pairs of long address-token prefixes and address-number/long-token combinations, so retrieval does not always require similar names. All keys include normalized country. Thus conflicting country labels are not retrieved; this is a limitation if country labels themselves are wrong.

Keys are xxHash 64-bit values stored with offsets into a canonical JSON-lines target record store. The base and supplementary indexes each contain 256 disk shards sorted by key and offset. Hash collisions only add candidates; full text features determine final decisions. Blocks containing more than 100 target records are skipped to avoid common-key explosions. Candidates are ranked by the number of distinct shared keys, with record offset as a deterministic tie-break, and capped at 240 per reference. These caps can lose true matches and are included in measured blocking recall. Empty candidate sets remain empty.

A final inexpensive filter uses token-set similarity of core names (N) and normalized addresses (A), retaining a pair when N >= 0.90, A >= 0.85, both N >= 0.35 and A >= 0.50, or an address is missing and N >= 0.50. A numerical tolerance of 1e-7 is used. This filter was selected using only 2,000 fitting-partition entities with the original base index: it retained all 6,040 retrieved true pairs in that probe while retaining about 49.85% of broad candidates. This observation is not a guaranteed recall bound for the final index; the held-out blocking metrics below include the filter and supplementary index. The final post-filter candidate set is precisely what is exported and scored. Strong address matches are preserved even when names use different scripts or trade names. The supplementary index was motivated by the original fitting-partition blocking recall of 0.8851; no audit-partition score was inspected while designing it.

- Sample-wide candidate pair recall: **{metrics['candidate_recall']:.6f}**.
- Held-out candidate recall: **{ceiling['micro_recall']:.6f}**.
- Held-out oracle macro-F0.5 ceiling using only retrieved true pairs: **{ceiling['macro_f0.5']:.6f}**.
- Final test candidate pairs: **{stats['candidates']:,}**.
- Reduction ratio versus all test reference/target pairs: **{reduction:.10f}**; mean candidates per reference: **{stats['candidates']/stats['entities']:.2f}**.
- Every exported candidate is scored exactly once by the final model. There is no additional hidden candidate filter. Every final match is a subset of the exported candidates.

## 4. Matching Model

**Model:** LightGBM gradient-boosted binary decision trees; {metrics['tree_count']} trees, at most 31 leaves per tree, learning rate 0.045, minimum 60 observations per leaf, L2 regularization 3, feature fraction 0.9, deterministic seed {metrics['seed']}. No class weighting or synthetic positives are used. All retrieved negative pairs for the sampled reference entities are included.

**Features:** {metrics['features']} numeric pair features. Full and legal-word-reduced names and addresses each contribute edit-ratio, partial-ratio, token-sort, token-set and weighted-ratio similarities, exact equality and relative lengths. Additional features describe name-token, address-token and numeric-token Jaccard/containment overlap; token counts; numeric disagreements; country equality; missing fields; and bidirectional best name-token edit alignment.

The final training matrix contains **{metrics['training_pairs']:,}** candidate pairs. Neither target source, entity IDs, ID digits nor memorized country labels are model features. The model can emit many matches per reference; there is no one-to-one assignment restriction or graph propagation.

**Threshold selection:** A fixed grid from 0.10 through 0.95 with step 0.025 was evaluated on the tuning partition using the exact singleton-aware macro-F0.5. Ties select the higher threshold. Selected threshold: **{metrics['threshold']:.6f}**. Model tree count was fixed in advance; tuning loss was logged but not used for early stopping.

**License and size:** LightGBM's installed distribution is MIT licensed; its license is included. The newly trained model is distributed under the project's MIT license. It has {metrics['total_leaves']:,} leaves in total, far below the 8-billion-parameter limit. No pretrained model was downloaded.

## 5. Results & Error Analysis

| Measure | Tuning partition | Held-out assessment |
|---|---:|---:|
| Reference entities | {tune['entities']:,} | {m['entities']:,} |
| Macro-F0.5 | {tune['macro_f0.5']:.6f} | {m['macro_f0.5']:.6f} |
| Micro pair precision (diagnostic) | {tune['micro_precision']:.6f} | {m['micro_precision']:.6f} |
| Micro pair recall (diagnostic) | {tune['micro_recall']:.6f} | {m['micro_recall']:.6f} |
| Correct singleton predictions | {tune['correct_singletons']}/{tune['singletons']} | {m['correct_singletons']}/{m['singletons']} |

Held-out false-positive pairs: {m['fp']:,}; false-negative pairs, including blocking misses: {m['fn']:,}. Always predicting empty gives held-out macro-F0.5 {metrics['held_out_empty_baseline']:.6f}. Micro pair precision and recall are diagnostics; they are not the challenge's macro score.

Pair-level classifier errors are recorded in `reports/validation_errors.json`; blocking misses contribute to reported recall but do not appear as scored pairs in that file. Potential weaknesses include short or common names without an address, severe spelling changes that share no blocking key, false similarities at shared addresses, and candidate caps on common names. Error records should be consulted rather than assuming every error has one of those causes.

Post-assessment inspection of the supplied records found the following overlapping error characteristics. These are descriptions, not proven causes. Among {analysis['false_positive']['total']} false-positive pairs, {analysis['false_positive']['high_address_similarity']} had address token-set similarity >= 0.90 and {analysis['false_positive']['numeric_disagreement']} disagreed on at least one numeric token. Among {analysis['scored_false_negative']['total']} scored false-negative pairs, {analysis['scored_false_negative']['missing_target_address']} had missing target addresses and {analysis['scored_false_negative']['numeric_disagreement']} had numeric disagreements. An additional {ceiling['fn']} held-out true links were missed by blocking. Detailed counts and representative raw record pairs are in `reports/error_analysis.json`. This inspection did not change the model or threshold.

The approximate entity-level 95% normal confidence interval for the held-out macro score is [{m['approximate_95pct_ci'][0]:.6f}, {m['approximate_95pct_ci'][1]:.6f}]. It assumes independent sampled reference entities and does not account for France distribution shift or all sources of model uncertainty.

Country-specific held-out results:

```json
{json.dumps(metrics['held_out_by_country'],indent=2)}
```

Country-transfer diagnostics (the indicated country's fitting and tuning labels are excluded; its audit entities are used only for assessment):

```json
{json.dumps(metrics['country_transfer_diagnostics'],indent=2)}
```

These diagnostics do not establish France accuracy. France has no provided training labels and the test set has no ground truth. No test accuracy or leaderboard score is claimed. The threshold/model were not selected using a public leaderboard score.

## 6. Conclusion

The pipeline provides actual trained-model inference, a faithful candidate audit file and reproducible evaluation with singleton-aware macro-F0.5. It prioritizes feasible retrieval and precision on laptop hardware. Performance is bounded by blocking misses, the representativeness of the supervised sample and unseen-country distribution shift; no perfect-output claim is made.

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/pipeline.py` is the end-to-end entry point. The README contains exact commands; dependencies are pinned in requirements.txt. The package includes the trained model, metrics, split membership, data hashes, errors, validation logs, and both output files. Temporary target indexes and feature caches are rebuilt from the original supplied data and are not shipped.

### B. Final Output and Validation

- Reference rows: {stats['entities']:,}.
- Predicted links: {stats['matches']:,}.
- Empty prediction rows: {stats['singletons']:,}.
- Candidate pairs: {stats['candidates']:,}.

The original validator is run on the matching output. The bounded-memory strict validator checks both full files, every target ID's existence, reference coverage and uniqueness, duplicate-free lists, correct prefixes and headers, and match inclusion in candidates. Logs are included in reports. The final test outputs are predictions, not labeled test evaluations.
'''
    (root/'Documentation_template.md').write_text(doc,encoding='utf8')
    manifest={}
    included=[root/'Documentation_template.md',*sorted((root/'output').glob('*.tsv'))]
    included += [f for f in sorted(code.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and '.venv' not in f.parts]
    for f in included:
        with f.open('rb') as stream: manifest[str(f.relative_to(root)).replace('\\','/')]=hashlib.file_digest(stream,'sha256').hexdigest()
    (root/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)); included.append(root/'SHA256SUMS.json')
    team_slug=''.join(c if c.isalnum() or c in '-_' else '_' for c in a.team).strip('_')[:80] or 'BusinessEntityResolution'
    archive=root/f'{team_slug}_submission.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in included: z.write(f,str(f.relative_to(root)))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    print(archive)

if __name__=='__main__': main()
