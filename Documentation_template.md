# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Chakravyuh  
**Team Members:** Kommu Hemasree (Leader), Kancharla Sunaina, Vakada Kushal, Vakkalanka Sai Shashank  
**Submission Date:** 2026-10-02

## 1. Executive Summary

A supervised LightGBM cascade resolves noisy business names and addresses from
the supplied records. The final v7 classifier rescores preliminary accepted
links with source-specific thresholds, exporting its exact scored candidate
set and one prediction row for every test reference.

## 2. Methodology

### 2.1 Problem Analysis

Name abbreviations, spelling/script variants, reordered or missing address
components, and ambiguous common names require multiple complementary signals.
The test set additionally contains France without training labels; country is
treated as an open string. Full dataset statistics are in reports/data_inventory.json.

### 2.2 Solution Strategy

**Approach Type:** Blocking plus supervised classifier cascade.  
**Core Innovation:** Population/competitor features, deterministic cross-script
retrieval and conservative final source-specific pair acceptance.

## 3. Candidate Generation (Blocking)

Exact/name-token/address/numeric keys and supplemental phonetic/romanized keys
reduce the full comparison space. Oversized blocks and ranked caps limit cost.
The last stage considers precisely 5,702,473 v4-positive pairs. Candidate misses
remain false negatives during assessment; perfect recall is not claimed.
Complete blocking stages, caps and the candidate audit are detailed below.

## 4. Matching Model

**Model type:** MIT-licensed LightGBM tree classifiers trained from scratch.  
**Features:** 154 final name/address similarities, token rarity/uniqueness,
numeric-order, phonetic/romanized and competing-reference signals.  
**Threshold selection:** Macro-F0.5 on 30,000 separate development references;
final thresholds 0.65 (Source 2), 0.75 (Source 3). No final-stage refitting.

## 5. Results & Error Analysis

Independent 10,000-reference macro-F0.5: 0.9740168111.
Baseline on the same references: 0.9728759304.
Missing addresses/common names can cause false merges; cross-script variation,
large-block caps and conservative final thresholds can cause missed matches.
These are offline scores; the v7 leaderboard score is not yet supplied.

## 6. Conclusion

The final cascade improves assessed precision while retaining complete test
reference coverage. Its exact final candidate inputs and measured validation
are exported for audit. Hidden-test accuracy and final ranking remain determined
by organizer evaluation.

## Appendix

### A. Code Artefacts

All source is in code/business_entity_resolution/src/. README.md provides exact
environment, training, offline evaluation, prediction and validation commands.
requirements.txt pins dependencies; model and license folders are included.

### B. Additional Results and Full Technical Methodology

# Business Entity Resolution: source-calibrated positive-link cascade (v7)

Team: Chakravyuh. Members: Kommu Hemasree (Leader), Kancharla Sunaina, Vakada Kushal, Vakkalanka Sai Shashank.

## Final method

The v4 submission achieved a participant-reported public score of 0.962. This version retains its complete preliminary prediction pipeline and then applies the frozen v4 LightGBM classifier to EVERY previously accepted pair. It does not add links. Source 2 and Source 3 use separately calibrated acceptance thresholds: 0.65 and 0.75, respectively. Source comes from the allowed ID prefix; numerical ID contents are never features. The country remains an open normalized string. All countries and every reference, including France, are exported.

This is model agreement with a calibrated final classifier: prior v3/v4 acceptance is required, and the final frozen v4 classifier must also accept the pair. The model uses the same 154 string, rarity, numeric-order and competing-reference features as v4, with the same neighbor selection and label-free population statistics. The final model is not retrained for this stage. The input for this last ML stage is the set of v4-positive pairs, so `candidate_pairs.tsv` now contains EXACTLY those pairs, including links rejected by the last classifier. Earlier larger candidate pools belong to preliminary stages and are not exported as final-stage candidates. Empty preliminary predictions have an empty final candidate list. This deliberately trades some recall for precision; all test outputs remain many-to-one from targets to the deduplicated reference framework, without imposing a forced target assignment.

An ID-to-offset lookup accelerates retrieval of the supplied records. A 64-bit hash locates a range, but the complete original ID is always checked, including collision handling. This is only a join/index mechanism, never a predictive ID feature. No external identity lookup, API, geocoder, business dataset or internet augmentation is used.

## Development and independent assessment

Thresholds are selected on 30,000 references previously held out from all model fitting. These are explicitly development data for v7, not a new independent score. The grid for each target source is 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.575, 0.65, 0.75 and 0.85; the pair maximizing macro-F0.5 is frozen. A separate new 10,000-reference sample excludes every reference from the original 30,000 sample, v3 60,000 sample, v4 60,000 sample and the 30,000 development sample. It is evaluated once with the frozen models and thresholds. No new fitting or tuning occurs on this final sample.

Both the existing v4 cascade and this exact rescoring cascade are evaluated on the same fresh references. Automatic production requires a positive lower endpoint of the approximate 95% paired improvement interval.

```json
{
  "baseline": {
    "macro_f0.5": 0.9728759303772009,
    "approximate_95pct_ci": [
      0.9708337643137189,
      0.9749180964406828
    ],
    "micro_precision": 0.991229131381563,
    "micro_recall": 0.9463501963501963,
    "tp": 32774,
    "fp": 290,
    "fn": 1858,
    "entities": 10000,
    "singletons": 553,
    "correct_singletons": 535
  },
  "proposed": {
    "macro_f0.5": 0.9740168110921779,
    "approximate_95pct_ci": [
      0.9720256510076666,
      0.9760079711766891
    ],
    "micro_precision": 0.9948747673815552,
    "micro_recall": 0.9416435666435666,
    "tp": 32611,
    "fp": 168,
    "fn": 2021,
    "entities": 10000,
    "singletons": 553,
    "correct_singletons": 541
  },
  "policy": {
    "source2_threshold": 0.65,
    "source3_threshold": 0.75,
    "macro_f0.5": 0.974406961160031
  },
  "paired_improvement": {
    "mean": 0.0011408807149769764,
    "approximate_95pct_ci": [
      0.00035802594052297546,
      0.0019237354894309773
    ]
  },
  "trials": [],
  "note": "Fresh validation with frozen thresholds."
}
```

The score is per-reference macro-F0.5, including singleton credit. Pair precision and recall are diagnostics. France has no provided labels, so its performance is not claimed from this local evaluation. The participant selected the uploaded v7 file for final submission; its leaderboard score has not yet been supplied for this document. A score of 0.992 or top-50 placement is not promised. The previous unsuccessful empty-recovery and simple ambiguity-rule experiments are not applied to this submitted file.

## Output verification

```json
{
  "prediction_stats": {
    "entities": 1732544,
    "candidates": 5702473,
    "matches": 5641306,
    "singletons": 105433,
    "removed_matches": 61167
  },
  "validation": {
    "status": "PASS",
    "entities": 1732544,
    "candidate_pairs": 5702473,
    "matched_pairs": 5641306,
    "countries": {
      "US": 663106,
      "France": 259452,
      "India": 809986
    },
    "all_target_ids_exist": true,
    "all_matches_are_candidates": true
  }
}
```

A benchmark also rescored 6,088 already-v4-accepted links from the first 10,000 test references without any unexpected rejection at the original threshold; its report is included. Both full output files passed streaming validation of IDs, source prefixes, reference coverage, headers, uniqueness and match inclusion in candidates. The official matching-file validator passed with ID existence checks. All final candidates are verified to equal the actual final classifier's input lists. The ZIP includes source, pinned dependencies, previously trained MIT-licensed models, methodology and reports. All models are far below 8 billion parameters. A SHA-256 manifest and ZIP CRC check verify package integrity.

## Exact reproduction

Use `src/reproduce_submission.py` with the exact installation and run commands in README.md. It runs all required baseline, v3, v4, development, fresh assessment and final test stages in an empty work root, using only the supplied seven data files and this code folder. Intermediate caches are regenerated. Original completed output hashes are in SHA256SUMS.json. Packaging checks original output identity and archive integrity; a new full from-scratch training run was not performed during packaging.

## Supplementary baseline architecture

The following describes the underlying v4 baseline. Its earlier test counts and assessment scores belong to v4; the final v7 counts and independent comparison appear above.

# Business Entity Resolution: v4 targeted correction cascade

Team: Chakravyuh. Members: Kommu Hemasree (Leader), Kancharla Sunaina, Vakada Kushal, Vakkalanka Sai Shashank.

## Methodology and data

The previous v3 submission scored 0.954 on the public leaderboard according to the participant. The new approach preserves that complete prediction set and replaces the final classifier for references selected by a deterministic cross-script gate. There are no internet business lookups, geocoders, registration services or external identity datasets. All candidate records, population statistics and labels come from the supplied challenge files.

The new supervised sample contains 60,000 Source 1 references, excluding every reference from the original 30,000 sample and the v3 60,000 sample. Seed 20260928 determines a country/singleton-stratified split: two thirds fitting, one sixth threshold/model selection, one sixth fresh assessment. All target records remain available for retrieval. Candidate misses count as false negatives. The final new model is refitted on the complete sample only after held-out assessment. Reusing label-free v3 reference-frequency indexes does not introduce held-out matching labels.

## Candidate retrieval and normalization

The original exact, token-combination, address and numeric hash indexes are supplemented with phonetic-name, name-token, name-prefix/suffix and romanized-address-pair indexes. Indic scripts are converted mechanically using Python Unicode character names, vowel-sign and virama handling. Phonetic folding reduces vowel and consonant-spelling variations. Latin accents are normalized too. This is a deterministic text transform, not a pretrained translation model or business-name dictionary. Transliteration is approximate and is only evidence for supervised matching, never an automatic merge rule.

Supplementary blocks larger than 500 records are skipped; at most 200 supplementary offsets are retained by shared-key count. They are unioned with the v3 initial candidates. Strong variants can expand retrieval. A cheap filter then retains plausible name/address or phonetic similarity, followed by the union of the top 60 candidates under four rankings: original compact name, address, combined similarity and phonetic name. At most 240 candidate pairs per corrected reference reach the new final classifier.

## Features and model

The new classifier uses 154 features: the previous 120 name/address, rarity, uniqueness, numeric and competing-reference features plus 34 romanized/phonetic fuzzy-similarity and ordered numeric features. Numeric evidence preserves first street/building numbers and alphanumeric units instead of relying only on an unordered number set. Two prespecified LightGBM configurations are compared on tuning data: 800 trees/63 leaves and 1,200 trees/127 leaves. Learning rate 0.035, minimum leaf size 50, L2 regularization 5 and feature fraction 0.9 are fixed. The threshold grid is 0.25 through 0.95 in steps of 0.025.

Selected new model: 1200 trees, maximum 127 leaves, threshold 0.5750000000000003. Supervised matrix: 8,082,420 pairs. The previous v3 model is also required for the cascade. Both models and project code are MIT licensed and far below 8 billion parameters; no pretrained model is used.

## Exact cascade rule and candidate audit

The gate selects a reference if supplementary retrieval finds an Indic-name candidate absent from its v3 predicted matches with phonetic token-sort similarity at least 80/100, or if a v3 predicted Indic-name match has phonetic token-sort similarity below 50/100. It uses records and v3 predictions, never labels. All other references retain their v3 final predictions and candidate sets. Selected references use the new model and its final scored candidate set. The candidate TSV therefore records the actual inputs to the final classifier branch used for each reference. Earlier gate and preliminary v3 computations are not claimed as final-model candidates for corrected references.

Test gate selection: 319,683 out of 1,732,544 references. Both source-2 and source-3 matches are many-to-one; no forced one-to-one assignment is used. Countries remain arbitrary strings. Every test reference, including France, is exported; unselected references are preserved.

## Offline evidence for this exact submitted cascade

The v3 model and the new cascade are evaluated on the SAME fresh references. The gate is applied using each assessment reference's actual v3 predictions and supplementary candidates.

```json
{
  "v3_same_holdout": {
    "macro_f0.5": 0.9658171726014594,
    "approximate_95pct_ci": [
      0.9635291141813624,
      0.9681052310215563
    ],
    "micro_precision": 0.9892277861561664,
    "micro_recall": 0.9305173561854029,
    "tp": 32141,
    "fp": 350,
    "fn": 2400,
    "entities": 10000,
    "singletons": 558,
    "correct_singletons": 530
  },
  "submitted_cascade_holdout": {
    "macro_f0.5": 0.9735730581701831,
    "approximate_95pct_ci": [
      0.9716379665798587,
      0.9755081497605076
    ],
    "micro_precision": 0.9908635604795872,
    "micro_recall": 0.9450797602848788,
    "tp": 32644,
    "fp": 301,
    "fn": 1897,
    "entities": 10000,
    "singletons": 558,
    "correct_singletons": 530
  },
  "gated_assessment_entities": 1603,
  "paired_improvement": {
    "mean": 0.007755885568723537,
    "approximate_95pct_ci": [
      0.0065052549752438234,
      0.00900651616220325
    ]
  },
  "new_model_on_all_assessment_entities_not_the_submitted_cascade": {
    "macro_f0.5": 0.9746779682130363,
    "approximate_95pct_ci": [
      0.9727902795265928,
      0.9765656568994798
    ],
    "micro_precision": 0.9900102613629504,
    "micro_recall": 0.9496829854375959,
    "tp": 32803,
    "fp": 331,
    "fn": 1738,
    "entities": 10000,
    "singletons": 558,
    "correct_singletons": 529
  },
  "new_blocking_ceiling": {
    "macro_f0.5": 0.9929488146674091,
    "approximate_95pct_ci": [
      0.9919398953749002,
      0.993957733959918
    ],
    "micro_precision": 1.0,
    "micro_recall": 0.9798500332937669,
    "tp": 33845,
    "fp": 0,
    "fn": 696,
    "entities": 10000,
    "singletons": 558,
    "correct_singletons": 558
  }
}
```

Macro-F0.5 averages 1.25*TP/(0.25*true_count+predicted_count) across reference entities, with 1 for a correctly empty singleton and 0 for a false merge on a singleton. Micro pair precision and recall are separate diagnostics. Local scores are not public/private leaderboard scores. The v4 public leaderboard score was 0.962, as reported by the participant. Neither 0.99 nor a rank is guaranteed. Remaining limitations include ambiguous businesses with missing addresses, approximate transliteration, blocked large candidate groups and unseen-country distribution shifts.

## Outputs and verification

```json
{
  "status": "PASS",
  "entities": 1732544,
  "candidate_pairs": 164269936,
  "matched_pairs": 5702473,
  "countries": {
    "US": 663106,
    "France": 259452,
    "India": 809986
  },
  "all_target_ids_exist": true,
  "all_matches_are_candidates": true
}
```

The bounded-memory validator checks both complete TSVs, all target-ID existence, unique reference coverage, duplicate-free lists, exact headers and match inclusion in final candidates. The official validator additionally checks the matching file; its optional candidate loading is omitted to control memory. The ZIP includes both TSVs, source, pinned dependencies, trained models, reports and this methodology. SHA-256 hashes cover every packaged input file; the archive is CRC-checked.

## Reproduction

Run from a workspace root with the supplied dataset. First reproduce v3 using `enhanced_pipeline.py --stage all` as documented in README.md; keep its `artifacts_v3` directory, models and prediction chunks. Then run `phonetic_pipeline.py --stage prepare`, followed by `--stage train`, using the same data directory and default artifact paths. Build test supplementary retrieval with `phonetic_index.build(Path('artifacts/test_index'))` and `phonetic_index.build_native(...)`. Finally run `hybrid_pipeline.py --stage all --data-dir DATASET`, validate `output_v4`, and run `package_hybrid.py`. Source for every component is included. Intermediate caches and full data are not packaged. A resumable deadline-aware runner is provided as `finish_phonetic.py`.

