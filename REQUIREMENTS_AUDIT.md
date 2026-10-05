# Submission requirements and training evidence

This audit concerns the original submission and the independently repeated offline run. The user reports an Unstop leaderboard score of **0.898**. That is different from the local held-out training-data score of **0.9274622785**. Neither number is a fabricated training score or a guarantee about the private leaderboard.

| Requirement | Evidence and result |
|---|---|
| Actual supervised training | The supplied terminal log records fitting 1,185,836 candidate pairs, including 59,629 positive pairs, for the development model. Validation binary log loss decreases from 0.0304959 at iteration 50 to 0.0162919 at iteration 450. |
| Independent offline reproduction | `artifacts/baseline_reproduction_proof.json` confirms that the original and repeated runs have identical model bytes, metrics, prediction statistics and matching-file SHA-256 hashes. |
| Correct evaluation formula | `pipeline.score` computes per-reference `1.25*TP/(0.25*true_count+predicted_count)`, awards 1 to a correctly empty singleton, and averages across reference entities. Unit tests cover singleton and non-singleton cases. |
| Held-out assessment | 6,000 reference entities were separate from model fitting and threshold tuning. The local score is 0.9274622785; precision and recall reported beside it are micro pair diagnostics, not the macro competition score. |
| Refit final model | The final model was genuinely refitted on 30,000 sampled reference entities and 1,982,636 candidate pairs. This is sampled training, not training on every one of the 2.2 million labeled reference entities. |
| Every test Source 1 entity | Full validation reports exactly 1,732,544 rows. |
| Unseen country coverage | Rows include US: 663,106; India: 809,986; France: 259,452. Countries are open strings; the code does not restrict them to the training-country set. |
| Required TSV format and headers | `validate_streaming.py` verifies exact headers, two columns, and one row per reference in source order. |
| Valid Source 2/3 IDs, no duplicate list entries | Full streaming validation checks every candidate target against the test source IDs and verifies duplicate-free lists and valid prefixes. |
| Matching is a subset of candidates | Full streaming validation reports `all_matches_are_candidates: true`. |
| Faithful final candidate set | `batch_pairs` creates the final filtered feature rows used by inference; the prediction worker writes the IDs associated with exactly those rows. A regeneration check reproduced the first 500 entities and all 35,084 scored pairs exactly. |
| Official validator | `artifacts/validation_official.log` records PASS for the matching file with target-ID existence enabled. Its memory-heavy candidate loading was avoided; the separate full streaming check passed both files. |
| MIT/Apache model and size limit | LightGBM's MIT license is included. The trained model has 450 trees and 13,950 leaves, well below 8 billion parameters. The project/model are distributed under MIT. |
| No external business lookup | Pipeline source has no business API, geocoder, internet identity lookup, or downloaded business-data dependency. All record retrieval uses the provided source files. Input SHA-256 hashes are recorded in the data inventory. |
| Runnable final package | `BusinessEntityResolution_submission.zip` includes both outputs, source code, pinned requirements, README, model, methodology and reports. Required archive entries and ZIP CRC integrity were checked. |

## Remaining limitations and administrative items

- The original blocker has held-out recall 0.955988 and an oracle macro-F0.5 ceiling of 0.984012. Even a perfect classifier cannot overcome its missing candidates.
- Country-specific local performance differs: US macro-F0.5 0.953641 and India 0.887920. France has no provided training labels. The precise cause of the leaderboard gap cannot be established without its hidden labels.
- The LightGBM deprecation and feature-name warnings in the pasted log did not abort training or inference. They are not a statement that the score is invalid.
- Team name and member names were not supplied. The original ZIP uses the temporary team name `BusinessEntityResolution` and marks member information as not supplied.
- A new, improved output must be validated and uploaded separately; the portal score cannot be inferred from a local validation score.
- A score near 0.99 is an optimization target, not a promised result. No test labels, external identity lookups, or leaderboard-label reconstruction are used to pursue it.

The revised experiments are kept in `artifacts_v3/` and `output_v3/`; the original scored submission is preserved.

## Revised v3 assessment (26 September 2026)

Actual training and model selection have completed on a new sample, disjoint from all original 30,000 supervised references. The split contains 40,000 fitting, 10,000 tuning, and 10,000 assessment references. Final refitting and full test inference are separate subsequent stages; these results do not imply that revised output validation or packaging has finished.

| Metric on the same fresh 10,000 references | Original | Revised |
|---|---:|---:|
| Macro-F0.5 | 0.928438 | 0.963971 |
| Micro pair precision | 0.975692 | 0.989225 |
| Micro pair recall | 0.860317 | 0.931957 |
| False-positive links | 739 | 350 |
| False-negative links, including blocking misses | 4,816 | 2,346 |
| Correct singletons | 523/568 | 532/568 |

Evidence: `artifacts_v3/metrics.json`, `train.log`, `assessment_model.txt`, and `split_integrity.json`. The revised assessment model has 800 trees, maximum 63 leaves, 120 features and a tuning-selected threshold of 0.625. The complete supervised matrix contains 4,830,066 pairs. Six unit tests passed. `cache_equivalence.log` verifies identical raw/cached candidates, groups and all features for 100 test references / 8,290 pairs.

The revised held-out blocking recall is 0.968183 and its oracle macro-F0.5 ceiling is 0.987327. Therefore this version has **not demonstrated 0.99 macro-F0.5**, even though its pair precision is close to 0.99. Its revised Unstop score is still unknown. `finish_enhanced.py` runs inference, both validation checks, and packaging sequentially after successful final training; `artifacts_v3/finish.log` records its status.
