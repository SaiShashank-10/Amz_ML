# Chakravyuh — Amazon ML Challenge: Business Entity Resolution

This repository contains the completed v7 pipeline, trained LightGBM models,
pinned dependencies, methodology and assessment reports.

Team: Kommu Hemasree (Leader), Kancharla Sunaina, Vakada Kushal,
and Vakkalanka Sai Shashank.

## Reproduce the solution

See [the pipeline README](code/business_entity_resolution/README.md) for exact
installation, training, offline assessment, prediction and validation commands.
See [the methodology](Documentation_template.md) for blocking and model details.
The supplied challenge dataset is required separately.

All source is under code/business_entity_resolution/src/. The models/, models_v3/
and models_v4/ folders contain completed model artifacts. reports*/ contains
assessment and validation evidence. requirements.txt pins dependencies; LICENSE
and licenses/ contain licensing notices.

Large generated indexes, feature matrices, prediction checkpoints, TSV outputs
and submission ZIPs are excluded from Git. They remain in the original local
workspace, including Chakravyuh_submission.zip. They can be regenerated using
the included code and supplied dataset.

The v7 independent offline macro-F0.5 is 0.9740168111 on 10,000 held-out
references. This is not a test leaderboard score; the v7 leaderboard score
has not been supplied. No score or rank is guaranteed.

The publishable branch starts with clean history so earlier generated files
are not uploaded. Original history is retained in a local backup branch.
