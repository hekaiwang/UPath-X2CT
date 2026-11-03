# Baseline results

This release includes main-comparison results for X2CT-GAN, PerX2CT, BX2S (HU),
XraySyn (HU), and DVG-Diffusion on exactly the three paper cohorts. It does not
bundle their third-party source trees or model weights. The trainable code and
weights supplied here are for UPath-X2CT.

`results/baseline_sources.json` records upstream URLs and revisions. The
`*_per_case.csv` files contain the five baselines and final UPath-X2CT, and
`main_results.csv` contains patient-level means. The unit test independently
recomputes these means from the per-case evidence.

BX2S and XraySyn are project adaptations producing HU volumes. XraySyn uses one
AP input; all other listed methods use AP and LAT. DVG-Diffusion is the project's
v1.4 retraining with the decoder unfrozen, not an unmodified official checkpoint.
These values should be described as common-protocol project comparisons rather
than numbers copied from the upstream papers.
