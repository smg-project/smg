# Engine and framework API drift log

Record every place the live engine or framework API differed from the
planning docs, with date, engine version, and what was done.

| Date | Engine/framework | Expected | Observed | Action |
|---|---|---|---|---|
| 2026-09-03 | SGLang 0.5.15.post1 | `/get_weight_version` | 404; version is in `/model_info` | discovery reads the `weight_version` registration label |
