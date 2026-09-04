# Engine and framework API drift log

Record every place the live engine or framework API differed from the
planning docs, with date, engine version, and what was done.

| Date | Engine/framework | Expected | Observed | Action |
|---|---|---|---|---|
| 2026-09-03 | SGLang 0.5.15.post1 | `/get_weight_version` | 404; version is in `/model_info` | discovery reads the `weight_version` registration label |
| 2026-09-03 | SGLang 0.5.15.post1 | `POST /pause_generation` with no body succeeds | 400 Bad Request; FastAPI requires a JSON body, `{}` is enough | callers must send `{}` on bodyless control routes; `examples/rl/refit_from_disk.py` needs `rl.fanout("pause_generation", {}, ...)` (the e2e test already passes `json={}`) |
