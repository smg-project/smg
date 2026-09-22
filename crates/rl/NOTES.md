# Engine and framework API drift log

Record every place the live engine or framework API differed from the
planning docs, with date, engine version, and what was done.

| Date | Engine/framework | Expected | Observed | Action |
|---|---|---|---|---|
| 2026-09-03 | SGLang 0.5.15.post1 | `/get_weight_version` | 404; version is in `/model_info` | discovery reads the `weight_version` registration label |
| 2026-09-03 | SGLang 0.5.15.post1 | `POST /pause_generation` with no body succeeds | 400 Bad Request; FastAPI requires a JSON body, `{}` is enough | callers must send `{}` on bodyless control routes; `examples/rl/refit_from_disk.py` needs `rl.fanout("pause_generation", {}, ...)` (the e2e test already passes `json={}`) |
| 2026-09-21 | TokenSpeed d1464d8f | `flush_cache` on GET and POST like SGLang | GET only on the in-engine app | TokenSpeed PR adds POST; older builds need `method="GET"` |
| 2026-09-21 | TokenSpeed d1464d8f | bodyless POST → 400 | six control routes answer 500 on an empty/malformed body | TokenSpeed PR moves body parsing inside the guard; older builds: always send a JSON object |
| 2026-09-21 | TokenSpeed d1464d8f | `pause_generation {"mode": …}` | mode ignored, always `wait`; `keep` unreachable | TokenSpeed PR honors `mode`; static row advertises `wait,abort` only |
| 2026-09-21 | TokenSpeed d1464d8f | `update_weights_from_tensor` end to end | route exists, CUDA-IPC receive path not implemented | engine advertises `rl.update_from=disk,distributed` |
| 2026-09-22 | TokenSpeed d1464d8f | `update_weights_from_disk`/`_tensor` end to end | the routes exist but the scheduler raises `NotImplementedError` and dies (engine down); the companion branch now answers 501 and advertises `distributed` only | static row lists `distributed`; refit TokenSpeed with `update_weights_from_distributed` (see `examples/rl` once the trainer-side example lands) |
