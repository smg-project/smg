# Engine and framework API drift log

Record every place the live engine or framework API differed from the
planning docs, with date, engine version, and what was done.

| Date | Engine/framework | Expected | Observed | Action |
|---|---|---|---|---|
| 2026-09-03 | SGLang 0.5.15.post1 | `/get_weight_version` | 404; version is in `/model_info` | discovery reads the `weight_version` registration label |
| 2026-09-03 | SGLang 0.5.15.post1 | `POST /pause_generation` with no body succeeds | 400 Bad Request; FastAPI requires a JSON body, `{}` is enough | callers must send `{}` on bodyless control routes; `examples/rl/refit_from_disk.py` needs `rl.fanout("pause_generation", {}, ...)` (the e2e test already passes `json={}`) |
| 2026-09-10 | SGLang 0.5.15.post1 | discovery reports a real version before the first refit | registration reports `weight_version: default` (the placeholder) until a refit lands | seeded as unversioned (`Version::from_label` treats `default` as `None`) so a first refit is not out-ordered by the placeholder |
| 2026-09-10 | vLLM (version unpinned) | refit routes carry a version field the passthrough observer can read | documented weight-transfer bodies carry no version field | passthrough observes control state only for vLLM; versions must come through the API (`set_version`/`set_fleet_version`); verify against a live vLLM when one is available |
| 2026-09-14 | SGLang 0.5.19 | `POST /release_memory_occupation` on a busy engine pauses it | a scheduler assertion aborts the whole engine process; idle, the route answers 200 in 3 ms | the proxy reports 502 `upstream_unreachable` and observes nothing (the table keeps `control: active`); pause the fleet before releasing memory, as the refit sequence already does |
| 2026-09-14 | SGLang 0.5.19 | `POST /flush_cache` answers 200 under traffic (0.5.15 did) | 400 on a busy server; a fleet flush under load is a 207 with every worker in `failed[]` | `flush_cache: true` inside a refit does not reliably flush under load; the version change still resets the gateway's cache-aware entries, so routing is unaffected |
