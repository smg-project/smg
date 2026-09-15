# smg-rl

RL control plane for the Shepherd Model Gateway. Enabled with `--enable-rl`.

| Route | Purpose |
|---|---|
| `GET /v1/rl/workers` | list workers with engine, topology, health, weight version, capabilities |
| `GET /v1/rl/workers/{id}` | one worker |
| `GET\|POST /v1/rl/workers/{id}/engine/{path}` | proxy one engine-native route to one worker |
| `GET\|POST /v1/rl/engine/{path}?selector=...` | the same call fanned out to every matching worker |
| `POST /v1/rl/workers/{id}/version` | record the weight version one worker holds (SMG-local; no engine call) |
| `POST /v1/rl/workers/{id}/state` | mark one worker `active`, `paused`, or `asleep` for routing (SMG-local) |
| `POST /v1/rl/version?selector=...` | the version write fanned out to every matching worker |
| `POST /v1/rl/state?selector=...` | the state write fanned out to every matching worker |

Response bodies are the `openai_protocol::rl` types (`RlWorkersResponse`,
`RlWorkerEntry`, `RlCallOutcome`, `RlFanoutResponse`), registered in
`clients/openapi-gen` so the generated SDKs carry them. `GET /v1/rl/workers`
reports `protocol_version` (currently 1), bumped only for incompatible changes.

Only HTTP workers can be proxied. A gRPC or ZMQ worker that matches a selector is
reported in `failed[]` as `unsupported_connection_mode` (HTTP 422 on the per-worker
route), so a fan-out over a mixed fleet answers 207 and `smg.rl.RL.fanout` raises
`FanoutError` unless `allow_partial=True`.

Every error answers `{"error": <code>, "message": ...}` with a stable code:
`invalid_body` (400, a malformed or absent JSON body), `invalid_version` (400,
a `weight_version` that is empty, over 128 bytes, or not printable ASCII),
`invalid_version_policy` (400), `invalid_engine_path` (400),
`selector_required` / `invalid_selector` / `no_workers_match` (400),
`worker_not_found` (404), `unsupported_connection_mode` (422),
`upstream_unreachable` (502), and `upstream_timeout` (504).

Flags: `--enable-rl`, `--rl-control-timeout-secs` (600), `--rl-fanout-concurrency` (32),
`--rl-version-policy` (any).
Recommended RL launch profile: `--enable-rl --disable-health-check --disable-circuit-breaker --request-timeout-secs 14400`.

## Versions and control state

SMG keeps a side table of `(weight_version, control_state)` per engine, keyed
by base URL (DP ranks share an entry). Three sources feed it, last write wins:
the refit and pause/sleep calls proxied through `/v1/rl` (`version_source:
passthrough`), the explicit API above (`version_source: api`), and the
`weight_version` registration label at discovery time (`version_source:
registration`; SGLang's `default` placeholder seeds as unversioned). `GET
/v1/rl/workers` reports the table's current view: `weight_version` (`null`
when unversioned), `version_source` (`null` when unversioned), and `control`
(`active`, `paused`, or `asleep`).

With `--enable-rl`, a worker the table shows `paused` or `asleep` is never
selected for inference; `--rl-version-policy` (`any`, `latest-only`,
`min-version:<v>`, `max-staleness:<k>`, or the per-request
`x-smg-version-policy` header) further narrows the candidates by version
against the model's known maximum.

Two response headers carry the table's view of what served a request:
`x-smg-weight-version` on every response from a versioned engine, and
`x-smg-mixed-version: true` on a buffered `/generate` response whose tokens
spanned two versions.

## Python client

```python
from smg.rl import RL, paused
rl = RL("http://smg:30000")                     # api_key="..." if control-plane auth is on
for w in rl.workers():
    print(w.id, w.engine, w.tp_size, w.health, w.weight_version)
rl.call(w.id, "server_info", method="GET")
with paused(rl, "engine=sglang"):               # pause_generation ... continue_generation
    rl.fanout("update_weights_from_disk", {"model_path": "/ckpt/42", "weight_version": "42"},
              selector="engine=sglang")
```

`paused` always resumes: a pause that failed on some engines, or a refit that
failed, still gets `continue_generation` so nothing stays paused. SGLang's
`pause_generation` and `continue_generation` require a JSON body; `paused`
sends `{}`, and so must direct `fanout` calls (a bodyless POST returns 400).

`fanout` raises `FanoutError` (with `.result.failed`) unless `allow_partial=True`.
Keep the client connected for the whole call: the gateway cancels outstanding
engine calls when the caller disconnects.

Design: `docs/superpowers/specs/2026-09-03-rl-m1-discovery-passthrough-design.md`.
