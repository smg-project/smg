# smg-rl

RL control plane for the Shepherd Model Gateway. Enabled with `--enable-rl`.

| Route | Purpose |
|---|---|
| `GET /v1/rl/workers` | list workers with engine, topology, health, weight version, capabilities |
| `GET /v1/rl/workers/{id}` | one worker |
| `GET\|POST /v1/rl/workers/{id}/engine/{path}` | proxy one engine-native route to one worker |
| `GET\|POST /v1/rl/engine/{path}?selector=...` | the same call fanned out to every matching worker |

Response bodies are the `openai_protocol::rl` types (`RlWorkersResponse`,
`RlWorkerEntry`, `RlCallOutcome`, `RlFanoutResponse`), registered in
`clients/openapi-gen` so the generated SDKs carry them. `GET /v1/rl/workers`
reports `protocol_version` (currently 1), bumped only for incompatible changes.

Only HTTP workers can be proxied. A gRPC or ZMQ worker that matches a selector is
reported in `failed[]` as `unsupported_connection_mode` (HTTP 422 on the per-worker
route), so a fan-out over a mixed fleet answers 207 and `smg.rl.RL.fanout` raises
`FanoutError` unless `allow_partial=True`.

Flags: `--enable-rl`, `--rl-control-timeout-secs` (600), `--rl-fanout-concurrency` (32).
Recommended RL launch profile: `--enable-rl --disable-health-check --disable-circuit-breaker --request-timeout-secs 14400`.

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
