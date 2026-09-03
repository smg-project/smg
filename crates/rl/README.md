# smg-rl

RL control plane for the Shepherd Model Gateway. Enabled with `--enable-rl`.

| Route | Purpose |
|---|---|
| `GET /v1/rl/workers` | list workers with engine, topology, health, weight version, capabilities |
| `GET /v1/rl/workers/{id}` | one worker |
| `GET\|POST /v1/rl/workers/{id}/engine/{path}` | proxy one engine-native route to one worker |
| `GET\|POST /v1/rl/engine/{path}?selector=...` | the same call fanned out to every matching worker |

Flags: `--enable-rl`, `--rl-control-timeout-secs` (600), `--rl-fanout-concurrency` (32).
Recommended RL launch profile: `--enable-rl --disable-health-check --disable-circuit-breaker --request-timeout-secs 14400`.

## Python client

```python
from smg.rl import RL
rl = RL("http://smg:30000")                     # api_key="..." if control-plane auth is on
for w in rl.workers():
    print(w.id, w.engine, w.tp_size, w.health, w.weight_version)
rl.call(w.id, "server_info", method="GET")
rl.fanout("pause_generation", selector="engine=sglang")
rl.fanout("update_weights_from_disk", {"model_path": "/ckpt/42", "weight_version": "42"},
          selector="engine=sglang")
rl.fanout("continue_generation", selector="engine=sglang")
```

`fanout` raises `FanoutError` (with `.result.failed`) unless `allow_partial=True`.
Keep the client connected for the whole call: the gateway cancels outstanding
engine calls when the caller disconnects.

Design: `docs/superpowers/specs/2026-09-03-rl-m1-discovery-passthrough-design.md`.
