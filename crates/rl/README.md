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

## Control endpoints

Control calls go to a worker's **control endpoint**, not necessarily its data
transport. An HTTP worker is controlled through itself; an `rl.control_url`
label on an HTTP worker is ignored. A gRPC or ZMQ worker needs the
`rl.control_url` label: TokenSpeed engines advertise it in server
info (SMG's discovery turns it into the label), and any worker can be given one
at `POST /workers` or through the worker update route. A wildcard bind host in
the advertised URL (`0.0.0.0`, `::`) is replaced by the worker's own host. The
worker's `api_key` is sent as the bearer to the control endpoint. The label is
trusted: whatever host `rl.control_url` names receives the worker's `api_key`
as a bearer, so only engines and operators you trust may set it. A worker with
no control endpoint is reported in `failed[]` as `no_control_endpoint` (HTTP
422 on the per-worker route), so a fan-out over such a fleet answers 207 and
`smg.rl.RL.fanout` raises `FanoutError` unless `allow_partial=True`.

Capabilities come from the `rl.*` labels when an engine (or operator) supplies
them (`"source": "label"`), else from the built-in table (`"source": "static"`).

Flags: `--enable-rl`, `--rl-control-timeout-secs` (600), `--rl-fanout-concurrency` (32).
Recommended RL launch profile: `--enable-rl --disable-health-check --disable-circuit-breaker --request-timeout-secs 14400`.
TokenSpeed rollout engines: `python3 -m smg_grpc_servicer.tokenspeed --model … --port 30000 --rl-control-host <reachable> --rl-control-port 30400 [--rl-control-api-key …]`, registered as `grpc://host:30000`. TokenSpeed refits are trainer-driven over NCCL (`init_weights_update_group` → `update_weights_from_distributed` → `destroy_weights_update_group`, each proxied per worker through `/v1/rl`); `update_weights_from_disk` answers HTTP 501 on TokenSpeed. See `docs/guides/rl-tokenspeed.md`.

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
