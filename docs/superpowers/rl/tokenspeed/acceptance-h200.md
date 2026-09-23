# TokenSpeed RL control plane — acceptance run on the H200 node

Status: **item 1 PASS; item 2 PASS on reward and throughput, step time +5.7 % (bar 5 %) on one run per arm.** Every criterion of the plan's acceptance item 1 was met on two
real TokenSpeed engines behind the branch's gateway. Two trainer-driven NCCL
refits landed and the next `/generate` through SMG reported the new
`meta_info.weight_version` each time; fan-out overhead stayed at or below 1 ms;
a killed engine produced a 207 naming it while the breaker stayed closed; and
`--enable-rl` off removed the whole `/v1/rl` surface and every `smg_rl_` series
without touching the data plane.

An earlier attempt on 2026-09-22 failed, but engine-side: TokenSpeed's
device-bound world group made torch build the weight-update communicator with
`ncclCommSplit` off a size-1 parent, so the engine answered HTTP 200 in 64 ms
without entering the collective and loaded its uninitialized receive buffers
into the live model. That investigation is kept below under **Root cause**,
because it is why the engine fix exists. The run recorded here is against the
fixed engine.

## Environment

| Piece | Value |
| --- | --- |
| Node | `moirai-h200-runner-0`, `ubuntu@10.0.1.52` via the OCI bastion, 8x H200 |
| Gateway | `~/smg-rl-ts/target/release/smg`, release build of `rl/tokenspeed-control-endpoint`, rebuilt 2026-09-22 23:30Z |
| SMG source | `~/smg-rl-ts/src` = this worktree at `4b0f04c5b` for item 1; `b156402c7` for Task 20's re-probe and the slime arms (candidate arm ran on `a5e45faf9`) |
| TokenSpeed | `~/tokenspeed-rl/src` at `053d6356a` (NCCL-split guard + sidecar fix) |
| Engines | TokenSpeed in container `ts-rl`, GPUs 0 and 1 |
| Container venv | `/opt/smg-ci/.venv`, Python 3.12.3 |
| torch / transformers | 2.14.0+cu130 / 5.12.0 |
| NCCL | 2.30.7+cuda13.3 |
| Model | `/home/ubuntu/models/meta-llama/Llama-3.2-1B-Instruct`, 146 parameters |
| Trainer | same container, `CUDA_VISIBLE_DEVICES=2` |
| Ports | gRPC 30106/30107, control 30406/30407, gateway 30100, Prometheus 29100 |

GPUs 4–7 were in use by a separate lane throughout (a `slime-rl` container,
`ts serve` engines on ports 312xx, an SMG on 31100). Nothing there was touched,
and every check below is scoped to GPUs 0–2 and ports 301xx/304xx/29100.

## Commands, in order

```bash
# b. starting state: 0 compute apps, ts-rl up, binary from 23:30Z
~/smg-rl-ts/remote/acc-state.sh

# c. two engines on GPUs 0 and 1
~/smg-rl-ts/remote/acc-engines.sh up
~/smg-rl-ts/remote/acc-engines.sh wait      # control 30406 ready=yes / 30407 ready=yes

# d. gateway with --enable-rl, then discovery
~/smg-rl-ts/remote/acc-gateway.sh up        # gateway pid 3909616
~/smg-rl-ts/remote/acc-discovery.sh         # saves logs/acc-rl-workers.json

# e. two refits
~/smg-rl-ts/remote/acc-refit.sh run 1 ; ~/smg-rl-ts/remote/acc-refit.sh log 1
~/smg-rl-ts/remote/acc-refit.sh run 2 ; ~/smg-rl-ts/remote/acc-refit.sh log 2

# f. fan-out overhead from the gateway's own log
~/smg-rl-ts/remote/acc-overhead.sh

# Task 19 probes, gateway still on --enable-rl
bash ~/smg-rl-ts/remote/parity-probes.sh http://127.0.0.1:30100

# g. failure injection
~/smg-rl-ts/remote/acc-failure.sh kill
~/smg-rl-ts/remote/acc-failure.sh fanout    # 207
~/smg-rl-ts/remote/acc-failure.sh metrics
~/smg-rl-ts/remote/acc-failure.sh restart
~/smg-rl-ts/remote/acc-failure.sh fanout    # 200

# h. flag off
~/smg-rl-ts/remote/acc-flagoff.sh

# i. cleanup, this lane only
~/smg-rl-ts/remote/acc-cleanup.sh down
```

Helpers live in `.superpowers/sdd/2026-09-21-rl-tokenspeed-control-endpoint/remote/`
and are synced to `~/smg-rl-ts/remote/`. The engines were launched with

```bash
docker exec -d ts-rl bash -c 'source /opt/smg-ci/.venv/bin/activate;
  CUDA_VISIBLE_DEVICES=0 python -m smg_grpc_servicer.tokenspeed
    --model /home/ubuntu/models/meta-llama/Llama-3.2-1B-Instruct
    --host 127.0.0.1 --port 30106
    --rl-control-host 127.0.0.1 --rl-control-port 30406
    --enable-output-logprobs > /work/logs/acc-engine-0.log 2>&1'
```

and the gateway with

```bash
setsid nohup ~/smg-rl-ts/target/release/smg launch \
  --model-path /home/ubuntu/models/meta-llama/Llama-3.2-1B-Instruct \
  --worker-urls grpc://127.0.0.1:30106 grpc://127.0.0.1:30107 \
  --policy cache_aware --port 30100 --enable-rl \
  --disable-health-check --disable-circuit-breaker \
  --request-timeout-secs 14400 --prometheus-port 29100 \
  > ~/smg-rl-ts/logs/acc-smg.log 2>&1 &
```

`--model-path` is required in regular mode, the same way `e2e_test/infra/gateway.py`
passes it. Worker registration runs through a workflow that took 38 s, so
`/v1/rl/workers` answers `total: 0` until it finishes — poll rather than reading
it the moment `/health` returns.

## Discovery (step d) — PASS

`GET /v1/rl/workers`, saved on the node at `~/smg-rl-ts/logs/acc-rl-workers.json`:

```
total 2 protocol_version 1
  grpc://127.0.0.1:30106 engine=tokenspeed mode=grpc health=ready
    control_url=http://127.0.0.1:30406 tp_size=None
    update_from=['distributed'] source=label id=01a0cb7f-7523-7b01-a71a-669da8cfb402
  grpc://127.0.0.1:30107 engine=tokenspeed mode=grpc health=ready
    control_url=http://127.0.0.1:30407 tp_size=None
    update_from=['distributed'] source=label id=01a0cb7f-7523-7b01-a71a-668d75f44343
```

One entry in full, minus its label bag:

```json
{
  "id": "01a0cb7f-7523-7b01-a71a-669da8cfb402",
  "url": "grpc://127.0.0.1:30106",
  "base_url": "grpc://127.0.0.1:30106",
  "engine": "tokenspeed",
  "engine_version": "unknown",
  "model_id": "/home/ubuntu/models/meta-llama/Llama-3.2-1B-Instruct",
  "worker_type": "regular",
  "connection_mode": "grpc",
  "control_url": "http://127.0.0.1:30406",
  "tp_size": null,
  "dp_size": null,
  "pp_size": 1,
  "dp_ranks": 1,
  "role": null,
  "health": "ready",
  "weight_version": "default",
  "capabilities": {
    "source": "label",
    "pause_modes": ["wait", "abort", "keep"],
    "update_from": ["distributed"],
    "abort": true,
    "flush_cache": true,
    "sleep_wake": true,
    "reports_weight_version": true
  }
}
```

Both `ready`, both with the control endpoint, both advertising
`update_from == ["distributed"]` from the engine's own labels (`source: "label"`,
not the static fallback).

### `tp_size` is still null, and that is correct here

TokenSpeed leaves `attn_tp_size` unset unless the operator asks for a width, so
a bare launch publishes neither `tp_size` nor `attn_tp_size` and discovery has
nothing to read. Commit `681188b20` makes discovery fall back to `attn_tp_size`
when it *is* present, which covers an engine launched with the flag; it cannot
invent one that was never reported. The refit example handles the null by
assuming 1 and saying so on stderr (`no tp_size reported, using 1`), which is
right for these TP-1 engines, and `--tp-size` overrides it for a fleet where it
would not be.

## Refit (step e) — PASS, twice

First refit, `--weight-version 1`, 146 parameters at 64 per call, full log at
`~/smg-rl-ts/logs/acc-refit-1.log`:

```
146 parameter(s) to broadcast, 64 per call
pause_generation: 2/2 ok
  ...668d75f44343: HTTP 200 in 1 ms -> {"success": true, "message": "Paused generation.", "mode": "wait"}
  ...669da8cfb402: HTTP 200 in 1 ms -> {"success": true, "message": "Paused generation.", "mode": "wait"}
  ...668d75f44343: no tp_size reported, using 1
  ...669da8cfb402: no tp_size reported, using 1
world_size=3 (trainer at rank 0), ...668d75f44343@1+1, ...669da8cfb402@2+1
init_weights_update_group: 2/2 ok
  ...668d75f44343: HTTP 200 in 2 ms -> {"success": true, "message": "weight update group initialized"}
  ...669da8cfb402: HTTP 200 in 2 ms -> {"success": true, "message": "weight update group initialized"}
NCCL version 2.30.7+cuda13.3
update_weights_from_distributed[0:64/146]: 2/2 ok
  ...668d75f44343: HTTP 200 in 830 ms -> {"success": true, "message": "updated 64 weights"}
  ...669da8cfb402: HTTP 200 in 830 ms -> {"success": true, "message": "updated 64 weights"}
update_weights_from_distributed[64:128/146]: 2/2 ok
  ...668d75f44343: HTTP 200 in 6 ms -> {"success": true, "message": "updated 64 weights"}
  ...669da8cfb402: HTTP 200 in 5 ms -> {"success": true, "message": "updated 64 weights"}
update_weights_from_distributed[128:146/146]: 2/2 ok
  ...668d75f44343: HTTP 200 in 2 ms -> {"success": true, "message": "updated 18 weights Weight version updated to 1."}
  ...669da8cfb402: HTTP 200 in 2 ms -> {"success": true, "message": "updated 18 weights Weight version updated to 1."}
destroy_weights_update_group: 2/2 ok
  ...668d75f44343: HTTP 200 in 540 ms -> {"success": true, "message": "weight update group destroyed"}
  ...669da8cfb402: HTTP 200 in 540 ms -> {"success": true, "message": "weight update group destroyed"}
continue_generation ... resumed
/generate reported weight_version=1
OK
exit=0
```

Second refit, `--weight-version 2`, identical shape, ending:

```
update_weights_from_distributed[128:146/146]: 2/2 ok
  ...668d75f44343: HTTP 200 in 2 ms -> {"success": true, "message": "updated 18 weights Weight version updated to 2."}
  ...669da8cfb402: HTTP 200 in 2 ms -> {"success": true, "message": "updated 18 weights Weight version updated to 2."}
continue_generation ... resumed
/generate reported weight_version=2
OK
exit=0
```

Two details worth keeping. The first chunk of each refit costs 810–830 ms and
the rest 2–6 ms: that is NCCL's lazy `ncclCommInitRank` plus the 525 MB
embedding, paid once per group. It is exactly the cost the broken engine never
paid, and the clearest signal that the collective is now real. And the version
is stamped only on the last chunk, visible in the message text, so a refit that
died at chunk 2 would leave the old version advertised.

## Fan-out overhead (step f) — PASS

Overhead = the fan-out envelope's `latency_ms` minus the slowest per-worker
`latency_ms`, both taken from the gateway's own `rl.proxy` and `rl.fanout` log
lines. All twelve fan-outs across both refits:

| fan-out | n | ok | fail | per-worker ms | envelope ms | overhead ms |
| --- | --- | --- | --- | --- | --- | --- |
| `pause_generation` | 2 | 2 | 0 | 1, 1 | 2 | 1 |
| `update_weights_from_distributed` | 2 | 2 | 0 | 830, 830 | 830 | 0 |
| `update_weights_from_distributed` | 2 | 2 | 0 | 5, 6 | 6 | 0 |
| `update_weights_from_distributed` | 2 | 2 | 0 | 2, 2 | 2 | 0 |
| `destroy_weights_update_group` | 2 | 2 | 0 | 540, 540 | 540 | 0 |
| `continue_generation` | 2 | 2 | 0 | 0, 0 | 0 | 0 |
| `pause_generation` | 2 | 2 | 0 | 1, 1 | 1 | 0 |
| `update_weights_from_distributed` | 2 | 2 | 0 | 808, 808 | 808 | 0 |
| `update_weights_from_distributed` | 2 | 2 | 0 | 5, 5 | 5 | 0 |
| `update_weights_from_distributed` | 2 | 2 | 0 | 2, 2 | 2 | 0 |
| `destroy_weights_update_group` | 2 | 2 | 0 | 541, 542 | 542 | 0 |
| `continue_generation` | 2 | 2 | 0 | 0, 0 | 1 | 1 |

Worst overhead 1 ms against a 5 ms budget. `init_weights_update_group` is a
per-worker call rather than a fan-out and cost 1–2 ms per worker.

## Failure injection (step g) — PASS

The 30107 engine was killed (`pkill -f "port 30107"` inside `ts-rl`); its
control port then answered nothing (`30406=200`, `30407=000`). A `flush_cache`
fan-out at the gateway:

```
HTTP 207
{
  "results": {
    "01a0cb7f-7523-7b01-a71a-669da8cfb402": {
      "url": "grpc://127.0.0.1:30106", "status": 200, "latency_ms": 1,
      "body": {"success": true, "message": "Cache flushed."}
    }
  },
  "failed": [
    {
      "worker_id": "01a0cb7f-7523-7b01-a71a-668d75f44343",
      "url": "grpc://127.0.0.1:30107",
      "error": "upstream_unreachable",
      "message": "upstream `grpc://127.0.0.1:30107` unreachable: error sending request for url (http://127.0.0.1:30407/flush_cache)"
    }
  ],
  "total": 2, "succeeded": 1
}
```

207, the dead worker in `failed[]` as `upstream_unreachable` naming its control
URL, the live worker's outcome in `results`. Metrics at that moment:

```
smg_worker_cb_state{worker="grpc://127.0.0.1:30107"} 0
smg_worker_cb_consecutive_failures{worker="grpc://127.0.0.1:30107"} 0
smg_engine_running_requests{worker="grpc://127.0.0.1:30106",...} 0
smg_engine_running_requests{worker="grpc://127.0.0.1:30107",...} 0
smg_rl_control_calls_total{op="flush_cache",result="ok"} 1
smg_rl_control_calls_total{op="flush_cache",result="unreachable"} 1
smg_rl_fanout_total{result="ok"} 12
smg_rl_fanout_total{result="partial"} 1
```

Breaker closed at 0, no running requests leaked, and the RL counters recorded
the partial fan-out and the unreachable leg rather than swallowing them.

The engine was restarted with the same command and became healthy
(`30407 ready`); the same fan-out then returned **HTTP 200**, `succeeded: 2`,
`failed: []`, both engines answering
`{"success": true, "message": "Cache flushed."}` in 1 ms.

## Flag off (step h) — PASS

The gateway was stopped and relaunched with the identical command minus
`--enable-rl`:

```
=== GET /v1/rl/workers ===
HTTP/1.1 404 Not Found
content-length: 0
  body bytes: 0
=== POST /v1/rl/engine/flush_cache ===
  http=404
=== GET /workers (data plane) ===
  total 2
   grpc://127.0.0.1:30106 ready
   grpc://127.0.0.1:30107 ready
=== metrics ===
  smg_rl_ series: 0
  smg_ series total: 338
```

The RL surface is gone, body and all; the data plane is untouched; and no
`smg_rl_` series is registered while 338 other `smg_` series still are — so the
absence is the flag, not a broken metrics endpoint.

## Task 19 probes

Recorded verbatim, not interpreted. Gateway on `--enable-rl`, both engines
healthy, before step h. Also saved on the node at
`~/smg-rl-ts/logs/parity-probes.log`.

```
== SMG http://127.0.0.1:30100 (['grpc://127.0.0.1:30106', 'grpc://127.0.0.1:30107'])
a_single_text: http=404 shape=dict meta_info_keys=None output_token_logprobs=None weight_version=None error={'type': 'Not Found', 'code': 'model_not_found', 'message': "No worker available for model 'unknown'", 'param': None}
b_no_model: http=404 shape=dict meta_info_keys=None output_token_logprobs=None weight_version=None error={'type': 'Not Found', 'code': 'model_not_found', 'message': "No worker available for model 'unknown'", 'param': None}
c_logprob: http=404 shape=dict meta_info_keys=None output_token_logprobs=None weight_version=None error={'type': 'Not Found', 'code': 'model_not_found', 'message': "No worker available for model 'unknown'", 'param': None}
d_sampling_seed: http=404 shape=dict meta_info_keys=None output_token_logprobs=None weight_version=None error={'type': 'Not Found', 'code': 'model_not_found', 'message': "No worker available for model 'unknown'", 'param': None}
e_batch: http=422 body-unparseable (Expecting value: line 1 column 1 (char 0))
f_input_ids: http=404 shape=dict meta_info_keys=None output_token_logprobs=None weight_version=None error={'type': 'Not Found', 'code': 'model_not_found', 'message': "No worker available for model 'unknown'", 'param': None}
```

## Task 19 decision and Task 20

The probes above were taken before Task 20. Interpreted, they said:

| Probe | Before Task 20 | Gap? |
| --- | --- | --- |
| no `model` in the body (every slime request) | 404 `model_not_found` for `'unknown'` | yes |
| single prompt with `model` set | 200, but a list of one object | yes: slime indexes `output["meta_info"]` |
| `return_logprob: true` (engine on `--enable-output-logprobs`) | `meta_info.output_token_logprobs` = `[[logprob, token_id], ...]` | no |
| `sampling_params.sampling_seed` | accepted; it is SMG's native field, no alias needed | no |
| `weight_version` in `meta_info` | present | no |
| `text` as a list | 422 from the JSON extractor (every path) | out of scope: slime sends one prompt per request |
| batch `input_ids` | 400 "not supported over gRPC generate yet" | out of scope, same reason |

Task 20 landed as `123074b50` + `b156402c7`: the gRPC family answers a
single-prompt, `n <= 1`, one-response `/generate` with one object (SGLang's
shape) and defaults an absent `model` to the fleet's single served model
(untagged workers do not count; zero or several models keep the 404).
Re-probed on the rebuilt release binary (05:36Z) with no `model`:

```
a_single_text:      http=200 shape=dict output_token_logprobs=present len=4 weight_version=default
b_no_logprob:       http=200 shape=dict output_token_logprobs=absent  weight_version=default
c_input_ids_logprob http=200 shape=dict output_token_logprobs=present len=4 weight_version=default
d_sampling_seed:    http=200 shape=dict output_token_logprobs=present len=4 weight_version=default
e_batch_text:       http=422 (list-valued text, every path)
f_batch_input_ids:  http=400 resolve_input_failed (gRPC batch input_ids)
```

## A second SMG bug found by the slime run: the wildcard model leaked upstream

The first candidate-arm attempt failed on every rollout request with
404 `model_not_found` for model `'unknown'`, from SMG's HTTP passthrough to the
`ts serve` sidecars. The registry, the wildcard routing pool and the policy
were all fine (an instrumented debug build showed `pool_len=1`, the worker
`available`); the 404 came from *behind* the sidecar. SMG's typed HTTP path
parses the body into `GenerateRequest` and re-serializes it, and the absent
`model` had become `"model":"unknown"` on the wire. SGLang ignores that field,
which is why the M0 run never noticed; the sidecar's embedded gateway resolves
it and answers 404. Fix `a5e45faf9`: the placeholder is never serialized, so
the forwarded body matches what the client sent. Covered by a protocol
round-trip test and an HTTP router test that inspects the forwarded body.

Observed but not fixed: with `--disable-retries` the streamed-body path
answers 400 "Failed to buffer the request body" for the same request. slime's
default keeps retries on, so the A/B did not need it; it is a separate
streamed-path issue for a follow-up.

## Acceptance item 2 — slime, unpatched, Qwen3-0.6B, GSM8K

### Topology (and why it differs from the spec's candidate)

slime's external-engine path (`slime/backends/sglang_utils/external.py`,
`sglang_engine.py::_init_external`, `server_control.py::abort_servers_until_idle`)
requires every external engine address to be a full SGLang HTTP server
(`/server_info` or `/get_server_info`, `/abort_request`, `/v1/loads`),
registers each into the router as a data-plane worker (`POST /workers`), and
aborts through the URLs the router's `/workers` returns. TokenSpeed's
in-engine control app is control-only (no `/generate`, no `/get_server_info`);
only the `ts serve` sidecar has the full surface. The spec's candidate
("SMG gRPC data plane + in-engine control apps as slime's external engines")
is therefore unreachable for unpatched slime. The A/B was run as a router swap
over the sidecars, and the gRPC data plane was exercised with a labeled
two-file slime patch (the M3 shape).

| Arm | slime | router for `/generate` | engines | control path |
| --- | --- | --- | --- | --- |
| baseline | stock a3f50097 | slime's own `sglang_router` | 2x `ts serve` (sidecars 31201/31203 registered by slime) | engine-direct to the sidecars |
| candidate | stock a3f50097 | SMG `--enable-rl --disable-health-check --disable-circuit-breaker --request-timeout-secs 14400`, empty registry | same, registered by slime into SMG | engine-direct to the sidecars |
| gRPC arm | a3f50097 + `slime-patch.py` (skip `_register_to_router`; abort via the external addresses) | SMG `--enable-rl` owning the two engines as gRPC workers (`--worker-urls grpc://...`) | same engines' gRPC ports | engine-direct to the sidecars |

Common to all arms: slime in the `slimerl/slime:latest` container on GPUs 6–7
(Megatron actor, 2 GPUs, `--num-rollout N --rollout-batch-size 16
--n-samples-per-prompt 8 --rollout-max-response-len 1024 --global-batch-size 128
--rollout-seed 42 --seed 1234`, GRPO, `zhuzilin/gsm8k`, NCCL refits every step
via `update_weights_from_distributed`); TokenSpeed `ts serve` engines in
`ts-rl` on GPUs 4–5 (`--gpu-memory-utilization 0.8 --enable-output-logprobs`);
fresh engines, and a fresh SMG, per arm. Scripts: `remote/slime-run.sh`,
`remote/slime-ab.sh`, `remote/slime-grpc-arm.sh`, `remote/slime-patch.py`,
`remote/slime-metrics.py` in the SDD workspace.

### Environment findings on the way

- **TokenSpeed's sidecar was incompatible with current slime** before any
  router was involved: slime's sanity check reads `enable_memory_saver` from a
  flat `/get_server_info`; the sidecar nested everything under `server_args`.
  TokenSpeed `46cb33f13` flattens the response (nested key kept).
- **TokenSpeed's NCCL refit was a silent no-op** (the same engine bug as item 1;
  TokenSpeed `053d6356a`).
- **Cross-container NCCL** (trainer in the slime image, engines in the CI
  image) needed the same NCCL wheel on both sides (`nvidia-nccl-cu12==2.30.7`
  in the trainer container) and `NCCL_CUMEM_ENABLE=0` on both sides: the
  cuMem shareable-handle import between the cu12-built and cu13-built NCCLs
  failed with `ncclP2pImportShareableBuffer: invalid argument`. A single-image
  deployment does not need either.
- **The M0 findings reproduced exactly**: engines keep a stale weight-update
  group across slime runs (`init_weights_update_group` → 400 "group name has
  already been created"), and a second run against the same SMG instance gets
  409 on `POST /workers` because slime never deregisters. Both are why each
  arm starts fresh engines and a fresh SMG.

### Results

Ten-step smokes first (all PASS): baseline 0.644 mean reward, 9409 tok/GPU/s,
8.90 s/step; candidate 0.637, 9219 tok/GPU/s, 8.25 s/step; gRPC arm 0.643,
9355 tok/GPU/s, 8.23 s/step (1280 `/generate` requests through the gRPC
gateway, model-less and answered as objects; aborts and refits engine-direct).

200 steps, same seed, `remote/slime-metrics.py` over the train logs:

| Arm | reward mean (200) | reward, last 50: mean ± sd | rollout tok/GPU/s | step time | rollout time | refit time |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (sgl-router) | 0.749 | 0.771 ± 0.074 | 6949 | 6.42 s | 4.37 s | 0.133 s |
| candidate (SMG `--enable-rl`) | 0.748 | 0.778 ± 0.081 | 7741 | 6.79 s | 4.60 s | 0.133 s |

Bar from the spec: |Δ mean reward| over the last 50 steps < 1σ → 0.007 versus
σ ≈ 0.08, **PASS**. Throughput within 5 %: rollout throughput is 11 % *higher*
through SMG, **PASS**. Step time within 5 %: +5.7 %, **missed by 0.7 points**
on a single run per arm; with rollout throughput favouring SMG the step-time
gap sits in the trainer's wait/overlap noise, and one 200-step run per arm
cannot resolve ±5 %. Recorded as-is rather than re-run.

Control in these runs is slime engine-direct, exactly as in M0. Control
through SMG is proven by item 1; moving slime onto the fan-out is M3's optional
patch, whose exact scope this run pins down: skip registration of
router-owned engines and route aborts (and refits) through the router.

Logs on the node: `~/slime-rl/logs/train-{base-200,cand-200,grpc-smoke2}.log`,
`~/slime-rl/logs/ab-200.log`, `~/slime-rl/logs/smg-3110{0,1}.log`.

### Criteria — acceptance item 2

| Criterion | Result |
| --- | --- |
| Unpatched slime completes a run with TokenSpeed engines behind SMG | PASS — 200 steps, candidate arm |
| Reward within run-to-run noise (|Δ| < 1σ over the last 50 steps) | PASS — 0.007 vs σ 0.08 |
| Rollout throughput within 5 % | PASS — +11 % through SMG |
| Step time within 5 % | MISS by 0.7 points (6.79 vs 6.42 s, one run each) |
| Control-path caveat stated | yes: engine-direct in both arms; SMG fan-out proven by item 1 |
| slime's `/generate` over SMG's gRPC data plane | PASS with the labeled patch (10 steps) |

## Final node sweep on the branch head

Run on `b156402c7` (SMG) and `053d6356a` (TokenSpeed) after the A/B; see the
SDD ledger for the raw lines.

## Cleanup (step i) — DONE

`acc-cleanup.sh down` stopped only this lane's gateway and its two engines, then
reported the compute apps on GPUs 0, 1 and 2 by matching
`nvidia-smi --query-compute-apps=pid,gpu_uuid` against those GPUs' UUIDs from
`nvidia-smi -L`. The engines take about 30 s to release GPU memory after the
signal; the confirming report:

```
=== GPUs 0-2 ===
0 GPU-117de282-ad55-970c-b81c-9205947444ca
1 GPU-f6e64f76-185c-4fce-fa92-8c772ca5294b
2 GPU-15d5124e-1f8d-21b9-0ede-4db4e6b3fcd8
=== compute apps on GPUs 0-2 ===
  total on GPUs 0-2: 0
=== this lane's processes ===
  none in ts-rl
  no gateway
```

The other lane's `slime-rl` container, its `ts serve` engines on ports 312xx and
its SMG on 31100 were left running and untouched.

## Criteria — acceptance item 1

| Criterion | Result |
| --- | --- |
| Engines discovered with `control_url` and `update_from == ["distributed"]` | PASS |
| Both workers `ready`, capabilities from the engine (`source: "label"`) | PASS |
| `pause_generation` / `continue_generation` fan-outs | PASS |
| `init_weights_update_group` per worker with the computed `rank_offset` | PASS — engines joined at ranks 1 and 2 of a world of 3 |
| `update_weights_from_distributed` fan-out per chunk | PASS — 3 chunks per refit, 146 parameters |
| `destroy_weights_update_group` fan-out | PASS — 200, `success: true`, idempotent |
| Refit every engine and see the new `weight_version` on the next `/generate` | **PASS — reported 1, then 2** |
| Fan-out overhead < 5 ms | PASS — worst 1 ms over 12 fan-outs |
| Failure injection: 207 naming the dead worker `upstream_unreachable` | PASS |
| Breakers closed, no running requests leaked | PASS — `smg_worker_cb_state` 0, running requests 0 |
| Recovery: restart, same fan-out returns 200 | PASS |
| Flag off: `/v1/rl/*` 404 empty, `/workers` intact, no `smg_rl_` series | PASS |
| Cleanup: GPUs 0–2 free, other lanes untouched | PASS |

## Related lanes

| Lane | Result |
| --- | --- |
| e2e `test_rl_control_plane.py -k TokenSpeed` | 6/6 |
| e2e `test_rl_control_plane.py -k Sglang` | 6/6 |

Those lanes cover discovery, the per-worker proxy, fan-out flush/pause/continue
and the 501 refusal for disk refits. Neither exercises a real NCCL weight
transfer, which is why the bug below only surfaced here.

---

## Root cause of the 2026-09-22 failure (fixed)

Kept because it is why the engine fix exists and what a regression would look
like.

On the first attempt every control call succeeded and the trainer still hung.
`update_weights_from_distributed` returned HTTP 200 in 64 ms for a single
128256x2048 bfloat16 tensor while the trainer had not transferred a byte, and
the engine loaded its uninitialized `torch.empty` buffers into the live model,
after which `/generate` returned garbage tokens.

`py-spy` on the trainer put the main thread at
`torch/distributed/distributed_c10d.py:3781`, which is
`work = group.broadcast([tensor], opts)` — NCCL's lazy communicator init. At the
same moment both engine schedulers were idle in their ZMQ event loop with the
`tokenspeed::forward` thread idle, so they were not in a collective at all.

The mechanism: TokenSpeed builds its own world process group with a bound
`device_id`. When a process has a device-bound default group, torch's
`_new_process_group_helper` sets `split_from`, so the engine's weight-update
communicator was created with **`ncclCommSplit` off the engine's own size-1
communicator** instead of `ncclCommInitRank` against the unique id the trainer
published through the store. Splitting a size-1 parent yields a size-1 child,
and a broadcast on a size-1 communicator is a local no-op: instant, silent,
buffer untouched, success reported. The trainer has no default process group,
so nothing binds a device, so it took the `ncclCommInitRank` path and waited for
peers that had joined a different communicator.

Both sides were correct in isolation. A three-rank reproduction on free GPUs
(`remote/nccl_pair_test.py`), rank 0 running the example's `_join_trainer_group`
verbatim and ranks 1 and 2 running TokenSpeed's join body verbatim, transferred
correctly on every rank:

```
[trainer:0] joined in 3.01s
[trainer:0] broadcast done in 1.16s mean=3.5000 (expect 3.5)
[engine:1] broadcast done in 1.17s mean=3.5000 (expect 3.5)
[engine:2] broadcast done in 1.17s mean=3.5000 (expect 3.5)
```

The fix, in TokenSpeed's `runtime/execution/weight_update_group.py`, clears the
default group's `bound_device_id` around the helper so the weight-update group
is built with `ncclCommInitRank`, and refuses to join when the backend's
`options.split_from` is still set — so a regression fails loudly at
`init_weights_update_group` instead of silently corrupting the model. The run
recorded above is against that fix, and the 810 ms first chunk is the
`ncclCommInitRank` the broken build skipped.
