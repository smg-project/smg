# TokenSpeed RL control plane — acceptance run on the H200 node

Status: **BLOCKED on acceptance item 1 (refit).** The control plane itself —
discovery, per-worker proxy, fan-out, pause/resume, group setup and teardown —
worked on every call. The weight transfer did not: TokenSpeed's engine answers
`update_weights_from_distributed` with HTTP 200 without ever entering the NCCL
collective, so the trainer blocks in `ncclCommInitRank` and the engine loads the
uninitialized receive buffers into the model. Evidence and the isolated
reproduction that rules out the trainer-side code are below.

The run was then cut short: the OCI bastion session expired mid-run and cannot
be renewed without an interactive `oci session authenticate`. Steps f–i were not
reached and the node was not cleaned up. See **State left on the node**.

## Environment

| Piece | Value |
| --- | --- |
| Node | `moirai-h200-runner-0`, `ubuntu@10.0.1.52` via the OCI bastion, 8x H200 |
| Gateway | `~/smg-rl-ts/target/release/smg`, release build of `rl/tokenspeed-control-endpoint` |
| Source on node | `~/smg-rl-ts/src` = this worktree at `fdae590e` |
| Engines | TokenSpeed in container `ts-rl`, image `ghcr.io/smg-project/smg:ci-tokenspeed-e87f7267c0b3-cd0bac2babbb` |
| Container venv | `/opt/smg-ci/.venv`, Python 3.12.3 |
| torch / transformers | 2.14.0+cu130 / 5.12.0 |
| NCCL | 2.30.7+cuda13.3 |
| Model | `/home/ubuntu/models/meta-llama/Llama-3.2-1B-Instruct`, 146 parameters |
| Trainer | same container, `CUDA_VISIBLE_DEVICES=2` |

## Commands, in order

```bash
# a. sync (from the worktree)
.superpowers/sdd/2026-09-21-rl-tokenspeed-control-endpoint/h200-rsync.sh -a --delete \
  --exclude target --exclude .git --exclude .superpowers --exclude .venv \
  --exclude __pycache__ --exclude .pytest_cache <worktree>/ h200:~/smg-rl-ts/src/

# b. GPU check
nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l     # -> 0

# c. two engines, GPUs 0 and 1
~/smg-rl-ts/remote/acc-engines.sh up
~/smg-rl-ts/remote/acc-engines.sh wait     # -> control 30406 ready=yes / 30407 ready=yes

# d. gateway
~/smg-rl-ts/remote/acc-gateway.sh up       # -> gateway pid 3675285
~/smg-rl-ts/remote/acc-gateway.sh wait     # -> health ok, 2 workers
curl -s http://127.0.0.1:30100/v1/rl/workers | python3 -m json.tool

# e. refit (failed, see below)
~/smg-rl-ts/remote/acc-refit.sh run 1            # chunk 64, hung
~/smg-rl-ts/remote/acc-refit.sh run diag1 1      # chunk 1, hung identically

# diagnosis
~/smg-rl-ts/remote/acc-stacks.sh                 # py-spy on trainer + schedulers
~/smg-rl-ts/remote/acc-pairtest.sh 29777         # isolated 3-rank NCCL repro
~/smg-rl-ts/remote/acc-reset.sh                  # kill trainer, destroy group, resume
```

The helper scripts live in
`.superpowers/sdd/2026-09-21-rl-tokenspeed-control-endpoint/remote/` and are
synced to `~/smg-rl-ts/remote/`.

The engines were launched with:

```bash
docker exec -d ts-rl bash -c 'source /opt/smg-ci/.venv/bin/activate;
  CUDA_VISIBLE_DEVICES=0 python -m smg_grpc_servicer.tokenspeed
    --model /home/ubuntu/models/meta-llama/Llama-3.2-1B-Instruct
    --host 127.0.0.1 --port 30106
    --rl-control-host 127.0.0.1 --rl-control-port 30406
    --enable-output-logprobs > /work/logs/acc-engine-0.log 2>&1'
```

and the gateway with:

```bash
setsid nohup ~/smg-rl-ts/target/release/smg launch \
  --model-path /home/ubuntu/models/meta-llama/Llama-3.2-1B-Instruct \
  --worker-urls grpc://127.0.0.1:30106 grpc://127.0.0.1:30107 \
  --policy cache_aware --port 30100 --enable-rl \
  --disable-health-check --disable-circuit-breaker \
  --request-timeout-secs 14400 --prometheus-port 29100 \
  > ~/smg-rl-ts/logs/acc-smg.log 2>&1 &
```

`--model-path` is required in regular mode (`e2e_test/infra/gateway.py` passes
it the same way); without it the gateway refuses to launch.

## Discovery (step d) — PASS

`GET /v1/rl/workers` (full JSON saved on the node at
`~/smg-rl-ts/logs/acc-rl-workers.json`), one entry per engine:

```json
{
  "id": "01a0cacf-e118-7952-b1df-3415a16614ff",
  "url": "grpc://127.0.0.1:30106",
  "engine": "tokenspeed",
  "connection_mode": "grpc",
  "control_url": "http://127.0.0.1:30406",
  "tp_size": null,
  "pp_size": 1,
  "dp_ranks": 1,
  "health": "ready",
  "weight_version": "default",
  "capabilities": {
    "source": "label",
    "pause_modes": ["wait", "abort", "keep"],
    "update_from": ["distributed"],
    "abort": true, "flush_cache": true,
    "sleep_wake": true, "reports_weight_version": true
  }
}
```

The second worker is identical with `grpc://127.0.0.1:30107` and
`http://127.0.0.1:30407`. Both `ready`, both with the control endpoint, both
advertising `update_from == ["distributed"]` from the engine's own labels
(`source: "label"`, not the static fallback). That is the whole of step d.

### Drift found: `tp_size` is null for a plain TokenSpeed launch

TokenSpeed's `server_args` carries `attn_tp_size=None` unless the operator sets
it, and SMG derives `tp_size` from `attn_tp_size`/`tp_size` in the engine's
server args (`model_gateway/src/routers/grpc/client.rs`, `TOKENSPEED_GRPC_KEYS`).
`pipeline_parallel_size=1` does map, so `pp_size` is `1` while `tp_size` is
`null`. The e2e lane asserts `w["tp_size"] is not None`
(`e2e_test/router/test_rl_control_plane.py`), so that lane must launch its
engines with the parallelism flag set; a bare launch does not. The refit example
handles it: it falls back to 1 per worker and says so on stderr
(`no tp_size reported, assuming 1`), which is correct for these TP-1 engines but
would silently mis-lay-out a real TP>1 fleet whose engines do not report it.

## Refit (step e) — FAIL, engine side

`examples/rl/refit_from_trainer.py` drove the sequence correctly and every
control call succeeded. The trainer then blocked forever in its first
`dist.broadcast`.

Trainer output (chunk 1 run, `--chunk 1`, log `~/smg-rl-ts/logs/acc-refit-diag1.log`):

```
loading /home/ubuntu/models/meta-llama/Llama-3.2-1B-Instruct onto cuda:0 ...
146 parameter(s) to broadcast, 1 per call
pause_generation: 2/2 ok
  01a0cacf-...-3405d51fc613: HTTP 200 in 1 ms -> {"success": true, "message": "Paused generation.", "mode": "wait"}
  01a0cacf-...-3415a16614ff: HTTP 200 in 1 ms -> {"success": true, "message": "Paused generation.", "mode": "wait"}
  01a0cacf-...-3405d51fc613: no tp_size reported, assuming 1
  01a0cacf-...-3415a16614ff: no tp_size reported, assuming 1
world_size=3 (trainer at rank 0), 01a0cacf-...-3405d51fc613@1+1, 01a0cacf-...-3415a16614ff@2+1
init_weights_update_group: 2/2 ok
  01a0cacf-...-3405d51fc613: HTTP 200 in 1 ms -> {"success": true, "message": "weight update group initialized"}
  01a0cacf-...-3415a16614ff: HTTP 200 in 1 ms -> {"success": true, "message": "weight update group initialized"}
NCCL version 2.30.7+cuda13.3
<hangs here, indefinitely>
```

Engine side, both engines (`~/smg-rl-ts/logs/acc-engine-{0,1}.log`):

```
[2026-09-22 20:50:02,800  ATTN TP RANK 0] - INFO - weight-update group joined:
    rank=2 world_size=3 device=cuda:0 group=smg_refit (model_runner.py:296)
```

(rank 1 on the other engine — the rank layout is exactly what the plan
specifies.) Nothing after that: no error, no warning, no traceback.

Gateway log, the `update_weights_from_distributed` fan-out that the trainer was
waiting on:

```
20:50:02  rl.proxy  path=update_weights_from_distributed status=200 latency_ms=64
20:50:02  rl.proxy  path=update_weights_from_distributed status=200 latency_ms=64
20:50:02  rl.fanout path=update_weights_from_distributed total=2 succeeded=2 failed=0 latency_ms=64
```

So each engine answered **200 in 64 ms for a single 128256x2048 bfloat16 tensor
(525 MB)** while the trainer had not yet transferred a byte.

### Where each side actually was

`py-spy dump` on the trainer, while hung:

```
Thread 3817 (active): "MainThread"
    broadcast (torch/distributed/distributed_c10d.py:3781)
    _stream_weights (refit_from_trainer.py:276)
    _refit (refit_from_trainer.py:363)
    main (refit_from_trainer.py:413)
```

Line 3781 of torch 2.14's `distributed_c10d.py` is
`work = group.broadcast([tensor], opts)` — the call that does NCCL's lazy
`ncclCommInitRank`. The trainer is waiting for peers to join the communicator.
No fan-out thread is alive (it already returned), and `ss` shows no connection
to the gateway, so nothing is in flight.

`py-spy dump` on both engine scheduler processes at the same moment:

```
Thread 2802 (active+gil): "MainThread"
    recv_multipart (zmq/sugar/socket.py:813)
    _drain_reqs (engine/request_handler.py:248)
    event_loop (engine/event_loop.py:1264)
Thread 2981 (idle): "tokenspeed::forward"
```

Both schedulers are back in their idle event loop and the forward thread — the
thread that runs the weight update (`device.py`, `self._thread.run(_apply_update)`)
— is idle. The engines are not in a collective at all.

### The weights that were "received" are uninitialized memory

After the first refit attempt (chunk 64, which reported `succeeded=2` for its
first batch of 64 parameters), a `/generate` through SMG returns garbage:

```
POST /generate {"text":"1+1=","sampling_params":{"max_new_tokens":4}}
-> {"text": " çerç-еヶ月 東京", "meta_info": {"weight_version": "default", ...}}
```

The engine's receive path allocates `torch.empty(shape, dtype, device)` and
feeds it to `load_weights` whether or not the broadcast delivered anything, so
a no-op broadcast writes uninitialized memory into the live model. The engine
still answers `{"success": true, "message": "updated 64 weights"}`.

### The trainer-side code is not the cause

`remote/nccl_pair_test.py` runs three processes on free GPUs (3, 6, 7): rank 0
uses the example's `_join_trainer_group` verbatim, ranks 1 and 2 use TokenSpeed's
`init_weights_update_group` body verbatim (same `rendezvous` +
`PrefixStore(group_name, store)` + `_new_process_group_helper` +
`_world.pg_group_ranks`), then all three do one `dist.broadcast(src=0)`:

```
[trainer:0] joined in 3.01s
[trainer:0] broadcast done in 1.16s mean=3.5000 (expect 3.5)
[engine:1] joined in 0.01s
[engine:1] broadcast done in 1.17s mean=3.5000 (expect 3.5)
[engine:2] joined in 0.01s
[engine:2] broadcast done in 1.17s mean=3.5000 (expect 3.5)
```

Correct data on every rank. The same join code that deadlocks inside the
TokenSpeed engine process works outside it, so the group construction, the rank
layout and the broadcast order in `refit_from_trainer.py` are right. Note also
the 1.17 s cost of the lazy NCCL init in this test versus the engine's 64 ms
round trip — the engine never paid that cost, which is what "it never entered
the collective" looks like from the outside.

### What to look at next in TokenSpeed

A non-root NCCL broadcast that returns immediately, leaves the buffer untouched
and raises nothing is what you get from a collective that was never issued to a
real multi-rank communicator. Candidates, in the order worth checking:

1. The effective size/rank of `self._weight_update_pg` at broadcast time
   (`dist.get_world_size(pg)` / `dist.get_rank(pg)`), versus the `world_size=3`
   the join logged. A size-1 communicator makes a broadcast a local no-op.
2. Whether the forward thread is inside an open NCCL group scope
   (`ncclGroupStart` / torch's `_coalescing_manager`) when the update runs,
   which defers collectives instead of executing them.
3. Whether the engine's own default process group and its bound device change
   how `_new_process_group_helper` builds the new group in that process
   (the engine has one; the trainer does not).

`remote/acc-instrument.py add` was applied on the node to log, inside
`update_weights_from_distributed`, `dist.get_world_size(pg)`,
`dist.get_rank(pg)`, `type(self.model).__name__`, and the mean of each received
buffer plus a receive count. The engines were restarted with it, but the bastion
session expired before that run could be read. Running
`~/smg-rl-ts/remote/acc-refit.sh run diag2 1` and reading
`~/smg-rl-ts/logs/acc-engine-0.log` for `RLDEBUG` should answer (1) and the
model-class question in one shot.

## Fan-out overhead (step f) — PASS on the calls that ran

Overhead = fan-out envelope `latency_ms` minus the slowest per-worker
`latency_ms`, both from the gateway's own `rl.proxy` / `rl.fanout` log lines.

| Fan-out | per-worker (ms) | fan-out (ms) | overhead (ms) |
| --- | --- | --- | --- |
| `pause_generation` | 2, 1 | 2 | 0 |
| `update_weights_from_distributed` (run 1, 64 names) | 73, 74 | 74 | 0 |
| `update_weights_from_distributed` (diag, 1 name) | 64, 64 | 64 | 0 |

All under the 5 ms budget. `init_weights_update_group` is a per-worker call, not
a fan-out, and cost 1–2 ms per worker.

## Failure injection (step g) — NOT RUN

Blocked: the bastion session expired before this step.

## Flag off (step h) — NOT RUN

Blocked: same.

## Cleanup (step i) — NOT DONE

Blocked: same. See below.

## State left on the node

Nothing here is destructive, but it needs a hand once access is back.

- Two TokenSpeed engines running in `ts-rl` on GPUs 0 and 1 (gRPC 30106/30107,
  control 30406/30407), started by `acc-engines.sh up` and **restarted with the
  RLDEBUG instrumentation**. Stop with `~/smg-rl-ts/remote/acc-engines.sh down`.
- The gateway on port 30100 (Prometheus 29100), pid 3675285 at launch. Stop with
  `~/smg-rl-ts/remote/acc-gateway.sh down`.
- `~/tokenspeed-rl/src/python/tokenspeed/runtime/execution/model_runner.py` has
  temporary `RLDEBUG` logging added. **Revert with
  `python3 ~/smg-rl-ts/remote/acc-instrument.py remove`** and restart the
  engines, or the next run of that tree keeps the diagnostic logging.
- Both engines' weights are corrupted by the partial refits. They must be
  restarted before serving anything.
- Two engines that are **not ours** were running on GPUs 4 and 5
  (`smg_grpc_servicer.tokenspeed ... Qwen/Qwen3-0.6B`, control ports
  31210/31211). They appeared after the initial GPU check came back clean, so
  another lane started them. They were left alone.

To resume: `oci session authenticate --profile iad`, recreate the bastion
session, then re-point `h200.sh` / `h200-rsync.sh` at the new session OCID.

## Criteria — acceptance item 1

| Criterion | Result |
| --- | --- |
| Engines discovered with `control_url` and `update_from == ["distributed"]` | PASS |
| Both workers `ready`, capabilities from the engine (`source: "label"`) | PASS |
| `pause_generation` / `continue_generation` fan-outs | PASS |
| `init_weights_update_group` per worker with the computed `rank_offset` | PASS (engines joined at ranks 1 and 2 of a world of 3) |
| `destroy_weights_update_group` fan-out | PASS (200, `success: true`, idempotent) |
| Refit every engine and see the new `weight_version` on the next `/generate` | **FAIL** — engine returns success without receiving; trainer deadlocks |
| Fan-out overhead < 5 ms | PASS on the three fan-outs that ran (0 ms each) |
| Failure injection: 207 + `upstream_unreachable`, breakers closed | NOT RUN |
| Flag off: `/v1/rl/*` 404, `/workers` intact, no `smg_rl_` metrics | NOT RUN |

## Related lanes already green

| Lane | Result |
| --- | --- |
| e2e `test_rl_control_plane.py -k TokenSpeed` | 6/6 |
| e2e `test_rl_control_plane.py -k Sglang` | 6/6 |

Those lanes cover discovery, the per-worker proxy, fan-out flush/pause/continue
and the 501 refusal for disk refits. None of them exercises a real NCCL weight
transfer, which is why this gap only shows up here.
