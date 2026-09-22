# RL examples

## Launch profile

```bash
smg launch --worker-urls http://rollout:30000 http://rollout:30001 --policy cache_aware \
  --enable-rl --disable-health-check --disable-circuit-breaker --request-timeout-secs 14400
```

`--disable-health-check` and `--disable-circuit-breaker` match what slime and vime
set on their own routers (a transient engine error must not open a breaker for a
whole training step); `--request-timeout-secs 14400` covers multi-hour agentic
rollouts. Raise `--rl-control-timeout-secs` (default 600) if a disk refit takes longer.

## refit_from_disk.py

Pauses, refits, and resumes every worker matching `--selector`, then confirms the
next `/generate` through SMG reports the new `meta_info.weight_version`.

```bash
python examples/rl/refit_from_disk.py --smg http://127.0.0.1:30000 \
  --model-path /ckpt/step-42 --weight-version 42
```

Exit code 0 on success, 1 on a version mismatch; a failed fan-out raises
`smg.rl.FanoutError` naming the workers that failed (the others completed).

SGLang only. TokenSpeed answers `update_weights_from_disk` with HTTP 501 and
keeps serving — use `refit_from_trainer.py` there.

## refit_from_trainer.py

The trainer-driven NCCL refit, which is slime's path and the only one
TokenSpeed implements. The trainer holds rank 0 of a process group every engine
rank joins and broadcasts each weight into the engines, which load them as they
arrive. Same pause/refit/resume/verify shape as the disk example.

```bash
python examples/rl/refit_from_trainer.py --smg http://127.0.0.1:30000 \
  --model-path /ckpt/step-42 --weight-version 42 --selector engine=tokenspeed
```

This one runs where a trainer runs: it needs `torch` built with CUDA and
`transformers`, and it loads `--model-path` onto `--device` (default `cuda:0`)
to stand in for a training loop's live weights. `--chunk` (default 64) is how
many parameters ride on one `update_weights_from_distributed` call, so a large
model streams in batches instead of one enormous request.

`--master-address` is this host's address **as the engines see it**: every
engine is handed it and dials it to join the trainer's process group. The
default, `127.0.0.1`, only works when the engines share a host with the
trainer. Against engines on other hosts, pass the trainer's routable address —
otherwise each engine dials its own loopback, none arrives, and the rendezvous
stalls for `--timeout` before failing. The script refuses to start when `--smg`
names a non-loopback host and `--master-address` is still loopback.

```bash
python examples/rl/refit_from_trainer.py --smg http://smg:30000 \
  --master-address trainer-0.internal \
  --model-path /ckpt/step-42 --weight-version 42 --selector engine=tokenspeed
```

Ranks are laid out from each worker's `tp_size`: the trainer takes rank 0 and
engine *k* takes `rank_offsets[k] .. rank_offsets[k] + tp_size - 1`, so two
TP-1 engines make a world of 3. `init_weights_update_group` goes to each worker
separately (every one needs its own `rank_offset`); the broadcast itself is one
fan-out per chunk, since the body is identical everywhere.

A TokenSpeed engine launched without an explicit parallelism flag reports no
`tp_size` at all, and the script then assumes 1 and says so on stderr. Pass
`--tp-size` when that assumption is wrong: a bad rank layout deadlocks the
group instead of failing.

Exit code 0 on success, 1 on any failure — including a refit that lands but
whose teardown does not.

An HTTP-level failure is cleaned up: the group is destroyed, the cache flushed
and the engines resumed on the way out, and the engines are named on stderr as
serving a part-old, part-new model if any chunk had already gone out. An engine
that stops participating in the collective is different. By then the trainer's
main thread is inside NCCL, so torch's watchdog takes the process down without
raising into Python, no cleanup runs, and the fleet stays paused. Un-pause it
by hand:

```python
from smg.rl import RL
RL("http://smg:30000").fanout("continue_generation", {}, selector="engine=tokenspeed")
```

## Selector cheatsheet

`engine=sglang` · `engine in (sglang,vllm)` · `role!=reward` · `url=http://rollout:30000`
(single worker) · `model=Qwen/Qwen3-8B, tp_size=1`
