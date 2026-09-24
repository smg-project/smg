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

## Selector cheatsheet

`engine=sglang` · `engine in (sglang,vllm)` · `role!=reward` · `url=http://rollout:30000`
(single worker) · `model=Qwen/Qwen3-8B, tp_size=1`
