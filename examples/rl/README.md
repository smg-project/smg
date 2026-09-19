# RL examples

## Launch profile

```bash
smg launch --worker-urls http://rollout:30000 http://rollout:30001 --policy cache_aware \
  --enable-rl --disable-health-check --disable-circuit-breaker --request-timeout-secs 14400 \
  --rl-version-policy latest-only --retry-max-retries 30 --retry-initial-backoff-ms 100 \
  --retry-max-backoff-ms 1000
```

`--disable-health-check` and `--disable-circuit-breaker` match what slime and vime
set on their own routers (a transient engine error must not open a breaker for a
whole training step); `--request-timeout-secs 14400` covers multi-hour agentic
rollouts. Raise `--rl-control-timeout-secs` (default 600) if a disk refit takes longer.

With `--enable-rl`, SMG never routes to an engine it has observed paused or
asleep, whether that came from a call proxied through `/v1/rl` or from
`smg.rl.RL.set_state`. `--rl-version-policy` (or the per-request
`x-smg-version-policy` header) then narrows the remaining candidates by
version: `any` (default), `latest-only`, `min-version:<v>`, or
`max-staleness:<k>`. An empty eligible set answers a retryable 503, so the
retry knobs above are the wait budget across a refit — at the values shown,
about 27 s. Every response SMG serves for a versioned engine carries
`x-smg-weight-version`; a buffered `/generate` response whose tokens spanned
two versions also carries `x-smg-mixed-version: true`. SGLang's `default`
label counts as unversioned. An engine restarted behind the same URL without
re-registering keeps its table entry until the next refit or `set_version`.

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
(single worker) · `model=Qwen/Qwen3-8B, tp_size=1` · `control=active` ·
`weight_version!=42`
