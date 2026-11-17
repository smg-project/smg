# 03. The RL example's launch profile

Dialect: CLI argv.

## Input

```
smg launch --worker-urls http://rollout-a:30000 http://rollout-b:30001 --policy cache_aware \
  --enable-rl --disable-health-check --disable-circuit-breaker --request-timeout-secs 14400
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score: [prefix-affinity, kv-pressure, least-load, warmup-slice]
    pick: best
bindings:
  - match: { model: "*" }
    profile: cache_aware
```
