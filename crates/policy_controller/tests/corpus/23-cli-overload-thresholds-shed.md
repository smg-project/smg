# 23. Cache-aware with tightened overload thresholds and shedding

Dialect: CLI argv.

## Input

```
smg launch --worker-urls http://worker-a:8000 http://worker-b:8000 --policy cache_aware \
  --worker-overload-waiting-requests 32 --worker-overload-token-usage 0.9 --worker-overload-shed
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
    overload:
      waiting_requests: 32
      token_usage: 0.9
      shed_when_all_vetoed: true
```
