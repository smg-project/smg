# 11. The data-parallel routing test: rank expansion under cache-aware routing

Dialect: launcher argv.

## Input

```
python -m smg.launch_router --host 127.0.0.1 --port 30000 --prometheus-port 29090 --policy cache_aware --dp-aware --log-level debug
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
