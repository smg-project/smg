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
    score:
      - warmup-slice: { weight: 8 }
      - prefix-affinity: { weight: 4 }
      - kv-pressure
      - least-load
    pick:
      best: { sample: 16 }
bindings:
  - match: { model: "*" }
    profile: cache_aware
```
