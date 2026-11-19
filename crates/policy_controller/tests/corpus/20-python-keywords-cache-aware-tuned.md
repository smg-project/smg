# 20. A programmatic Python caller: cache-aware with every tuning keyword given

Dialect: Python binding keywords.

## Input

```json
{
  "worker_urls": ["http://worker-a:8000", "http://worker-b:8000"],
  "policy": "cache_aware",
  "cache_threshold": 0.5,
  "balance_abs_threshold": 32,
  "balance_rel_threshold": 1.1,
  "eviction_interval_secs": 60,
  "max_tree_size": 16777216
}
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
index:
  "*": { sweep_s: 60, max_size: 16777216 }
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - warmup-slice: { weight: 8, miss_share: 0.5 }
      - prefix-affinity:
          weight: 4
          threshold: 0.5
          spill: { abs: 32, rel: 1.1 }
      - kv-pressure
      - least-load
    pick:
      best: { sample: 16 }
bindings:
  - match: { model: "*" }
    profile: cache_aware
```
