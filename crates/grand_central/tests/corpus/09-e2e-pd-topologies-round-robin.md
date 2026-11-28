# 09. The end-to-end prefill-decode topologies: round-robin on both sides

Dialect: launcher argv (`python -m smg.launch_router`).

## Input

```
python -m smg.launch_router --host 127.0.0.1 --port 30000 --prometheus-port 29090 --policy cache_aware \
  --pd-disaggregation --prefill http://prefill-a:8000 8998 --decode http://decode-a:8000 --decode http://decode-b:8000 \
  --prefill-policy round_robin --decode-policy round_robin --load-monitor-interval 2 \
  --health-check-interval-secs 1 --health-failure-threshold 2 --health-success-threshold 1
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  prefill:
    filter: [healthy, not-overloaded, has-capacity, not-tried]
    score: []
    pick: round-robin
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
    score: []
    pick: round-robin
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
  - match: { model: "*", phase: [prefill, decode] }
    plan: pd-pair
    profiles: { prefill: prefill, decode: decode }
  - match: { model: "*" }
    profile: cache_aware
```
