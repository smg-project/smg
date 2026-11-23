# 07. The gRPC stress script's gateway

Dialect: CLI argv.

## Input

```
smg --host 127.0.0.1 --port 30000 --enable-igw --policy cache_aware \
  --prometheus-host 127.0.0.1 --prometheus-port 29090 --runtime-worker-threads 8 --log-level warn
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  prefill:
    filter: [healthy, not-overloaded, has-capacity, not-tried]
    score:
      - warmup-slice: { weight: 8 }
      - prefix-affinity: { weight: 4 }
      - kv-pressure
      - least-load
    pick:
      best: { sample: 16 }
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
    score:
      - warmup-slice: { weight: 8 }
      - prefix-affinity: { weight: 4 }
      - kv-pressure
      - least-load
    pick:
      best: { sample: 16 }
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
