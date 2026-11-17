# 33. Prefill-decode with the admission window, queue and strict pairing

Dialect: CLI argv.

## Input

```
smg --pd-disaggregation --prefill http://prefill-a:8000 8998 --decode http://decode-a:8000 --decode http://decode-b:8000 \
  --policy cache_aware --pd-pairing-mode strict --prefill-max-inflight-requests-per-worker 4 \
  --prefill-queue-size 50 --prefill-queue-timeout-secs 10 --pd-admission-wait-secs 5
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  prefill:
    filter: [healthy, not-overloaded, has-capacity, not-tried]
    score: [prefix-affinity, kv-pressure, least-load, warmup-slice]
    pick: best
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
    score: [prefix-affinity, kv-pressure, least-load, warmup-slice]
    pick: best
plans:
  pd-pair: { pairing: strict }
bindings:
  - match: { model: "*", phase: [prefill, decode] }
    plan: pd-pair
    profiles: { prefill: prefill, decode: decode }
```
