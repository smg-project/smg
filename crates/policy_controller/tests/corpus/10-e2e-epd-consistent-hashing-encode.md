# 10. The end-to-end encode-prefill-decode gateway

Dialect: launcher argv.

## Input

```
python -m smg.launch_router --host 127.0.0.1 --port 30000 --prometheus-port 29090 --policy cache_aware \
  --epd-disaggregation --encode http://encode-a:8000 8998 --prefill http://prefill-a:8000 8998 \
  --decode http://decode-a:8000 --encode-policy consistent_hashing --multimodal-tensor-transport inline
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
view:
  headers: []
profiles:
  encode:
    filter: [healthy, not-overloaded, not-tried]
    score: [consistent-hash]
    pick: best
  prefill:
    filter: [healthy, not-overloaded, has-capacity, not-tried]
    score: [prefix-affinity, kv-pressure, least-load, warmup-slice]
    pick: best
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
    score: [prefix-affinity, kv-pressure, least-load, warmup-slice]
    pick: best
bindings:
  - match: { model: "*", phase: [encode, prefill, decode] }
    plan: epd
    profiles: { encode: encode, prefill: prefill, decode: decode }
```
