# 28. Cache-aware with the hash index and every event-driven knob set

Dialect: CLI argv.

## Input

```
smg launch --worker-urls http://worker-a:8000 http://worker-b:8000 --policy cache_aware \
  --cache-index hash --cache-boundaries 512,2048 --cache-ttl-secs 300 \
  --selection-temperature 0.05 --overlap-decay 0.5 \
  --balance-token-usage-threshold 0.3 --overload-token-usage-threshold 0.9 \
  --selection-accounting-ttl-ms 200 --balance-prefix-margin-tokens 1024 --balance-uncached-credit \
  --worker-warmup-secs 0
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
view:
  boundaries: [512, 2048]
index:
  "*": { kind: hash, ttl_s: 300 }
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - warmup-slice: { weight: 8, secs: 0 }
      - prefix-affinity:
          weight: 4
          decay: 0.5
          accounting_ttl_ms: 200
          spill: { prefix_margin_tokens: 1024, uncached_credit: true }
      - kv-pressure: { spread: 0.3, ceiling: 0.9 }
      - least-load
    pick: { best-with-temperature: { temperature: 0.05 } }
bindings:
  - match: { model: "*" }
    profile: cache_aware
```
