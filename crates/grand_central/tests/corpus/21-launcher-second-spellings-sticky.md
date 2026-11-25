# 21. The launcher's second spellings with sticky sessions

Dialect: launcher argv.

## Input

```
python -m smg.launch_router --worker-urls http://worker-a:8000 http://worker-b:8000 --policy cache_aware \
  --cache-match-threshold 0.4 --spill-abs-threshold 48 --spill-rel-threshold 1.3 \
  --eviction-interval-secs 90 --sticky-sessions --sticky-key-idle-secs 600
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
view:
  headers: [x-smg-routing-key]
index:
  "*": { sweep_s: 90 }
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - warmup-slice: { weight: 8, miss_share: 0.4 }
      - prefix-affinity:
          weight: 4
          threshold: 0.4
          spill: { abs: 48, rel: 1.3 }
      - kv-pressure
      - least-load
    pick:
      best: { sample: 16 }
bindings:
  - match: { model: "*" }
    profile: cache_aware
    sessions:
      sticky_header: x-smg-routing-key
      headers: [x-smg-routing-key]
      ttl_s: 600
      sweep_s: 90
      assignment: delegate
      inflight_cap: 2
      issue_session_id: false
```
