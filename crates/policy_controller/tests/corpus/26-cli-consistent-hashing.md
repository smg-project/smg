# 26. Consistent hashing

Dialect: CLI argv.

## Input

```
smg launch --worker-urls http://worker-a:8000 http://worker-b:8000 http://worker-c:8000 --policy consistent_hashing
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
view:
  headers: [x-smg-routing-key, x-smg-target-worker]
profiles:
  consistent_hashing:
    filter: [healthy, not-overloaded, not-tried]
    score: [consistent-hash]
    pick: best
bindings:
  - match: { model: "*" }
    profile: consistent_hashing
```
