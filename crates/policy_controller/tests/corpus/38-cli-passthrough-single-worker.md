# 38. Passthrough on a single worker

Dialect: CLI argv.

## Input

```
smg launch --worker-urls http://worker-a:8000 --policy passthrough
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  passthrough:
    filter: [healthy, not-overloaded, not-tried]
    score: []
    pick: passthrough
bindings:
  - match: { model: "*" }
    profile: passthrough
```
