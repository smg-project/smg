# 24. Round-robin with overload protection switched off

Dialect: CLI argv.

## Input

```
smg launch --worker-urls http://worker-a:8000 http://worker-b:8000 --policy round_robin --disable-worker-overload-protection
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  round_robin:
    filter: [healthy, not-overloaded, not-tried]
    score: []
    pick: round-robin
bindings:
  - match: { model: "*" }
    profile: round_robin
    overload: { enabled: false }
```
