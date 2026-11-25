# 37. The random policy

Dialect: CLI argv.

## Input

```
smg launch --worker-urls http://worker-a:8000 http://worker-b:8000 --policy random
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  random:
    filter: [healthy, not-overloaded, not-tried]
    score: []
    pick: random
bindings:
  - match: { model: "*" }
    profile: random
```
