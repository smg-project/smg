# 39. Power-of-two with the least-requests metric, through the count-first picker

Dialect: CLI argv.

## Input

```
smg launch --worker-urls http://worker-a:8000 http://worker-b:8000 --policy power_of_two --power-of-two-load-metric least_requests
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  power_of_two:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: count-first
bindings:
  - match: { model: "*" }
    profile: power_of_two
```
