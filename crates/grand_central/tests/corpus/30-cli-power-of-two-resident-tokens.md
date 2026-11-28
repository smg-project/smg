# 30. Power-of-two with the resident-tokens metric

Dialect: CLI argv.

## Input

```
smg launch --worker-urls http://worker-a:8000 http://worker-b:8000 --policy power_of_two --power-of-two-load-metric resident_tokens:512
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  power_of_two:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - resident-work: { request_tokens: 512 }
    pick: best
bindings:
  - match: { model: "*" }
    profile: power_of_two
```
