# 05. The A/B simulator's gateway with power_of_two

Dialect: CLI argv.

## Input

```
smg --host 127.0.0.1 --port 30000 --enable-igw --policy power_of_two
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  power_of_two:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: power-of-two
bindings:
  - match: { model: "*" }
    profile: power_of_two
```
