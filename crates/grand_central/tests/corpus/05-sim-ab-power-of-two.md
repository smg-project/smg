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
  encode:
    filter: [healthy, not-overloaded, not-tried]
    score: [consistent-hash]
    pick: best
  prefill:
    filter: [healthy, not-overloaded, has-capacity, not-tried]
    score: [least-load]
    pick: power-of-two
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
    score: [least-load]
    pick: power-of-two
  power_of_two:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: power-of-two
bindings:
  - match: { model: "*", phase: [encode, prefill, decode] }
    plan: epd
    profiles: { encode: encode, prefill: prefill, decode: decode }
  - match: { model: "*" }
    profile: power_of_two
```
