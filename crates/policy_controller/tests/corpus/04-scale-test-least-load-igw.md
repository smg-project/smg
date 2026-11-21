# 04. The scale test's gateway: IGW mode, least-load

Dialect: CLI argv.

## Input

```
smg --host 127.0.0.1 --port 30000 --enable-igw --policy least_load --disable-tokenizer-autoload
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  prefill:
    filter: [healthy, not-overloaded, has-capacity, not-tried]
    score: [least-load]
    pick: best
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
    score: [least-load]
    pick: best
  least_load:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
bindings:
  - match: { model: "*", phase: [prefill, decode] }
    plan: pd-pair
    profiles: { prefill: prefill, decode: decode }
  - match: { model: "*" }
    profile: least_load
```
