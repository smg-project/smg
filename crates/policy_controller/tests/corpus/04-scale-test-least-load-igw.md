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
  least_load:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
bindings:
  - match: { model: "*" }
    profile: least_load
```
