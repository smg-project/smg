# 27. Prefix hash with bounded load and cache boundaries

Dialect: CLI argv.

## Input

```
smg launch --worker-urls http://worker-a:8000 http://worker-b:8000 --policy prefix_hash \
  --prefix-token-count 512 --prefix-hash-load-factor 1.5 --prefix-hash-balance-abs-threshold 4 \
  --cache-boundaries 256,1024,4096
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
view:
  boundaries: [256, 1024, 4096]
profiles:
  prefix_hash:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - consistent-hash:
          weight: 8
          prefix_tokens: 512
          bounded_load: { factor: 1.5, abs_margin: 4 }
      - least-requests
    pick: best
bindings:
  - match: { model: "*" }
    profile: prefix_hash
    routing_key:
      sources:
        - header: x-smg-routing-key
```
