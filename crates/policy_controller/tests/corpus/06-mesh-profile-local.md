# 06. The local mesh profile: several gateways, cache-aware, mesh flags

Dialect: CLI argv.

## Input

```
smg --host 127.0.0.1 --port 30100 --policy cache_aware \
  --worker-urls http://127.0.0.1:30200 http://127.0.0.1:30201 \
  --enable-mesh --mesh-host 127.0.0.1 --mesh-port 39527 \
  --mesh-server-name gw-0 --mesh-peer-urls 127.0.0.1:39528 \
  --prometheus-port 29000 --prometheus-host 127.0.0.1 --log-level info
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score: [prefix-affinity, kv-pressure, least-load, warmup-slice]
    pick: best
bindings:
  - match: { model: "*" }
    profile: cache_aware
```
