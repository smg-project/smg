# 17. A Kubernetes deployment's argument list: cache-aware with a 256-token block size

Dialect: launcher argv (Kubernetes `args:` list).

## Input

```
- --port
- "30000"
- --prometheus-port
- "29090"
- --policy
- cache_aware
- --block-size
- "256"
- --reasoning-parser
- <parser>
- --tool-call-parser
- <parser>
- --model-alias
- alias=canonical
- --service-discovery
- --selector
- app=engine
- --service-discovery-port
- "8000"
- --worker-startup-timeout-secs
- "1800"
- --enable-trace
- --otlp-traces-endpoint
- http://collector:4317
- --log-json
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
view:
  block_size: 256
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score: [prefix-affinity, kv-pressure, least-load, warmup-slice]
    pick: best
bindings:
  - match: { model: "*" }
    profile: cache_aware
```
