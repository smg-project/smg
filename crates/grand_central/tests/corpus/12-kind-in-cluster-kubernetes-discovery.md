# 12. The in-cluster discovery manifest: Kubernetes discovery, round-robin

Dialect: launcher argv (a Kubernetes container `args:` list).

## Input

```
- --discovery-provider
- kubernetes
- --selector
- app=engines-incluster
- --service-discovery-port
- "8080"
- --service-discovery-namespace
- default
- --policy
- round_robin
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  round_robin:
    filter: [healthy, not-overloaded, not-tried]
    score: []
    pick: round-robin
bindings:
  - match: { model: "*" }
    profile: round_robin
```
