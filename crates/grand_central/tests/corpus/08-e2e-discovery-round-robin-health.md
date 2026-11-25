# 08. The discovery end-to-end harness: round-robin with tight health probing

Dialect: CLI argv.

## Input

```
smg --host 127.0.0.1 --port 31000 --prometheus-port 31001 \
  --discovery-provider file --discovery-file /run/smg/workers.json --discovery-check-interval-secs 1 \
  --policy round_robin --health-check-interval-secs 1 --health-failure-threshold 2 \
  --health-success-threshold 1 --drain-settle-secs 0 --worker-startup-timeout-secs 30
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
