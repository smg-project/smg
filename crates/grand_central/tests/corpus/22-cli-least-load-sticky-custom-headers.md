# 22. Least-load with sticky sessions on custom headers and a first-seen placement rule

Dialect: CLI argv.

## Input

```
smg launch --worker-urls http://worker-a:8000 http://worker-b:8000 --policy least_load \
  --routing-key-override --routing-key-headers x-session-id x-smg-routing-key \
  --max-idle-secs 600 --assignment-mode min_load --least-load-kv-pressure-weight 0.3
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
view:
  headers: [x-session-id, x-smg-routing-key]
profiles:
  least_load:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - least-load: { kv_pressure_weight: 0.3 }
    pick: best
bindings:
  - match: { model: "*" }
    profile: least_load
    sessions:
      sticky_header: x-session-id
      headers: [x-session-id, x-smg-routing-key]
      ttl_s: 600
      sweep_s: 120
      assignment: min_load
      inflight_cap: 2
      issue_session_id: false
```
