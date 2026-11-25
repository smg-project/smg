# 25. The manual policy with a min-group first placement

Dialect: CLI argv.

## Input

```
smg launch --worker-urls http://worker-a:8000 http://worker-b:8000 --policy manual \
  --assignment-mode min_group --max-idle-secs 3600 --eviction-interval 30
```

## Expected spec

```yaml
apiVersion: smg.policy/v1
generation: 1
view:
  headers: [x-smg-routing-key]
profiles:
  manual:
    filter: [healthy, not-overloaded, not-tried]
    score: []
    pick: manual
bindings:
  - match: { model: "*" }
    profile: manual
    sessions:
      sticky_header: x-smg-routing-key
      ttl_s: 3600
      sweep_s: 30
      assignment: min_group
      inflight_cap: 65535
      issue_session_id: false
```
