# Grand Central

Grand Central chooses the worker that serves each request. You write the rules in a *policy spec*, a YAML or JSON document. The same Rust library runs inside a host process and behind a standalone gRPC service.

## Try it

Run one decision from a checkout. It needs no running service.

```sh
cargo test -p smg-grand-central --locked --lib the_pick_orders_targets_best_first_and_names_the_survivors
```

The test asks for a worker among three, with five, one and three requests in flight. It checks that the worker with one request comes first.

This policy spec describes the profile the test runs:

```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [healthy, not-tried]
    score: [least-requests]
    pick: best
```

A *profile* is one way to choose a worker. A *binding* says which models use a profile. This spec has no bindings, so every model uses its one profile.

## How a decision works

A profile runs three steps over the workers that serve the request's model.

1. **Filter.** Each filter drops workers that should not take the request. A *hard* filter may drop every worker. A *soft* filter steps aside when it would drop the last ones.
2. **Score.** Each scorer rates the workers that are left from 0 to 1. Above, `least-requests` rates the worker with the fewest requests in flight highest.
3. **Pick.** One picker chooses from the scores. A pick names the chosen worker and up to three alternates, best first.

A decision that finds no worker is a *shed*: a refusal that names its reason and the stage that stopped it. By default a single profile first tries again with the overload veto lifted, then with liveness doubts lifted. See [Decisions and refusals](REFERENCE.md#decisions-and-refusals).

A profile can weigh one scorer over another:

```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - least-load: { weight: 3 }
      - least-requests
    pick: best
```

The combined score is the weighted average of the scorers. A bare name has weight 1, so `least-load` counts for three quarters here.

## Built-in stages

A *stage* is one filter, scorer or picker. The [stages page](docs/stages.md) has each stage's rules, defaults and limits.

### Filters

<!-- filters: begin -->
| Name | What it keeps | Parameters |
|---|---|---|
| `healthy` | Hard. Ready workers that no drain, breaker, stall or veto holds back. | None |
| `not-overloaded` | Hard. Workers that are not overloaded. | None |
| `has-capacity` | Hard. Workers with a free prefill admission slot. | None |
| `has-room` | Soft. Workers with room in the decode window, or under a rule the spec writes. | `used`, `capacity`, `headroom` |
| `not-tried` | Soft. Workers that no earlier attempt tried and the host does not exclude. | None |
<!-- filters: end -->

### Scorers

<!-- scorers: begin -->
| Name | What it favors | Parameters beside `weight` |
|---|---|---|
| `least-load` | The shortest estimated wait, from load reports and requests in flight. | `kv_pressure_weight`, `mean_prefill_tokens`, `default_throughput` |
| `least-requests` | The fewest requests in flight. | None |
| `fewest-sessions` | The fewest sessions with a request in flight. | None |
| `session-affinity` | The worker the request's session is pinned to. | None |
| `prefix-affinity` | The deepest holders of the prompt's prefix, within its spill and pressure rules. | `threshold`, `source`, `spill` |
| `kv-pressure` | No worker. It sets the KV pressure limits that the affinity stages read. | `spread`, `ceiling` |
| `warmup-slice` | Young or thin-cache workers, for a share of the cache misses. A single request warms nobody while the model's index holds no block. | `secs`, `share`, `blocks`, `thin_ratio`, `divert_every`, `miss_share` |
| `bucket` | The worker that owns the prompt's size range, at prefill. | `abs_threshold`, `rel_threshold`, `adjust_interval_s` |
| `consistent-hash` | The routing key's owner on a hash ring, then its successors. | `prefix_tokens`, `bounded_load` |
| `resident-work` | The least resident work. | `request_tokens` |
| `least-cost` | Workers within a margin of the lowest weighted sum of named fields. | `terms`, `margin`, `unreported` |
<!-- scorers: end -->

### Pickers

<!-- pickers: begin -->
| Name | What it picks | Parameters |
|---|---|---|
| `best` | The highest combined score, with a seeded draw among ties. | `sample` |
| `round-robin` | The next worker in rotation. | None |
| `random` | A worker drawn by the request's seed. | None |
| `power-of-two` | The better-scored of two random workers. | None. Needs a scorer. |
| `count-first` | The fewest requests in flight, then the highest score. | None |
| `power-of-two-count-first` | Of two random workers, the one with fewer requests in flight. | None. Needs a scorer. |
| `passthrough` | The first worker in the host's order. | None |
| `manual` | The worker that the binding's session rule assigns or pins. | None. Takes no scorer. |
| `best-with-temperature` | Nothing yet. The name is reserved and the validator refuses it. | `temperature` |
<!-- pickers: end -->

## Run it

Start the SMG gateway with `--grand-central on` to route by Grand Central, in the gateway's own process. Add `--policy-spec FILE` to run a policy spec of your own. A gateway whose every router is a third-party provider's starts no engine: see [the options](docs/design.md#93-the-options).

### In a host process

A host program links the library and decides in its own process. The example program is a small host:

```sh
cargo run -p smg-grand-central --features in-process --example in_process
```

It does three things:

- It starts the engine once, over its worker registry (`InProcess::start`).
- It asks for a worker with one call for each attempt of a request (`InProcess::place`).
- It reports when each request went out and how it ended.

A host that keeps its own list of workers can call `Engine::decide` directly, as the test under [Try it](#try-it) does.

### As a standalone service

The service is one binary. Build it from a checkout with `protoc` installed. Save the first policy spec above as `policy.yaml` in the checkout root, then run:

```sh
cargo build --release --locked -p smg-grand-central --features controller
target/release/smg-grand-central --controller-id controller-0 --spec-file policy.yaml \
  --gateway-listen 127.0.0.1:30100 --control-listen 127.0.0.1:30101 \
  --replication-listen 127.0.0.1:30102 --admin-listen 127.0.0.1:30103 \
  --identity-as-told gateway,control
```

- Register each worker with the `RegisterWorker` call on the control listener.
- Ask for a worker with `Session` and `Decide` on the gateway listener.
- The service checks a changed policy spec before it replaces the running one.

[The controller's wire test](tests/controller.rs) shows registration and the decision stream end to end. The reference lists every [flag](REFERENCE.md#standalone-service) and every [call](REFERENCE.md#remote-calls-from-a-gateway).

## Presets

A *preset* is a complete policy spec under a name. `PolicySpec::preset("least_load")` builds one.

| Preset | How it chooses |
|---|---|
| `random` | Draws among the workers that pass its filters. |
| `round_robin` | Rotates through the workers. |
| `passthrough` | Takes the first worker in the host's order. |
| `power_of_two` | Compares the estimated waits of two random workers. |
| `least_load` | Favors the shortest estimated wait. |
| `consistent_hashing` | Favors the routing key's owner on a hash ring. |
| `prefix_hash` | Places by the routing key, else by the prompt's prefix. |
| `manual` | Places a routing key at random once, then keeps it there. |
| `cache_aware` | Blends cache warm-up, prefix reuse and least load. |
| `bucket` | Groups prefill prompts by size. The standalone service refuses it. |

Each preset has one profile, the same three filters and a binding for every model. The reference shows [each preset's spec](REFERENCE.md#presets).

## Extend it

A host can add stages of its own in three ways:

- **A Rust stage.** Implement `Filter`, `Scorer` or `Picker` and register it in the host's stage catalog. See [Register a Rust stage](REFERENCE.md#register-a-rust-stage) and the [example program](../../examples/grand-central).
- **A WebAssembly component.** Compile a component for the `smg:policy` package and load it in the host. See [Load a WebAssembly component](REFERENCE.md#load-a-webassembly-component) and the [WIT file](contracts/smg-policy.wit).
- **A gRPC plugin.** Implement the `PolicyPlugin` service and let the host call it. See [Call a gRPC plugin](REFERENCE.md#call-a-grpc-plugin) and the [service contract](contracts/policy_plugin.proto).

## Reference

[REFERENCE.md](REFERENCE.md) holds the detail. It still calls the policy spec "the policy document".

- [The policy spec](docs/policy-spec.md): its fields, [bindings](docs/policy-spec.md#bindings) and [plans](docs/policy-spec.md#plans)
- [Prefill and decode pairs](docs/policy-spec.md#prefill-and-decode) and [encode with a pair](docs/policy-spec.md#encode-with-a-pair)
- [Overload, shadows and reserved fields](REFERENCE.md#overload-shadows-and-reserved-fields) and [session keys](REFERENCE.md#session-keys)
- [Signals a decision can read](REFERENCE.md#signals-a-decision-can-read)
- [Decisions and refusals](REFERENCE.md#decisions-and-refusals) and [updating a policy spec](REFERENCE.md#updating-a-document)
- [Explain and metrics](REFERENCE.md#explain-and-metrics)
- [Worker discovery](REFERENCE.md#worker-discovery) for the standalone service
- [Cost and regression gates](REFERENCE.md#cost-and-regression-gates)
- [The design document](docs/design.md)

## Not built yet

- An SMG gateway that asks a standalone service for its decisions.
- A binding that matches on a tenant, a priority, a header, a prompt size or labels.
- `canary`, `override: allowed` and `admission` on a binding.
- A policy spec field that sets up a plugin. Host code sets up every plugin.
- The `bucket` scorer in the standalone service. The library runs it.
- The `best-with-temperature` picker.
- Worker discovery in the stock image.

The reference has the [whole list](REFERENCE.md#what-is-not-built-yet).
