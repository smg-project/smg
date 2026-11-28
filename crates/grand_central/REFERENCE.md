# Grand Central

Grand Central chooses which workers serve a request. One versioned YAML or JSON document composes health and capacity filters, weighted scores for load or cache locality, and a picker with alternates. Bind profiles by model and phase, keep related requests on a worker, or choose compatible workers for split requests. The same Rust library runs in a host process and behind the standalone gRPC service. The service validates a new document before it replaces the running one and can explain sampled picks and sheds.

## First decision

From a checkout, run the small in-process example with three ready workers. It checks that the worker with one request in flight comes first, followed by the workers with three and five. It needs no running service.

```sh
cargo test -p smg-grand-central --locked --lib the_pick_orders_targets_best_first_and_names_the_survivors
```

This document describes that profile. With one profile, an omitted or empty `bindings` list uses it for every model and phase under the `single` plan.

```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [healthy, not-tried]
    score: [least-requests]
    pick: best
```

1. **Filter.** Start with the model's pool for the requested role. Hard filters run first, in their written order. An empty hard pass usually sheds. A soft filter that would empty the pool leaves it as it was before that filter. Overload and liveness have [fallback rules](docs/guarantees.md#decisions-and-refusals).
2. **Score.** Each scorer rates the rows it is asked to compare from 0 to 1, higher first. The combined score is a weighted average. A bare name has weight 1. A mapping puts the weight beside that scorer's other parameters. In the next document the weights give `least-load` three quarters of the combined score.
3. **Pick.** Every profile has one picker. A pick carries up to four ordered targets. A shed gives a reason and a stage or constraint: the one that stopped the decision, but for two sheds of a pair. The picker can narrow the comparison before scoring.

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

## Built-in stages

The stock filters, scorers and pickers stand on [a page of their own](docs/stages.md), each with its rules, its parameters and their defaults.

## Signals a decision can read

What feeds each signal, and the views a host supplies, stand on [the embedding page](docs/embedding.md#signals-a-decision-can-read).

## Run it

The standalone service stands on [a page of its own](docs/service.md): its flags, its worker discovery, the calls it serves and the client a host calls it with. A host that decides in its own process reads [Embed the engine](docs/embedding.md).

## Extend it

A Rust stage, a WebAssembly component and a gRPC plugin stand on [a page of their own](docs/extending.md).

### Register a Rust stage

[Register a Rust stage](docs/extending.md#register-a-rust-stage) on that page shows how a host registers one. [A stage with parameters](docs/extending.md#a-stage-with-parameters) shows a scorer with typed parameters.

## Policy document reference

The document's fields, its bindings and its plans stand on a page of their own: [The policy spec](docs/policy-spec.md).

## Guarantees and limits

What a decision books, what the service refuses and how a spec is updated stand on [a page of their own](docs/guarantees.md).

## Explain and metrics

Sampled decisions, the service's series, the plugins' stats and the gateway's routing metrics stand on [a page of their own](docs/metrics.md).

## Presets

`PolicySpec::preset("least_load")` builds a complete document with generation 1. Each preset has one profile, the same three filters and a binding for every model. The copies below leave the number to the source. `PolicySpec::power_of_two` also accepts `ExpectedWait`, `Requests`, `LeastRequests` or `ResidentTokens { request_tokens }`. The resident variant ranks the full set by resident work. `pair_presets` and `epd_presets` compose side profiles from preset names and add `has-capacity` to the prefill side and `has-room` to the decode side. The standalone controller places a pair at its prefill, and reads both bits open. [Preset constructors](src/document/spec/presets.rs)

<details>
<summary>Preset documents</summary>

### `random`

Draws among the workers that pass its filters. The picker uses no score. The overload filter can still use load reports.

<!-- preset: random -->
```yaml
apiVersion: smg.policy/v1
profiles:
  random:
    filter: [healthy, not-overloaded, not-tried]
    pick: random
bindings:
  - match: { model: "*" }
    profile: random
```

### `round_robin`

Rotates through a worker list or a survivor set. A single request's first attempt, held to no part of its list, takes the next worker of its list's rotation where the filters keep that worker: at every list size over a list that has a counter of its own, as the lists of the engine over worker lists (`InProcess`) have, and from 16 workers over any other. Every other placement rotates through its survivor set, a retry among them: changes to that set change the rotation, and retained rotations are bounded.

<!-- preset: round_robin -->
```yaml
apiVersion: smg.policy/v1
profiles:
  round_robin:
    filter: [healthy, not-overloaded, not-tried]
    pick: round-robin
bindings:
  - match: { model: "*" }
    profile: round_robin
```

### `passthrough`

Takes the first survivor in host order.

<!-- preset: passthrough -->
```yaml
apiVersion: smg.policy/v1
profiles:
  passthrough:
    filter: [healthy, not-overloaded, not-tried]
    pick: passthrough
bindings:
  - match: { model: "*" }
    profile: passthrough
```

### `power_of_two`

Compares the estimated waits of two draws. A full filter pass can still be needed.

<!-- preset: power_of_two -->
```yaml
apiVersion: smg.policy/v1
profiles:
  power_of_two:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: power-of-two
bindings:
  - match: { model: "*" }
    profile: power_of_two
```

### `least_load`

Favors the best estimated wait, including active KV pressure. Uses in-flight counts when the compared set has no reports.

<!-- preset: least_load -->
```yaml
apiVersion: smg.policy/v1
profiles:
  least_load:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
bindings:
  - match: { model: "*" }
    profile: least_load
```

### `consistent_hashing`

Favors the routing key's ring owner while it is usable. With no session key the request is placed by its client's key, read from `x-forwarded-for`, else `cookie`. With neither, the scores tie. The binding names the routing key header as the key's source, and the two client headers behind it. A sessions rule added to the binding reads the same key and names no header of its own.

<!-- preset: consistent_hashing -->
```yaml
apiVersion: smg.policy/v1
profiles:
  consistent_hashing:
    filter: [healthy, not-overloaded, not-tried]
    score: [consistent-hash]
    pick: best
bindings:
  - match: { model: "*" }
    profile: consistent_hashing
    routing_key:
      sources:
        - header: x-smg-routing-key
        - header: x-forwarded-for
          client: true
        - header: cookie
          client: true
        - header: x-smg-target-worker
          target: true
```

### `prefix_hash`

Uses the routing key first, then the prompt's prefix: every block from the first through the last whole block at the deepest reached boundary, or at the configured 256-token depth when no boundary is reached. On a ring the request's caller lends, the prefix ends at the token of that boundary or depth, and no block is rounded off. Two prompts that differ anywhere in that prefix take different keys, so they spread over the workers. When no key can be used, the in-flight-count scorer places the request. The binding names the routing key header as the key's source. A sessions rule added to the binding names no header of its own.

<!-- preset: prefix_hash -->
```yaml
apiVersion: smg.policy/v1
profiles:
  prefix_hash:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - consistent-hash:
          weight: 8
          prefix_tokens: 256
          bounded_load: { factor: 1.25, abs_margin: 10 }
      - least-requests
    pick: best
bindings:
  - match: { model: "*" }
    profile: prefix_hash
    routing_key:
      sources:
        - header: x-smg-routing-key
```

### `manual`

Uses a host-resolved key from the routing header with random first placement and preference for the original pin on recovery. This preset sweeps every 60 seconds and disables the in-flight cap.

<!-- preset: manual -->
```yaml
apiVersion: smg.policy/v1
view:
  headers: [x-smg-routing-key]
profiles:
  manual:
    filter: [healthy, not-overloaded, not-tried]
    pick: manual
bindings:
  - match: { model: "*" }
    profile: manual
    sessions:
      sticky_header: x-smg-routing-key
      assignment: random
      ttl_s: 14400
      sweep_s: 60
      inflight_cap: 65535
```

### `cache_aware`

Blends warm-up, prefix reuse and least-load. The default KV gate is off. The stock controller reads cache events from each gRPC worker whose row asks for them and names a model, and its decisions read them. There the preset works from the cached prefixes of each worker the index holds a block for, and from worker age and its load monitor's reports. It works from the prefixes sent to every worker for a text request and while the model's index holds no block, and on a pair's side from the prefixes sent to each worker the index holds no block for. While the model's index holds no block, a single request warms no worker; a pair's side still warms a young one.

The preset's prefix stage names no source, so it reads `merged`. Set `source: index` to read cache events alone, or `source: predictor` to read sent prefixes alone. A standalone controller's decision reads the cache events of its own workers' streams, as a gateway's does.

<!-- preset: cache_aware -->
```yaml
apiVersion: smg.policy/v1
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - warmup-slice: { weight: 8 }
      - prefix-affinity: { weight: 4 }
      - kv-pressure
      - least-load
    pick:
      best: { sample: 16 }
bindings:
  - match: { model: "*" }
    profile: cache_aware
```

### `bucket`

Groups prefill prompts by size in a host that reports the lifecycle. The standalone validator refuses this preset.

<!-- preset: bucket -->
```yaml
apiVersion: smg.policy/v1
profiles:
  bucket:
    filter: [healthy, not-overloaded, not-tried]
    score: [bucket]
    pick: best
bindings:
  - match: { model: "*" }
    profile: bucket
```

</details>

## Cost and regression gates

The decision benchmarks time view construction and policy execution separately, typically at 8, 64 and 512 candidates. They cover presets, dispatch credit, overloaded pools, sticky hits and misses, spec validation, cache locality, sampling and prefix keying. The store benchmarks time membership writes, publication, update journals and pairing. Library benches also measure encode placement, pair and encode class construction, concurrent callers, the sent-prefix predictor, election, and both plugin kinds. The `decide_stream` bench measures loopback round-trip percentiles and throughput under offered load, separately from in-process work. [Benchmark sources](benches)

```sh
cargo bench -p smg-grand-central --locked --bench decision
```

| Gate | Work measured | Threshold by row |
|---|---|---|
| [Engine](../../.github/bench-gates/grand-central-engine.toml) | View build, decisions and spec parse/validation. | 10–15% |
| [Affinity](../../.github/bench-gates/grand-central-engine-affinity.toml) | Affinity decisions and selected sizes of view builds. | 10% |
| [Bucket](../../.github/bench-gates/grand-central-engine-bucket.toml) | Bucket decisions, including prefill across models and with stale rows. | 10–20% |
| [Power of two](../../.github/bench-gates/grand-central-engine-power-of-two.toml) | The preset's load and count variants. | 10% |
| [Prefix hash](../../.github/bench-gates/grand-central-engine-prefix-hash.toml) | Ring owner and load spill paths, with and without boundaries. | 10–15% |
| [Store](../../.github/bench-gates/policy-store.toml) | Load and membership writes, publication and selected pair and many-model cases. | 10–40% |
| [Store journal](../../.github/bench-gates/policy-store-journal.toml) | Journaled writes and journal reads. | 10–30% |
| [Gateway policy selection](../../.github/bench-gates/policy-selection.toml) | The gateway's regular placement benchmarks. | 10% |

The gates run three interleaved measurements per side when timing is needed. Each benchmark ID uses the first matching threshold. A time regression must exceed that threshold in the median of the runs, with every changed run slower than every base run. Overlap is reported as suspect. Some gates also repeat suspect measurements or check for executable placement effects before failing a row. Missing gated IDs are detected. These are relative regressions for the measured paths. The plugin and wire round-trip benchmarks have no dedicated policy gate here. [Gate rules and report](../../scripts/bench_gate.py)

## A worker's row

A worker's row, its bounds, the door that writes it and the worker jobs of a host with a local store stand on [a page of their own](docs/worker-rows.md).

## What is not built yet

- A client that marks a request as asked again. The controller serves the mark. The crate's own client does not set it yet.
- A client that asks a served controller for a pair's later decode or for its decode call. The controller serves both. The crate's own client states neither yet.
- A choice by the media a worker can process, outside `epd`. No built-in filter reads it.
- A binding that matches on a tenant, a priority, a header, a prompt size or labels. The fields are in the schema. A document that writes one is refused as not in this build.
- A pair's shadows at an encode request's first decision, beside a side its sessions rule or its picker's assignment places and through a served controller, and plugin stages in a shadow profile.
- `canary`, `override: allowed` and `admission` on a binding, and `issue_session_id` in a sessions rule.
- A sessions rule under `pd-pair` or `epd` that states `rule_placed: untold`, through the stock controller. Its file of leases keeps no word of a pair's side. In a host's own process a rule runs on both sides of a pair under either plan, or under `pd-pair` on the one side `sides` names.
- A list of `sides` in a sessions rule under `epd`. The field is the `pd-pair` plan's, and under `epd` a rule covers the pair's two sides.
- The `best-with-temperature` picker, and three settings of `prefix-affinity`: `decay`, `accounting_ttl_ms` and `spill.uncached_credit`.
- The `bucket` scorer in the stock controller. The library runs it. The stock controller refuses a document that names it, because it does not yet report every request's end to a stage.
- The record of sent prefixes on a standby. A standby starts with an empty record.
- The wire's `prompt_text`. A controller does not hash a foreign host's text yet. It answers such a request `Busy`, as before.
- Cache events from an engine that asks for a key, or from a data-parallel worker's rows above rank 0. The stock controller builds its worker objects without credentials, so an engine that asks for a key gives it no stream. Every replica of the stock controller opens a cache-event stream to each gRPC worker whose row asks for events and names a model, and keeps those streams while the active document reads no index. A host built with the `worker-host` feature has the feed's two parts: `worker_host::KvAdapter` writes a worker's cache events to the index, and `worker_host::KvHost` gives each event subscription its row. A host that opens a row's feed without the row's cache group state clears the row's blocks and cursor first, so the subscription starts at the head of the stream.
- One more input in the stock controller. A worker's circuit-breaker bit is read from two places there: the worker's own object, and a veto of the asking gateway that holds, as its breaker's open window does. Such a veto takes the worker out of that gateway's decisions, and no rule lifts it. Its row reads as stalled too, as under every veto of that gateway, so a filter that reads the stalled bit refuses it. Nothing in the controller records a request's end on the object's breaker yet, so that bit reads closed.
- An engine connection over the host-local ZMQ wire. A controller built with the `worker-host` feature hands each worker object a gRPC connector alone, and an object connects to its engine at its first use. A worker that serves only the ZMQ wire gets no connection there, so nothing that controller reads from an engine reaches it for that worker. The controller sends such a worker no health probe: a row that names its one gateway is host-local, and the controller cannot dial it. Its row turns ready by its registration's connect signal, which a row whose policy turns probes off gets; with its probes on, nothing readies it yet. No field of a registration or of a worker object names a gateway yet, so the controller's doors refuse a worker of that wire.
- A probe with a key. The controller's probe of a worker carries no API key until a host hands the worker objects a keeper of credentials. An HTTP worker's URL userinfo goes out as Basic authentication, as the HTTP client sends it for every call. A gRPC worker's channel cuts the userinfo, so its probe carries no credential.
- A count of the controller's probes in its own metrics. The worker crate counts each probe in a recorder the controller does not install.
- A probe schedule kept across a change of leader. A new leader probes at once each worker that has no verdict on that replica, and each other worker after its own interval.
- A driver for the leader's own doubt of a worker. The store takes no such write, and a standalone controller's decision reads a worker's stalled bit on the worker's own object and in the asking gateway's vetoes. The object's bit is written by the worker crate's liveness check, which runs with its load monitor: a replica runs that monitor while it leads.
- A plugin named in the document. The format has no field for one.
- Two calls of the contract: `StepDown` and `WatchRegime`. The stock controller answers each as not served.
- Seven of the nine slices of the snapshot call, `GetSnapshot`. The stock controller serves the worker rows and the spec's status. It refuses a request for any other slice by the slice's name, and a request that names no slice asks for all nine. It also refuses a request for identifiers in clear.
- The audit mode of the apply call.
- A key for the workers discovery finds. The controller refuses one at its start. It keeps no worker key, and a replica that took the lead could not learn the key of a worker found before it: a pass registers no worker whose row stands.
- Flags for the timing of a discovery registration, and a found worker's cache-event word written again at an apply.
- A client identity and a private CA for the dial of a found worker.
- A registration that ends ahead of its dial where another maker holds the worker's row.
- Discovery in the stock image, which does not build the feature.
- A gateway that asks a standalone controller for its decisions.
