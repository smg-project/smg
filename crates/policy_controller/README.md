# Policy controller

The policy controller chooses which workers serve a request. One versioned YAML or JSON document composes health and capacity filters, weighted scores for load or cache locality, and a picker with alternates. Bind profiles by model and phase, keep related requests on a worker, or choose compatible workers for split requests. The same Rust library runs in a host process and behind the standalone gRPC service. The service validates a new document before it replaces the running one and can explain sampled picks and sheds.

## First decision

From a checkout, run the small in-process example with three ready workers. It checks that the worker with one request in flight comes first, followed by the workers with three and five. It needs no running service.

```sh
cargo test -p smg-policy-controller --locked --lib the_pick_orders_targets_best_first_and_names_the_survivors
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

1. **Filter.** Start with the model's pool for the requested role. Hard filters run first, in their written order. An empty hard pass usually sheds. A soft filter that would empty the pool leaves it as it was before that filter. Overload and liveness have [fallback rules](#decisions-and-refusals).
2. **Score.** Each scorer rates the rows it is asked to compare from 0 to 1, higher first. The combined score is a weighted average. A bare name has weight 1. A mapping puts the weight beside that scorer's other parameters. In the next document the weights give `least-load` three quarters of the combined score.
3. **Pick.** Every profile has one picker. A pick carries up to four ordered targets. A shed gives a reason and the stage or constraint that stopped the decision. The picker can narrow the comparison before scoring.

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

### Filters

<!-- filters: begin -->
| Name | What it reads and keeps | Parameters and defaults |
|---|---|---|
| `healthy` | Hard. Reads ready, draining, breaker and stall flags and the asking host's veto mask. Keeps ready workers clear of those vetoes. | None |
| `not-overloaded` | Hard. Reads the latched overload bit and the host's overload lift. | None on the stage. The binding's `overload` block sets thresholds, 8 queued requests or 0.8 active KV usage by default. |
| `has-capacity` | Hard. Reads the host's bit for a free prefill admission slot. | None |
| `has-room` | Soft. Reads the host's bit for free room in the decode running window. | None |
| `not-tried` | Soft. Reads earlier attempts and the host's exclusions. Keeps rows outside both lists while any remain. | None |
<!-- filters: end -->

### Scorers

Every scorer takes a finite positive `weight`, default 1. The sum and its reciprocal must be finite. Fractions in the table include their endpoints unless stated otherwise.

<!-- scorers: begin -->
| Name | What it reads and favors | Parameters and defaults |
|---|---|---|
| `least-load` | Load reports, in-flight counts and dispatches the reports have yet to reflect. Favors the shortest estimated wait from queued and running token work, drain rate and active KV pressure. With no reports it uses in-flight counts. | `kv_pressure_weight` 0.15 (finite, ≥0), `mean_prefill_tokens` 1024 (positive), `default_throughput` 2000 tokens/s (finite, positive, used when no compared row reports a live rate). |
| `least-requests` | In-flight request counts. Favors the lowest. | Weight only |
| `fewest-sessions` | Active session counts per worker. Favors the fewest keys with requests in flight. | Weight only |
| `session-affinity` | The request's pin from the host. Scores that worker 1 and the others 0. | Weight only |
| `prefix-affinity` | Matched prefix depth and request size, with in-flight counts and an optional active KV gate. Favors the deepest holders that meet the spill and pressure rules. Index hits count at any depth. A predicted hit must exceed `threshold`. | `threshold` 0.3 (0–1), `source` `index` (`predictor` uses sent prefixes and `merged` takes the deeper match per worker, both require host support), `spill.abs` 64 requests, `spill.rel` 1.5 (finite, ≥1), `spill.prefix_margin_tokens` 0. The validator refuses positive `decay`, positive `accounting_ttl_ms` and `spill.uncached_credit: true`. |
| `kv-pressure` | A flat score. The other affinity stages that use this gate read the spread and peak of reported active KV occupancy. | `spread` 1 and `ceiling` 1 (finite, positive). Each limit at 1 or above is off. |
| `warmup-slice` | Worker admission age, cache growth and size, prefix depth and in-flight counts. Gives young or thin workers a share of misses and can divert shallow hits to refill a thin cache. | `secs` 60 for age, `share` 0.25 of misses, `blocks` 1024 for growth, `thin_ratio` 0.5 of median index size, `divert_every` 8 hits, `miss_share` 0.3 for overlap treated as a miss. The three fractions are 0–1. Zero disables the corresponding age, thin-cache or diversion rule for `secs`, `thin_ratio` or `divert_every`. |
| `bucket` | Prefill prompt size and booked tokens. Favors a size range's owner or a nearby usable owner. It spills to the least booked worker when both spread thresholds are crossed. Other phases tie. | `abs_threshold` 64 tokens, `rel_threshold` 1.5 (finite, ≥1), `adjust_interval_s` 5 for the booking window and adjustment cadence (positive). The standalone validator refuses this stage because it requires lifecycle reports. |
| `consistent-hash` | The routing key, or a prefix key when configured. Favors the owner on a ring, then its successors. A load bound can pass the owner over. | `prefix_tokens` unset. `bounded_load` unset. With that block present, `factor` defaults to 1.25 (finite, ≥1) and `abs_margin` to 10. |
| `resident-work` | Reported resident context, requests, host counts and dispatch credit. Favors the lowest total work. Uses in-flight counts when no compared report counts resident tokens. | `request_tokens` required. A whole number including 0, the token cost added per request. |
| `least-cost` | Workers within the cost margin of the least weighted sum of the named fields. | `terms` (required), `margin` (zero), `unreported` (required when a term reads a report) |
<!-- scorers: end -->

`least-cost` adds each named value times its coefficient in the document's units. It reads `candidate.in_flight` from the host and `load.running`, `load.queued`, `load.waiting_uncached_tokens`, `load.kv_usage_permille`, `load.active_kv_permille`, and `load.resident_tokens` from the report. Resident tokens are present only when the report counts them. Waiting tokens are missing when they read zero beside a queue: a report of a queue's depth alone carries no token count.

For example, `terms: { load.queued: 11, candidate.in_flight: 2.5 }` adds 11 for each queued request and 2.5 for each request in flight.

With `unreported: { over-least: { request_cost: 13 } }`, a row missing a named value costs 13 per request in flight above the least reported cost, or above zero if no row reports. `unreported: missing-as-zero` sums the available named values. The absolute margin and the relative share of the magnitude of the least cost add together. Each row within that margin scores 1. With this scorer alone, `best` draws from those rows before the others.

Coefficients can be positive or negative and must be nonzero. Numeric settings are finite. The weight, coefficient magnitudes, absolute margin, and request cost are at most `1e15`, and the relative margin is at most `1e6`. Both margin parts and the request cost are nonnegative.

### Pickers

<!-- pickers: begin -->
| Name | What it reads and how it picks | Parameters and defaults |
|---|---|---|
| `best` | Combined scores. Highest first, with a seeded draw inside the score's tie band. | `sample` is unset by default. Use `best: { sample: 16 }` for a sample. Valid sizes are 1–64. |
| `round-robin` | The ordered survivor set. Advances a rotation for that set. | None |
| `random` | The survivors and the request's seed. Draws a worker and then alternates. | None |
| `power-of-two` | Scores of two random draws. Uses two engines when the host supplies suitable engine groups. | None. Requires a scorer. |
| `count-first` | In-flight counts, then combined score for the lowest-count rows. Orders higher-count alternates by count. | None |
| `power-of-two-count-first` | Two random rows, then in-flight count, then combined score on equal counts. | None. Requires a scorer. |
| `passthrough` | Survivor order. Takes the first worker in the host's order. | None |
| `manual` | The binding's session assignment and pins. Uses `min_load` with no rule or for `delegate`. | None. The validator refuses scorers with this picker. |
| `best-with-temperature` | Reserved name. The validator refuses it. | Its reserved `temperature` must be finite and ≥0. |
<!-- pickers: end -->

The sampled `best` path needs a host that names prefix holders and warming rows, and stages that can work on a draw. At 16 or more rows and more than twice the sample size, it makes at most twice that many draws to keep up to the sample size in distinct usable rows. The named lists join the draw. Small pools, an empty draw, a missing named view, or stages that need the whole pool use the full path. Power-of-two scores its compared set. A normalized scorer can cast its full weight as a vote between two rows.

## Signals a decision can read

The engine reports a profile's input requirements through `Needs`. The host supplies the request and worker views. Native stages can read the full view. Plugins see the projection their host configures. See [the Rust view types](src/view.rs) and [the wire records](contracts/policy_plugin.proto).

| Signal | Where it comes from |
|---|---|
| Model, phase, tenant, namespace seed, priority, streaming, output limit, prompt size, media counts, retry attempt and seed | The caller's `RequestFacts`. The remote caller sends these in `DecideRequest`. A frame with raw prompt text receives `Busy`. The built-in binding matcher reads model and phase. |
| Request hashes and headers | The caller sends token-block or text-window hashes, their kind and optionally their block size. The applied document bounds the prefix read. The caller projects headers as positions in `view.headers` and resolves the routing or session key. |
| Ready, draining, stall, breaker and reachability | Membership and health probes, plus the asking gateway's vetoes and wire constraints. A liveness veto is a doubt. A held veto represents the asking gateway's breaker and cannot be lifted by the liveness fallback. The standalone controller has no separate routing breaker for workers. |
| Running, queued and waiting uncached work, KV usage and active KV usage, drain rate, resident tokens, admission window, sample age and source | Host load reports. The standalone leader polls the worker's load RPC and can scrape HTTP metrics for a registered runtime with a supported parser. Fields depend on the reporter. Use `view.max_report_age_ms` to age out reports for the affected scorers and KV gates. Overload is a separate latched verdict. |
| Host requests in flight and dispatch credit | Lease bookings and lifecycle reports. Read counts through `RequestFacts::in_flight` so a stage also sees counts lent beside rows. The load scorers can account for dispatched work until a report covers it. |
| Pins, requests in flight for one key, and active-key counts | The host's session facts, or the engine's session rule and the store's pins of record. The standalone controller counts a key's requests in flight by the leases it holds, for a rule with an in-flight cap. Its decision view does not supply active-key counts or a premarked session-hit row. Its session rule keeps pins using the caller's key. |
| Cache depth, age, growth and size | An embedded host can submit KV events to `MemoryStore` and name prefix holders, or supply an affinity column itself. It can also supply sent-prefix predictions. With regular standalone ingestion, worker age is available and cache events and sent-prefix placements are absent. A controller built with the `worker-host` feature reads cache events from each gRPC worker whose row asks for them. |
| Role, ranks, wires, stable worker key, zone, engine family and labels | Membership. The document projects labels to bits, and zone and family to positions from 1. An unlisted or absent zone or family is 0. |
| Capacity, room, pair and encode compatibility, worker vision | The host supplies admission bits and worker facts. `MemoryStore` builds pairing tables and encode classes from membership for its library callers. Its candidate rows leave the admission bits unset. |
| Pool generation, size and membership, reachability and host constraints | The pool view and `Constraints` carry the pool's facts, exclusions, vetoes, a wire mask and an optional subset hint. A standalone request can narrow by common wires or specify a zero subset to exclude the pool. Hosts can supply further pool facts such as a regime or zone capacity. The standalone view supplies neither. |

An embedded host can implement or use the store's ingress traits for membership, health, loads, cache events, reservations, sessions and leases. `MemoryStore` has an event-fed chain index that holds up to 1024 workers per model. Its event sink reports sequence gaps and supports replay and resync. A host can read the validated `index` settings through `PolicySpec::index_for` and use them to configure its own index. The bundled store uses the same index layout for every document. Feed the predictor separately when a scorer needs predicted placements. The crate's `Predictor` keeps that record: it folds a request's keys once (`Predictor::fold_into`), reads by a shared reference, and writes with a tick of the host's clock. [Store API](src/store/mod.rs) · [Sent-prefix predictor](src/predictor.rs)

## Run it

### In a host process

Validate a `PolicySpec` to get an `Engine`. Call `Engine::decide` with request facts, rows, constraints and reusable `Scratch`. For a host that needs booking, apply an `AppliedSpec` to `MemoryStore` and call `Door` or `Handle`. The door runs on the caller's thread. It issues the lease for the first target. The host takes a target, reports dispatch and completion or release, and calls `Door::sweep` on its timer. The door decides a pair when asked at its prefill: that one request decides both sides. It returns `PairSide` for a decode or an encode sent on its own, and `Unleased` or `Faulted` if it cannot book or decide a pick. `Door::decide_encode_pair` also places an `epd` request's encode items. A pick can state what its first worker holds of the request's prefix (`Pick::overlap`): the leading blocks held, the request's blocks, where the match came from, and whether the prefix margin kept that worker in front. The profile's first `prefix-affinity` scorer says it. A host that calls the library asks through `Engine::decide_stated`. The controller's `Decide` answer does not carry it yet. [Engine example](src/framework.rs) · [Door API](src/door.rs) · [Handle API](src/handle.rs)

The crate defaults to `wasm`. Use `default-features = false` for the library without that runtime. `ingest` adds load polling and scrape helpers. `controller` adds the service, its protocol, command line and gRPC plugin host, and enables `ingest`. `client` enables the reference client and watcher and implies `controller`. `kubernetes` adds the namespaced Lease backend. `operator` adds the operators' door for a host's own process and builds no server. `worker-host` adds an optional registry of worker objects that follows membership. In the controller build those objects use a lazy gRPC connector. The feature also has the cache feed's two parts: `worker_host::KvAdapter` writes a worker's cache events to the index, and `worker_host::KvHost` gives each event subscription its row. A controller built with the feature fills its index itself: every replica opens a cache-event stream to each gRPC worker whose row asks for events. The stock controller is built without it. `test-util` supports the crate's tests and benches. [Cargo features](Cargo.toml)

### Standalone service

Build from a checkout with `protoc` available. The service feature uses the worker protocol files from the checkout. Save the first policy document above as `policy.yaml`, then run the following from the checkout root.

```sh
cargo build --release --locked -p smg-policy-controller --features controller
target/release/smg-policy-controller --controller-id controller-0 --spec-file policy.yaml \
  --gateway-listen 127.0.0.1:30100 --control-listen 127.0.0.1:30101 \
  --replication-listen 127.0.0.1:30102 --admin-listen 127.0.0.1:30103 \
  --identity-as-told gateway,control
```

Register workers with `RegisterWorker` on the control listener. Use `Session` and `Decide` on the gateway listener to ask for a worker. The service returns targets and leases for the caller to dispatch and report. For a complete local example of registration and the decision stream, see [the controller's wire test](tests/controller.rs). The deployment chart can mount `policyController.spec` as a file or `policyController.specStoreClaim` as a store. [Deployment chart](../../deploy/helm/smg/README.md#policy-controller)

| Flags | Default and effect |
|---|---|
| `--help`, `--version` | Show help or version and exit. |
| `--controller-id` | Reads `HOSTNAME` if omitted. Use letters, digits, `.`, `_` and `-`. |
| `--spec-file`, `--spec-store` | Choose exactly one. A watched YAML or JSON file, or an existing directory managed through the spec calls. |
| `--gateway-listen`, `--control-listen`, `--replication-listen`, `--admin-listen` | Bind all addresses by default at ports 30100, 30101, 30102 and 30103 respectively. Admin uses HTTP. |
| `--ingestion-listen` | Optional separate ingestion listener. The gateway listener carries that scope by default. Both ingestion calls are unimplemented. |
| `--identity-as-told` | Comma-separated listener names, empty by default. A named listener accepts one `x-smg-principal` value as the caller's identity. Use it on loopback or behind a trusted proxy. |
| `--tls-cert`, `--tls-key` | Optional PEM server chain and key together for the gRPC listeners. |
| `--gateway-client-ca`, `--ingestion-client-ca`, `--control-client-ca`, `--replication-client-ca` | Optional PEM client CA per scope. Both scopes of a shared listener must use the same CA or both omit it. Requires server TLS and conflicts with `--identity-as-told` on that listener. |
| `--lease-file`, `--lease-name`, `--lease-namespace` | A process-local lease by default. File lease for replicas on one host with an existing parent directory. A named Lease needs the `kubernetes` feature. Its namespace defaults to the client's namespace. The two shared lease types are exclusive. |
| `--advertise-address` | A dialable host or host:port. A bare host uses the gateway port. Required for a shared lease when the gateway binds a wildcard address. |
| `--lease-duration-ms`, `--lease-renew-deadline-ms`, `--lease-renew-period-ms` | 3000, 2000, 1000. The duration must exceed the deadline, which must exceed the positive period. The named Lease uses whole seconds. |
| `--load-monitor-interval` | 10 seconds, positive. Load polling interval and the freshness window for a stream report to stand in for a poll. |
| `--shutdown-settle-secs`, `--shutdown-grace-period-secs` | 5 and 30 seconds. Grace is positive and includes settle. Settle is at most half of grace. Readiness drops before the listeners close. |
| `--decide-shards` | 1–256, available parallelism capped at 8 by default. Each model maps to one shard with room for 128 jobs. |
| `--runtime-worker-threads` | 1–4096 when set. Unset uses the async runtime's default. |
| `--log-level`, `--log-json` | `info` by default, also accepts `debug`, `warn`, `error`. `RUST_LOG` can set the filter. Text output by default. `NO_COLOR`, `FORCE_COLOR` and `CLICOLOR_FORCE` control text color, with force taking precedence. |

Startup rejects missing or duplicate scopes, duplicate listener names or addresses, unusable TLS material, and invalid lease addresses or timings. A client CA requires a verified certificate at transport level. This build has no certificate-to-principal mapping. Calls requiring a principal use a listener configured with `--identity-as-told`. A stated subject is a visible ASCII token of at most 253 bytes without commas. The service checks the scope before the principal. [Flags](src/controller/run.rs) · [Listener configuration](src/controller/config.rs)

### Remote calls from a gateway

A gateway you host can call the standalone service. Use the generated client or the crate's `client::Client` or `client::blocking::Client`. The reference client dials and asks `Leader`, opens a session, sends one decision at a time, and reports `dispatched`, `completed` and `released` when its caller asks. Use `connect_with` for a configured TLS endpoint. For concurrent requests, clone a `client::Link`: it shares one session and one decide stream. An ask takes one credit and waits for its own answer inside its budget. Reports go through a bounded queue, and a full queue ends the link. The client exposes `welcome`, `term_floor`, `gateway_id`, `hashes_cut`, `session_end` and `watch_workers`. `Watcher` can apply worker events and resume positions to a local store. The host handles dispatch, fallbacks, retries, lease ownership and reconnection. There is no bundled operator CLI. [Reference client](src/client.rs) · [Worker watcher](src/watcher.rs)

In the gateway shipped in this checkout, `--print-policy-spec` can print the document implied by its routing flags and exit. Normal routing uses the gateway's existing policies. Setting its Rust configuration field `policy_controller` loads the generated document, then installs a router that returns `not_decided_by_policy_controller` (HTTP 501) for routing calls. The gateway binary has no switch to send routing requests to a remote controller in this build.

| Scope | Call | What this build serves |
|---|---|---|
| Any gRPC listener | `Leader` | Reports the known leader, address and term. One lease covers all models. It takes no principal. |
| Gateway | `Session` | Start with `Hello` naming the caller and a protocol range containing 1. `Welcome` returns identity and limits. Its spec generation is 0 and it leaves needs and hashing unset. Send dispatch, completion, release or gateway veto frames. A new session replaces the earlier one for that gateway. Hello can restore vetoes. Lease and pin restoration via `HelloPage`, `SpecAck` and pushing new specs to gateway sessions are unserved. |
| Gateway | `Decide` | Bidirectional stream for a present gateway. Accepts single decisions and prefill entry requests that ask for the prefill/decode pair in one answer with a lease per side under the same ID. Under `epd` the served prefill entry still picks only the pair. A separate decode ask under either pair binding, or an encode ask under `epd`, gives `Busy` with an unspecified reason. The wire carries no encode item, so use the in-process door or an embedded engine for the full encode plan. |
| Gateway | `WatchWorkers` | Snapshot followed by membership and status changes, for all models or a named set. Resume from a term and sequence. A position outside the retained history can require a new snapshot. The service refuses `with_needs`. |
| Ingestion | `PushLoad`, `PushKvEvents` | Return `UNIMPLEMENTED` after admission. |
| Control | `RegisterWorker` | `ADD` (default), `REPLACE`, or `UPSERT` by URL. An upsert with changed role, wire or models makes a new incarnation. `wait_ready` requires probes disabled. |
| Control | `PatchWorker`, `DeregisterWorker` | Guard by worker ID and revision. Patch sets and removes labels and can replace overload thresholds. Deregister can mark draining, or remove. A drain needs a later remove. |
| Control | `GetSnapshot` | Leader's redacted `SLICE_WORKERS` and `SLICE_SPEC`. Name the slices explicitly. Empty requests ask for all ten and are refused. The spec slice includes retained generations with a store. |
| Control | `Explain` | Read a sampled decision by gateway and request ID on the leader. |
| Control | `ApplyPolicySpec`, `GetPolicySpec`, `RollbackPolicySpec` | Validate, read or change documents as described under [Updating a document](#updating-a-document). |
| Control | `StepDown`, `WatchRegime` | Return `UNIMPLEMENTED` after admission. |
| Replication | `Replicate` | Standby follows the leader's term, replays retained events or restores a snapshot. Automatic following uses the advertised leader host and the same fixed replication port on each replica, in cleartext with `--identity-as-told replication`. |

The controller exposes the gRPC health service on each gRPC listener. The admin listener serves `/liveness` and `/health` even while draining, `/readiness` (200 for leader and ready follower, 503 otherwise), `/version` and `/metrics`. [Controller protocol](contracts/policy_controller.proto)

### Following a controller's workers

With the `client` feature, a host can keep a copy of the controller's worker rows in a `MemoryStore` through a `Watcher`. Use `Watcher::open_on(&link)` to take the snapshot and the changes that follow it over the link's channel. The watch opens within the same five-second bound as `Client::watch_workers`. Once open, it has its own stream, so the link can answer asks and carry reports while the watch waits. Use one link for the gateway id. A second session with that id takes the first one's place.

A watch can outlive the link. A host that follows only while the link stands can wait for either to end.

```rust
let mut following = watcher.open_on(&link).await?;
let end = loop {
    tokio::select! {
        biased;
        () = link.closed() => break None,
        event = following.next() => match event {
            Ok(true) => {}
            result => break Some(result),
        },
    }
};
```

`None` means the link ended. `Some` gives the watch's own end. `Link::ended()` gives the cause once the link ends. A closed send queue can refuse an ask or a report before its stream reader sees the end. A full report queue ends the link at once.

Drop the stream before opening the watch again. The watcher asks for changes after the last event it took, or for a snapshot if it has no place. When a link ends, the host dials a new one.

## Extend it

### Register a Rust stage

Implement `Filter`, `Scorer` or `Picker`. Add a factory to `Catalog::builtin()` and pass it to `PolicySpec::validate_in`, `Controller::bind_with` or `controller::run`. Each stage name is unique across the catalog. A simple scorer factory accepts the document's `weight`. It runs once at registration, where its stage's name is checked, and at every validation, including a dry run. Keep its effects inside the stage it returns.

The example registers a scorer with `stages.scorer("pack", || Box::new(Pack))?`. It favors workers with more reported running requests. The document names it in `score: [pack]` and uses `not-overloaded` to steer away at the threshold.

```sh
cargo build --release --locked -p policy-controller-example
target/release/policy-controller-example --controller-id controller-0 \
  --spec-file examples/policy-controller/policy.yaml
```

A host that holds a store of its own, with no controller around it, validates a document over the same catalog. `AppliedSpec::validate_in(&document, &stages, &denied_headers)` returns the generation that `MemoryStore::apply` takes. The host numbers its documents with `generation`. The store takes a number above the one that stands, so it refuses a document with no number.

A scorer can take settings of its own from the document. Register it with `scorer_with`. The factory reads the entry's parameters into a type of its own and returns the scorer, or refuses a value.

```rust
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PreferWorker {
    preferred: u64,
}

stages.scorer_with("prefer-worker", |build| {
    Ok(Box::new(build.params::<PreferWorker>()?))
})?;
```

```text
score:
  - least-requests
  - prefer-worker: { preferred: 89, weight: 7 }
```

- `weight` is the profile's and stays out of the parameters. A bare name hands the factory an empty map.
- The factory runs for each entry that names it, at every start, apply and dry run. A value it refuses with `StageError::bad` is reported at its field's path in the document. A parameter that fails to deserialize is reported at the entry's path.
- Derive `deny_unknown_fields` on the parameters' type. Without it a misspelled key is ignored and its default runs.
- The keys of a parameter map carry no order. The factory reads them in key order, and two files that differ in that order alone have one digest. A key written twice is refused.
- A filter and a picker take no parameters yet.
- The scorer's `name()` must return the name it is registered under. `scorer`, `filter` and `picker` run the factory once at registration and refuse another name there. `scorer_with` runs no factory then, so validation refuses the name.

The stock schema lists the stock catalog. Validate an extended document against the host's catalog. Stateful scorers and pickers can observe issue, dispatch, completion, release, membership and signals when the host supplies them. A stage that keeps a number of each worker's load report says so through `Lifecycle::keeps_loads`. The controller then tells it every report before a decision can read that report, through `Lifecycle::on_signal`. A stage that does not say so is told no report. In process the load signal names two more things: the worker's slot in the host's counts and the report's sample instant. A host that builds the signal itself fills both. A scorer of your own can state a pick's overlap too: implement `Scorer::states_overlap` and `Scorer::said`. Both have a default that says nothing. `said` is asked once a decision, of the first target alone. The standalone validator refuses stages that declare a need for the full lifecycle. [Runnable Rust example](../../examples/policy-controller) · [Catalog API](src/catalog.rs)

### Load a WebAssembly component

Compile a component for the `smg:policy` package, at the version the WIT file names. The `policy` world takes records. `policy-packed` takes byte blocks with the layout given in the WIT. The host refuses imports from WASI. It provides log, clock, counter and seeded random calls. A component implements `init`, a decision resource with filter, score and pick, and `observe`. A score answer can carry `tie-band` on the scores' 0 to 1 scale: rows within that band of the best score tie, and the host uses its default band where the guest sends none. The host keeps instances per decider and meters their calls.

This host function loads the repository's policy component, using the default session bonus, and makes a scorer with an in-flight-count fallback. Use the returned scorer in a `Weighted` entry of a `Profile`.

```rust
use std::{path::Path, sync::Arc};
use smg_policy_controller::{
    builtins::LeastRequests,
    plugins::OpenDecisions,
    wasm::{Plugin, PluginSpec, Runtime, WasmScorer},
};

fn component_scorer(
    decisions: &Arc<OpenDecisions>,
    path: &Path,
) -> Result<WasmScorer, Box<dyn std::error::Error>> {
    let runtime = Arc::new(Runtime::new(decisions)?);
    let spec = PluginSpec::new("component-score", std::fs::read(path)?);
    let plugin = Plugin::load(&runtime, spec)?;
    Ok(WasmScorer::new(plugin.bind(), Some(Box::new(LeastRequests))))
}
```

Use one `OpenDecisions` for the host's plugin runtimes. It accounts for the decision budget across plugins and sides. `PluginSpec` takes bytes, an optional ABI pin, opaque config, memory cap (16 MiB per instance by default), frame budget, fuel calibration (1000 units per microsecond by default), log limit (10 lines/s at level 2 by default), exposure, needs, counters, event subscription, fallback policy and breaker settings. A frame budget of zero takes the decision's remaining budget. The loader refuses zero memory or fuel, a component that fails to compile or link, an unsupported world, ABI or import, and a failing `init`. `Plugin::check_memory_budget` can check a host's total memory limit per decider. A trap or invalid output discards the instance. [Component build and host example](../../examples/wasm/wasm-guest-policy) · [WIT](contracts/smg-policy.wit) · [Host configuration](src/wasm/config.rs)

### Call a gRPC plugin

Implement the `PolicyPlugin` service with the supplied proto. Its `Decide` stream starts with a hello and an acknowledgement with a protocol version (1 or 2) and supported stages. It handles decision frames by ID. An optional `Observe` stream sends ordered lifecycle events. A filter or scorer can start its call while native stages work. A picker bound on its own receives the combined scores at the pick. A scoring answer can carry `tie_band` on the scores' 0 to 1 scale: rows within that band of the best score tie. The host uses its default band when the field is unset. Use a Unix socket with a checked peer user ID, or TCP with mutual TLS and a pinned server name.

This host function connects to a plugin that's listening on a Unix socket under the same user and makes a picker. Use it as a profile's picker.

```rust
use std::{path::PathBuf, sync::Arc};
use smg_policy_controller::{
    builtins::Best,
    grpc::{GrpcPicker, GrpcPluginSpec, Plugin, Runtime, Transport},
    plugins::OpenDecisions,
};

fn socket_picker(
    decisions: &Arc<OpenDecisions>,
    path: PathBuf,
) -> Result<GrpcPicker, Box<dyn std::error::Error>> {
    let runtime = Runtime::new(decisions)?;
    let spec = GrpcPluginSpec::new("socket-pick", Transport::Unix { path, peer_uid: None });
    let plugin = Plugin::load(&runtime, spec)?;
    Ok(GrpcPicker::new(plugin.bind(), Some(Box::new(Best))))
}
```

`GrpcPluginSpec` takes the transport, opaque config, exposure, needs, counter names, event subscription, failure policy, breaker settings, the per-decision budget and the connect budget (100 ms by default). A frame sees the whole candidate view up to `max_candidates` (128 by default). Beyond it the stage falls back. Cold or unavailable streams fall back while opening in the background. Call `Plugin::warm` on a deciding thread to establish its stream in advance. A failed initial handshake or initialization can refuse loading the plugin. The host bounds its waits and frames. [Service contract](contracts/policy_plugin.proto) · [Host configuration](src/grpc/config.rs) · [Test service](tests/support/grpc_plugin.rs)

### Plugin inputs and fallback

Host code sets up both plugin kinds. A policy document can name a registered stage adapter through the host's catalog. It has no `plugins` field. A host building a profile in Rust can bind several adapters to one plugin binding. Catalog registration requires a distinct stage name for each entry, and each adapter takes its name from its plugin.

Exposure defaults to hashed tenant and session identities and no headers. A host can withhold identities, expose the clear form it holds, and expose selected allow-listed header positions. Request hashes are passed only when declared in the plugin's needs, up to 8190. Lifecycle events default to none. Scorers and pickers receive the subscribed events. The gRPC observe queue holds 1024 events. Gaps in the sequence identify lost events.

A decline takes the configured stage fallback and leaves a healthy instance or stream in use. A trap, invalid output, timeout or open breaker can also trigger fallback. Surviving positions must be unique and in range. Each scored row needs one finite value, which the host clamps to 0–1. The pick must be a permitted position. With no fallback stage, the default `Open` policy keeps all rows, gives a flat score or ranks the combined scores, respectively. `Closed` sheds at a filter or picker and still gives a flat score at a scorer. The breaker defaults to five failures in ten seconds and a 30-second cooldown. Its latency rule can also open it when moving p50 exceeds half of the decision budget or p99 exceeds the budget. Hosts can read `Plugin::stats` and declared `Plugin::counters` in code. [Shared plugin settings](src/plugins/config.rs)

## Policy document reference

Use `apiVersion: smg.policy/v1` and at least one profile. Both YAML and JSON use the same types. Quote strings that look like numbers or booleans in YAML. A key without a value is null. Use `[]` or `{}` for an empty list or map. The [JSON schema](policyspec.schema.json) includes reserved fields and stock stage names. Use validation in the intended host to check which fields and stages can run.

| Document field | Meaning and limits |
|---|---|
| `apiVersion` | Required. Only `smg.policy/v1` is accepted. |
| `generation` | Unsigned 64-bit number, default 0. Zero asks the service to number it. See [Updating a document](#updating-a-document). |
| `profiles.<name>` | One to 65535 profiles. `filter` and `score` default to empty. `pick` is required. Stage mappings have one name. See [Built-in stages](#built-in-stages). |
| `view.headers` | Up to 256 unique lowercase HTTP header names, in wire position order. Default empty. Denied names are `authorization`, `proxy-authorization`, `cookie`, `set-cookie`, `x-api-key`, `api-key` and `x-goog-api-key`. Embedded validation can deny more, such as the gateway's tenant header. |
| `view.block_size` | Positive fallback block size when the request gives none. The prefix stages fall back to 16 if omitted. |
| `view.boundaries` | Positive, strictly ascending token positions, default empty. Used by prefix keying and required for a `hash` index entry. With a prefix-keyed scorer every boundary and `prefix_tokens` must cover at least one whole configured block and fit the hash cap. |
| `view.max_block_hashes` | 1–8190, default 256. The generation can read fewer, down to zero when no stage needs hashes. |
| `view.max_report_age_ms` | 1–3599999 ms, default unset. `least-load`, `resident-work`, `least-cost` and the load gates of `prefix-affinity` and `warmup-slice` use their unreported paths for older reports. Overload verdicts can still reflect older reports, plugins interpret report age themselves, and host-registered stages keep their own rules. A report at the limit still counts. |
| `view.labels`, `view.zones`, `view.engines` | Unique names, default empty. At most 64, 256 and 64 respectively. Labels are `key` or `key=value` with nonempty sides. |
| `index.<model glob>` | Defaults to an empty map. An entry has `kind` (`tree` by default, or `hash`), `ttl_s` (180 seconds for a hash placement), `max_size` (67108864 characters for text or tokens otherwise for a tree), and `sweep_s` (120 seconds). Numbers are positive. These are settings a host can read for its index. The crate's own store reads none of them. Exact keys win, then the longest literal prefix or suffix, then `*`. Equal-length prefix and suffix globs are refused. |
| `plans.pd-pair.pairing` | `off`, `lenient` (default), or `strict`. Used for prefill/decode compatibility, including the pair in `epd`. |
| `bindings` | Ordered, first match wins. Empty works with one profile. |
| `explain.sample` | 0–4294967295. Zero is the default and disables sampling. |

### Bindings and plans

`match.model` defaults to `*`. It takes an exact ID, `prefix*`, `*suffix`, or `*`. `match.phase` can list `single`, `prefill`, `decode`, and `encode`. An empty list matches all phases. A multi-sided plan with a nonempty phase list must include each of its sides.

```yaml
apiVersion: smg.policy/v1
profiles:
  chat:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
  batch:
    filter: [healthy]
    pick: round-robin
bindings:
  - match: { model: "batch-*" }
    profile: batch
  - match: { model: "*" }
    profile: chat
```

| `bindings[].plan` | Profiles the binding names | What the engine picks |
|---|---|---|
| `single` (default) | `profile` or `profiles.single`. | One worker in the requested role, with optional alternates. |
| `pd-pair` | `profiles.prefill` and `profiles.decode`. | A prefill worker and a compatible decode worker, in one decision. The decode hard filters first constrain which prefills have a partner. The decode profile then sees compatible workers for the selected prefill. If it yields no target the engine can try another prefill from the pick. |
| `epd` | `profiles.encode`, `profiles.prefill` and `profiles.decode`. | A pair with a prefill that can accept the encoders' output, then an encoder for each item over that prefill's compatible encoders. A request with no encode items needs no encoder. |

`bindings[].overload` defaults to `enabled: true`. Disable it to turn off binding-level thresholds. A worker registration can still set thresholds of its own. `waiting_requests` defaults to 8 and must be positive. `token_usage` defaults to 0.8 and must be above 0 and at most 1. The store judges the sum of queued requests and mean active KV usage over an engine's ranks. A report without that split carries its total usage there. Reaching either limit sets the veto. `shed_when_all_vetoed` defaults to false and controls the single-plan fallback and plain ring behavior. For a decode pool used by a pair, the binding selected at prefill supplies its overload policy.

`bindings[].sessions` configures sticky placement under `single`. `bindings[].routing_key` names the sources of a request's key, for a ring or a sessions rule. See [Session keys](#session-keys). A multi-sided plan refuses a sessions rule or a picker that places by session assignment.

A `single` binding can name a second profile with `shadow: next`. Its stages run through the library's `Engine::shadow` for comparison. Request handling runs the live profile. Use native stages and a profile that no binding uses live.

The validator checks reserved fields for shape and refuses their active use. This includes `match.tenant`, `match.priority`, `match.header`, `match.prompt_tokens`, a nonempty `match.labels`, `canary`, `override: allowed`, `admission.enabled: true`, and `sessions.issue_session_id: true`. The bounds use `min` and `max`. A header matcher uses `name` and `values`. A canary would name a `profile` and exactly one of `percent` (1–100), `tenants` or `header`. Use `override: locked` (the default) and leave admission and session ID issuance off.

### Prefill and decode

```yaml
apiVersion: smg.policy/v1
profiles:
  prefill:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
  decode:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
plans:
  pd-pair:
    pairing: lenient
bindings:
  - match: { model: "*" }
    plan: pd-pair
    profiles: { prefill: prefill, decode: decode }
```

An embedded host using the store can supply the runtime and a KV handoff descriptor. `lenient` rejects known differences. `strict` also rejects missing runtime or transport, a value missing on one side for a hard fact, and different versions when both are provided. Both sides may omit a KV layout fact. A shared explicit protocol settles the handoff after the runtime check. `off` skips descriptor checks. In every mode, HTTP-only workers pair within their wire class and workers on the other side of that division cannot pair if their known runtimes differ. The store derives compatibility from membership and rebuilds it when the pairing mode changes.

Over the standalone registration RPC, the pairing descriptor accepts only a bootstrap port, and the runtime comes from `spec.runtime`. Use `lenient` or `off` for workers registered this way. A host supplying its own admission facts can also add `has-capacity` and `has-room` to the profiles.

### Encode with a pair

```yaml
apiVersion: smg.policy/v1
profiles:
  encode:
    filter: [healthy, not-overloaded, not-tried]
    score: [consistent-hash]
    pick: best
  prefill:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
  decode:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
bindings:
  - match: { model: "*" }
    plan: epd
    profiles: { encode: encode, prefill: prefill, decode: decode }
```

Use this document with `Door::decide_encode_pair`, which places the request's encode items beside the pair. A host that decides over its own rows calls `Engine::decide_encode_pair` and supplies `EncodeItems`, the worker offers and the compatibility tables. The encode profile can key each item by content hash. A push handoff can place by one supplied key. Images require a vision-capable encoder, and a request can require a vision-capable decode worker. The engine writes encoder choices to `EncodePlaced`. All choices are made before dispatch. [Encode entry and refusal names](src/framework/encode.rs)

### Session keys

The host resolves a key and supplies it as `RequestFacts::session_key`, or as `facts.session_key` on the wire. The header fields specify the host's inputs. Sending the header in a `DecideRequest` alone does not create a key.

| `sessions` field | Default and behavior |
|---|---|
| `sticky_header` | Required without `routing_key`. It must be listed in `view.headers`. |
| `headers` | Empty uses the sticky header alone. A nonempty lookup list must contain it. Names must be unique and allow-listed. |
| `assignment` | `delegate` (default) uses the profile. `random` draws, `min_load` uses host requests in flight, and `min_group` uses active session counts. With `manual`, `delegate` uses `min_load` and a keyless request follows the same assignment without a pin. |
| `ttl_s` | 14400 seconds. Positive. Use 4294967295 for no expiry. |
| `sweep_s` | 120 seconds. Positive. With finite TTL, the sweep must be finite. |
| `inflight_cap` | 2. Positive. 65535 disables it. At the cap the session rule tries a fresh placement. A pin moves only to another worker below the cap. The cap reads the key's counts. The door and the standalone controller hold them from their leases, and a host that decides over its own rows feeds them. |
| `recovery` | `original` (default) prefers the first pin when both pins are usable again. `latest` prefers the most recently dispatched pin. |
| `issue_session_id` | False. True is refused. |

The stock controller counts a key's requests by the leases it holds. A request counts on the worker its lease books, from its decision to its end, over every gateway the controller serves. Decisions made at the same moment for one key each read the count before the others' leases are on file, so a burst of them can pass the cap. A request decided while its rule had no cap is not counted, so after a change that sets a cap a key can pass it until those requests end.

**Where the key comes from.** A binding names the sources of its requests' key under `routing_key`. The host reads them in order, and the first that holds a value is the key. A ring places by that key, and a sessions rule pins it.

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
        - rid: true
        - header: x-forwarded-for
        - header: cookie
          max_bytes: 8192
      strip_lineage: true
```

| Field | Default | Meaning |
|---|---|---|
| `sources` | required | One to eight entries, read in order. An entry is `header: <name>`, or `rid: true` for the request's own id. |
| `max_bytes`, on an entry | 128 | The most bytes of a value the source hands, 8192 at most. A longer value is passed over. So is an empty one, and one that is not UTF-8. |
| `strip_lineage` | `false` | Take the turn and retry suffixes off a key: `conv_t2_r1` keys as `conv`. |

A source need not be in `view.headers`: its value enters no view. A credential header is refused. A key from a header that `view.headers` does not list, `cookie` for one, leaves the host as its fingerprint alone, so no stage, plugin or record sees its text.

A binding names its sources once. With `routing_key`, a sessions rule leaves out `sticky_header` and `headers`. Without it, the rule's own headers are the sources. A binding with neither gives its requests no key.

A host reads a binding's sources with `AppliedSpec::key_sources_for` and derives a request's key with `KeySources::derive`.

A plain ring under `single` can keep a key on an otherwise usable worker across the overload veto. This requires the sole scorer to be `consistent-hash` without prefix or load bound, bare `best`, and a steering binding. Any session rule must delegate and either have its cap disabled or have the key's counts held. The door and the standalone controller hold them from their leases. A host that decides over its own rows sets `session_counted`. With shedding enabled, a vetoed owner passes the request to a usable pin or ring successor if one exists. With a bounded prefix ring, an owner strictly above both load bounds is passed over. The mean includes the incoming request. Where boundaries supply a shallower key it is tried before the profile's keyless scoring path.

## Guarantees and limits

### Decisions and refusals

A pick's first target is booked at issue by the door or service. The caller can dispatch an alternate and report which target it took. Report completion or release so the lease can end. An undispatched lease expires after five seconds. A repeat marked `again` ends an earlier lease still waiting for dispatch. The service checks the leadership term and budget as it makes the decision and checks the term and lease at send. Decisions read one published world and keep the issuing engine for lease events. A new generation builds fresh stages. Outstanding leases still belong to their issuing engine while the store holds them.

- A single profile's empty pass can rerun filters with the overload veto lifted, then with liveness doubts lifted. It favors fewer in-flight requests on fallback. `shed_when_all_vetoed: true` keeps overload in force and can still lift liveness alone. Draining, not-ready and breaker holds stay refused by the health check. Split plans shed at their own vetoed sides without this single-plan fallback.
- Unmatched bindings, no endpoints, an empty filter pass, a declined picker and incompatible pair or encode workers are sheds. The answer says the reason and the stopping stage. The built-ins emit no queue tickets. Hosts decide how to map sheds to request status and retry behavior.
- `Decide` has a 64 KiB incoming frame limit, including unknown fields. It requires a request ID of 1–256 bytes, known phase and hash kind, and at most 256 headers with positions in 0–255. A nonzero session hash must decode to a nonzero fingerprint for its model and phase. Zero means no key. A malformed or oversized frame can end the stream. The client cuts tail hashes to fit or returns an error when cutting them cannot make the message fit.
- `Welcome` grants 128 credits per decide stream and up to 256 streams per gateway. Requests spend credit. A credit frame replenishes it. `Busy` reasons are `queue_depth`, `no_credit`, `budget_gone_on_arrival`, `not_leader` and `unspecified`. They cover a full shard queue, exhausted stream credit, expired budgets or dispatch span, a fenced replica, and cases such as raw prompt text, an unsupported side, a decider fault or exit, spent lease IDs or a document still awaited. The caller owns any local fallback.
- The session expects `Hello` within five seconds, as its first frame, with a matching principal and protocol 1 in range. A gateway can keep one current session and get ten seconds of grace after disconnect. The table holds up to 4096 gateways, up to 4096 vetoes per gateway and eight kinds per worker per gateway. A `Hello` veto count must match the list. An unspecified veto kind, excess vetoes or an invalid lifecycle frame can end the stream. Old terms and replaced sessions can end calls. The reference client applies its term floor to decision and session answers. It waits up to five seconds per operation step, with a separately streamed watch.
- Workers registered over gRPC take one rank and one HTTP or gRPC wire. A new row must supply its role and wire. The service refuses a host-local wire, rank width above one, nonempty topology, or extra pairing descriptor fields. It accepts at most 128 model IDs and 128 labels per row, a 2048-byte URL, 256-byte names for model, runtime and zone, 128-byte label keys, 4096-byte label values and a 16 KiB encoded row. The URL and model IDs cannot contain blanks or control characters. A scrape needs an interval and uses the registered runtime's parser. Probes default to three failures, two successes, a 60-second interval and a five-second timeout. Duplicate adds, missing IDs or URLs, stale revisions and another maker's row have separate statuses.
- `WatchWorkers` allows four watches per principal. It can require a fresh snapshot when history no longer matches, and ends a stalled watch when polling resumes after 30 seconds. Worker watch pages hold up to 128 rows and 2 MiB of row data. Watches and `GetSnapshot` accept up to 128 model IDs. `GetSnapshot` can cap the worker rows returned and takes one snapshot at a time with up to 32 waiting. It refuses unknown or unavailable slices, invalid filter fields, and clear identifiers. `Explain` refuses a decision ID or an incomplete key and returns `NOT_FOUND` on a miss.
- A shared lease keeps rings of 4096 watched events and 65536 load events for replication. History gaps require a snapshot. The leader holds up to four snapshot restores at once and can release a stalled restore after 60 seconds.
- A call on the wrong scope returns `PERMISSION_DENIED`. A scoped call without a principal returns `UNAUTHENTICATED`. Leader-only operations can return `FAILED_PRECONDITION` with leader details, or `UNAVAILABLE` during a term transition. Unserved calls return `UNIMPLEMENTED` after admission.

### When a gateway loses its stream

A gateway asks the stock controller for decisions on a stream. Each decision is a **lease** on its worker: the worker counts as booked for that request. A lease that no dispatch names within five seconds is given up.

A stream can end with requests unanswered. Two rules keep one lease standing for one request.

- **The stream's end.** A decide stream that ends gives up the lease of every decision it had not sent yet. No gateway could use them.
- **A request asked again.** A gateway whose stream ended sends each unanswered request again and marks it `again` (`DecideRequest.again` in [`contracts`](contracts)). The controller ends the lease of the earlier ask at once, where it still waits for a dispatch, and decides the request anew.

What a gateway must hold to:

- The mark is for a request whose stream has ended, and the gateway reads nothing more from that stream. On a stream that is still open the earlier answer can still arrive, and it would carry a lease the mark has ended.
- A request sent twice without the mark is two requests and two leases.
- A repeat that the stream answers `Busy` before a decider takes it ends nothing. The next ask of that request carries the mark again.
- The earlier lease is found by a 64-bit fingerprint of the request's id, among that gateway's own leases.

Two counters tell what happened, on the admin listener's `/metrics`:

| counter | what it counts |
|---|---|
| `smg_policy_controller_decide_asked_again_total` | requests marked `again`, by `earlier_lease`: `ended` where the earlier lease still waited, `none` where no lease of the request waited |
| `smg_policy_controller_decide_answers_dropped_total` | answers no stream took, by `cause`: `stream_gone` where the stream had ended, before the answer was made or with it still in the stream's queue, and `queue_full` |

After a stream's end most repeats read `none`: the stream gave the lease up, or the first ask was never decided. That is no error.

### Updating a document

**File.** Use a path that leads to a regular file of at most 128 KiB. The reader follows symlinks. A `.json` suffix selects JSON. The controller checks the file at startup. The watch polls after one second and reads a changed identity after two matching looks. A rejected update leaves the running generation in place and is logged and counted once per settled file state while leading. Unchanged documents count as neither apply nor rejection. A fast same-size edit can escape detection if the file identity does not move. An update that stays unstable can be rejected.

**Store.** `--spec-store` uses an existing directory with a lock and support for hard links. It runs with a process-local lease. Startup loads the newest file or starts with no document in an empty store. An empty store can report ready and shed requests until the first apply. Enforce persists before publishing and retains the newest eight files normally. A competing start waits up to the shutdown grace for the lock.

**Generation.** A zero or omitted number starts at 1 and increments on change. A file can claim a higher number. Enforce through the store takes the next number. Changed content under the standing number, a stale number, or an out-of-order store number is refused. An identical document with the standing or automatic number is a no-op. The digest is a canonical `sha256:` digest of parsed content without the generation line.

**Calls.** `ApplyPolicySpec` requires a mode. Use `yaml` or `json` for the document format. `DRY_RUN` on either source checks the candidate under that source's rules. `ENFORCE` writes only under a store. Both can take the current `expected_digest` as a guard. A verdict can include before and after needs, or one issue with a field path when one applies. `AUDIT` is unserved. `GetPolicySpec` returns this replica's standing document, or a generation by number if held. It returns `NOT_FOUND` when none is held. `RollbackPolicySpec` under a store reapplies a named retained generation using a new number. Enforce and rollback under a file return `read_only` before document validation. Each dry-run and write queue has room for up to 32 waiting. Both use the same 128 KiB document cap as the file, with a separate bound on decoded YAML expansion.

**In process.** A host that embeds the store takes the same door with the crate's `operator` feature. `operator::SpecDoor::new(store, catalog, denied_headers)` makes it. `apply` takes the host's document as bytes, YAML or JSON. It checks the document over the host's catalog and denied headers, and runs it. A changed document runs as the next generation. `dry_run` checks a document and applies nothing. `status`, `document` and `standing` read the generation that runs, and `counts` the applies. `operator::snapshot::told` gives the worker rows an operator reads. An address and a provider's key are told without userinfo. No spec store is bound in process, so `enforce` and `roll_back` refuse. The feature builds on Linux and macOS. The gateway serves no route for these calls yet.

**Shared lease.** A standby follows the leader's document if replication is configured and working. It validates its own file at startup and watches without applying it while following. On promotion it may apply its file after it settles. A file already observed running since the last change is skipped. An unnumbered document the standby never saw run can take effect on promotion even if its file is older. An explicit generation keeps a stale number from taking over. A new leader awaiting its first document can return `Busy`. File lease renewals are durable writes. With the default timings, a slow renewal can stop decisions, close sessions or lose the lease. For slower storage, the lease flags can give more time, for example `--lease-duration-ms 15000 --lease-renew-deadline-ms 10000 --lease-renew-period-ms 2000`.

The validator refuses unknown document fields and a non-null `kind`, wrong API version, malformed or duplicate map keys, invalid names or ranges, missing or conflicting bindings, missing plan sides, unsafe headers, unknown stages, unsupported active fields and stage factory errors. The one exception is the parameters of a scorer you register: its own type reads them, and must deny unknown fields to refuse one. A refusal can be about the whole document. The validator messages in these two examples also appear in startup or apply errors.

<!-- refused -->
```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [ready]
    pick: random
```

```text
profile "default": unknown filter "ready"
```

<!-- refused -->
```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    score: [least-load]
    pick:
      best-with-temperature: { temperature: 0.5 }
```

```text
profiles.default.pick: the best-with-temperature picker is not in this build
```

### Explain and metrics

Set `explain: { sample: N }` to keep every Nth pick or shed per decider. Zero turns it off and one keeps every pick or shed. Each decider allocates its 1024-slot ring on its first sampled write and overwrites its oldest records. At R eligible decisions a second on that decider, a full ring covers about `1024 × N / R` seconds. A new generation restarts counting. Older samples can still be read under the same term. The remote lookup uses `gateway_id` and `request_id`. Use the subject the gateway authenticates as on the gateway listener for `gateway_id`, because `Explain` hashes it the same way as the decide stream. Use `Door::explain` in process with the store gateway ID. `Door::explained` returns that record with its names: the plan, the profile, and that profile's picker, filters and scorers in their declared order. The names are `None` once another generation stands, or when the request matched no binding. A shed made ahead of every profile names the plan alone, unless the binding's profile is the document's first. A term change makes old samples unavailable on the service.

A record echoes the lookup key and gives the decision and its trace without a lease. It keeps up to three top scores. The control call keeps the record's original binding position and trace. `binding` is 0 when no binding matched or when the first binding matched. The name fields are empty for unmatched decisions and records from older generations. It leaves the decision ID, full candidate list, view digest, plugin projections and lifecycle outcome unset. It omits request facts, headers and session keys.

The admin `/metrics` endpoint exports the following series. A spec source adds the two spec series.

| Metric | What it measures |
|---|---|
| `smg_policy_controller_state` | One gauge per `state` (`starting`, `leader`, `syncing`, `follower`, `draining`). The current state is 1. |
| `smg_policy_controller_start_time_seconds` | Startup time since the Unix epoch. |
| `smg_policy_controller_replication_synced` | 1 for a synced standby. |
| `smg_policy_controller_replication_lag_seconds` | Seconds since a frame last left the standby at the leader's head, or 0 when leading or before such a frame. |
| `smg_policy_controller_replication_restores_asked_total` | Standby's restore requests. |
| `smg_policy_controller_load_polls_total` | Poll arbitration per engine and interval, by `mode` (`poll`, `fallback`, `suppressed`). |
| `smg_policy_controller_load_poll_failures_total` | Poll failures by `reason`. |
| `smg_policy_controller_load_scrapes_total` | Successful HTTP load scrapes. |
| `smg_policy_controller_load_scrape_failures_total` | Scrape failures by `reason`. The poll and scrape label values are `transport`, `answer`, `deadline`, `no_load`, `unsupported`, and `client`. |
| `smg_policy_controller_load_rows_total` | Rank load rows written by the load driver. |
| `smg_policy_spec_generation` | Standing spec generation. |
| `smg_policy_spec_apply_total` | Applies by `result` (`applied` or `rejected`). |
| `smg_policy_controller_decide_busy_total` | Busy answers by wire `reason`. |
| `smg_policy_controller_decide_no_decision_total` | Unspecified Busy by `cause` (`fault`, `pair_side`, `prompt_text`, `decider_gone`, `lease_ids_spent`, or `document_awaited`). |
| `smg_policy_controller_decide_hashes_cut_total` | Requests whose hash list exceeded what the generation reads. |
| `smg_policy_controller_decide_answers_dropped_total` | Dropped answers by `cause` (`stream_gone` or `queue_full`). |
| `smg_policy_controller_decide_asked_again_total` | Requests marked `again`, by `earlier_lease` (`none` or `ended`). |
| `smg_policy_controller_decide_candidates_undialed_total` | Candidates removed by the asking gateway's wire constraints. |
| `smg_policy_controller_decide_budget_outlasted_total` | Decisions cut at the final decision check because the budget or dispatch span was too short. |
| `smg_policy_controller_members_outside_list` | Members not classified by a configured list, by `list` (`engines` or `zones`). |
| `smg_policy_controller_overload_flips_total` | Load writes that changed an overload verdict. |
| `smg_policy_controller_lifecycle_faults_total` | Panics contained while applying dispatch, completion, release or dispatch expiry. |

Plugin stats are available through the host APIs. Both kinds count answers, declines, exhausted decision budgets, breaker bypass and openings, stalls and observed events. WebAssembly also reports traps, fuel use and exhaustion, invalid replies, overlong calls, instantiations, lost events and unread timing. gRPC reports plugin Busy, invalid replies, overlarge pools, cold or refused frames, misses and their attribution, connections, stream failures, late answers and dropped observe events. Declared counters are separate from these host stats. [WebAssembly stats](src/wasm/plugin.rs) · [gRPC stats](src/grpc/plugin.rs)

## Gateway routing metrics

The embedded host counts each ask and times its call to the controller. Each ask counts as
`pick`, `shed`, `queue`, or `refused`. The `mode` is `embedded`. The `model` is the
resolved model. The `profile` is empty on a refusal, and for a request that is shed ahead of
every profile, as when no binding matches its model. It is also empty when the host needs a
name after that generation has been replaced. When a commit finds no sendable target, each
subsequent ask counts on its own.

The embedded column gives the gateway family to query. `Same` carries the classic names over.
The last column also records families whose embedded writers are pending. Four names in it
are planned, and no code writes them yet: `smg_policy_lease_expired_total{kind}`,
`smg_policy_leases{kind,state}`, `smg_policy_fallback_total{reason}` and
`smg_policy_deadline_misses_total`. The shipped dashboard uses the classic names. Decision
series appear after their first observation. The standalone controller keeps the
`smg_policy_controller_` prefix for its process metrics, including role, replication,
election, and load polling.

| Classic gateway | Embedded gateway | Class and plan |
| --- | --- | --- |
| `smg_worker_selection_total{policy}` | `smg_worker_selection_total{profile}` | Kept. The embedded profile label comes with the router's send path. |
| `smg_worker_overload_shed_total`, `smg_worker_overload_fallback_total`, `smg_worker_liveness_fallback_total` | Same | Kept. The embedded writers will use the decision's outcome and lifted veto. |
| `smg_pd_admission_waits_total`, `smg_pd_admission_sheds_total`, `smg_pd_admission_window`, `smg_pd_prefill_admission_inflight`, `smg_pd_prefill_admission_queued`, `smg_pd_prefill_admission_rejections_total`, `smg_pd_prefill_admission_wait_seconds` | Same | Kept. The gateway owns admission. |
| `smg_routing_key_source_total{source}` | Same | Kept. The gateway reads `header` or `rid` as the source. |
| `smg_cache_aware_policy_branch_total`, `smg_manual_policy_branch_total{branch}`, `smg_consistent_hashing_policy_branch_total`, `smg_prefix_hash_policy_branch_total` | `smg_policy_decisions_total{model,profile,outcome,mode}` | Renamed. The new family groups by outcome. The standalone controller's `smg_policy_controller_decide_no_decision_total` corresponds to `outcome="refused"`. |
| `smg_policy_inflight_reconciled_total{policy}` | — | Planned rename to `smg_policy_lease_expired_total{kind}`. No code writes the new name yet. |
| `smg_cache_aware_match_ratio`, `smg_kv_index_lookup_seconds`, `smg_cache_placement_entries` | Same | Moved to the process that holds the index. Match ratio is observed only when the scorer reports the chosen worker's overlap. Embedded observations from the index are pending. |
| `smg_cache_tree_chars`, `smg_cache_tree_tokens`, `smg_cache_tree_tenants`, `smg_manual_policy_cache_entries` | — | Ends with the classic trees and manual cache. |
| `smg_cache_aware_prefix_margin_held_total` | — | Open. An embedded successor requires the scorer to report whether the margin held without an extra read. Otherwise the counter ends with classic routing. |
| — | `smg_policy_decision_seconds{mode,outcome}` | Added. One observation per ask. |
| — | — | Planned additions. Embedded writers are pending for `smg_policy_leases{kind,state}`, `smg_policy_spec_generation`, and `smg_policy_spec_apply_total{result}`. The standalone controller already publishes the two spec families when it has a spec source. No code writes `smg_policy_leases` yet. |
| — | — | Planned renames of the standalone controller's `smg_policy_controller_decide_busy_total` to `smg_policy_fallback_total{reason}` for fallback and of its `smg_policy_controller_decide_budget_outlasted_total` to `smg_policy_deadline_misses_total` for decision deadlines. No code writes the new names yet. |
| `smg_router_requests_total`, `smg_router_request_errors_total`, `smg_router_request_duration_seconds`, `smg_router_tokens_total`, `smg_router_ttft_seconds`, `smg_router_tpot_seconds`, `smg_http_responses_total`, `smg_http_request_duration_seconds`, `smg_http_inflight_request_age_count`, `smg_http_connections_active` | Same | Kept. Request and connection metrics. |
| `smg_admission_inflight`, `smg_admission_queue_depth`, `smg_admission_queue_rejected_total`, `smg_scheduler_admit_total`, `smg_scheduler_inflight`, `smg_scheduler_queue_depth` | Same | Kept. Admission and scheduling. |
| `smg_engine_running_requests`, `smg_engine_waiting_requests`, `smg_engine_token_usage`, `smg_engine_cache_hit_rate`, `smg_worker_health`, `smg_worker_cb_state`, `smg_worker_health_checks_total`, `smg_worker_retries_total`, `smg_worker_retries_exhausted_total` | Same | Kept. Worker reports and resilience. |
| `smg_kv_event_lag_seconds`, `smg_kv_event_gaps_total`, `smg_kv_event_resyncs_total`, `smg_kv_event_missed_batches_total`, `smg_kv_event_stream_errors_total`, `smg_kv_event_degraded_ranks`, `smg_kv_index_entries` | Same | Kept. Gateway cache event and index metrics. |
| `smg_symphony_outputs_total`, `smg_symphony_malformed_total`, `smg_symphony_fallback_total`, `smg_symphony_refused_total`, `smg_symphony_calls_suppressed_total`, `smg_tokenizer_cache_lookups_total` | Same | Kept. Output parsing and tokenization. |
| `smg_tokio_worker_busy_ratio`, `smg_tokio_event_loop_stalls_total`, `smg_tokio_event_loop_delay_seconds`, `smg_tokio_alive_tasks`, `smg_tokio_global_queue_depth`, `smg_log_dropped_lines` | Same | Kept. Runtime and logging. |
| `smg_allocator_resident_bytes`, `smg_allocator_allocated_bytes` | Same | Kept. Available when the gateway runs with jemalloc statistics. |
| `smg_pd_prefill_duration_seconds`, `smg_pd_kv_transfer_duration_seconds`, `smg_pd_decode_ttft_seconds`, `smg_pd_ttft_seconds`, `smg_pd_rendezvous_failures_total`, `smg_pd_bootstrap_failures_total`, `smg_pd_kv_transfer_failures_total`, `smg_pd_kv_connector_mode_total` | Same | Kept. Prefill and decode dispatch. |

The manual policy branches `occupied_hit`, `occupied_miss`, `vacant`, and
`no_routing_id` correspond to `pick`. `no_healthy_workers` corresponds to `shed`.
`cap_respill` corresponds to `pick` when the policy returns a worker and `shed` when it
returns none. The outcome count groups these branches together.

An embedded selection uses the existing opt-in routing capture. Its `x-smg-cache-trace`
selection row names the profile in `policy`, uses origin `policy_controller`, and carries
a `decision` with up to three scored worker addresses. The gateway uses redacted addresses
for the selected worker and the top candidates. The same row appears in the dispatch log
when tracing is configured to log.

## Presets

`PolicySpec::preset("least_load")` builds a complete document with generation 1. Each preset has one profile, the same three filters and a binding for every model. The copies below leave the number to the source. `PolicySpec::power_of_two` also accepts `ExpectedWait`, `Requests`, `LeastRequests` or `ResidentTokens { request_tokens }`. The resident variant ranks the full set by resident work. `pair_presets` and `epd_presets` compose side profiles from preset names and add `has-capacity` to the prefill side and `has-room` to the decode side. Over `MemoryStore` or the standalone controller, which leave those bits unset, remove `has-capacity` or every prefill sheds. [Preset constructors](src/spec.rs)

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

Rotates through a survivor set. Changes to that set change the rotation, and retained rotations are bounded.

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

Favors the routing key's ring owner while it is usable. With no key the scores tie.

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
```

### `prefix_hash`

Uses the routing key first, then the last whole block at the deepest reached boundary or the configured 256-token depth when no boundary is reached. When no key can be used, the in-flight-count scorer places the request.

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

Blends warm-up, prefix reuse and least-load. The default KV gate is off. With the standalone controller's regular inputs it works from worker age and load. A controller built with the `worker-host` feature reads cache events from each gRPC worker whose row asks for them.

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

The decision benchmarks time view construction and policy execution separately, typically at 8, 64 and 512 candidates. They cover presets, dispatch credit, overloaded pools, sticky hits and misses, spec validation, cache locality, sampling and prefix keying. The store benchmarks time membership and load writes, publication, view building, update journals, KV ingestion and pairing. Library benches also measure encode placement, pair and encode class construction, concurrent callers, the sent-prefix predictor, election, and both plugin kinds. The `decide_stream` bench measures loopback round-trip percentiles and throughput under offered load, separately from in-process work. [Benchmark sources](benches)

```sh
cargo bench -p smg-policy-controller --locked --bench decision
```

| Gate | Work measured | Threshold by row |
|---|---|---|
| [Engine](../../.github/bench-gates/policy-engine.toml) | View build, decisions and spec parse/validation. | 10–15% |
| [Affinity](../../.github/bench-gates/policy-engine-affinity.toml) | Affinity decisions and selected sizes of view builds. | 10% |
| [Bucket](../../.github/bench-gates/policy-engine-bucket.toml) | Bucket decisions, including prefill across models and with stale rows. | 10–20% |
| [Power of two](../../.github/bench-gates/policy-engine-power-of-two.toml) | The preset's load and count variants. | 10% |
| [Prefix hash](../../.github/bench-gates/policy-engine-prefix-hash.toml) | Ring owner and load spill paths, with and without boundaries. | 10–15% |
| [Store](../../.github/bench-gates/policy-store.toml) | Store views, load and membership writes, publication and selected pair and many-model cases. | 10–40% |
| [Store journal](../../.github/bench-gates/policy-store-journal.toml) | Journaled writes, view build and journal reads. | 10–30% |
| [Gateway policy selection](../../.github/bench-gates/policy-selection.toml) | The gateway's regular placement benchmarks. | 10% |

The gates run three interleaved measurements per side when timing is needed. Each benchmark ID uses the first matching threshold. A time regression must exceed that threshold in the median of the runs, with every changed run slower than every base run. Overlap is reported as suspect. Some gates also repeat suspect measurements or check for executable placement effects before failing a row. Missing gated IDs are detected. These are relative regressions for the measured paths. The plugin and wire round-trip benchmarks have no dedicated policy gate here. [Gate rules and report](../../scripts/bench_gate.py)

## What is not built yet

- A client that marks a request as asked again. The controller serves the mark. The crate's own client does not set it yet.
- A plan that waits for the prefill worker's answer before it picks the decode worker. `pd-pair` and `epd` pick every worker in one decision.
- Encode placement through a served controller. The in-process door places the encode items with `Door::decide_encode_pair`. The served controller decides an `epd` request as its pair alone because the wire carries no item.
- A choice by the media a worker can process, outside `epd`. No built-in filter reads it.
- A binding that matches on a tenant, a priority, a header, a prompt size or labels. The fields are in the schema. A document that writes one is refused as not in this build.
- Shadow execution during request handling, shadow maps by side, and plugin stages in a shadow profile.
- `canary`, `override: allowed` and `admission` on a binding, and `issue_session_id` in a sessions rule.
- A sessions rule under `pd-pair` or `epd`.
- The `best-with-temperature` picker, and three settings of `prefix-affinity`: `decay`, `accounting_ttl_ms` and `spill.uncached_credit`.
- The `bucket` scorer in the stock controller. The library runs it. The stock controller refuses a document that names it, because it does not yet report every request's end to a stage.
- `prefix-affinity` with `source` set to `predictor` or `merged` in the stock controller. Both read the prefixes sent to each worker. The library validates them for a host that fills that column. The crate's `Predictor` keeps that record for such a host: it folds a request's keys once (`Predictor::fold_into`), reads by a shared reference, and writes with a tick of the host's clock. The stock controller refuses a document that names either, because it does not yet record the prefixes it sent.
- Cache events in the stock controller. Its `PushKvEvents` call is not served, and the stock build has nothing else that feeds its cache index. So `prefix-affinity` sees no worker as holding a prefix there. A controller built with the `worker-host` feature fills its index itself: every replica opens a cache-event stream to each gRPC worker whose row asks for events. It keeps those streams while the active document reads no index. The library places by cache for a host that feeds the index. A host built with the `worker-host` feature has the feed's two parts: `worker_host::KvAdapter` writes a worker's cache events to the index, and `worker_host::KvHost` gives each event subscription its row. A row above rank 0 of a data-parallel worker gets no feed there.
- One more input in the stock controller. A worker's circuit-breaker bit has one source there: a veto of the asking gateway that holds, as its breaker's open window does. Such a veto takes the worker out of that gateway's decisions, and no rule lifts it. Its row reads as stalled too, as under every veto of that gateway, so a filter that reads the stalled bit refuses it. The controller keeps no breaker of its own.
- An engine connection over the host-local ZMQ wire. A controller built with the `worker-host` feature hands each worker object a gRPC connector alone, and an object connects to its engine at its first use. A worker that serves only the ZMQ wire gets no connection there, so nothing that controller reads from an engine reaches it for that worker.
- A plugin named in the document. The format has no field for one.
- Four calls of the contract: `PushLoad`, `PushKvEvents`, `StepDown` and `WatchRegime`. The stock controller answers each as not served.
- Eight of the ten slices of the snapshot call, `GetSnapshot`. The stock controller serves the worker rows and the spec's status. It refuses a request for any other slice by the slice's name, and a request that names no slice asks for all ten. It also refuses a request for identifiers in clear.
- The audit mode of the apply call.
- The gateway's use of the controller.
