# smg-worker

Everything SMG knows about a worker, in one crate.

A **worker** is one serving process the gateway routes to. This crate holds
what a worker is on the wire, and the runtime a process needs to host
workers. The wire types are the default layer: a crate that only reads or
writes them links six crates and nothing else. The runtime sits behind the
`runtime` feature: the worker itself, the registry, the connection to its
engine, health, load monitoring, the KV event collector, the job queue and
metrics. The gateway hosts the runtime. Other crates of the workspace take the
layer they need.

What it holds:

- **Identity and the registry.** Who a worker is: its address, its models,
  its type. The registry keeps the workers of a process and tells its
  subscribers about every registration, removal, replacement and status
  change.
- **Connections.** The handle to a worker's engine over gRPC or ZMQ, and the
  HTTP clients that reach workers.
- **Health.** The probes and the status they drive, the circuit breaker,
  liveness and overload protection.
- **Load monitoring.** The requests in flight on a worker, and the monitor
  that collects each worker's load report.
- **KV cache events.** The collector of the KV cache events an engine pushes,
  and the port to the host's sink that applies them.
- **Jobs.** The worker jobs port that the health manager, discovery and
  registration ask through, and the queue that dispatches and tracks a host's
  jobs.
- **Metrics.** The worker's metric writers, and the store that drops a
  worker's series when the worker leaves.
- **The wire types and the proto.** A worker's spec, info, status and load
  report as the admin API carries them, the model card a worker serves, and
  `proto/common.proto`: the messages every engine's gRPC service shares.

What it does not hold: **routing**. No routing policy lives here. Choosing a
worker for a request is the job of the gateway and of the policy controller
(`crates/policy_controller`).

On this page:

- [Layers](#layers)
- [A host's first steps](#a-hosts-first-steps)
- [The wire types and the proto](#the-wire-types-and-the-proto)
- [The ports a host implements](#the-ports-a-host-implements)
- [Versioning](#versioning)
- [Where things are](#where-things-are)
- [How this page stays true](#how-this-page-stays-true)

## Layers

```toml
[dependencies]
smg-worker = { workspace = true, features = ["runtime"] }
```

That line is a host's, in a crate of this workspace. It asks for the runtime,
and `workspace = true` takes the version from the workspace's root manifest. A
crate outside the workspace names a version instead. The features are the
same. Without `features`, a crate gets the default layer, the first row below.
The default feature set is empty, so every other layer is asked for by name.

| Feature | What it adds | Turns on | Pulls in | Who needs it |
|---|---|---|---|---|
| none | The wire types, the settings and label vocabulary, a worker's address, the overload thresholds and the port between the KV event collector and a host's sink. | | `bitflags`, `schemars`, `serde`, `serde_json`, `serde_with`, `url` | A crate that only reads or writes the types. `openai-protocol` re-exports them. |
| `axum` | The HTTP responses of the admin types. | | `axum` | A crate that answers the admin API. |
| `metrics` | What the worker's metric writers share with a host's: the bounded label set, the one set of model labels and the handles a record keeps. | | `dashmap`, `rustc-hash` | A crate with metric writers of its own. `smg-external-router` is one. |
| `proto` | `common_proto`: the Rust types of `proto/common.proto`. The build then runs `protoc`. | | `prost`, `prost-build` | A crate that speaks gRPC to an engine. `smg-grpc-client` is one. |
| `runtime` | The worker runtime: the worker and its builder, the registry, health, the load monitor, the metric writers, the engine handle, the collector and apply rules for KV cache events, and the job queue. | `metrics`, `proto` | `anyhow`, `arc-swap`, `async-trait`, `chrono`, `futures`, `kv-index`, `metrics`, `metrics-exporter-prometheus`, `metrics-util`, `once_cell`, `openmetrics-parser`, `parking_lot`, `quanta`, `reqwest`, `rustls`, `thiserror`, `tokio`, `tonic`, `tracing`, `uuid` | The process that hosts the workers. The gateway is one. |
| `test-util` | `test_support`: what a test builds its workers with, a scripted engine among it. It also opens the test readers and hooks of the registry, the series store, the label interner, the KV event collector and the job queue. | `runtime` | `tonic-prost` | The tests of a crate that hosts the runtime. |
| `workflow` | `workflow`: the worker side of the workflows a host runs over its workers. A removal in registry order selects and drains workers, then removes them from the registry. In door order it selects workers and waits for them to settle between the host's drain and removal. It also holds a local worker's detection and creation at registration (connection and runtime detection, metadata and data-parallel reads, then the worker objects), the data the steps pass along, and the host trait they read. | `runtime` | `wfaas` | A host that registers or removes workers by a workflow. The gateway is one. |

"Turns on" names other features of this crate. "Pulls in" names the crates a
feature adds to the build. `runtime` turns on `proto`, so the build of a host
runs `protoc` too.

A test, `tests/default_layer.rs`, holds the first row at six crates. It also
keeps every layer from linking `openai-protocol`: that crate depends on this
one, and cargo refuses a cycle.

## A host's first steps

A host is a process that keeps workers. It builds a registry, registers
workers, reads their health and starts the background work. Both blocks need
the `runtime` feature.

```rust
use std::sync::Arc;

use smg_worker::{
    connection::engine::NoConnector,
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    registry::{Generation, WorkerRegistry},
    worker::WorkerStatus,
};

/// What this host keeps of one membership generation: the members alone.
#[derive(Debug)]
struct Members(Arc<[Arc<dyn Worker>]>);

impl Generation for Members {
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        Self(members)
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.0
    }
}

/// This host's name for the worker builder. Its workers speak HTTP, so they
/// need no engine connector.
type WorkerBuilder = BasicWorkerBuilder<NoConnector>;

fn main() {
    // 1. Build a registry.
    let registry = WorkerRegistry::<Members>::new();

    // 2. Register a worker.
    let worker = WorkerBuilder::new("http://127.0.0.1:8000")
        .model(ModelCard::new("example-model"))
        .build();
    let id = registry
        .register(Arc::new(worker))
        .expect("no worker holds this address yet");

    // 3. Read its health.
    let worker = registry.get(&id).expect("the worker was registered above");
    assert_eq!(worker.status(), WorkerStatus::Pending);
    assert!(!worker.is_healthy());
    assert_eq!(registry.get_by_model("example-model").len(), 1);
}
```

What the block shows:

- **`Generation` is the first thing a host writes.** The registry ends a
  generation at every membership change and builds the host's value from the
  members. `registry.generation(model)` hands a reader that value. A host
  that derives nothing from a generation keeps the members, as here.
- **The builder is named once.** Its type parameter says which engine
  connector a worker gets when the builder is handed none. `NoConnector` says
  none, which is enough for workers that speak HTTP. A gRPC or ZMQ worker
  needs a connector: see [the ports](#the-ports-a-host-implements).
- **`register` is create-only.** It returns the worker's id, or `None` when
  an active worker already holds the address.
- **A new worker is `Pending`.** It is registered, not yet proven healthy,
  and not routable. `is_healthy()` is true only for `Ready`. Before a request
  goes to a worker, ask `is_available()`: the worker is ready, its circuit
  breaker lets requests through, and no overload or stall veto stands on it.

The background work starts after the first workers are registered, inside a
tokio runtime:

```rust
use std::sync::Arc;

use smg_worker::{
    health::manager::{WorkerManager, WorkerManagerConfig},
    identity::basic::Worker,
    load::monitor::{LoadConsumer, WorkerMonitor},
    registry::{Generation, WorkerRegistry},
    settings::UnhealthyRemovalScope,
};

/// The routing side as the load monitor sees it. Nothing in this host reads
/// loads or books requests on a worker.
struct NoReaders;

impl LoadConsumer for NoReaders {
    fn needs_loads(&self) -> bool {
        false
    }

    fn reconcile_in_flight(&self, _worker: &dyn Worker) {}
}

/// Starts the health checks and the load monitor over `registry`.
fn start<G: Generation>(
    registry: &Arc<WorkerRegistry<G>>,
) -> (WorkerManager, Arc<WorkerMonitor<G>>) {
    // 4. Start the health checks.
    let health = WorkerManager::start(
        Arc::clone(registry),
        WorkerManagerConfig {
            default_check_interval_secs: 30,
            remove_unhealthy: UnhealthyRemovalScope::Disabled,
        },
        None,
    );

    // 5. Start the load monitor.
    let poll_interval_secs = 10;
    let engine_metrics = false;
    let conditional_polling = false;
    let monitor = Arc::new(WorkerMonitor::new(
        Arc::clone(registry),
        Arc::new(NoReaders),
        poll_interval_secs,
        engine_metrics,
        conditional_polling,
    ));
    monitor.start_event_loop();

    (health, monitor)
}
```

What the block shows:

- **The health manager moves a worker's status.** It probes each worker when
  its check is due. A `Pending` worker becomes `Ready` after enough probes
  pass in a row, two by default. A `Ready` worker becomes `NotReady` after
  enough fail in a row, three by default. Keep the returned `WorkerManager`:
  dropping it stops the checks. A host that keeps a worker's status elsewhere
  hands its sink to `WorkerManager::start_with_sink`, or to
  `WorkerManager::maybe_start_with_sink`, which starts the loop only where
  `WorkerManager::maybe_start` would.
- **`None` leaves the `WorkerJobs` port empty.** The manager then asks nobody
  to remove a failed worker or to read a worker's labels again.
  `UnhealthyRemovalScope::Disabled` says the same of removal: a failed worker
  stays registered and probed.
- **The load monitor follows the registry's events.** It polls the ready
  workers of each group, where a group is one model, one worker type and one
  connection mode. It publishes their reports as one immutable snapshot.
  `monitor.load_snapshot()` reads the current one, and `monitor.subscribe()`
  receives every new one.
- **The two flags are off here.** `engine_metrics` re-exports what each poll
  brings. `conditional_polling` polls a group only when something needs its
  loads; without it every group is polled.

## The wire types and the proto

```rust
use smg_worker::worker::{ConnectionMode, WorkerSpec, WorkerStatus, WorkerType};

let spec: WorkerSpec = serde_json::from_str(r#"{"url": "http://127.0.0.1:8000"}"#)
    .expect("a spec needs only its url");
assert_eq!(spec.worker_type, WorkerType::Regular);
assert_eq!(spec.connection_mode, ConnectionMode::Http);

let status = serde_json::to_string(&WorkerStatus::NotReady).expect("a status is a string");
assert_eq!(status, r#""not_ready""#);
```

The wire types need no feature. `WorkerSpec` is the one description of a
worker: the admin API takes it as input and returns it as output, so only
`url` is required. `WorkerInfo` is what the admin API answers for a worker.
`WorkerStatus` is its lifecycle: `pending`, `ready`, `not_ready`, `failed`,
`draining`. `WorkerLoadResponse` is the load report of one worker. A rank's
report in it (`SchedulerLoadSnapshot`) holds fixed figures and the values an
engine reports by name (`NamedValue`). The named values are those of the
rank's newest record: a pushed record's list takes the place of the last.
What a pushed record or a polled answer names past the list's bounds is
dropped and counted by worker in `smg_engine_load_values_dropped_total`. A
`values` key of another shape fails no answer: it reads as no list. These
types live in the `worker` module, and the model card in `model_card`.

The proto is one file, `proto/common.proto`, package `smg.grpc.common`. It
declares the messages every engine's gRPC service shares: the tokenizer's
chunks, the KV cache events, the engine load record, the cache flush, the
profiling calls and the tensor handles. A crate uses it in one of two ways.

**As Rust types.** The `proto` feature generates them into
`smg_worker::common_proto`.

**As a file**, to compile protos of its own that import it. The build script
of such a crate asks cargo where the file is:

```rust,no_run
// build.rs of a crate with `smg-worker` among its `[dependencies]`.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("DEP_SMG_WORKER_PROTO_DIR")?;
    println!("cargo::rerun-if-changed={dir}/common.proto");
    // Give `dir` to the proto compiler as an include path, beside the
    // crate's own proto directory.
    Ok(())
}
```

The variable's name has two parts. `links = "smg_worker_proto"` in this
crate's `Cargo.toml` gives `DEP_SMG_WORKER_PROTO`. This crate's build script
prints `cargo::metadata=dir=<path>`, which gives `_DIR`. Cargo hands the
variable to the build script of every crate that depends on this one
directly, and to no other. The path is a copy of the proto directory under
the build's output directory, so a dependent finds it the same way inside this
workspace and from a published copy of the crate.

`smg-grpc-client` does both. Its build script compiles the engine protos
with this directory on the include path. It maps the shared package to
`common_proto` with `extern_path`, so the shared messages are generated
once, here.

## The ports a host implements

A port is a trait this crate calls and a host implements.

| Port | Module | What the host says through it | In the gateway |
|---|---|---|---|
| `Generation` | `registry` | What it keeps of one membership generation. | The snapshot its routing reads. |
| `EngineConnector` | `connection::engine` | How to connect a gRPC or ZMQ worker to its engine, and how to dial a gRPC address once. | Its gRPC and ZMQ connector. |
| `EngineConnection` | `connection::engine` | The calls the runtime makes on a connected engine: liveness, the health and load reads, the labels the engine reports of its model and of itself, the admin calls and the two pushed streams. | Its client of an engine. |
| `ConnectorDefault` | `connection::engine` | The connector a builder uses when it is handed none. | The same connector. |
| `WorkerJobs` | `jobs` | How to remove workers at an address, how to read a worker's labels again, how to register a worker, and whether a job of an address is in flight. | Nothing to write: the crate's `JobQueue` implements it over the host's `JobHost`, for a job type that accepts a `WorkerJob`. |
| `JobHost` | `jobs::queue` | What jobs the queue takes and how they run. | Its application context. |
| `LoadSink` | `load::port` | What a pushed record or a poll's answer means for routing: the receipts and the poll decision, the overload verdict, what its readers cache. | Nothing to write: `WorkerMonitor::new` builds the crate's own sink over the registry and the routing side. A host with a load store of its own writes this trait and hands it to `WorkerMonitor::with_sink`. |
| `LoadConsumer` | `load::monitor` | Whether a reader wants loads, and how to trim what readers booked on a worker. | Its policy registry. |
| `HealthSink` | `health::port` | What a probe's outcome or a connect signal does to a worker's status, and that a worker's liveness veto moved. | Nothing to write: the crate's registry is the sink, and `WorkerManager::start` hands it. The collector calls `liveness_moved` when a worker's liveness veto may have moved. A host that keeps the veto outside the worker object implements it, and the default does nothing. |
| `RequestCompletionSink` | `identity::basic` | What to do when a request on a worker ends. | The sink its policy registry installs. |
| `KvEventSink` | `kv_events::port` | What to do with the KV cache events a worker's stream brings: keep each rank's cursor, apply what it admits, and drop a worker's blocks when it leaves. | The sink over its KV indexes. |
| `KvEventHost` | `kv_events` | What a worker's events subscription feeds: the sink its task opens, and what goes when a model's last events subscription is gone. | Its index side: the models' KV indexes. |
| `WorkflowHost` | `workflow` | What the steps of a worker workflow read of their host: its registry, its worker clients, its registration settings and its engine connector. It also names the type of the notes a host keeps on a registration. | Its application context. |

A host must write two of them: `Generation`, to have a registry, and
`LoadConsumer`, to run the load monitor over the crate's own sink. A host
with a load store of its own writes `LoadSink` in its place. The others are
for a host that needs them: one with gRPC or ZMQ workers, one that removes
failed workers, one whose routing books requests, one that keeps an index of
the engines' KV caches, one that runs the worker workflows.

The KV event collector keeps one task a worker (`KvEventCollector`). A host
adds a worker with the `SubscriptionKind` it wants and removes it. The
collector starts the task, restarts it when the worker is registered again
under its address, and ends it.

The collector is built over a `KvEventHost`, which says what an events
subscription feeds: `enroll` when the subscription is added, `open` when its
task opens the worker's place in the sink, and `model_left` when a model's
last events subscription is gone. A subscription for the pushed load records
alone (`SubscriptionKind::Loads`) calls none of the three.

## Versioning

A change to `proto/common.proto` asks more of a release than any other change
to this crate. The release check (`scripts/check_release_versions.py`) then
wants a version outside the caret of the last released one, whatever the
commits say: at `0.y.z` the minor number moves. The reason is the dependent
that builds its messages from the file: its published copy would otherwise
resolve to the new version and build messages that gained a field.

For any other change under `crates/worker` the step follows the commits: major
for a breaking one, minor for a `feat`, patch otherwise. The Python package
`smg-grpc-proto` ships the proto file through a link, so its version moves
with the file too.

## Where things are

Each line is a module of `src/lib.rs`. An empty feature cell means the module
is always there.

| Module | Feature | What it holds |
|---|---|---|
| `health` | | The overload thresholds. With `runtime`: the health manager, liveness, the circuit breaker, the resilience settings and the probe of a bootstrap port. |
| `identity` | | A worker's address, parsed once and keyed, and the sampling defaults a worker reports. With `runtime`: the `Worker` trait, `BasicWorker`, its builder, its metadata, the worker errors and the door a worker's address passes. |
| `kv_events` | | The port between the collector of the KV cache events an engine pushes and a host's sink. With `runtime`, the collector keeps one task a worker, the host supplies the sink, and `apply` puts events into an index. |
| `labels` | | The labels of a worker's spec that the worker's own code and its hosts read, and `flat_labels`, which turns an engine's answer into a flat map of labels. |
| `metrics_labels` | | The metric label values the worker's writers share with a host's. |
| `model_card` | | The model card a worker serves. |
| `model_type` | | The endpoints a model supports. |
| `models` | | The entry a model card becomes in a model listing. |
| `settings` | | The settings a host's configuration names and the worker's code reads. |
| `worker` | | The wire types: a worker's spec, info, status and load report, and the admin API's requests and answers. |
| `metrics` | `metrics` | The label mechanism shared with a host's writers. With `runtime`: the worker's metric writers and their series store. |
| `common_proto` | `proto` | The Rust types of `proto/common.proto`. |
| `connection` | `runtime` | The engine handle a host implements, the HTTP client factory and the cache of worker-directed clients. |
| `jobs` | `runtime` | The worker jobs port, the jobs it submits, and the queue that dispatches and tracks a host's jobs. |
| `load` | `runtime` | A worker's requests in flight, the load monitor, the snapshots it publishes, the fleet's capacity and the merge of scraped metrics. |
| `pd` | `runtime` | What a prefill or decode worker says of its side of a pair. |
| `periodic_task` | `runtime` | A task on a thread of its own. |
| `registry` | `runtime` | The registry and its events. |
| `runtime` | `runtime` | A worker's shared runtime state: what a replacement at the same address inherits. |
| `test_support` | `runtime`, `test-util` | What a test builds its workers with. |
| `workflow` | `workflow` | A removal in registry order selects and drains workers, then removes them from the registry. In door order it selects workers and waits for them to settle between the host's drain and removal. The module also holds a registration's detection and creation, the workflow data and the host trait. |

## How this page stays true

- Every Rust block on this page is a doc test of the crate.
  `cargo test -p smg-worker --features runtime` runs them.
- `tests/readme.rs` compares the feature table with `Cargo.toml` and the
  module map with `src/lib.rs`, both ways. It fails on a feature or a module
  that one side holds and the other does not, and on a row whose features
  differ.
- No test holds the ports table. A change that adds or removes a port edits
  its row by hand.

When either fails, change this page in the same pull request.
