# Policy Controller Example

A policy controller of your own: the stock controller with one scorer added
in Rust. The command line, the listeners, the election, the store and the
policy documents are the crate's (`crates/policy_controller`). The scorer is
this package's. This page is the guide: write a stage, register it, run.
The crate's [README](../../crates/policy_controller/README.md) explains the policy document.

Write a stage in code when a routing rule cannot be written as a policy
document over the built-in stages and you want it compiled into the
controller. It runs on the decision path as a built-in does: no sandbox, no
budget and no copy of the view. When the rule must change without a rebuild,
or comes from code you would not link into the controller, write a plugin
instead ([`wasm-guest-policy`](../wasm/wasm-guest-policy)).

## What it shows

- `Pack`, a scorer (`src/main.rs`): among the workers the profile's filters
  kept, the one with the most requests running, by its engine's own load
  report, scores highest. Light traffic fills one worker, the next opens
  when a filter drops the first, and whole workers stay idle.
- `stages()`: the stock build's catalog with `Pack` added under the name
  `pack`.
- `main`: the crate's command line and run loop over those stages.
- `policy.yaml`: a policy document that names `pack` as it names a built-in.
- The tests: a worker list as a controller holds one (the worker crate's
  worker objects, their status, their engines' load reports through a load
  feed) is handed to the engine with each request, and the request goes to
  the busy worker, then to the idle one once the busy one is taken out of
  rotation. A worker that joins late is in force at once. A worker with no
  report yet counts as idle, and an idle fleet ties. A route's own rule
  takes a worker out for its requests alone. A worker judged overloaded is
  passed over, and the idle one opens. The catalog a controller is bound
  with validates the document. Handed the document as its spec file, that
  controller runs it as its generation 1, the document decides over a
  worker list by the controller's own stages, and the stock controller
  refuses the file by the scorer's name. The built binary's start-up line
  names `pack` among its scorers, and started on the document it runs
  generation 1. It runs on the allocator it registers, with the settings it
  exports. Handed a worker through its door, it polls the engine behind it.
  Asked by a gateway to decide between two such workers, it names the one
  with requests running.
- `Cargo.toml`: the crate with its `controller` feature, and the allocator
  the binary registers.

## Write, register, run

1. Write the stage. Implement `Scorer`, `Filter` or `Picker` for a type of
   yours. A scorer writes one score for each row it is handed, from 0 to 1
   with higher better, and allocates nothing. A stage declares what it reads
   in `needs`: `Pack` declares the load reports. A stage that only places
   says so in `refuses_by_reports`, as `Pack` does. A stage that says nothing
   is taken as one that may refuse a request by its gateways' reports of
   leases, and each decision under it applies those reports first. A stage
   that keeps no state takes every default of `Lifecycle`.
2. Register it. Start from `StageRegistry::builtin()` and add the stage under a new
   name with `scorer`, `filter` or `picker`. The name must be free among the
   built-ins and across the three kinds, and it must be the name the stage
   answers.
3. Run. From a plain `fn main`, hand the catalog to `controller::run` and
   return its exit code. `run` builds the async runtime itself and panics
   inside another.

```rust
fn main() -> Result<ExitCode, StageRegistryError> {
    let mut stages = StageRegistry::builtin();
    stages.scorer("pack", || Box::new(Pack))?;
    Ok(controller::run(stages))
}
```

The closure is a factory. The catalog calls it once at registration, to read
the stage's name, and drops that stage. Each validation of a policy document
over the catalog calls it again, once for each place the document names the
stage, so every generation of the policy gets stages of its own. Keep the
factory cheap: its first call runs in `main`, before the command line is
read. State that must outlive a generation rides the closure.

## What a stage is handed

A stage reads one decision's view and nothing else: the same view a built-in
reads. The controller builds the workers' rows from its store, and its store
does not fill every field yet. A field it does not fill reads as its default,
with no error: a stage over one ties or keeps every worker.

| A stage reads | Where it comes from | In this controller today |
|---|---|---|
| `Candidate::health` | The prober's status of the worker, the pool's circuit breaker and the overload verdict on its engine's load reports. | The prober's status. A veto of the asking gateway is in the row only where the rows were written for the request: the stalled bit, and the breaker bit beside it where the veto holds, as the gateway's own open breaker does. Beside rows the store lends, both bits read clear and the veto stands in the constraints: the row's position in `Constraints::stalled`, and in `Constraints::held` where the veto holds. So a stage asks `Constraints::masks` beside the stalled bit, as `healthy` does. The controller keeps no breaker of its own. The overload bit reads as clear unless the controller's load monitor judges the engine's load reports over a threshold. |
| `Candidate::load` (`LoadRow`) | The engine's own load report: requests running and queued, KV-cache usage, the drain rate, the report's age. A stage that reads it declares `Needs::loads`. | Fed. |
| `Candidate::in_flight` | The requests the deciding host holds on the worker right now. | The leases on file that book the worker, read through `RequestFacts::in_flight`: beside the rows the store lends, in the row where the rows are written for the request. The field itself reads 0 beside lent rows, so a stage asks the request's facts. A decision reads one count a row. |
| `Candidate::session_hit`, `session_in_flight` | The request's session key: whether it is pinned to this worker, and its requests in flight there. | `session_hit` is set on the worker the ask names as pinned. `session_in_flight` is the row's own count of the key's requests in flight. Only a keyed request placed on one worker counts and is shown it: the controller steps the count on the worker's object while that request's lease is on file, and writes it into the row under a document that reads it, one with a stage that declares `Needs::session_in_flight` or a sessions rule with an in-flight cap. For any other request it reads 0. |
| `RequestFacts::affinity_at(index)` (`AffinityFacts`) | The cache index: the prefix blocks of the request the worker at `index` holds. An empty column reads as nothing held. Filled for a profile with a stage that declares `Needs::affinity`. | One entry for each worker. Its index facts are not fed yet and read 0. |
| `Candidate::role`, `wire_modes`, `zone`, `engine_kind`, `labels` | The worker's registration. | `role` and `wire_modes` are fed. `zone`, `engine_kind` and `labels` are fed against the lists the policy document declares (`view.zones`, `view.engines`, `view.labels`): a position from 1, and one bit for each label selector the worker meets. Each reads 0 where the document declares no list, and a worker whose zone or engine family its list does not name reads 0 there. |
| `DispatchCredit::read` (`CreditedDispatches`) | What the deciding host sent the worker that no load report of it shows yet: the requests and their prompt tokens. A scorer's factory asks for its profile's count with `StageBuild::dispatch_credit`. | Fed by the dispatch reports of every gateway that decides through this controller. A send stops counting at a report sampled at or after it, at the profile's `sent_ttl_ms` or at the worker's removal. When a send arrives with 8,192 already held on a row, the tally lets the row's oldest go, at any age. |
| `RequestFacts` | The request: the model, the tenant, the session key, the prompt's size and block hashes, the allow-listed headers. | The caller's, not the store's. |
| `PoolFacts`, at `DecisionScope::open` | The pool the model resolves to: its size, the policy's generation, the overload regime. A stage that holds a decision scope reads it when the scope opens. | The size and the generation. The overload regime is not fed yet and reads as none. |

`Pack` reads `Candidate::load` and says so in `needs`. The example's tests
hand the engine a worker list and its engines' load reports, and read where
a request goes.

## The policy document

```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [healthy, not-overloaded]
    score: [pack]
    pick: best
```

`Pack` takes a weight from the document and nothing else:
`- pack: { weight: 2 }`. A bare name weighs one. A scorer with settings of
its own registers with `scorer_with` and reads them from the document as
typed parameters: the crate's README shows one. The command line is the
crate's, and it refuses a flag it does not know.

The stock `not-overloaded` filter is where `pack` stops filling a worker. It
drops a worker judged overloaded: the queue or the KV-cache usage in its
engine's load reports stands at or over a threshold. A started controller's
load monitor judges each worker by the threshold its registration names, or
by the program's: 8 queued requests and a usage of 0.8. A worker of the
`external` runtime is judged by its registration's alone. The example's tests
show what the verdict does: a worker judged overloaded is passed over, and
the next one opens.

The stock build's catalog refuses this document by the scorer's name. The
published schema (`crates/policy_controller/policyspec.schema.json`) cannot
know a host's catalog and takes a scorer by any name, so the document passes
it. Only validation over a catalog tells the stock build from this one.

## Build and run

```sh
cargo build --release -p policy-controller-example
target/release/policy-controller-example --controller-id controller-0 \
  --spec-file examples/policy-controller/policy.yaml
```

`--spec-file` hands the controller its policy document. The file is read at
the start, judged over this build's stages and run as generation 1. It is
watched from then on: a changed file is applied as the next generation, and
one that does not run is rejected whole while the running generation stays.
The stock controller refuses the same file by the scorer's name.
`smg_policy_spec_generation` on the admin listener's `/metrics` names the
generation that runs.

The flags are the stock controller's, and `--help` lists them. `--log-level`
sets the level of the crate's own log lines. Lines your own crate writes show
at `warn` and above unless `RUST_LOG` says otherwise. `controller::run`
installs the process's log subscriber and panics when one is already set, so
`main` installs none.

## The allocator

The library registers no allocator: a controller built on it chooses its own.
This example registers the one the stock controller's binary runs on, with
the same settings: `src/main.rs` declares both, and `Cargo.toml` names one
dependency under a target rule. The rule leaves an Apple target out: the
allocator compiles no background thread there, which the settings ask for,
and the example runs on the system allocator. The choice shows under
registrations. Each one builds the store's world again, and the system
allocator keeps resident what those builds free. In a one-thread probe of the
store, 4,096 registrations of one model leave hundreds of megabytes resident
under the system allocator and about a dozen under this one. A controller
that registers none runs on the system allocator and pays that. On Linux
aarch64 the allocator is built for 64 KiB pages, so one binary runs on every
page size: the workspace's cargo configuration sets it, and a repository of
your own needs the same line.

## In a repository of your own

```toml
[dependencies]
smg-policy-controller = { git = "https://github.com/smg-project/smg", features = ["controller"] }
```

- `controller`: the listeners, the command line and `controller::run`.
- `worker-host`: the cache-event feed. `controller` turns it on: a controller reads cache events from each gRPC worker whose row asks for them and names a model.
- `kubernetes`: the Lease election the Helm chart's replicas use.
- `client`: the reference client of the protocol. The example's tests ask
  the started controller its decisions through it.
- The default features hold the WebAssembly plugin host.

The build wants the Rust release this repository pins (`rust-toolchain.toml`)
and `protoc` with the well-known types on its include path: the crate and the
gRPC client it links compile their contracts in their build scripts.

`own-repository.sh <directory>` writes this example as such a repository: its
source, document and tests, a manifest with the line above, the allocator's
dependency and the release profile the stock controller is built with, the
allocator's page setting (`.cargo/config.toml`), and this checkout's
toolchain pin and lock file. The line names no revision, so the first build
takes the crate at the tip of the repository's default branch and the lock
keeps it there. `cargo test` passes in a copy written from a checkout of that
tip. From an older checkout, name its `rev` or `tag` beside `git`. With
`--snapshot` the git source is a snapshot of this checkout, written beside
the copy. CI builds and tests the example that way, against the change under
test.

## The image

The repository's Dockerfile builds any workspace package that depends on the
crate. From the repository root:

```sh
docker build --file docker/policy-controller.Dockerfile \
  --build-arg PACKAGE=policy-controller-example --tag policy-controller-example .
```

CI builds this image beside the stock controller's, starts it and reads
`pack` in its start-up line. It builds it again for arm64 by the file's
cross build, where the final stage runs the binary. It pushes none of them.

`PACKAGE` names a package of the workspace the file builds in. A repository
of your own copies the file: it reads `rust-toolchain.toml` and `Cargo.lock`
at the root of its build context, where cargo also reads
`.cargo/config.toml`, and it builds a package that depends on the crate. No
job builds an image from such a copy: CI's images are this workspace's.

The Helm chart runs `registry/repository:tag` from
`policyController.image.registry`, `policyController.image.repository` and
`policyController.image.tag`. A registry left empty is the router's, and a
repository of your own needs its tag.

## What it leaves out

- Parameters. `Pack` takes none from the document beyond its weight. A scorer
  registered with `scorer_with`, a filter registered with `filter_with` and a
  picker registered with `picker_with` read typed parameters.
- A filter and a picker. They register as the scorer does.
- State. `Lifecycle` reports what became of each decision to a scorer that
  learns from it. A scorer that keeps a number of each worker's load report
  answers true to `keeps_loads`, and is then told every report before a
  decision reads it. `Pack` reads the view alone.
- A fallback. A plugin that fails falls back to a built-in stage. A stage in
  code is the controller's own code and has none.
- A name of its own. `--version` and `/version` answer the crate's name and
  version, as the stock controller does.
- Data the store does not feed yet. The table above names each field that
  reads as its default.
