# Policy Controller Example

A policy controller of your own: the stock controller with one scorer added
in Rust. The command line, the listeners, the election, the store and the
policy documents are the crate's (`crates/policy_controller`). The scorer is
this package's. This page is the guide: write a stage, register it, run.

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
- The tests: a store fed as the controller's ingestion feeds one
  (registration, the prober's status, the engines' load reports) gives the
  rows, and the decision picks the busy worker, then the idle one once the
  prober takes the busy one out. A worker with no report yet counts as idle,
  and an idle fleet ties. A gateway's veto takes its worker out for that
  gateway alone. A worker whose queue reaches its overload threshold is
  passed over, and the idle one opens. The catalog a controller is bound
  with validates the document. Handed the document as its spec file, that
  controller decides by it over its own store, and the stock controller
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
   in `needs`: `Pack` declares the load reports. A stage that keeps no state
   takes every default of `Lifecycle`.
2. Register it. Start from `Catalog::builtin()` and add the stage under a new
   name with `scorer`, `filter` or `picker`. The name must be free among the
   built-ins and across the three kinds, and it must be the name the stage
   answers.
3. Run. From a plain `fn main`, hand the catalog to `controller::run` and
   return its exit code. `run` builds the async runtime itself and panics
   inside another.

```rust
fn main() -> Result<ExitCode, CatalogError> {
    let mut stages = Catalog::builtin();
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
| `Candidate::health` | The prober's status of the worker, the pool's circuit breaker and the overload verdict on its engine's load reports. | The prober's status, and a stall where a veto of the asking gateway stands. The breaker is not fed yet and reads as clear. The overload bit reads as clear unless the store judges the engine's load reports over a threshold. |
| `Candidate::load` (`LoadRow`) | The engine's own load report: requests running and queued, KV-cache usage, the drain rate, the report's age. A stage that reads it declares `Needs::loads`. | Fed. |
| `Candidate::in_flight` | The requests the deciding host holds on the worker right now. | The leases on file that book the worker, read through `RequestFacts::in_flight`: beside the rows the store lends, in the row where the rows are written for the request. A decision reads one count a row. |
| `Candidate::session_hit`, `session_in_flight` | The request's session key: whether it is pinned to this worker, and its requests in flight there. | `session_hit` is set on the worker the ask names as pinned. `session_in_flight` is not fed yet and reads 0. |
| `RequestFacts::affinity_at(index)` (`AffinityFacts`) | The cache index: the prefix blocks of the request the worker at `index` holds. An empty column reads as nothing held. Filled for a profile with a stage that declares `Needs::affinity`. | One entry for each worker. Its index facts are not fed yet and read 0. |
| `Candidate::role`, `wire_modes`, `zone`, `engine_kind`, `labels` | The worker's registration. | `role` and `wire_modes` are fed. `zone`, `engine_kind` and `labels` are fed against the lists the policy document declares (`view.zones`, `view.engines`, `view.labels`): a position from 1, and one bit for each label selector the worker meets. Each reads 0 where the document declares no list, and a worker whose zone or engine family its list does not name reads 0 there. |
| `RequestFacts` | The request: the model, the tenant, the session key, the prompt's size and block hashes, the allow-listed headers. | The caller's, not the store's. |
| `PoolFacts`, at `DecisionScope::open` | The pool the model resolves to: its size, the policy's generation, the overload regime. A stage that holds a decision scope reads it when the scope opens. | The size and the generation. The overload regime is not fed yet and reads as none. |

`Pack` reads `Candidate::load` and says so in `needs`. The example's test
feeds a store through the same doors and reads the decision.

## The policy document

```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [healthy, not-overloaded]
    score: [pack]
    pick: best
```

A scorer in code takes a weight from the document and nothing else:
`- pack: { weight: 2 }`. A bare name weighs one. Settings of your own are
your binary's: read them in `main`, from the environment or a file, and move
them into the factory. The command line is the crate's, and it refuses a flag
it does not know.

The stock `not-overloaded` filter is where `pack` stops filling a worker. It
drops a worker whose engine the store judges overloaded: the queue or the
KV-cache usage in its load reports stands at or over a threshold. The
threshold is the worker's own, named at its registration, or its pool's,
from the policy document applied to the store: 8 queued requests and a usage
of 0.8 where the document names none. The example's tests show both: a fleet
whose workers each name a queue threshold, and a controller handed this
document, where the default stands.

The stock build's catalog refuses this document by the scorer's name. The
published schema (`crates/policy_controller/policyspec.schema.json`) lists
the stock build's stages, so the document fails that schema too and
validates over this build's catalog.

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

- Parameters. A stage in code takes none from the document beyond a scorer's
  weight.
- A filter and a picker. They register as the scorer does.
- State. `Lifecycle` reports what became of each decision to a scorer that
  learns from it. `Pack` reads the view alone.
- A fallback. A plugin that fails falls back to a built-in stage. A stage in
  code is the controller's own code and has none.
- A name of its own. `--version` and `/version` answer the crate's name and
  version, as the stock controller does.
- Data the store does not feed yet. The table above names each field that
  reads as its default.
