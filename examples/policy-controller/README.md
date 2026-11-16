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
  report, scores highest. Light traffic fills one worker before it opens the
  next, and whole workers stay idle.
- `stages()`: the stock build's catalog with `Pack` added under the name
  `pack`.
- `main`: the crate's command line and run loop over those stages.
- `policy.yaml`: a policy document that names `pack` as it names a built-in.
- The tests: a store fed as the controller's ingestion feeds one
  (registration, the prober's status, the engines' load reports) gives the
  rows, and the decision picks the busy worker, then the idle one once the
  prober takes the busy one out. A worker with no report yet counts as idle,
  and an idle fleet ties. A gateway's veto takes its worker out for that
  gateway alone. A controller bound with these stages takes the document,
  and the stock one refuses it by the scorer's name. The built binary's
  start-up line names `pack` among its scorers.
- `Cargo.toml`: one dependency, the crate with its `controller` feature.

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
the stage's name, and drops that stage. The controller calls it again each
time it validates a policy document, once for each place the document names
the stage, so every generation of the policy gets stages of its own. Keep the
factory cheap: its first call runs in `main`, before the command line is
read. State that must outlive a generation rides the closure.

## What a stage is handed

A stage reads one decision's view and nothing else: the same view a built-in
reads. The controller builds the workers' rows from its store, and its store
does not fill every field yet. A field it does not fill reads as its default,
with no error: a stage over one ties or keeps every worker.

| A stage reads | Where it comes from | In this controller today |
|---|---|---|
| `Candidate::health` | The prober's status of the worker, the pool's circuit breaker and the last load report's overload verdict. | The prober's status, and a stall where a veto of the asking gateway stands. The breaker and the overload verdict are not fed yet and read as clear. |
| `Candidate::load` (`LoadRow`) | The engine's own load report: requests running and queued, KV-cache usage, the drain rate, the report's age. A stage that reads it declares `Needs::loads`. | Fed. |
| `Candidate::in_flight` | The requests the deciding host holds on the worker right now. | Not fed yet: reads 0. |
| `Candidate::session_hit`, `session_in_flight` | The request's session key: whether it is pinned to this worker, and its requests in flight there. | `session_hit` is set on the worker the ask names as pinned. `session_in_flight` is not fed yet and reads 0. |
| `RequestFacts::affinity_at(index)` (`AffinityFacts`) | The cache index: the prefix blocks of the request the worker at `index` holds. An empty column reads as nothing held. Filled for a profile with a stage that declares `Needs::affinity`. | One entry for each worker. Its index facts are not fed yet and read 0. |
| `Candidate::role`, `zone`, `labels`, `wire_modes` | The worker's registration. | `role` and `wire_modes` are fed. `zone` and `labels` are not fed yet and read 0. |
| `RequestFacts` | The request: the model, the tenant, the session key, the prompt's size and block hashes, the allow-listed headers. | The caller's, not the store's. |
| `PoolFacts`, at `DecisionScope::open` | The pool the model resolves to: its size, the policy's generation, the overload regime. A stage that holds a decision scope reads it when the scope opens. | The size and the generation. The overload regime is not fed yet and reads as none. |

`Pack` reads `Candidate::load` and says so in `needs`. The example's test
feeds a store through the same doors and reads the decision.

## The policy document

```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [healthy]
    score: [pack]
    pick: best
```

A scorer in code takes a weight from the document and nothing else:
`- pack: { weight: 2 }`. A bare name weighs one. Settings of your own are
your binary's: read them in `main`, from the environment or a file, and move
them into the factory. The command line is the crate's, and it refuses a flag
it does not know.

The stock `not-overloaded` filter is left out: the store feeds no overload
verdict yet, so it would drop nothing here. Add it once the load reports
carry the verdict.

The stock controller refuses this document by the scorer's name. The
published schema (`crates/policy_controller/policyspec.schema.json`) lists
the stock build's stages, so the document fails that schema too and
validates in this controller.

## Build and run

```sh
cargo build --release -p policy-controller-example
target/release/policy-controller-example --controller-id controller-0
```

The flags are the stock controller's, and `--help` lists them. `--log-level`
sets the level of the crate's own log lines. Lines your own crate writes show
at `warn` and above unless `RUST_LOG` says otherwise. `controller::run`
installs the process's log subscriber and panics when one is already set, so
`main` installs none.

## In a repository of your own

```toml
[dependencies]
smg-policy-controller = { git = "https://github.com/smg-project/smg", features = ["controller"] }
```

- `controller`: the listeners, the command line and `controller::run`.
- `kubernetes`: the Lease election the Helm chart's replicas use.
- The default features hold the WebAssembly plugin host.

## The image

The repository's Dockerfile builds any workspace package that depends on the
crate. From the repository root:

```sh
docker build --file docker/policy-controller.Dockerfile \
  --build-arg PACKAGE=policy-controller-example --tag policy-controller-example .
```

The Helm chart runs the image that `policyController.image.repository` and
`policyController.image.tag` name.

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
  reads as its default, and the overload filter is out of the document with
  it.
- A decision through the controller. No call hands the controller a policy
  document yet, so it holds `pack` and judges a document against it, and
  runs no decision with it. The tests run `pack` over the store's rows.
