# The policy controller's wire contracts

Three files, two contracts, one vocabulary. Nothing here is compiled into the
gateway or the engine crate; a crate test compiles and parses the files and
holds them to the crate's own types.

| file | what it is | design |
|---|---|---|
| `policy_plugin.proto` | package `smg.policy.v1`: the view (`RequestFacts`, `Candidate`, `LoadRow`, `Constraints`, `PoolFacts`, `Identity`), the decision (`Decision`, `Pick`, `Target`, `Pair`, `Shed`, `Queue`, `Trace`, `Lease`, `Busy`), the lifecycle (`LifecycleEvent`) and the `PolicyPlugin` service a user policy serves over gRPC | 4.1, 4.2, 4.3, 4.5, appendix A |
| `policy_controller.proto` | the same package: the `PolicyController` service of the standalone controller in its four scopes (gateway, ingestion, control, replication), its frames, credits, worker events, snapshots, spec operations, leadership and the replication log; imports the file above and the engine wire's `common.proto` for the relayed load record and event batch | 5.1 to 5.5, 8, appendix A |
| `smg-policy.wit` | package `smg:policy@1.0.0`: the `types` and `host` interfaces and the two worlds a WebAssembly policy targets, `policy` (the view as records) and `policy-packed` (the view as bytes) | 4.5, appendix B |

Section numbers are the policy controller design's. The in-process form of
every view, decision and lifecycle message is the crate's own type in
`src/view.rs`, `src/decision.rs` and `src/lease.rs`, field for field and in
the same order; a wire message is that record without the fields the host
keeps for itself (a candidate's position and sample instant, a request's host
clock, a lease's issue instant), and each file's comments name them.

## Versioning and compatibility

- The proto package carries the major version. Within `v1` every change is
  additive: new fields take new numbers, removed fields are `reserved` by
  number and name, enum values are never renumbered, a `oneof` only grows. An
  unknown `oneof` case decodes as unset: a `Decision` with no outcome is a
  deadline miss, a `PluginDecision` that answers none of its asked stages a
  decline. An empty filter mask is a present, empty `survivors` message and
  sheds the request; it is never a decline. `Hello` carries the protocol
  range and `Welcome` the chosen version; the controller speaks N and N-1 and
  upgrades first. No absolute instant crosses the decide, session, plugin and
  lifecycle wires (budgets, ages and vetoes are durations relative to receipt);
  the control scope's audit records carry wall-clock instants on purpose.
- The WIT package carries the world version. The engine accepts
  `smg:policy@1.x` within a stated range and the spec may pin `abi: 1.0`. The
  component model types a record, variant, flags or function by its exact
  shape, so within `1.x` no type or signature changes: a `1.y` release adds
  host functions, interfaces or worlds a module may leave unused, a module's
  extra exports are ignored, and a changed shape is a new type or the next
  major. The loader's error names the module's version and the accepted
  range.
- Both are pre-release until the first shipped release that carries them; the
  rules above bind from then on.
- `tests/contracts.rs` pins the names: the RPC list of each service with its
  streaming shape, the cases of every `oneof`, the worlds, the interfaces and
  their functions, and the package names and versions. An additive change
  extends a list there on purpose; a removal or a rename fails the test.

## Consuming the contracts

- A gRPC plugin or a client of the controller compiles the two proto files
  with `protoc` and the generator of its language; the include path holds
  this directory and the engine wire's directory (`crates/grpc_client/proto`
  in this repository) for `common.proto`. In Rust the generated types come
  from `tonic-prost-build` as the repository's own gRPC clients do.
- A WebAssembly plugin targets one world of `smg-policy.wit` with
  `wit-bindgen` (or the generator of its language) and ships as a component.
  The engine refuses a component that imports any `wasi:` interface: a Rust
  guest builds without `std` and exports its own `cabi_realloc`. The host
  imports are `log`, `now-micros`, `counter` and `random-u64`; the exports are
  `init`, the `decision` resource (constructor, `filter`, `score`, `pick`) and
  `observe`. A `policy-packed` guest reads its three blocks in place: the
  offsets are in the WIT comments, the model name is padded so the u64
  sections stay 8-byte aligned for any name length, and the guest's
  `cabi_realloc` returns 8-aligned memory for the blocks (or it reads with
  unaligned loads).
- The validity rules a plugin must keep are in the files: surviving positions
  unique and in range, one finite score per candidate clamped to `0..1`, a
  pick inside the surviving set; a violation is a trap (the instance is
  re-instantiated), a `decline` runs the stage's fallback and keeps the
  instance.

## Conformance

A plugin author checks a module or a gRPC service with the conformance
harness (`smg policy record | test`, a later step). A `DecisionRecord`
carries the request, the scored rows with their ranks, the decision, each
plugin's exposure, the joined outcome and a digest of the view as decided; it
does not yet carry the candidate rows, the pool facts or the constraints, so
a replay rebuilds the view from the fixture, which keeps them beside the
record, checks it against `view_digest`, then runs the plugin in place of the
recorded stages. The verdicts cover determinism, validity, budget, fallback,
exposure, replay agreement, lifecycle, version, transport and the packed
layout. The fixtures (the records with their views) live in the fixture
repository (bellwether) under a pinned tree, never in this repository.

## Wire budgets

Measured with prost 0.14 over these files (typical frames; proto3 omits a
field at its default): a decision costs about 110 bytes plus 8 per block hash
and the headers in and about 170 bytes out (a pair with two decode alternates
260); a candidate row with its load row is about 90 bytes, so a gRPC plugin's
record view is about 42 KB at 512 candidates, hence `plugins[].max_candidates`
(default 128); a credit grant (a delta: each decide request spends one, no
answer returns one, a new stream starts at the welcome's window) is 6 bytes
in its envelope (20 on the wire); a
`HelloPage` of 100 leases and 50 pins is about 9.3 KB, so a 64 KB page holds
about 900 leases; the replication events are 40 bytes a load row, 78 a
reservation, 27 its end, 46 a pin and a generation its document (a few KB).

## The checks

`cargo test -p smg-policy-controller --test contracts` runs four tests: the
protocol compiles through `protoc` (on the path or named by `PROTOC`) and its
view, decision and lifecycle messages are the crate's records; the world
resolves and its records and enums are the crate's (by field and case names,
not their types: a width change such as `u8` to `u16` passes this check,
which a type table in a later head closes); the versioned names are the
pinned lists; the packed request block's padding keeps its u64 sections
8-byte aligned and the world's comment states the rule. No code is generated
and nothing at run time reads the files.
