# WASM Guest Policy Example

A routing policy written as a WebAssembly component: a filter, a scorer and a
picker for the policy controller's plugin host
(`crates/policy_controller/src/wasm`).

The policy: among the healthy workers that have capacity, prefer the one with
the fewest requests in flight, with a bonus for the worker that already holds
the request's session.

This guest is not gateway middleware. The other examples in this directory
target the gateway's middleware host and build for `wasm32-wasip2`. This one
targets the policy host and builds for `wasm32-unknown-unknown`: the policy
host has no WASI and refuses a component that imports any `wasi:` interface,
which a `wasm32-wasip2` build of the standard library carries.

## What it shows

- `init`: the plugin's configuration bytes, read once per instance. Here one
  byte, the session bonus in percent (default 20). An error fails the apply.
- The `decision` resource: the host builds one per request with the request,
  the candidate rows and the pool's facts, calls the stages the profile binds
  and drops it.
- `filter`: the surviving positions. With no usable worker the list is empty
  and the request is shed.
- `score`: one score per candidate, scaled to stay within 0 to 1, where the
  host clamps.
- `pick`: a position among the survivors.
- `observe`: the lifecycle of issued decisions. It counts completed
  dispatches in a counter the plugin's spec declares. The engine forwards the
  lifecycle to scorers and pickers: bound as a filter alone, a plugin
  observes nothing.
- Two of the four host imports: `log` and `counter`.

It reads four fields of a candidate row: `health`, `has-capacity`,
`in-flight` and `session-hit`. The host fills them. `session-hit` is true only
for the worker the host's session table pins the request's key to: where a
host marks none, the bonus does nothing and the policy is least loaded alone.

It keeps no state across decisions beyond its configuration. A policy that
learns from the lifecycle is a different example.

## The contract

The plugin implements the `policy` world of
[`smg-policy.wit`](../../../crates/policy_controller/contracts/smg-policy.wit),
the record form: the view arrives as typed records. The same file defines a
second world, `policy-packed`, where the view arrives as bytes the guest reads
in place. It is the form for a hot path: on the host's bench a three-stage
decision over 64 candidates takes about 19 microseconds through the record
world and about 6 through the packed one.

What the host enforces:

- A call runs under a budget. A stage that runs out, traps or breaks a rule
  below falls back to the profile's fallback stage, and the instance is
  rebuilt.
- `filter` returns positions that are unique and in range.
- `score` returns one finite value per candidate. The host clamps it to 0..1.
- `pick` returns a position among the survivors: the host passes the scores
  back, and a score below zero marks a row a filter removed.
- A stage that cannot judge a request declines (`Err(Decline::...)`). That is
  not a failure: the instance stays and the stage's fallback runs. With no
  fallback bound, what a decline does depends on the stage and on the
  plugin's failure mode (`open` is the default):

  | the stage that declines | `failure: open` | `failure: closed` |
  |---|---|---|
  | `filter` | every candidate stays | the request is shed |
  | `score` | a flat score: the plugin weighs nothing | the same |
  | `pick` | the host ranks by the combined scores | the request is shed |

  A score is a preference and has no veto, so a declined score sheds under
  neither mode: what must hold or shed is a filter. This example judges
  every request, so its stages never decline.

## Building

Prerequisites:

- The Rust toolchain, with the bare WebAssembly target:
  `rustup target add wasm32-unknown-unknown`
- `wasm-tools`: `cargo install wasm-tools`

```bash
cd examples/wasm/wasm-guest-policy
./build.sh
```

The script runs two commands:

```bash
cargo build --release --locked --target wasm32-unknown-unknown --target-dir target
wasm-tools component new \
  target/wasm32-unknown-unknown/release/wasm_guest_policy.wasm \
  -o target/wasm32-unknown-unknown/release/wasm_guest_policy.component.wasm
```

The first builds a core module that carries the world it was built against.
The second wraps it as the component the host loads, about 40 KB. The
`Cargo.lock` is committed, so the example builds from the same dependencies
every time. It moves by hand: the repository's dependency updates and its
audit read the root lock file alone.

## Loading it

A host loads the component through the crate's `wasm` module and binds it as
stages of a profile:

```rust
use std::sync::Arc;

use smg_policy_controller::wasm::{
    Plugin, PluginSpec, Runtime, WasmFilter, WasmPicker, WasmScorer,
};

let runtime = Arc::new(Runtime::new()?);
let mut spec = PluginSpec::new("least-loaded", std::fs::read(component_path)?);
spec.config = vec![20]; // the session bonus, in percent
spec.budget_micros = 10_000; // per call
spec.metrics = vec!["completed".to_string()]; // counter 0
let plugin = Plugin::load(&runtime, spec)?;

let bound = plugin.bind();
let filter = WasmFilter::new(Arc::clone(&bound), None);
let scorer = WasmScorer::new(Arc::clone(&bound), None);
let picker = WasmPicker::new(bound, None);
```

`crates/policy_controller/tests/wasm_example.rs` is the runnable form: it
loads this component into an engine and decides over a small pool. The policy
document does not name a plugin yet, so a host builds the `PluginSpec` itself.

## Testing

From the repository root (the example's directory is a workspace of its own):

```bash
examples/wasm/wasm-guest-policy/build.sh
cargo test -p smg-policy-controller --test wasm_example
```

The tests skip, and say so, when the component is not built. With
`SMG_REQUIRE_POLICY_GUEST=1` a missing component is a failure: CI sets it
where it builds this example.

## Writing your own

Copy this directory, keep the `wit_bindgen::generate!` call pointing at
`smg-policy.wit`, and change the three stages. A profile binds only the stages
it wants from a plugin, and a stage the policy does not implement can decline
every call. A plugin gets the load rows, the hashes, the headers and the
session counts only when its spec names them in `needs`: the host cannot read
what a module wants off the module. Keep `panic = "abort"` in the release
profile: a panic then ends the call as a trap the host handles, with no
unwinding code in the module.
