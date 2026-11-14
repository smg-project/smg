//! The WebAssembly plugin host (design section 9): user policies as
//! components of the committed `smg:policy` package, run as stages of the
//! framework.
//!
//! What a module is: one component exporting the record world
//! (`smg:policy/policy`, the view as records) or the packed world
//! (`smg:policy/policy-packed`, the view as the bytes of the WIT's layout
//! table); the host reads which from its exports. The host accepts
//! `smg:policy@1.x` up to [`Abi::SUPPORTED_MINOR`] and a spec may pin the
//! version. `examples/wasm/wasm-guest-policy` is a record-world module to
//! copy, with its build line; `tests/wasm_example.rs` loads what that line
//! builds.
//!
//! The sandbox: no WASI. A component importing any `wasi:` interface, or
//! anything but `smg:policy/host`, is refused at load. The host lends four
//! functions: a rate-limited log with a level cap, the decision's clock, the
//! counters the spec declared, and a draw from the decision's seed.
//!
//! How a module runs: one instance per module per deciding thread, created
//! at apply ([`Plugin::warm`] on every shard thread; a thread that was not
//! warmed instantiates at its first decision and the plugin counts it) and
//! kept. Each profile that binds the plugin holds one [`Bound`] set, taken
//! from its engine build's [`Binder`] (one adapter of a build forwards the
//! lifecycle, so an event reaches the plugin once per engine); the
//! engine opens it once per decision over the profile's view (a pair
//! decision opens one per side) and the stages read one `decision` resource,
//! dropped at the end. Every call runs under fuel: the plugin's frame budget
//! clamped to the decision's remaining budget (one remainder per thread per
//! decision, shared by every plugin and side it opens), times the host's calibration
//! of fuel to microseconds, with an epoch deadline of what the decision has
//! left (at most the frame) as the wall bound. The budget is wall time: a
//! call is charged the larger of its fuel at the calibration and its elapsed
//! time, so a bulk memory loop (one unit of fuel whatever its length) draws
//! the remainder down too, inside one linear memory under a cap. A frame budget gone on
//! arrival is never forwarded: the stages fall back and the breaker counts a
//! budget hit, the instance standing.
//!
//! What a failure does: a `decline` falls back to the stage's configured
//! fallback and keeps the instance; a trap, an exhausted budget or an answer
//! that breaks a validity rule (a duplicate or out-of-range filter position,
//! a non-finite score, a pick outside the survivors) falls back too, drops
//! the instance (rebuilt once at the decision's close, for a slot the
//! decision emptied; a faulted instance is never reused) and counts on the
//! breaker. The decision is the breaker's unit: a plugin fails a decision
//! once, however many of its opens fail in it (each side of a pair, each
//! binding), and its later opens of that decision fall back without a call.
//! A call the decision's remainder ends short of the plugin's own bounds (a
//! grant below its frame, by a plugin whose own spend did not consume the
//! budget alone) is the decision's budget spent, its instance dropped and
//! the breaker untouched, whichever scope ran first: under the default
//! frame a plugin is charged only when it alone consumed the budget. A cold
//! slot's `init` is judged under its own frame alone: one the remainder
//! clamps and ends (its charge reached what was left) is the decision's
//! budget spent and the close rebuilds the slot; one that ends by itself
//! short of the clamp is the plugin's own failure, as under the full frame.
//! A resource's destructor runs under a small fuel frame and a wall bound
//! of its own, so a spent budget drops no healthy instance and a cheap slow
//! destructor cannot stretch a decision.
//!
//! The breaker opens for a cool-down on a run of failures within its window,
//! or when the moving p50 of the plugin's decisions passes half the budget
//! or the p99 (the second-slowest of the ring) passes it: one sample per
//! decision, the plugin's own charge in it (its constructor and stage
//! calls, whatever ended them, and a lazy `init` the decision paid) over
//! the decision's budget, taken at its last close, the ring the thread's
//! across rebuilds. An opening writes one warning with the plugin, the rule
//! and the cool-down; a report made while the breaker is open, by a call
//! admitted before it opened, is dropped, so one incident is one opening.
//! Open, the stages take their fallbacks without a call. Without a fallback
//! stage, `failure: open` steps aside (keep all, flat scores, rank by score)
//! and `failure: closed` sheds (an emptied pool or a declined pick naming
//! the plugin).
//!
//! What is not here yet: the spec's `plugins` section that names a module,
//! its memory budget over every row and `max_candidates` (the row lands with
//! the configuration schema; until then a host builds a [`PluginSpec`]
//! itself and no document names a plugin), sourcing (a file or an image
//! reference pulled at validation), and the gRPC plugin service.

mod bindings;
mod breaker;
pub mod config;
mod counters;
mod plugin;
mod stages;
mod view;

use std::{
    collections::HashSet,
    sync::{Mutex, OnceLock},
};

pub use config::{Abi, BreakerSpec, Exposure, Failure, HeaderSet, LogLimit, PluginSpec, Reveal};
pub use counters::Counters;
pub use plugin::{Binder, Bound, LoadError, Plugin, Runtime, Stats, World};
pub use stages::{WasmFilter, WasmPicker, WasmScorer};
/// The packed form's layout, as the WIT's table names it.
pub use view::layout;

/// A stage's name lives as long as the process: the trace and the shed
/// carry `&'static str` names, and a spec that reloads a plugin under the
/// same name reuses the one allocation.
pub(crate) fn intern(name: &str) -> &'static str {
    static NAMES: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let names = NAMES.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = names
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(known) = guard.get(name) {
        return known;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    guard.insert(leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::intern;

    #[test]
    fn a_name_is_interned_once() {
        let a = intern("plugin-a");
        let b = intern("plugin-a");
        assert!(std::ptr::eq(a, b), "one allocation per distinct name");
        assert_ne!(intern("plugin-b"), a);
    }
}
