//! What every plugin kind shares: the decisions open on the deciding threads,
//! which each kind's runtime is built over (a decision's number, its budget
//! and whose bound a call ran out under), the breaker with its latency ring,
//! the counters a plugin declares, the parts of its row that are not about
//! its kind, the names its stages carry, and the three adapters that bind a
//! plugin's decision to a profile as the framework's filter, scorer and
//! picker, each with the fallback stage its row names.
//!
//! A kind (the WebAssembly host in the `wasm` module) implements
//! `BoundPlugin`, a trait of this crate alone, for the plugin it bound and
//! builds on these: the engine sees stages and a
//! [`DecisionScope`](crate::DecisionScope), never a kind. The module is built
//! when a kind is.

pub(crate) mod breaker;
pub mod config;
mod counters;
mod open;
mod stages;

use std::{
    collections::HashSet,
    sync::{Mutex, OnceLock},
};

pub use config::{BreakerSpec, Exposure, Failure, FrameBudget, HeaderSet, Reveal};
pub use counters::Counters;
pub use open::OpenDecisions;
pub(crate) use open::{own_bound, HalfOpen, OpenDecision, Part};
#[cfg(feature = "controller")]
pub(crate) use stages::Stage;
pub(crate) use stages::{BoundPlugin, Fallback};
pub use stages::{PluginFilter, PluginPicker, PluginScorer};

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
