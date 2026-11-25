//! What every plugin kind shares: the decisions open on the deciding threads,
//! which each kind's runtime is built over (a decision's number, its budget
//! and whose bound a call ran out under), the breaker with its latency ring,
//! the counters a plugin declares, the parts of its row that are not about
//! its kind, the block hashes of a request its view carries, the names its
//! stages carry, and the three adapters that bind a
//! plugin's decision to a profile as the framework's filter, scorer and
//! picker, each with the fallback stage its row names.
//!
//! A kind (the WebAssembly host in the `wasm` module) implements
//! `BoundPlugin`, a trait of this crate alone, for the plugin it bound and
//! builds on these: the engine sees stages and a
//! [`DecisionScope`](crate::engine::DecisionScope), never a kind. The module is built
//! when a kind is.

pub(crate) mod breaker;
pub(crate) mod config;
mod counters;
mod open;
mod stages;

pub use config::{BreakerSpec, Exposure, Failure, FrameBudget, HeaderSet, Reveal};
pub use counters::Counters;
pub use open::OpenDecisions;
pub(crate) use open::{own_bound, HalfOpen, OpenDecision, Part};
#[cfg(feature = "controller")]
pub(crate) use stages::Stage;
pub(crate) use stages::{BoundPlugin, Fallback};
pub use stages::{PluginFilter, PluginPicker, PluginScorer};

// A stage's name lives as long as the process: the trace and the shed carry
// `&'static str` names.
pub(crate) use crate::names::intern;
use crate::{
    document::spec::{Needs, MAX_BLOCK_HASHES},
    engine::view::RequestFacts,
};

/// The block hashes of a request a plugin's view carries, whatever its
/// kind: the request's where the plugin's row declares a request's hashes
/// ([`Needs::request_hashes`]), the first [`MAX_BLOCK_HASHES`] of a longer
/// list (the contracts' range), and none where the row declares neither
/// form, whatever the other stages of the generation read. So the row and
/// the contract bound what a plugin is handed, and a prompt's length sizes
/// no view of a plugin that reads none of it.
pub(crate) fn shown_hashes<'a>(facts: &RequestFacts<'a>, needs: Needs) -> &'a [u64] {
    if !needs.request_hashes() {
        return &[];
    }
    let most = usize::try_from(MAX_BLOCK_HASHES).unwrap_or(usize::MAX);
    facts.block_hashes.get(..most).unwrap_or(facts.block_hashes)
}

/// The request's size a plugin's view carries, whatever its kind: the
/// request's where the plugin's row declares it ([`Needs::prompt_size`]),
/// and zero where it does not, on every path. A host that sends a request
/// on without reading its prompt states a size of zero where no stage
/// declares the size. A plugin shown the size undeclared would read the
/// true size of a request its host read and zero of one it sent on: it
/// would pass its tests and misplace large bodies in service. Shown zero
/// always, it reads zero in its first test.
pub(crate) fn shown_size(facts: &RequestFacts<'_>, needs: Needs) -> u32 {
    if needs.prompt_size {
        facts.prompt_size()
    } else {
        0
    }
}
