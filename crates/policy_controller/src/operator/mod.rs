//! In-process calls over the policy document and the store it runs on.
//!
//! A host applies its start's document through [`SpecDoor::apply`]. It can
//! read the running document and status and check another document with a
//! dry run. Without a spec store, an enforce and a rollback are refused.
//! The calls run on the caller's thread.

mod answers;
mod door;
pub(crate) mod reader;
pub mod snapshot;
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) mod spec_store;
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
#[path = "no_spec_store.rs"]
pub(crate) mod spec_store;
#[cfg(test)]
pub(crate) mod test_support;
mod turn;

pub use answers::{
    Applied, Counts, Document, DryRun, Enforced, NeedsRow, NotHeld, OnFile, SpecRejected, Status,
    Unanswered,
};
#[cfg(all(test, feature = "controller"))]
pub(crate) use door::needs_rows;
pub use door::SpecDoor;
#[cfg(all(test, feature = "controller"))]
pub(crate) use door::Validate;
#[cfg(feature = "controller")]
pub(crate) use door::{above, over_bound, Checked, Held, Line};
#[cfg(feature = "controller")]
pub(crate) use door::{DRY_RUNS_WAITING, WRITES_WAITING};
pub use reader::{Format, MAX_DOCUMENT_BYTES};
pub use turn::Turn;
