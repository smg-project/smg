//! In-process calls over the policy document and the store it runs on.
//!
//! A host applies its start's document through [`DocumentManager::apply`]. It
//! can read the running document and status and check another document with a
//! dry run. Without a spec store, an enforce and a rollback are refused. A host
//! that runs a document and keeps no store for it checks another against the
//! one it runs with [`dry_run_against`]. The calls run on the caller's thread.

mod answers;
mod document_manager;
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
    ApplyCounts, ApplyOutcome, Document, DryRun, EnforceOutcome, EnforceRefused, NeedsRow, NotHeld,
    OnFile, SpecRejected, Status,
};
#[cfg(all(test, feature = "controller"))]
pub(crate) use document_manager::needs_rows;
#[cfg(all(test, feature = "controller"))]
pub(crate) use document_manager::Validate;
#[cfg(feature = "controller")]
pub(crate) use document_manager::{above, over_bound, Checked, Held, Line};
pub use document_manager::{dry_run_against, DocumentManager};
#[cfg(feature = "controller")]
pub(crate) use document_manager::{DRY_RUNS_WAITING, WRITES_WAITING};
pub use reader::{Format, MAX_DOCUMENT_BYTES};
pub use turn::SerialTurn;
