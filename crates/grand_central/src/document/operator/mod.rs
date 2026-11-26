//! In-process calls over the policy document and the store it runs on.
//!
//! A host applies its start's document through [`DocumentManager::apply`]. It
//! can read the running document and status and check another document with a
//! dry run. Without a spec store, an enforce and a rollback are refused. A host
//! that runs a document and keeps no store for it checks another against the
//! one it runs with [`dry_run_against`]. The calls run on the caller's thread.
//!
//! A host that starts from a file names the file's form with [`Format::of`]
//! and reads the file's bytes with [`Format::read`]: the read every source's
//! document passes, an apply's too.

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
#[path = "../../../tests/unit/document/operator/test_support.rs"]
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
pub(crate) use document_manager::{above, Checked, Entered, FollowerTurn, Held, Line};
pub use document_manager::{dry_run_against, DocumentManager};
#[cfg(feature = "controller")]
pub(crate) use document_manager::{DRY_RUNS_WAITING, WRITES_WAITING};
#[cfg(feature = "controller")]
pub(crate) use reader::over_bound;
pub use reader::{Format, MAX_DOCUMENT_BYTES};
pub use turn::SerialTurn;
