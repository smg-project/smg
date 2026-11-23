//! Shared worker registration steps used by both local and external workflows.
//!
//! These steps are designed to work with any worker type and can be composed
//! into different workflows using the DAG-based workflow engine.

mod activate;
mod register;
mod update_policies;

use std::sync::Arc;

pub use activate::ActivateWorkersStep;
pub(crate) use activate::ACTIVATE_WORKERS_STEP;
#[cfg(test)]
pub(crate) use register::registration_hold;
pub(crate) use register::REGISTER_WORKERS_STEP;
pub use register::{registered_entries, registered_in, RegisterWorkersStep};
use smg_worker::identity::basic::Worker;
pub(crate) use update_policies::start_kv_stream;
pub use update_policies::UpdatePoliciesStep;

/// Type alias for a collection of workers in workflow context.
/// Both local (single/DP-aware) and external (multi-model) workflows
/// use this unified type for consistency.
pub type WorkerList = Vec<Arc<dyn Worker>>;
