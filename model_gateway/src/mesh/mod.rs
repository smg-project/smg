//! Gateway-side glue for the v2 mesh: adapters that bridge the
//! typed `MeshKV` namespaces to local registries, plus bootstrap
//! and shutdown wiring added in later steps. [`discovery`] finds the
//! mesh's router peers.

pub mod adapters;
pub mod discovery;
pub mod wiring;

pub use adapters::{RateLimitSyncAdapter, TreeDelta, TreeSyncAdapter, WorkerSyncAdapter};
pub use wiring::MeshAdapters;
