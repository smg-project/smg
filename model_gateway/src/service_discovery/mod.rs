//! Worker service discovery.
//!
//! Split along responsibility boundaries so a second discovery source can be
//! added without touching the reconciliation logic:
//!
//! - `kubernetes` owns the Kubernetes client, reflector lifecycle, and the
//!   Pod-to-desired-worker conversion.
//! - `file` rereads a JSON manifest of workers when it changes, and on a
//!   fixed interval.
//! - `slurm` polls Slurm for the running jobs whose names map to worker
//!   types, and places a worker on each of their nodes.
//! - `consul` watches a Consul service with blocking queries, and makes a
//!   worker of each instance whose health checks pass.
//! - `snapshot` keeps the last good snapshot the file, Slurm and Consul
//!   providers read, which every pass reconciles, and logs what a read left
//!   out.
//! - `provider` defines the contract every provider produces: a
//!   `provider::DiscoveredWorker` per worker, plus the provider's kind.
//! - `reconciler` owns ownership, the registry diff and the `JobQueue`
//!   submissions. It sees only that contract — never a `Pod`, a Pod UID, or
//!   which provider is running beyond its kind.
//! - `runtime` resolves configuration into the selected provider's runtime
//!   settings, through the one conversion every input surface shares, and
//!   starts that provider.
//!
//! SMG mesh-router peer discovery is a different concern — it discovers router
//! peers, not inference workers — and lives in [`crate::mesh::discovery`].

mod consul;
mod file;
mod kubernetes;
mod provider;
mod reconciler;
mod runtime;
mod slurm;
mod snapshot;
#[cfg(test)]
pub(crate) mod testing;

pub use consul::ConsulProviderConfig;
pub use file::FileProviderConfig;
#[cfg(feature = "test-util")]
pub use kubernetes::start_service_discovery_with_client;
pub use kubernetes::{
    ModelIdSource, PodInfo, PodType, ServiceDiscoveryConfig, POD_NAME_LABEL, POD_UID_LABEL,
};
pub use reconciler::{
    DISCOVERY_ID_LABEL, DISCOVERY_LABEL_PREFIX, DISCOVERY_PROVIDER_LABEL, DISCOVERY_SPEC_HASH_LABEL,
};
pub use runtime::{start_service_discovery, RuntimeDiscoveryConfig};
pub use slurm::SlurmProviderConfig;
