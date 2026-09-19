//! Worker service discovery.
//!
//! Split along responsibility boundaries so a second discovery source can be
//! added without touching the reconciliation logic:
//!
//! - [`kubernetes`] owns the Kubernetes client, reflector lifecycle, and the
//!   Pod-to-desired-worker conversion.
//! - [`reconciler`] owns the registry diff and the `JobQueue` submissions. It
//!   takes an already-computed desired-worker snapshot and never sees a `Pod`.
//!
//! SMG mesh-router peer discovery is a different concern — it discovers router
//! peers, not inference workers — and lives in [`crate::mesh_discovery`].

mod kubernetes;
mod reconciler;
#[cfg(test)]
mod testing;

#[cfg(feature = "test-util")]
pub use kubernetes::start_service_discovery_with_client;
pub use kubernetes::{
    start_service_discovery, ModelIdSource, PodInfo, PodType, ServiceDiscoveryConfig,
};
pub use reconciler::{POD_NAME_LABEL, POD_UID_LABEL};
