//! Worker service discovery.
//!
//! Split along responsibility boundaries so a second discovery source can be
//! added without touching the reconciliation logic:
//!
//! - [`kubernetes`] owns the Kubernetes client, reflector lifecycle, and the
//!   Pod-to-desired-worker conversion.
//! - [`provider`] defines the contract every provider produces: a
//!   [`provider::DiscoveredWorker`] per worker, plus the provider's kind.
//! - [`reconciler`] owns ownership, the registry diff and the `JobQueue`
//!   submissions. It sees only that contract — never a `Pod`, a Pod UID, or
//!   which provider is running beyond its kind.
//! - [`runtime`] resolves configuration into the selected provider's runtime
//!   settings, through the one conversion every input surface shares, and
//!   starts that provider.
//!
//! SMG mesh-router peer discovery is a different concern — it discovers router
//! peers, not inference workers — and lives in [`crate::mesh_discovery`].

mod kubernetes;
mod provider;
mod reconciler;
mod runtime;
#[cfg(test)]
mod testing;

#[cfg(feature = "test-util")]
pub use kubernetes::start_service_discovery_with_client;
pub use kubernetes::{
    ModelIdSource, PodInfo, PodType, ServiceDiscoveryConfig, POD_NAME_LABEL, POD_UID_LABEL,
};
pub use reconciler::{DISCOVERY_ID_LABEL, DISCOVERY_PROVIDER_LABEL, DISCOVERY_SPEC_HASH_LABEL};
pub use runtime::{start_service_discovery, RuntimeDiscoveryConfig};
