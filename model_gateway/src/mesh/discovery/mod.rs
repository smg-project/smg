//! SMG mesh router-peer discovery.
//!
//! Distinct from [`crate::service_discovery`]: this finds *SMG router peers*
//! for the mesh cluster, not inference workers. It is Kubernetes-specific and
//! runs independently of whichever worker discovery provider is selected — or
//! of none at all.

mod kubernetes;

#[cfg(feature = "test-util")]
pub use kubernetes::start_mesh_discovery_with_client;
pub use kubernetes::{start_mesh_discovery, MeshDiscoveryConfig};
