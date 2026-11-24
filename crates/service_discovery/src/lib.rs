//! Worker service discovery.
//!
//! Split along responsibility boundaries so a second discovery source can be
//! added without touching the reconciliation logic:
//!
//! - `kubernetes` owns the Kubernetes client, reflector lifecycle, and the
//!   Pod-to-desired-worker conversion.
//! - `file` rereads a JSON manifest of workers when it changes, and on a
//!   fixed interval.
//! - `fs_watch` is the watch on a directory that tells it of a change. The
//!   watch stands in this crate because the file provider needs it; it is
//!   public so that a host's own file readers share it and keep no copy.
//! - `slurm` polls Slurm for the running jobs whose names map to worker
//!   types, and places a worker on each of their nodes.
//! - `consul` watches a Consul service with blocking queries, and makes a
//!   worker of each instance whose health checks pass.
//! - `snapshot` keeps the last good snapshot the file, Slurm and Consul
//!   providers read, which every pass reconciles, and logs what a read left
//!   out.
//! - `provider` defines the contract every provider produces: a
//!   `provider::DiscoveredWorker` per worker, plus the provider's kind.
//! - `reconciler` owns ownership, the registry diff and what is asked of the
//!   host's job queue. It sees only that contract — never a `Pod`, a Pod UID,
//!   or which provider is running beyond its kind.
//! - `host` defines what the providers ask of the process that hosts them:
//!   its registry, and its job queue.
//! - `config` holds the settings a host reads for each provider, and the
//!   error a provider refuses settings with.
//! - `series` holds the metric series the providers and the reconciler
//!   record.
//! - `runtime` checks configuration with `validation`, resolves it into the
//!   selected provider's runtime settings, and starts that provider.
//!
//! Each provider is a feature of the crate (`kubernetes`, `file`, `slurm`,
//! `consul`), and the directory watch is one of its own (`fs-watch`), which
//! `file` names: a host builds what it runs. `runtime` refuses settings for
//! a provider the build does not hold.
//!
//! SMG mesh-router peer discovery is a different concern — it discovers router
//! peers, not inference workers — and lives in the gateway's
//! `mesh::discovery` module.

// A build with no provider keeps the provider contract, the reconciler and the
// series' recorders with nothing to call them: it serves a host that reads the
// settings, the labels and the host trait alone.
#![cfg_attr(
    not(any(
        test,
        feature = "consul",
        feature = "file",
        feature = "kubernetes",
        feature = "slurm"
    )),
    expect(
        dead_code,
        reason = "no provider is built, so nothing calls the contract or the reconciler"
    )
)]

mod config;
#[cfg(feature = "consul")]
mod consul;
#[cfg(feature = "file")]
mod file;
#[cfg(feature = "fs-watch")]
pub mod fs_watch;
mod host;
#[cfg(feature = "kubernetes")]
mod kubernetes;
mod provider;
mod reconciler;
mod runtime;
mod series;
#[cfg(feature = "slurm")]
mod slurm;
#[cfg(any(feature = "consul", feature = "file", feature = "slurm"))]
mod snapshot;
#[cfg(any(test, feature = "test-util"))]
pub mod testing;
mod validation;

pub use config::{
    deserialize_discovery, ConsulDiscoveryConfig, DiscoveryConfig, DiscoveryConfigError,
    DiscoveryConfigResult, FileDiscoveryConfig, KubernetesDiscoveryConfig, SlurmDiscoveryConfig,
    DEFAULT_DISCOVERY_CHECK_INTERVAL_SECS,
};
#[cfg(feature = "consul")]
pub use consul::ConsulProviderConfig;
#[cfg(feature = "file")]
pub use file::{export_manifest, FileProviderConfig, Manifest};
pub use host::DiscoveryHost;
#[cfg(all(feature = "kubernetes", feature = "test-util"))]
pub use kubernetes::start_service_discovery_with_client;
#[cfg(feature = "kubernetes")]
pub use kubernetes::{
    build_watcher_config, matches_selector, pod_ip, pod_is_ready, selector_string, ModelIdSource,
    PodInfo, PodType, ServiceDiscoveryConfig, POD_NAME_LABEL, POD_UID_LABEL,
};
pub use provider::Roles;
pub use reconciler::{
    DISCOVERY_ID_LABEL, DISCOVERY_LABEL_PREFIX, DISCOVERY_PROVIDER_LABEL, DISCOVERY_SPEC_HASH_LABEL,
};
pub use runtime::{start_service_discovery, RuntimeDiscoveryConfig};
pub use series::describe_series;
#[cfg(feature = "slurm")]
pub use slurm::SlurmProviderConfig;
