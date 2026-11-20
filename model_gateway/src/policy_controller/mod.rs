//! The policy controller in this process.
//!
//! A request's worker has two possible deciders: the classic policies, and
//! the policy controller's engine over the controller's store. One
//! setting picks a decider for the whole process
//! ([`RouterConfig::policy_controller`]). The classic policies are the
//! default.
//!
//! The setting starts the store with the policy document for the routing
//! configuration. The context holds the membership feed over that store and
//! a host over the feed's holder. A regular HTTP router with that host asks
//! it to place typed requests for a named model. The gateway's workflows
//! leave the feed's door unused, so without a controller's address nothing
//! feeds the store yet. The server answers with a refusing router in the
//! gateway object's place ([`router`]), so its requests never reach the
//! host. With the setting off, the context holds neither the store, its
//! feed nor its host.
//!
//! One site reads the setting, [`start_with`]. Every other site asks what
//! the start left in the context.
//!
//! Under a controller's address a follower keeps the store's worker rows
//! as that controller's watch tells them ([`follower`]). The gateway then
//! discovers, registers and probes no worker of its own.

use std::sync::Arc;

use smg_policy_controller::{
    store::{MemoryStore, StaleGeneration},
    Catalog, PolicySpec, SpecError,
};
use tracing::info;

use crate::{
    app_context::AppContext,
    config::{PolicyDocumentError, RouterConfig},
    routers::RouterTrait,
};

// The follower dials through the controller crate's client, which two
// targets build (the manifest says why). Elsewhere its names stand in its
// place, and startup refuses a controller's address before one is started.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) mod follower;
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
#[path = "follower_absent.rs"]
pub(crate) mod follower;

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the gateway holds the feed for membership writes")
)]
pub(crate) mod feed;
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the rank accessor, explicit release and Held are exercised by tests"
    )
)]
pub(crate) mod host;
mod undecided;

pub(crate) use undecided::refused;
use undecided::Undecided;

/// Why the controller's store did not start.
#[derive(Debug, thiserror::Error)]
pub(crate) enum StartError {
    /// The routing configuration holds a setting no policy document says.
    #[error(transparent)]
    Document(#[from] PolicyDocumentError),
    /// The gateway's check of a document it hosts refuses this one: this
    /// build does not run a part of it, or its view reads a header the
    /// gateway keeps to itself.
    #[error(transparent)]
    Spec(#[from] SpecError),
    /// The store holds a generation that is not below the document's.
    #[error(transparent)]
    Stale(#[from] StaleGeneration),
}

/// [`start_with`] as the stock gateway starts: no stages of a host, and the
/// document the routing configuration means. The cases start their stores
/// through it.
#[cfg(test)]
pub(crate) fn start(config: &RouterConfig) -> Result<Option<Arc<MemoryStore>>, StartError> {
    start_with(config, None, RouterConfig::policy_document)
}

/// The controller's store for this process, holding the policy document the
/// routing configuration means as its one generation. `None` where the
/// classic policies decide: nothing is constructed then, and `document` is
/// not called.
///
/// `stages` is the catalog of a host's own stages. The stock gateway hands
/// none, and the document is then checked over the stock stages alone.
///
/// A configuration the document cannot say, and a document the gateway's
/// check of a hosted document refuses ([`RouterConfig::hosted_policy`]),
/// stop the start by name. Neither is served by the classic policies in
/// silence.
pub(crate) fn start_with(
    config: &RouterConfig,
    stages: Option<Catalog>,
    document: impl FnOnce(&RouterConfig) -> Result<PolicySpec, PolicyDocumentError>,
) -> Result<Option<Arc<MemoryStore>>, StartError> {
    if !config.policy_controller {
        return Ok(None);
    }
    let document = document(config)?;
    let generation = config.hosted_policy_in(&document, stages.as_ref())?;
    let store = MemoryStore::new();
    store.apply(generation)?;
    info!("The policy controller is picked for this gateway: no classic router is built");
    Ok(Some(Arc::new(store)))
}

/// The controller's router for this process: the one router its server
/// answers with. `None` where the classic policies decide.
///
/// It asks what the start left in the context and never reads the setting
/// again, so a router stands where a store started and nowhere else. It
/// refuses each request by name.
pub(crate) fn router(context: &AppContext) -> Option<Arc<dyn RouterTrait>> {
    context
        .policy_store
        .is_some()
        .then(|| Arc::new(Undecided) as Arc<dyn RouterTrait>)
}

#[cfg(test)]
mod tests;
