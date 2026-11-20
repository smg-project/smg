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
//! it to place typed requests: for the model a request names, or for the one
//! primary model the router's workers serve where it names none. A regular
//! gRPC router asks it in its selection stage, and the HTTP pair router asks
//! it for each attempt's pair. A gRPC pair router mounts a refusing router
//! under its usual id ([`undecided()`]) whenever the store started, with or
//! without a controller's address. The routes that pick a
//! worker by no routing policy serve as under the classic policies: a
//! provider's router, a realtime session and a proxied GET. The gateway's
//! workflows leave the feed's door unused, so without an address nothing
//! feeds the store yet. With the setting off, the context holds neither the
//! store, its feed nor its host.
//!
//! One site reads the setting, [`start_with`]. Every other site asks what
//! the start left in the context.
//!
//! Under a controller's address a follower keeps the store's worker rows
//! as that controller's watch tells them ([`follower`]). The gateway then
//! discovers, registers and probes no worker of its own. Each row has its
//! worker object in the registry before the gateway reports a fleet, and
//! the object's status is the row's ([`feeds`]).

use std::sync::Arc;

use smg_policy_controller::{
    store::{MemoryStore, StaleGeneration},
    Catalog, SpecError,
};
use smg_worker::connection::engine::EngineConnector;
use tracing::info;

use crate::{
    app_context::AppContext,
    config::{PolicyDocumentError, RouterConfig},
    routers::{common::snapshot::WorkerRegistry, RouterTrait},
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

use feed::Feed;
use follower::{Follower, Settings};
pub(crate) use undecided::refused;
use undecided::Undecided;

/// Why the controller's store did not start.
#[derive(Debug, thiserror::Error)]
pub(crate) enum StartError {
    /// The routing configuration holds a setting no policy document says, or
    /// one that disagrees with the document a user gave.
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

/// [`start_with`] as the stock gateway starts: no stages of a host. The
/// cases start their stores through it.
#[cfg(test)]
pub(crate) fn start(config: &RouterConfig) -> Result<Option<Arc<MemoryStore>>, StartError> {
    start_with(config, None)
}

/// The controller's store for this process, holding the policy document in
/// force ([`RouterConfig::policy_document_in_force`]) as its one generation.
/// `None` where the classic policies decide: nothing is constructed then,
/// and no document is read.
///
/// `stages` is the catalog of a host's own stages. The stock gateway hands
/// none, and the document is then checked over the stock stages alone.
///
/// A configuration the document cannot say, a routing setting that disagrees
/// with a given document, and a document the gateway's check of a hosted
/// document refuses ([`RouterConfig::hosted_policy`]), stop the start by
/// name. None of them is served by the classic policies in silence.
pub(crate) fn start_with(
    config: &RouterConfig,
    stages: Option<Catalog>,
) -> Result<Option<Arc<MemoryStore>>, StartError> {
    if !config.policy_controller {
        return Ok(None);
    }
    let document = config.policy_document_in_force()?;
    let generation = config.hosted_policy_in(&document, stages.as_ref())?;
    let store = MemoryStore::new();
    store.apply(generation)?;
    info!("The policy controller is picked for this gateway");
    Ok(Some(Arc::new(store)))
}

/// The store's membership feed and, under a controller's address, the
/// follower that writes the store in the door's place.
///
/// The two are made together because they are one rule. A gateway that
/// follows a controller probes no worker, so its holder copies each row's
/// status onto the row's object ([`feed::following`]), and its follower
/// settles the feed after each event. Elsewhere the door is the store's
/// writer, and it settles the feed itself.
pub(crate) fn feeds(
    store: &Arc<MemoryStore>,
    registry: &Arc<WorkerRegistry>,
    connector: Arc<dyn EngineConnector>,
    controller: Option<Settings>,
) -> (Feed, Option<Follower>) {
    match controller {
        Some(settings) => {
            let feed = feed::following(store, registry, connector);
            let follower = Follower::start(settings, feed.clone());
            (feed, Some(follower))
        }
        None => (feed::feed(store, registry, connector), None),
    }
}

/// The refusing router for a family the controller does not decide yet:
/// the gRPC pair routers, which need pair placement. It mounts under the
/// family's usual id so the gateway can select it. Returns `None` under the
/// classic policies.
pub(crate) fn undecided(context: &AppContext) -> Option<Box<dyn RouterTrait>> {
    context
        .policy_host
        .is_some()
        .then(|| Box::new(Undecided) as Box<dyn RouterTrait>)
}

#[cfg(test)]
mod tests;
