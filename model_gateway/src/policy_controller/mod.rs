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
//! a host over the feed's holder, and starts the host's timer, which runs
//! the door's clock and shows its counts as series, on no request's path
//! ([`upkeep`]). A regular HTTP router with that host asks
//! it to place typed requests: for the model a request names, or for the one
//! primary model the router's workers serve where it names none. A regular
//! gRPC router and a gRPC pair router ask it in their selection stage. An
//! HTTP pair router asks it for each attempt's pair, through prefill
//! admission. A pair router does where this gateway decides pairs in its
//! own process: the document in force plans pairs and the gateway follows
//! no controller. It mounts a refusing router elsewhere
//! ([`undecided_without()`]). A gRPC encode pair router mounts a refusing
//! router under its usual id ([`undecided()`]) whenever the store started,
//! with or without a controller's address. The routes that pick a
//! worker by no routing policy serve as under the classic policies: a
//! provider's router, a realtime session and a proxied GET. A local worker's
//! registration writes its row at the feed's door. With the setting off, the
//! context holds neither the store, its feed nor its host.
//!
//! The document in force reaches the store through the policy crate's
//! operators' door as YAML: the bytes `--print-policy-spec` prints where
//! the routing configuration means it, and the gateway's YAML of it where
//! it was given. The door stays in the context beside the store. The
//! control plane reads the document in force through it and checks another
//! against it ([`operators`]). Nothing is applied after the start.
//!
//! One site reads the setting, [`start_with`]. Every other site asks what
//! the start left in the context.
//!
//! Under a controller's address the host asks that controller for each
//! decision, over the one link the follower holds. Where the controller
//! gives no answer, the host decides at the door of this process
//! ([`host::Host::linked`]).
//!
//! Under a controller's address a follower keeps the store's worker rows
//! as that controller's watch tells them ([`follower`]). The gateway then
//! discovers, registers and probes no worker of its own, and its worker
//! workflows write no row: a write of its own would set the mirror apart.
//! Each row has its worker object in the registry before the gateway
//! reports a fleet, and the object's status is the row's ([`feeds`]).

use std::sync::Arc;

pub(crate) use smg_policy_controller::spec::PlanName;
use smg_policy_controller::{
    operator::{Format, SpecDoor, SpecRejected},
    store::MemoryStore,
    Catalog,
};
use smg_worker::connection::engine::EngineConnector;
use tracing::info;

use crate::{
    app_context::AppContext,
    config::{validation::ConfigValidator, ConfigError, PolicyDocumentError, RouterConfig},
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
pub(crate) mod operators;
mod undecided;
pub(crate) mod upkeep;

use feed::Feed;
use follower::{Follower, Settings};
use host::Host;
pub(crate) use undecided::refused;
use undecided::Undecided;

/// Why the controller's store did not start.
#[derive(Debug, thiserror::Error)]
pub(crate) enum StartError {
    /// A policy document that would not run as it was given: one beside
    /// the setting off, or a path whose document was never loaded. The
    /// validator's two refusals, in its words.
    #[error(transparent)]
    Unrun(#[from] ConfigError),
    /// The routing configuration holds a setting no policy document says, or
    /// one that disagrees with the document a user gave.
    #[error(transparent)]
    Document(#[from] PolicyDocumentError),
    /// The document did not write as the YAML the door reads.
    #[error("the policy document did not write as YAML: {0}")]
    Unwritten(#[from] serde_yaml::Error),
    /// The door refuses the document: this build does not run a part of
    /// it, or its view reads a header the gateway keeps to itself.
    #[error(transparent)]
    Spec(#[from] SpecRejected),
}

/// What a start leaves the process: the controller's store, and the
/// operators' door over it, which holds the start's document as written.
pub(crate) struct Started {
    pub(crate) store: Arc<MemoryStore>,
    pub(crate) door: Arc<SpecDoor>,
}

/// [`start_with`] as the stock gateway starts: no stages of a host. The
/// cases start their stores through it.
#[cfg(test)]
pub(crate) fn start(config: &RouterConfig) -> Result<Option<Arc<MemoryStore>>, StartError> {
    let started = start_with(config, None)?;
    Ok(started.map(|started| started.store))
}

/// The controller's store for this process, holding the policy document in
/// force ([`RouterConfig::policy_document_in_force`]) as its one generation.
/// `None` where the classic policies decide: nothing is constructed then,
/// and no document is read. `None` too where every router of the gateway is
/// a provider's and neither a document nor a controller's address is given:
/// the setting moves nothing there, and one log line says so.
///
/// A host may build its context from a configuration it never validated,
/// so the validator's two refusals of a document that would not run stand
/// here as well ([`ConfigValidator::validate_policy_document`]): a document
/// given beside the setting off, and a path whose document was never
/// loaded, stop the start in the validator's words.
///
/// `stages` is the catalog of a host's own stages. The stock gateway hands
/// none, and the door then checks over the stock stages alone.
///
/// The start returns the store with the door that applied the document
/// ([`Started`]). The door is handed the document as YAML by the one
/// function `--print-policy-spec` checks its own through ([`hosted`]), and
/// it checks a later dry run over the same stages. A configuration the
/// document cannot say, a routing setting that disagrees with a given
/// document, and a document that check refuses, stop the start by name.
/// None of them is served by the classic policies in silence.
pub(crate) fn start_with(
    config: &RouterConfig,
    stages: Option<Catalog>,
) -> Result<Option<Started>, StartError> {
    ConfigValidator::validate_policy_document(config)?;
    if !config.policy_controller {
        return Ok(None);
    }
    // A gateway whose every router is a provider's places its requests by
    // no routing policy, so no document says it and the classic line serves
    // it. A document given beside it has nothing to run and a controller's
    // address nothing to supply: the read below refuses each by name.
    if let (Some(mode), None, None) = (
        config.provider_alone(),
        &config.policy_spec,
        &config.policy_controller_address,
    ) {
        info!(
            "The policy controller is picked and moves nothing here: the {mode} provider mode places its requests by no routing policy"
        );
        return Ok(None);
    }
    let document = config.policy_document_in_force()?;
    let stages = stages.unwrap_or_else(Catalog::builtin);
    // A given document may hold what its YAML does not say, a `kind` line.
    // The typed check refuses it by the door's own word before the door
    // reads the YAML, over the same stages and denied headers.
    document
        .validate_in(&stages, &config.denied_view_headers())
        .map_err(SpecRejected::Invalid)?;
    let written = serde_yaml::to_string(&document)?;
    let started = hosted(config, Some(stages), &written)?;
    info!("The policy controller is picked for this gateway");
    Ok(Some(started))
}

/// A new store and the operators' door over it, holding `written` as the
/// store's one generation: the one check and apply of a document this
/// gateway hosts. The door bounds the YAML's size, reads it back and
/// validates it over `stages`, the stock ones where a host hands none, with
/// the gateway's own headers denied to the view. The start keeps what this
/// returns. `--print-policy-spec` calls it for its refusal alone, so the
/// print and the start never part.
pub(crate) fn hosted(
    config: &RouterConfig,
    stages: Option<Catalog>,
    written: &str,
) -> Result<Started, SpecRejected> {
    let store = Arc::new(MemoryStore::new());
    let denied = config.denied_view_headers().map(str::to_owned);
    let stages = stages.unwrap_or_else(Catalog::builtin);
    let door = SpecDoor::new(Arc::clone(&store), Arc::new(stages), denied.into());
    door.apply(written.as_bytes().to_vec(), Format::Yaml)?;
    Ok(Started {
        store,
        door: Arc::new(door),
    })
}

/// What stands beside a started store in this process.
pub(crate) struct Hosted {
    /// The store's membership feed.
    pub(crate) feed: Feed,
    /// The host the routers ask.
    pub(crate) host: Arc<Host>,
    /// Under a controller's address, the task that holds the link and
    /// writes the store in the door's place.
    pub(crate) follower: Option<Follower>,
}

/// The feed, the host and, under a controller's address, the follower of
/// `store`.
///
/// The three are made together because they are one rule. A gateway under
/// a controller's address probes no worker. Its holder copies each row's
/// status onto the row's object ([`feed::following`]), its follower
/// settles the feed after each event, and its host asks the controller
/// over the link the follower holds ([`Host::linked`]).
/// Elsewhere the door is the store's writer and settles the feed itself,
/// and the host decides at the door of this process.
pub(crate) fn feeds(
    store: &Arc<MemoryStore>,
    registry: &Arc<WorkerRegistry>,
    connector: Arc<dyn EngineConnector>,
    controller: Option<Settings>,
) -> Hosted {
    match controller {
        Some(settings) => {
            let feed = feed::following(store, registry, connector);
            let follower = Follower::start(settings, feed.clone());
            let host = Host::linked(
                Arc::clone(store),
                Arc::clone(&feed.objects),
                follower.linked(),
            );
            Hosted {
                feed,
                host: Arc::new(host),
                follower: Some(follower),
            }
        }
        None => {
            let feed = feed::feed(store, registry, connector);
            let host = Host::new(Arc::clone(store), Arc::clone(&feed.objects));
            Hosted {
                feed,
                host: Arc::new(host),
                follower: None,
            }
        }
    }
}

/// The refusing router for a family the controller does not decide yet:
/// the gRPC encode pair router. It mounts under
/// the family's usual id so the gateway can select it. Returns `None` under
/// the classic policies.
pub(crate) fn undecided(context: &AppContext) -> Option<Box<dyn RouterTrait>> {
    context
        .policy_host
        .is_some()
        .then(|| Box::new(Undecided) as Box<dyn RouterTrait>)
}

/// The refusing router for a family this gateway does not decide in its
/// own process. `None` under the classic policies, and where the gateway
/// decides the family.
///
/// A family is not decided where no binding of the document in force names
/// its `plan`. A request of it is then decided under another plan, and none
/// of its targets commits as the family's: a pair asked under a single's
/// plan answers that no target holds. A gateway of many routers mounts
/// every family over its one document, so a family reaches this where that
/// document binds no plan of it: a document a user supplies, or the routing
/// settings' own where it plans one family.
///
/// Nor is it decided where the gateway follows a controller. Such a request
/// is the controller's to decide, and no ask reaches it for these families
/// yet, so the family refuses by name there as it did.
pub(crate) fn undecided_without(
    context: &AppContext,
    plan: PlanName,
) -> Option<Box<dyn RouterTrait>> {
    let planned = || {
        context
            .router_config
            .policy_document_in_force()
            .is_ok_and(|document| {
                document
                    .bindings
                    .iter()
                    .any(|binding| binding.plan.as_ref() == Some(&plan))
            })
    };
    undecided(context).filter(|_| context.follows_controller() || !planned())
}

#[cfg(test)]
mod tests;
