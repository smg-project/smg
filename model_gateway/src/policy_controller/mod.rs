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
//! a policy engine over the feed's holder, and starts the engine's timer, which
//! runs the door's clock and shows its counts as series, on no request's path
//! ([`upkeep`]). A regular HTTP router with that policy engine asks
//! it to place typed requests: for the model a request names, or for the one
//! primary model the router's workers serve where it names none. A regular
//! gRPC router asks it in its selection stage. A pair router asks it too
//! where the document in force plans pairs and the gateway follows no
//! controller: a gRPC pair router in its selection stage, an HTTP pair
//! router for each attempt's pair, through prefill admission. Elsewhere
//! under that engine a pair family mounts a refusing router under its usual
//! id ([`undecided_without()`]), and the gRPC encode pair router always
//! does ([`undecided()`]). The routes that pick a
//! worker by no routing policy serve as under the classic policies: a
//! provider's router, a realtime session and a proxied GET. With the
//! setting off, the context holds neither the store, its feed nor its
//! policy engine.
//!
//! The document in force reaches the store through the policy crate's
//! operators' door as YAML: the bytes `--print-policy-spec` prints where
//! the routing configuration means it, and the gateway's YAML of it where
//! it was given. The door stays in the context beside the store. The
//! control plane reads the document in force through it and checks another
//! against it ([`operators`]). Nothing is applied after the start.
//!
//! One site reads the setting, the start. Every other site asks what the
//! start left in the context.
//!
//! With no controller's address the gateway decides in its own process
//! ([`start_in_process`]). The start then builds the policy engine of the
//! document in force and nothing else of what is described above: no
//! store, no feed, no door and no engine object of the gateway's own. The
//! gateway's worker registry, its health checks, its load monitor and its
//! worker workflows are the classic ones, and the classic policy registry
//! asks the engine where it asks a classic policy
//! ([`crate::policies::EnginePolicy`]). The control plane's policy routes
//! answer there from the engine's document, whose YAML the start holds to
//! it ([`checked`]), and from the worker registry ([`operators`]). The
//! store and what stands on it are a gateway's that follows a controller
//! ([`start_with`]).
//!
//! Under a controller's address a follower keeps the store's worker rows
//! as that controller's watch tells them ([`follower`]). The gateway then
//! discovers, registers and probes no worker of its own, and its worker
//! workflows write no row: a write of its own would set the mirror apart.
//! Each row has its worker object in the registry before the gateway
//! reports a fleet, and the object's status is the row's ([`feeds`]).

use std::{sync::Arc, time::SystemTime};

pub(crate) use smg_policy_controller::document::PlanName;
use smg_policy_controller::{
    document::{
        operator::{dry_run_against, DocumentManager, Format, SpecRejected},
        AppliedSpec, IndexKind, Needs, PolicySpec, RoutingSettingsError,
    },
    stages::StageRegistry,
    store::MemoryStore,
};
use smg_worker::connection::engine::EngineConnector;
use tracing::info;

use crate::{
    app_context::AppContext,
    config::{validation::ConfigValidator, ConfigError, RouterConfig},
    policies::DocumentOrigin,
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
    expect(
        dead_code,
        reason = "the rank accessor, explicit release and Held are exercised by tests"
    )
)]
pub(crate) mod engine;
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos")),
    expect(
        dead_code,
        reason = "the follower reads the feed, and none runs on this target"
    )
)]
pub(crate) mod feed;
pub(crate) mod operators;
mod undecided;
pub(crate) mod upkeep;

use feed::Feed;
use follower::{Follower, Settings};
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
    Document(#[from] RoutingSettingsError),
    /// The document did not write as the YAML the door reads.
    #[error("the policy document did not write as YAML: {0}")]
    Unwritten(#[from] serde_yaml::Error),
    /// The door refuses the document: this build does not run a part of
    /// it, or its view reads a header the gateway keeps to itself.
    #[error(transparent)]
    Spec(#[from] SpecRejected),
    /// The document's YAML does not read back as the document it was
    /// written from, so the control plane would tell another text than the
    /// one that decides.
    #[error("the policy document's YAML does not read back as the document it writes: {0}")]
    NotReadBack(String),
    /// The document its operator supplied holds a place the engine does
    /// not decide in this process yet. Nothing else would decide that
    /// place, so the start names it: the profile, the stage of it that
    /// asks more than one pick, and the binding and phase that reach it.
    #[error(
        "the policy document's profile `{profile}` is not decided in this process yet: \
         it holds {stage} (binding `{model}`, phase {phase})"
    )]
    NotAlone {
        profile: String,
        stage: String,
        model: String,
        phase: &'static str,
    },
    /// The document its operator supplied states a rule that nothing in
    /// this process would run ([`rule_not_run`]): the start names the
    /// binding, by its place in the document and its model pattern, and
    /// the field.
    #[error(
        "the policy document's binding {at} (`{model}`) states `{field}`, \
         which is not run in this process yet: {because}"
    )]
    RuleNotRun {
        at: usize,
        model: String,
        field: &'static str,
        because: &'static str,
    },
    /// The document its operator supplied states a rule of its own, apart
    /// from its bindings, that nothing in this process would run
    /// ([`rule_not_run`]): the start names the field, and says what its
    /// operator can do.
    #[error(
        "the policy document states `{field}`, which is not run in this process yet: {because}. \
         {instead}, or run the gateway under a controller's address"
    )]
    FieldNotRun {
        field: &'static str,
        because: &'static str,
        instead: String,
    },
    /// The document keeps a model's prefix index as a placement map by a
    /// prompt's head (`index.<key>.kind: hash`), which nothing in this
    /// process runs ([`hash_index`]): the start names the index's key and
    /// says which setting starts.
    #[error(
        "the policy document states `index.{model}.kind: hash`, which is not run in this \
         process yet: no placement map by a prompt's head is kept here. Start with \
         `--cache-index tree` or leave the flag out (in a document, state `kind: tree` \
         or remove the field)"
    )]
    HashIndexNotRun { model: String },
}

/// What a start in this process leaves it ([`start_in_process`]): the
/// policy engine of the document in force, which the classic policy
/// registry's policies ask, and what the control plane's policy routes
/// read of the document where no store holds it.
#[derive(Clone)]
pub(crate) struct InProcess {
    /// The engine of the document in force, under the number the
    /// operators' door would give it: its own, and 1 where it names none.
    pub(crate) engine: Arc<smg_policy_controller::engine::PolicyEngine>,
    /// How the document came to be.
    pub(crate) origin: DocumentOrigin,
    /// The document as the start wrote it, in YAML: the bytes
    /// `--print-policy-spec` prints where the routing settings mean it.
    pub(crate) written: String,
    /// When the start built the engine: the time the document took effect.
    pub(crate) applied: SystemTime,
    /// The stages the document was validated over, a host's own among
    /// them: a check of another document against this one reads them.
    pub(crate) stages: Arc<StageRegistry>,
}

impl InProcess {
    /// Whether the routers ask the policy engine for the whole decision of
    /// each request, behind the arms they ask a followed controller's
    /// engine through, and no classic policy places one.
    ///
    /// They do where the engine decides every request of the document in
    /// force whole today: single requests alone, under no sessions rule,
    /// each place picked by its picker alone or decided by a profile that
    /// asks nothing this process does not feed a whole decision
    /// ([`fed`]). Under any other document the
    /// routers keep their classic placement pass, whose policies ask the
    /// engine for the pick. The set grows with what the engine decides
    /// whole, and this question goes when it decides every document.
    pub(crate) fn decides_whole(&self) -> bool {
        self.engine.spec().engine().single_and_fed(fed())
    }
}

/// What this process feeds a whole decision of the engine from its own
/// data, beyond the worker objects' live state.
///
/// The routers hash a request's prompt, by its tokens or its text, where
/// the document's stages read the hashes.
/// They count the prompt's size where they read the prompt: a typed route
/// reads it for each request it asks the engine about, and a route that
/// sends a body on unread must read it where a stage declares the size
/// ([`Needs::prompt_size`]), as the classic arm buffers a body for a policy
/// that needs its text.
/// The engine reads the cache-event index this process keeps, and keeps
/// its own record of the prefixes it sent: together they name a request's
/// holders and a pool's warming workers. Each worker's load report is read
/// in the load monitor's snapshot, lent once an ask
/// ([`engine::PolicyEngine::read_loads`]), and a worker's count of session
/// keys in flight on the worker object, with the rest of its record.
///
/// Not fed yet, so a profile that reads one keeps the classic placement
/// pass: a hash ring, a header of the request, a named signal's cells, and
/// the media a request carries. No route states a request's media counts
/// to the engine: a stage that declared them would read none for every
/// request, with no error.
fn fed() -> Needs {
    Needs {
        token_hashes: true,
        text_hashes: true,
        prompt_size: true,
        loads: true,
        events: true,
        sent_prefixes: true,
        active_sessions: true,
        affinity: true,
        ..Needs::default()
    }
}

/// What a start leaves the process: the controller's store, and the
/// operators' door over it, which holds the start's document as written.
pub(crate) struct Started {
    pub(crate) store: Arc<MemoryStore>,
    pub(crate) door: Arc<DocumentManager>,
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
/// ([`Started`]). The door is handed the document as YAML ([`hosted`]), and
/// it checks a later dry run over the same stages. A configuration the
/// document cannot say, a routing setting that disagrees with a given
/// document, and a document that check refuses, stop the start by name.
/// None of them is served by the classic policies in silence.
pub(crate) fn start_with(
    config: &RouterConfig,
    stages: Option<StageRegistry>,
) -> Result<Option<Started>, StartError> {
    let Some(document) = document_in_force(config)? else {
        return Ok(None);
    };
    let stages = stages.unwrap_or_else(StageRegistry::builtin);
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

/// The start of a gateway that decides in its own process: the setting on
/// and no controller's address. It reads the setting and the document in
/// force as [`start_with`] does and refuses what that start refuses, in
/// the same words. It then builds the policy engine of the document, over
/// `stages` where a host hands its own, and nothing else: no store, no
/// door, no feed. `None` where the setting is off or moves nothing.
///
/// A document the routing settings mean is decided by the engine where it
/// picks the place's profile by itself, and by the classic policy of the
/// setting elsewhere. That holds for the settings' own document handed
/// back as a supplied one too: it says what they say. Any other document
/// its operator supplied has no setting behind a place, so a place of it
/// the engine does not pick by itself is refused here
/// ([`StartError::NotAlone`]), and so is a rule that nothing in this
/// process would run ([`rule_not_run`]). So neither of the two meets a
/// document the gateway wrote, beside the settings that wrote it.
///
/// Every check and refusal is [`engine_for`]'s, which answers for a
/// document the gateway does not run as well ([`would_start`]). The start
/// adds its one log line.
pub(crate) fn start_in_process(
    config: &RouterConfig,
    stages: Option<StageRegistry>,
) -> Result<Option<InProcess>, StartError> {
    let started = engine_for(config, stages.map(Arc::new))?;
    if started.is_some() {
        info!(
            "The policy controller is picked for this gateway and runs in this process, over the gateway's own worker data: \
             the policy engine decides each policy whose line says so"
        );
    }
    Ok(started)
}

/// Whether a start of this gateway in its own process would run
/// `document`, were it the document the gateway's operator supplies in the
/// place of the one it runs. `config` is the gateway's own configuration
/// and `stages` the ones its start validated over
/// ([`InProcess::stages`]). The answer is the start's own
/// ([`engine_for`]): a refusal comes in the start's words, and a document
/// this takes is one the gateway starts under.
///
/// The control plane's dry run of a document asks it where the gateway
/// decides in its own process, so a document that passes the dry run does
/// not stop the next start.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the control plane's dry run of a document asks it"
    )
)]
pub(crate) fn would_start(
    config: &RouterConfig,
    stages: &Arc<StageRegistry>,
    document: &PolicySpec,
) -> Result<(), StartError> {
    let supplied = RouterConfig {
        policy_spec: Some(document.clone()),
        ..config.clone()
    };
    engine_for(&supplied, Some(Arc::clone(stages))).map(drop)
}

/// The policy engine a gateway of `config` decides by in its own process,
/// with what its start keeps beside it: the one home of every check and
/// refusal of a start there. `stages` are a host's own, and the stock ones
/// where it hands none. `None` where the setting is off or moves nothing.
/// It says nothing of a document in the log and leaves nothing behind, so
/// it answers for a document the gateway does not run too.
fn engine_for(
    config: &RouterConfig,
    stages: Option<Arc<StageRegistry>>,
) -> Result<Option<InProcess>, StartError> {
    // The document comes under the number a gateway's first document
    // takes: its own, and 1 where it names none (the reader's rule).
    let Some(document) = document_in_force(config)? else {
        return Ok(None);
    };
    let stages = stages.unwrap_or_else(|| Arc::new(StageRegistry::builtin()));
    let written = serde_yaml::to_string(&document)?;
    let spec = checked(config, &stages, &document, &written)?;
    hash_index(&document)?;
    let engine = Arc::new(smg_policy_controller::engine::PolicyEngine::new(spec));
    // The document is its operator's own where it says another thing than
    // the routing settings mean. The number is no part of what it says.
    let mut settings = config.policy_document()?;
    settings.generation = document.generation;
    let origin = if settings == document {
        DocumentOrigin::Settings
    } else {
        DocumentOrigin::Supplied
    };
    if origin == DocumentOrigin::Supplied {
        // A place the engine does not pick alone is decided by its whole
        // pass where the routers ask the engine for every request of the
        // document, and by nothing where they do not.
        let whole = engine.spec().engine().single_and_fed(fed());
        if let Some(place) = engine.not_alone().filter(|_| !whole) {
            return Err(StartError::NotAlone {
                profile: place.profile.to_owned(),
                stage: place.stage.to_string(),
                model: place.model.to_owned(),
                phase: place.phase.name(),
            });
        }
        rule_not_run(config, &settings, &document)?;
    }
    Ok(Some(InProcess {
        engine,
        origin,
        written,
        applied: SystemTime::now(),
        stages,
    }))
}

/// Refuses `document` by its first prefix index kept as a placement map
/// (`index.<key>.kind: hash`), whoever wrote it: the routing settings
/// (`--cache-index hash`) or its operator.
///
/// The engine keeps the prefixes a model's workers were sent as a tree,
/// with no lifetime and no boundary, and it keeps that record for the tree
/// kind alone. Under the hash kind nothing here would keep a placement:
/// every request would read as held by no worker, and no line would say
/// so. The classic policy runs the kind, so the refusal says which setting
/// starts.
fn hash_index(document: &PolicySpec) -> Result<(), StartError> {
    let kept_as_map = document
        .index
        .iter()
        .find(|(_, index)| index.kind == IndexKind::Hash);
    match kept_as_map {
        Some((model, _)) => Err(StartError::HashIndexNotRun {
            model: model.clone(),
        }),
        None => Ok(()),
    }
}

/// Refuses a supplied `document` by its first rule that nothing in this
/// process would run. `settings` is the document the routing settings
/// mean.
///
/// The engine is asked for picks, and the gateway's own placement holds
/// every other rule of a request. It keeps sessions, reads a request's
/// routing key and judges a worker's overload as the routing settings say.
/// It runs no shadow profile. Its registry keeps one policy for each side
/// of a pair, for every model. So a binding that states `overload`,
/// `routing_key` or `sessions` otherwise than the settings' own document
/// does, a binding that names a `shadow`, a pair binding that names some
/// models and a second pair binding would each start with a rule no code
/// runs, and the gateway would run the settings' value in silence. So
/// would a binding of some models that covers a side's phase, in a
/// gateway that places pairs: the side's policy places those models'
/// pair requests, whatever that binding says. And so would a pair binding
/// that covers the single phase, in a gateway that places single
/// requests. It plans none and takes them from every binding behind it,
/// so the classic policy would place them and no line would say so: it
/// names its sides' phases alone.
///
/// The document's own rules stand beside its bindings'. The placement
/// pairs a prefill with a decode by the routing settings' mode and picks a
/// pair's decode with its prefill, and nothing here keeps a decision for
/// explain. So a pair plan's `pairing` other than the settings' mode, a
/// pair plan's `decode: after-prefill` and an `explain.sample` other than
/// zero are refused by the field ([`StartError::FieldNotRun`]).
fn rule_not_run(
    config: &RouterConfig,
    settings: &PolicySpec,
    document: &PolicySpec,
) -> Result<(), StartError> {
    use smg_policy_controller::{
        document::{BindingSpec, DecodePick, Pairing, PlanName},
        engine::Phase,
    };

    use crate::config::{PdPairingMode, RoutingMode};

    const BY_SETTINGS: &str = "the gateway's placement runs the routing settings' own";
    const ONE_A_SIDE: &str = "one policy places each side of a pair, for every model";
    const SINGLE_ALONE: &str = "one policy places each side of a pair, for every model: \
                                a binding of some models holds single requests alone (`phase: [single]`)";
    const PAIR_SIDES: &str = "this gateway places single requests and a pair binding plans none: \
                              a pair binding holds its sides' phases alone (`phase: [prefill, decode]`)";
    const EPD_SIDES: &str = "this gateway places single requests and a pair binding plans none: \
                             a pair binding holds its sides' phases alone (`phase: [encode, prefill, decode]`)";
    let pair_mode = matches!(
        config.mode,
        RoutingMode::PrefillDecode { .. } | RoutingMode::EncodePrefillDecode { .. }
    );
    // A pair mode, and a gateway of many routers, which mounts every family.
    let places_pairs = config.enable_igw || pair_mode;
    // Every other mode, and a gateway of many routers.
    let places_singles = config.enable_igw || !pair_mode;
    // What the routing settings say of a binding's own rules: the one
    // binding of their document.
    let run = settings.bindings.first().cloned().unwrap_or_default();
    let mut pair_planned = false;
    for (at, binding) in document.bindings.iter().enumerate() {
        let refused = |field, because| StartError::RuleNotRun {
            at: at + 1,
            model: binding.matcher.model.clone(),
            field,
            because,
        };
        // An absent overload block is the block's defaults.
        let overload = |binding: &BindingSpec| binding.overload.clone().unwrap_or_default();
        if overload(binding) != overload(&run) {
            return Err(refused("overload", BY_SETTINGS));
        }
        if binding.routing_key != run.routing_key {
            return Err(refused("routing_key", BY_SETTINGS));
        }
        if binding.sessions != run.sessions {
            return Err(refused("sessions", BY_SETTINGS));
        }
        if binding.shadow.is_some() {
            return Err(refused("shadow", "no shadow profile runs here"));
        }
        if matches!(binding.plan, Some(PlanName::PdPair | PlanName::Epd)) {
            if binding.matcher.model != "*" {
                return Err(refused("match.model", ONE_A_SIDE));
            }
            if pair_planned {
                return Err(refused("plan", "an earlier binding plans the pair"));
            }
            // A list that names no phase covers every phase.
            let phases = &binding.matcher.phase;
            if places_singles && (phases.is_empty() || phases.contains(&Phase::Single)) {
                let sides = if binding.plan == Some(PlanName::Epd) {
                    EPD_SIDES
                } else {
                    PAIR_SIDES
                };
                return Err(refused("match.phase", sides));
            }
            pair_planned = true;
        } else if places_pairs
            && binding.matcher.model != "*"
            && binding.matcher.phase != [Phase::Single]
        {
            return Err(refused("match.phase", SINGLE_ALONE));
        }
    }
    // The document's own rules. A pair plan's rules have a request only
    // where a binding plans the pair. Each refusal says what the operator
    // can do: a field left out reads as its default.
    const REMOVE: &str = "Remove the field";
    let stated = |field, because, instead: String| StartError::FieldNotRun {
        field,
        because,
        instead,
    };
    if pair_planned {
        let mode = config.pd_pairing_mode;
        let by_settings = match mode {
            PdPairingMode::Off => Pairing::Off,
            PdPairingMode::Lenient => Pairing::Lenient,
            PdPairingMode::Strict => Pairing::Strict,
        };
        if document.plans.pd_pair.pairing != by_settings {
            let instead = if by_settings == Pairing::default() {
                REMOVE.to_owned()
            } else {
                format!("State `{}`, the routing settings' mode", mode.as_str())
            };
            return Err(stated(
                "plans.pd-pair.pairing",
                "the gateway's placement pairs by the routing settings' mode",
                instead,
            ));
        }
        if document.plans.pd_pair.decode != DecodePick::WithPrefill {
            return Err(stated(
                "plans.pd-pair.decode",
                "a pair's decode is picked with its prefill",
                REMOVE.to_owned(),
            ));
        }
    }
    if document.explain.sample != 0 {
        return Err(stated(
            "explain.sample",
            "no decision is kept for explain",
            REMOVE.to_owned(),
        ));
    }
    Ok(())
}

/// `document` as a gateway that decides in its own process runs it, with
/// its YAML `written` held to it. The document is validated over `stages`
/// with the gateway's own headers denied to its view. A dry run of the text
/// against the validated document must then answer that document's own
/// number and digest ([`dry_run_against`]). So the text is inside the bound
/// a dry run's body is held to, the policy crate's reader reads it, and the
/// control plane's read tells the document that decides: a dry run of that
/// text under its own `ETag` answers no change.
///
/// `--print-policy-spec` checks its document here too, so the print and
/// this start never part.
pub(crate) fn checked(
    config: &RouterConfig,
    stages: &StageRegistry,
    document: &PolicySpec,
    written: &str,
) -> Result<AppliedSpec, StartError> {
    let denied = config.denied_view_headers();
    let spec =
        AppliedSpec::validate_in(document, stages, &denied).map_err(SpecRejected::Invalid)?;
    let read = dry_run_against(&spec, stages, &denied, "", written.as_bytes(), Format::Yaml)
        .map_err(|unanswered| StartError::NotReadBack(unanswered.to_string()))?;
    match read.would_run? {
        (generation, digest) if (generation, Some(digest)) == read.standing => Ok(spec),
        (generation, digest) => Err(StartError::NotReadBack(format!(
            "it reads as generation {generation} with digest {digest}"
        ))),
    }
}

/// The document a start runs. `None` where the setting is off, and where
/// it is on and moves nothing. The one site that reads the setting.
fn document_in_force(config: &RouterConfig) -> Result<Option<PolicySpec>, StartError> {
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
    Ok(Some(config.policy_document_in_force()?))
}

/// A new store and the operators' door over it, holding `written` as the
/// store's one generation: the one check and apply of a document this
/// gateway hosts. The door bounds the YAML's size, reads it back and
/// validates it over `stages`, the stock ones where a host hands none, with
/// the gateway's own headers denied to the view. The start keeps what this
/// returns.
pub(crate) fn hosted(
    config: &RouterConfig,
    stages: Option<StageRegistry>,
    written: &str,
) -> Result<Started, SpecRejected> {
    let store = Arc::new(MemoryStore::new());
    let denied = config.denied_view_headers().map(str::to_owned);
    let stages = stages.unwrap_or_else(StageRegistry::builtin);
    let door = DocumentManager::new(Arc::clone(&store), Arc::new(stages), denied.into());
    door.apply(written.as_bytes().to_vec(), Format::Yaml)?;
    Ok(Started {
        store,
        door: Arc::new(door),
    })
}

/// The store's membership feed, and the follower that writes the store as
/// the controller at `controller` tells it.
///
/// The two are made together because they are one rule. A gateway that
/// follows a controller probes no worker, so its holder copies each row's
/// status onto the row's object ([`feed::following`]), and its follower
/// settles the feed after each event.
pub(crate) fn feeds(
    store: &Arc<MemoryStore>,
    registry: &Arc<WorkerRegistry>,
    connector: Arc<dyn EngineConnector>,
    controller: Settings,
) -> (Feed, Follower) {
    let feed = feed::following(store, registry, connector);
    let follower = Follower::start(controller, feed.clone());
    (feed, follower)
}

/// The refusing router for a family the gateway's own policy engine does
/// not decide: the gRPC encode pair router. It mounts under the family's
/// usual id so the gateway can select it. `None` where the context holds no
/// such engine: the family is then its own router.
pub(crate) fn undecided(context: &AppContext) -> Option<Box<dyn RouterTrait>> {
    context
        .policy_engine
        .is_some()
        .then(|| Box::new(Undecided) as Box<dyn RouterTrait>)
}

/// The refusing router for a family the gateway's own policy engine does
/// not decide in this process. `None` where the context holds no such
/// engine, and where it decides the family.
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
/// yet, so the family refuses by name there.
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
mod start_in_process_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod worker_zone_tests;
