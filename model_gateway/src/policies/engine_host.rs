//! The gateway as host of the policy engine.
//!
//! The host builds the engine's bounded view from the pool's live state (one
//! routing-state read and one load-report lookup per candidate, into a
//! thread-local buffer), asks the engine for a decision outside its own
//! locks, and commits one target under them: the commit re-validates each
//! target through the caller's predicate and walks the ordered list on a
//! miss. The engine's accounting is fed at the commit (`on_dispatched`), so a
//! decision that never commits books nothing.
//!
//! The rows carry what the built-in stages read; a field no stage reads yet
//! (wire modes, zones, labels, the index's overlap) stays at its default
//! until the step that feeds it. The regular placement sites decide through
//! the host (`routers::common::placement::select_from`) and the encode items
//! of an encode-prefill-decode request through [`Decider::decide_encode_items`]
//! (the gRPC worker-selection stage); the pair and admission paths follow in
//! later steps. A removed or replaced worker's rows leave the engine's
//! accounting with it ([`Decider::on_worker_removed`] and
//! [`Decider::on_worker_replaced`], from the workflow's removal, registration
//! and property-update steps), and its session pins go with them: a key whose
//! owner was removed keeps its failover as the owner, a key with no pin left
//! is placed afresh. A worker re-registered at the same address is a new id
//! to the engine, so its old keys do not come back to it (the legacy maps pin
//! by URL and give them back). Mesh imports, which the registry rebuilds and
//! removes without a workflow, do not reach these hooks yet; they and the
//! rest of the membership feed (added, resized, snapshot) follow in a later
//! step.
//!
//! A session (the routing-key header, else the rid-derived key) is a
//! binding's `sessions` rule in the engine: the `manual` policy pins every
//! key, and the sticky routing-key override attaches the same rule to the
//! default policy's binding. The host hands the engine the key under the
//! model and phase it is for, as the legacy keyed paths namespace their pins,
//! and the engine's session table learns the pin at the commit from the
//! decision's own verdict, which the commit copies from the pick onto the
//! lease (a fresh key is claimed at its decision). The in-flight cap reads
//! the host's per-key count on each worker, the one the load guard keeps
//! under the raw key.

use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use openai_protocol::worker::WorkerLoadResponse;
use parking_lot::{Mutex, RwLock};
use smg_policy_controller::{
    fingerprint,
    spec::{
        BindingSpec, LeastLoadParams, PlanName, ProfileSpec, ScoreSpec, ScoreStage,
        MANUAL_SESSION_HEADER,
    },
    store::{identity, identity_of},
    Assignment, Candidate, Constraints, Decision, Engine, Health, Identity, Lease, LeaseId,
    LeaseKind, Lifecycle, LoadRow, MediaCounts, Membership, Needs, PairPools, Phase, PolicySpec,
    PoolFacts, Recovery, RequestFacts, Role, Scratch, SessionVerdict, SessionsSpec, Side,
    SignalSource, SpecError, Target, WorkerId,
};
use tracing::debug;

use super::registry::STICKY_INFLIGHT_CAP;
use crate::{
    config::{ManualAssignmentMode, PolicyConfig, PowerOfTwoLoadMetric, RoutingKeyOverrideConfig},
    policies::{host_stores::HostStores, normalize_model_key},
    routers::common::pd_admission,
    worker::{
        expected_wait::DEFAULT_MEAN_PREFILL_TOKENS,
        load_state::{LoadReceiver, LoadSnapshot},
        Worker, WorkerType,
    },
};

/// The crate's least-load scorer, the one stage that prices token-work.
const LEAST_LOAD: &str = "least-load";

/// Why the engine cannot serve a policy configuration in this build.
#[derive(Debug, thiserror::Error)]
pub enum EngineHostError {
    /// A policy, or a form of one, whose stages the engine does not carry
    /// yet.
    #[error("the policy engine does not carry the stages of {0} yet")]
    Unsupported(&'static str),
    #[error(transparent)]
    Spec(#[from] SpecError),
}

/// The host's per-worker capacity predicate a decision reads outside the
/// admission lock.
pub type CapacityCheck<'a> = &'a dyn Fn(&Arc<dyn Worker>) -> bool;

/// The engine preset a policy configuration is served as in this build;
/// `None` for a policy, or a form of one, whose stages the engine does not
/// carry yet. The validator refuses such a configuration under
/// `--policy-engine engine` at start.
pub fn preset_name(config: &PolicyConfig) -> Option<&'static str> {
    Some(match config {
        PolicyConfig::Random => "random",
        PolicyConfig::RoundRobin => "round_robin",
        PolicyConfig::Passthrough => "passthrough",
        PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::ResidentTokens { .. },
        } => return None,
        PolicyConfig::PowerOfTwo { .. } => "power_of_two",
        PolicyConfig::LeastLoad { .. } => "least_load",
        PolicyConfig::Manual { .. } => "manual",
        _ => return None,
    })
}

/// The engine preset an encode policy configuration is served as: the three
/// policies the encode side takes, each carried in this build; `None` for
/// anything else. The default side keeps [`preset_name`]: the consistent
/// hashing preset serves encode items, whose keys are content hashes, and
/// not yet the regular path, whose keyed form (the positional target header,
/// the implicit client keys) moves behind the view in a later step.
pub fn encode_preset_name(config: &PolicyConfig) -> Option<&'static str> {
    Some(match config {
        PolicyConfig::Random => "random",
        PolicyConfig::RoundRobin => "round_robin",
        PolicyConfig::ConsistentHashing => "consistent_hashing",
        _ => return None,
    })
}

/// The name of what the engine does not carry in `config`, for the error.
fn unsupported(config: &PolicyConfig) -> EngineHostError {
    EngineHostError::Unsupported(match config {
        PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::ResidentTokens { .. },
        } => "power_of_two with resident_tokens",
        other => other.name(),
    })
}

/// The preset profile of one policy configuration with its tunables carried
/// over: the least-load parameters, and the stages the power-of-two metrics
/// name (`requests` draws two and compares in-flight counts, `least_requests`
/// takes the fewest in flight over every candidate).
fn profile_of(config: &PolicyConfig) -> Result<ProfileSpec, EngineHostError> {
    let name = preset_name(config).ok_or_else(|| unsupported(config))?;
    let mut profile = PolicySpec::preset(name)?
        .profiles
        .remove(name)
        .ok_or(SpecError::Empty)?;
    tune(&mut profile, config);
    Ok(profile)
}

/// The mean prefill a profile prices an unknown prompt size at: its
/// least-load scorer's parameter, the scorer's own default when the preset
/// runs it untuned (every `power_of_two` metric ends its chain in the
/// expected wait), nothing when the profile prices no tokens. The crate's
/// untuned scorer and the legacy policies share the default of 1024 tokens.
fn mean_prefill_of(profile: &ProfileSpec) -> u32 {
    profile
        .score
        .iter()
        .find_map(|score| match score {
            ScoreSpec::Name(name) if name == LEAST_LOAD => Some(DEFAULT_MEAN_PREFILL_TOKENS),
            ScoreSpec::Name(_) => None,
            ScoreSpec::Params(ScoreStage::LeastLoad(params)) => Some(
                params
                    .mean_prefill_tokens
                    .unwrap_or(DEFAULT_MEAN_PREFILL_TOKENS),
            ),
            ScoreSpec::Params(_) => None,
        })
        .unwrap_or(0)
}

/// Carry a configuration's tunables into its preset profile (see
/// [`profile_of`]); a profile built by the crate's pair presets takes the
/// same adjustment. The two in-flight metrics of `power_of_two` keep the
/// policies' two stages through the engine's count-first pickers: the
/// fewest in flight, the expected wait among equal counts.
fn tune(profile: &mut ProfileSpec, config: &PolicyConfig) {
    match config {
        PolicyConfig::LeastLoad {
            kv_pressure_weight,
            mean_prefill_tokens,
            default_throughput,
        } => {
            profile.score = vec![ScoreSpec::Params(ScoreStage::LeastLoad(LeastLoadParams {
                weight: 1.0,
                kv_pressure_weight: Some(*kv_pressure_weight as f32),
                mean_prefill_tokens: Some(*mean_prefill_tokens),
                default_throughput: Some(*default_throughput as f32),
            }))];
        }
        PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::Requests,
        } => {
            profile.score = vec![ScoreSpec::Name("least-load".into())];
            profile.pick = "power-of-two-count-first".into();
        }
        PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::LeastRequests,
        } => {
            profile.score = vec![ScoreSpec::Name("least-load".into())];
            profile.pick = "count-first".into();
        }
        // The manual policy's assignment of a new key is its profile: random
        // draws, min_load and delegate (nothing underneath to delegate to)
        // take the fewest requests in flight, min_group the fewest sessions.
        PolicyConfig::Manual {
            assignment_mode, ..
        } => match assignment_mode {
            ManualAssignmentMode::Random => {}
            ManualAssignmentMode::MinLoad | ManualAssignmentMode::Delegate => {
                profile.score = vec![ScoreSpec::Name("least-requests".into())];
                profile.pick = "best".into();
            }
            ManualAssignmentMode::MinGroup => {
                profile.score = vec![ScoreSpec::Name("fewest-sessions".into())];
                profile.pick = "best".into();
            }
        },
        _ => {}
    }
}

/// The sessions rule of the default binding: the `manual` policy pins every
/// key, recovers to its oldest pin and caps nothing; under the sticky
/// routing-key override every other policy pins its keys at the override's
/// assignment and in-flight cap, recovering to the latest pin under
/// `delegate` and to the first pin under the other assignments, as the
/// legacy override fronts a delegated pick and keeps an assigned pin's
/// original first. A zero idle time or a zero sweep interval keeps entries
/// for ever, as either leaves the legacy eviction thread off. The override's
/// headers are the rule's, lowercased and each once; one on the engine's
/// deny-list refuses the start under the engine (a session header is an
/// allow-listed header of the view), where the legacy override takes it.
fn sessions_of(default: &PolicyConfig, sticky: &RoutingKeyOverrideConfig) -> Option<SessionsSpec> {
    // The schema's largest idle time or sweep is for ever, which is what a
    // zero of either legacy setting means (the eviction thread off).
    let lifetime = |max_idle_secs: u64, eviction_interval_secs: u64| {
        if max_idle_secs == 0 || eviction_interval_secs == 0 {
            (u32::MAX, u32::MAX)
        } else {
            let seconds = |secs: u64| u32::try_from(secs).unwrap_or(u32::MAX);
            (seconds(max_idle_secs), seconds(eviction_interval_secs))
        }
    };
    match default {
        PolicyConfig::Manual {
            eviction_interval_secs,
            max_idle_secs,
            ..
        } => {
            let (ttl_s, sweep_s) = lifetime(*max_idle_secs, *eviction_interval_secs);
            Some(SessionsSpec {
                sticky_header: MANUAL_SESSION_HEADER.into(),
                headers: vec![],
                ttl_s,
                sweep_s,
                assignment: Assignment::Delegate,
                // The schema's largest cap is no cap, as the manual policy has none.
                inflight_cap: u16::MAX,
                issue_session_id: false,
                recovery: Recovery::Original,
            })
        }
        _ if sticky.enabled => {
            let (ttl_s, sweep_s) = lifetime(sticky.max_idle_secs, sticky.eviction_interval_secs);
            // Lowercased and each name once, in order: the registry keeps
            // the operator's list as written, the view lists a name once.
            let mut headers: Vec<String> = Vec::with_capacity(sticky.headers.len());
            for header in &sticky.headers {
                let name = header.to_ascii_lowercase();
                if !headers.contains(&name) {
                    headers.push(name);
                }
            }
            let sticky_header = headers
                .first()
                .cloned()
                .unwrap_or_else(|| MANUAL_SESSION_HEADER.into());
            let recovery = if sticky.assignment_mode == ManualAssignmentMode::Delegate {
                Recovery::Latest
            } else {
                Recovery::Original
            };
            Some(SessionsSpec {
                sticky_header,
                headers: if headers.len() > 1 { headers } else { vec![] },
                ttl_s,
                sweep_s,
                assignment: assignment_of(sticky.assignment_mode),
                inflight_cap: STICKY_INFLIGHT_CAP,
                issue_session_id: false,
                recovery,
            })
        }
        _ => None,
    }
}

/// The engine's assignment of a new key for the override's mode.
fn assignment_of(mode: ManualAssignmentMode) -> Assignment {
    match mode {
        ManualAssignmentMode::Random => Assignment::Random,
        ManualAssignmentMode::MinLoad => Assignment::MinLoad,
        ManualAssignmentMode::MinGroup => Assignment::MinGroup,
        ManualAssignmentMode::Delegate => Assignment::Delegate,
    }
}

/// The scope a session key pins under: its model and phase, so one key is
/// one session per model and the prefill and decode sides stick
/// independently, as the legacy keyed paths namespace their pins.
fn session_scope(model: &str, phase: Phase) -> u64 {
    fingerprint(normalize_model_key(model).as_bytes())
        ^ (phase as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// The number standing for a worker's incarnation until the controller's
/// store numbers them: the handle's address, as the load feed's provenance
/// fence reads it. A same-URL replacement is a new handle.
pub fn incarnation(worker: &dyn Worker) -> u64 {
    (worker as *const dyn Worker).cast::<()>().addr() as u64
}

/// The stable id of a worker in the view: its registration key (the URL)
/// and its incarnation, under the store's identity rule, so a same-URL
/// replacement is a new id and the host's rows carry the view's ids.
pub fn worker_id(worker: &dyn Worker) -> WorkerId {
    identity(worker.url(), incarnation(worker))
}

/// The id of a worker handle and the worker's stable key for placement
/// ([`Candidate::key`]), both from one pass over the registration key: the
/// id under the store's identity rule, the key the fingerprint itself, so a
/// ring positions the worker alike across its handles and across gateway
/// replicas while the lifecycle still prunes by handle.
pub fn worker_identity(worker: &dyn Worker) -> (WorkerId, u32) {
    let stable = fingerprint(worker.url().as_bytes());
    (identity_of(stable, incarnation(worker)), stable as u32)
}

/// Whether stages with `needs` read the request body: its text, or the
/// prefix hashes of either kind the host computes from it. The token hashes
/// alone are computed from a request's token ids when it carries them.
pub(crate) fn wants_text(needs: Needs) -> bool {
    needs.text || needs.token_hashes || needs.text_hashes
}

thread_local! {
    /// The view of the decision in progress on this thread, reused across
    /// decisions so a decision allocates nothing after the first.
    static VIEW: RefCell<View> = RefCell::new(View::default());
}

#[derive(Default)]
struct View {
    candidates: Vec<Candidate>,
    decodes: Vec<Candidate>,
    tried: Vec<WorkerId>,
    scratch: Scratch,
}

/// What a decision reads from the request as the host holds it.
#[derive(Clone, Copy, Default)]
pub struct HostRequest<'a> {
    pub model: &'a str,
    /// The host's token count or estimate.
    pub prompt_tokens: u32,
    /// The request's session key as the host resolved it (the routing-key
    /// header, else the rid-derived key): the fact a binding's sessions rule
    /// pins by, under the request's model and phase.
    pub session_key: Option<&'a str>,
    /// Base URLs of the workers earlier attempts dispatched to.
    pub tried: &'a [String],
    pub overload_lifted: bool,
    /// The phase the decision is for: `Single` on the regular paths,
    /// `Prefill` for a prefill side decided alone (its profile carries the
    /// capacity filter).
    pub phase: Phase,
    /// The prefill capacity window as the host reads it, when the decision
    /// is a prefill side's; every candidate has capacity otherwise.
    pub has_capacity: Option<CapacityCheck<'a>>,
    /// The request's lease sequence when the router drew it before it
    /// captured its candidates ([`Decider::begin`]): the decision and the
    /// one re-decision over the same candidates run under it, so a handle
    /// retired after the draw is booked by neither commit. `None` numbers
    /// the lease when its view is built.
    pub sequence: Option<LeaseId>,
}

/// What the encode items of one request share, as the host holds it: the
/// two predicates the rows' bits are read from, and the request's images.
#[derive(Clone, Copy)]
pub struct EncodeRequest<'a> {
    pub model: &'a str,
    /// The encoder pairs with the chosen prefill side (its runtime and,
    /// under a push handoff, its EC roles and connector): the row's
    /// `ec_compatible` bit, the `epd` plan's first filter.
    pub pairs_with_prefill: &'a dyn Fn(&Arc<dyn Worker>) -> bool,
    /// The encoder has a vision encoder: the row's `vision` bit, the plan's
    /// second filter when the request carries images.
    pub vision_capable: &'a dyn Fn(&Arc<dyn Worker>) -> bool,
    /// The request's image items, as the view's media count; zero lifts
    /// the plan's vision rule.
    pub images: u16,
}

/// The engine in the gateway process with the host's feeds.
pub struct Decider {
    engine: Engine,
    loads: RwLock<Option<LoadReceiver>>,
    /// The handles the membership retired, each with the lease sequence at
    /// its retirement; the commit reads it so a lease decided before the
    /// retirement books nothing on the retired handle.
    retired: Mutex<Retired>,
    /// Test seam: runs after a commit's checks and before its bookings,
    /// under the lock they share.
    #[cfg(test)]
    between_check_and_booking: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    /// The store ingress over this process's structures, handed in under
    /// `--policy-engine engine`; nothing reads or writes through it until the
    /// step that moves the ingestion drivers.
    stores: RwLock<Option<Arc<HostStores>>>,
    /// The host's clock origin: every instant in the view is microseconds
    /// since it.
    origin: Instant,
    seq: AtomicU64,
    /// The token-work a request of unknown size is booked at: the mean
    /// prefill of the default profile's least-load scorer when it has one
    /// (`least_load`'s setting, the scorer's default under `power_of_two`),
    /// nothing otherwise.
    mean_prompt_tokens: u32,
    /// What the engine's stages read, from its needs at the build: the two
    /// per-key counts are map reads per row, so a row carries them only when
    /// a stage reads them.
    needs: Needs,
    /// The engine binds an encode profile: the `epd` plan decides the encode
    /// items of an encode-prefill-decode request.
    encode: bool,
}

impl std::fmt::Debug for Decider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decider")
            .field("generation", &self.engine.generation())
            .field("needs", &self.engine.needs())
            .field("decisions", &self.decisions())
            .finish_non_exhaustive()
    }
}

/// What every row of one view build shares: the load snapshot, the
/// instant and the request's session key.
#[derive(Clone, Copy)]
struct ViewBuild<'a> {
    snapshot: Option<&'a LoadSnapshot>,
    now: Instant,
    session_key: Option<&'a str>,
}

impl Decider {
    /// The engine for the gateway's policy configuration: the default policy
    /// bound to every model, the prefill and decode policies under the
    /// `pd-pair` plan with the admission filters in front of each side. The
    /// default binding pins sessions when the default policy is `manual`, or
    /// when the sticky routing-key override (`sticky`) is enabled
    /// (`sessions_of`).
    pub fn from_configs(
        default: &PolicyConfig,
        prefill: &PolicyConfig,
        decode: &PolicyConfig,
        sticky: &RoutingKeyOverrideConfig,
    ) -> Result<Self, EngineHostError> {
        Self::from_configs_with_encode(default, prefill, decode, None, sticky)
    }

    /// [`Decider::from_configs`] with the encode side of an
    /// encode-prefill-decode deployment: the three sides under the `epd`
    /// plan, the encode profile over the encode pool with the plan's
    /// compatibility and vision rules in front of it. An encode policy the
    /// encode side does not carry ([`encode_preset_name`]) is refused by
    /// name.
    pub fn from_configs_with_encode(
        default: &PolicyConfig,
        prefill: &PolicyConfig,
        decode: &PolicyConfig,
        encode: Option<&PolicyConfig>,
        sticky: &RoutingKeyOverrideConfig,
    ) -> Result<Self, EngineHostError> {
        let (prefill_name, decode_name) = (
            preset_name(prefill).ok_or_else(|| unsupported(prefill))?,
            preset_name(decode).ok_or_else(|| unsupported(decode))?,
        );
        let mut spec = match encode {
            Some(encode) => {
                let encode_name = encode_preset_name(encode).ok_or_else(|| unsupported(encode))?;
                PolicySpec::epd_presets(encode_name, prefill_name, decode_name)?
            }
            None => PolicySpec::pair_presets(prefill_name, decode_name)?,
        };
        if let Some(profile) = spec.profiles.get_mut("prefill") {
            tune(profile, prefill);
        }
        if let Some(profile) = spec.profiles.get_mut("decode") {
            tune(profile, decode);
        }
        spec.profiles
            .insert("default".to_string(), profile_of(default)?);
        // The plan's binding covers its phases (the pair's two, the encode
        // phase too under `epd`); every other phase resolves to the default
        // profile.
        let phases = if encode.is_some() {
            vec![Phase::Encode, Phase::Prefill, Phase::Decode]
        } else {
            vec![Phase::Prefill, Phase::Decode]
        };
        for binding in &mut spec.bindings {
            binding.matcher.phase.clone_from(&phases);
        }
        let sessions = sessions_of(default, sticky);
        if let Some(rule) = &sessions {
            // The validator wants every header a sessions rule names in the
            // view's list; the host feeds the key it resolved under them.
            spec.view.headers = if rule.headers.is_empty() {
                vec![rule.sticky_header.clone()]
            } else {
                rule.headers.clone()
            };
        }
        spec.bindings.push(BindingSpec {
            profile: Some("default".to_string()),
            sessions,
            ..BindingSpec::default()
        });
        Self::from_spec(&spec)
    }

    /// The engine for a document: `spec` validated and hosted. A request of
    /// unknown prompt size is booked at the mean prefill of the profile
    /// named `default` when it prices tokens, nothing otherwise.
    pub fn from_spec(spec: &PolicySpec) -> Result<Self, EngineHostError> {
        let engine = spec.validate()?;
        Ok(Self {
            needs: engine.needs(),
            engine,
            loads: RwLock::new(None),
            retired: Mutex::new(Retired::default()),
            #[cfg(test)]
            between_check_and_booking: Mutex::new(None),
            stores: RwLock::new(None),
            origin: Instant::now(),
            seq: AtomicU64::new(1),
            mean_prompt_tokens: spec.profiles.get("default").map_or(0, mean_prefill_of),
            encode: spec.bindings.iter().any(|b| b.plan == Some(PlanName::Epd)),
        })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Whether the engine binds an encode profile, so the encode items of an
    /// encode-prefill-decode request are its decisions.
    pub fn serves_encode(&self) -> bool {
        self.encode
    }

    /// A worker removed from the pool takes its rows in the engine's
    /// accounting with it (`Membership::Removed` under its id), as the legacy
    /// policies drop their per-worker state on removal; the policy registry's
    /// removal path calls this beside that pruning.
    pub fn on_worker_removed(&self, worker: &dyn Worker) {
        let id = worker_id(worker);
        self.retire(id);
        self.engine.on_membership(&Membership::Removed(id));
    }

    /// A worker replaced in place (the same URL, a new handle) retires its
    /// rows the same way: the engine keys its accounting by handle, so the
    /// old handle's rows would otherwise stay for good under churn.
    /// `Membership::Replaced` prunes them; the replacement starts with none.
    pub fn on_worker_replaced(&self, old: &dyn Worker, new: &dyn Worker) {
        let old = worker_id(old);
        self.retire(old);
        self.engine.on_membership(&Membership::Replaced {
            old,
            new: worker_id(new),
        });
    }

    /// Record `id`'s retirement at the next lease to be issued. A lease is
    /// numbered at the router's capture (`begin`) or before its view is
    /// built, so every decision that began before this call carries a lower
    /// sequence.
    fn retire(&self, id: WorkerId) {
        self.retired
            .lock()
            .retire(id, self.seq.load(Ordering::Relaxed));
    }

    /// Draw a request's lease sequence before its candidates are captured.
    /// A router that reads the pool after this call books nothing on a
    /// handle retired from then on, however the retirement interleaves with
    /// its capture and its decision: [`HostRequest::sequence`] carries the
    /// draw into [`Decider::decide`], and the one re-decision over the same
    /// candidates reuses it. A request without one is numbered when its
    /// view is built.
    pub fn begin(&self) -> LeaseId {
        LeaseId(self.seq.fetch_add(1, Ordering::Relaxed))
    }

    /// Lease sequences drawn so far: each `begin`, and each `decide` or
    /// `decide_pair` call without a drawn one; for tests and dumps.
    pub fn decisions(&self) -> u64 {
        self.seq.load(Ordering::Relaxed).saturating_sub(1)
    }

    /// The prompt's token count as the host books it on a lease: the
    /// request's tokens when the host has them, else the mean a request of
    /// unknown size is priced at (the least-load policy's
    /// `mean_prefill_tokens`, as the legacy policy prices it; nothing under a
    /// policy that does not price tokens).
    pub fn prompt_tokens(&self, tokens: Option<&[u32]>) -> u32 {
        tokens.map_or(self.mean_prompt_tokens, |tokens| {
            u32::try_from(tokens.len()).unwrap_or(u32::MAX)
        })
    }

    /// Hand the engine the worker monitor's load feed.
    pub fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        *self.loads.write() = rx;
    }

    /// Hand the engine the host's store ingress (the registry, the monitors
    /// and the admission counters behind the store traits).
    pub fn set_stores(&self, stores: Arc<HostStores>) {
        *self.stores.write() = Some(stores);
    }

    /// The host's store ingress, once handed in.
    pub fn stores(&self) -> Option<Arc<HostStores>> {
        self.stores.read().clone()
    }

    fn snapshot(&self) -> Option<Arc<LoadSnapshot>> {
        self.loads.read().as_ref().map(|rx| rx.borrow().clone())
    }

    fn micros_since_origin(&self, at: Instant) -> u64 {
        at.checked_duration_since(self.origin)
            .map_or(0, |d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX))
    }

    fn candidate(
        &self,
        build: &ViewBuild<'_>,
        position: u32,
        worker: &Arc<dyn Worker>,
        has_capacity: bool,
        pair_class: u32,
    ) -> Candidate {
        let ViewBuild {
            snapshot,
            now,
            session_key,
        } = *build;
        let state = worker.routing_state();
        let worker_type = *worker.worker_type();
        let role = match worker_type {
            WorkerType::Prefill => Role::Prefill,
            WorkerType::Decode => Role::Decode,
            WorkerType::Encode => Role::Encode,
            WorkerType::Regular => Role::Single,
        };
        let (load, load_sampled_at_micros) = snapshot.and_then(|s| s.report(worker.url())).map_or(
            (None, 0),
            |(report, sampled_at)| {
                (
                    Some(load_row(report, worker_type, now, sampled_at)),
                    self.micros_since_origin(sampled_at),
                )
            },
        );
        let (id, key) = worker_identity(worker.as_ref());
        Candidate {
            position,
            id,
            key,
            role,
            dp_rank: worker
                .dp_rank()
                .and_then(|r| u16::try_from(r).ok())
                .unwrap_or(0),
            dp_size: worker
                .dp_size()
                .and_then(|s| u16::try_from(s).ok())
                .unwrap_or(1),
            health: Health {
                healthy: state.healthy,
                breaker_open: !state.can_execute,
                stalled: state.stalled,
                // The gateway folds a drain into the status, which the
                // healthy bit reads.
                draining: false,
                overloaded: state.overloaded,
            },
            in_flight: u32::try_from(state.load).unwrap_or(u32::MAX),
            load,
            load_sampled_at_micros,
            session_in_flight: match session_key {
                Some(key) if self.needs.session_in_flight => {
                    u16::try_from(worker.routing_key_inflight(key)).unwrap_or(u16::MAX)
                }
                _ => 0,
            },
            active_sessions: if self.needs.active_sessions {
                u16::try_from(worker.routing_key_load()).unwrap_or(u16::MAX)
            } else {
                0
            },
            pair_class,
            has_capacity,
            has_room: role != Role::Decode || pd_admission::has_free_room(worker.as_ref()),
            ..Candidate::default()
        }
    }

    fn lease(&self, id: LeaseId, now: Instant, tokens: u32, session_key: Identity<'_>) -> Lease {
        Lease {
            id,
            kind: LeaseKind::Credit,
            side: Side::Single,
            target_index: 0,
            generation: self.engine.generation(),
            session_key: session_key.hash,
            session: SessionVerdict::None,
            issued_micros: self.micros_since_origin(now),
            tokens,
        }
    }

    fn seed(&self, now_micros: u64) -> u64 {
        now_micros ^ self.seq.load(Ordering::Relaxed).rotate_left(32)
    }

    /// Decide one worker over `workers`. `excluded` are the ids a commit of
    /// this request just refused, for the one re-decision.
    pub fn decide(
        &self,
        request: &HostRequest<'_>,
        workers: &[Arc<dyn Worker>],
        excluded: &[WorkerId],
    ) -> (Decision, Lease) {
        let now = Instant::now();
        // Under the sequence the router drew at its capture, or one drawn
        // here before the view is built: a retirement that lands after the
        // draw outdates this lease (see `Retired`).
        let id = request.sequence.unwrap_or_else(|| self.begin());
        let snapshot = self.snapshot();
        let session_key = request.session_key.map_or(Identity::NONE, |key| {
            Identity::scoped(key, session_scope(request.model, request.phase))
        });
        let build = ViewBuild {
            snapshot: snapshot.as_deref(),
            now,
            session_key: request.session_key,
        };
        VIEW.with(|view| {
            let mut view = view.borrow_mut();
            let View {
                candidates,
                tried,
                scratch,
                ..
            } = &mut *view;
            candidates.clear();
            tried.clear();
            for (position, worker) in workers.iter().enumerate() {
                let candidate = self.candidate(
                    &build,
                    position as u32,
                    worker,
                    request.has_capacity.is_none_or(|has| has(worker)),
                    0,
                );
                if request.tried.iter().any(|url| url == worker.base_url()) {
                    tried.push(candidate.id);
                }
                candidates.push(candidate);
            }
            let now_micros = self.micros_since_origin(now);
            let facts = RequestFacts {
                model: request.model,
                prompt_tokens: request.prompt_tokens,
                session_key,
                phase: request.phase,
                attempt: u8::try_from(request.tried.len()).unwrap_or(u8::MAX),
                tried,
                seed: self.seed(now_micros),
                now_micros,
                ..RequestFacts::default()
            };
            let pool = PoolFacts {
                pool_size: workers.len() as u32,
                generation: self.engine.generation(),
                ..PoolFacts::default()
            };
            let cx = Constraints {
                excluded,
                overload_lifted: request.overload_lifted,
                ..Constraints::default()
            };
            let decision = self.engine.decide(&facts, &pool, candidates, &cx, scratch);
            (
                decision,
                self.lease(id, now, request.prompt_tokens, session_key),
            )
        })
    }

    /// Decide the encode items of one request: one view of `encoders`, with
    /// each row's compatibility and vision bits read through the request's
    /// predicates, and one decision per key under the `epd` plan's encode
    /// profile. A key is an item's content hash and travels as the
    /// decision's session key, so under the consistent-hash preset an item
    /// lands on its key's ring owner and equal items share an encoder;
    /// `excluded` are the ids a commit of an item just refused, for the one
    /// re-decision of that item. The decisions come back in the keys' order
    /// with their leases.
    pub fn decide_encode_items(
        &self,
        request: &EncodeRequest<'_>,
        encoders: &[Arc<dyn Worker>],
        keys: &[u64],
        excluded: &[WorkerId],
    ) -> Vec<(Decision, Lease)> {
        let now = Instant::now();
        // Numbered before the view is built, one id per item: a retirement
        // that lands during the decisions outdates every lease of this
        // request (see `Retired`).
        let first = self.seq.fetch_add(keys.len() as u64, Ordering::Relaxed);
        let snapshot = self.snapshot();
        let build = ViewBuild {
            snapshot: snapshot.as_deref(),
            now,
            session_key: None,
        };
        VIEW.with(|view| {
            let mut view = view.borrow_mut();
            let View {
                candidates,
                scratch,
                ..
            } = &mut *view;
            candidates.clear();
            for (position, worker) in encoders.iter().enumerate() {
                let mut row = self.candidate(&build, position as u32, worker, true, 0);
                row.ec_compatible = (request.pairs_with_prefill)(worker);
                row.vision = (request.vision_capable)(worker);
                candidates.push(row);
            }
            let now_micros = self.micros_since_origin(now);
            let pool = PoolFacts {
                pool_size: encoders.len() as u32,
                generation: self.engine.generation(),
                ..PoolFacts::default()
            };
            let cx = Constraints {
                excluded,
                ..Constraints::default()
            };
            keys.iter()
                .enumerate()
                .map(|(item, &key)| {
                    let session_key = Identity::hashed(key);
                    let facts = RequestFacts {
                        model: request.model,
                        session_key,
                        phase: Phase::Encode,
                        media: MediaCounts {
                            images: request.images,
                            ..MediaCounts::default()
                        },
                        seed: self.seed(now_micros) ^ key,
                        now_micros,
                        ..RequestFacts::default()
                    };
                    let decision = self.engine.decide(&facts, &pool, candidates, &cx, scratch);
                    let id = LeaseId(first + item as u64);
                    (decision, self.lease(id, now, 0, session_key))
                })
                .collect()
        })
    }

    /// Decide a prefill and decode pair: the prefill profile over `prefills`
    /// (with `has_capacity` as the admission window reads it now), the
    /// decode profile over the partners of the chosen prefill. `partners(i)`
    /// lists the open partners of `prefills[i]`; the decode rows of each
    /// prefill form its pair class. The request's tried URLs and `excluded`
    /// reach both sides. The returned workers are the decode rows in
    /// position order, for the target's decode side.
    pub fn decide_pair(
        &self,
        request: &HostRequest<'_>,
        prefills: &[Arc<dyn Worker>],
        has_capacity: CapacityCheck<'_>,
        partners: &dyn Fn(usize) -> Vec<Arc<dyn Worker>>,
        lift_decode: bool,
        excluded: &[WorkerId],
    ) -> (Decision, Lease, Vec<Arc<dyn Worker>>) {
        let now = Instant::now();
        // Under the sequence the router drew at its capture, or one drawn
        // here before the view is built: a retirement that lands after the
        // draw outdates this lease (see `Retired`).
        let id = request.sequence.unwrap_or_else(|| self.begin());
        let snapshot = self.snapshot();
        let session_key = request.session_key.map_or(Identity::NONE, Identity::plain);
        let build = ViewBuild {
            snapshot: snapshot.as_deref(),
            now,
            session_key: request.session_key,
        };
        let mut decode_workers: Vec<Arc<dyn Worker>> = Vec::new();
        let decision = VIEW.with(|view| {
            let mut view = view.borrow_mut();
            let View {
                candidates,
                decodes,
                tried,
                scratch,
            } = &mut *view;
            candidates.clear();
            decodes.clear();
            tried.clear();
            for (i, prefill) in prefills.iter().enumerate() {
                let class = i as u32 + 1;
                let candidate =
                    self.candidate(&build, i as u32, prefill, has_capacity(prefill), class);
                if request.tried.iter().any(|url| url == prefill.base_url()) {
                    tried.push(candidate.id);
                }
                candidates.push(candidate);
                for decode in partners(i) {
                    let position = decode_workers.len() as u32;
                    let row = self.candidate(&build, position, &decode, true, class);
                    if request.tried.iter().any(|url| url == decode.base_url()) {
                        tried.push(row.id);
                    }
                    decodes.push(row);
                    decode_workers.push(decode);
                }
            }
            let now_micros = self.micros_since_origin(now);
            let facts = RequestFacts {
                model: request.model,
                prompt_tokens: request.prompt_tokens,
                session_key,
                phase: Phase::Prefill,
                attempt: u8::try_from(request.tried.len()).unwrap_or(u8::MAX),
                tried,
                seed: self.seed(now_micros),
                now_micros,
                ..RequestFacts::default()
            };
            let pool = PoolFacts {
                pool_size: prefills.len() as u32,
                generation: self.engine.generation(),
                ..PoolFacts::default()
            };
            let cx = Constraints {
                excluded,
                overload_lifted: request.overload_lifted,
                ..Constraints::default()
            };
            let decode_cx = Constraints {
                excluded,
                overload_lifted: lift_decode,
                ..Constraints::default()
            };
            self.engine.decide_pair(
                &facts,
                &pool,
                PairPools {
                    prefills: candidates,
                    decodes,
                },
                &cx,
                &decode_cx,
                scratch,
            )
        });
        (
            decision,
            self.lease(id, now, request.prompt_tokens, session_key),
            decode_workers,
        )
    }

    /// The commit: walk the ordered targets, take the first one `accept`
    /// re-validates under the host's lock, and feed the engine's accounting
    /// for the target taken, on `side` and, for a pair, on the decode side
    /// too. `None` when every target missed; the caller re-decides once with
    /// the misses excluded, then falls back.
    pub fn commit(
        &self,
        lease: &Lease,
        decision: &Decision,
        side: Side,
        mut accept: impl FnMut(&Target) -> bool,
    ) -> Option<Target> {
        let pick = decision.pick()?;
        for (index, target) in pick.targets.iter().enumerate() {
            if accept(target) {
                let since_decision = self
                    .micros_since_origin(Instant::now())
                    .saturating_sub(lease.issued_micros);
                let since_decision = u32::try_from(since_decision).unwrap_or(u32::MAX);
                let booked = Lease {
                    side,
                    target_index: u8::try_from(index).unwrap_or(u8::MAX),
                    session: pick.session,
                    ..*lease
                };
                // A lease decided before a side's retirement books nothing
                // on it: the rows the retirement pruned stay gone, and the
                // dispatch itself still goes to the handle the host holds.
                // The check and the booking hold the lock together, so a
                // retirement lands before both or after both.
                {
                    let retired = self.retired.lock();
                    let book_target = !retired.predates(target.id, lease.id);
                    let book_decode = target
                        .pair
                        .is_some_and(|pair| !retired.predates(pair.decode_id, lease.id));
                    #[cfg(test)]
                    if let Some(hook) = self.between_check_and_booking.lock().as_ref() {
                        hook();
                    }
                    if book_target {
                        self.engine
                            .on_dispatched(&booked, target, side, since_decision);
                    }
                    if book_decode {
                        let decode = Lease {
                            side: Side::Decode,
                            ..booked
                        };
                        self.engine
                            .on_dispatched(&decode, target, Side::Decode, since_decision);
                    }
                }
                return Some(*target);
            }
            debug!(
                position = target.position,
                "policy engine: a target missed its commit re-validation"
            );
        }
        None
    }
}

/// How many retirements the commit remembers by handle. Past the cap the
/// oldest entry leaves and its bound becomes the record's watermark: a lease
/// decided before the oldest remembered retirement books nothing, whatever
/// handle it names, since that handle may be one the record forgot. The
/// dispatch itself still goes; only its accounting is skipped, and a lease
/// that old is past every dispatch deadline.
const RETIRED_KEPT: usize = 1024;

/// The handles the membership retired (removed, or replaced at their URL),
/// each with the lease sequence at its retirement. A commit under a lease
/// decided before that sequence books nothing on the handle, so the rows the
/// retirement pruned do not come back through a late dispatch; a lease
/// decided after it books as usual, as the id may be a new handle's (the
/// allocator reuses addresses). The entries are kept in the order of their
/// bounds; the oldest leaves at the cap, and its bound stays as the
/// watermark every check reads first.
#[derive(Debug, Default)]
struct Retired {
    bound: HashMap<WorkerId, u64>,
    order: VecDeque<WorkerId>,
    /// The bound of the newest entry the cap dropped: the order follows the
    /// bounds, so a lease below it is older than every retirement the record
    /// still names.
    watermark: u64,
}

impl Retired {
    fn retire(&mut self, id: WorkerId, next_lease: u64) {
        // A handle retired again (a successor at the same URL can carry the
        // same id once the allocator reuses the address) moves to the back
        // with its new bound, so the order stays the order of the bounds.
        if self.bound.insert(id, next_lease).is_some() {
            if let Some(at) = self.order.iter().position(|kept| *kept == id) {
                self.order.remove(at);
            }
        }
        self.order.push_back(id);
        if self.order.len() > RETIRED_KEPT {
            let dropped = self
                .order
                .pop_front()
                .and_then(|oldest| self.bound.remove(&oldest));
            if let Some(bound) = dropped {
                self.watermark = self.watermark.max(bound);
            }
        }
    }

    /// Whether `lease` was decided before `id`'s retirement, or before the
    /// oldest retirement the record still names.
    fn predates(&self, id: WorkerId, lease: LeaseId) -> bool {
        lease.0 < self.watermark || self.bound.get(&id).is_some_and(|&bound| lease.0 < bound)
    }
}

/// One worker's load row from its report, in one pass over the ranks: the
/// request and token counts summed, the KV ratios averaged, the drain rate
/// as the policies read it today. A prefill's drain is its measured input
/// capacity, which only a complete report with every rank's rate states: a
/// partial report or a rank without a rate leaves it unknown (zero), and
/// the scorer prices the worker at the fleet's nominal rate. Any other
/// role's drain is the generation rate of the ranks with a request running.
pub(crate) fn load_row(
    report: &WorkerLoadResponse,
    worker_type: WorkerType,
    now: Instant,
    sampled_at: Instant,
) -> LoadRow {
    let mut running: i64 = 0;
    let mut queued: i64 = 0;
    let mut uncached: i64 = 0;
    let mut max_running: i64 = 0;
    let mut kv_usage = 0.0f64;
    let mut active_kv = 0.0f64;
    let mut live_rate = 0.0f64;
    let complete = !report.loads.is_empty()
        && report.loads.len() >= usize::try_from(report.dp_rank_count).unwrap_or(0);
    let mut prefill_rate: Option<f64> = complete.then_some(0.0);
    let mut resident: Option<i64> = Some(0);
    for rank in &report.loads {
        running += i64::from(rank.num_running_reqs);
        queued += i64::from(rank.num_waiting_reqs);
        uncached += i64::from(rank.num_waiting_uncached_tokens);
        max_running += i64::from(rank.max_running_requests);
        kv_usage += rank.token_usage;
        active_kv += rank.active_token_usage.unwrap_or(rank.token_usage);
        match worker_type {
            WorkerType::Prefill => {
                prefill_rate = prefill_rate
                    .zip(
                        rank.prefill_throughput
                            .filter(|rate| rate.is_finite() && *rate > 0.0),
                    )
                    .map(|(sum, rate)| sum + rate);
            }
            _ if rank.num_running_reqs > 0 => live_rate += rank.gen_throughput,
            _ => {}
        }
        resident = resident.zip(rank.num_resident_tokens).map(|(a, b)| a + b);
    }
    if matches!(worker_type, WorkerType::Prefill) {
        live_rate = prefill_rate.unwrap_or(0.0);
    }
    let ranks = report.loads.len().max(1) as f64;
    LoadRow {
        running: narrow(running),
        queued: narrow(queued),
        waiting_uncached_tokens: narrow(uncached),
        kv_usage_permille: permille(kv_usage / ranks),
        active_kv_permille: permille(active_kv / ranks),
        live_rate: if live_rate.is_finite() && live_rate > 0.0 {
            live_rate.min(f64::from(u32::MAX)) as u32
        } else {
            0
        },
        resident_tokens: resident.map_or(0, narrow),
        report_age_ms: u32::try_from(now.saturating_duration_since(sampled_at).as_millis())
            .unwrap_or(u32::MAX),
        // The monitor's feed merges the poll tick and the pushed records
        // without naming which produced an entry; the row says poll until
        // the feed carries its source.
        source: SignalSource::Poll,
        max_running: narrow(max_running),
    }
}

/// A summed engine count as the row's field: negative reads as none.
fn narrow(count: i64) -> u32 {
    u32::try_from(count.max(0)).unwrap_or(u32::MAX)
}

/// A ratio as the row's permille field.
fn permille(ratio: f64) -> u16 {
    (ratio * 1000.0).round().clamp(0.0, 1000.0) as u16
}

#[cfg(test)]
mod tests {
    use std::sync::{mpsc, Arc};

    use openai_protocol::worker::WorkerStatus;
    use tokio::sync::watch;

    use super::*;
    use crate::{
        config::types::ManualAssignmentMode,
        policies::{test_support, PolicyFactory},
        worker::{
            load_state::{one_rank_load, LoadSnapshot},
            test_support::worker_of,
            BasicWorkerBuilder, Worker,
        },
    };

    fn pool(n: usize) -> Vec<Arc<dyn Worker>> {
        let urls: Vec<String> = (0..n).map(|i| format!("http://w{i}:8000")).collect();
        let refs: Vec<&str> = urls.iter().map(String::as_str).collect();
        test_support::make_workers(&refs)
    }

    /// Two workers under the same idle report, so only the engine's own
    /// credit tells them apart.
    fn fed_pair(decider: &Decider) -> (Vec<Arc<dyn Worker>>, watch::Sender<Arc<LoadSnapshot>>) {
        let workers = pool(2);
        let (tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
            (workers[0].url().to_string(), one_rank_load(0, 0.1, 2000.0)),
            (workers[1].url().to_string(), one_rank_load(0, 0.1, 2000.0)),
        ]));
        decider.set_load_receiver(Some(rx));
        (workers, tx)
    }

    fn request() -> HostRequest<'static> {
        HostRequest {
            model: "m",
            ..HostRequest::default()
        }
    }

    fn decider(config: &PolicyConfig) -> Decider {
        Decider::from_configs(config, config, config, &RoutingKeyOverrideConfig::default())
            .expect("a carried policy")
    }

    /// The default policy under the sticky routing-key override.
    fn overridden(config: &PolicyConfig, sticky: RoutingKeyOverrideConfig) -> Decider {
        Decider::from_configs(
            config,
            config,
            config,
            &RoutingKeyOverrideConfig {
                enabled: true,
                ..sticky
            },
        )
        .expect("a carried policy under the override")
    }

    fn manual(assignment_mode: ManualAssignmentMode) -> PolicyConfig {
        PolicyConfig::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: 14_400,
            assignment_mode,
        }
    }

    /// Decide and commit one request under `key`; the committed position.
    fn place(decider: &Decider, workers: &[Arc<dyn Worker>], key: &str) -> usize {
        let keyed = HostRequest {
            session_key: Some(key),
            ..request()
        };
        let (decision, lease) = decider.decide(&keyed, workers, &[]);
        let target = decider
            .commit(&lease, &decision, Side::Single, |t| {
                workers[t.position as usize].is_available()
            })
            .expect("a target");
        target.position as usize
    }

    /// The host answers its static needs (the text questions, the load
    /// poll, the stream kind) from the engine for the policies it serves,
    /// so for every configuration the converter carries the engine's needs
    /// for a regular placement and for both pair sides must equal the flags
    /// of the legacy policy built from the same configuration. A
    /// configuration the converter refuses is never answered from the
    /// engine.
    #[test]
    fn a_carried_configurations_needs_equal_the_legacy_policys_flags() {
        let configs = [
            PolicyConfig::Random,
            PolicyConfig::RoundRobin,
            PolicyConfig::Passthrough,
            PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096),
            PolicyConfig::PowerOfTwo {
                load_metric: PowerOfTwoLoadMetric::ExpectedWait,
            },
            PolicyConfig::PowerOfTwo {
                load_metric: PowerOfTwoLoadMetric::Requests,
            },
            PolicyConfig::PowerOfTwo {
                load_metric: PowerOfTwoLoadMetric::LeastRequests,
            },
            PolicyConfig::PowerOfTwo {
                load_metric: PowerOfTwoLoadMetric::ResidentTokens {
                    request_tokens: 512,
                },
            },
            PolicyConfig::LeastLoad {
                kv_pressure_weight: 0.0,
                mean_prefill_tokens: 100,
                default_throughput: 1000.0,
            },
            PolicyConfig::Bucket {
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                bucket_adjust_interval_secs: 5,
            },
            PolicyConfig::Manual {
                eviction_interval_secs: 60,
                max_idle_secs: 600,
                assignment_mode: ManualAssignmentMode::default(),
            },
            PolicyConfig::ConsistentHashing,
            PolicyConfig::PrefixHash {
                prefix_token_count: 256,
                load_factor: 1.25,
                balance_abs_threshold: 32,
                cache_boundaries: vec![],
            },
        ];
        let mut carried = 0;
        for config in &configs {
            let name = config.name();
            let legacy = PolicyFactory::create_from_config(config);
            let Ok(decider) =
                Decider::from_configs(config, config, config, &RoutingKeyOverrideConfig::default())
            else {
                assert!(preset_name(config).is_none(), "{name} is carried");
                continue;
            };
            carried += 1;
            for phase in [Phase::Single, Phase::Prefill, Phase::Decode] {
                let needs = decider.engine().needs_for_phase(phase);
                assert_eq!(
                    wants_text(needs),
                    legacy.needs_request_text(),
                    "{name} {phase:?}: text"
                );
                assert_eq!(
                    needs.text || needs.text_hashes,
                    legacy.needs_request_text_with_tokens(),
                    "{name} {phase:?}: text with token ids"
                );
                assert_eq!(
                    needs.loads,
                    legacy.needs_backend_loads(),
                    "{name} {phase:?}: loads"
                );
                assert_eq!(
                    needs.events,
                    legacy.needs_kv_events(),
                    "{name} {phase:?}: KV events"
                );
            }
        }
        assert_eq!(carried, 8, "the six presets, power_of_two in three forms");
    }

    /// A removed worker's rows leave the engine's accounting with it: work
    /// booked on a worker steers the next decision away from it until the
    /// worker is removed, after which nothing booked follows its id.
    #[test]
    fn a_removed_workers_rows_leave_the_accounting_with_it() {
        let decider = decider(&PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        });
        let (workers, _reports) = fed_pair(&decider);
        let request = HostRequest {
            prompt_tokens: decider.prompt_tokens(None),
            ..request()
        };
        let book = |position: u32| {
            let (decision, lease) = decider.decide(&request, &workers, &[]);
            decider
                .commit(&lease, &decision, Side::Single, |t| t.position == position)
                .expect("the wanted worker is booked");
        };
        let leader = || {
            let (decision, _) = decider.decide(&request, &workers, &[]);
            decision
                .pick()
                .expect("a pick")
                .targets
                .iter()
                .next()
                .map(|t| t.position)
        };
        book(0);
        book(0);
        book(1);
        assert_eq!(leader(), Some(1), "the worker with less booked work leads");

        decider.on_worker_removed(workers[0].as_ref());
        assert_eq!(
            leader(),
            Some(0),
            "the removed worker's rows went with it, so it reads idle again"
        );
    }

    #[test]
    fn the_engine_decides_over_the_live_pool_and_the_commit_takes_the_first_accepted_target() {
        let decider = decider(&PolicyConfig::RoundRobin);
        let workers = pool(3);
        workers[1].set_status(WorkerStatus::NotReady);
        let mut seen = Vec::new();
        for _ in 0..4 {
            let (decision, lease) = decider.decide(&request(), &workers, &[]);
            let targets = &decision.pick().expect("a pick").targets;
            assert!(
                targets.iter().all(|t| t.position != 1),
                "the view's healthy bit keeps the not-ready worker out of the decision: {targets:?}"
            );
            let target = decider
                .commit(&lease, &decision, Side::Single, |t| {
                    workers[t.position as usize].is_available()
                })
                .expect("a target");
            assert_ne!(target.position, 1, "the not-ready worker is filtered");
            seen.push(target.position);
        }
        assert!(seen.contains(&0) && seen.contains(&2), "{seen:?}");
    }

    /// A refused commit answers `None` and books nothing; the re-decision
    /// with the missed target excluded takes the other worker (the exclusion
    /// is a soft stage: it steps aside only when nothing else is left).
    #[test]
    fn a_refused_commit_answers_none_and_the_re_decision_excludes_the_miss() {
        let decider = decider(&PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        });
        let (workers, _feed) = fed_pair(&decider);
        let request = HostRequest {
            prompt_tokens: 512,
            ..request()
        };
        let (decision, lease) = decider.decide(&request, &workers, &[]);
        assert!(decider
            .commit(&lease, &decision, Side::Single, |_| false)
            .is_none());
        let missed = decision.pick().expect("a pick").targets.first().copied();
        let missed = missed.expect("a first target");
        for _ in 0..3 {
            let (again, _) = decider.decide(&request, &workers, &[missed.id]);
            let next = again.pick().expect("a pick").targets.first().copied();
            assert_eq!(
                next.map(|t| t.position),
                Some(1 - missed.position),
                "nothing was booked, so only the exclusion moves the pick"
            );
        }
        let (again, _) = decider.decide(&request, &workers, &[]);
        assert_eq!(
            again.pick().map(|p| p.targets.len()),
            Some(2),
            "both workers stay in the list once the exclusion is gone"
        );
    }

    #[test]
    fn the_commit_books_the_dispatch_so_the_next_decision_prefers_the_other_worker() {
        let decider = decider(&PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        });
        let (workers, _feed) = fed_pair(&decider);
        let request = HostRequest {
            prompt_tokens: 512,
            ..request()
        };
        let (first, lease) = decider.decide(&request, &workers, &[]);
        let taken = decider
            .commit(&lease, &first, Side::Single, |_| true)
            .expect("a target");
        // Twenty decisions: without the booking the two workers tie and the
        // picker draws the tie from the seed.
        for _ in 0..20 {
            let (second, _) = decider.decide(&request, &workers, &[]);
            let next = second.pick().expect("a pick").targets.first().copied();
            assert_eq!(
                next.map(|t| t.position),
                Some(1 - taken.position),
                "the committed worker carries the credit until a later report"
            );
        }
    }

    #[test]
    fn a_policy_the_engine_does_not_carry_is_refused_by_name() {
        let carried = PolicyConfig::Random;
        let sticky = RoutingKeyOverrideConfig::default();
        let Err(error) = Decider::from_configs(
            &PolicyConfig::ConsistentHashing,
            &carried,
            &carried,
            &sticky,
        ) else {
            panic!("a policy the engine does not carry is refused");
        };
        assert!(error.to_string().contains("consistent_hashing"), "{error}");
        let resident = PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::ResidentTokens { request_tokens: 64 },
        };
        assert!(preset_name(&resident).is_none());
        let Err(error) = Decider::from_configs(&carried, &resident, &carried, &sticky) else {
            panic!("the resident-tokens metric is refused");
        };
        assert!(error.to_string().contains("resident_tokens"), "{error}");
    }

    #[test]
    fn the_in_flight_metrics_of_power_of_two_decide_by_the_host_count() {
        let workers = pool(2);
        for _ in 0..5 {
            workers[0].increment_load();
        }
        for metric in [
            PowerOfTwoLoadMetric::Requests,
            PowerOfTwoLoadMetric::LeastRequests,
        ] {
            let decider = decider(&PolicyConfig::PowerOfTwo {
                load_metric: metric,
            });
            for _ in 0..4 {
                let (decision, _) = decider.decide(&request(), &workers, &[]);
                let first = decision.pick().expect("a pick").targets.first().copied();
                assert_eq!(first.map(|t| t.position), Some(1), "{metric:?}");
            }
        }
    }

    /// The request's tried URLs reach the engine as the not-tried filter's
    /// ids: a worker an earlier attempt dispatched to is passed over while
    /// another remains, and the filter steps aside once every worker was
    /// tried.
    #[test]
    fn a_tried_worker_is_passed_over_while_another_remains() {
        let decider = decider(&PolicyConfig::RoundRobin);
        let workers = pool(2);
        let tried = [workers[0].base_url().to_string()];
        let once_tried = HostRequest {
            tried: &tried,
            ..request()
        };
        for _ in 0..20 {
            let (decision, _) = decider.decide(&once_tried, &workers, &[]);
            let first = decision.pick().expect("a pick").targets.first().copied();
            assert_eq!(first.map(|t| t.position), Some(1), "{decision:?}");
        }
        let both = [
            workers[0].base_url().to_string(),
            workers[1].base_url().to_string(),
        ];
        let all_tried = HostRequest {
            tried: &both,
            ..request()
        };
        let (decision, _) = decider.decide(&all_tried, &workers, &[]);
        assert!(
            decision.pick().is_some(),
            "the not-tried filter is soft: {decision:?}"
        );
    }

    /// Three workers tell the two in-flight metrics apart: `least_requests`
    /// takes the emptiest worker for every seed, `requests` samples two and
    /// never takes the fullest.
    #[test]
    fn least_requests_takes_the_emptiest_of_three_and_requests_never_the_fullest() {
        let workers = pool(3);
        for _ in 0..5 {
            workers[0].increment_load();
        }
        for _ in 0..3 {
            workers[1].increment_load();
        }
        let least = decider(&PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::LeastRequests,
        });
        let sampled = decider(&PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::Requests,
        });
        for _ in 0..20 {
            let (decision, _) = least.decide(&request(), &workers, &[]);
            let first = decision.pick().expect("a pick").targets.first().copied();
            assert_eq!(first.map(|t| t.position), Some(2), "{decision:?}");
            let (decision, _) = sampled.decide(&request(), &workers, &[]);
            let first = decision.pick().expect("a pick").targets.first().copied();
            assert_ne!(first.map(|t| t.position), Some(0), "{decision:?}");
        }
    }

    #[test]
    fn a_same_url_replacement_gets_a_new_id() {
        let a: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w:8000").build());
        let b: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w:8000").build());
        assert_ne!(worker_id(a.as_ref()), worker_id(b.as_ref()));
        assert_eq!(worker_id(a.as_ref()), worker_id(a.as_ref()));
    }

    /// The legacy policy's case through the host: an idle worker whose cache
    /// is full of reclaimable prefixes (usage 1.0, its running requests
    /// holding none) beats a busy half-full one under the pressure term, and
    /// a report without the split falls back to the usage it has.
    #[test]
    fn a_reclaimable_prefix_cache_does_not_make_an_idle_worker_busy_through_the_host() {
        let decider = decider(&PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.15,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        });
        let workers = pool(2);
        let busy = || {
            let mut busy = one_rank_load(0, 0.5, 2000.0);
            busy.loads[0].active_token_usage = Some(0.5);
            busy
        };
        let mut idle = one_rank_load(0, 1.0, 2000.0);
        idle.loads[0].num_running_reqs = 0;
        idle.loads[0].active_token_usage = Some(0.0);
        let (tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
            (workers[0].url().to_string(), idle),
            (workers[1].url().to_string(), busy()),
        ]));
        decider.set_load_receiver(Some(rx));
        let leader = || {
            let (decision, _) = decider.decide(&request(), &workers, &[]);
            decision
                .pick()
                .expect("a pick")
                .targets
                .first()
                .map(|t| t.position)
        };
        assert_eq!(
            leader(),
            Some(0),
            "the idle worker's reclaimable cache is no pressure"
        );
        // Without the split the usage stands in for the share: the first
        // worker now reads full and the half-full one leads.
        let mut unsplit = one_rank_load(0, 1.0, 2000.0);
        unsplit.loads[0].num_running_reqs = 0;
        tx.send(LoadSnapshot::from_loads_for_test(vec![
            (workers[0].url().to_string(), unsplit),
            (workers[1].url().to_string(), busy()),
        ]))
        .expect("the receiver lives");
        assert_eq!(
            leader(),
            Some(1),
            "a report without the split falls back to its usage"
        );
    }

    #[test]
    fn the_view_row_reads_the_load_feed_and_the_scorer_takes_the_idle_worker() {
        let decider = decider(&PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        });
        let workers = pool(2);
        let mut busy = one_rank_load(4096, 0.5, 2000.0);
        busy.loads[0].num_waiting_reqs = 3;
        busy.loads[0].active_token_usage = Some(0.25);
        busy.loads[0].max_running_requests = 16;
        busy.loads[0].num_resident_tokens = Some(777);
        let idle = one_rank_load(0, 0.1, 2000.0);
        let (_tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
            (workers[0].url().to_string(), busy),
            (workers[1].url().to_string(), idle),
        ]));
        decider.set_load_receiver(Some(rx));

        let snapshot = decider.snapshot().expect("the feed");
        let build = ViewBuild {
            snapshot: Some(&snapshot),
            now: Instant::now(),
            session_key: None,
        };
        let row = decider
            .candidate(&build, 0, &workers[0], true, 0)
            .load
            .expect("a row for a reported worker");
        assert_eq!(
            (row.running, row.queued, row.waiting_uncached_tokens),
            (1, 3, 4096)
        );
        assert_eq!((row.kv_usage_permille, row.active_kv_permille), (500, 250));
        assert_eq!(
            (row.live_rate, row.max_running, row.resident_tokens),
            (2000, 16, 777)
        );
        assert_eq!(row.source, SignalSource::Poll);

        for _ in 0..3 {
            let (decision, _) = decider.decide(&request(), &workers, &[]);
            let first = decision.pick().expect("a pick").targets.first().copied();
            assert_eq!(first.map(|t| t.position), Some(1), "the idle worker wins");
        }
    }

    /// A rank with no running request contributes no generation rate (its
    /// gauge is stale); a prefill reads its prefill throughput whatever runs.
    #[test]
    fn an_idle_rank_contributes_no_live_rate_until_a_request_runs() {
        let now = Instant::now();
        let mut report = one_rank_load(0, 0.1, 2000.0);
        report.loads[0].num_running_reqs = 0;
        assert_eq!(
            load_row(&report, WorkerType::Regular, now, now).live_rate,
            0
        );
        report.loads[0].num_running_reqs = 1;
        assert_eq!(
            load_row(&report, WorkerType::Regular, now, now).live_rate,
            2000
        );
        report.loads[0].num_running_reqs = 0;
        report.loads[0].prefill_throughput = Some(1500.0);
        assert_eq!(
            load_row(&report, WorkerType::Prefill, now, now).live_rate,
            1500
        );
    }

    /// The counts sum over the ranks; the KV fields average over them, the
    /// active share falling back to the token usage of a rank without one.
    #[test]
    fn the_counts_sum_and_the_kv_fields_average_over_the_ranks() {
        let now = Instant::now();
        let mut report = one_rank_load(0, 0.2, 2000.0);
        let mut second = report.loads[0].clone();
        second.dp_rank = 1;
        second.token_usage = 0.8;
        second.active_token_usage = Some(0.4);
        report.loads.push(second);
        let row = load_row(&report, WorkerType::Regular, now, now);
        assert_eq!((row.kv_usage_permille, row.active_kv_permille), (500, 300));
        assert_eq!((row.running, row.live_rate), (2, 4000));
    }

    /// A prefill's drain is unknown unless the report is complete and every
    /// rank states a rate: one rateless rank or a partial report reads zero
    /// (the fleet's nominal rate at the scorer), as the policies read it.
    #[test]
    fn a_partial_or_rateless_prefill_report_leaves_the_drain_unknown() {
        let now = Instant::now();
        let mut report = one_rank_load(0, 0.1, 2000.0);
        report.loads[0].prefill_throughput = Some(5000.0);
        let mut second = report.loads[0].clone();
        second.dp_rank = 1;
        second.prefill_throughput = None;
        report.loads.push(second);
        report.dp_rank_count = 2;
        assert_eq!(
            load_row(&report, WorkerType::Prefill, now, now).live_rate,
            0
        );
        report.loads[1].prefill_throughput = Some(3000.0);
        assert_eq!(
            load_row(&report, WorkerType::Prefill, now, now).live_rate,
            8000
        );
        report.dp_rank_count = 4;
        assert_eq!(
            load_row(&report, WorkerType::Prefill, now, now).live_rate,
            0
        );
        assert_eq!(
            load_row(&report, WorkerType::Regular, now, now).live_rate,
            4000,
            "a regular worker's drain is its ranks' generation rate"
        );
    }

    /// The two in-flight metrics keep the policies' two stages: the fewest
    /// in flight first, the expected wait among equal counts. Equal counts
    /// over a busy and an idle report take the idle worker on every seed;
    /// a lower count beats a lighter load.
    #[test]
    fn equal_counts_fall_to_the_expected_wait_and_a_lower_count_wins_over_load() {
        let workers = pool(2);
        let (_tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
            (
                workers[0].url().to_string(),
                one_rank_load(4096, 0.5, 2000.0),
            ),
            (workers[1].url().to_string(), one_rank_load(0, 0.1, 2000.0)),
        ]));
        let request = HostRequest {
            prompt_tokens: 512,
            ..request()
        };
        for metric in [
            PowerOfTwoLoadMetric::Requests,
            PowerOfTwoLoadMetric::LeastRequests,
        ] {
            let decider = decider(&PolicyConfig::PowerOfTwo {
                load_metric: metric,
            });
            decider.set_load_receiver(Some(rx.clone()));
            for _ in 0..20 {
                let (decision, _) = decider.decide(&request, &workers, &[]);
                let first = decision.pick().expect("a pick").targets.first().copied();
                assert_eq!(
                    first.map(|t| t.position),
                    Some(1),
                    "{metric:?}: equal counts, the idle report wins"
                );
            }
        }
        // The idle worker now carries the higher count: the count decides.
        workers[1].increment_load();
        workers[1].increment_load();
        let decider = decider(&PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::LeastRequests,
        });
        decider.set_load_receiver(Some(rx));
        for _ in 0..20 {
            let (decision, _) = decider.decide(&request, &workers, &[]);
            let first = decision.pick().expect("a pick").targets.first().copied();
            assert_eq!(
                first.map(|t| t.position),
                Some(0),
                "the lower count beats the lighter load"
            );
        }
    }

    /// Three workers at equal counts with waits far apart: the count-first
    /// pick falls to the wait and takes the idle worker on every seed under
    /// `least_requests`; `requests` never takes the heaviest.
    #[test]
    fn equal_counts_over_three_workers_still_take_the_idle_one() {
        let workers = pool(3);
        let (_tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
            (workers[0].url().to_string(), one_rank_load(0, 0.1, 2000.0)),
            (
                workers[1].url().to_string(),
                one_rank_load(1000, 0.1, 2000.0),
            ),
            (
                workers[2].url().to_string(),
                one_rank_load(20_000, 0.1, 2000.0),
            ),
        ]));
        let request = HostRequest {
            prompt_tokens: 512,
            ..request()
        };
        let least = decider(&PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::LeastRequests,
        });
        least.set_load_receiver(Some(rx.clone()));
        let sampled = decider(&PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::Requests,
        });
        sampled.set_load_receiver(Some(rx));
        for _ in 0..20 {
            let (decision, _) = least.decide(&request, &workers, &[]);
            let first = decision.pick().expect("a pick").targets.first().copied();
            assert_eq!(
                first.map(|t| t.position),
                Some(0),
                "the idle worker at five percent of the span: {decision:?}"
            );
            let (decision, _) = sampled.decide(&request, &workers, &[]);
            let first = decision.pick().expect("a pick").targets.first().copied();
            assert_ne!(first.map(|t| t.position), Some(2), "{decision:?}");
        }
    }

    /// The review's wide spread: counts 0, 1 and 2000 with the idle worker's
    /// wait the worst. The count decides before any wait: `least_requests`
    /// takes the idle worker on every seed, `requests` never leads a pair
    /// with the busiest worker.
    #[test]
    fn a_strictly_lower_count_wins_whatever_the_wait_span() {
        let workers = pool(3);
        workers[1].increment_load();
        for _ in 0..2000 {
            workers[2].increment_load();
        }
        let (_tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
            (
                workers[0].url().to_string(),
                one_rank_load(40_000, 0.1, 2000.0),
            ),
            (workers[1].url().to_string(), one_rank_load(0, 0.1, 2000.0)),
            (
                workers[2].url().to_string(),
                one_rank_load(20_000, 0.1, 2000.0),
            ),
        ]));
        let request = HostRequest {
            prompt_tokens: 512,
            ..request()
        };
        let least = decider(&PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::LeastRequests,
        });
        least.set_load_receiver(Some(rx.clone()));
        let sampled = decider(&PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::Requests,
        });
        sampled.set_load_receiver(Some(rx));
        for _ in 0..20 {
            let (decision, _) = least.decide(&request, &workers, &[]);
            let first = decision.pick().expect("a pick").targets.first().copied();
            assert_eq!(
                first.map(|t| t.position),
                Some(0),
                "the idle count beats the lightest wait: {decision:?}"
            );
            let (decision, _) = sampled.decide(&request, &workers, &[]);
            let first = decision.pick().expect("a pick").targets.first().copied();
            assert_ne!(first.map(|t| t.position), Some(2), "{decision:?}");
        }
    }

    /// Two workers where the samplers disagree: the lower-count worker
    /// carries the heavier wait (40,000 waiting tokens against an idle
    /// report). `requests` leads the pair with the lower count on every
    /// seed, as the count-first sampler does; the plain sampler would take
    /// the lighter wait. The other worker is the alternate.
    #[test]
    fn the_sampled_in_flight_metric_leads_its_pair_with_the_lower_count() {
        let workers = pool(2);
        workers[0].increment_load();
        workers[1].increment_load();
        workers[1].increment_load();
        let (_tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
            (
                workers[0].url().to_string(),
                one_rank_load(40_000, 0.1, 2000.0),
            ),
            (workers[1].url().to_string(), one_rank_load(0, 0.1, 2000.0)),
        ]));
        let request = HostRequest {
            prompt_tokens: 512,
            ..request()
        };
        let sampled = decider(&PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::Requests,
        });
        sampled.set_load_receiver(Some(rx));
        for _ in 0..20 {
            let (decision, _) = sampled.decide(&request, &workers, &[]);
            let positions: Vec<u32> = decision
                .pick()
                .expect("a pick")
                .targets
                .iter()
                .map(|t| t.position)
                .collect();
            assert_eq!(
                positions,
                [0, 1],
                "the lower count leads the pair whatever its wait: {decision:?}"
            );
        }
    }

    /// The sampled half of the `requests` mapping: three workers with
    /// counts 5, 3 and 0 under idle reports. A drawn pair of the two fuller
    /// workers comes up about one decision in three and the middle worker
    /// leads it, so 64 decisions lead with position 1 at least once (the
    /// best form, `count-first`, would lead with the idle worker every
    /// time); the fullest worker never leads a pair.
    #[test]
    fn the_sampled_in_flight_metric_sometimes_leads_with_a_drawn_middle_worker() {
        let workers = pool(3);
        for _ in 0..5 {
            workers[0].increment_load();
        }
        for _ in 0..3 {
            workers[1].increment_load();
        }
        let (_tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(
            workers
                .iter()
                .map(|w| (w.url().to_string(), one_rank_load(0, 0.1, 2000.0)))
                .collect(),
        ));
        let request = HostRequest {
            prompt_tokens: 512,
            ..request()
        };
        let sampled = decider(&PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::Requests,
        });
        sampled.set_load_receiver(Some(rx));
        let mut leads = Vec::new();
        for _ in 0..64 {
            let (decision, _) = sampled.decide(&request, &workers, &[]);
            let first = decision.pick().expect("a pick").targets.first().copied();
            leads.extend(first.map(|t| t.position));
        }
        assert!(
            !leads.contains(&0),
            "the fullest worker never leads a pair: {leads:?}"
        );
        assert!(
            leads.contains(&1),
            "requests samples two, so the middle worker leads a drawn pair: {leads:?}"
        );
    }

    /// The decode side reads the request's tried URLs and the exclusions as
    /// the prefill side does: a tried or excluded decode is passed over
    /// while a partner remains.
    #[test]
    fn a_pair_decision_passes_over_a_tried_or_excluded_decode() {
        let decider = decider(&PolicyConfig::RoundRobin);
        let prefills = vec![worker_of("http://p0:8000", WorkerType::Prefill)];
        let decodes = [
            worker_of("http://d0:8000", WorkerType::Decode),
            worker_of("http://d1:8000", WorkerType::Decode),
        ];
        let partners = |_: usize| -> Vec<Arc<dyn Worker>> { decodes.to_vec() };
        let tried = [decodes[0].base_url().to_string()];
        let after_d0 = HostRequest {
            phase: Phase::Prefill,
            tried: &tried,
            ..request()
        };
        let decode_of = |decision: &Decision| {
            decision
                .pick()
                .expect("a pair")
                .targets
                .first()
                .and_then(|t| t.pair)
                .map(|p| p.decode_position)
        };
        for _ in 0..20 {
            let (decision, _, _) =
                decider.decide_pair(&after_d0, &prefills, &|_| true, &partners, false, &[]);
            assert_eq!(
                decode_of(&decision),
                Some(1),
                "the tried decode: {decision:?}"
            );
        }
        let excluded = [worker_id(decodes[0].as_ref())];
        let fresh = HostRequest {
            phase: Phase::Prefill,
            ..request()
        };
        for _ in 0..20 {
            let (decision, _, _) =
                decider.decide_pair(&fresh, &prefills, &|_| true, &partners, false, &excluded);
            assert_eq!(
                decode_of(&decision),
                Some(1),
                "the excluded decode: {decision:?}"
            );
        }
    }

    #[test]
    fn a_pair_decision_pairs_a_prefill_with_one_of_its_partners_and_books_both_sides() {
        let decider = decider(&PolicyConfig::RoundRobin);
        let prefills = vec![
            worker_of("http://p0:8000", WorkerType::Prefill),
            worker_of("http://p1:8000", WorkerType::Prefill),
        ];
        let decodes = [
            worker_of("http://d0:8000", WorkerType::Decode),
            worker_of("http://d1:8000", WorkerType::Decode),
            worker_of("http://d2:8000", WorkerType::Decode),
        ];
        // p0 partners with d0 alone; p1 with d1 and d2.
        let partners = |i: usize| -> Vec<Arc<dyn Worker>> {
            match i {
                0 => vec![decodes[0].clone()],
                _ => vec![decodes[1].clone(), decodes[2].clone()],
            }
        };
        let request = HostRequest {
            phase: Phase::Prefill,
            ..request()
        };
        for _ in 0..4 {
            let (decision, lease, rows) =
                decider.decide_pair(&request, &prefills, &|_| true, &partners, false, &[]);
            let target = decider
                .commit(&lease, &decision, Side::Prefill, |_| true)
                .expect("a pair");
            let pair = target.pair.expect("the decode side");
            let decode = &rows[pair.decode_position as usize];
            assert!(
                partners(target.position as usize)
                    .iter()
                    .any(|w| w.url() == decode.url()),
                "the decode is a partner of the chosen prefill"
            );
            assert_eq!(pair.decode_id, worker_id(decode.as_ref()));
        }
    }

    /// A pair commit books the decode side too: under least-load on both
    /// sides with equal partner reports, the next pair takes the other decode.
    #[test]
    fn a_pair_commit_books_the_decode_side_so_the_next_pair_takes_the_other_decode() {
        let decider = decider(&PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        });
        let prefills = vec![worker_of("http://p0:8000", WorkerType::Prefill)];
        let decodes = [
            worker_of("http://d0:8000", WorkerType::Decode),
            worker_of("http://d1:8000", WorkerType::Decode),
        ];
        let idle = || one_rank_load(0, 0.1, 2000.0);
        let (_tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
            (prefills[0].url().to_string(), idle()),
            (decodes[0].url().to_string(), idle()),
            (decodes[1].url().to_string(), idle()),
        ]));
        decider.set_load_receiver(Some(rx));
        let partners = |_: usize| -> Vec<Arc<dyn Worker>> { decodes.to_vec() };
        let request = HostRequest {
            phase: Phase::Prefill,
            prompt_tokens: 512,
            ..request()
        };
        let (first, lease, _) =
            decider.decide_pair(&request, &prefills, &|_| true, &partners, false, &[]);
        let taken = decider
            .commit(&lease, &first, Side::Prefill, |_| true)
            .expect("a pair");
        let booked = taken.pair.expect("the decode side").decode_position;
        // Twenty decisions: without the booking the two decodes tie and the
        // picker draws the tie from the seed, so one or three decisions
        // could agree by chance.
        for _ in 0..20 {
            let (next, _, _) =
                decider.decide_pair(&request, &prefills, &|_| true, &partners, false, &[]);
            let pair = next
                .pick()
                .expect("a pair")
                .targets
                .first()
                .and_then(|t| t.pair);
            assert_eq!(
                pair.map(|p| p.decode_position),
                Some(1 - booked),
                "the booked decode carries the credit: {next:?}"
            );
        }
    }

    #[test]
    fn a_prefill_without_capacity_is_passed_over() {
        let decider = decider(&PolicyConfig::RoundRobin);
        let prefills = vec![
            worker_of("http://p0:8000", WorkerType::Prefill),
            worker_of("http://p1:8000", WorkerType::Prefill),
        ];
        let decode = worker_of("http://d0:8000", WorkerType::Decode);
        let partners = |_: usize| -> Vec<Arc<dyn Worker>> { vec![decode.clone()] };
        let full = prefills[0].url().to_string();
        let has_capacity = |w: &Arc<dyn Worker>| w.url() != full;
        let request = HostRequest {
            phase: Phase::Prefill,
            ..request()
        };
        for _ in 0..3 {
            let (decision, _, _) =
                decider.decide_pair(&request, &prefills, &has_capacity, &partners, false, &[]);
            let first = decision.pick().expect("a pair").targets.first().copied();
            assert_eq!(first.map(|t| t.position), Some(1));
        }
    }

    /// A request of unknown size is booked at the mean prefill the default
    /// profile's least-load scorer prices it at (the lease's contract):
    /// `least_load`'s setting, the scorer's default of 1024 under every
    /// `power_of_two` metric (the expected wait leads or breaks the count's
    /// ties); a counted prompt is booked as counted; a profile that prices no
    /// tokens books none.
    #[test]
    fn an_unknown_prompt_size_is_booked_at_the_mean_prefill() {
        let least_load = decider(&PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        });
        assert_eq!(least_load.prompt_tokens(None), 100);
        assert_eq!(least_load.prompt_tokens(Some(&[1, 2, 3])), 3);
        let expected_wait = decider(&PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::ExpectedWait,
        });
        assert_eq!(expected_wait.prompt_tokens(None), 1024);
        assert_eq!(expected_wait.prompt_tokens(Some(&[7])), 1);
        let requests = decider(&PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::Requests,
        });
        assert_eq!(
            requests.prompt_tokens(None),
            1024,
            "the count's ties break by expected wait, which prices tokens"
        );
        assert_eq!(decider(&PolicyConfig::RoundRobin).prompt_tokens(None), 0);
    }
    /// A commit under a lease decided before the worker's retirement books
    /// nothing on it, whether the worker was removed or replaced at its URL:
    /// the rows the retirement pruned do not come back through the late
    /// dispatch. A lease decided after the retirement books as usual, since
    /// the id may be a new handle's.
    #[test]
    fn a_commit_under_a_lease_older_than_the_workers_retirement_books_nothing() {
        let config = PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        };
        let retire_found = [
            |decider: &Decider, found: &dyn Worker| decider.on_worker_removed(found),
            |decider: &Decider, found: &dyn Worker| {
                let replacement = pool(1);
                decider.on_worker_replaced(found, replacement[0].as_ref());
            },
        ];
        for retire in retire_found {
            let decider = decider(&config);
            let (workers, _reports) = fed_pair(&decider);
            let request = HostRequest {
                prompt_tokens: decider.prompt_tokens(None),
                ..request()
            };
            let book = |position: u32| {
                let (decision, lease) = decider.decide(&request, &workers, &[]);
                decider
                    .commit(&lease, &decision, Side::Single, |t| t.position == position)
                    .expect("the wanted worker is booked");
            };
            let leader = || {
                let (decision, _) = decider.decide(&request, &workers, &[]);
                decision
                    .pick()
                    .expect("a pick")
                    .targets
                    .iter()
                    .next()
                    .map(|t| t.position)
            };
            // Two leases decided while the found worker stands...
            let stale = [
                decider.decide(&request, &workers, &[]),
                decider.decide(&request, &workers, &[]),
            ];
            // ...its retirement...
            retire(&decider, workers[0].as_ref());
            // ...and their commits after it: the dispatches go, nothing is booked.
            for (decision, lease) in &stale {
                decider
                    .commit(lease, decision, Side::Single, |t| t.position == 0)
                    .expect("the dispatch itself proceeds");
            }
            book(1);
            assert_eq!(
                leader(),
                Some(0),
                "the retired handle reads idle: the stale leases booked nothing"
            );
            // A lease decided after the retirement books under the same id.
            book(0);
            book(0);
            assert_eq!(
                leader(),
                Some(1),
                "a lease decided after the retirement books as usual"
            );
        }
    }
    /// A commit under a lease older than the oldest retirement the record
    /// still names books nothing either: the record forgets a handle past
    /// its cap, and a lease that old may name one it forgot. The dispatch
    /// itself still goes, and a lease decided after books as usual.
    #[test]
    fn a_commit_under_a_lease_older_than_the_oldest_remembered_retirement_books_nothing() {
        let config = PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        };
        let decider = decider(&config);
        let (workers, _reports) = fed_pair(&decider);
        let request = HostRequest {
            prompt_tokens: decider.prompt_tokens(None),
            ..request()
        };
        let book = |position: u32| {
            let (decision, lease) = decider.decide(&request, &workers, &[]);
            decider
                .commit(&lease, &decision, Side::Single, |t| t.position == position)
                .expect("the wanted worker is booked");
        };
        let leader = || {
            let (decision, _) = decider.decide(&request, &workers, &[]);
            decision
                .pick()
                .expect("a pick")
                .targets
                .iter()
                .next()
                .map(|t| t.position)
        };
        // Two leases decided while the found worker stands...
        let stale = [
            decider.decide(&request, &workers, &[]),
            decider.decide(&request, &workers, &[]),
        ];
        // ...its retirement, then enough others' that its entry leaves the
        // record (the handles stay alive, so no id is reused meanwhile)...
        decider.on_worker_removed(workers[0].as_ref());
        let churn = pool(RETIRED_KEPT);
        for gone in &churn {
            decider.on_worker_removed(gone.as_ref());
        }
        // ...and the stale commits after it: the dispatches go, nothing is booked.
        for (decision, lease) in &stale {
            decider
                .commit(lease, decision, Side::Single, |t| t.position == 0)
                .expect("the dispatch itself proceeds");
        }
        book(1);
        assert_eq!(
            leader(),
            Some(0),
            "the forgotten handle reads idle: the stale leases booked nothing"
        );
        // A lease decided after the churn books under the same id.
        book(0);
        book(0);
        assert_eq!(
            leader(),
            Some(1),
            "a lease decided after the oldest remembered retirement books as usual"
        );
    }
    /// The record keeps the bound of the entry it drops at the cap as its
    /// watermark: a lease below it predates whatever handle it names, a
    /// lease at or above it is judged by the handle's own bound alone.
    #[test]
    fn the_retired_record_keeps_a_watermark_past_its_cap() {
        let mut retired = Retired::default();
        retired.retire(WorkerId(0), 5);
        for id in 1..=u64::try_from(RETIRED_KEPT).expect("a small cap") {
            retired.retire(WorkerId(id), 10);
        }
        assert!(
            retired.predates(WorkerId(0), LeaseId(4)),
            "below the watermark: older than every remembered retirement"
        );
        assert!(
            !retired.predates(WorkerId(0), LeaseId(5)),
            "at the watermark, the handle forgotten: books as a new handle's"
        );
        assert!(
            retired.predates(WorkerId(1), LeaseId(9)),
            "a remembered handle keeps its own bound"
        );
        assert!(!retired.predates(WorkerId(1), LeaseId(10)));
    }
    /// A handle retired again moves to the back of the order with its new
    /// bound, so the entry the cap drops is always the oldest retirement
    /// still named and the watermark never jumps past a remembered bound.
    #[test]
    fn a_handle_retired_again_moves_behind_the_entries_it_once_preceded() {
        let mut retired = Retired::default();
        retired.retire(WorkerId(0), 5);
        retired.retire(WorkerId(1), 6);
        retired.retire(WorkerId(0), 7);
        // 1,023 more make 1,025: the cap drops one entry, the front, which
        // is 1's since 0 moved behind it.
        for id in 2..=u64::try_from(RETIRED_KEPT).expect("a small cap") {
            retired.retire(WorkerId(id), 10);
        }
        // An id the record never saw is judged by the watermark alone.
        let unknown = WorkerId(u64::MAX);
        assert!(
            !retired.predates(unknown, LeaseId(6)),
            "the watermark is the dropped entry's bound of 6, not the re-retired handle's 7"
        );
        assert!(retired.predates(unknown, LeaseId(5)));
        assert!(
            retired.predates(WorkerId(0), LeaseId(6)),
            "the re-retired handle is still named, under its new bound"
        );
        assert!(!retired.predates(WorkerId(0), LeaseId(7)));
    }
    /// A retirement that lands while a decision runs, after the view read
    /// the pool and before the engine decided, outdates that lease too: the
    /// lease is numbered before the view is built.
    #[test]
    fn a_commit_under_a_lease_whose_decision_saw_the_retirement_books_nothing() {
        let config = PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        };
        let decider = decider(&config);
        let (workers, _reports) = fed_pair(&decider);
        let leader = || {
            let (decision, _) = decider.decide(&request(), &workers, &[]);
            decision
                .pick()
                .expect("a pick")
                .targets
                .iter()
                .next()
                .map(|t| t.position)
        };
        // The capacity check runs inside the view loop, before the engine
        // decides: the found worker retires there, once.
        let retired = std::cell::Cell::new(false);
        let retire_during_the_view = |_: &Arc<dyn Worker>| {
            if !retired.replace(true) {
                decider.on_worker_removed(workers[0].as_ref());
            }
            true
        };
        let request_during = HostRequest {
            prompt_tokens: decider.prompt_tokens(None),
            has_capacity: Some(&retire_during_the_view),
            ..request()
        };
        let (decision, lease) = decider.decide(&request_during, &workers, &[]);
        assert!(retired.get(), "the retirement landed during the decision");
        decider
            .commit(&lease, &decision, Side::Single, |t| t.position == 0)
            .expect("the dispatch itself proceeds");
        // One small booking on the other worker: the retired handle, idle,
        // still leads; the stale lease's mean-size booking would have lost it.
        let small = HostRequest {
            prompt_tokens: 1,
            ..request()
        };
        let (decision, lease) = decider.decide(&small, &workers, &[]);
        decider
            .commit(&lease, &decision, Side::Single, |t| t.position == 1)
            .expect("the other worker is booked");
        assert_eq!(
            leader(),
            Some(0),
            "the retired handle reads idle: a lease numbered before the view booked nothing"
        );
    }
    /// A retirement that lands between a router's capture of its candidates
    /// and its decision outdates the lease when the router drew the request's
    /// sequence first: the captured slice still holds the retired handle, the
    /// decision and the one re-decision over that slice run under the drawn
    /// sequence, and neither commit books on it.
    #[test]
    fn a_commit_under_a_sequence_drawn_before_the_retirement_books_nothing() {
        let config = PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        };
        let decider = decider(&config);
        let (workers, _reports) = fed_pair(&decider);
        let leader = || {
            let (decision, _) = decider.decide(&request(), &workers, &[]);
            decision
                .pick()
                .expect("a pick")
                .targets
                .iter()
                .next()
                .map(|t| t.position)
        };
        // The router's order: the sequence, then the candidate slice (both
        // handles), then the first worker's retirement lands, then the
        // decision over the stale slice.
        let sequence = decider.begin();
        let captured = workers.clone();
        decider.on_worker_removed(workers[0].as_ref());
        let under_sequence = HostRequest {
            prompt_tokens: decider.prompt_tokens(None),
            sequence: Some(sequence),
            ..request()
        };
        let (decision, lease) = decider.decide(&under_sequence, &captured, &[]);
        assert_eq!(
            lease.id, sequence,
            "the decision runs under the drawn sequence"
        );
        decider
            .commit(&lease, &decision, Side::Single, |t| t.position == 0)
            .expect("the dispatch itself proceeds");
        // The one re-decision over the same slice, the live worker excluded
        // as a commit miss would exclude it, keeps the sequence.
        let live = worker_id(workers[1].as_ref());
        let (again, lease) = decider.decide(&under_sequence, &captured, &[live]);
        assert_eq!(
            lease.id, sequence,
            "the re-decision keeps the drawn sequence"
        );
        decider
            .commit(&lease, &again, Side::Single, |t| t.position == 0)
            .expect("the re-decision's dispatch proceeds too");
        // One small booking on the other worker: the retired handle, idle,
        // still leads; either stale booking at the mean size would have lost it.
        let small = HostRequest {
            prompt_tokens: 1,
            ..request()
        };
        let (decision, lease) = decider.decide(&small, &workers, &[]);
        decider
            .commit(&lease, &decision, Side::Single, |t| t.position == 1)
            .expect("the other worker is booked");
        assert_eq!(
            leader(),
            Some(0),
            "the retired handle reads idle: no commit under the drawn sequence booked on it"
        );
    }

    /// The pair path under a sequence drawn before the capture: the router
    /// draws, captures the prefills and the partners, a decode's retirement
    /// lands, then the pair decision over the stale partners runs under the
    /// drawn sequence and its commit books nothing on the retired decode. A
    /// pair request without a drawn sequence is numbered afresh.
    #[test]
    fn a_pair_commit_under_a_sequence_drawn_before_the_retirement_books_nothing() {
        let decider = decider(&PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        });
        let prefills = vec![worker_of("http://p0:8000", WorkerType::Prefill)];
        let decodes = [
            worker_of("http://d0:8000", WorkerType::Decode),
            worker_of("http://d1:8000", WorkerType::Decode),
        ];
        let idle = || one_rank_load(0, 0.1, 2000.0);
        let (_tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
            (prefills[0].url().to_string(), idle()),
            (decodes[0].url().to_string(), idle()),
            (decodes[1].url().to_string(), idle()),
        ]));
        decider.set_load_receiver(Some(rx));
        let partners = |_: usize| -> Vec<Arc<dyn Worker>> { decodes.to_vec() };
        let pair_request = |prompt_tokens| HostRequest {
            phase: Phase::Prefill,
            prompt_tokens,
            ..request()
        };
        let leading_decode = || {
            let (decision, _, _) = decider.decide_pair(
                &pair_request(512),
                &prefills,
                &|_| true,
                &partners,
                false,
                &[],
            );
            decision
                .pick()
                .expect("a pair")
                .targets
                .first()
                .and_then(|t| t.pair)
                .map(|p| p.decode_position)
        };
        // One small booking on the second decode first, so the first one
        // leads strictly and the pair below takes it.
        let (decision, lease, _) = decider.decide_pair(
            &pair_request(1),
            &prefills,
            &|_| true,
            &partners,
            false,
            &[],
        );
        decider
            .commit(&lease, &decision, Side::Prefill, |t| {
                t.pair.is_some_and(|p| p.decode_position == 1)
            })
            .expect("the second decode is booked");
        assert_eq!(
            leading_decode(),
            Some(0),
            "the first decode leads while idle"
        );
        // The router's order: the sequence, then the captures (the partners
        // still list the first decode), then its retirement lands, then the
        // decision over the stale partners.
        let sequence = decider.begin();
        decider.on_worker_removed(decodes[0].as_ref());
        let under_sequence = HostRequest {
            sequence: Some(sequence),
            ..pair_request(512)
        };
        let (decision, lease, _) =
            decider.decide_pair(&under_sequence, &prefills, &|_| true, &partners, false, &[]);
        assert_eq!(
            lease.id, sequence,
            "the pair decision runs under the drawn sequence"
        );
        let taken = decider
            .commit(&lease, &decision, Side::Prefill, |_| true)
            .expect("the dispatch itself proceeds");
        assert_eq!(
            taken.pair.map(|p| p.decode_position),
            Some(0),
            "the pair took the retired decode from the stale partners"
        );
        // The retired decode still leads: the request's booking would have
        // lost it to the one-token booking on the other decode.
        assert_eq!(
            leading_decode(),
            Some(0),
            "the retired decode reads idle: no commit under the drawn sequence booked on it"
        );
        // A pair request with no drawn sequence is numbered afresh.
        let (_, fresh, _) = decider.decide_pair(
            &pair_request(512),
            &prefills,
            &|_| true,
            &partners,
            false,
            &[],
        );
        assert!(
            fresh.id.0 > sequence.0,
            "a pair request without a drawn sequence draws a new one: {:?} after {:?}",
            fresh.id,
            sequence
        );
    }

    /// A retirement that lands while a commit runs, after its check and
    /// before its booking, waits for the booking: the check and the booking
    /// hold the retirement record's lock together, so the row the booking
    /// adds is the row the retirement then prunes.
    #[test]
    fn a_retirement_during_a_commits_booking_waits_for_it() {
        let config = PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        };
        let decider = Arc::new(decider(&config));
        let (workers, _reports) = fed_pair(&decider);
        let leader = || {
            let (decision, _) = decider.decide(&request(), &workers, &[]);
            decision
                .pick()
                .expect("a pick")
                .targets
                .iter()
                .next()
                .map(|t| t.position)
        };
        // One small booking on the other worker: the found handle leads
        // while it is idle and loses the lead with one mean-size row.
        let small = HostRequest {
            prompt_tokens: 1,
            ..request()
        };
        let (decision, lease) = decider.decide(&small, &workers, &[]);
        decider
            .commit(&lease, &decision, Side::Single, |t| t.position == 1)
            .expect("the other worker is booked");
        assert_eq!(leader(), Some(0), "the found handle leads while idle");
        let request = HostRequest {
            prompt_tokens: decider.prompt_tokens(None),
            ..request()
        };
        let (decision, lease) = decider.decide(&request, &workers, &[]);
        // From between the check and the booking: the found worker's removal
        // on another thread, waited for briefly. Under the shared lock it
        // waits for the booking instead; released early, it lands first.
        let removal: Arc<Mutex<Option<std::thread::JoinHandle<()>>>> = Arc::new(Mutex::new(None));
        {
            let hooked = Arc::clone(&decider);
            let found = Arc::clone(&workers[0]);
            let removal = Arc::clone(&removal);
            *decider.between_check_and_booking.lock() = Some(Box::new(move || {
                let (done, landed) = mpsc::channel();
                let decider = Arc::clone(&hooked);
                let found = Arc::clone(&found);
                let handle = std::thread::spawn(move || {
                    decider.on_worker_removed(found.as_ref());
                    let _ = done.send(());
                });
                let _ = landed.recv_timeout(std::time::Duration::from_millis(200));
                *removal.lock() = Some(handle);
            }));
        }
        decider
            .commit(&lease, &decision, Side::Single, |t| t.position == 0)
            .expect("the dispatch itself proceeds");
        *decider.between_check_and_booking.lock() = None;
        if let Some(handle) = removal.lock().take() {
            handle.join().expect("the removal thread");
        }
        assert_eq!(
            leader(),
            Some(0),
            "the retired handle reads idle: the retirement waited for the booking it then pruned"
        );
    }

    /// The manual policy pins a key at its first placement and keeps it
    /// there; an unroutable owner hands the key to a failover, and the key
    /// returns to its first owner once that worker is back: the pin list
    /// remembers the oldest first, as the legacy manual policy's does.
    #[test]
    fn the_manual_policy_pins_a_key_and_recovers_to_its_first_owner() {
        let decider = decider(&manual(ManualAssignmentMode::MinLoad));
        let workers = pool(3);
        let owner = place(&decider, &workers, "s1");
        // A vacant key would go to a lighter worker.
        for _ in 0..3 {
            workers[owner].increment_load();
        }
        for _ in 0..4 {
            assert_eq!(place(&decider, &workers, "s1"), owner);
        }
        workers[owner].set_status(WorkerStatus::NotReady);
        let failover = place(&decider, &workers, "s1");
        assert_ne!(failover, owner);
        for _ in 0..3 {
            assert_eq!(place(&decider, &workers, "s1"), failover);
        }
        workers[owner].set_status(WorkerStatus::Ready);
        assert_eq!(
            place(&decider, &workers, "s1"),
            owner,
            "the key returns to its first owner"
        );
        assert_eq!(decider.engine().sessions().len(), 1);
    }

    /// Under the sticky override the pin follows the failover: once the first
    /// owner is back the key stays where it last ran, as the legacy override
    /// pins the newest worker first; a policy that would rotate an unpinned
    /// key keeps the pin.
    #[test]
    fn the_sticky_override_keeps_a_key_on_its_latest_worker() {
        let decider = overridden(
            &PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                assignment_mode: ManualAssignmentMode::Delegate,
                ..RoutingKeyOverrideConfig::default()
            },
        );
        let workers = pool(3);
        let owner = place(&decider, &workers, "s1");
        for _ in 0..4 {
            assert_eq!(
                place(&decider, &workers, "s1"),
                owner,
                "round robin would rotate an unpinned key"
            );
        }
        workers[owner].set_status(WorkerStatus::NotReady);
        let failover = place(&decider, &workers, "s1");
        assert_ne!(failover, owner);
        workers[owner].set_status(WorkerStatus::Ready);
        for _ in 0..3 {
            assert_eq!(
                place(&decider, &workers, "s1"),
                failover,
                "the latest pin leads"
            );
        }
    }

    /// The legacy override keeps an assigned pin's original first under
    /// `random`, `min_load` and `min_group` (its `assign_pin`), so the key
    /// returns to its first worker once that worker is back; only a
    /// delegated pick fronts the latest worker.
    #[test]
    fn the_override_recovers_to_the_first_worker_under_every_assignment_but_delegate() {
        for mode in [
            ManualAssignmentMode::Random,
            ManualAssignmentMode::MinLoad,
            ManualAssignmentMode::MinGroup,
        ] {
            let decider = overridden(
                &PolicyConfig::RoundRobin,
                RoutingKeyOverrideConfig {
                    assignment_mode: mode,
                    ..RoutingKeyOverrideConfig::default()
                },
            );
            let workers = pool(3);
            let owner = place(&decider, &workers, "s1");
            workers[owner].set_status(WorkerStatus::NotReady);
            let failover = place(&decider, &workers, "s1");
            assert_ne!(failover, owner, "{mode:?}");
            workers[owner].set_status(WorkerStatus::Ready);
            assert_eq!(
                place(&decider, &workers, "s1"),
                owner,
                "{mode:?}: the first pin leads again"
            );
        }
    }

    /// A pin remembers two workers: with its first two owners down the key
    /// takes a third, which replaces the oldest; with every worker back the
    /// key is on the second owner, the oldest the list still holds.
    #[test]
    fn a_manual_pin_remembers_its_two_latest_workers_oldest_first() {
        let decider = decider(&manual(ManualAssignmentMode::Random));
        let workers = pool(3);
        let first = place(&decider, &workers, "s1");
        workers[first].set_status(WorkerStatus::NotReady);
        let second = place(&decider, &workers, "s1");
        workers[second].set_status(WorkerStatus::NotReady);
        let third = place(&decider, &workers, "s1");
        assert_eq!(third, 3 - first - second);
        for worker in &workers {
            worker.set_status(WorkerStatus::Ready);
        }
        for _ in 0..3 {
            assert_eq!(
                place(&decider, &workers, "s1"),
                second,
                "the first owner left the two-slot list"
            );
        }
    }

    /// Without a key the manual policy places by its assignment mode:
    /// min_load takes the fewest requests in flight, min_group the fewest
    /// sessions in flight (a count the row carries only when a stage reads
    /// it), and nothing is pinned.
    #[test]
    fn without_a_key_the_manual_policy_places_by_its_assignment_mode() {
        let workers = pool(2);
        for _ in 0..3 {
            workers[0].increment_load();
        }
        workers[1].increment_routing_key_load("a");
        workers[1].increment_routing_key_load("b");
        let min_load = decider(&manual(ManualAssignmentMode::MinLoad));
        let min_group = decider(&manual(ManualAssignmentMode::MinGroup));
        assert!(min_group.engine().needs().active_sessions);
        assert!(!min_load.engine().needs().active_sessions);
        assert!(
            !min_load.engine().needs().session_in_flight,
            "no cap, so no per-key count is read"
        );
        let first = |decider: &Decider| {
            let (decision, _) = decider.decide(&request(), &workers, &[]);
            decision
                .pick()
                .and_then(|pick| pick.targets.first().map(|t| t.position))
        };
        for _ in 0..4 {
            assert_eq!(first(&min_load), Some(1), "three requests run on the first");
            assert_eq!(first(&min_group), Some(0), "two sessions run on the second");
        }
        assert!(min_load.engine().sessions().is_empty());
    }

    /// One key is one session per model: the same key under two models pins
    /// independently, as the legacy override namespaces its map by model.
    #[test]
    fn a_session_key_is_scoped_by_its_model() {
        let decider = overridden(
            &PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig::default(),
        );
        let workers = pool(2);
        let place_for = |model: &str| {
            let keyed = HostRequest {
                model,
                session_key: Some("s1"),
                ..HostRequest::default()
            };
            let (decision, lease) = decider.decide(&keyed, &workers, &[]);
            decider
                .commit(&lease, &decision, Side::Single, |_| true)
                .expect("a target")
                .position
        };
        let a = place_for("a");
        let b = place_for("b");
        assert_ne!(
            a, b,
            "round robin gave the second model's session the other worker"
        );
        for _ in 0..3 {
            assert_eq!(place_for("a"), a);
            assert_eq!(place_for("b"), b);
        }
        assert_eq!(decider.engine().sessions().len(), 2);
    }

    /// An owner over the in-flight cap: the key respills to a worker under
    /// the cap and the pin follows it; with every worker over the cap the key
    /// stays on its owner. The cap counts the host's in-flight requests under
    /// the raw key, as the load guard keeps them.
    #[test]
    fn a_saturated_owner_respills_the_key_to_a_worker_under_the_cap() {
        let decider = overridden(
            &PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                assignment_mode: ManualAssignmentMode::MinLoad,
                ..Default::default()
            },
        );
        assert!(decider.engine().needs().session_in_flight);
        let workers = pool(2);
        let owner = place(&decider, &workers, "s1");
        let other = 1 - owner;
        for _ in 0..2 {
            workers[owner].increment_routing_key_load("s1");
        }
        assert_eq!(
            place(&decider, &workers, "s1"),
            other,
            "two of the key's requests in flight on the owner"
        );
        workers[owner].decrement_routing_key_load("s1");
        workers[owner].decrement_routing_key_load("s1");
        assert_eq!(
            place(&decider, &workers, "s1"),
            other,
            "the pin followed the respill"
        );
        for worker in &workers {
            for _ in 0..2 {
                worker.increment_routing_key_load("s1");
            }
        }
        assert_eq!(
            place(&decider, &workers, "s1"),
            other,
            "no worker under the cap: the key stays"
        );
    }

    /// The legacy eviction thread stays off when either of its two settings
    /// is zero; the engine's rule keeps such entries for ever. The override's
    /// rule carries its cap and recovers to the latest pin; without it a
    /// policy other than manual pins nothing.
    #[test]
    fn a_zero_idle_or_sweep_setting_keeps_sessions_for_ever() {
        let off = RoutingKeyOverrideConfig::default();
        let rule = |max_idle_secs, eviction_interval_secs| {
            sessions_of(
                &PolicyConfig::Manual {
                    eviction_interval_secs,
                    max_idle_secs,
                    assignment_mode: ManualAssignmentMode::Random,
                },
                &off,
            )
            .expect("manual pins")
        };
        assert_eq!(
            (rule(0, 60).ttl_s, rule(600, 0).ttl_s),
            (u32::MAX, u32::MAX)
        );
        assert_eq!(rule(600, 60).inflight_cap, u16::MAX, "no cap");
        assert_eq!(rule(600, 60).sticky_header, MANUAL_SESSION_HEADER);
        assert_eq!((rule(600, 60).ttl_s, rule(600, 60).sweep_s), (600, 60));
        assert_eq!(rule(600, 60).recovery, Recovery::Original);
        assert!(sessions_of(&PolicyConfig::RoundRobin, &off).is_none());
        let on = RoutingKeyOverrideConfig {
            enabled: true,
            max_idle_secs: 0,
            ..Default::default()
        };
        let rule = sessions_of(&PolicyConfig::RoundRobin, &on).expect("the override pins");
        assert_eq!(
            (rule.ttl_s, rule.inflight_cap, rule.recovery),
            (u32::MAX, STICKY_INFLIGHT_CAP, Recovery::Latest)
        );
        let listed = RoutingKeyOverrideConfig {
            enabled: true,
            headers: vec![
                "X-Routing-Key".into(),
                "x-routing-key".into(),
                "x-smg-routing-key".into(),
            ],
            assignment_mode: ManualAssignmentMode::MinLoad,
            ..Default::default()
        };
        let rule = sessions_of(&PolicyConfig::RoundRobin, &listed).expect("the override pins");
        assert_eq!(
            rule.sticky_header, "x-routing-key",
            "the first header, lowercase"
        );
        assert_eq!(
            rule.headers,
            ["x-routing-key", "x-smg-routing-key"],
            "each name once after lowercasing"
        );
        assert!(
            Decider::from_configs(
                &PolicyConfig::RoundRobin,
                &PolicyConfig::RoundRobin,
                &PolicyConfig::RoundRobin,
                &listed
            )
            .is_ok(),
            "a repeated name is one header of the view"
        );
        let denied = RoutingKeyOverrideConfig {
            headers: vec!["x-api-key".into()],
            ..listed
        };
        let error = Decider::from_configs(
            &PolicyConfig::RoundRobin,
            &PolicyConfig::RoundRobin,
            &PolicyConfig::RoundRobin,
            &denied,
        )
        .expect_err("a denied header refuses the build")
        .to_string();
        assert!(
            error.contains("x-api-key") && error.contains("deny-list"),
            "{error}"
        );
        assert_eq!(
            rule.recovery,
            Recovery::Original,
            "an assigned pin keeps its original first"
        );
    }

    /// Under the override's `min_group` assignment a new key goes to the
    /// worker with the fewest sessions in flight, not the fewest requests.
    #[test]
    fn the_override_under_min_group_places_a_new_key_by_the_fewest_sessions() {
        let decider = overridden(
            &PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                assignment_mode: ManualAssignmentMode::MinGroup,
                ..Default::default()
            },
        );
        assert!(decider.engine().needs().active_sessions);
        let workers = pool(2);
        for _ in 0..3 {
            workers[1].increment_load();
        }
        workers[0].increment_routing_key_load("a");
        workers[0].increment_routing_key_load("b");
        for key in ["k1", "k2", "k3"] {
            assert_eq!(
                place(&decider, &workers, key),
                1,
                "two sessions run on the first worker, three requests on the second"
            );
        }
    }

    /// The per-key in-flight count is read into a row only under a rule
    /// with a cap: the manual policy's rows carry none, the override's carry
    /// the guard's count under the raw key.
    #[test]
    fn the_per_key_count_is_fed_only_under_a_rule_with_a_cap() {
        let workers = pool(1);
        workers[0].increment_routing_key_load("s1");
        let row = |decider: &Decider| {
            let build = ViewBuild {
                snapshot: None,
                now: Instant::now(),
                session_key: Some("s1"),
            };
            decider
                .candidate(&build, 0, &workers[0], true, 0)
                .session_in_flight
        };
        assert_eq!(row(&decider(&manual(ManualAssignmentMode::Random))), 0);
        assert_eq!(
            row(&overridden(
                &PolicyConfig::RoundRobin,
                RoutingKeyOverrideConfig::default()
            )),
            1
        );
    }

    /// The first target's position of every decision.
    fn positions(decided: &[(Decision, Lease)]) -> Vec<u32> {
        decided
            .iter()
            .map(|(decision, _)| {
                decision
                    .pick()
                    .expect("a pick")
                    .targets
                    .first()
                    .expect("a target")
                    .position
            })
            .collect()
    }

    /// The encode items of one request are decided over one view of the
    /// encoders, each by its content hash: equal hashes land on one encoder
    /// and the same keys decide the same way later. The row bits come from
    /// the request's predicates: when one encoder pairs with the prefill
    /// side every item lands on it, whatever its key's owner; when one
    /// encoder has a vision encoder every image item lands on it; an image
    /// item over encoders without one is shed by the plan's rule, a text
    /// item takes any encoder.
    #[test]
    fn encode_items_are_decided_by_their_keys_over_the_encoders_that_pair_with_the_prefill() {
        let round_robin = PolicyConfig::RoundRobin;
        let decider = Decider::from_configs_with_encode(
            &round_robin,
            &round_robin,
            &round_robin,
            Some(&PolicyConfig::ConsistentHashing),
            &RoutingKeyOverrideConfig::default(),
        )
        .expect("the encode side is carried");
        assert!(decider.serves_encode());
        assert!(!self::decider(&round_robin).serves_encode());
        let encoders: Vec<Arc<dyn Worker>> = (0..4)
            .map(|i| worker_of(&format!("http://e{i}:8000"), WorkerType::Encode))
            .collect();
        let every = |_: &Arc<dyn Worker>| true;
        let request = EncodeRequest {
            model: "m",
            pairs_with_prefill: &every,
            vision_capable: &every,
            images: 1,
        };
        // Eight keys: by the ring they spread over the four encoders.
        let keys: Vec<u64> = (0..8u64)
            .map(|i| fingerprint(format!("image-{}", i % 7).as_bytes()))
            .collect();
        let decided = decider.decide_encode_items(&request, &encoders, &keys, &[]);
        assert_eq!(decided.len(), 8);
        let targets: Vec<Target> = decided
            .iter()
            .map(|(decision, lease)| {
                decider
                    .commit(lease, decision, Side::Encode, |_| true)
                    .expect("an encoder")
            })
            .collect();
        assert_eq!(
            targets[0].position, targets[7].position,
            "equal hashes share an encoder"
        );
        let owners: std::collections::BTreeSet<u32> = targets.iter().map(|t| t.position).collect();
        assert!(
            owners.len() > 1,
            "the keys spread over the encoders: {owners:?}"
        );
        assert_eq!(decider.decisions(), 8, "one decision per item");
        let before: Vec<u32> = targets.iter().map(|t| t.position).collect();
        let again = decider.decide_encode_items(&request, &encoders, &keys, &[]);
        assert_eq!(positions(&again), before, "the same keys land the same way");

        // One encoder pairs with the prefill side: every item lands on it,
        // whatever its key's owner (the bit's hand-off from the predicate).
        let only_two = encoders[2].url().to_string();
        let pairs_two = |worker: &Arc<dyn Worker>| worker.url() == only_two;
        let paired = EncodeRequest {
            pairs_with_prefill: &pairs_two,
            ..request
        };
        let decided = decider.decide_encode_items(&paired, &encoders, &keys, &[]);
        assert_eq!(
            positions(&decided),
            vec![2; 8],
            "only the paired encoder is taken"
        );

        // One encoder has a vision encoder: every image item lands on it.
        let only_one = encoders[1].url().to_string();
        let sees_one = |worker: &Arc<dyn Worker>| worker.url() == only_one;
        let sighted = EncodeRequest {
            vision_capable: &sees_one,
            ..request
        };
        let decided = decider.decide_encode_items(&sighted, &encoders, &keys, &[]);
        assert_eq!(
            positions(&decided),
            vec![1; 8],
            "only the sighted encoder takes images"
        );

        let blind = |_: &Arc<dyn Worker>| false;
        let no_vision = EncodeRequest {
            vision_capable: &blind,
            ..request
        };
        let shed = decider.decide_encode_items(&no_vision, &encoders, &keys[..1], &[]);
        match &shed[0].0 {
            Decision::Shed(shed) => assert_eq!(shed.emptied_by, "vision"),
            other => panic!("an image over encoders without a vision encoder answered {other:?}"),
        }
        let text = EncodeRequest {
            images: 0,
            ..no_vision
        };
        let decided = decider.decide_encode_items(&text, &encoders, &keys[..1], &[]);
        assert!(
            decided[0].0.pick().is_some(),
            "a text item takes any encoder: {:?}",
            decided[0].0
        );
    }

    /// A commit that refused a decision's targets re-decides with them
    /// excluded: the excluded owner's key lands on another encoder, and the
    /// excluded id is never among the new targets.
    #[test]
    fn an_excluded_encoder_passes_its_key_to_the_ring_s_next_member() {
        let round_robin = PolicyConfig::RoundRobin;
        let decider = Decider::from_configs_with_encode(
            &round_robin,
            &round_robin,
            &round_robin,
            Some(&PolicyConfig::ConsistentHashing),
            &RoutingKeyOverrideConfig::default(),
        )
        .expect("the encode side is carried");
        let encoders: Vec<Arc<dyn Worker>> = (0..4)
            .map(|i| worker_of(&format!("http://e{i}:8000"), WorkerType::Encode))
            .collect();
        let pairs = |_: &Arc<dyn Worker>| true;
        let request = EncodeRequest {
            model: "m",
            pairs_with_prefill: &pairs,
            vision_capable: &pairs,
            images: 0,
        };
        let keys = [fingerprint(b"image-c")];
        let first = decider.decide_encode_items(&request, &encoders, &keys, &[]);
        let owner = *first[0]
            .0
            .pick()
            .expect("a pick")
            .targets
            .as_slice()
            .first()
            .expect("a target");
        let again = decider.decide_encode_items(&request, &encoders, &keys, &[owner.id]);
        let pick = again[0].0.pick().expect("a pick");
        let next = pick.targets.as_slice().first().expect("a target");
        assert_ne!(next.position, owner.position);
        assert!(pick.targets.as_slice().iter().all(|t| t.id != owner.id));
    }

    #[test]
    fn an_encode_policy_the_engine_does_not_carry_is_refused_by_name() {
        let carried = PolicyConfig::RoundRobin;
        let uncarried = PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        };
        let Err(error) = Decider::from_configs_with_encode(
            &carried,
            &carried,
            &carried,
            Some(&uncarried),
            &RoutingKeyOverrideConfig::default(),
        ) else {
            panic!("an encode policy outside the encode side's three is refused");
        };
        assert!(error.to_string().contains("least_load"), "{error}");
        assert_eq!(
            encode_preset_name(&PolicyConfig::ConsistentHashing),
            Some("consistent_hashing")
        );
        assert_eq!(encode_preset_name(&PolicyConfig::Random), Some("random"));
        assert!(
            preset_name(&PolicyConfig::ConsistentHashing).is_none(),
            "the regular path's keyed form waits for a later step"
        );
    }

    #[test]
    fn a_candidate_s_id_and_key_come_from_one_pass_over_the_url() {
        let workers = pool(1);
        let worker = workers[0].as_ref();
        let (id, key) = worker_identity(worker);
        assert_eq!(id, worker_id(worker), "the id under the store's rule");
        assert_eq!(key, fingerprint(worker.url().as_bytes()) as u32);
    }
}
