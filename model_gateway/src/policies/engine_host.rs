//! The gateway as host of the policy engine.
//!
//! The host builds the bounded view from the pool's live state (one routing
//! state read and one load-report lookup per candidate, into a thread-local
//! buffer), asks the engine for a decision outside its own locks, and commits
//! one target under them: the commit re-validates the target per kind and
//! walks the ordered list on a miss. The engine's accounting is fed at the
//! commit (`on_dispatched`), so a decision that never commits books nothing.

use std::{
    cell::RefCell,
    hash::{BuildHasher, Hash, Hasher, RandomState},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use openai_protocol::worker::WorkerType;
use parking_lot::RwLock;
use smg_policy::{
    spec::{BindingSpec, ProfileSpec, ScoreParams, ScoreSpec},
    Candidate, Constraints, Decision, Engine, Health, Lease, LeaseId, LeaseKind, Lifecycle,
    LoadReport, PairPools, Phase, PolicySpec, PoolFacts, RequestFacts, Role, Scratch, Side,
    SpecError, Target, WorkerId, API_VERSION,
};
use tracing::debug;

use crate::{
    config::PolicyConfig,
    routers::common::pd_admission,
    worker::{
        load_state::{LoadReceiver, LoadSnapshot},
        Worker,
    },
};

/// Why a configuration cannot be served by the engine in this build.
/// The host's per-worker capacity predicate a decision reads outside the
/// admission lock.
pub type CapacityCheck<'a> = &'a dyn Fn(&Arc<dyn Worker>) -> bool;

#[derive(Debug, thiserror::Error)]
pub enum EngineHostError {
    #[error("the policy engine does not carry the stages of {0:?} yet")]
    Unsupported(&'static str),
    #[error(transparent)]
    Spec(#[from] SpecError),
}

/// The preset a legacy policy configuration maps to in this build; `None`
/// for a policy whose stages the engine does not carry yet.
pub fn preset_name(config: &PolicyConfig) -> Option<&'static str> {
    Some(match config {
        PolicyConfig::Random => "random",
        PolicyConfig::RoundRobin => "round_robin",
        PolicyConfig::Passthrough => "passthrough",
        PolicyConfig::PowerOfTwo { .. } => "power_of_two",
        PolicyConfig::LeastLoad { .. } => "least_load",
        _ => return None,
    })
}

/// The profile of one legacy configuration, with its tunables carried over.
fn profile_of(config: &PolicyConfig) -> Result<ProfileSpec, EngineHostError> {
    let name = preset_name(config).ok_or_else(|| EngineHostError::Unsupported(config.name()))?;
    let mut profile = PolicySpec::preset(name)?
        .profiles
        .remove(name)
        .ok_or(SpecError::Empty)?;
    if let PolicyConfig::LeastLoad {
        kv_pressure_weight,
        mean_prefill_tokens,
        default_throughput,
        ..
    } = config
    {
        let mut params = std::collections::BTreeMap::new();
        params.insert(
            "least-load".to_string(),
            ScoreParams {
                weight: 1.0,
                kv_pressure_weight: Some(*kv_pressure_weight as f32),
                mean_prefill_tokens: Some(*mean_prefill_tokens),
                default_throughput: Some(*default_throughput as f32),
            },
        );
        profile.score = vec![ScoreSpec::Named(params)];
    }
    Ok(profile)
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

/// What a single-worker decision reads from the request.
#[derive(Clone, Copy, Default)]
pub struct SingleRequest<'a> {
    pub model: &'a str,
    pub tokens: Option<&'a [u32]>,
    pub session_key: Option<&'a str>,
    /// Base URLs of the engines earlier attempts dispatched to.
    pub tried: &'a [String],
    pub overload_lifted: bool,
    /// The phase the decision is for: `Single` on the regular paths, `Prefill`
    /// for a prefill-side decision alone (its profile carries the capacity filter).
    pub phase: Phase,
    /// The prefill capacity window as the host reads it, when the decision
    /// is a prefill side's; every candidate has capacity otherwise.
    pub has_capacity: Option<CapacityCheck<'a>>,
}

/// The engine in the gateway process with the host's feeds.
pub struct Decider {
    engine: Engine,
    loads: RwLock<Option<LoadReceiver>>,
    origin: Instant,
    seq: AtomicU64,
    hasher: RandomState,
    default_label: &'static str,
    prefill_label: &'static str,
    decode_label: &'static str,
}

impl std::fmt::Debug for Decider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decider")
            .field("generation", &self.engine.generation())
            .field("needs", &self.engine.needs())
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
    /// The engine for the gateway's legacy configuration: the default policy
    /// bound to every model, the prefill and decode policies under the
    /// `pd-pair` plan with the admission filters in front of each side.
    pub fn from_configs(
        default: &PolicyConfig,
        prefill: Option<&PolicyConfig>,
        decode: Option<&PolicyConfig>,
    ) -> Result<Self, EngineHostError> {
        let prefill = prefill.unwrap_or(default);
        let decode = decode.unwrap_or(default);
        let mut prefill_profile = profile_of(prefill)?;
        prefill_profile.filter.insert(2, "has-capacity".into());
        let mut decode_profile = profile_of(decode)?;
        decode_profile.filter.insert(2, "has-room".into());
        let mut profiles = std::collections::BTreeMap::new();
        profiles.insert("default".to_string(), profile_of(default)?);
        profiles.insert("prefill".to_string(), prefill_profile);
        profiles.insert("decode".to_string(), decode_profile);
        let mut sides = std::collections::BTreeMap::new();
        sides.insert("prefill".to_string(), "prefill".to_string());
        sides.insert("decode".to_string(), "decode".to_string());
        let spec = PolicySpec {
            api_version: API_VERSION.to_string(),
            generation: 1,
            profiles,
            bindings: vec![
                BindingSpec {
                    matcher: smg_policy::spec::Match {
                        model: "*".into(),
                        phase: vec![Phase::Prefill, Phase::Decode],
                    },
                    plan: Some("pd-pair".into()),
                    profiles: sides,
                    profile: None,
                },
                BindingSpec {
                    profile: Some("default".into()),
                    ..BindingSpec::default()
                },
            ],
        };
        let engine = spec.validate()?;
        Ok(Self {
            engine,
            loads: RwLock::new(None),
            origin: Instant::now(),
            seq: AtomicU64::new(1),
            hasher: RandomState::new(),
            default_label: preset_name(default).unwrap_or("engine"),
            prefill_label: preset_name(prefill).unwrap_or("engine"),
            decode_label: preset_name(decode).unwrap_or("engine"),
        })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// The metrics label of the profile that decides for `phase`.
    pub fn label(&self, phase: Phase) -> &'static str {
        match phase {
            Phase::Prefill => self.prefill_label,
            Phase::Decode => self.decode_label,
            Phase::Single | Phase::Encode => self.default_label,
        }
    }

    /// Hand the engine the worker monitor's load feed.
    pub fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        *self.loads.write() = rx;
    }

    fn snapshot(&self) -> Option<Arc<LoadSnapshot>> {
        self.loads.read().as_ref().map(|rx| rx.borrow().clone())
    }

    fn micros_since_origin(&self, at: Instant) -> u64 {
        at.checked_duration_since(self.origin)
            .map_or(0, |d| d.as_micros() as u64)
    }

    /// The stable id of a worker: its URL and the incarnation behind the
    /// handle, so a same-URL replacement is a new id.
    pub fn worker_id(&self, worker: &dyn Worker) -> WorkerId {
        let mut hasher = self.hasher.build_hasher();
        worker.url().hash(&mut hasher);
        (worker as *const dyn Worker).cast::<()>().hash(&mut hasher);
        WorkerId(hasher.finish())
    }

    fn candidate(
        &self,
        build: &ViewBuild<'_>,
        position: u32,
        worker: &Arc<dyn Worker>,
        has_capacity: bool,
        partner_group: u32,
    ) -> Candidate {
        let ViewBuild {
            snapshot,
            now,
            session_key,
        } = *build;
        let state = worker.routing_state();
        let worker_type = worker.worker_type();
        let role = match worker_type {
            WorkerType::Prefill => Role::Prefill,
            WorkerType::Decode => Role::Decode,
            WorkerType::Encode => Role::Encode,
            WorkerType::Regular => Role::Single,
        };
        let report = snapshot
            .and_then(|s| s.report(worker.url()))
            .map(|(load, sampled_at)| {
                let live_rate = if *worker_type == WorkerType::Prefill {
                    load.loads
                        .iter()
                        .filter_map(|rank| rank.prefill_throughput)
                        .filter(|rate| rate.is_finite() && *rate > 0.0)
                        .sum::<f64>()
                } else {
                    load.loads
                        .iter()
                        .filter(|rank| rank.num_running_reqs > 0)
                        .map(|rank| rank.gen_throughput)
                        .sum::<f64>()
                };
                LoadReport {
                    running: u32::try_from(load.total_running_reqs()).unwrap_or(0),
                    waiting: u32::try_from(load.total_waiting_reqs()).unwrap_or(0),
                    waiting_uncached_tokens: u32::try_from(load.total_waiting_uncached_tokens())
                        .unwrap_or(u32::MAX),
                    kv_pressure: load.effective_kv_pressure() as f32,
                    token_usage: load.effective_token_usage() as f32,
                    live_rate: live_rate as f32,
                    age_micros: u32::try_from(
                        now.saturating_duration_since(sampled_at).as_micros(),
                    )
                    .unwrap_or(u32::MAX),
                    sampled_at_micros: self.micros_since_origin(sampled_at),
                }
            });
        Candidate {
            position,
            id: self.worker_id(worker.as_ref()),
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
                draining: false,
                overloaded: state.overloaded,
            },
            in_flight: u32::try_from(state.load).unwrap_or(u32::MAX),
            report,
            signal: if report.is_some() {
                smg_policy::SignalSource::Poll
            } else {
                smg_policy::SignalSource::None
            },
            session_in_flight: session_key.map_or(0, |key| {
                u32::try_from(worker.routing_key_inflight(key)).unwrap_or(0)
            }),
            partner_group,
            has_capacity,
            has_room: role != Role::Decode || pd_admission::has_free_room(worker.as_ref()),
            ..Candidate::default()
        }
    }

    fn lease(&self, now: Instant, tokens: u32, session_key: Option<&str>) -> Lease {
        Lease {
            id: LeaseId(self.seq.fetch_add(1, Ordering::Relaxed)),
            kind: LeaseKind::Credit,
            generation: self.engine.generation(),
            session_key: self.hasher.hash_one(session_key),
            issued_micros: self.micros_since_origin(now),
            tokens,
        }
    }

    fn seed(&self, now_micros: u64) -> u64 {
        now_micros ^ self.seq.load(Ordering::Relaxed).rotate_left(32)
    }

    /// Decide a single worker over `workers`. `excluded` are the ids a
    /// commit of this request just refused, for the one re-decision.
    pub fn decide_single(
        &self,
        request: &SingleRequest<'_>,
        workers: &[Arc<dyn Worker>],
        excluded: &[WorkerId],
    ) -> (Decision, Lease) {
        let now = Instant::now();
        let snapshot = self.snapshot();
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
                if request
                    .tried
                    .iter()
                    .any(|engine| engine == worker.base_url())
                {
                    tried.push(candidate.id);
                }
                candidates.push(candidate);
            }
            let now_micros = self.micros_since_origin(now);
            let tokens = request
                .tokens
                .map_or(0, |t| u32::try_from(t.len()).unwrap_or(u32::MAX));
            let facts = RequestFacts {
                model: request.model,
                prompt_tokens: tokens,
                session_key: request.session_key,
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
                subset_hint: None,
            };
            let decision = self.engine.decide(&facts, &pool, candidates, &cx, scratch);
            (decision, self.lease(now, tokens, request.session_key))
        })
    }

    /// Decide a prefill and decode pair: the prefill profile over `prefills`
    /// (with `has_capacity` as the admission window reads it now), the
    /// decode profile over the partners of the chosen prefill. `partners(i)`
    /// lists the open partners of `prefills[i]`.
    pub fn decide_pair(
        &self,
        request: &SingleRequest<'_>,
        prefills: &[Arc<dyn Worker>],
        has_capacity: &dyn Fn(&Arc<dyn Worker>) -> bool,
        partners: &dyn Fn(usize) -> Vec<Arc<dyn Worker>>,
        lift_decode: bool,
        excluded: &[WorkerId],
    ) -> (Decision, Lease, Vec<Arc<dyn Worker>>) {
        let now = Instant::now();
        let snapshot = self.snapshot();
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
                let group = i as u32 + 1;
                candidates.push(self.candidate(
                    &build,
                    i as u32,
                    prefill,
                    has_capacity(prefill),
                    group,
                ));
                for decode in partners(i) {
                    let position = decode_workers.len() as u32;
                    decodes.push(self.candidate(&build, position, &decode, true, group));
                    decode_workers.push(decode);
                }
            }
            let now_micros = self.micros_since_origin(now);
            let tokens = request
                .tokens
                .map_or(0, |t| u32::try_from(t.len()).unwrap_or(u32::MAX));
            let facts = RequestFacts {
                model: request.model,
                prompt_tokens: tokens,
                session_key: request.session_key,
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
                subset_hint: None,
            };
            let decode_cx = Constraints {
                excluded: &[],
                overload_lifted: lift_decode,
                subset_hint: None,
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
        let tokens = request
            .tokens
            .map_or(0, |t| u32::try_from(t.len()).unwrap_or(u32::MAX));
        (
            decision,
            self.lease(now, tokens, request.session_key),
            decode_workers,
        )
    }

    /// The commit: walk the ordered targets, take the first one `accept`
    /// re-validates under the host's lock, and feed the engine's accounting
    /// for the target taken. `None` when every target missed; the caller
    /// re-decides once with the misses excluded, then falls back.
    pub fn commit(
        &self,
        lease: &Lease,
        decision: &Decision,
        side: Side,
        mut accept: impl FnMut(&Target) -> bool,
    ) -> Option<Target> {
        let pick = decision.pick()?;
        for target in pick.targets.iter() {
            if accept(target) {
                self.engine.on_dispatched(lease, target, side);
                if let Some(decode) = target.decode {
                    let decode_target = Target {
                        position: decode.position,
                        id: decode.id,
                        dp_rank: decode.dp_rank,
                        decode: None,
                    };
                    self.engine
                        .on_dispatched(lease, &decode_target, Side::Decode);
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

    /// The ids of a decision's targets, for the re-decision's exclusion.
    pub fn target_ids(decision: &Decision) -> Vec<WorkerId> {
        decision
            .pick()
            .map(|pick| pick.targets.iter().map(|t| t.id).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::WorkerStatus;

    use super::*;
    use crate::{
        policies::test_support,
        worker::{BasicWorkerBuilder, Worker},
    };

    fn pool(n: usize) -> Vec<Arc<dyn Worker>> {
        let urls: Vec<String> = (0..n).map(|i| format!("http://w{i}:8000")).collect();
        let refs: Vec<&str> = urls.iter().map(String::as_str).collect();
        test_support::make_workers(&refs)
    }

    #[test]
    fn the_engine_decides_over_the_live_pool_and_the_commit_takes_the_first_accepted_target() {
        let decider =
            Decider::from_configs(&PolicyConfig::RoundRobin, None, None).expect("round robin");
        let workers = pool(3);
        workers[1].set_status(WorkerStatus::NotReady);
        let request = SingleRequest {
            model: "m",
            ..SingleRequest::default()
        };
        let mut seen = Vec::new();
        for _ in 0..4 {
            let (decision, lease) = decider.decide_single(&request, &workers, &[]);
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

    #[test]
    fn a_commit_that_refuses_every_target_answers_none_and_books_nothing() {
        let decider = Decider::from_configs(&PolicyConfig::Random, None, None).expect("random");
        let workers = pool(2);
        let request = SingleRequest {
            model: "m",
            ..SingleRequest::default()
        };
        let (decision, lease) = decider.decide_single(&request, &workers, &[]);
        assert!(decider
            .commit(&lease, &decision, Side::Single, |_| false)
            .is_none());
        let excluded = Decider::target_ids(&decision);
        assert_eq!(excluded.len(), 2);
    }

    #[test]
    fn a_policy_the_engine_does_not_carry_is_refused_by_name() {
        let Err(error) = Decider::from_configs(&PolicyConfig::ConsistentHashing, None, None) else {
            panic!("a policy the engine does not carry is refused");
        };
        assert!(error.to_string().contains("consistent_hashing"), "{error}");
    }

    #[test]
    fn a_same_url_replacement_gets_a_new_id() {
        let decider = Decider::from_configs(&PolicyConfig::Random, None, None).expect("random");
        let a: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w:8000").build());
        let b: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w:8000").build());
        assert_ne!(decider.worker_id(a.as_ref()), decider.worker_id(b.as_ref()));
        assert_eq!(decider.worker_id(a.as_ref()), decider.worker_id(a.as_ref()));
    }
}
