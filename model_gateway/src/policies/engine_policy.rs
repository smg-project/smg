//! A classic policy object that asks the policy engine.
//!
//! Where the gateway keeps the policy engine in its own process, the policy
//! registry hands out this object in the place of a classic policy: for a
//! model's requests, or for a side of a pair. The registry finds the
//! place's profile once, by the engine's own bindings, where it builds the
//! place's policy. Everything around the pick then runs as it runs for the
//! classic policy of that place:
//! the placement's pass with its candidate rules, the routing-key override,
//! the fallback for a pool every worker of which is vetoed, the selection
//! counter under the classic policy's name, and what the registry asks a
//! policy of itself. The object's own part is the pick among the workers it
//! is handed, which it asks of the engine.
//!
//! The engine reads the workers where a classic policy reads them: a
//! worker's health and its vetoes are one read of the worker object, and
//! nothing of a worker is copied. What the engine keeps of a list of
//! workers (a rotation's count) hangs on the routing snapshot beside that
//! list, in the cell the placement hands on with the list's name
//! ([`NamedList::members`]): no table is looked up for it and no lock is
//! taken.
//!
//! A place whose every request the routers ask the engine for needs no
//! pick here. Where nothing reads its classic policy either, the registry
//! holds the policy's name alone ([`DecidedByEngine`]).
//!
//! [`NamedList::members`]: super::NamedList::members

use std::{
    any::Any,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use smg_policy_controller::engine::{AloneProfile, PolicyEngine};
use smg_worker::identity::basic::Worker;
use tracing::error;

use super::{LoadBalancingPolicy, SelectWorkerInfo};

/// The policy object that asks the policy engine for its picks, under one
/// profile of the engine's document and in the name of one classic policy.
///
/// Built for a profile whose whole decision is one pick of its picker over
/// the workers the pass admits ([`PolicyEngine::place`]): the object tests
/// a worker's availability itself, by the pass's own test, so the placement
/// hands it the snapshot's list whole and a pick reads the workers it asks
/// about and no other. The classic policy's name only labels the object:
/// the selection counter counts under it, and no profile is found by it.
///
/// The object stands on cache lines of its own. The placement takes a
/// count of its policy at each request and drops it, so the count's line
/// is one every caller writes. A field beside that count would be read
/// from that line at each pick.
#[derive(Debug)]
#[repr(align(128))]
pub struct EnginePolicy {
    engine: Arc<PolicyEngine>,
    profile: AloneProfile,
    name: &'static str,
}

impl EnginePolicy {
    /// The object that decides by `profile`, which `engine` found for the
    /// place the object is built for ([`PolicyEngine::place`]), under the
    /// label `name`: the name of the classic policy of that place.
    pub fn of(engine: &Arc<PolicyEngine>, profile: AloneProfile, name: &'static str) -> Self {
        Self {
            engine: Arc::clone(engine),
            profile,
            name,
        }
    }
}

impl LoadBalancingPolicy for EnginePolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        // A snapshot's list named whole: what the engine keeps of it is in
        // the cell beside it, built by the list's first pick. The pick is
        // one turn of the list's own rotation and one read of that worker.
        if let Some(named) = info.named {
            if let (None, Some(cell)) = (named.positions, named.members) {
                debug_assert!(
                    std::ptr::eq(&named.list[..], workers),
                    "a list named whole is the slice handed"
                );
                let members = cell.get_or_init(|| Box::new(self.engine.pool_members(named.list)));
                return self
                    .engine
                    .pick(self.profile, members, |state| info.admits(state));
            }
        }
        // Workers nothing is kept of (a narrowed list, a part of a pair's
        // list): one read of each, and the pick among those that pass.
        self.engine
            .pick_among(self.profile, workers, |state| info.admits(state))
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// What the policy registry holds for a place whose every request the
/// routers ask the policy engine for, once nothing reads the place's
/// classic policy: the policy's name, and nothing else. It builds no state
/// of the policy and starts no task of it.
///
/// No placement asks it: a route that places a request of the place asks
/// the engine. One that asks all the same has asked a classic policy where
/// the engine decides. It is handed no worker, and the first such ask is
/// logged as an error under the policy's name.
///
/// This type goes with the registry's classic build for such a place, once
/// no policy's classic object has a reader left there.
#[derive(Debug)]
pub(crate) struct DecidedByEngine {
    name: &'static str,
    asked: AtomicBool,
}

impl DecidedByEngine {
    /// The object for the place of the classic policy `name`.
    pub(crate) fn of(name: &'static str) -> Self {
        Self {
            name,
            asked: AtomicBool::new(false),
        }
    }

    /// Whether a placement asked this object for a worker.
    #[cfg(test)]
    pub(crate) fn was_asked(&self) -> bool {
        self.asked.load(Ordering::Relaxed)
    }
}

impl LoadBalancingPolicy for DecidedByEngine {
    fn select_worker(&self, _: &[Arc<dyn Worker>], _: &SelectWorkerInfo) -> Option<usize> {
        if !self.asked.swap(true, Ordering::Relaxed) {
            error!(
                "Policy `{}`: a route asked the classic policy for a worker where the policy \
                 engine decides every request, and is handed none",
                self.name
            );
        }
        None
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Barrier, thread, time::Duration};

    use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
    use smg_policy_controller::document::{AppliedSpec, PolicySpec};
    use smg_worker::{
        identity::basic::{ConnectionMode, RuntimeType, WorkerType},
        test_support::{no_health_check, render_with_recorder},
    };

    use super::*;
    use crate::{
        config::{PolicyConfig, RoutingKeyOverrideConfig},
        policies::{
            policy_filters_unavailable_workers, DocumentOrigin, PolicyRegistry, RoundRobinPolicy,
            Side,
        },
        routers::{
            common::{
                pair_index::PdWire,
                placement::{select_pair, select_single, PlacementInputs},
                prefill_admission::{
                    PrefillAdmission, PrefillAdmissionAttempt, PrefillSelectionContext,
                },
                snapshot::{RoutingPool, WorkerRegistry, WorkerRegistryExt},
            },
            grpc::backend_client::BasicWorkerBuilder,
        },
    };

    const MODEL: &str = "asked-model";

    fn config(name: &str) -> PolicyConfig {
        serde_json::from_value(serde_json::json!({ "type": name })).expect("a policy's name")
    }

    /// A registry whose policies ask the engine of the preset `name`, the
    /// document a gateway's settings make for that policy.
    fn asking(name: &str) -> PolicyRegistry {
        let document = PolicySpec::preset(name).expect("a built-in preset");
        asking_under(&document, name, DocumentOrigin::Settings)
    }

    /// A registry whose default policy is `name` and whose policies ask the
    /// engine of `document`. No router stands over it that asks the engine
    /// for a whole decision.
    fn asking_under(document: &PolicySpec, name: &str, origin: DocumentOrigin) -> PolicyRegistry {
        asking_as(document, name, origin, false)
    }

    /// [`asking_under`], with whether the routers ask the engine for each
    /// request's whole decision under `document`.
    fn asking_as(
        document: &PolicySpec,
        name: &str,
        origin: DocumentOrigin,
        whole: bool,
    ) -> PolicyRegistry {
        let spec = AppliedSpec::validate(document).expect("the document validates");
        PolicyRegistry::with_policy_engine(
            config(name),
            RoutingKeyOverrideConfig::default(),
            Arc::new(PolicyEngine::new(spec)),
            origin,
            whole,
        )
    }

    /// Least load's preset with one more binding ahead of its own: the
    /// models `alpha-*` are placed by a profile of round robin's stages,
    /// named for nothing a policy is named.
    fn two_bindings() -> PolicySpec {
        let mut document = PolicySpec::preset("least_load").expect("a built-in preset");
        let turns = PolicySpec::preset("round_robin").expect("a built-in preset");
        let profile = turns.profiles["round_robin"].clone();
        document.profiles.insert("by-turns".to_string(), profile);
        let mut named = document.bindings[0].clone();
        named.matcher.model = "alpha-*".to_string();
        named.profile = Some("by-turns".to_string());
        document.bindings.insert(0, named);
        document
    }

    /// The workers of one side of a pair fleet, in their order.
    type Workers = Vec<Arc<dyn Worker>>;

    /// A pair fleet on the gRPC wire: `prefills` prefill workers and
    /// `decodes` decode workers of one model, each list in its order. A
    /// decode reports a running window of two rooms, so the rooms count in
    /// a decode's pick.
    fn pair_fleet(prefills: usize, decodes: usize) -> (WorkerRegistry, Workers, Workers) {
        let registry = WorkerRegistry::new();
        let side = |name: &str, worker_type: WorkerType, count: usize| -> Workers {
            (0..count)
                .map(|at| {
                    let mut labels = HashMap::new();
                    if worker_type == WorkerType::Decode {
                        labels.insert("max_running_requests".to_string(), "2".to_string());
                    }
                    let worker: Arc<dyn Worker> = Arc::new(
                        BasicWorkerBuilder::new(format!("grpc://{name}-{at}:9000"))
                            .model(ModelCard::new(MODEL))
                            .worker_type(worker_type)
                            .connection_mode(ConnectionMode::Grpc)
                            .runtime_type(RuntimeType::Vllm)
                            .labels(labels)
                            .health_config(no_health_check())
                            .build(),
                    );
                    worker.set_status(WorkerStatus::Ready);
                    assert!(registry.register(Arc::clone(&worker)).is_some());
                    worker
                })
                .collect()
        };
        let prefill = side("prefill", WorkerType::Prefill, prefills);
        let decode = side("decode", WorkerType::Decode, decodes);
        (registry, prefill, decode)
    }

    /// A pair gateway's registry under round robin on both sides, its side
    /// policies built by the registry and asking the engine of the pair
    /// document those settings make.
    fn asking_for_a_pair() -> PolicyRegistry {
        let document =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("built-in presets");
        let policies = asking_under(&document, "round_robin", DocumentOrigin::Settings);
        policies.set_side_policy(Side::Prefill, &PolicyConfig::RoundRobin);
        policies.set_side_policy(Side::Decode, &PolicyConfig::RoundRobin);
        // Every case over this registry is a case of the engine's object.
        for side in [policies.get_prefill_policy(), policies.get_decode_policy()] {
            assert!(side.as_any().is::<EnginePolicy>(), "{}", side.name());
        }
        policies
    }

    /// The place of `worker` in `side`.
    fn place_in(side: &[Arc<dyn Worker>], worker: &Arc<dyn Worker>) -> usize {
        side.iter()
            .position(|member| Arc::ptr_eq(member, worker))
            .expect("a worker of the side")
    }

    fn fleet(workers: usize) -> (Arc<WorkerRegistry>, Vec<Arc<dyn Worker>>) {
        let registry = Arc::new(WorkerRegistry::new());
        let fleet: Vec<Arc<dyn Worker>> = (0..workers)
            .map(|at| {
                let worker: Arc<dyn Worker> = Arc::new(
                    BasicWorkerBuilder::new(format!("http://asked-{at:03}:8000"))
                        .model(ModelCard::new(MODEL))
                        .health_config(no_health_check())
                        .build(),
                );
                worker.set_status(WorkerStatus::Ready);
                assert!(registry.register(Arc::clone(&worker)).is_some());
                worker
            })
            .collect();
        (registry, fleet)
    }

    /// The worker one request is placed on, by its place in `among`.
    fn placed(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        among: &[Arc<dyn Worker>],
        inputs: PlacementInputs<'_>,
    ) -> Option<usize> {
        let pool = RoutingPool::HttpRegular;
        let worker = select_single(registry, policies, MODEL, pool, None, inputs)?;
        among.iter().position(|member| Arc::ptr_eq(member, &worker))
    }

    /// The registry builds the object where the engine's document holds a
    /// profile the engine picks by its picker alone, under the classic
    /// policy's name, and the classic policy of the name where it does not.
    #[test]
    fn the_registry_builds_the_object_for_a_profile_the_engine_picks_alone() {
        let policies = asking("round_robin");
        let policy = policies.get_default_policy();
        assert!(policy.as_any().is::<EnginePolicy>());
        assert_eq!(policy.name(), "round_robin");
        // The object tests availability itself, as round robin does: the
        // placement hands it the snapshot's list whole.
        assert!(policy_filters_unavailable_workers(policy.as_ref()));
        // A model whose label names the default's type is placed by the
        // object too, and one whose label names another type by the
        // classic policy of that name.
        let labelled = policies.on_worker_added("labelled", Some("round_robin"));
        assert!(labelled.as_any().is::<EnginePolicy>());
        let other = policies.on_worker_added("other", Some("random"));
        assert_eq!(other.name(), "random");
        assert!(!other.as_any().is::<EnginePolicy>());

        // A profile that scores is not picked so yet: the classic policy.
        let by_load = asking("least_load");
        let policy = by_load.get_default_policy();
        assert!(!policy.as_any().is::<EnginePolicy>());
        assert_eq!(policy.name(), "least_load");

        // No engine: the classic policy, as ever.
        let classic = PolicyRegistry::new(PolicyConfig::RoundRobin);
        assert!(classic
            .get_default_policy()
            .as_any()
            .is::<RoundRobinPolicy>());
    }

    /// With every worker ready the object turns one strict rotation over
    /// the registry's pool through the classic placement, and the cell of
    /// the pool on the snapshot holds what the engine keeps of it.
    #[test]
    fn the_object_turns_one_rotation_over_the_pool_the_pass_hands_it() {
        for workers in [1, 2, 5, 64] {
            let (registry, _fleet) = fleet(workers);
            let policies = asking("round_robin");
            let pool = registry.get_routing_pool(MODEL, RoutingPool::HttpRegular);
            let next = || placed(&registry, &policies, &pool, PlacementInputs::default());
            let first = next().expect("a worker");
            for step in 1..=2 * workers {
                assert_eq!(next(), Some((first + step) % workers), "{workers} workers");
            }
            let snapshot = registry.get_routing_snapshot(MODEL);
            let members = snapshot
                .pool_members_cell(RoutingPool::HttpRegular)
                .get()
                .expect("the pool's first pick filled its cell");
            assert!(Arc::ptr_eq(
                members.workers(),
                snapshot.pool(RoutingPool::HttpRegular)
            ));
        }
    }

    /// What the engine keeps of a pool stands on the registry's snapshot,
    /// over the pool's own list: one value a membership, borrowed by every
    /// pick, and a new one after a change. Nothing is made for a pool before
    /// its first pick.
    #[test]
    fn the_engines_value_of_a_pool_stands_until_the_membership_changes() {
        let (registry, fleet) = fleet(3);
        let policies = asking("round_robin");
        let pool = RoutingPool::HttpRegular;
        let snapshot = registry.get_routing_snapshot(MODEL);
        assert!(snapshot.pool_members_cell(pool).get().is_none());
        let pick = || placed(&registry, &policies, &fleet, PlacementInputs::default());
        assert!(pick().is_some());
        let members = snapshot
            .pool_members_cell(pool)
            .get()
            .expect("the first pick filled the cell");
        assert!(pick().is_some());
        let again = registry.get_routing_snapshot(MODEL);
        let kept = again.pool_members_cell(pool).get().expect("the value");
        assert!(std::ptr::eq(&**members, &**kept), "one value a membership");

        let joined: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://asked-new:8000")
                .model(ModelCard::new(MODEL))
                .health_config(no_health_check())
                .build(),
        );
        joined.set_status(WorkerStatus::Ready);
        assert!(registry.register(joined).is_some());
        let changed = registry.get_routing_snapshot(MODEL);
        assert!(changed.pool_members_cell(pool).get().is_none());
        assert!(pick().is_some());
        let fresh = changed.pool_members_cell(pool).get().expect("a new value");
        assert_eq!((members.workers().len(), fresh.workers().len()), (3, 4));
    }

    /// A model nobody serves is placed on nobody, and nothing is kept for
    /// it on the snapshot every unknown model shares.
    #[test]
    fn a_model_nobody_serves_is_placed_on_nobody() {
        let (registry, _fleet) = fleet(2);
        let policies = asking("round_robin");
        let pool = RoutingPool::HttpRegular;
        let nobody = select_single(
            &registry,
            &policies,
            "no-such-model",
            pool,
            None,
            PlacementInputs::default(),
        );
        assert!(nobody.is_none());
        let shared = registry.get_routing_snapshot("no-such-model");
        assert!(shared.pool_members_cell(pool).get().is_none());
    }

    /// Callers that pick at the same moment do not pile on one worker.
    /// Four callers pick together, a barrier a round, over 256 idle
    /// workers, 64 rounds: every worker takes exactly one request, by the
    /// classic policy and by the object alike. No lock orders the callers
    /// on either path: a rotation is one atomic step.
    #[test]
    fn callers_that_pick_at_once_take_each_idle_worker_once() {
        const CALLERS: usize = 4;
        const ROUNDS: usize = 64;
        for by_engine in [false, true] {
            let (registry, fleet) = fleet(CALLERS * ROUNDS);
            let policies = if by_engine {
                asking("round_robin")
            } else {
                PolicyRegistry::new(PolicyConfig::RoundRobin)
            };
            let round = Barrier::new(CALLERS);
            let taken: Vec<usize> = thread::scope(|scope| {
                let callers: Vec<_> = (0..CALLERS)
                    .map(|_| {
                        scope.spawn(|| {
                            let pick = |_| {
                                round.wait();
                                placed(&registry, &policies, &fleet, PlacementInputs::default())
                                    .expect("an idle fleet takes every request")
                            };
                            (0..ROUNDS).map(pick).collect::<Vec<_>>()
                        })
                    })
                    .collect();
                callers
                    .into_iter()
                    .flat_map(|caller| caller.join().expect("a caller ends"))
                    .collect()
            });
            let mut times = vec![0usize; fleet.len()];
            for at in taken {
                times[at] += 1;
            }
            let piled = times.iter().filter(|&&times| times != 1).count();
            assert_eq!(
                piled, 0,
                "workers that took no request or more than one, by the engine: {by_engine}"
            );
        }
    }

    /// The object and the classic round robin share alike where workers are
    /// not ready: no request goes to such a worker, and the ready workers
    /// take equal shares. A small pool, a pool that earns turns past a
    /// refused worker, and a run of workers not ready longer than those
    /// turns.
    #[test]
    fn the_object_and_the_classic_policy_share_alike_past_workers_not_ready() {
        // Over a run of sixteen among sixty-four the object takes three
        // turns a pick: 53 picks a turn of the pool, and 48 turns of the
        // pool a round of both its rotations. The classic policy takes one
        // turn a pick: 64 picks a turn of the pool.
        const RUN: std::ops::Range<usize> = 20..36;
        // A fleet's size, which of its workers are ready, and the picks of
        // one round of the object's rotations over it.
        type Case = (usize, fn(usize) -> bool, usize);
        let cases: [Case; 3] = [
            (8, |at| at != 5, 8 * 7),
            (64, |at| at % 4 != 1, 64 * 48),
            (64, |at| !RUN.contains(&at), 53 * 48),
        ];
        for (case, (workers, ready, round)) in cases.into_iter().enumerate() {
            let serving = (0..workers).filter(|&at| ready(at)).count();
            for by_engine in [false, true] {
                let (registry, fleet) = fleet(workers);
                for (at, worker) in fleet.iter().enumerate() {
                    if !ready(at) {
                        worker.set_status(WorkerStatus::NotReady);
                    }
                }
                let policies = if by_engine {
                    asking("round_robin")
                } else {
                    PolicyRegistry::new(PolicyConfig::RoundRobin)
                };
                // Whole rounds of every rotation the policy turns.
                let requests = 4 * if by_engine { round } else { workers * serving };
                let mut times = vec![0usize; workers];
                for _ in 0..requests {
                    let at = placed(&registry, &policies, &fleet, PlacementInputs::default());
                    times[at.expect("a worker")] += 1;
                }
                // The object's first turn starts at a drawn worker. Where
                // that is inside the run, the first pass over the run is a
                // short one, and the count ends that much off a whole
                // round: at most two picks a worker.
                let slack = if by_engine && case == 2 { 2 } else { 0 };
                for (at, &times) in times.iter().enumerate() {
                    let share = if ready(at) { requests / serving } else { 0 };
                    assert!(
                        times.abs_diff(share) <= slack,
                        "worker {at} of {workers} took {times} of {requests}, by the engine: {by_engine}"
                    );
                }
            }
        }
    }

    /// The placement's own rules run around the object as around a classic
    /// policy. A candidate predicate holds the request to the workers it
    /// accepts, the engines a failed attempt tried step aside, and a pool
    /// every worker of which is over its thresholds goes to its least
    /// loaded worker by the placement's fallback: the object answers
    /// nobody there.
    #[test]
    fn the_pass_keeps_its_rules_around_the_object() {
        let (registry, fleet) = fleet(6);
        let policies = asking("round_robin");
        let odd = |worker: &dyn Worker| {
            fleet
                .iter()
                .position(|member| member.url() == worker.url())
                .is_some_and(|at| at % 2 == 1)
        };
        let tried = [fleet[1].base_url().to_owned()];
        let held = PlacementInputs {
            candidate_filter: Some(&odd),
            tried: &tried,
            ..PlacementInputs::default()
        };
        let mut taken = [0; 6];
        for _ in 0..8 {
            taken[placed(&registry, &policies, &fleet, held).expect("a worker")] += 1;
        }
        assert_eq!(taken, [0, 0, 0, 4, 0, 4]);

        // Every worker over its thresholds, the fewest requests on one.
        for (at, worker) in fleet.iter().enumerate() {
            assert!(registry.set_worker_overloaded(worker, true));
            for _ in 0..if at == 2 { 0 } else { at + 1 } {
                worker.increment_load();
            }
        }
        for _ in 0..4 {
            let at = placed(&registry, &policies, &fleet, PlacementInputs::default());
            assert_eq!(at, Some(2), "the least loaded worker takes the request");
        }
    }

    /// A selection by the object is counted as the classic policy's: the
    /// selection counter under the classic name, and no counter of the
    /// engine's own.
    #[test]
    fn a_selection_by_the_object_counts_under_the_classic_name() {
        let rendered = render_with_recorder(|| {
            let (registry, fleet) = fleet(4);
            let policies = asking("round_robin");
            for _ in 0..9 {
                let at = placed(&registry, &policies, &fleet, PlacementInputs::default());
                assert!(at.is_some());
            }
        });
        let counted = rendered
            .lines()
            .find(|line| line.starts_with("smg_worker_selection_total{") && line.contains(MODEL))
            .expect("the selections are counted");
        assert!(counted.contains("policy=\"round_robin\""), "{counted}");
        assert!(counted.ends_with(" 9"), "{counted}");
        assert!(
            !rendered.contains("smg_policy_decisions_total{"),
            "the object counts no decision of its own"
        );
    }

    /// A place of the registry is a model and a phase, and the engine finds
    /// its profile by its document's bindings: a model is placed by the
    /// binding that names it, whatever the profile is called. The default
    /// policy serves every model, so a binding of some models does not name
    /// its place.
    #[test]
    fn a_model_is_placed_by_the_binding_that_names_it() {
        let policies = asking_under(&two_bindings(), "least_load", DocumentOrigin::Supplied);
        // Every model's binding holds a profile that scores: the classic
        // policy of the default's name, until the engine picks it.
        let default = policies.get_default_policy();
        assert_eq!(default.name(), "least_load");
        assert!(!default.as_any().is::<EnginePolicy>());
        // A model the first binding names: its own object, labelled by its
        // profile's name as the supplied document writes it.
        let alpha = policies.on_worker_added("alpha-1", None);
        assert!(alpha.as_any().is::<EnginePolicy>());
        assert_eq!(alpha.name(), "by-turns");
        // Any other model: the default policy itself.
        let beta = policies.on_worker_added("beta-1", None);
        assert!(Arc::ptr_eq(&beta, &default));
    }

    /// Where a worker's `policy` label and a binding disagree: under a
    /// supplied document the binding places the model, and under the
    /// document the settings make the label chooses, as ever.
    #[test]
    fn a_supplied_documents_binding_places_a_labelled_model() {
        let supplied = asking_under(&two_bindings(), "least_load", DocumentOrigin::Supplied);
        let bound = supplied.on_worker_added("alpha-1", Some("random"));
        assert!(bound.as_any().is::<EnginePolicy>());
        // A model the bindings leave to a profile the engine does not pick
        // yet keeps its label's classic policy.
        let other = supplied.on_worker_added("beta-1", Some("random"));
        assert_eq!(other.name(), "random");

        let settings = asking_under(&two_bindings(), "least_load", DocumentOrigin::Settings);
        let labelled = settings.on_worker_added("alpha-1", Some("random"));
        assert_eq!(labelled.name(), "random");
        assert!(!labelled.as_any().is::<EnginePolicy>());
        // No label: the model's own place, under either origin. Under the
        // document the settings make the object keeps the classic policy's
        // name.
        let plain = settings.on_worker_added("alpha-2", None);
        assert!(plain.as_any().is::<EnginePolicy>());
        assert_eq!(plain.name(), "least_load");
    }

    /// A profile's name is a label and nothing reads it as a classic
    /// policy's. The sticky routing-key override leaves the classic
    /// `manual` policy its own handling of a key, and asks the policy for
    /// that. An object whose profile a supplied document calls `manual`
    /// turns a rotation, so the override places its keyed requests: one
    /// key stays on one worker.
    #[test]
    fn a_profile_named_for_a_classic_policy_is_not_read_as_that_policy() {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let profile = document
            .profiles
            .remove("round_robin")
            .expect("the preset's profile");
        document.profiles.insert("manual".to_string(), profile);
        document.bindings[0].profile = Some("manual".to_string());
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let policies = PolicyRegistry::with_policy_engine(
            config("round_robin"),
            RoutingKeyOverrideConfig {
                enabled: true,
                ..RoutingKeyOverrideConfig::default()
            },
            Arc::new(PolicyEngine::new(spec)),
            DocumentOrigin::Supplied,
            false,
        );
        let policy = policies.get_default_policy();
        assert!(policy.as_any().is::<EnginePolicy>());
        assert_eq!(policy.name(), "manual");
        assert!(!policy.honors_routing_key());

        let (_registry, fleet) = fleet(4);
        let keyed = SelectWorkerInfo {
            routing_key: Some("one-conversation"),
            ..SelectWorkerInfo::default()
        };
        let first = policies
            .select_worker_for_model(&policy, MODEL, &fleet, &keyed)
            .expect("a worker");
        for _ in 0..4 {
            let again = policies.select_worker_for_model(&policy, MODEL, &fleet, &keyed);
            assert_eq!(again, Some(first), "the key's worker");
        }
    }

    /// A model no binding of a supplied document names is placed by the
    /// classic policies, as with the setting off: its workers' label first,
    /// the default policy otherwise. Each such model is counted once where
    /// it gets its first worker. A model a pair binding names is not one
    /// of them: its requests are a pair's, which the side policies place.
    #[test]
    fn a_model_no_binding_names_is_placed_by_the_classic_policies_and_counted() {
        const UNBOUND: &str = "smg_policy_unbound_models_total";
        let rendered = render_with_recorder(|| {
            let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
            document.bindings[0].matcher.model = "alpha-*".to_string();
            let policies = asking_under(&document, "round_robin", DocumentOrigin::Supplied);
            let default = policies.get_default_policy();
            assert!(default.as_any().is::<RoundRobinPolicy>());
            let unbound = policies.on_worker_added("beta-1", None);
            assert!(Arc::ptr_eq(&unbound, &default));
            // A later worker of the model finds its policy in place.
            policies.on_worker_added("beta-1", None);
            let labelled = policies.on_worker_added("beta-2", Some("random"));
            assert_eq!(labelled.name(), "random");
            let bound = policies.on_worker_added("alpha-1", None);
            assert!(bound.as_any().is::<EnginePolicy>());
        });
        let counted = rendered
            .lines()
            .find(|line| line.starts_with(UNBOUND))
            .expect("the unbound models are counted");
        assert!(counted.ends_with(" 2"), "{counted}");

        let rendered = render_with_recorder(|| {
            let pair =
                PolicySpec::pair_presets("round_robin", "round_robin").expect("built-in presets");
            let policies = asking_under(&pair, "round_robin", DocumentOrigin::Supplied);
            let default = policies.get_default_policy();
            let paired = policies.on_worker_added(MODEL, None);
            assert!(Arc::ptr_eq(&paired, &default));
        });
        assert!(
            !rendered.lines().any(|line| line.starts_with(UNBOUND)),
            "a pair binding names the model"
        );
    }

    /// The registry builds a pair's side policies, from the side and the
    /// side's settings: each side's place is the side's phase, and its
    /// profile is the pair document's own, whatever it is called. With no
    /// engine the same call builds the classic policy.
    #[test]
    fn the_registry_builds_a_pairs_sides_by_their_places() {
        let policies = asking_for_a_pair();
        for side in [policies.get_prefill_policy(), policies.get_decode_policy()] {
            assert!(side.as_any().is::<EnginePolicy>());
            assert_eq!(side.name(), "round_robin");
        }
        // The pair document plans no single request: the classic policy.
        assert!(policies
            .get_default_policy()
            .as_any()
            .is::<RoundRobinPolicy>());
        // No side was set for the encode phase: its fallback, the classic
        // policy of its own name.
        assert_eq!(policies.get_encode_policy().name(), "consistent_hashing");

        let classic = PolicyRegistry::new(PolicyConfig::RoundRobin);
        classic.set_side_policy(Side::Prefill, &PolicyConfig::RoundRobin);
        classic.set_side_policy(Side::Decode, &config("random"));
        assert!(classic
            .get_prefill_policy()
            .as_any()
            .is::<RoundRobinPolicy>());
        assert_eq!(classic.get_decode_policy().name(), "random");
    }

    /// The log says who decides each policy the registry builds and keeps,
    /// once: the default at the start, a side where it is first set (a
    /// gateway of many routers asks for it from each of its pair routers),
    /// the encode fallback at its first read, and a model's policy that
    /// its workers' label names.
    #[test]
    #[tracing_test::traced_test]
    fn the_log_says_once_who_decides_each_policy_the_registry_keeps() {
        const LINES: [&str; 5] = [
            "Policy `round_robin` of every model (single requests): the classic policy decides it",
            "Policy `round_robin` of every model (prefill requests): the policy engine decides it",
            "Policy `round_robin` of every model (decode requests): the policy engine decides it",
            "Policy `consistent_hashing` of every model (encode requests): the classic policy decides it",
            "Policy `random` of model asked-model (single requests): the classic policy decides it, \
             its workers' policy label names it",
        ];
        let policies = asking_for_a_pair();
        let prefill = policies.get_prefill_policy();
        for _ in 0..2 {
            policies.set_side_policy(Side::Prefill, &PolicyConfig::RoundRobin);
        }
        assert!(Arc::ptr_eq(&prefill, &policies.get_prefill_policy()));
        let encode = policies.get_encode_policy();
        assert!(Arc::ptr_eq(&encode, &policies.get_encode_policy()));
        policies.on_worker_added(MODEL, Some("random"));

        logs_assert(|lines: &[&str]| {
            for line in LINES {
                let written = lines
                    .iter()
                    .filter(|written| written.contains(line))
                    .count();
                if written != 1 {
                    return Err(format!("{written} lines say: {line}"));
                }
            }
            Ok(())
        });
    }

    /// A policy the engine does not pick by itself is decided by the
    /// engine's whole pass where the routers ask the engine for each
    /// request, and the log says so in the classic line's place. Where they
    /// do not, the classic policy decides it and the line says that.
    #[test]
    #[tracing_test::traced_test]
    fn the_log_says_the_engine_decides_by_its_whole_pass_where_the_routers_ask_it() {
        let document = PolicySpec::preset("least_load").expect("a built-in preset");
        asking_as(&document, "least_load", DocumentOrigin::Settings, true);
        let document = PolicySpec::preset("power_of_two").expect("a built-in preset");
        asking_as(&document, "power_of_two", DocumentOrigin::Settings, false);
        logs_assert(|lines: &[&str]| {
            let said = |words: &str| lines.iter().filter(|line| line.contains(words)).count();
            let whole = said(
                "Policy `least_load` of every model (single requests): the policy engine decides \
                 it by its whole pass (the profile `least_load`)",
            );
            let classic = said(
                "Policy `power_of_two` of every model (single requests): the classic policy \
                 decides it, the policy engine does not pick the profile `power_of_two` by itself",
            );
            let wrong = said("Policy `least_load` of every model (single requests): the classic");
            match (whole, classic, wrong) {
                (1, 1, 0) => Ok(()),
                counts => Err(format!("{counts:?} lines")),
            }
        });
    }

    /// Where the routers ask the engine for every request, nothing reads
    /// the classic manual policy: the registry builds none and holds the
    /// policy's name alone, and the log says so. A placement that asks it
    /// all the same is handed no worker, and the first such ask is logged
    /// as an error. Where the routers keep their classic pass the classic
    /// manual policy is built as ever, and a policy that still has a reader
    /// is built under both.
    #[tokio::test]
    #[tracing_test::traced_test]
    async fn the_registry_builds_no_classic_manual_policy_where_the_engine_decides_whole() {
        use crate::policies::{LeastLoadPolicy, ManualPolicy};

        let document = PolicySpec::preset("manual").expect("a built-in preset");
        let whole = asking_as(&document, "manual", DocumentOrigin::Settings, true);
        let default = whole.get_default_policy();
        assert_eq!(default.name(), "manual");
        let held = default
            .as_any()
            .downcast_ref::<DecidedByEngine>()
            .expect("the policy's name alone");
        assert!(!held.was_asked());
        let (_registry, workers) = fleet(2);
        for _ in 0..2 {
            let picked = default.select_worker(&workers, &SelectWorkerInfo::default());
            assert_eq!(picked, None);
        }
        assert!(held.was_asked());

        let classic_pass = asking_as(&document, "manual", DocumentOrigin::Settings, false);
        let default = classic_pass.get_default_policy();
        assert!(default.as_any().is::<ManualPolicy>());

        let read = PolicySpec::preset("least_load").expect("a built-in preset");
        let whole = asking_as(&read, "least_load", DocumentOrigin::Settings, true);
        assert!(whole.get_default_policy().as_any().is::<LeastLoadPolicy>());

        logs_assert(|lines: &[&str]| {
            let said = |words: &str| lines.iter().filter(|line| line.contains(words)).count();
            let not_built = said(
                "Policy `manual` of every model (single requests): the policy engine decides it \
                 by its whole pass (the profile `manual`); no classic policy is built",
            );
            // The ask is an error's line, once.
            let asked = lines
                .iter()
                .filter(|line| line.contains("ERROR"))
                .filter(|line| line.contains("Policy `manual`: a route asked the classic policy"))
                .count();
            match (not_built, asked) {
                (1, 1) => Ok(()),
                counts => Err(format!("{counts:?} lines")),
            }
        });
    }

    /// The prefill side's profile holds the test of a free slot of the
    /// prefill window, and the object drops it: the pass holds it. Under
    /// prefill admission a prefill at its bound leaves the candidates before
    /// the side's object is asked, so the object never picks it. With no
    /// admission the pass asks nothing of the window, and neither does the
    /// object.
    #[test]
    fn the_pass_holds_the_prefill_window_ahead_of_the_prefill_sides_object() {
        let (registry, prefills, _decodes) = pair_fleet(3, 2);
        let policies = asking_for_a_pair();
        let pairs = registry
            .get_routing_snapshot(MODEL)
            .pd_pairs(PdWire::Grpc, policies.pd_pairing_mode());
        // One request in flight on the second prefill, and a bound of one.
        prefills[1].increment_load();
        let admission = PrefillAdmission::new(1, 0, Duration::ZERO);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("a runtime");
        let mut taken = [0usize; 3];
        for _ in 0..8 {
            let admitted = runtime.block_on(admission.admit(
                None,
                |capacity: &PrefillSelectionContext<'_>| {
                    let inputs = PlacementInputs::default();
                    let wire = None;
                    match select_pair(
                        &registry,
                        &policies,
                        MODEL,
                        &pairs,
                        wire,
                        Some(capacity),
                        inputs,
                    ) {
                        Ok(pair) => {
                            taken[place_in(&prefills, &pair.prefill)] += 1;
                            capacity.select(pair.prefill, ())
                        }
                        Err(_) => PrefillAdmissionAttempt::Unavailable,
                    }
                },
            ));
            // The admitted request is let go at once.
            drop(admitted);
        }
        assert_eq!(taken, [4, 0, 4], "the prefill at its bound takes nothing");

        // With no admission the window is nobody's test: every prefill
        // takes its turn.
        let mut taken = [0usize; 3];
        for _ in 0..9 {
            let inputs = PlacementInputs::default();
            let pair = select_pair(&registry, &policies, MODEL, &pairs, None, None, inputs)
                .expect("a pair");
            taken[place_in(&prefills, &pair.prefill)] += 1;
        }
        assert_eq!(taken, [3, 3, 3]);
    }

    /// The decode side's profile holds the test of a free decode room, and
    /// the object drops it: the pass holds it. A decode whose rooms are all
    /// claimed leaves the candidates before the side's object is asked,
    /// while a sibling has a room. With every decode full the pass leaves
    /// the list as it is, and the object picks among them all.
    #[test]
    fn the_pass_holds_the_decode_rooms_ahead_of_the_decode_sides_object() {
        let (registry, _prefills, decodes) = pair_fleet(2, 3);
        let policies = asking_for_a_pair();
        let pairs = registry
            .get_routing_snapshot(MODEL)
            .pd_pairs(PdWire::Grpc, policies.pd_pairing_mode());
        let decode_of = |policies: &PolicyRegistry| {
            let inputs = PlacementInputs::default();
            let pair = select_pair(&registry, policies, MODEL, &pairs, None, None, inputs)
                .expect("a pair");
            place_in(&decodes, &pair.decode)
        };
        // Both rooms of the second decode are claimed, and the others each
        // report a window with a free room.
        assert!(decodes[1].try_admit_pd(2, 2), "both rooms claimed");
        assert!(decodes[0].try_admit_pd(1, 2) && decodes[2].try_admit_pd(1, 2));
        let mut taken = [0usize; 3];
        for _ in 0..8 {
            taken[decode_of(&policies)] += 1;
        }
        assert_eq!(taken, [4, 0, 4], "the full decode takes nothing");

        // Every decode full: the list stands, and each takes its turn.
        assert!(decodes[0].try_admit_pd(1, 2) && decodes[2].try_admit_pd(1, 2));
        let mut taken = [0usize; 3];
        for _ in 0..9 {
            taken[decode_of(&policies)] += 1;
        }
        assert_eq!(taken, [3, 3, 3]);
    }
}
