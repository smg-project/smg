//! A classic policy object that asks the policy engine.
//!
//! Where the gateway keeps the policy engine in its own process, the policy
//! registry hands out this object in the place of the classic policy of a
//! name. Everything around the pick then runs as it runs for that policy:
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
//! [`NamedList::members`]: super::NamedList::members

use std::{any::Any, sync::Arc};

use smg_policy_controller::engine::{LiveProfile, PolicyEngine};
use smg_worker::identity::basic::Worker;

use super::{LoadBalancingPolicy, SelectWorkerInfo};

/// The policy object that asks the policy engine for its picks, under one
/// profile of the engine's document and in the name of one classic policy.
///
/// Built for a profile whose whole decision is one pick of its picker over
/// workers that pass the health tests ([`PolicyEngine::live_profile`]): the
/// object tests a worker's availability itself, as the classic policy of
/// its name does, so the placement hands it the snapshot's list whole and a
/// pick reads the workers it asks about and no other.
///
/// The object stands on cache lines of its own. The placement takes a
/// count of its policy at each request and drops it, so the count's line
/// is one every caller writes. A field beside that count would be read
/// from that line at each pick.
#[derive(Debug)]
#[repr(align(128))]
pub struct EnginePolicy {
    engine: Arc<PolicyEngine>,
    profile: LiveProfile,
    name: &'static str,
}

impl EnginePolicy {
    /// The object that stands for the classic policy `name` and decides by
    /// the profile named `profile` in `engine`'s document. `None` where the
    /// document holds no such profile, or one whose decision is more than
    /// one pick of its picker: the caller builds the classic policy.
    pub fn of(engine: &Arc<PolicyEngine>, profile: &str, name: &'static str) -> Option<Self> {
        let profile = engine.live_profile(profile)?;
        Some(Self {
            engine: Arc::clone(engine),
            profile,
            name,
        })
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
                let members =
                    cell.get_or_init(|| Box::new(self.engine.pool_members("", named.list)));
                return self
                    .engine
                    .pick(self.profile, members, info.overload_lifted);
            }
        }
        // Workers nothing is kept of (a narrowed list, a part of a pair's
        // list): one read of each, and the pick among those that pass.
        self.engine
            .pick_among(self.profile, workers, info.overload_lifted)
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
    use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
    use smg_policy_controller::document::{AppliedSpec, PolicySpec};
    use smg_worker::test_support::{no_health_check, render_with_recorder};

    use super::*;
    use crate::{
        config::{PolicyConfig, RoutingKeyOverrideConfig},
        policies::{policy_filters_unavailable_workers, PolicyRegistry, RoundRobinPolicy},
        routers::{
            common::{
                placement::{select_single, PlacementInputs},
                snapshot::{RoutingPool, WorkerRegistry, WorkerRegistryExt},
            },
            grpc::backend_client::BasicWorkerBuilder,
        },
    };

    const MODEL: &str = "asked-model";

    fn config(name: &str) -> PolicyConfig {
        serde_json::from_value(serde_json::json!({ "type": name })).expect("a policy's name")
    }

    /// A registry whose policies ask the engine of the preset `name`.
    fn asking(name: &str) -> PolicyRegistry {
        let mut document = PolicySpec::preset(name).expect("a built-in preset");
        document.generation = 1;
        let spec = AppliedSpec::validate(&document).expect("the preset validates");
        PolicyRegistry::with_policy_engine(
            config(name),
            RoutingKeyOverrideConfig::default(),
            Arc::new(PolicyEngine::new(spec)),
        )
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
}
