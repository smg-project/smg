//! Placement by the policy engine, over the registry's own data.
//!
//! Where the `policy_controller` setting is on and the gateway keeps the
//! engine itself, the placement asks the policy engine where it would ask a
//! classic policy ([`super::placement`]). Nothing around the decision
//! changes: the pool is the registry's routing pool for the model, the
//! workers are the registry's own objects, a worker's health and its
//! requests in flight are read on the worker object as the classic policies
//! read them, and the worker the engine names goes back to the router as a
//! classic selection does.
//!
//! What the engine derives from a pool's members is kept on the registry's
//! snapshot ([`ModelWorkerSnapshot::pool_members`]): built once a membership
//! change, borrowed by every request after it. A request every worker of
//! its pool may take builds nothing and copies nothing. A request the
//! router holds to some of the pool's workers (a retained plan's wire, a
//! candidate predicate, the engines a failed attempt tried) asks its rule of
//! each worker once, as the classic placement does, into a list the thread
//! keeps.
//!
//! [`ModelWorkerSnapshot::pool_members`]: super::snapshot::ModelWorkerSnapshot::pool_members

use std::{
    cell::{Cell, RefCell},
    sync::Arc,
    time::Instant,
};

use smg_policy_controller::engine::{
    Placement, PlacementRequest, PolicyEngine, PoolMembers, WorkerId,
};
use smg_worker::identity::basic::Worker;

use super::{
    placement::{PlacementInputs, WireConstraint},
    snapshot::{RoutingPool, WorkerRegistry, WorkerRegistryExt},
};
use crate::observability::metrics::{
    KeptPicks, Metrics, PolicyDecision, POLICY_DECISIONS_PER_TIMED,
};

/// The `mode` label of a decision made in this process.
const EMBEDDED: &str = "embedded";

/// What limits one request to some of a pool's workers, in the lists a
/// thread keeps for such requests: the rows the request may not take, and
/// the ids of the workers it tried.
#[derive(Default)]
struct Limits {
    refused: Vec<u32>,
    tried: Vec<WorkerId>,
}

thread_local! {
    static LIMITS: RefCell<Limits> = RefCell::new(Limits::default());
    /// The decisions this thread has yet to make before it times one.
    static UNTIMED: Cell<u32> = const { Cell::new(0) };
}

/// One worker for `model_id` from `pool`, named by the policy engine, or
/// `None` when it places the request on none. The decision is counted, and
/// one decision in [`POLICY_DECISIONS_PER_TIMED`] is timed.
///
/// The engine is told the model, the workers the request may not take
/// (`wire`, the candidate predicate) and the engines it tried. What else
/// `inputs` holds of the request (its text, tokens, headers, session key
/// and cache partition) is read by policies the engine does not yet decide
/// here: nothing sets the engine to a served gateway's placement until it
/// is told each.
pub(crate) fn single(
    engine: &PolicyEngine,
    registry: &WorkerRegistry,
    model_id: &str,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
    inputs: PlacementInputs<'_>,
) -> Option<Arc<dyn Worker>> {
    let snapshot = registry.get_routing_snapshot(model_id);
    let workers = snapshot.pool(pool);
    if workers.is_empty() {
        // Nobody serves the model on this route: there is nothing to
        // decide, and the snapshot shared by every unknown model keeps no
        // client's model id.
        return None;
    }
    let members = snapshot.pool_members(pool, model_id, engine);
    let picks = snapshot.pool_picks(pool);
    if wire.is_none() && inputs.candidate_filter.is_none() && inputs.tried.is_empty() {
        return decide(engine, &PlacementRequest::of(model_id), members, picks);
    }
    // Each worker is asked once what holds the request to some of them, as
    // the classic placement asks, and nothing is allocated for the answer.
    let decide_limited = |limits: &mut Limits| {
        limits.refused.clear();
        limits.tried.clear();
        for (position, worker) in workers.iter().enumerate() {
            let admitted = WireConstraint::admits(wire, worker.as_ref())
                && inputs
                    .candidate_filter
                    .is_none_or(|accepts| accepts(worker.as_ref()));
            if admitted {
                if inputs.tried.iter().any(|url| url == worker.base_url()) {
                    limits.tried.extend(members.id_at(position));
                }
            } else {
                limits
                    .refused
                    .push(u32::try_from(position).unwrap_or(u32::MAX));
            }
        }
        let request = PlacementRequest {
            tried: &limits.tried,
            refused: &limits.refused,
            ..PlacementRequest::of(model_id)
        };
        decide(engine, &request, members, picks)
    };
    LIMITS.with(|limits| match limits.try_borrow_mut() {
        Ok(mut limits) => decide_limited(&mut limits),
        // A candidate predicate that places a request of its own on this
        // thread: its lists are its own.
        Err(_) => decide_limited(&mut Limits::default()),
    })
}

/// One decision of `engine` for `request` over `members`, counted under the
/// request's model and timed where its turn comes. A pick counts through
/// the handles `picks` keeps for the pool, with no lookup: a gateway that
/// serves many models pays for none. A request placed on nobody is counted
/// through the calling thread's handles.
fn decide(
    engine: &PolicyEngine,
    request: &PlacementRequest<'_>,
    members: &PoolMembers,
    picks: &KeptPicks,
) -> Option<Arc<dyn Worker>> {
    let started = timed().then(Instant::now);
    let placed = engine.place(request, members);
    let seconds = started.map(|started| started.elapsed().as_secs_f64());
    let (profile, outcome) = match &placed {
        Ok(placement) => (placement.profile(), "pick"),
        Err(unplaced) if unplaced.queued => (unplaced.profile, "queue"),
        Err(unplaced) => (unplaced.profile, "shed"),
    };
    let decision = PolicyDecision {
        generation: engine.spec().generation(),
        profile,
        outcome,
        mode: EMBEDDED,
    };
    let profile_name = || {
        let profiles = engine.spec().engine().profiles();
        let profile = profiles.get(usize::from(profile?))?;
        Some(Arc::from(profile.name.as_str()))
    };
    match &placed {
        Ok(_) => Metrics::count_engine_pick(picks, request.model, decision, profile_name),
        Err(_) => Metrics::count_policy_decision(request.model, decision, profile_name),
    }
    if let Some(seconds) = seconds {
        Metrics::time_policy_decision(EMBEDDED, outcome, seconds);
    }
    placed.ok().map(Placement::into_worker)
}

/// Whether the calling thread times its next decision: its first, and one
/// in [`POLICY_DECISIONS_PER_TIMED`] from then on. Every decision is
/// counted; the two clock reads and the histogram's write are paid by the
/// timed ones alone.
fn timed() -> bool {
    UNTIMED.with(|untimed| match untimed.get() {
        0 => {
            untimed.set(POLICY_DECISIONS_PER_TIMED - 1);
            true
        }
        left => {
            untimed.set(left - 1);
            false
        }
    })
}

#[cfg(test)]
mod tests {
    use std::{sync::Barrier, thread};

    use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
    use smg_policy_controller::document::{AppliedSpec, PolicySpec};
    use smg_worker::{
        metrics::handles::force_on_this_thread,
        test_support::{no_health_check, render_with_recorder},
    };

    use super::*;
    use crate::{
        config::PolicyConfig,
        policies::PolicyRegistry,
        routers::{common::placement::select_single, grpc::backend_client::BasicWorkerBuilder},
    };

    const MODEL: &str = "placed-model";

    fn engine(preset: &str) -> Arc<PolicyEngine> {
        let mut document = PolicySpec::preset(preset).expect("a built-in preset");
        document.generation = 1;
        let spec = AppliedSpec::validate(&document).expect("the preset validates");
        Arc::new(PolicyEngine::new(spec))
    }

    fn worker(url: &str) -> Arc<dyn Worker> {
        worker_of(MODEL, url)
    }

    fn worker_of(model: &str, url: &str) -> Arc<dyn Worker> {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model))
                .health_config(no_health_check())
                .build(),
        );
        worker.set_status(WorkerStatus::Ready);
        worker
    }

    fn fleet(workers: usize) -> (Arc<WorkerRegistry>, Vec<Arc<dyn Worker>>) {
        let registry = Arc::new(WorkerRegistry::new());
        let fleet: Vec<Arc<dyn Worker>> = (0..workers)
            .map(|at| {
                let worker = worker(&format!("http://placed-{at}:8000"));
                assert!(registry.register(Arc::clone(&worker)).is_some());
                worker
            })
            .collect();
        (registry, fleet)
    }

    fn policies(preset: &str) -> PolicyRegistry {
        let policies = PolicyRegistry::new(PolicyConfig::Random);
        policies.set_policy_engine(engine(preset));
        policies
    }

    fn placed(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        inputs: PlacementInputs<'_>,
    ) -> Option<String> {
        select_single(
            registry,
            policies,
            MODEL,
            RoutingPool::HttpRegular,
            None,
            inputs,
        )
        .map(|worker| worker.url().to_owned())
    }

    /// With an engine set the classic placement entry asks it: round robin
    /// by the engine turns one strict rotation over the registry's pool,
    /// where the registry's own policy here is random.
    #[test]
    fn the_placement_asks_the_engine_where_one_is_set() {
        let (registry, fleet) = fleet(5);
        let policies = policies("round_robin");
        let first = placed(&registry, &policies, PlacementInputs::default()).expect("a worker");
        let pool = registry.get_routing_pool(MODEL, RoutingPool::HttpRegular);
        let start = pool
            .iter()
            .position(|worker| worker.url() == first)
            .expect("a worker of the pool");
        for step in 1..=10 {
            let next = placed(&registry, &policies, PlacementInputs::default());
            assert_eq!(
                next.as_deref(),
                Some(pool[(start + step) % fleet.len()].url())
            );
        }
    }

    /// The engine's rows hang on the registry's snapshot, over the pool's
    /// own list: one value a membership, and a new one after a change.
    #[test]
    fn the_members_are_kept_on_the_snapshot_until_the_membership_changes() {
        let (registry, _fleet) = fleet(3);
        let engine = engine("round_robin");
        let snapshot = registry.get_routing_snapshot(MODEL);
        let members = snapshot.pool_members(RoutingPool::HttpRegular, MODEL, &engine);
        assert!(Arc::ptr_eq(
            members.workers(),
            snapshot.pool(RoutingPool::HttpRegular)
        ));
        let again = registry.get_routing_snapshot(MODEL);
        assert!(std::ptr::eq(
            members,
            again.pool_members(RoutingPool::HttpRegular, MODEL, &engine)
        ));
        assert!(registry
            .register(worker("http://placed-new:8000"))
            .is_some());
        let changed = registry.get_routing_snapshot(MODEL);
        let fresh = changed.pool_members(RoutingPool::HttpRegular, MODEL, &engine);
        assert_eq!((members.len(), fresh.len()), (3, 4));
    }

    /// A request held to some workers is never placed on another: a
    /// candidate predicate's refusals are named to the engine, and the
    /// engines a failed attempt tried step aside while another stands.
    #[test]
    fn a_held_request_is_placed_among_the_workers_it_may_take() {
        let (registry, fleet) = fleet(6);
        let policies = policies("round_robin");
        let odd = |worker: &dyn Worker| {
            fleet
                .iter()
                .position(|member| member.url() == worker.url())
                .is_some_and(|at| at % 2 == 1)
        };
        let tried = [fleet[1].base_url().to_owned()];
        let inputs = PlacementInputs {
            candidate_filter: Some(&odd),
            tried: &tried,
            ..PlacementInputs::default()
        };
        for _ in 0..8 {
            let url = placed(&registry, &policies, inputs).expect("a worker");
            assert!(url == fleet[3].url() || url == fleet[5].url(), "{url}");
        }
        // Every worker it may take was tried: the pool it may take is used
        // whole, as the classic placement uses it.
        let all = [
            fleet[1].base_url().to_owned(),
            fleet[3].base_url().to_owned(),
            fleet[5].base_url().to_owned(),
        ];
        let inputs = PlacementInputs {
            candidate_filter: Some(&odd),
            tried: &all,
            ..PlacementInputs::default()
        };
        let url = placed(&registry, &policies, inputs).expect("a worker");
        assert!(odd(registry.get_by_url(&url).expect("registered").as_ref()));
        // Nobody admitted: nobody placed.
        let nobody = |_: &dyn Worker| false;
        let inputs = PlacementInputs {
            candidate_filter: Some(&nobody),
            ..PlacementInputs::default()
        };
        assert_eq!(placed(&registry, &policies, inputs), None);
    }

    /// A model nobody serves is placed on nobody, and nothing is kept for
    /// it on the snapshot every unknown model shares.
    #[test]
    fn a_model_nobody_serves_is_placed_on_nobody() {
        let (registry, _fleet) = fleet(2);
        let policies = policies("round_robin");
        let nobody = select_single(
            &registry,
            &policies,
            "no-such-model",
            RoutingPool::HttpRegular,
            None,
            PlacementInputs::default(),
        );
        assert!(nobody.is_none());
    }

    /// Callers that decide at the same moment do not pile on one worker
    /// under round robin. Four callers decide together, a barrier a round,
    /// over 256 idle workers, 64 rounds: every worker takes exactly one
    /// request, by the classic policy and by the engine alike. No lock
    /// orders the callers on either path: a rotation is one atomic step.
    #[test]
    fn callers_that_decide_at_once_under_round_robin_take_each_idle_worker_once() {
        const CALLERS: usize = 4;
        const ROUNDS: usize = 64;
        for by_engine in [false, true] {
            let (registry, fleet) = fleet(CALLERS * ROUNDS);
            let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
            if by_engine {
                policies.set_policy_engine(engine("round_robin"));
            }
            let round = Barrier::new(CALLERS);
            let taken: Vec<String> = thread::scope(|scope| {
                let callers: Vec<_> = (0..CALLERS)
                    .map(|_| {
                        scope.spawn(|| {
                            let decide = |_| {
                                round.wait();
                                placed(&registry, &policies, PlacementInputs::default())
                                    .expect("an idle fleet takes every request")
                            };
                            (0..ROUNDS).map(decide).collect::<Vec<_>>()
                        })
                    })
                    .collect();
                callers
                    .into_iter()
                    .flat_map(|caller| caller.join().expect("a caller ends"))
                    .collect()
            });
            let piled = fleet
                .iter()
                .filter(|worker| taken.iter().filter(|url| *url == worker.url()).count() != 1)
                .count();
            assert_eq!(taken.len(), fleet.len());
            assert_eq!(
                piled, 0,
                "workers that took no request or more than one, by the engine: {by_engine}"
            );
        }
    }

    /// Every decision is counted under its model, profile and outcome, and
    /// one in sixty-four is timed: the histogram's count is the sample's.
    #[test]
    fn every_decision_is_counted_and_one_in_sixty_four_is_timed() {
        let rendered = render_with_recorder(|| {
            // The thread of this case alone decides: its first decision
            // is timed, and each sixty-fourth after it.
            UNTIMED.with(|untimed| untimed.set(0));
            let (registry, _fleet) = fleet(4);
            let policies = policies("round_robin");
            for _ in 0..130 {
                assert!(placed(&registry, &policies, PlacementInputs::default()).is_some());
            }
        });
        let counted = rendered
            .lines()
            .find(|line| line.starts_with("smg_policy_decisions_total{") && line.contains(MODEL))
            .expect("the decisions are counted");
        assert!(counted.contains("outcome=\"pick\"") && counted.contains("mode=\"embedded\""));
        assert!(counted.ends_with(" 130"), "{counted}");
        let timed = rendered
            .lines()
            .find(|line| line.starts_with("smg_policy_decision_seconds_count{"))
            .expect("some decisions are timed");
        assert!(timed.ends_with(" 3"), "{timed}");
    }

    /// Requests for several models in turn find their counters with no
    /// lookup: each model's pool keeps the handles its picks count through,
    /// one link for the model id, and every pick is counted under its own
    /// model. A request placed on nobody keeps nothing with the pool.
    #[test]
    fn the_picks_of_each_model_count_through_the_handles_its_pool_keeps() {
        const MODELS: [&str; 3] = ["kept-a", "kept-b", "kept-c"];
        let rendered = render_with_recorder(|| {
            force_on_this_thread(Some(true));
            let registry = WorkerRegistry::new();
            for model in MODELS {
                for at in 0..2 {
                    let worker = worker_of(model, &format!("http://{model}-{at}:8000"));
                    assert!(registry.register(worker).is_some());
                }
            }
            let policies = policies("round_robin");
            let select = |model: &str, inputs| {
                let pool = RoutingPool::HttpRegular;
                select_single(&registry, &policies, model, pool, None, inputs)
            };
            for _ in 0..7 {
                for model in MODELS {
                    assert!(select(model, PlacementInputs::default()).is_some());
                }
            }
            let nobody = |_: &dyn Worker| false;
            let refused = PlacementInputs {
                candidate_filter: Some(&nobody),
                ..PlacementInputs::default()
            };
            assert!(select(MODELS[0], refused).is_none());
            for model in MODELS {
                let snapshot = registry.get_routing_snapshot(model);
                assert_eq!(snapshot.pool_picks(RoutingPool::HttpRegular).links(), 1);
            }
            force_on_this_thread(None);
        });
        for model in MODELS {
            let labelled = format!("model=\"{model}\"");
            let of = |name: &str, more: &str| {
                rendered
                    .lines()
                    .find(|line| {
                        line.starts_with(name) && line.contains(&labelled) && line.contains(more)
                    })
                    .expect("the model's series is rendered")
            };
            let picked = of("smg_policy_decisions_total{", "outcome=\"pick\"");
            assert!(picked.ends_with(" 7"), "{picked}");
        }
        let shed = rendered
            .lines()
            .find(|line| line.contains("outcome=\"shed\"") && line.contains("kept-a"))
            .expect("the request nobody took is counted");
        assert!(shed.ends_with(" 1"), "{shed}");
    }
}
