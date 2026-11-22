//! The gateway's engine object over the engine of this process, for a
//! pair: an attempt is decided over the registry's prefill list and decode
//! list of its route's wire, and a target's two workers are the ones at its
//! places in them.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::http::StatusCode;
use openai_protocol::{
    model_card::ModelCard,
    worker::{WorkerLoadResponse, WorkerStatus},
};
use smg_policy_controller::{
    document::{AppliedSpec, DecodePick, Pairing, PolicySpec},
    engine::{Phase, PolicyEngine as OwnEngine, ShedReason, WireModes},
};
use smg_worker::{
    identity::basic::{ConnectionMode, RuntimeType, Worker, WorkerType},
    load::state::{one_rank_load, TestFeed},
    test_support::no_health_check,
};

use super::{mark, profile_of, Ask, DecisionGuard, PolicyEngine, Prompt, Unplaced};
use crate::{
    observability::cache_trace,
    routers::{
        common::{
            placement::refusal_reason,
            prefill_admission::{PrefillAdmission, PrefillAdmissionRejection},
            snapshot::{RoutingPool, WorkerRegistry},
        },
        grpc::backend_client::BasicWorkerBuilder,
    },
};

pub(super) const MODEL: &str = "model-a";

/// A second model, served by workers of its own.
const OTHER: &str = "model-b";

/// One side of a pair: its workers, in the registry's order.
pub(super) type Side = Vec<Arc<dyn Worker>>;

/// The stock pair document, round robin on both sides.
fn pair_document() -> PolicySpec {
    PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets")
}

/// The object over the engine of `document`, as a gateway's first, over
/// `registry`.
pub(super) fn object_of(registry: &Arc<WorkerRegistry>, mut document: PolicySpec) -> PolicyEngine {
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    PolicyEngine::own(Arc::new(OwnEngine::new(spec)), Arc::clone(registry))
}

/// The object over the engine of the stock pair document, picking a pair's
/// decode as `decode` says, over `registry`.
pub(super) fn object(registry: &Arc<WorkerRegistry>, decode: DecodePick) -> PolicyEngine {
    let mut document = pair_document();
    document.plans.pd_pair.decode = decode;
    object_of(registry, document)
}

/// `count` ready HTTP workers of [`MODEL`] on one `side` of a pair,
/// registered in `registry`, their handoff on `connector` where one is
/// stated.
pub(super) fn side(
    registry: &WorkerRegistry,
    side: WorkerType,
    count: usize,
    connector: Option<&str>,
) -> Side {
    side_of(registry, MODEL, side, count, connector)
}

/// [`side`] for the workers of `model`.
fn side_of(
    registry: &WorkerRegistry,
    model: &str,
    side: WorkerType,
    count: usize,
    connector: Option<&str>,
) -> Side {
    let listed = |at: usize| {
        let mut builder = BasicWorkerBuilder::new(format!("http://{model}-{side:?}-{at}:8000"))
            .model(ModelCard::new(model))
            .worker_type(side)
            .connection_mode(ConnectionMode::Http)
            // A runtime whose handoff is its connector's, so a stated
            // connector is a known transport.
            .runtime_type(RuntimeType::Vllm)
            .health_config(no_health_check());
        if let Some(connector) = connector {
            builder = builder.kv_connector(connector);
        }
        let worker: Arc<dyn Worker> = Arc::new(builder.build());
        worker.set_status(WorkerStatus::Ready);
        assert!(registry.register(Arc::clone(&worker)).is_some());
        worker
    };
    (0..count).map(listed).collect()
}

/// The object with `prefills` prefills and `decodes` decodes in its
/// registry, every prefill pairing with every decode.
pub(super) fn own(
    prefills: usize,
    decodes: usize,
    decode: DecodePick,
) -> (PolicyEngine, Side, Side) {
    let registry = Arc::new(WorkerRegistry::new());
    let prefills = side(&registry, WorkerType::Prefill, prefills, None);
    let decodes = side(&registry, WorkerType::Decode, decodes, None);
    (object(&registry, decode), prefills, decodes)
}

/// A pair's ask on the HTTP route, as its router states it.
pub(super) fn pair_ask(id: &[u8]) -> Ask<'_> {
    Ask {
        phase: Phase::Prefill,
        decode_again: true,
        wires: WireModes::HTTP,
        ..Ask::of(id, MODEL)
    }
}

/// The place of `worker` among `workers`, the objects the registry lists.
pub(super) fn place_of(workers: &[Arc<dyn Worker>], worker: &Arc<dyn Worker>) -> usize {
    let place = workers
        .iter()
        .position(|listed| Arc::ptr_eq(listed, worker));
    place.expect("a worker the registry lists")
}

/// A pair under round robin is placed by the engine of this process over
/// the registry's two lists. The placement holds the worker objects the
/// registry lists, every worker of each side takes a turn, both sides are
/// issued by the one decision, and no store stands behind the object.
#[tokio::test]
async fn a_pair_is_placed_over_the_registrys_two_lists() {
    let (policy_engine, prefills, decodes) = own(2, 3, DecodePick::WithPrefill);
    assert!(policy_engine.door.store().is_none());
    let (mut on_prefill, mut on_decode) = ([0_u32; 2], [0_u32; 3]);
    for _ in 0..12 {
        let id = mark();
        let placed = policy_engine.place_pair(&pair_ask(&id), None, None).await;
        let (mut placed, _guard) = placed.expect("a pair");
        on_prefill[place_of(&prefills, placed.worker())] += 1;
        on_decode[place_of(&decodes, placed.decode().expect("a decode"))] += 1;
        assert!(!placed.decode_waits(), "issued with its prefill");
        let now = Instant::now();
        placed.dispatched(now);
        placed.decode_dispatched(now);
    }
    assert!(on_prefill.iter().chain(&on_decode).all(|&times| times > 0));
    assert_eq!(policy_engine.faults(), 0);
}

/// A request that asks for its decision record reads the pair's two picks:
/// the decision with the prefill taken, then the decode taken with it, each
/// under the profile the document resolves its side to.
#[tokio::test]
async fn a_traced_pair_names_both_picks() {
    let (policy_engine, _prefills, _decodes) = own(2, 2, DecodePick::WithPrefill);
    cache_trace::scope(true, async {
        let id = mark();
        let placed = policy_engine.place_pair(&pair_ask(&id), None, None).await;
        let (placed, _guard) = placed.expect("a pair");
        let rows = cache_trace::captured_selections();
        assert_eq!(rows.len(), 2, "{rows:?}");
        assert_eq!(rows[0]["policy"], "prefill");
        assert_eq!(rows[0]["worker"], placed.worker().url());
        assert_eq!(rows[1]["policy"], "decode");
        assert_eq!(rows[1]["worker"], placed.decode().expect("a decode").url());
        for row in &rows {
            assert_eq!(row["origin"], "policy_controller");
            assert_eq!(row["decision"]["outcome"], "pick");
            assert_eq!(row["decision"]["generation"], 1);
        }
    })
    .await;
}

/// A retry names the two workers its failed attempt was sent to, by the
/// ids its placement carried, and is placed on neither while another pair
/// stands: a retry steps off the pair that just failed.
#[tokio::test]
async fn a_retry_is_placed_off_the_pair_it_tried() {
    let (policy_engine, _prefills, _decodes) = own(2, 2, DecodePick::WithPrefill);
    let id = mark();
    let first = policy_engine.place_pair(&pair_ask(&id), None, None).await;
    let (first, guard) = first.expect("a pair");
    let tried = [first.id(), first.decode_id().expect("a decode")];
    let failed = [first.worker(), first.decode().expect("a decode")].map(Arc::clone);
    drop((first, guard));
    for _ in 0..6 {
        let retry = Ask {
            tried: &tried,
            attempt: Some(1),
            ..pair_ask(&id)
        };
        let placed = policy_engine.place_pair(&retry, None, None).await;
        let (placed, _guard) = placed.expect("another pair");
        assert!(!Arc::ptr_eq(placed.worker(), &failed[0]));
        assert!(!Arc::ptr_eq(placed.decode().expect("a decode"), &failed[1]));
    }
}

/// Prefills that are all at the router's limit empty the pair's decision,
/// and that request waits for room: it is told the queue's own verdict when
/// its wait ends, never a missing endpoint, and it is placed once a prefill
/// has room.
#[tokio::test]
async fn a_pair_waits_while_every_prefill_is_at_its_limit() {
    let (policy_engine, _prefills, _decodes) = own(1, 1, DecodePick::WithPrefill);
    let admission = PrefillAdmission::new(1, 1, Duration::from_millis(30));
    let id = mark();
    let ask = pair_ask(&id);
    let held = policy_engine.place_pair(&ask, Some(&admission), None).await;
    let held = held.expect("a pair");
    let waited = policy_engine.place_pair(&ask, Some(&admission), None).await;
    let timed_out = Unplaced::Admission(PrefillAdmissionRejection::QueueTimeout);
    assert_eq!(waited.err(), Some(timed_out));
    drop(held);
    let placed = policy_engine.place_pair(&ask, Some(&admission), None).await;
    assert!(placed.is_ok(), "{:?}", placed.err());
}

/// Under a document that picks the decode after the prefill, a pair's
/// decode stands chosen at its commit, and its second ask is decided over
/// the registry's lists of that instant. A decode that stopped being ready
/// behind the first pick is passed over, and the placement moves to the
/// worker at the new target's place.
#[tokio::test]
async fn the_decode_asked_again_is_placed_over_the_lists_of_its_own_instant() {
    let (policy_engine, _prefills, decodes) = own(1, 2, DecodePick::AfterPrefill);
    let id = mark();
    let ask = pair_ask(&id);
    let placed = policy_engine.place_pair(&ask, None, None).await;
    let (mut placed, _guard) = placed.expect("a pair");
    assert!(placed.decode_waits());
    let chosen = place_of(&decodes, placed.decode().expect("a decode"));
    let now = Instant::now();
    placed.dispatched(now);
    decodes[chosen].set_status(WorkerStatus::NotReady);
    let moved = placed.decode_again(&policy_engine, &ask, |_| true, now);
    let moved = moved.cloned().expect("the other decode");
    assert!(Arc::ptr_eq(&moved, &decodes[1 - chosen]));
    assert!(!placed.decode_waits(), "the call's pick issued the side");
    placed.decode_dispatched(now);
}

/// The profile a pair of `model` is decided under by `policy_engine`, or why
/// it has no pair.
async fn decided_under(policy_engine: &PolicyEngine, model: &str) -> Result<Option<u16>, Unplaced> {
    let id = mark();
    let ask = Ask {
        model,
        ..pair_ask(&id)
    };
    let (placed, _guard) = policy_engine.place_pair(&ask, None, None).await?;
    let decided = placed.handle.as_ref().map(DecisionGuard::decision);
    Ok(decided.and_then(profile_of))
}

/// A pair is decided under the binding that names its model: the engine of
/// this process finds the binding by the request's model. Under a document
/// whose first pair binding names one model and whose second names every
/// model, each model's pairs are decided under its own binding's profile.
/// Under the first binding alone, a pair of a model it does not name is
/// declined by its binding, which every router answers as a 503 that names
/// the binding, and the model it names is placed.
#[tokio::test]
async fn a_pair_is_decided_under_the_binding_that_names_its_model() {
    let registry = Arc::new(WorkerRegistry::new());
    for model in [MODEL, OTHER] {
        side_of(&registry, model, WorkerType::Prefill, 2, None);
        side_of(&registry, model, WorkerType::Decode, 2, None);
    }
    let mut named = pair_document();
    named.bindings[0].matcher.model = MODEL.to_owned();
    // A second pair binding, for every model, under profiles of its own.
    let mut both = named.clone();
    let mut every = pair_document().bindings.remove(0);
    for (side, profile) in &mut every.profiles {
        let own = format!("{side}-of-every-model");
        both.profiles
            .insert(own.clone(), both.profiles[profile.as_str()].clone());
        *profile = own;
    }
    both.bindings.push(every);

    let policy_engine = object_of(&registry, both);
    let under_named = decided_under(&policy_engine, MODEL).await;
    let under_every = decided_under(&policy_engine, OTHER).await;
    let (under_named, under_every) = (under_named.expect("a pair"), under_every.expect("a pair"));
    assert!(under_named.is_some() && under_every.is_some());
    assert_ne!(under_named, under_every);

    let policy_engine = object_of(&registry, named);
    assert!(decided_under(&policy_engine, MODEL).await.is_ok());
    let declined = decided_under(&policy_engine, OTHER).await;
    let declined = declined.expect_err("no binding names the model");
    assert!(
        matches!(
            declined,
            Unplaced::Shed {
                reason: ShedReason::PolicyDeclined,
                emptied_by: "binding",
                ..
            }
        ),
        "{declined:?}"
    );
    let answer = declined.answer(OTHER);
    assert_eq!(answer.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        refusal_reason(&answer),
        Some("the policy declined at binding")
    );
}

/// A request whose binding plans no side for its phase is declined by that
/// binding. A pair binding that names no phase covers the single phase,
/// where it plans nothing: a single request of its models is declined.
/// Under a single binding of one model ahead of the pair's, a pair of that
/// model has no side either, and another model's pair is placed.
#[tokio::test]
async fn a_request_with_no_side_under_its_binding_is_declined() {
    let registry = Arc::new(WorkerRegistry::new());
    side_of(&registry, MODEL, WorkerType::Regular, 1, None);
    for model in [MODEL, OTHER] {
        side_of(&registry, model, WorkerType::Prefill, 1, None);
        side_of(&registry, model, WorkerType::Decode, 1, None);
    }
    let by_binding = |declined: &Unplaced| {
        let declined_by = ShedReason::PolicyDeclined;
        matches!(declined, Unplaced::Shed { reason, emptied_by: "binding", .. } if *reason == declined_by)
    };

    let policy_engine = object_of(&registry, pair_document());
    let id = mark();
    let single = Ask {
        pool: Some(RoutingPool::HttpRegular),
        ..Ask::of(&id, MODEL)
    };
    let declined = policy_engine.place(&single, Instant::now()).await;
    let declined = declined.expect_err("a pair binding plans no single request");
    assert!(by_binding(&declined), "{declined:?}");

    let mut document = pair_document();
    let single = PolicySpec::preset("round_robin").expect("a built-in preset");
    document.profiles.extend(single.profiles.clone());
    let mut named = single.bindings[0].clone();
    named.matcher.model = MODEL.to_owned();
    document.bindings.insert(0, named);
    let policy_engine = object_of(&registry, document);
    let declined = decided_under(&policy_engine, MODEL).await;
    let declined = declined.expect_err("a single binding plans no pair");
    assert!(by_binding(&declined), "{declined:?}");
    assert!(decided_under(&policy_engine, OTHER).await.is_ok());
}

/// A fleet whose prefills pair with none of its decodes reads the pair
/// index's own refusal, as the selection without the policy controller
/// answers it, and not a missing endpoint.
#[tokio::test]
async fn a_fleet_that_pairs_nothing_reads_the_pair_indexs_refusal() {
    let registry = Arc::new(WorkerRegistry::new());
    side(&registry, WorkerType::Prefill, 1, Some("NixlConnector"));
    side(&registry, WorkerType::Decode, 1, Some("MooncakeConnector"));
    let policy_engine = object(&registry, DecodePick::WithPrefill);
    let id = mark();
    let unplaced = policy_engine.place_pair(&pair_ask(&id), None, None).await;
    let unplaced = unplaced.err();
    let transport = ["transport".to_owned()];
    assert!(
        matches!(&unplaced, Some(Unplaced::NoCompatiblePair { mismatches, .. }) if *mismatches == transport),
        "{unplaced:?}"
    );
}

/// A pair under least load on both sides is placed by the reports the load
/// monitor publishes. The engine object reads the monitor's feed, and each
/// decision is lent the snapshot current at that instant, which the records
/// of both lists read: the prefill and the decode that report an empty
/// queue take every pair, and the next publication moves the placement at
/// the next decision.
#[tokio::test]
async fn a_least_load_pair_is_placed_by_the_reports_the_monitor_publishes() {
    let registry = Arc::new(WorkerRegistry::new());
    let prefills = side(&registry, WorkerType::Prefill, 3, None);
    let decodes = side(&registry, WorkerType::Decode, 3, None);
    let document = PolicySpec::pair_presets("least_load", "least_load");
    let policy_engine = object_of(&registry, document.expect("the pair presets"));
    let feed = TestFeed::attached_to(|receiver| {
        policy_engine.read_loads(receiver.expect("the feed's receiver"));
    });
    assert!(policy_engine.reads_loads());
    // One worker a side reports an empty queue, every other a long one.
    let reports = |idle: [usize; 2]| -> HashMap<String, WorkerLoadResponse> {
        let listed = [&prefills, &decodes].into_iter().zip(idle);
        let reported = listed.flat_map(|(side, idle)| {
            side.iter().enumerate().map(move |(at, worker)| {
                let queued = if at == idle { 0 } else { 400_000 };
                (worker.url().to_owned(), one_rank_load(queued, 0.1, 1_000.0))
            })
        });
        reported.collect()
    };
    for idle in [[2, 0], [0, 1], [1, 2]] {
        feed.publish(&reports(idle));
        for _ in 0..8 {
            let id = mark();
            let placed = policy_engine.place_pair(&pair_ask(&id), None, None).await;
            let (placed, _guard) = placed.expect("a pair");
            let on = [
                place_of(&prefills, placed.worker()),
                place_of(&decodes, placed.decode().expect("a decode")),
            ];
            assert_eq!(on, idle);
        }
    }
    assert_eq!(policy_engine.faults(), 0);
}

/// A pair under the bucket policy states the size of its prompt: the bucket
/// stage places a prefill by that size and declares it. The route's hashing
/// counts the tokens it read, and the pair's lease states that count.
#[tokio::test]
async fn a_bucket_pair_states_the_size_of_its_prompt() {
    let registry = Arc::new(WorkerRegistry::new());
    side(&registry, WorkerType::Prefill, 2, None);
    side(&registry, WorkerType::Decode, 2, None);
    let document = PolicySpec::pair_presets("bucket", "round_robin");
    let policy_engine = object_of(&registry, document.expect("the pair presets"));
    let tokens: Vec<u32> = (0..600).collect();
    let hashes = policy_engine.hashes(MODEL, &Prompt::of(Some(&tokens), None));
    let id = mark();
    let ask = Ask {
        hashes: Some(&hashes),
        ..pair_ask(&id)
    };
    let placed = policy_engine.place_pair(&ask, None, None).await;
    let (placed, _guard) = placed.expect("a pair");
    assert_eq!(placed.lease_tokens(), Some(600));
}

/// A pair is made by the pairing rule of the document that decides it. The
/// fleet whose two sides state different transports pairs nothing under
/// the lenient rule, and is placed under a document that pairs on nothing.
#[tokio::test]
async fn a_pair_is_made_by_the_documents_own_pairing() {
    let registry = Arc::new(WorkerRegistry::new());
    side(&registry, WorkerType::Prefill, 1, Some("NixlConnector"));
    side(&registry, WorkerType::Decode, 1, Some("MooncakeConnector"));
    let id = mark();
    let lenient = object_of(&registry, pair_document());
    let unpaired = lenient.place_pair(&pair_ask(&id), None, None).await;
    assert!(matches!(
        unpaired.err(),
        Some(Unplaced::NoCompatiblePair { .. })
    ));
    let mut document = pair_document();
    document.plans.pd_pair.pairing = Pairing::Off;
    let on_nothing = object_of(&registry, document);
    let placed = on_nothing.place_pair(&pair_ask(&id), None, None).await;
    assert!(placed.is_ok(), "{:?}", placed.err());
}
