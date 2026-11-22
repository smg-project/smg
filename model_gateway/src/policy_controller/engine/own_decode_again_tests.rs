//! The gateway's engine object over the engine of this process, for a
//! pair's decode asked again behind its prefill's answer: the second
//! decision is made over the decodes the first was made over, as the
//! registry lists them at that instant.

use std::{collections::HashMap, sync::Arc, time::Instant};

use openai_protocol::{
    model_card::ModelCard,
    worker::{WorkerLoadResponse, WorkerStatus},
};
use smg_policy_controller::document::{DecodePick, PolicySpec};
use smg_worker::{
    identity::basic::{ConnectionMode, RuntimeType, Worker, WorkerType},
    load::state::{one_rank_load, TestFeed},
    test_support::no_health_check,
};

use super::{
    mark,
    own_pair_tests::{object, object_of, own, pair_ask, place_of, side, MODEL},
    Placed, PolicyEngine,
};
use crate::{
    observability::cache_trace,
    routers::{
        common::{
            pair_index::{PairSides, PdPairIndex},
            prefill_admission::PrefillLoadGuard,
            snapshot::{ModelWorkerSnapshot, WorkerRegistry},
        },
        grpc::backend_client::BasicWorkerBuilder,
    },
};

/// A pair of `policy_engine` for a request that may take the `allowed`
/// decodes alone, as its route narrows it, with its prefill sent: the
/// decode waits for its second ask.
async fn sent_among(
    policy_engine: &PolicyEngine,
    id: &[u8],
    prefills: &[Arc<dyn Worker>],
    allowed: &[Arc<dyn Worker>],
) -> (Placed, PrefillLoadGuard) {
    let sides: PairSides = [Arc::from(prefills), Arc::from(allowed)];
    let among = |_: &ModelWorkerSnapshot, _: &PdPairIndex| Some(sides.clone());
    let ask = pair_ask(id);
    let placed = policy_engine.place_pair_among(&ask, |_| true, Some(&among), None, None);
    let (mut placed, guard) = placed.await.expect("a pair");
    assert!(placed.decode_waits());
    placed.dispatched(Instant::now());
    (placed, guard)
}

/// A pair of `policy_engine` for a request that may take every worker of
/// the two pools, with its prefill sent.
async fn sent(policy_engine: &PolicyEngine, id: &[u8]) -> (Placed, PrefillLoadGuard) {
    let placed = policy_engine.place_pair(&pair_ask(id), None, None).await;
    let (mut placed, guard) = placed.expect("a pair");
    assert!(placed.decode_waits());
    placed.dispatched(Instant::now());
    (placed, guard)
}

/// A ready HTTP decode of [`MODEL`] at `url`, registered in `registry`
/// behind a pair's first ask.
fn joined(registry: &WorkerRegistry, url: &str) -> Arc<dyn Worker> {
    let builder = BasicWorkerBuilder::new(url)
        .model(ModelCard::new(MODEL))
        .worker_type(WorkerType::Decode)
        .connection_mode(ConnectionMode::Http)
        .runtime_type(RuntimeType::Vllm)
        .health_config(no_health_check());
    let worker: Arc<dyn Worker> = Arc::new(builder.build());
    worker.set_status(WorkerStatus::Ready);
    assert!(registry.register(Arc::clone(&worker)).is_some());
    worker
}

/// The decode `placed` names.
fn decode_of(placed: &Placed) -> Arc<dyn Worker> {
    Arc::clone(placed.decode().expect("a decode"))
}

/// A decode asked again is one the request's route allowed it at its first
/// ask. Here the one decode the request may take stops being ready behind
/// the first pick, and the pool holds another, ready, that the request may
/// not take. The second decision is made over the first one's decodes
/// alone: it sheds, the pair keeps its decode, still chosen, and that
/// decode's dispatch books it.
#[tokio::test]
async fn the_decode_asked_again_stays_off_a_decode_its_first_ask_was_not_allowed() {
    let (policy_engine, prefills, decodes) = own(1, 2, DecodePick::AfterPrefill);
    let id = mark();
    let (mut placed, _guard) = sent_among(&policy_engine, &id, &prefills, &decodes[..1]).await;
    assert!(Arc::ptr_eq(&decode_of(&placed), &decodes[0]));
    decodes[0].set_status(WorkerStatus::NotReady);
    let now = Instant::now();
    let moved = placed.decode_again(&policy_engine, &pair_ask(&id), |_| true, now);
    assert!(
        moved.is_none(),
        "moved to a decode the request may not take"
    );
    assert!(placed.decode_waits(), "a shed leaves the side chosen");
    assert!(Arc::ptr_eq(&decode_of(&placed), &decodes[0]));
    placed.decode_dispatched(now);
    assert!(!placed.decode_waits());
    assert_eq!(policy_engine.faults(), 0);
}

/// The first pick's decode stops being ready while its prefill runs: the
/// second ask moves the pair to another decode the first ask was allowed,
/// whatever other decodes the pool holds.
#[tokio::test]
async fn the_decode_asked_again_moves_to_another_decode_its_first_ask_was_allowed() {
    let (policy_engine, prefills, decodes) = own(1, 4, DecodePick::AfterPrefill);
    for _ in 0..8 {
        let id = mark();
        let allowed = &decodes[..2];
        let (mut placed, _guard) = sent_among(&policy_engine, &id, &prefills, allowed).await;
        let chosen = place_of(allowed, &decode_of(&placed));
        allowed[chosen].set_status(WorkerStatus::NotReady);
        let now = Instant::now();
        let moved = placed.decode_again(&policy_engine, &pair_ask(&id), |_| true, now);
        let moved = moved.cloned();
        allowed[chosen].set_status(WorkerStatus::Ready);
        let moved = moved.expect("the other decode the request may take");
        assert!(Arc::ptr_eq(&moved, &allowed[1 - chosen]));
        assert!(!placed.decode_waits(), "the call's pick issued the side");
        placed.decode_dispatched(now);
    }
}

/// A decode that joined the registry between a pair's two asks is no
/// candidate of the second: the request's route never read it. Here the
/// first pick's decode stops being ready, the pool's other decode leaves
/// and a third joins: the second decision is made over the first one's
/// decodes that the registry still lists, and the pair keeps its decode.
#[tokio::test]
async fn the_decode_asked_again_takes_no_decode_that_joined_behind_its_first_ask() {
    let registry = Arc::new(WorkerRegistry::new());
    side(&registry, WorkerType::Prefill, 1, None);
    let decodes = side(&registry, WorkerType::Decode, 2, None);
    let policy_engine = object(&registry, DecodePick::AfterPrefill);
    let id = mark();
    let (mut placed, _guard) = sent(&policy_engine, &id).await;
    let first = decode_of(&placed);
    let other = &decodes[1 - place_of(&decodes, &first)];
    assert!(registry.remove_by_url(other.url()).is_some());
    joined(&registry, "http://decode-joined:8000");
    first.set_status(WorkerStatus::NotReady);
    let now = Instant::now();
    let moved = placed.decode_again(&policy_engine, &pair_ask(&id), |_| true, now);
    assert!(
        moved.is_none(),
        "moved to a decode its first ask never read"
    );
    assert!(placed.decode_waits());
    assert!(Arc::ptr_eq(&decode_of(&placed), &first));
}

/// The first pick's decode leaves the registry while its prefill runs: the
/// second ask drops it, by the object, and moves the pair to a decode of
/// its first ask that the registry still lists.
#[tokio::test]
async fn the_decode_asked_again_leaves_a_decode_that_left_the_registry() {
    let registry = Arc::new(WorkerRegistry::new());
    side(&registry, WorkerType::Prefill, 1, None);
    let decodes = side(&registry, WorkerType::Decode, 2, None);
    let policy_engine = object(&registry, DecodePick::AfterPrefill);
    let id = mark();
    let (mut placed, _guard) = sent(&policy_engine, &id).await;
    let first = decode_of(&placed);
    let other = &decodes[1 - place_of(&decodes, &first)];
    assert!(registry.remove_by_url(first.url()).is_some());
    let now = Instant::now();
    let moved = placed.decode_again(&policy_engine, &pair_ask(&id), |_| true, now);
    let moved = moved.cloned().expect("the decode the registry still lists");
    assert!(Arc::ptr_eq(&moved, other));
    assert!(!placed.decode_waits());
    placed.decode_dispatched(now);
}

/// A call ahead of the prefill's dispatch is refused: it moves nothing and
/// spends nothing. The call behind the dispatch is then decided over the
/// first ask's decodes.
#[tokio::test]
async fn a_decode_asked_ahead_of_its_prefills_dispatch_moves_nothing() {
    let (policy_engine, prefills, decodes) = own(1, 3, DecodePick::AfterPrefill);
    let id = mark();
    let ask = pair_ask(&id);
    let sides: PairSides = [Arc::from(prefills.as_slice()), Arc::from(&decodes[..2])];
    let among = |_: &ModelWorkerSnapshot, _: &PdPairIndex| Some(sides.clone());
    let placed = policy_engine.place_pair_among(&ask, |_| true, Some(&among), None, None);
    let (mut placed, _guard) = placed.await.expect("a pair");
    let first = decode_of(&placed);
    let other = &decodes[1 - place_of(&decodes[..2], &first)];
    first.set_status(WorkerStatus::NotReady);
    let now = Instant::now();
    assert!(placed
        .decode_again(&policy_engine, &ask, |_| true, now)
        .is_none());
    assert!(placed.decode_waits(), "a refusal leaves the side chosen");
    assert!(Arc::ptr_eq(&decode_of(&placed), &first));
    placed.dispatched(now);
    let moved = placed.decode_again(&policy_engine, &ask, |_| true, now);
    let moved = moved
        .cloned()
        .expect("the other decode the request may take");
    assert!(Arc::ptr_eq(&moved, other));
}

/// Under round robin a pair's second decision keeps the decode of its
/// first, so the decode rotation is turned once a request. Requests that
/// come one at a time then reach every decode at their first decision, and
/// none moves. Decided anew, the second decision would turn the rotation
/// the first one turned: every first decision over two decodes would name
/// the same one.
#[tokio::test]
async fn the_decode_asked_again_under_round_robin_keeps_its_first_decode() {
    let (policy_engine, _prefills, decodes) = own(1, 2, DecodePick::AfterPrefill);
    let mut first = [0; 2];
    for _ in 0..8 {
        let id = mark();
        let (mut placed, _guard) = sent(&policy_engine, &id).await;
        let named = decode_of(&placed);
        first[place_of(&decodes, &named)] += 1;
        let now = Instant::now();
        let again = placed.decode_again(&policy_engine, &pair_ask(&id), |_| true, now);
        let again = again.cloned().expect("a second decision");
        assert!(Arc::ptr_eq(&again, &named), "the pair moved");
        assert!(!placed.decode_waits(), "the call's pick issued the side");
        placed.decode_dispatched(now);
    }
    assert_eq!(first, [4, 4]);
    assert_eq!(policy_engine.faults(), 0);
}

/// Under least load the second decision is made on the reports of its own
/// instant: the pair moves to the decode that reports an empty queue now,
/// away from a first decode that is still ready.
#[tokio::test]
async fn the_decode_asked_again_under_least_load_follows_the_reports() {
    let registry = Arc::new(WorkerRegistry::new());
    let prefills = side(&registry, WorkerType::Prefill, 1, None);
    let decodes = side(&registry, WorkerType::Decode, 2, None);
    let document = PolicySpec::pair_presets("least_load", "least_load");
    let mut document = document.expect("the pair presets");
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    let policy_engine = object_of(&registry, document);
    let feed = TestFeed::attached_to(|receiver| {
        policy_engine.read_loads(receiver.expect("the feed's receiver"));
    });
    // The decode at `idle` reports an empty queue, the other a long one.
    let reports = |idle: usize| -> HashMap<String, WorkerLoadResponse> {
        let queued = |at: usize| if at == idle { 0 } else { 400_000 };
        let decodes = decodes.iter().enumerate();
        let decodes = decodes.map(|(at, worker)| (worker.url().to_owned(), queued(at)));
        let prefill = (prefills[0].url().to_owned(), 0);
        let reported = decodes.chain([prefill]);
        reported
            .map(|(url, queued)| (url, one_rank_load(queued, 0.1, 1_000.0)))
            .collect()
    };
    feed.publish(&reports(0));
    let id = mark();
    let (mut placed, _guard) = sent(&policy_engine, &id).await;
    assert!(Arc::ptr_eq(&decode_of(&placed), &decodes[0]));
    feed.publish(&reports(1));
    let now = Instant::now();
    let moved = placed.decode_again(&policy_engine, &pair_ask(&id), |_| true, now);
    let moved = moved
        .cloned()
        .expect("the decode that reports an empty queue");
    assert!(Arc::ptr_eq(&moved, &decodes[1]));
    assert_eq!(policy_engine.faults(), 0);
}

/// A request that asks for its decision record reads a third line behind
/// its pair's second ask: the decode's own decision, marked as the decode
/// decided again, with the decode the request is sent to. The second line
/// keeps the decode the first decision named.
#[tokio::test]
async fn a_traced_pair_names_the_decode_its_second_ask_took() {
    let (policy_engine, _prefills, _decodes) = own(1, 2, DecodePick::AfterPrefill);
    cache_trace::scope(true, async {
        let id = mark();
        let (mut placed, _guard) = sent(&policy_engine, &id).await;
        let first = decode_of(&placed);
        first.set_status(WorkerStatus::NotReady);
        let now = Instant::now();
        let moved = placed.decode_again(&policy_engine, &pair_ask(&id), |_| true, now);
        let moved = moved.cloned().expect("the other decode");
        let rows = cache_trace::captured_selections();
        assert_eq!(rows.len(), 3, "{rows:?}");
        assert_eq!(rows[0]["worker"], placed.worker().url());
        assert_eq!(rows[1]["worker"], first.url());
        assert!(rows[1]["decision"].get("decode_again").is_none());
        assert_eq!(rows[2]["policy"], "decode");
        assert_eq!(rows[2]["origin"], "policy_controller");
        assert_eq!(rows[2]["worker"], moved.url());
        assert_eq!(rows[2]["decision"]["outcome"], "pick");
        assert_eq!(rows[2]["decision"]["decode_again"], true);
    })
    .await;
}

/// A second decision none of whose decodes the route's rule admits names
/// no worker in its line: the request goes to the decode of the line
/// before, and the decode side's lease is given up.
#[tokio::test]
async fn a_traced_second_decision_the_route_takes_no_decode_of_names_no_worker() {
    let (policy_engine, _prefills, _decodes) = own(1, 2, DecodePick::AfterPrefill);
    cache_trace::scope(true, async {
        let id = mark();
        let (mut placed, _guard) = sent(&policy_engine, &id).await;
        let first = decode_of(&placed);
        let now = Instant::now();
        let moved = placed.decode_again(&policy_engine, &pair_ask(&id), |_| false, now);
        assert!(moved.is_none());
        assert!(Arc::ptr_eq(&decode_of(&placed), &first));
        assert!(!placed.decode_waits(), "the side's lease is given up");
        let rows = cache_trace::captured_selections();
        assert_eq!(rows.len(), 3, "{rows:?}");
        assert_eq!(rows[1]["worker"], first.url());
        assert!(rows[2]["worker"].is_null());
        assert_eq!(rows[2]["decision"]["decode_again"], true);
    })
    .await;
}
