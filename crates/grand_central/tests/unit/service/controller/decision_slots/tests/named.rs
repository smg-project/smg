//! The cases of a pair's frame that names its worker through the deciding
//! threads: the later decode it states, and the decode call behind it.

use std::{sync::atomic::Ordering, time::Duration};

use super::{
    applied, bed, booked, busy, call, decode_on_file, outcome, pair_frame, pair_of, ready, sent,
    sides_of, Bed, Step, GATEWAY, MODEL,
};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::view::{Role, WireModes, WorkerId},
    service::proto::{self, busy::Reason, decision::Outcome},
    store::{LeaseState, LeasesStore},
    tracking::lease::{LeaseId, Side, DISPATCH_TTL},
};

/// The preset whose profile is the plain hash ring: the one a side reads
/// the number under.
const RING: &str = "consistent_hashing";

/// The stock pair of the presets `prefill` and `decode` as `generation`. A
/// side that is the hash ring brings the binding its target source.
fn pair_spec(prefill: &str, decode: &str, generation: u64) -> AppliedSpec {
    let mut document = PolicySpec::pair_presets(prefill, decode).expect("presets of this build");
    document.generation = generation;
    applied(&document)
}

/// A leader under the stock pair of two hash rings, behind one decider,
/// over two prefills and two decodes. With the bed: the prefills and then
/// the decodes, each side in the order of the list the bed's holder lends,
/// the order a number counts in.
fn rings_bed() -> (Bed, [Vec<WorkerId>; 2]) {
    let bed = bed(Some(pair_spec(RING, RING, 1)), true, (1, 8), []);
    for (name, role) in [("p1", Role::Prefill), ("d1", Role::Decode)] {
        ready(
            &bed.store,
            &bed.reads.objects,
            name,
            MODEL,
            role,
            WireModes::NONE,
        );
    }
    let objects = &bed.reads.objects;
    let generation = objects.registry().model_generation(MODEL);
    let generation = generation.expect("the model's members");
    let listed = [Role::Prefill, Role::Decode].map(|role| {
        let lent = objects.lent(&generation, role);
        let member = |url: &str| bed.store.by_key(url);
        let ids = lent
            .workers
            .iter()
            .map(|worker| member(worker.url()).expect("a member of the fleet").id);
        ids.collect()
    });
    (bed, listed)
}

/// `frame` with its request's worker named by `number`.
fn naming(mut frame: proto::DecideRequest, number: u32) -> proto::DecideRequest {
    frame.named_worker = Some(number);
    frame
}

/// `frame`'s answer, handed over and decided `at` after the take: the
/// case's clock shows that instant.
async fn asked_at(bed: &mut Bed, frame: proto::DecideRequest, at: Duration) -> proto::DecideAnswer {
    assert_eq!(bed.submit_at(frame, GATEWAY, at), None);
    bed.answer().await
}

/// A pair asked under `id` with its decode later, which names its worker
/// by `number` where the case names one, and its prefill dispatched: the
/// lease's id.
async fn held_later(bed: &mut Bed, id: u8, number: Option<u32>) -> u64 {
    let frame = pair_frame(id, true);
    let frame = match number {
        Some(number) => naming(frame, number),
        None => frame,
    };
    let answer = bed.ask(frame).await;
    let lease = sides_of(&answer).first().expect("the prefill's lease").0;
    let at = bed.script.taken;
    assert!(bed
        .store
        .dispatched(GATEWAY, LeaseId(lease), sent(Side::Prefill), at));
    lease
}

/// A pair's frame that names its two workers keeps its word of its later
/// decode. The frame's decision is held to the two workers, and its answer
/// carries the prefill side's lease alone: the decode side is filed chosen
/// with the one decode the request named, and books nothing. The same
/// frame with no later decode is issued with both sides, as any pair's is.
///
/// The prefill is dispatched and the clock passes the dispatch deadline. A
/// decision behind it ends the pair that was issued whole and never
/// dispatched, and the chosen side still stands: no deadline bounds it.
/// The decode call that names the worker is refused as one whose decode
/// never stood chosen, and moves nothing. A dispatch that names the decode
/// then books the side on the worker the request named.
#[tokio::test]
async fn a_named_pairs_later_decode_stands_chosen_past_the_dispatch_deadline() {
    let (mut bed, [prefills, decodes]) = rings_bed();
    let (prefill, decode) = (prefills[1], decodes[1]);
    let named = [(prefill.0, Some(decode.0))];
    let answer = bed.ask(naming(pair_frame(1, true), 1)).await;
    let (targets, sides) = pair_of(&answer);
    assert_eq!(targets, named, "held to the two workers");
    let later = sides[0].0;
    assert_eq!(sides, [(later, proto::Side::Prefill, 0)]);
    assert_eq!(decode_on_file(&bed, later), (LeaseState::Chosen, decode));
    assert_eq!(
        booked(&bed, [prefill, decode]),
        [1, 0],
        "chosen books nothing"
    );

    let answer = bed.ask(naming(pair_frame(2, false), 1)).await;
    let (targets, sides) = pair_of(&answer);
    assert_eq!(targets, named, "held to the two workers");
    let whole = sides[0].0;
    let both = [proto::Side::Prefill, proto::Side::Decode].map(|side| (whole, side, 0));
    assert_eq!(sides, both, "issued with both sides");
    assert_eq!(decode_on_file(&bed, whole), (LeaseState::ISSUED, decode));
    assert_eq!(booked(&bed, [prefill, decode]), [2, 1]);

    let at = bed.script.taken;
    let sent_to = |bed: &Bed, side: Side, at| {
        let store = &bed.store;
        store.dispatched(GATEWAY, LeaseId(later), sent(side), at)
    };
    assert!(sent_to(&bed, Side::Prefill, at));
    let late = DISPATCH_TTL + Duration::from_secs(1);
    bed.script.steps.lock().push_back(Step::At(late));
    let behind = asked_at(&mut bed, pair_frame(3, false), late).await;
    assert!(matches!(outcome(&behind), Outcome::Pick(_)), "{behind:?}");
    let on_file = bed.store.sides(GATEWAY, LeaseId(whole));
    assert_eq!(on_file, None, "abandoned whole");
    assert_eq!(decode_on_file(&bed, later), (LeaseState::Chosen, decode));

    let refused = asked_at(&mut bed, naming(call(4, later), 1), late).await;
    assert_eq!(busy(&refused), Reason::DecodeNotChosen);
    assert_eq!(decode_on_file(&bed, later), (LeaseState::Chosen, decode));
    let before = booked(&bed, [decode]);
    assert!(sent_to(&bed, Side::Decode, at + late));
    let (state, on) = decode_on_file(&bed, later);
    assert!(matches!(state, LeaseState::Dispatched { .. }), "{state:?}");
    assert_eq!(on, decode, "the decode the request named");
    assert_eq!(booked(&bed, [decode]), before.map(|held| held + 1));
}

/// A decode call that names its worker is refused where the decode side of
/// the binding in force at the call reads the number, and is judged by
/// nothing else: not by what the pair's first frame stated, and not by the
/// spec that decided it.
///
/// Under the pair of two hash rings a call whose frame leaves the number
/// out is decided again as any pair's decode, though its pair's first
/// frame named its worker. A call that names its worker is refused as one
/// whose decode never stood chosen, though its pair's first frame named
/// none, and its side stands chosen. Behind a spec whose decode side is no
/// ring, a call that names its worker is decided again. Behind the two
/// rings again, such a call is refused. Each refusal is counted once, at
/// its reason.
#[tokio::test]
async fn a_decode_call_that_names_its_worker_is_judged_by_the_spec_in_force_at_the_call() {
    let (mut bed, _) = rings_bed();
    let unnamed = held_later(&mut bed, 1, None).await;
    let named = held_later(&mut bed, 2, Some(0)).await;
    let third = held_later(&mut bed, 3, Some(1)).await;
    let fourth = held_later(&mut bed, 4, Some(0)).await;
    let decided = |answer: &proto::DecideAnswer, lease: u64| {
        assert!(matches!(outcome(answer), Outcome::Pick(_)), "{answer:?}");
        assert_eq!(sides_of(answer), [(lease, proto::Side::Decode, 0)]);
    };
    let chosen = |bed: &Bed, lease: u64| decode_on_file(bed, lease).0.is_chosen();

    decided(&bed.ask(call(5, named)).await, named);
    let refused = bed.ask(naming(call(6, unnamed), 0)).await;
    assert_eq!(busy(&refused), Reason::DecodeNotChosen);
    assert!(chosen(&bed, unnamed), "the side stands chosen");

    let turning = pair_spec(RING, "round_robin", 2);
    bed.store.apply(turning).expect("a later generation");
    decided(&bed.ask(naming(call(7, third), 1)).await, third);

    let rings = pair_spec(RING, RING, 3);
    bed.store.apply(rings).expect("a later generation");
    let refused = bed.ask(naming(call(8, fourth), 0)).await;
    assert_eq!(busy(&refused), Reason::DecodeNotChosen);
    assert!(chosen(&bed, fourth), "the side stands chosen");
    let counted = bed.counts.busy[Reason::DecodeNotChosen as usize].load(Ordering::Relaxed);
    assert_eq!(counted, 2, "each refusal at its reason");
}
