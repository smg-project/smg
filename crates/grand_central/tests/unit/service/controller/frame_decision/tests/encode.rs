//! The cases of an encode request's frame: its ask [`decide_encode`] as a
//! deciding thread asks it, over the three lists the store's holder lends
//! for its model, and the two that file its pick ask [`issue_pair`] behind
//! it.

use std::{
    collections::BTreeSet,
    sync::{atomic::Ordering, Arc},
    time::{Duration, Instant},
};

use smg_worker::load::state::LoadSnapshot;

use super::{
    a_frame_under_the_second_files_watch, holder, pairs_of, shed_of, staged, stock_pair_plan,
    taken_at, ByReports, Checked, FrameBuffers, Part, GATEWAY, OTHER,
};
use crate::{
    document::spec::PolicySpec,
    engine::{
        decision::{Decision, ShedReason, Targets, SUBSET_HINT, WIRES},
        decision_slot::DecisionSlot,
        framework::{Picker, TieBand},
        view::{Candidate, RequestFacts, Role, WireModes, WorkerId},
        DemandFacts,
    },
    service::{
        controller::{
            decision_slots::tests::encode::{
                encode_plan, encode_plan_in, frame_of, member_of, member_stating, Stated, ENCODED,
                PUSHING,
            },
            frame_decision::{decide_encode, issue_pair},
            wire,
        },
        proto,
    },
    stages::{lifecycle::Lifecycle, pickers::target_of, registry::StageRegistry},
    store::{
        GatewayId, HealthSink, LeaseState, LeasesStore, LookOutcome, MemoryStore, WorkersStore,
    },
    tracking::lease::Side,
    worker_host::{Members, Objects},
};

/// [`member_of`] for [`ENCODED`].
fn member(
    store: &MemoryStore,
    objects: &Objects<Members>,
    name: &str,
    role: Role,
    stated: Stated,
) -> WorkerId {
    member_of(store, objects, ENCODED, name, role, stated)
}

/// A store under the encode plan.
fn encode_store() -> Arc<MemoryStore> {
    let store = Arc::new(MemoryStore::new());
    store.apply(encode_plan()).expect("above the unbound start");
    store
}

/// [`frame_of`] for [`ENCODED`], from a gateway that states no wires and no
/// hint.
fn encode_frame(n: u8, keys: u64) -> proto::DecideRequest {
    frame_of(ENCODED, n, keys, WireModes::NONE, None)
}

/// One encode request's decision on `frames`, a deciding thread's buffers,
/// and where its answer says its items go (`DecideAnswer.encode`): what
/// the thread asks for `frame` as `gateway`'s at `now`. The load monitor
/// published nothing: the request reads no report.
fn decided_on(
    frames: &mut FrameBuffers,
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    (gateway, now): (GatewayId, Instant),
) -> (Decision, Option<proto::EncodePlaced>) {
    let request = Checked::new(frame).expect("a frame the record holds");
    assert!(request.carries_items(), "an encode request's frame");
    let own = Part::default();
    let loads = Arc::new(LoadSnapshot::default());
    let decision = {
        let taken = taken_at(&request, gateway, (now, now), None);
        let decided =
            store.deciding(|world| decide_encode(world, objects, &taken, frames, &loads, &own));
        decided.expect("the engine decides").0
    };
    let (id, placed) = frames.answered(request, &decision);
    let answer = wire::decided(id, 1, &decision, None, placed);
    (decision, answer.encode)
}

/// [`decided_on`] with buffers of the frame's own.
fn served_encode(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    gateway: GatewayId,
    now: Instant,
) -> (Decision, Option<proto::EncodePlaced>) {
    let mut frames = FrameBuffers::default();
    decided_on(&mut frames, store, objects, frame, (gateway, now))
}

/// The encoders an answer names, by id, in the keys' order.
fn encoders_of(placed: Option<&proto::EncodePlaced>) -> Vec<WorkerId> {
    let encoders = placed.into_iter().flat_map(|placed| &placed.encoders);
    encoders.map(|encoder| WorkerId(encoder.id)).collect()
}

/// An encode request's frame is decided for its pair and for each of its
/// items, over its model's three lists. The pick names the prefill with its
/// decode, and the answer names each key's encoder beside it, in the keys'
/// order, by the store's id and with no pair. An encoder that states no end
/// takes no item. A request that states the items' one key has that key
/// placed in the items' stead by a class that pushes, and the answer says
/// so.
#[test]
fn an_encode_request_is_decided_for_its_pair_and_for_each_items_encoder() {
    let store = encode_store();
    let objects = holder(&store);
    let p = member(&store, &objects, "p0", Role::Prefill, PUSHING);
    let d = member(&store, &objects, "d0", Role::Decode, PUSHING);
    let e = member(&store, &objects, "e0", Role::Encode, PUSHING);
    let unstated = Stated {
        pushes: false,
        ..PUSHING
    };
    member(&store, &objects, "e1", Role::Encode, unstated);
    let now = Instant::now();
    for n in 1..=16 {
        let frame = encode_frame(n, 2);
        let (answer, placed) = served_encode(&store, &objects, frame, GATEWAY, now);
        assert_eq!(pairs_of(&answer), [(p, d)], "frame {n}: {answer:?}");
        let placed = placed.expect("a pick of an encode request says where its items go");
        assert!(!placed.one, "frame {n}: each item on its own");
        let named = |encoder: &proto::Target| (WorkerId(encoder.id), encoder.pair.is_none());
        let encoders: Vec<_> = placed.encoders.iter().map(named).collect();
        assert_eq!(encoders, [(e, true), (e, true)], "frame {n}: a key an item");
    }
    let mut frame = encode_frame(17, 2);
    frame.encode.as_mut().expect("the case's items").one = Some(proto::Identity {
        form: Some(proto::identity::Form::Hash(99)),
    });
    let (answer, placed) = served_encode(&store, &objects, frame, GATEWAY, now);
    assert_eq!(pairs_of(&answer), [(p, d)]);
    let placed = placed.expect("a pick of an encode request says where its items go");
    assert_eq!((encoders_of(Some(&placed)), placed.one), (vec![e], true));
}

/// A request's items are taken as one where it states their one key, with
/// a form or with none: the form is only what that one placement is placed
/// by. So a one key that is stated with no form crosses a link as it is.
/// The crate's own client makes the frame. A class that pushes places the
/// key in the items' stead, with no affinity, as the engine's pass does in
/// process ([`Engine::decide_encode_pair`](crate::engine::Engine::decide_encode_pair)).
/// The client reads one encoder for the items, and that they were placed as
/// one. A host that states no one key is answered an encoder an item.
///
/// The classic router takes one encoder a prompt under a push handoff with
/// no image to fan out on, whatever the keys say. Its selection hands one
/// hash for all the items
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2680-2693`
/// on main cb9aca544), and its push plan refuses a second encoder
/// (`model_gateway/src/routers/grpc/common/stages/encode.rs:328-333`).
#[cfg(feature = "client")]
#[test]
fn a_one_key_with_no_form_crosses_a_link_as_one_placement() {
    use crate::{
        engine::{
            decision_engine::Request,
            framework::{EncodeItems, EncodePlaced},
            view::{Constraints, Identity, Phase},
        },
        service::client::wire::{encode_request, placed},
    };
    let store = encode_store();
    let objects = holder(&store);
    let p = member(&store, &objects, "p0", Role::Prefill, PUSHING);
    let d = member(&store, &objects, "d0", Role::Decode, PUSHING);
    let e0 = member(&store, &objects, "e0", Role::Encode, PUSHING);
    let e1 = member(&store, &objects, "e1", Role::Encode, PUSHING);
    let now = Instant::now();
    let keys = [11, 12, 13].map(Identity::hashed);
    let each = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let formless = EncodeItems {
        one: Some(Identity::NONE),
        ..each
    };
    let hashed = EncodeItems {
        one: Some(Identity::hashed(99)),
        ..each
    };
    let mut room = EncodePlaced::new();
    // One request over the link: the frame the client makes of it, the
    // controller's decision, and the answer as the client reads it.
    let mut linked = |n: u8, items: EncodeItems<'_>| {
        let id = [n];
        let asked = Request {
            id: &id,
            facts: RequestFacts {
                model: ENCODED,
                seed: u64::from(n),
                phase: Phase::Prefill,
                budget_micros: 200_000,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        };
        let (frame, _) = encode_request(&asked, items);
        let (answer, told) = served_encode(&store, &objects, frame, GATEWAY, now);
        assert_eq!(pairs_of(&answer), [(p, d)], "frame {n}: {answer:?}");
        placed(told.as_ref(), &answer, &items, &mut room).expect("an answer the client reads");
        let encoders: Vec<_> = room.encoders().iter().map(|encoder| encoder.id).collect();
        assert!(encoders.iter().all(|id| [e0, e1].contains(id)), "frame {n}");
        (encoders.len(), room.one())
    };
    for n in 1..=16 {
        assert_eq!(linked(n, formless), (1, true), "frame {n}: no form");
        assert_eq!(linked(n, hashed), (1, true), "frame {n}: a hash");
        assert_eq!(linked(n, each), (3, false), "frame {n}: no one key");
    }
}

/// The field's presence marks an encode request, with a key or with none.
/// One with no key is decided as its pair and takes no encoder: its answer
/// carries the field with no encoder in it. Nothing is read of the
/// encoders for it: it is picked over a model with no encoder at all, and
/// for a gateway that dials none of the encoders there are.
#[test]
fn an_encode_request_with_no_key_is_decided_as_its_pair_and_takes_no_encoder() {
    let now = Instant::now();
    let theirs = Stated {
        only: Some(OTHER),
        ..PUSHING
    };
    for encoders in [&[][..], &[theirs][..]] {
        let store = encode_store();
        let objects = holder(&store);
        let p = member(&store, &objects, "p0", Role::Prefill, PUSHING);
        let d = member(&store, &objects, "d0", Role::Decode, PUSHING);
        for (n, &stated) in encoders.iter().enumerate() {
            member(&store, &objects, &format!("e{n}"), Role::Encode, stated);
        }
        let frame = encode_frame(1, 0);
        let (answer, placed) = served_encode(&store, &objects, frame, GATEWAY, now);
        assert_eq!(pairs_of(&answer), [(p, d)], "{} encoders", encoders.len());
        let none = proto::EncodePlaced {
            encoders: Vec::new(),
            one: false,
        };
        assert_eq!(placed, Some(none), "{} encoders", encoders.len());
    }
}

/// A request with a key takes its encoders by their own list's places and
/// vetoes, as each side of its pair takes its own. Another gateway alone
/// dials one encoder and the asking gateway vetoes another: every item of
/// the asking gateway's goes to the one encoder left. The other gateway
/// dials all three and vetoes none: its items go to each of them.
#[test]
fn an_encode_request_takes_its_encoders_by_their_own_places_and_vetoes() {
    let store = encode_store();
    let objects = holder(&store);
    let p = member(&store, &objects, "p0", Role::Prefill, PUSHING);
    let d = member(&store, &objects, "d0", Role::Decode, PUSHING);
    let theirs = Stated {
        only: Some(OTHER),
        ..PUSHING
    };
    let encoders = [
        member(&store, &objects, "e0", Role::Encode, PUSHING),
        member(&store, &objects, "e1", Role::Encode, theirs),
        member(&store, &objects, "e2", Role::Encode, PUSHING),
    ];
    let now = Instant::now();
    store.gateway_veto(GATEWAY, encoders[2], Some(now + Duration::from_secs(60)));
    let named = |gateway: GatewayId| -> BTreeSet<WorkerId> {
        let mut named = BTreeSet::new();
        for n in 1..=64 {
            let frame = encode_frame(n, 2);
            let (answer, placed) = served_encode(&store, &objects, frame, gateway, now);
            assert_eq!(pairs_of(&answer), [(p, d)], "frame {n}: {answer:?}");
            let placed = encoders_of(placed.as_ref());
            assert_eq!(placed.len(), 2, "frame {n}: an encoder a key");
            named.extend(placed);
        }
        named
    };
    assert_eq!(named(GATEWAY), BTreeSet::from([encoders[0]]));
    assert_eq!(named(OTHER), BTreeSet::from(encoders));
}

/// The answers of an encode request's frame come in a pair's order, and a
/// request with a key takes its encoders' list as a third list in it. A
/// list the gateway dials and can take nothing of sheds the frame under
/// that stage, the encoders' as a side's, though the gateway dials no
/// worker of another list. Else a list it dials no worker of is decided
/// over no worker, the encoders' as a side's. A model with no encoder at
/// all is the engine's to answer, under the encode side's own name. A side
/// with no worker at all is decided over no worker ahead of them all.
#[test]
fn an_encode_requests_frame_answers_its_three_lists_in_a_pairs_order() {
    let now = Instant::now();
    let (http, grpc) = (WireModes::HTTP, WireModes::GRPC);
    let both = Stated {
        wires: http.union(grpc),
        ..PUSHING
    };
    let theirs = Stated {
        only: Some(OTHER),
        ..PUSHING
    };
    // GATEWAY's answer for one key over a fleet of one prefill, `decodes`
    // and `encoders`, stating `wires` and `hint`.
    let answered = |prefill: bool, decodes: &[Stated], encoders: &[Stated], wires, hint| {
        let store = encode_store();
        let objects = holder(&store);
        if prefill {
            member(&store, &objects, "p0", Role::Prefill, both);
        }
        for (n, &stated) in decodes.iter().enumerate() {
            member(&store, &objects, &format!("d{n}"), Role::Decode, stated);
        }
        for (n, &stated) in encoders.iter().enumerate() {
            member(&store, &objects, &format!("e{n}"), Role::Encode, stated);
        }
        let frame = frame_of(ENCODED, 1, 1, wires, hint);
        let (answer, placed) = served_encode(&store, &objects, frame, GATEWAY, now);
        assert_eq!(placed.is_some(), answer.pick().is_some(), "{answer:?}");
        answer
    };
    let over_none = Some((ShedReason::NoEndpoints, "none"));
    let by_wires = Some((ShedReason::NoEndpoints, WIRES));
    let by_hint = Some((ShedReason::NoEndpoints, SUBSET_HINT));
    // Every list taken: a pick.
    let picked = answered(true, &[both], &[both], grpc, None);
    assert!(matches!(picked, Decision::Pick(_)), "{picked:?}");
    // The encoders state the HTTP wire alone and the gateway states the
    // other: the encoders' list is emptied, and the frame is shed under the
    // wires. It is too where the gateway dials no decode.
    let emptied = answered(true, &[both], &[PUSHING], grpc, None);
    assert_eq!(shed_of(&emptied), by_wires);
    let emptied = answered(true, &[theirs], &[PUSHING], grpc, None);
    assert_eq!(shed_of(&emptied), by_wires, "ahead of a side not dialed");
    // A hint of 0 empties the prefills first, whatever the encoders.
    let hinted = answered(true, &[both], &[theirs], http, Some(0));
    assert_eq!(shed_of(&hinted), by_hint);
    // Encoders another gateway alone dials: over no worker, as a side.
    let undialed = answered(true, &[both], &[theirs], http, None);
    assert_eq!(shed_of(&undialed), over_none);
    let undialed = answered(true, &[theirs], &[both], http, None);
    assert_eq!(shed_of(&undialed), over_none, "a side not dialed");
    // No encoder at all: the engine's own answer, under the encode side.
    let none = answered(true, &[both], &[], http, None);
    assert_eq!(shed_of(&none), Some((ShedReason::NoEndpoints, "encoders")));
    // A side with no worker at all: over no worker, though the gateway's
    // wires take every encoder.
    for (prefill, decodes) in [(true, &[][..]), (false, &[both][..])] {
        let lone = answered(prefill, decodes, &[PUSHING], grpc, None);
        assert_eq!(shed_of(&lone), over_none, "a prefill: {prefill}");
    }
}

/// An encode request's pick is filed as its pair's: a lease a side under
/// one id, on the pick's prefill and on its decode, and none for an
/// encoder. Each side's worker is booked, and the items' encoder is not. A
/// frame that states the later decode files its decode side chosen, as a
/// pair's frame does.
#[test]
fn an_encode_requests_pick_is_filed_its_pairs_leases_and_none_for_an_encoder() {
    let store = encode_store();
    let objects = holder(&store);
    let p = member(&store, &objects, "p0", Role::Prefill, PUSHING);
    let d = member(&store, &objects, "d0", Role::Decode, PUSHING);
    let e = member(&store, &objects, "e0", Role::Encode, PUSHING);
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let leased = |n: u8, later: bool| {
        let mut frame = encode_frame(n, 1);
        frame.decode_later = later;
        let request = Checked::new(frame).expect("a frame the record holds");
        let (mut frames, own) = (FrameBuffers::default(), Part::default());
        let loads = Arc::new(LoadSnapshot::default());
        let (decision, filed) = {
            let taken = taken_at(&request, GATEWAY, (now, now), None);
            store.deciding(|world| {
                let decided = decide_encode(world, &objects, &taken, &mut frames, &loads, &own);
                let (decision, booked) = decided.expect("the engine decides");
                let filed = issue_pair(&store, world, &taken, 1, &decision, booked);
                (decision, filed.expect("the store's own term"))
            })
        };
        let (id, placed) = frames.answered(request, &decision);
        let answer = wire::decided(id, 1, &decision, None, placed);
        assert_eq!(encoders_of(answer.encode.as_ref()), [e], "frame {n}");
        filed.expect("a pick")
    };
    // What the leases on file book on each worker, as its object counts it.
    let booked = || {
        [p, d, e].map(|id| {
            let row = WorkersStore::get(&*store, id).expect("the row");
            let object = objects.registry().get_by_url(&row.facts.key);
            object.expect("its object").load()
        })
    };
    let (prefill, decode) = leased(1, false);
    let decode = decode.expect("issued with its prefill");
    assert_eq!((prefill.side, prefill.booked), (Side::Prefill, p));
    assert_eq!((decode.side, decode.booked), (Side::Decode, d));
    assert_eq!(prefill.id, decode.id, "one id for the pair");
    assert_eq!(booked(), [1, 1, 0], "no lease books an encoder");
    assert_eq!(store.live(GATEWAY), 1, "one lease on file");

    let (prefill, decode) = leased(2, true);
    assert_eq!(decode, None, "the decode comes later");
    let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
    let chosen = sides.decode.map(|side| (side.state, side.target));
    assert_eq!(chosen, Some((LeaseState::Chosen, d)));
    assert_eq!(booked(), [2, 1, 0], "a chosen side books nothing");
}

/// A frame with items is never decided without them. Under a plan with no
/// encode side the engine declines it by its binding: a pair's plan does
/// not decide it as a plain pair, and a single profile does not decide it
/// alone. No item is placed.
#[test]
fn a_frame_with_items_under_a_plan_with_no_encode_side_is_declined_by_its_binding() {
    let now = Instant::now();
    let single = {
        let document = PolicySpec::preset("random").expect("a preset of this build");
        staged(&document, &StageRegistry::builtin())
    };
    for (plan, spec) in [("pd-pair", stock_pair_plan()), ("single", single)] {
        let store = Arc::new(MemoryStore::new());
        let objects = holder(&store);
        store.apply(spec).expect("above the unbound start");
        member(&store, &objects, "p0", Role::Prefill, PUSHING);
        member(&store, &objects, "d0", Role::Decode, PUSHING);
        member(&store, &objects, "e0", Role::Encode, PUSHING);
        for keys in [0, 1] {
            let frame = encode_frame(1, keys);
            let (answer, placed) = served_encode(&store, &objects, frame, GATEWAY, now);
            let declined = Some((ShedReason::PolicyDeclined, "binding"));
            assert_eq!(shed_of(&answer), declined, "{plan}, {keys} keys");
            assert_eq!(placed, None, "{plan}, {keys} keys");
        }
    }
}

/// A picker that reads the request's key: of the candidates kept, the one
/// at the key's fingerprint over their count. An item's facts state the
/// item's own key, so two keys' encoders are told apart.
#[derive(Debug)]
struct ByKey;

impl Lifecycle for ByKey {}

impl Picker for ByKey {
    fn name(&self) -> &'static str {
        "by-key"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        let count = u64::try_from(kept.len()).expect("a count of candidates");
        let at = usize::try_from(facts.session_key.hash % count).expect("a place among them");
        out.push(target_of(&candidates[kept[at] as usize]));
    }

    fn refuses_by_reports(&self) -> bool {
        false
    }
}

/// The answer names each key's encoder in the keys' order. Under a pick
/// that reads an item's key, two keys go to two encoders. Asked for both,
/// the answer names the first key's encoder first, and asked for them the
/// other way round it names that encoder last.
#[test]
fn an_answer_names_its_encoders_in_the_order_of_the_frames_keys() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("by-key", || Box::new(ByKey))
        .expect("a free name");
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let plan = encode_plan_in(&catalog, 1, ["by-key", "random", "random"]);
    store.apply(plan).expect("above the unbound start");
    member(&store, &objects, "p0", Role::Prefill, PUSHING);
    member(&store, &objects, "d0", Role::Decode, PUSHING);
    member(&store, &objects, "e0", Role::Encode, PUSHING);
    member(&store, &objects, "e1", Role::Encode, PUSHING);
    let now = Instant::now();
    let asked = |n, keys: &[u64]| {
        let mut frame = encode_frame(n, 0);
        let stated = frame.encode.as_mut().expect("an encode request's frame");
        stated.keys = keys
            .iter()
            .map(|&key| proto::Identity {
                form: Some(proto::identity::Form::Hash(key)),
            })
            .collect();
        let (answer, placed) = served_encode(&store, &objects, frame, GATEWAY, now);
        assert!(matches!(answer, Decision::Pick(_)), "{keys:?}: {answer:?}");
        encoders_of(placed.as_ref())
    };
    let (a, b) = (6, 7);
    let (of_a, of_b) = (asked(1, &[a])[0], asked(2, &[b])[0]);
    assert_ne!(of_a, of_b, "the two keys' encoders are told apart");
    assert_eq!(asked(3, &[a, b]), [of_a, of_b]);
    assert_eq!(asked(4, &[b, a]), [of_b, of_a]);
    assert_eq!(asked(5, &[b, a, a]), [of_b, of_a, of_a]);
}

/// A deciding thread keeps where its last encode request's items went, and
/// an answer carries no item of an earlier frame. Behind a pick with a key,
/// a frame the engine is asked nothing for places none, and a pick with no
/// key names no encoder.
#[test]
fn an_answer_carries_no_item_of_the_threads_earlier_frame() {
    let store = encode_store();
    let objects = holder(&store);
    let p = member(&store, &objects, "p0", Role::Prefill, PUSHING);
    let d = member(&store, &objects, "d0", Role::Decode, PUSHING);
    let e = member(&store, &objects, "e0", Role::Encode, PUSHING);
    let now = Instant::now();
    let mut frames = FrameBuffers::default();
    let mut asked =
        |frame, gateway| decided_on(&mut frames, &store, &objects, frame, (gateway, now));
    let (answer, placed) = asked(encode_frame(1, 2), GATEWAY);
    assert_eq!(pairs_of(&answer), [(p, d)]);
    assert_eq!(encoders_of(placed.as_ref()), [e, e]);
    // A hint of 0: shed where the gateway's places are read.
    let hinted = frame_of(ENCODED, 2, 2, WireModes::NONE, Some(0));
    let (answer, placed) = asked(hinted, GATEWAY);
    assert_eq!(
        shed_of(&answer),
        Some((ShedReason::NoEndpoints, SUBSET_HINT))
    );
    assert_eq!(placed, None);
    // No key: a pick, and no encoder of the first frame's.
    let (answer, placed) = asked(encode_frame(3, 0), GATEWAY);
    assert_eq!(pairs_of(&answer), [(p, d)]);
    assert_eq!(placed.map(|placed| placed.encoders.len()), Some(0));
}

/// An encode request any of whose three sides holds a stage that may refuse
/// by a report of leases is decided behind an apply of what waits at the
/// other files, as a pair's frame is: the encode side asks it as each side
/// of the pair does. A request whose stages refuse by none takes no turn
/// there, and the frame waits for its own file's look.
#[test]
fn an_encode_request_any_of_whose_sides_may_refuse_by_reports_applies_what_waits() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker("by-reports", || Box::new(ByReports))
        .expect("a free name");
    let cases = [
        (["by-reports", "random", "random"], LookOutcome::Quiet),
        (["random", "by-reports", "random"], LookOutcome::Quiet),
        (["random", "random", "by-reports"], LookOutcome::Quiet),
        (["random", "random", "random"], LookOutcome::Fed),
    ];
    for (picks, waited) in cases {
        let store = Arc::new(MemoryStore::new().with_decision_slots(2));
        let objects = holder(&store);
        store
            .apply(encode_plan_in(&catalog, 1, picks))
            .expect("above the unbound start");
        let p = member(&store, &objects, "p0", Role::Prefill, PUSHING);
        member(&store, &objects, "d0", Role::Decode, PUSHING);
        member(&store, &objects, "e0", Role::Encode, PUSHING);
        LeasesStore::set_term(&*store, 1);
        let mut watch = a_frame_under_the_second_files_watch(&store, p);
        // The first file's decision, its place lent.
        let _acting = DecisionSlot::new(0).enter();
        let place = store.at_work();
        let now = Instant::now();
        let request = Checked::new(encode_frame(1, 1)).expect("a frame the record holds");
        let taken = taken_at(&request, GATEWAY, (now, now), Some(&place));
        let (mut frames, own) = (FrameBuffers::default(), Part::default());
        let loads = Arc::new(LoadSnapshot::default());
        let decided = store
            .deciding(|world| decide_encode(world, &objects, &taken, &mut frames, &loads, &own));
        let (decision, _) = decided.expect("the engine decides");
        assert!(
            matches!(decision, Decision::Pick(_)),
            "{picks:?}: {decision:?}"
        );
        assert_eq!(watch.look(), waited, "{picks:?}");
    }
}

/// The encoders the asking gateway's wires take are counted in its
/// decider's part with the sides', for a request with a key. A request
/// with no key reads no encoder: it counts none, an earlier frame's
/// neither.
#[test]
fn the_encoders_a_gateways_wires_take_are_counted_for_a_request_with_a_key() {
    let store = encode_store();
    let objects = holder(&store);
    let (http, grpc) = (WireModes::HTTP, WireModes::GRPC);
    let both = Stated {
        wires: http.union(grpc),
        ..PUSHING
    };
    member(&store, &objects, "p0", Role::Prefill, both);
    member(&store, &objects, "d0", Role::Decode, both);
    member(&store, &objects, "e0", Role::Encode, PUSHING);
    let other_wire = member(&store, &objects, "e1", Role::Encode, both);
    let now = Instant::now();
    let (mut frames, own) = (FrameBuffers::default(), Part::default());
    let loads = Arc::new(LoadSnapshot::default());
    let mut counted = |n: u8, keys: u64| {
        let frame = frame_of(ENCODED, n, keys, grpc, None);
        let request = Checked::new(frame).expect("a frame the record holds");
        let decision = {
            let taken = taken_at(&request, GATEWAY, (now, now), None);
            let decided = store.deciding(|world| {
                decide_encode(world, &objects, &taken, &mut frames, &loads, &own)
            });
            decided.expect("the engine decides").0
        };
        let (id, placed) = frames.answered(request, &decision);
        let answer = wire::decided(id, 1, &decision, None, placed);
        (
            encoders_of(answer.encode.as_ref()),
            own.undialed.load(Ordering::Relaxed),
        )
    };
    // The encoder on the HTTP wire alone is dialed and shares no wire.
    assert_eq!(counted(1, 1), (vec![other_wire], 1));
    assert_eq!(counted(2, 0), (Vec::new(), 1), "no key reads no encoder");
    assert_eq!(counted(3, 1), (vec![other_wire], 2));
}

/// An encode request's frame is read for no number its request names its
/// worker by. Under an encode plan whose pair's two sides are plain hash
/// rings, so that its binding names a target source, a frame that states a
/// number is answered the pair the same frame is answered with none,
/// whatever the number: the widest one sheds nothing.
#[test]
fn an_encode_requests_frame_is_read_for_no_number() {
    let ring = "consistent_hashing";
    let document = PolicySpec::epd_presets("random", ring, ring);
    let mut document = document.expect("presets of this build");
    document.generation = 1;
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(staged(&document, &StageRegistry::builtin()))
        .expect("above the unbound start");
    for name in ["p0", "p1", "p2"] {
        member(&store, &objects, name, Role::Prefill, PUSHING);
    }
    for name in ["d0", "d1", "d2"] {
        member(&store, &objects, name, Role::Decode, PUSHING);
    }
    member(&store, &objects, "e0", Role::Encode, PUSHING);
    let now = Instant::now();
    let asked = |number: Option<u32>| {
        let mut frame = encode_frame(1, 1);
        frame.named_worker = number;
        pairs_of(&served_encode(&store, &objects, frame, GATEWAY, now).0)
    };
    let unnamed = asked(None);
    assert!(!unnamed.is_empty(), "a pick");
    for number in [0, 1, 2, u32::MAX] {
        assert_eq!(asked(Some(number)), unnamed, "{number}");
    }
}

/// This frame states no demand, so what the workers of an encode request
/// state for one changes no decision either. Its
/// prefill, its decode and its encoder each state a runtime, a pipeline for
/// media by reference, a way to decide a constraint and a window of sixteen
/// tokens. The frame is answered the pair and the encoders it is answered
/// over workers that state nothing.
#[test]
fn an_encode_request_is_decided_over_rows_that_state_demand_facts_as_over_rows_that_state_none() {
    let answers = |stated: DemandFacts| {
        let store = encode_store();
        let objects = holder(&store);
        for member in [
            ("p0", Role::Prefill),
            ("d0", Role::Decode),
            ("e0", Role::Encode),
        ] {
            member_stating(&store, &objects, ENCODED, member, PUSHING, stated);
        }
        let frame = encode_frame(1, 2);
        let (answer, placed) = served_encode(&store, &objects, frame, GATEWAY, Instant::now());
        (pairs_of(&answer), encoders_of(placed.as_ref()))
    };
    let stated = DemandFacts {
        runtime: 7,
        references: true,
        host_pipeline: true,
        window: 16,
        decoding: 3,
    };
    let (pairs, encoders) = answers(stated);
    assert_eq!(
        (pairs.len(), encoders.len()),
        (1, 2),
        "a pair, a key an item"
    );
    assert_eq!((pairs, encoders), answers(DemandFacts::default()));
}
