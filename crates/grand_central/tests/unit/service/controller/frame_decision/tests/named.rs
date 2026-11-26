//! The cases of a frame whose request names its worker by a number
//! (`DecideRequest.named_worker`): its ask [`decide`](super::decide) as a
//! deciding thread asks it, under the stock hash ring document, whose
//! binding names a target source, and
//! [`decide_alone`](super::decide_alone) for a frame that stands alone. A
//! pair's frame has its cases in a file of their own.

use std::{
    collections::BTreeSet,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use super::{
    alone_at, file, first, frame_for, holder, keyed, preset, served, served_alone, served_booked,
    shed_of, side_member, staged, GATEWAY, OTHER, PAIRED,
};
use crate::{
    document::spec::PolicySpec,
    engine::{
        decision::{Decision, ShedReason},
        framework::Filter,
        view::{Candidate, Constraints, RequestFacts, Role, WorkerId},
    },
    service::proto,
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    store::{GatewayId, HealthSink, LeasesStore, MemoryStore},
    worker_host::{Members, Objects},
};

mod pair;

/// The shed of a request whose named worker cannot take it.
const MISSED: Option<(ShedReason, &str)> = Some((ShedReason::NoEndpoints, "target-worker"));

/// `frame` with its request's worker named by `number`.
fn naming(mut frame: proto::DecideRequest, number: u32) -> proto::DecideRequest {
    frame.named_worker = Some(number);
    frame
}

/// A store under the preset `name` as generation 1, which mints leases
/// under term 1 and holds no member yet.
fn store_under(name: &str) -> Arc<MemoryStore> {
    let store = Arc::new(MemoryStore::new());
    store
        .apply(preset(name, 1))
        .expect("above the unbound start");
    LeasesStore::set_term(&*store, 1);
    store
}

/// A store under the stock hash ring document, whose one binding names a
/// target source.
fn ring_store() -> Arc<MemoryStore> {
    store_under("consistent_hashing")
}

/// `members`, each by the name it was registered under, in the order of
/// the list `objects` lends for [`PAIRED`] in `role`: the order a number
/// counts in.
fn list_order(
    objects: &Objects<Members>,
    role: Role,
    members: &[(&str, WorkerId)],
) -> Vec<WorkerId> {
    let generation = objects.registry().model_generation(PAIRED);
    let generation = generation.expect("the model's members");
    let lent = objects.lent(&generation, role);
    let named = |url: &str| {
        let member = |&&(name, _): &&(&str, WorkerId)| format!("http://{name}:8000") == url;
        let found = members.iter().find(member);
        found.expect("a member of the case").1
    };
    let order: Vec<WorkerId> = lent.workers.iter().map(|at| named(at.url())).collect();
    assert_eq!(order.len(), members.len(), "the list holds each member");
    order
}

/// The worker a decision names first, or the shed it answers.
fn reached_by(answer: &Decision) -> Result<WorkerId, Option<(ShedReason, &'static str)>> {
    first(answer).ok_or_else(|| shed_of(answer))
}

/// A frame that names its request's worker is answered that worker, ahead
/// of its session key. Under the stock hash ring document, over five
/// workers its gateway can be sent to, each number takes its frame to the
/// worker at that place of the list the worker objects lend, whichever
/// worker owns the frame's key, and with no key too. The frame's decision
/// is held to that one place: its answer names one target. The lease of a
/// keyed frame's pick counts for its key, as any keyed pick's does.
///
/// A frame that states no number is decided as it was: it goes to its
/// key's owner, with the alternates a whole decision finds.
#[test]
fn a_frame_that_names_its_worker_is_answered_the_worker_its_number_reaches() {
    let store = ring_store();
    let objects = holder(&store);
    let member = |name: &'static str| {
        (
            name,
            side_member(&store, &objects, name, Role::Single, None),
        )
    };
    let members = ["s0", "s1", "s2", "s3", "s4"].map(member);
    let order = list_order(&objects, Role::Single, &members);
    let now = Instant::now();
    let asked = |frame: proto::DecideRequest| {
        let (answer, booked) = served_booked(&store, &objects, frame, GATEWAY, (now, now), None);
        let Decision::Pick(pick) = &answer else {
            panic!("no pick: {answer:?}");
        };
        (first(&answer), pick.targets.len(), booked.session.is_none())
    };
    let mut owners = BTreeSet::new();
    for key in 1..=6_u64 {
        let unnamed = keyed(frame_for(PAIRED, 1, key), key);
        let (owner, targets, unkeyed) = asked(unnamed.clone());
        assert!(targets > 1, "key {key}: a whole decision's alternates");
        assert!(!unkeyed, "key {key}: its lease counts for it");
        owners.insert(owner);
        for (place, &worker) in (0..).zip(&order) {
            let answer = asked(naming(unnamed.clone(), place));
            assert_eq!(answer, (Some(worker), 1, false), "key {key}, place {place}");
        }
    }
    assert!(owners.len() > 1, "the keys have more than one owner");
    for (place, &worker) in (0..).zip(&order) {
        let answer = asked(naming(frame_for(PAIRED, 1, 7), place));
        assert_eq!(answer, (Some(worker), 1, true), "no key, place {place}");
    }
}

/// A frame whose number reaches no worker is shed with no endpoint under
/// `target-worker`: a number past the workers its gateway can be sent to,
/// and the widest number, which a host states for a text that is no
/// number. With a key and with none.
#[test]
fn a_number_that_reaches_no_worker_is_shed_under_target_worker() {
    let store = ring_store();
    let objects = holder(&store);
    for name in ["s0", "s1", "s2"] {
        side_member(&store, &objects, name, Role::Single, None);
    }
    let now = Instant::now();
    for number in [3, 4, 1 << 20, u32::MAX] {
        for frame in [frame_for(PAIRED, 1, 9), keyed(frame_for(PAIRED, 1, 9), 9)] {
            let frame = naming(frame, number);
            let answer = served(&store, &objects, frame, GATEWAY, (now, now), None);
            assert_eq!(shed_of(&answer), MISSED, "{number}");
        }
    }
}

/// The number counts the workers the asking gateway can be sent to, in the
/// list's order, and passes over every other. A worker another gateway
/// alone dials takes no number for this one. A standing veto of the asking
/// gateway ahead of a place moves the count: the number reaches the worker
/// behind the vetoed one, the last number is shed, another gateway counts
/// as before, and the veto's end puts the count back. A worker that is not
/// available by its object's own test, one over its thresholds here, is
/// passed over the same way.
#[test]
fn a_standing_veto_ahead_of_the_place_moves_the_count() {
    let store = ring_store();
    let objects = holder(&store);
    let member = |name: &'static str, only| {
        (
            name,
            side_member(&store, &objects, name, Role::Single, only),
        )
    };
    let members = [
        member("s0", None),
        member("s1", None),
        member("s2", Some(OTHER)),
        member("s3", None),
        member("s4", None),
    ];
    let order = list_order(&objects, Role::Single, &members);
    let theirs = members[2].1;
    let dialed: Vec<WorkerId> = order.iter().copied().filter(|&id| id != theirs).collect();
    let now = Instant::now();
    let reached = |gateway: GatewayId, number: u32, at: Instant| {
        let frame = naming(frame_for(PAIRED, 1, 3), number);
        reached_by(&served(&store, &objects, frame, gateway, (at, at), None))
    };
    for (number, &worker) in (0..).zip(&dialed) {
        assert_eq!(reached(GATEWAY, number, now), Ok(worker), "{number}");
    }
    assert_eq!(reached(GATEWAY, 4, now), Err(MISSED), "it dials four");
    for (number, &worker) in (0..).zip(&order) {
        assert_eq!(reached(OTHER, number, now), Ok(worker), "theirs, {number}");
    }

    store.gateway_veto(GATEWAY, dialed[0], Some(now + Duration::from_secs(60)));
    assert_eq!(reached(GATEWAY, 0, now), Ok(dialed[1]), "behind the veto");
    assert_eq!(reached(GATEWAY, 2, now), Ok(dialed[3]));
    assert_eq!(reached(GATEWAY, 3, now), Err(MISSED), "three are left");
    assert_eq!(reached(OTHER, 0, now), Ok(order[0]), "the veto is its own");
    let ended = now + Duration::from_secs(61);
    assert_eq!(
        reached(GATEWAY, 0, ended),
        Ok(dialed[0]),
        "a veto that ended"
    );

    let (name, vetoed) = members
        .iter()
        .find(|(_, id)| *id == dialed[1])
        .expect("a member");
    let object = objects
        .registry()
        .get_by_url(&format!("http://{name}:8000"));
    let object = object.expect("its object");
    object.set_overloaded(true);
    assert!(!object.is_available(), "over its thresholds");
    assert_eq!(
        reached(GATEWAY, 1, ended),
        Ok(dialed[2]),
        "behind {vetoed:?}"
    );
    assert_eq!(reached(GATEWAY, 3, ended), Err(MISSED), "three are left");
    object.set_overloaded(false);
    assert_eq!(reached(GATEWAY, 1, ended), Ok(dialed[1]), "available again");
}

/// Where the asking gateway can be sent to no worker of the list, there is
/// none to count: the request leaves its worker's hold and is decided as
/// one that names no worker, by the binding's rule for workers that are
/// all vetoed, whatever its number. Over four workers that are all over
/// their thresholds it goes to the one with the fewest leases on file.
///
/// It carries no key there. The same key with no number stays on its owner
/// over the veto, and its lease counts for it. With a number it goes to
/// the least loaded worker, and the lease of that pick counts for no
/// session.
#[test]
fn where_no_worker_can_be_counted_the_request_leaves_its_hold_with_no_key() {
    let store = ring_store();
    let objects = holder(&store);
    let names = ["s0", "s1", "s2", "s3"];
    let members = names.map(|name| side_member(&store, &objects, name, Role::Single, None));
    let now = Instant::now();
    let held = members.iter().zip([3_u64, 2, 0, 1]);
    let targets = held.flat_map(|(&member, held)| (0..held).map(move |_| member));
    for (target, id) in targets.zip(1..) {
        file(&store, id, target, now);
    }
    for name in names {
        let object = objects
            .registry()
            .get_by_url(&format!("http://{name}:8000"));
        object.expect("its object").set_overloaded(true);
    }
    let least = Some(members[2]);
    let asked = |frame: proto::DecideRequest| {
        let (answer, booked) = served_booked(&store, &objects, frame, GATEWAY, (now, now), None);
        (first(&answer), booked.session.is_none())
    };
    for number in [0, 3, 9, u32::MAX] {
        let answer = asked(naming(frame_for(PAIRED, 1, 5), number));
        assert_eq!(answer, (least, true), "{number}");
    }
    let mut elsewhere = 0;
    for key in 1..=6_u64 {
        let unnamed = keyed(frame_for(PAIRED, 1, key), key);
        let (owner, unkeyed) = asked(unnamed.clone());
        assert!(owner.is_some() && !unkeyed, "key {key}: its owner's lease");
        elsewhere += usize::from(owner != least);
        let answer = asked(naming(unnamed, 0));
        assert_eq!(answer, (least, true), "key {key}: no key crosses the hold");
    }
    assert!(elsewhere > 0, "a key's owner is another worker");
}

/// A filter that keeps every worker but the one its case names.
#[derive(Debug)]
struct Refuses(Arc<Mutex<Option<WorkerId>>>);

impl Lifecycle for Refuses {}

impl Filter for Refuses {
    fn name(&self) -> &'static str {
        "refuses"
    }

    fn keep(&self, _: &RequestFacts<'_>, candidate: &Candidate, _: &Constraints<'_>) -> bool {
        Some(candidate.id) != *self.0.lock()
    }
}

/// A named worker that cannot take the request after all leaves it no
/// worker. Where a filter of the frame's profile refuses the worker the
/// number reaches, the frame's one decision sheds, and the answer is the
/// request's shed under `target-worker`, not under the filter's name: the
/// request moves to no other worker. A number that reaches another worker
/// is answered that worker, and a frame that names none goes to a worker
/// the filter keeps.
#[test]
fn a_named_worker_its_decision_refuses_is_shed_under_target_worker() {
    let refused = Arc::new(Mutex::new(None));
    let mut catalog = StageRegistry::builtin();
    let held = Arc::clone(&refused);
    let stage = move || -> Box<dyn Filter> { Box::new(Refuses(Arc::clone(&held))) };
    catalog.filter("refuses", stage).expect("a free name");
    let mut document = PolicySpec::preset("consistent_hashing").expect("a preset of this build");
    document.generation = 1;
    let profile = document.profiles.get_mut("consistent_hashing");
    let profile = profile.expect("the preset's profile");
    profile.filter.push("refuses".into());
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(staged(&document, &catalog))
        .expect("above the unbound start");
    let member = |name: &'static str| {
        (
            name,
            side_member(&store, &objects, name, Role::Single, None),
        )
    };
    let members = ["s0", "s1", "s2"].map(member);
    let order = list_order(&objects, Role::Single, &members);
    let now = Instant::now();
    let reached = |named: Option<u32>| {
        let frame = frame_for(PAIRED, 1, 4);
        let frame = match named {
            Some(number) => naming(frame, number),
            None => frame,
        };
        reached_by(&served(&store, &objects, frame, GATEWAY, (now, now), None))
    };
    assert_eq!(reached(Some(1)), Ok(order[1]), "no worker refused yet");
    *refused.lock() = Some(order[1]);
    assert_eq!(reached(Some(1)), Err(MISSED), "its worker or none");
    // The shed keeps the trace of the decision that ran: it names the
    // profile, as a shed made once a profile is resolved does.
    let frame = naming(frame_for(PAIRED, 1, 4), 1);
    let answer = served(&store, &objects, frame, GATEWAY, (now, now), None);
    let Decision::Shed(shed) = answer else {
        panic!("a shed, not {answer:?}");
    };
    assert_eq!(shed.trace.made_under(), Some(0), "the decision's own trace");
    assert_eq!(reached(Some(0)), Ok(order[0]));
    assert_eq!(reached(Some(2)), Ok(order[2]));
    let unnamed = reached(None).expect("a pick among the two kept");
    assert_ne!(unnamed, order[1]);
}

/// Under a document whose binding names no target source the number is not
/// read: its host read it under another document. Sixteen frames that name
/// one place rotate over the list, as sixteen that name none do.
#[test]
fn a_number_is_not_read_under_a_binding_that_names_no_target_source() {
    let store = store_under("round_robin");
    let objects = holder(&store);
    for name in ["s0", "s1", "s2", "s3"] {
        side_member(&store, &objects, name, Role::Single, None);
    }
    let now = Instant::now();
    let named = |number: Option<u32>| -> BTreeSet<WorkerId> {
        let turns = (1..=16).map(|n| {
            let frame = frame_for(PAIRED, n, u64::from(n));
            let frame = match number {
                Some(number) => naming(frame, number),
                None => frame,
            };
            let answer = served(&store, &objects, frame, GATEWAY, (now, now), None);
            first(&answer).expect("a pick")
        });
        turns.collect()
    };
    assert_eq!(named(Some(2)).len(), 4, "the number holds no frame");
    assert_eq!(named(Some(u32::MAX)).len(), 4, "and sheds none");
    assert_eq!(named(None).len(), 4);
}

/// A frame that stands alone at a side's phase counts over the list of its
/// own phase's role: a prefill frame's number reaches the prefill at that
/// place and a decode frame's the decode, and a number past that role's
/// workers is shed, whatever workers of another role serve the model.
#[test]
fn a_frame_that_stands_alone_counts_over_its_own_phases_list() {
    let store = ring_store();
    let objects = holder(&store);
    let member = |name: &'static str, role| (name, side_member(&store, &objects, name, role, None));
    let (prefill, decode) = (Role::Prefill, Role::Decode);
    let prefills = ["p0", "p1", "p2"].map(|name| member(name, prefill));
    let decodes = ["d0", "d1"].map(|name| member(name, decode));
    member("s0", Role::Single);
    let ahead = list_order(&objects, prefill, &prefills);
    let behind = list_order(&objects, decode, &decodes);
    let now = Instant::now();
    let reached = |phase: proto::Phase, number: u32| {
        let frame = naming(alone_at(1, phase), number);
        reached_by(&served_alone(&store, &objects, frame, (GATEWAY, now), None))
    };
    for (number, &worker) in (0..).zip(&ahead) {
        assert_eq!(reached(proto::Phase::Prefill, number), Ok(worker));
    }
    assert_eq!(reached(proto::Phase::Prefill, 3), Err(MISSED));
    for (number, &worker) in (0..).zip(&behind) {
        assert_eq!(reached(proto::Phase::Decode, number), Ok(worker));
    }
    assert_eq!(reached(proto::Phase::Decode, 2), Err(MISSED));
}

/// The number counts among the workers that meet the frame's hard demands,
/// as the one call counts it in a host's own process. Of the six workers
/// the asking gateway dials, three state the runtime 2 and the others the
/// runtime 1. A frame that stays on a runtime counts the workers that
/// state it, in the list's order, and a number past them is shed under
/// `target-worker`. A frame that states no demand counts all six.
///
/// A worker that meets the demand takes no number where the gateway cannot
/// be sent to it: one another gateway alone dials, and one under the
/// asking gateway's veto. Where no worker states the frame's runtime there
/// is none to count, and the frame is shed under the demand's own stage.
///
/// The one call:
/// `the_number_counts_among_the_workers_that_meet_the_requests_demands` of
/// `engine/in_process/place/tests/target.rs`.
#[test]
fn the_number_counts_among_the_workers_that_meet_the_frames_demands() {
    use super::{demanding, stating_member};
    use crate::engine::DemandFacts;

    let store = ring_store();
    let objects = holder(&store);
    let stating = |name: &'static str, runtime: u32, only| {
        let stated = DemandFacts {
            runtime,
            ..DemandFacts::default()
        };
        let id = stating_member(&store, &objects, name, Role::Single, only, stated);
        ((name, id), runtime)
    };
    let stated = [
        stating("s0", 1, None),
        stating("s1", 2, None),
        stating("s2", 1, None),
        stating("s3", 2, None),
        stating("s4", 2, None),
        stating("s5", 1, None),
        stating("s6", 2, Some(OTHER)),
    ];
    let members = stated.map(|(member, _)| member);
    let theirs = members[6].1;
    // The workers the asking gateway dials, in the order a number counts
    // in, and those of them that state `runtime`.
    let order = list_order(&objects, Role::Single, &members);
    let dialed: Vec<WorkerId> = order.iter().copied().filter(|&id| id != theirs).collect();
    let on = |runtime: u32| -> Vec<WorkerId> {
        let states = |id: &WorkerId| {
            let told = stated.iter().find(|((_, member), _)| member == id);
            told.is_some_and(|(_, told)| *told == runtime)
        };
        dialed.iter().copied().filter(states).collect()
    };
    let now = Instant::now();
    let reached = |number: u32, runtime: u32| {
        let demands = proto::Demands {
            runtime,
            ..proto::Demands::default()
        };
        let frame = demanding(naming(frame_for(PAIRED, 1, 3), number), demands);
        reached_by(&served(&store, &objects, frame, GATEWAY, (now, now), None))
    };
    let (two, one) = (on(2), on(1));
    assert_eq!((two.len(), one.len()), (3, 3));
    for (number, &worker) in (0..).zip(&two) {
        assert_eq!(reached(number, 2), Ok(worker), "the runtime 2, {number}");
    }
    assert_eq!(reached(3, 2), Err(MISSED), "three state the runtime");
    for (number, &worker) in (0..).zip(&one) {
        assert_eq!(reached(number, 1), Ok(worker), "the runtime 1, {number}");
    }
    for (number, &worker) in (0..).zip(&dialed) {
        assert_eq!(reached(number, 0), Ok(worker), "no demand, {number}");
    }

    store.gateway_veto(GATEWAY, two[1], Some(now + Duration::from_secs(60)));
    assert_eq!(reached(1, 2), Ok(two[2]), "behind the veto");
    assert_eq!(reached(2, 2), Err(MISSED), "two can be sent to");

    let wires = Some((ShedReason::NoEndpoints, "wires"));
    assert_eq!(reached(0, 3), Err(wires), "no worker states the runtime 3");
}

/// A frame that stands alone at a side's phase counts among the workers
/// that meet what its phase's role is held to, as its decision judges
/// them. One of three workers a side states a window of 16 tokens and the
/// others one of 4,096.
///
/// A prefill holds the input alone. A prefill frame whose completion takes
/// 8,192 tokens counts every prefill, and one whose input takes 1,024
/// counts the two wide ones. A decode holds both bounds: a decode frame
/// whose completion takes 1,024 tokens counts the two wide decodes.
#[test]
fn a_frame_that_stands_alone_counts_by_what_its_phases_role_is_held_to() {
    use super::{demanding, stating_member};
    use crate::engine::DemandFacts;

    let store = ring_store();
    let objects = holder(&store);
    let stating = |name: &'static str, role, window: u32| {
        let stated = DemandFacts {
            window,
            ..DemandFacts::default()
        };
        let id = stating_member(&store, &objects, name, role, None, stated);
        (name, id)
    };
    let (prefill, decode) = (Role::Prefill, Role::Decode);
    let prefills = [
        stating("p0", prefill, 4_096),
        stating("p1", prefill, 16),
        stating("p2", prefill, 4_096),
    ];
    let decodes = [
        stating("d0", decode, 4_096),
        stating("d1", decode, 16),
        stating("d2", decode, 4_096),
    ];
    let ahead = list_order(&objects, prefill, &prefills);
    let behind = list_order(&objects, decode, &decodes);
    // A side's workers but the narrow one, in the order a number counts in.
    let wide = |order: &[WorkerId], narrow: WorkerId| -> Vec<WorkerId> {
        order.iter().copied().filter(|&id| id != narrow).collect()
    };
    let now = Instant::now();
    let reached = |phase: proto::Phase, number: u32, demands: proto::Demands| {
        let frame = demanding(naming(alone_at(1, phase), number), demands);
        reached_by(&served_alone(&store, &objects, frame, (GATEWAY, now), None))
    };
    let none = proto::Demands::default();
    let (at_prefill, at_decode) = (proto::Phase::Prefill, proto::Phase::Decode);

    let completion = proto::Demands {
        output_window: 8_192,
        ..none
    };
    for (number, &worker) in (0..).zip(&ahead) {
        let answer = reached(at_prefill, number, completion);
        assert_eq!(answer, Ok(worker), "no prefill's to hold, {number}");
    }
    let input = proto::Demands {
        input_window: 1_024,
        ..none
    };
    for (number, &worker) in (0..).zip(&wide(&ahead, prefills[1].1)) {
        let answer = reached(at_prefill, number, input);
        assert_eq!(answer, Ok(worker), "the input, {number}");
    }
    assert_eq!(reached(at_prefill, 2, input), Err(MISSED), "two hold it");

    let short = proto::Demands {
        output_window: 1_024,
        ..none
    };
    for (number, &worker) in (0..).zip(&wide(&behind, decodes[1].1)) {
        let answer = reached(at_decode, number, short);
        assert_eq!(answer, Ok(worker), "the completion, {number}");
    }
    assert_eq!(reached(at_decode, 2, short), Err(MISSED), "two hold it");
}
