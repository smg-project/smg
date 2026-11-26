//! The cases of a pair's frame whose request names its worker by a number
//! (`DecideRequest.named_worker`): its ask
//! [`decide_pair`](crate::service::controller::frame_decision::decide_pair)
//! as a deciding thread asks it, under the stock pair documents.
//!
//! Each case names the case of the one call it answers as, in a host's
//! own process: those of `engine/in_process/place/tests/pair/named.rs`
//! (`N` below) and of its `named/decode_alone.rs` (`D` below).

use std::{
    collections::BTreeSet,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_worker::load::state::LoadSnapshot;

use super::{
    super::{
        at_prefill, file, holder, keyed, pairs_of, served_pair, shed_of, side_member, staged,
        taken_at, Checked, Part, GATEWAY, OTHER,
    },
    list_order, naming, Refuses, MISSED,
};
use crate::{
    document::spec::{OverloadSpec, PolicySpec},
    engine::{
        decision::{Decision, PairSide, ShedReason},
        framework::Filter,
        view::{Candidate, Constraints, RequestFacts, Role, WireModes, WorkerId},
    },
    service::{
        controller::frame_decision::{decide_pair, FrameBuffers, PairBooked},
        proto,
    },
    stages::{lifecycle::Lifecycle, registry::StageRegistry},
    store::{GatewayId, HealthSink, LeasesStore, MemoryStore},
    worker_host::{Members, Objects},
};

/// The preset whose profile is the plain hash ring: the one a side reads
/// the number under.
const RING: &str = "consistent_hashing";

/// A store under the stock pair of the presets `prefill` and `decode` as
/// generation 1, with the stages of `catalog` and what `edit` changes of
/// the document. It mints leases under term 1 and holds no member yet. A
/// side that is the hash ring brings the binding its target source.
fn pair_store_in(
    catalog: &StageRegistry,
    (prefill, decode): (&str, &str),
    edit: impl FnOnce(&mut PolicySpec),
) -> Arc<MemoryStore> {
    let mut document = PolicySpec::pair_presets(prefill, decode).expect("presets of this build");
    document.generation = 1;
    edit(&mut document);
    let store = Arc::new(MemoryStore::new());
    store
        .apply(staged(&document, catalog))
        .expect("above the unbound start");
    LeasesStore::set_term(&*store, 1);
    store
}

/// [`pair_store_in`] with this build's own stages and the document as its
/// presets make it.
fn pair_store(prefill: &str, decode: &str) -> Arc<MemoryStore> {
    pair_store_in(&StageRegistry::builtin(), (prefill, decode), |_| {})
}

/// The stock pair of two hash rings: both sides read the number.
fn rings() -> Arc<MemoryStore> {
    pair_store(RING, RING)
}

/// The ready members `names` on `role`'s side, written through `objects`,
/// each beside its name: one `only` alone dials where it names a gateway.
fn side<const N: usize>(
    store: &MemoryStore,
    objects: &Objects<Members>,
    role: Role,
    names: [(&'static str, Option<GatewayId>); N],
) -> [(&'static str, WorkerId); N] {
    names.map(|(name, only)| (name, side_member(store, objects, name, role, only)))
}

/// A fleet every gateway dials: the prefills `p0`.. and the decodes `d0`..,
/// each side in the order of the list the worker objects lend for it, the
/// order a number counts in, with the holder of those objects.
fn fleet<const P: usize, const D: usize>(
    store: &Arc<MemoryStore>,
    prefills: [&'static str; P],
    decodes: [&'static str; D],
) -> (Arc<Objects<Members>>, Vec<WorkerId>, Vec<WorkerId>) {
    let objects = holder(store);
    let prefills = side(
        store,
        &objects,
        Role::Prefill,
        prefills.map(|name| (name, None)),
    );
    let decodes = side(
        store,
        &objects,
        Role::Decode,
        decodes.map(|name| (name, None)),
    );
    let ahead = list_order(&objects, Role::Prefill, &prefills);
    let behind = list_order(&objects, Role::Decode, &decodes);
    (objects, ahead, behind)
}

/// The pair each number reaches over two sides that pair whole: the
/// number, then the prefill and the decode at that place of each side.
fn numbered<'a>(
    ahead: &'a [WorkerId],
    behind: &'a [WorkerId],
) -> impl Iterator<Item = (u32, (WorkerId, WorkerId))> + 'a {
    (0..).zip(ahead.iter().copied().zip(behind.iter().copied()))
}

/// The `n`th pair's frame at its prefill, with its request's worker named
/// by `number` where the case names one.
fn pair_frame(n: u8, number: Option<u32>) -> proto::DecideRequest {
    let frame = at_prefill(n, WireModes::NONE, None);
    match number {
        Some(number) => naming(frame, number),
        None => frame,
    }
}

/// The prefill and the decode a decision names first, or the shed it
/// answers.
fn reached(answer: &Decision) -> Result<(WorkerId, WorkerId), Option<(ShedReason, &'static str)>> {
    let first = pairs_of(answer).first().copied();
    first.ok_or_else(|| shed_of(answer))
}

/// Says whether the worker object of the member `name` is over its
/// thresholds.
fn overload(objects: &Objects<Members>, name: &str, over: bool) {
    let object = objects
        .registry()
        .get_by_url(&format!("http://{name}:8000"));
    object.expect("its object").set_overloaded(over);
}

/// The pair's decision of `frame` as [`GATEWAY`]'s at `now`, with what the
/// leases of its pick take of the frame.
fn booked_pair(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    now: Instant,
) -> (Decision, PairBooked) {
    let request = Checked::new(frame).expect("a frame the record holds");
    let taken = taken_at(&request, GATEWAY, (now, now), None);
    let (mut frames, own) = (FrameBuffers::default(), Part::default());
    let loads = Arc::new(LoadSnapshot::default());
    let decided =
        store.deciding(|world| decide_pair(world, objects, &taken, &mut frames, &loads, &own));
    decided.expect("the engine decides")
}

/// A pair's frame that names its worker is answered the prefill and the
/// decode at that number, ahead of its session key. Under the stock pair
/// of two hash rings, over three prefills and three decodes its gateway
/// can be sent to, each number takes the frame to the worker at that place
/// of each side's list, whichever workers own the frame's key, and with no
/// key too. The frame's one decision is held to the two places: its answer
/// names one target.
///
/// A frame that states no number is decided as it was: it goes to its
/// key's owners.
///
/// The one call: `a_pair_that_names_its_worker_goes_to_that_worker_on_each_side`
/// (`N`).
#[test]
fn a_pairs_frame_that_names_its_worker_is_answered_that_worker_on_each_side() {
    let store = rings();
    let (objects, ahead, behind) = fleet(&store, ["p0", "p1", "p2"], ["d0", "d1", "d2"]);
    let now = Instant::now();
    let asked = |frame: proto::DecideRequest| {
        let answer = served_pair(&store, &objects, frame, GATEWAY, now);
        let pairs = pairs_of(&answer);
        (pairs.first().copied(), pairs.len())
    };
    let mut owners = BTreeSet::new();
    for key in 1..=6_u64 {
        let unnamed = keyed(pair_frame(1, None), key);
        let (owner, _) = asked(unnamed.clone());
        assert!(owner.is_some(), "key {key}: a pick");
        owners.insert(owner);
        for (place, pair) in numbered(&ahead, &behind) {
            let answer = asked(naming(unnamed.clone(), place));
            assert_eq!(answer, (Some(pair), 1), "key {key}, place {place}");
        }
    }
    assert!(
        owners.len() > 1,
        "the keys have more than one pair of owners"
    );
    for (place, pair) in numbered(&ahead, &behind) {
        let answer = asked(pair_frame(7, Some(place)));
        assert_eq!(answer, (Some(pair), 1), "no key, place {place}");
    }
}

/// A pair's frame whose number reaches no worker on a side is shed with no
/// endpoint under `target-worker`, and the engine is asked for no other
/// pair. Over three prefills and two decodes: a number past the prefills,
/// the widest number, which a host states for a text that is no number,
/// and a number that reaches the third prefill and no third partner of it.
/// With a key and with none.
///
/// The one call: `a_number_that_reaches_no_pair_is_shed_under_its_own_name`
/// (`N`).
#[test]
fn a_number_that_reaches_no_worker_on_a_side_sheds_the_pair_under_target_worker() {
    let store = rings();
    let (objects, ahead, behind) = fleet(&store, ["p0", "p1", "p2"], ["d0", "d1"]);
    let now = Instant::now();
    let at = |number: u32, key: Option<u64>| {
        let frame = pair_frame(9, Some(number));
        let frame = match key {
            Some(key) => keyed(frame, key),
            None => frame,
        };
        reached(&served_pair(&store, &objects, frame, GATEWAY, now))
    };
    for key in [None, Some(9)] {
        for number in [3, 4, 1 << 20, u32::MAX] {
            assert_eq!(at(number, key), Err(MISSED), "{number}: past the prefills");
        }
        assert_eq!(at(2, key), Err(MISSED), "no third partner");
        for (number, pair) in numbered(&ahead, &behind) {
            assert_eq!(at(number, key), Ok(pair), "{number}");
        }
    }
}

/// The numbers count the workers the asking gateway can be sent to, on
/// each side by that side's own reach, vetoes and worker objects. A worker
/// another gateway alone dials takes no number for this one. A standing
/// veto of the asking gateway ahead of a place moves that side's count and
/// leaves the other side's, another gateway counts as before, and the
/// veto's end puts the count back. A decode that is not available by its
/// object's own test, one over its thresholds here, is passed over the same
/// way.
///
/// The one call: `the_numbers_count_the_workers_that_can_be_sent_to` and
/// `a_place_its_host_masks_takes_no_number` (`N`).
#[test]
fn each_sides_number_counts_the_workers_its_gateway_can_be_sent_to() {
    let store = rings();
    let objects = holder(&store);
    let theirs = Some(OTHER);
    let prefills = [("p0", None), ("p1", theirs), ("p2", None), ("p3", None)];
    let prefills = side(&store, &objects, Role::Prefill, prefills);
    let decodes = [("d0", None), ("d1", None), ("d2", None), ("d3", theirs)];
    let decodes = side(&store, &objects, Role::Decode, decodes);
    let ahead = list_order(&objects, Role::Prefill, &prefills);
    let behind = list_order(&objects, Role::Decode, &decodes);
    let dialed = |order: &[WorkerId], other: WorkerId| -> Vec<WorkerId> {
        order.iter().copied().filter(|&id| id != other).collect()
    };
    let (ours_ahead, ours_behind) = (dialed(&ahead, prefills[1].1), dialed(&behind, decodes[3].1));
    let now = Instant::now();
    let at = |gateway: GatewayId, number: u32, at: Instant| {
        let frame = pair_frame(3, Some(number));
        reached(&served_pair(&store, &objects, frame, gateway, at))
    };
    for (number, pair) in numbered(&ours_ahead, &ours_behind) {
        assert_eq!(at(GATEWAY, number, now), Ok(pair), "{number}");
    }
    assert_eq!(at(GATEWAY, 3, now), Err(MISSED), "it dials three a side");
    for (number, pair) in numbered(&ahead, &behind) {
        assert_eq!(at(OTHER, number, now), Ok(pair), "theirs, {number}");
    }

    // A veto on the first decode it dials: the decodes' count moves, and
    // the prefills' does not.
    store.gateway_veto(GATEWAY, ours_behind[0], Some(now + Duration::from_secs(60)));
    assert_eq!(at(GATEWAY, 0, now), Ok((ours_ahead[0], ours_behind[1])));
    assert_eq!(at(GATEWAY, 1, now), Ok((ours_ahead[1], ours_behind[2])));
    assert_eq!(at(GATEWAY, 2, now), Err(MISSED), "two decodes are left");
    assert_eq!(
        at(OTHER, 0, now),
        Ok((ahead[0], behind[0])),
        "the veto is its own"
    );
    let ended = now + Duration::from_secs(61);
    assert_eq!(at(GATEWAY, 0, ended), Ok((ours_ahead[0], ours_behind[0])));

    // A decode over its thresholds beside two that are not.
    let (name, _) = decodes
        .iter()
        .find(|(_, id)| *id == ours_behind[1])
        .expect("a member");
    overload(&objects, name, true);
    assert_eq!(at(GATEWAY, 1, ended), Ok((ours_ahead[1], ours_behind[2])));
    assert_eq!(at(GATEWAY, 2, ended), Err(MISSED), "two decodes are left");
    overload(&objects, name, false);
    assert_eq!(at(GATEWAY, 1, ended), Ok((ours_ahead[1], ours_behind[1])));
}

/// Where no prefill is counted there is nobody to name: the pair leaves
/// its hold and is decided as one that names no worker, whatever its
/// number. Over prefills that are all over their thresholds the frame is
/// answered as the same frame is with no number, and that answer names the
/// prefill with the fewest leases on file. The leases of its pick take the
/// frame's sessions, as those of the frame that names none do.
///
/// The one call: `where_no_prefill_is_counted_the_pair_leaves_its_hold`
/// (`N`).
#[test]
fn where_no_prefill_is_counted_the_pair_leaves_its_hold() {
    let store = rings();
    let (objects, ahead, _) = fleet(&store, ["p0", "p1"], ["d0", "d1"]);
    let now = Instant::now();
    file(&store, 1, ahead[1], now);
    for name in ["p0", "p1"] {
        overload(&objects, name, true);
    }
    let asked = |frame: proto::DecideRequest| {
        let (answer, booked) = booked_pair(&store, &objects, frame, now);
        let keyless = [booked.session.is_none(), booked.decode_session.is_none()];
        (pairs_of(&answer), keyless)
    };
    for key in 1..=6_u64 {
        let unnamed = keyed(pair_frame(1, None), key);
        let (pairs, keyless) = asked(unnamed.clone());
        let least = pairs.first().map(|&(prefill, _)| prefill);
        assert_eq!(least, Some(ahead[0]), "key {key}: the fewest leases");
        assert_eq!(keyless, [false; 2], "key {key}: its leases count for it");
        for number in [0, 1, 7, u32::MAX] {
            let named = asked(naming(unnamed.clone(), number));
            assert_eq!(named, (pairs.clone(), [false; 2]), "key {key}, {number}");
        }
    }
}

/// A filter that keeps every worker and tells, of each request it is
/// asked for, whether that request states a session key.
#[derive(Debug)]
struct SeesKey(Arc<Mutex<Vec<bool>>>);

impl Lifecycle for SeesKey {}

impl Filter for SeesKey {
    fn name(&self) -> &'static str {
        "sees-key"
    }

    fn keep(&self, facts: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        self.0.lock().push(!facts.session_key.is_none());
        true
    }
}

/// A pair that leaves its hold is decided with its frame's keys, unlike a
/// single request's frame: under the plain ring a pair's key keeps no
/// prefill over a veto. Over prefills that are all over their thresholds
/// no prefill is counted, and the frame is decided as one that names no
/// worker. The stages of that decision are asked for a request that states
/// the frame's session key, whatever number the frame states, as they are
/// for the unnamed frame: a filter ahead of the prefills' profile tells it
/// of every request it is asked for. A frame with no key shows that the
/// filter tells the two apart.
///
/// The one call: `where_no_prefill_is_counted_the_pair_leaves_its_hold`
/// (`N`), which keeps its routing value there.
#[test]
fn a_pair_that_leaves_its_hold_is_decided_with_its_keys() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut catalog = StageRegistry::builtin();
    let told = Arc::clone(&seen);
    let stage = move || -> Box<dyn Filter> { Box::new(SeesKey(Arc::clone(&told))) };
    catalog.filter("sees-key", stage).expect("a free name");
    let store = pair_store_in(&catalog, (RING, RING), |document| {
        let prefills = document.profiles.get_mut("prefill");
        let prefills = prefills.expect("the pair's prefill profile");
        prefills.filter.insert(0, "sees-key".into());
    });
    let (objects, ..) = fleet(&store, ["p0", "p1"], ["d0", "d1"]);
    let now = Instant::now();
    for name in ["p0", "p1"] {
        overload(&objects, name, true);
    }
    // Whether each request the filter was asked for stated a session key,
    // over the one decision of `frame`: a pick, as over such prefills.
    let asks = |frame: proto::DecideRequest| {
        seen.lock().clear();
        let answer = served_pair(&store, &objects, frame, GATEWAY, now);
        assert!(reached(&answer).is_ok(), "{answer:?}");
        std::mem::take(&mut *seen.lock())
    };
    let keyless = asks(pair_frame(1, None));
    assert!(
        !keyless.is_empty() && !keyless.contains(&true),
        "{keyless:?}"
    );
    let unnamed = asks(keyed(pair_frame(1, None), 5));
    assert!(
        !unnamed.is_empty() && !unnamed.contains(&false),
        "{unnamed:?}"
    );
    for number in [0, 1, 7, u32::MAX] {
        let named = asks(naming(keyed(pair_frame(1, None), 5), number));
        assert_eq!(named, unnamed, "number {number}");
    }
}

/// Where no decode can be sent to and the binding steers such a pool, a
/// decode over its thresholds alone counts as a prefill's partner: the
/// prefill's number is read and the decode's is not. The frame's decision
/// is held to the prefill alone, and its decode goes by the binding's rule
/// for decodes that are all over them, to the one with the fewest leases
/// on file. A number past the prefills sheds the pair. Under a binding
/// that sheds such a pool no decode counts so: no prefill is counted, the
/// number is not read, and the pair is shed as saturated, as the same
/// frame is with no number.
///
/// The one call:
/// `a_decode_pool_all_over_its_thresholds_keeps_the_prefills_number_alone`
/// (`N`).
#[test]
fn a_decode_pool_all_over_its_thresholds_keeps_the_prefills_number_alone() {
    let store = rings();
    let (objects, ahead, behind) = fleet(&store, ["p0", "p1"], ["d0", "d1"]);
    let now = Instant::now();
    file(&store, 1, behind[0], now);
    for name in ["d0", "d1"] {
        overload(&objects, name, true);
    }
    let at = |number: u32| {
        let frame = keyed(pair_frame(4, Some(number)), 4);
        reached(&served_pair(&store, &objects, frame, GATEWAY, now))
    };
    assert_eq!(at(0), Ok((ahead[0], behind[1])), "the fewest leases");
    assert_eq!(at(1), Ok((ahead[1], behind[1])), "the fewest leases");
    assert_eq!(at(2), Err(MISSED), "the prefills are counted");

    let shedding = pair_store_in(&StageRegistry::builtin(), (RING, RING), |document| {
        document.bindings[0].overload = Some(OverloadSpec {
            shed_when_all_vetoed: true,
            ..OverloadSpec::default()
        });
    });
    let (objects, ..) = fleet(&shedding, ["p0", "p1"], ["d0", "d1"]);
    for name in ["d0", "d1"] {
        overload(&objects, name, true);
    }
    let shed = |number: Option<u32>| {
        let frame = keyed(pair_frame(4, number), 4);
        shed_of(&served_pair(&shedding, &objects, frame, GATEWAY, now))
    };
    let unnamed = shed(None);
    let saturated = Some(ShedReason::Saturated);
    assert_eq!(unnamed.map(|(reason, _)| reason), saturated);
    for number in [0, 1, 2, u32::MAX] {
        assert_eq!(shed(Some(number)), unnamed, "{number}: as with no number");
    }
}

/// A side reads the number where its profile is the plain hash ring. Under
/// a ring prefill beside a round robin decode the prefill is the one named
/// and the decode takes its turns: the frame's decision is held to the
/// prefill's place alone. A pair none of whose sides is the ring is read
/// for no number, though its binding names a target source, and neither is
/// one under a document whose binding names none: sixteen frames that name
/// one place take their turns on both sides, and the widest number sheds
/// none. Two rings under a binding that names no target source read no
/// number either: a frame goes to its key's owners, whatever number it
/// states.
///
/// The one call: `a_side_is_held_where_its_profile_is_the_plain_ring`
/// (`N`).
#[test]
fn a_side_is_held_where_its_profile_is_the_plain_ring() {
    let now = Instant::now();
    let turns = |store: &Arc<MemoryStore>, number: Option<u32>| {
        let (objects, ahead, behind) = fleet(store, ["p0", "p1", "p2"], ["d0", "d1", "d2"]);
        let named = (1..=16).map(|n| {
            let frame = keyed(pair_frame(n, number), u64::from(n));
            let answer = served_pair(store, &objects, frame, GATEWAY, now);
            reached(&answer).expect("a pick")
        });
        let (prefills, decodes): (BTreeSet<WorkerId>, BTreeSet<WorkerId>) = named.unzip();
        (prefills, decodes, ahead, behind)
    };
    let (prefills, decodes, ahead, _) = turns(&pair_store(RING, "round_robin"), Some(1));
    assert_eq!(prefills, BTreeSet::from([ahead[1]]), "the prefill is held");
    assert_eq!(decodes.len(), 3, "the decodes take their turns");

    // The ring pair's target source over two sides that take their turns.
    let sourced = |document: &mut PolicySpec| {
        let rings = PolicySpec::pair_presets(RING, RING).expect("presets of this build");
        let block = rings.bindings[0].routing_key.clone();
        let mut block = block.expect("the ring's block of sources");
        // The routing key's header and, last, the target source.
        block.sources.drain(1..block.sources.len() - 1);
        assert!(block.sources.last().is_some_and(|source| source.target));
        document.bindings[0].routing_key = Some(block);
    };
    let builtin = StageRegistry::builtin();
    let turning = ("round_robin", "round_robin");
    for number in [1, u32::MAX] {
        let neither = pair_store_in(&builtin, turning, sourced);
        let (prefills, decodes, ..) = turns(&neither, Some(number));
        assert_eq!((prefills.len(), decodes.len()), (3, 3), "{number}: no ring");
        let unsourced = pair_store(turning.0, turning.1);
        let (prefills, decodes, ..) = turns(&unsourced, Some(number));
        assert_eq!(
            (prefills.len(), decodes.len()),
            (3, 3),
            "{number}: no source"
        );
    }

    // Two rings under a block that names no target source: the frame's
    // host read the number under another document.
    let rings = pair_store_in(&builtin, (RING, RING), |document| {
        let block = document.bindings[0].routing_key.as_mut();
        let block = block.expect("the ring's block of sources");
        block.sources.retain(|source| !source.target);
    });
    let (objects, ..) = fleet(&rings, ["p0", "p1", "p2"], ["d0", "d1", "d2"]);
    for key in 1..=6_u64 {
        let asked = |number: Option<u32>| {
            let frame = keyed(pair_frame(1, number), key);
            reached(&served_pair(&rings, &objects, frame, GATEWAY, now))
        };
        let owners = asked(None);
        assert!(owners.is_ok(), "key {key}: a pick");
        for number in [0, 1, 2, u32::MAX] {
            assert_eq!(asked(Some(number)), owners, "key {key}, {number}");
        }
    }
}

/// Where the decode's side alone is the plain hash ring, the prefill is
/// its own profile's pick, and the decision counts the number itself
/// behind that pick: the prefills take their turns, and the decode is the
/// one the number reaches among the decodes the gateway can be sent to,
/// whatever the frame's key. A decode another gateway alone dials and one
/// over its thresholds take no number. A number past the decodes counted,
/// and the widest one, shed the pair under `target-worker`.
///
/// The one call:
/// `a_decode_named_alone_is_counted_behind_the_prefill_its_profile_picks`,
/// `a_number_past_the_picks_partners_sheds_the_pair_behind_that_pick` and
/// `the_decodes_number_counts_the_decodes_that_can_be_sent_to` (`D`).
#[test]
fn a_decode_named_alone_is_counted_behind_the_prefill_the_decision_picks() {
    let store = pair_store("round_robin", RING);
    let objects = holder(&store);
    let prefills = [("p0", None), ("p1", None), ("p2", None)];
    let prefills = side(&store, &objects, Role::Prefill, prefills);
    let decodes = [
        ("d0", None),
        ("d1", Some(OTHER)),
        ("d2", None),
        ("d3", None),
    ];
    let decodes = side(&store, &objects, Role::Decode, decodes);
    let behind = list_order(&objects, Role::Decode, &decodes);
    let ours: Vec<WorkerId> = behind
        .iter()
        .copied()
        .filter(|&id| id != decodes[1].1)
        .collect();
    let now = Instant::now();
    let turns = |number: u32| {
        let named = (1..=16).map(|n| {
            let frame = keyed(pair_frame(n, Some(number)), u64::from(n));
            reached(&served_pair(&store, &objects, frame, GATEWAY, now))
        });
        let named: Result<Vec<(WorkerId, WorkerId)>, _> = named.collect();
        named.map(|pairs| pairs.into_iter().unzip::<_, _, BTreeSet<_>, BTreeSet<_>>())
    };
    for (number, &decode) in (0..).zip(&ours) {
        let (ahead, named) = turns(number).expect("sixteen picks");
        assert_eq!(named, BTreeSet::from([decode]), "{number}: the one decode");
        assert_eq!(ahead.len(), prefills.len(), "{number}: the prefills' turns");
    }
    for number in [3, 1 << 20, u32::MAX] {
        assert_eq!(turns(number), Err(MISSED), "{number}: past the decodes");
    }

    let (name, _) = decodes
        .iter()
        .find(|(_, id)| *id == ours[0])
        .expect("a member");
    overload(&objects, name, true);
    let (_, named) = turns(0).expect("sixteen picks");
    assert_eq!(named, BTreeSet::from([ours[1]]), "behind the one over them");
    assert_eq!(turns(2), Err(MISSED), "two decodes are left");
}

/// Where no decode can be sent to, the decode's number is not read: the
/// frame is decided as one that names no worker. Over decodes that are all
/// over their thresholds the binding's rule takes the one with the fewest
/// leases on file, whatever the number. So is it where no prefill opens
/// the pair. Here no prefill is available: the pair's default for such a
/// pool pairs by load, and the frame is answered the prefill with the
/// fewest leases beside the decode the same frame is answered with no
/// number.
///
/// The one call:
/// `where_no_decode_can_be_sent_to_the_decodes_number_is_not_read` and
/// `where_no_prefill_opens_the_pair_the_decodes_number_is_not_read` (`D`).
#[test]
fn where_no_prefill_opens_the_pair_the_decodes_number_is_not_read() {
    let store = pair_store("round_robin", RING);
    let (objects, ahead, behind) = fleet(&store, ["p0", "p1"], ["d0", "d1"]);
    let now = Instant::now();
    file(&store, 1, behind[0], now);
    file(&store, 2, ahead[1], now);
    let asked = |number: Option<u32>| {
        let frame = keyed(pair_frame(5, number), 5);
        reached(&served_pair(&store, &objects, frame, GATEWAY, now)).expect("a pick")
    };
    let numbers = [0, 1, 9, u32::MAX].map(Some);

    // No decode can be sent to: the decode of the least load, and the
    // prefills take their turns.
    for name in ["d0", "d1"] {
        overload(&objects, name, true);
    }
    for number in [None].into_iter().chain(numbers) {
        assert_eq!(asked(number).1, behind[1], "{number:?}: the fewest leases");
    }
    for name in ["d0", "d1"] {
        overload(&objects, name, false);
    }

    // No prefill opens the pair: the prefill of the least load.
    for name in ["p0", "p1"] {
        overload(&objects, name, true);
    }
    let unnamed = asked(None);
    assert_eq!(unnamed.0, ahead[0], "the fewest leases");
    for number in numbers {
        assert_eq!(asked(number), unnamed, "{number:?}");
    }
}

/// A prefill over its thresholds alone opens the pair where its profile
/// places among such prefills: one that weighs load against affinity by a
/// gate of its own, under a binding that steers. The decode's number is
/// read then, behind the prefill that profile picks, and a number past the
/// decodes sheds the pair.
///
/// The one call:
/// `a_prefill_over_its_thresholds_opens_the_pair_under_a_profile_that_weighs`
/// (`D`).
#[test]
fn a_prefill_over_its_thresholds_opens_the_pair_under_a_profile_that_weighs() {
    let store = pair_store("cache_aware", RING);
    let (objects, _, behind) = fleet(&store, ["p0", "p1"], ["d0", "d1", "d2"]);
    for name in ["p0", "p1"] {
        overload(&objects, name, true);
    }
    let now = Instant::now();
    let named = |n: u8, number: u32| {
        let frame = keyed(pair_frame(n, Some(number)), u64::from(n));
        let answer = served_pair(&store, &objects, frame, GATEWAY, now);
        reached(&answer).map(|(_, decode)| decode)
    };
    for n in 1..=8 {
        assert_eq!(named(n, 0), Ok(behind[0]), "names 0, frame {n}");
        assert_eq!(named(n, 2), Ok(behind[2]), "names 2, frame {n}");
        assert_eq!(named(n, 3), Err(MISSED), "names 3, frame {n}");
    }
}

/// A named worker that cannot take the request after all leaves the pair
/// no worker. Where a filter of a side's profile refuses the worker the
/// number reaches, the frame's one decision sheds, and the answer is the
/// pair's shed under `target-worker`, not under the filter's name: the
/// pair moves to no other worker. The shed keeps the trace of the decision
/// that ran: it names the profile that decision was made under, where the
/// shed of a number that reaches no worker names none, since the engine
/// was asked nothing. A number that reaches another pair is answered that
/// pair.
///
/// The shed names the side the worker was missed on, as the one call's
/// does: the prefills' for a number past the prefills, which are counted
/// first, and the side of the worker the filter refused. The side stays in
/// this process: the answer's message has no field for it.
///
/// The one call: `a_number_that_reaches_no_pair_is_shed_under_its_own_name`
/// (`N`), where the route's rule refuses the pair as the pick, and
/// `a_pair_shed_under_its_named_worker_names_the_side_it_was_missed_on`
/// (`N`) for the side.
#[test]
fn a_named_pair_its_decision_refuses_is_shed_under_target_worker() {
    let refused = Arc::new(Mutex::new(None));
    let mut catalog = StageRegistry::builtin();
    let held = Arc::clone(&refused);
    let stage = move || -> Box<dyn Filter> { Box::new(Refuses(Arc::clone(&held))) };
    catalog.filter("refuses", stage).expect("a free name");
    let store = pair_store_in(&catalog, (RING, RING), |document| {
        for profile in document.profiles.values_mut() {
            profile.filter.push("refuses".into());
        }
    });
    let (objects, ahead, behind) = fleet(&store, ["p0", "p1", "p2"], ["d0", "d1", "d2"]);
    let now = Instant::now();
    let asked = |number: u32| {
        let frame = pair_frame(4, Some(number));
        served_pair(&store, &objects, frame, GATEWAY, now)
    };
    let made_under = |answer: &Decision| match answer {
        Decision::Shed(shed) => shed.trace.made_under(),
        other => panic!("a shed, not {other:?}"),
    };
    assert_eq!(
        reached(&asked(1)),
        Ok((ahead[1], behind[1])),
        "nobody refused"
    );
    let side_of = |answer: &Decision| match answer {
        Decision::Shed(shed) => shed.side,
        other => panic!("a shed, not {other:?}"),
    };
    let past = asked(3);
    assert_eq!(shed_of(&past), MISSED, "past the workers");
    assert_eq!(made_under(&past), None, "the engine was asked nothing");
    assert_eq!(side_of(&past), Some(PairSide::Prefill), "past the prefills");
    for (side, left) in [(ahead[1], PairSide::Prefill), (behind[1], PairSide::Decode)] {
        *refused.lock() = Some(side);
        let answer = asked(1);
        assert_eq!(shed_of(&answer), MISSED, "its pair or none");
        assert!(made_under(&answer).is_some(), "the decision's own trace");
        assert_eq!(side_of(&answer), Some(left), "the side its filter emptied");
        assert_eq!(reached(&asked(0)), Ok((ahead[0], behind[0])));
        assert_eq!(reached(&asked(2)), Ok((ahead[2], behind[2])));
    }
}

/// The ready members of `role`'s side, each beside its name. `names` says
/// what each states for a request's hard demands: its name, its runtime
/// and its way to decide a constraint. Its last part is the one gateway
/// that dials it, where one alone does.
fn stating<const N: usize>(
    store: &MemoryStore,
    objects: &Objects<Members>,
    role: Role,
    names: [(&'static str, u32, u8, Option<GatewayId>); N],
) -> [(&'static str, WorkerId); N] {
    use super::super::stating_member;
    use crate::engine::DemandFacts;

    names.map(|(name, runtime, decoding, only)| {
        let stated = DemandFacts {
            runtime,
            decoding,
            ..DemandFacts::default()
        };
        let id = stating_member(store, objects, name, role, only, stated);
        (name, id)
    })
}

/// `order` but the worker `out`: a side's workers that a demand keeps, in
/// the order a number counts in.
fn but(order: &[WorkerId], out: WorkerId) -> Vec<WorkerId> {
    order.iter().copied().filter(|&id| id != out).collect()
}

/// The numbers count among the workers that meet the frame's hard demands,
/// each side by its role's part, as the one call counts them in a host's
/// own process. The asking gateway dials three workers a side: one states
/// the runtime 1 and the others the runtime 2. One decode alone states a
/// way to decide a constraint. Another gateway alone dials a fourth worker
/// a side, on the runtime 2: it takes no number.
///
/// A frame that stays on a runtime counts the workers that state it on
/// each side, and a number past them sheds the pair under `target-worker`.
/// A prefill is not held to the way, so that demand leaves the prefills'
/// count as it is: the one decode that states it is each prefill's one
/// partner. Where a demand leaves a side nobody there is none to count,
/// and the pair is shed under the demand's own stage, as any pair is.
///
/// The one call:
/// `the_numbers_count_among_the_workers_that_meet_the_requests_demands`
/// (`N`).
#[test]
fn the_numbers_count_among_the_workers_that_meet_the_frames_demands() {
    use super::super::demanding;

    let store = rings();
    let objects = holder(&store);
    let theirs = Some(OTHER);
    let prefills = [
        ("p0", 1, 0, None),
        ("p1", 2, 0, None),
        ("p2", 2, 0, None),
        ("p3", 2, 0, theirs),
    ];
    let prefills = stating(&store, &objects, Role::Prefill, prefills);
    let decodes = [
        ("d0", 1, 0, None),
        ("d1", 2, 0, None),
        ("d2", 2, 3, None),
        ("d3", 2, 0, theirs),
    ];
    let decodes = stating(&store, &objects, Role::Decode, decodes);
    // Each side's workers the asking gateway dials, in the order a number
    // counts in.
    let ahead = list_order(&objects, Role::Prefill, &prefills);
    let ahead = but(&ahead, prefills[3].1);
    let behind = list_order(&objects, Role::Decode, &decodes);
    let behind = but(&behind, decodes[3].1);
    let now = Instant::now();
    let asked = |number: u32, demands: proto::Demands| {
        let frame = demanding(pair_frame(1, Some(number)), demands);
        reached(&served_pair(&store, &objects, frame, GATEWAY, now))
    };
    let none = proto::Demands::default();
    for (number, pair) in numbered(&ahead, &behind) {
        assert_eq!(asked(number, none), Ok(pair), "no demand, {number}");
    }

    // A pinned pair's wire: two workers a side state its runtime.
    let wire = proto::Demands { runtime: 2, ..none };
    let (on_ahead, on_behind) = (but(&ahead, prefills[0].1), but(&behind, decodes[0].1));
    for (number, pair) in numbered(&on_ahead, &on_behind) {
        assert_eq!(asked(number, wire), Ok(pair), "the runtime 2, {number}");
    }
    assert_eq!(asked(2, wire), Err(MISSED), "two a side");
    let other = proto::Demands { runtime: 1, ..none };
    let lone = (prefills[0].1, decodes[0].1);
    assert_eq!(asked(0, other), Ok(lone), "the other runtime's one");
    assert_eq!(asked(1, other), Err(MISSED));

    // The prefills are not held to the way: all three are counted, and the
    // one decode that states it is the reached prefill's one partner.
    let decided = proto::Demands {
        decoding: 3,
        ..none
    };
    let one = (ahead[0], decodes[2].1);
    assert_eq!(asked(0, decided), Ok(one), "any prefill, the one decode");
    assert_eq!(asked(1, decided), Err(MISSED), "one decode");

    // No decode states the way 4: there is no partner to count a prefill
    // by, and the pair is shed as one that names no worker is.
    let unmet = proto::Demands {
        decoding: 4,
        ..none
    };
    let shed = Some((ShedReason::NoEndpoints, "decoding"));
    assert_eq!(asked(0, unmet), Err(shed));
}

/// Where the decode's side alone reads the number, the decodes' count
/// passes over a decode that fails what a decode is held to of the frame's
/// hard demands. The count of a pair named on both sides does the same.
///
/// The first decode states the runtime 1 and every other worker the
/// runtime 2. A frame that stays on the runtime 2 counts two decodes,
/// behind whichever prefill the decision picks. A number past them sheds
/// the pair under `target-worker`, and a frame that states no demand
/// counts all three.
///
/// The one call:
/// `a_decode_named_alone_takes_no_number_behind_a_mask_or_a_demand` (`D`).
#[test]
fn a_decode_named_alone_takes_no_number_behind_a_demand() {
    use super::super::demanding;

    let store = pair_store("round_robin", RING);
    let objects = holder(&store);
    let prefills = [("p0", 2, 0, None), ("p1", 2, 0, None)];
    let prefills = stating(&store, &objects, Role::Prefill, prefills);
    let decodes = [("d0", 1, 0, None), ("d1", 2, 0, None), ("d2", 2, 0, None)];
    let decodes = stating(&store, &objects, Role::Decode, decodes);
    let behind = list_order(&objects, Role::Decode, &decodes);
    let now = Instant::now();
    // The prefills and the decodes that sixteen frames are answered.
    let turns = |number: u32, demands: proto::Demands| {
        let named = (1..=16).map(|n| {
            let frame = keyed(pair_frame(n, Some(number)), u64::from(n));
            let frame = demanding(frame, demands);
            reached(&served_pair(&store, &objects, frame, GATEWAY, now))
        });
        let named: Result<Vec<(WorkerId, WorkerId)>, _> = named.collect();
        named.map(|pairs| pairs.into_iter().unzip::<_, _, BTreeSet<_>, BTreeSet<_>>())
    };
    let none = proto::Demands::default();
    for (number, &decode) in (0..).zip(&behind) {
        let (ahead, named) = turns(number, none).expect("sixteen picks");
        assert_eq!(named, BTreeSet::from([decode]), "no demand, {number}");
        assert_eq!(ahead.len(), prefills.len(), "the prefills' turns");
    }
    let wire = proto::Demands { runtime: 2, ..none };
    for (number, &decode) in (0..).zip(&but(&behind, decodes[0].1)) {
        let (ahead, named) = turns(number, wire).expect("sixteen picks");
        assert_eq!(named, BTreeSet::from([decode]), "the runtime 2, {number}");
        assert_eq!(ahead.len(), prefills.len(), "the prefills' turns");
    }
    assert_eq!(turns(2, wire), Err(MISSED), "two decodes meet the demand");
}
