//! The cases of an encode request's frame under a sessions rule on its
//! binding. The rule runs on the pair's two sides as under a pair's plan,
//! through the entry that issues no lease, and places no item. Each case
//! names the case it answers as: a pair's case of this folder's first file,
//! or the engine's own.

use std::{sync::Arc, time::Instant};

use parking_lot::Mutex;
use smg_worker::load::state::LoadSnapshot;

use super::{
    super::{file, file_of, keyed, keys_on, leased_call, pins_of, sent, Lists},
    encode_frame, encoders_of, holder, member, pairs_of, shed_of, staged, taken_at, Checked,
    FrameBuffers, Part, GATEWAY,
};
use crate::{
    document::{
        applied::AppliedSpec,
        spec::{Assignment, PolicySpec, SessionsSpec},
    },
    engine::{
        decision::Decision,
        framework::Filter,
        sticky_sessions::SessionVerdict,
        view::{Candidate, Constraints, Phase, RequestFacts, Role, WorkerId},
    },
    service::{
        controller::{
            decision_slots::tests::encode::PUSHING,
            frame_decision::{decide_encode, issue_pair},
            lifecycle::{self, tests::open},
            wire,
        },
        proto,
    },
    stages::registry::StageRegistry,
    store::{LeasesStore, MemoryStore},
    tracking::lease::{Lease, LeaseId, ReleaseReason},
    worker_host::{Members, Objects},
};

/// The encode plan's document under a sessions rule that states no cap,
/// with the stages of `catalog`. Each of an encode request's three sides
/// draws among the workers `filters` keep. The rule stands on the binding
/// as `set` leaves it, and runs on the pair's two sides.
fn ruled_encode_plan_in(
    catalog: &StageRegistry,
    filters: &str,
    set: impl FnOnce(&mut SessionsSpec),
) -> AppliedSpec {
    let document = format!(
        r#"{{
          "apiVersion": "smg.policy/v1",
          "generation": 1,
          "view": {{ "headers": ["x-session-id"] }},
          "profiles": {{
            "encoders": {{ "filter": {filters}, "pick": "random" }},
            "prefills": {{ "filter": {filters}, "pick": "random" }},
            "decodes": {{ "filter": {filters}, "pick": "random" }}
          }},
          "bindings": [
            {{ "plan": "epd",
              "profiles": {{ "encode": "encoders", "prefill": "prefills", "decode": "decodes" }},
              "sessions": {{ "sticky_header": "x-session-id", "inflight_cap": 65535 }} }}
          ]
        }}"#
    );
    let mut document: PolicySpec =
        serde_json::from_str(&document).expect("a document of this build");
    let rule = document.bindings[0].sessions.as_mut();
    set(rule.expect("the binding's rule"));
    staged(&document, catalog)
}

/// [`ruled_encode_plan_in`] as the stock controller's own catalog takes it,
/// each side among the workers its health filter keeps. The rule places a
/// new key by `assignment` under the in-flight cap `cap`, 65535 for none.
fn ruled_encode_plan(assignment: Assignment, cap: u16) -> AppliedSpec {
    let catalog = StageRegistry::builtin().without_lifecycle();
    ruled_encode_plan_in(&catalog, r#"["healthy"]"#, |rule| {
        (rule.assignment, rule.inflight_cap) = (assignment, cap);
    })
}

/// What a frame's pick is filed: the prefill side's lease, and the decode
/// side's where it was issued with it.
type Filed = Option<(Lease, Option<Lease>)>;

/// One encode request's frame of [`GATEWAY`]'s decided and its pick leased
/// under term 1, as a deciding thread does both under one load of the
/// world. It answers the decision, the encoders its answer names, and what
/// was filed. The load monitor published nothing.
fn leased_encode(
    store: &MemoryStore,
    objects: &Objects<Members>,
    frame: proto::DecideRequest,
    now: Instant,
) -> (Decision, Vec<WorkerId>, Filed) {
    let request = Checked::new(frame).expect("a frame the record holds");
    let (mut frames, own) = (FrameBuffers::default(), Part::default());
    let loads = Arc::new(LoadSnapshot::default());
    let (decision, filed) = {
        let taken = taken_at(&request, GATEWAY, (now, now), None);
        store.deciding(|world| {
            let decided = decide_encode(world, objects, &taken, &mut frames, &loads, &own);
            let (decision, booked) = decided.expect("the engine decides");
            let filed = issue_pair(store, world, &taken, 1, &decision, booked);
            (decision, filed.expect("the store's own term"))
        })
    };
    let (id, placed) = frames.answered(request, &decision);
    let answer = wire::decided(id, 1, &decision, None, placed);
    (decision, encoders_of(answer.encode.as_ref()), filed)
}

/// Tells the store that `gateway`'s frames sent both sides of the pair
/// `prefill` leads to its pick's first target.
fn sent_both(store: &MemoryStore, prefill: &Lease, now: Instant) {
    let hand = open(store, GATEWAY);
    for side in [proto::Side::Prefill, proto::Side::Decode] {
        let own = sent(prefill, side, 0);
        lifecycle::dispatched(store, &hand, &own, now).expect("the frame reads");
    }
}

/// A key's encode requests follow the pins its first one claimed, on both
/// sides of the pair, under the stock controller's own catalog. The first
/// request claims a prefill and a decode, and its gateway's two Dispatched
/// frames pin each side's key there. Every later request of the key is
/// answered that pair, with an item and with none, and each side's lease
/// states the pin.
///
/// The pair's case holds the same following in its last step:
/// `a_served_pairs_decode_dispatch_elsewhere_withdraws_the_decode_sides_own_claim`.
#[test]
fn a_served_encode_requests_key_follows_its_pins_with_an_item_and_with_none() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(ruled_encode_plan(Assignment::Delegate, u16::MAX))
        .expect("above the unbound start");
    let sides = [
        (Role::Prefill, ["p0", "p1", "p2"]),
        (Role::Decode, ["d0", "d1", "d2"]),
    ];
    for (role, names) in sides {
        for name in names {
            member(&store, &objects, name, role, PUSHING);
        }
    }
    let e = member(&store, &objects, "e0", Role::Encode, PUSHING);
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let leased = |n: u8, items: u64| {
        let frame = keyed(encode_frame(n, items), 7);
        let (decision, encoders, filed) = leased_encode(&store, &objects, frame, now);
        let (prefill, decode) = filed.expect("a pick");
        let decode = decode.expect("issued with its prefill");
        let verdicts = (prefill.session_verdict, decode.session_verdict);
        (pairs_of(&decision)[0], encoders, verdicts, prefill)
    };
    let (first, encoders, claims, prefill) = leased(1, 1);
    let claimed = (
        SessionVerdict::Claimed(first.0),
        SessionVerdict::Claimed(first.1),
    );
    assert_eq!((encoders, claims), (vec![e], claimed));
    sent_both(&store, &prefill, now);
    let pinned = (SessionVerdict::Pinned, SessionVerdict::Pinned);
    for n in 2..=17 {
        let items = u64::from(n % 2);
        let (pair, encoders, follows, _) = leased(n, items);
        assert_eq!((pair, follows), (first, pinned), "frame {n}");
        let named = if items == 0 { vec![] } else { vec![e] };
        assert_eq!(encoders, named, "frame {n}: an encoder an item");
    }
}

/// The count of its key's requests each row carried into a stage, with the
/// phase the stage decided for.
type Counts = Arc<Mutex<Vec<(Phase, u16)>>>;

/// A filter of a host's own build that keeps every row and notes the count
/// of the request's key it carries.
#[derive(Debug)]
struct ReadsCounts(Counts);

impl Filter for ReadsCounts {
    fn name(&self) -> &'static str {
        "reads-counts"
    }

    fn keep(&self, facts: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
        self.0.lock().push((facts.phase, row.session_in_flight));
        true
    }
}

/// A served encode request under a rule with an in-flight cap hands each
/// side's key on at the cap, as a served pair's rule does. The leases on
/// file count each side's key under the side's own session, and each side's
/// rows carry that count through the entry that issues no lease. The key's
/// first two requests stand on one prefill and one decode.
///
/// The third meets the cap on both sides. It is placed again, on the
/// prefill and the decode that carry none of the key's requests, and each
/// side's lease states the respill. The key's fourth request follows the
/// move on both sides.
///
/// An encoder's rows carry no count. Leases filed by hand book the encoder
/// under each side's session, up to the cap, and its rows read none of them:
/// an item goes by its side's policy.
///
/// The pair's case: `a_served_pair_hands_each_sides_key_on_at_its_rules_cap`.
#[test]
fn a_served_encode_request_hands_each_sides_key_on_at_its_rules_cap() {
    let counts = Counts::default();
    let kept = Arc::clone(&counts);
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("reads-counts", move || {
            Box::new(ReadsCounts(Arc::clone(&kept)))
        })
        .expect("a free filter name");
    let catalog = catalog.without_lifecycle();
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    // The rule places a key on the worker with the fewest requests: a
    // request placed again at the cap leaves the key's worker.
    let plan = ruled_encode_plan_in(&catalog, r#"["healthy", "reads-counts"]"#, |rule| {
        (rule.assignment, rule.inflight_cap) = (Assignment::MinLoad, 2);
    });
    store.apply(plan).expect("above the unbound start");
    for (role, names) in [(Role::Prefill, ["p0", "p1"]), (Role::Decode, ["d0", "d1"])] {
        for name in names {
            member(&store, &objects, name, role, PUSHING);
        }
    }
    let e = member(&store, &objects, "e0", Role::Encode, PUSHING);
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let leased = |n: u8| {
        counts.lock().clear();
        let frame = keyed(encode_frame(n, 1), 7);
        let (decision, encoders, filed) = leased_encode(&store, &objects, frame, now);
        assert_eq!(encoders, [e], "frame {n}");
        let (prefill, decode) = filed.expect("a pick");
        let decode = decode.expect("issued with its prefill");
        let verdicts = (prefill.session_verdict, decode.session_verdict);
        (pairs_of(&decision)[0], verdicts, prefill, decode)
    };
    // The largest count the last frame's stages read on their rows: the
    // prefills', the decodes', the encoders'.
    let most = || {
        [Phase::Prefill, Phase::Decode, Phase::Encode].map(|phase| {
            let seen = counts.lock();
            let of_phase = seen.iter().filter(|row| row.0 == phase).map(|row| row.1);
            of_phase.max()
        })
    };
    // The key's first request claims a worker a side, and its gateway's two
    // Dispatched frames pin each side's key there.
    let (first, claims, prefill, decode) = leased(1);
    let claimed = (
        SessionVerdict::Claimed(first.0),
        SessionVerdict::Claimed(first.1),
    );
    assert_eq!((claims, most()), (claimed, [Some(0); 3]));
    sent_both(&store, &prefill, now);
    // The cap's two leases on the encoder under each side's session.
    let sessions = [prefill.session, decode.session];
    for (id, session) in (1..).zip(sessions.into_iter().chain(sessions)) {
        file_of(&store, id, e, session, now);
    }
    // One request of the key in flight a side: the second stays.
    let pinned = (SessionVerdict::Pinned, SessionVerdict::Pinned);
    let (second, stays, ..) = leased(2);
    assert_eq!((second, stays), (first, pinned));
    assert_eq!(most(), [Some(1), Some(1), Some(0)], "one a side");
    // Two in flight on each side's worker: the third request is placed
    // again on both sides, and no Dispatched frame has come for it.
    let (third, respills, prefill, decode) = leased(3);
    assert!(
        third.0 != first.0 && third.1 != first.1,
        "{first:?} {third:?}"
    );
    assert_eq!(respills, (SessionVerdict::Respill, SessionVerdict::Respill));
    assert_eq!(most(), [Some(2), Some(2), Some(0)], "the cap a side");
    let pins = [prefill.session, decode.session].map(|session| pins_of(&store, session));
    let moved = [Some([third.0, first.0]), Some([third.1, first.1])];
    assert_eq!(
        pins, moved,
        "each side's pin, moved where the pair is decided"
    );
    // The fourth follows the move: each side's new worker carries one.
    let (fourth, follows, ..) = leased(4);
    assert_eq!((fourth, follows), (third, pinned));
}

/// A served encode request's keyed pick counts for its key on each side's
/// worker, under a sessions rule and the stock controller's own catalog.
/// The prefill's worker and the decode's each carry one key more while the
/// pair's lease stands. The encoder carries none, and a request with no key
/// counts for none.
///
/// A rule that places a new key on the worker with the fewest keys reads
/// those counts on each side. The next key's request takes the prefill and
/// the decode that carry none, and the key after a release takes the two
/// the release freed.
///
/// The pair's case:
/// `a_served_pairs_keyed_lease_counts_its_key_on_each_sides_worker`.
#[test]
fn a_served_encode_requests_keyed_lease_counts_its_key_on_each_sides_worker() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(ruled_encode_plan(Assignment::MinGroup, u16::MAX))
        .expect("above the unbound start");
    let side = |role: Role, names: [&str; 2]| {
        names.map(|name| member(&store, &objects, name, role, PUSHING))
    };
    let [p0, p1] = side(Role::Prefill, ["p0", "p1"]);
    let [d0, d1] = side(Role::Decode, ["d0", "d1"]);
    member(&store, &objects, "e0", Role::Encode, PUSHING);
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    // What the workers' objects count, and what the pairs that stand make:
    // the two prefills, the two decodes, then the encoder.
    let keys = || keys_on(&objects, ["p0", "p1", "d0", "d1", "e0"]);
    let made = |pairs: &[(WorkerId, WorkerId)]| {
        let carried = |id| {
            let carrying = pairs.iter().filter(|pair| pair.0 == id || pair.1 == id);
            carrying.count()
        };
        let [a, b, c, d] = [p0, p1, d0, d1].map(carried);
        [a, b, c, d, 0]
    };
    let leased = |n: u8, key: Option<u64>| {
        let frame = encode_frame(n, 1);
        let frame = match key {
            Some(key) => keyed(frame, key),
            None => frame,
        };
        let (decision, _, filed) = leased_encode(&store, &objects, frame, now);
        let (prefill, _) = filed.expect("a pick");
        (pairs_of(&decision)[0], prefill.id)
    };
    let released = |lease: LeaseId| {
        let gone = ReleaseReason::ClientGone;
        assert!(LeasesStore::released(&*store, GATEWAY, lease, None, gone));
    };
    let (_, keyless) = leased(1, None);
    assert_eq!(keys(), [0; 5], "a request with no key counts for none");
    released(keyless);
    let (first, held) = leased(2, Some(7));
    assert_eq!(keys(), made(&[first]), "a key on each side's worker");
    // The next key's request takes the prefill and the decode with no key.
    let (second, _) = leased(3, Some(8));
    let apart = second.0 != first.0 && second.1 != first.1;
    assert!(apart, "{first:?} {second:?}");
    assert_eq!(keys(), [1, 1, 1, 1, 0]);
    // A release frees the key on each side, and the next key lands there.
    released(held);
    assert_eq!(keys(), made(&[second]), "the released request's key left");
    let (third, _) = leased(4, Some(9));
    assert_eq!(third, first);
}

/// A filter of a host's own build that turns an encode request's second
/// item away. It keeps no encoder for the second key of [`encode_frame`]'s,
/// and every row of any other decision.
#[derive(Debug)]
struct RefusesTheSecondItem;

impl Filter for RefusesTheSecondItem {
    fn name(&self) -> &'static str {
        "refuses-the-second-item"
    }

    fn keep(&self, facts: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        facts.phase != Phase::Encode || facts.session_key.hash >> 8 != 2
    }
}

/// An item's answer that turns a served encode request away behind its pair
/// gives back the claims that pair's decision made for the request's key.
/// The rule places a new key's prefill and its decode on the workers that
/// hold the fewest requests, and claims both. The request's second item is
/// then turned away, so nothing is filed, and neither side's session keeps
/// a pin.
///
/// The key's next request is placed again, by the leases as they then
/// stand, and claims the other prefill and the other decode. A claim left
/// standing led that request to the pair the first never reached, as a pin.
///
/// The engine's case:
/// `an_item_turned_away_takes_back_the_claims_the_rules_assignment_made` in
/// `crates/grand_central/tests/unit/engine/framework/encode/tests/pins_and_claims.rs`.
#[test]
fn a_served_encode_request_an_item_turns_away_gives_its_keys_claims_back() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("refuses-the-second-item", || Box::new(RefusesTheSecondItem))
        .expect("a free filter name");
    let catalog = catalog.without_lifecycle();
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    let filters = r#"["healthy", "refuses-the-second-item"]"#;
    let plan = ruled_encode_plan_in(&catalog, filters, |rule| {
        rule.assignment = Assignment::MinLoad;
    });
    store.apply(plan).expect("above the unbound start");
    let side = |role: Role, names: [&str; 2]| {
        names.map(|name| member(&store, &objects, name, role, PUSHING))
    };
    let [p0, p1] = side(Role::Prefill, ["p0", "p1"]);
    let [d0, d1] = side(Role::Decode, ["d0", "d1"]);
    member(&store, &objects, "e0", Role::Encode, PUSHING);
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    // The second prefill and the second decode hold a lease each: the rule
    // places a new key on the first of each side.
    for (id, target) in (1..).zip([p1, d1]) {
        file(&store, id, target, now);
    }
    let asked = |n: u8, items: u64| {
        let frame = keyed(encode_frame(n, items), 7);
        leased_encode(&store, &objects, frame, now)
    };
    // Two items: the first is placed, and the second's stage turns the
    // request away behind its pair.
    let (turned, encoders, filed) = asked(1, 2);
    let stage = shed_of(&turned).map(|shed| shed.1);
    assert_eq!(stage, Some("refuses-the-second-item"));
    assert_eq!((encoders, filed), (vec![], None), "nothing is filed");
    // Each side's session of the key, as a frame's reader makes them and a
    // lease of the key states them below.
    let sessions = {
        let frame = keyed(encode_frame(9, 1), 7);
        let request = Checked::new(frame).expect("a frame the record holds");
        let mut lists = Lists::default();
        let read = lists.of(&request);
        let facts = request.facts(&read, 0, 0);
        [Phase::Prefill, Phase::Decode].map(|phase| RequestFacts { phase, ..facts }.session())
    };
    // Neither side's session holds a worker: the table keeps no pin.
    let unpinned = [WorkerId(0); 2];
    let pins = sessions.map(|session| pins_of(&store, session).unwrap_or(unpinned));
    assert_eq!(pins, [unpinned; 2], "the claims are given back");
    // The leases point at the second of each side by then.
    for (id, target) in (3..).zip([p0, p0, d0, d0]) {
        file(&store, id, target, now);
    }
    let (next, _, filed) = asked(2, 1);
    let (prefill, decode) = filed.expect("a pick");
    let decode = decode.expect("issued with its prefill");
    assert_eq!([prefill.session, decode.session], sessions);
    assert_eq!(pairs_of(&next).first(), Some(&(p1, d1)), "placed again");
    let claims = (prefill.session_verdict, decode.session_verdict);
    let claimed = (SessionVerdict::Claimed(p1), SessionVerdict::Claimed(d1));
    assert_eq!(claims, claimed, "the key followed a pair it never reached");
}

/// The decode call behind an encode request's later decode hands its side's
/// key on at the rule's cap, as a pair's call does. An encode request that
/// states the later decode is filed its decode side chosen, on the key's
/// decode, and that side counts nothing there. Two whole requests of the
/// key stand on that decode when the call comes.
///
/// The call's frame carries no item and is read at the decode phase. It
/// meets the cap on the key's decode and is placed again, on the other
/// one. The side it issues states the respill, the decode side's pin leads
/// the other decode, and the key counts on the decode the call picked.
///
/// The pair's cases:
/// `a_served_decode_call_hands_its_sides_key_on_at_its_rules_cap` and
/// `a_served_pairs_later_decode_counts_its_key_from_its_calls_issue`.
#[test]
fn a_decode_call_behind_an_encode_request_hands_its_sides_key_on_at_its_rules_cap() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(ruled_encode_plan(Assignment::MinLoad, 2))
        .expect("above the unbound start");
    let p = member(&store, &objects, "p0", Role::Prefill, PUSHING);
    let decodes = ["d0", "d1"].map(|name| member(&store, &objects, name, Role::Decode, PUSHING));
    member(&store, &objects, "e0", Role::Encode, PUSHING);
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let leased = |n: u8, later: bool| {
        let mut frame = keyed(encode_frame(n, 1), 7);
        frame.decode_later = later;
        let (decision, _, filed) = leased_encode(&store, &objects, frame, now);
        let (prefill, decode) = filed.expect("a pick");
        assert_eq!(decode.is_none(), later, "frame {n}");
        (pairs_of(&decision)[0], prefill)
    };
    // The key's first request claims a decode, and its gateway's two
    // Dispatched frames pin each side's key.
    let (first, whole) = leased(1, false);
    sent_both(&store, &whole, now);
    // The second request's decode is sent later: its side stands chosen on
    // the key's decode.
    let (second, held) = leased(2, true);
    assert_eq!(second, first);
    let hand = open(&store, GATEWAY);
    let own = sent(&held, proto::Side::Prefill, 0);
    lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
    // The third request is whole: the key's decode carries two of its
    // requests, the first one's and this one's.
    let (third, _) = leased(3, false);
    assert_eq!(third, first);
    // What the decodes' objects count of the key: the key's decode alone.
    let other = usize::from(first.1 == decodes[0]);
    let keys = || keys_on(&objects, ["d0", "d1"]);
    let mut counted = [1, 1];
    counted[other] = 0;
    assert_eq!(keys(), counted, "the key's decode alone");
    // The call states no item and names the second request's lease. It
    // meets the cap on the key's decode.
    let mut call = keyed(encode_frame(4, 0), 7);
    (call.encode, call.decode_of) = (None, Some(held.id.0));
    let (again, issued) = leased_call(&store, &objects, call, now);
    let placed = [(p, decodes[other])];
    assert_eq!(pairs_of(&again), placed, "the call is placed again");
    let (side, _) = issued.expect("a pick's issue");
    assert_eq!(side.session_verdict, SessionVerdict::Respill);
    let moved = Some([decodes[other], first.1]);
    assert_eq!(pins_of(&store, side.session), moved);
    assert_eq!(keys(), [1, 1], "the call's issue, on the decode it picked");
}
