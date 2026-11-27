//! The cases of a pair's frame and of a decode call under a sessions rule on
//! their binding. They hold the session verdict each side's lease keeps, the
//! claims a decision makes for a key, and where each claim leaves. They also
//! hold two counts the leases on file keep on each side's worker: the keys
//! it carries, and a key's requests under a rule's cap.

use super::*;

/// A served pair under a sessions rule keeps a session verdict a side on
/// file. The first request of a key claims a worker on each side. The
/// decode's dispatch on the pick's second target is its claiming decision's
/// commit elsewhere: it withdraws the decode side's claim, so the key's one
/// decode pin is the decode the dispatch took, and the key's next pair
/// follows it. The prefill side's claim is the prefill's own to settle: it
/// stands through the decode's dispatch. Each side's lease states its own
/// claim, on its own worker.
#[test]
fn a_served_pairs_decode_dispatch_elsewhere_withdraws_the_decode_sides_own_claim() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(ruled_pair_plan())
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    for name in ["d0", "d1"] {
        side_member(&store, &objects, name, Role::Decode, None);
    }
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let leased = |n: u8| {
        let frame = keyed(at_prefill(n, WireModes::NONE, None), 7);
        let (decision, prefill, decode) = leased_pair(&store, &objects, frame, now);
        let decode = decode.expect("issued with its prefill");
        (pairs_of(&decision), prefill, decode)
    };
    let (chosen, prefill, decode) = leased(1);
    assert!(chosen.len() > 1 && chosen[0].1 != chosen[1].1, "{chosen:?}");
    let (first, second) = (chosen[0].1, chosen[1].1);
    let pins = || [prefill.session, decode.session].map(|session| pins_of(&store, session));
    let alone = |id| Some([id, WorkerId(0)]);
    assert_eq!(pins(), [alone(p), alone(first)], "a claim a side");
    let hand = open(&store, GATEWAY);
    let elsewhere = sent(&prefill, proto::Side::Decode, 1);
    lifecycle::dispatched(&store, &hand, &elsewhere, now).expect("the frame reads");
    assert_eq!(pins(), [alone(p), alone(second)], "the decode's claim left");
    let claims = (prefill.session_verdict, decode.session_verdict);
    let own = (SessionVerdict::Claimed(p), SessionVerdict::Claimed(first));
    assert_eq!(claims, own, "each side's lease states its own claim");
    // The key's next pair follows the decode its first one went to.
    let (next, prefill, decode) = leased(2);
    assert_eq!(next, [(p, second)]);
    let followed = (prefill.session_verdict, decode.session_verdict);
    assert_eq!(followed, (SessionVerdict::Pinned, SessionVerdict::Pinned));
}

/// A decode call's side keeps the call's own session verdict. The pair's
/// first decision claims a decode for the key, and its side stands chosen
/// under that claim. The call follows the claim as the key's pin: the side
/// it issues states a pin, on file and in its lease. The issue's undo puts
/// the side back under the first decision's claim.
#[test]
fn a_served_decode_calls_side_keeps_the_calls_own_session_verdict() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(ruled_pair_plan())
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    for name in ["d0", "d1"] {
        side_member(&store, &objects, name, Role::Decode, None);
    }
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let mut frame = keyed(at_prefill(1, WireModes::NONE, None), 7);
    frame.decode_later = true;
    let (decision, prefill, decode) = leased_pair(&store, &objects, frame, now);
    assert_eq!(decode, None, "the decode side stands chosen");
    let first = pairs_of(&decision)[0].1;
    let on_file = || {
        let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
        let side = sides.decode.expect("a pair's lease");
        (side.state, side.session_verdict)
    };
    let claimed = (LeaseState::Chosen, SessionVerdict::Claimed(first));
    assert_eq!(on_file(), claimed);
    let hand = open(&store, GATEWAY);
    let own = sent(&prefill, proto::Side::Prefill, 0);
    lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
    let call = keyed(at_call(2, prefill.id, WireModes::NONE, None), 7);
    let (again, issued) = leased_call(&store, &objects, call, now);
    assert_eq!(pairs_of(&again), [(p, first)], "the call follows the claim");
    let (side, issue) = issued.expect("a pick's issue");
    assert_eq!(side.session_verdict, SessionVerdict::Pinned);
    assert_eq!(on_file(), (LeaseState::ISSUED, SessionVerdict::Pinned));
    assert!(store.unissue_decode(GATEWAY, prefill.id, issue));
    assert_eq!(on_file(), claimed);
}

/// A served pair whose commit refused every target keeps no claim on
/// either side. A frame that stated the later decode leaves its decode
/// side chosen under the claim the pair's first decision made, and no
/// stage heard that side's issue: the release that ends the prefill takes
/// the decode's claim back with it, and the key's next pair is placed as a
/// key no request carried. A release that names such a decode alone takes
/// its claim back at that word. A release for another reason takes no
/// claim back.
#[test]
fn a_served_refused_pair_keeps_no_claim_on_a_decode_that_stood_chosen() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(ruled_pair_plan())
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    for name in ["d0", "d1"] {
        side_member(&store, &objects, name, Role::Decode, None);
    }
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let later = |n: u8, key: u64| {
        let mut frame = keyed(at_prefill(n, WireModes::NONE, None), key);
        frame.decode_later = true;
        let (decision, prefill, _) = leased_pair(&store, &objects, frame, now);
        let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
        let decode = sides.decode.expect("a pair's lease");
        (pairs_of(&decision), prefill, decode)
    };
    let (chosen, prefill, decode) = later(1, 7);
    assert_eq!(decode.state, LeaseState::Chosen);
    let pins = |keys: [Session; 2]| keys.map(|session| pins_of(&store, session));
    let alone = |id| Some([id, WorkerId(0)]);
    let none = Some([WorkerId(0); 2]);
    let keys = [prefill.session, decode.session];
    assert_eq!(pins(keys), [alone(p), alone(chosen[0].1)], "a claim a side");
    let hand = open(&store, GATEWAY);
    let release = |lease: &Lease, side: proto::Side, reason: proto::released::Reason| {
        let word = proto::Released {
            what: Some(proto::released::What::LeaseId(lease.id.0)),
            reason: reason.into(),
            side: side.into(),
        };
        lifecycle::released(&store, &hand, &word).expect("the frame reads");
    };
    let whole = proto::Side::Unspecified;
    let refused = proto::released::Reason::CommitFailed;
    release(&prefill, whole, refused);
    assert_eq!(pins(keys), [none; 2], "neither claim stays");
    // The key's next pair claims anew, over both decodes, and a release
    // for another reason takes no claim back.
    let (next, again, decode) = later(2, 7);
    assert_eq!(next.len(), 2, "{next:?}");
    assert!(matches!(decode.session_verdict, SessionVerdict::Claimed(_)));
    release(&again, whole, proto::released::Reason::ClientGone);
    assert_eq!(pins(keys), [alone(p), alone(next[0].1)]);
    // A release that names the decode alone takes that side's claim back
    // at its word, and the prefill's release then takes the prefill's.
    let (other, prefill, decode) = later(3, 8);
    let keys = [prefill.session, decode.session];
    assert_eq!(pins(keys), [alone(p), alone(other[0].1)], "a claim a side");
    release(&prefill, proto::Side::Decode, refused);
    assert_eq!(pins(keys), [alone(p), none], "the decode's claim goes");
    release(&prefill, whole, refused);
    assert_eq!(pins(keys), [none; 2]);
}

/// A served pair whose commit refused every target keeps no claim on a
/// decode side a decode call issued. The pair's first decision claims a
/// decode for the key, and a call behind the prefill's dispatch follows
/// that claim as the key's pin: the side it issues states a pin and stands
/// in the place of the chosen side the claim was made for. The release
/// that ends the issued side ends that chosen side too, so the first
/// decision's claim goes with the pair's refusal, or with a release that
/// names the decode alone. A release that names a side the pair's lease
/// does not hold ends nothing and takes no claim. The prefill's dispatch
/// confirmed its side's claim, and that pin stays.
#[test]
fn a_served_refused_pair_keeps_no_claim_behind_a_decode_call() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(ruled_pair_plan())
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    for name in ["d0", "d1"] {
        side_member(&store, &objects, name, Role::Decode, None);
    }
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let hand = open(&store, GATEWAY);
    let alone = |id| Some([id, WorkerId(0)]);
    let none = Some([WorkerId(0); 2]);
    let refused = |lease: &Lease, side: proto::Side| {
        let word = proto::Released {
            what: Some(proto::released::What::LeaseId(lease.id.0)),
            reason: proto::released::Reason::CommitFailed.into(),
            side: side.into(),
        };
        lifecycle::released(&store, &hand, &word).expect("the frame reads");
    };
    // The pair's refusal alone, then behind a release of the decode alone.
    for (n, key, decode_first) in [(1, 7, false), (3, 8, true)] {
        let mut frame = keyed(at_prefill(n, WireModes::NONE, None), key);
        frame.decode_later = true;
        let (decision, prefill, _) = leased_pair(&store, &objects, frame, now);
        let first = pairs_of(&decision)[0].1;
        let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
        let decode = sides.decode.expect("a pair's lease");
        let pins = || [prefill.session, decode.session].map(|session| pins_of(&store, session));
        let own = sent(&prefill, proto::Side::Prefill, 0);
        lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
        let call = keyed(at_call(n + 1, prefill.id, WireModes::NONE, None), key);
        let (_, issued) = leased_call(&store, &objects, call, now);
        let (side, _) = issued.expect("a pick's issue");
        assert_eq!(side.session_verdict, SessionVerdict::Pinned);
        assert_eq!(
            pins(),
            [alone(p), alone(first)],
            "the call follows the claim"
        );
        // A release that names a side a pair's lease does not hold ends
        // nothing, and the claim stays.
        for other in [proto::Side::Single, proto::Side::Encode] {
            refused(&prefill, other);
            assert_eq!(
                pins(),
                [alone(p), alone(first)],
                "a release of another side leaves the claim"
            );
        }
        if decode_first {
            refused(&prefill, proto::Side::Decode);
            assert_eq!(
                pins(),
                [alone(p), none],
                "the claim goes at the decode's word"
            );
        }
        refused(&prefill, proto::Side::Unspecified);
        assert_eq!(pins(), [alone(p), none], "the first decision's claim goes");
    }
}

/// A refusal told late takes back no claim of a newer decision. A pair's
/// first decision claims a worker for each side's key, and its decode is
/// asked for later. Both keys idle out while the pair's lease lives, and a
/// newer pair of the session claims the same two workers. The first pair's
/// decode call follows the newer claim as the key's pin, and its commit
/// then refuses every target. The newer pair's claims stand at that
/// release, and the session's next pair follows them ahead of the newer
/// pair's dispatch.
#[test]
fn a_served_pairs_late_refusal_leaves_a_newer_pairs_claims() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    // A key idles out inside the life of a pair's lease.
    let idle = ruled_pair_plan_with(|rule| (rule.ttl_s, rule.sweep_s) = (1, 1));
    store.apply(idle).expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    let d = side_member(&store, &objects, "d0", Role::Decode, None);
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let hand = open(&store, GATEWAY);
    let alone = |id| Some([id, WorkerId(0)]);
    let claimed = (SessionVerdict::Claimed(p), SessionVerdict::Claimed(d));
    let pinned = (SessionVerdict::Pinned, SessionVerdict::Pinned);
    // The first pair, its decode asked for later: a claim a side.
    let mut frame = keyed(at_prefill(1, WireModes::NONE, None), 7);
    frame.decode_later = true;
    let (_, first, _) = leased_pair(&store, &objects, frame, now);
    let sides = LeasesStore::sides(&*store, GATEWAY, first.id).expect("on file");
    let chosen = sides.decode.expect("a pair's lease");
    assert_eq!((first.session_verdict, chosen.session_verdict), claimed);
    let pins = || [first.session, chosen.session].map(|session| pins_of(&store, session));
    assert_eq!(pins(), [alone(p), alone(d)], "a claim a side");
    let own = sent(&first, proto::Side::Prefill, 0);
    lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
    // Both keys idle out, and a newer pair of the session claims the same
    // two workers. No dispatch of it is told yet.
    let later = now + Duration::from_secs(3);
    let newer = keyed(at_prefill(2, WireModes::NONE, None), 7);
    let (_, prefill, decode) = leased_pair(&store, &objects, newer, later);
    let decode = decode.expect("issued with its prefill");
    assert_eq!((prefill.session_verdict, decode.session_verdict), claimed);
    // The first pair's decode call follows the newer claim as a pin, and
    // its commit then refuses every target.
    let call = keyed(at_call(3, first.id, WireModes::NONE, None), 7);
    let (_, issued) = leased_call(&store, &objects, call, later);
    let (side, _) = issued.expect("a pick's issue");
    assert_eq!(side.session_verdict, SessionVerdict::Pinned);
    let word = proto::Released {
        what: Some(proto::released::What::LeaseId(first.id.0)),
        reason: proto::released::Reason::CommitFailed.into(),
        side: proto::Side::Unspecified.into(),
    };
    lifecycle::released(&store, &hand, &word).expect("the frame reads");
    assert_eq!(
        pins(),
        [alone(p), alone(d)],
        "the newer pair's claims stand"
    );
    // The session's next pair follows them, with a second decode to draw.
    let other = side_member(&store, &objects, "d1", Role::Decode, None);
    let third = keyed(at_prefill(4, WireModes::NONE, None), 7);
    let (decision, prefill, decode) = leased_pair(&store, &objects, third, later);
    assert_eq!(pairs_of(&decision), [(p, d)], "no pair on {other:?}");
    let decode = decode.expect("issued with its prefill");
    assert_eq!((prefill.session_verdict, decode.session_verdict), pinned);
}

/// A decode call's claim leaves at its own release. The pair's first
/// decision claims a decode for the key, and the gateway then holds a veto
/// on that decode. The call, three seconds behind the pair's decision,
/// finds no pin it can follow and claims the other decode. That claim is
/// dated by the pair's lease, which the side's release is told under. So
/// the pair's refusal takes the call's claim back.
#[test]
fn a_served_decode_calls_claim_leaves_at_its_own_release() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(ruled_pair_plan())
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    for name in ["d0", "d1"] {
        side_member(&store, &objects, name, Role::Decode, None);
    }
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let hand = open(&store, GATEWAY);
    let mut frame = keyed(at_prefill(1, WireModes::NONE, None), 7);
    frame.decode_later = true;
    let (decision, prefill, _) = leased_pair(&store, &objects, frame, now);
    let chosen = pairs_of(&decision);
    assert!(chosen.len() > 1 && chosen[0].1 != chosen[1].1, "{chosen:?}");
    let (first, second) = (chosen[0].1, chosen[1].1);
    let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
    let decode = sides.decode.expect("a pair's lease");
    let pins = || pins_of(&store, decode.session);
    assert_eq!(pins(), Some([first, WorkerId(0)]), "the first claim");
    let own = sent(&prefill, proto::Side::Prefill, 0);
    lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
    // The gateway's veto stands on the claimed decode at the call.
    let later = now + Duration::from_secs(3);
    store.gateway_veto(GATEWAY, first, Some(later + Duration::from_secs(60)));
    let call = keyed(at_call(2, prefill.id, WireModes::NONE, None), 7);
    let (again, issued) = leased_call(&store, &objects, call, later);
    assert_eq!(pairs_of(&again), [(p, second)], "the other decode");
    let (side, _) = issued.expect("a pick's issue");
    assert_eq!(side.session_verdict, SessionVerdict::Claimed(second));
    assert_eq!(pins(), Some([first, second]), "the call's claim");
    let word = proto::Released {
        what: Some(proto::released::What::LeaseId(prefill.id.0)),
        reason: proto::released::Reason::CommitFailed.into(),
        side: proto::Side::Unspecified.into(),
    };
    lifecycle::released(&store, &hand, &word).expect("the frame reads");
    assert_eq!(pins(), Some([first, WorkerId(0)]), "the call's claim left");
}

/// A served pair's keyed pick counts for its key on each side's worker,
/// under the stock controller's own catalog: the prefill's worker and the
/// decode's each carry one key more while the pair's lease stands, and a
/// pair with no key counts for none. A rule that places a new key on the
/// worker with the fewest keys reads those counts on each side: the next
/// key's pair takes the prefill and the decode that carry none, and the
/// key after a release takes the two the release freed.
///
/// The classic router counts the same. Its load guard counts every
/// request that carries a key (`WorkerLoadGuard::from_acquired_load` in
/// `crates/worker/src/load/inflight.rs`). A pair's request is guarded on
/// each side under the route's key (`acquire_prefill` in
/// `model_gateway/src/routers/common/prefill_admission.rs`, `LoadGuards::new`
/// in `model_gateway/src/routers/grpc/context.rs`). Its sticky policy places
/// a new key on a worker with the fewest keys in flight (`min_group_select`
/// in `model_gateway/src/policies/manual.rs`).
#[test]
fn a_served_pairs_keyed_lease_counts_its_key_on_each_sides_worker() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(ruled_pair_plan_of(Assignment::MinGroup))
        .expect("above the unbound start");
    let [p0, p1] =
        ["p0", "p1"].map(|name| side_member(&store, &objects, name, Role::Prefill, None));
    let [d0, d1] = ["d0", "d1"].map(|name| side_member(&store, &objects, name, Role::Decode, None));
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    // What the workers' objects count, and what the pairs that stand make:
    // the two prefills, then the two decodes.
    let keys = || keys_on(&objects, ["p0", "p1", "d0", "d1"]);
    let made = |pairs: &[(WorkerId, WorkerId)]| {
        let carried = |id| {
            let carrying = pairs.iter().filter(|pair| pair.0 == id || pair.1 == id);
            carrying.count()
        };
        [p0, p1, d0, d1].map(carried)
    };
    let leased = |n: u8, key: Option<u64>| {
        let frame = at_prefill(n, WireModes::NONE, None);
        let frame = match key {
            Some(key) => keyed(frame, key),
            None => frame,
        };
        let (decision, prefill, _) = leased_pair(&store, &objects, frame, now);
        (pairs_of(&decision)[0], prefill.id)
    };
    let released = |lease: LeaseId| {
        let gone = ReleaseReason::ClientGone;
        assert!(LeasesStore::released(&*store, GATEWAY, lease, None, gone));
    };
    let (_, keyless) = leased(1, None);
    assert_eq!(keys(), [0; 4], "a pair with no key counts for none");
    released(keyless);
    let (first, held) = leased(2, Some(7));
    assert_eq!(keys(), made(&[first]), "a key on each side's worker");
    // The next key's pair takes the prefill and the decode with no key.
    let (second, _) = leased(3, Some(8));
    let apart = second.0 != first.0 && second.1 != first.1;
    assert!(apart, "{first:?} {second:?}");
    assert_eq!(keys(), [1; 4]);
    // A release frees the key on each side, and the next key lands there.
    released(held);
    assert_eq!(keys(), made(&[second]), "the released pair's key left");
    let (third, _) = leased(4, Some(9));
    assert_eq!(third, first);
}

/// A served pair under a rule with an in-flight cap hands each side's key
/// on at the cap, as a served single's rule does. The leases on file count
/// each side's key under the side's own session, and each side's decision
/// reads that count. The key's first two pairs stand on one prefill and one
/// decode. The third meets the cap on both sides. It is placed again, on
/// the prefill and the decode that carry none of the key's requests, and
/// each side's lease states the respill. Each side's pin leads there where
/// the pair is decided, ahead of any Dispatched frame, so the key's fourth
/// pair follows the move on both sides.
#[test]
fn a_served_pair_hands_each_sides_key_on_at_its_rules_cap() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    // The rule places a key on the worker with the fewest requests: a
    // request placed again at the cap leaves the key's worker.
    store
        .apply(ruled_pair_plan_under(Assignment::MinLoad, 2))
        .expect("above the unbound start");
    for (role, names) in [(Role::Prefill, ["p0", "p1"]), (Role::Decode, ["d0", "d1"])] {
        for name in names {
            side_member(&store, &objects, name, role, None);
        }
    }
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let leased = |n: u8| {
        let frame = keyed(at_prefill(n, WireModes::NONE, None), 7);
        let (decision, prefill, decode) = leased_pair(&store, &objects, frame, now);
        let decode = decode.expect("issued with its prefill");
        let verdicts = (prefill.session_verdict, decode.session_verdict);
        (pairs_of(&decision)[0], verdicts, prefill, decode)
    };
    // The key's first pair claims a worker a side, and its gateway's two
    // Dispatched frames pin each side's key there.
    let (first, claims, prefill, _) = leased(1);
    let claimed = (
        SessionVerdict::Claimed(first.0),
        SessionVerdict::Claimed(first.1),
    );
    assert_eq!(claims, claimed);
    let hand = open(&store, GATEWAY);
    for side in [proto::Side::Prefill, proto::Side::Decode] {
        let own = sent(&prefill, side, 0);
        lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
    }
    // One request of the key in flight a side: the second pair stays.
    let pinned = (SessionVerdict::Pinned, SessionVerdict::Pinned);
    let (second, stays, ..) = leased(2);
    assert_eq!((second, stays), (first, pinned));
    // Two in flight on each side's worker: the third pair is placed again
    // on both sides, and no Dispatched frame has come for it.
    let (third, respills, prefill, decode) = leased(3);
    assert!(
        third.0 != first.0 && third.1 != first.1,
        "{first:?} {third:?}"
    );
    assert_eq!(respills, (SessionVerdict::Respill, SessionVerdict::Respill));
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

/// A served decode call under a rule with an in-flight cap hands its side's
/// key on at the cap. A call is decided through the entry that issues no
/// lease, and its maker counts a key under the side's own session, as the
/// leases on file do. The key's second pair leaves its decode for later:
/// that side stands chosen on the key's decode and counts nothing there.
/// Two whole pairs of the key stand on that decode when the call comes. The
/// call meets the cap there and is placed again, on the other decode. The
/// side it issues states the respill, and the decode side's pin leads the
/// other decode where the call is decided.
#[test]
fn a_served_decode_call_hands_its_sides_key_on_at_its_rules_cap() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(ruled_pair_plan_under(Assignment::MinLoad, 2))
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    let decodes = ["d0", "d1"].map(|name| side_member(&store, &objects, name, Role::Decode, None));
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let hand = open(&store, GATEWAY);
    let leased = |n: u8, later: bool| {
        let mut frame = keyed(at_prefill(n, WireModes::NONE, None), 7);
        frame.decode_later = later;
        let (decision, prefill, _) = leased_pair(&store, &objects, frame, now);
        (pairs_of(&decision)[0], prefill)
    };
    // The key's first pair claims a decode, and its gateway's two
    // Dispatched frames pin each side's key.
    let (first, whole) = leased(1, false);
    for side in [proto::Side::Prefill, proto::Side::Decode] {
        let own = sent(&whole, side, 0);
        lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
    }
    // The second pair's decode is sent later: its side stands chosen on
    // the key's decode.
    let (second, held) = leased(2, true);
    assert_eq!(second, first);
    let own = sent(&held, proto::Side::Prefill, 0);
    lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
    // The third pair is whole: the key's decode carries two of its
    // requests, the first pair's and this one's.
    let (third, _) = leased(3, false);
    assert_eq!(third, first);
    // The call meets the cap on the key's decode.
    let call = keyed(at_call(4, held.id, WireModes::NONE, None), 7);
    let (again, issued) = leased_call(&store, &objects, call, now);
    let other = if first.1 == decodes[0] {
        decodes[1]
    } else {
        decodes[0]
    };
    assert_eq!(pairs_of(&again), [(p, other)], "the call is placed again");
    let (side, _) = issued.expect("a pick's issue");
    assert_eq!(side.session_verdict, SessionVerdict::Respill);
    let moved = Some([other, first.1]);
    assert_eq!(pins_of(&store, side.session), moved);
}

/// A served pair whose decode is sent later counts its key on the
/// prefill's worker at its filing, and on the decode's from the call that
/// issues the side: a side that stands chosen carries no request. The
/// decode the pair chose is not ready at the call, so the call picks the
/// other, and the key counts on the decode the call picked. The issue's
/// undo gives the decode's count up, and the prefill's stands. A pair
/// whose first frame states no key counts none of its own, and its call's
/// key counts all the same: each side counts the key its own frame states.
#[test]
fn a_served_pairs_later_decode_counts_its_key_from_its_calls_issue() {
    for first_key in [Some(7), None] {
        let store = Arc::new(MemoryStore::new());
        let objects = holder(&store);
        store
            .apply(ruled_pair_plan())
            .expect("above the unbound start");
        let p = side_member(&store, &objects, "p0", Role::Prefill, None);
        let decodes =
            ["d0", "d1"].map(|name| side_member(&store, &objects, name, Role::Decode, None));
        LeasesStore::set_term(&*store, 1);
        let now = Instant::now();
        let keys = || keys_on(&objects, ["p0", "d0", "d1"]);
        let frame = at_prefill(1, WireModes::NONE, None);
        let mut frame = match first_key {
            Some(key) => keyed(frame, key),
            None => frame,
        };
        frame.decode_later = true;
        let (decision, prefill, decode) = leased_pair(&store, &objects, frame, now);
        // The prefill's worker carries the first frame's key, where it states one.
        let held = usize::from(first_key.is_some());
        assert_eq!((decode, keys()), (None, [held, 0, 0]), "a chosen side");
        let hand = open(&store, GATEWAY);
        let own = sent(&prefill, proto::Side::Prefill, 0);
        lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
        let first = pairs_of(&decision)[0].1;
        stand(&objects, &store, first, WorkerStatus::NotReady);
        let call = keyed(at_call(2, prefill.id, WireModes::NONE, None), 7);
        let (again, issued) = leased_call(&store, &objects, call, now);
        let other = if first == decodes[0] { 1 } else { 0 };
        assert_eq!(pairs_of(&again), [(p, decodes[other])], "it picks again");
        let (_, issue) = issued.expect("a pick's issue");
        let mut counted = [held, 0, 0];
        counted[1 + other] = 1;
        assert_eq!(keys(), counted, "the call's issue, on the decode it picked");
        assert!(store.unissue_decode(GATEWAY, prefill.id, issue));
        assert_eq!(keys(), [held, 0, 0], "the issue taken back");
    }
}

/// The pair plan's document under a sessions rule that states no cap, as
/// the stock controller's own catalog takes it: each side draws among the
/// workers its health filter keeps, and each side's key follows the worker
/// its first request claimed.
fn ruled_pair_plan() -> AppliedSpec {
    ruled_pair_plan_of(Assignment::Delegate)
}

/// [`ruled_pair_plan`] where the rule places a new key by `assignment`.
fn ruled_pair_plan_of(assignment: Assignment) -> AppliedSpec {
    ruled_pair_plan_under(assignment, u16::MAX)
}

/// [`ruled_pair_plan_of`] under the in-flight cap `cap`, 65535 for none.
fn ruled_pair_plan_under(assignment: Assignment, cap: u16) -> AppliedSpec {
    ruled_pair_plan_with(|rule| (rule.assignment, rule.inflight_cap) = (assignment, cap))
}

/// [`ruled_pair_plan`] with its sessions rule as `set` leaves it.
fn ruled_pair_plan_with(set: impl FnOnce(&mut SessionsSpec)) -> AppliedSpec {
    let document = r#"{
      "apiVersion": "smg.policy/v1",
      "generation": 1,
      "view": { "headers": ["x-session-id"] },
      "profiles": {
        "prefills": { "filter": ["healthy"], "pick": "random" },
        "decodes": { "filter": ["healthy"], "pick": "random" }
      },
      "bindings": [
        { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair",
          "profiles": { "prefill": "prefills", "decode": "decodes" },
          "sessions": { "sticky_header": "x-session-id", "inflight_cap": 65535 } },
        { "profile": "prefills" }
      ]
    }"#;
    let mut document: PolicySpec =
        serde_json::from_str(document).expect("a document of this build");
    let rule = document.bindings[0].sessions.as_mut();
    set(rule.expect("the pair's rule"));
    staged(&document, &StageRegistry::builtin().without_lifecycle())
}

/// A decode call the rule places on another decode than its pair's first
/// decision claimed takes the key with it at its dispatch. Behind the call
/// the key holds both decodes, the one its first decision claimed ahead.
/// Behind the dispatch of the call's decode it holds that decode alone:
/// the decode the request left is no pin of the key. The pair's refusal
/// leaves the key on the first decode, with a release of the decode ahead
/// of it or without one. A call that follows an older pin of the key
/// claims nothing, and its dispatch leaves the key's pins as they stood.
/// A prefill dispatched on the pick's second target changes none of it.
#[test]
fn a_served_decode_call_on_another_decode_takes_the_keys_pin_with_it() {
    /// How the pair ends behind its decode call.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Ends {
        Refused,
        DecodeReleasedFirst,
        DecodeDispatched,
    }
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(ruled_pair_plan())
        .expect("above the unbound start");
    let p = side_member(&store, &objects, "p0", Role::Prefill, None);
    for name in ["d0", "d1"] {
        side_member(&store, &objects, name, Role::Decode, None);
    }
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let hand = open(&store, GATEWAY);
    let alone = |id| Some([id, WorkerId(0)]);
    let refused = |lease: &Lease, side: proto::Side| {
        let word = proto::Released {
            what: Some(proto::released::What::LeaseId(lease.id.0)),
            reason: proto::released::Reason::CommitFailed.into(),
            side: side.into(),
        };
        lifecycle::released(&store, &hand, &word).expect("the frame reads");
    };
    // Each pair by the number of its first frame, the key its frames
    // hand, the target its prefill's dispatch names, its end, and whether
    // the key holds an older pin on the decode its call takes.
    let pairs = [
        (1, 7, 0, Ends::Refused, false),
        (3, 8, 0, Ends::DecodeReleasedFirst, false),
        (5, 9, 0, Ends::DecodeDispatched, true),
        (9, 10, 0, Ends::DecodeDispatched, false),
        (11, 11, 1, Ends::DecodeDispatched, false),
    ];
    for (n, key, taken, ends, older) in pairs {
        // The key's older pin: an earlier pair of the key sent on both
        // sides, whose decode is not ready for the pair of this row.
        let older = older.then(|| {
            let frame = keyed(at_prefill(n + 2, WireModes::NONE, None), key);
            let (decision, earlier, _) = leased_pair(&store, &objects, frame, now);
            for side in [proto::Side::Prefill, proto::Side::Decode] {
                let word = sent(&earlier, side, 0);
                lifecycle::dispatched(&store, &hand, &word, now).expect("the frame reads");
            }
            let pinned = pairs_of(&decision)[0].1;
            stand(&objects, &store, pinned, WorkerStatus::NotReady);
            pinned
        });
        let mut frame = keyed(at_prefill(n, WireModes::NONE, None), key);
        frame.decode_later = true;
        let (decision, prefill, _) = leased_pair(&store, &objects, frame, now);
        let first = pairs_of(&decision)[0].1;
        let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
        let chosen = sides.decode.expect("a pair's lease").session;
        // The pins of the prefill side's key, then of the decode side's as
        // the pair's first frame states it.
        let pins = || [prefill.session, chosen].map(|session| pins_of(&store, session));
        let own = sent(&prefill, proto::Side::Prefill, taken);
        lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
        // The decode side's key ahead of the call: the first claim, behind
        // the older pin where the key holds one.
        let held = older.map_or_else(|| alone(first), |pinned| Some([pinned, first]));
        assert_eq!(pins(), [alone(p), held], "pair {n}: the first claim");
        stand(&objects, &store, first, WorkerStatus::NotReady);
        if let Some(pinned) = older {
            stand(&objects, &store, pinned, WorkerStatus::Ready);
        }
        let call = keyed(at_call(n + 1, prefill.id, WireModes::NONE, None), key);
        let (again, issued) = leased_call(&store, &objects, call, now);
        let other = pairs_of(&again)[0].1;
        assert_ne!(other, first, "pair {n}: the call names the other decode");
        let (side, _) = issued.expect("a pick's issue");
        // The call follows the key's older pin where one stands, and
        // claims its decode behind the first claim where none does.
        let (verdict, called) = match older {
            Some(_) => (SessionVerdict::Pinned, held),
            None => (SessionVerdict::Claimed(other), Some([first, other])),
        };
        assert_eq!(side.session_verdict, verdict, "pair {n}: the verdict");
        assert_eq!(
            pins_of(&store, side.session),
            called,
            "pair {n}: the call's claim"
        );
        assert_eq!(
            pins(),
            [alone(p), called],
            "pair {n}: behind the call's issue"
        );
        let left = match ends {
            Ends::Refused | Ends::DecodeReleasedFirst => {
                if ends == Ends::DecodeReleasedFirst {
                    refused(&prefill, proto::Side::Decode);
                    assert_eq!(
                        pins(),
                        [alone(p), alone(first)],
                        "pair {n}: the decode released"
                    );
                }
                refused(&prefill, proto::Side::Unspecified);
                alone(first)
            }
            Ends::DecodeDispatched => {
                let to = sent(&prefill, proto::Side::Decode, 0);
                lifecycle::dispatched(&store, &hand, &to, now).expect("the frame reads");
                // Every word the dispatch tells returned: none faulted.
                assert_eq!(store.lifecycle_faults(), 0, "pair {n}: no fault");
                older.map_or_else(|| alone(other), |_| held)
            }
        };
        assert_eq!(pins(), [alone(p), left], "pair {n}: at the end");
        // The decode the request left is ready again: the key reads the same.
        stand(&objects, &store, first, WorkerStatus::Ready);
        assert_eq!(pins(), [alone(p), left], "pair {n}: the first decode ready");
    }
}

/// A decode call that is taken back leaves the key's pins as the call
/// wrote them until the next word of its pair. The dispatch of the decode
/// the pair's first decision claimed, told under the pair's own lease,
/// leaves the key on that decode alone. A call whose commit failed leaves
/// the key there at once. A second call behind one taken back follows a
/// pin of the key and claims nothing: sent on the first call's decode, the
/// pair's decode leaves the key on that decode alone; sent on the first
/// decision's decode, it leaves both pins.
#[test]
fn a_served_decode_call_taken_back_takes_the_keys_pin_with_it_to_its_first_decode() {
    let store = Arc::new(MemoryStore::new());
    let objects = holder(&store);
    store
        .apply(ruled_pair_plan())
        .expect("above the unbound start");
    side_member(&store, &objects, "p0", Role::Prefill, None);
    for name in ["d0", "d1"] {
        side_member(&store, &objects, name, Role::Decode, None);
    }
    LeasesStore::set_term(&*store, 1);
    let now = Instant::now();
    let hand = open(&store, GATEWAY);
    let alone = |id| Some([id, WorkerId(0)]);
    // Each pair by the number of its first frame, the key its frames
    // hand, the reason its call is taken back under, and for a second call
    // behind the one taken back whether the first decode is ready for it.
    let pairs = [
        (1, 7, ReleaseReason::CommitFailed, None, "commit failed"),
        (3, 8, ReleaseReason::Abandoned, None, "abandoned"),
        (5, 9, ReleaseReason::LateDecision, None, "late decision"),
        (7, 10, ReleaseReason::Abandoned, Some(false), "called again"),
        (
            10,
            11,
            ReleaseReason::Abandoned,
            Some(true),
            "called again, ready",
        ),
    ];
    for (n, key, reason, second, name) in pairs {
        let mut frame = keyed(at_prefill(n, WireModes::NONE, None), key);
        frame.decode_later = true;
        let (decision, prefill, _) = leased_pair(&store, &objects, frame, now);
        let first = pairs_of(&decision)[0].1;
        let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
        let chosen = sides.decode.expect("a pair's lease").session;
        let own = sent(&prefill, proto::Side::Prefill, 0);
        lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
        stand(&objects, &store, first, WorkerStatus::NotReady);
        let call = keyed(at_call(n + 1, prefill.id, WireModes::NONE, None), key);
        let (again, _) = leased_call(&store, &objects, call, now);
        let other = pairs_of(&again)[0].1;
        assert_ne!(other, first, "{name}: the other decode");
        assert!(
            store.take_back_decode(GATEWAY, prefill.id, reason),
            "{name}: the call's issue is taken back"
        );
        // A failed commit takes the call's claim back with the issue.
        let behind = if matches!(reason, ReleaseReason::CommitFailed) {
            alone(first)
        } else {
            Some([first, other])
        };
        assert_eq!(pins_of(&store, chosen), behind, "{name}: taken back");
        // A second call follows a pin of the key and claims nothing: the
        // first call's decode while the first decode is not ready, the
        // first decode once it is.
        let second = second.map(|ready| {
            if ready {
                stand(&objects, &store, first, WorkerStatus::Ready);
            }
            let call = keyed(at_call(n + 2, prefill.id, WireModes::NONE, None), key);
            let (again, issued) = leased_call(&store, &objects, call, now);
            let (side, _) = issued.expect("a pick's issue");
            let verdict = side.session_verdict;
            assert_eq!(verdict, SessionVerdict::Pinned, "{name}: a pin followed");
            let decode = pairs_of(&again)[0].1;
            assert_eq!(decode, if ready { first } else { other }, "{name}: the pin");
            assert_eq!(pins_of(&store, chosen), behind, "{name}: called again");
            decode
        });
        // The pair's decode is sent. The first pick's own leaves the key on
        // it alone. A second call's on the first call's decode leaves the
        // key on that decode alone: the lease went where the call's claim
        // stands. A second call's on the first decode claimed nothing
        // there, and leaves both pins.
        let to = sent(&prefill, proto::Side::Decode, 0);
        lifecycle::dispatched(&store, &hand, &to, now).expect("the frame reads");
        let left = match second {
            None => alone(first),
            Some(decode) if decode == other => alone(other),
            Some(_) => Some([first, other]),
        };
        assert_eq!(pins_of(&store, chosen), left, "{name}: dispatched");
        stand(&objects, &store, first, WorkerStatus::Ready);
    }
}

/// A decode call takes the key's pin with it from the first claim of its
/// own pair alone. A key that idled out behind a pair's first decision
/// starts over, and a second request of the key that claims the same
/// decode holds a claim of its own. The first pair's call then claims its
/// decode behind that claim, and the dispatch of the call's decode leaves
/// both: the key reads the second request's decode ahead of the call's.
/// The second request's dispatch there, or its refusal, moves no pin.
#[test]
fn a_served_decode_call_leaves_the_claim_of_a_later_request_of_its_key() {
    for dispatched in [true, false] {
        let store = Arc::new(MemoryStore::new());
        let objects = holder(&store);
        // A key idles out inside the life of a pair's lease.
        let idle = ruled_pair_plan_with(|rule| (rule.ttl_s, rule.sweep_s) = (1, 1));
        store.apply(idle).expect("above the unbound start");
        side_member(&store, &objects, "p0", Role::Prefill, None);
        let d0 = side_member(&store, &objects, "d0", Role::Decode, None);
        let d1 = side_member(&store, &objects, "d1", Role::Decode, None);
        LeasesStore::set_term(&*store, 1);
        let now = Instant::now();
        let later = now + Duration::from_secs(3);
        let hand = open(&store, GATEWAY);
        let alone = |id| Some([id, WorkerId(0)]);
        // A pair of the key by the number of its first frame, its decode
        // asked for later: its lease and the decode side's key.
        let pair = |n: u8, at: Instant| {
            let mut frame = keyed(at_prefill(n, WireModes::NONE, None), 7);
            frame.decode_later = true;
            let (_, prefill, _) = leased_pair(&store, &objects, frame, at);
            let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
            (prefill, sides.decode.expect("a pair's lease").session)
        };
        // The first pair claims the one decode that is ready.
        stand(&objects, &store, d1, WorkerStatus::NotReady);
        let (first, key) = pair(1, now);
        let own = sent(&first, proto::Side::Prefill, 0);
        lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
        assert_eq!(pins_of(&store, key), alone(d0), "the first claim");
        // Past the key's idle time a second request claims that decode anew.
        let (second, again) = pair(3, later);
        assert_eq!(again, key, "one key");
        assert_eq!(pins_of(&store, key), alone(d0), "the second claim");
        // The first pair's call lands on the other decode and claims it.
        stand(&objects, &store, d0, WorkerStatus::NotReady);
        stand(&objects, &store, d1, WorkerStatus::Ready);
        let call = keyed(at_call(2, first.id, WireModes::NONE, None), 7);
        let (_, issued) = leased_call(&store, &objects, call, later);
        let (side, _) = issued.expect("a pick's issue");
        assert_eq!(
            side.session_verdict,
            SessionVerdict::Claimed(d1),
            "the call's verdict"
        );
        let both = Some([d0, d1]);
        assert_eq!(pins_of(&store, key), both, "the call's claim");
        // The call's decode is sent: the second request's claim stands.
        let to = sent(&first, proto::Side::Decode, 0);
        lifecycle::dispatched(&store, &hand, &to, later).expect("the frame reads");
        assert_eq!(pins_of(&store, key), both, "the call's decode sent");
        if dispatched {
            let to = sent(&second, proto::Side::Decode, 0);
            lifecycle::dispatched(&store, &hand, &to, later).expect("the frame reads");
        } else {
            let word = proto::Released {
                what: Some(proto::released::What::LeaseId(second.id.0)),
                reason: proto::released::Reason::CommitFailed.into(),
                side: proto::Side::Unspecified.into(),
            };
            lifecycle::released(&store, &hand, &word).expect("the frame reads");
        }
        assert_eq!(
            pins_of(&store, key),
            both,
            "the second request's end, dispatched {dispatched}"
        );
    }
}

/// A second request of the key that follows the first claim of a pair is
/// placed on that claim's decode. The pair's call then claims another
/// decode, and the dispatch of the call's decode leaves the key on it
/// alone. The second request's dispatch on the decode it followed puts
/// that decode back where it stood, ahead of the call's: the key was
/// served on both. A second request that is refused was sent nowhere, and
/// the key stays on the call's decode.
#[test]
fn a_served_decode_calls_dispatch_keeps_the_decode_a_later_request_followed() {
    for dispatched in [true, false] {
        let store = Arc::new(MemoryStore::new());
        let objects = holder(&store);
        // A key that does not idle out inside the life of a pair's lease.
        let plan = ruled_pair_plan_with(|_| ());
        store.apply(plan).expect("above the unbound start");
        side_member(&store, &objects, "p0", Role::Prefill, None);
        let d0 = side_member(&store, &objects, "d0", Role::Decode, None);
        let d1 = side_member(&store, &objects, "d1", Role::Decode, None);
        LeasesStore::set_term(&*store, 1);
        let now = Instant::now();
        let later = now + Duration::from_secs(3);
        let hand = open(&store, GATEWAY);
        let alone = |id| Some([id, WorkerId(0)]);
        // A pair of the key by the number of its first frame, its decode
        // asked for later: its lease and its decode side.
        let pair = |n: u8, at: Instant| {
            let mut frame = keyed(at_prefill(n, WireModes::NONE, None), 7);
            frame.decode_later = true;
            let (_, prefill, _) = leased_pair(&store, &objects, frame, at);
            let sides = LeasesStore::sides(&*store, GATEWAY, prefill.id).expect("on file");
            (prefill, sides.decode.expect("a pair's lease"))
        };
        // The first pair claims the one decode that is ready.
        stand(&objects, &store, d1, WorkerStatus::NotReady);
        let (first, side) = pair(1, now);
        let key = side.session;
        let own = sent(&first, proto::Side::Prefill, 0);
        lifecycle::dispatched(&store, &hand, &own, now).expect("the frame reads");
        assert_eq!(pins_of(&store, key), alone(d0), "the first claim");
        // A second request of the key follows that claim.
        let (second, side) = pair(3, later);
        assert_eq!(side.session, key, "one key");
        assert_eq!(side.session_verdict, SessionVerdict::Pinned, "it follows");
        assert_eq!(pins_of(&store, key), alone(d0), "no second claim");
        // The first pair's call lands on the other decode and claims it.
        stand(&objects, &store, d0, WorkerStatus::NotReady);
        stand(&objects, &store, d1, WorkerStatus::Ready);
        let call = keyed(at_call(2, first.id, WireModes::NONE, None), 7);
        let (_, issued) = leased_call(&store, &objects, call, later);
        let (side, _) = issued.expect("a pick's issue");
        assert_eq!(
            side.session_verdict,
            SessionVerdict::Claimed(d1),
            "the call's verdict"
        );
        let both = Some([d0, d1]);
        assert_eq!(pins_of(&store, key), both, "the call's claim");
        // The call's decode is sent: the key holds it alone.
        let to = sent(&first, proto::Side::Decode, 0);
        lifecycle::dispatched(&store, &hand, &to, later).expect("the frame reads");
        assert_eq!(pins_of(&store, key), alone(d1), "the call's decode sent");
        if dispatched {
            // The second request is sent on the decode it followed.
            let to = sent(&second, proto::Side::Decode, 0);
            lifecycle::dispatched(&store, &hand, &to, later).expect("the frame reads");
            assert_eq!(pins_of(&store, key), both, "where it stood");
        } else {
            // The second request is refused: it was sent nowhere.
            let word = proto::Released {
                what: Some(proto::released::What::LeaseId(second.id.0)),
                reason: proto::released::Reason::CommitFailed.into(),
                side: proto::Side::Unspecified.into(),
            };
            lifecycle::released(&store, &hand, &word).expect("the frame reads");
            assert_eq!(pins_of(&store, key), alone(d1), "sent nowhere");
        }
    }
}
