//! A pair's lease: its record of two sides, a chosen decode side and the
//! later call that issues it, the dispatch deadline over a pair's sides,
//! and what one side books, how it is dispatched, ended and told.

use super::*;

#[test]
fn a_pairs_record_stands_until_its_decode_side_completes() {
    let store = MemoryStore::new();
    let leases: &dyn LeasesStore = &store;
    let now = Instant::now();
    let gateway = GatewayId(1);
    let pair = lease(gateway, 5, 0, now);
    let decode = decode_side(WorkerId(8));
    assert_eq!(leases.issue_pair(pair, decode), Ok(()));
    let mut outcome = Outcome {
        side: Side::Prefill,
        served: WorkerId(7),
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    assert!(leases.completed(gateway, LeaseId(5), &outcome));
    assert!(
        leases.get(gateway, LeaseId(5)).is_some(),
        "the room lives on"
    );
    outcome.side = Side::Decode;
    assert!(leases.completed(gateway, LeaseId(5), &outcome));
    assert!(leases.get(gateway, LeaseId(5)).is_none());
}

/// A pair's record states no word of a request kept from its profile: the
/// door that files a pair clears it, as it writes the pair's mark, so both
/// sides of a pair are told whatever a caller wrote. A single's record
/// keeps the word as it was filed.
#[test]
fn the_door_that_files_a_pair_clears_the_word_of_a_kept_request() {
    let store = MemoryStore::new();
    let [p, d] = ["p", "d"].map(|name| ready(&store, &format!("http://{name}:1")));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let kept = LeaseRecord {
        untold: true,
        ..booking(gateway, 1, p, now)
    };
    leases.issue(kept).expect("free");
    let (record, decode) = pair(2, p, &[d], now);
    let kept = LeaseRecord {
        untold: true,
        ..record
    };
    issue_pair(leases, (kept, decode)).expect("free");
    let word = |id| leases.get(gateway, LeaseId(id)).expect("on file").untold;
    assert_eq!((word(1), word(2)), (true, false));
}

/// Where each side of the pair's lease under `id` stands.
fn sides(store: &MemoryStore, id: u64) -> Option<(LeaseState, LeaseState)> {
    let found = LeasesStore::sides(store, GatewayId(1), LeaseId(id))?;
    Some((found.record.state, found.decode?.state))
}

#[test]
fn a_chosen_decode_side_books_nothing_and_waits_under_no_deadline() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, id, now) = (GatewayId(1), LeaseId(1), Instant::now());
    leases.set_term(1);
    let (record, decode) = chosen_pair(1, p, &[d], now);
    issue_pair(leases, (record, decode)).expect("the key is free");
    assert_eq!(booked(&store, &[p, d]), [1, 0]);
    assert!(decode.state.is_chosen());
    assert!(!decode.state.is_live());
    assert!(!decode.state.is_issued());
    assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), now));
    assert!(leases.expire(now + Duration::from_secs(9)).is_empty());
    assert_eq!(store.queued(), 0);
    assert_eq!(
        sides(&store, 1),
        Some((LeaseState::Dispatched { at: now }, LeaseState::Chosen))
    );
}

#[test]
fn a_call_issues_the_chosen_side_afresh_and_books_it_once() {
    let store = MemoryStore::new().with_decision_slots(2);
    let [p, d0, d1] = ["http://p:1", "http://d0:1", "http://d1:1"].map(|key| ready(&store, key));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let (_, mut decode) = pair(0, p, &[d1, d0], now);
    decode.session = Session::from_raw(42);
    decode.tokens = 24;
    decode.target_index = 1;

    let id = LeaseId(1);
    issue_pair(leases, chosen_pair(id.0, p, &[d0], now)).expect("the key is free");
    assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), now));
    let prefill = leases.get(gateway, id);
    assert_eq!(leases.issue_decode(gateway, id, decode, now), Ok(()));
    let filed = leases.sides(gateway, id).expect("on file");
    assert_eq!(Some(filed.record), prefill);
    assert_eq!(
        filed.decode,
        Some(DecodeSide {
            target_index: 0,
            ..decode
        })
    );
    assert_eq!(filed.target_of(Side::Decode, 0), Some((d1, 0)));
    assert_eq!(filed.target_of(Side::Decode, 1), Some((d0, 0)));
    assert_eq!(booked(&store, &[p, d0, d1]), [1, 0, 1]);
    assert!(leases.released(gateway, id, None, ReleaseReason::ClientGone));

    // A lease of an older term in block one names the second file.
    let id = LeaseId(1 << 20 | 2);
    assert_eq!(store.file_of(id), 1);
    issue_pair(leases, chosen_pair(id.0, p, &[d0], now)).expect("the key is free");
    assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), now));
    let mut heard = None;
    assert!(store.unlocked());
    let (issued, _) = store
        .issue_decided_decode(gateway, id, (decode, None), now, |lease| {
            assert!(!store.unlocked(), "the issue is told under the turn");
            heard = Some(*lease);
        })
        .expect("the prefill holds the chosen side");
    assert_eq!(heard, Some(issued));
    assert_eq!(
        (
            issued.id,
            issued.side,
            issued.booked,
            issued.target_index,
            issued.session,
            issued.tokens
        ),
        (id, Side::Decode, d1, 0, decode.session, decode.tokens)
    );
    assert_eq!(
        booked(&store, &[p, d0, d1]),
        [1, 0, 0],
        "no slot was supplied"
    );
}

/// A pair's decode side keeps its own session verdict on file, beside the
/// prefill side's on the record's lease, and the side's lease states the
/// side's own. A dispatch that moves the side to another target leaves the
/// verdict as it was: a claim names the worker claimed at the issue. A
/// call's issue files the call's verdict with its side.
#[test]
fn a_decode_side_keeps_its_own_session_verdict_on_file() {
    let store = MemoryStore::new();
    let [p, d0, d1] = ["http://p:1", "http://d0:1", "http://d1:1"].map(|key| ready(&store, key));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let claimed = |id| Some(SessionVerdict::Claimed(id));
    // What the file holds of the lease `id`: the side's verdict, then the
    // one each side's lease states.
    let verdicts = |id: u64| {
        let filed = leases.sides(gateway, LeaseId(id)).expect("on file");
        let stated = |side| filed.side(side).map(|(lease, _)| lease.session_verdict);
        let kept = filed.decode.map(|side| side.session_verdict);
        [kept, stated(Side::Prefill), stated(Side::Decode)]
    };
    let (mut record, mut decode) = pair(1, p, &[d0, d1], now);
    record.lease.session_verdict = SessionVerdict::Claimed(p);
    decode.session_verdict = SessionVerdict::Claimed(d0);
    issue_pair(leases, (record, decode)).expect("the key is free");
    let own = [claimed(d0), claimed(p), claimed(d0)];
    assert_eq!(verdicts(1), own);
    assert!(leases.dispatched(gateway, LeaseId(1), on(Side::Decode, 1), now));
    let moved = leases.sides(gateway, LeaseId(1)).expect("on file").decode;
    assert_eq!(moved.map(|side| side.target), Some(d1));
    assert_eq!(verdicts(1), own, "the worker claimed at the issue");

    let (mut record, mut chosen) = chosen_pair(2, p, &[d0], now);
    record.lease.session_verdict = SessionVerdict::Claimed(p);
    chosen.session_verdict = SessionVerdict::Claimed(d0);
    issue_pair(leases, (record, chosen)).expect("the key is free");
    assert!(leases.dispatched(gateway, LeaseId(2), on(Side::Prefill, 0), now));
    let call = DecodeSide {
        session_verdict: SessionVerdict::Pinned,
        ..decode_side(d0)
    };
    assert_eq!(leases.issue_decode(gateway, LeaseId(2), call, now), Ok(()));
    let pinned = Some(SessionVerdict::Pinned);
    assert_eq!(verdicts(2), [pinned, claimed(p), pinned]);
}

#[test]
fn the_call_starts_the_dispatch_deadline_again() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, id, t0) = (GatewayId(1), LeaseId(1), Instant::now());
    leases.set_term(1);
    issue_pair(leases, chosen_pair(1, p, &[d], t0)).expect("the key is free");
    assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), t0));
    let pending = DecodeSide {
        state: LeaseState::Chosen,
        ..decode_side(d)
    };
    let now = t0 + Duration::from_secs(8);
    assert_eq!(leases.issue_decode(gateway, id, pending, now), Ok(()));
    assert_eq!(
        sides(&store, 1),
        Some((LeaseState::Dispatched { at: t0 }, LeaseState::ISSUED))
    );
    assert!(leases.expire(t0 + Duration::from_secs(9)).is_empty());
    assert_eq!(store.queued(), 1);
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    let swept = leases.expire(t0 + Duration::from_secs(14));
    let filed = leases.sides(gateway, id).expect("the prefill still stands");
    assert_eq!(swept, [filed]);
    assert_eq!(filed.record.state, LeaseState::Dispatched { at: t0 });
    assert_eq!(
        filed.decode.map(|side| side.state),
        Some(LeaseState::Released(ReleaseReason::Abandoned))
    );
    assert_eq!(booked(&store, &[p, d]), [1, 0]);
}

#[test]
fn the_call_is_refused_by_name() {
    use DecodeRefused::{DecodeStands, NoLease, NotChosen, PrefillNotHeld};

    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let (mut ended, decode) = chosen_pair(4, p, &[d], now);
    ended.state = LeaseState::Completed;
    issue_pair(leases, (ended, decode)).expect("the key is free");
    leases
        .issue(booking(gateway, 1, p, now))
        .expect("the key is free");
    issue_pair(leases, pair(2, p, &[d], now)).expect("the key is free");
    for id in [3, 5, 6] {
        issue_pair(leases, chosen_pair(id, p, &[d], now)).expect("the key is free");
    }
    for id in [2, 5, 6] {
        assert!(leases.dispatched(gateway, LeaseId(id), on(Side::Prefill, 0), now));
    }
    let decode = decode_side(d);
    assert_eq!(
        leases.issue_decode(gateway, LeaseId(5), decode, now),
        Ok(())
    );
    assert!(leases.dispatched(gateway, LeaseId(6), on(Side::Decode, 0), now));
    let refused = |owner, id, reason| {
        let before = (
            leases.sides(gateway, id),
            booked(&store, &[p, d]),
            store.queued(),
        );
        let later = now + Duration::from_secs(8);
        assert_eq!(
            leases.issue_decode(owner, id, decode, later),
            Err(reason),
            "{owner:?} {id:?}"
        );
        assert_eq!(
            (
                leases.sides(gateway, id),
                booked(&store, &[p, d]),
                store.queued()
            ),
            before
        );
    };
    refused(GatewayId(2), LeaseId(5), NoLease);
    refused(gateway, LeaseId(1), NotChosen);
    refused(gateway, LeaseId(2), NotChosen);
    refused(gateway, LeaseId(3), PrefillNotHeld);
    refused(gateway, LeaseId(4), PrefillNotHeld);
    refused(gateway, LeaseId(5), DecodeStands);
    refused(gateway, LeaseId(6), DecodeStands);
    for id in [2, 5, 6] {
        assert!(leases.completed(gateway, LeaseId(id), &done(Side::Decode)));
    }
    refused(gateway, LeaseId(2), NotChosen);
    refused(gateway, LeaseId(5), DecodeStands);
    refused(gateway, LeaseId(6), DecodeStands);
    assert!(leases.completed(gateway, LeaseId(3), &done(Side::Prefill)));
    refused(gateway, LeaseId(3), NoLease);
}

#[test]
fn a_dispatch_on_a_chosen_side_books_it_once_as_it_lands() {
    let store = MemoryStore::new();
    let [p, d0, d1] = ["http://p:1", "http://d0:1", "http://d1:1"].map(|key| ready(&store, key));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    for (index, target, counts) in [(1, d1, [1, 0, 1]), (0, d0, [1, 1, 0])] {
        let id = LeaseId(u64::from(index) + 1);
        let (record, decode) = chosen_pair(id.0, p, &[d0, d1], now);
        issue_pair(leases, (record, decode)).expect("the key is free");
        assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), now));
        assert!(leases.expire(now + DISPATCH_TTL).is_empty());
        assert_eq!(store.queued(), 0);
        let at = now + Duration::from_secs(8);
        assert!(leases.dispatched(gateway, id, on(Side::Decode, index), at));
        let filed = leases.sides(gateway, id).expect("on file");
        assert_eq!(
            filed.decode,
            Some(DecodeSide {
                target,
                target_index: index,
                state: LeaseState::Dispatched { at },
                ..decode
            })
        );
        assert_eq!(booked(&store, &[p, d0, d1]), counts);
        assert_eq!(store.queued(), 0);
        // A second word, on the pick's other target, moves nothing.
        assert!(leases.dispatched(gateway, id, on(Side::Decode, 1 - index), now));
        assert_eq!(leases.sides(gateway, id), Some(filed));
        assert_eq!(booked(&store, &[p, d0, d1]), counts);
        assert!(leases.released(gateway, id, None, ReleaseReason::ClientGone));
        assert_eq!(booked(&store, &[p, d0, d1]), [0, 0, 0]);
    }
}

/// A dispatch at an index the pick holds no target at names no worker. A
/// chosen side takes nothing of it: the side stays chosen on its pick's
/// first decode and books none, and the door still answers that the lease
/// is on file. Behind the word a call issues the side, a dispatch that
/// names a target books that target, and the side's release frees no
/// count. The pick is as wide as a record keeps one, four targets, so the
/// first index past it is the first that no record holds.
#[test]
fn a_dispatch_at_an_index_its_pick_holds_no_target_at_leaves_a_chosen_side_as_it_stood() {
    let store = MemoryStore::new();
    let keys = ["p", "d0", "d1", "d2", "d3"].map(|name| format!("http://{name}:1"));
    let workers = keys.map(|key| ready(&store, &key));
    let [p, d0, d1, d2, d3] = workers;
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let at = now + Duration::from_secs(8);
    // A pair of four targets with its prefill sent, and the word said past
    // them: at the first index the pick does not hold, and at a byte's
    // last.
    let past_its_pick = |id: u64| {
        let id = LeaseId(id);
        let (record, decode) = chosen_pair(id.0, p, &[d0, d1, d2, d3], now);
        issue_pair(leases, (record, decode)).expect("the key is free");
        assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), now));
        let filed = leases.sides(gateway, id);
        for index in [4, u8::MAX] {
            assert!(leases.dispatched(gateway, id, on(Side::Decode, index), at));
        }
        assert_eq!(leases.sides(gateway, id), filed);
        assert_eq!(filed.and_then(|sides| sides.decode), Some(decode));
        assert_eq!(booked(&store, &workers), [1, 0, 0, 0, 0]);
        id
    };

    // A call behind the word issues the side, on its own pick's first
    // decode.
    let id = past_its_pick(1);
    let (_, called) = pair(id.0, p, &[d1, d0], now);
    assert_eq!(leases.issue_decode(gateway, id, called, now), Ok(()));
    assert_eq!(booked(&store, &workers), [1, 0, 1, 0, 0]);
    assert!(leases.released(gateway, id, None, ReleaseReason::ClientGone));

    // A dispatch behind it that names a target books that target: here
    // the pick's last.
    let id = past_its_pick(2);
    assert!(leases.dispatched(gateway, id, on(Side::Decode, 3), at));
    let sent = (
        LeaseState::Dispatched { at: now },
        LeaseState::Dispatched { at },
    );
    assert_eq!(sides(&store, id.0), Some(sent));
    assert_eq!(booked(&store, &workers), [1, 0, 0, 0, 1]);
    assert!(leases.released(gateway, id, None, ReleaseReason::ClientGone));

    // A release of the side behind it ends the side and frees no count.
    let id = past_its_pick(3);
    assert!(leases.released(gateway, id, Some(Side::Decode), ReleaseReason::ClientGone));
    let gone = LeaseState::Released(ReleaseReason::ClientGone);
    let ended = (LeaseState::Dispatched { at: now }, gone);
    assert_eq!(sides(&store, id.0), Some(ended));
    assert_eq!(booked(&store, &workers), [1, 0, 0, 0, 0]);
    assert!(leases.completed(gateway, id, &done(Side::Prefill)));
    assert_eq!(booked(&store, &workers), [0, 0, 0, 0, 0]);
}

#[test]
fn a_chosen_side_ends_with_its_prefill() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, id, now) = (GatewayId(1), LeaseId(1), Instant::now());
    leases.set_term(1);
    issue_pair(leases, chosen_pair(1, p, &[d], now)).expect("the key is free");
    assert!(leases.dispatched(gateway, id, on(Side::Prefill, 0), now));
    // The gateway gives the chosen side up: its end stands and no call
    // issues it.
    assert!(leases.released(gateway, id, Some(Side::Decode), ReleaseReason::ClientGone));
    let gone = LeaseState::Released(ReleaseReason::ClientGone);
    assert_eq!(
        sides(&store, 1),
        Some((LeaseState::Dispatched { at: now }, gone))
    );
    let decode = chosen_pair(1, p, &[d], now).1;
    assert_eq!(
        leases.issue_decode(gateway, id, decode, now),
        Err(DecodeRefused::DecodeStands)
    );
    assert_eq!(booked(&store, &[p, d]), [1, 0]);
    assert!(leases.completed(gateway, id, &done(Side::Prefill)));
    assert_eq!(leases.sides(gateway, id), None);
    assert_eq!(store.paired(), [vec![], vec![]]);
    assert_eq!(booked(&store, &[p, d]), [0, 0]);
}

#[test]
fn a_chosen_side_dispatched_by_a_handed_frame_is_told_once() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (id, now) = (LeaseId(1), Instant::now());
    leases.set_term(1);
    store.tell_through(heard);
    issue_pair(leases, chosen_pair(1, p, &[d], now)).expect("the key is free");
    drop(told_sides());
    let hand = Hand::new(GatewayId(1));
    let sent = |side| Said::Dispatched(on(side, 0), now);
    store.hand(&hand, id, sent(Side::Prefill));
    store.hand(&hand, id, sent(Side::Decode));
    store.hand(&hand, id, sent(Side::Decode));
    // Each side's dispatch is told once, with the lease as its frame found
    // it: the decode side stood chosen.
    let told: Vec<_> = told_sides()
        .into_iter()
        .map(|(found, said)| (found.decode.map(|side| side.state), said))
        .collect();
    let chosen = Some(LeaseState::Chosen);
    assert_eq!(
        told,
        [(chosen, sent(Side::Prefill)), (chosen, sent(Side::Decode))]
    );
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
}

#[test]
fn a_release_of_every_side_frees_a_chosen_side_s_place() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, id, now) = (GatewayId(1), LeaseId(1), Instant::now());
    leases.set_term(1);
    issue_pair(leases, chosen_pair(1, p, &[d], now)).expect("the key is free");
    assert!(leases.released(gateway, id, None, ReleaseReason::ClientGone));
    assert_eq!(leases.sides(gateway, id), None);
    assert_eq!(store.paired(), [vec![], vec![]]);
    assert_eq!(booked(&store, &[p, d]), [0, 0]);
}

#[test]
fn the_sweep_s_answer_tells_a_decode_side_that_completed_from_one_it_abandoned() {
    let store = MemoryStore::new();
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    let (p, d) = (WorkerId(7), WorkerId(8));
    leases.set_term(1);
    store.tell_through(heard);
    drop(told_sides());
    for id in [1, 2] {
        issue_pair(leases, pair(id, p, &[d], now)).expect("the key is free");
    }
    assert!(leases.dispatched(gateway, LeaseId(1), on(Side::Decode, 0), now));
    assert!(leases.completed(gateway, LeaseId(1), &done(Side::Decode)));
    let expired = leases.expire(now + Duration::from_secs(2));
    let states = expired.iter().map(|sides| {
        (
            sides.record.lease.id,
            sides.record.state,
            sides.decode.map(|side| side.state),
        )
    });
    let abandoned = LeaseState::Released(ReleaseReason::Abandoned);
    assert_eq!(
        states.collect::<Vec<_>>(),
        [
            (LeaseId(1), abandoned, Some(LeaseState::Completed)),
            (LeaseId(2), abandoned, Some(abandoned)),
        ]
    );
    assert_eq!(leases.live(gateway), 0);
    let told = |id, side| {
        (
            LeaseId(id),
            Said::Released(Some(side), ReleaseReason::Abandoned),
        )
    };
    let ends = told_sides()
        .into_iter()
        .map(|(sides, said)| (sides.record.lease.id, said));
    assert_eq!(
        ends.collect::<Vec<_>>(),
        [
            told(1, Side::Prefill),
            told(2, Side::Prefill),
            told(2, Side::Decode)
        ]
    );
}

#[cfg(feature = "controller")]
#[test]
fn a_pair_s_decode_side_is_told_issued_on_the_decode_worker() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    store.set_term(1);
    let (mut record, mut decode) = pair(0, p, &[d], Instant::now());
    record.lease.booked = p;
    decode.session = Session::from_raw(41);
    decode.tokens = 24;
    let world = store.snapshot();
    let slots = [world.booked_of(p), world.booked_of(d)];
    let heard = RefCell::new(Vec::new());
    let prefill = store
        .issue_decided_pair(record, decode, slots, |lease| {
            heard.borrow_mut().push(*lease);
        })
        .expect("minted under the store's term");
    let decode = Lease {
        side: Side::Decode,
        booked: d,
        session: decode.session,
        tokens: decode.tokens,
        ..prefill
    };
    assert_eq!(heard.into_inner(), [prefill, decode]);
}

#[test]
fn a_pair_with_both_sides_waiting_takes_one_place_of_the_queue() {
    let store = MemoryStore::new();
    store.set_term(1);
    let (p, d) = (WorkerId(7), WorkerId(8));
    issue_pair(&store, pair(1, p, &[d], Instant::now())).expect("the key is free");
    assert_eq!(store.queued(), 1);
}

/// A pair filed with a side that ended books the side that lives, and
/// no count for the one that ended.
#[test]
fn a_pair_filed_with_an_ended_side_books_the_side_that_lives() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let ended = LeaseState::Completed;
    let (record, side) = pair(1, p, &[d], now);
    let half = DecodeSide {
        state: ended,
        ..side
    };
    leases.issue_pair(record, half).expect("the key is free");
    assert_eq!(booked(&store, &[p, d]), [1, 0]);
    let (mut half, side) = pair(2, p, &[d], now);
    half.state = ended;
    leases.issue_pair(half, side).expect("the key is free");
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    assert_eq!(leases.live(gateway), 2);
}

/// The store tells of a lease with the worker its record books. A single's
/// lease and a pair's prefill side name the record's target, a decode side
/// its own, and the lease on file names none.
#[test]
fn a_side_s_lease_names_the_worker_its_record_books() {
    let now = Instant::now();
    let (p, d) = (WorkerId(3), WorkerId(5));
    let booked = |lease: Option<Lease>| lease.map(|lease| lease.booked);
    let told = |sides: &Sides, side| booked(sides.side(side).map(|(lease, _)| lease));
    let record = lease(GatewayId(1), 1, 1, now);
    let single = Sides {
        record,
        decode: None,
    };
    assert_eq!(record.lease.booked, WorkerId(0), "filed with no word");
    assert_eq!(told(&single, Side::Single), Some(record.target));
    // A single's end is lent off its record, under the same worker.
    let whole = Said::Released(None, ReleaseReason::ClientGone);
    let lent = single.with_held_single(whole, |held| held.lease.booked);
    assert_eq!(lent, Some(record.target));
    let (record, decode) = pair(2, p, &[d], now);
    let both = Sides {
        record,
        decode: Some(decode),
    };
    assert_eq!(told(&both, Side::Prefill), Some(p));
    assert_eq!(told(&both, Side::Decode), Some(d));
    // A pair's end is lent once a side, the prefill's first: the lease
    // names the prefill, and the target's pair the decode.
    let mut ended = Vec::new();
    both.with_held_pair(whole, |held, said| {
        let decode = held.target.pair.map(|pair| pair.decode_id);
        ended.push((said, held.lease.booked, decode));
    });
    let of = |side| Said::Released(Some(side), ReleaseReason::ClientGone);
    let sides = [Side::Prefill, Side::Decode].map(|side| (of(side), p, Some(d)));
    assert_eq!(ended, sides);
}

#[test]
fn a_pairs_sides_are_dispatched_and_ended_each_by_its_own_frame() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, key, now) = (GatewayId(1), LeaseId(1), Instant::now());
    leases.set_term(1);
    issue_pair(leases, pair(1, p, &[d], now)).expect("the key is free");
    // One record, and each side books its own worker.
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    assert_eq!(leases.live(gateway), 1);
    // A side is dispatched by the frame that names it, and no other is.
    let at = now + Duration::from_millis(3);
    let dispatched = LeaseState::Dispatched { at };
    assert!(leases.dispatched(gateway, key, on(Side::Decode, 0), at));
    assert_eq!(sides(&store, 1), Some((LeaseState::ISSUED, dispatched)));
    assert!(leases.dispatched(gateway, key, on(Side::Prefill, 0), at));
    assert_eq!(sides(&store, 1), Some((dispatched, dispatched)));
    // A frame on a side the pair does not hold changes nothing.
    assert!(leases.dispatched(gateway, key, on(Side::Single, 0), now));
    assert!(leases.completed(gateway, key, &done(Side::Encode)));
    let gone = ReleaseReason::ClientGone;
    assert!(leases.released(gateway, key, Some(Side::Single), gone));
    assert_eq!(sides(&store, 1), Some((dispatched, dispatched)));
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    // The prefill's completion frees its worker's count, and the decode's
    // stands; a second completion of that side changes nothing.
    for _ in 0..2 {
        assert!(leases.completed(gateway, key, &done(Side::Prefill)));
        assert_eq!(booked(&store, &[p, d]), [0, 1]);
        assert_eq!(sides(&store, 1), Some((LeaseState::Completed, dispatched)));
        assert_eq!(leases.live(gateway), 1);
    }
    // The record leaves the file at its last side's end.
    assert!(leases.completed(gateway, key, &done(Side::Decode)));
    assert_eq!(booked(&store, &[p, d]), [0, 0]);
    assert_eq!(leases.get(gateway, key), None);
    assert!(!leases.completed(gateway, key, &done(Side::Decode)));
}

#[test]
fn a_decodes_dispatch_on_an_alternate_moves_the_decodes_booking_alone() {
    let store = MemoryStore::new();
    let [p, d0, d1] = ["http://p:1", "http://d0:1", "http://d1:1"].map(|key| ready(&store, key));
    let leases: &dyn LeasesStore = &store;
    let (gateway, key, now) = (GatewayId(1), LeaseId(1), Instant::now());
    leases.set_term(1);
    issue_pair(leases, pair(1, p, &[d0, d1], now)).expect("the key is free");
    assert_eq!(booked(&store, &[p, d0, d1]), [1, 1, 0]);
    // Every target of a pair's pick names the one prefill: its dispatch on
    // the alternate's index moves no booking.
    assert!(leases.dispatched(gateway, key, on(Side::Prefill, 1), now));
    assert_eq!(booked(&store, &[p, d0, d1]), [1, 1, 0]);
    assert!(leases.dispatched(gateway, key, on(Side::Decode, 1), now));
    assert_eq!(booked(&store, &[p, d0, d1]), [1, 0, 1]);
    let filed = leases.sides(gateway, key).expect("on file");
    let record = filed.record;
    assert_eq!((record.target, record.lease.target_index), (p, 0));
    let decode = filed.decode.expect("a pair's lease");
    assert_eq!((decode.target, decode.target_index), (d1, 1));
    assert_eq!(filed.target_of(Side::Decode, 1), Some((d1, 0)));
    // The decode's end frees the count of the worker it moved to.
    assert!(leases.completed(gateway, key, &done(Side::Decode)));
    assert_eq!(booked(&store, &[p, d0, d1]), [1, 0, 0]);
}

#[test]
fn a_release_ends_the_side_it_names_and_with_no_side_every_side() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now, gone) = (GatewayId(1), Instant::now(), ReleaseReason::ClientGone);
    leases.set_term(1);
    issue_pair(leases, pair(1, p, &[d], now)).expect("the key is free");
    // One side given up: its count is free and the other side stands.
    assert!(leases.released(gateway, LeaseId(1), Some(Side::Decode), gone));
    assert_eq!(booked(&store, &[p, d]), [1, 0]);
    let released = LeaseState::Released(gone);
    assert_eq!(sides(&store, 1), Some((LeaseState::ISSUED, released)));
    // The last side's release takes the record off the file.
    assert!(leases.released(gateway, LeaseId(1), Some(Side::Prefill), gone));
    assert_eq!(leases.get(gateway, LeaseId(1)), None);
    assert_eq!(booked(&store, &[p, d]), [0, 0]);
    // A release that names no side ends every side at once.
    issue_pair(leases, pair(2, p, &[d], now)).expect("the key is free");
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    assert!(leases.released(gateway, LeaseId(2), None, gone));
    assert_eq!(leases.get(gateway, LeaseId(2)), None);
    assert_eq!(booked(&store, &[p, d]), [0, 0]);
    assert_eq!(leases.live(gateway), 0);
}

/// A pair filed with its prefill part dispatched and its decode side
/// waiting is asked at its deadline all the same: the side stands behind
/// its prefill, and the lease's deadline is written one span on.
#[test]
fn a_pair_filed_with_its_decode_side_alone_waiting_is_asked_at_its_deadline() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    let dispatched = LeaseState::Dispatched { at: now };
    let (mut record, decode) = pair(1, p, &[d], now);
    record.state = dispatched;
    issue_pair(leases, (record, decode)).expect("the key is free");
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    // An issue past the deadline asks the lease and ends nothing of it.
    let later = now + Duration::from_secs(2);
    let world = store.snapshot();
    store
        .issue_decided(booking(gateway, 0, p, later), world.booked_of(p), |_| ())
        .expect("minted under the store's term");
    assert_eq!(sides(&store, 1), Some((dispatched, LeaseState::ISSUED)));
    assert_eq!(booked(&store, &[p, d]), [2, 1]);
    let filed = leases.get(gateway, LeaseId(1)).expect("the lease stands");
    assert_eq!(filed.dispatch_deadline, later + DISPATCH_TTL);
}

/// A place let go is the next one laid: a file's list of places is as
/// long as the most pairs that stood at once.
#[test]
fn the_list_of_places_is_as_long_as_the_most_pairs_that_stood_at_once() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    for id in 1..=3 {
        issue_pair(leases, pair(id, p, &[d], now)).expect("the key is free");
    }
    assert_eq!(store.places(), 3);
    // Each pair that ends leaves its place to the next one filed.
    for id in 4..=9 {
        assert!(leases.released(gateway, LeaseId(id - 3), None, ReleaseReason::ClientGone));
        issue_pair(leases, pair(id, p, &[d], now)).expect("the key is free");
        assert_eq!(store.places(), 3, "pair {id}");
    }
}

#[test]
fn the_dispatch_deadline_ends_each_side_that_waits_and_a_dispatched_side_stands() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    // Two pairs: the first's decode is dispatched, the second waits whole.
    for id in [1, 2] {
        issue_pair(leases, pair(id, p, &[d], now)).expect("the key is free");
    }
    assert!(leases.dispatched(gateway, LeaseId(1), on(Side::Decode, 0), now));
    assert_eq!(booked(&store, &[p, d]), [2, 2]);
    // An issue past the deadline ends what waits and tells of each side
    // it ended, the prefill's first.
    let later = now + Duration::from_secs(2);
    let world = store.snapshot();
    store.tell_through(heard);
    drop(told_sides());
    store
        .issue_decided(booking(gateway, 0, p, later), world.booked_of(p), |_| ())
        .expect("minted under the store's term");
    let ended = told_sides().into_iter().map(|(found, said)| {
        let Said::Released(side, ReleaseReason::Abandoned) = said else {
            panic!("this issue tells the deadline's ends alone: {said:?}");
        };
        // The word is lent off the lease as the end found it: once, of
        // the side it names, which waited.
        let mut lent = Vec::new();
        found.with_held_pair(said, |_, said| lent.push(said));
        assert_eq!(lent, [said]);
        (found.record.lease.id.0, side)
    });
    assert_eq!(
        ended.collect::<Vec<_>>(),
        [
            (1, Some(Side::Prefill)),
            (2, Some(Side::Prefill)),
            (2, Some(Side::Decode)),
        ]
    );
    let abandoned = LeaseState::Released(ReleaseReason::Abandoned);
    let dispatched = LeaseState::Dispatched { at: now };
    // The dispatched side stands with its count; the pair that waited
    // whole left the file.
    assert_eq!(sides(&store, 1), Some((abandoned, dispatched)));
    assert_eq!(leases.get(gateway, LeaseId(2)), None);
    assert_eq!(booked(&store, &[p, d]), [1, 1]);
    // The side that stands ends by its own frame.
    assert!(leases.completed(gateway, LeaseId(1), &done(Side::Decode)));
    assert_eq!(leases.get(gateway, LeaseId(1)), None);
    assert_eq!(booked(&store, &[p, d]), [1, 0]);
    // The store's own sweep returns both sides as it left them.
    issue_pair(leases, pair(3, p, &[d], later)).expect("the key is free");
    assert!(leases.dispatched(gateway, LeaseId(3), on(Side::Decode, 0), later));
    let swept = leases.expire(later + Duration::from_secs(9));
    let third =
        |found: &Sides| (found.record.lease.id, found.record.state) == (LeaseId(3), abandoned);
    assert!(swept.iter().any(third));
    let at = LeaseState::Dispatched { at: later };
    assert_eq!(sides(&store, 3), Some((abandoned, at)));
}

/// A pair's sides are dispatched one by one. A decode side that was issued
/// with its prefill stands at the dispatch deadline while that prefill is
/// dispatched, at a later issue and at the store's own sweep: no end is
/// told of it, it keeps its count, and its own dispatch finds it booked
/// however long the prefill ran.
#[test]
fn the_dispatch_deadline_ends_no_decode_side_that_waits_behind_its_dispatched_prefill() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    // The first pair's sides are dispatched one by one, the second's
    // prefill alone.
    for id in [1, 2] {
        issue_pair(leases, pair(id, p, &[d], now)).expect("the key is free");
    }
    let dispatched = LeaseState::Dispatched { at: now };
    assert!(leases.dispatched(gateway, LeaseId(1), on(Side::Prefill, 0), now));
    assert_eq!(sides(&store, 1), Some((dispatched, LeaseState::ISSUED)));
    assert!(leases.dispatched(gateway, LeaseId(1), on(Side::Decode, 0), now));
    assert_eq!(sides(&store, 1), Some((dispatched, dispatched)));
    assert!(leases.dispatched(gateway, LeaseId(2), on(Side::Prefill, 0), now));
    // An issue past the deadline ends nothing and tells nothing: the
    // decode side that waits stands behind its prefill, with its count.
    let later = now + Duration::from_secs(2);
    let world = store.snapshot();
    store.tell_through(heard);
    drop(told_sides());
    store
        .issue_decided(booking(gateway, 0, p, later), world.booked_of(p), |_| ())
        .expect("minted under the store's term");
    assert!(told_sides().is_empty(), "no end is told");
    assert_eq!(sides(&store, 1), Some((dispatched, dispatched)));
    assert_eq!(sides(&store, 2), Some((dispatched, LeaseState::ISSUED)));
    assert_eq!(booked(&store, &[p, d]), [3, 2]);
    // The lease keeps one place of the queue, a span behind that pass.
    let deadline = |id| {
        leases
            .get(gateway, LeaseId(id))
            .map(|filed| filed.dispatch_deadline)
    };
    assert_eq!(deadline(2), Some(later + DISPATCH_TTL));
    // The store's own sweep asks it at that deadline and holds it too:
    // of the three leases that wait, it ends the single's alone.
    let swept = leases.expire(later + DISPATCH_TTL);
    let ended: Vec<_> = swept.iter().map(|found| found.record.target).collect();
    assert_eq!(ended, [p], "the single's lease, issued at the first pass");
    assert_eq!(sides(&store, 2), Some((dispatched, LeaseState::ISSUED)));
    assert_eq!(deadline(2), Some(later + DISPATCH_TTL * 2));
    assert_eq!(booked(&store, &[p, d]), [2, 2]);
    // The decode is named a minute behind the pair's issue: it is booked.
    let sent = now + Duration::from_secs(60);
    assert!(leases.dispatched(gateway, LeaseId(2), on(Side::Decode, 0), sent));
    let decode_sent = LeaseState::Dispatched { at: sent };
    assert_eq!(sides(&store, 2), Some((dispatched, decode_sent)));
    assert_eq!(booked(&store, &[p, d]), [2, 2]);
    assert!(leases.expire(sent + DISPATCH_TTL).is_empty());
}

/// A decode side that was issued with its prefill has a whole deadline
/// behind that prefill's end. A deadline that falls due between the
/// prefill's end and the decode's dispatch a moment later ends nothing:
/// that pass starts the side's own deadline, and the dispatch finds the
/// side booked. A side that still waits at the end of its own deadline is
/// released as abandoned.
#[test]
fn a_decode_side_has_a_whole_deadline_behind_its_prefills_end() {
    let store = MemoryStore::new();
    let (p, d) = (ready(&store, "http://p:1"), ready(&store, "http://d:1"));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    // Two pairs sent prefill first: each prefill is dispatched and has
    // answered inside the lease's first deadline, a second from the issue.
    for id in [1, 2] {
        issue_pair(leases, pair(id, p, &[d], now)).expect("the key is free");
        assert!(leases.dispatched(gateway, LeaseId(id), on(Side::Prefill, 0), now));
        assert!(leases.completed(gateway, LeaseId(id), &done(Side::Prefill)));
    }
    let waiting = Some((LeaseState::Completed, LeaseState::ISSUED));
    assert_eq!((sides(&store, 1), sides(&store, 2)), (waiting, waiting));
    assert_eq!(booked(&store, &[p, d]), [0, 2]);
    // The first deadline falls due behind the prefills' ends: the pass
    // ends nothing, tells nothing and starts each side a whole deadline.
    let first = now + Duration::from_secs(1);
    store.tell_through(heard);
    drop(told_sides());
    assert!(leases.expire(first).is_empty());
    assert!(told_sides().is_empty(), "no end is told");
    assert_eq!((sides(&store, 1), sides(&store, 2)), (waiting, waiting));
    let deadline = |id| {
        leases
            .get(gateway, LeaseId(id))
            .map(|filed| filed.dispatch_deadline)
    };
    assert_eq!(deadline(1), Some(first + DISPATCH_TTL));
    assert_eq!(deadline(2), Some(first + DISPATCH_TTL));
    // The first pair's decode is named a moment behind that pass: the
    // side is booked on its worker, and no later deadline ends it.
    let sent = first + Duration::from_millis(1);
    assert!(leases.dispatched(gateway, LeaseId(1), on(Side::Decode, 0), sent));
    let decode_sent = LeaseState::Dispatched { at: sent };
    assert_eq!(sides(&store, 1), Some((LeaseState::Completed, decode_sent)));
    assert_eq!(booked(&store, &[p, d]), [0, 2]);
    drop(told_sides());
    // The second pair's decode has that one deadline and no second: a
    // moment inside it the side waits, and at its end the side is
    // released as abandoned and told so, and its lease leaves the file.
    let own = first + DISPATCH_TTL;
    assert!(leases.expire(own - Duration::from_millis(1)).is_empty());
    assert_eq!(sides(&store, 2), waiting);
    let swept = leases.expire(own);
    let abandoned = LeaseState::Released(ReleaseReason::Abandoned);
    let second = |found: &Sides| {
        (found.record.lease.id, found.record.state) == (LeaseId(2), LeaseState::Completed)
            && found.decode.is_some_and(|decode| decode.state == abandoned)
    };
    assert_eq!(swept.len(), 1);
    assert!(swept.iter().all(second));
    let told: Vec<_> = told_sides().into_iter().map(|(_, said)| said).collect();
    assert_eq!(
        told,
        [Said::Released(Some(Side::Decode), ReleaseReason::Abandoned)]
    );
    assert_eq!(leases.get(gateway, LeaseId(2)), None);
    assert_eq!(sides(&store, 1), Some((LeaseState::Completed, decode_sent)));
    assert_eq!(booked(&store, &[p, d]), [0, 1]);
}

/// The pair a decode call reads ahead of its decision is the lease's own:
/// its prefill's worker, then the worker and the rank of the decode its
/// first decision chose for the target the prefill's dispatch took, the
/// pick's first or the one that dispatch named. A side a call issued and a
/// take-back put back names the first decision's decode again. The read is
/// refused under the name its issue is refused under, it gives the file's
/// turn back with its answer, and it moves nothing.
///
/// Last in the answer comes the instant the record's own lease states it
/// was issued at.
#[cfg(feature = "controller")]
#[test]
fn the_held_pair_names_the_prefill_and_the_decode_of_the_target_its_dispatch_took() {
    use DecodeRefused::{DecodeStands, NoLease, NotChosen, PrefillNotHeld};

    let store = MemoryStore::new();
    let keys = ["http://p:1", "http://d0:1", "http://d1:1", "http://d2:1"];
    let [p, d0, d1, d2] = keys.map(|key| ready(&store, key));
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    // A pick of three targets on the one prefill, each decode at a rank of
    // its own: the pair filed with its decode side chosen where `later`.
    let filed = |id: u64, later: bool| {
        let (mut record, mut decode) = pair(id, p, &[d0, d1, d2], now);
        let mut targets = Targets::new();
        for (decode_id, decode_dp_rank) in [(d0, 2), (d1, 3), (d2, 5)] {
            let pair = decision::Pair {
                decode_id,
                decode_dp_rank,
                ..Default::default()
            };
            targets.push(Target {
                id: p,
                pair: Some(pair),
                ..Target::default()
            });
        }
        (record.picked, decode.picked) =
            (PickRecord::of(&targets), PickRecord::of_decodes(&targets));
        if later {
            decode.state = LeaseState::Chosen;
        }
        // Each lease is dated by its id, a thousand microseconds an id.
        record.lease.issued_micros = id * 1_000;
        issue_pair(&store, (record, decode)).expect("the key is free");
    };
    for id in 1..=3 {
        filed(id, true);
    }
    filed(4, false);
    store
        .issue(booking(gateway, 5, p, now))
        .expect("the key is free");
    let file = || {
        let sides = [1, 2, 3, 4, 5].map(|id| LeasesStore::sides(&store, gateway, LeaseId(id)));
        (sides, booked(&store, &[p, d0, d1, d2]), store.queued())
    };
    let held = |owner: GatewayId, id: u64| {
        let before = file();
        let held = store.held_pair(owner, LeaseId(id));
        assert!(store.unlocked(), "the turn is given back");
        assert_eq!(file(), before, "the read moves nothing");
        held.map(|(prefill, decode, _, issued)| (prefill, decode, issued))
    };
    // No prefill was sent yet.
    assert_eq!(held(gateway, 1), Err(PrefillNotHeld));
    // The prefill's dispatch names the pick's first target, then another.
    assert!(store.dispatched(gateway, LeaseId(1), on(Side::Prefill, 0), now));
    assert!(store.dispatched(gateway, LeaseId(2), on(Side::Prefill, 2), now));
    assert!(store.dispatched(gateway, LeaseId(4), on(Side::Prefill, 0), now));
    assert_eq!(held(gateway, 1), Ok((p, Some((d0, 2)), 1_000)));
    assert_eq!(held(gateway, 2), Ok((p, Some((d2, 5)), 2_000)));
    // Each refusal by its name: another gateway's id, a prefill not sent,
    // a pair filed whole and a single's lease.
    assert_eq!(held(GatewayId(2), 1), Err(NoLease));
    assert_eq!(held(gateway, 3), Err(PrefillNotHeld));
    assert_eq!(held(gateway, 4), Err(NotChosen));
    assert_eq!(held(gateway, 5), Err(NotChosen));
    // A call issues the side on another decode: the side stands, and the
    // read is refused as the next issue is.
    let moved = DecodeSide {
        picked: PickRecord::default(),
        ..decode_side(d1)
    };
    assert_eq!(store.issue_decode(gateway, LeaseId(2), moved, now), Ok(()));
    assert_eq!(held(gateway, 2), Err(DecodeStands));
    assert_eq!(
        store.issue_decode(gateway, LeaseId(2), moved, now),
        Err(DecodeStands)
    );
    // Taken back, the side names the first decision's decode again, at the
    // target the prefill's dispatch took.
    assert!(store.take_back_decode(gateway, LeaseId(2), ReleaseReason::Abandoned));
    assert_eq!(held(gateway, 2), Ok((p, Some((d2, 5)), 2_000)));
}

/// An undo takes back its own issue of a decode side and no other. Where a
/// take-back ended that issue, and where a later call issued the side
/// since, the first issue's undo moves nothing and frees nothing: the place
/// counts the side's issues, and an undo names its own. The undo of the
/// issue that stands puts the side back as that issue found it and frees
/// its booking. Behind the prefill's end it takes the lease off the file:
/// the side would stand chosen beside a prefill that ended, and no later
/// event ends a side that is not live.
#[cfg(feature = "controller")]
#[test]
fn an_undo_takes_back_its_own_issue_of_a_decode_side_and_no_other() {
    let store = MemoryStore::new();
    let [p, d0, d1] = ["http://p:1", "http://d0:1", "http://d1:1"].map(|key| ready(&store, key));
    let (gateway, id, now) = (GatewayId(1), LeaseId(1), Instant::now());
    store.set_term(1);
    issue_pair(&store, chosen_pair(id.0, p, &[d0, d1], now)).expect("the key is free");
    assert!(store.dispatched(gateway, id, on(Side::Prefill, 0), now));
    let file = || {
        let sides = LeasesStore::sides(&store, gateway, id);
        (sides, booked(&store, &[p, d0, d1]))
    };
    let chosen = file();
    assert_eq!(chosen.1, [1, 0, 0]);
    // One call's issue of the side on the second decode, with the slot of
    // the count it books.
    let call = || {
        let slot = store.snapshot().booked_of(d1);
        let issued = store.issue_decided_decode(gateway, id, (decode_side(d1), slot), now, |_| ());
        issued.expect("the prefill holds the chosen side").1
    };
    let first = call();
    assert_eq!(file().1, [1, 0, 1]);
    assert!(store.take_back_decode(gateway, id, ReleaseReason::Abandoned));
    assert_eq!(file(), chosen);
    assert!(!store.unissue_decode(gateway, id, first), "its issue ended");
    assert_eq!(file(), chosen);
    let second = call();
    let issued = file();
    assert_eq!(issued.1, [1, 0, 1]);
    assert!(!store.unissue_decode(gateway, id, first), "another's issue");
    assert_eq!(file(), issued);
    assert!(store.unissue_decode(gateway, id, second));
    assert_eq!(file(), chosen);
    assert!(!store.unissue_decode(gateway, id, second), "undone once");
    // Behind the prefill's end.
    let third = call();
    let gone = ReleaseReason::ClientGone;
    assert!(store.released(gateway, id, Some(Side::Prefill), gone));
    assert!(file().0.is_some(), "the issued decode stands");
    assert!(store.unissue_decode(gateway, id, third));
    assert_eq!(file(), (None, vec![0, 0, 0]));
}
