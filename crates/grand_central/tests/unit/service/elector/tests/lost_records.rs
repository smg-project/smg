//! A record gone by no holder's release: deleted, emptied or cleared in
//! place under its holder, and how long a candidate waits it out.

use super::*;

#[test]
fn a_take_of_a_record_that_vanished_leads_a_renew_deadline_later() {
    // The wait before the take covers the holder this run saw. A record
    // taken by another and removed again inside that wait had a holder
    // this run never saw, so this take's fence opens late as well.
    let mut lease = MemoryLease::new();
    let held = held_by("x@host:8", 1, 3_000);
    assert_eq!(
        lease.write(None, &held).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    lease.lose();
    assert_eq!(b.tick(t0 + S).expect("look"), TickOutcome::Gone { term: 1 });
    let took = t0 + 4 * S;
    assert_eq!(
        b.tick(took).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    let fence = b.fence();
    assert_eq!(fence.leader_term(took), None);
    assert_eq!(
        b.tick(took + S).expect("renew"),
        TickOutcome::Renewed { term: 2 }
    );
    assert_eq!(fence.leader_term(took + S), None);
    assert_eq!(
        b.tick(took + 2 * S).expect("renew"),
        TickOutcome::Renewed { term: 2 }
    );
    assert_eq!(fence.leader_term(took + 2 * S), Some(2));
}

#[test]
fn a_replica_started_while_a_live_holders_record_is_gone_leads_after_that_holders_fence() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    // The record goes right after the holder's renewal, and the holder
    // stalls: its fence is open to its renew deadline. A replica that
    // starts now never saw the lease held and takes at once.
    lease.lose();
    let mut c = candidate(&lease, "c@host:3");
    let took = t0 + S / 10;
    assert_eq!(
        c.tick(took).expect("take"),
        TickOutcome::Acquired { term: 1 }
    );
    let (holders, takers) = (a.fence(), c.fence());
    assert_eq!(
        c.tick(took + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    // The holder's fence at its last open instant, the taker's closed.
    assert_eq!(holders.leader_term(t0 + 2 * S), Some(1));
    assert_eq!(takers.leader_term(t0 + 2 * S), None);
    // One renew deadline after the take: the taker's opens, the holder's
    // closed a tenth of a second before.
    assert_eq!(
        c.tick(took + 2 * S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(holders.leader_term(took + 2 * S), None);
    assert_eq!(takers.leader_term(took + 2 * S), Some(1));
}

#[test]
fn a_record_removed_again_before_a_waiter_saw_its_taker_has_one_leader() {
    let mut lease = MemoryLease::new();
    let held = held_by("a@host:1", 1, 3_000);
    assert_eq!(
        lease.write(None, &held).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let (mut b, mut c) = (candidate(&lease, "b@host:2"), candidate(&lease, "c@host:3"));
    let t0 = Instant::now() + AHEAD;
    for waiter in [&mut b, &mut c] {
        assert!(matches!(
            waiter.tick(t0).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
    }
    lease.lose();
    for waiter in [&mut b, &mut c] {
        assert_eq!(
            waiter.tick(t0 + S).expect("look"),
            TickOutcome::Gone { term: 1 }
        );
    }
    // Both waits end at the same instant. b takes; the record goes again
    // before c looks, so c never saw b's record and takes under the same
    // term: a wait before the take cannot see that holder.
    let took = t0 + 4 * S;
    assert_eq!(
        b.tick(took).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    lease.lose();
    let again = took + S / 2;
    assert_eq!(
        c.tick(again).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    // Neither take opened a fence, and b's first renewal reads its loss:
    // one leader under the term, a renew deadline after its own take.
    assert_eq!(b.fence().leader_term(again), None);
    assert_eq!(c.fence().leader_term(again), None);
    assert_eq!(
        b.tick(took + S).expect("the renewal"),
        TickOutcome::Lost {
            term: 2,
            holder: "c@host:3".to_string()
        }
    );
    assert_eq!(
        c.tick(again + S).expect("renew"),
        TickOutcome::Renewed { term: 2 }
    );
    assert_eq!(c.fence().leader_term(again + S), None);
    assert_eq!(
        c.tick(again + 2 * S).expect("renew"),
        TickOutcome::Renewed { term: 2 }
    );
    assert_eq!(c.fence().leader_term(again + 2 * S), Some(2));
    assert_eq!(b.fence().leader_term(again + 2 * S), None);
}

#[test]
fn a_record_that_vanished_is_waited_out_and_taken_at_the_term_after_the_highest_seen() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("lease");
    let mut a =
        Elector::new(Box::new(FileLease::new(&path)), "a@host:1", settings()).expect("settings");
    let mut b =
        Elector::new(Box::new(FileLease::new(&path)), "b@host:2", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    assert!(a.release().expect("release"));
    assert_eq!(
        b.tick(t0 + S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert!(matches!(
        a.tick(t0 + S + S / 2).expect("follow"),
        TickOutcome::Following { term: 2, .. }
    ));
    // The record is deleted under the holder (an operator forcing a
    // re-election). No release made it so, and the holder's fence is
    // open until its own next step reads the lease: a candidate that saw
    // the record held waits that record's duration from its own first
    // sight of the loss, the old holder as well, and follows nobody. The
    // take then counts from the highest term either saw, never from one.
    std::fs::remove_file(&path).expect("the deletion");
    let a_saw = t0 + S + 3 * S / 4;
    assert_eq!(a.tick(a_saw).expect("look"), TickOutcome::Gone { term: 2 });
    assert_eq!(a.observed(), None, "nobody is named");
    assert_eq!(
        b.fence().leader_term(a_saw),
        Some(2),
        "the holder still leads when the candidate first sees the loss"
    );
    assert_eq!(
        b.tick(t0 + 2 * S).expect("look"),
        TickOutcome::Lost {
            term: 2,
            holder: String::new()
        }
    );
    assert_eq!(
        b.tick(t0 + 3 * S).expect("look"),
        TickOutcome::Gone { term: 2 }
    );
    assert_eq!(
        a.tick(a_saw + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Gone { term: 2 },
        "the record's duration counts from a's first sight of the loss"
    );
    assert_eq!(
        a.tick(a_saw + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 3 }
    );
    assert!(matches!(
        b.tick(t0 + 5 * S).expect("follow"),
        TickOutcome::Following { term: 3, .. }
    ));
}

#[test]
fn an_emptied_lease_file_is_waited_out_as_a_deleted_one() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("lease");
    let mut a =
        Elector::new(Box::new(FileLease::new(&path)), "a@host:1", settings()).expect("settings");
    let mut b =
        Elector::new(Box::new(FileLease::new(&path)), "b@host:2", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    assert!(matches!(
        b.tick(t0 + S / 2).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    // The file is emptied in place: it reads as no lease, as a deleted
    // one does, and no release wrote it.
    std::fs::write(&path, b"").expect("the emptying");
    let b_saw = t0 + 3 * S / 4;
    assert_eq!(b.tick(b_saw).expect("look"), TickOutcome::Gone { term: 1 });
    assert_eq!(
        a.fence().leader_term(b_saw),
        Some(1),
        "the holder still leads when the candidate first sees the loss"
    );
    assert_eq!(
        b.tick(b_saw + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Gone { term: 1 }
    );
    assert_eq!(
        a.fence().leader_term(b_saw + 3 * S),
        None,
        "the holder's fence closed inside the wait"
    );
    assert_eq!(
        b.tick(b_saw + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
}

/// The record of `holder` with its holder cleared in place, as a hand's
/// patch leaves it: the term, the stamps and the duration kept.
fn cleared(lease: &MemoryLease, holder: &Elector) {
    let held = holder.observed().expect("the holder's record").clone();
    let cleared = LeaseRecord {
        holder: String::new(),
        ..held.clone()
    };
    let mut hand = lease.clone();
    assert_eq!(
        hand.write(Some(&held), &cleared).expect("a write"),
        ConditionalWriteOutcome::Done
    );
}

#[test]
fn a_record_cleared_in_place_under_a_live_holder_is_waited_out() {
    // A holder's release says no duration. A record whose holder another
    // hand cleared keeps its term and its duration, and that holder
    // leads until its next step reads the lease: the record is waited
    // out for the duration it says, from the first sight of it.
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    let mut b = candidate(&lease, "b@host:2");
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    cleared(&lease, &a);
    let saw = t0 + S / 2;
    assert_eq!(b.tick(saw).expect("look"), TickOutcome::Gone { term: 1 });
    assert_eq!(a.fence().leader_term(saw), Some(1));
    assert_eq!(b.fence().leader_term(saw), None);
    // The holder reads its loss at its next step and waits as every
    // candidate does.
    assert_eq!(
        a.tick(t0 + S).expect("the renewal"),
        TickOutcome::Lost {
            term: 1,
            holder: String::new()
        }
    );
    assert_eq!(a.fence().leader_term(t0 + S), None);
    assert_eq!(
        a.tick(t0 + 2 * S).expect("look"),
        TickOutcome::Gone { term: 1 }
    );
    assert_eq!(
        b.tick(saw + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Gone { term: 1 }
    );
    assert_eq!(
        b.tick(saw + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
}

#[test]
fn a_candidate_that_never_saw_the_lease_held_waits_out_a_record_cleared_in_place() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    cleared(&lease, &a);
    // A replica that starts now never saw the lease held, and the holder
    // stalls with its fence open. The record still says its duration:
    // it is waited out from this first sight, and the holder's fence has
    // closed by itself before the take.
    let mut c = candidate(&lease, "c@host:3");
    let saw = t0 + S / 2;
    assert_eq!(c.tick(saw).expect("look"), TickOutcome::Gone { term: 1 });
    assert_eq!(a.fence().leader_term(saw), Some(1));
    assert_eq!(c.fence().leader_term(saw), None);
    assert_eq!(
        c.tick(saw + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Gone { term: 1 }
    );
    assert_eq!(
        c.tick(saw + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(a.fence().leader_term(saw + 3 * S), None);
    assert_eq!(c.fence().leader_term(saw + 3 * S), Some(2));
}

#[test]
fn a_record_cleared_in_place_under_a_passed_term_is_waited_out_for_its_own_duration() {
    // The wait is the cleared record's own whatever its term. One under
    // a term this run saw pass is a late writer's, and that writer may
    // lead for the duration its record says: here two seconds past the
    // wait for the holder seen last.
    let held = held_by("x@host:9", 4, 3_000);
    let mut lease = MemoryLease::new();
    assert_eq!(
        lease.write(None, &held).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 4, .. }
    ));
    let late = LeaseRecord {
        holder: String::new(),
        ..held_by("y@host:8", 2, 5_000)
    };
    assert_eq!(
        lease.write(Some(&held), &late).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let saw = t0 + S;
    for wait in [Duration::ZERO, 3 * S, 5 * S - Duration::from_micros(1)] {
        let look = b.tick(saw + wait).expect("look");
        assert!(
            matches!(look, TickOutcome::Gone { .. }),
            "{wait:?}: {look:?}"
        );
    }
    assert_eq!(
        b.tick(saw + 5 * S).expect("take"),
        TickOutcome::Acquired { term: 5 }
    );
    // No holder's release leaves a record under a passed term, so this
    // take waits as the take of an absent record does: its fence opens
    // one renew deadline later.
    let took = saw + 5 * S;
    assert_eq!(b.fence().leader_term(took), None);
    assert_eq!(
        b.tick(took + 2 * S).expect("renew"),
        TickOutcome::Renewed { term: 5 }
    );
    assert_eq!(b.fence().leader_term(took + 2 * S), Some(5));
}

#[test]
fn a_lease_replaced_by_an_empty_one_is_waited_out() {
    // A `Lease` replaced by a manifest with no record reads as released
    // under term zero. No release lowers its holder's term, so one under a
    // term below the record last seen held is no holder's release.
    let held = held_by("x@host:9", 4, 3_000);
    let mut lease = MemoryLease::new();
    assert_eq!(
        lease.write(None, &held).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 4, .. }
    ));
    let empty = LeaseRecord::default();
    assert_eq!(
        lease.write(Some(&held), &empty).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let saw = t0 + S;
    assert_eq!(b.tick(saw).expect("look"), TickOutcome::Gone { term: 4 });
    assert_eq!(
        b.tick(saw + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Gone { term: 4 }
    );
    assert_eq!(
        b.tick(saw + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 5 }
    );
    // An empty record has no holder to wait for either: the take opens
    // no fence, as over an absent one.
    assert_eq!(b.fence().leader_term(saw + 3 * S), None);
    // The taker's own record, replaced again before its next look, is
    // one it saw held: the wait counts anew from that look.
    let own = b.observed().expect("the take's record").clone();
    assert_eq!(
        lease.write(Some(&own), &empty).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let again = saw + 4 * S;
    assert_eq!(
        b.tick(again).expect("look"),
        TickOutcome::Lost {
            term: 5,
            holder: String::new()
        }
    );
    assert_eq!(
        b.tick(again + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Gone { term: 5 }
    );
    assert_eq!(
        b.tick(again + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 6 }
    );
}

#[test]
fn a_candidate_that_never_saw_the_lease_held_takes_an_empty_one_at_once() {
    // A first start over the empty `Lease` a chart renders, or over a
    // new lease file: there is no record to wait out.
    let mut lease = MemoryLease::new();
    assert_eq!(
        lease.write(None, &LeaseRecord::default()).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
}

#[test]
fn a_release_at_the_held_term_is_taken_at_once_by_a_run_that_saw_a_higher_term() {
    // The rule is keyed on the record last seen held, not on the highest
    // term: a store put back to a lower term, then released by the holder
    // it names, shows that holder's own release.
    let mut lease = MemoryLease::new();
    let (first, fallen) = (held_by("x@host:8", 5, 3_000), held_by("y@host:9", 3, 3_000));
    assert_eq!(
        lease.write(None, &first).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 5, .. }
    ));
    assert_eq!(
        lease.write(Some(&first), &fallen).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    assert!(matches!(
        b.tick(t0 + S).expect("follow"),
        TickOutcome::Following { term: 3, .. }
    ));
    let released = LeaseRecord {
        holder: String::new(),
        duration_ms: 0,
        renewed_at_micros: 2,
        ..fallen.clone()
    };
    assert_eq!(
        lease.write(Some(&fallen), &released).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    assert_eq!(
        b.tick(t0 + S + Duration::from_millis(1)).expect("take"),
        TickOutcome::Acquired { term: 6 },
        "at once, above the highest term this run saw"
    );
}

#[test]
fn a_held_record_under_a_lower_term_is_no_loss_and_its_own_loss_is_waited_from_that_sight() {
    // A store put back to a held record under a lower term shows a
    // holder, not a loss: the wait for the record lost after it is dated
    // at the first sight of that loss, never at the sight of the holder.
    let mut lease = MemoryLease::new();
    let (first, fallen) = (held_by("x@host:8", 5, 3_000), held_by("y@host:9", 3, 3_000));
    assert_eq!(
        lease.write(None, &first).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 5, .. }
    ));
    assert_eq!(
        lease.write(Some(&first), &fallen).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    assert!(matches!(
        b.tick(t0 + S).expect("follow"),
        TickOutcome::Following { term: 3, .. }
    ));
    lease.lose();
    let saw = t0 + 2 * S;
    assert_eq!(b.tick(saw).expect("look"), TickOutcome::Gone { term: 3 });
    assert_eq!(
        b.tick(saw + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Gone { term: 3 },
        "the duration counts from the first sight of the loss"
    );
    assert_eq!(
        b.tick(saw + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 6 },
        "above the highest term this run saw"
    );
}

#[test]
fn a_lost_record_is_waited_out_for_its_own_duration_not_the_waiters() {
    // Two replicas may run two durations. The wait covers the lost
    // holder's fence, so it is that record's duration, longer or shorter
    // than the three seconds this replica runs.
    let mut lease = MemoryLease::new();
    let long = held_by("x@host:8", 1, 6_000);
    assert_eq!(
        lease.write(None, &long).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    lease.lose();
    let saw = t0 + S;
    assert_eq!(b.tick(saw).expect("look"), TickOutcome::Gone { term: 1 });
    assert_eq!(
        b.tick(saw + 6 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Gone { term: 1 },
        "six seconds, the lost record's, where this replica runs three"
    );
    assert_eq!(
        b.tick(saw + 6 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    // A holder with a one second lease writes over this replica's
    // record, and that record is lost in turn.
    let own = b.observed().expect("the take's record").clone();
    let short = held_by("z@host:9", 3, 1_000);
    assert_eq!(
        lease.write(Some(&own), &short).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let lost = saw + 7 * S;
    assert!(matches!(
        b.tick(lost).expect("look"),
        TickOutcome::Lost { term: 2, .. }
    ));
    lease.lose();
    let again = lost + S / 2;
    assert_eq!(b.tick(again).expect("look"), TickOutcome::Gone { term: 3 });
    assert_eq!(
        b.tick(again + S - Duration::from_micros(1)).expect("look"),
        TickOutcome::Gone { term: 3 },
        "one second, the lost record's"
    );
    assert_eq!(
        b.tick(again + S).expect("take"),
        TickOutcome::Acquired { term: 4 }
    );
}

#[test]
fn a_record_taken_while_it_was_gone_ends_the_wait_and_is_followed() {
    // A replica that starts while the record is gone never saw it held
    // and takes at once; a candidate waiting the old record out follows
    // the new holder as it follows any.
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(
        a.tick(t0).expect("acquire"),
        TickOutcome::Acquired { term: 1 }
    );
    assert!(matches!(
        b.tick(t0 + S / 2).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    lease.lose();
    assert_eq!(b.tick(t0 + S).expect("look"), TickOutcome::Gone { term: 1 });
    let mut c = candidate(&lease, "c@host:3");
    assert_eq!(
        c.tick(t0 + S + S / 2).expect("take"),
        TickOutcome::Acquired { term: 1 }
    );
    assert_eq!(
        b.tick(t0 + 2 * S).expect("follow"),
        TickOutcome::Following {
            holder: "c@host:3".to_string(),
            term: 1
        }
    );
    // The new record is lost in turn: the wait for it counts from b's
    // first sight of this loss, never from the one before it.
    lease.lose();
    let again = t0 + 2 * S + S / 2;
    assert_eq!(b.tick(again).expect("look"), TickOutcome::Gone { term: 1 });
    assert_eq!(
        b.tick(again + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Gone { term: 1 }
    );
    assert_eq!(
        b.tick(again + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
}

#[test]
fn a_record_under_this_identity_that_came_back_lower_is_taken_above_the_highest_term() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("lease");
    let mut a =
        Elector::new(Box::new(FileLease::new(&path)), "a@host:1", settings()).expect("settings");
    let mut b =
        Elector::new(Box::new(FileLease::new(&path)), "b@host:2", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    assert!(a.release().expect("release"));
    assert_eq!(
        b.tick(t0 + S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert!(matches!(
        a.tick(t0 + S + S / 2).expect("follow"),
        TickOutcome::Following { term: 2, .. }
    ));
    // The record is deleted, and a second process under a's identity (a
    // restart racing the first) takes the empty lease at term one and
    // stalls. The first process saw term two: once that record has been
    // still for its duration it takes above the highest term it saw, a
    // take in every field, never one under the lower term. Its next
    // take after a deletion is above that again, once it waited its own
    // lost record out as every candidate that saw it does.
    std::fs::remove_file(&path).expect("the deletion");
    let mut again =
        Elector::new(Box::new(FileLease::new(&path)), "a@host:1", settings()).expect("settings");
    assert_eq!(
        again.tick(t0 + 2 * S).expect("take"),
        TickOutcome::Acquired { term: 1 }
    );
    let first_sight = t0 + 2 * S + S / 2;
    assert!(matches!(
        a.tick(first_sight).expect("look"),
        TickOutcome::Following { term: 1, .. }
    ));
    let twins = a.observed().expect("the twin's record").clone();
    assert_eq!(
        a.tick(first_sight + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 3 }
    );
    let taken = a.observed().expect("the take's record").clone();
    assert_eq!(taken.acquired_at_micros, taken.renewed_at_micros);
    assert!(taken.acquired_at_micros > twins.acquired_at_micros);
    std::fs::remove_file(&path).expect("the second deletion");
    assert_eq!(
        a.tick(first_sight + 4 * S).expect("look"),
        TickOutcome::Lost {
            term: 3,
            holder: String::new()
        }
    );
    assert_eq!(
        a.tick(first_sight + 7 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Gone { term: 3 }
    );
    assert_eq!(
        a.tick(first_sight + 7 * S).expect("take"),
        TickOutcome::Acquired { term: 4 }
    );
}
