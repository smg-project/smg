//! The election's transitions over a store that answers every write.
//! Its cases cover a take and the wait before it, a renewal, a release,
//! a restart, and a holder told a term to leave under.

use super::*;

#[test]
fn an_elector_alone_over_its_own_lease_leads_at_its_take() {
    // Nothing can hold a store that was handed to nobody: no holder
    // this run never saw may lead, and the fence opens at the take.
    let mut alone = Elector::alone("a@host:1", settings()).expect("valid settings");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(
        alone.tick(t0).expect("tick"),
        TickOutcome::Acquired { term: 1 }
    );
    assert_eq!(alone.fence().leader_term(t0), Some(1));
    // A store that is handed in keeps the wait, a memory one included:
    // its clones share it.
    let mut handed = candidate(&MemoryLease::new(), "b@host:2");
    assert_eq!(
        handed.tick(t0).expect("tick"),
        TickOutcome::Acquired { term: 1 }
    );
    assert_eq!(handed.fence().leader_term(t0), None);
}

#[test]
fn a_candidate_takes_an_absent_lease_at_term_one_and_leads_a_renew_deadline_later() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("tick"), TickOutcome::Acquired { term: 1 });
    assert!(a.is_holder());
    let record = a.observed().expect("the record").clone();
    assert_eq!(record.holder, "a@host:1");
    assert_eq!(record.duration_ms, 3_000);
    assert_eq!(record.acquired_at_micros, record.renewed_at_micros);
    assert!(record.acquired_at_micros > 0);
    // No record stood, so nobody was waited for, and a holder this run
    // never saw may lead to its renew deadline: the fence stays closed.
    let fence = a.fence();
    assert_eq!(fence.leader_term(t0), None);
    assert_eq!(fence.term(), 1, "the term is on the frames from the take");
    // The take alone carries the fence nowhere: its own renew deadline
    // ends where the lead would open.
    assert_eq!(fence.leader_term(t0 + 2 * S), None, "no renewal landed");
    assert_eq!(
        a.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(fence.leader_term(t0 + S), None, "inside the renew deadline");
    // The clock alone turns that answer twice from here: at the
    // opening still ahead, then at the window's end, and never after.
    assert_eq!(fence.turns_at(t0 + S), Some(t0 + 2 * S));
    assert_eq!(fence.turns_at(t0 + 2 * S), Some(t0 + 3 * S));
    assert_eq!(fence.turns_at(t0 + 3 * S), Some(t0 + 3 * S));
    let past = t0 + 3 * S + Duration::from_micros(1);
    assert_eq!(fence.turns_at(past), None);
    // The pair by one read is the two reads at one instant.
    for at in [t0 + S, t0 + 2 * S, t0 + 3 * S, past] {
        assert_eq!(
            fence.term_and_turn(at),
            (fence.leader_term(at), fence.turns_at(at))
        );
    }
    // One renew deadline after the take the fence opens by itself, at
    // no step: the renewal behind it carries the window past that
    // instant, to its own renew deadline.
    assert_eq!(
        fence.leader_term(t0 + 2 * S - Duration::from_micros(1)),
        None,
        "the instant before the opening"
    );
    assert_eq!(fence.leader_term(t0 + 2 * S), Some(1));
    assert_eq!(fence.leader_term(t0 + 3 * S), Some(1), "at the deadline");
    assert_eq!(
        fence.leader_term(t0 + 3 * S + Duration::from_micros(1)),
        None,
        "past the renew deadline the fence closes by itself"
    );
    // A renewal moves the end and never the opening: a reader whose
    // instant is behind the renewal's still reads the lead it is under.
    assert_eq!(
        a.tick(t0 + 3 * S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(fence.leader_term(t0 + 2 * S), Some(1));
    assert_eq!(fence.leader_term(t0 + 5 * S), Some(1), "at the deadline");
    assert_eq!(
        fence.leader_term(t0 + 5 * S + Duration::from_micros(1)),
        None
    );
}

#[test]
fn a_candidate_takes_an_empty_record_and_leads_a_renew_deadline_later() {
    // A `Lease` made with no record, or replaced by such a manifest
    // under a live holder, reads as released under term zero. No holder
    // wrote that: no release lowers its term and a take writes one or
    // more. A candidate that never saw the lease held takes it at once,
    // and a holder it never saw may lead to its renew deadline: the
    // fence stays closed that long, as over no record.
    let mut lease = MemoryLease::new();
    let empty = LeaseRecord::default();
    assert_eq!(
        lease.write(None, &empty).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("tick"), TickOutcome::Acquired { term: 1 });
    let fence = a.fence();
    assert_eq!(fence.leader_term(t0), None);
    assert_eq!(
        a.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(fence.leader_term(t0 + S), None, "inside the renew deadline");
    // The first renewal one renew deadline after the take opens it.
    assert_eq!(
        a.tick(t0 + 2 * S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(fence.leader_term(t0 + 2 * S), Some(1));
}

#[test]
fn a_follower_times_the_record_from_its_own_first_sight() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    a.tick(t0).expect("acquire");
    // b first sees the record half a second after it was written; the
    // record's clock says nothing to b.
    let t1 = t0 + S / 2;
    assert_eq!(
        b.tick(t1).expect("tick"),
        TickOutcome::Following {
            holder: "a@host:1".to_string(),
            term: 1
        }
    );
    assert!(matches!(
        b.tick(t0 + 3 * S).expect("tick"),
        TickOutcome::Following { .. }
    ));
    assert!(matches!(
        b.tick(t1 + 3 * S - Duration::from_micros(1)).expect("tick"),
        TickOutcome::Following { .. }
    ));
    assert_eq!(
        b.tick(t1 + 3 * S).expect("tick"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(b.fence().leader_term(t1 + 3 * S), Some(2));
}

#[test]
fn a_stalled_holders_renewal_is_refused_by_the_moved_record() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    a.tick(t0).expect("acquire");
    b.tick(t0).expect("follow");
    // a stalls past the duration; b takes the lease.
    assert_eq!(
        b.tick(t0 + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(
        a.fence().leader_term(t0 + 3 * S),
        None,
        "a fenced itself at 2 s"
    );
    assert_eq!(
        a.tick(t0 + 3 * S + S / 2).expect("the stalled renewal"),
        TickOutcome::Lost {
            term: 1,
            holder: "b@host:2".to_string()
        }
    );
    assert!(!a.is_holder());
    assert_eq!(a.fence().term(), 1, "the lost term stays on a's frames");
    assert_eq!(
        a.tick(t0 + 4 * S + S / 2).expect("tick"),
        TickOutcome::Following {
            holder: "b@host:2".to_string(),
            term: 2
        }
    );
    assert_eq!(
        b.tick(t0 + 4 * S).expect("renew"),
        TickOutcome::Renewed { term: 2 }
    );
}

#[test]
fn a_renewing_holder_keeps_its_lease_against_a_waiting_candidate() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    for i in 1..=10u32 {
        let now = t0 + i * S;
        assert_eq!(
            a.tick(now).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert!(matches!(
            b.tick(now).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        assert_eq!(a.fence().leader_term(now), Some(1));
        assert_eq!(b.fence().leader_term(now), None);
    }
}

#[test]
fn a_release_inside_the_takes_wait_leaves_the_wait_to_the_next_candidate() {
    // b took a record with no holder to wait for and gives it up before
    // its fence opened. A holder b never saw may still lead to its renew
    // deadline, and b's wait was all that covered it: the release says
    // that deadline as its duration, and the next candidate waits it out
    // from its first sight of the release.
    let lease = MemoryLease::new();
    let mut b = candidate(&lease, "b@host:2");
    let mut c = candidate(&lease, "c@host:3");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(b.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    assert!(matches!(
        c.tick(t0).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    assert!(b.release().expect("release"));
    let released = b.observed().expect("the release").clone();
    assert!(released.is_released());
    assert_eq!(released.duration_ms, 2_000);
    let saw = t0 + Duration::from_millis(1);
    assert_eq!(c.tick(saw).expect("look"), TickOutcome::Gone { term: 1 });
    assert_eq!(
        c.tick(saw + 2 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Gone { term: 1 }
    );
    assert_eq!(
        c.tick(saw + 2 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(c.fence().leader_term(saw + 2 * S), Some(2));
}

#[test]
fn a_releaser_waits_out_its_own_release_from_the_writes_return() {
    // b's take is handed an instant from before its read, so its own
    // sight of the record is dated there and its wait at the read's
    // return. Its release inside that wait is dated at the write's
    // return: b's own next take waits out what it handed on, as every
    // candidate does, and never the read's length short of it.
    let lease = MemoryLease::new();
    let mut b = candidate(&lease, "b@host:2");
    let before = Instant::now();
    thread::sleep(Duration::from_millis(20));
    assert_eq!(
        b.tick(before).expect("take"),
        TickOutcome::Acquired { term: 1 }
    );
    assert!(b.release().expect("release"));
    let released = Instant::now();
    assert_eq!(
        b.tick(before + 2 * S).expect("look"),
        TickOutcome::Gone { term: 1 }
    );
    assert_eq!(b.fence().leader_term(before + 2 * S), None);
    assert_eq!(
        b.tick(released + 2 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(b.fence().leader_term(released + 2 * S), Some(2));
}

#[test]
fn a_taker_whose_lease_went_waits_for_nothing() {
    // The wait belongs to the take. A loss ends it, and so does a
    // release: the reader says no from then on.
    let mut lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    assert!(a.waits_to_lead());
    let own = a.observed().expect("the take's record").clone();
    let other = held_by("x@host:8", 2, 3_000);
    assert_eq!(
        lease.write(Some(&own), &other).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    assert!(matches!(
        a.tick(t0 + S).expect("look"),
        TickOutcome::Lost { term: 1, .. }
    ));
    assert!(!a.waits_to_lead());
    let mut b = candidate(&MemoryLease::new(), "b@host:2");
    assert_eq!(b.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    assert!(b.waits_to_lead());
    assert!(b.release().expect("release"));
    assert!(!b.waits_to_lead());
}

#[test]
fn a_released_lease_is_taken_at_once_with_the_next_term() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let mut b = candidate(&lease, "b@host:2");
    // The holder leads before it releases: its release is a leader's
    // word, and the next candidate leads at its take.
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { .. }
    ));
    assert!(a.release().expect("release"));
    assert!(!a.is_holder());
    assert_eq!(
        a.fence().leader_term(t0),
        None,
        "released: the fence closes now"
    );
    assert!(!a.release().expect("a second release"), "nothing held");
    let record = b.observed().expect("b's last read");
    assert_eq!(record.holder, "a@host:1", "b has not looked again yet");
    assert_eq!(
        b.tick(t0 + Duration::from_millis(1)).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(
        b.fence().leader_term(t0 + Duration::from_millis(1)),
        Some(2),
        "a holder's release is led at once"
    );
    assert!(
        a.observed().expect("a's last write").is_released(),
        "a's last sight of the record is its own release"
    );
}

#[test]
fn a_restart_under_the_same_identity_takes_the_term_above() {
    let lease = MemoryLease::new();
    let t0 = Instant::now() + AHEAD;
    let acquired_at = {
        let mut a = candidate(&lease, "a@host:1");
        a.tick(t0).expect("acquire");
        a.observed().expect("the record").acquired_at_micros
    };
    // The process restarts: a fresh elector under the same identity. It
    // takes its earlier run's record once the record stood still for its
    // duration from its first sight, the wait of every candidate: a live
    // writer under the same name would have moved it by then. The take
    // is above the record's term: the earlier run led under that one,
    // and a term is taken once.
    let mut again = candidate(&lease, "a@host:1");
    let first_sight = t0 + S / 10;
    assert_eq!(
        again.tick(first_sight).expect("look"),
        TickOutcome::Following {
            holder: "a@host:1".to_string(),
            term: 1
        },
        "not yet: the earlier run may still be writing"
    );
    assert!(matches!(
        again
            .tick(first_sight + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Following { .. }
    ));
    assert_eq!(
        again.tick(first_sight + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    let record = again.observed().expect("the record");
    assert_eq!(record.term, 2);
    assert!(
        record.acquired_at_micros > acquired_at,
        "a take in every field"
    );
    assert_eq!(record.acquired_at_micros, record.renewed_at_micros);
    assert_eq!(again.fence().leader_term(first_sight + 3 * S), Some(2));
}

#[test]
fn a_restart_takes_only_once_the_stalled_earlier_runs_fence_is_closed() {
    let lease = MemoryLease::new();
    let t0 = Instant::now() + AHEAD;
    let mut earlier = candidate(&lease, "a@host:1");
    earlier.lead_at(t0);
    assert_eq!(
        earlier.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    // The earlier run stalls with its fence open to the renew deadline,
    // and a new run under the same identity starts at once. A take
    // after a renew period of stillness would open a second fence beside
    // it; the new run waits for the record's duration instead.
    let mut again = candidate(&lease, "a@host:1");
    let first_sight = t0 + S + S / 10;
    assert!(matches!(
        again.tick(first_sight).expect("look"),
        TickOutcome::Following { .. }
    ));
    let deadline = t0 + S + 2 * S;
    assert_eq!(earlier.fence().leader_term(deadline), Some(1));
    assert!(matches!(
        again.tick(deadline).expect("look"),
        TickOutcome::Following { .. }
    ));
    assert_eq!(again.fence().leader_term(deadline), None);
    assert!(matches!(
        again
            .tick(first_sight + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Following { .. }
    ));
    assert_eq!(
        again.tick(first_sight + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(earlier.fence().leader_term(first_sight + 3 * S), None);
    assert_eq!(again.fence().leader_term(first_sight + 3 * S), Some(2));
}

#[test]
fn terms_only_grow_across_takes_and_releases() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let mut b = candidate(&lease, "b@host:2");
    let mut c = candidate(&lease, "c@host:3");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    a.release().expect("release");
    assert_eq!(
        b.tick(t0 + S).expect("tick"),
        TickOutcome::Acquired { term: 2 }
    );
    c.tick(t0 + S).expect("c sees b");
    assert_eq!(
        c.tick(t0 + 4 * S).expect("b expired for c"),
        TickOutcome::Acquired { term: 3 }
    );
    c.release().expect("release");
    assert_eq!(
        a.tick(t0 + 5 * S).expect("tick"),
        TickOutcome::Acquired { term: 4 }
    );
    assert_eq!(
        b.tick(t0 + 5 * S).expect("b's stalled renewal"),
        TickOutcome::Lost {
            term: 2,
            holder: "a@host:1".to_string()
        }
    );
}

#[test]
fn a_release_past_the_takes_opening_hands_on_no_wait() {
    // The wait ends at its instant by the clock, whatever the steps
    // since. One renew deadline after the take's write no holder this
    // run never saw leads, so a release from then on says no duration:
    // the next candidate takes at once and leads at its take.
    let settings = ElectionSettings {
        duration: Duration::from_millis(60),
        renew_period: Duration::from_millis(10),
        renew_deadline: Duration::from_millis(40),
    };
    let lease = MemoryLease::new();
    let mut b = Elector::new(Box::new(lease.clone()), "b@host:2", settings).expect("settings");
    let take = b.tick(Instant::now()).expect("take");
    assert_eq!(take, TickOutcome::Acquired { term: 1 });
    assert!(b.waits_to_lead());
    // No step since the take: the clock alone ends the wait.
    thread::sleep(settings.renew_deadline);
    assert!(!b.waits_to_lead(), "the wait ended at its instant");
    assert!(b.release().expect("release"));
    let released = b.observed().expect("the release").clone();
    assert!(released.is_released());
    assert_eq!(released.duration_ms, 0);
    let mut c = Elector::new(Box::new(lease.clone()), "c@host:3", settings).expect("settings");
    let now = Instant::now();
    assert_eq!(
        c.tick(now).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(c.fence().leader_term(now), Some(2));
}

#[test]
fn a_holder_told_to_leave_releases_and_takes_nothing_for_the_duration_its_record_said() {
    // The hold is one lease duration, the one the holder's own record
    // said, from the word. Nobody else looks here, so the lease stands
    // free through it and the replica that left takes it back under the
    // next term.
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    let word = t0 + S / 2;
    assert!(a.step_down(None, word).expect("the release"));
    assert!(!a.is_holder());
    assert_eq!(a.fence().leader_term(word), None, "closed at the word");
    let stored = lease.clone().read().expect("read").expect("the record");
    assert_eq!(
        (stored.holder.as_str(), stored.term, stored.duration_ms),
        ("", 1, 0),
        "a holder's release under its own term"
    );
    for wait in [Duration::ZERO, S, 3 * S - Duration::from_micros(1)] {
        assert_eq!(
            a.tick(word + wait).expect("look"),
            TickOutcome::Waiting { term: 1 },
            "{wait:?}"
        );
    }
    assert_eq!(
        a.tick(word + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(a.fence().leader_term(word + 3 * S), Some(2));
    // The take ends the hold: the holder renews from here.
    assert_eq!(
        a.tick(word + 4 * S).expect("renew"),
        TickOutcome::Renewed { term: 2 }
    );
}

#[test]
fn a_replica_whose_hold_ended_over_a_release_that_still_says_a_duration_reads_it_gone() {
    // The waiting answer ends with the hold. A release another replica
    // wrote inside the hold says a duration that outlasts it: from the
    // hold's end this replica is a candidate as any other, and a record
    // with no holder and a duration is the gone record's answer again.
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    let word = t0 + S / 2;
    assert!(a.step_down(None, word).expect("the release"));
    let own = lease.clone().read().expect("read").expect("the record");
    let later = LeaseRecord {
        term: 2,
        renewed_at_micros: own.renewed_at_micros + 1,
        duration_ms: 60_000,
        ..own.clone()
    };
    assert_eq!(
        lease.clone().write(Some(&own), &later).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    assert_eq!(
        a.tick(word + S).expect("look"),
        TickOutcome::Waiting { term: 2 }
    );
    assert_eq!(
        a.tick(word + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Waiting { term: 2 }
    );
    assert_eq!(
        a.tick(word + 3 * S).expect("look"),
        TickOutcome::Gone { term: 2 }
    );
    assert!(!a.is_holder());
}

#[test]
fn a_standby_takes_a_stepped_down_holders_lease_at_its_next_look() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    let mut b = candidate(&lease, "b@host:2");
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    let word = t0 + S / 2;
    assert!(a.step_down(None, word).expect("the release"));
    // A holder's release: taken at once and led at once.
    assert_eq!(
        b.tick(word).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(b.fence().leader_term(word), Some(2));
    // The replica that left follows the taker, and still holds its own
    // takes back when the taker stops inside the hold.
    assert!(matches!(
        a.tick(word + S).expect("follow"),
        TickOutcome::Following { term: 2, .. }
    ));
    assert!(b.release().expect("the taker's stop"));
    assert_eq!(
        a.tick(word + 2 * S).expect("look"),
        TickOutcome::Waiting { term: 2 }
    );
    assert_eq!(
        a.tick(word + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 3 }
    );
}

#[test]
fn a_term_told_to_a_holder_goes_into_the_record_it_releases() {
    // A holder told a term above its own leaves, and its release carries
    // that term: whoever takes next leads above it. A release never
    // lowers the term, so no candidate reads this record as gone.
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    let mut b = candidate(&lease, "b@host:2");
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    let word = t0 + S / 2;
    assert!(a.step_down(Some(7), word).expect("the release"));
    let stored = lease.clone().read().expect("read").expect("the record");
    assert_eq!(
        (stored.holder.as_str(), stored.term, stored.duration_ms),
        ("", 7, 0)
    );
    assert_eq!(
        a.tick(word).expect("look"),
        TickOutcome::Waiting { term: 7 }
    );
    assert_eq!(
        b.tick(word).expect("take"),
        TickOutcome::Acquired { term: 8 }
    );
    assert_eq!(b.fence().leader_term(word), Some(8));
}

#[test]
fn an_elector_alone_told_a_term_takes_above_it_at_its_next_step() {
    // No other candidate can take a lease the elector made itself, so a
    // word holds nothing back: the fence closes at the word, and the
    // next step takes the lease again above the term said and leads.
    let mut a = Elector::alone("a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    assert!(a.step_down(Some(5), t0 + S).expect("the word"));
    assert_eq!(a.fence().leader_term(t0 + S), None);
    assert_eq!(
        a.tick(t0 + S).expect("take"),
        TickOutcome::Acquired { term: 6 }
    );
    assert_eq!(a.fence().leader_term(t0 + S), Some(6));
}

#[test]
fn a_word_inside_the_takes_wait_hands_the_wait_on_as_a_stop_does() {
    // A taker told to leave before its fence opened never led, so its
    // release pays no wait: the record says the second its wait still
    // owed as its duration, as at a stop inside the wait.
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    assert!(a.waits_to_lead());
    assert!(a.step_down(None, t0 + S).expect("the word"));
    let released = a.observed().expect("the release").clone();
    assert!(released.is_released());
    assert_eq!(released.duration_ms, 1_000);
    assert!(!a.waits_to_lead());
    // Its own next steps wait their turn: the release it reads is its
    // own, and no record is gone.
    assert_eq!(
        a.tick(t0 + S + Duration::from_millis(1)).expect("a step"),
        TickOutcome::Waiting { term: 1 }
    );
    // A term said while it waits is the term its wait says, the
    // release's own term below it.
    assert!(!a
        .step_down(Some(5), t0 + S + Duration::from_millis(2))
        .expect("a word"));
    assert_eq!(
        a.tick(t0 + S + Duration::from_millis(3)).expect("a step"),
        TickOutcome::Waiting { term: 5 }
    );
    // A word once the opening has passed finds no wait to hand on: no
    // holder this run never saw leads by then, and the release says no
    // duration.
    let mut b = candidate(&MemoryLease::new(), "b@host:2");
    assert_eq!(b.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    assert!(b.step_down(None, t0 + 2 * S).expect("the word"));
    assert_eq!(b.observed().expect("the release").duration_ms, 0);
}

#[test]
fn a_higher_term_said_after_the_release_goes_into_the_released_record() {
    // Three gateways show rising floors to one leader. The first word
    // releases the lease under five; the second, nine, finds no lease
    // held here and is written over the release; the third, twelve, is
    // written over the record the second wrote, which this run's sight
    // of the lease moved to. The candidate that takes next takes above
    // twelve, so no gateway refuses its lead.
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    let mut b = candidate(&lease, "b@host:2");
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    let word = t0 + S / 2;
    assert!(a.step_down(Some(5), word).expect("the release"));
    assert!(!a.step_down(Some(9), word).expect("the term is written"));
    let stored = lease.clone().read().expect("read").expect("the record");
    assert_eq!(
        (stored.holder.as_str(), stored.term, stored.duration_ms),
        ("", 9, 0)
    );
    assert!(!a.step_down(Some(12), word).expect("the term is written"));
    let stored = lease.clone().read().expect("read").expect("the record");
    assert_eq!(
        (stored.holder.as_str(), stored.term, stored.duration_ms),
        ("", 12, 0)
    );
    // A term at or under the record's writes nothing more.
    assert!(!a.step_down(Some(12), word).expect("no write"));
    assert!(!a.step_down(Some(7), word).expect("no write"));
    assert_eq!(lease.clone().read().expect("read"), Some(stored));
    assert_eq!(
        a.tick(word).expect("look"),
        TickOutcome::Waiting { term: 12 }
    );
    assert_eq!(
        b.tick(word).expect("take"),
        TickOutcome::Acquired { term: 13 }
    );
    assert_eq!(b.fence().leader_term(word), Some(13));
}

#[test]
fn a_term_said_over_a_release_that_moved_is_counted_and_writes_nothing() {
    // The candidate took the released lease before the higher term was
    // said: the record is that holder's and stays as it is. The gateway
    // that holds the term tells it to that holder.
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    let mut b = candidate(&lease, "b@host:2");
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    let word = t0 + S / 2;
    assert!(a.step_down(Some(5), word).expect("the release"));
    assert_eq!(
        b.tick(word).expect("take"),
        TickOutcome::Acquired { term: 6 }
    );
    assert!(!a.step_down(Some(9), word).expect("the record moved"));
    let stored = lease.clone().read().expect("read").expect("the record");
    assert_eq!((stored.holder.as_str(), stored.term), ("b@host:2", 6));
    assert_eq!(a.highest_term(), 9);
}

#[test]
fn a_term_said_over_a_record_gone_for_the_run_writes_nothing() {
    // A hand writes an old release over a holder's record. For the
    // candidate that saw that holder the record is gone: no holder's
    // release made it so, and the holder may still lead. A term said
    // inside the wait the candidate owes is counted and the record
    // stays as it is: written under the term it would read as the
    // holder's release and be taken at once, beside an open fence.
    let lease = MemoryLease::new();
    let mut w = candidate(&lease, "w@host:1");
    let t0 = Instant::now() + AHEAD;
    w.lead_at(t0);
    let mut x = candidate(&lease, "x@host:2");
    let mut c = candidate(&lease, "c@host:3");
    x.tick(t0).expect("follow");
    assert!(w.step_down(Some(4), t0 + S).expect("the release"));
    let released = lease.clone().read().expect("read").expect("the release");
    let took = x.tick(t0 + S).expect("take");
    assert!(matches!(took, TickOutcome::Acquired { .. }), "{took:?}");
    c.tick(t0 + 2 * S).expect("follow");
    x.tick(t0 + 2 * S).expect("renew");
    let held = lease.clone().read().expect("read").expect("the record");
    assert_eq!(
        lease
            .clone()
            .write(Some(&held), &released)
            .expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let at = t0 + 3 * S;
    let gone = c.tick(at).expect("look");
    assert!(!matches!(gone, TickOutcome::Acquired { .. }), "{gone:?}");
    let said = held.term;
    assert!(!c.step_down(Some(said), at).expect("the word"));
    assert_eq!(lease.clone().read().expect("read"), Some(released));
    assert_eq!(c.highest_term(), said);
    let next = c.tick(at).expect("look");
    assert!(!matches!(next, TickOutcome::Acquired { .. }), "{next:?}");
    assert_eq!(c.fence().leader_term(at), None);
    assert_eq!(x.fence().leader_term(at), Some(said));
}

#[test]
fn a_term_at_or_under_a_holders_own_and_a_word_to_a_candidate_move_no_lease() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    let mut b = candidate(&lease, "b@host:2");
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    // The holder's own term is no word to leave.
    assert!(!a.step_down(Some(1), t0).expect("no write"));
    assert_eq!(
        a.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(a.fence().leader_term(t0 + S), Some(1));
    // A candidate holds no lease: the term is counted and nothing is
    // held back, so its take at the holder's stop is at once and above
    // the term it was told.
    assert!(!b.step_down(Some(7), t0 + S).expect("no write"));
    assert!(!b.step_down(None, t0 + S).expect("no write"));
    assert!(a.release().expect("the holder's stop"));
    assert_eq!(
        b.tick(t0 + S).expect("take"),
        TickOutcome::Acquired { term: 8 }
    );
}

#[test]
fn a_take_that_would_reach_the_term_ceiling_is_an_error_at_every_step() {
    // The last term under the ceiling is taken; above it none is left.
    let free = LeaseRecord {
        holder: String::new(),
        ..held_by("x@host:9", TERM_CEILING - 2, 0)
    };
    let mut lease = MemoryLease::new();
    assert_eq!(
        lease.write(None, &free).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(
        b.tick(t0).expect("the last term"),
        TickOutcome::Acquired {
            term: TERM_CEILING - 1
        }
    );
    assert!(b.release().expect("the stop"));
    let mut c = candidate(&lease, "c@host:3");
    for step in 1..=2 {
        let err = c.tick(t0 + step * S).expect_err("no term is left");
        assert_eq!(
                err.to_string(),
                format!(
                    "election term: the next term {TERM_CEILING} is not below the ceiling {TERM_CEILING}"
                )
            );
        assert!(!c.is_holder());
    }
    let stored = lease.clone().read().expect("read").expect("the record");
    assert!(stored.is_released(), "nothing was written");
}

#[test]
fn a_take_over_the_largest_term_a_record_can_say_is_the_term_error() {
    // A record no elector wrote says the largest term: the next term's
    // sum stops there and never wraps to a term under the ceiling.
    let free = LeaseRecord {
        holder: String::new(),
        ..held_by("x@host:9", u64::MAX, 0)
    };
    let mut lease = MemoryLease::new();
    assert_eq!(
        lease.write(None, &free).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let mut b = candidate(&lease, "b@host:2");
    let refused = b
        .tick(Instant::now() + AHEAD)
        .expect_err("no term is left to take");
    assert!(matches!(refused, ElectError::Term(u64::MAX)), "{refused}");
    assert!(!b.is_holder());
    let stored = lease.clone().read().expect("read").expect("the record");
    assert!(stored.is_released(), "nothing was written");
}

#[test]
fn a_step_handed_an_earlier_instant_leaves_the_latest_one_standing() {
    // The loop steps by the clock, which only grows. A caller that
    // hands its own instants out of order takes nothing back: the
    // wait's reader answers by the latest instant a step was handed.
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    assert_eq!(
        a.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert!(a.waits_to_lead());
    assert_eq!(
        a.tick(t0 + 2 * S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert!(!a.waits_to_lead());
    assert_eq!(
        a.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert!(!a.waits_to_lead(), "the opening stays passed");
}

#[test]
fn a_taker_stopped_at_half_its_wait_hands_on_the_half_it_still_owed() {
    // The take's lead would open two seconds after it. Stopped a second
    // into that wait, the taker hands on the second it still owed: the
    // next candidate takes a second after its sight of the release and
    // leads at its take, when the taker would have and one look.
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    // A microsecond past the second: the wait still owed has a
    // fraction, and the record's millisecond rounds it up.
    let stopped = t0 + S + Duration::from_micros(1);
    assert_eq!(
        a.tick(stopped).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert!(a.waits_to_lead());
    assert!(a.release().expect("the stop"));
    let released = lease.clone().read().expect("read").expect("the record");
    assert!(released.is_released());
    assert_eq!(released.duration_ms, 1_000);
    let saw = t0 + S + Duration::from_millis(1);
    assert_eq!(b.tick(saw).expect("look"), TickOutcome::Gone { term: 1 });
    assert_eq!(
        b.tick(saw + S - Duration::from_micros(1)).expect("look"),
        TickOutcome::Gone { term: 1 }
    );
    assert_eq!(
        b.tick(saw + S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(b.fence().leader_term(saw + S), Some(2));
    assert!(!b.waits_to_lead());
}

#[test]
fn a_release_after_the_lease_moved_answers_false() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    a.tick(t0).expect("acquire");
    assert!(a.waits_to_lead());
    b.tick(t0).expect("follow");
    assert_eq!(
        b.tick(t0 + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert!(
        !a.release().expect("nothing to give up"),
        "the lease is b's"
    );
    assert!(!a.is_holder());
    assert!(!a.waits_to_lead(), "the wait went with the lease");
    assert_eq!(
        lease
            .clone()
            .read()
            .expect("read")
            .expect("the record")
            .holder,
        "b@host:2"
    );
}

#[test]
fn a_term_said_that_leaves_none_to_take_is_refused_before_it_is_counted() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    for said in [TERM_CEILING - 1, TERM_CEILING, u64::MAX] {
        let refused = a
            .step_down(Some(said), t0)
            .expect_err("no term is left to take");
        assert!(
            matches!(refused, ElectError::Term(next) if next == said.saturating_add(1)),
            "{refused}"
        );
    }
    // Nothing was counted and nothing released: the holder leads on
    // under its own term.
    assert!(a.is_holder());
    assert_eq!(a.fence().leader_term(t0), Some(1));
    assert_eq!(
        a.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    // The highest term a word may say leaves the last one to take.
    assert!(a
        .step_down(Some(TERM_CEILING - 2), t0 + S)
        .expect("the word"));
    let released = a.observed().expect("the release").clone();
    assert_eq!(released.term, TERM_CEILING - 2);
}
