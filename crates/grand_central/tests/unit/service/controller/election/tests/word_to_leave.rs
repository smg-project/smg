//! The words a replica is told: a term said to a leader, to a follower and to
//! a replica alone, the word to leave and a planned hand-over, the wait of a
//! replica told to leave, the release at a word, and the terms the door
//! refuses.

use super::*;

/// A term said behind a word's leave of a replica alone is counted ahead of
/// the take, and the take stands above it. A word to leave and a word of
/// term 7 stood on the channel before the loop began: the replica leaves its
/// first lease, counts the term, and takes term 8. No take stands between
/// the two: one leave was made, and two lines say a lead.
#[test]
fn a_term_said_behind_a_words_leave_is_counted_ahead_of_the_take() {
    let name = "a@127.0.0.1:1";
    let store = Arc::new(MemoryStore::new());
    let elector = Elector::alone(name, unhurried()).expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let ahead = vec![Word::StepDown(None), Word::StepDown(Some(7))];
    let looping = looping_behind(elector, Arc::clone(&store), standing, ahead);
    until("the take above the term said", || {
        let stands = stood.borrow();
        stands.holder == name && stands.term == 8
    });
    assert_eq!(store.term(), 8);
    until("the line of the second lead", || {
        looping.count(": leading") == 2
    });
    let (left, text) = looping.stopped_with("stepped down");
    assert_eq!(left, 1, "{text}");
    assert_eq!(text.matches(": leading").count(), 2, "{text}");
}

/// The same behind the log's leave. The replica alone took term 1 over a
/// log that followed term 5, and a word of term 7 stood on the channel
/// before the loop began. The loop leaves that take in its own step and
/// counts the term said ahead of the take that follows at once: that take
/// is term 8, and none stands at term 6. One line says a leave, and it
/// names the followed term: the log's leave, with none at the word. One
/// line says a lead.
#[test]
fn a_term_said_behind_the_logs_leave_is_counted_ahead_of_the_take() {
    let name = "a@127.0.0.1:1";
    let store = followed(5);
    let elector = Elector::alone(name, unhurried()).expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let ahead = vec![Word::StepDown(Some(7))];
    let looping = looping_behind(elector, Arc::clone(&store), standing, ahead);
    until("the take above the term said", || {
        let stands = stood.borrow();
        stands.holder == name && stands.term == 8
    });
    assert_eq!(store.term(), 8);
    until("the line of the lead", || looping.said(": leading"));
    let (said, text) = looping.stopped_with(RETAKES);
    assert_eq!(said, 1, "{text}");
    assert_eq!(left_under(&text), [Some(5)], "{text}");
    assert_eq!(lines_with(&text, ": leading"), 1, "{text}");
}

#[test]
fn a_leader_told_a_term_above_its_own_leaves_and_the_standby_leads_above_that_term() {
    let lease = MemoryLease::new();
    let candidate = |identity: &str| {
        Elector::new(Box::new(lease.clone()), identity, brisk()).expect("the settings are in order")
    };
    let (first, second) = (candidate("a@127.0.0.1:1"), candidate("b@127.0.0.1:2"));
    let (first_fence, second_fence) = (first.fence(), second.fence());
    let fences = [&*first_fence, &*second_fence];
    let second_store = Arc::new(MemoryStore::new());
    let (first_standing, first_stood) = watch::channel(ElectionObservation::default());
    let (second_standing, second_stood) = watch::channel(ElectionObservation::default());
    let first_running =
        start(first, Arc::new(MemoryStore::new()), first_standing).expect("a thread");
    wait_for("the first lead", fences, || {
        first_fence.leader_term(Instant::now()) == Some(1)
    });
    let second_running =
        start(second, Arc::clone(&second_store), second_standing).expect("a thread");
    wait_for("the second's first look", fences, || {
        second_stood.borrow().holder == "a@127.0.0.1:1"
    });

    // A term of seven is said to the leader, as a gateway's floor is:
    // it leaves, and the standby takes the released lease at its next
    // look, above the term the leader was told. No instant has two open
    // fences: every wait here asks both.
    first_running
        .door()
        .saw(7)
        .expect("a term that leaves one to take");
    wait_for("the standby's take above the term said", fences, || {
        second_fence.leader_term(Instant::now()) == Some(8) && second_store.term() == 8
    });
    wait_for("the replica that left follows the taker", fences, || {
        *first_stood.borrow()
            == ElectionObservation {
                holder: "b@127.0.0.1:2".to_owned(),
                term: 8,
                looked: true,
            }
    });
    assert_eq!(first_fence.leader_term(Instant::now()), None);
    first_running.stop();
    ended(first_running);
    second_running.stop();
    ended(second_running);
}

#[test]
fn a_replica_alone_over_its_own_lease_that_is_told_to_leave_leads_again_at_the_word() {
    // No other candidate can take a lease the elector made itself: a
    // word holds nothing back and the take follows it at once, a renew
    // period of a minute ahead of the next step by the clock. The line
    // at the word says that the next step takes the lease again.
    let elector = Elector::alone("a@127.0.0.1:1", unhurried()).expect("the settings are in order");
    let fence = elector.fence();
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping(elector, standing);
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    looping
        .stop
        .send(Word::StepDown(Some(7)))
        .expect("the loop runs");
    until("the retake above the term said", || {
        fence.leader_term(Instant::now()) == Some(8)
    });
    until("the standing after the retake", || {
        stood.borrow().holder == "a@127.0.0.1:1"
    });
    assert_eq!(stood.borrow().term, 8);
    let (said, text) = looping.stopped_with(RETAKES);
    assert_eq!(said, 1, "{text}");
    // The fence closed at the word, and the retake says it leads again.
    assert_eq!(text.matches(": leading").count(), 2, "{text}");
    assert!(!text.contains("waits its turn"), "{text}");
    assert!(
        !text.contains("takes nothing for one lease duration"),
        "{text}"
    );
}

/// A replica another candidate can follow takes no step at the word: a
/// standby's next look takes what it left, and its own next step comes
/// with the renew period, a minute away here. A step would write the
/// line of the turn it waits or of the record with no holder.
#[test]
fn a_replica_another_can_follow_takes_no_step_at_the_word_to_leave() {
    let elector = Elector::new(Box::new(MemoryLease::new()), "a@127.0.0.1:1", unhurried())
        .expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping(elector, standing);
    until("the first take", || {
        stood.borrow().holder == "a@127.0.0.1:1"
    });
    looping
        .stop
        .send(Word::StepDown(None))
        .expect("the loop runs");
    until("the leave", || {
        looping.said("takes nothing for one lease duration")
    });
    // The loop is past the word: a step that followed it is written
    // by now, and the stop writes none.
    thread::sleep(Duration::from_millis(200));
    let (_, text) = looping.stopped_with("stepped down");
    assert!(!text.contains("waits its turn"), "{text}");
    assert!(!text.contains("names no holder"), "{text}");
}

#[test]
fn a_leader_told_its_own_term_keeps_on_and_logs_no_second_line_of_leading() {
    // A word that moves no lease leaves the report's flag as it stood:
    // the leader renews on under its term, and no renewal says again
    // that it leads.
    let elector = Elector::alone("a@127.0.0.1:1", brisk()).expect("the settings are in order");
    let fence = elector.fence();
    let (standing, mut stood) = watch::channel(ElectionObservation::default());
    let looping = looping(elector, standing);
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    until("the line of leading", || looping.said(": leading"));
    looping
        .stop
        .send(Word::StepDown(Some(1)))
        .expect("the loop runs");
    looks(&mut stood, 3);
    assert_eq!(fence.leader_term(Instant::now()), Some(1));
    let (leading, text) = looping.stopped_with(": leading");
    assert_eq!(leading, 1, "{text}");
}

/// A planned hand-over is of one term's lease. Made under term one and
/// taken after a gateway's floor moved this replica to term three, it
/// leaves that take alone; made under the term held, it is taken.
#[test]
fn a_hand_over_planned_under_a_term_the_replica_left_releases_nothing() {
    let elector = Elector::alone("a@127.0.0.1:1", brisk()).expect("the settings are in order");
    let fence = elector.fence();
    let (standing, mut stood) = watch::channel(ElectionObservation::default());
    let running = start(elector, Arc::new(MemoryStore::new()), standing).expect("a thread");
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    let door = running.door();
    door.saw(2).expect("a term that leaves one to take");
    until("the take above the floor", || {
        fence.leader_term(Instant::now()) == Some(3)
    });
    // The hand-over of term one comes late: three steps on, the lease
    // of term three is still held.
    door.hand_over(1);
    looks(&mut stood, 3);
    assert_eq!(fence.leader_term(Instant::now()), Some(3));
    door.hand_over(3);
    until("the take after the hand-over of the term held", || {
        fence.leader_term(Instant::now()) == Some(4)
    });
    running.stop();
    ended(running);
}

#[test]
fn a_replica_with_no_standby_that_is_told_to_leave_waits_one_lease_duration_and_leads_again_above()
{
    let lease = MemoryLease::new();
    let settings = brisk();
    let elector = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", settings)
        .expect("the settings are in order");
    let fence = elector.fence();
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping(elector, standing);
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    // The fence opens between two steps: the next one writes the line.
    until("the line of leading", || looping.said(": leading"));

    // A term of seven is said: the replica leaves, and it names nobody
    // under that term while it waits its turn.
    let word = Instant::now();
    looping
        .stop
        .send(Word::StepDown(Some(7)))
        .expect("the loop runs");
    until("the leave", || {
        *stood.borrow()
            == ElectionObservation {
                holder: String::new(),
                term: 7,
                looked: true,
            }
    });
    assert_eq!(fence.leader_term(Instant::now()), None);
    // With no standby over a lease it shares, it takes the lease back
    // when its hold ends, one lease duration after the word and above
    // the term it was told: the price of the door where no standby
    // looks.
    until("the retake", || {
        fence.leader_term(Instant::now()) == Some(8)
    });
    let out = word.elapsed();
    assert!(
        out >= settings.duration,
        "led again {out:?} after the word, inside its hold"
    );
    until("the standing after the retake", || {
        stood.borrow().holder == "a@127.0.0.1:1"
    });
    assert_eq!(stood.borrow().term, 8);
    let (waits, text) =
        looping.stopped_with("the lease stands free and this replica waits its turn");
    assert_eq!(waits, 1, "{text}");
    let line = text
        .lines()
        .find(|line| line.contains("waits its turn"))
        .expect("the waiting line");
    assert!(
        line.contains(" INFO "),
        "a turn waited out is no warning: {line}"
    );
    assert_eq!(
        text.matches("stepped down; the lease is released").count(),
        1,
        "{text}"
    );
    assert_eq!(text.matches(": leading").count(), 2, "{text}");
    assert!(!text.contains("the lease's record is gone"), "{text}");
    assert!(!text.contains("the lease's term is under"), "{text}");
}

#[test]
fn a_replica_that_waits_its_turn_names_nobody_over_a_record_that_ran_out() {
    // A record that names a holder and ran out by this replica's clock
    // is free to take. A replica inside its hold takes nothing and
    // follows nobody: its standing names no holder, whatever the record
    // says.
    let mut lease = MemoryLease::new();
    let elector = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", brisk())
        .expect("the settings are in order");
    let fence = elector.fence();
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping(elector, standing);
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    looping
        .stop
        .send(Word::StepDown(Some(7)))
        .expect("the loop runs");
    until("the leave", || stood.borrow().term == 7);
    // Another holder's record lands over the release, says five renew
    // periods and is never renewed: the replica follows it while it
    // lasts, and its own hold outlasts it.
    let released = lease.read().expect("a read").expect("the release");
    assert!(released.is_released());
    let ran_out = LeaseRecord {
        holder: "x@127.0.0.1:9".to_owned(),
        term: 8,
        acquired_at_micros: released.renewed_at_micros + 1,
        renewed_at_micros: released.renewed_at_micros + 1,
        duration_ms: 100,
    };
    assert_eq!(
        lease.write(Some(&released), &ran_out).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    until("the record that ran out names nobody", || {
        *stood.borrow()
            == ElectionObservation {
                holder: String::new(),
                term: 8,
                looked: true,
            }
    });
    assert_eq!(fence.leader_term(Instant::now()), None);
    until("the retake above the record that ran out", || {
        fence.leader_term(Instant::now()) == Some(9)
    });
    let (waits, text) =
        looping.stopped_with("the lease stands free and this replica waits its turn");
    assert!(waits >= 1, "{text}");
}

#[test]
fn a_replica_that_waits_its_turn_counts_a_term_it_read_in_a_release() {
    // Inside this replica's hold a standby takes the lease and gives
    // it up between two looks: the replica reads only the release,
    // under term eight. The standing counts that term, so a holder
    // restored under term seven is not named.
    let mut lease = MemoryLease::new();
    let elector = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", brisk())
        .expect("the settings are in order");
    let fence = elector.fence();
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping(elector, standing);
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    looping
        .stop
        .send(Word::StepDown(Some(7)))
        .expect("the loop runs");
    until("the leave", || stood.borrow().term == 7);
    let released = lease.read().expect("a read").expect("the release");
    assert!(released.is_released());
    let later = LeaseRecord {
        term: 8,
        renewed_at_micros: released.renewed_at_micros + 1,
        duration_ms: 0,
        ..released.clone()
    };
    assert_eq!(
        lease.write(Some(&released), &later).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    until("the term read in the release is counted", || {
        *stood.borrow()
            == ElectionObservation {
                holder: String::new(),
                term: 8,
                looked: true,
            }
    });
    let restored = LeaseRecord {
        holder: "x@127.0.0.1:9".to_owned(),
        term: 7,
        acquired_at_micros: later.renewed_at_micros + 1,
        renewed_at_micros: later.renewed_at_micros + 1,
        duration_ms: 60_000,
    };
    assert_eq!(
        lease.write(Some(&later), &restored).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    until("the line of the fallen term", || {
        looping.said("the lease's term is under the highest")
    });
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 8,
            looked: true,
        }
    );
    let (fallen, text) = looping.stopped_with("the lease's term is under the highest");
    assert_eq!(fallen, 1, "{text}");
}

#[test]
fn a_term_told_stands_over_a_release_read_under_a_lower_term() {
    // A release under term three stands for a minute. A follower told
    // term nine reads it look after look, and the standing keeps the
    // term told.
    let free = LeaseRecord {
        term: 3,
        duration_ms: 60_000,
        ..LeaseRecord::default()
    };
    let mut lease = MemoryLease::new();
    assert_eq!(
        lease.write(None, &free).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let follower = Elector::new(Box::new(lease.clone()), "b@127.0.0.1:2", brisk())
        .expect("the settings are in order");
    let (standing, mut stood) = watch::channel(ElectionObservation::default());
    let looping = looping(follower, standing);
    until("the term read in the release", || stood.borrow().term == 3);
    looping
        .stop
        .send(Word::StepDown(Some(9)))
        .expect("the loop runs");
    until("the term counted", || stood.borrow().term == 9);
    looks(&mut stood, 5);
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 9,
            looked: true
        }
    );
    let (gone, text) = looping.stopped_with("the lease's record is gone or names no holder");
    assert_eq!(gone, 1, "{text}");
}

#[test]
fn a_replica_told_to_leave_inside_its_takes_wait_says_only_that_it_waits() {
    // The release such a replica reads at its next steps is its own:
    // it waits its turn, and no line says a record is gone.
    let elector = Elector::new(Box::new(MemoryLease::new()), "a@127.0.0.1:1", brisk())
        .expect("the settings are in order");
    let (standing, _stood) = watch::channel(ElectionObservation::default());
    let looping = looping(elector, standing);
    until("the line of the take", || {
        looping.said("took a lease with no holder to wait for")
    });
    looping
        .stop
        .send(Word::StepDown(None))
        .expect("the loop runs");
    until("the line of the wait", || looping.said("waits its turn"));
    let (gone, text) = looping.stopped_with("the lease's record is gone");
    assert_eq!(gone, 0, "{text}");
}

#[test]
fn a_word_past_the_door_that_leaves_no_term_to_take_counts_nothing() {
    // The controller's door refuses such a term before a word is said.
    // One that reaches the loop all the same is refused by the elector:
    // the holder leads on, and the standing counts nothing.
    let elector = Elector::new(Box::new(MemoryLease::new()), "a@127.0.0.1:1", brisk())
        .expect("the settings are in order");
    let fence = elector.fence();
    let (standing, mut stood) = watch::channel(ElectionObservation::default());
    let looping = looping(elector, standing);
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    looping
        .stop
        .send(Word::StepDown(Some(TERM_CEILING - 1)))
        .expect("the loop runs");
    until("the line of the refusal", || {
        looping.said("the word is refused")
    });
    looks(&mut stood, 3);
    assert_eq!(fence.leader_term(Instant::now()), Some(1));
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: "a@127.0.0.1:1".to_owned(),
            term: 1,
            looked: true
        }
    );
    let (refused, text) = looping.stopped_with("the word is refused");
    assert_eq!(refused, 1, "{text}");
    assert!(!text.contains("stepped down;"), "{text}");
}

#[test]
fn a_term_said_to_a_follower_is_counted_and_its_take_is_above_it() {
    // The holder's duration is a minute: the follower takes only at its
    // release.
    let held = LeaseRecord {
        holder: "x@127.0.0.1:8".to_owned(),
        term: 5,
        duration_ms: 60_000,
        ..LeaseRecord::default()
    };
    let mut lease = MemoryLease::new();
    assert_eq!(
        lease.write(None, &held).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let follower = Elector::new(Box::new(lease.clone()), "b@127.0.0.1:2", brisk())
        .expect("the settings are in order");
    let fence = follower.fence();
    let (standing, mut stood) = watch::channel(ElectionObservation::default());
    let looping = looping(follower, standing);
    until("the first look", || stood.borrow().holder == held.holder);

    // A caller says it saw term nine. No lease moves; the holder under
    // term five is named no longer, since no write stored it under nine.
    looping
        .stop
        .send(Word::StepDown(Some(9)))
        .expect("the loop runs");
    until("the term counted", || {
        *stood.borrow()
            == ElectionObservation {
                holder: String::new(),
                term: 9,
                looked: true,
            }
    });
    // Looks at the holder's record under the lower term: one line a
    // streak, nobody named, no write.
    looks(&mut stood, 5);
    assert!(stood.borrow().holder.is_empty());
    assert_eq!(stood.borrow().term, 9);
    assert_eq!(lease.read().expect("a read"), Some(held.clone()));
    // The holder releases: the follower takes at once, above the term it
    // was told, and it was never held back.
    let released = LeaseRecord {
        holder: String::new(),
        duration_ms: 0,
        ..held.clone()
    };
    assert_eq!(
        lease.write(Some(&held), &released).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    until("the take", || fence.leader_term(Instant::now()) == Some(10));
    let (under, text) =
        looping.stopped_with("the lease's term is under the highest this replica saw or was told");
    assert_eq!(under, 1, "{text}");
    assert!(!text.contains("stepped down"), "{text}");
    assert!(!text.contains("waits its turn"), "{text}");
}

/// A term said reaches the standing at the word, before the next step
/// could read it: a follower told term nine names the holder it
/// followed under term three no longer, a renew period ahead of its
/// next look.
#[test]
fn a_term_said_to_a_follower_stands_before_its_next_step() {
    let held = LeaseRecord {
        holder: "x@127.0.0.1:8".to_owned(),
        term: 3,
        duration_ms: 60_000,
        ..LeaseRecord::default()
    };
    let mut lease = MemoryLease::new();
    assert_eq!(
        lease.write(None, &held).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let follower = Elector::new(Box::new(lease), "b@127.0.0.1:2", unhurried())
        .expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping(follower, standing);
    until("the first look", || stood.borrow().holder == held.holder);
    looping
        .stop
        .send(Word::StepDown(Some(9)))
        .expect("the loop runs");
    until("the term said", || stood.borrow().term == 9);
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 9,
            looked: true,
        }
    );
    let (stepped, text) = looping.stopped_with("stepped down");
    assert_eq!(stepped, 0, "{text}");
}

/// A word's release that comes back moved reads the lease again, and
/// the standing counts that record's term at the word. A holder told to
/// leave under no term finds another replica's release under term
/// nine: nobody is named under term nine, before the next step.
#[test]
fn a_release_read_again_at_a_word_is_counted_before_the_next_step() {
    let store = Losable::default();
    let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", unhurried())
        .expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping(holder, standing);
    until("the take", || stood.borrow().holder == "a@127.0.0.1:1");
    store.ahead([Ahead::Write(LeaseRecord {
        term: 9,
        ..LeaseRecord::default()
    })]);
    looping
        .stop
        .send(Word::StepDown(None))
        .expect("the loop runs");
    until("the word taken", || stood.borrow().holder.is_empty());
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 9,
            looked: true,
        }
    );
    let (stepped, text) = looping.stopped_with("stepped down; the lease is released");
    assert_eq!(stepped, 1, "{text}");
}

#[test]
fn a_release_that_fails_at_the_word_is_tried_again_at_the_next_step_with_nobody_named() {
    let lease = MemoryLease::new();
    let releases = Arc::new(Mutex::new(Vec::new()));
    let store = FailsReleases {
        lease: lease.clone(),
        failing: 2,
        stalled: Duration::ZERO,
        releases: Arc::clone(&releases),
    };
    let elector =
        Elector::new(Box::new(store), "a@127.0.0.1:1", brisk()).expect("the settings are in order");
    let fence = elector.fence();
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping(elector, standing);
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    // The fence opens between two steps: the next one writes the line.
    until("the line of leading", || looping.said(": leading"));

    // The release fails at the word and at the step after it, and lands
    // at the third try. The fence is closed from the word, nobody is
    // named, and no step in between renews the lease.
    looping
        .stop
        .send(Word::StepDown(None))
        .expect("the loop runs");
    until("the release", || {
        lease
            .clone()
            .read()
            .expect("a read")
            .is_some_and(|record| record.is_released())
    });
    assert_eq!(fence.leader_term(Instant::now()), None);
    assert_eq!(releases.lock().len(), 3);
    assert!(stood.borrow().holder.is_empty());
    let (failed, text) = looping
        .stopped_with("stepped down; the release failed and is tried again at the next step");
    assert_eq!(failed, 1, "{text}");
    assert_eq!(text.matches(": leading").count(), 1, "{text}");
}

#[test]
fn the_highest_term_a_word_may_say_leaves_the_last_term_to_take() {
    let elector = Elector::new(Box::new(MemoryLease::new()), "a@127.0.0.1:1", brisk())
        .expect("the settings are in order");
    let fence = elector.fence();
    let (standing, _stood) = watch::channel(ElectionObservation::default());
    let running = start(elector, Arc::new(MemoryStore::new()), standing).expect("a thread");
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    // The replica leaves under the term said, and the last term under
    // the ceiling is its next take.
    running
        .door()
        .saw(TERM_CEILING - 2)
        .expect("the last term a word may say");
    until("the retake", || {
        fence.leader_term(Instant::now()) == Some(TERM_CEILING - 1)
    });
    running.stop();
    ended(running);
}

#[test]
fn a_term_said_at_the_door_is_the_handles_word() {
    let elector = Elector::alone("a@127.0.0.1:1", brisk()).expect("the settings are in order");
    let fence = elector.fence();
    let (standing, _stood) = watch::channel(ElectionObservation::default());
    let running = start(elector, Arc::new(MemoryStore::new()), standing).expect("a thread");
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    // The door says through the handle's words: a term said through it
    // is a word to leave that the thread takes.
    running
        .door()
        .saw(7)
        .expect("a term that leaves one to take");
    until("the retake above the term said", || {
        fence.leader_term(Instant::now()) == Some(8)
    });
    running.stop();
    ended(running);
}

#[test]
fn the_door_refuses_a_term_that_leaves_none_to_take() {
    let (door, said) = Said::door();
    let refused = door
        .saw(TERM_CEILING - 1)
        .expect_err("no term is left to take");
    assert!(
        matches!(refused, ElectError::Term(next) if next == TERM_CEILING),
        "{refused}"
    );
    assert!(said.terms().is_empty(), "no word is said");
}

#[test]
fn a_term_that_leaves_none_to_take_is_refused_at_the_door() {
    let elector = Elector::new(
        Box::new(MemoryLease::new()),
        "a@127.0.0.1:1",
        ElectionSettings::default(),
    )
    .expect("the settings are in order");
    let (standing, _stood) = watch::channel(ElectionObservation::default());
    let running = start(elector, Arc::new(MemoryStore::new()), standing).expect("a thread");
    let refused = running
        .door()
        .saw(TERM_CEILING - 1)
        .expect_err("no term is left to take");
    assert!(
        matches!(refused, ElectError::Term(next) if next == TERM_CEILING),
        "{refused}"
    );
    running.stop();
    ended(running);
}
