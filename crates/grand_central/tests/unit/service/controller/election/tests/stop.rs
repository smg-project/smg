//! How the loop ends: a stop, a dropped election and a closed channel, a stop
//! queued behind a leave, a step that panics, the release at a stop where the
//! lease moved or the write fails, and two candidates that hand over at a
//! stop.

use super::*;

/// A host that runs the election in its own process cancels it by
/// dropping it: the thread hears its stop channel close, gives the
/// lease up and ends, as at a stop.
#[test]
fn an_election_dropped_without_a_stop_gives_the_lease_up_and_ends() {
    let mut lease = MemoryLease::new();
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(50),
        renew_deadline: Duration::from_millis(400),
    };
    let candidate = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", settings)
        .expect("the settings are in order");
    let fence = candidate.fence();
    let (standing, _stood) = watch::channel(ElectionObservation::default());
    let mut running = start(candidate, Arc::new(MemoryStore::new()), standing).expect("a thread");
    until("the take", || fence.leader_term(Instant::now()) == Some(1));

    let mut end = running.end.take().expect("the end was not seen yet");
    drop(running);
    until("the thread's end", || end.try_recv().is_ok());
    assert_eq!(fence.leader_term(Instant::now()), None);
    let stored = lease.read().expect("a read");
    assert!(
        stored.is_some_and(|record| record.holder.is_empty()),
        "the lease is free"
    );
}

/// A lease that can be armed to panic on a read after it stores a take.
struct PanicsAfterTake {
    lease: MemoryLease,
    armed: Arc<AtomicBool>,
    took: bool,
}

impl LeaseBackend for PanicsAfterTake {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        assert!(
            !self.took || !self.armed.load(Ordering::SeqCst),
            "the read after the take panics"
        );
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        let written = self.lease.write(expected, record)?;
        if written == ConditionalWriteOutcome::Done && !record.holder.is_empty() {
            self.took = true;
        }
        Ok(written)
    }

    fn place(&self) -> String {
        self.lease.place()
    }
}

#[test]
fn a_panicking_step_closes_its_fence_and_releases_before_the_thread_ends() {
    let mut lease = MemoryLease::new();
    let armed = Arc::new(AtomicBool::new(false));
    let released = LeaseRecord {
        term: 1,
        ..LeaseRecord::default()
    };
    assert_eq!(
        lease.write(None, &released).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(50),
        renew_deadline: Duration::from_millis(400),
    };
    let candidate = Elector::new(
        Box::new(PanicsAfterTake {
            lease: lease.clone(),
            armed: armed.clone(),
            took: false,
        }),
        "a@127.0.0.1:1",
        settings,
    )
    .expect("the settings are in order");
    let fence = candidate.fence();
    let (standing, _stood) = watch::channel(ElectionObservation::default());
    let running = start(candidate, Arc::new(MemoryStore::new()), standing).expect("a thread");
    thread::sleep(Duration::from_millis(100));
    until("the take", || fence.leader_term(Instant::now()) == Some(2));
    armed.store(true, Ordering::SeqCst);

    ended(running);
    assert_eq!(fence.leader_term(Instant::now()), None);
    let stored = lease.read().expect("a read");
    assert!(
        stored.is_some_and(|record| record.holder.is_empty()),
        "the lease is free"
    );
}

/// A door given out keeps no election alive: the handle alone holds the
/// words. Dropped while the service's shared state still holds the
/// door, the election ends as it does with no door, and the door then
/// judges a term and says it to no one.
#[test]
fn an_election_dropped_while_a_door_is_held_gives_the_lease_up_and_ends() {
    let mut lease = MemoryLease::new();
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(50),
        renew_deadline: Duration::from_millis(400),
    };
    let candidate = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", settings)
        .expect("the settings are in order");
    let fence = candidate.fence();
    let (standing, _stood) = watch::channel(ElectionObservation::default());
    let mut running = start(candidate, Arc::new(MemoryStore::new()), standing).expect("a thread");
    until("the take", || fence.leader_term(Instant::now()) == Some(1));

    let door = running.door();
    let mut end = running.end.take().expect("the end was not seen yet");
    drop(running);
    until("the thread's end", || end.try_recv().is_ok());
    assert_eq!(fence.leader_term(Instant::now()), None);
    let stored = lease.read().expect("a read");
    assert!(
        stored.is_some_and(|record| record.holder.is_empty()),
        "the lease is free"
    );
    door.saw(7).expect("a term that leaves one to take");
    let refused = door
        .saw(TERM_CEILING - 1)
        .expect_err("no term is left to take");
    assert!(
        matches!(refused, ElectError::Term(next) if next == TERM_CEILING),
        "{refused}"
    );
}

/// What one stop showed.
struct Stop {
    /// How long after the stop each release began. The stop's instant
    /// is taken before the stop is sent, so no release began sooner:
    /// a hold of the host only makes these longer.
    releases: Vec<Duration>,
    /// How long after the stop the thread had ended.
    took: Duration,
    /// Whether the lease is free.
    free: bool,
}

/// A stop that waits behind the log's leave ends the loop with no further
/// take. The replica is alone over its lease and its log followed term 5:
/// its first take, term 1, starts no term and is left in its own step. The
/// stop stood on the channel before the loop began, so no clock decides: the
/// loop reads it ahead of the take that would follow at once. The store
/// stays under the first take's term, and no line says a lead.
#[test]
fn a_stop_queued_behind_the_logs_leave_ends_the_loop_with_no_further_take() {
    let name = "a@127.0.0.1:1";
    let store = followed(5);
    let elector = Elector::alone(name, unhurried()).expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping_behind(elector, Arc::clone(&store), standing, vec![Word::Stop]);
    let text = looping.ended();
    // One take, the first: the store was told its term and no other, and
    // its log leads none.
    assert_eq!(store.term(), 1, "{text}");
    assert_eq!(store.followed(), Some(5), "{text}");
    // The last standing names nobody, under the term the leave said.
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 5,
            looked: true,
        }
    );
    assert_eq!(lines_with(&text, "started no term"), 1, "{text}");
    assert_eq!(lines_with(&text, RETAKES), 1, "{text}");
    assert_eq!(lines_with(&text, ": leading"), 0, "{text}");
}

/// The same behind a word's leave. The replica alone leads under term 1. A
/// word to leave, a stop and a word of term 7 stood on the channel before
/// the loop began: the loop leaves at the first, and reads the stop ahead
/// of its next take. The word behind the stop is not worked: the last
/// standing holds term 1. One take was made, and one line says a lead.
#[test]
fn a_stop_queued_behind_a_words_leave_ends_the_loop_with_no_further_take() {
    let name = "a@127.0.0.1:1";
    let store = Arc::new(MemoryStore::new());
    let elector = Elector::alone(name, unhurried()).expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let ahead = vec![Word::StepDown(None), Word::Stop, Word::StepDown(Some(7))];
    let looping = looping_behind(elector, Arc::clone(&store), standing, ahead);
    let text = looping.ended();
    assert_eq!(store.term(), 1, "{text}");
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 1,
            looked: true,
        }
    );
    assert_eq!(lines_with(&text, RETAKES), 1, "{text}");
    assert_eq!(lines_with(&text, ": leading"), 1, "{text}");
}

/// The same where the word that leaves says a term. A word of term 7 and a
/// stop stood on the channel before the loop began: the replica counts the
/// term and leaves its first lease for it, and reads the stop ahead of the
/// take that would stand above the term. One take was made, and one line
/// says a lead.
#[test]
fn a_stop_queued_behind_a_said_terms_leave_ends_the_loop_with_no_further_take() {
    let name = "a@127.0.0.1:1";
    let store = Arc::new(MemoryStore::new());
    let elector = Elector::alone(name, unhurried()).expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let ahead = vec![Word::StepDown(Some(7)), Word::Stop];
    let looping = looping_behind(elector, Arc::clone(&store), standing, ahead);
    let text = looping.ended();
    assert_eq!(store.term(), 1, "{text}");
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 7,
            looked: true,
        }
    );
    assert_eq!(lines_with(&text, RETAKES), 1, "{text}");
    assert_eq!(lines_with(&text, ": leading"), 1, "{text}");
}

/// The same with a term said between the two, and the leave a planned
/// hand-over's. A hand-over of term 1, a word of term 7 and a stop stood on
/// the channel before the loop began: the loop leaves at the first, counts
/// the term of the second with no lease held, and reads the stop ahead of
/// the take that would stand above that term. One take was made, and one
/// line says a lead. One line says a leave, and it names no term: the
/// hand-over left the lease, and the word of term 7 found none to leave.
#[test]
fn a_stop_queued_behind_a_hand_over_and_a_said_term_ends_the_loop_with_no_further_take() {
    let name = "a@127.0.0.1:1";
    let store = Arc::new(MemoryStore::new());
    let elector = Elector::alone(name, unhurried()).expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let ahead = vec![Word::HandOver(1), Word::StepDown(Some(7)), Word::Stop];
    let looping = looping_behind(elector, Arc::clone(&store), standing, ahead);
    let text = looping.ended();
    assert_eq!(store.term(), 1, "{text}");
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 7,
            looked: true,
        }
    );
    assert_eq!(lines_with(&text, RETAKES), 1, "{text}");
    assert_eq!(left_under(&text), [None], "{text}");
    assert_eq!(lines_with(&text, ": leading"), 1, "{text}");
}

/// A channel closed behind a hand-over's leave ends the loop with no
/// further take, as a stop read there does. The replica alone leads under
/// term 1. A hand-over of that term stood on the channel, and the channel
/// was closed behind it, before the loop began: the loop leaves at the
/// word and finds the channel's end ahead of its next take. One take was
/// made, and the one leave's line names no term.
#[test]
fn a_channel_closed_behind_a_hand_over_ends_the_loop_with_no_further_take() {
    let name = "a@127.0.0.1:1";
    let store = Arc::new(MemoryStore::new());
    let elector = Elector::alone(name, unhurried()).expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let ahead = vec![Word::HandOver(1)];
    let looping = looping_on(elector, Arc::clone(&store), standing, ahead, true);
    let text = looping.ended();
    assert_eq!(store.term(), 1, "{text}");
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 1,
            looked: true,
        }
    );
    assert_eq!(left_under(&text), [None], "{text}");
    assert_eq!(lines_with(&text, ": leading"), 1, "{text}");
}

/// A leader whose lease moved after its last step finds it at its
/// stop's release: it releases nothing and says the lease moved under
/// the term it held, as a step that read the loss would. A replica that
/// held nothing at its stop says nothing of a move, and neither does a
/// holder whose release lands.
#[test]
fn a_stop_that_finds_the_lease_moved_says_so_and_releases_nothing() {
    const MOVED: &str = "the lease moved; the fence is closed";
    // A renew period of twenty seconds: the stop's release is the
    // first to read the lease after the take.
    let settings = ElectionSettings {
        duration: Duration::from_secs(60),
        renew_period: Duration::from_secs(20),
        renew_deadline: Duration::from_secs(40),
    };
    let store = Losable::default();
    let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", settings)
        .expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let leading = looping(holder, standing);
    until("the take", || stood.borrow().holder == "a@127.0.0.1:1");
    // Another candidate's record is stored over the holder's.
    let theirs = LeaseRecord {
        holder: "b@127.0.0.1:2".to_owned(),
        term: 2,
        duration_ms: 60_000,
        ..LeaseRecord::default()
    };
    store.put(&theirs);
    let (moved, text) = leading.stopped_with(MOVED);
    assert_eq!(moved, 1, "{text}");
    // The line carries the term this replica held, not the record's.
    let line = text
        .lines()
        .find(|line| line.contains(MOVED))
        .expect("the line");
    assert!(
        line.split_whitespace().any(|field| field == "term=1"),
        "{line}"
    );
    assert!(!text.contains("the lease is released"), "{text}");
    let mut lease = store.clone();
    assert_eq!(lease.read().expect("a read"), Some(theirs));

    let follower = Elector::new(Box::new(store.clone()), "c@127.0.0.1:3", settings)
        .expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let following = looping(follower, standing);
    until("the first look", || stood.borrow().looked);
    let (moved, text) = following.stopped_with(MOVED);
    assert_eq!(moved, 0, "{text}");

    // A holder that stops over a quiet store releases, and says nothing
    // of a move.
    let quiet = Elector::new(Box::new(MemoryLease::new()), "d@127.0.0.1:4", settings)
        .expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let released = looping(quiet, standing);
    until("the take", || stood.borrow().holder == "d@127.0.0.1:4");
    let (moved, text) = released.stopped_with(MOVED);
    assert_eq!(moved, 0, "{text}");
    assert!(text.contains("the lease is released"), "{text}");
}

/// A stop whose release comes back moved reads the lease again, and the
/// elector counts that record's term as at any step. The last standing
/// a stopping replica leaves counts it too: another replica's release
/// under term nine, stored after the holder's last step, is read by
/// the stop alone.
#[test]
fn a_release_read_again_at_a_stop_is_counted_by_the_last_standing() {
    let store = Losable::default();
    let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", unhurried())
        .expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let leading = looping(holder, standing);
    until("the take", || stood.borrow().holder == "a@127.0.0.1:1");
    assert_eq!(stood.borrow().term, 1);
    // The release under term nine is stored before the stop's own
    // write, which comes back moved; the read after it finds it.
    store.ahead([Ahead::Write(LeaseRecord {
        term: 9,
        ..LeaseRecord::default()
    })]);
    let (released, text) = leading.stopped_with("the lease is released");
    assert_eq!(released, 1, "{text}");
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 9,
            looked: true,
        },
        "{text}"
    );
}

/// One candidate takes the free lease over a store that fails its first
/// `failing` releases, the first of them after `stalled`, and is
/// stopped.
fn stopped_over(failing: u32, stalled: Duration, renew_period: Duration) -> Stop {
    let mut lease = MemoryLease::new();
    let releases = Arc::new(Mutex::new(Vec::new()));
    let store = FailsReleases {
        lease: lease.clone(),
        failing,
        stalled,
        releases: Arc::clone(&releases),
    };
    let settings = ElectionSettings {
        duration: renew_period * 6,
        renew_period,
        renew_deadline: renew_period * 4,
    };
    let candidate = Elector::new(Box::new(store), "a@127.0.0.1:1", settings)
        .expect("the settings are in order");
    let fence = candidate.fence();
    let (standing, _stood) = watch::channel(ElectionObservation::default());
    let running = start(candidate, Arc::new(MemoryStore::new()), standing).expect("a thread");
    until("the take", || fence.leader_term(Instant::now()) == Some(1));
    let stopped = Instant::now();
    running.stop();
    ended(running);
    let took = stopped.elapsed();
    let free = lease
        .read()
        .expect("a read")
        .is_some_and(|record| record.holder.is_empty());
    let releases = releases
        .lock()
        .iter()
        .map(|began| began.duration_since(stopped))
        .collect();
    Stop {
        releases,
        took,
        free,
    }
}

#[test]
fn a_release_that_fails_is_tried_again_a_renew_period_after_it_began() {
    unheld(
        "a release that fails",
        Duration::from_secs(1),
        failed_releases_tried_again,
    );
}

/// One try of the case above: a second of room on each upper bound.
fn failed_releases_tried_again() {
    // Every write fails, and the first takes the whole period: three
    // writes and no more, and the lease is left to expire by its
    // duration.
    let period = Duration::from_secs(2);
    let stop = stopped_over(u32::MAX, period, period);
    assert_eq!(stop.releases.len(), 3);
    assert!(!stop.free);
    // THE FLOOR, which no hold of the host breaks: a write begins no
    // sooner than a renew period after the one before began. The first
    // began after the stop, so the second is a period past the stop
    // and the third two.
    assert!(
        stop.releases[1] >= period && stop.releases[2] >= period * 2,
        "the releases began {:?} after the stop",
        stop.releases
    );
    // THE MOST, each with a second of room on this period. The write
    // that took the period is followed at once: the pause never adds
    // to a stall. And no pause follows the last attempt.
    bound!(
        stop.releases[1] < period + period / 2,
        "the second release began {:?} after the stop",
        stop.releases[1]
    );
    bound!(
        stop.took < period * 2 + period / 2,
        "the thread ended {:?} after its stop",
        stop.took
    );

    // Two failures: the third write releases.
    let stop = stopped_over(2, Duration::ZERO, Duration::from_millis(20));
    assert_eq!(stop.releases.len(), 3);
    assert!(stop.free, "the third write releases");

    // No failure: one write, and the thread ends with no pause.
    let stop = stopped_over(0, Duration::ZERO, period);
    assert_eq!(stop.releases.len(), 1);
    assert!(stop.free);
    bound!(
        stop.took < period,
        "the thread ended {:?} after its stop",
        stop.took
    );
}

#[test]
fn two_candidates_over_one_lease_lead_follow_and_hand_over_at_a_stop() {
    unheld(
        "two candidates over one lease",
        Duration::from_millis(600),
        two_candidates_hand_over_at_a_stop,
    );
}

/// One try of the case above: 600 ms of room, the lease's duration.
fn two_candidates_hand_over_at_a_stop() {
    let lease = MemoryLease::new();
    // The order of the design's settings at a fifth of their size.
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(50),
        renew_deadline: Duration::from_millis(400),
    };
    let candidate = |identity: &str| {
        Elector::new(Box::new(lease.clone()), identity, settings)
            .expect("the settings are in order")
    };
    let (first, second) = (candidate("a@127.0.0.1:1"), candidate("b@127.0.0.1:2"));
    let (first_fence, second_fence) = (first.fence(), second.fence());
    let fences = [&*first_fence, &*second_fence];
    let (first_store, second_store) = (Arc::new(MemoryStore::new()), Arc::new(MemoryStore::new()));
    let (first_standing, first_stood) = watch::channel(ElectionObservation::default());
    let (second_standing, second_stood) = watch::channel(ElectionObservation::default());

    // The first takes the free lease at its first step: the term is in
    // its store and its standing names itself.
    let first_running = start(first, Arc::clone(&first_store), first_standing).expect("a thread");
    wait_for("the first take", fences, || {
        first_stood.borrow().looked && first_fence.leader_term(Instant::now()) == Some(1)
    });
    assert_eq!(first_store.term(), 1);
    assert_eq!(first_stood.borrow().holder, "a@127.0.0.1:1");

    // The second follows: its standing names the first under its term.
    let second_running =
        start(second, Arc::clone(&second_store), second_standing).expect("a thread");
    wait_for("the second's first look", fences, || {
        second_stood.borrow().looked
    });
    // The bound on the clock: the first kept its lease to the second's
    // look. A host that held it past the lease's duration handed the
    // lease to the second, under the next term.
    let (holder, term) = {
        let look = second_stood.borrow();
        (look.holder.clone(), look.term)
    };
    bound!(
        holder == "a@127.0.0.1:1" && term == 1,
        "the second's first look names {holder:?} under term {term}"
    );
    assert_eq!(second_store.term(), 0, "a follower holds no term");

    // The planned stop: the first closes its fence, releases and ends;
    // the second takes the released lease under the next term.
    first_running.stop();
    // The thread opens its fence inside the take and writes its store's
    // term after it: the wait asks both.
    wait_for("the hand-over", fences, || {
        second_fence.leader_term(Instant::now()) == Some(2) && second_store.term() == 2
    });
    ended(first_running);
    assert_eq!(first_fence.leader_term(Instant::now()), None);
    assert_eq!(
        *first_stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 1,
            looked: true,
        },
        "a stopped replica names nobody"
    );

    // A stopped thread never takes again: the second keeps the lease
    // until its own stop, and the lease is then free.
    second_running.stop();
    ended(second_running);
    assert_eq!(second_fence.leader_term(Instant::now()), None);
    let mut reader = candidate("c@127.0.0.1:3");
    reader.tick(Instant::now()).expect("a look");
    assert_eq!(
        reader.observed().map(|record| record.term),
        Some(3),
        "the free lease is taken at once under the next term"
    );
}
