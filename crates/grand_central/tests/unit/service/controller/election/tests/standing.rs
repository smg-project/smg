//! The standing a look leaves: whom it names behind a failed look, a fallen
//! term and a lost lease, the term it counts that was read at the look that
//! loses the lease, in a release read again after a moved write, or at a step
//! that then failed, and a record lost under a live holder.

use super::*;

/// A lease whose reads fail while `away` is set.
struct Away {
    lease: MemoryLease,
    away: Arc<AtomicBool>,
}

impl LeaseBackend for Away {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        if self.away.load(Ordering::SeqCst) {
            return Err(ElectError::Io {
                place: self.place(),
                source: std::io::Error::other("the store is away"),
            });
        }
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        self.lease.write(expected, record)
    }

    fn place(&self) -> String {
        self.lease.place()
    }
}

#[test]
fn a_look_that_fails_names_nobody_until_one_lands() {
    unheld(
        "a look that fails",
        Duration::from_millis(600),
        a_failed_look_names_nobody,
    );
}

/// One try of the case above: 600 ms of room, the lease's duration.
fn a_failed_look_names_nobody() {
    let lease = MemoryLease::new();
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(50),
        renew_deadline: Duration::from_millis(400),
    };
    let away = Arc::new(AtomicBool::new(false));
    let leader = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", settings)
        .expect("the settings are in order");
    let follower = Elector::new(
        Box::new(Away {
            lease: lease.clone(),
            away: Arc::clone(&away),
        }),
        "b@127.0.0.1:2",
        settings,
    )
    .expect("the settings are in order");
    let (leader_fence, follower_fence) = (leader.fence(), follower.fence());
    let fences = [&*leader_fence, &*follower_fence];
    let (leader_standing, _led) = watch::channel(ElectionObservation::default());
    let (follower_standing, mut followed) = watch::channel(ElectionObservation::default());
    let leading = start(leader, Arc::new(MemoryStore::new()), leader_standing).expect("a thread");
    wait_for("the take", fences, || {
        leader_fence.leader_term(Instant::now()) == Some(1)
    });
    let following =
        start(follower, Arc::new(MemoryStore::new()), follower_standing).expect("a thread");
    // The bound on the clock, here and at the last wait: the holder
    // kept its lease. A host that held it past the lease's duration,
    // right after a look of the follower's, handed the lease to the
    // follower: its standing then reads the next term, and the wait
    // ends on that too.
    wait_for("the follower's look", fences, || {
        followed.borrow().holder == "a@127.0.0.1:1" || followed.borrow().term > 1
    });
    bound!(
        followed.borrow().term == 1,
        "the follower took the lease before its first look landed"
    );

    // The follower's store goes away. The second publish from here is
    // of a step that began after it did: that look failed, and it names
    // nobody under the term the looks before it saw.
    away.store(true, Ordering::SeqCst);
    for _ in 0..2 {
        followed.mark_unchanged();
        wait_for("a publish", fences, || {
            followed.has_changed().expect("the thread lives")
        });
    }
    assert_eq!(
        *followed.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 1,
            looked: true,
        }
    );

    // The store is back: the next look names the holder again.
    away.store(false, Ordering::SeqCst);
    wait_for("a look that lands", fences, || {
        followed.borrow().holder == "a@127.0.0.1:1" || followed.borrow().term > 1
    });
    bound!(
        followed.borrow().term == 1,
        "the follower took the lease before a look landed"
    );
    following.stop();
    ended(following);
    leading.stop();
    ended(leading);
}

#[test]
fn a_lease_whose_term_fell_names_nobody_under_the_highest_term_seen() {
    // A store whose term went back, as no elector writes it: a Lease
    // deleted or replaced, a lease file restored. The look keeps the
    // highest term it saw and names nobody: no write stored the new
    // holder under that term.
    let held = |holder: &str, term: u64| LeaseRecord {
        holder: holder.to_owned(),
        term,
        duration_ms: 60_000,
        ..LeaseRecord::default()
    };
    let mut lease = MemoryLease::new();
    let first = held("x@127.0.0.1:8", 5);
    let (fallen, back) = (held("y@127.0.0.1:9", 3), held("y@127.0.0.1:9", 5));
    assert_eq!(
        lease.write(None, &first).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(400),
    };
    let follower = Elector::new(Box::new(lease.clone()), "b@127.0.0.1:2", settings)
        .expect("the settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let running = start(follower, Arc::new(MemoryStore::new()), standing).expect("a thread");
    until("the first look", || stood.borrow().holder == first.holder);
    assert_eq!(stood.borrow().term, 5);

    assert_eq!(
        lease.write(Some(&first), &fallen).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    until("the look at the fallen term", || {
        stood.borrow().holder.is_empty()
    });
    assert_eq!(stood.borrow().term, 5);

    // A holder again at the first look that reads a term at the highest.
    assert_eq!(
        lease.write(Some(&fallen), &back).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    until("the look at the term come back", || {
        stood.borrow().holder == back.holder
    });
    assert_eq!(stood.borrow().term, 5);
    running.stop();
    ended(running);
}

/// A lease lost to another holder names that holder on the standing at
/// the look that reads the loss: only a loss to nobody reads as no
/// record.
#[test]
fn a_lease_lost_to_another_holder_names_that_holder_at_the_look_at_the_loss() {
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(400),
    };
    let store = Losable::default();
    let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", settings)
        .expect("the settings are in order");
    let fence = holder.fence();
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping(holder, standing);
    until("the fence's opening", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    // Another candidate's record is stored over the holder's, under the
    // next term. The first look that no longer names the holder read
    // the loss, and names the lease's holder.
    store.put(&LeaseRecord {
        holder: "b@127.0.0.1:2".to_owned(),
        term: 2,
        duration_ms: 60_000,
        ..LeaseRecord::default()
    });
    until("the look at the loss", || {
        stood.borrow().holder != "a@127.0.0.1:1"
    });
    let at_the_loss = stood.borrow().clone();
    let (_, text) = looping.stopped_with("the lease");
    assert_eq!(at_the_loss.holder, "b@127.0.0.1:2", "{text}");
    assert_eq!(at_the_loss.term, 2, "{text}");
}

/// A lease lost to nobody under a higher term. The tick of the look
/// that loses it says the term this replica held; the record it read,
/// a release another replica wrote, says the release's own. The
/// standing counts the record's, so a holder restored under a term
/// between the two is not named.
#[test]
fn a_term_read_at_the_look_that_loses_the_lease_stands_over_a_holder_restored_under_it() {
    let store = Losable::default();
    let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", brisk())
        .expect("the settings are in order");
    let fence = holder.fence();
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping(holder, standing);
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    // Between two looks another replica takes the lease and gives it up
    // under term nine: the next look loses the lease and names nobody.
    // Before the look after it a holder is restored under term seven.
    store.put_before_the_next_looks([
        LeaseRecord {
            term: 9,
            ..LeaseRecord::default()
        },
        LeaseRecord {
            holder: "x@127.0.0.1:9".to_owned(),
            term: 7,
            duration_ms: 60_000,
            ..LeaseRecord::default()
        },
    ]);
    until("the look at the restored holder", || {
        stood.borrow().holder == "x@127.0.0.1:9"
            || looping.said("the lease's term is under the highest")
    });
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 9,
            looked: true,
        }
    );
    let (fallen, text) = looping.stopped_with("the lease's term is under the highest");
    assert_eq!(fallen, 1, "{text}");
}

/// The same release seen only where a renewal's write came back moved.
/// It lands between the holder's read and its write, and the lease is
/// read again: the elector observes that record, so the standing counts
/// its term as at a plain read.
#[test]
fn a_release_read_again_after_a_moved_write_stands_over_a_holder_restored_under_it() {
    let store = Losable::default();
    let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", brisk())
        .expect("the settings are in order");
    let fence = holder.fence();
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping(holder, standing);
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    // The release under term nine is stored before the renewal's write;
    // the read after that write finds it, and the look after that one
    // finds a holder restored under term seven.
    store.ahead([
        Ahead::Write(LeaseRecord {
            term: 9,
            ..LeaseRecord::default()
        }),
        Ahead::Read(None),
        Ahead::Read(Some(LeaseRecord {
            holder: "x@127.0.0.1:9".to_owned(),
            term: 7,
            duration_ms: 60_000,
            ..LeaseRecord::default()
        })),
    ]);
    until("the look at the restored holder", || {
        stood.borrow().holder == "x@127.0.0.1:9"
            || looping.said("the lease's term is under the highest")
    });
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 9,
            looked: true,
        }
    );
    let (fallen, text) = looping.stopped_with("the lease's term is under the highest");
    assert_eq!(fallen, 1, "{text}");
}

/// A step that read a term and then failed. A follower reads a release
/// under term nine and its take's write fails, step after step: each
/// step is an error, and the elector counted the term it read. The
/// standing counts it with every step's, while the steps fail.
#[test]
fn a_term_read_at_a_step_that_then_failed_is_counted_while_the_steps_fail() {
    let store = Losable::default();
    store.put(&LeaseRecord {
        holder: "x@127.0.0.1:9".to_owned(),
        term: 3,
        duration_ms: 60_000,
        ..LeaseRecord::default()
    });
    let follower = Elector::new(Box::new(store.clone()), "b@127.0.0.1:2", brisk())
        .expect("the settings are in order");
    let (standing, mut stood) = watch::channel(ElectionObservation::default());
    let looping = looping(follower, standing);
    until("the holder is followed", || stood.borrow().term == 3);
    store.fail_writes(true);
    store.put_before_the_next_looks([LeaseRecord {
        term: 9,
        ..LeaseRecord::default()
    }]);
    until("the failing streak", || {
        looping.said("an election step failed")
    });
    looks(&mut stood, 2);
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 9,
            looked: true,
        }
    );
    let (failed, text) = looping.stopped_with("an election step failed");
    assert_eq!(failed, 1, "{text}");
}

/// A step that reads the lease twice. A holder told to leave whose
/// release is still owed reads another holder under term nine; its
/// release's write comes back moved over a holder restored under term
/// seven, which the read after it finds. The term is read before
/// anybody is named: that step names nobody, under term nine.
#[test]
fn a_step_that_reads_a_term_and_then_a_holder_restored_under_it_names_nobody() {
    const FALLEN: &str = "the lease's term is under the highest";
    let held = |holder: &str, term: u64| LeaseRecord {
        holder: holder.to_owned(),
        term,
        duration_ms: 60_000,
        ..LeaseRecord::default()
    };
    let store = Losable::default();
    let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", brisk())
        .expect("the settings are in order");
    let fence = holder.fence();
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let looping = looping(holder, standing);
    until("the first lead", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    // The word's release fails and stays owed: every step tries it
    // again in place of a renewal.
    store.fail_writes(true);
    looping
        .stop
        .send(Word::StepDown(None))
        .expect("the loop runs");
    until("the failed release", || {
        looping.said("the release failed and is tried again")
    });
    // The next step's read finds the holder under term nine, and its
    // release's write the holder restored under term seven. The script
    // is one call and its write lands under the failing mode: no step
    // of the loop reads the first with its own write still failing.
    store.ahead([
        Ahead::Read(Some(held("z@127.0.0.1:9", 9))),
        Ahead::Write(held("y@127.0.0.1:7", 7)),
    ]);
    until("the step that read both", || stood.borrow().term == 9);
    assert_eq!(
        *stood.borrow(),
        ElectionObservation {
            holder: String::new(),
            term: 9,
            looked: true,
        }
    );
    until("the line of the fallen term", || looping.said(FALLEN));
    let (fallen, text) = looping.stopped_with(FALLEN);
    assert_eq!(fallen, 1, "{text}");
}

/// A record lost under a live holder, by no release. The holder's fence
/// is open until its own next step reads the lease; a replica that saw
/// the record held waits that record's duration from its first sight of
/// the loss, so no instant has two open fences, and it names nobody
/// while it waits. Two election threads over one store, the standby's
/// steps half a period behind the holder's. Both wait the lost record
/// out and the first whose wait ends takes it: the standby on an idle
/// host, the old holder where the standby's thread is held. The case
/// reads the first take by either.
#[test]
fn a_record_lost_under_a_live_holder_is_taken_only_after_its_duration() {
    unheld(
        "a lost record",
        Duration::from_millis(400),
        a_lost_record_is_waited_out,
    );
}

/// One try of the case above: 400 ms of room, the renew deadline, the
/// shortest stop that breaks one of its bounds.
fn a_lost_record_is_waited_out() {
    let store = Losable::default();
    let settings = ElectionSettings {
        duration: Duration::from_millis(800),
        renew_period: Duration::from_millis(200),
        renew_deadline: Duration::from_millis(400),
    };
    let (holder, standby) = ("a@127.0.0.1:1", "b@127.0.0.1:2");
    let leader =
        Elector::new(Box::new(store.clone()), holder, settings).expect("the settings are in order");
    let follower = Elector::new(Box::new(store.clone()), standby, settings)
        .expect("the settings are in order");
    let (leader_fence, follower_fence) = (leader.fence(), follower.fence());
    let (leader_standing, led) = watch::channel(ElectionObservation::default());
    let (follower_standing, followed) = watch::channel(ElectionObservation::default());
    let leading = start(leader, Arc::new(MemoryStore::new()), leader_standing).expect("a thread");
    until("the take", || {
        leader_fence.leader_term(Instant::now()) == Some(1)
    });
    let renewals = store.landed();
    until("a renewal", || store.landed() > renewals);
    thread::sleep(settings.renew_period / 2);
    let following =
        start(follower, Arc::new(MemoryStore::new()), follower_standing).expect("a thread");
    until("the standby's look", || followed.borrow().holder == holder);

    // The loss, right after one of the holder's steps.
    let renewals = store.landed();
    until("a renewal", || store.landed() > renewals);
    // The bound on the clock: the holder led to the loss. A host that
    // held it a renew deadline closed its fence, and one that held it
    // past the lease's duration handed the lease on: the record lost
    // is then of a later term.
    bound!(
        leader_fence.leader_term(Instant::now()) == Some(1),
        "the holder did not lead under term 1 at the loss"
    );
    let lost = Instant::now();
    store.lose();
    // Both fences through the loss, the first read on both sides of the
    // second as `wait_for` reads them, until a renew deadline after the
    // first take under the next term: a holder's fence closes by itself
    // inside that. `unnamed` is each replica seen with its fence closed
    // and nobody named: the old holder first, the standby second.
    let (mut samples, mut both, mut unnamed) = (0_u32, 0_u32, (false, false));
    // The first take, under whatever later term: how long after the
    // loss its read was asked for, its term, whether the standby's, and
    // that instant. A fence answers a term only inside its window, so
    // that instant is no sooner than the take, and a stop between a
    // read and the line that dates it lengthens nothing.
    let mut taken: Option<(Duration, u64, bool, Instant)> = None;
    while taken.is_none_or(|(_, _, _, seen)| seen.elapsed() < settings.renew_deadline) {
        let asked_before = Instant::now();
        let before = leader_fence.leader_term(asked_before);
        let asked_between = Instant::now();
        let between = follower_fence.leader_term(asked_between);
        let asked_after = Instant::now();
        let after = leader_fence.leader_term(asked_after);
        samples += 1;
        both += u32::from(before.is_some() && between.is_some() && before == after);
        unnamed.0 |= before.is_none() && led.borrow().holder.is_empty();
        unnamed.1 |= between.is_none() && followed.borrow().holder.is_empty();
        // The loop runs to its end whatever that term: the bound on it is
        // read after the count of two open fences, never ahead of it.
        let later = [
            (before, asked_before),
            (between, asked_between),
            (after, asked_after),
        ]
        .into_iter()
        .find_map(|(term, asked)| term.filter(|term| *term > 1).map(|term| (term, asked)));
        if let (None, Some((term, asked))) = (taken, later) {
            taken = Some((
                asked.duration_since(lost),
                term,
                between == Some(term),
                asked,
            ));
        }
        assert!(
            lost.elapsed() < Duration::from_secs(20),
            "neither replica took the lost record in 20 s"
        );
        thread::sleep(Duration::from_micros(200));
    }
    let (took, term, by_standby, _) = taken.expect("a replica leads under a later term");
    let (taker, named_nobody, other) = if by_standby {
        (standby, unnamed.1, &led)
    } else {
        (holder, unnamed.0, &followed)
    };
    // No hold excuses two open fences or a take sooner than the lost
    // record's duration: both are read before the bound.
    assert_eq!(
        both, 0,
        "two open fences in {both} of {samples} samples after the loss"
    );
    assert!(
        took >= settings.duration,
        "{taker} led {took:?} after the loss, sooner than the lost record's duration"
    );
    // The other bound on the clock: the first take read is the next
    // term's. A host that held the test past the lease's duration
    // handed the lease on once more before a fence was read under
    // that term, and the first take read is then a later one.
    bound!(
        term == 2,
        "the first take read after the loss leads under term {term}"
    );
    // The page's row reads this line.
    writeln!(
        std::io::stdout(),
        "LOST_RECORD samples={samples} both_open={both} took_ms={} by={taker}",
        took.as_millis()
    )
    .expect("stdout");
    assert!(named_nobody, "{taker} named a holder through its wait");
    // The bound on the clock once more: the other replica follows the
    // taker from its first look after the take, under the take's term:
    // a look from before the loss names the old holder under term 1
    // and ends nothing. A host that held the test past the lease's
    // duration handed the lease on, and the other replica then reads a
    // later term: the wait ends on that too.
    until("the other replica follows the taker", || {
        let look = other.borrow();
        (look.holder == taker && look.term == 2) || look.term > 2
    });
    let read = other.borrow().term;
    bound!(
        read == 2,
        "the other replica reads term {read}, past the taker's"
    );
    following.stop();
    ended(following);
    leading.stop();
    ended(leading);
}
