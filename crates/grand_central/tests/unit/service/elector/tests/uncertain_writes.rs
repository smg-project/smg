//! A write whose outcome its writer cannot know. Every case here runs
//! over a store whose write stalls, loses a race, fails, lands with its
//! answer lost, or lands late.

use super::*;

/// A backend whose first write runs what happens between this
/// candidate's read and its write.
struct Stalled {
    lease: MemoryLease,
    between: Option<Box<dyn FnOnce() + Send>>,
}

impl LeaseBackend for Stalled {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        if let Some(between) = self.between.take() {
            between();
        }
        self.lease.write(expected, record)
    }

    fn place(&self) -> String {
        "stalled".to_string()
    }
}

#[test]
fn a_take_that_lands_after_a_later_holders_record_went_opens_after_that_holders_fence() {
    // b reads an absent lease and stalls before its write. c takes the
    // lease meanwhile and leads, and its record is removed: b's write
    // lands over an absent record again. c leads to one renew deadline
    // past its last renewal, which b never saw, so b's lead opens one
    // renew deadline after its own write returned, by the clock: no
    // instant has both fences open.
    let settings = ElectionSettings {
        duration: Duration::from_millis(60),
        renew_period: Duration::from_millis(10),
        renew_deadline: Duration::from_millis(40),
    };
    let lease = MemoryLease::new();
    let mut c = Elector::new(Box::new(lease.clone()), "c@host:3", settings).expect("settings");
    let c_fence = c.fence();
    let removed = lease.clone();
    let (last, renewed) = mpsc::channel();
    let between = move || {
        let take = c.tick(Instant::now()).expect("take");
        assert_eq!(take, TickOutcome::Acquired { term: 1 });
        let mut rounds = 0;
        let led = loop {
            thread::sleep(settings.renew_period);
            let now = Instant::now();
            assert_eq!(
                c.tick(now).expect("renew"),
                TickOutcome::Renewed { term: 1 }
            );
            if c.fence().leader_term(now).is_some() {
                break now;
            }
            rounds += 1;
            assert!(rounds < 100, "the other holder never led");
        };
        last.send(led).expect("the case waits");
        removed.lose();
    };
    let backend = Stalled {
        lease: lease.clone(),
        between: Some(Box::new(between)),
    };
    let mut b = Elector::new(Box::new(backend), "b@host:2", settings).expect("settings");
    let take = b.tick(Instant::now()).expect("take");
    let returned = Instant::now();
    assert_eq!(take, TickOutcome::Acquired { term: 1 });
    let renewed = renewed.recv().expect("the other holder's last renewal");
    // b renews while c may still lead, and reads closed to the end of
    // c's fence.
    let within = renewed + settings.renew_deadline / 2;
    assert_eq!(
        b.tick(within).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(c_fence.leader_term(within), Some(1));
    assert_eq!(b.fence().leader_term(within), None);
    let closes = renewed + settings.renew_deadline;
    assert_eq!(c_fence.leader_term(closes), Some(1));
    assert_eq!(b.fence().leader_term(closes), None);
    // b's lead opens past it, one renew deadline after its write
    // returned: the step that returned is no earlier than that write.
    let opens = returned + settings.renew_deadline;
    assert!(opens > closes);
    assert_eq!(
        b.tick(opens).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(b.fence().leader_term(opens), Some(1));
    assert_eq!(c_fence.leader_term(opens), None);
}

/// A backend that lets another candidate write between this candidate's
/// read and its write: the race a compare-and-swap exists for. A case
/// may set the rival after the candidate's take.
struct Interposed {
    lease: MemoryLease,
    rival: Arc<Mutex<Option<LeaseRecord>>>,
}

impl LeaseBackend for Interposed {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        if let Some(rival) = self.rival.lock().take() {
            assert_eq!(
                self.lease
                    .write(expected, &rival)
                    .expect("the rival's write"),
                ConditionalWriteOutcome::Done
            );
        }
        self.lease.write(expected, record)
    }

    fn place(&self) -> String {
        "interposed".to_string()
    }
}

#[test]
fn a_take_the_rival_won_is_a_race_and_the_next_tick_follows() {
    let lease = MemoryLease::new();
    let rival = LeaseRecord {
        holder: "r@host:9".to_string(),
        term: 1,
        acquired_at_micros: 7,
        renewed_at_micros: 7,
        duration_ms: 3_000,
    };
    let backend = Interposed {
        lease: lease.clone(),
        rival: Arc::new(Mutex::new(Some(rival.clone()))),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("tick"), TickOutcome::Raced);
    assert!(!a.is_holder());
    assert!(!a.waits_to_lead(), "a take a rival won waits for no lead");
    assert_eq!(a.fence().leader_term(t0), None);
    assert_eq!(
        a.tick(t0 + S).expect("tick"),
        TickOutcome::Following {
            holder: "r@host:9".to_string(),
            term: 1
        }
    );
    assert_eq!(lease.clone().read().expect("read"), Some(rival));
}

/// A holder over a store where another replica's write lands between
/// its next read and its next write: `theirs` makes that record from
/// the holder's own, and it is stored one wall instant after it.
fn led_before_a_rivals_write(
    t0: Instant,
    theirs: impl FnOnce(LeaseRecord) -> LeaseRecord,
) -> Elector {
    let lease = MemoryLease::new();
    let rival = Arc::new(Mutex::new(None));
    let backend = Interposed {
        lease: lease.clone(),
        rival: Arc::clone(&rival),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    a.lead_at(t0);
    let mine = lease.clone().read().expect("read").expect("the record");
    *rival.lock() = Some(LeaseRecord {
        renewed_at_micros: mine.renewed_at_micros + 1,
        ..theirs(mine)
    });
    a
}

/// A holder over a store where another replica's release under term
/// nine lands between its next read and its next write.
fn led_before_a_rivals_release(t0: Instant) -> Elector {
    led_before_a_rivals_write(t0, |mine| LeaseRecord {
        holder: String::new(),
        term: 9,
        duration_ms: 0,
        ..mine
    })
}

#[test]
fn a_release_that_lands_under_a_renewals_write_is_observed_at_the_read_after_it() {
    // The renewal's write comes back moved and the lease is read again.
    // That record is a sight as any other: the run counts its term, so
    // the standing reads it and the next take is above it.
    let t0 = Instant::now() + AHEAD;
    let mut a = led_before_a_rivals_release(t0);
    assert_eq!(
        a.tick(t0 + S).expect("the look that loses the lease"),
        TickOutcome::Lost {
            term: 1,
            holder: String::new()
        }
    );
    assert_eq!(a.observed().map(|record| record.term), Some(9));
    assert_eq!(a.highest_term(), 9);
    assert_eq!(
        a.tick(t0 + 2 * S).expect("take"),
        TickOutcome::Acquired { term: 10 }
    );
}

#[test]
fn a_release_that_lands_under_a_stops_write_is_observed_at_the_read_after_it() {
    // The same race at a stop: the release's write comes back moved, the
    // lease read again is released under a term above this run's, and
    // the run counts that term.
    let t0 = Instant::now() + AHEAD;
    let mut a = led_before_a_rivals_release(t0);
    assert!(a.release().expect("the release"));
    assert!(!a.is_holder());
    assert_eq!(a.observed().map(|record| record.term), Some(9));
    assert_eq!(a.highest_term(), 9);
}

#[test]
fn a_take_that_lands_under_a_renewals_write_is_observed_at_the_read_after_it() {
    // The rival's write is a take, a held record under the next term.
    // The look that loses the lease names its holder; the run counts
    // its term and keeps the record, and its wait for that record
    // counts from the read that found it.
    let t0 = Instant::now() + AHEAD;
    let mut a = led_before_a_rivals_write(t0, |mine| LeaseRecord {
        holder: "b@host:2".to_owned(),
        term: 2,
        ..mine
    });
    assert_eq!(
        a.tick(t0 + S).expect("the look that loses the lease"),
        TickOutcome::Lost {
            term: 1,
            holder: "b@host:2".to_owned()
        }
    );
    assert_eq!(
        a.observed()
            .map(|record| (record.holder.as_str(), record.term)),
        Some(("b@host:2", 2))
    );
    assert_eq!(a.highest_term(), 2);
    // The rival's record stood still for its duration since that read:
    // a look a microsecond under the wait's end still follows it, and
    // the take at its end is above the term counted.
    assert_eq!(
        a.tick(t0 + 4 * S - Duration::from_micros(1))
            .expect("a look inside the wait"),
        TickOutcome::Following {
            holder: "b@host:2".to_owned(),
            term: 2
        }
    );
    assert_eq!(
        a.tick(t0 + 4 * S).expect("the take after the wait"),
        TickOutcome::Acquired { term: 3 }
    );
}

/// A backend whose writes fail while `fails` is above zero, as a stalled
/// file system or an API server away would.
struct Failing {
    lease: MemoryLease,
    fails: Arc<AtomicU32>,
}

impl LeaseBackend for Failing {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        if self.fails.load(Ordering::Relaxed) > 0 {
            self.fails.fetch_sub(1, Ordering::Relaxed);
            return Err(ElectError::Io {
                place: self.place(),
                source: std::io::Error::other("the backend is away"),
            });
        }
        self.lease.write(expected, record)
    }

    fn place(&self) -> String {
        "failing".to_string()
    }
}

#[test]
fn a_failed_renewal_leaves_the_holder_its_record_and_the_fence_to_its_deadline() {
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(0));
    let backend = Failing {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    fails.store(1, Ordering::Relaxed);
    let err = a.tick(t0 + S).expect_err("the backend is away");
    assert_eq!(err.to_string(), "the lease at failing: the backend is away");
    assert!(a.is_holder(), "an error changes nothing");
    assert_eq!(a.observed().expect("the record").holder, "a@host:1");
    let fence = a.fence();
    assert_eq!(
        fence.leader_term(t0 + S),
        Some(1),
        "the fence runs to its deadline"
    );
    assert_eq!(
        fence.leader_term(t0 + 2 * S + Duration::from_micros(1)),
        None,
        "and closes on its own past it"
    );
    assert_eq!(
        a.tick(t0 + 2 * S).expect("the backend is back"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(
        fence.leader_term(t0 + 4 * S),
        Some(1),
        "reopened by the renewal"
    );
    assert_eq!(
        lease
            .clone()
            .read()
            .expect("read")
            .expect("the record")
            .holder,
        "a@host:1"
    );
}

#[test]
fn a_take_whose_renewals_do_not_land_never_leads() {
    // The lead of a take with no holder to wait for opens one renew
    // deadline after the take, and a fence runs one renew deadline past
    // its last renewal: with no renewal landed the two meet, and no
    // instant reads this term as the leader's.
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(0));
    let backend = Failing {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    fails.store(2, Ordering::Relaxed);
    a.tick(t0 + S).expect_err("the backend is away");
    a.tick(t0 + 2 * S).expect_err("the backend is away");
    let fence = a.fence();
    for after in [0, 2_000_000, 2_000_001, 5_000_000] {
        let at = t0 + Duration::from_micros(after);
        assert_eq!(fence.leader_term(at), None, "{after} us after the take");
    }
    assert!(!a.waits_to_lead(), "the wait ended at its instant");
    // A renewal that lands after all leads from its own instant, as a
    // renewal past a fence that closed by itself does.
    let late = t0 + 2 * S + Duration::from_millis(500);
    assert_eq!(
        a.tick(late).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(fence.leader_term(late - Duration::from_micros(1)), None);
    assert_eq!(fence.leader_term(late), Some(1));
    assert_eq!(fence.leader_term(late + 2 * S), Some(1), "at the deadline");
}

#[test]
fn a_take_whose_write_failed_waits_for_nothing_while_it_follows_a_rival() {
    // The wait belongs to a take that landed with its answer. An attempt
    // in doubt is no take: the reader says no from the failed write on,
    // and through the steps that follow the rival that took the lease.
    let lease = MemoryLease::new();
    let backend = Failing {
        lease: lease.clone(),
        fails: Arc::new(AtomicU32::new(1)),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let mut b = candidate(&lease, "b@host:2");
    let t0 = Instant::now() + AHEAD;
    a.tick(t0).expect_err("the backend is away");
    assert!(!a.waits_to_lead(), "an attempt in doubt is no take");
    assert_eq!(b.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    assert!(b.waits_to_lead());
    assert!(matches!(
        a.tick(t0 + S).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    assert!(!a.waits_to_lead(), "a follower waits for no lead");
}

#[test]
fn a_failed_release_keeps_the_record_for_a_retry_and_the_fence_closed() {
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(0));
    let backend = Failing {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.tick(t0).expect("acquire");
    fails.store(1, Ordering::Relaxed);
    a.release().expect_err("the backend is away");
    assert!(a.is_holder(), "kept for the retry");
    assert_eq!(a.fence().leader_term(t0), None, "closed by the first call");
    assert!(a.release().expect("the retry"), "released");
    assert!(!a.is_holder());
    assert!(lease
        .clone()
        .read()
        .expect("read")
        .expect("the record")
        .is_released());
    assert!(a.observed().expect("the last write").is_released());
}

#[test]
fn a_term_whose_write_failed_goes_into_the_release_at_the_next_word() {
    // The write of nine over the release fails: the term is counted and
    // the record stays under five. A lower word, seven, then writes the
    // highest term this run saw and not its own: the candidate that
    // takes next takes above nine.
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(0));
    let backend = Failing {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    let mut b = candidate(&lease, "b@host:2");
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    let word = t0 + S / 2;
    assert!(a.step_down(Some(5), word).expect("the release"));
    fails.store(1, Ordering::Relaxed);
    a.step_down(Some(9), word).expect_err("the backend is away");
    let stored = lease.clone().read().expect("read").expect("the record");
    assert_eq!((stored.term, a.highest_term()), (5, 9));
    assert!(!a.step_down(Some(7), word).expect("the highest is written"));
    let stored = lease.clone().read().expect("read").expect("the record");
    assert_eq!(
        (stored.holder.as_str(), stored.term, stored.duration_ms),
        ("", 9, 0)
    );
    assert_eq!(
        b.tick(word).expect("take"),
        TickOutcome::Acquired { term: 10 }
    );
}

#[test]
fn a_term_said_over_an_empty_record_writes_nothing() {
    // An empty record, released under term zero, is no holder's
    // release: a manifest put under a live holder reads so, and the
    // take over it opens its fence a renew deadline late for a holder
    // the taker never saw. A candidate whose take over it failed is
    // told a term: the term is counted and the record stays empty.
    // Written under the term it would read as a release, and the next
    // take would lead at once.
    let mut lease = MemoryLease::new();
    let empty = LeaseRecord::default();
    assert_eq!(
        lease.write(None, &empty).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let fails = Arc::new(AtomicU32::new(1));
    let backend = Failing {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut c = Elector::new(Box::new(backend), "c@host:3", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    c.tick(t0).expect_err("the backend is away");
    assert!(!c.step_down(Some(5), t0).expect("the word"));
    assert_eq!(lease.clone().read().expect("read"), Some(empty));
    assert_eq!(c.highest_term(), 5);
    // The next take is still a take over an empty record: at once, and
    // its fence a renew deadline late.
    let mut d = candidate(&lease, "d@host:4");
    assert_eq!(d.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    let fence = d.fence();
    assert_eq!(fence.leader_term(t0), None);
    assert_eq!(
        d.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(fence.leader_term(t0 + S), None, "inside the renew deadline");
    assert_eq!(
        d.tick(t0 + 2 * S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(fence.leader_term(t0 + 2 * S), Some(1));
}

#[test]
fn a_term_told_while_a_take_is_in_doubt_stands_over_the_take_that_landed() {
    // The take lands and its answer is lost: the run holds nothing it
    // knows of when a term of seven is said, and the term is counted.
    // The step that finds the take landed under term one leaves the
    // lease as the word would have: no step leads under a term below
    // the highest its run saw.
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(1));
    let backend = LandsThenErrors {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.tick(t0).expect_err("the answer was lost");
    assert!(!a.step_down(Some(7), t0).expect("no holder is known"));
    assert_eq!(
        a.tick(t0 + S).expect("the landed take is left"),
        TickOutcome::Waiting { term: 7 }
    );
    assert!(!a.is_holder());
    assert_eq!(a.fence().leader_term(t0 + S), None);
    // The take had no holder to wait for and never led: its release
    // says the term told and hands the take's wait on.
    let left = lease.clone().read().expect("read").expect("the record");
    assert!(left.is_released());
    assert_eq!((left.term, left.duration_ms), (7, 2_000));
    // The run takes nothing inside its hold, one lease duration from the
    // step that left, and leads above the term said after it.
    assert_eq!(
        a.tick(t0 + 4 * S - Duration::from_micros(1))
            .expect("inside the hold"),
        TickOutcome::Waiting { term: 7 }
    );
    assert_eq!(
        a.tick(t0 + 4 * S).expect("the take above"),
        TickOutcome::Acquired { term: 8 }
    );
    assert_eq!(a.fence().leader_term(t0 + 4 * S), Some(8));
}

#[test]
fn a_release_that_fails_at_the_word_is_tried_again_at_every_step_and_never_renewed_over() {
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(0));
    let backend = Failing {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    fails.store(2, Ordering::Relaxed);
    let word = t0 + S / 2;
    a.step_down(Some(7), word).expect_err("the backend is away");
    assert!(a.is_holder(), "kept for the retry");
    assert_eq!(a.fence().leader_term(word), None, "closed at the word");
    // The next step is the release again, not a renewal: the fence stays
    // closed and the record is not renewed under the term it left.
    a.tick(t0 + S).expect_err("the backend is still away");
    assert_eq!(a.fence().leader_term(t0 + S), None);
    let stored = lease.clone().read().expect("read").expect("the record");
    assert_eq!(
        stored.renewed_at_micros,
        a.observed().expect("its record").renewed_at_micros
    );
    assert_eq!(
        a.tick(t0 + 2 * S).expect("the release lands"),
        TickOutcome::Waiting { term: 7 }
    );
    let stored = lease.clone().read().expect("read").expect("the record");
    assert_eq!((stored.holder.as_str(), stored.term), ("", 7));
    // The hold counts from the word, not from the release that landed.
    assert_eq!(
        a.tick(word + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Waiting { term: 7 }
    );
    assert_eq!(
        a.tick(word + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 8 }
    );
}

#[test]
fn a_release_under_a_told_term_that_landed_but_errored_reads_as_released_at_the_next_step() {
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(0));
    let backend = LandsThenErrors {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    fails.store(1, Ordering::Relaxed);
    let word = t0 + S / 2;
    a.step_down(Some(7), word).expect_err("the answer was lost");
    assert!(a.is_holder(), "kept for the retry");
    let stored = lease.clone().read().expect("read").expect("the record");
    assert_eq!(
        (stored.holder.as_str(), stored.term),
        ("", 7),
        "the release landed"
    );
    // The next step finds its own release under the term it was told:
    // the lease is given up, and it waits its turn.
    assert_eq!(
        a.tick(t0 + S).expect("look"),
        TickOutcome::Waiting { term: 7 }
    );
    assert!(!a.is_holder());
    assert_eq!(
        a.tick(word + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 8 }
    );
}

#[test]
fn a_leaving_holder_whose_lease_was_taken_before_its_release_landed_has_lost_it() {
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(0));
    let backend = Failing {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    let mut b = candidate(&lease, "b@host:2");
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
    fails.store(1, Ordering::Relaxed);
    let word = t0 + S / 2;
    a.step_down(None, word).expect_err("the backend is away");
    // The record stands unrenewed for its duration and the standby
    // takes it; the leaving holder reads that as its loss.
    assert_eq!(
        b.tick(t0 + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(
        a.tick(t0 + 3 * S).expect("the loss"),
        TickOutcome::Lost {
            term: 1,
            holder: "b@host:2".to_string()
        }
    );
    assert!(!a.is_holder());
    // Its hold stands: when the taker stops inside it, the replica that
    // left waits its turn.
    assert!(b.release().expect("the taker's stop"));
    assert_eq!(
        a.tick(word + 3 * S - Duration::from_micros(1))
            .expect("look"),
        TickOutcome::Waiting { term: 2 }
    );
    assert_eq!(
        a.tick(word + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 3 }
    );
}

/// A backend whose write lands and then reports an error while `fails`
/// is above zero: a sync that failed after the rename, an update the
/// server applied before the call timed out.
struct LandsThenErrors {
    lease: MemoryLease,
    fails: Arc<AtomicU32>,
}

impl LeaseBackend for LandsThenErrors {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        let written = self.lease.write(expected, record)?;
        if self.fails.load(Ordering::Relaxed) > 0 {
            self.fails.fetch_sub(1, Ordering::Relaxed);
            return Err(ElectError::Io {
                place: self.place(),
                source: std::io::Error::other("the answer was lost"),
            });
        }
        Ok(written)
    }

    fn place(&self) -> String {
        "lands-then-errors".to_string()
    }
}

#[test]
fn a_write_that_landed_but_reported_an_error_is_this_runs_own() {
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(1));
    let backend = LandsThenErrors {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    // The take lands and the answer is lost.
    a.tick(t0).expect_err("the answer was lost");
    assert!(!a.is_holder());
    assert_eq!(
        a.fence().leader_term(t0),
        None,
        "no fence on an unanswered take"
    );
    assert_eq!(
        lease
            .clone()
            .read()
            .expect("read")
            .expect("the record")
            .holder,
        "a@host:1"
    );
    // The next look finds this run's own write and takes it at once. No
    // record stood at the attempt, so the fence opens one renew deadline
    // after that attempt: the renewal below, not this look.
    assert_eq!(
        a.tick(t0 + S).expect("own"),
        TickOutcome::Acquired { term: 1 }
    );
    assert_eq!(a.fence().leader_term(t0 + S), None);
    // A renewal lands and the answer is lost: the record is still this
    // run's own, renewed from there, never a loss.
    fails.store(1, Ordering::Relaxed);
    a.tick(t0 + 2 * S).expect_err("the answer was lost");
    assert!(a.is_holder());
    assert_eq!(
        a.tick(t0 + 3 * S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(a.fence().leader_term(t0 + 3 * S), Some(1));
    // A standby under another name saw a live record throughout.
    let mut b = candidate(&lease, "b@host:2");
    assert!(matches!(
        b.tick(t0 + 3 * S).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
}

#[test]
fn a_release_that_landed_but_errored_reads_as_released_on_the_retry() {
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(0));
    let backend = LandsThenErrors {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.tick(t0).expect("acquire");
    fails.store(1, Ordering::Relaxed);
    a.release().expect_err("the answer was lost");
    assert!(a.is_holder(), "kept for the retry");
    assert!(
        lease
            .clone()
            .read()
            .expect("read")
            .expect("the record")
            .is_released(),
        "the release landed"
    );
    assert!(a.release().expect("the retry"), "released after all");
    assert!(!a.is_holder());
    assert_eq!(a.fence().leader_term(t0), None);
}

#[test]
fn a_release_that_landed_but_errored_is_not_renewed_by_the_next_tick() {
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(0));
    let backend = LandsThenErrors {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    fails.store(1, Ordering::Relaxed);
    a.release().expect_err("the answer was lost");
    // The caller ticks instead of retrying: the released record is this
    // run's own release, not a lease to renew. The fence stays closed
    // under the old term while the candidates take the free lease.
    assert_eq!(
        a.tick(t0 + S).expect("look"),
        TickOutcome::Lost {
            term: 1,
            holder: String::new()
        }
    );
    assert!(!a.is_holder());
    assert_eq!(a.fence().leader_term(t0 + S), None);
    assert!(
        lease
            .clone()
            .read()
            .expect("read")
            .expect("the record")
            .is_released(),
        "still released"
    );
    let mut b = candidate(&lease, "b@host:2");
    assert_eq!(
        b.tick(t0 + S + S / 2).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert!(matches!(
        a.tick(t0 + 2 * S).expect("look"),
        TickOutcome::Following { term: 2, .. }
    ));
}

#[test]
fn a_release_after_a_renewal_that_landed_but_errored_gives_up_the_landed_record() {
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(0));
    let backend = LandsThenErrors {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    fails.store(1, Ordering::Relaxed);
    a.tick(t0 + S).expect_err("the answer was lost");
    // A planned stop: the release goes over the renewal that landed, not
    // over the record before it, and the lease is free at once.
    assert!(a.release().expect("release"), "released");
    assert!(!a.is_holder());
    assert_eq!(a.fence().leader_term(t0 + S), None);
    let stored = lease.clone().read().expect("read").expect("the record");
    assert!(stored.is_released());
    assert_eq!(stored.term, 1);
    let mut b = candidate(&lease, "b@host:2");
    assert_eq!(
        b.tick(t0 + S + S / 2).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
}

#[test]
fn a_release_inside_the_wait_over_a_renewal_that_landed_hands_the_wait_on() {
    // This run's renewal landed with its answer lost, inside the take's
    // wait. The release finds that renewal, releases over it, and still
    // says what the take's wait owed as the record's duration.
    let lease = MemoryLease::new();
    let fails = Arc::new(AtomicU32::new(0));
    let backend = LandsThenErrors {
        lease: lease.clone(),
        fails: Arc::clone(&fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    fails.store(1, Ordering::Relaxed);
    a.tick(t0 + S).expect_err("the answer was lost");
    assert!(a.waits_to_lead());
    assert!(a.release().expect("release"));
    assert!(!a.waits_to_lead(), "a release ends the wait");
    let stored = lease.clone().read().expect("read").expect("the record");
    assert!(stored.is_released());
    assert_eq!(stored.duration_ms, 1_000);
}

/// A backend whose reads fail while `read_fails` is above zero, whose
/// writes land and then report an error while `lost_answers` is above
/// zero, and whose writes fail before landing while `write_fails` is
/// above zero: one API server through its blips.
struct Blips {
    lease: MemoryLease,
    read_fails: Arc<AtomicU32>,
    lost_answers: Arc<AtomicU32>,
    write_fails: Arc<AtomicU32>,
}

impl LeaseBackend for Blips {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        if self.read_fails.load(Ordering::Relaxed) > 0 {
            self.read_fails.fetch_sub(1, Ordering::Relaxed);
            return Err(ElectError::Io {
                place: self.place(),
                source: std::io::Error::other("the read was lost"),
            });
        }
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        if self.write_fails.load(Ordering::Relaxed) > 0 {
            self.write_fails.fetch_sub(1, Ordering::Relaxed);
            return Err(ElectError::Io {
                place: self.place(),
                source: std::io::Error::other("the write was refused"),
            });
        }
        let written = self.lease.write(expected, record)?;
        if self.lost_answers.load(Ordering::Relaxed) > 0 {
            self.lost_answers.fetch_sub(1, Ordering::Relaxed);
            return Err(ElectError::Io {
                place: self.place(),
                source: std::io::Error::other("the answer was lost"),
            });
        }
        Ok(written)
    }

    fn place(&self) -> String {
        "blips".to_string()
    }
}

#[test]
fn a_release_whose_re_read_failed_keeps_the_record_for_the_retry() {
    let lease = MemoryLease::new();
    let read_fails = Arc::new(AtomicU32::new(0));
    let lost_answers = Arc::new(AtomicU32::new(0));
    let backend = Blips {
        lease: lease.clone(),
        read_fails: Arc::clone(&read_fails),
        lost_answers: Arc::clone(&lost_answers),
        write_fails: Arc::new(AtomicU32::new(0)),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.tick(t0).expect("acquire");
    // A renewal lands and its answer is lost.
    lost_answers.store(1, Ordering::Relaxed);
    a.tick(t0 + S).expect_err("the answer was lost");
    // A planned stop: the release's write is refused over the record
    // before that renewal, and the re-read that settles what moved fails
    // too. The record is kept, so a second call retries, finds its own
    // landed renewal and releases it.
    read_fails.store(1, Ordering::Relaxed);
    a.release().expect_err("the read was lost");
    assert!(a.is_holder(), "kept for the retry");
    assert_eq!(a.fence().leader_term(t0 + S), None);
    assert!(a.release().expect("the retry"), "released after all");
    assert!(!a.is_holder());
    let stored = lease.clone().read().expect("read").expect("the record");
    assert!(stored.is_released());
    assert_eq!(stored.term, 1);
}

#[test]
fn a_release_write_that_fails_after_a_landed_renewal_keeps_that_renewal_for_the_retry() {
    let lease = MemoryLease::new();
    let lost_answers = Arc::new(AtomicU32::new(0));
    let write_fails = Arc::new(AtomicU32::new(0));
    let backend = Blips {
        lease: lease.clone(),
        read_fails: Arc::new(AtomicU32::new(0)),
        lost_answers: Arc::clone(&lost_answers),
        write_fails: Arc::clone(&write_fails),
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    // A renewal lands and its answer is lost; then the release's own
    // write fails before it lands. The landed renewal stays the record
    // the retry adopts: the lease is free at once, not at its expiry.
    lost_answers.store(1, Ordering::Relaxed);
    a.tick(t0 + S).expect_err("the answer was lost");
    write_fails.store(1, Ordering::Relaxed);
    a.release().expect_err("the write was refused");
    assert!(a.is_holder(), "kept for the retry");
    assert!(a.release().expect("the retry"), "released after all");
    assert!(!a.is_holder());
    let stored = lease.clone().read().expect("read").expect("the record");
    assert!(stored.is_released());
    assert_eq!(stored.term, 1);
    let mut b = candidate(&lease, "b@host:2");
    assert_eq!(
        b.tick(t0 + 2 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
}

/// A backend whose write does not land while `delay` is set: it reports
/// an error and lands at the backend's next write, before that write is
/// judged, as a server applies a timed-out update late.
struct LandsLate {
    lease: MemoryLease,
    delay: Arc<AtomicBool>,
    pending: Option<(Option<LeaseRecord>, LeaseRecord)>,
}

impl LeaseBackend for LandsLate {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.lease.read()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        if let Some((expected, record)) = self.pending.take() {
            self.lease.write(expected.as_ref(), &record)?;
        }
        if self.delay.swap(false, Ordering::Relaxed) {
            self.pending = Some((expected.cloned(), record.clone()));
            return Err(ElectError::Io {
                place: self.place(),
                source: std::io::Error::other("the answer was lost"),
            });
        }
        self.lease.write(expected, record)
    }

    fn place(&self) -> String {
        "lands-late".to_string()
    }
}

#[test]
fn an_initial_take_that_landed_late_is_this_runs_own_on_the_next_tick() {
    let lease = MemoryLease::new();
    let delay = Arc::new(AtomicBool::new(true));
    let backend = LandsLate {
        lease: lease.clone(),
        delay: Arc::clone(&delay),
        pending: None,
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    // The first take reports an error and lands after the next read:
    // that read finds nothing, the retry's write is refused as moved,
    // and the tick after reads this run's own record and takes it.
    a.tick(t0).expect_err("the answer was lost");
    assert_eq!(a.tick(t0 + S).expect("retry"), TickOutcome::Raced);
    assert_eq!(
        a.tick(t0 + 2 * S).expect("own"),
        TickOutcome::Acquired { term: 1 }
    );
    assert!(a.is_holder());
    // No record stood at either attempt, and the record may have landed
    // at any time before this look: the fence opens one renew deadline
    // after the write that took it here, never after an attempt.
    assert_eq!(a.fence().leader_term(t0 + 2 * S), None);
    assert_eq!(
        a.tick(t0 + 3 * S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(a.fence().leader_term(t0 + 3 * S), None);
    assert_eq!(a.fence().leader_term(t0 + 4 * S), Some(1));
    assert_eq!(
        lease
            .clone()
            .read()
            .expect("read")
            .expect("the record")
            .holder,
        "a@host:1"
    );
}

#[test]
fn a_renewal_that_landed_late_is_renewed_over_not_lost() {
    let lease = MemoryLease::new();
    let delay = Arc::new(AtomicBool::new(false));
    let backend = LandsLate {
        lease: lease.clone(),
        delay: Arc::clone(&delay),
        pending: None,
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(
        a.tick(t0).expect("acquire"),
        TickOutcome::Acquired { term: 1 }
    );
    delay.store(true, Ordering::Relaxed);
    a.tick(t0 + S).expect_err("the answer was lost");
    assert!(a.is_holder());
    // The next tick reads the record before the lost renewal lands and
    // writes after it did: the record under its write is this run's
    // own, and the renewal goes over it within the tick.
    assert_eq!(
        a.tick(t0 + 2 * S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert!(a.is_holder());
    assert_eq!(a.fence().leader_term(t0 + 2 * S), Some(1));
    assert_eq!(
        lease
            .clone()
            .read()
            .expect("read")
            .expect("the record")
            .holder,
        "a@host:1"
    );
    assert_eq!(
        a.tick(t0 + 3 * S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    let mut b = candidate(&lease, "b@host:2");
    assert!(matches!(
        b.tick(t0 + 3 * S).expect("follow"),
        TickOutcome::Following { term: 1, .. }
    ));
}

/// A holder whose renewal reported an error and lands at the store's
/// next write, before that write is judged; the lease comes with it.
fn led_with_a_late_renewal(t0: Instant) -> (Elector, MemoryLease) {
    let lease = MemoryLease::new();
    let delay = Arc::new(AtomicBool::new(false));
    let backend = LandsLate {
        lease: lease.clone(),
        delay: Arc::clone(&delay),
        pending: None,
    };
    let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
    assert_eq!(
        a.tick(t0).expect("acquire"),
        TickOutcome::Acquired { term: 1 }
    );
    delay.store(true, Ordering::Relaxed);
    a.tick(t0 + S).expect_err("the answer was lost");
    (a, lease)
}

#[test]
fn a_renewal_adopted_at_the_read_after_a_moved_write_is_no_stranger() {
    // The late renewal lands under the next renewal's write, which
    // comes back moved. The read after it observes that record with
    // the attempt still kept: the record is this run's own.
    let t0 = Instant::now() + AHEAD;
    let (mut a, lease) = led_with_a_late_renewal(t0);
    assert_eq!(
        a.tick(t0 + 2 * S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    // Another run under this name writes the record once: the first
    // stranger, followed for the record's duration. A second would be
    // refused.
    let mine = lease.clone().read().expect("read").expect("the record");
    let theirs = LeaseRecord {
        renewed_at_micros: mine.renewed_at_micros + 1,
        ..mine.clone()
    };
    assert_eq!(
        lease.clone().write(Some(&mine), &theirs).expect("write"),
        ConditionalWriteOutcome::Done
    );
    assert_eq!(
        a.tick(t0 + 3 * S).expect("the look that loses the lease"),
        TickOutcome::Lost {
            term: 1,
            holder: "a@host:1".to_owned()
        }
    );
    assert_eq!(
        a.tick(t0 + 4 * S).expect("the first stranger"),
        TickOutcome::Following {
            holder: "a@host:1".to_owned(),
            term: 1
        }
    );
}

#[test]
fn a_renewal_adopted_at_a_releases_read_after_a_moved_write_is_no_stranger() {
    // The same adoption at a word to leave: the release's write comes
    // back moved over the late renewal, the read after it observes
    // that record with the attempt still kept, and the release goes
    // over it.
    let t0 = Instant::now() + AHEAD;
    let (mut a, lease) = led_with_a_late_renewal(t0);
    assert!(a.step_down(None, t0 + S).expect("the release"));
    let released = lease.clone().read().expect("read").expect("the record");
    assert!(released.is_released());
    // Another run under this name takes the lease: the first stranger.
    let theirs = LeaseRecord {
        holder: "a@host:1".to_owned(),
        term: 2,
        duration_ms: 3_000,
        renewed_at_micros: released.renewed_at_micros + 1,
        ..released.clone()
    };
    assert_eq!(
        lease
            .clone()
            .write(Some(&released), &theirs)
            .expect("write"),
        ConditionalWriteOutcome::Done
    );
    assert_eq!(
        a.tick(t0 + 2 * S).expect("the first stranger"),
        TickOutcome::Following {
            holder: "a@host:1".to_owned(),
            term: 2
        }
    );
}
