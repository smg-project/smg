//! What a reader of the fence sees: the lead's window at a take's step,
//! behind a renewal, across a slow read, from other threads and among
//! several candidates.

use super::*;

/// A take's step runs behind the take's stamp and reads the fence as
/// any reader does: the take's term where its window is open, and no
/// term where the lead still waits to open. A renewal is no take and
/// runs no step.
#[test]
fn a_takes_step_runs_behind_its_stamp() {
    let t0 = Instant::now() + AHEAD;
    // The steps a candidate ran, and whether the last was shown the
    // first term.
    let stepping = |elector: &mut Elector| {
        let fence = elector.fence();
        let (steps, shown) = (
            Arc::new(AtomicU32::new(0)),
            Arc::new(AtomicBool::new(false)),
        );
        let step = {
            let (steps, shown) = (Arc::clone(&steps), Arc::clone(&shown));
            move || {
                steps.fetch_add(1, Ordering::Relaxed);
                shown.store(fence.leader_term(t0) == Some(1), Ordering::Relaxed);
            }
        };
        elector.at_take(step);
        (steps, shown)
    };
    let mut alone = Elector::alone("a@host:1", settings()).expect("valid settings");
    let (steps, shown) = stepping(&mut alone);
    assert_eq!(
        alone.tick(t0).expect("tick"),
        TickOutcome::Acquired { term: 1 }
    );
    assert_eq!(steps.load(Ordering::Relaxed), 1);
    assert!(
        shown.load(Ordering::Relaxed),
        "the fence showed the take's term at the step"
    );
    assert_eq!(
        alone.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(steps.load(Ordering::Relaxed), 1, "a renewal is no take");

    // A take with no holder to wait for leads one renew deadline
    // later: its step is shown no term.
    let lease = MemoryLease::new();
    let mut waiting = candidate(&lease, "b@host:1");
    let (steps, shown) = stepping(&mut waiting);
    assert_eq!(
        waiting.tick(t0).expect("tick"),
        TickOutcome::Acquired { term: 1 }
    );
    assert_eq!(steps.load(Ordering::Relaxed), 1);
    assert!(
        !shown.load(Ordering::Relaxed),
        "the lead still waits to open"
    );
}

#[test]
fn a_renewal_reopens_the_fence_and_keeps_the_term() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    let first = a.observed().expect("the record").clone();
    assert_eq!(
        a.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    let second = a.observed().expect("the record").clone();
    assert!(
        second.renewed_at_micros > first.renewed_at_micros,
        "every renewal moves the record"
    );
    assert_eq!(second.acquired_at_micros, first.acquired_at_micros);
    assert_eq!(second.term, 1);
    let fence = a.fence();
    assert_eq!(
        fence.leader_term(t0 + 3 * S),
        Some(1),
        "the deadline counts from the renewal"
    );
    assert_eq!(
        fence.leader_term(t0 + 3 * S + Duration::from_micros(1)),
        None
    );
}

#[test]
fn two_electors_under_one_identity_never_both_lead() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let mut twin = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(
        a.tick(t0).expect("acquire"),
        TickOutcome::Acquired { term: 1 }
    );
    // The twin sees a live record under its own name: it waits for the
    // record's duration of stillness, and the holder's renewal turns the
    // wait into a refusal, never a take.
    assert!(matches!(
        twin.tick(t0 + S / 2).expect("look"),
        TickOutcome::Following { .. }
    ));
    assert_eq!(
        a.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    let err = twin
        .tick(t0 + S + S / 2)
        .expect_err("a second writer under this identity");
    assert_eq!(
        err.to_string(),
        "election identity: another candidate writes under this identity"
    );
    assert!(!twin.is_holder());
    for i in 2..6u32 {
        let now = t0 + i * S;
        assert_eq!(
            a.tick(now).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        twin.tick(now + S / 2)
            .expect_err("refused while the record moves");
        assert_eq!(a.fence().leader_term(now), Some(1));
        assert_eq!(twin.fence().leader_term(now + S / 2), None);
    }
    // The holder stalls after its renewal at `last`. Its fence is open to
    // the renew deadline; the twin keeps its distance for the record's
    // duration from its first sight of that renewal, then takes the
    // term above with the earlier fence closed: the refusal was not for
    // life.
    let last = t0 + 5 * S;
    let first_sight = last + S / 2;
    twin.tick(last + 2 * S)
        .expect_err("the record may still move");
    assert_eq!(a.fence().leader_term(last + 2 * S), Some(1));
    assert_eq!(twin.fence().leader_term(last + 2 * S), None);
    twin.tick(first_sight + 3 * S - Duration::from_micros(1))
        .expect_err("not yet");
    assert_eq!(
        twin.tick(first_sight + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(a.fence().leader_term(first_sight + 3 * S), None);
    assert_eq!(twin.fence().leader_term(first_sight + 3 * S), Some(2));
    // The stalled holder wakes to a record that moved under it.
    assert_eq!(
        a.tick(first_sight + 4 * S).expect("look"),
        TickOutcome::Lost {
            term: 1,
            holder: "a@host:1".to_string()
        }
    );
    assert_eq!(a.fence().leader_term(first_sight + 4 * S), None);
}

/// A backend whose first read stalls while the holder renews inside the
/// stall: the record it returns is newer than the instant the read began.
struct SlowRead {
    lease: MemoryLease,
    holder: Arc<Mutex<Elector>>,
    stall: Option<Duration>,
    /// The record goes after the holder's renewal, inside the stall.
    loses: bool,
}

impl LeaseBackend for SlowRead {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        if let Some(stall) = self.stall.take() {
            thread::sleep(stall);
            self.holder.lock().tick(Instant::now())?;
            if self.loses {
                self.lease.lose();
            }
            thread::sleep(Duration::from_millis(5));
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
        "slow".to_string()
    }
}

#[test]
fn a_slow_read_dates_the_followers_first_sight_when_the_read_returns() {
    // Real instants: the read's own delay is what this is about. A tenth
    // of the design's settings keeps the run under a second.
    let fast = ElectionSettings {
        duration: Duration::from_millis(120),
        renew_period: Duration::from_millis(40),
        renew_deadline: Duration::from_millis(80),
    };
    let lease = MemoryLease::new();
    let holder = Arc::new(Mutex::new(
        Elector::new(Box::new(lease.clone()), "a@host:1", fast).expect("settings"),
    ));
    let a_fence = {
        let mut a = holder.lock();
        assert_eq!(
            a.tick(Instant::now()).expect("acquire"),
            TickOutcome::Acquired { term: 1 }
        );
        a.fence()
    };
    let backend = SlowRead {
        lease,
        holder: Arc::clone(&holder),
        stall: Some(Duration::from_millis(90)),
        loses: false,
    };
    let mut b = Elector::new(Box::new(backend), "b@host:2", fast).expect("settings");
    // b's first look begins now; 90 ms into the read the holder renews,
    // so its fence runs 80 ms from that renewal; then the holder stalls
    // for good. Dated before the read, b's wait would end inside that
    // window.
    assert!(matches!(
        b.tick(Instant::now()).expect("look"),
        TickOutcome::Following { .. }
    ));
    let started = Instant::now();
    let took = loop {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "b never took the lease"
        );
        let now = Instant::now();
        if b.tick(now).expect("look") == (TickOutcome::Acquired { term: 2 }) {
            break now;
        }
        thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(
        a_fence.leader_term(took),
        None,
        "the holder's fence had closed before the take began"
    );
}

#[test]
fn a_slow_read_dates_the_takers_opening_when_the_read_returns() {
    // Real instants and a tenth of the design's settings, as above. The
    // holder renews 90 ms into b's first read and its record goes before
    // that read returns: b never saw the lease held, reads no record and
    // takes at once. The holder's fence runs 80 ms from that renewal, so
    // an opening dated before the read would end inside it.
    let fast = ElectionSettings {
        duration: Duration::from_millis(120),
        renew_period: Duration::from_millis(40),
        renew_deadline: Duration::from_millis(80),
    };
    let lease = MemoryLease::new();
    let holder = Arc::new(Mutex::new(
        Elector::new(Box::new(lease.clone()), "a@host:1", fast).expect("settings"),
    ));
    let a_fence = {
        let mut a = holder.lock();
        assert_eq!(
            a.tick(Instant::now()).expect("acquire"),
            TickOutcome::Acquired { term: 1 }
        );
        a.fence()
    };
    let backend = SlowRead {
        lease,
        holder: Arc::clone(&holder),
        stall: Some(Duration::from_millis(90)),
        loses: true,
    };
    let mut b = Elector::new(Box::new(backend), "b@host:2", fast).expect("settings");
    assert_eq!(
        b.tick(Instant::now()).expect("take"),
        TickOutcome::Acquired { term: 1 }
    );
    let b_fence = b.fence();
    let started = Instant::now();
    let opened = loop {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "b's fence never opened"
        );
        let now = Instant::now();
        b.tick(now).expect("renew");
        if b_fence.leader_term(now).is_some() {
            break now;
        }
        thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(
        a_fence.leader_term(opened),
        None,
        "the holder's fence had closed before the taker's opened"
    );
}

#[test]
fn a_shared_records_duration_outlasts_its_fence() {
    let duration = Duration::from_millis(u64::MAX);
    let settings = ElectionSettings {
        duration,
        renew_period: S,
        renew_deadline: duration - S,
    };
    let mut lease = MemoryLease::new();
    let released = held_by("", 1, 0);
    assert_eq!(
        lease.write(None, &released).expect("seed"),
        ConditionalWriteOutcome::Done
    );
    let mut a = Elector::new(Box::new(lease.clone()), "a@host:1", settings).expect("settings");
    let mut b = Elector::new(Box::new(lease.clone()), "b@host:2", settings).expect("settings");
    let t0 = Instant::now() + AHEAD;
    assert!(
        t0.checked_add(Duration::MAX).is_none(),
        "the greatest duration is past the clock's reach"
    );
    assert!(
        later(t0, Duration::MAX) > t0 + 3600 * S,
        "the clock-overflow fallback stays in the distant future"
    );
    assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 2 });
    assert_eq!(a.fence().leader_term(t0), Some(2));
    assert!(matches!(
        b.tick(t0).expect("follow"),
        TickOutcome::Following { .. }
    ));
    let Some(take) = t0.checked_add(duration) else {
        panic!("the record's greatest duration is inside the clock's reach");
    };
    assert_eq!(a.fence().leader_term(take), None);
    assert_eq!(
        b.tick(take).expect("take"),
        TickOutcome::Acquired { term: 3 }
    );
    assert_eq!(b.fence().leader_term(take), Some(3));
}

#[test]
fn a_read_that_finds_a_lead_finds_the_end_of_its_window() {
    // An elector that takes and leaves the lead as fast as it can, under
    // a reader of the pair. One read of the fence never shows a lead
    // beside no turn, as two reads astride a release would.
    let fence = Arc::new(Fence::new());
    let t0 = Instant::now();
    let until = t0 + 3_600 * S;
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let (fence, stop) = (Arc::clone(&fence), Arc::clone(&stop));
        thread::spawn(move || {
            let mut term = 0;
            while !stop.load(Ordering::Relaxed) {
                term += 1;
                fence.set(term, Some((t0, until)));
                fence.set(term, None);
            }
        })
    };
    let mut led = 0;
    while led < 50_000 && t0.elapsed() < 10 * S {
        let (term, turn) = fence.term_and_turn(Instant::now());
        if term.is_some() {
            led += 1;
            assert_eq!(turn, Some(until), "a lead with no instant to sleep to");
        }
    }
    stop.store(true, Ordering::Relaxed);
    writer.join().expect("the writer thread");
    assert!(led > 0, "the reader saw no lead");
}

#[test]
fn the_fence_is_read_from_other_threads_without_the_elector() {
    let lease = MemoryLease::new();
    let mut a = candidate(&lease, "a@host:1");
    let t0 = Instant::now() + AHEAD;
    a.lead_at(t0);
    let fence = a.fence();
    let read = thread::spawn(move || fence.leader_term(t0 + S))
        .join()
        .expect("the reader thread");
    assert_eq!(read, Some(1));
    fn assert_send<T: Send>(_: &T) {}
    assert_send(&a);
}

#[test]
fn four_candidates_never_lead_at_once() {
    // Every round is four seconds later, so every follower's sight of the
    // record is expired and all four write for it; the compare-and-swap
    // lets one through per round, and the fences agree.
    let lease = MemoryLease::new();
    let mut electors: Vec<Elector> = (0..4)
        .map(|i| candidate(&lease, &format!("c{i}@host:{i}")))
        .collect();
    let t0 = Instant::now() + AHEAD;
    let mut last_term = 0;
    for round in 0..24u32 {
        let now = t0 + round * 4 * S;
        for i in 0..4 {
            let elector = &mut electors[(i + round as usize) % 4];
            let tick = elector.tick(now).expect("tick");
            if let TickOutcome::Acquired { term } | TickOutcome::Renewed { term } = tick {
                assert!(
                    term >= last_term,
                    "terms only grow: {term} after {last_term}"
                );
                last_term = term;
            }
        }
        let leaders = electors
            .iter()
            .filter(|e| e.fence().leader_term(now).is_some())
            .count();
        assert!(leaders <= 1, "round {round}: {leaders} fenced leaders");
        let holders = electors.iter().filter(|e| e.is_holder()).count();
        assert_eq!(holders, 1, "round {round}: one record holder");
    }
    assert!(last_term >= 2, "the lease changed hands");
}
