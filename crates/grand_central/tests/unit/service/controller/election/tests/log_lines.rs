//! The loop's log lines: the one line a streak of a fallen term, of late
//! renewals and of a record that is gone, a take's lines of its wait and of
//! leading, and the lines of a holder whose record is emptied.

use super::*;

#[test]
fn a_fallen_term_logs_one_line_a_streak_and_one_more_when_it_falls_again() {
    let held = |holder: &str, term: u64| LeaseRecord {
        holder: holder.to_owned(),
        term,
        duration_ms: 60_000,
        ..LeaseRecord::default()
    };
    let mut lease = MemoryLease::new();
    let first = held("x@127.0.0.1:8", 5);
    let fallen = held("y@127.0.0.1:9", 3);
    let back = held("y@127.0.0.1:9", 5);
    let again = held("y@127.0.0.1:9", 4);
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
    let (standing, mut stood) = watch::channel(ElectionObservation::default());
    // The loop itself, on a thread of the case's own under a subscriber
    // that keeps its lines. A second subscriber stays registered beside
    // it: with one alone a line is on or off by the thread that reaches
    // it first, and a case beside this one reaches it under none.
    let lines = Logged(Arc::new(Mutex::new(Vec::new())));
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let (stop, stopped) = mpsc::channel();
    let looping = thread::spawn(move || {
        tracing::subscriber::with_default(subscriber, || {
            run(follower, &MemoryStore::new(), &standing, &stopped);
        });
    });
    until("the first look", || stood.borrow().holder == first.holder);

    // A streak of looks under the fallen term: one line.
    assert_eq!(
        lease.write(Some(&first), &fallen).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    until("the look at the fallen term", || {
        stood.borrow().holder.is_empty()
    });
    looks(&mut stood, 5);
    // The term comes back and falls again: one line more.
    assert_eq!(
        lease.write(Some(&fallen), &back).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    until("the look at the term come back", || {
        stood.borrow().holder == back.holder
    });
    assert_eq!(
        lease.write(Some(&back), &again).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    until("the look at the second fall", || {
        stood.borrow().holder.is_empty()
    });
    looks(&mut stood, 5);
    stop.send(Word::Stop).expect("the loop runs");
    // Bounded like every wait here: a loop that ignored its stop fails
    // this case by name.
    until("the loop's end after its stop", || looping.is_finished());
    looping.join().expect("the loop ends");

    let text = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let fell = text
        .lines()
        .filter(|line| line.contains("the lease's term is under the highest this replica saw"))
        .count();
    assert_eq!(fell, 2, "{text}");
}

#[test]
fn a_late_renewal_logs_one_line_a_streak_and_one_more_when_renewals_turn_late_again() {
    unheld(
        "a late renewal's lines",
        Duration::from_millis(300),
        late_renewals_log_one_line_a_streak,
    );
}

/// One try of the case above: 300 ms of room, three renew periods.
fn late_renewals_log_one_line_a_streak() {
    // A slow write takes four periods under a deadline of four: every
    // renewal of a slow streak lands on a fence that had closed. A fast
    // one lands a period after the stamp before it, three periods inside
    // the deadline: a held host has that room.
    let period = Duration::from_millis(100);
    let slow = Arc::new(AtomicBool::new(false));
    let began = Arc::new(Mutex::new(Vec::new()));
    let store = WritesSlowly {
        lease: MemoryLease::new(),
        takes: period * 4,
        slow: Arc::clone(&slow),
        began: Arc::clone(&began),
    };
    let settings = ElectionSettings {
        duration: period * 6,
        renew_period: period,
        renew_deadline: period * 4,
    };
    let leader = Elector::new(Box::new(store), "a@127.0.0.1:1", settings)
        .expect("the settings are in order");
    let (standing, _stood) = watch::channel(ElectionObservation::default());
    let looping = looping(leader, standing);
    // Waits until `more` more writes began: all but the last of them
    // have landed by then.
    let writes = |more: usize| {
        let from = began.lock().len();
        until("the writes", || began.lock().len() >= from + more);
    };
    writes(3);
    // A slow streak, two late renewals at the least: one line.
    slow.store(true, Ordering::SeqCst);
    writes(4);
    // Fast again: the streak ends at the first renewal that lands
    // inside the deadline.
    slow.store(false, Ordering::SeqCst);
    writes(4);
    // Late again: one line more.
    slow.store(true, Ordering::SeqCst);
    writes(3);
    slow.store(false, Ordering::SeqCst);
    let (late, text) = looping.stopped_with("a renewal landed on a fence that had closed");
    // The bound on the clock: every fast renewal landed inside the
    // deadline. A host that held one past it made one more late line.
    bound!(late == 2, "{late} late lines:\n{text}");
}

#[test]
fn a_record_that_is_gone_logs_one_line_a_streak_and_one_more_when_it_goes_again() {
    // The record's duration is a minute: the case ends inside the wait.
    let held = LeaseRecord {
        holder: "x@127.0.0.1:8".to_owned(),
        term: 5,
        duration_ms: 60_000,
        ..LeaseRecord::default()
    };
    let store = Losable::default();
    store.put(&held);
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(400),
    };
    let follower = Elector::new(Box::new(store.clone()), "b@127.0.0.1:2", settings)
        .expect("the settings are in order");
    let (standing, mut stood) = watch::channel(ElectionObservation::default());
    let looping = looping(follower, standing);
    until("the first look", || stood.borrow().holder == held.holder);

    // A streak of looks at a record that is gone: one line, nobody
    // named, the term kept.
    store.lose();
    until("the look at the loss", || stood.borrow().holder.is_empty());
    looks(&mut stood, 5);
    assert_eq!(stood.borrow().term, 5);
    // The record is back and goes again, replaced by an empty one this
    // time: one line more, and no line of a fallen term beside it.
    store.put(&held);
    until("the look at the record come back", || {
        stood.borrow().holder == held.holder
    });
    store.put(&LeaseRecord::default());
    until("the look at the second loss", || {
        stood.borrow().holder.is_empty()
    });
    looks(&mut stood, 5);
    let (gone, text) = looping.stopped_with("the lease's record is gone or names no holder");
    assert_eq!(gone, 2, "{text}");
    assert!(!text.contains("the lease's term fell"), "{text}");
    // The holder followed again after the first streak is named anew.
    assert_eq!(text.matches(": following").count(), 2, "{text}");
}

#[test]
fn a_take_whose_slow_step_closed_its_fence_logs_no_wait() {
    const TOOK: &str = "took a lease with no holder to wait for";
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(400),
    };
    // A holder's release is taken at once and waits for nobody. This
    // take's step began a second before its report, as over a store that
    // answers slowly: the fence reads closed at the report, past its
    // renew deadline, and no line names a wait that was not made.
    let mut lease = MemoryLease::new();
    let released = LeaseRecord {
        term: 1,
        ..LeaseRecord::default()
    };
    assert_eq!(
        lease.write(None, &released).expect("a write"),
        ConditionalWriteOutcome::Done
    );
    let mut taker = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", settings)
        .expect("the settings are in order");
    let began = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("the clock is past its first second");
    let tick = taker.tick(began).expect("the take");
    assert_eq!(tick, TickOutcome::Acquired { term: 2 });
    assert_eq!(taker.fence().leader_term(Instant::now()), None);
    let lines = Logged(Arc::new(Mutex::new(Vec::new())));
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let mut leading = false;
    tracing::subscriber::with_default(subscriber, || {
        super::super::report(
            &taker,
            &tick,
            &MemoryStore::new(),
            &mut String::new(),
            &mut super::super::Streak::default(),
            &mut leading,
        );
    });
    let text = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    assert_eq!(text.matches(TOOK).count(), 0, "{text}");
    assert!(!leading, "{text}");
}

#[test]
fn a_take_with_no_holder_to_wait_for_logs_its_take_and_leading_once_the_fence_opens() {
    const TOOK: &str = "took a lease with no holder to wait for";
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(400),
    };
    // Stopped inside the renew deadline of a take over no record: the
    // take's line alone. The deadline is half a minute: the case ends
    // inside it.
    let patient = ElectionSettings {
        duration: Duration::from_secs(60),
        renew_deadline: Duration::from_secs(30),
        ..settings
    };
    let lease = MemoryLease::new();
    let taker = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", patient)
        .expect("the settings are in order");
    let fence = taker.fence();
    let (standing, mut stood) = watch::channel(ElectionObservation::default());
    let inside = looping(taker, standing);
    until("the take", || stood.borrow().holder == "a@127.0.0.1:1");
    looks(&mut stood, 5);
    assert_eq!(fence.leader_term(Instant::now()), None);
    let (leading, text) = inside.stopped_with(": leading");
    assert_eq!((leading, text.matches(TOOK).count()), (0, 1), "{text}");

    // The same take under a deadline the case outlives: one line of
    // leading, after the take's, at the first step after the opening.
    let mut lease = MemoryLease::new();
    let taker = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", settings)
        .expect("the settings are in order");
    let fence = taker.fence();
    let (standing, _stood) = watch::channel(ElectionObservation::default());
    let outlived = looping(taker, standing);
    until("the fence's opening", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    // The fence opens between two steps: the next one writes the line.
    until("the line of leading", || outlived.said(": leading"));
    // Another holder takes the lease away and stalls. Its record is
    // waited out and taken above: a holder was waited for, so that take
    // leads at once, under the line of leading and no line of the take.
    let other = LeaseRecord {
        holder: "x@127.0.0.1:8".to_owned(),
        term: 2,
        duration_ms: 600,
        ..LeaseRecord::default()
    };
    until("the other holder's take", || {
        let current = lease.read().expect("a read");
        lease.write(current.as_ref(), &other).expect("a write") == ConditionalWriteOutcome::Done
    });
    until("the take above", || {
        fence.leader_term(Instant::now()) == Some(3)
    });
    let (leading, text) = outlived.stopped_with(": leading");
    assert_eq!((leading, text.matches(TOOK).count()), (2, 1), "{text}");
    assert!(text.find(TOOK) < text.find(": leading"), "{text}");
}

#[test]
fn a_holder_whose_record_is_emptied_logs_that_it_is_gone_and_no_fallen_term() {
    unheld(
        "an emptied record",
        Duration::from_millis(500),
        an_emptied_record_is_gone,
    );
}

/// One try of the case above, inside the wait: half a second of room.
fn an_emptied_record_is_gone() {
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(20),
        renew_deadline: Duration::from_millis(400),
    };
    let store = Losable::default();
    let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", settings)
        .expect("the settings are in order");
    let fence = holder.fence();
    let (standing, mut stood) = watch::channel(ElectionObservation::default());
    let looping = looping(holder, standing);
    until("the fence's opening", || {
        fence.leader_term(Instant::now()) == Some(1)
    });
    // The record is replaced by an empty one under its holder. The
    // holder's step reads a loss to nobody, the next one a record that
    // is gone: one line, and none of a fallen term, which never comes
    // back. The case ends inside the wait.
    let emptied = Instant::now();
    store.put(&LeaseRecord::default());
    // The look at the loss names nobody under the term it held. Past
    // the wait the holder takes again, under the next term: the wait
    // ends on that too, and the bound below tells the two apart.
    until("the look at the loss", || {
        let look = stood.borrow();
        look.holder.is_empty() || look.term > 1
    });
    looks(&mut stood, 5);
    let term = stood.borrow().term;
    let (gone, text) = looping.stopped_with("the lease's record is gone or names no holder");
    // No hold explains a fallen term, which never comes back: its line
    // is read before the bound.
    assert!(!text.contains("the lease's term fell"), "{text}");
    // The bound on the clock: the case ended inside the wait. A host
    // that held it past the wait's end let the holder take again.
    let took = emptied.elapsed();
    bound!(
        took < settings.duration,
        "the case ended {took:?} after the record was emptied, outside the wait"
    );
    assert_eq!(term, 1);
    assert_eq!(gone, 1, "{text}");
}
