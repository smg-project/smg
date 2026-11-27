//! A leader over a store that answers slowly: its fence while renewals take
//! under the margin and just under it, and the step that follows a slow one.

use super::*;

/// What a leader over a slow store showed, sampled every 5 ms.
struct Slow {
    samples: u32,
    /// The samples that read the fence closed.
    closed: u32,
    /// The samples whose standing did not name the leader.
    unnamed: u32,
    /// When each write of the loop began.
    began: Vec<Instant>,
}

/// One candidate takes the lease over a store whose every write takes
/// `takes`, leads through `renewals` renewals a period apart, and is
/// stopped. The deadline is two periods and the duration three, so the
/// fence's margin is one period by both of its forms.
fn led_over_a_slow_store(period: Duration, takes: Duration, renewals: u32) -> Slow {
    let identity = "a@127.0.0.1:1";
    let began = Arc::new(Mutex::new(Vec::new()));
    let store = WritesSlowly {
        lease: MemoryLease::new(),
        takes,
        slow: Arc::new(AtomicBool::new(true)),
        began: Arc::clone(&began),
    };
    let settings = ElectionSettings {
        duration: period * 3,
        renew_period: period,
        renew_deadline: period * 2,
    };
    let candidate =
        Elector::new(Box::new(store), identity, settings).expect("the settings are in order");
    let fence = candidate.fence();
    let (standing, stood) = watch::channel(ElectionObservation::default());
    let running = start(candidate, Arc::new(MemoryStore::new()), standing).expect("a thread");
    until("the take", || {
        fence.leader_term(Instant::now()) == Some(1) && stood.borrow().holder == identity
    });
    // Through the renewals the fence reads open and the standing names
    // the holder, at every sample: what readiness and the Leader answer
    // read.
    let through = Instant::now() + period * renewals + takes;
    let (mut samples, mut closed, mut unnamed) = (0_u32, 0_u32, 0_u32);
    while Instant::now() < through {
        samples += 1;
        closed += u32::from(fence.leader_term(Instant::now()).is_none());
        unnamed += u32::from(stood.borrow().holder != identity);
        thread::sleep(Duration::from_millis(5));
    }
    // The loop's writes alone: the release at the stop follows the last
    // step at once.
    let began = began.lock().clone();
    running.stop();
    ended(running);
    Slow {
        samples,
        closed,
        unnamed,
        began,
    }
}

#[test]
fn a_leader_whose_renewals_take_under_the_margin_never_closes_its_fence() {
    unheld(
        "a slow store under the margin",
        Duration::from_secs(1),
        renewals_under_the_margin,
    );
}

/// One try of the case above: a second of room.
fn renewals_under_the_margin() {
    // The fence's margin is the smaller of the deadline less the period
    // and half the deadline, both 2.5 s here. Every write takes 1.5 s.
    // That is over half the margin: steps that began a period after the
    // step before them RETURNED would land a renewal 5.5 s after the
    // stamp before it and close the fence for half a second in every
    // cycle. It is under the margin by a second: a held host has that
    // room.
    let period = Duration::from_millis(2500);
    let slow = led_over_a_slow_store(period, Duration::from_millis(1500), 3);
    let (samples, closed, unnamed) = (slow.samples, slow.closed, slow.unnamed);
    // The page's row reads this line.
    writeln!(
        std::io::stdout(),
        "SLOW_STORE samples={samples} closed={closed} unnamed={unnamed}"
    )
    .expect("the test's output");
    // No step met the store sooner than a period after the step before
    // it began. A hold only parts two writes further, so no hold
    // excuses this floor: it is read before the bounds.
    for pair in slow.began.windows(2) {
        let apart = pair[1].duration_since(pair[0]);
        assert!(apart >= period * 9 / 10, "two writes {apart:?} apart");
    }
    bound!(
        (closed, unnamed) == (0, 0),
        "{closed} closed and {unnamed} unnamed of {samples} samples"
    );
    bound!(slow.began.len() >= 4, "the take and three renewals");
}

#[test]
fn the_loop_holds_the_fence_just_under_the_margin_and_follows_a_slow_step_at_once() {
    unheld(
        "a slow store just under the margin",
        Duration::from_millis(500),
        just_under_the_margin,
    );
}

/// One try of the case above: half a second of room.
fn just_under_the_margin() {
    // A write of three quarters of the margin, 1.5 s under 2 s: half a
    // second of room for a held host. A wait that put any share of the
    // step's own time on top of the period would land the renewal after
    // the fence closed: half that time, here, a quarter of a second
    // after it.
    let slow = led_over_a_slow_store(Duration::from_millis(2000), Duration::from_millis(1500), 1);
    bound!(
        (slow.closed, slow.unnamed) == (0, 0),
        "{} closed and {} unnamed of {} samples",
        slow.closed,
        slow.unnamed,
        slow.samples
    );

    // A step that took the period or more is followed at once: the next
    // write begins when the one before it returned, never a period
    // after that. A hold only parts two writes further, so the closest
    // pair tells.
    let period = Duration::from_millis(200);
    let takes = Duration::from_millis(300);
    let began = Arc::new(Mutex::new(Vec::new()));
    let store = WritesSlowly {
        lease: MemoryLease::new(),
        takes,
        slow: Arc::new(AtomicBool::new(true)),
        began: Arc::clone(&began),
    };
    let settings = ElectionSettings {
        duration: period * 3,
        renew_period: period,
        renew_deadline: period * 2,
    };
    let candidate = Elector::new(Box::new(store), "a@127.0.0.1:1", settings)
        .expect("the settings are in order");
    let (standing, _stood) = watch::channel(ElectionObservation::default());
    let running = start(candidate, Arc::new(MemoryStore::new()), standing).expect("a thread");
    until("the take and four renewals", || began.lock().len() >= 5);
    let began = began.lock().clone();
    running.stop();
    ended(running);
    let closest = began
        .windows(2)
        .map(|pair| pair[1].duration_since(pair[0]))
        .min()
        .expect("five writes began");
    assert!(
        closest >= takes,
        "a write began {closest:?} after the one before it"
    );
    assert!(
            closest < takes + period / 2,
            "the closest two writes began {closest:?} apart: a step of {takes:?} was not followed at once"
        );
}
