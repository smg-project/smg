//! The handoff of the leases' turn: a frame its hand applies where nobody
//! is at work or leaves to the thread at work or to a watch, the look, the
//! sweep and the place's apply that take up what waits, and what a tell
//! hears under the turn.

use super::*;

/// The count a host keeps of the row `id`, which stands in the store.
fn standing(store: &MemoryStore, id: WorkerId) -> Arc<AtomicU32> {
    handed_count(store, id).expect("the row stands in the store")
}

#[test]
fn a_frame_handed_where_nobody_is_at_work_is_applied_by_its_hand() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(2, now);
    let hand = Hand::new(GatewayId(1));
    let dispatch = Said::Dispatched(Dispatch::default(), now);
    store.hand(&hand, ids[0], dispatch);
    // Told before any other take of the turn, with the record as the frame
    // found it.
    let kept = told();
    assert_eq!(kept.len(), 1);
    assert_eq!(
        (kept[0].0.lease.id, kept[0].0.state, kept[0].1),
        (ids[0], LeaseState::ISSUED, dispatch)
    );
    // A second dispatch, a second release and a lease not on file tell
    // nothing.
    store.hand(&hand, ids[0], dispatch);
    store.hand(&hand, ids[0], GONE);
    store.hand(&hand, ids[0], GONE);
    store.hand(&hand, LeaseId(99), GONE);
    assert_eq!(told_ids(), [(1, GONE)]);
    assert_eq!(in_flight(&store), [(a, 1)]);
}

#[test]
fn a_frame_handed_while_a_thread_is_at_work_is_left_to_that_thread() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(3, now);
    let hand = Hand::new(GatewayId(1));
    let at_work = store.at_work();
    store.hand(&hand, ids[0], GONE);
    assert_eq!((told_ids(), in_flight(&store)), (vec![], vec![(a, 3)]));
    // The thread at work applies it before it reads the counts,
    at_work.apply();
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(1, GONE)], vec![(a, 2)])
    );
    // at its next issue,
    store.hand(&hand, ids[1], GONE);
    let (_, ended) = issued_at(&store, a, now);
    assert_eq!((ended, told_ids()), (vec![], vec![(2, GONE)]));
    // and once more as it leaves its work.
    store.hand(&hand, ids[2], GONE);
    assert_eq!(told_ids(), []);
    drop(at_work);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(3, GONE)], vec![(a, 1)])
    );
}

thread_local! {
    /// The world each tell of [`under`] was handed, by its address. A case
    /// holds every world it compares, so no later world stands where one
    /// of them stood.
    static UNDER: RefCell<Vec<*const StoreSnapshot>> = const { RefCell::new(Vec::new()) };
    /// Whether the next tell of [`under`] publishes a later world.
    static LATER: Cell<bool> = const { Cell::new(true) };
}

/// A teller that keeps the world each tell is handed. Its first tell
/// publishes a later world before it returns, so a word told behind it
/// under the same load reads the earlier world.
fn under(store: &MemoryStore, world: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
    UNDER.with_borrow_mut(|under| under.push(std::ptr::from_ref(world)));
    if LATER.replace(false) {
        ready(store, "http://b:1");
    }
}

/// A store under term 1 that tells through [`under`], with one ready
/// worker and `leases` leases issued on it at `now`, five seconds to their
/// dispatch: the store, the worker, the leases' ids and the world that
/// stands.
fn under_bed(
    leases: u64,
    now: Instant,
) -> (Arc<MemoryStore>, WorkerId, Vec<LeaseId>, Arc<StoreSnapshot>) {
    let store = Arc::new(MemoryStore::new());
    store.tell_through(under);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let ids = (0..leases).map(|_| LeaseId(1 << 32 | issued_at(&store, a, now).0));
    let ids = ids.collect();
    let first = store.snapshot();
    (store, a, ids, first)
}

/// A thread with no job at hand keeps its watch while a lease stands on
/// its file: a hand leaves its frame in the ring, and the thread's next
/// look applies it, or the first step of its next job.
#[test]
fn a_frame_handed_under_a_watch_is_left_to_the_watching_thread() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(3, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(
        place.look(),
        LookOutcome::Quiet,
        "three leases stand on the file"
    );
    assert_eq!(store.watches(), 1);
    store.hand(&hand, ids[0], GONE);
    assert_eq!((told_ids(), in_flight(&store)), (vec![], vec![(a, 3)]));
    // The thread's next look applies the frame,
    assert_eq!(place.look(), LookOutcome::Fed);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(1, GONE)], vec![(a, 2)])
    );
    // and so does the first step of its next job.
    store.hand(&hand, ids[1], GONE);
    assert_eq!(told_ids(), []);
    place.work();
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(2, GONE)], vec![(a, 1)])
    );
}

/// A watch starts at a look that finds a lease on file, never at a job.
/// A thread that rests behind each job shows none, as a caller of a door
/// in process: its hands read who is at work, and its rest applies what
/// they left. A thread that looked keeps its watch through the jobs
/// behind the look.
#[test]
fn a_watch_starts_at_a_look_and_stands_through_a_job() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(3, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(store.watches(), 0, "at work, and no look yet");
    store.hand(&hand, ids[0], GONE);
    assert_eq!(told_ids(), [], "left to the thread at work");
    place.rest();
    assert_eq!((told_ids(), store.watches()), (vec![(1, GONE)], 0));
    place.work();
    assert_eq!(store.watches(), 0, "a job starts no watch");
    assert_eq!(
        place.look(),
        LookOutcome::Quiet,
        "two leases stand on the file"
    );
    assert_eq!(store.watches(), 1);
    place.work();
    assert_eq!(store.watches(), 1, "the watch stands through the job");
    store.hand(&hand, ids[1], GONE);
    assert_eq!(told_ids(), []);
    place.apply();
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(2, GONE)], vec![(a, 1)])
    );
}

/// A look says whether it applied a frame, for its host to set the time
/// to its next look by: a frame of a lease on file, and one that names no
/// lease on file, since its gateway spoke all the same. A frame the
/// thread applied as it took up a job is behind it at its next look.
#[test]
fn a_look_says_whether_it_applied_a_frame() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(3, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(place.look(), LookOutcome::Quiet, "nothing was handed");
    store.hand(&hand, ids[0], GONE);
    assert_eq!(place.look(), LookOutcome::Fed);
    assert_eq!(place.look(), LookOutcome::Quiet, "the ring is empty again");
    // A second release of the lease that left: a frame all the same.
    store.hand(&hand, ids[0], GONE);
    assert_eq!(place.look(), LookOutcome::Fed);
    // What the thread applies at a job's first step is not its look's.
    store.hand(&hand, ids[1], GONE);
    place.work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    assert_eq!(in_flight(&store), [(a, 1)]);
}

/// A look that finds no lease on file ends the watch: the thread owes the
/// file no further look, and a hand applies in place again.
#[test]
fn a_look_that_finds_no_lease_on_file_ends_the_watch() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(1, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    store.hand(&hand, ids[0], GONE);
    // The look applies the release of the file's last lease.
    assert_eq!(place.look(), LookOutcome::Ended);
    assert_eq!(store.watches(), 0);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(1, GONE)], vec![(a, 0)])
    );
    // A lease laid on a file nobody watches: its frame is applied by its
    // hand.
    let (second, _) = issued_at(&store, a, now);
    store.hand(&hand, LeaseId(1 << 32 | second), GONE);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(second, GONE)], vec![(a, 0)])
    );
    assert_eq!(
        place.look(),
        LookOutcome::Ended,
        "no lease stands, and no watch"
    );
}

/// A watch ends by its word first and one more look behind it. A frame a
/// hand left under the old word is applied by that look, and a lease it
/// finds on file keeps the watch: the cases lay the lease and the frame
/// that reach a file between a look that found none and the word's end.
#[test]
fn a_watch_ends_behind_one_more_look() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(2, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    store.hand(&hand, ids[0], GONE);
    assert_eq!(told_ids(), []);
    // The end's own look applies the frame and finds the second lease:
    // the watch stands, and the next frame is left to it.
    assert_eq!(place.end_watch(), LookOutcome::Fed);
    assert_eq!(
        (told_ids(), in_flight(&store), store.watches()),
        (vec![(1, GONE)], vec![(a, 1)], 1)
    );
    store.hand(&hand, ids[1], GONE);
    assert_eq!(told_ids(), [], "left to the watch that stands");
    // With the last lease's release in the ring the end applies it, and
    // the watch ends: the next frame is applied by its hand.
    assert_eq!(place.end_watch(), LookOutcome::Ended);
    assert_eq!(
        (told_ids(), in_flight(&store), store.watches()),
        (vec![(2, GONE)], vec![(a, 0)], 0)
    );
    let (third, _) = issued_at(&store, a, now);
    store.hand(&hand, LeaseId(1 << 32 | third), GONE);
    assert_eq!(told_ids(), [(third, GONE)]);
}

/// A thread that rests, or drops its place, ends its watch whatever
/// stands on its file: the hands apply in place from there.
#[test]
fn a_place_that_rests_or_drops_ends_its_watch() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(4, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    store.hand(&hand, ids[0], GONE);
    assert_eq!(told_ids(), []);
    // Its rest applies what was handed, and ends the watch over the three
    // leases that stand.
    place.rest();
    assert_eq!((told_ids(), store.watches()), (vec![(1, GONE)], 0));
    store.hand(&hand, ids[1], GONE);
    assert_eq!(told_ids(), [(2, GONE)]);
    // A place dropped on its watch does the same.
    place.work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    store.hand(&hand, ids[2], GONE);
    assert_eq!(told_ids(), []);
    drop(place);
    assert_eq!((told_ids(), store.watches()), (vec![(3, GONE)], 0));
    store.hand(&hand, ids[3], GONE);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(4, GONE)], vec![(a, 0)])
    );
}

/// The sweep applies at a watched file while the file's thread waits: a
/// dispatch left in the ring is applied before the sweep reads a
/// deadline, so its lease is not ended as one no gateway dispatched.
#[test]
fn the_sweep_applies_at_a_watched_file_whose_thread_waits() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(2, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    let dispatch = Said::Dispatched(Dispatch::default(), now);
    store.hand(&hand, ids[0], dispatch);
    assert_eq!(told_ids(), []);
    // Past both leases' dispatch deadline.
    let ended = LeasesStore::expire(&*store, now + Duration::from_secs(6));
    let ended: Vec<_> = ended.iter().map(|found| found.record.lease.id).collect();
    assert_eq!(ended, [ids[1]]);
    let abandoned = Said::Released(None, ReleaseReason::Abandoned);
    assert_eq!(told_ids(), [(1, dispatch), (2, abandoned)]);
    assert_eq!(in_flight(&store), [(a, 1)]);
    assert_eq!(
        place.look(),
        LookOutcome::Quiet,
        "the dispatched lease stands"
    );
}

/// The sweep applies a release left by a thread that unwound even where
/// the dispatched lease has no place left in the deadline queue.
#[test]
fn the_sweep_applies_a_release_left_before_an_unwind() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(1, now);
    let hand = Hand::new(GatewayId(1));
    let dispatch = Said::Dispatched(Dispatch::default(), now);
    store.hand(&hand, ids[0], dispatch);
    assert_eq!(told_ids(), [(1, dispatch)]);
    assert!(LeasesStore::expire(&*store, now + Duration::from_secs(6)).is_empty());
    assert_eq!(store.queued(), 0, "the dispatched lease's place is popped");

    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut place = store.at_work();
        assert_eq!(place.look(), LookOutcome::Quiet);
        store.hand(&hand, ids[0], GONE);
        assert_eq!(told_ids(), []);
        panic!("leave the release for the sweep");
    }));
    assert!(unwound.is_err());

    assert!(LeasesStore::expire(&*store, now + Duration::from_secs(6)).is_empty());
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(1, GONE)], vec![(a, 0)])
    );
}

/// A full ring under a watch: the hand that finds it applies where the
/// watching thread waits, in the order the frames were handed, and no
/// frame is dropped.
#[test]
fn a_full_ring_under_a_watch_is_applied_by_its_hand() {
    let now = Instant::now();
    let room = u64::try_from(MemoryStore::HANDOVER).expect("the ring's room is a count");
    let (store, a, ids) = handed_bed(room + 2, now);
    let hand = Hand::new(GatewayId(1));
    let mut place = store.at_work();
    assert_eq!(place.look(), LookOutcome::Quiet);
    for id in &ids[..MemoryStore::HANDOVER] {
        store.hand(&hand, *id, GONE);
    }
    assert_eq!(told_ids(), [], "the ring holds its room");
    // The next hand finds the ring full and applies it; its own frame
    // waits for the watch.
    store.hand(&hand, ids[MemoryStore::HANDOVER], GONE);
    let applied = told_ids();
    assert_eq!(applied.len(), MemoryStore::HANDOVER);
    assert!(applied.iter().map(|(lease, _)| *lease).eq(1..=room));
    assert_eq!(
        place.look(),
        LookOutcome::Fed,
        "one lease stands behind the look"
    );
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(room + 1, GONE)], vec![(a, 1)])
    );
}

/// A place applies what waits at each other file whose thread waits, for a
/// decision under a stage that may refuse by a report of leases. A file
/// whose thread is at work is left to that thread, as a hand leaves it.
/// The second file's thread is the case's own, on its watch and then at
/// work.
#[test]
fn a_place_applies_what_waits_at_the_other_files_whose_threads_wait() {
    let now = Instant::now();
    let store = Arc::new(MemoryStore::new().with_decision_slots(2));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let (said, heard_of) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the other
        // thread's waits end, and the case with it.
        let (heard_of, go) = (heard_of, go);
        let store = &store;
        scope.spawn(move || {
            let _acting = DecisionSlot::new(1).enter();
            let ids: Vec<_> = (0..2).map(|_| issued_at(store, a, now).0).collect();
            let mut place = store.at_work();
            assert_eq!(
                place.look(),
                LookOutcome::Quiet,
                "two leases stand on its file"
            );
            let second = ids[1];
            said.send(ids).expect("the case hears the ids");
            word.recv_timeout(PATIENCE).expect("the case's word");
            place.work();
            said.send(Vec::new()).expect("the case hears the work");
            word.recv_timeout(PATIENCE).expect("the case's word");
            drop(place);
            assert_eq!(
                told_ids(),
                [(second, GONE)],
                "what waited is its own as it leaves"
            );
        });
        let _acting = DecisionSlot::new(0).enter();
        let place = store.at_work();
        let ids = heard_of.recv_timeout(PATIENCE).expect("the watch stands");
        let hand = Hand::new(GatewayId(1));
        drop(told());
        // The other file's thread waits on its watch: the frame a hand
        // left there is applied by this place.
        store.hand(&hand, LeaseId(1 << 32 | ids[0]), GONE);
        assert_eq!(told_ids(), [], "left to the watch");
        place.apply_waiting();
        assert_eq!(told_ids(), [(ids[0], GONE)]);
        // That thread is at work: what waits is its own.
        go.send(()).expect("the other thread waits");
        heard_of.recv_timeout(PATIENCE).expect("the work began");
        store.hand(&hand, LeaseId(1 << 32 | ids[1]), GONE);
        place.apply_waiting();
        assert_eq!(told_ids(), [], "left to the thread at work");
        go.send(()).expect("the other thread waits");
    });
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A file that shows no watch holds nothing for a look, so a place's apply
/// passes it by its word and takes no turn there. The case's tell runs
/// under the second file's turn, where a hand applies in place, and
/// applies through the place at the first file: a take of the held turn
/// would be refused.
#[test]
fn a_places_apply_takes_no_turn_at_a_file_that_shows_no_watch() {
    thread_local! {
        static PLACE: RefCell<Option<AtWork>> = const { RefCell::new(None) };
        static TOLD: Cell<u32> = const { Cell::new(0) };
    }
    fn under(_: &MemoryStore, _: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
        PLACE.with_borrow(|place| place.as_ref().expect("the case's place").apply_waiting());
        TOLD.set(TOLD.get() + 1);
    }
    let now = Instant::now();
    let store = Arc::new(MemoryStore::new().with_decision_slots(2));
    store.tell_through(under);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    // A lease on the second file, issued as its decider.
    let lease = {
        let _acting = DecisionSlot::new(1).enter();
        issued_at(&store, a, now).0
    };
    let _acting = DecisionSlot::new(0).enter();
    PLACE.set(Some(store.at_work()));
    // The second file shows no watch and no thread works there: the hand
    // applies in place, and the tell runs under that file's turn.
    store.hand(&Hand::new(GatewayId(1)), LeaseId(1 << 32 | lease), GONE);
    assert_eq!(TOLD.get(), 1, "the release was told under its file's turn");
    PLACE.set(None);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A place's apply takes its own file's turn by a wait and applies what
/// was handed while its thread worked. The thread's look at a request is
/// one try of that turn, which another file's decision may hold for a
/// read: a decision under a stage that may refuse by a report of leases
/// does not read by that try alone.
#[test]
fn a_places_apply_applies_what_was_handed_at_its_own_file() {
    let (store, a, ids) = handed_bed(1, Instant::now());
    let place = store.at_work();
    // Handed while the thread works: left to it.
    store.hand(&Hand::new(GatewayId(1)), ids[0], GONE);
    assert_eq!(told_ids(), [], "left to the thread at work");
    place.apply_waiting();
    assert_eq!(told_ids(), [(ids[0].0 & 0xffff_ffff, GONE)]);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A place's apply reads behind what a watch kept, whatever the watching
/// thread does meanwhile. A hand leaves a release under the second
/// file's watch and returns; this place then applies what waits at the
/// other files and reads the worker's count once, as a decision over
/// worker lists does: the count its host keeps, with no lock taken. In
/// every other round the second file's thread takes up a job as that
/// read is made: it says it works under its turn, with its apply behind
/// the mark, so the read finds the release applied, by this place or by
/// the job's first step. A hand at a file with no watch would have
/// applied it before it returned.
#[test]
fn a_places_apply_reads_behind_a_watchs_frames_beside_a_job_that_begins() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};

    const ROUNDS: u64 = 20_000;
    // The widest pause, in spins, between the word that lets the other
    // thread take up its job and this place's apply: the rounds sweep it.
    const SPINS: u64 = 64;
    // The words of a round, said by each side in turn: the other thread's
    // watch stands over the round's lease, it may take up its job, it is
    // at work, the round is over.
    const WATCHES: u64 = 1;
    const TAKE_UP: u64 = 2;
    const WORKS: u64 = 3;
    const OVER: u64 = 4;

    // Says the case left, by its end or by an unwind: the other thread
    // waits for no word behind it.
    struct Leaves<'a>(&'a AtomicBool);
    impl Drop for Leaves<'_> {
        fn drop(&mut self) {
            self.0.store(true, SeqCst);
        }
    }

    let now = Instant::now();
    let store = Arc::new(MemoryStore::new().with_decision_slots(2));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let (word, lease, left) = (AtomicU64::new(0), AtomicU64::new(0), AtomicBool::new(false));
    std::thread::scope(|scope| {
        let (store, word, lease, left) = (&store, &word, &lease, &left);
        scope.spawn(move || {
            let _acting = DecisionSlot::new(1).enter();
            let mut place = store.at_work();
            // Waits for the case's word, with `pause` between its reads;
            // `false` where the case left. The wait for its job is one
            // side of the race and spins. The wait for the round's end
            // is outside it and yields, as the case's own waits do: on a
            // host with no core to spare a spin can cost a time slice.
            let heard = |wanted: u64, pause: fn()| {
                while word.load(SeqCst) != wanted {
                    if left.load(SeqCst) {
                        return false;
                    }
                    pause();
                }
                true
            };
            loop {
                // At work: the round's lease, then the watch over it.
                let id = issued_at(store, a, now).0;
                assert_ne!(
                    place.look(),
                    LookOutcome::Ended,
                    "a lease stands on the file"
                );
                lease.store(id, SeqCst);
                word.store(WATCHES, SeqCst);
                if !heard(TAKE_UP, std::hint::spin_loop) {
                    return;
                }
                // A job: the first step of its work applies what waits.
                place.work();
                drop(told());
                word.store(WORKS, SeqCst);
                if !heard(OVER, std::thread::yield_now) {
                    return;
                }
            }
        });
        let _leaves = Leaves(left);
        let _acting = DecisionSlot::new(0).enter();
        let place = store.at_work();
        let hand = Hand::new(GatewayId(1));
        let wait_for = |wanted: u64| {
            let since = Instant::now();
            while word.load(SeqCst) != wanted {
                assert!(
                    since.elapsed() < PATIENCE,
                    "the other thread's word {wanted}"
                );
                std::thread::yield_now();
            }
        };
        let counted = standing(store, a);
        for round in 1..=ROUNDS {
            wait_for(WATCHES);
            drop(told());
            store.hand(&hand, LeaseId(1 << 32 | lease.load(SeqCst)), GONE);
            // Every other round the other thread takes up its job as this
            // place applies and reads, a swept pause apart.
            let begins = round % 2 == 0;
            if begins {
                word.store(TAKE_UP, SeqCst);
                for _ in 0..round / 2 % SPINS {
                    std::hint::spin_loop();
                }
            }
            place.apply_waiting();
            assert_eq!(
                counted.load(Ordering::Relaxed),
                0,
                "round {round}: the read is behind the release's apply"
            );
            if !begins {
                assert!(!told().is_empty(), "round {round}: this place applied it");
                word.store(TAKE_UP, SeqCst);
            }
            wait_for(WORKS);
            word.store(OVER, SeqCst);
        }
    });
}

/// The floor of the price of a place's apply at the other files, measured:
/// a million calls with nothing waiting and no other thread, at one, two
/// and eight files that show no watch, then at two and eight where each
/// other file shows one. Its own file's turn is in each call.
#[test]
#[ignore = "a measurement for the Numbers table, run by name"]
fn a_places_apply_at_the_other_files_measured() {
    use std::io::Write as _;

    const CALLS: u32 = 1_000_000;
    let mut out = std::io::stdout();
    let now = Instant::now();
    for (files, watched) in [(1_u8, false), (2, false), (8, false), (2, true), (8, true)] {
        let store = Arc::new(MemoryStore::new().with_decision_slots(usize::from(files)));
        let a = ready(&store, "http://a:1");
        LeasesStore::set_term(&*store, 1);
        // A watch at each other file: a lease on it and its place's look.
        let others = if watched { 1..files } else { 0..0 };
        let _watches: Vec<AtWork> = others
            .map(|other| {
                let _acting = DecisionSlot::new(other).enter();
                let mut place = store.at_work();
                issued_at(&store, a, now);
                assert_ne!(
                    place.look(),
                    LookOutcome::Ended,
                    "a lease stands on the file"
                );
                place
            })
            .collect();
        let _acting = DecisionSlot::new(0).enter();
        let place = store.at_work();
        let since = Instant::now();
        for _ in 0..CALLS {
            place.apply_waiting();
        }
        let nanos = since.elapsed().as_nanos() / u128::from(CALLS);
        let shown = if watched {
            "a watch at each other"
        } else {
            "no watch"
        };
        writeln!(
            out,
            "a place's apply, {files} files, {shown}, nothing waiting: {nanos} ns a call"
        )
        .expect("stdout");
    }
}

/// The frames of one take are told under one world, the store's as it
/// stood at the take's first tell: a take loads it once, however many
/// frames it tells.
#[test]
fn the_frames_of_a_take_are_told_under_one_world() {
    let (store, _, ids, first) = under_bed(3, Instant::now());
    let hand = Hand::new(GatewayId(1));
    let at_work = store.at_work();
    store.hand(&hand, ids[0], GONE);
    store.hand(&hand, ids[1], GONE);
    at_work.apply();
    let later = store.snapshot();
    assert!(!Arc::ptr_eq(&later, &first), "a later world stands");
    let (first, later) = (Arc::as_ptr(&first), Arc::as_ptr(&later));
    assert_eq!(UNDER.take(), [first, first], "one load for the take");
    store.hand(&hand, ids[2], GONE);
    at_work.apply();
    assert_eq!(UNDER.take(), [later], "the next take's own");
}

/// The leases one issue ends are told under one world, the store's as it
/// stood at the first end's tell: an issue loads it once for its ends.
/// The next issue's ends are told under its own.
#[test]
fn the_leases_an_issue_ends_are_told_under_one_world() {
    let now = Instant::now();
    let (store, a, _, first) = under_bed(3, now);
    // Six seconds on, all three are past their dispatch deadline: this
    // issue ends two, and the one behind it the third.
    let late = now + Duration::from_secs(6);
    issued_at(&store, a, late);
    let later = store.snapshot();
    assert!(!Arc::ptr_eq(&later, &first), "a later world stands");
    let (first, later) = (Arc::as_ptr(&first), Arc::as_ptr(&later));
    assert_eq!(
        UNDER.take(),
        [first, first],
        "one load for the issue's ends"
    );
    issued_at(&store, a, late);
    assert_eq!(UNDER.take(), [later], "the next issue's own");
}

/// The leases a sweep ends in a file are told under one world, the
/// store's as it stood at the first end's tell.
#[test]
fn the_leases_a_sweep_ends_in_a_file_are_told_under_one_world() {
    let now = Instant::now();
    let (store, _, ids, first) = under_bed(3, now);
    let ended = LeasesStore::expire(&*store, now + Duration::from_secs(6));
    assert_eq!(ended.len(), ids.len());
    assert!(
        !Arc::ptr_eq(&store.snapshot(), &first),
        "a later world stands"
    );
    let first = Arc::as_ptr(&first);
    assert_eq!(
        UNDER.take(),
        [first, first, first],
        "one load for the sweep"
    );
}

/// An issue's frames and its ends are two runs, told under a load each:
/// the frames under the world that stood at the apply's first tell, the
/// ends under the one that stands at the first end's. This holds the
/// split as a known limit, not as an aim: where one load serves both
/// runs, the ends read the first world and this case turns with it.
#[test]
fn an_issues_frames_and_its_ends_are_told_under_a_load_each() {
    let now = Instant::now();
    let (store, a, ids, first) = under_bed(2, now);
    let hand = Hand::new(GatewayId(1));
    let at_work = store.at_work();
    store.hand(&hand, ids[0], GONE);
    // Six seconds on the other lease is past its dispatch deadline: this
    // issue applies the handed frame, then ends that lease.
    issued_at(&store, a, now + Duration::from_secs(6));
    drop(at_work);
    let later = store.snapshot();
    assert!(!Arc::ptr_eq(&later, &first), "a later world stands");
    assert_eq!(
        UNDER.take(),
        [Arc::as_ptr(&first), Arc::as_ptr(&later)],
        "a load for each run"
    );
}

/// A decider looks at what is handed before a decision reads its rows'
/// counts, and that look never waits: a turn it finds held it reads past.
#[test]
fn a_thread_at_work_does_not_wait_for_a_held_turn_before_its_reads() {
    let store = Arc::new(MemoryStore::new());
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let (t0, world) = (Instant::now(), store.snapshot());
    let (began, working) = mpsc::sync_channel(1);
    let (look, asked) = mpsc::sync_channel::<()>(1);
    let (reached, parked) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    let (looked, seen) = mpsc::sync_channel(1);
    let store = &store;
    std::thread::scope(|scope| {
        // A thread is at work: its first step took the turn while it was
        // free. It takes it once more as it leaves, once the holder let go.
        scope.spawn(move || {
            let at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            asked
                .recv_timeout(PATIENCE)
                .expect("the case asks for the look");
            at_work.apply();
            looked.send(()).expect("the case waits for the look");
        });
        working
            .recv_timeout(PATIENCE)
            .expect("the thread is at work");
        // Another holder is parked under the turn, in an issue's tell.
        scope.spawn(move || {
            let record = booking(GatewayId(1), 0, a, t0);
            let issued = store.issue_decided(record, world.booked_of(a), |_| {
                reached.send(()).expect("the case hears the tell");
                word.recv_timeout(PATIENCE)
                    .expect("the case lets the issuer go");
            });
            issued.expect("minted under the store's term");
        });
        parked.recv_timeout(PATIENCE).expect("the issue tells");
        // The thread at work looks and reads on.
        look.send(())
            .expect("the thread at work waits for the word");
        let early = seen.recv_timeout(Duration::from_secs(2));
        go.send(()).expect("the issuer holds the other end");
        assert!(early.is_ok(), "the look waited for the turn");
    });
    assert!(store.unlocked());
}

#[test]
fn a_dispatch_handed_in_time_saves_its_lease_from_the_deadlines_end() {
    let t0 = Instant::now();
    let (store, a, ids) = handed_bed(2, t0);
    let hand = Hand::new(GatewayId(1));
    let _at_work = store.at_work();
    let in_time = t0 + Duration::from_secs(1);
    store.hand(
        &hand,
        ids[0],
        Said::Dispatched(Dispatch::default(), in_time),
    );
    // An issue past both deadlines applies what is handed before it ends
    // a deadline: the dispatched lease stands, the other is abandoned.
    let late = t0 + Duration::from_secs(6);
    assert_eq!(issued_at(&store, a, late).1, [2]);
    let state = |id| LeasesStore::get(&*store, GatewayId(1), id).map(|record| record.state);
    assert_eq!(state(ids[0]), Some(LeaseState::Dispatched { at: in_time }));
    assert_eq!(state(ids[1]), None);
}

#[test]
fn a_full_ring_makes_its_caller_apply_and_no_frame_is_dropped() {
    let now = Instant::now();
    let room = u64::try_from(MemoryStore::HANDOVER).expect("the ring's room is a count");
    let (store, a, ids) = handed_bed(room + 1, now);
    let hand = Hand::new(GatewayId(1));
    // A thread at work that does not come by: the frames wait in the ring.
    let at_work = store.at_work();
    for id in &ids {
        store.hand(&hand, *id, GONE);
    }
    // The last found the ring full and applied it, in the order the frames
    // were handed; its own frame waits.
    let applied = told_ids();
    assert_eq!(applied.len(), MemoryStore::HANDOVER);
    assert!(applied.iter().map(|(lease, _)| *lease).eq(1..=room));
    drop(at_work);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(room + 1, GONE)], vec![(a, 0)])
    );
}

#[test]
fn a_streams_end_applies_what_it_handed() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(2, now);
    let at_work = store.at_work();
    let hand = Hand::new(GatewayId(1));
    store.hand(&hand, ids[0], GONE);
    store.hand(&hand, ids[1], GONE);
    assert_eq!(told_ids(), []);
    // The stream's end does not wait for the thread at work.
    store.hand_back(hand);
    assert_eq!(
        (told_ids(), in_flight(&store)),
        (vec![(1, GONE), (2, GONE)], vec![(a, 0)])
    );
    drop(at_work);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "taken first or not at all")]
fn a_tell_that_takes_a_door_of_the_leases_is_refused_in_a_debug_build() {
    fn takes_a_door(
        store: &MemoryStore,
        _: &StoreSnapshot,
        _: &Sides,
        _: Said,
        _: Option<DecisionSlot>,
    ) {
        let _live = LeasesStore::live(store, GatewayId(1));
    }
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(takes_a_door);
    let record = booking(GatewayId(1), 0, a, Instant::now());
    let issued = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
    let hand = Hand::new(GatewayId(1));
    store.hand(&hand, issued.expect("minted").id, GONE);
}

/// One hand against one thread that leaves its work, the two started
/// together by a spin and moved a little each round. When both have ended
/// no door of the leases is asked: the handed frame is told by then, by the
/// thread that left or by the hand. A leaver that looked before it said it
/// leaves would let a frame wait on a lull.
#[test]
fn a_frame_handed_as_a_thread_leaves_its_work_is_applied_by_one_of_the_two() {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::SeqCst};
    static HEARD: AtomicUsize = AtomicUsize::new(0);
    fn counted(_: &MemoryStore, _: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
        HEARD.fetch_add(1, SeqCst);
    }
    const ROUNDS: usize = 100_000;
    /// How long the rounds run at most: past it the case asks no more.
    const RUN: Duration = Duration::from_secs(20);
    const BOUND: Duration = Duration::from_secs(60);
    /// Tells the case's two threads to end, where the case unwinds too.
    struct Ends<'a>(&'a AtomicUsize);
    impl Drop for Ends<'_> {
        fn drop(&mut self) {
            self.0.store(usize::MAX, SeqCst);
        }
    }
    let store = Arc::new(MemoryStore::new());
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(counted);
    let (gateway, now, world) = (GatewayId(1), Instant::now(), store.snapshot());
    let marks = [const { AtomicUsize::new(0) }; 4];
    let lease = AtomicU64::new(0);
    let (mut short, mut ran) = (0_usize, 0_usize);
    std::thread::scope(|scope| {
        let (store, [round, set, go, done], lease) = (&store, &marks, &lease);
        let next = move |seen: usize| loop {
            let asked = round.load(SeqCst);
            if asked != seen {
                break asked;
            }
            std::hint::spin_loop();
        };
        let start = move |asked: usize, spins: usize| {
            set.fetch_add(1, SeqCst);
            // The case's end is a word to go, too.
            while go.load(SeqCst) != asked && round.load(SeqCst) != usize::MAX {
                std::hint::spin_loop();
            }
            for _ in 0..spins {
                std::hint::spin_loop();
            }
        };
        scope.spawn(move || {
            let mut asked = next(0);
            while asked != usize::MAX {
                let at_work = store.at_work();
                start(asked, asked / 24 % 24);
                drop(at_work);
                done.fetch_add(1, SeqCst);
                asked = next(asked);
            }
        });
        scope.spawn(move || {
            let hand = Hand::new(gateway);
            let mut asked = next(0);
            while asked != usize::MAX {
                let id = LeaseId(lease.load(SeqCst));
                start(asked, asked % 24);
                store.hand(&hand, id, GONE);
                done.fetch_add(1, SeqCst);
                asked = next(asked);
            }
        });
        // A round's waits are bounded and the two threads end where this
        // one unwinds: a failure ends the case.
        let _ends = Ends(round);
        let wait = |count: &AtomicUsize, asked: usize| {
            while count.load(SeqCst) != 2 * asked {
                assert!(now.elapsed() < BOUND, "a thread of the case stopped");
                std::hint::spin_loop();
            }
        };
        for asked in 1..=ROUNDS {
            if now.elapsed() > RUN {
                break;
            }
            let record = booking(gateway, 0, a, now);
            let issued = store.issue_decided(record, world.booked_of(a), |_| ());
            lease.store(issued.expect("minted under the store's term").id.0, SeqCst);
            let before = HEARD.load(SeqCst);
            round.store(asked, SeqCst);
            wait(set, asked);
            go.store(asked, SeqCst);
            wait(done, asked);
            short += usize::from(HEARD.load(SeqCst) == before);
            ran = asked;
        }
    });
    assert!(ran > 0, "no round ran");
    assert_eq!(short, 0, "of {ran} rounds, a handed frame waited on a lull");
}

/// A frame handed while another thread holds the turn is left to the
/// thread at work, and that thread waits for the turn as it leaves: the
/// frame is told once the holder lets go, with no door of the leases
/// asked behind it.
#[test]
fn a_thread_that_leaves_its_work_waits_for_a_held_turn() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    static HEARD: AtomicUsize = AtomicUsize::new(0);
    fn counted(_: &MemoryStore, _: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
        HEARD.fetch_add(1, SeqCst);
    }
    let store = Arc::new(MemoryStore::new());
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(counted);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let first = store
        .issue_decided(
            booking(gateway, 0, a, now),
            store.snapshot().booked_of(a),
            |_| (),
        )
        .expect("minted under the store's term");
    let (began, working) = mpsc::sync_channel(1);
    let (leave, asked) = mpsc::sync_channel::<()>(1);
    let (reached, parked) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    let (left, gone) = mpsc::sync_channel(1);
    let (early, before) = std::thread::scope(|scope| {
        let store = &store;
        // One thread is at work: a frame handed from here on is left to
        // it. It leaves its work at the case's word, and is not through
        // while the turn is held.
        scope.spawn(move || {
            let at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            asked
                .recv_timeout(PATIENCE)
                .expect("the case says when to leave");
            drop(at_work);
            left.send(()).expect("the case waits for the leaver");
        });
        working
            .recv_timeout(PATIENCE)
            .expect("the thread is at work");
        // Another holder is parked under the turn, in an issue's hook.
        scope.spawn(move || {
            let record = booking(gateway, 0, a, now);
            let issued = store.issue_decided(record, store.snapshot().booked_of(a), |_| {
                reached.send(()).expect("the case hears the issue");
                word.recv_timeout(PATIENCE)
                    .expect("the case lets the issuer go");
            });
            issued.expect("minted under the store's term");
        });
        parked.recv_timeout(PATIENCE).expect("the issue tells");
        let hand = Hand::new(gateway);
        let before = HEARD.load(SeqCst);
        store.hand(&hand, first.id, GONE);
        leave
            .send(())
            .expect("the thread at work waits for the word");
        let early = gone.recv_timeout(Duration::from_millis(250)).is_ok();
        go.send(()).expect("the issuer holds the other end");
        (early, before)
    });
    assert!(!early, "the leaver was through under a held turn");
    let told = HEARD.load(SeqCst) - before;
    assert_eq!(
        told, 1,
        "the leaver applies what was handed under the held turn"
    );
}

/// A line that prints the store from under the turn, a tell's or a panic's,
/// does not take the turn again: it reads no count of the leases there.
#[test]
fn the_store_is_printed_from_under_its_leases_turn() {
    thread_local!(static PRINTED: RefCell<String> = const { RefCell::new(String::new()) });
    fn prints(store: &MemoryStore, _: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
        PRINTED.with(|printed| *printed.borrow_mut() = format!("{store:?}"));
    }
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(prints);
    let record = booking(GatewayId(1), 0, a, Instant::now());
    let issued = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
    let hand = Hand::new(GatewayId(1));
    store.hand(&hand, issued.expect("minted").id, GONE);
    let under = PRINTED.with(|printed| printed.borrow().clone());
    assert!(under.contains("leases: under the turn"), "{under}");
    let free = format!("{store:?}");
    assert!(free.contains("leases: 0"), "{free}");
}

#[test]
fn frames_handed_from_other_threads_are_all_applied_whoever_is_at_work() {
    const STREAMS: usize = 4;
    const ROUNDS: usize = 2_000;
    let store = Arc::new(MemoryStore::new());
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    let (gateway, t0) = (GatewayId(1), Instant::now());
    std::thread::scope(|scope| {
        let mut streams = Vec::new();
        for _ in 0..STREAMS {
            let (issued, ids) = mpsc::channel::<LeaseId>();
            let store = &store;
            scope.spawn(move || {
                let hand = Hand::new(gateway);
                for id in ids {
                    store.hand(&hand, id, Said::Dispatched(Dispatch::default(), t0));
                    store.hand(&hand, id, GONE);
                }
                store.hand_back(hand);
            });
            streams.push(issued);
        }
        // The deciding thread: at work for two rounds of three, away for
        // the third, so both the thread at work and the hands apply.
        let world = store.snapshot();
        for round in 0..ROUNDS {
            let _at_work = (round % 3 != 0).then(|| store.at_work());
            for stream in &streams {
                let record = booking(gateway, 0, a, t0);
                let lease = store
                    .issue_decided(record, world.booked_of(a), |_| ())
                    .expect("minted under the store's term");
                stream.send(lease.id).expect("its stream reads on");
            }
        }
    });
    // Every stream applied its ring before it left: no lease stands.
    assert_eq!(LeasesStore::live(&*store, gateway), 0);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A lease's events are told under the decider whose file the lease is on,
/// whichever thread applies them: its issuer's, where the store's host
/// named its deciders. A store of one file that no decider is named for
/// tells under no name.
#[test]
fn a_leases_events_are_told_under_its_files_decider() {
    let issue = |store: &MemoryStore, a| {
        let record = booking(GatewayId(1), 0, a, Instant::now());
        let issued = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
        issued.expect("minted under the store's term").id
    };
    let store = MemoryStore::new().with_decision_slots(4);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(heard);
    // One lease issued by a thread that acts as the fourth decider, one by
    // a thread that names none: the first file's.
    let fourth = {
        let _acting = DecisionSlot::new(3).enter();
        issue(&store, a)
    };
    let first = issue(&store, a);
    // Their ends are applied by a thread that names none.
    let hand = Hand::new(GatewayId(1));
    store.hand(&hand, fourth, GONE);
    store.hand(&hand, first, GONE);
    assert_eq!(
        told_by(),
        [Some(DecisionSlot::new(3)), Some(DecisionSlot::new(0))]
    );
    let plain = MemoryStore::new();
    let a = ready(&plain, "http://a:1");
    LeasesStore::set_term(&plain, 1);
    plain.tell_through(heard);
    let named = {
        let _acting = DecisionSlot::new(3).enter();
        issue(&plain, a)
    };
    plain.hand(&Hand::new(GatewayId(1)), named, GONE);
    assert_eq!(told_by(), [None]);
}
