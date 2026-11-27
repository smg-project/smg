//! The leases' files, one a decider: the file a decider issues into and
//! tells at, the doors and the sweep that reach every file, the ids across
//! a rise of the term and a claim, a store laid for deciders, and the
//! streak of lifecycle faults a store counts.

use super::*;

/// What a store told on any thread, by the store's address, in the order
/// told: a lease's counter, and what was said of it or none for its issue.
static TOLD_ACROSS: Mutex<Vec<(usize, u64, Option<Said>)>> = Mutex::new(Vec::new());

fn told_across(store: &MemoryStore, lease: LeaseId, said: Option<Said>) {
    let entry = (
        std::ptr::from_ref(store).addr(),
        lease.0 & 0xffff_ffff,
        said,
    );
    TOLD_ACROSS
        .lock()
        .expect("no case fails under it")
        .push(entry);
}

#[test]
fn a_lease_is_told_issued_before_any_holder_ends_it() {
    fn ended(
        store: &MemoryStore,
        _: &StoreSnapshot,
        found: &Sides,
        said: Said,
        _: Option<DecisionSlot>,
    ) {
        told_across(store, found.record.lease.id, Some(said));
    }
    let store = MemoryStore::new();
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(ended);
    let (t0, world) = (Instant::now(), store.snapshot());
    let issue = |at: Instant, parked: &dyn Fn()| {
        let record = LeaseRecord {
            dispatch_deadline: t0 + Duration::from_millis(1),
            ..booking(GatewayId(1), 0, a, at)
        };
        let issued = store.issue_decided(record, world.booked_of(a), |lease| {
            told_across(&store, lease.id, None);
            parked();
        });
        issued.expect("minted under the store's term");
    };
    let issue = &issue;
    let (reached, parked) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    let (done, swept) = mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        // The first issuer is parked in its tell, under the turn.
        scope.spawn(move || {
            issue(t0, &|| {
                reached.send(()).expect("the case hears the tell");
                word.recv_timeout(PATIENCE)
                    .expect("the case lets the issuer go");
            });
        });
        parked
            .recv_timeout(PATIENCE)
            .expect("the first issue tells");
        // A second issue past the first lease's deadline would end it: it
        // waits for the turn, so it has swept nothing a moment later.
        scope.spawn(move || {
            issue(t0 + Duration::from_secs(1), &|| ());
            done.send(()).expect("the case waits for the sweep");
        });
        let early = swept.recv_timeout(Duration::from_millis(100));
        assert!(early.is_err(), "the sweep waits for the issue's tell");
        go.send(()).expect("the issuer holds the other end");
        swept.recv_timeout(PATIENCE).expect("the second issue ends");
    });
    let me = std::ptr::from_ref(&store).addr();
    let told = TOLD_ACROSS.lock().expect("no case fails under it");
    let mine = told.iter().filter(|(by, ..)| *by == me);
    let mine: Vec<_> = mine.map(|(_, lease, said)| (*lease, *said)).collect();
    let abandoned = Some(Said::Released(None, ReleaseReason::Abandoned));
    // The second lease is due as it is filed: its own issue ends it, and
    // tells it issued first.
    assert_eq!(mine, [(1, None), (2, None), (1, abandoned), (2, abandoned)]);
}

/// A store made for its deciders holds a file of leases each. A decider
/// issues into its own file under its own turn and the id it mints names
/// that file: what is handed of a lease is applied there, under that
/// decider's name, while another decider's turn is held.
#[test]
fn a_decider_issues_into_its_own_file_and_a_leases_id_names_it() {
    static NAMES: Mutex<Vec<(LeaseId, Option<DecisionSlot>)>> = Mutex::new(Vec::new());
    fn named(_: &MemoryStore, _: &StoreSnapshot, found: &Sides, _: Said, by: Option<DecisionSlot>) {
        let mut names = NAMES.lock().expect("no case fails under it");
        names.push((found.record.lease.id, by));
    }
    let store = Arc::new(MemoryStore::new().with_decision_slots(3));
    assert_eq!(store.decision_slots(), 3);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(named);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let issue = |decider: u8| {
        let _acting = DecisionSlot::new(decider).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id
    };
    let (second, third) = (issue(1), issue(2));
    assert!(
        third.0.abs_diff(second.0) > 1,
        "a file mints in a block of its own: {second:?} {third:?}"
    );
    let (reached, parked) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the issuer's
        // wait ends, and the case with it.
        let (parked, go) = (parked, go);
        let store = &store;
        // The third decider holds its own turn, parked in an issue's hook.
        scope.spawn(move || {
            let _acting = DecisionSlot::new(2).enter();
            let record = booking(gateway, 0, a, now);
            let issued = store.issue_decided(record, store.snapshot().booked_of(a), |_| {
                reached.send(()).expect("the case hears the issue");
                word.recv_timeout(PATIENCE)
                    .expect("the case lets the issuer go");
            });
            issued.expect("minted under the store's term");
        });
        parked.recv_timeout(PATIENCE).expect("the issue tells");
        // The second's lease is read and ended at its own file meanwhile.
        let on_file = LeasesStore::get(&**store, gateway, second);
        assert_eq!(on_file.map(|record| record.lease.id), Some(second));
        store.hand(&Hand::new(gateway), second, GONE);
        assert_eq!(LeasesStore::get(&**store, gateway, second), None);
        go.send(()).expect("the issuer holds the other end");
    });
    store.hand(&Hand::new(gateway), third, GONE);
    let names = NAMES.lock().expect("no case fails under it");
    let (second, third) = (
        (second, DecisionSlot::new(1)),
        (third, DecisionSlot::new(2)),
    );
    assert_eq!(*names, [second, third].map(|(lease, by)| (lease, Some(by))));
    drop(names);
    assert_eq!(LeasesStore::live(&*store, gateway), 1);
}

/// A restore hands each lease's end to its file. Where the file's decider
/// is at work the restore returns with the lease on file and nothing
/// told: that decider tells the end once, at its next take. A frame
/// handed ahead of the restore for a lease is told in its order, and the
/// restore's end of that lease then tells nothing.
#[test]
fn a_restore_leaves_a_working_deciders_leases_to_its_next_take() {
    let now = Instant::now();
    let (store, a, ids) = handed_bed(2, now);
    let leader = MemoryStore::new();
    assert_eq!(ready(&leader, "http://a:1"), a);
    let cut = leader.checkpoint();
    let (began, working) = mpsc::sync_channel(1);
    let (look, asked) = mpsc::sync_channel::<()>(1);
    let (looked, seen) = mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the thread at
        // work's wait ends, and the case with it.
        let (working, look, seen) = (working, look, seen);
        let store = &store;
        scope.spawn(move || {
            let at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            asked
                .recv_timeout(PATIENCE)
                .expect("the case asks for the look");
            at_work.apply();
            looked
                .send(told_ids())
                .expect("the case waits for the look");
        });
        working
            .recv_timeout(PATIENCE)
            .expect("the thread is at work");
        // A gateway's frame for the first lease, ahead of the restore.
        store.hand(&Hand::new(GatewayId(1)), ids[0], GONE);
        assert!(store.restore_from(cut));
        // The restore is back: both leases stand and nothing is told.
        assert_eq!(LeasesStore::live(&**store, GatewayId(1)), 2);
        assert_eq!(told_ids(), []);
        look.send(())
            .expect("the thread at work waits for the word");
        let removed = Said::Released(None, ReleaseReason::WorkerRemoved);
        assert_eq!(
            seen.recv_timeout(PATIENCE).expect("the look"),
            [(1, GONE), (2, removed)]
        );
    });
    assert_eq!(LeasesStore::live(&*store, GatewayId(1)), 0);
    assert_eq!(store.queued(), 0);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// While a decider is at work no other thread acts at its file: a
/// stream's end, the sweep of the due leases and a door each find it at
/// work under the turn and apply nothing. What waits is the decider's.
#[test]
fn no_thread_acts_at_a_file_beside_the_thread_at_work() {
    let now = Instant::now();
    let (store, _, ids) = handed_bed(2, now);
    let (began, working) = mpsc::sync_channel(1);
    let (look, asked) = mpsc::sync_channel::<()>(1);
    let (looked, seen) = mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the thread at
        // work's wait ends, and the case with it.
        let (working, look, seen) = (working, look, seen);
        let store = &store;
        scope.spawn(move || {
            let at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            asked
                .recv_timeout(PATIENCE)
                .expect("the case asks for the look");
            at_work.apply();
            looked
                .send(told_ids())
                .expect("the case waits for the look");
        });
        working
            .recv_timeout(PATIENCE)
            .expect("the thread is at work");
        let hand = Hand::new(GatewayId(1));
        store.hand(&hand, ids[0], GONE);
        // The stream's end, the sweep past every deadline and a door's read.
        store.hand_back(hand);
        let late = now + Duration::from_secs(60);
        assert!(LeasesStore::expire(&**store, late).is_empty());
        assert_eq!(LeasesStore::live(&**store, GatewayId(1)), 2);
        assert_eq!(told_ids(), []);
        // The thread at work applies what was handed at its next look.
        look.send(())
            .expect("the thread at work waits for the word");
        assert_eq!(seen.recv_timeout(PATIENCE).expect("the look"), [(1, GONE)]);
    });
    assert_eq!(LeasesStore::live(&*store, GatewayId(1)), 1);
}

/// A decider that takes up its work waits out a hand that applies at its
/// file. The hand found no thread at work, took the file's turn and tells
/// a lease's event under the decider's name: the decider says it works,
/// then takes that turn, so no thread tells under its name beside its
/// first decision.
#[test]
fn a_decider_takes_up_its_work_behind_a_hand_that_tells_at_its_file() {
    type Ends = (mpsc::SyncSender<()>, mpsc::Receiver<()>);
    static ENDS: Mutex<Option<Ends>> = Mutex::new(None);
    fn parks(_: &MemoryStore, _: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
        let ends = ENDS.lock().expect("no case fails under it").take();
        if let Some((reached, word)) = ends {
            reached.send(()).expect("the case hears the tell");
            word.recv_timeout(PATIENCE)
                .expect("the case lets the teller go");
        }
    }
    let store = Arc::new(MemoryStore::new().with_decision_slots(2));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(parks);
    let (gateway, now) = (GatewayId(1), Instant::now());
    // A lease on the second decider's file.
    let lease = {
        let _acting = DecisionSlot::new(1).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id
    };
    let (reached, parked) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    *ENDS.lock().expect("no case fails under it") = Some((reached, word));
    let (began, working) = mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the teller's
        // wait ends, and the case with it.
        let (parked, go, working) = (parked, go, working);
        let store = &store;
        // A hand finds no thread at work at the file: it takes the turn
        // and stands in the tell of the frame it applies.
        scope.spawn(move || store.hand(&Hand::new(gateway), lease, GONE));
        // Bounded: where no tell is made the case fails, and does not
        // wait on an end the teller holds.
        let told = parked.recv_timeout(PATIENCE);
        told.expect("the hand's tell");
        // The file's decider takes up its work behind that turn.
        scope.spawn(move || {
            let _acting = DecisionSlot::new(1).enter();
            let at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            drop(at_work);
        });
        let early = working.recv_timeout(Duration::from_millis(250));
        assert!(early.is_err(), "a decider worked beside a hand's tell");
        go.send(()).expect("the teller holds the other end");
        let at_work = working.recv_timeout(PATIENCE);
        at_work.expect("the decider is at work behind the tell");
    });
    assert_eq!(LeasesStore::live(&*store, gateway), 0);
}

/// A full ring stops its hand until a holder applied: the thread at work
/// at the file, which the hand yields to. No frame is dropped, and none is
/// applied by the hand beside that thread.
#[test]
fn a_full_ring_stops_its_hand_until_the_thread_at_work_applied() {
    let now = Instant::now();
    let room = u64::try_from(MemoryStore::HANDOVER).expect("the ring's room is a count");
    let (store, a, ids) = handed_bed(room + 1, now);
    let (began, working) = mpsc::sync_channel(1);
    let (look, asked) = mpsc::sync_channel::<()>(1);
    let (looked, seen) = mpsc::sync_channel(1);
    let (handed, through) = mpsc::sync_channel(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the thread at
        // work's wait ends, and the case with it.
        let (working, look, seen, through) = (working, look, seen, through);
        let (store, ids) = (&store, &ids);
        // A thread at work that does not come by until it is asked.
        scope.spawn(move || {
            let at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            asked
                .recv_timeout(PATIENCE)
                .expect("the case asks for the look");
            at_work.apply();
            drop(at_work);
            looked
                .send(told_ids())
                .expect("the case waits for the look");
        });
        working
            .recv_timeout(PATIENCE)
            .expect("the thread is at work");
        // A hand fills the ring and stops at the frame past it.
        scope.spawn(move || {
            let hand = Hand::new(GatewayId(1));
            for id in ids {
                store.hand(&hand, *id, GONE);
            }
            handed
                .send(told_ids())
                .expect("the case waits for the hand");
        });
        let early = through.recv_timeout(Duration::from_millis(250));
        assert!(early.is_err(), "the hand was through a full ring");
        look.send(())
            .expect("the thread at work waits for the word");
        let by_hand = through.recv_timeout(PATIENCE);
        let by_hand = by_hand.expect("the hand is through");
        // The ring was the thread at work's to apply. The frame past it
        // is that thread's too, or its hand's once the thread had left:
        // every frame is told once, in the order it was handed.
        let mut applied = seen.recv_timeout(PATIENCE).expect("the look");
        assert!(applied.len() >= MemoryStore::HANDOVER);
        applied.extend(by_hand);
        assert!(applied.iter().map(|(lease, _)| *lease).eq(1..=room + 1));
    });
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A file whose decider rests ends no lease at an issue: the sweep ends
/// its leases past their dispatch deadline, each told under the file's
/// decider. A file whose decider is at work is left to it.
#[test]
fn the_sweep_ends_a_resting_files_due_leases_under_its_deciders_name() {
    static NAMES: Mutex<Vec<(LeaseId, Option<DecisionSlot>)>> = Mutex::new(Vec::new());
    fn named(_: &MemoryStore, _: &StoreSnapshot, found: &Sides, _: Said, by: Option<DecisionSlot>) {
        let mut names = NAMES.lock().expect("no case fails under it");
        names.push((found.record.lease.id, by));
    }
    let store = Arc::new(MemoryStore::new().with_decision_slots(2));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&*store, 1);
    store.tell_through(named);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let issue = |decider: u8| {
        let _acting = DecisionSlot::new(decider).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id
    };
    let (first, second) = (issue(0), issue(1));
    let late = now + Duration::from_secs(60);
    let ended = |store: &MemoryStore| {
        let ended = LeasesStore::expire(store, late);
        ended
            .iter()
            .map(|found| found.record.lease.id)
            .collect::<Vec<_>>()
    };
    let (began, working) = mpsc::sync_channel(1);
    let (leave, asked) = mpsc::sync_channel::<()>(1);
    std::thread::scope(|scope| {
        // The case's own ends go with it where it fails: the decider's
        // wait ends, and the case with it.
        let (working, leave) = (working, leave);
        let store = &store;
        // The first decider is at work: the sweep leaves its file to it.
        scope.spawn(move || {
            let _acting = DecisionSlot::new(0).enter();
            let _at_work = store.at_work();
            began.send(()).expect("the case waits for the seat");
            asked
                .recv_timeout(PATIENCE)
                .expect("the case says when to leave");
        });
        working
            .recv_timeout(PATIENCE)
            .expect("the decider is at work");
        assert_eq!(ended(store), [second]);
        leave.send(()).expect("the decider waits for the word");
    });
    assert_eq!(ended(&store), [first]);
    let names = NAMES.lock().expect("no case fails under it");
    let told = [
        (second, DecisionSlot::new(1)),
        (first, DecisionSlot::new(0)),
    ];
    assert_eq!(*names, told.map(|(lease, by)| (lease, Some(by))));
    drop(names);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// A gateway's leases stand in the files of the deciders that issued them:
/// its live count, its restore and its end reach every file.
#[test]
fn a_gateways_doors_reach_its_leases_in_every_file() {
    let store = MemoryStore::new().with_decision_slots(3);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    let (ours, theirs, now) = (GatewayId(1), GatewayId(2), Instant::now());
    let issue = |decider: u8, gateway| {
        let _acting = DecisionSlot::new(decider).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id
    };
    let leases = [issue(0, ours), issue(1, ours), issue(2, ours)];
    let other = issue(1, theirs);
    let doors: &dyn LeasesStore = &store;
    assert_eq!((doors.live(ours), doors.live(theirs)), (3, 1));
    assert_eq!(in_flight(&store), [(a, 4)]);
    // A restore keeps the leases its gateway presents, each in the file
    // its id names, and ends the gateway's others in every file.
    let kept = doors.get(ours, leases[2]).expect("on its decider's file");
    assert_eq!(doors.restore(ours, &[kept]), 1);
    let on_file = leases.map(|lease| doors.get(ours, lease).is_some());
    assert_eq!(on_file, [false, false, true]);
    assert_eq!(in_flight(&store), [(a, 2)]);
    // The gateways' end takes each one's leases out of every file.
    assert_eq!(doors.drop_gateways(&[ours, theirs]), [1, 1]);
    assert_eq!(doors.get(theirs, other), None);
    assert_eq!(in_flight(&store), [(a, 0)]);
}

/// The start of a term the store follows raises the term on every file
/// and tells the blocks, as the term's own door does: the table of a
/// term that holds a lease is kept, and each file mints under the
/// followed term. A start told again, or an older term's, raises nothing.
#[test]
fn a_followed_terms_start_raises_every_file_and_starts_the_blocks_under_it() {
    let standby = MemoryStore::new().with_decision_slots(2);
    let term: &dyn LeasesStore = &standby;
    term.set_term(3);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let issue = |decider: u8, under: u64| {
        let _acting = DecisionSlot::new(decider).enter();
        standby.issue_decided(lease(gateway, 0, under, now), None, |_| ())
    };
    issue(1, 3).expect("minted under the term the standby stood at");
    standby.follow_term(Position { seq: 1, term: 7 });
    assert_eq!(term.term(), 7);
    assert_eq!(
        standby.kept_terms(),
        [3],
        "the rise is told the ids on file"
    );
    for decider in [0, 1] {
        let minted = issue(decider, 7).map(|lease| lease.id.0 >> 32);
        assert_eq!(minted, Ok(7), "the file of decider {decider}");
    }
    standby.follow_term(Position { seq: 2, term: 5 });
    assert_eq!(term.term(), 7);
}

/// A lease's id names its file for as long as the lease can be on file. A
/// frame for a lease of the term before reaches the file of the decider
/// that issued it, behind a rise under which another decider claimed the
/// same block.
#[test]
fn a_lease_of_the_term_before_is_found_in_its_file_behind_a_rise() {
    static NAMES: Mutex<Vec<(LeaseId, Option<DecisionSlot>)>> = Mutex::new(Vec::new());
    fn named(_: &MemoryStore, _: &StoreSnapshot, found: &Sides, _: Said, by: Option<DecisionSlot>) {
        let mut names = NAMES.lock().expect("no case fails under it");
        names.push((found.record.lease.id, by));
    }
    let store = MemoryStore::new().with_decision_slots(3);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(named);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let issue = |decider: u8, term: u64| {
        let _acting = DecisionSlot::new(decider).enter();
        let record = LeaseRecord {
            target: a,
            ..lease(gateway, 0, term, now)
        };
        let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id
    };
    // The third decider claims the first block of term 1, the second
    // decider the first block of term 2.
    let before = issue(2, 1);
    LeasesStore::set_term(&store, 2);
    let after = issue(1, 2);
    assert_eq!(before.0 & 0xffff_ffff, after.0 & 0xffff_ffff);
    assert_eq!(store.kept_terms(), [1]);
    let hand = Hand::new(gateway);
    store.hand(&hand, before, GONE);
    store.hand(&hand, after, GONE);
    let names = NAMES.lock().expect("no case fails under it");
    let told = [
        (before, DecisionSlot::new(2)),
        (after, DecisionSlot::new(1)),
    ];
    assert_eq!(*names, told.map(|(lease, by)| (lease, Some(by))));
    drop(names);
    assert_eq!(LeasesStore::live(&store, gateway), 0);
}

/// A streak of lifecycle calls that panicked says its first alone: each is
/// counted, and a call that returns ends the streak.
#[test]
fn a_streak_of_lifecycle_faults_says_its_first_alone() {
    let store = MemoryStore::new();
    assert!(store.call_faulted());
    assert!(!store.call_faulted());
    store.call_returned();
    assert_eq!(store.lifecycle_faults(), 2);
    assert!(store.call_faulted());
    assert_eq!(store.lifecycle_faults(), 3);
}

/// An id names its file for as long as its lease can be on file, so the
/// doors that take an id from their caller file none whose file a later
/// claim would name anew: an id of the store's term from a block no
/// decider claimed, or of a term the store can still take. An id of a
/// term before is filed by its block, and one of a claimed block in the
/// file that claimed it. A store of one file takes every id, as it did.
#[test]
fn a_door_files_no_id_whose_file_a_claim_would_name_anew() {
    let (gateway, now) = (GatewayId(1), Instant::now());
    let by_hand = |term: u64, counter: u64| lease(gateway, term << 32 | counter, term.min(3), now);
    let store = MemoryStore::new().with_decision_slots(2);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 3);
    let doors: &dyn LeasesStore = &store;
    // No decider minted under term 3 yet: its first block is no file's.
    assert_eq!(doors.issue(by_hand(3, 9)), Err(LeaseRefused::Unclaimed));
    assert_eq!(doors.issue(by_hand(4, 9)), Err(LeaseRefused::Unclaimed));
    assert_eq!(doors.restore(gateway, &[by_hand(3, 9), by_hand(4, 9)]), 0);
    // An id of a term before names its file by its block: for good.
    assert_eq!(doors.issue(by_hand(2, 9)), Ok(()));
    assert!(doors.get(gateway, LeaseId(2 << 32 | 9)).is_some());
    // The second decider claims the first block of term 3: an id of that
    // block names its file, and the doors file it there.
    let minted = {
        let _acting = DecisionSlot::new(1).enter();
        let record = LeaseRecord {
            target: a,
            ..lease(gateway, 0, 3, now)
        };
        let issued = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
        issued.expect("minted under the store's term").id
    };
    assert_eq!(minted, LeaseId(3 << 32 | 1));
    assert_eq!(doors.issue(by_hand(3, 9)), Ok(()));
    assert_eq!(doors.restore(gateway, &[by_hand(3, 9), by_hand(2, 9)]), 2);
    let hand = Hand::new(gateway);
    store.hand(&hand, LeaseId(3 << 32 | 9), GONE);
    assert_eq!(doors.get(gateway, LeaseId(3 << 32 | 9)), None);
    assert_eq!(doors.live(gateway), 1);
    // One file: every id names it.
    let one = MemoryStore::new();
    LeasesStore::set_term(&one, 3);
    assert_eq!(LeasesStore::issue(&one, by_hand(3, 9)), Ok(()));
    assert_eq!(LeasesStore::restore(&one, gateway, &[by_hand(4, 9)]), 1);
}

/// A door asks the file an id names once more under the turn it took. A
/// door that waited for the first file's turn while a decider claimed the
/// id's block files the lease in that decider's file, where its frames
/// look for it. No hook says the door reached its wait, so the case gives
/// it time: a door that came late reads the claimed byte and passes too.
#[test]
fn a_door_that_waited_behind_a_claim_files_its_id_where_the_claim_named_it() {
    let (gateway, now) = (GatewayId(1), Instant::now());
    let store = MemoryStore::new().with_decision_slots(2);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 3);
    let (store, booked) = (&store, store.snapshot().booked_of(a));
    let record = LeaseRecord {
        target: a,
        ..lease(gateway, 0, 3, now)
    };
    // An id of the term's second block, which no decider claimed yet: it
    // reads the first file.
    let by_hand = lease(gateway, 3 << 32 | 1 << 20 | 9, 3, now);
    let (reached, parked) = mpsc::sync_channel(1);
    let (go, word) = mpsc::sync_channel::<()>(1);
    let filed = std::thread::scope(|scope| {
        // The first decider claims the first block and is parked under its
        // turn, in its issue's tell.
        scope.spawn(move || {
            let _acting = DecisionSlot::new(0).enter();
            let issued = store.issue_decided(record, booked, |_| {
                reached.send(()).expect("the case hears the tell");
                word.recv().expect("the case lets the issuer go");
            });
            issued.expect("minted under the store's term");
        });
        parked.recv().expect("the issue tells");
        // The door reads the first file and waits for its turn.
        let door = scope.spawn(move || LeasesStore::issue(store, by_hand));
        std::thread::sleep(Duration::from_millis(200));
        // The second decider claims the second block, the id's.
        let _acting = DecisionSlot::new(1).enter();
        let second = store.issue_decided(record, booked, |_| ());
        go.send(()).expect("the issuer holds the other end");
        let filed = door.join().expect("the door returns");
        (second.map(|lease| lease.id), filed)
    });
    assert_eq!(filed, (Ok(LeaseId(3 << 32 | 1 << 20)), Ok(())));
    assert!(LeasesStore::get(store, gateway, by_hand.lease.id).is_some());
}

/// Once the store's term is past 31 bits no decider mints and no term
/// leads the blocks anew, so the doors file every id where it reads its
/// file: an id of the last term from a block no decider claimed, and an
/// id of a term that never led here. A restore keeps both.
#[test]
fn the_doors_file_every_id_once_the_term_is_past_31_bits() {
    let (gateway, now) = (GatewayId(1), Instant::now());
    let by_hand = |term: u64, counter: u64| lease(gateway, term << 32 | counter, term, now);
    let store = MemoryStore::new().with_decision_slots(2);
    LeasesStore::set_term(&store, 4);
    let doors: &dyn LeasesStore = &store;
    let (unclaimed, never) = (by_hand(4, 9), by_hand(10, 5 << 20 | 1));
    assert_eq!(doors.issue(unclaimed), Err(LeaseRefused::Unclaimed));
    LeasesStore::set_term(&store, 1 << 31);
    assert_eq!(
        (doors.issue(unclaimed), doors.issue(never)),
        (Ok(()), Ok(()))
    );
    let on_file = |record: LeaseRecord| doors.get(gateway, record.lease.id).is_some();
    assert!(on_file(unclaimed) && on_file(never));
    assert_eq!(doors.restore(gateway, &[unclaimed, never]), 2);
    assert!(on_file(unclaimed) && on_file(never));
}

/// The most deciders a host names each issue a first lease: every id is
/// far below its term's mark, three quarters of the counter, so no count
/// of deciders hands a term over at its first leases.
#[test]
fn the_first_leases_of_the_most_deciders_are_far_below_the_terms_mark() {
    let store = MemoryStore::new().with_decision_slots(256);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let first = |decider: u8| {
        let _acting = DecisionSlot::new(decider).enter();
        let record = booking(gateway, 0, a, now);
        let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
        lease.expect("minted under the store's term").id.0 & 0xffff_ffff
    };
    let counters: Vec<u64> = (0..=u8::MAX).map(first).collect();
    let mark = 3_u64 << 30;
    let highest = counters.iter().max().copied().unwrap_or(mark);
    assert!(highest * 12 <= mark, "{highest} beside the mark {mark}");
    assert_eq!(LeasesStore::live(&store, gateway), 256);
    assert_eq!(in_flight(&store), [(a, 256)]);
}

/// The files laid for a host's deciders keep the store's word to the
/// engine and its fault count.
#[test]
fn the_files_laid_for_deciders_keep_the_word_and_the_fault_count() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    static HEARD: AtomicUsize = AtomicUsize::new(0);
    fn counted(_: &MemoryStore, _: &StoreSnapshot, _: &Sides, _: Said, _: Option<DecisionSlot>) {
        HEARD.fetch_add(1, SeqCst);
    }
    let store = MemoryStore::new();
    store.tell_through(counted);
    assert!(store.call_faulted());
    let store = store.with_decision_slots(2);
    assert_eq!((store.decision_slots(), store.lifecycle_faults()), (2, 1));
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let record = booking(gateway, 0, a, now);
    let lease = store.issue_decided(record, store.snapshot().booked_of(a), |_| ());
    let lease = lease.expect("minted under the store's term").id;
    store.hand(&Hand::new(gateway), lease, GONE);
    assert_eq!(HEARD.load(SeqCst), 1, "told through the word given before");
}

/// A store with a lease on file is not laid for deciders: the lease's
/// file would go from under it.
#[test]
#[should_panic(expected = "a store's files are laid before a lease is filed")]
fn a_store_with_a_lease_on_file_is_not_laid_for_deciders() {
    let store = MemoryStore::new();
    let filed = LeasesStore::issue(&store, lease(GatewayId(1), 1, 0, Instant::now()));
    filed.expect("a lease");
    drop(store.with_decision_slots(2));
}

/// Nor is a store laid again with a lease on a later file: the guard
/// reads every file, each under its own turn.
#[test]
#[should_panic(expected = "a store's files are laid before a lease is filed")]
fn a_store_with_a_lease_on_a_later_file_is_not_laid_again() {
    let store = MemoryStore::new().with_decision_slots(4);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    {
        // A thread that acts as the fourth decider files on the fourth
        // file: the first file stays empty.
        let (_acting, world) = (DecisionSlot::new(3).enter(), store.snapshot());
        let record = booking(GatewayId(1), 0, a, Instant::now());
        let issued = store.issue_decided(record, world.booked_of(a), |_| ());
        issued.expect("minted under the store's term");
    }
    drop(store.with_decision_slots(8));
}
