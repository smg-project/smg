//! The count a store's host keeps for a worker ([`HostCount`]): the store
//! keeps it equal to the worker's cell at every door of the leases, a
//! lease filed ahead of it is on it, and it goes with its row.

use std::{
    sync::{
        atomic::{AtomicI64, AtomicU32, Ordering::SeqCst},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use super::{booked, booking, done, facts, followed, issue_pair, on, pair, ready};
use crate::{
    engine::{
        decision::{Target, Targets},
        view::WorkerId,
    },
    store::{
        GatewayId, JournalKinds, LeasesStore, MemoryStore, PickRecord, WorkerFacts, WorkersStore,
    },
    tracking::{
        lease::{LeaseId, ReleaseReason, Side},
        HostCount,
    },
};

/// What a host counts for one worker, read from outside the store: the
/// leases it holds, and how many of the counts handed over it the store
/// dropped.
#[derive(Default)]
struct Kept {
    held: AtomicI64,
    dropped: AtomicU32,
}

impl Kept {
    fn held(&self) -> i64 {
        self.held.load(SeqCst)
    }

    fn dropped(&self) -> u32 {
        self.dropped.load(SeqCst)
    }
}

/// A host's count: it takes each step, refuses an end no booking stands
/// for, and says when it is dropped. Two counts over one `Kept` step one
/// number.
struct Count(Arc<Kept>);

impl HostCount for Count {
    fn booked(&self) {
        self.0.held.fetch_add(1, SeqCst);
    }

    fn freed(&self) {
        let held = self.0.held.fetch_sub(1, SeqCst);
        assert!(held > 0, "an end with no booking under it");
    }
}

impl Drop for Count {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, SeqCst);
    }
}

/// Hands the store a count for `id` that steps `kept`.
fn hand(store: &MemoryStore, id: WorkerId, kept: &Arc<Kept>) {
    assert!(store.host_count(id, Box::new(Count(Arc::clone(kept)))));
}

/// Hands the store a count for `id`: what it holds.
fn handed(store: &MemoryStore, id: WorkerId) -> Arc<Kept> {
    let kept = Arc::new(Kept::default());
    hand(store, id, &kept);
    kept
}

/// Takes the row `id` out of the store.
fn remove(store: &MemoryStore, id: WorkerId) {
    let revision = WorkersStore::get(store, id).expect("the row").revision;
    WorkersStore::remove(store, id, revision).expect("removed");
}

/// Three ready workers with a count handed for each, and the read of both
/// counts: each worker's count is its cell's, under the step's name.
fn fleet() -> (
    MemoryStore,
    [WorkerId; 3],
    impl Fn(&MemoryStore, &str) -> Vec<u32>,
) {
    let store = MemoryStore::new();
    let workers = ["http://a:1", "http://b:1", "http://c:1"].map(|key| ready(&store, key));
    let counts = workers.map(|id| handed(&store, id));
    let both = move |store: &MemoryStore, step: &str| {
        let cells = booked(store, &workers);
        let hosts: Vec<i64> = counts.iter().map(|kept| kept.held()).collect();
        let cells_wide: Vec<i64> = cells.iter().map(|&cell| i64::from(cell)).collect();
        assert_eq!(hosts, cells_wide, "{step}: each host's count is its cell's");
        cells
    };
    (store, workers, both)
}

/// Every door of the leases steps a host's count with its cell: a filing,
/// a dispatch that names another target, a completion, a pair's two
/// sides, a release, the dispatch deadline, a new leader's restore and a
/// gone gateway.
#[test]
fn every_door_of_the_leases_steps_a_hosts_count_with_its_cell() {
    let (store, [a, b, c], both) = fleet();
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);

    // A single's lease, sent to its pick's second target, then ended.
    let mut targets = Targets::new();
    for id in [a, b] {
        targets.push(Target {
            id,
            ..Target::default()
        });
    }
    let mut single = booking(gateway, 1, a, now);
    single.picked = PickRecord::of(&targets);
    assert_eq!(leases.issue(single), Ok(()));
    assert_eq!(both(&store, "a filing"), [1, 0, 0]);
    assert!(leases.dispatched(gateway, LeaseId(1), on(Side::Single, 1), now));
    assert_eq!(both(&store, "a dispatch on another target"), [0, 1, 0]);
    assert!(leases.completed(gateway, LeaseId(1), &done(Side::Single)));
    assert_eq!(both(&store, "a completion"), [0, 0, 0]);

    // A pair: each side on its own worker, the decode sent to an alternate.
    issue_pair(leases, pair(2, a, &[b, c], now)).expect("the key is free");
    assert_eq!(both(&store, "a pair's filing"), [1, 1, 0]);
    assert!(leases.dispatched(gateway, LeaseId(2), on(Side::Decode, 1), now));
    assert_eq!(
        both(&store, "a decode's dispatch on an alternate"),
        [1, 0, 1]
    );
    let gone = ReleaseReason::ClientGone;
    assert!(leases.released(gateway, LeaseId(2), None, gone));
    assert_eq!(both(&store, "a release of every side"), [0, 0, 0]);

    // A lease no dispatch names by its deadline.
    assert_eq!(leases.issue(booking(gateway, 3, c, now)), Ok(()));
    assert_eq!(both(&store, "a second filing"), [0, 0, 1]);
    assert_eq!(leases.expire(now + Duration::from_secs(9)).len(), 1);
    assert_eq!(both(&store, "the dispatch deadline"), [0, 0, 0]);

    // A new leader's file: what a gateway's pages list, then the gateway gone.
    let pages = [booking(gateway, 4, b, now), booking(gateway, 5, b, now)];
    assert_eq!(leases.restore(gateway, &pages), 2);
    assert_eq!(both(&store, "a restore"), [0, 2, 0]);
    assert_eq!(leases.drop_gateways(&[gateway]), [2]);
    assert_eq!(both(&store, "a gone gateway"), [0, 0, 0]);
}

/// A lease filed ahead of its worker's count is on the first count handed,
/// and ends on it.
#[test]
fn a_lease_filed_ahead_of_its_workers_count_is_on_the_first_count_handed() {
    let store = MemoryStore::new();
    let worker = ready(&store, "http://a:1");
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    for id in 1..=2 {
        assert_eq!(leases.issue(booking(gateway, id, worker, now)), Ok(()));
    }
    assert!(leases.completed(gateway, LeaseId(1), &done(Side::Single)));
    assert_eq!(booked(&store, &[worker]), [1]);

    let kept = handed(&store, worker);
    assert_eq!(kept.held(), 1, "the lease that stands");
    assert!(leases.completed(gateway, LeaseId(2), &done(Side::Single)));
    assert_eq!((kept.held(), booked(&store, &[worker])), (0, vec![0]));
}

/// A count handed in the place of one that steps the same number leaves
/// the number as it stood: it takes the leases that stand as the one
/// before it gives them up. Their ends are its own from then on.
#[test]
fn a_count_over_the_number_of_the_one_before_it_leaves_the_number_as_it_stood() {
    let store = MemoryStore::new();
    let worker = ready(&store, "http://a:1");
    let kept = handed(&store, worker);
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    assert_eq!(leases.issue(booking(gateway, 1, worker, now)), Ok(()));
    assert_eq!(kept.held(), 1);

    // Another count over what the first one steps.
    hand(&store, worker, &kept);
    assert_eq!(kept.dropped(), 1, "the count before it is dropped");
    assert_eq!(kept.held(), 1, "the lease stands on the number");
    assert!(leases.completed(gateway, LeaseId(1), &done(Side::Single)));
    assert_eq!(kept.held(), 0, "the end is the new count's");
}

/// A count of a number of its own, handed in another's place, takes the
/// leases that stand, and the count before it gives them up.
#[test]
fn a_count_of_its_own_number_takes_the_leases_of_the_one_before_it() {
    let store = MemoryStore::new();
    let worker = ready(&store, "http://a:1");
    let first = handed(&store, worker);
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    for id in 1..=2 {
        assert_eq!(leases.issue(booking(gateway, id, worker, now)), Ok(()));
    }
    assert_eq!(first.held(), 2);

    let second = handed(&store, worker);
    assert_eq!((first.held(), first.dropped()), (0, 1), "given up");
    assert_eq!(second.held(), 2, "the leases that stand");
    assert!(leases.completed(gateway, LeaseId(1), &done(Side::Single)));
    assert_eq!((first.held(), second.held()), (0, 1));
}

/// A row that leaves takes the leases that book it off its count, and the
/// store lets the count go: their ends step no host's count.
#[test]
fn a_row_that_leaves_takes_its_leases_off_its_count() {
    let store = MemoryStore::new();
    let [idle, busy] = ["http://a:1", "http://b:1"].map(|key| ready(&store, key));
    let (kept_idle, kept_busy) = (handed(&store, idle), handed(&store, busy));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    for id in 1..=2 {
        assert_eq!(leases.issue(booking(gateway, id, busy, now)), Ok(()));
    }
    assert_eq!((kept_idle.held(), kept_busy.held()), (0, 2));

    remove(&store, idle);
    remove(&store, busy);
    assert_eq!((kept_idle.held(), kept_idle.dropped()), (0, 1));
    assert_eq!(kept_busy.held(), 0, "the two leases are given up");
    assert_eq!(kept_busy.dropped(), 1, "and the count is let go");
    for id in 1..=2 {
        assert!(leases.completed(gateway, LeaseId(id), &done(Side::Single)));
    }
    assert_eq!(kept_busy.held(), 0, "an end behind the row steps nothing");
}

/// A worker's new life takes a count of its own: the count of the life
/// before it gives up its leases, and the one handed for the new row
/// holds the new row's leases alone.
#[test]
fn a_workers_new_life_takes_a_count_of_its_own() {
    let store = MemoryStore::new();
    let old = ready(&store, "http://a:1");
    let kept_old = handed(&store, old);
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    for id in 1..=2 {
        assert_eq!(leases.issue(booking(gateway, id, old, now)), Ok(()));
    }

    let new = store.upsert(facts("http://a:1", 2)).id();
    store.publish_now();
    assert_ne!(old, new);
    assert_eq!((kept_old.held(), kept_old.dropped()), (0, 1));
    assert_eq!(leases.issue(booking(gateway, 3, new, now)), Ok(()));
    let kept_new = handed(&store, new);
    assert_eq!(kept_new.held(), 1, "the new row's lease");
    assert!(leases.completed(gateway, LeaseId(1), &done(Side::Single)));
    assert_eq!(kept_new.held(), 1, "an end of the life before it");
    assert!(leases.completed(gateway, LeaseId(3), &done(Side::Single)));
    assert_eq!(kept_new.held(), 0);
}

/// A snapshot takes the leases off the count of a row it drops, and of a
/// row it lays at a new place: a watch keeps the place of a row it sees
/// again, and a restore gives every row a new one.
#[test]
fn a_snapshot_takes_the_leases_off_the_count_of_a_row_it_moves() {
    let store = MemoryStore::new();
    let [left, stays] = ["http://a:1", "http://b:1"].map(|key| ready(&store, key));
    let (kept_left, kept_stays) = (handed(&store, left), handed(&store, stays));
    let leases: &dyn LeasesStore = &store;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    for (id, target) in [(1, left), (2, stays)] {
        assert_eq!(leases.issue(booking(gateway, id, target, now)), Ok(()));
    }

    let source = MemoryStore::new();
    assert_eq!(ready(&source, "http://b:1"), stays);
    assert!(store.watch_from(source.checkpoint(JournalKinds::Watched)));
    assert_eq!((kept_left.held(), kept_left.dropped()), (0, 1), "dropped");
    assert_eq!((kept_stays.held(), kept_stays.dropped()), (1, 0), "kept");

    assert!(store.restore_from(source.checkpoint(JournalKinds::All)));
    assert_eq!((kept_stays.held(), kept_stays.dropped()), (0, 1));
}

/// A follower's store takes its leader's removal and its leader's new
/// incarnation as its own writes take theirs: the count of the row that
/// left gives up its leases and goes.
#[test]
fn a_followed_leaving_takes_the_leases_off_the_count() {
    let (leader, follower) = (MemoryStore::new(), MemoryStore::new());
    let join = |facts: WorkerFacts| {
        let change = leader.upsert(facts);
        followed(&leader, &follower, change);
        change.id()
    };
    let (removed, replaced) = (join(facts("http://a:1", 1)), join(facts("http://b:1", 1)));
    follower.publish_now();
    let kept_removed = handed(&follower, removed);
    let kept_replaced = handed(&follower, replaced);
    let leases: &dyn LeasesStore = &follower;
    let (gateway, now) = (GatewayId(1), Instant::now());
    leases.set_term(1);
    for (id, target) in [(1, removed), (2, replaced)] {
        assert_eq!(leases.issue(booking(gateway, id, target, now)), Ok(()));
    }
    assert_eq!((kept_removed.held(), kept_replaced.held()), (1, 1));

    let revision = WorkersStore::get(&leader, removed)
        .expect("the row")
        .revision;
    let change = leader.remove(removed, revision).expect("removed");
    followed(&leader, &follower, change);
    assert_eq!((kept_removed.held(), kept_removed.dropped()), (0, 1));
    assert_ne!(join(facts("http://b:1", 2)), replaced);
    assert_eq!((kept_replaced.held(), kept_replaced.dropped()), (0, 1));
}

/// A row that is gone takes no count: one no write made, and one that
/// left.
#[test]
fn a_row_that_is_gone_takes_no_count() {
    let store = MemoryStore::new();
    let kept = Arc::new(Kept::default());
    let count = Box::new(Count(Arc::clone(&kept)));
    assert!(!store.host_count(WorkerId(7), count));
    assert_eq!(kept.dropped(), 1);

    let worker = ready(&store, "http://a:1");
    remove(&store, worker);
    let count = Box::new(Count(Arc::clone(&kept)));
    assert!(!store.host_count(worker, count), "a row that left");
    assert_eq!((kept.held(), kept.dropped()), (0, 2));
}

/// The gateways that file at once in a case, and the leases each files.
const WRITERS: u32 = 4;
const LEASES: u64 = 2_000;

/// Says a writer ended, however it ended.
struct Ended<'a>(&'a AtomicU32);

impl Drop for Ended<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, SeqCst);
    }
}

/// Four gateways file leases on `worker` and end every second one, each
/// on a thread of its own. `among` runs once leases stand, or once every
/// writer ended: the hand-over or the leaving a case sets among them.
fn among_filings(store: &MemoryStore, worker: WorkerId, among: impl FnOnce()) {
    let now = Instant::now();
    let (filed, ended) = (AtomicU32::new(0), AtomicU32::new(0));
    thread::scope(|scope| {
        for writer in 0..WRITERS {
            let (filed, ended) = (&filed, &ended);
            scope.spawn(move || {
                let _ended = Ended(ended);
                let gateway = GatewayId(u64::from(writer) + 1);
                for id in 1..=LEASES {
                    let lease = store.issue(booking(gateway, id, worker, now));
                    assert_eq!(lease, Ok(()));
                    // Every second lease ends, and the others stand.
                    if id % 2 == 0 {
                        assert!(store.completed(gateway, LeaseId(id), &done(Side::Single)));
                    }
                    filed.fetch_add(1, SeqCst);
                }
            });
        }
        while filed.load(SeqCst) < 256 && ended.load(SeqCst) < WRITERS {
            thread::yield_now();
        }
        among();
    });
}

/// Leases filed and ended on several threads while the worker's first
/// count is handed: no step is lost and none is taken twice.
#[test]
fn a_count_handed_among_filings_holds_the_leases_that_stand() {
    for _ in 0..8 {
        let store = MemoryStore::new();
        let worker = ready(&store, "http://a:1");
        LeasesStore::set_term(&store, 1);
        let kept = Arc::new(Kept::default());
        among_filings(&store, worker, || hand(&store, worker, &kept));
        let stand = u32::try_from(u64::from(WRITERS) * LEASES / 2).expect("a small count");
        assert_eq!(kept.held(), i64::from(stand), "the leases that stand");
        assert_eq!(booked(&store, &[worker]), [stand], "as the worker's cell");
    }
}

/// A row that leaves while several threads file and end leases on its
/// worker: its count gives up every lease it took, and none steps it
/// behind the leaving.
#[test]
fn a_count_that_leaves_among_filings_holds_no_lease() {
    for _ in 0..8 {
        let store = MemoryStore::new();
        let worker = ready(&store, "http://a:1");
        let kept = handed(&store, worker);
        LeasesStore::set_term(&store, 1);
        among_filings(&store, worker, || remove(&store, worker));
        assert_eq!(kept.held(), 0, "every lease it took is given up");
        assert_eq!(kept.dropped(), 1);
    }
}
