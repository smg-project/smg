//! A restore and a watch's snapshot: what a restore leaves of the lease file
//! and tells of its ends, what a watch's snapshot keeps, evicts and tells the
//! members, and a stage that panics at the store's word of membership.

use super::*;

/// A replica that restores holds no lease. What it filed in an earlier
/// lead is another history's: the file is empty behind the restore, its
/// deadline queue with it, and every count reads zero. Each lease's end
/// is told once, as a frame's release of a removed worker's lease is.
/// When the replica leads again the gateway's `Hello` files what it
/// lists, and each count reads that list once.
#[test]
fn a_restore_empties_the_lease_file_and_a_hello_books_what_it_lists_once() {
    let store = MemoryStore::new();
    store.tell_through(heard);
    let (a, b) = (ready(&store, "http://a:1"), ready(&store, "http://b:1"));
    let (gateway, now) = (GatewayId(1), Instant::now());
    store.set_term(1);
    for (id, target) in [(1, a), (2, a), (3, b)] {
        LeasesStore::issue(&store, booking(gateway, id, target, now)).expect("filed");
    }
    let of = |a_holds, b_holds| {
        let mut rows = [(a, a_holds), (b, b_holds)];
        rows.sort_unstable();
        rows
    };
    assert_eq!(in_flight(&store), of(2, 1));
    assert_eq!(store.queued(), 3);
    // Another replica led over the same rows, and this one restores from
    // its cut.
    let leader = MemoryStore::new();
    let theirs = (ready(&leader, "http://a:1"), ready(&leader, "http://b:1"));
    assert_eq!(theirs, (a, b));
    drop(told());
    assert!(store.restore_from(leader.checkpoint()));
    let leases: &dyn LeasesStore = &store;
    assert_eq!(leases.live(gateway), 0, "a lease stood behind the restore");
    assert!(leases.get(gateway, LeaseId(1)).is_none());
    assert_eq!(store.queued(), 0);
    assert_eq!(in_flight(&store), of(0, 0));
    // The three ends, each told once with the record the file held.
    let removed = Said::Released(None, ReleaseReason::WorkerRemoved);
    let mut ended: Vec<u64> = told()
        .into_iter()
        .map(|(record, said)| {
            assert_eq!(said, removed);
            assert_eq!(record.state, LeaseState::ISSUED);
            record.lease.id.0
        })
        .collect();
    ended.sort_unstable();
    assert_eq!(ended, [1, 2, 3]);
    // The file is empty: a second restore tells no end.
    assert!(store.restore_from(leader.checkpoint()));
    assert!(told().is_empty(), "a lease was told twice");
    // The lead taken again: the gateway lists the two leases it still
    // holds, one a worker.
    store.set_term(3);
    let listed = [booking(gateway, 1, a, now), booking(gateway, 3, b, now)];
    assert_eq!(leases.restore(gateway, &listed), 2);
    assert_eq!(in_flight(&store), of(1, 1));
    // An end the gateway reports then gives its count back.
    assert!(leases.released(gateway, LeaseId(1), None, ReleaseReason::ClientGone));
    assert_eq!(in_flight(&store), of(0, 1));
}

/// A restore tells each lease's end under the decider whose file holds
/// the lease, as a frame's end is told, whichever thread restores.
#[test]
fn a_restore_tells_each_end_under_the_decider_whose_file_holds_its_lease() {
    let store = MemoryStore::new().with_decision_slots(4);
    let a = ready(&store, "http://a:1");
    LeasesStore::set_term(&store, 1);
    store.tell_through(heard);
    let (now, world) = (Instant::now(), store.snapshot());
    let issue = || {
        let record = booking(GatewayId(1), 0, a, now);
        let issued = store.issue_decided(record, world.booked_of(a), |_| ());
        issued.expect("minted under the store's term").id
    };
    // One lease issued by a thread that acts as the fourth decider, one by
    // a thread that names none: the first file's.
    let named = {
        let _acting = DecisionSlot::new(3).enter();
        issue()
    };
    let unnamed = issue();
    // Another replica led over the same row, and this one restores from
    // its cut on a thread that names none.
    let leader = MemoryStore::new();
    assert_eq!(ready(&leader, "http://a:1"), a);
    assert!(store.restore_from(leader.checkpoint()));
    let mut ended: Vec<(u64, Option<DecisionSlot>)> = told()
        .into_iter()
        .map(|(record, _)| record.lease.id.0)
        .zip(told_by())
        .collect();
    ended.sort_unstable();
    let first = Some(DecisionSlot::new(0));
    let mut issued = [(named.0, Some(DecisionSlot::new(3))), (unnamed.0, first)];
    issued.sort_unstable();
    assert_eq!(ended, issued);
}

#[test]
fn a_watch_snapshot_keeps_a_members_local_tables() {
    let store = MemoryStore::new();
    let id = ready(&store, "http://a:1");
    store.publish_now();
    let (gateway, now) = (GatewayId(1), Instant::now());
    let lease = booking(gateway, 1, id, now);
    let leases: &dyn LeasesStore = &store;
    leases.set_term(1);
    leases.issue(lease).expect("filed");
    let slot = store.snapshot().booked_of(id).expect("the member's slot");
    store.gateway_veto(gateway, id, Some(now + Duration::from_secs(60)));

    let mut cut = store.checkpoint();
    cut.epoch += 1;
    cut.at.term = 7;
    cut.rows[0].admitted_at += Duration::from_secs(1);
    cut.rows[0].revision += 1;
    cut.rows[0].facts.priority = 8;
    // The row is the cut's, its later instant too: the store keeps none
    // of its own.
    let expected = cut.rows[0].clone();
    assert!(store.watch_from(cut));

    assert_eq!(leases.live(gateway), 1);
    assert_eq!(leases.get(gateway, lease.lease.id), Some(lease));
    assert_eq!(leases.term(), 1);
    assert_eq!(store.snapshot().booked_of(id), Some(slot));
    assert_eq!(in_flight(&store), [(id, 1)]);
    assert!(store.vetoed(gateway, id, now));
    assert_eq!(store.get(id), Some(expected));
}

#[test]
fn a_watch_snapshot_evicts_a_leavers_tables_and_its_lease_moves_no_other_count() {
    let store = MemoryStore::new();
    let left = store.upsert(facts("http://a:1", 1)).id();
    store.publish_now();
    let before = store.snapshot();
    let slot = before.booked_of(left).expect("the leaver's slot");
    let (gateway, now) = (GatewayId(1), Instant::now());
    let lease = booking(gateway, 1, left, now);
    let leases: &dyn LeasesStore = &store;
    leases.set_term(1);
    leases.issue(lease).expect("filed");
    store.gateway_veto(gateway, left, Some(now + Duration::from_secs(60)));

    let source = MemoryStore::new();
    let joined = ready(&source, "http://b:1");
    assert!(store.watch_from(source.checkpoint()));
    assert_eq!(leases.get(gateway, lease.lease.id), Some(lease));
    assert_ne!(store.snapshot().booked_of(joined), Some(slot));
    assert_eq!(in_flight(&store), [(joined, 0)]);
    assert!(store.by_key("http://a:1").is_none());
    assert!(!store.vetoed(gateway, left, now));
    assert!(leases.released(gateway, lease.lease.id, None, ReleaseReason::ClientGone));
    assert_eq!(in_flight(&store), [(joined, 0)]);
}

#[test]
fn a_watch_snapshot_gives_a_new_incarnation_its_own_count() {
    let store = MemoryStore::new();
    let key = "http://a:1";
    let old = ready(&store, key);
    let (gateway, now) = (GatewayId(1), Instant::now());
    let lease = booking(gateway, 1, old, now);
    let leases: &dyn LeasesStore = &store;
    leases.set_term(1);
    leases.issue(lease).expect("filed");

    let source = MemoryStore::new();
    assert_eq!(ready(&source, key), old);
    let new = source.upsert(facts(key, 2)).id();
    assert!(store.watch_from(source.checkpoint()));
    assert_eq!(store.by_key(key).map(|row| row.id), Some(new));
    assert!(store.get(old).is_none());
    assert_eq!(in_flight(&store), [(new, 0)]);
    assert!(leases.released(gateway, lease.lease.id, None, ReleaseReason::ClientGone));
    assert_eq!(in_flight(&store), [(new, 0)]);
}

#[test]
fn a_watch_snapshot_publishes_the_cut_at_once_and_tells_the_members_once() {
    use crate::{
        document::{AppliedSpec, PolicySpec},
        engine::{
            view::{Candidate, RequestFacts},
            Scorer, TieBand,
        },
        stages::{Lifecycle, StageRegistry},
    };

    #[derive(Debug)]
    struct Heard(Arc<Mutex<Vec<String>>>);

    impl Lifecycle for Heard {
        fn on_membership(&self, change: &Membership<'_>) {
            self.0
                .lock()
                .expect("the stage heard")
                .push(format!("{change:?}"));
        }
    }

    impl Scorer for Heard {
        fn name(&self) -> &'static str {
            "heard"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(0.0);
            TieBand::NORMALIZED
        }
    }

    let store = MemoryStore::new();
    let id = ready(&store, "http://a:1");
    let heard = Arc::new(Mutex::new(Vec::new()));
    let mut catalog = StageRegistry::builtin();
    let events = Arc::clone(&heard);
    catalog
        .scorer("heard", move || Box::new(Heard(Arc::clone(&events))))
        .expect("a free name");
    let document: PolicySpec = serde_json::from_str(
        r#"{"apiVersion":"smg.policy/v1","generation":1,"view":{"labels":["zone=b"]},"profiles":{"p":{"score":["heard"],"pick":"best"}}}"#,
    ).expect("a policy document");
    let digest = document.digest().expect("the digest");
    let spec =
        AppliedSpec::validate_digested(&document, digest, &catalog, &[]).expect("a generation");
    store.apply(spec).expect("the generation is applied");

    let mut cut = store.checkpoint();
    // The same revision can name other facts in another history.
    cut.rows[0]
        .facts
        .labels
        .insert("zone".to_owned(), "b".to_owned());
    cut.rows[0].status = WorkerStatus::Draining;
    cut.epoch += 2;
    let epoch = cut.epoch;
    store.upsert(facts("http://owed:1", 1));
    assert!(store.publish_due().is_some(), "a build is owed");
    heard.lock().expect("the stage heard").clear();
    let before = store.snapshot();
    assert!(store.watch_from(cut));
    assert!(
        !Arc::ptr_eq(&store.snapshot(), &before),
        "a world of the cut stands at once"
    );
    assert_eq!(store.epoch(), epoch);
    assert_eq!(store.publish_due(), None);
    assert_eq!(store.filed_members(), 0);
    let (_, rows) = store.all_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, id);
    let zone = rows[0].facts.labels.get("zone").map(String::as_str);
    assert_eq!(zone, Some("b"), "the cut's labels");
    assert_eq!(rows[0].status, WorkerStatus::Draining, "the cut's status");
    let snapshot = Membership::Snapshot(&[id]);
    assert_eq!(
        *heard.lock().expect("the stage heard"),
        [format!("{snapshot:?}")]
    );
}

/// A stage that panics at a store's word of membership is held and counted
/// by the engine that decides under the standing document, as at a
/// gateway's word: the store's write goes on with what it still owes. An
/// apply stands and tells the journal its generation, a registration leaves
/// its row on file and a removal takes it off. The stage behind the one
/// that panicked hears none of the words. Before, the panic unwound out of
/// the write: an apply left with its world shown and its journal untold.
#[cfg(feature = "in-process")]
#[test]
fn a_stage_that_panics_at_the_stores_word_of_membership_is_held_and_counted() {
    use crate::{
        document::{AppliedSpec, PolicySpec},
        engine::{
            view::{Candidate, RequestFacts},
            Scorer, TieBand,
        },
        stages::{Lifecycle, StageRegistry},
    };

    /// A scorer that scores every row alike. `Some` counts the words of
    /// membership it hears; `None` panics at each.
    #[derive(Debug)]
    struct Hears(&'static str, Option<Arc<AtomicU32>>);

    impl Lifecycle for Hears {
        fn on_membership(&self, _: &Membership<'_>) {
            let Some(heard) = &self.1 else {
                panic!("the case's own panic, at a word of membership");
            };
            heard.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl Scorer for Hears {
        fn name(&self) -> &'static str {
            self.0
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(0.0);
            TieBand::NORMALIZED
        }
    }

    let log = Arc::new(Log::new(8));
    let journal: Arc<Log> = Arc::clone(&log);
    let store = MemoryStore::journaled(journal);
    store.set_term(1);
    let behind = Arc::new(AtomicU32::new(0));
    let heard = Arc::clone(&behind);
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer("deaf", || Box::new(Hears("deaf", None)))
        .expect("a free name");
    catalog
        .scorer("behind", move || {
            Box::new(Hears("behind", Some(Arc::clone(&heard))))
        })
        .expect("a free name");
    let document: PolicySpec = serde_json::from_str(
        r#"{"apiVersion":"smg.policy/v1","generation":1,"profiles":{"p":{"score":["deaf","behind"],"pick":"best"}}}"#,
    )
    .expect("a policy document");
    let digest = document.digest().expect("the digest");
    let spec =
        AppliedSpec::validate_digested(&document, digest, &catalog, &[]).expect("a generation");
    let unheard = || store.deciding(|world| world.front().unheard());

    // An apply tells its engine the members that stand.
    let told = log.head().seq;
    store.apply(spec).expect("the generation is applied");
    assert_eq!(unheard(), 1);
    assert!(log.head().seq > told, "the journal is told the generation");

    // A registration and a removal each tell their word.
    let id = store.upsert(facts("http://a:1", 1)).id();
    assert_eq!(unheard(), 2);
    let row = store.get(id).expect("the row is on file");
    assert!(store.remove(id, row.revision).is_some());
    assert_eq!(unheard(), 3);
    assert!(store.get(id).is_none());
    assert_eq!(behind.load(Ordering::Relaxed), 0);
}

#[test]
fn a_watch_snapshot_the_journal_refuses_changes_nothing() {
    let store = MemoryStore::journaled(Arc::new(Log::new(8)));
    let id = ready(&store, "http://a:1");
    store.set_term(3);
    let before = store.checkpoint();
    let world = store.snapshot();
    let source = MemoryStore::new();
    ready(&source, "http://b:1");
    let mut cut = source.checkpoint();
    cut.at = Position { seq: 1, term: 2 };
    assert!(!store.watch_from(cut));
    let after = store.checkpoint();
    assert_eq!(
        (after.at, after.epoch, after.rows),
        (before.at, before.epoch, before.rows)
    );
    assert!(Arc::ptr_eq(&store.snapshot(), &world));
    assert_eq!(in_flight(&store), [(id, 0)]);
}

/// A journal that takes every place, and whose restart waits inside the
/// restore's locks: it says that it stands there, then waits for the
/// case's word.
struct WaitsAtRestart {
    reached: mpsc::Sender<()>,
    go: Mutex<mpsc::Receiver<()>>,
}

impl Journal for WaitsAtRestart {
    fn member(
        &self,
        _: Option<Position>,
        _: Option<u64>,
        _: u64,
        _: MembershipChange,
        _: Option<&WorkerRow>,
        _: Option<&WorkerRow>,
    ) -> bool {
        true
    }

    fn status(
        &self,
        _: Option<Position>,
        _: Option<u64>,
        _: u64,
        _: WorkerId,
        _: WorkerStatus,
        _: WorkerStatus,
        _: Arc<[String]>,
    ) -> bool {
        true
    }

    fn term_rose(&self, _: u64) {}

    fn generation(&self, _: Option<Position>, _: u64) {}

    fn term_followed(&self, _: Position) {}

    fn head(&self) -> Position {
        Position::default()
    }

    fn followed(&self) -> Option<u64> {
        None
    }

    fn restart(&self, _: Position) -> bool {
        // A case that is gone is waited for no longer.
        let _ = self.reached.send(());
        if let Ok(word) = self.go.lock() {
            let _ = word.recv();
        }
        true
    }
}

/// A take beside a restore stands whole behind it. The take's write waits
/// for its file's turn, and the restore holds every file's turn from its
/// start until it has read the ids on file: the first lease of the term
/// taken is filed behind that read, on the restored row's count, and the
/// restore ends no lease of that term.
#[test]
fn a_take_beside_a_restore_files_its_first_lease_behind_it() {
    let (reached, stands) = mpsc::channel();
    let (go, word) = mpsc::channel();
    let store = MemoryStore::journaled(Arc::new(WaitsAtRestart {
        reached,
        go: Mutex::new(word),
    }));
    let a = ready(&store, "http://a:1");
    let (gateway, now) = (GatewayId(1), Instant::now());
    // The lead before: one lease on file.
    store.set_term(1);
    LeasesStore::issue(&store, booking(gateway, 1, a, now)).expect("filed");
    // Another replica led over the same row, and this one restores from
    // its cut.
    let leader = MemoryStore::new();
    assert_eq!(ready(&leader, "http://a:1"), a);
    let cut = leader.checkpoint();
    let store = &store;
    let (inside, restored, filed) = std::thread::scope(move |scope| {
        let restore = scope.spawn(move || store.restore_from(cut));
        // The restore stands inside its locks, at its journal's restart.
        stands
            .recv_timeout(Duration::from_secs(10))
            .expect("the restore reaches its journal");
        // The replica takes the lease there: the take's write, then the
        // first lease of its term.
        let (landed, lands) = mpsc::channel();
        let take = scope.spawn(move || {
            store.set_term(2);
            let first = LeaseRecord {
                target: a,
                ..lease(gateway, 2, 2, now)
            };
            let filed = LeasesStore::issue(store, first);
            let _ = landed.send(());
            filed
        });
        // Nothing of the take lands while the restore stands there.
        let inside = lands.recv_timeout(Duration::from_millis(200)).is_ok();
        go.send(()).expect("the restore waits at its journal");
        (
            inside,
            restore.join().expect("the restore"),
            take.join().expect("the take"),
        )
    });
    assert!(!inside, "the take landed inside the restore");
    assert!(restored);
    filed.expect("the first lease of the term taken is filed");
    // The file holds that lease, on the restored row's count, and nothing
    // of the lead before.
    let leases: &dyn LeasesStore = store;
    assert!(leases.get(gateway, LeaseId(1)).is_none());
    assert!(leases.get(gateway, LeaseId(2)).is_some());
    assert_eq!(leases.term(), 2);
    assert_eq!(in_flight(store), [(a, 1)]);
}
