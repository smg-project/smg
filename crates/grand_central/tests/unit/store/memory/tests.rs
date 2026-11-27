use super::*;
use crate::{
    engine::{decision_slot::DecisionSlot, view::WorkerId},
    store::{
        leases::PickRecord,
        workers::{Registrant, WorkerPatch},
    },
    tracking::lease::Lease,
};

/// The cases' road to the table of records: the five write entries the
/// store's trait held, for a case that fills a store with no holder of
/// worker objects. Each takes the write's hold, decides by the rule
/// every writer decides by and hands the one body the change
/// ([`MemoryStore::decided`]), as a holder's write call does. They
/// leave with the table.
impl MemoryStore {
    /// The cases' road: a registration by a control call. It leaves with
    /// the table.
    pub(crate) fn upsert(&self, facts: WorkerFacts) -> MembershipChange {
        self.upsert_by(facts, Registrant::Control)
    }

    /// The cases' road: a registration or its refresh by `maker`, decided
    /// over the table's own row. It leaves with the table.
    pub(crate) fn upsert_by(&self, facts: WorkerFacts, maker: Registrant) -> MembershipChange {
        let workers = self.writing();
        let standing = workers.by_key.get(&facts.key);
        let standing = standing.and_then(|id| workers.rows.get(id));
        let standing = standing.map(|held| &held.row);
        let now = Instant::now();
        let (change, row) = WorkerRow::laid_over(facts, standing, maker, now);
        let fits = self.decided(workers, now, change, Some(row), None);
        assert!(fits, "a change decided under the membership lock fits");
        change
    }

    /// The cases' road: a patch of the row `id` where it stands at
    /// `revision`. It leaves with the table.
    pub(crate) fn patch(
        &self,
        id: WorkerId,
        patch: WorkerPatch,
        revision: u64,
    ) -> Option<MembershipChange> {
        let mut workers = self.writing();
        let mut row = workers.guarded(id, revision)?.row.clone();
        patch.laid_on(&mut row.facts);
        row.revision += 1;
        let change = MembershipChange::Relabelled { id };
        self.decided(workers, Instant::now(), change, Some(row), None)
            .then_some(change)
    }

    /// The cases' road: the removal of the row `id` where it stands at
    /// `revision`. It leaves with the table.
    pub(crate) fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
        let mut workers = self.writing();
        workers.guarded(id, revision)?;
        let change = MembershipChange::Removed { id };
        self.decided(workers, Instant::now(), change, None, None)
            .then_some(change)
    }

    /// The cases' road: a status of the row `id` where it stands at
    /// `revision`, as a holder tells one. It leaves with the table.
    pub(crate) fn set_status(
        &self,
        id: WorkerId,
        status: WorkerStatus,
        revision: u64,
    ) -> Option<MembershipChange> {
        let mut workers = self.writing();
        let from = workers.guarded(id, revision)?.row.status;
        // This road hands no models: the journal is told the table's
        // row's.
        match workers.set_status(id, status, None) {
            StatusSet::Moved => Some(MembershipChange::StatusChanged {
                id,
                from,
                to: status,
            }),
            StatusSet::Stood | StatusSet::Refused => None,
        }
    }
}

#[test]
fn the_lease_sweep_takes_only_a_file_with_a_due_lease() {
    let store = MemoryStore::new().with_decision_slots(256);
    let clear_takes = || store.files.iter().for_each(turn::File::clear_takes);
    let takes = || store.files.iter().map(turn::File::takes).sum::<usize>();
    let now = Instant::now();

    clear_takes();
    assert!(LeasesStore::expire(&store, now).is_empty());
    assert_eq!(takes(), 0, "empty files need no turn");

    LeasesStore::set_term(&store, 1);
    let record = LeaseRecord {
        lease: Lease::default(),
        gateway: GatewayId(1),
        request: 1,
        target: WorkerId(1),
        picked: PickRecord::default(),
        decode: None,
        untold: false,
        term: 1,
        issued_at: now,
        dispatch_deadline: now + Duration::from_secs(1),
        state: LeaseState::ISSUED,
    };
    let _acting = DecisionSlot::new(173).enter();
    let issued = store
        .issue_decided(record, None, |_| {})
        .expect("the lease is filed");
    clear_takes();
    let ended = LeasesStore::expire(&store, record.dispatch_deadline);
    assert_eq!(ended.len(), 1);
    assert_eq!(ended[0].record.lease.id, issued.id);
    assert_eq!(takes(), 1, "only the due lease's file is taken");
}

/// A lease issued at `now` that waits for its dispatch until `deadline`.
fn waiting(now: Instant, request: u64, deadline: Instant) -> LeaseRecord {
    LeaseRecord {
        lease: Lease::default(),
        gateway: GatewayId(1),
        request,
        target: WorkerId(1),
        picked: PickRecord::default(),
        decode: None,
        untold: false,
        term: 1,
        issued_at: now,
        dispatch_deadline: deadline,
        state: LeaseState::ISSUED,
    }
}

/// An issue reads no clock for the hint. A new first place asks the
/// sweep once, the sweep reads its deadline, and no sweep takes the
/// file again before it. A place behind the first asks none. The file
/// the sweep emptied is passed.
#[test]
fn a_file_is_asked_once_and_then_at_the_deadline_the_sweep_read() {
    let store = MemoryStore::new().with_decision_slots(256);
    let clear_takes = || store.files.iter().for_each(turn::File::clear_takes);
    let takes = || store.files.iter().map(turn::File::takes).sum::<usize>();
    let now = Instant::now();
    let after = |millis: u64| now + Duration::from_millis(millis);
    LeasesStore::set_term(&store, 1);
    let _acting = DecisionSlot::new(173).enter();
    let file = |request: u64, millis: u64| {
        let filed = store.issue_decided(waiting(now, request, after(millis)), None, |_| {});
        filed.expect("the lease is filed")
    };
    file(1, 1_000);
    clear_takes();
    assert!(LeasesStore::expire(&store, now).is_empty());
    assert_eq!(takes(), 1, "a new first place asks the sweep once");
    assert!(LeasesStore::expire(&store, after(500)).is_empty());
    assert_eq!(takes(), 1, "the sweep waits for the deadline it read");
    file(2, 1_500);
    clear_takes();
    assert!(LeasesStore::expire(&store, after(500)).is_empty());
    assert_eq!(takes(), 0, "a place behind the first asks no sweep");
    // A lease filed with an earlier deadline of its own is a new
    // first place: the sweep is asked ahead of the instant it read.
    let earlier = file(3, 600);
    clear_takes();
    let ended = LeasesStore::expire(&store, after(700));
    assert_eq!((ended.len(), takes()), (1, 1));
    assert_eq!(ended[0].record.lease.id, earlier.id);
    let ended = LeasesStore::expire(&store, after(1_000));
    assert_eq!((ended.len(), takes()), (1, 2));
    let ended = LeasesStore::expire(&store, after(1_500));
    assert_eq!((ended.len(), takes()), (1, 3));
    assert!(LeasesStore::expire(&store, after(2_000)).is_empty());
    assert_eq!(takes(), 3, "the emptied file needs no turn");
}

/// The sweep shows a file's deadline ahead of its questions. A file
/// whose host has not settled is asked once before that instant, and
/// from it at every pass until the host settles.
#[test]
fn a_file_its_host_has_not_settled_is_asked_once_before_its_deadline() {
    let store = MemoryStore::new();
    let clear_takes = || store.files.iter().for_each(turn::File::clear_takes);
    let takes = || store.files.iter().map(turn::File::takes).sum::<usize>();
    let now = Instant::now();
    let after = |millis: u64| now + Duration::from_millis(millis);
    LeasesStore::set_term(&store, 1);
    let filed = store.issue_decided(waiting(now, 1, after(1_000)), None, |_| {});
    filed.expect("the lease is filed");
    clear_takes();
    for millis in [100, 200, 300] {
        assert!(store.expire_where(after(millis), |_| false).is_empty());
    }
    assert_eq!(takes(), 1, "asked once before its deadline");
    for millis in [1_000, 1_100] {
        assert!(store.expire_where(after(millis), |_| false).is_empty());
    }
    assert_eq!(takes(), 3, "asked at every pass from its deadline");
    assert_eq!(store.expire_where(after(1_200), |_| true).len(), 1);
}

/// The sweep passes a file whose decider works by its first read, with
/// no turn held: it takes no turn of a thread at work. It ends the
/// file's due lease once the decider has left.
#[test]
fn the_sweep_takes_no_turn_of_a_file_whose_decider_works() {
    let store = Arc::new(MemoryStore::new());
    let now = Instant::now();
    let due = now + Duration::from_secs(1);
    LeasesStore::set_term(&*store, 1);
    let filed = store.issue_decided(waiting(now, 1, due), None, |_| {});
    filed.expect("the lease is filed");
    let place = store.at_work();
    store.files[0].clear_takes();
    let swept = std::thread::scope(|scope| {
        let sweep = scope.spawn(|| LeasesStore::expire(&*store, due).len());
        sweep.join().expect("the sweep returns")
    });
    assert_eq!(swept, 0, "the file is its decider's while it works");
    assert_eq!(store.files[0].takes(), 0, "no turn of a thread at work");
    drop(place);
    assert_eq!(LeasesStore::expire(&*store, due).len(), 1);
}

/// Waits until `file`'s turn was taken `count` times since the case
/// cleared its takes. A thread that takes the turn counts first and
/// waits for it then.
fn taken(file: &turn::File, count: usize) {
    let end = Instant::now() + Duration::from_secs(30);
    while file.takes() < count {
        assert!(Instant::now() < end, "take {count} of the turn never came");
        std::thread::yield_now();
    }
}

/// A decider that begins work behind the sweep's first read is read
/// again under the turn, and the sweep passes its file. The case holds
/// the turn while the sweep waits for it and the decider writes its
/// mark.
#[test]
fn the_sweep_passes_a_file_whose_decider_began_behind_its_first_read() {
    let store = Arc::new(MemoryStore::new());
    let now = Instant::now();
    let due = now + Duration::from_secs(1);
    LeasesStore::set_term(&*store, 1);
    let filed = store.issue_decided(waiting(now, 1, due), None, |_| {});
    filed.expect("the lease is filed");
    let file = &store.files[0];
    let held = file.take();
    file.clear_takes();
    let (leave, left) = mpsc::sync_channel::<()>(1);
    let swept = std::thread::scope(|scope| {
        // The case's own end goes with it where it fails: the
        // decider's wait ends, and the case with it.
        let (leave, store) = (leave, &store);
        // The sweep reads nobody at work and waits for the turn.
        let sweep = scope.spawn(move || LeasesStore::expire(&**store, due).len());
        taken(file, 1);
        // The decider writes its mark, then waits for the turn too.
        let decider = scope.spawn(move || {
            let place = store.at_work();
            left.recv().expect("the case says when to leave");
            drop(place);
        });
        taken(file, 2);
        drop(held);
        let swept = sweep.join().expect("the sweep returns");
        leave.send(()).expect("the decider waits for the word");
        decider.join().expect("the decider leaves");
        swept
    });
    assert_eq!(swept, 0, "the file is its decider's from its mark");
    assert_eq!(LeasesStore::expire(&*store, due).len(), 1);
}

/// Indexes that hold no index and name one block size: what a case
/// reads an engine's hold of them by.
#[cfg(feature = "controller")]
struct OneSize(u32);

#[cfg(feature = "controller")]
impl CacheIndexes for OneSize {
    fn index(&self, _: &str) -> Option<Arc<dyn crate::engine::CacheIndex>> {
        None
    }

    fn block_size(&self, _: &str) -> Option<u32> {
        Some(self.0)
    }
}

/// The preset `round_robin` at `generation`.
#[cfg(feature = "controller")]
fn round_robin(generation: u64) -> AppliedSpec {
    use crate::document::spec::PolicySpec;

    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = generation;
    AppliedSpec::validate(&document).expect("the document validates")
}

/// The indexes a store is lent reach its engine in that call, and the
/// engine holds them under each policy spec applied later: the store
/// keeps one engine for its life. The first lending stands: a second
/// is ignored, under the policy spec that stands and under the next.
#[cfg(feature = "controller")]
#[test]
fn the_stores_engine_holds_the_lent_indexes_across_applies() {
    let store = MemoryStore::new();
    let size = || store.snapshot().front().affinity().block_size("m");
    store.apply(round_robin(1)).expect("generation 1");
    assert_eq!(size(), None, "lent none yet");
    store.lend_cache_indexes(Arc::new(OneSize(16)));
    assert_eq!(size(), Some(16), "the engine is handed them");
    store.apply(round_robin(2)).expect("generation 2");
    assert_eq!(size(), Some(16), "it holds them under a later policy spec");
    store.lend_cache_indexes(Arc::new(OneSize(32)));
    assert_eq!(size(), Some(16), "the first lending stands");
    store.apply(round_robin(3)).expect("generation 3");
    assert_eq!(size(), Some(16), "under the next policy spec too");
}

/// The side an issue replaced is answered under its place and its
/// lease's key, read and left kept, while the side that the issue laid
/// is issued or dispatched. A place keeps what an issue replaced past
/// its lease's end. So a lease laid in the place later reads none of
/// it: not under another key, and not under the same key until a call
/// issues its side. An issued side that ended answers none.
#[cfg(feature = "controller")]
#[test]
fn the_side_an_issue_replaced_is_read_under_its_place_and_key_while_its_issue_stands() {
    use crate::engine::SessionVerdict;

    let key = |id: u64| (GatewayId(1), LeaseId(id));
    let side = |target: u64, state: LeaseState| DecodeSide {
        target: WorkerId(target),
        picked: PickRecord::default(),
        session: Session::NONE,
        session_verdict: SessionVerdict::None,
        state,
        tokens: 0,
        target_index: 0,
    };
    // A side on worker 8 as it stands in `state`, issued by `issues` calls.
    let laid = |issues: u8, state: LeaseState| FiledDecode {
        issues,
        ..FiledDecode::laid(side(8, state), None)
    };
    let first = side(7, LeaseState::Chosen);
    let mut decodes = Decodes::default();
    assert!(!decodes.keeps_any(), "no issue kept a side yet");
    // A lease ahead, so the pair's place is not the list's first.
    decodes.lay(key(1), laid(0, LeaseState::ISSUED));
    let at = decodes.lay(key(2), laid(1, LeaseState::ISSUED));
    assert_eq!(decodes.replaced_side(at, &key(2)), None, "nothing kept yet");

    decodes.keep(at, key(2), (first, Instant::now()));
    assert!(decodes.keeps_any());
    for _ in 0..2 {
        assert_eq!(decodes.replaced_side(at, &key(2)), Some(first));
    }
    assert_eq!(decodes.replaced_side(0, &key(1)), None, "no issue there");

    // The issued side ends where it stands, then stands dispatched.
    let ended = LeaseState::Released(ReleaseReason::Abandoned);
    *decodes.get_mut(at, &key(2)).expect("the pair's side") = laid(1, ended);
    assert_eq!(decodes.replaced_side(at, &key(2)), None, "its side ended");
    let dispatched = LeaseState::Dispatched { at: Instant::now() };
    *decodes.get_mut(at, &key(2)).expect("the pair's side") = laid(1, dispatched);
    assert_eq!(decodes.replaced_side(at, &key(2)), Some(first));

    // The lease leaves, and the next one laid takes its place.
    assert!(decodes.take(at, &key(2)).is_some());
    assert_eq!(decodes.lay(key(3), laid(1, LeaseState::ISSUED)), at);
    assert_eq!(decodes.replaced_side(at, &key(3)), None, "another key's");
    assert!(decodes.take(at, &key(3)).is_some());
    assert_eq!(decodes.lay(key(2), laid(0, LeaseState::ISSUED)), at);
    assert_eq!(decodes.replaced_side(at, &key(2)), None, "never issued");
}
