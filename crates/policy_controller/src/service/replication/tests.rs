//! The journal against the store: every journaled entry tells its event
//! once, in the store's order, and a cut with the tail after it is the next
//! cut, under writers on other threads.

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock, Weak,
    },
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use super::{Entry, Event, Log, LogPositionLost};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::view::{Health, LoadRow, Role, WireModes, WorkerId},
    store::{
        identity, pairing::PairingDescriptor, BatchFacts, BatchOutcome, ContentHash, GatewayId,
        HealthSink, Journal, KvEvent, KvEventSink, LeasesStore, LoadSink, LoadSource,
        MembershipChange, MembershipTail, MemoryStore, OutsideLists, Position, SequenceHash,
        Signals, StoreCheckpoint, StoredBlock, WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus,
        WorkersStore,
    },
    tracking::lease::TransportClass,
};

/// A store with its log, `events` retained.
fn journaled(events: usize) -> (MemoryStore, Arc<Log>) {
    let log = Arc::new(Log::new(events));
    let journal: Arc<dyn Journal> = Arc::clone(&log) as _;
    (MemoryStore::journaled(journal), log)
}

fn facts(key: &str, incarnation: u64) -> WorkerFacts {
    WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        incarnation,
        models: vec!["m".to_owned()],
        dp_size: 1,
        ..WorkerFacts::default()
    }
}

fn running(count: u32) -> LoadRow {
    LoadRow {
        running: count,
        ..LoadRow::default()
    }
}

/// A batch read after `from` ascends from it. A walk that reads from the
/// place of its last entry checks each batch: a read that handed the place
/// back would never come back empty, and the walk would append without end.
pub(super) fn ascends(from: Position, batch: &[Entry]) {
    let mut last = from.seq;
    for entry in batch {
        assert!(
            entry.at.seq > last,
            "{:?} read after position {last}",
            entry.at
        );
        last = entry.at.seq;
    }
}

/// Every entry of the log, each batch read from the place of the last
/// entry before it.
fn all(log: &Log) -> Vec<Entry> {
    let mut entries: Vec<Entry> = Vec::new();
    let mut out = Vec::new();
    loop {
        let after = entries
            .last()
            .map_or_else(Position::default, |entry| entry.at);
        log.read(after, 64, &mut out)
            .expect("the log retains the test's writes");
        if out.is_empty() {
            return entries;
        }
        ascends(after, &out);
        entries.append(&mut out);
    }
}

fn revision(store: &MemoryStore, id: WorkerId) -> u64 {
    WorkersStore::get(store, id).expect("the row").revision
}

/// A membership event's change, the row it left in the table and its
/// epoch.
fn member(entry: &Entry) -> (MembershipChange, &WorkerRow, u64) {
    match &entry.event {
        Event::Member {
            epoch,
            change,
            row: Some(row),
            ..
        } => (*change, row.as_ref(), *epoch),
        other => panic!("a membership event with its row, not {other:?}"),
    }
}

/// The row a membership event's write took out of the table.
fn left(entry: &Entry) -> Option<&WorkerRow> {
    match &entry.event {
        Event::Member { left, .. } => left.as_deref(),
        Event::Term | Event::Generation { .. } => None,
    }
}

#[test]
fn every_journaled_entry_tells_one_event_in_the_stores_order() {
    const KEY: &str = "http://a:8000";
    let (store, log) = journaled(64);
    store.set_term(2);
    let a = store.upsert(facts(KEY, 1)).id();
    store.set_status(a, WorkerStatus::Ready, revision(&store, a));
    let now = Instant::now();
    // The load writes that follow take no place: a load row is told to no
    // journal.
    store.record(a, 0, running(3), now, LoadSource::Stream);
    // A poll that began before that record: rank 0 pushed since, rank 1 did
    // not, so the report writes rank 1 alone.
    let polled = now.checked_sub(Duration::from_secs(1)).unwrap_or(now);
    store.report(
        &[(a, 0, Some(running(4))), (a, 1, Some(running(5)))],
        polled,
        LoadSource::Poll,
    );
    // An answer with no row for any rank of an engine writes nothing and
    // tells nothing: the word that no report came is in no entry. The poll
    // begins after rank 0's push, so the word is said.
    let later = now + Duration::from_millis(1);
    store.report(&[(a, 1, None)], later, LoadSource::Poll);
    let patch = WorkerPatch {
        priority: Some(7),
        ..WorkerPatch::default()
    };
    store.patch(a, patch, revision(&store, a));
    store.upsert(WorkerFacts {
        dp_size: 2,
        ..facts(KEY, 1)
    });
    store.set_term(5);
    let replaced = WorkersStore::get(&store, a).expect("the row");
    let b = store.upsert(facts(KEY, 2)).id();
    // Three failed probes take a ready worker out; the first two move no
    // status and tell nothing.
    for _ in 0..3 {
        store.probe_outcome(b, false, revision(&store, b), &|| true);
    }
    store.connected(b, revision(&store, b), &|| true);
    let found = WorkersStore::get(&store, b).expect("the row");
    store.remove(b, found.revision);

    let entries = all(&log);
    let places: Vec<Position> = entries.iter().map(|entry| entry.at).collect();
    let at = |seq, term| Position { seq, term };
    assert_eq!(
        places,
        [
            at(1, 2),
            at(2, 2),
            at(3, 2),
            at(4, 2),
            at(5, 2),
            at(6, 5),
            at(7, 5),
            at(8, 5),
            at(9, 5),
            at(10, 5)
        ]
    );
    assert_eq!(log.head(), at(10, 5));
    // Each term starts with its own position: the store took it.
    assert_eq!(entries[0].event, Event::Term);
    assert_eq!(entries[5].event, Event::Term);

    let (change, row, epoch) = member(&entries[1]);
    assert_eq!(change, MembershipChange::Registered { id: a });
    assert_eq!(
        (row.id, row.status, row.revision, epoch),
        (a, WorkerStatus::Pending, 1, 1)
    );

    let (change, row, epoch) = member(&entries[2]);
    assert_eq!(
        change,
        MembershipChange::StatusChanged {
            id: a,
            from: WorkerStatus::Pending,
            to: WorkerStatus::Ready
        }
    );
    assert_eq!(
        (row.status, row.revision, epoch),
        (WorkerStatus::Ready, 1, 2)
    );

    let (change, row, epoch) = member(&entries[3]);
    assert_eq!(change, MembershipChange::Relabelled { id: a });
    assert_eq!((row.facts.priority, epoch), (7, 3));

    let (change, row, epoch) = member(&entries[4]);
    assert_eq!(change, MembershipChange::Resized { id: a });
    assert_eq!((row.facts.dp_size, epoch), (2, 4));

    let (change, row, epoch) = member(&entries[6]);
    assert_eq!(change, MembershipChange::Replaced { old: a, new: b });
    assert_eq!(
        (row.id, row.facts.incarnation, row.revision, epoch),
        (b, 2, 1, 5)
    );
    assert_eq!(
        left(&entries[6]),
        Some(&replaced),
        "a replacement tells the incarnation it took out"
    );
    assert!(
        entries[..6].iter().all(|entry| left(entry).is_none()),
        "no other write took a row out"
    );

    let (change, row, _) = member(&entries[7]);
    assert_eq!(
        change,
        MembershipChange::StatusChanged {
            id: b,
            from: WorkerStatus::Ready,
            to: WorkerStatus::NotReady
        }
    );
    assert_eq!(row.status, WorkerStatus::NotReady);

    let (change, row, _) = member(&entries[8]);
    assert_eq!(
        change,
        MembershipChange::StatusChanged {
            id: b,
            from: WorkerStatus::NotReady,
            to: WorkerStatus::Ready
        }
    );
    assert_eq!(row.status, WorkerStatus::Ready);

    match &entries[9].event {
        Event::Member {
            epoch,
            change,
            row: None,
            left: Some(left),
        } => {
            assert_eq!(*change, MembershipChange::Removed { id: b });
            assert_eq!(left.as_ref(), &found, "a removal tells the row it took out");
            assert_eq!(*epoch, store.epoch());
        }
        other => panic!("a removal with the row it took out, not {other:?}"),
    }
}

#[test]
fn a_write_the_store_refused_and_an_entry_off_the_journal_tell_nothing() {
    let (store, log) = journaled(64);
    let a = store.upsert(facts("http://a:8000", 1)).id();
    let now = Instant::now();
    let stale = revision(&store, a) + 1;
    assert_eq!(store.set_status(a, WorkerStatus::Ready, stale), None);
    assert_eq!(store.remove(a, stale), None);
    assert_eq!(store.patch(a, WorkerPatch::default(), stale), None);
    // The status already stands.
    assert_eq!(
        store.set_status(a, WorkerStatus::Pending, revision(&store, a)),
        None
    );
    // No such member: a load for it is dropped.
    store.record(WorkerId(42), 0, running(1), now, LoadSource::Stream);
    store.report(
        &[(WorkerId(42), 0, Some(running(1)))],
        now,
        LoadSource::Poll,
    );
    // A gateway's own facts: its veto and its completed report.
    store.gateway_veto(GatewayId(1), a, Some(now + Duration::from_secs(5)));
    HealthSink::completed(&store, GatewayId(1), a, TransportClass::Ok, now);
    store.evict(a);
    store.publish(now + Duration::from_secs(1));
    assert_eq!(
        log.head(),
        Position { seq: 1, term: 0 },
        "the registration alone"
    );
}

/// The listener of the watched kinds is told once the store has let its
/// locks go, by every path a journaled write takes: what the listener
/// wakes stands in no writer's way, and in no reservation's. A write the
/// store refused, a term that did not rise and a load row tell it nothing.
#[test]
fn a_watched_write_is_told_once_the_store_let_its_locks_go() {
    const KEY: &str = "http://a:8000";
    // Whether the store's locks were free, at each telling.
    let told = Arc::new(Mutex::new(Vec::new()));
    let written: Arc<OnceLock<Weak<MemoryStore>>> = Arc::default();
    let log = {
        let (told, written) = (Arc::clone(&told), Arc::clone(&written));
        Log::new(64).on_watched(move || {
            let store = written.get().and_then(Weak::upgrade);
            told.lock()
                .push(store.expect("the store that wrote").unlocked());
        })
    };
    let store = Arc::new(MemoryStore::journaled(Arc::new(log)));
    assert!(written.set(Arc::downgrade(&store)).is_ok());
    // The paths told while a lock was held: every one is named, not the
    // first alone.
    let (mut seen, mut held) = (0, Vec::new());
    let mut tells = |path: &'static str, want: usize| {
        let told = told.lock();
        assert_eq!(told.len() - seen, want, "{path}: its tellings");
        if !told[seen..].iter().all(|free| *free) {
            held.push(path);
        }
        seen = told.len();
    };
    store.set_term(1);
    tells("a term's start", 1);
    let a = store.upsert(facts(KEY, 1)).id();
    tells("a registration", 1);
    store.upsert(WorkerFacts {
        dp_size: 2,
        ..facts(KEY, 1)
    });
    tells("a registration told again", 1);
    let patch = WorkerPatch {
        priority: Some(7),
        ..WorkerPatch::default()
    };
    store.patch(a, patch, revision(&store, a));
    tells("a patch", 1);
    store.set_status(a, WorkerStatus::Ready, revision(&store, a));
    tells("a status", 1);
    let b = store.upsert(facts(KEY, 2)).id();
    tells("a replacement", 1);
    // Three failed probes take a ready worker out: the first two move no
    // status.
    for _ in 0..3 {
        store.probe_outcome(b, false, revision(&store, b), &|| true);
    }
    tells("a probe's outcome", 1);
    store.connected(b, revision(&store, b), &|| true);
    tells("a connection", 1);
    let stale = revision(&store, b) + 1;
    assert_eq!(store.remove(b, stale), None);
    store.set_term(1);
    store.record(b, 0, running(1), Instant::now(), LoadSource::Stream);
    tells("a refused write, a term that did not rise, a load row", 0);
    store.remove(b, revision(&store, b));
    tells("a removal", 1);
    assert!(held.is_empty(), "told under a lock of the store: {held:?}");
}

#[test]
fn the_stores_term_reaches_its_journal_and_never_falls() {
    let (store, log) = journaled(8);
    store.set_term(3);
    store.set_term(2);
    assert_eq!((store.term(), log.term()), (3, 3));
    assert_eq!(
        log.head(),
        Position { seq: 1, term: 3 },
        "the take is the term's first position"
    );
    store.upsert(facts("http://a:8000", 1));
    assert_eq!(log.head(), Position { seq: 2, term: 3 });
}

/// A journal that keeps the terms it is told.
#[derive(Default)]
struct Told {
    terms: Mutex<Vec<u64>>,
}

impl Journal for Told {
    fn member(
        &self,
        _: Option<Position>,
        _: u64,
        _: MembershipChange,
        _: Option<&WorkerRow>,
        _: Option<&WorkerRow>,
    ) {
    }

    fn term_rose(&self, term: u64) {
        self.terms.lock().push(term);
    }

    fn generation(&self, _: Option<Position>, _: u64) {}

    fn term_followed(&self, _: Position) {}

    fn head(&self) -> Position {
        Position::default()
    }

    fn followed(&self) -> Option<u64> {
        None
    }

    fn restart(&self, _: Position) -> bool {
        true
    }
}

#[test]
fn the_store_tells_a_term_once_when_it_rises() {
    let told = Arc::new(Told::default());
    let store = MemoryStore::journaled(Arc::clone(&told) as _);
    for term in [2, 2, 1, 4, 4, 3] {
        store.set_term(term);
    }
    assert_eq!(store.term(), 4);
    assert_eq!(*told.terms.lock(), [2, 4]);
}

#[test]
fn a_load_write_takes_no_position_and_the_store_holds_its_row() {
    let log = Arc::new(Log::new(8));
    let store = MemoryStore::journaled(Arc::clone(&log) as _);
    let a = store.upsert(facts("http://a:8000", 1)).id();
    let now = Instant::now();
    store.record(a, 0, running(1), now, LoadSource::Stream);
    store.report(&[(a, 1, Some(running(2)))], now, LoadSource::Poll);
    assert_eq!(
        log.head(),
        Position { seq: 1, term: 0 },
        "the registration alone"
    );
    assert!(store.load(a, 0).is_some() && store.load(a, 1).is_some());
}

#[test]
fn a_store_with_no_journal_cuts_at_the_start_and_a_journaled_one_at_its_head() {
    let plain = MemoryStore::new();
    plain.upsert(facts("http://a:8000", 1));
    let cut = plain.checkpoint();
    assert_eq!((cut.at, cut.rows.len()), (Position::default(), 1));

    let (store, log) = journaled(8);
    store.set_term(4);
    let a = store.upsert(facts("http://b:8000", 1)).id();
    store.upsert(facts("http://a:8000", 1));
    store.record(a, 1, running(2), Instant::now(), LoadSource::Stream);
    // The cut stands at the last journaled write: the load row after it
    // took no place.
    let cut = store.checkpoint();
    assert_eq!(cut.at, log.head());
    assert_eq!(cut.at, Position { seq: 3, term: 4 });
    assert_eq!(cut.epoch, store.epoch());
    let keys: Vec<&str> = cut.rows.iter().map(|row| row.facts.key.as_str()).collect();
    assert_eq!(keys, ["http://a:8000", "http://b:8000"], "rows by key");
}

/// What a standby holds after a cut and the events it applied since: the
/// rows.
#[derive(Debug, PartialEq)]
struct Twin {
    rows: BTreeMap<WorkerId, WorkerRow>,
}

impl Twin {
    fn of(cut: &StoreCheckpoint) -> Self {
        Self {
            rows: cut.rows.iter().map(|row| (row.id, row.clone())).collect(),
        }
    }

    fn leave(&mut self, id: WorkerId) {
        self.rows.remove(&id);
    }

    fn apply(&mut self, event: &Event) {
        match event {
            Event::Member { change, row, .. } => match *change {
                MembershipChange::Removed { id } => self.leave(id),
                MembershipChange::Replaced { old, new } => {
                    self.leave(old);
                    let row = row.as_deref().expect("a replacement tells its new row");
                    self.rows.insert(new, row.clone());
                }
                other => {
                    let row = row.as_deref().expect("a membership write tells its row");
                    self.rows.insert(other.id(), row.clone());
                }
            },
            Event::Term | Event::Generation { .. } => {}
        }
    }
}

/// A small deterministic sequence.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, below: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % below
    }
}

const KEYS: usize = 6;

fn key(index: usize) -> String {
    format!("http://w{index}:8000")
}

/// Registers, flips, refreshes, replaces and removes workers over a few
/// keys. `lives[k]` names the incarnation key `k` is about to get, before
/// the write that makes it, so a load writer aims at a worker in the very
/// write that registers it.
fn churn(store: &MemoryStore, lives: &[AtomicU64], rounds: u32) {
    let mut draw = Lcg(7);
    for _ in 0..rounds {
        let index = draw.next(KEYS as u64) as usize;
        let name = key(index);
        match (store.by_key(&name), draw.next(6)) {
            (None, _) => {
                let life = lives[index].fetch_add(1, Ordering::SeqCst) + 1;
                store.upsert(facts(&name, life));
            }
            (Some(row), 0) => {
                let life = lives[index].fetch_add(1, Ordering::SeqCst) + 1;
                assert_ne!(row.facts.incarnation, life);
                store.upsert(facts(&name, life));
            }
            (Some(row), 1) => {
                store.remove(row.id, row.revision);
            }
            (Some(row), 2) => {
                store.upsert(WorkerFacts {
                    priority: row.facts.priority + 1,
                    ..row.facts
                });
            }
            (Some(row), _) => {
                let status = if row.status == WorkerStatus::Ready {
                    WorkerStatus::NotReady
                } else {
                    WorkerStatus::Ready
                };
                store.set_status(row.id, status, row.revision);
            }
        }
    }
}

/// Writes load rows for the workers the keys hold or are about to hold.
fn loads(store: &MemoryStore, lives: &[AtomicU64], seed: u64, writes: u32) {
    let mut draw = Lcg(seed);
    let now = Instant::now();
    for write in 0..writes {
        let index = draw.next(KEYS as u64) as usize;
        let life = lives[index].load(Ordering::SeqCst);
        let id = identity(&key(index), life);
        let rank = draw.next(2) as u16;
        if draw.next(4) == 0 {
            store.report(&[(id, rank, Some(running(write)))], now, LoadSource::Poll);
        } else {
            store.record(id, rank, running(write), now, LoadSource::Stream);
        }
    }
}

#[test]
fn a_cut_and_the_tail_after_it_are_the_next_cut_under_writers() {
    const LEGS: u32 = 4;
    const ROUNDS: u32 = 1_000;
    const WRITES: u32 = 5_000;
    const CUTS: usize = 500;
    let (store, log) = journaled(1 << 17);
    let lives: Vec<AtomicU64> = (0..KEYS).map(|_| AtomicU64::new(0)).collect();
    let mut cuts = vec![store.checkpoint()];
    // The writers' work in legs. While a leg runs, a cut is taken each time
    // the log has moved past the last one, a bounded number of them, and one
    // more behind the leg. The load writers write beside the churn: a load
    // row takes no place, and a cut waits for none of them.
    for _ in 0..LEGS {
        thread::scope(|scope| {
            let writers = [
                scope.spawn(|| churn(&store, &lives, ROUNDS)),
                scope.spawn(|| loads(&store, &lives, 11, WRITES)),
                scope.spawn(|| loads(&store, &lives, 13, WRITES)),
            ];
            let mut taken = 0;
            while taken < CUTS && writers.iter().any(|writer| !writer.is_finished()) {
                if cuts.last().is_some_and(|cut| cut.at == log.head()) {
                    thread::yield_now();
                    continue;
                }
                cuts.push(store.checkpoint());
                taken += 1;
            }
        });
        cuts.push(store.checkpoint());
    }

    let entries = all(&log);
    let place = |cut: &StoreCheckpoint| usize::try_from(cut.at.seq).expect("a count of events");
    let mut out = Vec::new();
    let (mut members, mut moved) = (0u32, 0u32);
    for pair in cuts.windows(2) {
        let (from, to) = (&pair[0], &pair[1]);
        let mut twin = Twin::of(from);
        let after = &entries[place(from)..place(to)];
        for entry in after {
            members += u32::from(matches!(entry.event, Event::Member { .. }));
            twin.apply(&entry.event);
        }
        moved += u32::from(!after.is_empty());
        assert_eq!(
            twin,
            Twin::of(to),
            "the cut at {:?} and the {} events after it are not the cut at {:?}",
            from.at,
            after.len(),
            to.at
        );
        // The cut's place is a place of the log: the tail is read from it.
        assert_eq!(log.read(to.at, 1, &mut out), Ok(()), "{:?}", to.at);
        assert_eq!(
            out.first().map(|entry| entry.at.seq),
            (to.at != log.head()).then_some(to.at.seq + 1)
        );
    }

    // The case is held to what it claims, by its legs and not by the
    // scheduler: every leg wrote membership events, no load row took a
    // place, and a cut stands behind each leg.
    assert_eq!(u64::from(members), log.head().seq);
    assert!(members >= LEGS * ROUNDS / 2, "{members} membership events");
    assert!(moved >= LEGS, "{moved} cuts with a tail, of {}", cuts.len());
}

#[test]
fn a_reader_of_a_restarted_store_starts_at_its_cut_and_reads_on() {
    let (store, log) = journaled(64);
    let mut out = Vec::new();
    // The log restarts at a leader's place, as a restore of the store has
    // it, and the store writes on under no term. A reader starts from a
    // cut, which stands at the store's own write, and reads what followed:
    // the cut with that tail is the next cut.
    log.restart(Position { seq: 2, term: 2 })
        .expect("a place the log takes");
    store.upsert(facts("http://a:8000", 1));
    let first = store.checkpoint();
    assert_eq!(first.at, Position { seq: 3, term: 0 });
    assert_eq!(log.read(first.at, 64, &mut out), Ok(()));
    assert!(out.is_empty());
    let b = store.upsert(facts("http://b:8000", 1)).id();
    store.record(b, 0, running(2), Instant::now(), LoadSource::Stream);
    assert_eq!(log.read(first.at, 64, &mut out), Ok(()));
    let mut twin = Twin::of(&first);
    for entry in &out {
        twin.apply(&entry.event);
    }
    let next = store.checkpoint();
    assert_eq!(out.last().map(|entry| entry.at), Some(next.at));
    assert_eq!(twin, Twin::of(&next));
    // A restart below those numbers: the first reader's place is gone for
    // good, and the cut after the next write stands past it.
    log.restart(Position { seq: 2, term: 3 })
        .expect("a place the log takes");
    store.upsert(facts("http://c:8000", 1));
    assert_eq!(log.read(first.at, 64, &mut out), Err(LogPositionLost));
    let again = store.checkpoint();
    assert_eq!(again.at, Position { seq: 5, term: 0 });
    assert_eq!(log.read(again.at, 64, &mut out), Ok(()));
    assert!(out.is_empty());
}

/// One of a leader's entries applied to a standby's store under the guard
/// of its position, the way the standby's applier does.
fn follows(standby: &MemoryStore, log: &Log, entry: &Entry) {
    let adopting = log.adopt(entry.at).expect("the next position");
    applies(standby, entry);
    assert!(adopting.landed(), "{entry:?}");
}

/// One of a leader's entries told to a standby's store by the door for a
/// followed write of its kind, at the entry's place.
fn applies(standby: &MemoryStore, entry: &Entry) {
    match &entry.event {
        Event::Member {
            epoch, change, row, ..
        } => {
            let (change, row) = told(*change, row.as_deref());
            standby.follow(entry.at, *epoch, change, row);
        }
        Event::Term => standby.follow_term(entry.at),
        Event::Generation { generation } => {
            standby.follow_generation_place(entry.at, *generation);
        }
    }
}

/// A membership change as the replication stream tells it: no row for a
/// removal or a status change, and a resize told as a relabel.
fn told(
    change: MembershipChange,
    row: Option<&WorkerRow>,
) -> (MembershipChange, Option<WorkerRow>) {
    match change {
        MembershipChange::Removed { .. } | MembershipChange::StatusChanged { .. } => (change, None),
        MembershipChange::Resized { id } => (MembershipChange::Relabelled { id }, row.cloned()),
        _ => (change, row.cloned()),
    }
}

/// The leader's writes of a few rounds: membership churn, then load rows.
fn rounds(leader: &MemoryStore, lives: &[AtomicU64], count: u64) {
    for round in 0..count {
        churn(leader, lives, 5);
        loads(leader, lives, round, 8);
    }
}

/// A standby's tables are the leader's: the place, the epoch and every row
/// under its key.
fn twins(standby: &MemoryStore, leader: &MemoryStore) {
    let (ours, theirs) = (standby.checkpoint(), leader.checkpoint());
    assert_eq!((ours.at, ours.epoch), (theirs.at, theirs.epoch));
    assert_eq!(ours.rows, theirs.rows);
    for row in &ours.rows {
        let found = standby.by_key(&row.facts.key).map(|found| found.id);
        assert_eq!(found, Some(row.id), "{}", row.facts.key);
    }
    let at = later();
    assert_eq!(pool(standby, at), pool(leader, at));
}

/// An instant past the window any earlier read of a pool opened: a second
/// after the last one handed out, so each read publishes what waits.
fn later() -> Instant {
    static READS: AtomicU64 = AtomicU64::new(0);
    Instant::now() + Duration::from_secs(READS.fetch_add(1, Ordering::SeqCst) + 1)
}

/// What the store's world holds of each worker on file at `at`: the id,
/// width and health bits of its published row, by id. A worker no pool
/// lists has none. A standby follows no load row, so the load rows are set
/// aside.
fn pool(store: &MemoryStore, at: Instant) -> Vec<(WorkerId, u16, Health)> {
    store.publish(at);
    let world = store.snapshot();
    let on_file = store.checkpoint().rows;
    let mut ids: Vec<WorkerId> = on_file.iter().map(|row| row.id).collect();
    ids.sort_unstable();
    ids.into_iter()
        .filter_map(|id| world.first_row_of(id))
        .map(|row| (row.id, row.dp_size, row.health))
        .collect()
}

#[test]
fn a_standby_that_follows_holds_the_leaders_tables_and_its_log() {
    let (leader, led) = journaled(4_096);
    let (standby, followed) = journaled(4_096);
    leader.set_term(3);
    let lives: Vec<AtomicU64> = (0..KEYS).map(|_| AtomicU64::new(0)).collect();
    // Each read finds the change before it the last one the standby
    // followed: none leans on a later write to be shown.
    let mut told = 0;
    let mut reads = |leader: &MemoryStore| {
        let entries = all(&led);
        for entry in entries.iter().skip(told) {
            follows(&standby, &followed, entry);
        }
        told = entries.len();
        twins(&standby, leader);
    };
    rounds(&leader, &lives, 40);
    reads(&leader);
    let wide = leader.upsert(facts("http://wide:8000", 1)).id();
    let revision = WorkersStore::get(&leader, wide).expect("the row").revision;
    leader.set_status(wide, WorkerStatus::Ready, revision);
    leader.record(wide, 0, running(3), Instant::now(), LoadSource::Stream);
    reads(&leader);
    // A width change, which the stream tells as a relabel.
    let wider = WorkerFacts {
        dp_size: 2,
        ..facts("http://wide:8000", 1)
    };
    assert!(matches!(
        leader.upsert(wider),
        MembershipChange::Resized { .. }
    ));
    reads(&leader);
    // The worker leaves: its row and its key leave the standby.
    let left = leader.by_key("http://wide:8000").expect("the row");
    assert!(leader.remove(left.id, left.revision).is_some());
    reads(&leader);
    assert!(standby.by_key("http://wide:8000").is_none());
    // It comes back under its key, which the removal freed.
    assert!(matches!(
        leader.upsert(facts("http://wide:8000", 1)),
        MembershipChange::Registered { .. }
    ));
    reads(&leader);
    let entries = all(&led);
    assert!(entries.len() > 200, "{} entries", entries.len());
    assert_eq!(all(&followed), entries);
    assert!(!pool(&standby, later()).is_empty());
    assert_eq!(standby.term(), 3, "the leader's term, told under its guard");
    assert_eq!(followed.followed(), Some(3));
}

#[test]
fn a_standbys_index_holds_a_lane_for_each_row_that_asks_for_events() {
    let asks = |key: &str, incarnation| WorkerFacts {
        signals: Signals {
            events: true,
            ..Signals::default()
        },
        ..facts(key, incarnation)
    };
    let (leader, led) = journaled(256);
    let (standby, followed) = journaled(256);
    leader.set_term(1);
    let a = leader.upsert(asks("http://a:8000", 1)).id();
    let b = leader.upsert(asks("http://b:8000", 1)).id();
    let quiet = leader.upsert(facts("http://quiet:8000", 1)).id();
    let lanes = |store: &MemoryStore, ids: &[WorkerId]| -> Vec<bool> {
        let lane = |id: &WorkerId| store.lane_counts(*id, 0).is_some();
        ids.iter().map(lane).collect()
    };
    // Each read: the standby follows what the leader wrote since the last,
    // and both hold a lane for exactly the rows that ask.
    let mut told = 0;
    let mut reads = |ids: &[WorkerId], want: &[bool]| {
        let entries = all(&led);
        for entry in entries.iter().skip(told) {
            follows(&standby, &followed, entry);
        }
        told = entries.len();
        assert_eq!(lanes(&leader, ids), want, "the leader");
        assert_eq!(lanes(&standby, ids), want, "the standby");
    };
    reads(&[a, b, quiet], &[true, true, false]);
    // A relabel that stops asking retires the lane, one that asks makes it.
    leader.upsert(facts("http://a:8000", 1));
    leader.upsert(asks("http://quiet:8000", 1));
    reads(&[a, b, quiet], &[false, true, true]);
    // A replaced row's lane leaves with it and the new row takes its own.
    let again = leader.upsert(asks("http://b:8000", 2)).id();
    reads(&[b, again], &[false, true]);
    // A removed row's lane leaves with it.
    let left = leader.by_key("http://quiet:8000").expect("the row");
    assert!(leader.remove(left.id, left.revision).is_some());
    reads(&[quiet, again], &[false, true]);
    // A restore: the lanes are the cut's rows', and a row that left gave
    // its lane up.
    let (other, _) = journaled(256);
    let own = other.upsert(asks("http://own:8000", 1)).id();
    assert_eq!(lanes(&other, &[own, again]), [true, false]);
    assert!(other.restore_from(leader.checkpoint()));
    assert_eq!(lanes(&other, &[own, again, a]), [false, true, false]);
    // A row that stays through a restore keeps what this replica heard for
    // it: its lane's blocks are the replica's own, in no cut.
    let heard: &dyn KvEventSink<Batch = [KvEvent]> = &other;
    let block = [KvEvent::Stored {
        parent: None,
        blocks: vec![StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: ContentHash(2),
        }],
    }];
    let first = BatchFacts {
        sequence: 1,
        ..BatchFacts::default()
    };
    assert_eq!(heard.admit(again, 0, first, &block), BatchOutcome::Applied);
    assert_eq!(other.indexed_blocks(again, 0), 1);
    assert!(other.restore_from(leader.checkpoint()));
    assert_eq!(other.indexed_blocks(again, 0), 1, "the lane was made again");
}

#[test]
fn a_restore_and_the_tail_after_it_are_the_leaders_tables() {
    let (leader, led) = journaled(4_096);
    leader.set_term(3);
    let lives: Vec<AtomicU64> = (0..KEYS).map(|_| AtomicU64::new(0)).collect();
    rounds(&leader, &lives, 20);
    // The standby held a row of its own, written under no term.
    let (standby, followed) = journaled(4_096);
    standby.upsert(facts("http://stale:8000", 1));
    let cut = leader.checkpoint();
    let at = cut.at;
    assert!(standby.restore_from(cut));
    assert_eq!(standby.term(), 3);
    assert_eq!(followed.head(), at);
    assert!(standby.by_key("http://stale:8000").is_none());
    twins(&standby, &leader);
    rounds(&leader, &lives, 20);
    for entry in all(&led).iter().filter(|entry| entry.at.seq > at.seq) {
        follows(&standby, &followed, entry);
    }
    twins(&standby, &leader);
}

/// A standby's two doors tell the journal's listener as a leader's writes
/// do, once the store has let its locks go: a restore restarts the journal
/// inside its locks, and a followed write is journaled under the membership
/// lock.
#[test]
fn a_restore_and_a_followed_write_are_told_once_the_store_let_its_locks_go() {
    let (leader, led) = journaled(64);
    leader.set_term(3);
    leader.upsert(facts("http://a:8000", 1));
    // Whether the standby's locks were free, at each telling.
    let told = Arc::new(Mutex::new(Vec::new()));
    let written: Arc<OnceLock<Weak<MemoryStore>>> = Arc::default();
    let followed = {
        let (told, written) = (Arc::clone(&told), Arc::clone(&written));
        Arc::new(Log::new(64).on_watched(move || {
            let store = written.get().and_then(Weak::upgrade);
            told.lock()
                .push(store.expect("the store that wrote").unlocked());
        }))
    };
    let journal: Arc<dyn Journal> = Arc::<Log>::clone(&followed);
    let standby = Arc::new(MemoryStore::journaled(journal));
    assert!(written.set(Arc::downgrade(&standby)).is_ok());
    let cut = leader.checkpoint();
    let at = cut.at;
    assert!(standby.restore_from(cut));
    assert_eq!(*told.lock(), [true], "a restore");
    leader.upsert(facts("http://b:8000", 1));
    for entry in all(&led).iter().filter(|entry| entry.at.seq > at.seq) {
        follows(&standby, &followed, entry);
    }
    assert_eq!(*told.lock(), [true, true], "a followed registration");
}

/// A replica restored from a leader's cut serves its log in three shapes:
/// before its first follow, leading alone under a term it took, and alone
/// under no term. In each a reader that stood at the replica's own write
/// before the restore is lost for good, and a reader that starts again
/// from a cut reads on: the cut with the tail after it is the next cut.
#[test]
fn a_reader_of_a_restored_replica_recovers_at_a_cut_in_every_shape_it_serves_in() {
    for shape in [
        "before its first follow",
        "leading alone",
        "alone under no term",
    ] {
        let (leader, led) = journaled(64);
        leader.set_term(3);
        leader.upsert(facts("http://a:8000", 1));
        // Three writes of the replica's own, under no term: a reader
        // stands at the last.
        let (replica, log) = journaled(64);
        for key in ["http://x:8000", "http://y:8000", "http://z:8000"] {
            replica.upsert(facts(key, 1));
        }
        let stale = replica.checkpoint().at;
        assert_eq!(stale, Position { seq: 3, term: 0 }, "{shape}");
        // The restore stands at the leader's second position.
        assert!(replica.restore_from(leader.checkpoint()), "{shape}");
        match shape {
            "before its first follow" => {}
            "leading alone" => {
                replica.set_term(4);
            }
            _ => {
                replica.upsert(facts("http://late:8000", 1));
            }
        }
        let mut out = Vec::new();
        assert_eq!(
            log.read(stale, 64, &mut out),
            Err(LogPositionLost),
            "{shape}"
        );
        let cut = replica.checkpoint();
        assert_eq!(log.read(cut.at, 64, &mut out), Ok(()), "{shape}");
        assert!(out.is_empty(), "{shape}");
        // One more write in the replica's shape: the leader's, followed, or
        // its own.
        if shape == "before its first follow" {
            leader.upsert(facts("http://b:8000", 1));
            let entries = all(&led);
            follows(&replica, &log, entries.last().expect("the leader's write"));
        } else {
            replica.upsert(facts("http://b:8000", 1));
        }
        assert_eq!(
            log.read(stale, 64, &mut out),
            Err(LogPositionLost),
            "{shape}"
        );
        assert_eq!(log.read(cut.at, 64, &mut out), Ok(()), "{shape}");
        assert_eq!(out.len(), 1, "{shape}");
        let mut twin = Twin::of(&cut);
        for entry in &out {
            twin.apply(&entry.event);
        }
        assert_eq!(twin, Twin::of(&replica.checkpoint()), "{shape}");
    }
}

#[test]
fn a_restore_the_journal_refuses_changes_nothing() {
    // The store leads term 3 with a row of its own.
    let (store, log) = journaled(64);
    store.set_term(3);
    store.upsert(facts("http://own:8000", 1));
    let held = |store: &MemoryStore| {
        let cut = store.checkpoint();
        (cut.at, cut.epoch, Twin::of(&cut), store.term())
    };
    let before = (held(&store), all(&log));
    // A cut under no term, under a term below its own and under the term
    // it leads: none is a history ahead of its own.
    for term in [0, 2, 3] {
        let (other, _) = journaled(64);
        other.set_term(term);
        other.upsert(facts("http://theirs:8000", 1));
        other.upsert(facts("http://more:8000", 1));
        assert!(!store.restore_from(other.checkpoint()), "term {term}");
        assert_eq!((held(&store), all(&log)), before, "term {term}");
    }
    // A store under no term takes no cut under no term: no leader wrote it.
    let (fresh, fresh_log) = journaled(64);
    let (other, _) = journaled(64);
    other.upsert(facts("http://theirs:8000", 1));
    assert!(!fresh.restore_from(other.checkpoint()));
    assert!(fresh.by_key("http://theirs:8000").is_none());
    assert!(all(&fresh_log).is_empty());
    // A term above its own is a leader's.
    let (leader, _) = journaled(64);
    leader.set_term(4);
    leader.upsert(facts("http://theirs:8000", 1));
    assert!(store.restore_from(leader.checkpoint()));
    assert_eq!(store.term(), 4);
    assert_eq!(log.followed(), Some(4));
}

#[test]
fn a_restore_never_lowers_the_stores_term() {
    // A store with no journal has no log to refuse a cut under a term below
    // its own.
    let store = MemoryStore::new();
    store.set_term(5);
    let (leader, _) = journaled(64);
    leader.set_term(3);
    leader.upsert(facts("http://theirs:8000", 1));
    assert!(store.restore_from(leader.checkpoint()));
    assert!(store.by_key("http://theirs:8000").is_some());
    assert_eq!(store.term(), 5);
}

#[test]
fn a_restore_leaves_nothing_of_the_tables_it_replaces() {
    const ASKER: GatewayId = GatewayId(1);
    let (leader, _led) = journaled(64);
    leader.set_term(1);
    let kept = leader.upsert(facts("http://kept:8000", 1)).id();
    // The store held the row that stays and one of its own, each with what
    // lives as long as a row.
    let (store, _log) = journaled(64);
    let gone = store.upsert(facts("http://gone:8000", 1)).id();
    store.upsert(facts("http://kept:8000", 1));
    let now = Instant::now();
    store.record(gone, 0, running(1), now, LoadSource::Stream);
    HealthSink::completed(&store, ASKER, kept, TransportClass::Ok, now);
    store.gateway_veto(ASKER, kept, Some(now + Duration::from_secs(3_600)));
    // Whether the published world hands the asker a held veto on the row.
    let held = |store: &MemoryStore| {
        let at = later();
        store.publish(at);
        let world = store.snapshot();
        let mut standing = world
            .gateway_vetoes("m", Role::Single, ASKER)
            .standing_at(at);
        standing.any(|veto| veto.id == kept && veto.held)
    };
    assert!(held(&store));
    assert!(store.restore_from(leader.checkpoint()));
    assert!(WorkersStore::get(&store, gone).is_none());
    assert!(store.load(gone, 0).is_none());
    assert!(!store.vetoed(ASKER, kept, now), "the veto in the table");
    assert!(!held(&store), "the veto in the world");
    assert!(!store.completed_ok_since(ASKER, kept, now));
    // A watcher of the store that was reads no change across the restore.
    assert_eq!(store.changes_since(0), MembershipTail::SnapshotRequired);
    // And the key that left is free: its next registration is a first one.
    assert_eq!(
        store.upsert(facts("http://gone:8000", 1)),
        MembershipChange::Registered { id: gone }
    );
}

#[test]
fn a_restore_reads_each_row_it_brings_under_the_lists_and_carries_no_field() {
    // A generation that lists one zone: a member outside it is counted.
    let listing = || {
        let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
        document.generation = 1;
        document.view.zones = vec!["east".to_owned()];
        AppliedSpec::validate(&document).expect("the document validates")
    };
    // Two leaders admit one key in two zones: each mints the same id at
    // the same revision.
    let lead = |zone: &str| {
        let (leader, _led) = journaled(64);
        leader.set_term(1);
        let member = WorkerFacts {
            zone: Some(zone.to_owned()),
            ..facts("http://a:8000", 1)
        };
        let id = leader.upsert(member).id();
        let at = revision(&leader, id);
        (leader, id, at)
    };
    let (inside, id, at) = lead("east");
    let (outside, again, revised) = lead("west");
    assert_eq!((again, revised), (id, at));
    let (standby, _log) = journaled(64);
    standby.apply(listing()).expect("applied");
    assert!(standby.restore_from(inside.checkpoint()));
    assert_eq!(standby.outside_lists(), OutsideLists::default());
    // The next cut brings a row at the id and the revision the standby's
    // world read one at, in the other zone. No field of the row that left
    // is carried: the row is read under the lists again.
    assert!(standby.restore_from(outside.checkpoint()));
    let counted = OutsideLists {
        engines: 0,
        zones: 1,
    };
    assert_eq!(standby.outside_lists(), counted);
}

#[test]
fn a_cut_beside_restores_is_one_restore_whole() {
    const RESTORES: usize = 2_000;
    // Two cuts of one leader, a few writes apart.
    let (leader, _led) = journaled(4_096);
    leader.set_term(1);
    let lives: Vec<AtomicU64> = (0..KEYS).map(|_| AtomicU64::new(0)).collect();
    rounds(&leader, &lives, 10);
    let first = leader.checkpoint();
    rounds(&leader, &lives, 10);
    let second = leader.checkpoint();
    let whole = |cut: &StoreCheckpoint| (cut.at, cut.epoch, Twin::of(cut));
    let cuts = [whole(&first), whole(&second)];
    assert_ne!(cuts[0], cuts[1]);
    let (standby, _followed) = journaled(4_096);
    assert!(standby.restore_from(first.clone()));
    thread::scope(|scope| {
        let restorer = scope.spawn(|| {
            for turn in 0..RESTORES {
                let cut = if turn % 2 == 0 { &second } else { &first };
                assert!(standby.restore_from(cut.clone()));
            }
        });
        // The place, the epoch and the tables of a cut are one restore's.
        loop {
            let seen = whole(&standby.checkpoint());
            assert!(
                cuts.contains(&seen),
                "a cut between two restores, at {:?}",
                seen.0
            );
            if restorer.is_finished() {
                break;
            }
        }
    });
}

#[test]
fn a_load_row_of_the_standbys_own_inside_an_adoption_takes_no_place() {
    let (leader, led) = journaled(64);
    leader.set_term(1);
    let a = leader.upsert(facts("http://a:8000", 1)).id();
    let before = all(&led).len();
    // The leader's next events, one of each kind: a member and the start of
    // its next term.
    leader.upsert(facts("http://b:8000", 1));
    leader.set_term(2);
    let entries = all(&led);
    assert_eq!(entries.len(), before + 2);
    for (index, event) in entries.iter().enumerate().skip(before) {
        let (standby, followed) = journaled(64);
        for entry in &entries[..index] {
            follows(&standby, &followed, entry);
        }
        // A poll's answer that was out when the standby stopped leading
        // lands while the applier holds the event's place. It tells the
        // journal nothing: the hold stands, and the leader's event lands.
        let adopting = followed.adopt(event.at).expect("the next position");
        standby.record(a, 0, running(9), Instant::now(), LoadSource::Poll);
        applies(&standby, event);
        assert!(adopting.landed(), "{event:?}");
        drop(adopting);
        assert_eq!(all(&followed), entries[..=index], "{event:?}");
    }
}

#[test]
fn a_change_that_does_not_fit_the_tables_is_not_applied_and_takes_no_place() {
    use MembershipChange::{Registered, Relabelled, Removed, Replaced, StatusChanged};
    let (leader, led) = journaled(64);
    leader.set_term(1);
    let a = leader.upsert(facts("http://a:8000", 1)).id();
    let c = leader.upsert(facts("http://c:8000", 1)).id();
    let (standby, followed) = journaled(64);
    for entry in &all(&led) {
        follows(&standby, &followed, entry);
    }
    let held = WorkersStore::get(&standby, a).expect("the row");
    let b = identity("http://b:8000", 1);
    let other = WorkerRow {
        id: b,
        facts: facts("http://b:8000", 1),
        ..held.clone()
    };
    let again = WorkerRow {
        id: identity("http://b:8000", 2),
        ..other.clone()
    };
    let next = standby.epoch() + 1;
    let (pending, ready) = (WorkerStatus::Pending, WorkerStatus::Ready);
    let misfits = [
        (
            "an epoch that is not the next",
            next + 1,
            Registered { id: b },
            Some(other.clone()),
        ),
        (
            "a registration of a held id",
            next,
            Registered { id: a },
            Some(held.clone()),
        ),
        (
            "a registration of a held id under another key",
            next,
            Registered { id: a },
            Some(WorkerRow {
                id: a,
                ..other.clone()
            }),
        ),
        (
            "a registration under a held key",
            next,
            Registered { id: b },
            Some(WorkerRow {
                id: b,
                ..held.clone()
            }),
        ),
        (
            "a registration that tells no row",
            next,
            Registered { id: b },
            None,
        ),
        (
            "a row that is not the change's",
            next,
            Registered { id: b },
            Some(again.clone()),
        ),
        (
            "a relabel of a row not held",
            next,
            Relabelled { id: b },
            Some(other.clone()),
        ),
        (
            "a replacement of a row not held",
            next,
            Replaced {
                old: b,
                new: again.id,
            },
            Some(again),
        ),
        (
            "a replacement under another key",
            next,
            Replaced { old: a, new: b },
            Some(other.clone()),
        ),
        (
            "a relabel that carries another row",
            next,
            Relabelled { id: a },
            Some(WorkerRow {
                id: b,
                ..held.clone()
            }),
        ),
        (
            "a relabel under another key",
            next,
            Relabelled { id: a },
            Some(WorkerRow {
                facts: facts("http://b:8000", 1),
                ..held.clone()
            }),
        ),
        (
            "a replacement that carries another row",
            next,
            Replaced { old: a, new: b },
            Some(WorkerRow {
                id: identity("http://a:8000", 9),
                ..held.clone()
            }),
        ),
        (
            "a replacement by a held id",
            next,
            Replaced { old: a, new: c },
            Some(WorkerRow {
                id: c,
                ..held.clone()
            }),
        ),
        ("a removal of a row not held", next, Removed { id: b }, None),
        (
            "a removal that tells a row",
            next,
            Removed { id: a },
            Some(held),
        ),
        (
            "a status change of a row not held",
            next,
            StatusChanged {
                id: b,
                from: pending,
                to: ready,
            },
            None,
        ),
        (
            "a status change from another status",
            next,
            StatusChanged {
                id: a,
                from: ready,
                to: pending,
            },
            None,
        ),
    ];
    let stands = |standby: &MemoryStore| {
        let cut = standby.checkpoint();
        (cut.at, cut.epoch, Twin::of(&cut), all(&followed))
    };
    let before = stands(&standby);
    let at = Position {
        seq: before.0.seq + 1,
        term: 1,
    };
    for (misfit, epoch, change, row) in misfits {
        let adopting = followed.adopt(at).expect("the next position");
        standby.follow(at, epoch, change, row);
        assert!(!adopting.landed(), "{misfit}");
        drop(adopting);
        assert_eq!(stands(&standby), before, "{misfit}");
    }
    // The change that fits takes the place.
    let adopting = followed.adopt(at).expect("the next position");
    standby.follow(at, next, Registered { id: b }, Some(other));
    assert!(adopting.landed());
    assert_eq!(standby.epoch(), next);
}

#[test]
fn a_followed_row_keeps_the_instant_it_was_admitted_at() {
    let (standby, followed) = journaled(64);
    let at = |seq| Position { seq, term: 1 };
    let row = WorkerRow {
        id: identity("http://a:8000", 1),
        facts: facts("http://a:8000", 1),
        status: WorkerStatus::Pending,
        revision: 1,
        admitted_at: Instant::now(),
        maker: Default::default(),
    };
    let id = row.id;
    let adopting = followed.adopt(at(1)).expect("the next position");
    standby.follow_term(at(1));
    drop(adopting);
    let adopting = followed.adopt(at(2)).expect("the next position");
    standby.follow(
        at(2),
        1,
        MembershipChange::Registered { id },
        Some(row.clone()),
    );
    drop(adopting);
    // An event's row crosses with the instant of its apply: the instant
    // the row was admitted at stays.
    let relabelled = WorkerRow {
        revision: 2,
        admitted_at: row.admitted_at + Duration::from_secs(5),
        ..row.clone()
    };
    let adopting = followed.adopt(at(3)).expect("the next position");
    standby.follow(
        at(3),
        2,
        MembershipChange::Relabelled { id },
        Some(relabelled),
    );
    assert!(adopting.landed());
    let held = WorkersStore::get(&standby, id).expect("the row");
    assert_eq!((held.revision, held.admitted_at), (2, row.admitted_at));
}

#[test]
fn a_followed_row_keeps_the_maker_its_leader_wrote() {
    use crate::store::Registrant;

    let (standby, followed) = journaled(64);
    let at = |seq| Position { seq, term: 1 };
    let found = Registrant::provider("cluster", "pods/a").expect("a whole maker");
    let row = WorkerRow {
        id: identity("http://a:8000", 1),
        facts: facts("http://a:8000", 1),
        status: WorkerStatus::Pending,
        revision: 1,
        admitted_at: Instant::now(),
        maker: found.clone(),
    };
    let id = row.id;
    let adopting = followed.adopt(at(1)).expect("the next position");
    standby.follow_term(at(1));
    drop(adopting);
    let adopting = followed.adopt(at(2)).expect("the next position");
    standby.follow(
        at(2),
        1,
        MembershipChange::Registered { id },
        Some(row.clone()),
    );
    drop(adopting);
    let relabelled = WorkerRow { revision: 2, ..row };
    let adopting = followed.adopt(at(3)).expect("the next position");
    standby.follow(
        at(3),
        2,
        MembershipChange::Relabelled { id },
        Some(relabelled),
    );
    assert!(adopting.landed());
    let held = WorkersStore::get(&standby, id).expect("the row");
    assert_eq!((held.revision, held.maker), (2, found));
}

/// A prefill or a decode of the model on one runtime's gRPC side, which
/// names its KV transport or none.
fn pair_facts(key: &str, role: Role, transport: Option<&str>) -> WorkerFacts {
    WorkerFacts {
        role,
        wire_modes: WireModes::GRPC,
        engine_kind: "rt-a".to_owned(),
        pairing: transport.map(|transport| {
            Arc::new(PairingDescriptor {
                transport: Some(transport.to_owned()),
                ..PairingDescriptor::default()
            })
        }),
        ..facts(key, 1)
    }
}

/// The id and the pair class of every prefill and then every decode the
/// model's classes name, each as its published row carries it.
fn pair_rows(store: &MemoryStore) -> Vec<(WorkerId, u32)> {
    let world = store.snapshot();
    let Some(classes) = world.pairs("m") else {
        return Vec::new();
    };
    let named = classes.prefills().iter().chain(classes.decodes());
    named
        .filter_map(|&(id, _)| world.first_row_of(id))
        .map(|row| (row.id, row.pair_class))
        .collect()
}

#[test]
fn a_followed_relabel_carries_the_pair_classes_unless_the_offer_moved() {
    // Two prefills of two transports and three decodes, one of each
    // transport and one that names none: two classes.
    let fleet = [
        ("grpc://pa:9000", Role::Prefill, Some("tx-a")),
        ("grpc://pb:9000", Role::Prefill, Some("tx-b")),
        ("grpc://da:9000", Role::Decode, Some("tx-a")),
        ("grpc://db:9000", Role::Decode, Some("tx-b")),
        ("grpc://dn:9000", Role::Decode, None),
    ];
    let first = || pair_facts("grpc://pa:9000", Role::Prefill, Some("tx-a"));
    // Each fact that places a worker in its pools or makes its offer.
    let moved = [
        (
            "the descriptor",
            pair_facts("grpc://pa:9000", Role::Prefill, Some("tx-b")),
        ),
        (
            "the runtime",
            WorkerFacts {
                engine_kind: "rt-b".to_owned(),
                ..first()
            },
        ),
        (
            "the wires",
            WorkerFacts {
                wire_modes: WireModes::HTTP,
                ..first()
            },
        ),
        (
            "the role",
            WorkerFacts {
                role: Role::Decode,
                ..first()
            },
        ),
        (
            "the models",
            WorkerFacts {
                models: vec!["other".to_owned()],
                ..first()
            },
        ),
    ];
    for (what, facts) in moved {
        let (leader, led) = journaled(64);
        let (standby, followed) = journaled(64);
        leader.set_term(1);
        let mut told = 0;
        let mut follow = || {
            let entries = all(&led);
            for entry in entries.iter().skip(told) {
                follows(&standby, &followed, entry);
            }
            told = entries.len();
            // Each store's build is asked: the case reads at once.
            leader.publish_now();
            standby.publish_now();
        };
        for (key, role, transport) in fleet {
            leader.upsert(pair_facts(key, role, transport));
        }
        follow();
        let before = pair_rows(&standby);
        assert_eq!(before, pair_rows(&leader), "{what}");
        // The world is kept, so its classes stand where they are: one
        // address in a later world is those classes carried.
        let world = standby.snapshot();
        let derived = world.pairs("m").expect("the model's classes");
        // A row told again with a fact no pair reads moves no class.
        leader.upsert(WorkerFacts {
            zone: Some("z".to_owned()),
            ..first()
        });
        follow();
        let carried = standby.snapshot();
        let pairs = carried.pairs("m").expect("the model's classes");
        assert!(std::ptr::eq(derived, pairs), "{what}: derived again");
        assert_eq!(pair_rows(&standby), before, "{what}");
        // The offer moves: the standby derives its leader's classes.
        leader.upsert(facts);
        follow();
        assert_eq!(pair_rows(&standby), pair_rows(&leader), "{what}");
        let (ours, theirs) = (standby.snapshot(), leader.snapshot());
        assert_eq!(ours.pairs("m"), theirs.pairs("m"), "{what}");
        assert_ne!(pair_rows(&standby), before, "{what}: the classes moved");
    }
}

#[test]
fn a_standbys_pair_classes_follow_each_membership_write_and_a_restore() {
    let (leader, led) = journaled(64);
    let (standby, followed) = journaled(64);
    leader.set_term(1);
    let mut told = 0;
    // The standby follows what the leader wrote since, and reads the pair
    // classes its leader reads.
    let mut alike = |what: &str| {
        let entries = all(&led);
        for entry in entries.iter().skip(told) {
            follows(&standby, &followed, entry);
        }
        told = entries.len();
        // Each store's build is asked: the case reads at once.
        leader.publish_now();
        standby.publish_now();
        assert_eq!(pair_rows(&standby), pair_rows(&leader), "{what}");
        let (ours, theirs) = (standby.snapshot(), leader.snapshot());
        assert_eq!(ours.pairs("m"), theirs.pairs("m"), "{what}");
        pair_rows(&standby)
    };
    let prefill = |transport| pair_facts("grpc://pa:9000", Role::Prefill, Some(transport));
    leader.upsert(prefill("tx-a"));
    leader.upsert(pair_facts("grpc://pb:9000", Role::Prefill, Some("tx-b")));
    leader.upsert(pair_facts("grpc://da:9000", Role::Decode, Some("tx-a")));
    let pair = alike("a pair");
    // A member joins, one comes back as a new incarnation with another
    // offer, one leaves: each write moves the classes.
    leader.upsert(pair_facts("grpc://db:9000", Role::Decode, Some("tx-b")));
    let joined = alike("a member joined");
    assert_ne!(joined, pair);
    leader.upsert(WorkerFacts {
        incarnation: 2,
        ..prefill("tx-b")
    });
    let replaced = alike("a new incarnation");
    assert_ne!(replaced, joined);
    let left = leader.by_key("grpc://pb:9000").expect("the row");
    assert!(leader.remove(left.id, left.revision).is_some());
    let fewer = alike("a member left");
    assert_ne!(fewer, replaced);
    // A store that derived other classes takes the leader's cut: its
    // classes are those of the cut's membership.
    let (other, _log) = journaled(64);
    other.upsert(prefill("tx-a"));
    other.upsert(pair_facts("grpc://dz:9000", Role::Decode, Some("tx-a")));
    other.publish_now();
    assert_ne!(pair_rows(&other), fewer);
    assert!(other.restore_from(leader.checkpoint()));
    assert_eq!(pair_rows(&other), fewer);
    let (ours, theirs) = (other.snapshot(), leader.snapshot());
    assert_eq!(ours.pairs("m"), theirs.pairs("m"));
}

/// A document that validates, under the generation `number`.
fn generation(number: u64) -> AppliedSpec {
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = number;
    AppliedSpec::validate(&document).expect("a preset validates")
}

/// The generations a log holds, each at its place.
fn generations(log: &Log) -> Vec<(Position, u64)> {
    all(log)
        .iter()
        .filter_map(|entry| match entry.event {
            Event::Generation { generation } => Some((entry.at, generation)),
            _ => None,
        })
        .collect()
}

#[test]
fn a_generation_takes_a_place_under_a_term_and_a_take_tells_the_standing_one() {
    let at = |seq, term| Position { seq, term };
    // A store that runs no document tells none at its take.
    let (unbound, log) = journaled(16);
    unbound.set_term(1);
    assert_eq!(generations(&log), []);
    assert_eq!(log.head(), at(1, 1));

    let (store, log) = journaled(16);
    // Applied before any take: the log is told, and keeps no place for it.
    store.apply(generation(3)).expect("above the unbound one");
    assert_eq!(log.head(), Position::default());
    // The take tells the term's start, then the generation that stands.
    store.set_term(1);
    let start = Entry {
        at: at(1, 1),
        event: Event::Term,
    };
    assert_eq!(all(&log)[0], start);
    assert_eq!(generations(&log), [(at(2, 1), 3)]);
    // An apply under the term takes the next place, among the membership
    // writes, and it is one of the watched kinds.
    store.upsert(facts("grpc://a:9000", 1));
    store.apply(generation(4)).expect("above the standing one");
    assert_eq!(generations(&log), [(at(2, 1), 3), (at(4, 1), 4)]);
    assert_eq!(log.head(), at(4, 1));
    // A generation the store refuses tells nothing.
    store
        .apply(generation(4))
        .expect_err("not above the standing one");
    assert_eq!(log.head(), at(4, 1));
    // The next take tells the standing one again, under its own term.
    store.set_term(2);
    assert_eq!(generations(&log).last(), Some(&(at(6, 2), 4)));
}

#[test]
fn a_standby_follows_a_generation_at_its_leaders_place() {
    let (leader, led) = journaled(16);
    let (standby, log) = journaled(16);
    leader.set_term(1);
    leader.apply(generation(2)).expect("above the unbound one");
    leader.apply(generation(5)).expect("above the standing one");
    let entries = all(&led);
    let (start, told) = entries.split_first().expect("the term's start");
    log.join(start.at).expect("the leader's first place");
    standby.follow_term(start.at);
    for entry in told {
        let Event::Generation { generation: number } = entry.event else {
            panic!("the leader wrote no row: {entry:?}");
        };
        let adopting = log.adopt(entry.at).expect("the next position");
        standby
            .follow_generation(entry.at, generation(number))
            .expect("above its own");
        assert!(adopting.landed(), "{entry:?}");
    }
    assert_eq!(all(&log), entries, "the leader's history, place by place");
    assert_eq!(standby.snapshot().spec().generation(), 5);
    assert_eq!(log.followed(), Some(1), "it leads nothing by it");

    // A generation the store refuses tells nothing: its guard does not
    // land, and the standby starts again from a snapshot.
    let next = Position {
        seq: log.head().seq + 1,
        term: 1,
    };
    let adopting = log.adopt(next).expect("the next position");
    standby
        .follow_generation(next, generation(5))
        .expect_err("not above its own");
    assert!(!adopting.landed());
    // An event that changes nothing journals its place alone.
    let adopting = log.adopt(next).expect("the next position");
    standby.follow_generation_place(next, 5);
    assert!(adopting.landed());
    assert_eq!(generations(&log).last(), Some(&(next, 5)));
    assert_eq!(standby.snapshot().spec().generation(), 5);
}
