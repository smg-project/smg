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

use super::{Entry, Event, Log, Lost};
use crate::{
    lease::TransportClass,
    store::{
        identity, Cut, GatewayId, HealthSink, Journal, Kinds, LeasesStore, LoadSample, LoadSink,
        LoadSource, MembershipChange, MemoryStore, Position, WorkerFacts, WorkerPatch, WorkerRow,
        WorkerStatus, WorkersStore,
    },
    view::{LoadRow, WorkerId},
};

/// A store whose log carries every kind, `events` of each retained.
fn journaled(events: usize) -> (MemoryStore, Arc<Log>) {
    let log = Arc::new(Log::new(events, events));
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
        log.read(Kinds::All, after, 64, &mut out)
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
        Event::Load { .. } | Event::Term => None,
    }
}

/// A load event's worker, rank and sample.
fn load(entry: &Entry) -> (WorkerId, u16, LoadSample) {
    match &entry.event {
        Event::Load { id, rank, sample } => (*id, *rank, *sample),
        other => panic!("a load event, not {other:?}"),
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
        store.probe_outcome(b, false, revision(&store, b));
    }
    store.connected(b, revision(&store, b));
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
            at(6, 2),
            at(7, 2),
            at(8, 5),
            at(9, 5),
            at(10, 5),
            at(11, 5),
            at(12, 5)
        ]
    );
    assert_eq!(log.head(Kinds::All), at(12, 5));
    // Each term starts with its own position: the store took it.
    assert_eq!(entries[0].event, Event::Term);
    assert_eq!(entries[7].event, Event::Term);

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

    let (id, rank, sample) = load(&entries[3]);
    assert_eq!(
        (id, rank, sample.row.running, sample.source),
        (a, 0, 3, LoadSource::Stream)
    );
    let (id, rank, sample) = load(&entries[4]);
    assert_eq!(
        (id, rank, sample.row.running, sample.source),
        (a, 1, 5, LoadSource::Poll)
    );

    let (change, row, epoch) = member(&entries[5]);
    assert_eq!(change, MembershipChange::Relabelled { id: a });
    assert_eq!((row.facts.priority, epoch), (7, 3));

    let (change, row, epoch) = member(&entries[6]);
    assert_eq!(change, MembershipChange::Resized { id: a });
    assert_eq!((row.facts.dp_size, epoch), (2, 4));

    let (change, row, epoch) = member(&entries[8]);
    assert_eq!(change, MembershipChange::Replaced { old: a, new: b });
    assert_eq!(
        (row.id, row.facts.incarnation, row.revision, epoch),
        (b, 2, 1, 5)
    );
    assert_eq!(
        left(&entries[8]),
        Some(&replaced),
        "a replacement tells the incarnation it took out"
    );
    assert!(
        entries[..8].iter().all(|entry| left(entry).is_none()),
        "no other write took a row out"
    );

    let (change, row, _) = member(&entries[9]);
    assert_eq!(
        change,
        MembershipChange::StatusChanged {
            id: b,
            from: WorkerStatus::Ready,
            to: WorkerStatus::NotReady
        }
    );
    assert_eq!(row.status, WorkerStatus::NotReady);

    let (change, row, _) = member(&entries[10]);
    assert_eq!(
        change,
        MembershipChange::StatusChanged {
            id: b,
            from: WorkerStatus::NotReady,
            to: WorkerStatus::Ready
        }
    );
    assert_eq!(row.status, WorkerStatus::Ready);

    match &entries[11].event {
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
        log.head(Kinds::All),
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
        Log::new(64, 64).on_watched(move |_| {
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
        store.probe_outcome(b, false, revision(&store, b));
    }
    tells("a probe's outcome", 1);
    store.connected(b, revision(&store, b));
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
        log.head(Kinds::All),
        Position { seq: 1, term: 3 },
        "the take is the term's first position"
    );
    store.upsert(facts("http://a:8000", 1));
    assert_eq!(log.head(Kinds::All), Position { seq: 2, term: 3 });
}

/// A journal that keeps the terms it is told and counts the load rows.
#[derive(Default)]
struct Told {
    terms: Mutex<Vec<u64>>,
    loads: AtomicU64,
    carries_loads: bool,
}

impl Journal for Told {
    fn member(&self, _: u64, _: MembershipChange, _: Option<&WorkerRow>, _: Option<&WorkerRow>) {}

    fn load(&self, _: WorkerId, _: u16, _: &LoadSample) {
        self.loads.fetch_add(1, Ordering::SeqCst);
    }

    fn term_rose(&self, term: u64) {
        self.terms.lock().push(term);
    }

    fn carries_loads(&self) -> bool {
        self.carries_loads
    }

    fn head(&self, _: Kinds) -> Position {
        Position::default()
    }

    fn followed(&self) -> Option<u64> {
        None
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
fn a_store_tells_no_load_row_to_a_journal_that_carries_none() {
    for carries_loads in [false, true] {
        let told = Arc::new(Told {
            carries_loads,
            ..Told::default()
        });
        let store = MemoryStore::journaled(Arc::clone(&told) as _);
        let a = store.upsert(facts("http://a:8000", 1)).id();
        let now = Instant::now();
        store.record(a, 0, running(1), now, LoadSource::Stream);
        store.report(&[(a, 1, Some(running(2)))], now, LoadSource::Poll);
        assert_eq!(
            told.loads.load(Ordering::SeqCst),
            if carries_loads { 2 } else { 0 }
        );
    }
    // A log of the watched kinds alone: a load write takes no position
    // and the store holds the row all the same.
    let log = Arc::new(Log::new(8, 0));
    let store = MemoryStore::journaled(Arc::clone(&log) as _);
    let a = store.upsert(facts("http://a:8000", 1)).id();
    store.record(a, 0, running(1), Instant::now(), LoadSource::Stream);
    assert_eq!(
        log.head(Kinds::All),
        Position { seq: 1, term: 0 },
        "the registration alone"
    );
    assert_eq!(store.cut(Kinds::All).loads.len(), 1);
}

#[test]
fn a_store_with_no_journal_cuts_at_the_start_and_a_journaled_one_at_its_head() {
    let plain = MemoryStore::new();
    plain.upsert(facts("http://a:8000", 1));
    let cut = plain.cut(Kinds::All);
    assert_eq!((cut.at, cut.rows.len()), (Position::default(), 1));

    let (store, log) = journaled(8);
    store.set_term(4);
    let a = store.upsert(facts("http://b:8000", 1)).id();
    store.upsert(facts("http://a:8000", 1));
    store.record(a, 1, running(2), Instant::now(), LoadSource::Stream);
    let cut = store.cut(Kinds::All);
    assert_eq!(cut.at, log.head(Kinds::All));
    assert_eq!(cut.at, Position { seq: 4, term: 4 });
    assert_eq!(cut.epoch, store.epoch());
    let keys: Vec<&str> = cut.rows.iter().map(|row| row.facts.key.as_str()).collect();
    assert_eq!(keys, ["http://a:8000", "http://b:8000"], "rows by key");
    let loads: Vec<(WorkerId, u16, u32)> = cut
        .loads
        .iter()
        .map(|(id, rank, sample)| (*id, *rank, sample.row.running))
        .collect();
    assert_eq!(loads, [(a, 1, 2)]);

    // A cut of the rows: the same rows at the last write of the membership
    // kinds, not at the load row after it.
    let rows = store.cut(Kinds::Watched);
    assert_eq!(rows.at, log.head(Kinds::Watched));
    assert_eq!(rows.at, Position { seq: 3, term: 4 });
    assert_eq!((rows.epoch, &rows.rows), (cut.epoch, &cut.rows));
    assert!(rows.loads.is_empty());
}

/// What a standby holds after a cut and the events it applied since: the
/// rows and the load rows, under the store's own rules (a load of no member
/// is dropped; a row that leaves takes its load rows).
#[derive(Debug, PartialEq)]
struct Twin {
    rows: BTreeMap<WorkerId, WorkerRow>,
    loads: BTreeMap<(WorkerId, u16), LoadSample>,
}

impl Twin {
    fn of(cut: &Cut) -> Self {
        Self {
            rows: cut.rows.iter().map(|row| (row.id, row.clone())).collect(),
            loads: cut
                .loads
                .iter()
                .map(|(id, rank, sample)| ((*id, *rank), *sample))
                .collect(),
        }
    }

    fn leave(&mut self, id: WorkerId) {
        self.rows.remove(&id);
        self.loads.retain(|(of, _), _| *of != id);
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
            Event::Load { id, rank, sample } => {
                if self.rows.contains_key(id) {
                    self.loads.insert((*id, *rank), *sample);
                }
            }
            Event::Term => {}
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

/// One load row for the worker each key holds. With no churn beside it,
/// the row of every worker a key holds is journaled.
fn load_each(store: &MemoryStore, lives: &[AtomicU64]) {
    let now = Instant::now();
    for (index, life) in lives.iter().enumerate() {
        let id = identity(&key(index), life.load(Ordering::SeqCst));
        store.record(id, 0, running(0), now, LoadSource::Stream);
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
    // A standby's cuts (every kind) and a watcher's (the rows), in turn.
    let mut cuts = vec![store.cut(Kinds::All)];
    let mut row_cuts = vec![store.cut(Kinds::Watched)];
    // The writers' work in legs. While a leg runs, a cut is taken each time
    // the log has moved past the last one, a bounded number of them. After
    // the leg each key's worker gets one load row and one more cut is
    // taken: cuts stand between writes of both kinds whatever the
    // scheduler does.
    for _ in 0..LEGS {
        thread::scope(|scope| {
            let writers = [
                scope.spawn(|| churn(&store, &lives, ROUNDS)),
                scope.spawn(|| loads(&store, &lives, 11, WRITES)),
                scope.spawn(|| loads(&store, &lives, 13, WRITES)),
            ];
            let mut taken = 0;
            while taken < CUTS && writers.iter().any(|writer| !writer.is_finished()) {
                if cuts
                    .last()
                    .is_some_and(|cut| cut.at == log.head(Kinds::All))
                {
                    thread::yield_now();
                    continue;
                }
                cuts.push(store.cut(Kinds::All));
                row_cuts.push(store.cut(Kinds::Watched));
                taken += 1;
            }
        });
        load_each(&store, &lives);
        cuts.push(store.cut(Kinds::All));
        row_cuts.push(store.cut(Kinds::Watched));
    }

    let entries = all(&log);
    let place = |cut: &Cut| usize::try_from(cut.at.seq).expect("a count of events");
    let mut out = Vec::new();
    let (mut members, mut load_rows, mut moved) = (0u32, 0u32, 0u32);
    for pair in cuts.windows(2) {
        let (from, to) = (&pair[0], &pair[1]);
        let mut twin = Twin::of(from);
        let after = &entries[place(from)..place(to)];
        for entry in after {
            match entry.event {
                Event::Member { .. } => members += 1,
                Event::Load { .. } => load_rows += 1,
                Event::Term => {}
            }
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
        // The cut's load rows stand by worker and rank.
        assert!(
            to.loads
                .windows(2)
                .all(|pair| (pair[0].0, pair[0].1) < (pair[1].0, pair[1].1)),
            "{:?}",
            to.at
        );
        // The cut's place is a place of the log: the tail is read from it.
        assert_eq!(
            log.read(Kinds::All, to.at, 1, &mut out),
            Ok(()),
            "{:?}",
            to.at
        );
        assert_eq!(
            out.first().map(|entry| entry.at.seq),
            (to.at != log.head(Kinds::All)).then_some(to.at.seq + 1)
        );
    }

    // A watcher's cuts: the rows at the last write of the watched kinds,
    // and the membership events after it are the next cut's rows.
    let mut rows_moved = 0u32;
    for pair in row_cuts.windows(2) {
        let (from, to) = (&pair[0], &pair[1]);
        let mut twin = Twin::of(from);
        let after = &entries[place(from)..place(to)];
        for entry in after {
            if matches!(entry.event, Event::Member { .. }) {
                twin.apply(&entry.event);
            }
        }
        rows_moved += u32::from(!after.is_empty());
        assert_eq!(
            twin,
            Twin::of(to),
            "the rows at {:?} and the membership events after them are not the rows at {:?}",
            from.at,
            to.at
        );
        assert!(to.loads.is_empty(), "a cut of the rows carries no load row");
        // It stands at a write of the watched kinds, and the log reads
        // those kinds on from it.
        let last = place(to)
            .checked_sub(1)
            .and_then(|index| entries.get(index));
        assert!(
            !last.is_some_and(|entry| matches!(entry.event, Event::Load { .. })),
            "{:?}",
            to.at
        );
        assert_eq!(
            log.read(Kinds::Watched, to.at, 1, &mut out),
            Ok(()),
            "{:?}",
            to.at
        );
    }

    // The case is held to what it claims, by its legs and not by the
    // scheduler: every leg wrote both kinds, and a cut stands after each.
    assert_eq!(u64::from(members + load_rows), log.head(Kinds::All).seq);
    assert!(members >= LEGS * ROUNDS / 2, "{members} membership events");
    assert!(
        load_rows >= LEGS,
        "{load_rows} load rows written to a member"
    );
    assert!(moved >= LEGS, "{moved} cuts with a tail, of {}", cuts.len());
    assert!(
        rows_moved >= LEGS,
        "{rows_moved} cuts of the rows with a tail"
    );
}

#[test]
fn a_reader_of_a_restarted_store_starts_at_its_cut_and_reads_on() {
    let (store, log) = journaled(64);
    let mut out = Vec::new();
    // The log restarts at a leader's place, as a restore of the store has
    // it, and the store writes on under no term. A reader starts from a
    // cut, which stands at the store's own write, and reads what followed:
    // the cut with that tail is the next cut.
    log.restart(Position { seq: 2, term: 2 });
    store.upsert(facts("http://a:8000", 1));
    let first = store.cut(Kinds::All);
    assert_eq!(first.at, Position { seq: 3, term: 0 });
    assert_eq!(log.read(Kinds::All, first.at, 64, &mut out), Ok(()));
    assert!(out.is_empty());
    let b = store.upsert(facts("http://b:8000", 1)).id();
    store.record(b, 0, running(2), Instant::now(), LoadSource::Stream);
    assert_eq!(log.read(Kinds::All, first.at, 64, &mut out), Ok(()));
    let mut twin = Twin::of(&first);
    for entry in &out {
        twin.apply(&entry.event);
    }
    let next = store.cut(Kinds::All);
    assert_eq!(out.last().map(|entry| entry.at), Some(next.at));
    assert_eq!(twin, Twin::of(&next));
    // A restart below those numbers: the first reader's place is gone for
    // good, and the cut after the next write stands past it.
    log.restart(Position { seq: 2, term: 3 });
    store.upsert(facts("http://c:8000", 1));
    assert_eq!(log.read(Kinds::All, first.at, 64, &mut out), Err(Lost));
    let again = store.cut(Kinds::All);
    assert_eq!(again.at, Position { seq: 6, term: 0 });
    assert_eq!(log.read(Kinds::All, again.at, 64, &mut out), Ok(()));
    assert!(out.is_empty());
}

#[test]
fn a_workers_load_rows_stand_between_its_registration_and_its_leaving() {
    const ROUNDS: u32 = 4_000;
    const WRITES: u32 = 40_000;
    let (store, log) = journaled(1 << 17);
    let lives: Vec<AtomicU64> = (0..KEYS).map(|_| AtomicU64::new(0)).collect();
    thread::scope(|scope| {
        scope.spawn(|| churn(&store, &lives, ROUNDS));
        scope.spawn(|| loads(&store, &lives, 17, WRITES));
        scope.spawn(|| loads(&store, &lives, 19, WRITES));
    });
    // One row more for each key's worker, the churn ended: the journal
    // holds load rows whatever the scheduler did.
    load_each(&store, &lives);
    // Each worker by the journal: live from its registration, gone from its
    // removal or its replacement.
    let mut live = BTreeMap::new();
    let mut load_rows = 0u32;
    for entry in all(&log) {
        match entry.event {
            Event::Member { change, .. } => match change {
                MembershipChange::Registered { id } => {
                    live.insert(id, true);
                }
                MembershipChange::Replaced { old, new } => {
                    live.insert(old, false);
                    live.insert(new, true);
                }
                MembershipChange::Removed { id } => {
                    live.insert(id, false);
                }
                _ => {}
            },
            Event::Load { id, .. } => {
                load_rows += 1;
                match live.get(&id) {
                    Some(true) => {}
                    Some(false) => {
                        panic!("the load at {:?} stands after its worker left", entry.at)
                    }
                    None => panic!(
                        "the load at {:?} stands before its worker's registration",
                        entry.at
                    ),
                }
            }
            Event::Term => {}
        }
    }
    assert!(load_rows > 0, "no load row was written to a member");
    let gone = live.values().filter(|live| !**live).count();
    assert!(gone > 0, "no worker left under the load writers");
}
