//! The build a membership write owes: who pays it and when, what the ask
//! does, and what stays in force at a write's own answer.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

use super::{
    ask, facts, load, member, running_of, stalled, status, view, Classes, ViewBuffers, ASKER,
    MODEL, PUBLISH_WINDOW,
};
use crate::{
    store::{
        HealthSink, LoadSink, LoadSource, MemoryStore, WorkerFacts, WorkerStatus, WorkersStore,
    },
    view::{Role, WorkerId},
};

fn worker(index: usize) -> WorkerFacts {
    facts(&format!("http://w{index}:8000"), Role::Single, &[MODEL])
}

/// The workers a decision at `now` may be handed: the pool's routable rows.
fn handed(store: &MemoryStore, now: Instant) -> Vec<WorkerId> {
    let (rows, _, _) = view(store, &ask(now));
    rows.iter()
        .filter(|row| row.health.routable())
        .map(|row| row.id)
        .collect()
}

fn revision(store: &MemoryStore, id: WorkerId) -> u64 {
    WorkersStore::get(store, id).expect("the row").revision
}

#[test]
fn a_registration_owes_its_build_and_the_ask_pays_it_at_once() {
    let store = MemoryStore::new();
    let id = store.upsert(worker(0)).id();
    // The answer is back and the row is on file. No build was paid, so no
    // decision reads the row yet, and the store names when one will.
    assert!(WorkersStore::get(&store, id).is_some());
    let mut out = ViewBuffers::default();
    let now = Instant::now();
    assert!(store
        .build_view(MODEL, Role::Single, &ask(now), &mut out)
        .is_none());
    assert_eq!(store.membership_builds(), 0);
    assert!(store.publish_due().is_some());
    store.publish_now();
    assert_eq!(view(&store, &ask(now)).0.len(), 1);
    assert_eq!((store.membership_builds(), store.publish_due()), (1, None));
    // Nothing is owed: the ask does nothing.
    store.publish_now();
    assert_eq!(store.membership_builds(), 1);
}

#[test]
fn the_writes_of_one_window_are_one_build_at_the_instant_the_store_names() {
    let store = MemoryStore::new();
    store.upsert(worker(0));
    // No window is open: the first build is due at once, and opens one.
    let first = store.publish_due().expect("a build is owed");
    store.publish(first);
    assert_eq!(store.membership_builds(), 1);
    for index in 1..64 {
        store.upsert(worker(index));
    }
    let end = store.publish_due().expect("a build is owed");
    assert!(end >= first + PUBLISH_WINDOW);
    // A call before the instant pays nothing: a storm builds once a window.
    store.publish(end - Duration::from_nanos(1));
    assert_eq!(store.membership_builds(), 1);
    assert_eq!(view(&store, &ask(end)).0.len(), 1);
    store.publish(end);
    assert_eq!(store.membership_builds(), 2);
    assert_eq!(view(&store, &ask(end)).0.len(), 64);
    assert_eq!(store.publish_due(), None);
}

#[test]
fn a_paid_build_opens_the_window_and_an_asked_one_leaves_it_as_it_stands() {
    let store = MemoryStore::new();
    let id = member(&store, "http://a:8000");
    // The ask opened no window: the first record is read at once.
    let t0 = Instant::now();
    store.record(id, 0, load(1, 0), t0, LoadSource::Stream);
    assert_eq!(running_of(&store, id, t0), Some(1));
    // The host pays the next build at the instant the store names. A record
    // inside the window that build opened waits for its end.
    store.upsert(worker(1));
    let due = store.publish_due().expect("a build is owed");
    store.publish(due);
    store.record(id, 0, load(2, 0), due, LoadSource::Stream);
    assert_eq!(running_of(&store, id, due), Some(1));
    store.publish(due + PUBLISH_WINDOW);
    assert_eq!(running_of(&store, id, due), Some(2));
}

#[test]
fn a_record_waits_while_a_build_is_owed_and_is_read_with_it() {
    let store = MemoryStore::new();
    let id = member(&store, "http://a:8000");
    let t0 = Instant::now();
    store.upsert(worker(1));
    // No window is open and the record still waits: the build shows it.
    store.record(id, 0, load(3, 0), t0, LoadSource::Stream);
    assert_eq!(running_of(&store, id, t0), None);
    store.publish_now();
    assert_eq!(running_of(&store, id, t0), Some(3));
}

#[test]
fn a_status_and_a_veto_are_read_at_their_answer_while_a_build_is_owed() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    store.upsert(worker(1));
    status(&store, id, WorkerStatus::Ready);
    assert_eq!(handed(&store, t0), [id]);
    store.gateway_veto(ASKER, id, Some(t0 + Duration::from_secs(1)));
    assert_eq!(stalled(&store, ASKER, t0), [id]);
    // Neither paid the build: the member's own ask made the one.
    assert_eq!(store.membership_builds(), 1);
    assert_eq!(view(&store, &ask(t0)).0.len(), 1);
}

#[test]
fn a_veto_past_its_end_leaves_at_the_build_a_windows_end_pays() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let before = t0 - Duration::from_millis(1);
    let ended = member(&store, "http://a:8000");
    store.gateway_veto(ASKER, ended, Some(t0));
    store.upsert(worker(1));
    assert_eq!(stalled(&store, ASKER, before), [ended]);
    // The host pays the build at or past the veto's end. The build is a
    // window's publication: a decision dated before that end reads it, and
    // it no longer holds the entry.
    let due = store.publish_due().expect("a build is owed");
    store.publish(due);
    assert!(stalled(&store, ASKER, before).is_empty());
    // The sink's own answer keeps its compare.
    assert!(store.vetoed(ASKER, ended, before));
}

#[test]
fn a_leave_is_in_force_at_its_answer_and_its_row_leaves_at_the_build() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let stays = member(&store, "http://a:8000");
    let leaves = member(&store, "http://b:8000");
    for id in [stays, leaves] {
        status(&store, id, WorkerStatus::Ready);
    }
    assert_eq!(handed(&store, t0).len(), 2);
    store
        .remove(leaves, revision(&store, leaves))
        .expect("removed");
    // No build was paid. The pool still lists the row and no decision is
    // handed it: a leave does not wait for the window.
    assert_eq!(view(&store, &ask(t0)).0.len(), 2);
    assert_eq!(handed(&store, t0), [stays]);
    store.publish_now();
    assert_eq!(view(&store, &ask(t0)).0.len(), 1);
}

#[test]
fn a_replaced_incarnation_leaves_at_the_answer_and_the_new_one_joins_at_the_build() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let old = member(&store, "http://a:8000");
    status(&store, old, WorkerStatus::Ready);
    let new = store
        .upsert(WorkerFacts {
            incarnation: 2,
            ..facts("http://a:8000", Role::Single, &[MODEL])
        })
        .id();
    assert_ne!(new, old);
    // The old life is handed to no decision, and the new one is not built.
    let (rows, _, _) = view(&store, &ask(t0));
    assert_eq!(rows.iter().map(|row| row.id).collect::<Vec<_>>(), [old]);
    assert!(handed(&store, t0).is_empty());
    store.publish_now();
    let (rows, _, _) = view(&store, &ask(t0));
    assert_eq!(rows.iter().map(|row| row.id).collect::<Vec<_>>(), [new]);
}

#[test]
fn no_decision_is_handed_a_worker_after_its_leaves_answer_under_a_storm() {
    const WORKERS: usize = 256;
    // A window far shorter than the storm: its builds land between leaves.
    let store = MemoryStore::with_publish_window(Duration::from_micros(200));
    let ids: Vec<WorkerId> = (0..WORKERS)
        .map(|index| {
            let id = store.upsert(worker(index)).id();
            store.publish_now();
            status(&store, id, WorkerStatus::Ready);
            id
        })
        .collect();
    let index_of: HashMap<WorkerId, usize> = ids.iter().copied().zip(0..).collect();
    let gone: Vec<AtomicBool> = (0..WORKERS).map(|_| AtomicBool::new(false)).collect();
    let done = AtomicBool::new(false);
    thread::scope(|scope| {
        // The storm: every other worker leaves, by its removal or by a new
        // life under its key, and a new worker registers after each removal.
        scope.spawn(|| {
            for index in (0..WORKERS).step_by(2) {
                if index % 4 == 0 {
                    store
                        .remove(ids[index], revision(&store, ids[index]))
                        .expect("removed");
                    gone[index].store(true, Ordering::SeqCst);
                    store.upsert(worker(WORKERS + index));
                } else {
                    store.upsert(WorkerFacts {
                        incarnation: 2,
                        ..worker(index)
                    });
                    gone[index].store(true, Ordering::SeqCst);
                }
            }
            done.store(true, Ordering::SeqCst);
        });
        // The host, as the driver's loop: it pays what is due.
        scope.spawn(|| {
            while !done.load(Ordering::SeqCst) {
                let now = Instant::now();
                if store.publish_due().is_some_and(|due| now >= due) {
                    store.publish(now);
                }
                thread::yield_now();
            }
        });
        for _ in 0..2 {
            scope.spawn(|| {
                let mut out = ViewBuffers::default();
                let mut left = vec![false; WORKERS];
                while !done.load(Ordering::SeqCst) {
                    // The leaves answered before this decision began.
                    for (left, gone) in left.iter_mut().zip(&gone) {
                        *left = gone.load(Ordering::SeqCst);
                    }
                    store
                        .build_view(MODEL, Role::Single, &ask(Instant::now()), &mut out)
                        .expect("the pool");
                    for row in out.candidates.iter().filter(|row| row.health.routable()) {
                        if let Some(&index) = index_of.get(&row.id) {
                            assert!(!left[index], "worker {index}: handed after its leave");
                        }
                    }
                }
            });
        }
    });
    // Every write is in the last build: the workers that stayed, the new
    // lives, which keep the status of the old, and the new workers, which
    // are pending.
    store.publish_now();
    let now = Instant::now();
    assert_eq!(view(&store, &ask(now)).0.len(), WORKERS);
    assert_eq!(handed(&store, now).len(), WORKERS / 2 + WORKERS / 4);
}

#[test]
fn the_writes_one_build_pays_ask_the_pair_classes_once() {
    let (a, b, c) = (WorkerId(1), WorkerId(2), WorkerId(3));
    let written = |left, joined| Classes::Written { left, joined };
    // A write that moved no offer changes nothing of what another asks.
    let joins = written(None, Some(a));
    assert_eq!(Classes::Carry.then(joins), joins);
    assert_eq!(joins.then(Classes::Carry), joins);
    // One worker's writes stay one worker's: what the last build held of
    // it, and what stands of it now.
    let moved = written(Some(a), Some(a));
    assert_eq!(moved.then(moved), moved);
    assert_eq!(
        written(Some(a), Some(b)).then(written(Some(b), Some(c))),
        written(Some(a), Some(c))
    );
    assert_eq!(
        joins.then(written(Some(a), Some(b))),
        written(None, Some(b))
    );
    assert_eq!(
        written(Some(a), Some(b)).then(written(Some(b), None)),
        written(Some(a), None)
    );
    // A worker that joined and left between two builds was in neither.
    assert_eq!(joins.then(written(Some(a), None)), Classes::Carry);
    // Two workers' writes derive whole, and so does every write after.
    assert_eq!(joins.then(written(None, Some(b))), Classes::Derive);
    assert_eq!(written(Some(a), None).then(joins), Classes::Derive);
    assert_eq!(Classes::Derive.then(Classes::Carry), Classes::Derive);
    assert_eq!(Classes::Derive.then(joins), Classes::Derive);
}
