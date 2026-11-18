//! The build a membership write owes: who pays it and when, what the ask
//! does, and what stays in force at a write's own answer.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

use super::{
    ask, block, facts, generation_with, load, member, overloaded, pressure, running_of, stalled,
    status, view, Classes, ViewBuffers, ASKER, MODEL, PUBLISH_WINDOW,
};
use crate::{
    store::{
        HealthSink, Kinds, LoadSink, LoadSource, MemoryStore, WorkerFacts, WorkerStatus,
        WorkersStore,
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
fn a_removal_pays_its_build_before_it_answers() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let stays = member(&store, "http://a:8000");
    let leaves = member(&store, "http://b:8000");
    for id in [stays, leaves] {
        status(&store, id, WorkerStatus::Ready);
    }
    assert_eq!(handed(&store, t0).len(), 2);
    let builds = store.membership_builds();
    store
        .remove(leaves, revision(&store, leaves))
        .expect("removed");
    // The removal built before it answered. The pool lists the row no more,
    // whatever a pass reads of a row's health, and nothing is left owed.
    let (rows, _, _) = view(&store, &ask(t0));
    assert_eq!(rows.iter().map(|row| row.id).collect::<Vec<_>>(), [stays]);
    assert_eq!(
        (store.membership_builds(), store.publish_due()),
        (builds + 1, None)
    );
}

#[test]
fn a_replaced_incarnation_pays_its_build_before_it_answers() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let old = member(&store, "http://a:8000");
    status(&store, old, WorkerStatus::Ready);
    let builds = store.membership_builds();
    let new = store
        .upsert(WorkerFacts {
            incarnation: 2,
            ..facts("http://a:8000", Role::Single, &[MODEL])
        })
        .id();
    assert_ne!(new, old);
    // The write built before it answered: the old life's row is in no pool,
    // the new life stands in its place with the status it inherits, and
    // nothing is left owed.
    assert_eq!(handed(&store, t0), [new]);
    assert_eq!(
        (store.membership_builds(), store.publish_due()),
        (builds + 1, None)
    );
}

/// A standby's follow door owes and pays as its leader's writes do: a
/// followed registration and a followed refresh leave their build owed,
/// the registration's row in no view before it, and a followed removal and
/// a followed new life pay their own build before the door returns.
#[test]
fn a_followed_join_owes_its_build_and_a_followed_leave_pays_its_own() {
    let leader = MemoryStore::new();
    let a = member(&leader, "http://a:8000");
    let b = member(&leader, "http://b:8000");
    let standby = MemoryStore::new();
    assert!(standby.restore_from(leader.cut(Kinds::All)));
    let now = Instant::now();
    let listed = |store: &MemoryStore| -> Vec<WorkerId> {
        let (rows, _, _) = view(store, &ask(now));
        rows.iter().map(|row| row.id).collect()
    };
    // A followed registration: on file, owed, in no view until the build.
    let joins = leader.upsert(worker(2));
    let row = WorkersStore::get(&leader, joins.id());
    standby.follow(WorkersStore::epoch(&leader), joins, row);
    assert_eq!(listed(&standby).len(), 2);
    assert!(standby.publish_due().is_some());
    standby.publish_now();
    assert_eq!(listed(&standby).len(), 3);
    // A followed refresh keeps the window as the registration does, and
    // the leave after it pays what it left owed.
    let refresh = leader.upsert(WorkerFacts {
        zone: Some("z".to_owned()),
        ..worker(2)
    });
    let row = WorkersStore::get(&leader, refresh.id());
    standby.follow(WorkersStore::epoch(&leader), refresh, row);
    assert!(standby.publish_due().is_some());
    // A followed removal: the row is in no view when the door returns.
    let leaves = leader.remove(a, revision(&leader, a)).expect("removed");
    standby.follow(WorkersStore::epoch(&leader), leaves, None);
    assert!(!listed(&standby).contains(&a));
    assert_eq!(standby.publish_due(), None);
    // A followed new life: it stands in the old one's place at once.
    let life = WorkerFacts {
        incarnation: 2,
        ..facts("http://b:8000", Role::Single, &[MODEL])
    };
    let replaces = leader.upsert(life);
    let new = replaces.id();
    let row = WorkersStore::get(&leader, new);
    standby.follow(WorkersStore::epoch(&leader), replaces, row);
    let rows = listed(&standby);
    assert!(rows.contains(&new) && !rows.contains(&b));
    assert_eq!(standby.publish_due(), None);
}

/// A restore builds at once, whatever the window: the cut's rows are in a
/// view when it returns, over a build the replica's own write left owed,
/// and nothing is left owed.
#[test]
fn a_restore_builds_its_world_at_once_and_leaves_nothing_owed() {
    let leader = MemoryStore::new();
    let mut cut = [
        member(&leader, "http://a:8000"),
        member(&leader, "http://b:8000"),
    ];
    cut.sort_unstable();
    let standby = MemoryStore::new();
    // A row of the replica's own, its build still owed.
    standby.upsert(worker(9));
    assert!(standby.publish_due().is_some());
    assert!(standby.restore_from(leader.cut(Kinds::All)));
    let (rows, _, _) = view(&standby, &ask(Instant::now()));
    let mut listed: Vec<WorkerId> = rows.iter().map(|row| row.id).collect();
    listed.sort_unstable();
    assert_eq!(listed, cut);
    assert_eq!(standby.publish_due(), None);
}

#[test]
fn a_worker_that_returns_under_its_id_inside_the_window_stays_out_of_the_pools_it_left() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let stays = member(&store, "http://a:8000");
    let leaves = member(&store, "http://b:8000");
    for id in [stays, leaves] {
        status(&store, id, WorkerStatus::Ready);
    }
    store
        .remove(leaves, revision(&store, leaves))
        .expect("removed");
    // The same key and incarnation, registered for another model: the same
    // id, told ready before the build its return owes.
    let back = store
        .upsert(facts("http://b:8000", Role::Single, &["other"]))
        .id();
    assert_eq!(back, leaves);
    status(&store, back, WorkerStatus::Ready);
    // The status is the new registration's. The pool it left lists no row
    // of it since the leave's own build, and the next build reads the
    // status with the registration.
    assert_eq!(handed(&store, t0), [stays]);
    store.publish_now();
    assert_eq!(handed(&store, t0), [stays]);
    let mut out = ViewBuffers::default();
    store
        .build_view("other", Role::Single, &ask(t0), &mut out)
        .expect("the pool it registered for");
    let ready = out.candidates.iter().filter(|row| row.health.routable());
    assert_eq!(ready.map(|row| row.id).collect::<Vec<_>>(), [back]);
}

#[test]
fn no_veto_on_a_worker_that_left_stands_over_the_one_that_returns_under_its_id() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    status(&store, id, WorkerStatus::Ready);
    store.gateway_veto(ASKER, id, Some(t0 + Duration::from_secs(1)));
    assert_eq!(stalled(&store, ASKER, t0), [id]);
    store.remove(id, revision(&store, id)).expect("removed");
    let back = store
        .upsert(facts("http://a:8000", Role::Single, &[MODEL]))
        .id();
    assert_eq!(back, id);
    status(&store, back, WorkerStatus::Ready);
    store.publish_now();
    // The veto left with the worker it was on, at the leave's answer: the
    // build finds none to put on the one that returned.
    assert_eq!(stalled(&store, ASKER, t0), []);
    assert_eq!(handed(&store, t0), [back]);
}

#[test]
fn the_word_that_no_report_came_leaves_with_the_worker_it_was_said_of() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    store.report(&[(id, 0, Some(pressure(9, 0)))], t0, LoadSource::Poll);
    store.report(&[(id, 0, None)], t0, LoadSource::Poll);
    assert_eq!(overloaded(&store, t0), []);
    store.remove(id, revision(&store, id)).expect("removed");
    let back = store
        .upsert(facts("http://a:8000", Role::Single, &[MODEL]))
        .id();
    // The one that returned reports a queue past the threshold before its
    // build. The build judges it from that record: the word said of the one
    // that left is not carried to it.
    store.record(back, 0, pressure(9, 0), t0, LoadSource::Stream);
    store.publish_now();
    assert_eq!(overloaded(&store, t0), [back]);
}

#[test]
fn a_poll_with_no_report_for_a_worker_that_returned_moves_no_verdict_before_its_build() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    store.report(&[(id, 0, Some(pressure(9, 0)))], t0, LoadSource::Poll);
    assert_eq!(
        (overloaded(&store, t0), store.overload_flips()),
        (vec![id], 1)
    );
    store.remove(id, revision(&store, id)).expect("removed");
    let back = store
        .upsert(facts("http://a:8000", Role::Single, &[MODEL]))
        .id();
    // The poll's word is the new registration's, whose engine is not built
    // yet: it does not reach the engine of the one that left.
    store.report(&[(back, 0, None)], t0, LoadSource::Poll);
    assert_eq!(store.overload_flips(), 1);
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
                    // Every row of the view, whatever its health: a profile
                    // with no health filter picks among them all.
                    for row in &out.candidates {
                        if let Some(&index) = index_of.get(&row.id) {
                            assert!(!left[index], "worker {index}: listed after its leave");
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

/// The contract names the window by its figure, for a caller that reads no
/// source: the sentence and the store's constant move together.
#[test]
fn the_contract_names_the_window_a_registration_is_in_force_within() {
    const CONTRACT: &str = include_str!("../../../../contracts/policy_controller.proto");
    let figure = format!(
        "publication window ({} ms) of their answer",
        PUBLISH_WINDOW.as_millis()
    );
    assert!(
        CONTRACT.contains(&figure),
        "the contract does not say: {figure}"
    );
}
