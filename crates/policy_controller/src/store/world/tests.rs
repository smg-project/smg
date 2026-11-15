//! The cases of the world and of a decision's build from it, over
//! [`MemoryStore`]: every instant is handed in, none is slept.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use super::{Ask, Built, ViewBuffers, World, Writer, PUBLISH_WINDOW};
use crate::{
    store::{
        GatewayId, HealthSink, LoadSink, LoadSource, MemoryStore, Reachability, WorkerFacts,
        WorkerPatch, WorkerRow, WorkerStatus, WorkersStore,
    },
    view::{fingerprint, AffinityFacts, Candidate, LoadRow, Phase, Role, WireModes, WorkerId},
    PolicySpec,
};

const MODEL: &str = "m";
const ASKER: GatewayId = GatewayId(1);

fn facts(key: &str, role: Role, models: &[&str]) -> WorkerFacts {
    WorkerFacts {
        key: key.to_string(),
        engine: key.to_string(),
        incarnation: 1,
        role,
        models: models.iter().map(|model| (*model).to_string()).collect(),
        dp_size: 1,
        ..WorkerFacts::default()
    }
}

/// A worker of the model, registered and not yet ready.
fn member(store: &MemoryStore, key: &str) -> WorkerId {
    store.upsert(facts(key, Role::Single, &[MODEL])).id()
}

fn status(store: &MemoryStore, id: WorkerId, status: WorkerStatus) {
    let revision = WorkersStore::get(store, id).expect("the row").revision;
    store.set_status(id, status, revision).expect("the status");
}

fn load(running: u32, age_ms: u32) -> LoadRow {
    LoadRow {
        running,
        report_age_ms: age_ms,
        ..LoadRow::default()
    }
}

fn ask(now: Instant) -> Ask {
    Ask {
        gateway: ASKER,
        now,
        pinned: None,
        affinity: false,
    }
}

/// One decision's view of the model's single pool.
fn view(store: &MemoryStore, ask: &Ask) -> (Vec<Candidate>, Vec<AffinityFacts>, Built) {
    let mut out = ViewBuffers::default();
    let built = store
        .build_view(MODEL, Role::Single, ask, &mut out)
        .expect("the pool");
    (out.candidates, out.affinity, built)
}

/// What the pool's first row reads as running, when it has a record.
fn running(store: &MemoryStore, now: Instant) -> Option<u32> {
    view(store, &ask(now)).0[0].load.map(|load| load.running)
}

/// The workers of the pool that `gateway`'s vetoes mask at `now`, by id.
fn stalled(store: &MemoryStore, gateway: GatewayId, now: Instant) -> Vec<WorkerId> {
    let asked = Ask {
        gateway,
        ..ask(now)
    };
    view(store, &asked)
        .0
        .iter()
        .filter(|row| row.health.stalled)
        .map(|row| row.id)
        .collect()
}

#[test]
fn a_pool_holds_every_member_of_its_model_and_role_by_id_whatever_its_status() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let ready = member(&store, "http://a:8000");
    let pending = member(&store, "http://b:8000");
    let draining = member(&store, "http://c:8000");
    let prefill = store
        .upsert(facts("http://p:8000", Role::Prefill, &[MODEL]))
        .id();
    store.upsert(facts("http://o:8000", Role::Single, &["other"]));
    status(&store, ready, WorkerStatus::Ready);
    status(&store, draining, WorkerStatus::Draining);

    let (rows, column, built) = view(&store, &ask(now));
    let mut ids = [ready, pending, draining];
    ids.sort_unstable();
    assert_eq!(rows.iter().map(|row| row.id).collect::<Vec<_>>(), ids);
    assert_eq!(
        rows.iter().map(|row| row.position).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    for row in &rows {
        assert_eq!(row.health.healthy, row.id == ready, "{row:?}");
        assert_eq!(row.health.draining, row.id == draining, "{row:?}");
        assert_eq!(row.role, Role::Single);
        assert_eq!(row.dp_size, 1);
        assert_ne!(row.key, 0);
        assert!(row.load.is_none());
    }
    assert!(column.is_empty());
    assert_eq!(built.pool_size, 3);
    assert_eq!(built.membership_epoch, store.world().epoch());

    let world = store.world();
    assert_eq!(world.members(MODEL, Role::Single), Some(3));
    assert_eq!(world.members(MODEL, Role::Prefill), Some(1));
    assert_eq!(world.members("other", Role::Single), Some(1));
    assert_eq!(world.members(MODEL, Role::Decode), None);
    let mut out = ViewBuffers::default();
    world
        .build(MODEL, Role::Prefill, &ask(now), &mut out)
        .expect("the prefill pool");
    assert_eq!(out.candidates.len(), 1);
    assert_eq!(out.candidates[0].id, prefill);
    assert_eq!(out.candidates[0].role, Role::Prefill);
    // No member serves it: nothing is built and the buffers are cleared.
    assert_eq!(world.build("none", Role::Single, &ask(now), &mut out), None);
    assert!(out.candidates.is_empty());
}

#[test]
fn another_gateways_host_local_row_is_not_written_and_the_slice_is_its_own_index() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let open = member(&store, "http://a:8000");
    let local = |key: &str, gateway: u64| {
        store
            .upsert(WorkerFacts {
                reachable_from: Reachability::Gateway(GatewayId(gateway)),
                ..facts(key, Role::Single, &[MODEL])
            })
            .id()
    };
    let theirs = local("unix:///b", 2);
    let ours = local("unix:///c", 1);
    let last = member(&store, "http://d:8000");
    let ids_of = |gateway: u64| {
        let mut out = ViewBuffers::default();
        let built = store
            .build_view(
                MODEL,
                Role::Single,
                &Ask {
                    gateway: GatewayId(gateway),
                    affinity: true,
                    ..ask(now)
                },
                &mut out,
            )
            .expect("the pool");
        assert_eq!(built.pool_size as usize, out.candidates.len());
        // The column lines up with the rows that were written.
        assert_eq!(out.affinity.len(), out.candidates.len());
        for (index, row) in out.candidates.iter().enumerate() {
            assert_eq!(row.position as usize, index);
        }
        let mut ids: Vec<WorkerId> = out.candidates.iter().map(|row| row.id).collect();
        assert!(ids.is_sorted(), "the slice is sorted by worker id");
        ids.sort_unstable();
        ids
    };
    let sorted = |mut ids: Vec<WorkerId>| {
        ids.sort_unstable();
        ids
    };
    assert_eq!(ids_of(1), sorted(vec![open, ours, last]));
    assert_eq!(ids_of(2), sorted(vec![open, theirs, last]));
    assert_eq!(ids_of(3), sorted(vec![open, last]));
    assert_eq!(store.world().members(MODEL, Role::Single), Some(4));

    // A fact that arrives by a worker's id is found in the slice the
    // decision is made over, with the other gateway's row left out of it.
    for pinned in [open, ours, last] {
        let (rows, _, _) = view(
            &store,
            &Ask {
                pinned: Some(pinned),
                ..ask(now)
            },
        );
        for row in &rows {
            assert_eq!(row.session_hit, row.id == pinned, "{row:?}");
            assert_eq!(row.host_local, row.id == ours, "{row:?}");
        }
    }
    // A pin on a row this gateway does not reach marks no row.
    let (rows, _, _) = view(
        &store,
        &Ask {
            pinned: Some(theirs),
            ..ask(now)
        },
    );
    assert!(rows.iter().all(|row| !row.session_hit));
}

#[test]
fn a_decision_reads_one_world_and_a_membership_write_builds_the_next() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let first = member(&store, "http://a:8000");
    let before = store.world();
    assert_eq!(before.epoch(), store.epoch());

    let second = member(&store, "http://b:8000");
    let after = store.world();
    assert!(after.epoch() > before.epoch());
    assert_eq!(after.epoch(), store.epoch());
    // The world a decision took still reads the membership it was built by.
    let mut out = ViewBuffers::default();
    before
        .build(MODEL, Role::Single, &ask(now), &mut out)
        .expect("the pool");
    assert_eq!(out.candidates.len(), 1);
    assert_eq!(out.candidates[0].id, first);
    after
        .build(MODEL, Role::Single, &ask(now), &mut out)
        .expect("the pool");
    assert_eq!(out.candidates.len(), 2);

    // A status change moves the store's epoch and no world's: its
    // publication derives nothing.
    status(&store, second, WorkerStatus::Ready);
    assert_eq!(store.world().epoch(), after.epoch());
    assert!(store.epoch() > after.epoch());

    // The last member of a pool takes the pool with it.
    for id in [first, second] {
        let revision = WorkersStore::get(&store, id).expect("the row").revision;
        store.remove(id, revision).expect("removed");
    }
    assert_eq!(store.world().members(MODEL, Role::Single), None);
    assert_eq!(
        store.build_view(MODEL, Role::Single, &ask(now), &mut out),
        None
    );
    assert_eq!(store.world().epoch(), store.epoch());
}

/// The first row of `model`'s pool as `world` holds it: its running count.
fn running_in(world: &World, model: &str, now: Instant) -> Option<u32> {
    let mut out = ViewBuffers::default();
    world
        .build(model, Role::Single, &ask(now), &mut out)
        .expect("the pool");
    out.candidates[0].load.map(|load| load.running)
}

#[test]
fn two_publications_of_one_membership_are_two_worlds_of_one_epoch_sharing_what_it_derived() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let one = store
        .upsert(facts("http://a:8000", Role::Single, &["one"]))
        .id();
    store.upsert(facts("http://b:8000", Role::Single, &["two"]));
    let before = store.world();
    // A load publication between two membership writes.
    store.record(one, 0, load(3, 0), t0, LoadSource::Stream);
    let after = store.world();
    assert!(
        !Arc::ptr_eq(&before, &after),
        "a publication is a new world"
    );
    // The epoch is the membership's: the one thing to compare.
    assert_eq!(before.epoch(), after.epoch());
    // Nothing derived was built again: the two worlds share it by pointer,
    assert!(Arc::ptr_eq(&before.derived, &after.derived));
    // and the pool no write was folded into keeps its rows by pointer.
    let rows = |world: &World, model: &str| {
        let index =
            world.derived.by_model[model][super::role_slot(Role::Single)].expect("the pool");
        Arc::clone(&world.rows[index as usize])
    };
    assert!(Arc::ptr_eq(&rows(&before, "two"), &rows(&after, "two")));
    assert!(!Arc::ptr_eq(&rows(&before, "one"), &rows(&after, "one")));
    // A world never changes once published: the one a decision took before
    // the record reads none.
    assert_eq!(running_in(&before, "one", t0), None);
    assert_eq!(running_in(&after, "one", t0), Some(3));

    // A membership write derives anew and moves the epoch.
    member(&store, "http://c:8000");
    let next = store.world();
    assert!(next.epoch() > after.epoch());
    assert!(!Arc::ptr_eq(&after.derived, &next.derived));
}

#[test]
fn one_publication_holds_every_pools_folds_whichever_write_made_it() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let one = store
        .upsert(facts("http://a:8000", Role::Single, &["one"]))
        .id();
    let two = store
        .upsert(facts("http://b:8000", Role::Single, &["two"]))
        .id();
    let shown = |model: &str| running_in(&store.world(), model, t0);
    // The first record opens the window; one more for each pool waits.
    store.record(one, 0, load(1, 0), t0, LoadSource::Stream);
    let inside = t0 + Duration::from_millis(1);
    store.record(one, 0, load(2, 0), inside, LoadSource::Stream);
    store.record(two, 0, load(3, 0), inside, LoadSource::Stream);
    assert_eq!((shown("one"), shown("two")), (Some(1), None));
    // The window's end is one world for the whole store.
    store.publish(t0 + PUBLISH_WINDOW);
    assert_eq!((shown("one"), shown("two")), (Some(2), Some(3)));

    // The publications made at once go through the same door: a status
    // change of one pool's worker shows what waited in the other,
    let again = t0 + PUBLISH_WINDOW + Duration::from_millis(1);
    store.record(two, 0, load(4, 0), again, LoadSource::Stream);
    assert_eq!(shown("two"), Some(3));
    status(&store, one, WorkerStatus::Ready);
    assert_eq!(shown("two"), Some(4));
    assert_eq!(store.publish_due(), None);
    // and so does a veto.
    store.record(two, 0, load(5, 0), again, LoadSource::Stream);
    assert_eq!(shown("two"), Some(4));
    store.gateway_veto(ASKER, one, Some(again + Duration::from_secs(1)));
    assert_eq!(shown("two"), Some(5));
    assert_eq!(store.publish_due(), None);
}

#[test]
fn a_decision_racing_the_publications_reads_both_sides_of_one_publication() {
    let turns = 20_000u32;
    let store = Arc::new(MemoryStore::new());
    let t0 = Instant::now();
    let prefill = store
        .upsert(facts("http://p:8000", Role::Prefill, &[MODEL]))
        .id();
    let decode = store
        .upsert(facts("http://d:8000", Role::Decode, &[MODEL]))
        .id();
    // Both sides hold turn 0 before a decision looks: the first record
    // opens the window, the second waits for its end.
    store.record(prefill, 0, load(0, 0), t0, LoadSource::Stream);
    store.record(decode, 0, load(0, 0), t0, LoadSource::Stream);
    store.publish(t0 + PUBLISH_WINDOW);
    let writer = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || {
            // A turn's two records land inside the window the turn before
            // opened, and its end shows them in one world.
            for turn in 1..=turns {
                let opened = t0 + PUBLISH_WINDOW * turn;
                let inside = opened + Duration::from_millis(1);
                store.record(prefill, 0, load(turn, 0), inside, LoadSource::Stream);
                store.record(decode, 0, load(turn, 0), inside, LoadSource::Stream);
                store.publish(opened + PUBLISH_WINDOW);
            }
        })
    };
    let side = |world: &World, role: Role, out: &mut ViewBuffers| {
        world.build(MODEL, role, &ask(t0), out).expect("the pool");
        out.candidates[0].load.map(|load| load.running)
    };
    let sides = |out: &mut ViewBuffers| {
        // One load: both sides are built from the world a decision took.
        let world = store.world();
        (
            side(&world, Role::Prefill, out),
            side(&world, Role::Decode, out),
        )
    };
    let mut out = ViewBuffers::default();
    while !writer.is_finished() {
        let (prefills, decodes) = sides(&mut out);
        assert_eq!(
            prefills, decodes,
            "the two sides of one world are of one publication"
        );
        std::hint::spin_loop();
    }
    writer.join().expect("the writer ends");
    assert_eq!(sides(&mut out), (Some(turns), Some(turns)));
}

#[test]
fn a_write_that_changes_nothing_a_decision_reads_publishes_no_world() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    let until = t0 + Duration::from_secs(1);
    store.gateway_veto(ASKER, id, Some(until));
    let world = store.world();
    // The same veto again, and the clearing of a veto nobody holds.
    store.gateway_veto(ASKER, id, Some(until));
    store.gateway_veto(GatewayId(2), id, None);
    assert!(Arc::ptr_eq(&world, &store.world()));
    // A change between two statuses a row reads alike: neither is ready,
    // neither is draining.
    status(&store, id, WorkerStatus::NotReady);
    assert!(Arc::ptr_eq(&world, &store.world()));
    // Each of these moves something a decision reads.
    store.gateway_veto(ASKER, id, Some(until + Duration::from_secs(1)));
    let moved = store.world();
    assert!(!Arc::ptr_eq(&world, &moved));
    status(&store, id, WorkerStatus::Ready);
    assert!(!Arc::ptr_eq(&moved, &store.world()));
}

#[test]
fn a_status_change_is_read_at_once_and_publishes_its_pool_whole() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    assert!(!view(&store, &ask(t0)).0[0].health.healthy);
    status(&store, id, WorkerStatus::Ready);
    assert!(view(&store, &ask(t0)).0[0].health.healthy);

    // A record that waits for the window's end rides the publication a
    // status change makes of its pool.
    store.record(id, 0, load(1, 0), t0, LoadSource::Stream);
    store.record(
        id,
        0,
        load(2, 0),
        t0 + Duration::from_millis(1),
        LoadSource::Stream,
    );
    assert_eq!(running(&store, t0), Some(1));
    assert!(store.publish_due().is_some());
    status(&store, id, WorkerStatus::Draining);
    let (rows, _, _) = view(&store, &ask(t0));
    assert!(!rows[0].health.healthy);
    assert!(rows[0].health.draining);
    assert_eq!(rows[0].load.map(|load| load.running), Some(2));
    assert_eq!(store.publish_due(), None);
}

#[test]
fn a_lone_record_inside_the_window_is_read_at_the_windows_end_which_the_sink_names() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    assert_eq!(running(&store, t0), None);
    assert_eq!(store.publish_due(), None);

    // The first record after a quiet window publishes by itself.
    store.record(id, 0, load(1, 0), t0, LoadSource::Stream);
    assert_eq!(running(&store, t0), Some(1));
    assert_eq!(store.publish_due(), None);

    // The last record of a burst lands inside the window: no later write
    // comes, and the sink names the instant it is due.
    let late = t0 + Duration::from_millis(5);
    store.record(id, 0, load(7, 0), late, LoadSource::Stream);
    assert_eq!(running(&store, late), Some(1));
    assert_eq!(store.publish_due(), Some(t0 + PUBLISH_WINDOW));

    // A call before the due instant publishes nothing.
    store.publish(t0 + PUBLISH_WINDOW - Duration::from_millis(1));
    assert_eq!(running(&store, late), Some(1));
    assert_eq!(store.publish_due(), Some(t0 + PUBLISH_WINDOW));

    store.publish(t0 + PUBLISH_WINDOW);
    assert_eq!(running(&store, t0 + PUBLISH_WINDOW), Some(7));
    assert_eq!(store.publish_due(), None);
    // The record is on file from its write, whatever the window.
    assert_eq!(store.load(id, 0).map(|sample| sample.row.running), Some(7));
}

#[test]
fn a_host_that_never_calls_the_window_shows_a_waiting_record_with_its_next_write() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let first = member(&store, "http://a:8000");
    let second = member(&store, "http://b:8000");
    let of = |id: WorkerId, now: Instant| {
        view(&store, &ask(now))
            .0
            .iter()
            .find(|row| row.id == id)
            .and_then(|row| row.load)
            .map(|load| load.running)
    };
    store.record(first, 0, load(1, 0), t0, LoadSource::Stream);
    let inside = t0 + Duration::from_millis(5);
    store.record(first, 0, load(2, 0), inside, LoadSource::Stream);
    assert_eq!(of(first, inside), Some(1));

    // A write a window after the last publication publishes by itself, and
    // what waited goes with it.
    let later = t0 + PUBLISH_WINDOW;
    store.record(second, 0, load(9, 0), later, LoadSource::Stream);
    assert_eq!(of(first, later), Some(2));
    assert_eq!(of(second, later), Some(9));
    assert_eq!(store.publish_due(), None);
}

#[test]
fn a_store_made_with_its_own_window_waits_that_long() {
    let window = Duration::from_millis(200);
    let store = MemoryStore::with_publish_window(window);
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    store.record(id, 0, load(1, 0), t0, LoadSource::Stream);
    store.record(id, 0, load(2, 0), t0 + PUBLISH_WINDOW, LoadSource::Stream);
    assert_eq!(running(&store, t0), Some(1));
    assert_eq!(store.publish_due(), Some(t0 + window));
    store.publish(t0 + window);
    assert_eq!(running(&store, t0), Some(2));
}

#[test]
fn a_window_the_clock_cannot_add_never_ends_and_a_status_change_still_shows_the_record() {
    let store = MemoryStore::with_publish_window(Duration::MAX);
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    // The first record opens the window and is read at once.
    store.record(id, 0, load(1, 0), t0, LoadSource::Stream);
    assert_eq!(running(&store, t0), Some(1));
    // The second waits; no instant is named for it and none ends its wait.
    let later = t0 + Duration::from_secs(1);
    store.record(id, 0, load(2, 0), later, LoadSource::Stream);
    assert_eq!(store.publish_due(), None);
    store.publish(later + Duration::from_secs(3_600));
    assert_eq!(running(&store, later), Some(1));
    // A status change publishes the pool whole, the waiting record in it.
    status(&store, id, WorkerStatus::Ready);
    assert_eq!(running(&store, later), Some(2));
}

#[test]
fn a_report_is_folded_whole_and_a_poll_asked_before_the_last_window_waits() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    store.record(id, 0, load(1, 0), t0, LoadSource::Stream);
    store.evict(id);
    // The eviction has no instant of its own: it waits for the window.
    assert_eq!(running(&store, t0), Some(1));
    assert_eq!(store.publish_due(), Some(t0 + PUBLISH_WINDOW));
    // A poll asked before the last publication lands inside its window.
    store.report(
        id,
        &[(0, load(4, 0)), (1, load(5, 0))],
        t0 - Duration::from_millis(3),
        LoadSource::Poll,
    );
    assert_eq!(running(&store, t0), Some(1));
    store.publish(t0 + PUBLISH_WINDOW);
    // The row reads its own rank's figures, never another rank's.
    assert_eq!(running(&store, t0), Some(4));
}

#[test]
fn a_report_whose_last_rank_is_not_the_members_still_shows_the_members_row() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    // The member is rank 0 and the report's last row is another rank's: the
    // report is shown for the row it reached, whichever row came last.
    store.report(
        id,
        &[(0, load(4, 0)), (1, load(5, 0)), (2, load(6, 0))],
        t0,
        LoadSource::Poll,
    );
    assert_eq!(running(&store, t0), Some(4));
    // Inside the window that opened, the next one waits and the sink names
    // its instant.
    store.report(
        id,
        &[(0, load(7, 0)), (1, load(8, 0))],
        t0 + Duration::from_millis(1),
        LoadSource::Poll,
    );
    assert_eq!(running(&store, t0), Some(4));
    assert_eq!(store.publish_due(), Some(t0 + PUBLISH_WINDOW));
    store.publish(t0 + PUBLISH_WINDOW);
    assert_eq!(running(&store, t0 + PUBLISH_WINDOW), Some(7));
}

#[test]
fn an_eviction_of_a_record_a_membership_write_carried_is_shown_at_once_when_no_window_opened() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let registered = WorkerFacts {
        dp_size: 2,
        ..facts("http://a:8000", Role::Single, &[MODEL])
    };
    let id = store.upsert(registered.clone()).id();
    // A record of a rank the row is not: on file, in no row, so no window
    // opens for it.
    store.record(id, 1, load(9, 0), t0, LoadSource::Stream);
    assert_eq!(running(&store, t0), None);
    assert_eq!(store.publish_due(), None);
    // The registration again, as that rank: the membership write carries
    // the record on file into the row.
    store.upsert(WorkerFacts {
        dp_rank: 1,
        ..registered
    });
    assert_eq!(running(&store, t0), Some(9));
    // No window ever opened, so the eviction has none to wait for.
    store.evict(id);
    assert_eq!(running(&store, t0), None);
    assert_eq!(store.publish_due(), None);
}

#[test]
fn an_evicted_member_reads_no_load_and_a_store_that_never_published_shows_it_at_once() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    // No window ever opened: the eviction of nothing changes nothing.
    store.evict(id);
    assert_eq!(running(&store, t0), None);
    assert_eq!(store.publish_due(), None);

    store.record(id, 0, load(3, 0), t0, LoadSource::Stream);
    store.evict(id);
    store.publish(t0 + PUBLISH_WINDOW);
    assert_eq!(running(&store, t0 + PUBLISH_WINDOW), None);
}

#[test]
fn the_age_is_the_decisions_own_and_saturates() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    store.record(id, 0, load(1, 250), t0, LoadSource::Stream);
    let sampled = t0 - Duration::from_millis(250);

    let now = t0 + Duration::from_secs(1);
    let (rows, _, built) = view(&store, &ask(now));
    assert_eq!(built.now_micros, store.micros(now));
    assert_eq!(rows[0].load_sampled_at_micros, store.micros(sampled));
    assert_eq!(rows[0].load.map(|load| load.report_age_ms), Some(1_250));
    // The published row is not written by a decision: a second decision
    // reads its own age.
    let later = t0 + Duration::from_secs(3);
    let (rows, _, _) = view(&store, &ask(later));
    assert_eq!(rows[0].load.map(|load| load.report_age_ms), Some(3_250));
    // A decision dated before the sample reads no age, never a wrapped one.
    let (rows, _, _) = view(&store, &ask(sampled - Duration::from_secs(1)));
    assert_eq!(rows[0].load.map(|load| load.report_age_ms), Some(0));
    // Seven weeks on, the age stands at its ceiling.
    let (rows, _, _) = view(&store, &ask(t0 + Duration::from_secs(50 * 86_400)));
    assert_eq!(rows[0].load.map(|load| load.report_age_ms), Some(u32::MAX));
}

#[test]
fn a_membership_write_keeps_the_instant_of_a_record_on_file() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    store.record(id, 0, load(1, 250), t0, LoadSource::Stream);
    let sampled = t0 - Duration::from_millis(250);
    let now = t0 + Duration::from_secs(3);
    let read = || {
        let (rows, _, _) = view(&store, &ask(now));
        let row = rows.iter().find(|row| row.id == id).expect("the member");
        (
            row.load_sampled_at_micros,
            row.load.map(|load| load.report_age_ms),
        )
    };
    assert_eq!(read(), (store.micros(sampled), Some(3_250)));
    // Another member registers and the rows are built again: the record on
    // file keeps its own instant, so its age reads as it did before.
    member(&store, "http://b:8000");
    assert_eq!(read(), (store.micros(sampled), Some(3_250)));
}

#[test]
fn a_gateways_veto_masks_the_worker_for_that_gateway_until_its_end_or_its_clear() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let vetoed = member(&store, "http://a:8000");
    let other = member(&store, "http://b:8000");
    let until = t0 + Duration::from_secs(1);
    store.gateway_veto(ASKER, vetoed, Some(until));
    let stalled = |gateway: GatewayId, now: Instant| stalled(&store, gateway, now);
    assert_eq!(stalled(ASKER, t0), [vetoed]);
    // The veto is its gateway's alone, and it ends at its instant.
    assert_eq!(stalled(GatewayId(2), t0), []);
    assert_eq!(stalled(ASKER, until), []);
    // The view agrees with the sink's own read at every instant.
    for now in [t0, until - Duration::from_millis(1), until] {
        assert_eq!(
            stalled(ASKER, now) == [vetoed],
            store.vetoed(ASKER, vetoed, now)
        );
    }

    store.gateway_veto(ASKER, other, Some(until));
    let mut both = [vetoed, other];
    both.sort_unstable();
    assert_eq!(stalled(ASKER, t0), both);
    store.gateway_veto(ASKER, vetoed, None);
    assert_eq!(stalled(ASKER, t0), [other]);

    // A veto leaves with its worker's row: the next incarnation starts
    // without it.
    let revision = WorkersStore::get(&store, other).expect("the row").revision;
    store.remove(other, revision).expect("removed");
    let again = member(&store, "http://b:8000");
    assert_eq!(again, other, "the same key and incarnation is the same id");
    assert_eq!(stalled(ASKER, t0), []);
}

#[test]
fn two_vetoes_of_one_gateway_set_in_falling_id_order_are_each_found_by_its_id() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let mut ids = [
        member(&store, "http://a:8000"),
        member(&store, "http://b:8000"),
    ];
    ids.sort_unstable();
    let [low, high] = ids;
    let until = t0 + Duration::from_secs(1);
    // The higher id first: a gateway's list is kept by id, not by arrival.
    store.gateway_veto(ASKER, high, Some(until));
    store.gateway_veto(ASKER, low, Some(until));
    assert_eq!(stalled(&store, ASKER, t0), [low, high]);
    // The first one set is cleared by its own id and the other still masks.
    store.gateway_veto(ASKER, high, None);
    assert_eq!(stalled(&store, ASKER, t0), [low]);
    assert!(!store.vetoed(ASKER, high, t0));
    assert!(store.vetoed(ASKER, low, t0));
    // Set again and given a new end, each is still found where it stands.
    store.gateway_veto(ASKER, high, Some(until));
    store.gateway_veto(ASKER, low, Some(t0 + Duration::from_millis(1)));
    assert_eq!(
        stalled(&store, ASKER, t0 + Duration::from_millis(1)),
        [high]
    );
}

#[test]
fn the_column_is_filled_for_a_profile_that_declares_it_and_for_no_other() {
    let store = MemoryStore::new();
    let ids = [
        member(&store, "http://a:8000"),
        member(&store, "http://b:8000"),
        member(&store, "http://c:8000"),
    ];
    let admitted = |id: WorkerId| WorkersStore::get(&store, id).expect("the row").admitted_at;
    let now = admitted(ids[2]) + Duration::from_secs(10);
    let needs = |preset: &str| {
        PolicySpec::preset(preset)
            .expect("a preset of this build")
            .validate()
            .expect("a preset validates")
            .needs_for(MODEL, Phase::Single)
    };

    // The builder is what holds a host to a declared need: under a profile
    // whose stages read the column, every candidate has its entry.
    assert!(needs("cache_aware").affinity);
    let (rows, column, _) = view(
        &store,
        &Ask {
            affinity: needs("cache_aware").affinity,
            ..ask(now)
        },
    );
    assert_eq!(column.len(), rows.len());
    for (row, held) in rows.iter().zip(&column) {
        // The rows' own clock on both instants, as the build takes the age:
        // each is whole microseconds since the store's origin.
        let age_micros = store.micros(now) - store.micros(admitted(row.id));
        assert_eq!(
            *held,
            AffinityFacts {
                admitted_age_ms: u32::try_from(age_micros / 1_000).expect("ten seconds"),
                ..AffinityFacts::default()
            }
        );
        assert!(held.admitted_age_ms >= 10_000);
    }

    // A profile that declares none of it copies none of it.
    assert!(!needs("round_robin").affinity);
    let (rows, column, _) = view(
        &store,
        &Ask {
            affinity: needs("round_robin").affinity,
            ..ask(now)
        },
    );
    assert_eq!(rows.len(), 3);
    assert!(column.is_empty());
}

#[test]
fn a_worker_serving_two_models_has_a_row_in_each_pool_and_one_record_reaches_both() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let shared = store
        .upsert(facts("http://a:8000", Role::Single, &["one", "two", "one"]))
        .id();
    let single = store
        .upsert(facts("http://b:8000", Role::Single, &["two"]))
        .id();
    let world = store.world();
    // A model listed twice is one row.
    assert_eq!(world.members("one", Role::Single), Some(1));
    assert_eq!(world.members("two", Role::Single), Some(2));

    store.record(shared, 0, load(6, 0), t0, LoadSource::Stream);
    status(&store, shared, WorkerStatus::Ready);
    let of = |model: &str, id: WorkerId| {
        let mut out = ViewBuffers::default();
        store
            .build_view(model, Role::Single, &ask(t0), &mut out)
            .expect("the pool");
        out.candidates
            .iter()
            .find(|row| row.id == id)
            .map(|row| (row.health.healthy, row.load.map(|load| load.running)))
    };
    assert_eq!(of("one", shared), Some((true, Some(6))));
    assert_eq!(of("two", shared), Some((true, Some(6))));
    assert_eq!(of("two", single), Some((false, None)));
}

#[test]
fn a_new_world_carries_the_records_on_file_and_a_replaced_incarnation_starts_without() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    status(&store, id, WorkerStatus::Ready);
    store.record(id, 0, load(1, 0), t0, LoadSource::Stream);
    // A record waits inside the window when a membership write builds the
    // next world: the world carries it, and nothing waits after.
    store.record(
        id,
        0,
        load(8, 0),
        t0 + Duration::from_millis(1),
        LoadSource::Stream,
    );
    assert_eq!(running(&store, t0), Some(1));
    member(&store, "http://b:8000");
    let of = |id: WorkerId| {
        view(&store, &ask(t0))
            .0
            .iter()
            .find(|row| row.id == id)
            .map(|row| (row.health.healthy, row.load.map(|load| load.running)))
    };
    assert_eq!(of(id), Some((true, Some(8))));
    assert_eq!(store.publish_due(), None);

    // The same key under a new incarnation is a new row: its status is
    // carried where the registration matches, its records are not.
    let next = store
        .upsert(WorkerFacts {
            incarnation: 2,
            ..facts("http://a:8000", Role::Single, &[MODEL])
        })
        .id();
    assert_ne!(next, id);
    assert_eq!(of(id), None);
    assert_eq!(of(next), Some((true, None)));
    // A record for the id that left reaches no row.
    store.record(
        id,
        0,
        load(5, 0),
        t0 + PUBLISH_WINDOW * 2,
        LoadSource::Stream,
    );
    assert_eq!(of(next), Some((true, None)));
    assert_eq!(store.publish_due(), None);
}

#[test]
fn a_row_carries_its_registrations_fixed_facts_and_reads_its_own_rank() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let key = "grpc://d:9000@3";
    let registered = WorkerFacts {
        engine: "grpc://d:9000".to_string(),
        dp_rank: 3,
        dp_size: 4,
        wire_modes: WireModes::GRPC.union(WireModes::HTTP),
        pair_class: 9,
        reachable_from: Reachability::Gateway(ASKER),
        ..facts(key, Role::Decode, &[MODEL])
    };
    let id = store.upsert(registered.clone()).id();
    let row_of = || {
        let mut out = ViewBuffers::default();
        store
            .build_view(MODEL, Role::Decode, &ask(t0), &mut out)
            .expect("the pool");
        assert_eq!(out.candidates.len(), 1);
        out.candidates[0]
    };
    let row = row_of();
    assert_eq!(row.id, id);
    assert_eq!(row.role, Role::Decode);
    assert_eq!((row.dp_rank, row.dp_size), (3, 4));
    assert_eq!(row.wire_modes, WireModes::GRPC.union(WireModes::HTTP));
    assert_eq!(row.pair_class, 9);
    assert!(row.host_local);
    // The low 32 bits of the registration key's fingerprint.
    assert_eq!(row.key, fingerprint(key.as_bytes()) as u32);
    // What no fact of the store feeds yet reads as the row's default.
    assert_eq!(
        (row.in_flight, row.session_in_flight, row.active_sessions),
        (0, 0, 0)
    );
    assert!(!row.has_capacity && !row.has_room && !row.health.overloaded);
    assert_eq!(
        (row.zone, row.labels, row.engine_kind, row.connector),
        (0, 0, 0, 0)
    );
    assert!(!row.vision && !row.ec_compatible && !row.session_hit);

    // The row reads the record of its own rank, never another rank's.
    store.record(id, 0, load(5, 0), t0, LoadSource::Stream);
    assert_eq!(row_of().load, None);
    store.record(id, 3, load(6, 0), t0 + PUBLISH_WINDOW, LoadSource::Stream);
    assert_eq!(row_of().load.map(|load| load.running), Some(6));

    // A refresh of the registration and a patch each build the next world.
    store.upsert(WorkerFacts {
        dp_size: 8,
        ..registered
    });
    assert_eq!(row_of().dp_size, 8);
    assert_eq!(row_of().load.map(|load| load.running), Some(6));
    assert_eq!(store.world().epoch(), store.epoch());
    let revision = WorkersStore::get(&store, id).expect("the row").revision;
    store
        .patch(
            id,
            WorkerPatch {
                priority: Some(1),
                ..WorkerPatch::default()
            },
            revision,
        )
        .expect("patched");
    assert_eq!(store.world().epoch(), store.epoch());
}

#[test]
fn a_pool_of_many_members_is_sorted_by_worker_id_and_each_is_found_by_it() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let ids: Vec<WorkerId> = (0..24)
        .map(|index| member(&store, &format!("http://w{index}:8000")))
        .collect();
    let (rows, _, _) = view(&store, &ask(now));
    let listed: Vec<WorkerId> = rows.iter().map(|row| row.id).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(listed, sorted);
    for &pinned in &ids {
        let (rows, _, _) = view(
            &store,
            &Ask {
                pinned: Some(pinned),
                ..ask(now)
            },
        );
        let hit: Vec<WorkerId> = rows
            .iter()
            .filter(|row| row.session_hit)
            .map(|row| row.id)
            .collect();
        assert_eq!(hit, [pinned]);
    }
}

#[test]
fn a_probes_verdict_and_a_connect_signal_are_read_at_once() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let id = member(&store, "http://a:8000");
    let healthy = || view(&store, &ask(now)).0[0].health.healthy;
    let revision = || WorkersStore::get(&store, id).expect("the row").revision;
    assert!(!healthy());

    // The connect signal of a handshake-promoted worker.
    assert!(store.connected(id, revision()).is_some());
    assert!(healthy());

    // The prober takes a ready worker out after its failure threshold.
    let threshold = WorkersStore::get(&store, id)
        .expect("the row")
        .facts
        .health
        .failure_threshold;
    for _ in 0..threshold {
        assert!(healthy());
        store.probe_outcome(id, false, revision());
    }
    assert!(!healthy());
}

#[test]
fn one_models_members_stand_in_the_pool_of_their_own_role() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    // One model served in every role, a different count each: a role read
    // from another's slot is seen by its size and by its rows.
    let roles = [
        (Role::Single, 1_u32),
        (Role::Prefill, 2),
        (Role::Decode, 3),
        (Role::Encode, 4),
    ];
    for (role, count) in roles {
        for index in 0..count {
            store.upsert(facts(
                &format!("http://{role:?}-{index}:8000"),
                role,
                &[MODEL],
            ));
        }
    }
    for (role, count) in roles {
        let mut out = ViewBuffers::default();
        let built = store
            .build_view(MODEL, role, &ask(t0), &mut out)
            .expect("the pool");
        assert_eq!(built.pool_size, count, "{role:?}");
        assert!(
            out.candidates.iter().all(|row| row.role == role),
            "{role:?}"
        );
    }
}

#[test]
fn a_veto_masks_its_workers_row_in_every_pool_it_serves() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + Duration::from_secs(1);
    let both = store
        .upsert(facts("http://a:8000", Role::Single, &["one", "two"]))
        .id();
    store.upsert(facts("http://b:8000", Role::Single, &["one"]));
    let two = store
        .upsert(facts("http://c:8000", Role::Single, &["two"]))
        .id();
    let masked = |model: &str| {
        let mut out = ViewBuffers::default();
        store
            .build_view(model, Role::Single, &ask(t0), &mut out)
            .expect("the pool");
        out.candidates
            .iter()
            .filter(|row| row.health.stalled)
            .map(|row| row.id)
            .collect::<Vec<_>>()
    };
    store.gateway_veto(ASKER, both, Some(until));
    assert_eq!(masked("one"), [both]);
    assert_eq!(masked("two"), [both]);
    // A veto on a worker of one pool masks nothing in the other.
    store.gateway_veto(ASKER, two, Some(until));
    assert_eq!(masked("one"), [both]);
    let mut pair = [both, two];
    pair.sort_unstable();
    assert_eq!(masked("two"), pair);
    store.gateway_veto(ASKER, both, None);
    assert_eq!(masked("one"), []);
    assert_eq!(masked("two"), [two]);
}

#[test]
fn a_veto_beside_host_local_rows_masks_its_own_row_in_each_gateways_slice() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + Duration::from_secs(1);
    let other = GatewayId(2);
    let register = |key: &str, reachable_from: Reachability| {
        store
            .upsert(WorkerFacts {
                reachable_from,
                ..facts(key, Role::Single, &[MODEL])
            })
            .id()
    };
    register("http://a:8000", Reachability::Any);
    let theirs = register("http://b:8000", Reachability::Gateway(other));
    let behind = register("http://c:8000", Reachability::Any);
    register("http://d:8000", Reachability::Gateway(ASKER));
    let last = register("http://e:8000", Reachability::Any);
    // The asking gateway's slice leaves the other gateway's row out: a veto
    // on a row behind it masks that row, never the one at its index.
    store.gateway_veto(ASKER, behind, Some(until));
    assert_eq!(stalled(&store, ASKER, t0), [behind]);
    // A veto on a row the gateway's slice does not hold masks nothing.
    store.gateway_veto(ASKER, theirs, Some(until));
    assert_eq!(stalled(&store, ASKER, t0), [behind]);
    // The other gateway's slice leaves out the asking gateway's row.
    store.gateway_veto(other, last, Some(until));
    assert_eq!(stalled(&store, other, t0), [last]);
    assert_eq!(stalled(&store, ASKER, t0), [behind]);
}

#[test]
fn the_pair_door_builds_both_sides_of_one_publication() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let prefill = store
        .upsert(facts("http://p:8000", Role::Prefill, &[MODEL]))
        .id();
    let decode = store
        .upsert(facts("http://d:8000", Role::Decode, &[MODEL]))
        .id();
    let (mut first, mut second) = (ViewBuffers::default(), ViewBuffers::default());
    let (left, right) = store
        .build_pair(MODEL, &ask(t0), &mut first, &mut second)
        .expect("both pools");
    assert_eq!((left.pool_size, right.pool_size), (1, 1));
    assert_eq!(left.membership_epoch, right.membership_epoch);
    assert_eq!(
        (first.candidates[0].id, second.candidates[0].id),
        (prefill, decode)
    );
    // A model served on one side alone has no pair, and neither buffer
    // keeps a view: not the side that has members, not the last pair's;
    // neither its rows nor, under an ask that declares it, its column.
    let with_column = Ask {
        affinity: true,
        ..ask(t0)
    };
    let empty = |first: &ViewBuffers, second: &ViewBuffers| {
        first.candidates.is_empty()
            && first.affinity.is_empty()
            && second.candidates.is_empty()
            && second.affinity.is_empty()
    };
    store.upsert(facts("http://s:8000", Role::Prefill, &["prefill-alone"]));
    assert_eq!(
        store.build_pair("prefill-alone", &with_column, &mut first, &mut second),
        None
    );
    assert!(empty(&first, &second));
    store
        .build_pair(MODEL, &with_column, &mut first, &mut second)
        .expect("both pools");
    assert_eq!((first.affinity.len(), second.affinity.len()), (1, 1));
    store.upsert(facts("http://t:8000", Role::Decode, &["decode-alone"]));
    assert_eq!(
        store.build_pair("decode-alone", &with_column, &mut first, &mut second),
        None
    );
    assert!(empty(&first, &second));
}

#[test]
fn the_pair_door_racing_the_publications_builds_both_sides_from_one_load() {
    /// How many publications the door is read across before the case ends.
    const SEEN: u32 = 256;
    let store = Arc::new(MemoryStore::new());
    let t0 = Instant::now();
    let prefill = store
        .upsert(facts("http://p:8000", Role::Prefill, &[MODEL]))
        .id();
    let decode = store
        .upsert(facts("http://d:8000", Role::Decode, &[MODEL]))
        .id();
    // Both sides hold turn 0 before the door is asked.
    store.record(prefill, 0, load(0, 0), t0, LoadSource::Stream);
    store.record(decode, 0, load(0, 0), t0, LoadSource::Stream);
    store.publish(t0 + PUBLISH_WINDOW);
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let (store, stop) = (Arc::clone(&store), Arc::clone(&stop));
        std::thread::spawn(move || {
            // One publication shows a turn's two records, turn after turn
            // until the door was read across enough of them.
            let mut turn = 0u32;
            while !stop.load(Ordering::Relaxed) {
                turn += 1;
                let opened = t0 + PUBLISH_WINDOW * turn;
                let inside = opened + Duration::from_millis(1);
                store.record(prefill, 0, load(turn, 0), inside, LoadSource::Stream);
                store.record(decode, 0, load(turn, 0), inside, LoadSource::Stream);
                store.publish(opened + PUBLISH_WINDOW);
            }
            turn
        })
    };
    let (mut first, mut second) = (ViewBuffers::default(), ViewBuffers::default());
    let mut sides = || {
        store
            .build_pair(MODEL, &ask(t0), &mut first, &mut second)
            .expect("both pools");
        let running = |side: &ViewBuffers| side.candidates[0].load.map(|load| load.running);
        (running(&first), running(&second))
    };
    // The case ends at an event, the door read across `SEEN` publications,
    // and holds the door's rule at every read: one load of the world feeds
    // both sides, so a publication between the two builds shows in neither.
    let (mut seen, mut last) = (0, None);
    while seen < SEEN {
        assert!(
            !writer.is_finished(),
            "the writer ended before the door was read across {SEEN} publications"
        );
        let (prefills, decodes) = sides();
        assert_eq!(
            prefills, decodes,
            "the door's two sides are of one load of the world"
        );
        if prefills != last {
            (seen, last) = (seen + 1, prefills);
        }
    }
    stop.store(true, Ordering::Relaxed);
    let turns = writer.join().expect("the writer ends");
    assert_eq!(sides(), (Some(turns), Some(turns)));
}

#[test]
fn two_ranks_that_push_at_two_instants_are_each_aged_from_their_own() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let rank = |rank: u16| {
        store
            .upsert(WorkerFacts {
                engine: "grpc://e:9000".to_string(),
                dp_rank: rank,
                dp_size: 2,
                ..facts(&format!("grpc://e:9000@{rank}"), Role::Single, &[MODEL])
            })
            .id()
    };
    let (first, second) = (rank(0), rank(1));
    let ages = |now: Instant| {
        let (rows, _, _) = view(&store, &ask(now));
        let age = |id: WorkerId| {
            rows.iter()
                .find(|row| row.id == id)
                .and_then(|row| row.load)
                .map(|load| load.report_age_ms)
        };
        (age(first), age(second))
    };
    store.record(first, 0, load(1, 0), t0, LoadSource::Stream);
    // A decision between the two pushes: the second rank has no record yet.
    assert_eq!(ages(t0 + Duration::from_millis(2)), (Some(2), None));
    let later = t0 + PUBLISH_WINDOW;
    store.record(second, 1, load(2, 0), later, LoadSource::Stream);
    store.publish(later);
    // Each row's age counts from its own record's instant, never from the
    // later one.
    let window_ms = u32::try_from(PUBLISH_WINDOW.as_millis()).expect("a window in milliseconds");
    assert_eq!(
        ages(later + Duration::from_millis(5)),
        (Some(window_ms + 5), Some(5))
    );
}

#[test]
fn a_publication_before_a_departing_workers_row_leaves_still_shows_its_veto() {
    let origin = Instant::now();
    let t0 = origin + Duration::from_secs(1);
    let row = |id: u64, key: &str| WorkerRow {
        id: WorkerId(id),
        facts: facts(key, Role::Single, &[MODEL]),
        status: WorkerStatus::Ready,
        revision: 1,
        admitted_at: origin,
    };
    let members = [row(1, "http://a:8000"), row(2, "http://b:8000")];
    let masked = |world: &World| {
        let mut out = ViewBuffers::default();
        world
            .build(MODEL, Role::Single, &ask(t0), &mut out)
            .expect("the pool");
        out.candidates
            .iter()
            .filter(|row| row.health.stalled)
            .map(|row| row.id)
            .collect::<Vec<_>>()
    };
    let leaving = WorkerId(1);
    let mut writer = Writer::new(PUBLISH_WINDOW, origin);
    writer.rebuild(1, members.iter(), |_, _| None);
    let vetoed = writer
        .veto(ASKER, leaving, Some(t0 + Duration::from_secs(60)))
        .expect("a new veto");
    assert_eq!(masked(&vetoed), [leaving]);
    // Another worker's load write publishes while the removal is under
    // way: the leaving worker's row still stands in that world, masked.
    let between = writer.publish_window(t0);
    assert_eq!(masked(&between), [leaving]);
    // The membership write takes the row and its vetoes out together,
    let after = writer.rebuild(2, members[1..].iter(), |_, _| None);
    assert_eq!(after.members(MODEL, Role::Single), Some(1));
    assert_eq!(masked(&after), []);
    // and the same id registered again starts without them.
    let again = writer.rebuild(3, members.iter(), |_, _| None);
    assert_eq!(again.members(MODEL, Role::Single), Some(2));
    assert_eq!(masked(&again), []);
}

#[test]
fn a_departing_workers_vetoes_leave_alone_and_every_other_veto_stands() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + Duration::from_secs(60);
    let other = GatewayId(2);
    let leaving = member(&store, "http://a:8000");
    let staying = member(&store, "http://b:8000");
    // The asking gateway's list names both workers, the other gateway's
    // the one that stays.
    store.gateway_veto(ASKER, leaving, Some(until));
    store.gateway_veto(ASKER, staying, Some(until));
    store.gateway_veto(other, staying, Some(until));
    let mut both = [leaving, staying];
    both.sort_unstable();
    assert_eq!(stalled(&store, ASKER, t0), both);
    assert_eq!(stalled(&store, other, t0), [staying]);
    let revision = WorkersStore::get(&store, leaving)
        .expect("the row")
        .revision;
    store.remove(leaving, revision).expect("removed");
    // The membership write drops the vetoes on the worker that left and no
    // other: the rest of that gateway's list stands, and so does the list
    // that named members alone.
    assert_eq!(stalled(&store, ASKER, t0), [staying]);
    assert_eq!(stalled(&store, other, t0), [staying]);
}

#[test]
fn a_membership_write_keeps_the_standing_veto_of_each_of_many_members() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + Duration::from_secs(60);
    let mut vetoed: Vec<WorkerId> = (0..64)
        .map(|n| member(&store, &format!("http://w{n}:8000")))
        .collect();
    for &id in &vetoed {
        store.gateway_veto(ASKER, id, Some(until));
    }
    vetoed.sort_unstable();
    assert_eq!(stalled(&store, ASKER, t0), vetoed);
    // One more registration derives every pool anew. Each vetoed worker is
    // still a member, in whatever order the store's map hands that write
    // its members: no veto leaves.
    member(&store, "http://joined:8000");
    assert_eq!(stalled(&store, ASKER, t0), vetoed);
}

#[test]
fn a_veto_past_its_end_leaves_at_the_next_windows_publication() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let ended = member(&store, "http://a:8000");
    let standing = member(&store, "http://b:8000");
    let until = t0 + Duration::from_millis(10);
    store.gateway_veto(ASKER, ended, Some(until));
    store.gateway_veto(ASKER, standing, Some(until + Duration::from_secs(1)));
    let mut both = [ended, standing];
    both.sort_unstable();
    assert_eq!(stalled(&store, ASKER, t0), both);
    // Dated past the first veto's end over the publication that still
    // lists it: lapsed by the build's own compare.
    assert_eq!(stalled(&store, ASKER, until), [standing]);
    // A window's publication at or past that end takes the entry out by
    // the store's own clock. A decision dated before that instant reads
    // the publication, which no longer holds it; the veto that stands
    // stays.
    store.record(ended, 0, load(1, 0), until, LoadSource::Stream);
    assert_eq!(stalled(&store, ASKER, t0), [standing]);
    // The sink's own answer keeps its compare.
    assert!(store.vetoed(ASKER, ended, t0));
}
