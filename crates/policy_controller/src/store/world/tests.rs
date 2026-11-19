//! The cases of the world and of a decision's build from it, over
//! [`MemoryStore`]: every instant is handed in, none is slept.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use super::{
    run_of, seek, side_of, Ask, Built, Classes, GatewayRun, OutsideLists, PairBuffers, PoolVeto,
    PoolVetoes, StaleGeneration, ViewBuffers, World, Writer, PUBLISH_WINDOW, WALKED_RUNS,
};
use crate::{
    counts::Slot,
    spec::{BindingSpec, Match, OverloadSpec, Pairing},
    store::{
        loads::LoadSample, pairing::Refusal, GatewayId, HealthSink, LeasesStore, LoadSink,
        LoadSource, MembershipTail, MemoryStore, Overload, PairingDescriptor, Reachability,
        VetoWrite, WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus, WorkersStore,
    },
    view::{
        fingerprint, AffinityFacts, Candidate, Constraints, LoadRow, Phase, RequestFacts, Role,
        WireModes, WorkerId,
    },
    AppliedSpec, PairPools, PairTable, PolicySpec, Scratch,
};

mod encode_classes;
mod encode_door;
mod engines;
mod list_room;
mod load_call;
mod membership_call;
mod mode_and_overload;
mod owed;
mod pool_binding;
mod repeated_models;
mod unnamed_rank;
mod written;

const MODEL: &str = "m";
const ASKER: GatewayId = GatewayId(1);

/// A store's write gives each member its slot. The cases here hand a writer
/// rows no store filed and read no count of theirs, so a row is given the
/// next slot of the writer's counts where it is handed.
trait Filed {
    fn filed(&self, writer: &Writer) -> std::vec::IntoIter<(&WorkerRow, Slot)>;
}

impl Filed for [WorkerRow] {
    fn filed(&self, writer: &Writer) -> std::vec::IntoIter<(&WorkerRow, Slot)> {
        let counts = writer.counts();
        let filed: Vec<_> = self.iter().map(|row| (row, counts.claim())).collect();
        filed.into_iter()
    }
}

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

/// The tests' way to a row a decision reads: registered, and the build its
/// write owes asked at once.
fn admit(store: &MemoryStore, facts: WorkerFacts) -> WorkerId {
    let id = store.upsert(facts).id();
    store.publish_now();
    id
}

/// A worker of the model, registered and not yet ready.
fn member(store: &MemoryStore, key: &str) -> WorkerId {
    admit(store, facts(key, Role::Single, &[MODEL]))
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
    lent_agrees(store, ask);
    (out.candidates, out.affinity, built)
}

/// Holds the lent view to the build's, under one load of the world: the
/// rows [`World::view`] hands, with what it hands beside them written in,
/// are the rows [`World::build`] writes, field for field. Every case that
/// builds its view through [`view`] holds the lent one to it here.
fn lent_agrees(store: &MemoryStore, ask: &Ask) {
    let (mut built, mut lent) = (ViewBuffers::default(), ViewBuffers::default());
    store.with_world(|world| {
        let read = world
            .build(MODEL, Role::Single, ask, &mut built)
            .expect("the pool");
        let view = world
            .view(MODEL, Role::Single, ask, false, &mut lent)
            .expect("the pool");
        let mut rows = view.candidates.to_vec();
        for (index, row) in rows.iter_mut().enumerate() {
            assert_eq!(row.position as usize, index);
            // The age at the decision: its instant less the sample's.
            let sampled_at = row.load_sampled_at_micros;
            if let Some(load) = &mut row.load {
                load.report_age_ms = super::age_ms(view.built.now_micros, sampled_at);
            }
        }
        assert!(view.stalled.is_sorted(), "the mask is in the rows' order");
        for &position in view.stalled {
            rows[position as usize].health.stalled = true;
        }
        if let Some(at) = view.pinned_at.checked_sub(1) {
            rows[at as usize].session_hit = true;
        }
        // A row is compared by its printed form: every field of it.
        assert_eq!(format!("{rows:?}"), format!("{:?}", built.candidates));
        let column = format!("{:?}", view.affinity);
        assert_eq!(column, format!("{:?}", built.affinity));
        assert_eq!(view.built, read);
    });
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
    store.publish_now();
    store.upsert(facts("http://o:8000", Role::Single, &["other"]));
    store.publish_now();
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
        admit(
            &store,
            WorkerFacts {
                reachable_from: Reachability::Gateway(GatewayId(gateway)),
                ..facts(key, Role::Single, &[MODEL])
            },
        )
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
fn a_build_names_the_asking_gateways_list_where_the_pool_holds_a_host_local_row() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let open = member(&store, "http://a:8000");
    let built = |gateway: u64| {
        let asked = Ask {
            gateway: GatewayId(gateway),
            ..ask(now)
        };
        let (rows, _, built) = view(&store, &asked);
        let mut ids: Vec<WorkerId> = rows.iter().map(|row| row.id).collect();
        ids.sort_unstable();
        (built.reach, ids)
    };
    // A pool every gateway dials whole is one list: no gateway's is named.
    assert_eq!([built(1), built(2)], [(0, vec![open]), (0, vec![open])]);

    let local = store
        .upsert(WorkerFacts {
            reachable_from: Reachability::Gateway(GatewayId(2)),
            ..facts("unix:///b", Role::Single, &[MODEL])
        })
        .id();
    store.publish_now();
    let mut both = vec![open, local];
    both.sort_unstable();
    // One host-local row, and the pool's lists are two under one epoch:
    // each gateway's is named by its id, the row's holder's and the others'.
    assert_eq!(
        [built(1), built(2), built(3)],
        [(1, vec![open]), (2, both), (3, vec![open])]
    );
}

#[test]
fn a_build_names_the_asking_gateways_list_where_its_veto_masks_a_row() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let vetoed = member(&store, "http://a:8000");
    let reach = |gateway: u64, at: Instant| {
        let asked = Ask {
            gateway: GatewayId(gateway),
            ..ask(at)
        };
        view(&store, &asked).2.reach
    };
    assert_eq!([reach(1, now), reach(2, now)], [0, 0]);

    // A veto masks a row of its holder's copy and of no other: that
    // gateway's list is its own, and the others still read the pool's one.
    let until = now + Duration::from_secs(1);
    store.gateway_veto(GatewayId(2), vetoed, Some(until));
    assert_eq!([reach(1, now), reach(2, now), reach(3, now)], [0, 2, 0]);
    // A veto that ended masks nothing.
    assert_eq!(reach(2, until), 0);
}

#[test]
fn two_gateways_lists_of_one_size_each_turn_their_own_rotation() {
    // Twelve rows every gateway dials and four host-local rows a gateway:
    // each of two gateways is handed sixteen rows under one membership
    // epoch, the other's four left out.
    let store = MemoryStore::new();
    let now = Instant::now();
    let ready = |facts: WorkerFacts| {
        let id = store.upsert(facts).id();
        store.publish_now();
        status(&store, id, WorkerStatus::Ready);
    };
    for row in 0..12 {
        ready(facts(
            &format!("http://w{row}:8000"),
            Role::Single,
            &[MODEL],
        ));
    }
    for gateway in [1, 2] {
        for row in 0..4 {
            ready(WorkerFacts {
                reachable_from: Reachability::Gateway(GatewayId(gateway)),
                ..facts(&format!("unix:///g{gateway}/{row}"), Role::Single, &[MODEL])
            });
        }
    }
    let engine = PolicySpec::preset("round_robin")
        .expect("a preset of this build")
        .validate()
        .expect("a preset validates");
    let (mut out, mut scratch) = (ViewBuffers::default(), Scratch::new());
    let mut drawn = [Vec::new(), Vec::new()];
    // Their asks alternate through one engine, as a controller's do.
    for turn in 0..32u64 {
        let asked = Ask {
            gateway: GatewayId(1 + turn % 2),
            ..ask(now)
        };
        let built = store
            .build_view(MODEL, Role::Single, &asked, &mut out)
            .expect("the pool");
        assert_eq!(out.candidates.len(), 16);
        let request = RequestFacts {
            model: MODEL,
            now_micros: built.now_micros,
            ..RequestFacts::default()
        };
        let decision = engine.decide(
            &request,
            &built.pool_facts(),
            &out.candidates,
            &Constraints::default(),
            &mut scratch,
        );
        assert!(decision.trace().drew_first());
        let pick = decision.pick().expect("a ready row");
        drawn[(turn % 2) as usize].push(pick.targets.iter().next().expect("a target").id);
    }
    // One rotation turned by both lists would hand a gateway every other
    // row of its own: sixteen decisions of a gateway visit its sixteen rows.
    for ids in &mut drawn {
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 16);
    }
}

#[test]
fn two_gateways_in_strict_turn_leave_each_others_vetoed_rows_their_share() {
    // Sixteen rows every gateway dials whole. Two gateways ask in strict
    // turn, and each holds a veto on a row at each parity: whichever
    // parity a gateway's turns land on, one of its vetoes stands there.
    let store = MemoryStore::new();
    let now = Instant::now();
    for row in 0..16 {
        let id = store
            .upsert(facts(
                &format!("http://w{row}:8000"),
                Role::Single,
                &[MODEL],
            ))
            .id();
        store.publish_now();
        status(&store, id, WorkerStatus::Ready);
    }
    let (rows, _, _) = view(&store, &ask(now));
    let ids: Vec<WorkerId> = rows.iter().map(|row| row.id).collect();
    let until = now + Duration::from_secs(60);
    for (at, &id) in ids[..4].iter().enumerate() {
        store.gateway_veto(GatewayId(1 + at as u64 / 2), id, Some(until));
    }
    let engine = PolicySpec::preset("round_robin")
        .expect("a preset of this build")
        .validate()
        .expect("a preset validates");
    let (mut out, mut scratch) = (ViewBuffers::default(), Scratch::new());
    let mut picked = [0u32; 16];
    for turn in 0..224u64 {
        let asked = Ask {
            gateway: GatewayId(1 + turn % 2),
            ..ask(now)
        };
        let built = store
            .build_view(MODEL, Role::Single, &asked, &mut out)
            .expect("the pool");
        let request = RequestFacts {
            model: MODEL,
            now_micros: built.now_micros,
            ..RequestFacts::default()
        };
        let decision = engine.decide(
            &request,
            &built.pool_facts(),
            &out.candidates,
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("a ready row");
        let id = pick.targets.iter().next().expect("a target").id;
        let at = ids.iter().position(|&row| row == id).expect("a row");
        picked[at] += 1;
    }
    // Each gateway turns its own rotation: its 112 asks hand each of its
    // fourteen kept rows eight. A row one gateway vetoes is the other's
    // alone. One rotation turned by both would leave a row vetoed at its
    // vetoer's parity one pick, the other gateway never landing on it.
    let mut want = [16u32; 16];
    want[..4].fill(8);
    assert_eq!(picked, want);
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
        store.publish_now();
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

/// The rows of `model`'s single pool as `world` lists them, under a count
/// of the caller's own.
fn rows_of(world: &World, model: &str) -> Arc<[Candidate]> {
    let index = world.derived.by_model[model][super::role_slot(Role::Single)].expect("the pool");
    Arc::clone(&world.rows[index as usize])
}

#[test]
fn two_publications_of_one_membership_are_two_worlds_of_one_epoch_sharing_what_it_derived() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let one = store
        .upsert(facts("http://a:8000", Role::Single, &["one"]))
        .id();
    store.publish_now();
    store.upsert(facts("http://b:8000", Role::Single, &["two"]));
    store.publish_now();
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
    assert!(Arc::ptr_eq(
        &rows_of(&before, "two"),
        &rows_of(&after, "two")
    ));
    assert!(!Arc::ptr_eq(
        &rows_of(&before, "one"),
        &rows_of(&after, "one")
    ));
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

/// What `id`'s row reads as running in the store's world.
fn running_of(store: &MemoryStore, id: WorkerId, now: Instant) -> Option<u32> {
    let (rows, _, _) = view(store, &ask(now));
    let row = rows.iter().find(|row| row.id == id).expect("the row");
    row.load.map(|load| load.running)
}

#[test]
fn a_pool_no_reader_holds_is_published_into_two_buffers_in_turn() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    member(&store, "http://b:8000");
    // Each record lands a window after the last and publishes by itself.
    let publish = |turn: u32| {
        let at = t0 + PUBLISH_WINDOW * turn;
        store.record(id, 0, load(turn, 0), at, LoadSource::Stream);
        assert_eq!(running_of(&store, id, at), Some(turn));
    };
    // Where the store's world lists the pool's rows, no hold kept on them.
    let listed = || rows_of(&store.world(), MODEL).as_ptr();
    let derived = listed();
    assert_eq!(
        store.publication_allocations(),
        0,
        "a membership write is not counted"
    );

    // The pool's first publication has no earlier rows to write over.
    publish(1);
    let second = listed();
    assert_ne!(second, derived);
    assert_eq!(store.publication_allocations(), 1);
    // Each one after it writes over the rows published before the last:
    // two buffers in turn, and nothing allocated.
    for turn in 2..=7 {
        publish(turn);
        let written = if turn % 2 == 0 { derived } else { second };
        assert_eq!(listed(), written, "turn {turn}");
    }
    assert_eq!(store.publication_allocations(), 1);

    // A membership write derives its pools anew, uncounted, and each pool
    // starts again without earlier rows. The rows it retired are the
    // writer's to free: the spare at once, the published at the next
    // publication, when the world that listed them has left the store.
    member(&store, "http://c:8000");
    assert_eq!((store.publication_allocations(), store.held_rows()), (1, 1));
    publish(8);
    assert_eq!((store.publication_allocations(), store.held_rows()), (2, 0));
    publish(9);
    publish(10);
    assert_eq!(store.publication_allocations(), 2);
}

#[test]
fn a_pool_written_once_keeps_its_rows_while_another_pool_publishes() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let one = store
        .upsert(facts("http://a:8000", Role::Single, &["one"]))
        .id();
    store.publish_now();
    let two = store
        .upsert(facts("http://b:8000", Role::Single, &["two"]))
        .id();
    store.publish_now();
    // One record into the first pool, published by itself.
    store.record(one, 0, load(1, 0), t0, LoadSource::Stream);
    let listed = |model: &str| rows_of(&store.world(), model).as_ptr();
    let written = listed("one");
    // The other pool publishes five times, a window apart. The first pool
    // moved in none of them: it is not published again, and the world
    // lists the rows it listed.
    for turn in 1..=5 {
        let at = t0 + PUBLISH_WINDOW * turn;
        store.record(two, 0, load(turn, 0), at, LoadSource::Stream);
        assert_eq!(running_in(&store.world(), "two", at), Some(turn));
        assert_eq!(listed("one"), written, "turn {turn}");
    }
    // The first pool was written over the rows the second registration
    // retired, and the other allocated at its own first publication alone.
    assert_eq!(store.publication_allocations(), 1);
}

#[test]
fn rows_a_reader_still_holds_are_left_whole_and_that_publication_allocates() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    let publish = |turn: u32| {
        let at = t0 + PUBLISH_WINDOW * turn;
        store.record(id, 0, load(turn, 0), at, LoadSource::Stream);
        assert_eq!(running_of(&store, id, at), Some(turn));
    };
    let counts = || (store.publication_allocations(), store.held_rows());
    publish(1);
    assert_eq!(counts(), (1, 0));

    // A reader takes the world of the first publication and keeps it
    // across two publications of its pool.
    let kept = store.world();
    let read = format!("{:?}", rows_of(&kept, MODEL));
    publish(2);
    assert_eq!(counts(), (1, 0));
    // The second would write over the rows that world lists: it leaves
    // them whole, allocates and is counted.
    publish(3);
    assert_eq!(counts(), (2, 1));
    assert_eq!(running_in(&kept, MODEL, t0), Some(1));
    assert_eq!(format!("{:?}", rows_of(&kept, MODEL)), read);

    // One world held is one buffer more: the two others go on in turn.
    publish(4);
    publish(5);
    assert_eq!(counts(), (2, 1));
    assert_eq!(format!("{:?}", rows_of(&kept, MODEL)), read);

    // The writer kept its own reference to the rows it left and its own
    // hold on the world that lists them, so the reader dropping its hold
    // frees nothing: three counts stay, the writer's, that world's and
    // this case's. The next publication lets the world go.
    let rows = rows_of(&kept, MODEL);
    drop(kept);
    assert_eq!(Arc::strong_count(&rows), 3);
    publish(6);
    assert_eq!((counts(), Arc::strong_count(&rows)), ((2, 1), 2));
    // The publication that finds them alone frees them.
    drop(rows);
    publish(7);
    assert_eq!(counts(), (2, 0));

    // A membership write keeps its outgoing rows the same way, uncounted:
    // the spare for its reader, the published rows for the world the store
    // showed till then. The pool's next publication frees both and
    // allocates.
    let kept = store.world();
    publish(8);
    member(&store, "http://b:8000");
    assert_eq!(counts(), (2, 2));
    let rows = rows_of(&kept, MODEL);
    drop(kept);
    assert_eq!(Arc::strong_count(&rows), 3);
    drop(rows);
    publish(9);
    assert_eq!(counts(), (3, 0));
}

#[test]
fn a_pool_with_no_spare_to_write_over_takes_rows_the_writer_holds_alone() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    let publish = |turn: u32| {
        let at = t0 + PUBLISH_WINDOW * turn;
        store.record(id, 0, load(turn, 0), at, LoadSource::Stream);
        assert_eq!(running_of(&store, id, at), Some(turn));
    };
    let counts = || (store.publication_allocations(), store.held_rows());
    publish(1);
    // A reader that holds the worlds of the last two publications: each
    // publication finds its pool's spare listed by the older one.
    let mut kept = [store.world(), store.world()];
    publish(2);
    kept[0] = store.world();
    // The first such publication has no other rows to take.
    publish(3);
    assert_eq!(counts(), (2, 1));
    // From then on the rows the reader let go come back in turn: three
    // buffers carry every window and nothing is allocated.
    for turn in 4..=9 {
        kept[(turn % 2) as usize] = store.world();
        publish(turn);
        assert_eq!(counts(), (2, 1), "turn {turn}");
    }
    drop(kept);
    publish(10);
    assert_eq!(counts(), (2, 0));

    // A membership write that leaves the pool its length: the pool's next
    // publication takes the rows the write retired.
    store.upsert(facts("http://z:8000", Role::Single, &["other"]));
    store.publish_now();
    assert_eq!(counts(), (2, 1));
    publish(11);
    assert_eq!(counts(), (2, 0));
}

#[test]
fn one_publication_holds_every_pools_folds_whichever_write_made_it() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let one = store
        .upsert(facts("http://a:8000", Role::Single, &["one"]))
        .id();
    store.publish_now();
    let two = store
        .upsert(facts("http://b:8000", Role::Single, &["two"]))
        .id();
    store.publish_now();
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
    store.publish_now();
    let decode = store
        .upsert(facts("http://d:8000", Role::Decode, &[MODEL]))
        .id();
    store.publish_now();
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
        &[(id, 0, Some(load(4, 0))), (id, 1, Some(load(5, 0)))],
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
        &[
            (id, 0, Some(load(4, 0))),
            (id, 1, Some(load(5, 0))),
            (id, 2, Some(load(6, 0))),
        ],
        t0,
        LoadSource::Poll,
    );
    assert_eq!(running(&store, t0), Some(4));
    // Inside the window that opened, the next one waits and the sink names
    // its instant.
    store.report(
        &[(id, 0, Some(load(7, 0))), (id, 1, Some(load(8, 0)))],
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
    store.publish_now();
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
    store.publish_now();
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
    store.publish_now();
    let again = member(&store, "http://b:8000");
    assert_eq!(again, other, "the same key and incarnation is the same id");
    assert_eq!(stalled(ASKER, t0), []);
}

/// The noted write shows the world what the plain one shows: a veto set
/// with no note behind it masks the worker for its gateway alone, a later
/// end behind a note that stands is the end shown, and a noted clear lifts
/// the mask.
#[test]
fn a_noted_veto_is_shown_and_lifted_in_the_published_world() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let vetoed = member(&store, "http://a:8000");
    let until = t0 + Duration::from_secs(1);
    let written = store.gateway_veto_noted(ASKER, vetoed, Some(until), false, t0);
    assert_eq!(written, VetoWrite::Written);
    assert_eq!(stalled(&store, ASKER, t0), [vetoed]);
    assert_eq!(stalled(&store, GatewayId(2), t0), []);
    assert_eq!(stalled(&store, ASKER, until), []);

    let later = until + Duration::from_secs(1);
    let written = store.gateway_veto_noted(ASKER, vetoed, Some(later), true, t0);
    assert_eq!(written, VetoWrite::Written);
    assert_eq!(stalled(&store, ASKER, until), [vetoed]);

    let written = store.gateway_veto_noted(ASKER, vetoed, None, true, t0);
    assert_eq!(written, VetoWrite::Written);
    assert_eq!(stalled(&store, ASKER, t0), []);
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
    store.publish_now();
    let single = store
        .upsert(facts("http://b:8000", Role::Single, &["two"]))
        .id();
    store.publish_now();
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
    store.publish_now();
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
        reachable_from: Reachability::Gateway(ASKER),
        ..facts(key, Role::Decode, &[MODEL])
    };
    let id = store.upsert(registered.clone()).id();
    store.publish_now();
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
    // A decode with no prefill of its model is outside any pairing.
    assert_eq!(row.pair_class, 0);
    assert!(row.host_local);
    // The low 32 bits of the registration key's fingerprint.
    assert_eq!(row.key, fingerprint(key.as_bytes()) as u32);
    // What no fact of the store feeds yet reads as the row's default.
    assert_eq!(
        (row.in_flight, row.session_in_flight, row.active_sessions),
        (0, 0, 0)
    );
    assert!(!row.has_capacity && !row.has_room && !row.health.overloaded);
    assert_eq!(row.connector, 0);
    // Under a generation that declares no list the listed fields read 0.
    assert_eq!((row.zone, row.labels, row.engine_kind), (0, 0, 0));
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
    store.publish_now();
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
    store.publish_now();
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
    assert!(store.connected(id, revision()).moved().is_some());
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
            store.publish_now();
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
    store.publish_now();
    store.upsert(facts("http://b:8000", Role::Single, &["one"]));
    store.publish_now();
    let two = store
        .upsert(facts("http://c:8000", Role::Single, &["two"]))
        .id();
    store.publish_now();
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
        admit(
            &store,
            WorkerFacts {
                reachable_from,
                ..facts(key, Role::Single, &[MODEL])
            },
        )
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
    store.publish_now();
    let decode = store
        .upsert(facts("http://d:8000", Role::Decode, &[MODEL]))
        .id();
    store.publish_now();
    let mut out = PairBuffers::default();
    let (left, right) = store
        .build_pair(MODEL, &ask(t0), &mut out)
        .expect("both pools");
    assert_eq!((left.pool_size, right.pool_size), (1, 1));
    assert_eq!(left.membership_epoch, right.membership_epoch);
    assert_eq!(
        (out.pools().prefills[0].id, out.pools().decodes[0].id),
        (prefill, decode)
    );
    assert!(!out.pools().table.is_empty());
    // A model served on one side alone has no pair, and no buffer keeps a
    // view: not the side that has members, not the last pair's; neither
    // its rows, nor its table, nor, under an ask that declares it, its
    // column.
    let with_column = Ask {
        affinity: true,
        ..ask(t0)
    };
    let empty = |out: &PairBuffers| {
        let pools = out.pools();
        pools.prefills.is_empty()
            && pools.prefill_affinity.is_empty()
            && pools.decodes.is_empty()
            && pools.decode_affinity.is_empty()
            && pools.table.is_empty()
    };
    store.upsert(facts("http://s:8000", Role::Prefill, &["prefill-alone"]));
    store.publish_now();
    assert_eq!(
        store.build_pair("prefill-alone", &with_column, &mut out),
        None
    );
    assert!(empty(&out));
    store
        .build_pair(MODEL, &with_column, &mut out)
        .expect("both pools");
    assert_eq!(
        (
            out.pools().prefill_affinity.len(),
            out.pools().decode_affinity.len()
        ),
        (1, 1)
    );
    store.upsert(facts("http://t:8000", Role::Decode, &["decode-alone"]));
    store.publish_now();
    assert_eq!(
        store.build_pair("decode-alone", &with_column, &mut out),
        None
    );
    assert!(empty(&out));
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
    store.publish_now();
    let decode = store
        .upsert(facts("http://d:8000", Role::Decode, &[MODEL]))
        .id();
    store.publish_now();
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
    let mut out = PairBuffers::default();
    let mut sides = || {
        store
            .build_pair(MODEL, &ask(t0), &mut out)
            .expect("both pools");
        let running = |side: &[Candidate]| side[0].load.map(|load| load.running);
        (running(out.pools().prefills), running(out.pools().decodes))
    };
    // The case ends at an event, the door read across `SEEN` publications,
    // and holds the door's rule at every read: one load of the world feeds
    // both sides, so a publication between the two builds shows in neither.
    // A store that stopped publishing fails the case at this bound: without
    // it the door would be read for ever.
    let bound = Instant::now() + Duration::from_secs(60);
    let (mut seen, mut last) = (0, None);
    while seen < SEEN {
        assert!(
            !writer.is_finished(),
            "the writer ended before the door was read across {SEEN} publications"
        );
        assert!(
            Instant::now() < bound,
            "the door was read across {seen} publications of {SEEN} in a minute"
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
        admit(
            &store,
            WorkerFacts {
                engine: "grpc://e:9000".to_string(),
                dp_rank: rank,
                dp_size: 2,
                ..facts(&format!("grpc://e:9000@{rank}"), Role::Single, &[MODEL])
            },
        )
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
        maker: Default::default(),
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
    writer.rebuild(1, members.filed(&writer), |_, _| None, Classes::Derive);
    let vetoed = writer
        .veto(ASKER, leaving, Some(t0 + Duration::from_secs(60)))
        .expect("a new veto");
    assert_eq!(masked(&vetoed), [leaving]);
    // Another worker's load write publishes while the removal is under
    // way: the leaving worker's row still stands in that world, masked.
    let between = writer.publish_window(t0);
    assert_eq!(masked(&between), [leaving]);
    // The membership write takes the row and its vetoes out together,
    let after = writer.rebuild(2, members[1..].filed(&writer), |_, _| None, Classes::Derive);
    assert_eq!(after.members(MODEL, Role::Single), Some(1));
    assert_eq!(masked(&after), []);
    // and the same id registered again starts without them.
    let again = writer.rebuild(3, members.filed(&writer), |_, _| None, Classes::Derive);
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
    store.publish_now();
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

#[test]
fn each_of_many_gateways_is_masked_by_its_own_vetoes_alone() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + Duration::from_secs(60);
    let mut workers: Vec<WorkerId> = (0..24)
        .map(|n| member(&store, &format!("http://w{n}:8000")))
        .collect();
    workers.sort_unstable();
    // Twenty-four gateways whose ids follow no order of their writes, each
    // with three vetoes; a worker is vetoed by three of them.
    let gateways: Vec<GatewayId> = (0..24).map(|n| GatewayId(n * 856 % 1_009 + 2)).collect();
    let own = |n: usize| {
        let mut own: Vec<WorkerId> = (0..3).map(|step| workers[(n + step * 5) % 24]).collect();
        own.sort_unstable();
        own
    };
    for (n, &gateway) in gateways.iter().enumerate().rev() {
        for id in own(n) {
            store.gateway_veto(gateway, id, Some(until));
        }
    }
    let each_reads_its_own = || {
        for (n, &gateway) in gateways.iter().enumerate() {
            assert_eq!(stalled(&store, gateway, t0), own(n), "gateway {n}");
        }
        // A gateway that holds no veto reads none among the others'.
        assert_eq!(stalled(&store, ASKER, t0), []);
    };
    // As each veto's write left the runs,
    each_reads_its_own();
    // as a membership write builds every run again,
    member(&store, "http://joined:8000");
    each_reads_its_own();
    // and after one gateway cleared its own: the others' stand.
    for id in own(7) {
        store.gateway_veto(gateways[7], id, None);
    }
    assert_eq!(stalled(&store, gateways[7], t0), []);
    assert_eq!(stalled(&store, gateways[6], t0), own(6));
    assert_eq!(stalled(&store, gateways[8], t0), own(8));
}

#[test]
fn a_gateways_run_is_found_by_the_walk_and_by_the_search_and_none_where_it_holds_none() {
    let until = Instant::now() + Duration::from_secs(60);
    // Runs of gateways three apart, on both sides of the count a build
    // walks: an asker stands under them, between two and above them. Each
    // run's one veto names its own place.
    for count in [
        0,
        1,
        2,
        WALKED_RUNS - 1,
        WALKED_RUNS,
        WALKED_RUNS + 1,
        3 * WALKED_RUNS,
    ] {
        let runs: Vec<GatewayRun> = (0..count)
            .map(|place| GatewayRun {
                gateway: GatewayId(3 * place as u64 + 3),
                vetoes: Arc::from([PoolVeto {
                    row: place as u32,
                    id: WorkerId(place as u64),
                    until,
                }]),
            })
            .collect();
        for asker in 0..=3 * count as u64 + 4 {
            let found = run_of(&runs, GatewayId(asker)).map(|run| run.vetoes[0].row);
            let holds = asker >= 3 && asker % 3 == 0 && asker / 3 <= count as u64;
            let held = holds.then(|| (asker / 3 - 1) as u32);
            assert_eq!(found, held, "{count} runs, gateway {asker}");
        }
    }
}

#[test]
fn one_worker_vetoed_by_every_gateway_is_masked_for_each_whatever_their_count() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + Duration::from_secs(60);
    let mut workers: Vec<WorkerId> = (0..4)
        .map(|n| member(&store, &format!("http://w{n}:8000")))
        .collect();
    workers.sort_unstable();
    // Gateways two apart. Each vetoes the one sick worker and, by its own
    // number, one of the three others: a build that took a neighbour's run
    // would mask another row.
    let gateway = |n: u64| GatewayId(2 * n + 2);
    let own = |n: u64| {
        let mut own = vec![workers[0], workers[1 + (n % 3) as usize]];
        own.sort_unstable();
        own
    };
    let walked = WALKED_RUNS as u64;
    let mut holders = 0;
    // Few gateways, as many as a build walks, one more, and many.
    for count in [1, 9, walked, walked + 1, 2 * walked] {
        for n in holders..count {
            for id in own(n) {
                store.gateway_veto(gateway(n), id, Some(until));
            }
        }
        holders = count;
        for n in 0..count {
            assert_eq!(
                stalled(&store, gateway(n), t0),
                own(n),
                "{count} gateways, gateway {n}"
            );
            // The gateway under each holder holds none.
            assert_eq!(
                stalled(&store, GatewayId(2 * n + 1), t0),
                [],
                "{count} gateways, under gateway {n}"
            );
        }
        assert_eq!(
            stalled(&store, GatewayId(2 * count + 3), t0),
            [],
            "{count} gateways, above them"
        );
    }
}

/// Every pool's runs as plain values: per gateway its entries.
fn runs_of(writer: &Writer) -> Vec<Vec<(GatewayId, Vec<PoolVeto>)>> {
    writer
        .pool_vetoes
        .iter()
        .map(|runs| {
            runs.iter()
                .map(|run| (run.gateway, run.vetoes.to_vec()))
                .collect()
        })
        .collect()
}

/// A member row of the writer's own cases: worker `id`, ready, serving
/// `models`.
fn row_serving(id: u64, models: &[&str], admitted_at: Instant) -> WorkerRow {
    WorkerRow {
        id: WorkerId(id),
        facts: facts(&format!("http://w{id}:8000"), Role::Single, models),
        status: WorkerStatus::Ready,
        revision: 1,
        admitted_at,
        maker: Default::default(),
    }
}

#[test]
fn a_spare_of_another_length_is_left_and_the_publication_allocates() {
    let origin = Instant::now();
    let members = [
        row_serving(1, &[MODEL], origin),
        row_serving(2, &[MODEL], origin),
    ];
    let mut writer = Writer::new(PUBLISH_WINDOW, origin);
    writer.rebuild(1, members.filed(&writer), |_, _| None, Classes::Derive);
    // No write leaves a pool a spare of another length: one is planted,
    // the rows of a pool of one, held by the pool alone.
    writer.spares[0] = Some(Arc::from(&writer.working[0].rows[..1]));
    let world = writer
        .fold_status(WorkerId(1), WorkerStatus::Draining)
        .expect("a publication");
    assert_eq!(world.rows[0].len(), 2);
    assert!(world.rows[0].iter().any(|row| row.health.draining));
    assert_eq!((writer.allocations, writer.held_rows()), (1, 1));
    // The next publication finds the planted rows free of every world and
    // of no pool's length: it frees them and writes over the pool's own.
    writer
        .fold_status(WorkerId(1), WorkerStatus::Ready)
        .expect("a publication");
    assert_eq!((writer.allocations, writer.held_rows()), (1, 0));
}

#[test]
fn a_membership_write_leaves_the_writers_lists_at_one_slot_a_pool() {
    let origin = Instant::now();
    let members = [
        row_serving(1, &[MODEL, "other"], origin),
        row_serving(2, &[MODEL, "other"], origin),
    ];
    let mut writer = Writer::new(PUBLISH_WINDOW, origin);
    for epoch in 1..=4 {
        writer.rebuild(epoch, members.filed(&writer), |_, _| None, Classes::Derive);
        // Two models, a pool each: one slot a pool in each list, however
        // many membership writes came before.
        assert_eq!(
            (
                writer.spares.len(),
                writer.published.len(),
                writer.working.len()
            ),
            (2, 2, 2),
            "epoch {epoch}"
        );
        // Both pools publish, so each holds a spare for the next membership
        // write to retire.
        writer
            .fold_status(WorkerId(1), WorkerStatus::Draining)
            .expect("a publication");
    }
}

#[test]
fn a_vetos_write_builds_its_own_gateways_run_and_shares_every_other() {
    let origin = Instant::now();
    let until = origin + Duration::from_secs(60);
    // Eight workers of the model; the first four serve a second model too.
    let members: Vec<WorkerRow> = (1..=8)
        .map(|id| {
            let models: &[&str] = if id <= 4 {
                &[MODEL, "second"]
            } else {
                &[MODEL]
            };
            row_serving(id, models, origin)
        })
        .collect();
    let mut writer = Writer::new(PUBLISH_WINDOW, origin);
    writer.rebuild(1, members.filed(&writer), |_, _| None, Classes::Derive);
    let (writing, other) = (GatewayId(3), GatewayId(7));
    for (gateway, id) in [(other, 2), (other, 6), (writing, 3)] {
        writer
            .veto(gateway, WorkerId(id), Some(until))
            .expect("a new veto");
    }
    let run = |runs: &PoolVetoes, gateway: GatewayId| {
        runs.iter()
            .find(|run| run.gateway == gateway)
            .map(|run| Arc::clone(&run.vetoes))
    };
    // One more veto of the writing gateway, on a worker of the first pool
    // alone.
    let before = Arc::clone(&writer.pool_vetoes);
    writer
        .veto(writing, WorkerId(7), Some(until))
        .expect("a new veto");
    let after = Arc::clone(&writer.pool_vetoes);
    let mut whole = 0;
    for (was, is) in before.iter().zip(after.iter()) {
        if Arc::ptr_eq(was, is) {
            // The pool the worker has no row in: not built again at all.
            whole += 1;
            continue;
        }
        // In the worker's pool the writing gateway's run is a new one with
        // the entry, and the other gateway's is the allocation it was.
        let (old, new) = (run(was, other), run(is, other));
        assert!(old
            .zip(new)
            .is_some_and(|(old, new)| Arc::ptr_eq(&old, &new)));
        assert_eq!(run(is, writing).map(|run| run.len()), Some(2));
    }
    assert_eq!((before.len(), whole), (2, 1));
    // The clear takes the entry out the same way; the last clear of a
    // gateway takes its run out of the pool, and the last of the store
    // leaves no list at all.
    let before = Arc::clone(&writer.pool_vetoes);
    writer.veto(writing, WorkerId(7), None).expect("a clear");
    let after = Arc::clone(&writer.pool_vetoes);
    assert_eq!(
        before
            .iter()
            .zip(after.iter())
            .filter(|(was, is)| Arc::ptr_eq(was, is))
            .count(),
        1
    );
    writer.veto(writing, WorkerId(3), None).expect("a clear");
    assert!(writer
        .pool_vetoes
        .iter()
        .all(|runs| run(runs, writing).is_none()));
    for id in [2, 6] {
        writer.veto(other, WorkerId(id), None).expect("a clear");
    }
    assert!(writer.pool_vetoes.is_empty());
}

#[test]
fn the_runs_each_write_leaves_are_the_runs_a_derive_builds_again() {
    let origin = Instant::now();
    let at = |ms: u64| origin + Duration::from_millis(ms);
    // Twelve workers of the model; each third one serves a second model.
    let all: Vec<WorkerRow> = (1..=12)
        .map(|id| {
            let models: &[&str] = if id % 3 == 0 {
                &[MODEL, "second"]
            } else {
                &[MODEL]
            };
            row_serving(id, models, origin)
        })
        .collect();
    let mut present = [true; 12];
    let mut writer = Writer::new(PUBLISH_WINDOW, origin);
    writer.rebuild(1, all.filed(&writer), |_, _| None, Classes::Derive);
    let counts = writer.counts();
    // A fixed sequence: the case makes the same writes at every run.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let mut clock = 1_000;
    let mut standing_most = 0;
    for step in 0..3_000_u64 {
        let (gateway, worker) = (GatewayId(next(5) * 11 + 2), WorkerId(next(12) + 1));
        match next(16) {
            // A membership write: one worker joins or leaves.
            0 => {
                let n = next(12) as usize;
                present[n] = !present[n];
                let members = all
                    .iter()
                    .zip(present)
                    .filter_map(|(row, here)| here.then_some((row, counts.claim())));
                writer.rebuild(step + 2, members, |_, _| None, Classes::Derive);
            }
            // A window's publication, which drops what its instant ended.
            1 => {
                clock += next(40);
                writer.publish_window(at(clock));
            }
            // A clear, of a veto that stands or of none.
            2..=5 => drop(writer.veto(gateway, worker, None)),
            // A veto, new or with a new end.
            _ => drop(writer.veto(gateway, worker, Some(at(clock + 1 + next(60))))),
        }
        let left = runs_of(&writer);
        for runs in &left {
            assert!(
                runs.windows(2).all(|pair| pair[0].0 < pair[1].0),
                "step {step}: a pool's runs stand by gateway"
            );
            for (_, run) in runs {
                assert!(!run.is_empty(), "step {step}: no run is empty");
                assert!(
                    run.windows(2).all(|pair| pair[0].row < pair[1].row),
                    "step {step}: a run stands by row"
                );
            }
        }
        let standing: usize = left.iter().flatten().map(|(_, run)| run.len()).sum();
        standing_most = standing_most.max(standing);
        writer.derive_vetoes();
        assert_eq!(left, runs_of(&writer), "step {step}");
    }
    // The sequence held vetoes of several gateways in both pools at once.
    assert!(standing_most >= 12, "{standing_most}");
    // Then every standing veto is cleared, one write each: each clear
    // leaves what a derive builds, the last one no list at all.
    for gateway in (0..5).map(|n| GatewayId(n * 11 + 2)) {
        for worker in (1..=12).map(WorkerId) {
            if writer.veto(gateway, worker, None).is_none() {
                continue;
            }
            let left = runs_of(&writer);
            writer.derive_vetoes();
            assert_eq!(left, runs_of(&writer), "{gateway:?} clears {worker:?}");
        }
    }
    assert!(writer.pool_vetoes.is_empty());
}

#[test]
fn a_seek_from_any_place_finds_what_a_search_of_the_whole_list_finds() {
    // Ids with gaps and repeats, in a list longer than the steps on foot.
    let ids = [
        1, 1, 3, 4, 4, 4, 7, 9, 10, 12, 15, 15, 18, 21, 22, 25, 30, 31, 31, 40,
    ]
    .map(WorkerId);
    for from in 0..=ids.len() {
        for id in (0..=41).map(WorkerId) {
            let whole = ids.partition_point(|&at| at < id).max(from);
            assert_eq!(seek(&ids, from, id, |&at| at), whole, "from {from}, {id:?}");
        }
    }
}

/// A document that validates, under the generation `number`.
fn generation(number: u64) -> AppliedSpec {
    generation_pairing(number, Pairing::default())
}

/// A document that validates, under the generation `number`, whose pair
/// plan names the pairing `mode`.
fn generation_pairing(number: u64, mode: Pairing) -> AppliedSpec {
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = number;
    document.plans.pd_pair.pairing = mode;
    AppliedSpec::validate(&document).expect("a preset validates")
}

/// A store under a first generation whose document names the pairing
/// `mode`: the one way a store takes a mode.
fn store_pairing(mode: Pairing) -> MemoryStore {
    let store = MemoryStore::new();
    store
        .apply(generation_pairing(1, mode))
        .expect("the first generation");
    store
}

#[test]
fn a_store_starts_under_the_unbound_generation_which_binds_nothing() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    member(&store, "http://a:8000");
    let world = store.world();
    assert_eq!(world.spec().generation(), 0);
    for phase in [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode] {
        assert!(world.spec().engine().binding(MODEL, phase).is_none());
    }
    // A build under it reads the same number with its rows.
    let built = view(&store, &ask(t0)).2;
    assert_eq!((built.generation, built.pool_facts().generation), (0, 0));
}

#[test]
fn a_generation_is_applied_above_the_standing_one_and_no_other() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    member(&store, "http://a:8000");
    // A document that names no generation is number 0: not above the
    // unbound start, which stays the only generation 0.
    assert_eq!(
        store.apply(generation(0)),
        Err(StaleGeneration {
            standing: 0,
            offered: 0
        })
    );
    store.apply(generation(3)).expect("above the unbound start");
    let built = view(&store, &ask(t0)).2;
    assert_eq!(
        (
            built.generation,
            built.pool_facts().generation,
            built.pool_size
        ),
        (3, 3, 1)
    );
    let standing = store.world();
    assert_eq!(standing.spec().generation(), 3);
    assert!(standing
        .spec()
        .engine()
        .binding(MODEL, Phase::Single)
        .is_some());
    // The number only rises: the standing one again and any under it are
    // refused by naming the standing one, and nothing is published.
    for offered in [3, 2, 0] {
        let refused = store.apply(generation(offered)).expect_err("not above");
        assert_eq!(
            refused,
            StaleGeneration {
                standing: 3,
                offered
            }
        );
        assert!(refused.to_string().ends_with("the standing generation 3"));
    }
    assert!(Arc::ptr_eq(&standing, &store.world()));
    store.apply(generation(4)).expect("above the standing one");
    assert_eq!(view(&store, &ask(t0)).2.generation, 4);
}

#[test]
fn a_generation_outlives_its_last_world_until_the_next_apply_hands_it_back() {
    let mut writer = Writer::new(PUBLISH_WINDOW, Instant::now());
    let first = Arc::new(generation(1));
    let held = Arc::downgrade(&first);
    let (world, before) = writer
        .apply(first, 0, std::iter::empty(), |_, _| None)
        .expect("above the unbound start");
    assert!(before.is_none(), "nothing was kept before the first apply");
    // A reader still holds a world of generation 1 when the next is
    // applied and the cell lets that world go; the unbound start, kept
    // since the first apply, is handed back.
    let reader = Arc::clone(&world);
    let (_, before) = writer
        .apply(Arc::new(generation(2)), 0, std::iter::empty(), |_, _| None)
        .expect("above the standing one");
    assert_eq!(before.map(|spec| spec.generation()), Some(0));
    writer.replaced(world);
    // The reader lets its world go: the writer keeps the generation, and
    // the world too until a publication lets it go, so the reader frees
    // no engine. Both holds are the writer's.
    drop(reader);
    assert_eq!(held.strong_count(), 2, "kept by the writer alone");
    // The next apply lets the world go and hands the generation back, and
    // the caller's drop is its end.
    let (_, before) = writer
        .apply(Arc::new(generation(3)), 0, std::iter::empty(), |_, _| None)
        .expect("above the standing one");
    let before = before.expect("the generation before the last");
    assert_eq!((before.generation(), held.strong_count()), (1, 1));
    drop(before);
    assert_eq!(held.strong_count(), 0);
}

#[test]
fn a_world_held_across_two_applies_takes_its_generation_to_a_publication() {
    let mut writer = Writer::new(PUBLISH_WINDOW, Instant::now());
    let first = Arc::new(generation(1));
    let held = Arc::downgrade(&first);
    let (world, _) = writer
        .apply(first, 0, std::iter::empty(), |_, _| None)
        .expect("above the unbound start");
    // A reader holds the world the cell lets go at the next apply.
    let reader = Arc::clone(&world);
    let applied = writer.apply(Arc::new(generation(2)), 0, std::iter::empty(), |_, _| None);
    drop(applied.expect("above the standing one"));
    writer.replaced(world);
    let applied = writer.apply(Arc::new(generation(3)), 0, std::iter::empty(), |_, _| None);
    drop(applied.expect("above the standing one"));
    // The second apply handed the generation back and its caller let it
    // go: the reader's world alone holds it now.
    assert_eq!(held.strong_count(), 1);
    // The reader frees no engine all the same: the writer lists its world,
    // and the engine goes with it at the publication that finds no reader
    // holding it, here the next apply's.
    drop(reader);
    assert_eq!(held.strong_count(), 1);
    let applied = writer.apply(Arc::new(generation(4)), 0, std::iter::empty(), |_, _| None);
    drop(applied.expect("above the standing one"));
    assert_eq!(held.strong_count(), 0);
}

#[test]
fn an_applied_generation_is_published_at_once_with_the_rows_as_they_stand() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    status(&store, id, WorkerStatus::Ready);
    store.gateway_veto(ASKER, id, Some(t0 + Duration::from_secs(60)));
    // The first record opens a window; the second waits inside it.
    store.record(id, 0, load(1, 0), t0, LoadSource::Stream);
    store.record(
        id,
        0,
        load(2, 0),
        t0 + Duration::from_millis(1),
        LoadSource::Stream,
    );
    assert_eq!(running(&store, t0), Some(1));
    let epoch = store.world().epoch();
    store.apply(generation(1)).expect("above the unbound start");
    // One publication: the generation, the membership and its veto as they
    // stood, and the record that waited.
    let (rows, _, built) = view(&store, &ask(t0));
    assert_eq!((built.generation, built.membership_epoch), (1, epoch));
    assert_eq!(rows[0].load.map(|load| load.running), Some(2));
    assert!(rows[0].health.healthy && rows[0].health.stalled);
}

#[test]
fn the_pair_door_builds_no_decode_side_beside_an_empty_prefill_pool() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let _decode = store.upsert(facts("http://d:8000", Role::Decode, &[MODEL]));
    store.publish_now();
    let mut out = PairBuffers::default();
    assert_eq!(store.build_pair(MODEL, &ask(t0), &mut out), None);
    // The decode pool has a member and its buffer never grew: the door
    // stopped at the prefill pool it did not find.
    assert_eq!(out.decode.candidates.capacity(), 0);
    let _prefill = store.upsert(facts("http://p:8000", Role::Prefill, &[MODEL]));
    store.publish_now();
    assert!(store.build_pair(MODEL, &ask(t0), &mut out).is_some());
    assert_eq!(
        (out.prefill.candidates.len(), out.decode.candidates.len()),
        (1, 1)
    );
}

/// One decision's view of one side of the model's pair.
fn side(store: &MemoryStore, role: Role, ask: &Ask) -> (Vec<Candidate>, Vec<AffinityFacts>, Built) {
    let mut out = ViewBuffers::default();
    let built = store
        .build_view(MODEL, role, ask, &mut out)
        .expect("the pool");
    (out.candidates, out.affinity, built)
}

/// A prefill or decode of the model on the runtime-specific side, of one
/// known runtime, that names its KV transport or none.
fn pair_facts(key: &str, role: Role, transport: Option<&str>) -> WorkerFacts {
    WorkerFacts {
        wire_modes: WireModes::GRPC,
        engine_kind: "rt-a".to_string(),
        pairing: transport.map(|transport| {
            Arc::new(PairingDescriptor {
                transport: Some(transport.to_string()),
                ..PairingDescriptor::default()
            })
        }),
        ..facts(key, role, &[MODEL])
    }
}

/// The id and the class of each row, in the rows' order.
fn classes(rows: &[Candidate]) -> Vec<(WorkerId, u32)> {
    rows.iter().map(|row| (row.id, row.pair_class)).collect()
}

/// The class on the row of worker `id`.
fn class_of(rows: &[Candidate], id: WorkerId) -> u32 {
    let row = rows.iter().find(|row| row.id == id);
    row.expect("the worker's row").pair_class
}

fn sorted(mut ids: Vec<WorkerId>) -> Vec<WorkerId> {
    ids.sort_unstable();
    ids
}

/// Two prefills of two transports and three decodes, one of each transport
/// and one that names none: leniently the silent decode pairs with both
/// prefills, one row of one class that both prefill classes list. The
/// prefills, then the decodes of the first transport, of the second and the
/// silent one.
fn two_classes(store: &MemoryStore) -> ([WorkerId; 2], [WorkerId; 3]) {
    let member = |key: &str, role: Role, transport: Option<&str>| {
        admit(store, pair_facts(key, role, transport))
    };
    let prefills = [
        member("grpc://pa:9000", Role::Prefill, Some("tx-a")),
        member("grpc://pb:9000", Role::Prefill, Some("tx-b")),
    ];
    let decodes = [
        member("grpc://da:9000", Role::Decode, Some("tx-a")),
        member("grpc://db:9000", Role::Decode, Some("tx-b")),
        member("grpc://dn:9000", Role::Decode, None),
    ];
    (prefills, decodes)
}

/// A pair plan whose stages read nothing of a row but its class and whether
/// the host excluded it.
const PAIR_PLAN: &str = r#"{
  "apiVersion": "smg.policy/v1",
  "profiles": { "p": { "filter": ["not-tried"], "pick": "best" } },
  "bindings": [
    { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "p", "decode": "p" } }
  ]
}"#;

/// How a pair decision is handed which classes pair.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pairs {
    /// As the door hands them: the rows and the table.
    ByTable,
    /// The door's rows with no table: a class pairs with its own number.
    ByNumber,
}

/// The decodes the engine's pair entry pairs `prefill` with over what the
/// store's pair door builds for `ask`, sorted by id: one decision after
/// another, every other prefill excluded and each decode already drawn
/// excluded from the next. The exclusion is soft, so the draws end when the
/// entry has nobody new to pair: it pairs another prefill, or a decode
/// again, or nobody.
fn partners_drawn(
    store: &MemoryStore,
    ask: &Ask,
    prefill: WorkerId,
    pairs: Pairs,
) -> Vec<WorkerId> {
    let spec: PolicySpec = serde_json::from_str(PAIR_PLAN).expect("a well-formed document");
    let engine = spec.validate().expect("the engine");
    let (mut out, mut scratch) = (PairBuffers::default(), Scratch::new());
    let mut drawn = Vec::new();
    loop {
        let (built, _) = store.build_pair(MODEL, ask, &mut out).expect("both pools");
        let others: Vec<WorkerId> = classes(out.pools().prefills)
            .into_iter()
            .map(|(id, _)| id)
            .filter(|&id| id != prefill)
            .collect();
        let request = RequestFacts {
            model: MODEL,
            phase: Phase::Prefill,
            now_micros: built.now_micros,
            ..RequestFacts::default()
        };
        let pools = match pairs {
            Pairs::ByTable => out.pools(),
            Pairs::ByNumber => PairPools {
                table: PairTable::default(),
                ..out.pools()
            },
        };
        let decision = engine.decide_pair(
            &request,
            &built.pool_facts(),
            pools,
            &Constraints {
                excluded: &others,
                ..Constraints::default()
            },
            &Constraints {
                excluded: &drawn,
                ..Constraints::default()
            },
            &mut scratch,
        );
        let target = decision
            .pick()
            .and_then(|pick| pick.targets.first().copied());
        let decode = target
            .filter(|target| target.id == prefill)
            .and_then(|target| target.pair)
            .map(|pair| pair.decode_id)
            .filter(|decode| !drawn.contains(decode));
        match decode {
            Some(decode) => drawn.push(decode),
            None => return sorted(drawn),
        }
    }
}

/// The step's one sentence: a pair built from the store pairs through the
/// engine's pair entry. Before, no row carried a class and the entry paired
/// nobody.
#[test]
fn a_pair_built_from_the_store_pairs_through_the_engines_pair_entry() {
    let store = MemoryStore::new();
    let now = Instant::now();
    two_classes(&store);
    let spec: PolicySpec = serde_json::from_str(PAIR_PLAN).expect("a well-formed document");
    let engine = spec.validate().expect("the engine");
    let (mut out, mut scratch) = (PairBuffers::default(), Scratch::new());
    let (built, _) = store
        .build_pair(MODEL, &ask(now), &mut out)
        .expect("both pools");
    let request = RequestFacts {
        model: MODEL,
        phase: Phase::Prefill,
        now_micros: built.now_micros,
        ..RequestFacts::default()
    };
    let decision = engine.decide_pair(
        &request,
        &built.pool_facts(),
        out.pools(),
        &Constraints::default(),
        &Constraints::default(),
        &mut scratch,
    );
    assert!(decision.pick().is_some(), "the pair routes: {decision:?}");
}

#[test]
fn the_pair_door_hands_the_table_its_rows_classes_are_numbered_for() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let ([pa, pb], [da, db, dn]) = two_classes(&store);
    // A prefill of another runtime: no decode pairs with it, so the entry
    // pairs another prefill before it.
    let apart = store
        .upsert(WorkerFacts {
            engine_kind: "rt-b".to_string(),
            ..pair_facts("grpc://px:9000", Role::Prefill, None)
        })
        .id();
    store.publish_now();

    // Through the door a prefill pairs with the decode of its transport and
    // with the silent one, and with no other.
    let drawn = |prefill: WorkerId, pairs: Pairs| partners_drawn(&store, &ask(now), prefill, pairs);
    assert_eq!(drawn(pa, Pairs::ByTable), sorted(vec![da, dn]));
    assert_eq!(drawn(pb, Pairs::ByTable), sorted(vec![db, dn]));
    assert_eq!(drawn(apart, Pairs::ByTable), []);
    // The table is what says so. Each side's classes are numbered by
    // themselves, so by its number alone a prefill class meets one decode
    // class, never its two.
    assert_eq!(drawn(pa, Pairs::ByNumber).len(), 1);
    assert_eq!(drawn(pb, Pairs::ByNumber).len(), 1);
}

#[test]
fn a_worker_has_one_row_of_one_class_and_the_world_holds_which_classes_pair() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let ([pa, pb], [da, db, dn]) = two_classes(&store);
    // A decode of another runtime: no prefill pairs with it.
    let apart = store
        .upsert(WorkerFacts {
            engine_kind: "rt-b".to_string(),
            ..pair_facts("grpc://dx:9000", Role::Decode, None)
        })
        .id();
    store.publish_now();

    // One row a worker on each side: the silent decode pairs with both
    // prefills and is still one row, with one position and one column entry.
    let (prefills, _, built) = side(&store, Role::Prefill, &ask(now));
    assert_eq!(built.pool_size, 2);
    let with_column = Ask {
        affinity: true,
        ..ask(now)
    };
    let (decodes, column, built) = side(&store, Role::Decode, &with_column);
    assert_eq!((built.pool_size, column.len()), (4, 4));
    assert_eq!(
        decodes.iter().map(|row| row.position).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );

    // Each side's classes are numbered as their first worker comes in id
    // order; the decode no prefill pairs with is class 0 and takes no number.
    let numbered = |ids: &[WorkerId]| -> Vec<(WorkerId, u32)> {
        sorted(ids.to_vec()).into_iter().zip(1..).collect()
    };
    assert_eq!(classes(&prefills), numbered(&[pa, pb]));
    let mut expected = numbered(&[da, db, dn]);
    expected.push((apart, 0));
    expected.sort_unstable();
    assert_eq!(classes(&decodes), expected);

    // The world holds what the rows were written from: a prefill class
    // lists the class of its transport's decode and the silent decode's.
    let world = store.world();
    let pairs = world.pairs(MODEL).expect("the model's classes");
    assert_eq!((pairs.classes(), pairs.pairs()), (2, 4));
    for (prefill, partners) in [(pa, [da, dn]), (pb, [db, dn])] {
        let mut listed = partners.map(|id| class_of(&decodes, id));
        listed.sort_unstable();
        assert_eq!(pairs.table().of(&class_of(&prefills, prefill)), listed);
    }
    assert_eq!(pairs.refusal(WireModes::GRPC), None);
    assert!(world.pairs("other").is_none());
}

#[test]
fn a_pool_less_another_gateways_host_local_rows_pairs_by_the_same_table() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let other = GatewayId(2);
    let prefill =
        |key: &str, transport: &str| admit(&store, pair_facts(key, Role::Prefill, Some(transport)));
    let (pa, pb) = (
        prefill("grpc://pa:9000", "tx-a"),
        prefill("grpc://pb:9000", "tx-b"),
    );
    // Host-local decodes of each gateway on the first transport, and a
    // silent decode every gateway dials.
    let local = |key: String, gateway: GatewayId| {
        let registered = WorkerFacts {
            reachable_from: Reachability::Gateway(gateway),
            ..pair_facts(&key, Role::Decode, Some("tx-a"))
        };
        admit(&store, registered)
    };
    let of_other: Vec<WorkerId> = (0..4)
        .map(|index| local(format!("ipc://o{index}"), other))
        .collect();
    let of_asker: Vec<WorkerId> = (0..4)
        .map(|index| local(format!("ipc://a{index}"), ASKER))
        .collect();
    let dn = store
        .upsert(pair_facts("grpc://dn:9000", Role::Decode, None))
        .id();
    store.publish_now();

    let world = store.world();
    let held = world.pairs(MODEL).expect("the model's classes");
    let lists = |table: PairTable<'_>| -> Vec<Vec<u32>> {
        (1..=held.classes())
            .map(|class| table.of(&class).to_vec())
            .collect()
    };
    for (gateway, own) in [(ASKER, &of_asker), (other, &of_other)] {
        let asked = Ask {
            gateway,
            ..ask(now)
        };
        // The gateway's decode pool is its own host-local rows and the
        // shared one. A row's position is the gateway's own; its class and
        // the table are the world's.
        let mut out = PairBuffers::default();
        store
            .build_pair(MODEL, &asked, &mut out)
            .expect("both pools");
        let mut reached = own.clone();
        reached.push(dn);
        let rows = classes(out.pools().decodes);
        assert_eq!(
            rows.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
            sorted(reached.clone()),
            "{gateway:?}"
        );
        assert!(
            rows.iter().all(|row| held.decodes().contains(row)),
            "{gateway:?}"
        );
        assert_eq!(lists(out.pools().table), lists(held.table()), "{gateway:?}");

        // The first transport's prefill pairs with every decode the gateway
        // reaches, the second's with the silent one alone.
        let drawn = |prefill: WorkerId| partners_drawn(&store, &asked, prefill, Pairs::ByTable);
        assert_eq!(drawn(pa), sorted(reached), "{gateway:?}");
        assert_eq!(drawn(pb), [dn], "{gateway:?}");
    }
}

#[test]
fn the_classes_follow_the_membership_and_are_each_models_own() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let class_in = |model: &str, role: Role, id: WorkerId| {
        let mut out = ViewBuffers::default();
        store.build_view(model, role, &ask(now), &mut out)?;
        let row = out.candidates.iter().find(|row| row.id == id)?;
        Some(row.pair_class)
    };
    // A prefill of two models and no decode: outside any pairing in both.
    let prefill = store
        .upsert(facts("grpc://p:9000", Role::Prefill, &[MODEL, "other"]))
        .id();
    store.publish_now();
    assert_eq!(class_in(MODEL, Role::Prefill, prefill), Some(0));
    assert!(store.world().pairs(MODEL).is_none());

    // A decode of one of the two models arrives: a class in that model.
    let decode = store
        .upsert(facts("grpc://d:9000", Role::Decode, &[MODEL]))
        .id();
    store.publish_now();
    assert_eq!(class_in(MODEL, Role::Prefill, prefill), Some(1));
    assert_eq!(class_in(MODEL, Role::Decode, decode), Some(1));
    assert_eq!(class_in("other", Role::Prefill, prefill), Some(0));
    let world = store.world();
    assert_eq!(world.pairs(MODEL).map(|pairs| pairs.classes()), Some(1));
    assert!(world.pairs("other").is_none());

    // It leaves: the prefill is outside any pairing again.
    let revision = WorkersStore::get(&store, decode).expect("the row").revision;
    store.remove(decode, revision).expect("removed");
    store.publish_now();
    assert_eq!(class_in(MODEL, Role::Prefill, prefill), Some(0));
    assert!(store.world().pairs(MODEL).is_none());

    // A decode no prefill pairs with: no class, and the world says why.
    let apart = store
        .upsert(WorkerFacts {
            engine_kind: "rt-b".to_string(),
            ..facts("grpc://e:9000", Role::Decode, &[MODEL])
        })
        .id();
    store.publish_now();
    store.upsert(WorkerFacts {
        engine_kind: "rt-a".to_string(),
        ..facts("grpc://p:9000", Role::Prefill, &[MODEL, "other"])
    });
    store.publish_now();
    assert_eq!(class_in(MODEL, Role::Prefill, prefill), Some(0));
    assert_eq!(class_in(MODEL, Role::Decode, apart), Some(0));
    let world = store.world();
    let pairs = world.pairs(MODEL).expect("both pools hold a worker");
    assert_eq!(pairs.classes(), 0);
    assert_eq!(
        pairs.refusal(WireModes::NONE),
        Some(&Refusal {
            prefill: vec!["rt-a/?/?".to_string()],
            decode: vec!["rt-b/?/?".to_string()],
            mismatches: vec!["runtime".to_string()],
        })
    );
}

#[test]
fn two_stores_fed_one_membership_in_two_orders_number_its_classes_alike_under_each_mode() {
    // Three prefills of three transports; a decode of each of two of them
    // and one that names none.
    let fleet = [
        ("grpc://pa:9000", Role::Prefill, Some("tx-a")),
        ("grpc://pb:9000", Role::Prefill, Some("tx-b")),
        ("grpc://pc:9000", Role::Prefill, Some("tx-c")),
        ("grpc://da:9000", Role::Decode, Some("tx-a")),
        ("grpc://db:9000", Role::Decode, Some("tx-b")),
        ("grpc://dn:9000", Role::Decode, None),
    ];
    let now = Instant::now();
    // The prefill classes, the decode classes and the prefills outside any
    // pairing.
    let expected = [
        (Pairing::Off, 1, 1, 0),
        (Pairing::Lenient, 3, 3, 0),
        (Pairing::Strict, 2, 2, 1),
    ];
    for (mode, prefill_classes, decode_classes, unpaired) in expected {
        let fed = |order: &mut dyn Iterator<Item = &(&str, Role, Option<&str>)>| {
            let store = store_pairing(mode);
            for &(key, role, transport) in order {
                store.upsert(pair_facts(key, role, transport));
                store.publish_now();
            }
            store
        };
        let (one, other) = (fed(&mut fleet.iter()), fed(&mut fleet.iter().rev()));
        let (prefills, _, _) = side(&one, Role::Prefill, &ask(now));
        let (decodes, _, _) = side(&one, Role::Decode, &ask(now));
        // One row a worker, whatever the mode pairs.
        assert_eq!((prefills.len(), decodes.len()), (3, 3), "{mode:?}");
        let largest = |rows: &[Candidate]| rows.iter().map(|row| row.pair_class).max();
        assert_eq!(largest(&decodes), Some(decode_classes), "{mode:?}");
        assert_eq!(
            prefills.iter().filter(|row| row.pair_class == 0).count(),
            unpaired,
            "{mode:?}"
        );
        let world = one.world();
        let pairs = world.pairs(MODEL).expect("the model's classes");
        assert_eq!(pairs.classes(), prefill_classes, "{mode:?}");

        // The other store numbers every class alike, row for row, and holds
        // the same table.
        assert_eq!(
            classes(&side(&other, Role::Prefill, &ask(now)).0),
            classes(&prefills),
            "{mode:?}"
        );
        assert_eq!(
            classes(&side(&other, Role::Decode, &ask(now)).0),
            classes(&decodes),
            "{mode:?}"
        );
        assert_eq!(other.world().pairs(MODEL), Some(pairs), "{mode:?}");
    }
}

#[test]
fn a_generation_of_another_pairing_mode_derives_the_classes_again_at_the_same_epoch() {
    let store = MemoryStore::new();
    let now = Instant::now();
    // A prefill and a decode of two known runtimes on the plain wire: a
    // comparing mode refuses the pair, off compares nothing of the handoff.
    for (key, role, runtime) in [
        ("w://p:1", Role::Prefill, "rt-a"),
        ("w://d:1", Role::Decode, "rt-b"),
    ] {
        store.upsert(WorkerFacts {
            wire_modes: WireModes::HTTP,
            engine_kind: runtime.to_string(),
            ..facts(key, role, &[MODEL])
        });
        store.publish_now();
    }
    let class = || side(&store, Role::Prefill, &ask(now)).0[0].pair_class;
    // The prefill's load record on file, as its row carries it.
    let record = || {
        let row = side(&store, Role::Prefill, &ask(now)).0[0];
        (
            row.load_sampled_at_micros,
            row.load.map(|load| load.running),
        )
    };
    let prefill = side(&store, Role::Prefill, &ask(now)).0[0].id;
    store.record(prefill, 0, load(3, 0), now, LoadSource::Stream);
    let on_file = record();
    assert_eq!(on_file.1, Some(3));
    let epoch = store.epoch();
    let unbound = store.world();
    assert_eq!(class(), 0, "the unbound generation holds the default mode");

    // A generation of the standing mode publishes the rows as they are:
    // what the membership derived is shared, by pointer.
    store.apply(generation(1)).expect("the first generation");
    let first = store.world();
    assert!(Arc::ptr_eq(&unbound.derived, &first.derived));
    assert_eq!((first.epoch(), first.spec().generation()), (epoch, 1));
    assert_eq!(class(), 0);

    // A generation of another mode, with no member moved: the classes are
    // derived again and published once under the new generation.
    store
        .apply(generation_pairing(2, Pairing::Off))
        .expect("above the first");
    let second = store.world();
    assert!(!Arc::ptr_eq(&first.derived, &second.derived));
    assert_eq!(class(), 1);
    assert_eq!(second.pairs(MODEL).map(|pairs| pairs.classes()), Some(1));
    assert_eq!(second.spec().generation(), 2);
    // Each row is derived with its load record on file.
    assert_eq!(record(), on_file);
    // The epoch is the membership's: it reads the same in the store and on
    // the world, and a watcher that saw it is told nothing.
    assert_eq!((store.epoch(), second.epoch()), (epoch, epoch));
    assert_eq!(
        store.changes_since(epoch),
        MembershipTail::Changes(Vec::new())
    );
    // The world a decision took before the apply keeps its own classes.
    assert_eq!(first.pairs(MODEL).map(|pairs| pairs.classes()), Some(0));

    // A membership write under the new mode derives under it,
    store.upsert(WorkerFacts {
        wire_modes: WireModes::HTTP,
        engine_kind: "rt-b".to_string(),
        ..facts("w://d:2", Role::Decode, &[MODEL])
    });
    store.publish_now();
    assert_eq!(class(), 1);
    assert_eq!(store.epoch(), epoch + 1);
    // and a refused generation changes no class.
    assert!(store.apply(generation_pairing(2, Pairing::Strict)).is_err());
    assert_eq!(class(), 1);
    store
        .apply(generation_pairing(3, Pairing::Lenient))
        .expect("above the second");
    assert_eq!(class(), 0);
    assert_eq!(store.epoch(), epoch + 1);
}

#[test]
fn the_store_hands_the_rule_each_workers_wires_and_its_own_mode() {
    let now = Instant::now();
    // A prefill and a decode of two known runtimes, both on `wires`, in a
    // store of `mode`: the class of the prefill's row.
    let class = |mode: Pairing, wires: WireModes| {
        let store = store_pairing(mode);
        for (key, role, runtime) in [
            ("w://p:1", Role::Prefill, "rt-a"),
            ("w://d:1", Role::Decode, "rt-b"),
        ] {
            store.upsert(WorkerFacts {
                wire_modes: wires,
                engine_kind: runtime.to_string(),
                ..facts(key, role, &[MODEL])
            });
            store.publish_now();
        }
        side(&store, Role::Prefill, &ask(now)).0[0].pair_class
    };
    // Off compares nothing of the handoff: the two pair on the plain wire.
    assert_eq!(class(Pairing::Off, WireModes::HTTP), 1);
    // A comparing mode refuses them there, and no mode pairs them where a
    // pair shares a runtime.
    assert_eq!(class(Pairing::Lenient, WireModes::HTTP), 0);
    assert_eq!(class(Pairing::Off, WireModes::GRPC), 0);

    // A prefill and a decode on two sides of the wire's fact never pair,
    // and nothing is refused: neither side holds both pools.
    let store = store_pairing(Pairing::Off);
    for (key, role, wires) in [
        ("http://p:1", Role::Prefill, WireModes::HTTP),
        ("grpc://d:1", Role::Decode, WireModes::GRPC),
    ] {
        store.upsert(WorkerFacts {
            wire_modes: wires,
            ..facts(key, role, &[MODEL])
        });
        store.publish_now();
    }
    assert_eq!(side(&store, Role::Prefill, &ask(now)).0[0].pair_class, 0);
    assert_eq!(side(&store, Role::Decode, &ask(now)).0[0].pair_class, 0);
    let world = store.world();
    let pairs = world.pairs(MODEL).expect("both pools hold a worker");
    assert_eq!(pairs.classes(), 0);
    assert_eq!(
        (
            pairs.refusal(WireModes::HTTP),
            pairs.refusal(WireModes::GRPC)
        ),
        (None, None)
    );
}

/// The id and the class of every prefill row and then every decode row of
/// the model, as views built from the store's world of this instant read
/// them.
fn pair_rows(store: &MemoryStore) -> Vec<(WorkerId, u32)> {
    let mut out = ViewBuffers::default();
    let mut rows = Vec::new();
    for role in [Role::Prefill, Role::Decode] {
        let built = store.build_view(MODEL, role, &ask(Instant::now()), &mut out);
        if built.is_some() {
            rows.extend(classes(&out.candidates));
        }
    }
    rows
}

#[test]
fn a_membership_write_that_moves_no_offer_carries_the_classes() {
    let store = MemoryStore::new();
    let (prefills, _) = two_classes(&store);
    // The world is kept, so its classes stand where they are: one address
    // in a later world is those classes carried, never others in their
    // place.
    let before = store.world();
    let derived = before.pairs(MODEL).expect("the model's classes");
    let rows = pair_rows(&store);
    let carried = |what: &str| {
        let world = store.world();
        let pairs = world.pairs(MODEL).expect("the model's classes");
        assert!(std::ptr::eq(derived, pairs), "{what}: derived again");
        assert_eq!(pair_rows(&store), rows, "{what}");
        world.epoch()
    };
    // The same registration again: a membership write all the same.
    store.upsert(pair_facts("grpc://pa:9000", Role::Prefill, Some("tx-a")));
    store.publish_now();
    assert!(carried("the same facts") > before.epoch());
    // Facts no pair reads.
    store.upsert(WorkerFacts {
        zone: Some("z".to_string()),
        max_running: Some(9),
        priority: 2,
        ..pair_facts("grpc://pa:9000", Role::Prefill, Some("tx-a"))
    });
    store.publish_now();
    carried("facts no pair reads");
    // A patch moves labels, a priority and an overload alone.
    let revision = WorkersStore::get(&store, prefills[0])
        .expect("the row")
        .revision;
    let patch = WorkerPatch {
        labels: None,
        priority: Some(3),
        overload: None,
    };
    store.patch(prefills[0], patch, revision).expect("patched");
    store.publish_now();
    carried("a patch");
}

#[test]
fn a_membership_write_that_moves_an_offer_derives_the_classes_again() {
    // Each fact that places a worker in its pools or makes its offer, moved
    // on one prefill by its registration refreshed.
    let first = || pair_facts("grpc://pa:9000", Role::Prefill, Some("tx-a"));
    let moved = [
        (
            "the descriptor",
            pair_facts("grpc://pa:9000", Role::Prefill, Some("tx-b")),
        ),
        (
            "the runtime",
            WorkerFacts {
                engine_kind: "rt-b".to_string(),
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
                models: vec!["other".to_string()],
                ..first()
            },
        ),
    ];
    for (what, facts) in moved {
        let store = MemoryStore::new();
        two_classes(&store);
        let before = pair_rows(&store);
        store.upsert(facts.clone());
        store.publish_now();
        // A store that never held the first version: every write of it
        // registers a worker, and derives.
        let fresh = MemoryStore::new();
        fresh.upsert(facts);
        fresh.publish_now();
        for (key, role, transport) in [
            ("grpc://pb:9000", Role::Prefill, Some("tx-b")),
            ("grpc://da:9000", Role::Decode, Some("tx-a")),
            ("grpc://db:9000", Role::Decode, Some("tx-b")),
            ("grpc://dn:9000", Role::Decode, None),
        ] {
            fresh.upsert(pair_facts(key, role, transport));
            fresh.publish_now();
        }
        let after = pair_rows(&store);
        assert_eq!(after, pair_rows(&fresh), "{what}");
        assert_eq!(
            store.world().pairs(MODEL),
            fresh.world().pairs(MODEL),
            "{what}"
        );
        assert_ne!(after, before, "{what}: the classes moved");
    }
}

/// The five workers of [`two_classes`], by key, role and transport.
const PAIR_FLEET: [(&str, Role, Option<&str>); 5] = [
    ("grpc://pa:9000", Role::Prefill, Some("tx-a")),
    ("grpc://pb:9000", Role::Prefill, Some("tx-b")),
    ("grpc://da:9000", Role::Decode, Some("tx-a")),
    ("grpc://db:9000", Role::Decode, Some("tx-b")),
    ("grpc://dn:9000", Role::Decode, None),
];

#[test]
fn a_member_that_leaves_a_model_of_two_sides_derives_its_classes_again() {
    let store = MemoryStore::new();
    let (_, decodes) = two_classes(&store);
    let before = pair_rows(&store);
    // The decode of the first transport leaves: both sides keep a member.
    let revision = WorkersStore::get(&store, decodes[0])
        .expect("the row")
        .revision;
    store.remove(decodes[0], revision).expect("removed");
    store.publish_now();
    // A store that never held it.
    let fresh = MemoryStore::new();
    for (key, role, transport) in PAIR_FLEET {
        if key != "grpc://da:9000" {
            fresh.upsert(pair_facts(key, role, transport));
            fresh.publish_now();
        }
    }
    let after = pair_rows(&store);
    assert_eq!(after, pair_rows(&fresh));
    assert_eq!(store.world().pairs(MODEL), fresh.world().pairs(MODEL));
    assert_ne!(after, before);
}

#[test]
fn a_new_incarnation_under_a_key_derives_the_classes_again() {
    let store = MemoryStore::new();
    let (prefills, _) = two_classes(&store);
    // The first prefill comes again under its key: a new id in its pool.
    let again = WorkerFacts {
        incarnation: 2,
        ..pair_facts("grpc://pa:9000", Role::Prefill, Some("tx-a"))
    };
    store.upsert(again.clone());
    store.publish_now();
    // A store that never held the first incarnation.
    let fresh = MemoryStore::new();
    fresh.upsert(again);
    fresh.publish_now();
    for &(key, role, transport) in &PAIR_FLEET[1..] {
        fresh.upsert(pair_facts(key, role, transport));
        fresh.publish_now();
    }
    let after = pair_rows(&store);
    assert_eq!(after, pair_rows(&fresh));
    assert_eq!(store.world().pairs(MODEL), fresh.world().pairs(MODEL));
    assert!(
        after.iter().all(|&(id, _)| id != prefills[0]),
        "the first incarnation left"
    );
    assert!(after.iter().all(|&(_, class)| class != 0), "{after:?}");
}

#[test]
fn a_write_that_moves_no_offer_carries_each_models_own_classes() {
    let store = MemoryStore::new();
    two_classes(&store);
    // A second model, of one prefill and one decode: classes of its own.
    store.upsert(facts("grpc://op:9000", Role::Prefill, &["other"]));
    store.publish_now();
    store.upsert(facts("grpc://od:9000", Role::Decode, &["other"]));
    store.publish_now();
    let classes_of = |world: &World, model: &str| {
        std::ptr::from_ref(world.pairs(model).expect("the model's classes"))
    };
    // The world is kept, so each model's classes stand where they are.
    let before = store.world();
    let (first, second) = (classes_of(&before, MODEL), classes_of(&before, "other"));
    assert_ne!(first, second);
    // A worker of each model writes its registration again.
    store.upsert(pair_facts("grpc://pa:9000", Role::Prefill, Some("tx-a")));
    store.publish_now();
    store.upsert(facts("grpc://od:9000", Role::Decode, &["other"]));
    store.publish_now();
    let after = store.world();
    assert_eq!(classes_of(&after, MODEL), first);
    assert_eq!(classes_of(&after, "other"), second);
}

#[test]
fn a_blank_value_of_a_descriptor_is_dropped_at_the_door() {
    // The fleet of `two_classes` written with blanks: a blank protocol and
    // version on every worker, each transport between blanks, and a blank
    // one where the fleet names none.
    let blanked = MemoryStore::new();
    let ids: Vec<WorkerId> = PAIR_FLEET
        .iter()
        .map(|&(key, role, transport)| {
            let transport = transport.map_or_else(|| " ".to_string(), |name| format!(" {name} "));
            let facts = WorkerFacts {
                pairing: Some(Arc::new(PairingDescriptor {
                    protocol: Some(String::new()),
                    transport: Some(transport),
                    version: Some(String::new()),
                    kv_layout: [Some(" ".to_string()), None, None, None],
                })),
                ..pair_facts(key, role, None)
            };
            admit(&blanked, facts)
        })
        .collect();
    let plain = MemoryStore::new();
    two_classes(&plain);
    // Two blank protocols settle no pair: the transports decide, as where no
    // worker sets a protocol.
    assert_eq!(pair_rows(&blanked), pair_rows(&plain));
    assert_eq!(blanked.world().pairs(MODEL), plain.world().pairs(MODEL));
    // The row of record holds each value as the rule reads it.
    let first = ids.first().copied().expect("a worker");
    let row = WorkersStore::get(&blanked, first).expect("the row");
    assert_eq!(
        row.facts.pairing,
        Some(Arc::new(PairingDescriptor {
            transport: Some("tx-a".to_string()),
            ..PairingDescriptor::default()
        }))
    );
}

#[test]
fn a_descriptor_the_rule_reads_is_shared_by_every_copy_of_its_row() {
    let store = MemoryStore::new();
    // Handed in as the rule reads it, and kept by the caller too.
    let handed = Arc::new(PairingDescriptor {
        transport: Some("tx-a".to_string()),
        ..PairingDescriptor::default()
    });
    let id = admit(
        &store,
        WorkerFacts {
            pairing: Some(Arc::clone(&handed)),
            ..pair_facts("grpc://pa:9000", Role::Prefill, None)
        },
    );
    // The door kept the descriptor where it stood, and each read by value
    // copies the row with it: one descriptor behind them all.
    for _ in 0..2 {
        let row = WorkersStore::get(&store, id).expect("the row");
        let held = row.facts.pairing.as_ref().expect("the descriptor");
        assert!(Arc::ptr_eq(held, &handed));
    }
}

// ------------------------------------------------------ the overload verdict

/// A generation whose one binding hands every pool the block `overload`.
fn generation_with(number: u64, overload: OverloadSpec) -> AppliedSpec {
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = number;
    let profile = document
        .profiles
        .keys()
        .next()
        .expect("the preset's profile")
        .clone();
    document.bindings = vec![BindingSpec {
        profile: Some(profile),
        overload: Some(overload),
        ..BindingSpec::default()
    }];
    AppliedSpec::validate(&document).expect("the document validates")
}

/// A block with both thresholds named.
fn block(waiting_requests: u32, token_usage: f32) -> OverloadSpec {
    OverloadSpec {
        waiting_requests: Some(waiting_requests),
        token_usage: Some(token_usage),
        ..OverloadSpec::default()
    }
}

/// A load row with a queue and an active KV usage.
fn pressure(queued: u32, active_kv_permille: u16) -> LoadRow {
    LoadRow {
        queued,
        active_kv_permille,
        ..LoadRow::default()
    }
}

/// The workers of the pool whose rows carry the overload verdict, by id.
fn overloaded(store: &MemoryStore, now: Instant) -> Vec<WorkerId> {
    view(store, &ask(now))
        .0
        .iter()
        .filter(|row| row.health.overloaded)
        .map(|row| row.id)
        .collect()
}

/// One rank of the two-rank engine `engine`.
fn rank_of(store: &MemoryStore, engine: &str, rank: u16) -> WorkerId {
    rank_among(store, engine, rank, 2)
}

/// One rank of the engine `engine`, which says it has `size` ranks.
fn rank_among(store: &MemoryStore, engine: &str, rank: u16, size: u16) -> WorkerId {
    admit(
        store,
        WorkerFacts {
            engine: engine.to_string(),
            dp_rank: rank,
            dp_size: size,
            ..facts(&format!("{engine}#{rank}"), Role::Single, &[MODEL])
        },
    )
}

#[test]
fn a_store_with_no_threshold_writes_no_verdict() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let a = member(&store, "http://a:8000");
    // Under the unbound generation no pool has a pair.
    store.record(a, 0, pressure(1_000, 1_000), t0, LoadSource::Stream);
    assert_eq!(overloaded(&store, t0), []);
    // A block switched off hands none, whatever values it still names.
    let off = OverloadSpec {
        enabled: false,
        ..block(1, 0.1)
    };
    store.apply(generation_with(1, off)).expect("applied");
    store.record(a, 0, pressure(1_000, 1_000), t0, LoadSource::Stream);
    assert_eq!(overloaded(&store, t0), []);
    assert_eq!(store.overload_flips(), 0, "no write moved a verdict");
}

#[test]
fn the_pass_over_every_engine_says_whether_it_moved_a_verdict() {
    let origin = Instant::now();
    // One member that names a count of its own, with three requests queued.
    let member = WorkerRow {
        facts: WorkerFacts {
            overload: Overload {
                waiting_requests: Some(2),
                kv_usage_permille: None,
            },
            ..facts("http://w1:8000", Role::Single, &[MODEL])
        },
        ..row_serving(1, &[MODEL], origin)
    };
    let sample = |_, _| {
        Some(LoadSample {
            row: pressure(3, 0),
            received_at: origin,
            sampled_at: origin,
            source: LoadSource::Stream,
        })
    };
    let mut writer = Writer::new(PUBLISH_WINDOW, origin);
    writer.rebuild(1, [member].filed(&writer), sample, Classes::Derive);
    let verdict = |writer: &Writer| writer.working[0].rows[0].health.overloaded;
    // The build judged every engine: the row carries its verdict, and a
    // pass finds it as that write left it.
    assert!(verdict(&writer));
    assert!(!writer.judge_every());
    // A verdict that stands against what is on file is one the pass moves,
    // and the pass says so: what a door behind its own judgment holds to
    // none.
    writer.working[0].rows[0].health.overloaded = false;
    assert!(writer.judge_every());
    assert!(verdict(&writer));
    assert!(!writer.judge_every());
}

#[test]
fn a_row_is_judged_at_and_above_its_pool_s_thresholds_and_its_own_stand_over_them() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let a = member(&store, "http://a:8000");
    let b = member(&store, "http://b:8000");
    // The third names a count of its own and takes the pool's usage.
    let c = store
        .upsert(WorkerFacts {
            overload: Overload {
                waiting_requests: Some(2),
                kv_usage_permille: None,
            },
            ..facts("http://c:8000", Role::Single, &[MODEL])
        })
        .id();
    store.publish_now();
    store
        .apply(generation_with(1, block(4, 0.5)))
        .expect("applied");

    // The count: at the threshold is over, one under is not.
    store.record(a, 0, pressure(4, 0), t0, LoadSource::Stream);
    store.record(b, 0, pressure(3, 0), t0, LoadSource::Stream);
    store.record(c, 0, pressure(2, 0), t0, LoadSource::Stream);
    let mut over = [a, c];
    over.sort_unstable();
    assert_eq!(overloaded(&store, t0), over);
    // The usage, in the row's unit: at the threshold is over, one
    // permille under is not; either signal alone vetoes.
    store.record(a, 0, pressure(0, 499), t0, LoadSource::Stream);
    store.record(b, 0, pressure(0, 500), t0, LoadSource::Stream);
    store.record(c, 0, pressure(1, 500), t0, LoadSource::Stream);
    let mut over = [b, c];
    over.sort_unstable();
    assert_eq!(overloaded(&store, t0), over);

    // A worker's own value still vetoes under a block switched off, and
    // the value it left to the pool is gone with the block.
    let off = OverloadSpec {
        enabled: false,
        ..OverloadSpec::default()
    };
    store.apply(generation_with(2, off)).expect("applied");
    assert_eq!(
        overloaded(&store, t0),
        [],
        "the pool's usage left with the block"
    );
    store.record(c, 0, pressure(2, 0), t0, LoadSource::Stream);
    store.record(a, 0, pressure(9, 0), t0, LoadSource::Stream);
    assert_eq!(overloaded(&store, t0), [c]);
    // A membership write under no pool's pair judges again the engine
    // whose only threshold is its registration's own, and its records
    // are judged after it.
    member(&store, "http://d:8000");
    assert_eq!(overloaded(&store, t0), [c]);
    store.record(c, 0, pressure(1, 0), t0, LoadSource::Stream);
    assert_eq!(overloaded(&store, t0), []);
    store.record(c, 0, pressure(2, 0), t0, LoadSource::Stream);
    assert_eq!(overloaded(&store, t0), [c]);
}

#[test]
fn an_engine_s_ranks_are_judged_as_one_by_their_sum_and_their_mean() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    let mut both = [zero, one];
    both.sort_unstable();
    store
        .apply(generation_with(1, block(8, 0.8)))
        .expect("applied");
    let answer = |first: LoadRow, second: LoadRow| {
        store.report(
            &[(zero, 0, Some(first)), (one, 1, Some(second))],
            t0,
            LoadSource::Poll,
        );
        overloaded(&store, t0)
    };
    // The queued requests are summed over the ranks: neither rank is at
    // the count alone.
    assert_eq!(answer(pressure(5, 0), pressure(3, 0)), both);
    assert_eq!(answer(pressure(4, 0), pressure(3, 0)), []);
    // The usage is the mean over the ranks, compared with no division:
    // 900 and 700 are at 800, 900 and 699 are under it.
    assert_eq!(answer(pressure(0, 900), pressure(0, 700)), both);
    assert_eq!(answer(pressure(0, 900), pressure(0, 699)), []);
    // The ranks on file are the ranks judged: with one evicted, the other
    // is the whole engine.
    assert_eq!(answer(pressure(0, 800), pressure(0, 0)), []);
    // An eviction has no instant of its own: its judgment is shown with
    // the window's end, as its fold is.
    store.evict(one);
    store.publish(t0);
    assert_eq!(overloaded(&store, t0), both, "one rank on file, at 800");
    // An engine with nothing on file reads clear.
    store.evict(zero);
    store.publish(t0);
    assert_eq!(overloaded(&store, t0), []);
}

#[test]
fn two_ranks_that_trade_their_queues_in_one_answer_move_no_verdict() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    store.report(
        &[
            (zero, 0, Some(pressure(6, 0))),
            (one, 1, Some(pressure(1, 0))),
        ],
        t0,
        LoadSource::Poll,
    );
    assert_eq!(
        (overloaded(&store, t0), store.overload_flips()),
        (vec![], 0)
    );
    // One answer in which the ranks trade: the sum never left 7.
    store.report(
        &[
            (zero, 0, Some(pressure(1, 0))),
            (one, 1, Some(pressure(6, 0))),
        ],
        t0,
        LoadSource::Poll,
    );
    assert_eq!(
        (overloaded(&store, t0), store.overload_flips()),
        (vec![], 0)
    );
    // The same trade as two writes reads one new rank beside one old: 6
    // and 6 between them, a verdict raised and taken back.
    store.report(&[(zero, 0, Some(pressure(6, 0)))], t0, LoadSource::Poll);
    assert_eq!(overloaded(&store, t0).len(), 2);
    store.report(&[(one, 1, Some(pressure(1, 0)))], t0, LoadSource::Poll);
    assert_eq!(
        (overloaded(&store, t0), store.overload_flips()),
        (vec![], 2)
    );
}

#[test]
fn the_word_that_no_report_came_clears_the_verdict_until_the_engine_is_heard_again() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let a = member(&store, "http://a:8000");
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    store.report(&[(a, 0, Some(pressure(9, 0)))], t0, LoadSource::Poll);
    assert_eq!(
        (overloaded(&store, t0), store.overload_flips()),
        (vec![a], 1)
    );

    // A poll that brought no report: the verdict is cleared and the row on
    // file stays.
    let t1 = t0 + Duration::from_secs(1);
    store.report(&[(a, 0, None)], t1, LoadSource::Poll);
    assert_eq!(
        (overloaded(&store, t1), store.overload_flips()),
        (vec![], 2)
    );
    assert_eq!(store.load(a, 0).map(|sample| sample.row.queued), Some(9));
    // The word is on file: a membership write judges again and does not
    // raise the verdict from the row that is still there.
    member(&store, "http://b:8000");
    assert_eq!(overloaded(&store, t1), []);
    // So does a new generation.
    store
        .apply(generation_with(2, block(8, 1.0)))
        .expect("applied");
    assert_eq!(overloaded(&store, t1), []);
    assert_eq!(
        store.overload_flips(),
        2,
        "neither is a write of the load table"
    );
    // The engine's next record is judged as any other.
    let t2 = t1 + Duration::from_secs(1);
    store.record(a, 0, pressure(9, 0), t2, LoadSource::Stream);
    assert_eq!(
        (overloaded(&store, t2), store.overload_flips()),
        (vec![a], 3)
    );

    // A rank whose stream pushed since the poll began is heard: the poll's
    // word that none came does not stand over its record.
    store.report(
        &[(a, 0, None)],
        t2 - Duration::from_millis(1),
        LoadSource::Poll,
    );
    assert_eq!(
        (overloaded(&store, t2), store.overload_flips()),
        (vec![a], 3)
    );
}

#[test]
fn a_registration_s_own_usage_stands_over_its_pool_s() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let own = |key: &str, permille: u16| {
        admit(
            &store,
            WorkerFacts {
                overload: Overload {
                    waiting_requests: None,
                    kv_usage_permille: Some(permille),
                },
                ..facts(key, Role::Single, &[MODEL])
            },
        )
    };
    let low = own("http://low:8000", 300);
    let high = own("http://high:8000", 900);
    let pool = member(&store, "http://pool:8000");
    // The count is out of reach: the usage alone judges.
    store
        .apply(generation_with(1, block(1_000, 0.5)))
        .expect("applied");
    let at = |permille: u16| {
        for id in [low, high, pool] {
            store.record(id, 0, pressure(0, permille), t0, LoadSource::Stream);
        }
        overloaded(&store, t0)
    };
    // Under the pool's half: the row that names a lower usage is over it.
    assert_eq!(at(400), [low]);
    // Over the pool's half, and under the usage the third row names.
    let mut over = [low, pool];
    over.sort_unstable();
    assert_eq!(at(700), over);
    let mut all = [low, high, pool];
    all.sort_unstable();
    assert_eq!(at(900), all);
}

#[test]
fn a_pool_takes_the_thresholds_of_the_binding_that_covers_its_phase() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let prefill = store
        .upsert(facts("http://p:8000", Role::Prefill, &[MODEL]))
        .id();
    store.publish_now();
    let decode = store
        .upsert(facts("http://d:8000", Role::Decode, &[MODEL]))
        .id();
    store.publish_now();
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = 1;
    let profile = document
        .profiles
        .keys()
        .next()
        .expect("the preset's profile")
        .clone();
    // One model, two bindings: each covers one phase with its own count.
    let binding = |phase: Phase, waiting: u32| BindingSpec {
        matcher: Match {
            phase: vec![phase],
            ..Match::default()
        },
        profile: Some(profile.clone()),
        overload: Some(block(waiting, 1.0)),
        ..BindingSpec::default()
    };
    document.bindings = vec![binding(Phase::Prefill, 2), binding(Phase::Decode, 9)];
    store
        .apply(AppliedSpec::validate(&document).expect("the document validates"))
        .expect("applied");
    let over = |role: Role| {
        let mut out = ViewBuffers::default();
        store
            .world()
            .build(MODEL, role, &ask(t0), &mut out)
            .expect("the pool");
        out.candidates
            .iter()
            .map(|row| row.health.overloaded)
            .collect::<Vec<_>>()
    };
    // The same queue on both sides: over the prefill pool's count alone.
    store.record(prefill, 0, pressure(5, 0), t0, LoadSource::Stream);
    store.record(decode, 0, pressure(5, 0), t0, LoadSource::Stream);
    assert_eq!(
        (over(Role::Prefill), over(Role::Decode)),
        (vec![true], vec![false])
    );
}

#[test]
fn a_failed_poll_does_not_silence_an_engine_whose_rank_pushed_since_it_began() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    let mut both = [zero, one];
    both.sort_unstable();
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    store.report(
        &[
            (zero, 0, Some(pressure(1, 0))),
            (one, 1, Some(pressure(1, 0))),
        ],
        t0,
        LoadSource::Poll,
    );
    assert_eq!(overloaded(&store, t0), []);

    // A poll goes out, and rank 0's stream reports a queue while it is
    // out: the engine is over its count.
    let asked = t0 + Duration::from_secs(1);
    let pushed = asked + Duration::from_millis(1);
    store.record(zero, 0, pressure(9, 0), pushed, LoadSource::Stream);
    assert_eq!(
        (overloaded(&store, pushed), store.overload_flips()),
        (both.to_vec(), 1)
    );
    // The poll fails. Its word that none came stands over no engine a
    // rank of which was heard since it began, in either order of ranks.
    let none = [(zero, 0, None), (one, 1, None)];
    store.report(&none, asked, LoadSource::Poll);
    assert_eq!(
        (overloaded(&store, pushed), store.overload_flips()),
        (both.to_vec(), 1)
    );
    store.report(&[none[1], none[0]], asked, LoadSource::Poll);
    assert_eq!(
        (overloaded(&store, pushed), store.overload_flips()),
        (both.to_vec(), 1)
    );
    // So for an answer that names the other rank alone: a poll is for
    // the ranks no stream serves, and a failed one names no other.
    store.report(&[none[1]], asked, LoadSource::Poll);
    assert_eq!(
        (overloaded(&store, pushed), store.overload_flips()),
        (both.to_vec(), 1)
    );
    // Nor when the rank that pushed is the other one: rank 1's record
    // comes while a poll for rank 0 alone is out.
    let again = pushed + Duration::from_secs(1);
    let since = again + Duration::from_millis(1);
    store.record(one, 1, pressure(1, 0), since, LoadSource::Stream);
    store.report(&[none[0]], again, LoadSource::Poll);
    assert_eq!(
        (overloaded(&store, since), store.overload_flips()),
        (both.to_vec(), 1)
    );
    // A poll asked after both records, and failed, is the engine's word:
    // whether a stream still serves a rank is its caller's to know.
    let later = since + Duration::from_secs(1);
    store.report(&none, later, LoadSource::Poll);
    assert_eq!(
        (overloaded(&store, later), store.overload_flips()),
        (vec![], 2)
    );
    // An answer that brings one rank's row and omits the other is no such
    // word: the engine is heard, and the omitted rank keeps what is on
    // file.
    store.report(
        &[(zero, 0, Some(pressure(7, 0))), (one, 1, None)],
        later,
        LoadSource::Poll,
    );
    assert_eq!(
        (overloaded(&store, later), store.overload_flips()),
        (both.to_vec(), 3)
    );
}

#[test]
fn the_word_that_no_report_came_is_kept_and_ended_while_no_threshold_stands() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let a = member(&store, "http://a:8000");
    let off = || OverloadSpec {
        enabled: false,
        ..OverloadSpec::default()
    };
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    store.report(&[(a, 0, Some(pressure(9, 0)))], t0, LoadSource::Poll);
    store.report(&[(a, 0, None)], t0, LoadSource::Poll);
    assert_eq!(
        (overloaded(&store, t0), store.overload_flips()),
        (vec![], 2)
    );

    // The block is switched off and the engine is heard under no
    // threshold: its record ends the word, so the generation that brings
    // the threshold back judges what is on file.
    store.apply(generation_with(2, off())).expect("applied");
    store.record(a, 0, pressure(9, 0), t0, LoadSource::Stream);
    store
        .apply(generation_with(3, block(8, 1.0)))
        .expect("applied");
    assert_eq!(overloaded(&store, t0), [a]);

    // The word that comes under no threshold is kept: the generation that
    // brings one back raises no verdict from the row that is still on
    // file.
    store.apply(generation_with(4, off())).expect("applied");
    let t1 = t0 + Duration::from_secs(1);
    store.report(&[(a, 0, None)], t1, LoadSource::Poll);
    store
        .apply(generation_with(5, block(8, 1.0)))
        .expect("applied");
    assert_eq!(overloaded(&store, t1), []);
    assert_eq!(
        store.overload_flips(),
        2,
        "no write of the load table moved a verdict under no threshold"
    );
}

#[test]
fn the_word_that_no_report_came_is_read_an_engine_at_a_time_in_a_mixed_answer() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let a = member(&store, "http://a:8000");
    let b = member(&store, "http://b:8000");
    let mut both = [a, b];
    both.sort_unstable();
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    store.report(
        &[(a, 0, Some(pressure(9, 0))), (b, 0, Some(pressure(9, 0)))],
        t0,
        LoadSource::Poll,
    );
    assert_eq!(
        (overloaded(&store, t0), store.overload_flips()),
        (both.to_vec(), 2)
    );
    // One call for two engines: the first brings a row and the second
    // none. The second engine's word is its own, whatever the first said.
    let t1 = t0 + Duration::from_secs(1);
    store.report(
        &[(a, 0, Some(pressure(9, 0))), (b, 0, None)],
        t1,
        LoadSource::Poll,
    );
    assert_eq!(
        (overloaded(&store, t1), store.overload_flips()),
        (vec![a], 3)
    );
    // A later engine of the answer is heard by any of its rows: a
    // two-rank engine named after another, its first rank with a row and
    // its second with none, is judged and keeps what is on file.
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    let mut over = [a, zero, one];
    over.sort_unstable();
    let t2 = t1 + Duration::from_secs(1);
    store.report(
        &[
            (a, 0, Some(pressure(9, 0))),
            (zero, 0, Some(pressure(9, 0))),
            (one, 1, None),
        ],
        t2,
        LoadSource::Poll,
    );
    assert_eq!(
        (overloaded(&store, t2), store.overload_flips()),
        (over.to_vec(), 4)
    );
}

#[test]
fn a_single_rank_worker_is_an_engine_alone_and_ranks_share_their_engine_s_key() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    // Two single-rank rows that name one engine. The facts' rule is that
    // a single-rank worker's engine is its own key: each is judged alone.
    let alone = |key: &str| {
        admit(
            &store,
            WorkerFacts {
                engine: "http://shared:8000".to_string(),
                ..facts(key, Role::Single, &[MODEL])
            },
        )
    };
    let (x, y) = (alone("http://x:8000"), alone("http://y:8000"));
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    let mut ranks = [zero, one];
    ranks.sort_unstable();
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    for (id, rank) in [(x, 0), (y, 0), (zero, 0), (one, 1)] {
        store.record(id, rank, pressure(5, 0), t0, LoadSource::Stream);
    }
    // Five and five: at the count as one engine's ranks, under it alone.
    assert_eq!(overloaded(&store, t0), ranks);
}

/// The report period the age cases tell.
const REPORT_PERIOD: Duration = Duration::from_secs(10);

#[test]
fn a_stale_low_usage_masks_a_hot_rank_only_until_its_sample_ends() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let at = |secs: u64| t0 + Duration::from_secs(secs);
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    store
        .apply(generation_with(1, block(8, 0.6)))
        .expect("applied");
    assert!(store.heard_every(REPORT_PERIOD));
    // A hot rank beside a cool one: their mean is under the threshold.
    store.record(one, 1, pressure(0, 100), at(0), LoadSource::Stream);
    store.record(zero, 0, pressure(0, 900), at(0), LoadSource::Stream);
    assert_eq!(overloaded(&store, at(0)), []);
    // The cool rank falls silent. Three periods behind the engine's
    // newest receipt its sample is still the engine's word.
    store.record(zero, 0, pressure(0, 900), at(30), LoadSource::Stream);
    assert_eq!(overloaded(&store, at(30)), []);
    // Past them it ends, and the hot rank is judged alone.
    store.record(zero, 0, pressure(0, 900), at(31), LoadSource::Stream);
    assert_eq!(
        (overloaded(&store, at(31)), store.overload_flips()),
        (vec![zero, one], 1)
    );
    // Heard again, the rank's sample counts again.
    store.record(one, 1, pressure(0, 100), at(40), LoadSource::Stream);
    assert_eq!(
        (overloaded(&store, at(40)), store.overload_flips()),
        (vec![], 2)
    );
}

#[test]
fn a_stale_high_count_holds_its_veto_only_until_its_sample_ends() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let at = |secs: u64| t0 + Duration::from_secs(secs);
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    assert!(store.heard_every(REPORT_PERIOD));
    store.record(one, 1, pressure(9, 0), at(0), LoadSource::Stream);
    store.record(zero, 0, pressure(0, 0), at(0), LoadSource::Stream);
    assert_eq!(overloaded(&store, at(0)), [zero, one]);
    // The rank with the queue falls silent: inside the bound its count
    // holds the veto, past it the engine's silence takes it back.
    store.record(zero, 0, pressure(0, 0), at(30), LoadSource::Stream);
    assert_eq!(overloaded(&store, at(30)), [zero, one]);
    store.record(zero, 0, pressure(0, 0), at(31), LoadSource::Stream);
    assert_eq!(
        (overloaded(&store, at(31)), store.overload_flips()),
        (vec![], 2)
    );
}

#[test]
fn no_sample_ends_by_its_age_until_a_report_period_is_told() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let at = |secs: u64| t0 + Duration::from_secs(secs);
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    store.record(one, 1, pressure(9, 0), at(0), LoadSource::Stream);
    store.record(zero, 0, pressure(0, 0), at(1_000), LoadSource::Stream);
    // No period was told: the silent rank's count stands, however old.
    assert_eq!(overloaded(&store, at(1_000)), [zero, one]);
    // The telling judges every engine again under the period, and is
    // shown as a write with no instant of its own is: at the window's
    // end.
    assert!(store.heard_every(REPORT_PERIOD));
    store.publish(at(1_000));
    assert_eq!(overloaded(&store, at(1_000)), []);
    // A period is told once: the same one stands, another changes
    // nothing, and the next judgment reads the first.
    assert!(store.heard_every(REPORT_PERIOD));
    assert!(!store.heard_every(Duration::from_secs(1_000)));
    store.record(zero, 0, pressure(0, 0), at(1_001), LoadSource::Stream);
    assert_eq!(overloaded(&store, at(1_001)), []);
}

#[test]
fn an_engine_none_of_whose_ranks_is_heard_keeps_its_verdict() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let at = |secs: u64| t0 + Duration::from_secs(secs);
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    assert!(store.heard_every(REPORT_PERIOD));
    store.record(one, 1, pressure(9, 0), at(0), LoadSource::Stream);
    store.record(zero, 0, pressure(0, 0), at(0), LoadSource::Stream);
    // No rank is heard again: no write judges the engine, and a
    // membership write judges it from samples as old as each other.
    assert_eq!(overloaded(&store, at(1_000)), [zero, one]);
    member(&store, "http://b:8000");
    assert_eq!(overloaded(&store, at(1_000)), [zero, one]);
}

#[test]
fn a_membership_write_ages_each_sample_from_its_own_receipt() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let at = |secs: u64| t0 + Duration::from_secs(secs);
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    assert!(store.heard_every(REPORT_PERIOD));
    store.record(one, 1, pressure(9, 0), at(0), LoadSource::Stream);
    store.record(zero, 0, pressure(0, 0), at(31), LoadSource::Stream);
    assert_eq!(overloaded(&store, at(31)), []);
    // Every row is made again and every engine judged again: the ended
    // sample stays ended, by the receipts of the samples on file.
    member(&store, "http://b:8000");
    assert_eq!(overloaded(&store, at(31)), []);
}

/// The bound is the third period whole: a rank exactly three periods
/// behind its engine's newest is still counted, and one a microsecond
/// further is not.
#[test]
fn a_rank_exactly_three_periods_behind_is_counted_and_a_microsecond_past_it_ends() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    assert!(store.heard_every(REPORT_PERIOD));
    let bound = REPORT_PERIOD * 3;
    // Rank one holds the queue that vetoes. Rank zero is heard at the
    // bound: rank one's sample still counts.
    store.record(one, 1, pressure(9, 0), t0, LoadSource::Stream);
    store.record(zero, 0, pressure(0, 0), t0 + bound, LoadSource::Stream);
    assert_eq!(overloaded(&store, t0 + bound), [zero, one]);
    // A microsecond further, and it has ended.
    let past = t0 + bound + Duration::from_micros(1);
    store.record(zero, 0, pressure(0, 0), past, LoadSource::Stream);
    assert_eq!(overloaded(&store, past), []);
}

/// Beside a rank past the bound, a rank exactly at it is still counted: the
/// walk that leaves the ended rank out keeps the one three periods behind.
#[test]
fn a_rank_at_the_bound_is_counted_beside_one_past_it() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let zero = rank_among(&store, "grpc://e:9000", 0, 3);
    let one = rank_among(&store, "grpc://e:9000", 1, 3);
    let two = rank_among(&store, "grpc://e:9000", 2, 3);
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    assert!(store.heard_every(REPORT_PERIOD));
    let bound = REPORT_PERIOD * 3;
    let tick = Duration::from_micros(1);
    // Rank two is heard a microsecond before rank one, which holds the queue
    // that vetoes. Rank zero is heard three periods after rank one: rank two
    // has ended, and rank one, exactly at the bound, still counts.
    store.record(two, 2, pressure(0, 0), t0, LoadSource::Stream);
    store.record(one, 1, pressure(9, 0), t0 + tick, LoadSource::Stream);
    let now = t0 + tick + bound;
    store.record(zero, 0, pressure(0, 0), now, LoadSource::Stream);
    // The verdict is the engine's: every rank's row reads it, in the pool's
    // order, the ended rank's too.
    let mut ranks = [zero, one, two];
    ranks.sort_unstable();
    assert_eq!(overloaded(&store, now), ranks);
}

/// A membership write finds each rank's receipt among single-rank
/// members: an engine's ranks do not stand together in the id order the
/// write walks, and a single-rank member has no receipt.
#[test]
fn a_membership_write_finds_each_rank_s_receipt_among_single_rank_members() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let at = |secs: u64| t0 + Duration::from_secs(secs);
    member(&store, "http://a:8000");
    let zero = rank_of(&store, "grpc://e:9000", 0);
    member(&store, "http://b:8000");
    let one = rank_of(&store, "grpc://e:9000", 1);
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    assert!(store.heard_every(REPORT_PERIOD));
    store.record(one, 1, pressure(9, 0), at(0), LoadSource::Stream);
    store.record(zero, 0, pressure(0, 0), at(31), LoadSource::Stream);
    // The write makes every row again: rank one's sample stays ended.
    member(&store, "http://c:8000");
    assert_eq!(overloaded(&store, at(31)), []);
    // Heard again, rank one is counted beside rank zero's sample, which
    // the next write still reads nine seconds behind it.
    store.record(one, 1, pressure(9, 0), at(40), LoadSource::Stream);
    member(&store, "http://d:8000");
    assert_eq!(overloaded(&store, at(40)), [zero, one]);
}

#[test]
fn a_generation_applied_over_standing_rows_ages_each_sample_from_its_own_receipt() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let at = |secs: u64| t0 + Duration::from_secs(secs);
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    assert!(store.heard_every(REPORT_PERIOD));
    // No threshold stands yet. One rank falls silent with a queue on
    // file, the other is heard long after it, and no membership write
    // comes between.
    store.record(one, 1, pressure(9, 0), at(0), LoadSource::Stream);
    store.record(zero, 0, pressure(0, 0), at(0), LoadSource::Stream);
    member(&store, "http://b:8000");
    store.record(zero, 0, pressure(0, 0), at(100), LoadSource::Stream);
    assert_eq!(overloaded(&store, at(100)), []);
    // The generation's first judgment reads the receipts of the samples
    // on file: the silent rank's count ended long ago.
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    assert_eq!(overloaded(&store, at(100)), []);
    // Heard again, the rank's count is its engine's word.
    store.record(one, 1, pressure(9, 0), at(101), LoadSource::Stream);
    assert_eq!(overloaded(&store, at(101)), [zero, one]);
}

#[test]
fn a_poll_s_rows_are_heard_at_the_instant_their_poll_began() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let at = |secs: u64| t0 + Duration::from_secs(secs);
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    assert!(store.heard_every(REPORT_PERIOD));
    store.report(
        &[
            (zero, 0, Some(pressure(0, 0))),
            (one, 1, Some(pressure(9, 0))),
        ],
        at(0),
        LoadSource::Poll,
    );
    assert_eq!(overloaded(&store, at(0)), [zero, one]);
    // Later answers omit rank 1, which keeps its sample: it ends three
    // periods behind the poll that last brought its engine a row.
    let omitting = [(zero, 0, Some(pressure(0, 0))), (one, 1, None)];
    store.report(&omitting, at(30), LoadSource::Poll);
    assert_eq!(overloaded(&store, at(30)), [zero, one]);
    store.report(&omitting, at(31), LoadSource::Poll);
    // The engine's verdict ends with the sample. The rank its answers
    // leave out beside rank 0 stays avoided.
    assert_eq!(overloaded(&store, at(31)), [one]);
}

#[test]
fn a_single_rank_worker_s_sample_is_its_engine_s_newest_at_any_age() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let at = |secs: u64| t0 + Duration::from_secs(secs);
    let a = member(&store, "http://a:8000");
    let zero = rank_of(&store, "grpc://e:9000", 0);
    let one = rank_of(&store, "grpc://e:9000", 1);
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    assert!(store.heard_every(REPORT_PERIOD));
    // Beside a data-parallel engine that is heard long after it, a
    // single-rank worker's one sample stands: no other engine's receipt
    // ages it, at a record's write or at a membership write.
    store.record(a, 0, pressure(9, 0), at(0), LoadSource::Stream);
    store.record(zero, 0, pressure(0, 0), at(1_000), LoadSource::Stream);
    store.record(one, 1, pressure(0, 0), at(1_000), LoadSource::Stream);
    assert_eq!(overloaded(&store, at(1_000)), [a]);
    member(&store, "http://b:8000");
    assert_eq!(overloaded(&store, at(1_000)), [a]);
}

#[test]
fn a_membership_write_and_a_new_generation_judge_again_from_what_is_on_file() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let a = member(&store, "http://a:8000");
    store.record(a, 0, pressure(5, 0), t0, LoadSource::Stream);
    assert_eq!(overloaded(&store, t0), []);
    // The record was on file before any threshold: a generation that
    // hands the pool one judges it.
    store
        .apply(generation_with(1, block(5, 1.0)))
        .expect("applied");
    assert_eq!(overloaded(&store, t0), [a]);
    // A membership write makes every row again and carries no bit over:
    // the verdict stands because the record on file is judged again. The
    // pool's rows are cut once, after that judgment: the write counts no
    // publication's copy.
    let copies = store.publication_allocations();
    member(&store, "http://b:8000");
    assert_eq!(
        (overloaded(&store, t0), store.publication_allocations()),
        (vec![a], copies)
    );
    // A generation with a higher count takes the verdict back.
    store
        .apply(generation_with(2, block(6, 1.0)))
        .expect("applied");
    assert_eq!(overloaded(&store, t0), []);
    assert_eq!(
        store.overload_flips(),
        0,
        "no write of the load table moved one"
    );
}

#[test]
fn the_deciding_door_hands_the_rows_and_the_engine_of_one_publication() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    member(&store, "http://a:8000");
    store.apply(generation(1)).expect("above the unbound start");
    store.deciding(|world| {
        // A generation applied and a member admitted while a decision holds
        // its world are the next decision's: this one reads the rows and
        // the engine it was handed, whole.
        store.apply(generation(2)).expect("above the standing one");
        member(&store, "http://b:8000");
        let mut out = ViewBuffers::default();
        let built = world
            .build(MODEL, Role::Single, &ask(t0), &mut out)
            .expect("the pool");
        assert_eq!(
            (built.generation, world.spec().generation(), built.pool_size),
            (1, 1, 1)
        );
        assert_eq!(world.spec().engine().generation(), 1);
    });
    let next = store.deciding(|world| {
        (
            world.spec().generation(),
            world.members(MODEL, Role::Single),
        )
    });
    assert_eq!(next, (2, Some(2)));
}

/// How many rows of a pool hold no report ([`Built::unreported`]), kept by
/// the writes and shown where the rows are published: a member before its
/// first report and a member whose report was dropped each count, and a
/// report without a rate does not. The build and the lent view read one
/// number, and the pool facts say that every row holds a report where it
/// is 0 ([`PoolFacts::reported`]).
#[test]
fn a_publication_counts_the_rows_of_a_pool_that_hold_no_report() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let at = |turn: u32| t0 + PUBLISH_WINDOW * turn;
    let rated = |live_rate: u32| LoadRow {
        live_rate,
        ..load(1, 0)
    };
    let unreported = |now: Instant| {
        let (asked, mut out) = (ask(now), ViewBuffers::default());
        let built = store
            .build_view(MODEL, Role::Single, &asked, &mut out)
            .expect("the pool");
        let lent = store.with_world(|world| {
            let view = world.view(MODEL, Role::Single, &asked, false, &mut out);
            view.expect("the pool").built.unreported
        });
        assert_eq!(built.unreported, lent);
        assert_eq!(built.pool_facts().reported, lent == 0);
        lent
    };
    let first = member(&store, "http://a:8000");
    let second = member(&store, "http://b:8000");
    // A membership write counts its rows: two members, no report yet.
    assert_eq!(unreported(t0), 2);
    store.record(first, 0, rated(1_900), at(1), LoadSource::Stream);
    assert_eq!(unreported(at(1)), 1);
    // A report without a rate is a report: the pool's last row holds one,
    // and the next with a rate changes nothing.
    store.record(second, 0, rated(0), at(2), LoadSource::Stream);
    assert_eq!(unreported(at(2)), 0);
    store.record(second, 0, rated(1_900), at(3), LoadSource::Stream);
    assert_eq!(unreported(at(3)), 0);
    assert_eq!(unreported(at(4)), 0);
    // A member whose report is dropped holds none from the next window on.
    store.evict(first);
    store.publish(at(5));
    assert_eq!(unreported(at(5)), 1);
    // A member that joins is counted beside the rows that were there.
    member(&store, "http://c:8000");
    assert_eq!(unreported(at(5)), 2);
}

/// The count is each pool's own, moved by the row a write folds and taken
/// again where a membership write builds the rows. A member of two models
/// moves both pools by its one report; a report on a row that holds one and
/// a drop on a row that holds none move nothing; a membership write counts
/// the reports its rows carry. No publication takes a pass for it: a debug
/// build holds each publication's count to a pass over its rows.
#[test]
fn a_report_moves_the_count_of_each_pool_its_row_stands_in() {
    const OTHER: &str = "n";
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let at = |turn: u32| t0 + PUBLISH_WINDOW * turn;
    let counts = |now: Instant| {
        let count = |model: &str| {
            let (asked, mut out) = (ask(now), ViewBuffers::default());
            let built = store.build_view(model, Role::Single, &asked, &mut out);
            built.expect("the pool").unreported
        };
        (count(MODEL), count(OTHER))
    };
    let both = admit(
        &store,
        facts("http://a:8000", Role::Single, &[MODEL, OTHER]),
    );
    let one = member(&store, "http://b:8000");
    assert_eq!(counts(t0), (2, 1));
    store.record(both, 0, load(1, 0), at(1), LoadSource::Stream);
    assert_eq!(counts(at(1)), (1, 0));
    // A report on a row that holds one moves nothing.
    store.record(both, 0, load(2, 0), at(2), LoadSource::Stream);
    assert_eq!(counts(at(2)), (1, 0));
    store.record(one, 0, load(1, 0), at(3), LoadSource::Stream);
    assert_eq!(counts(at(3)), (0, 0));
    // A dropped report leaves both pools, once.
    store.evict(both);
    store.publish(at(4));
    assert_eq!(counts(at(4)), (1, 1));
    store.evict(both);
    store.publish(at(5));
    assert_eq!(counts(at(5)), (1, 1));
    // A membership write builds every pool's rows again and counts the
    // reports they carry: the member that joins the second model holds
    // none, and the first model's one report stands.
    admit(&store, facts("http://c:8000", Role::Single, &[OTHER]));
    assert_eq!(counts(at(5)), (1, 2));
    store.record(both, 0, load(1, 0), at(6), LoadSource::Stream);
    assert_eq!(counts(at(6)), (0, 1));
}

/// One publication that writes two pools takes the list of counts again
/// where either pool's count moved. A pool whose count stands, written in
/// the same publication as one whose count moved, does not keep the old
/// list for both: held in both orders of the two pools.
#[test]
fn a_publication_of_two_pools_lists_the_count_that_moved_in_either() {
    const OTHER: &str = "n";
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let at = |turn: u32| t0 + PUBLISH_WINDOW * turn;
    let counts = |now: Instant| {
        let count = |model: &str| {
            let (asked, mut out) = (ask(now), ViewBuffers::default());
            let built = store.build_view(model, Role::Single, &asked, &mut out);
            built.expect("the pool").unreported
        };
        (count(MODEL), count(OTHER))
    };
    let first = member(&store, "http://a:8000");
    let second = admit(&store, facts("http://b:8000", Role::Single, &[OTHER]));
    assert_eq!(counts(t0), (1, 1));
    store.record(second, 0, load(1, 0), at(1), LoadSource::Stream);
    assert_eq!(counts(at(1)), (1, 0));
    // One window holds a first report in the first pool and one more in
    // the second: the first pool's count moves and the second's stands.
    let inside = at(1) + Duration::from_millis(1);
    store.record(first, 0, load(1, 0), inside, LoadSource::Stream);
    store.record(second, 0, load(2, 0), inside, LoadSource::Stream);
    store.publish(at(2));
    assert_eq!(counts(at(2)), (0, 0));
    // The other order. A member joins the second pool, and one window
    // holds one more report in the first pool and a first in the second.
    let third = admit(&store, facts("http://c:8000", Role::Single, &[OTHER]));
    assert_eq!(counts(at(2)), (0, 1));
    let inside = at(2) + Duration::from_millis(1);
    store.record(first, 0, load(2, 0), inside, LoadSource::Stream);
    store.record(third, 0, load(1, 0), inside, LoadSource::Stream);
    store.publish(at(3));
    assert_eq!(counts(at(3)), (0, 0));
}

/// Where the published rows of the model's single pool stand.
fn published(world: &World) -> *const Candidate {
    rows_of(world, MODEL).as_ptr()
}

#[test]
fn a_pool_every_gateway_reaches_whole_is_lent_where_it_stands_and_copied_when_asked() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let first = member(&store, "http://a:8000");
    let second = member(&store, "http://b:8000");
    store.gateway_veto(ASKER, second, Some(now + Duration::from_secs(1)));
    let asked = Ask {
        pinned: Some(first),
        ..ask(now)
    };
    let mut out = ViewBuffers::default();
    store.with_world(|world| {
        let view = world
            .view(MODEL, Role::Single, &asked, false, &mut out)
            .expect("the pool");
        // The rows are the publication's own, and nothing of the decision
        // is in them: the veto and the pin stand beside them.
        assert_eq!(view.candidates.as_ptr(), published(world));
        let own = |row: &Candidate| !row.health.stalled && !row.session_hit;
        assert!(view.candidates.iter().all(own));
        let at = |id: WorkerId| {
            let found = view.candidates.iter().position(|row| row.id == id);
            found.expect("a member") as u32
        };
        assert_eq!(view.stalled, [at(second)]);
        assert_eq!(view.pinned_at, at(first) + 1);
        assert_eq!(view.built.pool_size, 2);
    });
    assert!(out.candidates.is_empty(), "a lent view copies no row");

    // A decision that asks for written rows is handed the build's copy:
    // the bits are in the rows and nothing stands beside them.
    store.with_world(|world| {
        let view = world
            .view(MODEL, Role::Single, &asked, true, &mut out)
            .expect("the pool");
        assert_ne!(view.candidates.as_ptr(), published(world));
        assert!(view.stalled.is_empty());
        assert_eq!(view.pinned_at, 0);
        let bits = |id: WorkerId| {
            let found = view.candidates.iter().find(|row| row.id == id);
            let row = found.expect("a member");
            (row.health.stalled, row.session_hit)
        };
        assert_eq!((bits(first), bits(second)), ((false, true), (true, false)));
    });

    // No member, no view, and the buffers are cleared.
    store.with_world(|world| {
        let none = world.view("another", Role::Single, &asked, false, &mut out);
        assert!(none.is_none());
    });
    assert!(out.candidates.is_empty() && out.stalled.is_empty());
    // A build that finds no pool clears the mask a lent view left too.
    out.stalled.push(7);
    store.with_world(|world| {
        let none = world.build("another", Role::Single, &asked, &mut out);
        assert!(none.is_none());
    });
    assert!(out.candidates.is_empty() && out.stalled.is_empty());
}

#[test]
fn a_pin_that_names_no_member_of_the_pool_pins_no_lent_row() {
    // Workers of another model stand among this pool's members in the
    // store's order: the search for one of them over the pool's rows ends
    // between two rows, and not past the last.
    let store = MemoryStore::new();
    let now = Instant::now();
    let (mut members, mut others) = (Vec::new(), Vec::new());
    for index in 0..4 {
        members.push(member(&store, &format!("http://m{index}:8000")));
        let other = facts(&format!("http://o{index}:8000"), Role::Single, &["another"]);
        others.push(admit(&store, other));
    }
    let last = members.iter().copied().max().expect("four members");
    assert!(
        others.iter().any(|&id| id < last),
        "a worker of the other model sorts among the members"
    );
    let mut out = ViewBuffers::default();
    for outsider in others {
        let asked = Ask {
            pinned: Some(outsider),
            ..ask(now)
        };
        // The lent rows with what stands beside them are the build's rows
        // (the helper holds them to it), and the build wrote no session bit.
        let (rows, _, _) = view(&store, &asked);
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|row| !row.session_hit));
        store.with_world(|world| {
            let view = world
                .view(MODEL, Role::Single, &asked, false, &mut out)
                .expect("the pool");
            assert_eq!(view.candidates.as_ptr(), published(world));
            assert_eq!(view.pinned_at, 0);
        });
    }
}

#[test]
fn a_pool_that_holds_a_host_local_row_is_copied_for_every_gateway() {
    let store = MemoryStore::new();
    let now = Instant::now();
    member(&store, "http://a:8000");
    let local = admit(
        &store,
        WorkerFacts {
            reachable_from: Reachability::Gateway(GatewayId(2)),
            ..facts("unix:///b", Role::Single, &[MODEL])
        },
    );
    let mut out = ViewBuffers::default();
    for (gateway, rows) in [(ASKER, 1), (GatewayId(2), 2)] {
        let asked = Ask {
            gateway,
            ..ask(now)
        };
        store.with_world(|world| {
            let view = world
                .view(MODEL, Role::Single, &asked, false, &mut out)
                .expect("the pool");
            // The asking gateway's positions are its own here: its copy.
            assert_ne!(view.candidates.as_ptr(), published(world));
            assert_eq!(view.candidates.len(), rows);
            let holds = view.candidates.iter().any(|row| row.id == local);
            assert_eq!(holds, gateway == GatewayId(2));
            assert!(view.stalled.is_empty() && view.pinned_at == 0);
        });
    }
}

#[test]
fn a_lent_view_hands_its_own_gateways_standing_vetoes_in_the_rows_order() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let mut ids: Vec<WorkerId> = (0..6)
        .map(|n| member(&store, &format!("http://w{n}:8000")))
        .collect();
    // The pool's order: a row's position is its worker's place by id.
    ids.sort_unstable();
    let end = t0 + Duration::from_secs(10);
    let early = t0 + Duration::from_secs(1);
    // Set in falling order, one that ends early, one of another gateway.
    store.gateway_veto(ASKER, ids[4], Some(end));
    store.gateway_veto(ASKER, ids[1], Some(end));
    store.gateway_veto(ASKER, ids[2], Some(early));
    store.gateway_veto(GatewayId(2), ids[0], Some(end));
    let masked = |gateway: GatewayId, now: Instant| {
        let mut out = ViewBuffers::default();
        let asked = Ask {
            gateway,
            ..ask(now)
        };
        store.with_world(|world| {
            let view = world.view(MODEL, Role::Single, &asked, false, &mut out);
            view.expect("the pool").stalled.to_vec()
        })
    };
    assert_eq!(masked(ASKER, t0), [1, 2, 4]);
    // A veto past its end masks nothing, from its own instant on.
    assert_eq!(masked(ASKER, early), [1, 4]);
    assert!(masked(ASKER, end).is_empty());
    // No other gateway's run is read.
    assert_eq!(masked(GatewayId(2), t0), [0]);
    assert!(masked(GatewayId(3), t0).is_empty());
}

#[test]
fn a_dropped_gateways_vetoes_leave_in_one_publication_and_every_other_stands() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + Duration::from_secs(60);
    let one = member(&store, "http://a:8000");
    let two = member(&store, "http://b:8000");
    let other = GatewayId(2);
    store.gateway_veto(ASKER, one, Some(until));
    store.gateway_veto(ASKER, two, Some(until));
    store.gateway_veto(other, two, Some(until));
    let mut both = [one, two];
    both.sort_unstable();
    assert_eq!(stalled(&store, ASKER, t0), both);

    let before = store.world();
    store.drop_gateways(&[ASKER]);
    assert_eq!(stalled(&store, ASKER, t0), []);
    assert_eq!(
        stalled(&store, other, t0),
        [two],
        "another gateway's stands"
    );
    // Dropped again, and a gateway that never vetoed: no world is made.
    let lifted = store.world();
    assert!(
        !Arc::ptr_eq(&before, &lifted),
        "the lifting is a publication"
    );
    store.drop_gateways(&[ASKER]);
    store.drop_gateways(&[GatewayId(3)]);
    assert!(Arc::ptr_eq(&lifted, &store.world()));
}

#[test]
fn gateways_dropped_together_leave_in_one_publication() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + Duration::from_secs(60);
    let one = member(&store, "http://a:8000");
    let two = member(&store, "http://b:8000");
    let (second, other) = (GatewayId(7), GatewayId(8));
    store.gateway_veto(ASKER, one, Some(until));
    store.gateway_veto(second, two, Some(until));
    store.gateway_veto(other, two, Some(until));

    let published = store.publications();
    store.drop_gateways(&[second, ASKER]);
    assert_eq!(store.publications(), published + 1, "one world for the two");
    assert_eq!(stalled(&store, ASKER, t0), []);
    assert_eq!(stalled(&store, second, t0), []);
    assert_eq!(
        stalled(&store, other, t0),
        [two],
        "another gateway's stands"
    );
    // Gateways that hold no veto, and no gateway at all: no world is made.
    store.drop_gateways(&[second, ASKER, GatewayId(9)]);
    store.drop_gateways(&[]);
    assert_eq!(store.publications(), published + 1);
    // A gateway that holds no veto, listed ahead of one that does, hides
    // nothing: the holder's veto leaves in a world.
    store.drop_gateways(&[GatewayId(2), other]);
    assert_eq!(store.publications(), published + 2);
    assert_eq!(stalled(&store, other, t0), []);
}

/// A generation that declares the view's three lists.
fn generation_listing(
    number: u64,
    labels: &[&str],
    zones: &[&str],
    engines: &[&str],
) -> AppliedSpec {
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = number;
    let owned = |list: &[&str]| list.iter().map(|name| (*name).to_string()).collect();
    document.view.labels = owned(labels);
    document.view.zones = owned(zones);
    document.view.engines = owned(engines);
    AppliedSpec::validate(&document).expect("the document validates")
}

/// A worker of the model that names an engine family, a zone and labels.
fn listed_member(
    store: &MemoryStore,
    key: &str,
    engine: &str,
    zone: Option<&str>,
    labels: &[(&str, &str)],
) -> WorkerId {
    let labels = labels.iter();
    let facts = WorkerFacts {
        engine_kind: engine.to_string(),
        zone: zone.map(str::to_string),
        labels: labels
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect(),
        ..facts(key, Role::Single, &[MODEL])
    };
    admit(store, facts)
}

/// What a member's row reads in the three listed fields: its engine family,
/// its zone, its labels.
fn listed(world: &World, id: WorkerId) -> (u16, u16, u64) {
    let rows = rows_of(world, MODEL);
    let row = rows.iter().find(|row| row.id == id);
    let row = row.expect("the member's row");
    (row.engine_kind, row.zone, row.labels)
}

#[test]
fn a_row_names_its_engine_family_and_zone_by_position_and_holds_a_bit_a_label() {
    let store = MemoryStore::new();
    let lists = generation_listing(
        1,
        &["tier", "tier=gold", "pool=a"],
        &["east", "west"],
        &["alpha", "beta"],
    );
    store.apply(lists).expect("applied");
    let gold = listed_member(&store, "gold", "beta", Some("west"), &[("tier", "gold")]);
    let labels = [("tier", "silver"), ("pool", "a")];
    let silver = listed_member(&store, "silver", "alpha", None, &labels);
    // A position counts from 1 in the list's order. A bit a selector, in
    // the list's order: a key alone is met by any value of it.
    let world = store.world();
    assert_eq!(listed(&world, gold), (2, 2, 0b011));
    assert_eq!(listed(&world, silver), (1, 0, 0b101));
    assert_eq!(store.outside_lists(), OutsideLists::default());
    // A family or a zone the lists do not name is admitted, reads 0 and is
    // counted under its list. A worker that names neither is not counted.
    let stray = listed_member(&store, "stray", "gamma", Some("north"), &[("pool", "b")]);
    let bare = listed_member(&store, "bare", "", None, &[]);
    let world = store.world();
    assert_eq!(listed(&world, stray), (0, 0, 0));
    assert_eq!(listed(&world, bare), (0, 0, 0));
    assert_eq!(listed(&world, gold), (2, 2, 0b011));
    let outside = OutsideLists {
        engines: 1,
        zones: 1,
    };
    assert_eq!(store.outside_lists(), outside);
}

#[test]
fn a_worker_of_two_models_reads_the_lists_in_each_of_its_rows() {
    let store = MemoryStore::new();
    let zones = ["east", "west"];
    let engines = ["alpha", "beta"];
    store
        .apply(generation_listing(1, &["tier=gold"], &zones, &engines))
        .expect("applied");
    let both = WorkerFacts {
        engine_kind: "beta".to_string(),
        zone: Some("west".to_string()),
        labels: [("tier".to_string(), "gold".to_string())].into(),
        ..facts("both", Role::Single, &[MODEL, "second"])
    };
    let id = admit(&store, both);
    // One row a model, and each row reads the member's three fields.
    let read = |world: &World, model: &str| {
        let rows = rows_of(world, model);
        let row = rows.iter().find(|row| row.id == id);
        let row = row.expect("the member's row");
        (row.engine_kind, row.zone, row.labels)
    };
    let world = store.world();
    assert_eq!(read(&world, MODEL), (2, 2, 0b1));
    assert_eq!(read(&world, "second"), (2, 2, 0b1));
    // An apply whose lists differ reads both rows again.
    store
        .apply(generation_listing(2, &["tier=gold"], &["west"], &["beta"]))
        .expect("applied");
    let world = store.world();
    assert_eq!(read(&world, MODEL), (1, 1, 0b1));
    assert_eq!(read(&world, "second"), (1, 1, 0b1));
    assert_eq!(store.outside_lists(), OutsideLists::default());
    // An apply whose lists name neither of its fields counts the member once
    // a list, not once a row.
    store
        .apply(generation_listing(3, &["tier=gold"], &["east"], &["alpha"]))
        .expect("applied");
    let world = store.world();
    assert_eq!(read(&world, MODEL), (0, 0, 0b1));
    assert_eq!(read(&world, "second"), (0, 0, 0b1));
    let outside = OutsideLists {
        engines: 1,
        zones: 1,
    };
    assert_eq!(store.outside_lists(), outside);
    // A membership write counts at its own read of the members, at the build
    // it leaves owed: the worker is counted once there too, beside a worker
    // both lists name.
    let inside = WorkerFacts {
        engine_kind: "alpha".to_string(),
        zone: Some("east".to_string()),
        ..facts("inside", Role::Single, &[MODEL])
    };
    admit(&store, inside);
    assert_eq!(store.outside_lists(), outside);
}

/// The one door's pair is of one moment. Each member joins outside the
/// engines' list and then reports over its bound, so at every moment the
/// flips trail the members outside by one at most, and a second thread reads
/// the pair while the walk runs. Two holds of the lock would hand a reader a
/// pair no moment held.
#[test]
fn a_pair_read_beside_a_walk_of_joins_and_reports_is_of_one_moment() {
    const MEMBERS: u64 = 1_500;
    let store = Arc::new(MemoryStore::new());
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = 1;
    document.view.engines = vec!["alpha".to_string()];
    let profile = document
        .profiles
        .keys()
        .next()
        .expect("the preset's profile");
    document.bindings = vec![BindingSpec {
        profile: Some(profile.clone()),
        overload: Some(block(8, 1.0)),
        ..BindingSpec::default()
    }];
    let listing = AppliedSpec::validate(&document).expect("the document validates");
    store.apply(listing).expect("applied");
    let walk = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || {
            for at in 0..MEMBERS {
                let joins = WorkerFacts {
                    engine_kind: "beta".to_string(),
                    ..facts(&format!("w{at}"), Role::Single, &[MODEL])
                };
                let id = admit(&store, joins);
                let over = [(id, 0, Some(pressure(9, 0)))];
                store.report(&over, Instant::now(), LoadSource::Poll);
            }
        })
    };
    while !walk.is_finished() {
        let counts = store.counts();
        let (outside, flips) = (u64::from(counts.outside.engines), counts.overload_flips);
        assert!(
            flips <= outside && outside <= flips + 1,
            "a pair of no moment: {outside} outside beside {flips} flips"
        );
    }
    walk.join().expect("the walk");
    let end = store.counts();
    assert_eq!(
        (u64::from(end.outside.engines), end.overload_flips),
        (MEMBERS, MEMBERS)
    );
}

#[test]
fn an_apply_whose_lists_differ_reads_every_member_again_in_its_first_world() {
    let store = MemoryStore::new();
    let west = listed_member(&store, "w", "beta", Some("west"), &[("tier", "gold")]);
    let north = listed_member(&store, "n", "beta", Some("north"), &[]);
    // No generation, no list: every field reads 0 and nothing is outside.
    assert_eq!(listed(&store.world(), west), (0, 0, 0));
    assert_eq!(store.outside_lists(), OutsideLists::default());
    let zones = ["east", "west"];
    let engines = ["alpha", "beta"];
    store
        .apply(generation_listing(1, &["tier=gold"], &zones, &engines))
        .expect("applied");
    // The apply's own world holds the rows of its own lists.
    let first = store.world();
    assert_eq!(first.spec().generation(), 1);
    assert_eq!(listed(&first, west), (2, 2, 0b1));
    assert_eq!(listed(&first, north), (2, 0, 0));
    let outside = OutsideLists {
        engines: 0,
        zones: 1,
    };
    assert_eq!(store.outside_lists(), outside);
    // The same lists under the next number: no member is read again, and
    // the pool's rows are the ones the last world listed.
    store
        .apply(generation_listing(2, &["tier=gold"], &zones, &engines))
        .expect("applied");
    let second = store.world();
    assert_eq!(second.spec().generation(), 2);
    assert!(Arc::ptr_eq(
        &rows_of(&first, MODEL),
        &rows_of(&second, MODEL)
    ));
    assert_eq!(store.outside_lists(), outside);
    // Other lists: the positions and the bits move with them.
    store
        .apply(generation_listing(
            3,
            &["zone", "tier=gold"],
            &["north", "west"],
            &["beta"],
        ))
        .expect("applied");
    let third = store.world();
    assert_eq!(listed(&third, west), (1, 2, 0b10));
    assert_eq!(listed(&third, north), (1, 1, 0));
    assert_eq!(store.outside_lists(), OutsideLists::default());
    // A generation that declares none: every field reads 0 again.
    store.apply(generation(4)).expect("applied");
    let fourth = store.world();
    assert_eq!(listed(&fourth, west), (0, 0, 0));
    assert_eq!(listed(&fourth, north), (0, 0, 0));
    assert_eq!(store.outside_lists(), OutsideLists::default());
}

/// A generation that declares zones and engine families under the pairing
/// `mode`.
fn generation_listing_pairing(
    number: u64,
    mode: Pairing,
    zones: &[&str],
    engines: &[&str],
) -> AppliedSpec {
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = number;
    document.plans.pd_pair.pairing = mode;
    let owned = |list: &[&str]| list.iter().map(|name| (*name).to_string()).collect();
    document.view.zones = owned(zones);
    document.view.engines = owned(engines);
    AppliedSpec::validate(&document).expect("the document validates")
}

#[test]
fn an_apply_that_derives_the_classes_again_reads_every_member_under_its_own_lists() {
    let store = MemoryStore::new();
    let west = listed_member(&store, "w", "beta", Some("west"), &[]);
    let north = listed_member(&store, "n", "gamma", Some("north"), &[]);
    let lenient = Pairing::default();
    store
        .apply(generation_listing_pairing(1, lenient, &["west"], &["beta"]))
        .expect("applied");
    assert_eq!(listed(&store.world(), west), (1, 1, 0));
    let outside = OutsideLists {
        engines: 1,
        zones: 1,
    };
    assert_eq!(store.outside_lists(), outside);
    // Another mode and other lists in one generation: the classes are
    // derived by the membership write's door, whose rows are new, and its
    // one world holds them under the generation's own lists.
    let zones = ["north", "west"];
    store
        .apply(generation_listing_pairing(
            2,
            Pairing::Off,
            &zones,
            &["gamma"],
        ))
        .expect("applied");
    let second = store.world();
    assert_eq!(second.spec().generation(), 2);
    assert_eq!(listed(&second, north), (1, 1, 0));
    assert_eq!(listed(&second, west), (0, 2, 0));
    let outside = OutsideLists {
        engines: 1,
        zones: 0,
    };
    assert_eq!(store.outside_lists(), outside);
    // Another mode and no list: every field reads 0, and the member the
    // last generation counted outside a list is counted no more.
    store
        .apply(generation_pairing(3, Pairing::Strict))
        .expect("applied");
    let third = store.world();
    assert_eq!(listed(&third, north), (0, 0, 0));
    assert_eq!(listed(&third, west), (0, 0, 0));
    assert_eq!(store.outside_lists(), OutsideLists::default());
}

#[test]
fn a_membership_write_reads_the_member_it_writes_and_carries_every_others_fields() {
    let store = MemoryStore::new();
    // A document of zones and families alone lists a member too.
    let zones = ["east", "west"];
    let engines = ["alpha", "beta"];
    store
        .apply(generation_listing(1, &[], &zones, &engines))
        .expect("applied");
    let west = listed_member(&store, "w", "beta", Some("west"), &[("tier", "gold")]);
    let east = listed_member(&store, "e", "alpha", Some("east"), &[]);
    let world = store.world();
    assert_eq!(listed(&world, west), (2, 2, 0));
    assert_eq!(listed(&world, east), (1, 1, 0));
    // A refresh that states another zone and family moves the member's
    // positions and its count: the write reads the member it writes.
    let again = listed_member(&store, "w", "alpha", Some("north"), &[("tier", "gold")]);
    assert_eq!(again, west);
    let world = store.world();
    assert_eq!(listed(&world, west), (1, 0, 0));
    assert_eq!(listed(&world, east), (1, 1, 0));
    let outside = OutsideLists {
        engines: 0,
        zones: 1,
    };
    assert_eq!(store.outside_lists(), outside);
    // A patch of one member's labels moves its bits and no other's.
    store
        .apply(generation_listing(
            2,
            &["tier=gold", "lane"],
            &zones,
            &engines,
        ))
        .expect("applied");
    assert_eq!(listed(&store.world(), west), (1, 0, 0b01));
    let row = WorkersStore::get(&store, east).expect("the row");
    let lane = [("lane".to_string(), "a".to_string())];
    let patch = WorkerPatch {
        labels: Some(lane.into_iter().collect()),
        priority: None,
        overload: None,
    };
    store.patch(east, patch, row.revision).expect("patched");
    store.publish_now();
    let world = store.world();
    assert_eq!(listed(&world, east), (1, 1, 0b10));
    assert_eq!(listed(&world, west), (1, 0, 0b01));
    // A member that leaves takes its count with it, and the member that
    // stays reads as before.
    let row = WorkersStore::get(&store, west).expect("the row");
    store.remove(west, row.revision).expect("removed");
    store.publish_now();
    assert_eq!(listed(&store.world(), east), (1, 1, 0b10));
    assert_eq!(store.outside_lists(), OutsideLists::default());
    // It joins again under other facts, and reads as those say.
    let back = listed_member(&store, "w", "beta", Some("west"), &[]);
    assert_eq!(listed(&store.world(), back), (2, 2, 0));
}

#[test]
fn a_member_back_under_its_id_is_read_under_the_lists_again() {
    let store = MemoryStore::new();
    store
        .apply(generation_listing(1, &[], &["east", "west"], &["alpha"]))
        .expect("applied");
    let stays = listed_member(&store, "e", "alpha", Some("east"), &[]);
    let leaves = listed_member(&store, "w", "alpha", Some("west"), &[]);
    assert_eq!(listed(&store.world(), leaves), (1, 2, 0));
    let row = WorkersStore::get(&store, leaves).expect("the row");
    store.remove(leaves, row.revision).expect("removed");
    // The same key and incarnation, stating a zone the list does not name,
    // registered after the leave: the same id, and its row stands at the
    // revision the one that left was read at.
    let north = WorkerFacts {
        engine_kind: "alpha".to_string(),
        zone: Some("north".to_string()),
        ..facts("w", Role::Single, &[MODEL])
    };
    let back = store.upsert(north).id();
    let again = WorkersStore::get(&store, back).expect("the row");
    assert_eq!((back, again.revision), (leaves, row.revision));
    store.publish_now();
    // The build reads the registration that stands: no field of the one
    // that left is carried, and the count outside the list follows.
    let world = store.world();
    assert_eq!(listed(&world, back), (1, 0, 0));
    assert_eq!(listed(&world, stays), (1, 1, 0));
    let outside = OutsideLists {
        engines: 0,
        zones: 1,
    };
    assert_eq!(store.outside_lists(), outside);
}

#[test]
fn a_member_whose_rows_revision_stands_is_carried_and_one_whose_revision_moved_is_read() {
    let origin = Instant::now();
    let row = |id: u64, zone: &str, revision: u64| WorkerRow {
        id: WorkerId(id),
        facts: WorkerFacts {
            zone: Some(zone.to_string()),
            ..facts(&format!("http://w{id}:8000"), Role::Single, &[MODEL])
        },
        status: WorkerStatus::Ready,
        revision,
        admitted_at: origin,
        maker: Default::default(),
    };
    let zones = |world: &World| (listed(world, WorkerId(3)).1, listed(world, WorkerId(7)).1);
    let mut writer = Writer::new(PUBLISH_WINDOW, origin);
    let lists = Arc::new(generation_listing(1, &[], &["east", "west"], &[]));
    writer
        .apply(lists, 0, std::iter::empty(), |_, _| None)
        .expect("applied");
    let members = [row(3, "east", 1), row(7, "west", 1)];
    let first = writer.rebuild(1, members.filed(&writer), |_, _| None, Classes::Derive);
    assert_eq!(zones(&first), (1, 2));
    // The revision is what says a row's facts moved. At a standing revision
    // a member's fields are the last read's, whatever the row states: a
    // store hands no such row, and the case does, to tell a carry from a
    // read.
    let members = [row(3, "west", 1), row(7, "east", 1)];
    let carried = writer.rebuild(2, members.filed(&writer), |_, _| None, Classes::Derive);
    assert_eq!(zones(&carried), (1, 2));
    // At another revision a member is read, and the other is carried.
    let members = [row(3, "west", 1), row(7, "east", 2)];
    let read = writer.rebuild(3, members.filed(&writer), |_, _| None, Classes::Derive);
    assert_eq!(zones(&read), (1, 1));
    // An apply whose lists differ reads every member, at any revision and
    // in whatever order the store's table hands them,
    let other = Arc::new(generation_listing(2, &[], &["west", "east"], &[]));
    let members = [row(7, "east", 2), row(3, "east", 1)];
    let (relisted, _) = writer
        .apply(other, 3, members.filed(&writer), |_, _| None)
        .expect("applied");
    assert_eq!(zones(&relisted), (2, 2));
    // and the write after it carries that reading.
    let members = [row(3, "west", 1), row(7, "west", 2)];
    let after = writer.rebuild(4, members.filed(&writer), |_, _| None, Classes::Derive);
    assert_eq!(zones(&after), (2, 2));
    // A member that leaves takes its entry with it, on either side of a
    // member that stays. One that joins again at the id and the revision
    // it left at is read: no field of its last life is carried.
    let members = [row(7, "west", 2)];
    let left = writer.rebuild(5, members.filed(&writer), |_, _| None, Classes::Derive);
    assert_eq!(listed(&left, WorkerId(7)).1, 2);
    let members = [row(3, "west", 1), row(7, "west", 2)];
    let back = writer.rebuild(6, members.filed(&writer), |_, _| None, Classes::Derive);
    assert_eq!(zones(&back), (1, 2));
    let members = [row(3, "west", 1)];
    let left = writer.rebuild(7, members.filed(&writer), |_, _| None, Classes::Derive);
    assert_eq!(listed(&left, WorkerId(3)).1, 1);
    let members = [row(3, "west", 1), row(7, "west", 2)];
    let back = writer.rebuild(8, members.filed(&writer), |_, _| None, Classes::Derive);
    assert_eq!(zones(&back), (1, 1));
}

/// A side's whole entries are a list of its own, made once at its length.
/// The last model's side that lists every member is the entries
/// themselves: none is copied.
#[test]
fn the_last_models_side_of_every_member_takes_the_entries_whole() {
    let mut entries: Vec<u64> = (10..14).collect();
    let stood = entries.as_ptr();
    // Some members of the last model, then every member of an earlier one.
    let some = side_of(&mut entries, &[1, 3], true);
    assert_eq!((some.as_slice(), some.capacity()), (&[11, 13][..], 2));
    let every = side_of(&mut entries, &[0, 1, 2, 3], false);
    assert_eq!(every, [10, 11, 12, 13]);
    assert!(every.as_ptr() != stood && entries.len() == 4);
    let whole = side_of(&mut entries, &[0, 1, 2, 3], true);
    assert_eq!(whole, [10, 11, 12, 13]);
    assert!(whole.as_ptr() == stood && entries.is_empty());
}
