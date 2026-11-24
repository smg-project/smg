//! The cases of a publication's reader of one gateway's standing vetoes by
//! worker id ([`StoreSnapshot::gateway_vetoes`]): what a veto written
//! through the store's door reads as.
//!
//! Every case reads the reader's answer as a worker's id with whether the
//! veto's hold stands, never as a row of a pool.

use std::time::{Duration, Instant};

use super::{super::StandingVeto, admit, facts, load, member, StoreSnapshot, ASKER, MODEL};
use crate::{
    engine::view::{Role, WorkerId},
    store::{
        GatewayId, HealthSink, LeasesStore, LoadSink, LoadSource, MemoryStore, Reachability,
        VetoWrite, WorkerFacts, WorkersStore,
    },
};

/// A gateway that asks beside [`ASKER`].
const OTHER: GatewayId = GatewayId(2);

const MS: Duration = Duration::from_millis(1);
const S: Duration = Duration::from_secs(1);

/// A standing veto whose hold stands.
fn hold(id: WorkerId) -> StandingVeto {
    StandingVeto { id, held: true }
}

/// A standing veto with no hold, or past its hold's end.
fn doubt(id: WorkerId) -> StandingVeto {
    StandingVeto { id, held: false }
}

/// What `world`'s reader hands `gateway` of `model`'s single pool at `now`.
fn read_of(
    world: &StoreSnapshot,
    model: &str,
    gateway: GatewayId,
    now: Instant,
) -> Vec<StandingVeto> {
    world
        .gateway_vetoes(model, Role::Single, gateway)
        .standing_at(now)
        .collect()
}

/// The reader's answer in the store's standing publication.
fn read(store: &MemoryStore, model: &str, gateway: GatewayId, now: Instant) -> Vec<StandingVeto> {
    store.with_snapshot(|world| read_of(world, model, gateway, now))
}

/// A veto with no hold, written through the door a controller writes by.
fn doubted(store: &MemoryStore, gateway: GatewayId, id: WorkerId, until: Instant, now: Instant) {
    let written = store.gateway_veto_noted(gateway, id, Some(until), None, false, now);
    assert_eq!(written, VetoWrite::Written);
}

/// A veto written through today's door is read under its worker's id by
/// the gateway that wrote it, and by no other. The ids ascend whatever the
/// order of the writes.
#[test]
fn a_veto_written_through_the_door_is_read_by_its_workers_id_for_its_gateway_alone() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let mut ids =
        ["http://a:8000", "http://b:8000", "http://c:8000"].map(|key| member(&store, key));
    ids.sort_unstable();
    // No veto stands in the store: nothing to read.
    assert_eq!(read(&store, MODEL, ASKER, t0), []);

    let until = t0 + 10 * S;
    doubted(&store, ASKER, ids[2], until, t0);
    doubted(&store, ASKER, ids[0], until, t0);
    let own = [doubt(ids[0]), doubt(ids[2])];
    assert_eq!(read(&store, MODEL, ASKER, t0), own);
    assert_eq!(read(&store, MODEL, OTHER, t0), []);

    doubted(&store, OTHER, ids[1], until, t0);
    assert_eq!(read(&store, MODEL, OTHER, t0), [doubt(ids[1])]);
    assert_eq!(read(&store, MODEL, ASKER, t0), own);
}

/// A veto is read until its end and as held until its hold's end: at its
/// end's instant it stands no more, though the publication still holds its
/// entry. A hold noted past the veto's end is cut to it, and a veto told
/// with no kind holds whole.
#[test]
fn a_veto_is_read_until_its_end_and_held_until_its_holds_end() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    let at = |now: Instant| read(&store, MODEL, ASKER, now);
    let (hold_end, until) = (t0 + 2 * S, t0 + 5 * S);
    let written = store.gateway_veto_noted(ASKER, id, Some(until), Some(hold_end), false, t0);
    assert_eq!(written, VetoWrite::Written);
    assert_eq!(at(t0), [hold(id)]);
    assert_eq!(at(hold_end - MS), [hold(id)]);
    // From the hold's end to the veto's own the worker is doubted alone.
    assert_eq!(at(hold_end), [doubt(id)]);
    assert_eq!(at(until - MS), [doubt(id)]);
    assert_eq!(at(until), []);

    // A hold past the veto's end holds to that end and no longer.
    let later = until + 5 * S;
    let written = store.gateway_veto_noted(ASKER, id, Some(later), Some(later + 60 * S), false, t0);
    assert_eq!(written, VetoWrite::Written);
    assert_eq!(at(later - MS), [hold(id)]);
    assert_eq!(at(later), []);

    // The door that tells no kind: the whole veto holds.
    store.gateway_veto(ASKER, id, Some(until));
    assert_eq!(at(until - MS), [hold(id)]);
    assert_eq!(at(until), []);

    let written = store.gateway_veto_noted(ASKER, id, None, None, false, t0);
    assert_eq!(written, VetoWrite::Written);
    assert_eq!(at(t0), []);
}

/// A worker of two pools is read in each pool's ask. A model the store
/// does not know and a role with no pool read nothing.
#[test]
fn a_worker_of_two_pools_is_read_in_each_pools_ask() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + S;
    let both = admit(
        &store,
        facts("http://a:8000", Role::Single, &["one", "two"]),
    );
    let one = admit(&store, facts("http://b:8000", Role::Single, &["one"]));
    store.gateway_veto(ASKER, both, Some(until));
    store.gateway_veto(ASKER, one, Some(until));
    let read_in = |model: &str| read(&store, model, ASKER, t0);
    assert!(read_in("one").contains(&hold(both)) && read_in("one").contains(&hold(one)));
    assert!(read_in("two").contains(&hold(both)));
    assert_eq!(read_in("three"), []);
    let prefills = store.with_snapshot(|world| {
        let vetoes = world.gateway_vetoes("one", Role::Prefill, ASKER);
        vetoes.standing_at(t0).count()
    });
    assert_eq!(prefills, 0);
}

/// A veto that left is read no more: a departed worker's leaves with its
/// row, a dropped gateway's with the gateway, and one past its end at the
/// next window's publication.
#[test]
fn a_veto_that_left_is_read_no_more() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let mut ids =
        ["http://a:8000", "http://b:8000", "http://c:8000"].map(|key| member(&store, key));
    ids.sort_unstable();
    let [leaving, ending, staying] = ids;
    let (soon, until) = (t0 + 10 * MS, t0 + 60 * S);
    doubted(&store, ASKER, leaving, until, t0);
    doubted(&store, ASKER, ending, soon, t0);
    doubted(&store, ASKER, staying, until, t0);
    doubted(&store, OTHER, staying, until, t0);
    let all = [doubt(leaving), doubt(ending), doubt(staying)];
    assert_eq!(read(&store, MODEL, ASKER, t0), all);

    let revision = WorkersStore::get(&store, leaving)
        .expect("the row")
        .revision;
    store.remove(leaving, revision).expect("removed");
    store.publish_now();
    assert_eq!(
        read(&store, MODEL, ASKER, t0),
        [doubt(ending), doubt(staying)]
    );

    // A window's publication at the first veto's end takes its entry out:
    // an ask dated before that end reads it no more.
    store.record(staying, 0, load(1, 0), soon, LoadSource::Stream);
    assert_eq!(read(&store, MODEL, ASKER, t0), [doubt(staying)]);

    store.drop_gateways(&[ASKER]);
    assert_eq!(read(&store, MODEL, ASKER, t0), []);
    assert_eq!(read(&store, MODEL, OTHER, t0), [doubt(staying)]);
}

/// A pool names a worker another gateway alone reaches, and a gateway can
/// veto it. The reader hands the veto: its caller passes over a worker its
/// own list does not hold.
#[test]
fn a_veto_on_a_worker_its_gateway_does_not_reach_is_read() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + S;
    let register = |key: &str, reachable_from: Reachability| {
        admit(
            &store,
            WorkerFacts {
                reachable_from,
                ..facts(key, Role::Single, &[MODEL])
            },
        )
    };
    let open = register("http://a:8000", Reachability::Any);
    let theirs = register("http://b:8000", Reachability::Gateway(OTHER));
    store.gateway_veto(ASKER, open, Some(until));
    store.gateway_veto(ASKER, theirs, Some(until));
    let mut vetoed = [hold(open), hold(theirs)];
    vetoed.sort_unstable_by_key(|veto| veto.id);
    assert_eq!(read(&store, MODEL, ASKER, t0), vetoed);
}
