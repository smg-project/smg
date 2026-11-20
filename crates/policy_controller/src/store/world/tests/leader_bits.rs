//! The leader's own doubt of a worker ([`HealthSink::leader_bits`]): the
//! row's stalled bit for every gateway, with no hold, beside the gateways'
//! own vetoes. What it stands through, and what it leaves with.

use std::time::{Duration, Instant};

use super::{admit, ask, doubted, facts, held, member, stalled, status, view, Ask, ASKER, MODEL};
use crate::{
    store::{
        GatewayId, HealthSink, HealthWrite, LeaderBits, MemoryStore, VetoWrite, WorkerFacts,
        WorkerPatch, WorkerStatus, WorkersStore,
    },
    view::{Role, WorkerId},
};

/// A gateway that asks beside [`ASKER`].
const OTHER: GatewayId = GatewayId(2);
const DOUBTED: LeaderBits = LeaderBits { stalled: true };
const CLEAR: LeaderBits = LeaderBits { stalled: false };

fn revision(store: &MemoryStore, id: WorkerId) -> u64 {
    WorkersStore::get(store, id).expect("the row").revision
}

/// The leader's word on `id`, read under the revision its row stands at.
fn says(store: &MemoryStore, id: WorkerId, bits: LeaderBits) -> HealthWrite {
    store.leader_bits(id, revision(store, id), bits)
}

/// The workers of the pool a decision of `gateway` may be handed at `now`.
fn routable(store: &MemoryStore, gateway: GatewayId, now: Instant) -> Vec<WorkerId> {
    let asked = Ask {
        gateway,
        ..ask(now)
    };
    view(store, &asked)
        .0
        .iter()
        .filter(|row| row.health.routable())
        .map(|row| row.id)
        .collect()
}

#[test]
fn the_leaders_doubt_masks_the_worker_for_every_gateway_with_no_hold() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let doubtful = member(&store, "http://a:8000");
    let sound = member(&store, "http://b:8000");
    for id in [doubtful, sound] {
        status(&store, id, WorkerStatus::Ready);
    }
    // The write moves no status and says the one that stands.
    assert_eq!(
        says(&store, doubtful, DOUBTED),
        HealthWrite::Stood(WorkerStatus::Ready)
    );
    // Every gateway reads the doubt at once, and as a doubt alone: no hold
    // stands on the row, and no gateway vetoes the worker.
    for gateway in [ASKER, OTHER] {
        assert_eq!(doubted(&store, gateway, now), [doubtful]);
        assert_eq!(held(&store, gateway, now), []);
        assert_eq!(routable(&store, gateway, now), [sound]);
        assert!(!store.vetoed(gateway, doubtful, now));
    }
    assert_eq!(
        says(&store, doubtful, CLEAR),
        HealthWrite::Stood(WorkerStatus::Ready)
    );
    for gateway in [ASKER, OTHER] {
        assert_eq!(stalled(&store, gateway, now), []);
    }
}

#[test]
fn a_doubt_that_moves_publishes_once_and_a_word_that_stands_publishes_nothing() {
    let store = MemoryStore::new();
    let id = member(&store, "http://a:8000");
    let published = store.publications();
    assert_eq!(
        says(&store, id, CLEAR),
        HealthWrite::Stood(WorkerStatus::Pending)
    );
    assert_eq!(store.publications(), published, "no doubt stood");
    for (bits, publications) in [(DOUBTED, 1), (DOUBTED, 1), (CLEAR, 2), (CLEAR, 2)] {
        says(&store, id, bits);
        assert_eq!(store.publications(), published + publications);
    }
    // A word on a worker with no row is judged against nothing.
    let nobody = WorkerId(u64::MAX);
    assert_eq!(store.leader_bits(nobody, 1, DOUBTED), HealthWrite::NoRow);
    assert_eq!(store.publications(), published + 2);
}

#[test]
fn a_doubt_stands_through_a_build_a_status_and_the_rows_next_revision() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let id = member(&store, "http://a:8000");
    says(&store, id, DOUBTED);
    // Another member registers: every row is built again from the members.
    let builds = store.membership_builds();
    member(&store, "http://b:8000");
    assert_eq!(store.membership_builds(), builds + 1);
    assert_eq!(stalled(&store, ASKER, now), [id]);
    // A status moves two other bits of the row.
    status(&store, id, WorkerStatus::Ready);
    assert_eq!(stalled(&store, ASKER, now), [id]);
    // The row's facts are written again and its revision moves. The worker
    // is the one it was, so the doubt of it stands, through that build too.
    let read = revision(&store, id);
    let patch = WorkerPatch {
        priority: Some(1),
        ..WorkerPatch::default()
    };
    store.patch(id, patch, read).expect("patched");
    store.publish_now();
    assert_ne!(revision(&store, id), read);
    assert_eq!(stalled(&store, OTHER, now), [id]);
    // A word read under the revision the row left is dropped.
    assert_eq!(store.leader_bits(id, read, CLEAR), HealthWrite::Stale);
    assert_eq!(stalled(&store, ASKER, now), [id]);
    assert_eq!(
        says(&store, id, CLEAR),
        HealthWrite::Stood(WorkerStatus::Ready)
    );
    assert_eq!(stalled(&store, ASKER, now), []);
}

#[test]
fn a_doubt_leaves_with_its_row_and_the_keys_next_life_starts_with_none() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let key = "http://a:8000";
    let next_life = || WorkerFacts {
        incarnation: 2,
        ..facts(key, Role::Single, &[MODEL])
    };
    let first = member(&store, key);
    says(&store, first, DOUBTED);
    // The key's next incarnation is another worker: the leader said no
    // word of it, and a word on the one that left finds no row.
    let next = admit(&store, next_life());
    assert_ne!(next, first);
    assert_eq!(stalled(&store, ASKER, now), []);
    assert_eq!(store.leader_bits(first, 1, DOUBTED), HealthWrite::NoRow);
    // A row removed and registered again under its id is another life too.
    says(&store, next, DOUBTED);
    assert_eq!(stalled(&store, ASKER, now), [next]);
    store.remove(next, revision(&store, next)).expect("removed");
    store.publish_now();
    let again = admit(&store, next_life());
    assert_eq!(again, next, "the same key and incarnation is the same id");
    assert_eq!(stalled(&store, ASKER, now), []);
}

#[test]
fn a_gateways_veto_and_the_leaders_doubt_stand_beside_each_other() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + Duration::from_secs(1);
    let id = member(&store, "http://a:8000");
    member(&store, "http://b:8000");
    // The asking gateway holds the worker to `until`, as behind its open
    // breaker, and the leader doubts it.
    let hold = store.gateway_veto_noted(ASKER, id, Some(until), Some(until), false, t0);
    assert_eq!(hold, VetoWrite::Written);
    says(&store, id, DOUBTED);
    assert_eq!(held(&store, ASKER, t0), [id]);
    assert_eq!(doubted(&store, OTHER, t0), [id]);
    // The gateway's veto ends at its instant. The leader's doubt has no
    // end: from then the gateway reads it as every other does.
    assert_eq!(doubted(&store, ASKER, until), [id]);
    // The leader's doubt clears: the gateway's hold stands, for it alone.
    says(&store, id, CLEAR);
    assert_eq!(held(&store, ASKER, t0), [id]);
    assert_eq!(stalled(&store, OTHER, t0), []);
    assert_eq!(stalled(&store, ASKER, until), []);
}
