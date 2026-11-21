//! The leader's own doubt of a worker ([`HealthSink::leader_bits`]): the
//! row's stalled bit for every gateway, with no hold, beside the gateways'
//! own vetoes. What it stands through, and what it leaves with.

use std::time::{Duration, Instant};

use super::{
    admit, ask, doubted, facts, generation_pairing, held, member, stalled, status, view, Ask,
    ASKER, MODEL,
};
use crate::{
    document::spec::Pairing,
    engine::view::{Role, WorkerId},
    store::{
        GatewayId, HealthSink, HealthWrite, Kinds, LeaderBits, MemoryStore, VetoWrite, ViewBuffers,
        WorkerFacts, WorkerPatch, WorkerStatus, WorkersStore,
    },
};

/// A gateway that asks beside [`ASKER`].
const OTHER: GatewayId = GatewayId(2);
const DOUBTED: LeaderBits = LeaderBits { stalled: true };
const CLEAR: LeaderBits = LeaderBits { stalled: false };

fn revision(store: &MemoryStore, id: WorkerId) -> u64 {
    WorkersStore::get(store, id).expect("the row").revision
}

/// The word on `id` of a writer that leads, read under the revision its
/// row stands at.
fn says(store: &MemoryStore, id: WorkerId, bits: LeaderBits) -> HealthWrite {
    store.leader_bits(id, revision(store, id), bits, &|| true)
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
    assert_eq!(
        store.leader_bits(nobody, 1, DOUBTED, &|| true),
        HealthWrite::NoRow
    );
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
    assert_eq!(
        store.leader_bits(id, read, CLEAR, &|| true),
        HealthWrite::Stale
    );
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
    assert_eq!(
        store.leader_bits(first, 1, DOUBTED, &|| true),
        HealthWrite::NoRow
    );
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

/// A worker of two models has a row in each pool, and a word of the leader
/// reaches both in the one publication.
#[test]
fn a_doubt_reaches_the_workers_row_in_every_pool_it_stands_in() {
    let store = MemoryStore::new();
    let id = admit(
        &store,
        facts("http://a:8000", Role::Single, &[MODEL, "other"]),
    );
    let published = store.publications();
    let doubt_in = |model: &str| -> Vec<bool> {
        let mut out = ViewBuffers::default();
        store
            .build_view(model, Role::Single, &ask(Instant::now()), &mut out)
            .expect("the worker serves the model");
        out.candidates
            .iter()
            .map(|row| row.health.stalled)
            .collect()
    };
    says(&store, id, DOUBTED);
    assert_eq!(store.publications(), published + 1);
    assert_eq!(
        (doubt_in(MODEL), doubt_in("other")),
        (vec![true], vec![true])
    );
    says(&store, id, CLEAR);
    assert_eq!(store.publications(), published + 2);
    assert_eq!(
        (doubt_in(MODEL), doubt_in("other")),
        (vec![false], vec![false])
    );
}

/// A new generation builds the membership again, and the doubt is in the
/// rows it builds.
#[test]
fn a_doubt_stands_through_a_new_generation() {
    let store = MemoryStore::new();
    store
        .apply(generation_pairing(1, Pairing::Lenient))
        .expect("the first generation");
    let id = member(&store, "http://a:8000");
    says(&store, id, DOUBTED);
    let builds = store.membership_builds();
    store
        .apply(generation_pairing(2, Pairing::Off))
        .expect("the next generation");
    assert_eq!(store.membership_builds(), builds + 1);
    assert_eq!(stalled(&store, ASKER, Instant::now()), [id]);
}

/// The bits are the store's own host's. A watch keeps them for a worker
/// still present, as it keeps a gateway's vetoes. A cut carries none: a
/// store that restores from it starts with no doubt.
#[test]
fn a_watch_keeps_the_bits_of_a_worker_still_present_and_a_cut_carries_none() {
    let store = MemoryStore::new();
    let id = member(&store, "http://a:8000");
    says(&store, id, DOUBTED);
    let cut = store.cut(Kinds::All);
    let standby = MemoryStore::new();
    assert!(standby.restore_from(cut.clone()));
    assert!(store.watch_from(cut));
    let now = Instant::now();
    assert_eq!(stalled(&store, ASKER, now), [id]);
    assert_eq!(stalled(&standby, ASKER, now), []);
}

/// The writer has the last word, asked behind the store's locks. Where it
/// says no the word is stale: nothing is written and nothing published.
#[test]
fn a_word_whose_writer_leads_no_longer_at_the_locks_is_stale() {
    let store = MemoryStore::new();
    let id = member(&store, "http://a:8000");
    let published = store.publications();
    let read = revision(&store, id);
    assert_eq!(
        store.leader_bits(id, read, DOUBTED, &|| false),
        HealthWrite::Stale
    );
    assert_eq!(store.publications(), published);
    assert_eq!(stalled(&store, ASKER, Instant::now()), []);
    // A doubt that stands is not cleared by such a word either.
    says(&store, id, DOUBTED);
    assert_eq!(
        store.leader_bits(id, read, CLEAR, &|| false),
        HealthWrite::Stale
    );
    assert_eq!(stalled(&store, ASKER, Instant::now()), [id]);
}
