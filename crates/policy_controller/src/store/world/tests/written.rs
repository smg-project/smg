//! The cases of a membership write of one worker: the pair classes it
//! leaves are the classes a store that never held the write's past
//! derives, whatever the worker left and joined, and the writer keeps a
//! derivation for the models that pair and for no other.

use std::time::Instant;

use super::{facts, generation_pairing, pair_facts, pair_rows, MODEL};
use crate::{
    spec::Pairing,
    store::{
        world::{Classes, Writer, PUBLISH_WINDOW},
        MemoryStore, WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus, WorkersStore,
    },
    view::{Role, WireModes, WorkerId},
};

/// A store fed the standing rows of `store`, each as a first registration.
fn fresh_of(store: &MemoryStore) -> MemoryStore {
    let fresh = MemoryStore::new();
    for row in store.snapshot().1 {
        fresh.upsert(row.facts);
    }
    fresh
}

#[test]
fn a_walk_of_membership_writes_leaves_the_classes_a_fresh_store_derives() {
    const MODELS: [&[&str]; 3] = [&[MODEL], &["other"], &[MODEL, "other"]];
    for seed in 0..16_u64 {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut draw = |of: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % of as u64) as usize
        };
        let store = MemoryStore::new();
        let mut incarnations = [1_u64; 8];
        for step in 0..150 {
            let worker = draw(8);
            let key = format!("grpc://w{worker}:9000");
            let row = store.by_key(&key);
            match (draw(7), row) {
                // The worker leaves.
                (0, Some(row)) => {
                    store.remove(row.id, row.revision).expect("removed");
                }
                // A patch: no offer moves.
                (1, Some(row)) => {
                    let patch = WorkerPatch {
                        labels: None,
                        priority: Some(3),
                        overload: None,
                    };
                    store.patch(row.id, patch, row.revision).expect("patched");
                }
                // A registration: a first one, the same again, or one that
                // moves the role, the models or the offer, one in six under
                // a new incarnation.
                _ => {
                    incarnations[worker] += u64::from(draw(6) == 0);
                    let role = [Role::Prefill, Role::Decode][draw(2)];
                    let transport = [None, Some("tx-a"), Some("tx-b")][draw(3)];
                    let models = MODELS[draw(3)].iter().map(|model| (*model).to_string());
                    store.upsert(WorkerFacts {
                        incarnation: incarnations[worker],
                        models: models.collect(),
                        engine_kind: ["", "rt-a", "rt-b"][draw(3)].to_string(),
                        wire_modes: [WireModes::GRPC, WireModes::HTTP][usize::from(draw(4) == 0)],
                        ..pair_facts(&key, role, transport)
                    });
                }
            }
            let fresh = fresh_of(&store);
            for model in [MODEL, "other"] {
                assert_eq!(
                    store.world().pairs(model),
                    fresh.world().pairs(model),
                    "seed {seed}, step {step}, model {model}"
                );
            }
            assert_eq!(
                pair_rows(&store),
                pair_rows(&fresh),
                "seed {seed}, step {step}"
            );
        }
    }
}

#[test]
fn a_write_that_moves_an_offer_and_no_class_carries_the_classes() {
    // Two prefills and two decodes that name no runtime: one class a side.
    // Each prefill names its runtime and unnames it in turn, the first by
    // id among them: its offer moves at every write, and it pairs as it
    // did.
    let bare = |key: &str, role: Role, runtime: &str| WorkerFacts {
        engine_kind: runtime.to_string(),
        ..pair_facts(key, role, None)
    };
    let store = MemoryStore::new();
    let prefills = ["grpc://pa:9000", "grpc://pb:9000"];
    for key in prefills {
        store.upsert(bare(key, Role::Prefill, ""));
    }
    for key in ["grpc://da:9000", "grpc://db:9000"] {
        store.upsert(bare(key, Role::Decode, ""));
    }
    // The world is kept, so its classes stand where they are: one address
    // in a later world is those classes carried.
    let before = store.world();
    let derived = before.pairs(MODEL).expect("the model's classes");
    let rows = pair_rows(&store);
    for key in prefills {
        for runtime in ["rt-b", ""] {
            store.upsert(bare(key, Role::Prefill, runtime));
            let world = store.world();
            let pairs = world.pairs(MODEL).expect("the model's classes");
            assert!(
                std::ptr::eq(derived, pairs),
                "{key} {runtime:?}: derived again"
            );
            assert_eq!(pair_rows(&store), rows, "{key} {runtime:?}");
        }
    }
}

#[test]
fn a_worker_that_leaves_one_of_its_models_moves_that_models_classes_alone() {
    let store = MemoryStore::new();
    let both = |models: &[&str]| WorkerFacts {
        models: models.iter().map(|model| (*model).to_string()).collect(),
        ..pair_facts("grpc://pa:9000", Role::Prefill, Some("tx-a"))
    };
    store.upsert(both(&[MODEL, "other"]));
    for model in [MODEL, "other"] {
        for (key, role) in [("p", Role::Prefill), ("d", Role::Decode)] {
            store.upsert(facts(&format!("grpc://{key}-{model}:9000"), role, &[model]));
        }
    }
    let before = store.world();
    let kept = before.pairs(MODEL).expect("the model's classes");
    // The prefill of both models leaves the second: the first model's
    // classes are carried, and the second's are those of a store that never
    // held it there.
    store.upsert(both(&[MODEL]));
    let (after, fresh) = (store.world(), fresh_of(&store).world());
    assert!(std::ptr::eq(
        kept,
        after.pairs(MODEL).expect("the model's classes")
    ));
    assert_eq!(after.pairs("other"), fresh.pairs("other"));
    assert_ne!(after.pairs("other"), before.pairs("other"));
}

#[test]
fn the_writer_keeps_a_derivation_a_paired_model_and_none_past_it() {
    let origin = Instant::now();
    let row = |id: u64, role: Role, model: &str| WorkerRow {
        id: WorkerId(id),
        facts: facts(&format!("grpc://w{id}:9000"), role, &[model]),
        status: WorkerStatus::Ready,
        revision: 1,
        admitted_at: origin,
    };
    let moved = |id: u64| Classes::Written {
        left: Some(WorkerId(id)),
        joined: Some(WorkerId(id)),
    };
    let mut writer = Writer::new(PUBLISH_WINDOW, origin);
    // A model of one side keeps none, and its first pair makes one.
    let mut members = vec![row(1, Role::Prefill, "m0")];
    writer.rebuild(1, members.iter(), |_, _| None, moved(1));
    assert_eq!(writer.derivations.len(), 0);
    members.push(row(2, Role::Decode, "m0"));
    writer.rebuild(2, members.iter(), |_, _| None, moved(2));
    assert_eq!(writer.derivations.len(), 1);
    // The pair moves through a hundred models, the decode first: the
    // model it left keeps no derivation, and the one both stand in has
    // the only one.
    for n in 1..100_u64 {
        let model = format!("m{n}");
        members[1] = row(2, Role::Decode, &model);
        writer.rebuild(2 * n + 1, members.iter(), |_, _| None, moved(2));
        assert_eq!(writer.derivations.len(), 0, "{model}");
        members[0] = row(1, Role::Prefill, &model);
        writer.rebuild(2 * n + 2, members.iter(), |_, _| None, moved(1));
        let kept: Vec<&str> = writer.derivations.keys().map(|model| &**model).collect();
        assert_eq!(kept, [model.as_str()]);
    }
}

#[test]
fn every_kind_of_membership_write_is_answered_for_its_worker_alone() {
    let offered = |key: &str, role: Role, transport: &str, incarnation: u64| WorkerFacts {
        incarnation,
        ..pair_facts(key, role, Some(transport))
    };
    let store = MemoryStore::new();
    // The model's first pair makes the one whole derivation.
    store.upsert(offered("grpc://pa:9000", Role::Prefill, "tx-a", 1));
    assert_eq!(store.whole_derivations(MODEL), None);
    store.upsert(offered("grpc://da:9000", Role::Decode, "tx-a", 1));
    assert_eq!(store.whole_derivations(MODEL), Some(1));
    // Two workers that join, an offer that moves, a new incarnation, a
    // registration again as it stands and a worker that leaves: each write
    // is answered for its worker, and nothing is derived whole again.
    store.upsert(offered("grpc://pb:9000", Role::Prefill, "tx-b", 1));
    store.upsert(offered("grpc://db:9000", Role::Decode, "tx-b", 1));
    store.upsert(offered("grpc://pb:9000", Role::Prefill, "tx-a", 1));
    store.upsert(offered("grpc://pb:9000", Role::Prefill, "tx-b", 2));
    store.upsert(offered("grpc://db:9000", Role::Decode, "tx-b", 1));
    let row = store.by_key("grpc://db:9000").expect("the decode's row");
    store.remove(row.id, row.revision).expect("removed");
    assert_eq!(store.whole_derivations(MODEL), Some(1));
    assert_eq!(
        store.world().pairs(MODEL),
        fresh_of(&store).world().pairs(MODEL)
    );
    // A generation of another pairing mode is the second.
    store
        .apply(generation_pairing(1, Pairing::Strict))
        .expect("applied");
    assert_eq!(store.whole_derivations(MODEL), Some(2));
}
