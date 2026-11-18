//! The cases of the encode classes: what the world holds of a model's
//! encoders and prefills by the ends of the encode handoff they state.

use super::*;
use crate::store::{EncodeHandoff, EncodeOffer};

/// A worker of the model, of one known runtime, that states `offer` as its
/// end of the encode handoff.
pub(super) fn encode_facts(key: &str, role: Role, offer: EncodeOffer) -> WorkerFacts {
    WorkerFacts {
        engine_kind: "rt-a".to_string(),
        encode: offer,
        ..facts(key, role, &[MODEL])
    }
}

/// The end of a worker that pushes, or takes a push, on `connector`.
pub(super) fn push_on(connector: &str) -> EncodeOffer {
    EncodeOffer {
        handoff: EncodeHandoff::Push,
        producer: true,
        consumer: true,
        connector: Some(connector.to_string()),
        ..EncodeOffer::default()
    }
}

/// The class at worker `id`'s row: `classed` holds a class a row of `rows`.
fn class_at(rows: &[Candidate], classed: &[u32], id: WorkerId) -> u32 {
    let row = rows.iter().position(|row| row.id == id);
    classed[row.expect("the worker's row")]
}

#[test]
fn the_world_holds_a_models_encode_classes_by_the_rows_of_its_pools() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let upsert = |key: &str, role: Role, connector: &str| {
        store
            .upsert(encode_facts(key, role, push_on(connector)))
            .id()
    };
    let e1 = upsert("grpc://e1:9000", Role::Encode, "c1");
    let e2 = upsert("grpc://e2:9000", Role::Encode, "c2");
    let e3 = upsert("grpc://e3:9000", Role::Encode, "c2");
    let p1 = upsert("grpc://p1:9000", Role::Prefill, "c1");
    let p2 = upsert("grpc://p2:9000", Role::Prefill, "c2");

    // Each side stands in the classes as the rows of its pool.
    let (encoders, _, _) = side(&store, Role::Encode, &ask(now));
    let (prefills, _, _) = side(&store, Role::Prefill, &ask(now));
    let world = store.world();
    let classes = world.encodes(MODEL).expect("an end and a prefill");
    assert_eq!(classes.encoders().len(), encoders.len());
    assert_eq!(classes.prefills().len(), prefills.len());
    let first = class_at(&encoders, classes.encoders(), e1);
    let second = class_at(&encoders, classes.encoders(), e2);
    assert!(first != 0 && second != 0 && first != second);
    assert_eq!(class_at(&encoders, classes.encoders(), e3), second);
    // A prefill names the class of the encoders on its connector.
    assert_eq!(class_at(&prefills, classes.prefills(), p1), first);
    assert_eq!(class_at(&prefills, classes.prefills(), p2), second);
    // A list names its encoders by their rows in the encode pool.
    let listed = |class: u32| -> Vec<WorkerId> {
        let rows = classes.list(class).iter();
        sorted(rows.map(|&row| encoders[row as usize].id).collect())
    };
    assert_eq!(listed(first), [e1]);
    assert_eq!(listed(second), sorted(vec![e2, e3]));
    assert_eq!(classes.form(first), EncodeHandoff::Push);
}

#[test]
fn a_blank_connector_of_an_encode_offer_is_dropped_at_the_door() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let upsert = |key: &str, role: Role, connector: &str| {
        store
            .upsert(encode_facts(key, role, push_on(connector)))
            .id()
    };
    // One name written three ways, and a connector of blanks alone.
    let e1 = upsert("grpc://e1:9000", Role::Encode, " c1 ");
    let e2 = upsert("grpc://e2:9000", Role::Encode, "c1");
    let e3 = upsert("grpc://e3:9000", Role::Encode, "  ");
    let p1 = upsert("grpc://p1:9000", Role::Prefill, "c1 ");

    // The names between blanks are one end; the blank one states none.
    let (encoders, _, _) = side(&store, Role::Encode, &ask(now));
    let (prefills, _, _) = side(&store, Role::Prefill, &ask(now));
    let world = store.world();
    let classes = world.encodes(MODEL).expect("an end and a prefill");
    let class = class_at(&encoders, classes.encoders(), e1);
    assert_ne!(class, 0);
    assert_eq!(class_at(&encoders, classes.encoders(), e2), class);
    assert_eq!(class_at(&encoders, classes.encoders(), e3), 0);
    assert_eq!(class_at(&prefills, classes.prefills(), p1), class);
    // The row of record holds the connector as the rule reads it.
    let connector = |id: WorkerId| {
        let row = WorkersStore::get(&store, id).expect("the row");
        row.facts.encode.connector
    };
    assert_eq!(connector(e1).as_deref(), Some("c1"));
    assert_eq!(connector(e3), None);
}

#[test]
fn a_model_with_no_encoder_or_no_prefill_has_no_encode_classes() {
    let store = MemoryStore::new();
    store.upsert(encode_facts("grpc://e1:9000", Role::Encode, push_on("c1")));
    assert!(store.world().encodes(MODEL).is_none());
    store.upsert(encode_facts("grpc://p1:9000", Role::Prefill, push_on("c1")));
    let world = store.world();
    assert_eq!(
        world.encodes(MODEL).map(|classes| classes.classes()),
        Some(1)
    );
    assert!(world.encodes("other").is_none());

    // A model of prefills alone has none either.
    let alone = MemoryStore::new();
    alone.upsert(encode_facts("grpc://p1:9000", Role::Prefill, push_on("c1")));
    assert!(alone.world().encodes(MODEL).is_none());
}

#[test]
fn a_model_whose_encoders_state_no_end_has_no_encode_classes() {
    let store = MemoryStore::new();
    let silent = EncodeOffer::default();
    store.upsert(encode_facts("grpc://e1:9000", Role::Encode, silent));
    store.upsert(encode_facts("grpc://p1:9000", Role::Prefill, push_on("c1")));
    assert!(store.world().encodes(MODEL).is_none());
    // The first encoder to state an end makes the first class.
    store.upsert(encode_facts("grpc://e2:9000", Role::Encode, push_on("c1")));
    let world = store.world();
    let classes = world.encodes(MODEL).expect("an end and a prefill");
    assert_eq!((classes.classes(), classes.list(1).len()), (1, 1));
}

#[test]
fn two_stores_of_one_membership_number_the_encode_classes_alike() {
    let fleet = [
        ("grpc://e1:9000", Role::Encode, "c2"),
        ("grpc://e2:9000", Role::Encode, "c1"),
        ("grpc://e3:9000", Role::Encode, "c2"),
        ("grpc://p1:9000", Role::Prefill, "c1"),
        ("grpc://p2:9000", Role::Prefill, "c2"),
    ];
    let (forward, backward) = (MemoryStore::new(), MemoryStore::new());
    for &(key, role, connector) in &fleet {
        forward.upsert(encode_facts(key, role, push_on(connector)));
    }
    for &(key, role, connector) in fleet.iter().rev() {
        backward.upsert(encode_facts(key, role, push_on(connector)));
    }
    let (ours, theirs) = (forward.world(), backward.world());
    let classes = ours.encodes(MODEL).expect("an end and a prefill");
    assert_eq!(Some(classes), theirs.encodes(MODEL));
    // The classes are numbered as the encoders stand in their pool's rows:
    // the first row's class is the first.
    assert_eq!(classes.classes(), 2);
    assert_eq!(classes.encoders()[0], 1);
}

/// The world reads a member's runtime from its worker's engine kind: two
/// engine kinds that state the rendezvous form are two ends.
#[test]
fn two_engine_kinds_of_one_form_are_two_encode_classes_in_the_world() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let of_kind = |key: &str, role: Role, kind: &str| {
        let rendezvous = EncodeOffer {
            handoff: EncodeHandoff::Rendezvous,
            ..EncodeOffer::default()
        };
        let registered = WorkerFacts {
            engine_kind: kind.to_string(),
            ..encode_facts(key, role, rendezvous)
        };
        store.upsert(registered).id()
    };
    let e1 = of_kind("grpc://e1:9000", Role::Encode, "rt-a");
    let e2 = of_kind("grpc://e2:9000", Role::Encode, "rt-b");
    let p1 = of_kind("grpc://p1:9000", Role::Prefill, "rt-a");
    let p2 = of_kind("grpc://p2:9000", Role::Prefill, "rt-b");
    let (encoders, _, _) = side(&store, Role::Encode, &ask(now));
    let (prefills, _, _) = side(&store, Role::Prefill, &ask(now));
    let world = store.world();
    let classes = world.encodes(MODEL).expect("two ends and two prefills");
    let first = class_at(&encoders, classes.encoders(), e1);
    let second = class_at(&encoders, classes.encoders(), e2);
    assert_eq!(classes.classes(), 2);
    assert!(first != 0 && second != 0 && first != second);
    assert_eq!(class_at(&prefills, classes.prefills(), p1), first);
    assert_eq!(class_at(&prefills, classes.prefills(), p2), second);
}

#[test]
fn a_membership_write_that_moves_no_offer_carries_the_encode_classes() {
    let store = MemoryStore::new();
    let encoder = store
        .upsert(encode_facts("grpc://e1:9000", Role::Encode, push_on("c1")))
        .id();
    store.upsert(encode_facts("grpc://p1:9000", Role::Prefill, push_on("c1")));
    // The world is kept, so its classes stand where they are: one address
    // in a later world is those classes carried, never others in their
    // place.
    let before = store.world();
    let derived = before.encodes(MODEL).expect("the model's classes");
    let carried = |what: &str| {
        let world = store.world();
        let encodes = world.encodes(MODEL).expect("the model's classes");
        assert!(std::ptr::eq(derived, encodes), "{what}: derived again");
        world.epoch()
    };
    // The same registration again: a membership write all the same.
    store.upsert(encode_facts("grpc://e1:9000", Role::Encode, push_on("c1")));
    assert!(carried("the same facts") > before.epoch());
    // Facts no class reads.
    store.upsert(WorkerFacts {
        zone: Some("z".to_string()),
        priority: 2,
        ..encode_facts("grpc://p1:9000", Role::Prefill, push_on("c1"))
    });
    carried("facts no class reads");
    // A patch moves labels, a priority and an overload alone.
    let revision = WorkersStore::get(&store, encoder)
        .expect("the row")
        .revision;
    let patch = WorkerPatch {
        labels: None,
        priority: Some(3),
        overload: None,
    };
    store.patch(encoder, patch, revision).expect("patched");
    carried("a patch");
}

#[test]
fn a_membership_write_that_moves_an_end_derives_the_encode_classes_again() {
    // Each fact the encode classes read of a worker, moved on one encoder by
    // its registration refreshed.
    let first = || encode_facts("grpc://e1:9000", Role::Encode, push_on("c1"));
    let rendezvous = EncodeOffer {
        handoff: EncodeHandoff::Rendezvous,
        ..EncodeOffer::default()
    };
    let moved = [
        (
            "the connector",
            encode_facts("grpc://e1:9000", Role::Encode, push_on("c2")),
        ),
        (
            "the handoff",
            encode_facts("grpc://e1:9000", Role::Encode, rendezvous),
        ),
        (
            "the runtime",
            WorkerFacts {
                engine_kind: "rt-b".to_string(),
                ..first()
            },
        ),
    ];
    let rest = |store: &MemoryStore| {
        store.upsert(encode_facts("grpc://e2:9000", Role::Encode, push_on("c2")));
        store.upsert(encode_facts("grpc://p1:9000", Role::Prefill, push_on("c1")));
        store.upsert(encode_facts("grpc://p2:9000", Role::Prefill, push_on("c2")));
    };
    for (what, facts) in moved {
        let store = MemoryStore::new();
        store.upsert(first());
        rest(&store);
        let before = store.world();
        store.upsert(facts.clone());
        // A store that never held the first version: every write of it
        // registers a worker, and derives.
        let fresh = MemoryStore::new();
        fresh.upsert(facts);
        rest(&fresh);
        let (after, fresh) = (store.world(), fresh.world());
        assert_eq!(after.encodes(MODEL), fresh.encodes(MODEL), "{what}");
        assert_ne!(
            after.encodes(MODEL),
            before.encodes(MODEL),
            "{what}: the classes moved"
        );
    }
}
