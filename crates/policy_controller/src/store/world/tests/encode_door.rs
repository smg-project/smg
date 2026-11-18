//! The cases of the encode door: the pair, the encode pool and its columns
//! built from the world for the asking gateway, and a decision through it.

use super::{
    encode_classes::{encode_facts, push_on},
    *,
};
use crate::{
    store::{world::EncodePairBuffers, EncodeHandoff, EncodeOffer},
    EncodeItems, EncodePlaced, EncoderFacts, Identity,
};

/// A worker of the model that pairs over one transport and states `offer`
/// as its end of the encode handoff.
fn door_facts(key: &str, role: Role, offer: EncodeOffer) -> WorkerFacts {
    WorkerFacts {
        encode: offer,
        ..pair_facts(key, role, Some("tx-a"))
    }
}

/// Where worker `id` stands in `rows`.
fn row_at(rows: &[Candidate], id: WorkerId) -> usize {
    let row = rows.iter().position(|row| row.id == id);
    row.expect("the worker's row")
}

#[test]
fn the_encode_door_builds_the_pair_the_encode_pool_and_its_columns() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let upsert = |facts: WorkerFacts| store.upsert(facts).id();
    let e1 = upsert(door_facts("grpc://e1:9000", Role::Encode, push_on("c1")));
    // On the second connector one encoder takes references and images, and
    // one pushes without its role, so it states no end.
    let e2 = upsert(WorkerFacts {
        vision: true,
        ..door_facts(
            "grpc://e2:9000",
            Role::Encode,
            EncodeOffer {
                references: true,
                ..push_on("c2")
            },
        )
    });
    let e3 = upsert(door_facts(
        "grpc://e3:9000",
        Role::Encode,
        EncodeOffer {
            producer: false,
            ..push_on("c2")
        },
    ));
    let p1 = upsert(door_facts("grpc://p1:9000", Role::Prefill, push_on("c1")));
    let p2 = upsert(door_facts("grpc://p2:9000", Role::Prefill, push_on("c2")));
    let d1 = upsert(WorkerFacts {
        vision: true,
        ..door_facts("grpc://d1:9000", Role::Decode, EncodeOffer::default())
    });
    let d2 = upsert(door_facts(
        "grpc://d2:9000",
        Role::Decode,
        EncodeOffer::default(),
    ));

    let mut out = EncodePairBuffers::default();
    let built = store.build_encode_pair(MODEL, &ask(now), &mut out);
    assert!(built.is_some(), "both sides of the pair hold a member");
    let (pools, encode) = out.pools();
    assert_eq!((pools.prefills.len(), pools.decodes.len()), (2, 2));
    assert_eq!(encode.encoders.len(), 3);
    // One entry a row in each column.
    assert_eq!(encode.facts.len(), encode.encoders.len());
    assert_eq!(encode.prefill_classes.len(), pools.prefills.len());
    let fact = |id: WorkerId| encode.facts[row_at(encode.encoders, id)];
    let named = |id: WorkerId| encode.prefill_classes[row_at(pools.prefills, id)];
    // A prefill names the class of the encoders on its connector.
    assert!(fact(e1).class != 0 && fact(e2).class != 0);
    assert_ne!(fact(e1).class, fact(e2).class);
    assert_eq!(named(p1), fact(e1).class);
    assert_eq!(named(p2), fact(e2).class);
    // An encoder without its role has no class and still states its push.
    assert_eq!((fact(e3).class, fact(e3).pushes), (0, true));
    assert!(fact(e1).pushes && fact(e2).pushes);
    // The stated facts: who takes references, who takes images.
    assert_eq!(
        [e1, e2, e3].map(|id| fact(id).references),
        [false, true, false]
    );
    let vision = |rows: &[Candidate], id: WorkerId| rows[row_at(rows, id)].vision;
    assert!(vision(encode.encoders, e2) && !vision(encode.encoders, e1));
    assert!(vision(pools.decodes, d1) && !vision(pools.decodes, d2));
    // The columns are the publication's own classes.
    let world = store.world();
    let classes = world.encodes(MODEL).expect("two ends and two prefills");
    assert_eq!(encode.prefill_classes, classes.prefills());
    let classed: Vec<u32> = encode.facts.iter().map(|fact| fact.class).collect();
    assert_eq!(classed, classes.encoders());
}

#[test]
fn the_encode_doors_columns_follow_the_rows_of_the_asking_gateway() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let other = GatewayId(2);
    let upsert = |key: &str, role: Role, connector: &str, local: Option<GatewayId>| {
        let registered = WorkerFacts {
            reachable_from: local.map_or(Reachability::Any, Reachability::Gateway),
            ..door_facts(key, role, push_on(connector))
        };
        store.upsert(registered).id()
    };
    // On the first connector the workers every gateway dials; on the
    // second each gateway's own host-local encoder and prefill.
    let shared = (
        upsert("grpc://e:9000", Role::Encode, "c1", None),
        upsert("grpc://p:9000", Role::Prefill, "c1", None),
    );
    let of_other = (
        upsert("ipc://oe", Role::Encode, "c2", Some(other)),
        upsert("ipc://op", Role::Prefill, "c2", Some(other)),
    );
    let of_asker = (
        upsert("ipc://ae", Role::Encode, "c2", Some(ASKER)),
        upsert("ipc://ap", Role::Prefill, "c2", Some(ASKER)),
    );
    store.upsert(door_facts(
        "grpc://d:9000",
        Role::Decode,
        EncodeOffer::default(),
    ));
    for (gateway, own) in [(ASKER, of_asker), (other, of_other)] {
        let asked = Ask {
            gateway,
            ..ask(now)
        };
        let mut out = EncodePairBuffers::default();
        store
            .build_encode_pair(MODEL, &asked, &mut out)
            .expect("both sides of the pair");
        let (pools, encode) = out.pools();
        // The gateway's pools are the shared rows and its own.
        let ids = |rows: &[Candidate]| sorted(rows.iter().map(|row| row.id).collect());
        assert_eq!(
            ids(encode.encoders),
            sorted(vec![shared.0, own.0]),
            "{gateway:?}"
        );
        assert_eq!(
            ids(pools.prefills),
            sorted(vec![shared.1, own.1]),
            "{gateway:?}"
        );
        // Each column holds one entry a row of the gateway's view, and a
        // row's entry is its own: the classes still meet across the pools.
        assert_eq!(encode.facts.len(), 2, "{gateway:?}");
        assert_eq!(encode.prefill_classes.len(), 2, "{gateway:?}");
        let fact = |id: WorkerId| encode.facts[row_at(encode.encoders, id)];
        let named = |id: WorkerId| encode.prefill_classes[row_at(pools.prefills, id)];
        assert!(fact(shared.0).class != 0 && fact(own.0).class != 0);
        assert_ne!(fact(shared.0).class, fact(own.0).class, "{gateway:?}");
        assert_eq!(named(shared.1), fact(shared.0).class, "{gateway:?}");
        assert_eq!(named(own.1), fact(own.0).class, "{gateway:?}");
    }
}

#[test]
fn the_encode_door_builds_no_class_and_no_encoder_where_the_model_has_none() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let mut out = EncodePairBuffers::default();
    // One side of a pair alone builds nothing.
    store.upsert(door_facts("grpc://p1:9000", Role::Prefill, push_on("c1")));
    store.upsert(door_facts(
        "grpc://e1:9000",
        Role::Encode,
        EncodeOffer::default(),
    ));
    assert_eq!(store.build_encode_pair(MODEL, &ask(now), &mut out), None);
    // A pair whose one encoder states no end: every row names class 0.
    store.upsert(door_facts(
        "grpc://d1:9000",
        Role::Decode,
        EncodeOffer::default(),
    ));
    assert!(store
        .build_encode_pair(MODEL, &ask(now), &mut out)
        .is_some());
    let (pools, encode) = out.pools();
    assert_eq!((pools.prefills.len(), pools.decodes.len()), (1, 1));
    assert_eq!(encode.encoders.len(), 1);
    assert_eq!(encode.facts, [EncoderFacts::default()]);
    assert_eq!(encode.prefill_classes, [0]);
    // A model of a pair alone builds an empty encode pool.
    let alone = MemoryStore::new();
    alone.upsert(door_facts("grpc://p1:9000", Role::Prefill, push_on("c1")));
    alone.upsert(door_facts(
        "grpc://d1:9000",
        Role::Decode,
        EncodeOffer::default(),
    ));
    assert!(alone
        .build_encode_pair(MODEL, &ask(now), &mut out)
        .is_some());
    let (pools, encode) = out.pools();
    assert_eq!(pools.prefills.len(), 1);
    assert!(encode.encoders.is_empty() && encode.facts.is_empty());
    assert_eq!(encode.prefill_classes, [0]);
    // A model the store does not hold clears every buffer.
    assert_eq!(alone.build_encode_pair("other", &ask(now), &mut out), None);
    let (pools, encode) = out.pools();
    assert!(pools.prefills.is_empty() && pools.decodes.is_empty());
    assert!(encode.encoders.is_empty() && encode.facts.is_empty());
    assert!(encode.prefill_classes.is_empty());
}

/// An encoder that meets its prefill at a rendezvous is classed with it and
/// states no push: the door reads a push from the pushing form alone.
#[test]
fn the_encode_door_states_no_push_for_an_encoder_of_a_rendezvous() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let rendezvous = || EncodeOffer {
        handoff: EncodeHandoff::Rendezvous,
        ..EncodeOffer::default()
    };
    let upsert = |facts: WorkerFacts| store.upsert(facts).id();
    let e1 = upsert(door_facts("grpc://e1:9000", Role::Encode, rendezvous()));
    let e2 = upsert(door_facts("grpc://e2:9000", Role::Encode, push_on("c1")));
    let p1 = upsert(door_facts("grpc://p1:9000", Role::Prefill, rendezvous()));
    store.upsert(door_facts(
        "grpc://d1:9000",
        Role::Decode,
        EncodeOffer::default(),
    ));

    let mut out = EncodePairBuffers::default();
    let built = store.build_encode_pair(MODEL, &ask(now), &mut out);
    assert!(built.is_some(), "both sides of the pair hold a member");
    let (pools, encode) = out.pools();
    let fact = |id: WorkerId| encode.facts[row_at(encode.encoders, id)];
    // The rendezvous is an end: the encoder is of its prefill's class.
    assert_ne!(fact(e1).class, 0);
    assert_eq!(
        encode.prefill_classes[row_at(pools.prefills, p1)],
        fact(e1).class
    );
    // It states no push; the encoder beside it that pushes states one.
    assert!(!fact(e1).pushes);
    assert!(fact(e2).pushes);
}

/// An `epd` plan whose stages read nothing of a row but its status, its
/// class and whether the host excluded it: what the store feeds a row today.
const ENCODE_PLAN: &str = r#"{
  "apiVersion": "smg.policy/v1",
  "profiles": { "p": { "filter": ["healthy"], "pick": "best" } },
  "bindings": [
    { "plan": "epd", "profiles": { "encode": "p", "prefill": "p", "decode": "p" } }
  ]
}"#;

/// The door's one sentence: a request with an encode item is decided
/// through the store's door and the engine's entry. The pair is picked
/// among the prefills whose class holds a ready encoder, and the item is
/// placed on an encoder of the chosen prefill's class.
#[test]
fn a_request_with_an_item_is_decided_through_the_encode_door_and_the_engines_entry() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let ready = |key: &str, role: Role, offer: EncodeOffer| {
        let id = store.upsert(door_facts(key, role, offer)).id();
        status(&store, id, WorkerStatus::Ready);
        id
    };
    let e1 = ready("grpc://e1:9000", Role::Encode, push_on("c1"));
    // The second connector's one encoder is registered and not ready.
    store.upsert(door_facts("grpc://e2:9000", Role::Encode, push_on("c2")));
    let p1 = ready("grpc://p1:9000", Role::Prefill, push_on("c1"));
    ready("grpc://p2:9000", Role::Prefill, push_on("c2"));
    ready("grpc://d1:9000", Role::Decode, EncodeOffer::default());
    let spec: PolicySpec = serde_json::from_str(ENCODE_PLAN).expect("a well-formed document");
    let engine = spec.validate().expect("the engine");
    let (mut out, mut scratch) = (EncodePairBuffers::default(), Scratch::new());
    let mut placed = EncodePlaced::new();
    let keys = [Identity::hashed(7)];
    for seed in 0..8 {
        let (built, _) = store
            .build_encode_pair(MODEL, &ask(now), &mut out)
            .expect("both sides of the pair");
        let (pools, encode) = out.pools();
        let request = RequestFacts {
            model: MODEL,
            seed,
            now_micros: built.now_micros,
            ..RequestFacts::default()
        };
        let items = EncodeItems {
            keys: &keys,
            one: None,
            references: false,
            decode_images: false,
        };
        let cx = Constraints::default();
        let decision = engine.decide_encode_pair(
            &request,
            &built.pool_facts(),
            pools,
            encode,
            items,
            &cx,
            &cx,
            &cx,
            &mut scratch,
            &mut placed,
        );
        let Some(pick) = decision.pick() else {
            panic!("seed {seed}: the request routes: {decision:?}");
        };
        let target = pick.targets.first().expect("a target");
        assert_eq!(
            target.id, p1,
            "seed {seed}: the prefill a ready encoder serves"
        );
        let encoders: Vec<WorkerId> = placed.encoders().iter().map(|target| target.id).collect();
        assert_eq!(encoders, [e1], "seed {seed}");
    }
}

/// A gateway's view keeps the classes the whole pool numbered. Sixty-four
/// other gateways' host-local encoders and prefills stand first, each pair
/// on its host's own connector, so the asker's one encoder is of the
/// sixty-fifth class, above the count of the rows it is handed, and serves
/// the asker's prefill.
#[test]
fn a_view_with_rows_left_out_is_served_under_the_class_the_pool_numbered() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let ready = |key: &str, role: Role, offer: EncodeOffer, local: Option<GatewayId>| {
        let registered = WorkerFacts {
            reachable_from: local.map_or(Reachability::Any, Reachability::Gateway),
            ..door_facts(key, role, offer)
        };
        let id = store.upsert(registered).id();
        status(&store, id, WorkerStatus::Ready);
        id
    };
    for host in 0..64 {
        let (local, connector) = (Some(GatewayId(100 + host)), format!("c{host}"));
        ready(
            &format!("ipc://e{host}"),
            Role::Encode,
            push_on(&connector),
            local,
        );
        ready(
            &format!("ipc://p{host}"),
            Role::Prefill,
            push_on(&connector),
            local,
        );
    }
    let encoder = ready("ipc://ae", Role::Encode, push_on("asker"), Some(ASKER));
    let prefill = ready("ipc://ap", Role::Prefill, push_on("asker"), Some(ASKER));
    ready("grpc://d1:9000", Role::Decode, EncodeOffer::default(), None);
    let spec: PolicySpec = serde_json::from_str(ENCODE_PLAN).expect("a well-formed document");
    let engine = spec.validate().expect("the engine");
    let (mut out, mut scratch) = (EncodePairBuffers::default(), Scratch::new());
    let mut placed = EncodePlaced::new();
    let keys = [Identity::hashed(7)];
    let (built, _) = store
        .build_encode_pair(MODEL, &ask(now), &mut out)
        .expect("both sides of the pair");
    let (pools, encode) = out.pools();
    assert_eq!(encode.facts.len(), 1, "the asker's own encoder alone");
    assert_eq!(encode.facts[0].class, 65);
    assert_eq!(encode.prefill_classes, [65]);
    let request = RequestFacts {
        model: MODEL,
        seed: 1,
        now_micros: built.now_micros,
        ..RequestFacts::default()
    };
    let items = EncodeItems {
        keys: &keys,
        one: None,
        references: false,
        decode_images: false,
    };
    let cx = Constraints::default();
    let decision = engine.decide_encode_pair(
        &request,
        &built.pool_facts(),
        pools,
        encode,
        items,
        &cx,
        &cx,
        &cx,
        &mut scratch,
        &mut placed,
    );
    let Some(pick) = decision.pick() else {
        panic!("the request routes: {decision:?}");
    };
    let target = pick.targets.first().expect("a target");
    assert_eq!(target.id, prefill, "the asker's own prefill");
    let encoders: Vec<WorkerId> = placed.encoders().iter().map(|target| target.id).collect();
    assert_eq!(encoders, [encoder], "the asker's own encoder");
}

#[test]
fn a_membership_write_that_moves_no_offer_carries_the_encoders_facts() {
    let store = MemoryStore::new();
    let encoder = store
        .upsert(encode_facts("grpc://e1:9000", Role::Encode, push_on("c1")))
        .id();
    store.upsert(encode_facts("grpc://p1:9000", Role::Prefill, push_on("c1")));
    // What the encode pool's shape holds: one address in a later world is
    // those facts carried, never others in their place.
    let stated = || {
        let world = store.world();
        let shape = world.derived.shape(MODEL, Role::Encode);
        shape
            .and_then(|shape| shape.encoders.clone())
            .expect("the encoders' facts")
    };
    let before = stated();
    assert_eq!(before.len(), 1);
    // The same registration again: a membership write all the same.
    store.upsert(encode_facts("grpc://e1:9000", Role::Encode, push_on("c1")));
    assert!(
        Arc::ptr_eq(&before, &stated()),
        "the same facts: stated again"
    );
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
    assert!(Arc::ptr_eq(&before, &stated()), "a patch: stated again");
}

#[test]
fn a_membership_write_that_moves_an_encoders_end_states_its_facts_again() {
    let store = MemoryStore::new();
    store.upsert(encode_facts("grpc://e1:9000", Role::Encode, push_on("c1")));
    store.upsert(encode_facts("grpc://p1:9000", Role::Prefill, push_on("c1")));
    store.upsert(encode_facts(
        "grpc://d1:9000",
        Role::Decode,
        EncodeOffer::default(),
    ));
    let facts = || {
        let mut out = EncodePairBuffers::default();
        store
            .build_encode_pair(MODEL, &ask(Instant::now()), &mut out)
            .expect("a pair");
        let (_, encode) = out.pools();
        encode.facts.to_vec()
    };
    let before = facts();
    assert_eq!(before.len(), 1);
    assert!(before[0].pushes && !before[0].references);
    // The encoder registers again and takes references now: its end moved.
    let references = EncodeOffer {
        references: true,
        ..push_on("c1")
    };
    store.upsert(encode_facts("grpc://e1:9000", Role::Encode, references));
    let after = facts();
    assert!(after[0].references, "the door reads the end of this write");
    assert_eq!(after[0].class, before[0].class);
}
