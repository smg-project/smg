//! The cases of the encode columns a holder hands beside its lent encoder
//! and prefill lists.

use std::{ptr, sync::Arc, time::Instant};

use smg_worker::registry::Generation;

use super::{
    pairs::{pair_side, state, stray_prefill},
    Replica, DECODE, ENCODE, MODEL, PREFILL,
};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        Constraints, EncodeItems, EncodeLists, EncodePlaced, EncoderFacts, Identity, PairLists,
        Phase, PolicyEngine, Request, RequestFacts, Role, WorkerId,
    },
    store::{EncodeHandoff, EncodeOffer, WorkerFacts, WorkerStatus, WorkersStore},
    worker_host::OpenAdmission,
};

const ENCODE_1: &str = "http://encode-1:8000";
const ENCODE_2: &str = "http://encode-2:8000";
const ENCODE_3: &str = "http://encode-3:8000";
const PREFILL_1: &str = "http://prefill-1:8000";
const PREFILL_2: &str = "http://prefill-2:8000";
const PREFILL_3: &str = "http://prefill-3:8000";

/// What the row of the worker at `url` states: its role, its runtime and
/// its end of the encode handoff. Every worker offers a partner one KV
/// transport, so each prefill pairs with each decode.
fn end(url: &str, role: Role, runtime: &str, encode: EncodeOffer) -> WorkerFacts {
    WorkerFacts {
        engine_kind: runtime.to_string(),
        encode,
        ..pair_side(url, role, "t")
    }
}

/// A push on `connector`, in the role of `role`: an encoder states the
/// producer's role and a prefill the consumer's.
fn push(role: Role, connector: &str) -> EncodeOffer {
    EncodeOffer {
        handoff: EncodeHandoff::Push,
        producer: role == Role::Encode,
        consumer: role == Role::Prefill,
        connector: Some(connector.to_string()),
        references: false,
    }
}

/// What the row of the worker at `url` states under one runtime: a push on
/// `connector`, in its role's part.
fn pushing(url: &str, role: Role, connector: &str) -> WorkerFacts {
    end(url, role, "one", push(role, connector))
}

/// A rendezvous: no role and no connector.
fn rendezvous() -> EncodeOffer {
    EncodeOffer {
        handoff: EncodeHandoff::Rendezvous,
        ..EncodeOffer::default()
    }
}

/// What the encode side reads of an encoder of `class` that states a push.
fn pushes(class: u32) -> EncoderFacts {
    EncoderFacts {
        class,
        pushes: true,
        references: false,
    }
}

/// What the encode side reads of an encoder of `class` that states none.
fn meets(class: u32) -> EncoderFacts {
    EncoderFacts {
        class,
        ..EncoderFacts::default()
    }
}

/// The holder hands what an encode request's decision reads beside its
/// encoder list and its prefill list by the places of the two lists as
/// lent, under the store's encode rule over what the rows state. The
/// encoders of one end are one class, numbered where its first encoder is
/// lent, and a prefill names the class of the end it states. A prefill with
/// no row stands among the members and in no list, so it takes no place.
/// The columns are derived once.
#[test]
fn the_encode_columns_are_by_the_lent_lists_places_under_the_stores_rule() {
    let replica = Replica::new();
    let registry = replica.objects.registry();
    assert!(registry.register(stray_prefill()).is_some());
    let takes_references = EncodeOffer {
        references: true,
        ..push(Role::Encode, "a")
    };
    // The two roles are written by turns: a place is a worker's among the
    // workers of its own role.
    let rows = [
        (ENCODE, Role::Encode, "one", push(Role::Encode, "a")),
        (PREFILL, Role::Prefill, "one", rendezvous()),
        (ENCODE_1, Role::Encode, "one", rendezvous()),
        (PREFILL_1, Role::Prefill, "one", push(Role::Prefill, "a")),
        (ENCODE_2, Role::Encode, "one", takes_references),
        (PREFILL_2, Role::Prefill, "another", rendezvous()),
        (ENCODE_3, Role::Encode, "one", EncodeOffer::default()),
        (PREFILL_3, Role::Prefill, "one", EncodeOffer::default()),
    ];
    for (url, role, runtime, offer) in rows {
        state(&replica, end(url, role, runtime, offer));
    }
    let generation = replica.generation();
    assert_eq!(generation.members().len(), 9, "the worker with no row too");
    let columns = replica.objects.encode_columns(&generation);

    // The two encoders that push on one connector are the first class, and
    // the second of them takes an item's references. The encoder that
    // states no form has no end: class `0`.
    let references = EncoderFacts {
        references: true,
        ..pushes(1)
    };
    assert_eq!(columns.facts(), [pushes(1), meets(2), references, meets(0)]);
    // Each prefill names the class of its own end: the rendezvous of its
    // runtime, then the push on its connector. A rendezvous under a runtime
    // no encoder states names none, and neither does a prefill that states
    // no form.
    assert_eq!(columns.prefill_classes(), [2, 1, 0, 0]);
    let lent = |role: Role| replica.objects.lent(&generation, role).workers.len();
    assert_eq!((lent(Role::Encode), lent(Role::Prefill)), (4, 4));

    let again = replica.objects.encode_columns(&generation);
    assert!(ptr::eq(again, columns), "derived once");
    assert_eq!(replica.objects.unstated(), 1, "the lists were cut once");
}

/// Each column holds one entry a worker of its list where the rule holds
/// no class at all: encoders that state no end, and a model with no
/// encoder. The engine reads a column by its list's places, so a column of
/// another length would be refused.
#[test]
fn the_columns_hold_one_entry_a_worker_where_no_encoder_states_an_end() {
    let replica = Replica::new();
    for url in [PREFILL, PREFILL_1] {
        state(&replica, end(url, Role::Prefill, "one", rendezvous()));
    }
    let none = replica.generation();
    let columns = replica.objects.encode_columns(&none);
    assert_eq!(columns.facts(), []);
    assert_eq!(columns.prefill_classes(), [0, 0]);

    // An encoder of an unknown runtime states no end, whatever its form,
    // and one that states a push with no connector states none either.
    let bare = EncodeOffer {
        connector: None,
        ..push(Role::Encode, "a")
    };
    state(&replica, end(ENCODE, Role::Encode, "", rendezvous()));
    state(&replica, end(ENCODE_1, Role::Encode, "one", bare));
    let unstated = replica.generation();
    let columns = replica.objects.encode_columns(&unstated);
    assert_eq!(columns.facts(), [meets(0), pushes(0)]);
    assert_eq!(columns.prefill_classes(), [0, 0]);
}

/// The rule runs over whatever is lent. With no prefill lent, an encoder
/// keeps the class of the end it states and the prefill column is empty.
/// The store derives no class for such a model, and its pools read class
/// `0` on every encoder.
#[test]
fn an_encoder_keeps_the_class_of_its_end_where_no_prefill_is_lent() {
    let replica = Replica::new();
    state(&replica, pushing(ENCODE, Role::Encode, "a"));
    state(&replica, end(ENCODE_1, Role::Encode, "one", rendezvous()));
    let generation = replica.generation();
    let columns = replica.objects.encode_columns(&generation);
    assert_eq!(columns.facts(), [pushes(1), meets(2)]);
    assert!(columns.prefill_classes().is_empty());
}

/// A generation asked for its encode columns once the objects of workers
/// it lends were replaced reads no row for those workers: each states no
/// end at its place, an encoder of class `0` that states no push and a
/// prefill no encoder hands to. The workers that stand are classed as
/// their rows say.
#[test]
fn a_lent_worker_whose_object_was_replaced_states_no_end() {
    let replica = Replica::new();
    let sides = [
        (ENCODE, Role::Encode),
        (ENCODE_1, Role::Encode),
        (PREFILL, Role::Prefill),
        (PREFILL_1, Role::Prefill),
    ];
    let rows = sides.map(|(url, role)| state(&replica, pushing(url, role, "a")));
    let before = replica.generation();
    // The lists are cut ahead of the replacements.
    assert_eq!(replica.lent(&before, Role::Encode).len(), 2);
    for at in [0, 2] {
        let (url, role) = sides[at];
        let sees = WorkerFacts {
            vision: true,
            ..pushing(url, role, "a")
        };
        assert_eq!(state(&replica, sees), rows[at], "the row's id stands");
    }

    let stale = replica.objects.encode_columns(&before);
    assert_eq!(stale.facts(), [meets(0), pushes(1)]);
    assert_eq!(stale.prefill_classes(), [0, 1]);
    let after = replica.generation();
    let fresh = replica.objects.encode_columns(&after);
    assert_eq!(fresh.facts(), [pushes(1), pushes(1)]);
    assert_eq!(fresh.prefill_classes(), [1, 1]);
}

/// A row whose engine kind or whose encode offer moves gets a new object,
/// and the registry publishes another generation for it: the holder keeps
/// no object over a row that moved. So the generation that stood keeps the
/// columns it derived, and the new generation's say what the rows state
/// now.
#[test]
fn a_row_whose_encode_end_moves_gets_a_new_object_and_a_new_generation() {
    let replica = Replica::new();
    let encoder = state(&replica, pushing(ENCODE, Role::Encode, "a"));
    let prefill = |connector: &str| pushing(PREFILL, Role::Prefill, connector);
    let row = state(&replica, prefill("a"));
    let object_of = |row: WorkerId| {
        let id = replica.objects.object(row).expect("the row's object");
        replica.objects.registry().get(&id).expect("it stands")
    };
    let (before, stood) = (replica.generation(), object_of(row));
    let handed = replica.objects.encode_columns(&before);
    assert_eq!(handed.facts(), [pushes(1)]);
    assert_eq!(handed.prefill_classes(), [1]);

    // The prefill states its role on another connector, in place: its row
    // keeps its id.
    assert_eq!(state(&replica, prefill("b")), row, "the row's id stands");
    let moved = replica.generation();
    assert!(!Arc::ptr_eq(&before, &moved), "another generation");
    assert!(!Arc::ptr_eq(&stood, &object_of(row)), "another object");
    // An ask of the generation that stood derives nothing: its columns
    // stand.
    let again = replica.objects.encode_columns(&before);
    assert!(ptr::eq(again, handed), "one pair of columns a generation");
    let apart = replica.objects.encode_columns(&moved);
    assert_eq!(apart.facts(), [pushes(1)]);
    assert_eq!(apart.prefill_classes(), [0]);

    // The prefill comes back to the connector, and the encoder's engine
    // kind moves in place: the two state one end no longer.
    assert_eq!(state(&replica, prefill("a")), row);
    let together = replica.generation();
    let columns = replica.objects.encode_columns(&together);
    assert_eq!(columns.prefill_classes(), [1]);
    let stood = object_of(encoder);
    let another = end(ENCODE, Role::Encode, "another", push(Role::Encode, "a"));
    assert_eq!(state(&replica, another), encoder, "the row's id stands");
    assert!(!Arc::ptr_eq(&stood, &object_of(encoder)), "another object");
    let after = replica.generation();
    assert!(!Arc::ptr_eq(&together, &after), "another generation");
    let columns = replica.objects.encode_columns(&after);
    assert_eq!(columns.facts(), [pushes(1)]);
    assert_eq!(columns.prefill_classes(), [0]);
}

/// One fleet under the stock encode document, decided over the three lists
/// the holder lends, with the holder's pair classes and its encode columns
/// as they are. One encoder states an end, and one prefill states the same:
/// the decision picks that prefill and places the item on that encoder,
/// each named by its place in its list and by its row's id.
#[test]
fn the_lent_lists_and_the_holders_columns_place_an_item_of_one_fleet() {
    let mut replica = Replica::new();
    replica.objects = replica.objects.with_status_followed();
    let sides = [
        (PREFILL, Role::Prefill, push(Role::Prefill, "a")),
        (PREFILL_1, Role::Prefill, push(Role::Prefill, "b")),
        (DECODE, Role::Decode, EncodeOffer::default()),
        (ENCODE, Role::Encode, EncodeOffer::default()),
        (ENCODE_1, Role::Encode, push(Role::Encode, "b")),
    ];
    let rows = sides.map(|(url, role, offer)| {
        let facts = WorkerFacts {
            max_running: Some(4_096),
            ..end(url, role, "one", offer)
        };
        let id = replica.store.upsert(facts).id();
        replica.store.publish_now();
        let row = replica.store.by_key(url).expect("a row");
        let ready = WorkerStatus::Ready;
        assert!(replica.store.set_status(id, ready, row.revision).is_some());
        id
    });
    replica.objects.follow(&replica.store);
    let preset = "least_load";
    let document = PolicySpec::epd_presets(preset, preset, preset).expect("the stock presets");
    let applied = || AppliedSpec::validate(&document).expect("the stock document");
    let (spec, engine) = (applied(), PolicyEngine::new(applied()));
    let generation = replica.generation();
    let lent = |role: Role| replica.objects.lent(&generation, role);
    let columns = replica.objects.encode_columns(&generation);
    let lists = EncodeLists {
        pair: PairLists {
            prefill: lent(Role::Prefill),
            decode: lent(Role::Decode),
            classes: replica.objects.pair_classes(&generation, spec.pairing()),
            admission: &OpenAdmission,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        },
        encoders: lent(Role::Encode),
        encoders_places: None,
        encoders_stalled: &[],
        encoders_held: &[],
        facts: columns.facts(),
        prefill_classes: columns.prefill_classes(),
    };
    let mut request = Request {
        id: b"encode",
        facts: RequestFacts {
            model: MODEL,
            seed: 7,
            phase: Phase::Prefill,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    };
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let mut placed = EncodePlaced::new();
    let now = Instant::now();
    let decision = lists.decide(&engine, None, now, &mut request, items, &mut placed);

    let first = decision.pick().and_then(|pick| pick.targets.iter().next());
    let first = first.expect("a pick");
    let pair = first.pair.expect("a pair");
    assert_eq!((first.position, first.id), (1, rows[1]), "the prefill");
    assert_eq!((pair.decode_position, pair.decode_id), (0, rows[2]));
    let encoders = placed.encoders().iter();
    let encoders: Vec<(u32, WorkerId)> = encoders.map(|at| (at.position, at.id)).collect();
    assert_eq!(encoders, [(1, rows[4])], "the item's encoder");
}
