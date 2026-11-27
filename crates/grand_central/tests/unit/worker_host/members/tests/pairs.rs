//! The cases of the pair classes a holder hands beside its two lent lists.

use std::{ptr, sync::Arc, time::Instant};

use openai_protocol::model_card::ModelCard;
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    registry::Generation,
    worker::WorkerType,
};

use super::{Replica, DECODE, MODEL, PREFILL, STRAY};
// The cases of a debug build's own checks.
#[cfg(debug_assertions)]
use crate::store::pairing::{Member, PairClasses};
use crate::{
    document::spec::{AppliedSpec, Pairing, PolicySpec},
    engine::{
        Constraints, Decision, InProcess, PairAdmission, PairLists, Phase, Request, RequestFacts,
        Role, WireModes, WorkerId,
    },
    store::{pairing::PairingDescriptor, Registrant, WorkerFacts, WorkersStore},
    worker_host::OpenAdmission,
};

const PREFILL_1: &str = "http://prefill-1:8000";
const PREFILL_2: &str = "http://prefill-2:8000";
const DECODE_1: &str = "http://decode-1:8000";
const DECODE_2: &str = "http://decode-2:8000";

/// Write the row `facts` state through the holder: the row's id.
pub(super) fn state(replica: &Replica, facts: WorkerFacts) -> WorkerId {
    let written = replica
        .objects
        .upsert(&replica.store, facts, Registrant::Control);
    written.expect("the store takes the write").id()
}

/// What the row of a pair's worker at `url` states: its side, and the KV
/// transport it offers a partner.
pub(super) fn pair_side(url: &str, role: Role, transport: &str) -> WorkerFacts {
    let descriptor = PairingDescriptor {
        transport: Some(transport.to_string()),
        ..PairingDescriptor::default()
    };
    WorkerFacts {
        key: url.to_string(),
        engine: url.to_string(),
        incarnation: 1,
        role,
        models: vec![MODEL.to_string()],
        dp_size: 1,
        pairing: Some(Arc::new(descriptor)),
        ..WorkerFacts::default()
    }
}

/// The worker crate's word for no runtime with its first letter alone
/// upper: a letter case that is neither all lower nor all upper.
pub(super) fn no_runtime_in_title_case() -> String {
    let none = smg_worker::worker::RuntimeType::default().as_str();
    let (first, rest) = none.split_at(1);
    format!("{}{rest}", first.to_uppercase())
}

/// A prefill another writer registered: the holder lays no row for it.
pub(super) fn stray_prefill() -> Arc<dyn Worker> {
    let worker = <BasicWorkerBuilder>::new(STRAY).model(ModelCard::new(MODEL));
    Arc::new(worker.worker_type(WorkerType::Prefill).build())
}

/// The holder hands a pair's classes by the places of its two lists as
/// lent, under the store's pairing rule over what the rows state. A prefill
/// with no row stands among the members and in no list, so it takes no
/// place. Each pairing mode has a set of its own, derived once.
#[test]
fn a_pairs_classes_are_by_the_lent_lists_places_under_the_stores_rule() {
    let replica = Replica::new();
    let registry = replica.objects.registry();
    assert!(registry.register(stray_prefill()).is_some());
    let sides = [
        (PREFILL, Role::Prefill, "a"),
        (PREFILL_1, Role::Prefill, "b"),
        (PREFILL_2, Role::Prefill, "a"),
        (DECODE, Role::Decode, "b"),
        (DECODE_1, Role::Decode, "c"),
        (DECODE_2, Role::Decode, "a"),
    ];
    for (url, role, transport) in sides {
        state(&replica, pair_side(url, role, transport));
    }
    let generation = replica.generation();
    assert_eq!(generation.members().len(), 7, "the worker with no row too");
    let ask = |mode: Pairing| replica.objects.pair_classes(&generation, mode);

    // Two transports that differ do not pair. So each prefill pairs with
    // the one decode of its transport, and the decode of a third transport
    // with nobody. A class is numbered where its first worker is lent.
    let lenient = ask(Pairing::Lenient);
    assert_eq!(lenient.prefills(), [1, 2, 1]);
    assert_eq!(lenient.decodes(), [1, 0, 2]);
    assert_eq!(lenient.table().validate(), Ok(()));
    assert_eq!(
        (lenient.table().of(&1), lenient.table().of(&2)),
        (&[2][..], &[1][..])
    );
    let partners = |prefill: usize| -> Vec<usize> {
        let pairs = |decode: &usize| lenient.pairs(prefill, *decode);
        (0..3).filter(pairs).collect()
    };
    assert_eq!([partners(0), partners(1), partners(2)], [[2], [0], [2]]);

    // Strict refuses a pair whose runtime neither side names, and off
    // compares nothing.
    let strict = ask(Pairing::Strict);
    assert_eq!(
        (strict.prefills(), strict.decodes()),
        (&[0; 3][..], &[0; 3][..])
    );
    let off = ask(Pairing::Off);
    assert_eq!((off.prefills(), off.decodes()), (&[1; 3][..], &[1; 3][..]));
    let asked = [
        (Pairing::Lenient, lenient),
        (Pairing::Strict, strict),
        (Pairing::Off, off),
    ];
    for (mode, set) in asked {
        assert!(ptr::eq(ask(mode), set), "{mode:?}: derived once");
    }
    assert_eq!(replica.objects.unstated(), 1, "the lists were cut once");
}

/// The rule reads what a row states beside its descriptor too: its wires and
/// its engine kind. A worker on the HTTP wire alone pairs with no worker on
/// another wire, and two engine kinds that differ do not pair.
#[test]
fn a_pairs_classes_read_each_rows_wires_and_engine_kind() {
    let replica = Replica::new();
    let sides = [
        (PREFILL, Role::Prefill, WireModes::HTTP, "one"),
        (PREFILL_1, Role::Prefill, WireModes::GRPC, "one"),
        (PREFILL_2, Role::Prefill, WireModes::GRPC, "another"),
        (DECODE, Role::Decode, WireModes::GRPC, "one"),
    ];
    for (url, role, wire_modes, kind) in sides {
        let facts = WorkerFacts {
            wire_modes,
            engine_kind: kind.to_string(),
            ..pair_side(url, role, "a")
        };
        state(&replica, facts);
    }
    let generation = replica.generation();
    let classes = replica.objects.pair_classes(&generation, Pairing::Lenient);
    assert_eq!(
        (classes.prefills(), classes.decodes()),
        (&[0, 1, 0][..], &[1][..])
    );
}

/// A row keeps its runtime's name as its host wrote it, and the holder
/// hands the rule the worker crate's reading of that name. Two spellings of
/// one runtime are one name on either side, and so are two names that
/// crate reads as one runtime. Beside that name the two here state one
/// transport and no other fact of a handoff, and pair under every mode, on
/// a wire whose requests are one runtime's.
#[test]
fn two_spellings_of_one_runtime_are_one_name_to_a_pairs_classes() {
    use smg_worker::worker::RuntimeType;

    let name = RuntimeType::Generic.as_str();
    let spelled = [
        (name.to_uppercase(), name.to_string()),
        (name.to_string(), name.to_uppercase()),
        ("TensorRT-LLM".to_string(), "trtllm".to_string()),
    ];
    for (prefill, decode) in spelled {
        let replica = Replica::new();
        let sides = [
            (PREFILL, Role::Prefill, &prefill),
            (DECODE, Role::Decode, &decode),
        ];
        for (url, role, kind) in sides {
            let facts = WorkerFacts {
                wire_modes: WireModes::GRPC,
                engine_kind: kind.clone(),
                ..pair_side(url, role, "a")
            };
            state(&replica, facts);
        }
        let generation = replica.generation();
        for mode in [Pairing::Off, Pairing::Lenient, Pairing::Strict] {
            let classes = replica.objects.pair_classes(&generation, mode);
            assert_eq!(
                (classes.prefills(), classes.decodes()),
                (&[1][..], &[1][..]),
                "{prefill} and {decode} under {mode:?}"
            );
        }
    }
}

/// The worker crate's word for no runtime is no name to the rule, in any
/// letter case: such a row's runtime is unknown. Strict refuses a pair
/// whose runtime neither side names, though both state one protocol and
/// both write the word alike, and lenient pairs it. Two rows that write
/// the word in two letter cases are of no two runtimes. Beside a worker
/// that names its runtime such a row is of no different runtime: on a wire
/// whose requests are one runtime's, the two here report one transport and
/// pair leniently, and strict refuses the side that names none.
#[test]
fn the_word_for_no_runtime_is_no_name_to_a_pairs_classes() {
    use smg_worker::worker::RuntimeType;

    let none = RuntimeType::default().as_str();
    let upper = none.to_uppercase();
    let mixed = no_runtime_in_title_case();
    let ask = |replica: &Replica, mode: Pairing| {
        let generation = replica.generation();
        let classes = replica.objects.pair_classes(&generation, mode);
        (classes.prefills().to_vec(), classes.decodes().to_vec())
    };

    // Two rows on the HTTP wire that state one protocol.
    let agreed = |url: &str, role: Role, kind: &str| {
        let descriptor = PairingDescriptor {
            protocol: Some("p".to_string()),
            ..PairingDescriptor::default()
        };
        WorkerFacts {
            wire_modes: WireModes::HTTP,
            engine_kind: kind.to_string(),
            pairing: Some(Arc::new(descriptor)),
            ..pair_side(url, role, "a")
        }
    };
    // Both write the word alike, in three letter cases.
    for word in [none, upper.as_str(), mixed.as_str()] {
        let alike = Replica::new();
        state(&alike, agreed(PREFILL, Role::Prefill, word));
        state(&alike, agreed(DECODE, Role::Decode, word));
        assert_eq!(ask(&alike, Pairing::Strict), (vec![0], vec![0]), "{word}");
        assert_eq!(ask(&alike, Pairing::Lenient), (vec![1], vec![1]), "{word}");
    }
    // Each writes it in a letter case of its own.
    let apart = Replica::new();
    state(&apart, agreed(PREFILL, Role::Prefill, none));
    state(&apart, agreed(DECODE, Role::Decode, &mixed));
    assert_eq!(ask(&apart, Pairing::Strict), (vec![0], vec![0]));
    assert_eq!(ask(&apart, Pairing::Lenient), (vec![1], vec![1]));

    // Such a row beside a decode that names its runtime.
    let sided = |url: &str, role: Role, kind: &str| WorkerFacts {
        wire_modes: WireModes::GRPC,
        engine_kind: kind.to_string(),
        ..pair_side(url, role, "a")
    };
    let named = Replica::new();
    state(&named, sided(PREFILL, Role::Prefill, none));
    let known = RuntimeType::Generic.as_str();
    state(&named, sided(DECODE, Role::Decode, known));
    assert_eq!(ask(&named, Pairing::Lenient), (vec![1], vec![1]));
    assert_eq!(ask(&named, Pairing::Off), (vec![1], vec![1]));
    assert_eq!(ask(&named, Pairing::Strict), (vec![0], vec![0]));
}

/// A name the worker crate does not know stays the row's own word to the
/// rule, compared as the row wrote it. One that is not empty is one known
/// runtime on both sides, under strict too, and two spellings of it are
/// two runtimes, which no mode pairs on a wire whose requests are one
/// runtime's.
#[test]
fn a_name_the_worker_crate_does_not_know_is_compared_as_the_row_wrote_it() {
    let replica = Replica::new();
    let sides = [
        (PREFILL, Role::Prefill, "one"),
        (PREFILL_1, Role::Prefill, "ONE"),
        (DECODE, Role::Decode, "one"),
    ];
    for (url, role, kind) in sides {
        let facts = WorkerFacts {
            wire_modes: WireModes::GRPC,
            engine_kind: kind.to_string(),
            ..pair_side(url, role, "a")
        };
        state(&replica, facts);
    }
    let generation = replica.generation();
    for mode in [Pairing::Off, Pairing::Lenient, Pairing::Strict] {
        let classes = replica.objects.pair_classes(&generation, mode);
        assert_eq!(
            (classes.prefills(), classes.decodes()),
            (&[1, 0][..], &[1][..]),
            "{mode:?}"
        );
    }
}

/// A row that states no pairing descriptor pairs by what its worker object
/// offers: the pairing facts of the row's labels and engine kind, as the
/// worker's own rule reads them. A row that states a descriptor is read by
/// it, whatever its labels say. A row that is relabelled pairs by its new
/// labels in the generation its new object publishes.
#[test]
fn a_row_that_states_no_descriptor_pairs_by_its_labels() {
    let replica = Replica::new();
    let labelled = |transport: &str| -> WorkerFacts {
        let label = "disaggregation_transfer_backend".to_string();
        WorkerFacts {
            engine_kind: "sglang".to_string(),
            labels: [(label, transport.to_string())].into(),
            ..WorkerFacts::default()
        }
    };
    let silent = |url: &str, role: Role, transport: &str| WorkerFacts {
        pairing: None,
        ..WorkerFacts {
            engine_kind: labelled(transport).engine_kind,
            labels: labelled(transport).labels,
            ..pair_side(url, role, "stated")
        }
    };
    state(&replica, silent(PREFILL, Role::Prefill, "nixl"));
    state(&replica, silent(PREFILL_1, Role::Prefill, "mooncake"));
    state(&replica, silent(DECODE, Role::Decode, "mooncake"));
    // This decode's labels name one transport and its descriptor another.
    let stated = WorkerFacts {
        engine_kind: labelled("mooncake").engine_kind,
        labels: labelled("mooncake").labels,
        ..pair_side(DECODE_1, Role::Decode, "nixl")
    };
    state(&replica, stated);
    let generation = replica.generation();
    let lenient = replica.objects.pair_classes(&generation, Pairing::Lenient);
    let partners = |prefill: usize| -> Vec<usize> {
        let pairs = |decode: &usize| lenient.pairs(prefill, *decode);
        (0..2).filter(pairs).collect()
    };
    assert_eq!([partners(0), partners(1)], [[1], [0]]);
    // Strict reads the same facts: each side reports its transport.
    let strict = replica.objects.pair_classes(&generation, Pairing::Strict);
    assert!(
        strict.pairs(1, 0),
        "two rows whose labels report one transport"
    );
    assert!(!strict.pairs(0, 0), "two transports that differ");

    // A relabel gets the row a new object, and the model another
    // generation. There the two prefills report one transport and have one
    // partner, the same one. The set derived before stands as it was.
    state(&replica, silent(PREFILL, Role::Prefill, "mooncake"));
    let relabelled = replica.generation();
    let moved = replica.objects.pair_classes(&relabelled, Pairing::Lenient);
    let now = |prefill: usize| -> Vec<usize> {
        let pairs = |decode: &usize| moved.pairs(prefill, *decode);
        (0..2).filter(pairs).collect()
    };
    assert_eq!(now(0), now(1), "one transport, one partner");
    assert_eq!(now(0).len(), 1);
    assert_eq!([partners(0), partners(1)], [[1], [0]]);
}

/// A generation asked for its pair classes once the objects of workers it
/// lends were replaced reads no row for those workers: each pairs with
/// nobody, at its place on either side, and the workers that stand pair as
/// their rows say. None takes the class of the object that took its place.
#[test]
fn a_lent_worker_whose_object_was_replaced_pairs_with_nobody() {
    let replica = Replica::new();
    let sides = [
        (PREFILL, Role::Prefill),
        (PREFILL_1, Role::Prefill),
        (DECODE, Role::Decode),
        (DECODE_1, Role::Decode),
    ];
    let rows = sides.map(|(url, role)| replica.write(url, role, false));
    let before = replica.generation();
    // The lists are cut ahead of the replacements.
    assert_eq!(replica.lent(&before, Role::Decode).len(), 2);
    let replaced = [
        (PREFILL, Role::Prefill, rows[0]),
        (DECODE, Role::Decode, rows[2]),
    ];
    for (url, role, row) in replaced {
        assert_eq!(replica.write(url, role, true), row, "the row's id stands");
    }

    let stale = replica.objects.pair_classes(&before, Pairing::Lenient);
    assert_eq!(
        (stale.prefills(), stale.decodes()),
        (&[0, 1][..], &[0, 1][..])
    );
    let pairs =
        [(0, 0), (0, 1), (1, 0), (1, 1)].map(|(prefill, decode)| stale.pairs(prefill, decode));
    assert_eq!(pairs, [false, false, false, true]);
    let after = replica.generation();
    let fresh = replica.objects.pair_classes(&after, Pairing::Lenient);
    assert_eq!(
        (fresh.prefills(), fresh.decodes()),
        (&[1, 1][..], &[1, 1][..])
    );
}

/// A row whose pairing facts move gets a new object, and the registry
/// publishes another generation for it: the holder keeps no object over a
/// row that moved. So the generation that stood keeps the one set it
/// derived under the mode, and the new generation's classes say what the
/// row states now.
#[test]
fn a_row_whose_pairing_facts_move_gets_a_new_object_and_a_new_generation() {
    let replica = Replica::new();
    state(&replica, pair_side(PREFILL, Role::Prefill, "a"));
    let decode = state(&replica, pair_side(DECODE, Role::Decode, "a"));
    let object = || {
        let id = replica.objects.object(decode).expect("the row's object");
        replica.objects.registry().get(&id).expect("it stands")
    };
    let (before, stood) = (replica.generation(), object());
    let paired = replica.objects.pair_classes(&before, Pairing::Lenient);
    assert_eq!((paired.prefills(), paired.decodes()), (&[1][..], &[1][..]));

    // The decode offers another transport, in place: its row keeps its id.
    let moved = state(&replica, pair_side(DECODE, Role::Decode, "b"));
    assert_eq!(moved, decode, "the row's id stands");
    let after = replica.generation();
    assert!(!Arc::ptr_eq(&before, &after), "another generation");
    assert!(!Arc::ptr_eq(&stood, &object()), "another object");

    // An ask of the generation that stood derives nothing: its set stands.
    let again = replica.objects.pair_classes(&before, Pairing::Lenient);
    assert!(ptr::eq(again, paired), "one set a mode");
    let apart = replica.objects.pair_classes(&after, Pairing::Lenient);
    assert_eq!((apart.prefills(), apart.decodes()), (&[0][..], &[0][..]));
}

/// The classes the rule derived over `prefills` prefills and `decodes`
/// decodes that report nothing.
#[cfg(debug_assertions)]
fn derived_over(prefills: u64, decodes: u64) -> PairClasses {
    let side = |ids: std::ops::Range<u64>| -> Vec<Member<'static>> {
        let member = |id| Member::new(WorkerId(id), WireModes::default(), "", None);
        ids.map(member).collect()
    };
    let (prefills, decodes) = (side(0..prefills), side(prefills..prefills + decodes));
    PairClasses::derive(&prefills, &decodes, Pairing::Lenient)
}

/// The holder hands the rule's classes on by the places of its two lists,
/// and says of each place whether a member was derived for it. A debug
/// build holds it to that word: a place said to hold a member that no
/// derivation made is refused.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "a member derived for every place given")]
fn a_place_given_with_no_member_derived_is_refused_in_a_debug_build() {
    let _ = derived_over(1, 1).by_place([true, true], [true]);
}

/// And so is a member derived that the caller says of no place.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "a place given for every member derived")]
fn a_member_derived_with_no_place_given_is_refused_in_a_debug_build() {
    let _ = derived_over(1, 2).by_place([true], [false, true]);
}

/// One fleet under the stock pair document, decided over the lists the
/// holder lends, with the holder's pair classes and a controller's open
/// admission. One prefill and one decode share a transport, so the fleet
/// holds one pair, and the decision picks it.
#[test]
fn the_lent_lists_and_the_holders_classes_pick_the_one_pair_of_one_fleet() {
    let replica = Replica::new();
    let sides = [
        (PREFILL, Role::Prefill, "a"),
        (PREFILL_1, Role::Prefill, "b"),
        (DECODE, Role::Decode, "c"),
        (DECODE_1, Role::Decode, "b"),
    ];
    let rows = sides.map(|(url, role, transport)| {
        let facts = WorkerFacts {
            max_running: Some(4_096),
            ..pair_side(url, role, transport)
        };
        let written = replica
            .objects
            .upsert(&replica.store, facts, Registrant::Control);
        let id = written.expect("the store takes the write").id();
        replica.store.publish_now();
        let row = replica.store.by_key(url).expect("a row");
        replica
            .objects
            .connect_row(&replica.store, id, row.revision);
        id
    });
    let the_pair = Some((rows[1], rows[3]));
    let document = PolicySpec::pair_presets("least_load", "least_load").expect("the stock presets");
    let applied = || AppliedSpec::validate(&document).expect("the stock document");
    let mut request = Request {
        id: b"pair",
        facts: RequestFacts {
            model: MODEL,
            seed: 7,
            phase: Phase::Prefill,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    };
    // The pair a decision picked: its first target's prefill and decode.
    let picked = |decision: &Decision| -> Option<(WorkerId, WorkerId)> {
        let first = decision.pick()?.targets.iter().next()?;
        Some((first.id, first.pair?.decode_id))
    };

    // Every place of both lists is open to the request: the holder's
    // classes alone say who pairs.
    let (spec, engine) = (applied(), InProcess::new(applied()));
    let generation = replica.generation();
    let lent = |role: Role| replica.objects.lent(&generation, role);
    let lists = PairLists {
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
    };
    assert!(OpenAdmission.has_capacity(0) && OpenAdmission.has_room(1));
    let now = Instant::now();
    let over_lists = lists.decide(&engine, None, now, &mut request, None);
    assert_eq!(picked(&over_lists), the_pair, "over the lists lent");
}
