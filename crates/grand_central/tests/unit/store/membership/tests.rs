use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use super::{base, NotRegistrant, WorkerFactsChange, HALF};
use crate::{
    engine::{
        demands::DemandFacts,
        view::{Role, WireModes, WorkerId},
    },
    store::{
        identity, pairing::PairingDescriptor, EncodeOffer, MembershipChange, ProviderKeyError,
        Registrant, WorkerFacts, WorkerRow, WorkerStatus,
    },
};

#[test]
fn the_rows_maker_passes_and_any_other_caller_is_told_who_made_it() -> Result<(), ProviderKeyError>
{
    let callers = [
        Registrant::Control,
        Registrant::provider("pool", "first")?,
        Registrant::provider("pool", "second")?,
        Registrant::provider("other", "first")?,
    ];
    let control = Err((
        NotRegistrant::Control,
        "the row was made by a control call: its maker alone changes it",
    ));
    let another_key = Err((
        NotRegistrant::AnotherKey { provider: "pool" },
        "the row is provider pool's under another key: that key's caller changes it",
    ));
    let provider = Err((
        NotRegistrant::Provider { provider: "pool" },
        "the row is provider pool's: its worker is changed at the provider's source",
    ));
    for (maker, expected) in [
        (Registrant::Control, [Ok(()), control, control, control]),
        (
            Registrant::provider("pool", "first")?,
            [provider, Ok(()), another_key, provider],
        ),
    ] {
        let row = WorkerRow {
            maker,
            id: WorkerId(1),
            facts: WorkerFacts::default(),
            status: WorkerStatus::Pending,
            revision: 1,
            admitted_at: Instant::now(),
        };
        for (caller, expected) in callers.iter().zip(expected) {
            assert_eq!(
                row.made_by(caller).map_err(|not| (not, not.to_string())),
                expected.map_err(|(not, sentence)| (not, sentence.to_owned())),
                "maker {:?}, caller {caller:?}",
                row.maker
            );
        }
    }
    Ok(())
}

#[test]
fn facts_under_another_incarnation_are_never_unmoved() {
    let standing = WorkerFacts {
        key: "http://worker:8000".to_owned(),
        incarnation: 7,
        ..WorkerFacts::default()
    };
    let fresh = WorkerFacts {
        incarnation: 0,
        ..standing.clone()
    };
    assert_eq!(
        standing.compare_to_standing(&standing),
        WorkerFactsChange::Unmoved
    );
    assert_eq!(
        fresh.compare_to_standing(&standing),
        WorkerFactsChange::InPlace
    );
}

#[test]
fn equal_facts_move_nothing_a_label_changes_in_place_and_a_role_a_wire_or_a_model_is_another_life()
{
    let standing = WorkerFacts {
        role: Role::Single,
        wire_modes: WireModes::HTTP,
        models: vec!["one".to_owned(), "two".to_owned()],
        ..WorkerFacts::default()
    };
    assert_eq!(
        standing.clone().compare_to_standing(&standing),
        WorkerFactsChange::Unmoved
    );

    let mut relabelled = standing.clone();
    relabelled
        .labels
        .insert("pool".to_owned(), "blue".to_owned());
    assert_eq!(
        relabelled.compare_to_standing(&standing),
        WorkerFactsChange::InPlace
    );

    for (field, changed) in [
        (
            "role",
            WorkerFacts {
                role: Role::Decode,
                ..standing.clone()
            },
        ),
        (
            "wire modes",
            WorkerFacts {
                wire_modes: WireModes::GRPC,
                ..standing.clone()
            },
        ),
        (
            "models",
            WorkerFacts {
                models: vec!["one".to_owned(), "three".to_owned()],
                ..standing.clone()
            },
        ),
    ] {
        assert_eq!(
            changed.compare_to_standing(&standing),
            WorkerFactsChange::Another,
            "{field}"
        );
    }

    let mut reordered = standing.clone();
    reordered.models.reverse();
    assert_eq!(
        reordered.compare_to_standing(&standing),
        WorkerFactsChange::InPlace
    );
}

/// What a worker states for a request's hard demands says which
/// requests it can take at all, as its wires do: a registration that
/// changes it starts another life, whichever way the number moves.
#[test]
fn a_changed_demand_fact_is_another_life() {
    let stating = |runtime| WorkerFacts {
        demand_facts: DemandFacts {
            runtime,
            ..DemandFacts::default()
        },
        ..WorkerFacts::default()
    };
    for (standing, fresh) in [(7, 9), (7, 0), (0, 7)] {
        assert_eq!(
            stating(fresh).compare_to_standing(&stating(standing)),
            WorkerFactsChange::Another,
            "{standing} to {fresh}"
        );
    }
    assert_eq!(
        stating(7).compare_to_standing(&stating(7)),
        WorkerFactsChange::Unmoved
    );
}

/// Whether a worker takes images says which requests it can take at
/// all, as a demand fact does: the door reads the bit for a request
/// that needs vision. A registration that changes it starts another
/// life, whichever way it turns.
#[test]
fn a_changed_vision_is_another_life() {
    let seeing = |vision| WorkerFacts {
        vision,
        ..WorkerFacts::default()
    };
    for (standing, fresh) in [(true, false), (false, true)] {
        assert_eq!(
            seeing(fresh).compare_to_standing(&seeing(standing)),
            WorkerFactsChange::Another,
            "{standing} to {fresh}"
        );
    }
    assert_eq!(
        seeing(true).compare_to_standing(&seeing(true)),
        WorkerFactsChange::Unmoved
    );
}

#[test]
fn a_run_draws_its_base_below_the_bound() {
    assert!(base() < HALF);
    assert!(base() < HALF);
}

fn stated() -> WorkerFacts {
    WorkerFacts {
        key: "url".to_owned(),
        ..WorkerFacts::default()
    }
}

#[test]
fn a_fact_past_its_bound_is_refused_by_its_name() {
    use super::FactPastBound;

    type Set = fn(&mut WorkerFacts, usize);
    let limits: [(&str, usize, &str, Set); 8] = [
        ("key", 2_048, "a URL is at most 2048 bytes", |f, n| {
            f.key = "a".repeat(n);
        }),
        ("engine", 2_048, "a URL is at most 2048 bytes", |f, n| {
            f.engine = "a".repeat(n);
        }),
        (
            "engine_kind",
            256,
            "a runtime is at most 256 bytes",
            |f, n| {
                f.engine_kind = "a".repeat(n);
            },
        ),
        ("models", 256, "a model id is at most 256 bytes", |f, n| {
            f.models = vec!["a".repeat(n)];
        }),
        ("zone", 256, "a zone is at most 256 bytes", |f, n| {
            f.zone = Some(" ".repeat(n));
        }),
        ("labels", 128, "a row holds at most 128 labels", |f, n| {
            f.labels = (0..n).map(|key| (key.to_string(), String::new())).collect();
        }),
        (
            "labels",
            128,
            "a label's key is at most 128 bytes",
            |f, n| {
                f.labels = [(" ".repeat(n), String::new())].into();
            },
        ),
        (
            "labels",
            4_096,
            "a label's value is at most 4096 bytes",
            |f, n| {
                f.labels = [(String::new(), " ".repeat(n))].into();
            },
        ),
    ];
    let maker = Registrant::Control;
    for (fact, bound, why, set) in limits {
        let mut facts = stated();
        set(&mut facts, bound);
        assert_eq!(facts.bounded(&maker), Ok(()), "{fact} at {bound}");
        set(&mut facts, bound + 1);
        assert_eq!(facts.bounded(&maker), Err(FactPastBound { fact, why }));
    }

    let mut facts = stated();
    facts.key.clear();
    assert_eq!(
        facts.bounded(&maker),
        Err(FactPastBound {
            fact: "key",
            why: "a registration names its worker's URL"
        })
    );
    facts.engine = "é".repeat(1_025);
    facts.engine_kind = "two words".to_owned();
    facts.models = vec![String::new()];
    facts.zone = Some("a".repeat(257));
    facts.labels.insert("a".to_owned(), "a".repeat(4_097));

    for bad in ["url\u{2003}other", "url\u{7f}"] {
        facts.key = bad.to_owned();
        assert_eq!(
            facts.bounded(&maker),
            Err(FactPastBound {
                fact: "key",
                why: "a URL holds no blank and no control character",
            })
        );
    }
    facts.key = stated().key;
    assert_eq!(
        facts.bounded(&maker),
        Err(FactPastBound {
            fact: "engine",
            why: "a URL is at most 2048 bytes",
        })
    );
    for bad in ["one\u{2003}two", "one\u{7f}"] {
        facts.engine = bad.to_owned();
        assert_eq!(
            facts.bounded(&maker),
            Err(FactPastBound {
                fact: "engine",
                why: "a URL holds no blank and no control character",
            })
        );
        facts.engine.clear();
        facts.engine_kind = bad.to_owned();
        assert_eq!(
            facts.bounded(&maker),
            Err(FactPastBound {
                fact: "engine_kind",
                why: "a runtime holds no blank and no control character",
            })
        );
    }
    facts.engine_kind.clear();
    for bad in ["", "one\u{2003}two", "one\u{7f}"] {
        facts.models = vec!["m".to_owned(), bad.to_owned()];
        assert_eq!(
            facts.bounded(&maker),
            Err(FactPastBound {
                fact: "models",
                why: "a model id is not empty and holds no blank and no control character",
            })
        );
    }
    facts.models.clear();
    assert_eq!(
        facts.bounded(&maker),
        Err(FactPastBound {
            fact: "zone",
            why: "a zone is at most 256 bytes",
        })
    );
    facts.zone = None;
    assert_eq!(
        facts.bounded(&maker),
        Err(FactPastBound {
            fact: "labels",
            why: "a label's value is at most 4096 bytes",
        })
    );
}

#[test]
fn a_row_weighs_its_texts_as_often_as_it_tells_them() -> Result<(), ProviderKeyError> {
    let mut facts = WorkerFacts {
        engine: "engine".to_owned(),
        engine_kind: "kind".to_owned(),
        zone: Some("zone-a".to_owned()),
        models: vec!["m".to_owned(), "other".to_owned()],
        labels: [
            ("tier".to_owned(), "gold".to_owned()),
            ("policy".to_owned(), "fast".to_owned()),
        ]
        .into(),
        ..stated()
    };
    assert_eq!(facts.weight(&Registrant::Control), 270);
    facts.signals.scrape_interval_ms = 1;
    assert_eq!(facts.weight(&Registrant::Control), 279);
    assert_eq!(facts.weight(&Registrant::provider("pool", "work")?), 297);
    Ok(())
}

#[test]
fn a_long_list_of_model_ids_is_bounded_by_the_rows_size_alone() {
    use super::FactPastBound;

    let mut facts = stated();
    let maker = Registrant::Control;
    facts.models = (0..129).map(|id| id.to_string()).collect();
    assert_eq!(facts.bounded(&maker), Ok(()));

    // 186 bytes before the ids, then 62 ids of 261 bytes and one of 16.
    facts.models = vec!["m".repeat(256); 62];
    facts.models.push("m".repeat(11));
    assert_eq!(facts.weight(&maker), 16_384);
    assert_eq!(facts.bounded(&maker), Ok(()));
    facts.models[62].push('m');
    assert_eq!(
        facts.bounded(&maker),
        Err(FactPastBound {
            fact: "row",
            why: "a row is at most 16384 bytes on the wire",
        })
    );
    facts.models[62].pop();
    assert_eq!(
        facts.bounded(&Registrant::provider("pool", "work").expect("a maker")),
        Err(FactPastBound {
            fact: "row",
            why: "a row is at most 16384 bytes on the wire",
        })
    );
}

/// A row that stands under the key [`stated`] names, in `status`,
/// admitted at `admitted_at` and made by a provider.
fn standing(status: WorkerStatus, admitted_at: Instant) -> Result<WorkerRow, ProviderKeyError> {
    Ok(WorkerRow {
        id: WorkerId(9),
        facts: WorkerFacts {
            incarnation: 1,
            role: Role::Single,
            wire_modes: WireModes::HTTP,
            models: vec!["one".to_owned(), "two".to_owned()],
            dp_size: 1,
            ..stated()
        },
        status,
        revision: 4,
        admitted_at,
        maker: Registrant::provider("pool", "first")?,
    })
}

/// The rule's first arm: a key no row stands under.
#[test]
fn facts_laid_over_no_row_register_a_pending_row_by_their_maker() -> Result<(), ProviderKeyError> {
    let (maker, now) = (Registrant::provider("pool", "first")?, Instant::now());
    let facts = WorkerFacts {
        incarnation: 7,
        ..stated()
    };
    let id = identity("url", 7);
    let (change, row) = WorkerRow::laid_over(facts.clone(), None, maker.clone(), now);
    assert_eq!(change, MembershipChange::Registered { id });
    let registered = WorkerRow {
        id,
        facts,
        status: WorkerStatus::Pending,
        revision: 1,
        admitted_at: now,
        maker,
    };
    assert_eq!(row, registered);
    Ok(())
}

/// The rule's second arm: the standing row's own incarnation. The row
/// keeps what is no fact of a registration: its id, its status, when it
/// was admitted and who made it.
#[test]
fn facts_laid_over_their_own_incarnation_change_the_row_in_place() -> Result<(), ProviderKeyError> {
    let earlier = Instant::now();
    let now = earlier + Duration::from_secs(5);
    for status in [WorkerStatus::Ready, WorkerStatus::Draining] {
        let standing = standing(status, earlier)?;
        let id = standing.id;
        let mut facts = standing.facts.clone();
        facts.labels.insert("zone".to_owned(), "b".to_owned());
        let (change, row) =
            WorkerRow::laid_over(facts.clone(), Some(&standing), Registrant::Control, now);
        assert_eq!(change, MembershipChange::Relabelled { id });
        let relabelled = WorkerRow {
            facts: facts.clone(),
            revision: 5,
            ..standing.clone()
        };
        assert_eq!(row, relabelled, "over a row that is {status:?}");
        // A width that moved is told as a resize.
        facts.dp_size = 2;
        let (change, row) =
            WorkerRow::laid_over(facts.clone(), Some(&standing), Registrant::Control, now);
        assert_eq!(change, MembershipChange::Resized { id });
        let resized = WorkerRow {
            facts,
            revision: 5,
            ..standing
        };
        assert_eq!(row, resized, "over a row that is {status:?}");
    }
    Ok(())
}

/// The rule's third arm: another incarnation under the key. The same
/// worker's status is handed on, a drain never, and the key's maker
/// stands through its new life, whoever states it.
#[test]
fn facts_laid_over_another_incarnation_are_the_keys_next_life() -> Result<(), ProviderKeyError> {
    type State = fn(&mut WorkerFacts);
    // What each next life states, and whether it is the worker the row
    // was: its role, its models however they are ordered, its wires.
    let lives: [(State, bool); 6] = [
        (|_| {}, true),
        (|facts| facts.models.reverse(), true),
        (|facts| facts.vision = true, true),
        (|facts| facts.role = Role::Decode, false),
        (|facts| facts.wire_modes = WireModes::GRPC, false),
        (|facts| facts.models.push("three".to_owned()), false),
    ];
    let earlier = Instant::now();
    let now = earlier + Duration::from_secs(5);
    let new = identity("url", 2);
    for stood in [
        WorkerStatus::Ready,
        WorkerStatus::NotReady,
        WorkerStatus::Draining,
    ] {
        let standing = standing(stood, earlier)?;
        for (life, (state, same)) in lives.into_iter().enumerate() {
            let mut facts = WorkerFacts {
                incarnation: 2,
                ..standing.facts.clone()
            };
            state(&mut facts);
            let (change, row) =
                WorkerRow::laid_over(facts.clone(), Some(&standing), Registrant::Control, now);
            let old = standing.id;
            assert_eq!(change, MembershipChange::Replaced { old, new });
            let carried = same && stood != WorkerStatus::Draining;
            let next = WorkerRow {
                id: new,
                facts,
                status: if carried {
                    stood
                } else {
                    WorkerStatus::Pending
                },
                revision: 1,
                admitted_at: now,
                maker: standing.maker.clone(),
            };
            assert_eq!(row, next, "life {life} over a row that is {stood:?}");
        }
    }
    Ok(())
}

/// Ahead of every arm: a blank value of a descriptor or of an encode
/// offer is no report, and a descriptor that holds none is shared with
/// the row, never copied.
#[test]
fn facts_laid_over_any_row_hold_no_blank_of_a_descriptor_or_of_an_encode_offer(
) -> Result<(), ProviderKeyError> {
    let now = Instant::now();
    let standing = standing(WorkerStatus::Ready, now)?;
    let blank = PairingDescriptor {
        protocol: Some(" nixl ".to_owned()),
        transport: Some(" ".to_owned()),
        ..PairingDescriptor::default()
    };
    let read = Arc::new(PairingDescriptor {
        protocol: Some("nixl".to_owned()),
        ..PairingDescriptor::default()
    });
    // A new key, the standing row's own incarnation, and another.
    for (standing, incarnation) in [(None, 1), (Some(&standing), 1), (Some(&standing), 2)] {
        let facts = WorkerFacts {
            incarnation,
            pairing: Some(Arc::new(blank.clone())),
            encode: EncodeOffer {
                connector: Some("  ".to_owned()),
                ..EncodeOffer::default()
            },
            ..stated()
        };
        let (_, row) = WorkerRow::laid_over(facts, standing, Registrant::Control, now);
        assert_eq!(row.facts.pairing.as_ref(), Some(&read));
        assert_eq!(row.facts.encode.connector, None);
        let facts = WorkerFacts {
            incarnation,
            pairing: Some(Arc::clone(&read)),
            ..stated()
        };
        let (_, row) = WorkerRow::laid_over(facts, standing, Registrant::Control, now);
        let held = row.facts.pairing.expect("the descriptor stated");
        assert!(Arc::ptr_eq(&held, &read), "shared, not copied");
    }
    Ok(())
}
