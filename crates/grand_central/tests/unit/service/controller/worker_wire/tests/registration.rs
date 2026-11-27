//! A registration's reader: what a spec states field by field, what it
//! leaves to the standing row, and what the door refuses by its path.

use super::*;

#[test]
fn a_bare_registration_is_one_checked_rank_under_the_stores_defaults() {
    let made = facts(&spec(), None).expect("a bare registration");
    assert_eq!(
        made,
        WorkerFacts {
            key: URL.to_owned(),
            engine: URL.to_owned(),
            role: Role::Single,
            wire_modes: WireModes::GRPC,
            dp_size: 1,
            ..WorkerFacts::default()
        }
    );
    assert!(made.health.probes);
    assert_eq!(made.reachable_from, Reachability::Any);
}

/// One rule of a row's wire and its reach, whoever states the wire: a
/// registration's row keeps the reach of the row that stands.
#[test]
fn a_host_local_wire_is_held_to_the_reach_of_the_row_that_stands() {
    let stated = proto::WorkerSpec {
        wire: proto::WireMode::Zmq as i32,
        ..spec()
    };
    let unsaid = proto::WorkerSpec { wire: 0, ..spec() };
    let standing = |from| WorkerFacts {
        wire_modes: WireModes::ZMQ,
        reachable_from: from,
        ..facts(&spec(), None).expect("a row")
    };
    let one = Reachability::Gateway(GatewayId(7));
    for ask in [&stated, &unsaid] {
        // One gateway's row takes the wire, stated or left as it stood.
        let made = facts(ask, Some(&standing(one))).expect("one gateway's row");
        assert_eq!(
            (made.wire_modes, made.reachable_from),
            (WireModes::ZMQ, one)
        );
        // A row that every gateway reaches is refused it.
        let refusal = facts(ask, Some(&standing(Reachability::Any))).expect_err("no gateway");
        assert_eq!(refusal.path, "spec.wire");
    }
}

#[test]
fn a_registration_states_its_row_field_by_field() {
    let stated = proto::WorkerSpec {
        runtime: "runtime-a".to_owned(),
        models: vec!["model-a".to_owned()],
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        zone: "zone-1".to_owned(),
        overload: Some(proto::OverloadThresholds {
            queue_limit: Some(40),
            kv_usage_permille: None,
        }),
        kv_block_size: 16,
        pairing: Some(proto::PairingDescriptor {
            bootstrap_port: 8_998,
            ..Default::default()
        }),
        dp_size: 1,
        signals: Some(proto::Signals {
            events: true,
            loads: false,
            scrape: true,
            scrape_interval_ms: 3_000,
            scrape_family: "runtime-a".to_owned(),
        }),
        health_policy: Some(proto::HealthPolicy {
            failure_threshold: 0,
            success_threshold: 6,
            check_interval_ms: 0,
            timeout_ms: 250,
            disabled: true,
        }),
        ..spec()
    };
    let made = facts(&stated, None).expect("a full registration");
    let default = HealthPolicy::default();
    assert_eq!(
        made,
        WorkerFacts {
            key: URL.to_owned(),
            engine: URL.to_owned(),
            role: Role::Single,
            wire_modes: WireModes::GRPC,
            engine_kind: "runtime-a".to_owned(),
            models: vec!["model-a".to_owned()],
            dp_size: 1,
            labels: BTreeMap::from([("tier".to_owned(), "gold".to_owned())]),
            zone: Some("zone-1".to_owned()),
            kv_block_size: Some(16),
            bootstrap_port: Some(8_998),
            signals: Signals {
                events: true,
                loads: false,
                scrape_interval_ms: 3_000,
            },
            health: HealthPolicy {
                // A zero takes the default's value.
                failure_threshold: default.failure_threshold,
                success_threshold: 6,
                check_interval_ms: default.check_interval_ms,
                timeout_ms: 250,
                probes: false,
            },
            overload: Overload {
                // An unset threshold takes the store's default field.
                kv_usage_permille: Overload::default().kv_usage_permille,
                waiting_requests: Some(40),
            },
            ..WorkerFacts::default()
        }
    );
}

#[test]
fn a_descriptor_states_the_bootstrap_port_or_leaves_it_as_it_stands() {
    let with = |bootstrap_port| proto::WorkerSpec {
        pairing: Some(proto::PairingDescriptor {
            bootstrap_port,
            ..Default::default()
        }),
        ..spec()
    };
    let port = |stated: &proto::WorkerSpec, standing: Option<&WorkerFacts>| {
        facts(stated, standing)
            .expect("a registration")
            .bootstrap_port
    };
    assert_eq!(port(&spec(), None), None);
    assert_eq!(port(&with(0), None), None);
    assert_eq!(port(&with(8_998), None), Some(8_998));
    assert_eq!(port(&with(65_535), None), Some(65_535));
    // Over a standing row: a port stated replaces it, none stated keeps
    // it, with a descriptor or without.
    let standing = facts(&with(8_998), None).expect("a registration");
    assert_eq!(port(&with(9_001), Some(&standing)), Some(9_001));
    assert_eq!(port(&with(0), Some(&standing)), Some(8_998));
    assert_eq!(port(&spec(), Some(&standing)), Some(8_998));
}

/// Number 8 of a descriptor was its pair class, and the contract
/// reserves it. A peer built before that can still state it: the
/// generated reader passes over the number, so the descriptor reads as
/// one that states none and its registration is taken.
#[test]
fn a_descriptor_that_states_the_reserved_number_reads_as_one_that_states_none() {
    let stated = proto::PairingDescriptor {
        bootstrap_port: 8_998,
        ..Default::default()
    };
    // The descriptor's own bytes, then number 8 as a varint of 7: a
    // field's key is its number over three bits of wire type, and a
    // varint's type is 0.
    let mut sent = stated.encode_to_vec();
    sent.extend([8 << 3, 7]);
    let read = proto::PairingDescriptor::decode(sent.as_slice()).expect("a descriptor");
    assert_eq!(read, stated);
    let registered = proto::WorkerSpec {
        pairing: Some(read),
        ..spec()
    };
    let made = facts(&registered, None).expect("a registration");
    assert_eq!(made.bootstrap_port, Some(8_998));
}

#[test]
fn a_threshold_the_block_leaves_unset_takes_the_stores_default() {
    // Each line reads the store's default field, set or not: the case
    // stands whatever that default is.
    let default = Overload::default();
    let made = |queue_limit, kv_usage_permille| {
        let stated = proto::WorkerSpec {
            overload: Some(proto::OverloadThresholds {
                queue_limit,
                kv_usage_permille,
            }),
            ..spec()
        };
        facts(&stated, None).expect("a registration").overload
    };
    assert_eq!(made(None, None), default);
    assert_eq!(
        made(None, Some(900)),
        Overload {
            kv_usage_permille: Some(900),
            ..default
        }
    );
    assert_eq!(
        made(Some(40), None),
        Overload {
            waiting_requests: Some(40),
            ..default
        }
    );
}

/// What a worker states for a request's hard demands is its
/// registration's block, whole. A block left out keeps what the row
/// states, and an empty one states none.
#[test]
fn a_block_of_demand_facts_is_the_rows_whole_and_one_left_out_keeps_the_rows() {
    let with = |block: proto::DemandFacts| proto::WorkerSpec {
        demand_facts: Some(block),
        ..spec()
    };
    let stated = |stated: &proto::WorkerSpec, standing: Option<&WorkerFacts>| {
        facts(stated, standing).map(|made| made.demand_facts)
    };
    let every = proto::DemandFacts {
        runtime: 7,
        references: true,
        host_pipeline: true,
        window: 32_768,
        decoding: 255,
    };
    let held = DemandFacts {
        runtime: 7,
        references: true,
        host_pipeline: true,
        window: 32_768,
        decoding: 255,
    };
    assert_eq!(stated(&spec(), None), Ok(DemandFacts::default()));
    assert_eq!(stated(&with(every), None), Ok(held));

    // Over a standing row: no block keeps its facts, a block replaces
    // them whole, and an empty block states none.
    let standing = facts(&with(every), None).expect("a registration");
    assert_eq!(stated(&spec(), Some(&standing)), Ok(held));
    let window_alone = proto::DemandFacts {
        window: 4_096,
        ..Default::default()
    };
    assert_eq!(
        stated(&with(window_alone), Some(&standing)),
        Ok(DemandFacts {
            window: 4_096,
            ..DemandFacts::default()
        })
    );
    assert_eq!(
        stated(&with(proto::DemandFacts::default()), Some(&standing)),
        Ok(DemandFacts::default())
    );

    // The store holds the way to decide a constraint in a byte.
    let wide = proto::DemandFacts {
        decoding: 256,
        ..every
    };
    assert_eq!(
        stated(&with(wide), None),
        Err(refused(
            "spec.demand_facts.decoding",
            "a way to decide a constraint is at most 255",
        ))
    );
}

/// Whether a worker takes images is its registration's where it states
/// it. A registration that states none keeps the row's, and a new row
/// takes no images.
#[test]
fn a_registration_states_whether_its_worker_takes_images_and_one_that_states_none_keeps_the_rows() {
    let with = |vision: bool| proto::WorkerSpec {
        vision: Some(vision),
        ..spec()
    };
    let stated = |stated: &proto::WorkerSpec, standing: Option<&WorkerFacts>| {
        facts(stated, standing).map(|made| made.vision)
    };
    assert_eq!(stated(&spec(), None), Ok(false));
    assert_eq!(stated(&with(true), None), Ok(true));
    assert_eq!(stated(&with(false), None), Ok(false));

    // Over a standing row: a registration that states none keeps the
    // row's, and one that states it is the row's, either way.
    let seeing = facts(&with(true), None).expect("a registration");
    assert_eq!(stated(&spec(), Some(&seeing)), Ok(true));
    assert_eq!(stated(&with(false), Some(&seeing)), Ok(false));
    let blind = facts(&spec(), None).expect("a registration");
    assert_eq!(stated(&with(true), Some(&blind)), Ok(true));
}

/// The descriptor's number 11 left the contract: whether a worker takes
/// images is the registration's own field. A registration that still
/// states the number is read as one that states none, where a build
/// before this one refused it by its path.
#[test]
fn a_descriptor_that_still_states_number_eleven_reads_as_one_that_states_none() {
    let port_alone = proto::PairingDescriptor {
        bootstrap_port: 7_000,
        ..Default::default()
    };
    let mut bytes = port_alone.encode_to_vec();
    // Field 11 as a varint: its key, then one.
    bytes.extend_from_slice(&[11 << 3, 1]);
    let read = proto::PairingDescriptor::decode(bytes.as_slice()).expect("a descriptor");
    assert_eq!(read, port_alone);
    assert_eq!(bootstrap_port(&read), Ok(Some(7_000)));
}

/// A worker's end of the encode handoff is its registration's block,
/// whole. A block left out keeps what the row states, and an empty one
/// states no end. Its connector is the store's text: no rule holds its
/// words, and the row's bound is its one bound.
#[test]
fn a_block_of_an_encode_offer_is_the_rows_whole_and_one_left_out_keeps_the_rows() {
    let with = |block: proto::EncodeOffer| proto::WorkerSpec {
        encode: Some(block),
        ..spec()
    };
    let stated = |stated: &proto::WorkerSpec, standing: Option<&WorkerFacts>| {
        facts(stated, standing).map(|made| made.encode)
    };
    let every = proto::EncodeOffer {
        handoff: proto::EncodeHandoff::Push as i32,
        producer: true,
        consumer: true,
        connector: "connector-a".to_owned(),
        references: true,
    };
    let held = EncodeOffer {
        handoff: EncodeHandoff::Push,
        producer: true,
        consumer: true,
        connector: Some("connector-a".to_owned()),
        references: true,
    };
    assert_eq!(stated(&spec(), None), Ok(EncodeOffer::default()));
    assert_eq!(stated(&with(every.clone()), None), Ok(held.clone()));

    // Over a standing row: no block keeps its offer, a block replaces it
    // whole, and an empty block states no end.
    let standing = facts(&with(every.clone()), None).expect("a registration");
    assert_eq!(stated(&spec(), Some(&standing)), Ok(held));
    let meeting = proto::EncodeOffer {
        handoff: proto::EncodeHandoff::Rendezvous as i32,
        ..Default::default()
    };
    assert_eq!(
        stated(&with(meeting), Some(&standing)),
        Ok(EncodeOffer {
            handoff: EncodeHandoff::Rendezvous,
            ..EncodeOffer::default()
        })
    );
    assert_eq!(
        stated(&with(proto::EncodeOffer::default()), Some(&standing)),
        Ok(EncodeOffer::default())
    );

    // A registration's connector is the store's text, and no rule holds its
    // words. One past a name's bound, one that holds a blank and one that
    // holds a control character are each taken as stated.
    let named = |connector: &str| proto::EncodeOffer {
        connector: connector.to_owned(),
        ..every.clone()
    };
    let read = |connector: &str| stated(&with(named(connector)), None).map(|offer| offer.connector);
    for taken in [
        text(NAME_BYTES + 1),
        "connector a".to_owned(),
        "connector\u{7}".to_owned(),
    ] {
        assert_eq!(read(&taken), Ok(Some(taken.clone())));
    }
    // It is read as the store holds it: past the blanks around it, and a
    // blank one states none. So a registration that states the row's offer
    // with blanks around its connector states the row's own offer.
    assert_eq!(read("  connector a\t"), Ok(Some("connector a".to_owned())));
    assert_eq!(read(" \t "), Ok(None));
    assert_eq!(
        stated(&with(named(" connector-a ")), Some(&standing)),
        Ok(standing.encode.clone())
    );
    // The row's bound is the connector's one bound: one that carries the
    // row past it is refused by the row's path.
    assert_eq!(
        read(&text(ROW_BYTES)).map_err(|refused| refused.path),
        Err("spec")
    );
    // A form this build does not know is refused by its path.
    let unknown = proto::EncodeOffer {
        handoff: 3,
        ..every.clone()
    };
    let Refused { path, why } = stated(&with(unknown), None).expect_err("the form is refused");
    assert_eq!(path, "spec.encode.handoff");
    assert!(!why.is_empty());
}

/// The descriptor's five numbers of an encode offer left the contract: the
/// offer is the registration's own block. A registration that still states
/// one of them is read as one that states none, where a build before this
/// one refused it by its path.
#[test]
fn a_descriptor_that_still_states_a_number_of_the_encode_offer_reads_as_one_that_states_none() {
    let port_alone = proto::PairingDescriptor {
        bootstrap_port: 7_000,
        ..Default::default()
    };
    let mut bytes = port_alone.encode_to_vec();
    // Four of the five as a varint of one, then the connector as a text of
    // one letter: each field's key, then its value.
    for number in [9_u8, 10, 12, 14] {
        bytes.extend_from_slice(&[number << 3, 1]);
    }
    bytes.extend_from_slice(&[(13 << 3) | 2, 1, b'a']);
    let read = proto::PairingDescriptor::decode(bytes.as_slice()).expect("a descriptor");
    assert_eq!(read, port_alone);
    let registered = proto::WorkerSpec {
        pairing: Some(read),
        ..spec()
    };
    let made = facts(&registered, None).expect("a registration");
    assert_eq!(made.bootstrap_port, Some(7_000));
    assert_eq!(made.encode, EncodeOffer::default());
}

/// A worker's priority and its drain window are its registration's where
/// it states them. A registration that states neither keeps the row's, and
/// a new row stands at priority zero and waits the default window.
#[test]
fn a_registration_states_a_priority_and_a_drain_window_and_one_that_states_none_keeps_the_rows() {
    let with = |priority: Option<i32>, window: Option<u32>| proto::WorkerSpec {
        priority,
        drain_settle_secs: window,
        ..spec()
    };
    let stated = |stated: &proto::WorkerSpec, standing: Option<&WorkerFacts>| {
        facts(stated, standing).map(|made| (made.priority, made.drain_settle_secs))
    };
    assert_eq!(stated(&spec(), None), Ok((0, None)));
    assert_eq!(stated(&with(Some(-5), None), None), Ok((-5, None)));
    // A window of no wait is a window, and not the default one.
    assert_eq!(stated(&with(None, Some(0)), None), Ok((0, Some(0))));
    assert_eq!(stated(&with(Some(7), Some(45)), None), Ok((7, Some(45))));

    // Over a standing row: a registration that states neither keeps the
    // row's, and one that states either is the row's for that fact alone.
    let standing = facts(&with(Some(7), Some(45)), None).expect("a registration");
    assert_eq!(stated(&spec(), Some(&standing)), Ok((7, Some(45))));
    let zero = with(Some(0), None);
    assert_eq!(stated(&zero, Some(&standing)), Ok((0, Some(45))));
    let no_wait = with(None, Some(0));
    assert_eq!(stated(&no_wait, Some(&standing)), Ok((7, Some(0))));
}

#[test]
fn a_registration_over_a_standing_row_keeps_what_it_leaves_unsaid_save_its_labels_and_its_policy() {
    let standing = full().facts;
    assert!(!standing.health.probes, "the standing row is unchecked");
    let made = facts(&spec(), Some(&standing)).expect("a bare registration");
    assert_eq!(
        made,
        WorkerFacts {
            // What every registration states.
            role: Role::Single,
            wire_modes: WireModes::GRPC,
            // The two facts that are the registration's, whole: it
            // states no label, and no policy, so it is checked under
            // the defaults.
            labels: BTreeMap::new(),
            health: HealthPolicy::default(),
            ..standing
        }
    );
    assert!(made.health.probes);
    // The wire and the role are kept too where a registration names
    // neither, and a number this build does not know is not "unsaid".
    let standing = WorkerFacts {
        wire_modes: WireModes::HTTP,
        ..full().facts
    };
    let unsaid = proto::WorkerSpec {
        wire: 0,
        role: 0,
        ..spec()
    };
    let kept = facts(&unsaid, Some(&standing)).expect("a registration that names neither");
    assert_eq!(
        (kept.role, kept.wire_modes),
        (Role::Decode, WireModes::HTTP)
    );
    for (stated, path) in [
        (proto::WorkerSpec { wire: 3, ..spec() }, "spec.wire"),
        (proto::WorkerSpec { role: 5, ..spec() }, "spec.role"),
    ] {
        assert_eq!(
            facts(&stated, Some(&standing)).map_err(|refused| refused.path),
            Err(path)
        );
    }
}

#[test]
fn a_field_the_store_cannot_hold_is_refused_by_its_path() {
    let signals = |scrape, scrape_interval_ms, family: &str| {
        Some(proto::Signals {
            scrape,
            scrape_interval_ms,
            scrape_family: family.to_owned(),
            ..Default::default()
        })
    };
    let cases: [(proto::WorkerSpec, &str); 18] = [
        (
            proto::WorkerSpec {
                url: String::new(),
                ..spec()
            },
            "spec.url",
        ),
        (
            proto::WorkerSpec {
                url: "grpc://10.0.0.5:8000 ".to_owned(),
                ..spec()
            },
            "spec.url",
        ),
        (
            proto::WorkerSpec {
                url: "grpc://10.0.0.5:8000\u{7}".to_owned(),
                ..spec()
            },
            "spec.url",
        ),
        (
            proto::WorkerSpec {
                runtime: "runtime a".to_owned(),
                ..spec()
            },
            "spec.runtime",
        ),
        (
            proto::WorkerSpec {
                runtime: "runtime-a\n".to_owned(),
                ..spec()
            },
            "spec.runtime",
        ),
        (
            proto::WorkerSpec {
                models: vec!["model-a".to_owned(), " model-b".to_owned()],
                ..spec()
            },
            "spec.models",
        ),
        (
            proto::WorkerSpec {
                models: vec![String::new()],
                ..spec()
            },
            "spec.models",
        ),
        (proto::WorkerSpec { wire: 0, ..spec() }, "spec.wire"),
        (proto::WorkerSpec { wire: 3, ..spec() }, "spec.wire"),
        // A host-local wire: no registration names the gateway that
        // holds its socket.
        (
            proto::WorkerSpec {
                wire: proto::WireMode::Zmq as i32,
                ..spec()
            },
            "spec.wire",
        ),
        (proto::WorkerSpec { role: 0, ..spec() }, "spec.role"),
        (proto::WorkerSpec { role: 5, ..spec() }, "spec.role"),
        (
            proto::WorkerSpec {
                dp_size: 2,
                ..spec()
            },
            "spec.dp_size",
        ),
        (
            proto::WorkerSpec {
                topology: vec!["rack-1".to_owned()],
                ..spec()
            },
            "spec.topology",
        ),
        (
            proto::WorkerSpec {
                overload: Some(proto::OverloadThresholds {
                    queue_limit: None,
                    kv_usage_permille: Some(1_001),
                }),
                ..spec()
            },
            "spec.overload.kv_usage_permille",
        ),
        (
            proto::WorkerSpec {
                signals: signals(true, 0, ""),
                ..spec()
            },
            "spec.signals.scrape_interval_ms",
        ),
        (
            proto::WorkerSpec {
                signals: signals(false, 500, ""),
                ..spec()
            },
            "spec.signals.scrape",
        ),
        (
            proto::WorkerSpec {
                runtime: "runtime-a".to_owned(),
                signals: signals(true, 500, "runtime-b"),
                ..spec()
            },
            "spec.signals.scrape_family",
        ),
    ];
    for (stated, path) in cases {
        let Refused { path: found, why } =
            facts(&stated, None).expect_err("the registration is refused");
        assert_eq!(found, path, "{stated:?}");
        assert!(!why.is_empty());
    }
    // Of a pairing descriptor the store holds the bootstrap port alone:
    // every other field a caller states is refused by its own path.
    let named = |name: &str| name.to_owned();
    let descriptors: [(proto::PairingDescriptor, &str); 7] = [
        (
            proto::PairingDescriptor {
                protocol: named("protocol-a"),
                ..Default::default()
            },
            "spec.pairing.protocol",
        ),
        (
            proto::PairingDescriptor {
                connector: named("connector-a"),
                ..Default::default()
            },
            "spec.pairing.connector",
        ),
        (
            proto::PairingDescriptor {
                transport: named("transport-a"),
                ..Default::default()
            },
            "spec.pairing.transport",
        ),
        (
            proto::PairingDescriptor {
                version: named("1"),
                ..Default::default()
            },
            "spec.pairing.version",
        ),
        (
            proto::PairingDescriptor {
                kv_layout: [(named("layers"), named("32"))].into(),
                ..Default::default()
            },
            "spec.pairing.kv_layout",
        ),
        (
            proto::PairingDescriptor {
                bootstrap_host: named("10.0.0.5"),
                ..Default::default()
            },
            "spec.pairing.bootstrap_host",
        ),
        (
            proto::PairingDescriptor {
                bootstrap_port: 65_536,
                ..Default::default()
            },
            "spec.pairing.bootstrap_port",
        ),
    ];
    for (pairing, path) in descriptors {
        let stated = proto::WorkerSpec {
            pairing: Some(pairing),
            ..spec()
        };
        let Refused { path: found, why } =
            facts(&stated, None).expect_err("the descriptor is refused");
        assert_eq!(found, path, "{stated:?}");
        assert!(!why.is_empty());
    }
    // The two zeroes of a width both read one rank.
    for dp_size in [0, 1] {
        let made = facts(&proto::WorkerSpec { dp_size, ..spec() }, None);
        assert_eq!(made.map(|facts| facts.dp_size), Ok(1));
    }
    assert_eq!(
        refused("a", "b"),
        Refused {
            path: "a",
            why: "b"
        }
    );
}

#[test]
fn a_field_at_its_bound_is_taken_and_one_over_is_refused_by_its_path() {
    type Stating = fn(usize) -> proto::WorkerSpec;
    let cases: [(&str, usize, &str, Stating); 7] = [
        (
            "spec.url",
            URL_BYTES,
            "a URL is at most 2048 bytes",
            |bytes| proto::WorkerSpec {
                url: text(bytes),
                ..spec()
            },
        ),
        (
            "spec.runtime",
            NAME_BYTES,
            "a runtime is at most 256 bytes",
            |bytes| proto::WorkerSpec {
                runtime: text(bytes),
                ..spec()
            },
        ),
        (
            "spec.zone",
            NAME_BYTES,
            "a zone is at most 256 bytes",
            |bytes| proto::WorkerSpec {
                zone: text(bytes),
                ..spec()
            },
        ),
        (
            "spec.models",
            NAME_BYTES,
            "a model id is at most 256 bytes",
            |bytes| proto::WorkerSpec {
                models: vec!["m".to_owned(), text(bytes)],
                ..spec()
            },
        ),
        (
            "spec.labels",
            KEY_BYTES,
            "a label's key is at most 128 bytes",
            |bytes| proto::WorkerSpec {
                labels: [(text(bytes), String::new())].into(),
                ..spec()
            },
        ),
        (
            "spec.labels",
            VALUE_BYTES,
            "a label's value is at most 4096 bytes",
            |bytes| proto::WorkerSpec {
                labels: [("k".to_owned(), text(bytes))].into(),
                ..spec()
            },
        ),
        (
            "spec.labels",
            LABELS,
            "a row holds at most 128 labels",
            |count| {
                let labels = (0..count).map(|n| (n.to_string(), String::new()));
                proto::WorkerSpec {
                    labels: labels.collect(),
                    ..spec()
                }
            },
        ),
    ];
    for (path, bound, why, stating) in cases {
        assert!(facts(&stating(bound), None).is_ok(), "{path} at {bound}");
        assert_eq!(
            facts(&stating(bound + 1), None).map(|_| ()),
            Err(refused(path, why)),
            "{path} past {bound}"
        );
        assert!(why.contains(&bound.to_string()), "{why}");
    }
    let long_id = proto::WorkerSpec {
        models: vec![text(NAME_BYTES + 1); 129],
        ..spec()
    };
    assert_eq!(
        facts(&long_id, None).map(|_| ()),
        Err(refused("spec.models", "a model id is at most 256 bytes"))
    );
}

#[test]
fn a_registration_limits_model_ids_by_the_wires_row_weight() {
    let short = proto::WorkerSpec {
        models: (0..129).map(|n| n.to_string()).collect(),
        ..spec()
    };
    assert_eq!(
        facts(&short, None).expect("the ids fit").models,
        short.models
    );
    let heavy = proto::WorkerSpec {
        models: vec![text(256); 64],
        ..spec()
    };
    assert_eq!(
        facts(&heavy, None),
        Err(refused("spec", "a row is at most 16384 bytes on the wire"))
    );
}

#[test]
fn a_url_past_its_rules_is_refused_before_its_row_is_looked_up() {
    let looked = std::cell::Cell::new(0);
    let lookup = |_: &str| {
        looked.set(looked.get() + 1);
        None
    };
    let named = |url: String| proto::WorkerSpec { url, ..spec() };
    let refusals = [
        (String::new(), "a registration names its worker's URL"),
        (text(URL_BYTES + 1), "a URL is at most 2048 bytes"),
        (
            "grpc://10.0.0.5 :8000".to_owned(),
            "a URL holds no blank and no control character",
        ),
    ];
    for (url, why) in refusals {
        assert_eq!(
            registration(&named(url), lookup).map(|_| ()),
            Err(refused("spec.url", why))
        );
    }
    assert_eq!(looked.get(), 0, "a refused URL is never looked up");
    // A URL at its bound is looked up, once.
    assert!(registration(&named(text(URL_BYTES)), lookup).is_ok());
    assert_eq!(looked.get(), 1);
}
