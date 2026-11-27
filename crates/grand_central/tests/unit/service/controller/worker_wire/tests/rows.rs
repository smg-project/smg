//! The row on the wire: what its writer tells of a store's row, and what
//! its reader holds, reads past or refuses.

use super::*;

#[test]
fn the_wires_row_tells_every_fact_of_the_stores_row() {
    let stored = full();
    let told = row(&stored);
    let expected = proto::WorkerRow {
        id: stored.id.0,
        registration_key: URL.to_owned(),
        incarnation: 3,
        url: URL.to_owned(),
        wire: proto::WireMode::Grpc as i32,
        role: proto::Role::Decode as i32,
        runtime: "runtime-a".to_owned(),
        models: vec!["model-a".to_owned(), "model-b".to_owned()],
        dp_size: 4,
        labels: [("policy", "least_load"), ("tier", "gold")]
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value.to_owned()))
            .collect(),
        zone: "zone-1".to_owned(),
        status: proto::WorkerStatus::Draining as i32,
        revision: 11,
        health: proto::HealthFlag::Draining as u32,
        reachable: Some(proto::Reachability {
            scope: Some(proto::reachability::Scope::GatewayId(
                "0000000000000abc".to_owned(),
            )),
        }),
        pairing: Some(proto::PairingDescriptor {
            bootstrap_port: 8_998,
            ..Default::default()
        }),
        admitted_age_ms: 0,
        maker: None,
        overload: Some(proto::OverloadThresholds {
            queue_limit: None,
            kv_usage_permille: Some(875),
        }),
        signals: Some(proto::Signals {
            events: true,
            loads: true,
            scrape: true,
            scrape_interval_ms: 2_500,
            scrape_family: "runtime-a".to_owned(),
        }),
        kv_block_size: 16,
        policy: "least_load".to_owned(),
        topology: Vec::new(),
        max_running: 96,
        dp_rank: 2,
        engine: "grpc://10.0.0.5".to_owned(),
        health_policy: Some(proto::HealthPolicy {
            failure_threshold: 5,
            success_threshold: 4,
            check_interval_ms: 7_000,
            timeout_ms: 900,
            disabled: true,
        }),
        demand_facts: Some(proto::DemandFacts {
            runtime: 7,
            references: true,
            host_pipeline: false,
            window: 32_768,
            decoding: 3,
        }),
        vision: true,
        encode: Some(proto::EncodeOffer {
            handoff: proto::EncodeHandoff::Push as i32,
            producer: true,
            consumer: false,
            connector: "connector-a".to_owned(),
            references: true,
        }),
        priority: -5,
        drain_settle_secs: Some(45),
    };
    assert_eq!(told, expected);
}

#[test]
fn a_row_of_defaults_reads_any_gateway_and_no_scrape() {
    let stored = WorkerRow {
        id: identity(URL, 1),
        facts: WorkerFacts {
            key: URL.to_owned(),
            wire_modes: WireModes::HTTP.union(WireModes::GRPC),
            ..WorkerFacts::default()
        },
        status: WorkerStatus::Ready,
        revision: 1,
        admitted_at: Instant::now(),
        maker: Default::default(),
    };
    let told = row(&stored);
    // Two wires read as the sum of their numbers.
    assert_eq!(told.wire, 3);
    assert_eq!(told.health, proto::HealthFlag::Healthy as u32);
    assert_eq!(
        told.reachable,
        Some(proto::Reachability {
            scope: Some(proto::reachability::Scope::Any(true)),
        })
    );
    let signals = told.signals.expect("the row's signals");
    assert!(!signals.scrape && signals.scrape_family.is_empty());
    let default = Overload::default();
    assert_eq!(
        told.overload,
        Some(proto::OverloadThresholds {
            queue_limit: default.waiting_requests,
            kv_usage_permille: default.kv_usage_permille.map(u32::from),
        })
    );
    assert_eq!(
        told.health_policy.map(|policy| policy.disabled),
        Some(false)
    );
    // A worker that states nothing for a request's hard demands tells
    // no block: its row is the row a build before the block told.
    assert_eq!(told.demand_facts, None);
    // Neither does one whose encode offer states nothing.
    assert_eq!(told.encode, None);
}

#[test]
fn every_status_has_its_own_word_on_the_wire() {
    let told = [
        WorkerStatus::Pending,
        WorkerStatus::Ready,
        WorkerStatus::NotReady,
        WorkerStatus::Failed,
        WorkerStatus::Draining,
    ]
    .map(status);
    assert_eq!(
        told,
        [
            proto::WorkerStatus::Pending,
            proto::WorkerStatus::Ready,
            proto::WorkerStatus::NotReady,
            proto::WorkerStatus::Failed,
            proto::WorkerStatus::Draining,
        ]
    );
    let change = status_changed(identity(URL, 1), WorkerStatus::Pending, WorkerStatus::Ready);
    assert_eq!(
        event(change, 12, 9, 4),
        proto::WorkerEvent {
            seq: 12,
            membership_epoch: 9,
            event: Some(Event::Status(proto::StatusChanged {
                worker_id: identity(URL, 1).0,
                old: proto::WorkerStatus::Pending as i32,
                new: proto::WorkerStatus::Ready as i32,
                health: proto::HealthFlag::Healthy as u32,
            })),
            term: 4,
        }
    );
}

/// The wire's row tells whether its worker takes images, and its reader
/// holds it. A row of a worker that takes none tells no byte of the
/// field, so it is the row a build before the field told. Such a row
/// reads as taking none.
#[test]
fn a_row_tells_whether_its_worker_takes_images_and_one_that_tells_none_reads_as_taking_none() {
    let seeing = full();
    let blind = WorkerRow {
        facts: WorkerFacts {
            vision: false,
            ..full().facts
        },
        ..full()
    };
    let (told_seeing, told_blind) = (row(&seeing), row(&blind));
    assert!(told_seeing.vision);
    assert!(!told_blind.vision);
    let read = |told: &proto::WorkerRow| told_row(told, Instant::now()).map(|row| row.facts.vision);
    assert_eq!(read(&told_seeing), Ok(true));
    assert_eq!(read(&told_blind), Ok(false));
    // The bit is three bytes of the row that tells it, and none of the
    // row that does not.
    assert_eq!(told_seeing.encoded_len() - told_blind.encoded_len(), 3);
    let bytes = told_blind.encode_to_vec();
    let again = proto::WorkerRow::decode(bytes.as_slice()).expect("a row");
    assert_eq!(read(&again), Ok(false));
}

/// The wire's row tells its worker's end of the encode handoff, and its
/// reader holds it. A row of a worker whose offer states nothing tells no
/// byte of the block, so it is the row a build before the block told. Such
/// a row reads as one that states no end. An offer that names no form and
/// states something tells its block. A told row's connector is read as
/// told.
#[test]
fn a_row_tells_its_workers_encode_offer_and_one_that_tells_none_reads_as_stating_no_end() {
    let offering = full();
    let silent = WorkerRow {
        facts: WorkerFacts {
            encode: EncodeOffer::default(),
            ..full().facts
        },
        ..full()
    };
    let (told_offering, told_silent) = (row(&offering), row(&silent));
    assert_eq!(told_silent.encode, None);
    let read = |told: &proto::WorkerRow| told_row(told, Instant::now()).map(|row| row.facts.encode);
    assert_eq!(read(&told_offering), Ok(offering.facts.encode.clone()));
    assert_eq!(read(&told_silent), Ok(EncodeOffer::default()));
    let bytes = told_silent.encode_to_vec();
    let again = proto::WorkerRow::decode(bytes.as_slice()).expect("a row");
    assert_eq!(read(&again), Ok(EncodeOffer::default()));

    // The wire's weight counts the block as the wire frames it: with no
    // connector, and at each side of the second byte of the block's own
    // length and of the connector's.
    let told = |facts: &WorkerFacts| {
        let stored = WorkerRow {
            facts: facts.clone(),
            ..full()
        };
        row(&stored).encoded_len()
    };
    for connector in [0, 1, 119, 120, 127, 128, NAME_BYTES] {
        let with = WorkerFacts {
            encode: EncodeOffer {
                connector: (connector > 0).then(|| text(connector)),
                ..offering.facts.encode.clone()
            },
            ..offering.facts.clone()
        };
        assert_eq!(
            weight(&with) - weight(&silent.facts),
            told(&with) - told(&silent.facts),
            "a connector of {connector} bytes"
        );
    }

    // The writer tells each form by its own number, and the reader holds it:
    // a rendezvous comes back a rendezvous. An offer that names no form and
    // states references is no default one, so its block is told too.
    let offers = [
        EncodeOffer {
            handoff: EncodeHandoff::Rendezvous,
            ..EncodeOffer::default()
        },
        EncodeOffer {
            references: true,
            ..EncodeOffer::default()
        },
    ];
    for (offer, form) in offers.into_iter().zip([
        proto::EncodeHandoff::Rendezvous,
        proto::EncodeHandoff::Unspecified,
    ]) {
        let stored = WorkerRow {
            facts: WorkerFacts {
                encode: offer.clone(),
                ..full().facts
            },
            ..full()
        };
        let told = row(&stored);
        let block = told.encode.as_ref().expect("the block is told");
        assert_eq!(block.handoff, form as i32);
        assert_eq!(read(&told), Ok(offer));
    }

    // A told row's connector is read as told. One that holds a blank and
    // one past a registration's bound are texts a host's own write may
    // state, and a standby holds the row its leader took. A form this build
    // does not know is refused by the row's path.
    let told_as = |change: fn(&mut proto::EncodeOffer)| {
        let mut told = row(&offering);
        change(told.encode.as_mut().expect("the block"));
        told_row(&told, Instant::now())
            .map(|row| row.facts.encode.connector)
            .map_err(|refused| refused.path)
    };
    assert_eq!(
        told_as(|block| block.connector = "a b".to_owned()),
        Ok(Some("a b".to_owned()))
    );
    assert_eq!(
        told_as(|block| block.connector = text(NAME_BYTES + 1)),
        Ok(Some(text(NAME_BYTES + 1)))
    );
    assert_eq!(
        told_as(|block| block.handoff = 3),
        Err("row.encode.handoff")
    );
}

/// The wire's row tells its worker's priority and drain window, and its
/// reader holds them. A row of a worker that states neither tells no byte
/// of the two fields, so it is the row a build before the fields told.
/// Such a row reads at priority zero and the default window.
#[test]
fn a_row_tells_its_workers_priority_and_drain_window_and_one_that_tells_none_reads_the_defaults() {
    let stating = full();
    let plain = WorkerRow {
        facts: WorkerFacts {
            priority: 0,
            drain_settle_secs: None,
            ..full().facts
        },
        ..full()
    };
    let (told_stating, told_plain) = (row(&stating), row(&plain));
    let told = |told: &proto::WorkerRow| (told.priority, told.drain_settle_secs);
    assert_eq!(told(&told_stating), (-5, Some(45)));
    assert_eq!(told(&told_plain), (0, None));
    let read = |told: &proto::WorkerRow| {
        let read = told_row(told, Instant::now());
        read.map(|row| (row.facts.priority, row.facts.drain_settle_secs))
    };
    assert_eq!(read(&told_stating), Ok((-5, Some(45))));
    assert_eq!(read(&told_plain), Ok((0, None)));
    // Each number is three bytes of the row that tells it, and none of the
    // row that does not.
    assert_eq!(told_stating.encoded_len() - told_plain.encoded_len(), 6);
    let bytes = told_plain.encode_to_vec();
    let again = proto::WorkerRow::decode(bytes.as_slice()).expect("a row");
    assert_eq!(read(&again), Ok((0, None)));
    // A window of no wait crosses as a window.
    let no_wait = WorkerRow {
        facts: WorkerFacts {
            drain_settle_secs: Some(0),
            ..full().facts
        },
        ..full()
    };
    assert_eq!(read(&row(&no_wait)), Ok((-5, Some(0))));
}

/// A watcher builds its worker's object from the row it was told, and the
/// object waits the window that row tells. A row that tells none builds an
/// object of the default window.
#[test]
fn an_object_built_from_a_told_row_waits_the_window_the_row_tells() {
    use smg_worker::worker::HealthCheckConfig;

    let window = |stored: &WorkerRow| {
        let told = told_row(&row(stored), Instant::now()).expect("a row");
        let object = crate::worker_host::object(&told);
        object.metadata().health_config.drain_settle_secs
    };
    assert_eq!(window(&full()), 45);
    let plain = WorkerRow {
        facts: WorkerFacts {
            drain_settle_secs: None,
            ..full().facts
        },
        ..full()
    };
    assert_eq!(
        window(&plain),
        HealthCheckConfig::default().drain_settle_secs
    );
}

#[test]
fn a_row_crosses_with_its_maker() {
    use super::super::{maker_weight, room};
    use crate::store::{ProviderKeyError, Registrant};

    let now = Instant::now();
    // A control call's row says no maker, and a row that says none
    // reads as a control call's.
    let by_hand = full();
    let told = row(&by_hand);
    assert_eq!(told.maker, None);
    let back = told_row(&told, now).expect("the row");
    assert_eq!(back.maker, Registrant::Control);
    // A provider's row crosses whole: its name and its key for the
    // worker.
    let found = WorkerRow {
        maker: Registrant::provider("cluster", "pods/a").expect("a whole maker"),
        ..full()
    };
    let told = row(&found);
    let shown = proto::Maker {
        provider: "cluster".to_owned(),
        key: "pods/a".to_owned(),
    };
    assert_eq!(told.maker, Some(shown));
    let back = told_row(&told, now).expect("the row");
    assert_eq!(back.maker, found.maker);
    // The maker's bytes are the wire's, at a key's length on each
    // side of a length's second byte, and a row holds them inside its
    // bound.
    for bytes in [1, 110, 127, 128, 300] {
        let maker = Registrant::provider("cluster", "k".repeat(bytes)).expect("a whole maker");
        let with = WorkerRow {
            maker: maker.clone(),
            ..full()
        };
        let grown = row(&with).encoded_len() - row(&by_hand).encoded_len();
        assert_eq!(maker_weight(&maker), grown, "a key of {bytes}");
    }
    assert_eq!(maker_weight(&Registrant::Control), 0);
    let brim = facts(&weighing(ROW_BYTES), None).expect("a row at its bound");
    assert!(room(&brim, &brim.labels, &Registrant::Control, "spec").is_ok());
    assert_eq!(
        room(&brim, &brim.labels, &found.maker, "spec").map_err(|refused| refused.path),
        Err("spec")
    );
    let needs = maker_weight(&found.maker);
    let under = facts(&weighing(ROW_BYTES - needs), None).expect("a row under its bound");
    assert!(room(&under, &under.labels, &found.maker, "spec").is_ok());
    // One byte past the bound is refused, by the wire's own weight.
    let over = facts(&weighing(ROW_BYTES - needs + 1), None).expect("a row a byte past");
    assert_eq!(
        room(&over, &over.labels, &found.maker, "spec").map_err(|refused| refused.path),
        Err("spec")
    );
    let widest = WorkerRow {
        facts: under,
        ..found.clone()
    };
    assert!(row(&widest).encoded_len() <= ROW_BYTES);
    // A maker that breaks a part's rule is refused by that part's path,
    // in the words of the routine that makes one: no leader makes it.
    let wide = "p".repeat(Registrant::NAME_BYTES + 1);
    let long = "k".repeat(Registrant::KEY_BYTES + 1);
    for (provider, key, path) in [
        ("", "pods/a", "row.maker.provider"),
        ("cluster", "", "row.maker.key"),
        ("", "", "row.maker.provider"),
        (wide.as_str(), "pods/a", "row.maker.provider"),
        ("a cluster", "pods/a", "row.maker.provider"),
        ("a,b", "pods/a", "row.maker.provider"),
        ("\u{e9}", "pods/a", "row.maker.provider"),
        ("cluster", long.as_str(), "row.maker.key"),
    ] {
        let broken = proto::WorkerRow {
            maker: Some(proto::Maker {
                provider: provider.to_owned(),
                key: key.to_owned(),
            }),
            ..told.clone()
        };
        let refusal = told_row(&broken, now).expect_err("refused");
        let (ProviderKeyError::Provider(why) | ProviderKeyError::Key(why)) =
            Registrant::provider(provider, key).expect_err("no maker");
        assert_eq!((refusal.path, refusal.why), (path, why));
    }
}

#[test]
fn the_stores_row_comes_back_from_the_wire_fact_by_fact() {
    let stored = full();
    let now = stored.admitted_at + Duration::from_millis(1_500);
    let sent = proto::WorkerRow {
        admitted_age_ms: 1_500,
        ..row(&stored)
    };
    assert_eq!(told_row(&sent, now), Ok(stored));
    // A row of the store's defaults comes back as it is.
    let plain = WorkerRow {
        id: identity(URL, 1),
        facts: WorkerFacts {
            key: URL.to_owned(),
            ..WorkerFacts::default()
        },
        status: WorkerStatus::Pending,
        revision: 1,
        admitted_at: now,
        maker: Default::default(),
    };
    assert_eq!(told_row(&row(&plain), now), Ok(plain.clone()));
    // Each status and each role comes back as itself.
    for status in [
        WorkerStatus::Pending,
        WorkerStatus::Ready,
        WorkerStatus::NotReady,
        WorkerStatus::Failed,
        WorkerStatus::Draining,
    ] {
        let each = WorkerRow {
            status,
            ..plain.clone()
        };
        assert_eq!(told_row(&row(&each), now), Ok(each));
    }
    for role in [Role::Single, Role::Prefill, Role::Decode, Role::Encode] {
        let mut each = plain.clone();
        each.facts.role = role;
        assert_eq!(told_row(&row(&each), now), Ok(each));
    }
}

#[test]
fn a_row_this_build_cannot_hold_is_refused_by_its_path() {
    type Spoil = fn(&mut proto::WorkerRow);
    let sent = row(&full());
    let now = Instant::now();
    let spoiled: [(&str, Spoil); 16] = [
        ("row.status", |row| row.status = 99),
        ("row.status", |row| row.status = 0),
        ("row.role", |row| row.role = 0),
        ("row.wire", |row| row.wire = 8),
        ("row.wire", |row| row.wire = -1),
        ("row.reachable", |row| row.reachable = None),
        ("row.reachable", |row| {
            let scope = proto::reachability::Scope::Any(false);
            row.reachable = Some(proto::Reachability { scope: Some(scope) });
        }),
        ("row.reachable", |row| {
            let scope = proto::reachability::Scope::GatewayId("no number".to_owned());
            row.reachable = Some(proto::Reachability { scope: Some(scope) });
        }),
        ("row.overload", |row| row.overload = None),
        ("row.signals", |row| row.signals = None),
        ("row.health_policy", |row| row.health_policy = None),
        ("row.dp_rank", |row| row.dp_rank = 65_536),
        ("row.dp_size", |row| row.dp_size = 65_536),
        // A descriptor that tells more than the bootstrap port.
        ("row.pairing", |row| {
            row.pairing = Some(proto::PairingDescriptor {
                bootstrap_port: 8_998,
                version: "1".to_owned(),
                ..Default::default()
            });
        }),
        ("row.pairing.bootstrap_port", |row| {
            row.pairing = Some(proto::PairingDescriptor {
                bootstrap_port: 65_536,
                ..Default::default()
            });
        }),
        // A way to decide a constraint past the store's byte.
        ("row.demand_facts.decoding", |row| {
            row.demand_facts = Some(proto::DemandFacts {
                decoding: 256,
                ..Default::default()
            });
        }),
    ];
    for (path, spoil) in spoiled {
        let mut row = sent.clone();
        spoil(&mut row);
        let refused = told_row(&row, now).expect_err(path);
        assert_eq!(refused.path, path);
    }
    let mut wide = sent;
    wide.overload = Some(proto::OverloadThresholds {
        queue_limit: None,
        kv_usage_permille: Some(65_536),
    });
    let refused = told_row(&wide, now).expect_err("a permille past the width");
    assert_eq!(refused.path, "row.overload.kv_usage_permille");
}

/// Number 8 of a descriptor was its pair class, and the contract
/// reserves it. A leader built before that can still state it in a row
/// it tells: the generated reader passes over the number, so the row
/// reads as the row of its port alone, and a standby or a watcher takes
/// it.
#[test]
fn a_told_row_that_states_the_reserved_number_reads_as_the_row_of_its_port_alone() {
    let stored = full();
    let now = stored.admitted_at + Duration::from_millis(1_500);
    let mut sent = proto::WorkerRow {
        admitted_age_ms: 1_500,
        ..row(&stored)
    };
    // The told descriptor's own bytes, then number 8 as a varint of 7.
    let mut stated = sent
        .pairing
        .take()
        .expect("a row with a port tells a descriptor")
        .encode_to_vec();
    stated.extend([8 << 3, 7]);
    let read = proto::PairingDescriptor::decode(stated.as_slice()).expect("a descriptor");
    sent.pairing = Some(read);
    assert_eq!(told_row(&sent, now), Ok(stored));
}

#[test]
fn a_parts_load_rows_are_read_past() {
    let at = Instant::now();
    // A row no build reads: its source names no ingress. An older
    // leader's part carries load rows, and this build keeps none.
    let sent = proto::LoadSample {
        worker_id: 7,
        dp_rank: 2,
        load: Some(proto::LoadRow {
            running: 3,
            source: 99,
            ..proto::LoadRow::default()
        }),
        ..proto::LoadSample::default()
    };
    let part = proto::ReplicationSnapshot {
        workers: vec![row(&full())],
        loads: vec![sent],
        last: true,
        ..proto::ReplicationSnapshot::default()
    };
    let mut rows = Vec::new();
    told_part(&part, at, &mut rows).expect("the load rows are read past");
    assert_eq!(rows.len(), 1, "the part's rows are read");
}
