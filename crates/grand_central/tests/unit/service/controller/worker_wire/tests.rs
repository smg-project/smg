use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use prost::Message as _;

use super::{
    bootstrap_port, event, member, page, patch, refused, registration, removed, replaced,
    replication, row, start_again, status, status_changed, told_change, told_part, told_row,
    Refused, KEY_BYTES, LABELS, NAME_BYTES, ROW_BYTES, URL_BYTES, VALUE_BYTES,
};
use crate::{
    engine::{
        view::{Role, WireModes, WorkerId},
        DemandFacts,
    },
    service::{
        proto::{self, replication_event::Event as Sent, worker_event::Event},
        replication::{Entry, Event as Written},
    },
    store::{
        identity, EncodeHandoff, EncodeOffer, GatewayId, HealthPolicy, MembershipChange, Overload,
        Position, Reachability, Registrant, Signals, WorkerFacts, WorkerRow, WorkerStatus,
    },
};

const URL: &str = "grpc://10.0.0.5:8000";

fn spec() -> proto::WorkerSpec {
    proto::WorkerSpec {
        url: URL.to_owned(),
        wire: proto::WireMode::Grpc as i32,
        role: proto::Role::Single as i32,
        ..Default::default()
    }
}

/// What the row of these facts weighs under its own labels.
fn weight(facts: &WorkerFacts) -> usize {
    super::weight(facts, &facts.labels)
}

/// The facts of a registration over a row of `standing`, through the
/// door's own entry.
fn facts(spec: &proto::WorkerSpec, standing: Option<&WorkerFacts>) -> Result<WorkerFacts, Refused> {
    let row = standing.map(|facts| WorkerRow {
        facts: facts.clone(),
        ..full()
    });
    registration(spec, |_| row).map(|(facts, _)| facts)
}

/// A name of `bytes` bytes.
fn text(bytes: usize) -> String {
    "a".repeat(bytes)
}

/// A registration whose row weighs `bytes`: three labels at a value's
/// bound and a fourth that fills the rest.
fn weighing(bytes: usize) -> proto::WorkerSpec {
    let mut stated = proto::WorkerSpec {
        labels: ["a", "b", "c"]
            .map(|key| (key.to_owned(), text(VALUE_BYTES)))
            .into(),
        ..spec()
    };
    stated.labels.insert("fill".to_owned(), text(200));
    let base = weight(&facts(&stated, None).expect("a row under every bound"));
    stated
        .labels
        .insert("fill".to_owned(), text(200 + bytes - base));
    stated
}

/// A store row with every fact the wire's row tells set to a value of
/// its own.
fn full() -> WorkerRow {
    let facts = WorkerFacts {
        key: URL.to_owned(),
        engine: "grpc://10.0.0.5".to_owned(),
        incarnation: 3,
        role: Role::Decode,
        wire_modes: WireModes::GRPC,
        engine_kind: "runtime-a".to_owned(),
        models: vec!["model-a".to_owned(), "model-b".to_owned()],
        dp_rank: 2,
        dp_size: 4,
        labels: BTreeMap::from([
            ("policy".to_owned(), "least_load".to_owned()),
            ("tier".to_owned(), "gold".to_owned()),
        ]),
        zone: Some("zone-1".to_owned()),
        max_running: Some(96),
        kv_block_size: Some(16),
        bootstrap_port: Some(8_998),
        reachable_from: Reachability::Gateway(GatewayId(0xabc)),
        signals: Signals {
            events: true,
            loads: true,
            scrape_interval_ms: 2_500,
        },
        health: HealthPolicy {
            failure_threshold: 5,
            success_threshold: 4,
            check_interval_ms: 7_000,
            timeout_ms: 900,
            probes: false,
        },
        overload: Overload {
            kv_usage_permille: Some(875),
            waiting_requests: None,
        },
        demand_facts: DemandFacts {
            runtime: 7,
            references: true,
            host_pipeline: false,
            window: 32_768,
            decoding: 3,
        },
        vision: true,
        encode: EncodeOffer {
            handoff: EncodeHandoff::Push,
            producer: true,
            consumer: false,
            connector: Some("connector-a".to_owned()),
            references: true,
        },
        ..WorkerFacts::default()
    };
    WorkerRow {
        id: identity(URL, 3),
        facts,
        status: WorkerStatus::Draining,
        revision: 11,
        admitted_at: Instant::now(),
        maker: Default::default(),
    }
}

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

#[test]
fn a_log_entry_of_each_membership_write_has_its_case_at_the_entrys_place() {
    let old = full();
    let new = WorkerRow {
        id: identity(URL, 4),
        status: WorkerStatus::Ready,
        revision: old.revision + 3,
        ..full()
    };
    let at = Position { seq: 9, term: 4 };
    let entry = |change, row: Option<&WorkerRow>, left: Option<&WorkerRow>| Entry {
        at,
        event: Written::Member {
            epoch: 6,
            change,
            row: row.cloned().map(Arc::new),
            left: left.cloned().map(Arc::new),
        },
    };
    let told = |entry: &Entry| {
        let sent = member(entry).expect("a membership event");
        assert_eq!((sent.seq, sent.membership_epoch, sent.term), (9, 6, 4));
        sent.event.expect("a case")
    };

    let registered = MembershipChange::Registered { id: new.id };
    assert_eq!(
        told(&entry(registered, Some(&new), None)),
        Event::Registered(row(&new))
    );
    for change in [
        MembershipChange::Relabelled { id: new.id },
        MembershipChange::Resized { id: new.id },
    ] {
        assert_eq!(
            told(&entry(change, Some(&new), None)),
            Event::Relabelled(row(&new))
        );
    }
    // The new incarnation's status came over from the old one's, or it
    // is a fresh row's.
    let replacement = MembershipChange::Replaced {
        old: old.id,
        new: new.id,
    };
    assert_eq!(
        told(&entry(replacement, Some(&new), Some(&old))),
        replaced(&old, &new, true)
    );
    let fresh = WorkerRow {
        status: WorkerStatus::Pending,
        ..new.clone()
    };
    assert_eq!(
        told(&entry(replacement, Some(&fresh), Some(&old))),
        replaced(&old, &fresh, false)
    );
    assert_eq!(
        told(&entry(
            MembershipChange::Removed { id: old.id },
            None,
            Some(&old)
        )),
        removed(&old)
    );
    // A status change is told with the revision its write left the row
    // at, the row's own.
    let (from, to) = (WorkerStatus::Pending, WorkerStatus::Ready);
    let moved = MembershipChange::StatusChanged {
        id: new.id,
        from,
        to,
    };
    assert_eq!(
        told(&entry(moved, Some(&new), None)),
        status_changed(new.id, from, to)
    );
    // A shape the store never tells, and an entry that is no worker's
    // event.
    assert_eq!(member(&entry(registered, None, None)), None);
    assert_eq!(member(&entry(moved, None, Some(&old))), None);
    assert_eq!(
        member(&Entry {
            at,
            event: Written::Term
        }),
        None
    );
}

#[test]
fn a_page_carries_its_rows_at_the_cuts_place_and_the_word_to_start_again_carries_none() {
    let first = full();
    let second = WorkerRow {
        id: identity(URL, 2),
        ..full()
    };
    let at = Position { seq: 9, term: 4 };
    assert_eq!(
        page(vec![row(&first), row(&second)], false, at, 6),
        proto::WorkerEvent {
            seq: 9,
            membership_epoch: 6,
            event: Some(Event::Snapshot(proto::WorkerSnapshot {
                workers: vec![row(&first), row(&second)],
                last: false,
            })),
            term: 4,
        }
    );
    assert_eq!(
        page(Vec::new(), true, at, 6).event,
        Some(Event::Snapshot(proto::WorkerSnapshot {
            workers: Vec::new(),
            last: true,
        }))
    );
    // The oldest event the log retains is the one after its floor.
    assert_eq!(
        start_again(Position { seq: 40, term: 4 }),
        proto::WorkerEvent {
            seq: 0,
            membership_epoch: 0,
            event: Some(Event::SnapshotRequired(proto::SnapshotRequired {
                oldest_seq: 41,
            })),
            term: 0,
        }
    );
}

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
fn a_patch_merges_the_labels_and_replaces_the_overload_block() {
    // A label the patch leaves unsaid stays: the patch is merged.
    let mut standing = full().facts;
    standing.labels.insert("zone".to_owned(), "west".to_owned());
    let request = proto::PatchWorkerRequest {
        labels: [
            ("tier".to_owned(), "silver".to_owned()),
            ("lane".to_owned(), "b".to_owned()),
        ]
        .into(),
        remove_labels: vec!["policy".to_owned(), "absent".to_owned()],
        overload: Some(proto::OverloadThresholds {
            queue_limit: Some(12),
            kv_usage_permille: None,
        }),
        ..Default::default()
    };
    let made = patch(&request, &standing)
        .expect("a patch in order")
        .expect("a patch that changes the row");
    assert_eq!(
        made.labels,
        Some(BTreeMap::from([
            ("lane".to_owned(), "b".to_owned()),
            ("tier".to_owned(), "silver".to_owned()),
            ("zone".to_owned(), "west".to_owned()),
        ]))
    );
    assert_eq!(made.priority, None);
    assert_eq!(
        made.overload,
        Some(Overload {
            kv_usage_permille: Overload::default().kv_usage_permille,
            waiting_requests: Some(12),
        })
    );
}

#[test]
fn a_patch_that_changes_nothing_is_none_and_a_label_both_set_and_removed_is_refused() {
    // The row states both thresholds and so does each block below: a
    // block is compared with the row field by field, whatever the
    // store's defaults are.
    let mut standing = full().facts;
    standing.overload.waiting_requests = Some(12);
    let stating = |queue_limit| proto::PatchWorkerRequest {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        remove_labels: vec!["absent".to_owned()],
        overload: Some(proto::OverloadThresholds {
            queue_limit: Some(queue_limit),
            kv_usage_permille: Some(875),
        }),
        ..Default::default()
    };
    assert_eq!(patch(&stating(12), &standing), Ok(None));
    let empty = proto::PatchWorkerRequest::default();
    assert_eq!(patch(&empty, &standing), Ok(None));
    // A block that differs in one threshold is the patch's one change.
    let made = patch(&stating(13), &standing).expect("a patch in order");
    assert_eq!(
        made.map(|patch| (patch.labels, patch.overload)),
        Some((
            None,
            Some(Overload {
                kv_usage_permille: Some(875),
                waiting_requests: Some(13),
            })
        ))
    );

    let both = proto::PatchWorkerRequest {
        labels: [("tier".to_owned(), "silver".to_owned())].into(),
        remove_labels: vec!["tier".to_owned()],
        ..Default::default()
    };
    assert_eq!(
        patch(&both, &standing).map_err(|refused| refused.path),
        Err("remove_labels")
    );
    let over = proto::PatchWorkerRequest {
        overload: Some(proto::OverloadThresholds {
            queue_limit: None,
            kv_usage_permille: Some(4_000),
        }),
        ..Default::default()
    };
    assert_eq!(
        patch(&over, &standing).map_err(|refused| refused.path),
        Err("overload.kv_usage_permille")
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

#[test]
fn a_row_at_its_bound_is_taken_and_one_byte_over_is_refused_whole() {
    assert_eq!(ROW_BYTES, 16_384);
    let made = facts(&weighing(ROW_BYTES), None).expect("a row at its bound");
    assert_eq!(weight(&made), ROW_BYTES);
    assert_eq!(
        facts(&weighing(ROW_BYTES + 1), None).map(|_| ()),
        Err(refused("spec", "a row is at most 16384 bytes on the wire"))
    );
}

#[test]
fn a_patch_is_held_to_the_bounds_of_the_row_it_would_make() {
    type Asking = fn(usize) -> proto::PatchWorkerRequest;
    let keyed = |labels: Vec<(String, String)>| proto::PatchWorkerRequest {
        labels: labels.into_iter().collect(),
        ..Default::default()
    };
    let cases: [(&str, usize, &str, Asking); 5] = [
        (
            "labels",
            KEY_BYTES,
            "a label's key is at most 128 bytes",
            |bytes| proto::PatchWorkerRequest {
                labels: [(text(bytes), String::new())].into(),
                ..Default::default()
            },
        ),
        (
            "labels",
            VALUE_BYTES,
            "a label's value is at most 4096 bytes",
            |bytes| proto::PatchWorkerRequest {
                labels: [("k".to_owned(), text(bytes))].into(),
                ..Default::default()
            },
        ),
        (
            "labels",
            LABELS,
            "a row holds at most 128 labels",
            |count| {
                let labels = (0..count).map(|n| (n.to_string(), String::new()));
                proto::PatchWorkerRequest {
                    labels: labels.collect(),
                    ..Default::default()
                }
            },
        ),
        (
            "remove_labels",
            LABELS,
            "a patch removes at most 128 labels",
            |count| {
                let remove_labels = (0..count).map(|n| n.to_string()).collect();
                proto::PatchWorkerRequest {
                    remove_labels,
                    ..Default::default()
                }
            },
        ),
        (
            "remove_labels",
            KEY_BYTES,
            "a label's key is at most 128 bytes",
            |bytes| proto::PatchWorkerRequest {
                remove_labels: vec![text(bytes)],
                ..Default::default()
            },
        ),
    ];
    let bare = facts(&spec(), None).expect("a bare row");
    for (path, bound, why, asking) in cases {
        assert!(patch(&asking(bound), &bare).is_ok(), "{path} at {bound}");
        assert_eq!(
            patch(&asking(bound + 1), &bare).map(|_| ()),
            Err(refused(path, why)),
            "{path} past {bound}"
        );
        assert!(why.contains(&bound.to_string()), "{why}");
    }
    // The removals' count is read before the list is: past it a patch
    // is refused by the count, whatever its list holds.
    let both = proto::PatchWorkerRequest {
        labels: [("0".to_owned(), String::new())].into(),
        remove_labels: (0..=LABELS).map(|n| n.to_string()).collect(),
        ..Default::default()
    };
    assert_eq!(
        patch(&both, &bare).map(|_| ()),
        Err(refused(
            "remove_labels",
            "a patch removes at most 128 labels"
        ))
    );
    // The count is of the row the patch would make: a label it removes
    // makes room for the one it sets.
    let mut crowded = bare.clone();
    crowded.labels = (0..LABELS)
        .map(|n| (n.to_string(), String::new()))
        .collect();
    let more = keyed(vec![("more".to_owned(), String::new())]);
    assert_eq!(
        patch(&more, &crowded).map(|_| ()),
        Err(refused("labels", "a row holds at most 128 labels"))
    );
    let swap = proto::PatchWorkerRequest {
        remove_labels: vec!["0".to_owned()],
        ..more
    };
    assert!(patch(&swap, &crowded).is_ok_and(|made| made.is_some()));
    // The row whole: a label of 500 bytes under a one-byte key adds 509.
    let near = facts(&weighing(ROW_BYTES - 509), None).expect("a row under its bound");
    let filling = |bytes| keyed(vec![("p".to_owned(), text(bytes))]);
    let made = patch(&filling(500), &near).expect("a row at its bound");
    let mut after = near.clone();
    after.labels = made
        .and_then(|patch| patch.labels)
        .expect("the labels moved");
    assert_eq!(weight(&after), ROW_BYTES);
    assert_eq!(
        patch(&filling(501), &near).map(|_| ()),
        Err(refused(
            "labels",
            "a row is at most 16384 bytes on the wire"
        ))
    );
    // The policy label is told twice, as a label and under its own
    // field: one of 246 bytes on a row with none adds 510.
    let under = facts(&weighing(ROW_BYTES - 510), None).expect("a row under its bound");
    let policy = |bytes| keyed(vec![("policy".to_owned(), text(bytes))]);
    let made = patch(&policy(246), &under).expect("a row at its bound");
    let mut after = under.clone();
    after.labels = made
        .and_then(|patch| patch.labels)
        .expect("the labels moved");
    assert_eq!(weight(&after), ROW_BYTES);
    assert_eq!(
        patch(&policy(247), &under).map(|_| ()),
        Err(refused(
            "labels",
            "a row is at most 16384 bytes on the wire"
        ))
    );
    // A row at its bound takes a patch of the overload block alone and
    // stays within it at its widest: the block was weighed so.
    let brim = facts(&weighing(ROW_BYTES), None).expect("a row at its bound");
    let block = proto::PatchWorkerRequest {
        overload: Some(proto::OverloadThresholds {
            queue_limit: Some(u32::MAX),
            kv_usage_permille: Some(1_000),
        }),
        ..Default::default()
    };
    let made = patch(&block, &brim).expect("a patch in order");
    let stated = WorkerFacts {
        overload: made.and_then(|patch| patch.overload).expect("the block"),
        ..brim
    };
    let mut widest = row(&WorkerRow {
        id: WorkerId(u64::MAX),
        facts: stated.clone(),
        status: WorkerStatus::Ready,
        revision: u64::MAX,
        ..full()
    });
    widest.incarnation = u64::MAX;
    assert_eq!(widest.encoded_len(), ROW_BYTES);
    // And the row registers again as it stands, its block stated.
    let again = proto::WorkerSpec {
        overload: block.overload,
        ..weighing(ROW_BYTES)
    };
    assert!(facts(&again, Some(&stated)).is_ok_and(|made| made == stated));
    // And a label of the same bytes in other letters still fits it.
    let other = keyed(vec![("a".to_owned(), "b".repeat(VALUE_BYTES))]);
    assert!(patch(&other, &stated).is_ok_and(|made| made.is_some()));
}

#[test]
fn a_rows_weight_is_its_bytes_on_the_wire_with_its_numbers_at_their_widest() {
    // A row whose every text holds `bytes`, a model id and a label's key
    // left empty, under `labels` more labels of no value.
    let edge = |bytes: usize, labels: usize| WorkerRow {
        facts: WorkerFacts {
            key: text(bytes),
            engine: text(bytes),
            engine_kind: text(bytes),
            models: vec![String::new(), text(bytes)],
            zone: Some(text(bytes)),
            labels: (0..labels)
                .map(|n| (n.to_string(), String::new()))
                .chain([
                    ("policy".to_owned(), text(bytes)),
                    (String::new(), text(bytes)),
                ])
                .collect(),
            ..full().facts
        },
        ..full()
    };
    let bare = WorkerRow {
        facts: facts(&spec(), None).expect("a bare row"),
        ..full()
    };
    // A row that names a runtime and scrapes nothing tells no family.
    let idle = WorkerRow {
        facts: WorkerFacts {
            signals: Signals {
                scrape_interval_ms: 0,
                ..full().facts.signals
            },
            ..full().facts
        },
        ..full()
    };
    // The widest a row of these facts is told: the controller's numbers
    // at their most, under a status that sets a health bit, its overload
    // block stated in full. The sum is the wire's own length at each
    // side of a length's second byte, and with a field left empty.
    for stored in [
        full(),
        bare,
        idle,
        edge(127, 127),
        edge(128, 128),
        edge(300, 0),
    ] {
        let widest = WorkerRow {
            id: WorkerId(u64::MAX),
            status: WorkerStatus::Ready,
            revision: u64::MAX,
            ..stored
        };
        let mut told = row(&widest);
        told.incarnation = u64::MAX;
        told.overload = Some(proto::OverloadThresholds {
            queue_limit: Some(u32::MAX),
            kv_usage_permille: Some(1_000),
        });
        let key = widest.facts.key.len();
        assert_eq!(weight(&widest.facts), told.encoded_len(), "a key of {key}");
    }
    // No status a row moves to tells more, and no revision.
    for status in [
        WorkerStatus::Pending,
        WorkerStatus::Ready,
        WorkerStatus::NotReady,
        WorkerStatus::Failed,
        WorkerStatus::Draining,
    ] {
        let at = WorkerRow { status, ..full() };
        assert!(row(&at).encoded_len() <= weight(&at.facts), "{status:?}");
    }
}

#[test]
fn the_stores_measure_is_never_under_the_wires_weight() {
    let widest = WorkerFacts {
        incarnation: u64::MAX,
        role: Role::Encode,
        wire_modes: WireModes::HTTP.union(WireModes::GRPC).union(WireModes::ZMQ),
        dp_rank: u16::MAX,
        dp_size: u16::MAX,
        kv_block_size: Some(u32::MAX),
        max_running: Some(u32::MAX),
        bootstrap_port: Some(u16::MAX),
        priority: i32::MAX,
        reachable_from: Reachability::Gateway(GatewayId(u64::MAX)),
        overload: Overload {
            kv_usage_permille: Some(u16::MAX),
            waiting_requests: Some(u32::MAX),
        },
        health: HealthPolicy {
            failure_threshold: u32::MAX,
            success_threshold: u32::MAX,
            check_interval_ms: u32::MAX,
            timeout_ms: u32::MAX,
            probes: false,
        },
        signals: Signals {
            events: true,
            loads: true,
            scrape_interval_ms: u32::MAX,
        },
        ..WorkerFacts::default()
    };
    // The same row where its worker states each demand fact at its
    // widest: the store charges the wire's block beside the numbers.
    let stating = WorkerFacts {
        demand_facts: DemandFacts {
            runtime: u32::MAX,
            references: true,
            host_pipeline: true,
            window: u32::MAX,
            decoding: u8::MAX,
        },
        ..widest.clone()
    };
    let numeric = |facts: &WorkerFacts| {
        let mut numbers = row(&WorkerRow {
            facts: facts.clone(),
            id: WorkerId(u64::MAX),
            revision: u64::MAX,
            status: WorkerStatus::Ready,
            ..full()
        });
        numbers.admitted_age_ms = u64::MAX;
        numbers.encoded_len()
    };
    // The same row again where its worker takes images too: the store
    // charges the wire's bit beside the block.
    let seeing = WorkerFacts {
        vision: true,
        ..stating.clone()
    };
    // The same row again where its worker states every role of an encode
    // offer and no connector: the store charges the wire's block.
    let offering = WorkerFacts {
        encode: EncodeOffer {
            handoff: EncodeHandoff::Rendezvous,
            producer: true,
            consumer: true,
            connector: None,
            references: true,
        },
        ..seeing.clone()
    };
    let (plain, block) = (numeric(&widest), numeric(&stating));
    assert!(plain <= WorkerFacts::NUMBERS, "{plain} numeric bytes");
    // What the store charges for the block is the block at its widest.
    assert_eq!(block - plain, WorkerFacts::DEMAND_FACTS);
    // What it charges for the bit is the bit.
    assert_eq!(numeric(&seeing) - block, WorkerFacts::VISION);
    // The measure adds that charge for a row that takes images. A row's
    // texts leave more room than the bit takes, so no row below needs
    // the charge to fit its measure, and it is read here.
    let charged = |facts: &WorkerFacts| facts.weight(&Registrant::Control);
    assert_eq!(charged(&seeing) - charged(&stating), WorkerFacts::VISION);
    // What it charges for an offer is the offer's block with no connector,
    // and the measure adds it for a row whose offer states anything.
    assert_eq!(numeric(&offering) - numeric(&seeing), WorkerFacts::ENCODE);
    assert_eq!(charged(&offering) - charged(&seeing), WorkerFacts::ENCODE);
    // A connector is charged as any text is, beside the block: its bytes
    // and a text's frame, and no byte more.
    let on = WorkerFacts {
        encode: EncodeOffer {
            connector: Some(text(300)),
            ..offering.encode.clone()
        },
        ..offering.clone()
    };
    assert_eq!(charged(&on) - charged(&offering), 300 + WorkerFacts::FRAME);

    let makers = [
        Registrant::Control,
        Registrant::provider("p", "k").expect("the shortest maker"),
        Registrant::provider(text(128), text(2_048)).expect("the widest maker"),
    ];
    for long in [false, true] {
        let sized = |least, most| text(if long { most } else { least });
        // With a scrape and with none: the block stated and not, the bit
        // beside the block, and an offer beside both, whose connector is
        // a text of the row.
        let cells = [
            (false, &widest),
            (true, &widest),
            (false, &stating),
            (true, &stating),
            (false, &seeing),
            (true, &seeing),
            (false, &offering),
            (true, &offering),
        ];
        for (scrape, numbers) in cells {
            let facts = WorkerFacts {
                key: sized(1, 2_048),
                engine: sized(0, 2_048),
                engine_kind: sized(0, 256),
                models: vec![sized(1, 256)],
                zone: Some(sized(0, 256)),
                labels: [
                    (sized(0, 128), sized(0, 4_096)),
                    ("policy".to_owned(), sized(0, 4_096)),
                ]
                .into(),
                signals: Signals {
                    scrape_interval_ms: if scrape { u32::MAX } else { 0 },
                    ..numbers.signals
                },
                encode: EncodeOffer {
                    connector: (numbers.encode != EncodeOffer::default()).then(|| sized(1, 256)),
                    ..numbers.encode.clone()
                },
                ..numbers.clone()
            };
            for maker in &makers {
                let wire = weight(&facts) + super::maker_weight(maker);
                let stored = facts.weight(maker);
                assert!(
                    wire <= stored,
                    "long={long}, scrape={scrape}, wire={wire}, store={stored}"
                );
            }
        }
    }

    // The row whose texts leave the measure the least room: a long
    // address, engine and runtime, and no other text, as a snapshot
    // tells it, with its age. The texts' own room is less than the
    // block there, so the block's charge is what keeps the measure
    // over the row's bytes.
    let lean = WorkerFacts {
        key: text(2_048),
        engine: text(2_048),
        engine_kind: text(256),
        encode: EncodeOffer {
            connector: Some(text(256)),
            ..offering.encode.clone()
        },
        ..offering
    };
    let mut told = row(&WorkerRow {
        facts: lean.clone(),
        id: WorkerId(u64::MAX),
        revision: u64::MAX,
        status: WorkerStatus::Ready,
        ..full()
    });
    told.admitted_age_ms = u64::MAX;
    let (wire, stored) = (told.encoded_len(), lean.weight(&Registrant::Control));
    assert!(
        wire <= stored,
        "the leanest row: wire={wire}, store={stored}"
    );
}

#[test]
fn a_page_of_rows_at_their_bound_is_half_a_frame_and_an_answer_far_less() {
    // The contract's page (`WorkerSnapshot`) and a client's default frame.
    const PAGE: usize = 128;
    const FRAME: usize = 4 * 1024 * 1024;
    let fat = WorkerRow {
        facts: facts(&weighing(ROW_BYTES), None).expect("a row at its bound"),
        ..full()
    };
    let page = proto::WorkerSnapshot {
        workers: vec![row(&fat); PAGE],
        last: true,
    };
    // A page frames each row in four bytes.
    assert!(page.encoded_len() <= FRAME / 2 + 4 * PAGE + 2);
    // A replacement's answer carries two rows.
    let answer = event(replaced(&fat, &fat, true), u64::MAX, u64::MAX, u64::MAX);
    assert!(answer.encoded_len() <= 2 * ROW_BYTES + 64);
}

#[test]
fn every_kind_of_log_entry_is_sent_under_a_case_of_its_own() {
    let at = Position { seq: 3, term: 2 };
    let stored = full();
    let entries = [
        Written::Term,
        Written::Member {
            epoch: 1,
            change: MembershipChange::Registered { id: stored.id },
            row: Some(Arc::new(stored.clone())),
            left: None,
        },
    ]
    .map(|event| replication(&Entry { at, event }).event);
    assert_eq!(entries[0], Some(Sent::TermStart(proto::TermStart {})));
    assert_eq!(
        entries[1],
        Some(Sent::Membership(event(
            Event::Registered(row(&stored)),
            3,
            1,
            2
        )))
    );
    // A membership entry in a shape the store never tells is sent with
    // no case set: the standby reads an event it does not know.
    let untold = Written::Member {
        epoch: 1,
        change: MembershipChange::Registered { id: stored.id },
        row: None,
        left: None,
    };
    assert_eq!(replication(&Entry { at, event: untold }).event, None);
}

#[test]
fn a_row_crosses_with_its_maker() {
    use super::{maker_weight, room};
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
fn every_membership_event_comes_back_as_the_change_the_store_follows() {
    let old = full();
    let new = WorkerRow {
        id: identity(URL, 4),
        status: WorkerStatus::Ready,
        ..full()
    };
    let now = new.admitted_at;
    let sent = |change, row: Option<&WorkerRow>, left: Option<&WorkerRow>| {
        let entry = Entry {
            at: Position { seq: 9, term: 4 },
            event: Written::Member {
                epoch: 6,
                change,
                row: row.cloned().map(Arc::new),
                left: left.cloned().map(Arc::new),
            },
        };
        told_change(&member(&entry).expect("a membership event"), now)
    };
    let registered = MembershipChange::Registered { id: new.id };
    let relabelled = MembershipChange::Relabelled { id: new.id };
    let resized = MembershipChange::Resized { id: new.id };
    let replacement = MembershipChange::Replaced {
        old: old.id,
        new: new.id,
    };
    let removal = MembershipChange::Removed { id: old.id };
    let status = MembershipChange::StatusChanged {
        id: old.id,
        from: WorkerStatus::Ready,
        to: WorkerStatus::Draining,
    };
    let row = Some(new.clone());
    assert_eq!(
        sent(registered, Some(&new), None),
        Ok((6, registered, row.clone()))
    );
    assert_eq!(
        sent(relabelled, Some(&new), None),
        Ok((6, relabelled, row.clone()))
    );
    // A resize crosses as a relabel: the store reads the widths.
    assert_eq!(
        sent(resized, Some(&new), None),
        Ok((6, relabelled, row.clone()))
    );
    assert_eq!(
        sent(replacement, Some(&new), Some(&old)),
        Ok((6, replacement, row))
    );
    assert_eq!(sent(removal, None, Some(&old)), Ok((6, removal, None)));
    assert_eq!(sent(status, Some(&old), None), Ok((6, status, None)));
    // A watch's own cases and an event of no case are not a change.
    for event in [
        None,
        Some(Event::Snapshot(proto::WorkerSnapshot::default())),
    ] {
        let none = proto::WorkerEvent {
            event,
            ..proto::WorkerEvent::default()
        };
        let refused = told_change(&none, now).expect_err("no change");
        assert_eq!(refused.path, "membership.event");
    }
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
