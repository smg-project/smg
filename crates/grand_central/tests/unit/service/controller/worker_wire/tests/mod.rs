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

mod log;
mod patches;
mod registration;
mod rows;
mod weight;

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
        priority: -5,
        drain_settle_secs: Some(45),
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
