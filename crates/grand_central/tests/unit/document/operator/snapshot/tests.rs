use super::*;

/// An address is told without its userinfo and with every other byte as
/// registered. One with no userinfo is told as it is. One whose `@`
/// stands past the authority's end is told by the marker: no part of
/// what may be a userinfo is told.
#[test]
fn an_address_is_told_without_its_userinfo() {
    for (address, told) in [
        (
            "grpc://user:secret@10.0.0.5:8000",
            Some("grpc://10.0.0.5:8000"),
        ),
        (
            "https://user@host.example:8443/v1?x=1#top",
            Some("https://host.example:8443/v1?x=1#top"),
        ),
        ("user:secret@10.0.0.5:8000", Some("10.0.0.5:8000")),
        // An `@` inside the userinfo: the authority's last one ends it.
        (
            "grpc://us@er:se@cret@10.0.0.5:8000",
            Some("grpc://10.0.0.5:8000"),
        ),
        (
            "grpc+tls://user@10.0.0.5:8000",
            Some("grpc+tls://10.0.0.5:8000"),
        ),
        // A `://` behind the authority is no scheme's: the authority
        // starts at the address's first byte, and its userinfo goes.
        (
            "user:secret@10.0.0.5:8000/next/http://other",
            Some("10.0.0.5:8000/next/http://other"),
        ),
        (
            "user:secret@10.0.0.5:8000?next=http://other",
            Some("10.0.0.5:8000?next=http://other"),
        ),
        (
            "user:secret@10.0.0.5:8000#http://other",
            Some("10.0.0.5:8000#http://other"),
        ),
        (
            "grpc://user:secret@10.0.0.5:8000?next=http://other",
            Some("grpc://10.0.0.5:8000?next=http://other"),
        ),
        // No userinfo: every byte stays.
        ("grpc://10.0.0.5:8000", None),
        ("10.0.0.5:8000?next=http://other", None),
        ("http://[::1]:8000/v1?x=1", None),
        ("10.0.0.5:8000", None),
        ("1http://10.0.0.5:8000", None),
        ("", None),
        // A userinfo that holds a raw `/`, `?` or `#` reads like these,
        // and so does a path that holds an `@`.
        ("http://user:se/cret@10.0.0.5:8000", Some(UNPARSABLE)),
        ("http://user:se?cret@10.0.0.5:8000", Some(UNPARSABLE)),
        ("http://user:se#cret@10.0.0.5:8000", Some(UNPARSABLE)),
        ("http://10.0.0.5:8000/a@b", Some(UNPARSABLE)),
        // No scheme by its form stands before `://`: its `/` ends the
        // authority, and the `@` behind it stands past that end.
        ("user:se://cret@10.0.0.5:8000", Some(UNPARSABLE)),
        ("1http://user:secret@10.0.0.5:8000", Some(UNPARSABLE)),
    ] {
        assert_eq!(without_userinfo(address).as_deref(), told, "{address}");
    }
}

#[test]
fn a_plain_page_stops_at_its_bytes_even_for_its_first_row() {
    use crate::store::{MemoryStore, WorkerFacts, WorkersStore};

    let store = MemoryStore::new();
    let asked = WorkerRowsQuery {
        models: Vec::new(),
        limit: usize::MAX,
    };
    let now = Instant::now();
    for n in 0..2 {
        let key = format!("grpc://{n}.{}", "x".repeat((2 * 1024 * 1024) / 4));
        store.upsert(WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            dp_size: 1,
            models: vec!["m".to_owned()],
            ..WorkerFacts::default()
        });
    }
    // Every row, by key, with the count that serve: a lone read's
    // answer for a call with no model and no limit.
    let read = || {
        let rows = store.checkpoint().rows;
        let serving = rows.len();
        (rows, serving)
    };
    let (rows, serving) = read();
    let page = worker_rows_page(&asked, Rows::Told(rows, serving), now);
    assert_eq!(page.rows.len(), 1);
    assert!(page.truncated);

    let (mut rows, _) = read();
    rows.truncate(1);
    rows[0].facts.key = "x".repeat(2 * 1024 * 1024);
    let too_large = worker_rows_page(&asked, Rows::Told(rows, 1), now);
    assert!(too_large.rows.is_empty() && too_large.truncated);
}

/// A page's JSON, literally: a row that states every fact, a
/// provider's among them with a credential in its key and in its own,
/// and a row that states the least, admitted after the take.
#[test]
fn a_page_writes_its_rows_as_an_operator_reads_them() {
    use std::time::Duration;

    use crate::{
        engine::view::WorkerId,
        store::{
            pairing::PairingDescriptor, EncodeOffer, GatewayId, Overload, Signals, WorkerFacts,
            WorkerStatus,
        },
    };

    let admitted = Instant::now();
    let taken = admitted + Duration::from_millis(1500);
    let full = WorkerRow {
        id: WorkerId(0xab),
        facts: WorkerFacts {
            key: "grpc://user:secret@10.0.0.5:8000".to_owned(),
            engine: "grpc://user:secret@10.0.0.5:8000".to_owned(),
            incarnation: 7,
            role: Role::Prefill,
            wire_modes: WireModes::from_bits(3),
            engine_kind: "runtime-a".to_owned(),
            models: vec!["m1".to_owned(), "m2".to_owned()],
            dp_rank: 1,
            dp_size: 2,
            labels: BTreeMap::from([("policy".to_owned(), "default".to_owned())]),
            zone: Some("zone-a".to_owned()),
            pairing: Some(Arc::new(PairingDescriptor {
                protocol: Some("p1".to_owned()),
                transport: Some("t1".to_owned()),
                version: Some("v1".to_owned()),
                kv_layout: [
                    Some("float16".to_owned()),
                    None,
                    Some("a1".to_owned()),
                    None,
                ],
            })),
            encode: EncodeOffer {
                handoff: EncodeHandoff::Push,
                consumer: true,
                connector: Some("c1".to_owned()),
                ..EncodeOffer::default()
            },
            vision: true,
            max_running: Some(8),
            kv_block_size: Some(16),
            bootstrap_port: Some(9000),
            priority: 5,
            reachable_from: Reachability::Gateway(GatewayId(0x1f)),
            signals: Signals {
                events: true,
                loads: true,
                scrape_interval_ms: 0,
            },
            overload: Overload {
                kv_usage_permille: Some(900),
                waiting_requests: Some(64),
            },
            ..WorkerFacts::default()
        },
        status: WorkerStatus::Ready,
        revision: 3,
        admitted_at: admitted,
        maker: Registrant::provider("finder", "http://u:p@host/x").expect("a maker"),
    };
    let least = WorkerRow {
        id: WorkerId(1),
        facts: WorkerFacts {
            key: "http://b:1".to_owned(),
            engine: "http://b:1".to_owned(),
            incarnation: 1,
            dp_size: 1,
            ..WorkerFacts::default()
        },
        status: WorkerStatus::Pending,
        revision: 1,
        admitted_at: taken + Duration::from_secs(2),
        maker: Registrant::Control,
    };
    let asked = WorkerRowsQuery {
        models: Vec::new(),
        limit: usize::MAX,
    };
    let page = worker_rows_page(&asked, Rows::Told(vec![full, least], 2), taken);
    let probes = serde_json::json!({
        "failure_threshold": 3,
        "success_threshold": 2,
        "check_interval_ms": 60_000,
        "timeout_ms": 5_000,
        "disabled": false
    });
    let expected = serde_json::json!({
        "rows": [
            {
                "id": "00000000000000ab",
                "registration_key": "grpc://10.0.0.5:8000",
                "incarnation": 7,
                "engine": "grpc://10.0.0.5:8000",
                "wire": ["http", "grpc"],
                "role": "prefill",
                "runtime": "runtime-a",
                "models": ["m1", "m2"],
                "dp_rank": 1,
                "dp_size": 2,
                "labels": {"policy": "default"},
                "zone": "zone-a",
                "status": "ready",
                "revision": 3,
                "reachable": {"gateway_id": "000000000000001f"},
                "pairing": {
                    "protocol": "p1",
                    "transport": "t1",
                    "version": "v1",
                    "bootstrap_port": 9000,
                    "kv_layout": {"kv_cache_dtype": "float16", "attention_backend": "a1"},
                    "ec_producer": false,
                    "ec_consumer": true,
                    "vision": true,
                    "encode_handoff": "push",
                    "ec_connector": "c1",
                    "encode_references": false
                },
                "max_running": 8,
                "kv_block_size": 16,
                "priority": 5,
                "overload": {"queue_limit": 64, "kv_usage_permille": 900},
                "signals": {"events": true, "loads": true, "scrape_interval_ms": 0},
                "health_policy": probes,
                "admitted_age_ms": 1500,
                "maker": {"provider": "finder", "key": "http://host/x"}
            },
            {
                "id": "0000000000000001",
                "registration_key": "http://b:1",
                "incarnation": 1,
                "engine": "http://b:1",
                "wire": [],
                "role": "single",
                "runtime": "",
                "models": [],
                "dp_rank": 0,
                "dp_size": 1,
                "labels": {},
                "zone": null,
                "status": "pending",
                "revision": 1,
                "reachable": {"any": true},
                "pairing": null,
                "max_running": null,
                "kv_block_size": null,
                "priority": 0,
                "overload": {"queue_limit": null, "kv_usage_permille": null},
                "signals": {"events": false, "loads": false, "scrape_interval_ms": 0},
                "health_policy": probes,
                "admitted_age_ms": 0,
                "maker": null
            }
        ],
        "truncated": false
    });
    assert_eq!(
        serde_json::to_value(&page).expect("a page's JSON"),
        expected
    );
}
