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
    use crate::store::{MemoryStore, WorkerFacts};

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

/// A page's bytes count the block of demand facts by what each worker
/// states. Where a worker states every fact at its longest, its row's
/// measure grows by what its JSON grows. A page of short rows keeps its
/// JSON inside the page's bytes, whether its workers state the facts or
/// state none.
#[test]
fn a_page_counts_the_demand_facts_its_rows_tell() {
    use crate::{
        engine::{view::WorkerId, DemandFacts},
        store::{WorkerFacts, WorkerStatus},
    };

    let now = Instant::now();
    let longest = DemandFacts {
        runtime: u32::MAX,
        references: false,
        host_pipeline: false,
        window: u32::MAX,
        decoding: u8::MAX,
    };
    let row = |n: u64, demand_facts| {
        let address = format!("http://10.0.{}.{}:8000", n / 256, n % 256);
        WorkerRow {
            id: WorkerId(n),
            facts: WorkerFacts {
                key: address.clone(),
                engine: address,
                incarnation: 1,
                dp_size: 1,
                models: vec!["m".to_owned()],
                demand_facts,
                ..WorkerFacts::default()
            },
            status: WorkerStatus::Ready,
            revision: 1,
            admitted_at: now,
            maker: Registrant::Control,
        }
    };
    let json =
        |row: &WorkerRow| serde_json::to_string(&RowJson::of(row, now)).expect("a row's JSON");

    let (silent, stating) = (row(1, DemandFacts::default()), row(1, longest));
    let (told_none, told) = (json(&silent), json(&stating));
    assert!(
        told_none.ends_with(r#","demand_facts":null}"#),
        "{told_none}"
    );
    assert_eq!(r#","demand_facts":null"#.len(), NO_DEMAND_FACTS_BYTES);
    assert_eq!(
        told.len() - told_none.len(),
        DEMAND_FACTS_BYTES - NO_DEMAND_FACTS_BYTES,
        "{told}"
    );
    assert_eq!(
        measure(&stating) - measure(&silent),
        told.len() - told_none.len()
    );

    let asked = WorkerRowsQuery {
        models: Vec::new(),
        limit: usize::MAX,
    };
    for stated in [longest, DemandFacts::default()] {
        let rows: Vec<WorkerRow> = (0..4_000).map(|n| row(n, stated)).collect();
        let serving = rows.len();
        let page = worker_rows_page(&asked, Rows::Told(rows, serving), now);
        let bytes = serde_json::to_vec(&page).expect("a page's JSON").len();
        assert!(
            page.truncated && bytes <= PAGE_BYTES,
            "{stated:?}: {bytes} bytes in {} rows",
            page.rows.len()
        );
    }
}

/// A page's bytes count a text by what its JSON writes for it: two bytes
/// for a quote, for a backslash and for a control character JSON names by
/// a letter, six for any other control character, and every other byte as
/// it stands. A row's measure grows by what its JSON grows where its texts
/// hold such characters, and a page of rows with such a zone and such a
/// label keeps its JSON inside the page's bytes.
#[test]
fn a_page_counts_a_text_by_what_its_json_writes() {
    use crate::{
        engine::view::WorkerId,
        store::{EncodeOffer, WorkerFacts, WorkerStatus},
    };

    // Every character of one byte, and one each of two, three and four.
    let one_byte = (0..=0x7F_u8).map(|byte| char::from(byte).to_string());
    for text in one_byte.chain(["\u{e9}", "\u{2028}", "\u{1f600}"].map(str::to_owned)) {
        let json = serde_json::to_string(&text).expect("a text's JSON");
        assert_eq!(escaped_len(&text), json.len() - 2, "{text:?}");
    }

    let now = Instant::now();
    let row = |n: u64, zone: String, tier: String, connector: Option<String>| {
        let address = format!("http://10.0.{}.{}:8000", n / 256, n % 256);
        WorkerRow {
            id: WorkerId(n),
            facts: WorkerFacts {
                key: address.clone(),
                engine: address,
                incarnation: 1,
                dp_size: 1,
                models: vec!["m".to_owned()],
                labels: [("tier".to_owned(), tier)].into(),
                zone: Some(zone),
                encode: EncodeOffer {
                    connector,
                    ..EncodeOffer::default()
                },
                ..WorkerFacts::default()
            },
            status: WorkerStatus::Ready,
            revision: 1,
            admitted_at: now,
            maker: Registrant::Control,
        }
    };
    // Three texts a registration states as it likes: a zone and a label's
    // value, each as long as a row lets it be, and a connector. The plain
    // row holds a letter in every place. The other holds a zone of bells,
    // a value of quotes and backslashes, and a connector with a line feed
    // in every fourth place.
    let (zone, value) = (WorkerFacts::NAME_BYTES, WorkerFacts::LABEL_VALUE_BYTES);
    let (bells, quotes) = ("\u{7}".repeat(zone), r#""\"#.repeat(value / 2));
    let plain = row(
        1,
        "z".repeat(zone),
        "t".repeat(value),
        Some("c".repeat(1_024)),
    );
    let escaped = row(1, bells.clone(), quotes.clone(), Some("nfs\n".repeat(256)));
    let json =
        |row: &WorkerRow| serde_json::to_string(&RowJson::of(row, now)).expect("a row's JSON");
    let (told_plain, told) = (json(&plain), json(&escaped));
    // Five more bytes a bell, and one more a quote, a backslash and a
    // line feed.
    assert_eq!(told.len() - told_plain.len(), 5 * zone + value + 256);
    assert_eq!(
        measure(&escaped) - measure(&plain),
        told.len() - told_plain.len()
    );

    // A page of rows with that zone and that value. They state no pair and
    // no encode offer: what that block's own fields take in a row's JSON
    // is no part of a text's count.
    let asked = WorkerRowsQuery {
        models: Vec::new(),
        limit: usize::MAX,
    };
    let rows: Vec<WorkerRow> = (0..2_000)
        .map(|n| row(n, bells.clone(), quotes.clone(), None))
        .collect();
    let serving = rows.len();
    let page = worker_rows_page(&asked, Rows::Told(rows, serving), now);
    let bytes = serde_json::to_vec(&page).expect("a page's JSON").len();
    assert!(
        page.truncated && bytes <= PAGE_BYTES,
        "{bytes} bytes in {} rows",
        page.rows.len()
    );
}

/// A page's JSON, literally: a row that states every fact, a
/// provider's among them with a credential in its key and in its own,
/// and a row that states the least, admitted after the take. The first
/// tells what its worker states for a request's hard demands, each fact
/// a value of its own, and the second tells none. A third states a
/// window and a pipeline of its host's with no runtime, and tells its
/// block all the same.
#[test]
fn a_page_writes_its_rows_as_an_operator_reads_them() {
    use std::time::Duration;

    use crate::{
        engine::{view::WorkerId, DemandFacts},
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
            demand_facts: DemandFacts {
                runtime: 7,
                references: true,
                host_pipeline: false,
                window: 4_096,
                decoding: 3,
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
    let windowed = WorkerRow {
        id: WorkerId(2),
        facts: WorkerFacts {
            key: "http://c:1".to_owned(),
            engine: "http://c:1".to_owned(),
            demand_facts: DemandFacts {
                host_pipeline: true,
                window: 2_048,
                ..DemandFacts::default()
            },
            ..least.facts.clone()
        },
        ..least.clone()
    };
    let page = worker_rows_page(&asked, Rows::Told(vec![full, least, windowed], 3), taken);
    let probes = serde_json::json!({
        "failure_threshold": 3,
        "success_threshold": 2,
        "check_interval_ms": 60_000,
        "timeout_ms": 5_000,
        "disabled": false
    });
    let mut expected = serde_json::json!({
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
                "maker": {"provider": "finder", "key": "http://host/x"},
                "demand_facts": {
                    "runtime": 7,
                    "references": true,
                    "host_pipeline": false,
                    "window": 4096,
                    "decoding": 3
                }
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
                "maker": null,
                "demand_facts": null
            }
        ],
        "truncated": false
    });
    // The third row reads as the second but for its id, its two addresses
    // and its block.
    let mut third = expected["rows"][1].clone();
    third["id"] = "0000000000000002".into();
    third["registration_key"] = "http://c:1".into();
    third["engine"] = "http://c:1".into();
    third["demand_facts"] = serde_json::json!({
        "runtime": 0,
        "references": false,
        "host_pipeline": true,
        "window": 2048,
        "decoding": 0
    });
    expected["rows"]
        .as_array_mut()
        .expect("the rows")
        .push(third);
    assert_eq!(
        serde_json::to_value(&page).expect("a page's JSON"),
        expected
    );
}
