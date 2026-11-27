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

/// One kind of row: the row of a number, admitted at an instant.
type Kind = fn(u64, Instant) -> WorkerRow;

/// A short row that states the least, at an address of its number.
fn short(n: u64, admitted: Instant) -> WorkerRow {
    use crate::{
        engine::view::WorkerId,
        store::{WorkerFacts, WorkerStatus},
    };

    let address = format!("http://10.0.{}.{}:8000", n / 256, n % 256);
    WorkerRow {
        id: WorkerId(n),
        facts: WorkerFacts {
            key: address.clone(),
            engine: address,
            incarnation: 1,
            dp_size: 1,
            models: vec!["m".to_owned()],
            ..WorkerFacts::default()
        },
        status: WorkerStatus::Ready,
        revision: 1,
        admitted_at: admitted,
        maker: Registrant::Control,
    }
}

/// [`short`], stating a pair's descriptor, its port and an encode offer.
fn paired(n: u64, admitted: Instant) -> WorkerRow {
    use crate::store::{pairing::PairingDescriptor, EncodeOffer};

    let mut row = short(n, admitted);
    row.facts.role = Role::Prefill;
    row.facts.pairing = Some(Arc::new(PairingDescriptor {
        protocol: Some("p1".to_owned()),
        transport: Some("t1".to_owned()),
        version: Some("v1".to_owned()),
        kv_layout: [
            Some("float16".to_owned()),
            None,
            Some("a1".to_owned()),
            None,
        ],
    }));
    row.facts.bootstrap_port = Some(9_000);
    row.facts.vision = true;
    row.facts.encode = EncodeOffer {
        handoff: EncodeHandoff::Push,
        consumer: true,
        connector: Some("c1".to_owned()),
        ..EncodeOffer::default()
    };
    row
}

/// [`short`], with a zone of bells, a label whose name and value hold
/// quotes and backslashes, and a model id with a quote: texts a row may
/// state, each written longer than it stands.
fn escaped(n: u64, admitted: Instant) -> WorkerRow {
    use crate::store::WorkerFacts;

    let mut row = short(n, admitted);
    row.facts.zone = Some("\u{7}".repeat(WorkerFacts::NAME_BYTES));
    let value = r#""\"#.repeat(WorkerFacts::LABEL_VALUE_BYTES / 2);
    row.facts.labels = [(r#"a"b\"#.to_owned(), value)].into();
    row.facts.models = vec![r#"m"1"#.to_owned()];
    row
}

/// [`short`], with every number at its longest, the names of a role, a
/// status and three wires, a reach of one gateway's and every demand fact.
fn longest(n: u64, admitted: Instant) -> WorkerRow {
    use crate::{
        engine::{view::WorkerId, DemandFacts},
        store::{GatewayId, HealthPolicy, Overload, Signals, WorkerStatus},
    };

    let mut row = short(n, admitted);
    row.id = WorkerId(u64::MAX - n);
    row.status = WorkerStatus::NotReady;
    row.revision = u64::MAX;
    let facts = &mut row.facts;
    facts.incarnation = u64::MAX;
    facts.role = Role::Prefill;
    facts.wire_modes = WireModes::from_bits(7);
    (facts.dp_rank, facts.dp_size) = (u16::MAX, u16::MAX);
    (facts.max_running, facts.kv_block_size) = (Some(u32::MAX), Some(u32::MAX));
    facts.bootstrap_port = Some(u16::MAX);
    facts.priority = i32::MIN;
    facts.reachable_from = Reachability::Gateway(GatewayId(u64::MAX));
    facts.signals = Signals {
        events: false,
        loads: false,
        scrape_interval_ms: u32::MAX,
    };
    facts.health = HealthPolicy {
        failure_threshold: u32::MAX,
        success_threshold: u32::MAX,
        check_interval_ms: u32::MAX,
        timeout_ms: u32::MAX,
        probes: true,
    };
    facts.overload = Overload {
        kv_usage_permille: Some(u16::MAX),
        waiting_requests: Some(u32::MAX),
    };
    facts.demand_facts = DemandFacts {
        runtime: u32::MAX,
        references: false,
        host_pipeline: false,
        window: u32::MAX,
        decoding: u8::MAX,
    };
    row
}

/// A page's paging counts the bytes its JSON takes, to the byte: each row
/// as the page writes it, the commas between two rows and the page's own
/// bytes around them, with `truncated` at its longer word. A short row, a
/// pair's with an encode offer, a provider's with a credential in its two
/// addresses and in its key, a row of texts JSON escapes and a row with
/// every number at its longest count so, each alone and all in one page.
/// So does a page with no row. A page that left a row out writes one byte
/// fewer than its count.
#[test]
fn a_page_counts_its_json_to_the_byte() {
    use std::time::Duration;

    let admitted = Instant::now();
    // An age of twenty digits, a number's longest: the take stands 10^16
    // seconds after the admission, more than a narrow instant holds.
    let taken = admitted
        .checked_add(Duration::from_secs(10_000_000_000_000_000))
        .expect("an instant that holds 10^16 seconds more");
    let aged = serde_json::to_value(RowJson::of(&short(1, admitted), taken)).expect("a row's JSON");
    assert_eq!(aged["admitted_age_ms"], 10_000_000_000_000_000_000_u64);
    let mut provided = short(3, admitted);
    provided.facts.key = "grpc://user:secret@10.0.0.5:8000".to_owned();
    provided.facts.engine = provided.facts.key.clone();
    provided.maker = Registrant::provider("finder", "http://u:p@host/x").expect("a maker");
    let kinds = [
        ("a short row", short(1, admitted)),
        ("a pair's", paired(2, admitted)),
        ("a provider's", provided),
        ("texts JSON escapes", escaped(4, admitted)),
        ("numbers at their longest", longest(5, admitted)),
    ];
    let counted = |rows: Vec<WorkerRow>, limit| {
        let asked = WorkerRowsQuery {
            models: Vec::new(),
            limit,
        };
        let serving = rows.len();
        let (page, bytes) = counted_page(&asked, Rows::Told(rows, serving), taken);
        let json = serde_json::to_vec(&page).expect("a page's JSON");
        (page, bytes, json.len())
    };
    for (name, row) in &kinds {
        let (page, bytes, written) = counted(vec![row.clone()], usize::MAX);
        assert!(page.rows.len() == 1 && !page.truncated, "{name}");
        assert_eq!(bytes, written, "{name}");
    }
    let all: Vec<WorkerRow> = kinds.into_iter().map(|(_, row)| row).collect();
    let (page, bytes, written) = counted(all.clone(), usize::MAX);
    assert!(page.rows.len() == all.len() && !page.truncated);
    assert_eq!(bytes, written);
    let (page, bytes, written) = counted(Vec::new(), usize::MAX);
    assert!(page.rows.is_empty() && !page.truncated);
    assert_eq!(bytes, written);

    // Two rows of the five: the page writes `true` where its count holds
    // `false`.
    let (page, bytes, written) = counted(all, 2);
    assert!(page.rows.len() == 2 && page.truncated);
    assert_eq!(bytes, written + 1);
}

/// A full page holds the rows whose JSON fits, but for a row that would
/// end the cut page's JSON on the bound's last byte: the page is cut, its
/// count is its JSON's length and one, the count is inside the page's
/// bytes, and the next row would have passed them. Short rows, rows that
/// state every number at its longest, rows of a pair and rows of texts
/// JSON escapes each fill a page so.
#[test]
fn a_full_page_holds_the_rows_whose_json_fits() {
    let now = Instant::now();
    let asked = WorkerRowsQuery {
        models: Vec::new(),
        limit: usize::MAX,
    };
    let kinds: [(&str, u64, Kind); 4] = [
        ("short rows", 5_000, short),
        ("stated numbers at their longest", 5_000, longest),
        ("rows of a pair", 5_000, paired),
        ("texts JSON escapes", 500, escaped),
    ];
    for (name, count, make) in kinds {
        let rows: Vec<WorkerRow> = (0..count).map(|n| make(n, now)).collect();
        let serving = rows.len();
        let (page, counted) = counted_page(&asked, Rows::Told(rows, serving), now);
        let bytes = serde_json::to_vec(&page).expect("a page's JSON").len();
        assert!(page.truncated, "{name}: {} rows", page.rows.len());
        // The page writes `true` where its count holds `false`: a row that
        // did not fit adds nothing to the count.
        assert_eq!(counted, bytes + 1, "{name}");
        assert!(counted <= PAGE_BYTES, "{name}: {counted} bytes");
        // The first row left out, as the page would write it behind a
        // comma.
        let held = u64::try_from(page.rows.len()).expect("a count of rows");
        let next = serde_json::to_vec(&RowJson::of(&make(held, now), now)).expect("a row's JSON");
        assert!(
            counted + 1 + next.len() > PAGE_BYTES,
            "{name}: a row of {} bytes fits behind {counted}",
            next.len()
        );
    }
}

/// A shared cut pages by the count and the rule a lone read's rows page
/// by. The count is its page's JSON with `truncated` at its longer word,
/// the commas and the page's own bytes with it. So a page that left no row
/// out counts its JSON to the byte, with five rows, with four and with
/// none, and a page that left a row out, at the call's limit or at the
/// bound, writes one byte fewer than its count. A row told without its
/// credential counts as it is told, and a row that serves no model of the
/// call adds nothing. The first row must fit the bound too.
#[test]
fn a_shared_cut_counts_its_json_to_the_byte() {
    use crate::store::Position;

    let now = Instant::now();
    // A row with a credential in its two addresses.
    let mut secret = short(3, now);
    secret.facts.key = "grpc://user:secret@10.0.0.5:8000".to_owned();
    secret.facts.engine = secret.facts.key.clone();
    let mut other = paired(4, now);
    other.facts.models = vec!["other".to_owned()];
    let shared = |rows| {
        Arc::new(StoreCheckpoint {
            at: Position::default(),
            epoch: 1,
            rows,
        })
    };
    // Five rows by key, as a cut holds them.
    let cut = shared(vec![
        secret,
        short(1, now),
        paired(2, now),
        other,
        longest(5, now),
    ]);
    let counted = |cut: &Arc<StoreCheckpoint>, models: &[&str], limit: usize| {
        let asked = WorkerRowsQuery {
            models: models.iter().map(|&model| model.to_owned()).collect(),
            limit,
        };
        let (page, bytes) = counted_page(&asked, Rows::Shared(Arc::clone(cut)), now);
        let json = serde_json::to_vec(&page).expect("a page's JSON");
        (page, bytes, json.len())
    };

    let (page, bytes, written) = counted(&cut, &[], usize::MAX);
    assert!(page.rows.len() == 5 && !page.truncated);
    assert_eq!(bytes, written);
    // The first row is told, and so counted, without its credential.
    let told = &page.rows[0].facts;
    assert_eq!(
        (told.key.as_str(), told.engine.as_str()),
        ("grpc://10.0.0.5:8000", "grpc://10.0.0.5:8000")
    );

    // The fourth row serves another model: no byte and no comma for it.
    let (page, bytes, written) = counted(&cut, &["m"], usize::MAX);
    assert!(page.rows.len() == 4 && !page.truncated);
    assert!(page.rows.iter().all(|row| row.facts.models == ["m"]));
    assert_eq!(bytes, written);

    // No row serves the model: the count is the page's own bytes.
    let (page, bytes, written) = counted(&cut, &["none"], usize::MAX);
    assert!(page.rows.is_empty() && !page.truncated);
    assert_eq!(bytes, written);

    // Two rows of the five: the page writes `true` where its count holds
    // `false`.
    let (page, bytes, written) = counted(&cut, &[], 2);
    assert!(page.rows.len() == 2 && page.truncated);
    assert_eq!(bytes, written + 1);

    // A first row past the bound: the page is cut with no row, as a lone
    // read's is.
    let mut large = short(1, now);
    large.facts.key = "x".repeat(2 * 1024 * 1024);
    let (page, bytes, written) = counted(&shared(vec![large]), &[], usize::MAX);
    assert!(page.rows.is_empty() && page.truncated);
    assert_eq!(bytes, written + 1);
}

/// A page may take the bound, 2 MiB, to its last byte: three rows whose
/// JSON ends on that byte are all held, and the page's JSON is the bound
/// long. A cut page keeps that byte free, since its count holds
/// `truncated` at its longer word: with a fourth row behind them the three
/// are held still, in a page one byte short of the bound. One byte more in
/// the third row leaves that row out, with a fourth row behind it or none,
/// though a page cut behind the third would end on the bound's last byte.
#[test]
fn a_page_may_end_on_the_bounds_last_byte() {
    // The bound by its number, and not by the constant the paging reads.
    let bound = 2 * 1024 * 1024;
    let now = Instant::now();
    let asked = WorkerRowsQuery {
        models: Vec::new(),
        limit: usize::MAX,
    };
    // Two short rows and a third whose key holds a path too: a `/` and
    // `pad` bytes.
    let rows = |pad: usize| {
        let mut last = short(3, now);
        last.facts.key = format!("http://10.0.0.3:8000/{}", "x".repeat(pad));
        vec![short(1, now), short(2, now), last]
    };
    // The ids of the rows a page holds and whether it is cut, then its
    // count and its JSON's length.
    let counted = |rows: Vec<WorkerRow>| {
        let serving = rows.len();
        let (page, bytes) = counted_page(&asked, Rows::Told(rows, serving), now);
        let json = serde_json::to_vec(&page).expect("a page's JSON");
        let held: Vec<u64> = page.rows.iter().map(|row| row.id.0).collect();
        ((held, page.truncated), (bytes, json.len()))
    };
    let (_, (unpadded, _)) = counted(rows(0));
    let pad = bound - unpadded;

    let (held, lengths) = counted(rows(pad));
    assert_eq!(held, (vec![1, 2, 3], false));
    assert_eq!(lengths, (bound, bound));

    // A fourth row behind them: the page is cut, a byte short of the bound.
    let mut four = rows(pad);
    four.push(short(4, now));
    let (held, lengths) = counted(four);
    assert_eq!(held, (vec![1, 2, 3], true));
    assert_eq!(lengths, (bound, bound - 1));

    // One byte more in the third row.
    let (held, (bytes, written)) = counted(rows(pad + 1));
    assert_eq!(held, (vec![1, 2], true));
    assert_eq!(bytes, written + 1);

    let mut four = rows(pad + 1);
    four.push(short(4, now));
    let (held, _) = counted(four);
    assert_eq!(held, (vec![1, 2], true));
    // The page cut behind the third row, as it would write itself.
    let of_three = WorkerRowsPage {
        rows: rows(pad + 1),
        truncated: true,
        taken: now,
    };
    let json = serde_json::to_vec(&of_three).expect("a page's JSON");
    assert_eq!(json.len(), bound);
}

/// A count that failed reads as the largest number, past the page's
/// bound. The JSON writer refuses a map keyed by a pair of numbers, and
/// the count of one reads so. Where the first row must fit the bound too,
/// as in an operator's page, a row measured so is left out: its page is
/// cut there and counts nothing for it, and the rows ahead of it stay.
#[test]
fn a_count_that_failed_is_past_the_pages_bound() {
    let refused: BTreeMap<(u8, u8), u8> = [((1, 2), 3)].into();
    assert!(serde_json::to_vec(&refused).is_err());
    assert_eq!(written(&refused), usize::MAX);

    let now = Instant::now();
    let asked = WorkerRowsQuery {
        models: Vec::new(),
        limit: usize::MAX,
    };
    let first = written(&RowJson::of(&short(1, now), now));
    // Three rows under a host's measure whose count of one row fails: the
    // ids of the rows the page holds, whether it is cut, and its count.
    let paged_with = |failing: u64| {
        let rows = vec![short(1, now), short(2, now), short(3, now)];
        let framing = Framing::default();
        paged(&asked, Rows::Told(rows, 3), now, false, framing, |row| {
            let bytes = if row.id.0 == failing {
                written(&refused)
            } else {
                written(&RowJson::of(&row, now))
            };
            (row.id.0, bytes)
        })
    };
    // The second row's count fails: the first row stays, alone.
    assert_eq!(paged_with(2), (vec![1], true, first));
    // The first row's count fails: no row stays.
    assert_eq!(paged_with(1), (Vec::new(), true, 0));
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
