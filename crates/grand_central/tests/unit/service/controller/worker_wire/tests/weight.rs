//! A row's weight on the wire and the store's measure of it, and the
//! bounds a row and a page of rows are held to.

use super::*;

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
    // The same row again where its worker states a priority, and then a
    // drain window too, each at its widest. The store charges the wire's
    // number for each, and the measure adds each where the row states it.
    let ranked = WorkerFacts {
        priority: i32::MIN,
        ..offering.clone()
    };
    let waiting = WorkerFacts {
        drain_settle_secs: Some(u32::MAX),
        ..ranked.clone()
    };
    assert_eq!(numeric(&ranked) - numeric(&offering), WorkerFacts::PRIORITY);
    assert_eq!(charged(&ranked) - charged(&offering), WorkerFacts::PRIORITY);
    assert_eq!(
        numeric(&waiting) - numeric(&ranked),
        WorkerFacts::DRAIN_WINDOW
    );
    assert_eq!(
        charged(&waiting) - charged(&ranked),
        WorkerFacts::DRAIN_WINDOW
    );

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
            (false, &waiting),
            (true, &waiting),
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
                let wire = weight(&facts) + super::super::maker_weight(maker);
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
