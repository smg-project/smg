use super::{layout::*, *};
use crate::{
    document::spec::MAX_BLOCK_HASHES,
    engine::view::{AffinityHolder, AffinityRows, Header, MediaCounts, WorkerId},
    plugins::shared::config::HeaderSet,
};

fn candidate_row() -> Candidate {
    Candidate {
        position: 3,
        id: WorkerId(0x0102_0304_0506_0708),
        role: Role::Decode,
        dp_rank: 2,
        dp_size: 4,
        health: Health {
            healthy: true,
            overloaded: true,
            ..Health::default()
        },
        wire_modes: WireModes::GRPC.union(WireModes::ZMQ),
        engine_kind: 5,
        connector: 6,
        in_flight: 7,
        load: Some(LoadRow {
            running: 8,
            queued: 9,
            waiting_uncached_tokens: 10,
            kv_usage_permille: 11,
            active_kv_permille: 12,
            live_rate: 13,
            resident_tokens: 14,
            resident_counted: true,
            report_age_ms: 15,
            source: SignalSource::Relay,
            max_running: 16,
        }),
        load_sampled_at_micros: 99,
        session_hit: true,
        session_in_flight: 19,
        pair_class: 20,
        has_capacity: false,
        has_room: true,
        zone: 23,
        labels: 0xff00,
        ec_compatible: true,
        vision: false,
        host_local: true,
        active_sessions: 24,
        key: 25,
        engine: 27,
    }
}

/// The affinity facts the host holds for [`candidate_row`], beside it.
fn candidate_held() -> AffinityFacts {
    AffinityFacts {
        overlap_blocks: 17,
        predicted_blocks: 18,
        admitted_age_ms: 21,
        indexed_growth: 22,
        indexed_blocks: 26,
    }
}

fn le_u32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]])
}

fn le_u16(buf: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([buf[at], buf[at + 1]])
}

fn le_u64(buf: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[at..at + 8]);
    u64::from_le_bytes(b)
}

#[test]
fn the_record_row_carries_every_field_and_the_flags_by_name() {
    let c = candidate_row();
    let r = candidate(&c, candidate_held(), c.in_flight);
    assert_eq!(r.id, c.id.0);
    assert_eq!(r.role, wit::Role::Decode);
    assert!(r
        .health
        .contains(wit::Health::HEALTHY | wit::Health::OVERLOADED));
    assert!(!r.health.contains(wit::Health::DRAINING));
    assert!(r
        .wire_modes
        .contains(wit::WireModes::GRPC | wit::WireModes::ZMQ));
    assert!(!r.wire_modes.contains(wit::WireModes::HTTP));
    let l = r.load.expect("the row has a load");
    assert_eq!(
        (l.running, l.source, l.max_running),
        (8, wit::SignalSource::Relay, 16)
    );
    assert_eq!(
        (r.labels, r.key, r.active_sessions, r.indexed_blocks),
        (0xff00, 25, 24, 26)
    );
    assert_eq!(
        (
            r.overlap_blocks,
            r.predicted_blocks,
            r.admitted_age_ms,
            r.indexed_growth
        ),
        (17, 18, 21, 22)
    );
    assert_eq!(health_bits(c.health), 0b1_0001);
}

/// The record fill reads each row's entry of the request's column by the
/// row's index, as the packed fill does.
#[test]
fn the_record_fill_gives_each_row_its_own_entry_of_the_column() {
    let column = [AffinityFacts::default(), candidate_held()];
    let facts = RequestFacts {
        affinity: &column,
        ..RequestFacts::default()
    };
    let mut records = Records::default();
    records.fill(
        &facts,
        &[Candidate::default(), candidate_row()],
        &PoolFacts::default(),
    );
    let held = |r: &wit::Candidate| {
        (
            r.overlap_blocks,
            r.predicted_blocks,
            r.admitted_age_ms,
            r.indexed_growth,
            r.indexed_blocks,
        )
    };
    assert_eq!(held(&records.candidates[0]), (0, 0, 0, 0, 0));
    assert_eq!(held(&records.candidates[1]), (17, 18, 21, 22, 26));
}

/// Rows a host named in the column's place reach a guest as a column
/// would, on the record form and the packed one: a holder's row carries
/// its depth and every other row nothing.
#[test]
fn both_fills_give_a_named_holder_its_depth_and_every_other_row_nothing() {
    let facts = RequestFacts {
        named: Some(AffinityRows {
            holders: &[AffinityHolder {
                row: 1,
                overlap_blocks: 17,
                predicted_blocks: 0,
            }],
            ..AffinityRows::default()
        }),
        ..RequestFacts::default()
    };
    let rows = [Candidate::default(), candidate_row(), Candidate::default()];
    let mut records = Records::default();
    records.fill(&facts, &rows, &PoolFacts::default());
    let record: Vec<u32> = records
        .candidates
        .iter()
        .map(|r| r.overlap_blocks)
        .collect();
    assert_eq!(record, vec![0, 17, 0]);
    let mut packed = Packed::default();
    packed.fill_candidates(&facts, &rows);
    let packed: Vec<u32> = packed
        .candidates
        .as_chunks::<ROW>()
        .0
        .iter()
        .map(|row| le_u32(row, ROW_OVERLAP))
        .collect();
    assert_eq!(packed, vec![0, 17, 0]);
}

/// Both fills carry each candidate's requests in flight as the
/// request's facts read them: the row's own count.
#[test]
fn both_fills_give_a_row_the_count_its_request_reads() {
    let mut rows = [Candidate::default(), candidate_row(), Candidate::default()];
    for (row, position) in rows.iter_mut().zip(0..) {
        row.position = position;
    }
    rows[2].in_flight = 3;
    let facts = RequestFacts::default();
    let mut records = Records::default();
    records.fill(&facts, &rows, &PoolFacts::default());
    let record: Vec<u32> = records.candidates.iter().map(|r| r.in_flight).collect();
    assert_eq!(record, vec![0, 7, 3]);
    let mut packed = Packed::default();
    packed.fill_candidates(&facts, &rows);
    let packed: Vec<u32> = packed
        .candidates
        .as_chunks::<ROW>()
        .0
        .iter()
        .map(|row| le_u32(row, ROW_IN_FLIGHT))
        .collect();
    assert_eq!(packed, vec![0, 7, 3]);
}

/// Beside rows a host named in the column's place, the packed fill
/// carries each candidate's requests in flight: the row's own count.
#[test]
fn the_packed_fill_gives_a_row_its_count_beside_named_rows() {
    let facts = RequestFacts {
        named: Some(AffinityRows {
            holders: &[AffinityHolder {
                row: 1,
                overlap_blocks: 17,
                predicted_blocks: 0,
            }],
            ..AffinityRows::default()
        }),
        ..RequestFacts::default()
    };
    let mut rows = [Candidate::default(), candidate_row(), Candidate::default()];
    rows[2].in_flight = 3;
    let mut packed = Packed::default();
    packed.fill_candidates(&facts, &rows);
    let counts: Vec<u32> = packed
        .candidates
        .as_chunks::<ROW>()
        .0
        .iter()
        .map(|row| le_u32(row, ROW_IN_FLIGHT))
        .collect();
    assert_eq!(counts, vec![0, 7, 3]);
}

/// Both fills carry each candidate's count of the request's session
/// key, the row's own, and the two forms agree.
#[test]
fn both_fills_give_a_row_the_keys_count_its_request_reads() {
    let mut rows = [Candidate::default(), candidate_row(), Candidate::default()];
    for (row, position) in rows.iter_mut().zip(0..) {
        row.position = position;
    }
    let facts = RequestFacts::default();
    let mut records = Records::default();
    records.fill(&facts, &rows, &PoolFacts::default());
    let record = records.candidates.iter().map(|r| r.session_in_flight);
    let record: Vec<u16> = record.collect();
    let mut packed = Packed::default();
    packed.fill_candidates(&facts, &rows);
    let table = packed.candidates.as_chunks::<ROW>().0.iter();
    let table: Vec<u16> = table
        .map(|row| le_u16(row, ROW_SESSION_IN_FLIGHT))
        .collect();
    assert_eq!(record, table, "the two forms");
    assert_eq!(record, [0, 19, 0], "the rows' own");
}

#[test]
fn the_packed_row_puts_every_field_at_the_tables_offset() {
    let c = candidate_row();
    let mut packed = Packed::default();
    // The second row's affinity facts are the second entry of the
    // request's column: one index reads the row and its entry.
    let column = [AffinityFacts::default(), candidate_held()];
    let facts = RequestFacts {
        affinity: &column,
        ..RequestFacts::default()
    };
    packed.fill_candidates(&facts, &[Candidate::default(), c]);
    assert_eq!(packed.candidates.len(), 2 * ROW);
    let row = &packed.candidates[ROW..];
    assert_eq!(le_u64(row, ROW_ID), c.id.0);
    assert_eq!(le_u32(row, ROW_IN_FLIGHT), 7);
    assert_eq!(le_u32(row, ROW_RUNNING), 8);
    assert_eq!(le_u32(row, ROW_QUEUED), 9);
    assert_eq!(le_u32(row, ROW_WAITING_UNCACHED), 10);
    assert_eq!(le_u32(row, ROW_OVERLAP), 17);
    assert_eq!(le_u32(row, ROW_PREDICTED), 18);
    assert_eq!(le_u32(row, ROW_RESIDENT), 14);
    assert_eq!(le_u32(row, ROW_PAIR_CLASS), 20);
    assert_eq!(le_u32(row, ROW_ADMITTED_AGE), 21);
    assert_eq!(le_u32(row, ROW_REPORT_AGE), 15);
    assert_eq!(le_u32(row, ROW_INDEXED_GROWTH), 22);
    assert_eq!(le_u32(row, ROW_LIVE_RATE), 13);
    assert_eq!(le_u32(row, ROW_MAX_RUNNING), 16);
    assert_eq!(le_u16(row, ROW_KV_USAGE), 11);
    assert_eq!(le_u16(row, ROW_ACTIVE_KV), 12);
    assert_eq!(le_u16(row, ROW_DP_RANK), 2);
    assert_eq!(le_u16(row, ROW_DP_SIZE), 4);
    assert_eq!(le_u16(row, ROW_ZONE), 23);
    assert_eq!(le_u16(row, ROW_ENGINE_KIND), 5);
    assert_eq!(le_u16(row, ROW_CONNECTOR), 6);
    assert_eq!(le_u16(row, ROW_SESSION_IN_FLIGHT), 19);
    assert_eq!(row[ROW_HEALTH], 0b1_0001);
    assert_eq!(row[ROW_WIRE_MODES], 0b110);
    assert_eq!(row[ROW_ROLE], 2);
    assert_eq!(row[ROW_SOURCE], 3);
    assert_eq!(
        row[ROW_BITS],
        BIT_SESSION_HIT | BIT_HAS_ROOM | BIT_EC_COMPATIBLE | BIT_HOST_LOCAL | BIT_LOAD_PRESENT
    );
    assert_eq!(row[ROW_BITS + 1], 0);
    assert_eq!(le_u16(row, ROW_ACTIVE_SESSIONS), 24);
    assert_eq!(le_u32(row, ROW_KEY), 25);
    assert_eq!(le_u32(row, ROW_INDEXED_BLOCKS), 26);
    assert_eq!(le_u64(row, ROW_LABELS), 0xff00);
    let empty = &packed.candidates[..ROW];
    assert_eq!(
        empty[ROW_BITS] & BIT_LOAD_PRESENT,
        0,
        "no load row: the bit is clear"
    );
    // The first row reads the column's first entry, which holds nothing:
    // a fill that gave every row one entry would write the second's here.
    let held = [
        ROW_OVERLAP,
        ROW_PREDICTED,
        ROW_ADMITTED_AGE,
        ROW_INDEXED_GROWTH,
        ROW_INDEXED_BLOCKS,
    ]
    .map(|at| le_u32(empty, at));
    assert_eq!(held, [0; 5]);
}

/// An empty column is nothing held by any candidate: the five read zero
/// in every row, and the rest of each row is written.
#[test]
fn the_packed_fill_reads_an_empty_column_as_nothing_held() {
    let c = candidate_row();
    let mut packed = Packed::default();
    packed.fill_candidates(&RequestFacts::default(), &[c, c]);
    assert_eq!(packed.candidates.len(), 2 * ROW);
    for row in packed.candidates.as_chunks::<ROW>().0 {
        assert_eq!(le_u64(row, ROW_ID), c.id.0);
        let held = [
            ROW_OVERLAP,
            ROW_PREDICTED,
            ROW_ADMITTED_AGE,
            ROW_INDEXED_GROWTH,
            ROW_INDEXED_BLOCKS,
        ]
        .map(|at| le_u32(row, at));
        assert_eq!(held, [0; 5]);
    }
}

/// A column that holds an entry holds one a candidate. One that ends
/// before the candidates is a host's mistake and fails the fill, as an
/// index past its end does: a fill that stopped where the column ends
/// would hand the guest rows of zeros for the candidates left.
#[test]
#[should_panic(expected = "out of range")]
fn the_packed_fill_fails_on_a_column_that_ends_before_the_candidates() {
    let column = [candidate_held()];
    let facts = RequestFacts {
        affinity: &column,
        ..RequestFacts::default()
    };
    Packed::default().fill_candidates(&facts, &[Candidate::default(), candidate_row()]);
}

/// Both forms carry the request's block hashes for a row that declares
/// them, in the token form or the text form, the first
/// [`MAX_BLOCK_HASHES`] of a longer list, and none for a row that
/// declares neither, whatever else it declares.
#[test]
fn a_view_carries_the_block_hashes_its_row_declares() {
    let most = usize::try_from(MAX_BLOCK_HASHES).expect("the range fits a length");
    let longer: Vec<u64> = (0..=u64::from(MAX_BLOCK_HASHES)).collect();
    let facts = RequestFacts {
        block_hashes: &longer,
        ..RequestFacts::default()
    };
    let token = Needs {
        token_hashes: true,
        ..Needs::default()
    };
    let text = Needs {
        text_hashes: true,
        ..Needs::default()
    };
    let neither = Needs {
        text: true,
        loads: true,
        events: true,
        affinity: true,
        ..Needs::default()
    };
    for (needs, carried) in [
        (Needs::default(), 0),
        (neither, 0),
        (token, most),
        (text, most),
    ] {
        let record = request(&facts, &Exposure::default(), needs, &[], &[]);
        assert_eq!(record.block_hashes, &longer[..carried], "{needs:?}");
        let mut packed = Packed::default();
        packed
            .fill_request(&facts, &Exposure::default(), needs)
            .expect("the request packs");
        let b = &packed.request;
        assert_eq!(usize::from(le_u16(b, REQ_HASH_COUNT)), carried, "{needs:?}");
        // No name, no tried id, no header: the hashes are the block's tail.
        assert_eq!(b.len(), HEADER + 8 * carried, "{needs:?}");
        // Every hash at its own eight bytes, in the list's order.
        let read: Vec<u64> = (0..carried).map(|i| le_u64(b, HEADER + 8 * i)).collect();
        assert_eq!(read, &longer[..carried], "{needs:?}");
    }
    // A list inside the range is carried whole.
    let short = RequestFacts {
        block_hashes: &longer[..3],
        ..RequestFacts::default()
    };
    let record = request(&short, &Exposure::default(), token, &[], &[]);
    assert_eq!(record.block_hashes, &longer[..3]);
}

/// Both forms carry the request's size for a row that declares it, and
/// zero for a row that does not, whatever else it declares. A plugin
/// that reads the size undeclared reads zero at once: of a request its
/// host read whole as of one its host sent on unread.
#[test]
fn a_view_carries_the_requests_size_its_row_declares() {
    let facts = RequestFacts {
        prompt_tokens: 512,
        ..RequestFacts::default()
    };
    let sized = Needs {
        prompt_size: true,
        ..Needs::default()
    };
    let other = Needs {
        token_hashes: true,
        text_hashes: true,
        loads: true,
        ..Needs::default()
    };
    for (needs, shown) in [(sized, 512), (other, 0), (Needs::default(), 0)] {
        let record = request(&facts, &Exposure::default(), needs, &[], &[]);
        assert_eq!(record.prompt_tokens, shown, "{needs:?}");
        let mut packed = Packed::default();
        packed
            .fill_request(&facts, &Exposure::default(), needs)
            .expect("the request packs");
        assert_eq!(
            le_u32(&packed.request, REQ_PROMPT_TOKENS),
            shown,
            "{needs:?}"
        );
    }
    // A request of text alone is shown its text's bytes over four:
    // 1,024 two-byte characters are 2,048 bytes.
    let worded = RequestFacts {
        prompt_chars: 1_024,
        prompt_bytes: 2_048,
        ..RequestFacts::default()
    };
    let record = request(&worded, &Exposure::default(), sized, &[], &[]);
    assert_eq!(record.prompt_tokens, 512);
}

/// The two u64 sections are written a section at a time: every hash and
/// every tried id sits at its own eight bytes, little-endian and in
/// order, the tried ids after the hashes, at any length of either, in
/// a block that held a longer request before.
#[test]
fn the_packed_sections_hold_every_value_at_its_own_eight_bytes() {
    let value = |i: u64| (i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let needs = Needs {
        token_hashes: true,
        ..Needs::default()
    };
    let mut packed = Packed::default();
    for (hash_count, tried_count) in [(257, 2), (0, 0), (1, 0), (0, 1), (3, 5)] {
        let hashes: Vec<u64> = (0..hash_count).map(value).collect();
        let tried: Vec<WorkerId> = (0..tried_count)
            .map(|i| WorkerId(value(1_000 + i)))
            .collect();
        let facts = RequestFacts {
            model: "m",
            block_hashes: &hashes,
            tried: &tried,
            ..RequestFacts::default()
        };
        packed
            .fill_request(&facts, &Exposure::default(), needs)
            .expect("the request packs");
        let b = &packed.request;
        // A name of one byte: the sections start at the next multiple of 8.
        let at = (HEADER + 1).next_multiple_of(8);
        assert_eq!(b.len(), at + 8 * (hashes.len() + tried.len()));
        let read: Vec<u64> = (at..b.len()).step_by(8).map(|i| le_u64(b, i)).collect();
        let wanted: Vec<u64> = hashes
            .iter()
            .copied()
            .chain(tried.iter().map(|t| t.0))
            .collect();
        assert_eq!(read, wanted, "{hash_count} hashes, {tried_count} tried ids");
    }
}

#[test]
fn a_twelve_byte_model_name_puts_the_hashes_at_ninety_six() {
    // The WIT's own example of the alignment rule.
    let hashes = [0x11u64, 0x22];
    let tried = [WorkerId(0x33)];
    let headers = [
        Header {
            name: 2,
            value: "exposed",
        },
        Header {
            name: 9,
            value: "hidden",
        },
    ];
    let facts = RequestFacts {
        model: "twelve-bytes",
        model_fingerprint: crate::engine::ModelFingerprint::NONE,
        namespace_seed: 1,
        tenant: Identity::plain("acme"),
        priority: -7,
        streaming: true,
        max_output_tokens: 64,
        prompt_tokens: 128,
        block_hashes: &hashes,
        hash_kind: HashKind::Text,
        session_key: Identity::plain("s-1"),
        phase: Phase::Prefill,
        attempt: 2,
        tried: &tried,
        budget_micros: 500,
        headers: &headers,
        media: MediaCounts {
            images: 1,
            audio: 2,
            video: 3,
            bytes: 4,
        },
        seed: 9,
        block_size: 16,
        prompt_chars: 0,
        prompt_bytes: 0,
        now_micros: 0,
        session_counted: false,
        // Host-only: the request block carries no affinity column and
        // no rows named in its place.
        affinity: &[],
        named: None,
        signal_cells: None,
        client_key: 0,
        ring: None,
        ring_positions: &[],
        decode_ring_positions: None,
        session_key_from_rid: false,
    };
    let exposure = Exposure {
        tenant: Reveal::Plain,
        session_key: Reveal::Hash,
        headers: HeaderSet::of([2]),
    };
    // The row declares the hashes this request carries.
    let text_hashes = Needs {
        text_hashes: true,
        ..Needs::default()
    };
    let mut packed = Packed::default();
    packed
        .fill_request(&facts, &exposure, text_hashes)
        .expect("the request packs");
    let b = &packed.request;
    assert_eq!(le_u16(b, REQ_MODEL_LEN), 12);
    assert_eq!(&b[HEADER..HEADER + 12], b"twelve-bytes");
    assert_eq!(le_u64(b, 96), 0x11);
    assert_eq!(le_u64(b, 104), 0x22);
    assert_eq!(le_u64(b, 112), 0x33, "the tried ids follow the hashes");
    // One exposed header: (name u8, len u16, bytes).
    assert_eq!(b[120], 2);
    assert_eq!(le_u16(b, 121), 7);
    assert_eq!(&b[123..130], b"exposed");
    assert_eq!(le_u16(b, REQ_HEADER_COUNT), 1);
    // The plain tenant follows the headers; the session key stays a hash.
    assert_eq!(le_u16(b, 130), 4);
    assert_eq!(&b[132..136], b"acme");
    assert_eq!(b.len(), 136);
    assert_eq!(le_u64(b, REQ_TENANT), facts.tenant.hash);
    assert_eq!(
        le_u64(b, REQ_SESSION_KEY),
        facts.session().get(),
        "the session's value, as a lease of the request carries it"
    );
    assert_eq!(
        b[REQ_BITS],
        REQ_BIT_STREAMING | REQ_BIT_TENANT_PLAIN,
        "streaming and the plain tenant; no plain session key"
    );
    assert_eq!(le_u32(b, REQ_BUDGET), 500);
    assert_eq!(le_u32(b, REQ_BLOCK_SIZE), 16);
    assert_eq!(le_u32(b, REQ_PRIORITY), (-7i32) as u32);
    assert_eq!((b[REQ_PHASE], b[REQ_HASH_KIND], b[REQ_ATTEMPT]), (1, 1, 2));
    assert_eq!(le_u16(b, REQ_MEDIA_VIDEO), 3);
    assert_eq!(le_u32(b, REQ_MEDIA_BYTES), 4);
}

#[test]
fn the_record_request_carries_the_block_size_beside_the_seed() {
    let facts = RequestFacts {
        model: "m",
        seed: 9,
        block_size: 16,
        ..RequestFacts::default()
    };
    let record = request(&facts, &Exposure::default(), Needs::default(), &[], &[]);
    assert_eq!((record.seed, record.block_size), (9, 16));
}

#[test]
fn a_withheld_identity_packs_as_zero_and_the_record_form_agrees() {
    let facts = RequestFacts {
        model: "m",
        tenant: Identity::plain("acme"),
        session_key: Identity::hashed(77),
        ..RequestFacts::default()
    };
    let exposure = Exposure {
        tenant: Reveal::None,
        session_key: Reveal::Plain,
        headers: HeaderSet::NONE,
    };
    let mut packed = Packed::default();
    packed
        .fill_request(&facts, &exposure, Needs::default())
        .expect("the request packs");
    assert_eq!(le_u64(&packed.request, REQ_TENANT), 0);
    assert_eq!(
        le_u64(&packed.request, REQ_SESSION_KEY),
        facts.session().get(),
        "plain asked, hash held: the session's value crosses"
    );
    assert_eq!(packed.request[REQ_BITS], 0);
    assert_eq!(identity(&facts.tenant, Reveal::None), wit::Identity::None);
    assert_eq!(
        session_identity(&facts, Reveal::Plain),
        wit::Identity::Hash(facts.session().get())
    );
    assert_eq!(
        identity(&facts.tenant, Reveal::Plain),
        wit::Identity::Plain("acme")
    );
    assert_eq!(
        identity(&Identity::NONE, Reveal::Plain),
        wit::Identity::None
    );
}

/// The exposure is asked before the session is made: a plugin that is
/// not shown the session key costs the request no session, and neither
/// does a plain key in the record form. The carried fingerprint here is
/// another name's, which stops a debug build wherever the session is
/// made, so no form below made one.
#[test]
fn a_session_key_the_plugin_is_not_shown_makes_no_session() {
    let facts = RequestFacts {
        model: "model-a",
        model_fingerprint: crate::engine::ModelFingerprint::of("model-b"),
        session_key: Identity::plain("conversation-7"),
        ..RequestFacts::default()
    };
    let withheld = Exposure {
        session_key: Reveal::None,
        ..Exposure::default()
    };
    let record = request(&facts, &withheld, Needs::default(), &[], &[]);
    assert_eq!(record.session_key, wit::Identity::None);
    let mut packed = Packed::default();
    packed
        .fill_request(&facts, &withheld, Needs::default())
        .expect("the request packs");
    assert_eq!(le_u64(&packed.request, REQ_SESSION_KEY), 0);
    let plain = Exposure {
        session_key: Reveal::Plain,
        ..Exposure::default()
    };
    let record = request(&facts, &plain, Needs::default(), &[], &[]);
    assert_eq!(record.session_key, wit::Identity::Plain("conversation-7"));
}

/// A plain session key under the plain exposure packs the session's
/// value beside its string: the number a lease of the request carries,
/// so a plugin shown the key matches the two. The key's own fingerprint
/// is another number, and the case tells them apart.
#[test]
fn a_plain_session_key_packs_the_sessions_value_beside_its_string() {
    let facts = RequestFacts {
        model: "m",
        session_key: Identity::plain("conversation-7"),
        ..RequestFacts::default()
    };
    let exposure = Exposure {
        session_key: Reveal::Plain,
        ..Exposure::default()
    };
    let mut packed = Packed::default();
    packed
        .fill_request(&facts, &exposure, Needs::default())
        .expect("the request packs");
    let b = &packed.request;
    let session = facts.session().get();
    assert_ne!(session, facts.session_key.hash);
    assert_eq!(le_u64(b, REQ_SESSION_KEY), session);
    assert_eq!(b[REQ_BITS], REQ_BIT_SESSION_PLAIN);
    // The string follows the model's padded section: (len u16, bytes).
    assert_eq!(le_u16(b, HEADER + 8), 14);
    assert_eq!(&b[HEADER + 10..], b"conversation-7");
}

#[test]
fn a_name_past_the_u16_bound_is_too_long_for_the_packed_form() {
    let long = "x".repeat(usize::from(u16::MAX) + 1);
    let facts = RequestFacts {
        model: &long,
        ..RequestFacts::default()
    };
    let mut packed = Packed::default();
    assert_eq!(
        packed.fill_request(&facts, &Exposure::default(), Needs::default()),
        Err(TooLong)
    );
}

#[test]
fn the_pool_block_lists_its_zones_after_the_header() {
    let zones = [
        ZoneCapacity {
            zone: 1,
            healthy: 3,
            total: 4,
        },
        ZoneCapacity {
            zone: 2,
            healthy: 0,
            total: 1,
        },
    ];
    let pool = PoolFacts {
        pool_size: 8,
        membership_epoch: 5,
        ring_epoch: 6,
        regime: OverloadRegime::Partial,
        generation: 7,
        zones: &zones,
        reach: 0,
        reported: false,
        ..PoolFacts::default()
    };
    let mut packed = Packed::default();
    packed.fill_pool(&pool);
    let b = &packed.pool;
    assert_eq!(b.len(), POOL + 2 * ZONE);
    assert_eq!(le_u32(b, POOL_SIZE), 8);
    assert_eq!(b[POOL_REGIME], 1);
    assert_eq!(le_u64(b, POOL_GENERATION), 7);
    assert_eq!(le_u16(b, POOL_ZONE_COUNT), 2);
    assert_eq!(le_u16(b, POOL + ZONE + ZONE_ZONE), 2);
    assert_eq!(le_u32(b, POOL + ZONE_HEALTHY), 3);
    assert_eq!(le_u32(b, POOL + ZONE + ZONE_TOTAL), 1);
}

#[test]
fn a_vetos_instant_becomes_the_microseconds_it_has_left() {
    let veto = Signal::GatewayVeto {
        worker: WorkerId(4),
        until_micros: 1_500,
    };
    assert_eq!(signal(&veto, 1_000), wit::Signal::GatewayVeto((4, 500)));
    assert_eq!(signal(&veto, 2_000), wit::Signal::GatewayVeto((4, 0)));
}

/// A dispatch on another target of its pick is told with a lease that
/// names that target's index, and the guest's lease record carries it.
#[test]
fn a_lease_record_names_the_index_of_its_lease() {
    let told = Lease {
        target_index: 1,
        ..Lease::default()
    };
    assert_eq!(lease(&told, &Exposure::default()).target_index, 1);
}

/// The guest's lease record states the size its request record stated:
/// the count of the request's token ids, and for a request of text
/// alone its text's bytes over four, whatever its characters. A text
/// of 400 two-byte characters reads 200 in the request record and in
/// the lease record: a guest that books by the one and frees by the
/// other keeps its books.
#[test]
fn a_lease_record_states_the_size_its_request_record_stated() {
    let size = |tokens, chars, bytes| {
        let told = Lease {
            tokens,
            chars,
            bytes,
            ..Lease::default()
        };
        lease(&told, &Exposure::default()).tokens
    };
    let sizes = (size(7, 400, 800), size(0, 400, 800), size(0, 0, 0));
    assert_eq!(sizes, (7, 200, 0));
    let worded = RequestFacts {
        prompt_chars: 400,
        prompt_bytes: 800,
        ..RequestFacts::default()
    };
    let sized = Needs {
        prompt_size: true,
        ..Needs::default()
    };
    let record = request(&worded, &Exposure::default(), sized, &[], &[]);
    let held = size(0, worded.prompt_chars, worded.prompt_bytes);
    assert_eq!((record.prompt_tokens, held), (200, 200));
}

/// The guest's world carries the models a worker left as the case
/// `left`: the mapper hands a guest the worker's id with the names, and
/// the write's own word as before.
#[test]
fn a_guest_is_told_the_models_a_worker_left_as_the_case_left() {
    let (id, models) = (WorkerId(7), ["m".to_string()]);
    let left = Membership::Left {
        id,
        models: &models,
    };
    let (mut snapshot, mut names) = (Vec::new(), Vec::new());
    let word = membership(&left, &mut snapshot, &mut names);
    assert!(matches!(word, wit::Membership::Left((7, ["m"]))));
    let (mut snapshot, mut names) = (Vec::new(), Vec::new());
    let word = membership(&Membership::Relabelled(id), &mut snapshot, &mut names);
    assert!(matches!(word, wit::Membership::Relabelled(7)));
}
