use super::*;

/// `RequestFacts::in_flight` returns the row's own count.
#[test]
fn a_rows_requests_in_flight_are_the_rows_own() {
    let row = |position, in_flight| Candidate {
        position,
        in_flight,
        ..Candidate::default()
    };
    let rows = [row(0, 5), row(1, 2)];
    let facts = RequestFacts::default();
    let read: Vec<u32> = rows.iter().map(|row| facts.in_flight(row)).collect();
    assert_eq!(read, [5, 2]);
}

#[test]
fn health_flags_compose_into_the_two_routability_reads() {
    let ready = Health {
        healthy: true,
        ..Health::default()
    };
    assert!(ready.routable());
    let overloaded = Health {
        overloaded: true,
        ..ready
    };
    assert!(!overloaded.routable());
    assert!(overloaded.routable_ignoring_overload());
    let open = Health {
        breaker_open: true,
        ..ready
    };
    assert!(!open.routable_ignoring_overload());
}

#[test]
#[expect(clippy::print_stdout, reason = "the case reports the candidate layout")]
fn a_candidate_row_is_two_lines_and_its_affinity_facts_20_bytes() {
    // The filters walk the rows once a filter, so the rows' stride is a
    // cost every policy pays: the pin is exact, 128 bytes at a 64-byte
    // alignment, and a change of any number here carries the crate's
    // decision bench (`benches/decision.rs`) where the pull-request gate
    // runs as well as on a build host: the two read a narrower stride in
    // opposite directions.
    let size = size_of::<Candidate>();
    let alignment = align_of::<Candidate>();
    println!("Candidate size {size} bytes, alignment {alignment} bytes");
    assert_eq!((size, alignment), (128, 64));
    assert_eq!(size_of::<AffinityFacts>(), 20);
}

/// A record made ready is one the health and admission filters keep
/// and an engine of its own, and every field it does not write is the
/// default's. A default record is dropped by three of those filters.
#[test]
fn a_ready_record_can_be_sent_to_and_is_an_engine_of_its_own() {
    use crate::{
        engine::framework::Filter,
        stages::{HasCapacity, HasRoom, Healthy, NotOverloaded},
    };

    let made = Candidate::ready(3, WorkerId(9), Role::Decode);
    assert_eq!(
        (made.position, made.id, made.role, made.engine),
        (3, WorkerId(9), Role::Decode, 3)
    );
    let (facts, cx) = (RequestFacts::default(), Constraints::default());
    let stock: [&dyn Filter; 4] = [&Healthy, &NotOverloaded, &HasCapacity, &HasRoom];
    let dropped = |record: &Candidate| {
        let refuses = stock
            .iter()
            .filter(|stage| !stage.keep(&facts, record, &cx));
        refuses.map(|stage| stage.name()).collect::<Vec<_>>()
    };
    assert!(dropped(&made).is_empty(), "{:?}", dropped(&made));
    let unset = Candidate::default();
    assert_eq!(dropped(&unset), ["healthy", "has-capacity", "has-room"]);

    let rest = Candidate {
        position: 0,
        id: WorkerId::default(),
        role: Role::default(),
        health: Health::default(),
        has_capacity: false,
        has_room: false,
        engine: 0,
        ..made
    };
    assert_eq!(format!("{rest:?}"), format!("{unset:?}"));
}

/// A request's facts are copied where a decision narrows or resumes
/// them, so their size is a cost of those decisions: the pin is exact.
/// The signal cells are one word, for every signal, and the client's
/// key is one word more. The ring a caller lends is two words, where
/// the request stands on it two more, and a pair's decode side's own
/// places there two more. A text's two lengths are one word: its
/// characters fill what was spare, and its bytes take the word. The
/// word that says the request's own id handed its session key is one
/// byte of what is spare. A field the facts gain beyond the spare
/// bytes moves this number and carries the decision bench's cells
/// that copy the facts (`benches/decision.rs`).
#[test]
fn a_requests_facts_take_the_pinned_bytes() {
    assert_eq!(size_of::<RequestFacts<'_>>(), 320);
}

#[test]
fn a_candidate_rows_fields_take_107_bytes() {
    // The row is written once per candidate per decision, so what its
    // fields take is a cost every policy pays. The alignment leaves
    // spare bytes, so the row's size no longer says it, and a field
    // added to the row would move no size. This case names every field
    // (a new one does not compile until it is named here) and pins
    // where the last one ends. A fact one stage family reads belongs in
    // a column beside the rows, as the affinity facts are. The plugin
    // view's packed row is 104 bytes WITH the five facts in it: nothing
    // ties that number to this one, and each is pinned in its own file.
    fn end<T>(at: usize, _field: &T) -> usize {
        at + size_of::<T>()
    }
    macro_rules! ends {
            ($($field:ident),* $(,)?) => {{
                let Candidate { $($field),* } = Candidate::default();
                [$(end(std::mem::offset_of!(Candidate, $field), &$field)),*]
            }};
        }
    let ends = ends!(
        position,
        id,
        role,
        dp_rank,
        dp_size,
        health,
        wire_modes,
        engine_kind,
        connector,
        in_flight,
        load,
        load_sampled_at_micros,
        session_hit,
        session_in_flight,
        pair_class,
        has_capacity,
        has_room,
        zone,
        labels,
        ec_compatible,
        vision,
        host_local,
        active_sessions,
        key,
        engine,
    );
    assert_eq!(ends.into_iter().max(), Some(107));
}

/// A request's size in blocks is its tokens over its block size: the size
/// its host counted with where the request states one, the view's where
/// it states none. It is never under one, and never under the block
/// hashes the host sent. A request of text states its length and no
/// count of tokens: its windows are bytes.
#[test]
fn a_requests_blocks_are_its_tokens_over_its_block_size_and_never_under_its_hashes() {
    let hashes = [1u64, 2, 3];
    let sized = |prompt_tokens: u32, block_size: u32, hashed: usize| RequestFacts {
        prompt_tokens,
        block_size,
        block_hashes: &hashes[..hashed],
        ..RequestFacts::default()
    };
    // 100 tokens at the view's 16 are six whole blocks.
    assert_eq!(sized(100, 0, 0).blocks(16), 6);
    assert_eq!(sized(100, 0, 0).block_size_or(16), 16);
    // At the size its host counted with, the view's is not read.
    assert_eq!(sized(100, 32, 0).blocks(16), 3);
    assert_eq!(sized(100, 32, 0).block_size_or(16), 32);
    // Under one block, and with no token counted: one block.
    assert_eq!(sized(5, 0, 0).blocks(16), 1);
    assert_eq!(sized(0, 0, 0).blocks(16), 1);
    // Never under the hashes the host sent, and over them by its tokens.
    assert_eq!(sized(20, 0, 3).blocks(16), 3);
    assert_eq!(sized(100, 0, 3).blocks(16), 6);
    // A text of 200 two-byte characters is 400 bytes: six windows of 64.
    // Read by its characters it would be three.
    let text = RequestFacts {
        hash_kind: HashKind::Text,
        prompt_chars: 200,
        prompt_bytes: 400,
        ..sized(0, 64, 0)
    };
    assert_eq!(text.blocks(16), 6);
}

#[test]
fn an_empty_affinity_column_reads_as_nothing_held_for_every_candidate() {
    let empty = RequestFacts::default();
    for index in [0, 1, 511] {
        assert_eq!(empty.affinity_at(index), AffinityFacts::default());
    }
}

#[test]
fn a_filled_affinity_column_reads_the_entry_at_the_candidates_index() {
    let column = [3, 5, 8].map(|overlap_blocks| AffinityFacts {
        overlap_blocks,
        ..AffinityFacts::default()
    });
    let facts = RequestFacts {
        affinity: &column,
        ..RequestFacts::default()
    };
    let read = [0, 1, 2].map(|index| facts.affinity_at(index).overlap_blocks);
    assert_eq!(read, [3, 5, 8]);
}

/// A column that holds an entry is never read as a default past its
/// end: zero is a real answer (a cold worker), so a forgiving read would
/// turn a column filled for another pool into a routing change.
#[test]
#[should_panic(expected = "index out of bounds")]
fn a_read_past_a_filled_affinity_column_is_not_a_default() {
    let column = [AffinityFacts::default()];
    let facts = RequestFacts {
        affinity: &column,
        ..RequestFacts::default()
    };
    let _ = facts.affinity_at(1);
}

/// A pass's walk of the rows named reads each row as the search does, in
/// whatever order the pass asks: ascending, a row twice, a row behind
/// the cursor.
#[test]
fn a_walk_of_the_rows_named_reads_each_row_as_the_search_does() {
    let holders = [(1, 4, 0), (4, 9, 2), (5, 1, 7), (9, 3, 0)].map(
        |(row, overlap_blocks, predicted_blocks)| AffinityHolder {
            row,
            overlap_blocks,
            predicted_blocks,
        },
    );
    let named = AffinityRows {
        holders: &holders,
        ..AffinityRows::default()
    };
    let orders: [&[u32]; 4] = [
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        &[1, 1, 4, 4, 9, 9],
        &[10, 9, 5, 4, 1, 0],
        &[5, 0, 9, 4, 4, 1, 7, 10, 5],
    ];
    for order in orders {
        let mut walk = named.walk();
        for &row in order {
            assert_eq!(walk.entry(row), named.entry(row), "row {row} of {order:?}");
        }
    }
    // No row named: nothing held at any row.
    let mut walk = AffinityRows::default().walk();
    assert_eq!(walk.entry(0), AffinityFacts::default());
    assert_eq!(walk.entry(3), AffinityFacts::default());
}

/// Behind a draw a pass reads the rows named by a search: each row reads
/// as the walk of a whole read reads it, and no read looks behind a
/// cursor, where the walk asked in that order does.
#[test]
fn a_read_behind_a_draw_searches_the_rows_named_and_keeps_no_cursor() {
    let holders = [(1, 4), (4, 9), (9, 3)].map(|(row, overlap_blocks)| AffinityHolder {
        row,
        overlap_blocks,
        predicted_blocks: 0,
    });
    let whole = AffinityRows {
        holders: &holders,
        ..AffinityRows::default()
    };
    let drawn = AffinityRows {
        mean_in_flight: Some(1.0),
        ..whole
    };
    assert!(drawn.drawn() && !whole.drawn());
    let feed = Feed::Named(drawn);
    let order = [9, 0, 4, 4, 1, 10, 9];
    LOOKS_BEHIND.with(|looks| looks.set(0));
    let mut reads = feed.reads();
    for row in order {
        assert_eq!(
            feed.read::<true>(&mut reads, row),
            whole.entry(row),
            "row {row}"
        );
    }
    assert_eq!(
        LOOKS_BEHIND.with(std::cell::Cell::get),
        0,
        "a search looked behind"
    );
    for row in order {
        assert_eq!(
            feed.read::<false>(&mut reads, row),
            whole.entry(row),
            "row {row}"
        );
    }
    assert!(
        LOOKS_BEHIND.with(std::cell::Cell::get) > 0,
        "the walk never looked behind"
    );
}

/// Rows named in the column's place read through the same accessor: a
/// named row reads what the host said it holds, by both readings, and
/// every other row nothing held. Beside a column the column is read.
#[test]
fn rows_named_in_the_columns_place_read_what_each_holds() {
    let holders = [
        AffinityHolder {
            row: 1,
            overlap_blocks: 9,
            predicted_blocks: 0,
        },
        AffinityHolder {
            row: 4,
            overlap_blocks: 3,
            predicted_blocks: 7,
        },
    ];
    let facts = RequestFacts {
        named: Some(AffinityRows {
            holders: &holders,
            ..AffinityRows::default()
        }),
        ..RequestFacts::default()
    };
    let read = [0, 1, 2, 4, 511].map(|index| {
        let held = facts.affinity_at(index);
        (held.overlap_blocks, held.predicted_blocks)
    });
    assert_eq!(read, [(0, 0), (9, 0), (0, 0), (3, 7), (0, 0)]);
    let nobody = RequestFacts {
        named: Some(AffinityRows::default()),
        ..RequestFacts::default()
    };
    assert_eq!(nobody.affinity_at(3), AffinityFacts::default());
    assert_eq!(
        RequestFacts::default().affinity_at(1),
        AffinityFacts::default()
    );
    // Beside a column the column is read.
    let column = [
        AffinityFacts::default(),
        AffinityFacts {
            overlap_blocks: 3,
            ..AffinityFacts::default()
        },
    ];
    let beside = RequestFacts {
        affinity: &column,
        ..facts
    };
    assert_eq!(beside.affinity_at(1).overlap_blocks, 3);
}

#[test]
fn the_load_row_holds_its_resident_count_flag_in_its_padding() {
    // 34 bytes of fields in a row aligned to four: the flag adds no
    // byte, and an optional row needs no tag beside it.
    assert_eq!(size_of::<LoadRow>(), 36);
    assert_eq!(size_of::<Option<LoadRow>>(), 36);
}

#[test]
fn an_identity_fingerprint_is_stable_and_never_zero() {
    let a = Identity::plain("session-1");
    let b = Identity::plain("session-1");
    assert_eq!(a.hash, b.hash);
    assert_ne!(a.hash, Identity::plain("session-2").hash);
    assert!(!a.is_none());
    assert!(Identity::NONE.is_none());
    assert_eq!(Identity::hashed(a.hash).hash, a.hash);
    // The empty key is present too: a fingerprint is never the absent
    // marker.
    assert_ne!(fingerprint(b""), 0);
}

/// The session's value is a wire fact: these literals stop an edit of
/// the formula that keeps every relative case green.
#[test]
fn a_sessions_value_is_pinned_for_a_fixed_key_model_and_each_phase() {
    let of = |phase| {
        RequestFacts {
            model: "model-a",
            session_key: Identity::plain("conversation-7"),
            phase,
            ..RequestFacts::default()
        }
        .session()
        .get()
    };
    assert_eq!(fingerprint(b"conversation-7"), 0xd449_9ecc_c740_2a12);
    assert_eq!(of(Phase::Single), 0x6e88_ff71_cc98_2a42);
    assert_eq!(of(Phase::Prefill), 0x619f_dbb0_c262_9a55);
    assert_eq!(of(Phase::Decode), 0x3e92_fbbd_183d_f4b6);
    assert_eq!(of(Phase::Encode), 0x3780_ba54_336a_57d0);
    // The lift of a zero result is a term of the formula. The mix sends
    // 0 to 0 and no other value there, so one key of a model and phase
    // would read as no session: it reads 1, and is a session.
    let lifted = RequestFacts {
        model: "model-a",
        session_key: Identity::hashed(0x2bb0_c344_f139_5abd),
        phase: Phase::Single,
        ..RequestFacts::default()
    }
    .session();
    assert_eq!(lifted.get(), 1);
    assert!(!lifted.is_none());
}

/// The formula runs backwards: a session's value gives its key's
/// fingerprint within the same model and phase, on the contract's own
/// numbers and across keys, models and phases.
#[test]
fn a_sessions_value_gives_its_keys_fingerprint_back_within_its_model_and_phase() {
    let phases = [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode];
    let contract: [u64; 4] = [
        0x6e88_ff71_cc98_2a42,
        0x619f_dbb0_c262_9a55,
        0x3e92_fbbd_183d_f4b6,
        0x3780_ba54_336a_57d0,
    ];
    let model = fingerprint(b"model-a");
    for (phase, session) in phases.into_iter().zip(contract) {
        let key = key_of_session(session, model, phase);
        assert_eq!(key, 0xd449_9ecc_c740_2a12, "{phase:?}");
    }
    for (index, phase) in (0..4_000_u32).zip(phases.into_iter().cycle()) {
        let name = format!("model-{}", index % 7);
        let key = fingerprint(format!("key-{index}").as_bytes());
        let session = RequestFacts {
            model: &name,
            session_key: Identity::hashed(key),
            phase,
            ..RequestFacts::default()
        }
        .session()
        .get();
        let read = key_of_session(session, fingerprint(name.as_bytes()), phase);
        assert_eq!(read, key, "key-{index} under {name}, {phase:?}");
    }
    // No session has no key, and neither has the one value a model and
    // a phase give the fingerprint no key has.
    assert_eq!(key_of_session(0, model, Phase::Single), 0);
    assert_eq!(
        key_of_session(0xc771_96d2_b9cf_9497, model, Phase::Single),
        0
    );
    // The value the formula lifts a zero result to reads as the key it
    // sends there by itself, whose session it is too.
    let sent = key_of_session(1, model, Phase::Single);
    let lifted = RequestFacts {
        model: "model-a",
        session_key: Identity::hashed(sent),
        ..RequestFacts::default()
    };
    assert_eq!((sent, lifted.session().get()), (0xbddd_7c85_982a_9f6f, 1));
}

#[test]
fn a_session_is_one_key_in_one_model_and_phase_whatever_form_the_key_takes() {
    let plain = RequestFacts {
        model: "model-a",
        session_key: Identity::plain("conversation-7"),
        ..RequestFacts::default()
    };
    // A host that holds the key's fingerprint alone names the same
    // session.
    let hashed = RequestFacts {
        session_key: Identity::hashed(fingerprint(b"conversation-7")),
        ..plain
    };
    assert_eq!(plain.session(), hashed.session());
    // So does a host that carries the model's fingerprint, made once a
    // model, where this one fingerprints the name at the read.
    let carried = RequestFacts {
        model_fingerprint: ModelFingerprint::of("model-a"),
        ..plain
    };
    assert_eq!(plain.session(), carried.session());
    let other_model = RequestFacts {
        model: "model-b",
        ..plain
    };
    let other_phase = RequestFacts {
        phase: Phase::Decode,
        ..plain
    };
    let other_key = RequestFacts {
        session_key: Identity::plain("conversation-8"),
        ..plain
    };
    for other in [other_model, other_phase, other_key] {
        assert_ne!(plain.session(), other.session());
    }
    // The model's part is its name as the request carries it: no name
    // is a name too.
    let unnamed = RequestFacts { model: "", ..plain };
    assert!(!unnamed.session().is_none());
    assert_ne!(unnamed.session(), plain.session());
    // No key, no session.
    let keyless = RequestFacts {
        session_key: Identity::NONE,
        ..plain
    };
    assert!(keyless.session().is_none());
}

/// A session key's count stands on a worker's object under one of two
/// names. Under the key's text the name is the text the request
/// carries, and a key carried as its fingerprint alone has none. Under
/// the session's number it is the number's sixteen digits: one name for
/// the key in either form, another for another model. A request with
/// no key has no name under either.
#[test]
fn a_sessions_count_is_named_by_where_it_stands() {
    let plain = RequestFacts {
        model: "model-a",
        session_key: Identity::plain("conversation-7"),
        ..RequestFacts::default()
    };
    let hashed = RequestFacts {
        session_key: Identity::hashed(plain.session_key.hash),
        ..plain
    };
    let named = |under: SessionCountedUnder, facts: &RequestFacts<'_>| {
        let mut buffer = [0; Session::TEXT_LEN];
        under.name(facts, &mut buffer).map(str::to_owned)
    };
    let text = SessionCountedUnder::KeyText;
    let number = SessionCountedUnder::SessionNumber;
    assert_eq!(SessionCountedUnder::default(), text);
    assert_eq!(named(text, &plain).as_deref(), Some("conversation-7"));
    assert_eq!(named(text, &hashed), None);
    let digits = format!("{:016x}", plain.session().get());
    assert_eq!(named(number, &plain).as_deref(), Some(digits.as_str()));
    assert_eq!(named(number, &hashed).as_deref(), Some(digits.as_str()));
    let other = RequestFacts {
        model: "model-b",
        ..plain
    };
    assert_ne!(named(number, &other), Some(digits));
    let keyless = RequestFacts::default();
    assert_eq!(named(text, &keyless), None);
    assert_eq!(named(number, &keyless), None);
    // The number's text: every digit, the highest first.
    let text_of = |value| {
        let mut buffer = [0; Session::TEXT_LEN];
        Session::from_raw(value).text(&mut buffer).to_owned()
    };
    assert_eq!(text_of(0x1f), "000000000000001f");
    assert_eq!(text_of(u64::MAX), "ffffffffffffffff");
    assert_eq!(text_of(0x0123_4567_89ab_cdef), "0123456789abcdef");
}

/// A carried fingerprint is the host's word that it is the name's. A
/// debug build stops the read of one that is another name's; a release
/// build trusts it, and the request is then that other model's session.
#[test]
#[cfg_attr(
    debug_assertions,
    should_panic(expected = "the carried fingerprint is not the model name's")
)]
fn a_carried_fingerprint_of_another_name_stops_a_debug_build_and_is_trusted_by_a_release_one() {
    let other = RequestFacts {
        model: "model-b",
        session_key: Identity::plain("conversation-7"),
        ..RequestFacts::default()
    };
    let carried = RequestFacts {
        model: "model-a",
        model_fingerprint: ModelFingerprint::of("model-b"),
        ..other
    };
    assert_eq!(carried.session(), other.session());
}

#[test]
fn wire_modes_compose_as_bits() {
    let both = WireModes::HTTP.union(WireModes::GRPC);
    assert!(both.contains(WireModes::HTTP));
    assert!(!both.contains(WireModes::ZMQ));
    assert!(both.intersects(WireModes::GRPC.union(WireModes::ZMQ)));
    assert!(!WireModes::ZMQ.intersects(both));
    assert_eq!(WireModes::from_bits(0xff).bits(), 0b111);
    assert!(WireModes::NONE.is_empty());
}

#[test]
fn no_media_is_the_default_of_the_counts() {
    assert_eq!(MediaCounts::NONE, MediaCounts::default());
}

#[test]
fn a_load_row_reads_its_permille_as_a_ratio() {
    let row = LoadRow {
        kv_usage_permille: 250,
        ..LoadRow::default()
    };
    assert!((row.kv_usage() - 0.25).abs() < 1e-6);
}
