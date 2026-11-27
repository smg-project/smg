use super::*;

/// A filter that keeps an encoder under the encode phase alone.
#[derive(Debug)]
struct EncodersUnderTheirPhase;

impl framework::Filter for EncodersUnderTheirPhase {
    fn name(&self) -> &'static str {
        "encoders-under-their-phase"
    }

    fn keep(&self, facts: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
        !matches!(row.role, Role::Encode) || matches!(facts.phase, Phase::Encode)
    }
}

/// The bed's decision into a `placed` the caller holds.
fn decide_into(bed: &Bed, engine: &Engine, placed: &mut EncodePlaced) -> Decision {
    let (facts, pool) = bed.asked();
    let encode = EncodePool {
        encoders: &bed.encoders,
        facts: &bed.facts,
        prefill_classes: &bed.prefill_classes,
        affinity: &bed.encode_affinity,
        ..EncodePool::default()
    };
    let items = EncodeItems {
        keys: &bed.keys,
        one: bed.one,
        references: bed.references,
        decode_images: bed.decode_images,
        ring: None,
    };
    let cx = Constraints::default();
    engine.decide_encode_pair(
        &facts,
        &pool,
        bed.pools(),
        encode,
        items,
        &cx,
        &cx,
        &cx,
        &mut Scratch::new(),
        placed,
    )
}

/// A decision that is no pick holds nothing placed: an item's stage that
/// turns after an earlier item was placed, or after the one key was
/// reported, leaves no encoder and no one key behind.
#[test]
fn a_decision_that_is_no_pick_holds_nothing_placed() {
    let mut engine = engine();
    for profile in engine.profiles.iter_mut().map(own) {
        profile.filters.push(Box::new(RefusesTheKey));
    }
    // Two items: the first is placed, the second's stage turns.
    let mut bed = Bed::new();
    bed.keys = vec![Identity::hashed(11), Identity::hashed(12)];
    let (decision, placed) = bed.decide_with(&engine);
    let stage = match &decision {
        Decision::Shed(shed) => shed.emptied_by,
        other => panic!("a shed, not {other:?}"),
    };
    assert_eq!(stage, "refuses-the-key");
    assert!(
        placed.encoders().is_empty(),
        "the first item's encoder is held under a shed"
    );
    assert!(!placed.one());
    // The one key under a class that pushes: reported, then its stage turns.
    let mut bed = Bed::new();
    bed.one = Some(Identity::hashed(12));
    let (decision, placed) = bed.decide_with(&engine);
    assert!(decision.pick().is_none(), "a shed, not {decision:?}");
    assert!(!placed.one(), "the one key is reported under a shed");
    assert!(placed.encoders().is_empty());
    // The same engine places both items of two keys it keeps.
    let mut bed = Bed::new();
    bed.keys = vec![Identity::hashed(11), Identity::hashed(13)];
    let (decision, placed) = bed.decide_with(&engine);
    assert!(decision.pick().is_some(), "a pick, not {decision:?}");
    assert_eq!(placed.encoders().len(), 2);
}

/// The entry clears what the decision before placed: a `placed` the host
/// hands again reads the new decision alone.
#[test]
fn the_entry_clears_what_the_decision_before_placed() {
    let engine = engine();
    let mut placed = EncodePlaced::new();
    let mut pushed = Bed::new();
    pushed.one = Some(Identity::hashed(12));
    let first = decide_into(&pushed, &engine, &mut placed);
    assert!(first.pick().is_some(), "a pick, not {first:?}");
    assert_eq!((placed.encoders().len(), placed.one()), (1, true));
    // A request without items after it reads no encoder and places none.
    let mut bare = Bed::new();
    bare.keys.clear();
    let second = decide_into(&bare, &engine, &mut placed);
    assert!(second.pick().is_some(), "a pick, not {second:?}");
    assert_eq!(
        (placed.encoders().len(), placed.one()),
        (0, false),
        "the decision before is held"
    );
    // And a request of one item after the one key's holds its own alone.
    let third = decide_into(&pushed, &engine, &mut placed);
    assert!(third.pick().is_some(), "a pick, not {third:?}");
    let fourth = decide_into(&Bed::new(), &engine, &mut placed);
    assert!(fourth.pick().is_some(), "a pick, not {fourth:?}");
    assert_eq!((placed.encoders().len(), placed.one()), (1, false));
}

/// The encode side's stages are asked under the encode phase: a stage that
/// keeps an encoder under that phase alone leaves the item placed.
#[test]
fn the_encode_side_is_asked_under_the_encode_phase() {
    let mut engine = engine();
    for profile in engine.profiles.iter_mut().map(own) {
        profile.filters.push(Box::new(EncodersUnderTheirPhase));
    }
    let (decision, placed) = Bed::new().decide_with(&engine);
    assert!(decision.pick().is_some(), "a pick, not {decision:?}");
    assert_eq!(placed.encoders().len(), 1);
}

/// What [`ReadsHeld`] records: each row's worker id with the value it read.
type HeldReads = Arc<parking_lot::Mutex<Vec<(u64, Option<f64>)>>>;

/// A filter that keeps every row and records what the row reads of a
/// signal: its worker's id with the value.
#[derive(Debug)]
struct ReadsHeld(SignalColumn, HeldReads);

impl framework::Filter for ReadsHeld {
    fn name(&self) -> &'static str {
        "reads-held"
    }

    fn keep(&self, facts: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
        self.1.lock().push((row.id.0, self.0.read(facts, row)));
        true
    }
}

/// Every pool a decision reads has its cells, and a side the admission
/// narrows carries its admitted rows' cells. Here the admission leaves out
/// the first prefill and the first decode, so each side's admitted row
/// stands at position 0, where the host's cells hold its neighbour's
/// value. The request's own cells name no pool. Each row that is asked
/// reads its own value: a prefill 10 and its place, a decode 20 and its
/// place, an encoder 30 and its place.
#[test]
fn each_admitted_row_and_each_encoder_reads_its_own_cells() {
    let read = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut engine = engine();
    for profile in engine.profiles.iter_mut().map(own) {
        let held = SignalColumn::new("held", "tokens", 0, None);
        let reads = ReadsHeld(held, Arc::clone(&read));
        profile.filters.push(Box::new(reads));
    }
    let cell = |value| SignalCell {
        value,
        sampled_at_micros: 0,
    };
    let mut bed = Bed::new();
    // The idle prefill's class has no open encoder; the idle decode takes
    // no image and the request asks one of its decode.
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 9)];
    bed.prefill_classes = vec![1, 2];
    bed.decodes = vec![
        Candidate {
            vision: false,
            ..row(0, Role::Decode, 1, 0)
        },
        row(1, Role::Decode, 1, 9),
    ];
    bed.encoders = vec![
        unready(row(0, Role::Encode, 0, 0)),
        row(1, Role::Encode, 0, 3),
    ];
    bed.facts = vec![pushing(1), pushing(2)];
    bed.decode_images = true;
    bed.prefill_cells = vec![cell(10.0), cell(11.0)];
    bed.decode_cells = vec![cell(20.0), cell(21.0)];
    bed.encode_cells = vec![cell(30.0), cell(31.0)];
    let request = SignalCells::new(vec![cell(1.0); 8], 1);
    let (facts, pool) = bed.asked();
    let facts = RequestFacts {
        signal_cells: Some(&request),
        ..facts
    };
    let (decision, _) = bed.decide_asked(&engine, &facts, &pool);
    assert!(decision.pick().is_some(), "a pick, not {decision:?}");
    let read = read.lock();
    let own = |id: u64| Some(((id / 1_000) * 10 + id % 1_000) as f64);
    assert!(read.iter().all(|&(id, value)| value == own(id)), "{read:?}");
    // The admitted prefill, the admitted decode and the open encoder.
    for id in [1_001, 2_001, 3_001] {
        assert!(read.iter().any(|&(asked, _)| asked == id), "{id}: {read:?}");
    }
}

/// The decision of a bed whose admission narrows both sides, as the case
/// above lays it, with `prefill` and `decode` cells handed for two rows a
/// side.
#[cfg(debug_assertions)]
fn decided_over_cells(prefill: usize, decode: usize) {
    let cell = SignalCell {
        value: 1.0,
        sampled_at_micros: 0,
    };
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 9)];
    bed.prefill_classes = vec![1, 2];
    bed.decodes = vec![
        Candidate {
            vision: false,
            ..row(0, Role::Decode, 1, 0)
        },
        row(1, Role::Decode, 1, 9),
    ];
    bed.encoders = vec![
        unready(row(0, Role::Encode, 0, 0)),
        row(1, Role::Encode, 0, 3),
    ];
    bed.facts = vec![pushing(1), pushing(2)];
    bed.decode_images = true;
    bed.prefill_cells = vec![cell; prefill];
    bed.decode_cells = vec![cell; decode];
    bed.encode_cells = vec![cell; 2];
    let (facts, pool) = bed.asked();
    let _ = bed.decide_asked(&engine(), &facts, &pool);
}

/// A debug build refuses a side's cells of another shape at the entry,
/// whatever the admission. The copy for the admitted rows pads a short
/// set with cells of no value, and the pair's own assertion then reads
/// the copy.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the signal cells are those of the rows the decision was given")]
fn a_debug_build_refuses_the_prefills_cells_of_another_shape_at_the_encode_entry() {
    decided_over_cells(1, 2);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the signal cells are those of the rows the decision was given")]
fn a_debug_build_refuses_the_decodes_cells_of_another_shape_at_the_encode_entry() {
    decided_over_cells(2, 3);
}

/// A release build asserts no shape, so the host's cells may end before an
/// admitted row. The scratch is one a side for every pass: a pass's cells
/// are its own rows' alone.
#[test]
fn an_admitted_row_outside_the_hosts_cells_takes_cells_of_no_value() {
    let cell = |value| SignalCell {
        value,
        sampled_at_micros: 0,
    };
    let values = |admitted: &Admitted| -> Vec<Option<f64>> {
        (0..admitted.from.len() as u32)
            .map(|at| admitted.cells.row(at).map(|row| row[0].value))
            .collect()
    };
    let mut admitted = Admitted {
        from: vec![1, 2],
        ..Admitted::default()
    };
    admitted.copy_cells(&SignalCells::new(vec![cell(10.0), cell(11.0)], 1));
    let read = values(&admitted);
    assert_eq!(read[0], Some(11.0));
    assert!(read[1].is_some_and(f64::is_nan), "no value, not {read:?}");
    // The next pass admits one row: the cells of the pass before are gone.
    admitted.from = vec![0];
    admitted.copy_cells(&SignalCells::new(vec![cell(20.0), cell(21.0)], 1));
    assert_eq!(values(&admitted), [Some(20.0)]);
    assert!(admitted.cells.fits(1));
}

#[test]
fn a_hosts_count_past_the_cap_asks_the_admitted_copy_for_no_cell() {
    let mut admitted = Admitted {
        from: vec![0, 1, 2],
        ..Admitted::default()
    };
    // No cells, and a count no document holds: a copy under that count
    // would write as many empty cells for each admitted row.
    let past = spec::MAX_SIGNALS + 1;
    admitted.copy_cells(&SignalCells::new(Vec::new(), past));
    assert_eq!(admitted.cells.signals(), 0);
    assert!(admitted.cells.fits(3), "no cell for any admitted row");
}

/// The decision of `bed` over a fresh scratch under `table`, and the words
/// each of the encode side's four sets of bits holds after it: the open and
/// the member encode classes, the decodes' classes and those that take
/// images.
fn decided_widths(bed: &Bed, table: PairTable<'_>) -> (Decision, [usize; 4]) {
    let (facts, pool) = bed.asked();
    let pools = PairPools {
        table,
        ..bed.pools()
    };
    let encode = EncodePool {
        encoders: &bed.encoders,
        facts: &bed.facts,
        prefill_classes: &bed.prefill_classes,
        affinity: &bed.encode_affinity,
        ..EncodePool::default()
    };
    let items = EncodeItems {
        keys: &bed.keys,
        one: bed.one,
        references: bed.references,
        decode_images: bed.decode_images,
        ring: None,
    };
    let cx = Constraints::default();
    let mut scratch = Scratch::new();
    let decision = engine().decide_encode_pair(
        &facts,
        &pool,
        pools,
        encode,
        items,
        &cx,
        &cx,
        &cx,
        &mut scratch,
        &mut EncodePlaced::new(),
    );
    let side = scratch
        .encode
        .as_deref()
        .expect("the entry leaves its side in the scratch");
    let (encoders, decodes) = (&side.encoders, &side.decode_classes);
    let sets = [
        &encoders.open,
        &encoders.member,
        &decodes.registered,
        &decodes.vision,
    ];
    (decision, sets.map(|set| set.words.len()))
}

#[test]
fn a_class_number_above_the_fleet_sizes_no_set() {
    let none = PairTable::default();
    let far = |at: u32, below: u32| row(at, Role::Decode, u32::MAX - below, 0);
    // Under no table a class pairs by its number, whatever the number: the
    // largest number finds its pair among others as far, and no set is as
    // wide as any of them.
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, u32::MAX, 0)];
    bed.decodes = vec![far(0, 0), far(1, 1), far(2, 9)];
    let (decision, words) = decided_widths(&bed, none);
    assert!(decision.pick().is_some(), "{decision:?}");
    assert_eq!(words, [1, 1, 1, 1]);
    // A far number no decode names is no partner.
    bed.decodes = vec![far(0, 1), far(1, 9)];
    let (decision, words) = decided_widths(&bed, none);
    assert!(decision.pick().is_none(), "{decision:?}");
    assert_eq!(words[2..], [1, 1]);
    // Under a table a number above the table's count is no class of it, and
    // the sets are the table's width.
    let (ends, classes) = ([1], [1]);
    let mut bed = Bed::new();
    bed.decodes = vec![row(0, Role::Decode, 1, 0), far(1, 0)];
    let (decision, words) = decided_widths(&bed, PairTable::new(&ends, &classes));
    assert!(decision.pick().is_some(), "{decision:?}");
    assert_eq!(words[2..], [1, 1]);
    // An encode class far above the rows handed sizes no set either.
    let mut bed = Bed::new();
    bed.facts = vec![pushing(u32::MAX)];
    bed.prefill_classes = vec![u32::MAX];
    let (decision, words) = decided_widths(&bed, none);
    assert!(decision.pick().is_some(), "{decision:?}");
    assert_eq!(words[..2], [1, 1]);
}

#[test]
fn an_encode_class_above_the_rows_handed_is_a_class_still() {
    // A host states each encoder's class, and the framework does not hold
    // it to the count of the rows handed. A class above that count is a
    // class still, from the first number the sets' bits do not hold on.
    let none = PairTable::default();
    for class in [63, 64, 65, 4096] {
        let mut bed = Bed::new();
        bed.facts = vec![pushing(class)];
        bed.prefill_classes = vec![class];
        let (decision, _) = decided_widths(&bed, none);
        assert!(decision.pick().is_some(), "class {class}: {decision:?}");
    }
    // The numbers above the bits are found whatever the order of their
    // rows, and one no encoder names admits no prefill.
    let mut bed = Bed::new();
    bed.encoders = (0..3).map(|at| row(at, Role::Encode, 0, 0)).collect();
    bed.facts = [0, 1, 9].map(|below| pushing(u32::MAX - below)).to_vec();
    bed.prefill_classes = vec![u32::MAX];
    let (decision, _) = decided_widths(&bed, none);
    assert!(decision.pick().is_some(), "{decision:?}");
    bed.prefill_classes = vec![u32::MAX - 2];
    let (decision, _) = decided_widths(&bed, none);
    assert!(decision.pick().is_none(), "{decision:?}");
    // A prefill of a class no encoder names is left out beside the far
    // class's own, and the pair is picked among what is left.
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 0)];
    bed.prefill_classes = vec![7, u32::MAX];
    let (decision, _) = decided_widths(&bed, none);
    let Decision::Pick(Pick { targets, .. }) = decision else {
        panic!("a pick, not {decision:?}");
    };
    assert_eq!(targets.first().map(|target| target.position), Some(1));
}

/// The index of `decodes`, of a fleet of as many, under `table`.
fn index_of(decodes: &[Candidate], table: PairTable<'_>) -> DecodeClasses {
    let mut classes = DecodeClasses::default();
    classes.fill(decodes, decodes.len(), &table);
    classes
}

#[test]
fn a_far_decode_class_is_found_whatever_the_order_of_its_rows() {
    // Under no table the numbers above the bits stand in two lists, and
    // each is found whatever the order of its rows: four far classes, the
    // largest first, and the third takes no images.
    let far = |at: u32, below: u32| row(at, Role::Decode, u32::MAX - below, 0);
    let mut decodes = vec![far(0, 0), far(1, 1), far(2, 5), far(3, 9)];
    decodes[2].vision = false;
    // A decode of no class stands beside them.
    decodes.push(row(4, Role::Decode, 0, 0));
    let classes = index_of(&decodes, PairTable::default());
    for below in [0, 1, 5, 9] {
        let class = u32::MAX - below;
        assert!(classes.holds(class), "{below} below the largest");
        let vision = classes.holds_vision(class);
        assert_eq!(vision, below != 5, "{below} below the largest");
    }
    // A far number no decode names is no class, and neither is the number
    // of the decode outside any pairing.
    for class in [u32::MAX - 2, 0] {
        assert!(!classes.holds(class), "class {class}");
        assert!(!classes.holds_vision(class), "class {class}");
    }
}

#[test]
fn a_tables_decode_classes_stand_in_the_tables_own_width() {
    // Under a table the sets are the table's width, whatever the count of
    // the decodes: one decode of the last of a table's seventy classes.
    let listed: Vec<u32> = (1..=70).collect();
    let ends = [70];
    let decodes = [row(0, Role::Decode, 70, 0)];
    let classes = index_of(&decodes, PairTable::new(&ends, &listed));
    assert!(classes.holds(70));
    assert!(classes.holds_vision(70));
    let sets = [&classes.registered, &classes.vision];
    assert_eq!(sets.map(|set| set.words.len()), [2, 2]);
    // A number above the table's count is no class of it, and no list
    // keeps it.
    let (ends, listed) = ([1], [1]);
    let decodes = [
        row(0, Role::Decode, 1, 0),
        row(1, Role::Decode, u32::MAX, 0),
    ];
    let classes = index_of(&decodes, PairTable::new(&ends, &listed));
    assert!(classes.holds(1));
    assert!(!classes.holds(u32::MAX));
    assert!(!classes.holds_vision(u32::MAX));
}

#[test]
fn a_request_without_items_reads_no_membership() {
    // No answer of the membership pass is asked: the sets stay as a fresh
    // scratch holds them, and the pair is the pair entry's.
    let mut bed = Bed::new();
    bed.keys.clear();
    bed.images = 0;
    let (decision, words) = decided_widths(&bed, PairTable::default());
    assert_eq!(decision, bed.pair(&engine()));
    assert_eq!(words, [0, 0, 0, 0]);
}
