use super::*;

/// A mask numbers the rows of the pool it is handed with. Beside rows this
/// side numbered anew it is read by where each row stood: a masked row that
/// was left out masks none, and a masked row that was admitted is no
/// candidate. Beside a whole pool it is read as it is handed.
#[test]
fn a_mask_is_read_by_where_each_row_stood() {
    let engine = engine();
    let open = Constraints::default();
    let (first, second, third) = (masking(&[0]), masking(&[1]), masking(&[2]));
    // The first prefill's class has no encoder and the first decode takes
    // no image: each side is cut to its other two rows, the last the
    // fuller.
    let mut cut = Bed::new();
    cut.prefills = (0..3)
        .map(|at| row(at, Role::Prefill, 1, 9 * (at / 2)))
        .collect();
    cut.prefill_classes = vec![2, 1, 1];
    cut.decodes = (0..3)
        .map(|at| row(at, Role::Decode, 1, 9 * (at / 2)))
        .collect();
    cut.decodes[0].vision = false;
    cut.decode_images = true;
    let (facts, _) = cut.asked();
    let pair = |sides: [&Constraints<'_>; 3]| {
        pair_of(&decide_under(&cut, &engine, &facts, cut.pools(), sides).0)
    };
    assert_eq!(pair([&open; 3]), (1, 1));
    // A masked row that was left out masks none.
    assert_eq!(pair([&first, &open, &open]), (1, 1));
    assert_eq!(pair([&open, &first, &open]), (1, 1));
    // A masked row that was admitted is no candidate.
    assert_eq!(pair([&second, &open, &open]), (2, 1));
    assert_eq!(pair([&open, &second, &open]), (1, 2));
    assert_eq!(pair([&third, &third, &open]), (1, 1));
    // A mask of a row left out and of a row admitted masks the one admitted.
    let both = masking(&[0, 1]);
    assert_eq!(pair([&both, &both, &open]), (2, 2));
    // Two encoders of the pair's class, the second the fuller, and one of
    // another class: a mask leaves the key to the class's open encoder.
    let mut offered = Bed::new();
    offered.encoders = (0..3)
        .map(|at| row(at, Role::Encode, 0, 9 * (at % 2)))
        .collect();
    offered.facts = vec![pushing(1), pushing(1), pushing(2)];
    let encoder = |mask: &Constraints<'_>| {
        let sides = [&open, &open, mask];
        let (decision, placed) = decide_under(&offered, &engine, &facts, offered.pools(), sides);
        assert!(decision.pick().is_some(), "a pick, not {decision:?}");
        placed.encoders().first().map(|target| target.position)
    };
    assert_eq!(encoder(&open), Some(0));
    assert_eq!(encoder(&first), Some(1));
    assert_eq!(encoder(&second), Some(0));
    assert_eq!(encoder(&third), Some(0));
    // No cut: beside a whole pool a mask is read as it is handed, and the
    // masked prefill or decode is no candidate.
    let mut whole = Bed::new();
    whole.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 9)];
    whole.prefill_classes = vec![1, 1];
    whole.decodes = vec![row(0, Role::Decode, 1, 0), row(1, Role::Decode, 1, 9)];
    for (sides, pair) in [
        ([&first, &open, &open], (1, 0)),
        ([&open, &first, &open], (0, 1)),
    ] {
        let (decision, _) = decide_under(&whole, &engine, &facts, whole.pools(), sides);
        assert_eq!(pair_of(&decision), pair);
    }
}

/// A mask numbered anew is the decision's own: the next decision of one
/// scratch reads its own mask and no row of the last.
#[test]
fn a_mask_numbered_anew_is_forgotten_by_the_next_decision() {
    let engine = engine();
    // The first prefill's class has no encoder: the pool is cut to the
    // other two, the last the fuller.
    let mut bed = Bed::new();
    bed.prefills = (0..3)
        .map(|at| row(at, Role::Prefill, 1, 9 * (at / 2)))
        .collect();
    bed.prefill_classes = vec![2, 1, 1];
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
    let open = Constraints::default();
    let (mut scratch, mut placed) = (Scratch::new(), EncodePlaced::new());
    let mut prefill = |mask: &[u32]| {
        let (cx, pools) = (masking(mask), bed.pools());
        let decision = engine.decide_encode_pair(
            &facts,
            &pool,
            pools,
            encode,
            items,
            &cx,
            &open,
            &open,
            &mut scratch,
            &mut placed,
        );
        pair_of(&decision).0
    };
    assert_eq!(prefill(&[1]), 2);
    assert_eq!(prefill(&[2]), 1);
    assert_eq!(prefill(&[]), 1);
}

/// A soft filter that reads the mask: the pass that opens the encoders asks
/// the hard filters alone.
#[derive(Debug)]
struct MaskedSoftly;

impl framework::Filter for MaskedSoftly {
    fn name(&self) -> &'static str {
        "masked-softly"
    }

    fn keep(&self, _: &RequestFacts<'_>, row: &Candidate, cx: &Constraints<'_>) -> bool {
        !cx.masks(row.position)
    }

    fn soft(&self) -> bool {
        true
    }
}

/// Where no hard filter reads the mask a masked encoder is offered, and a
/// key's run reads the mask by the rows it is handed.
#[test]
fn a_mask_is_numbered_with_the_encoders_offered() {
    let mut engine = engine();
    let encode = engine
        .binding("m", Phase::Prefill)
        .and_then(|binding| match binding.plan {
            Plan::Epd { encode } => Some(usize::from(encode)),
            Plan::Single | Plan::PdPair => None,
        })
        .expect("the epd plan");
    let filters = &mut own(&mut engine.profiles[encode]).filters;
    filters.retain(|filter| filter.name() != "healthy");
    filters.push(Box::new(MaskedSoftly));
    engine.marks[encode] = framework::Marks::of(&engine.profiles[encode], engine.orders[encode]);
    // The first encoder is of another class: the two offered stand at 0 and
    // 1, the first of them masked and the second the fuller.
    let mut bed = Bed::new();
    bed.encoders = (0..3)
        .map(|at| row(at, Role::Encode, 0, 9 * (at / 2)))
        .collect();
    bed.facts = vec![pushing(2), pushing(1), pushing(1)];
    let (facts, _) = bed.asked();
    let (open, masked) = (Constraints::default(), masking(&[1]));
    let sides = [&open, &open, &masked];
    let (decision, placed) = decide_under(&bed, &engine, &facts, bed.pools(), sides);
    assert!(decision.pick().is_some(), "a pick, not {decision:?}");
    assert_eq!(placed.encoders().first().map(|t| t.position), Some(2));
}

/// Beside a whole pool a mask is read as it is handed. Both encoders are of
/// the prefill's class and no hard filter reads the mask, so the encoders
/// offered are the pool. A key's run then reads the list its host stated,
/// an entry that names no row among it: nothing is numbered anew.
#[test]
fn a_mask_over_every_encoder_offered_is_read_as_it_is_handed() {
    let mut engine = engine();
    let notes = noting(&mut engine);
    let encode = engine
        .binding("m", Phase::Prefill)
        .and_then(|binding| match binding.plan {
            Plan::Epd { encode } => Some(usize::from(encode)),
            Plan::Single | Plan::PdPair => None,
        })
        .expect("the epd plan");
    let filters = &mut own(&mut engine.profiles[encode]).filters;
    filters.retain(|filter| filter.name() != "healthy");
    engine.marks[encode] = framework::Marks::of(&engine.profiles[encode], engine.orders[encode]);
    let mut bed = Bed::new();
    bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
    bed.facts = vec![pushing(1), pushing(1)];
    let (facts, _) = bed.asked();
    let (open, masked) = (Constraints::default(), masking(&[1, 9]));
    let sides = [&open, &open, &masked];
    let (decision, _) = decide_under(&bed, &engine, &facts, bed.pools(), sides);
    assert!(decision.pick().is_some(), "a pick, not {decision:?}");
    let read = noted_under(&notes, Phase::Encode);
    // Each of the two encoders was asked, under the mask as handed.
    assert_eq!(read.iter().map(|note| note.at).collect::<Vec<_>>(), [0, 1]);
    assert!(read.iter().all(|note| note.mask == [1, 9]), "{read:?}");
}

/// A filter that refuses the rows the host holds and reads no mask: a hard
/// one, or a soft one that the pass opening the encoders does not ask.
#[derive(Debug)]
struct HeldOut {
    soft: bool,
}

impl framework::Filter for HeldOut {
    fn name(&self) -> &'static str {
        "held-out"
    }

    fn keep(&self, _: &RequestFacts<'_>, row: &Candidate, cx: &Constraints<'_>) -> bool {
        cx.held.binary_search(&row.position).is_err()
    }

    fn soft(&self) -> bool {
        self.soft
    }
}

/// The held rows of a mask are numbered with it. Beside rows a side
/// numbered anew, a stage reads a held row by where the row stands among
/// them: on the prefill side, on the decode side and among the encoders
/// offered. A mask that holds no row refuses none here.
#[test]
fn the_held_rows_of_a_mask_are_numbered_with_it() {
    let mut engine = engine();
    let binding = engine.binding("m", Phase::Prefill).expect("the binding");
    let Plan::Epd { encode } = binding.plan else {
        panic!("the epd plan");
    };
    let decode = binding.decode_profile.unwrap_or(binding.profile);
    for (profile, soft) in [(binding.profile, false), (decode, false), (encode, true)] {
        let profile = usize::from(profile);
        let filters = &mut own(&mut engine.profiles[profile]).filters;
        filters.retain(|filter| filter.name() != "healthy");
        filters.push(Box::new(HeldOut { soft }));
        engine.marks[profile] =
            framework::Marks::of(&engine.profiles[profile], engine.orders[profile]);
    }
    let open = Constraints::default();
    let (masked, held) = (
        masking(&[1]),
        Constraints {
            stalled: &[1],
            held: &[1],
            ..Constraints::default()
        },
    );
    // The first prefill's class has no encoder and the first decode takes
    // no image: each side is cut to its other two rows, the last the
    // fuller. The held row stands first among the rows admitted.
    let mut cut = Bed::new();
    cut.prefills = (0..3)
        .map(|at| row(at, Role::Prefill, 1, 9 * (at / 2)))
        .collect();
    cut.prefill_classes = vec![2, 1, 1];
    cut.decodes = (0..3)
        .map(|at| row(at, Role::Decode, 1, 9 * (at / 2)))
        .collect();
    cut.decodes[0].vision = false;
    cut.decode_images = true;
    let (facts, _) = cut.asked();
    let pair = |sides: [&Constraints<'_>; 3]| {
        pair_of(&decide_under(&cut, &engine, &facts, cut.pools(), sides).0)
    };
    assert_eq!(pair([&open; 3]), (1, 1));
    assert_eq!(pair([&masked, &masked, &open]), (1, 1));
    assert_eq!(pair([&held, &open, &open]), (2, 1));
    assert_eq!(pair([&open, &held, &open]), (1, 2));
    // One scratch through two decisions: the held rows numbered for the
    // first are forgotten by the second, whose mask holds none.
    let (_, pool) = cut.asked();
    let encode = EncodePool {
        encoders: &cut.encoders,
        facts: &cut.facts,
        prefill_classes: &cut.prefill_classes,
        affinity: &cut.encode_affinity,
        ..EncodePool::default()
    };
    let items = EncodeItems {
        keys: &cut.keys,
        one: cut.one,
        references: cut.references,
        decode_images: cut.decode_images,
        ring: None,
    };
    let (mut scratch, mut placed) = (Scratch::new(), EncodePlaced::new());
    let mut prefill = |cx: &Constraints<'_>| {
        let decision = engine.decide_encode_pair(
            &facts,
            &pool,
            cut.pools(),
            encode,
            items,
            cx,
            &open,
            &open,
            &mut scratch,
            &mut placed,
        );
        pair_of(&decision).0
    };
    assert_eq!(prefill(&held), 2);
    assert_eq!(prefill(&masked), 1);
    // The first encoder is of another class: the two offered stand at 0 and
    // 1, the first of them held and the second the fuller.
    let mut bed = Bed::new();
    bed.encoders = (0..3)
        .map(|at| row(at, Role::Encode, 0, 9 * (at / 2)))
        .collect();
    bed.facts = vec![pushing(2), pushing(1), pushing(1)];
    let (facts, _) = bed.asked();
    let encoder = |side: &Constraints<'_>| {
        let sides = [&open, &open, side];
        let (decision, placed) = decide_under(&bed, &engine, &facts, bed.pools(), sides);
        assert!(decision.pick().is_some(), "a pick, not {decision:?}");
        placed.encoders().first().map(|target| target.position)
    };
    assert_eq!(encoder(&masked), Some(1));
    assert_eq!(encoder(&held), Some(2));
}

/// A side's held list takes its room where a hold is first numbered, one
/// place a row admitted. A mask with no hold takes none, and a fuller hold
/// over as many rows grows nothing. The list of a side's places is sized
/// the same way, where places are first numbered.
#[test]
fn a_sides_held_list_is_sized_once_where_a_hold_is_numbered() {
    let mut side = Admitted::default();
    for at in 0..8 {
        side.push(&row(at, Role::Prefill, 1, 0), at as usize, &[]);
    }
    side.take_room();
    side.renumber(&[1, 3], &[], None);
    assert_eq!(side.held.capacity(), 0, "a mask with no hold takes no room");
    assert_eq!(side.places.capacity(), 0, "nor does one with no place");
    side.masked.clear();
    side.renumber(&[1, 3], &[3], None);
    assert_eq!((&side.masked[..], &side.held[..]), (&[1, 3][..], &[3][..]));
    let room = side.held.capacity();
    assert!(
        room >= side.rows.len(),
        "the first hold takes a place a row"
    );
    side.masked.clear();
    let every: Vec<u32> = (0..8).collect();
    side.renumber(&every, &every, Some(&[3, 9]));
    assert_eq!(side.held, every);
    assert_eq!(side.held.capacity(), room, "a fuller hold grew the list");
    // A place past the rows names none of them.
    assert_eq!(side.places, [3]);
    let room = side.places.capacity();
    assert!(
        room >= side.rows.len(),
        "the first places take a place a row"
    );
    side.masked.clear();
    side.renumber(&[], &[], Some(&every));
    assert_eq!(side.places, every);
    assert_eq!(side.places.capacity(), room, "fuller places grew the list");
    // The next decision's places are its own.
    side.renumber(&[], &[], None);
    assert!(side.places.is_empty());
}
