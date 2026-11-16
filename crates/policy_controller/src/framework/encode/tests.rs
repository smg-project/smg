//! The encode side of a pair decision, over small pools built by hand: what
//! the recorded sheet does not hold (a column of another length, a binding
//! of another plan, the rows read back from the admitted pools, the
//! affinity columns) and each gap by its stage.

use super::*;
use crate::{
    framework::PairTable,
    spec::PolicySpec,
    view::{Health, MediaCounts, Role, WorkerId},
    Pick,
};

/// The `epd` plan with every side taking the row that holds fewer requests.
fn engine() -> Engine {
    let mut spec = PolicySpec::epd_presets("power_of_two", "power_of_two", "power_of_two")
        .expect("the presets");
    for profile in spec.profiles.values_mut() {
        profile.pick = "count-first".into();
    }
    spec.validate().expect("the form validates")
}

/// A ready row of `role` at `at` in pair class `pair_class`, holding
/// `in_flight` requests.
fn row(at: u32, role: Role, pair_class: u32, in_flight: u32) -> Candidate {
    let id = match role {
        Role::Prefill => 1_000,
        Role::Decode => 2_000,
        _ => 3_000,
    } + u64::from(at);
    Candidate {
        position: at,
        id: WorkerId(id),
        role,
        health: Health {
            healthy: true,
            ..Health::default()
        },
        in_flight,
        pair_class,
        has_capacity: true,
        has_room: true,
        vision: true,
        key: id as u32,
        ..Candidate::default()
    }
}

fn unready(row: Candidate) -> Candidate {
    Candidate {
        health: Health::default(),
        ..row
    }
}

/// An encoder's facts: its class, pushing, taking no references.
fn pushing(class: u32) -> EncoderFacts {
    EncoderFacts {
        class,
        pushes: true,
        references: false,
    }
}

/// One fleet and one request, decided.
struct Bed {
    prefills: Vec<Candidate>,
    decodes: Vec<Candidate>,
    encoders: Vec<Candidate>,
    facts: Vec<EncoderFacts>,
    prefill_classes: Vec<u32>,
    prefill_affinity: Vec<AffinityFacts>,
    encode_affinity: Vec<AffinityFacts>,
    keys: Vec<Identity<'static>>,
    one: Option<Identity<'static>>,
    references: bool,
    decode_images: bool,
    images: u16,
}

impl Bed {
    /// One prefill and one decode of pair class 1, one pushing encoder of
    /// encode class 1 and a request of one item that carries an image.
    fn new() -> Self {
        Self {
            prefills: vec![row(0, Role::Prefill, 1, 0)],
            decodes: vec![row(0, Role::Decode, 1, 0)],
            encoders: vec![row(0, Role::Encode, 0, 0)],
            facts: vec![pushing(1)],
            prefill_classes: vec![1],
            prefill_affinity: Vec::new(),
            encode_affinity: Vec::new(),
            keys: vec![Identity::hashed(11)],
            one: None,
            references: false,
            decode_images: false,
            images: 1,
        }
    }

    /// The request's facts and the pool's, as both entries are handed them.
    fn asked(&self) -> (RequestFacts<'static>, PoolFacts<'static>) {
        let facts = RequestFacts {
            model: "m",
            media: MediaCounts {
                images: self.images,
                ..MediaCounts::default()
            },
            seed: 7,
            ..RequestFacts::default()
        };
        let pool = PoolFacts {
            pool_size: self.prefills.len() as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        (facts, pool)
    }

    fn pools(&self) -> PairPools<'_> {
        PairPools {
            prefills: &self.prefills,
            decodes: &self.decodes,
            prefill_affinity: &self.prefill_affinity,
            ..PairPools::default()
        }
    }

    /// The pair entry's own answer over the bed's two pools.
    fn pair(&self, engine: &Engine) -> Decision {
        let (facts, pool) = self.asked();
        let cx = Constraints::default();
        engine.decide_pair(&facts, &pool, self.pools(), &cx, &cx, &mut Scratch::new())
    }

    fn decide_with(&self, engine: &Engine) -> (Decision, EncodePlaced) {
        let (facts, pool) = self.asked();
        let pools = self.pools();
        let encode = EncodePool {
            encoders: &self.encoders,
            facts: &self.facts,
            prefill_classes: &self.prefill_classes,
            affinity: &self.encode_affinity,
        };
        let items = EncodeItems {
            keys: &self.keys,
            one: self.one,
            references: self.references,
            decode_images: self.decode_images,
        };
        let cx = Constraints::default();
        let mut placed = EncodePlaced::new();
        let decision = engine.decide_encode_pair(
            &facts,
            &pool,
            pools,
            encode,
            items,
            &cx,
            &cx,
            &cx,
            &mut Scratch::new(),
            &mut placed,
        );
        (decision, placed)
    }

    fn decide(&self) -> (Decision, EncodePlaced) {
        self.decide_with(&engine())
    }

    /// The stage a shed of this bed names.
    fn stage(&self) -> &'static str {
        match self.decide().0 {
            Decision::Shed(shed) => shed.emptied_by,
            other => panic!("a shed, not {other:?}"),
        }
    }

    /// The pick of this bed: the prefill's and the decode's rows, each
    /// placed key's encoder row, and whether the one key was placed.
    fn placed(&self) -> ((u32, u32), Vec<u32>, bool) {
        let (decision, placed) = self.decide();
        let Decision::Pick(Pick { targets, .. }) = decision else {
            panic!("a pick, not {decision:?}");
        };
        let target = targets.first().expect("a target");
        let decode = target.pair.expect("a pair").decode_position;
        let encoders = placed.encoders().iter().map(|t| t.position).collect();
        ((target.position, decode), encoders, placed.one())
    }
}

#[test]
fn an_item_is_placed_on_an_encoder_of_the_chosen_prefills_class() {
    let mut bed = Bed::new();
    // Two classes: the prefill of class 2 holds fewer requests and is the
    // pair's; the idle encoder of class 1 is not its encoder.
    bed.prefills = vec![row(0, Role::Prefill, 1, 5), row(1, Role::Prefill, 1, 0)];
    bed.prefill_classes = vec![1, 2];
    bed.encoders = vec![
        row(0, Role::Encode, 0, 0),
        row(1, Role::Encode, 0, 4),
        row(2, Role::Encode, 0, 2),
    ];
    bed.facts = vec![pushing(1), pushing(2), pushing(2)];
    bed.keys = vec![Identity::hashed(11), Identity::hashed(12)];
    assert_eq!(bed.placed(), ((1, 0), vec![2, 2], false));
}

#[test]
fn a_request_without_items_reads_no_encoder_and_places_none() {
    let mut bed = Bed::new();
    bed.keys.clear();
    bed.encoders = vec![unready(row(0, Role::Encode, 0, 0))];
    bed.facts = vec![EncoderFacts::default()];
    bed.prefill_classes = vec![0];
    assert_eq!(bed.placed(), ((0, 0), vec![], false));
}

#[test]
fn the_one_key_is_placed_under_a_class_that_pushes_and_under_no_other() {
    let mut bed = Bed::new();
    bed.keys = vec![Identity::hashed(11), Identity::hashed(12)];
    bed.one = Some(Identity::hashed(99));
    assert_eq!(bed.placed(), ((0, 0), vec![0], true));
    bed.facts = vec![EncoderFacts {
        class: 1,
        ..EncoderFacts::default()
    }];
    assert_eq!(bed.placed(), ((0, 0), vec![0, 0], false));
    // A class that pushes and a request that carries no one key.
    bed.facts = vec![pushing(1)];
    bed.one = None;
    assert_eq!(bed.placed(), ((0, 0), vec![0, 0], false));
}

#[test]
fn the_admitted_rows_answer_by_their_place_in_the_hosts_pools() {
    let mut bed = Bed::new();
    // The idle prefill's class has no open encoder; the idle decode takes
    // no image and the request asks one of its decode.
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 9)];
    bed.prefill_classes = vec![1, 2];
    bed.prefill_affinity = vec![AffinityFacts::default(); 2];
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
    bed.encode_affinity = vec![AffinityFacts::default(); 2];
    bed.decode_images = true;
    assert_eq!(bed.placed(), ((1, 1), vec![1], false));
}

#[test]
fn a_gap_no_retry_closes_is_named_by_membership() {
    // An encoder that takes the request is registered and not ready.
    let mut bed = Bed::new();
    bed.encoders = vec![unready(row(0, Role::Encode, 0, 0))];
    assert_eq!(bed.stage(), "encoders");
    // No encoder is registered.
    bed.encoders.clear();
    bed.facts.clear();
    assert_eq!(bed.stage(), "encoders");

    // The only encoder takes no image.
    let mut bed = Bed::new();
    bed.encoders = vec![Candidate {
        vision: false,
        ..row(0, Role::Encode, 0, 0)
    }];
    assert_eq!(bed.stage(), "encode-vision");
    // Without an image the same encoder serves.
    bed.images = 0;
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));

    // A pushing encoder that states no end.
    let mut bed = Bed::new();
    bed.facts = vec![pushing(0)];
    assert_eq!(bed.stage(), "ec-compatible");
    // An encoder states an end no prefill of a pair states.
    let mut bed = Bed::new();
    bed.prefill_classes = vec![0];
    assert_eq!(bed.stage(), "ec-compatible");

    // References, and no encoder that processes them.
    let mut bed = Bed::new();
    bed.references = true;
    assert_eq!(bed.stage(), "encode-media");
    bed.facts = vec![EncoderFacts {
        references: true,
        ..pushing(1)
    }];
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));

    // The decode must take images and none of a served pair does.
    let mut bed = Bed::new();
    bed.decodes = vec![Candidate {
        vision: false,
        ..row(0, Role::Decode, 1, 0)
    }];
    bed.decode_images = true;
    assert_eq!(bed.stage(), "decode-vision");
}

#[test]
fn a_shed_of_the_pair_is_returned_as_the_pair_gave_it() {
    let mut bed = Bed::new();
    bed.prefills = vec![unready(row(0, Role::Prefill, 1, 0))];
    assert_eq!(bed.stage(), "healthy");
    // A class with no open encoder is no candidate of the pair: beside an
    // open encoder of another class the prefill pool is empty.
    let mut bed = Bed::new();
    bed.encoders = vec![
        unready(row(0, Role::Encode, 0, 0)),
        row(1, Role::Encode, 0, 0),
    ];
    bed.facts = vec![pushing(1), pushing(2)];
    assert_eq!(bed.stage(), "none");
}

#[test]
fn a_column_of_another_length_and_a_binding_of_another_plan_decline() {
    let declined = |bed: &Bed, engine: &Engine| match bed.decide_with(engine).0 {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by),
        other => panic!("a shed, not {other:?}"),
    };
    let column = (ShedReason::PolicyDeclined, "encode-column");
    let mut bed = Bed::new();
    bed.facts.clear();
    assert_eq!(declined(&bed, &engine()), column);
    let mut bed = Bed::new();
    bed.prefill_classes.push(1);
    assert_eq!(declined(&bed, &engine()), column);
    let mut bed = Bed::new();
    bed.encode_affinity = vec![AffinityFacts::default(); 3];
    assert_eq!(declined(&bed, &engine()), column);

    let pair = PolicySpec::pair_presets("power_of_two", "power_of_two")
        .expect("the presets")
        .validate()
        .expect("the form validates");
    assert_eq!(
        declined(&Bed::new(), &pair),
        (ShedReason::PolicyDeclined, "binding")
    );
}

/// What the pair entry answers, this entry answers the same when it admits
/// every row: a pick with its alternates, a shed, and the overload veto,
/// with an item and without. No built-in stage issues a queue ticket yet,
/// so none is read here.
#[test]
fn the_pairs_own_answers_read_the_same_through_both_entries() {
    let engine = engine();
    let mut alternates = Bed::new();
    alternates.decodes = vec![row(0, Role::Decode, 1, 2), row(1, Role::Decode, 1, 1)];
    let mut shed = Bed::new();
    shed.prefills = vec![unready(row(0, Role::Prefill, 1, 0))];
    let mut vetoed = Bed::new();
    vetoed.decodes = vec![Candidate {
        health: Health {
            healthy: true,
            overloaded: true,
            ..Health::default()
        },
        ..row(0, Role::Decode, 1, 0)
    }];
    // The beds are what they are named for.
    assert!(
        matches!(alternates.pair(&engine), Decision::Pick(pick) if pick.targets.len() == 2),
        "{:?}",
        alternates.pair(&engine)
    );
    assert!(
        matches!(shed.pair(&engine), Decision::Shed(shed) if shed.emptied_by == "healthy"),
        "{:?}",
        shed.pair(&engine)
    );
    assert!(
        matches!(vetoed.pair(&engine), Decision::Shed(shed) if shed.reason == ShedReason::Saturated),
        "{:?}",
        vetoed.pair(&engine)
    );
    for (name, mut bed) in [
        ("alternates", alternates),
        ("a shed", shed),
        ("the veto", vetoed),
    ] {
        let pair = bed.pair(&engine);
        assert_eq!(bed.decide_with(&engine).0, pair, "{name}, with an item");
        bed.keys.clear();
        assert_eq!(bed.decide_with(&engine).0, pair, "{name}, without items");
    }
}

/// Each key draws its own sequence, derived from the request's seed: the
/// items of one request over a tie of encoders do not follow one draw to
/// one encoder.
#[test]
fn the_items_of_one_request_spread_over_a_tie_of_encoders() {
    let mut bed = Bed::new();
    bed.encoders = (0..8).map(|at| row(at, Role::Encode, 0, 0)).collect();
    bed.facts = vec![pushing(1); 8];
    bed.keys = (0..6).map(|item| Identity::hashed(100 + item)).collect();
    let (_, encoders, one) = bed.placed();
    assert!(!one);
    assert_eq!(encoders.len(), 6);
    let first = encoders[0];
    assert!(
        encoders.iter().any(|&encoder| encoder != first),
        "six keys over a tie of eight follow one draw: {encoders:?}"
    );
}

/// References that carry images are attended to by the decode: the pair is
/// picked among the decodes that take images when a served pair has one,
/// and among every decode when none has or the request carries no image.
#[test]
fn references_with_images_take_a_decode_with_vision_when_a_pair_has_one() {
    let blind = |at: u32, in_flight: u32| Candidate {
        vision: false,
        ..row(at, Role::Decode, 1, in_flight)
    };
    let mut bed = Bed::new();
    bed.references = true;
    bed.facts = vec![EncoderFacts {
        references: true,
        ..pushing(1)
    }];
    // The idle decode takes no image, the busy one does.
    bed.decodes = vec![blind(0, 0), row(1, Role::Decode, 1, 9)];
    assert_eq!(bed.placed(), ((0, 1), vec![0], false));
    // Without an image the references ask nothing of the decode.
    bed.images = 0;
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
    // No decode of the pair takes images: the request is placed all the same.
    bed.images = 1;
    bed.decodes = vec![blind(0, 0), blind(1, 9)];
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
}

#[test]
fn a_served_pair_without_vision_ahead_leaves_the_question_of_vision_open() {
    // Two pair classes under no table: the idle prefill's decode takes no
    // image, the busy prefill's does, and both prefills are served. The
    // pass goes on past the first served prefill, so a request that asks
    // an image of its decode is placed on the pair that takes one.
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 2, 9)];
    bed.prefill_classes = vec![1, 1];
    bed.decodes = vec![
        Candidate {
            vision: false,
            ..row(0, Role::Decode, 1, 0)
        },
        row(1, Role::Decode, 2, 9),
    ];
    bed.decode_images = true;
    assert_eq!(bed.placed(), ((1, 1), vec![0], false));
    // References with an image take that pair too.
    bed.decode_images = false;
    bed.references = true;
    bed.facts = vec![EncoderFacts {
        references: true,
        ..pushing(1)
    }];
    assert_eq!(bed.placed(), ((1, 1), vec![0], false));
}

/// A filter that refuses every row under the key of hash 12.
#[derive(Debug)]
struct RefusesTheKey;

impl crate::framework::Filter for RefusesTheKey {
    fn name(&self) -> &'static str {
        "refuses-the-key"
    }

    fn keep(&self, facts: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        facts.session_key != Identity::hashed(12)
    }
}

/// A filter that keeps an encoder under the encode phase alone.
#[derive(Debug)]
struct EncodersUnderTheirPhase;

impl crate::framework::Filter for EncodersUnderTheirPhase {
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
    };
    let items = EncodeItems {
        keys: &bed.keys,
        one: bed.one,
        references: bed.references,
        decode_images: bed.decode_images,
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
    for profile in &mut engine.profiles {
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
    for profile in &mut engine.profiles {
        profile.filters.push(Box::new(EncodersUnderTheirPhase));
    }
    let (decision, placed) = Bed::new().decide_with(&engine);
    assert!(decision.pick().is_some(), "a pick, not {decision:?}");
    assert_eq!(placed.encoders().len(), 1);
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
    };
    let items = EncodeItems {
        keys: &bed.keys,
        one: bed.one,
        references: bed.references,
        decode_images: bed.decode_images,
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
    let sets = [&side.open, &side.member, &side.decodes_of, &side.vision_of];
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
    // The store numbers the encode classes over the whole pool and hands a
    // gateway its view, the rows of another gateway's host left out. A
    // class then stands above the count of the rows handed: it is a class
    // still, from the first number the sets' bits do not hold on.
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
