//! The encode side of a pair decision, over small pools built by hand: what
//! the recorded sheet does not hold (a column of another length, a binding
//! of another plan, the rows read back from the admitted pools, the
//! affinity columns) and each gap by its stage.

use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use super::*;
use crate::{
    document::{spec, spec::PolicySpec},
    engine::{
        framework,
        framework::PairTable,
        view::{Health, MediaCounts, Role, WorkerId},
    },
    load::signals::{SignalCell, SignalCells, SignalColumn},
    tracking::counts::tests::Lent,
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

/// The position of the encode profile among `engine`'s profiles.
fn encode_profile(engine: &Engine) -> Option<u16> {
    let mut names = engine
        .profiles()
        .iter()
        .map(|profile| profile.name.as_str());
    u16::try_from(names.position(|name| name == "encode")?).ok()
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
    /// Each pool's signal cells, one a row under one signal; empty for a
    /// pool that is handed none.
    prefill_cells: Vec<SignalCell>,
    decode_cells: Vec<SignalCell>,
    encode_cells: Vec<SignalCell>,
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
            prefill_cells: Vec::new(),
            decode_cells: Vec::new(),
            encode_cells: Vec::new(),
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
        self.decide_asked(engine, &facts, &pool)
    }

    /// The encode entry's answer to `facts` over the bed's pools.
    fn decide_asked(
        &self,
        engine: &Engine,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
    ) -> (Decision, EncodePlaced) {
        fn cells(of: &[SignalCell]) -> Option<SignalCells> {
            (!of.is_empty()).then(|| SignalCells::new(of.to_vec(), 1))
        }
        let prefill = cells(&self.prefill_cells);
        let decode = cells(&self.decode_cells);
        let encoders = cells(&self.encode_cells);
        let pools = PairPools {
            prefill_signal_cells: prefill.as_ref(),
            decode_signal_cells: decode.as_ref(),
            ..self.pools()
        };
        let encode = EncodePool {
            encoders: &self.encoders,
            facts: &self.facts,
            prefill_classes: &self.prefill_classes,
            affinity: &self.encode_affinity,
            signal_cells: encoders.as_ref(),
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
            facts,
            pool,
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

/// A lent table numbers the rows of the request's own pool, and the encoders
/// are numbered again from zero: the encode side reads each encoder's own
/// count and nothing of the table.
#[test]
fn the_encode_side_reads_no_count_lent_for_another_pool() {
    let mut bed = Bed::new();
    // The first encoder holds five requests and the second none.
    bed.encoders = vec![row(0, Role::Encode, 0, 5), row(1, Role::Encode, 0, 0)];
    bed.facts = vec![pushing(1), pushing(1)];
    // Another pool's table says the opposite of those two positions.
    let mut others = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 9)];
    let lent = Lent::take(&mut others);
    let (facts, pool) = bed.asked();
    let (decision, placed) = bed.decide_asked(&engine(), &lent.beside(&facts), &pool);
    assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
    let encoders: Vec<u32> = placed.encoders().iter().map(|t| t.position).collect();
    assert_eq!(encoders, vec![1]);
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

/// A hard filter that refuses one row, by its id.
#[derive(Debug)]
struct RefusesOne(WorkerId);

impl framework::Filter for RefusesOne {
    fn name(&self) -> &'static str {
        "refuses-one"
    }

    fn keep(&self, _: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
        row.id != self.0
    }
}

/// A row over its thresholds.
fn overloaded(row: Candidate) -> Candidate {
    Candidate {
        health: Health {
            overloaded: true,
            ..row.health
        },
        ..row
    }
}

/// Encoders the overload veto left none of are a saturated fleet: the shed
/// names the veto's stage and its reason is saturation, as a pair's side
/// sheds its own rows. A stage ahead of the veto that leaves none is a
/// missing endpoint as before, and so is one behind it that refuses an
/// encoder the veto kept. An encoder that does not take the request is not
/// counted, and under a veto its host lifted the encoders serve.
#[test]
fn encoders_the_veto_left_none_of_shed_as_saturation() {
    let stock = engine();
    let saturated = (ShedReason::Saturated, "not-overloaded");
    let missing = (ShedReason::NoEndpoints, "encoders");
    // Every encoder that takes the request is over its thresholds.
    let mut bed = Bed::new();
    bed.encoders = vec![
        overloaded(row(0, Role::Encode, 0, 0)),
        overloaded(row(1, Role::Encode, 0, 0)),
    ];
    bed.facts = vec![pushing(1), pushing(1)];
    let shed = bed.decide().0;
    assert_eq!(shed_of(&shed), saturated);
    // The veto is a stage of the encode profile, and the trace says so.
    assert_eq!(shed.trace().made_under(), encode_profile(&stock));
    assert!(shed.trace().made_under().is_some());
    // A pair's side says the same of its own rows.
    let mut side = Bed::new();
    side.decodes = vec![overloaded(row(0, Role::Decode, 1, 0))];
    assert_eq!(shed_of(&side.pair(&stock)), saturated);
    // The host lifted the veto on the encode side: the item is placed.
    let (facts, _) = bed.asked();
    let cx = Constraints::default();
    let lifted = Constraints {
        overload_lifted: true,
        ..Constraints::default()
    };
    let sides = [&cx, &cx, &lifted];
    let (decision, placed) = decide_under(&bed, &stock, &facts, bed.pools(), sides);
    assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
    assert_eq!(placed.encoders().len(), 1);

    // An unready encoder beside one over its thresholds was refused ahead
    // of the veto: the veto still left none.
    bed.encoders[1] = unready(row(1, Role::Encode, 0, 0));
    assert_eq!(shed_of(&bed.decide().0), saturated);
    // No encoder reached the veto: a missing endpoint.
    bed.encoders[0] = unready(row(0, Role::Encode, 0, 0));
    assert_eq!(shed_of(&bed.decide().0), missing);

    // An encoder the veto kept, refused by a hard filter listed behind it:
    // a missing endpoint, as a pair's side names rows so filtered.
    bed.encoders = vec![
        overloaded(row(0, Role::Encode, 0, 0)),
        row(1, Role::Encode, 0, 0),
    ];
    let mut behind = engine();
    for profile in &mut behind.profiles {
        profile
            .filters
            .push(Box::new(RefusesOne(bed.encoders[1].id)));
    }
    assert_eq!(shed_of(&bed.decide_with(&behind).0), missing);

    // An encoder over its thresholds that takes no image is not counted:
    // the one that takes it was refused ahead of the veto.
    bed.encoders = vec![
        unready(row(0, Role::Encode, 0, 0)),
        Candidate {
            vision: false,
            ..overloaded(row(1, Role::Encode, 0, 0))
        },
    ];
    assert_eq!(shed_of(&bed.decide().0), missing);
    // Nor is one over its thresholds that states no end.
    bed.encoders[1] = overloaded(row(1, Role::Encode, 0, 0));
    bed.facts[1] = pushing(0);
    assert_eq!(shed_of(&bed.decide().0), missing);
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
    // A shed made once the encode profile is resolved says that profile.
    // One made where no binding gives a profile says none.
    let declined = |bed: &Bed, engine: &Engine| match bed.decide_with(engine).0 {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by, shed.trace.made_under()),
        other => panic!("a shed, not {other:?}"),
    };
    let encode = encode_profile(&engine());
    assert!(encode.is_some(), "the plan has an encode profile");
    let column = (ShedReason::PolicyDeclined, "encode-column", encode);
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
        (ShedReason::PolicyDeclined, "binding", None)
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
    vetoed.decodes = vec![overloaded(row(0, Role::Decode, 1, 0))];
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

/// A request with no item still asks what it asks of its decode. Images
/// asked of the decode keep it off the encode entry's early answer: the pair
/// is picked among the decodes that take images, and where none takes one
/// the shed names the stage.
#[test]
fn a_request_without_items_that_asks_images_of_its_decode_takes_a_decode_with_vision() {
    let blind = |at: u32, in_flight: u32| Candidate {
        vision: false,
        ..row(at, Role::Decode, 1, in_flight)
    };
    let mut bed = Bed::new();
    bed.keys.clear();
    bed.decode_images = true;
    // The idle decode takes no image, the busy one does.
    bed.decodes = vec![blind(0, 0), row(1, Role::Decode, 1, 9)];
    assert_eq!(bed.placed(), ((0, 1), vec![], false));
    bed.decodes = vec![blind(0, 0), blind(1, 9)];
    assert_eq!(bed.stage(), "decode-vision");
}

/// References that carry images ask a decode with vision of a request
/// with no item too, and nothing of one that carries no image.
#[test]
fn references_with_images_and_no_item_take_a_decode_with_vision() {
    let blind = Candidate {
        vision: false,
        ..row(0, Role::Decode, 1, 0)
    };
    let mut bed = Bed::new();
    bed.keys.clear();
    bed.references = true;
    // The idle decode takes no image, the busy one does.
    bed.decodes = vec![blind, row(1, Role::Decode, 1, 9)];
    assert_eq!(bed.placed(), ((0, 1), vec![], false));
    bed.images = 0;
    assert_eq!(bed.placed(), ((0, 0), vec![], false));
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

impl framework::Filter for RefusesTheKey {
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
    for profile in &mut engine.profiles {
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

/// A stage that shows its rows to a plugin and counts each time it is
/// asked: a scope to open, a row to keep.
#[derive(Debug)]
struct ShowsItsRows(Arc<AtomicU32>);

impl framework::Lifecycle for ShowsItsRows {}

impl framework::DecisionScope for ShowsItsRows {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn close(&self) {}
}

impl framework::Filter for ShowsItsRows {
    fn name(&self) -> &'static str {
        "shows-its-rows"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        self.0.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn scope(&self) -> Option<&dyn framework::DecisionScope> {
        Some(self)
    }
}

/// A filter that refuses an encoder to a request that carries a pin.
#[derive(Debug)]
struct NoEncoderUnderAPin;

impl framework::Filter for NoEncoderUnderAPin {
    fn name(&self) -> &'static str {
        "no-encoder-under-a-pin"
    }

    fn keep(&self, facts: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
        !matches!(row.role, Role::Encode) || facts.pinned_at == 0
    }
}

/// The bed's decision over `pools` under `facts` and each side's
/// constraints: the prefill's, the decode's and the encode's.
fn decide_under(
    bed: &Bed,
    engine: &Engine,
    facts: &RequestFacts<'_>,
    pools: PairPools<'_>,
    sides: [&Constraints<'_>; 3],
) -> (Decision, EncodePlaced) {
    let (_, pool) = bed.asked();
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
    };
    let [cx, decode_cx, encode_cx] = sides;
    let mut placed = EncodePlaced::new();
    let decision = engine.decide_encode_pair(
        facts,
        &pool,
        pools,
        encode,
        items,
        cx,
        decode_cx,
        encode_cx,
        &mut Scratch::new(),
        &mut placed,
    );
    (decision, placed)
}

/// The reason and the stage of a shed.
fn shed_of(decision: &Decision) -> (ShedReason, &'static str) {
    match decision {
        Decision::Shed(shed) => (shed.reason, shed.emptied_by),
        other => panic!("a shed, not {other:?}"),
    }
}

/// A request with items under an encode profile that shows its rows to a
/// plugin is declined before a stage of the profile is asked. Without items
/// the profile is not read, and the pair is decided.
#[test]
fn an_encode_profile_that_shows_its_rows_declines_a_request_with_items() {
    let asked = Arc::new(AtomicU32::new(0));
    let mut engine = engine();
    let encode = engine
        .binding("m", Phase::Prefill)
        .and_then(|binding| match binding.plan {
            Plan::Epd { encode } => Some(usize::from(encode)),
            Plan::Single | Plan::PdPair => None,
        })
        .expect("the epd plan");
    let shows = ShowsItsRows(Arc::clone(&asked));
    engine.profiles[encode].filters.push(Box::new(shows));
    engine.marks[encode] = framework::Marks::of(&engine.profiles[encode], engine.orders[encode]);

    let (decision, placed) = Bed::new().decide_with(&engine);
    assert_eq!(
        shed_of(&decision),
        (ShedReason::PolicyDeclined, "encode-scope")
    );
    assert!(placed.encoders().is_empty());
    let mut bare = Bed::new();
    bare.keys.clear();
    let (decision, _) = bare.decide_with(&engine);
    assert!(decision.pick().is_some(), "a pick, not {decision:?}");
    assert_eq!(asked.load(Ordering::Relaxed), 0, "no stage is asked");
}

/// The pair entry declines a mask beside the rows a side shows a plugin
/// (`written-rows`), and it reads the mask as this entry hands it: numbered
/// with the rows a cut leaves. A mask on a row the cut left out masks none
/// of them and declines nothing.
#[test]
fn a_mask_on_a_row_cut_out_declines_nothing_for_a_side_that_shows_its_rows() {
    let mut engine = engine();
    let binding = engine.binding("m", Phase::Prefill).expect("the binding");
    let prefill = usize::from(binding.profile);
    let shows = ShowsItsRows(Arc::new(AtomicU32::new(0)));
    engine.profiles[prefill].filters.push(Box::new(shows));
    engine.marks[prefill] = framework::Marks::of(&engine.profiles[prefill], engine.orders[prefill]);
    // The first prefill's class holds no encoder: the pair is decided over
    // the second alone.
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 0)];
    bed.prefill_classes = vec![2, 1];
    let (facts, _) = bed.asked();
    let cx = Constraints::default();
    let decide = |mask: &[u32]| {
        let masked = masking(mask);
        decide_under(&bed, &engine, &facts, bed.pools(), [&masked, &cx, &cx]).0
    };
    // The mask names the row the cut left out.
    assert_eq!(pair_of(&decide(&[0])), (1, 0));
    // It names the row admitted: a mask beside the rows shown.
    assert_eq!(
        shed_of(&decide(&[1])),
        (ShedReason::PolicyDeclined, "written-rows")
    );
}

/// A pin is a position in another pool, and the encode side's stages are
/// handed none.
#[test]
fn the_encode_side_is_handed_no_pin() {
    let mut engine = engine();
    for profile in &mut engine.profiles {
        profile.filters.push(Box::new(NoEncoderUnderAPin));
    }
    let bed = Bed::new();
    let pinned = RequestFacts {
        pinned_at: 1,
        ..bed.asked().0
    };
    let cx = Constraints::default();
    let (decision, placed) = decide_under(&bed, &engine, &pinned, bed.pools(), [&cx; 3]);
    assert!(decision.pick().is_some(), "a pick, not {decision:?}");
    assert_eq!(placed.encoders().len(), 1);
}

/// A pair column of another length than its pool declines as the pair
/// entry declines it, where a cut would leave a column that fits.
#[test]
fn a_pair_column_of_another_length_declines_before_a_cut() {
    let engine = engine();
    // The second prefill's class has no encoder and the second decode takes
    // no image: each side is cut to its first row.
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 0)];
    bed.prefill_classes = vec![1, 2];
    bed.decodes = vec![
        row(0, Role::Decode, 1, 0),
        Candidate {
            vision: false,
            ..row(1, Role::Decode, 1, 0)
        },
    ];
    bed.decode_images = true;
    let (facts, _) = bed.asked();
    let cx = Constraints::default();
    let none: &[AffinityFacts] = &[];
    let short = [AffinityFacts::default(); 1];
    let long = [AffinityFacts::default(); 3];
    for (prefill_affinity, decode_affinity) in [
        (&short[..], none),
        (none, &short[..]),
        (&long[..], none),
        (none, &long[..]),
    ] {
        let pools = PairPools {
            prefill_affinity,
            decode_affinity,
            ..bed.pools()
        };
        let (decision, _) = decide_under(&bed, &engine, &facts, pools, [&cx; 3]);
        assert_eq!(
            shed_of(&decision),
            (ShedReason::PolicyDeclined, "affinity-column")
        );
        assert_eq!(decision.trace().made_under(), None, "as the pair entry");
    }
}

/// A hard filter that counts what it is asked and keeps every row.
#[derive(Debug)]
struct CountsItsAsks(Arc<AtomicU32>);

impl framework::Filter for CountsItsAsks {
    fn name(&self) -> &'static str {
        "counts-its-asks"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        self.0.fetch_add(1, Ordering::Relaxed);
        true
    }
}

/// A request that carries a column of its own, or names rows, declines as
/// the pair entry declines it, before a stage of the encode side is asked.
#[test]
fn a_request_with_a_column_or_named_rows_declines_before_a_stage_is_asked() {
    let asked = Arc::new(AtomicU32::new(0));
    let mut engine = engine();
    for profile in &mut engine.profiles {
        let counts = CountsItsAsks(Arc::clone(&asked));
        profile.filters.push(Box::new(counts));
    }
    let bed = Bed::new();
    let cx = Constraints::default();
    let holders = [crate::NamedHolder {
        row: 1,
        overlap_blocks: 8,
        predicted_blocks: 0,
    }];
    let named = RequestFacts {
        named: Some(crate::NamedAffinity {
            holders: &holders,
            ..crate::NamedAffinity::default()
        }),
        ..bed.asked().0
    };
    let column = [AffinityFacts::default()];
    let carried = RequestFacts {
        affinity: &column,
        ..bed.asked().0
    };
    for facts in [&named, &carried] {
        let (decision, placed) = decide_under(&bed, &engine, facts, bed.pools(), [&cx; 3]);
        assert_eq!(
            shed_of(&decision),
            (ShedReason::PolicyDeclined, "affinity-column")
        );
        assert_eq!(decision.trace().made_under(), None, "as the pair entry");
        assert!(placed.encoders().is_empty());
        assert_eq!(asked.load(Ordering::Relaxed), 0, "no stage is asked");
    }
    // The same engine decides a request that carries neither, and asks.
    let (decision, _) = bed.decide_with(&engine);
    assert!(decision.pick().is_some(), "a pick, not {decision:?}");
    assert_ne!(asked.load(Ordering::Relaxed), 0);
}

/// The constraints of a host whose standing veto masks the rows `at`.
fn masking(at: &[u32]) -> Constraints<'_> {
    Constraints {
        stalled: at,
        ..Constraints::default()
    }
}

/// The prefill's and the decode's rows of a pick.
fn pair_of(decision: &Decision) -> (u32, u32) {
    let Decision::Pick(Pick { targets, .. }) = decision else {
        panic!("a pick, not {decision:?}");
    };
    let target = targets.first().expect("a target");
    (
        target.position,
        target.pair.expect("a pair").decode_position,
    )
}

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
    let filters = &mut engine.profiles[encode].filters;
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
        let filters = &mut engine.profiles[profile].filters;
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

/// The `epd` plan with every side turning its pool's rotation.
fn turning() -> Engine {
    let spec = PolicySpec::epd_presets("round_robin", "round_robin", "round_robin");
    spec.expect("the presets")
        .validate()
        .expect("the form validates")
}

/// The sizes a list is turned at: a small one, and the first at which a
/// single decision draws before it reads its pool.
const TURNED: [u32; 2] = [2, framework::DRAW_FROM as u32];

/// The encoders offered under two classes are two lists of one pool, and
/// each turns a rotation of its own, also where they have one size: a pair
/// that alternates between the classes reaches every encoder of each. A
/// key's run is a whole pass, which turns a rotation by the rows it keeps.
#[test]
fn the_encoders_offered_under_two_classes_turn_rotations_apart() {
    for size in TURNED {
        let engine = turning();
        // Two encode classes of `size` encoders each and a prefill of each,
        // one item a request.
        let mut bed = Bed::new();
        bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 0)];
        bed.prefill_classes = vec![1, 2];
        bed.encoders = (0..2 * size)
            .map(|at| row(at, Role::Encode, 0, 0))
            .collect();
        bed.facts = (0..2 * size).map(|at| pushing(1 + at / size)).collect();
        let mut reached = vec![0; bed.encoders.len()];
        for _ in 0..4 * size {
            let (decision, placed) = bed.decide_with(&engine);
            assert!(decision.pick().is_some(), "a pick, not {decision:?}");
            reached[placed.encoders()[0].position as usize] += 1;
        }
        assert!(
            reached.iter().all(|&turns| turns == 2),
            "{size}: {reached:?}"
        );
    }
}

/// Two cuts of the prefills are two lists of one pool: two kinds of media,
/// each served by a class of its own, alternate, and each kind's requests
/// reach every prefill of its class. A side's run is a whole pass too.
#[test]
fn two_cuts_of_the_prefills_turn_rotations_apart() {
    for size in TURNED {
        let engine = turning();
        // The first encoder takes images and the second references.
        let kind = |references: bool| {
            let mut bed = Bed::new();
            bed.prefills = (0..2 * size)
                .map(|at| row(at, Role::Prefill, 1, 0))
                .collect();
            bed.prefill_classes = (0..2 * size).map(|at| 1 + at / size).collect();
            bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
            bed.encoders[1].vision = false;
            bed.facts = vec![pushing(1), pushing(2)];
            bed.facts[1].references = true;
            bed.references = references;
            bed.images = u16::from(!references);
            bed
        };
        let kinds = [kind(false), kind(true)];
        let mut reached = vec![0; kinds[0].prefills.len()];
        for bed in kinds.iter().cycle().take(4 * size as usize) {
            let (decision, _) = bed.decide_with(&engine);
            reached[pair_of(&decision).0 as usize] += 1;
        }
        assert!(
            reached.iter().all(|&turns| turns == 2),
            "{size}: {reached:?}"
        );
    }
}

/// A side's held list takes its room where a hold is first numbered, one
/// place a row admitted. A mask with no hold takes none, and a fuller hold
/// over as many rows grows nothing.
#[test]
fn a_sides_held_list_is_sized_once_where_a_hold_is_numbered() {
    let mut side = Admitted::default();
    for at in 0..8 {
        side.push(&row(at, Role::Prefill, 1, 0), at as usize, &[]);
    }
    side.take_room();
    side.renumber(&[1, 3], &[]);
    assert_eq!(side.held.capacity(), 0, "a mask with no hold takes no room");
    side.masked.clear();
    side.renumber(&[1, 3], &[3]);
    assert_eq!((&side.masked[..], &side.held[..]), (&[1, 3][..], &[3][..]));
    let room = side.held.capacity();
    assert!(
        room >= side.rows.len(),
        "the first hold takes a place a row"
    );
    side.masked.clear();
    let every: Vec<u32> = (0..8).collect();
    side.renumber(&every, &every);
    assert_eq!(side.held, every);
    assert_eq!(side.held.capacity(), room, "a fuller hold grew the list");
}
