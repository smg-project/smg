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
    document::spec::{self, PolicySpec},
    engine::{
        framework::{self, PairTable},
        view::{Health, MediaCounts, Role, WorkerId},
        LiftedVeto, Pick, PlacedBy,
    },
    load::signals::{SignalCell, SignalCells, SignalColumn},
    policies::consistent_hashing::InOrder,
    tracking::lease::{Lease, Side},
};

/// A profile of an engine no other engine shares yet, to change in a case.
fn own(profile: &mut Arc<Profile>) -> &mut Profile {
    Arc::get_mut(profile).expect("a profile held once")
}

/// The document of [`engine`]: the `epd` plan with every side taking the row
/// that holds fewer requests.
fn document() -> PolicySpec {
    let mut spec = PolicySpec::epd_presets("power_of_two", "power_of_two", "power_of_two")
        .expect("the presets");
    for profile in spec.profiles.values_mut() {
        profile.pick = "count-first".into();
    }
    spec
}

/// The `epd` plan with every side taking the row that holds fewer requests.
fn engine() -> Engine {
    document().validate().expect("the form validates")
}

/// [`engine`] under a sessions rule with no cap, which leaves a new key to
/// the profile and says `word` of a request it places alone.
fn ruled(word: spec::RulePlaced) -> Engine {
    let mut spec = document();
    let stock = PolicySpec::preset("manual").expect("a built-in preset");
    let mut rule = stock.bindings[0].sessions.clone();
    let sessions = rule.as_mut().expect("the stock rule");
    (sessions.assignment, sessions.rule_placed) = (spec::Assignment::Delegate, word);
    spec.view.headers.push(sessions.sticky_header.clone());
    spec.bindings[0].sessions = rule;
    spec.validate().expect("a rule under the three-sided plan")
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
        in_flight,
        pair_class,
        vision: true,
        key: id as u32,
        ..Candidate::ready(at, WorkerId(id), role)
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
    /// The videos the request states beside its images; none until a case
    /// states one.
    video: u16,
    /// The wires the request's host states to the encode side; none until a
    /// case states one.
    wires: WireModes,
    /// The ring the request's caller lends for the items; none until a case
    /// lends one.
    ring: Option<Box<dyn LentRing>>,
    /// The hard demand the encode side's constraints state, and what each
    /// encoder states for one; none until a case states them.
    demands: Demands,
    stated: Vec<DemandFacts>,
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
            video: 0,
            wires: WireModes::NONE,
            ring: None,
            demands: Demands::default(),
            stated: Vec::new(),
        }
    }

    /// The request's facts and the pool's, as both entries are handed them.
    fn asked(&self) -> (RequestFacts<'static>, PoolFacts<'static>) {
        let facts = RequestFacts {
            model: "m",
            media: MediaCounts {
                images: self.images,
                video: self.video,
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
            prefill_affinity: &self.prefill_affinity,
            ..PairPools::of(&self.prefills, &self.decodes)
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
        self.decide_asked_in(engine, facts, pool, &mut Scratch::new())
    }

    /// The encode entry's answer over the bed's pools, made with `scratch`:
    /// the scratch keeps the decision's word of the list it shed over.
    fn decide_in(&self, engine: &Engine, scratch: &mut Scratch) -> Decision {
        let (facts, pool) = self.asked();
        self.decide_asked_in(engine, &facts, &pool, scratch).0
    }

    /// [`Self::decide_asked`], made with `scratch`.
    fn decide_asked_in(
        &self,
        engine: &Engine,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        scratch: &mut Scratch,
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
            demands: &self.stated,
        };
        let items = EncodeItems {
            keys: &self.keys,
            one: self.one,
            references: self.references,
            decode_images: self.decode_images,
            ring: self.ring.as_deref(),
        };
        let cx = Constraints::default();
        let encode_cx = Constraints {
            wire_modes: self.wires,
            demands: self.demands,
            ..Constraints::default()
        };
        let mut placed = EncodePlaced::new();
        let decision = engine.decide_encode_pair(
            facts,
            pool,
            pools,
            encode,
            items,
            &cx,
            &cx,
            &encode_cx,
            scratch,
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

/// Each shape of the encode pool that declines a request with an item: an
/// empty encode pool beside the bed's prefill, then each of the three
/// columns of another length than its list
/// (`a_column_of_another_length_and_a_binding_of_another_plan_decline`).
fn shapes() -> [fn(&mut Bed); 4] {
    [
        |bed| {
            bed.encoders.clear();
            bed.facts.clear();
            bed.prefill_classes.clear();
        },
        |bed| bed.facts.clear(),
        |bed| bed.prefill_classes.push(1),
        |bed| bed.encode_affinity = vec![AffinityFacts::default(); 3],
    ]
}

/// A request with no item that states `images` images, asked over each of
/// those shapes. Over each it takes the pair entry's pair, and no encoder
/// is placed.
fn a_request_without_items_takes_its_pair(images: u16, decode_images: bool) {
    for (at, shape) in shapes().iter().enumerate() {
        let mut bed = Bed::new();
        bed.keys.clear();
        bed.images = images;
        bed.decode_images = decode_images;
        shape(&mut bed);
        let (decision, placed) = bed.decide();
        assert!(
            matches!(decision, Decision::Pick(_)),
            "shape {at}, {images} images: {decision:?}"
        );
        assert_eq!(decision, bed.pair(&engine()), "shape {at}");
        assert!(placed.encoders().is_empty(), "shape {at}");
    }
}

/// A request with no item is answered whatever its host handed in the
/// encode pool's columns: one that states an image, and one of text alone.
/// The classic router holds such a request to no encoder. It lists the
/// encoders for every request, and each of its tests that refuses a
/// request for want of one, or keeps a pair to the workers one hands to,
/// stands behind the test that the request has items
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2149`,
/// `:2187` and `:2215` on main 384b8fd6b).
#[test]
fn a_request_without_items_is_answered_whatever_the_encode_pools_columns_hold() {
    a_request_without_items_takes_its_pair(1, false);
    a_request_without_items_takes_its_pair(0, false);
}

/// So is one that asks images of its decode: the bed's decode takes them,
/// and the pair is the pair entry's. Over the same shapes a decode that
/// takes none leaves the request no pair, and the shed names the decode's
/// stage: what a request asks of its decode is read whatever the encode
/// pool holds. The classic router refuses that request by its pairs'
/// decodes alone, since a request with no item asks no encoder of a pair
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2187`
/// and `:2202-2206` on main 384b8fd6b).
#[test]
fn a_request_without_items_that_asks_images_of_its_decode_reads_no_encode_column() {
    a_request_without_items_takes_its_pair(1, true);
    for (at, shape) in shapes().iter().enumerate() {
        let mut bed = Bed::new();
        bed.keys.clear();
        bed.decode_images = true;
        bed.decodes = vec![Candidate {
            vision: false,
            ..row(0, Role::Decode, 1, 0)
        }];
        shape(&mut bed);
        assert_eq!(bed.stage(), "decode-vision", "shape {at}");
    }
}

#[test]
fn the_one_key_is_placed_under_a_class_that_pushes_and_under_no_other() {
    let mut bed = Bed::new();
    bed.keys = vec![Identity::hashed(11), Identity::hashed(12)];
    bed.one = Some(Identity::hashed(99));
    assert_eq!(bed.placed(), ((0, 0), vec![0], true));
    // A one key with no form is a one key all the same: the items are
    // placed once, with no affinity.
    bed.one = Some(Identity::NONE);
    assert_eq!(bed.placed(), ((0, 0), vec![0], true));
    bed.one = Some(Identity::hashed(99));
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

/// An encoder at `at` that holds `in_flight` requests, dialed on `wire`.
fn dialed_on(wire: WireModes, at: u32, in_flight: u32) -> Candidate {
    Candidate {
        wire_modes: wire,
        ..row(at, Role::Encode, 0, in_flight)
    }
}

/// An encoder dialed on no wire the request's host states is left out, and
/// the item goes to one that is: here the idle encoder is dialed over gRPC
/// and the busy one over HTTP. A host that states no wire leaves none out,
/// and one that states both takes the idle one.
///
/// The classic router tests an encoder's wire with the rest of what the
/// request asks of it, and keeps the encoders that pass
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2138`
/// and `:2147` on main 384b8fd6b).
#[test]
fn an_encoder_dialed_on_no_wire_the_request_states_is_left_out() {
    let mut bed = Bed::new();
    bed.encoders = vec![
        dialed_on(WireModes::GRPC, 0, 0),
        dialed_on(WireModes::HTTP, 1, 4),
    ];
    bed.facts = vec![pushing(1), pushing(1)];
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
    bed.wires = WireModes::HTTP;
    assert_eq!(bed.placed(), ((0, 0), vec![1], false));
    bed.wires = WireModes::HTTP.union(WireModes::GRPC);
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
    // An encoder that is not ready is no encoder to send to, on its wire
    // or off it: the shed is the encode side's, by state.
    bed.wires = WireModes::HTTP;
    bed.encoders[1] = unready(bed.encoders[1]);
    assert_eq!(bed.stage(), "encoders");
}

/// An encoder dialed on two wires takes a request held to one of them: a
/// row states the set of its wires, and one stated wire in the set is
/// enough. The idle encoder is dialed over HTTP and gRPC, and the busy one
/// over gRPC alone: a request held to gRPC takes the idle one, and so does
/// a request held to HTTP, where it is the only one left.
///
/// The classic router compares the request's wire with the one wire its
/// worker is dialed on
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2138`
/// on main 384b8fd6b, by
/// `model_gateway/src/routers/common/placement.rs:88-93`): it holds no
/// encoder on two.
#[test]
fn an_encoder_dialed_on_two_wires_takes_a_request_held_to_one_of_them() {
    let mut bed = Bed::new();
    bed.encoders = vec![
        dialed_on(WireModes::HTTP.union(WireModes::GRPC), 0, 0),
        dialed_on(WireModes::GRPC, 1, 4),
    ];
    bed.facts = vec![pushing(1), pushing(1)];
    bed.wires = WireModes::GRPC;
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
    bed.wires = WireModes::HTTP;
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
}

/// A prefill whose class holds no encoder on the request's wires is no
/// candidate of the pair. The idle prefill's one encoder is dialed over
/// gRPC, and the busy prefill's over HTTP: a request held to HTTP takes
/// the busy prefill and its encoder.
///
/// The classic router keeps a pair to the workers an encoder it kept hands
/// to
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2213-2216`
/// on main 384b8fd6b, over the encoders kept at `:2147`).
#[test]
fn a_prefill_whose_encoders_are_on_other_wires_is_not_the_pairs() {
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 5)];
    bed.prefill_classes = vec![1, 2];
    bed.encoders = vec![
        dialed_on(WireModes::GRPC, 0, 0),
        dialed_on(WireModes::HTTP, 1, 0),
    ];
    bed.facts = vec![pushing(1), pushing(2)];
    assert_eq!(bed.placed(), ((0, 0), vec![0], false));
    bed.wires = WireModes::HTTP;
    assert_eq!(bed.placed(), ((1, 0), vec![1], false));
}

/// A gap is named by the encoders on the request's wires. The encoder
/// dialed over HTTP processes no references, and the one over gRPC
/// processes them and takes images: for a request of references with an
/// image held to HTTP, no encoder takes the media. With no wire stated the
/// second encoder takes the request. Where the encoder over gRPC pushes,
/// the gap is the one of an encoder that lacks its role, on whatever wire
/// that encoder is dialed.
///
/// The classic router names a gap by the encoders that pass its test of
/// the wire: with none that would take the request but for its image, the
/// request's media are not supported
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2153-2164`
/// and `:2171-2172` on main 384b8fd6b). Its test of a pushing encoder
/// stands between the two and reads no wire (`:2165-2170`).
#[test]
fn a_gap_is_named_by_the_encoders_on_the_requests_wires() {
    let processes = |references: bool| EncoderFacts {
        class: 1,
        pushes: false,
        references,
    };
    let mut bed = Bed::new();
    bed.encoders = vec![
        dialed_on(WireModes::HTTP, 0, 0),
        dialed_on(WireModes::GRPC, 1, 0),
    ];
    bed.facts = vec![processes(false), processes(true)];
    bed.references = true;
    assert_eq!(bed.placed(), ((0, 0), vec![1], false));
    bed.wires = WireModes::HTTP;
    assert_eq!(bed.stage(), "encode-media");
    bed.facts[1] = EncoderFacts {
        references: true,
        ..pushing(1)
    };
    assert_eq!(bed.stage(), "ec-compatible");
}

/// The pass keeps the hard demand its encode side states in its one test
/// of an encoder, against what each encoder states beside its row
/// ([`EncodePool::demands`]): the runtime as a wire, and vision with the
/// request's pixels. An encoder that states nothing meets no demand for a
/// runtime, as every encoder does under an empty column. A column of
/// another length than the encoders is declined for a request that reads
/// it, and is not asked of one that reads none of it. A window and a way
/// of deciding a constraint are no encoder's to meet.
///
/// The classic router holds an encoder to the request's wire in its one
/// test of an encoder
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2576`
/// on main ccc411cf0), and answers a pushing encoder that takes the media
/// off that wire as one that lacks its role (`:2603-2608`).
#[test]
fn an_encoder_is_held_to_the_encode_sides_demand_by_what_it_states() {
    let on = |runtime: u32| DemandFacts {
        runtime,
        ..DemandFacts::default()
    };
    let mut bed = Bed::new();
    bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 4)];
    bed.facts = vec![pushing(1), pushing(1)];
    bed.demands = Demands {
        runtime: 7,
        ..Demands::default()
    };
    // No column: no encoder states a runtime, and a pushing encoder that
    // takes the media stands off the request's.
    assert_eq!(bed.stage(), "ec-compatible");
    // The first states another runtime, and the second states nothing.
    bed.stated = vec![on(8), DemandFacts::default()];
    assert_eq!(bed.stage(), "ec-compatible");
    // A column of another length than the encoders is its host's mistake,
    // one entry short or one too many: the request is declined.
    bed.stated = vec![on(7)];
    assert_eq!(bed.stage(), "encode-column");
    bed.stated = vec![on(7); 3];
    assert_eq!(bed.stage(), "encode-column");
    // The second states the request's: the item goes there, past the idle
    // first.
    bed.stated = vec![on(8), on(7)];
    assert_eq!(bed.placed(), ((0, 0), vec![1], false));
    // A window and a way of deciding a constraint ask nothing of it.
    bed.demands = Demands {
        input_window: 9,
        output_window: 9,
        decoding: 3,
        ..bed.demands
    };
    assert_eq!(bed.placed(), ((0, 0), vec![1], false));
    // A demand for vision is the test of the request's pixels, and reads
    // no entry: a column of another length is not asked of it. The request
    // holds no image and demands vision, and the idle encoder takes none.
    bed.stated = vec![on(8)];
    bed.images = 0;
    bed.demands = Demands {
        vision: true,
        ..Demands::default()
    };
    bed.encoders[0].vision = false;
    assert_eq!(bed.placed(), ((0, 0), vec![1], false));
    bed.encoders[1].vision = false;
    assert_eq!(bed.stage(), "encode-vision");
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

/// An encode decision says, in the scratch it is made with, the list it
/// sheds its request over as saturated: the encoders where the overload
/// veto left none of them open, and else the list its pair's decision
/// says, where that decision was handed the pair's two lists whole. Where
/// the pass handed it a part of either list, the decision says no list,
/// of the list that was cut or of the other: the pair's decision judged
/// the part. Each decision takes back the word of the one before where it
/// begins, so an answer that is no such shed leaves none.
#[test]
fn an_encode_decision_says_the_list_it_sheds_over() {
    let mut shedding = engine();
    shedding.shed_when_all_vetoed(0);
    let saturated = (ShedReason::Saturated, "not-overloaded");
    let mut scratch = Scratch::new();
    // The one encoder is over its thresholds.
    let mut bed = Bed::new();
    bed.encoders[0] = overloaded(bed.encoders[0]);
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), Some(ShedOver::Encoders));
    // No encoder is ready: a missing endpoint, which says no list.
    bed.encoders[0] = unready(bed.encoders[0]);
    let missing = (ShedReason::NoEndpoints, "encoders");
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), missing);
    assert_eq!(scratch.shed_over(), None, "behind a missing endpoint");
    // The one prefill is over its thresholds: the pair's decision is
    // handed both lists whole, and says the prefills.
    let mut bed = Bed::new();
    bed.prefills[0] = overloaded(bed.prefills[0]);
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), Some(ShedOver::Prefills));
    // A second prefill under its thresholds hands to no encoder, so the
    // pair's decision is handed the first alone: its word of the prefills
    // says nothing of the list the caller lent, and goes.
    bed.prefills.push(row(1, Role::Prefill, 1, 0));
    bed.prefill_classes.push(2);
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), None, "a part of the prefills");
    // A pick says no list either.
    bed.prefills[0] = row(0, Role::Prefill, 1, 0);
    let decision = bed.decide_in(&shedding, &mut scratch);
    assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
    assert_eq!(scratch.shed_over(), None, "behind a pick");
    // The one decode is over its thresholds behind whole lists: the
    // pair's decision says the decodes.
    let mut bed = Bed::new();
    bed.decodes[0] = overloaded(bed.decodes[0]);
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), Some(ShedOver::Decodes));
    // Behind a part of the prefills that word goes too: the decision made
    // its shed for the prefills it was handed.
    bed.prefills.push(row(1, Role::Prefill, 1, 0));
    bed.prefill_classes.push(2);
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), None, "the prefills cut, the decodes");
    // The request's decode must take images, and a second decode under
    // its thresholds takes none: a part of the decodes, behind a shed over
    // the decodes and behind one over the prefills.
    let mut bed = Bed::new();
    bed.decode_images = true;
    bed.decodes[0] = overloaded(bed.decodes[0]);
    bed.decodes.push(Candidate {
        vision: false,
        ..row(1, Role::Decode, 1, 0)
    });
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), None, "the decodes cut, the decodes");
    bed.decodes[0] = row(0, Role::Decode, 1, 0);
    bed.prefills[0] = overloaded(bed.prefills[0]);
    assert_eq!(shed_of(&bed.decide_in(&shedding, &mut scratch)), saturated);
    assert_eq!(scratch.shed_over(), None, "the decodes cut, the prefills");
}

/// Under a binding that sheds a pool the overload veto empties, encoders
/// the veto left none of are a saturated fleet: the shed names the veto's
/// stage and its reason is saturation, as a pair's side sheds under that
/// binding. A stage ahead of the veto that leaves none is a missing endpoint
/// as before, and so is one behind it that refuses an encoder the veto
/// kept. An encoder that does not take the request is not counted, and
/// under a veto its host lifted the encoders serve.
#[test]
fn encoders_the_veto_left_none_of_shed_as_saturation_under_a_binding_that_sheds() {
    let mut shedding = engine();
    shedding.shed_when_all_vetoed(0);
    let saturated = (ShedReason::Saturated, "not-overloaded");
    let missing = (ShedReason::NoEndpoints, "encoders");
    // Every encoder that takes the request is over its thresholds.
    let mut bed = Bed::new();
    bed.encoders = vec![
        overloaded(row(0, Role::Encode, 0, 0)),
        overloaded(row(1, Role::Encode, 0, 0)),
    ];
    bed.facts = vec![pushing(1), pushing(1)];
    let shed = bed.decide_with(&shedding).0;
    assert_eq!(shed_of(&shed), saturated);
    // The veto is a stage of the encode profile, and the trace says so.
    assert_eq!(shed.trace().made_under(), encode_profile(&shedding));
    assert!(shed.trace().made_under().is_some());
    // A pair's decode side says the same of its own rows under that
    // binding.
    let mut side = Bed::new();
    side.decodes = vec![overloaded(row(0, Role::Decode, 1, 0))];
    assert_eq!(shed_of(&side.pair(&shedding)), saturated);
    // The host lifted the veto on the encode side: the item is placed.
    let (facts, _) = bed.asked();
    let cx = Constraints::default();
    let lifted = Constraints {
        overload_lifted: true,
        ..Constraints::default()
    };
    let sides = [&cx, &cx, &lifted];
    let (decision, placed) = decide_under(&bed, &shedding, &facts, bed.pools(), sides);
    assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
    assert_eq!(placed.encoders().len(), 1);

    // An unready encoder beside one over its thresholds was refused ahead
    // of the veto: the veto still left none.
    bed.encoders[1] = unready(row(1, Role::Encode, 0, 0));
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), saturated);
    // No encoder reached the veto: a missing endpoint.
    bed.encoders[0] = unready(row(0, Role::Encode, 0, 0));
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), missing);

    // An encoder the veto kept, refused by a hard filter listed behind it:
    // a missing endpoint, as a pair's side names rows so filtered.
    bed.encoders = vec![
        overloaded(row(0, Role::Encode, 0, 0)),
        row(1, Role::Encode, 0, 0),
    ];
    let mut behind = engine();
    behind.shed_when_all_vetoed(0);
    for profile in behind.profiles.iter_mut().map(own) {
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
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), missing);
    // A video alone asks the same of an encoder as an image does.
    (bed.images, bed.video) = (0, 1);
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), missing);
    (bed.images, bed.video) = (1, 0);
    // Nor is one over its thresholds that states no end.
    bed.encoders[1] = overloaded(row(1, Role::Encode, 0, 0));
    bed.facts[1] = pushing(0);
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), missing);
    // Nor is one over its thresholds dialed on no wire the request states.
    bed.encoders = vec![
        unready(dialed_on(WireModes::HTTP, 0, 0)),
        overloaded(dialed_on(WireModes::GRPC, 1, 0)),
    ];
    bed.facts[1] = pushing(1);
    bed.wires = WireModes::HTTP;
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), missing);
    // With no wire stated the veto left none.
    bed.wires = WireModes::NONE;
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), saturated);
    // Nor is one over its thresholds that states another runtime than the
    // request stays on.
    let on = |runtime: u32| DemandFacts {
        runtime,
        ..DemandFacts::default()
    };
    bed.demands = Demands {
        runtime: 7,
        ..Demands::default()
    };
    bed.stated = vec![on(7), on(8)];
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), missing);
    // Both on the request's runtime: the veto left none.
    bed.stated = vec![on(7), on(7)];
    assert_eq!(shed_of(&bed.decide_with(&shedding).0), saturated);
}

/// Under a binding that steers, encoders the overload veto left none of are
/// a missing endpoint under `encoders`: no encoder enters the all-vetoed
/// rule, so none is steered to, and the request is answered as one whose
/// encoders are not ready. The shed is made under the encode profile all
/// the same. A pair's decode side picks among its rows under that binding,
/// and under a veto its host lifted the encoders serve.
#[test]
fn encoders_the_veto_left_none_of_are_a_missing_endpoint_under_a_binding_that_steers() {
    let stock = engine();
    let missing = (ShedReason::NoEndpoints, "encoders");
    // Every encoder that takes the request is over its thresholds.
    let mut bed = Bed::new();
    bed.encoders = vec![
        overloaded(row(0, Role::Encode, 0, 0)),
        overloaded(row(1, Role::Encode, 0, 0)),
    ];
    bed.facts = vec![pushing(1), pushing(1)];
    let shed = bed.decide().0;
    assert_eq!(shed_of(&shed), missing);
    assert_eq!(shed.trace().made_under(), encode_profile(&stock));
    assert!(shed.trace().made_under().is_some());
    // A pair's decode side over its thresholds is picked among under that
    // binding: the rule steers there, and it is asked of no encoder.
    let mut side = Bed::new();
    side.decodes = vec![overloaded(row(0, Role::Decode, 1, 0))];
    let pair = side.pair(&stock);
    assert!(matches!(pair, Decision::Pick(_)), "{pair:?}");
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
    // An unready encoder beside one over its thresholds reads the same.
    bed.encoders[1] = unready(row(1, Role::Encode, 0, 0));
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

/// An encode request names no worker: a decision that states a number on
/// its decode side ([`Constraints::target_worker`]) is declined under
/// `target-worker`, beside places of that side and beside none, with an
/// item and with none. The pair entry counts such a number.
#[test]
fn a_number_on_the_decode_side_declines_an_encode_request() {
    let (bed, engine) = (Bed::new(), engine());
    let (facts, pool) = bed.asked();
    let shed = |keys: &[Identity<'static>], decode_cx: &Constraints<'_>| {
        let encode = EncodePool {
            encoders: &bed.encoders,
            facts: &bed.facts,
            prefill_classes: &bed.prefill_classes,
            affinity: &bed.encode_affinity,
            ..EncodePool::default()
        };
        let items = EncodeItems {
            keys,
            one: None,
            references: false,
            decode_images: false,
            ring: None,
        };
        let cx = Constraints::default();
        let decision = engine.decide_encode_pair(
            &facts,
            &pool,
            bed.pools(),
            encode,
            items,
            &cx,
            decode_cx,
            &cx,
            &mut Scratch::new(),
            &mut EncodePlaced::new(),
        );
        match decision {
            Decision::Shed(shed) => Some((shed.reason, shed.emptied_by)),
            Decision::Pick(_) | Decision::Queue(_) => None,
        }
    };
    let numbered = |places| Constraints {
        places,
        target_worker: Some(0),
        ..Constraints::default()
    };
    let declined = Some((ShedReason::PolicyDeclined, "target-worker"));
    for keys in [&bed.keys[..], &[]] {
        assert_eq!(shed(keys, &numbered(Some(&[0]))), declined);
        assert_eq!(shed(keys, &numbered(None)), declined);
        assert_eq!(shed(keys, &Constraints::default()), None, "a pick");
    }
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
        matches!(vetoed.pair(&engine), Decision::Pick(pick) if pick.trace.lifted == LiftedVeto::Overload),
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

/// No item is placed by where its request stands on a lent ring: that
/// place is the request's routing value's, and the ring is the request's
/// own decision's. A key with no text stays on the encode stage's own ring
/// under every such place. The classic gateway keys an item's encoder by
/// the item's content alone
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs`,
/// lines 2308 to 2358 at f55a0a6bf), so one content keeps its encoder
/// under every routing value.
#[test]
fn an_item_keeps_its_encoder_wherever_its_request_stands_on_a_lent_ring() {
    let document = PolicySpec::epd_presets("consistent_hashing", "power_of_two", "power_of_two");
    let engine = document.expect("the presets").validate();
    let engine = engine.expect("the form validates");
    let mut bed = Bed::new();
    bed.encoders = (0..4).map(|at| row(at, Role::Encode, 0, 0)).collect();
    bed.facts = vec![pushing(1); 4];
    // Six keys far apart, as the hashes of six contents are.
    let spread = |item: u64| Identity::hashed((item + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    bed.keys = (0..6).map(spread).collect();
    let (facts, pool) = bed.asked();
    let encoders = |facts: &RequestFacts<'_>| -> Vec<u32> {
        let (_, placed) = bed.decide_asked(&engine, facts, &pool);
        placed.encoders().iter().map(|t| t.position).collect()
    };
    // No ring lent: each item on the encoder its own key reaches.
    let own = encoders(&facts);
    assert_eq!(own.len(), 6);
    let first = own[0];
    assert!(
        own.iter().any(|&encoder| encoder != first),
        "six keys far apart on one encoder of four: {own:?}"
    );
    // A ring lent that holds no encoder: a key placed by the request's
    // place would go to the encoder at that place modulo the four.
    let ring = InOrder(vec!["grpc://prefill-0:9000".to_owned()]);
    for place in 0..4 {
        let places = [place];
        let lent = RequestFacts {
            ring: Some(&ring),
            ring_positions: &places,
            ..facts
        };
        assert_eq!(encoders(&lent), own, "the request stands at {place}");
    }
}

/// The address the encoder at `at` of a case's list is registered at.
fn encoder_at(at: u32) -> String {
    format!("grpc://encoder-{at}:9000")
}

/// `count` ready encoders, each carrying its address's id as a row of a
/// gateway's list does: a lent ring names a worker by its address.
fn addressed(count: u32) -> Vec<Candidate> {
    (0..count)
        .map(|at| Candidate {
            id: WorkerId::from_address(&encoder_at(at)),
            ..row(at, Role::Encode, 0, 0)
        })
        .collect()
}

/// A caller's ring that reaches the encoders at `places` in that order.
fn reaches(places: &[u32]) -> InOrder {
    InOrder(places.iter().map(|&at| encoder_at(at)).collect())
}

/// The texts of eight items, as a host holds the hex of an item's content
/// hash. The cases' ring places a text by its fingerprint, whose last bits
/// read each byte's last bits alone: these eight stand at eight places
/// modulo eight, so they spread over the encoders of every case's ring.
const TEXTS: [&str; 8] = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"];

/// The `epd` plan whose encode side is the preset `encode`.
fn encoding_by(encode: &str) -> Engine {
    let document = PolicySpec::epd_presets(encode, "power_of_two", "power_of_two");
    let engine = document.expect("the presets").validate();
    engine.expect("the form validates")
}

/// Each placed key's encoder under `engine`, by its row in the encode pool.
fn encoders_of(bed: &Bed, engine: &Engine) -> Vec<u32> {
    let (decision, placed) = bed.decide_with(engine);
    assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
    placed.encoders().iter().map(|t| t.position).collect()
}

/// An item that carries its text is placed on the ring its request's caller
/// lends for the items: on the first encoder offered to it that the ring's
/// walk reaches from where the item's own text stands. The classic gateway
/// asks its encode policy for each item with the hex of the item's content
/// hash as the routing key, on the model's ring where that policy reads one
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs`,
/// lines 2637 to 2649 and 2722 to 2772 at 6aa129a5c), so a content keeps
/// the encoder that ring gives it. An encoder the ring names that is not
/// offered, one that is not ready or one of another prefill's class, is
/// passed. A key with no text states no place: the stage's own ring keeps
/// it where it stood with no ring lent.
#[test]
fn an_item_goes_where_the_ring_lent_for_the_items_reaches_from_its_own_text() {
    let engine = encoding_by("consistent_hashing");
    let mut bed = Bed::new();
    bed.encoders = addressed(6);
    bed.facts = vec![pushing(1); 6];
    let texts = TEXTS;
    bed.keys = texts.map(Identity::plain).to_vec();
    // No ring lent: each key on the encoder the stage's own ring gives it.
    let own = encoders_of(&bed, &engine);
    let order = [4, 1, 5, 0, 3, 2];
    let ring = reaches(&order);
    // The first of `offered` the ring's walk reaches from `text`'s place.
    let reached = |text: &str, offered: &[u32]| {
        let from = (ring.position(text) % 6) as usize;
        let mut walk = (0..6).map(|step| order[(from + step) % 6]);
        walk.find(|at| offered.contains(at))
            .expect("an encoder offered")
    };
    let walked = |offered: &[u32]| -> Vec<u32> {
        let of_text = |text: &&str| reached(text, offered);
        texts.iter().map(of_text).collect()
    };
    bed.ring = Some(Box::new(reaches(&order)));
    let lent = encoders_of(&bed, &engine);
    assert_eq!(lent, walked(&[0, 1, 2, 3, 4, 5]));
    assert_ne!(lent, own, "the two rings agree on every key");
    // The first key's owner is not ready, and the encoder behind it on the
    // ring hands to another prefill's class: neither is offered, and the
    // key's walk goes past both.
    let at = order.iter().position(|&at| at == lent[0]);
    let at = at.expect("an encoder of the ring");
    let (down, other) = (order[at], order[(at + 1) % 6]);
    bed.encoders[down as usize] = unready(bed.encoders[down as usize]);
    bed.facts[other as usize] = pushing(2);
    let offered: Vec<u32> = (0..6).filter(|at| ![down, other].contains(at)).collect();
    let past = encoders_of(&bed, &engine);
    assert_eq!(past, walked(&offered));
    assert_eq!(past[0], order[(at + 2) % 6], "the first key passes both");
    // The same contents with no text: no place is stated, and each stays on
    // the encoder the stage's own ring gave it.
    let mut bare = Bed::new();
    bare.encoders = addressed(6);
    bare.facts = vec![pushing(1); 6];
    bare.keys = texts
        .map(|text| Identity::hashed(Identity::plain(text).hash))
        .to_vec();
    bare.ring = Some(Box::new(reaches(&order)));
    assert_eq!(encoders_of(&bare, &engine), own);
}

/// A ring lent for the items that reaches no encoder offered holds none of
/// them yet, or is of another membership than the list. Under the plain
/// ring a key then goes to the encoder at its text's place modulo the
/// encoders offered, in their list's order, as the gateway's own ring
/// policy places a key with no ring
/// (`model_gateway/src/policies/consistent_hashing.rs`,
/// `find_by_consistent_hash`): one content keeps one encoder there too.
#[test]
fn a_ring_that_reaches_no_offered_encoder_places_an_item_by_its_texts_place_over_them() {
    let engine = encoding_by("consistent_hashing");
    let mut bed = Bed::new();
    bed.encoders = addressed(5);
    bed.facts = vec![pushing(1); 5];
    // The second encoder is not ready: four are offered.
    bed.encoders[1] = unready(bed.encoders[1]);
    let offered = [0u32, 2, 3, 4];
    let texts = TEXTS;
    bed.keys = texts.map(Identity::plain).to_vec();
    let elsewhere = || InOrder(vec!["grpc://prefill-0:9000".to_owned()]);
    let at = |text: &&str| offered[(elsewhere().position(text) % 4) as usize];
    let by_place: Vec<u32> = texts.iter().map(at).collect();
    assert!(
        offered.iter().all(|encoder| by_place.contains(encoder)),
        "the texts leave an offered encoder out: {by_place:?}"
    );
    bed.ring = Some(Box::new(elsewhere()));
    assert_eq!(encoders_of(&bed, &engine), by_place);
    // A ring that holds no worker answers a key's place too.
    bed.ring = Some(Box::new(InOrder(Vec::new())));
    assert_eq!(encoders_of(&bed, &engine), by_place);
}

/// The one key placed in the items' stead goes where the ring lent for the
/// items reaches from its own text, as an item's key does: the classic
/// gateway places the items of a request without image fan-out by the hex
/// of one digest over them, under a pushing handoff
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs`,
/// lines 2622 to 2635 at 6aa129a5c).
#[test]
fn the_one_key_goes_where_the_lent_ring_reaches_from_its_own_text() {
    let engine = encoding_by("consistent_hashing");
    let mut bed = Bed::new();
    bed.encoders = addressed(6);
    bed.facts = vec![pushing(1); 6];
    bed.keys = TEXTS.map(Identity::plain).to_vec();
    let order = [2, 5, 0, 4, 1, 3];
    let ring = reaches(&order);
    let mut homes = Vec::new();
    for digest in ["d0", "d1", "d2", "d3", "d4", "d5"] {
        bed.one = Some(Identity::plain(digest));
        bed.ring = Some(Box::new(reaches(&order)));
        let (decision, placed) = bed.decide_with(&engine);
        assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
        assert!(placed.one(), "{digest}");
        let encoders: Vec<u32> = placed.encoders().iter().map(|t| t.position).collect();
        let home = order[(ring.position(digest) % 6) as usize];
        assert_eq!(encoders, [home], "{digest}");
        homes.push(home);
    }
    assert!(
        homes.iter().any(|&home| home != homes[0]),
        "six digests at one encoder of six: {homes:?}"
    );
}

/// A lender that places a key's text and no head: the trait's own answer
/// for a head.
#[derive(Debug)]
struct PlacesNoHead(InOrder);

impl LentRing for PlacesNoHead {
    fn position(&self, key: &str) -> u64 {
        self.0.position(key)
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        self.0.walk(position, take);
    }
}

/// Under the prefix form of the ring the lender is asked where an item's
/// text stands as a head, and never where it stands as a key: a routing
/// value leads its request's prompt there, and the gateway's own policy of
/// that form looks a header's key up as it looks a head up
/// (`model_gateway/src/policies/prefix_hash.rs`, `select_worker_impl`).
/// The crate's test ring answers another number for a head than for a key,
/// so an item placed by the wrong question lands elsewhere. A lender that
/// places no head lends nothing: the stage's own ring places the item, as
/// with no ring lent.
#[test]
fn under_the_prefix_form_an_item_goes_where_its_text_stands_as_a_head() {
    let engine = encoding_by("prefix_hash");
    let mut bed = Bed::new();
    bed.encoders = addressed(6);
    bed.facts = vec![pushing(1); 6];
    let texts = TEXTS;
    bed.keys = texts.map(Identity::plain).to_vec();
    let own = encoders_of(&bed, &engine);
    let order = [3, 0, 4, 2, 5, 1];
    let ring = reaches(&order);
    let by = |place: u64| order[(place % 6) as usize];
    let as_head = |text: &&str| {
        by(ring
            .head_position(Head::Text(text))
            .expect("a head's place"))
    };
    let as_heads: Vec<u32> = texts.iter().map(as_head).collect();
    let as_keys: Vec<u32> = texts.iter().map(|text| by(ring.position(text))).collect();
    assert_ne!(as_heads, as_keys, "the two questions answer alike");
    bed.ring = Some(Box::new(reaches(&order)));
    assert_eq!(encoders_of(&bed, &engine), as_heads);
    bed.ring = Some(Box::new(PlacesNoHead(reaches(&order))));
    assert_eq!(encoders_of(&bed, &engine), own);
}

/// A lender that counts what it is asked: a key's place, a head's place and
/// a walk.
#[derive(Debug)]
struct Counted(InOrder, Arc<[AtomicU32; 3]>);

impl LentRing for Counted {
    fn position(&self, key: &str) -> u64 {
        self.1[0].fetch_add(1, Ordering::Relaxed);
        self.0.position(key)
    }

    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        self.1[1].fetch_add(1, Ordering::Relaxed);
        self.0.head_position(head)
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        self.1[2].fetch_add(1, Ordering::Relaxed);
        self.0.walk(position, take);
    }
}

/// The ring lent for the items is asked once a key that carries its text,
/// by the encode profile's word: a key's place under a stage that places a
/// routing value by its own text, a head's place under the prefix form, and
/// nothing under a profile that reads no ring. A key with no text asks
/// nothing, and neither does a request without items. The classic gateway
/// looks the model's ring up once an item, and reads that ring only where
/// its encode policy reads one
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs`,
/// lines 2722 to 2772 and 2637 to 2649 at 6aa129a5c).
#[test]
fn the_lent_ring_is_asked_once_a_key_and_only_under_a_profile_that_reads_one() {
    let order = [1, 3, 0, 2];
    // What the lender was asked for one request of `keys` under the encode
    // preset `encode`: keys' places, heads' places, walks.
    let asked = |encode: &str, keys: &[Identity<'static>]| {
        let counts: Arc<[AtomicU32; 3]> = Arc::default();
        let mut bed = Bed::new();
        bed.encoders = addressed(4);
        bed.facts = vec![pushing(1); 4];
        bed.keys = keys.to_vec();
        bed.ring = Some(Box::new(Counted(reaches(&order), Arc::clone(&counts))));
        let (decision, _) = bed.decide_with(&encoding_by(encode));
        assert!(
            matches!(decision, Decision::Pick(_)),
            "{encode}: {decision:?}"
        );
        [0, 1, 2].map(|at| counts[at].load(Ordering::Relaxed))
    };
    let mut keys = TEXTS.map(Identity::plain).to_vec();
    keys.push(Identity::hashed(77));
    assert_eq!(asked("consistent_hashing", &keys), [8, 0, 8]);
    assert_eq!(asked("prefix_hash", &keys), [0, 8, 8]);
    assert_eq!(asked("power_of_two", &keys), [0, 0, 0]);
    assert_eq!(asked("consistent_hashing", &[]), [0, 0, 0]);
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

/// A video asks vision as an image does, of the encoder that takes the
/// request and, beside references, of its decode. A request whose one item
/// is a video is refused under the vision gap where the only encoder takes
/// no images, and its item goes to the encoder that takes them where one
/// does, busy as it is. References beside a video take a decode that takes
/// images, with an item and with none.
///
/// The classic router asks vision for any pixel item, an image or a video
/// (`is_pixel_modality` and `request_needs_vision`,
/// `model_gateway/src/routers/grpc/multimodal/capability.rs:325-344` on
/// main d6517a4fe), and holds each encoder to it
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:1530`).
#[test]
fn a_video_alone_asks_vision_as_an_image_does() {
    let blind = |at: u32, role: Role, in_flight: u32| Candidate {
        vision: false,
        ..row(at, role, u32::from(role == Role::Decode), in_flight)
    };
    let video = || {
        let mut bed = Bed::new();
        bed.images = 0;
        bed.video = 1;
        bed
    };
    // The only encoder takes no image.
    let mut bed = video();
    bed.encoders = vec![blind(0, Role::Encode, 0)];
    assert_eq!(bed.stage(), "encode-vision");
    // Beside one that takes images the item goes there, busy as it is.
    bed.encoders = vec![blind(0, Role::Encode, 0), row(1, Role::Encode, 0, 9)];
    bed.facts = vec![pushing(1), pushing(1)];
    assert_eq!(bed.placed(), ((0, 0), vec![1], false));

    // References beside a video: the idle decode takes no image, the busy
    // one does.
    let mut bed = video();
    bed.references = true;
    bed.facts = vec![EncoderFacts {
        references: true,
        ..pushing(1)
    }];
    bed.decodes = vec![blind(0, Role::Decode, 0), row(1, Role::Decode, 1, 9)];
    assert_eq!(bed.placed(), ((0, 1), vec![0], false));
    // And with no item.
    bed.keys.clear();
    assert_eq!(bed.placed(), ((0, 1), vec![], false));
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
    let held = &side.decode_classes;
    let sets = [&side.open, &side.member, &held.registered, &held.vision];
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
        ring: None,
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
    own(&mut engine.profiles[encode])
        .filters
        .push(Box::new(shows));
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
    own(&mut engine.profiles[prefill])
        .filters
        .push(Box::new(shows));
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
    for profile in engine.profiles.iter_mut().map(own) {
        let counts = CountsItsAsks(Arc::clone(&asked));
        profile.filters.push(Box::new(counts));
    }
    let bed = Bed::new();
    let cx = Constraints::default();
    let holders = [crate::engine::AffinityHolder {
        row: 1,
        overlap_blocks: 8,
        predicted_blocks: 0,
    }];
    let named = RequestFacts {
        named: Some(crate::engine::AffinityRows {
            holders: &holders,
            ..crate::engine::AffinityRows::default()
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

/// The constraints of a host that holds the request to the rows `at` of a
/// pool.
fn holding(at: &[u32]) -> Constraints<'_> {
    Constraints {
        places: Some(at),
        ..Constraints::default()
    }
}

/// The workers of a pick's first target: the prefill's and the decode's.
fn workers_of(decision: &Decision) -> (WorkerId, WorkerId) {
    let Decision::Pick(Pick { targets, .. }) = decision else {
        panic!("a pick, not {decision:?}");
    };
    let target = targets.first().expect("a target");
    (target.id, target.pair.expect("a pair").decode_id)
}

/// The places a side's host holds the request to number the rows of the
/// pool it is handed with. Beside rows this side numbered anew they are
/// read by where each row stood: a place whose row was left out opens none,
/// and the pair is picked among the rows admitted that stand at a place. So
/// a side answers as it does for a host that hands the rows admitted
/// itself, with its places among them. Beside a whole pool the places are
/// read as they are handed.
#[test]
fn a_sides_places_are_read_by_where_each_row_stood() {
    let engine = engine();
    let open = Constraints::default();
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
    let decide =
        |sides: [&Constraints<'_>; 3]| decide_under(&cut, &engine, &facts, cut.pools(), sides).0;
    let pair = |sides: [&Constraints<'_>; 3]| pair_of(&decide(sides));
    assert_eq!(pair([&open; 3]), (1, 1));
    // The fuller row alone stands at a place: the idle one is not the
    // pair's.
    let last = holding(&[2]);
    assert_eq!(pair([&last, &open, &open]), (2, 1));
    assert_eq!(pair([&open, &last, &open]), (1, 2));
    // A place whose row was left out opens none: no other row stands in
    // its stead.
    let ends = holding(&[0, 2]);
    assert_eq!(pair([&ends, &open, &open]), (2, 1));
    assert_eq!(pair([&open, &ends, &open]), (1, 2));
    assert_eq!(pair([&ends, &ends, &open]), (2, 2));
    // Both rows admitted stand at a place: the idle one.
    let admitted = holding(&[1, 2]);
    assert_eq!(pair([&admitted, &admitted, &open]), (1, 1));
    // A mask beside the places is numbered with them: the idle row at a
    // place is masked, and the fuller one is the pair's.
    let masked = Constraints {
        stalled: &[1],
        ..holding(&[0, 1, 2])
    };
    assert_eq!(pair([&masked, &masked, &open]), (2, 2));

    // A host that hands the rows admitted itself states its places among
    // them, and the same workers answer.
    let moved = |row: &Candidate, position: u32| Candidate { position, ..*row };
    let mut handed = Bed::new();
    handed.prefills = vec![moved(&cut.prefills[1], 0), moved(&cut.prefills[2], 1)];
    handed.prefill_classes = vec![1, 1];
    handed.decodes = vec![moved(&cut.decodes[1], 0), moved(&cut.decodes[2], 1)];
    handed.decode_images = true;
    let theirs = holding(&[1]);
    for (ours, theirs) in [
        ([&ends, &open, &open], [&theirs, &open, &open]),
        ([&open, &ends, &open], [&open, &theirs, &open]),
        ([&ends, &ends, &open], [&theirs, &theirs, &open]),
    ] {
        let (answer, _) = decide_under(&handed, &engine, &facts, handed.pools(), theirs);
        assert_eq!(workers_of(&decide(ours)), workers_of(&answer));
    }

    // No cut: beside a whole pool the places are read as they are handed.
    let mut whole = Bed::new();
    whole.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 9)];
    whole.prefill_classes = vec![1, 1];
    whole.decodes = vec![row(0, Role::Decode, 1, 0), row(1, Role::Decode, 1, 9)];
    let second = holding(&[1]);
    for (sides, pair) in [
        ([&second, &open, &open], (1, 0)),
        ([&open, &second, &open], (0, 1)),
    ] {
        let (decision, _) = decide_under(&whole, &engine, &facts, whole.pools(), sides);
        assert_eq!(pair_of(&decision), pair);
    }
}

/// Places numbered anew are the decision's own: the next decision of one
/// scratch reads its own places and none of the last.
#[test]
fn places_numbered_anew_are_forgotten_by_the_next_decision() {
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
    let mut prefill = |cx: &Constraints<'_>| {
        let decision = engine.decide_encode_pair(
            &facts,
            &pool,
            bed.pools(),
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
    assert_eq!(prefill(&holding(&[2])), 2);
    assert_eq!(prefill(&open), 1);
    assert_eq!(prefill(&masking(&[1])), 2);
    assert_eq!(prefill(&holding(&[1, 2])), 1);
}

/// An item is placed on an encoder at one of the places the encoders' host
/// holds the request to. An encoder outside them takes no key and opens no
/// class, so a prefill it alone hands to is no candidate of the pair. A
/// key's run reads the mask numbered with the encoders offered, and no
/// place.
#[test]
fn an_item_is_placed_on_an_encoder_at_one_of_its_requests_places() {
    let engine = engine();
    let open = Constraints::default();
    // Two encoders of the pair's class, the second the fuller, and one of
    // another class.
    let mut bed = Bed::new();
    bed.encoders = (0..3)
        .map(|at| row(at, Role::Encode, 0, 9 * (at % 2)))
        .collect();
    bed.facts = vec![pushing(1), pushing(1), pushing(2)];
    let (facts, _) = bed.asked();
    let encoder = |held: &Constraints<'_>| {
        let sides = [&open, &open, held];
        let (decision, placed) = decide_under(&bed, &engine, &facts, bed.pools(), sides);
        assert!(decision.pick().is_some(), "a pick, not {decision:?}");
        placed.encoders().first().map(|target| target.position)
    };
    assert_eq!(encoder(&open), Some(0));
    assert_eq!(encoder(&holding(&[1])), Some(1));
    assert_eq!(encoder(&holding(&[1, 2])), Some(1));
    assert_eq!(encoder(&holding(&[0, 1, 2])), Some(0));
    // A place past the pool's end names no encoder.
    assert_eq!(encoder(&holding(&[1, 7])), Some(1));
    // The idle encoder of the class stands outside the places, and a mask
    // names it: the mask is read by where each encoder stood.
    let masked = Constraints {
        stalled: &[0],
        ..holding(&[1, 2])
    };
    assert_eq!(encoder(&masked), Some(1));

    // The idle prefill's one encoder stands outside the places: the pair is
    // the busy prefill, and its encoder takes the item.
    let mut classes = Bed::new();
    classes.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 5)];
    classes.prefill_classes = vec![1, 2];
    classes.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
    classes.facts = vec![pushing(1), pushing(2)];
    let placed = |held: &Constraints<'_>| {
        let sides = [&open, &open, held];
        let (decision, placed) = decide_under(&classes, &engine, &facts, classes.pools(), sides);
        let encoders: Vec<u32> = placed.encoders().iter().map(|t| t.position).collect();
        (pair_of(&decision), encoders)
    };
    assert_eq!(placed(&open), ((0, 0), vec![0]));
    assert_eq!(placed(&holding(&[1])), ((1, 0), vec![1]));
    assert_eq!(placed(&holding(&[0, 1])), ((0, 0), vec![0]));
}

/// A request whose places on a list name workers and none that takes it,
/// though the list holds one, is held off every worker that takes it: it is
/// shed under the places' name. No gap of the membership says it, since the
/// model takes the request. A gap is read over every worker, with places as
/// without. Where a worker at the places takes the request and is not open,
/// the shed is the list's own, by the worker's state.
///
/// Places that name no worker of a list are no such case: nobody stands at
/// them for the pass to leave out. Such places of a side shed the request
/// as they shed its pair alone, beside the rows admitted as beside a whole
/// list. Such places of the encoders shed it as a pool with no open encoder
/// does.
#[test]
fn a_request_held_off_every_worker_that_takes_it_is_shed_under_the_places_name() {
    let stock = engine();
    let open = Constraints::default();
    let held_off = (ShedReason::NoEndpoints, "places");
    let first = holding(&[0]);
    // No place, the place at the end of a list of two, and a place past
    // it.
    let nobody = [holding(&[]), holding(&[2]), holding(&[7])];
    // The first place, which names a worker, and a place that names none.
    let first_and_nobody = holding(&[0, 7]);
    let shed = |bed: &Bed, sides: [&Constraints<'_>; 3]| {
        let (facts, _) = bed.asked();
        shed_of(&decide_under(bed, &stock, &facts, bed.pools(), sides).0)
    };
    // The pair entry's own answer under each side's constraints.
    let alone = |bed: &Bed, [cx, decode_cx]: [&Constraints<'_>; 2]| {
        let (facts, pool) = bed.asked();
        let mut scratch = Scratch::new();
        shed_of(&stock.decide_pair(&facts, &pool, bed.pools(), cx, decode_cx, &mut scratch))
    };

    // The first decode takes no image, and the request asks images of its
    // decode.
    let mut bed = Bed::new();
    bed.decodes = vec![row(0, Role::Decode, 1, 0), row(1, Role::Decode, 1, 0)];
    bed.decodes[0].vision = false;
    bed.decode_images = true;
    assert_eq!(shed(&bed, [&open, &first, &open]), held_off);
    assert_eq!(shed(&bed, [&open, &first_and_nobody, &open]), held_off);
    for nobody in &nobody {
        let pair = alone(&bed, [&open, nobody]);
        assert_ne!(pair.1, "places");
        assert_eq!(shed(&bed, [&open, nobody, &open]), pair);
    }
    // No decode takes images: the gap is the membership's.
    bed.decodes[1].vision = false;
    assert_eq!(shed(&bed, [&open, &first, &open]).1, "decode-vision");
    assert_eq!(shed(&bed, [&open; 3]).1, "decode-vision");

    // The first prefill's class has no encoder.
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 0)];
    bed.prefill_classes = vec![2, 1];
    assert_eq!(shed(&bed, [&first, &open, &open]), held_off);
    assert_eq!(shed(&bed, [&first_and_nobody, &open, &open]), held_off);
    for nobody in &nobody {
        let pair = alone(&bed, [nobody, &open]);
        assert_ne!(pair.1, "places");
        assert_eq!(shed(&bed, [nobody, &open, &open]), pair);
    }
    // No prefill's class has one.
    bed.prefill_classes = vec![2, 2];
    assert_eq!(shed(&bed, [&first, &open, &open]).1, "ec-compatible");
    assert_eq!(shed(&bed, [&open; 3]).1, "ec-compatible");

    // The first encoder takes no image, and the request states one.
    let mut bed = Bed::new();
    bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
    bed.encoders[0].vision = false;
    bed.facts = vec![pushing(1), pushing(1)];
    assert_eq!(shed(&bed, [&open, &open, &first]), held_off);
    let by_state = (ShedReason::NoEndpoints, "encoders");
    for nobody in &nobody {
        assert_eq!(shed(&bed, [&open, &open, nobody]), by_state);
    }
    // No encoder takes images.
    bed.encoders[1].vision = false;
    assert_eq!(shed(&bed, [&open, &open, &first]).1, "encode-vision");
    assert_eq!(shed(&bed, [&open; 3]).1, "encode-vision");
    // The encoder at the place states no end: it takes no request,
    // whatever its row says, and the one outside the place takes this one.
    bed.encoders[1].vision = true;
    bed.encoders[0].vision = true;
    bed.facts = vec![pushing(0), pushing(1)];
    assert_eq!(shed(&bed, [&open, &open, &first]), held_off);

    // A side that admits nobody leaves nobody out: its places are the
    // pair's to answer, as without them. The prefill's class has one
    // encoder, which is not ready, beside a ready encoder of a class no
    // prefill names.
    let mut bed = Bed::new();
    bed.encoders = vec![
        unready(row(0, Role::Encode, 0, 0)),
        row(1, Role::Encode, 0, 0),
    ];
    bed.facts = vec![pushing(1), pushing(2)];
    let unheld = shed(&bed, [&open; 3]);
    assert_ne!(unheld.1, "places");
    assert_eq!(shed(&bed, [&first, &open, &open]), unheld);

    // The encoder at the place takes the request and is not ready: the
    // shed is the encode side's, by state. Over its thresholds it reads
    // the same under a binding that steers, and is a saturated fleet of
    // one under a binding that sheds, whatever the encoder outside the
    // place is.
    let mut bed = Bed::new();
    bed.encoders = vec![
        unready(row(0, Role::Encode, 0, 0)),
        row(1, Role::Encode, 0, 0),
    ];
    bed.facts = vec![pushing(1), pushing(1)];
    assert_eq!(shed(&bed, [&open, &open, &first]), by_state);
    bed.encoders[0] = overloaded(row(0, Role::Encode, 0, 0));
    assert_eq!(shed(&bed, [&open, &open, &first]), by_state);
    let saturated = (ShedReason::Saturated, "not-overloaded");
    let mut shedding = engine();
    shedding.shed_when_all_vetoed(0);
    let (facts, _) = bed.asked();
    let held = decide_under(&bed, &shedding, &facts, bed.pools(), [&open, &open, &first]);
    assert_eq!(shed_of(&held.0), saturated);
    // A hard filter behind the veto refuses the encoder outside the place.
    // Over every encoder that is a missing endpoint: the request held to
    // the place reads the encoder there alone.
    let mut behind = engine();
    behind.shed_when_all_vetoed(0);
    for profile in behind.profiles.iter_mut().map(own) {
        profile
            .filters
            .push(Box::new(RefusesOne(bed.encoders[1].id)));
    }
    let under = |sides: [&Constraints<'_>; 3]| {
        shed_of(&decide_under(&bed, &behind, &facts, bed.pools(), sides).0)
    };
    assert_eq!(under([&open; 3]), by_state);
    assert_eq!(under([&open, &open, &first]), saturated);
}

/// A gap the pass reads of the pairs' membership is answered ahead of the
/// encoders' places: the model's workers say it whatever the places are,
/// and no other place closes it. So a request that its encoders' places
/// hold off every encoder that takes it is shed under the gap where one
/// stands, as it is with no place and with a side's places. An encoder
/// outside the places that is not ready changes nothing: the places hold
/// the request off it. What the pair's own decision answers stays behind
/// the pass, under places as under none.
#[test]
fn a_gap_of_the_pairs_membership_stands_ahead_of_the_encoders_places() {
    let stock = engine();
    let open = Constraints::default();
    let first = holding(&[0]);
    let shed = |bed: &Bed, sides: [&Constraints<'_>; 3]| {
        let (facts, _) = bed.asked();
        shed_of(&decide_under(bed, &stock, &facts, bed.pools(), sides).0)
    };
    // The first encoder takes no image, and the request states one: the
    // encoders' place holds it off the one encoder that takes it.
    let held_off = || {
        let mut bed = Bed::new();
        bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
        bed.encoders[0].vision = false;
        bed.facts = vec![pushing(1), pushing(1)];
        bed
    };
    let mut bed = held_off();
    assert_eq!(shed(&bed, [&open, &open, &first]).1, "places");
    // No decode takes images, and the request asks images of its decode.
    bed.decodes[0].vision = false;
    bed.decode_images = true;
    for sides in [[&open; 3], [&open, &first, &open], [&open, &open, &first]] {
        assert_eq!(shed(&bed, sides).1, "decode-vision");
    }
    // The encoders hand to a class that no prefill names.
    let mut bed = held_off();
    bed.prefill_classes = vec![2];
    for sides in [[&open; 3], [&first, &open, &open], [&open, &open, &first]] {
        assert_eq!(shed(&bed, sides).1, "ec-compatible");
    }
    // The one encoder that takes the request is not ready. With no place
    // the request has no open encoder, and under the place it is held off
    // the encoder all the same.
    let mut bed = held_off();
    bed.encoders[1] = unready(bed.encoders[1]);
    assert_eq!(shed(&bed, [&open; 3]).1, "encoders");
    assert_eq!(shed(&bed, [&open, &open, &first]).1, "places");
    // No prefill pairs with a decode: the pair's own decision says so, and
    // it is asked behind the pass.
    let mut bed = held_off();
    bed.decodes = vec![row(0, Role::Decode, 2, 0)];
    assert_eq!(shed(&bed, [&open; 3]).1, "partnered");
    assert_eq!(shed(&bed, [&open, &open, &first]).1, "places");
}

/// The encoders' places are numbered anew by each decision, as a side's
/// are: the next decision of one scratch holds its item to its own places
/// and to none of the last decision's.
#[test]
fn the_encoders_places_are_forgotten_by_the_next_decision() {
    let engine = engine();
    // Two encoders of the pair's class.
    let mut bed = Bed::new();
    bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
    bed.facts = vec![pushing(1), pushing(1)];
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
    let mut encoder = |held: &Constraints<'_>| {
        let decision = engine.decide_encode_pair(
            &facts,
            &pool,
            bed.pools(),
            encode,
            items,
            &open,
            &open,
            held,
            &mut scratch,
            &mut placed,
        );
        assert!(decision.pick().is_some(), "a pick, not {decision:?}");
        placed.encoders().first().map(|target| target.position)
    };
    assert_eq!(encoder(&holding(&[1])), Some(1));
    assert_eq!(encoder(&holding(&[0])), Some(0));
    assert_eq!(encoder(&holding(&[1])), Some(1));
}

/// A debug build refuses places that do not ascend where the encode pass
/// numbers them beside a list it narrowed. Read by one walk beside rows
/// that ascend, the places of the last two rows stated backwards would
/// hold the request to the last row alone and say nothing.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the places are ascending, each once")]
fn a_debug_build_refuses_a_narrowed_sides_places_out_of_order() {
    // The first prefill's class has no encoder: the side is narrowed.
    let mut bed = Bed::new();
    bed.prefills = (0..3).map(|at| row(at, Role::Prefill, 1, 0)).collect();
    bed.prefill_classes = vec![2, 1, 1];
    let (facts, _) = bed.asked();
    let open = Constraints::default();
    let sides = [&holding(&[2, 1]), &open, &open];
    let _ = decide_under(&bed, &engine(), &facts, bed.pools(), sides);
}

/// The encoders' places are refused the same way where the pass holds the
/// open encoders to them.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the places are ascending, each once")]
fn a_debug_build_refuses_the_encoders_places_out_of_order() {
    let mut bed = Bed::new();
    bed.encoders = vec![row(0, Role::Encode, 0, 0), row(1, Role::Encode, 0, 0)];
    bed.facts = vec![pushing(1), pushing(1)];
    let (facts, _) = bed.asked();
    let open = Constraints::default();
    let sides = [&open, &open, &holding(&[1, 0])];
    let _ = decide_under(&bed, &engine(), &facts, bed.pools(), sides);
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

/// Tells `engine`'s session table that both sides of `pick` went out for
/// `facts`, a side at a time as a host commits a pair: each side's lease
/// states that side's own session and its own verdict, the decode side's
/// as the first target's pair keeps it.
fn dispatched(engine: &Engine, facts: &RequestFacts<'_>, pick: &Pick) {
    let target = pick.targets.first().expect("a target");
    let pair = target.pair.expect("a pair");
    let decode = pair.decode_verdict.on(pair.decode_id);
    let sides = [
        (Side::Prefill, Phase::Prefill, pick.session_verdict),
        (Side::Decode, Phase::Decode, decode),
    ];
    for (side, phase, session_verdict) in sides {
        let lease = Lease {
            session: RequestFacts { phase, ..*facts }.session(),
            session_verdict,
            ..Lease::default()
        };
        engine.dispatched_table(&lease, target, side);
    }
}

/// Each side's word of a pick: who placed it, and whether it is kept from
/// its profile. The prefill side's, then the decode side's, which every
/// pair of the pick states.
fn words(decision: &Decision) -> [(PlacedBy, bool); 2] {
    let Decision::Pick(pick) = decision else {
        panic!("a pick, not {decision:?}");
    };
    let mut pairs = pick.targets.iter().filter_map(|target| target.pair);
    let decode = pairs.next().expect("a decode").decode_placing;
    assert!(pairs.all(|pair| pair.decode_placing == decode));
    [
        (pick.placed_by(), pick.untold()),
        (decode.placed_by(), decode.untold()),
    ]
}

/// The request of `bed` under the session key of hash `key`.
fn keyed(bed: &Bed, key: u64) -> RequestFacts<'static> {
    RequestFacts {
        session_key: Identity::hashed(key),
        ..bed.asked().0
    }
}

/// A pick keeps each side's word of who placed it beside a list this side
/// numbered anew, as beside the host's own: the prefills cut to the classes
/// an encoder serves, the decodes cut to the ones that take images, and
/// both. A new key's pair is the stages' on both sides. The key's next
/// request follows both pins, by the host's rows and past the loads: its
/// pair is the rule's on both sides, and under the rule's word `untold`
/// each side is kept from its profile. A list built anew for the host's
/// rows said the stages of the prefill side, and a profile then heard of a
/// request its rule keeps from it.
///
/// A gateway's sticky map follows a pin under its cap among the workers
/// its three-sided selection holds eligible, and asks the role's policy
/// nothing there (`model_gateway/src/policies/registry.rs:833-836`; the
/// eligible workers at
/// `model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2651-2654`).
/// The lines are main's at `9b72a221d`.
#[test]
fn a_rules_pick_states_who_placed_it_beside_a_pool_that_was_cut() {
    let stages = (PlacedBy::Stages, false);
    for word in [spec::RulePlaced::Told, spec::RulePlaced::Untold] {
        let rule = (PlacedBy::Rule, word == spec::RulePlaced::Untold);
        // Which lists are cut: none, the prefills, the decodes, both.
        for (prefills, decodes) in [(false, false), (true, false), (false, true), (true, true)] {
            let case = format!("{word:?}: prefills cut {prefills}, decodes cut {decodes}");
            let engine = ruled(word);
            // Three rows a side, the second the emptiest. A side that is
            // cut leaves its first row out: the first prefill's class has
            // no encoder, and the first decode takes no image.
            let loaded = |role| {
                let rows = [5, 0, 9].into_iter().zip(0..);
                rows.map(|(load, at)| row(at, role, 1, load)).collect()
            };
            let mut bed = Bed::new();
            (bed.prefills, bed.decodes) = (loaded(Role::Prefill), loaded(Role::Decode));
            bed.prefill_classes = if prefills { vec![2, 1, 1] } else { vec![1; 3] };
            (bed.decodes[0].vision, bed.decode_images) = (!decodes, decodes);
            let (facts, pool) = (keyed(&bed, 7), bed.asked().1);
            let (first, _) = bed.decide_asked(&engine, &facts, &pool);
            assert_eq!(pair_of(&first), (1, 1), "{case}");
            assert_eq!(words(&first), [stages; 2], "{case}: a new key");
            dispatched(&engine, &facts, first.pick().expect("a pair"));
            // The loads point away from each side's second row by then.
            (bed.prefills[1].in_flight, bed.decodes[1].in_flight) = (20, 20);
            let (next, _) = bed.decide_asked(&engine, &facts, &pool);
            assert_eq!(pair_of(&next), (1, 1), "{case}: the pins");
            assert_eq!(words(&next), [rule; 2], "{case}");
        }
    }
}

/// An item's answer that turns a request away behind its pair takes back
/// the claims that pair's decision made for the request's key, each side's
/// under its own session. The pair's decision claims a new key's prefill
/// and its decode where it places them, ahead of any dispatch, and the
/// request then goes to neither. So the key's next request finds no pin:
/// the stages place both its sides again, by the loads as they then stand.
/// A claim left standing led that request to the pair the first never
/// reached, as the rule's own. A key that was served keeps its pins behind
/// such an answer: the decision followed them and claimed nothing.
///
/// A gateway's sticky map differs here, and the design records it under
/// its decision 13. The map writes a role's pin where the role is selected
/// (`model_gateway/src/policies/registry.rs:875`, and `:856` at the cap).
/// The three-sided selection picks its pair first
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2655`)
/// and assigns the encoders behind it (`:2700-2707`): an assignment that
/// fails returns with both roles' pins where the pair's selection wrote
/// them. The pair's own selection leaves its prefill's pin so behind a
/// decode its policy declines
/// (`model_gateway/src/routers/common/placement.rs:3147-3148`,
/// `:3210-3212`). The lines are main's at `9b72a221d`.
#[test]
fn an_item_turned_away_behind_its_pair_takes_back_the_pairs_claims() {
    let mut engine = ruled(spec::RulePlaced::Told);
    for profile in engine.profiles.iter_mut().map(own) {
        profile.filters.push(Box::new(RefusesTheKey));
    }
    // Two rows a side, the first the emptier. A request of two items: the
    // first is placed, and the second's stage turns the request away.
    let both = vec![Identity::hashed(11), Identity::hashed(12)];
    let kept = vec![Identity::hashed(11)];
    let mut bed = Bed::new();
    bed.prefills = vec![row(0, Role::Prefill, 1, 0), row(1, Role::Prefill, 1, 5)];
    bed.prefill_classes = vec![1, 1];
    bed.decodes = vec![row(0, Role::Decode, 1, 0), row(1, Role::Decode, 1, 5)];
    bed.keys.clone_from(&both);
    let (new, pool) = (keyed(&bed, 7), bed.asked().1);
    let (turned, _) = bed.decide_asked(&engine, &new, &pool);
    assert_eq!(shed_of(&turned).1, "refuses-the-key");
    // The key's next request carries an item the engine keeps, and the
    // loads point at each side's second row by then.
    bed.keys.clone_from(&kept);
    (bed.prefills[0].in_flight, bed.decodes[0].in_flight) = (9, 9);
    let (next, _) = bed.decide_asked(&engine, &new, &pool);
    assert_eq!(
        pair_of(&next),
        (1, 1),
        "the key followed a pair it never reached"
    );
    assert_eq!(words(&next), [(PlacedBy::Stages, false); 2]);
    // Another key is served on that pair, and the loads then point at each
    // side's first row. Its request that is turned away moves no pin.
    let served = keyed(&bed, 8);
    let (first, _) = bed.decide_asked(&engine, &served, &pool);
    assert_eq!(pair_of(&first), (1, 1));
    dispatched(&engine, &served, first.pick().expect("a pair"));
    (bed.prefills[1].in_flight, bed.decodes[1].in_flight) = (20, 20);
    bed.keys.clone_from(&both);
    let (turned, _) = bed.decide_asked(&engine, &served, &pool);
    assert_eq!(shed_of(&turned).1, "refuses-the-key");
    bed.keys.clone_from(&kept);
    let (next, _) = bed.decide_asked(&engine, &served, &pool);
    assert_eq!(pair_of(&next), (1, 1), "a served key's pins");
    assert_eq!(words(&next), [(PlacedBy::Rule, false); 2]);
}
