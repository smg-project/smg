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
    document::{
        spec::{self, PolicySpec},
        ModeSettings, PolicySettings, RoutingSettings,
    },
    engine::{
        framework::{self, PairTable},
        view::{Health, MediaCounts, Role, WorkerId},
        LiftedVeto, Pick, PlacedBy,
    },
    load::signals::{SignalCell, SignalCells, SignalColumn},
    policies::consistent_hashing::tests::InOrder,
    tracking::lease::{Lease, Side},
};

mod item_placement;
mod lent_rings;
mod masks;
mod pins_and_claims;
mod places;
mod rotations;
mod scratch;
mod sheds_and_declines;
mod vision;
mod wires;

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

/// An encoder at `at` that holds `in_flight` requests, dialed on `wire`.
fn dialed_on(wire: WireModes, at: u32, in_flight: u32) -> Candidate {
    Candidate {
        wire_modes: wire,
        ..row(at, Role::Encode, 0, in_flight)
    }
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
