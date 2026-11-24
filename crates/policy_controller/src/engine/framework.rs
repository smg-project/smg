//! The framework (design 4.4): profiles of filters, weighted scorers and one
//! picker; plans that sequence profiles; the engine that resolves a request
//! to its binding and decides. A scorer or picker allocates nothing per
//! decision: scratch lives in the per-thread [`Scratch`] the host keeps.

use std::{
    any::Any,
    collections::HashMap,
    fmt::Debug,
    sync::{atomic::AtomicU64, Arc},
};

use crate::{
    document::spec::{Assignment, Needs},
    engine::{
        decision::{
            Decision, LiftedVeto, Overlap, Pick, Shed, ShedReason, Survivors, Target, Targets,
            Trace, MAX_TARGETS, TRACE_TOP,
        },
        sticky_sessions::{self, Claim, SessionRule, SessionTable, SessionVerdict},
        view::{
            mean_in_flight, AffinityFacts, AffinityRows, Candidate, Constraints, LoadReportCutoff,
            Phase, PoolFacts, RequestFacts, Session, WorkerId,
        },
        warm_up::WarmUpRule,
    },
    load::signals::SignalCells,
    policies::{prefix_hash::HeadRule, random::Random},
    stages::{
        filters::{HasCapacity, RowTests},
        lifecycle::Lifecycle,
        pickers::{target_of, Best},
        scorers::{FewestSessions, LeastRequests},
    },
    tracking::{
        dispatch_credit::DispatchCredit,
        lease::{EventKind, EventSet, Lease, Membership, Outcome, ReleaseReason, Side, Signal},
    },
    Draw,
};

mod alone;
mod encode;
mod engines;
mod keyed;
mod places;
mod vetoed;

pub use alone::{AloneProfile, InTheWay, NotAlone, Place};
pub use encode::{EncodeItems, EncodePlaced, EncodePool, EncoderFacts};
use engines::{run_engines, run_owned};
pub use keyed::PoolRows;
use vetoed::{AllVetoed, Half};

/// Drops candidates before scoring. A `soft` filter steps aside when it would
/// leave nothing: the engines a retry already tried step aside while another
/// candidate remains, and a pool with nothing else is used whole.
///
/// `Any`: the engine's build knows its own filters by their type, and a
/// decision that draws first asks a profile made of them alone about a row
/// in one test, with no call a filter.
pub trait Filter: Any + Send + Sync + Debug {
    fn name(&self) -> &'static str;

    fn keep(&self, facts: &RequestFacts<'_>, candidate: &Candidate, cx: &Constraints<'_>) -> bool;

    fn soft(&self) -> bool {
        false
    }

    /// Whether a decision that draws first ([`Picker::draw_first`]) may ask
    /// this filter about the rows it drew alone, with no pass over the
    /// rest: [`Filter::keep`] of one row is the filter's whole word on it.
    /// A row-wise filter reads each row alone and is promised no order of
    /// the rows it is asked about: they come ascending where the pool is
    /// read whole, and in the draw's order behind a draw. The default is
    /// no, and a profile that holds such a filter runs its whole pass: a
    /// plugin's filter is shown the whole view.
    fn row_wise(&self) -> bool {
        false
    }

    /// What the host must feed for this filter to mean anything. A filter
    /// that reads the affinity column ([`RequestFacts::affinity`]) declares
    /// [`Needs::affinity`]; one that does not may be handed an empty
    /// column, which reads as nothing held. So with the request's block
    /// hashes: a stage that reads them declares [`Needs::token_hashes`] or
    /// [`Needs::text_hashes`], and one that declares neither may be handed
    /// none.
    ///
    /// Asked only while the stage's generation is built, perhaps more than
    /// once there. The answer stands for that generation: no decision and
    /// no membership write asks.
    fn needs(&self) -> Needs {
        Needs::default()
    }

    /// The per-decision scope this stage reads, if it holds one.
    fn scope(&self) -> Option<&dyn DecisionScope> {
        None
    }

    /// Whether this stage may refuse a request by what its gateways
    /// report of leases: a count of requests in flight it is shown
    /// ([`RequestFacts::in_flight`]), or what it keeps of the leases'
    /// events ([`Lifecycle`]). Either may be late on a lull where another
    /// deciding thread's leases write it. A stage that places by it
    /// places elsewhere; one that refuses by it would turn away a request
    /// its host admits. So a host that applies its gateways' reports late
    /// applies them first for a decision under such a stage
    /// ([`Engine::reads_for`]). On a served controller that costs the
    /// decision one more take of its own file's turn. With several
    /// deciders it also costs a read of each other decider's word, and
    /// that decider's turn where it keeps a watch.
    ///
    /// A stage that says nothing is taken as one that may, so nothing a
    /// host or a plugin writes turns away what its host admits by a
    /// late report. A host's own stage that only places, or that
    /// refuses by nothing a report writes, says `false` and leaves that
    /// price. Each built-in stage of this crate says so. A plugin's
    /// stage cannot say it yet: its wrapper keeps the default. The
    /// answer is read once, where the engine is built ([`Engine::new`]):
    /// it is the stage's for the engine's whole life.
    fn refuses_by_reports(&self) -> bool {
        true
    }

    /// Drops from `kept` (positions into `candidates`, in order) the rows this
    /// filter refuses, in place: one call per decision, with [`Filter::keep`]
    /// inlined into the loop. A filter that answers for the whole pool at
    /// once overrides it, and must drop exactly the rows [`Filter::keep`]
    /// refuses: the pair plan's partner and routable passes still ask `keep`
    /// row by row.
    fn retain(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        kept: &mut Vec<u32>,
    ) {
        kept.retain(|&i| self.keep(facts, &candidates[i as usize], cx));
    }
}

/// The band within which scores tie: rows whose combined score is within it
/// of the best remaining score are one tie, drawn from uniformly by the
/// decision's seed. A scorer returns the band of the values it wrote, on
/// their `0.0..=1.0` scale, so a scorer whose natural band is not the
/// normalized one (a least-load scorer ties within a microsecond of expected
/// wait, which is a narrow band on a spread pool and a wide one on a
/// near-equal pool) publishes its own; a profile's band is the weighted sum
/// of its scorers' bands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TieBand(pub f32);

impl TieBand {
    /// The band of a scorer whose written values are exact in their own
    /// right: equal is a tie, a rounding difference is not.
    pub const NORMALIZED: Self = Self(1e-6);
}

/// How the values a scorer writes for the rows of one decision lie
/// ([`Scorer::spread`]). Further spreads add their cases: a stage names
/// the one it writes, and a `match` outside this crate keeps an arm for
/// the rest.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ScoreSpread {
    /// Anywhere in `0.0..=1.0`.
    Any,
    /// On levels in `0.0..=1.0` at least this far apart: two rows read one
    /// value, or two that differ by this much or more.
    Levels(f32),
    /// One value for every row, 1.0: the stage orders nothing and widens no
    /// tie, so the band it returns is none, `TieBand(0.0)`. Among its
    /// leading ranks a decision by rank does not ask such a stage: its
    /// share stands in every row's sum unasked. Past them it is asked like
    /// any stage and a band it returned would be read: a debug build checks
    /// it at every ask.
    Alike,
}

/// What a rank's answer left in front, said beside its scores
/// ([`Scorer::score_front`]). Further words add their cases: a stage
/// names the one it says, and a `match` outside this crate keeps an arm
/// for the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScoreFront {
    /// Nothing is said of the scores.
    Unsaid,
    /// Every row was written the ceiling, 1.0: the stage parts nothing.
    Alike,
    /// This many rows were written the ceiling, and every other row a level
    /// under it.
    At(u32),
    /// The stage does not answer for these rows alone, as where it scores
    /// a row against what the others score: its answer stands only where
    /// they are every row the decision compares, so it must be asked
    /// whole. Nothing is said of the scores, and the stage need not have
    /// written them.
    Whole,
}

impl ScoreFront {
    /// The word a stage's scores read as under its spread, in one pass over
    /// them. Under levels: [`ScoreFront::Alike`] where every row stands at the
    /// ceiling, [`ScoreFront::At`] where some do and every other row stands a
    /// whole level or more under it, and nothing of any other scores. A
    /// stage that orders nothing reads alike, and one that promises no
    /// levels says nothing.
    #[must_use]
    pub fn of(spread: ScoreSpread, stage: &[f32]) -> Self {
        let step = match spread {
            ScoreSpread::Levels(step) => step,
            ScoreSpread::Alike => return Self::Alike,
            ScoreSpread::Any => return Self::Unsaid,
        };
        // Two counts and no way out early: the pass stays free of branches
        // on the rows.
        let mut ceilings = 0u32;
        let mut under = 0u32;
        for &value in stage {
            ceilings += u32::from(value >= 1.0);
            under += u32::from(value <= 1.0 - step && value < 1.0);
        }
        let rows = stage.len();
        if ceilings as usize == rows {
            Self::Alike
        } else if (ceilings + under) as usize == rows {
            Self::At(ceilings)
        } else {
            Self::Unsaid
        }
    }
}

/// Scores the set the profile's picker compares: every survivor of the
/// filters, unless the picker narrows them ([`Picker::narrow`]). `out[i]` is
/// the score of `kept[i]`, normalized to `0.0..=1.0` with higher better; a
/// scorer normalizes against the rows it is handed, writes in place and
/// allocates nothing. The narrowing bounds those rows and nothing else: a
/// scorer that by its own rule reads the pool's facts, or a level it keeps
/// for the fleet, still does. A plugin scorer's guest is one such: its
/// interface builds it with every candidate of the decision and asks one
/// score a candidate, and the host reads the scores of the rows handed, so
/// what the guest takes from a candidate outside them is the guest's own.
/// The returned [`TieBand`] is the band of the written values.
pub trait Scorer: Lifecycle + Debug {
    fn name(&self) -> &'static str;

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand;

    /// Whether the scorer may be handed the rows a picker drew first
    /// ([`Picker::draw_first`]), where no pass over the pool ran before it.
    /// A row-wise scorer reads each row alone and is promised no order of
    /// the rows it is handed: they stand ascending where the pool is read
    /// whole, and in the draw's order behind a draw. The default is no,
    /// and a profile that holds such a scorer runs its whole pass: a
    /// plugin's scorer is shown the whole view.
    fn row_wise(&self) -> bool {
        false
    }

    /// Whether the scorer may be asked of the rows a rank put in front
    /// ([`Self::spread`]) and no other. It orders them as it would among
    /// every row, reading of the rows behind them what it needs, or says
    /// that it must be asked whole ([`Self::score_front`],
    /// [`ScoreFront::Whole`]). The promise is the order and not the values: a
    /// scorer that spreads its values over the rows asked writes a front's
    /// rows further apart, and two of them that a sum over every row
    /// rounds together are ordered there, where the whole pass draws among
    /// them. A draw's rows are scored among themselves
    /// ([`Self::row_wise`]) and a front's are not: a scorer whose order of
    /// two rows turns on which rows it is asked of says no. The default is
    /// no, and behind a rank such a stage is asked of every row.
    fn front_wise(&self) -> bool {
        false
    }

    /// Whether this scorer places a keyed request by itself
    /// ([`Self::own`]); the default is no. The engine reads it once per
    /// profile when it is built, and asks such a scorer first where it is
    /// the profile's one scorer.
    fn owns(&self) -> bool {
        false
    }

    /// The rows of a request this scorer places by itself, with no pool
    /// read whole, over a pool whose host lists its engines
    /// ([`PoolFacts::engines`]): it finds the request's rows by its own
    /// rule, asks `keep`, every filter of the profile, about each row it
    /// reads, and leaves in `kept` the rows it hands out, the pick first
    /// and its alternates after it, each one kept by `keep` and none
    /// twice. The rows it hands out are the ones the profile's run would,
    /// in its order: only the rows read differ. On `false` nothing is
    /// decided, what `kept` holds is dropped and the profile's run
    /// decides: the scorer places nothing by itself, the request or the
    /// pool gives it nothing to go by, or its rule ran out. It reads no
    /// value of the seed's stream.
    fn own(
        &self,
        _facts: &RequestFacts<'_>,
        _pool: &PoolFacts<'_>,
        _candidates: &[Candidate],
        _keep: &mut dyn FnMut(&Candidate) -> bool,
        _kept: &mut Vec<u32>,
    ) -> bool {
        false
    }

    /// [`Self::own`] for a host that reads a row when it is asked about
    /// one, and writes none ahead of the decision. The scorer finds the
    /// request's workers by its own rule and learns from `place_of` where
    /// each stands in the pool, which holds `rows` rows: `None` for a
    /// worker that is no row of it, and for every worker of a pool where
    /// two rows carry one id ([`PoolRows::place_of`]), which so names no
    /// row. It asks `keep`, every filter of the profile, about each place
    /// once, and of as many places as its rule goes through: the hook
    /// bounds no walk. Returns how many rows it hands out: the places
    /// `keep` kept, in the order it was asked, the pick first and its
    /// alternates behind it. They are the rows the profile's run over the
    /// whole pool hands out, in its order. `0` where it places nothing by
    /// itself or its rule ran out: the profile's run decides. It reads no
    /// value of the seed's stream. The default places nothing.
    fn own_by_place(
        &self,
        _facts: &RequestFacts<'_>,
        _rows: usize,
        _place_of: &dyn Fn(WorkerId) -> Option<u32>,
        _keep: &mut dyn FnMut(u32) -> bool,
    ) -> usize {
        0
    }

    /// How this scorer's values lie ([`ScoreSpread`]), asked once when the
    /// engine is built. A scorer that writes levels is a rank where its weight
    /// outweighs every stage behind it: a plain decision then asks those stages
    /// of the rows at its ceiling alone, where they fill the decision's targets
    /// and are at most half of the rows it was asked of, where each of those
    /// stages orders nothing or answers for a front alone
    /// ([`Self::front_wise`]), and where at most one of them that is no rank
    /// orders the rows. A stage that cannot promise its levels says nothing,
    /// the default.
    fn spread(&self) -> ScoreSpread {
        ScoreSpread::Any
    }

    /// As [`Self::score`], with what the answer left in front ([`ScoreFront`]),
    /// over three lists: the pool, the kept rows, which are the rows
    /// [`Self::score`] is handed, and the rows asked, a part of the kept
    /// rows in their order. `out[i]` is the score of `asked[i]`: a stage
    /// may read a kept row it is not asked of, and writes the rows asked
    /// alone. A plain decision asks it of a stage that is a rank
    /// ([`Self::spread`]), of every kept row: the two lists are one there,
    /// and it reads nothing of the scores where every row stands alike. It
    /// asks it too of a stage behind a rank, of the rows the rank put in
    /// front, which says there that it must be asked whole
    /// ([`ScoreFront::Whole`]) where it does not answer for them alone. A
    /// scorer that says its front has one body behind both entries, so a
    /// row scores alike by either. It says the word from what that body
    /// knows, or reads it off the scores ([`ScoreFront::of`]). A debug build
    /// checks the word against the scores. The default scores the rows
    /// asked and says nothing.
    fn score_front(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        _kept: &[u32],
        asked: &[u32],
        out: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        (
            self.score(facts, candidates, asked, out),
            ScoreFront::Unsaid,
        )
    }

    /// Whether this scorer says what a pick's first target holds of its
    /// request ([`Self::said`]); the default is no. No entry of the engine
    /// reads it: an entry states a pick's overlap itself, from the rows the
    /// decision was given ([`Pick::overlap`]).
    fn states_overlap(&self) -> bool {
        false
    }

    /// What `row` holds of the request ([`Overlap`]), asked behind the
    /// decision that picked the row, on the thread that made it, and once
    /// a decision: a stage may spend what its pass judged by on the first
    /// call, so a second row asked of the same decision may read no margin
    /// held. The default says nothing.
    ///
    /// No entry of the engine asks it: an entry states a pick's overlap
    /// itself, from the rows the decision was given ([`Pick::overlap`]),
    /// and not by this word.
    fn said(&self, _facts: &RequestFacts<'_>, _candidates: &[Candidate], _row: u32) -> Overlap {
        Overlap::default()
    }

    /// What the host must feed for this scorer to mean anything. A scorer
    /// that reads the affinity column ([`RequestFacts::affinity`]) declares
    /// [`Needs::affinity`]; one that does not may be handed an empty
    /// column, which reads as nothing held. So with the request's block
    /// hashes: a stage that reads them declares [`Needs::token_hashes`] or
    /// [`Needs::text_hashes`], and one that declares neither may be handed
    /// none.
    ///
    /// Asked only while the stage's generation is built, perhaps more than
    /// once there. The answer stands for that generation: no decision and
    /// no membership write asks.
    fn needs(&self) -> Needs {
        Needs::default()
    }

    /// The per-decision scope this stage reads, if it holds one.
    fn scope(&self) -> Option<&dyn DecisionScope> {
        None
    }

    /// As [`Filter::refuses_by_reports`], for a scorer.
    fn refuses_by_reports(&self) -> bool {
        true
    }

    /// Whether the stage places a request at the place its caller states
    /// for the routing value's own text on a ring the caller lends
    /// ([`RequestFacts::ring`]): the hash ring that is keyed by no prefix.
    /// What makes a request for its caller states that place under such a
    /// stage ([`Engine::states_key_place`]): another form of the ring is
    /// asked for another text, and says which
    /// ([`Self::places_by_heads`]). A stage that reads a ring
    /// ([`Scorer::needs`]) and answers `false` here is taken for such a
    /// form.
    fn places_by_key_text(&self) -> bool {
        false
    }

    /// How the stage cuts a request into the heads it places it by on a
    /// ring its caller lends ([`RequestFacts::ring`]): the prefix form of
    /// the hash ring, which reads where a head stands
    /// ([`LentRing::head_position`](crate::policies::consistent_hashing::LentRing::head_position))
    /// and never where the routing value's own text does. What makes a
    /// request for its caller states a place for each head under such a
    /// stage ([`Engine::states_head_places`]). `None` from every other
    /// stage. The answer stands for the stage's generation, as
    /// [`Self::needs`] does: it is asked at most once where an engine is
    /// built, and no decision asks.
    fn places_by_heads(&self) -> Option<HeadRule<'_>> {
        None
    }

    /// The warm-up ruling this stage applies, where it applies one: what a
    /// host that names a pool's warming rows enters
    /// ([`WarmUpRule::list_warming_rows`]). No other stage holds one.
    fn warm_up_rule(&self) -> Option<WarmUpRule> {
        None
    }
}

/// Two counts over the survivors a picker's narrowing reordered, not a list
/// of its own: the first `compared` are
/// the set it compares, and the next `alternates` are survivors it passed
/// over that still follow its pick as alternates, in the order it left them.
/// They follow a pick and never stand in for one: when the picker pushes no
/// target, none is appended.
///
/// The record gains counts: code outside this crate makes it by
/// [`Shortlist::all`] or [`Shortlist::new`], and matches it with `..`.
///
/// ```
/// use smg_policy_controller::engine::Shortlist;
/// let shortlist = Shortlist::new(2, 1);
/// assert_eq!((shortlist.compared, shortlist.alternates), (2, 1));
/// assert_eq!(Shortlist::all(5), Shortlist::new(5, 0));
/// ```
///
/// ```compile_fail
/// use smg_policy_controller::engine::Shortlist;
/// let _ = Shortlist { compared: 2, alternates: 1 };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Shortlist {
    /// How many survivors, from the front, the picker compares.
    pub compared: usize,
    /// How many survivors behind those follow the pick as alternates.
    pub alternates: usize,
}

impl Shortlist {
    /// Every survivor is compared.
    pub const fn all(survivors: usize) -> Self {
        Self {
            compared: survivors,
            alternates: 0,
        }
    }

    /// The first `compared` survivors are compared, and the next
    /// `alternates` follow the pick as alternates.
    #[must_use]
    pub const fn new(compared: usize, alternates: usize) -> Self {
        Self {
            compared,
            alternates,
        }
    }
}

/// The most rows a decision that draws its one or two picks first
/// ([`Picker::draw_first`]) may see refused before it stops drawing: the
/// bound those pickers count against. Such a draw asks the filters about no
/// more rows than the targets it hands out and these, and a decision about
/// no more than those and the whole pass's. A picker that draws a sample
/// counts none: its own draws, twice the sample, bound what it asks.
pub const DRAW_BOUND: u32 = 8;

/// The smallest pool at which a decision draws first for speed, set by
/// run. A draw pays more than the whole pass for each row it asks about, so
/// a small pool is read whole where the two place a request by one rule
/// (the last paragraph names the picker whose draw is asked under it). With
/// every row ready the draw reads faster than the pass from eight rows on,
/// and with one row in seventeen not ready from sixteen, for every built-in
/// that draws: sixteen is the first size the `draw` bench reads at which it
/// wins on both. Rows refused do not move
/// this size, and under twice this size they can cost more than the draw
/// gains: with every second row refused, a pool under 32 rows can pay more
/// than its whole pass: under a picker that draws two rows, and under
/// `round_robin` too in a whole decision with its view's build. The bench's
/// cells `power_of_two/half/16` and `power_of_two/half/32` hold that shape
/// and where it ends.
///
/// A picker that takes turns ([`Picker::takes_turns`]) is asked to draw
/// under this size too, over a pool that lends its own word
/// ([`PoolFacts::word`]): its draw takes the list's own turn, which the
/// whole pass does not, so there the size is no matter of speed.
pub const DRAW_FROM: usize = 16;

/// The pool size from which a profile whose leading stages are ranks is
/// decided by rank ([`Scorer::spread`]); a smaller pool is asked whole. Rows
/// stand in front where four fill the targets and are at most half, so in a
/// small pool a front spares the stages behind few rows, four at eight rows,
/// while a decision with no row in front pays for the ranks' reading: at
/// eight rows 67 instructions, and 1.04 to 1.09 of the whole pass by time.
/// The size is the draw's ([`DRAW_FROM`]): a decision reads one floor for
/// both.
pub const RANK_FROM: usize = DRAW_FROM;

// A decision compares the pool's size with one floor for the draw and the
// ranks.
const _: () = assert!(RANK_FROM == DRAW_FROM);

/// What a picker draws ([`Picker::unit`]). Further units add their cases:
/// a picker names its own, and a `match` outside this crate keeps an arm
/// for the rest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum DrawUnit {
    /// A row: a worker, or one rank of a data-parallel worker.
    #[default]
    Row,
    /// An engine: a worker, or the ranks of a data-parallel worker
    /// together. Over a pool whose host lists its engines
    /// ([`PoolFacts::engines`]) the picker's draw is over them, and the
    /// decision reads the rows of the engine it takes and of no other.
    /// Over any other pool the picker draws rows.
    Engine,
}

/// Turns the survivors into ordered targets, in two steps the engine runs
/// around the scorers: `narrow` names the set the picker compares, the
/// scorers are handed that set alone, and `pick` orders it. `scores[i]`
/// belongs to `kept[i]` and `band` is the profile's tie band; the picker
/// pushes its pick first and its alternates after it, and allocates nothing.
pub trait Picker: Lifecycle + Debug {
    fn name(&self) -> &'static str;

    /// Names the set the picker compares, before the scorers run: it moves
    /// that set to the front of `kept` and says how many survivors behind it
    /// follow the pick as alternates. It permutes `kept` and writes no other
    /// value. A survivor the picker passed over is handed to no scorer, so
    /// it moves no input of a built-in's estimate. The affinity stages
    /// read the rows a host named fastest where the compared set stands
    /// ascending: a row asked behind the one before it costs a search.
    /// The default compares every survivor.
    fn narrow(
        &self,
        _facts: &RequestFacts<'_>,
        _candidates: &[Candidate],
        kept: &mut [u32],
    ) -> Shortlist {
        Shortlist::all(kept.len())
    }

    /// Orders the rows it is handed. The engine hands the set `narrow` left
    /// and no other survivor. A caller that scored every survivor may hand
    /// them all, as a plugin picker does to its fallback: the picker's whole
    /// rule then runs here, and a picker that compares a part takes that
    /// part itself, by the draws its narrowing would have made. The scores
    /// on that path are the pool's: every survivor was scored, so the part
    /// is compared by scores a worker outside it has moved. A profile that
    /// wants the estimate made over the part names the narrowing picker as
    /// its picker.
    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    );

    /// Whether the picker draws first ([`Picker::draw_first`]); the default
    /// is no. The engine reads it once per profile when it is built, and
    /// asks such a picker to draw over a pool of [`DRAW_FROM`] rows or more,
    /// and over a smaller one where its draw is a turn
    /// ([`Picker::takes_turns`]).
    fn draws_first(&self) -> bool {
        false
    }

    /// The unit the picker draws ([`DrawUnit`]); a row by default. The engine
    /// reads it once per profile when it is built. A picker that says the
    /// engine draws first ([`Picker::draws_first`]) and compares two rows:
    /// it is handed the two rows that stand for the engines drawn, through
    /// [`Picker::order_drawn`], and the engine of the first target it
    /// pushes is the one taken.
    fn unit(&self) -> DrawUnit {
        DrawUnit::Row
    }

    /// Finds the rows the picker hands out without the survivors' list: it
    /// draws rows of `candidates` by its own rule and asks `keep`, every
    /// filter of the profile, about each. On `true`, `kept` holds the rows
    /// it hands out, each one kept by `keep` and none twice: the scorers are
    /// handed them and [`Picker::order_drawn`] orders them. On `false`
    /// nothing is decided, what `kept` holds is dropped and the whole pass
    /// runs: the picker does not draw first, its rule leaves this request to
    /// the pass, as a rotation leaves a retry, or its bound ran out before it
    /// held its pick ([`DRAW_BOUND`] refused rows, or a sample's draws). A
    /// draw that keeps nobody says
    /// nothing of the pool: the pass alone sheds. `draw` is the seed's stream: the
    /// picker takes its values from it and from nowhere else, and whatever
    /// follows the draw reads the stream past them.
    fn draw_first(
        &self,
        _facts: &RequestFacts<'_>,
        _pool: &PoolFacts<'_>,
        _candidates: &[Candidate],
        _keep: &mut dyn FnMut(&Candidate) -> bool,
        _kept: &mut Vec<u32>,
        _draw: &mut Draw,
    ) -> bool {
        false
    }

    /// Orders the rows [`Picker::draw_first`] left in `kept`; `scores[i]`
    /// belongs to `kept[i]`. The default is [`Picker::pick`] over them. A
    /// picker whose `pick` would draw or turn again hands them out as it
    /// drew them.
    fn order_drawn(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        self.pick(facts, candidates, kept, scores, band, out);
    }

    /// Whether the rows a host named in the affinity column's place
    /// ([`AffinityRows`]) join the rows this picker draws first: the
    /// engine keeps, behind the rows drawn, each named row every filter
    /// keeps, and hands the stages the drawn rows' mean in flight, so a
    /// named row does not move the mean it is judged against. The engine
    /// asks such a picker to draw for a request that names rows in the
    /// column's place, and for no other. The default is no: what the picker
    /// drew is all it compares.
    fn joined_by_named(&self) -> bool {
        false
    }

    /// Whether the picker takes its targets from the best of the rows it is
    /// handed and reads no row behind them. A profile's ranks may then hand
    /// it the rows they put in front alone, with the sums every stage gave
    /// them: each row behind lies under every row in front by more than
    /// the tie band. The default is no: a picker may read
    /// every score, so its profile asks every row every stage. The engine
    /// reads it once per profile when it is built.
    fn front_wise(&self) -> bool {
        false
    }

    /// Whether the picker compares every row it is handed and takes its
    /// targets from the best scores among them, in their order: it narrows
    /// to no set of its own and draws between equal scores alone. A
    /// profile whose one scorer places a request by itself
    /// ([`Scorer::own`]) may then hand out that scorer's rows, the ones
    /// this picker would take. The default is no: a picker that narrows,
    /// samples or draws does not follow a scorer's order.
    fn takes_best(&self) -> bool {
        false
    }

    /// What the host must feed for this picker to mean anything. A picker
    /// that reads the affinity column ([`RequestFacts::affinity`]) declares
    /// [`Needs::affinity`]; one that does not may be handed an empty
    /// column, which reads as nothing held. So with the request's block
    /// hashes: a stage that reads them declares [`Needs::token_hashes`] or
    /// [`Needs::text_hashes`], and one that declares neither may be handed
    /// none.
    ///
    /// Asked only while the stage's generation is built, perhaps more than
    /// once there. The answer stands for that generation: no decision and
    /// no membership write asks.
    fn needs(&self) -> Needs {
        Needs::default()
    }

    /// The per-decision scope this stage reads, if it holds one.
    fn scope(&self) -> Option<&dyn DecisionScope> {
        None
    }

    /// As [`Filter::refuses_by_reports`], for a picker.
    fn refuses_by_reports(&self) -> bool {
        true
    }

    /// Whether the picker leaves the placement to its binding's sessions
    /// assignment (the `manual` picker): the engine reads it once per
    /// profile when it is built, and places such a profile's requests
    /// through the assignment's stage without asking the picker.
    fn by_assignment(&self) -> bool {
        false
    }

    /// Whether the picker places a request over a pool's workers by itself
    /// ([`Picker::pick_alone`]): its own rule is its whole pick, and it
    /// reads no row, no score and no count. A caller that keeps a worker's
    /// health on the worker and holds every other rule of a placement
    /// itself asks such a picker for one pick and nothing else of a
    /// decision ([`Engine::place_for`]). The default is no. The engine reads
    /// it where such a caller asks for a place's profile, and at a pair's
    /// decode decided again under a scorer ([`Engine::decide_decode`]): at
    /// no other decision.
    fn picks_alone(&self) -> bool {
        false
    }

    /// Whether each pick takes the next turn of an order the picker keeps,
    /// as a rotation's does: a second pick for one request spends a turn
    /// that serves no request. Such a picker's draw ([`Picker::draw_first`])
    /// must read no value of the seed's stream: the engine hands it the
    /// request's own seed and asserts in a debug build that none was read.
    /// The default is no. The engine reads it at a pair's decode decided
    /// again, handed the decode of the pair's first decision
    /// ([`Engine::decide_decode`]), and once at its build, for a profile
    /// that draws for every request. The row such a picker's draw lands on
    /// is its rule's answer, so the draw is asked at every pool size over a
    /// pool that lends its own word ([`PoolFacts::word`]), and a session a
    /// sessions rule leaves to the profile ([`Assignment::Delegate`]) takes
    /// the pool's turn.
    fn takes_turns(&self) -> bool {
        false
    }

    /// One pick over `pool` by the picker's own rule alone
    /// ([`Picker::picks_alone`]). `keep` asks the caller's test of the
    /// worker at a position of the pool: one read of that worker. `Some`
    /// is the position picked, which `keep` kept. `None` where the rule
    /// kept no worker. A rule that did not ask every worker, as a rotation
    /// whose one turn lands on a refused worker, leaves the pool to its
    /// caller, which reads it whole and asks [`Picker::pick_among`] of the
    /// workers that pass. A rule that asked every worker settles it
    /// ([`Picker::asks_every_worker`]). The default picks nobody.
    fn pick_alone(
        &self,
        pool: &PoolWord<'_>,
        keep: &mut dyn FnMut(usize) -> bool,
    ) -> Option<usize> {
        let _ = (pool, keep);
        None
    }

    /// Whether [`Picker::pick_alone`] asks `keep` of every worker of the
    /// pool before it picks nobody, as a scan in the pool's order does and
    /// a rotation's one turn does not. Its `None` then says that no
    /// worker of the pool passes, and the caller does not read the pool a
    /// second time to learn it. The default is no.
    fn asks_every_worker(&self) -> bool {
        false
    }

    /// The pick among the `passed` workers a whole pass left, as a place in
    /// their list, which stands in the pool's order. `set` names the set
    /// of them: a picker that keeps a rotation a set turns that set's own.
    /// Asked with one worker or more. The default takes the first.
    fn pick_among(&self, set: u64, passed: usize) -> usize {
        let _ = (set, passed);
        0
    }
}

/// A pool as a picker that places over its workers by itself is handed it
/// ([`Picker::pick_alone`]): its size and its own word.
#[derive(Clone, Copy, Debug)]
pub struct PoolWord<'a> {
    /// How many workers the pool holds, in an order that stands for as long
    /// as the pool does.
    pub rows: usize,
    /// The pool's own word, kept by the caller beside the pool for as long
    /// as the pool stands, for the picker that places over it: a rotation
    /// counts its turns in it, so the pool's rotation is found with the
    /// pool and no table is looked up. It starts at a position the caller
    /// drew.
    pub word: &'a AtomicU64,
}

/// Per-decision state a stage set keeps across the stages of one profile
/// (a plugin's `decision` resource over the view): the engine opens it once
/// before the profile's first stage of a decision reads it and closes it
/// after the last, on every path out, so a stage set used by several stages
/// of one profile, or asked row by row, reads one state. The unwinding path
/// is one of them: when a scope panics at its open, the engine closes the
/// scopes whose open returned, and neither that scope nor any after it.
/// So a `close` always follows an `open` that returned, and an `open`
/// that stops half-way undoes what it did itself. Its twin is the
/// implementer's: a `close` that stops half-way leaves its scope closed,
/// so it forgets what it holds open before a step that may not return. A
/// `close` that panics keeps no other scope open: the engine runs every
/// close it owes, each under a catch, and the first panic of the decision
/// is the one the host sees. A built-in stage holds none; the profile's flag
/// ([`Profile::scoped`]) is read once in the engine, so a profile without
/// one pays nothing per decision.
pub trait DecisionScope: Send + Sync + Debug {
    fn open(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], pool: &PoolFacts<'_>);

    fn close(&self);
}

/// A scorer with its weight in the profile's combination.
#[derive(Debug)]
pub struct WeightedScorer {
    pub scorer: Box<dyn Scorer>,
    pub weight: f32,
}

impl WeightedScorer {
    /// The stage's share of the weights of a profile whose one scorer it
    /// is, as the whole pass folds it into a score ([`combine`]): the
    /// weight times its own inverse. A weight's share of itself need not
    /// be one, and under a weight too small to invert it is not finite.
    /// What places a request in front of the whole pass writes its scores
    /// by this share, and only where it is finite.
    #[inline]
    fn sole_share(&self) -> f32 {
        self.weight * (1.0 / self.weight)
    }
}

/// One profile: an ordered set of filters, scorers and exactly one picker.
#[derive(Debug)]
pub struct Profile {
    pub name: String,
    pub filters: Vec<Box<dyn Filter>>,
    pub scorers: Vec<WeightedScorer>,
    pub picker: Box<dyn Picker>,
}

impl Profile {
    /// What this profile's stages read from the host beyond the facts every
    /// view carries: every stage of every kind, so a stage that reads the
    /// affinity column as a filter or as the picker counts as one that
    /// scores does, and a host that fills the column where
    /// [`Needs::affinity`] is declared leaves it empty only where nothing
    /// reads it. The engine's summary ([`Engine::needs`]) adds the view's
    /// own list of headers.
    pub fn needs(&self) -> Needs {
        let filters = self.filters.iter().map(|f| f.needs());
        let scorers = self.scorers.iter().map(|w| w.scorer.needs());
        let needs = filters.chain(scorers).fold(self.picker.needs(), Needs::or);
        // A sample is the need of a profile that may draw. Under a stage
        // that is not row-wise the pool is read whole, and lists capped at
        // the sample would hide holders from that pass.
        Needs {
            sample: if draws_first(self) { needs.sample } else { 0 },
            ..needs
        }
    }

    /// The rulings of this profile's warm-up stages ([`Scorer::warm_up_rule`]),
    /// in the stages' order.
    fn warm_rules(&self) -> impl Iterator<Item = WarmUpRule> + '_ {
        let stages = self.scorers.iter();
        stages.filter_map(|stage| stage.scorer.warm_up_rule())
    }

    /// The cap of each list of rows a host names in the affinity column's
    /// place for a decision of this profile ([`Engine::affinity_rows_cap`]):
    /// its sample, [`u32::MAX`] where it takes none, and `0` where it is named
    /// no row.
    fn named_cap(&self) -> u32 {
        let needs = self.needs();
        let mut rules = self.warm_rules();
        let first = rules.next();
        let one = first.is_none_or(|first| rules.all(|rule| rule == first));
        if !(needs.affinity && one) {
            return 0;
        }
        match needs.sample {
            // A profile that shows its rows to a plugin takes no sample.
            0 if self.scoped() => 0,
            0 => u32::MAX,
            sample => sample,
        }
    }

    /// Whether the filters may be asked of a row ahead of its naming
    /// ([`Engine::keeps_for`]), where `folded` says the engine holds every
    /// one of them as one test. Each answers row by row, and where they are
    /// asked by a call no stage holds a scope the call would run ahead of.
    /// None declares what the door writes behind the test, the naming's
    /// column or the key's counts: ahead of them no row holds either, and
    /// such a filter would refuse the holders it keeps once they are
    /// written. Asked once, at the engine's build.
    fn asked_by_row(&self, folded: bool) -> bool {
        let behind = |needs: Needs| needs.affinity || needs.session_in_flight;
        let later = self.filters.iter().any(|filter| behind(filter.needs()));
        let called = || !self.scoped() && self.filters.iter().all(|filter| filter.row_wise());
        !later && (folded || called())
    }

    /// Whether a scorer of this profile reads sent prefixes and no cache
    /// events ([`Engine::reads_sent_alone_for`]): asked of its scorers at
    /// the engine's build.
    ///
    /// The cache events meant are a request's holders, which a scorer
    /// reads from the index where it declares token hashes beside the
    /// events. One scorer that reads them there keeps the index's holders
    /// first, whatever another reads. A scorer that reads the events for a
    /// row's warm-up reads no holder and turns nothing here.
    fn reads_sent_alone(&self) -> bool {
        let needs = || self.scorers.iter().map(|stage| stage.scorer.needs());
        needs().any(|needs| needs.sent_prefixes)
            && !needs().any(|needs| needs.events && needs.token_hashes)
    }

    /// What a host names this profile's rows by ([`NamedBy`]), asked of
    /// its stages once at the engine's build. `folded` holds its filters
    /// as one test, where the engine holds every one of them itself.
    fn named_by(&self, folded: Option<RowTests>) -> NamedBy {
        let cap = self.named_cap();
        let mut rules = self.warm_rules();
        let first = rules.next();
        let rule = first.filter(|&first| rules.all(|rule| rule == first));
        let held = match folded {
            _ if !self.asked_by_row(folded.is_some()) => Held::Unasked,
            Some(tests) => Held::Folded(tests),
            None => Held::Called,
        };
        let needs = self.needs();
        NamedBy {
            cap,
            rule,
            sent: needs.affinity && needs.sent_prefixes,
            sent_alone: self.reads_sent_alone(),
            held,
        }
    }

    /// Whether any stage holds a [`DecisionScope`].
    pub fn scoped(&self) -> bool {
        self.scopes().next().is_some()
    }

    fn scopes(&self) -> impl Iterator<Item = &dyn DecisionScope> {
        self.filters
            .iter()
            .filter_map(|f| f.scope())
            .chain(self.scorers.iter().filter_map(|w| w.scorer.scope()))
            .chain(self.picker.scope())
    }

    /// Whether any stage may refuse a request by a report of leases
    /// ([`Filter::refuses_by_reports`]).
    fn refuses_by_reports(&self) -> bool {
        self.filters.iter().any(|f| f.refuses_by_reports())
            || self.scorers.iter().any(|w| w.scorer.refuses_by_reports())
            || self.picker.refuses_by_reports()
    }

    /// Opens every stage's scope over the view; the guard closes them. Out
    /// of line, so a profile without a scope keeps the engine's decision path
    /// as it was.
    #[cold]
    #[inline(never)]
    fn open<'a>(
        &'a self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        pool: &PoolFacts<'_>,
    ) -> Opened<'a> {
        // While the scopes open, a guard of this function's own counts each
        // scope whose open returned: if an open unwinds, those are closed,
        // and neither the scope that panicked nor any after it.
        let mut opening = Opening {
            profile: self,
            returned: 0,
        };
        for scope in self.scopes() {
            scope.open(facts, candidates, pool);
            opening.returned += 1;
        }
        // Every open returned: the decision's guard closes them all.
        opening.returned = 0;
        Opened(self)
    }
}

/// The open scopes of one profile for one decision, closed on drop.
struct Opened<'a>(&'a Profile);

impl Drop for Opened<'_> {
    fn drop(&mut self) {
        close_scopes(self.0);
    }
}

/// Closes every scope of the profile, each under a catch, so a close that
/// panics keeps no scope after it open. The first panic of the decision is
/// the one the host sees. Where one is already unwinding (a stage's, the
/// open of a pair's other side, or a close of that side's), a close's panic
/// stops here: let out, it would end the process. Where none is, the first
/// close's panic goes on once every scope is closed. A pair's decode side
/// closes first: where a close of each side panics, the decode side's goes
/// on and the prefill side's stops inside it. A payload that stops here is
/// released.
#[cold]
#[inline(never)]
fn close_scopes(profile: &Profile) {
    let mut first = None;
    for scope in profile.scopes() {
        let closed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| scope.close()));
        let Err(payload) = closed else { continue };
        if first.is_none() && !std::thread::panicking() {
            first = Some(payload);
        } else {
            release(payload);
        }
    }
    if let Some(payload) = first {
        std::panic::resume_unwind(payload);
    }
}

/// The scopes of one profile whose open returned while a decision opens
/// them: the first `returned` in the profile's order. Its drop closes those,
/// which is what the unwinding of a later scope's open runs; once every open
/// returned the count is cleared and `Opened` holds the scopes.
struct Opening<'a> {
    profile: &'a Profile,
    returned: usize,
}

impl Drop for Opening<'_> {
    fn drop(&mut self) {
        for scope in self.profile.scopes().take(self.returned) {
            // A close here runs inside the unwinding of a later scope's
            // open: a close that panics stops there, the rest are still
            // closed, and the open's panic is the one that reaches the host.
            contained(|| scope.close());
        }
    }
}

/// Runs `f` on a path that is already unwinding, where a second panic let
/// out would end the process: a panic of `f` stops here, and its payload is
/// released.
pub(crate) fn contained(f: impl FnOnce()) {
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        release(payload);
    }
}

/// The payloads [`release`] frees in a row: the caught one, then the
/// payload each drop before it raised. Two free a payload whose drop
/// panics once; the bound is what ends a drop that always raises another.
const RELEASE_DEPTH: usize = 4;

/// Frees a caught payload the engine does not resume. A payload's own
/// destructor can panic, and a panic let out here would end the process or
/// take the first panic's place: the drop runs under a catch of its own.
/// The payload of a drop that panicked is freed the same way, down to
/// [`RELEASE_DEPTH`] payloads in a row. Only the one past that depth is
/// forgotten, since its drop could panic in turn.
#[cold]
#[inline(never)]
pub(crate) fn release(payload: Box<dyn Any + Send>) {
    let mut next = Some(payload);
    for _ in 0..RELEASE_DEPTH {
        let Some(payload) = next.take() else { return };
        next = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(payload))).err();
    }
    std::mem::forget(next);
}

/// Per-decision scratch the framework owns; kept by the host per thread so a
/// decision allocates nothing after the first.
#[derive(Debug, Default)]
pub struct Scratch {
    single: Buffers,
    decode: Buffers,
    /// The prefill classes with a partner the decode profile keeps, sorted;
    /// once the prefill pool sheds, the decode classes its routable prefills
    /// pair with.
    classes: Vec<u32>,
    /// The decode classes of a pair table, a bit a class: the ones with a
    /// decode the decode profile keeps, then the ones the routable prefills
    /// pair with.
    marks: ClassBits,
    /// The prefill classes with an open first partner, sorted
    /// ([`FirstPartners`]). Where the pair is made among them they stand in
    /// `classes`, and this holds every class with a partner.
    firsts: Vec<u32>,
    /// The encode side of an encode-pair decision, boxed: the entry moves
    /// one pointer out of the scratch while the pair entry holds the rest.
    encode: Option<Box<encode::EncodeScratch>>,
    /// What the last decision made through the keeping entry read of its
    /// session key, kept for the request's shadow ([`SessionsFound`]).
    sessions_found: SessionsFound,
    /// Whether the keeping entry made the last decision with this scratch.
    /// Only a debug build keeps it, for the shadow's stop on a mix of the
    /// two entries ([`Engine::shadow`]).
    #[cfg(debug_assertions)]
    kept_last: bool,
}

/// What a live decision under a sessions rule learned of its key, step
/// by step: the key's pins at its read, and the pins the table held
/// where its claim was not taken. Each is what the table held, never
/// an answer worked out of the live decision's own pick. The decision's
/// commit moves the pins, so its shadow asks for neither again: it takes
/// each at the step the live decision took it, against its own pins and
/// its own pick ([`Engine::shadow`]). The keeping entry writes it
/// ([`Engine::decide_keeping`]), and forgets the one before it first: the
/// scratch holds a reading of its last keeping decision alone.
#[derive(Clone, Copy, Debug, Default)]
struct SessionsFound {
    /// The request the reading is of: its key, its clock and its seed. A
    /// key of `0` is no reading: a present key is never `0`.
    key: u64,
    now_micros: u64,
    seed: u64,
    /// The key's pins as the table held them at the decision's read.
    pins: [WorkerId; 2],
    /// The decision's claim was not taken, and `standing` holds the
    /// table's answer. The read clears it, so no answer of an earlier
    /// reading is taken for this one's.
    stood: bool,
    /// The pins the table held where the decision's claim was not taken.
    standing: [WorkerId; 2],
}

impl SessionsFound {
    /// Whether the reading is of this request.
    fn is_of(&self, key: u64, facts: &RequestFacts<'_>) -> bool {
        self.key == key && self.now_micros == facts.now_micros && self.seed == facts.seed
    }

    /// Hold no reading.
    fn forget(&mut self) {
        self.key = 0;
    }
}

/// The buffers of one profile run: the survivors, the combined scores of the
/// set the picker compares, one scorer's stage and the survivors a soft
/// filter restores when it would leave nothing. A decision by rank lends
/// two of them more ([`combine_ranked`]): the stage's buffer holds the sums
/// of the rows in front behind a stage's scores of them, and the soft
/// filter's backup holds their list, which keeps its length from one
/// decision to the next. A pair decision keeps the
/// prefill's apart from the decode side's, so the prefill's survivors stay
/// for the next prefill target.
#[derive(Debug, Default)]
struct Buffers {
    kept: Vec<u32>,
    scores: Vec<f32>,
    stage: Vec<f32>,
    before_soft: Vec<u32>,
    /// The rows as they read with no doubt, for a pass that lifts the
    /// liveness veto over rows that carry one: the all-vetoed rule's
    /// ([`vetoed::emptied`]) and the pair default's ([`vetoed::paired`]).
    /// The first such pass sizes it, and no other decision writes it.
    undoubted: Vec<Candidate>,
    /// The veto lifted for the rows a pass kept: the mark of its pick's
    /// trace ([`vetoed::steered`]). The all-vetoed rule writes it where it
    /// keeps rows ([`vetoed::emptied`]), and a pair's default writes its
    /// tier on both sides' buffers ([`vetoed::paired`]). No other pass
    /// clears it, so a decision that reads it clears it first: a keyed
    /// request's ([`Engine::keep`]), and a pair's for the prefill's buffer
    /// ([`Engine::decide_pair`]). There the default alone writes it again,
    /// and the prefill's shadow reads it to know the default made the pair
    /// ([`Engine::shadow_side`]).
    lifted: LiftedVeto,
    /// The pool size every buffer holds room for ([`Buffers::reserve`]).
    room: usize,
}

impl Buffers {
    /// Room in every buffer for the whole pass over a pool of `rows`. A
    /// decision that draws first writes a few rows and asks for the room
    /// all the same: the pass it widens to may be many decisions away, and
    /// it allocates nothing the thread's first decision did not. A whole
    /// pass by the profile's one test of a row asks for it too
    /// ([`run_whole`]). A decision by rank asks for no more than the whole
    /// pass: what it keeps of the rows in front stands in these buffers
    /// ([`combine_few`]).
    fn reserve(&mut self, rows: usize) {
        if self.room < rows {
            self.grow(rows);
        }
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, rows: usize) {
        fn room<T>(buffer: &mut Vec<T>, rows: usize) {
            buffer.reserve(rows.saturating_sub(buffer.len()));
        }
        room(&mut self.kept, rows);
        room(&mut self.scores, rows);
        room(&mut self.stage, rows);
        room(&mut self.before_soft, rows);
        self.room = rows;
    }
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A set of class numbers, a pair table's decode classes or a pool's encode
/// classes: one bit a number.
#[derive(Debug, Default)]
struct ClassBits {
    words: Vec<u64>,
}

impl ClassBits {
    /// Empties the set and sizes it for the class numbers below `width`.
    fn reset(&mut self, width: usize) {
        self.words.clear();
        self.words.resize(width.div_ceil(64), 0);
    }

    /// Adds `class` and says whether the set holds it. A number the set was
    /// not sized for is left out: what it means is each caller's to say.
    fn insert(&mut self, class: u32) -> bool {
        let Some(word) = self.words.get_mut(class as usize / 64) else {
            return false;
        };
        *word |= 1 << (class % 64);
        true
    }

    fn has(&self, class: u32) -> bool {
        Self::has_in(&self.words, class)
    }

    /// The set's words, for a pass that reads the set once a row: bound
    /// ahead of the pass, no row loads their place or their count again.
    fn words(&self) -> &[u64] {
        &self.words
    }

    /// [`Self::has`] over the words a pass bound by [`Self::words`].
    fn has_in(words: &[u64], class: u32) -> bool {
        words
            .get(class as usize / 64)
            .is_some_and(|word| word >> (class % 64) & 1 == 1)
    }

    /// The classes of the set, ascending.
    fn classes(&self) -> impl Iterator<Item = u32> + '_ {
        self.words.iter().zip(0u32..).flat_map(|(&word, at)| {
            std::iter::successors((word != 0).then_some(word), |&left| {
                let left = left & (left - 1);
                (left != 0).then_some(left)
            })
            .map(move |left| at * 64 + left.trailing_zeros())
        })
    }
}

/// Combined scores of the rows in `kept`, written into `scores`; the
/// profile's tie band comes back. A profile's one stage, where its share
/// is the whole sum, writes into `scores` itself, and its values and band
/// are brought to the sum's own terms where they stand. Inlined into its
/// two callers, as it was into its one before a draw's rows were scored
/// too.
#[inline(always)]
fn combine(
    profile: &Profile,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    kept: &[u32],
    scores: &mut Vec<f32>,
    stage: &mut Vec<f32>,
) -> TieBand {
    scores.clear();
    scores.resize(kept.len(), 0.0);
    if profile.scorers.is_empty() {
        return TieBand::NORMALIZED;
    }
    if let [weighted] = profile.scorers.as_slice() {
        // One stage whose share is the whole sum writes the scores where
        // they stand: each value and the band are the sum's own terms, a
        // zero plus the stage's clamped value.
        if weighted.weight > 0.0 && weighted.sole_share() == 1.0 {
            let stage_band = weighted.scorer.score(facts, candidates, kept, scores);
            debug_assert!(
                bandless(weighted, stage_band),
                "{} orders nothing and returned a band",
                weighted.scorer.name()
            );
            for sum in scores.iter_mut() {
                *sum = 0.0 + sum.clamp(0.0, 1.0);
            }
            return TieBand(0.0 + stage_band.0.clamp(0.0, 1.0));
        }
    }
    // The sum of one weight above zero is that weight, so one stage's share
    // below is its [`WeightedScorer::sole_share`], the value tested above.
    let total: f32 = profile.scorers.iter().map(|w| w.weight).sum();
    let norm = if total > 0.0 { 1.0 / total } else { 1.0 };
    let mut band = 0.0f32;
    for weighted in &profile.scorers {
        stage.clear();
        stage.resize(kept.len(), 0.0);
        let stage_band = weighted.scorer.score(facts, candidates, kept, stage);
        // Among its leading ranks a decision by rank does not ask a stage
        // that orders nothing, so the band of one is none here too
        // ([`ScoreSpread::Alike`]).
        debug_assert!(
            bandless(weighted, stage_band),
            "{} orders nothing and returned a band",
            weighted.scorer.name()
        );
        let share = weighted.weight * norm;
        band += share * stage_band.0.clamp(0.0, 1.0);
        for (sum, s) in scores.iter_mut().zip(stage.iter()) {
            *sum += share * s.clamp(0.0, 1.0);
        }
    }
    TieBand(band)
}

/// The scorers and the picker over the survivors in `buf.kept`: the one place
/// a profile's estimate is made. The picker names the set it compares, the
/// scorers are handed that set alone, the picker orders it, and the
/// survivors it passed over follow as alternates while targets are left.
/// Alternates follow a pick and never stand in for one: a picker that pushes
/// no target has declined, and no survivor it passed over is appended. On
/// return the compared set leads `buf.kept` and `buf.scores` holds its
/// scores.
fn score_and_pick(
    profile: &Profile,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    buf: &mut Buffers,
    targets: &mut Targets,
) {
    let narrowed = profile.picker.narrow(facts, candidates, &mut buf.kept);
    let compared = narrowed.compared.min(buf.kept.len());
    let alternates = narrowed.alternates.min(buf.kept.len() - compared);
    let band = combine(
        profile,
        facts,
        candidates,
        &buf.kept[..compared],
        &mut buf.scores,
        &mut buf.stage,
    );
    let before = targets.len();
    profile.picker.pick(
        facts,
        candidates,
        &buf.kept[..compared],
        &buf.scores,
        band,
        targets,
    );
    if targets.len() == before {
        return;
    }
    // A full list keeps what it has: the walk is bounded by the list's size.
    for &i in buf.kept[compared..compared + alternates]
        .iter()
        .take(MAX_TARGETS)
    {
        targets.push(target_of(&candidates[i as usize]));
    }
}

/// Whether a stage's band is one its spread allows: a stage that says it
/// orders nothing returns none ([`ScoreSpread::Alike`]). Read in debug builds
/// at every ask of a stage: a leading one that orders nothing is not asked
/// by rank, and one behind a stage that orders the rows is.
fn bandless(weighted: &WeightedScorer, band: TieBand) -> bool {
    weighted.scorer.spread() != ScoreSpread::Alike || band.0 == 0.0
}

/// Whether a rank's scores are what it said of them ([`ScoreFront`]): at the
/// ceiling in the count it named, and every other row a step of its spread
/// under it. Read in debug builds, where a stage that says a wrong word
/// fails its author's case; a build without the check trusts the word.
fn as_said(spread: ScoreSpread, said: ScoreFront, stage: &[f32]) -> bool {
    let ceilings = stage.iter().filter(|&&value| value >= 1.0).count();
    match said {
        ScoreFront::Unsaid | ScoreFront::Whole => true,
        ScoreFront::Alike => ceilings == stage.len(),
        ScoreFront::At(n) => {
            let step = match spread {
                ScoreSpread::Levels(step) => step,
                ScoreSpread::Any | ScoreSpread::Alike => 0.0,
            };
            ceilings == n as usize
                && stage
                    .iter()
                    .all(|&value| value >= 1.0 || value <= 1.0 - step)
        }
    }
}

/// The combined scores of a profile whose leading stages are ranks
/// ([`Order`]), for the rows in `kept`, written into `scores`. The
/// profile's tie band comes back, and how many rows stand in front: zero
/// where every row was asked every stage.
///
/// A rank that leaves every row at its ceiling adds its share to what each
/// row holds, and a leading stage that orders nothing is not asked. A rank
/// parts the rows where some stand at its ceiling and some under it. The rows
/// at the ceiling stand in front where they fill a decision's targets, are at
/// most half of the rows asked, and the build let the rank put rows in
/// front ([`Order`]): the stages behind are asked of them alone
/// ([`combine_few`]), and a row left behind keeps the sum the ranks gave
/// it. That sum lies under every row in front by more than the later
/// stages add, so the targets come from the rows in front. A stage asked
/// of them alone orders them as it would among every row, or says that it
/// must be asked whole ([`ScoreFront::Whole`]), so their order is the weighted
/// sum's over every row wherever that sum tells them apart. It does not
/// everywhere: a stage that spreads its values over the rows it is handed
/// writes a front's rows further apart, and two of them whose values among
/// every row meet in one `f32` sum stand apart here. The whole pass draws
/// among such rows, and the front orders them by the stage's own rule.
/// Where a stage must be asked whole, where the tie band is wider than the
/// ranks' room, and anywhere else, every row is asked on from the rank
/// that parted them, and the scores are the weighted sum's to the bit: a
/// row's sum is taken in the stages' order either way.
#[expect(
    clippy::too_many_arguments,
    reason = "combine's inputs, the build's word and the list of the rows in front"
)]
fn combine_ranked(
    profile: &Profile,
    order: Order,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    kept: &[u32],
    scores: &mut Vec<f32>,
    stage: &mut Vec<f32>,
    tied: &mut Vec<u32>,
) -> (TieBand, usize) {
    let rows = kept.len();
    scores.clear();
    scores.resize(rows, 0.0);
    // Room for the list of the rows a rank may put in front, their
    // candidates and their places: at most the rows. The list lives in the
    // buffer a soft filter fills, and a profile without one never sized it.
    // A front's sums take no room of their own: they stand in the stage's
    // buffer ([`combine_few`]).
    if tied.capacity() < rows {
        tied.reserve(rows - tied.len());
    }
    if rows == 0 {
        return (TieBand::NORMALIZED, 0);
    }
    let stages = &profile.scorers[..];
    let total: f32 = stages.iter().map(|w| w.weight).sum();
    let norm = if total > 0.0 { 1.0 / total } else { 1.0 };
    let ranks = usize::from(order.ranks).min(stages.len());
    let mut band = 0.0f32;
    // What every row holds while all are tied.
    let mut held = 0.0f32;
    // The leading ranks, while every row is tied. `Alike` stands for no
    // rank having parted the rows.
    let mut front = ScoreFront::Alike;
    let mut i = 0;
    while i < ranks {
        let weighted = &stages[i];
        let share = weighted.weight * norm;
        i += 1;
        if order.alike(i - 1) {
            // Not asked: every row holds its share.
            held += share;
            continue;
        }
        stage.clear();
        stage.resize(rows, 0.0);
        // A rank is asked of every row: one list twice.
        let (mut stage_band, mut said) = weighted
            .scorer
            .score_front(facts, candidates, kept, kept, stage);
        if said == ScoreFront::Whole {
            // It was handed every row: its plain answer is its word.
            stage_band = weighted.scorer.score(facts, candidates, kept, stage);
            said = ScoreFront::Unsaid;
        }
        debug_assert!(
            as_said(weighted.scorer.spread(), said, stage),
            "{} said {said:?} of scores that read otherwise",
            weighted.scorer.name()
        );
        band += share * stage_band.0.clamp(0.0, 1.0);
        if said == ScoreFront::Alike {
            held += share;
            continue;
        }
        // The rows part here: each holds its own sum from now on.
        for (sum, value) in scores.iter_mut().zip(stage.iter()) {
            *sum = held + share * value.clamp(0.0, 1.0);
        }
        front = said;
        break;
    }
    match front {
        ScoreFront::Alike => scores.fill(held),
        ScoreFront::At(n)
            if order.fronts(i - 1)
                && n as usize >= MAX_TARGETS
                && n as usize * 2 <= rows
                && i < stages.len() =>
        {
            let n = n as usize;
            // The list keeps its length from one decision to the next and
            // is filled only as it grows: every entry read of it was
            // written for this decision first. It is sliced to the front, a
            // half for the candidates and a half for their places: the walk
            // below stops at that count.
            if tied.len() < 2 * n {
                tied.resize(2 * n, 0);
            }
            let (asked, places) = tied[..2 * n].split_at_mut(n);
            let mut next = 0;
            for (at, &value) in stage.iter().enumerate() {
                if value >= 1.0 && next < n {
                    asked[next] = kept[at];
                    places[next] = at as u32;
                    next += 1;
                }
            }
            // A scorer that counted wrong puts nobody in front.
            if next == n {
                let decided = combine_few(
                    profile, order, facts, candidates, kept, scores, stage, tied, n, i, norm, band,
                );
                if let Some(decided) = decided {
                    return decided;
                }
            }
        }
        ScoreFront::At(_) | ScoreFront::Unsaid | ScoreFront::Whole => {}
    }
    // Every row is asked on, as a profile without ranks asks: no row stood
    // in front, or the rows in front could not decide alone.
    for (at, weighted) in stages.iter().enumerate().skip(i) {
        let share = weighted.weight * norm;
        if order.alike(at) {
            scores.iter_mut().for_each(|sum| *sum += share);
            continue;
        }
        stage.clear();
        stage.resize(rows, 0.0);
        let stage_band = weighted.scorer.score(facts, candidates, kept, stage);
        debug_assert!(
            bandless(weighted, stage_band),
            "{} orders nothing and returned a band",
            weighted.scorer.name()
        );
        band += share * stage_band.0.clamp(0.0, 1.0);
        for (sum, value) in scores.iter_mut().zip(stage.iter()) {
            *sum += share * value.clamp(0.0, 1.0);
        }
    }
    (TieBand(band), 0)
}

/// The stages from `from` on, asked of the `few` rows in front alone: `tied`
/// holds their candidates, then their places among the rows. A later rank
/// puts fewer in front where they still fill the targets ([`fewer`]) and no
/// stage has parted them since they stood in front. The profile's tie band
/// comes back, and how many rows stand in front at the
/// end: their sums are written among the rows' then, and a row a later rank
/// left behind keeps what the ranks asked of every row gave it. Until then
/// each row in front holds its sum in the stage's buffer, behind the scores
/// a stage writes of those rows: room the rank's own ask took. A stage
/// asked here is shown `kept` beside the rows in front, the rows the ranks
/// were asked of: it may read them, and scores the rows in front alone.
///
/// No stage of this build is asked here as a rank: its ranks say nothing of
/// a front alone, and `least-load`, which answers for one, is no rank. A
/// later rank that leaves fewer in front is a host's, and so are the rows
/// it must find level ([`fewer`]).
///
/// `None` where the rows in front cannot decide alone, the rows' sums left
/// as the ranks wrote them: a stage said it must be asked whole
/// ([`ScoreFront::Whole`]), or the tie band is wider than the ranks' room
/// ([`Order`]) and reaches a row behind. Out of line: a decision that puts
/// nobody in front carries none of this.
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "combine's inputs and what the ranks before left"
)]
fn combine_few(
    profile: &Profile,
    order: Order,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    kept: &[u32],
    scores: &mut [f32],
    stage: &mut Vec<f32>,
    tied: &mut [u32],
    mut few: usize,
    from: usize,
    norm: f32,
    mut band: f32,
) -> Option<(TieBand, usize)> {
    // What each row in front holds, kept in the stage's buffer behind the
    // room a stage's scores of those rows take, until the rows in front
    // have decided. The buffer held a score for every row when the rank
    // was asked, and the rows in front are at most half of them: nothing
    // is sized here, whatever the thread's first decision of this size was.
    let first = few;
    debug_assert!(
        stage.capacity() >= 2 * first,
        "the stage's buffer holds a score for every row"
    );
    stage.clear();
    stage.resize(2 * first, 0.0);
    let (outs, sums) = stage.split_at_mut(first);
    for (sum, &place) in sums.iter_mut().zip(&tied[first..2 * first]) {
        *sum = scores[place as usize];
    }
    let last = profile.scorers.len() - 1;
    let mut whole = false;
    // Whether every row in front holds one sum: a rank puts fewer in front
    // only of rows no stage has parted since they stood there.
    let mut level = true;
    for (i, weighted) in profile.scorers.iter().enumerate().skip(from) {
        let share = weighted.weight * norm;
        if order.alike(i) {
            // Not asked: every row in front holds its share.
            for sum in &mut sums[..few] {
                *sum += share;
            }
            continue;
        }
        // A row a stage leaves unwritten reads nothing, as in the zeroed
        // buffer the whole pass hands: not the score of the stage before.
        let out = &mut outs[..few];
        out.fill(0.0);
        let asked = &tied[..few];
        let (stage_band, said) = weighted
            .scorer
            .score_front(facts, candidates, kept, asked, out);
        // Checked ahead of the way out: a band beside `ScoreFront::Whole` is
        // not read, and is none all the same.
        debug_assert!(
            bandless(weighted, stage_band),
            "{} orders nothing and returned a band",
            weighted.scorer.name()
        );
        if said == ScoreFront::Whole {
            whole = true;
            break;
        }
        debug_assert!(
            as_said(weighted.scorer.spread(), said, out),
            "{} said {said:?} of scores that read otherwise",
            weighted.scorer.name()
        );
        band += share * stage_band.0.clamp(0.0, 1.0);
        for (sum, value) in sums[..few].iter_mut().zip(out.iter()) {
            *sum += share * value.clamp(0.0, 1.0);
        }
        match said {
            ScoreFront::At(front) if level && order.fronts(i) && i < last => {
                let left = fewer(out, tied, sums, few, front as usize);
                // The rows it left in front stand at its ceiling. Where it
                // left them all it parted them, unless all stand there.
                level = left < few || front as usize == few;
                few = left;
            }
            ScoreFront::Alike => {}
            ScoreFront::At(_) | ScoreFront::Unsaid | ScoreFront::Whole => level = false,
        }
    }
    if whole || band > order.room {
        return None;
    }
    for (&sum, &place) in sums[..few].iter().zip(&tied[few..2 * few]) {
        scores[place as usize] = sum;
    }
    Some((TieBand(band), few))
}

/// The rows in front after a rank asked of `few` of them wrote `stage`: the
/// ones at its ceiling, listed in place of the `few` with what each holds
/// (`sums`), where they fill a decision's targets and are at most half of
/// them; else the `few` stay.
fn fewer(stage: &[f32], tied: &mut [u32], sums: &mut [f32], few: usize, front: usize) -> usize {
    let at = stage.iter().filter(|&&value| value >= 1.0).count();
    if at != front || front < MAX_TARGETS || front * 2 > few {
        return few;
    }
    // The candidates and their sums first, then the places: each is written
    // at or before the entry it is read from.
    let mut next = 0;
    for at in 0..few {
        if stage[at] >= 1.0 {
            tied[next] = tied[at];
            sums[next] = sums[at];
            next += 1;
        }
    }
    let mut next = 0;
    for at in 0..few {
        if stage[at] >= 1.0 {
            tied[front + next] = tied[few + at];
            next += 1;
        }
    }
    front
}

/// The single plan's whole pass for a profile whose leading stages are
/// ranks ([`Order`]): the filter half as [`run_single`] runs it, then the
/// scoring half with the stages asked by rank ([`combine_ranked`]). Where
/// rows stand in front they fill the targets, and the picker compares them
/// alone. A keyed request under a sessions rule never comes here: it is the
/// rule's, in [`Engine::decide`]. Nor does an encode item, whose pass is its
/// own ([`run_encode`]), nor a request the profile's picker draws for.
///
/// Out of line, so a decision of a profile without ranks keeps the engine's
/// decision path as it was.
#[inline(never)]
fn run_ranked(
    engine: &Engine,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    set: Set<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Decision {
    // The profile and its ranks are read here, by the index, as a draw
    // reads its profile and its fold.
    let at = usize::from(index);
    let (profile, order) = (&engine.profiles[at], engine.orders[at]);
    let candidates = set.candidates;
    if let Some(emptied_by) = filter(profile, facts, cx, buf, set) {
        return single_emptied(index, facts, pool, candidates, emptied_by, buf);
    }
    let narrowed = profile.picker.narrow(facts, candidates, &mut buf.kept);
    let compared = narrowed.compared.min(buf.kept.len());
    let alternates = narrowed.alternates.min(buf.kept.len() - compared);
    let (band, few) = combine_ranked(
        profile,
        order,
        facts,
        candidates,
        &buf.kept[..compared],
        &mut buf.scores,
        &mut buf.stage,
        &mut buf.before_soft,
    );

    let mut targets = Targets::new();
    if few == 0 {
        profile.picker.pick(
            facts,
            candidates,
            &buf.kept[..compared],
            &buf.scores,
            band,
            &mut targets,
        );
    } else {
        // The rows in front fill the targets: the picker compares them
        // alone, by the sums every stage gave them.
        let (asked, places) = buf.before_soft.split_at(few);
        buf.stage.clear();
        buf.stage
            .extend(places.iter().take(few).map(|&at| buf.scores[at as usize]));
        profile
            .picker
            .pick(facts, candidates, asked, &buf.stage, band, &mut targets);
    }
    let trace = trace_of(index, facts, pool, candidates, &buf.kept, &buf.scores);
    if targets.is_empty() {
        return shed(ShedReason::PolicyDeclined, profile.picker.name(), trace);
    }
    // A full list keeps what it has: the walk is bounded by the list's size.
    for &i in buf.kept[compared..compared + alternates]
        .iter()
        .take(MAX_TARGETS)
    {
        targets.push(target_of(&candidates[i as usize]));
    }
    Decision::Pick(Pick {
        targets,
        trace,
        session_verdict: SessionVerdict::None,
    })
}

/// Whether `class` is one of the sorted `classes`. Class 0 is outside any
/// pairing, whatever a list names.
fn classed(classes: &[u32], class: u32) -> bool {
    class != 0 && classes.binary_search(&class).is_ok()
}

/// The class of a list of one class, the usual list: a pass over the rows
/// compares each row's class with it and searches no list, as a decision
/// by number always did. A list of class 0 alone names no class.
fn only(classes: &[u32]) -> Option<u32> {
    match classes {
        [one] if *one != 0 => Some(*one),
        _ => None,
    }
}

/// The filters over `candidates`, stage by stage in profile order, each one
/// pass over the survivors so far: the classes a chosen prefill pairs with,
/// the partnered test, the `epd` plan's compatibility and vision rules, the
/// hard filters, then each soft filter as long as it leaves something.
/// Returns the name of the stage that emptied the pool, if one did (design
/// 4.2: a shed names it): a pool with no candidate was emptied by no stage,
/// a prefill whose classes hold no decode by the partners.
///
/// A set opens with the places the request is held to, where its host holds
/// it to any, and with no other row ([`places`]): what a class or a filter
/// then keeps is among them, under any profile.
///
/// A pool a hard filter empties is put to the set's all-vetoed rule before
/// its stage is returned ([`vetoed::emptied`]). Where the rule steers, the
/// rows it kept stay in `buf.kept` behind the stage's name, and the plan's
/// home for an emptied pool tells the two by them ([`single_emptied`]): no
/// caller holds anything for the rule past this call.
fn filter(
    profile: &Profile,
    facts: &RequestFacts<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
    set: Set<'_>,
) -> Option<&'static str> {
    let Set {
        candidates,
        class,
        partnered,
        encode,
        ..
    } = set;
    let kept = &mut buf.kept;
    kept.clear();
    match class {
        // A side held to places opens with those of them its classes keep:
        // a class opens no row outside them.
        Some(classes) if places::stated(cx) => {
            places::open(kept, candidates.len(), cx);
            kept.retain(|&i| classed(classes, candidates[i as usize].pair_class));
            if kept.is_empty() {
                return Some("partners");
            }
        }
        Some(classes) => {
            let rows = 0..candidates.len() as u32;
            match only(classes) {
                Some(one) => {
                    kept.extend(rows.filter(|&i| candidates[i as usize].pair_class == one));
                }
                None => kept
                    .extend(rows.filter(|&i| classed(classes, candidates[i as usize].pair_class))),
            }
            if kept.is_empty() {
                return Some("partners");
            }
        }
        None => places::open(kept, candidates.len(), cx),
    }
    if kept.is_empty() {
        return Some("none");
    }
    if let Some(p) = partnered {
        match only(p) {
            Some(one) => kept.retain(|&i| candidates[i as usize].pair_class == one),
            None => kept.retain(|&i| classed(p, candidates[i as usize].pair_class)),
        }
        if kept.is_empty() {
            return Some("partnered");
        }
    }
    if encode {
        kept.retain(|&i| candidates[i as usize].ec_compatible);
        if kept.is_empty() {
            return Some("ec-compatible");
        }
        kept.retain(|&i| vision_fits(facts, &candidates[i as usize]));
        if kept.is_empty() {
            return Some(VISION);
        }
    }
    for hard in profile.filters.iter().filter(|f| !f.soft()) {
        hard.retain(facts, candidates, cx, kept);
        if kept.is_empty() {
            return Some(vetoed::emptied(hard.name(), profile, facts, cx, buf, set));
        }
    }
    for soft in profile.filters.iter().filter(|f| f.soft()) {
        buf.before_soft.clear();
        buf.before_soft.extend_from_slice(kept);
        soft.retain(facts, candidates, cx, kept);
        if kept.is_empty() {
            // Copied back, not swapped: a swap would hand `kept` the backup's
            // smaller buffer, and the next decision would grow it again.
            kept.extend_from_slice(&buf.before_soft);
        }
    }
    None
}

/// The single plan's opening for a request a sessions rule pins or places:
/// the profile's filters over the set's whole pool, the survivors left in
/// `buf.kept`. `Some` is the plan's answer when a filter leaves nobody
/// ([`single_emptied`]).
fn single_survivors(
    profile: &Profile,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    set: Set<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Option<Decision> {
    let emptied_by = filter(profile, facts, cx, buf, set)?;
    Some(single_emptied(
        index,
        facts,
        pool,
        set.candidates,
        emptied_by,
        buf,
    ))
}

/// What the single plan decides over a pool its filter half emptied: the
/// shed that names the stage, or the all-vetoed rule's pick where the rule
/// left the rows it kept in `buf.kept` ([`vetoed::steered`]). Every path of
/// the plan ends here when a stage leaves nobody, a plain request
/// ([`run_single`]) and the two a sessions rule or an assignment places
/// ([`single_survivors`]), and so does a side of a pair ([`run_side`]):
/// what a plan asks of an emptied pool has this one home.
///
/// Cold and out of line, its arguments in the order the shed takes them: a
/// pass that keeps a row sets up nothing for this call.
#[cold]
#[inline(never)]
fn single_emptied(
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    emptied_by: &'static str,
    buf: &mut Buffers,
) -> Decision {
    if !buf.kept.is_empty() {
        return vetoed::steered(index, facts, pool, candidates, buf);
    }
    shed(
        reason_of(emptied_by),
        emptied_by,
        trace_of(index, facts, pool, candidates, &[], &[]),
    )
}

/// The trace of a decision that counted its pool: `kept` holds every
/// survivor of the filters, and `scores` the scores of the set compared.
fn trace_of(
    profile: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    kept: &[u32],
    scores: &[f32],
) -> Trace {
    trace_by::<false>(profile, facts, pool, candidates, kept, scores)
}

/// [`trace_of`] behind a draw. The rows stand in the draw's order, so
/// their best are read as a draw's ([`top_drawn`]); and a draw reads a
/// part of the pool, so its survivors are not counted.
fn trace_drawn(
    profile: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    kept: &[u32],
    scores: &[f32],
) -> Trace {
    trace_by::<true>(profile, facts, pool, candidates, kept, scores)
}

/// The trace of a decision behind a draw that handed its rows out in an
/// order of its own: `best` names them best first, each with its score,
/// and the trace keeps the first of them. No survivor is counted, as
/// [`trace_drawn`] counts none, and no pass reads the scores again.
fn trace_ordered(
    profile: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    best: impl IntoIterator<Item = (WorkerId, f32)>,
) -> Trace {
    let mut trace = Trace {
        profile,
        survivors: Survivors::NOT_COUNTED,
        seed: facts.seed,
        generation: pool.generation,
        ..Trace::default()
    };
    let mut len = 0;
    for (slot, row) in trace.top.iter_mut().zip(best) {
        *slot = row;
        len += 1;
    }
    trace.top_len = len;
    trace
}

/// The trace of a decision over the rows `kept` and their `scores`: a
/// pool read whole, or the rows of a draw (`DRAWN`). One body, the read
/// of the best scores chosen where it is entered.
#[inline(always)]
fn trace_by<const DRAWN: bool>(
    profile: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    kept: &[u32],
    scores: &[f32],
) -> Trace {
    let mut trace = Trace {
        profile,
        survivors: if DRAWN {
            Survivors::NOT_COUNTED
        } else {
            Survivors::counted(kept.len() as u32)
        },
        seed: facts.seed,
        generation: pool.generation,
        ..Trace::default()
    };
    let mut top = [(0, f32::NEG_INFINITY); TRACE_TOP];
    let len = if DRAWN {
        top_drawn(scores, &mut top)
    } else {
        top_pass(scores, &mut top)
    };
    for (slot, &(i, score)) in top.iter().enumerate().take(len) {
        trace.top[slot] = (candidates[kept[i] as usize].id, score);
    }
    trace.top_len = len as u8;
    trace
}

/// The `K` best of `scores` in descending order, each with its place in
/// the scan, into the first slots of `top`; the answer is how many there
/// are, and a slot past them is left as it was. A row enters when it
/// beats the lowest row kept, and moves above a kept row only when
/// strictly better, so equal scores keep the scan's order (the earlier
/// row ahead).
///
/// This is the read of a pool read whole. Its rows come in the host's
/// order decision after decision: a row that beats nobody kept is passed
/// over by one test, which a processor predicts.
///
/// The slots are the caller's, not an array handed back: by count, the
/// array by value cost a whole pass a copy a decision in the ranked push
/// and four instructions a row in the trace.
#[inline(always)]
pub(crate) fn top_pass<const K: usize>(scores: &[f32], top: &mut [(usize, f32); K]) -> usize {
    let mut len = 0usize;
    for (i, &score) in scores.iter().enumerate() {
        if len == K && score <= top[K - 1].1 {
            continue;
        }
        let mut slot = if len < K {
            len += 1;
            len - 1
        } else {
            K - 1
        };
        while slot > 0 && score > top[slot - 1].1 {
            top[slot] = top[slot - 1];
            slot -= 1;
        }
        top[slot] = (i, score);
    }
    len
}

/// [`top_pass`] behind a draw: the same answer at another cost. The rows
/// come in the draw's order and no test of a score is predicted, so each
/// row is placed by its key ([`top_keyed`]), the same work whatever it
/// scores. Three lists are read as the pass reads them: one under
/// [`KEYED_FROM`] rows, one that holds a score without an order, and one
/// past the keys' places.
#[inline(always)]
pub(crate) fn top_drawn<const K: usize>(scores: &[f32], top: &mut [(usize, f32); K]) -> usize {
    if scores.len() >= KEYED_FROM
        && scores.len() <= 1 << PLACE_BITS
        && !scores.iter().fold(false, |nan, score| nan | score.is_nan())
    {
        return top_keyed(scores, top);
    }
    top_pass(scores, top)
}

/// From how many rows a draw's best scores are read by keys. A shorter
/// list holds few tests to mispredict, and the pass's form reads it in
/// fewer instructions: by count the four rows another picker draws cost
/// 92 more a decision by keys. Keys were measured over lists of sixteen
/// rows and more.
const KEYED_FROM: usize = 16;

/// How many bits of a key name a row's place: a list of up to
/// `1 << PLACE_BITS` rows is read by keys.
const PLACE_BITS: u32 = 20;

/// [`top_drawn`] over scores that order. Each row is one key: its score's
/// bits in the scores' order, above its place counted from the end, so the
/// greater key is the better score and, between equal scores, the earlier
/// row. The best `K` keys are kept by a maximum and a minimum a slot, the
/// same work whatever a row scores. A key is a whole number
/// under 2^52 and exact as an `f64`: a processor takes the greater of two
/// such numbers by one instruction, and of two integers by a conditional
/// move, which a compiler turns back into a branch inside a loop.
#[inline(always)]
fn top_keyed<const K: usize>(scores: &[f32], top: &mut [(usize, f32); K]) -> usize {
    let places = (1u64 << PLACE_BITS) - 1;
    let mut keys = [0.0f64; K];
    for (i, &score) in scores.iter().enumerate() {
        // A negative zero reads as zero. The sign bit is turned, and under
        // a set sign every other bit, so the bits order as the scores do.
        let bits = (score + 0.0).to_bits();
        let order = bits ^ (((bits as i32 >> 31) as u32) | 0x8000_0000);
        let mut key = ((u64::from(order) << PLACE_BITS | (places - i as u64)) as i64) as f64;
        for slot in &mut keys {
            let held = *slot;
            *slot = held.max(key);
            key = held.min(key);
        }
    }
    let len = scores.len().min(K);
    for (slot, &key) in top.iter_mut().zip(&keys).take(len) {
        let i = (places - (key as u64 & places)) as usize;
        *slot = (i, scores[i]);
    }
    len
}

fn shed(reason: ShedReason, emptied_by: &'static str, trace: Trace) -> Decision {
    Decision::Shed(Shed {
        reason,
        emptied_by,
        retry_after_secs: 1,
        wait_allowed: false,
        trace,
    })
}

/// The reason a shed carries, read from the stage that emptied the pool: the
/// overload filter empties a pool only when every row it saw was over the
/// thresholds with the veto in force, so that stage alone means saturation.
fn reason_of(emptied_by: &str) -> ShedReason {
    if emptied_by == "not-overloaded" {
        ShedReason::Saturated
    } else {
        ShedReason::NoEndpoints
    }
}

/// A pool to decide over: narrowed to the rows of the classes a chosen
/// prefill's class pairs with when the decision is a pair's decode side
/// (`class`: those classes, sorted), or to the rows whose class is in the
/// sorted partnered list when it is the prefill side.
#[derive(Clone, Copy)]
struct Set<'a> {
    candidates: &'a [Candidate],
    class: Option<&'a [u32]>,
    partnered: Option<&'a [u32]>,
    /// The profile's filters as one test of a row, where the pass's caller
    /// holds the engine's fold of them: a whole pass hands it to its stages
    /// ([`run_whole`]), and the all-vetoed rule's pass with the overload
    /// veto lifted asks it about a set that is the whole pool
    /// ([`vetoed::emptied`]). `None` from every other caller: that pass of
    /// the rule is staged.
    fold: Option<RowTests>,
    /// The `epd` plan's filters apply: the pool is the encode pool and the
    /// rows' compatibility and vision bits are read.
    encode: bool,
    /// What the binding says of a pool a soft veto empties ([`AllVetoed`]):
    /// the filter half puts such a pool to the rule by this word
    /// ([`vetoed::emptied`]).
    vetoed: AllVetoed,
}

impl<'a> Set<'a> {
    /// Whether the set holds the single plan's whole pool: no class
    /// narrows it and no rule of the `epd` plan reads its rows, so the
    /// profile's filters alone decide which rows stand. A whole pass asks
    /// the profile's fold about such a set alone ([`run_whole`]).
    const fn plain(&self) -> bool {
        self.class.is_none() && self.partnered.is_none() && !self.encode
    }

    /// One side of a pair: `candidates` narrowed by the classes a chosen
    /// prefill pairs with or by the partnered list. A side is put to no rule
    /// ([`AllVetoed::Unasked`]) until its caller says its binding's word
    /// ([`Self::ruled`]), as a live decode side does. A prefill side says
    /// none, nor does a pass that only names a shed's stage, a shadow's run
    /// or an encode item's: a pool a veto empties there is shed.
    const fn side(
        candidates: &'a [Candidate],
        class: Option<&'a [u32]>,
        partnered: Option<&'a [u32]>,
    ) -> Self {
        Self {
            candidates,
            class,
            partnered,
            fold: None,
            encode: false,
            vetoed: AllVetoed::Unasked,
        }
    }

    /// The same rows put to no rule ([`AllVetoed::Unasked`]): a pass over
    /// them sheds a pool a veto empties, whatever the binding says. The
    /// rule's own passes run over such a set, and so does each pass of a
    /// kept key that reads the overload veto as it stands
    /// ([`Engine::keep`]).
    const fn shedding(self) -> Self {
        Self {
            vetoed: AllVetoed::Unasked,
            ..self
        }
    }

    /// The same rows put to their binding's all-vetoed rule by its word
    /// (`vetoed`): a pair's live decode side. Where a soft veto leaves such
    /// a side nobody, the rule steers it or sheds it as it does the single
    /// plan's pool ([`vetoed::emptied`]), over the side's own rows.
    const fn ruled(self, vetoed: AllVetoed) -> Self {
        Self { vetoed, ..self }
    }
}

/// The stage a pool is shed under when none of its rows takes the pixels
/// its request holds, an image or a video: the `epd` plan's vision rule
/// sheds under it, and so does a request's vision demand
/// ([`UnmetDemand::Vision`](crate::engine::demands::UnmetDemand::Vision)).
pub(crate) const VISION: &str = "vision";

/// The `epd` plan's vision rule: a request that holds pixels, an image or
/// a video ([`MediaCounts::holds_pixels`]), takes an encoder with a vision
/// encoder; a request that holds none takes any.
///
/// [`MediaCounts::holds_pixels`]: crate::engine::view::MediaCounts::holds_pixels
fn vision_fits(facts: &RequestFacts<'_>, c: &Candidate) -> bool {
    !facts.media.holds_pixels() || c.vision
}

/// The scoring half of a profile's run, over the survivors its filter half
/// left in `buf.kept`: the scorers and the picker ([`score_and_pick`]), then
/// the trace. A picker that pushes no target declined. A pair's decode
/// decided again hands `first`, the decode its first decision named: where
/// that decode is a survivor it leads the targets and the picker is not
/// asked ([`keep_first`]), while the profile's scorers still read the
/// survivors. Inlined: each of its callers is one profile's run, as it was
/// before the halves had names, and one that hands no `first` holds no
/// test of it.
#[inline(always)]
fn score_survivors(
    profile: &Profile,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    buf: &mut Buffers,
    first: Option<(WorkerId, u16)>,
) -> Decision {
    let mut targets = Targets::new();
    if !first.is_some_and(|first| keep_first(candidates, buf, first, &mut targets)) {
        score_and_pick(profile, facts, candidates, buf, &mut targets);
    } else if !profile.scorers.is_empty() {
        // A kept decode spares the picker's turn alone: the scorers read
        // the survivors as ahead of a pick, and the trace holds their scores.
        combine(
            profile,
            facts,
            candidates,
            &buf.kept,
            &mut buf.scores,
            &mut buf.stage,
        );
    }
    // The trace counts every survivor of the filters; its top rows are of
    // the compared set, which leads the survivors and alone holds scores.
    let trace = trace_of(index, facts, pool, candidates, &buf.kept, &buf.scores);
    if targets.is_empty() {
        return shed(ShedReason::PolicyDeclined, profile.picker.name(), trace);
    }
    Decision::Pick(Pick {
        targets,
        trace,
        session_verdict: SessionVerdict::None,
    })
}

/// One profile over one side of a pair: the filter half of its run, then
/// the scoring half. The pair plans run their sides through here; the single
/// plan runs the same two halves from its own opening ([`run_single`]).
/// A side its filters emptied ends where the single plan's pool does
/// ([`single_emptied`]): in the all-vetoed rule's pick where the side's set
/// carries its binding's word ([`Set::ruled`]) and the rule kept rows, else
/// in the shed that names the stage.
fn run_profile(
    profile: &Profile,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    set: Set<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Decision {
    run_side(profile, index, facts, pool, set, cx, buf, None)
}

/// The body of [`run_profile`], which hands no `first`. A pair's decode
/// decided again enters here with the decode its first decision named,
/// where its picker takes turns, under no scorer or one it does not read
/// ([`Engine::decide_decode`]): the scoring half keeps that decode where it
/// is a survivor ([`score_survivors`]).
/// Inlined into the two, so a run that hands none is the code it was.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "a side's run, beside the decode the pair's first decision named"
)]
fn run_side(
    profile: &Profile,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    set: Set<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
    first: Option<(WorkerId, u16)>,
) -> Decision {
    let candidates = set.candidates;
    if let Some(emptied_by) = filter(profile, facts, cx, buf, set) {
        return single_emptied(index, facts, pool, candidates, emptied_by, buf);
    }
    score_survivors(profile, index, facts, pool, candidates, buf, first)
}

/// The targets of a pair's decode decided again where `first`, the worker
/// and the rank of the decode the pair's first decision named, is one of
/// the survivors in `buf.kept`: that decode, then the survivors behind it
/// in their order, wrapping, to the count a pick names. They are the
/// alternates a rotation's pick at that decode hands, and the scores are a
/// pick's under no scorer. `false` where `first` is none of the survivors,
/// with nothing written: the picker then picks among them.
fn keep_first(
    candidates: &[Candidate],
    buf: &mut Buffers,
    first: (WorkerId, u16),
    targets: &mut Targets,
) -> bool {
    let stands = buf.kept.iter().position(|&row| {
        let row = &candidates[row as usize];
        (row.id, row.dp_rank) == first
    });
    let Some(at) = stands else {
        return false;
    };
    let (before, from) = buf.kept.split_at(at);
    for &row in from.iter().chain(before).take(MAX_TARGETS) {
        targets.push(target_of(&candidates[row as usize]));
    }
    buf.scores.clear();
    buf.scores.resize(buf.kept.len(), 0.0);
    true
}

/// The staged pass of the single plan over `set` (an `epd` plan's encode
/// item among its sets, its rules in `set`): the filter half of the
/// profile's run, a stage at a time, then the scoring half; a pool the
/// filters emptied ends in the plan's one home for it ([`single_emptied`]).
/// The filter half has put such a pool to the binding's all-vetoed rule by
/// then, so the pass holds nothing for the rule. Its callers: the whole
/// pass where the fold keeps no row ([`run_whole`]), an encode item, the
/// draw over a pool's engines, and the walk under a lifted veto. Inlined
/// into each.
#[inline(always)]
fn run_single(
    profile: &Profile,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    set: Set<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Decision {
    let candidates = set.candidates;
    if let Some(emptied_by) = filter(profile, facts, cx, buf, set) {
        return single_emptied(index, facts, pool, candidates, emptied_by, buf);
    }
    score_survivors(profile, index, facts, pool, candidates, buf, None)
}

/// The whole pass at the decision's own door, and the one a draw widens to
/// ([`run_drawn`]). The profile and its fold are read here, by the index,
/// as a draw reads them. Where the build folded the profile's filters into
/// one test of a row ([`RowTests::of`]) and the set is the plan's whole
/// pool ([`Set::plain`]), the filter half is one pass of that test
/// ([`keep_folded`]). Where that pass keeps no row, or the profile holds a
/// filter the engine does not fold, the stages run one by one
/// ([`run_single`]). A side's set and an encode item's are not the whole
/// pool's: their own rules stand ahead of the profile's, stage by stage.
///
/// The fold is asked only where one of the pool's first rows passes it
/// ([`leads`]): a pool whose every row is refused is put to the stages at
/// once, which alone say what refused it. The stages take the fold with
/// their set: where the overload veto empties the pool, the all-vetoed
/// rule's pass with that veto lifted asks it ([`vetoed::emptied`]).
///
/// Out of line: the decision's entry holds nothing of the pass. By count,
/// inlined there the pass read 27 instructions fewer over 8 rows, and a
/// decision of any other kind 7 to 9 more.
#[inline(never)]
fn run_whole(
    engine: &Engine,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    mut set: Set<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Decision {
    let at = usize::from(index);
    let (profile, rows) = (&engine.profiles[at], engine.row_tests[at]);
    // The stages take the fold with their set. Written in place and
    // first: by count, a copy of the set made for the stages behind the
    // fold's pass cost a pool put to them 17 instructions.
    set.fold = rows;
    let candidates = set.candidates;
    if let Some(tests) = rows {
        let tests = tests.asked(facts, cx);
        if set.plain() && leads(tests, facts, candidates, cx) {
            // The fold's pass asks for every buffer's room, as a draw
            // does: no pass of another kind behind it grows a list.
            buf.reserve(candidates.len());
            if keep_folded(tests, facts, candidates, cx, &mut buf.kept) {
                return score_survivors(profile, index, facts, pool, candidates, buf, None);
            }
        }
    }
    run_single(profile, index, facts, pool, set, cx, buf)
}

/// The rows at a pool's front that settle whether the fold is asked about
/// the pool ([`leads`]). By count, a pool of 8 rows that sheds read 177
/// instructions more than the stages alone where the fold ran its whole
/// pass first and kept nobody, and reads 95 more where four rows settle
/// it, at any size; the check costs a pool that keeps a row 38. With every
/// second row refused, fifteen pools of sixteen keep the fold.
const FOLD_LEAD: usize = 4;

/// Whether one of the pool's first [`FOLD_LEAD`] rows passes `tests`, the
/// profile's fold as the decision asks it ([`RowTests::asked`]). The fold's
/// pass is a bet that the pool holds a row every filter keeps, and a pass
/// that keeps nobody is paid for twice: the stages run behind it. Where the
/// front rows are all refused the stages are asked at once. The host's mask
/// is not asked here: a front row it names only sends the pool to the fold,
/// which asks it.
#[inline(always)]
fn leads(
    tests: RowTests,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    cx: &Constraints<'_>,
) -> bool {
    candidates
        .iter()
        .take(FOLD_LEAD)
        .any(|c| tests.keeps(facts, c, cx))
}

/// The filter half of a plain request's whole pass by the profile's one
/// test of a row, where the build folded its filters into one
/// ([`RowTests::of`]): the rows every filter keeps, in the pool's order, are
/// left in `kept`. Says whether a row stands.
///
/// Where one does, these are the rows the staged pass leaves ([`filter`]).
/// A row every filter keeps outlives every stage, so no stage empties the
/// pool and no soft filter steps aside, and each stage drops the rows its
/// own `keep` refuses. Where none does, the pass says nothing of the pool:
/// the staged pass alone names the stage that emptied it, lets a soft
/// filter step aside and asks the all-vetoed rule. It keeps no row for a
/// request its host holds to places ([`places`]): the staged pass opens its
/// set with them.
///
/// `tests` is the set as the decision asks it: the request's own facts
/// are read once, by the caller ([`RowTests::asked`]). The host's mask is
/// asked about a row only where a veto stands, by the row's place as the
/// health filter's pass asks it. Out of line: the loop holds its own
/// registers. By count, inlined into the decision it read 72 instructions
/// more over a pool of 8 rows.
#[inline(never)]
fn keep_folded(
    tests: RowTests,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    cx: &Constraints<'_>,
    kept: &mut Vec<u32>,
) -> bool {
    kept.clear();
    if places::stated(cx) {
        return false;
    }
    if tests.masked(cx) {
        for (i, c) in (0u32..).zip(candidates) {
            if tests.keeps(facts, c, cx) && !cx.masks(i) {
                kept.push(i);
            }
        }
    } else {
        for (i, c) in (0u32..).zip(candidates) {
            if tests.keeps(facts, c, cx) {
                kept.push(i);
            }
        }
    }
    !kept.is_empty()
}

/// Whether a profile's single-plan decision may draw first ([`run_drawn`]):
/// its picker draws first and every other stage answers row by row.
fn draws_first(profile: &Profile) -> bool {
    profile.picker.draws_first()
        && profile.filters.iter().all(|f| f.row_wise())
        && profile.scorers.iter().all(|w| w.scorer.row_wise())
}

/// The rows a host named join the rows a picker drew
/// ([`Picker::joined_by_named`]): each named row every filter keeps is
/// pushed behind the draw, none twice, the holders before the warming rows.
/// Returns the mean in flight of the rows drawn, which the spill gate reads
/// in place of the mean of the rows handed. A named row past the pool names
/// no candidate and is passed over.
fn join_named(
    facts: &RequestFacts<'_>,
    named: &AffinityRows<'_>,
    candidates: &[Candidate],
    keep: &mut dyn FnMut(&Candidate) -> bool,
    kept: &mut Vec<u32>,
) -> f32 {
    let mean = mean_in_flight(facts, candidates, kept);
    let holders = named.holders.iter().map(|holder| holder.row);
    for row in holders.chain(named.warming.iter().copied()) {
        if !kept.contains(&row) && candidates.get(row as usize).is_some_and(&mut *keep) {
            kept.push(row);
        }
    }
    mean
}

/// The picker's order of a draw the rows a host named join
/// ([`Picker::joined_by_named`]): the named rows join the rows drawn
/// ([`join_named`]) and the stages are handed the drawn rows' mean in
/// flight. Out of line: a draw nothing joins keeps the code it had.
#[inline(never)]
fn order_joined(
    profile: &Profile,
    after: &RequestFacts<'_>,
    named: AffinityRows<'_>,
    candidates: &[Candidate],
    keep: &mut dyn FnMut(&Candidate) -> bool,
    buf: &mut Buffers,
    out: &mut Targets,
) {
    let mean = join_named(after, &named, candidates, keep, &mut buf.kept);
    let joined = RequestFacts {
        named: Some(AffinityRows {
            mean_in_flight: Some(mean),
            ..named
        }),
        ..*after
    };
    let band = combine(
        profile,
        &joined,
        candidates,
        &buf.kept,
        &mut buf.scores,
        &mut buf.stage,
    );
    profile
        .picker
        .order_drawn(&joined, candidates, &buf.kept, &buf.scores, band, out);
}

/// A single-plan decision by a picker that draws first
/// ([`Picker::draw_first`]), in front of the plan's opening. The picker
/// draws rows and every filter of the profile is asked about each through
/// the filter's own `keep`, hard and soft alike. A row every filter keeps is
/// a row the whole pass keeps, so every target handed out is one the pass
/// could have handed out; the scorers are handed the rows drawn and the
/// picker orders them. Where the picker takes them
/// ([`Picker::joined_by_named`]), the rows a host named join the rows drawn
/// before the scorers run ([`order_joined`]). A draw that keeps nobody says
/// nothing of the pool: the decision widens to the whole pass
/// ([`run_whole`]), whose stages alone shed, name a stage and ask the
/// all-vetoed rule. Nothing is drawn for a request its host holds to places
/// ([`places`]): the whole pass decides it.
///
/// Where the build folded the profile's filters into one test of a row
/// (`row_tests`), a row is asked about in that one test, and about the host's
/// mask only where a veto stands. A profile with a filter of a host's is
/// asked a call a filter.
///
/// Whatever follows the draw reads the seed's stream past the values the
/// draw read. The stream is continued, never restarted: a pass behind
/// refused draws that read those values again would decide by the values
/// that refused them, and skew the picker's law on that path. The trace
/// keeps the request's seed, from which the whole decision replays.
///
/// Out of line, so a decision that draws nothing keeps the engine's decision
/// path as it was.
#[inline(never)]
fn run_drawn(
    engine: &Engine,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    set: Set<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Decision {
    // The profile and its fold are read here, by the index: the call takes
    // the engine where a staged pass takes the profile, and no argument more.
    let at = usize::from(index);
    let (profile, rows) = (&engine.profiles[at], engine.row_tests[at]);
    let candidates = set.candidates;
    buf.reserve(candidates.len());
    buf.kept.clear();
    let mut draw = Draw::new(facts.seed);
    let mut draw_by = |ask: &mut dyn FnMut(&Candidate) -> bool| {
        profile
            .picker
            .draw_first(facts, pool, candidates, ask, &mut buf.kept, &mut draw)
    };
    let drew = match rows {
        // A request held to places is the whole pass's, which opens its
        // set with them: nothing is drawn, and the pass below decides.
        _ if places::stated(cx) => false,
        Some(tests) if tests.masked(cx) => {
            draw_by(&mut |c| tests.keeps(facts, c, cx) && !cx.masks(c.position))
        }
        Some(tests) => draw_by(&mut |c| tests.keeps(facts, c, cx)),
        None => draw_by(&mut |c| profile.filters.iter().all(|f| f.keep(facts, c, cx))),
    };
    // The host caps both lists at the profile's sample: past it a decision
    // reads dearer, never wrong, so a debug build alone asserts it.
    debug_assert!(
        facts.named.is_none_or(|named| {
            let cap = engine.profile_needs[at].sample as usize;
            cap == 0 || named.holders.len().max(named.warming.len()) <= cap
        }),
        "each list of rows named stands at or under the profile's sample"
    );
    // A picker that read no value of the stream left the seed as it was.
    let resumed;
    let after = if draw.state() == facts.seed {
        facts
    } else {
        resumed = RequestFacts {
            seed: draw.state(),
            ..*facts
        };
        &resumed
    };
    if drew {
        let mut targets = Targets::new();
        match after.named {
            // The rows the host named join the rows drawn where the picker
            // takes them: out of line, so a draw nothing joins keeps the
            // code it had.
            Some(named) if profile.picker.joined_by_named() => {
                // The rows named are few: each is asked a call a filter,
                // which answers as the folded test does.
                let mut keep = |c: &Candidate| profile.filters.iter().all(|f| f.keep(facts, c, cx));
                order_joined(
                    profile,
                    after,
                    named,
                    candidates,
                    &mut keep,
                    buf,
                    &mut targets,
                );
            }
            _ => {
                let band = combine(
                    profile,
                    after,
                    candidates,
                    &buf.kept,
                    &mut buf.scores,
                    &mut buf.stage,
                );
                profile.picker.order_drawn(
                    after,
                    candidates,
                    &buf.kept,
                    &buf.scores,
                    band,
                    &mut targets,
                );
            }
        }
        if !targets.is_empty() {
            let trace = trace_drawn(index, facts, pool, candidates, &buf.kept, &buf.scores);
            return Decision::Pick(Pick {
                targets,
                trace,
                session_verdict: SessionVerdict::None,
            });
        }
    }
    let mut decision = run_whole(engine, index, after, pool, set, cx, buf);
    match &mut decision {
        Decision::Pick(pick) => pick.trace.seed = facts.seed,
        Decision::Shed(shed) => shed.trace.seed = facts.seed,
        Decision::Queue(queue) => queue.trace.seed = facts.seed,
    }
    decision
}

/// The crate's own pickers that draw first, by name. Each one's draw reads
/// a row of the pool through the test it is handed and in no other way: it
/// reads the pool's size, draws places, and asks the test about the row at
/// each. So its draw may be asked over rows whose records are not written
/// yet ([`run_drawn_written`]). A host's stage takes no built-in's name and
/// states its own ([`StageRegistry`](crate::stages::StageRegistry)), so a
/// picker of one of these names is the crate's, and a host's own picker
/// that draws first is handed every record written.
#[cfg(feature = "in-process")]
const DRAWS_BY_ITS_TEST: [&str; 6] = [
    "power-of-two",
    "power-of-two-count-first",
    "best",
    "random",
    "passthrough",
    "round-robin",
];

/// A plain single decision that is a draw's, as [`Engine::drawn_for`] found
/// it: the profile that draws and its binding's word for a pool the
/// overload veto empties.
#[cfg(feature = "in-process")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Drawn {
    index: u16,
    vetoed: AllVetoed,
}

/// A record no decision wrote, as a debug build leaves every row before a
/// draw over unwritten rows: its fixed half, every veto set and a count no
/// worker holds. A stage that read it would refuse the row or cost it past
/// every written one, and the cases that hold such a draw to the whole
/// write's pick would part.
#[cfg(all(debug_assertions, feature = "in-process"))]
fn unwritten(fixed: &Candidate) -> Candidate {
    let mut row = *fixed;
    row.health.healthy = false;
    row.health.breaker_open = true;
    row.health.stalled = true;
    row.health.draining = true;
    row.health.overloaded = true;
    row.in_flight = u32::MAX;
    row
}

/// [`run_drawn`] for a host that keeps the fixed half of its rows and
/// writes a row's record when the decision asks about the row: the picker
/// draws over `fixed`, the rows as they stand with no live value, and the
/// row a draw lands on is written into `rows` at its own place (`write`,
/// handed the fixed row and the place, writes the whole record there)
/// before the profile's filters are asked about the written record. The scorers and the picker's order
/// then read `rows` at the places the draw kept, and nowhere else: the
/// profile draws first, so each of its filters and scorers answers row by
/// row ([`draws_first`]), and its picker is one whose draw reads a row
/// through its test alone ([`DRAWS_BY_ITS_TEST`]). The rows a host named
/// are written before they join the rows drawn. A draw that keeps nobody
/// widens to the whole pass as [`run_drawn`]'s does, over every row
/// written.
///
/// The decision is the one [`run_drawn`] makes over the same rows written
/// whole: the same draws of the seed's stream, the same test of each row
/// drawn, the same scores of the rows kept. A row the draw did not ask
/// about holds whatever an earlier decision of the thread left in `rows`;
/// a debug build overwrites it first ([`unwritten`]).
///
/// In a gateway's own process alone: no other host keeps its rows in two
/// halves.
#[cfg(feature = "in-process")]
#[expect(
    clippy::too_many_arguments,
    reason = "the draw's inputs and the host's two halves of a row: the fixed rows, the buffer and the write"
)]
#[inline(never)]
fn run_drawn_written(
    engine: &Engine,
    drawn: Drawn,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    fixed: &[Candidate],
    rows: &mut [Candidate],
    write: &mut dyn FnMut(&Candidate, &mut Candidate),
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Decision {
    let Drawn { index, vetoed } = drawn;
    let at = usize::from(index);
    let (profile, tests) = (&engine.profiles[at], engine.row_tests[at]);
    debug_assert_eq!(fixed.len(), rows.len(), "one buffer row a fixed row");
    #[cfg(debug_assertions)]
    for (row, fixed) in rows.iter_mut().zip(fixed) {
        *row = unwritten(fixed);
    }
    buf.reserve(fixed.len());
    buf.kept.clear();
    let mut draw = Draw::new(facts.seed);
    let masked = tests.is_some_and(|tests| tests.masked(cx));
    // A request held to places is the whole pass's, which opens its set
    // with them: nothing is drawn, and every row is written below.
    let drew = !places::stated(cx) && {
        // The row a draw lands on, written at its place: the written
        // record is the one the filters are asked about.
        let mut ask = |fixed: &Candidate| {
            let Some(row) = rows.get_mut(fixed.position as usize) else {
                return false;
            };
            write(fixed, row);
            let row = &*row;
            match tests {
                Some(tests) => tests.keeps(facts, row, cx) && !(masked && cx.masks(row.position)),
                None => profile.filters.iter().all(|f| f.keep(facts, row, cx)),
            }
        };
        profile
            .picker
            .draw_first(facts, pool, fixed, &mut ask, &mut buf.kept, &mut draw)
    };
    // The host caps both lists at the profile's sample, as it does for a
    // draw over written rows: a debug build alone asserts it.
    debug_assert!(
        facts.named.is_none_or(|named| {
            let cap = engine.profile_needs[at].sample as usize;
            cap == 0 || named.holders.len().max(named.warming.len()) <= cap
        }),
        "each list of rows named stands at or under the profile's sample"
    );
    // A picker that read no value of the stream left the seed as it was.
    let resumed;
    let after = if draw.state() == facts.seed {
        facts
    } else {
        resumed = RequestFacts {
            seed: draw.state(),
            ..*facts
        };
        &resumed
    };
    if drew {
        let mut targets = Targets::new();
        match after.named {
            Some(named) if profile.picker.joined_by_named() => {
                // The rows named are few: each is written before the join
                // asks about it, but a row the draw kept, which is written.
                let holders = named.holders.iter().map(|holder| holder.row);
                for place in holders.chain(named.warming.iter().copied()) {
                    if buf.kept.contains(&place) {
                        continue;
                    }
                    let at = place as usize;
                    if let (Some(fixed), Some(row)) = (fixed.get(at), rows.get_mut(at)) {
                        write(fixed, row);
                    }
                }
                let mut keep = |c: &Candidate| profile.filters.iter().all(|f| f.keep(facts, c, cx));
                order_joined(profile, after, named, rows, &mut keep, buf, &mut targets);
            }
            _ => {
                let band = combine(
                    profile,
                    after,
                    rows,
                    &buf.kept,
                    &mut buf.scores,
                    &mut buf.stage,
                );
                profile
                    .picker
                    .order_drawn(after, rows, &buf.kept, &buf.scores, band, &mut targets);
            }
        }
        if !targets.is_empty() {
            let trace = trace_drawn(index, facts, pool, rows, &buf.kept, &buf.scores);
            return Decision::Pick(Pick {
                targets,
                trace,
                session_verdict: SessionVerdict::None,
            });
        }
    }
    // The draw says nothing of the pool: the whole pass reads every row,
    // and the pool's facts say of the rows written what a host that wrote
    // them first says ([`PoolFacts::reported`]).
    let mut reported = true;
    for (row, fixed) in rows.iter_mut().zip(fixed) {
        write(fixed, row);
        reported &= row.load.is_some();
    }
    let pool = &PoolFacts { reported, ..*pool };
    let set = Set {
        candidates: rows,
        class: None,
        partnered: None,
        fold: None,
        encode: false,
        vetoed,
    };
    let mut decision = run_whole(engine, index, after, pool, set, cx, buf);
    match &mut decision {
        Decision::Pick(pick) => pick.trace.seed = facts.seed,
        Decision::Shed(shed) => shed.trace.seed = facts.seed,
        Decision::Queue(queue) => queue.trace.seed = facts.seed,
    }
    decision
}

/// The whole pass of an encode item of the `epd` plan: the plan's rules
/// filter the encode pool before the profile's ([`filter`]). Out of line,
/// so the decision's entry builds one set for every other request and the
/// flag is this call's alone.
#[inline(never)]
fn run_encode(
    profile: &Profile,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    set: Set<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Decision {
    let set = Set {
        encode: true,
        ..set
    };
    run_single(profile, index, facts, pool, set, cx, buf)
}

/// The plan: the sequence of profiles a request runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Plan {
    /// One profile over one pool.
    #[default]
    Single,
    /// The prefill profile over the open prefills, then the decode profile
    /// over the chosen prefill's partners.
    PdPair,
    /// The encode profile named here, once per encode item over the encode
    /// pool with the pair's EC compatibility and the request's vision demand
    /// as the plan's own filters, then the pair as under `PdPair`: the
    /// binding's `profile` is the prefill side and `decode_profile` the
    /// decode side. The encode side rides the plan, so a binding keeps the
    /// pair's two fields.
    Epd {
        /// The encode profile, by position.
        encode: u16,
    },
}

impl Plan {
    /// Whether `phase` is one of the sides this plan decides together, at
    /// the pair's prefill. A request of any other phase stands alone: one
    /// profile over the one list of its phase.
    #[cfg(feature = "in-process")]
    pub(crate) const fn pairs(self, phase: Phase) -> bool {
        match self {
            Self::Single => false,
            Self::PdPair => matches!(phase, Phase::Prefill | Phase::Decode),
            Self::Epd { .. } => !matches!(phase, Phase::Single),
        }
    }
}

/// The profile a binding resolves a phase to, by position; `None` when the
/// binding's plan has no side for the phase: a single or an encode request
/// under the pair plan, a single request under `epd`. The host and the spec
/// disagree there, and the decision is declined by the binding, as a pair
/// decision under a single binding is.
fn profile_index(binding: &Binding, phase: Phase) -> Option<u16> {
    match (binding.plan, phase) {
        (Plan::Single, _) | (Plan::PdPair | Plan::Epd { .. }, Phase::Prefill) => {
            Some(binding.profile)
        }
        (Plan::PdPair | Plan::Epd { .. }, Phase::Decode) => {
            Some(binding.decode_profile.unwrap_or(binding.profile))
        }
        (Plan::Epd { encode }, Phase::Encode) => Some(encode),
        (Plan::PdPair, Phase::Single | Phase::Encode) | (Plan::Epd { .. }, Phase::Single) => None,
    }
}

/// `entry` where its binding's plan pairs (`pd-pair` or `epd`): what makes
/// a model's prefill entry its pair entry ([`Engine::pair_entry`]).
fn paired(entry: &Entry) -> Option<&Entry> {
    matches!(entry.binding.plan, Plan::PdPair | Plan::Epd { .. }).then_some(entry)
}

/// The shadow profile a binding names for `phase`, by position: a
/// single's for each phase, a pair's a side. `None` for a pair's encode
/// side, which names none.
fn shadow_index(binding: &Binding, phase: Phase) -> Option<u16> {
    match (binding.plan, phase) {
        (Plan::Single, _) | (Plan::PdPair | Plan::Epd { .. }, Phase::Prefill) => binding.shadow,
        (Plan::PdPair | Plan::Epd { .. }, Phase::Decode) => binding.decode_shadow,
        (Plan::PdPair | Plan::Epd { .. }, Phase::Single | Phase::Encode) => None,
    }
}

/// The overload thresholds a binding hands the pools decided under it
/// ([`Engine::pool_binding`]): the crate's two defaults where its block is
/// switched on, resolved once by validation. A block names no number.
/// The host judges an engine's load reports against them and latches the
/// verdict on the engine's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverloadThresholds {
    /// Queued requests, summed over an engine's ranks, at or above which
    /// the engine's rows are vetoed.
    pub waiting_requests: u32,
    /// KV usage in permille, the mean over an engine's ranks, at or above
    /// which the engine's rows are vetoed.
    pub kv_usage_permille: u16,
}

/// A binding: which profile (or plan of profiles) a request resolves to.
#[derive(Clone, Debug)]
pub struct Binding {
    /// A model glob: `*`, `prefix*`, `*suffix` or an exact id.
    pub model: String,
    /// The phases the binding covers; empty covers every phase.
    pub phases: Vec<Phase>,
    pub plan: Plan,
    /// The profile for a single decision, and the prefill side of a pair.
    pub profile: u16,
    /// The decode profile of the `pd-pair` plan; `None` under `single`.
    pub decode_profile: Option<u16>,
    /// The shadow of `profile`, by position.
    pub shadow: Option<u16>,
    /// The shadow of `decode_profile`, by position.
    pub decode_shadow: Option<u16>,
    /// Sticky sessions: a keyed request stays on the worker its key was
    /// placed on (design 4.6), under the `single` plan and on each side of
    /// a pair, by the side's own session.
    pub sessions: Option<SessionRule>,
    /// The overload thresholds of the pools decided under the binding
    /// ([`Engine::pool_binding`]); `None` when its block is switched off,
    /// and for a binding built by hand that names none.
    pub overload: Option<OverloadThresholds>,
}

impl Binding {
    fn single_shadow(&self) -> Option<u16> {
        (self.plan == Plan::Single).then_some(self.shadow).flatten()
    }

    fn matches(&self, model: &str, phase: Phase) -> bool {
        let model_ok = match (self.model.strip_prefix('*'), self.model.strip_suffix('*')) {
            _ if self.model == "*" => true,
            (Some(suffix), _) => model.ends_with(suffix),
            (_, Some(prefix)) => model.starts_with(prefix),
            _ => self.model == model,
        };
        model_ok && (self.phases.is_empty() || self.phases.contains(&phase))
    }

    /// Whether the binding's phase list covers `phase`: an empty list
    /// covers each.
    fn covers(&self, phase: Phase) -> bool {
        self.phases.is_empty() || self.phases.contains(&phase)
    }

    /// The model pattern, read as [`Self::matches`] reads it.
    fn pattern(&self) -> Pattern<'_> {
        match (self.model.strip_prefix('*'), self.model.strip_suffix('*')) {
            _ if self.model == "*" => Pattern::Any,
            (Some(suffix), _) => Pattern::Suffix(suffix),
            (_, Some(prefix)) => Pattern::Prefix(prefix),
            _ => Pattern::Exact(&self.model),
        }
    }

    /// Whether one model id matches this binding's pattern and `other`'s:
    /// each of the two can be the first to match a pool of that model.
    fn shares_a_model(&self, other: &Self) -> bool {
        use Pattern::{Any, Exact, Prefix, Suffix};
        match (self.pattern(), other.pattern()) {
            // Every id, and the id that starts with the one's prefix and
            // ends with the other's suffix.
            (Any, _) | (_, Any) | (Prefix(_), Suffix(_)) | (Suffix(_), Prefix(_)) => true,
            (Prefix(a), Prefix(b)) => a.starts_with(b) || b.starts_with(a),
            (Suffix(a), Suffix(b)) => a.ends_with(b) || b.ends_with(a),
            (Exact(a), Exact(b)) => a == b,
            (Exact(id), Prefix(prefix)) | (Prefix(prefix), Exact(id)) => id.starts_with(prefix),
            (Exact(id), Suffix(suffix)) | (Suffix(suffix), Exact(id)) => id.ends_with(suffix),
        }
    }
}

/// A binding's model pattern: every id, the ids that end with a suffix,
/// the ids that start with a prefix, or one id.
enum Pattern<'a> {
    Any,
    Suffix(&'a str),
    Prefix(&'a str),
    Exact(&'a str),
}

/// What a host names a profile's rows by, where it names them in the
/// affinity column's place: the cap of each list
/// ([`Engine::affinity_rows_cap`]), the one ruling of the warming rows
/// ([`Engine::warm_up_rule_for`]), and the record of sent prefixes, whether
/// the profile's pool keeps it ([`Engine::reads_sent_for`]) and whether
/// its holders are named ahead of the index's
/// ([`Engine::reads_sent_alone_for`]). A naming that
/// fills its cap is held to a test of the profile's own too
/// ([`Engine::keeps_for`]). Every input of a naming that is a profile's
/// own stands here: two profiles are named the same rows where
/// [`Self::names_as`] says so.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NamedBy {
    /// The cap of each list: `0` where the profile is named no row.
    cap: u32,
    /// The one ruling its warm-up stages rule by: `None` where it holds
    /// no such stage, and where two of them rule by different numbers.
    rule: Option<WarmUpRule>,
    /// Whether its pool keeps the record of sent prefixes for it: a stage
    /// of it reads them, in the rows it is named or in the column it is
    /// handed whole.
    sent: bool,
    /// Whether it reads a request's holders from the record of sent
    /// prefixes alone: a scorer of it reads the record and none reads the
    /// holders from the index.
    sent_alone: bool,
    /// The test a naming that fills its cap is held to: the profile's
    /// filters, asked of a row ahead of its naming.
    held: Held,
}

impl NamedBy {
    /// Whether a profile named by `other` is named the rows this one is:
    /// by one cap, one ruling and one reading of the record. A naming
    /// that fills its cap is held to its profile's own test of a row. So
    /// under a cap that can fill, the two are held to one test too.
    fn names_as(&self, other: &Self) -> bool {
        let fills = self.cap != u32::MAX;
        (self.cap, self.rule) == (other.cap, other.rule)
            && (self.sent, self.sent_alone) == (other.sent, other.sent_alone)
            && (!fills || self.held.same_test(other.held))
    }
}

/// What a naming that fills its cap is held to: the test its profile's
/// standing pass asks of a row ([`Engine::keeps_for`]), as far as one
/// profile's can be told from another's. The key's mark is no part of
/// it: the mark is set behind the build, and the test reads it where it
/// is made.
#[derive(Clone, Copy, Debug)]
enum Held {
    /// To no test: the pass alone can say which rows stay
    /// ([`Profile::asked_by_row`]), and the naming stands as it is made.
    Unasked,
    /// To the profile's filters as the one test the build folded them
    /// into: two profiles of one set refuse the same rows.
    Folded(RowTests),
    /// To the profile's filters by a call each: no two such profiles are
    /// known to refuse the same rows.
    Called,
}

impl Held {
    /// Whether a naming held to `other` is held to the same test.
    fn same_test(self, other: Self) -> bool {
        match (self, other) {
            (Self::Unasked, Self::Unasked) => true,
            (Self::Folded(own), Self::Folded(others)) => own == others,
            _ => false,
        }
    }
}

/// How a binding's shadow reads the rows its live decision was handed
/// ([`Engine::shadow_for`]). A host names rows for the live profile by
/// what that profile says it is named by ([`NamedBy`]). They are the
/// shadow profile's own rows where the two say one value. Where the two
/// differ by the cap alone, or by the test a full cap is held to, or the
/// live profile is named no row, a host names the shadow's rows again
/// ([`Self::Own`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShadowRows {
    /// As the live decision read them: the shadow profile would be handed
    /// the same.
    Same,
    /// Less the rows named for the live profile: the shadow profile reads
    /// no column, and would be named none.
    Unnamed,
    /// By rows the host names again for it, under the shadow profile's
    /// own cap ([`Engine::shadow_cap_at`]) and its own test of a row: that
    /// profile names by another cap than the live one, a cap the two
    /// share can fill and they hold it to different tests, or the live
    /// one is named no row.
    /// A pool keeps one ruling of its warming rows and one record of
    /// sent prefixes, and both are the shadow profile's own there
    /// ([`Engine::pool_rule_for`], [`Engine::pool_record_for`]). The
    /// word carries no cap: every decision of a shadowed generation
    /// reads the word, and only one that names such rows reads the cap.
    Own,
    /// Not at all: the shadow profile would be named rows of another
    /// warm-up ruling or another record than the pool keeps, or handed a
    /// column the live decision was not. So too where the pool would
    /// keep the record for the shadow alone while a live profile reads
    /// the record of the same model ([`live_reader_beside`]): a model
    /// keeps one record under one bound. A pair's decode side is such a
    /// reader in its decode pool. Take a decode binding listed ahead of
    /// the pair's whose live profile is named no row: beside that side
    /// its shadow is not asked where the shadow reads the record,
    /// whichever way it reads it.
    Apart,
    /// A side at a time: the binding is a pair's, and a side that names
    /// a shadow is decided once more by it ([`Engine::shadow_side`]). A
    /// pair's sides read their pools' columns and are named no rows.
    Sides,
}

impl ShadowRows {
    /// How the shadow of `binding` reads its live decision's rows, by what
    /// each of the two profiles is named by and whether the shadow's reads
    /// the column at all. `beside` says whether a live profile reads the
    /// record of sent prefixes for a model the binding decides
    /// ([`live_reader_beside`]). It is asked where the pool would keep
    /// the record for the shadow alone.
    fn of(
        binding: &Binding,
        named_by: &[NamedBy],
        needs: &[Needs],
        beside: impl FnOnce() -> bool,
    ) -> Self {
        let Some(shadow) = binding.single_shadow().map(usize::from) else {
            return Self::Apart;
        };
        let live = usize::from(binding.profile);
        let (Some(live), Some(by)) = (named_by.get(live), named_by.get(shadow)) else {
            return Self::Apart;
        };
        let column = needs.get(shadow).is_some_and(|needs| needs.affinity);
        match (live.cap, by.cap) {
            // A shadow that reads the record in the column it is handed
            // whole: the pool of a live profile that reads none keeps none.
            (0, 0) if by.sent && !live.sent => Self::Apart,
            // Neither is named a row: both read the view as it is handed.
            (0, 0) => Self::Same,
            // The live profile alone is named rows. A shadow that reads
            // no column decides without them, as it would live.
            (_, 0) if !column => Self::Unnamed,
            (_, 0) => Self::Apart,
            // One value names the same rows: a cap that can fill is
            // held to one test of a row by both.
            _ if live.names_as(by) => Self::Same,
            // The shadow profile is named rows of its own. The pool's
            // warming rows are of one ruling and its record of one
            // profile: the live one's. A live profile that is named no
            // row and reads no record leaves the record to the shadow
            // there, and the ruling too where that profile rules by
            // none. The record is the model's, though, under one bound:
            // entries kept for a shadow never stand under the bound a
            // live profile's stand under, where they would move which
            // of those the record keeps.
            (live_cap, _) => {
                let left = live_cap == 0;
                let ruled = live.rule == by.rule || (left && live.rule.is_none());
                let recorded = if left && !live.sent {
                    !(by.sent && beside())
                } else {
                    (live.sent, live.sent_alone) == (by.sent, by.sent_alone)
                };
                if ruled && recorded {
                    Self::Own
                } else {
                    Self::Apart
                }
            }
        }
    }
}

/// The shadow profile of each side of a pair's binding, the prefill's
/// then the decode's, where the build finds it free to run beside the
/// live one: no binding decides by it live and it shows its rows to no
/// plugin, as a single's shadow ([`Engine::shadow_for`]). None for a
/// binding of one profile.
fn side_shadows(binding: &Binding, live: &[bool], marks: &[Marks]) -> [Option<u16>; 2] {
    if !matches!(binding.plan, Plan::PdPair | Plan::Epd { .. }) {
        return [None; 2];
    }
    [binding.shadow, binding.decode_shadow].map(|shadow| {
        shadow.filter(|&index| {
            let at = usize::from(index);
            live.get(at) == Some(&false) && marks.get(at).is_some_and(|marks| !marks.scoped())
        })
    })
}

/// Whether the shadow profile at `shadow`, of `side` of a pair's
/// `binding`, would be handed another column than its live side's pool
/// carries. A pair names no rows: each side reads its pool's column. A
/// pool is read by one ruling of its warming rows and keeps one record
/// of sent prefixes: those of the profile a request of the pool's phase
/// asked alone resolves to ([`Engine::pool_rule_for`],
/// [`Engine::pool_record_for`]). That profile is the pair's side profile
/// unless a binding of that phase is listed ahead of the pair's. Even
/// then a decode pool keeps the record the pair's decode profile reads,
/// where that binding's profile reads none. A shadow profile of another
/// ruling than its side's, or of another reading of the record, would
/// be handed another column live. One that reads no column is handed
/// none either way.
///
/// Two limits stand. A pair decided over a store's rows is handed each
/// row's admission age and no overlap: a side reads no holder of a
/// prefix there, live or in its shadow, so the test skips more than
/// that column tells apart. A prefill-decode pair decided over worker
/// lists reads its holders, a side at a time. And the test compares the
/// binding's own two profiles: under a binding of the side's phase
/// listed ahead of the pair's, the pool carries that binding's ruling,
/// and its record where its profile reads one, for the live side and
/// for its shadow alike, and the test still answers by the pair's two.
fn side_reads_another_column(
    binding: &Binding,
    side: PairSide,
    shadow: u16,
    named_by: &[NamedBy],
    needs: &[Needs],
) -> bool {
    let live = match side {
        PairSide::Prefill => binding.profile,
        PairSide::Decode => binding.decode_profile.unwrap_or(binding.profile),
    };
    // The ruling, and the record where the pool keeps one: whether its
    // holders are named ahead of the index's.
    let carried = |index: u16| {
        let by = named_by.get(usize::from(index))?;
        Some((by.rule, by.sent.then_some(by.sent_alone)))
    };
    let column = needs
        .get(usize::from(shadow))
        .is_some_and(|needs| needs.affinity);
    column && carried(live) != carried(shadow)
}

/// Whether a live profile reads the record of sent prefixes for a model
/// the binding at `at` decides a pool of, by another binding of
/// `bindings`. A model keeps one record under one bound, and each of its
/// pools that keeps the record writes there
/// ([`Engine::pool_record_for`]): a pool that kept it for a shadow alone
/// would write under the bound that live profile's entries stand under.
///
/// The answer is of the document, by the bindings' patterns, phase lists
/// and order: no membership is asked. It is one word for the binding, so
/// it says yes for each of its models where it is so for one. It reads
/// two bindings at a time, so it says yes too where a third takes the
/// pools those two would share. A profile counts as a reader where its
/// pool keeps the record for it ([`Engine::reads_sent_for`]).
///
/// A pair's decode profile counts wherever the pair is found by its
/// prefill, whichever binding's phase list covers the decode phase. The
/// pair decides its decode side under its own binding, and the decode
/// pool keeps the record for that side where the profile a decode
/// request asked alone resolves to reads none.
fn live_reader_beside(bindings: &[Binding], at: usize, named_by: &[NamedBy]) -> bool {
    let Some(own) = bindings.get(at) else {
        return false;
    };
    let sent = |other: &Binding, phase: Phase| {
        profile_index(other, phase)
            .and_then(|index| named_by.get(usize::from(index)))
            .is_some_and(|by| by.sent)
    };
    let reads = |other: &Binding, phase: Phase| other.covers(phase) && sent(other, phase);
    // A pair is found by its prefill alone, and its decode side reads
    // the decode pool whatever its own phase list says of that phase.
    let side = |other: &Binding| {
        matches!(other.plan, Plan::PdPair | Plan::Epd { .. })
            && other.covers(Phase::Prefill)
            && sent(other, Phase::Decode)
    };
    bindings.iter().enumerate().any(|(other_at, other)| {
        if other_at == at || !own.shares_a_model(other) {
            return false;
        }
        if other_at < at {
            // The other binding is found first. Where it covers each
            // phase this one covers, it takes every pool of a model the
            // two share, and this one decides none of that model.
            let takes = PHASES
                .iter()
                .all(|&phase| !own.covers(phase) || other.covers(phase));
            !takes && (side(other) || PHASES.iter().any(|&phase| reads(other, phase)))
        } else {
            // This binding is found first: the other decides a pool of
            // a model of its own only in a phase this one leaves. A pair
            // left the prefill lookup decides its decode side too, in
            // the decode pool, whether or not this one covers that phase.
            let left = |&phase: &Phase| !own.covers(phase) && reads(other, phase);
            (!own.covers(Phase::Prefill) && side(other)) || PHASES.iter().any(left)
        }
    })
}

/// The engine: validated profiles and the ordered bindings over them, and the
/// session table the bindings' sessions rules read.
#[derive(Debug)]
pub struct Engine {
    profiles: Vec<Profile>,
    /// The bindings in the spec's order, each beside what the build derived
    /// from it ([`Entry`]).
    bindings: Vec<Entry>,
    /// A shadow's placing assignment beside each binding, in the same
    /// order. Empty when no binding names a shadow.
    shadow_placing: Box<[Option<Assignment>]>,
    /// How each binding's shadow reads the rows its live decision was
    /// handed ([`ShadowRows`]), in the same order and as empty.
    shadow_rows: Box<[ShadowRows]>,
    /// Each side's shadow profile beside each pair's binding, the
    /// prefill's then the decode's ([`side_shadows`]), in the same order
    /// and as empty.
    shadow_sides: Box<[[Option<u16>; 2]]>,
    /// Per pair's binding and side, whether the side's shadow profile is
    /// named and not asked ([`side_reads_another_column`]), in the same
    /// order and as empty.
    side_apart: Box<[[bool; 2]]>,
    generation: u64,
    needs: Needs,
    /// The needs of every binding that covers a phase, its profile's stages
    /// and its sessions rule, folded per phase ([`Self::needs_for_phase`]),
    /// indexed by [`slot`].
    phase_needs: [Needs; PHASES.len()],
    /// Per phase, whether a binding that covers it under the single plan
    /// pins a key, or has a profile whose one scorer places a keyed
    /// request by itself on a ring: the phases whose requests the keyed
    /// decision over rows read one at a time answers
    /// ([`Self::decide_keyed`]), indexed by [`slot`]. Every other phase's
    /// request is turned away there by this one read.
    keyed: [bool; PHASES.len()],
    /// The session table this engine's decisions read and write: its own,
    /// made with it, or the one its host keeps across documents
    /// ([`Self::share_sessions`]).
    sessions: Arc<SessionTable>,
    assignment: AssignmentStages,
    /// Per profile, what its stages ask of a decision beyond the profile's
    /// own run ([`Marks`]).
    marks: Vec<Marks>,
    /// Per profile, how its scorers place a request by its heads on a lent
    /// ring ([`HeadPlaces`]).
    head_places: Vec<HeadPlaces>,
    /// Per profile, what its stages ask of a host ([`Profile::needs`]),
    /// asked of them once at the build: a decision reads its answer here.
    profile_needs: Vec<Needs>,
    /// Per profile, what a host names the rows of its decisions by
    /// ([`NamedBy`]), said once at the build: the cap
    /// ([`Self::affinity_rows_cap`]), the warm-up ruling
    /// ([`Self::warm_up_rule_for`]), the record of sent prefixes
    /// ([`Self::reads_sent_for`], [`Self::reads_sent_alone_for`]) and
    /// the test a full cap is held to ([`Self::keeps_for`]).
    named_by: Vec<NamedBy>,
    /// Per profile, its filters as one test of a row, where the engine
    /// holds every one itself ([`RowTests::of`]). A decision that draws
    /// first asks it about a row in place of a call a filter, and a whole
    /// pass keeps the rows of the single plan's pool by it
    /// ([`keep_folded`]).
    row_tests: Vec<Option<RowTests>>,
    /// Per profile, which of its leading stages are ranks ([`Order`]). A
    /// decision reads it where the profile's mark says so, and no other
    /// decision does.
    orders: Vec<Order>,
    /// Per profile, whether its draw may be asked before a row's record is
    /// written ([`Engine::drawn_for`]): it draws first, and its picker is
    /// one of the crate's own, whose draw reads a row through the test it
    /// is handed and in no other way ([`DRAWS_BY_ITS_TEST`]).
    #[cfg(feature = "in-process")]
    written_draws: Box<[bool]>,
    /// What the door needs of a seat, zero on the plain path.
    seat_flags: u8,
    explain_sample: u32,
    max_report_age_ms: Option<u64>,
    /// The stages that keep a number of the reports
    /// ([`Lifecycle::keeps_loads`]), each by its place: listed once at the
    /// build, so a publication asks no stage whether it keeps them.
    keepers: Vec<Keeper>,
    /// One tally per profile that reads sends, in the order of the
    /// profiles' names: a later engine finds a profile's by its name
    /// ([`Engine::succeed`]).
    pub(crate) sent_credits: Vec<Arc<DispatchCredit>>,
    /// The kinds of event the document takes, but for its counts of sends
    /// ([`Lifecycle::hears`] adds those): each stage's answer, asked once at
    /// the build, with the engine's own. A word of a request is told to
    /// every profile's stages, so the answer is the document's.
    hears: EventSet,
}

/// A stage that keeps a number of the reports, by its place in the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Keeper {
    /// The scorer at `stage` of the profile at `profile`.
    Scorer { profile: u16, stage: u16 },
    /// The picker of the profile at `profile`.
    Picker { profile: u16 },
}

impl Keeper {
    /// The stages of `profiles` that say they keep the reports, in the
    /// profiles' order, a profile's scorers ahead of its picker.
    fn of(profiles: &[Profile]) -> Vec<Self> {
        let mut keepers = Vec::new();
        // Inclusive ranges: a walk over as many stages as a place can
        // name ends with the list, where an open range steps past its end.
        for (at, profile) in (0..=u16::MAX).zip(profiles) {
            for (stage, weighted) in (0..=u16::MAX).zip(&profile.scorers) {
                if weighted.scorer.keeps_loads() {
                    keepers.push(Self::Scorer { profile: at, stage });
                }
            }
            if profile.picker.keeps_loads() {
                keepers.push(Self::Picker { profile: at });
            }
        }
        keepers
    }
}

/// A stage keeps state for each decision slot, so each model decides at its
/// own slot.
pub(crate) const KEEPS_SLOT_STATE: u8 = 1;
/// The generation samples decisions at the seat where they were made.
pub(crate) const SAMPLES_DECISIONS: u8 = 2;
/// A binding of the generation names a shadow.
pub(crate) const SHADOWS: u8 = 4;

/// Which of a profile's leading stages are ranks, read once when the engine
/// is built: a stage that writes levels ([`ScoreSpread::Levels`]) and whose
/// weight outweighs every stage behind it, or a stage that orders nothing
/// ([`ScoreSpread::Alike`]). A rank leaves two rows tied, or apart by more than
/// every later stage can add, so the stages behind it are asked of the rows
/// it left tied ([`combine_ranked`]).
///
/// A rank is a stage here. An engine's data-parallel rank is a candidate
/// row ([`Candidate::dp_rank`]), and no stage or plan of its own.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Order {
    /// How many leading stages are ranks; none, and the profile's score is
    /// the weighted sum of every stage over every row.
    ranks: u8,
    /// The ranks that order nothing, a bit a stage.
    alike: u8,
    /// The ranks that may put rows in front, a bit a stage: one that writes
    /// levels where every stage behind it orders nothing or answers for a
    /// front alone ([`Scorer::front_wise`]), so each may be asked of those
    /// rows alone.
    fronts: u8,
    /// The widest tie band under which the rows a rank put in front stay
    /// apart from the rows behind them: half the least a rank's step
    /// outweighs the stages behind it by, as a share of the profile's
    /// weights. A decision whose band is wider asks every row every stage.
    room: f32,
}

impl Order {
    /// The most leading stages read as ranks: each has a bit in `alike`.
    const MOST: usize = 8;
    /// The least share of the profile's weights a rank's step outweighs the
    /// stages behind it by. Under it the rounding of two rows' sums could
    /// meet, and the stage is read as no rank.
    const LEAST: f32 = 1e-4;

    fn of(profile: &Profile) -> Self {
        let stages = &profile.scorers;
        let mut order = Self::default();
        let sound = stages
            .iter()
            .all(|stage| stage.weight > 0.0 && stage.weight.is_finite());
        if !sound {
            return order;
        }
        let total: f32 = stages.iter().map(|stage| stage.weight).sum();
        let mut room = f32::INFINITY;
        for (i, stage) in stages.iter().enumerate().take(Self::MOST) {
            match stage.scorer.spread() {
                ScoreSpread::Alike => order.alike |= 1 << i,
                ScoreSpread::Levels(step) => {
                    // The most the stages behind can add between two rows.
                    let behind: f32 = stages[i + 1..]
                        .iter()
                        .filter(|later| later.scorer.spread() != ScoreSpread::Alike)
                        .map(|later| later.weight)
                        .sum();
                    // By how much its step outweighs them, of the whole.
                    let over = (stage.weight * step - behind) / total;
                    if over.is_nan() || over < Self::LEAST {
                        break;
                    }
                    room = room.min(over / 2.0);
                    let alone = stages[i + 1..].iter().all(|later| {
                        later.scorer.spread() == ScoreSpread::Alike || later.scorer.front_wise()
                    });
                    if alone {
                        order.fronts |= 1 << i;
                    }
                }
                ScoreSpread::Any => break,
            }
            order.ranks = i as u8 + 1;
        }
        // Two stages behind the ranks that each order the rows: asked of the
        // rows in front alone, each would be weighed against the other anew,
        // so no rank of the profile puts rows in front.
        let plain = stages[usize::from(order.ranks)..]
            .iter()
            .filter(|later| later.scorer.spread() != ScoreSpread::Alike)
            .count();
        if plain > 1 {
            order.fronts = 0;
        }
        // A profile whose ranks all order nothing puts no row in front.
        order.room = if room.is_finite() { room } else { 0.0 };
        order
    }

    /// Whether stage `i` is a rank.
    const fn ranks(self, i: usize) -> bool {
        i < self.ranks as usize
    }

    /// Whether a decision by rank has anything to leave unasked: a rank
    /// that may put rows in front, or one that orders nothing.
    const fn gains(self) -> bool {
        self.fronts != 0 || self.alike != 0
    }

    /// Whether stage `i` is a rank that may put rows in front. A stage past
    /// the ranks is none, whatever its index: the bits are the ranks' alone.
    const fn fronts(self, i: usize) -> bool {
        self.ranks(i) && self.fronts & (1 << i) != 0
    }

    /// Whether stage `i` is a rank that orders nothing.
    const fn alike(self, i: usize) -> bool {
        self.ranks(i) && self.alike & (1 << i) != 0
    }
}

/// The positions of the two profiles a request is placed by on a lent
/// ring ([`Engine::ring_profiles`]): the one that decides it and a pair's
/// decode profile, each where there is one.
pub(crate) type RingProfiles = (Option<usize>, Option<usize>);

/// How a profile's scorers place a request by its heads on a lent ring
/// ([`Scorer::places_by_heads`]), asked of them once when the engine is
/// built. What makes a request reads the answer here and asks no stage
/// ([`Engine::states_head_places`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum HeadPlaces {
    /// No scorer of the profile reads a ring.
    #[default]
    Unread,
    /// Every scorer of the profile that reads a ring places by heads cut
    /// by this one rule: its depth, and the view's boundaries.
    Rule(u32, Box<[u32]>),
    /// A scorer reads a ring by another place than a head's, or two
    /// scorers name two rules: no one list of heads' places is the
    /// profile's.
    Apart,
}

impl HeadPlaces {
    fn of(profile: &Profile) -> Self {
        let mut found = Self::Unread;
        for stage in &profile.scorers {
            let stated = match stage.scorer.places_by_heads() {
                Some(rule) => Self::Rule(rule.depth, rule.boundaries.into()),
                None if stage.scorer.needs().ring => return Self::Apart,
                None => continue,
            };
            if found != Self::Unread && found != stated {
                return Self::Apart;
            }
            found = stated;
        }
        found
    }

    /// The one rule of this profile and of `other`, a pair's second side
    /// where the request has one: a side that reads no ring holds nothing
    /// back, and two rules are none.
    fn with<'a>(&'a self, other: Option<&'a Self>) -> Option<HeadRule<'a>> {
        let one = match (self, other) {
            (one, None | Some(Self::Unread)) => one,
            (Self::Unread, Some(one)) => one,
            (own, Some(other)) if own == other => own,
            _ => return None,
        };
        match one {
            Self::Rule(depth, boundaries) => Some(HeadRule {
                depth: *depth,
                boundaries,
            }),
            Self::Unread | Self::Apart => None,
        }
    }
}

/// What a profile's stages ask of a decision beyond the profile's own run,
/// read once when the engine is built. A decision reads the word in one
/// compare, so a profile that asks nothing pays for no mark.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Marks(u16);

impl Marks {
    /// A stage holds a [`DecisionScope`] to open.
    const SCOPED: u16 = 1;
    /// The picker leaves the placement to the binding's sessions assignment
    /// ([`Picker::by_assignment`]).
    const BY_ASSIGNMENT: u16 = 1 << 1;
    /// The picker draws first and every other stage answers row by row
    /// ([`draws_first`]): a pool of [`DRAW_FROM`] rows or more is decided by
    /// the rows drawn. It asks nothing before the profile's run.
    const DRAWS_FIRST: u16 = 1 << 2;
    /// The rows a host named join the rows the picker draws
    /// ([`Picker::joined_by_named`]): such a profile draws for a request
    /// that names rows in the column's place, and for no other. Set only
    /// beside [`Self::DRAWS_FIRST`].
    const JOINED: u16 = 1 << 3;
    /// The profile's leading stages are ranks ([`Order`]) and nothing else
    /// is asked before its run: a plain decision over a pool of
    /// [`RANK_FROM`] rows or more, each said to hold a report
    /// ([`PoolFacts::reported`]), asks its stages by rank ([`run_ranked`]).
    /// On a profile that draws for a request that names rows
    /// ([`Self::JOINED`]) that request is the draw's, and every other is
    /// the ranks'. Never set on a profile that draws for every request:
    /// the draw is asked first there.
    const RANKED: u16 = 1 << 4;
    /// The plain ring alone places a keyed request by its session key:
    /// under a binding that steers, such a request keeps its key's worker
    /// over the overload veto ([`Engine::keep`]), where the kept key's test
    /// passes it ([`vetoed::keeps_its_key`]). The validator sets it for
    /// that one shape of profile, no stage does ([`Self::with_key`]).
    const KEYED: u16 = 1 << 5;
    /// The scope of a profile under the key's mark: its [`Self::SCOPED`]
    /// word stands here. A decision's opening then asks the scope's and the
    /// assignment's marks as it does without a key, and the key's arm opens
    /// the scope.
    const KEYED_SCOPE: u16 = 1 << 6;
    /// The picker's unit is the engine ([`Picker::unit`]): a pool whose host
    /// lists its engines ([`PoolFacts::engines`]) is decided over them
    /// ([`run_engines`]), at any size. Set only beside
    /// [`Self::DRAWS_FIRST`]: the rows such a decision reads are handed to
    /// the stages as a draw's.
    const ENGINES: u16 = 1 << 7;
    /// The profile's one scorer places a request by itself
    /// ([`Scorer::own`]) and has a weight above zero whose share of itself
    /// is finite ([`WeightedScorer::sole_share`]), every filter answers
    /// row by row and the picker takes the best scores of every row
    /// ([`Picker::takes_best`]): a picker that narrows first does not
    /// follow the scorer's order, a weight that orders nothing leaves the
    /// rows to the picker's draw, a weight too small to invert gives the
    /// whole pass no score to order by, and the profile runs. Under the
    /// mark a request that carries a session key, over a pool whose host
    /// lists its engines, is the scorer's to place ([`run_owned`]).
    const OWNS: u16 = 1 << 8;
    /// A stage may refuse a request by a report of leases
    /// ([`Filter::refuses_by_reports`]). The engine's run reads nothing of
    /// it: the decision's host does, by its one lookup
    /// ([`Engine::reads_for`]).
    const BY_REPORTS: u16 = 1 << 9;
    /// A scorer places a request at the place stated for its routing
    /// value's own text on a lent ring ([`Scorer::places_by_key_text`]).
    /// The engine's run reads nothing of it: what makes the request does,
    /// by its one lookup ([`Engine::states_key_place`]).
    const KEY_PLACE: u16 = 1 << 10;
    /// A scorer reads a ring ([`Needs::ring`](crate::document::spec::Needs::ring))
    /// and places by another text than the routing value's own: the place
    /// stated for that text is not its place. Read by the same lookup.
    const OTHER_PLACE: u16 = 1 << 11;
    /// The profile draws for every request and its picker keeps an order
    /// it takes turns of ([`Picker::takes_turns`]). The row its draw lands
    /// on is its rule's answer, where another picker's draw is a faster
    /// way to the whole pass's: so the draw is asked under the draw's
    /// floor too, over a pool that lends its own word
    /// ([`PoolFacts::word`]), and of a session its rule leaves to the
    /// profile ([`Engine::pool_turn`]). One read tells a decision so,
    /// where the picker's own answer is a call.
    const TURNS: u16 = 1 << 12;

    fn of(profile: &Profile, order: Order) -> Self {
        let scoped = if profile.scoped() { Self::SCOPED } else { 0 };
        let by_assignment = if profile.picker.by_assignment() {
            Self::BY_ASSIGNMENT
        } else {
            0
        };
        let draws_first = if draws_first(profile) {
            Self::DRAWS_FIRST
        } else {
            0
        };
        let joined = if draws_first != 0 && profile.picker.joined_by_named() {
            Self::JOINED
        } else {
            0
        };
        let draws_always = draws_first != 0 && joined == 0;
        // The picker says it itself: a name can be answered by a host's
        // picker that reads every score. A profile that draws for every
        // request holds no rank mark: the draw is asked first there, and a
        // decision knows the profile by one compare of its marks.
        let front = profile.picker.front_wise() && !draws_always;
        let ranked = if (scoped | by_assignment) == 0 && order.gains() && front {
            Self::RANKED
        } else {
            0
        };
        let engines = if draws_first != 0 && profile.picker.unit() == DrawUnit::Engine {
            Self::ENGINES
        } else {
            0
        };
        let alone = match &profile.scorers[..] {
            // The share, not the weight: a weight without end has none,
            // and neither has one too small to invert.
            [stage] => stage.weight > 0.0 && stage.sole_share().is_finite() && stage.scorer.owns(),
            _ => false,
        };
        let row_wise = profile.filters.iter().all(|f| f.row_wise());
        let takes_best = profile.picker.takes_best();
        let owns = if (scoped | by_assignment) == 0 && alone && row_wise && takes_best {
            Self::OWNS
        } else {
            0
        };
        let by_reports = if profile.refuses_by_reports() {
            Self::BY_REPORTS
        } else {
            0
        };
        let by_key_text = |stage: &WeightedScorer| stage.scorer.places_by_key_text();
        let key_place = if profile.scorers.iter().any(by_key_text) {
            Self::KEY_PLACE
        } else {
            0
        };
        let by_other_text = |stage: &WeightedScorer| {
            stage.scorer.needs().ring && !stage.scorer.places_by_key_text()
        };
        let other_place = if profile.scorers.iter().any(by_other_text) {
            Self::OTHER_PLACE
        } else {
            0
        };
        let turns = if draws_always && profile.picker.takes_turns() {
            Self::TURNS
        } else {
            0
        };
        // The three marks no run of the engine reads.
        let for_hosts = by_reports | key_place | other_place;
        // What says when the profile draws, and how.
        let draws = draws_first | joined | turns;
        Self(scoped | by_assignment | draws | ranked | engines | owns | for_hosts)
    }

    /// A profile whose requests take their list's own turn first.
    const fn turns(self) -> bool {
        self.0 & Self::TURNS != 0
    }

    /// Whether no stage asks anything before the profile's run: no scope
    /// to open, no assignment that places, no key that lifts the veto.
    const fn asks_nothing_first(self) -> bool {
        self.0 & (Self::SCOPED | Self::BY_ASSIGNMENT | Self::KEYED) == 0
    }

    /// A scope to open, under the key's mark or under none.
    const fn scoped(self) -> bool {
        self.0 & (Self::SCOPED | Self::KEYED_SCOPE) != 0
    }

    /// A stage that may refuse by a report of leases.
    const fn by_reports(self) -> bool {
        self.0 & Self::BY_REPORTS != 0
    }

    /// A scorer that places by the place stated for a key's own text.
    const fn key_place(self) -> bool {
        self.0 & Self::KEY_PLACE != 0
    }

    /// A ring scorer that places by another text than the key's own.
    const fn other_place(self) -> bool {
        self.0 & Self::OTHER_PLACE != 0
    }

    /// Whether a profile of these marks reads a lent ring by another kind
    /// of place than a profile of `live`'s: one of the two places by the
    /// routing value's own text and the other reads the ring by another
    /// text. The place a request states is the live profile's kind, so it
    /// is no place of such a profile. A profile that reads no ring is
    /// apart from none.
    const fn ring_apart(self, live: Self) -> bool {
        let form = Self::KEY_PLACE | Self::OTHER_PLACE;
        let (own, live) = (self.0 & form, live.0 & form);
        own != 0 && live != 0 && own != live
    }

    /// A scope to open and no key's mark: the opening's first arm.
    const fn opens_first(self) -> bool {
        self.0 & Self::SCOPED != 0
    }

    /// What may stand ahead of the profile's whole pass in a plain
    /// decision, its marks in one masked read: the draw's two and the
    /// ranks' ([`Self::RANKED`]). Of the draw's, [`Self::DRAWS_FIRST`]
    /// stands alone where the profile draws for every request, and beside
    /// [`Self::JOINED`] where it draws for a request that names rows. A
    /// decision asks once, so a profile that neither draws nor ranks pays
    /// the one compare it paid before the third mark, and one that draws
    /// for every request reads `DRAWS_FIRST` alone: it holds no rank mark.
    const fn ahead(self) -> u16 {
        self.0 & (Self::DRAWS_FIRST | Self::JOINED | Self::RANKED)
    }

    const fn ranked(self) -> bool {
        self.0 & Self::RANKED != 0
    }

    const fn engines(self) -> bool {
        self.0 & Self::ENGINES != 0
    }

    const fn owns(self) -> bool {
        self.0 & Self::OWNS != 0
    }

    /// What a plain decision asks of the marks in front of the profile's
    /// whole pass, in one masked read: what may stand ahead of it
    /// ([`Self::ahead`]), whether the picker's unit is the engine and
    /// whether the scorer places a request by itself. A profile that asks
    /// none of it pays the one compare it paid for the draw and the ranks.
    const fn first(self) -> u16 {
        self.0 & (Self::DRAWS_FIRST | Self::JOINED | Self::RANKED | Self::ENGINES | Self::OWNS)
    }

    const fn by_assignment(self) -> bool {
        self.0 & Self::BY_ASSIGNMENT != 0
    }

    /// The scope the key's arm opens.
    const fn keyed_scope(self) -> bool {
        self.0 & Self::KEYED_SCOPE != 0
    }

    const fn keyed(self) -> bool {
        self.0 & Self::KEYED != 0
    }

    /// The marks of a profile the plain ring alone places by the session key.
    /// A picker that leaves the placement to its rule asks no stage where a
    /// key goes, so its marks stand as they are.
    const fn with_key(self) -> Self {
        if self.by_assignment() {
            return self;
        }
        let scope = if self.0 & Self::SCOPED != 0 {
            Self::KEYED_SCOPE
        } else {
            0
        };
        Self((self.0 & !Self::SCOPED) | Self::KEYED | scope)
    }
}

/// The assignment that places a request: a profile whose picker leaves the
/// placement to the rule has no stages to delegate to, and reads the fewest
/// requests in flight under `delegate`.
const fn placed_by(assignment: Assignment, by_assignment: bool) -> Assignment {
    match assignment {
        Assignment::Delegate if by_assignment => Assignment::MinLoad,
        other => other,
    }
}

/// A binding as the engine holds it: the binding as the spec declared it,
/// which is what [`Engine::binding`] answers, and beside it what the build
/// derived from it for the engine's own decisions.
#[derive(Debug)]
struct Entry {
    binding: Binding,
    /// The assignment that places a request of the binding ([`placed_by`]):
    /// the rule's own, or `delegate` where no rule stands, as the binding's
    /// profile reads it under the single plan. Written once when the engine
    /// is built, so no decision asks and the binding keeps the rule as
    /// declared.
    placing: Assignment,
    /// What a decision of the binding does when the overload veto leaves it
    /// no worker ([`AllVetoed`]): the document's word, or the default. A
    /// decision copies it into its set ([`Set`]), and the filter half reads
    /// it where a hard filter left nobody and nowhere else.
    vetoed: AllVetoed,
    /// Whether a side of the binding's pair may be placed by its sessions
    /// rule or by the assignment a picker leaves the placement to
    /// ([`Engine::places_side`]): a rule stands, or a side's picker is such
    /// a picker. Written once when the engine is built, and read by a
    /// pair's decision alone. A pair under no rule whose pickers pick
    /// tests this one byte a decision and runs its two profiles as it did.
    sides_placed: bool,
}

/// The sides of a pair that run under the pair's binding whatever binding
/// covers their phase first: the decode side of every pair, and the encode
/// side of an `epd` pair. The prefill side is the lookup's own.
const PAIR_SIDES: [Phase; 2] = [Phase::Decode, Phase::Encode];

/// Every phase, in the order [`Engine::phase_needs`] keeps them.
const PHASES: [Phase; 4] = [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode];

/// A phase's slot in [`Engine::phase_needs`] (and in a plugin's open
/// decisions, one per side of a pair): total, so no read falls back to
/// another phase's; [`PHASES`] lists the phases in this order.
pub(crate) const fn slot(phase: Phase) -> usize {
    match phase {
        Phase::Single => 0,
        Phase::Prefill => 1,
        Phase::Decode => 2,
        Phase::Encode => 3,
    }
}

// `PHASES` and `slot` agree on every phase's slot, or the crate does not build.
const _: () = {
    let mut i = 0;
    while i < PHASES.len() {
        assert!(
            slot(PHASES[i]) == i,
            "PHASES and slot disagree on a phase's slot"
        );
        i += 1;
    }
};

/// What a binding's sessions rule reads beyond its profile's stages: the
/// key under the view's headers, and the per-key counts its assignment and
/// cap compare; the validator's summary, per binding.
fn session_needs(binding: &Binding) -> Needs {
    binding
        .sessions
        .as_ref()
        .map_or_else(Needs::default, |rule| Needs {
            headers: true,
            active_sessions: rule.assignment == Assignment::MinGroup,
            session_in_flight: rule.inflight_cap > 0,
            ..Needs::default()
        })
}

/// The shortest sweep interval among the bindings' sessions rules, the
/// session table's cadence; `0` when no rule sweeps.
fn shortest_sweep(bindings: &[Binding]) -> u64 {
    bindings
        .iter()
        .filter_map(|b| b.sessions.as_ref())
        .map(|rule| rule.sweep_micros)
        .filter(|sweep| *sweep > 0)
        .min()
        .unwrap_or(0)
}

/// The shards of the session table of an engine of `bindings`
/// ([`SessionTable::with_shards`]): four a core of the host where a binding
/// has a sessions rule, since every keyed decision takes its key's shard's
/// mutex, and one where no rule looks a key up.
fn session_shards(bindings: &[Binding]) -> usize {
    if bindings.iter().any(|binding| binding.sessions.is_some()) {
        sticky_sessions::shards_for_this_host()
    } else {
        1
    }
}

/// How long a worker that left keeps its pins in the session table
/// ([`SessionTable::left`]): the longest TTL among the bindings' sessions
/// rules, a rule with none counting as [`ABSENT_WORKER_PIN_TTL_MICROS`], which
/// is the time too where no binding has a rule.
///
/// [`ABSENT_WORKER_PIN_TTL_MICROS`]: crate::engine::sticky_sessions::ABSENT_WORKER_PIN_TTL_MICROS
fn absence(bindings: &[Binding]) -> u64 {
    let stated = sticky_sessions::ABSENT_WORKER_PIN_TTL_MICROS;
    let rules = bindings.iter().filter_map(|b| b.sessions.as_ref());
    let longest = rules.map(|rule| rule.ttl_micros.unwrap_or(stated));
    longest.max().unwrap_or(stated)
}

/// The stages a sessions rule places a key by when its binding's profile does
/// not decide (every assignment but `delegate`).
#[derive(Debug, Default)]
struct AssignmentStages {
    random: Random,
    least_requests: LeastRequests,
    fewest_sessions: FewestSessions,
    best: Best,
}

/// The two pools of a pair decision: the open prefills and the decodes, each
/// row carrying its pair class; the pair table, which says which classes
/// pair; and beside each pool its affinity column, by the candidate's index,
/// empty for a side nothing reads it on, and its signal cells, none for a
/// side whose stages read no signal.
#[derive(Clone, Copy, Default)]
pub struct PairPools<'a> {
    pub prefills: &'a [Candidate],
    pub decodes: &'a [Candidate],
    /// The prefills' affinity facts: what the prefill side's stages find in
    /// [`RequestFacts::affinity`].
    pub prefill_affinity: &'a [AffinityFacts],
    /// The decodes' affinity facts, the same for the decode side's stages.
    pub decode_affinity: &'a [AffinityFacts],
    /// The prefills' signal cells: what the prefill side's stages find in
    /// [`RequestFacts::signal_cells`].
    pub prefill_signal_cells: Option<&'a SignalCells>,
    /// The decodes' signal cells, the same for the decode side's stages.
    pub decode_signal_cells: Option<&'a SignalCells>,
    /// The decode classes each prefill class pairs with. Left at its
    /// default, classes pair by their number.
    pub table: PairTable<'a>,
    /// The partners each prefill class is paired with first. Left at its
    /// default, no partner goes first.
    pub first: FirstPartners<'a>,
}

/// Which pair classes pair: for each prefill class, the decode classes it
/// pairs with. A worker stands once in its pool and carries one class
/// ([`Candidate::pair_class`]): prefills that pair with the same decodes
/// share one, and so do decodes that pair with the same prefills. The table
/// names classes, never rows or positions, so one table serves every pool
/// narrowed from the same workers.
///
/// The default is no table: a prefill class then pairs with the decode
/// class of its own number, as under a host that gives a decode the class
/// of the prefills it partners.
///
/// A host builds a table ([`PairTable::new`]) and can ask once whether it
/// is one the engine honours whole ([`PairTable::validate`]). How a decision
/// reads it is the engine's own.
#[derive(Clone, Copy, Debug, Default)]
pub struct PairTable<'a> {
    /// Where each prefill class's list ends in `classes`.
    ends: &'a [u32],
    /// The lists, one after another.
    classes: &'a [u32],
}

impl<'a> PairTable<'a> {
    /// A table over the host's two slices. `classes` holds the lists one
    /// after another, prefill class 1's first, and `ends[k - 1]` is where
    /// prefill class `k`'s list ends in it. A list names decode classes in
    /// ascending order.
    ///
    /// A host numbers its decode classes densely from 1, each named by some
    /// list, so a list's classes are the numbers from 1 to the count of the
    /// lists' entries. What a list names outside them is no class of the
    /// table, and no decode pairs through it: class 0, which is outside any
    /// pairing, and a number above the count. A decision keeps one bit a
    /// class number up to that count, never more.
    ///
    /// Nothing is checked here, so a decision of a release build pays for no
    /// check: a table that fails [`PairTable::validate`] costs it the pairs the
    /// table misnames and a set of at most the table's own size, never the
    /// process.
    pub const fn new(ends: &'a [u32], classes: &'a [u32]) -> Self {
        Self { ends, classes }
    }

    /// No table was handed: classes pair by their number.
    pub(crate) const fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// The decode classes prefill class `class` pairs with, ascending: its
    /// list under a table, its own number under none. Every read of a pair
    /// decision asks here, but for the width's walk: it reads each list's
    /// last entry by the same cut ([`Self::cut`]), and asks here list by
    /// list once a last entry stands above the count. Class 0 pairs with
    /// nothing, as does a class the table holds no list for, and a list is
    /// read up to the count of the lists' entries: a number above it is no
    /// class of the table. The class is lent, so that under no table the
    /// list is the number itself.
    pub(crate) fn of<'c>(&self, class: &'c u32) -> &'c [u32]
    where
        'a: 'c,
    {
        if *class == 0 {
            return &[];
        }
        if self.is_empty() {
            return std::slice::from_ref(class);
        }
        let list = self.list(*class as usize - 1);
        // A list ascends, so the numbers above the count are its end.
        let held = |listed: &u32| *listed as usize <= self.classes.len();
        match list.last() {
            Some(last) if !held(last) => &list[..list.partition_point(held)],
            _ => list,
        }
    }

    /// The list of the prefill class at `at`, as the host handed it; empty
    /// for a class the table holds no list for.
    fn list(&self, at: usize) -> &'a [u32] {
        let Some(&end) = self.ends.get(at) else {
            return &[];
        };
        let start = at.checked_sub(1).map_or(0, |before| self.ends[before]);
        self.cut(start as usize, end as usize)
    }

    /// The entries from `start` up to `end`: the one cut of a list out of
    /// the lists' entries, empty where the two ends do not hold.
    /// [`Self::list`] cuts by a class's own two ends, and the width's walk
    /// by the end it passed last.
    fn cut(&self, start: usize, end: usize) -> &'a [u32] {
        self.classes.get(start..end).unwrap_or(&[])
    }

    /// The table's prefill classes.
    fn prefill_classes(&self) -> std::ops::RangeInclusive<u32> {
        1..=self.ends.len() as u32
    }

    /// The width of a set of bits over the table's decode classes: one past
    /// the largest class a list names, which is at most the count of the
    /// lists' entries ([`PairTable::of`]). A table costs a decision a set
    /// of its own size, whatever numbers it names. Lists ascend, so a
    /// list's last entry is its largest; the width is capped at the count
    /// all the same, so the bound stands for a list out of order too.
    ///
    /// One walk of the lists' ends reads each list's last entry by the cut
    /// a list's own read makes ([`Self::cut`]). The walk stays out of line
    /// and calls nothing on its way, so the routine that asks is compiled
    /// as it was without a table's walk in it. The first last entry above
    /// the count ends the walk: [`Self::width_by_lists`] answers.
    #[inline(never)]
    fn width(&self) -> usize {
        let count = self.classes.len();
        let (mut largest, mut start) = (0u32, 0usize);
        for &end in self.ends {
            let end = end as usize;
            if let Some(&last) = self.cut(start, end).last() {
                if last as usize > count {
                    return self.width_by_lists();
                }
                largest = largest.max(last);
            }
            start = end;
        }
        Self::capped(largest, count)
    }

    /// The width of a table with a list's last entry above the count of
    /// the entries: each list's own read says where it ends
    /// ([`PairTable::of`]), list by list. Cold and out of line: a table a
    /// host numbers from one up never comes here.
    #[cold]
    #[inline(never)]
    fn width_by_lists(&self) -> usize {
        let largest = self
            .prefill_classes()
            .filter_map(|class| self.of(&class).last().copied())
            .max()
            .unwrap_or(0);
        Self::capped(largest, self.classes.len())
    }

    /// One past `largest`, and at most one past `count`: the width's cap,
    /// apart from the lists so that a case asks it. A list's read ends at
    /// the count ([`PairTable::of`]), so no table hands the width a larger
    /// number while the library's search answers as it does today.
    fn capped(largest: u32, count: usize) -> usize {
        (largest as usize).min(count) + 1
    }

    /// Whether this is a table the engine honours whole, and the first
    /// clause it fails when it is not: the ends ascend and the last stands
    /// where the lists end, and each list names its classes in ascending
    /// order, none of them class 0 and none above the count of the lists'
    /// entries. The default, no table, is valid.
    ///
    /// One question, two askers: a host asks it once where it builds a
    /// table, in every build, and a debug build asks it where a decision
    /// takes one.
    pub fn validate(&self) -> Result<(), PairTableError> {
        self.laid_out(true)
    }

    /// The clauses of [`Self::validate`], the last of them only where
    /// `counted`: lists that name some of a table's classes hold numbers
    /// above their own count ([`FirstPartners::within`]).
    fn laid_out(&self, counted: bool) -> Result<(), PairTableError> {
        if !self.ends.windows(2).all(|pair| pair[0] <= pair[1]) {
            return Err(PairTableError::EndsDescend);
        }
        // Every entry stands in a list, so the count a read holds a list
        // to is the lists' own.
        let entries = self.classes.len();
        let end = self.ends.last().map_or(0, |&end| end as usize);
        if end > entries {
            return Err(PairTableError::EndPastLists);
        }
        if end < entries {
            return Err(PairTableError::EntriesPastEnds);
        }
        for list in (0..self.ends.len()).map(|at| self.list(at)) {
            if !list.windows(2).all(|pair| pair[0] < pair[1]) {
                return Err(PairTableError::ListOutOfOrder);
            }
            if list.first() == Some(&0) {
                return Err(PairTableError::ClassZero);
            }
            if counted && list.last().is_some_and(|&last| last as usize > entries) {
                return Err(PairTableError::NumberAboveCount);
            }
        }
        Ok(())
    }

    /// The prefill classes with a partner, ascending, into `classes`. `open`
    /// is the class of each decode the decode profile's hard filters keep.
    /// Under a table the question is asked once a prefill class, each
    /// stopping at the first open class it lists, and of no class when no
    /// decode is open. Under none the open classes are the answer: a class
    /// pairs with its own number.
    fn partnered(
        &self,
        open: impl Iterator<Item = u32>,
        classes: &mut Vec<u32>,
        marks: &mut ClassBits,
    ) {
        classes.clear();
        if self.is_empty() {
            classes.extend(open);
            classes.sort_unstable();
            classes.dedup();
            return;
        }
        // A number above the table's width is no class of the table.
        marks.reset(self.width());
        let mut any = false;
        for class in open {
            any |= marks.insert(class);
        }
        if any {
            classes.extend(
                self.prefill_classes()
                    .filter(|class| self.of(class).iter().any(|&listed| marks.has(listed))),
            );
        }
    }

    /// Turns the prefill classes in `classes` into the decode classes they
    /// pair with, both ascending. Under no table nothing moves: a class
    /// pairs with its own number.
    fn listed(&self, classes: &mut Vec<u32>, marks: &mut ClassBits) {
        if self.is_empty() {
            return;
        }
        marks.reset(self.width());
        for class in classes.iter() {
            for &listed in self.of(class) {
                marks.insert(listed);
            }
        }
        classes.clear();
        classes.extend(marks.classes());
    }
}

/// The clause of [`PairTable::validate`] a host's table fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PairTableError {
    #[error("an end stands before the end of the list before it")]
    EndsDescend,
    #[error("an end stands past the lists")]
    EndPastLists,
    #[error("the lists hold entries past the last end")]
    EntriesPastEnds,
    #[error("a list names its classes out of ascending order")]
    ListOutOfOrder,
    #[error("a list names class 0")]
    ClassZero,
    #[error("a list names a number above the count of the lists' entries")]
    NumberAboveCount,
}

/// The partners each prefill class is paired with first: for each prefill
/// class, a part of the decode classes its list in the pair table names
/// ([`PairTable`]). While one of them is open, the pair is made among the
/// prefills that have one open and their first partners, and a request
/// waits where such a prefill is held back by its admission limit alone.
/// Every other pair is made only while no prefill has a first partner
/// open ([`Engine::decide_pair`]). The default is none: every partner
/// stands alike.
///
/// The lists name classes of the pair table and not all of them, so they
/// are no table of their own. A host asks once, where it builds the two,
/// whether they stand within its table ([`FirstPartners::within`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct FirstPartners<'a> {
    /// The lists in a table's storage, read as the host handed them.
    lists: PairTable<'a>,
}

impl<'a> FirstPartners<'a> {
    /// The first partners over the host's two slices, laid out as a pair
    /// table's are ([`PairTable::new`]): `classes` holds the lists one
    /// after another, prefill class 1's first, and `ends[k - 1]` is where
    /// prefill class `k`'s list ends in it. A list names decode classes in
    /// ascending order. Nothing is checked here.
    pub const fn new(ends: &'a [u32], classes: &'a [u32]) -> Self {
        Self {
            lists: PairTable::new(ends, classes),
        }
    }

    /// The first partners of prefill class `class`, as the host handed
    /// them: none for class 0 and for a class the lists do not reach.
    pub(crate) fn of(&self, class: u32) -> &'a [u32] {
        match class.checked_sub(1) {
            Some(at) => self.lists.list(at as usize),
            None => &[],
        }
    }

    /// The prefill classes with an open first partner, ascending, into
    /// `classes`, and whether there is one. `marks` holds the open decode
    /// classes as `table`'s own pass left them ([`PairTable::partnered`]).
    /// Under no table that pass marks nothing, and no first partner is
    /// read.
    fn open(&self, table: &PairTable<'_>, marks: &ClassBits, classes: &mut Vec<u32>) -> bool {
        if table.is_empty() {
            return false;
        }
        classes.clear();
        let open = |class: &u32| self.of(*class).iter().any(|&listed| marks.has(listed));
        classes.extend(self.lists.prefill_classes().filter(open));
        !classes.is_empty()
    }

    /// Whether these are first partners of `table`, and the first clause
    /// they fail when they are not. The lists are laid out as a table's
    /// are, by the table's own clauses but for its count
    /// ([`PairTable::validate`]): the ends ascend and the last stands where
    /// the lists end, and each list names its classes in ascending order,
    /// none of them class 0. Then each class a list names is a partner of
    /// its prefill class in `table`. The default, none, stands within
    /// every table.
    ///
    /// A host asks it once where it builds the two, and a debug build asks
    /// it where a decision takes them.
    pub fn within(&self, table: &PairTable<'_>) -> Result<(), FirstPartnersError> {
        self.lists
            .laid_out(false)
            .map_err(FirstPartnersError::Lists)?;
        let partners = |class: u32| {
            let listed = table.of(&class);
            let partner = |named: &u32| listed.binary_search(named).is_ok();
            self.of(class).iter().all(partner)
        };
        if self.lists.prefill_classes().all(partners) {
            Ok(())
        } else {
            Err(FirstPartnersError::NoPartner)
        }
    }
}

/// The clause of [`FirstPartners::within`] a host's first partners fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FirstPartnersError {
    #[error("the first partners' lists: {0}")]
    Lists(PairTableError),
    #[error("a first partner is no partner of its prefill class in the pair table")]
    NoPartner,
}

/// One side of a pair decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairSide {
    Prefill,
    Decode,
}

impl PairSide {
    /// Both sides, the prefill first: the order a pair decides them in.
    pub const BOTH: [Self; 2] = [Self::Prefill, Self::Decode];

    /// The side's place in a table of the two ([`Self::BOTH`]).
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Prefill => 0,
            Self::Decode => 1,
        }
    }

    /// The side's label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

impl<'a> PairPools<'a> {
    /// The two pools with nothing beside them: no affinity column, no
    /// signal cells and no pair table, so classes pair by their number.
    /// A caller writes what it holds beside its pools over it:
    /// `PairPools { table, ..PairPools::of(prefills, decodes) }`.
    #[must_use]
    pub fn of(prefills: &'a [Candidate], decodes: &'a [Candidate]) -> Self {
        Self {
            prefills,
            decodes,
            ..Self::default()
        }
    }

    /// The decode classes a prefill of `class` is paired among: its first
    /// partners where the pair is made among those (`firsts`), else every
    /// class its list in the table names.
    fn partners<'c>(&self, class: &'c u32, firsts: bool) -> &'c [u32]
    where
        'a: 'c,
    {
        if firsts {
            self.first.of(*class)
        } else {
            self.table.of(class)
        }
    }

    /// Whether the pair `live` names was made among first partners: its
    /// decode is a first partner of its prefill's class `class`. A side's
    /// shadow is shown the partners its live side was shown.
    fn made_first(&self, class: u32, live: &Pick) -> bool {
        let decode = live.targets.first().and_then(|target| target.pair);
        let decode = decode.and_then(|pair| self.decodes.get(pair.decode_position as usize));
        decode.is_some_and(|decode| classed(self.first.of(class), decode.pair_class))
    }

    /// One side of the pair as its stages read it: the side's pool, and the
    /// request's facts with the side's phase, its draw sequence and its own
    /// pool's affinity column and signal cells, and with no rows named:
    /// those index neither pool. `None` when that column was not filled
    /// for that pool ([`column_fits`]). The one place a pair's column and
    /// cells meet their pool, so no call site hands one side's with the
    /// other side's candidates.
    fn side(
        self,
        side: PairSide,
        facts: &RequestFacts<'a>,
    ) -> Option<(&'a [Candidate], RequestFacts<'a>)> {
        let (pool, affinity, signal_cells, phase, seed) = match side {
            PairSide::Prefill => (
                self.prefills,
                self.prefill_affinity,
                self.prefill_signal_cells,
                Phase::Prefill,
                facts.seed,
            ),
            // The decode side draws its own sequence, derived from the
            // request's seed: one seed on both sides made the decode picker
            // replay the prefill picker's draws, so the decode chosen
            // followed the prefill chosen (random over four prefills of four
            // decodes each reached four of the sixteen). The decision still
            // replays from the trace's seed.
            PairSide::Decode => (
                self.decodes,
                self.decode_affinity,
                self.decode_signal_cells,
                Phase::Decode,
                Draw::new(facts.seed).next_u64(),
            ),
        };
        debug_assert_indexed(pool);
        debug_assert_cells(signal_cells, pool);
        column_fits(affinity, pool).then(|| {
            let facts = RequestFacts {
                phase,
                seed,
                affinity,
                named: None,
                // A pin and lent counts are positions in one pool: neither
                // side takes them.
                pinned_at: 0,
                lent_in_flight: None,
                // The side's own cells: the request's would be read by
                // another pool's positions.
                signal_cells,
                ..*facts
            };
            (pool, facts)
        })
    }
}

/// The stage a decision names when it is refused for its affinity column:
/// what a host counts its sheds under.
const AFFINITY_COLUMN: &str = "affinity-column";

/// Whether the request carries a column of its own or names rows. A pair
/// reads neither: its columns ride its two pools ([`PairPools`]), and
/// either on the request names neither pool.
fn carries_a_column(facts: &RequestFacts<'_>) -> bool {
    !facts.affinity.is_empty() || facts.named.is_some()
}

/// The stage a decision names when a profile that shows its rows to a
/// plugin was handed rows that carry a mask or a pin beside them.
const WRITTEN_ROWS: &str = "written-rows";

/// The decline of a decision whose rows were not written for the plugin
/// they are shown to, under the profile that declines. Cold and out of
/// line: an entry's path past the refusal keeps its weight, and the passes
/// it calls stay inlined. Built in place, the refusal cost a pair decision
/// under no mask 26 instructions.
#[cold]
#[inline(never)]
fn rows_not_written(bare: Trace, profile: u16) -> Decision {
    let trace = Trace { profile, ..bare };
    shed(ShedReason::PolicyDeclined, WRITTEN_ROWS, trace)
}

/// Whether an affinity column stands for `pool`: empty, or one entry a
/// candidate, in the candidates' order. Asked once where a decision takes
/// its inputs and never at a read. A column of any other length was filled
/// for another pool: nothing held is a real answer (a cold worker), so
/// reading such a column as zeros would turn a host's mistake into a routing
/// change no case sees.
///
/// An empty column is of another kind and is taken, under a profile that
/// declared the need as under one that did not: it is the host's word that
/// it holds nothing for any candidate. The engine checks the column's shape,
/// never its presence.
fn column_fits(column: &[AffinityFacts], pool: &[Candidate]) -> bool {
    column.is_empty() || column.len() == pool.len()
}

/// A debug build asserts that the rows a host named in the column's place
/// ([`AffinityRows`]) stand for `pool`, once where a
/// decision takes its inputs: both lists ascending and under the pool's
/// size, beside an empty column, and no mean in flight, which is the
/// engine's to set behind a draw. A list out of order hides a holder from
/// the stages' search, a column beside the rows is read in their place,
/// and a mean the host wrote is read for the mean of the rows wherever
/// the pool is read whole. A release build does not check what the host
/// derived: the test would ride every decision, a keyed one among them.
fn debug_assert_named(facts: &RequestFacts<'_>, pool: &[Candidate]) {
    let under = |row: u32| (row as usize) < pool.len();
    debug_assert!(
        facts.named.is_none_or(|named| {
            facts.affinity.is_empty()
                && named.mean_in_flight.is_none()
                && named.holders.is_sorted_by(|a, b| a.row < b.row)
                && named.holders.last().is_none_or(|holder| under(holder.row))
                && named.warming.is_sorted_by(|a, b| a < b)
                && named.warming.last().is_none_or(|&row| under(row))
        }),
        "rows named in the column's place stand ascending and under the pool's size, beside an empty column and no mean in flight"
    );
}

/// A debug build asserts that every row's `position` is its index in `pool`,
/// once where a decision takes its inputs. A row is found by where it stands
/// in the pool the decision was given, and the readers that still take a
/// row's position for its index (the plugin's filter adapter, the pair
/// plan's lookup of a chosen prefill and its row-by-row filter passes) would
/// answer for another row. The assertion leaves with the last of those
/// readers.
fn debug_assert_indexed(pool: &[Candidate]) {
    debug_assert!(
        pool.iter()
            .zip(0u32..)
            .all(|(row, index)| row.position == index),
        "a row's position is its index in the pool the decision was given"
    );
}

/// A debug build's check of the signal cells a decision was handed: the
/// cells of the pool's rows, the document's signals a row. A release build
/// tests nothing before the stages run, so a decision that reads no signal
/// pays nothing for the cells: a read outside them answers no value.
fn debug_assert_cells(cells: Option<&SignalCells>, pool: &[Candidate]) {
    debug_assert!(
        cells.is_none_or(|cells| cells.fits(pool.len())),
        "the signal cells are those of the rows the decision was given"
    );
}

/// The test the standing pass of one decision's profile asks of a row
/// ([`Engine::keeps_for`]): the profile's filters, the request's facts and
/// the host's constraints, held for the decision's time.
pub(crate) struct RowTest<'a> {
    /// The filters as one test, as this decision asks them; `None` where
    /// the build did not fold them.
    tests: Option<RowTests>,
    /// A veto of the host stands, and the folded health test asks its mask.
    masked: bool,
    filters: &'a [Box<dyn Filter>],
    facts: &'a RequestFacts<'a>,
    cx: &'a Constraints<'a>,
}

impl RowTest<'_> {
    /// Whether the pass keeps `row`. In its caller's loop: the folded test
    /// is a few bits of the row.
    #[inline(always)]
    pub(crate) fn keeps(&self, row: &Candidate) -> bool {
        match self.tests {
            Some(tests) => {
                tests.keeps(self.facts, row, self.cx)
                    && !(self.masked && self.cx.masks(row.position))
            }
            None => self.each(row),
        }
    }

    /// The test a call a filter, for a profile the build did not fold.
    /// Out of line: a folded profile's loop holds none of it.
    #[inline(never)]
    fn each(&self, row: &Candidate) -> bool {
        let mut filters = self.filters.iter();
        filters.all(|filter| filter.keep(self.facts, row, self.cx))
    }
}

impl Engine {
    /// The engine of `profiles` under `bindings`. `needs` is what no stage
    /// declares: a sessions rule's counts and the view's list of headers.
    /// Every profile's stages are folded in here, so the summary
    /// ([`Self::needs`]) holds a stage's needs however the engine was built.
    /// A binding built in code can shadow with a separate profile without
    /// a decision scope under the single plan. An engine holds at most
    /// 65,535 profiles: the position past them marks a decision made ahead
    /// of every profile ([`Trace::NO_PROFILE`]).
    pub fn new(
        profiles: Vec<Profile>,
        bindings: Vec<Binding>,
        generation: u64,
        needs: Needs,
    ) -> Self {
        debug_assert!(
            profiles.len() <= usize::from(Trace::NO_PROFILE),
            "the position past 65,535 profiles is the mark of no profile"
        );
        // Each profile's needs, asked of its stages once: the summary and
        // the phase answers below fold them.
        let profile_needs: Vec<Needs> = profiles.iter().map(Profile::needs).collect();
        let needs = profile_needs.iter().fold(needs, |all, &own| all.or(own));
        let orders: Vec<Order> = profiles.iter().map(Order::of).collect();
        let marks: Vec<Marks> = profiles
            .iter()
            .zip(&orders)
            .map(|(profile, &order)| Marks::of(profile, order))
            .collect();
        // Per phase, the bindings that cover it in the spec's order; a
        // binding behind a catch-all model glob decides no request of the
        // phase and adds nothing (the gateway's own document binds every
        // model's pair phases before its catch-all for the regular ones).
        let mut phase_needs = [Needs::default(); PHASES.len()];
        let mut keyed = [false; PHASES.len()];
        for phase in PHASES {
            for binding in bindings
                .iter()
                .filter(|b| b.phases.is_empty() || b.phases.contains(&phase))
            {
                let index = profile_index(binding, phase).map(usize::from);
                if let Some(&own) = index.and_then(|i| profile_needs.get(i)) {
                    let shadow = shadow_index(binding, phase)
                        .and_then(|i| profile_needs.get(usize::from(i)))
                        .copied()
                        .unwrap_or_default();
                    // A rule that pins a key, or a profile whose one scorer
                    // places a keyed request by itself on a ring: what a
                    // keyed decision answers for. A ring stage that reads
                    // the pool whole, as under a bounded-load rule, leaves
                    // its phase to the one test.
                    let walks = own.ring
                        && index
                            .and_then(|i| marks.get(i))
                            .is_some_and(|marks| marks.owns());
                    keyed[slot(phase)] |=
                        binding.plan == Plan::Single && (binding.sessions.is_some() || walks);
                    phase_needs[slot(phase)] = phase_needs[slot(phase)]
                        .or(own)
                        .or(shadow)
                        .or(session_needs(binding));
                }
                if binding.model == "*" {
                    break;
                }
            }
        }
        // A pair's decode side, and an `epd` pair's encode side, are decided
        // by the binding the prefill lookup resolves to
        // ([`Self::decide_pair`], [`Self::decide_encode_pair`]), not by the
        // first binding that covers the side's phase: each side's slot also
        // carries that side's profile of every pair binding a prefill
        // lookup can reach. A `pd-pair` binding names no encode side.
        for binding in bindings
            .iter()
            .filter(|b| b.phases.is_empty() || b.phases.contains(&Phase::Prefill))
        {
            if matches!(binding.plan, Plan::PdPair | Plan::Epd { .. }) {
                for side in PAIR_SIDES {
                    if let Some(&own) =
                        profile_index(binding, side).and_then(|i| profile_needs.get(usize::from(i)))
                    {
                        // The side's shadow reads what its profile needs,
                        // beside the live one. An encode side names none.
                        let shadow = shadow_index(binding, side)
                            .and_then(|i| profile_needs.get(usize::from(i)))
                            .copied()
                            .unwrap_or_default();
                        let needs = &mut phase_needs[slot(side)];
                        *needs = needs.or(own).or(shadow);
                    }
                }
            }
            if binding.model == "*" {
                break;
            }
        }
        let sessions = SessionTable::new(shortest_sweep(&bindings))
            .with_absence(absence(&bindings))
            .with_shards(session_shards(&bindings));
        let sessions = Arc::new(sessions);
        let head_places: Vec<HeadPlaces> = profiles.iter().map(HeadPlaces::of).collect();
        let keeping = marks.iter().any(|marks| marks.scoped());
        #[cfg(feature = "in-process")]
        let written_draws = profiles
            .iter()
            .map(|profile| {
                draws_first(profile) && DRAWS_BY_ITS_TEST.contains(&profile.picker.name())
            })
            .collect();
        let row_tests: Vec<Option<RowTests>> = profiles
            .iter()
            .map(|profile| RowTests::of(&profile.filters))
            .collect();
        let named_by: Vec<NamedBy> = profiles
            .iter()
            .zip(&row_tests)
            .map(|(profile, &folded)| profile.named_by(folded))
            .collect();
        let keepers = Keeper::of(&profiles);
        // The engine's own: the session table follows a dispatch where a
        // binding states a sessions rule. At an issue the engine does
        // nothing but tell its stages: the record of dispatched prefixes
        // is its host's to write, where the host decides.
        let own = if bindings.iter().any(|binding| binding.sessions.is_some()) {
            EventSet::NONE.with(EventKind::Dispatched)
        } else {
            EventSet::NONE
        };
        let hears = profiles.iter().fold(own, |hears, profile| {
            let stages = profile
                .scorers
                .iter()
                .map(|weighted| weighted.scorer.hears());
            stages.fold(hears.union(profile.picker.hears()), EventSet::union)
        });
        let shadows = bindings
            .iter()
            .any(|binding| binding.shadow.is_some() || binding.decode_shadow.is_some());
        let mut shadow_placing = Vec::new();
        let mut shadow_rows = Vec::new();
        let mut shadow_sides = Vec::new();
        let mut side_apart = Vec::new();
        let mut live = Vec::new();
        if shadows {
            shadow_placing.reserve_exact(bindings.len());
            live.resize(profiles.len(), false);
            for binding in &bindings {
                for phase in PHASES {
                    if let Some(used) =
                        profile_index(binding, phase).and_then(|i| live.get_mut(usize::from(i)))
                    {
                        *used = true;
                    }
                }
            }
            // A binding's word is of the whole list: whether a live
            // profile reads the record beside it. A pair's binding has
            // the word of its sides.
            let sides = bindings.iter();
            shadow_sides = sides
                .map(|binding| side_shadows(binding, &live, &marks))
                .collect();
            let apart = |(binding, sides): (&Binding, &[Option<u16>; 2])| {
                PairSide::BOTH.map(|side| {
                    sides[side.index()].is_some_and(|shadow| {
                        let (by, needs) = (&named_by[..], &profile_needs[..]);
                        side_reads_another_column(binding, side, shadow, by, needs)
                    })
                })
            };
            side_apart = bindings.iter().zip(&shadow_sides).map(apart).collect();
            let rows = bindings.iter().enumerate().map(|(at, binding)| {
                if shadow_sides[at] != [None; 2] {
                    return ShadowRows::Sides;
                }
                let beside = || live_reader_beside(&bindings, at, &named_by);
                ShadowRows::of(binding, &named_by, &profile_needs, beside)
            });
            shadow_rows = rows.collect();
        }
        // A picker that leaves the placement to its rule has no stages to
        // delegate to: the assignment that places is written beside each
        // binding here, once, so no decision asks. The binding itself keeps
        // the rule as the spec declared it.
        let bindings = bindings
            .into_iter()
            .map(|binding| {
                let by_assignment = binding.plan == Plan::Single
                    && marks
                        .get(usize::from(binding.profile))
                        .is_some_and(|marks| marks.by_assignment());
                let declared = binding
                    .sessions
                    .map_or(Assignment::Delegate, |rule| rule.assignment);
                if shadows {
                    // Bindings built in code must keep the same isolation
                    // as bindings built from a document.
                    let placing = binding.single_shadow().and_then(|i| {
                        let i = usize::from(i);
                        let marks = marks.get(i)?;
                        (!live[i] && !marks.scoped())
                            .then(|| placed_by(declared, marks.by_assignment()))
                    });
                    shadow_placing.push(placing);
                }
                // A side of a pair is placed by the rule, or by the
                // assignment where its picker leaves the placement to one.
                let assigned = |profile: u16| {
                    marks
                        .get(usize::from(profile))
                        .is_some_and(|marks| marks.by_assignment())
                };
                let sides_placed = binding.sessions.is_some()
                    || assigned(binding.profile)
                    || binding.decode_profile.is_some_and(assigned);
                Entry {
                    placing: placed_by(declared, by_assignment),
                    vetoed: AllVetoed::default(),
                    sides_placed,
                    binding,
                }
            })
            .collect();
        Self {
            profiles,
            bindings,
            shadow_placing: shadow_placing.into_boxed_slice(),
            shadow_rows: shadow_rows.into_boxed_slice(),
            shadow_sides: shadow_sides.into_boxed_slice(),
            side_apart: side_apart.into_boxed_slice(),
            generation,
            needs,
            phase_needs,
            keyed,
            sessions,
            assignment: AssignmentStages::default(),
            marks,
            head_places,
            profile_needs,
            named_by,
            row_tests,
            orders,
            #[cfg(feature = "in-process")]
            written_draws,
            seat_flags: (u8::from(keeping) * KEEPS_SLOT_STATE) | (u8::from(shadows) * SHADOWS),
            explain_sample: 0,
            max_report_age_ms: None,
            keepers,
            sent_credits: Vec::new(),
            hears,
        }
    }

    pub(crate) fn limit_report_age(&mut self, max_age_ms: Option<u64>) {
        self.max_report_age_ms = max_age_ms;
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Takes over what the standing engine's counts of sends hold: each
    /// count takes the rows' entries of the standing count that a profile
    /// of the same name made. Profiles that share a count carry it under
    /// the name of the first of them, so a count starts empty where that
    /// name made no standing count. The two engines share those entries
    /// from here on: a decision still running under the standing engine
    /// reads the sends a later one counts, and each count reads them under
    /// its own time limit and report delay.
    ///
    /// [`MemoryStore::apply`](crate::store::MemoryStore::apply) calls it
    /// before it shows the new engine. A host with a store of its own
    /// calls it where it swaps engines, before this engine takes a
    /// decision or a dispatch: what this engine's counts held until then
    /// is let go. A rank the standing count first hears of behind this
    /// call is entered in the standing count alone.
    pub fn succeed(&self, standing: &Engine) {
        for credit in &self.sent_credits {
            if let Ok(at) = standing
                .sent_credits
                .binary_search_by(|prior| prior.profile.cmp(&credit.profile))
            {
                credit.succeed(&standing.sent_credits[at]);
            }
        }
    }

    /// The session table: the pins of the keys it holds, each placed by a
    /// sessions rule.
    pub fn sessions(&self) -> &SessionTable {
        &self.sessions
    }

    /// This engine reads and writes `table` from here on: the table a host
    /// keeps for its life, where it keeps one engine object and gives it
    /// documents. The table takes this document's two times
    /// ([`SessionTable::set_times`]) and keeps its keys, its shards and the
    /// workers it holds as absent. The table this engine was made with is
    /// let go, and a pin it held would go with it: its host calls this
    /// before the engine takes a decision, which a debug build checks.
    #[cfg(feature = "in-process")]
    pub(crate) fn share_sessions(&mut self, table: Arc<SessionTable>) {
        debug_assert!(
            self.sessions.is_empty(),
            "the engine placed a key ahead of the hand-over of its session table"
        );
        table.set_times(self.sessions.sweep_micros(), self.sessions.absence_micros());
        self.sessions = table;
    }

    /// A worker that left membership stands again as `new`: its key was
    /// registered again, and a registration makes a new id. The session
    /// pins of `old` are `new`'s from here, so a key's recovery rules over
    /// them as over two routable pins. A host says it behind the join's
    /// own word ([`Membership::Added`], or [`Membership::Replaced`] where
    /// the row never left), for a registration that is the same worker:
    /// its role, its models and its wires.
    pub fn pins_follow(&self, old: WorkerId, new: WorkerId) {
        self.sessions.rename_worker(old, new);
    }

    /// Workers that left membership do not return as themselves: their
    /// session pins leave now, ahead of the absence's time, in one walk of
    /// the table for them all. A key one of them owned keeps its failover
    /// as its owner, and one with no pin left is placed afresh.
    pub fn pins_forget(&self, workers: &[WorkerId]) {
        self.sessions.forget_workers(workers);
    }

    /// How long a worker that left membership keeps its session pins, in
    /// microseconds of the decision clock: the longest TTL among the
    /// bindings' sessions rules, four hours where a rule has none.
    pub fn pins_absence_micros(&self) -> u64 {
        self.sessions.absence_micros()
    }

    /// What the host must feed before any request exists: every profile's
    /// stages and the view's own list of headers.
    pub fn needs(&self) -> Needs {
        self.needs
    }

    /// What a decision of `phase` for `model` reads: the needs of the stages
    /// of its live and shadow profiles and of the binding's sessions rule,
    /// nothing when no binding matches. The decode answer also carries the
    /// needs of the pair's decode profile where the prefill lookup names a
    /// pair ([`Self::decide_pair`]), and the encode answer those of the
    /// pair's encode profile where that lookup names an `epd` pair
    /// ([`Self::decide_encode_pair`]): both sides run under the pair's
    /// binding, whatever binding covers their phase first. The host reads
    /// it at registration to choose the worker's stream.
    pub fn needs_for(&self, model: &str, phase: Phase) -> Needs {
        self.reads_for(model, phase).0
    }

    /// [`Self::needs_for`], and whether the profile that decision resolves
    /// to shows its rows to a plugin ([`Profile::scoped`]), by one lookup.
    /// A host hands such a decision rows written for it: a plugin's view
    /// carries each row's bits and ages as the row's own fields, and
    /// [`Self::decide`] declines such a profile over rows that carry a mask
    /// or a pin beside them ([`Constraints::stalled`],
    /// [`RequestFacts::pinned_at`]). [`Self::decide_pair`] declines a mask
    /// beside a side that shows its rows: for a decode decision of a model
    /// that also pairs, the word carries the pair's decode side too, as the
    /// needs do, and for an encode decision the encode side of its `epd`
    /// pair. Written rows are the safe side.
    ///
    /// The last word says whether that profile holds a stage that may
    /// refuse a request by a report of leases
    /// ([`Filter::refuses_by_reports`]): a host that applies its gateways'
    /// reports late applies them first for such a decision. It carries a
    /// pair's side as the second word does. A profile kept its
    /// word when the engine was built, so a decision under no such stage
    /// reads nothing more for it.
    pub fn reads_for(&self, model: &str, phase: Phase) -> (Needs, bool, bool) {
        let (own, marks) = self
            .resolve(model, phase)
            .map(|(entry, index, _)| {
                (
                    self.profile_needs[usize::from(index)]
                        .or(self.shadow_needs(&entry.binding, phase))
                        .or(session_needs(&entry.binding)),
                    self.marks[usize::from(index)],
                )
            })
            .unwrap_or_default();
        match phase {
            Phase::Decode | Phase::Encode => {
                let (needs, pair) = self.pair_side(model, phase);
                (
                    own.or(needs),
                    marks.scoped() || pair.scoped(),
                    marks.by_reports() || pair.by_reports(),
                )
            }
            Phase::Single | Phase::Prefill => (own, marks.scoped(), marks.by_reports()),
        }
    }

    /// Whether any profile of this generation holds a stage that keeps
    /// state a decider ([`crate::engine`]): a plugin's kept instance is
    /// its decider's, and every profile's stages hear every lease. Under
    /// such a generation a host that seats its callers by thread decides
    /// each request at its model's decider, so that state has one writer
    /// a model and hears a model's leases whole. It is one read: under a
    /// generation with none a host looks nothing up.
    ///
    /// The fact is read from the mark of a stage that opens a decision's
    /// scope. A stage kind that keeps state a decider and opens none
    /// joins this fact: without it such a host would split that state by
    /// thread, with no error.
    pub fn keeps_slot_state(&self) -> bool {
        self.seat_flags & KEEPS_SLOT_STATE != 0
    }

    #[cfg_attr(
        not(any(feature = "in-process", test)),
        expect(
            dead_code,
            reason = "the one reader is the engine over worker lists, built under `in-process`"
        )
    )]
    pub(crate) fn seat_flags(&self) -> u8 {
        self.seat_flags
    }

    pub(crate) fn set_explain_sample(&mut self, sample: u32) {
        self.explain_sample = sample;
        self.seat_flags =
            (self.seat_flags & !SAMPLES_DECISIONS) | (u8::from(sample != 0) * SAMPLES_DECISIONS);
    }

    pub(crate) fn explain_sample(&self) -> u32 {
        self.explain_sample
    }

    /// The sample of the profile that decides a request of `phase` for
    /// `model` ([`Needs::sample`]), `0` where that profile never draws and
    /// where no binding matches. A host that names a decision's rows takes
    /// its cap from [`Self::affinity_rows_cap`], not from here: a profile that
    /// keeps its column has a sample too, and one that takes none is
    /// named without a cap. [`Self::needs_for`] answers for a
    /// worker's stream, where the larger of two needs is the safe one: for a
    /// decode decision of a model that also pairs it carries the pair's
    /// decode side, whose sample may be the larger. A cap's safe side is
    /// the smaller.
    pub fn sample_for(&self, model: &str, phase: Phase) -> u32 {
        self.resolve(model, phase).map_or(0, |(_, index, _)| {
            self.profile_needs[usize::from(index)].sample
        })
    }

    /// Whether what makes a request of `phase` for `model` states where
    /// the routing value's own text stands on a ring its caller lends: a
    /// scorer of the profile that decides it places by that
    /// ([`Scorer::places_by_key_text`]), or, for a pair asked at its
    /// prefill, a scorer of its decode profile does. And no scorer of
    /// either reads a ring by another text: a pair's two sides are handed
    /// one ring and one place, so a side of another ring form would take
    /// that place for its own. `false` where no binding matches. Only a
    /// request that lends a ring pays for the answer: one lookup of its
    /// binding, which names a pair's decode profile too. A decode asked
    /// again is answered for a decode request's own binding, which a
    /// document may order ahead of the pair's, as [`Self::sample_for`]
    /// answers. A caller that writes a request's facts itself asks it
    /// before it takes the place with the request's keys
    /// ([`KeySources::derive_keys`](crate::document::KeySources::derive_keys)).
    #[must_use]
    pub fn states_key_place(&self, model: &str, phase: Phase) -> bool {
        self.key_place_of(self.ring_profiles(model, phase))
    }

    /// The profiles a request of `phase` for `model` is placed by on a
    /// ring its caller lends: the one that decides it and, for a pair
    /// asked at its prefill, its decode profile. One lookup of its
    /// binding: a prefill's own entry is its model's pair entry where
    /// the plan pairs ([`Self::pair_entry`]), and the decode profile is
    /// read off it. What makes a request looks them up once and asks
    /// both of [`Self::key_place_of`] and [`Self::head_rule_of`].
    pub(crate) fn ring_profiles(&self, model: &str, phase: Phase) -> RingProfiles {
        let Some((entry, own, _)) = self.resolve(model, phase) else {
            return (None, None);
        };
        let decode = (phase == Phase::Prefill)
            .then(|| paired(entry))
            .flatten()
            .and_then(|pair| profile_index(&pair.binding, Phase::Decode));
        (Some(usize::from(own)), decode.map(usize::from))
    }

    /// [`Self::states_key_place`] by the two profiles of
    /// [`Self::ring_profiles`].
    pub(crate) fn key_place_of(&self, (own, decode): RingProfiles) -> bool {
        let marks = |at: Option<usize>| {
            let marks = at.and_then(|at| self.marks.get(at));
            marks.copied().unwrap_or_default()
        };
        let (own, decode) = (marks(own), marks(decode));
        (own.key_place() || decode.key_place()) && !(own.other_place() || decode.other_place())
    }

    /// [`Self::states_head_places`] by the two profiles of
    /// [`Self::ring_profiles`].
    pub(crate) fn head_rule_of(&self, (own, decode): RingProfiles) -> Option<HeadRule<'_>> {
        let own = self.head_places.get(own?)?;
        own.with(decode.and_then(|at| self.head_places.get(at)))
    }

    /// How what makes a request of `phase` for `model` cuts it into the
    /// heads whose places it states on a ring its caller lends: the rule
    /// of a scorer of the profile that decides the request
    /// ([`Scorer::places_by_heads`]), or, for a pair asked at its prefill,
    /// of a scorer of its decode profile. [`Self::states_key_place`]
    /// answers for the plain ring by the same two profiles, and this is
    /// its mirror: every scorer of either that places by heads names the
    /// one rule, and no scorer of either reads a ring by another text. A
    /// pair's two sides are handed one ring and one list of places, so a
    /// side that reads no ring holds nothing back, and a pair of two
    /// depths or of two ring forms states nothing. `None` then, and where
    /// no binding matches. The scorers answered when the engine was built,
    /// and no stage is asked here. Only a request that lends a ring to a
    /// profile with no place for its key's text pays for the answer: one
    /// lookup of its binding, which names a pair's decode profile too.
    #[must_use]
    pub fn states_head_places(&self, model: &str, phase: Phase) -> Option<HeadRule<'_>> {
        self.head_rule_of(self.ring_profiles(model, phase))
    }

    /// What `model`'s pair binding ([`Self::pair_binding`]) holds for
    /// `side`, one of [`PAIR_SIDES`]: the needs of that side's profile with
    /// its shadow's, and the profile's marks. Nothing where the model has
    /// no pair binding, and for the encode side of a `pd-pair` binding,
    /// which names none. One lookup of the binding answers both.
    fn pair_side(&self, model: &str, side: Phase) -> (Needs, Marks) {
        let held = self.pair_binding(model).and_then(|binding| {
            let index = usize::from(profile_index(binding, side)?);
            let needs = self.profile_needs.get(index).copied().unwrap_or_default();
            let marks = self.marks.get(index).copied().unwrap_or_default();
            Some((needs.or(self.shadow_needs(binding, side)), marks))
        });
        held.unwrap_or_default()
    }

    /// The pair binding of `model`: the binding its prefill lookup resolves
    /// to, where that binding's plan pairs (`pd-pair` or `epd`). Both sides
    /// of a pair decision run under it ([`Self::decide_pair`]), whatever
    /// binding covers the decode phase first.
    fn pair_binding(&self, model: &str) -> Option<&Binding> {
        self.pair_entry(model).map(|entry| &entry.binding)
    }

    /// [`Self::pair_binding`] as the engine holds it: the binding beside
    /// its word for a pool a soft veto empties, which a pair's decode side
    /// is put to the all-vetoed rule by. Its prefill side is put to none.
    fn pair_entry(&self, model: &str) -> Option<&Entry> {
        self.entry(model, Phase::Prefill).and_then(paired)
    }

    /// The binding a `phase` pool of `model` is decided under, whose
    /// overload block judges the pool. Every decision resolves by its own
    /// phase but a pair's decode side and an `epd` pair's encode side,
    /// which run under the pair binding: a decode pool is that binding's
    /// wherever the model has one, and an encode pool wherever the model
    /// has one of the `epd` plan. A binding of the side's phase ahead of
    /// it still decides a request of that phase asked alone, and its block
    /// is not read for the pool.
    pub fn pool_binding(&self, model: &str, phase: Phase) -> Option<&Binding> {
        let pair = match phase {
            Phase::Decode => self.pair_binding(model),
            Phase::Encode => self
                .pair_binding(model)
                .filter(|pair| matches!(pair.plan, Plan::Epd { .. })),
            Phase::Single | Phase::Prefill => None,
        };
        pair.or_else(|| self.binding(model, phase))
    }

    /// What some decision of `phase` may read, over every binding that
    /// covers the phase: the host's answer to "does any regular placement
    /// read the request text" and its kin, read at apply. A single decision
    /// over a host's worker lists skips its list's affinity table on this
    /// answer alone, so it never says less than [`Self::needs_for`] says of
    /// a request of the phase.
    pub fn needs_for_phase(&self, phase: Phase) -> Needs {
        self.phase_needs[slot(phase)]
    }

    /// What a host must feed, as rows: one for each model glob the bindings
    /// write and each phase with something to feed, in the bindings' order
    /// (design 5.1's needs summary). A row is an obligation and not the
    /// lookup: a host that feeds, for a model and a phase, every row whose
    /// glob matches the model feeds what any decision of that model and
    /// phase reads ([`Self::needs_for`]), and copies neither the bindings'
    /// order nor the pair's rule. The union feeds over where an earlier
    /// binding with fewer needs shadows a later one.
    ///
    /// A binding's row is its live and shadow stages and its sessions rule,
    /// for each phase it covers. A pair's decode profile is a decode row of the
    /// pair binding's glob, whichever binding a decode lookup resolves to,
    /// and an `epd` pair's encode profile an encode row of it the same way.
    /// Two bindings of one glob write one row a phase.
    pub fn needs_rows(&self) -> Vec<(&str, Phase, Needs)> {
        let mut rows: Vec<(&str, Phase, Needs)> = Vec::new();
        let mut held: HashMap<(&str, Phase), usize> = HashMap::new();
        for entry in &self.bindings {
            let binding = &entry.binding;
            let covers = |phase| binding.phases.is_empty() || binding.phases.contains(&phase);
            let own = |phase| {
                profile_index(binding, phase)
                    .and_then(|index| self.profile_needs.get(usize::from(index)))
                    .copied()
            };
            let pairs = matches!(binding.plan, Plan::PdPair | Plan::Epd { .. });
            for phase in PHASES {
                let mut needs = Needs::default();
                if let Some(own) = own(phase).filter(|_| covers(phase)) {
                    needs = own
                        .or(self.shadow_needs(binding, phase))
                        .or(session_needs(binding));
                }
                // The decode side of every pair this binding's prefill
                // lookup starts, and the encode side of an `epd` pair, each
                // with that side's shadow.
                if PAIR_SIDES.contains(&phase) && pairs && covers(Phase::Prefill) {
                    needs = needs
                        .or(own(phase).unwrap_or_default())
                        .or(self.shadow_needs(binding, phase));
                }
                if needs == Needs::default() {
                    continue;
                }
                let key = (binding.model.as_str(), phase);
                if let Some(&row) = held.get(&key) {
                    rows[row].2 = rows[row].2.or(needs);
                } else {
                    held.insert(key, rows.len());
                    rows.push((key.0, phase, needs));
                }
            }
        }
        rows
    }

    /// The needs of the shadow profile `binding` names for `phase`
    /// ([`shadow_index`]), nothing where it names none.
    fn shadow_needs(&self, binding: &Binding, phase: Phase) -> Needs {
        shadow_index(binding, phase)
            .and_then(|index| self.profile_needs.get(usize::from(index)))
            .copied()
            .unwrap_or_default()
    }

    pub fn profiles(&self) -> &[Profile] {
        &self.profiles
    }

    /// The first binding that matches, as the spec orders them, and as the
    /// spec declared it: a sessions rule reads back with the assignment the
    /// document names, whichever assignment the engine places by.
    pub fn binding(&self, model: &str, phase: Phase) -> Option<&Binding> {
        self.entry(model, phase).map(|entry| &entry.binding)
    }

    /// The first binding that matches, beside what the build derived from
    /// it: the engine's own lookup.
    fn entry(&self, model: &str, phase: Phase) -> Option<&Entry> {
        self.bindings
            .iter()
            .find(|entry| entry.binding.matches(model, phase))
    }

    /// The position of the first matching binding in this generation.
    pub(crate) fn binding_position(&self, model: &str, phase: Phase) -> Option<u32> {
        self.bindings
            .iter()
            .position(|entry| entry.binding.matches(model, phase))
            .and_then(|at| u32::try_from(at).ok())
    }

    pub(crate) fn binding_at(&self, at: u32) -> Option<&Binding> {
        self.bindings.get(at as usize).map(|entry| &entry.binding)
    }

    /// The cap of each list of rows a host names again for the shadow of
    /// the binding at `at` ([`ShadowRows::Own`]): the shadow profile's
    /// own, as [`Self::affinity_rows_cap`] says a live profile's. `0` where the
    /// binding names no shadow. A host asks it where it names such rows,
    /// and for no other decision.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "no host names a shadow rows of its own until the engine over worker lists does"
        )
    )]
    pub(crate) fn shadow_cap_at(&self, at: u32) -> u32 {
        self.bindings
            .get(at as usize)
            .and_then(|entry| entry.binding.shadow)
            .and_then(|shadow| self.named_by.get(usize::from(shadow)))
            .map_or(0, |by| by.cap)
    }

    /// The position of the binding a decode decided after its prefill
    /// runs under ([`Self::decide_decode`]): the pair binding its model's
    /// prefill lookup resolves to ([`Self::pair_binding`]), whatever
    /// phase the call states. A binding of the decode phase listed ahead
    /// of the pair's decides a decode asked alone, and no pair's. A host
    /// reads that call's shadow word ([`Self::shadow_at`]) and names its
    /// sampled record by this position. `None` where the prefill lookup
    /// finds no pair.
    #[cfg_attr(
        not(feature = "controller"),
        expect(
            dead_code,
            reason = "the served controller names a decode call's sampled record by it"
        )
    )]
    pub(crate) fn pair_position(&self, model: &str) -> Option<u32> {
        let at = self.binding_position(model, Phase::Prefill)?;
        let pairs = matches!(self.binding_at(at)?.plan, Plan::PdPair | Plan::Epd { .. });
        pairs.then_some(at)
    }

    /// Whether a request of `model` at `phase` is a side of a pair: the
    /// first binding that matches, by the phase's own lookup, states a plan
    /// that decides the phase with its pair ([`Plan::pairs`]). Any other
    /// request stands alone, and one that no binding matches does too: one
    /// profile decides it over the one list of its phase. A binding of one
    /// profile listed ahead of the pair's for the decode phase so decides
    /// a decode asked alone.
    #[cfg(feature = "in-process")]
    pub(crate) fn pairs(&self, model: &str, phase: Phase) -> bool {
        self.binding(model, phase)
            .is_some_and(|binding| binding.plan.pairs(phase))
    }

    /// The shadow word of the binding at `at`, where its shadow is asked
    /// ([`Self::shadow`], and [`Self::shadow_side`] for a pair's): the
    /// binding's position in this generation, and how that shadow reads
    /// the rows a host handed the live decision ([`ShadowRows`]). `None`
    /// where the binding names no shadow: one read says so, and a host
    /// orders such a request as it does under a generation that shadows
    /// nowhere.
    pub(crate) fn shadow_at(&self, at: u32) -> Option<(u32, ShadowRows)> {
        let rows = *self.shadow_rows.get(at as usize)?;
        // A single's placing stands where the build found the shadow
        // profile free to run beside the live one. A pair's word says
        // the same of a side of it.
        if rows != ShadowRows::Sides {
            self.shadow_placing.get(at as usize).copied().flatten()?;
        }
        Some((at, rows))
    }

    /// [`Self::shadow_at`] for the binding a request of `phase` for
    /// `model` resolves to.
    pub(crate) fn shadow_for(&self, model: &str, phase: Phase) -> Option<(u32, ShadowRows)> {
        self.shadow_at(self.binding_position(model, phase)?)
    }

    /// The profile a request resolves to for its phase; `None` when no
    /// binding matches or the binding's plan has no side for the phase.
    pub fn profile_for(&self, model: &str, phase: Phase) -> Option<(u16, &Profile)> {
        self.resolve(model, phase)
            .map(|(_, index, profile)| (index, profile))
    }

    /// The warm-up ruling a decision of `phase` for `model` applies: the one
    /// its profile's warm-up stages rule by, `None` where that profile holds
    /// no such stage, where two of them rule by different numbers, or no
    /// binding matches. A host that names a pool's warming rows enters it
    /// ([`WarmUpRule::list_warming_rows`]): it names a pool by one ruling, and
    /// a profile that applies two is named no rows
    /// ([`Self::affinity_rows_cap`]). A binding is found by model and phase
    /// alone, so a pool is decided by one profile and has one ruling.
    pub fn warm_up_rule_for(&self, model: &str, phase: Phase) -> Option<WarmUpRule> {
        let (_, index, _) = self.resolve(model, phase)?;
        self.named_by[usize::from(index)].rule
    }

    /// The ruling a host reads a pool's warming rows by, for the decisions
    /// of `phase` for `model`: the live profile's
    /// ([`Self::warm_up_rule_for`]) and, where that profile rules by none and
    /// its binding's shadow is named rows of its own, the shadow
    /// profile's. A live profile that rules by none reads no warming row,
    /// so the rows it is handed do not move.
    pub fn pool_rule_for(&self, model: &str, phase: Phase) -> Option<WarmUpRule> {
        self.warm_up_rule_for(model, phase).or_else(|| {
            let (at, rows) = self.shadow_for(model, phase)?;
            if rows != ShadowRows::Own {
                return None;
            }
            let shadow = self.bindings.get(at as usize)?.binding.shadow?;
            self.named_by.get(usize::from(shadow))?.rule
        })
    }

    /// Whether the profile a decision resolves to reads sent prefixes: a
    /// stage of it reads them, in the rows its decisions are named
    /// ([`Self::affinity_rows_cap`]) or, where it is named no row, in the
    /// column it is handed whole. The answer is of what the
    /// build kept: a membership write asks it of every pool, and no stage
    /// is asked again.
    pub(crate) fn reads_sent_for(&self, model: &str, phase: Phase) -> bool {
        self.needs.sent_prefixes
            && self
                .resolve(model, phase)
                .is_some_and(|(_, index, _)| self.named_by[usize::from(index)].sent)
    }

    /// Whether that profile reads a request's holders from the record of
    /// sent prefixes alone: a scorer of it reads the record, as the prefix
    /// stage does under the source that reads the record alone, and no
    /// prefix stage reads the index. A host names the record's holders
    /// ahead of the index's then. One prefix stage that reads cache events
    /// keeps the index's holders first, whatever another reads: under a
    /// cap it would lose them. A stage that reads cache events for a row's
    /// warm-up reads no holder and turns nothing here.
    /// Kept at the build, as [`Self::reads_sent_for`] is.
    pub(crate) fn reads_sent_alone_for(&self, model: &str, phase: Phase) -> bool {
        self.resolve(model, phase)
            .is_some_and(|(_, index, _)| self.named_by[usize::from(index)].sent_alone)
    }

    /// The record of sent prefixes a host keeps for a pool, for the
    /// decisions of `phase` for `model`: `None` where it keeps none, and
    /// whether the record's holders are named ahead of the index's. It
    /// is the live profile's word ([`Self::reads_sent_for`],
    /// [`Self::reads_sent_alone_for`]) and, where that profile is named
    /// no row and its binding's shadow is named rows of its own, the
    /// shadow profile's. A live profile that is named no row and reads
    /// no sent prefix reads nothing of the record its pool keeps there.
    /// A document that names no such shadow keeps none there. A shadow
    /// is named no such rows where a live profile reads the record of
    /// the same model ([`live_reader_beside`]): the entries kept for a
    /// shadow share a bound with no live reader's.
    ///
    /// A decode pool is read by its model's pair too: the pair's decode
    /// side runs under the pair's binding, whatever binding a decode
    /// request asked alone resolves to ([`Self::pool_binding`]). Where
    /// that request's profile reads no record, the pool keeps the one
    /// the pair's decode profile reads: that side is a live reader of
    /// the model's record. So a decode pool's word is, in this order,
    /// that request's profile's where it reads the record, the pair's
    /// decode profile's, and the shadow's as above.
    pub(crate) fn pool_record_for(&self, model: &str, phase: Phase) -> Option<bool> {
        if self.reads_sent_for(model, phase) {
            return Some(self.reads_sent_alone_for(model, phase));
        }
        // A pair's decode side reads the decode pool, whichever binding
        // the phase resolves to.
        let side = (phase == Phase::Decode)
            .then(|| self.pair_binding(model))
            .flatten()
            .and_then(|binding| profile_index(binding, Phase::Decode))
            .and_then(|index| self.named_by.get(usize::from(index)));
        if let Some(by) = side.filter(|by| by.sent) {
            return Some(by.sent_alone);
        }
        // The shadow's rows are its own with another record than the
        // live profile's only where that profile is named no row
        // ([`ShadowRows::of`]).
        let (at, rows) = self.shadow_for(model, phase)?;
        if rows != ShadowRows::Own {
            return None;
        }
        let shadow = self.bindings.get(at as usize)?.binding.shadow?;
        let by = self.named_by.get(usize::from(shadow))?;
        by.sent.then_some(by.sent_alone)
    }

    /// The cap of each list of rows a host names in the affinity column's
    /// place for a decision of `phase` for `model`, where the profile that
    /// decision resolves to reads the column by its own stages and its
    /// warm-up stages rule by one ruling: that profile's sample
    /// ([`Self::sample_for`]), and [`u32::MAX`] where it takes none. A
    /// sample caps how many rows are named and decides nothing else: a
    /// profile that takes none reads every row, so it is named every
    /// holder and every warming row. `0` otherwise, where the profile
    /// shows its rows to a plugin, whose stage reads the column whole, and
    /// where no binding matches: such a decision is named no row and
    /// handed the column where its needs declare it. The profile answers
    /// for itself here: [`Self::needs_for`] is merged over a pair's two
    /// phases and a binding's sessions rule, and says where to ask. The cap
    /// is the profile's own, said once at the build: a decision reads it.
    pub fn affinity_rows_cap(&self, model: &str, phase: Phase) -> u32 {
        self.resolve(model, phase)
            .map_or(0, |(_, index, _)| self.named_by[usize::from(index)].cap)
    }

    /// Whether the profile a decision of `phase` for `model` resolves to
    /// reads affinity by its own stages: the word of [`Self::needs_for`]
    /// less the binding's shadow profile, its sessions rule and, for a
    /// decode decision, the model's pair binding. `false` where no binding
    /// matches. A pick's overlap is stated by this word: rows a host names
    /// for a shadow alone say nothing of the live pick.
    #[cfg(feature = "in-process")]
    pub(crate) fn profile_reads_affinity(&self, model: &str, phase: Phase) -> bool {
        self.resolve(model, phase)
            .is_some_and(|(_, index, _)| self.profile_needs[usize::from(index)].affinity)
    }

    /// The test the standing pass of the decision's profile asks of a row
    /// ([`RowTest::keeps`]),
    /// for a host to ask of a holder before it names it. It is the test a
    /// draw asks ([`run_drawn`]): the profile's filters as one test where
    /// the build folded them, with the host's mask where a veto stands, and
    /// a call a filter otherwise.
    ///
    /// `None` where the pass alone can say which rows stay: a filter of the
    /// profile does not answer row by row ([`Filter::row_wise`]), a filter
    /// declares what the door writes behind the test (the naming's column,
    /// the key's counts), a stage holds a decision's scope
    /// ([`DecisionScope`]) while the filters are asked by a call, which
    /// would run ahead of the scope's opening, or the profile stands under
    /// the key's mark, where a keyed request's pass runs with the overload
    /// veto lifted ([`Engine::places_by_key`]). The mark alone decides: the
    /// function reads nothing of the request's key. The profile's own part
    /// is said once at the build ([`Profile::asked_by_row`]). `None` too
    /// where no binding matches.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "No host asks a holder's test before it names the holder yet."
        )
    )]
    pub(crate) fn keeps_for<'a>(
        &'a self,
        model: &str,
        phase: Phase,
        facts: &'a RequestFacts<'a>,
        cx: &'a Constraints<'a>,
    ) -> Option<RowTest<'a>> {
        let (_, index, _) = self.resolve(model, phase)?;
        self.keeps_at(index, facts, cx)
    }

    /// The test the standing pass of the profile at `index` asks of a
    /// row ([`Self::keeps_for`]).
    fn keeps_at<'a>(
        &'a self,
        index: u16,
        facts: &'a RequestFacts<'a>,
        cx: &'a Constraints<'a>,
    ) -> Option<RowTest<'a>> {
        if self.marks.get(usize::from(index))?.keyed() {
            return None;
        }
        self.row_test_at(index, facts, cx)
    }

    /// The test the profile at `index` asks of one row under `cx`, whatever
    /// the key's mark says: [`Self::keeps_at`] for a caller that reads the
    /// overload veto as its own pass does, lifted in `cx` for a kept key
    /// ([`Self::decide_keyed`]). `None` where the pass alone can say which
    /// rows stay ([`Profile::asked_by_row`]).
    fn row_test_at<'a>(
        &'a self,
        index: u16,
        facts: &'a RequestFacts<'a>,
        cx: &'a Constraints<'a>,
    ) -> Option<RowTest<'a>> {
        let at = usize::from(index);
        let filters = &self.profiles.get(at)?.filters[..];
        let tests = match self.named_by.get(at)?.held {
            Held::Unasked => return None,
            Held::Folded(tests) => Some(tests),
            Held::Called => None,
        };
        Some(RowTest {
            // The set as this decision asks it, narrowed once as the
            // draw narrows it.
            tests: tests.map(|tests| tests.asked(facts, cx)),
            masked: tests.is_some_and(|tests| tests.masked(cx)),
            filters,
            facts,
            cx,
        })
    }

    /// The binding a request resolves to, with its profile for the phase: one
    /// walk of the bindings per decision; `None` when no binding matches or
    /// the binding's plan has no side for the phase.
    fn resolve(&self, model: &str, phase: Phase) -> Option<(&Entry, u16, &Profile)> {
        let entry = self.entry(model, phase)?;
        let index = profile_index(&entry.binding, phase)?;
        let profile = self.profiles.get(usize::from(index))?;
        Some((entry, index, profile))
    }

    /// One profile over `candidates`: the `single` plan, or one encode item
    /// of the `epd` plan (the request's phase is `Encode` and the plan's
    /// compatibility and vision rules filter the pool before the profile);
    /// under a `single` binding's sessions rule a keyed request goes to its
    /// pinned worker first, and a profile whose picker leaves the placement
    /// to that rule is placed by its assignment. Every other request of a
    /// profile whose picker draws first ([`Picker::draw_first`]) is decided
    /// by the rows it draws where the pool is large enough, and by the whole
    /// pass under that size or when the draw keeps nobody. A picker that
    /// takes turns ([`Picker::takes_turns`]) draws under that size too, over
    /// a pool that lends its own word ([`PoolFacts::word`]). A picker the rows
    /// named join ([`Picker::joined_by_named`]) draws for a request that
    /// names rows and for no other. A pass the overload veto leaves nobody is
    /// put to the binding's all-vetoed rule: the request goes to the
    /// least-vetoed worker, or is shed where the binding says so. A pool the
    /// liveness doubt leaves nobody is put to the rule's second tier under
    /// either word: the request goes to a doubted worker, and where the
    /// binding sheds to one under its thresholds alone. A shed stands where
    /// the pool stays empty with those vetoes lifted. Where the binding
    /// steers, a keyed request of a profile that places by the session key is
    /// decided with the overload veto lifted, so the key keeps its worker;
    /// under a sessions rule with an in-flight cap, only where the host holds
    /// the key's counts ([`RequestFacts::session_counted`]). The request's
    /// affinity column is empty or holds one entry a candidate; any other is
    /// refused, as a decline under the stage name `affinity-column`. Rows
    /// named in the column's place ([`AffinityRows`]) are read as the host
    /// derived them: a debug build asserts that they stand for the pool. A
    /// profile that shows its rows to a plugin is declined over rows that
    /// carry a mask or a pin beside them, under the stage name
    /// `written-rows` ([`Self::reads_for`]).
    pub fn decide(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        self.decide_as::<false>(facts, pool, candidates, cx, scratch)
    }

    /// [`Self::decide`] for a request whose shadow will be asked
    /// ([`Self::shadow`]). It makes the same decision. Under a sessions
    /// rule it also keeps in the scratch what it learned of the request's
    /// key: the key's pins, and the pins the table held at a claim it
    /// lost. The shadow takes each where this decision took it, since the
    /// commit between the two moves the pins. A decision
    /// that answers before it reads its key keeps none, and leaves none
    /// of an earlier decision.
    /// A host that shadows a request decides it through this entry, each
    /// time it decides it.
    pub fn decide_keeping(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        // No reading yet: one an earlier decision left is no word of this
        // request, which may answer before it reads its key.
        scratch.sessions_found.forget();
        self.decide_as::<true>(facts, pool, candidates, cx, scratch)
    }

    /// The draw a plain single decision of `facts` is, over a pool of
    /// `rows` rows whose host writes a row's record when the decision asks
    /// about the row ([`Self::decide_drawn`]). `None` where
    /// [`Self::decide`] would not decide these facts by its draw at once,
    /// or the profile's draw may not be asked before the records
    /// ([`DRAWS_BY_ITS_TEST`]): that host writes every row and asks
    /// [`Self::decide`].
    ///
    /// The conditions are the ones [`Self::decide`] comes to its draw
    /// under, in its order: the binding the request resolves to; no column
    /// of affinity facts, which is one entry a written row; no scope to
    /// open, no assignment that places and no key that lifts a veto; no
    /// sessions rule over a keyed request; no encode item; no list of the
    /// pool's engines for a profile that decides over them; a pool of
    /// [`DRAW_FROM`] rows or more; and a profile that draws for every
    /// request, or for this one, which names rows. A smaller pool is
    /// written whole, also where [`Self::decide`] then draws over it for
    /// a picker that takes turns. `pool` is read for its list of engines
    /// alone.
    #[cfg(feature = "in-process")]
    pub(crate) fn drawn_for(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        rows: usize,
    ) -> Option<Drawn> {
        if rows < DRAW_FROM || !facts.affinity.is_empty() {
            return None;
        }
        let (entry, index, _) = self.resolve(facts.model, facts.phase)?;
        let at = usize::from(index);
        if !self.written_draws.get(at).copied().unwrap_or(false) {
            return None;
        }
        let marks = self.marks[at];
        let binding = &entry.binding;
        let keyed = binding.sessions.is_some()
            && binding.plan == Plan::Single
            && !facts.session_key.is_none();
        let encode = matches!(
            (binding.plan, facts.phase),
            (Plan::Epd { .. }, Phase::Encode)
        );
        if !marks.asks_nothing_first() || keyed || encode {
            return None;
        }
        if marks.first() & (Marks::ENGINES | Marks::OWNS) != 0 && !pool.engines.is_empty() {
            return None;
        }
        let ahead = marks.ahead();
        let draws =
            ahead == Marks::DRAWS_FIRST || (facts.named.is_some() && ahead & Marks::JOINED != 0);
        draws.then_some(Drawn {
            index,
            vetoed: entry.vetoed,
        })
    }

    /// [`Self::decide`] for a decision that is a draw's ([`Self::drawn_for`]
    /// said so of these facts and this pool), by a host that keeps the
    /// fixed half of its rows and writes a row's record when the decision
    /// asks about the row.
    ///
    /// `fixed` are the pool's rows with no live value, in the pool's order,
    /// and `rows` is the host's buffer of as many rows. `write` is handed a
    /// fixed row and the buffer's row at its place, and writes the whole
    /// record there: it is called for each row the draw asks about, for
    /// each row the request names, and for every row where the draw keeps
    /// nobody and the whole pass decides. The decision is the one
    /// [`Self::decide`] makes over the same rows written whole. No lock
    /// and no allocation beyond the stages' own.
    #[cfg(feature = "in-process")]
    #[expect(
        clippy::too_many_arguments,
        reason = "one decision's inputs, and the host's two halves of a row: the fixed rows, the buffer and the write"
    )]
    pub(crate) fn decide_drawn(
        &self,
        drawn: Drawn,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        fixed: &[Candidate],
        rows: &mut [Candidate],
        write: &mut dyn FnMut(&Candidate, &mut Candidate),
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        // The plain entry decides: a debug build keeps it for the shadow's
        // stop on a mix of the two.
        #[cfg(debug_assertions)]
        {
            scratch.kept_last = false;
        }
        run_drawn_written(
            self,
            drawn,
            facts,
            pool,
            fixed,
            rows,
            write,
            cx,
            &mut scratch.single,
        )
    }

    /// The body of the two entries. `KEEPS` is the keeping entry's form.
    /// It is a parameter of the type, so neither entry tests it. Always
    /// inlined: each entry is this body.
    #[inline(always)]
    fn decide_as<const KEEPS: bool>(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        // Which entry decides: a debug build keeps it for the shadow's
        // stop on a mix of the two.
        #[cfg(debug_assertions)]
        {
            scratch.kept_last = KEEPS;
        }
        let bare = Trace {
            seed: facts.seed,
            generation: pool.generation,
            ..Trace::default()
        };
        debug_assert_indexed(candidates);
        debug_assert_cells(facts.signal_cells, candidates);
        debug_assert_named(facts, candidates);
        if !column_fits(facts.affinity, candidates) {
            return shed(ShedReason::PolicyDeclined, AFFINITY_COLUMN, bare);
        }
        let Some((entry, index, profile)) = self.resolve(facts.model, facts.phase) else {
            return shed(ShedReason::PolicyDeclined, "binding", bare);
        };
        let Entry {
            binding,
            placing,
            vetoed,
            sides_placed: _,
        } = entry;
        // The single plan's rows, under the binding's word for a pool the
        // overload veto empties: every path below filters this set.
        let set = Set {
            candidates,
            class: None,
            partnered: None,
            fold: None,
            encode: false,
            vetoed: *vetoed,
        };
        let marks = self.marks[usize::from(index)];
        // A picker that leaves the placement to its rule is not asked: the
        // rule's pin takes a keyed request first, below, and the binding's
        // placing assignment places every other, under a rule or under none.
        let placed = || {
            marks.by_assignment()
                && binding.plan == Plan::Single
                && (binding.sessions.is_none() || facts.session_key.is_none())
        };
        // A plugin is shown each row as a record of the row's own fields:
        // its host hands rows written for the decision
        // ([`Self::reads_for`]). A mask or a pin beside the rows says they
        // are not, and the arm that would open the scope declines.
        let unwritten = || !cx.stalled.is_empty() || facts.pinned_at != 0;
        // One arm a mark, the scope's and the assignment's first: a profile
        // under those two alone meets no test for the key.
        let _opened = if marks.asks_nothing_first() {
            None
        } else if marks.opens_first() {
            if unwritten() {
                return rows_not_written(bare, index);
            }
            let opened = profile.open(facts, candidates, pool);
            if placed() {
                return self.assign(profile, index, *placing, facts, pool, set, cx, scratch);
            }
            Some(opened)
        } else if marks.by_assignment() {
            if placed() {
                return self.assign(profile, index, *placing, facts, pool, set, cx, scratch);
            }
            None
        } else {
            // The key's mark: a keyed request keeps its key's worker over
            // the overload veto, where the binding steers.
            if marks.keyed_scope() && unwritten() {
                return rows_not_written(bare, index);
            }
            let opened = marks
                .keyed_scope()
                .then(|| profile.open(facts, candidates, pool));
            if vetoed::keeps_its_key(binding, *placing, *vetoed, facts)
                && vetoed::regular(candidates, cx)
            {
                let rule = binding.sessions.as_ref();
                // The lifted run stands in for the rule's overload tier. A
                // pool it leaves nobody is put to the liveness tier, by the
                // binding's own set.
                return self.keep::<KEEPS>(profile, index, rule, facts, pool, set, cx, scratch);
            }
            opened
        };
        if let Some(rule) = binding.sessions {
            if binding.plan == Plan::Single && !facts.session_key.is_none() {
                return self.decide_session::<false, KEEPS>(
                    profile, index, &rule, *placing, facts, pool, set, cx, scratch,
                );
            }
        }
        let buf = &mut scratch.single;
        // An encode item's rules are the whole pass's own: it never draws.
        if matches!(
            (binding.plan, facts.phase),
            (Plan::Epd { .. }, Phase::Encode)
        ) {
            return run_encode(profile, index, facts, pool, set, cx, buf);
        }
        // What may stand ahead of the whole pass is asked of the marks
        // once: the unit's runs, a draw or the ranks.
        let first = marks.first();
        if first != 0 {
            // Over a pool whose host lists its engines, at any pool size:
            // a picker whose unit is the engine decides by its draw over
            // them, and a scorer that places a keyed request by itself is
            // asked first. Where none is listed, or the scorer places
            // nothing, the paths below decide.
            if first & (Marks::ENGINES | Marks::OWNS) != 0 && !pool.engines.is_empty() {
                if marks.engines() {
                    return run_engines(self, index, facts, pool, set, cx, buf);
                }
                if marks.owns() && facts.ring_key() != 0 {
                    let owned = run_owned(self, index, facts, pool, candidates, cx, buf);
                    if let Some(decision) = owned {
                        return decision;
                    }
                }
            }
            // A draw or the ranks over a pool of `DRAW_FROM` rows or more:
            // the ranks' floor is the draw's. A turn's draw under the floor
            // is asked behind them.
            let ahead = marks.ahead();
            if ahead != 0 && candidates.len() >= DRAW_FROM {
                // A profile that draws for every request is the draw's at one
                // compare, the one it paid before the ranks: it holds no rank
                // mark, so its marks read `DRAWS_FIRST` alone ([`Marks::of`]).
                if ahead == Marks::DRAWS_FIRST {
                    return run_drawn(self, index, facts, pool, set, cx, buf);
                }
                // A picker the rows named join draws for a request that names
                // rows. The request is asked first: one that names none skips
                // the mark.
                if facts.named.is_some() && ahead & Marks::JOINED != 0 {
                    return run_drawn(self, index, facts, pool, set, cx, buf);
                }
                // A profile whose leading stages are ranks is asked by rank
                // where the draw left the decision and the host says that every
                // row of the pool holds a report ([`PoolFacts::reported`]). The
                // load stage costs a row without one from its reporting peers,
                // so of one in front it would say that it must be asked whole,
                // after the ranks were read: a pool not said to hold none is
                // asked whole at once. A configured age limit can make a
                // reported row use that same path.
                if marks.ranked()
                    && pool.reported
                    && self.max_report_age_ms.is_none_or(|max_age| {
                        all_reports_current(candidates, facts.now_micros, max_age)
                    })
                {
                    return run_ranked(self, index, facts, pool, set, cx, buf);
                }
            }
            // Under the floor a draw is asked of one kind of profile: one
            // whose picker takes turns, over a pool that lends its own
            // word. Its draw is the list's own turn, which the whole pass
            // does not take: it is the same decision at every pool size.
            // From the floor up such a profile's draw was asked above.
            if marks.turns() && pool.word.is_some() {
                return run_drawn(self, index, facts, pool, set, cx, buf);
            }
        }
        run_whole(self, index, facts, pool, set, cx, buf)
    }

    /// Make the binding's decision with its shadow profile on the same
    /// rows. The run only updates the shadow's stages and scratch. It
    /// claims and renews no session pin, and it is lent no word of the pool
    /// ([`PoolFacts::word`]): the live decision alone turns the list's
    /// rotation. It is lent the ring its request lends
    /// ([`RequestFacts::ring`]) only where it reads a ring by the live
    /// profile's kind of place: the place stated is the live profile's, so
    /// a shadow that reads a ring by another text places on its own ring.
    ///
    /// Make the request's live decision through [`Self::decide_keeping`]
    /// and hand this call its scratch, before the scratch decides again.
    /// Under a sessions rule that decision keeps there what it learned of
    /// the key: the key's pins, and the pins the table held at a claim it
    /// lost.
    /// The shadow of the same request (the same key, clock and seed) asks
    /// for neither again: it takes each where the live decision took
    /// it, and reads it against its own pins and its own pick. A claim's
    /// answer is never replayed: where no pin came in, that answer is the
    /// live pick, and a shadow of another profile keeps its own. So it
    /// decides by the sessions as the live decision found them, whether
    /// or not that decision is committed first: a commit moves the pins.
    /// With no keeping decision of the request on this scratch (behind
    /// [`Self::decide`] alone, or with another scratch), the shadow reads
    /// the pins as they stand.
    ///
    /// A host that shadows a request decides it through the keeping
    /// entry, each time it decides it. Behind a plain decision the
    /// scratch can still hold an earlier keeping decision's reading of
    /// the request. The shadow would start from it, and a debug build
    /// stops there.
    ///
    /// Returns `None` where the phase resolves to no shadow. When the live
    /// and shadow profiles need different sets of named affinity rows,
    /// supply the affinity column.
    pub fn shadow(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Option<Decision> {
        let (entry, placing) = self
            .bindings
            .iter()
            .zip(&self.shadow_placing)
            .find(|(entry, _)| entry.binding.matches(facts.model, facts.phase))?;
        let placing = (*placing)?;
        let index = entry.binding.shadow?;
        let profile = &self.profiles[usize::from(index)];
        let marks = self.marks[usize::from(index)];
        debug_assert_indexed(candidates);
        debug_assert_cells(facts.signal_cells, candidates);
        debug_assert_named(facts, candidates);
        if !column_fits(facts.affinity, candidates) {
            let trace = Trace {
                seed: facts.seed,
                generation: pool.generation,
                ..Trace::default()
            };
            return Some(shed(ShedReason::PolicyDeclined, AFFINITY_COLUMN, trace));
        }
        // The place a request states on a lent ring is the live profile's
        // kind of place. A shadow that reads a ring by the other kind is
        // lent none: the place is not its own, and it places on its own
        // ring. So is a shadow of the prefix form that cuts a request by
        // another rule than the live profile's: the places stated are the
        // live profile's heads'. A live profile that reads no ring states
        // no place, and its shadow is apart from none.
        let live = profile_index(&entry.binding, facts.phase).map(usize::from);
        let apart = live.is_some_and(|live| {
            let form = self
                .marks
                .get(live)
                .is_some_and(|&live| marks.ring_apart(live));
            let own = self.head_places.get(usize::from(index));
            let live = self.head_places.get(live);
            let reads = |places: Option<&HeadPlaces>| {
                places.is_some_and(|places| *places != HeadPlaces::Unread)
            };
            form || (reads(own) && reads(live) && own != live)
        });
        let unlent;
        let facts = if facts.ring.is_some() && apart {
            unlent = RequestFacts {
                ring: None,
                ring_positions: &[],
                ..*facts
            };
            &unlent
        } else {
            facts
        };
        // The pool's own word counts the live decision's turns alone: a
        // shadow is lent none, and its picker turns a rotation of its own.
        let pool = &PoolFacts {
            word: None,
            ..*pool
        };
        let set = Set {
            candidates,
            class: None,
            partnered: None,
            fold: None,
            encode: false,
            vetoed: entry.vetoed,
        };
        let binding = &entry.binding;
        let keyed = !facts.session_key.is_none();
        let rule = binding.sessions.as_ref().filter(|_| keyed);
        if marks.by_assignment() && rule.is_none() {
            return Some(self.assign(profile, index, placing, facts, pool, set, cx, scratch));
        }
        if marks.keyed()
            && vetoed::keeps_its_key(binding, placing, entry.vetoed, facts)
            && vetoed::regular(candidates, cx)
        {
            // The binding's own set, as the live decision hands it: a pool
            // the lifted run leaves nobody is put to the liveness tier.
            return Some(match rule {
                Some(rule) => self.decide_with_session::<true, false, false>(
                    profile, index, rule, placing, facts, pool, set, cx, scratch,
                ),
                None => self.keep::<false>(profile, index, None, facts, pool, set, cx, scratch),
            });
        }
        if let Some(rule) = rule {
            return Some(self.decide_with_session::<false, false, false>(
                profile, index, rule, placing, facts, pool, set, cx, scratch,
            ));
        }
        let buf = &mut scratch.single;
        if !pool.engines.is_empty() {
            if marks.engines() {
                return Some(run_engines(self, index, facts, pool, set, cx, buf));
            }
            if marks.owns() && facts.ring_key() != 0 {
                if let Some(decision) = run_owned(self, index, facts, pool, candidates, cx, buf) {
                    return Some(decision);
                }
            }
        }
        let ahead = marks.ahead();
        if ahead != 0 && candidates.len() >= DRAW_FROM {
            if ahead == Marks::DRAWS_FIRST || (facts.named.is_some() && ahead & Marks::JOINED != 0)
            {
                return Some(run_drawn(self, index, facts, pool, set, cx, buf));
            }
            if marks.ranked()
                && pool.reported
                && self.max_report_age_ms.is_none_or(|max_age| {
                    all_reports_current(candidates, facts.now_micros, max_age)
                })
            {
                return Some(run_ranked(self, index, facts, pool, set, cx, buf));
            }
        }
        Some(run_whole(self, index, facts, pool, set, cx, buf))
    }

    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the profile run's inputs, as run_profile takes them"
    )]
    fn decide_session<const KEPT: bool, const KEEPS: bool>(
        &self,
        profile: &Profile,
        index: u16,
        rule: &SessionRule,
        placing: Assignment,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        set: Set<'_>,
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        self.decide_with_session::<KEPT, true, KEEPS>(
            profile, index, rule, placing, facts, pool, set, cx, scratch,
        )
    }

    /// A keyed decision under a sessions rule (design 4.6 and the host's
    /// keyed path before it): the profile's filters narrow the pool; the
    /// key's preferred pin, or its failover when the preferred one is
    /// unroutable, is the decision while it is under the in-flight cap; a pin
    /// at the cap puts the request to the stages again over every candidate,
    /// and moves only to another worker under the cap; a key with no
    /// routable pin is placed by the stages. The stages are the profile's own
    /// under `delegate`, else the assignment's, and `placing` names which:
    /// the binding's placing assignment, not the rule's declared one. The
    /// pin is written by the dispatch (`on_dispatched`). Two writes come
    /// ahead of it here, the live decision's alone: a key with no routable
    /// pin claims the worker it is placed on, and a pin at the cap moves to
    /// the worker a respill takes ([`Self::moved`]).
    ///
    /// `KEPT` is the form of a profile that places by the session key, under
    /// a binding that steers ([`Engine::keep`]): the pin's test reads the
    /// overload veto as it stands, and a key no pin takes is placed by the
    /// profile with the veto lifted. Where the pass with the veto standing
    /// kept nobody, that placement says the lifted veto in its trace
    /// ([`LiftedVeto::Overload`]). A placement that lands on one of the key's
    /// pins again follows that pin under the rule's in-flight cap
    /// ([`Self::respill`]): nothing is claimed, and a key handed on reads
    /// the veto as it stands before the lifted pass's survivors. Handed on
    /// to a survivor of the lifted pass alone, it says the lifted veto too.
    ///
    /// With `WRITE_SESSIONS` off the decision is a shadow's: it writes no
    /// pin. It starts from what the live decision of the same request
    /// read of the key, where the scratch holds that reading
    /// ([`SessionsFound`]). Where it does not, the shadow reads the table's
    /// pins as they stand.
    ///
    /// `KEEPS` is the form of the keeping entry
    /// ([`Engine::decide_keeping`]). Its live decision keeps that reading
    /// in the scratch, since its commit moves the pins and the shadow is
    /// asked behind it. The plain entry's keeps none, and a shadow keeps
    /// none.
    #[expect(
        clippy::too_many_arguments,
        reason = "the profile run's inputs, as run_profile takes them"
    )]
    fn decide_with_session<const KEPT: bool, const WRITE_SESSIONS: bool, const KEEPS: bool>(
        &self,
        profile: &Profile,
        index: u16,
        rule: &SessionRule,
        placing: Assignment,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        set: Set<'_>,
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let candidates = set.candidates;
        let buf = &mut scratch.single;
        // Whether the veto as it stands left the pool nobody: the form of a
        // kept key reads it, for the mark of its lifted placement.
        let mut all_vetoed = false;
        if KEPT {
            // A pass that keeps nobody leaves no pin standing, and the
            // lifted pass answers below: this pass is put to no rule.
            all_vetoed = filter(profile, facts, cx, buf, set.shedding()).is_some();
        } else if let Some(answer) = single_survivors(profile, index, facts, pool, set, cx, buf) {
            return answer;
        }
        let key = facts.session();
        // A shadow starts from what the live decision of the same request
        // read of the key, where the scratch is that decision's.
        let found = scratch.sessions_found;
        let handed = !WRITE_SESSIONS && found.is_of(key.get(), facts);
        // A host that shadows a request decides it through the keeping
        // entry. Behind a plain decision the scratch can still hold an
        // earlier keeping decision's reading of the same request.
        #[cfg(debug_assertions)]
        debug_assert!(
            !handed || scratch.kept_last,
            "the shadow takes a reading behind a plain decision"
        );
        let pins = if WRITE_SESSIONS {
            self.sessions.lookup(key, facts.now_micros, rule)
        } else if handed {
            found.pins
        } else {
            self.sessions.pins_at(key, facts.now_micros)
        };
        if WRITE_SESSIONS && KEEPS {
            // What the table gave at the read: the shadow starts from it.
            // The table's answer to a claim that is not taken is kept
            // where this decision takes it, and the shadow takes it at
            // the same step.
            let reading = &mut scratch.sessions_found;
            reading.key = key.get();
            reading.now_micros = facts.now_micros;
            reading.seed = facts.seed;
            reading.pins = pins;
            reading.stood = false;
        }
        // The first pin the filters kept, by the pins' order.
        let owner = pins
            .iter()
            .filter(|p| **p != WorkerId(0))
            .find_map(|p| buf.kept.iter().find(|&&i| candidates[i as usize].id == *p))
            .copied();
        if let Some(i) = owner {
            let c = &candidates[i as usize];
            if Self::at_cap(rule, c) {
                // At the cap: the stages place the request again, over every
                // candidate the filters kept.
                if let Some(respilled) = self.respill(
                    profile, index, rule, placing, facts, pool, set, cx, buf, c.id,
                ) {
                    return self.moved::<WRITE_SESSIONS>(key, respilled);
                }
            }
            let mut targets = Targets::new();
            targets.push(Target {
                position: c.position,
                id: c.id,
                dp_rank: c.dp_rank,
                pair: None,
            });
            // The rule's pin placed the request: no stage was asked.
            targets.by_rule(rule.rule_placed);
            let mut trace = Trace {
                profile: index,
                survivors: Survivors::counted(buf.kept.len() as u32),
                seed: facts.seed,
                generation: pool.generation,
                ..Trace::default()
            };
            trace.top[0] = (c.id, 1.0);
            trace.top_len = 1;
            return Decision::Pick(Pick {
                targets,
                trace,
                session_verdict: SessionVerdict::Pinned,
            });
        }
        if KEPT {
            // No pin stands under the veto: the profile places the key with
            // the veto lifted, and the claim's lead below reads that pass's
            // survivors.
            if let Some(shed) = vetoed::lifted(profile, index, facts, pool, set, cx, buf) {
                return shed;
            }
        }
        let mut decision = self.place(profile, index, placing, facts, pool, set, cx, buf);
        if let Decision::Pick(pick) = &mut decision {
            // Under any assignment but `delegate` the rule's own stage
            // placed the key: no stage of the profile was asked.
            if placing != Assignment::Delegate {
                pick.targets.by_rule(rule.rule_placed);
            }
            if WRITE_SESSIONS {
                pick.session_verdict = SessionVerdict::Pinned;
            }
            if all_vetoed {
                // The lifted pass alone kept a worker: the placement stands
                // by the lift and says so, as the all-vetoed rule's pick does.
                pick.trace.lifted = LiftedVeto::Overload;
            }
            // No routable pin (a fresh key, or every pin down): the key is
            // claimed at the decision, so concurrent requests agree on the
            // first claim when its worker is among this decision's survivors.
            if let Some(first) = pick.targets.first().map(|t| t.id) {
                // A kept key the lifted pass put back on one of its pins is
                // no claim: the pin stands as it was read. A claim would
                // mark it provisional, and an alternate's dispatch would
                // withdraw it. The plain form never names a pin here, and
                // carries no test for one.
                if KEPT && pins.contains(&first) {
                    pick.session_verdict = SessionVerdict::Pinned;
                    // A dispatch on an alternate writes that worker among
                    // the key's pins: none at the rule's cap stands behind
                    // the pin.
                    Self::drop_alternates_at_cap(rule, candidates, &buf.kept, &mut pick.targets);
                    // The followed pin is put to the rule's cap as the pin
                    // read first is: the lift is of the overload veto alone.
                    // A rule with no cap asks no walk for the pin's row.
                    let capped = rule.inflight_cap != 0
                        && buf
                            .kept
                            .iter()
                            .find(|&&i| candidates[i as usize].id == first)
                            .is_some_and(|&i| Self::at_cap(rule, &candidates[i as usize]));
                    if capped {
                        if let Some(handed) = self.hand_on::<KEPT>(
                            profile, index, rule, placing, facts, pool, set, cx, buf, all_vetoed,
                            first,
                        ) {
                            return self.moved::<WRITE_SESSIONS>(key, handed);
                        }
                    }
                    return decision;
                }
                let claim = if WRITE_SESSIONS {
                    self.sessions.claim(key, pins, first)
                } else if handed && found.stood {
                    // The live decision's claim was not taken: its shadow
                    // reads the pins the table held there against its own
                    // pins and its own pick, as the table does. Where no
                    // pin came in, the table's answer was the live pick:
                    // no pin for the shadow to follow, and it keeps its
                    // own. Where the pins stood as read, the live pick
                    // was a pin already under the id that stands for its
                    // worker. A shadow asks no id that stands: it keeps
                    // its own pick there too, put to no cap. Where the
                    // live claim took a rename's step for the key first,
                    // the pin it renamed reads as the newcomer.
                    match Claim::standing(&found.standing, &pins, first) {
                        Some(owner) => Claim::Standing(owner, found.standing),
                        None => return decision,
                    }
                } else {
                    return decision;
                };
                match claim {
                    Claim::Made(id) => pick.session_verdict = SessionVerdict::Claimed(id),
                    // A full table records nothing of the key: the
                    // placement stands as the profile made it.
                    Claim::NoRoom => {}
                    // The table took no claim of this decision's: another
                    // request's stands, the pins moved under the
                    // decision, or its worker is a pin of the key already,
                    // under the id that stands or under one a rename's
                    // walk had yet to reach. The worker it names is a
                    // pin of the key where it is among the survivors: the
                    // rule's cap holds for it as for the pin read first,
                    // and it leads where the placement stood elsewhere.
                    Claim::Standing(owner, held) => {
                        if WRITE_SESSIONS && KEEPS {
                            // The pins the table held, kept for the shadow.
                            scratch.sessions_found.standing = held;
                            scratch.sessions_found.stood = true;
                        }
                        if let Some(&i) = buf
                            .kept
                            .iter()
                            .find(|&&i| candidates[i as usize].id == owner)
                        {
                            let c = &candidates[i as usize];
                            if owner != first {
                                pick.targets.lead(Target {
                                    position: c.position,
                                    id: c.id,
                                    dp_rank: c.dp_rank,
                                    pair: None,
                                });
                            }
                            // The alternates behind the claimed worker, as
                            // behind a followed pin: none at the rule's cap.
                            // Read here, ahead of the hand-on's own passes
                            // over the buffers.
                            Self::drop_alternates_at_cap(
                                rule,
                                candidates,
                                &buf.kept,
                                &mut pick.targets,
                            );
                            if Self::at_cap(rule, c) {
                                if let Some(handed) = self.hand_on::<KEPT>(
                                    profile, index, rule, placing, facts, pool, set, cx, buf,
                                    all_vetoed, owner,
                                ) {
                                    return self.moved::<WRITE_SESSIONS>(key, handed);
                                }
                            }
                        }
                    }
                }
            }
        }
        decision
    }

    /// A respill's move, written where the request is decided: the worker a
    /// pin at its cap hands the key on to leads the key's pins from this
    /// decision on ([`SessionTable::respilled`]), ahead of the request's
    /// dispatch. A gateway's sticky map moves a pin where it selects the
    /// worker, so the key's next request, decided ahead of that dispatch,
    /// follows the move there too. The live decision's alone: with
    /// `WRITE_SESSIONS` off the decision is a shadow's and writes no pin.
    /// A placement on the pin again, and one on another worker at the cap,
    /// move no pin ([`Self::respill`]). `key` is the key the decision
    /// looked up.
    #[inline]
    fn moved<const WRITE_SESSIONS: bool>(&self, key: Session, handed: Decision) -> Decision {
        if WRITE_SESSIONS {
            if let Decision::Pick(pick) = &handed {
                if pick.session_verdict == SessionVerdict::Respill {
                    if let Some(first) = pick.targets.first() {
                        self.sessions.respilled(key, first.id);
                    }
                }
            }
        }
        handed
    }

    /// A request of a profile whose picker leaves the placement to the
    /// binding's sessions rule, with no key to pin or no rule to pin it by:
    /// the profile's filters narrow the pool and the assignment's stage
    /// places the request, as it places a keyed request with no routable
    /// pin. Nothing is pinned. Out of line, so a profile with a picker of
    /// its own keeps the engine's decision path as it was.
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the profile run's inputs, as run_profile takes them"
    )]
    fn assign(
        &self,
        profile: &Profile,
        index: u16,
        assignment: Assignment,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        set: Set<'_>,
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let buf = &mut scratch.single;
        if let Some(answer) = single_survivors(profile, index, facts, pool, set, cx, buf) {
            return answer;
        }
        self.place(profile, index, assignment, facts, pool, set, cx, buf)
    }

    /// Whether a pin carries its rule's in-flight cap of the key's requests,
    /// by the count its row carries ([`Candidate::session_in_flight`]). A
    /// cap of zero is none.
    fn at_cap(rule: &SessionRule, pin: &Candidate) -> bool {
        rule.inflight_cap != 0 && pin.session_in_flight >= rule.inflight_cap
    }

    /// Drops the targets behind the first that carry the rule's cap of the
    /// key's requests. A commit writes the worker it lands on among the
    /// key's pins by the decision's verdict, and the cap bounds where a pin
    /// may go: an alternate at the cap is not offered. The first target
    /// stays, the worker the decision follows or places. A rule with no cap
    /// asks no walk.
    fn drop_alternates_at_cap(
        rule: &SessionRule,
        candidates: &[Candidate],
        kept: &[u32],
        targets: &mut Targets,
    ) {
        if rule.inflight_cap == 0 || targets.len() < 2 {
            return;
        }
        let placed = *targets;
        targets.clear();
        for (nth, target) in placed.iter().enumerate() {
            let capped = nth > 0
                && kept.iter().any(|&k| {
                    let c = &candidates[k as usize];
                    c.id == target.id && Self::at_cap(rule, c)
                });
            if !capped {
                targets.push(*target);
            }
        }
    }

    /// A pin at its rule's in-flight cap ([`Self::at_cap`]) puts the request
    /// to the stages again, over every survivor: the cap bounds where a pin
    /// may move, not where a request may go. A placement on `pin` again is
    /// the pin taken ([`SessionVerdict::Pinned`]), and the decision names no
    /// other target: at the cap a pin moves by a respill alone. One on
    /// another worker under the cap is marked [`SessionVerdict::Respill`].
    /// The live decision fronts that worker at once ([`Self::moved`]), and
    /// its commit fronts the worker it lands on, whichever of its targets:
    /// a respill's alternates are the workers under the cap alone. One on
    /// another worker at the cap carries no verdict: its commit moves no
    /// pin. `None` where no survivor stands. A pin the decision follows
    /// asks in its form's order ([`Self::hand_on`]).
    #[expect(
        clippy::too_many_arguments,
        reason = "the placement's inputs, as place takes them"
    )]
    fn respill(
        &self,
        profile: &Profile,
        index: u16,
        rule: &SessionRule,
        placing: Assignment,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        set: Set<'_>,
        cx: &Constraints<'_>,
        buf: &mut Buffers,
        pin: WorkerId,
    ) -> Option<Decision> {
        if buf.kept.is_empty() {
            return None;
        }
        let candidates = set.candidates;
        let mut decision = self.place(profile, index, placing, facts, pool, set, cx, buf);
        if let Decision::Pick(pick) = &mut decision {
            // As a key with no pin: the rule's own stage placed it.
            if placing != Assignment::Delegate {
                pick.targets.by_rule(rule.rule_placed);
            }
            let taken = pick.targets.first().map(|target| target.id);
            let under_the_cap = |id: WorkerId| {
                buf.kept.iter().any(|&k| {
                    let c = &candidates[k as usize];
                    c.id == id && !Self::at_cap(rule, c)
                })
            };
            pick.session_verdict = match taken {
                Some(id) if id == pin => SessionVerdict::Pinned,
                Some(id) if under_the_cap(id) => SessionVerdict::Respill,
                _ => SessionVerdict::None,
            };
            // A commit moves the pins by the decision's verdict, whichever of
            // its targets it lands on. So a respill names the workers under
            // the cap alone, and the pin taken again is the decision's one
            // target: at the cap a pin moves by a respill alone. A placement
            // with no verdict keeps its alternates: its commit moves no pin.
            match pick.session_verdict {
                SessionVerdict::Respill => {
                    Self::drop_alternates_at_cap(rule, candidates, &buf.kept, &mut pick.targets);
                }
                SessionVerdict::Pinned => {
                    let placed = pick.targets;
                    pick.targets.clear();
                    if let Some(first) = placed.first() {
                        pick.targets.push(*first);
                    }
                }
                _ => {}
            }
        }
        Some(decision)
    }

    /// A pin the decision follows hands the key on at its rule's cap, in the
    /// order of the decision's form: the pin a kept key's lifted pass lands
    /// on, and another request's claim that either form answers with. The
    /// plain form asks its survivors ([`Self::respill`]). The kept form
    /// reads the veto as it stands first, as the pin read first does, and
    /// the lifted pass's survivors after: a placement of theirs says the
    /// lift in its trace. `None` where neither holds a worker: the key stays
    /// on `pin`, the pin followed. Out of line: the hand-on is off a
    /// decision's common path.
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the profile run's inputs, as run_profile takes them"
    )]
    fn hand_on<const KEPT: bool>(
        &self,
        profile: &Profile,
        index: u16,
        rule: &SessionRule,
        placing: Assignment,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        set: Set<'_>,
        cx: &Constraints<'_>,
        buf: &mut Buffers,
        all_vetoed: bool,
        pin: WorkerId,
    ) -> Option<Decision> {
        if !KEPT {
            return self.respill(
                profile, index, rule, placing, facts, pool, set, cx, buf, pin,
            );
        }
        // The veto as it stands: that pass's survivors decide. The pass is
        // put to no rule.
        if !all_vetoed && filter(profile, facts, cx, buf, set.shedding()).is_none() {
            if let Some(respilled) = self.respill(
                profile, index, rule, placing, facts, pool, set, cx, buf, pin,
            ) {
                return Some(respilled);
            }
        }
        // It keeps nobody: the lifted pass's survivors decide, and the trace
        // says the lift.
        if vetoed::lifted(profile, index, facts, pool, set, cx, buf).is_none() {
            if let Some(mut respilled) = self.respill(
                profile, index, rule, placing, facts, pool, set, cx, buf, pin,
            ) {
                if let Decision::Pick(pick) = &mut respilled {
                    pick.trace.lifted = LiftedVeto::Overload;
                }
                return Some(respilled);
            }
        }
        None
    }

    /// The pool's own turn for a request a sessions rule leaves to the
    /// profile, where the profile's picker takes turns
    /// ([`Picker::takes_turns`]): the draw a request with no session value
    /// is decided by ([`run_drawn`]), each row asked that draw's own test,
    /// every filter of the profile, hard and soft alike, and taken where it
    /// is one of the survivors in `buf.kept` too. A new session and a
    /// session at its cap so take the next turn of the rotation such a
    /// request turns, and none of the survivors' own. The draw is asked
    /// where that request's is: from the draw's floor up ([`DRAW_FROM`]),
    /// and under it over a pool that lends its own word
    /// ([`PoolFacts::word`]). `None` where the
    /// profile's plain request is not the draw's, or the draw keeps nobody:
    /// the stages decide ([`score_and_pick`]), as the whole pass does behind
    /// that draw. So where a soft filter refuses every row, both kinds of
    /// request are refused at the draw and fall to the survivors' one
    /// rotation. `None` too over a set narrowed to a side of a pair
    /// ([`Set::plain`]), whose request with no session value never draws:
    /// its sessions rotate through the survivors, as that request does.
    ///
    /// The rows drawn stand in `buf.before_soft`, which nothing reads behind
    /// the filters: `buf.kept` stays the survivors, which the callers read.
    /// A picker that takes turns draws by its rotation and reads no value of
    /// the seed's stream, so the stages are handed the request's own seed.
    /// The profile is read here, by the index, as [`run_drawn`] reads it.
    fn pool_turn(
        &self,
        index: u16,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        set: Set<'_>,
        cx: &Constraints<'_>,
        buf: &mut Buffers,
    ) -> Option<Decision> {
        // The plain path's own test of a turn: the single plan's whole
        // pool, of a profile that draws for every request and takes turns,
        // over `DRAW_FROM` rows or more or over a pool that lends its own
        // word, and a request its host holds to no places. A side of a
        // pair takes no turn of a list: its request with no session value
        // never draws, so its new session rotates through the survivors
        // too. The set says which it is. A request's phase does not: a
        // binding of the single plan decides a request of any phase.
        let (at, candidates) = (usize::from(index), set.candidates);
        let sized = candidates.len() >= DRAW_FROM || pool.word.is_some();
        if !(set.plain() && sized && self.marks[at].turns() && !places::stated(cx)) {
            return None;
        }
        let (profile, tests) = (&self.profiles[at], self.row_tests[at]);
        let Buffers {
            kept,
            scores,
            stage,
            before_soft: drawn,
            ..
        } = buf;
        drawn.clear();
        // A row the draw takes passes two tests, and neither says the other.
        // The draw's own, as a request with no session value is asked it:
        // the profile's fold where the build made one, else a call a
        // filter. Where the pass put back the rows a soft filter refused
        // together, or a kept session's pass ran again with the overload
        // veto lifted, every survivor fails it: the draw keeps none, as
        // that request's does, and the stages place over the survivors.
        // And the row is one of the survivors. A filter may read what
        // another decision moves, so a row the pass refused can pass the
        // first test here, and the session's cap is read of the survivors
        // alone ([`Self::respill`]). The host's mask, which that request's
        // draw asks beside a fold that holds the health filter, is not
        // asked again beside the fold: that filter asked it in the pass,
        // under these constraints, and no row it kept is under the mask.
        // The survivors name a row once, so as many as the rows is every row.
        let all = kept.len() == candidates.len();
        let mut keeps = |row: &Candidate| {
            let passes = match tests {
                Some(tests) => tests.keeps(facts, row, cx),
                None => profile.filters.iter().all(|f| f.keep(facts, row, cx)),
            };
            let at = candidates.element_offset(row);
            passes && (all || at.is_some_and(|at| kept.contains(&(at as u32))))
        };
        let mut draw = Draw::new(facts.seed);
        let picker = &profile.picker;
        let drew = picker.draw_first(facts, pool, candidates, &mut keeps, drawn, &mut draw);
        // Asked on both arms: the stages behind a draw that keeps nobody are
        // handed the request's own seed too.
        debug_assert_eq!(draw.state(), facts.seed, "a turn reads no drawn value");
        if !drew {
            return None;
        }
        let mut targets = Targets::new();
        let band = combine(profile, facts, candidates, drawn, scores, stage);
        picker.order_drawn(facts, candidates, drawn, scores, band, &mut targets);
        if targets.is_empty() {
            return None;
        }
        // The pass ran before the draw: its survivors are counted.
        let mut trace = trace_drawn(index, facts, pool, candidates, drawn, scores);
        trace.survivors = Survivors::counted(kept.len() as u32);
        Some(Decision::Pick(Pick {
            targets,
            trace,
            session_verdict: SessionVerdict::None,
        }))
    }

    /// The stages of an assignment over the survivors in `buf.kept`: the
    /// profile's own scorers and picker under `delegate`, else the
    /// assignment's stage over the same survivors. Under `delegate` a
    /// picker that takes turns is asked for the pool's turn first
    /// ([`Self::pool_turn`]). The assignment is the
    /// one that places ([`placed_by`]): every caller hands the binding's
    /// placing assignment, which the engine's build wrote for a binding
    /// under a rule and for one under none. Who placed the pick
    /// ([`Targets::by_rule`]) is its caller's word: a rule's, for its key
    /// alone.
    #[expect(
        clippy::too_many_arguments,
        reason = "the profile run's inputs, as run_profile takes them"
    )]
    fn place(
        &self,
        profile: &Profile,
        index: u16,
        assignment: Assignment,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        set: Set<'_>,
        cx: &Constraints<'_>,
        buf: &mut Buffers,
    ) -> Decision {
        let candidates = set.candidates;
        let mut targets = Targets::new();
        let picker: &dyn Picker = match assignment {
            // The profile's own stages, through the profile run's one home.
            Assignment::Delegate => {
                if let Some(turned) = self.pool_turn(index, facts, pool, set, cx, buf) {
                    return turned;
                }
                score_and_pick(profile, facts, candidates, buf, &mut targets);
                profile.picker.as_ref()
            }
            Assignment::Random => {
                buf.scores.clear();
                buf.scores.resize(buf.kept.len(), 0.0);
                let picker: &dyn Picker = &self.assignment.random;
                picker.pick(
                    facts,
                    candidates,
                    &buf.kept,
                    &buf.scores,
                    TieBand::NORMALIZED,
                    &mut targets,
                );
                picker
            }
            Assignment::MinLoad | Assignment::MinGroup => {
                buf.scores.clear();
                buf.scores.resize(buf.kept.len(), 0.0);
                let scorer: &dyn Scorer = if assignment == Assignment::MinLoad {
                    &self.assignment.least_requests
                } else {
                    &self.assignment.fewest_sessions
                };
                let band = scorer.score(facts, candidates, &buf.kept, &mut buf.scores);
                let picker: &dyn Picker = &self.assignment.best;
                picker.pick(
                    facts,
                    candidates,
                    &buf.kept,
                    &buf.scores,
                    band,
                    &mut targets,
                );
                picker
            }
        };
        let trace = trace_of(index, facts, pool, candidates, &buf.kept, &buf.scores);
        if targets.is_empty() {
            return shed(ShedReason::PolicyDeclined, picker.name(), trace);
        }
        Decision::Pick(Pick {
            targets,
            trace,
            session_verdict: SessionVerdict::None,
        })
    }

    /// Whether a side of a pair is placed by its binding's sessions rule or
    /// by the assignment its own picker leaves the placement to, and not by
    /// the side's profile alone ([`Self::placed_side`]): a request with a
    /// session key under a rule, and every request of such a picker. The
    /// entry's byte is asked first ([`Entry::sides_placed`]): a pair under
    /// no rule, each side's picker its own, runs its two profiles as it
    /// did, behind that one test.
    #[inline(always)]
    fn places_side(&self, entry: &Entry, index: u16, facts: &RequestFacts<'_>) -> bool {
        entry.sides_placed
            && ((entry.binding.sessions.is_some() && !facts.session_key.is_none())
                || self.marks[usize::from(index)].by_assignment())
    }

    /// One side of a pair that its binding's rule or its picker's
    /// assignment places ([`Self::places_side`]), as the single plan places
    /// such a request: under the rule a request with a session key goes to
    /// its key's pin first, by the side's own session
    /// ([`Self::decide_with_session`]), and every other is placed by the
    /// assignment ([`Self::assign`]). Both run over the single plan's
    /// buffers: with `DECODE` the decode side's are swapped in for the run,
    /// so the prefill's stay as its run left them. A key's pin is tested
    /// with the overload veto as it stands: the form of a kept key
    /// ([`Self::keep`]) is the single plan's alone.
    ///
    /// Out of line: a pair under no rule holds nothing of it.
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the profile run's inputs, as run_profile takes them"
    )]
    fn placed_side<const DECODE: bool>(
        &self,
        binding: &Binding,
        index: u16,
        profile: &Profile,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        set: Set<'_>,
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        // The assignment that places, as the build writes it for a binding
        // of the single plan ([`placed_by`]): by the side's own picker.
        let declared = binding
            .sessions
            .as_ref()
            .map_or(Assignment::Delegate, |rule| rule.assignment);
        let placing = placed_by(declared, self.marks[usize::from(index)].by_assignment());
        if DECODE {
            std::mem::swap(&mut scratch.single, &mut scratch.decode);
        }
        let decision = match &binding.sessions {
            Some(rule) if !facts.session_key.is_none() => self
                .decide_with_session::<false, true, false>(
                    profile, index, rule, placing, facts, pool, set, cx, scratch,
                ),
            _ => self.assign(profile, index, placing, facts, pool, set, cx, scratch),
        };
        if DECODE {
            std::mem::swap(&mut scratch.single, &mut scratch.decode);
        }
        decision
    }

    /// The prefill side of a pair by its rule or its picker's assignment
    /// ([`Self::placed_side`]), over the prefills of the classes it is
    /// shown: the first partners' where `firsts`, else every open
    /// partner's. The rule's run takes the whole scratch, so the classes
    /// its set reads are lent for the run and stand in the scratch again
    /// behind it.
    ///
    /// Out of line: a pair whose prefill side its profile runs holds the
    /// test of it at each run and the two runs of its profile, as it did.
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the profile run's inputs, as run_profile takes them"
    )]
    fn placed_prefill(
        &self,
        firsts: bool,
        binding: &Binding,
        profile: &Profile,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        prefills: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let shown = if firsts {
            &mut scratch.firsts
        } else {
            &mut scratch.classes
        };
        let classes = std::mem::take(shown);
        let set = Set::side(prefills, None, Some(&classes));
        let index = binding.profile;
        let decision =
            self.placed_side::<false>(binding, index, profile, facts, pool, set, cx, scratch);
        if firsts {
            scratch.firsts = classes;
        } else {
            scratch.classes = classes;
        }
        decision
    }

    /// The `pd-pair` plan, and the pair half of `epd`: the prefill profile
    /// over `prefills`, then the decode profile over the decodes of the
    /// classes the chosen prefill's class pairs with ([`PairTable`]). A
    /// prefill whose partners leave no decode is passed over for the next
    /// prefill target; the targets list carries the chosen pair first and the
    /// alternate decodes of the same prefill after it. Pair class `0` is
    /// outside any pairing on either side: such a decode partners nobody and
    /// such a prefill is dropped at the `partnered` stage, so a host that
    /// leaves the classes unset sheds instead of pairing all with all. Each
    /// side's affinity column rides its pool ([`PairPools`]); the request
    /// carries none here and names no rows, and either is refused under
    /// the stage name `affinity-column`.
    ///
    /// A side whose profile shows its rows to a plugin is handed rows
    /// written for the decision ([`Self::reads_for`]). A mask beside that
    /// side's rows ([`Constraints::stalled`]) is declined under the stage
    /// name `written-rows` before either scope opens, as [`Self::decide`]
    /// declines it. A pair side carries no pin.
    ///
    /// Under the binding's sessions rule each side follows its key's pin by
    /// the side's own session, and a side whose picker leaves the placement
    /// to the rule is placed by the assignment. The pick states the prefill
    /// side's verdict, and its first target's pair the decode side's
    /// ([`Pair::decode_verdict`](super::decision::Pair::decode_verdict)).
    /// A claim of the prefill side names the prefill the pair formed on,
    /// before any dispatch, and a decision that forms no pair leaves none.
    ///
    /// The decode side is put to its binding's all-vetoed rule for the
    /// overload veto, as the pool of a single request is ([`Self::decide`]).
    /// Where no decode is open and some are over their thresholds alone, the
    /// rule lifts that veto once for the pool, and every decode the hard
    /// filters then keep opens its prefills. A prefill is chosen among them
    /// by its own profile, its partners go to the one of them that holds the
    /// fewest requests, and the pair's trace says the lifted veto
    /// ([`LiftedVeto::Overload`]). On a retry the profile's soft filter keeps
    /// that pick to the partners the request has not tried while one is
    /// left. Under a binding that sheds, such a pool is shed as saturated.
    /// The prefill side's own run is put to no rule.
    ///
    /// Where that run sheds, a pool whose every usable worker is vetoed
    /// goes to the pair's default: the usable prefill and partner of the
    /// least load, the pairs the overload veto alone refuses first, with the
    /// admission limit asked of the prefill taken. A prefill pool refused
    /// whole pairs so, and so does a decode pool the liveness veto refuses
    /// on every worker. Each side's workers are those at the places its
    /// host states.
    ///
    /// The prefill admission limit is judged over the prefills that still
    /// have a partner to go to. Where one of them stands but for its limit,
    /// the limit alone left the run nobody, and the request is shed under
    /// the limit's stage, on which its host waits for room. Outside the
    /// pair's default, a prefill at its limit with no open partner makes no
    /// request wait: its refusal is named by what its partners' own stages
    /// say, as the refusal of a prefill under its limit is.
    ///
    /// Where the host states first partners ([`PairPools::first`]), the
    /// pair is made among the prefills with a first partner open and those
    /// partners, while there is one. The prefill profile is shown those
    /// prefills alone, the admission limit is judged over them, and the
    /// decode is one of the pick's first partners. Where the profile's own
    /// stages refuse each such prefill, the pair is made among every
    /// prefill with an open partner, as it is where no first partner is
    /// open. The pair's default reads every partner alike. The gateway's
    /// own pair selection orders its candidates so
    /// (`model_gateway/src/routers/common/placement.rs:2479-2502`,
    /// `:2532-2545`, `:2615-2618`, and its default at `:1804-1813`, on main
    /// `30f636706`).
    pub fn decide_pair(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        cx: &Constraints<'_>,
        decode_cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let bare = Trace {
            seed: facts.seed,
            generation: pool.generation,
            ..Trace::default()
        };
        // A pair's columns ride its two pools. One on the request, or rows
        // named on it, name neither pool: refused, where they would be
        // dropped in silence and both sides scored as if nothing were held.
        if carries_a_column(facts) {
            return shed(ShedReason::PolicyDeclined, AFFINITY_COLUMN, bare);
        }
        // Each side's stages read their own pool and its own column.
        let (Some((prefills, prefill_facts)), Some((decodes, decode_facts))) = (
            pools.side(PairSide::Prefill, facts),
            pools.side(PairSide::Decode, facts),
        ) else {
            return shed(ShedReason::PolicyDeclined, AFFINITY_COLUMN, bare);
        };
        // One binding for both sides: the prefill's lookup, whose decode
        // profile is the pair's decode side. A binding of the single plan
        // declines a pair decision: the host and the spec disagree.
        let Some(entry) = self.pair_entry(facts.model) else {
            return shed(ShedReason::PolicyDeclined, "binding", bare);
        };
        // The binding's word for a pool a soft veto empties: the live
        // decode side is put to the all-vetoed rule by it ([`Set::ruled`]).
        let (binding, vetoed) = (&entry.binding, entry.vetoed);
        let decode_index = binding.decode_profile.unwrap_or(binding.profile);
        let (Some(prefill_profile), Some(decode_profile)) = (
            self.profiles.get(usize::from(binding.profile)),
            self.profiles.get(usize::from(decode_index)),
        ) else {
            return shed(ShedReason::PolicyDeclined, "binding", bare);
        };
        // A plugin reads a row's own fields: a side that shows its rows is
        // handed rows written for the decision. A mask beside that side's
        // rows says they are not, and the decision is declined before
        // either scope opens. A side that shows nothing reads no mask here.
        let prefill_shown = self.marks[usize::from(binding.profile)].scoped();
        let decode_shown = self.marks[usize::from(decode_index)].scoped();
        if (prefill_shown && !cx.stalled.is_empty())
            || (decode_shown && !decode_cx.stalled.is_empty())
        {
            // The side whose shown rows stand beside a mask is the one that
            // declines: the prefill's first.
            let prefill = prefill_shown && !cx.stalled.is_empty();
            let declining = if prefill {
                binding.profile
            } else {
                decode_index
            };
            return rows_not_written(bare, declining);
        }
        // One scope per side: the WIT view has one candidate list, so the
        // prefill stages read the prefill pool and the decode stages the
        // decode pool, through every decode run and the partner passes.
        let _prefill_opened =
            prefill_shown.then(|| prefill_profile.open(&prefill_facts, prefills, pool));
        let _decode_opened =
            decode_shown.then(|| decode_profile.open(&decode_facts, decodes, pool));
        // The prefill pool narrows to the prefills with a partner the decode
        // profile's hard filters keep, before the prefill picker truncates to
        // its targets: a pair that exists is found. A decode of class 0 is
        // outside any pairing and partners nobody.
        debug_assert!(
            pools.table.validate().is_ok(),
            "the pair table is not valid: {}",
            pools
                .table
                .validate()
                .map_or_else(|clause| clause.to_string(), |()| String::new())
        );
        // A decode outside its side's places partners nobody.
        let open = places::rows(decodes, decode_cx)
            .filter(|d| {
                d.pair_class != 0
                    && decode_profile
                        .filters
                        .iter()
                        .filter(|f| !f.soft())
                        .all(|f| f.keep(&decode_facts, d, decode_cx))
            })
            .map(|d| d.pair_class);
        pools
            .table
            .partnered(open, &mut scratch.classes, &mut scratch.marks);
        // No prefill class has an open decode. The decode pool is put to
        // the binding's all-vetoed rule once. Where the overload veto alone
        // left it nobody, every decode the hard filters keep with that veto
        // lifted opens its prefills. A soft filter of that pass hides no
        // class: a room narrows a prefill's partners, not the prefills. The
        // decode run of the chosen prefill then meets its partners emptied
        // the same way, and the rule picks among them. Decodes the rule
        // keeps past a doubt open no prefill: the prefill side sheds, and
        // that pool goes to the pair's default below.
        if scratch.classes.is_empty() {
            let whole = Set::side(decodes, None, None).ruled(vetoed);
            let buf = &mut scratch.decode;
            let emptied = filter(decode_profile, &decode_facts, decode_cx, buf, whole);
            let lifted = !buf.kept.is_empty() && buf.lifted == LiftedVeto::Overload;
            if emptied.is_some() && lifted {
                // The facts and the constraints of the rule's own pass
                // ([`vetoed::emptied`]).
                let all = RequestFacts {
                    tried: &[],
                    ..decode_facts
                };
                let pass = Constraints {
                    overload_lifted: true,
                    ..*decode_cx
                };
                let hard = || decode_profile.filters.iter().filter(|f| !f.soft());
                // Here too a decode outside its side's places partners
                // nobody: the rule's pass above opened with those places.
                // So does a decode of class 0.
                let kept = places::rows(decodes, decode_cx)
                    .filter(|d| d.pair_class != 0 && hard().all(|f| f.keep(&all, d, &pass)));
                pools.table.partnered(
                    kept.map(|d| d.pair_class),
                    &mut scratch.classes,
                    &mut scratch.marks,
                );
            }
        }
        // The prefill buffer's mark of a lifted veto is cleared for this
        // decision. The pair's default writes it again
        // ([`vetoed::paired`]), and so does a pick the weighing lift alone
        // made possible, below: the prefill's shadow reads it
        // ([`Self::shadow_side`]).
        scratch.single.lifted = LiftedVeto::None;
        // A prefill profile that weighs load against affinity runs with the
        // overload veto lifted ([`Self::weighing_lift`]): its run, the
        // admission limit's tests and the count of its routable prefills
        // read `run_cx`. The lifted constraints are a second value beside
        // the stated ones, which the pair's default still reads: it judges
        // the pool as it stands with no lift.
        let lifted_run = self.weighing_lift(binding.profile, vetoed, cx);
        let run_cx = lifted_run.as_ref().unwrap_or(cx);
        debug_assert!(
            pools.first.within(&pools.table).is_ok(),
            "the first partners do not stand within the pair table: {:?}",
            pools.first.within(&pools.table)
        );
        // While a prefill has a first partner open, the pair is made among
        // those prefills and their first partners ([`FirstPartners`]): the
        // prefill profile is shown their classes alone.
        let mut firsts = pools
            .first
            .open(&pools.table, &scratch.marks, &mut scratch.firsts);
        let shown: &[u32] = if firsts {
            &scratch.firsts
        } else {
            &scratch.classes
        };
        // The prefill side's run over the classes it is shown, the first
        // partners' or every open partner's: by its rule or its picker's
        // assignment where one places the side ([`Self::placed_prefill`]),
        // else by its profile. The classes are chosen ahead of that test,
        // where the first partners' answer stands, and the test is asked
        // again where the second run needs it. Chosen behind the test, the
        // classes were chosen by value, and a pair under no rule paid 9
        // instructions for it.
        let mut prefill = if self.places_side(entry, binding.profile, &prefill_facts) {
            self.placed_prefill(
                firsts,
                binding,
                prefill_profile,
                &prefill_facts,
                pool,
                prefills,
                run_cx,
                scratch,
            )
        } else {
            run_profile(
                prefill_profile,
                binding.profile,
                &prefill_facts,
                pool,
                Set::side(prefills, None, Some(shown)),
                run_cx,
                &mut scratch.single,
            )
        };
        if firsts {
            let stands = match &prefill {
                Decision::Pick(_) => true,
                // One of those prefills that stands but for its admission
                // limit keeps the request for them: it waits, though a
                // prefill with no first partner open has room. Where none
                // stands, the pairs with any open partner are next.
                Decision::Shed(refused) if refused.reason != ShedReason::PolicyDeclined => {
                    let takes = Half::new(
                        prefill_profile,
                        binding.profile,
                        &prefill_facts,
                        prefills,
                        run_cx,
                    );
                    let listed = |c: &&Candidate| classed(&scratch.firsts, c.pair_class);
                    let mut among = takes.places().filter(listed);
                    if among.any(|c| takes.but_for_the_limit(c)) {
                        let limit = HasCapacity.name();
                        return shed(reason_of(limit), limit, refused.trace);
                    }
                    false
                }
                // A picker's decline passes through.
                _ => return prefill,
            };
            if stands {
                // The classes the prefill profile was shown stand in
                // `classes`, as a side's shadow reads them.
                std::mem::swap(&mut scratch.classes, &mut scratch.firsts);
            } else {
                firsts = false;
                prefill = if self.places_side(entry, binding.profile, &prefill_facts) {
                    self.placed_prefill(
                        false,
                        binding,
                        prefill_profile,
                        &prefill_facts,
                        pool,
                        prefills,
                        run_cx,
                        scratch,
                    )
                } else {
                    run_profile(
                        prefill_profile,
                        binding.profile,
                        &prefill_facts,
                        pool,
                        Set::side(prefills, None, Some(&scratch.classes)),
                        run_cx,
                        &mut scratch.single,
                    )
                };
            }
        }
        // The prefill's pick is read where its decision stands: a pick is
        // 256 bytes, and taking it out copies them.
        let Decision::Pick(prefill_pick) = &prefill else {
            // A prefill pool its filters emptied says what the decode pool's
            // own stages say about the partners of the prefills that stand
            // but for their admission limit (those were dropped by the
            // partnered test alone): a saturated decode fleet is saturation,
            // not a missing endpoint, and a class with no decode at all is
            // "partnered", as is a routable prefill of class 0. When no
            // prefill stands so, the prefill side's stage is named instead.
            // The limit names the shed for a prefill with an open partner
            // alone. A picker's decline passes through.
            if let Decision::Shed(prefill_shed) = &prefill {
                if prefill_shed.reason != ShedReason::PolicyDeclined {
                    // A refused pool vetoed on every usable worker goes to
                    // the pair of the least load ([`vetoed::paired`]). The
                    // default reads the classes with an open partner, so it
                    // is asked before they are written over.
                    let (first, second) = (binding.profile, decode_index);
                    let sides = [
                        Half::new(prefill_profile, first, &prefill_facts, prefills, cx),
                        Half::new(decode_profile, second, &decode_facts, decodes, decode_cx),
                    ];
                    let weighs = self.profile_needs[usize::from(first)].affinity;
                    let words = (vetoed == AllVetoed::Steer, weighs);
                    if let Some(pair) = vetoed::paired(sides, &pools.table, words, pool, scratch) {
                        return pair;
                    }
                    // The admission limit is judged over the prefills with
                    // an open partner. Where one of them stands but for its
                    // limit, the limit alone left the run nobody: the shed
                    // is the limit's, and a host waits on it. Asked before
                    // the classes with an open partner are written over,
                    // of a side that reads the constraints the run itself
                    // read: the default above read the stated ones.
                    let takes = Half::new(prefill_profile, first, &prefill_facts, prefills, run_cx);
                    let open = |c: &&Candidate| classed(&scratch.classes, c.pair_class);
                    let mut partnered = takes.places().filter(open);
                    if partnered.any(|c| takes.but_for_the_limit(c)) {
                        let limit = HasCapacity.name();
                        return shed(reason_of(limit), limit, prefill_shed.trace);
                    }
                    let mut routable = 0usize;
                    scratch.classes.clear();
                    // Of the prefills the request may take, the limit set
                    // aside: one outside its side's places is no routable
                    // prefill of it, and one at its limit with no open
                    // partner is named by its partners' stage as any other.
                    // A prefill is routable under the constraints the run
                    // itself read.
                    for c in takes.places().filter(|c| takes.but_for_the_limit(c)) {
                        routable += 1;
                        if c.pair_class != 0 {
                            scratch.classes.push(c.pair_class);
                        }
                    }
                    scratch.classes.sort_unstable();
                    scratch.classes.dedup();
                    if routable == 0 {
                        let stage = filter(
                            prefill_profile,
                            &prefill_facts,
                            run_cx,
                            &mut scratch.single,
                            Set::side(prefills, None, None),
                        )
                        .unwrap_or("none");
                        return shed(reason_of(stage), stage, prefill_shed.trace);
                    }
                    if scratch.classes.is_empty() {
                        return shed(ShedReason::NoEndpoints, "partnered", prefill_shed.trace);
                    }
                    // Their partners are the decodes of the classes they
                    // pair with.
                    pools.table.listed(&mut scratch.classes, &mut scratch.marks);
                    // An empty decode pool is that class for every prefill:
                    // its "none" falls through to the prefill's shed.
                    if let Some(stage) = filter(
                        decode_profile,
                        &decode_facts,
                        decode_cx,
                        &mut scratch.decode,
                        Set::side(decodes, None, Some(&scratch.classes)),
                    )
                    .filter(|&stage| stage != "none")
                    {
                        return shed(reason_of(stage), stage, prefill_shed.trace);
                    }
                }
            }
            return prefill;
        };
        // A pick the weighing lift alone made possible says so, as a kept
        // key's pick does ([`Self::keep`]): where the veto as it stands
        // leaves the prefills nobody, the pair's trace names the lifted
        // overload veto. The prefill buffer takes the mark too, so the
        // prefill's shadow says none beside such a pair
        // ([`Self::shadow_side`]). A pick beside a prefill under its
        // thresholds carries no mark.
        let lift_placed = lifted_run.is_some() && {
            let shown = Set::side(prefills, None, Some(&scratch.classes));
            let buf = &mut scratch.single;
            vetoed::all_vetoed(prefill_profile, &prefill_facts, shown, cx, buf)
        };
        if lift_placed {
            scratch.single.lifted = LiftedVeto::Overload;
        }
        let mut last_shed: Option<Shed> = None;
        for prefill_target in prefill_pick.targets.iter() {
            // A target's position is the host's; a row outside the slice
            // costs that target its turn, never the process.
            let Some(class) = prefills
                .get(prefill_target.position as usize)
                .map(|c| c.pair_class)
            else {
                continue;
            };
            // The decode profile runs over the chosen prefill's partners
            // alone: the decodes of the classes its class pairs with. A
            // class is a per-candidate fact and the table the decision's
            // own, so no host callback is needed to narrow the set.
            let partners = pools.partners(&class, firsts);
            let partners = Set::side(decodes, Some(partners), None).ruled(vetoed);
            let mut decision = if self.places_side(entry, decode_index, &decode_facts) {
                self.placed_side::<true>(
                    binding,
                    decode_index,
                    decode_profile,
                    &decode_facts,
                    pool,
                    partners,
                    decode_cx,
                    scratch,
                )
            } else {
                run_profile(
                    decode_profile,
                    decode_index,
                    &decode_facts,
                    pool,
                    partners,
                    decode_cx,
                    &mut scratch.decode,
                )
            };
            match &mut decision {
                // The decode's pick becomes the pair's where it stands: its
                // targets name the prefill, its trace is the prefill's, and
                // that decision is the one returned.
                Decision::Pick(paired) => {
                    // Every field of a pick is named here, with no rest:
                    // a field added later fails this build until the pair
                    // says whose value it keeps.
                    let Pick {
                        targets,
                        trace,
                        session_verdict,
                    } = paired;
                    // Each side's own verdict: the decode side's in the
                    // first target's pair, the prefill side's as the pick's.
                    targets.pair_under(prefill_target, *session_verdict);
                    // The pair's trace is the prefill's, with the mark of
                    // the veto the rule lifted for the decode side, or of
                    // the overload veto the prefill's own run was placed
                    // past.
                    let lifted = trace.lifted;
                    *trace = prefill_pick.trace;
                    trace.lifted = match lifted {
                        LiftedVeto::None if lift_placed => LiftedVeto::Overload,
                        lifted => lifted,
                    };
                    *session_verdict = prefill_pick.session_verdict;
                    // The prefill side's run claimed the prefill it placed
                    // first. Where the pair formed on another, the claim
                    // moves to that one here, ahead of any dispatch: the
                    // key's next decision follows a prefill that pairs.
                    // Where the table moves none, the claim stands
                    // unconfirmed no longer or that prefill is a pin
                    // already: the pick states a pin's verdict, and its
                    // dispatch moves the pins as a pin's alternate does.
                    if let SessionVerdict::Claimed(claimed) = prefill_pick.session_verdict {
                        if claimed != prefill_target.id {
                            let (key, paired) = (prefill_facts.session(), prefill_target.id);
                            *session_verdict = self.sessions.move_claim(key, claimed, paired);
                        }
                    }
                    return decision;
                }
                Decision::Shed(shed) => last_shed = Some(*shed),
                Decision::Queue(_) => return decision,
            }
        }
        // No prefill paired: the claim the prefill side's run made is
        // withdrawn, and the key's next request is placed again.
        if let SessionVerdict::Claimed(claimed) = prefill_pick.session_verdict {
            let key = prefill_facts.session();
            self.sessions.move_claim(key, claimed, WorkerId(0));
        }
        match last_shed {
            Some(last) => Decision::Shed(last),
            None => shed(ShedReason::NoEndpoints, "partners", prefill_pick.trace),
        }
    }

    /// The constraints a pair's prefill profile at `index` runs under where
    /// it weighs load against affinity: `cx` with the overload veto lifted.
    /// `None` where the stated ones stand: under a binding that sheds
    /// (`rule`), and under a profile none of whose stages reads affinity.
    ///
    /// A profile whose stages read affinity is taken to weigh load against
    /// it itself: its own gate moves a request off a holder loaded well
    /// past the pool. The veto would move a session off the prefill that
    /// holds its prefix each time that prefill crosses a threshold, and
    /// under sustained load the prefills cross them in turn. So such a
    /// profile places among the prefills over their thresholds too, as the
    /// gateway's own pair selection leaves the veto off a prefill pool
    /// whose policy weighs so
    /// (`model_gateway/src/routers/common/placement.rs:2375-2394`, at main
    /// `55c35100d`). The lift is the constraint a kept key's run is given
    /// ([`Self::keep`]). A profile that reads no affinity keeps the veto.
    fn weighing_lift<'a>(
        &self,
        index: u16,
        rule: AllVetoed,
        cx: &Constraints<'a>,
    ) -> Option<Constraints<'a>> {
        let needs = self.profile_needs.get(usize::from(index));
        let weighs = rule == AllVetoed::Steer && needs.is_some_and(|needs| needs.affinity);
        // Every pair's decision asks, so the copy is made only where the
        // lift stands.
        if !weighs {
            return None;
        }
        Some(Constraints {
            overload_lifted: true,
            ..*cx
        })
    }

    /// Whether the pair binding at `at` names a shadow for `side` that
    /// the build found free to run beside the live profile
    /// ([`side_shadows`]).
    pub(crate) fn shadows_side(&self, at: u32, side: PairSide) -> bool {
        let sides = self.shadow_sides.get(at as usize);
        sides.is_some_and(|sides| sides[side.index()].is_some())
    }

    /// Whether the shadow the pair binding at `at` names for `side` is
    /// not asked: its profile would be handed another column than the
    /// live side's pool carries ([`side_reads_another_column`]). A host
    /// says such a side's shadow as skipped.
    pub(crate) fn side_apart(&self, at: u32, side: PairSide) -> bool {
        let sides = self.side_apart.get(at as usize);
        sides.is_some_and(|sides| sides[side.index()])
    }

    /// One side of a pair's decision, made once more with that side's
    /// shadow profile over the rows its live side decided over. The
    /// prefill's shadow reads the prefills the live prefill was shown,
    /// those with a partner: their classes stand in `scratch` as the
    /// pair's decision left them, so a caller hands the scratch that
    /// decision wrote. The decode's shadow reads the partners of the
    /// live pair's prefill, never of a prefill of its own. `at` is
    /// the pair binding's position ([`Self::shadow_for`]) and `live` the
    /// pair's pick, made over the same `pools`. The run writes the
    /// scratch and the shadow profile's stages, and nothing a commit
    /// reads.
    ///
    /// A side's shadow is handed the ring its request lends whatever its
    /// ring form: the rule of a single's shadow ([`Self::shadow`]) is not
    /// asked here. No host asks a pair's shadows yet. The change that asks
    /// them holds a side's shadow to that rule against both live sides'
    /// marks together, as [`Self::states_key_place`] reads them: a pair is
    /// lent one list of places for its two sides.
    ///
    /// `None` where the binding names no shadow for the side, where the
    /// build did not find that profile free to run beside the live one
    /// ([`side_shadows`]), and where it would be handed another column
    /// than the live side's pool carries ([`Self::side_apart`]). `None`
    /// too beside a pair a lifted veto placed: a shadow's run is put to no
    /// rule, so it sheds a pool its live side was steered in, and decides
    /// by its profile where the live side was taken by load. The decode's
    /// shadow says none beside a pair whose trace names a lifted veto, by
    /// the binding's all-vetoed rule or by the pair's default
    /// ([`vetoed::paired`]). The prefill's shadow says none beside a pair
    /// the default made, which the mark on the prefill's buffer says: it
    /// speaks beside a pair whose decode side alone the rule steered.
    #[expect(
        clippy::too_many_arguments,
        reason = "the pair's entry and its inputs, beside the side and the live pick"
    )]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "no host asks a pair's shadows until the engine over worker lists does"
        )
    )]
    pub(crate) fn shadow_side(
        &self,
        at: u32,
        side: PairSide,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        cx: &Constraints<'_>,
        decode_cx: &Constraints<'_>,
        live: &Pick,
        scratch: &mut Scratch,
    ) -> Option<Decision> {
        let index = self.shadow_sides.get(at as usize)?[side.index()]?;
        if self.side_apart(at, side) {
            return None;
        }
        let profile = self.profiles.get(usize::from(index))?;
        let (prefills, prefill_facts) = pools.side(PairSide::Prefill, facts)?;
        let (decodes, decode_facts) = pools.side(PairSide::Decode, facts)?;
        Some(match side {
            PairSide::Prefill => {
                if scratch.single.lifted != LiftedVeto::None {
                    return None;
                }
                // The prefills the live prefill profile was shown: the
                // classes with a partner stand in the scratch as the
                // pair's decision left them ([`Self::decide_pair`]). No
                // stage of a live profile is asked again. Its decision's
                // scope closed with it, and a stage asked behind it
                // answers by its fallback. The shadow's run is lifted by
                // its own profile's word, as the live run is by the live
                // profile's ([`Self::weighing_lift`]).
                let rule = self.bindings.get(at as usize)?.vetoed;
                let lifted_run = self.weighing_lift(index, rule, cx);
                run_profile(
                    profile,
                    index,
                    &prefill_facts,
                    pool,
                    Set::side(prefills, None, Some(&scratch.classes)),
                    lifted_run.as_ref().unwrap_or(cx),
                    &mut scratch.single,
                )
            }
            PairSide::Decode => {
                if live.trace.lifted != LiftedVeto::None {
                    return None;
                }
                // The partners of the live pair's prefill.
                let prefill = live.targets.first()?;
                let class = prefills.get(prefill.position as usize)?.pair_class;
                let firsts = pools.made_first(class, live);
                run_profile(
                    profile,
                    index,
                    &decode_facts,
                    pool,
                    Set::side(decodes, Some(pools.partners(&class, firsts)), None),
                    decode_cx,
                    &mut scratch.decode,
                )
            }
        })
    }

    /// The decode side of a decode decided after its prefill
    /// ([`Self::decide_decode`]), made once more with the side's shadow
    /// profile over the rows that call decided over: the decodes that
    /// pair with the held `prefill`, on the call's own draw sequence.
    /// `at` is the pair binding's position ([`Self::pair_position`]). The
    /// run writes the scratch and the shadow profile's stages, and
    /// nothing a commit reads. The shadow is handed the ring its request
    /// lends as [`Self::shadow_side`] hands it, under the same word for
    /// the change that asks it.
    ///
    /// `None` where [`Self::shadow_side`] says none for the decode side,
    /// and where the held prefill left the rows. The same beside a decode
    /// the binding's all-vetoed rule placed, which the trace of `live`,
    /// that call's pick, says.
    #[expect(
        clippy::too_many_arguments,
        reason = "the decode entry and its inputs, beside the binding"
    )]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "no host asks a later decode's shadow until the engine over worker lists does"
        )
    )]
    pub(crate) fn shadow_decode_later(
        &self,
        at: u32,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        prefill: WorkerId,
        live: &Pick,
        decode_cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Option<Decision> {
        let side = PairSide::Decode;
        let index = self.shadow_sides.get(at as usize)?[side.index()]?;
        if self.side_apart(at, side) || live.trace.lifted != LiftedVeto::None {
            return None;
        }
        let profile = self.profiles.get(usize::from(index))?;
        let (decodes, mut decode_facts) = pools.side(side, facts)?;
        let prefill = pools.prefills.iter().find(|row| row.id == prefill)?;
        // The call's own draw sequence, as its live decode's.
        decode_facts.seed = facts.seed;
        let class = prefill.pair_class;
        let firsts = pools.made_first(class, live);
        Some(run_profile(
            profile,
            index,
            &decode_facts,
            pool,
            Set::side(decodes, Some(pools.partners(&class, firsts)), None),
            decode_cx,
            &mut scratch.decode,
        ))
    }

    /// Decides the decode side for a held prefill. The model's pair binding
    /// runs its decode profile over that prefill's partners in `pools`.
    /// Each picked target names the held prefill and one of those decodes.
    /// A missing prefill is shed under `prefill-gone`, whatever the call's
    /// constraints state: it is looked for before they are read. A shed
    /// made once the decode profile resolves names that profile. `pool`
    /// holds the facts of the decode pool, where the profile places. The
    /// decodes are held to the places `decode_cx` states. The held
    /// prefill is found by its id, in or outside any places. Partners that
    /// are all over their thresholds, or doubted, are put to the binding's
    /// all-vetoed rule, both its tiers: the rule picks afresh among them,
    /// whatever decode the first decision named. Over partners all over
    /// their thresholds the profile's soft filter keeps that pick to the
    /// ones the request has not tried while one is left, as the rule's
    /// pick in a first decision does. A pair's first decision lifts the
    /// overload veto alone by the rule, and a pool it still refuses goes to
    /// the pair's default, which asks no soft filter
    /// ([`Self::decide_pair`]). Here the prefill has run, and a refusal
    /// fails its request.
    ///
    /// Where the host states first partners ([`PairPools::first`]), the
    /// partners are the held prefill's first partners while one of them is
    /// open at this call's instant, by the test a first decision makes, and
    /// every partner of the prefill else.
    ///
    /// `first_decode` is the worker and the rank of the decode the pair's
    /// first decision named, where the caller holds them. Under a decode
    /// profile that holds no scorer and whose picker takes turns, the call
    /// answers that decode first while the profile's filters keep it, the
    /// decodes they keep behind it as its alternates, and the picker is not
    /// asked: a second pick would turn a rotation a second time for one
    /// request, and move the request for no reason. A scorer moves no pick
    /// of a picker that takes turns and reads no score
    /// ([`Picker::picks_alone`]): a profile that holds one beside such a
    /// picker keeps that decode as well. Its scorers read the decodes the
    /// filters keep, as ahead of a pick, and the trace holds their scores.
    /// Where a filter refuses that decode, or none of the partners is that
    /// worker, the picker picks among the decodes the filters keep. So it
    /// does under every other profile, and for a caller that hands none.
    #[expect(
        clippy::too_many_arguments,
        reason = "the request and the pair's pools, beside the pair the call is held by"
    )]
    pub fn decide_decode(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        prefill: WorkerId,
        first_decode: Option<(WorkerId, u16)>,
        decode_cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let bare = Trace {
            seed: facts.seed,
            generation: pool.generation,
            ..Trace::default()
        };
        if carries_a_column(facts) {
            return shed(ShedReason::PolicyDeclined, AFFINITY_COLUMN, bare);
        }
        let Some((decodes, mut decode_facts)) = pools.side(PairSide::Decode, facts) else {
            return shed(ShedReason::PolicyDeclined, AFFINITY_COLUMN, bare);
        };
        let Some(entry) = self.pair_entry(facts.model) else {
            return shed(ShedReason::PolicyDeclined, "binding", bare);
        };
        let binding = &entry.binding;
        let index = binding.decode_profile.unwrap_or(binding.profile);
        let Some(profile) = self.profiles.get(usize::from(index)) else {
            return shed(ShedReason::PolicyDeclined, "binding", bare);
        };
        // The held prefill is looked for before any constraint is read: a
        // call whose prefill left the lists says so, whatever it states.
        let Some(prefill) = pools.prefills.iter().find(|row| row.id == prefill) else {
            // The decode profile is resolved, so the shed names it.
            let under = Trace {
                profile: index,
                ..bare
            };
            return shed(ShedReason::NoEndpoints, "prefill-gone", under);
        };
        let shown = self.marks[usize::from(index)].scoped();
        if shown && !decode_cx.stalled.is_empty() {
            return rows_not_written(bare, index);
        }
        // This call has one draw sequence, seeded by its own request.
        decode_facts.seed = facts.seed;
        let _opened = shown.then(|| profile.open(&decode_facts, decodes, pool));
        debug_assert!(
            pools.first.within(&pools.table).is_ok(),
            "the first partners do not stand within the pair table: {:?}",
            pools.first.within(&pools.table)
        );
        // The held prefill's partners are put to the binding's all-vetoed
        // rule, its liveness tier included. They are its first partners
        // while one of those is open at this instant, by the test a pair's
        // first decision makes ([`FirstPartners`]): one the profile's hard
        // filters keep, or, where they keep no partner of the prefill, one
        // they keep with the overload veto lifted as the rule lifts it.
        // Else every partner.
        let class = prefill.pair_class;
        let first = pools.first.of(class);
        let firsts = !first.is_empty() && {
            let hard = || profile.filters.iter().filter(|f| !f.soft());
            let open = |among: &[u32], facts: &RequestFacts<'_>, cx: &Constraints<'_>| {
                let kept = |d: &Candidate| hard().all(|f| f.keep(facts, d, cx));
                places::rows(decodes, decode_cx).any(|d| classed(among, d.pair_class) && kept(d))
            };
            // The facts and the constraints of the rule's own pass
            // ([`vetoed::emptied`]).
            let all = RequestFacts {
                tried: &[],
                ..decode_facts
            };
            let pass = Constraints {
                overload_lifted: true,
                ..*decode_cx
            };
            open(first, &decode_facts, decode_cx)
                || (!open(pools.table.of(&class), &decode_facts, decode_cx)
                    && open(first, &all, &pass))
        };
        let partners = pools.partners(&class, firsts);
        let partners = Set::side(decodes, Some(partners), None).ruled(entry.vetoed);
        let mut decision = if self.places_side(entry, index, &decode_facts) {
            // The rule's run reads no `first_decode`: a session's first
            // decision claimed its decode in the session table, and the
            // key's pin answers this call.
            self.placed_side::<true>(
                binding,
                index,
                profile,
                &decode_facts,
                pool,
                partners,
                decode_cx,
                scratch,
            )
        } else {
            run_side(
                profile,
                index,
                &decode_facts,
                pool,
                partners,
                decode_cx,
                &mut scratch.decode,
                first_decode.filter(|_| {
                    profile.picker.takes_turns()
                        && (profile.scorers.is_empty() || profile.picker.picks_alone())
                }),
            )
        };
        // Each target names the held prefill with one decode, as the
        // targets of a pair's first decision do. They are turned where the
        // decision stands, and its trace stays the decode's: taken out and
        // built beside it, the pick was copied twice. The decode side's
        // verdict goes into the first target's pair, and the pick states
        // none: this call decided no prefill.
        if let Decision::Pick(pick) = &mut decision {
            let decode = std::mem::take(&mut pick.session_verdict);
            pick.targets.pair_under(&target_of(prefill), decode);
        }
        decision
    }
}

#[inline(never)]
fn all_reports_current(candidates: &[Candidate], now_micros: u64, max_age_ms: u64) -> bool {
    let cutoff = LoadReportCutoff::new(now_micros, max_age_ms);
    candidates.iter().all(|row| cutoff.load(row).is_some())
}

impl Engine {
    /// The session table's part of a dispatch ([`Lifecycle::on_dispatched`]): the
    /// session pin follows the dispatched worker of the side, the decode
    /// of a pair target on the decode side and the target itself
    /// otherwise. The controller tells the table first and the stages last
    /// ([`Self::dispatched_stages`]): a stage's call that unwinds leaves
    /// the table's entry written. A host tells a dispatch whole, by the
    /// trait.
    pub(crate) fn dispatched_table(&self, lease: &Lease, target: &Target, side: Side) {
        let dispatched = match (side, target.pair) {
            (Side::Decode, Some(pair)) => pair.decode_id,
            _ => target.id,
        };
        if lease.session_verdict != SessionVerdict::None {
            self.sessions
                .dispatched(lease.session, lease.session_verdict, dispatched);
        }
    }

    /// Records a dispatch in each profile tally that stages read. A tally
    /// asks `present` only where the send would enter a rank it does not
    /// know: a served controller asks its worker objects there, so a late
    /// report enters no rank behind a removal. An engine whose profiles
    /// read no sends asks nothing.
    #[inline]
    pub(crate) fn dispatched_sent(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
        present: impl Fn() -> bool,
    ) {
        match self.sent_credits.as_slice() {
            [] => {}
            // One count is the common document: its record runs in line
            // behind the caller's telling.
            [one] => one.dispatched(lease, target, side, since_decision_micros, present),
            _ => self.credit_dispatch(lease, target, side, since_decision_micros, present),
        }
    }

    /// Out of line: a document of several counts is rare. One count is
    /// told in line ([`Self::dispatched_sent`]).
    #[inline(never)]
    fn credit_dispatch(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
        present: impl Fn() -> bool,
    ) {
        for credit in &self.sent_credits {
            credit.dispatched(lease, target, side, since_decision_micros, &present);
        }
    }

    /// The stages' part of a dispatch: each profile's scorers and its
    /// picker hear it.
    pub(crate) fn dispatched_stages(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        for profile in &self.profiles {
            for w in &profile.scorers {
                w.scorer
                    .on_dispatched(lease, target, side, since_decision_micros);
            }
            profile
                .picker
                .on_dispatched(lease, target, side, since_decision_micros);
        }
    }
}

impl Engine {
    /// Tells a host's reports, each as the host shows it to decisions, to
    /// every stage that keeps a number of the reports
    /// ([`Lifecycle::keeps_loads`]), and to no other stage. `reports` is
    /// run once for each such stage and hands every report to the call it
    /// is given. An engine with no such stage never runs it: a publication
    /// that tells nobody walks no row. A plugin that watches signals hears
    /// of a world's reports no more than it did before a stage kept any.
    pub(crate) fn tell_loads(&self, reports: impl Fn(&mut dyn FnMut(&Signal<'_>))) {
        for keeper in &self.keepers {
            match *keeper {
                Keeper::Scorer { profile, stage } => {
                    let scorer = &self.profiles[usize::from(profile)].scorers[usize::from(stage)];
                    reports(&mut |report| scorer.scorer.on_signal(report));
                }
                Keeper::Picker { profile } => {
                    let picker = &self.profiles[usize::from(profile)].picker;
                    reports(&mut |report| picker.on_signal(report));
                }
            }
        }
    }

    /// The stages' part of a change of membership
    /// ([`Lifecycle::on_membership`]): the tallies of sends and each
    /// profile's scorers and its picker hear it, and the session table does
    /// not. The engine over a host's worker lists tells this part alone: a
    /// leave there takes no pin away. A session's pin stands on a worker
    /// that is gone until the session's idle time or two other workers end
    /// it, and is a routable pin again once a worker stands under its id.
    pub(crate) fn membership_stages(&self, change: &Membership<'_>) {
        for credit in &self.sent_credits {
            credit.membership(change);
        }
        for profile in &self.profiles {
            for w in &profile.scorers {
                w.scorer.on_membership(change);
            }
            profile.picker.on_membership(change);
        }
    }
}

impl Lifecycle for Engine {
    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        for profile in &self.profiles {
            for w in &profile.scorers {
                w.scorer.on_decision_issued(lease, decision);
            }
            profile.picker.on_decision_issued(lease, decision);
        }
    }

    #[inline(always)]
    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        self.dispatched_table(lease, target, side);
        self.dispatched_sent(lease, target, side, since_decision_micros, || true);
        self.dispatched_stages(lease, target, side, since_decision_micros);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        for profile in &self.profiles {
            for w in &profile.scorers {
                w.scorer.on_completed(lease, outcome);
            }
            profile.picker.on_completed(lease, outcome);
        }
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        for profile in &self.profiles {
            for w in &profile.scorers {
                w.scorer.on_released(lease, reason);
            }
            profile.picker.on_released(lease, reason);
        }
    }

    fn on_membership(&self, change: &Membership<'_>) {
        match change {
            // The pins of a worker that left stand for the absence's
            // time: its key may be registered again, and a host that
            // knows says which way ([`Engine::pins_follow`],
            // [`Engine::pins_forget`]).
            Membership::Removed(worker) => self.sessions.left(*worker),
            Membership::Replaced { old, new } => {
                self.sessions.left(*old);
                self.sessions.returned(*new);
            }
            // A join under an id that left ends its absence, whoever tells
            // it: the worker is back as it was, and its pins stand.
            Membership::Added(worker) => self.sessions.returned(*worker),
            Membership::Snapshot(workers) => self.sessions.returned_all(workers),
            _ => {}
        }
        self.membership_stages(change);
    }

    fn on_signal(&self, signal: &Signal<'_>) {
        for profile in &self.profiles {
            for w in &profile.scorers {
                w.scorer.on_signal(signal);
            }
            profile.picker.on_signal(signal);
        }
    }

    fn keeps_loads(&self) -> bool {
        !self.keepers.is_empty()
    }

    fn needs_lifecycle(&self) -> bool {
        self.profiles.iter().any(|profile| {
            profile.picker.needs_lifecycle()
                || profile.scorers.iter().any(|w| w.scorer.needs_lifecycle())
        })
    }

    /// The words the document takes: a stage's, the session table's at a
    /// dispatch under a sessions rule, the record of dispatched prefixes
    /// at an issue, and a count of sends at a dispatch. Folded at the
    /// build: one read here.
    fn hears(&self) -> EventSet {
        if self.sent_credits.is_empty() {
            self.hears
        } else {
            self.hears.with(EventKind::Dispatched)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{mpsc, Arc, Mutex, Once},
        time::Duration,
    };

    use super::*;
    use crate::{
        document::spec::{self, PolicySpec, Recovery, RulePlaced},
        engine::{
            decision::{self, Pair},
            sticky_sessions::DecodeVerdict,
            view::{self, Health, Identity, MediaCounts, Role, Session, WorkerId},
        },
        policies::{
            manual::Manual,
            passthrough::Passthrough,
            power_of_two::{PowerOfTwo, PowerOfTwoCountFirst, PowerOfTwoMetric},
            random::Random,
            round_robin::RoundRobin,
        },
        stages::{
            filters::{HasRoom, Healthy, NotOverloaded, NotTried},
            pickers::{self, Best, CountFirst},
            scorers::{self, LeastRequests},
        },
        tracking::in_flight::tests::Lent,
    };

    fn candidate(position: u32, in_flight: u32) -> Candidate {
        Candidate {
            position,
            id: WorkerId(100 + u64::from(position)),
            key: 100 + position,
            health: Health {
                healthy: true,
                ..Health::default()
            },
            in_flight,
            ..Candidate::default()
        }
    }

    fn least_requests_profile() -> Profile {
        Profile {
            name: "least".into(),
            filters: vec![Box::new(Healthy), Box::new(NotTried)],
            scorers: vec![WeightedScorer {
                scorer: Box::new(LeastRequests),
                weight: 1.0,
            }],
            picker: Box::new(Best),
        }
    }

    fn engine() -> Engine {
        Engine::new(
            vec![least_requests_profile()],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        )
    }

    fn with_shadow(shadow: Profile, sessions: Option<SessionRule>) -> Engine {
        Engine::new(
            vec![least_requests_profile(), shadow],
            vec![Binding {
                shadow: Some(1),
                sessions,
                ..single_binding()
            }],
            23,
            Needs::default(),
        )
    }

    fn by_sessions() -> Profile {
        Profile {
            name: "by-sessions".into(),
            scorers: vec![WeightedScorer {
                scorer: Box::new(FewestSessions),
                weight: 1.0,
            }],
            ..least_requests_profile()
        }
    }

    /// A keyed request of model `m` at `now`, its key's counts held.
    fn keyed_at(key: u64, now: u64) -> RequestFacts<'static> {
        RequestFacts {
            model: "m",
            session_key: Identity::hashed(key),
            session_counted: true,
            now_micros: now,
            ..RequestFacts::default()
        }
    }

    /// The worker of a decision's first target.
    fn first_of(decision: &Decision) -> Option<WorkerId> {
        decision
            .pick()
            .and_then(|pick| pick.targets.first())
            .map(|target| target.id)
    }

    /// Dispatch the target at `at` of a live decision, as a host commits it.
    fn commit(engine: &Engine, facts: &RequestFacts<'_>, live: &Decision, at: usize) -> WorkerId {
        let pick = live.pick().expect("a pick");
        let target = pick.targets.as_slice()[at];
        let lease = Lease {
            session: facts.session(),
            session_verdict: pick.session_verdict,
            issued_micros: facts.now_micros,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &target, Side::Single, 0);
        target.id
    }

    /// A live profile that reads no ring states no place on one, and its
    /// shadow is apart from none: a ring the request's caller lends is the
    /// shadow's, under the plain ring and under the prefix form alike. The
    /// shadow's first target is the worker that ring reaches first,
    /// whichever it is.
    #[test]
    fn a_shadow_beside_a_live_profile_that_reads_no_ring_keeps_the_lent_ring() {
        use crate::policies::{
            cache_aware::Gates,
            consistent_hashing::{ConsistentHash, InOrder},
        };

        let mut rows = [candidate(0, 3), candidate(1, 5), candidate(2, 9)];
        for (at, row) in rows.iter_mut().enumerate() {
            row.id = WorkerId::from_address(&format!("w{at}"));
        }
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        for prefixed in [false, true] {
            let ring = ConsistentHash::new(None);
            let ring = if prefixed {
                ring.keyed_by_prefix(256, Gates::default(), &[])
            } else {
                ring
            };
            let engine = with_shadow(
                Profile {
                    name: "ring".into(),
                    scorers: vec![WeightedScorer {
                        scorer: Box::new(ring),
                        weight: 1.0,
                    }],
                    ..least_requests_profile()
                },
                None,
            );
            for first in [2usize, 0] {
                let order = [first, 1, 2 - first].map(|at| format!("w{at}"));
                let lent = InOrder(order.to_vec());
                let facts = RequestFacts {
                    model: "m",
                    session_key: Identity::hashed(9),
                    ring: Some(&lent),
                    ring_positions: &[0],
                    ..RequestFacts::default()
                };
                let shadow = engine
                    .shadow(&facts, &pool, &rows, &cx, &mut scratch)
                    .expect("a shadow decision");
                assert_eq!(
                    first_of(&shadow),
                    Some(rows[first].id),
                    "prefixed: {prefixed}"
                );
            }
        }
    }

    /// One profile whose two ring scorers cut a request by two rules has no
    /// one list of heads' places: the engine states no rule for it, and a
    /// request under it is lent no ring. Two scorers of one rule state that
    /// rule.
    #[test]
    fn a_profile_whose_ring_scorers_name_two_head_rules_states_none() {
        use crate::policies::{cache_aware::Gates, consistent_hashing::ConsistentHash};

        let depth_of = |depths: [u32; 2]| {
            let scorers = depths.map(|depth| {
                let ring = ConsistentHash::new(None);
                WeightedScorer {
                    scorer: Box::new(ring.keyed_by_prefix(depth, Gates::default(), &[])),
                    weight: 1.0,
                }
            });
            let profile = Profile {
                name: "ring".into(),
                scorers: scorers.into(),
                ..least_requests_profile()
            };
            let engine = Engine::new(vec![profile], vec![single_binding()], 1, Needs::default());
            let rule = engine.states_head_places("m", Phase::Single);
            rule.map(|rule| rule.depth)
        };
        assert_eq!(depth_of([256, 256]), Some(256));
        assert_eq!(depth_of([256, 128]), None);
    }

    #[test]
    fn shadow_picks_with_its_own_profile_over_the_same_rows() {
        let engine = with_shadow(by_sessions(), None);
        let mut busy = candidate(1, 3);
        busy.active_sessions = 17;
        let rows = [candidate(0, 13), busy];
        let (facts, pool, cx) = (
            RequestFacts::default(),
            PoolFacts::default(),
            Constraints::default(),
        );
        let mut scratch = Scratch::new();
        let live = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        let first = |decision: &Decision| {
            decision
                .pick()
                .and_then(|pick| pick.targets.first())
                .map(|t| t.id)
        };
        assert_eq!(first(&live), Some(rows[1].id));
        assert_eq!(first(&shadow), Some(rows[0].id));
        assert_eq!(shadow.trace().profile, 1);
        assert_eq!(
            decision::ShadowKind::of(&live, &shadow),
            decision::ShadowKind::OtherTarget
        );

        let malformed = RequestFacts {
            affinity: &[AffinityFacts::default()],
            ..facts
        };
        assert_eq!(
            engine.shadow(&malformed, &pool, &rows, &cx, &mut scratch),
            Some(engine.decide(&malformed, &pool, &rows, &cx, &mut scratch))
        );
    }

    /// A shadow is lent no word of the pool. Where the live profile and its
    /// shadow both rotate, the live decisions take the list's rows one after
    /// the other, and the word counts them alone, whatever the shadows
    /// between them take.
    #[test]
    fn a_shadow_turns_no_rotation_of_the_live_list() {
        use std::sync::atomic::Ordering;

        let engine = Engine::new(
            vec![round_robin_profile(), round_robin_profile()],
            vec![Binding {
                shadow: Some(1),
                ..single_binding()
            }],
            23,
            Needs::default(),
        );
        let rows = pool_of(DRAW_FROM as u32, |_| false);
        let word = AtomicU64::new(3);
        let pool = PoolFacts {
            word: Some(&word),
            ..PoolFacts::default()
        };
        let (facts, cx) = (request(0), Constraints::default());
        let mut scratch = Scratch::new();
        let mut lives = Vec::new();
        for _ in 0..6 {
            let live = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
            lives.push(positions(&live)[0]);
            let shadow = engine.shadow(&facts, &pool, &rows, &cx, &mut scratch);
            assert!(shadow.expect("a shadow decision").pick().is_some());
        }
        assert_eq!(lives, [3, 4, 5, 6, 7, 8]);
        assert_eq!(word.load(Ordering::Relaxed), 9);
    }

    #[test]
    fn shadow_uses_the_bindings_all_vetoed_rule() {
        let profile = Profile {
            filters: vec![
                Box::new(Healthy),
                Box::new(NotOverloaded),
                Box::new(NotTried),
            ],
            ..least_requests_profile()
        };
        let mut engine = with_shadow(profile, None);
        let mut rows = [candidate(0, 29), candidate(1, 7)];
        for row in &mut rows {
            row.health.overloaded = true;
        }
        let (facts, pool, cx) = (
            RequestFacts::default(),
            PoolFacts::default(),
            Constraints::default(),
        );
        let mut scratch = Scratch::new();
        let steered = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("the binding shadows");
        assert!(
            matches!(steered, Decision::Pick(pick) if pick.trace.lifted == LiftedVeto::Overload)
        );
        engine.shed_when_all_vetoed(0);
        let shed = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("the binding shadows");
        assert!(matches!(shed, Decision::Shed(shed) if shed.reason == ShedReason::Saturated));
    }

    #[test]
    fn shadow_places_by_its_pickers_assignment() {
        let engine = with_shadow(
            Profile {
                scorers: vec![],
                picker: Box::new(Manual),
                ..least_requests_profile()
            },
            Some(SESSIONS),
        );
        let rows = [candidate(0, 21), candidate(1, 6)];
        let decision = engine
            .shadow(
                &RequestFacts::default(),
                &PoolFacts::default(),
                &rows,
                &Constraints::default(),
                &mut Scratch::new(),
            )
            .expect("the binding shadows");
        assert_eq!(
            decision
                .pick()
                .and_then(|pick| pick.targets.first())
                .map(|t| t.id),
            Some(rows[1].id)
        );
    }

    #[test]
    fn shadow_sampling_draws_from_the_seed_and_leaves_the_live_draw_alone() {
        let mut document = PolicySpec::preset("random").expect("a preset");
        let profile = document.profiles["random"].clone();
        document.profiles.insert("second".into(), profile);
        document.bindings[0].shadow = Some(spec::ShadowSpec::Profile("second".into()));
        let engine = document.validate().expect("a shadow binding");
        let control = document.validate().expect("the same binding");
        let rows: Vec<_> = (0..23).map(|row| candidate(row, row)).collect();
        let facts = RequestFacts {
            seed: 137,
            ..RequestFacts::default()
        };
        let pool = PoolFacts::default();
        let cx = Constraints::default();
        let mut scratch = Scratch::new();
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow draw");
        assert!(shadow.trace().drew_first());
        let again = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("the same shadow draw");
        assert_eq!(shadow, again);
        let without = control.decide(&facts, &pool, &rows, &cx, &mut Scratch::new());
        assert_eq!(
            shadow.pick().map(|pick| pick.targets),
            without.pick().map(|pick| pick.targets)
        );
        assert_eq!(
            engine.decide(&facts, &pool, &rows, &cx, &mut scratch),
            without
        );
    }

    /// A kept key's shadow is decided over the binding's own set, as the
    /// live decision is. Over a pool the liveness doubt leaves nobody it
    /// goes to the fewest in flight by the lift of that veto, under a
    /// sessions rule and under none, and claims no pin.
    #[test]
    fn shadow_puts_a_kept_keys_doubted_pool_to_the_liveness_tier() {
        for sessions in [None, Some(SESSIONS)] {
            let profile = Profile {
                filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
                ..least_requests_profile()
            };
            let mut engine = with_shadow(profile, sessions);
            engine.places_by_key(1);
            let mut rows = [candidate(0, 13), candidate(1, 3)];
            for row in &mut rows {
                row.health.stalled = true;
            }
            let facts = RequestFacts {
                model: "m",
                session_key: Identity::hashed(59),
                session_counted: true,
                now_micros: 71,
                ..RequestFacts::default()
            };
            let decision = engine
                .shadow(
                    &facts,
                    &PoolFacts::default(),
                    &rows,
                    &Constraints::default(),
                    &mut Scratch::new(),
                )
                .expect("a shadow decision");
            assert_eq!(decision.trace().lifted, LiftedVeto::Liveness);
            assert_eq!(
                decision
                    .pick()
                    .and_then(|pick| pick.targets.first())
                    .map(|t| t.id),
                Some(rows[1].id)
            );
            assert!(engine.sessions().is_empty());
        }
    }

    /// A twin: the shadow profile holds the live profile's stages under
    /// another place. Asked behind the live decision, made through the
    /// keeping entry, with that decision's scratch, it decides as the live
    /// one does on every path: wherever the shadow's entry leaves the live
    /// one's, the two disagree here.
    #[test]
    fn shadow_twin_decides_as_the_live_profile_on_every_path() {
        let stages = || Profile {
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            ..least_requests_profile()
        };
        let twin = |sessions: Option<SessionRule>, keeps: bool| {
            let mut engine = Engine::new(
                vec![stages(), stages()],
                vec![Binding {
                    shadow: Some(1),
                    sessions,
                    ..single_binding()
                }],
                23,
                Needs::default(),
            );
            if keeps {
                engine.places_by_key(0);
                engine.places_by_key(1);
            }
            engine
        };
        // What a decision says, apart from the profile that made it and
        // the claim a live decision alone makes.
        let said = |decision: &Decision| match decision {
            Decision::Pick(pick) => (
                pick.targets.iter().map(|t| t.id).collect::<Vec<_>>(),
                None,
                pick.trace.lifted,
            ),
            Decision::Shed(shed) => (
                Vec::new(),
                Some((shed.reason, shed.emptied_by)),
                shed.trace.lifted,
            ),
            Decision::Queue(queue) => (Vec::new(), None, queue.trace.lifted),
        };
        let free = [candidate(0, 13), candidate(1, 3)];
        let with = |mark: fn(&mut Candidate)| {
            let mut rows = free;
            rows.iter_mut().for_each(mark);
            rows
        };
        let over = with(|row| row.health.overloaded = true);
        let doubted = with(|row| row.health.stalled = true);
        let down = with(|row| row.health.healthy = false);
        let mut one_over = free;
        one_over[1].health.overloaded = true;
        // Prefill workers: no key is kept over them.
        let mut prefill = with(|row| row.role = Role::Prefill);
        prefill[1].health.overloaded = true;
        // Nor over decode or encode workers: every role but the regular
        // one passes the key on, as the gateway's own placement asks that
        // every candidate is a regular worker (`select_pass`, at main
        // `450084124`, `model_gateway/src/routers/common/placement.rs:1337`).
        let mut decode = with(|row| row.role = Role::Decode);
        decode[1].health.overloaded = true;
        let mut encode = with(|row| row.role = Role::Encode);
        encode[1].health.overloaded = true;
        let plain = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let keyed = RequestFacts {
            session_key: Identity::hashed(67),
            session_counted: true,
            now_micros: 71,
            ..plain
        };
        // Each path: its rule, whether the key is kept, the request, the
        // rows, and whether the key holds a pin where it is decided.
        let paths = [
            ("plain", None, false, &plain, &free, false),
            (
                "plain, every worker over",
                None,
                false,
                &plain,
                &over,
                false,
            ),
            (
                "plain, every worker doubted",
                None,
                false,
                &plain,
                &doubted,
                false,
            ),
            (
                "plain, every worker down",
                None,
                false,
                &plain,
                &down,
                false,
            ),
            (
                "keyed under a rule",
                Some(SESSIONS),
                false,
                &keyed,
                &free,
                false,
            ),
            (
                "keyed under a rule, a pin",
                Some(SESSIONS),
                false,
                &keyed,
                &free,
                true,
            ),
            (
                "keyed under a rule, every worker down",
                Some(SESSIONS),
                false,
                &keyed,
                &down,
                false,
            ),
            ("kept", None, true, &keyed, &one_over, false),
            ("kept, every worker over", None, true, &keyed, &over, false),
            (
                "kept, every worker doubted",
                None,
                true,
                &keyed,
                &doubted,
                false,
            ),
            (
                "kept under a rule",
                Some(SESSIONS),
                true,
                &keyed,
                &one_over,
                false,
            ),
            (
                "kept under a rule, a pin over",
                Some(SESSIONS),
                true,
                &keyed,
                &one_over,
                true,
            ),
            (
                "kept under a rule, every worker over",
                Some(SESSIONS),
                true,
                &keyed,
                &over,
                false,
            ),
            (
                "kept under a rule, every worker doubted",
                Some(SESSIONS),
                true,
                &keyed,
                &doubted,
                false,
            ),
            ("kept, prefill workers", None, true, &keyed, &prefill, false),
            (
                "kept under a rule, prefill workers",
                Some(SESSIONS),
                true,
                &keyed,
                &prefill,
                false,
            ),
            ("kept, decode workers", None, true, &keyed, &decode, false),
            ("kept, encode workers", None, true, &keyed, &encode, false),
        ];
        let cx = &Constraints::default();
        for (path, sessions, keeps, facts, rows, pinned) in paths {
            let pool = PoolFacts::default();
            let mut scratch = Scratch::new();
            let made = || {
                let engine = twin(sessions, keeps);
                if pinned {
                    // The key's pin, as an earlier dispatch left it.
                    let table = engine.sessions();
                    table.lookup(facts.session(), 10, &SESSIONS);
                    table.dispatched(facts.session(), SessionVerdict::Pinned, free[1].id);
                }
                engine
            };
            // What the live decision alone leaves of the key.
            let alone = made();
            let _ = alone.decide_keeping(facts, &pool, rows, cx, &mut scratch);
            let engine = made();
            let live = engine.decide_keeping(facts, &pool, rows, cx, &mut scratch);
            let shadow = engine
                .shadow(facts, &pool, rows, cx, &mut scratch)
                .expect("a shadow decision");
            assert_eq!(said(&shadow), said(&live), "{path}");
            assert_eq!(shadow.trace().profile, 1, "{path}");
            // A worker of another role over its thresholds passes the key
            // on: a kept key would take it.
            if rows[0].role != Role::Single {
                assert_eq!(said(&live).0, [free[0].id], "{path}");
            }
            if !facts.session_key.is_none() {
                assert_eq!(
                    engine.sessions().held(facts.session()),
                    alone.sessions().held(facts.session()),
                    "{path}: the shadow wrote"
                );
            }
        }
    }

    /// A commit moves the key's pins, so a shadow asked behind it starts
    /// from the pins its live decision read. A key no pin held is placed by
    /// the shadow's own stages behind the claim's commit: the two disagree
    /// where the profiles prefer different rows. The key's next request
    /// reads the pin, and the two agree.
    #[test]
    fn shadow_behind_a_claims_commit_places_the_key_by_its_own_stages() {
        let engine = with_shadow(by_sessions(), Some(SESSIONS));
        let mut busy = candidate(1, 3);
        busy.active_sessions = 17;
        let rows = [candidate(0, 13), busy];
        let facts = keyed_at(61, 71);
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert!(matches!(
            live.pick().map(|pick| pick.session_verdict),
            Some(SessionVerdict::Claimed(_))
        ));
        assert_eq!(commit(&engine, &facts, &live, 0), rows[1].id);
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(first_of(&shadow), Some(rows[0].id));
        assert_eq!(
            decision::ShadowKind::of(&live, &shadow),
            decision::ShadowKind::OtherTarget
        );
        // The key's next request: the pin stood before its decision.
        let next = keyed_at(61, 89);
        let live = engine.decide_keeping(&next, &pool, &rows, &cx, &mut scratch);
        assert_eq!(
            live.pick().map(|pick| pick.session_verdict),
            Some(SessionVerdict::Pinned)
        );
        assert_eq!(commit(&engine, &next, &live, 0), rows[1].id);
        let shadow = engine
            .shadow(&next, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(first_of(&shadow), Some(rows[1].id));
    }

    /// A live respill fronts the worker it takes where it is decided, and
    /// its commit fronts the worker it landed on. The shadow asked behind
    /// it still reads the pin at its cap, and hands the key on by its own
    /// stages: a respill of its own, which moves no pin.
    #[test]
    fn shadow_behind_a_respills_commit_hands_the_key_on_by_its_own_stages() {
        let engine = with_shadow(by_sessions(), Some(SESSIONS));
        let mut rows = [candidate(0, 1), candidate(1, 3), candidate(2, 13)];
        assert_eq!(place_keyed(&engine, &rows, 61, 71).id, rows[0].id);
        // The pin at its cap. The live stages prefer the second row, the
        // shadow's the third.
        rows[0].session_in_flight = 2;
        rows[0].in_flight = 5;
        rows[0].active_sessions = 9;
        rows[1].active_sessions = 17;
        let facts = keyed_at(61, 89);
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert_eq!(
            live.pick().map(|pick| pick.session_verdict),
            Some(SessionVerdict::Respill)
        );
        let moved = Some([rows[1].id, rows[0].id]);
        let pins = || engine.sessions().pins(facts.session());
        assert_eq!(pins(), moved, "ahead of the live commit");
        assert_eq!(commit(&engine, &facts, &live, 0), rows[1].id);
        assert_eq!(pins(), moved);
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(first_of(&shadow), Some(rows[2].id));
        assert_eq!(
            shadow.pick().map(|pick| pick.session_verdict),
            Some(SessionVerdict::Respill)
        );
        assert_eq!(pins(), moved, "the shadow's respill moves no pin");
    }

    /// A claim's commit on an alternate withdraws the claim and fronts the
    /// alternate. A twin asked behind it still names the live decision's
    /// targets, in the plain form and in the form of a kept key.
    #[test]
    fn shadow_twin_behind_a_commit_on_an_alternate_names_the_live_targets() {
        let stages = || Profile {
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            ..least_requests_profile()
        };
        let targets = |decision: &Decision| {
            decision
                .pick()
                .map(|pick| pick.targets.iter().map(|t| t.id).collect::<Vec<_>>())
        };
        for keeps in [false, true] {
            let mut engine = Engine::new(
                vec![stages(), stages()],
                vec![Binding {
                    shadow: Some(1),
                    sessions: Some(SESSIONS),
                    ..single_binding()
                }],
                23,
                Needs::default(),
            );
            if keeps {
                engine.places_by_key(0);
                engine.places_by_key(1);
            }
            let rows = [candidate(0, 13), candidate(1, 3)];
            let facts = keyed_at(61, 71);
            let (pool, cx) = (PoolFacts::default(), Constraints::default());
            let mut scratch = Scratch::new();
            let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
            assert_eq!(
                targets(&live),
                Some(vec![rows[1].id, rows[0].id]),
                "kept: {keeps}"
            );
            assert_eq!(commit(&engine, &facts, &live, 1), rows[0].id);
            assert_eq!(
                engine.sessions().pins(facts.session()),
                Some([rows[0].id, WorkerId(0)]),
                "kept: {keeps}"
            );
            let shadow = engine
                .shadow(&facts, &pool, &rows, &cx, &mut scratch)
                .expect("a shadow decision");
            assert_eq!(targets(&shadow), targets(&live), "kept: {keeps}");
        }
    }

    /// The live filters dropped the key's pin, and the live decision
    /// claimed another worker. A shadow whose filters keep the pin still
    /// follows it behind the claim's commit: the pins the live decision
    /// read hold it.
    #[test]
    fn shadow_behind_a_claims_commit_follows_the_pin_its_own_filters_keep() {
        let live_stages = Profile {
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            ..least_requests_profile()
        };
        let engine = Engine::new(
            vec![live_stages, least_requests_profile()],
            vec![Binding {
                shadow: Some(1),
                sessions: Some(SESSIONS),
                ..single_binding()
            }],
            23,
            Needs::default(),
        );
        let mut rows = [candidate(0, 1), candidate(1, 3), candidate(2, 13)];
        assert_eq!(place_keyed(&engine, &rows, 61, 71).id, rows[0].id);
        rows[0].health.overloaded = true;
        let facts = keyed_at(61, 89);
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert!(matches!(
            live.pick().map(|pick| pick.session_verdict),
            Some(SessionVerdict::Claimed(_))
        ));
        assert_eq!(commit(&engine, &facts, &live, 0), rows[1].id);
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(first_of(&shadow), Some(rows[0].id));
    }

    /// The plain entry keeps no reading, under a binding that names a
    /// shadow or under none: it is the decision it was. The keeping entry
    /// keeps one.
    #[test]
    fn shadow_reading_is_kept_by_the_keeping_entry_alone() {
        let rows = [candidate(0, 13), candidate(1, 3)];
        let facts = keyed_at(61, 71);
        let key = facts.session().get();
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        for engine in [
            session_engine(SESSIONS),
            with_shadow(by_sessions(), Some(SESSIONS)),
        ] {
            let _ = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
            assert!(!scratch.sessions_found.is_of(key, &facts));
        }
        let engine = with_shadow(by_sessions(), Some(SESSIONS));
        let _ = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert!(scratch.sessions_found.is_of(key, &facts));
    }

    /// A keeping decision that answers before it reads its key leaves no
    /// reading of an earlier one. The same request is decided again, and
    /// the live filters keep nobody. Its shadow, whose filters keep the
    /// pool, reads the pins as they stand: the worker the first decision's
    /// commit pinned, where its own stages would place another.
    #[test]
    fn shadow_behind_a_decision_that_read_nothing_takes_no_earlier_reading() {
        let live_stages = Profile {
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            ..least_requests_profile()
        };
        let engine = Engine::new(
            vec![live_stages, least_requests_profile()],
            vec![Binding {
                shadow: Some(1),
                sessions: Some(SESSIONS),
                ..single_binding()
            }],
            23,
            Needs::default(),
        );
        let mut rows = [candidate(0, 1), candidate(1, 3)];
        let facts = keyed_at(61, 71);
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert_eq!(commit(&engine, &facts, &live, 0), rows[0].id);
        rows[0].in_flight = 9;
        for row in &mut rows {
            row.health.overloaded = true;
        }
        let _ = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert!(!scratch.sessions_found.is_of(facts.session().get(), &facts));
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(first_of(&shadow), Some(rows[0].id));
    }

    /// A host that shadows a request decides it through the keeping entry.
    /// Here one mixes the two: a keeping decision is committed, and the
    /// same request is decided again through the entry that keeps none.
    /// The scratch still holds the first decision's reading, and a debug
    /// build stops where the shadow would take it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "the shadow takes a reading behind a plain decision")]
    fn shadow_behind_a_plain_decision_of_a_kept_request_stops_a_debug_build() {
        let engine = with_shadow(least_requests_profile(), Some(SESSIONS));
        let mut rows = [candidate(0, 1), candidate(1, 3)];
        let facts = keyed_at(61, 71);
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert_eq!(commit(&engine, &facts, &live, 0), rows[0].id);
        rows[0].in_flight = 9;
        let again = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
        assert_eq!(first_of(&again), Some(rows[0].id));
        let _ = engine.shadow(&facts, &pool, &rows, &cx, &mut scratch);
    }

    /// The order a host that shadows keeps: each decision of the request
    /// through the keeping entry. Behind the second one the shadow starts
    /// from what that decision read, the pin the first one's commit left,
    /// where its own stages would place another worker.
    #[test]
    fn shadow_behind_a_second_keeping_decision_starts_from_its_reading() {
        let engine = with_shadow(least_requests_profile(), Some(SESSIONS));
        let mut rows = [candidate(0, 1), candidate(1, 3)];
        let facts = keyed_at(61, 71);
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert_eq!(commit(&engine, &facts, &live, 0), rows[0].id);
        rows[0].in_flight = 9;
        let again = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert_eq!(first_of(&again), Some(rows[0].id));
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(first_of(&shadow), Some(rows[0].id));
    }

    /// The reading holds both pins the live decision read. The key's first
    /// pin is unhealthy, the live decision fails over to the second, and
    /// the shadow fails over to it too: with the second pin lost it would
    /// place the key on an unpinned worker by its own stages.
    #[test]
    fn shadow_fails_over_to_the_second_pin_the_live_decision_read() {
        let engine = with_shadow(least_requests_profile(), Some(SESSIONS));
        let mut rows = [candidate(0, 13), candidate(1, 3), candidate(2, 1)];
        let facts = keyed_at(61, 71);
        let key = facts.session();
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        engine.sessions().lookup(key, facts.now_micros, &SESSIONS);
        for row in [&rows[1], &rows[0]] {
            engine
                .sessions()
                .dispatched(key, SessionVerdict::Respill, row.id);
        }
        assert_eq!(engine.sessions().pins(key), Some([rows[0].id, rows[1].id]));
        rows[0].health.healthy = false;
        let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert_eq!(first_of(&live), Some(rows[1].id));
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(first_of(&shadow), Some(rows[1].id));
    }

    /// A reading is of one request: its key, its clock and its seed.
    /// Another request of the same key at the same clock takes none of
    /// it. Its shadow reads the pins as they stand, the worker the first
    /// request's commit pinned, where the reading held no pin.
    #[test]
    fn shadow_of_a_request_of_another_seed_takes_no_reading() {
        let engine = with_shadow(by_sessions(), Some(SESSIONS));
        let mut busy = candidate(1, 3);
        busy.active_sessions = 17;
        let rows = [candidate(0, 13), busy];
        let facts = keyed_at(61, 71);
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert_eq!(commit(&engine, &facts, &live, 0), rows[1].id);
        let next = RequestFacts { seed: 1, ..facts };
        let shadow = engine
            .shadow(&next, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(first_of(&shadow), Some(rows[1].id));
    }

    /// The reading holds the key's pins as the table held them, whether or
    /// not the live filters keep their workers. Here they drop the pin's
    /// worker and the live decision claims another. The shadow, whose
    /// filters keep that worker, follows the pin the live decision read:
    /// the claim's commit has moved the pins since.
    #[test]
    fn shadow_follows_a_pin_the_live_filters_left_out() {
        let live_stages = Profile {
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            ..least_requests_profile()
        };
        let engine = Engine::new(
            vec![live_stages, least_requests_profile()],
            vec![Binding {
                shadow: Some(1),
                sessions: Some(SESSIONS),
                ..single_binding()
            }],
            23,
            Needs::default(),
        );
        let mut rows = [candidate(0, 13), candidate(1, 3)];
        rows[0].health.overloaded = true;
        let facts = keyed_at(61, 71);
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let table = engine.sessions();
        table.lookup(facts.session(), 10, &SESSIONS);
        table.dispatched(facts.session(), SessionVerdict::Pinned, rows[0].id);
        let mut scratch = Scratch::new();
        let live = engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch);
        assert_eq!(commit(&engine, &facts, &live, 0), rows[1].id);
        assert_eq!(table.pins(facts.session()), Some([rows[1].id, rows[0].id]));
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(first_of(&shadow), Some(rows[0].id));
    }

    /// The best survivor, picked once another thread has had its turn at
    /// the table: the stages place a request between its read of the key
    /// and its claim, and one thread has no seam there.
    #[derive(Debug)]
    struct Turned {
        go: mpsc::Sender<()>,
        done: Mutex<mpsc::Receiver<()>>,
        once: Once,
    }

    impl Turned {
        /// The picker, with the two ends the other thread holds: it waits
        /// on the first for the pick, has its turn at the table, and says
        /// so on the second.
        fn new() -> (Self, mpsc::Receiver<()>, mpsc::Sender<()>) {
            let (go, went) = mpsc::channel();
            let (did, done) = mpsc::channel();
            let picker = Self {
                go,
                done: Mutex::new(done),
                once: Once::new(),
            };
            (picker, went, did)
        }
    }

    impl Lifecycle for Turned {}

    impl Picker for Turned {
        fn name(&self) -> &'static str {
            "turned"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            self.once.call_once(|| {
                let _ = self.go.send(());
                if let Ok(done) = self.done.lock() {
                    let _ = done.recv_timeout(Duration::from_secs(10));
                }
            });
            Best.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// Another request's respill turns the key's pins between a decision's
    /// read and its claim. The claim is not taken and no pin came in, so
    /// the table answers with the decision's own pick. That pick is no
    /// pin of the key. A shadow of another profile reads the pins the
    /// claim found against its own pick and keeps its own placement: it
    /// counts the other target it chose, and no false agreement.
    #[test]
    fn shadow_of_another_profile_keeps_its_pick_where_the_claim_found_no_new_pin() {
        let (picker, went, did) = Turned::new();
        let live = Profile {
            picker: Box::new(picker),
            ..least_requests_profile()
        };
        let other = Profile {
            name: "first".into(),
            picker: Box::new(FirstKept),
            ..least_requests_profile()
        };
        let engine = Engine::new(
            vec![live, other],
            vec![Binding {
                shadow: Some(1),
                sessions: Some(SESSIONS),
                ..single_binding()
            }],
            23,
            Needs::default(),
        );
        let mut rows = [
            candidate(0, 0),
            candidate(1, 0),
            candidate(2, 1),
            candidate(3, 0),
        ];
        rows[0].health.healthy = false;
        rows[1].health.healthy = false;
        let facts = keyed_at(61, 71);
        let pool = PoolFacts::default();
        let key = facts.session();
        let table = engine.sessions();
        table.lookup(key, 10, &SESSIONS);
        table.dispatched(key, SessionVerdict::Pinned, rows[0].id);
        table.dispatched(key, SessionVerdict::Pinned, rows[1].id);
        assert_eq!(table.pins(key), Some([rows[1].id, rows[0].id]));
        let turned = rows[0].id;
        let cx = Constraints::default();
        let mut scratch = Scratch::new();
        let live = std::thread::scope(|scope| {
            scope.spawn(move || {
                // Another request's respill, once the live decision has
                // read the key: both pins stay, turned.
                if went.recv_timeout(Duration::from_secs(10)).is_ok() {
                    table.dispatched(key, SessionVerdict::Respill, turned);
                }
                let _ = did.send(());
            });
            engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch)
        });
        // The claim was not taken and nothing was recorded: the pins stand
        // as the respill left them.
        assert_eq!(table.pins(key), Some([rows[0].id, rows[1].id]));
        assert_eq!(first_of(&live), Some(rows[3].id));
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(
            decision::ShadowKind::of(&live, &shadow),
            decision::ShadowKind::OtherTarget
        );
        assert_eq!(first_of(&shadow), Some(rows[2].id));
    }

    /// Another request's pin comes in between a decision's read of the key
    /// and its claim. The key holds a pin already, on a worker that is not
    /// ready, and the newcomer joins behind it. The claim is not taken,
    /// the table answers with the newcomer, and the live decision follows
    /// it. A shadow of another profile, whose own stages place the key
    /// elsewhere, reads the pins the claim found against the pins it
    /// started from and follows the newcomer too. It takes them at the
    /// claim, not at its read. In the kept form the newcomer is over its
    /// thresholds: read first, the pins would hold no worker the shadow's
    /// standing pass keeps, and it would keep its own placement.
    #[test]
    fn shadow_follows_the_pin_that_came_in_where_the_claim_was_not_taken() {
        let rule = SessionRule {
            recovery: Recovery::Original,
            ..SESSIONS
        };
        for kept in [false, true] {
            let (picker, went, did) = Turned::new();
            let live = Profile {
                filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
                picker: Box::new(picker),
                ..least_requests_profile()
            };
            let other = Profile {
                name: "first".into(),
                filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
                picker: Box::new(FirstKept),
                ..least_requests_profile()
            };
            let mut engine = Engine::new(
                vec![live, other],
                vec![Binding {
                    shadow: Some(1),
                    sessions: Some(rule),
                    ..single_binding()
                }],
                23,
                Needs::default(),
            );
            if kept {
                engine.places_by_key(0);
                engine.places_by_key(1);
            }
            // The shadow's stages place the key on the first row and the
            // live ones on the third. The newcomer is the row between
            // them, and the key's pin is the last row, which is not ready.
            let mut rows = [
                candidate(0, 5),
                candidate(1, 3),
                candidate(2, 0),
                candidate(3, 0),
            ];
            rows[1].health.overloaded = kept;
            rows[3].health.healthy = false;
            let (newcomer, pinned) = (rows[1].id, rows[3].id);
            let facts = keyed_at(61, 71);
            let pool = PoolFacts::default();
            let key = facts.session();
            let table = engine.sessions();
            table.lookup(key, 10, &rule);
            table.dispatched(key, SessionVerdict::Pinned, pinned);
            let cx = Constraints::default();
            let mut scratch = Scratch::new();
            let live = std::thread::scope(|scope| {
                scope.spawn(move || {
                    // Another request's dispatch, once the live decision
                    // has read the key: the newcomer joins behind the pin.
                    if went.recv_timeout(Duration::from_secs(10)).is_ok() {
                        table.dispatched(key, SessionVerdict::Pinned, newcomer);
                    }
                    let _ = did.send(());
                });
                engine.decide_keeping(&facts, &pool, &rows, &cx, &mut scratch)
            });
            // The claim was not taken and nothing was recorded.
            assert_eq!(table.pins(key), Some([pinned, newcomer]), "kept: {kept}");
            assert_eq!(first_of(&live), Some(newcomer), "kept: {kept}");
            let shadow = engine
                .shadow(&facts, &pool, &rows, &cx, &mut scratch)
                .expect("a shadow decision");
            assert_eq!(first_of(&shadow), Some(newcomer), "kept: {kept}");
            assert_eq!(
                decision::ShadowKind::of(&live, &shadow),
                decision::ShadowKind::Agreed,
                "kept: {kept}"
            );
        }
    }

    #[test]
    fn shadow_uses_session_pins_without_claiming_touching_or_sweeping() {
        let rule = SessionRule {
            ttl_micros: Some(41),
            sweep_micros: 11,
            ..SESSIONS
        };
        let engine = with_shadow(by_sessions(), Some(rule));
        let control = with_shadow(by_sessions(), Some(rule));
        let mut busy = candidate(1, 3);
        busy.active_sessions = 17;
        let rows = [candidate(0, 13), busy];
        let mut facts = RequestFacts {
            model: "m",
            session_key: Identity::hashed(59),
            now_micros: 71,
            ..RequestFacts::default()
        };
        let key = facts.session();
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let first = |decision: &Decision| {
            decision
                .pick()
                .and_then(|pick| pick.targets.first())
                .map(|t| t.id)
        };
        let fresh = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow placement");
        assert_eq!(first(&fresh), Some(rows[0].id));
        assert_eq!(
            fresh.pick().map(|pick| pick.session_verdict),
            Some(SessionVerdict::None)
        );
        assert!(engine.sessions().is_empty());
        let live = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
        assert_eq!(first(&live), Some(rows[1].id));
        assert_eq!(
            live,
            control.decide(&facts, &pool, &rows, &cx, &mut Scratch::new())
        );
        let pinned = control.sessions().held(key);
        assert!(pinned.is_some());
        assert_eq!(engine.sessions().held(key), pinned);

        facts.now_micros += 17;
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a pinned shadow");
        assert_eq!(first(&shadow), first(&live));
        assert_eq!(
            shadow.pick().map(|pick| pick.session_verdict),
            Some(SessionVerdict::Pinned)
        );
        assert_eq!(engine.sessions().held(key), pinned);

        // Idle until 112: the rule's next sweep is at 121.
        facts.now_micros += 33;
        let expired = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("an expired pin leaves a fresh placement");
        assert_eq!(first(&expired), Some(rows[0].id));
        assert_eq!(engine.sessions().held(key), pinned);

        let untimed = with_shadow(by_sessions(), Some(SESSIONS));
        let live = untimed.decide(&facts, &pool, &rows, &cx, &mut scratch);
        facts.now_micros += 71;
        let shadow = untimed
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a pin with no expiry");
        assert_eq!(first(&shadow), first(&live));
    }

    #[test]
    fn shadow_can_keep_a_key_over_the_overload_veto_and_writes_no_pin() {
        for sessions in [None, Some(SESSIONS)] {
            let profile = Profile {
                filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
                ..least_requests_profile()
            };
            let mut engine = with_shadow(profile, sessions);
            engine.places_by_key(1);
            let mut rows = [candidate(0, 3), candidate(1, 13)];
            rows[0].health.overloaded = true;
            let facts = RequestFacts {
                model: "m",
                session_key: Identity::hashed(59),
                session_counted: true,
                now_micros: 71,
                ..RequestFacts::default()
            };
            let decision = engine
                .shadow(
                    &facts,
                    &PoolFacts::default(),
                    &rows,
                    &Constraints::default(),
                    &mut Scratch::new(),
                )
                .expect("a shadow placement");
            assert_eq!(
                decision
                    .pick()
                    .and_then(|pick| pick.targets.first())
                    .map(|t| t.id),
                Some(rows[0].id)
            );
            assert!(engine.sessions().is_empty());
        }
    }

    #[test]
    fn shadow_stages_count_live_dispatches_but_a_shadow_pick_sends_nothing() {
        use std::sync::atomic::{AtomicU32, Ordering};

        #[derive(Debug)]
        struct Dispatches(Arc<AtomicU32>);

        impl Lifecycle for Dispatches {
            fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        impl Scorer for Dispatches {
            fn name(&self) -> &'static str {
                "dispatch-count"
            }

            fn score(
                &self,
                _: &RequestFacts<'_>,
                rows: &[Candidate],
                kept: &[u32],
                out: &mut [f32],
            ) -> TieBand {
                let count = self.0.load(Ordering::Relaxed);
                for (score, &i) in out.iter_mut().zip(kept) {
                    *score = if rows[i as usize].position == count {
                        1.0
                    } else {
                        0.0
                    };
                }
                TieBand::NORMALIZED
            }
        }

        let count = Arc::new(AtomicU32::new(0));
        let engine = with_shadow(
            Profile {
                scorers: vec![WeightedScorer {
                    scorer: Box::new(Dispatches(Arc::clone(&count))),
                    weight: 1.0,
                }],
                ..least_requests_profile()
            },
            None,
        );
        let rows = [candidate(0, 3), candidate(1, 13)];
        let (facts, pool, cx) = (
            RequestFacts::default(),
            PoolFacts::default(),
            Constraints::default(),
        );
        let mut scratch = Scratch::new();
        let live = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
        let target = live
            .pick()
            .and_then(|pick| pick.targets.first())
            .expect("a live target");
        assert_eq!(target.id, rows[0].id);
        engine.on_dispatched(&Lease::default(), target, Side::Single, 0);
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("the binding shadows");
        assert_eq!(
            shadow
                .pick()
                .and_then(|pick| pick.targets.first())
                .map(|t| t.id),
            Some(rows[1].id)
        );
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn shadow_flags_compose_with_sampling_and_plain_bindings_have_no_shadow() {
        let mut plain = engine();
        let facts = RequestFacts::default();
        assert_eq!(plain.seat_flags(), 0);
        plain.set_explain_sample(17);
        assert_eq!(plain.seat_flags(), 2);
        assert!(plain
            .shadow(
                &facts,
                &PoolFacts::default(),
                &[],
                &Constraints::default(),
                &mut Scratch::new(),
            )
            .is_none());
        let mut shadowed = with_shadow(by_sessions(), None);
        assert_eq!(shadowed.seat_flags(), 4);
        shadowed.set_explain_sample(17);
        assert_eq!(shadowed.seat_flags(), 6);
        shadowed.set_explain_sample(0);
        assert_eq!(shadowed.seat_flags(), 4);
    }

    #[test]
    fn shadow_respects_binding_order_and_separate_profile_state() {
        let fallback = Binding {
            shadow: Some(1),
            ..single_binding()
        };
        let first = Binding {
            model: "first".into(),
            ..single_binding()
        };
        let build = |first, fallback| {
            Engine::new(
                vec![least_requests_profile(), by_sessions()],
                vec![first, fallback],
                23,
                Needs::default(),
            )
        };
        let run = |engine: &Engine, model| {
            engine.shadow(
                &RequestFacts {
                    model,
                    ..RequestFacts::default()
                },
                &PoolFacts::default(),
                &[],
                &Constraints::default(),
                &mut Scratch::new(),
            )
        };
        let engine = build(first.clone(), fallback.clone());
        assert!(run(&engine, "first").is_none());
        assert!(run(&engine, "second").is_some());
        let reused = build(
            Binding {
                profile: 1,
                ..first
            },
            fallback,
        );
        assert!(run(&reused, "second").is_none());
    }

    /// A host names rows for the live profile, by its cap, its warm-up
    /// ruling and the record of sent prefixes. A shadow reads them where
    /// its own profile would be named the same, and reads the view
    /// without them where that profile reads no column. It is named rows
    /// of its own where it differs by its cap, or the live profile is
    /// named none, and the pool's ruling and record are its own. Behind
    /// a live profile that is named no row the record is the shadow's,
    /// whatever it reads a request's holders from. It is not asked
    /// anywhere else.
    #[test]
    fn a_shadow_reads_its_live_rows_only_where_it_would_be_handed_them() {
        use crate::document::spec::{
            AffinitySource, BestParams, PickStage, PrefixAffinityParams, ProfileSpec, ScoreStage,
            WarmupSliceParams,
        };
        type Turn<'a> = &'a dyn Fn(&mut ProfileSpec);
        let rows = |live: &str, shadow: &str, turn: Turn<'_>| {
            let mut document = PolicySpec::preset(live).expect("a preset");
            let next = PolicySpec::preset(shadow).expect("a preset");
            let profile = next.profiles.into_values().next();
            let mut profile = profile.expect("the preset's profile");
            turn(&mut profile);
            document.profiles.insert("second".into(), profile);
            document.bindings[0].shadow = Some(spec::ShadowSpec::Profile("second".into()));
            let engine = document.validate().expect("a binding with a shadow");
            let due = engine.shadow_for("m", Phase::Single);
            due.map(|(at, rows)| (rows, engine.shadow_cap_at(at)))
        };
        let as_it_is = |_: &mut ProfileSpec| {};
        let sample = |profile: &mut ProfileSpec| {
            profile.pick = PickStage::Best(BestParams { sample: 8 }).into();
        };
        let ruling = |profile: &mut ProfileSpec| {
            let slice = WarmupSliceParams {
                weight: 8.0,
                secs: Some(120),
                ..WarmupSliceParams::default()
            };
            profile.score[0] = ScoreStage::WarmupSlice(slice).into();
        };
        // The prefix stage reads the record of sent prefixes alone.
        let record = |profile: &mut ProfileSpec| {
            let prefix = PrefixAffinityParams {
                weight: 4.0,
                source: Some(AffinitySource::Predictor),
                ..PrefixAffinityParams::default()
            };
            profile.score[1] = ScoreStage::PrefixAffinity(prefix).into();
        };
        // The prefix stage reads the index of cache events alone.
        let index = |profile: &mut ProfileSpec| {
            let prefix = PrefixAffinityParams {
                weight: 4.0,
                source: Some(AffinitySource::Index),
                ..PrefixAffinityParams::default()
            };
            profile.score[1] = ScoreStage::PrefixAffinity(prefix).into();
        };
        // Each pairing's word, and the shadow profile's own cap beside it.
        let (aware, least) = ("cache_aware", "least_load");
        let cases: [(&str, &str, Turn<'_>, ShadowRows, u32); 10] = [
            (aware, aware, &as_it_is, ShadowRows::Same, 16),
            (aware, aware, &record, ShadowRows::Apart, 16),
            (aware, aware, &index, ShadowRows::Apart, 16),
            (least, "random", &as_it_is, ShadowRows::Same, 0),
            (aware, aware, &sample, ShadowRows::Own, 8),
            (aware, aware, &ruling, ShadowRows::Apart, 16),
            (least, aware, &as_it_is, ShadowRows::Own, 16),
            (least, aware, &ruling, ShadowRows::Own, 16),
            (least, aware, &record, ShadowRows::Own, 16),
            (aware, least, &as_it_is, ShadowRows::Unnamed, 0),
        ];
        for (live, shadow, turn, wanted, cap) in cases {
            assert_eq!(
                rows(live, shadow, turn),
                Some((wanted, cap)),
                "{shadow} behind {live}"
            );
        }
    }

    /// A naming that fills its cap is held to its profile's own test of a
    /// row: a shadow of other filters than its live profile's would be
    /// named other rows where a full cap is refused whole. Under a sample
    /// such a shadow is named rows of its own. So is one whose filters
    /// are asked by a call each: no two such tests are known to be one.
    /// Under no sample no cap fills, and the shadow reads its live
    /// decision's rows whatever its filters. Two profiles whose filters
    /// are asked of no row share one naming whatever their filters: no
    /// test reads past either's cap. One such profile beside one whose
    /// filters are asked does not.
    #[test]
    fn a_shadow_of_other_filters_is_named_rows_of_its_own_under_a_sample() {
        use crate::{
            document::spec::{BestParams, PickStage, ProfileSpec, ShadowSpec},
            stages::registry::StageRegistry,
        };
        /// A filter that is shown the whole view: the default word. The
        /// engine asks its profile's filters of no row ahead of a naming.
        #[derive(Debug)]
        struct Whole;
        impl Filter for Whole {
            fn name(&self) -> &'static str {
                "whole"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }
        }
        type Turn<'a> = &'a dyn Fn(&mut ProfileSpec);
        let mut catalog = StageRegistry::builtin();
        let added = catalog.filter("whole", || Box::new(Whole));
        added.expect("a free name");
        let rows = |sample: u32, both: Turn<'_>, shadow: Turn<'_>| {
            let mut document = PolicySpec::preset("cache_aware").expect("a preset");
            let live = document.profiles.values_mut().next();
            let live = live.expect("the preset's profile");
            live.pick = match sample {
                0 => "best".into(),
                sample => PickStage::Best(BestParams { sample }).into(),
            };
            both(live);
            let mut second = live.clone();
            shadow(&mut second);
            document.profiles.insert("second".into(), second);
            document.bindings[0].shadow = Some(ShadowSpec::Profile("second".into()));
            let engine = document.validate_in(&catalog, &[]);
            let engine = engine.expect("a binding with a shadow");
            engine.shadow_for("m", Phase::Single).map(|(_, rows)| rows)
        };
        let as_it_is = |_: &mut ProfileSpec| {};
        let whole = |profile: &mut ProfileSpec| profile.filter.push("whole".into());
        // The preset's first filter refuses the rows that are not ready.
        let first_gone = |profile: &mut ProfileSpec| {
            profile.filter.remove(0);
        };
        // The preset's last filter refuses the rows a request tried.
        let fewer = |profile: &mut ProfileSpec| {
            profile.filter.pop();
        };
        // A rule of the document is a stage of its own: the engine folds
        // its profile into no one test.
        let ruled = |profile: &mut ProfileSpec| {
            let rule = r#"{"has-room":{"used":{"candidate.in_flight":1},"capacity":8}}"#;
            profile
                .filter
                .push(serde_json::from_str(rule).expect("a rule"));
        };
        let cases: [(u32, Turn<'_>, Turn<'_>, ShadowRows); 7] = [
            (16, &as_it_is, &as_it_is, ShadowRows::Same),
            (16, &as_it_is, &fewer, ShadowRows::Own),
            (16, &as_it_is, &ruled, ShadowRows::Own),
            (16, &ruled, &as_it_is, ShadowRows::Own),
            (0, &as_it_is, &fewer, ShadowRows::Same),
            // Neither profile's filters are asked of a row: both namings
            // stand as made, whatever the filters.
            (16, &whole, &first_gone, ShadowRows::Same),
            // The live profile's are not, and its shadow's are: the
            // shadow drops the filter that is shown the whole view.
            (16, &whole, &fewer, ShadowRows::Own),
        ];
        for (at, (sample, both, shadow, wanted)) in cases.into_iter().enumerate() {
            assert_eq!(rows(sample, both, shadow), Some(wanted), "pairing {at}");
        }
    }

    /// A pool has one ruling of its warming rows: the live profile's, and
    /// where that profile rules by none and its shadow is named rows of
    /// its own, the shadow profile's.
    #[test]
    fn a_pool_is_ruled_for_its_shadow_where_the_live_profile_rules_by_none() {
        let aware = PolicySpec::preset("cache_aware").expect("a preset");
        let second = aware.profiles.values().next();
        let second = second.expect("the preset's profile").clone();
        let aware = aware.validate().expect("the preset validates");
        let rule = aware.warm_up_rule_for("m", Phase::Single);
        assert!(rule.is_some());
        assert_eq!(aware.pool_rule_for("m", Phase::Single), rule);

        let mut document = PolicySpec::preset("least_load").expect("a preset");
        let alone = document.validate().expect("the preset validates");
        assert_eq!(alone.pool_rule_for("m", Phase::Single), None);
        document.profiles.insert("second".into(), second);
        document.bindings[0].shadow = Some(spec::ShadowSpec::Profile("second".into()));
        let engine = document.validate().expect("a binding with a shadow");
        assert_eq!(engine.warm_up_rule_for("m", Phase::Single), None);
        assert_eq!(engine.pool_rule_for("m", Phase::Single), rule);
    }

    /// A pool keeps one record of sent prefixes: the live profile's, and
    /// where that profile is named no row and its shadow is named rows of
    /// its own, the shadow profile's. A document that names no such
    /// shadow keeps none there.
    #[test]
    fn a_pool_keeps_the_record_for_its_shadow_where_the_live_profile_is_named_no_row() {
        use crate::document::spec::{
            AffinitySource, PrefixAffinityParams, ScoreStage, ShadowSpec, WarmupSliceParams,
        };
        let aware = PolicySpec::preset("cache_aware").expect("a preset");
        let second = aware.profiles.values().next();
        let second = second.expect("the preset's profile").clone();
        let aware = aware.validate().expect("the preset validates");
        // The preset reads each worker by what it has: the index's
        // holders keep their place ahead of the record's.
        assert_eq!(aware.pool_record_for("m", Phase::Single), Some(false));

        let mut document = PolicySpec::preset("least_load").expect("a preset");
        let alone = document.validate().expect("the preset validates");
        assert_eq!(alone.pool_record_for("m", Phase::Single), None);
        document.profiles.insert("second".into(), second.clone());
        document.bindings[0].shadow = Some(ShadowSpec::Profile("second".into()));
        let engine = document.validate().expect("a binding with a shadow");
        assert!(!engine.reads_sent_for("m", Phase::Single));
        assert_eq!(engine.pool_record_for("m", Phase::Single), Some(false));

        // A shadow that reads the record alone is named its holders
        // ahead of the index's.
        let mut by_record = second;
        let prefix = PrefixAffinityParams {
            weight: 4.0,
            source: Some(AffinitySource::Predictor),
            ..PrefixAffinityParams::default()
        };
        by_record.score[1] = ScoreStage::PrefixAffinity(prefix).into();
        document.profiles.insert("second".into(), by_record);
        let engine = document.validate().expect("a binding with a shadow");
        assert_eq!(engine.pool_record_for("m", Phase::Single), Some(true));

        // A live profile that is named no row and reads the record keeps
        // it by its own word: here two warm-up stages of different
        // windows leave it the column whole. A shadow that reads the
        // record another way is not named rows of its own beside it.
        let aware = PolicySpec::preset("cache_aware").expect("a preset");
        let live = aware.profiles.into_values().next();
        let mut live = live.expect("the preset's profile");
        let slice = WarmupSliceParams {
            weight: 1.0,
            secs: Some(7),
            ..WarmupSliceParams::default()
        };
        live.score.push(ScoreStage::WarmupSlice(slice).into());
        document.profiles.insert("live".into(), live.clone());
        let unread = document.bindings[0].profile.replace("live".into());
        let engine = document.validate().expect("a live profile named no row");
        assert_eq!(engine.affinity_rows_cap("m", Phase::Single), 0);
        assert_eq!(engine.pool_record_for("m", Phase::Single), Some(false));
        let rows = engine.shadow_for("m", Phase::Single);
        assert_eq!(rows.map(|(_, rows)| rows), Some(ShadowRows::Apart));

        // That profile as the shadow of one that reads no record: its
        // pool keeps none, so the shadow is not asked over a column that
        // lacks the record it would read live.
        document.profiles.insert("second".into(), live);
        document.bindings[0].profile = unread;
        let engine = document.validate().expect("a shadow named no row");
        assert_eq!(engine.pool_record_for("m", Phase::Single), None);
        let rows = engine.shadow_for("m", Phase::Single);
        assert_eq!(rows.map(|(_, rows)| rows), Some(ShadowRows::Apart));
    }

    /// A model keeps one record of sent prefixes under one bound. A
    /// shadow named rows of its own behind a live profile that is named
    /// none would have its pool keep the record for it, and write under
    /// the bound of every live reader of that model. So it is not asked
    /// where another binding's live profile reads the record for a model
    /// its binding decides, and its pool keeps none. It is asked where
    /// that profile reads the index alone, where the two bindings share
    /// no model, and where one of them takes every pool of the models
    /// they share.
    #[test]
    fn a_shadow_is_not_asked_where_a_live_profile_reads_its_models_record() {
        use crate::document::spec::{AffinitySource, PrefixAffinityParams, ScoreStage, ShadowSpec};
        type Side<'a> = (&'a str, &'a [Phase]);
        let word = |own: Side<'_>, other: Side<'_>, first: bool, index: bool, model: &str| {
            let aware = PolicySpec::preset("cache_aware").expect("a preset");
            let aware = aware.profiles.into_values().next();
            let aware = aware.expect("the preset's profile");
            let mut live = aware.clone();
            if index {
                // The live profile's prefix stage reads the index alone.
                let prefix = PrefixAffinityParams {
                    weight: 4.0,
                    source: Some(AffinitySource::Index),
                    ..PrefixAffinityParams::default()
                };
                live.score[1] = ScoreStage::PrefixAffinity(prefix).into();
            }
            let mut document = PolicySpec::preset("least_load").expect("a preset");
            document.profiles.insert("second".into(), aware);
            document.profiles.insert("live".into(), live);
            let mut reader = document.bindings[0].clone();
            reader.matcher.model = other.0.into();
            reader.matcher.phase = other.1.to_vec();
            reader.profile = Some("live".into());
            let shadowed = &mut document.bindings[0];
            shadowed.matcher.model = own.0.into();
            shadowed.matcher.phase = own.1.to_vec();
            shadowed.shadow = Some(ShadowSpec::Profile("second".into()));
            document.bindings.insert(usize::from(first), reader);
            let engine = document.validate().expect("two bindings");
            let rows = engine.shadow_for(model, Phase::Single);
            let record = engine.pool_record_for(model, Phase::Single);
            (rows.map(|(_, rows)| rows), record)
        };
        let (single, prefill) = (&[Phase::Single][..], &[Phase::Prefill][..]);
        let each: &[Phase] = &[];
        let (own, apart) = (ShadowRows::Own, ShadowRows::Apart);
        // The shadowed binding, the other one, whether the shadowed one
        // is listed first, whether the other reads the index alone, the
        // model asked of, and the word.
        let cases: [(Side<'_>, Side<'_>, bool, bool, &str, ShadowRows); 8] = [
            // Another pool of the model is a live reader's.
            (("*", single), ("*", prefill), true, false, "m", apart),
            (("*", single), ("*", prefill), false, false, "m", apart),
            (("m*", single), ("*x", prefill), true, false, "m", apart),
            // One word a binding: another of its models is skipped too.
            (("*", each), ("m", prefill), false, false, "n", apart),
            // No record is read live beside it.
            (("*", single), ("*", prefill), true, true, "m", own),
            (("m", single), ("n", prefill), true, false, "m", own),
            // One binding takes every pool of the models they share.
            (("m", each), ("*", each), true, false, "m", own),
            (("*", each), ("m", each), false, false, "n", own),
        ];
        for (shadowed, other, first, index, model, wanted) in cases {
            let record = (wanted == own).then_some(false);
            assert_eq!(
                word(shadowed, other, first, index, model),
                (Some(wanted), record),
                "{shadowed:?} beside {other:?}"
            );
        }
    }

    /// A decode pool is read by its model's pair, whose decode side runs
    /// under the pair's binding. So the pool keeps the record that side's
    /// profile reads where a binding of the decode phase listed ahead
    /// reads none, and a shadow of that binding is not asked beside the
    /// side. The first binding's word stands where its own profile reads
    /// the record, a pair whose decode side reads none adds nothing, and
    /// a decode binding listed behind the pair's changes nothing.
    #[test]
    fn a_decode_pool_keeps_the_record_its_models_pair_reads() {
        use crate::document::spec::{AffinitySource, PrefixAffinityParams, ScoreStage, ShadowSpec};
        // A profile that reads the record alone: its holders are named
        // ahead of the index's, which the preset's are not.
        let aware = PolicySpec::preset("cache_aware").expect("a preset");
        let aware = aware.profiles.into_values().next();
        let aware = aware.expect("the preset's profile");
        let mut by_record = aware.clone();
        let prefix = PrefixAffinityParams {
            weight: 4.0,
            source: Some(AffinitySource::Predictor),
            ..PrefixAffinityParams::default()
        };
        by_record.score[1] = ScoreStage::PrefixAffinity(prefix).into();
        // The pair of `round_robin` prefills and `decode` decodes, behind
        // a `round_robin` binding of the decode phase alone.
        let behind = |decode: &str| {
            let pair = PolicySpec::pair_presets("round_robin", decode);
            let mut document = pair.expect("the pair presets");
            let stock = PolicySpec::preset("round_robin").expect("a preset");
            let mut first = stock.bindings[0].clone();
            first.matcher.phase = vec![Phase::Decode];
            first.profile = Some("first".into());
            document.bindings.insert(0, first);
            let turns = stock.profiles.into_values().next();
            let turns = turns.expect("the preset's profile");
            document.profiles.insert("first".into(), turns);
            document
        };
        // The first binding's shadow word and the pool's record.
        let word = |document: &PolicySpec| {
            let engine = document.validate().expect("a valid document");
            let rows = engine.shadow_for("m", Phase::Decode);
            let record = engine.pool_record_for("m", Phase::Decode);
            (rows.map(|(_, rows)| rows), record)
        };

        // A decode request asked alone resolves to the first binding,
        // whose profile reads no record: the pool keeps the pair's.
        let pairs = behind("cache_aware");
        let engine = pairs.validate().expect("two bindings");
        assert!(!engine.reads_sent_for("m", Phase::Decode));
        assert_eq!(engine.pool_record_for("m", Phase::Prefill), None);
        assert_eq!(word(&pairs), (None, Some(false)));
        assert_eq!(word(&behind("round_robin")), (None, None));

        // The first binding's own word stands where its profile reads
        // the record.
        let mut reader = behind("cache_aware");
        reader.profiles.insert("first".into(), by_record.clone());
        assert_eq!(word(&reader), (None, Some(true)));

        // The same binding listed behind the pair's takes no decode
        // request: the pool's word is the pair's, as under the pair alone.
        let mut after = behind("cache_aware");
        after.profiles.insert("first".into(), by_record.clone());
        after.bindings.swap(0, 1);
        let alone = PolicySpec::pair_presets("round_robin", "cache_aware");
        let alone = alone.expect("the pair presets");
        assert_eq!(word(&after), (None, Some(false)));
        assert_eq!(word(&alone), (None, Some(false)));

        // A shadow of the first binding has the pool keep the record for
        // it alone where the pair's decode side reads none. Beside a side
        // that reads the record it is not asked, and the pool's is the
        // side's.
        let shadowed = |decode: &str, shadow| {
            let mut document = behind(decode);
            document.profiles.insert("second".into(), shadow);
            document.bindings[0].shadow = Some(ShadowSpec::Profile("second".into()));
            document
        };
        let (own, apart) = (ShadowRows::Own, ShadowRows::Apart);
        let quiet = shadowed("round_robin", by_record.clone());
        assert_eq!(word(&quiet), (Some(own), Some(true)));
        let beside = shadowed("cache_aware", by_record);
        assert_eq!(word(&beside), (Some(apart), Some(false)));
        // A shadow that reads the record as the side does is not asked
        // either: one rule for every reading.
        let same = shadowed("cache_aware", aware);
        assert_eq!(word(&same), (Some(apart), Some(false)));
    }

    /// A pair is found by its prefill, whatever its own phase list says
    /// of the decode phase: a binding built in code may name the prefill
    /// alone. Its decode profile counts as a reader beside a binding of
    /// the decode phase all the same, in either order. A pair whose
    /// decode profile reads no record does not count.
    #[test]
    fn a_pairs_decode_side_is_a_reader_by_its_prefill_alone() {
        let named = |sent: bool| NamedBy {
            cap: u32::from(sent) * 16,
            rule: None,
            sent,
            sent_alone: false,
            held: Held::Unasked,
        };
        // Profile 0 reads no record and profile 1 reads it.
        let named_by = [named(false), named(true)];
        let alone = || Binding {
            phases: vec![Phase::Decode],
            ..single_binding()
        };
        let pair = |phases: &[Phase], decode: u16| Binding {
            phases: phases.to_vec(),
            decode_profile: Some(decode),
            ..pair_binding()
        };
        let beside = |bindings: [Binding; 2], at| live_reader_beside(&bindings, at, &named_by);
        let both: &[Phase] = &[Phase::Prefill, Phase::Decode];
        let (each, prefill): (&[Phase], &[Phase]) = (&[], &[Phase::Prefill]);
        // The decode binding first: the pair is left the prefill lookup.
        for phases in [both, each, prefill] {
            assert!(beside([alone(), pair(phases, 1)], 0), "{phases:?}");
            assert!(!beside([alone(), pair(phases, 0)], 0), "{phases:?}");
        }
        // The pair first: it takes the decode pool where its list covers
        // the phase, and the decode binding decides none of the model.
        assert!(!beside([pair(both, 1), alone()], 1));
        assert!(!beside([pair(each, 1), alone()], 1));
        assert!(beside([pair(prefill, 1), alone()], 1));
        assert!(!beside([pair(prefill, 0), alone()], 1));
    }

    /// Two bindings share a model where one id matches both patterns, as
    /// [`Binding::matches`] reads each: by every id of one to four
    /// letters.
    #[test]
    fn two_bindings_share_a_model_where_one_id_matches_both() {
        let binding = |model: &str| Binding {
            model: model.into(),
            phases: vec![],
            ..single_binding()
        };
        let ids: Vec<String> = (1..=4u32)
            .flat_map(|len| (0..1u32 << len).map(move |bits| (len, bits)))
            .map(|(len, bits)| {
                let letter = |at| if bits >> at & 1 == 0 { 'a' } else { 'b' };
                (0..len).map(letter).collect()
            })
            .collect();
        let patterns = [
            "*", "a*", "ab*", "b*", "*a", "*ba", "*b", "a", "ab", "ba", "b",
        ];
        for one in patterns {
            for other in patterns {
                let (one, other) = (binding(one), binding(other));
                let both = |id: &String| {
                    one.matches(id, Phase::Single) && other.matches(id, Phase::Single)
                };
                assert_eq!(
                    one.shares_a_model(&other),
                    ids.iter().any(both),
                    "{} and {}",
                    one.model,
                    other.model
                );
            }
        }
    }

    #[test]
    fn shadow_needs_join_the_matching_bindings_live_needs() {
        let mut document = PolicySpec::preset("least_load").expect("a preset");
        let mut next = PolicySpec::preset("cache_aware").expect("a preset");
        let mut profile = next
            .profiles
            .remove("cache_aware")
            .expect("the affinity profile");
        profile.pick = spec::PickStage::Best(spec::BestParams { sample: 13 }).into();
        document.profiles.insert("second".into(), profile);
        document.bindings[0].matcher.phase = vec![Phase::Single];
        document.bindings[0].shadow = Some(spec::ShadowSpec::Profile("second".into()));
        let engine = document
            .validate()
            .expect("a binding with affinity in its shadow");
        assert!(!engine.profiles()[0].needs().affinity);
        assert!(engine.needs_for("m", Phase::Single).affinity);
        assert!(engine.needs_for_phase(Phase::Single).affinity);
        assert!(!engine.needs_for("m", Phase::Decode).affinity);
        assert!(!engine.needs_for_phase(Phase::Decode).affinity);
        assert!(
            matches!(engine.needs_rows().as_slice(), [("*", Phase::Single, needs)] if needs.affinity)
        );
        assert_eq!(engine.named_by[1].cap, 13);
        assert!(engine
            .shadow(
                &RequestFacts {
                    phase: Phase::Decode,
                    ..RequestFacts::default()
                },
                &PoolFacts::default(),
                &[],
                &Constraints::default(),
                &mut Scratch::new(),
            )
            .is_none());
    }

    #[test]
    fn the_pick_orders_targets_best_first_and_names_the_survivors() {
        let engine = engine();
        let pool = [candidate(0, 5), candidate(1, 1), candidate(2, 3)];
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("a pick");
        let order: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
        assert_eq!(order, vec![1, 2, 0]);
        assert_eq!(pick.trace.survivors.get(), Some(3));
        assert_eq!(pick.trace.top[0].0, WorkerId(101));
        assert_eq!(pick.trace.top_len, 3);
    }

    #[test]
    fn the_trace_lists_equal_scores_in_the_pools_order_and_keeps_the_three_best() {
        // Four idle workers behind one busy one: the three idle rows the
        // trace names come in the pool's order, and the busy row stays out.
        let engine = engine();
        let pool = [
            candidate(0, 3),
            candidate(1, 0),
            candidate(2, 0),
            candidate(3, 0),
            candidate(4, 0),
        ];
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut scratch,
        );
        let trace = decision.trace();
        assert_eq!(trace.top_len, 3);
        let named: Vec<WorkerId> = trace.top.iter().map(|(id, _)| *id).collect();
        assert_eq!(named, vec![WorkerId(101), WorkerId(102), WorkerId(103)]);
        assert!(trace.top.iter().all(|(_, score)| *score == 1.0));
    }

    #[test]
    fn a_soft_filter_steps_aside_when_it_would_empty_the_pool() {
        let engine = engine();
        let pool = [candidate(0, 1)];
        let tried = [WorkerId(100)];
        let facts = RequestFacts {
            model: "m",
            tried: &tried,
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut scratch,
        );
        assert_eq!(
            decision
                .pick()
                .and_then(|p| p.targets.first().map(|t| t.position)),
            Some(0)
        );
    }

    #[test]
    fn a_hard_filter_that_empties_the_pool_sheds_with_its_name() {
        let engine = engine();
        let mut pool = [candidate(0, 1)];
        pool[0].health.healthy = false;
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        match engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut scratch,
        ) {
            Decision::Shed(shed) => {
                assert_eq!(shed.reason, ShedReason::NoEndpoints);
                assert_eq!(shed.emptied_by, "healthy");
                assert_eq!(shed.trace.made_under(), Some(0));
            }
            other => panic!("an unhealthy pool answered {other:?}"),
        }
    }

    fn strict_profile() -> Profile {
        Profile {
            name: "strict".into(),
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            scorers: vec![],
            picker: Box::new(Best),
        }
    }

    fn strict_engine() -> Engine {
        Engine::new(
            vec![strict_profile()],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        )
    }

    fn decide_single(engine: &Engine, pool: &[Candidate]) -> Decision {
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        engine.decide(
            &facts,
            &PoolFacts::default(),
            pool,
            &Constraints::default(),
            &mut scratch,
        )
    }

    fn pair_binding() -> Binding {
        Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::PdPair,
            profile: 0,
            decode_profile: Some(0),
            shadow: None,
            decode_shadow: None,
            sessions: None,
            overload: None,
        }
    }

    fn decide_pair(engine: &Engine, prefills: &[Candidate], decodes: &[Candidate]) -> Decision {
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        engine.decide_pair(
            &facts,
            &PoolFacts::default(),
            PairPools {
                prefills,
                decodes,
                ..PairPools::default()
            },
            &Constraints::default(),
            &Constraints::default(),
            &mut scratch,
        )
    }

    /// A scorer for the affinity column's cases: each kept row scores the
    /// overlap its own entry of the column holds, and declares that it reads
    /// the column.
    #[derive(Debug)]
    struct ByOverlap;

    impl Lifecycle for ByOverlap {}

    impl Scorer for ByOverlap {
        fn name(&self) -> &'static str {
            "by-overlap"
        }

        fn score(
            &self,
            facts: &RequestFacts<'_>,
            _: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            for (score, &i) in out.iter_mut().zip(kept) {
                *score = facts.affinity_at(i).overlap_blocks as f32 / 100.0;
            }
            TieBand::NORMALIZED
        }

        fn needs(&self) -> Needs {
            Needs {
                affinity: true,
                ..Needs::default()
            }
        }
    }

    fn single_binding() -> Binding {
        Binding {
            plan: Plan::Single,
            decode_profile: None,
            ..pair_binding()
        }
    }

    /// A pair's decode side runs under the model's pair binding, so the
    /// pool's lookup names that binding for a decode pool whatever covers
    /// the decode phase first. An `epd` pair's encode side runs under it
    /// too, so an encode pool is that binding's the same way. An encode
    /// pool of a model whose pair is of the `pd-pair` plan is the encode
    /// phase's own: that plan names no encode side. Every other pool is
    /// its own phase's.
    #[test]
    fn a_decode_pool_is_decided_under_its_models_pair_binding() {
        let phase_first = |phase: Phase| Binding {
            phases: vec![phase],
            ..single_binding()
        };
        let pair = |plan: Plan| Binding {
            model: "m".into(),
            plan,
            ..pair_binding()
        };
        let engine = |bindings: Vec<Binding>| {
            Engine::new(
                vec![least_requests_profile()],
                bindings,
                1,
                Needs::default(),
            )
        };
        let plan_of = |engine: &Engine, model: &str, phase: Phase| {
            engine.pool_binding(model, phase).map(|b| b.plan)
        };

        let pd = engine(vec![phase_first(Phase::Decode), pair(Plan::PdPair)]);
        // A decode request asked alone still resolves to the phase's first
        // binding. The pool follows the pair, which is what decides over it.
        assert_eq!(
            pd.binding("m", Phase::Decode).map(|b| b.plan),
            Some(Plan::Single)
        );
        assert_eq!(plan_of(&pd, "m", Phase::Decode), Some(Plan::PdPair));
        assert_eq!(plan_of(&pd, "m", Phase::Prefill), Some(Plan::PdPair));
        // A model the pair binding does not match has the phase's own.
        assert_eq!(plan_of(&pd, "n", Phase::Decode), Some(Plan::Single));
        assert_eq!(plan_of(&pd, "n", Phase::Prefill), None);
        // A binding of another plan that covers prefill ahead of the pair
        // binding takes the model's prefill lookup: no pair decision runs
        // for the model, and its decode pool is the decode phase's own.
        let ahead = engine(vec![
            phase_first(Phase::Prefill),
            phase_first(Phase::Decode),
            pair(Plan::PdPair),
        ]);
        let decode = ahead.pool_binding("m", Phase::Decode);
        assert_eq!(
            decode.map(|b| (b.plan, b.phases.clone())),
            Some((Plan::Single, vec![Phase::Decode]))
        );

        let epd = Plan::Epd { encode: 0 };
        let with_encoders = engine(vec![
            phase_first(Phase::Decode),
            phase_first(Phase::Encode),
            pair(epd),
        ]);
        assert_eq!(plan_of(&with_encoders, "m", Phase::Decode), Some(epd));
        // An encode item asked alone still resolves to the phase's first
        // binding. The pool follows the pair, whose one call decides over
        // it.
        assert_eq!(
            with_encoders.binding("m", Phase::Encode).map(|b| b.plan),
            Some(Plan::Single)
        );
        assert_eq!(plan_of(&with_encoders, "m", Phase::Encode), Some(epd));
        assert_eq!(
            plan_of(&with_encoders, "n", Phase::Encode),
            Some(Plan::Single)
        );
        // A pair of two sides names no encode side: its model's encode
        // pool is the encode phase's own.
        let two_sides = engine(vec![phase_first(Phase::Encode), pair(Plan::PdPair)]);
        assert_eq!(plan_of(&two_sides, "m", Phase::Encode), Some(Plan::Single));
    }

    /// One profile that keeps the healthy rows and leads with the deepest
    /// overlap, under `binding`.
    fn overlap_engine(binding: Binding) -> Engine {
        let profile = Profile {
            name: "overlap".into(),
            filters: vec![Box::new(Healthy)],
            scorers: vec![WeightedScorer {
                scorer: Box::new(ByOverlap),
                weight: 1.0,
            }],
            picker: Box::new(Best),
        };
        Engine::new(vec![profile], vec![binding], 1, Needs::default())
    }

    /// A column whose entries hold `overlaps`, in the candidates' order.
    fn held(overlaps: &[u32]) -> Vec<AffinityFacts> {
        overlaps
            .iter()
            .map(|&overlap_blocks| AffinityFacts {
                overlap_blocks,
                ..AffinityFacts::default()
            })
            .collect()
    }

    /// `n` routable candidates of pair class 1.
    fn paired(n: u32) -> Vec<Candidate> {
        (0..n)
            .map(|position| Candidate {
                pair_class: 1,
                ..candidate(position, 0)
            })
            .collect()
    }

    /// A pair's trace is its prefill's: the prefill profile's index and the
    /// rows that profile's filters left of the prefill pool. The decode ran
    /// another profile over another pool, and its trace is not the pair's.
    /// Each target names the prefill with a decode, and no session verdict
    /// stands on a pair.
    #[test]
    fn a_pairs_trace_is_its_prefills() {
        let engine = Engine::new(
            vec![least_requests_profile(), least_requests_profile()],
            vec![Binding {
                decode_profile: Some(1),
                ..pair_binding()
            }],
            1,
            Needs::default(),
        );
        let decision = decide_pair(&engine, &paired(3), &paired(2));
        let pick = decision.pick().expect("a pair");
        assert_eq!(pick.trace.profile, 0);
        assert_eq!(pick.trace.survivors.get(), Some(3));
        assert!(!pick.targets.is_empty());
        assert!(pick.targets.iter().all(|target| target.pair.is_some()));
        assert_eq!(pick.session_verdict, SessionVerdict::None);
    }

    /// The position a decision leads with.
    fn lead(decision: &Decision) -> Option<u32> {
        decision
            .pick()
            .and_then(|pick| pick.targets.first().map(|target| target.position))
    }

    fn refused_for_its_column(decision: &Decision) -> bool {
        matches!(
            decision,
            Decision::Shed(shed)
                if (shed.reason, shed.emptied_by) == (ShedReason::PolicyDeclined, "affinity-column")
        )
    }

    #[test]
    fn a_column_of_another_length_is_refused_at_the_regular_entry() {
        let engine = overlap_engine(Binding {
            model: "m".into(),
            ..single_binding()
        });
        let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        let decide = |model: &'static str, column: &[AffinityFacts]| {
            let facts = RequestFacts {
                model,
                affinity: column,
                ..RequestFacts::default()
            };
            engine.decide(
                &facts,
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut Scratch::new(),
            )
        };
        // One entry a candidate: the holder of the deepest overlap leads.
        assert_eq!(lead(&decide("m", &held(&[10, 90, 20]))), Some(1));
        // An empty column decides too: nothing held, every row alike.
        assert!(decide("m", &[]).pick().is_some());
        // Shorter or longer: refused, under the column's own stage name.
        for column in [held(&[10, 90]), held(&[10, 90, 20, 30])] {
            let decision = decide("m", &column);
            assert!(
                refused_for_its_column(&decision),
                "a column of {} for three candidates answered {decision:?}",
                column.len()
            );
            assert_eq!(decision.trace().made_under(), None);
        }
        // A decision's inputs are checked before anything is resolved from
        // them: a model no binding covers is refused for its column when the
        // column is another pool's, and for its binding only beside a column
        // that fits.
        assert!(refused_for_its_column(&decide("unbound", &held(&[10, 90]))));
        assert!(matches!(
            decide("unbound", &held(&[10, 90, 20])),
            Decision::Shed(shed) if shed.emptied_by == "binding"
        ));
    }

    /// A holder a host named in the column's place leads as the deepest
    /// overlap of a column does, and a request that names nothing reads its
    /// column.
    #[test]
    fn a_holder_a_host_named_leads_as_the_deepest_overlap_of_a_column_does() {
        use crate::engine::{AffinityHolder, AffinityRows};
        let engine = overlap_engine(Binding {
            model: "m".into(),
            ..single_binding()
        });
        let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        let decide = |column: &[AffinityFacts], named: Option<AffinityRows<'_>>| {
            let facts = RequestFacts {
                model: "m",
                affinity: column,
                named,
                ..RequestFacts::default()
            };
            engine.decide(
                &facts,
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut Scratch::new(),
            )
        };
        let holders = [
            AffinityHolder {
                row: 0,
                overlap_blocks: 10,
                predicted_blocks: 0,
            },
            AffinityHolder {
                row: 1,
                overlap_blocks: 90,
                predicted_blocks: 0,
            },
        ];
        let named = AffinityRows {
            holders: &holders,
            ..AffinityRows::default()
        };
        assert_eq!(lead(&decide(&[], Some(named))), Some(1));
        // Nobody named decides too.
        assert!(decide(&[], Some(AffinityRows::default())).pick().is_some());
        // A request that names nothing reads its column.
        assert_eq!(lead(&decide(&held(&[10, 90, 20]), None)), Some(1));
    }

    /// The regular entry takes the named rows under an assertion, as it
    /// takes the rows' positions: a debug build stops at a list out of
    /// order, at a row named twice, at a row past the pool and at rows
    /// named beside a column, each by the assertion's own words. A release
    /// build does not check, so the case is a debug build's alone.
    #[cfg(debug_assertions)]
    #[test]
    fn rows_named_out_of_order_past_the_pool_or_beside_a_column_stop_a_debug_build() {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        use crate::engine::{AffinityHolder, AffinityRows};
        let engine = overlap_engine(Binding {
            model: "m".into(),
            ..single_binding()
        });
        let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        let column = held(&[10, 90, 20]);
        let shapes = [
            ("holders out of order", &[][..], &[2, 1][..], &[][..]),
            ("a holder named twice", &[][..], &[1, 1][..], &[][..]),
            ("a holder past the pool", &[][..], &[1, 3][..], &[][..]),
            ("warming rows out of order", &[][..], &[][..], &[1, 0][..]),
            ("a warming row named twice", &[][..], &[][..], &[1, 1][..]),
            ("a warming row past the pool", &[][..], &[][..], &[3][..]),
            ("rows named beside a column", &column[..], &[1][..], &[][..]),
        ];
        for (what, column, holders, warming) in shapes {
            let holders: Vec<AffinityHolder> = holders
                .iter()
                .map(|&row| AffinityHolder {
                    row,
                    overlap_blocks: 90,
                    predicted_blocks: 0,
                })
                .collect();
            let stopped = catch_unwind(AssertUnwindSafe(|| {
                let facts = RequestFacts {
                    model: "m",
                    affinity: column,
                    named: Some(AffinityRows {
                        holders: &holders,
                        warming,
                        thin: false,
                        mean_in_flight: None,
                    }),
                    ..RequestFacts::default()
                };
                engine.decide(
                    &facts,
                    &PoolFacts::default(),
                    &pool,
                    &Constraints::default(),
                    &mut Scratch::new(),
                )
            }))
            .expect_err(what);
            let said = stopped
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| stopped.downcast_ref::<&str>().copied())
                .unwrap_or_default();
            assert!(
                said.contains("rows named in the column's place"),
                "{what} stopped by another word: {said}"
            );
        }
    }

    /// The regular entry takes its rows under the assertion the pair entry
    /// takes each side's under: a debug build stops at a row whose position
    /// is not its index, the second one here. A release build does not
    /// check, so the case is a debug build's alone.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "a row's position is its index")]
    fn a_misplaced_row_stops_a_debug_build_at_the_regular_entry() {
        let engine = overlap_engine(single_binding());
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[1].position = 5;
        let decision = engine.decide(
            &RequestFacts::default(),
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        panic!("a misplaced row answered {decision:?}");
    }

    /// The pair entry of the cases below: `prefills` and `decodes` beside the
    /// two columns as the call hands them.
    fn decide_pair_beside(
        engine: &Engine,
        prefills: &[Candidate],
        decodes: &[Candidate],
        prefill_affinity: &[AffinityFacts],
        decode_affinity: &[AffinityFacts],
    ) -> Decision {
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        engine.decide_pair(
            &facts,
            &PoolFacts::default(),
            PairPools {
                prefills,
                decodes,
                prefill_affinity,
                decode_affinity,
                ..PairPools::default()
            },
            &Constraints::default(),
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }

    #[test]
    fn a_column_of_another_length_is_refused_at_each_side_of_the_pair_entry() {
        let engine = overlap_engine(pair_binding());
        let (prefills, decodes) = (paired(2), paired(3));
        let (two, three) = (held(&[10, 90]), held(&[10, 90, 20]));
        let refused = |prefill_affinity: &[AffinityFacts], decode_affinity: &[AffinityFacts]| {
            refused_for_its_column(&decide_pair_beside(
                &engine,
                &prefills,
                &decodes,
                prefill_affinity,
                decode_affinity,
            ))
        };
        // Each pool beside its own column, or beside none, is decided.
        assert!(!refused(&two, &three));
        assert!(!refused(&[], &[]));
        assert!(!refused(&two, &[]));
        // The prefills beside a column of the decodes' length, then the
        // decodes beside one of the prefills': each side is refused alone.
        assert!(refused(&three, &three));
        assert!(refused(&two, &two));
    }

    /// A pair's columns ride its pools. A column on the request itself names
    /// neither pool, whatever its length: refused, never dropped in silence.
    #[test]
    fn a_column_on_the_request_is_refused_at_the_pair_entry() {
        let engine = overlap_engine(pair_binding());
        let (prefills, decodes) = (paired(2), paired(2));
        let decide = |on_the_request: &[AffinityFacts]| {
            let facts = RequestFacts {
                model: "m",
                affinity: on_the_request,
                ..RequestFacts::default()
            };
            engine.decide_pair(
                &facts,
                &PoolFacts::default(),
                PairPools {
                    prefills: &prefills,
                    decodes: &decodes,
                    ..PairPools::default()
                },
                &Constraints::default(),
                &Constraints::default(),
                &mut Scratch::new(),
            )
        };
        assert!(decide(&[]).pick().is_some());
        // As long as either pool: its length fits both, and it is refused.
        assert!(refused_for_its_column(&decide(&held(&[10, 90]))));
    }

    /// Rows named on a pair's request name neither pool, as a column on it
    /// does: refused.
    #[test]
    fn rows_named_on_the_request_are_refused_at_the_pair_entry() {
        let engine = overlap_engine(pair_binding());
        let (prefills, decodes) = (paired(2), paired(2));
        let facts = RequestFacts {
            model: "m",
            named: Some(AffinityRows::default()),
            ..RequestFacts::default()
        };
        let decision = engine.decide_pair(
            &facts,
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            &Constraints::default(),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(refused_for_its_column(&decision), "{decision:?}");
    }

    /// A pair's side carries no rows named, whatever its request carries:
    /// named rows index neither pool.
    #[test]
    fn a_pair_side_reads_no_rows_named_on_the_request() {
        let (prefills, decodes) = (paired(2), paired(2));
        let holders = [crate::engine::AffinityHolder {
            row: 1,
            overlap_blocks: 8,
            predicted_blocks: 0,
        }];
        let facts = RequestFacts {
            model: "m",
            named: Some(AffinityRows {
                holders: &holders,
                ..AffinityRows::default()
            }),
            // A pin is a position in one pool: no side takes it either.
            pinned_at: 2,
            ..RequestFacts::default()
        };
        // Nor the counts a host lent: they are read by a position too.
        let mut counted = [0, 1].map(|position| Candidate {
            position,
            ..Candidate::default()
        });
        let lent = Lent::take(&mut counted);
        let facts = lent.beside(&facts);
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        for (side, name) in [(PairSide::Prefill, "prefill"), (PairSide::Decode, "decode")] {
            let (_, read) = pools.side(side, &facts).expect("an empty column fits");
            assert!(read.named.is_none(), "the {name} side");
            assert_eq!(read.pinned_at, 0, "the {name} side");
            assert!(read.lent_in_flight.is_none(), "the {name} side");
        }
    }

    /// Each side of a pair reads its own pool's column: with the two columns
    /// swapped between pools of equal length, the pair moves.
    #[test]
    fn each_side_of_a_pair_reads_its_own_pools_column() {
        let engine = overlap_engine(pair_binding());
        let (prefills, decodes) = (paired(2), paired(2));
        let (rising, falling) = (held(&[10, 90]), held(&[90, 10]));
        let pair = |prefill_affinity: &[AffinityFacts], decode_affinity: &[AffinityFacts]| {
            let decision = decide_pair_beside(
                &engine,
                &prefills,
                &decodes,
                prefill_affinity,
                decode_affinity,
            );
            let target = *decision
                .pick()
                .and_then(|pick| pick.targets.first())
                .expect("a pair");
            (
                target.position,
                target.pair.expect("a decode").decode_position,
            )
        };
        assert_eq!(pair(&rising, &falling), (1, 0));
        assert_eq!(pair(&falling, &rising), (0, 1));
    }

    /// One index, one worker: a pool whose filters drop its FIRST candidates
    /// reads each survivor's own entry of the column, not the entry at the
    /// survivor's place among the kept.
    #[test]
    fn a_narrowed_pool_reads_each_survivors_own_entry_of_the_column() {
        let engine = overlap_engine(single_binding());
        let mut pool = [
            candidate(0, 0),
            candidate(1, 0),
            candidate(2, 0),
            candidate(3, 0),
        ];
        pool[0].health.healthy = false;
        pool[1].health.healthy = false;
        let column = held(&[90, 80, 10, 70]);
        let facts = RequestFacts {
            model: "m",
            affinity: &column,
            ..RequestFacts::default()
        };
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert_eq!(lead(&decision), Some(3));
    }

    /// A profile says by the decision's one lookup whether a stage of it
    /// may refuse by what its gateways report of leases. A stage that
    /// says nothing is taken as one that may, and is no plugin's by
    /// that: its profile shows no plugin its rows. A profile of this
    /// crate's stages says it holds none.
    #[test]
    fn a_profile_whose_stage_may_refuse_by_reports_says_so_at_the_lookup() {
        #[derive(Debug)]
        struct Counted;

        impl Filter for Counted {
            fn name(&self) -> &'static str {
                "counted"
            }

            fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
                facts.in_flight(c) == 0
            }
        }

        #[derive(Debug)]
        struct Scored;

        impl Lifecycle for Scored {}

        impl Scorer for Scored {
            fn name(&self) -> &'static str {
                "scored"
            }

            fn score(
                &self,
                _: &RequestFacts<'_>,
                _: &[Candidate],
                _: &[u32],
                out: &mut [f32],
            ) -> TieBand {
                out.fill(0.0);
                TieBand::NORMALIZED
            }
        }

        let engine_of =
            |profile| Engine::new(vec![profile], vec![single_binding()], 1, Needs::default());
        let plain = engine_of(least_requests_profile());
        assert!(!plain.reads_for("m", Phase::Single).2);
        // A filter that says nothing, and a scorer that says nothing: each
        // is taken as a stage that may refuse, and its word is the
        // profile's.
        let filtered = || Profile {
            filters: vec![Box::new(Counted)],
            ..least_requests_profile()
        };
        let scored = Profile {
            scorers: vec![WeightedScorer {
                scorer: Box::new(Scored),
                weight: 1.0,
            }],
            ..least_requests_profile()
        };
        let (_, shown, by_reports) = engine_of(filtered()).reads_for("m", Phase::Single);
        assert!(by_reports, "the filter's word is the profile's");
        assert!(!shown, "no plugin is shown its rows");
        let scored = engine_of(scored).reads_for("m", Phase::Single).2;
        assert!(scored, "the scorer's word is the profile's");
        // A decode binding with no pair carries its own profile's word.
        let decode = || Binding {
            phases: vec![Phase::Decode],
            ..single_binding()
        };
        let alone = Engine::new(vec![filtered()], vec![decode()], 1, Needs::default());
        assert!(alone.reads_for("m", Phase::Decode).2);
        // A decode binding of a profile with no such stage stands ahead of
        // a pair binding whose decode side holds one: the decode decision
        // carries the pair's word, as it carries the pair's needs.
        let ahead = Engine::new(
            vec![least_requests_profile(), filtered()],
            vec![
                decode(),
                Binding {
                    decode_profile: Some(1),
                    ..pair_binding()
                },
            ],
            1,
            Needs::default(),
        );
        assert!(!ahead.reads_for("m", Phase::Prefill).2);
        assert!(ahead.reads_for("m", Phase::Decode).2);
        // An encode binding of a profile with no such stage stands ahead of
        // an `epd` binding whose encode side holds one: the encode decision
        // carries the pair's word the same way. A pair of two sides names
        // no encode side and adds none.
        let encode = || Binding {
            phases: vec![Phase::Encode],
            ..single_binding()
        };
        let behind = |plan: Plan| {
            let pair = Binding {
                plan,
                ..pair_binding()
            };
            Engine::new(
                vec![least_requests_profile(), filtered()],
                vec![encode(), pair],
                1,
                Needs::default(),
            )
        };
        let three = behind(Plan::Epd { encode: 1 });
        assert!(three.reads_for("m", Phase::Encode).2);
        assert!(!three.reads_for("m", Phase::Decode).2);
        assert!(!behind(Plan::PdPair).reads_for("m", Phase::Encode).2);
    }

    /// A profile that shows its rows to a plugin says so by the engine's one
    /// lookup, and its decision over rows that carry a mask or a pin beside
    /// them is declined under a name of its own, in every build.
    #[test]
    fn a_profile_that_shows_its_rows_is_told_and_declines_rows_not_written_for_it() {
        #[derive(Debug)]
        struct Shown;

        impl Lifecycle for Shown {}

        impl DecisionScope for Shown {
            fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

            fn close(&self) {}
        }

        impl Filter for Shown {
            fn name(&self) -> &'static str {
                "shown"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn scope(&self) -> Option<&dyn DecisionScope> {
                Some(self)
            }
        }

        let engine_of =
            |profile| Engine::new(vec![profile], vec![single_binding()], 1, Needs::default());
        let plain = engine_of(least_requests_profile());
        let needs = plain.needs_for("m", Phase::Single);
        assert_eq!(plain.reads_for("m", Phase::Single), (needs, false, false));
        let engine = engine_of(Profile {
            filters: vec![Box::new(Shown)],
            ..least_requests_profile()
        });
        assert!(engine.reads_for("m", Phase::Single).1);
        let pool = [candidate(0, 0), candidate(1, 0)];
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let pinned = RequestFacts {
            pinned_at: 1,
            ..facts
        };
        let masked = Constraints {
            stalled: &[0],
            ..Constraints::default()
        };
        let written = Constraints::default();
        let decide = |facts: &RequestFacts<'_>, cx: &Constraints<'_>| {
            engine.decide(facts, &PoolFacts::default(), &pool, cx, &mut Scratch::new())
        };
        assert!(lead(&decide(&facts, &written)).is_some());
        for (facts, cx, beside) in [(&facts, &masked, "a mask"), (&pinned, &written, "a pin")] {
            let Decision::Shed(shed) = decide(facts, cx) else {
                panic!("{beside} beside the rows is declined");
            };
            assert_eq!(
                (shed.reason, shed.emptied_by),
                (ShedReason::PolicyDeclined, WRITTEN_ROWS),
                "{beside}"
            );
            assert_eq!(shed.trace.made_under(), Some(0), "{beside}");
        }
        // A profile that shows nothing reads both.
        assert!(lead(&plain.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &masked,
            &mut Scratch::new()
        ))
        .is_some());
        // Under a pair binding the word is the phase's own profile's: the
        // decode side shows its rows here, and the prefill side does not.
        let pair = Engine::new(
            vec![
                least_requests_profile(),
                Profile {
                    filters: vec![Box::new(Shown)],
                    ..least_requests_profile()
                },
            ],
            vec![Binding {
                decode_profile: Some(1),
                ..pair_binding()
            }],
            1,
            Needs::default(),
        );
        assert!(!pair.reads_for("m", Phase::Prefill).1);
        assert!(pair.reads_for("m", Phase::Decode).1);
    }

    /// The pair entry declines as the single entry does. A side whose
    /// profile shows its rows is handed rows written for the decision: a
    /// mask beside that side's rows is declined before either scope opens,
    /// and rows that carry the veto in their own field are shown with it. A
    /// mask beside a side that shows nothing is read by that side's filters.
    /// The host is told of the pair's decode side where a decode binding of
    /// another profile stands ahead of it, as it is told that side's needs,
    /// and of an `epd` pair's encode side the same way.
    #[test]
    fn a_pair_side_that_shows_its_rows_declines_a_mask_beside_them() {
        /// The scopes opened, and the rows they were shown stalled.
        #[derive(Debug)]
        struct Shown(Arc<Mutex<(u32, u32)>>);

        impl Lifecycle for Shown {}

        impl DecisionScope for Shown {
            fn open(&self, _: &RequestFacts<'_>, rows: &[Candidate], _: &PoolFacts<'_>) {
                let mut seen = self.0.lock().expect("the lock");
                seen.0 += 1;
                seen.1 += rows.iter().filter(|row| row.health.stalled).count() as u32;
            }

            fn close(&self) {}
        }

        impl Filter for Shown {
            fn name(&self) -> &'static str {
                "shown"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn scope(&self) -> Option<&dyn DecisionScope> {
                Some(self)
            }
        }

        let seen = Arc::new(Mutex::new((0u32, 0u32)));
        let shows = || Profile {
            filters: vec![Box::new(Healthy), Box::new(Shown(Arc::clone(&seen)))],
            ..least_requests_profile()
        };
        let read = || *seen.lock().expect("the lock");
        let pair_of = |prefill: Profile, decode: Profile| {
            let binding = Binding {
                decode_profile: Some(1),
                ..pair_binding()
            };
            Engine::new(vec![prefill, decode], vec![binding], 1, Needs::default())
        };
        let (prefills, mut decodes) = (paired(2), paired(2));
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let masked = Constraints {
            stalled: &[0],
            ..Constraints::default()
        };
        let written = Constraints::default();
        let decide = |engine: &Engine, decodes: &[Candidate], sides: [&Constraints<'_>; 2]| {
            let pools = PairPools {
                prefills: &prefills,
                decodes,
                ..PairPools::default()
            };
            let pool = PoolFacts::default();
            engine.decide_pair(
                &facts,
                &pool,
                pools,
                sides[0],
                sides[1],
                &mut Scratch::new(),
            )
        };
        let declined = |decision: &Decision| {
            matches!(decision, Decision::Shed(shed)
                if (shed.reason, shed.emptied_by) == (ShedReason::PolicyDeclined, WRITTEN_ROWS))
        };
        // The decode side shows its rows, the prefill side shows nothing.
        let engine = pair_of(least_requests_profile(), shows());
        assert!(lead(&decide(&engine, &decodes, [&written, &written])).is_some());
        assert_eq!(read(), (1, 0));
        // A mask beside the decode side's rows: declined, and no scope opens.
        let decision = decide(&engine, &decodes, [&written, &masked]);
        assert!(declined(&decision), "{decision:?}");
        assert_eq!(decision.trace().made_under(), Some(1));
        assert_eq!(read(), (1, 0), "no scope opens for a declined decision");
        // A mask beside the prefill side, which shows nothing: its filters
        // read it, and the masked prefill is passed over.
        assert_eq!(
            lead(&decide(&engine, &decodes, [&masked, &written])),
            Some(1)
        );
        assert_eq!(read(), (2, 0));
        // Rows written for the decision carry the veto in their own field:
        // the plugin is shown it, and the built-in filter reads it there.
        decodes[0].health.stalled = true;
        let decision = decide(&engine, &decodes, [&written, &written]);
        let decode = decision
            .pick()
            .and_then(|pick| pick.targets.first().and_then(|target| target.pair));
        assert_eq!(
            decode.map(|pair| pair.decode_position),
            Some(1),
            "{decision:?}"
        );
        assert_eq!(read(), (3, 1));
        decodes[0].health.stalled = false;

        // The other way round: the prefill side shows its rows.
        let turned = pair_of(shows(), least_requests_profile());
        let decision = decide(&turned, &decodes, [&masked, &written]);
        assert!(declined(&decision), "{decision:?}");
        let decision = decide(&turned, &decodes, [&written, &masked]);
        let decode = decision
            .pick()
            .and_then(|pick| pick.targets.first().and_then(|target| target.pair));
        assert_eq!(
            decode.map(|pair| pair.decode_position),
            Some(1),
            "{decision:?}"
        );

        // A decode binding of a profile that shows nothing stands ahead of
        // the pair binding: the pair's decode side still runs under the pair
        // binding, and the host is told so.
        let ahead = Engine::new(
            vec![least_requests_profile(), shows()],
            vec![
                Binding {
                    phases: vec![Phase::Decode],
                    ..single_binding()
                },
                Binding {
                    decode_profile: Some(1),
                    ..pair_binding()
                },
            ],
            1,
            Needs::default(),
        );
        assert!(!ahead.reads_for("m", Phase::Prefill).1);
        assert!(ahead.reads_for("m", Phase::Decode).1);
        // An encode binding of a profile that shows nothing stands ahead of
        // an `epd` binding whose encode profile shows its rows: the encode
        // side of the pair's one call runs under the pair binding, and the
        // host is told so too.
        let encoders = Engine::new(
            vec![least_requests_profile(), shows()],
            vec![
                Binding {
                    phases: vec![Phase::Encode],
                    ..single_binding()
                },
                Binding {
                    plan: Plan::Epd { encode: 1 },
                    ..pair_binding()
                },
            ],
            1,
            Needs::default(),
        );
        assert!(!encoders.reads_for("m", Phase::Decode).1);
        assert!(encoders.reads_for("m", Phase::Encode).1);
    }

    /// A stage that reads the affinity column as a filter alone, or as the
    /// picker alone, counts in the profile's needs as one that scores does:
    /// a host that fills the column where it is declared leaves it empty
    /// only where nothing reads it.
    #[test]
    fn a_profiles_needs_fold_its_filters_and_its_picker() {
        #[derive(Debug)]
        struct Reader;

        impl Lifecycle for Reader {}

        impl Filter for Reader {
            fn name(&self) -> &'static str {
                "reader"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn needs(&self) -> Needs {
                Needs {
                    affinity: true,
                    ..Needs::default()
                }
            }
        }

        impl Picker for Reader {
            fn name(&self) -> &'static str {
                "reader"
            }

            fn pick(
                &self,
                facts: &RequestFacts<'_>,
                candidates: &[Candidate],
                kept: &[u32],
                scores: &[f32],
                band: TieBand,
                out: &mut Targets,
            ) {
                Best.pick(facts, candidates, kept, scores, band, out);
            }

            fn needs(&self) -> Needs {
                Needs {
                    affinity: true,
                    ..Needs::default()
                }
            }
        }

        let as_filter = Profile {
            filters: vec![Box::new(Reader)],
            ..least_requests_profile()
        };
        let as_picker = Profile {
            picker: Box::new(Reader),
            ..least_requests_profile()
        };
        assert!(!least_requests_profile().needs().affinity);
        assert!(as_filter.needs().affinity, "a filter's needs");
        assert!(as_picker.needs().affinity, "the picker's needs");
        // The engine's summary holds them however the engine was built: a
        // caller that hands no needs of its own still gets its stages'.
        for (profile, bound_as) in [(as_filter, "a filter"), (as_picker, "the picker")] {
            let engine = Engine::new(vec![profile], vec![single_binding()], 1, Needs::default());
            assert!(engine.needs().affinity, "the summary, bound as {bound_as}");
            assert!(
                engine.needs_for("m", Phase::Single).affinity,
                "the decision's needs, bound as {bound_as}"
            );
        }
    }

    #[test]
    fn an_all_overloaded_pool_sheds_as_saturated() {
        let mut engine = strict_engine();
        engine.shed_when_all_vetoed(0);
        let mut pool = [candidate(0, 1), candidate(1, 1)];
        for c in &mut pool {
            c.health.overloaded = true;
        }
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        match engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut scratch,
        ) {
            Decision::Shed(shed) => {
                assert_eq!(shed.reason, ShedReason::Saturated);
                assert_eq!(
                    shed.emptied_by, "not-overloaded",
                    "the stage that emptied the pool is named"
                );
            }
            other => panic!("an overloaded pool answered {other:?}"),
        }
    }

    #[test]
    fn a_pool_emptied_behind_the_healthy_filter_names_the_later_stage() {
        let mut engine = strict_engine();
        engine.shed_when_all_vetoed(0);
        let mut pool = [candidate(0, 1), candidate(1, 1)];
        pool[0].health.healthy = false;
        pool[1].health.overloaded = true;
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        match engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut scratch,
        ) {
            Decision::Shed(shed) => {
                assert_eq!(shed.emptied_by, "not-overloaded");
                assert_eq!(
                    shed.reason,
                    ShedReason::Saturated,
                    "every routable row was over the thresholds"
                );
            }
            other => panic!("a pool with no routable row answered {other:?}"),
        }
    }

    #[test]
    fn a_draining_overloaded_pool_is_emptied_by_healthy_and_not_saturated() {
        let mut pool = [candidate(0, 1), candidate(1, 1)];
        for c in &mut pool {
            c.health.overloaded = true;
            c.health.draining = true;
        }
        match decide_single(&strict_engine(), &pool) {
            Decision::Shed(shed) => {
                assert_eq!(shed.emptied_by, "healthy");
                assert_eq!(shed.reason, ShedReason::NoEndpoints);
            }
            other => panic!("a draining pool answered {other:?}"),
        }
    }

    #[test]
    fn an_empty_pool_is_emptied_by_no_stage() {
        match decide_single(&strict_engine(), &[]) {
            Decision::Shed(shed) => {
                assert_eq!(shed.emptied_by, "none");
                assert_eq!(shed.reason, ShedReason::NoEndpoints);
            }
            other => panic!("an empty pool answered {other:?}"),
        }
    }

    #[test]
    fn a_saturated_class_sheds_as_saturated_whatever_the_other_classes_hold() {
        // The prefills' classes are 7 and 8; class 7's decode is overloaded
        // and class 9's healthy decode partners no prefill: saturation.
        let engine = Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let mut prefills = [candidate(0, 0), candidate(1, 0)];
        prefills[0].pair_class = 7;
        prefills[1].pair_class = 8;
        let mut decodes = [candidate(0, 0), candidate(1, 0)];
        decodes[0].pair_class = 7;
        decodes[0].health.overloaded = true;
        decodes[1].pair_class = 9;
        match decide_pair(&engine, &prefills, &decodes) {
            Decision::Shed(shed) => {
                assert_eq!(shed.emptied_by, "not-overloaded");
                assert_eq!(shed.reason, ShedReason::Saturated);
            }
            other => panic!("a saturated class answered {other:?}"),
        }
    }

    #[test]
    fn an_unroutable_prefills_partners_do_not_name_the_shed() {
        // A routable prefill of class 1 with an unhealthy partner, a draining
        // prefill of class 2 with an overloaded one: the routable prefill's
        // partner names the shed, not the draining prefill's.
        let engine = Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let mut prefills = [candidate(0, 0), candidate(1, 0)];
        prefills[0].pair_class = 1;
        prefills[1].pair_class = 2;
        prefills[1].health.draining = true;
        let mut decodes = [candidate(0, 0), candidate(1, 0)];
        decodes[0].pair_class = 1;
        decodes[0].health.healthy = false;
        decodes[1].pair_class = 2;
        decodes[1].health.overloaded = true;
        match decide_pair(&engine, &prefills, &decodes) {
            Decision::Shed(shed) => {
                assert_eq!(shed.emptied_by, "healthy");
                assert_eq!(shed.reason, ShedReason::NoEndpoints);
            }
            other => panic!("a pool with one routable prefill answered {other:?}"),
        }
        // Every prefill draining over a saturated decode fleet: the prefill
        // side's own stage is named.
        prefills[0].health.draining = true;
        decodes[0].health.healthy = true;
        decodes[0].health.overloaded = true;
        match decide_pair(&engine, &prefills, &decodes) {
            Decision::Shed(shed) => {
                assert_eq!(shed.emptied_by, "healthy");
                assert_eq!(shed.reason, ShedReason::NoEndpoints);
            }
            other => panic!("a draining prefill fleet answered {other:?}"),
        }
    }

    #[test]
    fn a_routable_prefills_saturated_partner_names_the_shed_past_an_unroutable_one() {
        // The mirror case: the draining prefill's partner is healthy, so the
        // prefill walk never reaches "partnered"; the routable prefill's
        // partner is overloaded and names the shed.
        let engine = Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let mut prefills = [candidate(0, 0), candidate(1, 0)];
        prefills[0].pair_class = 1;
        prefills[1].pair_class = 2;
        prefills[1].health.draining = true;
        let mut decodes = [candidate(0, 0), candidate(1, 0)];
        decodes[0].pair_class = 1;
        decodes[0].health.overloaded = true;
        decodes[1].pair_class = 2;
        match decide_pair(&engine, &prefills, &decodes) {
            Decision::Shed(shed) => {
                assert_eq!(shed.emptied_by, "not-overloaded");
                assert_eq!(shed.reason, ShedReason::Saturated);
            }
            other => panic!("a routable prefill with a saturated partner answered {other:?}"),
        }
    }

    #[test]
    fn a_saturated_decode_fleet_sheds_as_saturated() {
        let mut engine = Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        engine.shed_when_all_vetoed(0);
        let mut prefills = [candidate(0, 0)];
        prefills[0].pair_class = 7;
        let mut decodes = [candidate(0, 0), candidate(1, 0)];
        for d in &mut decodes {
            d.pair_class = 7;
            d.health.overloaded = true;
        }
        match decide_pair(&engine, &prefills, &decodes) {
            Decision::Shed(shed) => {
                assert_eq!(shed.emptied_by, "not-overloaded");
                assert_eq!(shed.reason, ShedReason::Saturated);
            }
            other => panic!("a saturated decode fleet answered {other:?}"),
        }
    }

    #[test]
    fn a_prefill_whose_partners_fail_the_decode_filters_is_dropped_as_unpartnered() {
        // The decode side never runs for a prefill without a kept partner:
        // the shed names the decode stage that dropped the partners, or
        // "partnered" when decodes survive and none has a prefill's class.
        let engine = Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let mut prefills = [candidate(0, 0)];
        prefills[0].pair_class = 7;
        let mut decodes = [candidate(0, 0)];
        decodes[0].pair_class = 7;
        decodes[0].health.healthy = false;
        match decide_pair(&engine, &prefills, &decodes) {
            Decision::Shed(shed) => {
                assert_eq!(
                    shed.emptied_by, "healthy",
                    "the decode stage that dropped the partner"
                );
                assert_eq!(shed.reason, ShedReason::NoEndpoints);
            }
            other => panic!("an unroutable partner answered {other:?}"),
        }
        decodes[0].health.healthy = true;
        decodes[0].pair_class = 8;
        match decide_pair(&engine, &prefills, &decodes) {
            Decision::Shed(shed) => {
                assert_eq!(shed.emptied_by, "partnered", "no decode of the class");
            }
            other => panic!("an empty class answered {other:?}"),
        }
    }

    #[test]
    fn a_pair_with_no_decode_at_all_is_dropped_as_unpartnered() {
        // An empty decode pool is a class with no decode for every prefill:
        // the prefill's own "partnered" shed stands, where the decode side's
        // filter alone would name "none".
        let engine = Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let mut prefills = [candidate(0, 0)];
        prefills[0].pair_class = 7;
        match decide_pair(&engine, &prefills, &[]) {
            Decision::Shed(shed) => {
                assert_eq!(shed.emptied_by, "partnered", "no decode of any class");
                assert_eq!(shed.reason, ShedReason::NoEndpoints);
            }
            other => panic!("an empty decode pool answered {other:?}"),
        }
    }

    #[test]
    fn the_pair_takes_its_decode_profile_from_the_binding_the_prefill_resolved() {
        // A decode-phase binding to the strict profile sits before the pair
        // binding, whose decode side is lenient; the pair uses its own.
        let lenient = Profile {
            name: "lenient".into(),
            filters: vec![Box::new(Healthy)],
            scorers: vec![],
            picker: Box::new(Best),
        };
        let engine = Engine::new(
            vec![lenient, strict_profile()],
            vec![
                Binding {
                    model: "*".into(),
                    phases: vec![Phase::Decode],
                    plan: Plan::Single,
                    profile: 1,
                    decode_profile: None,
                    shadow: None,
                    decode_shadow: None,
                    sessions: None,
                    overload: None,
                },
                pair_binding(),
            ],
            1,
            Needs::default(),
        );
        let mut prefills = [candidate(0, 0)];
        prefills[0].pair_class = 7;
        let mut decodes = [candidate(0, 0)];
        decodes[0].pair_class = 7;
        decodes[0].health.overloaded = true;
        let decision = decide_pair(&engine, &prefills, &decodes);
        assert!(
            decision.pick().is_some(),
            "the pair binding's lenient decode side keeps the overloaded decode: {decision:?}"
        );
    }

    #[test]
    fn a_pair_that_exists_is_found_past_the_prefill_pickers_targets() {
        // Five tied prefills and one decode, partnered with the fifth alone:
        // the prefill pool narrows to that one before the picker's four.
        let engine = Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let mut prefills = [candidate(0, 0); 5];
        for (i, p) in prefills.iter_mut().enumerate() {
            p.position = i as u32;
            p.id = WorkerId(100 + i as u64);
            p.pair_class = i as u32;
        }
        let mut decodes = [candidate(0, 0)];
        decodes[0].pair_class = 4;
        for seed in 0..20u64 {
            let facts = RequestFacts {
                model: "m",
                seed,
                ..RequestFacts::default()
            };
            let mut scratch = Scratch::new();
            let decision = engine.decide_pair(
                &facts,
                &PoolFacts::default(),
                PairPools {
                    prefills: &prefills,
                    decodes: &decodes,
                    ..PairPools::default()
                },
                &Constraints::default(),
                &Constraints::default(),
                &mut scratch,
            );
            let pick = decision.pick().expect("the partnered prefill is found");
            assert_eq!(
                pick.targets.first().map(|t| t.position),
                Some(4),
                "seed {seed}"
            );
        }
        decodes[0].health.healthy = false;
        match decide_pair(&engine, &prefills, &decodes) {
            Decision::Shed(shed) => {
                assert_eq!(
                    shed.emptied_by, "healthy",
                    "the decode stage that left no partner"
                );
            }
            other => panic!("a pool without a partnered prefill answered {other:?}"),
        }
    }

    #[test]
    fn a_single_binding_declines_a_pair_decision() {
        let prefills = [candidate(0, 0)];
        let decodes = [candidate(0, 0)];
        match decide_pair(&strict_engine(), &prefills, &decodes) {
            Decision::Shed(shed) => {
                assert_eq!(shed.reason, ShedReason::PolicyDeclined);
                assert_eq!(shed.emptied_by, "binding");
                assert_eq!(shed.trace.survivors.get(), Some(0), "no pool was read");
                assert_eq!(shed.trace.made_under(), None);
            }
            other => panic!("a single binding answered a pair with {other:?}"),
        }
    }

    /// A row whose position is not its index is a host's mistake. A debug
    /// build stops at the assertion where a decision takes its inputs; the
    /// assertion reads every row, and the misplaced one here is the second.
    /// A release build keeps the pair plan's guard: the lookup of the chosen
    /// prefill finds no row at that position and the target loses its turn,
    /// never the process, and never to another row's class. The first pool
    /// holds the last clause: its first prefill is of the decode's class and
    /// busier, so the plan chooses the misplaced one and keeps the first as
    /// its alternate; the misplaced one loses its turn and the alternate
    /// leads, where a lookup that fell back to any row would pair the
    /// misplaced one. The second pool has no alternate, and the decision is
    /// shed. No hosted job builds these cases for release: the release side
    /// runs by hand with every change of the pair entry, until the last
    /// reader of a row's position goes and the assertion with it.
    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "a row's position is its index")
    )]
    fn a_misplaced_prefill_stops_a_debug_build_and_costs_a_release_build_its_target() {
        let engine = Engine::new(
            vec![least_requests_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let mut decodes = [candidate(0, 0)];
        decodes[0].pair_class = 7;
        // Both prefills pair with the decode and the first is busier: the
        // plan chooses the second, the misplaced one, then the first.
        let mut prefills = [candidate(0, 9), candidate(1, 0)];
        prefills[0].pair_class = 7;
        prefills[1].position = 5;
        prefills[1].pair_class = 7;
        let decision = decide_pair(&engine, &prefills, &decodes);
        assert_eq!(
            lead(&decision),
            Some(0),
            "the alternate leads where the misplaced prefill lost its turn: {decision:?}"
        );
        // The first prefill outside any pairing: the misplaced one is the
        // only target, and its lost turn sheds the decision.
        prefills[0].pair_class = 0;
        match decide_pair(&engine, &prefills, &decodes) {
            Decision::Shed(shed) => assert_eq!(shed.emptied_by, "partners"),
            other => panic!("a misplaced prefill answered {other:?}"),
        }
    }

    #[test]
    fn class_zero_is_outside_any_pairing_on_either_side() {
        // A host that leaves the classes unset: a class-0 prefill over a
        // healthy class-0 decode sheds at the partnered stage instead of
        // pairing all with all.
        let engine = Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let prefills = [candidate(0, 0)];
        let decodes = [candidate(0, 0)];
        match decide_pair(&engine, &prefills, &decodes) {
            Decision::Shed(shed) => {
                assert_eq!(shed.emptied_by, "partnered");
                assert_eq!(shed.reason, ShedReason::NoEndpoints);
            }
            other => panic!("two class-0 rows answered {other:?}"),
        }
        // A prefill of a class over decodes of class 0 alone: no partner.
        let mut classed = [candidate(0, 0)];
        classed[0].pair_class = 7;
        match decide_pair(&engine, &classed, &decodes) {
            Decision::Shed(shed) => assert_eq!(shed.emptied_by, "partnered"),
            other => panic!("a class-0 decode partnered {other:?}"),
        }
    }

    #[test]
    fn a_class_zero_prefill_is_passed_over_for_a_classed_one() {
        // A just-admitted prefill with no class yet beside a classed one
        // whose decode is up: the pair is the classed one's, for every seed,
        // and the class-0 decode is no alternate.
        let engine = Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let mut prefills = [candidate(0, 0), candidate(1, 0)];
        prefills[1].pair_class = 7;
        let mut decodes = [candidate(0, 0), candidate(1, 0)];
        decodes[1].pair_class = 7;
        for seed in 0..20u64 {
            let facts = RequestFacts {
                model: "m",
                seed,
                ..RequestFacts::default()
            };
            let mut scratch = Scratch::new();
            let decision = engine.decide_pair(
                &facts,
                &PoolFacts::default(),
                PairPools {
                    prefills: &prefills,
                    decodes: &decodes,
                    ..PairPools::default()
                },
                &Constraints::default(),
                &Constraints::default(),
                &mut scratch,
            );
            let pick = decision.pick().expect("the classed prefill pairs");
            let first = pick.targets.first().expect("a target");
            assert_eq!(first.position, 1, "seed {seed}");
            assert_eq!(
                first.pair.map(|p| p.decode_position),
                Some(1),
                "seed {seed}"
            );
            assert_eq!(pick.targets.len(), 1, "the class-0 decode is no alternate");
        }
    }

    #[test]
    fn a_retry_passes_over_the_tried_worker_while_others_remain() {
        let engine = engine();
        let pool = [candidate(0, 1), candidate(1, 5)];
        let tried = [WorkerId(100)];
        let facts = RequestFacts {
            model: "m",
            attempt: 1,
            tried: &tried,
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            &pool,
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("the untried worker is picked");
        let order: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
        assert_eq!(
            order,
            vec![1],
            "the tried worker is dropped while another remains"
        );
        assert_eq!(pick.trace.survivors.get(), Some(1));
        assert_eq!(pick.trace.made_under(), Some(0));
    }

    #[test]
    fn a_request_no_binding_covers_is_declined_by_the_binding() {
        let engine = Engine::new(
            vec![least_requests_profile()],
            vec![Binding {
                model: "pd/*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        let facts = RequestFacts {
            model: "chat",
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        match engine.decide(
            &facts,
            &PoolFacts::default(),
            &[candidate(0, 0)],
            &Constraints::default(),
            &mut scratch,
        ) {
            Decision::Shed(shed) => {
                assert_eq!(shed.reason, ShedReason::PolicyDeclined);
                assert_eq!(shed.emptied_by, "binding");
                assert_eq!(shed.trace.survivors.get(), Some(0), "no pool was read");
                assert_eq!(shed.trace.made_under(), None);
            }
            other => panic!("an unbound model answered {other:?}"),
        }
    }

    /// A pair's decode profile is a decode row of the pair binding's glob,
    /// whatever phases the binding covers: an engine built by hand may bind
    /// a pair for the prefill phase alone, and a decode decision of that
    /// model carries the pair's decode profile. A pair bound off the prefill
    /// phase starts no pair: it writes no decode row.
    #[test]
    fn a_pairs_decode_profile_is_a_needs_row_of_the_pairs_glob() {
        #[derive(Debug)]
        struct Fed;

        impl Filter for Fed {
            fn name(&self) -> &'static str {
                "fed"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn needs(&self) -> Needs {
                Needs {
                    loads: true,
                    ..Needs::default()
                }
            }
        }

        let quiet = least_requests_profile();
        let fed = Profile {
            filters: vec![Box::new(Fed)],
            ..least_requests_profile()
        };
        let decode = fed.needs();
        assert_eq!(quiet.needs(), Needs::default());
        assert!(decode.loads);
        let engine = Engine::new(
            vec![quiet, fed],
            vec![Binding {
                model: "pd/*".into(),
                phases: vec![Phase::Prefill],
                plan: Plan::PdPair,
                profile: 0,
                decode_profile: Some(1),
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        assert_eq!(engine.needs_for("pd/x", Phase::Decode), decode);
        assert_eq!(engine.needs_rows(), [("pd/*", Phase::Decode, decode)]);
        // An `epd` pair's encode profile is an encode row of the glob the
        // same way: an encode decision of the model carries it, and so
        // does the encode phase's own answer.
        let fed = Profile {
            filters: vec![Box::new(Fed)],
            ..least_requests_profile()
        };
        let engine = Engine::new(
            vec![least_requests_profile(), fed],
            vec![Binding {
                model: "pd/*".into(),
                phases: vec![Phase::Prefill],
                plan: Plan::Epd { encode: 1 },
                profile: 0,
                decode_profile: Some(0),
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        let encode = decode;
        assert_eq!(engine.needs_for("pd/x", Phase::Encode), encode);
        assert_eq!(engine.needs_for_phase(Phase::Encode), encode);
        assert_eq!(engine.needs_for("pd/x", Phase::Decode), Needs::default());
        assert_eq!(engine.needs_rows(), [("pd/*", Phase::Encode, encode)]);
        // No prefill lookup resolves to a pair bound for another phase.
        let fed = Profile {
            filters: vec![Box::new(Fed)],
            ..least_requests_profile()
        };
        let engine = Engine::new(
            vec![least_requests_profile(), fed],
            vec![Binding {
                model: "pd/*".into(),
                phases: vec![Phase::Encode],
                plan: Plan::PdPair,
                profile: 0,
                decode_profile: Some(1),
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        assert_eq!(engine.needs_for("pd/x", Phase::Decode), Needs::default());
        assert!(engine.needs_rows().is_empty(), "{:?}", engine.needs_rows());
    }

    #[test]
    fn the_pair_plan_pairs_the_prefill_with_a_decode_of_its_class() {
        let engine = Engine::new(
            vec![Profile {
                name: "least".into(),
                filters: vec![Box::new(Healthy)],
                scorers: vec![WeightedScorer {
                    scorer: Box::new(LeastRequests),
                    weight: 1.0,
                }],
                picker: Box::new(Best),
            }],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::PdPair,
                profile: 0,
                decode_profile: Some(0),
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        let mut prefills = [candidate(0, 2), candidate(1, 0)];
        prefills[0].pair_class = 1;
        prefills[1].pair_class = 2;
        let mut decodes = [candidate(0, 0), candidate(1, 7), candidate(2, 1)];
        decodes[0].pair_class = 1;
        decodes[1].pair_class = 2;
        decodes[2].pair_class = 2;
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide_pair(
            &facts,
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            &Constraints::default(),
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("a pair");
        let first = pick.targets.first().expect("a target");
        assert_eq!(first.position, 1, "the idle prefill");
        assert_eq!(
            first.pair.map(|p| p.decode_position),
            Some(2),
            "its least-loaded partner"
        );
        assert_eq!(
            pick.targets.len(),
            2,
            "the alternate decode of the same prefill"
        );
        assert_eq!(
            pick.targets.as_slice()[1].pair.map(|p| p.decode_position),
            Some(1)
        );
    }

    #[test]
    fn the_decode_picker_does_not_replay_the_prefill_pickers_draws() {
        // Random on both sides, four prefills of four decodes each. With one
        // seed on both sides the decode index inside the class followed the
        // prefill index, and four of the sixteen decodes took every pair.
        let engine = PolicySpec::pair_presets("random", "random")
            .expect("the pair spec")
            .validate()
            .expect("the engine");
        let mut prefills = [candidate(0, 0); 4];
        for (i, p) in prefills.iter_mut().enumerate() {
            p.position = i as u32;
            p.id = WorkerId(100 + i as u64);
            p.pair_class = i as u32 + 1;
            p.has_capacity = true;
        }
        let mut decodes = [candidate(0, 0); 16];
        for (i, d) in decodes.iter_mut().enumerate() {
            d.position = i as u32;
            d.id = WorkerId(200 + i as u64);
            d.pair_class = i as u32 / 4 + 1;
            d.has_room = true;
        }
        let mut picked = [0u32; 16];
        for seed in 0..4096u64 {
            let facts = RequestFacts {
                model: "m",
                seed,
                ..RequestFacts::default()
            };
            let mut scratch = Scratch::new();
            let decision = engine.decide_pair(
                &facts,
                &PoolFacts::default(),
                PairPools {
                    prefills: &prefills,
                    decodes: &decodes,
                    ..PairPools::default()
                },
                &Constraints::default(),
                &Constraints::default(),
                &mut scratch,
            );
            let pick = decision.pick().expect("a pair");
            let first = pick.targets.first().expect("a target");
            let decode = first.pair.expect("a decode").decode_position as usize;
            assert_eq!(
                decodes[decode].pair_class, prefills[first.position as usize].pair_class,
                "seed {seed}: the decode is of the prefill's class"
            );
            picked[decode] += 1;
        }
        let idle: Vec<usize> = picked
            .iter()
            .enumerate()
            .filter(|(_, &n)| n == 0)
            .map(|(i, _)| i)
            .collect();
        assert!(
            idle.is_empty(),
            "decodes never picked over 4096 seeds: {idle:?}"
        );
    }

    #[test]
    fn a_prefill_without_partners_is_passed_over_for_the_next() {
        let engine = Engine::new(
            vec![least_requests_profile()],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::PdPair,
                profile: 0,
                decode_profile: Some(0),
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        let mut prefills = [candidate(0, 0), candidate(1, 5)];
        prefills[0].pair_class = 7;
        prefills[1].pair_class = 8;
        let mut decodes = [candidate(0, 0)];
        decodes[0].pair_class = 8;
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide_pair(
            &facts,
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            &Constraints::default(),
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("the busier prefill still pairs");
        assert_eq!(pick.targets.first().map(|t| t.position), Some(1));
    }

    /// A scorer that keeps the rows it was handed and scores a row by its
    /// position, the later the better.
    #[derive(Debug)]
    struct Handed(Arc<Mutex<Vec<u32>>>);

    impl Lifecycle for Handed {}

    impl Scorer for Handed {
        fn name(&self) -> &'static str {
            "handed"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            assert_eq!(out.len(), kept.len(), "one score a row handed");
            *self.0.lock().expect("the probe's rows") = kept.to_vec();
            for (o, &row) in out.iter_mut().zip(kept) {
                *o = candidates[row as usize].position as f32 / 8.0;
            }
            TieBand::NORMALIZED
        }
    }

    /// The probe scorer under `picker` over the healthy workers, with the
    /// rows the scorer was handed last.
    fn probed(
        picker: Box<dyn Picker>,
        sessions: Option<SessionRule>,
    ) -> (Engine, Arc<Mutex<Vec<u32>>>) {
        let handed = Arc::new(Mutex::new(Vec::new()));
        let engine = Engine::new(
            vec![Profile {
                name: "probed".into(),
                filters: vec![Box::new(Healthy)],
                scorers: vec![WeightedScorer {
                    scorer: Box::new(Handed(Arc::clone(&handed))),
                    weight: 1.0,
                }],
                picker,
            }],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                shadow: None,
                decode_shadow: None,
                sessions,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        (engine, handed)
    }

    /// The scorers are handed the set the picker compares and no other
    /// survivor: the two drawn under a two-draw picker, the fewest in flight
    /// under count-first, every survivor under a picker that does not
    /// narrow. The trace counts the filters' survivors all the same, and its
    /// top rows are of the set compared.
    #[test]
    fn the_scorers_are_handed_the_set_the_picker_compares() {
        // Five workers up, the first two holding the fewest, and one down.
        let mut pool = [
            candidate(0, 1),
            candidate(1, 1),
            candidate(2, 3),
            candidate(3, 3),
            candidate(4, 3),
            candidate(5, 0),
        ];
        pool[5].health.healthy = false;
        let run = |picker: Box<dyn Picker>, seed: u64| {
            let (engine, handed) = probed(picker, None);
            let facts = RequestFacts {
                model: "m",
                seed,
                ..RequestFacts::default()
            };
            let decision = engine.decide(
                &facts,
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut Scratch::new(),
            );
            let pick = *decision.pick().expect("a pick");
            let rows = handed.lock().expect("the probe's rows").clone();
            let order: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
            let top: Vec<u64> = pick.trace.top[..usize::from(pick.trace.top_len)]
                .iter()
                .map(|(id, _)| id.0 - 100)
                .collect();
            assert_eq!(pick.trace.survivors.get(), Some(5), "the filters' count");
            (rows, order, top)
        };
        let mut pairs = std::collections::BTreeSet::new();
        for seed in 0..64u64 {
            let (rows, order, top) = run(Box::new(PowerOfTwo), seed);
            assert_eq!(rows.len(), 2, "seed {seed}: the two drawn, {rows:?}");
            assert!(rows[0] != rows[1] && rows.iter().all(|&row| row < 5));
            let (better, other) = (rows[0].max(rows[1]), rows[0].min(rows[1]));
            assert_eq!(order, [better, other], "seed {seed}: the later row leads");
            assert_eq!(top, [u64::from(better), u64::from(other)], "seed {seed}");
            pairs.insert((rows[0], rows[1]));

            let (rows, order, top) = run(Box::new(PowerOfTwoCountFirst), seed);
            assert_eq!(rows.len(), 2, "seed {seed}: the two drawn, {rows:?}");
            let count = |row: u32| pool[row as usize].in_flight;
            let lead = *rows
                .iter()
                .min_by_key(|&&row| (count(row), std::cmp::Reverse(row)))
                .expect("two rows");
            assert_eq!(
                order[0], lead,
                "seed {seed}: the fewer in flight, then the later"
            );
            assert_eq!(top.len(), 2, "seed {seed}");

            let (rows, order, top) = run(Box::new(CountFirst), seed);
            assert_eq!(rows.len(), 2, "seed {seed}: {rows:?}");
            assert!(
                rows.contains(&0) && rows.contains(&1),
                "the two that hold one"
            );
            assert_eq!(
                order[..2],
                [1, 0],
                "seed {seed}: the later of the fewest leads"
            );
            assert_eq!(order.len(), 4, "seed {seed}: two passed over follow");
            assert!(order[2] != order[3] && order[2..].iter().all(|row| (2..5).contains(row)));
            assert_eq!(top, [1, 0], "seed {seed}");

            let (rows, order, top) = run(Box::new(Best), seed);
            assert_eq!(rows, [0, 1, 2, 3, 4], "seed {seed}: every survivor");
            assert_eq!(order, [4, 3, 2, 1], "seed {seed}");
            assert_eq!(top, [4, 3, 2], "seed {seed}");
        }
        assert!(pairs.len() > 4, "the draw moves the pair: {pairs:?}");
    }

    /// A picker whose narrowing names more rows than the survivors hold, of
    /// either kind, costs no decision its process: the engine compares what
    /// there is and lets what there is follow.
    #[test]
    fn a_picker_that_claims_more_than_the_survivors_is_held_to_them() {
        /// The first row compared leads; the narrowing claims what it holds.
        #[derive(Debug)]
        struct Claims(Shortlist);
        impl Lifecycle for Claims {}
        impl Picker for Claims {
            fn name(&self) -> &'static str {
                "claims"
            }
            fn narrow(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &mut [u32]) -> Shortlist {
                self.0
            }
            fn pick(
                &self,
                _: &RequestFacts<'_>,
                candidates: &[Candidate],
                kept: &[u32],
                _: &[f32],
                _: TieBand,
                out: &mut Targets,
            ) {
                if let Some(&row) = kept.first() {
                    out.push(target_of(&candidates[row as usize]));
                }
            }
        }
        let pool = [candidate(0, 1), candidate(1, 2), candidate(2, 3)];
        let all = Shortlist {
            compared: usize::MAX,
            alternates: usize::MAX,
        };
        let one = Shortlist {
            compared: 1,
            alternates: usize::MAX,
        };
        for (claim, compared, order) in
            [(all, vec![0, 1, 2], vec![0]), (one, vec![0], vec![0, 1, 2])]
        {
            let (engine, handed) = probed(Box::new(Claims(claim)), None);
            let facts = RequestFacts {
                model: "m",
                ..RequestFacts::default()
            };
            let decision = engine.decide(
                &facts,
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut Scratch::new(),
            );
            let pick = decision.pick().expect("a pick");
            assert_eq!(
                *handed.lock().expect("the probe's rows"),
                compared,
                "{claim:?}"
            );
            let targets: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
            assert_eq!(targets, order, "{claim:?}");
        }
    }

    /// Alternates follow a pick and never stand in for one. A picker that
    /// compares one worker, keeps the rest as alternates and pushes no
    /// target has declined: the decision is a shed that names the picker,
    /// under the profile run and under a sessions rule that delegates, and
    /// no worker it passed over is picked in its place.
    #[test]
    fn a_narrowing_picker_that_declines_sheds_whatever_it_passed_over() {
        /// Compares the first survivor, keeps the rest and declines.
        #[derive(Debug)]
        struct Declines;
        impl Lifecycle for Declines {}
        impl Picker for Declines {
            fn name(&self) -> &'static str {
                "declines"
            }
            fn narrow(&self, _: &RequestFacts<'_>, _: &[Candidate], kept: &mut [u32]) -> Shortlist {
                Shortlist {
                    compared: 1,
                    alternates: kept.len().saturating_sub(1),
                }
            }
            fn pick(
                &self,
                _: &RequestFacts<'_>,
                _: &[Candidate],
                _: &[u32],
                _: &[f32],
                _: TieBand,
                _: &mut Targets,
            ) {
            }
        }
        let pool = [candidate(0, 1), candidate(1, 2), candidate(2, 3)];
        for sessions in [None, Some(SESSIONS)] {
            let keyed = sessions.is_some();
            let site = if keyed {
                "a delegating rule"
            } else {
                "the profile run"
            };
            let (engine, handed) = probed(Box::new(Declines), sessions);
            let decision = if keyed {
                decide_keyed(&engine, &pool, 41, 1_000)
            } else {
                let facts = RequestFacts {
                    model: "m",
                    ..RequestFacts::default()
                };
                engine.decide(
                    &facts,
                    &PoolFacts::default(),
                    &pool,
                    &Constraints::default(),
                    &mut Scratch::new(),
                )
            };
            assert_eq!(*handed.lock().expect("the probe's rows"), [0], "{site}");
            match decision {
                Decision::Shed(shed) => {
                    assert_eq!(shed.reason, ShedReason::PolicyDeclined, "{site}");
                    assert_eq!(shed.emptied_by, "declines", "{site}");
                }
                picked => panic!("{site}: a decline answered {picked:?}"),
            }
        }
    }

    /// The first survivor, unless it is of the refused pair class: then the
    /// picker declines.
    #[derive(Debug)]
    struct RefusesClass(u32);

    impl Lifecycle for RefusesClass {}

    impl Picker for RefusesClass {
        fn name(&self) -> &'static str {
            "refuses-class"
        }

        fn pick(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            _: &[f32],
            _: TieBand,
            out: &mut Targets,
        ) {
            let Some(first) = kept.first().map(|&row| &candidates[row as usize]) else {
                return;
            };
            if first.pair_class != self.0 {
                out.push(target_of(first));
            }
        }
    }

    /// Under the pair plan the prefill side's targets are walked until one
    /// pairs. A narrowing picker keeps the survivors it passed over as
    /// alternates, so when the decode side declines the partners of the
    /// prefill compared, a prefill passed over still pairs: a pair that was a
    /// pick before the narrowing is one after it.
    #[test]
    fn a_prefill_the_picker_passed_over_still_pairs_when_the_one_compared_cannot() {
        // The idle prefill's decode is refused; the busy prefill's is not.
        let mut prefills = [candidate(0, 0), candidate(1, 5)];
        prefills[0].pair_class = 7;
        prefills[1].pair_class = 8;
        let mut decodes = [candidate(0, 0), candidate(1, 0)];
        decodes[0].pair_class = 7;
        decodes[1].pair_class = 8;
        let pickers: [fn() -> Box<dyn Picker>; 3] = [
            || Box::new(CountFirst),
            || Box::new(PowerOfTwo),
            || Box::new(PowerOfTwoCountFirst),
        ];
        for picker in pickers {
            let picker = picker();
            let name = picker.name();
            let engine = Engine::new(
                vec![
                    Profile {
                        name: "prefill".into(),
                        filters: vec![Box::new(Healthy)],
                        scorers: vec![WeightedScorer {
                            scorer: Box::new(LeastRequests),
                            weight: 1.0,
                        }],
                        picker,
                    },
                    Profile {
                        name: "decode".into(),
                        filters: vec![Box::new(Healthy)],
                        scorers: vec![],
                        picker: Box::new(RefusesClass(7)),
                    },
                ],
                vec![Binding {
                    model: "*".into(),
                    phases: vec![],
                    plan: Plan::PdPair,
                    profile: 0,
                    decode_profile: Some(1),
                    shadow: None,
                    decode_shadow: None,
                    sessions: None,
                    overload: None,
                }],
                1,
                Needs::default(),
            );
            for seed in 0..32u64 {
                let facts = RequestFacts {
                    model: "m",
                    seed,
                    ..RequestFacts::default()
                };
                let decision = engine.decide_pair(
                    &facts,
                    &PoolFacts::default(),
                    PairPools {
                        prefills: &prefills,
                        decodes: &decodes,
                        ..PairPools::default()
                    },
                    &Constraints::default(),
                    &Constraints::default(),
                    &mut Scratch::new(),
                );
                let Some(pick) = decision.pick() else {
                    panic!("{name}, seed {seed}: no pair in {decision:?}")
                };
                let first = pick.targets.first().expect("a target");
                assert_eq!(
                    (first.position, first.pair.map(|pair| pair.decode_position)),
                    (1, Some(1)),
                    "{name}, seed {seed}"
                );
            }
        }
    }

    /// Under the pair plan the decode side opens a stream of its own from
    /// the request's seed. The order count-first gives equal counts among
    /// the prefills it passed over shares no value with that stream: when
    /// the lead prefill's decode run sheds, each of two tied alternates
    /// pairs with each of its two decodes about as often. (Ordered by the
    /// decode side's own stream, two of the four pairs are never reached.)
    #[test]
    fn a_passed_over_prefill_pairs_with_every_decode_of_its_class() {
        /// The `random` picker, unless the first survivor is of the refused
        /// pair class: then the picker declines.
        #[derive(Debug)]
        struct RandomUnlessClass(u32);
        impl Lifecycle for RandomUnlessClass {}
        impl Picker for RandomUnlessClass {
            fn name(&self) -> &'static str {
                "random-unless-class"
            }
            fn pick(
                &self,
                facts: &RequestFacts<'_>,
                candidates: &[Candidate],
                kept: &[u32],
                scores: &[f32],
                band: TieBand,
                out: &mut Targets,
            ) {
                let first = kept.first().map(|&row| candidates[row as usize].pair_class);
                if first != Some(self.0) {
                    Random.pick(facts, candidates, kept, scores, band, out);
                }
            }
        }
        // The idle prefill leads and its decode side declines; the two busy
        // prefills tie and hold a class of two decodes each.
        let mut prefills = [candidate(0, 0), candidate(1, 5), candidate(2, 5)];
        for (prefill, class) in prefills.iter_mut().zip([7, 8, 9]) {
            prefill.pair_class = class;
        }
        let mut decodes = [
            candidate(0, 0),
            candidate(1, 0),
            candidate(2, 0),
            candidate(3, 0),
            candidate(4, 0),
        ];
        for (decode, class) in decodes.iter_mut().zip([7, 8, 8, 9, 9]) {
            decode.pair_class = class;
        }
        let engine = Engine::new(
            vec![
                Profile {
                    name: "prefill".into(),
                    filters: vec![Box::new(Healthy)],
                    scorers: vec![WeightedScorer {
                        scorer: Box::new(LeastRequests),
                        weight: 1.0,
                    }],
                    picker: Box::new(CountFirst),
                },
                Profile {
                    name: "decode".into(),
                    filters: vec![Box::new(Healthy)],
                    scorers: vec![],
                    picker: Box::new(RandomUnlessClass(7)),
                },
            ],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::PdPair,
                profile: 0,
                decode_profile: Some(1),
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        // How often each busy prefill pairs with each decode of its class.
        let mut pairs = [[0u32; 2]; 2];
        for seed in 0..4_000u64 {
            let facts = RequestFacts {
                model: "m",
                seed,
                ..RequestFacts::default()
            };
            let decision = engine.decide_pair(
                &facts,
                &PoolFacts::default(),
                PairPools {
                    prefills: &prefills,
                    decodes: &decodes,
                    ..PairPools::default()
                },
                &Constraints::default(),
                &Constraints::default(),
                &mut Scratch::new(),
            );
            let Some(pick) = decision.pick() else {
                panic!("seed {seed}: no pair in {decision:?}")
            };
            let first = pick.targets.first().expect("a target");
            let decode = first.pair.expect("a pair").decode_position;
            assert!(
                (1..=2).contains(&first.position) && (1..=4).contains(&decode),
                "seed {seed}: {first:?}"
            );
            pairs[(first.position - 1) as usize][((decode - 1) % 2) as usize] += 1;
        }
        assert!(
            pairs.iter().flatten().all(|n| (850..=1150).contains(n)),
            "{pairs:?}"
        );
    }

    /// The two slices of a table whose prefill class `k` pairs with the
    /// decode classes `lists[k - 1]`.
    fn table_of(lists: &[&[u32]]) -> (Vec<u32>, Vec<u32>) {
        let ends = lists
            .iter()
            .scan(0u32, |end, list| {
                *end += list.len() as u32;
                Some(*end)
            })
            .collect();
        (ends, lists.concat())
    }

    /// A pair decision under the table of `lists` ([`table_of`]).
    fn decide_tabled(
        engine: &Engine,
        prefills: &[Candidate],
        decodes: &[Candidate],
        lists: &[&[u32]],
    ) -> Decision {
        let (ends, classes) = table_of(lists);
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        engine.decide_pair(
            &facts,
            &PoolFacts::default(),
            PairPools {
                prefills,
                decodes,
                table: PairTable::new(&ends, &classes),
                ..PairPools::default()
            },
            &Constraints::default(),
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }

    /// Routable idle rows of the given pair classes, in order.
    fn classed_rows(classes: &[u32]) -> Vec<Candidate> {
        classes
            .iter()
            .zip(0u32..)
            .map(|(&pair_class, position)| Candidate {
                pair_class,
                ..candidate(position, 0)
            })
            .collect()
    }

    /// The decode positions a pick's targets name, ascending.
    fn decodes_of(decision: &Decision) -> Vec<u32> {
        let mut decodes: Vec<u32> = decision
            .pick()
            .into_iter()
            .flat_map(|pick| pick.targets.iter())
            .filter_map(|target| target.pair.map(|pair| pair.decode_position))
            .collect();
        decodes.sort_unstable();
        decodes
    }

    fn first_decode(decision: &Decision) -> Option<u32> {
        Some(decision.pick()?.targets.first()?.pair?.decode_position)
    }

    /// The reason and the stage of a shed.
    fn shed_of(decision: &Decision) -> Option<(ShedReason, &'static str)> {
        match decision {
            Decision::Shed(shed) => Some((shed.reason, shed.emptied_by)),
            _ => None,
        }
    }

    fn strict_pair_engine() -> Engine {
        Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        )
    }

    #[test]
    fn a_decode_decided_alone_is_picked_among_the_partners_of_its_prefill() {
        let engine = strict_pair_engine();
        let mut prefills = classed_rows(&[1, 2]);
        prefills[1].dp_rank = 5;
        let mut decodes = classed_rows(&[1, 2, 1]);
        decodes[2].dp_rank = 3;
        let (ends, classes) = table_of(&[&[2], &[1]]);
        let decision = engine.decide_decode(
            &request(7),
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                table: PairTable::new(&ends, &classes),
                ..PairPools::default()
            },
            prefills[1].id,
            None,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let targets = &decision
            .pick()
            .expect("partners of the held prefill")
            .targets;
        assert_eq!(decodes_of(&decision), [0, 2]);
        for target in targets.iter() {
            assert_eq!(
                (target.id, target.position, target.dp_rank),
                (prefills[1].id, 1, 5)
            );
            let pair = target.pair.expect("a decode at each target");
            let row = &decodes[pair.decode_position as usize];
            assert_eq!((pair.decode_id, pair.decode_dp_rank), (row.id, row.dp_rank));
        }
    }

    #[test]
    fn a_decode_that_joined_since_is_picked() {
        let engine = Engine::new(
            vec![least_requests_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let prefills = classed_rows(&[1]);
        let mut decodes = classed_rows(&[1, 1]);
        let mut scratch = Scratch::new();
        let facts = request(7);
        let pool = PoolFacts::default();
        let cx = Constraints::default();
        let old = engine.decide_pair(
            &facts,
            &pool,
            PairPools {
                prefills: &prefills,
                decodes: &decodes[..1],
                ..PairPools::default()
            },
            &cx,
            &cx,
            &mut scratch,
        );
        assert_eq!(decodes_of(&old), [0]);
        decodes[0].in_flight = 7;
        let new = engine.decide_decode(
            &facts,
            &pool,
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            prefills[0].id,
            None,
            &cx,
            &mut scratch,
        );
        assert_eq!(first_decode(&new), Some(1));
        assert_eq!(new.trace().seed, facts.seed);
    }

    /// Round robin over the two health tests, with no scorer.
    fn round_robin_profile() -> Profile {
        Profile {
            name: "rotated".into(),
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            scorers: vec![],
            picker: Box::new(RoundRobin::new()),
        }
    }

    /// The engine of one pair profile, on both sides.
    fn pair_engine(profile: Profile) -> Engine {
        Engine::new(vec![profile], vec![pair_binding()], 1, Needs::default())
    }

    /// One request's two decisions over one prefill and `decodes`: the
    /// pair's first, then its decode decided again, handed what `held`
    /// makes of the worker and the rank the first named. `meanwhile` is
    /// what happens to the decodes between the two, told that decode's
    /// position. The position the first names, then those the second does.
    fn decided_twice(
        engine: &Engine,
        decodes: &mut [Candidate],
        held: impl FnOnce((WorkerId, u16)) -> Option<(WorkerId, u16)>,
        meanwhile: impl FnOnce(&mut [Candidate], usize),
    ) -> (u32, Vec<u32>) {
        let prefills = classed_rows(&[1]);
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let (facts, mut scratch) = (request(7), Scratch::new());
        let pools = PairPools {
            prefills: &prefills,
            decodes,
            ..PairPools::default()
        };
        let first = engine.decide_pair(&facts, &pool, pools, &cx, &cx, &mut scratch);
        let target = first.pick().and_then(|pick| pick.targets.first().copied());
        let target = target.expect("a pair");
        let named = target.pair.expect("a decode");
        meanwhile(&mut *decodes, named.decode_position as usize);
        let pools = PairPools {
            prefills: &prefills,
            decodes,
            ..PairPools::default()
        };
        let held = held((named.decode_id, named.decode_dp_rank));
        let again = engine.decide_decode(&facts, &pool, pools, target.id, held, &cx, &mut scratch);
        let again = again.pick().expect("a decode decided again");
        let again = again.targets.iter().filter_map(|target| target.pair);
        let again = again.map(|pair| pair.decode_position).collect();
        (named.decode_position, again)
    }

    /// A rotation is turned once a request where the decode decided again
    /// is handed the decode of its first decision. Requests one at a time
    /// over two decodes: the first decisions reach both evenly and each
    /// second answer is its first. With none handed the call turns the
    /// rotation its first decision turned: every first decision names one
    /// decode, and every second the other, which then serves every request.
    /// The two decodes are two workers, or one worker's two ranks: a first
    /// decode is kept at its rank, and one at a rank no row holds is not.
    /// A scorer beside the picker changes none of it: the picker reads no
    /// score. The scorer still reads the decodes of a kept answer: behind a
    /// busy first decode the idle one tops the trace.
    #[test]
    fn a_decode_decided_again_under_round_robin_keeps_its_first_decode() {
        type Held = fn((WorkerId, u16)) -> Option<(WorkerId, u16)>;
        let cases: [(bool, Held, bool); 4] = [
            (false, Some, true),
            (true, Some, true),
            (false, |_| None, false),
            (true, |(worker, rank)| Some((worker, rank + 2)), false),
        ];
        let cases = cases.map(|case| [(false, case), (true, case)]);
        for (scored, (ranks, held, kept)) in cases.into_iter().flatten() {
            let mut profile = round_robin_profile();
            if scored {
                profile.scorers = least_requests_profile().scorers;
            }
            let engine = pair_engine(profile);
            let mut decodes = classed_rows(&[1, 1]);
            if ranks {
                (decodes[1].id, decodes[1].dp_rank) = (decodes[0].id, 1);
            }
            let mut sent = [0; 2];
            for _ in 0..8 {
                let (first, again) = decided_twice(&engine, &mut decodes, held, |_, _| {});
                assert_eq!(again[0] == first, kept, "{ranks} {scored}");
                sent[again[0] as usize] += 1;
            }
            sent.sort_unstable();
            assert_eq!(sent, if kept { [4, 4] } else { [0, 8] }, "{ranks} {scored}");
        }
        let engine = pair_engine(Profile {
            scorers: least_requests_profile().scorers,
            ..round_robin_profile()
        });
        let (prefills, mut decodes) = (classed_rows(&[1]), classed_rows(&[1, 1]));
        decodes[0].in_flight = 5;
        let kept = engine.decide_decode(
            &request(7),
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            prefills[0].id,
            Some((decodes[0].id, 0)),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert_eq!(first_decode(&kept), Some(0));
        assert_eq!(kept.trace().top[0].0, decodes[1].id);
    }

    /// The kept answer names what a pick names over the same survivors:
    /// the first decode leads, and the survivors behind it follow in their
    /// order, wrapping, to a pick's count. Five decodes and a request at
    /// each: four decodes from the first decision's on. Where no other
    /// decode stays healthy, the first is the whole answer.
    #[test]
    fn a_kept_decode_leads_the_survivors_behind_it() {
        let engine = pair_engine(round_robin_profile());
        let mut decodes = classed_rows(&[1; 5]);
        for _ in 0..5 {
            let (first, again) = decided_twice(&engine, &mut decodes, Some, |_, _| {});
            let behind = (0..MAX_TARGETS as u32).map(|k| (first + k) % 5);
            assert_eq!(again, behind.collect::<Vec<_>>());
        }
        let alone = |rows: &mut [Candidate], at: usize| {
            for (i, row) in rows.iter_mut().enumerate() {
                row.health.healthy = i == at;
            }
        };
        let (first, again) = decided_twice(&engine, &mut decodes, Some, alone);
        assert_eq!(again, [first]);
    }

    /// A first decode that a filter refuses at the second call is not
    /// kept: the picker picks among the decodes the filters keep, by one
    /// turn of their rotation. Each request's first decode stops being
    /// healthy behind its first decision and is healthy again behind its
    /// second: both decisions still reach the three decodes evenly.
    #[test]
    fn a_first_decode_a_filter_refuses_is_picked_again_by_one_turn() {
        let engine = pair_engine(round_robin_profile());
        let mut decodes = classed_rows(&[1, 1, 1]);
        let (mut firsts, mut seconds) = ([0; 3], [0; 3]);
        for _ in 0..6 {
            let unhealthy = |rows: &mut [Candidate], at: usize| rows[at].health.healthy = false;
            let (first, again) = decided_twice(&engine, &mut decodes, Some, unhealthy);
            assert_ne!(again[0], first);
            decodes[first as usize].health.healthy = true;
            firsts[first as usize] += 1;
            seconds[again[0] as usize] += 1;
        }
        assert_eq!((firsts, seconds), ([2; 3], [2; 3]));
    }

    /// A profile whose picker reads its scores decides again, whatever it is
    /// handed. Under least requests the second answer is the first while
    /// nothing moved, and it follows the counts of its own instant away
    /// from a first decode that is still healthy.
    #[test]
    fn a_decode_decided_again_under_a_scorer_follows_the_loads() {
        let engine = pair_engine(least_requests_profile());
        let mut decodes = classed_rows(&[1, 1]);
        decodes[1].in_flight = 3;
        let (first, again) = decided_twice(&engine, &mut decodes, Some, |_, _| {});
        assert_eq!((first, again[0]), (0, 0));
        let busy = |rows: &mut [Candidate], at: usize| rows[at].in_flight = 7;
        let (first, again) = decided_twice(&engine, &mut decodes, Some, busy);
        assert_eq!((first, again[0]), (0, 1));
    }

    /// `passthrough` takes no turn, so it decides again whatever it is
    /// handed: the first healthy decode of the list as it stands at the
    /// second call, with the healthy decodes behind it in the list's order.
    /// The first decode of the list is not healthy at the pair's first
    /// decision, which names the second, and is healthy by the second call.
    #[test]
    fn a_decode_decided_again_under_passthrough_is_the_lists_first() {
        type Held = fn((WorkerId, u16)) -> Option<(WorkerId, u16)>;
        let handed: [Held; 2] = [Some, |_| None];
        for held in handed {
            let engine = pair_engine(Profile {
                picker: Box::new(Passthrough),
                ..round_robin_profile()
            });
            let mut decodes = classed_rows(&[1, 1, 1]);
            decodes[0].health.healthy = false;
            let ahead = |rows: &mut [Candidate], _: usize| rows[0].health.healthy = true;
            let (first, again) = decided_twice(&engine, &mut decodes, held, ahead);
            assert_eq!((first, again), (1, vec![0, 1, 2]));
        }
    }

    #[test]
    fn a_prefill_that_left_is_shed_by_name() {
        let engine = strict_pair_engine();
        let prefills = classed_rows(&[1]);
        let decodes = classed_rows(&[1]);
        let facts = request(7);
        let decision = engine.decide_decode(
            &facts,
            &PoolFacts {
                generation: 8,
                ..PoolFacts::default()
            },
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            },
            WorkerId(987),
            None,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert_eq!(
            shed_of(&decision),
            Some((ShedReason::NoEndpoints, "prefill-gone"))
        );
        assert_eq!(decision.trace().made_under(), Some(0));
        assert_eq!(
            (decision.trace().seed, decision.trace().generation),
            (facts.seed, 8)
        );
    }

    #[test]
    fn a_decode_decided_alone_runs_no_prefill_stage() {
        let engine = Engine::new(
            vec![strict_profile(), least_requests_profile()],
            vec![Binding {
                decode_profile: Some(1),
                ..pair_binding()
            }],
            1,
            Needs::default(),
        );
        let mut prefills = classed_rows(&[1]);
        prefills[0].health.healthy = false;
        let mut decodes = classed_rows(&[1, 1]);
        decodes[0].in_flight = 7;
        let facts = request(7);
        assert_eq!(
            shed_of(&decide_pair(&engine, &prefills, &decodes)),
            Some((ShedReason::NoEndpoints, "healthy"))
        );
        let decision = engine.decide_decode(
            &facts,
            &PoolFacts::default(),
            PairPools {
                prefills: &prefills,
                decodes: &decodes,
                // This column is only read by prefill stages.
                prefill_affinity: &held(&[0, 0]),
                ..PairPools::default()
            },
            prefills[0].id,
            None,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let pick = decision.pick().expect("the decode runs on its own");
        assert_eq!(first_decode(&decision), Some(1));
        assert_eq!(pick.session_verdict, SessionVerdict::None);
    }

    #[test]
    fn a_decode_call_checks_the_decode_view_and_binding() {
        #[derive(Debug)]
        struct Shown(Arc<Mutex<Vec<Phase>>>);

        impl Lifecycle for Shown {}

        impl DecisionScope for Shown {
            fn open(&self, facts: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
                if let Ok(mut phases) = self.0.lock() {
                    phases.push(facts.phase);
                }
            }

            fn close(&self) {}
        }

        impl Filter for Shown {
            fn name(&self) -> &'static str {
                "shown"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn scope(&self) -> Option<&dyn DecisionScope> {
                Some(self)
            }
        }

        let phases = Arc::new(Mutex::new(Vec::new()));
        let engine = Engine::new(
            vec![Profile {
                filters: vec![Box::new(Shown(Arc::clone(&phases)))],
                ..least_requests_profile()
            }],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let prefills = classed_rows(&[1]);
        let decodes = classed_rows(&[1]);
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let facts = request(3);
        let cx = Constraints::default();
        let decide = |engine: &Engine,
                      facts: &RequestFacts<'_>,
                      pools: PairPools<'_>,
                      cx: &Constraints<'_>| {
            engine.decide_decode(
                facts,
                &PoolFacts::default(),
                pools,
                prefills[0].id,
                None,
                cx,
                &mut Scratch::new(),
            )
        };
        let column = held(&[1, 2]);
        let bad_pool = PairPools {
            decode_affinity: &column,
            ..pools
        };
        let bad_req = RequestFacts {
            affinity: &column[..1],
            ..facts
        };
        let mask = Constraints {
            stalled: &[0],
            ..cx
        };
        for (decision, stage) in [
            (decide(&engine, &bad_req, pools, &cx), "affinity-column"),
            (decide(&engine, &facts, bad_pool, &cx), "affinity-column"),
            (decide(&strict_engine(), &facts, pools, &cx), "binding"),
            (decide(&engine, &facts, pools, &mask), "written-rows"),
        ] {
            assert_eq!(
                shed_of(&decision),
                Some((ShedReason::PolicyDeclined, stage))
            );
            // A decline ahead of the decode profile names none. One past it
            // names it.
            let under = (stage == "written-rows").then_some(0);
            assert_eq!(decision.trace().made_under(), under, "{stage}");
        }
        // A call whose prefill left the lists says so ahead of its mask: the
        // prefill is looked for before a constraint is read. The gateway's
        // classic router asks no decode again
        // (`model_gateway/src/routers/common/placement.rs:1227-1275`, at
        // main `c738c8d3b`).
        let left = engine.decide_decode(
            &facts,
            &PoolFacts::default(),
            pools,
            WorkerId(u64::MAX),
            None,
            &mask,
            &mut Scratch::new(),
        );
        assert_eq!(
            shed_of(&left),
            Some((ShedReason::NoEndpoints, "prefill-gone"))
        );
        assert!(phases.lock().expect("the phases shown").is_empty());
        assert!(decide(&engine, &facts, pools, &cx).pick().is_some());
        assert_eq!(*phases.lock().expect("the phases shown"), [Phase::Decode]);
    }

    #[test]
    fn a_class_lists_what_the_table_holds_for_it_and_its_own_number_under_none() {
        let none = PairTable::default();
        assert!(none.is_empty());
        assert_eq!(none.of(&7), [7], "by number");
        assert!(none.of(&0).is_empty(), "class 0 pairs with nothing");

        let (ends, classes) = table_of(&[&[1, 3], &[], &[2]]);
        let table = PairTable::new(&ends, &classes);
        assert!(!table.is_empty());
        assert_eq!(table.of(&1), [1, 3]);
        assert!(table.of(&2).is_empty(), "an empty list");
        assert_eq!(table.of(&3), [2]);
        assert!(table.of(&0).is_empty(), "class 0 under a table");
        assert!(table.of(&4).is_empty(), "a class past the table");
        assert_eq!(table.width(), 4, "one past the largest class named");

        // Ends the lists do not hold cost a class its list.
        let past = PairTable::new(&[3], &[1, 2]);
        assert!(past.of(&1).is_empty());
        let backwards = PairTable::new(&[2, 1], &[1, 2, 3]);
        assert_eq!(backwards.of(&1), [1, 2]);
        assert!(backwards.of(&2).is_empty());
    }

    /// The width's walk reads each list's last entry, whichever list holds
    /// the largest, and the first last entry above the count hands the
    /// table to the list by list read. Both name one width: one past the
    /// largest class a list's own read holds.
    #[test]
    fn a_tables_width_is_one_past_the_largest_class_its_lists_hold() {
        let agree = |table: &PairTable<'_>, width: usize, what: &str| {
            assert_eq!(table.width(), width, "{what}");
            assert_eq!(table.width_by_lists(), width, "{what}, list by list");
        };
        let tables: [(&[&[u32]], usize); 10] = [
            // The largest class in the first list, a middle one, the last.
            (&[&[1, 4], &[], &[2, 3]], 5),
            (&[&[1], &[2, 4], &[3]], 5),
            (&[&[1, 2], &[3, 4]], 5),
            // In a list of one entry behind an empty list.
            (&[&[1], &[], &[2]], 3),
            // An empty list names no class: two entries, the width of one.
            (&[&[1], &[1], &[]], 2),
            // A last entry above the count, in the first list and in the
            // last: its list ends at the classes it holds.
            (&[&[1, 9], &[2, 3]], 4),
            (&[&[2, 3], &[1, 9]], 4),
            // One above the count is above it.
            (&[&[1, 3]], 2),
            // No list holds a class, and no list holds an entry.
            (&[&[8], &[9]], 1),
            (&[&[], &[]], 1),
        ];
        for (lists, width) in tables {
            let (ends, classes) = table_of(lists);
            agree(
                &PairTable::new(&ends, &classes),
                width,
                &format!("{lists:?}"),
            );
        }
        // Ends the entries do not hold: an end past them costs its list,
        // and each list is cut by its own two ends, whatever the last did.
        agree(&PairTable::new(&[3], &[1, 2]), 1, "an end past the entries");
        agree(&PairTable::new(&[3, 0, 1], &[2, 1]), 3, "ends out of order");
    }

    /// The question a host asks once where it builds a table: each clause
    /// is refused by its name, the first a table fails.
    #[test]
    fn a_hosts_validation_names_the_clause_a_table_fails() {
        let validate = |ends: &[u32], classes: &[u32]| PairTable::new(ends, classes).validate();
        assert_eq!(PairTable::default().validate(), Ok(()), "no table");
        assert_eq!(
            validate(&[2, 2, 3], &[1, 3, 2]),
            Ok(()),
            "an empty list too"
        );
        assert_eq!(
            validate(&[2, 1], &[1, 2, 3]),
            Err(PairTableError::EndsDescend)
        );
        assert_eq!(validate(&[3], &[1, 2]), Err(PairTableError::EndPastLists));
        // Ends written as each list's start: every list reads the one before.
        for (ends, classes) in [
            (&[0, 2][..], &[1, 2, 3, 4][..]),
            (&[1], &[1, 2]),
            (&[], &[1]),
        ] {
            assert_eq!(
                validate(ends, classes),
                Err(PairTableError::EntriesPastEnds),
                "{ends:?}"
            );
        }
        for misnamed in [[2, 1], [1, 1]] {
            assert_eq!(
                validate(&[2], &misnamed),
                Err(PairTableError::ListOutOfOrder),
                "{misnamed:?}"
            );
        }
        for zero in [&[0][..], &[0, 2]] {
            let ends = [zero.len() as u32];
            assert_eq!(validate(&ends, zero), Err(PairTableError::ClassZero));
        }
        assert_eq!(
            validate(&[2], &[1, 3]),
            Err(PairTableError::NumberAboveCount)
        );
        // The first clause a table fails is the one named.
        assert_eq!(validate(&[3, 1], &[2, 1]), Err(PairTableError::EndsDescend));
        assert_eq!(validate(&[2], &[0, 0]), Err(PairTableError::ListOutOfOrder));
        assert_eq!(validate(&[2], &[0, 3]), Err(PairTableError::ClassZero));
        // Each clause reads as what it found.
        assert_eq!(
            PairTableError::NumberAboveCount.to_string(),
            "a list names a number above the count of the lists' entries"
        );
    }

    /// The question a host asks once where it builds first partners
    /// beside its table: each clause is refused by its name, the first
    /// they fail.
    #[test]
    fn a_hosts_question_names_the_clause_first_partners_fail() {
        // Two prefill classes that pair with decode classes 1 and 2, and a
        // third with class 3 alone.
        let table = PairTable::new(&[2, 4, 5], &[1, 2, 1, 2, 3]);
        let within =
            |ends: &[u32], classes: &[u32]| FirstPartners::new(ends, classes).within(&table);
        let lists = |clause| Err(FirstPartnersError::Lists(clause));
        assert_eq!(FirstPartners::default().within(&table), Ok(()), "none");
        assert_eq!(within(&[1, 1], &[1]), Ok(()), "one class's alone");
        assert_eq!(within(&[2, 3, 4], &[1, 2, 2, 3]), Ok(()), "every partner");
        // A list names classes of the table, so a number above the lists'
        // own count stands.
        assert_eq!(within(&[0, 0, 1], &[3]), Ok(()), "above the count");
        assert_eq!(within(&[1, 0], &[1]), lists(PairTableError::EndsDescend));
        assert_eq!(within(&[2], &[1]), lists(PairTableError::EndPastLists));
        assert_eq!(
            within(&[1], &[1, 2]),
            lists(PairTableError::EntriesPastEnds)
        );
        assert_eq!(within(&[2], &[2, 1]), lists(PairTableError::ListOutOfOrder));
        assert_eq!(within(&[1], &[0]), lists(PairTableError::ClassZero));
        // A class of the table that is no partner of the list's prefill
        // class, and a list for a prefill class the table does not hold.
        let no_partner = Err(FirstPartnersError::NoPartner);
        assert_eq!(within(&[1], &[3]), no_partner);
        assert_eq!(within(&[0, 0, 0, 1], &[1]), no_partner);
        // Under no table a class pairs with its own number.
        let none = PairTable::default();
        assert_eq!(FirstPartners::new(&[1], &[1]).within(&none), Ok(()));
        assert_eq!(FirstPartners::new(&[1], &[2]).within(&none), no_partner);
        assert_eq!(
            FirstPartnersError::NoPartner.to_string(),
            "a first partner is no partner of its prefill class in the pair table"
        );
    }

    /// One invariant: a host's table costs a decision at most the table's
    /// own size.
    #[test]
    fn a_number_above_the_count_of_a_tables_entries_is_no_class_of_the_table() {
        // Two entries: the count itself is a class, as the largest number
        // of a dense table is.
        let dense = PairTable::new(&[2], &[1, 2]);
        assert_eq!(dense.of(&1), [1, 2]);
        assert_eq!(dense.validate(), Ok(()));
        // One above the count is no class.
        let above = PairTable::new(&[2], &[1, 3]);
        assert_eq!(above.of(&1), [1]);
        assert_eq!(above.validate(), Err(PairTableError::NumberAboveCount));

        // Three entries, two of them near the top of the number's range.
        let far = u32::MAX - 1;
        let entries = [1, far, u32::MAX];
        let sparse = PairTable::new(&[3], &entries);
        assert_eq!(sparse.of(&1), [1], "every read asks here");
        assert_eq!(sparse.validate(), Err(PairTableError::NumberAboveCount));
        assert_eq!(sparse.width(), 2);
        // The partner question: a decode of such a number opens nothing,
        // and the set of bits stays the table's size.
        let mut scratch = Scratch::new();
        let (classes, marks) = (&mut scratch.classes, &mut scratch.marks);
        sparse.partnered([far].into_iter(), classes, marks);
        assert!(classes.is_empty(), "{classes:?}");
        sparse.partnered([far, 1].into_iter(), classes, marks);
        assert_eq!(*classes, [1]);
        // The shed's list: the routable class lists the one class it holds.
        sparse.listed(classes, marks);
        assert_eq!(*classes, [1]);
        assert_eq!(marks.words.len(), 1, "one word of bits");

        // A list of such numbers alone is an empty list.
        let lone = PairTable::new(&[1], &[u32::MAX]);
        assert!(lone.of(&1).is_empty());
        assert_eq!(lone.width(), 1);

        // The cap itself, asked apart: no table's read hands it a number
        // above the count.
        assert_eq!(PairTable::capped(3, 3), 4, "the count is a class");
        assert_eq!(PairTable::capped(far, 3), 4);
        assert_eq!(PairTable::capped(u32::MAX, 0), 1);

        // A list out of order is cut wherever the search lands: the set
        // stays the table's size all the same.
        for unordered in [[u32::MAX, 1, far], [far, u32::MAX, 1], [2, u32::MAX, far]] {
            let table = PairTable::new(&[3], &unordered);
            assert!(table.width() <= 4, "{unordered:?}: {}", table.width());
            assert_eq!(table.validate(), Err(PairTableError::ListOutOfOrder));
        }
    }

    /// Class 0 is no class of the table, whatever a list names: the row
    /// tests pass it over, and the validation names the clause.
    #[test]
    fn a_listed_class_zero_is_no_class_of_the_table() {
        // Alone it is a list of no class: no pass compares a row with it.
        let alone = PairTable::new(&[1], &[0]);
        assert_eq!(alone.of(&1), [0], "a read answers the list as it stands");
        assert_eq!(only(alone.of(&1)), None);
        assert!(!classed(alone.of(&1), 0));
        assert_eq!(alone.validate(), Err(PairTableError::ClassZero));
        // Beside a class it adds no row to that class's.
        let beside = PairTable::new(&[2], &[0, 2]);
        assert_eq!(beside.of(&1), [0, 2]);
        assert_eq!(only(beside.of(&1)), None);
        assert!(classed(beside.of(&1), 2));
        assert!(!classed(beside.of(&1), 0));
        assert_eq!(beside.validate(), Err(PairTableError::ClassZero));
        // The partner question: an open decode class finds the prefill
        // class through the class beside, never through the 0.
        let mut scratch = Scratch::new();
        let (classes, marks) = (&mut scratch.classes, &mut scratch.marks);
        alone.partnered([2].into_iter(), classes, marks);
        assert!(classes.is_empty(), "{classes:?}");
        beside.partnered([2].into_iter(), classes, marks);
        assert_eq!(*classes, [1]);
        // The shed's list holds the 0 a list names, and a decode of class 0
        // is none of the decodes that name the shed.
        beside.listed(classes, marks);
        assert_eq!(*classes, [0, 2]);
        assert!(classed(classes, 2));
        assert!(!classed(classes, 0));
    }

    #[test]
    fn a_set_of_classes_holds_the_numbers_it_was_sized_for() {
        let mut set = ClassBits::default();
        set.reset(130);
        for class in [0, 1, 63, 64, 65, 129] {
            assert!(set.insert(class), "{class}");
        }
        assert!(!set.insert(192), "a number past the set's last word");
        assert!([0, 1, 63, 64, 65, 129].iter().all(|&class| set.has(class)));
        assert!(![2, 62, 66, 128, 192, u32::MAX]
            .iter()
            .any(|&class| set.has(class)));
        assert_eq!(set.classes().collect::<Vec<_>>(), [0, 1, 63, 64, 65, 129]);
        // A reset empties it, whatever its new width.
        set.reset(1);
        assert_eq!(set.classes().count(), 0);
        assert!(!set.has(1));
    }

    #[test]
    fn a_prefill_pairs_with_the_decodes_of_every_class_its_class_lists() {
        let engine = Engine::new(
            vec![least_requests_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        // Decode class 3 pairs with both prefill classes and stands once;
        // the last decode is outside any pairing.
        let mut prefills = classed_rows(&[1, 2]);
        let decodes = classed_rows(&[1, 2, 3, 0]);
        let lists: [&[u32]; 2] = [&[1, 3], &[2, 3]];
        prefills[1].in_flight = 5;
        let first = decide_tabled(&engine, &prefills, &decodes, &lists);
        assert_eq!(lead(&first), Some(0));
        assert_eq!(decodes_of(&first), [0, 2], "class 1's decodes: {first:?}");
        prefills[0].in_flight = 9;
        let second = decide_tabled(&engine, &prefills, &decodes, &lists);
        assert_eq!(lead(&second), Some(1));
        assert_eq!(decodes_of(&second), [1, 2], "class 2's decodes: {second:?}");
    }

    #[test]
    fn a_table_says_which_classes_pair_where_none_leaves_it_to_the_numbers() {
        let engine = strict_pair_engine();
        let prefills = classed_rows(&[1]);
        // The second decode's class stands at the edge of a word of bits.
        let decodes = classed_rows(&[1, 64]);
        let by_number = decide_pair(&engine, &prefills, &decodes);
        assert_eq!(decodes_of(&by_number), [0], "{by_number:?}");
        // A second prefill class names the 63 classes under it: the table
        // numbers densely.
        let under: Vec<u32> = (1..=63).collect();
        let by_table = decide_tabled(&engine, &prefills, &decodes, &[&[64], &under]);
        assert_eq!(decodes_of(&by_table), [1], "{by_table:?}");
        // A class the table lists nothing for has no partner, whatever its
        // number; nor has a class past the table.
        let unpartnered = Some((ShedReason::NoEndpoints, "partnered"));
        assert_eq!(
            shed_of(&decide_tabled(&engine, &prefills, &decodes, &[&[]])),
            unpartnered
        );
        let past = classed_rows(&[2]);
        assert_eq!(
            shed_of(&decide_tabled(&engine, &past, &decodes, &[&[1]])),
            unpartnered
        );
    }

    #[test]
    fn a_prefill_class_has_a_partner_when_any_class_it_lists_holds_an_open_decode() {
        let engine = strict_pair_engine();
        let prefills = classed_rows(&[1, 2]);
        let mut decodes = classed_rows(&[1, 2, 3]);
        let lists: [&[u32]; 2] = [&[1, 2], &[3]];
        // Class 1's first listed class is down and its second is up; class
        // 2's one class is down.
        decodes[0].health.healthy = false;
        decodes[2].health.healthy = false;
        let decision = decide_tabled(&engine, &prefills, &decodes, &lists);
        assert_eq!(lead(&decision), Some(0), "{decision:?}");
        assert_eq!(decodes_of(&decision), [1]);
        // The other way round: class 2 alone has a partner.
        decodes[1].health.healthy = false;
        decodes[2].health.healthy = true;
        let decision = decide_tabled(&engine, &prefills, &decodes, &lists);
        assert_eq!(lead(&decision), Some(1), "{decision:?}");
        assert_eq!(decodes_of(&decision), [2]);
    }

    #[test]
    fn under_a_table_a_shed_is_named_by_the_decodes_a_routable_prefill_lists() {
        let engine = strict_pair_engine();
        // A routable prefill of class 1 lists class 2, whose decode is
        // unhealthy; an unhealthy prefill of class 2 lists that class and
        // class 3, whose decode is overloaded; a healthy decode of class 1,
        // the routable prefill's own number, is listed by neither.
        let mut prefills = classed_rows(&[1, 2]);
        prefills[1].health.healthy = false;
        let mut decodes = classed_rows(&[2, 3, 1]);
        decodes[0].health.healthy = false;
        decodes[1].health.overloaded = true;
        let lists: [&[u32]; 2] = [&[2], &[2, 3]];
        assert_eq!(
            shed_of(&decide_tabled(&engine, &prefills, &decodes, &lists)),
            Some((ShedReason::NoEndpoints, "healthy")),
            "the routable prefill's partner names the shed"
        );
        // Both prefills routable: the stage that drops the last of their
        // partners names it, and that one is saturation.
        prefills[1].health.healthy = true;
        assert_eq!(
            shed_of(&decide_tabled(&engine, &prefills, &decodes, &lists)),
            Some((ShedReason::Saturated, "not-overloaded"))
        );
        // By number the routable prefill's partner is the decode of its own
        // class, whatever the number: an overloaded one is saturation under
        // a binding that sheds.
        let mut engine = engine;
        engine.shed_when_all_vetoed(0);
        let prefills = classed_rows(&[100]);
        let mut decodes = classed_rows(&[100]);
        decodes[0].health.overloaded = true;
        assert_eq!(
            shed_of(&decide_pair(&engine, &prefills, &decodes)),
            Some((ShedReason::Saturated, "not-overloaded"))
        );
    }

    #[test]
    fn by_number_a_class_is_found_wherever_its_decode_stands_in_the_pool() {
        let engine = strict_pair_engine();
        let prefills = classed_rows(&[1]);
        // The classes of the open decodes arrive out of order.
        let decodes = classed_rows(&[2, 3, 1]);
        let decision = decide_pair(&engine, &prefills, &decodes);
        assert_eq!(decodes_of(&decision), [2], "{decision:?}");
    }

    /// The partner question asks the hard filters alone: a class whose one
    /// decode a retry just tried keeps its prefill, and the soft filter then
    /// steps aside.
    #[test]
    fn a_decode_a_retry_just_tried_still_partners_its_prefill() {
        let engine = Engine::new(
            vec![least_requests_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let prefills = classed_rows(&[1]);
        let decodes = classed_rows(&[1]);
        let tried = [decodes[0].id];
        let decode_cx = Constraints {
            excluded: &tried,
            ..Constraints::default()
        };
        for table in [PairTable::default(), PairTable::new(&[1], &[1])] {
            let decision = engine.decide_pair(
                &RequestFacts {
                    model: "m",
                    ..RequestFacts::default()
                },
                &PoolFacts::default(),
                PairPools {
                    prefills: &prefills,
                    decodes: &decodes,
                    table,
                    ..PairPools::default()
                },
                &Constraints::default(),
                &decode_cx,
                &mut Scratch::new(),
            );
            assert_eq!(decodes_of(&decision), [0], "{decision:?}");
        }
    }

    /// Hands the first survivor alone: no alternate follows it.
    #[derive(Debug)]
    struct FirstKept;

    impl Lifecycle for FirstKept {}

    impl Picker for FirstKept {
        fn name(&self) -> &'static str {
            "first-kept"
        }

        fn pick(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            _: &[f32],
            _: TieBand,
            out: &mut Targets,
        ) {
            if let Some(&row) = kept.first() {
                out.push(target_of(&candidates[row as usize]));
            }
        }
    }

    /// Hands the last survivor alone.
    #[derive(Debug)]
    struct LastKept;

    impl Lifecycle for LastKept {}

    impl Picker for LastKept {
        fn name(&self) -> &'static str {
            "last-kept"
        }

        fn pick(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            _: &[f32],
            _: TieBand,
            out: &mut Targets,
        ) {
            if let Some(&row) = kept.last() {
                out.push(target_of(&candidates[row as usize]));
            }
        }
    }

    /// A pair's side is decided once more by its shadow profile, over the
    /// rows its live side decided over. The prefill's shadow reads the
    /// prefills that have a partner. The decode's shadow reads the
    /// partners of the live pair's prefill: a decode that pairs with
    /// another prefill alone is never its pick. A side the binding names
    /// no shadow for is not asked, and a pair that names none has no
    /// word for the door.
    #[test]
    fn a_pairs_side_is_decided_once_more_over_the_rows_its_live_side_saw() {
        let profile = |name: &str, picker: Box<dyn Picker>| Profile {
            name: name.into(),
            filters: Vec::new(),
            scorers: Vec::new(),
            picker,
        };
        let engine = |shadow: Option<u16>, decode_shadow: Option<u16>| {
            let binding = Binding {
                shadow,
                decode_shadow,
                ..pair_binding()
            };
            let profiles = vec![
                profile("live", Box::new(FirstKept)),
                profile("prefill-next", Box::new(LastKept)),
                profile("decode-next", Box::new(LastKept)),
            ];
            Engine::new(profiles, vec![binding], 1, Needs::default())
        };
        // The first prefill pairs with the first two decodes, the second
        // with the third alone, and the last with none.
        let of_class = |position, pair_class| Candidate {
            pair_class,
            ..candidate(position, 0)
        };
        let prefills = [of_class(0, 1), of_class(1, 2), of_class(2, 3)];
        let decodes = [of_class(0, 1), of_class(1, 1), of_class(2, 2)];
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let both = engine(Some(1), Some(2));
        let live = both.decide_pair(&facts, &pool, pools, &cx, &cx, &mut scratch);
        let pick = live.pick().expect("a pair");
        let first = pick.targets.first().expect("the pair's first target");
        let decode = first.pair.map(|pair| pair.decode_position);
        assert_eq!((first.position, decode), (0, Some(0)));
        let word = both.shadow_for("m", Phase::Prefill);
        assert_eq!(word, Some((0, ShadowRows::Sides)));
        let mut side = |engine: &Engine, side| {
            let shadow =
                engine.shadow_side(0, side, &facts, &pool, pools, &cx, &cx, pick, &mut scratch);
            shadow.map(|shadow| lead(&shadow))
        };
        // The last prefill with a partner: never the one with none.
        assert_eq!(side(&both, PairSide::Prefill), Some(Some(1)));
        // The last partner of the live prefill: never the third decode.
        assert_eq!(side(&both, PairSide::Decode), Some(Some(1)));
        let decode_alone = engine(None, Some(2));
        assert_eq!(side(&decode_alone, PairSide::Prefill), None);
        assert_eq!(side(&decode_alone, PairSide::Decode), Some(Some(1)));
        assert!(decode_alone.shadows_side(0, PairSide::Decode));
        assert!(!decode_alone.shadows_side(0, PairSide::Prefill));
        // A decode decided after its prefill: its shadow reads the
        // partners of the prefill that is held, whichever the first
        // decision's shadow took.
        let call = RequestFacts {
            phase: Phase::Decode,
            ..facts
        };
        let mut later = |engine: &Engine, held: WorkerId| {
            let shadow =
                engine.shadow_decode_later(0, &call, &pool, pools, held, pick, &cx, &mut scratch);
            shadow.map(|shadow| lead(&shadow))
        };
        assert_eq!(later(&both, prefills[0].id), Some(Some(1)));
        assert_eq!(later(&both, prefills[1].id), Some(Some(2)));
        assert_eq!(later(&both, WorkerId::default()), None, "the prefill left");
        let prefill_alone = engine(Some(1), None);
        assert_eq!(later(&prefill_alone, prefills[0].id), None, "not named");
        assert_eq!(engine(None, None).shadow_for("m", Phase::Prefill), None);
    }

    /// The prefill's shadow asks no stage of a live profile again. The
    /// live decision's scope closes with it, and a stage asked behind it
    /// answers by its fallback. The live decode profile's filter here
    /// keeps a row only while its decision stands open: asked again, it
    /// would refuse every partner, and the shadow would be shown no
    /// prefill. The classes the live prefill was shown stand in the
    /// scratch, and the shadow reads them.
    #[test]
    fn a_pair_sides_shadow_asks_no_stage_of_a_live_profile_again() {
        use std::sync::atomic::{AtomicBool, Ordering};

        /// Keeps every row while its decision stands open, and none
        /// behind it.
        #[derive(Debug)]
        struct WhileOpen(Arc<AtomicBool>);

        impl Filter for WhileOpen {
            fn name(&self) -> &'static str {
                "while-open"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                self.0.load(Ordering::Relaxed)
            }
        }

        let open = Arc::new(AtomicBool::new(true));
        let live = Profile {
            name: "live".into(),
            filters: vec![Box::new(WhileOpen(Arc::clone(&open)))],
            scorers: Vec::new(),
            picker: Box::new(FirstKept),
        };
        let next = |name: &str| Profile {
            name: name.into(),
            filters: Vec::new(),
            scorers: Vec::new(),
            picker: Box::new(LastKept),
        };
        let binding = Binding {
            shadow: Some(1),
            decode_shadow: Some(2),
            ..pair_binding()
        };
        let profiles = vec![live, next("prefill-next"), next("decode-next")];
        let engine = Engine::new(profiles, vec![binding], 1, Needs::default());
        // The first prefill pairs with the first two decodes, and the
        // second with the third alone.
        let of_class = |position, pair_class| Candidate {
            pair_class,
            ..candidate(position, 0)
        };
        let prefills = [of_class(0, 1), of_class(1, 2)];
        let decodes = [of_class(0, 1), of_class(1, 1), of_class(2, 2)];
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let decided = engine.decide_pair(&facts, &pool, pools, &cx, &cx, &mut scratch);
        let pick = decided.pick().expect("a pair");
        // The live decision is over: its stages answer as behind a
        // closed scope.
        open.store(false, Ordering::Relaxed);
        let mut side = |side| {
            let shadow =
                engine.shadow_side(0, side, &facts, &pool, pools, &cx, &cx, pick, &mut scratch);
            shadow.map(|shadow| lead(&shadow))
        };
        assert_eq!(side(PairSide::Prefill), Some(Some(1)));
        assert_eq!(side(PairSide::Decode), Some(Some(1)));
    }

    /// The `cache_aware` preset's profile with its prefix stage alone
    /// ahead of the load, reading cached blocks from the index: a profile
    /// that reads the column by no warm-up ruling and no record.
    fn by_index() -> spec::ProfileSpec {
        use crate::document::spec::{AffinitySource, PrefixAffinityParams, ScoreStage};
        let aware = PolicySpec::preset("cache_aware").expect("a preset");
        let profile = aware.profiles.into_values().next();
        let mut profile = profile.expect("the preset's profile");
        let prefix = PrefixAffinityParams {
            weight: 4.0,
            source: Some(AffinitySource::Index),
            ..PrefixAffinityParams::default()
        };
        profile.score = vec![
            ScoreStage::PrefixAffinity(prefix).into(),
            "least-load".into(),
        ];
        profile
    }

    /// The stock pair's document under `decode` as its decode preset,
    /// with `shadow` as the decode side's shadow profile.
    fn decode_shadowed_document(decode: &str, shadow: spec::ProfileSpec) -> PolicySpec {
        use crate::document::spec::ShadowSpec;
        let document = PolicySpec::pair_presets("least_load", decode);
        let mut document = document.expect("the stock presets");
        document.profiles.insert("decode-next".into(), shadow);
        let sides = [("decode".to_owned(), "decode-next".to_owned())];
        document.bindings[0].shadow = Some(ShadowSpec::Sides(sides.into()));
        document
    }

    /// The engine of that document.
    fn decode_shadowed(decode: &str, shadow: spec::ProfileSpec) -> Engine {
        let document = decode_shadowed_document(decode, shadow);
        document.validate().expect("a pair with a decode shadow")
    }

    /// A side's shadow reads what its profile needs: its needs join the
    /// side's, so the column it reads is filled for the pair's pool. The
    /// other side's needs stay its own. Under a binding of the decode
    /// phase listed ahead of the pair's, the decode pool still carries
    /// what the pair's decode shadow reads.
    #[test]
    fn a_pair_sides_shadow_joins_its_needs_to_the_sides() {
        use crate::document::spec::{BindingSpec, Match};
        let alone = PolicySpec::pair_presets("least_load", "least_load");
        let alone = alone.expect("the stock presets").validate();
        let alone = alone.expect("the pair document");
        assert!(!alone.reads_for("m", Phase::Decode).0.affinity);
        let engine = decode_shadowed("least_load", by_index());
        assert!(
            !engine.side_apart(0, PairSide::Decode),
            "the shadow is asked"
        );
        assert!(engine.reads_for("m", Phase::Decode).0.affinity);
        assert!(engine.needs_for_phase(Phase::Decode).affinity);
        assert!(!engine.reads_for("m", Phase::Prefill).0.affinity);
        // A decode asked alone resolves to the binding ahead, which reads
        // no column: the pair's decode side brings its shadow's needs.
        let mut ahead = decode_shadowed_document("least_load", by_index());
        let lone = ahead.profiles.get("decode").cloned();
        let lone = lone.expect("the decode side's profile");
        ahead.profiles.insert("lone".into(), lone);
        let first = BindingSpec {
            matcher: Match {
                phase: vec![Phase::Decode],
                ..Match::default()
            },
            profile: Some("lone".into()),
            ..BindingSpec::default()
        };
        ahead.bindings.insert(0, first);
        let ahead = ahead.validate().expect("a decode binding ahead");
        assert!(ahead.reads_for("m", Phase::Decode).0.affinity);
    }

    /// A pair names no rows: a side reads the column its pool carries,
    /// by the live profile's ruling and record. A side's shadow of
    /// another ruling or another record is named and not asked. One that
    /// reads the column as the pool carries it is asked, and so is one
    /// that reads no column.
    #[test]
    fn a_pair_sides_shadow_of_another_column_is_not_asked() {
        let preset = |name: &str| {
            let preset = PolicySpec::preset(name).expect("a preset");
            let profile = preset.profiles.into_values().next();
            profile.expect("the preset's profile")
        };
        let apart = |decode: &str, shadow: spec::ProfileSpec| {
            let engine = decode_shadowed(decode, shadow);
            assert!(engine.shadows_side(0, PairSide::Decode), "named");
            assert!(!engine.side_apart(0, PairSide::Prefill), "not named");
            engine.side_apart(0, PairSide::Decode)
        };
        // Behind a side that reads no cached blocks: its pool has no
        // ruling and keeps no record.
        assert!(apart("least_load", preset("cache_aware")));
        assert!(!apart("least_load", by_index()));
        assert!(!apart("least_load", preset("least_load")));
        // Behind the preset that rules its warming rows and reads the
        // record.
        assert!(!apart("cache_aware", preset("cache_aware")));
        assert!(apart("cache_aware", by_index()));
        assert!(!apart("cache_aware", preset("least_load")));
    }

    #[test]
    fn a_scratch_carries_no_class_from_one_pair_decision_into_the_next() {
        let engine = Engine::new(
            vec![Profile {
                name: "first".into(),
                filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
                scorers: vec![],
                picker: Box::new(FirstKept),
            }],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let prefills = classed_rows(&[1, 2]);
        let mut decodes = classed_rows(&[1, 2]);
        let (ends, classes) = table_of(&[&[1], &[2]]);
        for table in [PairTable::default(), PairTable::new(&ends, &classes)] {
            let mut scratch = Scratch::new();
            let mut decide = |decodes: &[Candidate]| {
                engine.decide_pair(
                    &RequestFacts {
                        model: "m",
                        ..RequestFacts::default()
                    },
                    &PoolFacts::default(),
                    PairPools {
                        prefills: &prefills,
                        decodes,
                        table,
                        ..PairPools::default()
                    },
                    &Constraints::default(),
                    &Constraints::default(),
                    &mut scratch,
                )
            };
            decodes[0].health.healthy = true;
            assert_eq!(lead(&decide(&decodes)), Some(0), "each class has a partner");
            // The first class's decode goes down: the second prefill alone
            // has a partner, and the picker hands no alternate.
            decodes[0].health.healthy = false;
            let next = decide(&decodes);
            assert_eq!(lead(&next), Some(1), "{next:?}");
        }
    }

    /// Class 0 pairs with nothing at the decode run too: a picker that hands
    /// a class-0 prefill the filters dropped gets no decode for it, under no
    /// table as under one.
    #[test]
    fn a_class_zero_prefill_pairs_with_no_decode_whoever_hands_it() {
        /// Hands the pool's first row, whichever rows the filters kept.
        #[derive(Debug)]
        struct FirstRow;
        impl Lifecycle for FirstRow {}
        impl Picker for FirstRow {
            fn name(&self) -> &'static str {
                "first-row"
            }
            fn pick(
                &self,
                _: &RequestFacts<'_>,
                candidates: &[Candidate],
                _: &[u32],
                _: &[f32],
                _: TieBand,
                out: &mut Targets,
            ) {
                if let Some(first) = candidates.first() {
                    out.push(target_of(first));
                }
            }
        }
        let engine = Engine::new(
            vec![
                Profile {
                    name: "prefill".into(),
                    filters: vec![Box::new(Healthy)],
                    scorers: vec![],
                    picker: Box::new(FirstRow),
                },
                strict_profile(),
            ],
            vec![Binding {
                decode_profile: Some(1),
                ..pair_binding()
            }],
            1,
            Needs::default(),
        );
        // The second prefill pairs with the second decode, so the prefill
        // pool keeps a row; the picker hands the first, of class 0, beside a
        // decode of class 0.
        let prefills = classed_rows(&[0, 1]);
        let decodes = classed_rows(&[0, 1]);
        let lists: [&[u32]; 1] = [&[1]];
        for (decision, how) in [
            (decide_pair(&engine, &prefills, &decodes), "by number"),
            (
                decide_tabled(&engine, &prefills, &decodes, &lists),
                "by a table",
            ),
        ] {
            assert_eq!(
                shed_of(&decision),
                Some((ShedReason::NoEndpoints, "partners")),
                "{how}: {decision:?}"
            );
        }
    }

    /// A list that names class 0 is not valid: a debug build stops at the
    /// clause, and a release build pairs no decode of class 0 through it.
    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "the pair table is not valid: a list names class 0")
    )]
    fn a_listed_class_zero_stops_a_debug_build_and_pairs_nothing_in_a_release_build() {
        // Alone it is no partner, and beside a class it adds no decode to
        // that class's.
        let engine = strict_pair_engine();
        let prefills = classed_rows(&[1]);
        let decodes = classed_rows(&[0, 2]);
        assert_eq!(
            shed_of(&decide_tabled(&engine, &prefills, &decodes, &[&[0]])),
            Some((ShedReason::NoEndpoints, "partnered"))
        );
        let beside = decide_tabled(&engine, &prefills, &decodes, &[&[0, 2]]);
        assert_eq!(decodes_of(&beside), [1], "{beside:?}");
        // Nor is a decode of class 0 a partner the decode profile lost: an
        // unhealthy one beside that list names no stage of the shed.
        let mut down = classed_rows(&[0, 2]);
        down[0].health.healthy = false;
        assert_eq!(
            shed_of(&decide_tabled(&engine, &prefills, &down, &[&[0]])),
            Some((ShedReason::NoEndpoints, "partnered"))
        );
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(
            expected = "the pair table is not valid: a list names its classes out of ascending order"
        )
    )]
    fn a_misnamed_table_stops_a_debug_build_and_costs_a_release_build_its_pairs() {
        let engine = strict_pair_engine();
        let prefills = classed_rows(&[1]);
        // A list out of order: the process stands, whatever the list pairs.
        let decodes = classed_rows(&[1, 2]);
        let decision = decide_tabled(&engine, &prefills, &decodes, &[&[2, 1]]);
        assert!(decision.pick().is_some() || shed_of(&decision).is_some());
    }

    /// One invariant, over drawn fleets: a table over one row a decode
    /// decides as the same fleet spelled without one, a decode row for each
    /// prefill class it pairs with: the same shed by reason and stage, the
    /// same prefills and the same decodes in the same order.
    #[test]
    fn a_table_decides_as_the_same_fleet_spelled_a_decode_row_a_prefill_class() {
        fn engine_of(picker: Box<dyn Picker>) -> Engine {
            let profile = Profile {
                name: "drawn".into(),
                filters: vec![
                    Box::new(Healthy),
                    Box::new(NotOverloaded),
                    Box::new(NotTried),
                ],
                scorers: vec![WeightedScorer {
                    scorer: Box::new(LeastRequests),
                    weight: 1.0,
                }],
                picker,
            };
            Engine::new(vec![profile], vec![pair_binding()], 1, Needs::default())
        }
        /// A row whose class runs from 0 (outside any pairing) to one past
        /// its side's last class (a class the table does not hold).
        fn row(draw: &mut Draw, position: u32, classes: u32) -> Candidate {
            Candidate {
                pair_class: draw.below(classes as usize + 2) as u32,
                health: Health {
                    healthy: draw.below(5) != 0,
                    overloaded: draw.below(4) == 0,
                    ..Health::default()
                },
                ..candidate(position, draw.below(6) as u32)
            }
        }
        /// What a decision says, by worker: a shed's reason and stage, or
        /// each target's prefill and decode; and the survivors it counted.
        type Said = (
            Option<(ShedReason, &'static str)>,
            Vec<(WorkerId, Option<WorkerId>)>,
            Survivors,
        );
        fn said(decision: &Decision) -> Said {
            match decision {
                Decision::Pick(pick) => (
                    None,
                    pick.targets
                        .iter()
                        .map(|target| (target.id, target.pair.map(|pair| pair.decode_id)))
                        .collect(),
                    pick.trace.survivors,
                ),
                Decision::Shed(shed) => (
                    Some((shed.reason, shed.emptied_by)),
                    Vec::new(),
                    shed.trace.survivors,
                ),
                Decision::Queue(_) => panic!("no profile here queues"),
            }
        }
        let engines = [
            engine_of(Box::new(Best)),
            engine_of(Box::new(PowerOfTwo)),
            engine_of(Box::new(CountFirst)),
        ];
        let mut draw = Draw::new(0x7ab1e);
        let (mut picks, mut sheds) = (0u32, 0u32);
        for fleet in 0..600u64 {
            let prefill_classes = 1 + draw.below(4) as u32;
            let decode_classes = 1 + draw.below(5) as u32;
            let mut lists: Vec<Vec<u32>> = (0..prefill_classes)
                .map(|_| {
                    (1..=decode_classes)
                        .filter(|_| draw.below(2) == 0)
                        .collect()
                })
                .collect();
            // A table numbers densely: a class no list drew joins one.
            for class in 1..=decode_classes {
                if !lists.iter().any(|list| list.contains(&class)) {
                    let list = &mut lists[draw.below(prefill_classes as usize)];
                    list.push(class);
                    list.sort_unstable();
                }
            }
            let prefills: Vec<Candidate> = (0..draw.below(6) as u32)
                .map(|position| row(&mut draw, position, prefill_classes))
                .collect();
            let decodes: Vec<Candidate> = (0..draw.below(9) as u32)
                .map(|position| row(&mut draw, position, decode_classes))
                .collect();
            // The same fleet without a table: a decode row for each prefill
            // class that lists its class, of that class's number; a decode
            // no class lists keeps one row, outside any pairing.
            let mut rows: Vec<Candidate> = Vec::new();
            for decode in &decodes {
                let before = rows.len();
                rows.extend(
                    lists
                        .iter()
                        .zip(1u32..)
                        .filter(|(list, _)| {
                            decode.pair_class != 0 && list.contains(&decode.pair_class)
                        })
                        .map(|(_, pair_class)| Candidate {
                            pair_class,
                            ..*decode
                        }),
                );
                if rows.len() == before {
                    rows.push(Candidate {
                        pair_class: 0,
                        ..*decode
                    });
                }
            }
            for (row, position) in rows.iter_mut().zip(0u32..) {
                row.position = position;
            }
            let lists: Vec<&[u32]> = lists.iter().map(Vec::as_slice).collect();
            let (ends, classes) = table_of(&lists);
            // Some workers were just tried: the soft filter steps in.
            let excluded: Vec<WorkerId> = (0..draw.below(3) as u64)
                .map(|_| WorkerId(100 + draw.below(9) as u64))
                .collect();
            let cx = Constraints {
                excluded: &excluded,
                ..Constraints::default()
            };
            for (engine, seed) in engines.iter().zip(fleet..) {
                let facts = RequestFacts {
                    model: "m",
                    seed,
                    ..RequestFacts::default()
                };
                let decide = |pools: PairPools<'_>| {
                    engine.decide_pair(
                        &facts,
                        &PoolFacts::default(),
                        pools,
                        &cx,
                        &cx,
                        &mut Scratch::new(),
                    )
                };
                let tabled = decide(PairPools {
                    prefills: &prefills,
                    decodes: &decodes,
                    table: PairTable::new(&ends, &classes),
                    ..PairPools::default()
                });
                let spelled = decide(PairPools {
                    prefills: &prefills,
                    decodes: &rows,
                    ..PairPools::default()
                });
                assert_eq!(
                    said(&tabled),
                    said(&spelled),
                    "fleet {fleet}, seed {seed}: lists {lists:?} over {prefills:?} and {decodes:?}"
                );
                match tabled {
                    Decision::Pick(_) => picks += 1,
                    _ => sheds += 1,
                }
            }
        }
        assert!(picks >= 300 && sheds >= 300, "{picks} picks, {sheds} sheds");
    }

    /// A sessions rule that delegates runs the profile's own stages through
    /// the profile run's one home: a keyed request's placement hands the
    /// scorers the set the picker compares, as an unkeyed decision does.
    #[test]
    fn a_delegating_sessions_rule_narrows_as_the_profile_run_does() {
        let (engine, handed) = probed(Box::new(CountFirst), Some(SESSIONS));
        let pool = [
            candidate(0, 4),
            candidate(1, 2),
            candidate(2, 2),
            candidate(3, 9),
        ];
        let decision = decide_keyed(&engine, &pool, 41, 1_000);
        let pick = decision.pick().expect("a pick");
        assert_eq!(
            *handed.lock().expect("the probe's rows"),
            [1, 2],
            "the two that hold two"
        );
        let order: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
        assert_eq!(
            order,
            [2, 1, 0, 3],
            "the later of the fewest, then by count"
        );
        assert_eq!(pick.trace.survivors.get(), Some(4));
    }

    #[test]
    fn bindings_match_globs_and_phases_in_order() {
        let b = Binding {
            model: "pd/*".into(),
            phases: vec![Phase::Prefill],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: None,
            overload: None,
        };
        assert!(b.matches("pd/a", Phase::Prefill));
        assert!(!b.matches("pd/a", Phase::Single));
        assert!(!b.matches("x/a", Phase::Prefill));
        let any = Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: None,
            overload: None,
        };
        assert!(any.matches("anything", Phase::Decode));
        let suffix = Binding {
            model: "*-chat".into(),
            ..any
        };
        assert!(suffix.matches("x-chat", Phase::Single));
        assert!(!suffix.matches("x-code", Phase::Single));
    }

    #[test]
    fn a_profile_band_is_the_weighted_sum_of_its_scorers_bands() {
        /// A scorer that writes equal scores and publishes a wide band.
        #[derive(Debug)]
        struct Wide;
        impl Lifecycle for Wide {}
        impl Scorer for Wide {
            fn name(&self) -> &'static str {
                "wide"
            }
            fn score(
                &self,
                _: &RequestFacts<'_>,
                _: &[Candidate],
                _: &[u32],
                out: &mut [f32],
            ) -> TieBand {
                out.fill(0.5);
                TieBand(0.2)
            }
        }
        let profile = Profile {
            name: "mixed".into(),
            filters: vec![],
            scorers: vec![
                WeightedScorer {
                    scorer: Box::new(Wide),
                    weight: 3.0,
                },
                WeightedScorer {
                    scorer: Box::new(LeastRequests),
                    weight: 1.0,
                },
            ],
            picker: Box::new(Best),
        };
        let pool = [candidate(0, 1), candidate(1, 1)];
        let kept = [0u32, 1];
        let (mut scores, mut stage) = (Vec::new(), Vec::new());
        let band = combine(
            &profile,
            &RequestFacts::default(),
            &pool,
            &kept,
            &mut scores,
            &mut stage,
        );
        let expected = 0.75 * 0.2 + 0.25 * TieBand::NORMALIZED.0;
        assert!((band.0 - expected).abs() < 1e-7, "{band:?}");
        assert!(scores
            .iter()
            .all(|&s| (s - (0.75 * 0.5 + 0.25)).abs() < 1e-6));
    }

    /// A profile of one stage whose share is the whole sum has that stage
    /// write the scores where they stand, and each value and the band are
    /// still the sum's own terms, bit for bit: a value past either end is
    /// brought in, a negative zero reads as zero, and a weight whose share
    /// of itself does not round to one scores through the sum.
    #[test]
    fn a_profile_of_one_stage_scores_as_the_sum_scores_it() {
        /// The values a stage may write, some outside a score's range.
        const GIVEN: [f32; 5] = [-0.0, 1.5, -3.0, 0.25, 1.0];
        /// A scorer that writes [`GIVEN`] under a band outside the range.
        #[derive(Debug)]
        struct Given;
        impl Lifecycle for Given {}
        impl Scorer for Given {
            fn name(&self) -> &'static str {
                "given"
            }
            fn score(
                &self,
                _: &RequestFacts<'_>,
                _: &[Candidate],
                _: &[u32],
                out: &mut [f32],
            ) -> TieBand {
                out.copy_from_slice(&GIVEN);
                TieBand(2.0)
            }
        }
        let pool: Vec<Candidate> = (0..5).map(|id| candidate(id, 1)).collect();
        let kept = [0u32, 1, 2, 3, 4];
        for weight in [1.0f32, 3.0, 41.0] {
            let profile = Profile {
                name: "one".into(),
                filters: vec![],
                scorers: vec![WeightedScorer {
                    scorer: Box::new(Given),
                    weight,
                }],
                picker: Box::new(Best),
            };
            let (mut scores, mut stage) = (Vec::new(), Vec::new());
            let band = combine(
                &profile,
                &RequestFacts::default(),
                &pool,
                &kept,
                &mut scores,
                &mut stage,
            );
            // The sum's own terms: a zero plus the stage's share of the
            // value brought into range. The share is the one a decision in
            // front of the whole pass writes its scores by.
            let share = profile.scorers[0].sole_share();
            assert_eq!(share.to_bits(), (weight * (1.0 / weight)).to_bits());
            let term = |value: f32| (0.0 + share * value.clamp(0.0, 1.0)).to_bits();
            let written: Vec<u32> = scores.iter().map(|score| score.to_bits()).collect();
            let summed: Vec<u32> = GIVEN.iter().map(|&value| term(value)).collect();
            assert_eq!(written, summed, "weight {weight}");
            assert_eq!(band.0.to_bits(), term(2.0), "weight {weight}");
        }
        // The weights stand on both sides: the share of 41 is not one.
        assert_eq!(3.0f32 * (1.0 / 3.0f32), 1.0);
        assert_ne!(41.0f32 * (1.0 / 41.0f32), 1.0);
    }

    const SESSIONS: SessionRule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 2,
        ttl_micros: None,
        sweep_micros: 0,
        recovery: Recovery::Latest,
        rule_placed: RulePlaced::Told,
    };

    /// The least-requests profile under a sessions rule.
    fn session_engine(rule: SessionRule) -> Engine {
        session_engine_of(least_requests_profile(), rule)
    }

    /// `profile` under a sessions rule.
    fn session_engine_of(profile: Profile, rule: SessionRule) -> Engine {
        Engine::new(
            vec![profile],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                shadow: None,
                decode_shadow: None,
                sessions: Some(rule),
                overload: None,
            }],
            1,
            Needs::default(),
        )
    }

    /// The session `key` names in the keyed cases' requests (model `m`, the
    /// single phase): what a lease carries and the table is keyed by.
    fn session(key: u64) -> Session {
        RequestFacts {
            model: "m",
            session_key: Identity::hashed(key),
            ..RequestFacts::default()
        }
        .session()
    }

    /// Decide for `key` at `now` without committing.
    fn decide_keyed(engine: &Engine, pool: &[Candidate], key: u64, now: u64) -> Decision {
        let facts = RequestFacts {
            model: "m",
            session_key: Identity::hashed(key),
            now_micros: now,
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        engine.decide(
            &facts,
            &PoolFacts::default(),
            pool,
            &Constraints::default(),
            &mut scratch,
        )
    }

    /// Decide for `key` at `now` and dispatch the first target, as a host
    /// commits it.
    fn place_keyed(engine: &Engine, pool: &[Candidate], key: u64, now: u64) -> Target {
        let facts = RequestFacts {
            model: "m",
            session_key: Identity::hashed(key),
            now_micros: now,
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            pool,
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("a pick");
        let target = *pick.targets.first().expect("a target");
        // The lease takes the request's session and the pick's verdict, as a
        // host commits them.
        let lease = Lease {
            session: facts.session(),
            session_verdict: pick.session_verdict,
            issued_micros: now,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &target, Side::Single, 0);
        target
    }

    /// A key is placed by the profile once, then stays on its pin past the
    /// load; an unroutable pin hands the key to a failover; once both are
    /// routable the rule's recovery says which leads.
    #[test]
    fn a_keyed_request_stays_on_its_pin_and_recovers_by_the_rule() {
        for (recovery, back_to_first) in [(Recovery::Latest, false), (Recovery::Original, true)] {
            let engine = session_engine(SessionRule {
                recovery,
                ..SESSIONS
            });
            let mut pool = vec![candidate(0, 5), candidate(1, 0), candidate(2, 3)];
            assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 1);
            pool[1].in_flight = 9;
            for t in 1..4 {
                assert_eq!(
                    place_keyed(&engine, &pool, 7, 10 + t).position,
                    1,
                    "the pin holds past the load ({recovery:?})"
                );
            }
            pool[1].health.healthy = false;
            assert_eq!(
                place_keyed(&engine, &pool, 7, 20).position,
                2,
                "the emptiest routable worker takes the key"
            );
            pool[1].health.healthy = true;
            let back = place_keyed(&engine, &pool, 7, 30);
            assert_eq!(back.position == 1, back_to_first, "{recovery:?}");
            assert_eq!(engine.sessions().len(), 1);
        }
    }

    /// Two pins, the owner leading and at the cap on the key. Where the
    /// stages pick the owner again the request stays on it. Where they pick
    /// the failover, under the cap, the respill goes there and the pin
    /// follows it, so the failover leads once the owner is under the cap
    /// again.
    #[test]
    fn a_saturated_owner_respills_onto_its_failover_and_the_failover_leads_after() {
        let engine = session_engine(SessionRule {
            recovery: Recovery::Original,
            ..SESSIONS
        });
        let mut pool = vec![candidate(0, 9), candidate(1, 0), candidate(2, 1)];
        assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 1);
        pool[1].health.healthy = false;
        assert_eq!(
            place_keyed(&engine, &pool, 7, 20).position,
            2,
            "the failover"
        );
        pool[1].health.healthy = true;
        assert_eq!(
            place_keyed(&engine, &pool, 7, 30).position,
            1,
            "the first owner leads under original recovery"
        );
        pool[1].session_in_flight = 2;
        assert_eq!(
            place_keyed(&engine, &pool, 7, 40).position,
            1,
            "the fewest requests are the owner's: the stages pick it again"
        );
        pool[1].in_flight = 5;
        assert_eq!(place_keyed(&engine, &pool, 7, 45).position, 2);
        pool[1].session_in_flight = 0;
        assert_eq!(
            place_keyed(&engine, &pool, 7, 50).position,
            2,
            "the respill moved the pin to the failover"
        );
    }

    /// At the cap the stages place the request over every candidate. A pick
    /// on another worker at the cap takes the request and leaves the pin;
    /// a pick on another worker under the cap takes the pin too.
    #[test]
    fn at_the_cap_a_request_goes_where_the_stages_say_and_the_pin_moves_only_under_the_cap() {
        let engine = session_engine(SESSIONS);
        let mut pool = vec![candidate(0, 0), candidate(1, 3), candidate(2, 3)];
        assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
        for c in &mut pool {
            c.session_in_flight = 2;
        }
        assert_eq!(
            place_keyed(&engine, &pool, 7, 20).position,
            0,
            "the stages pick the owner again"
        );
        // The fewest requests are now on a worker at the cap.
        pool[0].in_flight = 9;
        pool[1].in_flight = 2;
        let decision = decide_keyed(&engine, &pool, 7, 30);
        let pick = decision.pick().expect("the stages place");
        assert_eq!(pick.targets.first().map(|t| t.position), Some(1));
        assert_eq!(pick.session_verdict, SessionVerdict::None, "no pin moves");
        assert_eq!(place_keyed(&engine, &pool, 7, 40).position, 1);
        for c in &mut pool {
            c.session_in_flight = 0;
        }
        assert_eq!(
            place_keyed(&engine, &pool, 7, 50).position,
            0,
            "the pin stood on the owner"
        );
        // The owner at the cap again, the stages' pick under it.
        pool[0].session_in_flight = 2;
        assert_eq!(place_keyed(&engine, &pool, 7, 60).position, 1);
        pool[0].session_in_flight = 0;
        pool[0].in_flight = 0;
        assert_eq!(
            place_keyed(&engine, &pool, 7, 70).position,
            1,
            "the respill moved the pin"
        );
    }

    /// Who placed a keyed request's pick, at each place the whole decision
    /// answers under a sessions rule. A key with no pin and a pin at its cap
    /// are placed by the profile's stages under `delegate` and by the rule's
    /// own stage under its own assignment. A pin followed is the rule's
    /// under both.
    #[test]
    fn a_keyed_pick_says_who_placed_it() {
        use decision::PlacedBy;
        for (assignment, placed) in [
            (Assignment::Delegate, PlacedBy::Stages),
            (Assignment::MinLoad, PlacedBy::Rule),
        ] {
            let engine = session_engine(SessionRule {
                assignment,
                ..SESSIONS
            });
            let mut pool = vec![candidate(0, 0), candidate(1, 3), candidate(2, 3)];
            let by = |decision: &Decision| decision.pick().expect("a pick").placed_by();
            let new = decide_keyed(&engine, &pool, 7, 10);
            assert_eq!(by(&new), placed, "{assignment:?}: a key with no pin");
            assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
            let pinned = decide_keyed(&engine, &pool, 7, 20);
            assert_eq!(by(&pinned), PlacedBy::Rule, "{assignment:?}: a pin");
            // The pin at its cap and the busiest: the request is placed
            // again, on a worker under the cap.
            pool[0].session_in_flight = 2;
            pool[0].in_flight = 9;
            let again = decide_keyed(&engine, &pool, 7, 30);
            let pick = again.pick().expect("the respill places");
            assert_eq!(pick.session_verdict, SessionVerdict::Respill);
            assert_eq!(by(&again), placed, "{assignment:?}: a pin at its cap");
        }
    }

    /// A respill's commit fronts the worker it lands on, the first target or
    /// an alternate behind it. So no worker at the cap stands among a
    /// respill's targets: the pinned worker and a peer at the cap leave the
    /// list, and an alternate's dispatch moves the pin under the cap. The
    /// alternate then leads, and the worker the decision first took, which
    /// it fronted where it was made, stands behind it.
    #[test]
    fn a_respill_names_no_alternate_at_the_cap() {
        let engine = session_engine(SESSIONS);
        let mut pool = vec![
            candidate(0, 0),
            candidate(1, 1),
            candidate(2, 2),
            candidate(3, 3),
        ];
        assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
        // The pin at the cap and the busiest; the stages' second choice at
        // the cap too.
        pool[0].session_in_flight = 2;
        pool[0].in_flight = 9;
        pool[2].session_in_flight = 2;
        let decision = decide_keyed(&engine, &pool, 7, 20);
        let pick = decision.pick().expect("the respill places");
        assert_eq!(pick.session_verdict, SessionVerdict::Respill);
        let placed: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
        assert_eq!(placed, [1, 3], "the workers under the cap alone");
        // The alternate's dispatch, under the decision's verdict.
        let alternate = pick.targets.as_slice()[1];
        let lease = Lease {
            session: session(7),
            session_verdict: pick.session_verdict,
            issued_micros: 20,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &alternate, Side::Single, 0);
        let first = pick.targets.as_slice()[0];
        assert_eq!(
            engine.sessions().pins(session(7)),
            Some([alternate.id, first.id]),
            "the alternate, then the worker first taken"
        );
        pool[0].session_in_flight = 0;
        assert_eq!(
            place_keyed(&engine, &pool, 7, 30).position,
            3,
            "the pin moved under the cap"
        );
    }

    /// A respill fronts the worker it takes where it is decided, ahead of
    /// its dispatch: the key's next request follows the move under the cap,
    /// as a gateway's sticky map moves a pin where it selects the worker.
    /// The commit on that worker finds the pins in place.
    #[test]
    fn a_respill_fronts_its_worker_where_it_is_decided() {
        let engine = session_engine(SESSIONS);
        let mut pool = vec![candidate(0, 0), candidate(1, 1), candidate(2, 2)];
        let home = place_keyed(&engine, &pool, 7, 10);
        assert_eq!(home.position, 0);
        let pins = || engine.sessions().pins(session(7));
        // The pin at the cap and the busiest, and every other worker at the
        // cap too: the placement carries no verdict and moves no pin.
        pool[0].in_flight = 9;
        for row in &mut pool {
            row.session_in_flight = 2;
        }
        let over = decide_keyed(&engine, &pool, 7, 15);
        let pick = over.pick().expect("a placement at the cap");
        assert_eq!(pick.session_verdict, SessionVerdict::None);
        assert_eq!(pins(), Some([home.id, WorkerId(0)]), "no verdict");
        // The others under the cap: the respill takes the second row.
        (pool[1].session_in_flight, pool[2].session_in_flight) = (0, 0);
        let respill = decide_keyed(&engine, &pool, 7, 20);
        let pick = respill.pick().expect("the respill places");
        assert_eq!(pick.session_verdict, SessionVerdict::Respill);
        let taken = *pick.targets.first().expect("a target");
        assert_eq!(taken.position, 1);
        assert_eq!(pins(), Some([taken.id, home.id]), "ahead of its dispatch");
        // The key's next request, decided ahead of that dispatch.
        let next = decide_keyed(&engine, &pool, 7, 30);
        let pick = next.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
        assert_eq!(pick.targets.first().map(|target| target.id), Some(taken.id));
        let lease = Lease {
            session: session(7),
            session_verdict: SessionVerdict::Respill,
            issued_micros: 20,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &taken, Side::Single, 0);
        assert_eq!(pins(), Some([taken.id, home.id]), "behind its dispatch");
    }

    /// The pin taken again at the cap is the decision's one target. A
    /// dispatch on an alternate would move the pin by the key's recovery,
    /// onto a worker at the cap as onto any other.
    #[test]
    fn the_pin_taken_again_at_the_cap_names_no_alternate() {
        let engine = session_engine(SESSIONS);
        let mut pool = vec![
            candidate(0, 0),
            candidate(1, 1),
            candidate(2, 2),
            candidate(3, 3),
        ];
        assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
        // Every worker at the cap, the pin with the fewest requests: the
        // stages pick it again.
        for c in &mut pool {
            c.session_in_flight = 2;
        }
        let decision = decide_keyed(&engine, &pool, 7, 20);
        let pick = decision.pick().expect("the stages place");
        assert_eq!(pick.session_verdict, SessionVerdict::Pinned);
        // The dispatch of the decision's last target, under its verdict.
        let last = *pick.targets.as_slice().last().expect("a target");
        let lease = Lease {
            session: session(7),
            session_verdict: pick.session_verdict,
            issued_micros: 20,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &last, Side::Single, 0);
        for c in &mut pool {
            c.session_in_flight = 0;
        }
        assert_eq!(
            place_keyed(&engine, &pool, 7, 30).position,
            0,
            "the pin stayed"
        );
        let placed: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
        assert_eq!(placed, [0], "the pin alone");
    }

    /// A key with no routable pin is placed by the rule's assignment: the
    /// profile's own stages under `delegate`, the fewest requests under
    /// `min_load`, the fewest sessions under `min_group`, any routable worker
    /// under `random`.
    #[test]
    fn a_key_without_a_routable_pin_is_placed_by_the_rules_assignment() {
        let mut pool = vec![candidate(0, 1), candidate(1, 4), candidate(2, 2)];
        pool[0].active_sessions = 7;
        pool[1].active_sessions = 1;
        pool[2].active_sessions = 3;
        let by = |assignment| {
            let engine = session_engine(SessionRule {
                assignment,
                ..SESSIONS
            });
            place_keyed(&engine, &pool, 7, 10).position
        };
        assert_eq!(by(Assignment::Delegate), 0, "the profile's fewest requests");
        assert_eq!(by(Assignment::MinLoad), 0);
        assert_eq!(by(Assignment::MinGroup), 1, "the fewest sessions");
        assert!(by(Assignment::Random) < 3);
    }

    /// A join under an id that left ends its absence, whoever tells it:
    /// the join's own word, a replacement's new id, a snapshot that lists
    /// the worker.
    #[test]
    fn a_join_under_an_id_that_left_ends_its_absence() {
        let engine = session_engine(SESSIONS);
        let pool = vec![candidate(0, 1), candidate(1, 4)];
        // A key in the table: a table that holds none files no absence.
        place_keyed(&engine, &pool, 7, 10);
        let (a, b) = (WorkerId(100), WorkerId(101));
        engine.on_membership(&Membership::Removed(a));
        assert_eq!(engine.sessions.absent(), 1);
        engine.on_membership(&Membership::Added(a));
        assert_eq!(engine.sessions.absent(), 0, "a join's own word");
        engine.on_membership(&Membership::Removed(a));
        engine.on_membership(&Membership::Removed(b));
        engine.on_membership(&Membership::Snapshot(&[a]));
        assert_eq!(engine.sessions.absent(), 1, "the snapshot lists one");
        let replaced = Membership::Replaced {
            old: WorkerId(300),
            new: b,
        };
        engine.on_membership(&replaced);
        assert_eq!(engine.sessions.absent(), 1, "the old life alone");
    }

    #[test]
    fn a_removed_workers_keys_fall_to_their_failover_and_go_back_with_the_worker() {
        let engine = session_engine(SESSIONS);
        let mut pool = vec![
            candidate(0, 1),
            candidate(1, 4),
            candidate(2, 2),
            candidate(3, 3),
        ];
        // Key 7 lives on worker 100 alone; key 8 moves to 102 while 100 is
        // down and keeps 100 as its failover.
        assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
        assert_eq!(place_keyed(&engine, &pool, 8, 10).position, 0);
        pool[0].health.healthy = false;
        assert_eq!(place_keyed(&engine, &pool, 8, 20).position, 2);
        pool[0].health.healthy = true;
        pool[0].in_flight = 9;
        assert_eq!(
            place_keyed(&engine, &pool, 7, 30).position,
            0,
            "a pin holds under load"
        );
        assert_eq!(place_keyed(&engine, &pool, 8, 30).position, 2);
        // A host that drops a row numbers the rest again: a row's position
        // is its index in the pool a decision is given.
        let renumber = |pool: &mut [Candidate]| {
            for (row, index) in pool.iter_mut().zip(0u32..) {
                row.position = index;
            }
        };
        // The host removes 102: key 8's pins stand through the leave, its
        // owner reads as not routable and its failover serves, loaded or
        // not.
        engine.on_membership(&Membership::Removed(WorkerId(102)));
        pool.remove(2);
        renumber(&mut pool);
        assert_eq!(
            engine.sessions.pins(session(8)),
            Some([WorkerId(102), WorkerId(100)]),
            "the pins stand through the leave"
        );
        assert_eq!(place_keyed(&engine, &pool, 8, 40).id, WorkerId(100));
        // The host registers the key of 102 again, as the worker it was: a
        // new id, and its pins with it.
        engine.on_membership(&Membership::Added(WorkerId(202)));
        engine.pins_follow(WorkerId(102), WorkerId(202));
        pool.push(Candidate {
            id: WorkerId(202),
            ..candidate(3, 5)
        });
        assert_eq!(
            engine.sessions.pins(session(8)),
            Some([WorkerId(202), WorkerId(100)]),
            "the pins follow the key"
        );
        assert_eq!(place_keyed(&engine, &pool, 8, 50).id, WorkerId(202));
        // The host replaces 100 and says the new life is another worker:
        // key 7 has no pin left and is placed afresh by the rule's
        // assignment (the fewest requests), and key 8 keeps its owner.
        engine.on_membership(&Membership::Replaced {
            old: WorkerId(100),
            new: WorkerId(200),
        });
        assert_eq!(
            engine.sessions.pins(session(7)),
            Some([WorkerId(100), WorkerId(0)]),
            "a replacement's pins stand until the host says which way"
        );
        engine.pins_forget(&[WorkerId(100)]);
        pool.remove(0);
        renumber(&mut pool);
        assert_eq!(engine.sessions.pins(session(7)), None);
        assert_eq!(
            engine.sessions.pins(session(8)),
            Some([WorkerId(202), WorkerId(0)])
        );
        assert_eq!(place_keyed(&engine, &pool, 7, 60).id, WorkerId(103));
        assert_eq!(place_keyed(&engine, &pool, 8, 60).id, WorkerId(202));
    }

    /// A decision reads its rows ahead of its claim. Where its worker is
    /// registered again between the two, the key is claimed under the id
    /// that stands: the key's next request, decided on the rows that stand
    /// ahead of the first one's dispatch, follows the claim. The first
    /// request's verdict and its dispatch name the id its decision read,
    /// and the dispatch confirms the claim.
    #[test]
    fn a_claim_behind_its_workers_rename_leads_under_the_id_that_stands() {
        let engine = session_engine(SESSIONS);
        // The rows the first decision read, and the rows that stand: the
        // host replaced the first worker's row and says it is the worker
        // it was.
        let read = [candidate(0, 0), candidate(1, 1)];
        let mut rows = read;
        rows[0].id = WorkerId(900);
        engine.on_membership(&Membership::Replaced {
            old: read[0].id,
            new: rows[0].id,
        });
        engine.pins_follow(read[0].id, rows[0].id);
        let decision = decide_keyed(&engine, &read, 7, 10);
        let pick = decision.pick().expect("a placement");
        let first = *pick.targets.first().expect("a target");
        assert_eq!(first.id, read[0].id);
        assert_eq!(pick.session_verdict, SessionVerdict::Claimed(read[0].id));
        let pins = || engine.sessions.pins(session(7));
        assert_eq!(
            pins(),
            Some([rows[0].id, WorkerId(0)]),
            "the id that stands"
        );
        // The worker is the busiest now: the key's pin alone sends there.
        rows[0].in_flight = 9;
        let next = decide_keyed(&engine, &rows, 7, 20);
        let pick = next.pick().expect("a pick");
        assert_eq!(pick.targets.first().map(|t| t.id), Some(rows[0].id));
        let lease = Lease {
            session: session(7),
            session_verdict: SessionVerdict::Claimed(read[0].id),
            issued_micros: 10,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &first, Side::Single, 0);
        assert_eq!(pins(), Some([rows[0].id, WorkerId(0)]), "confirmed");
    }

    /// A key pinned on a worker and decided on rows read before the worker
    /// was registered again: the pin names the id that stands, no row
    /// read holds it, and the rotation's turn lands on the worker under
    /// the id it had. The table answers that the worker is a pin of the
    /// key already, and the decision takes its pick as the pin it is:
    /// nothing is claimed, and the rule's cap holds for it. At the cap the
    /// key is handed on to the rotation's next turn, as from a pin read
    /// first, and that worker leads the pins ahead of the dispatch.
    #[test]
    fn a_pick_answered_as_a_pin_already_is_put_to_the_rules_cap() {
        let engine = session_engine_of(round_robin_profile(), SESSIONS);
        let read = [
            candidate(0, 0),
            candidate(1, 0),
            candidate(2, 0),
            candidate(3, 0),
        ];
        let home = place_keyed(&engine, &read, 7, 10);
        let stands = WorkerId(900);
        engine.on_membership(&Membership::Replaced {
            old: home.id,
            new: stands,
        });
        engine.pins_follow(home.id, stands);
        let pins = || engine.sessions.pins(session(7));
        assert_eq!(pins(), Some([stands, WorkerId(0)]));
        // A request with no key takes one turn of the rotation.
        let turn = |now: u64| {
            let facts = RequestFacts {
                model: "m",
                now_micros: now,
                ..RequestFacts::default()
            };
            let (shared, cx) = (PoolFacts::default(), Constraints::default());
            let decision = engine.decide(&facts, &shared, &read, &cx, &mut Scratch::new());
            let pick = decision.pick().expect("a pick");
            pick.targets.first().expect("a target").position
        };
        // Three turns bring the rotation back to the key's worker. Under
        // the cap its turn is the pin: no claim, and nothing is written.
        let behind: Vec<u32> = (0..3).map(|step| turn(20 + step)).collect();
        assert!(!behind.contains(&home.position), "{behind:?}");
        let under = decide_keyed(&engine, &read, 7, 30);
        let pick = under.pick().expect("a pick");
        assert_eq!(pick.targets.first().map(|t| t.id), Some(home.id));
        assert_eq!(pick.session_verdict, SessionVerdict::Pinned, "no claim");
        assert_eq!(pins(), Some([stands, WorkerId(0)]), "nothing is written");
        // At the cap, a rotation later: the key is handed on.
        let behind: Vec<u32> = (0..3).map(|step| turn(40 + step)).collect();
        assert!(!behind.contains(&home.position), "{behind:?}");
        let mut capped = read;
        capped[home.position as usize].session_in_flight = SESSIONS.inflight_cap;
        let handed = decide_keyed(&engine, &capped, 7, 50);
        let pick = handed.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::Respill);
        let taken = *pick.targets.first().expect("a target");
        assert_ne!(taken.id, home.id);
        assert_eq!(pins(), Some([taken.id, stands]), "ahead of its dispatch");
    }

    #[test]
    fn two_first_decisions_of_a_fresh_key_agree_on_the_first_claim() {
        let engine = session_engine(SESSIONS);
        let key = session(7);
        let mut pool = vec![candidate(0, 1), candidate(1, 4), candidate(2, 2)];
        let first = decide_keyed(&engine, &pool, 7, 10);
        let first_pick = first.pick().expect("a pick");
        assert_eq!(first_pick.targets.first().map(|t| t.position), Some(0));
        let claimed = first_pick.targets.first().expect("a target").id;
        assert_eq!(first_pick.session_verdict, SessionVerdict::Claimed(claimed));
        // Before the first commits, a second decision under a changed load
        // would place the key elsewhere; it reads the claim as the pin and
        // takes it as a hit. (Between a decision's own reading and its claim
        // another claim is followed the same way, when among its survivors.)
        pool[0].in_flight = 9;
        let second = decide_keyed(&engine, &pool, 7, 11);
        let targets = second.pick().expect("a pick").targets;
        assert_eq!(
            targets.first().map(|t| t.position),
            Some(0),
            "the claim leads"
        );
        assert_eq!(targets.len(), 1, "a hit on the claim");
        assert_eq!(
            engine.sessions.pins(key),
            Some([WorkerId(100), WorkerId(0)])
        );
        // The commits, in either order, keep the one pin.
        let lease = |decision: &Decision, now| Lease {
            session: key,
            session_verdict: decision.pick().expect("a pick").session_verdict,
            issued_micros: now,
            ..Lease::default()
        };
        let target = |decision: &Decision| {
            *decision
                .pick()
                .expect("a pick")
                .targets
                .first()
                .expect("a target")
        };
        engine.on_dispatched(&lease(&second, 11), &target(&second), Side::Single, 0);
        engine.on_dispatched(&lease(&first, 10), &target(&first), Side::Single, 0);
        assert_eq!(
            engine.sessions.pins(key),
            Some([WorkerId(100), WorkerId(0)])
        );
    }

    /// The owner is down; the decision claims the best survivor, the commit
    /// refuses it and dispatches the alternate: the owner keeps its pin, the
    /// alternate is the failover, and the owner leads again once it is back.
    #[test]
    fn an_alternate_commit_withdraws_the_claim_and_keeps_the_owner() {
        let engine = session_engine(SessionRule {
            recovery: Recovery::Original,
            ..SESSIONS
        });
        let key = session(7);
        let mut pool = vec![candidate(0, 1), candidate(1, 4), candidate(2, 2)];
        let owner = place_keyed(&engine, &pool, 7, 10);
        assert_eq!(owner.position, 0);
        pool[0].health.healthy = false;
        let decision = decide_keyed(&engine, &pool, 7, 20);
        let pick = decision.pick().expect("a pick");
        let claimed = pick.targets.first().expect("a target").id;
        let alternate = pick.targets.as_slice()[1];
        assert_ne!(alternate.id, claimed);
        assert_eq!(engine.sessions.pins(key), Some([owner.id, claimed]));
        let lease = Lease {
            session: key,
            session_verdict: pick.session_verdict,
            issued_micros: 20,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &alternate, Side::Single, 0);
        assert_eq!(
            engine.sessions.pins(key),
            Some([owner.id, alternate.id]),
            "the claim is withdrawn, the owner keeps the lead"
        );
        pool[0].health.healthy = true;
        assert_eq!(place_keyed(&engine, &pool, 7, 30).position, 0);
    }

    /// A pair's dispatch moves its key's pin to the worker of the side it
    /// names: the pair's decode worker on the decode side, the target
    /// itself on the prefill side.
    #[test]
    fn a_pairs_dispatch_pins_the_worker_of_its_side() {
        let engine = session_engine(SESSIONS);
        let target = Target {
            id: WorkerId(3),
            pair: Some(Pair {
                decode_id: WorkerId(5),
                ..Pair::default()
            }),
            ..Target::default()
        };
        let lease = |key| Lease {
            session: session(key),
            session_verdict: SessionVerdict::Pinned,
            ..Lease::default()
        };
        engine.on_dispatched(&lease(7), &target, Side::Decode, 0);
        engine.on_dispatched(&lease(8), &target, Side::Prefill, 0);
        let pin = |key| engine.sessions.pins(session(key)).map(|pins| pins[0]);
        assert_eq!((pin(7), pin(8)), (Some(WorkerId(5)), Some(WorkerId(3))));
    }

    /// Two bindings' rules share the engine's table, which sweeps at the
    /// shorter of their intervals: once the short rule's TTL and interval
    /// passed, its idle key is gone from the table while the hourly key
    /// stays, whichever binding's decision ran the sweep.
    #[test]
    fn the_table_sweeps_at_the_shortest_interval_of_the_bindings_rules() {
        let hourly = SessionRule {
            ttl_micros: Some(3_600_000_000),
            sweep_micros: 3_600_000_000,
            ..SESSIONS
        };
        let short = SessionRule {
            ttl_micros: Some(1_000),
            sweep_micros: 1_000,
            ..SESSIONS
        };
        let binding = |model: &str, rule: SessionRule| Binding {
            model: model.into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: Some(rule),
            overload: None,
        };
        let engine = Engine::new(
            vec![least_requests_profile()],
            vec![binding("a", hourly), binding("b", short)],
            1,
            Needs::default(),
        );
        let pool = vec![candidate(0, 1), candidate(1, 4)];
        let place = |model: &'static str, key: u64, now: u64| {
            let facts = RequestFacts {
                model,
                session_key: Identity::hashed(key),
                now_micros: now,
                ..RequestFacts::default()
            };
            let mut scratch = Scratch::new();
            let decision = engine.decide(
                &facts,
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut scratch,
            );
            let pick = decision.pick().expect("a pick");
            let target = *pick.targets.first().expect("a target");
            let lease = Lease {
                session: facts.session(),
                session_verdict: pick.session_verdict,
                issued_micros: now,
                ..Lease::default()
            };
            engine.on_dispatched(&lease, &target, Side::Single, 0);
        };
        place("a", 1, 0);
        place("b", 2, 0);
        place("b", 3, 2_000);
        assert_eq!(
            engine.sessions().len(),
            2,
            "the hourly key and the fresh one stay, the short key was swept"
        );
        let swept = RequestFacts {
            model: "b",
            session_key: Identity::hashed(2),
            ..RequestFacts::default()
        };
        assert!(engine.sessions.pins(swept.session()).is_none());
    }

    #[test]
    fn a_key_whose_pins_are_all_down_is_claimed_at_its_decision_too() {
        let engine = session_engine(SESSIONS);
        let key = session(7);
        let mut pool = vec![candidate(0, 1), candidate(1, 4), candidate(2, 2)];
        assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
        pool[0].health.healthy = false;
        // The owner is down: the first decision claims the fewest-requests
        // worker before any commit; a second decision under a changed load
        // reads that claim as the pin and takes it.
        let first = decide_keyed(&engine, &pool, 7, 20);
        let lead = |d: &Decision| {
            d.pick()
                .expect("a pick")
                .targets
                .first()
                .map(|t| t.position)
        };
        assert_eq!(lead(&first), Some(2));
        assert_eq!(
            engine.sessions.pins(key),
            Some([WorkerId(102), WorkerId(100)])
        );
        pool[2].in_flight = 9;
        let second = decide_keyed(&engine, &pool, 7, 21);
        assert_eq!(lead(&second), Some(2), "the claim is the pin");
    }

    #[test]
    fn a_keyed_request_under_a_binding_without_a_sessions_rule_pins_nothing() {
        let engine = Engine::new(
            vec![least_requests_profile()],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        let pool = vec![candidate(0, 1), candidate(1, 4)];
        let decision = decide_keyed(&engine, &pool, 7, 10);
        let pick = decision.pick().expect("a pick");
        assert_eq!(pick.session_verdict, SessionVerdict::None);
        let lease = Lease {
            session: session(7),
            session_verdict: pick.session_verdict,
            issued_micros: 10,
            ..Lease::default()
        };
        engine.on_dispatched(
            &lease,
            pick.targets.first().expect("a target"),
            Side::Single,
            0,
        );
        assert!(engine.sessions.is_empty(), "no rule, no row");
    }

    /// One engine, two hosts' requests for one key: the first holds the
    /// plain key, the second its fingerprint alone. Built apart, they name
    /// one session, so the second meets the first's pin past the load; the
    /// key under another model, or in another phase, is another session and
    /// is placed afresh.
    #[test]
    fn two_hosts_requests_for_one_key_model_and_phase_meet_on_one_pin() {
        let engine = session_engine(SESSIONS);
        let mut pool = vec![candidate(0, 1), candidate(1, 4), candidate(2, 2)];
        let commit = |facts: &RequestFacts<'_>, pool: &[Candidate]| {
            let mut scratch = Scratch::new();
            let decision = engine.decide(
                facts,
                &PoolFacts::default(),
                pool,
                &Constraints::default(),
                &mut scratch,
            );
            let pick = decision.pick().expect("a pick");
            let target = *pick.targets.first().expect("a target");
            let lease = Lease {
                session: facts.session(),
                session_verdict: pick.session_verdict,
                issued_micros: facts.now_micros,
                ..Lease::default()
            };
            engine.on_dispatched(&lease, &target, Side::Single, 0);
            (target.position, pick.session_verdict)
        };
        let plain = RequestFacts {
            model: "m",
            session_key: Identity::plain("conversation-7"),
            now_micros: 10,
            ..RequestFacts::default()
        };
        let (first, verdict) = commit(&plain, &pool);
        assert_eq!(first, 0, "the fewest requests");
        assert_eq!(verdict, SessionVerdict::Claimed(WorkerId(100)));
        // The other host holds the key's fingerprint alone, and the pinned
        // worker is the busiest by now.
        pool[0].in_flight = 9;
        let hashed = RequestFacts {
            session_key: Identity::hashed(view::fingerprint(b"conversation-7")),
            now_micros: 20,
            ..plain
        };
        assert_eq!(
            commit(&hashed, &pool),
            (0, SessionVerdict::Pinned),
            "one pin for both forms of the key"
        );
        assert_eq!(engine.sessions().len(), 1);
        let other_model = RequestFacts {
            model: "n",
            now_micros: 30,
            ..plain
        };
        assert_eq!(
            commit(&other_model, &pool).0,
            2,
            "placed afresh, by the load"
        );
        let other_phase = RequestFacts {
            phase: Phase::Decode,
            now_micros: 40,
            ..plain
        };
        assert_eq!(
            commit(&other_phase, &pool).0,
            2,
            "placed afresh, by the load"
        );
        assert_eq!(engine.sessions().len(), 3, "three sessions of one key");
    }

    /// The `manual` picker's profile: the filters, no scorer.
    fn manual_profile() -> Profile {
        Profile {
            name: "manual".into(),
            filters: vec![Box::new(Healthy), Box::new(NotTried)],
            scorers: vec![],
            picker: Box::new(Manual),
        }
    }

    fn manual_engine(sessions: Option<SessionRule>) -> Engine {
        Engine::new(
            vec![manual_profile()],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                shadow: None,
                decode_shadow: None,
                sessions,
                overload: None,
            }],
            1,
            Needs::default(),
        )
    }

    /// One decision for `key` (`0`: no key) drawn by `seed`: the worker and
    /// what the decision did with the key.
    fn decide_seeded(
        engine: &Engine,
        pool: &[Candidate],
        key: u64,
        seed: u64,
    ) -> (u32, SessionVerdict) {
        let facts = RequestFacts {
            model: "m",
            session_key: Identity::hashed(key),
            seed,
            now_micros: 10,
            ..RequestFacts::default()
        };
        let mut scratch = Scratch::new();
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            pool,
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("a pick");
        (
            pick.targets.first().expect("a target").position,
            pick.session_verdict,
        )
    }

    /// Under the `manual` picker the rule's assignment places a keyless
    /// request and a key with no pin alike, through one stage: the two
    /// agree pick for pick on every seed. Without a key nothing is pinned.
    #[test]
    fn the_manual_picker_places_a_keyed_miss_and_a_keyless_request_by_one_assignment() {
        let mut pool = vec![
            candidate(0, 1),
            candidate(1, 4),
            candidate(2, 1),
            candidate(3, 2),
        ];
        pool[0].active_sessions = 7;
        pool[1].active_sessions = 1;
        pool[2].active_sessions = 3;
        pool[3].active_sessions = 1;
        // The workers an assignment may take: any, the fewest requests in
        // flight, the fewest session keys.
        for (assignment, may) in [
            (Assignment::Random, &[0u32, 1, 2, 3][..]),
            (Assignment::MinLoad, &[0, 2][..]),
            (Assignment::MinGroup, &[1, 3][..]),
            // No stages of its own to delegate to: the fewest in flight.
            (Assignment::Delegate, &[0, 2][..]),
        ] {
            let rule = SessionRule {
                assignment,
                ..SESSIONS
            };
            let mut taken = [false; 4];
            for seed in 1..=64u64 {
                let (keyless, verdict) = decide_seeded(&manual_engine(Some(rule)), &pool, 0, seed);
                assert_eq!(verdict, SessionVerdict::None, "{assignment:?}");
                let (keyed, verdict) = decide_seeded(&manual_engine(Some(rule)), &pool, 7, seed);
                assert!(
                    matches!(verdict, SessionVerdict::Claimed(_)),
                    "{assignment:?}: {verdict:?}"
                );
                assert_eq!(keyless, keyed, "{assignment:?}, seed {seed}");
                assert!(may.contains(&keyless), "{assignment:?} took {keyless}");
                taken[keyless as usize] = true;
            }
            for &position in may {
                assert!(
                    taken[position as usize],
                    "{assignment:?} never took {position}"
                );
            }
        }
    }

    #[test]
    fn a_keyless_request_under_the_manual_picker_pins_nothing() {
        let engine = manual_engine(Some(SESSIONS));
        let pool = vec![candidate(0, 3), candidate(1, 1)];
        let target = place_keyed(&engine, &pool, 0, 10);
        assert_eq!(target.position, 1, "the fewest in flight under delegate");
        assert!(engine.sessions.is_empty(), "no key, no row");
    }

    /// A binding with no sessions rule names no assignment: the picker
    /// takes the fewest in flight, with a key or without, and pins nothing.
    #[test]
    fn without_a_rule_the_manual_picker_takes_the_fewest_in_flight() {
        let engine = manual_engine(None);
        let pool = vec![candidate(0, 3), candidate(1, 1), candidate(2, 2)];
        for key in [0, 7] {
            for seed in 1..=8 {
                assert_eq!(
                    decide_seeded(&engine, &pool, key, seed),
                    (1, SessionVerdict::None)
                );
            }
        }
        assert!(engine.sessions.is_empty());
    }

    /// A reader reads back the document: the binding the engine answers
    /// holds its rule as it was handed in, under every assignment and under
    /// none. Under `delegate` the `manual` picker's requests still go to
    /// the fewest in flight: the assignment that places is the engine's own
    /// and no reader of the binding sees it.
    #[test]
    fn the_engine_answers_a_binding_as_it_was_declared() {
        for assignment in [
            Assignment::Random,
            Assignment::MinLoad,
            Assignment::MinGroup,
            Assignment::Delegate,
        ] {
            let rule = SessionRule {
                assignment,
                ..SESSIONS
            };
            let engine = manual_engine(Some(rule));
            let read_back = engine
                .binding("m", Phase::Single)
                .and_then(|binding| binding.sessions);
            assert_eq!(read_back, Some(rule), "{assignment:?}");
        }
        let unruled = manual_engine(None);
        let read_back = unruled
            .binding("m", Phase::Single)
            .map(|binding| binding.sessions);
        assert_eq!(read_back, Some(None), "no rule declared, none read back");
        // One engine, both halves: the rule reads back `delegate` and the
        // request is placed by the fewest in flight.
        let engine = manual_engine(Some(SESSIONS));
        let declared = engine
            .binding("m", Phase::Single)
            .and_then(|binding| binding.sessions)
            .map(|rule| rule.assignment);
        assert_eq!(declared, Some(Assignment::Delegate));
        let pool = vec![candidate(0, 3), candidate(1, 1)];
        assert_eq!(place_keyed(&engine, &pool, 0, 10).position, 1);
    }

    /// The keyed path under the `manual` picker is the engine's own: the
    /// key stays on its worker past the load, moves when the worker is
    /// unroutable, and the first owner leads again once it is back.
    #[test]
    fn a_key_under_the_manual_picker_stays_on_its_worker_and_the_first_owner_leads_again() {
        let engine = manual_engine(Some(SessionRule {
            assignment: Assignment::MinLoad,
            recovery: Recovery::Original,
            ..SESSIONS
        }));
        let mut pool = vec![candidate(0, 5), candidate(1, 0), candidate(2, 3)];
        assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 1);
        pool[1].in_flight = 9;
        assert_eq!(place_keyed(&engine, &pool, 7, 20).position, 1);
        pool[1].health.healthy = false;
        assert_eq!(
            place_keyed(&engine, &pool, 7, 30).position,
            2,
            "the fewest in flight of the routable workers"
        );
        pool[1].health.healthy = true;
        assert_eq!(place_keyed(&engine, &pool, 7, 40).position, 1);
        assert_eq!(
            engine.sessions.pins(session(7)),
            Some([WorkerId(101), WorkerId(102)])
        );
    }

    /// A pin over the rule's in-flight cap respills by the binding's placing
    /// assignment too: under the `manual` picker a `delegate` rule hands the
    /// key to the fewest in flight of the candidates, and the decision says
    /// a respill where that is another worker under the cap.
    #[test]
    fn a_respill_under_the_manual_picker_takes_the_fewest_in_flight() {
        let engine = manual_engine(Some(SESSIONS));
        let mut pool = vec![candidate(0, 5), candidate(1, 0), candidate(2, 3)];
        assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 1);
        pool[1].session_in_flight = SESSIONS.inflight_cap;
        pool[1].in_flight = 9;
        let decision = decide_keyed(&engine, &pool, 7, 20);
        let pick = decision.pick().expect("the respill places");
        assert_eq!(pick.session_verdict, SessionVerdict::Respill);
        assert_eq!(pick.targets.first().map(|t| t.position), Some(2));
    }

    /// `profiles` under the pair plan and a sessions rule with no cap that
    /// places a new session by `assignment`: the first profile on the
    /// prefill side and the last on the decode side.
    fn pair_session_engine(profiles: Vec<Profile>, assignment: Assignment) -> Engine {
        let rule = SessionRule {
            assignment,
            inflight_cap: 0,
            ..SESSIONS
        };
        let binding = Binding {
            decode_profile: Some(profiles.len() as u16 - 1),
            sessions: Some(rule),
            ..pair_binding()
        };
        Engine::new(profiles, vec![binding], 1, Needs::default())
    }

    /// A pair's request, under the session key `key` where one is given.
    fn pair_request(key: Option<u64>) -> RequestFacts<'static> {
        RequestFacts {
            session_key: key.map_or(Identity::NONE, Identity::hashed),
            ..request(7)
        }
    }

    /// Each side's own session of a pair's request: the prefill side's,
    /// then the decode side's.
    fn side_sessions(facts: &RequestFacts<'_>) -> [Session; 2] {
        [Phase::Prefill, Phase::Decode].map(|phase| RequestFacts { phase, ..*facts }.session())
    }

    /// Prefills and decodes that pair all with all, each side under ids of
    /// its own and the loads given: the prefills, then the decodes.
    fn two_sides(prefill_loads: &[u32], decode_loads: &[u32]) -> [Vec<Candidate>; 2] {
        let side = |loads: &[u32], first: u64| -> Vec<Candidate> {
            let mut rows = classed_rows(&vec![1; loads.len()]);
            for (row, &load) in rows.iter_mut().zip(loads) {
                (row.id, row.in_flight) = (WorkerId(first + u64::from(row.position)), load);
            }
            rows
        };
        [side(prefill_loads, 100), side(decode_loads, 200)]
    }

    /// One pair decided for `facts` and dispatched a side at a time, as a
    /// host commits it: each side's lease states that side's session and
    /// its verdict, the decode side's as the first target's pair keeps it.
    /// The two places, then the two verdicts.
    fn pair_placed(
        engine: &Engine,
        facts: &RequestFacts<'_>,
        [prefills, decodes]: &[Vec<Candidate>; 2],
        scratch: &mut Scratch,
    ) -> ((u32, u32), (SessionVerdict, DecodeVerdict)) {
        let pools = PairPools {
            prefills,
            decodes,
            ..PairPools::default()
        };
        pair_placed_in(engine, facts, pools, scratch)
    }

    /// [`pair_placed`] over the pools as a host hands them, with its pair
    /// table and its first partners.
    fn pair_placed_in(
        engine: &Engine,
        facts: &RequestFacts<'_>,
        pools: PairPools<'_>,
        scratch: &mut Scratch,
    ) -> ((u32, u32), (SessionVerdict, DecodeVerdict)) {
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let decision = engine.decide_pair(facts, &pool, pools, &cx, &cx, scratch);
        let pick = decision.pick().expect("a pair");
        let target = *pick.targets.first().expect("a target");
        let pair = target.pair.expect("a decode");
        let [prefill_session, decode_session] = side_sessions(facts);
        let prefill = Lease {
            session: prefill_session,
            session_verdict: pick.session_verdict,
            ..Lease::default()
        };
        let went = Target {
            pair: None,
            ..target
        };
        engine.on_dispatched(&prefill, &went, Side::Prefill, 0);
        let decode = Lease {
            session: decode_session,
            session_verdict: pair.decode_verdict.on(pair.decode_id),
            ..Lease::default()
        };
        let went = Target {
            position: pair.decode_position,
            id: pair.decode_id,
            dp_rank: pair.decode_dp_rank,
            pair: None,
        };
        engine.on_dispatched(&decode, &went, Side::Decode, 0);
        (
            (target.position, pair.decode_position),
            (pick.session_verdict, pair.decode_verdict),
        )
    }

    /// Under a sessions rule each side of a pair follows its key's pin by
    /// the side's own session. A key's first pair claims a worker on each
    /// side: the pick states the prefill side's claim and the first
    /// target's pair the decode side's. Each side's dispatch settles its
    /// own claim, so the key's next pair names the same two workers as
    /// pins, wherever the loads point by then. A request with no key is
    /// placed as a pair under no rule is, states no verdict and pins
    /// nothing.
    ///
    /// The classic manual policy keeps one key's entry apart under each
    /// role of a pair (`model_gateway/src/policies/manual.rs:345-352`).
    #[test]
    fn each_side_of_a_pair_follows_its_keys_pin_by_its_own_session() {
        let engine = pair_session_engine(vec![least_requests_profile()], Assignment::Delegate);
        let mut scratch = Scratch::new();
        // The least loaded prefill is the second, the least loaded decode
        // the third.
        let first = two_sides(&[5, 0, 5], &[5, 5, 0]);
        let keyed = pair_request(Some(7));
        let (places, verdicts) = pair_placed(&engine, &keyed, &first, &mut scratch);
        assert_eq!(places, (1, 2));
        let claimed = SessionVerdict::Claimed(first[0][1].id);
        assert_eq!(verdicts, (claimed, DecodeVerdict::CLAIMED));
        let [prefill_session, decode_session] = side_sessions(&keyed);
        assert_ne!(prefill_session, decode_session);
        let pinned = |session| engine.sessions.pins(session).map(|pins| pins[0]);
        assert_eq!(pinned(prefill_session), Some(first[0][1].id));
        assert_eq!(pinned(decode_session), Some(first[1][2].id));
        // The loads point elsewhere on both sides: the two pins hold.
        let moved = two_sides(&[0, 9, 9], &[0, 9, 9]);
        let (places, verdicts) = pair_placed(&engine, &keyed, &moved, &mut scratch);
        assert_eq!(places, (1, 2));
        assert_eq!(verdicts, (SessionVerdict::Pinned, DecodeVerdict::PINNED));
        // No key: the pair a binding under no rule gives, and no verdict.
        let plain = pair_request(None);
        let (places, verdicts) = pair_placed(&engine, &plain, &moved, &mut scratch);
        assert_eq!(places, (0, 0));
        assert_eq!(verdicts, (SessionVerdict::None, DecodeVerdict::NONE));
        let unruled = pair_engine(least_requests_profile());
        let (same, _) = pair_placed(&unruled, &plain, &moved, &mut Scratch::new());
        assert_eq!(same, places);
        // A key under no rule is placed as a request with none is.
        let (same, none) = pair_placed(&unruled, &keyed, &moved, &mut Scratch::new());
        assert_eq!(same, places);
        assert_eq!(none, (SessionVerdict::None, DecodeVerdict::NONE));
        // The rule's assignment places a new session alone. A request with
        // no key is the profile's under any: here the idle worker of each
        // side holds the most keys, and takes it.
        let by_keys = pair_session_engine(vec![least_requests_profile()], Assignment::MinGroup);
        let mut keyed_rows = two_sides(&[0, 9, 9], &[0, 9, 9]);
        for side in &mut keyed_rows {
            side[0].active_sessions = 5;
        }
        let (same, none) = pair_placed(&by_keys, &plain, &keyed_rows, &mut Scratch::new());
        assert_eq!(same, places);
        assert_eq!(none, (SessionVerdict::None, DecodeVerdict::NONE));
        // A new key goes where the rule's assignment says: to the worker
        // of each side that holds the fewest keys, the last.
        for side in &mut keyed_rows {
            side[1].active_sessions = 1;
        }
        let (by_rule, claims) = pair_placed(&by_keys, &keyed, &keyed_rows, &mut Scratch::new());
        assert_eq!(by_rule, (2, 2));
        let claimed = SessionVerdict::Claimed(keyed_rows[0][2].id);
        assert_eq!(claims, (claimed, DecodeVerdict::CLAIMED));
    }

    /// A side whose picker leaves the placement to the rule is placed by
    /// the rule's assignment, a request with a key and one with none
    /// alike: under `min_load` each side takes its fewest in flight. The
    /// key is pinned on each side and stays past the loads. The request
    /// with no key follows the loads and pins nothing. Under no rule such
    /// a side takes the fewest in flight too.
    ///
    /// The classic manual policy places a request with no key by its
    /// assignment under any role
    /// (`model_gateway/src/policies/manual.rs:357-360`), and a new key the
    /// same way (`:317`).
    #[test]
    fn a_manual_side_of_a_pair_is_placed_by_the_rules_assignment() {
        let engine = pair_session_engine(vec![manual_profile()], Assignment::MinLoad);
        let mut scratch = Scratch::new();
        let first = two_sides(&[3, 1, 2], &[2, 3, 1]);
        let keyed = pair_request(Some(7));
        let (places, verdicts) = pair_placed(&engine, &keyed, &first, &mut scratch);
        assert_eq!(places, (1, 2));
        let claimed = SessionVerdict::Claimed(first[0][1].id);
        assert_eq!(verdicts, (claimed, DecodeVerdict::CLAIMED));
        let moved = two_sides(&[0, 5, 2], &[1, 3, 9]);
        let (places, verdicts) = pair_placed(&engine, &keyed, &moved, &mut scratch);
        assert_eq!(places, (1, 2));
        assert_eq!(verdicts, (SessionVerdict::Pinned, DecodeVerdict::PINNED));
        let plain = pair_request(None);
        let (places, verdicts) = pair_placed(&engine, &plain, &moved, &mut scratch);
        assert_eq!(places, (0, 0));
        assert_eq!(verdicts, (SessionVerdict::None, DecodeVerdict::NONE));
        // Under no rule the picker's side reads the fewest in flight.
        let unruled = Engine::new(
            vec![manual_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        for facts in [&plain, &keyed] {
            let (places, verdicts) = pair_placed(&unruled, facts, &moved, &mut Scratch::new());
            assert_eq!(places, (0, 0));
            assert_eq!(verdicts, (SessionVerdict::None, DecodeVerdict::NONE));
        }
        // So it does on one side alone, beside a side whose picker picks:
        // on the decode side, then on the prefill side.
        for profiles in [
            vec![least_requests_profile(), manual_profile()],
            vec![manual_profile(), least_requests_profile()],
        ] {
            let one_side = Engine::new(
                profiles,
                vec![Binding {
                    decode_profile: Some(1),
                    ..pair_binding()
                }],
                1,
                Needs::default(),
            );
            let (places, _) = pair_placed(&one_side, &plain, &moved, &mut Scratch::new());
            assert_eq!(places, (0, 0));
        }
    }

    /// A prefill picker's decline passes through among first partners too:
    /// the pair's decision is the decline, by the picker's name, and no
    /// request waits for a prefill the picker was shown.
    #[test]
    fn a_prefill_pickers_decline_passes_through_among_first_partners() {
        let declining = Profile {
            name: "prefill".into(),
            filters: vec![Box::new(Healthy)],
            scorers: vec![],
            picker: Box::new(RefusesClass(1)),
        };
        let engine = pair_engine(declining);
        let mut prefill = candidate(0, 0);
        prefill.pair_class = 1;
        let mut decode = candidate(0, 0);
        decode.pair_class = 1;
        let (prefills, decodes) = ([prefill], [decode]);
        let first = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            table: PairTable::new(&[1], &[1]),
            first: FirstPartners::new(&[1], &[1]),
            ..PairPools::default()
        };
        let facts = RequestFacts {
            model: "m",
            ..RequestFacts::default()
        };
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let among = engine.decide_pair(&facts, &pool, first, &cx, &cx, &mut Scratch::new());
        for decision in [decide_pair(&engine, &prefills, &decodes), among] {
            match decision {
                Decision::Shed(shed) => {
                    assert_eq!(shed.reason, ShedReason::PolicyDeclined);
                    assert_eq!(shed.emptied_by, "refuses-class");
                }
                other => panic!("a prefill picker that declines answered {other:?}"),
            }
        }
    }

    /// A rule places a pair's prefill side over the classes the side is
    /// shown, in each of the two runs a host's first partners make.
    ///
    /// Prefill class 1 pairs with both decode classes, and its first
    /// partner is decode class 1. Prefill class 2 pairs with decode class
    /// 2 and has no first partner. While no first partner is open, a new
    /// key is placed over every open partner: on the idle prefill, of
    /// class 2. Once the first partner opens, the side is shown the
    /// prefills of class 1 alone. The key's pin stands outside them, so
    /// the rule places the key again among them, and its decode is the
    /// first partner. A pin's pick names the pin alone: shown every
    /// class, the rule would follow the pin to a prefill with no first
    /// partner, and the key would find no pair. Where the first run sheds,
    /// the second places a new key over every open partner, by the rule
    /// again: the pick states its claim. Each run leaves the classes it
    /// was shown in the scratch, where a side's shadow reads them.
    ///
    /// The classic router hands its policy the prefills that pair on a
    /// known runtime alone (the candidates of `pair_pass` in
    /// `model_gateway/src/routers/common/placement.rs`), and the classic
    /// manual policy selects anew for a key whose workers are not among
    /// those it is handed (`model_gateway/src/policies/manual.rs:303-308`).
    #[test]
    fn a_rule_places_a_pairs_prefill_among_first_partners_then_over_every_partner() {
        let engine = pair_session_engine(vec![least_requests_profile()], Assignment::Delegate);
        let mut scratch = Scratch::new();
        let table = PairTable::new(&[2, 3], &[1, 2, 2]);
        let first = FirstPartners::new(&[1, 1], &[1]);
        assert_eq!(first.within(&table), Ok(()));
        // A side's two workers, of class 1 and of class 2, under ids from
        // `first_id` on and the loads given.
        let rows = |loads: [u32; 2], first_id: u64| -> Vec<Candidate> {
            let mut rows = classed_rows(&[1, 2]);
            for (row, load) in rows.iter_mut().zip(loads) {
                (row.id, row.in_flight) = (WorkerId(first_id + u64::from(row.position)), load);
            }
            rows
        };
        // The idle worker of each side is the one of class 2.
        let (mut prefills, mut decodes) = (rows([5, 0], 100), rows([5, 0], 200));
        // The pair of `key`'s next request with its two verdicts, and the
        // classes its prefill side was shown.
        let placed = |key, prefills: &[Candidate], decodes: &[Candidate], scratch: &mut Scratch| {
            let pools = PairPools {
                prefills,
                decodes,
                table,
                first,
                ..PairPools::default()
            };
            let answer = pair_placed_in(&engine, &pair_request(Some(key)), pools, scratch);
            (answer, scratch.classes.clone())
        };
        let claim = DecodeVerdict::CLAIMED;
        let claimed = |id| (SessionVerdict::Claimed(WorkerId(id)), claim);
        // No first partner is open: decode class 1 has no healthy worker.
        decodes[0].health.healthy = false;
        let (answer, shown) = placed(7, &prefills, &decodes, &mut scratch);
        assert_eq!((answer, shown), (((1, 1), claimed(101)), vec![1, 2]));
        // The first partner opens, and the key's pin is not among the
        // prefills that have one.
        decodes[0].health.healthy = true;
        let (answer, shown) = placed(7, &prefills, &decodes, &mut scratch);
        assert_eq!((answer, shown), (((0, 0), claimed(100)), vec![1]));
        // The one prefill with a first partner open is not healthy: the
        // first run sheds, and the second is the rule's too.
        prefills[0].health.healthy = false;
        let (answer, shown) = placed(8, &prefills, &decodes, &mut scratch);
        assert_eq!((answer, shown), (((1, 1), claimed(101)), vec![1, 2]));
    }

    /// A side of a pair takes no turn of a list. Its request with no
    /// session value never draws, at any pool size, so a new session of a
    /// pair rotates through the survivors as that request does: on each
    /// side the two kinds of request take one rotation, turn by turn.
    ///
    /// The classic router asks one function for both kinds of request on a
    /// side, and it steps the counter of the list it is handed
    /// (`model_gateway/src/policies/round_robin.rs:142-167`).
    #[test]
    fn a_new_session_on_a_pairs_side_takes_the_survivors_rotation() {
        let rows = DRAW_FROM + 4;
        let engine = pair_session_engine(
            vec![round_robin_profile(), round_robin_profile()],
            Assignment::Delegate,
        );
        let sides = two_sides(&vec![0; rows], &vec![0; rows]);
        let mut scratch = Scratch::new();
        let keys = [None, Some(1), None, Some(2), Some(3), None];
        let places: Vec<(u32, u32)> = keys
            .into_iter()
            .map(|key| pair_placed(&engine, &pair_request(key), &sides, &mut scratch).0)
            .collect();
        let (prefill, decode) = places[0];
        let in_turn: Vec<(u32, u32)> = (0..keys.len() as u32)
            .map(|step| {
                (
                    (prefill + step) % rows as u32,
                    (decode + step) % rows as u32,
                )
            })
            .collect();
        assert_eq!(places, in_turn);
    }

    /// A decode profile whose picker declines the partners of a prefill
    /// where the first of them is of pair class 7.
    fn declining_decode_profile() -> Profile {
        Profile {
            name: "decode".into(),
            filters: vec![Box::new(Healthy)],
            scorers: vec![],
            picker: Box::new(RefusesClass(7)),
        }
    }

    /// A prefill the decode side declines is passed over under a rule too,
    /// and the claim its side's run made moves with the pair. The pick
    /// states the claim of the prefill that paired, and the key's pin names
    /// that prefill before any dispatch. So a second decision of the key,
    /// made before the first is dispatched, follows the prefill that pairs.
    /// The first decision's dispatch there settles the claim.
    #[test]
    fn a_claim_on_a_prefill_that_is_passed_over_moves_to_the_one_that_pairs() {
        // The idle prefill's decode is refused; the busy prefill's is not.
        let mut sides = two_sides(&[0, 5], &[0, 0]);
        for side in &mut sides {
            (side[0].pair_class, side[1].pair_class) = (7, 8);
        }
        let profiles = vec![least_requests_profile(), declining_decode_profile()];
        let engine = pair_session_engine(profiles, Assignment::Delegate);
        let keyed = pair_request(Some(7));
        let [prefill_session, _] = side_sessions(&keyed);
        let pools = PairPools {
            prefills: &sides[0],
            decodes: &sides[1],
            ..PairPools::default()
        };
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let paired = sides[0][1].id;
        // Two decisions of the key, neither dispatched: the first claims
        // the prefill that pairs, and the second follows that claim.
        let first = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
        let first = first.pick().expect("a pair");
        let went = *first.targets.first().expect("a target");
        assert_eq!(went.id, paired);
        assert_eq!(first.session_verdict, SessionVerdict::Claimed(paired));
        let pinned = Some([paired, WorkerId(0)]);
        assert_eq!(engine.sessions.pins(prefill_session), pinned);
        let second = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
        let second = second.pick().expect("the key's second pair");
        assert_eq!(second.targets.first().map(|target| target.id), Some(paired));
        assert_eq!(second.session_verdict, SessionVerdict::Pinned);
        // The first decision's dispatch on the prefill settles its claim.
        let lease = Lease {
            session: prefill_session,
            session_verdict: first.session_verdict,
            ..Lease::default()
        };
        let went = Target { pair: None, ..went };
        engine.on_dispatched(&lease, &went, Side::Prefill, 0);
        assert_eq!(engine.sessions.pins(prefill_session), pinned);
    }

    /// A pair's decision that forms no pair leaves no claim. Where the
    /// decode side declines the partners of every prefill placed, the claim
    /// the prefill side's run made is withdrawn, and the key holds no pin.
    /// Its next request is placed again, and pairs with a prefill whose
    /// decode is usable by then.
    #[test]
    fn a_pair_that_sheds_withdraws_the_claim_its_prefill_side_made() {
        // Every decode is of the refused class: no prefill pairs.
        let mut sides = two_sides(&[0, 5], &[0, 0]);
        for row in sides.iter_mut().flatten() {
            row.pair_class = 7;
        }
        let profiles = vec![least_requests_profile(), declining_decode_profile()];
        let engine = pair_session_engine(profiles, Assignment::Delegate);
        let keyed = pair_request(Some(7));
        let [prefill_session, _] = side_sessions(&keyed);
        let pools = PairPools {
            prefills: &sides[0],
            decodes: &sides[1],
            ..PairPools::default()
        };
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let mut scratch = Scratch::new();
        let declined = engine.decide_pair(&keyed, &pool, pools, &cx, &cx, &mut scratch);
        assert!(declined.pick().is_none(), "{declined:?}");
        let none = Some([WorkerId(0); 2]);
        assert_eq!(engine.sessions.pins(prefill_session), none);
        // The busy prefill's decode is usable by the key's next request.
        for side in &mut sides {
            side[1].pair_class = 8;
        }
        let (places, verdicts) = pair_placed(&engine, &keyed, &sides, &mut scratch);
        assert_eq!(places, (1, 1));
        assert_eq!(verdicts.0, SessionVerdict::Claimed(sides[0][1].id));
    }

    /// A decode side its rule places writes the decode side's buffers, as
    /// its profile's run does, and the prefill's buffer stays as the
    /// prefill's run left it. So the prefill's shadow speaks beside a pair
    /// whose decode side alone was steered past the overload veto, under a
    /// rule as under none: a lifted veto's mark on the prefill's buffer is
    /// the prefill's own.
    #[test]
    fn a_decode_side_under_a_rule_leaves_the_prefills_buffer_to_the_prefill() {
        let engine = |sessions: Option<SessionRule>| {
            let binding = Binding {
                shadow: Some(1),
                sessions,
                ..pair_binding()
            };
            let profiles = vec![strict_profile(), strict_profile()];
            Engine::new(profiles, vec![binding], 1, Needs::default())
        };
        // Every decode is over its thresholds: the binding's rule steers
        // the decode side, and the pair's trace names the lifted veto.
        let [prefills, mut decodes] = two_sides(&[0, 0], &[0, 0]);
        for decode in &mut decodes {
            decode.health.overloaded = true;
        }
        let rule = SessionRule {
            inflight_cap: 0,
            ..SESSIONS
        };
        for (sessions, key) in [(None, None), (Some(rule), Some(7))] {
            let (engine, facts) = (engine(sessions), pair_request(key));
            let pools = PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            };
            let (pool, cx) = (PoolFacts::default(), Constraints::default());
            let mut scratch = Scratch::new();
            let live = engine.decide_pair(&facts, &pool, pools, &cx, &cx, &mut scratch);
            let pick = live.pick().expect("a pair");
            assert_eq!(pick.trace.lifted, LiftedVeto::Overload, "{key:?}");
            let side = PairSide::Prefill;
            let shadow =
                engine.shadow_side(0, side, &facts, &pool, pools, &cx, &cx, pick, &mut scratch);
            assert!(
                shadow.is_some_and(|shadow| shadow.pick().is_some()),
                "{key:?}"
            );
        }
    }

    /// A later decode of a key follows the key's decode pin: the call
    /// names the decode the pair's first decision claimed, whether or not
    /// it is handed that decode, and states the pin in the first target's
    /// pair. The pick states no verdict: the call decided no prefill.
    #[test]
    fn a_later_decode_follows_its_keys_pin_and_states_it_in_its_pair() {
        let engine = pair_session_engine(vec![least_requests_profile()], Assignment::Delegate);
        let mut scratch = Scratch::new();
        let first = two_sides(&[0], &[5, 0, 5]);
        let keyed = pair_request(Some(7));
        let (places, _) = pair_placed(&engine, &keyed, &first, &mut scratch);
        assert_eq!(places, (0, 1));
        // The loads point at the third decode by the later call.
        let [prefills, decodes] = two_sides(&[0], &[5, 9, 0]);
        let named = (decodes[1].id, decodes[1].dp_rank);
        for held in [Some(named), None] {
            let pools = PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            };
            let (pool, cx) = (PoolFacts::default(), Constraints::default());
            let again = engine.decide_decode(
                &keyed,
                &pool,
                pools,
                prefills[0].id,
                held,
                &cx,
                &mut scratch,
            );
            let again = again.pick().expect("a decode decided again");
            let pair = again.targets.first().and_then(|target| target.pair);
            let pair = pair.expect("a decode");
            assert_eq!(
                (pair.decode_position, pair.decode_verdict),
                (1, DecodeVerdict::PINNED)
            );
            assert_eq!(again.session_verdict, SessionVerdict::None);
        }
        // With no key the later call follows the loads and states nothing.
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let (pool, cx) = (PoolFacts::default(), Constraints::default());
        let plain = pair_request(None);
        let again = engine.decide_decode(
            &plain,
            &pool,
            pools,
            prefills[0].id,
            None,
            &cx,
            &mut scratch,
        );
        let pair = again.pick().and_then(|pick| pick.targets.first()?.pair);
        let pair = pair.expect("a decode");
        assert_eq!(
            (pair.decode_position, pair.decode_verdict),
            (2, DecodeVerdict::NONE)
        );
    }

    /// The profile's filters narrow the pool before the assignment places a
    /// request with no key: a worker that is down takes none under any
    /// assignment, or with no rule, though it holds the fewest requests and
    /// the fewest keys.
    #[test]
    fn a_keyless_request_under_the_manual_picker_passes_the_profiles_filters() {
        let mut pool = vec![candidate(0, 0), candidate(1, 3), candidate(2, 2)];
        pool[0].health.healthy = false;
        pool[1].active_sessions = 5;
        pool[2].active_sessions = 4;
        let rules = [
            Assignment::Random,
            Assignment::MinLoad,
            Assignment::MinGroup,
            Assignment::Delegate,
        ]
        .map(|assignment| {
            Some(SessionRule {
                assignment,
                ..SESSIONS
            })
        });
        for rule in rules.into_iter().chain([None]) {
            let engine = manual_engine(rule);
            for seed in 1..=64u64 {
                let (taken, _) = decide_seeded(&engine, &pool, 0, seed);
                assert_ne!(taken, 0, "{rule:?}, seed {seed}: the worker that is down");
            }
        }
    }

    /// A lease takes its session from the request and its verdict from the
    /// pick, for every keyed request: under a sessions rule the dispatch
    /// pins by the two, and with no rule the same lease moves no pin, the
    /// verdict alone deciding it.
    #[test]
    fn a_lease_takes_the_requests_session_and_the_picks_verdict() {
        let pool = vec![candidate(0, 3), candidate(1, 1)];
        let facts = RequestFacts {
            model: "m",
            session_key: Identity::hashed(7),
            now_micros: 10,
            ..RequestFacts::default()
        };
        let pick_of = |engine: &Engine| {
            let mut scratch = Scratch::new();
            *engine
                .decide(
                    &facts,
                    &PoolFacts::default(),
                    &pool,
                    &Constraints::default(),
                    &mut scratch,
                )
                .pick()
                .expect("a pick")
        };
        let lease_of = |pick: &Pick| Lease {
            session: facts.session(),
            session_verdict: pick.session_verdict,
            issued_micros: 10,
            ..Lease::default()
        };
        let ruled = manual_engine(Some(SESSIONS));
        let miss = pick_of(&ruled);
        let target = *miss.targets.first().expect("a target");
        ruled.on_dispatched(&lease_of(&miss), &target, Side::Single, 0);
        let hit = pick_of(&ruled);
        assert_eq!(hit.session_verdict, SessionVerdict::Pinned);
        assert_eq!(hit.targets.first().map(|t| t.id), Some(target.id));
        let bare = manual_engine(None);
        let unruled = pick_of(&bare);
        assert_eq!(unruled.session_verdict, SessionVerdict::None);
        let target = *unruled.targets.first().expect("a target");
        bare.on_dispatched(&lease_of(&unruled), &target, Side::Single, 0);
        assert!(bare.sessions.is_empty(), "no rule, no row");
    }

    /// The assignment's path is the single plan's alone: under `epd` an
    /// encode item of a `manual` profile is not placed by it, so the picker
    /// is asked and declines by its name, as on a side of a pair.
    #[test]
    fn the_manual_picker_declines_an_encode_item_under_the_epd_plan() {
        let engine = Engine::new(
            vec![strict_profile(), manual_profile()],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Epd { encode: 1 },
                profile: 0,
                decode_profile: Some(0),
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        );
        let mut scratch = Scratch::new();
        match engine.decide(
            &encode_facts(1),
            &PoolFacts::default(),
            &[encode_candidate(0, true, true)],
            &Constraints::default(),
            &mut scratch,
        ) {
            Decision::Shed(shed) => {
                assert_eq!(shed.reason, ShedReason::PolicyDeclined);
                assert_eq!(shed.emptied_by, "manual");
            }
            other => panic!("a manual encode profile answered {other:?}"),
        }
    }

    fn encode_candidate(position: u32, ec_compatible: bool, vision: bool) -> Candidate {
        Candidate {
            role: Role::Encode,
            ec_compatible,
            vision,
            ..candidate(position, 0)
        }
    }

    /// Profile 0 is the strict pair side, profile 1 the encode profile.
    fn epd_engine() -> Engine {
        Engine::new(
            vec![strict_profile(), least_requests_profile()],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Epd { encode: 1 },
                profile: 0,
                decode_profile: Some(0),
                shadow: None,
                decode_shadow: None,
                sessions: None,
                overload: None,
            }],
            1,
            Needs::default(),
        )
    }

    fn encode_facts(images: u16) -> RequestFacts<'static> {
        RequestFacts {
            model: "m",
            phase: Phase::Encode,
            media: MediaCounts {
                images,
                ..MediaCounts::default()
            },
            ..RequestFacts::default()
        }
    }

    /// A single or an encode request under the pair plan, and a single one
    /// under `epd`, resolve to no profile and are declined by the binding,
    /// as a pair decision under a single binding is; the pair phases and
    /// `epd`'s encode phase resolve to their sides.
    #[test]
    fn a_single_or_encode_request_under_the_pair_plan_is_declined_by_the_binding() {
        let pair = Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        for phase in [Phase::Single, Phase::Encode] {
            assert!(pair.profile_for("m", phase).is_none(), "{phase:?}");
            let facts = RequestFacts {
                model: "m",
                phase,
                ..RequestFacts::default()
            };
            let mut scratch = Scratch::new();
            match pair.decide(
                &facts,
                &PoolFacts::default(),
                &[candidate(0, 0)],
                &Constraints::default(),
                &mut scratch,
            ) {
                Decision::Shed(shed) => {
                    assert_eq!(shed.reason, ShedReason::PolicyDeclined, "{phase:?}");
                    assert_eq!(shed.emptied_by, "binding", "{phase:?}");
                }
                other => panic!("{phase:?} under the pair plan answered {other:?}"),
            }
        }
        assert!(pair.profile_for("m", Phase::Prefill).is_some());
        assert!(pair.profile_for("m", Phase::Decode).is_some());
        let epd = epd_engine();
        assert!(epd.profile_for("m", Phase::Single).is_none());
        let side = |phase: Phase| {
            epd.profile_for("m", phase)
                .map(|(i, p)| (i, p.name.clone()))
        };
        assert_eq!(side(Phase::Encode), Some((1, "least".to_string())));
        assert_eq!(side(Phase::Prefill), Some((0, "strict".to_string())));
        assert_eq!(side(Phase::Decode), Some((0, "strict".to_string())));
    }

    /// The `epd` plan reads the rows' compatibility and vision bits before
    /// the encode profile's own stages, and a pool they empty names them; a
    /// single-plan binding never reads the bits.
    #[test]
    fn the_epd_plan_filters_encoders_by_ec_compatibility_and_vision_and_names_the_stage() {
        let epd = epd_engine();
        let decide = |facts: &RequestFacts<'_>, pool: &[Candidate]| {
            let mut scratch = Scratch::new();
            epd.decide(
                facts,
                &PoolFacts::default(),
                pool,
                &Constraints::default(),
                &mut scratch,
            )
        };
        let pool = [
            encode_candidate(0, false, true),
            encode_candidate(1, true, true),
            encode_candidate(2, true, false),
        ];
        let text = decide(&encode_facts(0), &pool);
        let pick = text.pick().expect("a pick");
        assert!(pick.targets.iter().all(|t| t.position != 0), "{pick:?}");
        assert_eq!(
            pick.trace.survivors.get(),
            Some(2),
            "the incompatible encoder is out"
        );
        let image = decide(&encode_facts(1), &pool);
        let pick = image.pick().expect("a pick");
        let order: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
        assert_eq!(order, [1], "an image takes the vision encoder alone");
        let shed_of = |facts: &RequestFacts<'_>, pool: &[Candidate]| match decide(facts, pool) {
            Decision::Shed(shed) => (shed.emptied_by, shed.reason),
            other => panic!("an emptied pool answered {other:?}"),
        };
        assert_eq!(
            shed_of(&encode_facts(0), &[encode_candidate(0, false, true)]),
            ("ec-compatible", ShedReason::NoEndpoints)
        );
        assert_eq!(
            shed_of(&encode_facts(1), &[encode_candidate(0, true, false)]),
            ("vision", ShedReason::NoEndpoints)
        );
        assert!(
            decide(&encode_facts(0), &[encode_candidate(0, true, false)])
                .pick()
                .is_some(),
            "a request without images takes an encoder without a vision encoder"
        );
        let mut sick = encode_candidate(0, true, true);
        sick.health.healthy = false;
        assert_eq!(
            shed_of(&encode_facts(0), &[sick, encode_candidate(1, false, true)]).0,
            "healthy",
            "the profile's stage is named after the plan's"
        );
        let single = engine();
        let mut scratch = Scratch::new();
        let regular = [encode_candidate(0, false, false)];
        assert!(
            single
                .decide(
                    &encode_facts(1),
                    &PoolFacts::default(),
                    &regular,
                    &Constraints::default(),
                    &mut scratch,
                )
                .pick()
                .is_some(),
            "a single-plan binding reads no compatibility bit"
        );
    }

    /// A video asks vision as an image does: a request of the encode phase
    /// whose media is video alone takes the vision encoder alone, and a
    /// pool with none is shed under the rule's stage. Audio alone asks
    /// none. The classic router asks vision for any pixel item, an image
    /// or a video (`is_pixel_modality` and `request_needs_vision`,
    /// `model_gateway/src/routers/grpc/multimodal/capability.rs:325-344`
    /// on main d6517a4fe).
    #[test]
    fn a_video_alone_takes_the_vision_encoder_under_the_epd_plan() {
        let epd = epd_engine();
        let decide = |media: MediaCounts, pool: &[Candidate]| {
            let facts = RequestFacts {
                media,
                ..encode_facts(0)
            };
            let cx = Constraints::default();
            epd.decide(
                &facts,
                &PoolFacts::default(),
                pool,
                &cx,
                &mut Scratch::new(),
            )
        };
        let video = MediaCounts {
            video: 1,
            ..MediaCounts::default()
        };
        let audio = MediaCounts {
            audio: 1,
            ..MediaCounts::default()
        };
        let pool = [
            encode_candidate(0, true, false),
            encode_candidate(1, true, true),
        ];
        let seen = decide(video, &pool);
        let pick = seen.pick().expect("a pick");
        let order: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
        assert_eq!(order, [1], "a video takes the vision encoder alone");
        match decide(video, &pool[..1]) {
            Decision::Shed(shed) => assert_eq!(
                (shed.emptied_by, shed.reason),
                ("vision", ShedReason::NoEndpoints)
            ),
            other => panic!("a pool with no vision encoder answered {other:?}"),
        }
        assert!(
            decide(audio, &pool[..1]).pick().is_some(),
            "audio alone takes an encoder without a vision encoder"
        );
    }

    #[test]
    fn the_epd_binding_serves_its_pair_half_through_decide_pair() {
        let epd = epd_engine();
        let mut prefills = [candidate(0, 0)];
        prefills[0].pair_class = 7;
        let mut decodes = [candidate(0, 0)];
        decodes[0].pair_class = 7;
        let decision = decide_pair(&epd, &prefills, &decodes);
        let pick = decision
            .pick()
            .expect("the epd binding's pair half decides");
        assert_eq!(
            pick.targets
                .first()
                .and_then(|t| t.pair)
                .map(|p| p.decode_position),
            Some(0)
        );
    }

    // ---- a decision that draws first

    /// `rows` ready candidates in the host's order, the rows `refused`
    /// names not ready.
    fn pool_of(rows: u32, refused: impl Fn(u32) -> bool) -> Vec<Candidate> {
        (0..rows)
            .map(|i| Candidate {
                health: Health {
                    healthy: !refused(i),
                    ..Health::default()
                },
                ..candidate(i, i % 5)
            })
            .collect()
    }

    /// A preset's engine. `power_of_two/requests` is that preset's form
    /// that compares its two draws by the requests in flight.
    fn preset(name: &str) -> Engine {
        let document = match name {
            "power_of_two/requests" => PolicySpec::power_of_two(PowerOfTwoMetric::Requests),
            name => PolicySpec::preset(name),
        };
        document
            .expect("a preset")
            .validate()
            .expect("the preset validates")
    }

    fn request(seed: u64) -> RequestFacts<'static> {
        RequestFacts {
            model: "m",
            seed: Draw::new(seed).next_u64(),
            ..RequestFacts::default()
        }
    }

    /// The whole pass of the engine's profile over `pool`, as a decision
    /// under [`DRAW_FROM`] rows runs it.
    fn whole_pass(engine: &Engine, facts: &RequestFacts<'_>, pool: &[Candidate]) -> Decision {
        let (entry, index, _) = engine
            .resolve(facts.model, Phase::Single)
            .expect("a binding");
        run_whole(
            engine,
            index,
            facts,
            &PoolFacts::default(),
            Set {
                candidates: pool,
                class: None,
                partnered: None,
                fold: None,
                encode: false,
                vetoed: entry.vetoed,
            },
            &Constraints::default(),
            &mut Buffers::default(),
        )
    }

    fn drawn(engine: &Engine, facts: &RequestFacts<'_>, pool: &[Candidate]) -> Decision {
        engine.decide(
            facts,
            &PoolFacts::default(),
            pool,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }

    /// A decision over a pool whose host says that every row holds a
    /// report ([`PoolFacts::reported`]): what a profile is asked by rank
    /// under.
    fn reported(engine: &Engine, facts: &RequestFacts<'_>, pool: &[Candidate]) -> Decision {
        let said = PoolFacts {
            reported: true,
            ..PoolFacts::default()
        };
        engine.decide(
            facts,
            &said,
            pool,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }

    fn positions(decision: &Decision) -> Vec<u32> {
        decision.pick().map_or_else(Vec::new, |pick| {
            pick.targets.iter().map(|t| t.position).collect()
        })
    }

    /// The fleets a draw is held against the whole pass over: every row
    /// ready, one in seventeen not, every second not, one row ready, none.
    const REFUSED: [fn(u32) -> bool; 5] = [
        |_| false,
        |i| i % 17 == 5,
        |i| i % 2 == 1,
        |i| i != 6,
        |_| true,
    ];

    #[test]
    fn a_draw_hands_out_only_rows_the_whole_pass_keeps() {
        let mut drew = 0u32;
        let mut widened = 0u32;
        const NAMES: [&str; 5] = [
            "passthrough",
            "random",
            "round_robin",
            "power_of_two",
            "power_of_two/requests",
        ];
        let mut drew_by = [0u32; NAMES.len()];
        for (which, name) in NAMES.into_iter().enumerate() {
            // Round robin turns a rotation at each decision: the pass that
            // answers for it runs on a twin, so neither turns the other's.
            let (engine, twin) = (preset(name), preset(name));
            for rows in [8u32, 9, 17, 64, 257] {
                for (fleet, refused) in REFUSED.into_iter().enumerate() {
                    let pool = pool_of(rows, refused);
                    for seed in 0..40u64 {
                        // Every third request already tried a row: the
                        // soft filter is asked of the rows drawn too.
                        let tried = [pool[(seed as usize * 7) % pool.len()].id];
                        let facts = RequestFacts {
                            tried: if seed % 3 == 0 { &tried } else { &[] },
                            ..request(seed ^ u64::from(rows) << 32)
                        };
                        let untried = |c: &&Candidate| !facts.tried.contains(&c.id);
                        let ready = || pool.iter().filter(|c| c.health.healthy);
                        // What the pass keeps: the ready rows not tried, or
                        // every ready row when that leaves none.
                        let kept: Vec<u32> = if ready().any(|c| untried(&c)) {
                            ready().filter(untried).map(|c| c.position).collect()
                        } else {
                            ready().map(|c| c.position).collect()
                        };
                        let got = drawn(&engine, &facts, &pool);
                        let want = whole_pass(&twin, &facts, &pool);
                        let at = format!("{name}, {rows} rows, fleet {fleet}, seed {seed}");
                        assert_eq!(got.trace().seed, facts.seed, "{at}: the request's seed");
                        if kept.is_empty() {
                            assert_eq!(got, want, "{at}: a shed is the whole pass's own");
                            continue;
                        }
                        let pick = got.pick().unwrap_or_else(|| panic!("{at}: a pick"));
                        if pick.trace.drew_first() {
                            drew += 1;
                            drew_by[which] += 1;
                            assert_eq!(pick.trace.survivors.get(), None, "{at}: no count");
                        } else {
                            widened += 1;
                            assert_eq!(pick.trace.survivors.get(), Some(kept.len() as u32), "{at}");
                        }
                        let (got, want) = (positions(&got), positions(&want));
                        assert!(
                            got.iter().all(|row| kept.contains(row)),
                            "{at}: {got:?} holds a row a filter refuses (kept {kept:?})"
                        );
                        let mut once = got.clone();
                        once.sort_unstable();
                        once.dedup();
                        assert_eq!(once.len(), got.len(), "{at}: a row handed out twice");
                        assert!(got.len() <= want.len(), "{at}: {got:?} against {want:?}");
                        if fleet == 0 && facts.tried.is_empty() {
                            assert_eq!(got.len(), want.len(), "{at}: every target, no row refused");
                        }
                        if name == "passthrough" {
                            assert_eq!(got, want[..got.len()], "{at}: row for row");
                        }
                    }
                }
            }
        }
        assert!(
            drew > 1_000 && widened > 100,
            "{drew} drew, {widened} widened"
        );
        // Each of the five draws, the requests form of `power_of_two` too.
        assert!(
            drew_by.iter().all(|&drew| drew > 100),
            "{drew_by:?} drew, by picker"
        );
    }

    #[test]
    fn a_pool_under_the_draws_size_is_read_whole() {
        for name in ["passthrough", "random", "round_robin", "power_of_two"] {
            let engine = preset(name);
            let facts = request(3);
            let under = pool_of(DRAW_FROM as u32 - 1, |_| false);
            let trace = *drawn(&engine, &facts, &under).trace();
            assert!(!trace.drew_first(), "{name}");
            assert_eq!(trace.survivors.get(), Some(DRAW_FROM as u32 - 1), "{name}");
            let at = pool_of(DRAW_FROM as u32, |_| false);
            let trace = *drawn(&engine, &facts, &at).trace();
            assert!(trace.drew_first(), "{name}");
            assert_eq!(
                trace.survivors.get(),
                None,
                "{name}: a draw counts no survivor"
            );
        }
    }

    #[test]
    fn a_profile_with_a_stage_that_is_not_row_wise_never_draws() {
        /// A filter that keeps every row and does not say its word on one
        /// row is its whole word: a plugin's filter is such a one.
        #[derive(Debug)]
        struct Whole;
        impl Filter for Whole {
            fn name(&self) -> &'static str {
                "whole"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }
        }
        // `least-requests` does not say it may be handed drawn rows.
        let engine = Engine::new(
            vec![Profile {
                name: "two".into(),
                filters: vec![Box::new(Healthy)],
                scorers: vec![WeightedScorer {
                    scorer: Box::new(LeastRequests),
                    weight: 1.0,
                }],
                picker: Box::new(PowerOfTwo),
            }],
            vec![single_binding()],
            1,
            Needs::default(),
        );
        let pool = pool_of(64, |_| false);
        let trace = *drawn(&engine, &request(1), &pool).trace();
        assert!(!trace.drew_first());
        assert_eq!(trace.survivors.get(), Some(64));
        // Nor does a picker that compares every survivor.
        let trace = *drawn(&preset("least_load"), &request(1), &pool).trace();
        assert!(!trace.drew_first());
        // Nor does a filter that is not row-wise, under a picker that draws
        // and at the first size it draws from.
        let engine = Engine::new(
            vec![Profile {
                name: "whole".into(),
                filters: vec![Box::new(Healthy), Box::new(Whole)],
                scorers: Vec::new(),
                picker: Box::new(Passthrough),
            }],
            vec![single_binding()],
            1,
            Needs::default(),
        );
        let pool = pool_of(DRAW_FROM as u32, |_| false);
        let trace = *drawn(&engine, &request(1), &pool).trace();
        assert!(!trace.drew_first());
        assert_eq!(trace.survivors.get(), Some(DRAW_FROM as u32));
    }

    #[test]
    fn the_drawing_pickers_keep_their_law_over_the_rows_the_pass_keeps() {
        // Thirteen rows kept of sixteen, equal loads: the pick is uniform
        // over them, under `random` and under a tie of the two drawn.
        let pool: Vec<Candidate> = pool_of(16, |i| [3, 7, 11].contains(&i))
            .into_iter()
            .map(|c| Candidate { in_flight: 0, ..c })
            .collect();
        const DECISIONS: u32 = 26_000;
        for name in ["random", "power_of_two"] {
            let engine = preset(name);
            let mut scratch = Scratch::new();
            let mut first = [0u32; 16];
            let mut any = [0u32; 16];
            for seed in 0..u64::from(DECISIONS) {
                let decision = engine.decide(
                    &request(seed),
                    &PoolFacts::default(),
                    &pool,
                    &Constraints::default(),
                    &mut scratch,
                );
                let rows = positions(&decision);
                first[rows[0] as usize] += 1;
                for row in rows {
                    any[row as usize] += 1;
                }
            }
            // 2,000 a row expected, a standard deviation of 43: five of them.
            for row in 0..16usize {
                if [3, 7, 11].contains(&row) {
                    assert_eq!(any[row], 0, "{name}: a refused row handed out");
                } else {
                    assert!(
                        first[row].abs_diff(DECISIONS / 13) < 220,
                        "{name}: {first:?}"
                    );
                }
            }
            let (low, high) = (any.iter().filter(|n| **n > 0).min(), any.iter().max());
            let (low, high) = (*low.expect("a row"), *high.expect("a row"));
            assert!(
                high - low < high / 10,
                "{name}: every target, by row: {any:?}"
            );
        }
    }

    #[test]
    fn round_robin_turns_the_pool_and_shares_the_refused_rows_turns() {
        let engine = preset("round_robin");
        let mut scratch = Scratch::new();
        let mut turn = |pool: &[Candidate], model: &'static str| {
            let facts = RequestFacts {
                model,
                ..request(0)
            };
            let decision = engine.decide(
                &facts,
                &PoolFacts::default(),
                pool,
                &Constraints::default(),
                &mut scratch,
            );
            positions(&decision)
        };
        // No row refused: a turn of the pool visits every row once, in the
        // host's order, and the rows behind the pick are its alternates.
        let ready = pool_of(32, |_| false);
        let start = turn(&ready, "a")[0];
        for step in 1..=32u32 {
            let rows = turn(&ready, "a");
            let want: Vec<u32> = (0..4).map(|k| (start + step + k) % 32).collect();
            assert_eq!(rows, want, "step {step}");
        }
        // Another model's pool turns its own rotation: its decisions take
        // no turn of the first.
        let other = turn(&ready, "b")[0];
        assert_eq!(turn(&ready, "a")[0], (start + 33) % 32);
        assert_eq!(turn(&ready, "b")[0], (other + 1) % 32);
        // Three rows refused, twenty turns of the pool: a kept row takes its
        // own twenty turns and an even share of the refused rows' sixty.
        let pool = pool_of(32, |i| [3, 7, 11].contains(&i));
        let mut picks = [0u32; 32];
        for _ in 0..32 * 20 {
            picks[turn(&pool, "c")[0] as usize] += 1;
        }
        for (row, &n) in picks.iter().enumerate() {
            if [3, 7, 11].contains(&row) {
                assert_eq!(n, 0, "a refused row picked");
            } else {
                assert!((22..=23).contains(&n), "row {row}: {picks:?}");
            }
        }
    }

    /// Under a sessions rule that leaves a new session to the profile
    /// (`delegate`), round robin turns one rotation for every request it
    /// places. A request with no session value, a new session and a session
    /// at its cap take the pool's rows one after the other: under the
    /// draw's floor, at it and above it, and under a profile whose plain
    /// request is not the draw's, since its scorer reads the rows together.
    /// A session's pass counted its survivors, and a turn that lands on a
    /// refused row is left to the stages. So is a request its host holds to
    /// places, as its plain request draws nothing. Where a soft filter
    /// refuses every row and steps aside, the draw keeps nobody for either
    /// kind of request, and both take the survivors' rotation. On a fresh
    /// engine each, a new session is decided as a request with no session
    /// value is: by the draw where the turn's row is kept, its trace the
    /// draw's with the pass's count, and by the stages where a filter, a
    /// host's filter or the host's veto refuses that row. A row the pass
    /// refused is not taken at the turn, whatever its filter says of it
    /// there. A host's picker that takes turns and orders nothing of its
    /// draw is the stages' too.
    /// `random` takes no turn: the stages alone place its new session.
    ///
    /// The classic router asks one function for all three kinds of request,
    /// and so steps the same counters
    /// (`model_gateway/src/policies/round_robin.rs:142-167`): for a request
    /// with no session value at `model_gateway/src/policies/registry.rs:471`,
    /// for a new session at `:581` and for a session over its cap at `:552`.
    #[test]
    fn round_robin_under_delegate_turns_one_rotation_for_every_request() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let scored = || Profile {
            scorers: vec![WeightedScorer {
                scorer: Box::new(LeastRequests),
                weight: 1.0,
            }],
            ..round_robin_profile()
        };
        assert!(draws_first(&round_robin_profile()) && !draws_first(&scored()));
        let sizes = [DRAW_FROM - 1, DRAW_FROM, 64].map(|rows| (rows, false));
        for profile in [round_robin_profile as fn() -> Profile, scored] {
            // The last pool is held to its every place.
            for (rows, held) in sizes.into_iter().chain([(DRAW_FROM, true)]) {
                let engine = session_engine_of(profile(), SESSIONS);
                let pool = pool_of(rows as u32, |_| false);
                let every: Vec<u32> = (0..rows as u32).collect();
                let mut scratch = Scratch::new();
                let mut turn = |pool: &[Candidate], key: Option<u64>| {
                    let mut facts = request(0);
                    if let Some(key) = key {
                        facts.session_key = Identity::hashed(key);
                    }
                    let shared = PoolFacts::default();
                    let cx = Constraints {
                        places: held.then_some(&every[..]),
                        ..Constraints::default()
                    };
                    let decision = engine.decide(&facts, &shared, pool, &cx, &mut scratch);
                    let pick = decision.pick().expect("a pick");
                    let target = *pick.targets.first().expect("a target");
                    if key.is_some() {
                        assert!(!pick.trace.drew_first(), "a session's pass counted");
                        let lease = Lease {
                            session: facts.session(),
                            session_verdict: pick.session_verdict,
                            ..Lease::default()
                        };
                        engine.on_dispatched(&lease, &target, Side::Single, 0);
                    }
                    target.position as usize
                };
                let first = turn(&pool, None);
                let pinned = turn(&pool, Some(1));
                let mut places = vec![first, pinned, turn(&pool, None)];
                // The session's worker carries the rule's cap of its requests.
                let mut capped = pool.clone();
                capped[pinned].session_in_flight = SESSIONS.inflight_cap;
                places.push(turn(&capped, Some(1)));
                places.push(turn(&pool, None));
                places.push(turn(&pool, Some(2)));
                let in_turn: Vec<usize> = (0..6).map(|step| (first + step) % rows).collect();
                assert_eq!(places, in_turn, "{rows} rows, held {held}");
                // A turn that lands on a refused row is left to the stages,
                // and every other turn of the pool is taken where it is asked.
                let ailing = pool_of(rows as u32, |row| row == 3);
                let keys = 10..10 + rows as u64;
                let lands: Vec<usize> = keys.map(|key| turn(&ailing, Some(key))).collect();
                assert!(!lands.contains(&3), "{rows} rows, held {held}: {lands:?}");
                let asked = draws_first(&profile()) && rows >= DRAW_FROM && !held;
                let taken = |(step, &row): (usize, &usize)| {
                    !asked || [3, row].contains(&((first + 6 + step) % rows))
                };
                assert!(lands.iter().enumerate().all(taken), "{rows}: {lands:?}");
            }
        }
        // A soft filter that refuses every row steps aside, and the pass
        // keeps them all. A request with no session value is refused at its
        // draw there, which asks every filter of the row it lands on, and
        // falls to the survivors' rotation. A new session falls to it too.
        let roomless = || Profile {
            filters: vec![Box::new(Healthy), Box::new(HasRoom)],
            ..round_robin_profile()
        };
        assert!(draws_first(&roomless()));
        let engine = session_engine_of(roomless(), SESSIONS);
        let pool = pool_of(DRAW_FROM as u32, |_| false);
        let mut scratch = Scratch::new();
        let mut turn = |key: Option<u64>| {
            let mut facts = request(0);
            if let Some(key) = key {
                facts.session_key = Identity::hashed(key);
            }
            let (shared, cx) = (PoolFacts::default(), Constraints::default());
            let decision = engine.decide(&facts, &shared, &pool, &cx, &mut scratch);
            positions(&decision)[0] as usize
        };
        let first = turn(None);
        let keyed = |step: u64| (step % 2 == 1).then_some(step);
        let places: Vec<usize> = (1..6).map(|step| turn(keyed(step))).collect();
        let in_turn: Vec<usize> = (1..6).map(|step| (first + step) % DRAW_FROM).collect();
        assert_eq!(places, in_turn, "a soft filter refuses every row");
        // A fresh engine's request with no session value and a fresh
        // engine's new session are decided alike. Where the turn's row is
        // kept, the session's targets are that request's draw, and its
        // trace is the draw's with the count of the session's own pass.
        // Where the turn's row is refused, the stages place both on the same
        // rows: by a filter of the fold, by a host's filter, by the host's
        // veto, with every row by a soft filter of the fold or of a host,
        // and with every row under the host's veto, which the rule lifts.
        let hosted = || Profile {
            filters: vec![Box::new(Hosted(Box::new(Healthy)))],
            ..round_robin_profile()
        };
        let hosted_roomless = || Profile {
            filters: vec![
                Box::new(Hosted(Box::new(Healthy))),
                Box::new(Hosted(Box::new(HasRoom))),
            ],
            ..round_robin_profile()
        };
        assert!(draws_first(&hosted()) && draws_first(&hosted_roomless()));
        // A host's filter whose answer moves between two asks of one row, as
        // a filter that reads what another decision moves can: it refuses
        // `row` the first time it is asked of it, and keeps it after.
        #[derive(Debug)]
        struct Flips {
            row: u32,
            asked: AtomicBool,
        }
        impl Filter for Flips {
            fn name(&self) -> &'static str {
                "flips"
            }

            fn row_wise(&self) -> bool {
                true
            }

            fn keep(&self, _: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
                c.position != self.row || self.asked.swap(true, Ordering::Relaxed)
            }
        }
        let shared = PoolFacts {
            generation: 7,
            ..PoolFacts::default()
        };
        let decide = |profile: Profile, pool: &[Candidate], cx: &Constraints<'_>, keyed: bool| {
            let engine = session_engine_of(profile, SESSIONS);
            let mut facts = request(5);
            if keyed {
                facts.session_key = Identity::hashed(1);
            }
            let decision = engine.decide(&facts, &shared, pool, cx, &mut Scratch::new());
            (positions(&decision), *decision.trace())
        };
        let free = Constraints::default();
        for rows in [DRAW_FROM as u32, 64] {
            let full = pool_of(rows, |_| false);
            let turn = decide(round_robin_profile(), &full, &free, false).0[0];
            // The turn's row kept, by the fold's test and by a host's filter.
            let apart = pool_of(rows, |row| row == (turn + 8) % rows);
            for profile in [round_robin_profile as fn() -> Profile, hosted] {
                let (plain, drew) = decide(profile(), &apart, &free, false);
                let (keyed, counted) = decide(profile(), &apart, &free, true);
                assert!(drew.drew_first() && plain[0] == turn, "{rows}: {plain:?}");
                assert_eq!(
                    (drew.made_under(), drew.seed, drew.generation),
                    (Some(0), request(5).seed, 7)
                );
                let pass = Trace {
                    survivors: Survivors::counted(rows - 1),
                    ..drew
                };
                assert_eq!((keyed, counted), (plain, pass), "{rows} rows");
            }
            let ailing = pool_of(rows, |row| row == turn);
            let stalled = [turn];
            let vetoed = Constraints {
                stalled: &stalled,
                ..Constraints::default()
            };
            let every: Vec<u32> = (0..rows).collect();
            let doubted = Constraints {
                stalled: &every,
                ..Constraints::default()
            };
            // A profile, its pool and what the host holds the request to.
            type State<'a> = (fn() -> Profile, &'a [Candidate], &'a Constraints<'a>);
            let refused: [State<'_>; 6] = [
                (round_robin_profile, &ailing, &free),
                (hosted, &ailing, &free),
                (round_robin_profile, &full, &vetoed),
                (roomless, &full, &free),
                (hosted_roomless, &full, &free),
                (round_robin_profile, &full, &doubted),
            ];
            for (state, (profile, pool, cx)) in refused.into_iter().enumerate() {
                let (plain, whole) = decide(profile(), pool, cx, false);
                let (keyed, session) = decide(profile(), pool, cx, true);
                assert!(!whole.drew_first(), "{rows} rows, state {state}");
                assert!(!plain.is_empty(), "{rows} rows, state {state}: a pick");
                assert_eq!(keyed, plain, "{rows} rows, state {state}");
                // The stages place both requests, by one trace: it says the
                // lift where every row is under the host's veto.
                let lifted = if state == 5 {
                    LiftedVeto::Liveness
                } else {
                    LiftedVeto::None
                };
                assert_eq!(whole.lifted, lifted, "{rows} rows, state {state}");
                assert_eq!(session, whole, "{rows} rows, state {state}");
            }
            // The row a filter refused at the pass is no survivor, whatever
            // the filter says of it at the turn: the turn does not take it.
            let flips = Profile {
                filters: vec![Box::new(Flips {
                    row: turn,
                    asked: AtomicBool::new(false),
                })],
                ..round_robin_profile()
            };
            let (keyed, counted) = decide(flips, &full, &free, true);
            assert!(!keyed.contains(&turn), "{rows} rows: {keyed:?}");
            assert_eq!(counted.survivors.get(), Some(rows - 1), "{rows} rows");
        }
        // A host's picker that takes turns and hands out nothing of the
        // row it drew: the stages place its new session, as they place its
        // request with no session value.
        #[derive(Debug)]
        struct Mute;
        impl Lifecycle for Mute {}
        impl Picker for Mute {
            fn name(&self) -> &'static str {
                "mute"
            }

            fn pick(
                &self,
                _: &RequestFacts<'_>,
                candidates: &[Candidate],
                kept: &[u32],
                _: &[f32],
                _: TieBand,
                out: &mut Targets,
            ) {
                if let Some(&row) = kept.last() {
                    out.push(target_of(&candidates[row as usize]));
                }
            }

            fn draws_first(&self) -> bool {
                true
            }

            fn takes_turns(&self) -> bool {
                true
            }

            fn draw_first(
                &self,
                _: &RequestFacts<'_>,
                _: &PoolFacts<'_>,
                candidates: &[Candidate],
                keep: &mut dyn FnMut(&Candidate) -> bool,
                kept: &mut Vec<u32>,
                _: &mut Draw,
            ) -> bool {
                kept.extend(
                    candidates
                        .iter()
                        .find(|row| keep(row))
                        .map(|row| row.position),
                );
                !kept.is_empty()
            }

            fn order_drawn(
                &self,
                _: &RequestFacts<'_>,
                _: &[Candidate],
                _: &[u32],
                _: &[f32],
                _: TieBand,
                _: &mut Targets,
            ) {
            }
        }
        let mute = || Profile {
            picker: Box::new(Mute),
            ..round_robin_profile()
        };
        assert!(draws_first(&mute()));
        let full = pool_of(DRAW_FROM as u32, |_| false);
        let last = vec![DRAW_FROM as u32 - 1];
        assert_eq!(decide(mute(), &full, &free, false).0, last);
        assert_eq!(decide(mute(), &full, &free, true).0, last);
        // `random` takes no turn. Over a pool with a row refused its draw and
        // its pick over the survivors name other rows for one seed, and its
        // new session lands where the stages alone put it: on the picker's
        // own pick over the survivors, by the request's seed.
        let random = || Profile {
            picker: Box::new(Random),
            ..round_robin_profile()
        };
        let ailing = pool_of(DRAW_FROM as u32, |row| row == 3);
        let survivors: Vec<u32> = (0..DRAW_FROM as u32).filter(|row| *row != 3).collect();
        for seed in 0..8 {
            let facts = RequestFacts {
                session_key: Identity::hashed(1),
                ..request(seed)
            };
            let placed = drawn(&session_engine_of(random(), SESSIONS), &facts, &ailing);
            let mut alone = Targets::new();
            Random.pick(&facts, &ailing, &survivors, &[], TieBand(0.0), &mut alone);
            let alone: Vec<u32> = alone.iter().map(|target| target.position).collect();
            assert_eq!(positions(&placed), alone, "seed {seed}");
        }
    }

    /// A binding of the single plan decides a request of any phase, and
    /// its round robin turns one rotation for both kinds of request at
    /// each: a request with no session value draws the pool's turn whatever
    /// its phase says, so a new session takes that turn too. A set narrowed
    /// to a side of a pair alone takes none.
    #[test]
    fn round_robin_under_delegate_turns_one_rotation_at_every_phase_of_the_single_plan() {
        let rows = DRAW_FROM + 4;
        let pool = pool_of(rows as u32, |_| false);
        for phase in PHASES {
            let engine = session_engine_of(round_robin_profile(), SESSIONS);
            let mut scratch = Scratch::new();
            let mut turn = |key: Option<u64>| {
                let facts = RequestFacts {
                    phase,
                    session_key: key.map_or(Identity::NONE, Identity::hashed),
                    ..request(0)
                };
                let (shared, cx) = (PoolFacts::default(), Constraints::default());
                let decision = engine.decide(&facts, &shared, &pool, &cx, &mut scratch);
                positions(&decision)[0] as usize
            };
            let first = turn(None);
            let keys = [Some(1), None, Some(2), Some(3), None];
            let places: Vec<usize> = keys.into_iter().map(&mut turn).collect();
            let in_turn: Vec<usize> = (1..6).map(|step| (first + step) % rows).collect();
            assert_eq!(places, in_turn, "{phase:?}");
        }
    }

    #[test]
    fn a_pick_that_stands_keeps_its_turn_behind_a_run_of_refused_rows() {
        // Row 4 is kept and the twelve rows behind it are not: the walk for
        // alternates meets the bound, and the pick stands with none.
        let pool = pool_of(32, |i| (5..17).contains(&i));
        let engine = preset("round_robin");
        let mut scratch = Scratch::new();
        let mut picks = [0u32; 32];
        let mut alone = 0;
        for _ in 0..32 * 10 {
            let decision = engine.decide(
                &request(0),
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut scratch,
            );
            let rows = positions(&decision);
            picks[rows[0] as usize] += 1;
            if rows == [4] {
                alone += 1;
                assert!(decision.trace().drew_first());
            }
        }
        assert_eq!(alone, 10, "row 4's own turns, one a turn of the pool");
        // Twelve refused turns a turn of the pool, shared by twenty rows.
        for (row, &n) in picks.iter().enumerate() {
            if (5..17).contains(&row) {
                assert_eq!(n, 0);
            } else {
                assert!((16..=16).contains(&n), "row {row}: {picks:?}");
            }
        }
    }

    #[test]
    fn the_pass_behind_a_draw_reads_the_seeds_stream_past_the_draws_values() {
        // Four rows ready of sixty-four: `random` spends its bound on the
        // pick, and where all eight draws were refused the whole pass runs
        // on the stream past those eight values, never from its start.
        let pool = pool_of(64, |i| ![5, 20, 40, 60].contains(&i));
        let (engine, twin) = (preset("random"), preset("random"));
        let mut widened = 0;
        for seed in 0..200 {
            let facts = request(seed);
            let got = drawn(&engine, &facts, &pool);
            if got.trace().drew_first() {
                continue;
            }
            widened += 1;
            let mut stream = Draw::new(facts.seed);
            for _ in 0..DRAW_BOUND {
                stream.next_u64();
            }
            let past = RequestFacts {
                seed: stream.state(),
                ..facts
            };
            let want = whole_pass(&twin, &past, &pool);
            assert_eq!(positions(&got), positions(&want), "seed {seed}");
            assert_eq!(
                got.trace().seed,
                facts.seed,
                "the trace keeps the request's seed"
            );
        }
        assert!(widened > 50, "{widened} of 200 decisions widened");
    }

    #[test]
    fn two_tied_rows_drawn_past_a_refused_one_are_ordered_on_the_stream_past_the_draws() {
        // Every second row ready and every ready row alike: the two rows
        // drawn tie, and the stream orders them from where the draws left
        // it, never from a value a draw already read.
        let pool: Vec<Candidate> = (0..64)
            .map(|i| Candidate {
                health: Health {
                    healthy: i % 2 == 0,
                    ..Health::default()
                },
                ..candidate(i, 0)
            })
            .collect();
        let engine = preset("power_of_two");
        let (mut past_a_refusal, mut second_first) = (0u32, 0u32);
        for seed in 0..400 {
            let facts = request(seed);
            let got = drawn(&engine, &facts, &pool);
            if !got.trace().drew_first() {
                continue;
            }
            // The two draws, replayed: each value of the stream is one row
            // asked, and the second draw passes over the first's row.
            let mut stream = Draw::new(facts.seed);
            let mut asked = 0u32;
            let mut draw_one = |except: Option<usize>| loop {
                let mut row = stream.below(pool.len() - usize::from(except.is_some()));
                if except.is_some_and(|taken| row >= taken) {
                    row += 1;
                }
                asked += 1;
                if pool[row].health.healthy {
                    break row;
                }
            };
            let first = draw_one(None);
            let second = draw_one(Some(first));
            if asked == 2 {
                continue;
            }
            past_a_refusal += 1;
            let past = RequestFacts {
                seed: stream.state(),
                ..facts
            };
            let kept = [first as u32, second as u32];
            let mut want = Targets::new();
            PowerOfTwo.pick(
                &past,
                &pool,
                &kept,
                &[0.5, 0.5],
                TieBand::NORMALIZED,
                &mut want,
            );
            let want: Vec<u32> = want.iter().map(|t| t.position).collect();
            assert_eq!(positions(&got), want, "seed {seed}");
            second_first += u32::from(want.first() == Some(&kept[1]));
        }
        assert!(
            second_first > 20 && past_a_refusal - second_first > 20,
            "{past_a_refusal} decisions drew past a refused row, {second_first} put the second first"
        );
    }

    #[test]
    fn the_two_draws_of_one_decision_share_its_bound_of_refused_rows() {
        use std::sync::atomic::{AtomicU32, Ordering};

        static ASKS: AtomicU32 = AtomicU32::new(0);
        static PASSES: AtomicU32 = AtomicU32::new(0);
        /// Keeps the fourth row it is asked about and no other; a whole
        /// pass keeps the pool as it stands and is counted apart.
        #[derive(Debug)]
        struct FourthAsk;
        impl Filter for FourthAsk {
            fn name(&self) -> &'static str {
                "fourth-ask"
            }

            fn row_wise(&self) -> bool {
                true
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                ASKS.fetch_add(1, Ordering::Relaxed) + 1 == 4
            }

            fn retain(
                &self,
                _: &RequestFacts<'_>,
                _: &[Candidate],
                _: &Constraints<'_>,
                _: &mut Vec<u32>,
            ) {
                PASSES.fetch_add(1, Ordering::Relaxed);
            }
        }
        let engine = Engine::new(
            vec![Profile {
                name: "two".into(),
                filters: vec![Box::new(FourthAsk)],
                scorers: Vec::new(),
                picker: Box::new(PowerOfTwo),
            }],
            vec![single_binding()],
            1,
            Needs::default(),
        );
        let pool = pool_of(64, |_| false);
        let got = drawn(&engine, &request(1), &pool);
        // Three rows refused, the fourth kept, then the second draw's rows
        // refused until the DECISION's bound: one ask more than the bound,
        // not the bound again for the second draw, and the whole pass
        // decides.
        assert_eq!(ASKS.load(Ordering::Relaxed), DRAW_BOUND + 1);
        assert_eq!(PASSES.load(Ordering::Relaxed), 1);
        assert!(!got.trace().drew_first());
        assert!(got.pick().is_some());
    }

    #[test]
    fn a_draw_asks_about_its_targets_and_its_bound_of_refused_rows_and_no_row_more() {
        use std::sync::atomic::{AtomicU32, Ordering};

        static ASKS: AtomicU32 = AtomicU32::new(0);
        /// The health filter's word, with each ask about one row counted;
        /// a whole pass is the health filter's own and is not counted.
        #[derive(Debug)]
        struct Counted;
        impl Filter for Counted {
            fn name(&self) -> &'static str {
                "counted"
            }

            fn row_wise(&self) -> bool {
                true
            }

            fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, cx: &Constraints<'_>) -> bool {
                ASKS.fetch_add(1, Ordering::Relaxed);
                Healthy.keep(facts, c, cx)
            }

            fn retain(
                &self,
                facts: &RequestFacts<'_>,
                candidates: &[Candidate],
                cx: &Constraints<'_>,
                kept: &mut Vec<u32>,
            ) {
                Healthy.retain(facts, candidates, cx, kept);
            }
        }
        let pickers: [(&str, Box<dyn Picker>); 5] = [
            ("passthrough", Box::new(Passthrough)),
            ("random", Box::new(Random)),
            ("round-robin", Box::new(RoundRobin::new())),
            ("power-of-two", Box::new(PowerOfTwo)),
            ("power-of-two-count-first", Box::new(PowerOfTwoCountFirst)),
        ];
        for (name, picker) in pickers {
            let engine = Engine::new(
                vec![Profile {
                    name: name.into(),
                    filters: vec![Box::new(Counted)],
                    scorers: Vec::new(),
                    picker,
                }],
                vec![single_binding()],
                1,
                Needs::default(),
            );
            for (fleet, refused) in REFUSED.into_iter().enumerate() {
                let pool = pool_of(64, refused);
                for seed in 0..50 {
                    ASKS.store(0, Ordering::Relaxed);
                    let decision = drawn(&engine, &request(seed), &pool);
                    let asks = ASKS.load(Ordering::Relaxed) as usize;
                    // The first fleet is ready whole: a draw there asks
                    // about the rows it hands out and about no other.
                    let most = if fleet == 0 {
                        assert!(decision.trace().drew_first(), "{name}, seed {seed}");
                        MAX_TARGETS
                    } else {
                        MAX_TARGETS + DRAW_BOUND as usize
                    };
                    assert!(
                        (1..=most).contains(&asks),
                        "{name}, fleet {fleet}, seed {seed}: {asks} asks"
                    );
                }
            }
        }
    }

    /// A host's filter around one of the engine's: the same word on
    /// every row, and no filter the build knows.
    #[derive(Debug)]
    struct Hosted(Box<dyn Filter>);
    impl Filter for Hosted {
        fn name(&self) -> &'static str {
            self.0.name()
        }

        fn soft(&self) -> bool {
            self.0.soft()
        }

        fn row_wise(&self) -> bool {
            self.0.row_wise()
        }

        fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, cx: &Constraints<'_>) -> bool {
            self.0.keep(facts, c, cx)
        }
    }
    /// An engine of one profile: `filters`, no scorer and `picker`.
    fn engine_of(filters: Vec<Box<dyn Filter>>, picker: Box<dyn Picker>) -> Engine {
        Engine::new(
            vec![Profile {
                name: "p".into(),
                filters,
                scorers: Vec::new(),
                picker,
            }],
            vec![single_binding()],
            1,
            Needs::default(),
        )
    }
    /// The pickers the cases of the folded test decide by.
    const PICKERS: [fn() -> Box<dyn Picker>; 4] = [
        || Box::new(Passthrough),
        || Box::new(Random),
        || Box::new(RoundRobin::new()),
        || Box::new(PowerOfTwo),
    ];

    /// The rows the cases of a row's test read, with the host's mask. By
    /// position modulo eight: a row each of four bits refuses and four the
    /// rows' own bits keep, of which the mask names one.
    fn tested_rows() -> (Vec<Candidate>, Vec<u32>) {
        let pool = (0..64u32)
            .map(|i| {
                let mut c = Candidate {
                    has_capacity: true,
                    has_room: true,
                    ..candidate(i, i % 5)
                };
                match i % 8 {
                    1 => c.health.healthy = false,
                    2 => c.health.overloaded = true,
                    3 => c.has_capacity = false,
                    4 => c.has_room = false,
                    _ => {}
                }
                c
            })
            .collect();
        (pool, (0..64).filter(|i| i % 8 == 7).collect())
    }

    #[test]
    fn a_profile_of_the_engines_own_filters_draws_as_one_asked_a_call_a_filter() {
        use crate::stages::filters::filters_of;
        // A request names one row tried, and the host one excluded.
        let (pool, mask) = tested_rows();
        let (mut drew, mut masked) = (0u32, 0u32);
        for set in 0..32u8 {
            for (which, picker) in PICKERS.into_iter().enumerate() {
                let folded = engine_of(filters_of(set), picker());
                let hosted = filters_of(set)
                    .into_iter()
                    .map(|filter| Box::new(Hosted(filter)) as Box<dyn Filter>)
                    .collect();
                let each = engine_of(hosted, picker());
                assert!(folded.row_tests[0].is_some(), "set {set:#07b}: folded");
                // A profile of no filter folds into the empty test whoever built it.
                assert_eq!(
                    each.row_tests[0].is_none(),
                    set != 0,
                    "set {set:#07b}: a host's"
                );
                for seed in 0..24u64 {
                    let tried = [pool[(seed as usize * 8 + 5) % pool.len()].id];
                    let excluded = [pool[(seed as usize * 8 + 6) % pool.len()].id];
                    let facts = RequestFacts {
                        tried: if seed % 2 == 0 { &tried } else { &[] },
                        ..request(seed ^ u64::from(set) << 32)
                    };
                    let cx = Constraints {
                        excluded: if seed % 3 == 0 { &excluded } else { &[] },
                        stalled: if seed % 4 < 2 { mask.as_slice() } else { &[] },
                        overload_lifted: seed % 5 == 0,
                        ..Constraints::default()
                    };
                    let decide = |engine: &Engine| {
                        engine.decide(
                            &facts,
                            &PoolFacts::default(),
                            &pool,
                            &cx,
                            &mut Scratch::new(),
                        )
                    };
                    let (got, want) = (decide(&folded), decide(&each));
                    let at = format!("set {set:#07b}, picker {which}, seed {seed}");
                    assert_eq!(got, want, "{at}");
                    if got.trace().drew_first() {
                        drew += 1;
                        masked += u32::from(set & 1 != 0 && !cx.stalled.is_empty());
                    }
                    if set & 1 != 0 {
                        assert!(
                            positions(&got).iter().all(|&row| !cx.masks(row)),
                            "{at}: a row the host's mask names"
                        );
                    }
                }
            }
        }
        // Most decisions drew, some of them under the health filter and a
        // standing veto, where the mask is asked beside the folded test.
        assert!(drew > 1_000 && masked > 200, "{drew} drew, {masked} masked");
    }

    /// The test a host asks of a holder before it names it is the draw's
    /// own. Over every set of the engine's filters, as the build folds them
    /// and as a host's own, the function and a call a filter agree on every
    /// row: under a mask, a row tried, a row excluded and the overload veto
    /// lifted.
    #[test]
    fn the_namings_test_is_the_draws_own() {
        use crate::stages::filters::filters_of;
        let (pool, mask) = tested_rows();
        let mut refused = 0u32;
        for set in 0..32u8 {
            let folded = engine_of(filters_of(set), Box::new(Random));
            let hosted = filters_of(set)
                .into_iter()
                .map(|filter| Box::new(Hosted(filter)) as Box<dyn Filter>)
                .collect();
            let each = engine_of(hosted, Box::new(Random));
            let asked = filters_of(set);
            for seed in 0..24u64 {
                let tried = [pool[(seed as usize * 8 + 5) % pool.len()].id];
                let excluded = [pool[(seed as usize * 8 + 6) % pool.len()].id];
                let facts = RequestFacts {
                    tried: if seed % 2 == 0 { &tried } else { &[] },
                    ..request(seed)
                };
                let cx = Constraints {
                    excluded: if seed % 3 == 0 { &excluded } else { &[] },
                    stalled: if seed % 4 < 2 { mask.as_slice() } else { &[] },
                    overload_lifted: seed % 5 == 0,
                    ..Constraints::default()
                };
                for (built, engine) in [("folded", &folded), ("a host's", &each)] {
                    let keeps = engine.keeps_for(facts.model, facts.phase, &facts, &cx);
                    let keeps = keeps.expect("every filter answers row by row");
                    for row in &pool {
                        let want = asked.iter().all(|filter| filter.keep(&facts, row, &cx));
                        let at = format!("set {set:#07b}, {built}, seed {seed}");
                        assert_eq!(keeps.keeps(row), want, "{at}, row {}", row.position);
                        refused += u32::from(!want);
                    }
                }
            }
        }
        assert!(refused > 10_000, "{refused} rows refused");
    }

    /// The function answers none where the pass alone can say which rows
    /// stay: under a filter that does not answer row by row, under one
    /// that holds a decision's scope, and for a profile under the key's
    /// mark, whatever the request carries.
    #[test]
    fn the_namings_test_is_none_where_the_pass_alone_can_say() {
        use crate::stages::filters::filters_of;
        /// A filter that is shown the whole view: the default word.
        #[derive(Debug)]
        struct Whole;
        impl Filter for Whole {
            fn name(&self) -> &'static str {
                "whole"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }
        }
        /// A filter that answers row by row, out of a decision's scope
        /// where it says it holds one.
        #[derive(Debug)]
        struct Scoped(bool);
        impl Lifecycle for Scoped {}
        impl DecisionScope for Scoped {
            fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

            fn close(&self) {}
        }
        impl Filter for Scoped {
            fn name(&self) -> &'static str {
                "scoped"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn row_wise(&self) -> bool {
                true
            }

            fn scope(&self) -> Option<&dyn DecisionScope> {
                self.0.then_some(self as &dyn DecisionScope)
            }
        }
        let cx = Constraints::default();
        let plain = request(7);
        let keyed = RequestFacts {
            session_key: Identity::hashed(59),
            ..plain
        };
        let tests = |engine: &Engine, facts: &RequestFacts<'_>| {
            let keeps = engine.keeps_for(facts.model, facts.phase, facts, &cx);
            keeps.is_some()
        };
        let whole = engine_of(vec![Box::new(Whole)], Box::new(Random));
        assert!(!tests(&whole, &plain), "a filter shown the whole view");
        // A scope opens at the decision, behind the naming: a filter that
        // answers row by row out of one is asked by its pass alone.
        let scoped = engine_of(vec![Box::new(Scoped(true))], Box::new(Random));
        assert!(
            !tests(&scoped, &plain),
            "a row-wise filter that holds a scope"
        );
        let unscoped = engine_of(vec![Box::new(Scoped(false))], Box::new(Random));
        assert!(tests(&unscoped, &plain), "the same filter with no scope");
        // The key's mark, set by hand: the validator sets it for the plain
        // ring's shape alone, which is named no row. A keyed request is
        // tested without the mark, and none is under it.
        let mut marked = engine_of(filters_of(0b11), Box::new(Random));
        assert!(tests(&marked, &keyed), "no mark: a key is tested");
        marked.places_by_key(0);
        assert!(!tests(&marked, &keyed), "a key under the mark");
        assert!(!tests(&marked, &plain), "no key under the mark");
    }

    #[test]
    fn a_whole_pass_by_the_folded_test_decides_as_one_asked_a_stage_at_a_time() {
        use crate::stages::filters::filters_of;
        // A row's shape: ready, or refused by one of four bits.
        fn shaped(rows: u32, shape: impl Fn(u32) -> u32) -> Vec<Candidate> {
            (0..rows)
                .map(|i| {
                    let mut c = Candidate {
                        has_capacity: true,
                        has_room: true,
                        ..candidate(i, i % 5)
                    };
                    match shape(i) {
                        1 => c.health.healthy = false,
                        2 => c.health.overloaded = true,
                        3 => c.has_capacity = false,
                        4 => c.has_room = false,
                        _ => {}
                    }
                    c
                })
                .collect()
        }
        // Pools under the draw's floor, so each decision is a whole pass:
        // rows all of one shape, one ready row among rows of one refusal,
        // and mixed rows.
        let mut pools = Vec::new();
        for rows in [1, 2, 7, DRAW_FROM as u32 - 1] {
            for shape in 0..5 {
                pools.push(shaped(rows, |_| shape));
            }
            for shape in 1..5 {
                pools.push(shaped(rows, |i| if i == rows / 2 { 0 } else { shape }));
            }
            for mix in 0..4 {
                pools.push(shaped(rows, |i| (i * 7 + mix * 3) % 6 % 5));
            }
        }
        let (mut stood, mut fell, mut shed, mut aside) = (0u32, 0u32, 0u32, 0u32);
        for set in 0..32u8 {
            for (which, picker) in PICKERS.into_iter().enumerate() {
                let folded = engine_of(filters_of(set), picker());
                let hosted = filters_of(set)
                    .into_iter()
                    .map(|filter| Box::new(Hosted(filter)) as Box<dyn Filter>)
                    .collect();
                let each = engine_of(hosted, picker());
                let filters = filters_of(set);
                for (number, pool) in pools.iter().enumerate() {
                    let middle = pool.len() / 2;
                    let ids: Vec<WorkerId> = pool.iter().map(|c| c.id).collect();
                    let places: Vec<u32> = (0..pool.len() as u32).collect();
                    for seed in 0..10u64 {
                        let facts = RequestFacts {
                            tried: match seed % 3 {
                                0 => &[],
                                1 => &ids[middle..=middle],
                                _ => &ids,
                            },
                            ..request(seed ^ u64::from(set) << 32)
                        };
                        let cx = Constraints {
                            excluded: if seed % 4 == 3 {
                                &ids[middle..=middle]
                            } else {
                                &[]
                            },
                            stalled: match seed % 5 {
                                3 => &places[middle..=middle],
                                4 => &places,
                                _ => &[],
                            },
                            overload_lifted: seed % 7 == 0,
                            ..Constraints::default()
                        };
                        let decide = |engine: &Engine| {
                            engine.decide(
                                &facts,
                                &PoolFacts::default(),
                                pool,
                                &cx,
                                &mut Scratch::new(),
                            )
                        };
                        let (got, want) = (decide(&folded), decide(&each));
                        let at =
                            format!("set {set:#07b}, picker {which}, pool {number}, seed {seed}");
                        assert_eq!(got, want, "{at}");
                        assert!(!got.trace().drew_first(), "{at}: a whole pass");
                        // Whether a row stands that every filter keeps:
                        // the folded test decides such a pool, and the
                        // stages one by one any other.
                        let stands = pool
                            .iter()
                            .any(|c| filters.iter().all(|f| f.keep(&facts, c, &cx)));
                        let picked = matches!(got, Decision::Pick(_));
                        stood += u32::from(stands);
                        fell += u32::from(!stands);
                        shed += u32::from(!picked);
                        aside += u32::from(!stands && picked);
                    }
                }
            }
        }
        // Both paths decided often: pools the folded test kept rows of,
        // pools a stage emptied and shed, and pools a soft filter or the
        // all-vetoed rule kept rows of behind an empty fold.
        assert!(
            stood > 10_000 && fell > 10_000 && shed > 5_000 && aside > 1_000,
            "{stood} stood, {fell} fell, {shed} shed, {aside} aside"
        );
    }

    /// `has-room` written bare is the host's bit and folds into the row
    /// test. With a rule it is a stage of its own: its profile folds into
    /// nothing, and a picker that draws first still draws first.
    #[test]
    fn has_room_with_a_rule_folds_into_nothing_and_still_draws_first() {
        let engine = |filter: &str| {
            let document: PolicySpec = serde_json::from_str(&format!(
                r#"{{"apiVersion":"smg.policy/v1","profiles":{{"p":{{"filter":["healthy",{filter}],"pick":"random"}}}}}}"#
            ))
            .expect("a document");
            document.validate().expect("validates")
        };
        let bare = engine(r#""has-room""#);
        assert!(bare.row_tests[0].is_some(), "the host's bit folds");
        let rule = engine(r#"{"has-room":{"used":{"candidate.in_flight":1},"capacity":8}}"#);
        assert!(rule.row_tests[0].is_none(), "the rule is asked as a stage");
        assert!(draws_first(&bare.profiles[0]) && draws_first(&rule.profiles[0]));
    }

    /// A set that is not the plan's whole pool is asked a stage at a time:
    /// an encode item's own rules stand ahead of the profile's filters, and
    /// the fold holds none of them.
    #[test]
    fn a_whole_pass_over_an_encode_items_set_is_asked_a_stage_at_a_time() {
        use crate::stages::filters::filters_of;
        let engine = engine_of(filters_of(0b11111), Box::new(Passthrough));
        assert!(engine.row_tests[0].is_some(), "the profile folds");
        // Two rows every filter keeps. The item's rule refuses the first.
        let pool: Vec<Candidate> = (0..2u32)
            .map(|i| Candidate {
                has_capacity: true,
                has_room: true,
                ec_compatible: i == 1,
                ..candidate(i, 0)
            })
            .collect();
        let facts = request(7);
        let (entry, index, profile) = engine
            .resolve(facts.model, Phase::Single)
            .expect("a binding");
        let item = Set {
            candidates: &pool,
            class: None,
            partnered: None,
            fold: None,
            encode: true,
            vetoed: entry.vetoed,
        };
        let whole = |set: Set<'_>| {
            let (shown, cx) = (PoolFacts::default(), Constraints::default());
            run_whole(
                &engine,
                index,
                &facts,
                &shown,
                set,
                &cx,
                &mut Buffers::default(),
            )
        };
        let (shown, cx) = (PoolFacts::default(), Constraints::default());
        let staged = run_single(
            profile,
            index,
            &facts,
            &shown,
            item,
            &cx,
            &mut Buffers::default(),
        );
        assert!(matches!(staged, Decision::Pick(_)), "the second row stands");
        assert_eq!(whole(item), staged, "the item's set");
        // The rule decides this pool: the plan's whole pool hands the first.
        let plain = Set {
            encode: false,
            ..item
        };
        assert_ne!(whole(plain), staged, "the whole pool");

        // The same under the all-vetoed rule: both rows over their
        // thresholds, so the rule's lifted pass decides. The whole pass
        // hands the rule its fold, and the rule asks it about the whole
        // pool alone: an item's set is lifted a stage at a time, and its
        // own rule still refuses the first row.
        let over: Vec<Candidate> = pool
            .iter()
            .map(|row| Candidate {
                health: Health {
                    overloaded: true,
                    ..row.health
                },
                ..*row
            })
            .collect();
        let hosted = filters_of(0b11111)
            .into_iter()
            .map(|filter| Box::new(Hosted(filter)) as Box<dyn Filter>)
            .collect();
        let each = engine_of(hosted, Box::new(Passthrough));
        let lifted = |engine: &Engine, encode: bool| {
            let (entry, index, _) = engine
                .resolve(facts.model, Phase::Single)
                .expect("a binding");
            let set = Set {
                candidates: &over,
                class: None,
                partnered: None,
                fold: None,
                encode,
                vetoed: entry.vetoed,
            };
            let (shown, cx) = (PoolFacts::default(), Constraints::default());
            run_whole(
                engine,
                index,
                &facts,
                &shown,
                set,
                &cx,
                &mut Buffers::default(),
            )
        };
        let item = lifted(&engine, true);
        assert!(
            matches!(&item, Decision::Pick(pick) if pick.trace.lifted == LiftedVeto::Overload),
            "the rule steers the item"
        );
        assert_eq!(item, lifted(&each, true), "the item's set, lifted");
        assert_eq!(
            lifted(&engine, false),
            lifted(&each, false),
            "the pool, lifted"
        );
        assert_ne!(item, lifted(&engine, false), "the item's rule decides");
    }

    /// A folded engine beside a staged one over pools of every kind, each
    /// on a scratch kept from decision to decision: every decision is
    /// equal, and a whole pass's survivors are the stages' by a model of
    /// them. A pass that left a list behind for the next decision to read
    /// would part the two, so the scratches change hands every seventh
    /// decision. The all-vetoed rule's pick is the fold's often.
    #[test]
    fn a_folded_engine_decides_as_a_staged_one_on_scratches_that_are_kept() {
        use crate::stages::filters::filters_of;
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        let (mut whole, mut drawn, mut emptied) = (0u32, 0u32, 0u32);
        let (mut steered, mut aside) = (0u32, 0u32);
        for set in 0..32u8 {
            for turn in 0..3usize {
                for (which, picker) in PICKERS.into_iter().enumerate() {
                    // The profile's filters in an order of the walk's own.
                    let order = |mut filters: Vec<Box<dyn Filter>>| {
                        if !filters.is_empty() {
                            let by = turn % filters.len();
                            filters.rotate_left(by);
                            if turn == 2 {
                                filters.reverse();
                            }
                        }
                        filters
                    };
                    let folded = engine_of(order(filters_of(set)), picker());
                    let hosted: Vec<Box<dyn Filter>> = order(filters_of(set))
                        .into_iter()
                        .map(|filter| Box::new(Hosted(filter)) as Box<dyn Filter>)
                        .collect();
                    let each = engine_of(hosted, picker());
                    let filters = order(filters_of(set));
                    // One scratch an engine, kept from decision to decision.
                    let (mut ours, mut theirs) = (Scratch::new(), Scratch::new());
                    for step in 0..160u32 {
                        let rows = 1 + u32::try_from(next(24)).expect("a small pool");
                        let pool: Vec<Candidate> = (0..rows)
                            .map(|i| {
                                let held = u32::try_from(next(5)).expect("a small count");
                                let mut c = Candidate {
                                    has_capacity: next(4) != 0,
                                    has_room: next(3) != 0,
                                    ..candidate(i, held)
                                };
                                c.health.healthy = next(6) != 0;
                                c.health.overloaded = next(3) == 0;
                                c.health.stalled = next(9) == 0;
                                c.health.draining = next(11) == 0;
                                c.health.breaker_open = next(13) == 0;
                                c
                            })
                            .collect();
                        let (mut tried, mut excluded) = (Vec::new(), Vec::new());
                        let mut stalled = Vec::new();
                        let (t, e, m) = (next(3), next(4), next(4));
                        for c in &pool {
                            if t == 0 && next(3) == 0 {
                                tried.push(c.id);
                            }
                            if e == 0 && next(4) == 0 {
                                excluded.push(c.id);
                            }
                            if m == 0 && next(3) == 0 {
                                stalled.push(c.position);
                            }
                        }
                        let facts = RequestFacts {
                            tried: &tried,
                            ..request(next(1 << 20))
                        };
                        let cx = Constraints {
                            excluded: &excluded,
                            stalled: &stalled,
                            overload_lifted: next(6) == 0,
                            ..Constraints::default()
                        };
                        let shown = PoolFacts::default();
                        let got = folded.decide(&facts, &shown, &pool, &cx, &mut ours);
                        let want = each.decide(&facts, &shown, &pool, &cx, &mut theirs);
                        let at =
                            format!("set {set:#07b}, turn {turn}, picker {which}, step {step}");
                        assert_eq!(got, want, "{at}");
                        if step % 7 == 6 {
                            std::mem::swap(&mut ours, &mut theirs);
                        }
                        // The model of the staged pass: the hard filters,
                        // then each soft filter while it leaves a row.
                        let keeps = |f: &dyn Filter, i: u32| f.keep(&facts, &pool[i as usize], &cx);
                        let mut stand: Vec<u32> = (0..rows).collect();
                        for f in filters.iter().filter(|f| !f.soft()) {
                            stand.retain(|&i| keeps(f.as_ref(), i));
                        }
                        if stand.is_empty() {
                            // Equal to the staged twin, not modelled: a shed,
                            // or the rule's pick behind its lifted pass.
                            emptied += 1;
                            steered += u32::from(got.trace().lifted == LiftedVeto::Overload);
                            continue;
                        }
                        let mut stepped = false;
                        for f in filters.iter().filter(|f| f.soft()) {
                            let left: Vec<u32> = stand
                                .iter()
                                .copied()
                                .filter(|&i| keeps(f.as_ref(), i))
                                .collect();
                            if left.is_empty() {
                                stepped = true;
                            } else {
                                stand = left;
                            }
                        }
                        aside += u32::from(stepped);
                        let Decision::Pick(pick) = &got else {
                            panic!("{at}: no pick over rows that stand: {got:?}")
                        };
                        let named: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
                        assert!(
                            named.iter().all(|row| stand.contains(row)),
                            "{at}: {named:?} of {stand:?}"
                        );
                        if got.trace().drew_first() {
                            drawn += 1;
                        } else {
                            whole += 1;
                            let counted = got.trace().survivors.get().map(|n| n as usize);
                            assert_eq!(counted, Some(stand.len()), "{at}: the survivors counted");
                        }
                    }
                }
            }
        }
        // The walk reached every kind of pass: whole passes and draws over
        // rows that stand, pools the hard filters emptied, the rule's
        // picks among them, and passes where a soft filter stepped aside.
        assert!(
            whole > 10_000 && drawn > 10_000 && emptied > 500 && steered > 500 && aside > 500,
            "{whole} whole, {drawn} drawn, {emptied} emptied, {steered} steered, {aside} aside"
        );
    }

    /// A profile may hold as many scorers as a place can name: the walk
    /// that lists the stages that keep reports ends with the list.
    #[test]
    fn an_engine_of_as_many_scorers_as_a_place_names_is_built() {
        let scorers = (0..u16::MAX)
            .map(|_| WeightedScorer {
                scorer: Box::new(LeastRequests),
                weight: 1.0,
            })
            .collect();
        let engine = Engine::new(
            vec![Profile {
                name: "wide".into(),
                filters: Vec::new(),
                scorers,
                picker: Box::new(Best),
            }],
            vec![single_binding()],
            1,
            Needs::default(),
        );
        assert!(!engine.keeps_loads(), "no stage of it keeps a report");
    }

    #[test]
    fn a_draw_asks_the_folded_test_of_its_own_profile() {
        // Two profiles of one engine, each folded: the first keeps a row
        // over its ceiling and the second refuses it. A draw under either
        // asks that profile's own test.
        let profile = |name: &str, filters: Vec<Box<dyn Filter>>| Profile {
            name: name.into(),
            filters,
            scorers: Vec::new(),
            picker: Box::new(Random),
        };
        let binding = |model: &str, profile: u16| Binding {
            model: model.into(),
            profile,
            ..single_binding()
        };
        let engine = Engine::new(
            vec![
                profile("lenient", vec![Box::new(Healthy)]),
                profile("strict", vec![Box::new(Healthy), Box::new(NotOverloaded)]),
            ],
            vec![binding("lenient", 0), binding("strict", 1)],
            1,
            Needs::default(),
        );
        assert!(engine.row_tests[0].is_some() && engine.row_tests[1].is_some());
        assert_ne!(engine.row_tests[0], engine.row_tests[1]);
        // Every second row is over its ceiling.
        let pool: Vec<Candidate> = (0..64u32)
            .map(|i| {
                let mut c = candidate(i, 0);
                c.health.overloaded = i % 2 == 0;
                c
            })
            .collect();
        let (mut drew, mut over) = ([0u32; 2], [0usize; 2]);
        for (at, model) in ["lenient", "strict"].into_iter().enumerate() {
            for seed in 0..64u64 {
                let facts = RequestFacts {
                    model,
                    ..request(seed)
                };
                let decision = engine.decide(
                    &facts,
                    &PoolFacts::default(),
                    &pool,
                    &Constraints::default(),
                    &mut Scratch::new(),
                );
                drew[at] += u32::from(decision.trace().drew_first());
                let handed = positions(&decision);
                over[at] += handed.iter().filter(|&&row| row % 2 == 0).count();
            }
        }
        assert!(drew[0] == 64 && drew[1] > 48, "{drew:?} drew");
        // The first profile hands such rows out and the second none.
        assert!(over[0] > 0, "the lenient profile handed none");
        assert_eq!(
            over[1], 0,
            "the strict profile handed a row over its ceiling"
        );
    }

    /// A draw asks the folded test of the profile its binding names: two
    /// profiles are bound crosswise, so a binding's place and its profile's
    /// index part. The lenient profile hands out overloaded rows and the
    /// strict one hands out none.
    #[test]
    fn a_draw_asks_the_test_of_the_profile_its_binding_names() {
        let profile = |name: &str, filters: Vec<Box<dyn Filter>>| Profile {
            name: name.into(),
            filters,
            scorers: Vec::new(),
            picker: Box::new(Random),
        };
        let binding = |model: &str, profile: u16| Binding {
            model: model.into(),
            profile,
            ..single_binding()
        };
        let engine = Engine::new(
            vec![
                profile("lenient", vec![Box::new(Healthy)]),
                profile("strict", vec![Box::new(Healthy), Box::new(NotOverloaded)]),
            ],
            vec![binding("strict", 1), binding("lenient", 0)],
            1,
            Needs::default(),
        );
        // Every other row is overloaded.
        let pool: Vec<Candidate> = (0..64u32)
            .map(|i| {
                let mut row = candidate(i, 0);
                row.health.overloaded = i % 2 == 0;
                row
            })
            .collect();
        let (mut drew, mut overloaded) = ([0u32; 2], [0usize; 2]);
        for (at, model) in ["lenient", "strict"].into_iter().enumerate() {
            for seed in 0..64u64 {
                let facts = RequestFacts {
                    model,
                    ..request(seed)
                };
                let decision = drawn(&engine, &facts, &pool);
                drew[at] += u32::from(decision.trace().drew_first());
                overloaded[at] += positions(&decision)
                    .iter()
                    .filter(|&&row| row % 2 == 0)
                    .count();
            }
        }
        assert!(drew[0] == 64 && drew[1] > 48, "{drew:?} decisions drew");
        assert!(
            overloaded[0] > 0 && overloaded[1] == 0,
            "overloaded rows handed out: {overloaded:?}"
        );
    }

    #[test]
    fn a_pool_under_a_new_membership_epoch_turns_its_own_rotation() {
        let engine = preset("round_robin");
        let ready = pool_of(32, |_| false);
        let mut scratch = Scratch::new();
        let mut turn = |epoch: u64| {
            let pool = PoolFacts {
                membership_epoch: epoch,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            let decision = engine.decide(&request(0), &pool, &ready, &cx, &mut scratch);
            positions(&decision)[0]
        };
        let old = turn(1);
        for step in 1..=5 {
            assert_eq!(turn(1), (old + step) % 32, "step {step}");
        }
        // A membership write moves the epoch: another list of rows stands
        // behind the pool's key, and it turns a rotation of its own.
        let new = turn(2);
        for step in 1..=3 {
            assert_eq!(turn(2), (new + step) % 32, "step {step}");
        }
        // The old epoch's rotation stands where its last decision left it.
        assert_eq!(turn(1), (old + 6) % 32);
    }

    #[test]
    fn two_lists_of_one_pool_each_turn_their_own_rotation() {
        // Two lists behind one model and phase, asked through one engine,
        // each pair apart by one word of the key: two callers' lists of one
        // pool that the host names apart, as where a pool holds host-local
        // rows; two lists of two sizes; a pool's list before a membership
        // write and after it.
        let engine = preset("round_robin");
        let mut scratch = Scratch::new();
        for (sizes, reaches, epochs) in [
            ([16u32, 16], [1, 2], [1, 1]),
            ([16, 32], [0, 0], [1, 1]),
            ([16, 16], [0, 0], [2, 3]),
        ] {
            let lists = sizes.map(|rows| pool_of(rows, |_| false));
            let mut drawn = sizes.map(|rows| vec![0u32; rows as usize]);
            let mut ask = |list: usize, drawn: &mut [Vec<u32>; 2]| {
                let pool = PoolFacts {
                    membership_epoch: epochs[list],
                    reach: reaches[list],
                    ..PoolFacts::default()
                };
                let cx = Constraints::default();
                let decision = engine.decide(&request(0), &pool, &lists[list], &cx, &mut scratch);
                assert!(decision.trace().drew_first());
                drawn[list][positions(&decision)[0] as usize] += 1;
            };
            // Their asks reach one engine in turn: one rotation turned by
            // both would draw every other row of each list.
            for turn in 0..64 {
                ask(turn % 2, &mut drawn);
            }
            for (rows, drawn) in sizes.iter().zip(&drawn) {
                assert!(
                    drawn.iter().all(|&n| n == 32 / rows),
                    "{sizes:?}: {drawn:?}"
                );
            }
            // And as a coin falls: each list's turn is still its own, so no
            // two of its rows are more than one decision apart.
            let mut coin = Draw::new(11);
            for _ in 0..320 {
                ask((coin.next_u64() & 1) as usize, &mut drawn);
            }
            for drawn in &drawn {
                let least = drawn.iter().copied().min().unwrap_or(0);
                let most = drawn.iter().copied().max().unwrap_or(0);
                assert!(most - least <= 1, "{sizes:?}: {drawn:?}");
            }
        }
    }

    #[test]
    fn an_encode_item_never_draws() {
        // The `epd` plan's compatibility rule is the whole pass's own: under
        // an encode profile whose picker draws first, an encode item over a
        // pool of the draw's size is still read whole, and the incompatible
        // encoder that leads the pool is handed to nobody.
        let engine = PolicySpec::epd_presets("passthrough", "least_load", "least_load")
            .expect("the presets")
            .validate()
            .expect("the spec validates");
        let pool: Vec<Candidate> = (0..16).map(|i| encode_candidate(i, i != 0, true)).collect();
        let decision = drawn(&engine, &encode_facts(0), &pool);
        assert_eq!(positions(&decision), [1, 2, 3, 4]);
        assert!(!decision.trace().drew_first());
        assert_eq!(decision.trace().survivors.get(), Some(15));
    }

    #[test]
    fn a_drawn_decision_replays_from_its_seed_and_a_widened_one_too() {
        for name in ["random", "power_of_two"] {
            let engine = preset(name);
            for (fleet, refused) in REFUSED.into_iter().enumerate() {
                let pool = pool_of(64, refused);
                for seed in 0..200 {
                    let facts = request(seed);
                    let first = drawn(&engine, &facts, &pool);
                    assert_eq!(
                        first,
                        drawn(&engine, &facts, &pool),
                        "{name}, fleet {fleet}"
                    );
                }
            }
        }
    }

    /// `release` frees a caught payload whole. Where a payload's drop
    /// panics, the payload of that panic is freed the same way, so a storm
    /// of such faults leaves none alive. A chain of drops that each raise
    /// the next is freed down to the depth, and the one payload past it is
    /// forgotten.
    #[test]
    fn a_storm_of_payloads_whose_drop_panics_leaves_none_alive() {
        use std::{
            panic::resume_unwind,
            sync::atomic::{AtomicUsize, Ordering},
        };

        static ALIVE: AtomicUsize = AtomicUsize::new(0);

        /// A payload counted while it is alive, whose drop raises the next
        /// of its chain `left` times. No hook runs for a payload raised by
        /// `resume_unwind`, so a storm writes nothing.
        struct Chained {
            left: usize,
        }

        impl Chained {
            fn boxed(left: usize) -> Box<Self> {
                ALIVE.fetch_add(1, Ordering::Relaxed);
                Box::new(Self { left })
            }
        }

        impl Drop for Chained {
            fn drop(&mut self) {
                ALIVE.fetch_sub(1, Ordering::Relaxed);
                if self.left > 0 {
                    resume_unwind(Self::boxed(self.left - 1));
                }
            }
        }

        for _ in 0..100_000 {
            release(Chained::boxed(1));
        }
        assert_eq!(
            ALIVE.load(Ordering::Relaxed),
            0,
            "payloads alive after a storm of drops that panic"
        );
        // The two chains by their numbers, never by the constant: a depth
        // one off either way fails one of them, and a payload past the
        // depth that is dropped where it should be forgotten fails the
        // second at once, since the last of a chain raises nothing.
        release(Chained::boxed(3));
        assert_eq!(
            ALIVE.load(Ordering::Relaxed),
            0,
            "a chain of the depth's four payloads is freed whole"
        );
        release(Chained::boxed(4));
        assert_eq!(
            ALIVE.load(Ordering::Relaxed),
            1,
            "a chain one past the depth ends there, with one payload forgotten"
        );
    }

    // ---- a draw the rows a host named join

    fn holding(row: u32, overlap_blocks: u32) -> crate::engine::AffinityHolder {
        crate::engine::AffinityHolder {
            row,
            overlap_blocks,
            predicted_blocks: 0,
        }
    }

    /// A request of sixteen blocks whose host names `holders`.
    fn naming(seed: u64, holders: &[crate::engine::AffinityHolder]) -> RequestFacts<'_> {
        static HASHES: [u64; 16] = [0; 16];
        RequestFacts {
            block_hashes: &HASHES,
            prompt_tokens: 256,
            named: Some(AffinityRows {
                holders,
                ..AffinityRows::default()
            }),
            ..request(seed)
        }
    }

    /// Behind the draw of a `best` that takes a sample, the rows a host
    /// named join the rows drawn: the deepest named holder leads whether the
    /// draw reached it or not, one a filter refuses leaves the lead to the
    /// next, and a named row the draw reached too is handed out once.
    #[test]
    fn the_rows_a_host_named_join_the_rows_a_sampled_best_drew() {
        let engine = preset("cache_aware");
        assert_eq!(engine.needs().sample, 16);
        let holders = [holding(20, 8), holding(40, 12)];
        view::LOOKS_BEHIND.with(|looks| looks.set(0));
        for (down, lead) in [(64u32, 40u32), (40, 20)] {
            let pool = pool_of(64, |i| i == down);
            for seed in 0..200u64 {
                let got = drawn(&engine, &naming(seed, &holders), &pool);
                assert!(got.trace().drew_first(), "seed {seed}: the decision drew");
                let rows = positions(&got);
                assert_eq!(rows.first(), Some(&lead), "seed {seed}, row {down} down");
                let mut once = rows.clone();
                once.sort_unstable();
                once.dedup();
                assert_eq!(once.len(), rows.len(), "seed {seed}: {rows:?}, a row twice");
            }
        }
        // Behind the draw the stages search the rows named: no walk looked
        // behind its cursor in any of the decisions.
        assert_eq!(view::LOOKS_BEHIND.with(std::cell::Cell::get), 0);
    }

    /// The resident-work stage answers row by row, as the least-load stage
    /// does, so a `best` that takes a sample draws under it: for a request
    /// whose host names rows, over a pool of more than twice the sample and
    /// of sixteen rows or more, the decision draws, and the stage costs the
    /// rows drawn and the rows named among themselves. The row of least work
    /// among the rows handed leads, so the pool's least row leads where the
    /// sample reached it and not otherwise. A request that names no row
    /// reads the pool whole, and there that row leads.
    #[test]
    fn a_sampled_best_draws_under_the_resident_work_stage() {
        let engine = Engine::new(
            vec![Profile {
                name: "sampled".into(),
                filters: vec![Box::new(Healthy)],
                scorers: vec![WeightedScorer {
                    scorer: Box::new(scorers::ResidentWork::new(0)),
                    weight: 1.0,
                }],
                picker: Box::new(pickers::SampledBest::new(4)),
            }],
            vec![Binding {
                model: "m".into(),
                profile: 0,
                ..single_binding()
            }],
            1,
            Needs::default(),
        );
        // Every row holds one request and reports its resident tokens, fewer
        // the further its place: no two rows hold the same work, and the
        // last row holds the least of the pool.
        let pool: Vec<Candidate> = pool_of(64, |_| false)
            .into_iter()
            .map(|row| Candidate {
                in_flight: 1,
                load: Some(view::LoadRow {
                    running: 1,
                    resident_tokens: 1_000 * (64 - row.position),
                    resident_counted: true,
                    ..view::LoadRow::default()
                }),
                ..row
            })
            .collect();
        let holders = [holding(20, 8)];
        let mut led = 0;
        for seed in 0..64u64 {
            let got = drawn(&engine, &naming(seed, &holders), &pool);
            assert!(got.trace().drew_first(), "seed {seed}: the decision drew");
            let handed = positions(&got);
            assert_eq!(
                handed.first(),
                handed.iter().max(),
                "seed {seed}: the furthest of the rows handed leads"
            );
            led += u32::from(handed.first() == Some(&63));
        }
        assert!(
            led < 32,
            "the pool's least row led {led} of 64 decisions over a sample of four"
        );
        let whole = drawn(&engine, &request(1), &pool);
        assert!(!whole.trace().drew_first(), "no row named: the pool whole");
        assert_eq!(positions(&whole).first(), Some(&63));
    }

    /// A host caps each list of rows it names at the picker's sample. Past
    /// it a decision reads dearer and is never wrong, so a debug build
    /// alone stops, by the assertion's own words.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "at or under the profile's sample")]
    fn rows_named_past_the_sample_stop_a_debug_build_where_the_picker_draws() {
        let engine = preset("cache_aware");
        let holders: Vec<crate::engine::AffinityHolder> =
            (0..17).map(|row| holding(row, 12)).collect();
        let decision = drawn(&engine, &naming(1, &holders), &pool_of(64, |_| false));
        panic!("seventeen rows named answered {decision:?}");
    }

    /// A pool of at most twice the sample is read whole, rows named or not,
    /// and so is a larger pool whose host names none: the draw is for the
    /// pools a host names rows in.
    #[test]
    fn a_sampled_best_reads_a_small_pool_and_a_pool_with_no_rows_named_whole() {
        let (engine, twin) = (preset("cache_aware"), preset("cache_aware"));
        let holders = [holding(3, 12)];
        for seed in 0..40u64 {
            // Twice the sample, rows named: the whole pass, row for row.
            let pool = pool_of(32, |i| i % 17 == 5);
            let facts = naming(seed, &holders);
            let got = drawn(&engine, &facts, &pool);
            assert!(!got.trace().drew_first(), "seed {seed}");
            assert_eq!(got, whole_pass(&twin, &facts, &pool), "seed {seed}");
            // One row more: the decision draws.
            let pool = pool_of(33, |i| i % 17 == 5);
            let got = drawn(&engine, &facts, &pool);
            assert!(got.trace().drew_first(), "seed {seed}");
            // No rows named: the whole pass, whatever the pool's size.
            let pool = pool_of(257, |i| i % 17 == 5);
            let facts = request(seed);
            let got = drawn(&engine, &facts, &pool);
            assert!(!got.trace().drew_first(), "seed {seed}");
            assert_eq!(got, whole_pass(&twin, &facts, &pool), "seed {seed}");
        }
    }

    /// The joined mark stands only beside the draw's: under a filter that is
    /// not row-wise a `best` that takes a sample reads its pool whole,
    /// whatever rows the host names.
    #[test]
    fn a_joined_picker_under_a_stage_that_is_not_row_wise_never_draws() {
        /// Keeps every row, and its word on one row is not its whole word.
        #[derive(Debug)]
        struct Whole;
        impl Filter for Whole {
            fn name(&self) -> &'static str {
                "whole"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }
        }
        let engine = Engine::new(
            vec![Profile {
                name: "whole".into(),
                filters: vec![Box::new(Healthy), Box::new(Whole)],
                scorers: Vec::new(),
                picker: Box::new(pickers::SampledBest::new(16)),
            }],
            vec![single_binding()],
            1,
            Needs::default(),
        );
        // It says no sample: a stage that is not row-wise reads the pool whole.
        assert_eq!(engine.needs().sample, 0);
        let holders = [holding(3, 12)];
        let pool = pool_of(64, |_| false);
        for seed in 0..20u64 {
            let got = drawn(&engine, &naming(seed, &holders), &pool);
            assert!(!got.trace().drew_first(), "seed {seed}");
            assert_eq!(got.trace().survivors.get(), Some(64), "seed {seed}");
        }
    }

    /// The mean the spill gate reads behind a draw is the drawn rows': four
    /// named holders with seven in flight spill against rows drawn at two,
    /// where the mean of the rows handed, the holders among them, would
    /// keep each of them in the lead.
    #[test]
    fn the_rows_named_do_not_move_the_mean_they_are_judged_against() {
        let engine = preset("cache_aware");
        let holders = [
            holding(8, 12),
            holding(24, 12),
            holding(40, 12),
            holding(56, 12),
        ];
        let pool: Vec<Candidate> = (0..64u32)
            .map(|i| candidate(i, if i % 16 == 8 { 7 } else { 2 }))
            .collect();
        let mut led = 0u32;
        for seed in 0..200u64 {
            let got = drawn(&engine, &naming(seed, &holders), &pool);
            assert!(got.trace().drew_first(), "seed {seed}");
            led += u32::from(positions(&got).first().is_some_and(|row| row % 16 == 8));
        }
        // A draw that takes all four holders reads a mean they pass: rare.
        assert!(led < 20, "{led} of 200 decisions were led by a busy holder");
    }

    /// The mean the join hands on is of the drawn rows' counts as the
    /// request's facts read them: the lent count where a table is lent.
    #[test]
    fn the_join_hands_on_the_mean_of_the_drawn_rows_lent_counts() {
        let mut pool: Vec<Candidate> = (0..)
            .zip([6, 2, 9, 9])
            .map(|(position, in_flight)| Candidate {
                position,
                in_flight,
                ..Candidate::default()
            })
            .collect();
        let lent = Lent::take(&mut pool);
        let facts = lent.beside(&RequestFacts::default());
        let mut kept = vec![0, 1];
        let mean = join_named(
            &facts,
            &AffinityRows::default(),
            &pool,
            &mut |_: &Candidate| true,
            &mut kept,
        );
        assert_eq!(mean, 4.0);
    }

    /// A named row past the pool names no candidate: the join passes over
    /// it in every build and asks no filter about it. A decision of a debug
    /// build stops before the join ([`debug_assert_named`]).
    #[test]
    fn the_join_passes_over_a_row_named_past_the_pool() {
        let pool = pool_of(4, |i| i == 2);
        let holders = [holding(1, 12), holding(2, 12), holding(9, 12)];
        let named = AffinityRows {
            holders: &holders,
            warming: &[3, 4, u32::MAX],
            ..AffinityRows::default()
        };
        let mut asked = Vec::new();
        let mut keep = |row: &Candidate| {
            asked.push(row.position);
            row.health.healthy
        };
        let mut kept = vec![0];
        join_named(
            &RequestFacts::default(),
            &named,
            &pool,
            &mut keep,
            &mut kept,
        );
        assert_eq!(kept, [0, 1, 3], "the pool's rows a filter keeps join");
        assert_eq!(
            asked,
            [1, 2, 3],
            "a filter is asked of the pool's rows alone"
        );
    }

    /// Two profiles whose `best` takes a sample of four and of sixteen, for
    /// the models `narrow` and `wide`.
    fn two_samples() -> Engine {
        let bound = |model: &str, profile| Binding {
            model: model.into(),
            profile,
            ..single_binding()
        };
        sampled_under(vec![bound("narrow", 0), bound("wide", 1)])
    }

    /// Two profiles that draw, under `bindings`: a sample of four (profile
    /// 0) and a sample of sixteen (profile 1).
    fn sampled_under(bindings: Vec<Binding>) -> Engine {
        let sampled = |name: &str, sample| Profile {
            name: name.into(),
            filters: vec![Box::new(Healthy)],
            scorers: Vec::new(),
            picker: Box::new(pickers::SampledBest::new(sample)),
        };
        Engine::new(
            vec![sampled("narrow", 4), sampled("wide", 16)],
            bindings,
            1,
            Needs::default(),
        )
    }

    /// A host caps the rows it names by the sample of the profile that
    /// decides, which [`Engine::sample_for`] says; the engine's summary
    /// holds the largest sample of every profile. The cache-aware preset
    /// says sixteen for a model's single decision.
    #[test]
    fn a_host_reads_the_cap_of_the_rows_it_names_from_the_profile_that_decides() {
        let engine = two_samples();
        assert_eq!(engine.needs().sample, 16);
        let pool = pool_of(64, |_| false);
        let holders: Vec<crate::engine::AffinityHolder> =
            (0..16).map(|row| holding(row, 12)).collect();
        for (model, sample) in [("narrow", 4usize), ("wide", 16)] {
            let cap = engine.sample_for(model, Phase::Single) as usize;
            assert_eq!(cap, sample, "{model}");
            let facts = RequestFacts {
                model,
                ..naming(1, &holders[..cap])
            };
            let got = drawn(&engine, &facts, &pool);
            assert!(got.trace().drew_first(), "{model}: capped so, it draws");
        }
        let cache_aware = preset("cache_aware");
        assert_eq!(cache_aware.sample_for("m", Phase::Single), 16);
    }

    /// Capped by the engine's summary instead, the rows a host names under
    /// the narrower profile stand past its picker's sample: a debug build
    /// stops, by the assertion's own words.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "at or under the profile's sample")]
    fn rows_capped_by_the_engines_summary_stop_a_debug_build_under_a_narrower_profile() {
        let engine = two_samples();
        let holders: Vec<crate::engine::AffinityHolder> = (0..engine.needs().sample)
            .map(|row| holding(row, 12))
            .collect();
        let facts = RequestFacts {
            model: "narrow",
            ..naming(1, &holders)
        };
        let decision = drawn(&engine, &facts, &pool_of(64, |_| false));
        panic!("sixteen rows named under a sample of four answered {decision:?}");
    }

    /// The mean in flight is the engine's to set behind a draw. A host that
    /// writes one stops a debug build, by the assertion's own words: on a
    /// pool read whole the stages would read its number for their rows'.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "beside an empty column and no mean in flight")]
    fn a_mean_in_flight_a_host_wrote_stops_a_debug_build() {
        let engine = preset("cache_aware");
        let holders = [holding(3, 12)];
        let facts = RequestFacts {
            named: Some(AffinityRows {
                holders: &holders,
                mean_in_flight: Some(2.0),
                ..AffinityRows::default()
            }),
            ..request(1)
        };
        let decision = drawn(&engine, &facts, &pool_of(8, |_| false));
        panic!("a mean the host wrote answered {decision:?}");
    }

    /// A sampled `best` counts no refused rows: under a filter that refuses
    /// half of the pool every decision still draws, and none of its targets
    /// is a refused row. A pool refused whole is the whole pass's to shed.
    #[test]
    fn a_sampled_draw_stands_under_a_filter_that_refuses_half_of_the_pool() {
        let engine = preset("cache_aware");
        let holders = [holding(5, 12)];
        let pool = pool_of(64, |i| i % 2 == 0);
        for seed in 0..500u64 {
            let got = drawn(&engine, &naming(seed, &holders), &pool);
            assert!(got.trace().drew_first(), "seed {seed}");
            let rows = positions(&got);
            assert!(rows.iter().all(|row| row % 2 == 1), "seed {seed}: {rows:?}");
        }
        let got = drawn(&engine, &naming(1, &holders), &pool_of(64, |_| true));
        assert!(!got.trace().drew_first());
        assert!(got.pick().is_none());
    }

    /// A draw's read of the best scores is the whole pass's, slot for slot:
    /// ties, scores under zero, scores without an order and every count of
    /// rows around the count kept. The keys' own read is held to the pass's
    /// at every count, under the count a draw reads by keys too.
    #[test]
    fn the_best_scores_of_a_draw_are_the_best_scores_of_a_pass() {
        fn bits<const K: usize>(
            read: impl Fn(&[f32], &mut [(usize, f32); K]) -> usize,
            scores: &[f32],
        ) -> ([(usize, u32); K], usize) {
            let mut top = [(usize::MAX, f32::NEG_INFINITY); K];
            let len = read(scores, &mut top);
            (top.map(|(i, score)| (i, score.to_bits())), len)
        }
        fn held<const K: usize>(scores: &[f32]) {
            assert_eq!(
                bits(top_drawn::<K>, scores),
                bits(top_pass::<K>, scores),
                "{scores:?}"
            );
            if !scores.iter().any(|score| score.is_nan()) {
                assert_eq!(
                    bits(top_keyed::<K>, scores),
                    bits(top_pass::<K>, scores),
                    "{scores:?}"
                );
            }
        }
        let values = [
            0.0f32,
            -0.0,
            0.25,
            0.5,
            0.5,
            0.75,
            1.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
        ];
        // Scores under zero alone, a tiny one among them: their order is the
        // keys' turn of every bit under a set sign.
        let under = [
            -0.0f32,
            -1.0e-40,
            -0.25,
            -0.25,
            -1.0,
            -3.5,
            f32::MIN,
            f32::NEG_INFINITY,
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for round in 0..4_000usize {
            let scores: Vec<f32> = (0..round % 41)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let pick = (state >> 33) as usize;
                    match round % 4 {
                        // Every fourth list holds scores under zero alone.
                        3 => under[pick % under.len()],
                        // Every second list holds no score without an order.
                        _ => values[pick % (values.len() - round % 2)],
                    }
                })
                .collect();
            held::<1>(&scores);
            held::<TRACE_TOP>(&scores);
            held::<5>(&scores);
        }
    }

    /// A list past the keys' places is read as the pass reads it, and a list
    /// that ends at their last place by keys: its last row's place is zero.
    #[test]
    fn a_list_past_the_keys_places_is_read_as_the_pass_reads_it() {
        let last = 1usize << PLACE_BITS;
        for len in [last - 1, last, last + 1, last + 7] {
            // One score everywhere but at both ends: the best rows stand at
            // the first place and at the last ones, ties among them.
            let mut scores = vec![0.25f32; len];
            scores[0] = 1.0;
            for (back, score) in [1.0f32, 0.5, 0.75, 1.0, 0.5].into_iter().enumerate() {
                scores[len - 1 - back] = score;
            }
            let mut drawn = [(usize::MAX, f32::NEG_INFINITY); 5];
            let mut pass = drawn;
            assert_eq!(top_drawn(&scores, &mut drawn), top_pass(&scores, &mut pass));
            assert_eq!(drawn, pass, "{len} rows");
        }
    }

    /// A draw's trace names the best of its rows as a pass's trace does,
    /// equal scores among them, and does not count its survivors.
    #[test]
    fn a_draws_trace_names_its_best_rows_as_a_passes_trace_does() {
        let pool = pool_of(60, |_| false);
        let facts = request(7);
        // A draw's order, and scores that tie: twenty rows, which a draw
        // reads by keys.
        let kept: Vec<u32> = (0..60u32).rev().step_by(3).collect();
        assert!(kept.len() >= KEYED_FROM);
        let scores: Vec<f32> = kept.iter().map(|&row| (row % 7) as f32 / 8.0).collect();
        let pass = trace_of(3, &facts, &PoolFacts::default(), &pool, &kept, &scores);
        let draw = trace_drawn(3, &facts, &PoolFacts::default(), &pool, &kept, &scores);
        assert_eq!(usize::from(pass.top_len), TRACE_TOP);
        assert_eq!(draw.survivors, Survivors::NOT_COUNTED);
        assert_eq!(
            Trace {
                survivors: pass.survivors,
                ..draw
            },
            pass
        );
    }

    /// A decode decision of a model that also pairs is decided by its own
    /// profile, and the rows a host names for it are capped by that
    /// profile's sample ([`Engine::sample_for`]). A decode worker's needs
    /// carry the pair's decode side too ([`Engine::needs_for`]), whose
    /// sample is the larger here: a list capped by them stands past the
    /// cap.
    #[test]
    fn a_decode_decision_of_a_model_that_also_pairs_is_capped_by_its_own_profile() {
        let alone = Binding {
            model: "mixed".into(),
            phases: vec![Phase::Decode],
            ..single_binding()
        };
        let pair = Binding {
            model: "mixed".into(),
            profile: 1,
            decode_profile: Some(1),
            ..pair_binding()
        };
        let engine = sampled_under(vec![alone, pair]);
        assert_eq!(engine.needs_for("mixed", Phase::Decode).sample, 16);
        assert_eq!(engine.sample_for("mixed", Phase::Decode), 4);
        assert_eq!(engine.sample_for("nobody", Phase::Decode), 0);
        let holders: Vec<crate::engine::AffinityHolder> =
            (0..4).map(|row| holding(row, 12)).collect();
        let facts = RequestFacts {
            model: "mixed",
            phase: Phase::Decode,
            ..naming(1, &holders)
        };
        let got = drawn(&engine, &facts, &pool_of(64, |_| false));
        assert!(
            got.trace().drew_first(),
            "capped by its own profile, it draws"
        );
    }

    /// The engine asks a draw of a pool of [`DRAW_FROM`] rows or more,
    /// whatever the sample: under a sample of four a pool of nine to
    /// fifteen rows, more than twice the sample, is still read whole.
    #[test]
    fn a_sample_under_eight_draws_from_sixteen_rows_on() {
        let engine = two_samples();
        let holders = [holding(0, 12)];
        let facts = RequestFacts {
            model: "narrow",
            ..naming(1, &holders)
        };
        for (rows, drew) in [(9, false), (12, false), (15, false), (16, true)] {
            let got = drawn(&engine, &facts, &pool_of(rows, |_| false));
            assert_eq!(got.trace().drew_first(), drew, "{rows} rows");
        }
    }

    /// A rank of the tests: a row reads the level named for its position,
    /// whichever rows it is asked of.
    /// Where it `says`, it says its spread and its front, as a rank does;
    /// where it does not it is a stage of any values, and its profile's
    /// decision is the weighted sum's.
    #[derive(Debug)]
    struct Levelled {
        levels: Vec<f32>,
        step: f32,
        says: bool,
    }

    impl Lifecycle for Levelled {}

    impl Scorer for Levelled {
        fn name(&self) -> &'static str {
            "levelled"
        }

        fn score(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            self.score_front(facts, candidates, kept, kept, out).0
        }

        fn spread(&self) -> ScoreSpread {
            if self.says {
                ScoreSpread::Levels(self.step)
            } else {
                ScoreSpread::Any
            }
        }

        fn front_wise(&self) -> bool {
            true
        }

        fn score_front(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            _: &[u32],
            asked: &[u32],
            out: &mut [f32],
        ) -> (TieBand, ScoreFront) {
            let mut top = 0;
            for (o, &row) in out.iter_mut().zip(asked) {
                *o = self.levels[candidates[row as usize].position as usize];
                top += u32::from(*o >= 1.0);
            }
            let front = if self.says {
                ScoreFront::At(top)
            } else {
                ScoreFront::Unsaid
            };
            (TieBand(0.0), front)
        }
    }

    /// A stage of any values: a row reads the value named for its position,
    /// under the tie band the stage was given.
    #[derive(Debug)]
    struct Valued(Vec<f32>, f32);

    impl Lifecycle for Valued {}

    impl Scorer for Valued {
        fn name(&self) -> &'static str {
            "valued"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            for (o, &row) in out.iter_mut().zip(kept) {
                *o = self.0[candidates[row as usize].position as usize];
            }
            TieBand(self.1)
        }

        fn front_wise(&self) -> bool {
            true
        }
    }

    /// The probe of the rows a stage is handed, answering row by row: a
    /// later position scores higher, and the rows handed last are kept.
    #[derive(Debug)]
    struct Asked(Arc<Mutex<Vec<u32>>>);

    impl Lifecycle for Asked {}

    impl Scorer for Asked {
        fn name(&self) -> &'static str {
            "asked"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            *self.0.lock().expect("the probe's rows") = kept.to_vec();
            for (o, &row) in out.iter_mut().zip(kept) {
                *o = candidates[row as usize].position as f32 / 8.0;
            }
            TieBand::NORMALIZED
        }

        fn front_wise(&self) -> bool {
            true
        }
    }

    /// A stage that orders nothing, counting the times it is asked.
    #[derive(Debug)]
    struct Flat(Arc<Mutex<u32>>);

    impl Lifecycle for Flat {}

    impl Scorer for Flat {
        fn name(&self) -> &'static str {
            "flat"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            *self.0.lock().expect("the count") += 1;
            out.fill(1.0);
            TieBand(0.0)
        }

        fn spread(&self) -> ScoreSpread {
            ScoreSpread::Alike
        }
    }

    /// A picker that reads every score: the lowest first.
    #[derive(Debug)]
    struct Lowest;

    impl Lifecycle for Lowest {}

    impl Picker for Lowest {
        fn name(&self) -> &'static str {
            "lowest"
        }

        fn pick(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            _: TieBand,
            out: &mut Targets,
        ) {
            let mut order: Vec<usize> = (0..kept.len()).collect();
            order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
            for at in order.into_iter().take(MAX_TARGETS) {
                out.push(target_of(&candidates[kept[at] as usize]));
            }
        }
    }

    /// A host's picker that draws for every request and says that it takes
    /// its targets from a front.
    #[derive(Debug)]
    struct Eager;

    impl Lifecycle for Eager {}

    impl Picker for Eager {
        fn name(&self) -> &'static str {
            "eager"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Lowest.pick(facts, candidates, kept, scores, band, out);
        }

        fn draws_first(&self) -> bool {
            true
        }

        fn front_wise(&self) -> bool {
            true
        }
    }

    /// One profile over the healthy rows: `stages` under `picker`.
    fn staged(stages: Vec<WeightedScorer>, picker: Box<dyn Picker>) -> Engine {
        Engine::new(
            vec![Profile {
                name: "staged".into(),
                filters: vec![Box::new(Healthy)],
                scorers: stages,
                picker,
            }],
            vec![single_binding()],
            1,
            Needs::default(),
        )
    }

    /// A rank of `levels` with a step of a half, and a stage of any values.
    fn rank(levels: &[f32], weight: f32, says: bool) -> WeightedScorer {
        WeightedScorer {
            scorer: Box::new(Levelled {
                levels: levels.to_vec(),
                step: 0.5,
                says,
            }),
            weight,
        }
    }

    fn valued(values: &[f32], band: f32) -> WeightedScorer {
        WeightedScorer {
            scorer: Box::new(Valued(values.to_vec(), band)),
            weight: 1.0,
        }
    }

    /// `rows` ready candidates, then candidates that are not ready up to
    /// the count a decision is taken by rank from ([`RANK_FROM`]): the
    /// filters leave the `rows`.
    fn ranked_pool(rows: u32) -> Vec<Candidate> {
        (0..rows.max(RANK_FROM as u32))
            .map(|i| Candidate {
                health: Health {
                    healthy: i < rows,
                    ..Health::default()
                },
                ..candidate(i, 0)
            })
            .collect()
    }

    /// What the build reads of a profile's stages: which lead as ranks, the
    /// band's room under them, and the mark, which is the front pickers'
    /// alone.
    #[test]
    fn the_build_reads_which_leading_stages_are_ranks() {
        let order = |stages: Vec<WeightedScorer>| staged(stages, Box::new(Best)).orders[0];
        let level = |step: f32, weight: f32| WeightedScorer {
            scorer: Box::new(Levelled {
                levels: Vec::new(),
                step,
                says: true,
            }),
            weight,
        };
        let any = |weight: f32| WeightedScorer {
            scorer: Box::new(LeastRequests),
            weight,
        };
        let flat = |weight: f32| WeightedScorer {
            scorer: Box::new(Flat(Arc::default())),
            weight,
        };
        let wise = |weight: f32| WeightedScorer {
            weight,
            ..valued(&[], 0.0)
        };
        let ranks = |ranks: u8, alike: u8, fronts: u8, room: f32| Order {
            ranks,
            alike,
            fronts,
            room,
        };
        let none = ranks(0, 0, 0, 0.0);
        // A level that outweighs what stands behind it is a rank, and the
        // band's room is half of what it outweighs it by; one that only
        // equals it is none.
        assert_eq!(order(vec![level(1.0, 3.0), any(1.0)]), ranks(1, 0, 0, 0.25));
        // It may put rows in front where every stage behind it answers row
        // by row, or orders nothing.
        assert_eq!(
            order(vec![level(1.0, 3.0), wise(1.0)]),
            ranks(1, 0, 0b1, 0.25)
        );
        assert_eq!(
            order(vec![level(0.5, 4.0), flat(1.0), wise(1.0)]),
            ranks(2, 0b10, 0b01, 1.0 / 6.0 / 2.0)
        );
        // Two stages behind it that each order the rows: every row is asked.
        assert_eq!(
            order(vec![level(1.0, 4.0), wise(1.0), wise(1.0)]),
            ranks(1, 0, 0, 1.0 / 3.0 / 2.0)
        );
        assert_eq!(order(vec![level(1.0, 1.0), any(1.0)]), none);
        assert_eq!(order(vec![level(0.5, 2.0), any(1.0)]), none);
        // One that outweighs it by less than the least share is none.
        assert_eq!(order(vec![level(1.0, 1.000_01), any(1.0)]), none);
        // A stage that orders nothing is a rank where it leads, and weighs
        // nothing behind one.
        assert_eq!(
            order(vec![level(0.5, 4.0), flat(1.0), any(1.0)]),
            ranks(2, 0b10, 0, 1.0 / 6.0 / 2.0)
        );
        assert_eq!(order(vec![flat(1.0), any(1.0)]), ranks(1, 0b1, 0, 0.0));
        // Behind the first stage that is no rank, none is.
        assert_eq!(order(vec![any(1.0), level(1.0, 8.0)]), none);
        // A weight that is not above zero leaves the profile its sum.
        assert_eq!(order(vec![level(1.0, 2.0), any(0.0)]), none);
        // The mark is the front pickers': one that may read every score
        // has its profile asked whole, and so has one that draws first.
        let marked = |picker: Box<dyn Picker>| {
            let engine = staged(vec![level(1.0, 3.0), wise(1.0)], picker);
            assert_eq!(engine.orders[0].ranks, 1);
            engine.marks[0].0 & Marks::RANKED != 0
        };
        // A profile whose ranks leave nothing unasked decides by its sum.
        let plain = staged(vec![level(1.0, 3.0), any(1.0)], Box::new(Best));
        assert_eq!(plain.marks[0].0 & Marks::RANKED, 0);
        assert!(marked(Box::new(Best)) && marked(Box::new(CountFirst)));
        assert!(!marked(Box::new(Lowest)) && !marked(Box::new(Random)));
        // A profile whose picker draws for every request holds no mark
        // either, whatever that picker says of a front: the draw is asked
        // first, and a decision knows the profile by one compare of its
        // marks ([`Marks::ahead`]).
        let eager = Profile {
            name: "eager".into(),
            filters: Vec::new(),
            scorers: Vec::new(),
            picker: Box::new(Eager),
        };
        let gains = staged(vec![level(1.0, 3.0), wise(1.0)], Box::new(Best)).orders[0];
        assert_eq!(Marks::of(&eager, gains).ahead(), Marks::DRAWS_FIRST);
        // The cache-aware preset: the warm-up slice and the holders' tiers
        // are ranks, the gate's stage orders nothing, and the load is asked
        // of the rows they put in front.
        let engine = preset("cache_aware");
        let order = engine.orders[0];
        assert_eq!((order.ranks, order.alike, order.fronts), (3, 0b100, 0b010));
        assert!(order.room > 0.0);
        assert_ne!(engine.marks[0].0 & Marks::RANKED, 0);
        // A profile of one scorer of any values, or of none, holds no rank.
        for name in ["least_load", "round_robin", "consistent_hashing"] {
            let engine = preset(name);
            assert_eq!(engine.orders, vec![none], "{name}");
            assert_eq!(engine.marks[0].0 & Marks::RANKED, 0, "{name}");
        }
    }

    /// Who a stage behind a rank is asked of. The rows at the rank's
    /// ceiling stand in front where they fill a decision's four targets and
    /// are at most half of the rows: the stage behind is asked of them
    /// alone. Fewer than the targets, or more than half, and every row is
    /// asked. The probe scores a later position higher.
    #[test]
    fn a_stage_behind_a_rank_is_asked_of_the_rows_in_front_alone() {
        let handed = Arc::new(Mutex::new(Vec::new()));
        let pool = ranked_pool(9);
        // The targets by position, and the rows the stage behind was handed.
        let asked = |levels: &[f32]| {
            let probe = WeightedScorer {
                scorer: Box::new(Asked(Arc::clone(&handed))),
                weight: 1.0,
            };
            let engine = staged(vec![rank(levels, 4.0, true), probe], Box::new(Best));
            assert_eq!(engine.orders[0].ranks, 1);
            let decision = reported(&engine, &request(7), &pool);
            let rows = handed.lock().expect("the probe's rows").clone();
            (positions(&decision), rows)
        };
        let every: Vec<u32> = (0..9).collect();
        // Four of nine fill the targets and are under half: the four alone.
        let (order, rows) = asked(&[1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5]);
        assert_eq!(rows, vec![0, 2, 4, 6]);
        assert_eq!(order, vec![6, 4, 2, 0]);
        // Five of nine are more than half: every row.
        let (order, rows) = asked(&[1.0, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5]);
        assert_eq!(rows, every);
        assert_eq!(order, vec![6, 4, 2, 1]);
        // Three of nine are fewer than the targets: every row, and the
        // fourth target is the best of the rows a level under.
        let (order, rows) = asked(&[1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]);
        assert_eq!(rows, every);
        assert_eq!(order, vec![4, 2, 0, 8]);
        // No row at the ceiling: every row.
        let (order, rows) = asked(&[0.5; 9]);
        assert_eq!(rows, every);
        assert_eq!(order, vec![8, 7, 6, 5]);
        // Behind a stage that does not answer for a front alone no row
        // stands in front: four of nine, and every row is asked.
        let whole = WeightedScorer {
            scorer: Box::new(Handed(Arc::clone(&handed))),
            weight: 1.0,
        };
        let levels = [1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5];
        let engine = staged(vec![rank(&levels, 4.0, true), whole], Box::new(Best));
        assert_eq!((engine.orders[0].ranks, engine.orders[0].fronts), (1, 0));
        reported(&engine, &request(7), &pool);
        assert_eq!(*handed.lock().expect("the probe's rows"), every);
    }

    /// A request that names rows is the draw's only under a picker that
    /// draws for it ([`Marks::JOINED`]). Under one that never draws, a
    /// profile of ranks is asked by rank: the stage behind the rank of the
    /// rows in front alone.
    #[test]
    fn a_request_that_names_rows_is_asked_by_rank_where_the_picker_never_draws() {
        let handed = Arc::new(Mutex::new(Vec::new()));
        let probe = WeightedScorer {
            scorer: Box::new(Asked(Arc::clone(&handed))),
            weight: 1.0,
        };
        let levels = [1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5];
        let engine = staged(vec![rank(&levels, 4.0, true), probe], Box::new(Best));
        assert_eq!(engine.marks[0].ahead(), Marks::RANKED);
        let pool = ranked_pool(9);
        let facts = RequestFacts {
            named: Some(AffinityRows::default()),
            ..request(7)
        };
        let decision = reported(&engine, &facts, &pool);
        assert_eq!(*handed.lock().expect("the probe's rows"), vec![0, 2, 4, 6]);
        assert_eq!(positions(&decision), vec![6, 4, 2, 0]);
    }

    /// A stage of the tests as a draw may be handed it: the stage's own
    /// answers, and the word that it reads each row alone
    /// ([`Scorer::row_wise`]).
    #[derive(Debug)]
    struct RowWise<S>(S);

    impl<S: Scorer> Lifecycle for RowWise<S> {}

    impl<S: Scorer> Scorer for RowWise<S> {
        fn name(&self) -> &'static str {
            self.0.name()
        }

        fn score(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            self.0.score(facts, candidates, kept, out)
        }

        fn row_wise(&self) -> bool {
            true
        }

        fn front_wise(&self) -> bool {
            self.0.front_wise()
        }

        fn spread(&self) -> ScoreSpread {
            self.0.spread()
        }

        fn score_front(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            asked: &[u32],
            out: &mut [f32],
        ) -> (TieBand, ScoreFront) {
            self.0.score_front(facts, candidates, kept, asked, out)
        }

        fn needs(&self) -> Needs {
            self.0.needs()
        }

        fn scope(&self) -> Option<&dyn DecisionScope> {
            self.0.scope()
        }

        fn warm_up_rule(&self) -> Option<WarmUpRule> {
            self.0.warm_up_rule()
        }
    }

    /// A host's picker the rows named join, which reads every score: it
    /// says nothing of a front, so its profile holds no rank mark.
    #[derive(Debug)]
    struct Joining;

    impl Lifecycle for Joining {}

    impl Picker for Joining {
        fn name(&self) -> &'static str {
            "joining"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Lowest.pick(facts, candidates, kept, scores, band, out);
        }

        fn draws_first(&self) -> bool {
            true
        }

        fn joined_by_named(&self) -> bool {
            true
        }
    }

    /// Under a picker the rows named join ([`Marks::JOINED`]), over a rated
    /// pool of [`DRAW_FROM`] rows: a request that names rows is the draw's
    /// ahead of the ranks, and one that names none is the ranks' by the
    /// profile's rank mark alone. Without the mark every row is asked.
    #[test]
    fn a_joined_picker_draws_for_the_rows_named_and_ranks_by_the_mark_alone() {
        let handed = Arc::new(Mutex::new(Vec::new()));
        let rows = || handed.lock().expect("the probe's rows").clone();
        // A rank that puts four of sixteen rows in front, and the probe of
        // the rows the stage behind it is handed.
        let stages = || {
            let ranks = Levelled {
                levels: (0..16).map(|row| if row < 4 { 1.0 } else { 0.5 }).collect(),
                step: 0.5,
                says: true,
            };
            vec![
                WeightedScorer {
                    scorer: Box::new(RowWise(ranks)),
                    weight: 4.0,
                },
                WeightedScorer {
                    scorer: Box::new(RowWise(Asked(Arc::clone(&handed)))),
                    weight: 1.0,
                },
            ]
        };
        let pool = ranked_pool(16);
        let names = RequestFacts {
            named: Some(AffinityRows::default()),
            ..request(7)
        };
        // Under the built-in picker of that kind the profile holds the
        // three marks. A request that names none is asked by rank: the
        // stage behind the rank of the four rows in front.
        let sampled = pickers::SampledBest::new(2);
        let engine = staged(stages(), Box::new(sampled));
        assert_eq!(
            engine.marks[0].ahead(),
            Marks::DRAWS_FIRST | Marks::JOINED | Marks::RANKED
        );
        let ranked = reported(&engine, &request(7), &pool);
        assert!(!ranked.trace().drew_first());
        assert_eq!(rows(), vec![0, 1, 2, 3]);
        // One that names rows is the draw's, whatever the ranks would put in
        // front: the stage is handed the sample's two rows.
        let drew = reported(&engine, &names, &pool);
        assert!(drew.trace().drew_first());
        assert_eq!(rows().len(), 2);
        // The same stages under a picker that reads every score hold no
        // rank mark, though the first of them is a rank: a request that
        // names none is asked whole, every row.
        let engine = staged(stages(), Box::new(Joining));
        assert_eq!(engine.orders[0].fronts, 0b1);
        assert_eq!(engine.marks[0].ahead(), Marks::DRAWS_FIRST | Marks::JOINED);
        let whole = reported(&engine, &request(7), &pool);
        assert!(!whole.trace().drew_first());
        assert_eq!(rows(), (0..16).collect::<Vec<u32>>());
    }

    /// A leading stage that orders nothing is not asked by a ranked
    /// decision: its share stands in every row's sum, and the decision is
    /// the one the weighted sum reads, which asks it once.
    #[test]
    fn a_stage_that_orders_nothing_is_not_asked() {
        let asks = Arc::new(Mutex::new(0));
        let stages = vec![
            rank(&[1.0, 0.5, 0.0, 0.5, 1.0], 4.0, true),
            WeightedScorer {
                scorer: Box::new(Flat(Arc::clone(&asks))),
                weight: 1.0,
            },
            WeightedScorer {
                scorer: Box::new(Asked(Arc::default())),
                weight: 1.0,
            },
        ];
        let engine = staged(stages, Box::new(Best));
        let order = engine.orders[0];
        assert_eq!((order.ranks, order.alike), (2, 0b10));
        let pool = ranked_pool(5);
        let facts = request(3);
        let ranked = reported(&engine, &facts, &pool);
        assert_eq!(*asks.lock().expect("the count"), 0);
        let summed = whole_pass(&engine, &facts, &pool);
        assert_eq!(*asks.lock().expect("the count"), 1);
        assert_eq!(ranked, summed);
    }

    /// The word a stage's scores read as under its spread ([`ScoreFront::of`]):
    /// alike where every row stands at the ceiling, a count where every
    /// other row stands a whole level or more under it, and nothing of any
    /// other scores.
    #[test]
    fn a_word_is_read_off_the_scores_by_the_spread() {
        let levels = ScoreSpread::Levels(0.5);
        assert_eq!(ScoreFront::of(levels, &[1.0, 1.0]), ScoreFront::Alike);
        assert_eq!(ScoreFront::of(levels, &[]), ScoreFront::Alike);
        assert_eq!(
            ScoreFront::of(levels, &[1.0, 0.5, 0.0, 1.0]),
            ScoreFront::At(2)
        );
        assert_eq!(ScoreFront::of(levels, &[0.5, 0.0]), ScoreFront::At(0));
        // A row less than a whole level under the ceiling, or no number.
        assert_eq!(ScoreFront::of(levels, &[1.0, 0.75]), ScoreFront::Unsaid);
        assert_eq!(ScoreFront::of(levels, &[1.0, f32::NAN]), ScoreFront::Unsaid);
        assert_eq!(
            ScoreFront::of(ScoreSpread::Any, &[1.0, 0.0]),
            ScoreFront::Unsaid
        );
        assert_eq!(
            ScoreFront::of(ScoreSpread::Alike, &[1.0, 1.0]),
            ScoreFront::Alike
        );
    }

    /// A pool under the count a decision is taken by rank from is asked as
    /// a profile without ranks asks: the stage that orders nothing is asked
    /// at fifteen rows and is not at sixteen.
    #[test]
    fn a_pool_under_sixteen_rows_is_not_decided_by_rank() {
        let asks = Arc::new(Mutex::new(0));
        let levels: Vec<f32> = (0..16).map(|i| if i < 4 { 1.0 } else { 0.5 }).collect();
        let stages = vec![
            rank(&levels, 4.0, true),
            WeightedScorer {
                scorer: Box::new(Flat(Arc::clone(&asks))),
                weight: 1.0,
            },
            WeightedScorer {
                scorer: Box::new(Asked(Arc::default())),
                weight: 1.0,
            },
        ];
        let engine = staged(stages, Box::new(Best));
        assert_ne!(engine.marks[0].0 & Marks::RANKED, 0);
        let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
        let facts = request(3);
        let under = reported(&engine, &facts, &pool[..RANK_FROM - 1]);
        assert_eq!(*asks.lock().expect("the count"), 1);
        reported(&engine, &facts, &pool);
        assert_eq!(*asks.lock().expect("the count"), 1);
        assert_eq!(under, whole_pass(&engine, &facts, &pool[..RANK_FROM - 1]));
    }

    /// Under each picker of the catalog, and under one that reads every
    /// score, a decision whose leading stage is a rank is the decision of
    /// the weighted sum of every stage over every row: the targets, the
    /// alternates and the trace, over random rows, levels, values, weights,
    /// tie bands and seeds. The twin's stages score alike and say nothing,
    /// so its profile holds no rank. A picker the validator builds from its
    /// parameters is not of this build yet.
    #[test]
    fn a_ranked_decision_is_the_weighted_sums_under_every_picker() {
        use crate::stages::registry::{Found, StageRegistry};
        let catalog = StageRegistry::builtin();
        let build = |name: &str| match catalog.find_picker(name) {
            Some(Found::Built(picker)) => Some(picker),
            Some(Found::Typed | Found::With(_)) => None,
            None => Some(Box::new(Lowest) as Box<dyn Picker>),
        };
        let mut marked = Vec::new();
        for name in catalog.pickers().chain(["lowest"]) {
            let mut draw = Draw::new(0x5eed);
            for round in 0..300u64 {
                let (Some(by_rank), Some(by_sum)) = (build(name), build(name)) else {
                    break;
                };
                let rows = 1 + draw.below(40) as u32;
                let levels: Vec<f32> = (0..rows)
                    .map(|_| [0.0, 0.5, 0.5, 1.0][draw.below(4)])
                    .collect();
                let values: Vec<f32> = (0..rows).map(|_| draw.below(9) as f32 / 8.0).collect();
                // A band of none, a small one, and two a rank's room may
                // not hold.
                let band = [0.0, 1e-6, 0.3, 1.0][draw.below(4)];
                // A lead that is no rank, and two that are.
                let lead = [1.5, 3.0, 8.0][draw.below(3)];
                let stages = |says| vec![rank(&levels, lead, says), valued(&values, band)];
                let (ranked, summed) =
                    (staged(stages(true), by_rank), staged(stages(false), by_sum));
                assert_eq!(summed.marks[0].0 & Marks::RANKED, 0, "{name}");
                if ranked.marks[0].0 & Marks::RANKED != 0 && !marked.contains(&name) {
                    marked.push(name);
                }
                let pool: Vec<Candidate> = (0..rows)
                    .map(|i| candidate(i, draw.below(3) as u32))
                    .collect();
                for seed in 0..4 {
                    let facts = request(round * 4 + seed);
                    assert_eq!(
                        reported(&ranked, &facts, &pool),
                        reported(&summed, &facts, &pool),
                        "{name}, round {round}, seed {seed}: levels {levels:?}, values \
                         {values:?}, band {band}, lead {lead}"
                    );
                }
            }
        }
        // The front pickers decided by rank: the case is not of sums alone.
        assert_eq!(marked, ["best", "count-first"]);
    }

    /// A tie band wider than a rank's room reaches from the rows in front
    /// to a row behind them, and the decision asks every row every stage.
    /// Four rows stand at the ceiling with the stage behind at nothing, and
    /// one a level under them with it at its most: 0.75 and 0.625 of the
    /// sum, one tie under a band of 0.25. Over the seeds the row behind
    /// leads some decisions, as it does by the weighted sum.
    #[test]
    fn a_tie_band_wider_than_a_ranks_room_asks_every_row_every_stage() {
        let levels = [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let values = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let engine = |says| {
            let stages = vec![rank(&levels, 3.0, says), valued(&values, 1.0)];
            staged(stages, Box::new(Best))
        };
        let (ranked, summed) = (engine(true), engine(false));
        assert_eq!(
            ranked.orders[0],
            Order {
                ranks: 1,
                alike: 0,
                fronts: 0b1,
                room: 0.0625
            }
        );
        let pool = ranked_pool(10);
        let mut led = Vec::new();
        for seed in 0..64 {
            let facts = request(seed);
            let decision = reported(&ranked, &facts, &pool);
            assert_eq!(decision, reported(&summed, &facts, &pool), "seed {seed}");
            led.push(positions(&decision)[0]);
        }
        assert!(led.contains(&4), "{led:?}");
    }

    /// A rank of the tests that says `1` of the scores `0`, which its
    /// first rows read; every other row reads nothing.
    #[derive(Debug)]
    struct Says(Vec<f32>, ScoreFront);

    impl Lifecycle for Says {}

    impl Scorer for Says {
        fn name(&self) -> &'static str {
            "says"
        }

        fn score(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            self.score_front(facts, candidates, kept, kept, out).0
        }

        fn spread(&self) -> ScoreSpread {
            ScoreSpread::Levels(0.5)
        }

        fn score_front(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            _: &[u32],
            out: &mut [f32],
        ) -> (TieBand, ScoreFront) {
            out.fill(0.0);
            out[..self.0.len()].copy_from_slice(&self.0);
            (TieBand(0.0), self.1)
        }
    }

    /// A decision over twelve rows under a rank that says `said` of
    /// `scores`, ahead of a stage that answers for a front alone: the
    /// profile is decided by rank, so the engine reads the word.
    fn told(scores: &[f32], said: ScoreFront) {
        let stages = vec![
            WeightedScorer {
                scorer: Box::new(Says(scores.to_vec(), said)),
                weight: 4.0,
            },
            valued(&[0.5; 12], 0.0),
        ];
        let engine = staged(stages, Box::new(Best));
        let pool = ranked_pool(12);
        reported(&engine, &request(1), &pool);
    }

    /// A rank that says a count of ceilings its scores do not hold fails a
    /// debug build at the engine's check of the word: its author's case,
    /// never a wrong decision in the field.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "says said At(5) of scores that read otherwise")]
    fn a_rank_that_miscounts_its_front_fails_a_debug_build() {
        told(&[1.0, 1.0, 1.0, 1.0], ScoreFront::At(5));
    }

    /// And so does one whose row stands within a step of the ceiling: the
    /// rows behind a front lie a step of the spread under it.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "says said At(4) of scores that read otherwise")]
    fn a_rank_with_a_row_within_a_step_of_its_ceiling_fails_a_debug_build() {
        told(&[1.0, 1.0, 1.0, 1.0, 0.75], ScoreFront::At(4));
    }

    /// The cache-aware preset by rank is its weighted sum. Over pools of 4
    /// to 64 rows where every row holds the request's prefix, one in two,
    /// one in eight or none, under random loads and seeds, the decision is
    /// the whole pass's: its targets, its alternates and the rows its trace
    /// names. The whole pass asks each stage by its `score`, the decision by
    /// rank asks the ranks by their `score_front`: a row scores alike by
    /// either. The load is asked of the holders in front alone and scores
    /// them among themselves, so their scores are not the whole pass's.
    #[test]
    fn the_cache_aware_preset_decides_by_rank_what_its_weighted_sum_decides() {
        let engine = preset("cache_aware");
        let mut draw = Draw::new(0xace);
        let mut fronts = 0;
        for round in 0..300u64 {
            let rows = 4 + draw.below(61) as u32;
            let pool: Vec<Candidate> = (0..rows)
                .map(|i| candidate(i, draw.below(4) as u32))
                .collect();
            let odds = [1, 2, 8, 1 << 20][draw.below(4)];
            let overlaps: Vec<u32> = (0..rows)
                .map(|_| if draw.below(odds) == 0 { 4 } else { 0 })
                .collect();
            let column = held(&overlaps);
            let facts = RequestFacts {
                prompt_tokens: 512,
                now_micros: 1_000_000,
                affinity: &column,
                ..request(round)
            };
            let holders = overlaps.iter().filter(|&&blocks| blocks > 0).count();
            fronts += usize::from(holders >= MAX_TARGETS && holders * 2 <= pool.len());
            let (ranked, summed) = (
                reported(&engine, &facts, &pool),
                whole_pass(&engine, &facts, &pool),
            );
            assert_eq!(
                positions(&ranked),
                positions(&summed),
                "round {round}: {overlaps:?}"
            );
            let named = |decision: &Decision| {
                let trace = decision.pick().expect("a pick").trace;
                (trace.top.map(|(id, _)| id), trace.top_len, trace.survivors)
            };
            assert_eq!(named(&ranked), named(&summed), "round {round}");
        }
        // In some rounds the holders fill the targets and are under half.
        assert!(fronts > 20, "{fronts}");
    }

    /// The cache-aware preset over fleets whose load reports are mixed: by
    /// rank it decides what its whole pass decides, in every round of four
    /// mixes. One row in eight is not ready and reports the pool's fastest
    /// rate: a filter drops it, and no cost reads it. Where every row in
    /// front holds a report the load is asked of the holders in front
    /// alone, and a report without a rate drains at the mean rate of the
    /// rows the filters kept. A row without a report is costed from its
    /// reporting peers: in front it makes the load say that it must be
    /// asked whole, and it is.
    #[test]
    fn the_cache_aware_preset_decides_by_rank_its_whole_pass_under_mixed_reports() {
        let engine = preset("cache_aware");
        let mut draw = Draw::new(0xace);
        // The rounds whose holders stand in front, and those of them the
        // front decided alone: the trace's scores are not the whole pass's.
        let (mut fronts, mut alone) = ([0u32; 4], [0u32; 4]);
        for round in 0..40_000u64 {
            // Every row rated; a third of the reports without a rate; a
            // third of the rows without a report; a third of the reports
            // without a rate and one row without a report.
            let mix = (round % 4) as usize;
            let rows = 8 + draw.below(57) as u32;
            let dark = draw.below(rows as usize) as u32;
            let pool: Vec<Candidate> = (0..rows)
                .map(|i| {
                    let mut row = candidate(i, draw.below(4) as u32);
                    row.health.healthy = draw.below(8) != 0;
                    let reports = match mix {
                        2 => draw.below(3) != 0,
                        3 => i != dark,
                        _ => true,
                    };
                    let rated = !matches!(mix, 1 | 3) || draw.below(3) != 0;
                    if reports {
                        row.load = Some(view::LoadRow {
                            running: draw.below(8) as u32,
                            queued: draw.below(4) as u32,
                            waiting_uncached_tokens: draw.below(8192) as u32,
                            live_rate: match (rated, row.health.healthy) {
                                (false, _) => 0,
                                (true, true) => 500 + draw.below(4000) as u32,
                                (true, false) => 60_000,
                            },
                            ..Default::default()
                        });
                    }
                    row
                })
                .collect();
            let odds = [2, 4, 8][draw.below(3)];
            let overlaps: Vec<u32> = (0..rows)
                .map(|_| if draw.below(odds) == 0 { 4 } else { 0 })
                .collect();
            let column = held(&overlaps);
            let facts = RequestFacts {
                prompt_tokens: 512,
                now_micros: 1_000_000,
                affinity: &column,
                ..request(round)
            };
            let kept = pool.iter().filter(|row| row.health.healthy).count();
            let holders = pool
                .iter()
                .zip(&overlaps)
                .filter(|(row, blocks)| row.health.healthy && **blocks > 0)
                .count();
            let front = holders >= MAX_TARGETS && holders * 2 <= kept;
            let (ranked, summed) = (
                reported(&engine, &facts, &pool),
                whole_pass(&engine, &facts, &pool),
            );
            assert_eq!(
                positions(&ranked),
                positions(&summed),
                "round {round}, mix {mix}: {overlaps:?}"
            );
            let top = |decision: &Decision| decision.pick().expect("a pick").trace.top;
            let named = |decision: &Decision| {
                let trace = decision.pick().expect("a pick").trace;
                (trace.top.map(|(id, _)| id), trace.top_len, trace.survivors)
            };
            assert_eq!(named(&ranked), named(&summed), "round {round}, mix {mix}");
            fronts[mix] += u32::from(front);
            alone[mix] += u32::from(front && top(&ranked) != top(&summed));
        }
        // In each mix the holders fill the targets and are under half in
        // most rounds.
        assert!(fronts.iter().all(|&rounds| rounds > 5_000), "{fronts:?}");
        // A front whose rows each hold a report decides alone, with a rate
        // or without one. A row without a report in it has it asked whole:
        // one such row stands in some fronts, and a third of the rows in
        // nearly every one.
        assert!(alone[0] > 5_000 && alone[1] > 5_000, "{alone:?}");
        assert!(alone[3] > 3_500 && alone[3] < alone[1], "{alone:?}");
        assert!(alone[2] < 1_000, "{alone:?}");
    }

    /// The cache-aware preset where the whole pass cannot tell the rows in
    /// front apart. Eight of thirty-two rows hold the request's prefix. Each
    /// runs one request and their rates step by one token a second from
    /// 20,000, so their waits lie 2.56 microseconds apart: outside the
    /// load's tie of one microsecond. With no queue behind them the
    /// decision by rank is the whole pass's under every seed. With a queue
    /// of 200 on one row behind, the pool's costs span a hundred seconds:
    /// the whole pass spreads the load over that span, the eight's sums
    /// meet in one value and its picker draws among them. By rank the load
    /// is asked of the eight alone and orders them by their wait: the
    /// targets are the four fastest under every seed, and each is a row the
    /// whole pass draws among.
    #[test]
    fn the_rows_in_front_are_ordered_where_the_whole_pass_draws_among_them() {
        let engine = preset("cache_aware");
        let pool_of = |queued: u32| -> Vec<Candidate> {
            (0..32u32)
                .map(|at| {
                    let mut row = candidate(at, 0);
                    row.load = Some(view::LoadRow {
                        running: 1,
                        queued: if at == 31 { queued } else { 0 },
                        live_rate: if at < 8 { 20_000 + at } else { 2_000 },
                        ..Default::default()
                    });
                    row
                })
                .collect()
        };
        let overlaps: Vec<u32> = (0..32).map(|at| if at < 8 { 4 } else { 0 }).collect();
        let column = held(&overlaps);
        // The targets of 64 seeds, each list of four once: by rank, and by
        // the whole pass.
        let read = |pool: &[Candidate]| {
            let (mut by_rank, mut by_sum) = (Vec::new(), Vec::new());
            for seed in 0..64u64 {
                let facts = RequestFacts {
                    prompt_tokens: 512,
                    now_micros: 1_000_000,
                    affinity: &column,
                    ..request(seed)
                };
                by_rank.push(positions(&reported(&engine, &facts, pool)));
                by_sum.push(positions(&whole_pass(&engine, &facts, pool)));
            }
            by_rank.sort_unstable();
            by_rank.dedup();
            by_sum.sort_unstable();
            by_sum.dedup();
            (by_rank, by_sum)
        };
        let fastest = [vec![7, 6, 5, 4]];
        let (by_rank, by_sum) = read(&pool_of(0));
        assert_eq!(by_rank, fastest);
        assert_eq!(by_sum, fastest);
        let (by_rank, by_sum) = read(&pool_of(200));
        assert_eq!(by_rank, fastest);
        assert!(by_sum.len() > 1, "{by_sum:?}");
        assert!(by_sum.iter().flatten().all(|&at| at < 8), "{by_sum:?}");
    }

    /// A pool said to hold a report on every row, and a host that says it
    /// where one row holds none. A report without a rate among the rows in
    /// front drains at the mean rate of the rows the filters kept, as in
    /// the whole pass: the load stage folds it over them and the rows in
    /// front decide alone, so the targets are the whole pass's and the
    /// trace's scores are the front's own. That mean is 2,000 tokens a
    /// second here. Over the rows in front alone it is 1,000, and over the
    /// pool, with the rows a filter dropped, 6,000: either would put that
    /// row elsewhere among the targets. A row without a report among the
    /// rows in front makes the stage say that it must be asked whole: the
    /// decision is the whole pass's, and so are its trace's scores. Behind
    /// the rows in front neither is read: they are costed alone, each from
    /// its own report.
    #[test]
    fn a_row_in_front_without_a_rate_drains_at_the_kept_rows_mean_rate() {
        let engine = preset("cache_aware");
        // Thirty-two rows. The first eight hold the request's prefix and
        // run at 1,000 tokens a second, the next twenty at 2,350, and the
        // last four at 33,000 and are not ready. The row at `bare` reports
        // no rate, or does not report.
        let work = [1, 2, 4, 3, 6, 7, 8, 9];
        let pool_of = |bare: u32, reports: bool| -> Vec<Candidate> {
            (0..32u32)
                .map(|at| {
                    let mut row = candidate(at, 0);
                    row.health.healthy = at < 28;
                    let rate = [1_000, 2_350, 33_000][usize::from(at >= 8) + usize::from(at >= 28)];
                    if at != bare || reports {
                        row.load = Some(view::LoadRow {
                            running: work.get(at as usize).copied().unwrap_or(1 + at % 4),
                            live_rate: if at == bare { 0 } else { rate },
                            ..Default::default()
                        });
                    }
                    row
                })
                .collect()
        };
        let overlaps: Vec<u32> = (0..32).map(|at| if at < 8 { 4 } else { 0 }).collect();
        let column = held(&overlaps);
        let facts = RequestFacts {
            prompt_tokens: 512,
            now_micros: 1_000_000,
            affinity: &column,
            ..request(11)
        };
        // The targets by rank, which are the whole pass's, and whether the
        // trace's scores by rank are the whole pass's too.
        let decided = |pool: &[Candidate]| {
            let (ranked, summed) = (
                reported(&engine, &facts, pool),
                whole_pass(&engine, &facts, pool),
            );
            assert_eq!(positions(&ranked), positions(&summed));
            let top = |decision: &Decision| decision.pick().expect("a pick").trace.top;
            (positions(&ranked), top(&ranked) == top(&summed))
        };
        // The fourth row runs three requests. Without a rate it drains at
        // the kept rows' mean and waits behind the first row alone: the
        // front decides. At its own 1,000 it waits behind the second too.
        assert_eq!(decided(&pool_of(3, true)), (vec![0, 3, 1, 2], false));
        // Without a report it takes its reporting peers' least cost, a row
        // behind the front's: the whole pass's decision, and its scores.
        assert_eq!(decided(&pool_of(3, false)), (vec![3, 0, 1, 2], true));
        // Behind the rows in front, either way: the front decides alone.
        assert_eq!(decided(&pool_of(20, true)), (vec![0, 1, 3, 2], false));
        assert_eq!(decided(&pool_of(20, false)), (vec![0, 1, 3, 2], false));
    }

    /// A later rank puts fewer rows in front where they still fill the
    /// targets and are at most half of the rows the rank before left. Of
    /// sixteen rows the first rank leaves eight and the second four of
    /// those: the stage behind both is asked of the four alone. Five of the
    /// eight are more than half, and the eight are asked.
    #[test]
    fn a_later_rank_puts_fewer_rows_in_front() {
        let handed = Arc::new(Mutex::new(Vec::new()));
        let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
        let first: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
        let asked = |second: &[f32]| {
            let probe = WeightedScorer {
                scorer: Box::new(Asked(Arc::clone(&handed))),
                weight: 1.0,
            };
            let stages = vec![rank(&first, 12.0, true), rank(second, 3.0, true), probe];
            let engine = staged(stages, Box::new(Best));
            assert_eq!(engine.orders[0].ranks, 2);
            let decision = reported(&engine, &request(9), &pool);
            let rows = handed.lock().expect("the probe's rows").clone();
            (positions(&decision), rows)
        };
        // Three of the eight are fewer than the targets: the eight are
        // asked, and the fourth target is the best of the other five.
        let mut second = vec![0.5; 16];
        second[..3].fill(1.0);
        let (order, rows) = asked(&second);
        assert_eq!(rows, (0..8).collect::<Vec<u32>>());
        assert_eq!(order, vec![2, 1, 0, 7]);
        second[3] = 1.0;
        let (order, rows) = asked(&second);
        assert_eq!(rows, vec![0, 1, 2, 3]);
        assert_eq!(order, vec![3, 2, 1, 0]);
        second[4] = 1.0;
        let (order, rows) = asked(&second);
        assert_eq!(rows, (0..8).collect::<Vec<u32>>());
        assert_eq!(order, vec![4, 3, 2, 1]);
    }

    /// A rank that puts fewer rows in front keeps what each of them holds.
    /// The second rank leaves the last four of the eight in front, above
    /// rows it wrote two other levels: the stage behind orders the four.
    #[test]
    fn a_narrowing_rank_keeps_the_sums_of_the_rows_it_leaves_in_front() {
        let handed = Arc::new(Mutex::new(Vec::new()));
        let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
        let first: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
        let mut second = vec![0.5; 16];
        second[2..4].fill(0.0);
        second[4..8].fill(1.0);
        let probe = WeightedScorer {
            scorer: Box::new(Asked(Arc::clone(&handed))),
            weight: 1.0,
        };
        let stages = vec![rank(&first, 12.0, true), rank(&second, 3.0, true), probe];
        let engine = staged(stages, Box::new(Best));
        let decision = reported(&engine, &request(9), &pool);
        assert_eq!(*handed.lock().expect("the probe's rows"), vec![4, 5, 6, 7]);
        assert_eq!(positions(&decision), vec![7, 6, 5, 4]);
    }

    /// A stage that writes only the rows it has a score for, and keeps the
    /// rows it was handed last: a negative entry leaves its row unwritten.
    #[derive(Debug)]
    struct Sparse(Vec<f32>, Arc<Mutex<Vec<u32>>>);

    impl Lifecycle for Sparse {}

    impl Scorer for Sparse {
        fn name(&self) -> &'static str {
            "sparse"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            let mut handed = self.1.lock().expect("the probe's rows");
            handed.clear();
            for (o, &row) in out.iter_mut().zip(kept) {
                let position = candidates[row as usize].position;
                handed.push(position);
                let value = self.0[position as usize];
                if value >= 0.0 {
                    *o = value;
                }
            }
            TieBand(0.0)
        }

        fn front_wise(&self) -> bool {
            true
        }
    }

    /// A row a stage leaves unwritten reads nothing by rank too. The first
    /// rank puts eight rows in front, and the second, asked of them, leaves
    /// four there, each written its ceiling. The stage behind is handed
    /// those four and writes three: the fourth reads nothing, as in the
    /// zeroed buffer the whole pass hands, and stands last. A front that
    /// kept the rank's output would hand it that ceiling and stand it first.
    #[test]
    fn a_row_a_stage_leaves_unwritten_reads_nothing_among_the_rows_in_front() {
        let handed = Arc::new(Mutex::new(Vec::new()));
        let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
        let first: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
        let second: Vec<f32> = (0..16).map(|i| if i < 4 { 1.0 } else { 0.5 }).collect();
        let mut third = vec![-1.0; 16];
        third[..3].copy_from_slice(&[0.25, 0.5, 0.75]);
        let sparse = WeightedScorer {
            scorer: Box::new(Sparse(third, Arc::clone(&handed))),
            weight: 1.0,
        };
        let stages = vec![rank(&first, 12.0, true), rank(&second, 3.0, true), sparse];
        let engine = staged(stages, Box::new(Best));
        let by_rank = positions(&reported(&engine, &request(9), &pool));
        assert_eq!(*handed.lock().expect("the probe's rows"), vec![0, 1, 2, 3]);
        assert_eq!(by_rank, vec![2, 1, 0, 3]);
        let by_sum = positions(&whole_pass(&engine, &request(9), &pool));
        assert_eq!(by_rank, by_sum);
    }

    /// A rank parts the rows in front without leaving fewer there, the
    /// three at its ceiling being under the targets. The rank behind it
    /// then leaves no fewer either, although four stand at its own ceiling:
    /// the stage behind is asked of the eight, and the targets are the
    /// weighted sum's.
    #[test]
    fn a_rank_behind_a_parted_front_leaves_no_fewer_in_front() {
        let handed = Arc::new(Mutex::new(Vec::new()));
        let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
        let first: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
        let mut second = vec![0.5; 16];
        second[..3].fill(1.0);
        let mut third = vec![0.5; 16];
        third[2..6].fill(1.0);
        let stages = |says| {
            let probe = WeightedScorer {
                scorer: Box::new(Asked(Arc::clone(&handed))),
                weight: 1.0,
            };
            vec![
                rank(&first, 48.0, says),
                rank(&second, 12.0, says),
                rank(&third, 3.0, says),
                probe,
            ]
        };
        let (ranked, summed) = (
            staged(stages(true), Box::new(Best)),
            staged(stages(false), Box::new(Best)),
        );
        assert_eq!(ranked.orders[0].ranks, 3);
        let facts = request(9);
        let decision = reported(&ranked, &facts, &pool);
        assert_eq!(
            *handed.lock().expect("the probe's rows"),
            (0..8).collect::<Vec<u32>>()
        );
        assert_eq!(positions(&decision), vec![2, 1, 0, 5]);
        assert_eq!(
            positions(&decision),
            positions(&reported(&summed, &facts, &pool))
        );
    }

    /// A profile is asked by rank only over a pool whose host says that
    /// every row holds a report ([`PoolFacts::reported`]): the stage behind
    /// the rank is handed the four rows in front there, and every row where
    /// the host does not say it. The turn is read in both directions over
    /// one engine, and the targets are the same either way.
    #[test]
    fn a_pool_not_said_to_hold_a_report_on_every_row_is_asked_whole_at_once() {
        let handed = Arc::new(Mutex::new(Vec::new()));
        let probe = WeightedScorer {
            scorer: Box::new(Asked(Arc::clone(&handed))),
            weight: 1.0,
        };
        let levels = [1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5];
        let engine = staged(vec![rank(&levels, 4.0, true), probe], Box::new(Best));
        let pool = ranked_pool(9);
        // The targets by position, and the rows the stage behind was handed.
        let asked = |reported: bool| {
            let facts = PoolFacts {
                reported,
                ..PoolFacts::default()
            };
            let decision = engine.decide(
                &request(7),
                &facts,
                &pool,
                &Constraints::default(),
                &mut Scratch::new(),
            );
            let rows = handed.lock().expect("the probe's rows").clone();
            (positions(&decision), rows)
        };
        let targets = vec![6, 4, 2, 0];
        let front = (targets.clone(), vec![0, 2, 4, 6]);
        assert_eq!(asked(true), front);
        // One row without a rate, a fresh worker before its first sample
        // among them, and the host stops saying it: every row.
        assert_eq!(asked(false), (targets, (0..9).collect::<Vec<u32>>()));
        // It reports: the four in front again.
        assert_eq!(asked(true), front);
    }

    /// The list of the rows in front keeps its length from one decision to
    /// the next over one scratch, and is filled only as it grows. A front
    /// of four, then of six, then of four again: each decision hands the
    /// stage behind the rank its own front and decides as the whole pass,
    /// whatever the list held before.
    #[test]
    fn a_front_is_listed_anew_over_a_scratch_that_held_another() {
        let handed = Arc::new(Mutex::new(Vec::new()));
        let front_of = |levels: &[f32]| {
            let probe = WeightedScorer {
                scorer: Box::new(Asked(Arc::clone(&handed))),
                weight: 1.0,
            };
            staged(vec![rank(levels, 4.0, true), probe], Box::new(Best))
        };
        let four: Vec<f32> = (0..16)
            .map(|i| if i < 8 && i % 2 == 0 { 1.0 } else { 0.5 })
            .collect();
        let six: Vec<f32> = (0..16).map(|i| if i < 6 { 1.0 } else { 0.5 }).collect();
        let (four, six) = (front_of(&four), front_of(&six));
        let pool = ranked_pool(16);
        let said = PoolFacts {
            reported: true,
            ..PoolFacts::default()
        };
        let mut scratch = Scratch::new();
        // The targets by position, and the rows the stage behind was handed.
        let mut asked = |engine: &Engine| {
            let facts = request(7);
            let decision =
                engine.decide(&facts, &said, &pool, &Constraints::default(), &mut scratch);
            let rows = handed.lock().expect("the probe's rows").clone();
            assert_eq!(
                positions(&decision),
                positions(&whole_pass(engine, &facts, &pool))
            );
            (positions(&decision), rows)
        };
        assert_eq!(asked(&four), (vec![6, 4, 2, 0], vec![0, 2, 4, 6]));
        assert_eq!(asked(&six), (vec![5, 4, 3, 2], vec![0, 1, 2, 3, 4, 5]));
        assert_eq!(asked(&four), (vec![6, 4, 2, 0], vec![0, 2, 4, 6]));
    }

    /// A stage of the tests that answers for a front alone and notes the
    /// two lists each ask by the front entry hands it: the kept rows, and
    /// the rows asked.
    #[derive(Debug)]
    struct Shown(Arc<Mutex<Vec<[Vec<u32>; 2]>>>);

    impl Lifecycle for Shown {}

    impl Scorer for Shown {
        fn name(&self) -> &'static str {
            "shown"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            for (o, &row) in out.iter_mut().zip(kept) {
                *o = candidates[row as usize].position as f32 / 16.0;
            }
            TieBand::NORMALIZED
        }

        fn front_wise(&self) -> bool {
            true
        }

        fn score_front(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            asked: &[u32],
            out: &mut [f32],
        ) -> (TieBand, ScoreFront) {
            let lists = [kept.to_vec(), asked.to_vec()];
            self.0.lock().expect("the asks").push(lists);
            (
                self.score(facts, candidates, asked, out),
                ScoreFront::Unsaid,
            )
        }
    }

    /// A stage behind a rank is shown the rows the filters kept beside the
    /// rows in front, and no row a filter dropped. Of twenty rows sixteen
    /// are ready: the first rank leaves eight of them in front and the
    /// second four of those, and the stage behind both is asked of the
    /// four beside the sixteen.
    #[test]
    fn a_stage_behind_a_rank_is_shown_the_kept_rows_beside_the_rows_in_front() {
        let shown = Arc::new(Mutex::new(Vec::new()));
        let pool: Vec<Candidate> = (0..20)
            .map(|i| Candidate {
                health: Health {
                    healthy: i < 16,
                    ..Health::default()
                },
                ..candidate(i, 0)
            })
            .collect();
        let first: Vec<f32> = (0..20).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
        let second: Vec<f32> = (0..20).map(|i| if i < 4 { 1.0 } else { 0.5 }).collect();
        let probe = WeightedScorer {
            scorer: Box::new(Shown(Arc::clone(&shown))),
            weight: 1.0,
        };
        let stages = vec![rank(&first, 12.0, true), rank(&second, 3.0, true), probe];
        let engine = staged(stages, Box::new(Best));
        let decision = reported(&engine, &request(9), &pool);
        let kept: Vec<u32> = (0..16).collect();
        assert_eq!(
            *shown.lock().expect("the asks"),
            vec![[kept, vec![0, 1, 2, 3]]]
        );
        assert_eq!(positions(&decision), vec![3, 2, 1, 0]);
    }

    /// A stage of the tests that answers row by row, and says it must be
    /// asked whole where it is handed fewer rows than its count. It notes
    /// how many rows each ask handed it.
    #[derive(Debug)]
    struct Wholly(Arc<Mutex<Vec<usize>>>, usize);

    impl Lifecycle for Wholly {}

    impl Scorer for Wholly {
        fn name(&self) -> &'static str {
            "wholly"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            self.0.lock().expect("the asks").push(kept.len());
            for (o, &row) in out.iter_mut().zip(kept) {
                *o = candidates[row as usize].position as f32 / 16.0;
            }
            TieBand::NORMALIZED
        }

        fn front_wise(&self) -> bool {
            true
        }

        fn score_front(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            _: &[u32],
            asked: &[u32],
            out: &mut [f32],
        ) -> (TieBand, ScoreFront) {
            if asked.len() < self.1 {
                self.0.lock().expect("the asks").push(asked.len());
                return (TieBand::NORMALIZED, ScoreFront::Whole);
            }
            (
                self.score(facts, candidates, asked, out),
                ScoreFront::Unsaid,
            )
        }
    }

    /// A stage behind a rank that says it must be asked whole is asked
    /// again of every row, and the decision is the weighted sum's: the rows
    /// in front decide alone only where the stage answers for them alone.
    #[test]
    fn a_stage_that_must_be_asked_whole_is_asked_of_every_row() {
        let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
        let levels: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
        let decided = |least: usize| {
            let asks = Arc::new(Mutex::new(Vec::new()));
            let stage = WeightedScorer {
                scorer: Box::new(Wholly(Arc::clone(&asks), least)),
                weight: 1.0,
            };
            let engine = staged(vec![rank(&levels, 4.0, true), stage], Box::new(Best));
            assert_ne!(engine.marks[0].0 & Marks::RANKED, 0);
            let decision = reported(&engine, &request(3), &pool);
            let asks = asks.lock().expect("the asks").clone();
            (positions(&decision), asks)
        };
        // It answers for the eight in front alone: asked once, of them.
        let (alone, asks) = decided(8);
        assert_eq!(asks, vec![8]);
        // It must be asked whole: asked of the eight, then of every row,
        // and the targets are the same four.
        let (whole, asks) = decided(16);
        assert_eq!(asks, vec![8, 16]);
        assert_eq!(whole, alone);
        assert_eq!(whole, vec![7, 6, 5, 4]);
    }

    /// A rank of the tests that says it must be asked whole and writes
    /// nothing then: its plain answer is the level named for a row's
    /// position.
    #[derive(Debug)]
    struct Shy(Vec<f32>);

    impl Lifecycle for Shy {}

    impl Scorer for Shy {
        fn name(&self) -> &'static str {
            "shy"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            for (o, &row) in out.iter_mut().zip(kept) {
                *o = self.0[candidates[row as usize].position as usize];
            }
            TieBand(0.0)
        }

        fn spread(&self) -> ScoreSpread {
            ScoreSpread::Levels(0.5)
        }

        fn score_front(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            _: &[u32],
            _: &mut [f32],
        ) -> (TieBand, ScoreFront) {
            (TieBand(0.0), ScoreFront::Whole)
        }
    }

    /// A pool the overload veto empties is put to the binding's all-vetoed
    /// rule by a decision asked by rank, as by the whole pass: the filter
    /// half is one, and so is the plan's home for an emptied pool. The rule
    /// steers the request by default, and sheds it where the binding says so.
    #[test]
    fn a_decision_by_rank_puts_an_all_vetoed_pool_to_the_bindings_rule() {
        let mut engine = preset("cache_aware");
        assert_ne!(engine.marks[0].0 & Marks::RANKED, 0);
        let mut pool: Vec<Candidate> = (0..32).map(|i| candidate(i, i % 5)).collect();
        for c in &mut pool {
            c.health.overloaded = true;
        }
        let facts = RequestFacts {
            prompt_tokens: 512,
            now_micros: 1_000_000,
            ..request(7)
        };
        let steered = reported(&engine, &facts, &pool);
        assert!(steered.pick().is_some(), "{steered:?}");
        assert_eq!(steered, whole_pass(&engine, &facts, &pool));
        engine.shed_when_all_vetoed(0);
        let shed = reported(&engine, &facts, &pool);
        assert!(shed.pick().is_none(), "{shed:?}");
        assert_eq!(shed, whole_pass(&engine, &facts, &pool));
    }

    /// A stage of the tests that orders nothing and notes how many rows
    /// each ask handed it.
    #[derive(Debug)]
    struct Still(Arc<Mutex<Vec<usize>>>);

    impl Lifecycle for Still {}

    impl Scorer for Still {
        fn name(&self) -> &'static str {
            "still"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            self.0.lock().expect("the asks").push(kept.len());
            out.fill(1.0);
            TieBand(0.0)
        }

        fn spread(&self) -> ScoreSpread {
            ScoreSpread::Alike
        }
    }

    /// A profile of ten stages: a rank, seven stages that order nothing, a
    /// ninth that says its front and a tenth that orders nothing. The build
    /// reads eight stages as ranks, so the ninth is none and its word
    /// leaves no fewer rows in front: the tenth is asked of the eight rows
    /// the leading rank put there, and the decision is the weighted sum's.
    /// A stage past the eighth has no bit of the build's to be read by.
    /// Read by its index, the ninth stopped a debug build, and a release
    /// build read the leading rank's bit and asked the tenth of four rows.
    #[test]
    fn a_stage_past_the_eighth_that_says_its_front_is_no_rank() {
        let lead: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
        let last: Vec<f32> = (0..16)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.5 })
            .collect();
        let handed = Arc::new(Mutex::new(Vec::new()));
        let mut stages = vec![rank(&lead, 8.0, true)];
        stages.extend((0..7).map(|_| WeightedScorer {
            scorer: Box::new(Flat(Arc::default())),
            weight: 1.0,
        }));
        stages.push(rank(&last, 1.0, true));
        stages.push(WeightedScorer {
            scorer: Box::new(Still(Arc::clone(&handed))),
            weight: 1.0,
        });
        let engine = staged(stages, Box::new(Best));
        assert_ne!(engine.marks[0].0 & Marks::RANKED, 0);
        assert_eq!(usize::from(engine.orders[0].ranks), Order::MOST);
        let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
        let facts = request(5);
        let ranked = reported(&engine, &facts, &pool);
        assert_eq!(*handed.lock().expect("the asks"), vec![8]);
        // The four rows at both ceilings tie, and the seed draws their order.
        let mut targets = positions(&ranked);
        targets.sort_unstable();
        assert_eq!(targets, vec![0, 2, 4, 6]);
        assert_eq!(
            positions(&ranked),
            positions(&whole_pass(&engine, &facts, &pool))
        );
    }

    /// A level stage of the tests that answers among the rows it is handed:
    /// the last of them reads the ceiling. It does not say that it may be
    /// asked of a front alone, and notes how many rows each ask handed it.
    #[derive(Debug)]
    struct Among(Arc<Mutex<Vec<usize>>>);

    impl Lifecycle for Among {}

    impl Scorer for Among {
        fn name(&self) -> &'static str {
            "among"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            self.0.lock().expect("the asks").push(kept.len());
            out.fill(0.5);
            if let Some(last) = out.last_mut() {
                *last = 1.0;
            }
            TieBand(0.0)
        }

        fn spread(&self) -> ScoreSpread {
            ScoreSpread::Levels(0.5)
        }
    }

    /// A rank that parts the rows and may not put them in front, since a
    /// rank behind it answers among the rows it is handed: every row is
    /// asked on, and the decision is the weighted sum's. The build's word is
    /// read where the rows part.
    #[test]
    fn a_rank_that_may_not_put_rows_in_front_asks_every_row_on() {
        let asks = Arc::new(Mutex::new(Vec::new()));
        let lead: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
        let early: Vec<f32> = (0..16).map(|i| 1.0 - i as f32 / 16.0).collect();
        let stages = vec![
            rank(&lead, 8.0, true),
            WeightedScorer {
                scorer: Box::new(Among(Arc::clone(&asks))),
                weight: 2.0,
            },
            WeightedScorer {
                scorer: Box::new(Valued(early, 0.0)),
                weight: 0.5,
            },
        ];
        let engine = staged(stages, Box::new(Best));
        assert_ne!(engine.marks[0].0 & Marks::RANKED, 0);
        // The second rank may put rows in front, the first may not.
        assert_eq!((engine.orders[0].ranks, engine.orders[0].fronts), (2, 0b10));
        let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
        let facts = request(11);
        let ranked = reported(&engine, &facts, &pool);
        assert_eq!(*asks.lock().expect("the asks"), vec![16]);
        assert_eq!(positions(&ranked), vec![0, 1, 2, 3]);
        assert_eq!(
            positions(&ranked),
            positions(&whole_pass(&engine, &facts, &pool))
        );
    }

    /// A host's picker that answers a built-in's name and reads every score:
    /// the lowest first.
    #[derive(Debug)]
    struct Namesake;

    impl Lifecycle for Namesake {}

    impl Picker for Namesake {
        fn name(&self) -> &'static str {
            "best"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Lowest.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// A picker is handed a front alone by its own word
    /// ([`Picker::front_wise`]), never by its name: a host's picker that
    /// answers `best` and reads every score is handed every row, and its
    /// profile is not decided by rank.
    #[test]
    fn a_picker_is_handed_a_front_by_its_word_and_not_by_its_name() {
        let lead: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
        let stages = |probe: &Arc<Mutex<Vec<u32>>>| {
            vec![
                rank(&lead, 4.0, true),
                WeightedScorer {
                    scorer: Box::new(Asked(Arc::clone(probe))),
                    weight: 1.0,
                },
            ]
        };
        let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
        let facts = request(13);
        let handed = Arc::new(Mutex::new(Vec::new()));
        let host = staged(stages(&handed), Box::new(Namesake));
        assert_eq!(host.marks[0].0 & Marks::RANKED, 0);
        // The lowest sums are the rows behind the rank's ceiling.
        assert_eq!(
            positions(&reported(&host, &facts, &pool)),
            vec![8, 9, 10, 11]
        );
        assert_eq!(handed.lock().expect("the probe's rows").len(), 16);
        let built = staged(stages(&handed), Box::new(Best));
        assert_ne!(built.marks[0].0 & Marks::RANKED, 0);
        assert_eq!(
            positions(&reported(&built, &facts, &pool)),
            vec![7, 6, 5, 4]
        );
        assert_eq!(handed.lock().expect("the probe's rows").len(), 8);
    }

    /// A stage of the tests that says it orders nothing and returns a band.
    #[derive(Debug)]
    struct Banded;

    impl Lifecycle for Banded {}

    impl Scorer for Banded {
        fn name(&self) -> &'static str {
            "banded"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(1.0);
            TieBand(0.5)
        }

        fn spread(&self) -> ScoreSpread {
            ScoreSpread::Alike
        }
    }

    /// A stage that says it orders nothing returns no band
    /// ([`ScoreSpread::Alike`]): a decision by rank does not ask it, so a band
    /// of its own would widen the whole pass's ties and no other's. One that
    /// returns a band fails a debug build where it is asked: its author's case,
    /// never two decisions in the field.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "banded orders nothing and returned a band")]
    fn a_stage_that_orders_nothing_and_returns_a_band_fails_a_debug_build() {
        let lead: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.5 }).collect();
        let stages = vec![
            rank(&lead, 4.0, true),
            WeightedScorer {
                scorer: Box::new(Banded),
                weight: 1.0,
            },
        ];
        let engine = staged(stages, Box::new(Best));
        let pool: Vec<Candidate> = (0..16).map(|i| candidate(i, 0)).collect();
        whole_pass(&engine, &request(17), &pool);
    }

    /// The stage of the case above behind a stage that orders the rows: by
    /// rank it is asked, and the decision ends at its band.
    #[cfg(debug_assertions)]
    fn banded_behind(levels: &[f32]) {
        let stages = vec![
            rank(levels, 8.0, true),
            WeightedScorer {
                scorer: Box::new(Asked(Arc::default())),
                weight: 1.0,
            },
            WeightedScorer {
                scorer: Box::new(Banded),
                weight: 1.0,
            },
        ];
        let engine = staged(stages, Box::new(Best));
        assert_eq!(engine.orders[0].ranks, 1);
        reported(&engine, &request(7), &ranked_pool(9));
    }

    /// Behind a stage that orders the rows, a stage that says it orders
    /// nothing is asked by rank: of the rows in front where four of nine
    /// stand at the rank's ceiling. A band of its own fails a debug build
    /// there too.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "banded orders nothing and returned a band")]
    fn a_band_from_a_stage_that_orders_nothing_fails_a_debug_build_behind_a_front() {
        banded_behind(&[1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5]);
    }

    /// And of every row where three of nine stand at the rank's ceiling,
    /// fewer than a decision's targets.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "banded orders nothing and returned a band")]
    fn a_band_from_a_stage_that_orders_nothing_fails_a_debug_build_where_every_row_is_asked_on() {
        banded_behind(&[1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]);
    }

    /// A stage of the tests that says it orders nothing and returns no band
    /// where every row is asked. Handed a front, it returns a band beside
    /// the word that it must be asked whole.
    #[derive(Debug)]
    struct BandedWhole;

    impl Lifecycle for BandedWhole {}

    impl Scorer for BandedWhole {
        fn name(&self) -> &'static str {
            "banded-whole"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(1.0);
            TieBand(0.0)
        }

        fn spread(&self) -> ScoreSpread {
            ScoreSpread::Alike
        }

        fn score_front(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            _: &[u32],
            out: &mut [f32],
        ) -> (TieBand, ScoreFront) {
            out.fill(1.0);
            (TieBand(0.5), ScoreFront::Whole)
        }
    }

    /// The check stands ahead of the way out of the rows in front. A stage
    /// that says it orders nothing and answers that it must be asked whole
    /// fails a debug build for the band beside that word, although that
    /// band is not read and its answer where every row is asked holds none.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "banded-whole orders nothing and returned a band")]
    fn a_band_beside_the_word_to_be_asked_whole_fails_a_debug_build() {
        let stages = vec![
            rank(&[1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.5], 8.0, true),
            WeightedScorer {
                scorer: Box::new(Asked(Arc::default())),
                weight: 1.0,
            },
            WeightedScorer {
                scorer: Box::new(BandedWhole),
                weight: 1.0,
            },
        ];
        let engine = staged(stages, Box::new(Best));
        assert_eq!(engine.orders[0].ranks, 1);
        reported(&engine, &request(7), &ranked_pool(9));
    }

    /// A leading rank that says it must be asked whole was handed every
    /// row: it is asked again plainly, and the decision is the weighted
    /// sum's.
    #[test]
    fn a_leading_rank_that_must_be_asked_whole_is_asked_plainly() {
        let levels = [1.0, 0.5, 1.0, 0.0, 0.5, 1.0, 1.0, 0.0, 1.0, 0.5];
        let values = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.5, 0.0];
        let shy = WeightedScorer {
            scorer: Box::new(Shy(levels.to_vec())),
            weight: 3.0,
        };
        let ranked = staged(vec![shy, valued(&values, 0.0)], Box::new(Best));
        assert_ne!(ranked.marks[0].0 & Marks::RANKED, 0);
        let stages = vec![rank(&levels, 3.0, false), valued(&values, 0.0)];
        let summed = staged(stages, Box::new(Best));
        let pool = ranked_pool(10);
        for seed in 0..16 {
            let facts = request(seed);
            assert_eq!(
                positions(&reported(&ranked, &facts, &pool)),
                positions(&reported(&summed, &facts, &pool)),
                "seed {seed}"
            );
        }
        // The five at the ceiling in the order of the stage behind.
        let led = positions(&reported(&ranked, &request(0), &pool));
        assert_eq!(led, vec![5, 8, 6, 2]);
    }
}
