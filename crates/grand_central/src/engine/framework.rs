//! The framework (design 4.4): profiles of filters, weighted scorers and one
//! picker; plans that sequence profiles; the engine that resolves a request
//! to its binding and decides. A scorer or picker allocates nothing per
//! decision: scratch lives in the per-thread [`Scratch`] the host keeps.

use std::{
    any::Any,
    cell::OnceCell,
    collections::HashMap,
    fmt::Debug,
    sync::{atomic::AtomicU64, Arc},
};

use crate::{
    document::spec::{Assignment, Needs},
    engine::{
        decision::{
            Decision, LiftedVeto, Overlap, PairSide, Pick, Placing, Shed, ShedReason, Survivors,
            Target, Targets, Trace, AFFINITY_COLUMN, BINDING, EC_COMPATIBLE, MAX_TARGETS, NO_STAGE,
            PARTNERED, PARTNERS, PREFILL_GONE, TARGET_WORKER, TRACE_TOP, VISION, WRITTEN_ROWS,
        },
        sticky_sessions::{self, Claim, SessionRule, SessionTable, SessionVerdict},
        view::{
            mean_in_flight, AffinityFacts, AffinityRows, Candidate, Constraints, LoadReportCutoff,
            Phase, PoolFacts, RequestFacts, Session, WorkerId,
        },
        warm_up::WarmUpRule,
    },
    load::signals::SignalCells,
    policies::{
        consistent_hashing::LentRing, prefix_hash::HeadRule, random::Random,
        round_robin::RoundRobin,
    },
    stages::{
        filters::{HasCapacity, NotOverloaded, RowTests},
        lifecycle::Lifecycle,
        pickers::{push_turn, target_of, Best},
        scorers::{FewestSessions, LeastRequests},
    },
    tracking::{
        dispatch_credit::DispatchCredit,
        lease::{EventKind, EventSet, Lease, Membership, Outcome, ReleaseReason, Side, Signal},
    },
    Draw,
};

mod alone;
mod builder;
mod encode;
mod engines;
mod keyed;
mod places;
mod vetoed;

pub use alone::{AloneProfile, InTheWay, NotAlone, Place};
pub use encode::{EncodeItems, EncodePlaced, EncodePool, EncoderFacts};
use engines::{run_engines, run_owned};
pub use keyed::PoolRows;
use vetoed::{AllVetoed, Count, Half};

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
    ///
    /// A row-wise filter that holds no decision scope may be asked about a
    /// row once more behind its decision: a pair's prefill shadow asks the
    /// live decode profile's hard filters so. Its word on a row must not
    /// turn on how often it is asked.
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
    /// A filter that reads the ring its request lends
    /// ([`RequestFacts::ring`]) declares [`Needs::ring`], and a live
    /// profile one of whose scorers places by the ring shows it the ring
    /// and the request's place on it. A pair side's shadow that reads a
    /// ring where neither live side does is lent that ring. Where one of
    /// its filters declares the need, every stage of the shadow's own run
    /// is shown what its lender answers: the ring and the shadow's place
    /// on it, or the facts as they stand where it answers none. So are
    /// the hard filters a prefill's shadow asks once more at its admission
    /// limit's test. Where no filter declares it, the scoring half alone
    /// is: the filters are shown no ring and no place. Neither are they in
    /// the steps of the pair's default that a side's shadow follows,
    /// whatever they declare.
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
#[expect(clippy::exhaustive_structs, reason = "its one value is the record")]
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

    /// Whether the scorer weighs load against affinity by a gate of its
    /// own: it places a request by who holds its prefix, and its gate
    /// bounds where it sends the request by the workers' load. The default
    /// is no.
    ///
    /// Under a binding that steers, a pair's prefill profile with such a
    /// scorer runs with the overload veto lifted
    /// ([`Constraints::overload_lifted`]). The veto would move a request
    /// off its holder each time that worker crosses a threshold, and the
    /// gate already answers a worker loaded too far. What a stage is fed
    /// says nothing of a gate: one that reads the affinity column
    /// ([`Self::needs`]) and answers no here keeps the veto for its
    /// profile.
    ///
    /// Asked where an engine is built, as [`Self::needs`] is: no decision
    /// asks.
    fn balances_load_against_affinity(&self) -> bool {
        false
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
    /// ([`crate::policies::consistent_hashing::LentRing::head_position`])
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
/// use smg_grand_central::engine::Shortlist;
/// let shortlist = Shortlist::new(2, 1);
/// assert_eq!((shortlist.compared, shortlist.alternates), (2, 1));
/// assert_eq!(Shortlist::all(5), Shortlist::new(5, 0));
/// ```
///
/// ```compile_fail
/// use smg_grand_central::engine::Shortlist;
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
/// (the last two paragraphs name the pickers whose draw is asked under it). With
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
///
/// A picker that draws two rows is asked under this size too, from eight
/// rows, by a host that writes a row's record when the decision asks about
/// the row (`WRITTEN_PAIR_FROM`): there a draw saves the records it does
/// not write, and it is asked only where its pick of a seed is the pass's.
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
    /// ([`Picker::takes_turns`]) or a pick of two before the records of the
    /// pool are written ([`DRAW_FROM`]'s last paragraph).
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
    /// held its pick ([`DRAW_BOUND`] refused rows, a smaller bound of the
    /// picker's own, or a sample's draws). A
    /// draw that keeps nobody says
    /// nothing of the pool: the pass alone sheds. `draw` is the seed's stream: the
    /// picker takes its values from it and from nowhere else, and whatever
    /// follows the draw reads the stream past them, or as it came where the
    /// picker drew from a copy to hand out the whole pass's own pick.
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
    /// decode decided again under a scorer: of the call's decode profile
    /// ([`Engine::decide_decode`]), and of the decode's shadow profile asked
    /// beside that call. It reads it at no other decision.
    fn picks_alone(&self) -> bool {
        false
    }

    /// Whether each pick takes the next turn of an order the picker keeps,
    /// as a rotation's does: a second pick for one request spends a turn
    /// that serves no request. Such a picker's draw ([`Picker::draw_first`])
    /// must read no value of the seed's stream: the engine hands it the
    /// request's own seed and asserts in a debug build that none was read.
    /// The default is no. The engine reads it at a pair's decode decided
    /// again, handed the decode of the pair's first decision: of the call's
    /// decode profile ([`Engine::decide_decode`]), and of the decode's
    /// shadow profile asked beside that call. It reads it once at its build
    /// too, for a profile that draws for every request. The row such a
    /// picker's draw lands on is its rule's answer, so the draw is asked at
    /// every pool size over a pool that lends its own word
    /// ([`PoolFacts::word`]), and a session a sessions rule leaves to the
    /// profile ([`Assignment::Delegate`]) takes the pool's turn.
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
#[non_exhaustive]
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

impl<'a> PoolWord<'a> {
    /// A pool's size and its own word. It takes every field whose silence
    /// would change an answer with no word, and neither of these has a neutral
    /// value: a later field with a true neutral value joins by assignment. A
    /// program that implements [`Picker::pick_alone`] builds one to call its
    /// picker. It checks nothing. Code outside this crate starts here: the
    /// record gains fields, so no literal of it builds there.
    #[must_use]
    pub const fn of(rows: usize, word: &'a AtomicU64) -> Self {
        Self { rows, word }
    }
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
#[non_exhaustive]
pub struct WeightedScorer {
    pub scorer: Box<dyn Scorer>,
    pub weight: f32,
}

impl WeightedScorer {
    /// A scorer with its weight: the record's two fields, so nothing is left
    /// for a caller to assign today. Code outside this crate starts here:
    /// the record gains fields, so no literal of it builds there.
    #[must_use]
    pub fn of(scorer: Box<dyn Scorer>, weight: f32) -> Self {
        Self { scorer, weight }
    }

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
#[non_exhaustive]
pub struct Profile {
    pub name: String,
    pub filters: Vec<Box<dyn Filter>>,
    pub scorers: Vec<WeightedScorer>,
    pub picker: Box<dyn Picker>,
}

impl Profile {
    /// A profile of one picker, with no filter and no scorer: a caller
    /// assigns the two lists over it. Code outside this crate starts here:
    /// the record gains fields, so no literal of it builds there.
    #[must_use]
    pub fn of(name: String, picker: Box<dyn Picker>) -> Self {
        Self {
            name,
            filters: Vec::new(),
            scorers: Vec::new(),
            picker,
        }
    }

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
    /// events. One scorer that reads them there has the index name a
    /// single request of token ids alone where the model's index holds a
    /// block, and keeps the index's holders first for every other
    /// request, whatever another reads. A scorer that reads the events for
    /// a row's warm-up reads no holder and turns nothing here.
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

/// Runs `f` where a panic of it is to stop: on a path that is already
/// unwinding, where a second panic let out would end the process, and
/// behind a caught panic its caller resumes, where one let out would take
/// that panic's place. A panic of `f` stops here, and its payload is
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
    /// Whether the last pair decision's decode side read first partners
    /// ([`FirstPartners`]): a pair made among them at its first decision,
    /// or a decode decided again among its held prefill's. The entry that
    /// chooses the partners writes it ([`Engine::decide_pair`],
    /// [`Engine::decide_decode`]), and the decode's shadow reads the same
    /// partners by it ([`Engine::shadow_side`],
    /// [`Engine::shadow_decode_later`]). It is what the decision chose and
    /// is never worked out of its pick: a pick the all-vetoed rule made
    /// past a doubt can name a first partner where every partner was read.
    among_first: bool,
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
    /// The list the last pair's decision or encode decision made with
    /// this scratch shed its request over as saturated
    /// ([`Self::shed_over`]).
    shed_over: Option<ShedOver>,
}

/// The list a decision shed its request over as saturated, by the
/// decision's own word ([`Scratch::shed_over`]). A decision says one under
/// a binding that sheds a list wholly over its thresholds, and under no
/// other. A shed's trace does not say it: a pair's shed made over its
/// decodes carries the prefill's trace, as one made over its prefills
/// does.
///
/// A pair's decision that sheds behind its prefill side's shed is
/// saturated exactly where the list the gateway's own pair selection
/// judges is wholly over its thresholds, by that selection's own count
/// (`vetoed::Half::count`), and it says that list: no such shed is
/// saturated with no word, and none says a word the count does not bear.
///
/// That holds over whole lists. Behind a part of a list
/// ([`PairPools::part`]) the question is not asked, and the stage that
/// ended the pass names the shed. A shed the overload veto named says
/// its list there as it did before the question stood: the prefills
/// that pair where no prefill stands, and else the decode list whole,
/// each where wholly over its thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShedOver {
    /// An encode request's encoders: the overload veto left none of them
    /// open ([`Engine::decide_encode_pair`]).
    Encoders,
    /// A pair's prefills: of the prefills at the request's places that
    /// pair with a decode of the pool none stands as it is, its limit
    /// aside, and they are wholly over their thresholds. A prefill that
    /// pairs with no decode is no part of that list: the gateway's own
    /// selection holds no such prefill ([`vetoed::pairs_in`]).
    Prefills,
    /// A pair's decodes, beside a prefill that pairs with a decode of the
    /// pool and stands: the whole decode list at the request's places is
    /// wholly over its thresholds, a decode that pairs with no prefill
    /// too.
    Decodes,
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
    /// ([`Engine::decide_pair`]). There the pair's default alone writes it
    /// again. Each side's shadow reads on the prefill's buffer whether the
    /// default made the pair, and in which tier. The prefill's shadow
    /// leaves that mark as it found it ([`Engine::shadow_side`]).
    lifted: LiftedVeto,
    /// The pool size every buffer holds room for ([`Buffers::reserve`]).
    room: usize,
    /// Whether nobody reads the best rows of a trace made with these
    /// buffers: the word a front states to the scratch.
    unread: bool,
    /// What a front said of the rows it wrote whole, for the decisions it
    /// asks over them with these buffers ([`Scratch::whole`]).
    whole: WholeRows,
}

/// What a front that wrote a list's every row says of them to the scratch
/// ([`Scratch::whole`]) ahead of the decisions it asks over them: the key
/// its list made once, which a pass would read every row to make. A whole
/// pass reads the word for the very rows it was said of and for no other
/// ([`Self::of`]). The default says nothing of any rows.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WholeRows {
    /// Where the rows stand and how many they are.
    at: usize,
    len: usize,
    /// The key of the rows in their order, as the crate's own rotation
    /// keys the rows a pass kept: the list makes it once.
    key: u64,
}

impl WholeRows {
    /// The word of `rows` as their front wrote them, with the list's `key`.
    #[cfg(feature = "in-process")]
    pub(crate) fn said(rows: &[Candidate], key: u64) -> Self {
        Self {
            at: rows.as_ptr().addr(),
            len: rows.len(),
            key,
        }
    }

    /// The word, where it was said of `candidates`: the very slice, by its
    /// place and its count. The word outlives the decisions it was said
    /// for: it stands until the next borrow of the thread's buffers, which
    /// takes it back before it hands out the scratch
    /// ([`Scratch::no_whole`]). So a pass reads only a word its own front
    /// said on the borrow the pass runs on, behind the write of the rows
    /// it names, and that front holds those rows shared until its
    /// decisions are answered: a slice at the word's place and of its
    /// count is those very rows, as they were written. The count is
    /// compared first: a word of no rows, which is every word under a
    /// document without the rotation, ends there.
    #[inline(always)]
    fn of(self, candidates: &[Candidate]) -> Option<Self> {
        (self.len == candidates.len() && self.len != 0 && self.at == candidates.as_ptr().addr())
            .then_some(self)
    }
}

impl Buffers {
    /// The scores a trace reads its best rows from: none where nobody
    /// reads them, so no pass over the scores runs for them.
    #[inline(always)]
    fn traced(&self) -> &[f32] {
        if self.unread {
            &[]
        } else {
            &self.scores
        }
    }

    /// Takes the word of whether nobody reads the best rows of a trace
    /// made with these buffers, as the scratch states it
    /// ([`Scratch::unread`]). [`Buffers::traced`] reads it.
    #[cfg(feature = "in-process")]
    #[inline(always)]
    fn set_unread(&mut self, unread: bool) {
        self.unread = unread;
    }

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

    /// States whether nobody reads the best rows of a trace made with this
    /// scratch from here on: a single decision kept behind a placement,
    /// which hands out no trace, under a document none of whose stages
    /// takes its issue and that keeps no sample, and the shadow profile
    /// decided behind it. A trace that reads its best rows of the scores
    /// then names none ([`Trace::top_len`] is `0`) and is whole in every
    /// other field; a decision that names its rows without that read names
    /// them as before. The word stands until it is stated again.
    #[cfg(feature = "in-process")]
    #[inline(always)]
    pub(crate) fn unread(&mut self, unread: bool) {
        self.single.set_unread(unread);
    }

    /// States what a front wrote of the rows it hands this scratch's
    /// decisions from here on ([`WholeRows`]): a single request's rows,
    /// written whole. The word stands until it is stated again, and the
    /// borrow of a thread's buffers takes it back where it begins
    /// ([`Self::no_whole`]), as it does [`Self::unread`]'s. Nothing takes
    /// it back behind the front's decisions: between two borrows it
    /// stands unread, as the scratch a front states it to is the calling
    /// thread's, which that borrow alone hands out.
    #[cfg(feature = "in-process")]
    #[inline(always)]
    pub(crate) fn whole(&mut self, rows: WholeRows) {
        self.single.whole = rows;
    }

    /// Takes back the word of [`Self::whole`] by its count alone: a word of
    /// no rows is read for no slice ([`WholeRows::of`]).
    #[cfg(feature = "in-process")]
    #[inline(always)]
    pub(crate) fn no_whole(&mut self) {
        self.single.whole.len = 0;
    }

    /// The list the last pair's decision or encode decision made with
    /// this scratch shed its request over as saturated, where it says
    /// one ([`ShedOver`]). Each such decision takes the word back where it
    /// begins and writes it at the one place it makes that shed, so the
    /// word is that decision's alone: `None` behind its every other
    /// answer. An encode decision that hands its pair's decision a part
    /// of a list says none of the pair's two: that decision judged the
    /// part, or asked no question behind it ([`PairPools::part`]). A
    /// caller reads the word behind a decision's shed as
    /// saturated, and behind nothing else: a request that is answered
    /// ahead of both decisions leaves the word of the one before.
    #[cfg_attr(
        not(any(test, feature = "in-process")),
        expect(
            dead_code,
            reason = "the one reader is the engine over worker lists, built under `in-process`"
        )
    )]
    pub(crate) fn shed_over(&self) -> Option<ShedOver> {
        self.shed_over
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
    score_and(
        profile,
        facts,
        candidates,
        buf,
        targets,
        |kept, scores, band, targets| {
            profile
                .picker
                .pick(facts, candidates, kept, scores, band, targets);
        },
    );
}

/// [`score_and_pick`] with its pick step handed in: `pick` is given the
/// compared set, its scores and the profile's band, and pushes the targets
/// the picker's `pick` would. Inlined into each of its two callers, so the
/// pass of a profile's own `pick` holds no test of the other
/// ([`turn_whole`]).
#[inline(always)]
fn score_and(
    profile: &Profile,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    buf: &mut Buffers,
    targets: &mut Targets,
    pick: impl FnOnce(&[u32], &[f32], TieBand, &mut Targets),
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
    pick(&buf.kept[..compared], &buf.scores, band, targets);
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
    let trace = trace_of(index, facts, pool, candidates, &buf.kept, buf.traced());
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
/// then keeps is among them, under any profile. A pair's decode side that
/// states a number beside its places opens with the one row the number
/// reaches among those its classes keep
/// ([`Constraints::target_worker`]), here and so in every pass that asks
/// this half again: `target-worker` where it reaches none.
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
                return Some(PARTNERS);
            }
            // The decode a pair's request names: the row its number reaches
            // among these, and the side's one row. A number past them
            // leaves no row, and none for the all-vetoed rule to steer to.
            if let Some(number) = cx.target_worker {
                if !named_row(kept, number) {
                    return Some(TARGET_WORKER);
                }
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
                return Some(PARTNERS);
            }
        }
        None => places::open(kept, candidates.len(), cx),
    }
    if kept.is_empty() {
        return Some(NO_STAGE);
    }
    if let Some(p) = partnered {
        match only(p) {
            Some(one) => kept.retain(|&i| candidates[i as usize].pair_class == one),
            None => kept.retain(|&i| classed(p, candidates[i as usize].pair_class)),
        }
        if kept.is_empty() {
            return Some(PARTNERED);
        }
    }
    if encode {
        kept.retain(|&i| candidates[i as usize].ec_compatible);
        if kept.is_empty() {
            return Some(EC_COMPATIBLE);
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
        side: None,
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
    /// ([`Self::ruled`]), as a live decode side and its shadow do. A prefill
    /// side says none, nor does its shadow, a pass that only names a shed's
    /// stage or an encode item's: a pool a veto empties there is shed.
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
    /// (`vetoed`): a pair's live decode side, and that side's shadow. Where
    /// a soft veto leaves such a side nobody, the rule steers it or sheds it
    /// as it does the single plan's pool ([`vetoed::emptied`]), over the
    /// side's own rows.
    const fn ruled(self, vetoed: AllVetoed) -> Self {
        Self { vetoed, ..self }
    }
}

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
    let trace = trace_of(index, facts, pool, candidates, &buf.kept, buf.traced());
    if targets.is_empty() {
        return shed(ShedReason::PolicyDeclined, profile.picker.name(), trace);
    }
    Decision::Pick(Pick {
        targets,
        trace,
        session_verdict: SessionVerdict::None,
    })
}

/// The scoring half of a whole pass that kept every row of a list under
/// the crate's own rotation ([`TURNS_ITS_SET`]), where the front that wrote
/// those very rows said the list's key ([`WholeRows`]): what
/// [`score_survivors`] answers with no `first`, with the rotation's turn
/// taken by `set`, that key. The rotation's `pick` takes the turn of the
/// same set by the same key, which it makes by reading every row kept
/// ([`push_turn`] hands out the rows for both); the list made it once. A
/// picker that narrowed the rows is asked for its `pick` as ever.
///
/// Out of line, and with the end of [`score_survivors`] written again:
/// the pass of any other profile holds one test of the scratch's word and
/// nothing of this ([`run_whole`]). By count, that end shared as one
/// function that takes the targets cost every whole pass a copy of them,
/// 30 instructions.
#[inline(never)]
fn turn_whole(
    profile: &Profile,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    buf: &mut Buffers,
    set: u64,
) -> Decision {
    let mut targets = Targets::new();
    score_and(
        profile,
        facts,
        candidates,
        buf,
        &mut targets,
        |kept, scores, band, targets| {
            if kept.len() != candidates.len() {
                return profile
                    .picker
                    .pick(facts, candidates, kept, scores, band, targets);
            }
            debug_assert_eq!(
                set,
                RoundRobin::set_key(candidates, kept),
                "the list's key is the key of every row kept"
            );
            let turn = profile.picker.pick_among(set, kept.len());
            push_turn(candidates, kept, turn, targets);
        },
    );
    let trace = trace_of(index, facts, pool, candidates, &buf.kept, buf.traced());
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
/// where its profile keeps such a decode ([`keeps_its_first`],
/// [`Engine::decide_decode`]): the scoring half keeps that decode where
/// it is a survivor ([`score_survivors`]). That call's shadow hands the
/// decode to its own run the same way ([`run_side_lent`],
/// [`Engine::shadow_decode_later`]).
/// Inlined into each, so a run that hands none is the code it was.
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

/// `facts` with the ring and the places a lender answered, where it
/// answered any. A shadow's run ([`run_side_lent`]) and the admission
/// limit's test behind it (`shadow_side`) both state the request through
/// this, so the test judges on the facts the run showed its filters.
fn with_lent<'f>(
    facts: &RequestFacts<'f>,
    lent: Option<(&'f dyn LentRing, &'f [u64])>,
) -> Option<RequestFacts<'f>> {
    lent.map(|(ring, places)| RequestFacts {
        ring: Some(ring),
        ring_positions: places,
        ..*facts
    })
}

/// [`run_side`] for a pair side's shadow, which may be stated a place of
/// its own on the ring its request lends ([`Engine::side_ring_profile`]).
/// Under a profile none of whose filters declares the ring need
/// ([`Filter::needs`]), `lent` is asked between the two halves, behind the
/// filter half. A set the filters emptied ends where every side's does
/// ([`single_emptied`]), in the all-vetoed rule's pick or in a shed, and
/// asks nothing: no stage would read the place. The filter half reads
/// `facts` as they stand, with no ring and no place, and the scoring half
/// reads them with the ring and the places `lent` answers, where it
/// answers any. Under a profile with such a filter (`ahead`), `lent` is
/// asked ahead of the filter half and both halves read what it answers,
/// as the stages of a live profile do, or `facts` as they stand where it
/// answers none: the lender is then asked for a set the filters may
/// empty. Out of line: a shadow's run alone enters here.
#[expect(
    clippy::too_many_arguments,
    reason = "a side's run, beside what its shadow is lent"
)]
fn run_side_lent<'l>(
    profile: &Profile,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    set: Set<'_>,
    cx: &Constraints<'_>,
    buf: &mut Buffers,
    first: Option<(WorkerId, u16)>,
    ahead: bool,
    lent: &dyn Fn() -> Option<(&'l dyn LentRing, &'l [u64])>,
) -> Decision {
    let candidates = set.candidates;
    let state = || with_lent(facts, lent());
    let stated = if ahead { state() } else { None };
    let shown = stated.as_ref().unwrap_or(facts);
    if let Some(emptied_by) = filter(profile, shown, cx, buf, set) {
        return single_emptied(index, shown, pool, candidates, emptied_by, buf);
    }
    let stated = if ahead { stated } else { state() };
    let facts = stated.as_ref().unwrap_or(facts);
    score_survivors(profile, index, facts, pool, candidates, buf, first)
}

/// Whether a decode decided again under `profile` keeps the decode its
/// pair's first decision named, where the call is handed one: its picker
/// takes turns, under no scorer or one the picker does not read. The live
/// call asks it of its decode profile ([`Engine::decide_decode`]), and
/// that call's shadow of its own ([`Engine::shadow_decode_later`]).
#[inline]
fn keeps_its_first(profile: &Profile) -> bool {
    profile.picker.takes_turns() && (profile.scorers.is_empty() || profile.picker.picks_alone())
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
                // The front that wrote these very rows whole said the
                // list's key ([`WholeRows`]): where the fold kept every
                // row and the profile's picker is the crate's own
                // rotation, the pass turns the set by it. The word is
                // asked first: no front states one under a document
                // without the rotation, whose pass pays that one test.
                let turned = buf
                    .whole
                    .of(candidates)
                    .filter(|_| engine.set_turns[at] && buf.kept.len() == candidates.len());
                if let Some(whole) = turned {
                    return turn_whole(profile, index, facts, pool, candidates, buf, whole.key);
                }
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
/// that refused them, and skew the picker's law on that path. One draw
/// reads a copy of the stream instead: a pick of two over a pool under
/// [`DRAW_FROM`] rows, which is asked only where every row is expected
/// kept and gives up at its first refused row. There the draw's pick of a
/// seed is the pass's own, so the pass behind it decides as if nothing had
/// been drawn and no seed's pick moves. The trace
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
            let trace = trace_drawn(index, facts, pool, candidates, &buf.kept, buf.traced());
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

/// The crate's own picker that keys a rotation by the rows a whole pass
/// kept, by name: its `pick` takes the turn of the set of those rows
/// ([`Picker::pick_among`]) and hands out the row at the turn and the rows
/// kept behind it. In a document a host's stage takes no built-in's name
/// and states its own ([`StageRegistry`](crate::stages::StageRegistry)), so
/// there a picker of this name is the crate's. An engine built in code
/// holds whatever picker its host handed it, and no word a picker says of
/// itself tells that its turn among a set is the turn its `pick` takes: so
/// the name is read only for an engine whose builder says a catalog made
/// its pickers ([`Engine::made_by_a_catalog`]).
const TURNS_ITS_SET: &str = "round-robin";

impl Engine {
    /// States that a catalog made this engine's every picker, as the
    /// document road does: a picker of the rotation's name is then the
    /// crate's own ([`TURNS_ITS_SET`]), and a whole pass that kept every
    /// row of a list its front wrote whole turns the set by the key the
    /// list made once ([`turn_whole`]). An engine built in code states
    /// nothing: its picker of that name is asked for its `pick`, as ever.
    pub(crate) fn made_by_a_catalog(&mut self) {
        for (turns, profile) in self.set_turns.iter_mut().zip(&self.profiles) {
            *turns = profile.picker.name() == TURNS_ITS_SET;
        }
        #[cfg(feature = "in-process")]
        {
            self.turns_a_set = self.set_turns.contains(&true);
        }
    }

    /// Whether a profile's picker is the crate's own rotation by its
    /// builder's word ([`Self::made_by_a_catalog`]): a front that wrote a
    /// list's rows whole says them to the scratch under such a document,
    /// and under no other ([`Scratch::whole`]).
    #[cfg(feature = "in-process")]
    #[inline(always)]
    pub(crate) fn turns_a_set(&self) -> bool {
        self.turns_a_set
    }
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

/// The crate's own pickers, by name, that read no count a worker object
/// alone holds: neither a row's requests in flight nor a count of its
/// sessions. A profile of the engine's own filters, no scorer and one of
/// these reads a row's fixed half, its report and its vetoes, and nothing
/// else of it ([`Engine::rows_stand`]). A picker of one of these names is
/// the crate's, as one of [`DRAWS_BY_ITS_TEST`] is.
#[cfg(feature = "in-process")]
const PICKS_BY_NO_COUNT: [&str; 2] = ["random", "round-robin"];

/// The crate's own pickers that draw two rows and compare them, by name.
/// Such a picker's draw and its whole pass place a request by one rule at
/// every pool size: two distinct uniform rows among the rows every filter
/// keeps, the better one leading and the other the alternate. With every
/// row kept the two draw the same two rows of a seed, and the better leads
/// in both. Two that tie are ordered by the request's own seed in a pool
/// under [`DRAW_FROM`] rows, by the draw as by the pass: there a seed's pick
/// is one pick, whichever of the two made it. From that size on, where the
/// pass decides only behind a draw that gave up, the draw orders two that
/// tie by a later value of its stream. With a row refused the pass draws
/// among the survivors and the draw among every row, so the two no longer
/// hand out one pick of a seed: under [`DRAW_FROM`] rows the draw gives the
/// pool to the pass at its first refused row and leaves the stream as it
/// came, and from that size on it draws again, as a draw over rows handed
/// written does. No other picker of [`DRAWS_BY_ITS_TEST`] is
/// listed: under
/// [`DRAW_FROM`] rows a sample's draw compares fewer rows than its pass, a
/// draw of one row hands out fewer alternates behind refused rows, and a
/// turn's draw is asked there by its own rule.
#[cfg(feature = "in-process")]
const DRAWS_TWO: [&str; 2] = ["power-of-two", "power-of-two-count-first"];

/// One side of a pair asked by its draw: the
/// fixed rows of its list, the buffer a record is written into, and the
/// writer of one record.
#[cfg(feature = "in-process")]
pub(crate) type DrawnSide<'a> = (
    &'a [Candidate],
    &'a mut [Candidate],
    &'a mut dyn FnMut(&Candidate, &mut Candidate),
);

/// What [`Engine::pair_drawn_for`] answers: the pair's two profiles.
#[cfg(feature = "in-process")]
#[derive(Clone, Copy)]
pub(crate) struct PairDrawn {
    profiles: [u16; 2],
}

/// The smallest pool over which a picker that draws two rows
/// ([`DRAWS_TWO`]) is asked to draw before the pool's records are written
/// ([`Engine::drawn_for`]), in a document whose every profile draws first
/// under such a picker and keeps a row by the engine's one folded test of
/// it (`RowTests`). Beside a profile of any other kind a pair's draw keeps
/// [`DRAW_FROM`]: the least floor would have that profile's request
/// resolved twice over a small pool, and a filter the engine does not fold
/// may refuse a row by a fact no word of the list says. A pool whose host
/// lists its engines is not meant under `power-of-two`, which decides a
/// pair over them at any size; `power-of-two-count-first` names no unit
/// and draws that pool's rows from [`DRAW_FROM`] on. A host that
/// writes a row's record when the decision asks about the row pays the
/// whole pass one write a row beside the pass's own reading, and the draw
/// the rows it lands on: so its draw wins under [`DRAW_FROM`] rows, where a
/// draw over rows handed written does not. Eight is the size
/// [`DRAW_FROM`]'s run names for a pool with every row ready. A pool of
/// fewer rows is written whole and read by the pass. Under
/// [`DRAW_FROM`] rows the host of the records asks the draw only where it
/// expects every row kept: of a request that names no worker tried or
/// excluded, carries no session key, is held to no places and stands under
/// no veto of its host,
/// over a list whose every worker publishes its routing state available
/// (`PoolRecords::stands`) and that holds no engine's ranks
/// (`PoolRecords::has_ranks`). Every other decision there is written whole
/// and read by the pass: no refused row costs a draw, and the pick of a
/// seed is the pass's in each. What it costs: a worker whose state moves
/// behind the list's answer may still refuse a row to the draw, which
/// gives up at that row, and the whole pass runs behind the rows it asked
/// about.
#[cfg(feature = "in-process")]
pub(super) const WRITTEN_PAIR_FROM: usize = 8;

// The written floor of a pair's draw stands under the floor of a draw over
// rows handed written, and holds the two rows it draws.
#[cfg(feature = "in-process")]
const _: () = assert!(2 <= WRITTEN_PAIR_FROM && WRITTEN_PAIR_FROM <= DRAW_FROM);

/// A plain single decision that is a draw's, as [`Engine::drawn_for`] found
/// it: the profile that draws and its binding's word for a pool the
/// overload veto empties.
#[cfg(feature = "in-process")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Drawn {
    index: u16,
    vetoed: AllVetoed,
    /// Whether the draw places a keyed request under a sessions rule and
    /// claims its worker for the key ([`Engine::decide_claimed`]).
    claims: bool,
}

#[cfg(feature = "in-process")]
impl Drawn {
    /// Whether the draw is a keyed request's under a sessions rule: its
    /// host asks [`Engine::decide_claimed`], which claims behind the draw.
    pub(crate) const fn claims(self) -> bool {
        self.claims
    }
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
    index: u16,
    vetoed: AllVetoed,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    fixed: &[Candidate],
    rows: &mut [Candidate],
    write: &mut dyn FnMut(&Candidate, &mut Candidate),
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Decision {
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
            let trace = trace_drawn(index, facts, pool, rows, &buf.kept, buf.traced());
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

/// One side of a pair by its draw ([`Engine::decide_pair_drawn`]): the two
/// rows the seed's stream draws over `fixed`, each written at its place
/// (`write`) and put to the profile's tests, then scored and ordered as
/// [`run_drawn_written`] orders a single's.
///
/// The whole pass decides a pair's side at every pool size, so the draw
/// keeps the pass's rule at every size, as a single's does under
/// [`DRAW_FROM`] rows alone: the request's seed is read as it came, so two
/// rows that tie are ordered by it, and the first row a test refuses ends
/// the draw. `None` where the draw names no pick: the caller decides the
/// pair whole, by a stream nothing was drawn from.
#[cfg(feature = "in-process")]
#[expect(
    clippy::too_many_arguments,
    reason = "the draw's inputs and the host's two halves of a row: the fixed rows, the buffer and the write"
)]
#[inline(never)]
fn run_side_drawn_written(
    engine: &Engine,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    fixed: &[Candidate],
    rows: &mut [Candidate],
    write: &mut dyn FnMut(&Candidate, &mut Candidate),
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Option<Pick> {
    let at = usize::from(index);
    let (profile, tests) = (&engine.profiles[at], engine.row_tests[at]);
    debug_assert_eq!(fixed.len(), rows.len(), "one buffer row a fixed row");
    #[cfg(debug_assertions)]
    for (row, fixed) in rows.iter_mut().zip(fixed) {
        *row = unwritten(fixed);
    }
    // A request held to places is the whole pass's.
    if places::stated(cx) {
        return None;
    }
    buf.reserve(fixed.len());
    buf.kept.clear();
    let masked = tests.is_some_and(|tests| tests.masked(cx));
    let mut refused = false;
    // The row a draw lands on, written at its place: the written record
    // is the one the tests are asked about.
    let mut ask = |fixed: &Candidate| {
        let kept = rows.get_mut(fixed.position as usize).is_some_and(|row| {
            write(fixed, row);
            let row = &*row;
            match tests {
                Some(tests) => tests.keeps(facts, row, cx) && !(masked && cx.masks(row.position)),
                None => profile.filters.iter().all(|f| f.keep(facts, row, cx)),
            }
        });
        refused |= !kept;
        kept
    };
    // The stream is the draw's own: the order below reads the seed of
    // `facts`, which no draw moved.
    let mut draw = Draw::new(facts.seed);
    let drew = profile
        .picker
        .draw_first(facts, pool, fixed, &mut ask, &mut buf.kept, &mut draw);
    if !drew || refused {
        return None;
    }
    let band = combine(
        profile,
        facts,
        rows,
        &buf.kept,
        &mut buf.scores,
        &mut buf.stage,
    );
    let mut targets = Targets::new();
    profile
        .picker
        .order_drawn(facts, rows, &buf.kept, &buf.scores, band, &mut targets);
    if targets.is_empty() {
        return None;
    }
    let trace = trace_drawn(index, facts, pool, rows, &buf.kept, buf.traced());
    Some(Pick {
        targets,
        trace,
        session_verdict: SessionVerdict::None,
    })
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
#[non_exhaustive]
pub enum Plan {
    /// One profile over one pool.
    #[default]
    Single,
    /// The prefill profile over the open prefills, then the decode profile
    /// over the chosen prefill's partners.
    PdPair,
    /// The pair as under `PdPair`, then the encode profile named here over
    /// the open encoders of the chosen prefill's class that take the
    /// request's media, once per encode item or once for the items taken
    /// as one. A request with items that no encoder is open to is refused
    /// ahead of the pair. The pair of a request with items is picked among
    /// the prefills whose class holds an open encoder that takes its media.
    /// A decode without the vision a request asks of it is no candidate of
    /// the pair either. The binding's `profile` is the prefill side and
    /// `decode_profile` the decode side. The encode side rides the plan,
    /// so a binding keeps the pair's two fields.
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

/// The live profiles of a pair binding's two sides, by position, the
/// prefill's first. A binding that names no decode profile runs its one
/// profile on both sides.
fn live_sides(binding: &Binding) -> [u16; 2] {
    let prefill = binding.profile;
    [prefill, binding.decode_profile.unwrap_or(prefill)]
}

/// The live profiles the shadow of `side` is held against for the ring its
/// request lends ([`Engine::unlent`]): both of `lives` where one list of
/// places is both sides', and the side's own where each side is stated
/// its own ([`RequestFacts::decode_ring_positions`]), whose list was
/// stated by that side's word.
fn held_to<'l>(lives: &'l [u16; 2], side: PairSide, facts: &RequestFacts<'_>) -> &'l [u16] {
    match facts.decode_ring_positions {
        Some(_) => std::slice::from_ref(&lives[side.index()]),
        None => lives,
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

/// A binding: which profile (or plan of profiles) a request resolves to.
#[derive(Clone, Debug)]
#[non_exhaustive]
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
    /// a pair the rule covers, by the side's own session. A document states
    /// the rule: code outside this crate builds none ([`SessionRule`]), so
    /// a binding it makes leaves this `None`, or takes the rule of a
    /// document's binding ([`Engine::binding`]).
    pub sessions: Option<SessionRule>,
}

impl Binding {
    /// The binding of the models `model` names to the profile at `profile`,
    /// as a document's binding that states those two and nothing else:
    /// every phase, the `single` plan, no shadow and no session rule. A
    /// caller assigns the rest over it. Code outside this crate starts
    /// here: the record gains fields, so no literal of it builds there.
    #[must_use]
    pub fn of(model: String, profile: u16) -> Self {
        Self {
            model,
            phases: Vec::new(),
            plan: Plan::Single,
            profile,
            decode_profile: None,
            shadow: None,
            decode_shadow: None,
            sessions: None,
        }
    }

    fn single_shadow(&self) -> Option<u16> {
        (self.plan == Plan::Single).then_some(self.shadow).flatten()
    }

    /// Whether a request of `model` in `phase` resolves to the binding,
    /// where no binding in front of it matches. `glob` is what the
    /// binding's pattern asks ([`Self::pattern`]): an engine reads it once,
    /// when it is built ([`Entry`]).
    #[inline(always)]
    fn matches_as(&self, glob: &Glob, model: &str, phase: Phase) -> bool {
        // One id is the commonest pattern: asked first, by a plain branch.
        let model_ok = if matches!(glob, Glob::Exact) {
            self.model == model
        } else {
            glob.covers(&self.model, model)
        };
        model_ok && self.covers(phase)
    }

    /// [`Self::matches_as`], of a binding no engine holds.
    #[cfg(test)]
    fn matches(&self, model: &str, phase: Phase) -> bool {
        self.matches_as(&self.pattern().glob(), model, phase)
    }

    /// Whether the binding's phase list covers `phase`: an empty list
    /// covers each.
    fn covers(&self, phase: Phase) -> bool {
        self.phases.is_empty() || self.phases.contains(&phase)
    }

    /// The model pattern, as its stars read.
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

impl Pattern<'_> {
    fn glob(&self) -> Glob {
        match *self {
            Self::Any => Glob::Any,
            Self::Suffix(suffix) => Glob::Suffix(suffix.into()),
            Self::Prefix(prefix) => Glob::Prefix(prefix.into()),
            Self::Exact(_) => Glob::Exact,
        }
    }
}

/// What a binding's model pattern asks of a model id ([`Pattern`]), with
/// the text a suffix or a prefix compares. The engine reads it once when
/// it is built ([`Entry`]), so a walk of the bindings reads no pattern's
/// stars.
#[derive(Debug)]
enum Glob {
    Any,
    Suffix(Box<str>),
    Prefix(Box<str>),
    /// One id: the pattern's own text, which the binding holds.
    Exact,
}

impl Glob {
    /// Whether `pattern`, which asks this, covers `model`.
    #[inline(always)]
    fn covers(&self, pattern: &str, model: &str) -> bool {
        match self {
            Self::Any => true,
            Self::Suffix(suffix) => model.ends_with(&**suffix),
            Self::Prefix(prefix) => model.starts_with(&**prefix),
            Self::Exact => pattern == model,
        }
    }
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
/// [`Self::names_as`] says so. `==` asks another thing, whether two
/// builds made one value: what a keep asks (`Engine::read_as`).
#[derive(Clone, Copy, Debug, PartialEq)]
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
/// is made. `==` asks whether two builds made one value, what a keep
/// asks (`Engine::read_as`): whether two namings are held to one test is
/// [`Self::same_test`].
#[derive(Clone, Copy, Debug, PartialEq)]
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
/// where that binding's profile reads none. A side reads the record its
/// pool keeps by its own profile's word of it
/// ([`Engine::side_reads_sent_alone`]). A shadow profile of another
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
    /// Each profile behind a counted reference: the engine of a later
    /// spec keeps a profile's object where nothing it was built from
    /// changed ([`Engine::keep_from`]), and the two engines then run one
    /// set of stages.
    profiles: Vec<Arc<Profile>>,
    /// Beside each profile, the generation whose build made its object:
    /// this engine's own, or an earlier one's for a profile it kept.
    #[cfg(feature = "in-process")]
    born: Vec<u64>,
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
    /// Per profile, whether its picker is the crate's own rotation
    /// ([`TURNS_ITS_SET`]), by the word of the builder that made the
    /// pickers ([`Self::made_by_a_catalog`]): a whole pass that kept every
    /// row of a list turns the set by the key the list made once
    /// ([`turn_whole`]). No profile's is in an engine built in code.
    set_turns: Box<[bool]>,
    /// Whether any profile's is ([`Self::turns_a_set`]).
    #[cfg(feature = "in-process")]
    turns_a_set: bool,
    /// Per profile, which of its leading stages are ranks ([`Order`]). A
    /// decision reads it where the profile's mark says so, and no other
    /// decision does.
    orders: Vec<Order>,
    /// Per profile, the smallest pool over which its draw may be asked
    /// before a row's record is written ([`Engine::drawn_for`]), and
    /// `usize::MAX` for a profile whose draw is never asked so. A profile
    /// is asked where it draws first and its picker is one of the crate's
    /// own, whose draw reads a row through the test it is handed and in no
    /// other way ([`DRAWS_BY_ITS_TEST`]): from [`WRITTEN_PAIR_FROM`] rows
    /// under a picker that draws two rows ([`DRAWS_TWO`]) where every
    /// profile of the document draws first under such a picker and keeps a
    /// row by the one folded test, and from [`DRAW_FROM`] in every other
    /// case. So the profiles of one document that are asked hold one
    /// floor, [`Self::written_least`]: it stands per profile for the one
    /// compare that tells [`Engine::drawn_for`] both whether a profile is
    /// asked and from what size.
    #[cfg(feature = "in-process")]
    written_from: Box<[usize]>,
    /// The least of [`Self::written_from`]: a smaller pool is no draw's
    /// under any profile, and is told so at one compare, before its
    /// request's binding is resolved.
    #[cfg(feature = "in-process")]
    written_least: usize,
    /// Whether some binding's pair may be asked by its draw
    /// ([`Entry::pair_drawn`]): a document with no such binding is told so
    /// at one load, before a pair's decision asks anything of its request.
    #[cfg(feature = "in-process")]
    pairs_drawn: bool,
    /// Whether no stage of any profile and no sessions rule's assignment
    /// reads a count a worker object alone holds ([`Self::rows_stand`]).
    #[cfg(feature = "in-process")]
    rows_stand: bool,
    /// What the door needs of a seat, zero on the plain path.
    seat_flags: u8,
    explain_sample: u32,
    /// The age limit of load reports a decision by rank checks a report
    /// by: the limit its document's build started under
    /// ([`Engine::built`]), and none for an engine made in code.
    max_report_age_ms: Option<u64>,
    /// The stages that keep a number of the reports
    /// ([`Lifecycle::keeps_loads`]), each by its place: listed once at the
    /// build, so a publication asks no stage whether it keeps them.
    keepers: Vec<StagePlace>,
    /// The stages that take each word of a lease, each by its place.
    takers: Takers,
    /// One tally for each two send settings a profile's stages read
    /// under: a later engine's tally under the same two shares its
    /// entries, and one under two that no tally here has takes a copy of
    /// the sends that stand ([`Engine::succeed`]).
    pub(crate) sent_credits: Vec<Arc<DispatchCredit>>,
    /// Beside each profile, the place in `sent_credits` of the tally its
    /// stages read, where one asked for it at the build.
    #[cfg(feature = "in-process")]
    sent_of: Vec<Option<usize>>,
    /// Beside each tally, the generation whose build made its object.
    /// Empty until the engine keeps one ([`Engine::keep_from`]): every
    /// tally is its own.
    #[cfg(feature = "in-process")]
    sent_born: Vec<u64>,
    /// The number of the stage table a spec's build made the stages over
    /// (`StageRegistry::own`): none for an engine made in code.
    #[cfg(feature = "in-process")]
    stage_table: Option<u64>,
    /// The kinds of event the document takes, but for its counts of sends
    /// ([`Lifecycle::hears`] adds those): each stage's answer, asked once at
    /// the build, with the engine's own. A request's word may go to a
    /// stage of any profile that takes it ([`Takers`]), so the answer is
    /// the document's.
    hears: EventSet,
    /// The stages' answers alone, as [`Self::hears`] folds them, without
    /// the engine's own ([`Self::hears_beside_table`]).
    stages_hear: EventSet,
}

/// A stage by its place in the engine. The engine lists by it the stages
/// that keep a number of the reports ([`Lifecycle::keeps_loads`]) and the
/// stages that take each word of a lease ([`Takers`]). A place names any
/// scorer a profile holds, and its picker: a list leaves out none of them
/// for where it stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StagePlace {
    /// The scorer at `stage` of the profile at `profile`.
    Scorer { profile: usize, stage: usize },
    /// The picker of the profile at `profile`.
    Picker { profile: usize },
}

impl StagePlace {
    /// The stages of `profiles` that say they keep the reports, by their
    /// places: in the profiles' order, a profile's scorers ahead of its
    /// picker.
    fn keepers(profiles: &[Profile]) -> Vec<Self> {
        let mut keepers = Vec::new();
        for (at, profile) in profiles.iter().enumerate() {
            for (stage, weighted) in profile.scorers.iter().enumerate() {
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

    /// The stage at this place of `profiles`, the list the place was read
    /// of.
    #[inline(always)]
    fn stage(self, profiles: &[Arc<Profile>]) -> &dyn Lifecycle {
        match self {
            Self::Scorer { profile, stage } => &*profiles[profile].scorers[stage].scorer,
            Self::Picker { profile } => &*profiles[profile].picker,
        }
    }

    /// The place of the profile the stage is of.
    #[cfg(feature = "in-process")]
    #[inline(always)]
    fn profile(self) -> usize {
        let (Self::Scorer { profile, .. } | Self::Picker { profile }) = self;
        profile
    }

    /// The stage's place in its profile: a scorer's among the scorers,
    /// and none for the picker.
    #[cfg(feature = "in-process")]
    fn within(self) -> Option<usize> {
        match self {
            Self::Scorer { stage, .. } => Some(stage),
            Self::Picker { .. } => None,
        }
    }
}

/// The stages that take each word of a lease ([`Lifecycle::hears`]), by
/// their places and in the engine's order. They are listed once, at the
/// build. A word told to the stages walks its list and no other stage: a
/// document of many profiles pays such a word by the stages that take it,
/// and not by the profiles it holds.
#[derive(Debug, Default)]
struct Takers {
    issued: Vec<StagePlace>,
    dispatched: Vec<StagePlace>,
    completed: Vec<StagePlace>,
    released: Vec<StagePlace>,
}

impl Takers {
    /// Lists the stage at `place` under each word of a lease in `hears`,
    /// the stage's own answer.
    fn list(&mut self, place: StagePlace, hears: EventSet) {
        let lists = [
            (EventKind::Issued, &mut self.issued),
            (EventKind::Dispatched, &mut self.dispatched),
            (EventKind::Completed, &mut self.completed),
            (EventKind::Released, &mut self.released),
        ];
        for (kind, list) in lists {
            if hears.contains(kind) {
                list.push(place);
            }
        }
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
    /// A scorer weighs load against affinity by a gate of its own
    /// ([`Scorer::balances_load_against_affinity`]). A pair's decision
    /// reads it for its prefill profile ([`Engine::weighs`]). No single
    /// decision does: each of its masked reads names other marks.
    const WEIGHS: u16 = 1 << 13;
    /// A filter declares the ring need ([`Filter::needs`]). The run of a
    /// pair side's shadow that is lent a ring reads it: it asks its
    /// lender ahead of such a profile's filter half ([`run_side_lent`]).
    /// No other decision reads it.
    const RING_FILTER: u16 = 1 << 14;

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
        let weighing = |stage: &WeightedScorer| stage.scorer.balances_load_against_affinity();
        let weighs = if profile.scorers.iter().any(weighing) {
            Self::WEIGHS
        } else {
            0
        };
        let ring_filter = if profile.filters.iter().any(|filter| filter.needs().ring) {
            Self::RING_FILTER
        } else {
            0
        };
        // The three marks no run of the engine reads.
        let for_hosts = by_reports | key_place | other_place;
        // What says when the profile draws, and how.
        let draws = draws_first | joined | turns;
        // The two marks a pair's decision alone reads.
        let for_pairs = weighs | ring_filter;
        Self(scoped | by_assignment | draws | ranked | engines | owns | for_hosts | for_pairs)
    }

    /// A profile whose requests take their list's own turn first.
    const fn turns(self) -> bool {
        self.0 & Self::TURNS != 0
    }

    /// A scorer that weighs load against affinity by a gate of its own.
    const fn weighs(self) -> bool {
        self.0 & Self::WEIGHS != 0
    }

    /// A filter that declares the ring need.
    const fn ring_filter(self) -> bool {
        self.0 & Self::RING_FILTER != 0
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
    /// What the binding's model pattern asks ([`Binding::pattern`]), read
    /// once when the engine is built.
    glob: Glob,
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
    /// Whether each side of the binding's pair may be asked by its draw
    /// ([`Engine::pair_drawn_for`]): the plan pairs, and each side's profile
    /// is a pick of two that draws first, asks nothing ahead of its draw
    /// and keeps a row by the engine's one folded test of it. Written once
    /// when the engine is built: a pair's decision reads this one byte and
    /// then asks of its request and its two lists alone.
    #[cfg(feature = "in-process")]
    pair_drawn: bool,
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

/// What a binding's sessions rule reads for a decision of `phase` beyond
/// its profile's stages: the key under the view's headers, and the per-key
/// counts its assignment and cap compare. It is the validator's summary,
/// per binding. Nothing for the encode side of the `epd` plan: there a rule
/// places the pair's two sides and no item, which goes by its side's
/// policy. A single binding's rule places a request of every phase it
/// covers, an encode request among them.
fn session_needs(binding: &Binding, phase: Phase) -> Needs {
    let item = phase == Phase::Encode && matches!(binding.plan, Plan::Epd { .. });
    binding
        .sessions
        .as_ref()
        .filter(|_| !item)
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
#[non_exhaustive]
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
    /// Whether the pools are a part of the lists the request's caller
    /// lent. Two things make a part. A hard demand its door judged left
    /// each side the workers that meet it. Or the encode pass is laid for
    /// the request, which states an item or holds its decode to the ones
    /// that take images: the pass hands the pair the workers its
    /// encoders and its images leave. Three writers say so, each from
    /// the test it already makes: the pair's door and the encode door
    /// where the request states a demand, and the encode pass where it
    /// is laid.
    ///
    /// Nothing else makes a part. The places a host states for a side are
    /// its own word of its list. The workers a route's rule takes are the
    /// call's places for its later decisions. Under both the pools are
    /// whole.
    ///
    /// Behind a part a pair's decision does not ask its one question of a
    /// shed ([`Engine::decide_pair`]). Private to the crate: a host of its
    /// own builds the pools from their default, which says whole.
    pub(crate) part: bool,
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
        let width = self.width();
        marks.reset(width);
        // The decodes are read until every number from one to the width
        // less one has an open decode: no decode behind that one opens a
        // class, and `open` is not asked for it.
        let mut closed = width.saturating_sub(1);
        let mut any = false;
        for class in open {
            if class == 0 || class as usize >= width || marks.has(class) {
                continue;
            }
            marks.insert(class);
            any = true;
            closed -= 1;
            if closed == 0 {
                break;
            }
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
#[non_exhaustive]
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
#[non_exhaustive]
pub enum FirstPartnersError {
    #[error("the first partners' lists: {0}")]
    Lists(PairTableError),
    #[error("a first partner is no partner of its prefill class in the pair table")]
    NoPartner,
}

impl<'a> PairPools<'a> {
    /// The two pools with nothing beside them: no affinity column, no
    /// signal cells and no pair table, so classes pair by their number.
    /// A caller assigns what it holds beside its pools over it:
    /// `let mut pools = PairPools::of(prefills, decodes); pools.table = table;`.
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

    /// One side of the pair as its stages read it: the side's pool, and the
    /// request's facts with the side's phase, its draw sequence and its own
    /// pool's affinity column and signal cells, and with no rows named:
    /// those index neither pool. `None` when that column was not filled
    /// for that pool ([`column_fits`]). The one place a pair's column and
    /// cells meet their pool, so no call site hands one side's with the
    /// other side's candidates.
    ///
    /// A side reads its own places on a lent ring in the one list
    /// ([`RequestFacts::ring_positions`]): the decode side's own where the
    /// request states them apart
    /// ([`RequestFacts::decode_ring_positions`]), which neither side's
    /// facts carry on. So a stage is never shown the other side's list.
    fn side(
        self,
        side: PairSide,
        facts: &RequestFacts<'a>,
    ) -> Option<(&'a [Candidate], RequestFacts<'a>)> {
        let (pool, affinity, signal_cells, phase, seed, ring_positions) = match side {
            PairSide::Prefill => (
                self.prefills,
                self.prefill_affinity,
                self.prefill_signal_cells,
                Phase::Prefill,
                facts.seed,
                facts.ring_positions,
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
                facts.decode_ring_positions.unwrap_or(facts.ring_positions),
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
                // The side's own cells: the request's would be read by
                // another pool's positions.
                signal_cells,
                // The side's own places, in the one list a stage reads.
                ring_positions,
                decode_ring_positions: None,
                ..*facts
            };
            (pool, facts)
        })
    }
}

/// Whether the request carries a column of its own or names rows. A pair
/// reads neither: its columns ride its two pools ([`PairPools`]), and
/// either on the request names neither pool.
fn carries_a_column(facts: &RequestFacts<'_>) -> bool {
    !facts.affinity.is_empty() || facts.named.is_some()
}

/// The one row `number` reaches among `kept`, left as the set's one row:
/// `false`, and no row, where it reaches none. Cold and out of line: a
/// side that states no number runs one test of the word.
#[cold]
#[inline(never)]
fn named_row(kept: &mut Vec<u32>, number: u32) -> bool {
    let named = kept.get(number as usize).copied();
    kept.clear();
    kept.extend(named);
    named.is_some()
}

/// Whether a pair's decode side states a number beside no places
/// ([`Constraints::target_worker`]): the number counts the places a host
/// states, so such a decision is declined under `target-worker`.
const fn counts_no_place(decode_cx: &Constraints<'_>) -> bool {
    decode_cx.target_worker.is_some() && decode_cx.places.is_none()
}

/// The decline of a decision whose decode side states a number beside no
/// places ([`counts_no_place`]), under `profile`. The decode's entry
/// names the profile it resolved. The pair's entry declines ahead of its
/// binding's lookup and names none: it hands on the profile of its bare
/// trace ([`Trace::profile`]). Cold and out of line, as
/// [`rows_not_written`] is: an entry's path past the refusal keeps its
/// weight.
#[cold]
#[inline(never)]
fn number_not_counted(bare: Trace, profile: u16) -> Decision {
    let trace = Trace { profile, ..bare };
    shed(ShedReason::PolicyDeclined, TARGET_WORKER, trace)
}

/// What a pair answers where its request names its decode by a number
/// ([`Constraints::target_worker`]) and the decode side's run named no
/// worker. Its shed for want of that worker, with no endpoint or for a
/// worker over its thresholds, takes the target's name: the request has
/// the one decode. Any other answer stands as it is. Cold and out of
/// line: a decode side that picks runs none of it.
#[cold]
#[inline(never)]
fn target_missed(decision: Decision) -> Decision {
    match decision {
        Decision::Shed(Shed {
            reason: ShedReason::NoEndpoints | ShedReason::Saturated,
            trace,
            ..
        }) => shed(ShedReason::NoEndpoints, TARGET_WORKER, trace),
        other => other,
    }
}

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

/// Whether a column its host hands stands for `pool`: empty, or one entry a
/// candidate, in the candidates' order. Asked once where a decision takes
/// its inputs and never at a read. A column of any other length was filled
/// for another pool: nothing held is a real answer of an affinity column (a
/// cold worker), and nothing stated is one of what an encoder states for a
/// demand, so reading such a column as zeros would turn a host's mistake
/// into a routing change no case sees.
///
/// An empty column is of another kind and is taken, under a profile that
/// declared the need as under one that did not: it is the host's word that
/// it holds nothing for any candidate. The engine checks the column's shape,
/// never its presence.
fn column_fits<T>(column: &[T], pool: &[Candidate]) -> bool {
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
                        .or(session_needs(binding, phase));
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
        // A pair's draw is asked under `DRAW_FROM` rows only in a document
        // whose every profile draws two: beside another profile the least
        // floor would resolve that profile's request twice over a small
        // pool, once to be told its own floor and once in the whole pass.
        // And only where every profile keeps a row by the folded test: a
        // filter the engine does not fold may refuse a row of a list that
        // stands, and behind a refused row the draw is not the pass's pick.
        #[cfg(feature = "in-process")]
        let pairs_alone = profiles.iter().all(|profile| {
            draws_first(profile)
                && DRAWS_TWO.contains(&profile.picker.name())
                && RowTests::of(&profile.filters).is_some()
        });
        #[cfg(feature = "in-process")]
        let written_from: Box<[usize]> = profiles
            .iter()
            .map(|profile| {
                let name = profile.picker.name();
                if !draws_first(profile) || !DRAWS_BY_ITS_TEST.contains(&name) {
                    usize::MAX
                } else if pairs_alone && DRAWS_TWO.contains(&name) {
                    WRITTEN_PAIR_FROM
                } else {
                    DRAW_FROM
                }
            })
            .collect();
        #[cfg(feature = "in-process")]
        let written_least = written_from.iter().copied().min().unwrap_or(usize::MAX);
        let row_tests: Vec<Option<RowTests>> = profiles
            .iter()
            .map(|profile| RowTests::of(&profile.filters))
            .collect();
        // No profile's until the builder that made the pickers says whose
        // they are ([`Self::made_by_a_catalog`]): a picker handed in here
        // may be a host's, whatever name it answers.
        let set_turns: Box<[bool]> = profiles.iter().map(|_| false).collect();
        // A sessions rule that places a new key by a count scores the rows
        // itself, beside every profile ([`Self::place`]).
        #[cfg(feature = "in-process")]
        let rows_stand = profiles.iter().zip(&row_tests).all(|(profile, tests)| {
            tests.is_some()
                && profile.scorers.is_empty()
                && PICKS_BY_NO_COUNT.contains(&profile.picker.name())
        }) && bindings.iter().all(|binding| {
            binding.sessions.as_ref().is_none_or(|rule| {
                matches!(rule.assignment, Assignment::Delegate | Assignment::Random)
            })
        });
        let named_by: Vec<NamedBy> = profiles
            .iter()
            .zip(&row_tests)
            .map(|(profile, &folded)| profile.named_by(folded))
            .collect();
        let keepers = StagePlace::keepers(&profiles);
        // The engine's own: the session table follows a dispatch, and at a
        // release takes back the claim of a decision its commit refused,
        // where a binding states a sessions rule. At an issue the engine
        // does nothing but tell its stages: the record of dispatched
        // prefixes is its host's to write, where the host decides.
        let own = if bindings.iter().any(|binding| binding.sessions.is_some()) {
            EventSet::NONE
                .with(EventKind::Dispatched)
                .with(EventKind::Released)
        } else {
            EventSet::NONE
        };
        // Each scorer and picker is asked once what it takes, a profile's
        // picker ahead of its scorers. Its answer is in the document's,
        // and lists it under each word of a lease it names: a profile's
        // scorers ahead of its picker, the order the words are told in.
        let mut takers = Takers::default();
        let mut stages_hear = EventSet::NONE;
        for (at, profile) in profiles.iter().enumerate() {
            let picker = profile.picker.hears();
            for (stage, weighted) in profile.scorers.iter().enumerate() {
                let hears = weighted.scorer.hears();
                stages_hear = stages_hear.union(hears);
                takers.list(StagePlace::Scorer { profile: at, stage }, hears);
            }
            stages_hear = stages_hear.union(picker);
            takers.list(StagePlace::Picker { profile: at }, picker);
        }
        let hears = stages_hear.union(own);
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
        #[cfg(feature = "in-process")]
        let mut pairs_drawn = false;
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
                // A side its rule leaves out is its profile's alone, and a
                // picker that leaves the placement to the rule has no
                // stages to place it by. A document's check refuses that
                // binding. One built in code stops a debug build here.
                debug_assert!(
                    binding.sessions.is_none_or(|rule| {
                        let decode = binding.decode_profile.unwrap_or(binding.profile);
                        (rule.sides.covers(false) || !assigned(binding.profile))
                            && (rule.sides.covers(true) || !assigned(decode))
                    }),
                    "a picker that places by the sessions rule on a side the rule's sides leave out"
                );
                // Each side of a pair may be asked by its draw where its
                // profile is a pick of two that draws first, asks nothing
                // ahead of the draw and folds its filters into one test.
                #[cfg(feature = "in-process")]
                let pair_drawn = matches!(binding.plan, Plan::PdPair | Plan::Epd { .. })
                    && [
                        binding.profile,
                        binding.decode_profile.unwrap_or(binding.profile),
                    ]
                    .into_iter()
                    .all(|index| {
                        let at = usize::from(index);
                        profiles
                            .get(at)
                            .is_some_and(|profile| DRAWS_TWO.contains(&profile.picker.name()))
                            && row_tests.get(at).is_some_and(Option::is_some)
                            && marks.get(at).is_some_and(|marks| {
                                marks.asks_nothing_first()
                                    && marks.ahead() == Marks::DRAWS_FIRST
                                    && !marks.by_assignment()
                            })
                    });
                #[cfg(feature = "in-process")]
                {
                    pairs_drawn |= pair_drawn;
                }
                Entry {
                    glob: binding.pattern().glob(),
                    placing: placed_by(declared, by_assignment),
                    vetoed: AllVetoed::default(),
                    sides_placed,
                    #[cfg(feature = "in-process")]
                    pair_drawn,
                    binding,
                }
            })
            .collect();
        Self {
            #[cfg(feature = "in-process")]
            born: vec![generation; profiles.len()],
            profiles: profiles.into_iter().map(Arc::new).collect(),
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
            set_turns,
            #[cfg(feature = "in-process")]
            turns_a_set: false,
            orders,
            #[cfg(feature = "in-process")]
            written_from,
            #[cfg(feature = "in-process")]
            written_least,
            #[cfg(feature = "in-process")]
            pairs_drawn,
            #[cfg(feature = "in-process")]
            rows_stand,
            seat_flags: (u8::from(keeping) * KEEPS_SLOT_STATE) | (u8::from(shadows) * SHADOWS),
            explain_sample: 0,
            max_report_age_ms: None,
            keepers,
            takers,
            sent_credits: Vec::new(),
            #[cfg(feature = "in-process")]
            sent_of: Vec::new(),
            #[cfg(feature = "in-process")]
            sent_born: Vec::new(),
            #[cfg(feature = "in-process")]
            stage_table: None,
            hears,
            stages_hear,
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Takes over what the standing engine's counts of sends hold. A
    /// count takes the rows' entries of the standing count under the same
    /// two settings, whichever profiles read that one, and the two
    /// engines share those entries from here on: a decision still running
    /// under the standing engine reads the sends a later one counts. A
    /// count under two settings that no standing count has shares none: a
    /// standing count has let go what its own settings ended. It takes a
    /// copy of the sends that stand, of each rank those of the standing
    /// count that holds the most there, each at its instant and its size,
    /// and its own reads let go what its own settings end. The engine
    /// records a send in every count and a count lets go a prefix by
    /// time, so the fullest holds what the others hold, but for a send
    /// that an earlier copy missed. A count starts empty where no count
    /// stands.
    ///
    /// Three limits stand for a count that takes a copy. A send every
    /// standing count has let go is gone for it too, though its own
    /// settings would count it still. It shares no entry with the
    /// standing engine: a decision still running there reads none of the
    /// sends the later engine counts in it. And a send told to the
    /// standing engine behind the copy's read of its rank is in no copy.
    /// A thread that read the standing engine ahead of the swap may tell
    /// it one: neither
    /// [`MemoryStore::apply`](crate::store::MemoryStore::apply) nor the
    /// apply of an engine in its host's process holds a lock that a word
    /// of a lease takes. The standing counts' rows of a rank are read one
    /// after another, so a send told between two of those reads is in the
    /// copy only where the longest read holds it. A count lacks a send
    /// its copy missed for that send's life: where that count holds the
    /// most at a later call, the copy made there lacks the send too,
    /// though another standing count holds it.
    ///
    /// [`MemoryStore::apply`](crate::store::MemoryStore::apply) has it
    /// called under its workers lock and its load lock, where the
    /// generation is put in place of the standing one and before the
    /// store shows it: a copy takes as long as the sends it copies, and a
    /// taker of either lock waits for it. A host with a store of its own
    /// calls it where it swaps engines, before this engine takes a
    /// decision or a dispatch: what this engine's counts held until then
    /// is let go. A rank a standing count first hears of behind this call
    /// is entered in the standing count alone. A copy reads every
    /// standing count's row of a rank and takes the longest read. It
    /// holds one standing row's locks at a time, for the read of its
    /// sends: a send to that row and a release from it wait meanwhile.
    pub fn succeed(&self, standing: &Engine) {
        let prior = &standing.sent_credits;
        for credit in &self.sent_credits {
            let under = |count: &&Arc<DispatchCredit>| count.settings == credit.settings;
            match prior.iter().find(under) {
                // A count this engine kept is the standing one itself.
                Some(stood) if Arc::ptr_eq(credit, stood) => {}
                Some(stood) => credit.succeed(stood),
                None => credit.copy_fullest(prior),
            }
        }
    }

    /// Keeps the standing engine's profile objects where nothing they were
    /// built from changed: a profile of the same name for which `same`
    /// holds, asked by that name, and whose stages say of themselves what
    /// the standing ones say. Such a profile's stages keep what they
    /// hold, and a decision under either engine runs the one set of them.
    /// What a stage says of itself is its name, in the stages' order, its
    /// needs, the words it takes, whether it keeps load reports and
    /// whether it needs the lifecycle's calls, and what the build read of
    /// the profile ([`Self::read_as`]). The spec's validation and this
    /// engine's tables were folded from the stages built for it: a
    /// profile whose factory built it a stage that says otherwise, under
    /// the same name and text, is built new.
    /// A count of sends stands where a profile that reads it is kept: this
    /// engine goes on with the standing count, and the count its build
    /// made follows that one ([`DispatchCredit::follow`]), so a reader
    /// built new and a reader kept read one set of entries and a rank
    /// first heard behind this call stands once. A count none of whose
    /// readers is kept is this engine's own, and takes over what the
    /// standing counts hold ([`Self::succeed`]). A profile kept keeps the
    /// generation its object was made under ([`Engine::stood_by`]).
    ///
    /// Nothing is kept across two stage tables that may build another
    /// stage under one name: the stock table's number, or one table of a
    /// host's own that took no stage between the two builds, is the same
    /// under both. Its caller calls it before this engine takes a decision
    /// or a word: what a replaced object held until then is let go.
    #[cfg(feature = "in-process")]
    pub(crate) fn keep_from(&mut self, standing: &Engine, same: impl Fn(&str) -> bool) {
        if self.stage_table.is_none() || self.stage_table != standing.stage_table {
            return;
        }
        // What a stage says of itself to a validation and to a making: its
        // name, its needs, the words it takes, and whether it keeps load
        // reports and needs the lifecycle's calls.
        let said = |of: &Profile| {
            let filters = of.filters.iter();
            let filters = filters.map(|f| (f.name(), f.needs(), EventSet::NONE, [false; 2]));
            let scorers = of.scorers.iter().map(|w| &w.scorer).map(|s| {
                let calls = [s.keeps_loads(), s.needs_lifecycle()];
                (s.name(), s.needs(), s.hears(), calls)
            });
            let picker = &of.picker;
            let calls = [picker.keeps_loads(), picker.needs_lifecycle()];
            let picker = (picker.name(), picker.needs(), picker.hears(), calls);
            filters.chain(scorers).chain([picker]).collect::<Vec<_>>()
        };
        // The standing profiles' places by their names, read once: each
        // profile finds its twin at one lookup. The walk of the counts
        // below reads the profiles for each count's readers.
        let mut places = HashMap::with_capacity(standing.profiles.len());
        for (at, stood) in standing.profiles.iter().enumerate() {
            places.entry(stood.name.as_str()).or_insert(at);
        }
        // Beside each profile, the place of the standing one it keeps.
        let twin = |(at, built): (usize, &Arc<Profile>)| {
            let stood = *places.get(built.name.as_str())?;
            let like = same(&built.name)
                && said(built) == said(&standing.profiles[stood])
                && self.read_as(at, standing, stood);
            like.then_some(stood)
        };
        let profiles = self.profiles.iter().enumerate();
        let kept: Vec<Option<usize>> = profiles.map(twin).collect();
        // A count's readers, by their places. Where one of them is kept,
        // the engine goes on with the standing count that reader holds, and
        // the count the build made follows it: a reader built new reads
        // and writes the standing count's entries through its own.
        debug_assert_eq!(self.sent_of.len(), self.profiles.len());
        let read = |of: &Self, at: usize| of.sent_of.get(at).copied().flatten();
        self.sent_born = vec![self.generation; self.sent_credits.len()];
        for count in 0..self.sent_credits.len() {
            let mut readers = (0..self.profiles.len()).filter(|&at| read(self, at) == Some(count));
            let stood = readers.find_map(|at| read(standing, kept[at]?));
            let Some(stood) = stood else { continue };
            // One pair of settings, one standing count: the other readers
            // kept hold the same one.
            let same = |twin| read(standing, twin) == Some(stood);
            debug_assert!(readers.all(|at| kept[at].is_none_or(same)));
            let leader = &standing.sent_credits[stood];
            self.sent_credits[count].follow(leader);
            let born = standing.sent_born.get(stood).copied();
            self.sent_born[count] = born.unwrap_or(standing.generation);
            self.sent_credits[count] = Arc::clone(leader);
        }
        for (at, stood) in kept.into_iter().enumerate() {
            let Some(stood) = stood else { continue };
            self.profiles[at] = Arc::clone(&standing.profiles[stood]);
            self.born[at] = standing.born[stood];
        }
    }

    /// Whether the build read of the stages of the profile at `at` what
    /// `standing` holds of its profile at `stood`: each table an engine
    /// folds from a profile's stages at its making and reads in their
    /// place, and whether a stage asked its build for the profile's count
    /// of sends. The lists of the stages that keep reports and that take
    /// each word are such tables ([`Self::listed`]): a stage kept is told
    /// as the build that made it listed it, and not as its twin of the
    /// later build answered. Two tables are not compared. The profile's
    /// needs are its stages' own, which its caller compares stage by
    /// stage, less a sample the marks rule. Whether its draw may be asked
    /// first is read of the marks and the picker's name. A table added to
    /// the making is added here.
    #[cfg(feature = "in-process")]
    fn read_as(&self, at: usize, standing: &Self, stood: usize) -> bool {
        let asked = |of: &Self, at: usize| of.sent_of.get(at).is_some_and(Option::is_some);
        // Asked last: the two engines' lists are searched only where the
        // profile's other tables agree.
        let listed = || {
            let mut lists = self.listed(at).into_iter().zip(standing.listed(stood));
            lists.all(|(here, there)| here.eq(there))
        };
        self.marks[at] == standing.marks[stood]
            && self.orders[at] == standing.orders[stood]
            && self.named_by[at] == standing.named_by[stood]
            && self.row_tests[at] == standing.row_tests[stood]
            && self.head_places[at] == standing.head_places[stood]
            && asked(self, at) == asked(standing, stood)
            && listed()
    }

    /// What the build listed of the profile at `at`: of the list of the
    /// stages that keep reports and of each word's list, the profile's
    /// stages by their places in it. Each list is in the engine's order,
    /// so a profile's stages stand together in it.
    #[cfg(feature = "in-process")]
    fn listed(&self, at: usize) -> [impl Iterator<Item = Option<usize>> + '_; 5] {
        let Takers {
            issued,
            dispatched,
            completed,
            released,
        } = &self.takers;
        [&self.keepers, issued, dispatched, completed, released].map(|list| {
            let from = list.partition_point(|place| place.profile() < at);
            let upto = list.partition_point(|place| place.profile() <= at);
            list[from..upto].iter().map(|place| place.within())
        })
    }

    /// An engine of no binding that holds this engine's profiles and
    /// counts of sends whose objects were made under a generation `made`
    /// passes, over the same session table. It decides nothing: a word
    /// that is not for every stage of this engine is told to it. The
    /// workers that stand at an apply are told as joined to the stages
    /// built new. An older request's words take no engine of this kind:
    /// they are told by the generation of their lease ([`Self::stood_by`]).
    ///
    /// It is made with no profile and then handed four things of this
    /// engine: the profile objects, the generations they were made
    /// under, the counts of sends and the session table. Every other
    /// table of it is an empty engine's and says nothing of these
    /// profiles: a word that reads one is not for it, as a load report's
    /// telling reads its keepers and a lease's word the stages that take
    /// it.
    #[cfg(feature = "in-process")]
    pub(crate) fn hearing(&self, made: impl Fn(u64) -> bool) -> Engine {
        let mut hearing = Engine::new(Vec::new(), Vec::new(), self.generation, Needs::default());
        let stood = (0..self.profiles.len()).filter(|&at| made(self.born[at]));
        let stood: Vec<usize> = stood.collect();
        let object = |at: &usize| Arc::clone(&self.profiles[*at]);
        hearing.profiles = stood.iter().map(object).collect();
        hearing.born = stood.iter().map(|&at| self.born[at]).collect();
        let born = |at: usize| self.sent_born.get(at).copied().unwrap_or(self.generation);
        let counts = (0..self.sent_credits.len()).filter(|&at| made(born(at)));
        let count = |at| Arc::clone(&self.sent_credits[at]);
        hearing.sent_credits = counts.map(count).collect();
        hearing.sessions = Arc::clone(&self.sessions);
        hearing
    }

    /// What this engine kept of the stages that stood by `generation`, an
    /// earlier one than its own: its session table, and each profile whose
    /// object was made under that generation or an earlier one. A word of
    /// a lease issued under that generation is told to it: a stage built
    /// since never heard the lease's issue, and hears no word of it.
    ///
    /// It is written for an earlier generation alone. Asked the engine's
    /// own generation or one above it, it answers every stage the engine
    /// kept and no stage its own build made.
    #[cfg(feature = "in-process")]
    #[inline]
    pub(crate) fn stood_by(&self, generation: u64) -> Stood<'_> {
        Stood {
            engine: self,
            below: Some(generation.saturating_add(1).min(self.generation)),
        }
    }

    /// This engine whole, as a hearer of a lease's words that is named as
    /// [`Self::stood_by`] names one: its session table and every profile.
    /// A host that tells leases of any generation off a file of its own
    /// names the hearers of each in the one type.
    #[cfg(feature = "controller")]
    #[inline]
    pub(crate) fn stands(&self) -> Stood<'_> {
        Stood {
            engine: self,
            below: None,
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

    /// Whether the rows of a decision under this document may be written
    /// with no read of a worker object where every worker of its lists is
    /// available: no stage of any profile, live or shadow, and no sessions
    /// rule reads a row's requests in flight or a count of its sessions.
    /// Each profile holds the engine's own filters alone, no scorer, and a
    /// picker that reads no count ([`PICKS_BY_NO_COUNT`]); a plugin's stage
    /// and a host's own are taken as reading one. No sessions rule places
    /// a new key on the worker with the fewest requests or the fewest
    /// keys, and none caps a key's requests. The engine's own rule for an
    /// emptied pool reads the count too, and never over rows none of which
    /// carries a veto.
    #[cfg(feature = "in-process")]
    pub(crate) fn rows_stand(&self) -> bool {
        self.rows_stand && !self.needs.session_in_flight && !self.needs.active_sessions
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
    /// beside them ([`Constraints::stalled`]). So does [`Self::decide_pair`]
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
                        .or(session_needs(&entry.binding, phase)),
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
    /// its decider's, and a stage may hear a word it takes of a lease
    /// whichever profile decided it. Under such a generation a host that
    /// seats its callers by thread decides each request at its model's
    /// decider, so that state has one writer a model and hears a model's
    /// leases whole. It is one read: under a generation with none a host
    /// looks nothing up.
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

    /// The words the document takes of a pick its session table has no
    /// word for ([`SessionVerdict::None`]: [`Self::dispatched_table`]
    /// passes it by): a stage's, and a count of sends at a dispatch.
    #[cfg_attr(
        not(feature = "in-process"),
        expect(
            dead_code,
            reason = "the one reader is the engine over worker lists, built under `in-process`"
        )
    )]
    pub(crate) fn hears_beside_table(&self) -> EventSet {
        self.with_sends(self.stages_hear)
    }

    /// `hears` with the dispatch a count of sends takes.
    fn with_sends(&self, hears: EventSet) -> EventSet {
        if self.sent_credits.is_empty() {
            hears
        } else {
            hears.with(EventKind::Dispatched)
        }
    }

    pub(crate) fn set_explain_sample(&mut self, sample: u32) {
        self.explain_sample = sample;
        self.seat_flags =
            (self.seat_flags & !SAMPLES_DECISIONS) | (u8::from(sample != 0) * SAMPLES_DECISIONS);
    }

    /// The document's `explain.sample`, zero where it states none: under
    /// zero no decision is kept to be read later. A host that answers a
    /// kept decision reads here whether the document keeps any.
    pub fn explain_sample(&self) -> u32 {
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
    /// either reads a ring by another text: the answer is one place for
    /// a pair's two sides, and a side of another ring form would take
    /// that place for its own. A pair whose sides read the ring by two
    /// words is answered `false`, and what makes its request states each
    /// side its own places, by that side's profile alone
    /// ([`RequestFacts::decode_ring_positions`]). `false` where no
    /// binding matches. Only a
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

    /// The shadow profile of the single binding at `at` that is stated a
    /// place of its own on a lent ring, for the request of `facts` over
    /// `candidates`: one that reads a ring where no stage of the profile
    /// that decides the request does, whatever its kind. That request
    /// states no place when it is made, so the shadow's is asked inside
    /// the shadow's own run
    /// ([`RequestMaker::shadow_ring`](crate::engine::decision_engine::RequestMaker::shadow_ring)),
    /// and the shadow walks the lent ring by the place it would be
    /// stated live. `None` under every other binding: beside a profile
    /// that reads the ring, a shadow reads that profile's place
    /// ([`Self::shadow`]). And `None` where [`Self::shadow`] sheds ahead
    /// of its profile, over no row or an affinity column that does not
    /// fit them: nothing is asked for a place no row takes.
    #[cfg(feature = "in-process")]
    pub(crate) fn shadow_ring_profile(
        &self,
        at: u32,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
    ) -> Option<u16> {
        let binding = self.binding_at(at)?;
        let shadow = binding.single_shadow()?;
        let reads = |profile: u16| Some(self.profile_needs.get(usize::from(profile))?.ring);
        if !reads(shadow)? {
            return None;
        }
        let live = profile_index(binding, facts.phase)?;
        let placed = !candidates.is_empty() && column_fits(facts.affinity, candidates);
        (placed && !reads(live)?).then_some(shadow)
    }

    /// The shadow profile of `side` of the pair binding at `at` that is
    /// stated a place of its own on a lent ring: one that reads a ring
    /// where neither live side of the pair does. Such a pair's request
    /// states no place when it is made, so the side's shadow is asked for
    /// its own inside its run, as a single binding's is
    /// ([`Self::shadow_ring_profile`]). A side's run asks behind its
    /// filter half where no filter of its profile declares the ring need
    /// ([`run_side_lent`]): a shadow that says none ahead of its profile
    /// ([`Self::shadow_side`]) and, there, one whose set the filters
    /// emptied are asked nothing. `None` under every other binding: beside
    /// a live side that reads the ring, a side's shadow reads the place
    /// the two live sides state, or none ([`Self::unlent`]). Each side's
    /// shadow is asked by its own word: two that cut a request by two
    /// rules are each stated the places of their own heads.
    fn side_ring_profile(&self, at: u32, side: PairSide) -> Option<u16> {
        let sides = self.shadow_sides.get(at as usize)?;
        let shadow = sides[side.index()]?;
        let lives = live_sides(&self.bindings.get(at as usize)?.binding);
        let reads = |profile: u16| Some(self.profile_needs.get(usize::from(profile))?.ring);
        let alone = reads(shadow)? && !reads(lives[0])? && !reads(lives[1])?;
        alone.then_some(shadow)
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

    /// Which sides of `model`'s pair a target source of its binding holds
    /// to the worker a request names: the prefill's, then the decode's. A
    /// pair's binding names one list of sources for its two sides, and the
    /// source is the side's whose profile holds a scorer that places by
    /// the routing value's own text on a ring
    /// ([`Scorer::places_by_key_text`]): the hash ring keyed by no prefix.
    /// Neither side where no such profile stands beside the source, and
    /// where the model's prefill resolves to no pair's binding. One lookup
    /// of the binding ([`Self::ring_profiles`]), asked by a pair's call
    /// under a document that names a target source.
    #[cfg(feature = "in-process")]
    pub(crate) fn names_worker_on(&self, model: &str) -> [bool; 2] {
        let (prefill, decode) = self.ring_profiles(model, Phase::Prefill);
        let plain = |at: Option<usize>| {
            let marks = at.and_then(|at| self.marks.get(at));
            marks.is_some_and(|marks| marks.key_place())
        };
        // A binding that pairs names a decode profile: no other has sides.
        [decode.is_some() && plain(prefill), plain(decode)]
    }

    /// Whether the binding of `model`'s pair steers a pool the overload
    /// veto empties to its least-vetoed worker ([`AllVetoed::Steer`]), and
    /// does not shed it: `false` where the model's prefill resolves to no
    /// binding. One lookup of the binding, asked off every decision that
    /// places: by the call of a pair that names its worker where no decode
    /// can be sent to, and by the encode entry where the overload veto left
    /// no encoder open.
    pub(crate) fn steers_all_vetoed(&self, model: &str) -> bool {
        let entry = self.resolve(model, Phase::Prefill);
        entry.is_some_and(|(entry, _, _)| entry.vetoed == AllVetoed::Steer)
    }

    /// Whether the prefill profile of `model`'s pair places among the
    /// prefills over their thresholds too: its binding steers a pool the
    /// overload veto empties, and the profile weighs load against affinity
    /// by a gate of its own ([`Self::weighing_lift`]). `false` where the
    /// model's prefill resolves to no binding. One lookup of the binding,
    /// asked by the call of a pair that names its decode alone where no
    /// available prefill has a partner to count.
    #[cfg(feature = "in-process")]
    pub(crate) fn lifts_prefills(&self, model: &str) -> bool {
        let entry = self.resolve(model, Phase::Prefill);
        entry
            .is_some_and(|(entry, index, _)| entry.vetoed == AllVetoed::Steer && self.weighs(index))
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
    /// one rule, and no scorer of either reads a ring by another text. The
    /// answer is one list of places for a pair's two sides, so a side
    /// that reads no ring holds nothing back, and a pair of two depths or
    /// of two ring forms has no such list. `None` then, where each side
    /// is stated its own places
    /// ([`RequestFacts::decode_ring_positions`]), and where no binding
    /// matches. The scorers answered when the engine was built,
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
                        .or(session_needs(binding, phase));
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

    pub fn profiles(&self) -> &[Arc<Profile>] {
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
            .find(|entry| entry.binding.matches_as(&entry.glob, model, phase))
    }

    /// The position of the first matching binding in this generation.
    pub(crate) fn binding_position(&self, model: &str, phase: Phase) -> Option<u32> {
        self.bindings
            .iter()
            .position(|entry| entry.binding.matches_as(&entry.glob, model, phase))
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
    /// build kept: the in-process engine asks it as it reads a pool's
    /// table, and no stage is asked again.
    #[cfg(any(test, feature = "in-process"))]
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
    /// ahead of the index's then, and the engine over worker lists reads
    /// and writes its record for every worker of that profile's list,
    /// whatever the index holds for the worker. One prefix stage that
    /// reads cache events
    /// keeps the index's holders first, whatever another reads: under a
    /// cap it would lose them. For a single request of token ids under a
    /// model whose index holds a block they are the only holders named. A
    /// stage that reads cache events for a row's warm-up reads no holder
    /// and turns nothing here.
    /// Kept at the build, as [`Self::reads_sent_for`] is.
    #[cfg(any(test, feature = "in-process"))]
    pub(crate) fn reads_sent_alone_for(&self, model: &str, phase: Phase) -> bool {
        self.resolve(model, phase)
            .is_some_and(|(_, index, _)| self.named_by[usize::from(index)].sent_alone)
    }

    /// [`Self::reads_sent_alone_for`] of the profile that decides `side` of
    /// `model`'s pair: the side's profile of the model's pair binding
    /// ([`Self::pair_binding`]). That is the profile a request of the
    /// side's phase asked alone resolves to, unless a binding of that
    /// phase is listed ahead of the pair's. The side's column is filled by
    /// this word, whichever profile its pool keeps the record for
    /// ([`Self::pool_record_for`]): the pool's table reads it once, beside
    /// that record's word, and no decision asks it here. `false` where the
    /// model has no pair binding, and for a phase its plan has no side for.
    #[cfg(feature = "in-process")]
    pub(crate) fn side_reads_sent_alone(&self, model: &str, side: Phase) -> bool {
        self.pair_binding(model)
            .and_then(|binding| profile_index(binding, side))
            .and_then(|index| self.named_by.get(usize::from(index)))
            .is_some_and(|by| by.sent_alone)
    }

    /// The record of sent prefixes a host keeps for a pool, for the
    /// decisions of `phase` for `model`: `None` where it keeps none, and
    /// whether the reader of a single decision over the pool reads sent
    /// prefixes alone. The record's holders are named ahead of the
    /// index's then, and the record speaks to that decision for every
    /// worker of the pool. It
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
    /// side runs under the pair's binding ([`Self::pair_binding`]),
    /// whatever binding a decode request asked alone resolves to. Where
    /// that request's profile reads no record, the pool keeps the one
    /// the pair's decode profile reads: that side is a live reader of
    /// the model's record. So a decode pool keeps a record, in this
    /// order, for that request's profile where it reads one, for the
    /// pair's decode profile, and for the shadow as above.
    ///
    /// The word is the word of the single decision's reader over the
    /// pool, and never a pair's side's: a side reads the record by its
    /// own profile's word ([`Self::side_reads_sent_alone`]). So for the
    /// pair's decode profile the pool keeps the record and the word is
    /// `false`: a decode request asked alone is decided there by a
    /// profile that reads no record, and no single reader reads alone.
    /// That decision keeps the index's holders first, and reads and
    /// writes the record as a profile that reads cache events does.
    #[cfg(any(test, feature = "in-process"))]
    pub(crate) fn pool_record_for(&self, model: &str, phase: Phase) -> Option<bool> {
        if self.reads_sent_for(model, phase) {
            return Some(self.reads_sent_alone_for(model, phase));
        }
        // A pair's decode side reads the decode pool, whichever binding
        // the phase resolves to. Its word is its own: the single decision
        // over the pool is no reader of the record.
        let side = (phase == Phase::Decode)
            .then(|| self.pair_binding(model))
            .flatten()
            .and_then(|binding| profile_index(binding, Phase::Decode))
            .and_then(|index| self.named_by.get(usize::from(index)));
        if side.is_some_and(|by| by.sent) {
            return Some(false);
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
    /// carry a mask beside them, under the stage name `written-rows`
    /// ([`Self::reads_for`]).
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

    /// Whether a draw over a pool of `rows` rows is asked only where its
    /// host expects every row kept (a list that stands, under a request
    /// that names no row refused): a pool under the floor of every
    /// profile's draw
    /// ([`DRAW_FROM`]) and at or over the least a draw is asked from
    /// before the records are written ([`WRITTEN_PAIR_FROM`], where every
    /// profile of the document draws two rows; no size in any other
    /// document). The host of the records asks its list first and
    /// [`Self::drawn_for`] behind it: two compares say so, and no binding
    /// is resolved for a list that does not stand.
    #[cfg(feature = "in-process")]
    #[inline]
    pub(crate) fn asks_standing(&self, rows: usize) -> bool {
        (self.written_least..DRAW_FROM).contains(&rows)
    }

    /// The draw a plain single decision of `facts` is, over a pool of
    /// `rows` rows whose host writes a row's record when the decision asks
    /// about the row ([`Self::decide_drawn`]). `None` where
    /// [`Self::decide`] would not decide these facts by its draw at once,
    /// or the profile's draw may not be asked before the records
    /// ([`DRAWS_BY_ITS_TEST`]): that host writes every row and asks
    /// [`Self::decide`]. One draw is asked where [`Self::decide`] reads
    /// the pool whole: a picker that draws two rows ([`DRAWS_TWO`]) draws
    /// over a pool of [`WRITTEN_PAIR_FROM`] rows or more in a document
    /// whose every profile draws so, by the rule its whole pass places the
    /// request by.
    ///
    /// The conditions are the ones [`Self::decide`] comes to its draw
    /// under, in its order: the binding the request resolves to; no column
    /// of affinity facts, which is one entry a written row; no scope to
    /// open, no assignment that places and no key that lifts a veto; no
    /// sessions rule over a keyed request, but one that leaves a key with
    /// no pin to a profile that takes turns and whose filters are one test
    /// of a row: that draw claims ([`Drawn::claims`]), and its host asks
    /// [`Self::decide_claimed`]; no encode item; no list of the
    /// pool's engines for a profile that decides over them; a pool of
    /// [`DRAW_FROM`] rows or more, or of [`WRITTEN_PAIR_FROM`] under a
    /// picker that draws two rows beside no other kind of profile; and a
    /// profile that draws for every
    /// request, or for this one, which names rows. A smaller pool is
    /// written whole, also where [`Self::decide`] then draws over it for
    /// a picker that takes turns. `pool` is read for its list of engines
    /// alone.
    ///
    /// A pool under [`DRAW_FROM`] rows is asked about only where its host
    /// expects every row kept ([`Self::asks_standing`]).
    #[cfg(feature = "in-process")]
    pub(crate) fn drawn_for(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        rows: usize,
    ) -> Option<Drawn> {
        if rows < self.written_least || !facts.affinity.is_empty() {
            return None;
        }
        let (entry, index, _) = self.resolve(facts.model, facts.phase)?;
        let at = usize::from(index);
        if rows < self.written_from.get(at).copied().unwrap_or(usize::MAX) {
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
        if !marks.asks_nothing_first() || encode {
            return None;
        }
        if marks.first() & (Marks::ENGINES | Marks::OWNS) != 0 && !pool.engines.is_empty() {
            return None;
        }
        let ahead = marks.ahead();
        let draws =
            ahead == Marks::DRAWS_FIRST || (facts.named.is_some() && ahead & Marks::JOINED != 0);
        if !draws {
            return None;
        }
        // Under a rule that leaves a key with no pin to the profile, the
        // whole decision places such a key by the list's own turn behind
        // its pass ([`Self::pool_turn`]): that turn is this draw. Held to
        // a profile whose filters fold into one test of a row, which says
        // of a row what the pass says of it. The key is asked about last:
        // a request with none pays the tests it paid before.
        if keyed
            && !(marks.turns()
                && entry.placing == Assignment::Delegate
                && self.row_tests[at].is_some())
        {
            return None;
        }
        Some(Drawn {
            index,
            vetoed: entry.vetoed,
            claims: keyed,
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
    /// [`Self::decide`] makes over the same rows written whole, from
    /// [`DRAW_FROM`] rows on. Under that size [`Self::decide`] reads such
    /// a pool whole and this is its picker's draw ([`DRAWS_TWO`]): the
    /// same rule and, with every row kept, the same two rows; behind it
    /// the trace counts no survivors, as behind every draw. No lock and no
    /// allocation beyond the stages' own.
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
        // The draw's two words go apart, a register each: the three
        // fields of a `Drawn` are handed as one packed word.
        run_drawn_written(
            self,
            drawn.index,
            drawn.vetoed,
            facts,
            pool,
            fixed,
            rows,
            write,
            cx,
            &mut scratch.single,
        )
    }

    /// [`Self::decide_drawn`] for a draw that claims ([`Drawn::claims`]): a
    /// keyed request under a sessions rule that leaves a key with no pin to
    /// a profile that takes turns. The steps are the whole decision's, in
    /// its order ([`Self::decide`]), with the row the list's word points at
    /// in the place of the pass over the pool: that row is read with
    /// nothing added to the word, and where the profile's filters keep it
    /// the pool is not emptied; the key is read, which starts its idle
    /// time; the draw takes the list's turn; and the worker the draw leads
    /// with is claimed for the key. A claim another request took first is
    /// settled as the whole decision settles it: its worker leads where
    /// the filters keep its row, and no alternate at the rule's cap stands
    /// behind it.
    ///
    /// `None` where the whole decision answers by another way, with no
    /// turn taken and nothing written that it does not write again at the
    /// same instant: a pool that lends no word, a retry, a request held to
    /// places, a key that holds a pin, and a turn whose row is refused,
    /// which says nothing of the pool. The host asks it over rows that each
    /// carry an id of their own: behind a standing claim the rule's cap is
    /// asked of the rows the draw kept. `None` too, behind the draw, where
    /// the standing claim's worker is at the rule's cap: the whole decision
    /// hands the key on by one more placement. The host then writes every
    /// row and asks [`Self::decide`].
    ///
    /// Out of line: a draw that claims nothing holds one test and none of
    /// this.
    #[cfg(feature = "in-process")]
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "one decision's inputs, and the host's two halves of a row, as decide_drawn takes them"
    )]
    pub(crate) fn decide_claimed(
        &self,
        drawn: Drawn,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        fixed: &[Candidate],
        rows: &mut [Candidate],
        write: &mut dyn FnMut(&Candidate, &mut Candidate),
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Option<Decision> {
        use std::sync::atomic::Ordering;

        let (word, tests) = (pool.word?, self.row_tests[usize::from(drawn.index)]?);
        if facts.attempt != 0 || !facts.tried.is_empty() || places::stated(cx) {
            return None;
        }
        let (entry, ..) = self.resolve(facts.model, facts.phase)?;
        let rule = entry.binding.sessions.as_ref()?;
        let key = facts.session();
        let free = [WorkerId(0); 2];
        // A pin that is down, at its cap or moved is the whole decision's.
        if self.sessions.pins_at(key, facts.now_micros) != free {
            return None;
        }
        let keeps = |row: &Candidate| {
            tests.keeps(facts, row, cx) && !(tests.masked(cx) && cx.masks(row.position))
        };
        let place = word.load(Ordering::Relaxed) as usize % fixed.len().max(1);
        let (head, row) = (fixed.get(place)?, rows.get_mut(place)?);
        write(head, row);
        let turned = *row;
        if !keeps(&turned) {
            return None;
        }
        // The whole decision reads the key behind a pass that kept a row.
        let pins = self.sessions.lookup(key, facts.now_micros, rule);
        if pins != free {
            return None;
        }
        // One read a row: the draw is handed the record read above.
        let mut once = |fixed: &Candidate, row: &mut Candidate| {
            if fixed.position == head.position {
                *row = turned;
            } else {
                write(fixed, row);
            }
        };
        let mut decision =
            self.decide_drawn(drawn, facts, pool, fixed, rows, &mut once, cx, scratch);
        // A shed and the all-vetoed rule's pick claim nothing.
        let Decision::Pick(pick) = &mut decision else {
            return Some(decision);
        };
        let first = pick.targets.first().map(|target| target.id);
        let (Some(first), LiftedVeto::None) = (first, pick.trace.lifted) else {
            return Some(decision);
        };
        pick.session_verdict = SessionVerdict::Pinned;
        match self.sessions.claim(key, pins, first, facts.now_micros) {
            Claim::Made(id) => pick.session_verdict = SessionVerdict::Claimed(id),
            // A full table records nothing of the key.
            Claim::NoRoom => {}
            Claim::Standing(owner, _) => {
                // The first row of the standing worker the pass kept, in
                // the pool's order. Behind a draw that widened the rows
                // kept are every survivor. Behind a draw they are the rows
                // drawn: each row of the worker is found by its id, read
                // where the draw did not keep it, and asked the draw's
                // test.
                let kept = &scratch.single.kept;
                let stands = if pick.trace.drew_first() {
                    let of_owner = fixed.iter().filter(|head| head.id == owner);
                    of_owner
                        .filter_map(|head| {
                            let row = rows.get_mut(head.position as usize)?;
                            if !kept.contains(&head.position) {
                                once(head, row);
                            }
                            Some(*row)
                        })
                        .find(keeps)
                } else {
                    kept.iter()
                        .map(|&row| rows[row as usize])
                        .find(|row| row.id == owner)
                };
                if let Some(row) = stands {
                    if Self::at_cap(rule, &row) {
                        return None;
                    }
                    if owner != first {
                        pick.targets.lead(Target {
                            position: row.position,
                            id: row.id,
                            dp_rank: row.dp_rank,
                            pair: None,
                        });
                    }
                    Self::drop_alternates_at_cap(rule, rows, kept, &mut pick.targets);
                }
            }
        }
        Some(decision)
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
            return shed(ShedReason::PolicyDeclined, BINDING, bare);
        };
        let Entry {
            binding,
            glob: _,
            placing,
            vetoed,
            sides_placed: _,
            #[cfg(feature = "in-process")]
                pair_drawn: _,
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
        // ([`Self::reads_for`]). A mask beside the rows says they are not,
        // and the arm that would open the scope declines.
        let unwritten = || !cx.stalled.is_empty();
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
        let (model, phase) = (facts.model, facts.phase);
        let (entry, placing) = self
            .bindings
            .iter()
            .zip(&self.shadow_placing)
            .find(|(entry, _)| entry.binding.matches_as(&entry.glob, model, phase))?;
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
        // The place a request states on a lent ring is the live profile's:
        // a shadow apart from it is lent none ([`Self::unlent`]).
        let live = profile_index(&entry.binding, facts.phase);
        let unlent = self.unlent(index, live.as_slice(), facts);
        let facts = unlent.as_ref().unwrap_or(facts);
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
                Some(rule) => self.decide_with_session::<true, false, false, false>(
                    profile, index, rule, placing, facts, pool, set, cx, scratch,
                ),
                None => self.keep::<false>(profile, index, None, facts, pool, set, cx, scratch),
            });
        }
        if let Some(rule) = rule {
            return Some(self.decide_with_session::<false, false, false, false>(
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

    /// `facts` with no ring lent and no place stated, a pair's decode
    /// side's own among them, where the shadow profile at `shadow`
    /// reads a lent ring by another place than one of the live profiles at
    /// `lives` states on it. The two read a ring by different kinds of
    /// place ([`Marks::ring_apart`]), or both place by heads and cut a
    /// request by different rules ([`HeadPlaces`]). The place a request
    /// states is its live profiles', so it is no place of such a shadow,
    /// which places on its own ring. A profile that reads no ring is apart
    /// from none, and a live profile that reads none states no place: its
    /// shadow under a single binding is stated one of its own inside its
    /// run ([`Self::shadow_ring_profile`]), and so is a side's shadow
    /// under a pair neither side of which reads a ring
    /// ([`Self::side_ring_profile`]).
    ///
    /// `None` where the facts stand: the request lends no ring, or the
    /// shadow is apart from no live profile.
    fn unlent<'a>(
        &self,
        shadow: u16,
        lives: &[u16],
        facts: &RequestFacts<'a>,
    ) -> Option<RequestFacts<'a>> {
        facts.ring?;
        let marks = |at: u16| self.marks.get(usize::from(at)).copied().unwrap_or_default();
        let reads = |at: u16| {
            let places = self.head_places.get(usize::from(at));
            places.filter(|places| **places != HeadPlaces::Unread)
        };
        let apart = |&live: &u16| {
            let rules = (reads(shadow), reads(live));
            let heads = matches!(rules, (Some(own), Some(live)) if own != live);
            marks(shadow).ring_apart(marks(live)) || heads
        };
        lives.iter().any(apart).then_some(RequestFacts {
            ring: None,
            ring_positions: &[],
            decode_ring_positions: None,
            ..*facts
        })
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
        self.decide_with_session::<KEPT, true, true, KEEPS>(
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
    /// With `MOVES` off a pin at the cap is not moved here. The decision is
    /// a pair's prefill side's: it is made ahead of its pair, and the
    /// pair's decision moves the pin where the pair forms
    /// ([`Self::decide_pair`]). A shadow's form states it off as well.
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
    fn decide_with_session<
        const KEPT: bool,
        const WRITE_SESSIONS: bool,
        const MOVES: bool,
        const KEEPS: bool,
    >(
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
                    return self.moved::<WRITE_SESSIONS, MOVES>(key, facts, respilled);
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
                // no claim: the pin stands as it was read. The table
                // answers the same for a pin its decision read: this test
                // spares its lock, and a shadow reads no table. The plain
                // form never names a pin here, and carries no test for one.
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
                            return self.moved::<WRITE_SESSIONS, MOVES>(key, facts, handed);
                        }
                    }
                    return decision;
                }
                let claim = if WRITE_SESSIONS {
                    self.sessions.claim(key, pins, first, facts.now_micros)
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
                                    return self.moved::<WRITE_SESSIONS, MOVES>(key, facts, handed);
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
    /// With `MOVES` off it is a pair's prefill side's, and writes none
    /// either: a prefill whose partners leave no decode is passed over and
    /// takes no request, so the pair's decision writes the move on the
    /// prefill the pair forms on ([`Self::decide_pair`]).
    /// A placement on the pin again, and one on another worker at the cap,
    /// move no pin ([`Self::respill`]). `key` is the key the decision
    /// looked up in `facts`, and the move is told under that request's
    /// instant.
    #[inline]
    fn moved<const WRITE_SESSIONS: bool, const MOVES: bool>(
        &self,
        key: Session,
        facts: &RequestFacts<'_>,
        handed: Decision,
    ) -> Decision {
        if WRITE_SESSIONS && MOVES {
            if let Decision::Pick(pick) = &handed {
                if pick.session_verdict == SessionVerdict::Respill {
                    if let Some(first) = pick.targets.first() {
                        self.sessions.respilled(key, first.id, facts.now_micros);
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
    /// a pair's prefill side where its pair forms ([`Self::decide_pair`]).
    /// Its commit fronts the worker it lands on, whichever of its targets:
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
        let mut trace = trace_drawn(
            index,
            facts,
            pool,
            candidates,
            &buf.before_soft,
            buf.traced(),
        );
        trace.survivors = Survivors::counted(buf.kept.len() as u32);
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
        let trace = trace_of(index, facts, pool, candidates, &buf.kept, buf.traced());
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
    /// session key under a rule that covers the side
    /// ([`SessionRule::sides`]), and every request of such a picker. The
    /// entry's byte is asked first ([`Entry::sides_placed`]): a pair under
    /// no rule, each side's picker its own, runs its two profiles as it
    /// did, behind that one test. A side the rule does not cover is its
    /// profile's: the rule's run alone reads a pin, claims one or leaves a
    /// verdict for a dispatch to write by. A document's check keeps a
    /// picker that leaves the placement to the rule on a side it covers.
    #[inline(always)]
    fn places_side(
        &self,
        entry: &Entry,
        index: u16,
        side: PairSide,
        facts: &RequestFacts<'_>,
    ) -> bool {
        let covers = |rule: &SessionRule| rule.sides.covers(side == PairSide::Decode);
        entry.sides_placed
            && ((entry.binding.sessions.as_ref().is_some_and(covers)
                && !facts.session_key.is_none())
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
    /// A decode side's pin at the cap moves where the side is decided
    /// ([`Self::moved`]): a decode's pick forms the pair. A prefill side's
    /// does not: the prefill it places first may be passed over, and the
    /// pair's decision moves the pin where the pair forms.
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
                .decide_with_session::<false, true, DECODE, false>(
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
    /// Under the binding's sessions rule each side the rule covers follows
    /// its key's pin by the side's own session. A side whose picker leaves
    /// the placement to the rule is placed by the assignment. The pick
    /// states the prefill side's verdict, and its first target's pair the
    /// decode side's
    /// ([`Pair::decode_verdict`](super::decision::Pair::decode_verdict)).
    /// A claim of the prefill side names the prefill the pair formed on,
    /// before any dispatch, and a decision that sheds leaves none.
    /// A prefill side's pin at the cap moves to that prefill where the
    /// pair forms, and a decision that forms no pair moves none.
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
    /// left. Under a binding that sheds, such a pool opens no prefill, and
    /// the request is shed by the one question of a pair's shed, below.
    /// The prefill side's own run is put to no rule. Under a binding that
    /// steers, a prefill profile that weighs load against affinity runs with
    /// the overload veto lifted (`Engine::weighing_lift`), and a pair no
    /// prefill of which stood without the lift says that veto in its trace.
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
    /// Behind the pair's default and its question of saturation, where no
    /// prefill at the request's places stands but for its limit, the limit
    /// names no shed: its name is the word a host waits on for room. The
    /// prefill side's stage is read there with the limit set aside
    /// (`vetoed::Half::emptied_limit_aside`): a prefill at its limit stays
    /// in that pass until another filter refuses it. So where the prefill
    /// side's profile lists the limit among its filters changes no answer
    /// there, under filters whose word on a row does not turn on how
    /// often they are asked. The gateway's own pair selection asks its
    /// overload shed ahead of its capacity test (`pair_pass` in
    /// `model_gateway/src/routers/common/placement.rs`).
    ///
    /// Where the prefill side's run sheds, no default pairs and no prefill
    /// waits on its limit alone, one question says whether the request is
    /// shed as saturated: the count of the one list the gateway's own pair
    /// selection judges (`vetoed::Half::count`). That list is every decode
    /// at the request's places, a decode that pairs with no prefill too,
    /// where a prefill at those places that pairs with a decode of the pool
    /// stands as it is, its limit aside and with no lift. Else it is those
    /// prefills. Under a binding that sheds, the shed is saturated exactly
    /// where that list is wholly over its thresholds, and the decision
    /// says the list (`ShedOver`). Under a binding that steers no such
    /// shed is saturated. A picker that declines keeps its own reason.
    /// Every other shed there has no endpoint, under
    /// the name of the stage that left a pass nobody. Where that stage is
    /// the overload veto, whose name is the word for saturation, the shed
    /// takes the name of the stage that refused a worker with room of the
    /// list judged (`vetoed::Half::refused_room`): the health stage for a
    /// worker under a liveness doubt, and `partnered` for one that stands
    /// and pairs with nobody that stands. The gateway's own pair selection
    /// judges so (`pair_pass` and `failure_from` in
    /// `model_gateway/src/routers/common/placement.rs`: the pass's two
    /// branches, where no prefill is eligible and where none is open).
    ///
    /// The question is asked over whole lists alone. Behind a part of a
    /// list (`PairPools::part`: a hard demand its door judged, or a
    /// request the encode pass is laid for) it is not asked. The shed
    /// keeps the name of the stage that left a pass nobody: it is
    /// saturated where that stage is the overload veto. It says its list
    /// there as it did before the question stood: the prefills that pair
    /// where no prefill stands, and else the decode list whole, each
    /// where wholly over its thresholds. That selection judges another
    /// list there than this decision is handed (`admitted_pool` in the
    /// same file), so the question over the part would answer for
    /// another list.
    ///
    /// Each shed of that question names the side that selection refuses
    /// for ([`Shed::side`](crate::engine::decision::Shed::side)). Three
    /// tests of the lists whole stand ahead of its question in
    /// `pair_pass`, in this order. A prefill pool that holds no worker is
    /// refused for the prefill side. A decode pool that holds none is
    /// refused for the decode side. So is a pool no prefill of which
    /// pairs with any decode, by its index's refusal (`PairingRefusal` in
    /// `model_gateway/src/routers/common/pair_index.rs`). Else the side is
    /// the list the question judges. Behind a part the same tests name
    /// the side over the part, and it may be another than that
    /// selection's role: that selection tests its lists whole.
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
    ///
    /// Where the decode side states a number
    /// ([`Constraints::target_worker`] of `decode_cx`), the pair's decode
    /// is the one that number reaches among the partners of the prefill
    /// chosen, at the decode side's places: its first partners where the
    /// pair is made among those. Where the number reaches none, or the
    /// decode it reaches is shed for want of an endpoint or as over its
    /// thresholds, the pair is shed under `target-worker` and the partners
    /// of no other prefill are asked. The pair's default reads no number.
    /// A number beside no places of the decode side is declined under that
    /// name. The gateway's own pair selection counts the open partners of
    /// its one pick so, and reads no header for its default
    /// (`model_gateway/src/routers/common/placement.rs:2884-2887`,
    /// `:2914-2920`, `:2729-2747`, on main `4b41f39a7`).
    pub fn decide_pair(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        cx: &Constraints<'_>,
        decode_cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        // The word of the list a shed is made over is this decision's: one
        // an earlier decision left goes here ([`Scratch::shed_over`]).
        scratch.shed_over = None;
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
        if counts_no_place(decode_cx) {
            return number_not_counted(bare, bare.profile);
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
            return shed(ShedReason::PolicyDeclined, BINDING, bare);
        };
        // The binding's word for a pool a soft veto empties: the live
        // decode side is put to the all-vetoed rule by it ([`Set::ruled`]).
        let (binding, vetoed) = (&entry.binding, entry.vetoed);
        let decode_index = binding.decode_profile.unwrap_or(binding.profile);
        let (Some(prefill_profile), Some(decode_profile)) = (
            self.profiles.get(usize::from(binding.profile)),
            self.profiles.get(usize::from(decode_index)),
        ) else {
            return shed(ShedReason::PolicyDeclined, BINDING, bare);
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
        let mut open = places::rows(decodes, decode_cx)
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
            .partnered(open.by_ref(), &mut scratch.classes, &mut scratch.marks);
        // The pass may end ahead of the last decode. A filter the engine
        // does not hold itself is asked about every decode all the same,
        // in the pass's order: what it keeps of its asks stands.
        let asked_whole = self.row_tests[usize::from(decode_index)].is_none();
        if asked_whole {
            open.for_each(drop);
        }
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
                let mut kept = places::rows(decodes, decode_cx)
                    .filter(|d| d.pair_class != 0 && hard().all(|f| f.keep(&all, d, &pass)))
                    .map(|d| d.pair_class);
                pools
                    .table
                    .partnered(kept.by_ref(), &mut scratch.classes, &mut scratch.marks);
                if asked_whole {
                    kept.for_each(drop);
                }
            }
        }
        // The prefill buffer's mark of a lifted veto is cleared for this
        // decision. The pair's default alone writes it again, by the tier
        // it paired in ([`vetoed::paired`]): each side's shadow reads there
        // whether the default made the pair ([`Self::shadow_side`]).
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
        let mut prefill =
            if self.places_side(entry, binding.profile, PairSide::Prefill, &prefill_facts) {
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
                    if takes.waits_among(&scratch.firsts) {
                        let limit = HasCapacity.name();
                        return shed(reason_of(limit), limit, refused.trace)
                            .of_side(PairSide::Prefill);
                    }
                    false
                }
                // A picker's decline passes through, as the prefill side's.
                _ => return prefill.of_side(PairSide::Prefill),
            };
            if stands {
                // The classes the prefill profile was shown stand in
                // `classes`, as a side's shadow reads them.
                std::mem::swap(&mut scratch.classes, &mut scratch.firsts);
            } else {
                firsts = false;
                prefill = if self.places_side(
                    entry,
                    binding.profile,
                    PairSide::Prefill,
                    &prefill_facts,
                ) {
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
        // Whether the pair is made among first partners, left for the
        // decode's shadow: it reads the partners this decision's decode
        // side reads ([`Self::shadow_side`]).
        scratch.among_first = firsts;
        // The prefill's pick is read where its decision stands: a pick is
        // 256 bytes, and taking it out copies them.
        let Decision::Pick(prefill_pick) = &prefill else {
            // A prefill pool its filters emptied goes to the pair's default,
            // then to the admission limit's shed, for a prefill with an open
            // partner alone, and then to the one question that says whether
            // the request is shed as saturated: the count of the list the
            // gateway's own pair selection judges, asked over whole lists
            // alone. Every other shed is named by a stage. It is the
            // prefill side's where no prefill stands but for its admission
            // limit, read with that limit set aside, and "none" where that
            // pass ends at no stage. It is "partnered" for a class with no
            // decode at all and for a routable prefill of class 0, and else
            // what the decode pool's own stages say about the partners of
            // the prefills that stand so (those were dropped by the
            // partnered test alone). Where those stages name none, the
            // prefill's own shed stands. Over whole lists each such shed
            // has no endpoint, and where its stage is the overload veto,
            // the list judged names it again ([`Half::refused_room`]).
            // Behind a part a pass the overload veto ended is shed as
            // saturated. A picker's decline passes through.
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
                    // The decision says the list it sheds over as saturated
                    // under a binding that sheds a list wholly over its
                    // thresholds, and under no other ([`ShedOver`]).
                    let says = vetoed == AllVetoed::Shed;
                    let words = (vetoed == AllVetoed::Steer, self.weighs(first));
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
                    if takes.waits_among(&scratch.classes) {
                        let limit = HasCapacity.name();
                        return shed(reason_of(limit), limit, prefill_shed.trace)
                            .of_side(PairSide::Prefill);
                    }
                    // No stage placed the pair, no default paired it and no
                    // prefill waits on its limit alone. The prefills the
                    // request may take that stand, the limit set aside: one
                    // outside its side's places is no routable prefill of
                    // it, and one at its limit with no open partner is named
                    // by its partners' stage as any other. A prefill is
                    // routable under the constraints the run itself read.
                    // Their classes are written over the classes with an
                    // open partner, which nothing below reads.
                    let has_partner = |c: &Candidate| vetoed::pairs_in(c, &pools.table, decodes);
                    // Whether a prefill of the pool pairs with a decode of
                    // it at all, in whatever state, asked once: where none
                    // does, nothing below looks for a prefill's partner
                    // again.
                    let any_pairs = prefills.iter().any(&has_partner);
                    let pairs = |c: &Candidate| any_pairs && has_partner(c);
                    // Whether one of them that pairs with a decode of the
                    // pool stands in the pool as it is with no lift: behind
                    // a weighing profile's lifted run the stated
                    // constraints are asked of it once more, as the default
                    // above read them.
                    let unlifted =
                        |c: &Candidate| lifted_run.is_none() || sides[0].but_for_the_limit(c);
                    let (mut routable, mut pair_stands) = (0usize, false);
                    scratch.classes.clear();
                    for c in takes.places().filter(|c| takes.but_for_the_limit(c)) {
                        routable += 1;
                        if c.pair_class != 0 {
                            scratch.classes.push(c.pair_class);
                        }
                        pair_stands = pair_stands || (pairs(c) && unlifted(c));
                    }
                    scratch.classes.sort_unstable();
                    scratch.classes.dedup();
                    // One question says whether the shed is saturation, the
                    // gateway's own pair selection's (`pair_pass`,
                    // `model_gateway/src/routers/common/placement.rs`,
                    // where no prefill is eligible or open). It judges one
                    // list: every decode at the request's places, a decode
                    // that pairs with no prefill too, where a prefill at
                    // those places that pairs with a decode of the pool
                    // stands; and else those prefills. Under a binding that
                    // sheds, the shed is saturated exactly where that list
                    // is wholly over its thresholds ([`Half::count`]), and
                    // the decision says the list. Under a binding that
                    // steers no shed here is saturated: that selection
                    // sheds for overload under its shed option alone
                    // (`failure_from` there). The count reads the
                    // list's rows once more, by the health stage's and the
                    // overload veto's tests, behind a shed alone. For the
                    // prefills a decode that pairs with each is looked for
                    // too, at worst every decode once a prefill.
                    let judged = || {
                        if pair_stands {
                            (sides[1].count(sides[1].places()), ShedOver::Decodes)
                        } else {
                            let paired = sides[0].places().filter(|&c| pairs(c));
                            (sides[0].count(paired), ShedOver::Prefills)
                        }
                    };
                    // Every shed below names one side, the role that
                    // selection refuses for. It asks three tests of its
                    // lists whole ahead of its question, in this order: a
                    // prefill pool that holds no worker is refused for the
                    // prefill's role, a decode pool that holds none for the
                    // decode's, and a pool no prefill of which pairs with
                    // any decode for the decode's too. Else the side is the
                    // list the question judges.
                    let side = if pair_stands {
                        PairSide::Decode
                    } else if prefills.is_empty() || any_pairs {
                        PairSide::Prefill
                    } else {
                        PairSide::Decode
                    };
                    // The question is asked over whole lists alone. Behind a
                    // part ([`PairPools::part`]) that selection judges a
                    // wider list than this decision is handed: the
                    // workers on the request's wire, and every worker
                    // where none is on it (`admitted_pool` there). So no
                    // shed behind a part is made saturated by the count,
                    // and a pass the overload veto ended keeps its name
                    // below.
                    let part = pools.part;
                    let asked = (says && !part).then(&judged);
                    if let Some((Count::WhollyOver(_), over)) = asked {
                        let veto = NotOverloaded.name();
                        scratch.shed_over = Some(over);
                        return shed(reason_of(veto), veto, prefill_shed.trace).of_side(side);
                    }
                    // Else the shed takes a first name. Where no prefill of
                    // the request's stands but for its limit it is the
                    // stage that leaves the prefill side's pass nobody, the
                    // limit aside, and `none` where that pass ends at no
                    // stage. Where those that stand pair with no decode it
                    // is `partnered`. Else it is what the decode pool's own
                    // stages say about the partners of the prefills that
                    // stand, and the prefill's own shed stands where they
                    // name none. A first name that is the overload veto's
                    // sheds as saturated behind a part, and over whole
                    // lists gives way to the name the list judged gives
                    // it, below. Every other shed has no endpoint.
                    let named = if routable == 0 {
                        // The limit's name says a wait for room, and no
                        // prefill here stands but for its limit. So the
                        // side's stage is read by a pass that sets the
                        // limit aside, wherever the profile lists it: a
                        // prefill at its limit stays in that pass until
                        // another filter refuses it.
                        let buf = &mut scratch.single;
                        Some(takes.emptied_limit_aside(buf).unwrap_or(NO_STAGE))
                    } else if scratch.classes.is_empty() {
                        Some(PARTNERED)
                    } else {
                        // Their partners are the decodes of the classes they
                        // pair with. An empty decode pool is that class for
                        // every prefill: its "none" leaves the prefill's own
                        // shed standing.
                        pools.table.listed(&mut scratch.classes, &mut scratch.marks);
                        let partners = Set::side(decodes, None, Some(&scratch.classes));
                        let buf = &mut scratch.decode;
                        filter(decode_profile, &decode_facts, decode_cx, buf, partners)
                            .filter(|&stage| stage != NO_STAGE)
                    };
                    let reason = named.map_or(prefill_shed.reason, reason_of);
                    // A pass the overload veto ended names that veto, the
                    // engine's word for saturation, which the count did not
                    // bear. The shed has no endpoint, under the stage that
                    // refused a worker with room of the list judged
                    // ([`Half::refused_room`]). Behind a part no list was
                    // judged, and the pass's name stands. The decision
                    // says its list there as it did before the question
                    // stood, where a stage named the shed: the prefills
                    // that pair where no prefill stands, and else the
                    // decode list whole, each where wholly over its
                    // thresholds.
                    if reason == ShedReason::Saturated && part && says && named.is_some() {
                        let (read, over) = if routable == 0 {
                            let paired = takes.places().filter(|&c| pairs(c));
                            (takes.count(paired), ShedOver::Prefills)
                        } else {
                            (sides[1].count(sides[1].places()), ShedOver::Decodes)
                        };
                        if let Count::WhollyOver(_) = read {
                            scratch.shed_over = Some(over);
                        }
                    }
                    if reason == ShedReason::Saturated && !part {
                        let (read, over) = asked.unwrap_or_else(&judged);
                        let judged = &sides[usize::from(over == ShedOver::Decodes)];
                        let stage = judged.refused_room(read);
                        return shed(ShedReason::NoEndpoints, stage, prefill_shed.trace)
                            .of_side(side);
                    }
                    if let Some(stage) = named {
                        return shed(reason, stage, prefill_shed.trace).of_side(side);
                    }
                    // The prefill's own shed stands, where the decodes at
                    // the request's places hold nobody or keep a partner.
                    return prefill.of_side(side);
                }
            }
            // A picker's decline passes through, as the prefill side's.
            return prefill.of_side(PairSide::Prefill);
        };
        // A pick the weighing lift alone made possible says so, as a kept
        // key's pick does ([`Self::keep`]): where the veto as it stands
        // leaves the prefills nobody, the pair's trace names the lifted
        // overload veto. The prefill buffer takes no mark: the pair is the
        // profile's, and each side's shadow runs beside it
        // ([`Self::shadow_side`]). A pick beside a prefill under its
        // thresholds carries no mark.
        let lift_placed = lifted_run.is_some() && {
            let shown = Set::side(prefills, None, Some(&scratch.classes));
            let buf = &mut scratch.single;
            vetoed::all_vetoed(prefill_profile, &prefill_facts, shown, cx, buf)
        };
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
            let mut decision =
                if self.places_side(entry, decode_index, PairSide::Decode, &decode_facts) {
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
                    // Each side's own verdict and its own word of who
                    // placed it: the decode side's verdict in the first
                    // target's pair and its word in every pair, the
                    // prefill side's as the pick's.
                    let placed = prefill_pick.targets.placing();
                    targets.pair_under(prefill_target, *session_verdict, placed);
                    // The pair's trace is the prefill's, with the mark of
                    // the veto the rule lifted for the decode side, or of
                    // the overload veto the prefill's own run was placed
                    // past. The decode side's own mark stands where it names
                    // one. A first decision puts that side to the rule's
                    // overload tier alone, so the two marks differ only
                    // under a filter whose `retain` drops a row its `keep`
                    // keeps, which [`Filter::retain`] forbids: the liveness
                    // mark of such a decode side is kept.
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
                            let at = prefill_facts.now_micros;
                            *session_verdict = self.sessions.move_claim(key, claimed, paired, at);
                        }
                    }
                    // The prefill side's run met its key's pin at the cap
                    // and placed the request again, and moved no pin
                    // ([`Self::moved`]). The pin moves here, where the
                    // pair forms and ahead of any dispatch, to the
                    // prefill the request goes to. A prefill passed over
                    // is a pin at no instant, so no pin is taken back,
                    // and none another request was sent to meanwhile.
                    if prefill_pick.session_verdict == SessionVerdict::Respill {
                        let (key, at) = (prefill_facts.session(), prefill_facts.now_micros);
                        self.sessions.respilled(key, prefill_target.id, at);
                    }
                    return decision;
                }
                Decision::Shed(shed) => {
                    last_shed = Some(*shed);
                    // The decode a request names is counted behind the one
                    // prefill: where it is shed, no other prefill's
                    // partners are asked.
                    if decode_cx.target_worker.is_some() {
                        break;
                    }
                }
                // A queue returns with the prefill side's claim standing.
                // No stage issues one today.
                Decision::Queue(_) => return decision,
            }
        }
        // No prefill paired: the claim the prefill side's run made is
        // withdrawn, and the key's next request is placed again.
        if let SessionVerdict::Claimed(claimed) = prefill_pick.session_verdict {
            let (key, at) = (prefill_facts.session(), prefill_facts.now_micros);
            self.sessions.move_claim(key, claimed, WorkerId(0), at);
        }
        // A prefill was picked and no decode pairs with it: the decode side
        // is the one left with nobody.
        match last_shed {
            // The shed of the one decode a request names takes the
            // target's name.
            Some(last) if decode_cx.target_worker.is_some() => target_missed(Decision::Shed(last)),
            Some(last) => Decision::Shed(last),
            None => shed(ShedReason::NoEndpoints, PARTNERS, prefill_pick.trace),
        }
        .of_side(PairSide::Decode)
    }

    /// Whether some binding's pair may be asked by its draw: what a pair's
    /// decision asks first, at one load ([`Self::pair_drawn_for`] asks the
    /// rest of the request's own binding).
    #[cfg(feature = "in-process")]
    pub(crate) const fn pairs_drawn(&self) -> bool {
        self.pairs_drawn
    }

    /// Whether each side of a pair may be
    /// asked by its draw ([`run_side_drawn_written`]), which writes the records
    /// it asks about and no other. Both profiles are a pick of two whose
    /// filters fold into one test of a row, each list holds its profile's
    /// floor of rows, and the request states nothing the pair's whole
    /// decision reads beside its two picks. Asked before a record is
    /// written; answers the two profiles.
    #[cfg(feature = "in-process")]
    pub(crate) fn pair_drawn_for(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        rows: [usize; 2],
        cx: &Constraints<'_>,
    ) -> Option<PairDrawn> {
        if carries_a_column(facts) || counts_no_place(cx) || !facts.session_key.is_none() {
            return None;
        }
        let entry = self.pair_entry(facts.model)?;
        if !entry.pair_drawn {
            return None;
        }
        let binding = &entry.binding;
        let decode_index = binding.decode_profile.unwrap_or(binding.profile);
        for (index, rows) in [(binding.profile, rows[0]), (decode_index, rows[1])] {
            let at = usize::from(index);
            if rows < self.written_from.get(at).copied().unwrap_or(usize::MAX)
                || (self.marks[at].first() & (Marks::ENGINES | Marks::OWNS) != 0
                    && !pool.engines.is_empty())
            {
                return None;
            }
        }
        if self
            .weighing_lift(binding.profile, entry.vetoed, cx)
            .is_some()
        {
            return None;
        }
        Some(PairDrawn {
            profiles: [binding.profile, decode_index],
        })
    }

    /// A pair's decision where each side is
    /// asked by its draw: [`Self::pair_drawn_for`] said yes, both lists
    /// stand, and every row of a side pairs with every row of the other.
    /// The prefill by the draw of its profile, then the decode by the draw
    /// of its own, joined as [`Self::decide_pair`] joins its two picks.
    /// `None` where a side's draw answers no pick: the caller then decides
    /// whole, over records it writes itself.
    #[cfg(feature = "in-process")]
    #[expect(
        clippy::too_many_arguments,
        reason = "the pair's facts, each side's rows with its writer, and the engine's scratch"
    )]
    pub(crate) fn decide_pair_drawn(
        &self,
        drawn: PairDrawn,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        prefills: DrawnSide<'_>,
        decodes: DrawnSide<'_>,
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Option<Decision> {
        let PairDrawn {
            profiles: [prefill_index, decode_index],
        } = drawn;
        scratch.shed_over = None;
        scratch.single.lifted = LiftedVeto::None;
        let prefill_facts = RequestFacts {
            phase: Phase::Prefill,
            named: None,
            signal_cells: None,
            decode_ring_positions: None,
            ..*facts
        };
        let decode_facts = RequestFacts {
            phase: Phase::Decode,
            seed: Draw::new(facts.seed).next_u64(),
            named: None,
            signal_cells: None,
            ring_positions: facts.decode_ring_positions.unwrap_or(facts.ring_positions),
            decode_ring_positions: None,
            ..*facts
        };
        let (fixed, rows, write) = prefills;
        let prefill_rows = fixed.len();
        let prefill = run_side_drawn_written(
            self,
            prefill_index,
            &prefill_facts,
            pool,
            fixed,
            rows,
            write,
            cx,
            &mut scratch.single,
        );
        let prefill_pick = prefill?;
        let prefill_target = prefill_pick.targets.iter().next()?;
        let (fixed, rows, write) = decodes;
        let mut decision = Decision::Pick(run_side_drawn_written(
            self,
            decode_index,
            &decode_facts,
            pool,
            fixed,
            rows,
            write,
            cx,
            &mut scratch.decode,
        )?);
        let Decision::Pick(paired) = &mut decision else {
            return None;
        };
        let Pick {
            targets,
            trace,
            session_verdict,
        } = paired;
        let placed = prefill_pick.targets.placing();
        targets.pair_under(prefill_target, *session_verdict, placed);
        let lifted = trace.lifted;
        *trace = prefill_pick.trace;
        // The whole pass counts the prefills it keeps, and a pair that
        // draws is asked only where it keeps every one of them.
        trace.survivors = Survivors::counted(prefill_rows as u32);
        trace.lifted = lifted;
        *session_verdict = prefill_pick.session_verdict;
        Some(decision)
    }

    /// A pair's answer over two lists of which one holds no row: one side
    /// alone is no pair, and the engine answers over no candidate, under
    /// the request's own constraints `cx` for both sides. `pool` holds
    /// the prefill list's facts, and `prefills` says whether that list
    /// holds a row. The refusal names the side of the list with none, the
    /// prefills' first. The engine read neither list, so where its refusal
    /// names a side it names the prefills': where that list holds a row,
    /// the list with none is the decodes'.
    ///
    /// A pair's decision in process answers so ahead of the request's
    /// demand. An encode request's answers so where its demand left the
    /// other list nobody, with an item behind its encoders' own answers:
    /// its caller reads the answer of its pair alone. Cold and out of
    /// line: its code stays off the path of a pair both of whose lists
    /// hold a row, and so does the arm that calls it.
    #[cold]
    #[inline(never)]
    pub(crate) fn decide_over_an_empty_list(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        prefills: bool,
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let none = PoolFacts {
            generation: pool.generation,
            ..PoolFacts::default()
        };
        let mut decision = self.decide_pair(facts, &none, PairPools::default(), cx, cx, scratch);
        if let Decision::Shed(shed) = &mut decision {
            if shed.side.is_some() && prefills {
                shed.side = Some(PairSide::Decode);
            }
        }
        decision
    }

    /// Whether the profile at `index` weighs load against affinity by a
    /// gate of its own: one of its scorers says so
    /// ([`Scorer::balances_load_against_affinity`]), read into the
    /// profile's marks where the engine is built. The one reader of that
    /// mark. A pair's decision asks it twice of its prefill profile: for
    /// the lifted run ([`Self::weighing_lift`]) and for the word its
    /// default is handed ([`vetoed::paired`]). What a stage is fed is
    /// another word ([`Needs::affinity`]): a stage that reads the column
    /// and has no gate lifts nothing.
    pub(crate) fn weighs(&self, index: u16) -> bool {
        let marks = self.marks.get(usize::from(index));
        marks.is_some_and(|marks| marks.weighs())
    }

    /// The constraints a pair's prefill profile at `index` runs under where
    /// it weighs load against affinity: `cx` with the overload veto lifted.
    /// `None` where the stated ones stand: under a binding that sheds
    /// (`rule`), and under a profile none of whose scorers says that it
    /// weighs ([`Self::weighs`]).
    ///
    /// A scorer that weighs has a gate of its own, which moves a request
    /// off a holder loaded well past the pool. The veto would move a
    /// session off the prefill that holds its prefix each time that prefill
    /// crosses a threshold, and under sustained load the prefills cross
    /// them in turn. So such a profile places among the prefills over their
    /// thresholds too, as the gateway's own pair selection leaves the veto
    /// off a prefill pool whose policy says the same word of itself
    /// (`model_gateway/src/routers/common/placement.rs:2575-2577`, at main
    /// `82a135b1b`). The lift is the constraint a kept key's run is given
    /// ([`Self::keep`]). A profile whose stages only read the affinity
    /// column keeps the veto: a need says what a stage is fed, and nothing
    /// of a gate.
    fn weighing_lift<'a>(
        &self,
        index: u16,
        rule: AllVetoed,
        cx: &Constraints<'a>,
    ) -> Option<Constraints<'a>> {
        let weighs = rule == AllVetoed::Steer && self.weighs(index);
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
    /// ([`side_shadows`]). That shadow is then due beside each decision
    /// the pair's profiles make, and a host says one that
    /// [`Self::shadow_side`] does not run as skipped.
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
    /// live pair's prefill, never of a prefill of its own: its first
    /// partners where the pair was made among them, by the word that
    /// decision left in the scratch ([`Scratch::among_first`]). `at` is
    /// the pair binding's position ([`Self::shadow_for`]) and `live` the
    /// pair's pick, made over the same `pools`. The run writes the
    /// scratch and the shadow profile's stages, and nothing a commit
    /// reads.
    ///
    /// A side's shadow is lent the ring its request lends by the rule of a
    /// single's shadow ([`Self::shadow`]), held against the word its
    /// side's list was stated by. Where the pair is lent one list of
    /// places for its two sides ([`Self::states_key_place`]), that is
    /// both live sides' word: a shadow that reads a ring by another place
    /// than either live side states is lent none. Where each side is
    /// stated its own places ([`RequestFacts::decode_ring_positions`]),
    /// it is the side's own live profile's word: a shadow that reads the
    /// ring as its live side does is lent the ring with its side's
    /// places ([`PairPools::side`]), and one that reads it as the other
    /// side does is lent none. A shadow lent none places on its own ring
    /// ([`Self::unlent`]).
    ///
    /// A shadow that reads a ring where neither live side does is stated
    /// its own place on the ring the request lends
    /// ([`Self::side_ring_profile`]), asked of `lends` behind the filter
    /// half of its run ([`run_side_lent`]): a shadow whose set the filters
    /// emptied, which the all-vetoed rule steers or sheds, asks nothing.
    /// Neither does the pair's default that a side's shadow follows: no
    /// step of it scores.
    ///
    /// A shadow profile with a filter that declares the ring need
    /// ([`Filter::needs`]) asks ahead of the filter half of its run,
    /// whatever its filters then keep, and once a decision: a prefill's
    /// shadow that runs again reads the answer of its first run. The
    /// pair's default asks nothing there either, and its steps show the
    /// shadow's filters no ring and no place.
    ///
    /// The decode's shadow is put to the binding's all-vetoed rule by the
    /// binding's word, as its live side is ([`Set::ruled`]): where a veto
    /// leaves the live prefill's partners nobody, the rule steers the
    /// shadow or sheds it as it does that side. The prefill's shadow is
    /// put to no all-vetoed rule, as its live side is put to none.
    ///
    /// A side's shadow follows the pair's default as its live side does
    /// ([`vetoed::paired`]). It starts where the pair's decision ended, on
    /// the scratch that decision left. A prefill's shadow is shown the
    /// classes its live side was shown last. Where the live side shed among
    /// first partners and ran again over every open class, the shadow's
    /// one run reads every open class.
    ///
    /// A prefill's shadow that sheds goes the live decision's way
    /// ([`Self::decide_pair`]), in this order:
    ///
    /// - Among first partners, the admission limit's test of the prefills
    ///   shown. Where one of them waits on its limit, the shed is the
    ///   limit's and names the prefill side. The test reads the facts the
    ///   shadow's run showed its filters.
    /// - Among first partners, its run over every open class.
    /// - The default's first step, which answers its prefill
    ///   ([`vetoed::shadow_prefill`]). The live decode profile stands on
    ///   the other side.
    ///
    /// Where the default answers nothing, the run's own refusal stands: a
    /// shadow is compared by its outcome.
    ///
    /// The default asks the live decode profile's hard filters behind the
    /// decision they stood in. It is asked only where each of them reads a
    /// row alone and holds no decision scope ([`Half::reads_rows_alone`]).
    /// Another filter may answer otherwise there, and the side says `None`.
    ///
    /// The mark on the prefill's buffer says a pair the default made.
    /// Beside such a pair the decode's shadow is the default's second step
    /// for the live pair's prefill ([`vetoed::paired_decode`]). It reads
    /// the decodes its own profile's hard filters keep in the pass that
    /// paired. Where they keep no partner of that prefill, it sheds under
    /// `partners`, as a decode side with no partner does.
    ///
    /// The prefill's shadow leaves the classes and the mark as it found
    /// them, where it panics too. The decode's shadow runs behind it on
    /// one scratch.
    ///
    /// A pair that a weighing prefill profile's lifted run placed
    /// ([`Self::weighing_lift`]) is a pair of the profiles. Each shadow
    /// runs beside it as beside any other. A prefill's shadow that does
    /// not weigh asks the default there.
    ///
    /// `None` where the binding names no shadow for the side, where the
    /// build did not find that profile free to run beside the live one
    /// ([`side_shadows`]), and where it would be handed another column
    /// than the live side's pool carries ([`Self::side_apart`]).
    ///
    /// `None` as well beside a side its binding's sessions rule or its
    /// picker's assignment placed ([`Self::places_side`]): a shadow's run
    /// is put to neither, and its bare profile set beside a key's pin
    /// would say nothing true of the profile. A request with no key under
    /// a rule is its profiles', and its sides' shadows run.
    ///
    /// `None` too for a prefill's shadow whose default may not be asked,
    /// as said above, and only where the default would be asked. Its own
    /// pick stands, and so does its limit's shed.
    #[expect(
        clippy::too_many_arguments,
        reason = "the pair's entry and its inputs, beside the side and the live pick"
    )]
    #[cfg_attr(
        not(any(feature = "in-process", test)),
        expect(
            dead_code,
            reason = "the one caller is the engine over worker lists, built under `in-process`"
        )
    )]
    pub(crate) fn shadow_side<'l>(
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
        lends: &dyn Fn(u16) -> Option<(&'l dyn LentRing, &'l [u64])>,
    ) -> Option<Decision> {
        let index = self.shadow_sides.get(at as usize)?[side.index()]?;
        if self.side_apart(at, side) {
            return None;
        }
        let profile = self.profiles.get(usize::from(index))?;
        let entry = self.bindings.get(at as usize)?;
        let lives = live_sides(&entry.binding);
        // A side its binding's sessions rule or its picker's assignment
        // placed: a shadow's run is put to neither.
        if self.places_side(entry, lives[side.index()], side, facts) {
            return None;
        }
        // The place stated on a lent ring is the two live sides' where
        // one list is both sides', and the side's own live profile's
        // where each side is stated its own.
        let unlent = self.unlent(index, held_to(&lives, side, facts), facts);
        let facts = unlent.as_ref().unwrap_or(facts);
        let (prefills, prefill_facts) = pools.side(PairSide::Prefill, facts)?;
        let (decodes, decode_facts) = pools.side(PairSide::Decode, facts)?;
        // The shadow's own place, asked behind the filter half, or ahead
        // of it where a filter of the profile declares the ring need. It
        // is asked once: a run of the side that follows reads the answer.
        let asked = OnceCell::new();
        let lent = || *asked.get_or_init(|| self.side_ring_profile(at, side).and_then(lends));
        let ahead = self.marks[usize::from(index)].ring_filter();
        // The mark the pair's decision left on the prefill's buffer: the
        // tier the pair's default paired in, and none beside a pair the
        // profiles made ([`Self::decide_pair`]).
        let defaulted = scratch.single.lifted;
        let steers = entry.vetoed == AllVetoed::Steer;
        Some(match side {
            PairSide::Prefill => {
                // The prefills the live prefill profile was shown: the
                // classes with a partner stand in the scratch as the
                // pair's decision left them ([`Self::decide_pair`]). The
                // shadow's own runs ask no stage of a live profile again.
                // Its decision's scope closed with it, and a stage asked
                // behind it answers by its fallback. The default below
                // asks the live decode profile's hard filters, and only
                // where each reads a row alone. The shadow's run is lifted
                // by its own profile's word, as the live run is by the
                // live profile's ([`Self::weighing_lift`]).
                let lifted_run = self.weighing_lift(index, entry.vetoed, cx);
                let run_cx = lifted_run.as_ref().unwrap_or(cx);
                // A refusal the pair's decision goes on from: a shed that is
                // no picker's decline.
                let refused = |run: &Decision| match run {
                    Decision::Shed(shed) => shed.reason != ShedReason::PolicyDeclined,
                    Decision::Pick(_) | Decision::Queue(_) => false,
                };
                // The shadow's own run over the prefills of `classes`. It asks
                // for its place behind its filter half, or ahead of it
                // ([`run_side_lent`]).
                let own = |classes: &[u32], buf: &mut Buffers| {
                    let set = Set::side(prefills, None, Some(classes));
                    run_side_lent(
                        profile,
                        index,
                        &prefill_facts,
                        pool,
                        set,
                        run_cx,
                        buf,
                        None,
                        ahead,
                        &lent,
                    )
                };
                let mut run = own(&scratch.classes, &mut scratch.single);
                // A run that sheds goes the live decision's way. Among
                // first partners, one of the prefills shown that stands
                // but for its admission limit keeps the request under the
                // limit's stage. Else the side runs again over every open
                // class, which stand in `firsts` behind that decision's
                // swap. The limit's test asks the hard filters on the facts
                // the run showed them: with what the lender answered that
                // run where one of them declares the ring need. The answer
                // stands, and the lender is asked nothing more.
                let firsts_shed = scratch.among_first && refused(&run);
                let waits = firsts_shed && {
                    let stated = with_lent(&prefill_facts, if ahead { lent() } else { None });
                    let shown = stated.as_ref().unwrap_or(&prefill_facts);
                    let takes = Half::new(profile, index, shown, prefills, run_cx);
                    takes.waits_among(&scratch.classes)
                };
                if waits {
                    // The limit's shed names the prefill side, as the live
                    // decision's does.
                    let limit = HasCapacity.name();
                    run = shed(reason_of(limit), limit, *run.trace()).of_side(PairSide::Prefill);
                } else if firsts_shed {
                    run = own(&scratch.firsts, &mut scratch.single);
                }
                // A run that still sheds goes to the pair's default for
                // its prefill, the live decode profile on the other side.
                // The default reads every open class and the constraints
                // as they are stated. The decode side's facts are this
                // run's, with no ring where the shadow is lent none: that
                // step asks the decode profile's hard filters alone, and
                // none of the crate's own reads a ring.
                if !waits && refused(&run) {
                    let live_decode = lives[PairSide::Decode.index()];
                    let live_decode_profile = self.profiles.get(usize::from(live_decode))?;
                    let sides = [
                        Half::new(profile, index, &prefill_facts, prefills, cx),
                        Half::new(
                            live_decode_profile,
                            live_decode,
                            &decode_facts,
                            decodes,
                            decode_cx,
                        ),
                    ];
                    // Those filters stood in the pair's decision, which is
                    // over. One that reads more than a row, or that holds
                    // a decision scope, may answer otherwise behind it:
                    // the side says none.
                    if !sides[1].reads_rows_alone() {
                        return None;
                    }
                    // The step leaves the classes and the mark as it found
                    // them: the decode's shadow reads both behind this one.
                    let words = (steers, self.weighs(index));
                    let table = &pools.table;
                    let taken =
                        vetoed::shadow_prefill(sides, table, words, pool, scratch, defaulted);
                    run = taken.unwrap_or(run);
                }
                run
            }
            PairSide::Decode => {
                let prefill = live.targets.first()?;
                let class = prefills.get(prefill.position as usize)?.pair_class;
                if defaulted == LiftedVeto::None {
                    // The partners of the live pair's prefill, put to the
                    // binding's all-vetoed rule as the live decode side's
                    // are: its first partners where the pair's decision
                    // read those, by the word it left
                    // ([`Scratch::among_first`]).
                    let firsts = scratch.among_first;
                    let partners = Set::side(decodes, Some(pools.partners(&class, firsts)), None);
                    run_side_lent(
                        profile,
                        index,
                        &decode_facts,
                        pool,
                        partners.ruled(entry.vetoed),
                        decode_cx,
                        &mut scratch.decode,
                        None,
                        ahead,
                        &lent,
                    )
                } else {
                    // The pair's default made the live pair: its decode is
                    // the live prefill's usable partner of the fewest
                    // requests, in the pass of the tier the mark names.
                    // The shadow's is that step's answer over the decodes
                    // its own profile's hard filters keep in that pass,
                    // every partner alike.
                    let half = Half::new(profile, index, &decode_facts, decodes, decode_cx);
                    half.usable(defaulted, steers, &mut scratch.decode);
                    let (table, buf) = (&pools.table, &mut scratch.decode);
                    match vetoed::paired_decode::<true>(&half, table, class, pool, buf) {
                        // The shadow's filters kept no partner of the live
                        // prefill in that pass, and the step's pick names
                        // nobody. The shadow sheds as a decode side with
                        // no partner does, and no veto was lifted for it.
                        Decision::Pick(pick) if pick.targets.is_empty() => {
                            let trace = Trace {
                                lifted: LiftedVeto::None,
                                ..pick.trace
                            };
                            shed(ShedReason::NoEndpoints, PARTNERS, trace).of_side(PairSide::Decode)
                        }
                        step => step,
                    }
                }
            }
        })
    }

    /// The decode side of a decode decided after its prefill
    /// ([`Self::decide_decode`]), made once more with the side's shadow
    /// profile over the rows that call decided over: the decodes that
    /// pair with the held `prefill`, on the call's own draw sequence. They
    /// are put to the binding's all-vetoed rule as that call's are, its
    /// liveness tier included ([`Set::ruled`]).
    /// `at` is the pair binding's position ([`Self::pair_position`]). The
    /// run writes the scratch and the shadow profile's stages, and
    /// nothing a commit reads. The shadow is lent the ring its request
    /// lends by the rule of a side's at the pair's decision
    /// ([`Self::shadow_side`]): held against both live sides, or against
    /// the decode's own alone where each side is stated its own places
    /// ([`Self::unlent`]). And it is asked its own place of `lends` as a
    /// side's is there: behind the filter half of its run, or ahead of it
    /// where a filter of the profile declares the ring need.
    ///
    /// `first_decode` is the decode the pair's first decision named, as
    /// that call is handed it. A shadow profile that keeps such a decode
    /// ([`keeps_its_first`]) answers it while its own filters keep it, and
    /// its picker takes no turn there, as the live profile's takes none: a
    /// rotation's shadow stays a turn a request beside its live side. The
    /// test is the shadow profile's own: one that keeps no such decode
    /// decides afresh, whatever its live side keeps.
    ///
    /// `None` where the binding names no shadow for the decode side that
    /// is free to run ([`side_shadows`]), where that shadow would be handed
    /// another column ([`Self::side_apart`]), and where the held prefill
    /// left the rows. The same beside a decode its sessions rule or its
    /// picker's assignment placed ([`Self::places_side`]).
    ///
    /// The run follows that call over the same `scratch`: the call left
    /// there whether it read the held prefill's first partners
    /// ([`Scratch::among_first`]), and the shadow reads the same partners
    /// by that word.
    #[expect(
        clippy::too_many_arguments,
        reason = "the decode entry and its inputs, beside the binding"
    )]
    #[cfg_attr(
        not(any(feature = "in-process", test)),
        expect(
            dead_code,
            reason = "the one caller is the engine over worker lists, built under `in-process`"
        )
    )]
    pub(crate) fn shadow_decode_later<'l>(
        &self,
        at: u32,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        prefill: WorkerId,
        first_decode: Option<(WorkerId, u16)>,
        decode_cx: &Constraints<'_>,
        scratch: &mut Scratch,
        lends: &dyn Fn(u16) -> Option<(&'l dyn LentRing, &'l [u64])>,
    ) -> Option<Decision> {
        let side = PairSide::Decode;
        let index = self.shadow_sides.get(at as usize)?[side.index()]?;
        if self.side_apart(at, side) {
            return None;
        }
        // A decode its binding's sessions rule or its picker's assignment
        // placed, as at the pair's decision.
        let entry = self.bindings.get(at as usize)?;
        let lives = live_sides(&entry.binding);
        if self.places_side(entry, lives[side.index()], side, facts) {
            return None;
        }
        let profile = self.profiles.get(usize::from(index))?;
        // The place stated on a lent ring, by the rule of a side's at the
        // pair's first decision.
        let unlent = self.unlent(index, held_to(&lives, side, facts), facts);
        let facts = unlent.as_ref().unwrap_or(facts);
        let (decodes, mut decode_facts) = pools.side(side, facts)?;
        let prefill = pools.prefills.iter().find(|row| row.id == prefill)?;
        // The call's own draw sequence, as its live decode's.
        decode_facts.seed = facts.seed;
        // The partners the live call read: the held prefill's first
        // partners where it read those, by the word it left.
        let class = prefill.pair_class;
        let firsts = scratch.among_first;
        let partners = Set::side(decodes, Some(pools.partners(&class, firsts)), None);
        Some(run_side_lent(
            profile,
            index,
            &decode_facts,
            pool,
            partners.ruled(entry.vetoed),
            decode_cx,
            &mut scratch.decode,
            first_decode.filter(|_| keeps_its_first(profile)),
            self.marks[usize::from(index)].ring_filter(),
            &|| self.side_ring_profile(at, side).and_then(lends),
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
    ///
    /// A number `decode_cx` states ([`Constraints::target_worker`]) is
    /// counted among those partners at the decode side's places, as a
    /// pair's first decision counts it ([`Self::decide_pair`]): the call
    /// answers the decode it reaches, is shed under `target-worker` where
    /// that decode cannot be named, and is declined under that name where
    /// the number stands beside no places.
    ///
    /// Under a sessions rule the call may claim its decode for the decode
    /// side's key, at this call's `now_micros`. A host that asks this entry
    /// itself tells the side it issues under a lease that states that
    /// instant, or none ([`Lease::issued_micros`]).
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
            return shed(ShedReason::PolicyDeclined, BINDING, bare);
        };
        let binding = &entry.binding;
        let index = binding.decode_profile.unwrap_or(binding.profile);
        let Some(profile) = self.profiles.get(usize::from(index)) else {
            return shed(ShedReason::PolicyDeclined, BINDING, bare);
        };
        // The held prefill is looked for before any constraint is read: a
        // call whose prefill left the lists says so, whatever it states.
        let Some(prefill) = pools.prefills.iter().find(|row| row.id == prefill) else {
            // The decode profile is resolved, so the shed names it.
            let under = Trace {
                profile: index,
                ..bare
            };
            // The held prefill is the side left with nobody.
            return shed(ShedReason::NoEndpoints, PREFILL_GONE, under).of_side(PairSide::Prefill);
        };
        let shown = self.marks[usize::from(index)].scoped();
        if shown && !decode_cx.stalled.is_empty() {
            return rows_not_written(bare, index);
        }
        if counts_no_place(decode_cx) {
            return number_not_counted(bare, index);
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
        // Left for the decode's shadow, which reads the same partners
        // ([`Self::shadow_decode_later`]).
        scratch.among_first = firsts;
        let partners = pools.partners(&class, firsts);
        let partners = Set::side(decodes, Some(partners), None).ruled(entry.vetoed);
        let mut decision = if self.places_side(entry, index, PairSide::Decode, &decode_facts) {
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
                first_decode.filter(|_| keeps_its_first(profile)),
            )
        };
        // Each target names the held prefill with one decode, as the
        // targets of a pair's first decision do. They are turned where the
        // decision stands, and its trace stays the decode's: taken out and
        // built beside it, the pick was copied twice. The decode side's
        // verdict goes into the first target's pair and its word of who
        // placed it into every pair, and the pick states neither of a
        // prefill: this call decided none. A host that holds the pair's
        // first pick keeps that pick's word of the prefill side beside
        // this one.
        if let Decision::Pick(pick) = &mut decision {
            let decode = std::mem::take(&mut pick.session_verdict);
            pick.targets
                .pair_under(&target_of(prefill), decode, Placing::Stages);
            return decision;
        }
        // The decode's run named no worker: the refusal is the decode side's.
        if decode_cx.target_worker.is_some() {
            decision = target_missed(decision);
        }
        decision.of_side(PairSide::Decode)
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
    /// otherwise. A dispatch is told in parts: the counts of sends, this
    /// table, then the stages ([`Self::dispatched_stages`]), so a stage's
    /// call that unwinds leaves the table's entry written. Every teller of
    /// a dispatch keeps that order, the trait's own entry among them.
    ///
    /// The table is handed the instant the lease was issued under
    /// ([`Lease::issued_micros`]). A dispatch elsewhere withdraws the claim
    /// that lease's decision made, and no later decision's claim on the
    /// same worker.
    pub(crate) fn dispatched_table(&self, lease: &Lease, target: &Target, side: Side) {
        let dispatched = match (side, target.pair) {
            (Side::Decode, Some(pair)) => pair.decode_id,
            _ => target.id,
        };
        if lease.session_verdict != SessionVerdict::None {
            let (verdict, at) = (lease.session_verdict, lease.issued_micros);
            self.sessions
                .dispatched(lease.session, verdict, dispatched, at);
        }
    }

    /// The session table's part of a release ([`Lifecycle::on_released`]): a
    /// commit that refused every target leaves the decision's claim no
    /// worker to move to, and the claim goes with it. A release is told in
    /// parts, as a dispatch is: this table, then the stages
    /// ([`Self::released_stages`]). An engine in its host's process tells
    /// the table each side a pair's release ends ahead of any stage, a
    /// decode stated for later among them: no stage hears that side's end,
    /// and the pair's first decision made its claim.
    ///
    /// The table is handed the instant the lease was issued under
    /// ([`Lease::issued_micros`]). The claim that goes is the one that
    /// lease's decision made: a refusal told late leaves a later decision's
    /// claim on the same worker.
    pub(crate) fn released_table(&self, lease: &Lease, reason: ReleaseReason) {
        if reason == ReleaseReason::CommitFailed {
            if let SessionVerdict::Claimed(claimed) = lease.session_verdict {
                let at = lease.issued_micros;
                self.sessions
                    .move_claim(lease.session, claimed, WorkerId(0), at);
            }
        }
    }

    /// The session table's part of a decode call that states a verdict of
    /// its key ([`SessionTable::told_under`]). The call read `key` at `at`,
    /// its own instant, and tells its words under `issued`, the instant of
    /// the lease its pair's first decision issued. A claim it made, of
    /// `claimed`, takes that date. `first` is the worker that decision
    /// claimed for `key`, `WorkerId(0)` where it claimed none. The pair's
    /// decode entries ask here, so they read no table of the engine's.
    #[cfg(feature = "in-process")]
    #[inline]
    pub(crate) fn told_under(
        &self,
        key: Session,
        claimed: Option<WorkerId>,
        at: u64,
        issued: u64,
        first: WorkerId,
    ) {
        self.sessions.told_under(key, claimed, at, issued, first);
    }

    /// The stages' part of a release, behind the table's
    /// ([`Self::released_table`]).
    pub(crate) fn released_stages(&self, lease: &Lease, reason: ReleaseReason) {
        tell_released(self.stages(&self.takers.released), lease, reason);
    }

    /// The stages of `takers`, one of this engine's lists of the stages
    /// that take a word ([`Takers`]), in the list's order.
    #[inline(always)]
    fn stages<'a>(&'a self, takers: &'a [StagePlace]) -> impl Iterator<Item = &'a dyn Lifecycle> {
        takers.iter().map(|place| place.stage(&self.profiles))
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

    /// The stages' part of a dispatch: the stages that take it hear it.
    // In line at its callers: as a call it costs a pair's decision 41
    // instructions, counted with Callgrind.
    #[inline(always)]
    pub(crate) fn dispatched_stages(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        let stages = self.stages(&self.takers.dispatched);
        tell_dispatched(stages, lease, target, side, since_decision_micros);
    }
}

/// A release told to `stages`, the ones that take it, in their order:
/// the engine's, a profile's scorers ahead of its picker.
#[inline(always)]
fn tell_released<'p>(
    stages: impl IntoIterator<Item = &'p dyn Lifecycle>,
    lease: &Lease,
    reason: ReleaseReason,
) {
    for stage in stages {
        stage.on_released(lease, reason);
    }
}

/// A dispatch told to `stages`, the ones that take it, in their order.
#[inline(always)]
fn tell_dispatched<'p>(
    stages: impl IntoIterator<Item = &'p dyn Lifecycle>,
    lease: &Lease,
    target: &Target,
    side: Side,
    since_decision_micros: u32,
) {
    for stage in stages {
        stage.on_dispatched(lease, target, side, since_decision_micros);
    }
}

/// A lease's issue told to `stages`, the ones that take it, in their
/// order.
#[inline(always)]
fn tell_issued<'p>(
    stages: impl IntoIterator<Item = &'p dyn Lifecycle>,
    lease: &Lease,
    decision: &Decision,
) {
    for stage in stages {
        stage.on_decision_issued(lease, decision);
    }
}

/// An end told to `stages`, the ones that take it, in their order.
#[inline(always)]
fn tell_completed<'p>(
    stages: impl IntoIterator<Item = &'p dyn Lifecycle>,
    lease: &Lease,
    outcome: &Outcome,
) {
    for stage in stages {
        stage.on_completed(lease, outcome);
    }
}

/// What an engine kept of the stages that stood by an earlier generation
/// ([`Engine::stood_by`]): the engine's session table, and each of its
/// profiles whose object was made below a generation. It hears the words
/// of a lease issued under that earlier generation.
///
/// It holds nothing of its own. A word told to the stages walks the
/// engine's list of the stages that take it ([`Takers`]), beside the
/// generations their profiles were made under, and tells the ones whose
/// profile stood. So an engine keeps no list for an earlier generation,
/// whatever the number of generations its profiles were made under: a
/// document that gained one profile at each of a thousand applies holds a
/// thousand profiles once.
///
/// A host that tells leases off a file of its own names the engine whole
/// as one too (`Engine::stands`): every profile stood, under no
/// generation's bound, so a profile made under the last generation a
/// document can bear hears as any other.
#[cfg(feature = "in-process")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stood<'a> {
    engine: &'a Engine,
    /// A profile made under a generation below this one stood. With
    /// none, every profile stood: the engine whole.
    below: Option<u64>,
}

#[cfg(feature = "in-process")]
impl<'a> Stood<'a> {
    /// The stages of `takers`, one of the engine's lists of the stages
    /// that take a word ([`Takers`]), whose profile stood: in the list's
    /// order.
    #[inline]
    fn stages(self, takers: &'a [StagePlace]) -> impl Iterator<Item = &'a dyn Lifecycle> {
        let Self { engine, below } = self;
        let stood = move |place: &&StagePlace| {
            below.is_none_or(|below| engine.born[place.profile()] < below)
        };
        let stood = takers.iter().filter(stood);
        stood.map(move |place| place.stage(&engine.profiles))
    }
}

/// What hears a lease's words at the engine that issued it: that engine's
/// session table, and its stages. An engine whole is one: the engine in
/// force hears the leases of its own generation. A host that holds its
/// decisions names it as the engine itself, and a host with a file of
/// leases as a [`Stood`] under no bound (`Engine::stands`). What a later
/// engine kept of the stages that stood by the lease's generation is the
/// other (a [`Stood`] by [`Engine::stood_by`]): the same table, and no
/// stage built since. The rule of a lease's words tells either in its one
/// order ([`lease_events`](super::lease_events)).
#[cfg(feature = "in-process")]
pub(crate) trait Issuing: Copy {
    /// The session table's part of a dispatch
    /// ([`Engine::dispatched_table`]).
    fn dispatched_table(self, lease: &Lease, target: &Target, side: Side);

    /// The stages' part of a dispatch ([`Engine::dispatched_stages`]).
    fn dispatched_stages(self, lease: &Lease, target: &Target, side: Side, since_micros: u32);

    /// The session table's part of a release ([`Engine::released_table`]).
    fn released_table(self, lease: &Lease, reason: ReleaseReason);

    /// The stages' part of a release ([`Engine::released_stages`]).
    fn released_stages(self, lease: &Lease, reason: ReleaseReason);

    /// A lease's issue, told to the stages
    /// ([`Lifecycle::on_decision_issued`]).
    fn decision_issued(self, lease: &Lease, decision: &Decision);

    /// A lease's end, told to the stages ([`Lifecycle::on_completed`]).
    fn completed(self, lease: &Lease, outcome: &Outcome);

    /// A release whole, as [`Lifecycle::on_released`] tells it: the
    /// session table, then the stages.
    #[inline(always)]
    fn released(self, lease: &Lease, reason: ReleaseReason) {
        self.released_table(lease, reason);
        self.released_stages(lease, reason);
    }
}

#[cfg(feature = "in-process")]
impl Issuing for &Engine {
    #[inline(always)]
    fn dispatched_table(self, lease: &Lease, target: &Target, side: Side) {
        Engine::dispatched_table(self, lease, target, side);
    }

    #[inline(always)]
    fn dispatched_stages(self, lease: &Lease, target: &Target, side: Side, since_micros: u32) {
        Engine::dispatched_stages(self, lease, target, side, since_micros);
    }

    #[inline(always)]
    fn released_table(self, lease: &Lease, reason: ReleaseReason) {
        Engine::released_table(self, lease, reason);
    }

    #[inline(always)]
    fn released_stages(self, lease: &Lease, reason: ReleaseReason) {
        Engine::released_stages(self, lease, reason);
    }

    #[inline(always)]
    fn decision_issued(self, lease: &Lease, decision: &Decision) {
        self.on_decision_issued(lease, decision);
    }

    #[inline(always)]
    fn completed(self, lease: &Lease, outcome: &Outcome) {
        self.on_completed(lease, outcome);
    }
}

// The session table is the engine's own, whatever generation a lease is
// of. A word's stages are the ones that take it and whose profile stood.
#[cfg(feature = "in-process")]
impl Issuing for Stood<'_> {
    #[inline]
    fn dispatched_table(self, lease: &Lease, target: &Target, side: Side) {
        self.engine.dispatched_table(lease, target, side);
    }

    fn dispatched_stages(self, lease: &Lease, target: &Target, side: Side, since_micros: u32) {
        let stages = self.stages(&self.engine.takers.dispatched);
        tell_dispatched(stages, lease, target, side, since_micros);
    }

    #[inline]
    fn released_table(self, lease: &Lease, reason: ReleaseReason) {
        self.engine.released_table(lease, reason);
    }

    fn released_stages(self, lease: &Lease, reason: ReleaseReason) {
        tell_released(self.stages(&self.engine.takers.released), lease, reason);
    }

    fn decision_issued(self, lease: &Lease, decision: &Decision) {
        tell_issued(self.stages(&self.engine.takers.issued), lease, decision);
    }

    fn completed(self, lease: &Lease, outcome: &Outcome) {
        tell_completed(self.stages(&self.engine.takers.completed), lease, outcome);
    }
}

impl Engine {
    /// Tells a host's reports, each as the host shows it to decisions, to
    /// every stage that keeps a number of the reports
    /// ([`Lifecycle::keeps_loads`]), and to no other stage. `reports` is
    /// run once for each such stage and hands every report to the call it
    /// is given. An engine with no such stage never runs it: a load
    /// snapshot that tells nobody walks no row. A plugin that watches
    /// signals hears of a snapshot's reports no more than it did before a
    /// stage kept any. The engine over worker lists is the one host that
    /// tells reports.
    #[cfg(feature = "in-process")]
    pub(crate) fn tell_loads(&self, reports: impl Fn(&mut dyn FnMut(&Signal<'_>))) {
        for keeper in &self.keepers {
            let stage = keeper.stage(&self.profiles);
            reports(&mut |report| stage.on_signal(report));
        }
    }

    /// Whether a stage that keeps the reports ([`Lifecycle::keeps_loads`])
    /// keeps a signal's column too ([`Needs::signal_cells`]). Such a stage
    /// reads a told report's named values by its column. A host states the
    /// values of a report it tells under this word: every stage that keeps
    /// reports is told them, or none is. It converts none under an engine
    /// that answers no. It asks once, where it makes its value of the
    /// document.
    #[cfg(feature = "in-process")]
    pub(crate) fn keepers_read_signals(&self) -> bool {
        self.keepers.iter().any(|keeper| match *keeper {
            StagePlace::Scorer { profile, stage } => {
                let scorer = &self.profiles[profile].scorers[stage];
                scorer.scorer.needs().signal_cells
            }
            StagePlace::Picker { profile } => {
                let picker = &self.profiles[profile].picker;
                picker.needs().signal_cells
            }
        })
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
        tell_issued(self.stages(&self.takers.issued), lease, decision);
    }

    #[inline(always)]
    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        self.dispatched_sent(lease, target, side, since_decision_micros, || true);
        self.dispatched_table(lease, target, side);
        self.dispatched_stages(lease, target, side, since_decision_micros);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        tell_completed(self.stages(&self.takers.completed), lease, outcome);
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        // The session table, then the stages. What hears a lease's words
        // spells the same two steps once more, under `in-process`
        // (`Issuing::released`): this entry stands without that feature.
        self.released_table(lease, reason);
        self.released_stages(lease, reason);
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
    /// dispatch and at a release under a sessions rule, the record of
    /// dispatched prefixes at an issue, and a count of sends at a
    /// dispatch. Folded at the build: one read here.
    fn hears(&self) -> EventSet {
        self.with_sends(self.hears)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/framework/tests/mod.rs"]
mod tests;
