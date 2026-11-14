//! The framework (design 4.4): profiles of filters, weighted scorers and one
//! picker; plans that sequence profiles; the engine that resolves a request
//! to its binding and decides. A scorer or picker allocates nothing per
//! decision: scratch lives in the per-thread [`Scratch`] the host keeps.

use std::fmt::Debug;

use crate::{
    builtins::{target_of, Best, FewestSessions, LeastRequests, Random},
    decision::{
        Decision, Pair, Pick, Shed, ShedReason, Target, Targets, Trace, MAX_TARGETS, TRACE_TOP,
    },
    lease::{Lease, Lifecycle, Membership, Outcome, ReleaseReason, Side, Signal},
    sessions::{Assignment, Claim, SessionRule, SessionTable, SessionVerdict},
    spec::Needs,
    view::{AffinityFacts, Candidate, Constraints, Phase, PoolFacts, RequestFacts, WorkerId},
    Draw,
};

/// Drops candidates before scoring. A `soft` filter steps aside when it would
/// leave nothing: the engines a retry already tried step aside while another
/// candidate remains, and a pool with nothing else is used whole.
pub trait Filter: Send + Sync + Debug {
    fn name(&self) -> &'static str;

    fn keep(&self, facts: &RequestFacts<'_>, candidate: &Candidate, cx: &Constraints<'_>) -> bool;

    fn soft(&self) -> bool {
        false
    }

    /// What the host must feed for this filter to mean anything. A filter
    /// that reads the affinity column ([`RequestFacts::affinity`]) declares
    /// [`Needs::affinity`]; one that does not may be handed an empty
    /// column, which reads as nothing held.
    fn needs(&self) -> Needs {
        Needs::default()
    }

    /// The per-decision scope this stage reads, if it holds one.
    fn scope(&self) -> Option<&dyn DecisionScope> {
        None
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

    /// What the host must feed for this scorer to mean anything. A scorer
    /// that reads the affinity column ([`RequestFacts::affinity`]) declares
    /// [`Needs::affinity`]; one that does not may be handed an empty
    /// column, which reads as nothing held.
    fn needs(&self) -> Needs {
        Needs::default()
    }

    /// The per-decision scope this stage reads, if it holds one.
    fn scope(&self) -> Option<&dyn DecisionScope> {
        None
    }
}

/// What a picker's narrowing left of the survivors: the first `compared` are
/// the set it compares, and the next `alternates` are survivors it passed
/// over that still follow its pick as alternates, in the order it left them.
/// They follow a pick and never stand in for one: when the picker pushes no
/// target, none is appended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Narrowed {
    /// How many survivors, from the front, the picker compares.
    pub compared: usize,
    /// How many survivors behind those follow the pick as alternates.
    pub alternates: usize,
}

impl Narrowed {
    /// Every survivor is compared.
    pub const fn all(survivors: usize) -> Self {
        Self {
            compared: survivors,
            alternates: 0,
        }
    }
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
    /// it moves no input of a built-in's estimate. The default compares
    /// every survivor.
    fn narrow(
        &self,
        _facts: &RequestFacts<'_>,
        _candidates: &[Candidate],
        kept: &mut [u32],
    ) -> Narrowed {
        Narrowed::all(kept.len())
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

    /// What the host must feed for this picker to mean anything. A picker
    /// that reads the affinity column ([`RequestFacts::affinity`]) declares
    /// [`Needs::affinity`]; one that does not may be handed an empty
    /// column, which reads as nothing held.
    fn needs(&self) -> Needs {
        Needs::default()
    }

    /// The per-decision scope this stage reads, if it holds one.
    fn scope(&self) -> Option<&dyn DecisionScope> {
        None
    }

    /// Whether the picker leaves the placement to its binding's sessions
    /// assignment (the `manual` picker): the engine reads it once per
    /// profile when it is built, and places such a profile's requests
    /// through the assignment's stage without asking the picker.
    fn by_assignment(&self) -> bool {
        false
    }
}

/// Per-decision state a stage set keeps across the stages of one profile
/// (a plugin's `decision` resource over the view): the engine opens it once
/// before the profile's first stage of a decision reads it and closes it
/// after the last, on every path out, so a stage set used by several stages
/// of one profile, or asked row by row, reads one state. A built-in stage
/// holds none; the profile's flag ([`Profile::scoped`]) is read once in the
/// engine, so a profile without one pays nothing per decision.
pub trait DecisionScope: Send + Sync + Debug {
    fn open(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], pool: &PoolFacts<'_>);

    fn close(&self);
}

/// A scorer with its weight in the profile's combination.
#[derive(Debug)]
pub struct Weighted {
    pub scorer: Box<dyn Scorer>,
    pub weight: f32,
}

/// One profile: an ordered set of filters, scorers and exactly one picker.
#[derive(Debug)]
pub struct Profile {
    pub name: String,
    pub filters: Vec<Box<dyn Filter>>,
    pub scorers: Vec<Weighted>,
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
        filters.chain(scorers).fold(self.picker.needs(), Needs::or)
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
        for scope in self.scopes() {
            scope.open(facts, candidates, pool);
        }
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

#[cold]
#[inline(never)]
fn close_scopes(profile: &Profile) {
    for scope in profile.scopes() {
        scope.close();
    }
}

/// Per-decision scratch the framework owns; kept by the host per thread so a
/// decision allocates nothing after the first.
#[derive(Debug, Default)]
pub struct Scratch {
    single: Buffers,
    decode: Buffers,
    /// The pair classes with a decode the decode profile keeps, sorted.
    classes: Vec<u32>,
}

/// The buffers of one profile run: the survivors, the combined scores of the
/// set the picker compares, one scorer's stage and the survivors a soft
/// filter restores when it would leave nothing. A pair decision keeps the
/// prefill's apart from the decode side's, so the prefill's survivors stay
/// for the next prefill target.
#[derive(Debug, Default)]
struct Buffers {
    kept: Vec<u32>,
    scores: Vec<f32>,
    stage: Vec<f32>,
    before_soft: Vec<u32>,
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Combined scores of the rows in `kept`, written into `scores`; the
/// profile's tie band comes back.
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
    let total: f32 = profile.scorers.iter().map(|w| w.weight).sum();
    let norm = if total > 0.0 { 1.0 / total } else { 1.0 };
    let mut band = 0.0f32;
    for weighted in &profile.scorers {
        stage.clear();
        stage.resize(kept.len(), 0.0);
        let stage_band = weighted.scorer.score(facts, candidates, kept, stage);
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

/// The filters over `candidates`, stage by stage in profile order, each one
/// pass over the survivors so far: the pair class, the partnered test, the
/// `epd` plan's compatibility and vision rules, the hard filters, then each
/// soft filter as long as it leaves something.
/// Returns the name of the stage that emptied the pool, if one did (design
/// 4.2: a shed names it): a pool with no candidate was emptied by no stage,
/// a class with no decode by the partners.
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
    } = set;
    let kept = &mut buf.kept;
    kept.clear();
    match class {
        Some(k) => {
            kept.extend(
                (0..candidates.len() as u32).filter(|&i| candidates[i as usize].pair_class == k),
            );
            if kept.is_empty() {
                return Some("partners");
            }
        }
        None => kept.extend(0..candidates.len() as u32),
    }
    if kept.is_empty() {
        return Some("none");
    }
    if let Some(p) = partnered {
        kept.retain(|&i| p.binary_search(&candidates[i as usize].pair_class).is_ok());
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
            return Some("vision");
        }
    }
    for hard in profile.filters.iter().filter(|f| !f.soft()) {
        hard.retain(facts, candidates, cx, kept);
        if kept.is_empty() {
            return Some(hard.name());
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
/// the profile's filters over the whole pool, the survivors left in
/// `buf.kept`. `Some` is the shed, when a filter leaves nobody.
fn single_survivors(
    profile: &Profile,
    index: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Option<Decision> {
    let set = Set {
        candidates,
        class: None,
        partnered: None,
        encode: false,
    };
    let emptied_by = filter(profile, facts, cx, buf, set)?;
    Some(shed(
        reason_of(emptied_by),
        emptied_by,
        trace_of(index, facts, pool, candidates, &[], &[]),
    ))
}

fn trace_of(
    profile: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    kept: &[u32],
    scores: &[f32],
) -> Trace {
    let mut trace = Trace {
        profile,
        survivors: kept.len() as u32,
        seed: facts.seed,
        generation: pool.generation,
        ..Trace::default()
    };
    // One pass: a row enters the top when it beats the lowest row kept, and
    // moves above a kept row only when strictly better, so equal scores keep
    // the scan's order (the earlier row ahead).
    let mut top: [(usize, f32); TRACE_TOP] = [(0, f32::NEG_INFINITY); TRACE_TOP];
    let mut len = 0usize;
    for (i, &score) in scores.iter().enumerate() {
        if len == TRACE_TOP && score <= top[TRACE_TOP - 1].1 {
            continue;
        }
        let mut slot = if len < TRACE_TOP {
            len += 1;
            len - 1
        } else {
            TRACE_TOP - 1
        };
        while slot > 0 && score > top[slot - 1].1 {
            top[slot] = top[slot - 1];
            slot -= 1;
        }
        top[slot] = (i, score);
    }
    for (slot, &(i, score)) in top.iter().enumerate().take(len) {
        trace.top[slot] = (candidates[kept[i] as usize].id, score);
    }
    trace.top_len = len as u8;
    trace
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

/// A pool to decide over: narrowed to one pair class when the decision is a
/// pair's decode side, or to the rows whose class is in the sorted partnered
/// list when it is the prefill side.
#[derive(Clone, Copy)]
struct Set<'a> {
    candidates: &'a [Candidate],
    class: Option<u32>,
    partnered: Option<&'a [u32]>,
    /// The `epd` plan's filters apply: the pool is the encode pool and the
    /// rows' compatibility and vision bits are read.
    encode: bool,
}

/// The `epd` plan's vision rule: a request carrying images takes an encoder
/// with a vision encoder; a request without them takes any.
fn vision_fits(facts: &RequestFacts<'_>, c: &Candidate) -> bool {
    facts.media.images == 0 || c.vision
}

/// One profile over one pool: the filters, the scorers, the picker and the
/// trace. Both plans run their sides through here.
fn run_profile(
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
        return shed(
            reason_of(emptied_by),
            emptied_by,
            trace_of(index, facts, pool, candidates, &[], &[]),
        );
    }
    let mut targets = Targets::new();
    score_and_pick(profile, facts, candidates, buf, &mut targets);
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
    /// Sticky sessions: a keyed request of the `single` plan stays on the
    /// worker its key was placed on (design 4.6).
    pub sessions: Option<SessionRule>,
}

impl Binding {
    fn matches(&self, model: &str, phase: Phase) -> bool {
        let model_ok = match (self.model.strip_prefix('*'), self.model.strip_suffix('*')) {
            _ if self.model == "*" => true,
            (Some(suffix), _) => model.ends_with(suffix),
            (_, Some(prefix)) => model.starts_with(prefix),
            _ => self.model == model,
        };
        model_ok && (self.phases.is_empty() || self.phases.contains(&phase))
    }
}

/// The engine: validated profiles and the ordered bindings over them, and the
/// session table the bindings' sessions rules read.
#[derive(Debug)]
pub struct Engine {
    profiles: Vec<Profile>,
    bindings: Vec<Binding>,
    generation: u64,
    needs: Needs,
    /// The needs of every binding that covers a phase, its profile's stages
    /// and its sessions rule, folded per phase ([`Self::needs_for_phase`]),
    /// indexed by [`slot`].
    phase_needs: [Needs; PHASES.len()],
    sessions: SessionTable,
    assignment: AssignmentStages,
    /// Per profile, what its stages ask of a decision beyond the profile's
    /// own run ([`Marks`]).
    marks: Vec<Marks>,
}

/// What a profile's stages ask of a decision beyond the profile's own run,
/// read once when the engine is built. A decision reads the byte in one
/// compare, so a profile that asks nothing pays for neither mark.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Marks(u8);

impl Marks {
    /// Nothing asked: the profile's run is the decision.
    const NONE: Self = Self(0);
    /// A stage holds a [`DecisionScope`] to open.
    const SCOPED: u8 = 1;
    /// The picker leaves the placement to the binding's sessions assignment
    /// ([`Picker::by_assignment`]).
    const BY_ASSIGNMENT: u8 = 1 << 1;

    fn of(profile: &Profile) -> Self {
        let scoped = if profile.scoped() { Self::SCOPED } else { 0 };
        let by_assignment = if profile.picker.by_assignment() {
            Self::BY_ASSIGNMENT
        } else {
            0
        };
        Self(scoped | by_assignment)
    }

    const fn scoped(self) -> bool {
        self.0 & Self::SCOPED != 0
    }

    const fn by_assignment(self) -> bool {
        self.0 & Self::BY_ASSIGNMENT != 0
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
/// decode row carrying the pair class it belongs to; and beside each pool its
/// affinity column, by the candidate's index, empty for a side nothing reads
/// it on.
#[derive(Clone, Copy, Default)]
pub struct PairPools<'a> {
    pub prefills: &'a [Candidate],
    pub decodes: &'a [Candidate],
    /// The prefills' affinity facts: what the prefill side's stages find in
    /// [`RequestFacts::affinity`].
    pub prefill_affinity: &'a [AffinityFacts],
    /// The decodes' affinity facts, the same for the decode side's stages.
    pub decode_affinity: &'a [AffinityFacts],
}

/// One side of a pair decision.
#[derive(Clone, Copy)]
enum PairSide {
    Prefill,
    Decode,
}

impl<'a> PairPools<'a> {
    /// One side of the pair as its stages read it: the side's pool, and the
    /// request's facts with the side's phase, its draw sequence and its own
    /// pool's affinity column. `None` when that column was not filled for
    /// that pool ([`column_fits`]). The one place a pair's column meets its
    /// pool, so no call site hands one side's column with the other side's
    /// candidates.
    fn side(
        self,
        side: PairSide,
        facts: &RequestFacts<'a>,
    ) -> Option<(&'a [Candidate], RequestFacts<'a>)> {
        let (pool, affinity, phase, seed) = match side {
            PairSide::Prefill => (
                self.prefills,
                self.prefill_affinity,
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
                Phase::Decode,
                Draw::new(facts.seed).next_u64(),
            ),
        };
        debug_assert_indexed(pool);
        column_fits(affinity, pool).then(|| {
            let facts = RequestFacts {
                phase,
                seed,
                affinity,
                ..*facts
            };
            (pool, facts)
        })
    }
}

/// The stage a decision names when it is refused for its affinity column:
/// what a host counts its sheds under.
const AFFINITY_COLUMN: &str = "affinity-column";

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

impl Engine {
    /// The engine of `profiles` under `bindings`. `needs` is what no stage
    /// declares: a sessions rule's counts and the view's list of headers.
    /// Every profile's stages are folded in here, so the summary
    /// ([`Self::needs`]) holds a stage's needs however the engine was built.
    pub fn new(
        profiles: Vec<Profile>,
        mut bindings: Vec<Binding>,
        generation: u64,
        needs: Needs,
    ) -> Self {
        // Each profile's needs, asked of its stages once: the summary and
        // the phase answers below fold them.
        let profile_needs: Vec<Needs> = profiles.iter().map(Profile::needs).collect();
        let needs = profile_needs.iter().fold(needs, |all, &own| all.or(own));
        // Per phase, the bindings that cover it in the spec's order; a
        // binding behind a catch-all model glob decides no request of the
        // phase and adds nothing (the gateway's own document binds every
        // model's pair phases before its catch-all for the regular ones).
        let mut phase_needs = [Needs::default(); PHASES.len()];
        for phase in PHASES {
            for binding in bindings
                .iter()
                .filter(|b| b.phases.is_empty() || b.phases.contains(&phase))
            {
                if let Some(&own) =
                    profile_index(binding, phase).and_then(|i| profile_needs.get(usize::from(i)))
                {
                    phase_needs[slot(phase)] =
                        phase_needs[slot(phase)].or(own).or(session_needs(binding));
                }
                if binding.model == "*" {
                    break;
                }
            }
        }
        // A pair's decode side is decided by the binding its prefill lookup
        // resolves to ([`Self::decide_pair`]), not by the first binding that
        // covers the decode phase: the decode slot also carries the decode
        // profile of every pair binding a prefill lookup can reach.
        let decode_slot = slot(Phase::Decode);
        for binding in bindings
            .iter()
            .filter(|b| b.phases.is_empty() || b.phases.contains(&Phase::Prefill))
        {
            if matches!(binding.plan, Plan::PdPair | Plan::Epd { .. }) {
                if let Some(&own) = profile_index(binding, Phase::Decode)
                    .and_then(|i| profile_needs.get(usize::from(i)))
                {
                    phase_needs[decode_slot] = phase_needs[decode_slot].or(own);
                }
            }
            if binding.model == "*" {
                break;
            }
        }
        let sessions = SessionTable::new(shortest_sweep(&bindings));
        let marks: Vec<Marks> = profiles.iter().map(Marks::of).collect();
        // A picker that leaves the placement to its rule has no stages to
        // delegate to: the rule's assignment is written here as the one
        // that places, once, so no decision asks.
        for binding in &mut bindings {
            let by_assignment = binding.plan == Plan::Single
                && marks
                    .get(usize::from(binding.profile))
                    .is_some_and(|marks| marks.by_assignment());
            if let Some(rule) = &mut binding.sessions {
                rule.assignment = placed_by(rule.assignment, by_assignment);
            }
        }
        Self {
            profiles,
            bindings,
            generation,
            needs,
            phase_needs,
            sessions,
            assignment: AssignmentStages::default(),
            marks,
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The session table: the pins of every key a sessions rule placed.
    pub fn sessions(&self) -> &SessionTable {
        &self.sessions
    }

    /// What the host must feed before any request exists: every profile's
    /// stages and the view's own list of headers.
    pub fn needs(&self) -> Needs {
        self.needs
    }

    /// What a decision of `phase` for `model` reads: the needs of the stages
    /// of the profile its binding resolves to and of the binding's sessions
    /// rule, nothing when no binding matches. A decode decision may also be the decode side of a pair,
    /// whose profile the prefill lookup names ([`Self::decide_pair`]); the
    /// decode answer carries that profile's needs too. The host reads it per
    /// worker, at registration, to choose the worker's stream.
    pub fn needs_for(&self, model: &str, phase: Phase) -> Needs {
        let own = self
            .resolve(model, phase)
            .map(|(binding, _, profile)| profile.needs().or(session_needs(binding)))
            .unwrap_or_default();
        match phase {
            Phase::Decode => own.or(self.pair_decode_needs(model)),
            _ => own,
        }
    }

    /// The needs of the decode profile of the pair binding `model`'s prefill
    /// lookup resolves to, nothing when that binding is not a pair.
    fn pair_decode_needs(&self, model: &str) -> Needs {
        self.binding(model, Phase::Prefill)
            .filter(|b| matches!(b.plan, Plan::PdPair | Plan::Epd { .. }))
            .and_then(|b| profile_index(b, Phase::Decode))
            .and_then(|i| self.profiles.get(usize::from(i)))
            .map(Profile::needs)
            .unwrap_or_default()
    }

    /// What some decision of `phase` may read, over every binding that
    /// covers the phase: the host's answer to "does any regular placement
    /// read the request text" and its kin, read at apply.
    pub fn needs_for_phase(&self, phase: Phase) -> Needs {
        self.phase_needs[slot(phase)]
    }

    pub fn profiles(&self) -> &[Profile] {
        &self.profiles
    }

    /// The first binding that matches, as the spec orders them.
    pub fn binding(&self, model: &str, phase: Phase) -> Option<&Binding> {
        self.bindings.iter().find(|b| b.matches(model, phase))
    }

    /// The profile a request resolves to for its phase; `None` when no
    /// binding matches or the binding's plan has no side for the phase.
    pub fn profile_for(&self, model: &str, phase: Phase) -> Option<(u16, &Profile)> {
        self.resolve(model, phase)
            .map(|(_, index, profile)| (index, profile))
    }

    /// The binding a request resolves to, with its profile for the phase: one
    /// walk of the bindings per decision; `None` when no binding matches or
    /// the binding's plan has no side for the phase.
    fn resolve(&self, model: &str, phase: Phase) -> Option<(&Binding, u16, &Profile)> {
        let binding = self.binding(model, phase)?;
        let index = profile_index(binding, phase)?;
        let profile = self.profiles.get(usize::from(index))?;
        Some((binding, index, profile))
    }

    /// One profile over `candidates`: the `single` plan, or one encode item
    /// of the `epd` plan (the request's phase is `Encode` and the plan's
    /// compatibility and vision rules filter the pool before the profile);
    /// under a `single` binding's sessions rule a keyed request goes to its
    /// pinned worker first, and a profile whose picker leaves the placement
    /// to that rule is placed by its assignment. The request's affinity
    /// column is empty or holds one entry a candidate; any other is
    /// refused, as a decline under the stage name `affinity-column`.
    pub fn decide(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let bare = Trace {
            seed: facts.seed,
            generation: pool.generation,
            ..Trace::default()
        };
        debug_assert_indexed(candidates);
        if !column_fits(facts.affinity, candidates) {
            return shed(ShedReason::PolicyDeclined, AFFINITY_COLUMN, bare);
        }
        let Some((binding, index, profile)) = self.resolve(facts.model, facts.phase) else {
            return shed(ShedReason::PolicyDeclined, "binding", bare);
        };
        let marks = self.marks[usize::from(index)];
        let _opened = if marks == Marks::NONE {
            None
        } else {
            let opened = marks
                .scoped()
                .then(|| profile.open(facts, candidates, pool));
            // A picker that leaves the placement to its rule is not asked:
            // the rule's pin takes a keyed request first, below, and the
            // assignment places every other.
            let pinned_first = binding.sessions.is_some() && !facts.session_key.is_none();
            if marks.by_assignment() && binding.plan == Plan::Single && !pinned_first {
                // With no rule the assignment is `delegate`, as placed for
                // such a picker.
                let unruled = placed_by(Assignment::Delegate, true);
                let assignment = binding.sessions.map_or(unruled, |rule| rule.assignment);
                return self.assign(
                    profile, index, assignment, facts, pool, candidates, cx, scratch,
                );
            }
            opened
        };
        if let Some(rule) = binding.sessions {
            if binding.plan == Plan::Single && !facts.session_key.is_none() {
                return self
                    .decide_session(profile, index, &rule, facts, pool, candidates, cx, scratch);
            }
        }
        let encode = matches!(
            (binding.plan, facts.phase),
            (Plan::Epd { .. }, Phase::Encode)
        );
        run_profile(
            profile,
            index,
            facts,
            pool,
            Set {
                candidates,
                class: None,
                partnered: None,
                encode,
            },
            cx,
            &mut scratch.single,
        )
    }

    /// A keyed decision under a sessions rule (design 4.6 and the host's
    /// keyed path before it): the profile's filters narrow the pool; the
    /// key's preferred pin, or its failover when the preferred one is
    /// unroutable, is the decision while it is under the in-flight cap; a pin
    /// over the cap is reassigned by the stages over the candidates under the
    /// cap (the pin stays when every candidate is over it); a key with no
    /// routable pin is placed by the stages. The stages are the profile's own
    /// under `delegate`, else the assignment's. The pin is written by the
    /// dispatch (`on_dispatched`), never here.
    #[expect(
        clippy::too_many_arguments,
        reason = "the profile run's inputs, as run_profile takes them"
    )]
    fn decide_session(
        &self,
        profile: &Profile,
        index: u16,
        rule: &SessionRule,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let buf = &mut scratch.single;
        if let Some(shed) = single_survivors(profile, index, facts, pool, candidates, cx, buf) {
            return shed;
        }
        let key = facts.session();
        let pins = self.sessions.lookup(key, facts.now_micros, rule);
        // The first pin the filters kept, by the pins' order.
        let owner = pins
            .iter()
            .filter(|p| **p != WorkerId(0))
            .find_map(|p| buf.kept.iter().find(|&&i| candidates[i as usize].id == *p))
            .copied();
        if let Some(i) = owner {
            let c = &candidates[i as usize];
            let under_cap = rule.inflight_cap == 0 || c.session_in_flight < rule.inflight_cap;
            if !under_cap {
                // Over the cap: the candidates under it decide; with none
                // under it the key stays on its pin.
                if buf
                    .kept
                    .iter()
                    .any(|&k| candidates[k as usize].session_in_flight < rule.inflight_cap)
                {
                    buf.kept
                        .retain(|&k| candidates[k as usize].session_in_flight < rule.inflight_cap);
                    let mut decision = self.place(
                        profile,
                        index,
                        rule.assignment,
                        facts,
                        pool,
                        candidates,
                        buf,
                    );
                    if let Decision::Pick(pick) = &mut decision {
                        pick.session_verdict = SessionVerdict::Respill;
                    }
                    return decision;
                }
            }
            let mut targets = Targets::new();
            targets.push(Target {
                position: c.position,
                id: c.id,
                dp_rank: c.dp_rank,
                pair: None,
            });
            let mut trace = Trace {
                profile: index,
                survivors: buf.kept.len() as u32,
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
        let mut decision = self.place(
            profile,
            index,
            rule.assignment,
            facts,
            pool,
            candidates,
            buf,
        );
        if let Decision::Pick(pick) = &mut decision {
            pick.session_verdict = SessionVerdict::Pinned;
            // No routable pin (a fresh key, or every pin down): the key is
            // claimed at the decision, so concurrent requests agree on the
            // first claim when its worker is among this decision's survivors.
            if let Some(first) = pick.targets.first().map(|t| t.id) {
                let owner = match self.sessions.claim(key, pins, first) {
                    Claim::Made(id) => {
                        pick.session_verdict = SessionVerdict::Claimed(id);
                        id
                    }
                    Claim::Standing(id) => id,
                };
                if owner != first {
                    if let Some(&i) = buf
                        .kept
                        .iter()
                        .find(|&&i| candidates[i as usize].id == owner)
                    {
                        let c = &candidates[i as usize];
                        pick.targets.lead(Target {
                            position: c.position,
                            id: c.id,
                            dp_rank: c.dp_rank,
                            pair: None,
                        });
                    }
                }
            }
        }
        decision
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
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let buf = &mut scratch.single;
        if let Some(shed) = single_survivors(profile, index, facts, pool, candidates, cx, buf) {
            return shed;
        }
        self.place(profile, index, assignment, facts, pool, candidates, buf)
    }

    /// The stages of an assignment over the survivors in `buf.kept`: the
    /// profile's own scorers and picker under `delegate`, else the
    /// assignment's stage over the same survivors. The assignment is the
    /// one that places ([`placed_by`]), as the engine's build wrote it.
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
        candidates: &[Candidate],
        buf: &mut Buffers,
    ) -> Decision {
        let mut targets = Targets::new();
        let picker: &dyn Picker = match assignment {
            // The profile's own stages, through the profile run's one home.
            Assignment::Delegate => {
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

    /// The `pd-pair` plan, and the pair half of `epd`: the prefill profile
    /// over `prefills`, then the decode profile over the decodes of the
    /// chosen prefill's pair class. A
    /// prefill whose partners leave no decode is passed over for the next
    /// prefill target; the targets list carries the chosen pair first and the
    /// alternate decodes of the same prefill after it. Pair class `0` is
    /// outside any pairing on either side: such a decode partners nobody and
    /// such a prefill is dropped at the `partnered` stage, so a host that
    /// leaves the classes unset sheds instead of pairing all with all. Each
    /// side's affinity column rides its pool ([`PairPools`]); the request
    /// carries none here, and a non-empty one is refused under the stage
    /// name `affinity-column`.
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
        // A pair's columns ride its two pools. One on the request names
        // neither pool: refused, where it would be dropped in silence and
        // both sides scored as if nothing were held.
        if !facts.affinity.is_empty() {
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
        let Some(binding) = self
            .binding(facts.model, Phase::Prefill)
            .filter(|b| matches!(b.plan, Plan::PdPair | Plan::Epd { .. }))
        else {
            return shed(ShedReason::PolicyDeclined, "binding", bare);
        };
        let decode_index = binding.decode_profile.unwrap_or(binding.profile);
        let (Some(prefill_profile), Some(decode_profile)) = (
            self.profiles.get(usize::from(binding.profile)),
            self.profiles.get(usize::from(decode_index)),
        ) else {
            return shed(ShedReason::PolicyDeclined, "binding", bare);
        };
        // One scope per side: the WIT view has one candidate list, so the
        // prefill stages read the prefill pool and the decode stages the
        // decode pool, through every decode run and the partner passes.
        let _prefill_opened = self.marks[usize::from(binding.profile)]
            .scoped()
            .then(|| prefill_profile.open(&prefill_facts, prefills, pool));
        let _decode_opened = self.marks[usize::from(decode_index)]
            .scoped()
            .then(|| decode_profile.open(&decode_facts, decodes, pool));
        // The prefill pool narrows to the prefills with a partner the decode
        // profile's hard filters keep, before the prefill picker truncates to
        // its targets: a pair that exists is found. A decode of class 0 is
        // outside any pairing and partners nobody.
        scratch.classes.clear();
        scratch.classes.extend(
            decodes
                .iter()
                .filter(|d| {
                    d.pair_class != 0
                        && decode_profile
                            .filters
                            .iter()
                            .filter(|f| !f.soft())
                            .all(|f| f.keep(&decode_facts, d, decode_cx))
                })
                .map(|d| d.pair_class),
        );
        scratch.classes.sort_unstable();
        scratch.classes.dedup();
        let prefill = run_profile(
            prefill_profile,
            binding.profile,
            &prefill_facts,
            pool,
            Set {
                candidates: prefills,
                class: None,
                partnered: Some(&scratch.classes),
                encode: false,
            },
            cx,
            &mut scratch.single,
        );
        let Decision::Pick(prefill_pick) = prefill else {
            // A prefill pool its filters emptied says what the decode pool's
            // own stages say about the partners of the prefills the prefill
            // filters keep (those were dropped by the partnered test alone):
            // a saturated decode fleet is saturation, not a missing endpoint,
            // and a class with no decode at all is "partnered", as is a
            // routable prefill of class 0. When no prefill passes its own
            // filters, the prefill side's stage is named instead. A picker's
            // decline passes through.
            if let Decision::Shed(prefill_shed) = &prefill {
                if prefill_shed.reason != ShedReason::PolicyDeclined {
                    let mut routable = 0usize;
                    scratch.classes.clear();
                    for c in prefills.iter().filter(|c| {
                        prefill_profile
                            .filters
                            .iter()
                            .filter(|f| !f.soft())
                            .all(|f| f.keep(&prefill_facts, c, cx))
                    }) {
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
                            cx,
                            &mut scratch.single,
                            Set {
                                candidates: prefills,
                                class: None,
                                partnered: None,
                                encode: false,
                            },
                        )
                        .unwrap_or("none");
                        return shed(reason_of(stage), stage, prefill_shed.trace);
                    }
                    if scratch.classes.is_empty() {
                        return shed(ShedReason::NoEndpoints, "partnered", prefill_shed.trace);
                    }
                    // An empty decode pool is that class for every prefill:
                    // its "none" falls through to the prefill's shed.
                    if let Some(stage) = filter(
                        decode_profile,
                        &decode_facts,
                        decode_cx,
                        &mut scratch.decode,
                        Set {
                            candidates: decodes,
                            class: None,
                            partnered: Some(&scratch.classes),
                            encode: false,
                        },
                    )
                    .filter(|&stage| stage != "none")
                    {
                        return shed(reason_of(stage), stage, prefill_shed.trace);
                    }
                }
            }
            return prefill;
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
            // alone; the class is a per-candidate fact, so no host callback
            // is needed to narrow the set.
            let decision = run_profile(
                decode_profile,
                decode_index,
                &decode_facts,
                pool,
                Set {
                    candidates: decodes,
                    class: Some(class),
                    partnered: None,
                    encode: false,
                },
                decode_cx,
                &mut scratch.decode,
            );
            match decision {
                Decision::Pick(decode_pick) => {
                    let mut targets = Targets::new();
                    for decode in decode_pick.targets.iter() {
                        targets.push(Target {
                            position: prefill_target.position,
                            id: prefill_target.id,
                            dp_rank: prefill_target.dp_rank,
                            pair: Some(Pair {
                                decode_position: decode.position,
                                decode_id: decode.id,
                                decode_dp_rank: decode.dp_rank,
                            }),
                        });
                    }
                    return Decision::Pick(Pick {
                        targets,
                        trace: prefill_pick.trace,
                        session_verdict: SessionVerdict::None,
                    });
                }
                Decision::Shed(shed) => last_shed = Some(shed),
                Decision::Queue(queue) => return Decision::Queue(queue),
            }
        }
        match last_shed {
            Some(last) => Decision::Shed(last),
            None => shed(ShedReason::NoEndpoints, "partners", prefill_pick.trace),
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

    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        // The session pin follows the dispatched worker of the side: the
        // decode of a pair target on the decode side, the target itself
        // otherwise.
        let dispatched = match (side, target.pair) {
            (Side::Decode, Some(pair)) => pair.decode_id,
            _ => target.id,
        };
        if lease.session_verdict != SessionVerdict::None {
            self.sessions
                .dispatched(lease.session, lease.session_verdict, dispatched);
        }
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
            Membership::Removed(worker) | Membership::Replaced { old: worker, .. } => {
                self.sessions.forget_worker(*worker);
            }
            _ => {}
        }
        for profile in &self.profiles {
            for w in &profile.scorers {
                w.scorer.on_membership(change);
            }
            profile.picker.on_membership(change);
        }
    }

    fn on_signal(&self, signal: &Signal) {
        for profile in &self.profiles {
            for w in &profile.scorers {
                w.scorer.on_signal(signal);
            }
            profile.picker.on_signal(signal);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{
        builtins::{
            Best, CountFirst, Healthy, LeastRequests, Manual, NotOverloaded, NotTried, PowerOfTwo,
            PowerOfTwoCountFirst,
        },
        sessions::Recovery,
        spec::PolicySpec,
        view::{Health, Identity, MediaCounts, Role, Session, WorkerId},
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
            scorers: vec![Weighted {
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
                sessions: None,
            }],
            1,
            Needs::default(),
        )
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
        assert_eq!(pick.trace.survivors, 3);
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
                sessions: None,
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
            sessions: None,
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

    /// One profile that keeps the healthy rows and leads with the deepest
    /// overlap, under `binding`.
    fn overlap_engine(binding: Binding) -> Engine {
        let profile = Profile {
            name: "overlap".into(),
            filters: vec![Box::new(Healthy)],
            scorers: vec![Weighted {
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
        let engine = strict_engine();
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
        let engine = strict_engine();
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
        let engine = Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
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
                    sessions: None,
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
        assert_eq!(pick.trace.survivors, 1);
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
                sessions: None,
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
            }
            other => panic!("an unbound model answered {other:?}"),
        }
    }

    #[test]
    fn the_pair_plan_pairs_the_prefill_with_a_decode_of_its_class() {
        let engine = Engine::new(
            vec![Profile {
                name: "least".into(),
                filters: vec![Box::new(Healthy)],
                scorers: vec![Weighted {
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
                sessions: None,
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
                sessions: None,
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
                scorers: vec![Weighted {
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
                sessions,
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
            assert_eq!(pick.trace.survivors, 5, "the filters' count");
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
        struct Claims(Narrowed);
        impl Lifecycle for Claims {}
        impl Picker for Claims {
            fn name(&self) -> &'static str {
                "claims"
            }
            fn narrow(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &mut [u32]) -> Narrowed {
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
        let all = Narrowed {
            compared: usize::MAX,
            alternates: usize::MAX,
        };
        let one = Narrowed {
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
            fn narrow(&self, _: &RequestFacts<'_>, _: &[Candidate], kept: &mut [u32]) -> Narrowed {
                Narrowed {
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
                        scorers: vec![Weighted {
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
                    sessions: None,
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
                    scorers: vec![Weighted {
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
                sessions: None,
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
        assert_eq!(pick.trace.survivors, 4);
    }

    #[test]
    fn bindings_match_globs_and_phases_in_order() {
        let b = Binding {
            model: "pd/*".into(),
            phases: vec![Phase::Prefill],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            sessions: None,
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
            sessions: None,
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
                Weighted {
                    scorer: Box::new(Wide),
                    weight: 3.0,
                },
                Weighted {
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

    const SESSIONS: SessionRule = SessionRule {
        assignment: Assignment::Delegate,
        inflight_cap: 2,
        ttl_micros: None,
        sweep_micros: 0,
        recovery: Recovery::Latest,
    };

    /// The least-requests profile under a sessions rule.
    fn session_engine(rule: SessionRule) -> Engine {
        Engine::new(
            vec![least_requests_profile()],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::Single,
                profile: 0,
                decode_profile: None,
                sessions: Some(rule),
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

    /// Two pins, the owner leading and saturated on the key, the failover
    /// under the cap: the respill goes to the failover and the pin follows
    /// it, so the failover leads once the owner is under the cap again.
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
        assert_eq!(place_keyed(&engine, &pool, 7, 40).position, 2);
        pool[1].session_in_flight = 0;
        assert_eq!(
            place_keyed(&engine, &pool, 7, 50).position,
            2,
            "the respill moved the pin to the failover"
        );
    }

    /// With every candidate over the cap the key stays on its owner; one
    /// candidate under the cap takes the respill.
    #[test]
    fn with_every_candidate_over_the_cap_the_key_stays_on_its_owner() {
        let engine = session_engine(SESSIONS);
        let mut pool = vec![candidate(0, 0), candidate(1, 3), candidate(2, 3)];
        assert_eq!(place_keyed(&engine, &pool, 7, 10).position, 0);
        for c in &mut pool {
            c.session_in_flight = 2;
        }
        assert_eq!(place_keyed(&engine, &pool, 7, 20).position, 0);
        pool[2].session_in_flight = 1;
        assert_eq!(place_keyed(&engine, &pool, 7, 30).position, 2);
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

    #[test]
    fn a_removed_workers_keys_fall_to_their_failover_or_are_placed_afresh() {
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
        // The host removes 102: key 8's failover is its owner, loaded or not.
        engine.on_membership(&Membership::Removed(WorkerId(102)));
        pool.remove(2);
        renumber(&mut pool);
        assert_eq!(
            engine.sessions.pins(session(8)),
            Some([WorkerId(100), WorkerId(0)]),
            "the removed owner left the pins, not just the routable set"
        );
        assert_eq!(place_keyed(&engine, &pool, 8, 40).id, WorkerId(100));
        // The host replaces 100: both keys have no pin left and are placed
        // afresh by the rule's assignment (the fewest requests).
        engine.on_membership(&Membership::Replaced {
            old: WorkerId(100),
            new: WorkerId(200),
        });
        pool.remove(0);
        renumber(&mut pool);
        assert_eq!(engine.sessions.pins(session(7)), None);
        assert_eq!(engine.sessions.pins(session(8)), None);
        assert_eq!(place_keyed(&engine, &pool, 7, 50).id, WorkerId(103));
        assert_eq!(place_keyed(&engine, &pool, 8, 50).id, WorkerId(103));
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
            sessions: Some(rule),
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
                sessions: None,
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
            session_key: Identity::hashed(crate::view::fingerprint(b"conversation-7")),
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
                sessions,
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

    /// On a side of a pair no sessions rule runs, so nothing places a
    /// `manual` profile's requests and the picker declines by its name. The
    /// validator refuses such a document before an engine is built.
    #[test]
    fn the_manual_picker_declines_on_a_side_of_a_pair() {
        let engine = Engine::new(
            vec![manual_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let mut prefill = candidate(0, 0);
        prefill.pair_class = 1;
        let mut decode = candidate(0, 0);
        decode.pair_class = 1;
        match decide_pair(&engine, &[prefill], &[decode]) {
            Decision::Shed(shed) => {
                assert_eq!(shed.reason, ShedReason::PolicyDeclined);
                assert_eq!(shed.emptied_by, "manual");
            }
            other => panic!("a manual side of a pair answered {other:?}"),
        }
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
                sessions: None,
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
                sessions: None,
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
        assert_eq!(pick.trace.survivors, 2, "the incompatible encoder is out");
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
}
