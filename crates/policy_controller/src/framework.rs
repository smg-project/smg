//! The framework (design 4.4): profiles of filters, weighted scorers and one
//! picker; plans that sequence profiles; the engine that resolves a request
//! to its binding and decides. A scorer or picker allocates nothing per
//! decision: scratch lives in the per-thread [`Scratch`] the host keeps.

use std::fmt::Debug;

use crate::{
    builtins::{Best, FewestSessions, LeastRequests, Random},
    decision::{Decision, Pair, Pick, Shed, ShedReason, Target, Targets, Trace, TRACE_TOP},
    lease::{Lease, Lifecycle, Membership, Outcome, ReleaseReason, Side, Signal},
    sessions::{Assignment, Claim, SessionRule, SessionTable, SessionVerdict},
    spec::Needs,
    view::{Candidate, Constraints, Phase, PoolFacts, RequestFacts, WorkerId},
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

/// Scores the surviving candidates. `out[i]` is the score of `kept[i]`,
/// normalized to `0.0..=1.0` with higher better; a scorer that sees the whole
/// pool normalizes against it, writes in place and allocates nothing. The
/// returned [`TieBand`] is the band of the written values.
pub trait Scorer: Lifecycle + Debug {
    fn name(&self) -> &'static str;

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand;

    /// What the host must feed for this scorer to mean anything.
    fn needs(&self) -> Needs {
        Needs::default()
    }
}

/// Turns the scored survivors into ordered targets. `scores[i]` belongs to
/// `kept[i]` and `band` is the profile's tie band; the picker pushes its pick
/// first and its alternates after it, and allocates nothing.
pub trait Picker: Lifecycle + Debug {
    fn name(&self) -> &'static str;

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    );
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
    /// view carries. The engine's summary ([`Engine::needs`]) adds the view's
    /// own list of headers.
    pub fn needs(&self) -> Needs {
        self.scorers
            .iter()
            .fold(Needs::default(), |needs, w| needs.or(w.scorer.needs()))
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

/// The buffers of one profile run: the survivors, their combined scores, one
/// scorer's stage and the survivors a soft filter restores when it would
/// leave nothing. A pair decision keeps the prefill's apart from the decode
/// side's, so the prefill's survivors stay for the next prefill target.
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

/// Combined scores of the survivors, written into `scores`; the profile's tie
/// band comes back.
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
    let band = combine(
        profile,
        facts,
        candidates,
        &buf.kept,
        &mut buf.scores,
        &mut buf.stage,
    );
    let mut targets = Targets::new();
    profile.picker.pick(
        facts,
        candidates,
        &buf.kept,
        &buf.scores,
        band,
        &mut targets,
    );
    let trace = trace_of(index, facts, pool, candidates, &buf.kept, &buf.scores);
    if targets.is_empty() {
        return shed(ShedReason::PolicyDeclined, profile.picker.name(), trace);
    }
    Decision::Pick(Pick {
        targets,
        trace,
        session: SessionVerdict::None,
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
}

/// Every phase, in the order [`Engine::phase_needs`] keeps them.
const PHASES: [Phase; 4] = [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode];

/// A phase's slot in [`Engine::phase_needs`]: total, so no read falls back
/// to another phase's needs; [`PHASES`] lists the phases in this order.
const fn slot(phase: Phase) -> usize {
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
/// decode row carrying the pair class it belongs to.
#[derive(Clone, Copy)]
pub struct PairPools<'a> {
    pub prefills: &'a [Candidate],
    pub decodes: &'a [Candidate],
}

impl Engine {
    pub fn new(
        profiles: Vec<Profile>,
        bindings: Vec<Binding>,
        generation: u64,
        needs: Needs,
    ) -> Self {
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
                if let Some(profile) =
                    profile_index(binding, phase).and_then(|i| profiles.get(usize::from(i)))
                {
                    phase_needs[slot(phase)] = phase_needs[slot(phase)]
                        .or(profile.needs())
                        .or(session_needs(binding));
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
                if let Some(profile) =
                    profile_index(binding, Phase::Decode).and_then(|i| profiles.get(usize::from(i)))
                {
                    phase_needs[decode_slot] = phase_needs[decode_slot].or(profile.needs());
                }
            }
            if binding.model == "*" {
                break;
            }
        }
        let sessions = SessionTable::new(shortest_sweep(&bindings));
        Self {
            profiles,
            bindings,
            generation,
            needs,
            phase_needs,
            sessions,
            assignment: AssignmentStages::default(),
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
    /// pinned worker first.
    pub fn decide(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let Some((binding, index, profile)) = self.resolve(facts.model, facts.phase) else {
            return shed(
                ShedReason::PolicyDeclined,
                "binding",
                Trace {
                    seed: facts.seed,
                    generation: pool.generation,
                    ..Trace::default()
                },
            );
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
        let set = Set {
            candidates,
            class: None,
            partnered: None,
            encode: false,
        };
        let buf = &mut scratch.single;
        if let Some(emptied_by) = filter(profile, facts, cx, buf, set) {
            return shed(
                reason_of(emptied_by),
                emptied_by,
                trace_of(index, facts, pool, candidates, &[], &[]),
            );
        }
        let key = facts.session_key.hash;
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
                    let mut decision =
                        self.place(profile, index, rule, facts, pool, candidates, buf);
                    if let Decision::Pick(pick) = &mut decision {
                        pick.session = SessionVerdict::Respill;
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
                session: SessionVerdict::Pinned,
            });
        }
        let mut decision = self.place(profile, index, rule, facts, pool, candidates, buf);
        if let Decision::Pick(pick) = &mut decision {
            pick.session = SessionVerdict::Pinned;
            // No routable pin (a fresh key, or every pin down): the key is
            // claimed at the decision, so concurrent requests agree on the
            // first claim when its worker is among this decision's survivors.
            if let Some(first) = pick.targets.first().map(|t| t.id) {
                let owner = match self.sessions.claim(key, pins, first) {
                    Claim::Made(id) => {
                        pick.session = SessionVerdict::Claimed(id);
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

    /// The stages of a sessions rule over the survivors in `buf.kept`: the
    /// profile's own scorers and picker under `delegate`, else the
    /// assignment's stage over the same survivors.
    #[expect(
        clippy::too_many_arguments,
        reason = "the profile run's inputs, as run_profile takes them"
    )]
    fn place(
        &self,
        profile: &Profile,
        index: u16,
        rule: &SessionRule,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        buf: &mut Buffers,
    ) -> Decision {
        let kept = &buf.kept;
        let (band, picker): (TieBand, &dyn Picker) = match rule.assignment {
            Assignment::Delegate => (
                combine(
                    profile,
                    facts,
                    candidates,
                    kept,
                    &mut buf.scores,
                    &mut buf.stage,
                ),
                profile.picker.as_ref(),
            ),
            Assignment::Random => {
                buf.scores.clear();
                buf.scores.resize(kept.len(), 0.0);
                (TieBand::NORMALIZED, &self.assignment.random)
            }
            Assignment::MinLoad | Assignment::MinGroup => {
                buf.scores.clear();
                buf.scores.resize(kept.len(), 0.0);
                let scorer: &dyn Scorer = if rule.assignment == Assignment::MinLoad {
                    &self.assignment.least_requests
                } else {
                    &self.assignment.fewest_sessions
                };
                (
                    scorer.score(facts, candidates, kept, &mut buf.scores),
                    &self.assignment.best,
                )
            }
        };
        let mut targets = Targets::new();
        picker.pick(facts, candidates, kept, &buf.scores, band, &mut targets);
        let trace = trace_of(index, facts, pool, candidates, kept, &buf.scores);
        if targets.is_empty() {
            return shed(ShedReason::PolicyDeclined, picker.name(), trace);
        }
        Decision::Pick(Pick {
            targets,
            trace,
            session: SessionVerdict::None,
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
    /// leaves the classes unset sheds instead of pairing all with all.
    pub fn decide_pair(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        pools: PairPools<'_>,
        cx: &Constraints<'_>,
        decode_cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let PairPools { prefills, decodes } = pools;
        let bare = Trace {
            seed: facts.seed,
            generation: pool.generation,
            ..Trace::default()
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
        let prefill_facts = RequestFacts {
            phase: Phase::Prefill,
            ..*facts
        };
        // The decode side draws its own sequence, derived from the request's
        // seed: one seed on both sides made the decode picker replay the
        // prefill picker's draws, so the decode chosen followed the prefill
        // chosen (random over four prefills of four decodes each reached four
        // of the sixteen). The decision still replays from the trace's seed.
        let decode_facts = RequestFacts {
            phase: Phase::Decode,
            seed: Draw::new(facts.seed).next_u64(),
            ..*facts
        };
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
                        session: SessionVerdict::None,
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
        if lease.session != SessionVerdict::None {
            self.sessions
                .dispatched(lease.session_key, lease.session, dispatched);
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
    use super::*;
    use crate::{
        builtins::{Best, Healthy, LeastRequests, NotTried},
        sessions::Recovery,
        spec::PolicySpec,
        view::{Health, Identity, WorkerId},
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
            filters: vec![Box::new(Healthy), Box::new(crate::builtins::NotOverloaded)],
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
            PairPools { prefills, decodes },
            &Constraints::default(),
            &Constraints::default(),
            &mut scratch,
        )
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

    #[test]
    fn a_prefill_position_outside_the_slice_costs_the_target_and_not_the_process() {
        let engine = Engine::new(
            vec![strict_profile()],
            vec![pair_binding()],
            1,
            Needs::default(),
        );
        let mut prefills = [candidate(0, 0)];
        prefills[0].position = 5;
        prefills[0].pair_class = 7;
        let mut decodes = [candidate(0, 0)];
        decodes[0].pair_class = 7;
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
            },
            &Constraints::default(),
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("the busier prefill still pairs");
        assert_eq!(pick.targets.first().map(|t| t.position), Some(1));
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
        let target = *decision
            .pick()
            .expect("a pick")
            .targets
            .first()
            .expect("a target");
        let lease = Lease {
            session_key: key,
            session: decision.pick().map_or(SessionVerdict::None, |p| p.session),
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
        // The host removes 102: key 8's failover is its owner, loaded or not.
        engine.on_membership(&Membership::Removed(WorkerId(102)));
        pool.remove(2);
        assert_eq!(
            engine.sessions.pins(Identity::hashed(8).hash),
            Some([WorkerId(100), WorkerId(0)]),
            "the removed owner left the pins, not just the routable set"
        );
        assert_eq!(place_keyed(&engine, &pool, 8, 40).position, 0);
        // The host replaces 100: both keys have no pin left and are placed
        // afresh by the rule's assignment (the fewest requests).
        engine.on_membership(&Membership::Replaced {
            old: WorkerId(100),
            new: WorkerId(200),
        });
        pool.remove(0);
        assert_eq!(engine.sessions.pins(Identity::hashed(7).hash), None);
        assert_eq!(engine.sessions.pins(Identity::hashed(8).hash), None);
        assert_eq!(place_keyed(&engine, &pool, 7, 50).position, 3);
        assert_eq!(place_keyed(&engine, &pool, 8, 50).position, 3);
    }

    #[test]
    fn two_first_decisions_of_a_fresh_key_agree_on_the_first_claim() {
        let engine = session_engine(SESSIONS);
        let key = Identity::hashed(7).hash;
        let mut pool = vec![candidate(0, 1), candidate(1, 4), candidate(2, 2)];
        let first = decide_keyed(&engine, &pool, 7, 10);
        let first_pick = first.pick().expect("a pick");
        assert_eq!(first_pick.targets.first().map(|t| t.position), Some(0));
        let claimed = first_pick.targets.first().expect("a target").id;
        assert_eq!(first_pick.session, SessionVerdict::Claimed(claimed));
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
            session_key: key,
            session: decision.pick().expect("a pick").session,
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
        let key = Identity::hashed(7).hash;
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
            session_key: key,
            session: pick.session,
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
                session_key: Identity::hashed(key).hash,
                session: pick.session,
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
        assert!(engine.sessions.pins(Identity::hashed(2).hash).is_none());
    }

    #[test]
    fn a_key_whose_pins_are_all_down_is_claimed_at_its_decision_too() {
        let engine = session_engine(SESSIONS);
        let key = Identity::hashed(7).hash;
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
        assert_eq!(pick.session, SessionVerdict::None);
        let lease = Lease {
            session_key: Identity::hashed(7).hash,
            session: pick.session,
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

    fn encode_candidate(position: u32, ec_compatible: bool, vision: bool) -> Candidate {
        Candidate {
            role: crate::view::Role::Encode,
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
            media: crate::view::MediaCounts {
                images,
                ..crate::view::MediaCounts::default()
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
