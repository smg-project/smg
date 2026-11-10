//! The framework (design 4.4): profiles of filters, weighted scorers and one
//! picker; plans that sequence profiles; the engine that resolves a request
//! to its binding and decides. A scorer or picker allocates nothing per
//! decision: scratch lives in the per-thread [`Scratch`] the host keeps.

use std::fmt::Debug;

use crate::{
    decision::{Decision, Pair, Pick, Shed, ShedReason, Target, Targets, Trace, TRACE_TOP},
    lease::{Lease, Lifecycle, Membership, Outcome, ReleaseReason, Side, Signal},
    spec::Needs,
    view::{Candidate, Constraints, Phase, PoolFacts, RequestFacts},
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

/// Per-decision scratch the framework owns; kept by the host per thread so a
/// decision allocates nothing after the first.
#[derive(Debug, Default)]
pub struct Scratch {
    single: Buffers,
    decode: Buffers,
    /// The pair classes with a decode the decode profile keeps, sorted.
    classes: Vec<u32>,
}

/// The buffers of one profile run: the survivors, their combined scores and
/// one scorer's stage. A pair decision keeps the prefill's apart from the
/// decode side's, so the prefill's survivors stay for the next prefill target.
#[derive(Debug, Default)]
struct Buffers {
    kept: Vec<u32>,
    scores: Vec<f32>,
    stage: Vec<f32>,
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

/// The filters over `candidates`: the hard ones, then each soft one as long
/// as it leaves something. Returns the name of the filter that emptied the
/// pool, if one did.
fn filter(
    profile: &Profile,
    facts: &RequestFacts<'_>,
    cx: &Constraints<'_>,
    kept: &mut Vec<u32>,
    set: Set<'_>,
) -> Option<&'static str> {
    let Set {
        candidates,
        class,
        partnered,
    } = set;
    kept.clear();
    kept.extend((0..candidates.len() as u32).filter(|&i| {
        let c = &candidates[i as usize];
        class.is_none_or(|k| c.pair_class == k)
            && partnered.is_none_or(|p| p.binary_search(&c.pair_class).is_ok())
            && profile
                .filters
                .iter()
                .filter(|f| !f.soft())
                .all(|f| f.keep(facts, c, cx))
    }));
    if kept.is_empty() {
        return Some(emptied_by(profile, facts, cx, kept, set));
    }
    for soft in profile.filters.iter().filter(|f| f.soft()) {
        if kept
            .iter()
            .any(|&i| soft.keep(facts, &candidates[i as usize], cx))
        {
            kept.retain(|&i| soft.keep(facts, &candidates[i as usize], cx));
        }
    }
    None
}

/// The hard filter that left nothing (design 4.2: a shed names the stage that
/// emptied the pool). The stages run again in profile order over the pool
/// (the pair class, then the partnered test, then the hard filters) and the
/// first after which no row survives is named; the pool is empty at this
/// point, so the second walk costs a pick nothing. A pool with no candidate
/// was emptied by no stage, a class with no decode by the partners.
fn emptied_by(
    profile: &Profile,
    facts: &RequestFacts<'_>,
    cx: &Constraints<'_>,
    kept: &mut Vec<u32>,
    set: Set<'_>,
) -> &'static str {
    let Set {
        candidates,
        class,
        partnered,
    } = set;
    kept.clear();
    kept.extend(
        (0..candidates.len() as u32)
            .filter(|&i| class.is_none_or(|k| candidates[i as usize].pair_class == k)),
    );
    if kept.is_empty() {
        return if class.is_some() { "partners" } else { "none" };
    }
    if let Some(p) = partnered {
        kept.retain(|&i| p.binary_search(&candidates[i as usize].pair_class).is_ok());
        if kept.is_empty() {
            return "partnered";
        }
    }
    for hard in profile.filters.iter().filter(|f| !f.soft()) {
        kept.retain(|&i| hard.keep(facts, &candidates[i as usize], cx));
        if kept.is_empty() {
            return hard.name();
        }
    }
    kept.clear();
    "none"
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
    let mut order: [usize; TRACE_TOP] = [usize::MAX; TRACE_TOP];
    for slot in 0..TRACE_TOP.min(kept.len()) {
        let mut best: Option<usize> = None;
        for (i, &score) in scores.iter().enumerate() {
            if order[..slot].contains(&i) {
                continue;
            }
            if best.is_none_or(|b| score > scores[b]) {
                best = Some(i);
            }
        }
        let Some(i) = best else {
            break;
        };
        order[slot] = i;
        trace.top[slot] = (candidates[kept[i] as usize].id, scores[i]);
        trace.top_len = slot as u8 + 1;
    }
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
    if let Some(emptied_by) = filter(profile, facts, cx, &mut buf.kept, set) {
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
    Decision::Pick(Pick { targets, trace })
}

impl Profile {
    /// Decide over `candidates` with this profile alone (the `single` plan).
    pub fn decide(
        &self,
        index: u16,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        run_profile(
            self,
            index,
            facts,
            pool,
            Set {
                candidates,
                class: None,
                partnered: None,
            },
            cx,
            &mut scratch.single,
        )
    }
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

/// The engine: validated profiles and the ordered bindings over them.
#[derive(Debug)]
pub struct Engine {
    profiles: Vec<Profile>,
    bindings: Vec<Binding>,
    generation: u64,
    needs: Needs,
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
        Self {
            profiles,
            bindings,
            generation,
            needs,
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// What the host must feed before any request exists.
    pub fn needs(&self) -> Needs {
        self.needs
    }

    pub fn profiles(&self) -> &[Profile] {
        &self.profiles
    }

    /// The first binding that matches, as the spec orders them.
    pub fn binding(&self, model: &str, phase: Phase) -> Option<&Binding> {
        self.bindings.iter().find(|b| b.matches(model, phase))
    }

    /// The profile a request resolves to for its phase.
    pub fn profile_for(&self, model: &str, phase: Phase) -> Option<(u16, &Profile)> {
        let binding = self.binding(model, phase)?;
        let index = match (binding.plan, phase) {
            (Plan::PdPair, Phase::Decode) => binding.decode_profile.unwrap_or(binding.profile),
            _ => binding.profile,
        };
        self.profiles.get(usize::from(index)).map(|p| (index, p))
    }

    /// The `single` plan: one profile over `candidates`.
    pub fn decide(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let Some((index, profile)) = self.profile_for(facts.model, facts.phase) else {
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
        profile.decide(index, facts, pool, candidates, cx, scratch)
    }

    /// The `pd-pair` plan: the prefill profile over `prefills`, then the
    /// decode profile over the decodes of the chosen prefill's pair class. A
    /// prefill whose partners leave no decode is passed over for the next
    /// prefill target; the targets list carries the chosen pair first and the
    /// alternate decodes of the same prefill after it.
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
            .filter(|b| b.plan == Plan::PdPair)
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
        let decode_facts = RequestFacts {
            phase: Phase::Decode,
            ..*facts
        };
        // The prefill pool narrows to the prefills with a partner the decode
        // profile's hard filters keep, before the prefill picker truncates to
        // its targets: a pair that exists is found.
        scratch.classes.clear();
        scratch.classes.extend(
            decodes
                .iter()
                .filter(|d| {
                    decode_profile
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
            },
            cx,
            &mut scratch.single,
        );
        let Decision::Pick(prefill_pick) = prefill else {
            // A prefill pool its filters emptied says what the decode pool's
            // own stages say about the partners of the prefills the prefill
            // filters keep (those were dropped by the partnered test alone):
            // a saturated decode fleet is saturation, not a missing endpoint,
            // and a class with no decode at all is "partnered". When no
            // prefill passes its own filters, the prefill side's stage is
            // named instead. A picker's decline passes through.
            if let Decision::Shed(prefill_shed) = &prefill {
                if prefill_shed.reason != ShedReason::PolicyDeclined {
                    scratch.classes.clear();
                    scratch.classes.extend(
                        prefills
                            .iter()
                            .filter(|c| {
                                prefill_profile
                                    .filters
                                    .iter()
                                    .filter(|f| !f.soft())
                                    .all(|f| f.keep(&prefill_facts, c, cx))
                            })
                            .map(|c| c.pair_class),
                    );
                    scratch.classes.sort_unstable();
                    scratch.classes.dedup();
                    if scratch.classes.is_empty() {
                        let stage = emptied_by(
                            prefill_profile,
                            &prefill_facts,
                            cx,
                            &mut scratch.single.kept,
                            Set {
                                candidates: prefills,
                                class: None,
                                partnered: None,
                            },
                        );
                        return shed(reason_of(stage), stage, prefill_shed.trace);
                    }
                    let stage = emptied_by(
                        decode_profile,
                        &decode_facts,
                        decode_cx,
                        &mut scratch.decode.kept,
                        Set {
                            candidates: decodes,
                            class: None,
                            partnered: Some(&scratch.classes),
                        },
                    );
                    if stage != "none" {
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
        view::{Health, WorkerId},
    };

    fn candidate(position: u32, in_flight: u32) -> Candidate {
        Candidate {
            position,
            id: WorkerId(100 + u64::from(position)),
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
        let decodes = [candidate(0, 0)];
        match decide_pair(&engine, &prefills, &decodes) {
            Decision::Shed(shed) => assert_eq!(shed.emptied_by, "partners"),
            other => panic!("a misplaced prefill answered {other:?}"),
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
    fn a_prefill_without_partners_is_passed_over_for_the_next() {
        let engine = Engine::new(
            vec![least_requests_profile()],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
                plan: Plan::PdPair,
                profile: 0,
                decode_profile: Some(0),
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
}
