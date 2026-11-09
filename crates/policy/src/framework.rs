//! The framework: profiles of filters, weighted scorers and one picker; plans
//! that sequence profiles; the engine that resolves a request to its binding
//! and decides.

use std::fmt::Debug;

use crate::{
    decision::{Decision, DecodeTarget, Pick, Shed, ShedReason, Target, Targets, Trace, TRACE_TOP},
    lease::{Lease, Lifecycle, Membership, Outcome, Side, Signal},
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

/// Scores the surviving candidates. `out[i]` is the score of `kept[i]`,
/// normalized to `0.0..=1.0` with higher better; a scorer that sees the whole
/// pool normalizes against it.
pub trait Scorer: Lifecycle + Debug {
    fn name(&self) -> &'static str;

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    );

    /// What the host must feed for this scorer to mean anything.
    fn needs(&self) -> Needs {
        Needs::default()
    }
}

/// Turns the scored survivors into ordered targets. `scores[i]` belongs to
/// `kept[i]`; the picker pushes its pick first and its alternates after it.
pub trait Picker: Lifecycle + Debug {
    fn name(&self) -> &'static str;

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
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
    kept: Vec<u32>,
    scores: Vec<f32>,
    stage: Vec<f32>,
    decode_kept: Vec<u32>,
    decode_scores: Vec<f32>,
    decode_stage: Vec<f32>,
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Combined scores of the survivors, written into `scores`.
fn combine(
    profile: &Profile,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    kept: &[u32],
    scores: &mut Vec<f32>,
    stage: &mut Vec<f32>,
) {
    scores.clear();
    scores.resize(kept.len(), 0.0);
    if profile.scorers.is_empty() {
        return;
    }
    let total: f32 = profile.scorers.iter().map(|w| w.weight).sum();
    let norm = if total > 0.0 { 1.0 / total } else { 1.0 };
    for weighted in &profile.scorers {
        stage.clear();
        stage.resize(kept.len(), 0.0);
        weighted.scorer.score(facts, candidates, kept, stage);
        for (sum, s) in scores.iter_mut().zip(stage.iter()) {
            *sum += weighted.weight * norm * s.clamp(0.0, 1.0);
        }
    }
}

/// The filters over `candidates`: the hard ones, then each soft one as long
/// as it leaves something. Returns the name of the filter that emptied the
/// pool, if one did.
fn filter(
    profile: &Profile,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    cx: &Constraints<'_>,
    kept: &mut Vec<u32>,
) -> Option<&'static str> {
    kept.clear();
    kept.extend((0..candidates.len() as u32).filter(|&i| {
        profile
            .filters
            .iter()
            .filter(|f| !f.soft())
            .all(|f| f.keep(facts, &candidates[i as usize], cx))
    }));
    if kept.is_empty() {
        return Some(
            profile
                .filters
                .iter()
                .find(|f| !f.soft())
                .map_or("none", |f| f.name()),
        );
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

fn trace_of(
    profile: u16,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts,
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
    let mut taken = [false; TRACE_TOP];
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
        taken[slot] = true;
        trace.top[slot] = (candidates[kept[i] as usize].id, scores[i]);
        trace.top_len = slot as u8 + 1;
    }
    trace
}

impl Profile {
    /// Decide over `candidates` with this profile alone (the `single` plan).
    pub fn decide(
        &self,
        index: u16,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        if let Some(emptied_by) = filter(self, facts, candidates, cx, &mut scratch.kept) {
            return Decision::Shed(Shed {
                reason: if candidates.is_empty() {
                    ShedReason::NoEndpoints
                } else if candidates.iter().all(|c| c.health.overloaded) {
                    ShedReason::Saturated
                } else {
                    ShedReason::NoEndpoints
                },
                emptied_by,
                retry_after_secs: 1,
                wait_allowed: false,
                trace: trace_of(index, facts, pool, candidates, &[], &[]),
            });
        }
        combine(
            self,
            facts,
            candidates,
            &scratch.kept,
            &mut scratch.scores,
            &mut scratch.stage,
        );
        let mut targets = Targets::new();
        self.picker.pick(
            facts,
            candidates,
            &scratch.kept,
            &scratch.scores,
            &mut targets,
        );
        let trace = trace_of(
            index,
            facts,
            pool,
            candidates,
            &scratch.kept,
            &scratch.scores,
        );
        if targets.is_empty() {
            return Decision::Shed(Shed {
                reason: ShedReason::PolicyDeclined,
                emptied_by: self.picker.name(),
                retry_after_secs: 1,
                wait_allowed: false,
                trace,
            });
        }
        Decision::Pick(Pick { targets, trace })
    }
}

/// A binding: which profile (or plan of profiles) a request resolves to.
#[derive(Clone, Debug)]
pub struct Binding {
    /// A model glob: `*`, `prefix*`, `*suffix` or an exact id.
    pub model: String,
    /// The phases the binding covers; empty covers every phase.
    pub phases: Vec<Phase>,
    /// The profile for a single decision, and the prefill phase of a pair.
    pub profile: u16,
    /// The decode profile of the `pd-pair` plan, when the binding has one.
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

/// The two pools of a pair decision: the open prefills and the decodes
/// (each decode row carrying the partner group it belongs to).
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
        let index = match phase {
            Phase::Decode => binding.decode_profile.unwrap_or(binding.profile),
            _ => binding.profile,
        };
        self.profiles.get(usize::from(index)).map(|p| (index, p))
    }

    /// The `single` plan: one profile over `candidates`.
    pub fn decide(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let Some((index, profile)) = self.profile_for(facts.model, facts.phase) else {
            return Decision::Shed(Shed {
                reason: ShedReason::PolicyDeclined,
                emptied_by: "binding",
                retry_after_secs: 1,
                wait_allowed: false,
                trace: Trace {
                    seed: facts.seed,
                    generation: pool.generation,
                    ..Trace::default()
                },
            });
        };
        profile.decide(index, facts, pool, candidates, cx, scratch)
    }

    /// The `pd-pair` plan: the prefill profile over `prefills`, then the
    /// decode profile over the decodes of the chosen prefill's partner group.
    /// A prefill whose partners leave no decode is passed over for the next
    /// prefill target; the targets list carries the chosen pair first and the
    /// alternate decodes of the same prefill after it.
    pub fn decide_pair(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts,
        pools: PairPools<'_>,
        cx: &Constraints<'_>,
        decode_cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        let PairPools { prefills, decodes } = pools;
        let prefill_facts = RequestFacts {
            phase: Phase::Prefill,
            ..*facts
        };
        let prefill = self.decide(&prefill_facts, pool, prefills, cx, scratch);
        let Decision::Pick(prefill_pick) = prefill else {
            return prefill;
        };
        let decode_facts = RequestFacts {
            phase: Phase::Decode,
            ..*facts
        };
        let Some((decode_index, decode_profile)) = self.profile_for(facts.model, Phase::Decode)
        else {
            return Decision::Shed(Shed {
                reason: ShedReason::PolicyDeclined,
                emptied_by: "binding",
                retry_after_secs: 1,
                wait_allowed: false,
                trace: prefill_pick.trace,
            });
        };
        let mut last_shed: Option<Shed> = None;
        for prefill_target in prefill_pick.targets.iter() {
            let group = prefills[prefill_target.position as usize].partner_group;
            // The decode profile runs over the chosen prefill's partners
            // alone; the group is a per-candidate fact, so no host callback
            // is needed to narrow the set.
            scratch.decode_kept.clear();
            let partners = PartnerSet { decodes, group };
            let decision = partners.decide(
                decode_profile,
                decode_index,
                &decode_facts,
                pool,
                decode_cx,
                scratch,
            );
            match decision {
                Decision::Pick(decode_pick) => {
                    let mut targets = Targets::new();
                    for decode in decode_pick.targets.iter() {
                        targets.push(Target {
                            position: prefill_target.position,
                            id: prefill_target.id,
                            dp_rank: prefill_target.dp_rank,
                            decode: Some(DecodeTarget {
                                position: decode.position,
                                id: decode.id,
                                dp_rank: decode.dp_rank,
                            }),
                        });
                    }
                    return Decision::Pick(Pick {
                        targets,
                        trace: prefill_pick.trace,
                    });
                }
                Decision::Shed(shed) => last_shed = Some(shed),
            }
        }
        Decision::Shed(last_shed.unwrap_or(Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: "partners",
            retry_after_secs: 1,
            wait_allowed: false,
            trace: prefill_pick.trace,
        }))
    }
}

/// The decodes of one partner group, decided with the decode scratch.
struct PartnerSet<'a> {
    decodes: &'a [Candidate],
    group: u32,
}

impl PartnerSet<'_> {
    fn decide(
        &self,
        profile: &Profile,
        index: u16,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts,
        cx: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Decision {
        // A partner filter in front of the profile's own, with the decode
        // scratch so the prefill's survivors stay intact for the next prefill
        // target.
        let group = self.group;
        scratch.decode_kept.clear();
        scratch
            .decode_kept
            .extend((0..self.decodes.len() as u32).filter(|&i| {
                let c = &self.decodes[i as usize];
                c.partner_group == group
                    && profile
                        .filters
                        .iter()
                        .filter(|f| !f.soft())
                        .all(|f| f.keep(facts, c, cx))
            }));
        if scratch.decode_kept.is_empty() {
            return Decision::Shed(Shed {
                reason: ShedReason::NoEndpoints,
                emptied_by: "partners",
                retry_after_secs: 1,
                wait_allowed: false,
                trace: Trace {
                    profile: index,
                    seed: facts.seed,
                    generation: pool.generation,
                    ..Trace::default()
                },
            });
        }
        for soft in profile.filters.iter().filter(|f| f.soft()) {
            if scratch
                .decode_kept
                .iter()
                .any(|&i| soft.keep(facts, &self.decodes[i as usize], cx))
            {
                scratch
                    .decode_kept
                    .retain(|&i| soft.keep(facts, &self.decodes[i as usize], cx));
            }
        }
        combine(
            profile,
            facts,
            self.decodes,
            &scratch.decode_kept,
            &mut scratch.decode_scores,
            &mut scratch.decode_stage,
        );
        let mut targets = Targets::new();
        profile.picker.pick(
            facts,
            self.decodes,
            &scratch.decode_kept,
            &scratch.decode_scores,
            &mut targets,
        );
        let trace = trace_of(
            index,
            facts,
            pool,
            self.decodes,
            &scratch.decode_kept,
            &scratch.decode_scores,
        );
        if targets.is_empty() {
            return Decision::Shed(Shed {
                reason: ShedReason::PolicyDeclined,
                emptied_by: profile.picker.name(),
                retry_after_secs: 1,
                wait_allowed: false,
                trace,
            });
        }
        Decision::Pick(Pick { targets, trace })
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

    fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side) {
        for profile in &self.profiles {
            for w in &profile.scorers {
                w.scorer.on_dispatched(lease, target, side);
            }
            profile.picker.on_dispatched(lease, target, side);
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

    fn on_released(&self, lease: &Lease) {
        for profile in &self.profiles {
            for w in &profile.scorers {
                w.scorer.on_released(lease);
            }
            profile.picker.on_released(lease);
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

    fn engine() -> Engine {
        Engine::new(
            vec![Profile {
                name: "least".into(),
                filters: vec![Box::new(Healthy), Box::new(NotTried)],
                scorers: vec![Weighted {
                    scorer: Box::new(LeastRequests),
                    weight: 1.0,
                }],
                picker: Box::new(Best),
            }],
            vec![Binding {
                model: "*".into(),
                phases: vec![],
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
            Decision::Pick(_) => panic!("an unhealthy pool picked"),
        }
    }

    #[test]
    fn the_pair_plan_pairs_the_prefill_with_a_decode_of_its_group() {
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
                profile: 0,
                decode_profile: Some(0),
            }],
            1,
            Needs::default(),
        );
        let mut prefills = [candidate(0, 2), candidate(1, 0)];
        prefills[0].partner_group = 1;
        prefills[1].partner_group = 2;
        let mut decodes = [candidate(0, 0), candidate(1, 7), candidate(2, 1)];
        decodes[0].partner_group = 1;
        decodes[1].partner_group = 2;
        decodes[2].partner_group = 2;
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
            first.decode.map(|d| d.position),
            Some(2),
            "its least-loaded partner"
        );
        assert_eq!(
            pick.targets.len(),
            2,
            "the alternate decode of the same prefill"
        );
        assert_eq!(
            pick.targets.as_slice()[1].decode.map(|d| d.position),
            Some(1)
        );
    }

    #[test]
    fn bindings_match_globs_and_phases_in_order() {
        let b = Binding {
            model: "pd/*".into(),
            phases: vec![Phase::Prefill],
            profile: 0,
            decode_profile: None,
        };
        assert!(b.matches("pd/a", Phase::Prefill));
        assert!(!b.matches("pd/a", Phase::Single));
        assert!(!b.matches("x/a", Phase::Prefill));
        let any = Binding {
            model: "*".into(),
            phases: vec![],
            profile: 0,
            decode_profile: None,
        };
        assert!(any.matches("anything", Phase::Decode));
    }
}
