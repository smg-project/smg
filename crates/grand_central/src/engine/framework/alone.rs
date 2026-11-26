//! Which places of a document one pick of a picker decides.
//!
//! A caller that holds every rule of a placement itself (a gateway's own
//! placement pass) builds one policy for each place it places for: a model
//! and a phase. [`Engine::place_for`] says what the document holds for such a
//! place, found once where that policy is built and never at a request.

use std::fmt;

use super::{profile_index, Binding, Engine, Marks, PairSide, Plan, Profile};
use crate::{document::spec::Needs, engine::view::Phase, stages::filters::RowTests};

/// A profile whose whole decision is one pick of its picker
/// ([`Engine::place_for`]), by its place in the document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AloneProfile(pub(crate) u16);

/// What a document holds for one place a caller builds a policy for: a
/// model and a phase ([`Engine::place_for`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Place<'a> {
    /// No binding names the model for the phase.
    Unbound,
    /// A binding names the model for the phase, and its plan has no side
    /// for it: a single request under a pair's plan, an encode request
    /// under a plan that names no encode side. A caller that places
    /// requests of that phase for the model has no profile for them.
    NoSide,
    /// The place's profile is decided by one pick of its picker over the
    /// workers the caller hands: the caller asks that pick and nothing else.
    Alone(AloneProfile),
    /// The place asks more of a decision than that pick gives.
    NotAlone {
        /// The profile's name.
        profile: &'a str,
        /// What asks more: the first such stage of the profile, else the
        /// binding's sessions rule or its shadow.
        in_the_way: InTheWay,
    },
}

/// A place a binding of a document names that asks more of a decision than
/// one pick of its profile's picker ([`Engine::not_alone`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct NotAlone<'a> {
    /// The model pattern of the binding that names the place.
    pub model: &'a str,
    /// The place's phase.
    pub phase: Phase,
    /// The profile the binding resolves the phase to.
    pub profile: &'a str,
    /// What asks more: the first such stage of the profile, else the
    /// binding's sessions rule or its shadow.
    pub in_the_way: InTheWay,
}

/// What stands between a place and a pick its profile's picker makes alone:
/// a stage of the profile, by the stage's name, or what the binding that
/// names the place states beside its profile: a sessions rule, a shadow.
/// The list grows with what a binding can state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum InTheWay {
    /// A filter that is neither a test of a worker's health nor a test the
    /// caller's own pass holds for the place.
    Filter(&'static str),
    /// One of the two tests of a worker's health is not among the profile's
    /// filters, by that filter's name. The pick tests a worker's health
    /// whole, so it would add the test.
    NoFilter(&'static str),
    /// A scorer.
    Scorer(&'static str),
    /// The picker: it does not place over a pool by itself, or its pick
    /// reads a fact of the request.
    Picker(&'static str),
    /// The binding's sessions rule: it pins a keyed request ahead of the
    /// profile, which one pick would step past.
    Sessions,
    /// The binding's shadow: its profile runs beside a whole decision, over
    /// that decision's records, and one pick would make none.
    Shadow,
}

impl fmt::Display for InTheWay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Filter(name) => write!(f, "the filter `{name}`"),
            Self::NoFilter(name) => write!(f, "no `{name}` filter"),
            Self::Scorer(name) => write!(f, "the scorer `{name}`"),
            Self::Picker(name) => write!(f, "the picker `{name}`"),
            Self::Sessions => f.write_str("the binding's sessions rule"),
            Self::Shadow => f.write_str("the binding's shadow"),
        }
    }
}

/// One place a binding names: the binding with its position in the
/// document, the phase, and the profile the binding resolves the phase to
/// with that profile's position.
#[derive(Clone, Copy)]
struct Named<'a> {
    at: u32,
    binding: &'a Binding,
    phase: Phase,
    index: u16,
    profile: &'a Profile,
}

impl Engine {
    /// What the document holds for the place of `model` and `phase`: the
    /// profile the first matching binding resolves the phase to
    /// ([`Self::profile_for`]), and whether one pick of that profile's
    /// picker over a pool's workers is the place's whole decision
    /// ([`Picker::pick_alone`]).
    ///
    /// It is, where the picker says so, the profile holds no scorer, and
    /// its filters are the two tests of a worker's health with no other
    /// but the tests the caller's own pass holds for the place. That pass
    /// tests a worker as the worker tests itself for routing, which is the
    /// two health tests. It steps past the workers a retry tried by its own
    /// rule at every place. At a prefill place it has narrowed the
    /// candidates to the prefills with a free slot of their window, and at
    /// a decode place to the decodes with a free room, before it asks: both
    /// are bits a host supplies, and that pass is the host. Any other
    /// filter is one the pick would drop, so a profile that holds one is
    /// not picked so, and the answer names it.
    ///
    /// The binding is asked too, behind the profile's stages. A sessions
    /// rule pins a keyed request ahead of the profile, and a shadow profile
    /// runs beside a whole decision: one pick would do neither, so a place
    /// under either is not picked so, and the answer names which
    /// ([`InTheWay::Sessions`], [`InTheWay::Shadow`]).
    ///
    /// [`Picker::pick_alone`]: super::Picker::pick_alone
    #[must_use]
    pub fn place_for(&self, model: &str, phase: Phase) -> Place<'_> {
        let found = self.binding_position(model, phase);
        let Some((at, binding)) = found.and_then(|at| Some((at, self.binding_at(at)?))) else {
            return Place::Unbound;
        };
        let Some(index) = profile_index(binding, phase) else {
            return Place::NoSide;
        };
        let Some(profile) = self.profiles.get(usize::from(index)) else {
            return Place::NoSide;
        };
        let place = Named {
            at,
            binding,
            phase,
            index,
            profile,
        };
        match self.in_the_way(place) {
            None => Place::Alone(AloneProfile(index)),
            Some(in_the_way) => Place::NotAlone {
                profile: &profile.name,
                in_the_way,
            },
        }
    }

    /// The first place a binding of the document names that is not decided
    /// by one pick of its profile's picker ([`Self::place_for`]): each
    /// binding in the document's order, at each phase it covers and its
    /// plan has a side for. `None` where every such place is picked so. A
    /// caller that asks the engine for picks alone refuses a document that
    /// holds such a place, by what this names.
    #[must_use]
    pub fn not_alone(&self) -> Option<NotAlone<'_>> {
        self.places().find_map(|place| {
            let in_the_way = self.in_the_way(place)?;
            Some(NotAlone {
                model: &place.binding.model,
                phase: place.phase,
                profile: &place.profile.name,
                in_the_way,
            })
        })
    }

    /// Every place a binding of the document names: each binding in the
    /// document's order, at each phase it covers and its plan has a side
    /// for.
    fn places(&self) -> impl Iterator<Item = Named<'_>> {
        const PHASES: [Phase; 4] = [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode];
        let bindings = (0_u32..).zip(&self.bindings);
        bindings.flat_map(move |(at, entry)| {
            let binding = &entry.binding;
            PHASES.into_iter().filter_map(move |phase| {
                if !binding.covers(phase) {
                    return None;
                }
                let index = profile_index(binding, phase)?;
                let profile = self.profiles.get(usize::from(index))?;
                Some(Named {
                    at,
                    binding,
                    phase,
                    index,
                    profile,
                })
            })
        })
    }

    /// Whether every request of the document is a single request that a
    /// host which feeds `fed` has decided whole in its own process. Each
    /// binding plans single requests alone and states no sessions rule,
    /// and each place of it is one of two:
    ///
    /// - one pick of its picker decides it ([`Self::not_alone`]), which
    ///   asks nothing of the host;
    /// - its stages declare nothing a host must feed outside `fed`
    ///   ([`Needs::within`]): the whole pass of a gateway's own process
    ///   decides it over records written from the gateway's own data. A
    ///   shadow the place asks runs in that pass, over the same records,
    ///   so its profile's stages are held to `fed` too.
    ///
    /// A host that feeds nothing still has decided whole a place whose
    /// stages read a worker's record and nothing more. The set of places
    /// grows with what a host feeds a whole decision.
    #[must_use]
    pub fn single_and_fed(&self, fed: Needs) -> bool {
        let single = |entry: &super::Entry| {
            entry.binding.plan == Plan::Single && entry.binding.sessions.is_none()
        };
        let decided = |place: Named<'_>| {
            if self.in_the_way(place).is_none() {
                return true;
            }
            let own = self.profile_needs.get(usize::from(place.index));
            let shadow = if self.shadowed(place) {
                self.shadow_needs(place.binding, place.phase)
            } else {
                Needs::default()
            };
            own.is_some_and(|own| own.or(shadow).within(fed))
        };
        self.bindings.iter().all(single) && self.places().all(decided)
    }

    /// What stands between `place` and one pick of its profile's picker:
    /// the first stage of the profile that asks more
    /// ([`Self::stage_in_the_way`]), else the binding's sessions rule, else
    /// its shadow. `None` where nothing does. The one function that says
    /// so: [`Self::place_for`] and [`Self::not_alone`] answer by it, and
    /// the engine of a gateway's own process answers a plain request a
    /// place by theirs.
    ///
    /// A stage is named ahead of the binding, so a place that named a
    /// stage before the binding was asked names the same stage.
    fn in_the_way(&self, place: Named<'_>) -> Option<InTheWay> {
        if let Some(stage) = self.stage_in_the_way(place.index, place.profile, place.phase) {
            return Some(stage);
        }
        let binding = place.binding;
        if binding.plan == Plan::Single && binding.sessions.is_some() {
            return Some(InTheWay::Sessions);
        }
        self.shadowed(place).then_some(InTheWay::Shadow)
    }

    /// Whether a decision of `place` asks its binding's shadow: a single's
    /// by the binding's word ([`Self::shadow_at`]), a pair's by the side of
    /// the place's phase ([`Self::shadows_side`]). One read under a
    /// document that names no shadow.
    fn shadowed(&self, place: Named<'_>) -> bool {
        match (place.binding.plan, place.phase) {
            (Plan::Single, _) => self.shadow_at(place.at).is_some(),
            (_, Phase::Prefill) => self.shadows_side(place.at, PairSide::Prefill),
            (_, Phase::Decode) => self.shadows_side(place.at, PairSide::Decode),
            (_, Phase::Single | Phase::Encode) => false,
        }
    }

    /// The first stage of `profile`, which stands at `index`, that asks
    /// more of a decision than one pick of its picker at a place of
    /// `phase`. `None` where no stage does.
    fn stage_in_the_way(&self, index: u16, profile: &Profile, phase: Phase) -> Option<InTheWay> {
        let held = RowTests::held_by_a_pass(phase);
        let mut tests = RowTests::none();
        for filter in &profile.filters {
            match RowTests::of(std::slice::from_ref(filter)) {
                Some(test) if test.health_or(held) => tests = tests.with(test),
                _ => return Some(InTheWay::Filter(filter.name())),
            }
        }
        if let Some(name) = tests.health_test_left_out() {
            return Some(InTheWay::NoFilter(name));
        }
        if let Some(weighted) = profile.scorers.first() {
            return Some(InTheWay::Scorer(weighted.scorer.name()));
        }
        // The turns' mark says how the picker's draw picks: it asks no
        // more of a decision than the draw's mark it stands beside.
        let asks_more = self
            .marks
            .get(usize::from(index))
            .is_none_or(|marks| marks.0 & !(Marks::DRAWS_FIRST | Marks::TURNS) != 0);
        if asks_more || !profile.picker.picks_alone() {
            return Some(InTheWay::Picker(profile.picker.name()));
        }
        None
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/framework/alone/tests.rs"]
mod tests;
