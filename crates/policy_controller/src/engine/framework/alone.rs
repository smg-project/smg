//! Which places of a document one pick of a picker decides.
//!
//! A caller that holds every rule of a placement itself (a gateway's own
//! placement pass) builds one policy for each place it places for: a model
//! and a phase. [`Engine::place_for`] says what the document holds for such a
//! place, found once where that policy is built and never at a request.

use std::{fmt, sync::Arc};

use super::{profile_index, Binding, Engine, Marks, Plan, Profile};
use crate::{
    document::spec::{Needs, PlanName},
    engine::view::Phase,
    stages::filters::RowTests,
    tracking::{dispatch_credit::DispatchCredit, lease::EventSet},
};

/// A profile whose whole decision is one pick of its picker
/// ([`Engine::place_for`]), by its place in the document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AloneProfile(pub(crate) u16);

/// What a document holds for one place a caller builds a policy for: a
/// model and a phase ([`Engine::place_for`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// The place's profile asks more of a decision than that pick gives.
    NotAlone {
        /// The profile's name.
        profile: &'a str,
        /// The first of its stages that asks more.
        stage: InTheWay,
    },
}

/// A place a binding of a document names whose profile asks more of a
/// decision than one pick of its picker ([`Engine::not_alone`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotAlone<'a> {
    /// The model pattern of the binding that names the place.
    pub model: &'a str,
    /// The place's phase.
    pub phase: Phase,
    /// The profile the binding resolves the phase to.
    pub profile: &'a str,
    /// The first of the profile's stages that asks more.
    pub stage: InTheWay,
}

/// What stands between a profile and a pick its picker makes alone, by the
/// stage's name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

impl fmt::Display for InTheWay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Filter(name) => write!(f, "the filter `{name}`"),
            Self::NoFilter(name) => write!(f, "no `{name}` filter"),
            Self::Scorer(name) => write!(f, "the scorer `{name}`"),
            Self::Picker(name) => write!(f, "the picker `{name}`"),
        }
    }
}

impl Engine {
    /// What the document holds for the place of `model` and `phase`: the
    /// profile the first matching binding resolves the phase to
    /// ([`Self::profile_for`]), and whether one pick of that profile's
    /// picker over a pool's workers is its whole decision
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
    /// [`Picker::pick_alone`]: super::Picker::pick_alone
    #[must_use]
    pub fn place_for(&self, model: &str, phase: Phase) -> Place<'_> {
        let Some(entry) = self.entry(model, phase) else {
            return Place::Unbound;
        };
        let Some(index) = profile_index(&entry.binding, phase) else {
            return Place::NoSide;
        };
        let Some(profile) = self.profiles.get(usize::from(index)) else {
            return Place::NoSide;
        };
        match self.in_the_way(index, profile, phase) {
            None => Place::Alone(AloneProfile(index)),
            Some(stage) => Place::NotAlone {
                profile: &profile.name,
                stage,
            },
        }
    }

    /// The first place a binding of the document names whose profile is
    /// not decided by one pick of its picker ([`Self::place_for`]): each
    /// binding in the document's order, at each phase it covers and its
    /// plan has a side for. `None` where every such place is picked so. A
    /// caller that asks the engine for picks alone refuses a document that
    /// holds such a place, by what this names.
    #[must_use]
    pub fn not_alone(&self) -> Option<NotAlone<'_>> {
        self.places().find_map(|(binding, phase, index, profile)| {
            let stage = self.in_the_way(index, profile, phase)?;
            Some(NotAlone {
                model: &binding.model,
                phase,
                profile: &profile.name,
                stage,
            })
        })
    }

    /// Every place a binding of the document names, with its profile and
    /// the profile's place in the document: each binding in the document's
    /// order, at each of its places ([`Self::places_of`]).
    fn places(&self) -> impl Iterator<Item = (&Binding, Phase, u16, &Profile)> {
        self.bindings.iter().flat_map(move |entry| {
            let binding = &entry.binding;
            self.places_of(binding)
                .map(move |(phase, index, profile)| (binding, phase, index, profile))
        })
    }

    /// Every place `binding` names, with its profile and the profile's
    /// place in the document: each phase the binding covers and its plan
    /// has a side for.
    fn places_of<'a>(
        &'a self,
        binding: &'a Binding,
    ) -> impl Iterator<Item = (Phase, u16, &'a Profile)> {
        const PHASES: [Phase; 4] = [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode];
        PHASES.into_iter().filter_map(move |phase| {
            if !binding.covers(phase) {
                return None;
            }
            let index = profile_index(binding, phase)?;
            let profile = self.profiles.get(usize::from(index))?;
            Some((phase, index, profile))
        })
    }

    /// Whether a host that asks the engine in its own process can have it
    /// decide every request the document plans as `plan`, given the
    /// `inputs` the host provides for a decision of that plan. `false`
    /// where no binding plans requests so. The host says which plans it
    /// hands the engine, and nothing here names one. Each binding of the
    /// plan states no sessions rule, and each place of it is one of two:
    ///
    /// - one pick of its picker decides it ([`Self::not_alone`]), which
    ///   asks nothing of the host;
    /// - its stages declare no need outside `inputs` ([`Needs::within`]):
    ///   the engine decides it over records written from the host's own
    ///   data.
    ///
    /// A host that provides nothing can still have a place decided whose
    /// stages read a worker's record and nothing more. The set of places
    /// grows with the inputs a host provides.
    ///
    /// The encode side of an encode plan has one more condition. Such a
    /// host has an encode item decided under no lease, as it holds no load
    /// guard on an encoder, so no stage is told an event of the item. A
    /// side whose profile depends on a lease's events would decide on
    /// state nothing keeps, so the engine cannot decide it: one whose
    /// scorer or picker hears a lease's issue, dispatch, completion or
    /// release ([`Lifecycle::hears`]), and one a stage of which reads the
    /// profile's count of sends, which a dispatch's event feeds.
    ///
    /// [`Lifecycle::hears`]: crate::stages::Lifecycle::hears
    ///
    /// A host that serves each plan's requests by a route of their own
    /// asks this a plan at a time: the routes of a plan the engine cannot
    /// decide keep the host's other placement, beside the routes of a plan
    /// it can.
    #[must_use]
    pub fn can_decide(&self, plan: PlanName, inputs: Needs) -> bool {
        self.has_plan(plan)
            && self
                .bindings_with_plan(plan)
                .all(|binding| self.can_decide_binding(binding, inputs))
    }

    /// Whether a binding of the document plans requests as `plan`.
    #[must_use]
    pub fn has_plan(&self, plan: PlanName) -> bool {
        self.bindings_with_plan(plan).next().is_some()
    }

    /// The bindings of the document that plan requests as `plan`.
    fn bindings_with_plan(&self, plan: PlanName) -> impl Iterator<Item = &Binding> {
        self.bindings
            .iter()
            .map(|entry| &entry.binding)
            .filter(move |binding| binding.plan.name() == plan)
    }

    /// Whether such a host can have the engine decide every request of the
    /// document: every plan a binding names ([`Self::can_decide`]).
    /// `inputs` answers what the host provides for a decision of a plan,
    /// and `None` for a plan whose requests it does not hand the engine.
    #[must_use]
    pub fn can_decide_every_plan(&self, inputs: impl Fn(PlanName) -> Option<Needs>) -> bool {
        self.bindings
            .iter()
            .map(|entry| &entry.binding)
            .all(|binding| {
                inputs(binding.plan.name())
                    .is_some_and(|inputs| self.can_decide_binding(binding, inputs))
            })
    }

    /// Whether such a host can have the engine decide every request of
    /// `binding`, given the `inputs` it provides for a decision of the
    /// binding's plan ([`Self::can_decide`]).
    fn can_decide_binding(&self, binding: &Binding, inputs: Needs) -> bool {
        let encode_side =
            |phase| phase == Phase::Encode && matches!(binding.plan, Plan::Epd { .. });
        let decided = |(phase, index, profile): (Phase, u16, &Profile)| {
            // No event of an encode item reaches a stage.
            if encode_side(phase) && self.depends_on_lease_events(profile) {
                return false;
            }
            let needs = self.profile_needs.get(usize::from(index));
            self.in_the_way(index, profile, phase).is_none()
                || needs.is_some_and(|needs| needs.within(inputs))
        };
        binding.sessions.is_none() && self.places_of(binding).all(decided)
    }

    /// Whether a stage of `profile` depends on a lease's events
    /// ([`Self::can_decide`]): a scorer or the picker hears one, or a
    /// stage reads the profile's count of sends. A filter hears no event.
    fn depends_on_lease_events(&self, profile: &Profile) -> bool {
        let scorers = profile.scorers.iter();
        let heard = scorers.map(|weighted| weighted.scorer.hears());
        let heard = heard.fold(profile.picker.hears(), EventSet::union);
        let counts_sends = || {
            let named = |sent: &Arc<DispatchCredit>| sent.profile == profile.name;
            self.sent_credits.iter().any(named)
        };
        heard.intersects(EventSet::LEASE) || counts_sends()
    }

    /// The first stage of `profile`, which stands at `index`, that asks
    /// more of a decision than one pick of its picker at a place of
    /// `phase`. `None` where no stage does.
    fn in_the_way(&self, index: u16, profile: &Profile, phase: Phase) -> Option<InTheWay> {
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
        let asks_more = self
            .marks
            .get(usize::from(index))
            .is_none_or(|marks| marks.0 & !Marks::DRAWS_FIRST != 0);
        if asks_more || !profile.picker.picks_alone() {
            return Some(InTheWay::Picker(profile.picker.name()));
        }
        None
    }
}
