//! Who is told a word of a lease at the engine's own entries: the stages
//! that name the word ([`Lifecycle::hears`]), in the engine's order, and
//! no other stage.

use super::*;
use crate::tracking::{
    lease::EventKind::{self, Completed, Dispatched, Issued, Released},
    TransportClass,
};

/// The words told, each beside the name of the stage that was told it.
type Told = Arc<Mutex<Vec<(&'static str, EventKind)>>>;

/// A stage that writes each word of a lease it is told, and names the
/// words it is made with. It implements every call: an engine that told
/// it a word it does not name would show here.
#[derive(Debug)]
struct Writes {
    name: &'static str,
    takes: EventSet,
    told: Told,
}

impl Writes {
    fn boxed(name: &'static str, takes: &[EventKind], told: &Told) -> Box<Self> {
        Box::new(Self {
            name,
            takes: EventSet::of(takes.iter().copied()),
            told: Arc::clone(told),
        })
    }

    fn write(&self, kind: EventKind) {
        let mut told = self.told.lock().expect("the words told");
        told.push((self.name, kind));
    }
}

impl Lifecycle for Writes {
    fn on_decision_issued(&self, _: &Lease, _: &Decision) {
        self.write(Issued);
    }

    fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
        self.write(Dispatched);
    }

    fn on_completed(&self, _: &Lease, _: &Outcome) {
        self.write(Completed);
    }

    fn on_released(&self, _: &Lease, _: ReleaseReason) {
        self.write(Released);
    }

    fn hears(&self) -> EventSet {
        self.takes
    }
}

impl Scorer for Writes {
    fn name(&self) -> &'static str {
        self.name
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (score, _) in out.iter_mut().zip(kept) {
            *score = 1.0;
        }
        TieBand::NORMALIZED
    }
}

impl Picker for Writes {
    fn name(&self) -> &'static str {
        self.name
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

/// An engine of two profiles whose stages name different words. The
/// first holds a scorer that takes the dispatch, one that takes the end
/// and the release, and a picker that takes every word. The second holds
/// a scorer that takes the issue and the release, and a picker that names
/// none. The second decides the case's request.
fn written(told: &Told) -> Engine {
    let scorer = |name, takes: &[EventKind]| WeightedScorer {
        scorer: Writes::boxed(name, takes, told),
        weight: 1.0,
    };
    let first = Profile {
        name: "first".into(),
        filters: vec![],
        scorers: vec![
            scorer("first-sends", &[Dispatched]),
            scorer("first-ends", &[Completed, Released]),
        ],
        picker: Writes::boxed(
            "first-picker",
            &[Issued, Dispatched, Completed, Released],
            told,
        ),
    };
    let second = Profile {
        name: "second".into(),
        filters: vec![],
        scorers: vec![scorer("second-issues", &[Issued, Released])],
        picker: Writes::boxed("second-picker", &[], told),
    };
    let binding = |model: &str, profile| Binding {
        model: model.into(),
        profile,
        ..single_binding()
    };
    let bindings = vec![binding("elsewhere", 0), binding("*", 1)];
    Engine::new(vec![first, second], bindings, 1, Needs::default())
}

/// At the engine's own entries each word of a lease is told to the stages
/// that name it and to no other, in the engine's order: the profiles',
/// and a profile's scorers ahead of its picker. The request is the second
/// profile's, and the stages of the first that take a word hear it: a
/// word's list holds the stages of every profile that take it. Every
/// stage here implements every call, and a picker that names no word is
/// told none. The four lists differ, so a word that walked another word's
/// list would show.
///
/// A host with a file of leases tells a lease of the engine's own
/// generation through what stands of the engine ([`Engine::stands`]): the
/// same stages hear, in the same order.
#[test]
fn a_word_of_a_lease_is_told_to_the_stages_that_name_it_in_the_engines_order() {
    let told = Told::default();
    let engine = written(&told);
    let rows = [candidate(0, 3), candidate(1, 13)];
    let (facts, cx) = (RequestFacts::default(), Constraints::default());
    // The pool's facts state the engine's generation, as its host states
    // it: the decision bears it, and the lease its host issues of that
    // decision bears the decision's.
    let pool = PoolFacts {
        generation: engine.generation(),
        ..PoolFacts::default()
    };
    let mut scratch = Scratch::new();
    let decision = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
    let target = decision.pick().and_then(|pick| pick.targets.first());
    let target = *target.expect("the second profile's pick");
    assert!(told.lock().expect("the words told").is_empty());

    let lease = Lease {
        booked: target.id,
        generation: decision.trace().generation,
        ..Lease::default()
    };
    assert_eq!(lease.generation, engine.generation());
    let outcome = Outcome::reported(200, TransportClass::Ok);
    // The names of the stages told since the last read, under one word.
    let heard = |kind: EventKind| {
        let mut told = told.lock().expect("the words told");
        let told = std::mem::take(&mut *told);
        assert!(told.iter().all(|(_, word)| *word == kind), "{told:?}");
        told.into_iter().map(|(name, _)| name).collect::<Vec<_>>()
    };
    let issued = ["first-picker", "second-issues"];
    let dispatched = ["first-sends", "first-picker"];
    let completed = ["first-ends", "first-picker"];
    let released = ["first-ends", "first-picker", "second-issues"];

    engine.on_decision_issued(&lease, &decision);
    assert_eq!(heard(Issued), issued);
    engine.on_dispatched(&lease, &target, Side::Single, 0);
    assert_eq!(heard(Dispatched), dispatched);
    engine.on_completed(&lease, &outcome);
    assert_eq!(heard(Completed), completed);
    engine.on_released(&lease, ReleaseReason::ClientGone);
    assert_eq!(heard(Released), released);

    #[cfg(feature = "controller")]
    {
        let stands = engine.stands();
        stands.decision_issued(&lease, &decision);
        assert_eq!(heard(Issued), issued);
        stands.dispatched_stages(&lease, &target, Side::Single, 0);
        assert_eq!(heard(Dispatched), dispatched);
        stands.completed(&lease, &outcome);
        assert_eq!(heard(Completed), completed);
        stands.released(&lease, ReleaseReason::ClientGone);
        assert_eq!(heard(Released), released);
    }
}

/// A word's list holds a scorer wherever it stands in its profile. The
/// profile holds 65,537 scorers, one more than sixteen bits count, and
/// the last alone names a word. The document's answer holds that word,
/// and the engine tells it to that scorer.
#[test]
fn a_word_reaches_a_scorer_past_sixteen_bits_of_its_profiles_scorers() {
    let told = Told::default();
    let scorer = |name, takes: &[EventKind]| WeightedScorer {
        scorer: Writes::boxed(name, takes, &told),
        weight: 1.0,
    };
    let mut scorers: Vec<_> = (0..=u16::MAX).map(|_| scorer("ahead", &[])).collect();
    scorers.push(scorer("last", &[Released]));
    assert_eq!(scorers.len(), 65_537);
    let wide = Profile {
        name: "wide".into(),
        filters: vec![],
        scorers,
        picker: Writes::boxed("picker", &[], &told),
    };
    let engine = Engine::new(vec![wide], vec![single_binding()], 1, Needs::default());
    assert!(engine.hears().contains(Released), "the document's answer");

    let lease = Lease {
        generation: engine.generation(),
        ..Lease::default()
    };
    engine.on_released(&lease, ReleaseReason::ClientGone);
    let told = told.lock().expect("the words told");
    assert_eq!(*told, [("last", Released)]);
}

/// The build asks each scorer and each picker once what it takes: a
/// profile's picker ahead of its scorers, the scorers in their order, and
/// the profiles in theirs.
#[test]
fn the_build_asks_a_profiles_picker_ahead_of_its_scorers_and_each_once() {
    /// A stage that writes its name where it is asked what it takes.
    #[derive(Debug)]
    struct Asked(&'static str, Arc<Mutex<Vec<&'static str>>>);
    impl Lifecycle for Asked {
        fn hears(&self) -> EventSet {
            self.1.lock().expect("the askings").push(self.0);
            EventSet::NONE
        }
    }
    impl Scorer for Asked {
        fn name(&self) -> &'static str {
            self.0
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
    impl Picker for Asked {
        fn name(&self) -> &'static str {
            self.0
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
    let asked = Arc::new(Mutex::new(Vec::new()));
    let profile = |name: &str, scorers: &[&'static str], picker| Profile {
        name: name.into(),
        filters: vec![],
        scorers: scorers
            .iter()
            .map(|&scorer| WeightedScorer {
                scorer: Box::new(Asked(scorer, Arc::clone(&asked))),
                weight: 1.0,
            })
            .collect(),
        picker: Box::new(Asked(picker, Arc::clone(&asked))),
    };
    let profiles = vec![
        profile("first", &["first-a", "first-b"], "first-picker"),
        profile("second", &["second-a"], "second-picker"),
    ];
    let _engine = Engine::new(profiles, vec![single_binding()], 1, Needs::default());
    let asked = asked.lock().expect("the askings");
    let in_order = [
        "first-picker",
        "first-a",
        "first-b",
        "second-picker",
        "second-a",
    ];
    assert_eq!(*asked, in_order);
}
