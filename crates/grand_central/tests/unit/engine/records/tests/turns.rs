//! A rotation behind a whole pass over the records a decision wrote whole:
//! the set it turns, the key the set is turned by, and which picker is
//! asked for a turn of it.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::{
    document::spec::{Needs, ScoreSpec},
    engine::{
        decision::{Targets, MAX_TARGETS},
        framework::{Binding, Engine, Picker, Profile, Scorer, TieBand, WeightedScorer},
    },
    policies::round_robin::RoundRobin,
    stages::{
        filters::{Healthy, NotOverloaded, NotTried},
        registry::StageRegistry,
        Lifecycle,
    },
};

/// Every target of a decision, by position and id; none for a shed.
fn targets(decision: &Decision) -> Vec<(u32, WorkerId)> {
    decision.pick().map_or_else(Vec::new, |pick| {
        pick.targets
            .iter()
            .map(|target| (target.position, target.id))
            .collect()
    })
}

/// A scorer of a host's own build that says nothing of itself: every
/// default of the trait, and one score for every row. A rotation behind
/// it is decided by the whole pass over every record.
#[derive(Debug)]
struct Level;

impl Lifecycle for Level {}

impl Scorer for Level {
    fn name(&self) -> &'static str {
        "level"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.5);
        TieBand::NORMALIZED
    }
}

/// The crate's own rotation behind a whole pass turns the set of the rows
/// the pass kept, by a key of those rows. Where [`decide`] wrote the list's
/// every record and the pass kept them all, the pass takes the key the list
/// made once; over rows no front spoke of ([`PoolRecords::write`], then the
/// engine) the rotation makes the key of the rows. It is one key: an engine
/// asked at the two entries by turns takes the turns of one rotation, which
/// are the turns of an engine made alike that is asked over written rows
/// alone. With a worker that is not ready the front says the list's key all
/// the same and the pass, which kept a part of the list, does not read it:
/// the rotation makes the key of the rows kept, and the two entries still
/// turn one set.
#[test]
fn a_rotation_behind_a_whole_pass_takes_one_sets_turns_at_either_entry() {
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer("level", || Box::new(Level))
        .expect("a free name");
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    document.generation = 1;
    for profile in document.profiles.values_mut() {
        profile.score.push(ScoreSpec::Name("level".to_owned()));
    }
    let applied = || {
        AppliedSpec::validate_in(&document, &catalog, &[]).expect("the catalog holds the scorer")
    };
    for not_ready in [None, Some(2)] {
        // Two engines made alike, so a rotation turns alike.
        let (both, written) = (applied(), applied());
        // A document's build says that its stage table made the pickers.
        assert!(both.engine().turns_a_set() && written.engine().turns_a_set());
        let workers = fleet(8);
        if let Some(at) = not_ready {
            workers[at].set_status(WorkerStatus::NotReady);
        }
        let clock = DecisionClock::new();
        let both_pool = PoolRecords::new(&workers, &both, &clock);
        let written_pool = PoolRecords::new(&workers, &written, &clock);
        let cx = Constraints::default();
        let (mut rows, mut scratch) = (Vec::new(), Scratch::default());
        let mut over_rows = |spec: &AppliedSpec, pool: &PoolRecords, now: Instant| {
            let live = Live::at(spec.engine(), &clock, now, None, None);
            let mut facts = single();
            facts.now_micros = live.now_micros();
            let (written, pool_facts) = pool.write(&workers, &live, &mut rows);
            let engine = spec.engine();
            targets(&engine.decide(&facts, &pool_facts, written, &cx, &mut scratch))
        };
        for turn in 0..40 {
            let now = Instant::now();
            let asked = if turn % 2 == 0 {
                let live = Live::at(both.engine(), &clock, now, None, None);
                let mut facts = single();
                targets(&decide(
                    both.engine(),
                    &both_pool,
                    &workers,
                    &live,
                    &mut facts,
                    &cx,
                ))
            } else {
                over_rows(&both, &both_pool, now)
            };
            let alone = over_rows(&written, &written_pool, now);
            let under = format!("turn {turn}, not ready {not_ready:?}");
            assert_eq!(alone.len(), MAX_TARGETS, "{under}");
            assert_eq!(asked, alone, "{under}");
        }
    }
}

/// How often a [`Wrapped`] picker was asked for a pick over rows and for a
/// turn among a set.
#[derive(Debug, Default)]
struct Asked {
    pick: AtomicUsize,
    among: AtomicUsize,
}

/// A host's picker round the crate's public rotation, which answers the
/// rotation's name: its `pick` is the rotation's. It states the two words
/// of the trait the rotation states where `words` says so, and every other
/// word of the trait is the default's. Its turn among a set is the
/// rotation's only where `among` says so: else it answers as the trait's
/// default does, the first. It counts what it is asked.
#[derive(Debug)]
struct Wrapped {
    rotation: RoundRobin,
    words: bool,
    among: bool,
    asked: Arc<Asked>,
}

impl Lifecycle for Wrapped {}

impl Picker for Wrapped {
    fn name(&self) -> &'static str {
        "round-robin"
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
        self.asked.pick.fetch_add(1, Ordering::Relaxed);
        self.rotation
            .pick(facts, candidates, kept, scores, band, out);
    }

    fn picks_alone(&self) -> bool {
        self.words
    }

    fn takes_turns(&self) -> bool {
        self.words
    }

    fn pick_among(&self, set: u64, passed: usize) -> usize {
        self.asked.among.fetch_add(1, Ordering::Relaxed);
        if self.among {
            self.rotation.pick_among(set, passed)
        } else {
            0
        }
    }
}

/// An engine built in code, as a host builds one: one profile of the
/// engine's own three filters, a host's scorer and a [`Wrapped`] picker,
/// bound to every model. The counts are the picker's.
fn built_in_code(words: bool, among: bool) -> (Engine, Arc<Asked>) {
    let asked = Arc::new(Asked::default());
    let picker = Wrapped {
        rotation: RoundRobin::new(),
        words,
        among,
        asked: Arc::clone(&asked),
    };
    let mut profile = Profile::of("turns".to_owned(), Box::new(picker));
    profile.filters = vec![
        Box::new(Healthy),
        Box::new(NotOverloaded),
        Box::new(NotTried),
    ];
    profile
        .scorers
        .push(WeightedScorer::of(Box::new(Level), 1.0));
    let binding = Binding::of("*".to_owned(), 0);
    let engine = Engine::new(vec![profile], vec![binding], 1, Needs::default());
    (engine, asked)
}

/// The worker each of `turns` decisions of `engine` leads with, by its
/// place in a list of eight workers, asked at the front that writes the
/// list's every record ([`decide`]). The workers are all ready but the one
/// at `not_ready`.
fn leads_at_the_front(engine: &Engine, turns: usize, not_ready: Option<usize>) -> Vec<u32> {
    let workers = fleet(8);
    if let Some(at) = not_ready {
        workers[at].set_status(WorkerStatus::NotReady);
    }
    let clock = DecisionClock::new();
    let pool = PoolRecords::new(&workers, &applied("round_robin"), &clock);
    let cx = Constraints::default();
    (0..turns)
        .map(|_| {
            let live = Live::at(engine, &clock, Instant::now(), None, None);
            let mut facts = single();
            picked(&decide(engine, &pool, &workers, &live, &mut facts, &cx)).0
        })
        .collect()
}

/// Whether each lead stands one place behind the one before it, round a
/// list of eight: the turns of a rotation over the whole list.
fn turns_the_list(leads: &[u32]) -> bool {
    leads.windows(2).all(|pair| pair[1] == (pair[0] + 1) % 8)
}

/// A host's picker that answers the rotation's name and forwards `pick`
/// alone is asked for its `pick` over the rows a decision wrote whole, as
/// over any rows: an engine built in code holds whatever picker its host
/// handed it, and a name says nothing of what a picker answers when it is
/// asked for a turn among a set. Its requests take the turns of its
/// rotation, and none of them is placed by that other answer, which is the
/// default's: the first row.
#[test]
fn a_hosts_picker_of_the_rotations_name_is_asked_for_its_pick_at_the_front() {
    let (engine, asked) = built_in_code(false, false);
    let leads = leads_at_the_front(&engine, 24, None);
    assert!(turns_the_list(&leads), "the leads: {leads:?}");
    assert_eq!(asked.pick.load(Ordering::Relaxed), 24);
    assert_eq!(asked.among.load(Ordering::Relaxed), 0);
    assert!(
        !engine.turns_a_set(),
        "an engine built in code is told nothing"
    );
}

/// The reach of the turn by the list's key, held by what a picker is asked.
/// In an engine whose builder says a catalog made its pickers
/// ([`Engine::made_by_a_catalog`], the word of a document's build), a whole
/// pass that kept every row of a list its front wrote whole asks the picker
/// for one turn among the set and never for a `pick`, and the requests take
/// the turns of the rotation. With a worker that is not ready the pass kept
/// a part of the list and asks for the `pick`, as it asks every picker of
/// an engine that was told nothing.
#[test]
fn a_whole_pass_of_a_catalogs_rotation_asks_one_turn_among_the_list_written_whole() {
    let (mut engine, asked) = built_in_code(true, true);
    engine.made_by_a_catalog();
    let leads = leads_at_the_front(&engine, 24, None);
    assert!(turns_the_list(&leads), "the leads: {leads:?}");
    assert_eq!(asked.among.load(Ordering::Relaxed), 24);
    assert_eq!(asked.pick.load(Ordering::Relaxed), 0);
    leads_at_the_front(&engine, 24, Some(2));
    assert_eq!(asked.among.load(Ordering::Relaxed), 24);
    assert_eq!(asked.pick.load(Ordering::Relaxed), 24);
}

/// The same picker with the two words of the trait the rotation states
/// ([`Picker::takes_turns`], [`Picker::picks_alone`]) and no turn among a
/// set of its own: the words do not say that its turn among a set is the
/// turn its `pick` takes, so it too is asked for its `pick`.
#[test]
fn a_hosts_picker_that_states_the_rotations_words_is_asked_for_its_pick_at_the_front() {
    let (engine, asked) = built_in_code(true, false);
    let leads = leads_at_the_front(&engine, 24, None);
    assert!(turns_the_list(&leads), "the leads: {leads:?}");
    assert_eq!(asked.pick.load(Ordering::Relaxed), 24);
    assert_eq!(asked.among.load(Ordering::Relaxed), 0);
}
