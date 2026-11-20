//! The shadow at the door. A binding's shadow profile is asked beside a
//! live decision, ahead of the decision's commit. What it said is counted
//! by the binding's position and kept with a sampled record. Nothing of it
//! changes, fails or books the live answer.

use std::{
    error::Error,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    thread,
    time::Instant,
};

use super::{
    counted, dispatched, fleet, items, keyed, pair_fleet, pair_request, place_one, placed,
    push_fleet, release, report, request, seated, GATEWAY,
};
use crate::{
    builtins::target_of,
    catalog::Catalog,
    decision::{Decision, ShadowKind, Targets},
    door::{Door, ON_SHADOW, SHADOW_FLOOR},
    framework::{Picker, TieBand},
    lease::{LeaseId, Lifecycle, Side},
    sessions::SessionVerdict,
    spec::{
        AffinitySource, AppliedSpec, BestParams, PickStage, PolicySpec, PrefixAffinityParams,
        ProfileSpec, ScoreStage, ShadowSpec, WarmupSliceParams,
    },
    store::{Dispatch, LeasesStore, MemoryStore, Said},
    view::{Candidate, RequestFacts, WorkerId},
    Assignment,
};

/// What a case's picker does with a decision.
#[derive(Clone, Copy, Debug)]
enum Rule {
    /// Takes the row at this place among the rows kept, and the row
    /// behind it as its alternate.
    Place(usize),
    /// Takes nobody: its profile declines.
    Nobody,
    /// Panics.
    Panics,
}

/// A picker under a case's rule, counting the decisions that ask it.
#[derive(Debug)]
struct Takes {
    name: &'static str,
    rule: Rule,
    asked: Arc<AtomicUsize>,
}

impl Lifecycle for Takes {}

impl Picker for Takes {
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
        self.asked.fetch_add(1, Ordering::Relaxed);
        match self.rule {
            Rule::Place(place) => {
                for &row in kept.iter().skip(place).take(2) {
                    out.push(target_of(&candidates[row as usize]));
                }
            }
            Rule::Nobody => {}
            Rule::Panics => panic!("the case's own panic, inside a shadow"),
        }
    }
}

/// Applies `generation` of `preset`'s document with the picker `live` in
/// its pick's place and, as its binding's shadow, a second profile of the
/// same stages under the picker `shadow`. `edit` has the document before
/// it is applied. The count is of the decisions that asked the shadow
/// profile's picker.
fn shadowing(
    store: &MemoryStore,
    (generation, preset): (u64, &str),
    (live, shadow): (Rule, Rule),
    edit: impl FnOnce(&mut PolicySpec),
) -> Arc<AtomicUsize> {
    let asked = Arc::new(AtomicUsize::new(0));
    let picker = |name: &'static str, rule: Rule, asked: &Arc<AtomicUsize>| {
        let asked = Arc::clone(asked);
        move || -> Box<dyn Picker> {
            let asked = Arc::clone(&asked);
            Box::new(Takes { name, rule, asked })
        }
    };
    let mut catalog = Catalog::builtin();
    let added = catalog.picker("takes-live", picker("takes-live", live, &Arc::default()));
    added.expect("a free name");
    let added = catalog.picker("takes-shadow", picker("takes-shadow", shadow, &asked));
    added.expect("a free name");
    let mut document = PolicySpec::preset(preset).expect("a preset of this build");
    document.generation = generation;
    let profile = document.profiles.values_mut().next();
    let profile = profile.expect("the preset's profile");
    profile.pick = "takes-live".into();
    let mut twin = profile.clone();
    twin.pick = "takes-shadow".into();
    document.profiles.insert("twin".into(), twin);
    document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    edit(&mut document);
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
    let applied = store.apply(spec.expect("the document validates"));
    applied.expect("a later generation");
    asked
}

/// A door over the fleet's three members under the `random` preset's
/// stages, deciding by `live` and shadowing by `shadow`, and the count of
/// the decisions that asked the shadow's picker.
fn bed(live: Rule, shadow: Rule) -> (Door, Arc<AtomicUsize>) {
    let store = fleet(2);
    let asked = shadowing(&store, (2, "random"), (live, shadow), |_| {});
    (Door::new(store), asked)
}

/// The door's counts of the first binding's shadows: agreed, another
/// target, another outcome, skipped.
fn kinds(door: &Door) -> [u64; 4] {
    kinds_at(door, 0)
}

/// [`kinds`] of the binding at `binding`.
fn kinds_at(door: &Door, binding: u32) -> [u64; 4] {
    use ShadowKind::{Agreed, OtherOutcome, OtherTarget, Skipped};
    [Agreed, OtherTarget, OtherOutcome, Skipped].map(|kind| door.shadowed(binding, kind))
}

/// A picker that takes the first row kept, counting the decisions that
/// hand it named rows.
#[derive(Debug)]
struct SeesNames(Arc<AtomicUsize>);

impl Lifecycle for SeesNames {}

impl Picker for SeesNames {
    fn name(&self) -> &'static str {
        "sees-names"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        let named = usize::from(facts.named.is_some());
        self.0.fetch_add(named, Ordering::Relaxed);
        if let Some(&row) = kept.first() {
            out.push(target_of(&candidates[row as usize]));
        }
    }
}

/// A shadow that takes another row is counted as one, and the answer is
/// the live profile's: the pick a shadow that agrees leaves too, one
/// lease, and one request booked on the pick's worker.
#[test]
fn a_shadow_that_takes_another_row_is_counted_beside_the_live_answer() {
    let (door, asked) = bed(Rule::Place(0), Rule::Place(1));
    let (_, first, _) = placed(&door, &request(b"r-1", 7));
    assert_eq!(kinds(&door), [0, 1, 0, 0]);
    assert_eq!(asked.load(Ordering::Relaxed), 1);
    assert_eq!(LeasesStore::live(&**door.store(), GATEWAY), 1);
    assert_eq!(counted(door.store()), [(first, 1)]);
    let (twin, _) = bed(Rule::Place(0), Rule::Place(0));
    assert_eq!(placed(&twin, &request(b"r-1", 7)).1, first);
    assert_eq!(kinds(&twin), [1, 0, 0, 0]);
}

/// A shadow that declines leaves the live pick and its booking as they
/// were decided. A live decision that declines is answered as its shed,
/// with no lease, whatever its shadow picks. Each is another outcome.
#[test]
fn a_shadow_of_another_outcome_changes_nothing_of_the_live_answer() {
    let (door, _) = bed(Rule::Place(0), Rule::Nobody);
    let (_, first, _) = placed(&door, &request(b"r-1", 7));
    assert_eq!(counted(door.store()), [(first, 1)]);
    assert_eq!(kinds(&door), [0, 0, 1, 0]);
    let (door, asked) = bed(Rule::Nobody, Rule::Place(0));
    let decided = door.decide(GATEWAY, &request(b"r-1", 7), Instant::now());
    let decided = decided.expect("a decision");
    assert!(matches!(decided.decision, Decision::Shed(_)), "{decided:?}");
    assert!(decided.lease.is_none());
    assert_eq!(counted(door.store()), []);
    assert_eq!(asked.load(Ordering::Relaxed), 1);
    assert_eq!(kinds(&door), [0, 0, 1, 0]);
}

/// A request with less than the floor of its budget left keeps it for its
/// answer: its shadow is not asked, and is counted as skipped. At the
/// floor, and with no budget stated, the shadow is asked.
#[test]
fn a_budget_under_the_floor_skips_the_shadow_and_the_answer_leaves() {
    let (door, asked) = bed(Rule::Place(0), Rule::Place(1));
    let floor = u32::try_from(SHADOW_FLOOR.as_micros()).expect("a floor in microseconds");
    let budgeted = |id: &'static [u8], micros: u32| {
        let mut stated = request(id, 7);
        stated.facts.budget_micros = micros;
        stated
    };
    let (_, first, _) = placed(&door, &budgeted(b"short", floor - 1));
    assert_eq!(asked.load(Ordering::Relaxed), 0);
    assert_eq!(kinds(&door), [0, 0, 0, 1]);
    for stated in [budgeted(b"held", floor), request(b"unstated", 7)] {
        assert_eq!(placed(&door, &stated).1, first);
    }
    assert_eq!(asked.load(Ordering::Relaxed), 2);
    assert_eq!(kinds(&door), [0, 2, 0, 1]);
}

/// A shadow is counted behind its decision's commit. A decision whose
/// lease the store refused is ordered again and its shadow asked again:
/// the one answer counts one shadow.
#[test]
fn a_decision_ordered_twice_counts_its_shadow_once() {
    let (door, asked) = bed(Rule::Place(0), Rule::Place(1));
    door.store().spend_ids(0);
    placed(&door, &request(b"r-1", 7));
    assert_eq!(asked.load(Ordering::Relaxed), 2);
    assert_eq!(kinds(&door), [0, 1, 0, 0]);
}

/// A sampled decision's record holds what its shadow said and the
/// shadow's first target: the worker the shadow's rule takes where it is
/// the live one.
#[test]
fn a_sampled_record_holds_the_shadows_kind_and_first_target() {
    let store = fleet(2);
    let rules = (Rule::Place(0), Rule::Place(1));
    shadowing(&store, (2, "random"), rules, |document| {
        document.explain.sample = 1;
    });
    let door = Door::new(store);
    placed(&door, &request(b"kept", 7));
    let record = door.explain(GATEWAY, b"kept");
    let record = record.expect("the sampled decision");
    let (turned, _) = bed(Rule::Place(1), Rule::Place(0));
    let (_, shadows, _) = placed(&turned, &request(b"kept", 7));
    assert_eq!(record.shadow, Some(ShadowKind::OtherTarget));
    assert_eq!(record.shadow_target, Some(shadows));
}

/// A shadow that panics costs its request nothing: the answer and its
/// lease are the live decision's, the fault is counted with the door's,
/// no shadow is counted, and the next request decides.
#[test]
fn a_shadow_that_panics_leaves_the_live_answer_and_counts_a_fault() {
    let (door, _) = bed(Rule::Place(0), Rule::Panics);
    let (_, first, _) = placed(&door, &request(b"r-1", 7));
    assert_eq!(counted(door.store()), [(first, 1)]);
    assert_eq!(door.faults(), 1);
    assert_eq!(kinds(&door), [0, 0, 0, 0]);
    assert_eq!(placed(&door, &request(b"r-2", 7)).1, first);
    assert_eq!(counted(door.store()), [(first, 2)]);
    assert_eq!(door.faults(), 2);
}

/// A twin of the live profile agrees with it on every decision of a run.
/// Both read the counts as they stand ahead of the decision's own lease.
/// Asked behind the commit, the twin would read the live pick one request
/// fuller and take another worker at each tie.
#[test]
fn a_twin_of_the_live_profile_agrees_at_every_tie_of_counts() {
    let store = fleet(2);
    let mut document = PolicySpec::preset("least_load").expect("a preset of this build");
    document.generation = 2;
    let live = document.profiles.values().next();
    let twin = live.expect("the preset's profile").clone();
    document.profiles.insert("twin".into(), twin);
    document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    store.apply(spec).expect("a later generation");
    let door = Door::new(store);
    for id in [b"r-1", b"r-2", b"r-3", b"r-4", b"r-5", b"r-6"] {
        placed(&door, &request(id, 7));
    }
    assert_eq!(kinds(&door), [6, 0, 0, 0]);
}

/// A fresh key under a sessions rule is the shadow profile's to place
/// too. It reads the sessions as the live decision found them, with no
/// pin, and takes its own row: asked at the plain entry it would read the
/// claim the live decision just made, and agree. A key pinned before the
/// decision keeps both on its worker.
#[test]
fn a_fresh_keys_shadow_places_it_and_a_pinned_keys_shadow_agrees() {
    let store = fleet(2);
    let rules = (Rule::Place(0), Rule::Place(1));
    shadowing(&store, (2, "manual"), rules, |document| {
        let rule = document.bindings[0].sessions.as_mut();
        rule.expect("the preset's rule").assignment = Assignment::Delegate;
    });
    let door = Door::new(store);
    let (lease, owner, verdict) = placed(&door, &keyed(b"r-1", 7, 1));
    assert_eq!(verdict, SessionVerdict::Claimed(owner));
    assert_eq!(kinds(&door), [0, 1, 0, 0]);
    // The dispatch pins the key, and its next request stays: the shadow
    // reads that pin as the live decision does.
    assert_eq!(report(&door, lease, dispatched()), 1);
    let (_, worker, verdict) = placed(&door, &keyed(b"r-2", 7, 2));
    assert_eq!((worker, verdict), (owner, SessionVerdict::Pinned));
    assert_eq!(kinds(&door), [1, 1, 0, 0]);
}

/// A door under the `manual` preset's rule with the key's worker left to
/// the profile, where both profiles take the first row kept. The first
/// request of key 7 is decided at the first seat, and the caller is put
/// at the second: the claim stands, and its lease is on a file no decider
/// works at afterwards. Returns the door, that lease and the key's worker.
fn claimed_at_another_seat() -> (Door, LeaseId, WorkerId) {
    let store = fleet(2);
    let rules = (Rule::Place(0), Rule::Place(0));
    shadowing(&store, (2, "manual"), rules, |document| {
        let rule = document.bindings[0].sessions.as_mut();
        rule.expect("the preset's rule").assignment = Assignment::Delegate;
    });
    let door = Door::new(store);
    let (earlier, owner, _) = thread::scope(|scope| {
        let first = scope.spawn(|| {
            seated(0);
            placed(&door, &keyed(b"r-0", 7, 1))
        });
        first.join().expect("the first decider")
    });
    seated(1);
    (door, earlier, owner)
}

/// Decides the key's next request at this seat and returns its worker.
/// Between its first live decision and that decision's shadow another
/// decider sends the key's earlier request out on its alternate, and the
/// key's pin moves there.
fn placed_while_the_pin_moves(door: &Door, earlier: LeaseId) -> WorkerId {
    let (at_gap, gap) = mpsc::channel();
    let (moved, resume) = mpsc::channel();
    thread::scope(|scope| {
        scope.spawn(move || {
            gap.recv().expect("the live decision was made");
            let alternate = Dispatch {
                target_index: 1,
                side: Side::Single,
                since_decision_micros: 250,
            };
            door.report(
                GATEWAY,
                earlier,
                Said::Dispatched(alternate, Instant::now()),
            );
            moved.send(()).expect("the decider waits at its shadow");
        });
        ON_SHADOW.set(Some(Box::new(move || {
            at_gap.send(()).expect("the other decider waits");
            resume.recv().expect("the pin moved");
        })));
        placed(door, &keyed(b"r-1", 7, 2)).1
    })
}

/// Another decider commits the key between a live decision and its
/// shadow: an earlier request of the key goes out on its alternate, and
/// the key's pin moves there. The shadow still decides from the pins its
/// live decision read, and agrees with it. Asked at the plain entry it
/// would read the pin where it moved to, and take the alternate.
#[test]
fn a_shadow_decides_from_the_pins_its_live_decision_read() {
    let (door, earlier, owner) = claimed_at_another_seat();
    assert_eq!(placed_while_the_pin_moves(&door, earlier), owner);
    assert_eq!(kinds(&door), [2, 0, 0, 0]);
    // The pin did move: the key's next request takes the alternate.
    assert_ne!(placed(&door, &keyed(b"r-2", 7, 3)).1, owner);
}

/// A decision the door makes again is shadowed from its own reading. The
/// store refuses the first lease, the key's pin has moved by then, and
/// the decision made again follows the pin. Its shadow starts from that
/// second reading and agrees. From the first decision's reading it would
/// keep the key's old worker, and be counted as another target.
#[test]
fn a_decision_made_again_is_shadowed_from_its_own_reading() {
    let (door, earlier, owner) = claimed_at_another_seat();
    // The spent ids are named for the first seat's file, where the
    // earlier lease is: its report is still applied at once.
    door.store().spend_ids(0);
    assert_ne!(placed_while_the_pin_moves(&door, earlier), owner);
    assert_eq!(kinds(&door), [2, 0, 0, 0]);
}

/// Rows are named for the live profile alone, by its sample, its
/// warm-up ruling and the record of sent prefixes. A shadow profile that
/// would be named other rows is not asked, and is counted as skipped: one
/// of another sample, one of another ruling, one that reads the record
/// where the live profile reads the index, and one that reads cached
/// blocks behind a live profile that is named no row. A shadow of the
/// live profile's own sample and ruling is asked over the same rows.
#[test]
fn a_shadow_that_would_be_named_other_rows_is_skipped() {
    let shadowed = |live: &str, turn: &dyn Fn(&mut ProfileSpec)| {
        let store = fleet(2);
        let mut document = PolicySpec::preset(live).expect("a preset of this build");
        document.generation = 2;
        let aware = PolicySpec::preset("cache_aware").expect("a preset of this build");
        let twin = aware.profiles.into_values().next();
        let mut twin = twin.expect("the preset's profile");
        turn(&mut twin);
        document.profiles.insert("twin".into(), twin);
        document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        store.apply(spec).expect("a later generation");
        let door = Door::new(store);
        placed(&door, &request(b"r-1", 7));
        kinds(&door)
    };
    let sample = |sample: u32| {
        move |twin: &mut ProfileSpec| twin.pick = PickStage::Best(BestParams { sample }).into()
    };
    let ruling = |twin: &mut ProfileSpec| {
        let slice = WarmupSliceParams {
            weight: 8.0,
            secs: Some(120),
            ..WarmupSliceParams::default()
        };
        twin.score[0] = ScoreStage::WarmupSlice(slice).into();
    };
    let record = |twin: &mut ProfileSpec| {
        let prefix = PrefixAffinityParams {
            weight: 4.0,
            source: Some(AffinitySource::Predictor),
            ..PrefixAffinityParams::default()
        };
        twin.score[1] = ScoreStage::PrefixAffinity(prefix).into();
    };
    assert_eq!(shadowed("cache_aware", &sample(8)), [0, 0, 0, 1]);
    assert_eq!(shadowed("cache_aware", &ruling), [0, 0, 0, 1]);
    assert_eq!(shadowed("cache_aware", &record), [0, 0, 0, 1]);
    assert_eq!(shadowed("random", &|_| {}), [0, 0, 0, 1]);
    let same = shadowed("cache_aware", &sample(16));
    assert_eq!((same.iter().sum::<u64>(), same[3]), (1, 0), "{same:?}");
}

/// A profile that reads no column is named no row when it decides live.
/// As a shadow behind a live profile that is named rows it is asked
/// without them: the names would turn a sampling picker's pick to a draw.
#[test]
fn a_shadow_that_reads_no_column_is_asked_without_the_live_rows() {
    let store = fleet(2);
    let named = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&named);
    let mut catalog = Catalog::builtin();
    let added = catalog.picker("sees-names", move || -> Box<dyn Picker> {
        Box::new(SeesNames(Arc::clone(&seen)))
    });
    added.expect("a free name");
    let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
    document.generation = 2;
    let plain = PolicySpec::preset("random").expect("a preset of this build");
    let twin = plain.profiles.into_values().next();
    let mut twin = twin.expect("the preset's profile");
    twin.pick = "sees-names".into();
    document.profiles.insert("twin".into(), twin);
    document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
    let applied = store.apply(spec.expect("the document validates"));
    applied.expect("a later generation");
    let door = Door::new(store);
    placed(&door, &request(b"r-1", 7));
    let kinds = kinds(&door);
    assert_eq!((kinds.iter().sum::<u64>(), kinds[3]), (1, 0), "{kinds:?}");
    assert_eq!(named.load(Ordering::Relaxed), 0);
}

/// A request whose binding names no shadow takes the plain order under a
/// generation that shadows another binding. A pair beside a shadowed
/// single is decided by either entry as it is by the other, with what was
/// placed cleared, and no shadow is counted. A shadowed single that comes
/// by the pair's entry is shadowed as by the plain one, and what was
/// placed is cleared.
#[test]
fn a_pair_beside_a_shadowed_single_is_ordered_as_before() -> Result<(), Box<dyn Error>> {
    let (donor, _) = push_fleet()?;
    let now = Instant::now();
    let store = pair_fleet(1, 1)?;
    let mut document = PolicySpec::pair_presets("least_load", "least_load")?;
    document.generation = 2;
    for profile in document.profiles.values_mut() {
        profile.filter.clear();
    }
    let single = PolicySpec::preset("random")?;
    let live = single.profiles.values().next();
    let live = live.ok_or("the preset's profile")?;
    document.profiles.insert("single".into(), live.clone());
    document.profiles.insert("twin".into(), live.clone());
    let mut binding = single.bindings[0].clone();
    "another-model".clone_into(&mut binding.matcher.model);
    binding.profile = Some("single".into());
    binding.shadow = Some(ShadowSpec::Profile("twin".into()));
    document.bindings.insert(0, binding);
    store.apply(AppliedSpec::validate(&document)?)?;
    let door = Door::new(store);
    let asked = pair_request(b"pair");
    let ordinary = door.decide(GATEWAY, &asked, now).expect("a decision");
    release(&door, &ordinary);
    let mut placed = place_one(&donor, now).expect("a decision");
    let answer = door
        .decide_encode_pair(GATEWAY, &asked, items(), &mut placed, now)
        .expect("a decision");
    assert_eq!(answer.decision, ordinary.decision);
    assert!(answer.pair() && !placed.one() && placed.encoders().is_empty());
    assert_eq!([kinds_at(&door, 0), kinds_at(&door, 1)], [[0; 4]; 2]);

    let (door, shadow_asked) = bed(Rule::Place(0), Rule::Place(1));
    let mut placed = place_one(&donor, now).expect("a decision");
    let answer = door
        .decide_encode_pair(GATEWAY, &request(b"r-1", 7), items(), &mut placed, now)
        .expect("a decision");
    assert!(!answer.pair() && !placed.one() && placed.encoders().is_empty());
    assert_eq!(kinds(&door), [0, 1, 0, 0]);
    assert_eq!(shadow_asked.load(Ordering::Relaxed), 1);
    Ok(())
}
