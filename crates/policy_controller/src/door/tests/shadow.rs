//! The shadow at the door. A binding's shadow profile is asked beside a
//! live decision, ahead of the decision's commit. What it said is counted
//! by the binding's position and kept with a sampled record. Nothing of it
//! changes, fails or books the live answer.

use std::{
    collections::BTreeMap,
    error::Error,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

use kv_index::chain_prefix_hash;

use super::{
    booked, counted, dispatched, encode_document, fleet, items, keyed, later_request, pair_counts,
    pair_dispatch, pair_fleet, pair_request, place_one, placed, push_fleet, release, report,
    request, seated, GATEWAY, MODEL,
};
use crate::{
    builtins::target_of,
    catalog::Catalog,
    decision::{Decision, ShadowKind, Targets},
    door::{Decided, Door, ON_SHADOW, SHADOW_FLOOR},
    framework::{EncodePlaced, PairSide, Picker, TieBand},
    lease::{LeaseId, Lifecycle, Side},
    sessions::SessionVerdict,
    spec::{
        AffinitySource, AppliedSpec, BestParams, PickStage, PolicySpec, PrefixAffinityParams,
        ProfileSpec, ScoreStage, ShadowSpec, WarmupSliceParams,
    },
    store::{
        BatchFacts, BatchOutcome, ContentHash, Dispatch, KvEvent, KvEventSink, LeasesStore,
        LoadSink, LoadSource, MemoryStore, Said, SequenceHash, Signals, StoredBlock, WorkerFacts,
        WorkerStatus, WorkersStore,
    },
    view::{Candidate, LoadRow, Phase, RequestFacts, Role, WorkerId},
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

/// A request whose demand leaves none of a pool that has rows is shed by
/// the door under the demand's own stage, as it is where no shadow is
/// bound. No profile decided it: the shadow's picker is not asked, and no
/// shadow is counted.
#[test]
fn a_pool_a_demand_emptied_is_shed_under_the_demand_and_is_not_shadowed() {
    use crate::{decision::ShedReason, Demands, Unmet};

    let (door, asked) = bed(Rule::Place(0), Rule::Place(1));
    let mut order = request(b"r-1", 7);
    order.constraints.demands = Demands {
        vision: true,
        ..Demands::default()
    };
    let decided = door.decide(GATEWAY, &order, Instant::now());
    let shed = match decided.expect("a decision").decision {
        Decision::Shed(shed) => Ok((shed.reason, shed.emptied_by)),
        decision => Err(decision.pick().is_some()),
    };
    assert_eq!(shed, Ok((ShedReason::NoEndpoints, Unmet::Vision.stage())));
    assert_eq!(asked.load(Ordering::Relaxed), 0);
    assert_eq!(kinds(&door), [0, 0, 0, 0]);
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

/// A pool keeps one ruling of its warming rows and one record of sent
/// prefixes: the live profile's. A shadow profile of another ruling, or
/// one that reads the record alone where the live profile reads cache
/// events beside it, is not asked, and is counted as skipped. A shadow
/// of another sample, and
/// one that reads cached blocks behind a live profile that is named no
/// row, are asked over rows named for them alone. A shadow of the live
/// profile's own sample and ruling is asked over the same rows.
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
    assert_eq!(shadowed("cache_aware", &ruling), [0, 0, 0, 1]);
    assert_eq!(shadowed("cache_aware", &record), [0, 0, 0, 1]);
    let asked = [
        shadowed("cache_aware", &sample(8)),
        shadowed("random", &|_| {}),
        shadowed("cache_aware", &sample(16)),
    ];
    for kinds in asked {
        assert_eq!((kinds.iter().sum::<u64>(), kinds[3]), (1, 0), "{kinds:?}");
    }
}

/// A store of one decider under `document` whose members publish their
/// cache events. `holders` of them hold the chain of blocks 1 to 1,100,
/// and `bare` hold none: under a ruling those are the pool's warming rows
/// at the next window. The members' ids, the holders first.
fn cached(document: &PolicySpec, holders: usize, bare: usize) -> (Arc<MemoryStore>, Vec<WorkerId>) {
    let store = MemoryStore::new().with_deciders(1);
    let spec = AppliedSpec::validate(document).expect("the document validates");
    store.apply(spec).expect("above the unbound start");
    let filled: Vec<u64> = (1..=1_100).collect();
    let mut ids = Vec::new();
    for at in 0..holders + bare {
        let key = format!("http://c{at}:8000");
        let facts = WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            role: Role::Single,
            models: vec![MODEL.to_owned()],
            dp_size: 1,
            max_running: Some(4_096),
            signals: Signals {
                events: true,
                ..Signals::default()
            },
            ..WorkerFacts::default()
        };
        let id = store.upsert(facts).id();
        let row = WorkersStore::get(&store, id).expect("the row");
        if row.status != WorkerStatus::Ready {
            let ready = store.set_status(id, WorkerStatus::Ready, row.revision);
            ready.expect("the row stands");
        }
        store.publish_now();
        if at < holders {
            let mut blocks: Vec<StoredBlock> = Vec::with_capacity(filled.len());
            for &content in &filled {
                let seq_hash = match blocks.last() {
                    Some(parent) => chain_prefix_hash(parent.seq_hash, ContentHash(content)),
                    None => SequenceHash(content),
                };
                blocks.push(StoredBlock {
                    seq_hash,
                    content_hash: ContentHash(content),
                });
            }
            let events = [KvEvent::Stored {
                parent: None,
                blocks,
            }];
            let batch = BatchFacts {
                sequence: 1,
                ..BatchFacts::default()
            };
            assert_eq!(store.admit(id, 0, batch, &events), BatchOutcome::Applied);
        }
        ids.push(id);
    }
    let window = Instant::now() + Duration::from_secs(1);
    store.record(ids[0], 0, LoadRow::default(), window, LoadSource::Stream);
    (Arc::new(store), ids)
}

/// The `least_load` preset with its decisions sampled and, where
/// `shadowed`, the `cache_aware` preset's profile as its binding's
/// shadow: a live profile that is named no row, and a shadow that is
/// named rows of its own.
fn own_rows(shadowed: bool) -> PolicySpec {
    let mut document = PolicySpec::preset("least_load").expect("a preset of this build");
    document.explain.sample = 1;
    if shadowed {
        let aware = PolicySpec::preset("cache_aware").expect("a preset of this build");
        let twin = aware.profiles.into_values().next();
        let twin = twin.expect("the preset's profile");
        document.profiles.insert("twin".into(), twin);
        document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    }
    document
}

/// A shadow profile that reads cached blocks, behind a live profile that
/// is named no row, is asked over rows the door names for it alone: the
/// request's holder, and the pool's warming rows by the shadow profile's
/// own ruling. It takes the worker the same profile takes live over the
/// same state. The live answer is the one given with no shadow.
#[test]
fn a_shadow_named_rows_of_its_own_is_asked_over_them() {
    let hashes = [1_u64];
    let mut asked = request(b"own", 7);
    asked.facts.block_hashes = &hashes;
    let place = |ids: &[WorkerId], id: WorkerId| ids.iter().position(|&at| at == id);

    let aware = PolicySpec::preset("cache_aware").expect("a preset of this build");
    let (store, ids) = cached(&aware, 1, 2);
    let as_live = place(&ids, placed(&Door::new(store), &asked).1);
    let (store, ids) = cached(&own_rows(false), 1, 2);
    let unshadowed = place(&ids, placed(&Door::new(store), &asked).1);

    let (store, ids) = cached(&own_rows(true), 1, 2);
    let door = Door::new(store);
    let (_, live, _) = placed(&door, &asked);
    assert_eq!(place(&ids, live), unshadowed);
    assert_eq!(kinds(&door)[3], 0, "the shadow is asked");
    let record = door.explain(GATEWAY, b"own");
    let record = record.expect("the sampled decision");
    assert_eq!(record.shadow_target.and_then(|id| place(&ids, id)), as_live);
    let named = door.own_named[0].named.lock();
    assert_eq!((named.holders.len(), named.warming.len()), (1, 2));
}

/// A shadow of another sample than the live profile's is named by its own
/// cap: of twelve holders the live profile is named every one under its
/// sixteen, and the shadow eight.
#[test]
fn a_shadow_of_another_sample_is_named_by_its_own_cap() {
    let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
    let live = document.profiles.values().next();
    let mut twin = live.expect("the preset's profile").clone();
    twin.pick = PickStage::Best(BestParams { sample: 8 }).into();
    document.profiles.insert("twin".into(), twin);
    document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    let (store, _) = cached(&document, 12, 0);
    let door = Door::new(store);
    let hashes = [1_u64];
    let mut asked = request(b"cap", 7);
    asked.facts.block_hashes = &hashes;
    placed(&door, &asked);
    assert_eq!(kinds(&door)[3], 0, "the shadow is asked");
    let named = (
        door.seats[0].slot.lock().buffers.named.holders.len(),
        door.own_named[0].named.lock().holders.len(),
    );
    assert_eq!(named, (12, 8));
}

/// The floor is asked ahead of the shadow's own rows: a request under it
/// names none and skips its shadow, and one at it is named its holder.
#[test]
fn a_budget_under_the_floor_names_no_rows_for_the_shadow() {
    let (store, _) = cached(&own_rows(true), 1, 2);
    let door = Door::new(store);
    let floor = u32::try_from(SHADOW_FLOOR.as_micros()).expect("a floor in microseconds");
    let hashes = [1_u64];
    let budgeted = |id: &'static [u8], micros: u32| {
        let mut stated = request(id, 7);
        stated.facts.block_hashes = &hashes;
        stated.facts.budget_micros = micros;
        stated
    };
    let named = |door: &Door| door.own_named[0].named.lock().holders.len();
    placed(&door, &budgeted(b"short", floor - 1));
    assert_eq!((kinds(&door), named(&door)), ([0, 0, 0, 1], 0));
    placed(&door, &budgeted(b"held", floor));
    assert_eq!((kinds(&door)[3], named(&door)), (1, 1));
}

/// A shadow's need of the affinity column fills none for a live decision
/// whose rows it does not read. Behind least load, a shadow that is named
/// rows of its own leaves the live decision no column, whatever it reads
/// a request's holders from. A live profile that reads the column itself
/// is filled it.
#[test]
fn a_column_is_filled_for_the_live_profile_and_for_no_shadow_alone() {
    let filled = |document: &PolicySpec| {
        let (store, _) = cached(document, 1, 2);
        let door = Door::new(store);
        let hashes = [1_u64];
        let mut asked = request(b"col", 7);
        asked.facts.block_hashes = &hashes;
        placed(&door, &asked);
        let skipped = kinds(&door)[3];
        let slot = door.seats[0].slot.lock();
        (slot.buffers.view.affinity.len(), skipped)
    };
    assert_eq!(filled(&own_rows(true)), (0, 0), "rows of its own");
    let mut alone = own_rows(true);
    let twin = alone.profiles.get_mut("twin").expect("the shadow profile");
    let prefix = PrefixAffinityParams {
        weight: 4.0,
        source: Some(AffinitySource::Predictor),
        ..PrefixAffinityParams::default()
    };
    twin.score[1] = ScoreStage::PrefixAffinity(prefix).into();
    assert_eq!(
        filled(&alone),
        (0, 0),
        "by the record of sent prefixes alone"
    );
    // A live profile of two warm-up rulings is named no row and reads
    // the column whole, with the cache-aware preset's profile behind it.
    let mut whole = PolicySpec::preset("cache_aware").expect("a preset of this build");
    whole.explain.sample = 1;
    let live = whole.profiles.values_mut().next();
    let live = live.expect("the preset's profile");
    let twin = live.clone();
    let slice = WarmupSliceParams {
        weight: 8.0,
        secs: Some(120),
        ..WarmupSliceParams::default()
    };
    live.score.push(ScoreStage::WarmupSlice(slice).into());
    whole.profiles.insert("twin".into(), twin);
    whole.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    assert_eq!(filled(&whole), (3, 0), "the live profile's own column");
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

/// The names of a case's pickers on a pair's two sides: the live one's,
/// then its shadow's.
const SIDE_PICKERS: [(&str, &str); 2] = [
    ("takes-prefill", "takes-prefill-next"),
    ("takes-decode", "takes-decode-next"),
];

/// Applies `generation` of the stock pair document with a case's picker
/// on each side: `live[0]` decides the prefill and `live[1]` the decode.
/// Each side `next` names is shadowed by a profile of the same stages
/// under that side's own picker. `edit` has the document before it is
/// applied. The counts are of the decisions that asked each side's
/// shadow picker, the prefill's then the decode's.
fn pair_shadowing(
    store: &MemoryStore,
    generation: u64,
    live: [Rule; 2],
    next: [Option<Rule>; 2],
    edit: impl FnOnce(&mut PolicySpec),
) -> [Arc<AtomicUsize>; 2] {
    let asked = [Arc::default(), Arc::default()];
    let picker = |name: &'static str, rule: Rule, asked: &Arc<AtomicUsize>| {
        let asked = Arc::clone(asked);
        move || -> Box<dyn Picker> {
            let asked = Arc::clone(&asked);
            Box::new(Takes { name, rule, asked })
        }
    };
    let mut catalog = Catalog::builtin();
    let document = PolicySpec::pair_presets("least_load", "least_load");
    let mut document = document.expect("the stock presets");
    document.generation = generation;
    let mut sides = BTreeMap::new();
    for (at, side) in ["prefill", "decode"].into_iter().enumerate() {
        let (own, shadow) = SIDE_PICKERS[at];
        let added = catalog.picker(own, picker(own, live[at], &Arc::default()));
        added.expect("a free name");
        let profile = document.profiles.get_mut(side);
        let profile = profile.expect("the side's profile");
        profile.filter.clear();
        profile.pick = own.into();
        let Some(rule) = next[at] else { continue };
        let added = catalog.picker(shadow, picker(shadow, rule, &asked[at]));
        added.expect("a free name");
        let mut twin = profile.clone();
        twin.pick = shadow.into();
        let name = format!("{side}-next");
        document.profiles.insert(name.clone(), twin);
        sides.insert(side.to_owned(), name);
    }
    if !sides.is_empty() {
        document.bindings[0].shadow = Some(ShadowSpec::Sides(sides));
    }
    edit(&mut document);
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
    let applied = store.apply(spec.expect("the document validates"));
    applied.expect("a later generation");
    asked
}

/// A door over two prefills and two decodes that pair each with each,
/// deciding a pair's sides by `live` and shadowing the sides `next`
/// names, with the counts of the decisions that asked each shadow.
fn pair_bed(live: [Rule; 2], next: [Option<Rule>; 2]) -> (Door, [Arc<AtomicUsize>; 2]) {
    let store = pair_fleet(2, 2).expect("the pair fleet");
    let asked = pair_shadowing(&store, 2, live, next, |_| {});
    (Door::new(store), asked)
}

/// The door's counts of the shadows of `side` of the first binding's
/// pairs: agreed, another target, another outcome, skipped.
fn side_kinds(door: &Door, side: PairSide) -> [u64; 4] {
    use ShadowKind::{Agreed, OtherOutcome, OtherTarget, Skipped};
    [Agreed, OtherTarget, OtherOutcome, Skipped].map(|kind| door.shadowed_side(0, side, kind))
}

/// The counts of both sides, the prefill's then the decode's.
fn both_kinds(door: &Door) -> [[u64; 4]; 2] {
    PairSide::BOTH.map(|side| side_kinds(door, side))
}

/// A pair decided at `door`, and the prefill and the decode it leads
/// with.
fn paired(door: &Door, id: &'static [u8], now: Instant) -> (Decided, (WorkerId, WorkerId)) {
    let decided = door.decide(GATEWAY, &pair_request(id), now);
    let decided = decided.expect("a decision");
    let pair = pair_of(&decided);
    (decided, pair)
}

/// The prefill and the decode a pair's decision names first.
fn pair_of(decided: &Decided) -> (WorkerId, WorkerId) {
    let pick = decided.decision.pick();
    let first = pick.and_then(|pick| pick.targets.first());
    let first = *first.expect("a pair");
    let decode = first.pair.expect("the pair's decode").decode_id;
    (first.id, decode)
}

/// How many decisions asked each side's shadow.
fn asks(asked: &[Arc<AtomicUsize>; 2]) -> [usize; 2] {
    asked.each_ref().map(|asked| asked.load(Ordering::Relaxed))
}

/// A pair's binding names a shadow a side. With the decode side alone
/// named, the decode is compared and the prefill is not: one count
/// moves, the decode side's. The pair is the live profiles' own, and a
/// pair counts nothing with the singles' shadows.
#[test]
fn a_pairs_decode_shadow_alone_is_compared_and_counted_by_its_side() {
    let now = Instant::now();
    let (plain, _) = pair_bed([Rule::Place(0); 2], [None; 2]);
    let (_, live) = paired(&plain, b"pair", now);
    assert_eq!(both_kinds(&plain), [[0; 4]; 2]);
    let (door, asked) = pair_bed([Rule::Place(0); 2], [None, Some(Rule::Place(1))]);
    let (decided, pair) = paired(&door, b"pair", now);
    assert!(decided.pair());
    assert_eq!(pair, live, "the pair is the live profiles' own");
    assert_eq!(both_kinds(&door), [[0; 4], [0, 1, 0, 0]]);
    assert_eq!(kinds(&door), [0; 4]);
    assert_eq!(asks(&asked), [0, 1]);
}

/// With both sides named each is compared with its own: two counts, one
/// a side. A shadow under its live side's own rule agrees with it.
#[test]
fn a_pair_with_both_sides_named_counts_one_a_side() {
    let next = [Some(Rule::Place(0)), Some(Rule::Place(1))];
    let (door, asked) = pair_bed([Rule::Place(0); 2], next);
    paired(&door, b"pair", Instant::now());
    assert_eq!(both_kinds(&door), [[1, 0, 0, 0], [0, 1, 0, 0]]);
    assert_eq!(asks(&asked), [1, 1]);
    // A side's shadow that declines is another outcome.
    let next = [Some(Rule::Nobody), Some(Rule::Place(0))];
    let (door, _) = pair_bed([Rule::Place(0); 2], next);
    paired(&door, b"pair", Instant::now());
    assert_eq!(both_kinds(&door), [[0, 0, 1, 0], [1, 0, 0, 0]]);
}

/// A shadowed pair books and leases what the live pair alone does. The
/// store's counts and leases behind it are those of the same pair under
/// a document that names no shadow, and the workers the shadows took
/// hold nothing.
#[test]
fn a_shadowed_pair_leaves_the_counts_and_leases_of_the_live_pair() {
    let now = Instant::now();
    let (turned, _) = pair_bed([Rule::Place(1); 2], [None; 2]);
    let (_, others) = paired(&turned, b"pair", now);
    let left = |next: [Option<Rule>; 2]| {
        let (door, _) = pair_bed([Rule::Place(0); 2], next);
        let (_, live) = paired(&door, b"pair", now);
        assert_ne!((live.0, live.1), others, "the shadows take other workers");
        let workers = [live.0, live.1, others.0, others.1];
        let store = door.store();
        let leases = LeasesStore::live(&**store, GATEWAY);
        (live, pair_counts(store, workers), leases)
    };
    let plain = left([None; 2]);
    assert_eq!(plain.1, [1, 1, 0, 0]);
    assert_eq!(left([Some(Rule::Place(1)); 2]), plain);
}

/// A pair with less than the floor of its budget left keeps it for its
/// answer: no side's shadow is asked, and each named side is counted as
/// skipped.
#[test]
fn a_pairs_budget_under_the_floor_skips_each_named_side() {
    let (door, asked) = pair_bed([Rule::Place(0); 2], [Some(Rule::Place(1)); 2]);
    let floor = u32::try_from(SHADOW_FLOOR.as_micros()).expect("a floor in microseconds");
    let mut short = pair_request(b"short");
    short.facts.budget_micros = floor - 1;
    let decided = door.decide(GATEWAY, &short, Instant::now());
    assert!(decided.expect("a decision").pair());
    assert_eq!(both_kinds(&door), [[0, 0, 0, 1]; 2]);
    assert_eq!(asks(&asked), [0, 0]);
    let mut held = pair_request(b"held");
    held.facts.budget_micros = floor;
    let decided = door.decide(GATEWAY, &held, Instant::now());
    assert!(decided.expect("a decision").pair());
    assert_eq!(both_kinds(&door), [[0, 1, 0, 1]; 2]);
}

/// A side's shadow that panics costs its pair nothing: the pair and its
/// leases are the live profiles', the fault is counted with the door's,
/// the side behind it is not asked, and the next pair decides.
#[test]
fn a_pairs_shadow_that_panics_leaves_the_pair_and_counts_a_fault() {
    let now = Instant::now();
    let next = [Some(Rule::Panics), Some(Rule::Place(1))];
    let (door, asked) = pair_bed([Rule::Place(0); 2], next);
    let (decided, live) = paired(&door, b"pair", now);
    assert!(decided.pair() && decided.lease.is_some());
    assert_eq!(pair_counts(door.store(), [live.0, live.1]), [1, 1]);
    assert_eq!(door.faults(), 1);
    assert_eq!(both_kinds(&door), [[0; 4]; 2]);
    assert_eq!(asks(&asked), [1, 0], "the side behind a fault is not asked");
    assert_eq!(paired(&door, b"next", now).1, live);
    assert_eq!(door.faults(), 2);
}

/// A sampled pair's record holds what each side's shadow said and that
/// shadow's first target: the prefill's in the record's first place, the
/// decode's beside it.
#[test]
fn a_sampled_pairs_record_holds_each_sides_kind_and_first_target() {
    let store = pair_fleet(2, 2).expect("the pair fleet");
    let next = [Some(Rule::Place(1)), Some(Rule::Place(0))];
    pair_shadowing(&store, 2, [Rule::Place(0); 2], next, |document| {
        document.explain.sample = 1;
    });
    let door = Door::new(store);
    let (_, (prefill, decode)) = paired(&door, b"kept", Instant::now());
    let record = door.explain(GATEWAY, b"kept");
    let record = record.expect("the sampled pair");
    assert_eq!(record.shadow, Some(ShadowKind::OtherTarget));
    assert!(record.shadow_target.is_some_and(|other| other != prefill));
    assert_ne!(record.shadow_target, Some(decode), "the prefill side's own");
    assert_eq!(record.decode_shadow, Some(ShadowKind::Agreed));
    assert_eq!(record.decode_shadow_target, Some(decode));
}

/// A pair's shadows are counted behind the pair's commit. A pair whose
/// lease the store refused is ordered again and each side's shadow is
/// asked again: the one answer counts one shadow a side.
#[test]
fn a_pair_ordered_twice_counts_each_sides_shadow_once() {
    let (door, asked) = pair_bed([Rule::Place(0); 2], [Some(Rule::Place(0)); 2]);
    seated(0);
    door.store().spend_ids(0);
    let (decided, _) = paired(&door, b"retry", Instant::now());
    assert!(decided.lease.is_some());
    assert_eq!(asks(&asked), [2, 2]);
    assert_eq!(both_kinds(&door), [[1, 0, 0, 0]; 2]);
}

/// A pair of a plan that takes no encode items is decided by the encode
/// entry as by the plain one: each named side's shadow is asked and
/// counted there too, the pair is the same, and what was placed is
/// cleared.
#[test]
fn a_pair_by_the_encode_entry_asks_its_sides_shadows() -> Result<(), Box<dyn Error>> {
    let (donor, _) = push_fleet()?;
    let now = Instant::now();
    let next = [None, Some(Rule::Place(1))];
    let (plain, _) = pair_bed([Rule::Place(0); 2], next);
    let (_, live) = paired(&plain, b"pair", now);
    let (door, asked) = pair_bed([Rule::Place(0); 2], next);
    let mut placed = place_one(&donor, now).expect("a decision");
    let request = pair_request(b"pair");
    let answer = door.decide_encode_pair(GATEWAY, &request, items(), &mut placed, now);
    let answer = answer.expect("a decision");
    assert!(answer.pair() && !placed.one() && placed.encoders().is_empty());
    assert_eq!(pair_of(&answer), live, "the pair the plain entry decides");
    assert_eq!(both_kinds(&door), [[0; 4], [0, 1, 0, 0]]);
    assert_eq!(both_kinds(&door), both_kinds(&plain));
    assert_eq!(asks(&asked), [0, 1]);
    Ok(())
}

/// An `epd` pair is decided with its encode items, and asks no shadow of
/// its sides: under a binding that names one, the encode entry places
/// the request's items as it does under a binding that names none, and
/// no side's count moves.
#[test]
fn a_pair_decided_with_its_encode_items_asks_no_shadow() -> Result<(), Box<dyn Error>> {
    let (door, encoder) = push_fleet()?;
    let mut document = encode_document()?;
    document.generation = 2;
    let twin = document.profiles.get("decode").cloned();
    let twin = twin.ok_or("the decode side's profile")?;
    document.profiles.insert("decode-next".into(), twin);
    let sides = BTreeMap::from([("decode".to_owned(), "decode-next".to_owned())]);
    document.bindings[0].shadow = Some(ShadowSpec::Sides(sides));
    door.store().apply(AppliedSpec::validate(&document)?)?;
    let mut placed = EncodePlaced::new();
    let request = pair_request(b"pair");
    let now = Instant::now();
    let answer = door.decide_encode_pair(GATEWAY, &request, items(), &mut placed, now);
    assert!(answer.expect("a decision").pair());
    let encoders: Vec<_> = placed.encoders().iter().map(|placed| placed.id).collect();
    assert_eq!(encoders, [encoder, encoder]);
    assert_eq!(both_kinds(&door), [[0; 4]; 2]);
    Ok(())
}

/// A side's shadow profile that would be handed another column than its
/// live side's pool carries is not asked. The pool is read by the live
/// profile's ruling and keeps the live profile's record: a shadow that
/// reads cached blocks, beside a side that reads none, is counted as
/// skipped. The other side's shadow is asked.
#[test]
fn a_pair_sides_shadow_of_another_column_is_skipped_and_counted() {
    let store = pair_fleet(2, 2).expect("the pair fleet");
    let next = [Some(Rule::Place(1)); 2];
    let asked = pair_shadowing(&store, 2, [Rule::Place(0); 2], next, |document| {
        let aware = PolicySpec::preset("cache_aware").expect("a preset of this build");
        let aware = aware.profiles.into_values().next();
        let shadow = document.profiles.get_mut("decode-next");
        shadow.expect("the decode's shadow").score = aware.expect("the preset's profile").score;
    });
    let door = Door::new(store);
    paired(&door, b"pair", Instant::now());
    assert_eq!(both_kinds(&door), [[0, 1, 0, 0], [0, 0, 0, 1]]);
    assert_eq!(asks(&asked), [1, 0]);
}

/// The door's counts of the shadows of the first binding's decodes that
/// were decided after their prefill: agreed, another target, another
/// outcome, skipped.
fn later_kinds(door: &Door) -> [u64; 4] {
    use ShadowKind::{Agreed, OtherOutcome, OtherTarget, Skipped};
    [Agreed, OtherTarget, OtherOutcome, Skipped].map(|kind| door.shadowed_later(0, kind))
}

/// A door over one prefill and two decodes whose decode side is decided
/// by the first row kept and shadowed under `next`, with a pair decided
/// for a later decode and its prefill dispatched: the door, the pair's
/// lease and the counts of the decisions that asked each side's shadow.
fn held_pair(next: Rule) -> (Door, LeaseId, [Arc<AtomicUsize>; 2]) {
    let store = pair_fleet(1, 2).expect("one prefill and two decodes");
    let next = [None, Some(next)];
    let asked = pair_shadowing(&store, 2, [Rule::Place(0); 2], next, |document| {
        document.explain.sample = 1;
    });
    let door = Door::new(store);
    let now = Instant::now();
    let first = door.decide(GATEWAY, &later_request(b"pair"), now);
    let lease = first.expect("a pair").lease.expect("a lease").lease.id;
    door.report(GATEWAY, lease, pair_dispatch(Side::Prefill, now));
    (door, lease, asked)
}

/// A decode decided after its prefill asks its side's shadow at that
/// second decision, over the partners of the held prefill, and compares
/// it with the decode the call picked. The count is its own: the pair's
/// first decision counted its decode side beside the decode it named
/// first, and that count stands.
#[test]
fn a_decode_decided_later_asks_its_shadow_at_its_own_call() {
    let (door, lease, asked) = held_pair(Rule::Place(1));
    assert_eq!(side_kinds(&door, PairSide::Decode), [0, 1, 0, 0]);
    assert_eq!(later_kinds(&door), [0; 4]);
    let call = later_request(b"pair");
    let decided = door.decide_decode(GATEWAY, &call, lease, Instant::now());
    let decided = decided.expect("a decode");
    let first = decided
        .decision
        .pick()
        .and_then(|pick| pick.targets.first());
    let decode = first
        .and_then(|target| target.pair)
        .map(|pair| pair.decode_id);
    assert_eq!(later_kinds(&door), [0, 1, 0, 0]);
    assert_eq!(side_kinds(&door, PairSide::Decode), [0, 1, 0, 0]);
    assert_eq!(kinds(&door), [0; 4]);
    assert_eq!(asks(&asked), [0, 2]);
    let record = door.explain(GATEWAY, b"pair");
    let record = record.expect("the sampled call");
    assert_eq!(record.decode_shadow, Some(ShadowKind::OtherTarget));
    assert!(record.decode_shadow_target.is_some() && record.decode_shadow_target != decode);
}

/// A later decode's shadow costs its call nothing. Under the floor of
/// its budget the shadow is not asked and is counted as skipped. One
/// that panics is counted with the door's faults, and the call answers
/// with its lease.
#[test]
fn a_later_decodes_shadow_under_the_floor_or_in_a_panic_leaves_the_call() {
    let (door, lease, asked) = held_pair(Rule::Place(1));
    let floor = u32::try_from(SHADOW_FLOOR.as_micros()).expect("a floor in microseconds");
    let mut short = later_request(b"pair");
    short.facts.budget_micros = floor - 1;
    let decided = door.decide_decode(GATEWAY, &short, lease, Instant::now());
    assert!(decided.expect("a decode").lease.is_some());
    assert_eq!(later_kinds(&door), [0, 0, 0, 1]);
    assert_eq!(asks(&asked), [0, 1], "the first decision's ask alone");

    let (door, lease, _) = held_pair(Rule::Panics);
    assert_eq!(door.faults(), 1, "the first decision's shadow");
    let call = later_request(b"pair");
    let decided = door.decide_decode(GATEWAY, &call, lease, Instant::now());
    let side = decided
        .expect("a decode")
        .lease
        .map(|leased| leased.lease.side);
    assert_eq!(side, Some(Side::Decode));
    assert_eq!(door.faults(), 2);
    assert_eq!(later_kinds(&door), [0; 4]);
}

/// A later decode's shadow books nothing: the store's counts behind the
/// call are those of the same call under a document that names no
/// shadow, the call's decode alone on its side.
#[test]
fn a_later_decodes_shadow_books_nothing() {
    let counts = |shadowed: bool| {
        let store = pair_fleet(1, 2).expect("one prefill and two decodes");
        let next = [None, shadowed.then_some(Rule::Place(1))];
        pair_shadowing(&store, 2, [Rule::Place(0); 2], next, |_| {});
        let door = Door::new(Arc::clone(&store));
        let now = Instant::now();
        let first = door.decide(GATEWAY, &later_request(b"pair"), now);
        let lease = first.expect("a pair").lease.expect("a lease").lease.id;
        door.report(GATEWAY, lease, pair_dispatch(Side::Prefill, now));
        let call = later_request(b"pair");
        let decided = door.decide_decode(GATEWAY, &call, lease, now);
        assert!(decided.expect("a decode").lease.is_some());
        (booked(&store, Role::Prefill), booked(&store, Role::Decode))
    };
    let plain = counts(false);
    assert_eq!(counts(true), plain);
    let decodes: u32 = plain.1.iter().map(|&(_, count)| count).sum();
    assert_eq!(decodes, 1, "the call's decode alone");
}

/// A decode call's shadow and its sampled record are found by the pair
/// binding the call decides under, whatever phase the call states. A
/// binding of the decode phase stands ahead of the pair's here and names
/// no shadow: it decides a decode asked alone, and no pair's. A call
/// that states the decode phase still asks the pair's decode shadow, and
/// counts it under the pair binding's position. Its record names that
/// binding, shadowed or not, so the record's names are the pair's.
#[test]
fn a_decode_call_finds_its_shadow_by_the_pair_binding_it_decides_under() {
    use crate::spec::{BindingSpec, Match};

    for shadowed in [true, false] {
        let store = pair_fleet(1, 2).expect("one prefill and two decodes");
        let next = [None, shadowed.then_some(Rule::Place(1))];
        pair_shadowing(&store, 2, [Rule::Place(0); 2], next, |document| {
            let alone = document.profiles.get("decode").cloned();
            let alone = alone.expect("the decode side's profile");
            document.profiles.insert("alone".into(), alone);
            let decode_first = BindingSpec {
                matcher: Match {
                    phase: vec![Phase::Decode],
                    ..Match::default()
                },
                profile: Some("alone".into()),
                ..BindingSpec::default()
            };
            document.bindings.insert(0, decode_first);
            document.explain.sample = 1;
        });
        let door = Door::new(store);
        let now = Instant::now();
        let first = door.decide(GATEWAY, &later_request(b"pair"), now);
        let lease = first.expect("a pair").lease.expect("a lease").lease.id;
        door.report(GATEWAY, lease, pair_dispatch(Side::Prefill, now));
        let mut call = later_request(b"pair");
        call.facts.phase = Phase::Decode;
        let decided = door.decide_decode(GATEWAY, &call, lease, now);
        let decided = decided.expect("a decode");
        assert!(decided.lease.is_some());
        // The pair's binding is the document's second.
        let counted = u64::from(shadowed);
        assert_eq!(door.shadowed_later(1, ShadowKind::OtherTarget), counted);
        assert_eq!(door.shadowed_later(0, ShadowKind::OtherTarget), 0);
        let explained = door.explained(GATEWAY, b"pair");
        let explained = explained.expect("the sampled call");
        let record = explained.record;
        assert_eq!(record.decision, decided.decision, "the call's own record");
        assert_eq!(record.binding, Some(1));
        let said = shadowed.then_some(ShadowKind::OtherTarget);
        assert_eq!(record.decode_shadow, said);
        let names = explained.names.expect("the generation that decided");
        assert_eq!((names.plan, names.profile.as_str()), ("pd-pair", "decode"));
    }
}
