//! The cases of the load reports the engine's task tells: each spawns the
//! task as a host does, on a paused clock, over the worker crate's own
//! registry, with its test feed as the load monitor and a picker that keeps
//! what it is told.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, OnceLock, Weak,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    load::state::{one_rank_load, LoadSnapshot},
    model_card::ModelCard,
    registry::{event::WorkerEvent, WorkerId as ObjectId},
    test_support::{no_health_check, ready_worker, WorkerRegistry},
    worker::{SchedulerLoadSnapshot, WorkerLoadResponse, WorkerStatus, WorkerType},
};
use tokio::{
    sync::broadcast::{self, Receiver},
    time::sleep,
};

use crate::{
    document::spec::{AppliedSpec, PolicySpec, ShadowSpec},
    engine::{
        decision::Targets,
        framework::{Picker, TieBand},
        policy_engine::{
            membership::Members,
            task::tests::{monitor, running, settle},
            PolicyEngine,
        },
        records::{worker_id, Live},
        view::{Candidate, FixedFacts, LoadRow, RequestFacts, WorkerId},
    },
    policies::random::Random,
    stages::{registry::StageRegistry, Lifecycle},
    tracking::{
        in_flight::WorkerSlot,
        lease::{Membership, Signal},
    },
};

const MODEL: &str = "model-a";

/// One report a stage was told: the worker, its rank, the row, the slot of
/// its count and the sample's instant. With them, the snapshot the engine
/// kept at hand while the stage heard it, by its place in memory, and zero
/// for none, and whether the stage had been told the worker as joined.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Told {
    worker: WorkerId,
    dp_rank: u16,
    row: LoadRow,
    slot: WorkerSlot,
    sampled_at_micros: u64,
    at_hand: usize,
    joined: bool,
}

/// What the stages of one engine heard: the reports, and the workers each
/// stage was told as joined and not as removed since, under the stage's
/// name. And of the stage named `panics`: the id of the worker at whose
/// report it panics now, zero for none, and how many reports it heard.
#[derive(Debug, Default)]
struct Kept {
    told: Mutex<Vec<Told>>,
    joined: Mutex<Vec<(&'static str, WorkerId)>>,
    panics_at: AtomicU64,
    ahead: AtomicUsize,
}

type Heard = Arc<Kept>;

/// The `random` picker as a stage that keeps load reports: it keeps each
/// report it is told. The one named `panics` counts each instead, and
/// panics at the report of the worker the case names.
#[derive(Debug)]
struct Keeps {
    name: &'static str,
    heard: Heard,
    engine: Arc<OnceLock<Weak<PolicyEngine>>>,
}

impl Lifecycle for Keeps {
    fn keeps_loads(&self) -> bool {
        true
    }

    fn on_membership(&self, change: &Membership<'_>) {
        let mut joined = self.heard.joined.lock();
        match *change {
            Membership::Added(id) => joined.push((self.name, id)),
            Membership::Removed(id) => joined.retain(|held| *held != (self.name, id)),
            _ => {}
        }
    }

    fn on_signal(&self, signal: &Signal) {
        let Signal::Load {
            worker,
            dp_rank,
            row,
            slot,
            sampled_at_micros,
        } = *signal
        else {
            return;
        };
        if self.name == "panics" {
            let panics = worker.0 == self.heard.panics_at.load(Ordering::Relaxed);
            assert!(!panics, "the case's own panic, at a report");
            self.heard.ahead.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let engine = self.engine.get().and_then(Weak::upgrade);
        let kept = engine.and_then(|engine| at_hand(&engine));
        let joined = self.heard.joined.lock().contains(&(self.name, worker));
        self.heard.told.lock().push(Told {
            worker,
            dp_rank,
            row,
            slot,
            sampled_at_micros,
            at_hand: kept.map_or(0, |kept| Arc::as_ptr(&kept) as usize),
            joined,
        });
    }
}

impl Picker for Keeps {
    fn name(&self) -> &'static str {
        self.name
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
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

/// The snapshot `engine` keeps at hand for its decisions: what a decision
/// whose request gives none reads.
fn at_hand(engine: &PolicyEngine) -> Option<Arc<LoadSnapshot>> {
    let newest = engine.loads.newest()?;
    (*newest).clone()
}

/// The engine of `preset` with [`Keeps`] in its pick's place, as a host
/// holds it for its task, and what that stage heard. `least_load` reads
/// load reports at its decisions, and `random` reads none. Where
/// `behind_a_panic`, the stage that keeps what it hears is the pick of a
/// second profile, the binding's shadow, and the first profile's pick is
/// the stage that panics at the report of the worker the case names: the
/// engine tells that one first.
fn keeping(preset: &str, behind_a_panic: bool) -> (Arc<PolicyEngine>, Heard) {
    let heard = Heard::default();
    let own = Arc::new(OnceLock::new());
    let mut stages = StageRegistry::builtin();
    for name in ["keeps", "panics"] {
        let (heard, own) = (Arc::clone(&heard), Arc::clone(&own));
        let added = stages.picker(name, move || -> Box<dyn Picker> {
            Box::new(Keeps {
                name,
                heard: Arc::clone(&heard),
                engine: Arc::clone(&own),
            })
        });
        added.expect("a free name");
    }
    let mut document = PolicySpec::preset(preset).expect("a built-in preset");
    document.generation = 1;
    let profile = document.profiles.get_mut(preset);
    let profile = profile.expect("the preset's profile");
    if behind_a_panic {
        profile.pick = "panics".into();
        let mut twin = profile.clone();
        twin.pick = "keeps".into();
        document.profiles.insert("twin".into(), twin);
        document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    } else {
        profile.pick = "keeps".into();
    }
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = Arc::new(PolicyEngine::new(
        spec.expect("the stages hold the pickers"),
    ));
    own.set(Arc::downgrade(&engine)).expect("set once");
    (engine, heard)
}

fn url(at: usize) -> String {
    format!("http://worker-{at}:8000")
}

/// The id a list's records name the worker at `url(at)` by: the fingerprint
/// of its address.
fn id(at: usize) -> WorkerId {
    worker_id(ready_worker(&url(at), MODEL).as_ref())
}

/// A registry in which the workers at `url(0)` to `url(workers - 1)` stand.
fn registry(workers: usize) -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new());
    for at in 0..workers {
        let registered = registry.register(ready_worker(&url(at), MODEL));
        registered.expect("a free address");
    }
    registry
}

/// The instant of the sample numbered `step`: a millisecond a step from one
/// origin, so two steps name two instants and one step always the same.
fn sample(step: u64) -> Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now) + Duration::from_millis(step)
}

/// One report a worker, each of `queued` waiting tokens: the worker at
/// `url(at)` sampled at the step `steps[at]`.
fn reports(steps: &[u64], queued: i32) -> HashMap<String, WorkerLoadResponse> {
    let report = |step: u64| WorkerLoadResponse {
        sampled_at: Some(sample(step)),
        ..one_rank_load(queued, 0.1, 1_000.0)
    };
    let stated = steps.iter().enumerate();
    stated.map(|(at, &step)| (url(at), report(step))).collect()
}

/// What the stage heard since the last read, in the order of the samples:
/// each report of a worker it had been told as joined.
fn told_since(heard: &Heard) -> Vec<Told> {
    let mut told = std::mem::take(&mut *heard.told.lock());
    assert!(told.iter().all(|word| word.joined), "behind its join");
    told.sort_by_key(|word| word.sampled_at_micros);
    told
}

/// The worker and the sample's step of each report in `told`, on `engine`'s
/// clock.
fn steps_of(engine: &PolicyEngine, told: &[Told]) -> Vec<(WorkerId, u64)> {
    let origin = engine.clock().micros(sample(0));
    let step = |word: &Told| (word.sampled_at_micros - origin) / 1_000;
    told.iter().map(|word| (word.worker, step(word))).collect()
}

/// A stage that keeps load reports is told the report of each worker it was
/// told as joined, as a list's record states it, from the task's first
/// step on: under the id the list names the worker by,
/// with its rank, the same row and the same sample instant on the engine's
/// clock, and with no slot. The row's age is at rest, where a record states
/// it at its decision. The third worker is a prefill and the second rank of
/// its engine: its report converts by its role, so it states no rate where
/// the first's states one, and it is told under its rank. The second
/// worker's report holds two ranks, and is told as their sum.
#[tokio::test(start_paused = true)]
async fn a_stage_that_keeps_reports_is_told_each_report_as_a_record_states_it() {
    let (engine, heard) = keeping("least_load", false);
    let registry = registry(2);
    let prefill = <BasicWorkerBuilder>::new(url(2))
        .worker_type(WorkerType::Prefill)
        .model(ModelCard::new(MODEL))
        .dp_config(1, 2)
        .health_config(no_health_check())
        .build();
    prefill.set_status(WorkerStatus::Ready);
    let prefill: Arc<dyn Worker> = Arc::new(prefill);
    let registered = registry.register(Arc::clone(&prefill));
    registered.expect("a free address");
    engine.set_registry(Arc::clone(&registry) as Arc<dyn Members>);
    // A rank's report stands under the rank's own address.
    let mut stated = reports(&[1, 2, 3], 4_096);
    let third = stated.remove(&url(2)).expect("the third report");
    stated.insert(prefill.url().to_string(), third);
    let second = stated.get_mut(&url(1)).expect("the second report");
    let mut rank = second.loads[0].clone();
    rank.dp_rank = 1;
    second.loads.push(rank);
    second.dp_rank_count = 2;
    let (_feed, channel) = monitor(&stated);
    engine.set_loads(channel);
    let task = running(&engine).await;

    let told = told_since(&heard);
    assert_eq!(told.len(), 3, "one report a standing worker");
    let worker = |at: usize| registry.get_by_url(&url(at)).expect("a worker");
    let list: Arc<[Arc<dyn Worker>]> = (0..2).map(worker).chain([prefill]).collect();
    let snapshot = at_hand(&engine).expect("the snapshot at hand");
    let (spec, clock) = (engine.spec(), engine.clock());
    let live = Live::at(spec.engine(), clock, Instant::now(), Some(&snapshot), None);
    let members = engine.pool_members(&list);
    for (position, word) in (0..).zip(&told) {
        let record = members.records(spec, clock).read(&list, &live, position);
        let record = record.expect("the worker's record");
        let at_rest = LoadRow {
            report_age_ms: 0,
            ..record.load.expect("its report")
        };
        let stated = (record.id, record.dp_rank, at_rest);
        assert_eq!((word.worker, word.dp_rank, word.row), stated, "{position}");
        assert_eq!(word.sampled_at_micros, record.load_sampled_at_micros);
        let waiting = if position == 1 { 8_192 } else { 4_096 };
        let stated = (word.slot, word.row.waiting_uncached_tokens);
        assert_eq!(stated, (WorkerSlot::NONE, waiting), "{position}");
    }
    let rates = (told[0].row.live_rate, told[2].row.live_rate);
    assert_eq!(rates, (1_000, 0), "a prefill states no such rate");
    let ranks: Vec<u16> = told.iter().map(|word| word.dp_rank).collect();
    assert_eq!(ranks, [0, 0, 1], "each under its rank");
    task.abort();
}

/// A document that reads no load report and no named signal keeps its
/// host's channel for a stage that keeps reports, and for it alone: the
/// stage is told each new report, and a decision whose request gives no
/// snapshot finds none.
#[tokio::test(start_paused = true)]
async fn a_document_that_reads_no_report_keeps_the_channel_for_its_stage_alone() {
    let (engine, heard) = keeping("random", false);
    let needs = engine.spec().needs();
    assert!(!needs.loads && !needs.signal_cells, "the preset reads none");
    engine.set_registry(registry(2) as Arc<dyn Members>);
    let (feed, channel) = monitor(&reports(&[1, 2], 0));
    engine.set_loads(channel);
    let task = running(&engine).await;
    let told = steps_of(&engine, &told_since(&heard));
    assert_eq!(told, [(id(0), 1), (id(1), 2)]);

    feed.publish(&reports(&[1, 3], 0));
    settle().await;
    assert_eq!(steps_of(&engine, &told_since(&heard)), [(id(1), 3)]);
    assert!(
        at_hand(&engine).is_none(),
        "none stands for a decision to read"
    );
    task.abort();
}

/// A report is told ahead of the snapshot a decision reads: while a stage
/// hears the reports of a snapshot, the engine still keeps the snapshot
/// before it at hand, and none at the first.
#[tokio::test(start_paused = true)]
async fn a_report_is_told_ahead_of_the_snapshot_a_decision_reads() {
    let (engine, heard) = keeping("least_load", false);
    engine.set_registry(registry(2) as Arc<dyn Members>);
    let (feed, channel) = monitor(&reports(&[1, 2], 0));
    engine.set_loads(channel);
    let task = running(&engine).await;
    let kept = |told: Vec<Told>| -> Vec<usize> { told.iter().map(|word| word.at_hand).collect() };
    assert_eq!(kept(told_since(&heard)), [0, 0], "no snapshot at hand yet");

    let first = at_hand(&engine).expect("the first snapshot");
    feed.publish(&reports(&[3, 4], 0));
    settle().await;
    let first = Arc::as_ptr(&first) as usize;
    assert_eq!(kept(told_since(&heard)), [first, first]);
    let second = at_hand(&engine).expect("the second snapshot");
    assert_ne!(
        Arc::as_ptr(&second) as usize,
        first,
        "kept behind the telling"
    );
    task.abort();
}

/// The report told last of a worker is not told again, however many
/// snapshots show it one after another, also behind a snapshot that lacked
/// it. A report that arrives with the same counts and a new sample instant
/// is a new report, and so is a sample earlier than the one told last,
/// also one told before: both are told.
#[tokio::test(start_paused = true)]
async fn a_report_is_told_once_and_a_new_sample_of_the_same_counts_again() {
    let (engine, heard) = keeping("least_load", false);
    engine.set_registry(registry(3) as Arc<dyn Members>);
    let (feed, channel) = monitor(&reports(&[1, 2, 3], 512));
    engine.set_loads(channel);
    let task = running(&engine).await;
    let told = steps_of(&engine, &told_since(&heard));
    assert_eq!(told, [(id(0), 1), (id(1), 2), (id(2), 3)]);

    // The third worker reports again: the same counts, a new sample.
    feed.publish(&reports(&[1, 2, 4], 512));
    settle().await;
    let told = told_since(&heard);
    assert_eq!(steps_of(&engine, &told), [(id(2), 4)]);
    assert_eq!(told[0].row.waiting_uncached_tokens, 512);

    feed.publish(&reports(&[1, 2, 4], 512));
    settle().await;
    assert_eq!(told_since(&heard), [], "no report is new");

    // Shown again behind a snapshot that lacked it, it is still the report
    // told last.
    let mut lacking = reports(&[1, 2, 4], 512);
    lacking.remove(&url(2)).expect("the third report");
    for stated in [lacking, reports(&[1, 2, 4], 512)] {
        feed.publish(&stated);
        settle().await;
        assert_eq!(told_since(&heard), [], "told once");
    }
    // An earlier sample is another report.
    feed.publish(&reports(&[1, 2, 3], 512));
    settle().await;
    assert_eq!(steps_of(&engine, &told_since(&heard)), [(id(2), 3)]);
    task.abort();
}

/// The host's channel keeps one snapshot. Of two publications between two
/// turns of the task the stage is told the later report alone, and nothing
/// it is told shows the one passed over.
#[tokio::test(start_paused = true)]
async fn of_two_publications_between_two_turns_the_later_report_is_told() {
    let (engine, heard) = keeping("least_load", false);
    engine.set_registry(registry(1) as Arc<dyn Members>);
    let (feed, channel) = monitor(&reports(&[1], 100));
    engine.set_loads(channel);
    let task = running(&engine).await;
    assert_eq!(steps_of(&engine, &told_since(&heard)), [(id(0), 1)]);

    feed.publish(&reports(&[2], 200));
    feed.publish(&reports(&[3], 300));
    settle().await;
    let told = told_since(&heard);
    assert_eq!(steps_of(&engine, &told), [(id(0), 3)], "the later one");
    assert_eq!(told[0].row.waiting_uncached_tokens, 300);
    task.abort();
}

/// The workers told are those the task told as joined, by the registry the
/// host handed. A worker registered since the last snapshot is told its
/// report, also one the snapshot before held already, and a worker removed
/// is told no more, whatever the snapshot holds at its address. A worker
/// whose object the registry replaces is told its report once more. A joined
/// worker with no report in a snapshot is told none. An engine handed no
/// registry tells no report. A registry handed late is followed from the
/// task's next upkeep, and the first snapshot behind it tells every report.
#[tokio::test(start_paused = true)]
async fn the_workers_the_task_told_as_joined_are_told_their_reports() {
    let (engine, heard) = keeping("least_load", false);
    let registry = registry(2);
    engine.set_registry(Arc::clone(&registry) as Arc<dyn Members>);
    let (feed, channel) = monitor(&reports(&[1, 2, 3], 0));
    engine.set_loads(channel);
    let task = running(&engine).await;
    let told = steps_of(&engine, &told_since(&heard));
    assert_eq!(told, [(id(0), 1), (id(1), 2)], "the third does not stand");

    // Its report stood in the snapshot before, untold: it is its first.
    let joined = registry.register(ready_worker(&url(2), MODEL));
    joined.expect("a free address");
    feed.publish(&reports(&[1, 2, 3], 0));
    settle().await;
    assert_eq!(steps_of(&engine, &told_since(&heard)), [(id(2), 3)]);

    // A joined worker whose report a snapshot lacks is told none of it.
    let mut lacking = reports(&[5, 0, 6], 0);
    lacking.remove(&url(1)).expect("the second report");
    feed.publish(&lacking);
    settle().await;
    let told = steps_of(&engine, &told_since(&heard));
    assert_eq!(told, [(id(0), 5), (id(2), 6)], "none of the second");

    let first = registry.get_id_by_url(&url(0)).expect("a worker there");
    registry.remove(&first).expect("the worker was removed");
    feed.publish(&reports(&[7, 8, 9], 0));
    settle().await;
    let told = steps_of(&engine, &told_since(&heard));
    assert_eq!(told, [(id(1), 8), (id(2), 9)], "the first is gone");

    // The follow writes what it holds of a replaced worker again.
    let second = registry.get_id_by_url(&url(1)).expect("a worker there");
    let replacement = ready_worker(&url(1), MODEL);
    assert!(registry.replace(&second, replacement), "it stands");
    feed.publish(&reports(&[7, 8, 9], 0));
    settle().await;
    let told = steps_of(&engine, &told_since(&heard));
    assert_eq!(told, [(id(1), 8)], "once more behind its replacement");
    task.abort();

    let (alone, heard) = keeping("least_load", false);
    let (feed, channel) = monitor(&reports(&[1, 2], 0));
    alone.set_loads(channel);
    let task = running(&alone).await;
    assert!(at_hand(&alone).is_some(), "the snapshot is kept");
    assert_eq!(told_since(&heard), [], "no registry, no report told");

    // A registry handed late is followed from the task's next upkeep, and
    // by no snapshot ahead of it.
    alone.set_registry(self::registry(2) as Arc<dyn Members>);
    feed.publish(&reports(&[1, 3], 0));
    settle().await;
    assert_eq!(told_since(&heard), [], "no worker is told as joined yet");
    // None counts as told yet, so the first telling behind the upkeep holds
    // every worker's report: the same ones, shown again.
    sleep(Duration::from_secs(1)).await;
    feed.publish(&reports(&[1, 3], 0));
    settle().await;
    let told = steps_of(&alone, &told_since(&heard));
    assert_eq!(told, [(id(0), 1), (id(1), 3)], "the first one too");
    task.abort();
}

/// A host's registry that states an id of its own for each worker, or
/// [`ONE`] for every worker where `one_id`, and counts what it is asked: a
/// subscription, its standing workers, and a worker's id. While `closed`,
/// each channel it hands is a closed one. While `panics`, its answer of the
/// standing workers panics. While `own_words`, it hands `words`, a channel
/// of its own that keeps two words, and says no word of its registry.
struct Counted {
    registry: Arc<WorkerRegistry>,
    asks: AtomicUsize,
    one_id: bool,
    closed: AtomicBool,
    panics: AtomicBool,
    words: broadcast::Sender<WorkerEvent>,
    own_words: AtomicBool,
}

impl Counted {
    fn over(registry: Arc<WorkerRegistry>, one_id: bool) -> Arc<Self> {
        Arc::new(Self {
            registry,
            asks: AtomicUsize::new(0),
            one_id,
            closed: AtomicBool::new(false),
            panics: AtomicBool::new(false),
            words: broadcast::channel(2).0,
            own_words: AtomicBool::new(false),
        })
    }

    fn asks(&self) -> usize {
        self.asks.load(Ordering::Relaxed)
    }
}

/// The id a host states for every worker of a case.
const ONE: WorkerId = WorkerId(77);

/// The id the host of [`Counted`] states of the worker at `url(at)`.
fn stated(at: usize) -> WorkerId {
    WorkerId(id(at).0.wrapping_add(7))
}

impl Members for Counted {
    fn events(&self) -> Receiver<WorkerEvent> {
        self.asks.fetch_add(1, Ordering::Relaxed);
        if self.closed.load(Ordering::Relaxed) {
            // Its sender is dropped here.
            return broadcast::channel(1).1;
        }
        if self.own_words.load(Ordering::Relaxed) {
            return self.words.subscribe();
        }
        Members::events(&*self.registry)
    }

    fn standing(&self) -> Vec<(ObjectId, Arc<dyn Worker>)> {
        self.asks.fetch_add(1, Ordering::Relaxed);
        assert!(!self.panics.load(Ordering::Relaxed), "the host's own code");
        Members::standing(&*self.registry)
    }

    fn facts_of(&self, worker: &Arc<dyn Worker>) -> FixedFacts {
        self.asks.fetch_add(1, Ordering::Relaxed);
        let own = WorkerId(worker_id(worker.as_ref()).0.wrapping_add(7));
        FixedFacts {
            id: Some(if self.one_id { ONE } else { own }),
            ..FixedFacts::default()
        }
    }
}

/// A telling with no word of the registry ready asks the host nothing: the
/// workers are those the follow told as joined, each under the id the host
/// stated at its join.
/// The registry is read whole at the follow's start and by no snapshot
/// after.
#[tokio::test(start_paused = true)]
async fn a_telling_asks_the_registry_nothing() {
    let (engine, heard) = keeping("least_load", false);
    let counted = Counted::over(registry(2), false);
    engine.set_registry(Arc::clone(&counted) as Arc<dyn Members>);
    let (feed, channel) = monitor(&reports(&[1, 2], 0));
    engine.set_loads(channel);
    let task = running(&engine).await;
    let told = steps_of(&engine, &told_since(&heard));
    assert_eq!(told, [(stated(0), 1), (stated(1), 2)]);
    let started = counted.asks();
    for step in [3, 5, 7] {
        feed.publish(&reports(&[step, step + 1], 0));
        settle().await;
        let told = steps_of(&engine, &told_since(&heard));
        assert_eq!(told, [(stated(0), step), (stated(1), step + 1)]);
    }
    assert_eq!(counted.asks(), started, "no ask of the host at a snapshot");
    task.abort();
}

/// A telling runs no step of the follow's upkeep: a subscription and a
/// whole read the follow owes wait for the task's next upkeep, however many
/// snapshots come first. So a host that hands closed channels is subscribed
/// to and read whole once a second, and a host whose read panics counts one
/// fault a second. The workers the follow holds are told their reports all
/// the while.
#[tokio::test(start_paused = true)]
async fn a_telling_leaves_what_the_follow_owes_to_the_upkeep() {
    let (engine, heard) = keeping("least_load", false);
    let counted = Counted::over(registry(2), false);
    // Every channel the host hands is a closed one.
    counted.closed.store(true, Ordering::Relaxed);
    engine.set_registry(Arc::clone(&counted) as Arc<dyn Members>);
    let (feed, channel) = monitor(&reports(&[1, 2], 0));
    engine.set_loads(channel);
    let task = running(&engine).await;
    let both = |step: u64| [(stated(0), step), (stated(1), step + 1)];
    assert_eq!(steps_of(&engine, &told_since(&heard)), both(1));
    // The follow's start asks four things: a subscription, the standing
    // workers, and the id of each of the two.
    assert_eq!(counted.asks(), 4, "the follow's start");

    for step in [3, 5, 7] {
        feed.publish(&reports(&[step, step + 1], 0));
        settle().await;
        assert_eq!(steps_of(&engine, &told_since(&heard)), both(step));
    }
    assert_eq!(counted.asks(), 4, "no subscription, no read at a snapshot");
    sleep(Duration::from_secs(1)).await;
    assert_eq!(counted.asks(), 8, "both at the next upkeep");

    // The host's channels stand again, and its read panics.
    counted.closed.store(false, Ordering::Relaxed);
    counted.panics.store(true, Ordering::Relaxed);
    sleep(Duration::from_secs(1)).await;
    assert_eq!((counted.asks(), engine.upkeep_faults()), (10, 1));
    for step in [9, 11, 13] {
        feed.publish(&reports(&[step, step + 1], 0));
        settle().await;
        assert_eq!(steps_of(&engine, &told_since(&heard)), both(step));
    }
    let owed = (counted.asks(), engine.upkeep_faults());
    assert_eq!(owed, (10, 1), "the read is tried again at no snapshot");
    sleep(Duration::from_secs(1)).await;
    let tried = (counted.asks(), engine.upkeep_faults());
    assert_eq!(tried, (11, 2), "at the next upkeep alone");
    task.abort();
}

/// A word that says the channel dropped events reads the registry whole
/// where it is heard, at a telling too: a worker the follow learns of by
/// that read alone is told its report in the snapshot of that telling. The
/// task takes its turns in no set order, so sixteen rounds would pass a
/// report over if a telling left the dropped events unread. Each whole read
/// writes what the follow holds of every worker again, so every report is
/// told once more behind it.
#[tokio::test(start_paused = true)]
async fn a_telling_that_hears_of_dropped_words_reads_the_registry_whole() {
    let (engine, heard) = keeping("least_load", false);
    let counted = Counted::over(registry(1), false);
    counted.own_words.store(true, Ordering::Relaxed);
    engine.set_registry(Arc::clone(&counted) as Arc<dyn Members>);
    let mut steps = vec![1];
    let (feed, channel) = monitor(&reports(&steps, 0));
    engine.set_loads(channel);
    let task = running(&engine).await;
    assert_eq!(steps_of(&engine, &told_since(&heard)), [(stated(0), 1)]);

    let standing = Members::standing(&*counted.registry);
    let (first_id, first) = standing.into_iter().next().expect("the first worker");
    let mut every = vec![(stated(0), 1)];
    for at in 1..=16 {
        let joined = counted.registry.register(ready_worker(&url(at), MODEL));
        joined.expect("a free address");
        // Three words of a worker the follow holds: one more than the
        // host's channel keeps.
        for _ in 0..3 {
            let word = WorkerEvent::Registered {
                worker_id: first_id.clone(),
                worker: Arc::clone(&first),
            };
            assert!(counted.words.send(word).is_ok(), "a subscription stands");
        }
        steps.push(10 + at as u64);
        every.push((stated(at), 10 + at as u64));
        feed.publish(&reports(&steps, 0));
        settle().await;
        let told = steps_of(&engine, &told_since(&heard));
        assert_eq!(told, every, "the worker {at}, behind a whole read");
    }
    task.abort();
}

/// A host whose answer of a worker's id says one more word of its first
/// worker on its own channel, `left` times over: each word heard makes the
/// next ready. It notes at each answer which snapshot the engine keeps at
/// hand, by its place in memory, and zero for none.
struct Echoing {
    registry: Arc<WorkerRegistry>,
    words: broadcast::Sender<WorkerEvent>,
    left: AtomicUsize,
    engine: Weak<PolicyEngine>,
    at_hand: Mutex<Vec<usize>>,
}

impl Echoing {
    /// One word of the first worker, which the follow holds: a
    /// registration that tells no stage anything.
    fn say(&self) {
        let standing = Members::standing(&*self.registry);
        let first = standing.into_iter().next();
        let (worker_id, worker) = first.expect("a standing worker");
        drop(
            self.words
                .send(WorkerEvent::Registered { worker_id, worker }),
        );
    }
}

impl Members for Echoing {
    fn events(&self) -> Receiver<WorkerEvent> {
        self.words.subscribe()
    }

    fn standing(&self) -> Vec<(ObjectId, Arc<dyn Worker>)> {
        Members::standing(&*self.registry)
    }

    fn facts_of(&self, _worker: &Arc<dyn Worker>) -> FixedFacts {
        let kept = self.engine.upgrade().and_then(|engine| at_hand(&engine));
        let kept = kept.map_or(0, |kept| Arc::as_ptr(&kept) as usize);
        self.at_hand.lock().push(kept);
        // One thread runs the task and this answer.
        let left = self.left.load(Ordering::Relaxed);
        if left > 0 {
            self.left.store(left - 1, Ordering::Relaxed);
            self.say();
        }
        FixedFacts::default()
    }
}

/// A telling hears at most the words that stood ready when it began, so a
/// registry whose every word makes another ready holds no snapshot back.
/// Forty-one words follow one another here, each made by the hearing of the
/// one before, and the snapshot published beside the first is at hand
/// before the last is heard.
#[tokio::test(start_paused = true)]
async fn a_telling_hears_no_more_words_than_stood_ready_at_its_start() {
    let (engine, heard) = keeping("least_load", false);
    let host = Arc::new(Echoing {
        registry: registry(1),
        words: broadcast::channel(8).0,
        left: AtomicUsize::new(0),
        engine: Arc::downgrade(&engine),
        at_hand: Mutex::default(),
    });
    engine.set_registry(Arc::clone(&host) as Arc<dyn Members>);
    let (feed, channel) = monitor(&reports(&[1], 0));
    engine.set_loads(channel);
    let task = running(&engine).await;
    assert_eq!(steps_of(&engine, &told_since(&heard)), [(id(0), 1)]);

    host.left.store(40, Ordering::Relaxed);
    host.say();
    feed.publish(&reports(&[2], 0));
    settle().await;
    assert_eq!(host.left.load(Ordering::Relaxed), 0, "every word was heard");
    assert_eq!(steps_of(&engine, &told_since(&heard)), [(id(0), 2)]);
    let second = at_hand(&engine).expect("the second snapshot");
    let last = host.at_hand.lock().last().copied();
    let second = Arc::as_ptr(&second) as usize;
    assert_eq!(last, Some(second), "kept ahead of the last word");
    task.abort();
}

/// A telling first hears the registry's words that stand ready: a worker
/// registered beside a publication is told its report in it, behind its
/// join. The task takes its turns in no set order, so thirty-two rounds
/// would pass a report over without that. The same at a hand-over: a
/// worker registered beside a new channel is told the report it holds.
#[tokio::test(start_paused = true)]
async fn a_worker_registered_beside_a_publication_is_told_its_report() {
    let (engine, heard) = keeping("least_load", false);
    let registry = registry(1);
    engine.set_registry(Arc::clone(&registry) as Arc<dyn Members>);
    let mut steps = vec![1];
    let (feed, channel) = monitor(&reports(&steps, 0));
    engine.set_loads(channel);
    let task = running(&engine).await;
    assert_eq!(steps_of(&engine, &told_since(&heard)), [(id(0), 1)]);

    for at in 1..=32 {
        let joined = registry.register(ready_worker(&url(at), MODEL));
        joined.expect("a free address");
        steps.push(10 + at as u64);
        feed.publish(&reports(&steps, 0));
        settle().await;
        let told = steps_of(&engine, &told_since(&heard));
        assert_eq!(told, [(id(at), 10 + at as u64)], "the worker {at}");
    }

    let mut monitors = Vec::new();
    for at in 33..=48 {
        let joined = registry.register(ready_worker(&url(at), MODEL));
        joined.expect("a free address");
        steps.push(10 + at as u64);
        let (next, channel) = monitor(&reports(&steps, 0));
        engine.set_loads(channel);
        monitors.push(next);
        sleep(Duration::from_secs(1)).await;
        let told = steps_of(&engine, &told_since(&heard));
        assert_eq!(told, [(id(at), 10 + at as u64)], "the worker {at}");
    }
    task.abort();
}

/// While the follow holds two workers under one id, the id's next life
/// ahead of the earlier one's removal, neither is told a report: nothing
/// says which of the two the id is. The one that stands alone is told.
#[tokio::test(start_paused = true)]
async fn of_two_workers_under_one_id_neither_is_told_until_one_stands_alone() {
    let (engine, heard) = keeping("least_load", false);
    let counted = Counted::over(registry(1), true);
    engine.set_registry(Arc::clone(&counted) as Arc<dyn Members>);
    let (feed, channel) = monitor(&reports(&[1, 2], 0));
    engine.set_loads(channel);
    let task = running(&engine).await;
    assert_eq!(steps_of(&engine, &told_since(&heard)), [(ONE, 1)]);

    let next = counted.registry.register(ready_worker(&url(1), MODEL));
    next.expect("a free address");
    feed.publish(&reports(&[3, 4], 0));
    settle().await;
    assert_eq!(told_since(&heard), [], "two workers under one id");

    let first = counted.registry.get_id_by_url(&url(0));
    let removed = counted.registry.remove(&first.expect("a worker there"));
    removed.expect("the worker was removed");
    feed.publish(&reports(&[5, 6], 0));
    settle().await;
    let told = steps_of(&engine, &told_since(&heard));
    assert_eq!(told, [(ONE, 6)], "the one that stands alone");
    task.abort();
}

/// A report whose conversion panics is held and counted, once: it is told
/// to no stage and counts as told, the other worker's report is told, the
/// task goes on and keeps the snapshot, and the worker's next sample is
/// told. The sum of two ranks' resident tokens overflows where overflow is
/// checked.
#[tokio::test(start_paused = true)]
#[cfg_attr(
    not(debug_assertions),
    ignore = "the sum wraps where overflow is not checked"
)]
async fn a_report_that_panics_at_its_conversion_ends_no_task() {
    let (engine, heard) = keeping("least_load", false);
    engine.set_registry(registry(2) as Arc<dyn Members>);
    let mut stated = reports(&[1, 2], 0);
    let rank = |resident: i64| SchedulerLoadSnapshot {
        num_resident_tokens: Some(resident),
        ..SchedulerLoadSnapshot::default()
    };
    let second = stated.get_mut(&url(1)).expect("the second report");
    second.loads = vec![rank(i64::MAX), rank(1)];
    let (feed, channel) = monitor(&stated);
    engine.set_loads(channel);
    let task = running(&engine).await;
    let told = steps_of(&engine, &told_since(&heard));
    assert_eq!(told, [(id(0), 1)], "the other worker's report is told");
    assert_eq!(engine.upkeep_faults(), 1, "held and counted");
    assert!(at_hand(&engine).is_some(), "the snapshot is kept");

    // The report that panicked counts as told: it is not tried again.
    feed.publish(&stated);
    settle().await;
    assert_eq!(told_since(&heard), [], "no report is new");
    assert_eq!(engine.upkeep_faults(), 1, "counted once");

    feed.publish(&reports(&[1, 3], 0));
    settle().await;
    let told = steps_of(&engine, &told_since(&heard));
    assert_eq!(told, [(id(1), 3)], "the worker's next sample");
    assert!(!task.is_finished(), "the task goes on");
    task.abort();
}

/// A stage that panics at a report is held and counted, each report under
/// a catch of its own: that stage hears the ones beside it, the stage
/// behind it is told every report, the task goes on and the snapshot is
/// kept at hand. Nothing is told twice: the report counts as told, and
/// the worker's next sample is told.
#[tokio::test(start_paused = true)]
async fn a_stage_that_panics_at_a_report_takes_none_from_the_stage_behind_it() {
    let (engine, heard) = keeping("least_load", true);
    engine.set_registry(registry(3) as Arc<dyn Members>);
    let (feed, channel) = monitor(&reports(&[1, 2, 3], 0));
    engine.set_loads(channel);
    let task = running(&engine).await;
    let told = steps_of(&engine, &told_since(&heard));
    assert_eq!(told, [(id(0), 1), (id(1), 2), (id(2), 3)]);
    // The reports the stage told first heard, and the panics counted.
    let ahead = || (heard.ahead.load(Ordering::Relaxed), engine.upkeep_faults());
    assert_eq!(ahead(), (3, 0), "both stages, whole");

    // The stage told first panics at one worker's report, each worker in
    // turn: whatever the order of a telling, it hears the two beside it.
    for (round, at) in (1..).zip(0..3) {
        heard.panics_at.store(id(at).0, Ordering::Relaxed);
        let step = 10 * round;
        feed.publish(&reports(&[step, step + 1, step + 2], 0));
        settle().await;
        let told = steps_of(&engine, &told_since(&heard));
        let whole = [(id(0), step), (id(1), step + 1), (id(2), step + 2)];
        assert_eq!(told, whole, "the stage behind is told every report");
        let heard = 3 + 2 * round as usize;
        assert_eq!(ahead(), (heard, round), "it misses that report alone");
    }
    let kept = at_hand(&engine).expect("the snapshot is kept");
    let second = kept.report(&url(1)).map(|(_, at)| at);
    assert_eq!(second, Some(sample(31)), "the newest one");

    feed.publish(&reports(&[30, 31, 32], 0));
    settle().await;
    assert_eq!(told_since(&heard), [], "nothing is told twice");
    assert_eq!(ahead(), (9, 3), "the report is not tried again");

    feed.publish(&reports(&[30, 31, 40], 0));
    settle().await;
    assert_eq!(steps_of(&engine, &told_since(&heard)), [(id(2), 40)]);
    assert_eq!(ahead(), (9, 4), "its next sample");
    task.abort();
}
