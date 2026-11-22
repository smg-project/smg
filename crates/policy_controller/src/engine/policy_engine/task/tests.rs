//! The cases of the engine's own task: each spawns it as a host does, on a
//! paused clock, with the worker crate's test feed as the load monitor.

use std::{
    collections::HashMap,
    future::Future,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use smg_worker::{
    identity::basic::Worker,
    load::state::{one_rank_load, LoadReceiver, LoadSnapshot, TestFeed},
    test_support::ready_worker,
    worker::WorkerLoadResponse,
};
use tokio::{
    task::JoinHandle,
    time::{advance, sleep},
};

use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision_engine::{DecisionEngine, Request, RequestMaker, SweepOutcome, WorkerLists},
        policy_engine::{PolicyEngine, PoolMembers},
        prompt_hashes::HashSeeds,
        view::{Constraints, Identity, RequestFacts, WorkerId},
    },
    store::GatewayId,
    tracking::{
        lease::{Outcome, Side, TransportClass},
        DecisionGuard,
    },
};

const MODEL: &str = "model-a";

/// A stock document as `generation`.
fn stock(name: &str, generation: u64) -> AppliedSpec {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = generation;
    AppliedSpec::validate(&document).expect("the document validates")
}

/// The engine of a stock document, as a host that spawns its task holds it.
fn preset(name: &str) -> Arc<PolicyEngine> {
    Arc::new(PolicyEngine::new(stock(name, 1)))
}

fn url(at: usize) -> String {
    format!("http://worker-{at}:8000")
}

fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
    (0..workers)
        .map(|at| ready_worker(&url(at), MODEL))
        .collect()
}

/// One report a worker: the worker at `short` has no queue, every other a
/// long one.
fn reports(workers: usize, short: usize) -> HashMap<String, WorkerLoadResponse> {
    (0..workers)
        .map(|at| {
            let queued = if at == short { 0 } else { 90_000 };
            (url(at), one_rank_load(queued, 0.1, 1_000.0))
        })
        .collect()
}

/// A load monitor's stand-in that published `reports`, and its channel as
/// a monitor's subscription hands it behind a publication: what was
/// published is marked seen, so the channel says nothing until the next.
fn monitor(reports: &HashMap<String, WorkerLoadResponse>) -> (TestFeed, LoadReceiver) {
    let mut channel = None;
    let feed = TestFeed::attached_to(|loads| channel = loads);
    feed.publish(reports);
    let mut channel = channel.expect("the feed's channel");
    channel.mark_unchanged();
    (feed, channel)
}

/// A request stated ahead over one list, with the snapshot its host gives
/// it, where it gives one.
struct Asks<'a> {
    request: Request<'a>,
    members: &'a PoolMembers,
    loads: Option<Arc<LoadSnapshot>>,
}

impl RequestMaker for Asks<'_> {
    fn model(&self) -> &str {
        self.request.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.request
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.members)
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.loads.as_ref()
    }
}

/// A request of `seed` over `members` that gives `loads`.
fn asks(seed: u64, members: &PoolMembers, loads: Option<Arc<LoadSnapshot>>) -> Asks<'_> {
    let facts = RequestFacts {
        model: MODEL,
        seed,
        ..RequestFacts::default()
    };
    Asks {
        request: Request {
            id: b"request",
            facts,
            constraints: Constraints::default(),
        },
        members,
        loads,
    }
}

/// The place in its list of the worker a request's whole decision names
/// first.
fn picked(engine: &PolicyEngine, asks: &Asks<'_>) -> u32 {
    let decided = engine.decide_under(GatewayId::LOCAL, asks, Instant::now());
    let whole = decided.expect("an answer").whole();
    let decision = whole.expect("a whole decision").0.decision;
    let pick = decision.pick().expect("a pick");
    pick.targets.first().expect("a target").position
}

/// Lets the task take what stands ready for it: no second passes.
async fn settle() {
    sleep(Duration::from_millis(1)).await;
}

/// A task spawned as a host spawns the engine's, past its first step.
async fn spawned(task: impl Future<Output = ()> + Send + 'static) -> JoinHandle<()> {
    #[expect(
        clippy::disallowed_methods,
        reason = "the case holds the task's handle and ends the task"
    )]
    let task = tokio::spawn(task);
    settle().await;
    task
}

/// The engine's task, spawned as a host spawns it, past its first step.
async fn running(engine: &Arc<PolicyEngine>) -> JoinHandle<()> {
    let engine = Arc::clone(engine);
    spawned(async move { engine.run().await }).await
}

/// The task keeps the newest snapshot of its host's channel at hand: a
/// decision whose request gives none picks by the reports the monitor
/// published last, with no call of the host between a publication and the
/// decision. Until the task runs the engine holds no snapshot and reads no
/// report, and a monitor that ended leaves its last snapshot standing.
#[tokio::test(start_paused = true)]
async fn the_task_keeps_the_newest_snapshot_for_a_request_that_gives_none() {
    let engine = preset("least_load");
    let list = pool(4);
    let members = engine.pool_members(&list);
    let (feed, channel) = monitor(&reports(4, 2));
    engine.set_loads(channel);
    let mut unread = [0_u32; 4];
    for seed in 0..64 {
        unread[picked(&engine, &asks(seed, &members, None)) as usize] += 1;
    }
    assert!(unread.iter().all(|&picks| picks > 0), "{unread:?}");

    let task = running(&engine).await;
    for seed in 0..32 {
        let place = picked(&engine, &asks(seed, &members, None));
        assert_eq!(place, 2, "seed {seed}: the shortest queue published");
    }
    feed.publish(&reports(4, 0));
    settle().await;
    for seed in 0..32 {
        let place = picked(&engine, &asks(seed, &members, None));
        assert_eq!(place, 0, "seed {seed}: the shortest queue published last");
    }
    drop(feed);
    sleep(Duration::from_secs(3)).await;
    for seed in 0..32 {
        let place = picked(&engine, &asks(seed, &members, None));
        assert_eq!(place, 0, "seed {seed}: the monitor's last snapshot stands");
    }
    task.abort();
}

/// A channel handed after the task's start is taken at the task's next
/// upkeep, within a second: the order of a host's two calls is free.
#[tokio::test(start_paused = true)]
async fn the_task_finds_a_channel_handed_after_its_start() {
    let engine = preset("least_load");
    let list = pool(4);
    let members = engine.pool_members(&list);
    let task = running(&engine).await;
    let (_feed, channel) = monitor(&reports(4, 2));
    engine.set_loads(channel);
    sleep(Duration::from_secs(1)).await;
    for seed in 0..32 {
        let place = picked(&engine, &asks(seed, &members, None));
        assert_eq!(place, 2, "seed {seed}: the shortest queue published");
    }
    task.abort();
}

/// A later hand-over replaces the channel: the task follows the newest one
/// from its next upkeep, and the monitor of the one before is heard no
/// more. A host hands another channel where it builds another monitor.
#[tokio::test(start_paused = true)]
async fn a_later_hand_over_replaces_the_channel() {
    let engine = preset("least_load");
    let list = pool(4);
    let members = engine.pool_members(&list);
    let (first, channel) = monitor(&reports(4, 2));
    engine.set_loads(channel);
    let task = running(&engine).await;
    assert_eq!(picked(&engine, &asks(7, &members, None)), 2);
    let (_second, channel) = monitor(&reports(4, 3));
    engine.set_loads(channel);
    sleep(Duration::from_secs(1)).await;
    for seed in 0..32 {
        let place = picked(&engine, &asks(seed, &members, None));
        assert_eq!(place, 3, "seed {seed}: the channel handed last");
    }
    first.publish(&reports(4, 0));
    settle().await;
    let place = picked(&engine, &asks(7, &members, None));
    assert_eq!(place, 3, "the first monitor is heard no more");
    task.abort();
}

/// Two hand-overs between two runs of the upkeep leave the one handed
/// last: the task takes it at its next run, and the monitor of the one
/// before, which it never followed, is not heard.
#[tokio::test(start_paused = true)]
async fn two_hand_overs_between_two_upkeeps_leave_the_last() {
    let engine = preset("least_load");
    let list = pool(4);
    let members = engine.pool_members(&list);
    let task = running(&engine).await;
    let (first, channel) = monitor(&reports(4, 2));
    engine.set_loads(channel);
    let (_second, channel) = monitor(&reports(4, 3));
    engine.set_loads(channel);
    sleep(Duration::from_secs(1)).await;
    for seed in 0..32 {
        let place = picked(&engine, &asks(seed, &members, None));
        assert_eq!(place, 3, "seed {seed}: the channel handed last");
    }
    first.publish(&reports(4, 0));
    settle().await;
    let place = picked(&engine, &asks(7, &members, None));
    assert_eq!(place, 3, "the first monitor was never followed");
    task.abort();
}

/// A decision reads the snapshot its request gives where it gives one, and
/// the engine's own otherwise: a host that still hands a snapshot is
/// answered by it, whatever the task holds.
#[tokio::test(start_paused = true)]
async fn a_requests_own_snapshot_is_read_ahead_of_the_engines() {
    let engine = preset("least_load");
    let list = pool(4);
    let members = engine.pool_members(&list);
    let (_feed, channel) = monitor(&reports(4, 2));
    engine.set_loads(channel);
    let task = running(&engine).await;
    let given = LoadSnapshot::from_loads_for_test(reports(4, 1).into_iter().collect());
    for seed in 0..32 {
        let own = picked(&engine, &asks(seed, &members, Some(Arc::clone(&given))));
        assert_eq!(own, 1, "seed {seed}: the request's snapshot");
        let kept = picked(&engine, &asks(seed, &members, None));
        assert_eq!(kept, 2, "seed {seed}: the engine's");
    }
    task.abort();
}

/// An engine whose document reads no load report keeps no channel and no
/// snapshot: its draws look no report up, so its list never converts one.
#[tokio::test(start_paused = true)]
async fn an_engine_whose_document_reads_no_report_keeps_no_snapshot() {
    let engine = preset("random");
    let list = pool(32);
    let members = engine.pool_members(&list);
    let (_feed, channel) = monitor(&reports(32, 2));
    engine.set_loads(channel);
    let task = running(&engine).await;
    assert!(engine.loads.newest().is_none(), "no channel is kept");
    for seed in 0..256 {
        let _ = picked(&engine, &asks(seed, &members, None));
    }
    assert_eq!(members.conversions(), 0, "no report was looked up");
    task.abort();
}

fn served(worker: WorkerId) -> Outcome {
    Outcome {
        side: Side::Single,
        served: worker,
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    }
}

/// The task runs the engine's upkeep with no call of its host: a session
/// key's pin is held through the rule's idle time and leaves the table at
/// the rule's sweep behind it, by the task's own pass.
#[tokio::test(start_paused = true)]
async fn the_task_runs_the_upkeep_with_no_call_of_its_host() {
    let engine = preset("manual");
    let list = pool(4);
    let members = engine.pool_members(&list);
    let task = running(&engine).await;
    let mut keyed = asks(51, &members, None);
    keyed.request.facts.session_key = Identity::hashed(51);
    let now = Instant::now();
    let mut first = DecisionGuard::decide_under(engine.as_ref(), GatewayId::LOCAL, &keyed, now)
        .expect("a decision");
    let pinned = first.take(0).expect("the first target");
    first.dispatched(now);
    first.completed(&served(pinned.id), now);
    drop(first);
    let document = engine.document.load_full();
    let sessions = document.spec.engine().sessions();
    assert_eq!(sessions.len(), 1);
    // The stock rule idles a key four hours and sweeps once a minute: the
    // key ends at the first sweep at or after its idle time, and the pass
    // that takes it out runs within a minute of that.
    sleep(Duration::from_secs(14_000)).await;
    assert_eq!(sessions.len(), 1, "inside the idle time");
    sleep(Duration::from_secs(600)).await;
    assert_eq!(sessions.len(), 0, "past its end");
    task.abort();
}

/// The stock sessions rule with its idle time and its sweep interval set,
/// as `generation`.
fn ruled(ttl_s: u32, sweep_s: u32, generation: u64) -> AppliedSpec {
    let mut document = PolicySpec::preset("manual").expect("a built-in preset");
    document.generation = generation;
    let rule = document.bindings[0].sessions.as_mut();
    let rule = rule.expect("the stock rule");
    (rule.ttl_s, rule.sweep_s) = (ttl_s, sweep_s);
    AppliedSpec::validate(&document).expect("the document validates")
}

/// The engine's table takes the applied document's two times and keeps its
/// keys, and the task's upkeep runs on behind the apply. A key pinned under
/// the document before ends at that document's sweep, on the hour, and the
/// table's own hourly pass is due half an hour later. Under the applied
/// document's interval of ten seconds the key leaves within ten seconds of
/// its end, by the task's own pass.
#[tokio::test(start_paused = true)]
async fn a_key_of_the_document_before_leaves_at_a_pass_of_the_applied_document() {
    const SECOND: u64 = 1_000_000;
    // Idle after a minute, swept once an hour.
    let engine = Arc::new(PolicyEngine::new(ruled(60, 3_600, 1)));
    // The task starts half an hour into the engine's clock, and the table's
    // first pass with it: its hourly passes fall on the half hours.
    sleep(Duration::from_secs(1_800)).await;
    let task = running(&engine).await;
    let list = pool(4);
    let members = engine.pool_members(&list);
    let mut keyed = asks(51, &members, None);
    keyed.request.facts.session_key = Identity::hashed(51);
    let now = tokio::time::Instant::now().into_std();
    let mut first = DecisionGuard::decide_under(engine.as_ref(), GatewayId::LOCAL, &keyed, now)
        .expect("a decision");
    let pinned = first.take(0).expect("the first target");
    first.dispatched(now);
    first.completed(&served(pinned.id), now);
    drop(first);
    let table = &engine.sessions;
    let times = || (table.sweep_micros(), table.absence_micros());
    assert_eq!(times(), (3_600 * SECOND, 60 * SECOND));

    // Idle after half a minute, swept every ten seconds.
    let two = ruled(30, 10, 2);
    engine.apply(two).expect("a higher generation");
    assert_eq!(
        times(),
        (10 * SECOND, 30 * SECOND),
        "the applied document's"
    );
    assert_eq!(table.len(), 1, "the table keeps its keys");
    sleep(Duration::from_secs(1_700)).await;
    assert_eq!(table.len(), 1, "ahead of the key's end, on the hour");
    sleep(Duration::from_secs(120)).await;
    assert_eq!(table.len(), 0, "gone at a pass of the applied document");
    task.abort();
}

/// The task's upkeep takes the step of the records of sent prefixes too:
/// the placements of a worker told gone wait in the record, and leave it at
/// the task's next runs, with no call of the host.
#[tokio::test(start_paused = true)]
async fn the_upkeep_takes_the_step_of_the_sent_prefixes() {
    let engine = preset("cache_aware");
    let list = pool(8);
    let members = engine.pool_members(&list);
    let task = running(&engine).await;
    let tokens: Vec<u32> = (0..1_024).collect();
    let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
    let mut sent = asks(7, &members, None);
    hashes.write_to(&mut sent.request.facts);
    let place = picked(&engine, &sent) as usize;
    let held = engine.affinity.placements();
    assert!(held > 0, "the prompt's blocks are recorded for its worker");
    engine.forget(list[place].url());
    assert_eq!(engine.affinity.placements(), held, "they wait for a sweep");
    sleep(Duration::from_secs(3)).await;
    assert_eq!(engine.affinity.placements(), 0, "taken out by the task");
    task.abort();
}

/// A panic in the upkeep is held and counted by itself, beside the
/// decisions' count, and the task goes on: the step that panics runs again
/// each second, and a snapshot published afterwards is still kept at hand.
#[tokio::test(start_paused = true)]
async fn a_panic_in_the_upkeep_is_counted_and_the_task_goes_on() {
    let engine = preset("least_load");
    let list = pool(4);
    let members = engine.pool_members(&list);
    let (feed, channel) = monitor(&reports(4, 2));
    engine.set_loads(channel);
    let step = Arc::clone(&engine);
    let task = spawned(async move { step.run_with(|_| panic!("the upkeep's step")).await }).await;
    assert_eq!(engine.upkeep_faults(), 1, "the first run");
    sleep(Duration::from_secs(3)).await;
    assert_eq!(engine.upkeep_faults(), 4, "one run a second");
    feed.publish(&reports(4, 0));
    settle().await;
    for seed in 0..32 {
        let place = picked(&engine, &asks(seed, &members, None));
        assert_eq!(place, 0, "seed {seed}: the shortest queue published last");
    }
    assert_eq!(engine.upkeep_faults(), 4, "one run a second and no more");
    assert_eq!(engine.faults(), 0, "no decision panicked");
    task.abort();
}

/// A task that was not run for a while runs its upkeep once for the seconds
/// it missed, dated behind them, and again a second later.
#[tokio::test(start_paused = true)]
async fn the_upkeep_runs_once_for_the_seconds_a_stall_missed() {
    let engine = preset("least_load");
    let (runs, dated) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    // The runtime's clock, which the case holds still and moves itself.
    let start = tokio::time::Instant::now().into_std();
    let (step, counted, last) = (Arc::clone(&engine), Arc::clone(&runs), Arc::clone(&dated));
    let count = move |now: Instant| {
        counted.fetch_add(1, Ordering::Relaxed);
        let since = now.saturating_duration_since(start);
        last.store(since.as_secs(), Ordering::Relaxed);
        SweepOutcome::default()
    };
    let task = spawned(async move { step.run_with(count).await }).await;
    assert_eq!(runs.load(Ordering::Relaxed), 1, "the first run");
    // Ten seconds pass with the task not run.
    advance(Duration::from_secs(10)).await;
    settle().await;
    assert_eq!(runs.load(Ordering::Relaxed), 2, "one run for ten seconds");
    assert_eq!(dated.load(Ordering::Relaxed), 10, "dated behind them");
    sleep(Duration::from_secs(1)).await;
    assert_eq!(runs.load(Ordering::Relaxed), 3, "and one a second later");
    task.abort();
}

/// A monitor that ended does not end the task: the upkeep runs on behind
/// it, and a channel handed afterwards is read within a second. A host
/// builds another monitor where its first one ended.
#[tokio::test(start_paused = true)]
async fn the_task_outlives_its_monitor_and_reads_the_next_one() {
    let engine = preset("least_load");
    let list = pool(4);
    let members = engine.pool_members(&list);
    let (first, channel) = monitor(&reports(4, 2));
    engine.set_loads(channel);
    let runs = Arc::new(AtomicU64::new(0));
    let (step, counted) = (Arc::clone(&engine), Arc::clone(&runs));
    let count = move |_: Instant| {
        counted.fetch_add(1, Ordering::Relaxed);
        SweepOutcome::default()
    };
    let task = spawned(async move { step.run_with(count).await }).await;
    assert_eq!(picked(&engine, &asks(7, &members, None)), 2);
    drop(first);
    settle().await;
    let before = runs.load(Ordering::Relaxed);
    sleep(Duration::from_secs(3)).await;
    let behind = runs.load(Ordering::Relaxed);
    assert_eq!(behind, before + 3, "the upkeep runs on behind the monitor");
    let (_second, channel) = monitor(&reports(4, 3));
    engine.set_loads(channel);
    sleep(Duration::from_secs(1)).await;
    for seed in 0..32 {
        let place = picked(&engine, &asks(seed, &members, None));
        assert_eq!(place, 3, "seed {seed}: the monitor built afterwards");
    }
    task.abort();
}
