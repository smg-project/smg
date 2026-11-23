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
    task::{yield_now, JoinHandle},
    time::{advance, sleep},
};

use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision_engine::{DecisionEngine, Request, RequestMaker, WorkerLists},
        list_cell::tests::PublishedList,
        policy_engine::{
            membership::tests::{handed, id, in_id_order, registry, told},
            single::tests::hearing,
            PolicyEngine,
        },
        view::{Constraints, Identity, RequestFacts, WorkerId},
        PlaceRequest,
    },
    store::GatewayId,
    tracking::{
        lease::{Outcome, Side, TransportClass},
        DecisionGuard,
    },
};

pub(in crate::engine::policy_engine) const MODEL: &str = "model-a";

/// The engine of a stock document, as a host that spawns its task holds it.
fn preset(name: &str) -> Arc<PolicyEngine> {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    Arc::new(PolicyEngine::new(spec))
}

pub(in crate::engine::policy_engine) fn url(at: usize) -> String {
    format!("http://worker-{at}:8000")
}

fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
    (0..workers)
        .map(|at| ready_worker(&url(at), MODEL))
        .collect()
}

/// One report a worker: the worker at `short` has no queue, every other a
/// long one.
pub(in crate::engine::policy_engine) fn reports(
    workers: usize,
    short: usize,
) -> HashMap<String, WorkerLoadResponse> {
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
pub(in crate::engine::policy_engine) fn monitor(
    reports: &HashMap<String, WorkerLoadResponse>,
) -> (TestFeed, LoadReceiver) {
    let mut channel = None;
    let feed = TestFeed::attached_to(|loads| channel = loads);
    feed.publish(reports);
    let mut channel = channel.expect("the feed's channel");
    channel.mark_unchanged();
    (feed, channel)
}

/// A request stated ahead over one list, with the snapshot its host gives
/// it, where it gives one.
pub(in crate::engine::policy_engine) struct Asks<'a> {
    pub(in crate::engine::policy_engine) request: Request<'a>,
    members: &'a PublishedList,
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
        WorkerLists::Single(self.members.lent())
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.loads.as_ref()
    }
}

/// A request of `seed` over `members` that gives `loads`.
pub(in crate::engine::policy_engine) fn asks(
    seed: u64,
    members: &PublishedList,
    loads: Option<Arc<LoadSnapshot>>,
) -> Asks<'_> {
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
pub(in crate::engine::policy_engine) fn picked(engine: &PolicyEngine, asks: &Asks<'_>) -> u32 {
    let decided = engine.decide_under(GatewayId::LOCAL, asks, Instant::now());
    let whole = decided.expect("an answer").whole();
    let decision = whole.expect("a whole decision").0.decision;
    let pick = decision.pick().expect("a pick");
    pick.targets.first().expect("a target").position
}

/// Lets the task take what stands ready for it: no second passes.
pub(in crate::engine::policy_engine) async fn settle() {
    sleep(Duration::from_millis(1)).await;
}

/// A task spawned as a host spawns the engine's, past its first step.
pub(in crate::engine::policy_engine) async fn spawned(
    task: impl Future<Output = ()> + Send + 'static,
) -> JoinHandle<()> {
    #[expect(
        clippy::disallowed_methods,
        reason = "the case holds the task's handle and ends the task"
    )]
    let task = tokio::spawn(task);
    settle().await;
    task
}

/// The engine's task, spawned as a host spawns it, past its first step.
pub(in crate::engine::policy_engine) async fn running(
    engine: &Arc<PolicyEngine>,
) -> JoinHandle<()> {
    let engine = Arc::clone(engine);
    spawned(async move { engine.run().await }).await
}

/// The engine's task, spawned as a host spawns it and polled once, with
/// the count of its upkeep's runs. The clock is first moved off a whole
/// millisecond, where a program's clock stands: a tick due at once still
/// waits there on the runtime's timer, and one yield does not reach it.
async fn polled_once(engine: &Arc<PolicyEngine>) -> (JoinHandle<()>, Arc<AtomicU64>) {
    advance(Duration::from_micros(500)).await;
    let runs = Arc::new(AtomicU64::new(0));
    let (step, counted) = (Arc::clone(engine), Arc::clone(&runs));
    let count = move |_: Instant| {
        counted.fetch_add(1, Ordering::Relaxed);
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "the case holds the task's handle and ends the task"
    )]
    let task = tokio::spawn(async move { step.run_with(count).await });
    yield_now().await;
    (task, runs)
}

/// The task's first upkeep runs at its first poll, and with it the task
/// reads the registry its host handed: a host that spawns the task and
/// yields once has stages that were told the workers that stand.
#[tokio::test(start_paused = true)]
async fn the_task_tells_the_workers_that_stand_at_its_first_poll() {
    let (engine, heard) = hearing(false);
    let engine = Arc::new(engine);
    let registry = registry(2);
    engine.set_registry(handed(&registry));
    let (task, runs) = polled_once(&engine).await;
    assert_eq!(runs.load(Ordering::Relaxed), 1, "the first upkeep");
    let stand = vec![("joined", id(0)), ("joined", id(1))];
    assert_eq!(in_id_order(told(&heard)), in_id_order(stand));
    task.abort();
}

/// The task takes the load channel its host handed at its first poll: a
/// host that spawns the task and yields once has an engine that reads the
/// channel's reports for a request that gives none.
#[tokio::test(start_paused = true)]
async fn the_task_takes_the_load_channel_at_its_first_poll() {
    let engine = preset("least_load");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let (_feed, channel) = monitor(&reports(4, 2));
    engine.set_loads(channel);
    let (task, runs) = polled_once(&engine).await;
    assert_eq!(runs.load(Ordering::Relaxed), 1, "the first upkeep");
    for seed in 0..32 {
        let place = picked(&engine, &asks(seed, &members, None));
        assert_eq!(place, 2, "seed {seed}: the shortest queue handed");
    }
    task.abort();
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
    let members = PublishedList::new(Arc::clone(&list));
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
    let members = PublishedList::new(Arc::clone(&list));
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
    let members = PublishedList::new(Arc::clone(&list));
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

/// A decision reads the snapshot its request gives where it gives one, and
/// the engine's own otherwise: a host that still hands a snapshot is
/// answered by it, whatever the task holds.
#[tokio::test(start_paused = true)]
async fn a_requests_own_snapshot_is_read_ahead_of_the_engines() {
    let engine = preset("least_load");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
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

/// An engine whose document reads no load report and no named signal, and
/// none of whose stages keeps a report, keeps no channel and no snapshot:
/// its draws look no report up, so its list never converts one.
#[tokio::test(start_paused = true)]
async fn an_engine_whose_document_reads_no_report_keeps_no_snapshot() {
    let engine = preset("random");
    let list = pool(32);
    let members = PublishedList::new(Arc::clone(&list));
    let (_feed, channel) = monitor(&reports(32, 2));
    engine.set_loads(channel);
    let task = running(&engine).await;
    assert!(engine.loads.feed.lock().1.is_none(), "no channel is kept");
    assert!(engine.loads.newest().is_none(), "and no snapshot");
    for seed in 0..256 {
        let _ = picked(&engine, &asks(seed, &members, None));
    }
    let converted = engine.conversions(members.lent());
    assert_eq!(converted, 0, "no report was looked up");
    task.abort();
}

/// The engine of a document that reads two named signals and no load
/// report: its one rule keeps the workers whose report says they have
/// room.
fn by_signals_alone() -> Arc<PolicyEngine> {
    let document: PolicySpec = serde_json::from_str(
        r#"{"apiVersion":"smg.policy/v1","generation":1,
            "signals":{"kv_tokens_capacity":{"unit":"tokens"},
                       "kv_tokens_used":{"unit":"tokens"}},
            "profiles":{"p":{"filter":["healthy",{"has-room":{
                "used":{"signal.kv_tokens_used":1},
                "capacity":"signal.kv_tokens_capacity"}}],"pick":"random"}}}"#,
    )
    .expect("a well-formed document");
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let needs = spec.needs();
    assert!(needs.signal_cells && !needs.loads, "signals and no report");
    Arc::new(PolicyEngine::new(spec))
}

/// One report a worker, each stating the two token counts a report offers
/// by name: the worker at `roomy` has room, every other is full.
fn rooms(workers: usize, roomy: usize) -> HashMap<String, WorkerLoadResponse> {
    (0..workers)
        .map(|at| {
            let mut report = one_rank_load(0, 0.1, 1_000.0);
            for rank in &mut report.loads {
                rank.num_used_tokens = if at == roomy { 100 } else { 2_000 };
                rank.max_total_num_tokens = 1_000;
            }
            (url(at), report)
        })
        .collect()
}

/// An engine whose document reads a named signal and no load report keeps
/// its host's channel: a signal's values stand in the same reports. A
/// decision whose request gives no snapshot reads them in the newest the
/// task keeps, and one whose request gives a snapshot reads them there.
#[tokio::test(start_paused = true)]
async fn an_engine_whose_document_reads_a_signal_keeps_the_channel() {
    let engine = by_signals_alone();
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let (feed, channel) = monitor(&rooms(4, 2));
    engine.set_loads(channel);
    let task = running(&engine).await;
    assert!(engine.loads.newest().is_some(), "the channel is kept");
    for seed in 0..32 {
        let place = picked(&engine, &asks(seed, &members, None));
        assert_eq!(place, 2, "seed {seed}: the one worker with room");
    }
    feed.publish(&rooms(4, 0));
    settle().await;
    let given = LoadSnapshot::from_loads_for_test(rooms(4, 1).into_iter().collect());
    for seed in 0..32 {
        let kept = picked(&engine, &asks(seed, &members, None));
        assert_eq!(kept, 0, "seed {seed}: the one with room published last");
        let own = picked(&engine, &asks(seed, &members, Some(Arc::clone(&given))));
        assert_eq!(own, 1, "seed {seed}: the request's snapshot");
    }
    task.abort();
}

/// A request placed with the engine's one call lends no snapshot and no
/// cells ([`PolicyEngine::place`]). Its decision reads the named values in
/// the newest snapshot the task keeps: the one worker whose report says it
/// has room takes every request, and the next publication moves them.
#[tokio::test(start_paused = true)]
async fn a_request_placed_with_one_call_reads_the_signals_the_task_keeps() {
    let engine: &'static Arc<PolicyEngine> = Box::leak(Box::new(by_signals_alone()));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let (feed, channel) = monitor(&rooms(4, 2));
    engine.set_loads(channel);
    let task = running(engine).await;
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    for turn in 0..32 {
        let placement = engine.place(&request).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[2]), "turn {turn}");
    }
    feed.publish(&rooms(4, 0));
    settle().await;
    for turn in 0..32 {
        let placement = engine.place(&request).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[0]), "turn {turn}");
    }
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
    let members = PublishedList::new(Arc::clone(&list));
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
    let sessions = engine.spec().engine().sessions();
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

/// A panic in the upkeep is held and counted by itself, beside the
/// decisions' count, and the task goes on: the step that panics runs again
/// each second, and a snapshot published afterwards is still kept at hand.
#[tokio::test(start_paused = true)]
async fn a_panic_in_the_upkeep_is_counted_and_the_task_goes_on() {
    let engine = preset("least_load");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
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
    let members = PublishedList::new(Arc::clone(&list));
    let (first, channel) = monitor(&reports(4, 2));
    engine.set_loads(channel);
    let runs = Arc::new(AtomicU64::new(0));
    let (step, counted) = (Arc::clone(&engine), Arc::clone(&runs));
    let count = move |_: Instant| {
        counted.fetch_add(1, Ordering::Relaxed);
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
