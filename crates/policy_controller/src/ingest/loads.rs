//! The load path driver (design 5.1, the loads row; 6, the ingestion): the
//! `GetLoads` poll of every engine whose stream is not serving it, under
//! the store's poll arbitration, and the metrics scrape of an engine
//! without a servicer, both written to the store's [`LoadSink`].
//!
//! One scheduler tick per second reads the membership (regrouped only when
//! the store's epoch moved) and starts what is due: an engine's poll every
//! `poll_interval`, its scrape every `scrape_interval_ms` of its row. A tick
//! waits for no engine: each call runs under its own deadline, its answer
//! is written when it lands, and an engine with a call out is not asked
//! again before, so one engine's silence never delays another engine's
//! sample. A poll asks the engine once, at the engine's own address, and
//! writes one row per rank whose own records are not serving it, each
//! rank's figures scoped from the answer. A scrape asks the metrics of an
//! engine of one rank and writes that rank's row; the gauges are
//! engine-wide, so a data-parallel engine without a servicer is not scraped
//! and its ranks have no row until a per-rank source names them: a row
//! holds a figure measured for its rank, never an aggregate. A failed read
//! writes no row: the row on file ages, the engine's overload verdict
//! reads clear until the engine is heard again unless a rank of it is
//! still heard on its stream, and health is the prober's (a later PR). A
//! read that lands after its worker left writes nothing: the store keeps
//! a worker's load rows exactly as long as its row.
//! A row reachable only through a gateway is never dialed: its records
//! arrive relayed. The engines are asked through a [`LoadProbe`], so the
//! tests drive the arbitration without a socket and the binary dials the
//! engines.
//!
//! Where it is built the driver tells the store the period every ready
//! rank is heard in, by its stream or by the poll: the longer of the poll
//! interval and the poll's deadline, plus a tick. The store ends the
//! sample of a rank that lies three such periods behind its engine's
//! newest ([`LoadSink::heard_every`]).
//!
//! The loop also keeps the instant the store names for its publication
//! ([`LoadSink::publish_due`]): a row that lands inside the store's window
//! is shown to decisions at the window's end, with no later write.

use std::{
    collections::{BTreeMap, HashSet},
    future::Future,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use openai_protocol::worker::{RuntimeType, WorkerLoadResponse};
use parking_lot::Mutex;
use tokio::{
    sync::watch,
    task::{JoinError, JoinSet},
    time::MissedTickBehavior,
};
use tracing::{debug, warn};

use super::{rows::load_row, scrape};
use crate::{
    store::{LoadSink, LoadSource, PollMode, Reachability, WorkerRow, WorkerStatus, WorkersStore},
    view::{Role, SignalSource, WireModes, WorkerId},
};

/// The driver's timings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadDriverConfig {
    /// How often an engine is polled while its stream is not serving it;
    /// also the freshness bound of the arbitration (a rank whose newest
    /// record is older than this is polled), and with a tick the period
    /// every ready rank is heard in, unless the poll's deadline is longer
    /// ([`Self::report_period`]). The gateway's
    /// `--load-monitor-interval`, 10 s. An interval the clock cannot add to
    /// the present polls an engine once.
    pub poll_interval: Duration,
    /// The bound on one poll, dial included: how long an engine's one call
    /// stays out before the engine can be asked again. A poll past it is a
    /// slow engine, not a dead one.
    pub poll_deadline: Duration,
    /// The bound on one scrape.
    pub scrape_timeout: Duration,
    /// The scheduler's tick: the resolution of every interval above.
    pub tick: Duration,
}

impl LoadDriverConfig {
    /// The period the driver tells the sink ([`LoadSink::heard_every`]),
    /// true of the slowest read of a sample that can end by its age: the
    /// longer of the poll interval and the poll's deadline, plus a tick.
    /// A poll starts on a tick, so two starts of an engine's polls lie at
    /// most that far apart, and a poll that takes its whole deadline
    /// lands inside the period it is counted in. A rank that loses one
    /// report is so heard again within two periods and a deadline, inside
    /// the three a sample is kept through. The sum saturates at the
    /// largest interval or deadline a host can set. A sample is aged
    /// against its engine's newest, and a scrape reads only an engine of
    /// one rank, whose sample is its engine's newest at any age: the
    /// scrape's timeout is not in the period.
    pub fn report_period(&self) -> Duration {
        self.poll_interval
            .max(self.poll_deadline)
            .saturating_add(self.tick)
    }
}

impl Default for LoadDriverConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(10),
            poll_deadline: Duration::from_secs(3),
            scrape_timeout: Duration::from_secs(5),
            tick: Duration::from_secs(1),
        }
    }
}

/// Where a poll goes: the engine's dialable address and its family, which
/// picks the gRPC client.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    /// The engine's address as the store holds it (`WorkerFacts::engine`,
    /// one per engine and shared by the rows of its ranks), which answers
    /// once for every rank. A rank row's own key is not dialed.
    pub endpoint: String,
    pub kind: RuntimeType,
}

/// Why a read produced no report. The variant is the failure's label.
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    /// The dial failed or the connection was lost.
    #[error("transport: {0}")]
    Transport(String),
    /// The engine answered the call with a status.
    #[error("the engine answered: {0}")]
    Answer(String),
    /// The deadline passed.
    #[error("no answer within {0:?}")]
    Deadline(Duration),
    /// The metrics body carried no load for this family.
    #[error("no load in the metrics of a {0} engine")]
    NoLoad(String),
    /// The family has no `GetLoads`.
    #[error("no GetLoads on a {0} engine")]
    Unsupported(String),
}

impl ReadError {
    /// The failure's label, for the counters.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Transport(_) => "transport",
            Self::Answer(_) => "answer",
            Self::Deadline(_) => "deadline",
            Self::NoLoad(_) => "no_load",
            Self::Unsupported(_) => "unsupported",
        }
    }
}

/// What the driver asks of an engine.
pub trait LoadProbe: Send + Sync + 'static {
    /// The engine's `GetLoads` answer, every rank as the engine reports it,
    /// within `deadline`.
    fn poll(
        &self,
        target: &Target,
        deadline: Duration,
    ) -> impl Future<Output = Result<WorkerLoadResponse, ReadError>> + Send;

    /// The text of the engine's metrics endpoint, within `timeout`.
    fn scrape(
        &self,
        url: &str,
        timeout: Duration,
    ) -> impl Future<Output = Result<String, ReadError>> + Send;

    /// The engine left the membership: whatever the probe holds for it goes.
    fn forget(&self, endpoint: &str);
}

/// What the driver did, as counters the metrics render.
#[derive(Debug, Default)]
pub struct Stats {
    /// Polls decided `poll` (no record on file).
    pub polls: AtomicU64,
    /// Polls decided `fallback` (a rank's records went quiet).
    pub fallbacks: AtomicU64,
    /// Polls skipped because every rank's stream was fresh, one per engine
    /// and poll interval: the arbitration is asked at every tick while a
    /// poll is due, and the count moves on the scale of the two above.
    pub suppressed: AtomicU64,
    /// Polls that produced no report, by label.
    pub poll_failures: Failures,
    /// Scrapes that produced a row.
    pub scrapes: AtomicU64,
    /// Scrapes that produced no row, by label.
    pub scrape_failures: Failures,
    /// Rank rows handed to the store by polls and scrapes. A rank whose own
    /// record arrived while the read was out keeps that record, so the rows
    /// on file can be fewer.
    pub rows: AtomicU64,
}

/// Failed reads by [`ReadError::label`].
#[derive(Debug, Default)]
pub struct Failures {
    pub transport: AtomicU64,
    pub answer: AtomicU64,
    pub deadline: AtomicU64,
    pub no_load: AtomicU64,
    pub unsupported: AtomicU64,
}

impl Failures {
    /// The labels of the series, in the order of [`Self::each`]: each is its
    /// variant's [`ReadError::label`].
    pub const LABELS: [&'static str; 5] =
        ["transport", "answer", "deadline", "no_load", "unsupported"];

    /// The counters, in the order of [`Self::LABELS`].
    fn counters(&self) -> [&AtomicU64; 5] {
        [
            &self.transport,
            &self.answer,
            &self.deadline,
            &self.no_load,
            &self.unsupported,
        ]
    }

    /// Counts one failure under its variant: a variant added to the error
    /// has no counter until this match names one, and does not compile.
    fn count(&self, error: &ReadError) {
        let counter = match error {
            ReadError::Transport(_) => &self.transport,
            ReadError::Answer(_) => &self.answer,
            ReadError::Deadline(_) => &self.deadline,
            ReadError::NoLoad(_) => &self.no_load,
            ReadError::Unsupported(_) => &self.unsupported,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// The counts, with their labels.
    pub fn each(&self) -> [(&'static str, u64); 5] {
        let mut out = [("", 0); 5];
        for (slot, (counter, name)) in out
            .iter_mut()
            .zip(self.counters().into_iter().zip(Self::LABELS))
        {
            *slot = (name, counter.load(Ordering::Relaxed));
        }
        out
    }
}

/// The driver over a store and a probe.
pub struct LoadDriver<S, P> {
    store: Arc<S>,
    probe: Arc<P>,
    config: LoadDriverConfig,
    stats: Arc<Stats>,
    /// The engines the probe was asked about and not told to forget since:
    /// it may hold a connection for each. The set outlives a run, as the
    /// probe does, so an engine that leaves the membership between two
    /// runs is let go at the next run's first tick.
    asked: Mutex<HashSet<String>>,
}

/// The engines of one membership epoch.
struct Grouped {
    epoch: u64,
    engines: Arc<Vec<Engine>>,
}

/// One engine's rows as the driver reads them from the membership.
#[derive(Debug)]
struct Engine {
    key: String,
    role: Role,
    /// The family, when the handshake named one the host knows.
    kind: Option<RuntimeType>,
    /// The engine's data-parallel width from its rows' facts: what decides
    /// whether an answer is scoped per rank, whatever rows are ready.
    dp_size: u16,
    /// Where a poll goes: the engine's address, when a ready row every
    /// gateway dials speaks gRPC.
    poll: Option<Target>,
    /// Where a scrape goes, and how often: an HTTP engine of one rank
    /// without a poll (no servicer) with a scrape interval on its row.
    scrape: Option<(String, Duration)>,
    /// The ready rows, by rank.
    ranks: Vec<(u16, WorkerId)>,
}

/// What a run of polls keeps between its ticks: the membership as last
/// grouped, when each engine is next due, and the calls that are out.
#[derive(Default)]
struct Schedule {
    /// The membership grouped by engine, kept until the store's epoch
    /// moves. An epoch names one membership within a run of polls alone:
    /// between two runs a restore may leave the store at the same epoch
    /// with another, so a run of polls starts with no grouping.
    grouped: Option<Grouped>,
    due: BTreeMap<String, Due>,
    /// The calls that have not answered yet. A tick starts calls and waits
    /// for none: an answer is written when it lands, and dropping the
    /// schedule (the loop's end) cancels what is still out.
    calls: JoinSet<Done>,
}

/// When an engine's next poll and scrape are due, and whether its call is
/// out.
struct Due {
    /// `None` after a poll whose interval the clock cannot add to the
    /// present: the engine is not polled again.
    poll: Option<tokio::time::Instant>,
    scrape: tokio::time::Instant,
    /// When a suppressed poll counts next (see [`Stats::suppressed`]);
    /// `None` as for `poll`: it counted once.
    suppressed: Option<tokio::time::Instant>,
    /// The engine's one call (its poll, or its scrape when it has no
    /// servicer) is out until this instant, the call's deadline: set when
    /// the call goes, cleared when its answer lands. The engine is not asked
    /// again before, so a slow engine holds one call, never a queue of them.
    out: Option<tokio::time::Instant>,
}

impl Due {
    /// An engine's first tick: everything due at `at`, nothing out.
    fn at(at: tokio::time::Instant) -> Self {
        Self {
            poll: Some(at),
            scrape: at,
            suppressed: Some(at),
            out: None,
        }
    }

    /// Whether the engine's call is still out at `at`.
    fn is_out(&self, at: tokio::time::Instant) -> bool {
        self.out.is_some_and(|until| at < until)
    }
}

/// One call's outcome, written when it lands.
struct Done {
    /// The membership the call was made under, and its engine's place in
    /// it.
    engines: Arc<Vec<Engine>>,
    engine: usize,
    /// The engine's out-mark this call set: the landing clears that mark
    /// and no later call's.
    mark: tokio::time::Instant,
    /// When the call began, for the store's pushed-since rule.
    polled_at: Instant,
    read: Read,
}

/// What a call read.
enum Read {
    Poll {
        /// The ranks the arbitration sent the poll for.
        ranks: Vec<(u16, WorkerId)>,
        /// The arbitration left a ready rank of the engine out: its own
        /// records serve it, by the freshness read when the poll went.
        served: bool,
        answer: Result<WorkerLoadResponse, ReadError>,
    },
    Scrape {
        text: Result<String, ReadError>,
    },
}

/// The host's clock through tokio's, so a test under paused time moves
/// every instant the store compares.
fn now() -> Instant {
    tokio::time::Instant::now().into_std()
}

/// Wait for the instant; for ever when there is none.
async fn at(instant: Option<Instant>) {
    match instant {
        Some(instant) => tokio::time::sleep_until(tokio::time::Instant::from_std(instant)).await,
        None => std::future::pending().await,
    }
}

/// The gate's next word; for ever once its teller is gone.
async fn turned(polls: &mut watch::Receiver<bool>) -> bool {
    if polls.changed().await.is_err() {
        std::future::pending::<()>().await;
    }
    *polls.borrow_and_update()
}

/// What a log line calls an engine: the host and port of its address, never
/// the address whole (a worker URL may carry userinfo, which no log line
/// gets; the gRPC client names its peers the same way). An address with an
/// `@` past its first `/`, `?` or `#` is named by the marker the gateway
/// writes for an address it cannot parse: a userinfo that holds one of the
/// three raw reads like a path or a query that holds an `@`, and a guess
/// could print part of the userinfo.
fn peer(engine: &str) -> &str {
    let rest = engine
        .split_once("://")
        .map_or(engine, |(_scheme, rest)| rest);
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(end);
    if tail.contains('@') {
        return "<unparsable address>";
    }
    authority
        .rsplit_once('@')
        .map_or(authority, |(_userinfo, host)| host)
}

impl<S, P> LoadDriver<S, P>
where
    S: WorkersStore + LoadSink + 'static,
    P: LoadProbe,
{
    /// A driver over `store`. The store is told the report period
    /// ([`LoadDriverConfig::report_period`]) here, where the driver is built
    /// and not at its first poll: a store is aged by the period from its
    /// first judged write.
    pub fn new(store: Arc<S>, probe: Arc<P>, config: LoadDriverConfig) -> Self {
        let period = config.report_period();
        if !store.heard_every(period) {
            warn!(
                ?period,
                "the load sink keeps the report period it was told first"
            );
        }
        Self {
            store,
            probe,
            config,
            stats: Arc::new(Stats::default()),
            asked: Mutex::new(HashSet::new()),
        }
    }

    /// The counters, shared with whoever renders them.
    pub fn stats(&self) -> Arc<Stats> {
        Arc::clone(&self.stats)
    }

    /// Ticks until `stop` reads `true`. A tick starts what is due and waits
    /// for no engine: each call runs under its own deadline and its answer
    /// is written when it lands, so an engine that does not answer delays
    /// its own rows alone, never the fleet's or the stop, which cancels the
    /// calls still out.
    pub async fn run(&self, stop: watch::Receiver<bool>) {
        // No role to read: the polls run for the loop's length.
        let (_always, polls) = watch::channel(true);
        self.run_while(stop, polls).await;
    }

    /// [`LoadDriver::run`] on a replica that may follow a leader: the loop
    /// runs for the process's life and polls only while `polls` reads
    /// `true`. It pays the store's publications either way, whoever wrote
    /// the rows: a follower's applier writes them, and the instant the
    /// store names is read again at each tick. A turn to `false` ends the
    /// run of polls, with its calls still out and its grouping of the
    /// membership; a turn to `true` starts the next with a tick at once.
    pub async fn run_while(
        &self,
        mut stop: watch::Receiver<bool>,
        mut polls: watch::Receiver<bool>,
    ) {
        let mut ticker = tokio::time::interval(self.config.tick);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut schedule = Schedule::default();
        let mut polling = *polls.borrow_and_update();
        loop {
            // The instant the store names for the rows its publication
            // window still holds, read after every write of this loop: the
            // last row of a burst is shown within one window of its
            // receipt, with no later write. A row another writer brought,
            // a follower's applier, is read at the next tick.
            let due = self.store.publish_due();
            // The stop first; then the answers before the tick, which finds
            // the engines whose calls landed free to be asked again.
            tokio::select! {
                biased;
                _ = stop.wait_for(|stop| *stop) => return,
                word = turned(&mut polls) => {
                    if word != polling {
                        polling = word;
                        schedule = Schedule::default();
                        if polling {
                            self.tick(&mut schedule);
                        }
                    }
                }
                Some(joined) = schedule.calls.join_next() => self.land(&mut schedule.due, joined),
                () = at(due) => self.store.publish(now()),
                _ = ticker.tick() => {
                    if polling {
                        self.tick(&mut schedule);
                    }
                }
            }
        }
    }

    /// The membership grouped by engine: the snapshot is read and regrouped
    /// only when the store's epoch moved since the run's last tick (every
    /// write, a status change included, moves it).
    fn engines(&self, grouped: &mut Option<Grouped>) -> Arc<Vec<Engine>> {
        let epoch = self.store.epoch();
        if let Some(kept) = grouped.as_ref() {
            if kept.epoch == epoch {
                return Arc::clone(&kept.engines);
            }
        }
        let (epoch, rows) = self.store.snapshot();
        let engines = Arc::new(engines(rows));
        *grouped = Some(Grouped {
            epoch,
            engines: Arc::clone(&engines),
        });
        engines
    }

    /// One tick: the membership read, the engines that left dropped, the due
    /// calls started. Nothing here waits on an engine.
    fn tick(&self, schedule: &mut Schedule) {
        let engines = self.engines(&mut schedule.grouped);
        let tick_at = tokio::time::Instant::now();
        let live: HashSet<&str> = engines.iter().map(|engine| engine.key.as_str()).collect();
        schedule.due.retain(|key, _| live.contains(key.as_str()));
        let mut asked = self.asked.lock();
        asked.retain(|key| {
            let kept = live.contains(key.as_str());
            if !kept {
                self.probe.forget(key);
            }
            kept
        });
        for (index, engine) in engines.iter().enumerate() {
            if !asked.contains(&engine.key) {
                asked.insert(engine.key.clone());
            }
            let slot = schedule
                .due
                .entry(engine.key.clone())
                .or_insert_with(|| Due::at(tick_at));
            if let Some(target) = &engine.poll {
                // The arbitration is asked at every tick the poll is due and
                // the due instant moves only when a call goes: an engine whose
                // ranks are all served by their stream is asked again at the
                // next tick, so a rank whose records stop is polled within
                // one tick of the freshness bound, not one interval later.
                if slot.poll.is_some_and(|due| tick_at >= due) && !slot.is_out(tick_at) {
                    if let Some(ranks) = self.ranks_to_poll(engine) {
                        let served = ranks.len() < engine.ranks.len();
                        let mark = tick_at + self.config.poll_deadline;
                        slot.poll = tick_at.checked_add(self.config.poll_interval);
                        slot.suppressed = slot.poll;
                        slot.out = Some(mark);
                        let probe = Arc::clone(&self.probe);
                        let target = target.clone();
                        let deadline = self.config.poll_deadline;
                        let engines = Arc::clone(&engines);
                        let polled_at = now();
                        schedule.calls.spawn(async move {
                            let answer = probe.poll(&target, deadline).await;
                            Done {
                                engines,
                                engine: index,
                                mark,
                                polled_at,
                                read: Read::Poll {
                                    ranks,
                                    served,
                                    answer,
                                },
                            }
                        });
                    } else if slot.suppressed.is_some_and(|due| tick_at >= due) {
                        // Asked at every tick, counted once an interval: the
                        // scale of the polls that go.
                        slot.suppressed = tick_at.checked_add(self.config.poll_interval);
                        self.stats.suppressed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            if let Some((url, every)) = &engine.scrape {
                // The arbitration first, as for a poll, and the due instant
                // moves only when a call goes: an engine whose rank is
                // served by its own records is not scraped, and is scraped
                // within one tick of the freshness bound once they stop.
                if tick_at >= slot.scrape && !slot.is_out(tick_at) && self.scrape_wanted(engine) {
                    let mark = tick_at + self.config.scrape_timeout;
                    slot.scrape = tick_at + *every;
                    slot.out = Some(mark);
                    let probe = Arc::clone(&self.probe);
                    let url = url.clone();
                    let timeout = self.config.scrape_timeout;
                    let engines = Arc::clone(&engines);
                    let polled_at = now();
                    schedule.calls.spawn(async move {
                        let text = probe.scrape(&url, timeout).await;
                        Done {
                            engines,
                            engine: index,
                            mark,
                            polled_at,
                            read: Read::Scrape { text },
                        }
                    });
                }
            }
        }
    }

    /// One call landed: its engine is free to be asked again and its answer
    /// is written. A call that ended abnormally frees its engine when the
    /// mark it set passes, at the call's deadline. A call whose engine left
    /// while it was out has the probe forget the engine once more: the
    /// connection its dial made after the first forget would stay for good.
    fn land(&self, due: &mut BTreeMap<String, Due>, joined: Result<Done, JoinError>) {
        let done = match joined {
            Ok(done) => done,
            Err(join) => {
                warn!(%join, "a load read ended abnormally");
                return;
            }
        };
        let Some(engine) = done.engines.get(done.engine) else {
            return;
        };
        match due.get_mut(&engine.key) {
            Some(slot) => {
                if slot.out == Some(done.mark) {
                    slot.out = None;
                }
            }
            // The engine left while its call was out. The tick that dropped
            // it told the probe to forget it, before the call's own dial had
            // anything to forget: what the call left behind goes now, and no
            // later tick would name the engine again.
            None => self.probe.forget(&engine.key),
        }
        self.apply(engine, done.polled_at, done.read);
    }

    /// The ranks a poll is for: every ready rank whose own records are not
    /// serving it. `None` when the stream serves them all, and no call goes.
    fn ranks_to_poll(&self, engine: &Engine) -> Option<Vec<(u16, WorkerId)>> {
        let now = now();
        let mut wanted = Vec::with_capacity(engine.ranks.len());
        let mut fallback = false;
        for &(rank, id) in &engine.ranks {
            match self.store.poll_mode(id, now, self.config.poll_interval) {
                PollMode::Suppressed => {}
                PollMode::Fallback => {
                    fallback = true;
                    wanted.push((rank, id));
                }
                PollMode::Poll => wanted.push((rank, id)),
            }
        }
        if wanted.is_empty() {
            return None;
        }
        let counter = if fallback {
            &self.stats.fallbacks
        } else {
            &self.stats.polls
        };
        counter.fetch_add(1, Ordering::Relaxed);
        Some(wanted)
    }

    /// Whether a scrape is for anyone: a rank of the engine that its own
    /// records are not serving. Only an engine of one rank is scraped, so
    /// this asks that rank.
    fn scrape_wanted(&self, engine: &Engine) -> bool {
        let now = now();
        engine.ranks.iter().any(|&(_, id)| {
            self.store.poll_mode(id, now, self.config.poll_interval) != PollMode::Suppressed
        })
    }

    /// Writes one call's answer in one write: a poll's rows for the ranks
    /// it was for, a scrape's sample for its engine's rank unless the
    /// rank's own records are fresher. A failure counts. A failed poll and
    /// an answer with no row are written as the word that no report came
    /// when the poll was for each ready rank of the engine: a poll that
    /// left out a rank its own records serve says nothing of the engine,
    /// which is heard. The store drops a row for a worker that left while
    /// the call was out.
    fn apply(&self, engine: &Engine, polled_at: Instant, read: Read) {
        match read {
            Read::Poll {
                ranks,
                served,
                answer,
            } => {
                let report = match answer {
                    Ok(report) => report,
                    Err(error) => {
                        debug!(engine = peer(&engine.key), %error, "the load poll produced no report");
                        self.stats.poll_failures.count(&error);
                        if !served {
                            let none: Vec<_> =
                                ranks.iter().map(|&(rank, id)| (id, rank, None)).collect();
                            self.store.report(&none, polled_at, LoadSource::Poll);
                        }
                        return;
                    }
                };
                let received = now();
                // A data-parallel engine's answer is scoped to each rank,
                // whatever rows are ready; a rank the answer omits is
                // unavailable: nothing is written for it, never another
                // rank's figures.
                let parallel = engine.dp_size > 1;
                let answer: Vec<_> = ranks
                    .into_iter()
                    .map(|(rank, id)| {
                        let row = report
                            .clone()
                            .for_dp_rank(parallel.then_some(usize::from(rank)))
                            .map(|scoped| {
                                load_row(
                                    &scoped,
                                    engine.role,
                                    received,
                                    received,
                                    SignalSource::Poll,
                                )
                            });
                        (id, rank, row)
                    })
                    .collect();
                let rows = answer.iter().filter(|(_, _, row)| row.is_some()).count();
                // The engine's answer in one write: its ranks are judged
                // as the answer left them. An answer with no row writes
                // none and is the word that no report came, which a poll
                // that left out a served rank does not say.
                if rows > 0 || !served {
                    self.store.report(&answer, polled_at, LoadSource::Poll);
                }
                self.stats.rows.fetch_add(rows as u64, Ordering::Relaxed);
            }
            Read::Scrape { text } => {
                let kind = engine.kind.unwrap_or_default();
                let snapshot = text.and_then(|text| {
                    scrape::snapshot(kind, &text)
                        .ok_or_else(|| ReadError::NoLoad(kind.as_str().to_owned()))
                });
                let snapshot = match snapshot {
                    Ok(snapshot) => snapshot,
                    Err(error) => {
                        debug!(engine = peer(&engine.key), %error, "the scrape produced no row");
                        self.stats.scrape_failures.count(&error);
                        let none: Vec<_> = engine
                            .ranks
                            .iter()
                            .map(|&(rank, id)| (id, rank, None))
                            .collect();
                        self.store.report(&none, polled_at, LoadSource::Scrape);
                        return;
                    }
                };
                let received = now();
                let response = scrape::single_rank(snapshot);
                let mut wrote = false;
                // The gauges are engine-wide and only an engine of one rank
                // is scraped (the grouping schedules none for a
                // data-parallel engine), so the sample is that rank's own
                // and no rank reads an aggregate. A rank served by its
                // stream keeps its record.
                for &(rank, id) in &engine.ranks {
                    if self
                        .store
                        .poll_mode(id, received, self.config.poll_interval)
                        == PollMode::Suppressed
                    {
                        continue;
                    }
                    let row = load_row(
                        &response,
                        engine.role,
                        received,
                        received,
                        SignalSource::Scrape,
                    );
                    self.store
                        .report(&[(id, rank, Some(row))], polled_at, LoadSource::Scrape);
                    self.stats.rows.fetch_add(1, Ordering::Relaxed);
                    wrote = true;
                }
                // The count is the scrapes that produced a row: one whose
                // rank took a record of its own while the call was out wrote
                // none.
                if wrote {
                    self.stats.scrapes.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

/// The engines of the membership: the ready rows grouped by their engine
/// key, each with where its poll and its scrape go.
fn engines(rows: Vec<WorkerRow>) -> Vec<Engine> {
    let mut by_key: BTreeMap<String, Engine> = BTreeMap::new();
    // The scrape candidates (engine key, interval), settled once every row is
    // read: the scrape stands in for a servicer, so an engine with a poll
    // has none.
    let mut scrapes: BTreeMap<String, u32> = BTreeMap::new();
    for row in rows {
        if row.status != WorkerStatus::Ready {
            continue;
        }
        let facts = row.facts;
        let engine = by_key
            .entry(facts.engine.clone())
            .or_insert_with(|| Engine {
                key: facts.engine.clone(),
                role: facts.role,
                kind: facts.engine_kind.parse().ok(),
                dp_size: 1,
                poll: None,
                scrape: None,
                ranks: Vec::new(),
            });
        engine.dp_size = engine.dp_size.max(facts.dp_size);
        let dialable = facts.reachable_from == Reachability::Any;
        if engine.poll.is_none() && dialable && facts.wire_modes.contains(WireModes::GRPC) {
            engine.poll = Some(Target {
                endpoint: facts.engine.clone(),
                kind: engine.kind.unwrap_or_default(),
            });
        }
        if dialable
            && facts.wire_modes.contains(WireModes::HTTP)
            && facts.signals.scrape_interval_ms > 0
        {
            scrapes
                .entry(facts.engine.clone())
                .or_insert(facts.signals.scrape_interval_ms);
        }
        engine.ranks.push((facts.dp_rank, row.id));
    }
    let mut engines: Vec<Engine> = by_key.into_values().collect();
    for engine in &mut engines {
        engine.ranks.sort_unstable();
        // The gauges are engine-wide: a data-parallel engine without a
        // servicer is not scraped, and its ranks have no row until a
        // per-rank source (its stream, a `GetLoads`) names them. A row holds
        // a figure measured for its rank, never an aggregate or a share of
        // one.
        if engine.poll.is_none() && engine.dp_size <= 1 {
            if let Some(every) = scrapes.remove(&engine.key) {
                engine.scrape = Some((engine.key.clone(), Duration::from_millis(u64::from(every))));
            }
        }
    }
    engines
}

#[cfg(test)]
mod tests;
