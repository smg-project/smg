//! Lane pool: per-worker FIFO queues served by a fixed set of lanes that steal whole workers.
//!
//! Events of one worker (an engine rank) apply in the order the engine sent them, so the unit of
//! scheduling is the worker, never the event: a worker with queued events sits in exactly one
//! lane's ready list, and the lane that takes it from there applies its events alone until the
//! queue drains or the lane's batch is up. Any lane may take any ready worker, so a lane whose
//! own workers are quiet drains the backlog of a busy one, and no event waits behind another
//! worker's events. The per-worker state (SMG's interned id and the worker's block map) moves with
//! the worker: the claim transitions hand it from lane to lane.
//!
//! # Producers
//!
//! [`LanePool::enqueue`] never blocks and never drops. A worker holds at most `depth_cap` queued
//! events; past that the event comes back as [`QueueFull`] and the producer applies its rule:
//!
//! - the gateway's event monitor stops reading that rank's stream (its per-rank cursor does not
//!   advance, so the engine sees flow control) and retries once [`LanePool::depth`] fell, or
//!   declares the rank stale and resyncs through the recovery path. Dropping one event is never
//!   an option: the index would stay wrong for as long as the block lives;
//! - the replay adapters hold the event in a per-lane backlog and leave the rest in the harness's
//!   own queue, which is what the harness's queue-depth row measures.
//!
//! [`PoolMetrics::rejected`] counts the refusals, `max_depth` the deepest worker queue seen.
//!
//! # Lanes
//!
//! A lane is any thread that calls [`LanePool::run_lane`]: in the replay harnesses the harness's
//! own lane threads (thread counts, pinning and per-lane CPU accounting stay the harness's), in the
//! gateway threads of the pool's owner. Each turn a lane runs the caller's non-blocking `pump`
//! (ingress of the caller's choosing, usually draining a channel into `enqueue`), takes the oldest
//! ready worker of its own list or steals the oldest of a lane that is busy applying another
//! worker (a lane that is not serving empties its own list within microseconds, so taking from
//! it would only move a worker's cache state between cores), applies up to `batch` events
//! and releases the worker; with nothing ready anywhere it runs the caller's `wait` (blocking
//! ingress, or [`LanePool::park_lane`]); the three come as one [`LaneHooks`] value. Idle lanes
//! cost nothing: a parked lane is woken by the enqueue that gives it work, directly when the
//! worker lands on its own list and as the chosen thief when it lands on a busy lane's list, so
//! the gateway's `wait` is a park with a long safety timeout rather than a poll.
//!
//! # Gateway handoff
//!
//! `KvEventMonitor::process_stream` applies the events of a batch inline today. With the pool it
//! enqueues them under the rank's slot, advances its sequence cursor only once the whole batch is
//! queued, and the rank's `WorkerIndexState` becomes the pool's per-worker state; removing a worker
//! is one more queued event that the apply closure answers by taking the state out. Nothing else
//! in the monitor changes.

use std::{
    collections::VecDeque,
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering},
        Mutex, MutexGuard, PoisonError,
    },
    thread::Thread,
    time::{Duration, Instant},
};

use crossbeam_utils::CachePadded;

/// No queued events and not in any ready list.
const IDLE: u8 = 0;
/// In exactly one lane's ready list.
const READY: u8 = 1;
/// One lane is applying the worker's events.
const RUNNING: u8 = 2;

/// Shape of a pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LanePoolConfig {
    /// Threads that will call [`LanePool::run_lane`], each with its own ready list.
    pub lanes: usize,
    /// Worker slots; `enqueue` and `depth` take a slot index below this.
    pub max_workers: usize,
    /// Queued events a worker may hold; `enqueue` refuses the next one.
    pub depth_cap: usize,
    /// Events a lane applies from one worker before releasing it, so a stolen backlog does not
    /// starve the lane's own ready workers.
    pub batch: usize,
    /// Events a ready worker may have queued on a lane that is not serving before any lane may
    /// take it: the case of a lane that is not running (preempted, or pinned to a core something
    /// else holds), which today's rule never reaches because only a serving lane with more on its
    /// list is stolen from. Zero, the default, keeps today's rule exactly. The bound is in events
    /// queued, never in time, so a quiet pool never steals and a stalled lane's backlog is
    /// taken once it is `steal_after` deep.
    pub steal_after: usize,
}

/// The event handed back by [`LanePool::enqueue`] when the worker's queue is at its cap.
pub struct QueueFull<E>(pub E);

impl<E> fmt::Debug for QueueFull<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("QueueFull")
    }
}

/// What a lane does after its `pump` or `wait` hook.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Continue,
    Stop,
}

/// A worker handed to [`LaneHooks::apply`]: its slot index and its state, which is `None` until
/// the hook creates it and `None` again once the hook takes it out.
pub struct Claimed<'a, W> {
    pub worker: u32,
    pub state: &'a mut Option<W>,
}

/// What a lane thread brings to [`LanePool::run_lane`]: how to apply an event and where its
/// ingress comes from.
pub trait LaneHooks<W, E> {
    /// Apply one event of one worker; the pool holds the worker for this call alone.
    fn apply(&mut self, claimed: Claimed<'_, W>, event: E);

    /// Non-blocking ingress at the top of every turn (drain a channel into
    /// [`LanePool::enqueue`], for instance).
    fn pump(&mut self) -> Control {
        Control::Continue
    }

    /// No worker is ready on any lane: block for a bounded time (on the ingress, or in
    /// [`LanePool::park_lane`]).
    fn wait(&mut self) -> Control;
}

/// Closures as hooks: `(apply, pump, wait)`.
impl<W, E, A, P, T> LaneHooks<W, E> for (A, P, T)
where
    A: FnMut(Claimed<'_, W>, E),
    P: FnMut() -> Control,
    T: FnMut() -> Control,
{
    fn apply(&mut self, claimed: Claimed<'_, W>, event: E) {
        (self.0)(claimed, event);
    }

    fn pump(&mut self) -> Control {
        (self.1)()
    }

    fn wait(&mut self) -> Control {
        (self.2)()
    }
}

/// Counters of a pool, summed over lanes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolMetrics {
    pub enqueued: u64,
    pub applied: u64,
    /// Events refused by the depth cap (handed back, never dropped).
    pub rejected: u64,
    /// Ready workers taken from another lane's list.
    pub steals: u64,
    /// Events waiting now (those being applied by a lane count until the batch ends).
    pub queued: usize,
    /// Deepest queue one worker reached.
    pub max_depth: usize,
    /// Most events waiting at once across all workers (sampled every few hundred batches per
    /// lane).
    pub max_queued: usize,
    /// Time lanes spent inside the apply closure.
    pub busy_ns: u64,
    /// Time lanes spent inside their `wait` hook.
    pub idle_ns: u64,
}

/// Queue capacity a worker keeps once its burst has drained.
const KEEP_CAPACITY: usize = 256;

struct WorkerSlot<W, E> {
    state: AtomicU8,
    queue: Mutex<VecDeque<E>>,
    /// Only the lane that moved the slot from READY to RUNNING takes this lock, so it is never
    /// contended; it is a lock rather than a cell so the exclusivity is the type system's.
    held: Mutex<Option<W>>,
}

/// One lane's list and counters. Every counter is the lane's own, so the hot paths touch no line
/// another lane writes; the pool-wide figures are sums.
struct Lane {
    ready: Mutex<VecDeque<u32>>,
    parked: AtomicBool,
    /// Inside `serve`: the lane is applying one worker's batch, so the others on its list wait.
    /// Thieves take from serving lanes only; a lane that is pumping or waiting empties its own
    /// list within microseconds, and moving a worker's state to another core for that would
    /// cost more than it saves.
    serving: AtomicBool,
    thread: Mutex<Option<Thread>>,
    busy_ns: AtomicU64,
    idle_ns: AtomicU64,
    /// Events this lane enqueued, refused, and applied.
    enqueued: AtomicU64,
    rejected: AtomicU64,
    applied: AtomicU64,
    steals: AtomicU64,
    max_depth: AtomicUsize,
    max_queued: AtomicUsize,
}

/// Batches between two samples of the pool-wide queue depth by one lane (a sum over every
/// lane's counters, so not something to take per event).
const SAMPLE_EVERY: u64 = 256;

pub struct LanePool<W, E> {
    config: LanePoolConfig,
    workers: Box<[CachePadded<WorkerSlot<W, E>>]>,
    lanes: Box<[CachePadded<Lane>]>,
    /// One bit per lane: set while the lane is serving a worker and has more on its list, which
    /// is exactly where a thief can take from. A hint: a stale bit costs the thief one empty
    /// look, a missed one delays a steal by a batch.
    stealable: Box<[CachePadded<AtomicU64>]>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl<W: Send, E: Send> LanePool<W, E> {
    /// # Panics
    ///
    /// When the configuration has no lanes, no workers, a zero cap or a zero batch.
    #[must_use]
    pub fn new(config: LanePoolConfig) -> Self {
        assert!(config.lanes > 0, "a lane pool needs at least one lane");
        assert!(
            config.max_workers > 0,
            "a lane pool needs at least one worker slot"
        );
        assert!(
            config.depth_cap > 0,
            "a lane pool needs a positive depth cap"
        );
        assert!(config.batch > 0, "a lane pool needs a positive batch");
        Self {
            config,
            workers: (0..config.max_workers)
                .map(|_| {
                    CachePadded::new(WorkerSlot {
                        state: AtomicU8::new(IDLE),
                        queue: Mutex::new(VecDeque::new()),
                        held: Mutex::new(None),
                    })
                })
                .collect(),
            stealable: (0..config.lanes.div_ceil(64))
                .map(|_| CachePadded::new(AtomicU64::new(0)))
                .collect(),
            lanes: (0..config.lanes)
                .map(|_| {
                    CachePadded::new(Lane {
                        ready: Mutex::new(VecDeque::new()),
                        parked: AtomicBool::new(false),
                        serving: AtomicBool::new(false),
                        thread: Mutex::new(None),
                        busy_ns: AtomicU64::new(0),
                        idle_ns: AtomicU64::new(0),
                        enqueued: AtomicU64::new(0),
                        rejected: AtomicU64::new(0),
                        applied: AtomicU64::new(0),
                        steals: AtomicU64::new(0),
                        max_depth: AtomicUsize::new(0),
                        max_queued: AtomicUsize::new(0),
                    })
                })
                .collect(),
        }
    }

    #[must_use]
    pub fn config(&self) -> &LanePoolConfig {
        &self.config
    }

    /// Queue `event` for `worker`; an idle worker becomes ready on `lane`'s list.
    ///
    /// Never blocks and never drops: at `depth_cap` queued events the event comes back and the
    /// caller holds it (see the module documentation for the rule each producer applies).
    ///
    /// # Errors
    ///
    /// [`QueueFull`] with the event when the worker's queue is at the cap.
    ///
    /// # Panics
    ///
    /// When `lane` or `worker` is out of range.
    pub fn enqueue(&self, lane: usize, worker: u32, event: E) -> Result<(), QueueFull<E>> {
        let slot = &self.workers[worker as usize];
        let me = &self.lanes[lane];
        let depth = {
            let mut queue = lock(&slot.queue);
            if queue.len() >= self.config.depth_cap {
                drop(queue);
                me.rejected.fetch_add(1, Ordering::Relaxed);
                return Err(QueueFull(event));
            }
            queue.push_back(event);
            queue.len()
        };
        me.enqueued.fetch_add(1, Ordering::Relaxed);
        me.max_depth.fetch_max(depth, Ordering::Relaxed);
        if slot
            .state
            .compare_exchange(IDLE, READY, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            self.make_ready(lane, worker);
        }
        Ok(())
    }

    /// Events queued for `worker` now.
    #[must_use]
    pub fn depth(&self, worker: u32) -> usize {
        lock(&self.workers[worker as usize].queue).len()
    }

    /// Events queued across all workers now (enqueued minus applied, summed over lanes).
    #[must_use]
    pub fn queued(&self) -> usize {
        let (mut enqueued, mut applied) = (0u64, 0u64);
        for lane in &self.lanes {
            enqueued += lane.enqueued.load(Ordering::Relaxed);
            applied += lane.applied.load(Ordering::Relaxed);
        }
        usize::try_from(enqueued.saturating_sub(applied)).unwrap_or(usize::MAX)
    }

    #[must_use]
    pub fn metrics(&self) -> PoolMetrics {
        let mut metrics = PoolMetrics::default();
        for lane in &self.lanes {
            metrics.enqueued += lane.enqueued.load(Ordering::Relaxed);
            metrics.rejected += lane.rejected.load(Ordering::Relaxed);
            metrics.applied += lane.applied.load(Ordering::Relaxed);
            metrics.steals += lane.steals.load(Ordering::Relaxed);
            metrics.busy_ns += lane.busy_ns.load(Ordering::Relaxed);
            metrics.idle_ns += lane.idle_ns.load(Ordering::Relaxed);
            metrics.max_depth = metrics
                .max_depth
                .max(lane.max_depth.load(Ordering::Relaxed));
            metrics.max_queued = metrics
                .max_queued
                .max(lane.max_queued.load(Ordering::Relaxed));
        }
        metrics.queued = metrics
            .enqueued
            .saturating_sub(metrics.applied)
            .try_into()
            .unwrap_or(usize::MAX);
        metrics
    }

    fn make_ready(&self, lane: usize, worker: u32) {
        let target = &self.lanes[lane];
        lock(&target.ready).push_back(worker);
        if target.serving.load(Ordering::Relaxed) {
            self.mark_stealable(lane, true);
        }
        if target.parked.swap(false, Ordering::SeqCst) {
            Self::unpark(target);
        } else {
            // The target lane is busy: wake one parked lane, which finds its own list empty and
            // steals the worker. Clearing the flag on the sleeper's behalf makes each wake-up
            // reach a different lane.
            for (index, other) in self.lanes.iter().enumerate() {
                if index != lane && other.parked.swap(false, Ordering::SeqCst) {
                    Self::unpark(other);
                    break;
                }
            }
        }
    }

    fn unpark(lane: &Lane) {
        if let Some(thread) = lock(&lane.thread).as_ref() {
            thread.unpark();
        }
    }

    /// Serve `lane` until the hooks' `pump` or `wait` returns [`Control::Stop`].
    ///
    /// `pump` runs at the top of every turn and must not block; `wait` runs when no worker is
    /// ready on any lane and may block for a bounded time; `apply` sees one worker at a time with
    /// exclusive access to that worker's state.
    ///
    /// # Panics
    ///
    /// When `lane` is out of range.
    pub fn run_lane(&self, lane: usize, hooks: &mut impl LaneHooks<W, E>) {
        let me = &self.lanes[lane];
        *lock(&me.thread) = Some(std::thread::current());
        let mut seed = (lane as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        loop {
            if hooks.pump() == Control::Stop {
                break;
            }
            let next = lock(&me.ready)
                .pop_front()
                .or_else(|| self.steal(lane, &mut seed));
            let Some(worker) = next else {
                let started = Instant::now();
                let control = hooks.wait();
                me.idle_ns
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if control == Control::Stop {
                    break;
                }
                continue;
            };
            self.serve(lane, worker, hooks);
        }
        *lock(&me.thread) = None;
    }

    /// Sleep until an enqueue makes a worker ready (on this lane's list, or on a busy lane's list
    /// with this lane chosen to steal it) or `timeout` passes. The wake is a `Thread::unpark`
    /// from the producer, so an idle fleet costs no wake-ups; the timeout is the safety net.
    pub fn park_lane(&self, lane: usize, timeout: Duration) {
        let me = &self.lanes[lane];
        me.parked.store(true, Ordering::SeqCst);
        // An enqueue that pushed before this check is seen here; one that pushes after it sees
        // the flag (both go through the list's lock) and unparks, which a later park consumes.
        if lock(&me.ready).is_empty() {
            std::thread::park_timeout(timeout);
        }
        me.parked.store(false, Ordering::SeqCst);
    }

    /// Whether some lane is serving with more workers on its list: the only reason for an idle
    /// lane to wake before its own ingress does. One word per 64 lanes.
    #[must_use]
    pub fn has_stealable(&self) -> bool {
        self.stealable
            .iter()
            .any(|word| word.load(Ordering::Relaxed) != 0)
    }

    fn mark_stealable(&self, lane: usize, on: bool) {
        let word = &self.stealable[lane / 64];
        let bit = 1u64 << (lane % 64);
        if on {
            word.fetch_or(bit, Ordering::Relaxed);
        } else {
            word.fetch_and(!bit, Ordering::Relaxed);
        }
    }

    /// Take the oldest ready worker of a lane whose bit says it is serving with more on its
    /// list; one word read when there is none.
    fn steal(&self, lane: usize, seed: &mut u64) -> Option<u32> {
        if self.lanes.len() == 1 {
            return None;
        }
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let start = (*seed % self.stealable.len() as u64) as usize;
        for step in 0..self.stealable.len() {
            let index = (start + step) % self.stealable.len();
            let mut word = self.stealable[index].load(Ordering::Relaxed);
            if index == lane / 64 {
                word &= !(1u64 << (lane % 64));
            }
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                let victim = index * 64 + bit;
                let Ok(mut ready) = self.lanes[victim].ready.try_lock() else {
                    continue;
                };
                let taken = ready.pop_front();
                if ready.is_empty() {
                    self.mark_stealable(victim, false);
                }
                drop(ready);
                if let Some(worker) = taken {
                    self.lanes[lane].steals.fetch_add(1, Ordering::Relaxed);
                    return Some(worker);
                }
            }
        }
        if self.config.steal_after == 0 {
            return None;
        }
        // Nothing is stealable by the serving rule: look for a ready worker waiting on a lane that
        // is not serving with `steal_after` events behind it, which is a lane that is not running.
        // One try-lock per lane of the pool, only on an idle turn that found nothing above.
        let count = self.lanes.len();
        let start = (*seed % count as u64) as usize;
        for step in 0..count {
            let victim = (start + step) % count;
            if victim == lane || self.lanes[victim].serving.load(Ordering::Relaxed) {
                continue;
            }
            let Ok(mut ready) = self.lanes[victim].ready.try_lock() else {
                continue;
            };
            let Some(&worker) = ready.front() else {
                continue;
            };
            if lock(&self.workers[worker as usize].queue).len() < self.config.steal_after {
                continue;
            }
            ready.pop_front();
            drop(ready);
            self.lanes[lane].steals.fetch_add(1, Ordering::Relaxed);
            return Some(worker);
        }
        None
    }

    fn serve(&self, lane: usize, worker: u32, hooks: &mut impl LaneHooks<W, E>) {
        let slot = &self.workers[worker as usize];
        let claimed = slot
            .state
            .compare_exchange(READY, RUNNING, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok();
        debug_assert!(claimed, "a ready worker is claimed by exactly one lane");
        let me = &self.lanes[lane];
        me.serving.store(true, Ordering::Relaxed);
        if !lock(&me.ready).is_empty() {
            self.mark_stealable(lane, true);
        }
        let started = Instant::now();
        let mut applied = 0;
        {
            let mut held = lock(&slot.held);
            while applied < self.config.batch {
                let event = {
                    let mut queue = lock(&slot.queue);
                    let event = queue.pop_front();
                    if event.is_none() && queue.capacity() > KEEP_CAPACITY {
                        // The burst is over: give its buffer back.
                        queue.shrink_to(KEEP_CAPACITY);
                    }
                    event
                };
                let Some(event) = event else {
                    break;
                };
                hooks.apply(
                    Claimed {
                        worker,
                        state: &mut held,
                    },
                    event,
                );
                applied += 1;
            }
        }
        me.busy_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let total = me.applied.fetch_add(applied as u64, Ordering::Relaxed) + applied as u64;
        me.serving.store(false, Ordering::Relaxed);
        self.mark_stealable(lane, false);
        if total / SAMPLE_EVERY != (total - applied as u64) / SAMPLE_EVERY {
            me.max_queued.fetch_max(self.queued(), Ordering::Relaxed);
        }
        slot.state.store(IDLE, Ordering::Release);
        // Whoever observes the queue non-empty after this point schedules the worker once: an
        // enqueue that saw RUNNING left scheduling to us, one that comes later wins the exchange.
        let pending = !lock(&slot.queue).is_empty();
        if pending
            && slot
                .state
                .compare_exchange(IDLE, READY, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        {
            self.make_ready(lane, worker);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicBool, Ordering},
        time::Duration,
    };

    use super::{Claimed, Control, LanePool, LanePoolConfig};

    struct Log {
        seen: Vec<u64>,
    }

    /// Every worker's events apply in order and under one lane at a time while three lanes
    /// steal from each other and two producers wait for room instead of losing events.
    #[test]
    fn events_of_a_worker_apply_in_order_under_one_lane_at_a_time() {
        const WORKERS: u32 = 16;
        const EVENTS: u64 = 2_000;
        const LANES: usize = 3;
        let pool: LanePool<Log, u64> = LanePool::new(LanePoolConfig {
            lanes: LANES,
            max_workers: WORKERS as usize,
            depth_cap: 64,
            batch: 8,
            steal_after: 0,
        });
        let produced = AtomicBool::new(false);
        let claims: Vec<AtomicBool> = (0..WORKERS).map(|_| AtomicBool::new(false)).collect();
        std::thread::scope(|scope| {
            for lane in 0..LANES {
                let (pool, produced, claims) = (&pool, &produced, &claims);
                scope.spawn(move || {
                    pool.run_lane(
                        lane,
                        &mut (
                            |claimed: Claimed<'_, Log>, event: u64| {
                                let worker = claimed.worker as usize;
                                assert!(
                                    !claims[worker].swap(true, Ordering::SeqCst),
                                    "two lanes inside worker {worker}"
                                );
                                claimed
                                    .state
                                    .get_or_insert_with(|| Log { seen: Vec::new() })
                                    .seen
                                    .push(event);
                                std::thread::yield_now();
                                claims[worker].store(false, Ordering::SeqCst);
                            },
                            || Control::Continue,
                            || {
                                if produced.load(Ordering::SeqCst) && pool.queued() == 0 {
                                    return Control::Stop;
                                }
                                // Long enough that progress depends on the producers' wake-ups.
                                pool.park_lane(lane, Duration::from_millis(50));
                                Control::Continue
                            },
                        ),
                    );
                });
            }
            let producers: Vec<_> = (0..2u32)
                .map(|half| {
                    let pool = &pool;
                    scope.spawn(move || {
                        for event in 0..EVENTS {
                            for worker in (half..WORKERS).step_by(2) {
                                let mut pending = event;
                                loop {
                                    match pool.enqueue((worker as usize) % LANES, worker, pending) {
                                        Ok(()) => break,
                                        Err(full) => {
                                            pending = full.0;
                                            std::thread::yield_now();
                                        }
                                    }
                                }
                            }
                        }
                    })
                })
                .collect();
            for producer in producers {
                producer.join().expect("producer");
            }
            produced.store(true, Ordering::SeqCst);
        });
        let metrics = pool.metrics();
        assert_eq!(metrics.enqueued, u64::from(WORKERS) * EVENTS);
        assert_eq!(metrics.applied, u64::from(WORKERS) * EVENTS);
        assert_eq!(metrics.queued, 0);
        assert!(metrics.max_depth <= 64);
        for worker in 0..WORKERS {
            let seen = super::lock(&pool.workers[worker as usize].held)
                .as_ref()
                .map(|log| log.seen.clone())
                .unwrap_or_default();
            assert_eq!(seen, (0..EVENTS).collect::<Vec<_>>(), "worker {worker}");
        }
    }

    /// The cap hands the event back instead of dropping it, and counts the refusal.
    #[test]
    fn the_depth_cap_refuses_without_dropping() {
        let pool: LanePool<(), &'static str> = LanePool::new(LanePoolConfig {
            lanes: 1,
            max_workers: 2,
            depth_cap: 2,
            batch: 4,
            steal_after: 0,
        });
        assert!(pool.enqueue(0, 1, "a").is_ok());
        assert!(pool.enqueue(0, 1, "b").is_ok());
        let refused = pool
            .enqueue(0, 1, "c")
            .expect_err("third event is over the cap");
        assert_eq!(refused.0, "c");
        assert_eq!(pool.depth(1), 2);
        assert_eq!(pool.depth(0), 0);
        let metrics = pool.metrics();
        assert_eq!(
            (metrics.enqueued, metrics.rejected, metrics.queued),
            (2, 1, 2)
        );
        assert_eq!(metrics.max_depth, 2);
    }

    /// A lane serves its own ready workers a batch at a time; while a lane is stuck inside one
    /// worker's event, another lane takes the rest of its list, whole workers at a time, and
    /// every worker's events stay in order. Nothing is taken from a lane that is not serving.
    #[test]
    fn an_idle_lane_steals_whole_workers_from_a_serving_one() {
        let pool: LanePool<Vec<u32>, u32> = LanePool::new(LanePoolConfig {
            lanes: 2,
            max_workers: 4,
            depth_cap: 16,
            batch: 2,
            steal_after: 0,
        });
        for round in 0..3 {
            for worker in 0..4u32 {
                // Workers 0 and 1 are lane 1's own, 2 and 3 are lane 0's.
                let home = usize::from(worker >= 2);
                pool.enqueue(1 - home, worker, round * 10 + worker).unwrap();
            }
        }
        // Lane 1 alone: lane 0 is not serving, so its workers stay where they are.
        let mut order = Vec::new();
        pool.run_lane(
            1,
            &mut (
                |claimed: Claimed<'_, Vec<u32>>, event: u32| {
                    claimed.state.get_or_insert_with(Vec::new).push(event);
                    order.push((claimed.worker, event));
                },
                || Control::Continue,
                || Control::Stop,
            ),
        );
        assert_eq!(pool.metrics().steals, 0);
        assert_eq!(pool.metrics().applied, 6);
        // Batches of two: worker 0 gave the lane to worker 1 after two events and came back.
        let lanes_own: Vec<u32> = order.iter().map(|(w, _)| *w).collect();
        assert_eq!(lanes_own, vec![0, 0, 1, 1, 0, 1]);

        // Lane 0 enters worker 2 and blocks there; lane 1 then takes worker 3 from its list.
        let blocked = AtomicBool::new(true);
        let stolen = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            let (pool, blocked) = (&pool, &blocked);
            scope.spawn(move || {
                pool.run_lane(
                    0,
                    &mut (
                        |claimed: Claimed<'_, Vec<u32>>, event: u32| {
                            while blocked.load(Ordering::SeqCst) {
                                std::thread::yield_now();
                            }
                            claimed.state.get_or_insert_with(Vec::new).push(event);
                        },
                        || Control::Continue,
                        || Control::Stop,
                    ),
                );
            });
            while !pool.lanes[0].serving.load(Ordering::Relaxed) {
                std::thread::yield_now();
            }
            pool.run_lane(
                1,
                &mut (
                    |claimed: Claimed<'_, Vec<u32>>, event: u32| {
                        claimed.state.get_or_insert_with(Vec::new).push(event);
                        super::lock(&stolen).push((claimed.worker, event));
                    },
                    || Control::Continue,
                    || Control::Stop,
                ),
            );
            blocked.store(false, Ordering::SeqCst);
        });
        let metrics = pool.metrics();
        assert_eq!(metrics.steals, 1, "{metrics:?}");
        assert_eq!(metrics.applied, 12);
        assert_eq!(*super::lock(&stolen), vec![(3, 3), (3, 13), (3, 23)]);
        let worker_two = super::lock(&pool.workers[2].held)
            .clone()
            .unwrap_or_default();
        assert_eq!(worker_two, vec![2, 12, 22]);
    }

    /// A ready worker on a lane that is not running is taken by another lane once its queue is
    /// `steal_after` deep, and never under the default: lane 0 is never run here, which is what a
    /// preempted lane looks like to the pool.
    #[test]
    fn a_stalled_lane_is_drained_once_its_backlog_reaches_steal_after() {
        for (steal_after, expect_applied, expect_steals) in [(0usize, 0u64, 0u64), (3, 4, 1)] {
            let pool: LanePool<(), u32> = LanePool::new(LanePoolConfig {
                lanes: 2,
                max_workers: 1,
                depth_cap: 16,
                batch: 8,
                steal_after,
            });
            for event in 0..4u32 {
                pool.enqueue(0, 0, event).unwrap();
            }
            let mut applied = Vec::new();
            let mut turns = 0u32;
            pool.run_lane(
                1,
                &mut (
                    |_claimed: Claimed<'_, ()>, event: u32| applied.push(event),
                    || Control::Continue,
                    || {
                        turns += 1;
                        if turns > 20 {
                            return Control::Stop;
                        }
                        pool.park_lane(1, Duration::from_millis(1));
                        Control::Continue
                    },
                ),
            );
            assert_eq!(
                applied.len() as u64,
                expect_applied,
                "steal_after {steal_after}: lane 1 took the stalled lane's backlog"
            );
            if expect_applied > 0 {
                assert_eq!(applied, vec![0, 1, 2, 3], "in order");
            }
            assert_eq!(pool.metrics().steals, expect_steals);
            assert_eq!(pool.metrics().queued, 4 - expect_applied as usize);
        }
    }

    /// Under `steal_after`, a backlog shallower than the setting stays with its lane: a lane that
    /// is merely slow to its next turn is not robbed of its own work.
    #[test]
    fn a_shallow_backlog_stays_with_its_lane() {
        let pool: LanePool<(), u32> = LanePool::new(LanePoolConfig {
            lanes: 2,
            max_workers: 1,
            depth_cap: 16,
            batch: 8,
            steal_after: 3,
        });
        pool.enqueue(0, 0, 1).unwrap();
        pool.enqueue(0, 0, 2).unwrap();
        let mut turns = 0u32;
        let mut applied = 0u64;
        pool.run_lane(
            1,
            &mut (
                |_claimed: Claimed<'_, ()>, _event: u32| applied += 1,
                || Control::Continue,
                || {
                    turns += 1;
                    if turns > 5 {
                        return Control::Stop;
                    }
                    Control::Continue
                },
            ),
        );
        assert_eq!(applied, 0);
        assert_eq!(pool.metrics().steals, 0);
        assert_eq!(pool.depth(0), 2);
    }
}
