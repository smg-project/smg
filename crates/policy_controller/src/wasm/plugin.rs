//! A loaded plugin and how the host runs it: one compiled component, one
//! kept instance per decider, one `decision` resource per decision
//! per profile that binds the plugin, every call under fuel, and the rules
//! for what a trap, an exhausted budget, an invalid answer or a decline does.

use std::{
    cell::{Cell, RefCell, RefMut},
    collections::VecDeque,
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        Arc, OnceLock,
    },
    thread::ThreadId,
    time::Instant,
};

use parking_lot::Mutex;
use thread_local::ThreadLocal;
use wasmtime::{
    component::{Component, HasSelf, Linker, ResourceAny},
    Config, Engine, OptLevel, Store, StoreLimitsBuilder, Trap, UpdateDeadline,
};

use super::{
    bindings::{packed, record, wit, HostState, LogGate, Watch},
    config::{Abi, EventKind, EventSet, Exposure, Failure, FrameBudget, PluginSpec},
    view::{self as marshal, Packed, Records, TooLong},
};
use crate::{
    decider::Decider,
    decision::{Decision, Target, MAX_TARGETS},
    framework::{self, DecisionScope},
    lease::{Lease, Membership, Outcome, ReleaseReason, Side, Signal},
    plugins::{
        breaker::{Breaker, Latency},
        intern, own_bound, BoundPlugin, Counters, Fallback, HalfOpen, OpenDecision, OpenDecisions,
        Part,
    },
    spec::Needs,
    view::{Candidate, Phase, PoolFacts, RequestFacts, WorkerId},
    Draw,
};

const MIB: usize = 1 << 20;

/// The frame of an `init` call: instantiation is an apply-time action, so
/// it gets a long one.
const INIT_FRAME_MICROS: u32 = 100_000;

/// The frame of an `observe` call for a plugin whose own budget is unset
/// (there is no request budget to take).
const OBSERVE_FRAME_MICROS: u32 = 1_000;
/// The most elements one table of an instance may hold: a policy's tables
/// are its indirect calls and its handles, hundreds at most.
const TABLE_ELEMENTS: usize = 1 << 16;
/// The epoch ticker's period. A tick that finds a guest call in flight reads
/// its thread's CPU clock ([`tick`]): a call's bound on the clock counts the
/// time its thread had the CPU, and the call ends at the first tick that
/// finds it has had the bound. Fuel is the fine bound (one unit per
/// operator, a bulk memory operation by its size once it has run); the
/// ticks catch what the calibration underprices and the CPU a host call
/// spends.
const EPOCH_TICK_MICROS: u64 = 1_000;

/// A bound of `bound_micros` on the wall clock in whole ticks, rounded up:
/// no deadline there ends a call sooner.
fn whole_ticks(bound_micros: u64) -> u64 {
    bound_micros.div_ceil(EPOCH_TICK_MICROS)
}

/// The epoch deadline of a call whose bound is `frame_micros` on the wall
/// clock: the bound in whole ticks, plus the tick in flight.
fn deadline_ticks(frame_micros: u32) -> u64 {
    whole_ticks(u64::from(frame_micros)) + 1
}

/// The first epoch deadline of a call, in ticks from its start. A watched
/// call with no reading of its start meets its next tick, which takes the
/// reading its bound counts from. A call whose start was read meets the tick
/// its deadline on the wall clock falls on and none before it, so that
/// reading ends only a call that is late: no call the wall clock would let
/// run is ended because its thread is armed. A call nothing watches (an
/// `init`; a host with no thread clock) ends at that deadline.
fn first_deadline(watched: bool, start_read: bool, bound_micros: u32) -> u64 {
    if watched && !start_read {
        1
    } else {
        deadline_ticks(bound_micros)
    }
}

/// The calling thread's CPU time, in microseconds; `None` on a host with no
/// thread clock, where a call's bound stays on the wall clock.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn thread_cpu_micros() -> Option<u64> {
    let now = rustix::time::clock_gettime(rustix::time::ClockId::ThreadCPUTime);
    let secs = u64::try_from(now.tv_sec).ok()?;
    let nanos = u64::try_from(now.tv_nsec).ok()?;
    Some(secs.saturating_mul(1_000_000).saturating_add(nanos / 1_000))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn thread_cpu_micros() -> Option<u64> {
    None
}

/// One tick with a guest call in flight on its store, run on the call's own
/// thread at the guest's next check. A watched call ends at the tick that
/// finds it has had its bound on the CPU since a reading, and runs on
/// otherwise, however late the wall clock says it is: the guest of a thread
/// the host held finishes in its own microseconds when the thread has the
/// CPU back. The reading is the call's start where the thread took one;
/// otherwise the call's first tick takes it, and the bound counts from
/// there. A first tick knows nothing of the time before it: it can end a
/// call (a call that has its bound on the CPU after it), it never says a
/// call was held. A call nothing watches ends here, on the wall clock.
fn tick(watch: &mut Watch) -> wasmtime::Result<UpdateDeadline> {
    let (true, Some(cpu)) = (watch.watched, thread_cpu_micros()) else {
        return Err(Trap::Interrupt.into());
    };
    let since = match watch.start {
        Some(start) => start,
        None => *watch.first_tick.get_or_insert(cpu),
    };
    if cpu.saturating_sub(since) >= watch.bound_micros {
        return Err(Trap::Interrupt.into());
    }
    Ok(UpdateDeadline::Continue(1))
}

/// How a call ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    /// It answered or declined.
    Answered,
    /// It failed by itself: a trap, or its fuel ran out.
    Failed,
    /// Its bound on the clock ended it.
    OnTheClock,
}

/// What the end of a call says of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// The call's time is the plugin's own: it was not late, or its bound
    /// ended it, or a reading of its start says it had its bound, or it
    /// failed by itself.
    Own,
    /// Late, answered or declined, and a reading of its start says its
    /// thread had less than the bound on the CPU: the host held it.
    Held,
    /// Late, not ended, and no reading of its start: the host held it or it
    /// overran a bound its fuel did not price, and nothing says which.
    Unread,
}

/// A call's own time in its plugin's two sums of a decision, in
/// microseconds. With no reading a call's time is an interval, from its
/// fuel to its charge, and the two sums take its two ends; a reading makes
/// it one number. The low end is lent against a reading to come: on a host
/// with no thread clock none can, and a call's time is its charge in both.
/// The `init` a decision pays on a cold slot is timed the same way: its two
/// ends join the decision's sample, never the plugin's own spend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct OwnTime {
    /// For the decision's latency sample: what a reading says the call had;
    /// with none, its charge. The ring wants more than one slow sample, so
    /// it takes the high end: a guest slow in what its fuel does not price
    /// reads slow there.
    sample: u64,
    /// For the plugin's own spend, which a later call that runs out is
    /// judged against: what a reading says the call had; with none, its
    /// fuel. One run-out is a verdict, so it convicts on the low end: the
    /// time a host held a call is never the plugin's spend. A host with no
    /// thread clock has no low end to lend: there it is the charge.
    spend: u64,
}

impl OwnTime {
    /// The time of a call or an `init` a reading covers, one number: what it
    /// could spend, its fuel or the CPU its thread `had`, never more than
    /// its charge.
    fn read(fuel_micros: u64, had: u64, charged: u64) -> Self {
        let read = fuel_micros.max(had).min(charged);
        Self {
            sample: read,
            spend: read,
        }
    }

    /// The time of one no reading covers, its two ends. With no thread
    /// clock (`watched` is false) nothing will ever read it, and its bound
    /// is the wall clock's: the time it took is its spend.
    fn unread(watched: bool, fuel_micros: u64, charged: u64) -> Self {
        Self {
            sample: charged,
            spend: if watched { fuel_micros } else { charged },
        }
    }

    /// The two sums with `more` added.
    fn plus(self, more: Self) -> Self {
        Self {
            sample: self.sample.saturating_add(more.sample),
            spend: self.spend.saturating_add(more.spend),
        }
    }
}

/// One thread's latency ring and its arm: whether the thread reads its CPU
/// clock at the start and the end of the plugin's calls
/// ([`ThreadRing::arm`]). The sample of a late call no reading covers is
/// the time the call took, which a hold of the host makes slow, so the ring
/// that is about to take one is armed until that sample has left it, and at
/// most one of them is ever in a window. A call that runs out under a
/// decision's remainder arms it too where the two ends of the time nothing
/// read part: by the time the plugin's unread calls took the run-out is its
/// own, by their fuel it is not. So does a sample slow at its high end alone
/// ([`ThreadRing::uncovered`]), late call or none: a window never holds two
/// slow samples no reading covered.
#[derive(Debug)]
struct ThreadRing {
    ring: Latency,
    /// The samples the ring still takes armed.
    armed: usize,
}

impl ThreadRing {
    const fn new() -> Self {
        Self {
            ring: Latency::new(),
            armed: 0,
        }
    }

    /// Arms the ring's thread for the ring's length in samples: the sample
    /// of the decision in flight and the [`Latency::SAMPLES`] - 1 after it.
    /// The sample that follows those overwrites this decision's.
    fn arm(&mut self) {
        self.armed = Latency::SAMPLES;
    }

    /// Whether the thread reads its CPU clock at the start and the end of a
    /// call.
    fn armed(&self) -> bool {
        self.armed != 0
    }

    /// Whether a sample of `micros` is past a mark that `low_micros` is
    /// short of, both under a frame of `frame_micros`. The second is the
    /// same decision at the low end of the time nothing read, so such a
    /// sample is slow by that time alone: no reading covers it. A sample
    /// slow by both ends is not one, nor is a call that is no sample.
    fn uncovered(micros: u32, low_micros: u32, frame_micros: u32) -> bool {
        Latency::share(micros, frame_micros).map(Latency::marks)
            != Latency::share(low_micros, frame_micros).map(Latency::marks)
    }

    /// Records one call in the ring ([`Latency::record`]); `true` when a
    /// window it completed reads slow. The arm falls by one where a sample
    /// is written, so it is counted in the ring's own unit, and the ring
    /// says what a sample is ([`Latency::share`]).
    fn record(&mut self, micros: u32, frame_micros: u32) -> bool {
        if Latency::share(micros, frame_micros).is_some() {
            self.armed = self.armed.saturating_sub(1);
        }
        self.ring.record(micros, frame_micros)
    }
}

/// A call's own time and its verdict, judged once, at its end. A reading
/// covers a call whose start was read (its thread is armed): `cpu` reads the
/// thread's clock again, and the call's time is what it could spend, its
/// fuel or the CPU its thread had, never the time the host held it, late or
/// not. Late, such a call is the host's when its thread had less than the
/// bound and it answered or declined, the plugin's otherwise. A hold excuses
/// lateness, never a fault: a call that failed by itself is the plugin's
/// whatever its thread had. One call there is not read again: not late,
/// with fuel that covers the time it took, its time is its fuel whatever
/// its thread had. No reading covers any other call. It is sampled at its
/// charge (`fuel_micros` or `elapsed_micros`, whichever is more), what it
/// was before the bound counted the CPU, and late it is unread unless its
/// bound ended it; it spends its fuel, for nothing says how much of its
/// time its thread had. Time nothing read draws the decision's remainder,
/// never the plugin's own spend. A host with no thread clock watches no
/// call and can never read one: there a call spends its charge, the time
/// the wall clock that bounds it says it took.
///
/// Late is the wall clock at or past the bound in whole ticks, the earliest
/// a deadline there could have ended the call. Not a tick more: a call the
/// host held for between the two would be charged its hold and counted
/// nowhere, and two such samples in a ring read slow. Not the bound itself:
/// under a frame shorter than a tick a call inside its fuel runs past its
/// bound on the wall clock and is ended by nothing.
// A call that is not late on a thread that is not armed pays two compares
// here: inlined, it pays no call.
#[inline]
fn judge(
    watch: &Watch,
    elapsed_micros: u64,
    fuel_micros: u64,
    end: End,
    cpu: impl FnOnce() -> Option<u64>,
) -> (OwnTime, Verdict) {
    let charged = fuel_micros.max(elapsed_micros);
    let late = watch.watched
        && elapsed_micros >= whole_ticks(watch.bound_micros).saturating_mul(EPOCH_TICK_MICROS);
    // Not late and covered by its fuel, a call takes its fuel in both arms
    // below: no reading can move its time, so none is taken.
    let Some(had) = watch
        .start
        .filter(|_| watch.watched && (late || fuel_micros < elapsed_micros))
        .and_then(|start| Some(cpu()?.saturating_sub(start)))
    else {
        let verdict = if late && end != End::OnTheClock {
            Verdict::Unread
        } else {
            Verdict::Own
        };
        return (
            OwnTime::unread(watch.watched, fuel_micros, charged),
            verdict,
        );
    };
    let verdict = if late && had < watch.bound_micros && end == End::Answered {
        Verdict::Held
    } else {
        Verdict::Own
    };
    (OwnTime::read(fuel_micros, had, charged), verdict)
}

/// Tables one store may hold: a component built from one module has the
/// module's and the shim's.
const TABLES: usize = 4;
/// Core instances one store may hold: a component's module, shim and fixup.
const INSTANCES: usize = 8;
/// The fuel of a decision resource's destructor, in microseconds at the
/// plugin's calibration: the guest's code, run for the host's cleanup, so it
/// has a small fuel frame of its own outside the decision's remainder (a
/// spent budget drops no healthy instance).
const DESTRUCTOR_FRAME_MICROS: u32 = 100;
/// The bound of a destructor on the clock: its own, in milliseconds, so a
/// destructor the calibration underprices, or one long on the CPU in its
/// host calls, cannot stretch a decision by a frame. Like every call's it
/// counts the time the thread had the CPU: a destructor the host held is
/// not ended.
const DESTRUCTOR_BOUND_MILLIS: u32 = 2;
const DESTRUCTOR_BOUND_MICROS: u32 = DESTRUCTOR_BOUND_MILLIS * 1_000;

/// What the epoch ticker and the calls share. The calls in flight are
/// counted per thread, so a call touches its own line and the one word the
/// ticker writes when it parks, never a counter every deciding thread bumps.
#[derive(Default)]
struct Ticking {
    /// Set when the runtime drops: the ticker exits at its next wake.
    stop: AtomicBool,
    /// The ticker is parked: a call that starts wakes it.
    parked: AtomicBool,
    /// Guest calls in flight on each thread: the ticker runs while any is.
    busy: ThreadLocal<AtomicU32>,
}

impl Ticking {
    fn any_busy(&self) -> bool {
        self.busy.iter().any(|b| b.load(Ordering::SeqCst) != 0)
    }
}

/// A guest call in flight on this thread: the epoch ticker runs while any
/// thread has one, parked otherwise.
struct Running<'a>(&'a AtomicU32);

impl<'a> Running<'a> {
    fn start(runtime: &'a Runtime) -> Self {
        let busy = runtime.ticking.busy.get_or_default();
        busy.fetch_add(1, Ordering::SeqCst);
        // The ticker parks only after a re-read of every thread's count, so
        // either it sees this call, or this call sees it parked and wakes it
        // (the token makes a park that follows return at once).
        if runtime.ticking.parked.load(Ordering::SeqCst) {
            runtime.ticker.unpark();
        }
        Self(busy)
    }
}

impl Drop for Running<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The runtime every plugin of a process shares: the compiler, the linker
/// with the four host imports, and the host's clock. It is built over the
/// deciding host's [`OpenDecisions`], which hold the decision open on each
/// thread and its budget: a decision that opens plugins of two runtimes
/// built over the same one is one decision to them all, with one remainder.
pub struct Runtime {
    engine: Engine,
    linker: Linker<HostState>,
    clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    /// The epoch ticker's state, shared with its thread.
    ticking: Arc<Ticking>,
    /// The ticker thread, to wake it when a call starts.
    ticker: std::thread::Thread,
    /// The decision open on each thread and its budget: shared with every
    /// runtime built over the same ones.
    decisions: Arc<OpenDecisions>,
    /// Whether this host has a thread CPU clock: a call's bound then counts
    /// the time its thread had the CPU, and is the wall clock's otherwise.
    thread_clock: bool,
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.ticking.stop.store(true, Ordering::Relaxed);
        self.ticker.unpark();
    }
}

impl fmt::Debug for Runtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Runtime").finish_non_exhaustive()
    }
}

impl Runtime {
    /// A runtime over `decisions` whose live clock counts microseconds from
    /// its creation. A host whose view clock has another origin passes it to
    /// [`Runtime::with_clock`], so `now-micros` inside `observe` and a
    /// veto's remaining time read on the same clock as the decisions.
    pub fn new(decisions: &Arc<OpenDecisions>) -> Result<Self, LoadError> {
        let origin = Instant::now();
        Self::with_clock(decisions, move || {
            u64::try_from(origin.elapsed().as_micros()).unwrap_or(u64::MAX)
        })
    }

    /// A runtime over `decisions` on the host's clock: `clock` returns
    /// microseconds since the host's origin, the clock of
    /// [`RequestFacts::now_micros`].
    pub fn with_clock(
        decisions: &Arc<OpenDecisions>,
        clock: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> Result<Self, LoadError> {
        let mut config = Config::new();
        configure(&mut config);
        let engine = Engine::new(&config).map_err(|e| LoadError::Runtime(e.to_string()))?;
        let mut linker = Linker::<HostState>::new(&engine);
        record::smg::policy::host::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)
            .map_err(|e| LoadError::Runtime(e.to_string()))?;
        // The clock of every call: one thread per runtime advances the
        // engine's epoch each tick while a call is in flight (parked while
        // none is), and a tick that finds a call in flight runs `tick` on
        // the call's thread.
        let ticking = Arc::new(Ticking::default());
        let shared = (Arc::clone(&ticking), engine.clone());
        // The name fits the kernel's fifteen characters, so `/proc` shows it
        // whole.
        let handle = std::thread::Builder::new()
            .name("wasm-epoch".into())
            .spawn(move || {
                let (ticking, engine) = shared;
                while !ticking.stop.load(Ordering::Relaxed) {
                    if ticking.any_busy() {
                        std::thread::sleep(std::time::Duration::from_micros(EPOCH_TICK_MICROS));
                        engine.increment_epoch();
                        continue;
                    }
                    // A call that started since the read above saw `parked`
                    // unset and did not wake us: the re-read sees its count.
                    ticking.parked.store(true, Ordering::SeqCst);
                    if !ticking.any_busy() {
                        std::thread::park();
                    }
                    ticking.parked.store(false, Ordering::SeqCst);
                }
            })
            .map_err(|e| LoadError::Runtime(e.to_string()))?;
        Ok(Self {
            engine,
            linker,
            clock: Arc::new(clock),
            ticking,
            ticker: handle.thread().clone(),
            decisions: Arc::clone(decisions),
            thread_clock: thread_cpu_micros().is_some(),
        })
    }

    /// The host's clock now.
    pub fn now_micros(&self) -> u64 {
        (self.clock)()
    }
}

/// What the host asks of the engine: the component model, fuel and epochs
/// for a call's two bounds, and no state for concurrent calls. The host
/// calls a plugin one call at a time and lends it no async import. The
/// runtime reads that switch at each call, and it defaults to on wherever
/// its async component support is compiled in, which any crate of a build
/// can ask for. Said here, a call makes the same allocator calls in every
/// build, and a component that needs that support is refused.
fn configure(config: &mut Config) {
    config.wasm_component_model(true);
    config.concurrency_support(false);
    config.consume_fuel(true);
    config.epoch_interruption(true);
    config.cranelift_opt_level(OptLevel::Speed);
}

/// Why a component is refused at load.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LoadError {
    #[error("the WebAssembly runtime failed: {0}")]
    Runtime(String),
    #[error("plugin {name}: the component does not compile: {reason}")]
    Compile { name: String, reason: String },
    #[error("plugin {name}: imports {import}; no WASI interface is lent to a policy plugin")]
    Wasi { name: String, import: String },
    #[error("plugin {name}: imports {import}; a policy plugin imports smg:policy/host and the package's own smg:policy/types, nothing else")]
    ForeignImport { name: String, import: String },
    #[error(
        "plugin {name}: does not link against the host's imports or the world's shape: {reason}"
    )]
    Link { name: String, reason: String },
    #[error("plugin {name}: exports neither smg:policy/plugin nor smg:policy/plugin-packed")]
    NoWorld { name: String },
    #[error("plugin {name}: targets smg:policy@{found}; this host accepts {accepted}")]
    Abi {
        name: String,
        found: String,
        accepted: String,
    },
    #[error("plugin {name}: targets smg:policy@{found}; the spec pins abi {pinned}")]
    AbiPin {
        name: String,
        found: Abi,
        pinned: Abi,
    },
    #[error("plugin {name}: memory_mib must be at least 1")]
    Memory { name: String },
    #[error("plugin {name}: fuel_per_micro must be at least 1")]
    Fuel { name: String },
    #[error("the plugins ask {asked} MiB of memory per decider; the budget is {budget} MiB")]
    MemoryBudget { asked: u64, budget: u64 },
    #[error("plugin {name}: init failed: {reason}")]
    Init { name: String, reason: String },
    #[error("plugin {name}: trapped at instantiation: {reason}")]
    Trap { name: String, reason: String },
}

/// Which world a module exports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum World {
    /// `smg:policy/policy`: the view as records.
    Record,
    /// `smg:policy/policy-packed`: the view as bytes.
    Packed,
}

enum Pre {
    Record(record::PolicyPre<HostState>),
    Packed(packed::PolicyPackedPre<HostState>),
}

enum Exports {
    Record(record::Policy),
    Packed(packed::PolicyPacked),
}

/// One kept instance: its store, its exports and its marshalling buffers.
struct Instance {
    store: Store<HostState>,
    exports: Exports,
    records: Records,
    packed: Packed,
    snapshot: Vec<u64>,
    /// The place in its plugin's membership feed the instance heard up to;
    /// zero: it is new to the feed.
    heard: u64,
    /// The feed's list, and its changes, as copied out for one telling.
    members: Vec<WorkerId>,
    changes: Vec<Membership<'static>>,
    /// The instance's number on its plugin: a resource is live on the store
    /// that made it and on no other.
    generation: u64,
}

/// What one guest call runs under: the fuel cap of its frame and the
/// decision's remaining fuel (the grant is the smaller), the frame's latency
/// bound, the decision's clock (`None` reads the live one) and its draw
/// sequence (`None` leaves the instance's own).
struct Frame<'a> {
    cap: u64,
    /// The decision's remaining budget, in microseconds; a grant draws it
    /// down by the call's fuel at the plugin's calibration.
    remaining_micros: &'a mut u64,
    fuel_per_micro: u64,
    frame_micros: u32,
    now: Option<u64>,
    draw: Option<&'a mut Draw>,
    /// A call whose answer is nobody's (the destructor): neither answered
    /// nor declined is counted; its faults are.
    quiet: bool,
    /// The decision's number on this thread, zero outside one: the plugin's
    /// own spend in the decision is kept by it.
    decision: u64,
    /// The decision's budget as seeded (zero outside one).
    seed_micros: u64,
}

/// What became of one guest call.
enum Call<T> {
    Answered(T),
    /// The guest declined the stage; the instance stands.
    Declined,
    /// A trap, an exhausted budget or an invalid answer; the instance is
    /// gone and the breaker counted it.
    Failed,
    /// The decision's budget has no fuel left for this call: the rest of
    /// the decision falls back.
    BudgetGone,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    Trap,
    /// The frame ran out, on fuel or on the clock. Under a grant the
    /// decision's remainder had clamped it is the plugin's too: its calls
    /// together exceeded the decision's budget.
    Fuel,
    Invalid,
}

/// The counters of one plugin, read through [`Plugin::stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Stage calls the guest answered.
    pub answered: u64,
    /// Stage calls the guest declined.
    pub declined: u64,
    /// Calls that trapped.
    pub traps: u64,
    /// Calls that ran past the plugin's own frame: out of fuel under a grant
    /// its own spend bounded, or their bound had on the thread's CPU clock
    /// (on the wall clock where a host has no thread clock).
    pub fuel_exhausted: u64,
    /// Decisions whose frame budget was gone on arrival: no call, the stages
    /// fell back, the breaker counted a budget hit.
    pub budget_gone: u64,
    /// Opens and calls the decision's remainder ended short of the plugin's
    /// own bounds (a call granted less than its frame, by a plugin that did
    /// not consume the budget alone): no call, a call out of fuel (its
    /// instance goes, mid-call) or a cold slot's `init` clamped below its
    /// cost; the stage fell back, nothing counted on the breaker, which
    /// judges a plugin against its own frames. What a plugin consumed holds
    /// no time nothing read: a call no reading covered spent its fuel, so a
    /// call starved by a hold before it is counted here (and in `unread`,
    /// where the time the earlier calls took would have made it the
    /// plugin's own).
    pub budget_spent: u64,
    /// Calls whose answer broke a validity rule.
    pub invalid: u64,
    /// Decisions bypassed by the open breaker.
    pub bypassed: u64,
    /// Decisions a packed plugin skipped for a request its form cannot carry.
    pub too_long: u64,
    /// Instances created, in all.
    pub instantiations: u64,
    /// Instances made outside the apply's warm-up: a cold slot's next
    /// decision or event, or the rebuild at a decision's close after a fault.
    pub lazy_instantiations: u64,
    /// Fuel the answered, declined and failed calls consumed, in all.
    pub fuel_used: u64,
    /// `observe` calls made.
    pub observed: u64,
    /// A lease's events and signals dropped: told on a thread that names no
    /// decider and is none of its own (no instance of the plugin came to
    /// stand in its slot), so no instance was theirs to reach and none was
    /// made. The plugin's first writes its one warning. A membership is fed
    /// and never dropped.
    pub events_dropped: u64,
    /// Late calls the host held that answered or declined, by a reading: the
    /// wall clock put the call at or past its bound in whole ticks, and the
    /// thread's CPU clock, read at the call's start and at its end, says the
    /// thread had less than the bound. The host's signal (a starved or
    /// throttled thread), never a failure of the call: the hold itself drops
    /// no instance and moves no breaker. A hold excuses lateness, never a
    /// fault: a held call that failed by itself (a trap, its fuel) is that
    /// failure alone, and an answer the stage then refuses is counted in
    /// `invalid`, the answer's own count, beside its stall. A thread reads
    /// its clock at a call's start only while it is armed (`unread`), so the
    /// first hold of a quiet thread is not here.
    pub stalls: u64,
    /// Events no reading covers, each of which arms its thread. A late
    /// call: the wall clock put it at or past its bound in whole ticks, the
    /// bound did not end it, and its thread had not read its clock at the
    /// call's start. The host held it, or it overran a bound its fuel did
    /// not price; nothing says which, so the call is sampled at the time it
    /// took. And a call that ran out under a remainder where the plugin's
    /// unread calls decide whose it was: by their fuel the budget's, by the
    /// time they took the plugin's. It is the budget's, and `budget_spent`
    /// counts it too. One call can be both, and counts twice. And a
    /// decision's sample that is slow by time nothing read alone, on a
    /// thread nothing armed: past half the budget, or past the budget, by
    /// the time its calls and a cold slot's `init` took, and short of that
    /// mark by their fuel. Through the next 128 decisions the plugin samples
    /// on an armed thread, each of its calls and a cold slot's `init` reads
    /// the clock at its start, and at its end unless it is a call that is not
    /// late and whose fuel covers the time it took. That window is short on a
    /// busy thread (128 ms at a thousand decisions a second), so a host that
    /// holds a thread less often leaves every hold here and none in
    /// `stalls`: read the two together.
    pub unread: u64,
    /// Times the breaker opened.
    pub breaker_opens: u64,
}

#[derive(Default)]
struct Tally {
    answered: AtomicU64,
    declined: AtomicU64,
    traps: AtomicU64,
    fuel_exhausted: AtomicU64,
    budget_gone: AtomicU64,
    budget_spent: AtomicU64,
    invalid: AtomicU64,
    bypassed: AtomicU64,
    too_long: AtomicU64,
    instantiations: AtomicU64,
    lazy_instantiations: AtomicU64,
    fuel_used: AtomicU64,
    observed: AtomicU64,
    events_dropped: AtomicU64,
    stalls: AtomicU64,
    unread: AtomicU64,
}

fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

/// How many membership changes a plugin's feed keeps for the instances
/// behind it. An instance further behind hears the list as it stands.
const FEED_CHANGES: usize = 64;

/// The membership feed of a plugin whose row lists the kind: the pool's one
/// hand-over to every instance of the plugin. The host tells a change once,
/// on whatever thread, and no guest call is made there ([`Plugin::hand`]);
/// each instance hears what is past its own place before its decider's next
/// decision or told event, one call a change ([`Plugin::catch_up`]).
struct Feed {
    /// The place of the last change told, zero before the first: what an
    /// instance compares its own with, without the lock.
    place: AtomicU64,
    told: Mutex<Told>,
}

/// What the host told of the membership.
struct Told {
    /// The membership as it stands: the last snapshot told, with each change
    /// since folded in.
    members: Vec<WorkerId>,
    /// The last changes told, the newest last, [`FEED_CHANGES`] at most. A
    /// snapshot is `None`: the list above holds it.
    changes: VecDeque<Option<Membership<'static>>>,
}

impl Feed {
    fn new() -> Self {
        Self {
            place: AtomicU64::new(0),
            told: Mutex::new(Told {
                members: Vec::new(),
                changes: VecDeque::with_capacity(FEED_CHANGES),
            }),
        }
    }
}

/// What a named decider keeps of a plugin between its decisions: what
/// outlives one, the instance and the latency ring.
struct Rest {
    instance: Option<Instance>,
    ring: ThreadRing,
}

impl Rest {
    fn new() -> Box<Self> {
        Box::new(Self {
            instance: None,
            ring: ThreadRing::new(),
        })
    }
}

/// One named decider's place of rest: empty until the decider first acts.
type Resting = Mutex<Option<Box<Rest>>>;

thread_local! {
    /// How many plugins' named deciders the thread holds the kept state of,
    /// from a bring to its give-back: what a thread under no name reads
    /// before it acts as itself.
    static HOLDS: Cell<u32> = const { Cell::new(0) };
}

/// What a thread is to a plugin, beside what stands in its working slots.
#[derive(Clone, Copy, Default)]
struct Standing {
    /// An instance came to stand in the thread's slot under no name (the
    /// thread decided, warmed or took the apply's own): it is a decider of
    /// its own, and what it tells reaches its instance.
    own: bool,
    /// The named decider whose kept state stands in the thread's working
    /// slots, from the time it is brought to the time it is given back.
    lent: Option<Decider>,
}

/// The apply's own instance (`init` ran at the load, and its failure was the
/// apply's), kept apart until a decider needs one: the thread that loaded
/// takes it at its first need, as if it had stood in its slot since the
/// load, and a named decider that needs an instance takes it where it asks
/// first. So a host that names its deciders makes as many instances as it
/// has deciders, whichever thread loads.
struct Spare {
    instance: Mutex<Option<Instance>>,
    /// Whether the instance still waits: read before the lock.
    left: AtomicBool,
    /// The thread that loaded.
    by: ThreadId,
}

/// A named decider's kept state in a thread's working slots for the span of
/// one call outside a decision: given back at the drop, where this hold
/// brought it.
struct Lent<'a> {
    plugin: &'a Plugin,
    brought: bool,
}

impl Drop for Lent<'_> {
    fn drop(&mut self) {
        if self.brought {
            self.plugin.rest();
        }
    }
}

/// A loaded plugin, shared by every profile that binds it and every thread
/// that decides.
pub struct Plugin {
    runtime: Arc<Runtime>,
    name: &'static str,
    abi: Abi,
    world: World,
    pre: Pre,
    config: Vec<u8>,
    memory_mib: u32,
    frame: FrameBudget,
    exposure: Exposure,
    needs: Needs,
    observes: EventSet,
    failure: Failure,
    log: super::config::LogLimit,
    counters: Counters,
    breaker: Breaker,
    tally: Tally,
    /// A thread's working slot, by `thread_local`'s thread id: the kept
    /// instance of a thread that names no decider (it is a decider of its
    /// own), and the place a named decider's instance is brought to for a
    /// decision or a told event. That crate recycles the id of a thread that
    /// exited, so a later thread may adopt the instance a dead one left: an
    /// instance is `Send`, one live thread holds an id at a time, and the
    /// adopter keeps it warm.
    instances: ThreadLocal<RefCell<Option<Instance>>>,
    /// Where a named decider's kept state rests between its decisions, one
    /// place a name ([`Decider`]), made at the first named use. A place's
    /// lock is taken where a decision brings the state and where it gives it
    /// back, and by no stage call. It is free unless a host lets two threads
    /// act as one decider at once: the second then finds no instance, makes
    /// one, and the later of the two to give back drops the other's.
    rests: OnceLock<Box<[Resting]>>,
    /// Per thread: the thread's own instance (it also decides under no
    /// name) while a named decider's stands in its working slot.
    aside: ThreadLocal<RefCell<Option<Instance>>>,
    /// Per thread: what the thread is to this plugin.
    standing: ThreadLocal<Cell<Standing>>,
    /// The apply's own instance, until a decider needs one.
    spare: Spare,
    /// Whether the plugin's one warning of a dropped event is written.
    drop_said: AtomicBool,
    /// The membership feed, for a plugin whose row lists the kind.
    feed: Option<Feed>,
    /// Instances made so far, numbering each.
    generations: AtomicU64,
    /// Per thread: this decision emptied the slot (a dropped instance), so
    /// its later opens make no instance and its close rebuilds it.
    emptied: ThreadLocal<Cell<bool>>,
    /// Per thread: the decision's number and this plugin's own time in it
    /// (its constructor and stage calls): its spend, so a call the budget
    /// ends is charged to the plugin only once that spend reached the
    /// decision's budget, and its part of the decision's latency sample.
    spent: ThreadLocal<Cell<(u64, OwnTime)>>,
    /// Per thread: the latency ring, one sample per decision (the plugin's
    /// own charged time in it and a lazy `init` the decision paid, over the
    /// decision's budget), kept across rebuilds so a plugin dropped every
    /// decision still completes a window, and the ring's arm.
    latency: ThreadLocal<RefCell<ThreadRing>>,
    /// Per thread: this plugin's part in the open decision (its number, the
    /// plugin's opens of it still to close, each side of a pair and each
    /// binding, so the decision's sample is taken at the plugin's last close
    /// of it, and whether the plugin already failed it: the decision is the
    /// breaker's unit, so it counts one failure however many opens fail),
    /// and the time of a lazy `init` the decision paid at its two ends (part
    /// of its sample, not of its own spend).
    decision_opens: ThreadLocal<Cell<(Part, OwnTime)>>,
}

impl fmt::Debug for Plugin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plugin")
            .field("name", &self.name)
            .field("abi", &self.abi)
            .field("world", &self.world)
            .finish_non_exhaustive()
    }
}

impl Plugin {
    /// Compiles and checks `spec`'s component, instantiates it once on this
    /// thread and runs `init`; an `init` failure fails the load, so a spec
    /// apply that loads its plugins fails whole.
    pub fn load(runtime: &Arc<Runtime>, spec: PluginSpec) -> Result<Arc<Self>, LoadError> {
        let name = spec.name.clone();
        if spec.memory_mib == 0 {
            return Err(LoadError::Memory { name });
        }
        if spec.fuel_per_micro == 0 {
            return Err(LoadError::Fuel { name });
        }
        let component =
            Component::new(&runtime.engine, &spec.component).map_err(|e| LoadError::Compile {
                name: name.clone(),
                reason: format!("{e:#}"),
            })?;
        let ty = component.component_type();
        for (import, _) in ty.imports(&runtime.engine) {
            if import.starts_with("wasi:") {
                return Err(LoadError::Wasi {
                    name,
                    import: import.to_string(),
                });
            }
            // A component of the world imports the package's `types` for its
            // definitions and `host` for its functions; nothing else.
            if import_version(import, "smg:policy/host").is_none()
                && import_version(import, "smg:policy/types").is_none()
            {
                return Err(LoadError::ForeignImport {
                    name,
                    import: import.to_string(),
                });
            }
        }
        let mut found = None;
        for (export, _) in ty.exports(&runtime.engine) {
            if let Some(version) = import_version(export, "smg:policy/plugin-packed") {
                found = Some((World::Packed, version));
            } else if let Some(version) = import_version(export, "smg:policy/plugin") {
                found = Some((World::Record, version));
            }
        }
        let Some((world, version)) = found else {
            return Err(LoadError::NoWorld { name });
        };
        let abi = Abi::parse(version)
            .filter(|a| a.accepted())
            .ok_or_else(|| LoadError::Abi {
                name: name.clone(),
                found: version.to_string(),
                accepted: format!(
                    "{}.0 to {}.{}",
                    Abi::MAJOR,
                    Abi::MAJOR,
                    Abi::SUPPORTED_MINOR
                ),
            })?;
        if let Some(pinned) = spec.abi {
            if pinned != abi {
                return Err(LoadError::AbiPin {
                    name,
                    found: abi,
                    pinned,
                });
            }
        }
        let link = |e: wasmtime::Error| LoadError::Link {
            name: name.clone(),
            reason: format!("{e:#}"),
        };
        let pre = runtime.linker.instantiate_pre(&component).map_err(link)?;
        let pre = match world {
            World::Record => Pre::Record(record::PolicyPre::new(pre).map_err(link)?),
            World::Packed => Pre::Packed(packed::PolicyPackedPre::new(pre).map_err(link)?),
        };
        let frame = spec.frame();
        let plugin = Arc::new(Self {
            runtime: Arc::clone(runtime),
            name: intern(&spec.name),
            abi,
            world,
            pre,
            config: spec.config,
            memory_mib: spec.memory_mib,
            frame,
            exposure: spec.view,
            needs: spec.needs,
            observes: spec.observes,
            failure: spec.failure,
            log: spec.log,
            counters: Counters::new(spec.metrics),
            breaker: Breaker::new(spec.breaker),
            tally: Tally::default(),
            instances: ThreadLocal::new(),
            rests: OnceLock::new(),
            aside: ThreadLocal::new(),
            standing: ThreadLocal::new(),
            spare: Spare {
                instance: Mutex::new(None),
                left: AtomicBool::new(false),
                by: std::thread::current().id(),
            },
            drop_said: AtomicBool::new(false),
            feed: spec
                .observes
                .contains(EventKind::Membership)
                .then(Feed::new),
            generations: AtomicU64::new(0),
            emptied: ThreadLocal::new(),
            spent: ThreadLocal::new(),
            latency: ThreadLocal::new(),
            decision_opens: ThreadLocal::new(),
        });
        // The apply's own instance: `init` runs here, and its failure is the
        // apply's. It waits apart for the first decider that needs one.
        let instance = plugin.instantiate(plugin.init_fuel())?;
        *plugin.spare.instance.lock() = Some(instance);
        plugin.spare.left.store(true, Ordering::Relaxed);
        Ok(plugin)
    }

    /// The total of `memory_mib` over `specs` against the budget, per
    /// decider: what the spec's `plugins.memory_budget_mib` checks at
    /// validation.
    pub fn check_memory_budget(specs: &[PluginSpec], budget_mib: u32) -> Result<(), LoadError> {
        let asked: u64 = specs.iter().map(|s| u64::from(s.memory_mib)).sum();
        if asked > u64::from(budget_mib) {
            return Err(LoadError::MemoryBudget {
                asked,
                budget: u64::from(budget_mib),
            });
        }
        Ok(())
    }

    /// Creates the kept instance of the decider the calling thread acts as
    /// (the thread's own where it names none) when it has none: the apply
    /// calls it as every decider, so no decision pays the instantiation. The
    /// first to ask takes the apply's own instance where it may.
    pub fn warm(&self) -> Result<(), LoadError> {
        let _lent = self.lent_here();
        let slot = self.slot();
        if slot.borrow().is_none() {
            let instance = match self.adopt() {
                Some(instance) => instance,
                None => self.instantiate(self.init_fuel())?,
            };
            *slot.borrow_mut() = Some(instance);
            self.stand();
        }
        // A plugin fed the membership hears it as it stands here, not at
        // the decider's first decision. A telling that faults the instance
        // leaves the slot cold: the fault is the breaker's, and the next
        // decision makes an instance. The mark the fault set is cleared as
        // a told event clears it: the warm-up is no decision's.
        self.catch_up(&mut slot.borrow_mut());
        self.emptied.get_or_default().set(false);
        Ok(())
    }

    /// Whether the decider the calling thread acts as (the thread itself
    /// where it names none) holds a kept instance, or would take the apply's
    /// own: its next decision makes none.
    pub fn is_warm(&self) -> bool {
        let _lent = self.lent_here();
        let kept = self
            .instances
            .get()
            .is_some_and(|slot| slot.borrow().is_some());
        kept || self.spare_is_mine()
    }

    /// Forgets the kept instance of the decider the calling thread acts as
    /// (the thread's own where it names none), so the next decision or
    /// `observe` on it instantiates again: what a host without a kept
    /// instance would pay per decision, for the measurements.
    #[doc(hidden)]
    pub fn forget_instance(&self) {
        let _lent = self.lent_here();
        drop(self.adopt());
        *self.slot().borrow_mut() = None;
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn abi(&self) -> Abi {
        self.abi
    }

    pub fn world(&self) -> World {
        self.world
    }

    pub fn failure(&self) -> Failure {
        self.failure
    }

    /// What the host must feed for the plugin's stages, as declared. Each
    /// stage the plugin is bound as carries these into its profile's needs
    /// ([`crate::Profile::needs`]), and so into the engine's summary.
    pub fn needs(&self) -> Needs {
        self.needs
    }

    /// The kinds of lifecycle event the plugin hears, as its row lists
    /// them.
    pub fn observes(&self) -> EventSet {
        self.observes
    }

    /// The counters the plugin adds to.
    pub fn counters(&self) -> &Counters {
        &self.counters
    }

    pub fn stats(&self) -> Stats {
        let t = &self.tally;
        Stats {
            answered: t.answered.load(Ordering::Relaxed),
            declined: t.declined.load(Ordering::Relaxed),
            traps: t.traps.load(Ordering::Relaxed),
            fuel_exhausted: t.fuel_exhausted.load(Ordering::Relaxed),
            budget_gone: t.budget_gone.load(Ordering::Relaxed),
            budget_spent: t.budget_spent.load(Ordering::Relaxed),
            invalid: t.invalid.load(Ordering::Relaxed),
            bypassed: t.bypassed.load(Ordering::Relaxed),
            too_long: t.too_long.load(Ordering::Relaxed),
            instantiations: t.instantiations.load(Ordering::Relaxed),
            lazy_instantiations: t.lazy_instantiations.load(Ordering::Relaxed),
            fuel_used: t.fuel_used.load(Ordering::Relaxed),
            observed: t.observed.load(Ordering::Relaxed),
            events_dropped: t.events_dropped.load(Ordering::Relaxed),
            stalls: t.stalls.load(Ordering::Relaxed),
            unread: t.unread.load(Ordering::Relaxed),
            breaker_opens: self.breaker.opens(),
        }
    }

    /// One engine build's hold on the plugin: every profile of the build
    /// that binds the plugin takes its bound set from the one binder, and a
    /// later build takes a binder of its own.
    pub fn binder(self: &Arc<Self>) -> Binder {
        Binder {
            plugin: Arc::clone(self),
            observer_taken: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Binds the plugin to the one profile of a build that binds it once:
    /// the stage set that shares one `decision` resource per decision. A
    /// build that binds it to several profiles takes each set from one
    /// [`Plugin::binder`]; taken here, each set would forward the lifecycle.
    pub fn bind(self: &Arc<Self>) -> Arc<Bound> {
        self.binder().bind()
    }

    fn slot(&self) -> &RefCell<Option<Instance>> {
        self.instances.get_or_default()
    }

    /// This thread's latency ring.
    fn ring(&self) -> &RefCell<ThreadRing> {
        self.latency.get_or(|| RefCell::new(ThreadRing::new()))
    }

    /// Whether the apply's own instance still waits and the caller may take
    /// it: the thread that loaded, or a thread acting as a named decider.
    fn spare_is_mine(&self) -> bool {
        self.spare.left.load(Ordering::Relaxed)
            && (Decider::acting().is_some() || std::thread::current().id() == self.spare.by)
    }

    /// The apply's own instance, where it still waits for the caller: no
    /// `init` runs and nothing is charged.
    fn adopt(&self) -> Option<Instance> {
        if !self.spare_is_mine() {
            return None;
        }
        let taken = self.spare.instance.lock().take();
        if taken.is_some() {
            self.spare.left.store(false, Ordering::Relaxed);
            self.stand();
        }
        taken
    }

    /// An instance came to stand in this thread's working slot. Under no
    /// name the thread is a decider of its own from here on: what it tells
    /// reaches its instance.
    fn stand(&self) {
        if Decider::acting().is_none() {
            let standing = self.standing.get_or_default();
            standing.set(Standing {
                own: true,
                ..standing.get()
            });
        }
    }

    /// Where `decider`'s kept state rests.
    fn rest_of(&self, decider: Decider) -> &Resting {
        let rests = self
            .rests
            .get_or_init(|| (0..Decider::COUNT).map(|_| Mutex::new(None)).collect());
        &rests[usize::from(decider.number())]
    }

    /// Brings the kept state of the decider the calling thread acts as into
    /// the thread's working slots, where the thread names one and does not
    /// hold it already, and says whether it did. A decision's first open
    /// brings it and the plugin's last close of the decision gives it back;
    /// a told event holds it for its one call. What the thread never gave
    /// back (a close that did not return) goes back first, under another
    /// name or under none: a thread decides on no instance but the one of
    /// the decider it acts as.
    fn lend(&self) -> bool {
        let Some(decider) = Decider::acting() else {
            // Under no name the thread acts as itself. One count says
            // whether it holds any named decider's state, so a host that
            // names none pays one load here.
            if HOLDS.get() != 0 {
                self.rest();
            }
            return false;
        };
        let standing = self.standing.get_or_default();
        let mut now = standing.get();
        match now.lent {
            Some(held) if held == decider => return false,
            Some(held) => self.give_back(held),
            None => {}
        }
        let mut rest = self.rest_of(decider).lock();
        let rest = rest.get_or_insert_with(Rest::new);
        let mut slot = self.slot().borrow_mut();
        if slot.is_some() {
            // The thread's own instance (it also decides under no name)
            // stands aside while the thread acts as the decider.
            *self.aside.get_or_default().borrow_mut() = slot.take();
        }
        *slot = rest.instance.take();
        std::mem::swap(&mut rest.ring, &mut *self.ring().borrow_mut());
        now.lent = Some(decider);
        standing.set(now);
        HOLDS.set(HOLDS.get() + 1);
        true
    }

    /// Gives the working slots back to the named decider whose kept state
    /// stands in them, where one does.
    fn rest(&self) {
        let Some(standing) = self.standing.get() else {
            return;
        };
        let mut now = standing.get();
        if let Some(held) = now.lent.take() {
            standing.set(now);
            self.give_back(held);
        }
    }

    /// The working slots' instance and ring go to `decider`'s rest, and the
    /// thread's own come back. An instance already resting there was made by
    /// a second thread acting as the decider at the same time, against the
    /// turn's rule: it is dropped here.
    fn give_back(&self, decider: Decider) {
        HOLDS.set(HOLDS.get().saturating_sub(1));
        let mut rest = self.rest_of(decider).lock();
        let rest = rest.get_or_insert_with(Rest::new);
        let mut slot = self.slot().borrow_mut();
        rest.instance = slot.take();
        if let Some(aside) = self.aside.get() {
            *slot = aside.borrow_mut().take();
        }
        std::mem::swap(&mut rest.ring, &mut *self.ring().borrow_mut());
    }

    /// The acting decider's kept state in this thread's working slots until
    /// the guard drops.
    fn lent_here(&self) -> Lent<'_> {
        Lent {
            plugin: self,
            brought: self.lend(),
        }
    }

    /// The start of a told event: the kept state of the decider the calling
    /// thread acts as is this thread's until the guard drops. `None` where
    /// the event has no instance to reach: the thread names no decider and
    /// is none of its own (no instance came to stand in its slot, and the
    /// apply's is not its to take), so it only tells, and an instance made
    /// for it would hear what no decision reads. Such an event is dropped
    /// and counted, and the plugin's first writes its one warning.
    fn telling(&self) -> Option<Lent<'_>> {
        if Decider::acting().is_none()
            && !self
                .standing
                .get()
                .is_some_and(|standing| standing.get().own)
            && !self.spare_is_mine()
        {
            bump(&self.tally.events_dropped);
            if !self.drop_said.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    plugin = self.name,
                    "a lifecycle event was told on a thread that names no decider and holds no \
                     instance of the plugin: it is dropped and counted, as each such event is; a \
                     host tells a lease's events as the decider that issued the lease"
                );
            }
            return None;
        }
        Some(self.lent_here())
    }

    /// The fuel of `init` when nothing clamps it.
    fn init_fuel(&self) -> u64 {
        self.frame.fuel_of(INIT_FRAME_MICROS)
    }

    /// A new instance, its `init` run under `init_cap` fuel.
    fn instantiate(&self, init_cap: u64) -> Result<Instance, LoadError> {
        self.instantiate_metered(init_cap).0
    }

    /// The same, with the fuel `init` used however it ended: what a decision
    /// that instantiates lazily is charged.
    fn instantiate_metered(&self, init_cap: u64) -> (Result<Instance, LoadError>, u64) {
        bump(&self.tally.instantiations);
        let limits = StoreLimitsBuilder::new()
            .memory_size(
                usize::try_from(self.memory_mib)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(MIB),
            )
            // One linear memory per instance, so the cap is the instance's;
            // tables stop at a size no policy needs, so `table.grow` is bounded.
            .memories(1)
            .tables(TABLES)
            .instances(INSTANCES)
            .table_elements(TABLE_ELEMENTS)
            .trap_on_grow_failure(true)
            .build();
        let now = Instant::now();
        let mut store = Store::new(
            &self.runtime.engine,
            HostState {
                limits,
                name: self.name,
                decision_micros: None,
                draw: Draw::new(0),
                counters: self.counters.clone(),
                log: LogGate::new(self.log, now),
                clock: Arc::clone(&self.runtime.clock),
                watch: Watch::default(),
            },
        );
        store.limiter(|state| &mut state.limits);
        // A tick that finds a guest call in flight on this store.
        store.epoch_deadline_callback(|mut context| tick(&mut context.data_mut().watch));
        let fuel = init_cap;
        if let Err(e) = store.set_fuel(fuel) {
            return (Err(LoadError::Runtime(e.to_string())), 0);
        }
        let init_micros =
            u32::try_from(init_cap / self.frame.fuel_per_micro.max(1)).unwrap_or(u32::MAX);
        store.set_epoch_deadline(deadline_ticks(init_micros));
        let _running = Running::start(&self.runtime);
        let made = self.init_in(&mut store);
        let used = fuel.saturating_sub(store.get_fuel().unwrap_or(0));
        self.tally.fuel_used.fetch_add(used, Ordering::Relaxed);
        let instance = made.map(|exports| Instance {
            store,
            exports,
            records: Records::default(),
            packed: Packed::default(),
            snapshot: Vec::new(),
            heard: 0,
            members: Vec::new(),
            changes: Vec::new(),
            generation: self.generations.fetch_add(1, Ordering::Relaxed) + 1,
        });
        (instance, used)
    }

    /// Instantiates the component in `store` and runs its `init`.
    fn init_in(&self, store: &mut Store<HostState>) -> Result<Exports, LoadError> {
        let trap = |e: wasmtime::Error| LoadError::Trap {
            name: self.name.to_string(),
            reason: format!("{e:#}"),
        };
        match &self.pre {
            Pre::Record(pre) => {
                let bindings = pre.instantiate(&mut *store).map_err(trap)?;
                let init = bindings
                    .smg_policy_plugin()
                    .call_init(&mut *store, &self.config)
                    .map_err(trap)?;
                if let Err(reason) = init {
                    return Err(LoadError::Init {
                        name: self.name.to_string(),
                        reason,
                    });
                }
                Ok(Exports::Record(bindings))
            }
            Pre::Packed(pre) => {
                let bindings = pre.instantiate(&mut *store).map_err(trap)?;
                let init = bindings
                    .smg_policy_plugin_packed()
                    .call_init(&mut *store, &self.config)
                    .map_err(trap)?;
                if let Err(reason) = init {
                    return Err(LoadError::Init {
                        name: self.name.to_string(),
                        reason,
                    });
                }
                Ok(Exports::Packed(bindings))
            }
        }
    }

    /// Runs one guest call on this thread's instance under `frame`: the
    /// grant is the frame's fuel or the decision's remainder, whichever is
    /// less; the latency counts against the frame, the fuel against the
    /// decision; the failure rules apply.
    fn call<T>(
        &self,
        instance: &mut Option<Instance>,
        mut frame: Frame<'_>,
        f: impl FnOnce(&mut Instance) -> wasmtime::Result<Result<T, wit::Decline>>,
    ) -> Call<T> {
        let Some(inst) = instance.as_mut() else {
            return Call::Failed;
        };
        let fuel_per_micro = frame.fuel_per_micro.max(1);
        let grant = frame
            .cap
            .min(frame.remaining_micros.saturating_mul(fuel_per_micro));
        if grant == 0 {
            // The decision's earlier calls spent its remainder: no call is
            // made, the stage falls back, the instance stands; nothing counts
            // on the breaker, which judged the calls that spent it.
            bump(&self.tally.budget_spent);
            return Call::BudgetGone;
        }
        if inst.store.set_fuel(grant).is_err() {
            return Call::Failed;
        }
        // The bound on the clock is what the decision has left, at most the
        // frame. On a host with a thread clock it counts the time this thread
        // has the CPU since a reading (`tick`): the call's start on a thread
        // whose ring is armed, the reading the call's end is judged on;
        // otherwise the first tick the call meets. Elsewhere the call ends at
        // its deadline on the wall clock.
        let bound = frame
            .frame_micros
            .min(u32::try_from(*frame.remaining_micros).unwrap_or(u32::MAX));
        let watched = self.runtime.thread_clock;
        let start = if watched && self.armed() {
            thread_cpu_micros()
        } else {
            None
        };
        inst.store
            .set_epoch_deadline(first_deadline(watched, start.is_some(), bound));
        let state = inst.store.data_mut();
        state.decision_micros = frame.now;
        if let Some(draw) = frame.draw.as_deref_mut() {
            std::mem::swap(&mut state.draw, draw);
        }
        state.watch = Watch {
            bound_micros: u64::from(bound),
            watched,
            start,
            first_tick: None,
        };
        let started = Instant::now();
        let answer = {
            let _running = Running::start(&self.runtime);
            f(inst)
        };
        let elapsed = u32::try_from(started.elapsed().as_micros()).unwrap_or(u32::MAX);
        let used = grant.saturating_sub(inst.store.get_fuel().unwrap_or(0));
        // The decision's deadline is on the wall clock: the call draws the
        // remainder by the larger of its fuel at the calibration and its
        // elapsed time, so a call the calibration underprices, a call long in
        // its host calls and a call the host held all draw it down.
        let fuel_micros = used.div_ceil(fuel_per_micro);
        let charged = fuel_micros.max(u64::from(elapsed));
        *frame.remaining_micros = frame.remaining_micros.saturating_sub(charged);
        let state = inst.store.data_mut();
        state.decision_micros = None;
        let watch = std::mem::take(&mut state.watch);
        if let Some(draw) = frame.draw {
            std::mem::swap(&mut state.draw, draw);
        }
        let end = match &answer {
            Ok(_) => End::Answered,
            Err(error) if error.downcast_ref::<Trap>() == Some(&Trap::Interrupt) => End::OnTheClock,
            Err(_) => End::Failed,
        };
        // The plugin's own time in the call is what it could spend; a late
        // call is judged once, here, with its answer read. One a reading
        // says the host held, and that answered, is the host's signal, not
        // the plugin's failure; one that failed by itself is that failure
        // alone. One no reading covers is counted by its own name and arms
        // the thread: the next calls there have a reading.
        let (own, verdict) = judge(
            &watch,
            u64::from(elapsed),
            fuel_micros,
            end,
            thread_cpu_micros,
        );
        match verdict {
            Verdict::Own => {}
            Verdict::Held => bump(&self.tally.stalls),
            Verdict::Unread => {
                bump(&self.tally.unread);
                self.arm();
            }
        }
        // The plugin's own time in the decision before this call: its two
        // sums.
        let before = if frame.decision == 0 {
            OwnTime::default()
        } else {
            self.spend(frame.decision, own)
        };
        self.tally.fuel_used.fetch_add(used, Ordering::Relaxed);
        match answer {
            Ok(Ok(value)) => {
                if !frame.quiet {
                    bump(&self.tally.answered);
                }
                Call::Answered(value)
            }
            Ok(Err(_decline)) => {
                if !frame.quiet {
                    bump(&self.tally.declined);
                }
                Call::Declined
            }
            Err(error) => {
                // Out of fuel, or its bound had on the clock: the frame ran
                // out.
                let frame_ran_out = matches!(
                    error.downcast_ref::<Trap>(),
                    Some(Trap::OutOfFuel | Trap::Interrupt)
                );
                // A call the budget ended is charged to the plugin only when
                // the bound it ran out under was its own: the call was granted
                // its whole frame and burned it, or the plugin's own spend in
                // the decision (its constructor, its earlier calls and this
                // grant in full) reaches the budget, so it alone consumed it.
                // Short of both, the remainder clamped the call (other plugins
                // drew it, or the host held a call of this one), and it is
                // the decision's budget spent, whichever scope ran first: the
                // interrupted instance goes (its memory is mid-call), the
                // breaker is untouched, and chronic slowness opens it through
                // the decision's latency sample instead. Judged on the grant,
                // not the clock, so a stalled call is not charged for the
                // stall; and against the spend, the low end of the time the
                // earlier calls took ([`OwnTime`]), so a call a hold starved is
                // not charged for the hold before it. A host with no thread
                // clock has no low end: there the spend is the time they took.
                let granted = grant.div_ceil(fuel_per_micro);
                let whole = grant == frame.cap;
                let under_its_own = own_bound(whole, before.spend, granted, frame.seed_micros);
                if frame_ran_out && frame.decision != 0 && !under_its_own {
                    tracing::debug!(
                        plugin = self.name,
                        "a call ran out under the decision's remainder"
                    );
                    bump(&self.tally.budget_spent);
                    if own_bound(whole, before.sample, granted, frame.seed_micros) {
                        // The two ends part: the time the plugin's unread
                        // calls took reaches the budget, their fuel does
                        // not, and nothing says whether a hold or their own
                        // time past their fuel is the difference. The
                        // run-out is the budget's, counted with the calls
                        // nothing read, and it arms the thread: its next
                        // calls are read, and a run-out behind them has one
                        // answer.
                        bump(&self.tally.unread);
                        self.arm();
                    }
                    self.drop_instance(instance);
                    return Call::Failed;
                }
                let fault = if frame_ran_out {
                    Fault::Fuel
                } else {
                    Fault::Trap
                };
                tracing::debug!(plugin = self.name, error = %format!("{error:#}"), "a guest call failed");
                self.fault(instance, fault);
                Call::Failed
            }
        }
    }

    /// A trap, an exhausted budget or an invalid answer: the instance is
    /// dropped (the decision's close rebuilds the slot), the breaker counts it.
    fn fault(&self, instance: &mut Option<Instance>, fault: Fault) {
        match fault {
            Fault::Trap => bump(&self.tally.traps),
            Fault::Fuel => bump(&self.tally.fuel_exhausted),
            Fault::Invalid => bump(&self.tally.invalid),
        }
        self.drop_instance(instance);
        self.fail(self.runtime.now_micros());
    }

    /// One failure on the breaker, marked on the decision it fell in. The
    /// decision is the breaker's unit: once a plugin failed one, its later
    /// opens of it bypass and its close counts no second failure, so a
    /// threshold of N failures is N decisions on a pair as on a single
    /// placement. Outside a decision (a lifecycle event) nothing is marked.
    fn fail(&self, now: u64) {
        let opens = self.decision_opens.get_or_default();
        let (mut part, init) = opens.get();
        part.fail();
        opens.set((part, init));
        self.report(now);
    }

    /// One failure reported to the breaker; an opening writes its line.
    fn report(&self, now: u64) {
        if self.breaker.failure(now) {
            self.opened("failures");
        }
    }

    /// The breaker opened under `rule`: the one line of an opening. For the
    /// cool-down every stage of the plugin takes its fallback, and nothing
    /// else says so as it happens. A report made while the breaker is open
    /// opens nothing, so an incident is one line: one a cool-down at most.
    fn opened(&self, rule: &'static str) {
        tracing::warn!(
            plugin = self.name,
            rule,
            cooldown_micros = self.breaker.cooldown_micros(),
            "the plugin's breaker opened: its stages take their fallback for the cool-down"
        );
    }

    /// Whether this plugin already failed decision `decision` on this
    /// thread; true until the thread's next decision opens it. Read at the
    /// close, and only when a rebuild failed.
    fn failed_in(&self, decision: u64) -> bool {
        let (part, _) = self.decision_opens.get_or_default().get();
        part.failed_in(decision)
    }

    /// One of this plugin's opens of decision `decision` on this thread;
    /// says whether the plugin already failed that decision (the one read
    /// of the cell an open makes).
    fn note_open(&self, decision: u64) -> bool {
        let opens = self.decision_opens.get_or_default();
        let (mut part, init) = opens.get();
        let init = if part.is_of(decision) {
            init
        } else {
            OwnTime::default()
        };
        let failed = part.open(decision);
        opens.set((part, init));
        failed
    }

    /// The time of a lazy `init` decision `decision` paid on this thread:
    /// judged under the init frame, never the plugin's own spend, but part of
    /// what the decision cost it, at its two ends like a call's.
    fn note_init(&self, decision: u64, own: OwnTime) {
        let opens = self.decision_opens.get_or_default();
        let (part, init) = opens.get();
        if part.is_of(decision) {
            opens.set((part, init.plus(own)));
        }
    }

    /// One of this plugin's closes of decision `decision`: the last one takes
    /// the decision's latency sample, the plugin's own charged time in the
    /// decision (its constructor and stage calls, whatever ended them) plus
    /// the time of a lazy `init` the decision paid, over the decision's
    /// budget. A decision that charged the plugin nothing records nothing.
    /// Says whether this close was the plugin's last of the decision.
    fn note_close(&self, decision: u64, seed_micros: u64) -> bool {
        let opens = self.decision_opens.get_or_default();
        let (mut part, init) = opens.get();
        let last = part.close(decision);
        opens.set((part, init));
        if !last {
            return false;
        }
        let (spent_stamp, spent) = self.spent.get_or_default().get();
        let own = if spent_stamp == decision {
            spent.plus(init)
        } else {
            init
        };
        if own.sample > 0 {
            self.sample(own, seed_micros);
        }
        true
    }

    /// One latency sample into the thread's ring: a decision's time as a
    /// share of its budget, at the high end of what nothing read; a window
    /// that reads slow opens the breaker. A sample slow at that end alone
    /// (past a mark of the ring its low end is short of) is slow by time
    /// nothing read, a hold of the host or the guest's own, and nothing says
    /// which. It is recorded as it is, counted with the unread calls, and
    /// arms the thread: the decisions that share its window are read, so no
    /// window holds two such samples. That bounds the samples no reading
    /// covered, not what one of them completes: the ring's allowance of one
    /// outlier is the window's, and a hold of the host spends it.
    fn sample(&self, own: OwnTime, seed_micros: u64) {
        let micros = |micros: u64| u32::try_from(micros).unwrap_or(u32::MAX);
        let (sample, low, seed) = (micros(own.sample), micros(own.spend), micros(seed_micros));
        let slow = {
            let mut ring = self
                .latency
                .get_or(|| RefCell::new(ThreadRing::new()))
                .borrow_mut();
            if !ring.armed() && ThreadRing::uncovered(sample, low, seed) {
                bump(&self.tally.unread);
                ring.arm();
            }
            ring.record(sample, seed)
        };
        if slow && self.breaker.slow(self.runtime.now_micros()) {
            self.opened("latency");
        }
    }

    /// Whether this thread reads its CPU clock at the start and the end of
    /// the plugin's calls: its ring is armed.
    fn armed(&self) -> bool {
        self.latency.get().is_some_and(|ring| ring.borrow().armed())
    }

    /// Arms this thread's ring. A late call here had no reading of its
    /// start, and its decision's sample is the time the call took; or a call
    /// here ran out under a remainder that the time its plugin's unread
    /// calls took makes its own and their fuel does not. (A sample slow by
    /// time nothing read arms the ring where it is taken.)
    fn arm(&self) {
        self.latency
            .get_or(|| RefCell::new(ThreadRing::new()))
            .borrow_mut()
            .arm();
    }

    /// Drops this thread's instance: its store is gone with it, and this
    /// decision's close rebuilds the slot.
    fn drop_instance(&self, instance: &mut Option<Instance>) {
        *instance = None;
        self.emptied.get_or_default().set(true);
    }

    /// Adds a call's own time to this plugin's two sums of decision
    /// `decision` on this thread and returns the sums before it.
    fn spend(&self, decision: u64, own: OwnTime) -> OwnTime {
        let spent = self.spent.get_or_default();
        let (stamp, before) = spent.get();
        let before = if stamp == decision {
            before
        } else {
            OwnTime::default()
        };
        spent.set((decision, before.plus(own)));
        before
    }

    /// One membership change of the host, told on whatever thread: folded
    /// into the feed's list and kept among its last changes. No guest call
    /// is made here and no instance: each instance hears it before its
    /// decider's next decision or told event ([`Plugin::catch_up`]).
    fn hand(&self, change: &Membership<'_>) {
        let Some(feed) = &self.feed else {
            return;
        };
        let mut told = feed.told.lock();
        let kept = match *change {
            Membership::Snapshot(members) => {
                told.members.clear();
                told.members.extend_from_slice(members);
                None
            }
            Membership::Added(worker) => {
                if !told.members.contains(&worker) {
                    told.members.push(worker);
                }
                Some(Membership::Added(worker))
            }
            Membership::Removed(worker) => {
                told.members.retain(|member| *member != worker);
                Some(Membership::Removed(worker))
            }
            Membership::Replaced { old, new } => {
                if let Some(member) = told.members.iter_mut().find(|member| **member == old) {
                    *member = new;
                } else if !told.members.contains(&new) {
                    told.members.push(new);
                }
                Some(Membership::Replaced { old, new })
            }
            Membership::Relabelled(worker) => Some(Membership::Relabelled(worker)),
            Membership::Resized(worker) => Some(Membership::Resized(worker)),
        };
        if told.changes.len() == FEED_CHANGES {
            told.changes.pop_front();
        }
        told.changes.push_back(kept);
        // Under the lock: the changes a place names are the ones the lock
        // shows whoever read that place.
        feed.place.fetch_add(1, Ordering::Release);
    }

    /// Tells the instance in `slot` what the membership feed holds past the
    /// instance's place, before its decider reads it: nothing where it is
    /// level (one load and one compare); each change, in the host's order,
    /// where the feed still keeps them all; the list as it stands, as one
    /// snapshot, where the instance is new to the feed (made at the warm-up,
    /// lazily or after a fault), further behind than the changes kept, or
    /// behind a snapshot. So an instance hears each change once, or a
    /// snapshot that holds it. A change that faults the instance ends the
    /// telling: the instance made after it is new to the feed. Each call is
    /// a told event's ([`Plugin::tell`]), also where a decision's open makes
    /// it: a frame of its own, outside the decision's sums.
    fn catch_up(&self, slot: &mut Option<Instance>) {
        let (Some(feed), Some(instance)) = (&self.feed, slot.as_mut()) else {
            return;
        };
        if instance.heard == feed.place.load(Ordering::Acquire) {
            return;
        }
        let mut ids = std::mem::take(&mut instance.snapshot);
        let mut members = std::mem::take(&mut instance.members);
        let mut changes = std::mem::take(&mut instance.changes);
        members.clear();
        changes.clear();
        // The copy is made under the lock and the calls after it: no teller
        // waits on a guest.
        let whole = {
            let told = feed.told.lock();
            let place = feed.place.load(Ordering::Relaxed);
            let behind = usize::try_from(place - instance.heard).unwrap_or(usize::MAX);
            let whole = instance.heard == 0
                || behind > told.changes.len()
                || told.changes.iter().rev().take(behind).any(Option::is_none);
            if whole {
                members.extend_from_slice(&told.members);
            } else {
                let level = told.changes.len() - behind;
                changes.extend(told.changes.iter().skip(level).flatten());
            }
            instance.heard = place;
            whole
        };
        if whole {
            let event = marshal::membership(&Membership::Snapshot(&members), &mut ids);
            self.tell(slot, wit::LifecycleEvent::Membership(event));
        } else {
            for change in &changes {
                if slot.is_none() {
                    break;
                }
                let event = marshal::membership(change, &mut ids);
                self.tell(slot, wit::LifecycleEvent::Membership(event));
            }
        }
        if let Some(instance) = slot.as_mut() {
            instance.snapshot = ids;
            instance.members = members;
            instance.changes = changes;
        }
    }

    /// Delivers one lifecycle event to the kept instance of the decider the
    /// calling thread acts as: the thread's own where it names none and is
    /// one. A thread that only tells delivers nothing ([`Plugin::telling`]).
    fn observe(&self, event: wit::LifecycleEvent<'_>) {
        let Some(_lent) = self.telling() else {
            return;
        };
        let slot = self.slot();
        let mut guard = slot.borrow_mut();
        let now = self.runtime.now_micros();
        if self.breaker.is_open(now) {
            return;
        }
        if guard.is_none() {
            *guard = self.adopt();
        }
        if guard.is_none() {
            match self.instantiate(self.init_fuel()) {
                Ok(instance) => {
                    bump(&self.tally.lazy_instantiations);
                    *guard = Some(instance);
                    self.stand();
                }
                Err(_) => {
                    self.report(now);
                    return;
                }
            }
        }
        // What the membership feed holds past this instance's place comes
        // first: the instance hears in the order the host told.
        self.catch_up(&mut guard);
        if guard.is_some() {
            self.tell(&mut guard, event);
        }
        // Outside a decision: a dropped instance is made again lazily by the
        // next event or open. The mark is a decision's, for its close; a
        // decision never rebuilds a slot it did not touch.
        self.emptied.get_or_default().set(false);
    }

    /// One `observe` call on the instance in `slot`, outside every
    /// decision's sums: the live clock, the instance's own draw, the plugin's
    /// frame (the event frame where its budget is unset).
    fn tell(&self, slot: &mut Option<Instance>, event: wit::LifecycleEvent<'_>) {
        let frame = if self.frame.micros == 0 {
            OBSERVE_FRAME_MICROS
        } else {
            self.frame.micros
        };
        let fuel = self.frame.fuel_of(frame);
        bump(&self.tally.observed);
        // Outside a decision: the live clock, the instance's own draw.
        let mut remaining_micros = u64::from(frame);
        let lifecycle = Frame {
            cap: fuel,
            remaining_micros: &mut remaining_micros,
            fuel_per_micro: self.frame.fuel_per_micro,
            frame_micros: frame,
            now: None,
            draw: None,
            quiet: false,
            decision: 0,
            seed_micros: 0,
        };
        let _ = self.call(slot, lifecycle, |inst| {
            let result = match &inst.exports {
                Exports::Record(b) => b.smg_policy_plugin().call_observe(&mut inst.store, event),
                Exports::Packed(b) => b
                    .smg_policy_plugin_packed()
                    .call_observe(&mut inst.store, event),
            };
            result.map(Ok)
        });
    }
}

/// The version of `name@major.minor.patch` when `name` is the interface
/// asked for.
fn import_version<'a>(full: &'a str, name: &str) -> Option<&'a str> {
    let rest = full.strip_prefix(name)?;
    rest.strip_prefix('@')
}

/// The decision a bound plugin has open on this thread for one phase: the
/// resource, the budget of its calls, and the answers already taken (a
/// filter answers once per decision, a scorer once; the pair plan repeats
/// the pick per prefill target). The buffers keep their capacity across
/// decisions.
struct Open {
    resource: Option<ResourceAny>,
    /// The generation of the instance that made `resource`.
    generation: u64,
    is_open: bool,
    /// The decision's number, as this open of it returned it.
    decision: u64,
    /// Every stage of this decision takes its fallback: the breaker is open,
    /// the budget is gone, the instance failed, or the request does not pack.
    bypass: bool,
    count: u32,
    /// The fuel cap of one call: the plugin's frame, clamped to the request's
    /// budget (the decision's remainder is the thread's, shared).
    frame_fuel: u64,
    frame_micros: u32,
    /// The decision's clock and draw sequence: what `now-micros` and
    /// `random-u64` read inside this decision's calls, and no other's.
    now: u64,
    draw: Draw,
    /// A declined filter or scorer declines for the whole decision: the
    /// guest is not asked again row by row.
    filter_declined: bool,
    score_declined: bool,
    /// The filter's answer as bits over the candidates, once `mask_set`.
    mask: Vec<u64>,
    mask_set: bool,
    /// The scorer's answer, one per candidate, clamped, once `scores_set`.
    scores: Vec<f32>,
    scores_set: bool,
    /// The picker's input: the combined scores, `-1.0` on a filtered row.
    scores_in: Vec<f32>,
}

impl Default for Open {
    fn default() -> Self {
        Self {
            resource: None,
            is_open: false,
            decision: 0,
            bypass: false,
            count: 0,
            generation: 0,
            frame_fuel: 0,
            frame_micros: 0,
            now: 0,
            draw: Draw::new(0),
            filter_declined: false,
            score_declined: false,
            mask: Vec::new(),
            mask_set: false,
            scores: Vec::new(),
            scores_set: false,
            scores_in: Vec::new(),
        }
    }
}

impl Open {
    /// Forgets the decision; the buffers keep their capacity.
    fn reset(&mut self) {
        self.resource = None;
        self.is_open = false;
        self.decision = 0;
        self.bypass = false;
        self.count = 0;
        self.generation = 0;
        self.frame_fuel = 0;
        self.frame_micros = 0;
        self.now = 0;
        self.filter_declined = false;
        self.score_declined = false;
        self.mask.clear();
        self.mask_set = false;
        self.scores.clear();
        self.scores_set = false;
    }

    /// What this decision's next call runs under, drawing on the thread's
    /// decision budget at the plugin's calibration.
    fn frame<'a>(&'a mut self, budget: &'a mut OpenDecision, fuel_per_micro: u64) -> Frame<'a> {
        Frame {
            cap: self.frame_fuel,
            remaining_micros: &mut budget.remaining_micros,
            fuel_per_micro,
            frame_micros: self.frame_micros,
            now: Some(self.now),
            draw: Some(&mut self.draw),
            quiet: false,
            decision: self.decision,
            seed_micros: budget.seed_micros,
        }
    }

    /// The decision's resource, live on the kept instance: none once the
    /// decision is bypassed, or once the store that made it is gone (a fault
    /// dropped it, and a rebuilt instance would refuse the handle).
    fn live(&self, slot: Option<&Instance>) -> Result<ResourceAny, Fallback> {
        if self.bypass || slot.is_none_or(|i| i.generation != self.generation) {
            return Err(Fallback);
        }
        self.resource.ok_or(Fallback)
    }

    /// An answer outside the contract is a fault of the plugin: the instance
    /// goes, and the rest of this decision falls back.
    fn invalid(&mut self, plugin: &Plugin, instance: &mut Option<Instance>) -> Fallback {
        plugin.fault(instance, Fault::Invalid);
        self.bypass = true;
        self.resource = None;
        Fallback
    }

    /// Reads a call's outcome into the open decision: a failure bypasses the
    /// rest of the decision (the resource went with the instance).
    fn answered<T>(&mut self, call: Call<T>) -> Result<T, Fallback> {
        match call {
            Call::Answered(value) => Ok(value),
            Call::Declined => Err(Fallback),
            Call::BudgetGone => {
                self.bypass = true;
                Err(Fallback)
            }
            Call::Failed => {
                self.bypass = true;
                self.resource = None;
                Err(Fallback)
            }
        }
    }
}

/// One open decision per phase, in the framework's phase slots: the pair
/// plan opens the prefill side and the decode side of one decision, and a
/// binding may name one profile for both, so a bound set keeps the two apart
/// by the phase the stages run under.
#[derive(Default)]
struct Opens([Open; 4]);

/// One engine build's hold on a plugin. The engine hands every lifecycle
/// event to the scorers and the picker of every profile, and a plugin bound
/// to several profiles hears it once: the bound sets a build takes from its
/// one binder share the build's observer, the first scorer or picker adapter
/// made from any of them. A later build over the same loaded plugin (a spec
/// applied again) takes a binder of its own, so its adapters forward for it
/// while the earlier build stands and once it is gone.
#[derive(Debug)]
pub struct Binder {
    plugin: Arc<Plugin>,
    observer_taken: Arc<AtomicBool>,
}

impl Binder {
    /// Binds the plugin to one profile of this build: the stage set that
    /// shares one `decision` resource per decision.
    pub fn bind(&self) -> Arc<Bound> {
        Arc::new(Bound {
            plugin: Arc::clone(&self.plugin),
            opens: ThreadLocal::new(),
            observer_taken: Arc::clone(&self.observer_taken),
        })
    }
}

/// A plugin bound to one profile: the three stage adapters of a profile
/// share it, so one decision constructs one `decision` resource per phase
/// whichever stages the profile asks of the plugin.
pub struct Bound {
    plugin: Arc<Plugin>,
    opens: ThreadLocal<RefCell<Opens>>,
    /// The build's observer, shared with the build's other bound sets.
    observer_taken: Arc<AtomicBool>,
}

impl fmt::Debug for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bound")
            .field("plugin", &self.plugin.name)
            .finish_non_exhaustive()
    }
}

impl Bound {
    pub fn plugin(&self) -> &Arc<Plugin> {
        &self.plugin
    }

    fn state(&self, phase: Phase) -> RefMut<'_, Open> {
        RefMut::map(self.opens.get_or_default().borrow_mut(), |opens| {
            &mut opens.0[framework::slot(phase)]
        })
    }

    /// Opens the decision of the request's phase: the resource over the
    /// view, once per decision (the later stages of the profile find it
    /// open). A gone budget or an open breaker opens nothing: the stages
    /// fall back.
    pub(crate) fn open(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        pool: &PoolFacts<'_>,
    ) {
        let mut open = self.state(facts.phase);
        if open.is_open {
            return;
        }
        open.reset();
        open.is_open = true;
        open.count = u32::try_from(candidates.len()).unwrap_or(u32::MAX);
        let plugin = &self.plugin;
        let frame = plugin.frame;
        open.frame_fuel = frame.fuel_for(facts.budget_micros);
        open.frame_micros = frame.frame_micros(facts.budget_micros);
        open.now = facts.now_micros;
        open.draw = Draw::new(facts.seed);
        // The decision's budget is one a thread: its first open seeds it,
        // every open of the decision (each side, each plugin of every
        // runtime built over the same open decisions) draws on it.
        let mut budget = plugin.runtime.decisions.on_this_thread();
        let decision = budget.open(facts.budget_micros);
        open.decision = decision;
        if plugin.note_open(decision) {
            // The plugin already failed this decision (the other side of a
            // pair, another binding): its part in the decision is over, with
            // no call, no `init` and no second count.
            open.bypass = true;
            return;
        }
        let now = plugin.runtime.now_micros();
        if plugin.breaker.is_open(now) {
            bump(&plugin.tally.bypassed);
            open.bypass = true;
            return;
        }
        if open.frame_fuel == 0 {
            // The frame budget is gone on arrival: no frame is forwarded, no
            // instance is made or fed, the breaker counts a budget hit.
            bump(&plugin.tally.budget_gone);
            plugin.fail(now);
            open.bypass = true;
            return;
        }
        if budget.remaining_micros == 0 {
            // An earlier open of this decision spent the remainder: no frame
            // is forwarded, the stage falls back, the breaker is untouched.
            bump(&plugin.tally.budget_spent);
            open.bypass = true;
            return;
        }
        // A named decider's kept state comes to this thread for the
        // decision; the plugin's last close of it gives it back.
        plugin.lend();
        let slot = plugin.slot();
        let mut guard = slot.borrow_mut();
        if guard.is_none() {
            if plugin.emptied.get_or_default().get() {
                // An earlier open or call of this decision dropped the
                // instance: the plugin's part in the decision is over, and
                // the close rebuilds the slot under the full init frame.
                open.bypass = true;
                return;
            }
            // The apply's own instance, where it still waits for this
            // decider: no `init` runs and nothing is charged.
            *guard = plugin.adopt();
        }
        if guard.is_none() {
            // A cold slot: `init` runs under what the decision has left (a
            // slot still empty at the close is rebuilt under the full frame).
            // The budget is wall time here as in a call: the instantiation is
            // charged the larger of its fuel at the calibration and its
            // elapsed time, however it ends. Its own time has two ends like
            // a call's. Nothing watches an `init` (its bound stays on the
            // wall clock), so no tick reads it: its low end is its fuel, and
            // an armed thread reads its clock around it, which makes the two
            // one: a held `init` is no slow sample there.
            let fuel_per_micro = frame.fuel_per_micro.max(1);
            let left = budget.remaining_micros.saturating_mul(fuel_per_micro);
            let init_fuel = plugin.init_fuel();
            let watched = plugin.runtime.thread_clock;
            let start = if watched && plugin.armed() {
                thread_cpu_micros()
            } else {
                None
            };
            let started = Instant::now();
            let (made, used) = plugin.instantiate_metered(init_fuel.min(left));
            let elapsed = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
            let fuel_micros = used.div_ceil(fuel_per_micro);
            let charged = fuel_micros.max(elapsed);
            let own = match start.and_then(|start| Some(thread_cpu_micros()?.saturating_sub(start)))
            {
                Some(had) => OwnTime::read(fuel_micros, had, charged),
                None => OwnTime::unread(watched, fuel_micros, charged),
            };
            match made {
                Ok(instance) => {
                    bump(&plugin.tally.lazy_instantiations);
                    budget.remaining_micros = budget.remaining_micros.saturating_sub(charged);
                    plugin.note_init(decision, own);
                    *guard = Some(instance);
                    plugin.stand();
                }
                Err(LoadError::Trap { .. })
                    if left < init_fuel && charged >= budget.remaining_micros =>
                {
                    // The remainder clamped `init` below its own frame and
                    // the clamp ended it: the charge reached what was left
                    // (out of fuel uses the whole grant, and the deadline
                    // never fires before the remainder has passed). The
                    // decision's budget is spent (the whole remainder is
                    // drawn, so no later open of the decision runs on it
                    // again), the slot stays cold for the close to rebuild
                    // under the full frame, and the breaker, which judges
                    // `init` under its own frame, is untouched. The plugin's
                    // sample takes the time of this `init` as of any other:
                    // a charge that reached the remainder, so one long on the
                    // clock reads as long here too, unless a reading says
                    // its thread did not have that time.
                    let init_micros = left.div_ceil(fuel_per_micro).max(charged);
                    budget.remaining_micros = budget.remaining_micros.saturating_sub(init_micros);
                    plugin.note_init(decision, own);
                    bump(&plugin.tally.budget_spent);
                    plugin.emptied.get_or_default().set(true);
                    open.bypass = true;
                    return;
                }
                Err(_) => {
                    // `init` failed by itself: under its own frame, or short
                    // of a clamp (a charge below the remainder: it would
                    // have ended the same under the full frame). The failure
                    // is the breaker's, the charge the decision's and part
                    // of the plugin's sample.
                    budget.remaining_micros = budget.remaining_micros.saturating_sub(charged);
                    plugin.note_init(decision, own);
                    plugin.fail(now);
                    open.bypass = true;
                    return;
                }
            }
        }
        // What the membership feed told since this instance last heard comes
        // before the decision reads the instance, as told events: outside
        // the decision's sums. A change that faults it ends the plugin's
        // part in the decision, and the close rebuilds the slot.
        plugin.catch_up(&mut guard);
        if guard.is_none() {
            open.bypass = true;
            return;
        }
        if let Some(inst) = guard.as_mut() {
            match &inst.exports {
                Exports::Record(_) => inst.records.fill(facts, candidates, pool),
                Exports::Packed(_) => {
                    let filled =
                        inst.packed
                            .fill(facts, candidates, pool, &plugin.exposure, plugin.needs);
                    if filled == Err(TooLong) {
                        bump(&plugin.tally.too_long);
                        open.bypass = true;
                        return;
                    }
                }
            }
        }
        let count = open.count;
        let exposure = &plugin.exposure;
        let needs = plugin.needs;
        let frame = open.frame(&mut budget, frame.fuel_per_micro);
        let constructed = plugin.call(&mut guard, frame, |inst| {
            let result = match &inst.exports {
                Exports::Record(b) => {
                    let headers = marshal::headers(facts, exposure);
                    let request =
                        marshal::request(facts, exposure, needs, &inst.records.tried, &headers);
                    let pool = marshal::pool(pool, &inst.records.zones);
                    b.smg_policy_plugin().decision().call_constructor(
                        &mut inst.store,
                        request,
                        &inst.records.candidates,
                        pool,
                    )
                }
                Exports::Packed(b) => b.smg_policy_plugin_packed().decision().call_constructor(
                    &mut inst.store,
                    &inst.packed.request,
                    &inst.packed.candidates,
                    count,
                    &inst.packed.pool,
                ),
            };
            result.map(Ok)
        });
        match constructed {
            Call::Answered(resource) => {
                open.resource = Some(resource);
                open.generation = guard.as_ref().map_or(0, |i| i.generation);
            }
            Call::Declined | Call::Failed | Call::BudgetGone => open.bypass = true,
        }
    }

    /// Closes every open phase of this thread: the resources are dropped
    /// under their frame's fuel (a destructor is the guest's code too), the
    /// answers forgotten. The scope is closed first: every phase stops
    /// reading as open, and its place in the thread's decision is given
    /// back, before the first destructor runs. A destructor is called under
    /// the host's own code (the clock a build handed the runtime, an
    /// import's body): whatever such a call does, the thread's next decision
    /// opens as its own.
    pub(crate) fn close(&self) {
        /// The destructors' walk. If it does not return, the slot it marked
        /// for the close's rebuild is forgotten: the thread's next decision
        /// finds a cold slot, not one its own calls emptied.
        struct Walk<'a> {
            plugin: &'a Plugin,
            returned: bool,
        }
        impl Drop for Walk<'_> {
            fn drop(&mut self) {
                if !self.returned {
                    self.plugin.emptied.get_or_default().set(false);
                }
            }
        }
        let plugin = &self.plugin;
        let mut opens = self.opens.get_or_default().borrow_mut();
        let mut budget = plugin.runtime.decisions.on_this_thread();
        // One bit a phase that was open.
        let mut closing = 0u8;
        for (phase, open) in opens.0.iter_mut().enumerate() {
            if open.is_open {
                open.is_open = false;
                budget.close();
                closing |= 1 << phase;
            }
        }
        let mut walk = Walk {
            plugin: self.plugin.as_ref(),
            returned: false,
        };
        // The decision these opens are of: each open of it returned the one
        // number.
        let mut decision = 0;
        // Whether the plugin's last open of that decision closes here.
        let mut last = false;
        // The phases that were open, the lowest first.
        while closing != 0 {
            let open = &mut opens.0[closing.trailing_zeros() as usize];
            closing &= closing - 1;
            if let Some(resource) = open.resource.take() {
                let mut guard = plugin.slot().borrow_mut();
                // A resource whose store went with a fault (another binding's
                // close of this decision may have rebuilt the slot since) has
                // nothing to drop: a rebuilt instance would refuse the handle.
                let same_store = guard
                    .as_ref()
                    .is_some_and(|i| i.generation == open.generation);
                if same_store {
                    // The destructor is the guest's code, run for the host's
                    // cleanup: a small fuel frame of its own, outside the
                    // decision's remainder, so a spent budget drops no healthy
                    // instance; a bound on the clock of its own in milliseconds,
                    // so a destructor long on the clock cannot stretch the
                    // decision by a frame; not a latency sample. Running out
                    // of either drops the instance through the call's own
                    // fault path.
                    let mut remaining = u64::from(DESTRUCTOR_BOUND_MICROS);
                    let frame = Frame {
                        cap: plugin.frame.fuel_of(DESTRUCTOR_FRAME_MICROS),
                        remaining_micros: &mut remaining,
                        fuel_per_micro: plugin.frame.fuel_per_micro,
                        frame_micros: DESTRUCTOR_BOUND_MICROS,
                        now: Some(open.now),
                        draw: Some(&mut open.draw),
                        quiet: true,
                        decision: 0,
                        seed_micros: 0,
                    };
                    let _ = plugin.call(&mut guard, frame, |inst| {
                        resource.resource_drop(&mut inst.store).map(Ok)
                    });
                }
            }
            decision = open.decision;
            open.reset();
            last = plugin.note_close(decision, budget.seed_micros);
        }
        walk.returned = true;
        drop(budget);
        // A slot this decision emptied (a call interrupted or trapped, an
        // answer outside the contract, a destructor that ran out) is rebuilt
        // once here, under the full init frame, so the next decision finds it
        // warm: never a slot the decision did not touch (a cold thread's next
        // open instantiates lazily), never while the breaker has the plugin
        // bypassed. Another open of this decision still holding a resource of
        // the dropped store (one plugin, two binding profiles) finds the
        // generation changed.
        if plugin.emptied.get_or_default().replace(false) {
            let now = plugin.runtime.now_micros();
            if !plugin.breaker.is_open(now) {
                let mut guard = plugin.slot().borrow_mut();
                if guard.is_none() {
                    match plugin.instantiate(plugin.init_fuel()) {
                        Ok(instance) => {
                            bump(&plugin.tally.lazy_instantiations);
                            *guard = Some(instance);
                            // An instance came to stand here as at every
                            // other place that makes one: under no name the
                            // thread is a decider of its own from this close.
                            plugin.stand();
                        }
                        Err(error) => {
                            tracing::warn!(plugin = plugin.name, %error, "the rebuild of a dropped instance failed");
                            // One failure a decision: a rebuild that fails after
                            // the fault that emptied the slot was counted is not
                            // a second one.
                            if !plugin.failed_in(decision) {
                                plugin.report(now);
                            }
                        }
                    }
                }
            }
        }
        // The decision is over for the plugin: a named decider's kept state
        // goes back to its rest, rebuilt here where the decision emptied it.
        if last {
            plugin.rest();
        }
    }

    /// One guest call of the open decision on this thread's instance, under
    /// the decision's fuel frame: `call` gets the instance and the decision's
    /// live resource. The slot's guard comes back with the answer, for the
    /// caller's own reading of an invalid one; a decision whose resource is
    /// not live falls back with no call made.
    fn ask<T>(
        &self,
        open: &mut Open,
        call: impl FnOnce(&mut Instance, ResourceAny) -> wasmtime::Result<Result<T, wit::Decline>>,
    ) -> Result<(Call<T>, RefMut<'_, Option<Instance>>), Fallback> {
        let plugin = &self.plugin;
        let mut guard = plugin.slot().borrow_mut();
        let resource = open.live(guard.as_ref())?;
        let mut budget = plugin.runtime.decisions.on_this_thread();
        let frame = open.frame(&mut budget, plugin.frame.fuel_per_micro);
        let answer = plugin.call(&mut guard, frame, |inst| call(inst, resource));
        Ok((answer, guard))
    }

    fn ensure_mask(&self, open: &mut Open) -> Result<(), Fallback> {
        if open.mask_set {
            return Ok(());
        }
        if open.filter_declined {
            return Err(Fallback);
        }
        let count = open.count;
        let (answer, mut guard) = self.ask(open, |inst, resource| match &inst.exports {
            Exports::Record(b) => b
                .smg_policy_plugin()
                .decision()
                .call_filter(&mut inst.store, resource),
            Exports::Packed(b) => b
                .smg_policy_plugin_packed()
                .decision()
                .call_filter(&mut inst.store, resource),
        })?;
        open.filter_declined = matches!(answer, Call::Declined);
        let positions = open.answered(answer)?;
        // Unique and in range, else the answer is invalid.
        open.mask.clear();
        open.mask.resize((count as usize).div_ceil(64), 0);
        for &p in &positions {
            if p >= count || bit(&open.mask, p) {
                return Err(open.invalid(&self.plugin, &mut guard));
            }
            open.mask[(p / 64) as usize] |= 1 << (p % 64);
        }
        open.mask_set = true;
        Ok(())
    }
}

impl BoundPlugin for Bound {
    fn name(&self) -> &'static str {
        self.plugin.name()
    }

    fn failure(&self) -> Failure {
        self.plugin.failure()
    }

    fn needs(&self) -> Needs {
        self.plugin.needs()
    }

    fn observes(&self) -> EventSet {
        self.plugin.observes()
    }

    /// The first scorer or picker adapter made from any bound set of the
    /// build forwards the lifecycle; the others do not, so an event reaches
    /// the plugin once per build, whichever of its profiles bind the plugin.
    fn take_observer(&self) -> bool {
        !self.observer_taken.swap(true, Ordering::Relaxed)
    }

    /// The filter's answer for the phase's decision, computed once: `kept`
    /// keeps the positions the guest named.
    fn retain(&self, facts: &RequestFacts<'_>, kept: &mut Vec<u32>) -> Result<(), Fallback> {
        let mut open = self.state(facts.phase);
        self.ensure_mask(&mut open)?;
        kept.retain(|&i| bit(&open.mask, i));
        Ok(())
    }

    /// One row of the filter's answer.
    fn keep(&self, facts: &RequestFacts<'_>, position: u32) -> Result<bool, Fallback> {
        let mut open = self.state(facts.phase);
        self.ensure_mask(&mut open)?;
        Ok(bit(&open.mask, position))
    }

    /// The scorer's answer for the phase's decision, computed once: `out[i]`
    /// is the score of `kept[i]`.
    fn score(
        &self,
        facts: &RequestFacts<'_>,
        kept: &[u32],
        out: &mut [f32],
    ) -> Result<(), Fallback> {
        let mut open = self.state(facts.phase);
        if !open.scores_set {
            if open.score_declined {
                return Err(Fallback);
            }
            let count = open.count;
            let (answer, mut guard) =
                self.ask(&mut open, |inst, resource| match &inst.exports {
                    Exports::Record(b) => b
                        .smg_policy_plugin()
                        .decision()
                        .call_score(&mut inst.store, resource),
                    Exports::Packed(b) => b
                        .smg_policy_plugin_packed()
                        .decision()
                        .call_score(&mut inst.store, resource),
                })?;
            open.score_declined = matches!(answer, Call::Declined);
            let mut scores = open.answered(answer)?;
            // One finite value per candidate, clamped by the host.
            if scores.len() != count as usize || scores.iter().any(|s| !s.is_finite()) {
                return Err(open.invalid(&self.plugin, &mut guard));
            }
            for s in &mut scores {
                *s = s.clamp(0.0, 1.0);
            }
            // The guest's list is the buffer: no second allocation.
            open.scores = scores;
            open.scores_set = true;
        }
        for (o, &i) in out.iter_mut().zip(kept) {
            *o = open.scores.get(i as usize).copied().unwrap_or(0.0);
        }
        Ok(())
    }

    /// The picker's answer over the survivors: `scores[i]` belongs to
    /// `kept[i]`; the guest sees one score per candidate, `-1.0` on a row
    /// the filters dropped, and names a surviving position or declines (a
    /// `none` is the pick's decline).
    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        kept: &[u32],
        scores: &[f32],
    ) -> Result<u32, Fallback> {
        let mut open = self.state(facts.phase);
        let count = open.count;
        // A bypassed decision falls back before the buffer is built.
        if open.bypass {
            return Err(Fallback);
        }
        let mut scores_in = std::mem::take(&mut open.scores_in);
        scores_in.clear();
        scores_in.resize(count as usize, -1.0);
        for (&i, &s) in kept.iter().zip(scores) {
            if let Some(slot) = scores_in.get_mut(i as usize) {
                *slot = s;
            }
        }
        let asked = self.ask(&mut open, |inst, resource| {
            let picked = match &inst.exports {
                Exports::Record(b) => b.smg_policy_plugin().decision().call_pick(
                    &mut inst.store,
                    resource,
                    &scores_in,
                ),
                Exports::Packed(b) => b.smg_policy_plugin_packed().decision().call_pick(
                    &mut inst.store,
                    resource,
                    &scores_in,
                ),
            };
            picked.map(|answer| match answer {
                Ok(Some(position)) => Ok(position),
                Ok(None) => Err(wit::Decline::NotApplicable),
                Err(decline) => Err(decline),
            })
        });
        let (answer, mut guard) = match asked {
            Ok(asked) => asked,
            Err(fallback) => {
                // The buffer stays with the decision when no call is made.
                open.scores_in = scores_in;
                return Err(fallback);
            }
        };
        // Within the surviving set, else the answer is invalid.
        let within = match &answer {
            Call::Answered(p) => scores_in.get(*p as usize).is_some_and(|s| *s >= 0.0),
            _ => true,
        };
        open.scores_in = scores_in;
        let picked = open.answered(answer)?;
        if !within {
            return Err(open.invalid(&self.plugin, &mut guard));
        }
        Ok(picked)
    }

    fn observe_issued(&self, lease: &Lease, decision: &Decision) {
        let mut targets: [wit::Target; MAX_TARGETS] = std::array::from_fn(|_| wit::Target {
            id: 0,
            dp_rank: 0,
            pair: None,
        });
        let n = marshal::targets(decision, &mut targets);
        self.plugin.observe(wit::LifecycleEvent::Issued((
            marshal::lease(lease, &self.plugin.exposure),
            &targets[..n],
        )));
    }

    fn observe_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        self.plugin
            .observe(wit::LifecycleEvent::Dispatched(wit::Dispatched {
                lease: marshal::lease(lease, &self.plugin.exposure),
                target: marshal::target(target),
                side: marshal::side(side),
                since_decision_micros,
            }));
    }

    fn observe_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.plugin
            .observe(wit::LifecycleEvent::Completed(marshal::completed(
                lease,
                outcome,
                &self.plugin.exposure,
            )));
    }

    fn observe_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.plugin.observe(wit::LifecycleEvent::Released((
            marshal::lease(lease, &self.plugin.exposure),
            marshal::release_reason(reason),
        )));
    }

    fn observe_membership(&self, change: &Membership<'_>) {
        self.plugin.hand(change);
    }

    fn observe_signal(&self, signal: &Signal) {
        let now = self.plugin.runtime.now_micros();
        self.plugin
            .observe(wit::LifecycleEvent::Signal(marshal::signal(signal, now)));
    }
}

/// This open marks its phase open and counts the decision before its
/// instance's steps, so it runs under the guard that undoes an open that
/// stops half-way.
impl DecisionScope for Bound {
    fn open(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], pool: &PoolFacts<'_>) {
        let mut half_open = HalfOpen::new(self);
        Bound::open(self, facts, candidates, pool);
        half_open.returned();
    }

    fn close(&self) {
        Bound::close(self);
    }
}

fn bit(mask: &[u64], position: u32) -> bool {
    mask.get((position / 64) as usize)
        .is_some_and(|word| word & (1 << (position % 64)) != 0)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use wasmtime::{Trap, UpdateDeadline};

    use super::{
        bit, deadline_ticks, first_deadline, import_version, judge, thread_cpu_micros, tick, End,
        Latency, ThreadRing, Verdict, Watch, EPOCH_TICK_MICROS,
    };

    /// Whether a tick ended the call; a tick that lets it run asks for the
    /// next one.
    fn ended(tick: wasmtime::Result<UpdateDeadline>) -> bool {
        match tick {
            Ok(UpdateDeadline::Continue(1)) => false,
            Ok(_) => panic!("a tick that lets a call run asks for the next tick"),
            Err(error) => {
                assert_eq!(error.downcast_ref::<Trap>(), Some(&Trap::Interrupt));
                true
            }
        }
    }

    /// Keeps the calling thread on the CPU for `micros` of its own clock.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn burn(micros: u64) {
        let from = thread_cpu_micros().expect("a thread clock");
        while thread_cpu_micros().expect("a thread clock") < from + micros {
            std::hint::spin_loop();
        }
    }

    /// A tick ends a call on the CPU its thread had since a reading, never
    /// on the wall clock: a thread that slept past the bound (the host held
    /// it) runs on, and the same thread once it has burned the bound is
    /// ended. With no reading of the start the first tick takes one and ends
    /// nothing; with one, no stand-in is taken.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn a_tick_ends_a_call_on_the_cpu_its_thread_had_since_a_reading() {
        const BOUND: u64 = 2_000;
        let held = std::time::Duration::from_micros(3 * BOUND);

        let mut unread = Watch {
            bound_micros: BOUND,
            watched: true,
            start: None,
            first_tick: None,
        };
        assert!(!ended(tick(&mut unread)), "a first tick ends nothing");
        assert!(unread.first_tick.is_some(), "it took the reading");
        std::thread::sleep(held);
        assert!(
            !ended(tick(&mut unread)),
            "three bounds on the wall clock, off the CPU: not ended"
        );
        burn(BOUND);
        assert!(
            ended(tick(&mut unread)),
            "the bound had on the CPU since the first tick"
        );

        let mut read = Watch {
            bound_micros: BOUND,
            watched: true,
            start: thread_cpu_micros(),
            first_tick: None,
        };
        std::thread::sleep(held);
        assert!(
            !ended(tick(&mut read)),
            "held since its start: short of the bound on the CPU"
        );
        assert!(
            read.first_tick.is_none(),
            "a start reading needs no stand-in"
        );
        burn(BOUND);
        assert!(
            ended(tick(&mut read)),
            "the bound had on the CPU since the start"
        );
    }

    /// A call nothing watches ends at its deadline on the wall clock: an
    /// `init`, and every call on a host with no thread clock, which takes
    /// the same arm.
    #[test]
    fn a_tick_ends_a_call_nothing_watches() {
        assert!(ended(tick(&mut Watch::default())), "an init's watch");
        let mut unwatched = Watch {
            bound_micros: u64::MAX,
            watched: false,
            start: None,
            first_tick: None,
        };
        assert!(ended(tick(&mut unwatched)), "whatever its bound");
    }

    /// A watched call with no reading of its start meets its next tick,
    /// which takes one. A call whose start was read meets the tick its
    /// deadline on the wall clock falls on and none before it, like a call
    /// nothing watches: a reading never ends a call the wall clock would let
    /// run.
    #[test]
    fn a_start_reading_meets_no_tick_before_the_wall_clocks_deadline() {
        assert_eq!(
            (
                deadline_ticks(1),
                deadline_ticks(1_000),
                deadline_ticks(1_001)
            ),
            (2, 2, 3),
            "the bound in whole ticks and the tick in flight"
        );
        for bound in [1, 20, 1_000, 1_001, 10_000] {
            assert_eq!(first_deadline(true, false, bound), 1, "{bound}: unread");
            assert_eq!(
                first_deadline(true, true, bound),
                deadline_ticks(bound),
                "{bound}: its start read"
            );
            assert_eq!(
                first_deadline(false, false, bound),
                deadline_ticks(bound),
                "{bound}: nothing watched"
            );
        }
    }

    /// The judgement of a call at its end is a function of its readings and
    /// of how it ended: the wall time, the reading at its start or none, the
    /// reading at its end, its fuel and the bound. Every branch, each at its
    /// boundary, and the calls that read the clock again: every call whose
    /// start was read, but one that is not late and whose fuel covers the
    /// time it took. With no reading a call is sampled at its charge and
    /// spends its fuel (its charge on a host with no thread clock); a
    /// reading gives it one time, late or not.
    #[test]
    fn a_late_call_is_judged_once_on_its_readings() {
        // A bound that is no whole number of ticks, and the wall time that
        // makes its call late: the bound in whole ticks.
        const BOUND: u64 = 1_500;
        const LATE: u64 = 2 * EPOCH_TICK_MICROS;
        const START: u64 = 100;
        let watch = |start: Option<u64>| Watch {
            bound_micros: BOUND,
            watched: true,
            start,
            first_tick: None,
        };
        let read = watch(Some(START));
        let unread = watch(None);
        let ticked = Watch {
            first_tick: Some(START),
            ..unread
        };
        let unwatched = Watch {
            watched: false,
            ..read
        };
        // The case; the watch; the CPU clock at the end; the wall time; the
        // fuel in microseconds; how the call ended; the own time (its
        // sample, its spend), the verdict and whether the clock was read
        // again.
        let rows = [
            (
                "past its bound, a microsecond short of late, read: no stall, and no time but its fuel",
                read,
                START,
                LATE - 1,
                7,
                End::Answered,
                (7, 7, Verdict::Own, 1),
            ),
            (
                "not late, read, the CPU at the wall time: its charge",
                read,
                START + 900,
                900,
                7,
                End::Answered,
                (900, 900, Verdict::Own, 1),
            ),
            (
                "not late, read, the CPU short of the wall time: a hold is no time of the call",
                read,
                START + 40,
                900,
                7,
                End::Answered,
                (40, 40, Verdict::Own, 1),
            ),
            (
                "not late, read, the fuel above the CPU: its fuel",
                read,
                START + 40,
                900,
                600,
                End::Answered,
                (600, 600, Verdict::Own, 1),
            ),
            (
                "not late, read, the CPU above the wall time: never more than the charge",
                read,
                START + 5 * BOUND,
                900,
                7,
                End::Failed,
                (900, 900, Verdict::Own, 1),
            ),
            (
                "not late, read, the fuel at the wall time: its fuel, and the clock is not read again",
                read,
                START + 40,
                600,
                600,
                End::Answered,
                (600, 600, Verdict::Own, 0),
            ),
            (
                "not late, read, the fuel above the wall time: its fuel, and no reading",
                read,
                START + 40,
                10,
                900,
                End::Failed,
                (900, 900, Verdict::Own, 0),
            ),
            (
                "not late, read, the wall time a microsecond past the fuel: read again",
                read,
                START + 40,
                601,
                600,
                End::Answered,
                (600, 600, Verdict::Own, 1),
            ),
            (
                "late, read, the fuel above the wall time: read again, for a reading judges lateness",
                read,
                START + 40,
                LATE,
                LATE + 5,
                End::Answered,
                (LATE + 5, LATE + 5, Verdict::Held, 1),
            ),
            (
                "late at its bound in whole ticks, the CPU one short of the bound: held",
                read,
                START + BOUND - 1,
                LATE,
                7,
                End::Answered,
                (BOUND - 1, BOUND - 1, Verdict::Held, 1),
            ),
            (
                "a bound of whole ticks: late at the bound itself",
                Watch {
                    bound_micros: 10 * EPOCH_TICK_MICROS,
                    ..unread
                },
                START,
                10 * EPOCH_TICK_MICROS,
                7,
                End::Answered,
                (10 * EPOCH_TICK_MICROS, 7, Verdict::Unread, 0),
            ),
            (
                "a bound under a tick, twice had on the wall clock: not late",
                Watch {
                    bound_micros: 20,
                    ..unread
                },
                START,
                45,
                7,
                End::Answered,
                (45, 7, Verdict::Own, 0),
            ),
            (
                "late, the CPU at the bound: the plugin's, at its CPU",
                read,
                START + BOUND,
                LATE,
                7,
                End::Answered,
                (BOUND, BOUND, Verdict::Own, 1),
            ),
            (
                "held, the fuel above the CPU: its own time is its fuel",
                read,
                START + 40,
                9 * BOUND,
                600,
                End::Answered,
                (600, 600, Verdict::Held, 1),
            ),
            (
                "held on the CPU, failed by itself: the plugin's, at its CPU",
                read,
                START + BOUND - 1,
                LATE,
                7,
                End::Failed,
                (BOUND - 1, BOUND - 1, Verdict::Own, 1),
            ),
            (
                "held on the CPU, ended on the clock: the plugin's",
                read,
                START + BOUND - 1,
                LATE,
                7,
                End::OnTheClock,
                (BOUND - 1, BOUND - 1, Verdict::Own, 1),
            ),
            (
                "the CPU above the wall time: never more than the charge",
                read,
                START + 5 * BOUND,
                LATE,
                7,
                End::Answered,
                (LATE, LATE, Verdict::Own, 1),
            ),
            (
                "late, read, ended on the clock: the plugin's, never the hold",
                read,
                START + BOUND + 300,
                9 * BOUND,
                7,
                End::OnTheClock,
                (BOUND + 300, BOUND + 300, Verdict::Own, 1),
            ),
            (
                "late, no reading of its start, not ended: unread, sampled at its charge, spent at its fuel",
                unread,
                START,
                9 * BOUND,
                7,
                End::Answered,
                (9 * BOUND, 7, Verdict::Unread, 0),
            ),
            (
                "late, no reading of its start, failed by itself: unread all the same",
                unread,
                START,
                9 * BOUND,
                7,
                End::Failed,
                (9 * BOUND, 7, Verdict::Unread, 0),
            ),
            (
                "late, no reading of its start, ended on the clock: the plugin's",
                unread,
                START,
                9 * BOUND,
                7,
                End::OnTheClock,
                (9 * BOUND, 7, Verdict::Own, 0),
            ),
            (
                "a first tick's reading is no reading of the start",
                ticked,
                START + 40,
                9 * BOUND,
                7,
                End::Answered,
                (9 * BOUND, 7, Verdict::Unread, 0),
            ),
            (
                "nothing watched: the wall clock's call, its charge twice, and no reading",
                unwatched,
                START,
                9 * BOUND,
                7,
                End::Answered,
                (9 * BOUND, 9 * BOUND, Verdict::Own, 0),
            ),
            (
                "not late, no reading, the fuel above the wall time: the charge is the fuel",
                unread,
                START,
                10,
                900,
                End::Answered,
                (900, 900, Verdict::Own, 0),
            ),
        ];
        for (case, watch, cpu_at_end, elapsed, fuel, end, wanted) in rows {
            let reads = Cell::new(0);
            let (own, verdict) = judge(&watch, elapsed, fuel, end, || {
                reads.set(reads.get() + 1);
                Some(cpu_at_end)
            });
            assert_eq!(
                (own.sample, own.spend, verdict, reads.get()),
                wanted,
                "{case}"
            );
        }
    }

    /// The engine the host asks for keeps no state for concurrent calls and
    /// admits no component that needs it, in every build. The runtime's
    /// switch defaults to on wherever its async component support is
    /// compiled in, which any crate of a build can ask for: there this case
    /// is red without the switch's line. A build without that support passes
    /// with or without the line.
    #[test]
    fn the_runtime_keeps_no_state_for_concurrent_calls() {
        // An engine as the runtime builds its own: a config, then the
        // host's settings.
        let mut config = wasmtime::Config::new();
        super::configure(&mut config);
        let engine = wasmtime::Engine::new(&config).expect("an engine");
        // The least component that needs the support: one export lifted the
        // async way.
        let needs_it = wat::parse_str(
            r#"(component
                (core module $m
                    (func (export "run") (result i32) i32.const 0)
                    (func (export "cb") (param i32 i32 i32) (result i32) i32.const 0))
                (core instance $i (instantiate $m))
                (type $run (func async))
                (func (export "run") (type $run)
                    (canon lift (core func $i "run") async (callback (core func $i "cb")))))"#,
        )
        .expect("the component assembles");
        assert!(
            wasmtime::component::Component::new(&engine, needs_it).is_err(),
            "a component with an async export compiled"
        );
        assert!(
            !engine.get_concurrency_support(),
            "the runtime keeps state for concurrent calls"
        );
    }

    #[test]
    fn an_interface_name_yields_its_version_and_nothing_else_matches() {
        assert_eq!(
            import_version("smg:policy/plugin@1.0.0", "smg:policy/plugin"),
            Some("1.0.0")
        );
        assert_eq!(
            import_version("smg:policy/plugin-packed@1.0.0", "smg:policy/plugin"),
            None,
            "the packed world is not the record world"
        );
        assert_eq!(
            import_version("wasi:io/poll@0.2.0", "smg:policy/host"),
            None
        );
        assert_eq!(import_version("smg:policy/host", "smg:policy/host"), None);
    }

    #[test]
    fn a_mask_reads_its_bits_and_nothing_past_its_end() {
        let mask = [1u64 << 3, 1];
        assert!(bit(&mask, 3));
        assert!(bit(&mask, 64));
        assert!(!bit(&mask, 4));
        assert!(!bit(&mask, 200));
    }

    /// A sample is uncovered where it is past a mark of the ring that its
    /// low end is short of: each mark at its edge, in the ring's own unit.
    #[test]
    fn a_sample_past_a_mark_its_low_end_is_short_of_is_uncovered() {
        // A sample, its low end and the frame, and whether no reading
        // covers the sample.
        let table = [
            // At half the frame the high end is past no mark.
            (500, 0, 1_000, false),
            // Past half the frame by the high end alone.
            (501, 0, 1_000, true),
            (501, 500, 1_000, true),
            // Past half the frame by both ends, and short of the frame by
            // both.
            (501, 501, 1_000, false),
            (1_000, 501, 1_000, false),
            // Past the frame by the high end alone, whatever half says.
            (1_001, 0, 1_000, true),
            (1_001, 501, 1_000, true),
            (1_001, 1_000, 1_000, true),
            // Past the frame by both ends.
            (1_001, 1_001, 1_000, false),
            (9_000, 1_001, 1_000, false),
            // The share is the ring's: a frame that is no thousand.
            (700, 5, 400, true),
            (200, 5, 400, false),
            (201, 5, 400, true),
            (2, 1, 3, true),
            // No frame, no sample.
            (700, 5, 0, false),
        ];
        for (micros, low_micros, frame_micros, uncovered) in table {
            assert_eq!(
                ThreadRing::uncovered(micros, low_micros, frame_micros),
                uncovered,
                "{micros} over {low_micros} under a frame of {frame_micros}"
            );
        }
    }

    /// An armed ring stays armed for its own length in samples and no
    /// longer: through the sample of the decision that armed it and the 127
    /// after it, so the next sample a late call can leave unread overwrites
    /// the first one's. A decision that records nothing does not count.
    #[test]
    fn an_armed_ring_stays_armed_for_its_length_in_samples() {
        let mut ring = ThreadRing::new();
        assert!(!ring.armed(), "a new ring is not armed");
        ring.arm();
        for sample in 0..Latency::SAMPLES - 1 {
            assert!(ring.armed(), "armed before sample {sample}");
            assert!(!ring.record(10, 0), "no frame: not a sample");
            assert!(
                ring.armed(),
                "a decision that records nothing does not count"
            );
            ring.record(10, 100);
        }
        assert!(
            ring.armed(),
            "the call 127 samples after the first is still read"
        );
        ring.record(10, 100);
        assert!(!ring.armed(), "the arm fell with the ring's length");
    }

    /// One slow sample a window is never its p99. A ring armed by each late
    /// call no reading covers takes such a sample once a length at most, so
    /// those samples alone never read slow.
    #[test]
    fn one_unread_sample_a_ring_length_never_reads_slow() {
        let mut ring = ThreadRing::new();
        let mut tripped = false;
        for round in 0..4 {
            assert!(!ring.armed(), "round {round}: the last arm has fallen");
            // The unread call's decision: its sample is the time the call
            // took, five hundred times its budget. It arms the ring.
            ring.arm();
            tripped |= ring.record(50_000, 100);
            // Armed, a late call is read: its sample is the plugin's own.
            for _ in 1..Latency::SAMPLES {
                tripped |= ring.record(10, 100);
            }
        }
        assert!(!tripped, "one slow sample in every window of the ring");
    }
}
