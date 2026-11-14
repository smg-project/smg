//! A loaded plugin and how the host runs it: one compiled component, one
//! kept instance per deciding thread, one `decision` resource per decision
//! per profile that binds the plugin, every call under fuel, and the rules
//! for what a trap, an exhausted budget, an invalid answer or a decline does.

use std::{
    cell::{Cell, RefCell, RefMut},
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use thread_local::ThreadLocal;
use wasmtime::{
    component::{Component, HasSelf, Linker, ResourceAny},
    Config, Engine, OptLevel, Store, StoreLimitsBuilder, Trap, UpdateDeadline,
};

use super::{
    bindings::{packed, record, wit, HostState, LogGate, Watch},
    breaker::{Breaker, Latency},
    config::{Abi, Exposure, Failure, FrameBudget, PluginSpec},
    counters::Counters,
    intern,
    view::{self as marshal, Packed, Records, TooLong},
};
use crate::{
    decision::{Decision, Target, MAX_TARGETS},
    framework::{self, DecisionScope},
    lease::{Lease, Membership, Outcome, ReleaseReason, Side, Signal},
    spec::Needs,
    view::{Candidate, Phase, PoolFacts, RequestFacts},
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
/// ticks catch what the calibration underprices and the time a host call
/// takes.
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

/// What the end of a call says of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// The call's time is the plugin's own: it was not late, or its bound
    /// ended it, or a reading of its start says it had its bound.
    Own,
    /// Late, and a reading of its start says its thread had less than the
    /// bound on the CPU: the host held it.
    Held,
    /// Late, not ended, and no reading of its start: the host held it or it
    /// overran a bound its fuel did not price, and nothing says which.
    Unread,
}

/// A call's own time, in microseconds, and its verdict. A late call is
/// judged once, at its end, on a reading: with one of its start, `cpu` reads
/// the thread's clock again, the call is the host's when its thread had less
/// than the bound and the plugin's otherwise, and its own time is what it
/// could spend, its fuel or that CPU, never the time the host held it. With
/// none the call is what it was before the bound counted the CPU: charged
/// the time it took, and unread unless its bound ended it. Any other call's
/// own time is its charge (`fuel_micros` or `elapsed_micros`, whichever is
/// more).
///
/// Late is the wall clock at or past the bound in whole ticks, the earliest
/// a deadline there could have ended the call. Not a tick more: a call the
/// host held for between the two would be charged its hold and counted
/// nowhere, and two such samples in a ring read slow. Not the bound itself:
/// under a frame shorter than a tick a call inside its fuel runs past its
/// bound on the wall clock and is ended by nothing.
fn judge(
    watch: &Watch,
    elapsed_micros: u64,
    fuel_micros: u64,
    ended_on_the_clock: bool,
    cpu: impl FnOnce() -> Option<u64>,
) -> (u64, Verdict) {
    let charged = fuel_micros.max(elapsed_micros);
    let late = watch.watched
        && elapsed_micros >= whole_ticks(watch.bound_micros).saturating_mul(EPOCH_TICK_MICROS);
    if !late {
        return (charged, Verdict::Own);
    }
    let Some(had) = watch
        .start
        .and_then(|start| Some(cpu()?.saturating_sub(start)))
    else {
        let verdict = if ended_on_the_clock {
            Verdict::Own
        } else {
            Verdict::Unread
        };
        return (charged, verdict);
    };
    let verdict = if had < watch.bound_micros {
        Verdict::Held
    } else {
        Verdict::Own
    };
    (fuel_micros.max(had).min(charged), verdict)
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
/// destructor the calibration underprices, or one long in its host calls,
/// cannot stretch a decision by a frame. Like every call's it counts the
/// time the thread had the CPU: a destructor the host held is not ended.
const DESTRUCTOR_BOUND_MILLIS: u32 = 2;
const DESTRUCTOR_BOUND_MICROS: u32 = DESTRUCTOR_BOUND_MILLIS * 1_000;

/// The budget of the decision open on a thread, in microseconds of the
/// request's: seeded by its first open, drawn down by every call of every
/// plugin and side the decision opens, forgotten by its last close.
#[derive(Default)]
struct DecisionBudget {
    remaining_micros: u64,
    /// The request's budget as seeded: a plugin's own spend reads against it.
    seed_micros: u64,
    /// The decision's number on this thread: a plugin's per-decision spend is
    /// keyed by it.
    stamp: u64,
    opens: u32,
}

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
/// with the four host imports, and the host's clock.
pub struct Runtime {
    engine: Engine,
    linker: Linker<HostState>,
    clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    /// The epoch ticker's state, shared with its thread.
    ticking: Arc<Ticking>,
    /// The ticker thread, to wake it when a call starts.
    ticker: std::thread::Thread,
    /// The budget of the decision open on each thread.
    budgets: ThreadLocal<RefCell<DecisionBudget>>,
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
    /// A runtime whose live clock counts microseconds from its creation. A
    /// host whose view clock has another origin passes it to
    /// [`Runtime::with_clock`], so `now-micros` inside `observe` and a
    /// veto's remaining time read on the same clock as the decisions.
    pub fn new() -> Result<Self, LoadError> {
        let origin = Instant::now();
        Self::with_clock(move || u64::try_from(origin.elapsed().as_micros()).unwrap_or(u64::MAX))
    }

    /// A runtime on the host's clock: `clock` returns microseconds since the
    /// host's origin, the clock of [`RequestFacts::now_micros`].
    pub fn with_clock(clock: impl Fn() -> u64 + Send + Sync + 'static) -> Result<Self, LoadError> {
        let mut config = Config::new();
        config.wasm_component_model(true);
        config.consume_fuel(true);
        config.epoch_interruption(true);
        config.cranelift_opt_level(OptLevel::Speed);
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
            budgets: ThreadLocal::new(),
            thread_clock: thread_cpu_micros().is_some(),
        })
    }

    /// The host's clock now.
    pub fn now_micros(&self) -> u64 {
        (self.clock)()
    }
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
    #[error(
        "the plugins ask {asked} MiB of memory per deciding thread; the budget is {budget} MiB"
    )]
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
    /// judges a plugin against its own frames.
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
    /// Late calls the host held, by a reading: the wall clock put the call
    /// at or past its bound in whole ticks, and the thread's CPU clock, read
    /// at the call's start and at its end, says the thread had less than the
    /// bound. The host's signal (a starved or throttled thread), never a
    /// count of the plugin's failures: the instance stays and the breaker
    /// does not move. A thread reads its clock at a call's start only while
    /// it is armed (`unread`), so the first hold of a quiet thread is not
    /// here.
    pub stalls: u64,
    /// Late calls no reading covers: the wall clock put the call at or past
    /// its bound in whole ticks, the bound did not end it, and its thread
    /// had not read its clock at the call's start. The host held it, or it
    /// overran a bound its fuel did not price; nothing says which, so the
    /// call is charged the time it took, and it arms its thread: through the
    /// next 128 decisions the plugin samples there, each of its calls reads
    /// the clock at its start. That window is short on a busy thread (128 ms
    /// at a thousand decisions a second), so a host that holds a thread less
    /// often leaves every hold here and none in `stalls`: read the two
    /// together.
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
    stalls: AtomicU64,
    unread: AtomicU64,
}

fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
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
    failure: Failure,
    log: super::config::LogLimit,
    counters: Counters,
    breaker: Breaker,
    tally: Tally,
    /// One kept instance per thread, by `thread_local`'s thread id. That crate
    /// recycles the id of a thread that exited, so a later thread may adopt
    /// the instance a dead one left: an instance is `Send`, one live thread
    /// holds an id at a time, and the adopter keeps it warm.
    instances: ThreadLocal<RefCell<Option<Instance>>>,
    /// Instances made so far, numbering each.
    generations: AtomicU64,
    /// Per thread: this decision emptied the slot (a dropped instance), so
    /// its later opens make no instance and its close rebuilds it.
    emptied: ThreadLocal<Cell<bool>>,
    /// Per thread: the decision's number and this plugin's own spend in it
    /// (its constructor and stage calls, in microseconds), so a call the
    /// budget ends is charged to the plugin only once that spend reached its
    /// own frame.
    spent: ThreadLocal<Cell<(u64, u64)>>,
    /// Per thread: the latency ring, one sample per decision (the plugin's
    /// own charged spend in it and a lazy `init` the decision paid, over the
    /// decision's budget), kept across rebuilds so a plugin dropped every
    /// decision still completes a window.
    latency: ThreadLocal<RefCell<Latency>>,
    /// Per thread: the decision's number, this plugin's opens of it still to
    /// close (each side of a pair, each binding), so the decision's sample is
    /// taken at the plugin's last close of it, the charge of a lazy `init`
    /// the decision paid (part of its sample, not of its own spend), and
    /// whether the plugin already failed the decision: the decision is the
    /// breaker's unit, so it counts one failure however many opens fail.
    decision_opens: ThreadLocal<Cell<(u64, u32, u64, bool)>>,
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
            failure: spec.failure,
            log: spec.log,
            counters: Counters::new(spec.metrics),
            breaker: Breaker::new(spec.breaker),
            tally: Tally::default(),
            instances: ThreadLocal::new(),
            generations: AtomicU64::new(0),
            emptied: ThreadLocal::new(),
            spent: ThreadLocal::new(),
            latency: ThreadLocal::new(),
            decision_opens: ThreadLocal::new(),
        });
        // The apply's own instance: `init` runs here, and its failure is the
        // apply's.
        let instance = plugin.instantiate(plugin.init_fuel())?;
        *plugin.slot().borrow_mut() = Some(instance);
        Ok(plugin)
    }

    /// The total of `memory_mib` over `specs` against the budget, per
    /// deciding thread: what the spec's `plugins.memory_budget_mib` checks
    /// at validation.
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

    /// Creates this thread's kept instance when it has none: the apply calls
    /// it on every deciding thread, so no decision pays the instantiation.
    pub fn warm(&self) -> Result<(), LoadError> {
        let slot = self.slot();
        if slot.borrow().is_none() {
            let instance = self.instantiate(self.init_fuel())?;
            *slot.borrow_mut() = Some(instance);
        }
        Ok(())
    }

    /// Whether this thread holds a kept instance.
    pub fn is_warm(&self) -> bool {
        self.slot().borrow().is_some()
    }

    /// Forgets this thread's kept instance, so the next decision or
    /// `observe` on it instantiates again: what a host without a kept
    /// instance would pay per decision, for the measurements.
    #[doc(hidden)]
    pub fn forget_instance(&self) {
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

    /// What the host must feed for the plugin's stages, as declared. The
    /// engine's summary folds the needs of a profile's scorers, so a plugin
    /// bound as a scorer carries these into it; a plugin bound as a filter
    /// or a picker alone does not (those stage kinds declare no needs), and
    /// a host that must feed it reads them here.
    pub fn needs(&self) -> Needs {
        self.needs
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
        // whose ring is armed, the one reading a late call is judged on;
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
        let ended_on_the_clock = answer
            .as_ref()
            .err()
            .and_then(|error| error.downcast_ref::<Trap>())
            == Some(&Trap::Interrupt);
        // The plugin's own time in the call is what it could spend; a late
        // call is judged once, here. One a reading says the host held is the
        // host's signal, not the plugin's failure. One no reading covers is
        // counted by its own name and arms the thread: the next late call
        // there has a reading.
        let (own, verdict) = judge(
            &watch,
            u64::from(elapsed),
            fuel_micros,
            ended_on_the_clock,
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
        // The plugin's own spend in the decision before this call.
        let own_before = if frame.decision == 0 {
            0
        } else {
            self.spend(frame.decision, own).saturating_sub(own)
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
                // Short of both, the remainder other plugins drew clamped the
                // call, and it is the decision's budget spent, whichever scope
                // ran first: the interrupted instance goes (its memory is
                // mid-call), the breaker is untouched, and chronic slowness
                // opens it through the decision's latency sample instead.
                // Judged on the grant, not the clock, so a stalled call is not
                // charged for the stall.
                let own_with_grant = own_before.saturating_add(grant.div_ceil(fuel_per_micro));
                let own_bound = grant == frame.cap || own_with_grant >= frame.seed_micros;
                if frame_ran_out && frame.decision != 0 && !own_bound {
                    tracing::debug!(
                        plugin = self.name,
                        "a call ran out under the decision's remainder"
                    );
                    bump(&self.tally.budget_spent);
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
        let (stamp, count, init, _) = opens.get();
        if count != 0 {
            opens.set((stamp, count, init, true));
        }
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
        let (stamp, _, _, failed) = self.decision_opens.get_or_default().get();
        stamp == decision && failed
    }

    /// One of this plugin's opens of decision `decision` on this thread;
    /// says whether the plugin already failed that decision (the one read
    /// of the cell an open makes).
    fn note_open(&self, decision: u64) -> bool {
        let opens = self.decision_opens.get_or_default();
        let (stamp, count, init, failed) = opens.get();
        let (count, init, failed) = if stamp == decision {
            (count, init, failed)
        } else {
            (0, 0, false)
        };
        opens.set((decision, count.saturating_add(1), init, failed));
        failed
    }

    /// The charge of a lazy `init` decision `decision` paid on this thread:
    /// judged under the init frame, never the plugin's own spend, but part of
    /// what the decision cost it.
    fn note_init(&self, decision: u64, micros: u64) {
        let opens = self.decision_opens.get_or_default();
        let (stamp, count, init, failed) = opens.get();
        if stamp == decision {
            opens.set((stamp, count, init.saturating_add(micros), failed));
        }
    }

    /// One of this plugin's closes of decision `decision`: the last one takes
    /// the decision's latency sample, the plugin's own charged spend in the
    /// decision (its constructor and stage calls, whatever ended them) plus
    /// the charge of a lazy `init` the decision paid, over the decision's
    /// budget. A decision that charged the plugin nothing records nothing.
    fn note_close(&self, decision: u64, seed_micros: u64) {
        let opens = self.decision_opens.get_or_default();
        let (stamp, count, init, failed) = opens.get();
        if stamp != decision {
            return;
        }
        let count = count.saturating_sub(1);
        opens.set((decision, count, init, failed));
        if count != 0 {
            return;
        }
        let (spent_stamp, spent) = self.spent.get_or_default().get();
        let spent = if spent_stamp == decision { spent } else { 0 };
        let charged = spent.saturating_add(init);
        if charged > 0 {
            self.sample(charged, seed_micros);
        }
    }

    /// One latency sample into the thread's ring: a decision's charge as a
    /// share of its budget; a window that reads slow opens the breaker.
    fn sample(&self, charged: u64, seed_micros: u64) {
        let sample = u32::try_from(charged).unwrap_or(u32::MAX);
        let seed = u32::try_from(seed_micros).unwrap_or(u32::MAX);
        let slow = self
            .latency
            .get_or(|| RefCell::new(Latency::new()))
            .borrow_mut()
            .record(sample, seed);
        if slow && self.breaker.slow(self.runtime.now_micros()) {
            self.opened("latency");
        }
    }

    /// Whether this thread reads its CPU clock at the start of the plugin's
    /// calls: its ring is armed.
    fn armed(&self) -> bool {
        self.latency.get().is_some_and(|ring| ring.borrow().armed())
    }

    /// Arms this thread's ring: a late call here had no reading of its
    /// start, and its decision's sample is the time the call took.
    fn arm(&self) {
        self.latency
            .get_or(|| RefCell::new(Latency::new()))
            .borrow_mut()
            .arm();
    }

    /// Drops this thread's instance: its store is gone with it, and this
    /// decision's close rebuilds the slot.
    fn drop_instance(&self, instance: &mut Option<Instance>) {
        *instance = None;
        self.emptied.get_or_default().set(true);
    }

    /// Adds `micros` to this plugin's own spend in decision `decision` on
    /// this thread and returns the spend so far.
    fn spend(&self, decision: u64, micros: u64) -> u64 {
        let spent = self.spent.get_or_default();
        let (stamp, before) = spent.get();
        let before = if stamp == decision { before } else { 0 };
        let after = before.saturating_add(micros);
        spent.set((decision, after));
        after
    }

    /// Delivers one lifecycle event to this thread's kept instance.
    fn observe(&self, event: wit::LifecycleEvent<'_>) {
        let slot = self.slot();
        let mut guard = slot.borrow_mut();
        let now = self.runtime.now_micros();
        if self.breaker.is_open(now) {
            return;
        }
        if guard.is_none() {
            match self.instantiate(self.init_fuel()) {
                Ok(instance) => {
                    bump(&self.tally.lazy_instantiations);
                    *guard = Some(instance);
                }
                Err(_) => {
                    self.report(now);
                    return;
                }
            }
        }
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
        let _ = self.call(&mut guard, lifecycle, |inst| {
            let result = match &inst.exports {
                Exports::Record(b) => b.smg_policy_plugin().call_observe(&mut inst.store, event),
                Exports::Packed(b) => b
                    .smg_policy_plugin_packed()
                    .call_observe(&mut inst.store, event),
            };
            result.map(Ok)
        });
        // Outside a decision: a dropped instance is made again lazily by the
        // next event or open. The mark is a decision's, for its close; a
        // decision never rebuilds a slot it did not touch.
        self.emptied.get_or_default().set(false);
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
    fn frame<'a>(&'a mut self, budget: &'a mut DecisionBudget, fuel_per_micro: u64) -> Frame<'a> {
        Frame {
            cap: self.frame_fuel,
            remaining_micros: &mut budget.remaining_micros,
            fuel_per_micro,
            frame_micros: self.frame_micros,
            now: Some(self.now),
            draw: Some(&mut self.draw),
            quiet: false,
            decision: budget.stamp,
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

/// The stage takes its fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fallback;

impl Bound {
    pub fn plugin(&self) -> &Arc<Plugin> {
        &self.plugin
    }

    /// The first scorer or picker adapter made from any bound set of the
    /// build forwards the lifecycle; the others do not, so an event reaches
    /// the plugin once per build, whichever of its profiles bind the plugin.
    pub(crate) fn take_observer(&self) -> bool {
        !self.observer_taken.swap(true, Ordering::Relaxed)
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
        // The decision's budget is the thread's: its first open seeds it,
        // every open of the decision (each side, each plugin) draws on it.
        let mut budget = plugin.runtime.budgets.get_or_default().borrow_mut();
        budget.opens += 1;
        if budget.opens == 1 {
            budget.remaining_micros = u64::from(facts.budget_micros);
            budget.seed_micros = u64::from(facts.budget_micros);
            budget.stamp = budget.stamp.wrapping_add(1).max(1);
        }
        if plugin.note_open(budget.stamp) {
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
            // A cold slot: `init` runs under what the decision has left (a
            // slot still empty at the close is rebuilt under the full frame).
            // The budget is wall time here as in a call: the instantiation is
            // charged the larger of its fuel at the calibration and its
            // elapsed time, however it ends.
            let fuel_per_micro = frame.fuel_per_micro.max(1);
            let left = budget.remaining_micros.saturating_mul(fuel_per_micro);
            let init_fuel = plugin.init_fuel();
            let started = Instant::now();
            let (made, used) = plugin.instantiate_metered(init_fuel.min(left));
            let elapsed = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
            let charged = used.div_ceil(fuel_per_micro).max(elapsed);
            match made {
                Ok(instance) => {
                    bump(&plugin.tally.lazy_instantiations);
                    budget.remaining_micros = budget.remaining_micros.saturating_sub(charged);
                    plugin.note_init(budget.stamp, charged);
                    *guard = Some(instance);
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
                    // sample is the larger of the remainder and the charge
                    // above, so an `init` long on the clock reads as long
                    // here too.
                    let init_micros = left.div_ceil(fuel_per_micro).max(charged);
                    budget.remaining_micros = budget.remaining_micros.saturating_sub(init_micros);
                    plugin.note_init(budget.stamp, init_micros);
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
                    plugin.note_init(budget.stamp, charged);
                    plugin.fail(now);
                    open.bypass = true;
                    return;
                }
            }
        }
        if let Some(inst) = guard.as_mut() {
            match &inst.exports {
                Exports::Record(_) => inst.records.fill(facts, candidates, pool),
                Exports::Packed(_) => {
                    if inst.packed.fill(facts, candidates, pool, &plugin.exposure) == Err(TooLong) {
                        bump(&plugin.tally.too_long);
                        open.bypass = true;
                        return;
                    }
                }
            }
        }
        let count = open.count;
        let exposure = &plugin.exposure;
        let frame = open.frame(&mut budget, frame.fuel_per_micro);
        let constructed = plugin.call(&mut guard, frame, |inst| {
            let result = match &inst.exports {
                Exports::Record(b) => {
                    let headers = marshal::headers(facts, exposure);
                    let request = marshal::request(facts, exposure, &inst.records.tried, &headers);
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
    /// answers forgotten.
    pub(crate) fn close(&self) {
        let plugin = &self.plugin;
        let mut opens = self.opens.get_or_default().borrow_mut();
        let mut budget = plugin.runtime.budgets.get_or_default().borrow_mut();
        for open in opens.0.iter_mut().filter(|o| o.is_open) {
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
            open.reset();
            budget.opens = budget.opens.saturating_sub(1);
            plugin.note_close(budget.stamp, budget.seed_micros);
        }
        let decision = budget.stamp;
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
    }

    /// The filter's answer for the phase's decision, computed once: `kept`
    /// keeps the positions the guest named.
    pub(crate) fn retain(
        &self,
        facts: &RequestFacts<'_>,
        kept: &mut Vec<u32>,
    ) -> Result<(), Fallback> {
        let mut open = self.state(facts.phase);
        self.ensure_mask(&mut open)?;
        kept.retain(|&i| bit(&open.mask, i));
        Ok(())
    }

    /// One row of the filter's answer.
    pub(crate) fn keep(&self, facts: &RequestFacts<'_>, position: u32) -> Result<bool, Fallback> {
        let mut open = self.state(facts.phase);
        self.ensure_mask(&mut open)?;
        Ok(bit(&open.mask, position))
    }

    fn ensure_mask(&self, open: &mut Open) -> Result<(), Fallback> {
        if open.mask_set {
            return Ok(());
        }
        if open.filter_declined {
            return Err(Fallback);
        }
        let plugin = &self.plugin;
        let count = open.count;
        let mut guard = plugin.slot().borrow_mut();
        let resource = open.live(guard.as_ref())?;
        let mut budget = plugin.runtime.budgets.get_or_default().borrow_mut();
        let frame = open.frame(&mut budget, plugin.frame.fuel_per_micro);
        let answer = plugin.call(&mut guard, frame, |inst| match &inst.exports {
            Exports::Record(b) => b
                .smg_policy_plugin()
                .decision()
                .call_filter(&mut inst.store, resource),
            Exports::Packed(b) => b
                .smg_policy_plugin_packed()
                .decision()
                .call_filter(&mut inst.store, resource),
        });
        open.filter_declined = matches!(answer, Call::Declined);
        let positions = open.answered(answer)?;
        // Unique and in range, else the answer is invalid.
        open.mask.clear();
        open.mask.resize((count as usize).div_ceil(64), 0);
        for &p in &positions {
            if p >= count || bit(&open.mask, p) {
                return Err(open.invalid(plugin, &mut guard));
            }
            open.mask[(p / 64) as usize] |= 1 << (p % 64);
        }
        open.mask_set = true;
        Ok(())
    }

    /// The scorer's answer for the phase's decision, computed once: `out[i]`
    /// is the score of `kept[i]`.
    pub(crate) fn score(
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
            let plugin = &self.plugin;
            let count = open.count;
            let mut guard = plugin.slot().borrow_mut();
            let resource = open.live(guard.as_ref())?;
            let mut budget = plugin.runtime.budgets.get_or_default().borrow_mut();
            let frame = open.frame(&mut budget, plugin.frame.fuel_per_micro);
            let answer = plugin.call(&mut guard, frame, |inst| match &inst.exports {
                Exports::Record(b) => b
                    .smg_policy_plugin()
                    .decision()
                    .call_score(&mut inst.store, resource),
                Exports::Packed(b) => b
                    .smg_policy_plugin_packed()
                    .decision()
                    .call_score(&mut inst.store, resource),
            });
            open.score_declined = matches!(answer, Call::Declined);
            let mut scores = open.answered(answer)?;
            // One finite value per candidate, clamped by the host.
            if scores.len() != count as usize || scores.iter().any(|s| !s.is_finite()) {
                return Err(open.invalid(plugin, &mut guard));
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
    pub(crate) fn pick(
        &self,
        facts: &RequestFacts<'_>,
        kept: &[u32],
        scores: &[f32],
    ) -> Result<u32, Fallback> {
        let mut open = self.state(facts.phase);
        let plugin = &self.plugin;
        let count = open.count;
        let mut guard = plugin.slot().borrow_mut();
        let resource = open.live(guard.as_ref())?;
        let mut scores_in = std::mem::take(&mut open.scores_in);
        scores_in.clear();
        scores_in.resize(count as usize, -1.0);
        for (&i, &s) in kept.iter().zip(scores) {
            if let Some(slot) = scores_in.get_mut(i as usize) {
                *slot = s;
            }
        }
        let mut budget = plugin.runtime.budgets.get_or_default().borrow_mut();
        let frame = open.frame(&mut budget, plugin.frame.fuel_per_micro);
        let answer = plugin.call(&mut guard, frame, |inst| {
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
        // Within the surviving set, else the answer is invalid.
        let within = match &answer {
            Call::Answered(p) => scores_in.get(*p as usize).is_some_and(|s| *s >= 0.0),
            _ => true,
        };
        open.scores_in = scores_in;
        let picked = open.answered(answer)?;
        if !within {
            return Err(open.invalid(plugin, &mut guard));
        }
        Ok(picked)
    }

    pub(crate) fn observe_issued(&self, lease: &Lease, decision: &Decision) {
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

    pub(crate) fn observe_dispatched(
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

    pub(crate) fn observe_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.plugin
            .observe(wit::LifecycleEvent::Completed(marshal::completed(
                lease,
                outcome,
                &self.plugin.exposure,
            )));
    }

    pub(crate) fn observe_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.plugin.observe(wit::LifecycleEvent::Released((
            marshal::lease(lease, &self.plugin.exposure),
            marshal::release_reason(reason),
        )));
    }

    pub(crate) fn observe_membership(&self, change: &Membership<'_>) {
        let plugin = &self.plugin;
        // The snapshot's ids are copied into the instance's own buffer; a
        // borrow of the slot for the copy and another for the call would
        // alias, so the copy is taken apart and handed back after.
        let mut snapshot = {
            let slot = plugin.slot();
            let mut guard = slot.borrow_mut();
            guard
                .as_mut()
                .map(|inst| std::mem::take(&mut inst.snapshot))
                .unwrap_or_default()
        };
        plugin.observe(wit::LifecycleEvent::Membership(marshal::membership(
            change,
            &mut snapshot,
        )));
        if let Some(inst) = plugin.slot().borrow_mut().as_mut() {
            inst.snapshot = snapshot;
        }
    }

    pub(crate) fn observe_signal(&self, signal: &Signal) {
        let now = self.plugin.runtime.now_micros();
        self.plugin
            .observe(wit::LifecycleEvent::Signal(marshal::signal(signal, now)));
    }
}

impl DecisionScope for Bound {
    fn open(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], pool: &PoolFacts<'_>) {
        Bound::open(self, facts, candidates, pool);
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
        bit, deadline_ticks, first_deadline, import_version, judge, thread_cpu_micros, tick,
        Verdict, Watch, EPOCH_TICK_MICROS,
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

    /// The judgement of a call at its end is a function of its readings: the
    /// wall time, the reading at its start or none, the reading at its end,
    /// its fuel and the bound. Every branch, each at its boundary, and the
    /// one case that reads the clock again: a late call whose start was
    /// read.
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
        // fuel in microseconds; ended on the clock; the own time, the
        // verdict and whether the clock was read again.
        let rows = [
            (
                "past its bound, a microsecond short of late: its charge, no second reading",
                read,
                START,
                LATE - 1,
                7,
                false,
                (LATE - 1, Verdict::Own, 0),
            ),
            (
                "late at its bound in whole ticks, the CPU one short of the bound: held",
                read,
                START + BOUND - 1,
                LATE,
                7,
                false,
                (BOUND - 1, Verdict::Held, 1),
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
                false,
                (10 * EPOCH_TICK_MICROS, Verdict::Unread, 0),
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
                false,
                (45, Verdict::Own, 0),
            ),
            (
                "late, the CPU at the bound: the plugin's, at its CPU",
                read,
                START + BOUND,
                LATE,
                7,
                false,
                (BOUND, Verdict::Own, 1),
            ),
            (
                "held, the fuel above the CPU: its own time is its fuel",
                read,
                START + 40,
                9 * BOUND,
                600,
                false,
                (600, Verdict::Held, 1),
            ),
            (
                "the CPU above the wall time: never more than the charge",
                read,
                START + 5 * BOUND,
                LATE,
                7,
                false,
                (LATE, Verdict::Own, 1),
            ),
            (
                "late, read, ended on the clock: the plugin's, never the hold",
                read,
                START + BOUND + 300,
                9 * BOUND,
                7,
                true,
                (BOUND + 300, Verdict::Own, 1),
            ),
            (
                "late, no reading of its start, not ended: unread, at its charge",
                unread,
                START,
                9 * BOUND,
                7,
                false,
                (9 * BOUND, Verdict::Unread, 0),
            ),
            (
                "late, no reading of its start, ended on the clock: the plugin's",
                unread,
                START,
                9 * BOUND,
                7,
                true,
                (9 * BOUND, Verdict::Own, 0),
            ),
            (
                "a first tick's reading is no reading of the start",
                ticked,
                START + 40,
                9 * BOUND,
                7,
                false,
                (9 * BOUND, Verdict::Unread, 0),
            ),
            (
                "nothing watched: the wall clock's call",
                unwatched,
                START,
                9 * BOUND,
                7,
                false,
                (9 * BOUND, Verdict::Own, 0),
            ),
            (
                "not late, the fuel above the wall time: the charge is the fuel",
                unread,
                START,
                10,
                900,
                false,
                (900, Verdict::Own, 0),
            ),
        ];
        for (case, watch, cpu_at_end, elapsed, fuel, ended, wanted) in rows {
            let reads = Cell::new(0);
            let (own, verdict) = judge(&watch, elapsed, fuel, ended, || {
                reads.set(reads.get() + 1);
                Some(cpu_at_end)
            });
            assert_eq!((own, verdict, reads.get()), wanted, "{case}");
        }
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
}
