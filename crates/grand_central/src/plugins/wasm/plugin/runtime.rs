//! The runtime every plugin of a process shares, and the epoch ticker that
//! runs while a guest call is in flight. A child of the plugin's module, so
//! that its items reach the plugin, its binding and their cases, and nothing
//! more of the crate.

use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc,
    },
    time::Instant,
};

use thread_local::ThreadLocal;
use wasmtime::{
    component::{HasSelf, Linker},
    Config, Engine, OptLevel,
};

use super::{
    super::bindings::{record, HostState},
    clock::{thread_cpu_micros, EPOCH_TICK_MICROS},
    LoadError,
};
use crate::plugins::shared::OpenDecisions;

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
pub(super) struct Running<'a>(&'a AtomicU32);

impl<'a> Running<'a> {
    pub(super) fn start(runtime: &'a Runtime) -> Self {
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
    pub(super) engine: Engine,
    pub(super) linker: Linker<HostState>,
    pub(super) clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    /// The epoch ticker's state, shared with its thread.
    ticking: Arc<Ticking>,
    /// The ticker thread, to wake it when a call starts.
    ticker: std::thread::Thread,
    /// The decision open on each thread and its budget: shared with every
    /// runtime built over the same ones.
    pub(super) decisions: Arc<OpenDecisions>,
    /// Whether this host has a thread CPU clock: a call's bound then counts
    /// the time its thread had the CPU, and is the wall clock's otherwise.
    pub(super) thread_clock: bool,
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
    /// [`RequestFacts::now_micros`](crate::engine::RequestFacts::now_micros).
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
pub(super) fn configure(config: &mut Config) {
    config.wasm_component_model(true);
    config.concurrency_support(false);
    config.consume_fuel(true);
    config.epoch_interruption(true);
    config.cranelift_opt_level(OptLevel::Speed);
}
