//! Execution of CPU-bound vision preprocessing: how work splits into tasks
//! and where those tasks run.
//!
//! The crate never touches rayon's global pool. A host chooses once, at
//! startup, between a dedicated pool of its own size ([`Parallelism::Pool`])
//! and running every task inline on the calling thread
//! ([`Parallelism::Inline`]), for a host that already runs requests
//! concurrently and wants preprocessing to cost exactly one thread each.
//! Unconfigured, the crate builds a small dedicated pool on first use.
//!
//! Why not the global pool: its worker count follows the machine, and every
//! entry wakes a share of those workers whether or not there is work for
//! them. Preprocessing a single image is sub-millisecond, so on a large host
//! the wake-ups cost several times the work itself.
//!
//! Under load the pool is shared by every request. A split is bounded by the
//! pool's free share, counted from the tasks spawned and not yet returned, so
//! one image on an idle host spreads across the pool while eight concurrent
//! images each keep their caller's thread instead of queueing behind one
//! another's bands. The bytes do not depend on the split.
//!
//! The pool's size therefore bounds one image's split, not the total: an
//! operation that finds the pool busy runs on its caller's thread, so
//! preprocessing under load uses about the pool's threads plus one per
//! concurrent request.

use std::{
    marker::PhantomData,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    },
};

const PARALLEL_MIN_BYTES: usize = 1 << 19;
const MAX_TASKS_PER_OPERATION: usize = 8;
/// Threads in the dedicated pool when the host names no size: enough to
/// split one large image, few enough to leave the host its cores.
const DEFAULT_POOL_THREADS: usize = 8;
/// Overrides the dedicated pool's size (a positive thread count): how many
/// threads one image's stages may spread over. An operation that finds the
/// pool busy runs on its caller's thread rather than waiting for it, so this
/// is not a cap on the preprocessing CPU under load.
pub const POOL_THREADS_ENV: &str = "SMG_MM_THREADS";
/// Prefix of the dedicated pool's thread names.
pub const POOL_THREAD_NAME_PREFIX: &str = "smg-mm-";

/// Where preprocessing tasks run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parallelism {
    /// A dedicated pool of this many threads, built on first use.
    Pool(NonZeroUsize),
    /// Every task runs on the calling thread, in order; no threads are
    /// created. For a host that supplies the concurrency itself.
    Inline,
}

impl Parallelism {
    /// The pool the crate uses when the host names none: `SMG_MM_THREADS`
    /// when set to a positive count, else the smaller of the machine's
    /// parallelism and eight.
    pub fn default_pool() -> Self {
        let from_env = std::env::var(POOL_THREADS_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .and_then(NonZeroUsize::new);
        let threads = from_env.unwrap_or_else(|| {
            let cores = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
            NonZeroUsize::new(cores.clamp(1, DEFAULT_POOL_THREADS)).unwrap_or(NonZeroUsize::MIN)
        });
        Self::Pool(threads)
    }
}

static PARALLELISM: OnceLock<Parallelism> = OnceLock::new();
static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
/// Tasks spawned on the pool and not yet returned: the pool's busy share,
/// which a new operation's split leaves alone.
static BUSY_TASKS: AtomicUsize = AtomicUsize::new(0);

/// One task counted in [`BUSY_TASKS`] from its spawn until it returns or
/// unwinds.
struct BusyTask;

impl BusyTask {
    fn begin() -> Self {
        BUSY_TASKS.fetch_add(1, Ordering::Relaxed);
        BusyTask
    }
}

impl Drop for BusyTask {
    fn drop(&mut self) {
        BUSY_TASKS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Choose where preprocessing runs, once per process, before the first
/// preprocessing call. Returns the mode already in force when it differs
/// from the one asked for (a later caller cannot change it).
pub fn configure_parallelism(mode: Parallelism) -> Result<(), Parallelism> {
    match PARALLELISM.set(mode) {
        Ok(()) => Ok(()),
        Err(_) => {
            let current = parallelism();
            if current == mode {
                Ok(())
            } else {
                Err(current)
            }
        }
    }
}

/// The mode in force (the default pool when the host chose none).
pub fn parallelism() -> Parallelism {
    *PARALLELISM.get_or_init(Parallelism::default_pool)
}

/// The pool, built on first use; `None` when its threads could not be
/// created, in which case tasks run inline on the calling thread.
fn pool(threads: NonZeroUsize) -> Option<&'static rayon::ThreadPool> {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.get())
            .thread_name(|index| format!("{POOL_THREAD_NAME_PREFIX}{index}"))
            .build()
            .map_err(|error| {
                tracing::warn!(%error, "could not build the preprocessing pool; running inline");
            })
            .ok()
    })
    .as_ref()
}

/// Spawns the tasks of one [`scope`]: onto the pool, or inline in order.
pub struct Spawner<'a, 'scope> {
    inner: SpawnerInner<'a, 'scope>,
}

enum SpawnerInner<'a, 'scope> {
    Pool(&'a rayon::Scope<'scope>),
    Inline(PhantomData<&'scope ()>),
}

impl<'scope> Spawner<'_, 'scope> {
    /// Run `task` within the scope: concurrently on the pool, or right now
    /// inline. The scope ends when every task has.
    pub fn spawn<F>(&self, task: F)
    where
        F: FnOnce(&Spawner<'_, 'scope>) + Send + 'scope,
    {
        match self.inner {
            SpawnerInner::Pool(scope) => {
                let busy = BusyTask::begin();
                scope.spawn(move |scope| {
                    let _busy = busy;
                    task(&Spawner {
                        inner: SpawnerInner::Pool(scope),
                    });
                });
            }
            SpawnerInner::Inline(_) => task(&Spawner {
                inner: SpawnerInner::Inline(PhantomData),
            }),
        }
    }
}

/// Run `operation` with a spawner for its tasks; returns once every task has
/// finished.
pub(crate) fn scope<'scope, OP, R>(operation: OP) -> R
where
    OP: FnOnce(&Spawner<'_, 'scope>) -> R + Send,
    R: Send,
{
    let pool = match parallelism() {
        Parallelism::Inline => None,
        Parallelism::Pool(threads) => pool(threads),
    };
    match pool {
        Some(pool) => pool.scope(|scope| {
            operation(&Spawner {
                inner: SpawnerInner::Pool(scope),
            })
        }),
        None => operation(&Spawner {
            inner: SpawnerInner::Inline(PhantomData),
        }),
    }
}

/// How many tasks to split an operation into: one unless the output is
/// large, the mode runs tasks concurrently and the pool has threads to
/// spare; an operation that finds the pool busy runs whole on its caller's
/// thread rather than queueing its bands behind other operations' tasks.
pub(crate) fn task_count(
    output_bytes: usize,
    work_items: usize,
    min_items_per_task: usize,
) -> usize {
    debug_assert!(min_items_per_task > 0);
    let threads = match parallelism() {
        Parallelism::Inline => return 1,
        Parallelism::Pool(threads) => threads.get(),
    };
    if output_bytes < PARALLEL_MIN_BYTES || work_items < 2 * min_items_per_task {
        return 1;
    }
    let free = threads.saturating_sub(BUSY_TASKS.load(Ordering::Relaxed));
    if free <= 1 {
        return 1;
    }
    (work_items / min_items_per_task)
        .min(free)
        .clamp(1, MAX_TASKS_PER_OPERATION)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_operations_stay_serial() {
        assert_eq!(task_count(PARALLEL_MIN_BYTES - 1, 1_024, 1), 1);
        assert_eq!(task_count(PARALLEL_MIN_BYTES, 63, 32), 1);
    }

    #[test]
    fn task_count_respects_operation_and_pool_limits() {
        let tasks = task_count(PARALLEL_MIN_BYTES, usize::MAX, 1);
        assert!(tasks <= MAX_TASKS_PER_OPERATION);
        if let Parallelism::Pool(threads) = parallelism() {
            assert!(tasks <= threads.get());
        } else {
            assert_eq!(tasks, 1);
        }
    }

    #[test]
    fn the_default_pool_is_small() {
        let Parallelism::Pool(threads) = Parallelism::default_pool() else {
            panic!("the default is a pool");
        };
        assert!(threads.get() <= DEFAULT_POOL_THREADS);
    }

    /// The two tests of this module that hold the shared pool take this, so
    /// neither holds it under the other; tests elsewhere in the binary spawn
    /// on the pool too, which is why the counts below are read with slack.
    static POOL_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn a_busy_pool_keeps_a_new_operation_whole() {
        use std::{
            sync::{mpsc, Mutex},
            time::{Duration, Instant},
        };
        let _serial = POOL_TESTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Parallelism::Pool(threads) = parallelism() else {
            return;
        };
        let threads = threads.get();
        let before = BUSY_TASKS.load(Ordering::Relaxed);
        let (release, held) = mpsc::channel::<()>();
        let held = Mutex::new(held);
        let split = std::thread::scope(|threads_scope| {
            threads_scope.spawn(|| {
                scope(|spawner| {
                    for _ in 0..threads {
                        spawner.spawn(|_| {
                            // Holds a pool thread until released.
                            let _ = held
                                .lock()
                                .expect("receiver")
                                .recv_timeout(Duration::from_secs(10));
                        });
                    }
                });
            });
            let deadline = Instant::now() + Duration::from_secs(5);
            while BUSY_TASKS.load(Ordering::Relaxed) < before + threads && Instant::now() < deadline
            {
                std::thread::sleep(Duration::from_millis(1));
            }
            let split = task_count(PARALLEL_MIN_BYTES, usize::MAX, 1);
            // Released before any assertion, so a failure cannot wait out
            // the holds.
            for _ in 0..threads {
                let _ = release.send(());
            }
            drop(release);
            split
        });
        assert_eq!(
            split, 1,
            "a pool held by {threads} tasks splits a new operation no further"
        );
        // The count returns once the held tasks do. Other tests of the binary
        // spawn on the pool as well, so the count is not exact at an instant:
        // what holds is that it comes back down to, or below, its reading
        // before the holds, within the time their tasks take.
        let deadline = Instant::now() + Duration::from_secs(5);
        while BUSY_TASKS.load(Ordering::Relaxed) > before && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            BUSY_TASKS.load(Ordering::Relaxed) <= before,
            "the count did not return once the held tasks did"
        );
    }

    #[test]
    fn a_scope_runs_every_task_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let _serial = POOL_TESTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let count = AtomicUsize::new(0);
        scope(|spawner| {
            for _ in 0..16 {
                spawner.spawn(|_| {
                    count.fetch_add(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(count.load(Ordering::SeqCst), 16);
    }
}
