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

use std::{marker::PhantomData, num::NonZeroUsize, sync::OnceLock};

const PARALLEL_MIN_BYTES: usize = 1 << 19;
const MAX_TASKS_PER_OPERATION: usize = 8;
/// Threads in the dedicated pool when the host names no size: enough to
/// split one large image, few enough to leave the host its cores.
const DEFAULT_POOL_THREADS: usize = 8;
/// Overrides the dedicated pool's size (a positive thread count).
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
            SpawnerInner::Pool(scope) => scope.spawn(move |scope| {
                task(&Spawner {
                    inner: SpawnerInner::Pool(scope),
                });
            }),
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
/// large and the mode runs tasks concurrently.
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
    (work_items / min_items_per_task)
        .min(threads)
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

    #[test]
    fn a_scope_runs_every_task_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
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
