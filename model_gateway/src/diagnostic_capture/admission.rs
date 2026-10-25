//! Shared admission and blocking-writer lifetime accounting for one capture run.

use std::{
    io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, PoisonError},
    thread::{self, JoinHandle},
};

use serde::Serialize;
use tokio::{runtime::Handle, sync::Semaphore};

use super::{CaptureFile, CaptureQueue, CaptureWriter, BUFFER_BYTES, RESERVED_BYTES};

/// Explicit bounds, validated before any filesystem work or task submission.
#[derive(Clone, Copy)]
pub struct CaptureLimits {
    /// Maximum sessions, counting initialization, owners and unfinished writers.
    pub max_active: usize,
    /// Total admitted attempts, including failed creation and abandoned captures.
    pub max_attempts: u64,
    /// Maximum logical bytes reserved for all attempts; reservations are never refunded.
    pub max_retained_bytes: u64,
    /// Per-file logical byte allowance, including the storage marker reservation.
    pub file_bytes: u64,
    /// Per-session queued/serializing record slots.
    pub queue_capacity: usize,
    /// Maximum serialized payload bytes per record, excluding its storage envelope.
    pub record_bytes: usize,
}

/// Share clones of one pool across a gateway run; independent pools have independent budgets.
///
/// Use an existing, absolute, operator-controlled root. These limits cover only this pool's
/// attempts, not old files, other writers, filesystem block overhead or later process runs.
/// External retention remains necessary. No serving path enables capture automatically.
#[derive(Clone)]
pub struct CapturePool(Arc<Shared>);

struct Shared {
    root: PathBuf,
    limits: CaptureLimits,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    active: usize,
    attempts: u64,
    reserved: u64,
}

struct Lease(Arc<Shared>);

impl Drop for Lease {
    fn drop(&mut self) {
        // Cleanup may recover a poisoned guard; subsequent admission still fails closed.
        self.0
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .active -= 1;
    }
}

/// A charged capture owner; its active slot is shared with the dedicated writer thread.
pub struct CaptureSession {
    directory: PathBuf,
    queue: CaptureQueue,
    worker: JoinHandle<()>,
    lease: Arc<Lease>,
}

impl CapturePool {
    /// Validate limits without IO. Retained bytes and attempts are charged before creation.
    ///
    /// Active payload storage is bounded by `max_active * (queue_capacity + 2) * record_bytes`,
    /// including one transient reallocation per producer, plus fixed file buffers, allocator,
    /// channel/task overhead, dedicated thread stacks and caller-provided serializer allocations.
    pub fn new(root: PathBuf, limits: CaptureLimits) -> io::Result<Self> {
        let memory = limits
            .queue_capacity
            .checked_add(2)
            .and_then(|n| n.checked_mul(limits.record_bytes))
            .and_then(|n| n.checked_add(BUFFER_BYTES))
            .and_then(|n| n.checked_mul(limits.max_active));
        if !root.is_absolute()
            || limits.queue_capacity == 0
            || limits.queue_capacity > Semaphore::MAX_PERMITS
            || limits.record_bytes == 0
            || limits.max_active == 0
            || limits.max_attempts == 0
            || limits.file_bytes < RESERVED_BYTES
            || limits.max_retained_bytes < limits.file_bytes
            || memory.is_none()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid capture pool limits or root",
            ));
        }
        Ok(Self(Arc::new(Shared {
            root,
            limits,
            state: Mutex::new(State::default()),
        })))
    }

    /// Admit immediately or return `WouldBlock` (active slots) / `QuotaExceeded` (run budget).
    ///
    /// Requires Tokio for file creation; each writer has a dedicated thread so an idle capture
    /// cannot pin the runtime's shared blocking pool. Cancellation
    /// does not refund an admitted attempt: active capacity returns only after both the owner
    /// and writer exit. Failed initialization still consumes its disk/attempt reservation.
    pub async fn start(&self) -> io::Result<CaptureSession> {
        self.start_with(CaptureWriter::run).await
    }

    async fn start_with(
        &self,
        run: impl FnOnce(CaptureWriter) + Send + 'static,
    ) -> io::Result<CaptureSession> {
        let handle = Handle::try_current().map_err(io::Error::other)?;
        let lease = self.reserve()?;
        let shared = self.0.clone();
        handle
            .spawn_blocking(move || {
                let limits = shared.limits;
                let file = CaptureFile::create(&shared.root, limits.file_bytes)?;
                let directory = file.directory().to_owned();
                let (queue, writer) =
                    CaptureQueue::new(file, limits.queue_capacity, limits.record_bytes)?;
                let writer_lease = lease.clone();
                let worker = thread::Builder::new()
                    .name("smg-capture".into())
                    .spawn(move || {
                        let _lease = writer_lease;
                        run(writer);
                    })
                    .map_err(io::Error::other)?;
                Ok(CaptureSession {
                    directory,
                    queue,
                    worker,
                    lease,
                })
            })
            .await
            .map_err(io::Error::other)?
    }

    fn reserve(&self) -> io::Result<Arc<Lease>> {
        let limits = self.0.limits;
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| io::Error::other("capture budget poisoned"))?;
        if state.attempts == limits.max_attempts
            || limits.file_bytes > limits.max_retained_bytes - state.reserved
        {
            return Err(io::Error::new(
                io::ErrorKind::QuotaExceeded,
                "capture run budget exhausted",
            ));
        }
        if state.active == limits.max_active {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "capture sessions busy",
            ));
        }
        state.active += 1;
        state.attempts += 1;
        state.reserved += limits.file_bytes;
        Ok(Arc::new(Lease(self.0.clone())))
    }
}

impl CaptureSession {
    /// Generated evidence directory, also available before a failed or abandoned completion.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Admit one bounded record without waiting for disk; queue errors permanently invalidate it.
    pub fn record<T: Serialize + ?Sized>(&mut self, payload: &T) -> io::Result<()> {
        self.queue.record(payload)
    }

    /// Invalidate observation coverage while retaining the writer acknowledgement and lease.
    pub(super) fn invalidate(&mut self) {
        self.queue.invalidate();
    }

    /// Seal and join the writer before releasing admission; requires a Tokio runtime.
    ///
    /// Polling authorizes sealing even if this waiter is later cancelled. The active lease then
    /// remains with the writer until it exits. Cancelled owners detach writer threads; runtime
    /// shutdown does not join them. Await finish before process exit for durable completion.
    /// Storage success still does not prove coverage.
    pub async fn finish(self) -> io::Result<PathBuf> {
        let handle = Handle::try_current().map_err(io::Error::other)?;
        let Self {
            queue,
            worker,
            lease,
            ..
        } = self;
        let result = queue.finish().await;
        // Join only after the writer has sent its result (or unwound), never while idle.
        handle
            .spawn_blocking(move || {
                let _lease = lease;
                worker
                    .join()
                    .map_err(|_| io::Error::other("capture writer panicked"))
            })
            .await
            .map_err(io::Error::other)??;
        result
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        future::Future,
        sync::mpsc,
        task::{Context, Waker},
        time::Duration,
    };

    use tokio::{
        sync::{oneshot, Barrier},
        task::JoinSet,
        time::{sleep, timeout},
    };

    use super::*;

    fn limits() -> CaptureLimits {
        CaptureLimits {
            max_active: 2,
            max_attempts: 4,
            max_retained_bytes: 4096,
            file_bytes: 1024,
            queue_capacity: 2,
            record_bytes: 64,
        }
    }

    fn make_pool(root: &Path, edit: impl FnOnce(&mut CaptureLimits)) -> CapturePool {
        let mut bounds = limits();
        edit(&mut bounds);
        CapturePool::new(root.to_owned(), bounds).unwrap()
    }

    async fn rejected(pool: &CapturePool, kind: io::ErrorKind) {
        assert_eq!(pool.start().await.err().unwrap().kind(), kind);
    }

    async fn wait_until(ready: impl Fn() -> bool) {
        timeout(Duration::from_secs(5), async {
            while !ready() {
                sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
    }

    #[test]
    fn invalid_limits_are_rejected_without_creating_files() {
        let root = tempfile::tempdir().unwrap();
        let invalid: [fn(&mut CaptureLimits); 9] = [
            |l| l.max_active = 0,
            |l| l.max_attempts = 0,
            |l| l.file_bytes = 511,
            |l| l.max_retained_bytes = 1023,
            |l| l.queue_capacity = 0,
            |l| l.queue_capacity = usize::MAX,
            |l| l.record_bytes = 0,
            |l| l.record_bytes = usize::MAX,
            |l| l.max_active = usize::MAX,
        ];
        for edit in invalid {
            let mut bad = limits();
            edit(&mut bad);
            assert!(
                matches!(CapturePool::new(root.path().to_owned(), bad), Err(e) if e.kind() == io::ErrorKind::InvalidInput)
            );
        }
        assert!(CapturePool::new(PathBuf::from("relative"), limits()).is_err());
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn concurrent_clones_share_active_limits_and_release_after_finish() {
        let root = tempfile::tempdir().unwrap();
        let pool = CapturePool::new(root.path().to_owned(), limits()).unwrap();
        let barrier = Arc::new(Barrier::new(12));
        let mut tasks = JoinSet::new();
        for _ in 0..12 {
            let (pool, barrier) = (pool.clone(), barrier.clone());
            tasks.spawn(async move {
                barrier.wait().await;
                pool.start().await
            });
        }
        let mut sessions = Vec::new();
        while let Some(result) = tasks.join_next().await {
            match result.unwrap() {
                Ok(session) => sessions.push(session),
                Err(error) => assert_eq!(error.kind(), io::ErrorKind::WouldBlock),
            }
        }
        assert_eq!(sessions.len(), 2);
        for session in sessions {
            session.finish().await.unwrap();
        }
        pool.start().await.unwrap().finish().await.unwrap();
    }

    #[tokio::test]
    async fn retained_disk_and_attempt_budgets_are_never_refunded() {
        let root = tempfile::tempdir().unwrap();
        let pool = make_pool(root.path(), |l| l.max_retained_bytes = 2048);
        for _ in 0..2 {
            let mut session = pool.start().await.unwrap();
            session.record(&1).unwrap();
            session.finish().await.unwrap();
        }
        rejected(&pool, io::ErrorKind::QuotaExceeded).await;
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 2);
        let missing = root.path().join("missing");
        let pool = make_pool(&missing, |l| l.max_attempts = 1);
        rejected(&pool, io::ErrorKind::NotFound).await;
        fs::create_dir(&missing).unwrap();
        rejected(&pool, io::ErrorKind::QuotaExceeded).await;
        assert_eq!(pool.0.state.lock().unwrap().active, 0);
    }

    #[tokio::test]
    async fn writer_failure_keeps_owner_charged_and_finish_preserves_error() {
        let root = tempfile::tempdir().unwrap();
        let pool = make_pool(root.path(), |l| {
            l.max_active = 1;
            l.file_bytes = 512;
        });
        let mut session = pool.start().await.unwrap();
        session.record(&1).unwrap();
        wait_until(|| session.worker.is_finished()).await;
        rejected(&pool, io::ErrorKind::WouldBlock).await;
        assert_eq!(
            session.finish().await.unwrap_err().kind(),
            io::ErrorKind::FileTooLarge
        );
        let session = pool
            .start_with(|_writer| panic!("writer trap"))
            .await
            .unwrap();
        let directory = session.directory().to_owned();
        assert!(session.finish().await.is_err());
        assert!(!directory.join("complete.json").exists());
        pool.start().await.unwrap().finish().await.unwrap();
    }

    #[tokio::test]
    async fn abandoned_and_cancelled_owners_stay_charged_until_writer_exit() {
        for seal in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let pool = make_pool(root.path(), |l| l.max_active = 1);
            let (release, blocked) = mpsc::channel();
            let session = pool
                .start_with(move |writer| {
                    blocked.recv_timeout(Duration::from_secs(5)).unwrap();
                    writer.run();
                })
                .await
                .unwrap();
            let directory = session.directory().to_owned();
            let mut finishing = Box::pin(session.finish());
            if seal {
                // One poll accepts the seal; the controlled writer cannot acknowledge it yet.
                assert!(finishing
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending());
            }
            drop(finishing);
            rejected(&pool, io::ErrorKind::WouldBlock).await;
            release.send(()).unwrap();
            wait_until(|| pool.0.state.lock().unwrap().active == 0).await;
            assert_eq!(directory.join("complete.json").exists(), seal);
            pool.start().await.unwrap().finish().await.unwrap();
        }
    }
    #[test]
    fn active_writer_does_not_pin_the_shared_blocking_pool() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(1)
            .enable_time()
            .build()
            .unwrap();
        runtime.block_on(async {
            let root = tempfile::tempdir().unwrap();
            let pool = make_pool(root.path(), |l| l.max_active = 1);
            let (ready, started) = oneshot::channel();
            let session = pool
                .start_with(move |writer| {
                    ready.send(()).unwrap();
                    writer.run();
                })
                .await
                .unwrap();
            started.await.unwrap();
            let probe = tokio::task::spawn_blocking(|| ());
            let progressed = timeout(Duration::from_secs(1), probe).await.is_ok();
            // Release the writer even on regression, before failing or dropping the runtime.
            session.finish().await.unwrap();
            assert!(progressed, "capture writer starved unrelated blocking work");
        });
    }
}
