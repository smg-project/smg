//! Observation handles for one admitted session; the owner alone can authorize completion.

use std::{
    io,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, PoisonError, TryLockError, Weak,
    },
};

use serde::Serialize;

use super::CaptureSession;

/// The only completion authority for a session shared by multiple observation boundaries.
///
/// Drop all observers before finishing. Observers keep admission charged after an owner is
/// abandoned, but reject further records; dropping the final observer abandons the session. This
/// coordinates storage only: callers still must record and validate configuration, boundary
/// coverage and terminal outcomes. No serving path creates these handles automatically.
pub struct SharedCapture {
    inner: Arc<Shared>,
    directory: PathBuf,
    _on_drop: InvalidateOnDrop,
}

struct Shared {
    session: Mutex<CaptureSession>,
    failed: AtomicBool,
    record_bytes_limit: usize,
}

// A weak guard does not add a strong observer reference. After a successful try_unwrap its
// upgrade fails, so dropping it cannot invalidate the session transferred to finish.
struct InvalidateOnDrop(Weak<Shared>);

impl Drop for InvalidateOnDrop {
    fn drop(&mut self) {
        if let Some(shared) = self.0.upgrade() {
            shared.failed.store(true, Ordering::Release);
        }
    }
}

/// A cloneable observation handle with no permission to seal the capture.
///
/// Recording tries the lock once, never waits for a competing serializer or disk. Contention,
/// poisoned locks and queue errors permanently invalidate the shared capture. One successful
/// observer serializes/enqueues at a time, preserving storage order. The underlying session's
/// byte/queue limits apply; handle/serializer allocations and serializer CPU time are excluded.
#[derive(Clone)]
pub struct CaptureObserver(Arc<Shared>);

impl SharedCapture {
    /// Share an already admitted session without adding a writer or changing its budgets.
    pub fn new(session: CaptureSession) -> Self {
        let directory = session.directory().to_owned();
        let inner = Arc::new(Shared {
            record_bytes_limit: session.record_bytes_limit(),
            session: Mutex::new(session),
            failed: AtomicBool::new(false),
        });
        Self {
            _on_drop: InvalidateOnDrop(Arc::downgrade(&inner)),
            inner,
            directory,
        }
    }

    /// Evidence location, including the incomplete prefix after an observation failure.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Create a handle that must be dropped before the owner can finish.
    pub fn observer(&self) -> CaptureObserver {
        CaptureObserver(Arc::clone(&self.inner))
    }

    /// Finish only after all observers are gone, then await the writer and release admission.
    ///
    /// Live observers return `WouldBlock`, invalidate further recording and abandon completion.
    /// A failed observer prevents sealing, but the writer is still joined so its original IO
    /// error is retained. An unpolled future abandons the owner. Once the underlying finish is
    /// polled, cancellation has the same semantics as [`CaptureSession::finish`]. Success is
    /// storage completion, not an inference pass or proof that all boundaries were observed.
    pub async fn finish(self) -> io::Result<PathBuf> {
        let Self {
            inner, _on_drop, ..
        } = self;
        let Shared {
            session, failed, ..
        } = Arc::try_unwrap(inner).map_err(|shared| {
            shared.failed.store(true, Ordering::Release);
            io::Error::new(
                io::ErrorKind::WouldBlock,
                "capture observers are still alive",
            )
        })?;
        drop(_on_drop);
        let poisoned = session.is_poisoned();
        // A panicking serializer already closes the queue. Recover only to await its writer;
        // neither mutex recovery nor a later finish may turn that partial record into a pass.
        let mut session = session.into_inner().unwrap_or_else(PoisonError::into_inner);
        if poisoned || failed.into_inner() {
            session.invalidate();
        }
        session.finish().await
    }
}

impl CaptureObserver {
    /// Bound temporary source encodings before the capped writer checks final JSON overhead.
    pub(crate) fn record_bytes_limit(&self) -> usize {
        self.0.record_bytes_limit
    }

    /// A boundary failed before recording; the owner must not seal this observation prefix.
    pub(crate) fn invalidate(&self) {
        self.0.failed.store(true, Ordering::Release);
    }

    /// Try one observation without waiting; any rejected record makes completion impossible.
    ///
    /// Success means queued, not persisted. A concurrent record can already be in flight when
    /// another observer fails; that record may be retained as incomplete evidence. Keep the
    /// owner outside a caught inference panic and check its finish result before using evidence.
    pub fn record<T: Serialize + ?Sized>(&self, payload: &T) -> io::Result<()> {
        if self.0.failed.load(Ordering::Acquire) {
            return Err(io::Error::other("capture observation is incomplete"));
        }
        let result = match self.0.session.try_lock() {
            Ok(mut session) => session.record(payload),
            Err(TryLockError::WouldBlock) => Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "capture observation is busy",
            )),
            Err(TryLockError::Poisoned(_)) => {
                Err(io::Error::other("capture observer lock poisoned"))
            }
        };
        if result.is_err() {
            self.0.failed.store(true, Ordering::Release);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        panic::{catch_unwind, AssertUnwindSafe},
        sync::mpsc,
        thread,
        time::Duration,
    };

    use serde::{ser::Error as _, Serializer};
    use serde_json::{json, Value};
    use tokio::time::{sleep, timeout};

    use super::*;
    use crate::diagnostic_capture::{CaptureLimits, CapturePool};

    fn pool(root: &Path, file_bytes: u64) -> io::Result<CapturePool> {
        CapturePool::new(
            root.to_owned(),
            CaptureLimits {
                max_active: 1,
                max_attempts: 4,
                max_retained_bytes: file_bytes * 4,
                file_bytes,
                queue_capacity: 64,
                record_bytes: 256,
            },
        )
    }

    async fn after_release(pool: &CapturePool) -> io::Result<CaptureSession> {
        timeout(Duration::from_secs(5), async {
            loop {
                match pool.start().await {
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        sleep(Duration::from_millis(1)).await;
                    }
                    result => return result,
                }
            }
        })
        .await
        .map_err(io::Error::other)?
    }

    struct Panics;
    impl Serialize for Panics {
        fn serialize<S: Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            panic!("deliberate observer serializer panic")
        }
    }

    struct Held {
        entered: mpsc::Sender<()>,
        resume: mpsc::Receiver<()>,
    }
    impl Serialize for Held {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.entered.send(()).map_err(S::Error::custom)?;
            self.resume
                .recv_timeout(Duration::from_secs(5))
                .map_err(S::Error::custom)?;
            serializer.serialize_str("first observation")
        }
    }

    #[tokio::test]
    async fn shared_handles_keep_one_order_and_only_the_owner_seals() {
        let root = tempfile::tempdir().unwrap();
        let pool = pool(root.path(), 8192).unwrap();
        let owner = SharedCapture::new(pool.start().await.unwrap());
        let directory = owner.directory().to_owned();
        let first = owner.observer();
        let second = first.clone();
        let third = owner.observer();
        for n in 0..12 {
            [&first, &second, &third][n % 3]
                .record(&json!({"n": n}))
                .unwrap();
        }
        assert!(!directory.join("complete.json").exists());
        drop((first, second, third));
        assert_eq!(owner.finish().await.unwrap(), directory);
        let data = fs::read_to_string(directory.join("records.jsonl")).unwrap();
        let records: Vec<Value> = data
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(records.len(), 12);
        for (n, record) in records.iter().enumerate() {
            assert_eq!(record, &json!({"sequence": n, "payload": {"n": n}}));
        }
        assert!(directory.join("complete.json").is_file());
        drop(pool.start().await.unwrap());
    }

    #[tokio::test]
    async fn a_live_observer_blocks_finish_and_keeps_admission_charged() {
        let root = tempfile::tempdir().unwrap();
        let pool = pool(root.path(), 8192).unwrap();
        let owner = SharedCapture::new(pool.start().await.unwrap());
        let directory = owner.directory().to_owned();
        let observer = owner.observer();
        assert_eq!(
            owner.finish().await.unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
        assert!(
            observer.record(&Panics).is_err(),
            "failed owner rejects before serialization"
        );
        assert_eq!(
            pool.start().await.err().unwrap().kind(),
            io::ErrorKind::WouldBlock
        );
        drop(observer);
        drop(after_release(&pool).await.unwrap());
        assert!(!directory.join("complete.json").exists());
    }

    #[tokio::test]
    async fn contention_rejects_before_serializing_and_never_seals() {
        let root = tempfile::tempdir().unwrap();
        let pool = pool(root.path(), 8192).unwrap();
        let owner = SharedCapture::new(pool.start().await.unwrap());
        let directory = owner.directory().to_owned();
        let first = owner.observer();
        let second = owner.observer();
        let (entered, ready) = mpsc::channel();
        let (resume, receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            first.record(&Held {
                entered,
                resume: receiver,
            })
        });
        ready.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(
            second.record(&Panics).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
        resume.send(()).unwrap();
        worker.join().unwrap().unwrap();
        assert!(second.record(&Panics).is_err());
        drop(second);
        assert!(owner.finish().await.is_err());
        assert!(!directory.join("complete.json").exists());
        drop(pool.start().await.unwrap());
    }

    #[tokio::test]
    async fn a_caught_serializer_panic_cannot_complete_after_mutex_recovery() {
        let root = tempfile::tempdir().unwrap();
        let pool = pool(root.path(), 8192).unwrap();
        let owner = SharedCapture::new(pool.start().await.unwrap());
        let directory = owner.directory().to_owned();
        let observer = owner.observer();
        assert!(catch_unwind(AssertUnwindSafe(|| observer.record(&Panics))).is_err());
        drop(observer);
        assert!(owner.finish().await.is_err());
        assert!(!directory.join("complete.json").exists());
        drop(pool.start().await.unwrap());
    }

    #[tokio::test]
    async fn writer_failure_survives_later_observation_errors() {
        let root = tempfile::tempdir().unwrap();
        let pool = pool(root.path(), 600).unwrap();
        let owner = SharedCapture::new(pool.start().await.unwrap());
        let directory = owner.directory().to_owned();
        let observer = owner.observer();
        observer.record(&"x".repeat(100)).unwrap(); // Fits record cap, exceeds file allowance.
        for _ in 0..65 {
            if observer.record(&"later").is_err() {
                break;
            }
        }
        drop(observer);
        assert_eq!(
            owner.finish().await.unwrap_err().kind(),
            io::ErrorKind::FileTooLarge
        );
        assert!(!directory.join("complete.json").exists());
        drop(pool.start().await.unwrap());
    }

    #[tokio::test]
    async fn oversized_observations_invalidate_every_handle() {
        let root = tempfile::tempdir().unwrap();
        let pool = pool(root.path(), 8192).unwrap();
        let owner = SharedCapture::new(pool.start().await.unwrap());
        let directory = owner.directory().to_owned();
        let observer = owner.observer();
        let another = observer.clone();
        assert_eq!(
            observer.record(&"x".repeat(1024)).unwrap_err().kind(),
            io::ErrorKind::FileTooLarge
        );
        assert!(another.record(&Panics).is_err());
        drop((observer, another));
        assert!(owner.finish().await.is_err());
        assert!(!directory.join("complete.json").exists());
    }

    #[tokio::test]
    async fn dropped_and_unpolled_owners_never_authorize_completion() {
        for unpolled in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let pool = pool(root.path(), 8192).unwrap();
            let owner = SharedCapture::new(pool.start().await.unwrap());
            let directory = owner.directory().to_owned();
            let observer = owner.observer();
            observer.record(&"prefix").unwrap();
            if unpolled {
                drop(owner.finish());
            } else {
                drop(owner);
            }
            assert!(
                observer.record(&Panics).is_err(),
                "abandonment rejects before serialization"
            );
            assert_eq!(
                pool.start().await.err().unwrap().kind(),
                io::ErrorKind::WouldBlock
            );
            drop(observer);
            drop(after_release(&pool).await.unwrap());
            assert!(!directory.join("complete.json").exists());
        }
    }
}
