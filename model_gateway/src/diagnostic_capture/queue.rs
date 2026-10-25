//! Single-producer mailbox for a separately scheduled blocking capture writer.

use std::{
    io::{self, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use serde::Serialize;
use serde_json::value::RawValue;
use tokio::sync::{mpsc, oneshot, Semaphore};

use super::CaptureFile;

/// Non-cloneable capture owner. Records require exclusive access and never wait for disk.
///
/// A failed record permanently closes admission. Dropping the owner without polling `finish`
/// leaves incomplete evidence. Scheduling, process-wide quotas and semantic completeness are
/// still the caller's responsibility; no serving path uses this mailbox automatically.
pub struct CaptureQueue {
    sender: Option<mpsc::Sender<Vec<u8>>>,
    completion: oneshot::Receiver<io::Result<PathBuf>>,
    sealed: Arc<AtomicBool>,
    max_record_bytes: usize,
}

/// Blocking consumer paired with exactly one `CaptureQueue`.
pub struct CaptureWriter {
    receiver: mpsc::Receiver<Vec<u8>>,
    completion: oneshot::Sender<io::Result<PathBuf>>,
    sealed: Arc<AtomicBool>,
    file: CaptureFile,
}

impl CaptureQueue {
    /// Pair an already created file with a bounded mailbox; this does not start a thread.
    ///
    /// Both limits must be positive. Admission reserves a slot before allocating or serializing.
    /// Queued, serializing and actively written payload buffers together hold at most
    /// `(capacity + 1) * max_record_bytes` bytes, excluding allocator/channel overhead, the
    /// file's fixed buffer and allocations performed by the caller's `Serialize` implementation.
    /// The record limit excludes the storage envelope, which has its own total file budget.
    pub fn new(
        file: CaptureFile,
        capacity: usize,
        max_record_bytes: usize,
    ) -> io::Result<(Self, CaptureWriter)> {
        if capacity == 0
            || capacity > Semaphore::MAX_PERMITS
            || max_record_bytes == 0
            || capacity
                .checked_add(1)
                .and_then(|n| n.checked_mul(max_record_bytes))
                .is_none()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid capture queue limits",
            ));
        }
        let (sender, receiver) = mpsc::channel(capacity);
        let (completion_tx, completion) = oneshot::channel();
        let sealed = Arc::new(AtomicBool::new(false));
        Ok((
            Self {
                sender: Some(sender),
                completion,
                sealed: sealed.clone(),
                max_record_bytes,
            },
            CaptureWriter {
                receiver,
                completion: completion_tx,
                sealed,
                file,
            },
        ))
    }

    /// Serialize one capped payload after reserving capacity, without waiting for the writer.
    ///
    /// Full/closed queues, serialization errors or panics permanently invalidate the capture.
    /// Success means queued, not durably stored. The writer's result is returned by `finish`.
    pub fn record<T: Serialize + ?Sized>(&mut self, payload: &T) -> io::Result<()> {
        // Keep admission closed on every early return, including an unwinding serializer.
        let sender = self.sender.take().ok_or_else(incomplete)?;
        {
            let permit = sender.try_reserve().map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => {
                    io::Error::new(io::ErrorKind::WouldBlock, "capture queue full")
                }
                mpsc::error::TrySendError::Closed(_) => {
                    io::Error::new(io::ErrorKind::BrokenPipe, "capture writer closed")
                }
            })?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(self.max_record_bytes)
                .map_err(io::Error::other)?;
            let mut buffer = Payload {
                bytes,
                limit: self.max_record_bytes,
            };
            serde_json::to_writer(&mut buffer, payload).map_err(io::Error::from)?;
            permit.send(buffer.bytes);
        }
        self.sender = Some(sender);
        Ok(())
    }

    /// Explicitly seal admission, drain accepted records and await durable storage completion.
    ///
    /// Once polled, this authorizes sealing even if the awaiting future is subsequently dropped.
    /// An unpolled future instead drops the owner without sealing. A caller must observe `Ok`
    /// before reporting success; a storage marker still does not establish boundary coverage.
    pub async fn finish(mut self) -> io::Result<PathBuf> {
        let sender = self.sender.take().ok_or_else(incomplete)?;
        self.sealed.store(true, Ordering::Release);
        drop(sender);
        self.completion
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "capture writer lost"))?
    }
}

impl CaptureWriter {
    /// Drain on a blocking thread (for example `spawn_blocking`), never an async runtime worker.
    ///
    /// All storage failures are returned through the owner's `finish`. If that owner was dropped,
    /// the receipt has no observer; this never upgrades abandoned input to complete evidence.
    pub fn run(mut self) {
        let result = (|| {
            while let Some(payload) = self.receiver.blocking_recv() {
                let raw: &RawValue = serde_json::from_slice(&payload).map_err(io::Error::from)?;
                self.file.record(raw)?;
            }
            if !self.sealed.load(Ordering::Acquire) {
                return Err(incomplete());
            }
            self.file.finish()
        })();
        // The explicit sealing request remains valid if its acknowledgement receiver is gone.
        let _ = self.completion.send(result);
    }
}

fn incomplete() -> io::Error {
    io::Error::other("capture admission is incomplete")
}

struct Payload {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for Payload {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit - self.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                "capture payload too large",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        future::Future,
        panic::{catch_unwind, AssertUnwindSafe},
        task::{Context, Waker},
    };

    use serde_json::{json, Value};

    use super::*;

    fn pair(
        root: &std::path::Path,
        slots: usize,
        bytes: usize,
    ) -> (CaptureQueue, CaptureWriter, PathBuf) {
        let file = CaptureFile::create(root, 4096).unwrap();
        let path = file.directory().to_owned();
        let (queue, writer) = CaptureQueue::new(file, slots, bytes).unwrap();
        (queue, writer, path)
    }

    #[tokio::test]
    async fn seal_drains_in_order_and_acknowledges_marker() {
        let root = tempfile::tempdir().unwrap();
        let (mut queue, writer, path) = pair(root.path(), 2, 64);
        queue.record(&json!({"text": "héllo"})).unwrap();
        queue.record(&json!({"done": true})).unwrap();
        assert!(!path.join("complete.json").exists());
        let worker = tokio::task::spawn_blocking(|| writer.run());
        assert_eq!(queue.finish().await.unwrap(), path);
        worker.await.unwrap();
        let records: Vec<Value> = fs::read_to_string(path.join("records.jsonl"))
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        assert_eq!(
            records,
            vec![
                json!({"sequence": 0, "payload": {"text": "héllo"}}),
                json!({"sequence": 1, "payload": {"done": true}})
            ]
        );
        assert!(path.join("complete.json").exists());
    }

    struct Panic;
    impl Serialize for Panic {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            panic!("serializer entered");
        }
    }

    struct Invalid;
    impl Serialize for Invalid {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("invalid payload"))
        }
    }

    #[test]
    fn full_queue_rejects_before_serializing_and_preserves_incomplete_prefix() {
        let root = tempfile::tempdir().unwrap();
        let (mut queue, writer, path) = pair(root.path(), 1, 4);
        queue.record(&"é").unwrap(); // Exactly four UTF-8 JSON bytes.
        assert_eq!(
            queue.record(&Panic).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
        assert!(queue.record(&0).is_err());
        writer.run();
        assert!(!fs::read(path.join("records.jsonl")).unwrap().is_empty());
        assert!(!path.join("complete.json").exists());
    }

    #[test]
    fn oversized_payload_and_serializer_panic_close_admission() {
        let root = tempfile::tempdir().unwrap();
        let (mut queue, writer, path) = pair(root.path(), 1, 3);
        assert_eq!(
            queue.record(&"é").unwrap_err().kind(),
            io::ErrorKind::FileTooLarge
        );
        assert!(queue.record(&0).is_err());
        writer.run();
        assert!(!path.join("complete.json").exists());
        let (mut queue, writer, path) = pair(root.path(), 1, 64);
        assert_eq!(
            queue.record(&Invalid).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert!(queue.record(&0).is_err());
        writer.run();
        assert!(!path.join("complete.json").exists());
        let (mut queue, writer, path) = pair(root.path(), 1, 64);
        assert!(catch_unwind(AssertUnwindSafe(|| queue.record(&Panic))).is_err());
        assert!(queue.record(&0).is_err());
        writer.run();
        assert!(!path.join("complete.json").exists());
    }

    #[test]
    fn dropping_owner_or_unpolled_finish_does_not_seal() {
        let root = tempfile::tempdir().unwrap();
        let (mut queue, writer, path) = pair(root.path(), 1, 64);
        queue.record(&1).unwrap();
        drop(queue);
        writer.run();
        assert!(!path.join("complete.json").exists());
        let (queue, writer, path) = pair(root.path(), 1, 64);
        drop(queue.finish());
        writer.run();
        assert!(!path.join("complete.json").exists());
    }

    #[test]
    fn accepted_seal_survives_lost_acknowledgement() {
        let root = tempfile::tempdir().unwrap();
        let (queue, writer, path) = pair(root.path(), 1, 64);
        let mut completion = Box::pin(queue.finish());
        assert!(completion
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending());
        drop(completion);
        writer.run();
        assert!(path.join("complete.json").exists());
    }

    #[tokio::test]
    async fn storage_failure_or_lost_worker_cannot_report_success() {
        let root = tempfile::tempdir().unwrap();
        let file = CaptureFile::create(root.path(), 512).unwrap();
        let path = file.directory().to_owned();
        let (mut queue, writer) = CaptureQueue::new(file, 1, 64).unwrap();
        queue.record(&1).unwrap();
        let worker = tokio::task::spawn_blocking(|| writer.run());
        assert_eq!(
            queue.finish().await.unwrap_err().kind(),
            io::ErrorKind::FileTooLarge
        );
        worker.await.unwrap();
        assert!(!path.join("complete.json").exists());
        let (queue, writer, _) = pair(root.path(), 1, 64);
        drop(writer);
        assert_eq!(
            queue.finish().await.unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
        let (mut queue, writer, _) = pair(root.path(), 1, 64);
        drop(writer);
        assert_eq!(
            queue.record(&1).unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
    }

    #[test]
    fn invalid_limits_return_errors_without_panicking() {
        let root = tempfile::tempdir().unwrap();
        for (slots, bytes) in [(0, 1), (1, 0), (usize::MAX, 1), (1, usize::MAX)] {
            let file = CaptureFile::create(root.path(), 512).unwrap();
            assert!(
                matches!(CaptureQueue::new(file, slots, bytes), Err(e) if e.kind() == io::ErrorKind::InvalidInput)
            );
        }
    }
}
