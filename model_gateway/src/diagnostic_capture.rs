//! Bounded Unix file storage for diagnostic records. No serving path invokes it automatically.
//!
//! Run `CaptureFile` on a blocking writer, not an inference task. `CaptureQueue` supplies a
//! bounded single-producer mailbox. `CapturePool` schedules writers under shared run limits.
//! Sampling, source metadata, capture hooks and retention across runs remain the caller's job.
//! A completion marker attests only that the accepted records were stored: it cannot prove
//! that every required engine/parser/adapter boundary was captured or that inference passed.
//!
//! `CaptureSnapshot` validates bounded storage bytes before exposing immutable payloads for
//! replay. A matching checksum is not authentication or evidence of semantic completeness.
//!
//! Dropped or failed writers leave an incomplete directory for inspection. Stream and marker
//! contents are synced before the marker is atomically published. A crash may lose the final
//! directory entry; consumers must treat a missing marker as incomplete.

use std::{
    fs::{self, DirBuilder, File, OpenOptions},
    io::{self, BufWriter, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

mod reader;
pub use reader::{CaptureReadLimits, CaptureSnapshot};

mod admission;
mod queue;
pub use admission::{CaptureLimits, CapturePool, CaptureSession};
pub use queue::{CaptureQueue, CaptureWriter};

pub use crate::routers::grpc::regular::symphony::FormatRecipe;

// The temporary and published marker names can both appear in an artifact inventory.
const MARKER_BYTES: u64 = 256;
const RESERVED_BYTES: u64 = 2 * MARKER_BYTES;
const BUFFER_BYTES: usize = 8192;

/// An explicitly created diagnostic stream with a permanent failure state and a byte budget.
pub struct CaptureFile {
    root: PathBuf,
    directory: PathBuf,
    file: BufWriter<File>,
    stream_limit: u64,
    bytes: u64,
    records: u64,
    digest: blake3::Hasher,
    failed: bool,
}

#[derive(Serialize)]
struct Record<'a, T: ?Sized> {
    sequence: u64,
    payload: &'a T,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Completion {
    version: u32,
    records: u64,
    bytes: u64,
    blake3: String,
}

impl CaptureFile {
    /// Create a fresh private directory under an existing, operator-controlled root.
    ///
    /// Names are generated internally, never derived from a request header or model output.
    /// `max_bytes` bounds logical file contents, including two reserved 256-byte marker names;
    /// filesystem metadata/block overhead is not included. At least 512 bytes are required.
    pub fn create(root: &Path, max_bytes: u64) -> io::Result<Self> {
        let stream_limit = max_bytes.checked_sub(RESERVED_BYTES).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "capture budget must be at least 512 bytes",
            )
        })?;
        let root = root.canonicalize()?;
        let directory = root.join(uuid::Uuid::now_v7().to_string());
        DirBuilder::new().mode(0o700).create(&directory)?;
        let file = private_file(&directory.join("records.jsonl"))?;
        Ok(Self {
            root,
            directory,
            file: BufWriter::with_capacity(BUFFER_BYTES, file),
            stream_limit,
            bytes: 0,
            records: 0,
            digest: blake3::Hasher::new(),
            failed: false,
        })
    }

    /// Location of this stream, including partial evidence after a failure or abandoned writer.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Append one ordered JSON record without allocating a serialization-sized intermediate.
    ///
    /// Serialization errors or panics, byte-limit and IO errors invalidate the stream. A partial
    /// final line is retained, but neither later writes nor `finish` can turn it into a pass.
    pub fn record<T: Serialize + ?Sized>(&mut self, payload: &T) -> io::Result<()> {
        if self.failed {
            return Err(io::Error::other("capture is already incomplete"));
        }
        // Stay incomplete even if the caller catches a serializer panic after partial output.
        self.failed = true;
        let result = (|| {
            let next = self
                .records
                .checked_add(1)
                .ok_or_else(|| io::Error::other("capture sequence exhausted"))?;
            let mut writer = Limited {
                inner: &mut self.file,
                bytes: &mut self.bytes,
                limit: self.stream_limit,
                digest: &mut self.digest,
            };
            serde_json::to_writer(
                &mut writer,
                &Record {
                    sequence: self.records,
                    payload,
                },
            )
            .map_err(io::Error::from)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
            self.records = next;
            Ok(())
        })();
        self.failed = result.is_err();
        result
    }

    /// Sync accepted records and atomically publish `complete.json`, without overwriting it.
    ///
    /// No fallible cleanup follows publication. The hidden temporary marker remains a second
    /// name for the same inode; both names fit the reserved budget. Only `complete.json` seals
    /// storage. Consumers still need the caller's source/coverage checks before accepting a run.
    pub fn finish(mut self) -> io::Result<PathBuf> {
        if self.failed {
            return Err(io::Error::other("capture is incomplete"));
        }
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        let completion = Completion {
            version: 1,
            records: self.records,
            bytes: self.bytes,
            blake3: self.digest.finalize().to_hex().to_string(),
        };
        let data = serde_json::to_vec(&completion).map_err(io::Error::other)?;
        if data.len() as u64 > MARKER_BYTES {
            return Err(io::Error::other("capture marker exceeds reserved bytes"));
        }
        let temporary = self.directory.join(".complete.json");
        let mut marker = private_file(&temporary)?;
        marker.write_all(&data)?;
        marker.flush()?;
        marker.sync_all()?;
        File::open(&self.directory)?.sync_all()?;
        File::open(&self.root)?.sync_all()?;
        // Same-directory hard linking is atomic and fails if the destination already exists.
        // Data and marker bytes are durable first; an interrupted publication stays incomplete.
        fs::hard_link(&temporary, self.directory.join("complete.json"))?;
        Ok(self.directory)
    }
}

fn private_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

struct Limited<'a, W> {
    inner: &'a mut W,
    bytes: &'a mut u64,
    limit: u64,
    digest: &'a mut blake3::Hasher,
}

impl<W: Write> Write for Limited<'_, W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if buffer.len() as u64 > self.limit.saturating_sub(*self.bytes) {
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                "capture byte budget exhausted",
            ));
        }
        let written = self.inner.write(buffer)?;
        self.digest.update(&buffer[..written]);
        *self.bytes += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use serde_json::{json, Value};

    use super::*;

    #[test]
    fn completion_matches_exact_ordered_bytes_and_private_permissions() {
        let root = tempfile::tempdir().unwrap();
        let mut capture = CaptureFile::create(root.path(), 4096).unwrap();
        capture
            .record(&json!({"text": "héllo\n", "tokens": [1, 2]}))
            .unwrap();
        capture.record(&json!({"finish": "stop"})).unwrap();
        assert!(!capture.directory().join("complete.json").exists());
        let data_path = capture.directory().join("records.jsonl");
        assert!(!fs::read(&data_path).unwrap().is_empty());
        let directory = capture.finish().unwrap();
        let data = fs::read(data_path).unwrap();
        let records: Vec<Value> = data
            .split_inclusive(|b| *b == b'\n')
            .map(|s| serde_json::from_slice(s).unwrap())
            .collect();
        assert_eq!(
            records,
            vec![
                json!({"sequence": 0, "payload": {"text": "héllo\n", "tokens": [1, 2]}}),
                json!({"sequence": 1, "payload": {"finish": "stop"}})
            ]
        );
        let marker: Value =
            serde_json::from_slice(&fs::read(directory.join("complete.json")).unwrap()).unwrap();
        assert_eq!(
            marker,
            json!({"version": 1, "records": 2, "bytes": data.len(), "blake3": blake3::hash(&data).to_hex().to_string()})
        );
        for (path, mode) in [
            (directory.clone(), 0o700),
            (directory.join("records.jsonl"), 0o600),
            (directory.join("complete.json"), 0o600),
        ] {
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                mode
            );
        }
    }

    #[test]
    fn budget_includes_the_newline_and_failure_is_permanent() {
        let root = tempfile::tempdir().unwrap();
        let line = b"{\"sequence\":0,\"payload\":\"x\"}\n";
        let mut exact =
            CaptureFile::create(root.path(), RESERVED_BYTES + line.len() as u64).unwrap();
        exact.record("x").unwrap();
        assert_eq!(
            fs::read(exact.finish().unwrap().join("records.jsonl")).unwrap(),
            line
        );
        let mut short =
            CaptureFile::create(root.path(), RESERVED_BYTES + line.len() as u64 - 1).unwrap();
        let directory = short.directory().to_owned();
        assert_eq!(
            short.record("x").unwrap_err().kind(),
            io::ErrorKind::FileTooLarge
        );
        assert!(short.record(&0).is_err());
        assert!(short.finish().is_err());
        assert!(!directory.join("complete.json").exists());
        assert!(fs::metadata(directory.join("records.jsonl")).unwrap().len() < line.len() as u64);
        assert!(CaptureFile::create(root.path(), RESERVED_BYTES - 1).is_err());
    }

    #[test]
    fn oversized_payload_and_serialization_error_never_publish_completion() {
        struct Reject;
        impl Serialize for Reject {
            fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("injected serialization failure"))
            }
        }
        let root = tempfile::tempdir().unwrap();
        let mut huge = CaptureFile::create(root.path(), 1024).unwrap();
        let directory = huge.directory().to_owned();
        assert!(huge.record(&"x".repeat(100_000)).is_err());
        assert!(huge.finish().is_err());
        assert!(fs::metadata(directory.join("records.jsonl")).unwrap().len() <= 512);
        assert!(!directory.join("complete.json").exists());
        let mut rejected = CaptureFile::create(root.path(), 4096).unwrap();
        let directory = rejected.directory().to_owned();
        assert_eq!(
            rejected.record(&Reject).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert!(rejected.record(&1).is_err());
        assert!(rejected.finish().is_err());
        assert!(!directory.join("complete.json").exists());
    }

    #[test]
    fn abandoned_writer_and_marker_failure_leave_incomplete_evidence() {
        let root = tempfile::tempdir().unwrap();
        let mut abandoned = CaptureFile::create(root.path(), 4096).unwrap();
        abandoned.record(&1).unwrap();
        let directory = abandoned.directory().to_owned();
        drop(abandoned);
        assert!(!directory.join("complete.json").exists());
        let capture = CaptureFile::create(root.path(), 4096).unwrap();
        let directory = capture.directory().to_owned();
        fs::create_dir(directory.join(".complete.json")).unwrap();
        assert!(capture.finish().is_err());
        assert!(!directory.join("complete.json").exists());
    }

    #[test]
    fn completion_never_overwrites_an_existing_marker() {
        let root = tempfile::tempdir().unwrap();
        let capture = CaptureFile::create(root.path(), 4096).unwrap();
        let marker = capture.directory().join("complete.json");
        fs::write(&marker, b"existing").unwrap();
        assert!(capture.finish().is_err());
        assert_eq!(fs::read(marker).unwrap(), b"existing");
    }

    #[test]
    fn caught_serialization_panic_cannot_seal_partial_records() {
        struct Panics;
        impl Serialize for Panics {
            fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                panic!("injected serializer panic")
            }
        }
        let root = tempfile::tempdir().unwrap();
        let mut capture = CaptureFile::create(root.path(), 4096).unwrap();
        let directory = capture.directory().to_owned();
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| capture.record(&Panics)));
        assert!(result.is_err());
        assert!(capture.record(&1).is_err());
        assert!(capture.finish().is_err());
        assert!(!directory.join("complete.json").exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn write_and_sync_failures_never_publish_completion() {
        let root = tempfile::tempdir().unwrap();
        for (device, size) in [
            ("/dev/full", 1),
            ("/dev/full", BUFFER_BYTES * 2),
            ("/dev/null", 1),
        ] {
            let mut capture = CaptureFile::create(root.path(), 32768).unwrap();
            let directory = capture.directory().to_owned();
            capture.file = BufWriter::with_capacity(
                BUFFER_BYTES,
                OpenOptions::new().write(true).open(device).unwrap(),
            );
            let result = capture.record(&"x".repeat(size));
            assert_eq!(result.is_err(), device == "/dev/full");
            if device == "/dev/full" {
                assert_eq!(result.unwrap_err().kind(), io::ErrorKind::StorageFull);
                assert!(capture.record(&1).is_err());
            }
            assert!(capture.finish().is_err());
            assert!(!directory.join("complete.json").exists());
        }
    }

    #[test]
    fn short_writes_count_only_bytes_actually_written() {
        struct Short(Vec<u8>);
        impl Write for Short {
            fn write(&mut self, data: &[u8]) -> io::Result<usize> {
                let n = data.len().min(2);
                self.0.extend_from_slice(&data[..n]);
                Ok(n)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut target = Short(Vec::new());
        let mut bytes = 0;
        let mut digest = blake3::Hasher::new();
        let mut limited = Limited {
            inner: &mut target,
            bytes: &mut bytes,
            limit: 8,
            digest: &mut digest,
        };
        limited.write_all(b"abcdefgh").unwrap();
        assert!(limited.write_all(b"i").is_err());
        assert_eq!(bytes, 8);
        assert_eq!(target.0, b"abcdefgh");
        assert_eq!(digest.finalize(), blake3::hash(b"abcdefgh"));
    }
}
