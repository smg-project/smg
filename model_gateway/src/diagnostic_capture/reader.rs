//! Bounded, immutable readback of storage-complete diagnostic records.

use std::{
    fs::File,
    io::{self, Read},
    ops::Range,
    os::fd::OwnedFd,
    path::Path,
};

use rustix::fs::{open, openat, Mode, OFlags};
use serde::Deserialize;
use serde_json::value::RawValue;

use super::{Completion, MARKER_BYTES};

/// Caller-selected readback bounds, independent of any limits claimed by the artifact.
#[derive(Clone, Copy)]
pub struct CaptureReadLimits {
    /// Maximum bytes in `records.jsonl`; the marker has its own fixed 256-byte limit.
    pub file_bytes: usize,
    /// Maximum bytes in one stored record, including its envelope and final newline.
    pub record_bytes: usize,
    /// Maximum records, bounding the snapshot's payload-offset index.
    pub records: usize,
}

/// Verified storage bytes held privately in memory, unaffected by later filesystem changes.
///
/// Checksums detect inconsistency, not authenticity: a producer can replace both files.
/// Callers must still validate source revision, payload schemas and semantic completeness.
pub struct CaptureSnapshot {
    jsonl: String,
    payloads: Vec<Range<usize>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredRecord<'a> {
    sequence: u64,
    #[serde(borrow)]
    payload: &'a RawValue,
}

impl CaptureSnapshot {
    /// Read and validate a complete artifact on a blocking thread before exposing any payload.
    ///
    /// Opens the directory once and its two regular files relative to that descriptor, refusing
    /// symlinks at those final components. Ancestor paths remain the caller's responsibility.
    /// All limits must be positive. Retains at most `file_bytes` data bytes and `records`
    /// payload ranges, excluding allocator overhead and the bounded marker/parser state.
    /// This does not execute captured code or certify that inference succeeded.
    pub fn read(directory: &Path, limits: CaptureReadLimits) -> io::Result<Self> {
        if limits.file_bytes == 0 || limits.record_bytes == 0 || limits.records == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read limits must be positive",
            ));
        }
        let directory = open(
            directory,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )?;
        let marker: Completion = serde_json::from_slice(&read_regular(
            &directory,
            "complete.json",
            MARKER_BYTES as usize,
        )?)
        .map_err(io::Error::from)?;
        if marker.version != 1 {
            return Err(invalid("unsupported capture version"));
        }
        if marker.bytes > limits.file_bytes as u64 || marker.records > limits.records as u64 {
            return Err(too_large());
        }
        let bytes = read_regular(&directory, "records.jsonl", limits.file_bytes)?;
        if bytes.len() as u64 != marker.bytes
            || blake3::hash(&bytes).to_hex().as_str() != marker.blake3
        {
            return Err(invalid("capture size or checksum mismatch"));
        }
        // Each record needs at least a newline; reject impossible counts before index allocation.
        if marker.records > marker.bytes || (!bytes.is_empty() && !bytes.ends_with(b"\n")) {
            return Err(invalid("capture count or final newline is invalid"));
        }
        let jsonl = String::from_utf8(bytes).map_err(|_| invalid("capture is not UTF-8"))?;
        let mut payloads = Vec::new();
        payloads
            .try_reserve_exact(marker.records as usize)
            .map_err(io::Error::other)?;
        let mut offset = 0;
        while offset < jsonl.len() {
            if payloads.len() as u64 >= marker.records {
                return Err(invalid("capture contains extra records"));
            }
            let end = offset.saturating_add(limits.record_bytes).min(jsonl.len());
            let window = &jsonl.as_bytes()[offset..end];
            let mut records =
                serde_json::Deserializer::from_slice(window).into_iter::<StoredRecord<'_>>();
            // RawValue can contain pretty JSON: parse a bounded object, not a physical line.
            let record = records
                .next()
                .ok_or_else(|| invalid("missing capture record"))?
                .map_err(|error| boundary_error(error, window, end < jsonl.len()))?;
            let length = records.byte_offset();
            if length == limits.record_bytes {
                return Err(too_large());
            }
            if jsonl.as_bytes()[offset] != b'{'
                || jsonl.as_bytes().get(offset + length) != Some(&b'\n')
            {
                return Err(invalid("capture record delimiter is invalid"));
            }
            offset += length + 1;
            if record.sequence != payloads.len() as u64 {
                return Err(invalid("capture sequence is not contiguous"));
            }
            // RawValue borrows this exact immutable string; store byte ranges without copying JSON.
            let payload = record.payload.get();
            let start = payload.as_ptr() as usize - jsonl.as_ptr() as usize;
            payloads.push(start..start + payload.len());
        }
        if payloads.len() as u64 != marker.records {
            return Err(invalid("capture is missing records"));
        }
        Ok(Self { jsonl, payloads })
    }

    /// Validated JSON payloads in storage sequence order, borrowing the verified snapshot.
    ///
    /// Payload contents are still untrusted; deserialize only the supported replay schema.
    pub fn payloads(&self) -> impl ExactSizeIterator<Item = &str> {
        self.payloads.iter().map(|range| &self.jsonl[range.clone()])
    }
}

fn read_regular(directory: &OwnedFd, name: &str, limit: usize) -> io::Result<Vec<u8>> {
    // NONBLOCK keeps a FIFO open from waiting for a writer; metadata rejects nonregular files.
    let mut file = File::from(openat(
        directory,
        name,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )?);
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(invalid("capture entry is not a regular file"));
    }
    let length = usize::try_from(metadata.len()).map_err(|_| too_large())?;
    if length > limit {
        return Err(too_large());
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(io::Error::other)?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)?;
    if file.read(&mut [0])? != 0 {
        return Err(invalid("capture entry changed size while reading"));
    }
    Ok(bytes)
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn boundary_error(error: serde_json::Error, window: &[u8], truncated: bool) -> io::Error {
    // RawValue reports a numeric prefix cut after e.g. '-' or '1e+' as Syntax, not EOF.
    let number_cut = error.is_syntax()
        && matches!(window.last(), Some(b'-' | b'+' | b'.' | b'e' | b'E'))
        && error.line() == window.iter().filter(|&&byte| byte == b'\n').count() + 1
        && error.column()
            == window
                .rsplit(|&byte| byte == b'\n')
                .next()
                .unwrap_or_default()
                .len();
    if truncated && (error.is_eof() || number_cut) {
        too_large()
    } else {
        io::Error::from(error)
    }
}

fn too_large() -> io::Error {
    io::Error::new(io::ErrorKind::FileTooLarge, "capture exceeds read limits")
}

#[cfg(test)]
mod tests {
    use std::{fs, os::unix::fs::symlink, path::PathBuf};

    use serde_json::{json, Value};

    use super::*;
    use crate::diagnostic_capture::CaptureFile;

    fn limits() -> CaptureReadLimits {
        CaptureReadLimits {
            file_bytes: 4096,
            record_bytes: 1024,
            records: 16,
        }
    }

    fn sealed(root: &Path) -> PathBuf {
        let mut file = CaptureFile::create(root, 4096).unwrap();
        file.record(
            &json!({"text": "héllo\n", "tokens": [3, 4], "numbers": [-0.25, 1.25e30, 1.25e-30]}),
        )
        .unwrap();
        file.record(&json!({"finish": true})).unwrap();
        file.finish().unwrap()
    }

    fn replace(directory: &Path, data: &[u8], records: u64) {
        fs::write(directory.join("records.jsonl"), data).unwrap();
        let marker = json!({"version": 1, "records": records, "bytes": data.len(), "blake3": blake3::hash(data).to_hex().to_string()});
        fs::write(
            directory.join("complete.json"),
            serde_json::to_vec(&marker).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn snapshot_preserves_payload_order_and_survives_later_file_changes() {
        let root = tempfile::tempdir().unwrap();
        let expected = [
            json!(null),
            json!({"text": "héllo\n", "tokens": [3, 4]}),
            json!([true, 1.25, "é"]),
        ];
        let mut file = CaptureFile::create(root.path(), 4096).unwrap();
        for value in &expected {
            let raw = RawValue::from_string(serde_json::to_string_pretty(value).unwrap()).unwrap();
            file.record(&raw).unwrap();
        }
        let directory = file.finish().unwrap();
        let snapshot = CaptureSnapshot::read(&directory, limits()).unwrap();
        fs::write(directory.join("records.jsonl"), "changed").unwrap();
        fs::remove_file(directory.join("complete.json")).unwrap();
        let values: Vec<Value> = snapshot
            .payloads()
            .map(|p| serde_json::from_str(p).unwrap())
            .collect();
        assert_eq!(values, expected);
        assert!(CaptureSnapshot::read(&directory, limits()).is_err());
    }

    #[test]
    fn empty_storage_is_valid_but_abandoned_storage_is_not() {
        let root = tempfile::tempdir().unwrap();
        let file = CaptureFile::create(root.path(), 4096).unwrap();
        assert!(CaptureSnapshot::read(file.directory(), limits()).is_err());
        let directory = file.finish().unwrap();
        assert_eq!(
            CaptureSnapshot::read(&directory, limits())
                .unwrap()
                .payloads()
                .len(),
            0
        );
    }

    #[test]
    fn changed_truncated_and_appended_data_fail_the_marker_check() {
        let root = tempfile::tempdir().unwrap();
        let directory = sealed(root.path());
        let original = fs::read(directory.join("records.jsonl")).unwrap();
        let mut changed = original.clone();
        changed[0] = b'[';
        let mut appended = original.clone();
        appended.push(b'\n');
        for bytes in [
            changed.as_slice(),
            &original[..original.len() - 1],
            &appended,
        ] {
            fs::write(directory.join("records.jsonl"), bytes).unwrap();
            assert!(CaptureSnapshot::read(&directory, limits()).is_err());
        }
    }

    #[test]
    fn matching_checksums_do_not_hide_invalid_envelopes_or_counts() {
        let root = tempfile::tempdir().unwrap();
        let directory = sealed(root.path());
        for (data, records) in [
            ("{\"sequence\":1,\"payload\":null}\n", 1),
            (
                "{\"sequence\":0,\"payload\":null}\n{\"sequence\":0,\"payload\":null}\n",
                2,
            ),
            ("{\"sequence\":0,\"payload\":null,\"extra\":1}\n", 1),
            ("{\"sequence\":0,\"payload\":null,\"payload\":true}\n", 1),
            ("{\"sequence\":0,\"payload\":null}", 1),
            ("{\"sequence\":0,\"payload\":}\n", 1),
            ("\n", 1),
            ("{\"sequence\":0,\"payload\":null}\n", 0),
            ("{\"sequence\":0,\"payload\":null}\n", 2),
        ] {
            replace(&directory, data.as_bytes(), records);
            assert!(
                CaptureSnapshot::read(&directory, limits()).is_err(),
                "{data}"
            );
        }
        replace(&directory, b"\xff\n", 1);
        assert!(CaptureSnapshot::read(&directory, limits()).is_err());
        let malformed = b"{\"sequence\":0,\"payload\":x-}\n";
        replace(&directory, malformed, 1);
        let bounded = CaptureReadLimits {
            record_bytes: malformed.len() - 2,
            ..limits()
        };
        assert_eq!(
            CaptureSnapshot::read(&directory, bounded)
                .err()
                .unwrap()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn strict_marker_version_shape_and_size_are_required() {
        let root = tempfile::tempdir().unwrap();
        let directory = sealed(root.path());
        let original: Value =
            serde_json::from_slice(&fs::read(directory.join("complete.json")).unwrap()).unwrap();
        for (key, value) in [
            ("version", json!(2)),
            ("extra", json!(1)),
            ("blake3", json!("bad")),
            ("bytes", json!(0)),
        ] {
            let mut marker = original.clone();
            marker[key] = value;
            fs::write(
                directory.join("complete.json"),
                serde_json::to_vec(&marker).unwrap(),
            )
            .unwrap();
            assert!(CaptureSnapshot::read(&directory, limits()).is_err());
        }
        fs::write(
            directory.join("complete.json"),
            vec![b' '; MARKER_BYTES as usize + 1],
        )
        .unwrap();
        assert!(CaptureSnapshot::read(&directory, limits()).is_err());
    }

    #[test]
    fn byte_line_and_count_limits_include_the_exact_boundary() {
        let root = tempfile::tempdir().unwrap();
        let directory = sealed(root.path());
        let data = fs::read_to_string(directory.join("records.jsonl")).unwrap();
        let exact = CaptureReadLimits {
            file_bytes: data.len(),
            record_bytes: data.split_inclusive('\n').map(str::len).max().unwrap(),
            records: 2,
        };
        assert!(CaptureSnapshot::read(&directory, exact).is_ok());
        use io::ErrorKind::{FileTooLarge, InvalidInput};
        for record_bytes in 1..exact.record_bytes {
            let error = CaptureSnapshot::read(
                &directory,
                CaptureReadLimits {
                    record_bytes,
                    ..exact
                },
            )
            .err()
            .unwrap();
            assert_eq!(
                error.kind(),
                FileTooLarge,
                "record_bytes={record_bytes}: {error:?}"
            );
        }
        let (bytes, record) = (exact.file_bytes, exact.record_bytes);
        for (file_bytes, record_bytes, records, expected) in [
            (bytes - 1, record, 2, FileTooLarge),
            (bytes, record - 1, 2, FileTooLarge),
            (bytes, record, 1, FileTooLarge),
            (0, record, 2, InvalidInput),
            (bytes, 0, 2, InvalidInput),
            (bytes, record, 0, InvalidInput),
        ] {
            let limits = CaptureReadLimits {
                file_bytes,
                record_bytes,
                records,
            };
            assert_eq!(
                CaptureSnapshot::read(&directory, limits)
                    .err()
                    .unwrap()
                    .kind(),
                expected
            );
        }
    }

    #[test]
    fn symlinks_directories_and_fifos_are_not_capture_files() {
        let root = tempfile::tempdir().unwrap();
        let directory = sealed(root.path());
        let linked = root.path().join("linked");
        symlink(&directory, &linked).unwrap();
        assert!(CaptureSnapshot::read(&linked, limits()).is_err());
        for name in ["records.jsonl", "complete.json"] {
            let directory = sealed(root.path());
            let path = directory.join(name);
            let saved = directory.join("saved");
            fs::rename(&path, &saved).unwrap();
            symlink(&saved, &path).unwrap();
            assert!(CaptureSnapshot::read(&directory, limits()).is_err());
            fs::remove_file(&path).unwrap();
            fs::create_dir(&path).unwrap();
            assert!(CaptureSnapshot::read(&directory, limits()).is_err());
            fs::remove_dir(&path).unwrap();
            let dir = File::open(&directory).unwrap();
            rustix::fs::mkfifoat(&dir, name, Mode::RUSR | Mode::WUSR).unwrap();
            assert!(CaptureSnapshot::read(&directory, limits()).is_err());
        }
    }
}
