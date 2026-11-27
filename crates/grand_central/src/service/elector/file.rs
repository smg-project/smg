//! The lease file (design 10): one JSON line at a path on a local
//! filesystem, rewritten whole at every renewal. The compare-and-swap is a
//! read under an exclusive advisory lock on the lock file beside it
//! (`<path>.lock`, held for one write), a write of `<path>.tmp` synced to
//! the disk, a rename over the path and a sync of the directory: a reader
//! without the lock (an operator, a candidate's `read`) sees the old record
//! or the new one, never a torn or empty file, and a host crash leaves one
//! of the two as well, so the term survives the host. A deleted or emptied
//! file is waited out by a replica that saw its record held, for that
//! record's duration, and taken above the highest term that replica saw. A
//! take of a file that holds no record leads one renew deadline later: a
//! holder still alive has fenced itself by then. A replica started after the
//! loss takes it at term one. A gateway's term floor above that term then
//! moves the lease above it, one hand-over a rising floor, while the floor
//! is believed (up to 65,536 above the term); past that the floors refuse
//! the controller until the gateways restart: keep the file.
//! Local filesystems only: the lock is the standard library's advisory lock,
//! which a network filesystem need not honour between hosts.

use std::{
    ffi::OsStr,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
};

use super::{ConditionalWriteOutcome, ElectError, LeaseBackend, LeaseRecord};

/// The lease at a path; see the module documentation.
#[derive(Debug)]
pub struct FileLease {
    path: PathBuf,
    lock: PathBuf,
    tmp: PathBuf,
}

impl FileLease {
    /// The lease at `path`, with `<path>.lock` and `<path>.tmp` beside it.
    /// Nothing is opened until the first call; the directory must exist.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let lock = with_suffix(&path, ".lock");
        let tmp = with_suffix(&path, ".tmp");
        Self { path, lock, tmp }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn io(&self, source: io::Error) -> ElectError {
        ElectError::Io {
            place: self.place(),
            source,
        }
    }

    fn read_record(&self) -> Result<Option<LeaseRecord>, ElectError> {
        let bytes = match fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(self.io(err)),
        };
        // An empty file (a `touch`ed path) is a lease nobody holds.
        if bytes.iter().all(u8::is_ascii_whitespace) {
            return Ok(None);
        }
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|err| ElectError::Malformed {
                place: self.place(),
                detail: err.to_string(),
            })
    }
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path
        .file_name()
        .map(OsStr::to_os_string)
        .unwrap_or_default();
    name.push(suffix);
    path.with_file_name(name)
}

impl LeaseBackend for FileLease {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.read_record()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        // Opened, never truncated: a truncate rewrites the lock file's inode
        // at every write of the lease, and on a disk under load that waits
        // for the filesystem where an open of a file that exists does not.
        let lock = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.lock)
            .map_err(|err| self.io(err))?;
        // Released when `lock` drops, on every path out of here.
        lock.lock().map_err(|err| self.io(err))?;
        if self.read_record()?.as_ref() != expected {
            return Ok(ConditionalWriteOutcome::Moved);
        }
        let mut line = serde_json::to_vec(record).map_err(|err| ElectError::Malformed {
            place: self.place(),
            detail: err.to_string(),
        })?;
        line.push(b'\n');
        let mut tmp = File::create(&self.tmp).map_err(|err| self.io(err))?;
        tmp.write_all(&line).map_err(|err| self.io(err))?;
        tmp.sync_all().map_err(|err| self.io(err))?;
        drop(tmp);
        fs::rename(&self.tmp, &self.path).map_err(|err| self.io(err))?;
        // The record is written once renamed. The directory sync makes the
        // rename durable across a crash (a crash before it surfaces the old
        // record, never an empty file); it is best effort, because the write
        // has landed whatever it answers, and a filesystem that refuses to
        // sync a directory would otherwise report every landed write as
        // failed.
        let _ = File::open(directory_of(&self.path)).and_then(|dir| dir.sync_all());
        Ok(ConditionalWriteOutcome::Done)
    }

    fn place(&self) -> String {
        self.path.display().to_string()
    }
}

/// The directory a lease path lives in: the working directory for a bare
/// file name.
fn directory_of(path: &Path) -> &Path {
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/elector/file/tests.rs"]
mod tests;
