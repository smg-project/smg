//! The lease file (design 10): one JSON line at a path on a local
//! filesystem, rewritten whole at every renewal. The compare-and-swap is a
//! read under an exclusive advisory lock on the lock file beside it
//! (`<path>.lock`, held for one write), a write of `<path>.tmp` synced to
//! the disk, a rename over the path and a sync of the directory: a reader
//! without the lock (an operator, a candidate's `read`) sees the old record
//! or the new one, never a torn or empty file, and a host crash leaves one
//! of the two as well, so the term survives the host. A deleted or emptied
//! file restarts the term at one; the gateways' term floors then refuse the
//! controller until they restart, so the file is not something to clean up.
//! Local filesystems only: the lock is the standard library's advisory lock,
//! which a network filesystem need not honour between hosts.

use std::{
    ffi::OsStr,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};

use super::{ElectError, LeaseBackend, LeaseRecord, Written};

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
    ) -> Result<Written, ElectError> {
        let lock = File::create(&self.lock).map_err(|err| self.io(err))?;
        // Released when `lock` drops, on every path out of here.
        lock.lock().map_err(|err| self.io(err))?;
        if self.read_record()?.as_ref() != expected {
            return Ok(Written::Moved);
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
        // The rename is durable once the directory is: a crash before this
        // sync surfaces the old record, never an empty file.
        if let Some(dir) = self.path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            File::open(dir)
                .and_then(|dir| dir.sync_all())
                .map_err(|err| self.io(err))?;
        }
        Ok(Written::Done)
    }

    fn place(&self) -> String {
        self.path.display().to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc, Barrier,
        },
        thread,
        time::{Duration, Instant},
    };

    use super::*;
    use crate::elector::{ElectionSettings, Elector, Tick};

    const S: Duration = Duration::from_secs(1);

    fn record(holder: &str, term: u64, renewed_at_micros: u64) -> LeaseRecord {
        LeaseRecord {
            holder: holder.to_string(),
            term,
            acquired_at_micros: 1,
            renewed_at_micros,
            duration_ms: 3_000,
        }
    }

    fn candidate(path: &Path, identity: &str) -> Elector {
        Elector::new(
            Box::new(FileLease::new(path)),
            identity,
            ElectionSettings::default(),
        )
        .expect("valid settings")
    }

    #[test]
    fn the_record_is_one_json_line_and_the_side_files_sit_beside_it() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("policy.lease");
        let mut lease = FileLease::new(&path);
        assert_eq!(lease.path(), path);
        assert_eq!(lease.read().expect("read"), None, "no file yet");
        let first = record("a@host:1", 1, 5);
        assert_eq!(lease.write(None, &first).expect("write"), Written::Done);
        let text = fs::read_to_string(&path).expect("the file");
        assert_eq!(
            text,
            "{\"holder\":\"a@host:1\",\"term\":1,\"acquired_at_micros\":1,\"renewed_at_micros\":5,\"duration_ms\":3000}\n"
        );
        assert!(
            dir.path().join("policy.lease.lock").exists(),
            "the lock file"
        );
        assert!(
            !dir.path().join("policy.lease.tmp").exists(),
            "renamed over the path"
        );
        assert_eq!(lease.read().expect("read"), Some(first));
        assert_eq!(lease.place(), path.display().to_string());
    }

    #[test]
    fn an_empty_file_is_no_lease_and_a_malformed_one_is_an_error() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("lease");
        fs::write(&path, "\n").expect("touch");
        let mut a = candidate(&path, "a@host:1");
        assert_eq!(
            a.tick(Instant::now()).expect("tick"),
            Tick::Acquired { term: 1 }
        );

        fs::write(&path, "{\"holder\":").expect("a torn line by hand");
        let mut b = candidate(&path, "b@host:2");
        let err = b.tick(Instant::now()).expect_err("malformed");
        assert!(
            matches!(&err, ElectError::Malformed { place, .. } if place == &path.display().to_string()),
            "{err}"
        );
        assert!(!b.is_holder(), "nothing taken");
        assert_eq!(
            fs::read_to_string(&path).expect("the file"),
            "{\"holder\":",
            "nothing written"
        );
        assert!(err.to_string().contains("cannot read"));
    }

    #[test]
    fn a_missing_directory_is_an_io_error_naming_the_path() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("absent").join("lease");
        let mut a = candidate(&path, "a@host:1");
        let err = a.tick(Instant::now()).expect_err("no directory");
        assert!(
            matches!(&err, ElectError::Io { place, .. } if place == &path.display().to_string()),
            "{err}"
        );
        assert!(!a.is_holder());
    }

    #[test]
    fn two_candidates_over_one_file_elect_one_and_hand_over_on_expiry() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("lease");
        let mut a = candidate(&path, "a@host:1");
        let mut b = candidate(&path, "b@host:2");
        let t0 = Instant::now();
        assert_eq!(a.tick(t0).expect("tick"), Tick::Acquired { term: 1 });
        assert_eq!(
            b.tick(t0).expect("tick"),
            Tick::Following {
                holder: "a@host:1".to_string(),
                term: 1
            }
        );
        for i in 1..=3u32 {
            assert_eq!(
                a.tick(t0 + i * S).expect("renew"),
                Tick::Renewed { term: 1 }
            );
            assert!(matches!(
                b.tick(t0 + i * S).expect("follow"),
                Tick::Following { .. }
            ));
        }
        // a stalls; b's sight of the last renewal is three seconds old.
        assert_eq!(
            b.tick(t0 + 6 * S).expect("take"),
            Tick::Acquired { term: 2 }
        );
        assert_eq!(a.fence().leader_term(t0 + 6 * S), None);
        assert_eq!(b.fence().leader_term(t0 + 6 * S), Some(2));
        assert_eq!(
            a.tick(t0 + 6 * S).expect("the stalled renewal"),
            Tick::Lost {
                term: 1,
                holder: "b@host:2".to_string()
            }
        );
        let stored: LeaseRecord =
            serde_json::from_str(&fs::read_to_string(&path).expect("the file")).expect("a record");
        assert_eq!(stored.holder, "b@host:2");
        assert_eq!(stored.term, 2);
    }

    #[test]
    fn a_rewrite_is_atomic_for_readers_without_the_lock() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("lease");
        let stop = Arc::new(AtomicBool::new(false));
        let readers: Vec<_> = (0..3)
            .map(|_| {
                let path = path.clone();
                let stop = Arc::clone(&stop);
                thread::spawn(move || {
                    let mut lease = FileLease::new(&path);
                    let mut seen = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        if let Some(record) = lease.read().expect("a whole record or none") {
                            assert_eq!(record.holder, "w@host:1");
                            seen += 1;
                        }
                    }
                    seen
                })
            })
            .collect();
        let mut writer = FileLease::new(&path);
        let mut previous: Option<LeaseRecord> = None;
        for i in 1..=500u64 {
            let next = record("w@host:1", 1, i);
            assert_eq!(
                writer.write(previous.as_ref(), &next).expect("write"),
                Written::Done
            );
            previous = Some(next);
        }
        stop.store(true, Ordering::Relaxed);
        let reads: u64 = readers
            .into_iter()
            .map(|r| r.join().expect("a reader"))
            .sum();
        assert!(reads > 0, "the readers read while the writer wrote");
        assert!(!dir.path().join("lease.tmp").exists());
    }

    #[test]
    fn concurrent_writers_serialize_on_the_lock_and_one_wins_each_round() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("lease");
        let rounds = 40u64;
        // Every writer is spawned before any is joined, and all four start
        // at the barrier: the rounds are contended, never run one after
        // another.
        let start = Arc::new(Barrier::new(4));
        let writers: Vec<_> = (0..4u64)
            .map(|w| {
                let path = path.clone();
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    let mut lease = FileLease::new(&path);
                    let mut won = 0u64;
                    start.wait();
                    for round in 1..=rounds {
                        // Every writer reads the round's record and writes
                        // the next one over it: the lock makes one of them
                        // first and the others see Moved.
                        loop {
                            let current = lease.read().expect("read");
                            let term = current.as_ref().map_or(0, |r| r.term);
                            if term >= round {
                                break;
                            }
                            let next = record(&format!("w{w}"), term + 1, round);
                            if lease.write(current.as_ref(), &next).expect("write") == Written::Done
                            {
                                won += 1;
                                break;
                            }
                        }
                    }
                    won
                })
            })
            .collect();
        let winners: Vec<u64> = writers
            .into_iter()
            .map(|h| h.join().expect("a writer"))
            .collect();
        assert_eq!(winners.iter().sum::<u64>(), rounds, "one write per round");
        let last = FileLease::new(&path)
            .read()
            .expect("read")
            .expect("the record");
        assert_eq!(last.term, rounds);
    }
}
