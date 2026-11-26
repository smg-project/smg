use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Barrier,
    },
    thread,
    time::{Duration, Instant},
};

use super::*;
use crate::service::elector::{ElectionSettings, Elector, TickOutcome};

const S: Duration = Duration::from_secs(1);

/// How far ahead of the clock a case that steps through a wait starts its
/// instants: the elector dates a first sight at the later of the caller's
/// instant and its own clock, and an instant the real clock overtakes
/// moves the wait.
const AHEAD: Duration = S.saturating_mul(3_600);

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
fn a_bare_file_name_lives_in_the_working_directory() {
    assert_eq!(directory_of(Path::new("policy.lease")), Path::new("."));
    assert_eq!(
        directory_of(Path::new("/var/lib/smg/policy.lease")),
        Path::new("/var/lib/smg")
    );
    assert_eq!(directory_of(Path::new("state/lease")), Path::new("state"));
}

#[test]
fn the_record_is_one_json_line_and_the_side_files_sit_beside_it() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("policy.lease");
    let mut lease = FileLease::new(&path);
    assert_eq!(lease.path(), path);
    assert_eq!(lease.read().expect("read"), None, "no file yet");
    let first = record("a@host:1", 1, 5);
    assert_eq!(
        lease.write(None, &first).expect("write"),
        ConditionalWriteOutcome::Done
    );
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

/// The lock file is opened at every write and never rewritten: a write
/// of the lease leaves its bytes as they stand, whether the write lands
/// or finds the record moved. A truncate would change the file at every
/// renewal, and on a disk under load that waits for the filesystem.
#[test]
fn a_write_of_the_lease_leaves_the_lock_file_as_it_stands() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("policy.lease");
    let lock = dir.path().join("policy.lease.lock");
    let mut lease = FileLease::new(&path);
    let first = record("a@host:1", 1, 5);
    assert_eq!(
        lease.write(None, &first).expect("write"),
        ConditionalWriteOutcome::Done
    );
    assert_eq!(fs::read(&lock).expect("the lock file"), b"", "made empty");
    // What stands in the lock file is no part of the lease: the mark
    // shows a write that rewrites it.
    fs::write(&lock, b"kept").expect("the mark");
    let second = record("a@host:1", 1, 6);
    assert_eq!(
        lease.write(Some(&first), &second).expect("write"),
        ConditionalWriteOutcome::Done
    );
    assert_eq!(fs::read(&lock).expect("the lock file"), b"kept");
    let theirs = record("b@host:2", 2, 7);
    assert_eq!(
        lease.write(Some(&first), &theirs).expect("write"),
        ConditionalWriteOutcome::Moved
    );
    assert_eq!(fs::read(&lock).expect("the lock file"), b"kept");
    assert_eq!(lease.read().expect("read"), Some(second));
}

#[test]
fn an_empty_file_is_no_lease_and_a_malformed_one_is_an_error() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("lease");
    fs::write(&path, "\n").expect("touch");
    let mut a = candidate(&path, "a@host:1");
    assert_eq!(
        a.tick(Instant::now()).expect("tick"),
        TickOutcome::Acquired { term: 1 }
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
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("tick"), TickOutcome::Acquired { term: 1 });
    assert_eq!(
        b.tick(t0).expect("tick"),
        TickOutcome::Following {
            holder: "a@host:1".to_string(),
            term: 1
        }
    );
    for i in 1..=3u32 {
        assert_eq!(
            a.tick(t0 + i * S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert!(matches!(
            b.tick(t0 + i * S).expect("follow"),
            TickOutcome::Following { .. }
        ));
    }
    // a stalls; b's sight of the last renewal is three seconds old.
    assert_eq!(
        b.tick(t0 + 6 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(a.fence().leader_term(t0 + 6 * S), None);
    assert_eq!(b.fence().leader_term(t0 + 6 * S), Some(2));
    assert_eq!(
        a.tick(t0 + 6 * S).expect("the stalled renewal"),
        TickOutcome::Lost {
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
            ConditionalWriteOutcome::Done
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
                        if lease.write(current.as_ref(), &next).expect("write")
                            == ConditionalWriteOutcome::Done
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
