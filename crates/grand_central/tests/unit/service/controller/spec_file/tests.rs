use std::{
    cell::{Cell, RefCell},
    process::Command,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::Instant,
};

use tempfile::TempDir;
use tokio::task::JoinSet;

use super::*;
use crate::{
    document::{
        operator::test_support::Validate,
        spec::{AppliedSpec, Digest, PolicySpec, SpecError},
    },
    service::{
        controller::generations::tests::{
            bound_to, election, election_under, faulting, long_text_under_aliases, padded,
            with_line, LEAST_LOAD, RANDOM,
        },
        elector::{Elector, Fence, TickOutcome},
    },
    stages::StageRegistry,
    store::MemoryStore,
};

mod polls;
mod read_whole;
mod shared_lease;
mod start;
mod watch_loop;

impl SpecFile {
    /// [`SpecFile::poll_looking`] with the file system's own look: a case's
    /// step.
    fn poll(&mut self) -> Look {
        self.poll_looking(look)
    }
}

/// A directory with a `policy.yaml`, and the generations of a store of
/// its own.
struct Bed {
    dir: TempDir,
    path: PathBuf,
    generations: Arc<Generations>,
}

fn bed(document: &[u8]) -> Bed {
    bed_validating(document, AppliedSpec::validate_digested)
}

/// [`bed`] with the validator of its generations.
fn bed_validating(document: &[u8], validate: Validate) -> Bed {
    bed_of(document, validate, None)
}

fn bed_of(document: &[u8], validate: Validate, fence: Option<Arc<Fence>>) -> Bed {
    let dir = tempfile::tempdir().expect("a directory");
    let path = dir.path().join("policy.yaml");
    fs::write(&path, document).expect("the file");
    let store = Arc::new(MemoryStore::new());
    let catalog = Arc::new(StageRegistry::builtin());
    let generations = Generations::validating(store, catalog, validate, fence);
    Bed {
        dir,
        path,
        generations: Arc::new(generations),
    }
}

impl Bed {
    /// A start on the bed's file, watched from then on.
    fn start(&self) -> SpecFile {
        SpecFile::start(self.path.clone(), Arc::clone(&self.generations)).expect("a valid document")
    }

    /// A start on `path`, by the generation it left standing.
    fn start_on(&self, path: &Path) -> Result<u64, SpecRejected> {
        SpecFile::start(path.to_path_buf(), Arc::clone(&self.generations))
            .map(|_| self.generation())
    }

    fn generation(&self) -> u64 {
        self.generations.generation()
    }

    fn rejected(&self) -> u64 {
        self.generations.apply_counts().rejected
    }

    /// The number, the format and the bytes the read call answers.
    fn answered(&self) -> (u64, String, Vec<u8>) {
        let answer = self.generations.document(0).expect("the standing document");
        (
            answer.generation,
            answer.format.as_str().to_owned(),
            answer.document,
        )
    }
}

/// Replaces the file as a writer that renames a finished file over it:
/// a new inode, never a half-written file.
fn replace(path: &Path, document: &[u8]) {
    let staged = path.with_extension("staged");
    fs::write(&staged, document).expect("the staged file");
    fs::rename(&staged, path).expect("the rename");
}

/// The two looks a moved file takes: the one that sees it moved, and the
/// one that finds it standing and reads it.
fn settle(file: &mut SpecFile) -> Result<ApplyOutcome, SpecRejected> {
    assert_eq!(
        file.poll(),
        Look::Settling,
        "a moved file is not read at once"
    );
    match file.poll() {
        Look::Read(outcome) => outcome,
        other => panic!("the file stood still and the look was {other:?}"),
    }
}

/// Reads through the descriptor seam and counts attempts to take bytes
/// from a descriptor other than the looked file.
fn read_counting_unexpected(
    identity: &Identity,
    unexpected_reads: &AtomicUsize,
) -> Option<Result<Vec<u8>, SpecRejected>> {
    read_with(identity, fs::File::metadata, |file, bytes| {
        let opened = file.metadata().ok().and_then(|metadata| {
            Identity::from_metadata(identity.resolved.clone(), &metadata).ok()
        });
        if opened.as_ref() != Some(identity) {
            unexpected_reads.fetch_add(1, Ordering::Relaxed);
        }
        read_to_bound(file, bytes)
    })
}

/// A start on `other` is refused as no regular file, and so is the
/// watched file once its name turns to `other`: one rejected state, the
/// running generation kept. The start and the looks run on a thread of
/// their own and are waited five seconds on the real clock, so a source
/// that waits on `other` fails the case where it would hang it.
#[cfg(unix)]
fn refused_as_no_file(bed: Bed, other: PathBuf) {
    assert_eq!(
        look(&other),
        Seen::Unreadable(NOT_A_FILE.to_owned()),
        "the look refuses it, unopened"
    );
    let (answer, answered) = mpsc::channel();
    thread::spawn(move || {
        let started = bed.start_on(&other);
        let bound = bed.generations.bound();
        // The watched file's name turns to the other.
        let mut file = bed.start();
        fs::rename(&other, &bed.path).expect("the rename");
        let watched = settle(&mut file);
        let next = file.poll();
        let counts = (bed.rejected(), bed.generation());
        let _ = answer.send((started, bound, watched, next, counts));
    });
    let (started, bound, watched, next, counts) = answered
        .recv_timeout(Duration::from_secs(5))
        .expect("the start and the looks answer: none waits on the file");
    let refused = || SpecRejected::Unreadable(NOT_A_FILE.to_owned());
    assert_eq!(started, Err(refused()));
    assert!(!bound, "a refused start binds nothing");
    assert_eq!(watched, Err(refused()));
    assert_eq!(next, Look::Same, "a rejected state is one event");
    assert_eq!(counts, (1, 1));
}

#[cfg(unix)]
pub(crate) use crate::document::operator::test_support::pipe_at;

/// A pipe is refused by its kind, at the start and under the watch,
/// unopened: this one has no writer, and an open would wait for one.
#[cfg(unix)]
#[test]
fn a_pipe_is_refused_by_its_kind_unread() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let pipe = bed.dir.path().join("pipe.yaml");
    pipe_at(&pipe);
    refused_as_no_file(bed, pipe);
}

/// A name that turns to a pipe between a look and its read is no state of
/// the looked file: the read opens without waiting and reads no byte of
/// what is no regular file. The read runs on a thread of its own, waited
/// five seconds, so an open that waits fails the case.
#[cfg(unix)]
#[test]
fn a_name_turned_to_a_pipe_under_a_read_is_not_read() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let looked = identity(&bed.path).expect("the file");
    let pipe = bed.dir.path().join("pipe.yaml");
    pipe_at(&pipe);
    fs::rename(&pipe, &bed.path).expect("the rename");

    let unexpected_reads = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&unexpected_reads);
    let (answer, answered) = mpsc::channel();
    thread::spawn(move || {
        let _ = answer.send(read_counting_unexpected(&looked, &counted));
    });
    let read = answered
        .recv_timeout(Duration::from_secs(5))
        .expect("the read answers: its open does not wait");
    assert_eq!(read, None);
    assert_eq!(unexpected_reads.load(Ordering::Relaxed), 0);
}

/// A device is refused as a pipe is, behind a link: a terminal's own
/// end gives a read nothing and holds it, and the zero device's bytes
/// never end.
#[cfg(unix)]
#[test]
fn a_device_is_refused_by_its_kind_unread() {
    use std::os::unix::fs::symlink;

    for device in ["/dev/ptmx", "/dev/zero"] {
        let bed = bed(LEAST_LOAD.as_bytes());
        let link = bed.dir.path().join("device.yaml");
        symlink(device, &link).expect("the link");
        refused_as_no_file(bed, link);
    }
}

/// An open takes no terminal: a process that leads a session of its own
/// opens a terminal's end and still has none, where an open without the
/// flag takes it. A process leads a session once, so the case runs
/// itself again as that process.
#[cfg(target_os = "linux")]
#[test]
fn an_open_takes_no_terminal() {
    use std::{env, ffi::OsStr, mem::ManuallyDrop, os::unix::ffi::OsStrExt};

    use rustix::{
        io::Errno,
        pty::{grantpt, openpt, ptsname, unlockpt, OpenptFlags},
    };

    const NAME: &str = "service::controller::spec_file::tests::an_open_takes_no_terminal";
    const LEADS: &str = "SMG_POLICY_SPEC_LEADS_A_SESSION";

    if env::var_os(LEADS).is_none() {
        let this = env::current_exe().expect("this executable");
        let child = Command::new(this)
            .args(["--exact", NAME, "--nocapture", "--test-threads=1"])
            .env(LEADS, "1")
            .output()
            .expect("the session's own process");
        let mut said = String::from_utf8_lossy(&child.stdout).into_owned();
        said.push_str(&String::from_utf8_lossy(&child.stderr));
        // One case ran there: a name that runs none passes too.
        assert!(
            child.status.success() && said.contains("1 passed"),
            "{said}"
        );
        return;
    }

    rustix::process::setsid().expect("a session of the process's own");
    let own = || {
        fs::File::open("/dev/tty")
            .map(drop)
            .map_err(|refused| refused.raw_os_error())
    };
    let none = Err(Some(Errno::NXIO.raw_os_error()));
    assert_eq!(own(), none, "a new session has no terminal");

    // The pair stays open to the process's end: its close hangs the
    // terminal up, which ends a process that took it.
    let pair = openpt(OpenptFlags::RDWR | OpenptFlags::NOCTTY).expect("a terminal pair");
    let pair = ManuallyDrop::new(pair);
    grantpt(&*pair).expect("the pair's other end is granted");
    unlockpt(&*pair).expect("the pair's other end is unlocked");
    let name = ptsname(&*pair, Vec::new()).expect("the other end's name");
    let end = Path::new(OsStr::from_bytes(name.to_bytes()));

    let opened = open(end).expect("the terminal's end opens");
    assert_eq!(own(), none, "the open took the terminal");
    drop(opened);
    // The control: here an open without the flag takes it.
    let _plain = fs::File::open(end).expect("the end opens again");
    assert_eq!(own(), Ok(()), "an open without the flag takes the terminal");
}

/// The loop's end, waited on the real clock for five seconds at most: a
/// held clock does not move while a look is in flight, so a timeout on
/// it would wait for ever on a loop that does not end.
async fn ended(watching: &mut JoinSet<Result<(), JoinError>>) -> Result<(), JoinError> {
    for _ in 0..5_000 {
        if let Some(joined) = watching.try_join_next() {
            return joined.expect("the watch's task");
        }
        tokio::task::yield_now().await;
        thread::sleep(Duration::from_millis(1));
    }
    panic!("the watch did not end at the stop");
}
