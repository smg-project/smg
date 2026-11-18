//! The policy spec's file source (design 4.6): the document is a file, read
//! at the start and watched from then on.
//!
//! The start reads the file as it stands, through every link on its path (a
//! mounted ConfigMap's file is a link), and applies its document as the
//! source's first ([`Generations`]). A file that cannot be read, a file over
//! the bound (refused by its size, unread) and a document that does not run
//! each refuse the start. So does a pipe, a device or a directory: the look
//! refuses it by its kind, unopened. A name may turn between a look and its
//! read, so the read opens without waiting and never as a controlling
//! terminal, and reads only what the opened file says is a regular file.
//!
//! The watch is a poll. A second after each look ended the path is resolved
//! and the file's identity read: where the path leads once every link is
//! followed, the file's size and times, and on Unix its inode. Two looks
//! are never closer than a poll, however long one took. A file is read
//! only when its identity moved from the one last read and then stood still
//! for one poll: an update that swaps a link is a new identity at once, and
//! an edit in place is not read half written. Every state the file settles
//! in is read once and has one outcome: applied, unchanged or rejected. A
//! rejected state (a document that does not run, a file over the bound, a
//! file that cannot be read) is one event and one count however long it
//! stands, and the running generation stays. A file that does not stand
//! still, look after look, is never read: it is told once, as a rejected
//! state is.
//!
//! A file is read whole or not at all, at the start and under the watch: its
//! identity is looked at before and after the read, and what a read gave
//! while the file moved, bytes or an error, is no state of the file: a
//! mount's rotation takes a version's directory from under an open. The
//! start reads such a file again, and the watch reads it once it stood
//! still.
//!
//! Limit. An identity is the file system's word: an edit in place that keeps
//! the file's size, made within the timestamp grain of the last look, is
//! seen at the file's next change. A swapped link is always seen.

use std::{
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime},
};

use tokio::{sync::watch, task::JoinError, time::sleep};

use super::generations::{Applied, Format, Generations, SpecRejected, MAX_DOCUMENT_BYTES};

/// Between two looks at the file.
pub(crate) const POLL: Duration = Duration::from_secs(1);

/// How many times a start reads a file that moves under its read before it
/// refuses.
const START_READS: usize = 3;

/// How many looks in a row find the file moved before the watch tells it as
/// a file that does not stand still.
const RESTLESS_LOOKS: u32 = 5;

/// What a look or a read says of a path that leads to no regular file.
const NOT_A_FILE: &str = "the path leads to no regular file";

/// What one look at the path found.
#[derive(Clone, Debug, Eq, PartialEq)]
enum Seen {
    /// A file, by its identity.
    File(Identity),
    /// No file a look could read the identity of, by the error's text.
    Unreadable(String),
}

/// A file's identity: two looks that read the same one saw the same bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Identity {
    /// Where the path leads once every link is followed.
    resolved: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    inode: Inode,
}

/// The file's inode, with the time of its last change.
#[cfg(unix)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Inode {
    device: u64,
    number: u64,
    changed: (i64, i64),
}

fn look(path: &Path) -> Seen {
    match identity(path) {
        Ok(identity) => Seen::File(identity),
        Err(error) => Seen::Unreadable(error.to_string()),
    }
}

fn identity(path: &Path) -> io::Result<Identity> {
    let resolved = fs::canonicalize(path)?;
    let metadata = fs::metadata(&resolved)?;
    // A pipe, a device or a directory is refused here, unopened. The read
    // asks the opened file again: the name may turn between the two.
    if !metadata.is_file() {
        return Err(io::Error::other(NOT_A_FILE));
    }
    Ok(Identity {
        len: metadata.len(),
        modified: metadata.modified().ok(),
        #[cfg(unix)]
        inode: {
            use std::os::unix::fs::MetadataExt;
            Inode {
                device: metadata.dev(),
                number: metadata.ino(),
                changed: (metadata.ctime(), metadata.ctime_nsec()),
            }
        },
        resolved,
    })
}

/// Opens the file without waiting, and never as the process's controlling
/// terminal. An open of a pipe waits for a writer otherwise, a start's
/// without end, and a session leader that has no terminal takes the first
/// terminal's device it opens: the kind is read after the open.
fn open(path: &Path) -> io::Result<fs::File> {
    use rustix::fs::{Mode, OFlags};

    let flags = OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOCTTY | OFlags::CLOEXEC;
    let opened = rustix::io::retry_on_intr(|| rustix::fs::open(path, flags, Mode::empty()))?;
    Ok(opened.into())
}

/// Opens a regular file for a read without waiting ([`open`]) and refuses
/// what is none by its kind, unread: a read of a pipe or of a terminal's
/// device waits on a writer. The opened file states the kind, not its path.
pub(super) fn open_regular(path: &Path) -> io::Result<fs::File> {
    let file = open(path)?;
    if !file.metadata()?.is_file() {
        return Err(io::Error::other(NOT_A_FILE));
    }
    Ok(file)
}

/// The file's bytes, whole. A file over the bound is refused by its size,
/// unread, and one that is no regular file by its kind, unread: a read of a
/// pipe or of a terminal's device waits on a writer.
fn read(identity: &Identity) -> Result<Vec<u8>, SpecRejected> {
    if identity.len > MAX_DOCUMENT_BYTES {
        return Err(SpecRejected::TooLarge(identity.len));
    }
    let unreadable = |error: io::Error| SpecRejected::Unreadable(error.to_string());
    // The opened file has the word, not the path a look found: the name may
    // have turned between the two.
    let file = open_regular(&identity.resolved).map_err(unreadable)?;
    let mut bytes = Vec::with_capacity(usize::try_from(identity.len).unwrap_or(0));
    // One byte past the bound and no more. A file that grew since the look
    // is another file at the look after the read. A file whose size a look
    // cannot read (one the kernel writes as it is read) and that streams
    // past the bound hands the door one byte over it and is refused by its
    // size, never cut to a prefix that may parse.
    file.take(MAX_DOCUMENT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(unreadable)?;
    Ok(bytes)
}

/// What one look of the watch did.
#[derive(Debug, Eq, PartialEq)]
enum Look {
    /// The file is as it was last read.
    Same,
    /// The file moved: it is read once it has stood still for one poll.
    Settling,
    /// The file moved at [`RESTLESS_LOOKS`] looks in a row and was never
    /// read: told at this look, once.
    Restless,
    /// The file was read, with this outcome.
    Read(Result<Applied, SpecRejected>),
}

/// The file source: the path, and where its watch stands.
pub(crate) struct SpecFile {
    path: PathBuf,
    format: Format,
    generations: Arc<Generations>,
    /// The state last read: seen again, it is not read again.
    read: Seen,
    /// A state other than the last read, seen once: read when the next look
    /// finds it standing.
    moved: Option<Seen>,
    /// The looks in a row that found the file moved since the look before.
    unsettled: u32,
}

impl SpecFile {
    /// The start: reads the file at `path` as it stands and applies its
    /// document as the source's first. A file that cannot be read, or a
    /// document that does not run, refuses the start.
    pub(crate) fn start(
        path: PathBuf,
        generations: Arc<Generations>,
    ) -> Result<Self, SpecRejected> {
        Self::start_looking(path, generations, look)
    }

    /// [`SpecFile::start`] with the look as its argument: the identity is
    /// the file system's word, and a case hands in a file that moves under
    /// its read.
    fn start_looking(
        path: PathBuf,
        generations: Arc<Generations>,
        mut look: impl FnMut(&Path) -> Seen,
    ) -> Result<Self, SpecRejected> {
        let format = Format::of(&path);
        // The look after the read has the word. A file that moved under its
        // read is read again, the read's bytes or its error dropped; one
        // that stood still gives its bytes or its own error. One that never
        // rests is no document.
        for _ in 0..START_READS {
            let found = look(&path);
            let document = match &found {
                Seen::File(identity) => read(identity),
                Seen::Unreadable(error) => Err(SpecRejected::Unreadable(error.clone())),
            };
            if look(&path) == found {
                generations.apply(document?, format)?;
                return Ok(Self {
                    path,
                    format,
                    generations,
                    read: found,
                    moved: None,
                    unsettled: 0,
                });
            }
        }
        Err(SpecRejected::Unreadable(format!(
            "the file moved under each of {START_READS} reads"
        )))
    }

    /// [`SpecFile::poll_looking`] with the file system's own look: a case's
    /// step.
    #[cfg(test)]
    fn poll(&mut self) -> Look {
        self.poll_looking(look)
    }

    /// One look at the file: the watch's whole step, which the loop takes a
    /// poll after the last one ended and a case takes when it chooses. The
    /// look is its argument: the identity is the file system's word, and a
    /// case hands in a file that moves under its read. No clock is read:
    /// "stood still for one poll" is two looks in a row that agree.
    fn poll_looking(&mut self, mut look: impl FnMut(&Path) -> Seen) -> Look {
        let now = look(&self.path);
        if now == self.read {
            self.moved = None;
            self.unsettled = 0;
            return Look::Same;
        }
        if self.moved.as_ref() != Some(&now) {
            self.moved = Some(now);
            return self.settling();
        }
        let outcome = match &now {
            Seen::Unreadable(error) => Err(SpecRejected::Unreadable(error.clone())),
            Seen::File(identity) => {
                let document = read(identity);
                let after = look(&self.path);
                if after != now {
                    // It moved under the read: what the read gave, bytes or
                    // an error, is no state of the file.
                    self.moved = Some(after);
                    return self.settling();
                }
                document.and_then(|document| self.generations.apply(document, self.format))
            }
        };
        self.read = now;
        self.moved = None;
        self.unsettled = 0;
        Look::Read(outcome.map_err(|rejected| self.generations.reject(rejected)))
    }

    /// A look that found the file moved since the look before. A file that
    /// moved at [`RESTLESS_LOOKS`] looks in a row is told at that look, as a
    /// rejected state is: one event and one count, however long it goes on,
    /// until a look reads it or finds it as it was last read.
    fn settling(&mut self) -> Look {
        self.unsettled = self.unsettled.saturating_add(1);
        if self.unsettled != RESTLESS_LOOKS {
            return Look::Settling;
        }
        self.generations.reject(SpecRejected::Unreadable(format!(
            "the file moved at each of {RESTLESS_LOOKS} looks"
        )));
        Look::Restless
    }

    /// Looks at the file a [`POLL`] after each look ended, until `stop`
    /// turns. A look reads the file system, so each runs off the runtime's
    /// workers, one at a time. The stop ends the watch at once: a look in
    /// flight ends by itself and the watch does not wait on it. The runtime
    /// that ran it does, at its own end: the controller's `run` bounds that
    /// wait.
    pub(crate) async fn run(self, stop: watch::Receiver<bool>) -> Result<(), JoinError> {
        self.run_looking(stop, look).await
    }

    /// [`SpecFile::run`] with the look as its argument.
    async fn run_looking<L>(
        mut self,
        mut stop: watch::Receiver<bool>,
        mut look: L,
    ) -> Result<(), JoinError>
    where
        L: FnMut(&Path) -> Seen + Send + 'static,
    {
        loop {
            // A poll after the last look ended: two looks are never closer,
            // so "stood still for one poll" holds after a look that stalled.
            tokio::select! {
                () = sleep(POLL) => {}
                _ = stop.wait_for(|stop| *stop) => return Ok(()),
            }
            let looking = tokio::task::spawn_blocking(move || {
                self.poll_looking(&mut look);
                (self, look)
            });
            let (file, looked) = tokio::select! {
                joined = looking => joined?,
                _ = stop.wait_for(|stop| *stop) => return Ok(()),
            };
            self = file;
            look = looked;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        process::Command,
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        thread,
    };

    use tempfile::TempDir;
    use tokio::task::JoinSet;

    use super::*;
    use crate::{
        controller::generations::{
            tests::{faulting, long_text_under_aliases, padded, with_line, LEAST_LOAD, RANDOM},
            Validate,
        },
        spec::{AppliedSpec, SpecError},
        store::MemoryStore,
    };

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
        let dir = tempfile::tempdir().expect("a directory");
        let path = dir.path().join("policy.yaml");
        fs::write(&path, document).expect("the file");
        let store = Arc::new(MemoryStore::new());
        let catalog = Arc::new(crate::Catalog::builtin());
        let generations = Arc::new(Generations::validating(store, catalog, validate));
        Bed {
            dir,
            path,
            generations,
        }
    }

    impl Bed {
        /// A start on the bed's file, watched from then on.
        fn start(&self) -> SpecFile {
            SpecFile::start(self.path.clone(), Arc::clone(&self.generations))
                .expect("a valid document")
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
            self.generations.counts().rejected
        }

        /// The number, the format and the bytes the read call answers.
        fn answered(&self) -> (u64, String, Vec<u8>) {
            let answer = self.generations.document(0).expect("the standing document");
            (answer.generation, answer.format, answer.document)
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
    fn settle(file: &mut SpecFile) -> Result<Applied, SpecRejected> {
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

    /// The start reads the file as it stands: a document with no line is
    /// generation 1 and a written line is the number, in the form the
    /// file's name says. A file of exactly the bound starts: the bound is
    /// the largest document.
    #[test]
    fn a_start_applies_the_file_as_it_stands() {
        let bed = bed(LEAST_LOAD.as_bytes());
        assert_eq!(bed.start_on(&bed.path), Ok(1));
        assert_eq!(bed.answered(), (1, "yaml".to_owned(), LEAST_LOAD.into()));

        let json = br#"{"apiVersion":"smg.policy/v1","generation":4,"profiles":{"default":{"pick":"random"}}}"#;
        let path = bed.dir.path().join("policy.json");
        fs::write(&path, json).expect("the file");
        assert_eq!(bed.start_on(&path), Ok(4));
        assert_eq!(bed.answered(), (4, "json".to_owned(), json.to_vec()));

        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
        let path = bed.dir.path().join("largest.yaml");
        fs::write(&path, padded(RANDOM, bound)).expect("the file");
        assert_eq!(bed.start_on(&path), Ok(5));
        assert_eq!(bed.answered().2.len(), bound);
    }

    /// A mounted ConfigMap's file is a link through `..data` into a
    /// version's directory: the start follows it.
    #[cfg(unix)]
    #[test]
    fn a_start_follows_the_links_of_a_mounted_file() {
        use std::os::unix::fs::symlink;

        let bed = bed(b"");
        let mount = bed.dir.path().join("mount");
        fs::create_dir_all(mount.join("..version_1")).expect("the version's directory");
        fs::write(mount.join("..version_1/policy.yaml"), with_line(RANDOM, 6)).expect("the file");
        symlink("..version_1", mount.join("..data")).expect("the data link");
        symlink("..data/policy.yaml", mount.join("policy.yaml")).expect("the file's link");

        assert_eq!(bed.start_on(&mount.join("policy.yaml")), Ok(6));
        assert_eq!(bed.answered().2, with_line(RANDOM, 6));
    }

    /// A file that is not there, a directory, a file over the bound (by its
    /// size, unread) and a document the validator refuses each refuse the
    /// start by name; nothing is applied and nothing counted.
    #[test]
    fn a_start_refuses_a_file_it_cannot_run() {
        let bed = bed(b"");
        fs::remove_file(&bed.path).expect("the file goes");
        match bed.start_on(&bed.path) {
            Err(SpecRejected::Unreadable(error)) => {
                assert!(error.contains("No such file"), "{error}");
            }
            other => panic!("started as {other:?}"),
        }
        assert!(matches!(
            bed.start_on(bed.dir.path()),
            Err(SpecRejected::Unreadable(_))
        ));

        // Five bytes over: the refusal names the file's size, which no read
        // gave it (a read stops one byte past the bound).
        let over = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits") + 5;
        fs::write(&bed.path, vec![b'#'; over]).expect("the file");
        assert_eq!(
            bed.start_on(&bed.path),
            Err(SpecRejected::TooLarge(MAX_DOCUMENT_BYTES + 5))
        );

        fs::write(&bed.path, LEAST_LOAD.replace("[healthy]", "[nowhere]")).expect("the file");
        let unknown_filter = SpecError::UnknownFilter {
            profile: "default".to_owned(),
            name: "nowhere".to_owned(),
        };
        assert_eq!(
            bed.start_on(&bed.path),
            Err(SpecRejected::Invalid(unknown_filter))
        );
        assert!(!bed.generations.bound());
        assert_eq!((bed.generation(), bed.rejected()), (0, 0));
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

    /// A pipe at `path`, with no writer. The program makes it: the linked
    /// system-call crate has no call for one on every system the crate
    /// builds on.
    #[cfg(unix)]
    pub(crate) fn pipe_at(path: &Path) {
        let made = Command::new("mkfifo")
            .arg(path)
            .status()
            .expect("mkfifo runs");
        assert!(made.success(), "mkfifo: {made}");
    }

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

    /// A name that turns to a pipe between a look and its read is refused by
    /// the opened file's kind: the read opens without waiting and reads no
    /// byte of what is no regular file. The read runs on a thread of its
    /// own, waited five seconds, so an open that waits fails the case.
    #[cfg(unix)]
    #[test]
    fn a_name_turned_to_a_pipe_under_a_read_is_refused_unread() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let looked = identity(&bed.path).expect("the file");
        let pipe = bed.dir.path().join("pipe.yaml");
        pipe_at(&pipe);
        fs::rename(&pipe, &bed.path).expect("the rename");

        let (answer, answered) = mpsc::channel();
        thread::spawn(move || {
            let _ = answer.send(read(&looked));
        });
        let read = answered
            .recv_timeout(Duration::from_secs(5))
            .expect("the read answers: its open does not wait");
        assert_eq!(read, Err(SpecRejected::Unreadable(NOT_A_FILE.to_owned())));
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

        const NAME: &str = "controller::spec_file::tests::an_open_takes_no_terminal";
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

    /// The identity reads the file's change time: an edit in place that
    /// keeps the file's size and puts its modified time back is a new
    /// identity. The change time has the file system's grain, so the edit
    /// is made again until it shows, a bounded number of times.
    #[cfg(unix)]
    #[test]
    fn an_edit_that_keeps_the_size_and_the_modified_time_is_a_new_identity() {
        use std::io::Write as _;

        let bed = bed(LEAST_LOAD.as_bytes());
        let before = identity(&bed.path).expect("the file");
        let modified = before.modified.expect("a modified time");
        let edited = LEAST_LOAD.replace("best", "tseb");
        let mut after = before.clone();
        for _ in 0..2_000 {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .open(&bed.path)
                .expect("the file");
            file.write_all(edited.as_bytes()).expect("the edit");
            file.set_modified(modified)
                .expect("the modified time put back");
            drop(file);
            after = identity(&bed.path).expect("the file");
            if after != before {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert_ne!(after, before, "the edit never showed");
        assert_eq!(
            (after.len, after.modified, after.inode.number),
            (before.len, before.modified, before.inode.number),
            "the change time alone moved"
        );
        assert_ne!(after.inode.changed, before.inode.changed);
    }

    /// A read stops one byte past the bound, whatever the file grew to since
    /// its look: the byte over tells the door a document over the bound.
    #[test]
    fn a_read_stops_one_byte_past_the_bound() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let looked = identity(&bed.path).expect("the file");
        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
        fs::write(&bed.path, vec![b'#'; bound + 5]).expect("the file grows");
        assert_eq!(read(&looked).map(|bytes| bytes.len()), Ok(bound + 1));
    }

    /// The identity reads where the path leads: a link turned to another
    /// name of the same file (one inode, one size, the same times) is a new
    /// identity.
    #[cfg(unix)]
    #[test]
    fn a_link_turned_to_another_name_of_the_file_is_a_new_identity() {
        use std::os::unix::fs::symlink;

        let bed = bed(LEAST_LOAD.as_bytes());
        let (other, link) = (
            bed.dir.path().join("other.yaml"),
            bed.dir.path().join("link.yaml"),
        );
        fs::hard_link(&bed.path, &other).expect("another name of the file");
        symlink(&bed.path, &link).expect("the link");
        let before = identity(&link).expect("the file");
        fs::remove_file(&link).expect("the link goes");
        symlink(&other, &link).expect("the link, turned");
        let after = identity(&link).expect("the file");
        assert_ne!(after, before);
        assert_eq!(
            (after.len, after.modified, after.inode),
            (before.len, before.modified, before.inode),
            "where the path leads alone moved"
        );
    }

    /// What a look made before a mount's rotation saw: the file, in a
    /// version's directory the rotation has removed since.
    fn before_the_rotation(dir: &Path, path: &Path) -> Seen {
        match look(path) {
            Seen::File(identity) => Seen::File(Identity {
                resolved: dir.join("..version_1/policy.yaml"),
                ..identity
            }),
            unreadable @ Seen::Unreadable(_) => unreadable,
        }
    }

    /// A rotation takes the version a look resolved from under the open:
    /// the read fails and the file moved, so the start reads it again where
    /// the path leads next. The same for a look that could not read the
    /// path between two versions. An error is the file's own where the file
    /// stood still: one read between two looks that agree, and the start is
    /// refused by the error's text, not as a file that moved.
    #[test]
    fn a_file_gone_at_its_read_is_read_where_the_path_leads_next() {
        let rotated = bed(LEAST_LOAD.as_bytes());
        let mut looks = 0;
        let file = SpecFile::start_looking(
            rotated.path.clone(),
            Arc::clone(&rotated.generations),
            |path| {
                looks += 1;
                if looks == 1 {
                    before_the_rotation(rotated.dir.path(), path)
                } else {
                    look(path)
                }
            },
        );
        assert!(file.is_ok(), "the second read stood still");
        assert_eq!(looks, 4, "the read that failed and the one that stood");
        assert_eq!(rotated.answered().2, LEAST_LOAD.as_bytes());

        let between = bed(RANDOM.as_bytes());
        let mut looks = 0;
        let file = SpecFile::start_looking(
            between.path.clone(),
            Arc::clone(&between.generations),
            |path| {
                looks += 1;
                if looks == 1 {
                    Seen::Unreadable("between two versions".to_owned())
                } else {
                    look(path)
                }
            },
        );
        assert!(file.is_ok(), "the read stood still");
        assert_eq!(looks, 4, "the look that failed, and a read between two");
        assert_eq!(between.answered().2, RANDOM.as_bytes());

        let still = bed(LEAST_LOAD.as_bytes());
        let mut looks = 0;
        let refused =
            SpecFile::start_looking(still.path.clone(), Arc::clone(&still.generations), |path| {
                looks += 1;
                before_the_rotation(still.dir.path(), path)
            });
        match refused.err() {
            Some(SpecRejected::Unreadable(error)) => {
                assert!(error.contains("No such file"), "{error}");
            }
            other => panic!("started as {other:?}"),
        }
        assert_eq!(looks, 2, "one read between two looks");
        assert!(!still.generations.bound());
    }

    /// Under the watch a read that fails while the file moves is no state
    /// of the file: nothing is rejected and nothing counted, and the file
    /// is read once it stood still. An error of a file that stood still is
    /// a rejected state.
    #[test]
    fn a_read_that_fails_under_a_moving_file_is_no_rejected_state() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = bed.start();
        replace(&bed.path, RANDOM.as_bytes());
        // Two looks from before a rotation: the version they lead to is gone
        // at the open, and the look after the read finds the next one.
        let dir = bed.dir.path();
        let gone = |path: &Path| before_the_rotation(dir, path);
        assert_eq!(file.poll_looking(gone), Look::Settling);
        let mut looks = 0;
        let failed = file.poll_looking(|path| {
            looks += 1;
            if looks == 1 {
                gone(path)
            } else {
                look(path)
            }
        });
        assert_eq!(failed, Look::Settling, "the read's error while it moved");
        assert_eq!((bed.rejected(), bed.generation()), (0, 1));
        assert_eq!(file.poll(), Look::Read(Ok(Applied::Generation(2))));

        // The same lost version at every look: the file stood still.
        assert_eq!(file.poll_looking(gone), Look::Settling);
        match file.poll_looking(gone) {
            Look::Read(Err(SpecRejected::Unreadable(error))) => {
                assert!(error.contains("No such file"), "{error}");
            }
            other => panic!("the look was {other:?}"),
        }
        assert_eq!((bed.rejected(), bed.generation()), (1, 2));
    }

    /// A file is read whole or not at all. At the start one that moved
    /// under its read is read again, and one that moves under every read
    /// refuses the start; under the watch the bytes read while it moved are
    /// not applied, and the file is read once it stood still.
    #[test]
    fn a_file_that_moves_under_its_read_is_read_again() {
        let started = bed(LEAST_LOAD.as_bytes());
        let mut looks = 0;
        let file = SpecFile::start_looking(
            started.path.clone(),
            Arc::clone(&started.generations),
            |path| {
                looks += 1;
                if looks == 2 {
                    // After the first read: a writer replaces the document.
                    fs::write(path, RANDOM).expect("the file");
                }
                look(path)
            },
        );
        assert!(file.is_ok(), "the second read stood still");
        assert_eq!(looks, 4, "two reads, each between two looks");
        assert_eq!(started.answered().2, RANDOM.as_bytes());

        let restless = bed(LEAST_LOAD.as_bytes());
        let mut looks = 0;
        let refused = SpecFile::start_looking(
            restless.path.clone(),
            Arc::clone(&restless.generations),
            |path| {
                looks += 1;
                if looks % 2 == 0 {
                    let document = if looks % 4 == 0 { LEAST_LOAD } else { RANDOM };
                    fs::write(path, document).expect("the file");
                }
                look(path)
            },
        );
        let moved = SpecRejected::Unreadable("the file moved under each of 3 reads".to_owned());
        assert_eq!(refused.err(), Some(moved));
        assert!(!restless.generations.bound());

        let watched = bed(LEAST_LOAD.as_bytes());
        let mut file = watched.start();
        replace(&watched.path, RANDOM.as_bytes());
        assert_eq!(file.poll(), Look::Settling);
        let mut looks = 0;
        let moving = file.poll_looking(|path| {
            looks += 1;
            if looks == 2 {
                // After the read: the document is replaced again.
                fs::write(path, with_line(LEAST_LOAD, 9)).expect("the file");
            }
            look(path)
        });
        assert_eq!(moving, Look::Settling, "the bytes read while it moved");
        assert_eq!(watched.generation(), 1);
        assert_eq!(file.poll(), Look::Read(Ok(Applied::Generation(9))));
    }

    /// A changed file is read once it stood still for one poll, never at the
    /// look that first sees it moved; a file that moves again before it
    /// stood still waits one poll more, so an edit in place is not read half
    /// written.
    #[test]
    fn a_changed_file_is_read_once_it_stood_still_for_one_poll() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = bed.start();
        assert_eq!(file.poll(), Look::Same);

        replace(&bed.path, RANDOM.as_bytes());
        assert_eq!(file.poll(), Look::Settling);
        assert_eq!(bed.generation(), 1, "not read at the look that saw it move");
        assert_eq!(file.poll(), Look::Read(Ok(Applied::Generation(2))));
        assert_eq!(bed.answered().2, RANDOM.as_bytes());
        assert_eq!(file.poll(), Look::Same);

        // An edit in place, in two writes with a look between them.
        fs::write(&bed.path, with_line(LEAST_LOAD, 9)).expect("the first write");
        assert_eq!(file.poll(), Look::Settling);
        fs::write(&bed.path, with_line(RANDOM, 11)).expect("the second write");
        assert_eq!(file.poll(), Look::Settling, "it moved since the last look");
        assert_eq!(bed.generation(), 2);
        assert_eq!(file.poll(), Look::Read(Ok(Applied::Generation(11))));
        assert_eq!(file.poll(), Look::Same);
    }

    /// A state that left is forgotten: a file seen moved at one look and as
    /// it was last read at the next has not stood still, and when the same
    /// new state comes back it waits its poll again.
    #[cfg(unix)]
    #[test]
    fn a_state_that_left_is_forgotten() {
        use std::os::unix::fs::symlink;

        let bed = bed(b"");
        let mount = bed.dir.path().join("mount");
        for (version, document) in [("..v1", LEAST_LOAD), ("..v2", RANDOM)] {
            fs::create_dir_all(mount.join(version)).expect("the version's directory");
            fs::write(mount.join(version).join("policy.yaml"), document).expect("the file");
        }
        let turn = |version: &str| {
            let staged = mount.join("..data_tmp");
            symlink(version, &staged).expect("the staged link");
            fs::rename(&staged, mount.join("..data")).expect("the swap");
        };
        turn("..v1");
        symlink("..data/policy.yaml", mount.join("policy.yaml")).expect("the file's link");
        let mut file = SpecFile::start(mount.join("policy.yaml"), Arc::clone(&bed.generations))
            .expect("a valid document");

        turn("..v2");
        assert_eq!(file.poll(), Look::Settling);
        turn("..v1");
        assert_eq!(file.poll(), Look::Same, "as it was last read");
        turn("..v2");
        assert_eq!(file.poll(), Look::Settling, "it waits its poll again");
        assert_eq!(bed.generation(), 1);
        assert_eq!(file.poll(), Look::Read(Ok(Applied::Generation(2))));
    }

    /// A mounted ConfigMap's update: the file is a link through `..data`,
    /// and an update swaps `..data` to a new directory in one rename. The
    /// path then leads elsewhere, which is a new identity whatever the
    /// file's size and times are.
    #[cfg(unix)]
    #[test]
    fn a_swapped_link_is_the_next_generation() {
        use std::os::unix::fs::symlink;

        let bed = bed(b"");
        let mount = bed.dir.path().join("mount");
        fs::create_dir(&mount).expect("the mount");
        let publish = |version: &str, document: &str| {
            fs::create_dir(mount.join(version)).expect("the version's directory");
            fs::write(mount.join(version).join("policy.yaml"), document).expect("the file");
            symlink(version, mount.join("..data_tmp")).expect("the staged link");
            fs::rename(mount.join("..data_tmp"), mount.join("..data")).expect("the swap");
        };
        publish("..version_1", LEAST_LOAD);
        symlink("..data/policy.yaml", mount.join("policy.yaml")).expect("the file's link");

        let mut file = SpecFile::start(mount.join("policy.yaml"), Arc::clone(&bed.generations))
            .expect("a valid document");
        assert_eq!(bed.generation(), 1);
        assert_eq!(file.poll(), Look::Same);

        publish("..version_2", RANDOM);
        fs::remove_dir_all(mount.join("..version_1")).expect("the old version goes");
        assert_eq!(settle(&mut file), Ok(Applied::Generation(2)));
        assert_eq!(bed.answered().2, RANDOM.as_bytes());
        assert_eq!(file.poll(), Look::Same);
    }

    /// The events are bounded by the file's states: a document that does
    /// not run is rejected once, at the look that reads it, and every look
    /// after it finds the file as it was last read. The running generation
    /// and its document stay, and the next valid file is the next
    /// generation.
    #[test]
    fn a_rejected_state_is_one_event_however_long_it_stands() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = bed.start();

        replace(
            &bed.path,
            LEAST_LOAD.replace("[healthy]", "[nowhere]").as_bytes(),
        );
        assert!(matches!(
            settle(&mut file),
            Err(SpecRejected::Invalid(SpecError::UnknownFilter { .. }))
        ));
        for _ in 0..10 {
            assert_eq!(file.poll(), Look::Same);
        }
        assert_eq!((bed.rejected(), bed.generation()), (1, 1));
        assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());

        replace(&bed.path, RANDOM.as_bytes());
        assert_eq!(settle(&mut file), Ok(Applied::Generation(2)));
        assert_eq!(bed.rejected(), 1);
    }

    /// A file that is gone, and a file over the bound, are rejected states
    /// like a document that does not run: one event each, the running
    /// generation kept. The file over the bound is refused by its size,
    /// unread.
    #[test]
    fn a_file_that_is_gone_or_over_the_bound_is_a_rejected_state() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = bed.start();

        fs::remove_file(&bed.path).expect("the file goes");
        assert!(matches!(
            settle(&mut file),
            Err(SpecRejected::Unreadable(_))
        ));
        assert_eq!(file.poll(), Look::Same);
        assert_eq!((bed.rejected(), bed.generation()), (1, 1));

        let over = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits") + 5;
        fs::write(&bed.path, vec![b'#'; over]).expect("the file");
        assert_eq!(
            settle(&mut file),
            Err(SpecRejected::TooLarge(MAX_DOCUMENT_BYTES + 5))
        );
        assert_eq!(file.poll(), Look::Same);
        assert_eq!((bed.rejected(), bed.generation()), (2, 1));

        replace(&bed.path, RANDOM.as_bytes());
        assert_eq!(settle(&mut file), Ok(Applied::Generation(2)));
        assert_eq!(bed.rejected(), 2);
    }

    /// What the door refuses of a start it refuses of a running controller,
    /// as a rejected state: the standing document with a `kind` line added
    /// (its digest is the standing one's, and the validator refuses it), and
    /// a file its aliases read past the bound. One event and one count
    /// each, and the running generation stays.
    #[test]
    fn a_file_the_door_refuses_is_a_rejected_state_though_its_digest_stands() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = bed.start();

        replace(
            &bed.path,
            format!("kind: PolicySpec\n{LEAST_LOAD}").as_bytes(),
        );
        assert_eq!(
            settle(&mut file),
            Err(SpecRejected::Invalid(SpecError::Kind(
                "PolicySpec".to_owned()
            )))
        );
        assert_eq!((bed.rejected(), bed.generation()), (1, 1));

        replace(&bed.path, long_text_under_aliases(8_192, 2_048).as_bytes());
        match settle(&mut file) {
            Err(SpecRejected::Unparsed { format: "yaml", .. }) => {}
            other => panic!("the look read it as {other:?}"),
        }
        assert_eq!(file.poll(), Look::Same);
        assert_eq!((bed.rejected(), bed.generation()), (2, 1));
        assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
    }

    /// A file whose document faults its own pass is a rejected state like
    /// another: one event and one count, the running generation kept, and
    /// the watch looks on.
    #[test]
    fn a_file_whose_document_faults_is_a_rejected_state() {
        let bed = bed_validating(LEAST_LOAD.as_bytes(), faulting);
        let mut file = bed.start();

        replace(&bed.path, RANDOM.as_bytes());
        assert_eq!(
            settle(&mut file),
            Err(SpecRejected::Faulted("a fault of the validator".to_owned()))
        );
        assert_eq!(file.poll(), Look::Same);
        assert_eq!((bed.rejected(), bed.generation()), (1, 1));

        replace(&bed.path, &with_line(LEAST_LOAD, 3));
        assert_eq!(settle(&mut file), Ok(Applied::Generation(3)));
        assert_eq!(bed.rejected(), 1);
    }

    /// A file that does not stand still is never read, and it is told once:
    /// at the fifth look in a row that finds it moved, as a rejected state
    /// is, and at no look after it. Once it rests it is read, and a file
    /// that goes restless again is told again.
    #[test]
    fn a_file_that_never_rests_is_told_once() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = bed.start();
        // A writer that grows the file before every look.
        let mut writes = 0;
        let mut restless = |file: &mut SpecFile| {
            file.poll_looking(|path| {
                writes += 1;
                let grown = format!("{RANDOM}#{}\n", "x".repeat(writes));
                fs::write(path, grown).expect("the file");
                look(path)
            })
        };

        for _ in 1..RESTLESS_LOOKS {
            assert_eq!(restless(&mut file), Look::Settling);
        }
        assert_eq!(restless(&mut file), Look::Restless);
        for _ in 0..100 {
            assert_eq!(restless(&mut file), Look::Settling);
        }
        assert_eq!((bed.rejected(), bed.generation()), (1, 1));

        // It rests: the next look finds it standing and reads it.
        assert_eq!(file.poll(), Look::Read(Ok(Applied::Generation(2))));
        assert_eq!(file.poll(), Look::Same);
        for _ in 1..RESTLESS_LOOKS {
            assert_eq!(restless(&mut file), Look::Settling);
        }
        assert_eq!(restless(&mut file), Look::Restless);
        assert_eq!((bed.rejected(), bed.generation()), (2, 2));
    }

    /// A look that finds the file as it was last read ends a run of moving
    /// looks: the count starts again from the next look that finds it
    /// moved, and a file is told at the fifth such look in a row, by the
    /// number.
    #[test]
    fn a_look_that_finds_the_file_as_last_read_ends_the_restless_count() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = bed.start();
        // A look that finds another state each time, no file behind it.
        let mut moves = 0;
        let mut moved = |file: &mut SpecFile| {
            file.poll_looking(|_| {
                moves += 1;
                Seen::Unreadable(format!("moved {moves}"))
            })
        };

        for _ in 0..4 {
            assert_eq!(moved(&mut file), Look::Settling);
        }
        assert_eq!(file.poll(), Look::Same);
        for _ in 0..4 {
            assert_eq!(moved(&mut file), Look::Settling, "the run began again");
        }
        assert_eq!(moved(&mut file), Look::Restless, "the fifth look in a row");
        assert_eq!((bed.rejected(), bed.generation()), (1, 1));
    }

    /// A read ends a run of moving looks as a look at the file last read
    /// does: the look after the read that finds the file moved is the first
    /// of a new run, and the fifth of that run tells it.
    #[test]
    fn a_read_ends_the_restless_count() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = bed.start();
        // A writer that grows the file before every look.
        let mut writes = 0;
        let mut restless = |file: &mut SpecFile| {
            file.poll_looking(|path| {
                writes += 1;
                let grown = format!("{RANDOM}#{}\n", "x".repeat(writes));
                fs::write(path, grown).expect("the file");
                look(path)
            })
        };

        for _ in 1..RESTLESS_LOOKS {
            assert_eq!(restless(&mut file), Look::Settling);
        }
        // It rests one look short of being told, and is read.
        assert_eq!(file.poll(), Look::Read(Ok(Applied::Generation(2))));
        for _ in 1..RESTLESS_LOOKS {
            assert_eq!(restless(&mut file), Look::Settling, "the run began again");
        }
        assert_eq!(
            restless(&mut file),
            Look::Restless,
            "the fifth look in a row"
        );
        assert_eq!((bed.rejected(), bed.generation()), (1, 2));
    }

    /// Waits on the real clock, five seconds at most, until look `n` is in
    /// flight.
    async fn in_flight(looks: &AtomicUsize, n: usize) {
        for _ in 0..5_000 {
            if looks.load(Ordering::SeqCst) >= n {
                return;
            }
            tokio::task::yield_now().await;
            thread::sleep(Duration::from_millis(1));
        }
        panic!("look {n} did not start");
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

    /// The serve's loop on a clock the case holds. It takes one look a poll
    /// after the last one ended: a look that stalls for three polls is
    /// followed a poll after it ends, never at once. The stop ends the loop
    /// with a look in flight. The look is handed in and writes down the
    /// clock's instant: the verdict is those instants, read off the held
    /// clock. A loop that looks without end fails at the look past the
    /// case's bound, and the loop's end is waited on the real clock, so the
    /// case hangs on neither.
    #[tokio::test(start_paused = true)]
    async fn the_watch_looks_a_poll_after_the_last_look_and_ends_at_the_stop() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let file = bed.start();
        let began = tokio::time::Instant::now();
        let looks = Arc::new(AtomicUsize::new(0));
        let stalls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let (release, released) = mpsc::channel::<()>();
        let counted = {
            let (looks, stalls, seen) =
                (Arc::clone(&looks), Arc::clone(&stalls), Arc::clone(&seen));
            move |path: &Path| {
                seen.lock().push(began.elapsed());
                let n = looks.fetch_add(1, Ordering::SeqCst) + 1;
                assert!(n <= 20, "the loop looks without end");
                if n == stalls.load(Ordering::SeqCst) {
                    released.recv().expect("the case lets the look go");
                }
                look(path)
            }
        };
        let (stop, stopped) = watch::channel(false);
        let mut watching = JoinSet::new();
        watching.spawn(file.run_looking(stopped, counted));

        // No look before a poll has passed, then a step a poll: the first
        // sees the file moved, the second reads it between two looks.
        replace(&bed.path, RANDOM.as_bytes());
        sleep(POLL / 2).await;
        assert!(seen.lock().is_empty(), "no look before a poll has passed");
        sleep(2 * POLL).await;
        assert_eq!(*seen.lock(), [POLL, 2 * POLL, 2 * POLL]);
        assert_eq!(bed.generation(), 2);

        // The third step's look stalls while three polls pass: the next
        // comes a poll after it ended, not at once.
        stalls.store(4, Ordering::SeqCst);
        tokio::time::advance(POLL / 2).await;
        in_flight(&looks, 4).await;
        tokio::time::advance(3 * POLL).await;
        let let_go = began.elapsed();
        release.send(()).expect("the look waits");
        sleep(POLL * 3 / 2).await;
        let next = seen.lock()[4..].to_vec();
        assert_eq!(next, [let_go + POLL], "a poll after the stalled look");

        // The stop ends the loop with a look in flight.
        stalls.store(6, Ordering::SeqCst);
        tokio::time::advance(POLL / 2).await;
        in_flight(&looks, 6).await;
        stop.send_replace(true);
        ended(&mut watching).await.expect("the loop's last join");
        release.send(()).expect("the look waits");
    }

    /// The stop ends the loop between two looks as it ends it with a look
    /// in flight: at once. The held clock has not moved when the loop has
    /// ended, and no look followed the stop.
    #[tokio::test(start_paused = true)]
    async fn the_stop_ends_the_watch_between_two_looks() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let file = bed.start();
        let began = tokio::time::Instant::now();
        let looks = Arc::new(AtomicUsize::new(0));
        let counted = {
            let looks = Arc::clone(&looks);
            move |path: &Path| {
                looks.fetch_add(1, Ordering::SeqCst);
                look(path)
            }
        };
        let (stop, stopped) = watch::channel(false);
        let mut watching = JoinSet::new();
        watching.spawn(file.run_looking(stopped, counted));

        // One look, and half a poll into the wait for the next.
        sleep(POLL * 3 / 2).await;
        assert_eq!(looks.load(Ordering::SeqCst), 1);
        let stopped_at = began.elapsed();
        stop.send_replace(true);
        ended(&mut watching).await.expect("the loop's last join");
        assert_eq!(began.elapsed(), stopped_at, "the loop waited no poll out");
        assert_eq!(looks.load(Ordering::SeqCst), 1, "no look after the stop");
    }
}
