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
//! an edit in place is not read half written. A leader's read applies the
//! document, leaves it unchanged or rejects it. A rejected state (a
//! document that does not run, a file over the bound, a file that cannot
//! be read) is counted once while it stands under a lead, and the running
//! generation stays. A file that keeps moving is never read and is told
//! as a rejected state.
//!
//! Under a shared lease the watch looks whether its replica leads or
//! follows. A leader's look hands the document to the door, which applies
//! it. The look of a replica that follows shows the door the document
//! ([`Generations::shown`]): the door passes it and keeps none of its
//! bytes, nothing is applied, and nothing is said of a file that cannot be
//! read or does not run. So the door knows
//! whether the file moved before or after the generations its replica
//! followed, and a take runs the file only where it holds the newer word.
//!
//! A file is read whole or not at all, at the start and under the watch: its
//! identity is looked at before and after the read, and the opened file's
//! identity is checked before and after its bytes. What a read gave while
//! the file moved, bytes or an error, is no state of the file: a mount's
//! rotation takes a version's directory from under an open. The start reads
//! such a file again, and the watch reads it once it stood still.
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
use tracing::debug;

use super::generations::{ApplyOutcome, Format, Generations, SpecRejected, MAX_DOCUMENT_BYTES};
use crate::document::operator::spec_store::NOT_A_FILE;

/// Between two looks at the file.
pub(crate) const POLL: Duration = Duration::from_secs(1);

/// How many times a start reads a file that moves under its read before it
/// refuses.
const START_READS: usize = 3;

/// How many looks in a row find the file moved before the watch tells it as
/// a file that does not stand still.
const RESTLESS_LOOKS: u32 = 5;

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
    Identity::from_metadata(resolved, &metadata)
}

impl Identity {
    fn from_metadata(resolved: PathBuf, metadata: &fs::Metadata) -> io::Result<Self> {
        // A pipe, a device or a directory is no regular-file identity. This
        // is checked both for a path look and for an opened descriptor.
        if !metadata.is_file() {
            return Err(io::Error::other(NOT_A_FILE));
        }
        Ok(Self {
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

/// The file's bytes, whole, when its opened descriptor was the looked file
/// and kept that identity through the read, and `None` where it was not: a
/// file that moved. A file over the bound is refused by its size, unread. A
/// descriptor that is no regular file is no looked file, so it is `None`,
/// unread: a read of a pipe or of a terminal's device waits on a writer.
fn read(identity: &Identity) -> Option<Result<Vec<u8>, SpecRejected>> {
    read_with(identity, fs::File::metadata, read_to_bound)
}

/// [`read`] with descriptor operations as arguments, so cases can observe a
/// byte read and make descriptor metadata fail.
fn read_with(
    identity: &Identity,
    mut metadata: impl FnMut(&fs::File) -> io::Result<fs::Metadata>,
    read_bytes: impl FnOnce(&fs::File, &mut Vec<u8>) -> io::Result<usize>,
) -> Option<Result<Vec<u8>, SpecRejected>> {
    if identity.len > MAX_DOCUMENT_BYTES {
        return Some(Err(SpecRejected::TooLarge(identity.len)));
    }
    let unreadable = |error: io::Error| SpecRejected::Unreadable(error.to_string());
    // The opened file has the word, not the path a look found: the name may
    // have turned between the two.
    let file = match open(&identity.resolved) {
        Ok(file) => file,
        Err(error) => {
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
            ) {
                return None;
            }
            return Some(Err(unreadable(error)));
        }
    };
    let before = match metadata(&file) {
        Ok(metadata) => metadata,
        Err(error) => return Some(Err(unreadable(error))),
    };
    let opened = Identity::from_metadata(identity.resolved.clone(), &before).ok()?;
    if opened != *identity {
        return None;
    }
    let mut bytes = Vec::with_capacity(usize::try_from(identity.len).unwrap_or(0));
    // One byte past the bound and no more. A file that grew since the look
    // is another file at the look after the read. A file whose size a look
    // cannot read (one the kernel writes as it is read) and that streams
    // past the bound hands the door one byte over it and is refused by its
    // size, never cut to a prefix that may parse.
    let document = read_bytes(&file, &mut bytes)
        .map(|_| bytes)
        .map_err(unreadable);
    let after = match metadata(&file) {
        Ok(metadata) => metadata,
        Err(error) => return Some(Err(unreadable(error))),
    };
    let after = Identity::from_metadata(identity.resolved.clone(), &after).ok()?;
    (after == opened).then_some(document)
}

fn read_to_bound(file: &fs::File, bytes: &mut Vec<u8>) -> io::Result<usize> {
    file.take(MAX_DOCUMENT_BYTES + 1).read_to_end(bytes)
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
    /// The file was read while its replica follows: the door was shown what
    /// the file holds and applied nothing ([`Generations::shown`]).
    Shown,
    /// The file was read and the door read nothing of it: its replica
    /// stopped leading under the look, or leads under another term by now,
    /// and the number is the standing one ([`ApplyOutcome::Waits`]). The state
    /// stays unread.
    Waits(u64),
    /// The file was read and its document handed to the door, with this
    /// outcome.
    Read(Result<ApplyOutcome, SpecRejected>),
}

/// The file source: the path, and where its watch stands.
pub(crate) struct SpecFile {
    path: PathBuf,
    format: Format,
    generations: Arc<Generations>,
    /// The state last read, by a leader's look or by the look of a replica
    /// that follows: seen again, it is read again only by a leader's look,
    /// while its document has not stood on this replica since the file
    /// took it ([`Generations::source_stood`]). A leader runs what its
    /// file says, and a mount that lags brings nothing older back.
    read: Seen,
    /// The door's count of takes at the last read by a leader's look, or
    /// by the start. A state that such a read handed to the door is read
    /// again behind a take alone, as a start reads it, though it never
    /// moved: a rejected state stays one event under one lead.
    took: u64,
    /// Whether the state last read was shown to the door by the look of a
    /// replica that follows, and by no leader's look since: the next look
    /// that leads reads it again, behind a take or behind none. A lead
    /// that a renewal brings back runs what its file says as well.
    shown: bool,
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

    /// The start of a replica under a shared lease: the file is read as it
    /// stands and its document passed, never applied
    /// ([`Generations::defer`]). A file that cannot be read, or a document
    /// that does not run, refuses the start all the same. Until it leads,
    /// a replica that follows a leader runs what the leader's log gives
    /// it. Under its first lead the door applies these bytes where the
    /// replica runs nothing, at the take or behind the lead's opening
    /// ([`Generations::deferred_now`]). Behind a take the watch reads the
    /// file as it stands then, though it never moved, while its document
    /// has not stood on this replica since the file took it: a replica
    /// that followed another document runs its own file's once it leads.
    pub(crate) fn start_deferred(
        path: PathBuf,
        generations: Arc<Generations>,
    ) -> Result<Self, SpecRejected> {
        Self::start_by(path, generations, look, read, Generations::defer)
    }

    /// [`SpecFile::start`] with the look as its argument: the identity is
    /// the file system's word, and a case hands in a file that moves under
    /// its read.
    fn start_looking(
        path: PathBuf,
        generations: Arc<Generations>,
        look: impl FnMut(&Path) -> Seen,
    ) -> Result<Self, SpecRejected> {
        Self::start_reading(path, generations, look, read)
    }

    /// [`SpecFile::start_looking`] with the descriptor read as its argument.
    fn start_reading(
        path: PathBuf,
        generations: Arc<Generations>,
        look: impl FnMut(&Path) -> Seen,
        read: impl FnMut(&Identity) -> Option<Result<Vec<u8>, SpecRejected>>,
    ) -> Result<Self, SpecRejected> {
        let apply = |door: &Generations, document, format| door.apply(document, format).map(drop);
        Self::start_by(path, generations, look, read, apply)
    }

    /// The start's read, and the document handed to the door by `hand`: an
    /// apply, or the pass alone of a replica under a shared lease.
    fn start_by(
        path: PathBuf,
        generations: Arc<Generations>,
        mut look: impl FnMut(&Path) -> Seen,
        mut read: impl FnMut(&Identity) -> Option<Result<Vec<u8>, SpecRejected>>,
        hand: impl FnOnce(&Generations, Vec<u8>, Format) -> Result<(), SpecRejected>,
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
                Seen::Unreadable(error) => Some(Err(SpecRejected::Unreadable(error.clone()))),
            };
            let Some(document) = document else {
                continue;
            };
            if look(&path) == found {
                hand(&generations, document?, format)?;
                let took = generations.takes();
                return Ok(Self {
                    path,
                    format,
                    generations,
                    read: found,
                    took,
                    shown: false,
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
        self.poll_reading(&mut look, read)
    }

    /// [`SpecFile::poll_looking`] with the descriptor read as its argument.
    fn poll_reading(
        &mut self,
        mut look: impl FnMut(&Path) -> Seen,
        mut read: impl FnMut(&Identity) -> Option<Result<Vec<u8>, SpecRejected>>,
    ) -> Look {
        // The lead this look's bytes are read under: the door applies them
        // while its fence shows that term, and under no other. The start's
        // document, where it still waits for the replica's first lead, goes
        // through the door first.
        let lead = self.generations.lead();
        self.generations.deferred_now();
        let now = look(&self.path);
        let takes = self.generations.takes();
        // A state seen again is read again by a leader's look alone, while
        // its document has not stood on this replica since the file took
        // it: behind a take, or at any look where a replica that followed
        // showed it last.
        let again = lead.is_some()
            && (self.shown || takes != self.took)
            && !self.generations.source_stood();
        if now == self.read && !again {
            self.moved = None;
            self.unsettled = 0;
            return Look::Same;
        }
        if self.moved.as_ref() != Some(&now) {
            self.moved = Some(now);
            return self.settling(lead.is_some());
        }
        let document = match &now {
            Seen::Unreadable(error) => Err(SpecRejected::Unreadable(error.clone())),
            Seen::File(identity) => {
                let Some(document) = read(identity) else {
                    return self.settling(lead.is_some());
                };
                let after = look(&self.path);
                if after != now {
                    // It moved under the read: what the read gave, bytes or
                    // an error, is no state of the file.
                    self.moved = Some(after);
                    return self.settling(lead.is_some());
                }
                document
            }
        };
        self.moved = None;
        self.unsettled = 0;
        if lead.is_none() {
            // A replica that follows: the door is shown what the file
            // holds and applies nothing, and nothing is said of a file the
            // look could not read, but for one debug line. The next look
            // that leads reads the state again where its document has not
            // stood since.
            if let Err(unread) = &document {
                debug!(%unread, "policy spec file not read on a standby: its take tells it");
            }
            self.generations
                .shown(document.as_deref().ok(), self.format);
            self.read = now;
            self.shown = true;
            return Look::Shown;
        }
        let outcome = match document {
            Ok(document) => self.generations.apply_under(lead, document, self.format),
            Err(unread) => {
                // No document: the file holds none that stood.
                self.generations.shown(None, self.format);
                Err(unread)
            }
        };
        // The door read nothing: this replica stopped leading while the
        // look was in flight, or leads under another term by now. The state
        // stays unread, so the next look reads it.
        if let Ok(ApplyOutcome::Waits(standing)) = outcome {
            return Look::Waits(standing);
        }
        self.read = now;
        self.took = takes;
        self.shown = false;
        Look::Read(outcome.map_err(|rejected| self.generations.reject(rejected)))
    }

    /// The loop's step behind a sleep of `nap`. Woken short of a poll, at
    /// the opening of a lead the start's document waited for, it hands the
    /// door that document and makes no look: two looks are never closer
    /// than a poll. Behind a whole poll it looks.
    fn woken(&mut self, nap: Duration, look: impl FnMut(&Path) -> Seen) {
        if nap < POLL {
            self.generations.deferred_now();
        } else {
            self.poll_looking(look);
        }
    }

    /// A look that found the file moved since the look before. A file that
    /// moved at [`RESTLESS_LOOKS`] looks in a row is told at that look, as a
    /// rejected state is: one event and one count, however long it goes on,
    /// until a look reads it or finds it as it was last read. A replica
    /// that follows says nothing of its file: its count waits one short of
    /// the event, and the look that tells it is a leader's (`leads`).
    fn settling(&mut self, leads: bool) -> Look {
        if !leads {
            if self.unsettled < RESTLESS_LOOKS - 1 {
                self.unsettled += 1;
            }
            return Look::Settling;
        }
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
    ///
    /// A replica under a shared lease that does not lead looks all the
    /// same, and applies nothing: its look shows the door what the file
    /// holds, passed and by its digest and its own line
    /// ([`Generations::shown`]). The door holds who
    /// writes, and a look that was in flight when the replica stopped
    /// leading applies nothing there ([`ApplyOutcome::Waits`]). So two replicas
    /// that read one file never apply it twice under two numbers. The
    /// first looks after a take read the file as a start does, though it
    /// never moved, while its document has not stood on this replica since
    /// the file took it ([`Generations::source_stood`]).
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
            // The sleep is shorter where the start's document waits for a
            // lead that opens sooner.
            let nap = self.generations.nap(POLL);
            tokio::select! {
                () = sleep(nap) => {}
                _ = stop.wait_for(|stop| *stop) => return Ok(()),
            }
            let looking = tokio::task::spawn_blocking(move || {
                self.woken(nap, &mut look);
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
        document::spec::{AppliedSpec, Digest, PolicySpec, SpecError},
        service::{
            controller::generations::{
                tests::{
                    bound_to, election, election_under, faulting, long_text_under_aliases, padded,
                    with_line, LEAST_LOAD, RANDOM,
                },
                Validate,
            },
            elector::{Elector, TickOutcome},
        },
        stages::StageRegistry,
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
        let catalog = Arc::new(StageRegistry::builtin());
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

    /// The read of a file that left its path and came back to it, and that
    /// stood as `before` when it left. A rename stamps the file's change
    /// time, and the stamp has the file system's grain. Where it shows, the
    /// file moved and takes the two looks of [`settle`]. Where it does not,
    /// the watch finds the file as it last saw it, and the next look reads
    /// it.
    fn returned(file: &mut SpecFile, before: &Seen) -> Result<ApplyOutcome, SpecRejected> {
        if look(&file.path) != *before {
            return settle(file);
        }
        match file.poll() {
            Look::Read(outcome) => outcome,
            other => panic!("the file stood as last seen and the look was {other:?}"),
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

    /// A file that grew since its look is not read through a descriptor that
    /// no longer has the looked identity.
    #[test]
    fn a_file_that_grew_since_its_look_is_not_read() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let looked = identity(&bed.path).expect("the file");
        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
        fs::write(&bed.path, vec![b'#'; bound + 5]).expect("the file grows");
        let unexpected_reads = AtomicUsize::new(0);
        assert_eq!(read_counting_unexpected(&looked, &unexpected_reads), None);
        assert_eq!(unexpected_reads.load(Ordering::Relaxed), 0);
    }

    /// A read stops one byte past the bound, whatever the file holds: the
    /// byte over tells the door a document over the bound.
    #[test]
    fn a_read_stops_one_byte_past_the_bound() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
        fs::write(&bed.path, vec![b'#'; bound + 5]).expect("the file grows");
        let file = fs::File::open(&bed.path).expect("the file");
        let mut bytes = Vec::new();
        read_to_bound(&file, &mut bytes).expect("the read");
        assert_eq!(bytes.len(), bound + 1);
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
    /// refused by the error's text, not as a file that moved. An open that
    /// says the looked file disappeared contradicts that look and is retried.
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
        assert_eq!(looks, 3, "the missing open and the read that stood");
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
        assert_eq!(
            refused.err(),
            Some(SpecRejected::Unreadable(
                "the file moved under each of 3 reads".to_owned()
            ))
        );
        assert_eq!(looks, 3, "each missing open is retried");
        assert!(!still.generations.bound());
    }

    /// Under the watch a read that fails while the file moves is no state
    /// of the file: nothing is rejected and nothing counted, and the file
    /// is read once it stood still. A missing open also contradicts looks
    /// that claim the file stands there.
    #[test]
    fn a_read_that_fails_under_a_moving_file_is_no_rejected_state() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = bed.start();
        replace(&bed.path, RANDOM.as_bytes());
        // An injected look names a version already gone at the open. The
        // missing open contradicts the look, so no second look is made.
        let dir = bed.dir.path();
        let gone = |path: &Path| before_the_rotation(dir, path);
        assert_eq!(file.poll_looking(gone), Look::Settling);
        let mut looks = 0;
        let failed = file.poll_looking(|path| {
            looks += 1;
            gone(path)
        });
        assert_eq!(looks, 1, "the missing open ends this read");
        assert_eq!(failed, Look::Settling, "the missing open is retried");
        assert_eq!((bed.rejected(), bed.generation()), (0, 1));
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));

        // The same impossible lost version at every look: its missing open
        // still contradicts the injected look.
        assert_eq!(file.poll_looking(gone), Look::Settling);
        assert_eq!(file.poll_looking(gone), Look::Settling);
        assert_eq!((bed.rejected(), bed.generation()), (0, 2));
    }

    /// A missing open between two looks of the same standing file is no
    /// state of that file: the start retries it after the name comes back.
    #[test]
    fn a_missing_open_between_matching_start_looks_is_retried() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let standing = look(&bed.path);
        let saved = bed.dir.path().join("standing.yaml");
        let mut looks = 0;

        let started =
            SpecFile::start_looking(bed.path.clone(), Arc::clone(&bed.generations), |path| {
                looks += 1;
                match looks {
                    1 => {
                        fs::rename(path, &saved).expect("the file leaves before the open");
                        standing.clone()
                    }
                    2 => {
                        fs::rename(&saved, path).expect("the file returns before the next look");
                        standing.clone()
                    }
                    _ => look(path),
                }
            });

        assert!(started.is_ok(), "the missing open is retried");
        assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
    }

    /// A missing open between two looks of the same settled file is no state
    /// of that file: the watch retries it without rejecting the standing file.
    #[test]
    fn a_missing_open_between_matching_watch_looks_is_retried() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = bed.start();
        replace(&bed.path, RANDOM.as_bytes());
        assert_eq!(file.poll(), Look::Settling);
        let standing = look(&bed.path);
        let saved = bed.dir.path().join("standing.yaml");
        let mut looks = 0;

        let missing = file.poll_looking(|path| {
            looks += 1;
            match looks {
                1 => {
                    fs::rename(path, &saved).expect("the file leaves before the open");
                    standing.clone()
                }
                2 => {
                    fs::rename(&saved, path).expect("the file returns before the next look");
                    standing.clone()
                }
                _ => look(path),
            }
        });

        assert_eq!(missing, Look::Settling, "the missing open is retried");
        assert_eq!((bed.rejected(), bed.generation()), (0, 1));
        if saved.exists() {
            fs::rename(&saved, &bed.path).expect("the file returns after the missing open");
        }
        assert_eq!(
            returned(&mut file, &standing),
            Ok(ApplyOutcome::Generation(2))
        );
        assert_eq!(bed.answered().2, RANDOM.as_bytes());
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
        assert_eq!(file.poll(), Look::Read(Ok(ApplyOutcome::Generation(9))));
    }

    /// A start does not accept bytes from a replacement that appears only
    /// between two looks of the standing file.
    #[test]
    fn an_aba_replacement_during_start_is_not_accepted() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let standing = look(&bed.path);
        let saved = bed.dir.path().join("standing.yaml");
        fs::rename(&bed.path, &saved).expect("the standing file is saved");
        fs::write(&bed.path, RANDOM).expect("the transient replacement");
        let mut looks = 0;
        let unexpected_reads = AtomicUsize::new(0);

        let started = SpecFile::start_reading(
            bed.path.clone(),
            Arc::clone(&bed.generations),
            |path| {
                looks += 1;
                match looks {
                    1 => standing.clone(),
                    2 => {
                        fs::rename(&saved, path).expect("the standing file is restored");
                        standing.clone()
                    }
                    _ => look(path),
                }
            },
            |identity| read_counting_unexpected(identity, &unexpected_reads),
        );

        assert!(started.is_ok(), "the standing file starts after the retry");
        // The look whose read met the replacement, and the two around the
        // read of the restored file. Where the stamp of the renames shows,
        // that read is made once more.
        assert!(looks >= 3, "the transient read is retried");
        assert_eq!(unexpected_reads.load(Ordering::Relaxed), 0);
        assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
    }

    /// A watch does not accept bytes from a replacement that appears only
    /// between two looks of the settled file.
    #[test]
    fn an_aba_replacement_during_watch_is_not_accepted() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = bed.start();
        replace(&bed.path, RANDOM.as_bytes());
        assert_eq!(file.poll(), Look::Settling);

        let settled = look(&bed.path);
        let saved = bed.dir.path().join("settled.yaml");
        fs::rename(&bed.path, &saved).expect("the settled file is saved");
        fs::write(&bed.path, with_line(LEAST_LOAD, 3)).expect("the transient replacement");
        let mut looks = 0;
        let unexpected_reads = AtomicUsize::new(0);
        let read = file.poll_reading(
            |path| {
                looks += 1;
                match looks {
                    1 => settled.clone(),
                    2 => {
                        fs::rename(&saved, path).expect("the settled file is restored");
                        settled.clone()
                    }
                    _ => look(path),
                }
            },
            |identity| read_counting_unexpected(identity, &unexpected_reads),
        );

        assert_eq!(read, Look::Settling, "the transient read is retried");
        assert_eq!(unexpected_reads.load(Ordering::Relaxed), 0);
        assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
        if saved.exists() {
            fs::rename(&saved, &bed.path).expect("the settled file is restored");
        }
        assert_eq!(
            returned(&mut file, &settled),
            Ok(ApplyOutcome::Generation(2))
        );
        assert_eq!(bed.answered().2, RANDOM.as_bytes());
    }

    /// An error reading the opened descriptor's metadata is the standing
    /// file's error at the start, not evidence that the file moved.
    #[test]
    fn a_descriptor_metadata_error_during_start_is_reported() {
        const ERROR: &str = "descriptor metadata failed";

        let bed = bed(LEAST_LOAD.as_bytes());
        let started = SpecFile::start_reading(
            bed.path.clone(),
            Arc::clone(&bed.generations),
            look,
            |identity| read_with(identity, |_| Err(io::Error::other(ERROR)), read_to_bound),
        );

        assert_eq!(
            started.err(),
            Some(SpecRejected::Unreadable(ERROR.to_owned()))
        );
        assert!(!bed.generations.bound());
    }

    /// An error reading descriptor metadata after its bytes is the standing
    /// file's error under the watch, not evidence that the file moved.
    #[test]
    fn a_descriptor_metadata_error_after_a_watch_read_is_reported() {
        const ERROR: &str = "descriptor metadata failed after the read";

        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = bed.start();
        replace(&bed.path, RANDOM.as_bytes());
        assert_eq!(file.poll(), Look::Settling);
        let metadata_calls = Cell::new(0);

        let failed = file.poll_reading(look, |identity| {
            read_with(
                identity,
                |file| {
                    metadata_calls.set(metadata_calls.get() + 1);
                    if metadata_calls.get() == 2 {
                        Err(io::Error::other(ERROR))
                    } else {
                        file.metadata()
                    }
                },
                read_to_bound,
            )
        });

        assert_eq!(
            failed,
            Look::Read(Err(SpecRejected::Unreadable(ERROR.to_owned())))
        );
        assert_eq!((bed.rejected(), bed.generation()), (1, 1));
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
        assert_eq!(file.poll(), Look::Read(Ok(ApplyOutcome::Generation(2))));
        assert_eq!(bed.answered().2, RANDOM.as_bytes());
        assert_eq!(file.poll(), Look::Same);

        // An edit in place, in two writes with a look between them.
        fs::write(&bed.path, with_line(LEAST_LOAD, 9)).expect("the first write");
        assert_eq!(file.poll(), Look::Settling);
        fs::write(&bed.path, with_line(RANDOM, 11)).expect("the second write");
        assert_eq!(file.poll(), Look::Settling, "it moved since the last look");
        assert_eq!(bed.generation(), 2);
        assert_eq!(file.poll(), Look::Read(Ok(ApplyOutcome::Generation(11))));
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
        assert_eq!(file.poll(), Look::Read(Ok(ApplyOutcome::Generation(2))));
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
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
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
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
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
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
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
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(3)));
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
        assert_eq!(file.poll(), Look::Read(Ok(ApplyOutcome::Generation(2))));
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
        assert_eq!(file.poll(), Look::Read(Ok(ApplyOutcome::Generation(2))));
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

    /// The watch of a replica under a shared lease looks whether its
    /// replica leads or follows. While the replica follows, a look shows
    /// the door what the file became and applies nothing: the log writes at
    /// its door. Behind the replica's take the file is read again, though
    /// it never moved since: its document never stood there, and it is
    /// applied as a leader's is. Back under a leader, the watch looks on
    /// and applies nothing.
    #[tokio::test(start_paused = true)]
    async fn the_watch_of_a_replica_that_follows_looks_and_applies_nothing() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut elector = election();
        bed.generations.under(elector.fence());
        let door = Arc::clone(&bed.generations);
        elector.at_take(move || door.at_take());
        let file = SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        follow(&bed, &[RANDOM, &bound_to(1)]);
        let looks = Arc::new(AtomicUsize::new(0));
        let counted = {
            let looks = Arc::clone(&looks);
            move |path: &Path| {
                let n = looks.fetch_add(1, Ordering::SeqCst) + 1;
                assert!(n <= 20, "the loop looks without end");
                look(path)
            }
        };
        let (stop, stopped) = watch::channel(false);
        let mut watching = JoinSet::new();
        watching.spawn(file.run_looking(stopped, counted));

        // Between two polls, so the take and a look never share an instant.
        replace(&bed.path, RANDOM.as_bytes());
        sleep(9 * POLL / 2).await;
        // Four polls: the look that sees the file moved, the read between
        // two looks, and two looks that find the file as it was shown.
        assert_eq!(looks.load(Ordering::SeqCst), 5, "it looks while it follows");
        assert_eq!(bed.generation(), 2, "and applies nothing");
        assert_eq!(bed.rejected(), 0);

        // The take: the file stands as it was shown, and its document never
        // stood here. One look that settles it, then the read between two
        // looks.
        elector.tick(Instant::now()).expect("a take");
        sleep(2 * POLL).await;
        assert_eq!(looks.load(Ordering::SeqCst), 8);
        assert_eq!(bed.generation(), 3, "applied as a leader's file is");
        assert_eq!(bed.answered().2, RANDOM.as_bytes());

        assert!(elector.release().expect("a release"), "it held the lease");
        follow(&bed, &[RANDOM, &bound_to(1), &bound_to(2), &bound_to(3)]);
        replace(&bed.path, LEAST_LOAD.as_bytes());
        sleep(5 * POLL).await;
        assert_eq!(
            looks.load(Ordering::SeqCst),
            14,
            "it looks on once it follows"
        );
        assert_eq!(bed.generation(), 4, "and applies nothing");

        stop.send_replace(true);
        ended(&mut watching).await.expect("the loop's last join");
    }

    /// `bed` follows the standing generation of a leader that applied
    /// `documents` in their order, as a stream that begins now brings it.
    fn follow(bed: &Bed, documents: &[&str]) {
        follow_at(&bed.generations, documents);
    }

    /// [`follow`] on a door.
    fn follow_at(door: &Generations, documents: &[&str]) {
        let leader = bed(b"");
        for document in documents {
            let applied = leader
                .generations
                .apply(document.as_bytes().to_vec(), Format::Yaml);
            applied.expect("a document that runs");
        }
        let told = leader.generations.standing_sent().expect("it stands");
        let following = door.following(&told);
        following
            .and_then(|following| following.land(None))
            .expect("above no generation");
    }

    /// The start of a replica under a shared lease passes its file and
    /// applies nothing: the document is applied under the replica's first
    /// lead, here at its take, where the replica followed nothing. Behind
    /// a take the watch reads the file as a start does, though it never
    /// moved, while its document has not stood on the replica since the
    /// file took it: a replica that followed another document applies its
    /// own file's as the next generation. A file whose document stood
    /// there since, applied or told, is a mount that lags: it is left for
    /// its next move. A file that does not run refuses this start as it
    /// refuses the other.
    #[test]
    fn a_start_under_a_shared_lease_passes_its_file_and_the_first_take_applies_it() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut file = SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        assert_eq!(bed.generation(), 0, "passed, not applied");
        assert_eq!(file.poll(), Look::Same, "no take, and the file never moved");
        bed.generations.at_take();
        assert_eq!(bed.generation(), 1);
        assert_eq!(file.poll(), Look::Same, "the file's document stands");

        // The rollout: a replica that starts over a new file follows its
        // old leader's document. The take applies none of the start's
        // bytes, and behind it the file is read though it never moved: the
        // new document stands as the next generation.
        let bed = self::bed(LEAST_LOAD.as_bytes());
        let mut file = SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        follow(&bed, &[RANDOM]);
        bed.generations.at_take();
        assert_eq!(bed.generation(), 1, "the leader's, and none of the start's");
        assert_eq!(bed.answered().2, RANDOM.as_bytes());
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
        assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
        // Behind its next take the file's document stands: no read.
        bed.generations.at_take();
        assert_eq!(file.poll(), Look::Same);
        // A file that moved is read at any look.
        replace(&bed.path, bound_to(1).as_bytes());
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(3)));

        // A mount that lags: a standby's file holds the document it was
        // told at its join, and its leader applied another since. The take
        // brings nothing older back: the file is left for its next move.
        let bed = self::bed(LEAST_LOAD.as_bytes());
        let mut file = SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        follow(&bed, &[LEAST_LOAD]);
        follow(&bed, &[LEAST_LOAD, RANDOM]);
        bed.generations.at_take();
        assert_eq!(file.poll(), Look::Same, "an older document of its leader's");
        assert_eq!((bed.generation(), bed.answered().2), (2, RANDOM.into()));
        replace(&bed.path, RANDOM.as_bytes());
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Unchanged(2)));

        let unrun = self::bed(LEAST_LOAD.replace("healthy", "no-such-stage").as_bytes());
        let refused = SpecFile::start_deferred(unrun.path.clone(), Arc::clone(&unrun.generations));
        assert!(refused.is_err(), "a stage this build lacks");
        unrun.generations.at_take();
        assert_eq!(unrun.generation(), 0);
    }

    /// A file that went bad between the start and the first lead leaves the
    /// start's document. The watch hands that document first, and the bad
    /// file is one rejected state behind it.
    #[test]
    fn a_file_gone_bad_since_the_start_leaves_the_starts_document() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut elector = election();
        bed.generations.under(elector.fence());
        let mut file = SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        replace(&bed.path, b"{ profiles");
        // A take with no step: the start's document waits under an open
        // fence, and the watch's look hands it.
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(bed.generation(), 0);
        let read = settle(&mut file);
        assert!(read.is_err(), "the bad file is refused");
        assert_eq!(bed.generation(), 1, "the start's document stands");
        assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
        assert_eq!(bed.rejected(), 1, "the bad file is one rejected state");
        assert_eq!(file.poll(), Look::Same, "told once");
    }

    /// A start's document that its take's pass refuses is counted and let
    /// go, and the mark of a document awaited stays: no decision is made.
    /// The watch reads the file behind the take and lands its document, and
    /// the mark goes.
    #[test]
    fn a_starts_document_refused_at_its_take_is_landed_by_the_next_look() {
        thread_local! {
            static PASSES: Cell<u32> = const { Cell::new(0) };
        }
        // The start's pass runs, the take's faults, and every later one
        // runs.
        fn faulting_once(
            document: &PolicySpec,
            digest: Digest,
            catalog: &StageRegistry,
            denied_headers: &[&str],
        ) -> Result<AppliedSpec, SpecError> {
            let passed = PASSES.with(|passes| passes.replace(passes.get() + 1));
            assert_ne!(passed, 1, "a fault of the validator");
            AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
        }
        let dir = tempfile::tempdir().expect("a directory");
        let path = dir.path().join("policy.yaml");
        fs::write(&path, LEAST_LOAD).expect("the file");
        let store = Arc::new(MemoryStore::new());
        let catalog = Arc::new(StageRegistry::builtin());
        let door = Arc::new(Generations::validating(
            Arc::clone(&store),
            catalog,
            faulting_once,
        ));
        let awaits = || store.snapshot().spec().awaits_document();
        let mut elector = election();
        door.under(elector.fence());
        let mut file = SpecFile::start_deferred(path, Arc::clone(&door)).expect("a file that runs");
        assert!(awaits(), "the start's document waits");
        let stepped = Arc::clone(&door);
        elector.at_take(move || stepped.at_take());
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(door.generation(), 0, "refused at the take");
        assert_eq!(door.apply_counts().rejected, 1);
        assert!(awaits(), "the mark stays");
        assert!(door.bound(), "the door keeps its source");
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(1)));
        assert!(!awaits(), "the mark went with the unbound start");
    }

    /// A look that resumes behind a lost lead applies nothing: it is the
    /// look of a replica that follows, and shows the door the file. The
    /// looks of the replica's next lead read the state again, behind a
    /// take or behind none: its document has not stood there since.
    #[test]
    fn a_look_that_resumes_behind_a_lost_lead_shows_the_file_and_applies_nothing() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut elector = election();
        bed.generations.under(elector.fence());
        let mut file = SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        let door = Arc::clone(&bed.generations);
        elector.at_take(move || door.at_take());
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(bed.generation(), 1, "the start's document, at the take");
        replace(&bed.path, bound_to(1).as_bytes());
        assert_eq!(file.poll(), Look::Settling);
        // The replica is deposed here and follows its next leader, whose
        // generation is above its own.
        assert!(elector.release().expect("a release"), "it held the lease");
        follow(&bed, &[RANDOM, &bound_to(2)]);
        assert_eq!(bed.generation(), 2);
        // The look resumes: the door is shown the file and applies none of
        // it.
        assert_eq!(file.poll(), Look::Shown);
        assert_eq!(file.poll(), Look::Same, "shown once");
        assert_eq!(bed.generation(), 2);
        assert_eq!(bed.answered().2, bound_to(2).as_bytes());
        assert_eq!(bed.rejected(), 0);
        // Behind its next take the state is read again: its document has
        // not stood here.
        elector.tick(Instant::now()).expect("a take");
        assert!(bed.generations.source_writes(), "it leads again");
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(3)));
        assert_eq!(bed.answered().2, bound_to(1).as_bytes());

        // A lead comes back with no take as well: a fence closes by itself
        // at the renew deadline, and the holder's next renewal opens it
        // again under the same term. A take whose write returned past its
        // window shows no term: it publishes nothing, and the start's
        // document waits on. A look in between is a follower's, and shows
        // the file. Behind the renewal that opens the fence the watch
        // brings both: the start's document, then the file as it was shown.
        let deadline = Duration::from_secs(1);
        let bed = self::bed(LEAST_LOAD.as_bytes());
        let mut elector = election_under(deadline);
        bed.generations.under(elector.fence());
        let mut file = SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        let door = Arc::clone(&bed.generations);
        elector.at_take(move || door.at_take());
        // The take's write lands two deadlines late: its fence is closed.
        let late = Instant::now()
            .checked_sub(2 * deadline)
            .expect("a host that is up for two seconds");
        elector.tick(late).expect("a take");
        assert!(!bed.generations.source_writes(), "past its deadline");
        assert_eq!(bed.generation(), 0, "no term showed at the take's step");
        assert!(bed.generations.bound(), "the door keeps its source");
        replace(&bed.path, bound_to(1).as_bytes());
        assert_eq!(file.poll(), Look::Settling);
        assert_eq!(file.poll(), Look::Shown);
        assert_eq!(bed.generation(), 0, "nothing is applied");
        // A round whose window closed again ahead of its read waits, or
        // finds the file as it was shown.
        let read = (0..8).find_map(|_| {
            let renewed = elector.tick(Instant::now());
            assert!(
                matches!(renewed, Ok(TickOutcome::Renewed { .. })),
                "a renewal"
            );
            assert!(matches!(file.poll(), Look::Settling | Look::Same));
            Some(file.poll()).filter(|look| matches!(look, Look::Read(_)))
        });
        assert_eq!(read, Some(Look::Read(Ok(ApplyOutcome::Generation(2)))));
        assert_eq!(bed.answered().2, bound_to(1).as_bytes());
        assert_eq!(bed.generations.takes(), 1, "no take since the first");
    }

    /// A file shown while its leader's fence stood closed is read behind
    /// the renewal that opens the fence. No take stands between the two
    /// leads, and the leader's own look had read behind its take: the look
    /// reads the state again because a follower's look showed it last.
    #[test]
    fn a_file_shown_under_a_closed_fence_is_read_behind_the_renewal_that_opens_it() {
        let deadline = Duration::from_secs(30);
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut elector = election_under(deadline);
        bed.generations.under(elector.fence());
        let mut file = SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        let door = Arc::clone(&bed.generations);
        elector.at_take(move || door.at_take());
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(bed.generation(), 1, "the start's document, at the take");
        // A leader's look reads behind the take: no later look reads for
        // the take's sake.
        replace(&bed.path, bound_to(1).as_bytes());
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
        // A renewal whose write returns two deadlines late stamps a window
        // that has ended: the replica holds its lease and shows no term.
        // The file moves, and the look is a follower's.
        let late = Instant::now()
            .checked_sub(2 * deadline)
            .expect("a host that is up for a minute");
        let renewed = elector.tick(late);
        assert!(
            matches!(renewed, Ok(TickOutcome::Renewed { .. })),
            "a renewal"
        );
        assert!(!bed.generations.source_writes(), "past its deadline");
        shown(&bed, &mut file, bound_to(2).as_bytes());
        // The next renewal opens the fence under the same term, and the
        // look reads the state it was shown: its document has not stood
        // here.
        let renewed = elector.tick(Instant::now());
        assert!(
            matches!(renewed, Ok(TickOutcome::Renewed { .. })),
            "a renewal"
        );
        assert!(bed.generations.source_writes(), "it leads again");
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(3)));
        assert_eq!(bed.answered().2, bound_to(2).as_bytes());
        assert_eq!(bed.generations.takes(), 1, "no take since the first");
        assert_eq!(file.poll(), Look::Same, "read once");
    }

    /// The loop sleeps no poll ahead of the start's document where its lead
    /// is open: the document is applied at once, by the loop's own clock.
    #[tokio::test(start_paused = true)]
    async fn the_loop_brings_the_starts_document_with_no_polls_sleep_ahead_of_it() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut elector = election();
        bed.generations.under(elector.fence());
        let file = SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        // A take with no step of its own: the start's document waits under
        // an open fence.
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(bed.generation(), 0);
        let (stop, stopped) = watch::channel(false);
        let mut watching = JoinSet::new();
        let began = tokio::time::Instant::now();
        watching.spawn(file.run(stopped));
        while bed.generation() == 0 {
            assert!(began.elapsed() < 20 * POLL, "the loop brings the document");
            sleep(POLL / 100).await;
        }
        assert!(
            began.elapsed() < POLL,
            "ahead of a poll's sleep: {:?}",
            began.elapsed()
        );
        stop.send_replace(true);
        ended(&mut watching).await.expect("the loop's last join");
    }

    /// Woken short of a poll, at the opening of a lead the start's document
    /// waited for, the watch hands the door that document and looks at
    /// nothing. Behind a whole poll it looks.
    #[test]
    fn woken_at_a_leads_opening_the_watch_hands_over_the_starts_document_and_looks_at_nothing() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut elector = election();
        bed.generations.under(elector.fence());
        let mut file = SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        // A take with no step of its own: the start's document waits under
        // an open fence.
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(bed.generation(), 0);
        assert_eq!(bed.generations.nap(POLL), Duration::ZERO);
        let looks = AtomicUsize::new(0);
        let counted = |path: &Path| {
            looks.fetch_add(1, Ordering::SeqCst);
            look(path)
        };
        file.woken(Duration::ZERO, &counted);
        assert_eq!(bed.generation(), 1, "the start's document");
        assert_eq!(looks.load(Ordering::SeqCst), 0, "no look");
        file.woken(POLL, &counted);
        assert_eq!(looks.load(Ordering::SeqCst), 1, "a whole poll looks");
        assert_eq!(bed.generation(), 1);
    }

    /// A look whose replica was deposed and took again under its read
    /// applies nothing: its bytes were read under the term before. The
    /// state stays unread, and the next looks read the file as it stands.
    #[test]
    fn a_look_whose_lead_changed_under_its_read_applies_nothing_and_reads_again() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut elector = election();
        bed.generations.under(elector.fence());
        let mut file = SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        let door = Arc::clone(&bed.generations);
        elector.at_take(move || door.at_take());
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(bed.generation(), 1, "the start's document, at the take");
        replace(&bed.path, bound_to(1).as_bytes());
        assert_eq!(file.poll(), Look::Settling);
        // The second look of a poll stands behind its read: there the
        // replica is deposed and takes again.
        let mut looks = 0;
        let retaken = file.poll_looking(|path| {
            looks += 1;
            if looks == 2 {
                assert!(elector.release().expect("a release"), "it held the lease");
                elector.tick(Instant::now()).expect("a second take");
            }
            look(path)
        });
        assert_eq!(retaken, Look::Waits(1));
        assert_eq!(bed.generation(), 1);
        assert!(bed.generations.source_writes(), "it leads again");
        // Read again under the term that shows now, the file is applied.
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
        assert_eq!(bed.answered().2, bound_to(1).as_bytes());
        assert_eq!(bed.rejected(), 0);
    }

    /// A replica under a shared lease that follows, over the file it
    /// started with: its watch and its election, the take's step wired.
    fn standby(bed: &Bed) -> (SpecFile, Elector) {
        let mut elector = election();
        bed.generations.under(elector.fence());
        let file = SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        let door = Arc::clone(&bed.generations);
        elector.at_take(move || door.at_take());
        (file, elector)
    }

    /// The file moves to `document`, and the watch of a replica that
    /// follows shows it to the door: two looks, nothing applied and nothing
    /// said.
    fn shown(bed: &Bed, file: &mut SpecFile, document: &[u8]) {
        let before = (bed.generation(), bed.rejected());
        replace(&bed.path, document);
        assert_eq!(file.poll(), Look::Settling);
        assert_eq!(file.poll(), Look::Shown);
        assert_eq!(file.poll(), Look::Same, "shown once");
        assert_eq!((bed.generation(), bed.rejected()), before);
    }

    /// A file that states its own generation stood only under that number:
    /// a digest reads no line. The standby's file states 10, and its
    /// leader runs the same document as generation 1 and then replaces it.
    /// The take reads the file, and its higher claim runs.
    #[test]
    fn a_numbered_file_above_the_generations_it_followed_runs_at_the_take() {
        let numbered = with_line(LEAST_LOAD, 10);
        let bed = bed(&numbered);
        let (mut file, mut elector) = standby(&bed);
        follow(&bed, &[LEAST_LOAD]);
        follow(&bed, &[LEAST_LOAD, RANDOM]);
        assert_eq!(bed.generation(), 2);
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(10)));
        assert_eq!(bed.answered().2, numbered);
        assert_eq!(bed.rejected(), 0);
    }

    /// A file the validator refuses never stood, whatever its digest. The
    /// standby's file gains a `kind` line, which no digest reads, under the
    /// document its replica runs. The take reads the file and tells the
    /// refusal once, and the running generation stays.
    #[test]
    fn a_file_the_validator_refuses_is_told_at_the_take_though_its_digest_stands() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let (mut file, mut elector) = standby(&bed);
        follow(&bed, &[LEAST_LOAD]);
        shown(
            &bed,
            &mut file,
            format!("kind: PolicySpec\n{LEAST_LOAD}").as_bytes(),
        );
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(
            settle(&mut file),
            Err(SpecRejected::Invalid(SpecError::Kind(
                "PolicySpec".to_owned()
            )))
        );
        assert_eq!((bed.rejected(), bed.generation()), (1, 1));
        assert_eq!(file.poll(), Look::Same, "told once");
    }

    /// A generation that enters while a standby passes its file leaves the
    /// file's document one that stood: the pass found it standing. The
    /// file is a mount one step behind, and the take leaves it.
    #[test]
    fn a_generation_that_enters_under_a_standbys_pass_leaves_the_file_stood() {
        thread_local! {
            static DOOR: RefCell<Option<Arc<Generations>>> = const { RefCell::new(None) };
        }
        // The pass of the file's document brings the leader's next
        // generation first, once: it enters under the look's pass.
        fn entered_under(
            document: &PolicySpec,
            digest: Digest,
            catalog: &StageRegistry,
            denied_headers: &[&str],
        ) -> Result<AppliedSpec, SpecError> {
            if let Some(door) = DOOR.with(|door| door.borrow_mut().take()) {
                follow_at(&door, &[LEAST_LOAD, RANDOM]);
            }
            AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
        }
        let bed = bed_validating(LEAST_LOAD.as_bytes(), entered_under);
        let (mut file, mut elector) = standby(&bed);
        follow(&bed, &[LEAST_LOAD]);
        DOOR.with(|door| *door.borrow_mut() = Some(Arc::clone(&bed.generations)));
        // The mount turns to the same document, and the look reads it.
        replace(&bed.path, LEAST_LOAD.as_bytes());
        assert_eq!(file.poll(), Look::Settling);
        assert_eq!(file.poll(), Look::Shown);
        assert_eq!(bed.generation(), 2, "it entered under the pass");
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(file.poll(), Look::Same, "a mount one step behind");
        assert_eq!(bed.generation(), 2);
        assert_eq!(bed.answered().2, RANDOM.as_bytes());
    }

    /// A file that gains a number while its replica follows has not stood
    /// under it. The standby's file takes the running document under the
    /// line 10: the take reads it and runs it under that number.
    #[test]
    fn a_file_that_gains_a_number_while_its_replica_follows_runs_at_the_take() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let (mut file, mut elector) = standby(&bed);
        follow(&bed, &[LEAST_LOAD]);
        let numbered = with_line(LEAST_LOAD, 10);
        shown(&bed, &mut file, &numbered);
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(10)));
        assert_eq!(bed.answered().2, numbered);
    }

    /// A mount one step behind. The standby's file takes the first of two
    /// changes, and its leader applies both. That document stood on the
    /// replica since the file took it, so the take leaves the file: nothing
    /// older comes back. The file's next move is read as a leader's is.
    #[test]
    fn a_mount_one_step_behind_brings_nothing_older_back_at_a_take() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let (mut file, mut elector) = standby(&bed);
        follow(&bed, &[LEAST_LOAD]);
        // The first change reaches this mount ahead of its leader's word of
        // it. The second reaches the leader's mount alone.
        shown(&bed, &mut file, RANDOM.as_bytes());
        follow(&bed, &[LEAST_LOAD, RANDOM]);
        follow(&bed, &[LEAST_LOAD, RANDOM, &bound_to(1)]);
        assert_eq!(bed.generation(), 3);
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(file.poll(), Look::Same, "its document stood here since");
        assert_eq!(file.poll(), Look::Same);
        assert_eq!(
            (bed.generation(), bed.answered().2),
            (3, bound_to(1).into())
        );
        assert_eq!(bed.rejected(), 0);
        // The mount's sync is a move: it is read, the standing document
        // again.
        replace(&bed.path, bound_to(1).as_bytes());
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Unchanged(3)));

        // The same where the leader's word came first: the file takes the
        // document that stands, and its leader replaces it.
        let bed = self::bed(LEAST_LOAD.as_bytes());
        let (mut file, mut elector) = standby(&bed);
        follow(&bed, &[LEAST_LOAD, RANDOM]);
        shown(&bed, &mut file, RANDOM.as_bytes());
        follow(&bed, &[LEAST_LOAD, RANDOM, &bound_to(1)]);
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(file.poll(), Look::Same);
        assert_eq!(
            (bed.generation(), bed.answered().2),
            (3, bound_to(1).into())
        );
    }

    /// A rollback that reaches the standby's mount first. Its leader ran a
    /// document and replaced it, and then the standby's file takes the
    /// first one again. That document has not stood on the replica since
    /// the file took it: it is the newer word, and the take runs it as the
    /// next generation. The bytes are those of a mount that lags: the
    /// order of the hand and the generations tells the two apart.
    #[test]
    fn a_rollback_that_reaches_the_standby_first_runs_at_the_take() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let (mut file, mut elector) = standby(&bed);
        follow(&bed, &[LEAST_LOAD]);
        shown(&bed, &mut file, RANDOM.as_bytes());
        follow(&bed, &[LEAST_LOAD, RANDOM]);
        // The rollback: the file holds the first document again, behind the
        // generation that replaced it.
        shown(&bed, &mut file, LEAST_LOAD.as_bytes());
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(3)));
        assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
        assert_eq!(file.poll(), Look::Same);
    }

    /// A replica that follows says nothing of a file it cannot read or
    /// parse: its look shows the door a file with no document, and no
    /// rejected state is told. The take reads the file again and tells it
    /// once, as a leader does, and the replica keeps what it followed.
    #[test]
    fn a_file_a_standby_cannot_read_or_parse_is_told_once_at_the_take() {
        for gone in [false, true] {
            let bed = bed(LEAST_LOAD.as_bytes());
            let (mut file, mut elector) = standby(&bed);
            follow(&bed, &[RANDOM]);
            if gone {
                fs::remove_file(&bed.path).expect("the file goes");
                assert_eq!(file.poll(), Look::Settling);
                assert_eq!(file.poll(), Look::Shown);
            } else {
                shown(&bed, &mut file, b"{ no policy");
            }
            assert_eq!(file.poll(), Look::Same);
            assert_eq!(bed.rejected(), 0, "a standby's silence is no verdict");
            elector.tick(Instant::now()).expect("a take");
            let told = settle(&mut file);
            if gone {
                assert!(matches!(told, Err(SpecRejected::Unreadable(_))), "{told:?}");
            } else {
                assert!(
                    matches!(told, Err(SpecRejected::Unparsed { .. })),
                    "{told:?}"
                );
            }
            assert_eq!(bed.rejected(), 1, "told once, at the take");
            assert_eq!(file.poll(), Look::Same, "one event under one lead");
            assert_eq!((bed.generation(), bed.answered().2), (1, RANDOM.into()));
        }
    }

    /// The look of a replica that follows publishes nothing, and none of
    /// the bytes it read are published later: the door takes their digest
    /// and keeps no byte. The take reads the file as it stands then, under
    /// its own term.
    #[test]
    fn a_standbys_look_publishes_nothing_and_its_bytes_are_never_published() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let (mut file, mut elector) = standby(&bed);
        follow(&bed, &[RANDOM]);
        shown(&bed, &mut file, bound_to(1).as_bytes());
        assert_eq!((bed.generation(), bed.answered().2), (1, RANDOM.into()));
        // The file moves again, and no look sees it ahead of the take.
        replace(&bed.path, bound_to(2).as_bytes());
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
        assert_eq!(bed.answered().2, bound_to(2).as_bytes());
    }

    /// What a document's own number does to a mount that lags. A file whose
    /// document never stood on the replica counts as the newer word: a
    /// replica that started over an older file runs it at its take. Where
    /// that document states its generation, a number at or under the
    /// standing one is refused by name and the standing generation stays:
    /// an author who numbers the documents gets no older one back.
    #[test]
    fn a_numbered_file_under_the_standing_generation_is_refused_at_the_take() {
        // No number: the older file comes back, as the next generation.
        let bed = bed(LEAST_LOAD.as_bytes());
        let (mut file, mut elector) = standby(&bed);
        follow(&bed, &[RANDOM, &bound_to(1)]);
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(3)));
        assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());

        // Numbered 1 under its leader's generation 2: refused, told once.
        let older = with_line(LEAST_LOAD, 1);
        let bed = self::bed(&older);
        let (mut file, mut elector) = standby(&bed);
        follow(&bed, &[RANDOM, &bound_to(1)]);
        elector.tick(Instant::now()).expect("a take");
        let refused = settle(&mut file);
        assert!(
            matches!(refused, Err(SpecRejected::Stale(_))),
            "{refused:?}"
        );
        assert_eq!((bed.generation(), bed.rejected()), (2, 1));
        assert_eq!(bed.answered().2, bound_to(1).as_bytes());
        assert_eq!(file.poll(), Look::Same);
    }

    /// A file that never rests under a replica that follows is told by no
    /// look of it: the count waits one short of the event. The first look
    /// of the replica's lead that finds the file still moving tells it,
    /// once.
    #[test]
    fn a_restless_file_is_told_by_no_look_of_a_replica_that_follows() {
        let bed = bed(LEAST_LOAD.as_bytes());
        let (mut file, mut elector) = standby(&bed);
        follow(&bed, &[RANDOM]);
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
        for _ in 0..3 * RESTLESS_LOOKS {
            assert_eq!(restless(&mut file), Look::Settling);
        }
        assert_eq!(bed.rejected(), 0, "a standby says nothing of its file");
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(restless(&mut file), Look::Restless, "its lead's first look");
        assert_eq!(restless(&mut file), Look::Settling);
        assert_eq!((bed.rejected(), bed.generation()), (1, 1));
    }

    /// A file a leader's look cannot run is told once under that lead. The
    /// door knows the file holds no document that stood, so behind the
    /// replica's next take the file is read and told again, though it never
    /// moved.
    #[test]
    fn a_file_a_leader_cannot_run_is_told_once_under_each_lead() {
        for gone in [false, true] {
            let bed = bed(LEAST_LOAD.as_bytes());
            let (mut file, mut elector) = standby(&bed);
            elector.tick(Instant::now()).expect("a take");
            assert_eq!(bed.generation(), 1, "the start's document, at the take");
            assert_eq!(file.poll(), Look::Same);
            if gone {
                fs::remove_file(&bed.path).expect("the file goes");
            } else {
                replace(&bed.path, b"{ no policy");
            }
            for told in 1..=2 {
                let refused = settle(&mut file);
                assert!(refused.is_err(), "{refused:?}");
                assert_eq!(bed.rejected(), told);
                assert_eq!(file.poll(), Look::Same, "one event under one lead");
                // Deposed, and back by a take: the file never moved.
                assert!(elector.release().expect("a release"), "it held the lease");
                assert_eq!(file.poll(), Look::Same, "nothing new to show");
                elector.tick(Instant::now()).expect("a take again");
            }
            assert_eq!(bed.generation(), 1);
        }
    }

    /// A take that never led publishes nothing, so its replica follows the
    /// next leader under any number, the first included: no number of its
    /// own stands in the way.
    #[test]
    fn a_take_past_its_window_publishes_nothing_and_follows_any_number() {
        let deadline = Duration::from_secs(1);
        let bed = bed(LEAST_LOAD.as_bytes());
        let mut elector = election_under(deadline);
        bed.generations.under(elector.fence());
        SpecFile::start_deferred(bed.path.clone(), Arc::clone(&bed.generations))
            .expect("a file that runs");
        let door = Arc::clone(&bed.generations);
        elector.at_take(move || door.at_take());
        let late = Instant::now()
            .checked_sub(2 * deadline)
            .expect("a host that is up for two seconds");
        elector.tick(late).expect("a take");
        assert_eq!(bed.generation(), 0, "no term showed at the take's step");
        // Another replica leads meanwhile, under generation 1 of another
        // document.
        follow(&bed, &[RANDOM]);
        assert_eq!(bed.generation(), 1);
        assert_eq!(bed.answered().2, RANDOM.as_bytes());
        assert_eq!(bed.rejected(), 0);
    }
}
