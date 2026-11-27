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
    /// The start: reads the file at `path` as it stands and hands its
    /// document to the door, which knows the start its replica makes
    /// ([`Generations::start_with`]). A replica alone over its lease
    /// applies the document as the source's first. A file that cannot be
    /// read, or a document that does not run, refuses the start.
    ///
    /// A replica under a shared lease passes the document and never
    /// applies it at the start. A file that cannot be read, or a document
    /// that does not run, refuses the start all the same. Until it leads,
    /// a replica that follows a leader runs what the leader's log gives
    /// it. Under its first lead the door applies these bytes where the
    /// replica runs nothing, at the take or behind the lead's opening
    /// ([`Generations::deferred_now`]). Behind a take the watch reads the
    /// file as it stands then, though it never moved, while its document
    /// has not stood on this replica since the file took it: a replica
    /// that followed another document runs its own file's once it leads.
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
        look: impl FnMut(&Path) -> Seen,
    ) -> Result<Self, SpecRejected> {
        Self::start_reading(path, generations, look, read)
    }

    /// [`SpecFile::start_looking`] with the descriptor read as its
    /// argument: the start's read, and the document handed to the door.
    fn start_reading(
        path: PathBuf,
        generations: Arc<Generations>,
        mut look: impl FnMut(&Path) -> Seen,
        mut read: impl FnMut(&Identity) -> Option<Result<Vec<u8>, SpecRejected>>,
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
                generations.start_with(document?, format)?;
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
#[path = "../../../tests/unit/service/controller/spec_file/tests.rs"]
pub(crate) mod tests;
