//! The policy spec's store (design 4.6): the directory an applied document
//! is persisted in before it runs, where the spec's source is the apply
//! call.
//!
//! A generation is one file, named by its number and its form
//! (`00000000000000000042.yaml`). The bytes are the document as its caller
//! sent it: the number lives in the name, never inside those bytes. A file
//! is written whole under another name and synced, then linked under its
//! generation's name, the other name removed and the directory synced, in
//! that order: a name of that shape holds a whole document or does not
//! exist. A link takes a name once and replaces none, so no write puts a
//! file over a generation on file, whoever wrote it; a number on file in
//! the other form is refused where the write finds it. The door
//! ([`DocumentManager`]) persists a document here and only then calls the
//! store's entry, one apply at a time: one process writes the directory,
//! under the door's lock alone.
//!
//! One process holds a directory: a store takes the lock of a file it keeps
//! there (`.lock`) before it reads a name, and holds it until it is
//! dropped; the system lets it go at any exit. The store holds the
//! directory itself, its path resolved once at the open: the lock file, the
//! names, every read, every write and every removal go through that
//! directory, wherever the path leads later, and a write is refused where
//! the path leads to another one as it begins. A start that finds the lock
//! held waits a bound for its holder to stop and takes the directory then.
//! One whose directory was renamed under that wait is refused once the lock
//! is held: by name where another is laid at its path, which a new start
//! takes, and as a path to no directory where its path leads to none. One
//! still held at the bound's end is refused by name, and one whose lock the
//! system does not give is refused in the system's own words: the lock asks
//! that its file is there or can be made, and no write of it on a file
//! system that locks as a local one does. So a start reads the directory
//! once the process before it writes no more, and runs that one's last
//! apply. The lock is on the file, not on its name: the file is made once
//! and never removed, and one removed under a running store lets a second
//! one in. It is the open file's, too: a child the process forks holds a
//! copy until it runs its program, so the lock may outlast a drop by that
//! moment, and a start that follows a drop in one process waits like any
//! other. The directory stands on a file system that links and locks. Where
//! no link is made, every write is refused at the link in the system's own
//! words, and nothing of it stays on file.
//!
//! The lock holds two apart as far as the file system's locks reach:
//! between two hosts of a network file system only where it carries them.
//! A second process such a directory lets in writes it too, and the write
//! itself keeps the two from one number. Its apply of a number the first
//! has on file is refused by name, and from then on it takes no write
//! until it starts again: a store reads its directory at its start alone,
//! so that process runs the generation it started with. Two that write one
//! number at one moment in the two forms never both stay: each looks for
//! the other form's name after its own link and backs out where it stands,
//! so one lands or neither. A write that backs out says so, apart from a
//! number that is on file: where both back out the number is free, and the
//! next apply of either lands. A start removes the temporary names it may,
//! a live writer's among them: that write is refused, unless its link was
//! made already. A write refused after its link removes its name and syncs
//! the directory again, and says which of the two failed. A name that stays
//! may run at a start, or refuse it beside the other form's name. A name
//! that is gone where the directory did not sync may come back after a
//! crash. A generation the store's entry refuses once it is on file goes
//! the same way, told the same. One whose name stays is no generation of
//! this store's from then on: no read lists it, and the next write tries
//! its removal first. A process that starts while the name stands cannot
//! tell it from a generation that ran: it reads it as one on file, and
//! runs it where it is the newest.
//!
//! A start reads the directory: the highest number is the generation to
//! run, and a file a write left under its temporary name is removed, or
//! left in place where this user may not remove it. One that is gone by
//! then is taken as removed, and a directory under such a name refuses the
//! start. Any other name is left alone, the lock file's among them. Two
//! files under one number, a directory another process holds, one renamed
//! under the start's wait, one with no lock file where none can be made, or
//! a path that leads to no directory, refuse the start by name. The
//! [`RETAINED`] highest numbers are kept; an apply removes the files under
//! them. A generation on file is read by the read call and by a rollback;
//! the read call reads its file through the directory itself ([`read_at`]),
//! under no lock of the door.
//!
//! [`DocumentManager`]: super::DocumentManager

#![cfg_attr(
    not(any(test, feature = "controller")),
    expect(dead_code, reason = "Only a host with a spec store opens a directory.")
)]

use std::{
    fs,
    io::{self, Read as _, Write as _},
    path::{Path, PathBuf},
    process,
    sync::Arc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use tracing::{info, warn};

use super::{Format, SpecRejected, MAX_DOCUMENT_BYTES};

/// What a read says of a path that leads to no regular file.
pub(crate) const NOT_A_FILE: &str = "the path leads to no regular file";

/// How many generations a store keeps: the standing one and the seven
/// before it.
pub(crate) const RETAINED: usize = 8;

/// The first letters of a name a write is in flight under: no start reads
/// it.
const WRITING: &str = ".writing-";

/// The name of the lock file in a store's directory: no generation's name
/// and no temporary one, so a start neither reads it nor removes it.
const HELD: &str = ".lock";

/// Between two tries of a lock another process holds.
const HELD_PACE: Duration = Duration::from_millis(50);

/// What a write is refused by where the store's path leads to another
/// directory than the one it holds.
const MOVED: &str = "the path leads to another directory than the one this store holds";

/// How many digits a generation's file name writes its number in: the
/// widest number's.
const DIGITS: usize = 20;

/// Why a spec store did not become a controller's source.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum StoreRefused {
    /// The path leads to a file, through one, or nowhere.
    #[error("the path leads to no directory")]
    NotADirectory,
    /// The directory could not be read.
    #[error("not read: {0}")]
    Unreadable(String),
    /// Another controller holds the directory's lock, past the wait the
    /// start was given.
    #[error(
        "another controller holds the directory, waited on for {0:?}: one controller holds a spec store"
    )]
    Held(Duration),
    /// The directory's lock was not taken, in the system's own words: the
    /// lock file does not open, or the file system takes no lock.
    #[error("not locked: {0}")]
    Unlocked(String),
    /// The directory holds no lock file and this start may not make one,
    /// in the system's words for the making: a store's first start makes
    /// the file, and needs a directory it may write.
    #[error(
        "no lock file, and none can be made: {0}: a store's first start needs a directory it may write"
    )]
    Unmade(String),
    /// The directory moved while the start waited for its lock: by the
    /// time the lock is held the path leads to another one, which a new
    /// start takes.
    #[error(
        "the directory moved while this start waited for its lock: the path leads to another one, which a new start takes"
    )]
    Moved,
    /// Two files, one in each form, are named by one number.
    #[error("generation {0} stands under two names")]
    TwoNames(u64),
    /// The newest generation on file does not run.
    #[error("generation {generation}: {source}")]
    Rejected {
        /// The generation, by its file's name.
        generation: u64,
        /// Why its document was not applied.
        #[source]
        source: SpecRejected,
    },
    /// The election's lease is one another replica can take.
    #[error(
        "another replica can take this controller's lease, and a standby follows a file's generations alone in this build: keep the lease in the process, or hand the spec as a file"
    )]
    #[cfg(feature = "controller")]
    BesideASharedLease,
    /// The controller took a spec source before this one.
    #[error("the controller has a spec source already")]
    SecondSource,
}

/// One generation on file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Persisted {
    pub(crate) generation: u64,
    pub(crate) format: Format,
}

/// The generation a file's name states, where it is a generation's name.
fn named(name: &str) -> Option<Persisted> {
    let (number, form) = name.split_once('.')?;
    let format = match form {
        "yaml" => Format::Yaml,
        "json" => Format::Json,
        _ => return None,
    };
    let digits = number.len() == DIGITS && number.bytes().all(|byte| byte.is_ascii_digit());
    let generation = number.parse().ok().filter(|&number| digits && number > 0)?;
    Some(Persisted { generation, format })
}

/// The name of the file of `persisted` in its directory.
fn file_name(persisted: Persisted) -> String {
    let Persisted { generation, format } = persisted;
    let form = format.as_str();
    format!("{generation:0DIGITS$}.{form}")
}

/// Whether a name stands in the directory `root`, asked through the
/// directory itself. A link that leads nowhere stands for no name.
fn on_file(root: &fs::File, name: &str) -> io::Result<bool> {
    match rustix::fs::statat(root, name, rustix::fs::AtFlags::empty()) {
        Ok(_) => Ok(true),
        Err(rustix::io::Errno::NOENT) => Ok(false),
        Err(errno) => Err(errno.into()),
    }
}

/// Removes `name` from the directory `root`, through the directory itself.
/// A name that is gone already is taken as removed, which is what the
/// removal wanted: another hand took it away, a second process among them
/// where the file system's locks do not hold two apart.
fn remove_at(root: &fs::File, name: &str) -> io::Result<()> {
    match rustix::fs::unlinkat(root, name, rustix::fs::AtFlags::empty()) {
        Ok(()) | Err(rustix::io::Errno::NOENT) => Ok(()),
        Err(errno) => Err(errno.into()),
    }
}

/// The file under `name` in the directory `root`, opened through the
/// directory itself and without waiting. What is no regular file is
/// refused by its kind, unread: a pipe under the name holds no call.
fn open_at(
    root: &fs::File,
    name: &str,
    flags: rustix::fs::OFlags,
    mode: rustix::fs::Mode,
) -> io::Result<fs::File> {
    use rustix::fs::OFlags;

    let quiet = flags | OFlags::NONBLOCK | OFlags::NOCTTY | OFlags::CLOEXEC;
    let opened = rustix::io::retry_on_intr(|| rustix::fs::openat(root, name, quiet, mode))?;
    let file = fs::File::from(opened);
    if !file.metadata()?.is_file() {
        return Err(io::Error::other(NOT_A_FILE));
    }
    Ok(file)
}

/// The bytes on file for `persisted` in the directory `root`, read through
/// the directory itself, one past the size bound at most: a longer file is
/// refused by its size, not read whole. What is no regular file is refused
/// by its kind, unread ([`open_at`]): a pipe under a generation's name
/// holds no start and no read call.
pub(crate) fn read_at(root: &fs::File, persisted: Persisted) -> io::Result<Vec<u8>> {
    use rustix::fs::{Mode, OFlags};

    let mut document = Vec::new();
    open_at(root, &file_name(persisted), OFlags::RDONLY, Mode::empty())?
        .take(MAX_DOCUMENT_BYTES + 1)
        .read_to_end(&mut document)?;
    Ok(document)
}

/// The form a generation's file is not in.
fn other_form(format: Format) -> Format {
    match format {
        Format::Yaml => Format::Json,
        Format::Json => Format::Yaml,
    }
}

/// What a write says where the other form's name stood after its own link:
/// it backs out. The number is not on file in this write's form, and in
/// the other only where that writer did not back out too.
fn backs_out(generation: u64) -> io::Error {
    let text = format!(
        "generation {generation} was written in the other form as this write was: this write backs out"
    );
    io::Error::new(io::ErrorKind::AlreadyExists, text)
}

/// What a refused generation leaves where its name did not come off file
/// again, said beside its refusal, a write's own failure or the store's
/// entry's: the removal's failure, then what a start may meet. A start
/// runs the newest generation on file, which this one may be, and beside
/// the other form's name the two refuse a start.
fn stays(generation: u64, left: &io::Error) -> String {
    format!(
        "generation {generation} was not taken off file again ({left}): a start may run it, and \
         beside the other form's name the two refuse a start"
    )
}

/// What a refused generation leaves where its name came off file again and
/// the directory did not sync after it, said beside its refusal: the
/// sync's failure. The name is gone, and a crash may bring it back.
fn unsynced(generation: u64, left: &io::Error) -> String {
    format!(
        "generation {generation} was taken off file again, but the directory did not sync \
         ({left}): a crash may bring its name back"
    )
}

/// The store: the directory and the generations on file in it.
#[derive(Debug)]
pub(crate) struct SpecStore {
    /// The directory's path, resolved once at the open: a link in it that
    /// turns later leads this store nowhere else.
    dir: PathBuf,
    /// The directory itself, open: the lock file, every read, every write
    /// and every removal go through it, and a write is refused where the
    /// path leads to another directory as it begins. The read call holds
    /// it too.
    root: Arc<fs::File>,
    /// The lock file, open: the directory's lock is held until the store
    /// is dropped.
    _held: fs::File,
    /// The generations on file, ascending.
    kept: Vec<Persisted>,
    /// The generations the store's entry refused whose names did not come
    /// off file ([`SpecStore::forget`]): none is among the kept, and the
    /// next write tries their removal first.
    refused: Vec<Persisted>,
    /// When this store was opened, in nanoseconds: with the process's id it
    /// tells this store's temporary names from another's on the directory.
    /// A process id alone does not: it is 1 in each of two containers.
    opened: u128,
    /// The writes this store began: a temporary name's own number.
    writes: u64,
}

impl SpecStore {
    /// [`SpecStore::open_within`] as the cases open theirs: a start that
    /// waits a case's room, as the controller's waits its grace period. A
    /// directory whose store a case dropped is free a moment later where
    /// another case's child process holds a copy of the lock file.
    #[cfg(test)]
    pub(crate) fn open(dir: PathBuf) -> Result<Self, StoreRefused> {
        Self::open_within(dir, Duration::from_secs(30))
    }

    /// Takes the directory at `dir` and reads it: the generations on file.
    /// A file left under a temporary name is removed, its write never
    /// ended, or left in place where this user may not remove it. Where
    /// another process holds the directory's lock it is tried again every
    /// [`HELD_PACE`]: the directory is taken once the holder lets go, and
    /// a start still held when `wait` has passed is refused by name. A
    /// process that stops lets the lock go at its exit, so a start given
    /// the time a stop takes reads the directory after that process's last
    /// write. A directory renamed while the start waited is refused once
    /// the lock is held: by name where another is laid at its path
    /// ([`StoreRefused::Moved`]), which a new start takes, and as a path to
    /// no directory where none is.
    pub(crate) fn open_within(dir: PathBuf, wait: Duration) -> Result<Self, StoreRefused> {
        let (dir, root) = Self::rooted(&dir)?;
        let held = Self::hold(&dir, &root, wait)?;
        Self::named_still(&dir, &root)?;
        Self::read_under(dir, root, held)
    }

    /// A second store on a directory another holds, its lock file open and
    /// not locked, as a file system whose locks do not hold between two
    /// hosts lets one in: the cases of two writers open theirs so.
    #[cfg(test)]
    pub(crate) fn open_beside(dir: PathBuf) -> Result<Self, StoreRefused> {
        let (dir, root) = Self::rooted(&dir)?;
        let unheld = Self::lock_file(&root)?;
        Self::read_under(dir, root, unheld)
    }

    /// What the system reads at a path that leads to a directory. A path
    /// that leads to none is refused by that name: one to a file, one
    /// through a file, one to nothing.
    fn found(dir: &Path) -> Result<fs::Metadata, StoreRefused> {
        match fs::metadata(dir) {
            Ok(found) if found.is_dir() => Ok(found),
            Ok(_) => Err(StoreRefused::NotADirectory),
            Err(error) => match error.kind() {
                // Nothing is there, or a file stands in the path's way.
                io::ErrorKind::NotFound | io::ErrorKind::NotADirectory => {
                    Err(StoreRefused::NotADirectory)
                }
                _ => Err(StoreRefused::Unreadable(error.to_string())),
            },
        }
    }

    /// The directory at `dir`, its path resolved and itself open: what the
    /// store locks, reads and writes from here on. A link in the path that
    /// turns while a start waits leads that start nowhere else.
    fn rooted(dir: &Path) -> Result<(PathBuf, fs::File), StoreRefused> {
        use rustix::fs::{Mode, OFlags};

        Self::found(dir)?;
        let unreadable = |error: io::Error| StoreRefused::Unreadable(error.to_string());
        let dir = fs::canonicalize(dir).map_err(unreadable)?;
        let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC;
        let root = rustix::io::retry_on_intr(|| rustix::fs::open(&dir, flags, Mode::empty()))
            .map_err(|errno| unreadable(errno.into()))?;
        Ok((dir, root.into()))
    }

    /// Whether an open to read serves where the open to read and write was
    /// refused as `refused`: the user's bits and a file system mounted to
    /// read alone refuse the write alone, and the lock asks none.
    fn read_serves(refused: &io::Error) -> bool {
        matches!(
            refused.kind(),
            io::ErrorKind::PermissionDenied | io::ErrorKind::ReadOnlyFilesystem
        )
    }

    /// The lock file of the directory `root`, open: made empty where it is
    /// not there, and never truncated, since what stands in it is no part
    /// of the lock. It is opened through the directory the store holds, so
    /// the lock is that directory's, whatever its path leads to by then.
    /// Neither open waits, and what is no regular file is refused by its
    /// kind: a pipe under the name holds no start and takes no lock. On a
    /// file system that locks as a local one does, the lock asks no write
    /// of the file: one that stands under bits this user may not write, as
    /// after a change of the controller's user, or on a file system
    /// mounted to read alone, is opened to read. Where a lock is made of a
    /// byte-range lock, as a network file system's client may make it, it
    /// asks a file opened to write: the lock of a file this user may not
    /// write is not given there, and the refusal is the system's for the
    /// lock. Where the file may not be made and the open to read fails
    /// too, the file not there or no regular file, the refusal is the
    /// system's for the making, and where the file is not there it says
    /// what a first start needs.
    fn lock_file(root: &fs::File) -> Result<fs::File, StoreRefused> {
        use rustix::fs::{Mode, OFlags};

        let open = |flags: OFlags, mode: Mode| open_at(root, HELD, flags, mode);
        let made = open(OFlags::RDWR | OFlags::CREATE, Mode::from_raw_mode(0o666));
        let unlocked = |error: io::Error| StoreRefused::Unlocked(error.to_string());
        match made {
            Err(denied) if Self::read_serves(&denied) => {
                open(OFlags::RDONLY, Mode::empty()).map_err(|read| match read.kind() {
                    // No lock file is there, and this start may not make one.
                    io::ErrorKind::NotFound => StoreRefused::Unmade(denied.to_string()),
                    _ => unlocked(denied),
                })
            }
            opened => opened.map_err(unlocked),
        }
    }

    /// The directory's lock: an exclusive advisory lock on the lock file,
    /// tried every [`HELD_PACE`] until `wait` has passed. The system lets
    /// the lock go when the file is dropped, and at any exit.
    fn hold(dir: &Path, root: &fs::File, wait: Duration) -> Result<fs::File, StoreRefused> {
        let lock = Self::lock_file(root)?;
        let started = Instant::now();
        let mut told = false;
        loop {
            match lock.try_lock() {
                Ok(()) => return Ok(lock),
                Err(fs::TryLockError::WouldBlock) => {
                    let Some(left) = wait.checked_sub(started.elapsed()) else {
                        return Err(StoreRefused::Held(wait));
                    };
                    if !told {
                        info!(
                            path = %dir.display(),
                            ?wait,
                            "spec store held by another controller: waiting for it to stop"
                        );
                        told = true;
                    }
                    thread::sleep(left.min(HELD_PACE));
                }
                Err(fs::TryLockError::Error(error)) => {
                    return Err(StoreRefused::Unlocked(error.to_string()))
                }
            }
        }
    }

    /// Removes the temporary name `name` from the directory `root`: a
    /// write that never ended. A name that is gone already is taken as
    /// removed ([`remove_at`]): a live writer took its own name away after
    /// its link. A file this user may not remove, by its bits or on a file
    /// system mounted to read alone, is left in place: no start reads it.
    /// Any other refusal refuses the start, and so does a directory under
    /// such a name, however the system words the refusal of its removal: by
    /// its kind, or as a permission's on some systems and behind the
    /// store's bits on all.
    fn clear(root: &fs::File, name: &str) -> io::Result<()> {
        use rustix::{
            fs::{AtFlags, FileType},
            io::Errno,
        };

        match remove_at(root, name) {
            Err(refused) if Self::read_serves(&refused) => {
                match rustix::fs::statat(root, name, AtFlags::SYMLINK_NOFOLLOW) {
                    Ok(left) if FileType::from_raw_mode(left.st_mode) == FileType::Directory => {
                        Err(Errno::ISDIR.into())
                    }
                    Ok(_) | Err(Errno::NOENT) => Ok(()),
                    Err(errno) => Err(errno.into()),
                }
            }
            done => done,
        }
    }

    /// Reads the directory `root`, whose path is `dir`, under `held`, its
    /// lock file. The names are read through the directory itself. A
    /// temporary name's write never ended: [`Self::clear`] removes it, or
    /// says what stays in place and what refuses the start.
    fn read_under(dir: PathBuf, root: fs::File, held: fs::File) -> Result<Self, StoreRefused> {
        let unreadable = |error: io::Error| StoreRefused::Unreadable(error.to_string());
        let mut kept = Vec::new();
        let names = rustix::fs::Dir::read_from(&root).map_err(|errno| unreadable(errno.into()))?;
        for entry in names {
            let entry = entry.map_err(|errno| unreadable(errno.into()))?;
            let Ok(name) = entry.file_name().to_str() else {
                continue;
            };
            if name.starts_with(WRITING) {
                Self::clear(&root, name).map_err(unreadable)?;
            } else if let Some(persisted) = named(name) {
                kept.push(persisted);
            }
        }
        kept.sort_by_key(|persisted| persisted.generation);
        if let Some(pair) = kept
            .windows(2)
            .find(|pair| pair[0].generation == pair[1].generation)
        {
            return Err(StoreRefused::TwoNames(pair[0].generation));
        }
        let opened = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        Ok(Self {
            dir,
            root: Arc::new(root),
            _held: held,
            kept,
            refused: Vec::new(),
            opened,
            writes: 0,
        })
    }

    /// Refuses where the store's path leads to another directory than the
    /// one it holds, or to none: an apply there would stand on file where
    /// no start by that path reads it. The path is asked as a start asks
    /// it: a file laid there or in its way, or nothing, is a path to no
    /// directory.
    fn same(&self) -> io::Result<()> {
        // The refusal goes on as its words alone. Boxed whole it kept its
        // debug text in the release binary: 20 KB for this one line.
        let at_path =
            Self::found(&self.dir).map_err(|refused| io::Error::other(refused.to_string()))?;
        if Self::is_held(&at_path, &self.root)? {
            Ok(())
        } else {
            Err(io::Error::other(MOVED))
        }
    }

    /// Whether `at_path`, what the system read at a path, is the directory
    /// `root`, by the system's own identity of the two.
    fn is_held(at_path: &fs::Metadata, root: &fs::File) -> io::Result<bool> {
        use std::os::unix::fs::MetadataExt as _;

        let held = root.metadata()?;
        Ok((at_path.dev(), at_path.ino()) == (held.dev(), held.ino()))
    }

    /// Refuses a start whose path leads to another directory than the one
    /// it locked, or to none, once the lock is held: the directory was
    /// renamed or removed while the start waited, and the one held is no
    /// longer the one its path names. The path is asked once, as a new
    /// start asks it, and the two identities are compared on that answer:
    /// a file laid there is a path to no directory, as it is to that
    /// start. A link in the path that turned passes: the path was resolved
    /// once, before the wait.
    fn named_still(dir: &Path, root: &fs::File) -> Result<(), StoreRefused> {
        let at_path = Self::found(dir)?;
        match Self::is_held(&at_path, root) {
            Ok(true) => Ok(()),
            Ok(false) => Err(StoreRefused::Moved),
            Err(error) => Err(StoreRefused::Unreadable(error.to_string())),
        }
    }

    /// The directory's path.
    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    /// The generations on file, ascending.
    pub(crate) fn kept(&self) -> &[Persisted] {
        &self.kept
    }

    /// The directory itself, as the read call holds it: a generation on
    /// file is read through it under no lock of the door.
    pub(crate) fn root(&self) -> Arc<fs::File> {
        Arc::clone(&self.root)
    }

    /// The bytes on file for `persisted`, read through the directory this
    /// store holds ([`read_at`]): what a start listed there is what it
    /// reads, wherever the path leads by then.
    pub(crate) fn read(&self, persisted: Persisted) -> io::Result<Vec<u8>> {
        read_at(&self.root, persisted)
    }

    /// When the file of `persisted` was written: when its generation was
    /// applied. The file is asked through the directory this store holds.
    pub(crate) fn written(&self, persisted: Persisted) -> io::Result<SystemTime> {
        use rustix::fs::{Mode, OFlags};

        open_at(
            &self.root,
            &file_name(persisted),
            OFlags::RDONLY,
            Mode::empty(),
        )?
        .metadata()?
        .modified()
    }

    /// Removes the file of `persisted` from the directory this store
    /// holds, through the directory itself. One that is gone already is
    /// taken as removed.
    fn remove(&self, persisted: Persisted) -> io::Result<()> {
        remove_at(&self.root, &file_name(persisted))
    }

    /// Persists `document` as `persisted`: written whole under a temporary
    /// name and synced, linked under the generation's name, the temporary
    /// name removed, then the directory synced. A link replaces no name: a
    /// number on file already takes no write, in this form at the link and
    /// in the other where the write finds it, before its link or after it:
    /// two writers of one number in the two forms never both stay. A write
    /// that fails leaves no file of its own behind and removes no other's:
    /// a name it linked is removed and the directory synced again, and
    /// where that fails the write says so beside its own failure, since a
    /// start runs the newest generation on file. A write is refused where
    /// the store's path leads to another directory than the one it holds
    /// ([`SpecStore::same`]); behind that check every step of it goes
    /// through the directory held ([`SpecStore::write_held`]). Ahead of
    /// both, a generation the entry refused whose name stayed on file is
    /// tried again ([`SpecStore::forget_again`]).
    pub(crate) fn persist(&mut self, persisted: Persisted, document: &[u8]) -> io::Result<()> {
        self.persist_synced(persisted, document, fs::File::sync_all)
    }

    /// [`SpecStore::persist`] with the directory's sync as its argument: a
    /// case hands in one that fails.
    fn persist_synced(
        &mut self,
        persisted: Persisted,
        document: &[u8],
        mut sync: impl FnMut(&fs::File) -> io::Result<()>,
    ) -> io::Result<()> {
        self.forget_again(&mut sync);
        self.same()?;
        self.write_held(persisted, document, sync)
    }

    /// The write behind its check, through the directory this store holds:
    /// its looks for a name, its temporary name, its link, its removals
    /// and its syncs. A path that turns once the check has passed leads no
    /// step elsewhere: the write lands where the lock is held, as one a
    /// moment before the turn does, and a back-out removes this write's
    /// own names there and no other directory's.
    fn write_held(
        &mut self,
        persisted: Persisted,
        document: &[u8],
        mut sync: impl FnMut(&fs::File) -> io::Result<()>,
    ) -> io::Result<()> {
        use rustix::fs::{AtFlags, Mode, OFlags};

        let Persisted { generation, format } = persisted;
        let taken = || {
            let text = format!("generation {generation} is on file already");
            io::Error::new(io::ErrorKind::AlreadyExists, text)
        };
        let root = &*self.root;
        let name = file_name(persisted);
        let other = file_name(Persisted {
            generation,
            format: other_form(format),
        });
        if on_file(root, &name)? || on_file(root, &other)? {
            // Another process wrote the directory, or a refused
            // generation stayed on file.
            return Err(taken());
        }
        self.writes += 1;
        let (id, opened, write) = (process::id(), self.opened, self.writes);
        let writing = format!("{WRITING}{id}-{opened}-{write}");
        let mut linked = false;
        let written = (|| {
            // Opened to read and write, as the lock file is: the write
            // alone uses it.
            let made = OFlags::RDWR | OFlags::CREATE | OFlags::EXCL;
            let mut file = open_at(root, &writing, made, Mode::from_raw_mode(0o666))?;
            file.write_all(document)?;
            file.sync_all()?;
            drop(file);
            rustix::fs::linkat(root, &writing, root, &name, AtFlags::empty()).map_err(|errno| {
                match io::Error::from(errno) {
                    // The name appeared since the look above: its file stays.
                    error if error.kind() == io::ErrorKind::AlreadyExists => taken(),
                    error => error,
                }
            })?;
            linked = true;
            // A start beside this write may have taken the temporary name
            // away already: the link stands, and the write goes on.
            remove_at(root, &writing)?;
            // The other form's name since the look above. A writer of that
            // form looks for this one after its own link, so one of the two
            // backs out at least: the two never both stay.
            if on_file(root, &other)? {
                return Err(backs_out(generation));
            }
            sync(root)
        })();
        let Err(failed) = written else {
            self.kept.push(persisted);
            return Ok(());
        };
        // No name of this write stays, whichever step failed. A generation's
        // name this write did not link is another's.
        let _ = remove_at(root, &writing);
        if !linked {
            return Err(failed);
        }
        // The link may have reached the disk, and a start runs the newest
        // generation on file: the removal is synced as the link was. A name
        // that stays and a removal that did not sync are each told, in its
        // own words, beside the write's own failure.
        let left = match remove_at(root, &name) {
            Err(left) => stays(generation, &left),
            Ok(()) => match sync(root) {
                Ok(()) => return Err(failed),
                Err(left) => unsynced(generation, &left),
            },
        };
        Err(io::Error::other(format!("{failed}; {left}")))
    }

    /// Takes the newest generation off file again: the store's entry
    /// refused it after it was persisted. The file goes from the directory
    /// this store holds, wherever the path leads by then: no start there
    /// reads a generation the entry refused. The store lists it no more,
    /// whatever the removal says. A name that does not come off and a
    /// removal that does not sync are each told, in its own words
    /// ([`stays`], [`unsynced`]); a name that stays is tried again ahead
    /// of the next write ([`SpecStore::forget_again`]).
    pub(crate) fn forget(&mut self) -> io::Result<()> {
        let Some(newest) = self.kept.pop() else {
            return Ok(());
        };
        let Persisted { generation, .. } = newest;
        if let Err(left) = self.remove(newest) {
            self.refused.push(newest);
            return Err(io::Error::other(stays(generation, &left)));
        }
        self.root
            .sync_all()
            .map_err(|left| io::Error::other(unsynced(generation, &left)))
    }

    /// Tries again the removal of each generation the entry refused whose
    /// name stayed on file ([`SpecStore::forget`]), and syncs the directory
    /// where one went, by the write's `sync`. One that stays still is tried
    /// at the next write, and each failure here is logged: no call answers
    /// it.
    fn forget_again(&mut self, sync: &mut impl FnMut(&fs::File) -> io::Result<()>) {
        let root = &*self.root;
        let mut went = Vec::new();
        self.refused.retain(|&refused| {
            let generation = refused.generation;
            let Err(error) = remove_at(root, &file_name(refused)) else {
                went.push(generation);
                return false;
            };
            warn!(generation, %error, "a refused generation stays on file: a start may run it");
            true
        });
        if !went.is_empty() {
            // The removal's own sync: the write that follows may not land.
            if let Err(error) = sync(root) {
                warn!(generations = ?went, %error, "a refused generation went off file, but the directory did not sync: a crash may bring its name back");
            }
        }
    }

    /// Removes the generations under the [`RETAINED`] newest, and returns
    /// their numbers with what a removal failed by. A file that stays is
    /// tried again at the next apply. The files go from the directory this
    /// store holds.
    pub(crate) fn prune(&mut self) -> Vec<(u64, io::Result<()>)> {
        let over = self.kept.len().saturating_sub(RETAINED);
        let mut pruned = Vec::with_capacity(over);
        let mut stays = Vec::new();
        for old in self.kept.drain(..over).collect::<Vec<_>>() {
            let removed = self.remove(old);
            if removed.is_err() {
                stays.push(old);
            }
            pruned.push((old.generation, removed));
        }
        if !pruned.is_empty() {
            let _ = self.root.sync_all();
        }
        // A file that stays is still on file, under the retained ones.
        stays.append(&mut self.kept);
        self.kept = stays;
        pruned
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/document/operator/spec_store/tests.rs"]
pub(crate) mod tests;
