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
//! ([`Generations`]) persists a document here and only then calls the
//! store's entry, one apply at a time: one process writes the directory,
//! under the door's lock alone.
//!
//! One process holds a directory: a store takes the lock of a file it
//! keeps there (`.lock`) before it reads a name, and holds it until it is
//! dropped; the system lets it go at any exit. A start that finds the lock
//! held waits a bound for its holder to stop and takes the directory then.
//! One still held at the bound's end is refused by name, and one whose
//! lock the system does not give is refused in the system's own words: the
//! lock asks that its file is there or can be made, and no write of it on
//! a file system that locks as a local one does. So a start reads the
//! directory once the process before it writes no more, and runs that
//! one's last apply. The lock is on the file, not on its name: the file is
//! made once and never removed, and one removed under a running store lets
//! a second one in. It is the open file's, too: a child the process forks
//! holds a copy until it runs its program, so the lock may outlast a drop
//! by that moment, and a start that follows a drop in one process waits
//! like any other. The directory stands on a file system that links and
//! locks. Where no link is made, every write is refused at the link in the
//! system's own words, and nothing of it stays on file.
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
//! next apply of either lands. A start removes every temporary name, a
//! live writer's among them: that write is refused. A write refused after
//! its link removes its name and syncs the directory again, and says which
//! of the two failed. A name that stays may run at a start, or refuse it
//! beside the other form's name. A name that is gone where the directory
//! did not sync may come back after a crash.
//!
//! A start reads the directory: the highest number is the generation to
//! run, and a file a write left under its temporary name is removed. Any
//! other name is left alone, the lock file's among them. Two files under
//! one number, a directory another process holds, or a path that
//! leads to no directory, refuse the start by name. The [`RETAINED`]
//! highest numbers are kept; an apply removes the files under them. A
//! generation on file is read by the read call and by a rollback; the read
//! call reads its file by the directory's path alone ([`read_in`]), under
//! no lock of the door.
//!
//! [`Generations`]: super::generations::Generations

use std::{
    fs,
    io::{self, Read as _, Write as _},
    path::{Path, PathBuf},
    process, thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use tracing::info;

use super::{
    generations::{Format, SpecRejected, MAX_DOCUMENT_BYTES},
    spec_file::open_regular,
};

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

/// How many digits a generation's file name writes its number in: the
/// widest number's.
const DIGITS: usize = 20;

/// Why a spec store did not become a controller's source.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum StoreRefused {
    /// The path is a file, or leads nowhere.
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
        "another replica can take this controller's lease, and a standby does not follow an applied generation in this build: keep the lease in the process, or hand the spec as a file"
    )]
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

/// The path of `persisted` in the directory at `dir`.
fn path_in(dir: &Path, persisted: Persisted) -> PathBuf {
    let Persisted { generation, format } = persisted;
    let form = format.as_str();
    dir.join(format!("{generation:0DIGITS$}.{form}"))
}

/// The bytes on file for `persisted` in the directory at `dir`, one past
/// the size bound at most: a longer file is refused by its size, not read
/// whole. The file is opened as the spec file is, without waiting, and
/// what is no regular file is refused by its kind, unread: a pipe under a
/// generation's name holds no start and no read call.
pub(crate) fn read_in(dir: &Path, persisted: Persisted) -> io::Result<Vec<u8>> {
    let mut document = Vec::new();
    open_regular(&path_in(dir, persisted))?
        .take(MAX_DOCUMENT_BYTES + 1)
        .read_to_end(&mut document)?;
    Ok(document)
}

/// Syncs the directory itself, so that a rename in it outlasts a crash.
/// Where a directory does not open as a file there is no such call.
fn sync_dir(dir: &Path) -> io::Result<()> {
    if cfg!(unix) {
        fs::File::open(dir)?.sync_all()
    } else {
        Ok(())
    }
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

/// What a write says where its refused generation's name did not come off
/// file again: the write's own failure, then the removal's, then what a
/// start may meet. A start runs the newest generation on file, which this
/// one may be, and beside the other form's name the two refuse a start.
fn stays(generation: u64, failed: &io::Error, left: &io::Error) -> io::Error {
    let text = format!(
        "{failed}; generation {generation} was not taken off file again ({left}): a start may run it, \
         and beside the other form's name the two refuse a start"
    );
    io::Error::other(text)
}

/// What a write says where its refused generation's name came off file
/// again and the directory did not sync after it: the write's own failure,
/// then the sync's. The name is gone, and a crash may bring it back.
fn unsynced(generation: u64, failed: &io::Error, left: &io::Error) -> io::Error {
    let text = format!(
        "{failed}; generation {generation} was taken off file again, but the directory did not sync \
         ({left}): a crash may bring its name back"
    );
    io::Error::other(text)
}

/// The store: the directory and the generations on file in it.
#[derive(Debug)]
pub(crate) struct SpecStore {
    dir: PathBuf,
    /// The lock file, open: the directory's lock is held until the store
    /// is dropped.
    _held: fs::File,
    /// The generations on file, ascending.
    kept: Vec<Persisted>,
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
    /// A file left under a temporary name is removed: its write never
    /// ended. Where another process holds the directory's lock it is tried
    /// again every [`HELD_PACE`]: the directory is taken once the holder
    /// lets go, and a start still held when `wait` has passed is refused by
    /// name. A process that stops lets the lock go at its exit, so a start
    /// given the time a stop takes reads the directory after that process's
    /// last write.
    pub(crate) fn open_within(dir: PathBuf, wait: Duration) -> Result<Self, StoreRefused> {
        Self::found(&dir)?;
        let held = Self::hold(&dir, wait)?;
        Self::read_under(dir, held)
    }

    /// A second store on a directory another holds, its lock file open and
    /// not locked, as a file system whose locks do not hold between two
    /// hosts lets one in: the cases of two writers open theirs so.
    #[cfg(test)]
    pub(crate) fn open_beside(dir: PathBuf) -> Result<Self, StoreRefused> {
        Self::found(&dir)?;
        let unheld = Self::lock_file(&dir)?;
        Self::read_under(dir, unheld)
    }

    /// Refuses a path that leads to no directory.
    fn found(dir: &Path) -> Result<(), StoreRefused> {
        match fs::metadata(dir) {
            Ok(found) if found.is_dir() => Ok(()),
            Ok(_) => Err(StoreRefused::NotADirectory),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                Err(StoreRefused::NotADirectory)
            }
            Err(error) => Err(StoreRefused::Unreadable(error.to_string())),
        }
    }

    /// The lock file of `dir`, open: made empty where it is not there, and
    /// never truncated, since what stands in it is no part of the lock. On
    /// a file system that locks as a local one does, the lock asks no write
    /// of the file either: one that stands under bits this user may not
    /// write, as after a change of the controller's user, is opened to
    /// read. Where a lock is made of a byte-range lock, as a network file
    /// system's client may make it, it asks a file opened to write: the
    /// lock of a file this user may not write is not given there, and the
    /// refusal is the system's for the lock. Where the file is not there
    /// and may not be made, the refusal is the system's for the making.
    fn lock_file(dir: &Path) -> Result<fs::File, StoreRefused> {
        let path = dir.join(HELD);
        let made = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path);
        match made {
            Err(denied) if denied.kind() == io::ErrorKind::PermissionDenied => {
                fs::File::open(&path).map_err(|_| denied)
            }
            opened => opened,
        }
        .map_err(|error| StoreRefused::Unlocked(error.to_string()))
    }

    /// The directory's lock: an exclusive advisory lock on the lock file,
    /// tried every [`HELD_PACE`] until `wait` has passed. The system lets
    /// the lock go when the file is dropped, and at any exit.
    fn hold(dir: &Path, wait: Duration) -> Result<fs::File, StoreRefused> {
        let lock = Self::lock_file(dir)?;
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

    /// Reads the directory at `dir` under `held`, its lock file.
    fn read_under(dir: PathBuf, held: fs::File) -> Result<Self, StoreRefused> {
        let unreadable = |error: io::Error| StoreRefused::Unreadable(error.to_string());
        let mut kept = Vec::new();
        for entry in fs::read_dir(&dir).map_err(unreadable)? {
            let name = entry.map_err(unreadable)?.file_name();
            let Some(name) = name.to_str() else { continue };
            if name.starts_with(WRITING) {
                fs::remove_file(dir.join(name)).map_err(unreadable)?;
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
            _held: held,
            kept,
            opened,
            writes: 0,
        })
    }

    /// The directory's path.
    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    /// The generations on file, ascending.
    pub(crate) fn kept(&self) -> &[Persisted] {
        &self.kept
    }

    fn path(&self, persisted: Persisted) -> PathBuf {
        path_in(&self.dir, persisted)
    }

    /// The bytes on file for `persisted` ([`read_in`]).
    pub(crate) fn read(&self, persisted: Persisted) -> io::Result<Vec<u8>> {
        read_in(&self.dir, persisted)
    }

    /// When the file of `persisted` was written: when its generation was
    /// applied.
    pub(crate) fn written(&self, persisted: Persisted) -> io::Result<SystemTime> {
        fs::metadata(self.path(persisted))?.modified()
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
    /// start runs the newest generation on file.
    pub(crate) fn persist(&mut self, persisted: Persisted, document: &[u8]) -> io::Result<()> {
        self.persist_synced(persisted, document, sync_dir)
    }

    /// [`SpecStore::persist`] with the directory's sync as its argument: a
    /// case hands in one that fails.
    fn persist_synced(
        &mut self,
        persisted: Persisted,
        document: &[u8],
        mut sync: impl FnMut(&Path) -> io::Result<()>,
    ) -> io::Result<()> {
        let Persisted { generation, format } = persisted;
        let taken = || {
            let text = format!("generation {generation} is on file already");
            io::Error::new(io::ErrorKind::AlreadyExists, text)
        };
        let name = self.path(persisted);
        let other = self.path(Persisted {
            generation,
            format: other_form(format),
        });
        if name.try_exists()? || other.try_exists()? {
            // Another process wrote the directory, or a refused
            // generation stayed on file.
            return Err(taken());
        }
        self.writes += 1;
        let (id, opened, write) = (process::id(), self.opened, self.writes);
        let writing = self.dir.join(format!("{WRITING}{id}-{opened}-{write}"));
        let mut linked = false;
        let written = (|| {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&writing)?;
            file.write_all(document)?;
            file.sync_all()?;
            drop(file);
            fs::hard_link(&writing, &name).map_err(|error| match error.kind() {
                // The name appeared since the look above: its file stays.
                io::ErrorKind::AlreadyExists => taken(),
                _ => error,
            })?;
            linked = true;
            fs::remove_file(&writing)?;
            // The other form's name since the look above. A writer of that
            // form looks for this one after its own link, so one of the two
            // backs out at least: the two never both stay.
            if other.try_exists()? {
                return Err(backs_out(generation));
            }
            sync(&self.dir)
        })();
        let Err(failed) = written else {
            self.kept.push(persisted);
            return Ok(());
        };
        // No name of this write stays, whichever step failed. A generation's
        // name this write did not link is another's.
        let _ = fs::remove_file(&writing);
        if !linked {
            return Err(failed);
        }
        // The link may have reached the disk, and a start runs the newest
        // generation on file: the removal is synced as the link was. A name
        // that stays and a removal that did not sync are each told, in its
        // own words, beside the write's own failure.
        if let Err(left) = fs::remove_file(&name) {
            return Err(stays(generation, &failed, &left));
        }
        match sync(&self.dir) {
            Ok(()) => Err(failed),
            Err(left) => Err(unsynced(generation, &failed, &left)),
        }
    }

    /// Takes the newest generation off file again: the store's entry
    /// refused it after it was persisted.
    pub(crate) fn forget(&mut self) -> io::Result<()> {
        let Some(newest) = self.kept.pop() else {
            return Ok(());
        };
        fs::remove_file(self.path(newest))?;
        sync_dir(&self.dir)
    }

    /// Removes the generations under the [`RETAINED`] newest, and returns
    /// their numbers with what a removal failed by. A file that stays is
    /// tried again at the next apply.
    pub(crate) fn prune(&mut self) -> Vec<(u64, io::Result<()>)> {
        let over = self.kept.len().saturating_sub(RETAINED);
        let mut pruned = Vec::with_capacity(over);
        let mut stays = Vec::new();
        for old in self.kept.drain(..over).collect::<Vec<_>>() {
            let removed = fs::remove_file(self.path(old));
            if removed.is_err() {
                stays.push(old);
            }
            pruned.push((old.generation, removed));
        }
        if !pruned.is_empty() {
            let _ = sync_dir(&self.dir);
        }
        // A file that stays is still on file, under the retained ones.
        stays.append(&mut self.kept);
        self.kept = stays;
        pruned
    }
}

#[cfg(test)]
pub(crate) mod tests {
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    use std::{collections::BTreeSet, sync::mpsc};

    use super::*;
    use crate::controller::membership::tests::Logged;
    #[cfg(unix)]
    use crate::controller::spec_file::tests::pipe_at;

    const YAML: Format = Format::Yaml;

    fn at(generation: u64, format: Format) -> Persisted {
        Persisted { generation, format }
    }

    /// The names in `dir`, sorted, the store's own lock file apart.
    pub(crate) fn names(dir: &Path) -> BTreeSet<String> {
        fs::read_dir(dir)
            .expect("the directory reads")
            .map(|entry| entry.expect("an entry").file_name())
            .map(|name| name.into_string().expect("a name in UTF-8"))
            .filter(|name| name != HELD)
            .collect()
    }

    /// The name a generation's file takes.
    pub(crate) fn name_of(generation: u64, form: &str) -> String {
        format!("{generation:020}.{form}")
    }

    #[test]
    fn a_persisted_generation_is_one_whole_file_under_its_number() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut store = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
        assert_eq!(store.kept(), []);

        store.persist(at(42, YAML), b"first").expect("it persists");
        store
            .persist(at(43, Format::Json), b"{}")
            .expect("it persists");
        // One file a generation, by its number and its form, and no name a
        // write was in flight under.
        let on_file = BTreeSet::from([name_of(42, "yaml"), name_of(43, "json")]);
        assert_eq!(names(dir.path()), on_file);
        assert_eq!(store.kept(), [at(42, YAML), at(43, Format::Json)]);
        assert_eq!(store.read(at(42, YAML)).expect("it reads"), b"first");
        assert_eq!(store.read(at(43, Format::Json)).expect("it reads"), b"{}");
    }

    #[test]
    fn a_start_reads_the_generations_on_file_and_removes_a_write_that_never_ended() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let write = |name: &str| fs::write(dir.path().join(name), name).expect("the file");
        let unended = format!("{WRITING}7-3");
        let others = [
            "notes.txt",
            "3.yaml",
            &name_of(0, "yaml"),
            &name_of(5, "toml"),
        ];
        for name in [&name_of(10, "json"), &name_of(3, "yaml"), &unended] {
            write(name);
        }
        for name in others {
            write(name);
        }

        let store = SpecStore::open(dir.path().to_owned()).expect("it opens");
        // Ascending, whatever order the directory lists them in; a name
        // that is no generation's is no generation, and stays.
        assert_eq!(store.kept(), [at(3, YAML), at(10, Format::Json)]);
        let left = names(dir.path());
        assert!(!left.contains(&unended), "{left:?}");
        for name in others {
            assert!(left.contains(name), "{name} of {left:?}");
        }
    }

    #[test]
    fn two_names_under_one_number_and_a_path_to_no_directory_refuse_by_name() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let file = dir.path().join("a-file");
        fs::write(&file, "").expect("the file");
        for path in [file, dir.path().join("nowhere")] {
            let refused = SpecStore::open(path).expect_err("no directory");
            assert_eq!(refused, StoreRefused::NotADirectory);
            assert_eq!(refused.to_string(), "the path leads to no directory");
        }

        for form in ["yaml", "json"] {
            fs::write(dir.path().join(name_of(7, form)), "").expect("the file");
        }
        let refused = SpecStore::open(dir.path().to_owned()).expect_err("two names");
        assert_eq!(refused, StoreRefused::TwoNames(7));
        assert_eq!(refused.to_string(), "generation 7 stands under two names");
    }

    /// One store holds a directory: a second open is refused by name while
    /// the first stands, at once or at the end of the wait it was given,
    /// and lands once the first is dropped. The lock file is the store's
    /// own: made empty, and a start leaves its bytes as it finds them.
    #[test]
    fn a_held_directory_refuses_a_second_store_until_its_holder_is_dropped() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let open = |wait| SpecStore::open_within(dir.path().to_owned(), wait);
        let first = open(Duration::ZERO).expect("it opens");
        let lock = dir.path().join(HELD);
        assert_eq!(fs::read(&lock).expect("the lock file"), b"");

        let refused = open(Duration::ZERO).expect_err("it is held");
        assert_eq!(refused, StoreRefused::Held(Duration::ZERO));
        let wait = 3 * HELD_PACE;
        let asked = Instant::now();
        let refused = open(wait).expect_err("it is held");
        assert!(asked.elapsed() >= wait, "{:?}", asked.elapsed());
        assert_eq!(refused, StoreRefused::Held(wait));
        assert_eq!(
            refused.to_string(),
            "another controller holds the directory, waited on for 150ms: one controller holds a spec store"
        );

        // What stands in the lock file is no part of the lock. A start
        // after the drop waits: a child process of another case may hold a
        // copy of the open file for a moment.
        fs::write(&lock, "a note").expect("the file");
        drop(first);
        let second = open(Duration::from_secs(30)).expect("the directory is free");
        assert_eq!(second.kept(), []);
        assert_eq!(fs::read(&lock).expect("the lock file"), b"a note");
    }

    /// A start that waits reads the directory only once it holds it: what
    /// the holder wrote while the start waited is on file for it. The wait
    /// is said in one line.
    #[test]
    fn a_waiting_start_reads_the_directory_once_its_holder_let_go() {
        const ROOM: Duration = Duration::from_secs(30);
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut first = SpecStore::open(dir.path().to_owned()).expect("it opens");
        // The waiter's own subscriber keeps its thread's lines, a second
        // one registered beside it, as in the door's audit case.
        let lines = Logged::default();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let (path, writer) = (dir.path().to_owned(), lines.clone());
        let waiter = thread::spawn(move || {
            let subscriber = tracing_subscriber::fmt()
                .with_ansi(false)
                .with_writer(move || writer.clone())
                .finish();
            tracing::subscriber::with_default(subscriber, || SpecStore::open_within(path, ROOM))
        });
        let said = || {
            let text = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
            text.matches("held by another controller: waiting for it to stop")
                .count()
        };
        // The waiter found the directory held, and has read no name yet.
        let asked = Instant::now();
        while said() == 0 {
            assert!(asked.elapsed() < ROOM, "the waiter never said it waits");
            thread::sleep(Duration::from_millis(5));
        }
        // Three more tries pass under the lock: a line at every try would
        // be said again.
        thread::sleep(3 * HELD_PACE);
        first.persist(at(1, YAML), b"first").expect("it persists");
        drop(first);
        let second = waiter
            .join()
            .expect("the waiter's thread")
            .expect("the directory is free");
        assert_eq!(second.kept(), [at(1, YAML)]);
        assert_eq!(said(), 1, "one line for the wait");
    }

    /// On a file system that locks as a local one does, the lock asks no
    /// write of its file: one that stands under bits its user may not
    /// write, as after a change of the controller's user, is opened to read
    /// and locks all the same.
    #[cfg(unix)]
    #[test]
    #[expect(
        clippy::print_stdout,
        reason = "a run the bits do not hold in reports it on stdout under --nocapture"
    )]
    fn a_lock_file_its_user_may_not_write_locks_a_local_directory() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = tempfile::tempdir().expect("a temp dir");
        let lock = dir.path().join(HELD);
        fs::write(&lock, "").expect("the file");
        fs::set_permissions(&lock, fs::Permissions::from_mode(0o444)).expect("its bits");
        if fs::OpenOptions::new().write(true).open(&lock).is_ok() {
            // The bits do not hold for this user (the superuser): the first
            // open serves, and there is nothing of the second to read here.
            println!(
                "a lock file its user may not write: this run asserted another lawful world: \
                 the file's bits do not hold for this user"
            );
            return;
        }
        let open = || SpecStore::open_within(dir.path().to_owned(), Duration::ZERO);
        let first = open().expect("the lock is taken on a file opened to read");
        assert_eq!(
            open().expect_err("it is held"),
            StoreRefused::Held(Duration::ZERO)
        );
        drop(first);
    }

    /// A lock the system does not give refuses the start in the system's
    /// own words: here a directory stands under the lock file's name.
    #[test]
    fn a_lock_the_system_does_not_give_refuses_the_start_in_its_words() {
        let dir = tempfile::tempdir().expect("a temp dir");
        fs::create_dir(dir.path().join(HELD)).expect("a directory");
        let refused = SpecStore::open(dir.path().to_owned()).expect_err("no lock");
        assert!(
            matches!(&refused, StoreRefused::Unlocked(told) if told.contains("directory")),
            "{refused:?}"
        );
        assert!(refused.to_string().starts_with("not locked: "), "{refused}");
    }

    /// The bound of what a store keeps: at no apply does it hold more than
    /// the retained generations, on file or in its own list.
    #[test]
    fn a_store_keeps_the_newest_generations_and_removes_the_ones_under_them() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut store = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
        let last = u64::try_from(RETAINED).expect("a small count") + 3;
        for generation in 1..=last {
            store
                .persist(at(generation, YAML), b"document")
                .expect("it persists");
            let pruned = store.prune();
            assert!(pruned.iter().all(|(_, removed)| removed.is_ok()));
            assert!(store.kept().len() <= RETAINED, "{:?}", store.kept());
            assert!(names(dir.path()).len() <= RETAINED);
            // One apply removes one generation at most, the oldest.
            let oldest = generation.checked_sub(last - 3);
            let removed: Vec<u64> = pruned.iter().map(|(number, _)| *number).collect();
            assert_eq!(
                removed,
                oldest
                    .filter(|&number| number > 0)
                    .into_iter()
                    .collect::<Vec<_>>()
            );
        }
        let kept: Vec<u64> = store.kept().iter().map(|kept| kept.generation).collect();
        assert_eq!(kept, (4..=last).collect::<Vec<_>>());
        let on_file = (4..=last).map(|number| name_of(number, "yaml")).collect();
        assert_eq!(names(dir.path()), on_file);
    }

    /// A number on file already, in either form, takes no second write, and
    /// its file stays as it was.
    #[test]
    fn a_number_on_file_is_not_written_over() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut store = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
        store.persist(at(1, YAML), b"one").expect("it persists");
        for format in [YAML, Format::Json] {
            let refused = store
                .persist(at(1, format), b"two")
                .expect_err("the number is taken");
            assert_eq!(refused.kind(), io::ErrorKind::AlreadyExists);
            assert_eq!(refused.to_string(), "generation 1 is on file already");
        }
        assert_eq!(names(dir.path()), BTreeSet::from([name_of(1, "yaml")]));
        assert_eq!(store.read(at(1, YAML)).expect("it reads"), b"one");
        assert_eq!(store.kept(), [at(1, YAML)]);
    }

    /// Two stores on one directory, as two processes hold one, write one
    /// number at one moment: one write lands and the other is refused by
    /// name, whichever looked first. The file is the winner's, whole, and
    /// the refused write's own cleanup leaves it there.
    #[test]
    fn two_stores_that_write_one_number_at_one_moment_land_one_file() {
        const ROUNDS: u64 = 100;
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut first = SpecStore::open(dir.path().to_owned()).expect("it opens");
        // A second writer the lock did not keep out.
        let mut second = SpecStore::open_beside(dir.path().to_owned()).expect("it opens");
        // Two processes' temporary names differ, whatever the clock read.
        second.writes = 1 << 32;
        let barrier = std::sync::Barrier::new(2);
        for generation in 1..=ROUNDS {
            let write = |store: &mut SpecStore, bytes: &'static [u8]| {
                barrier.wait();
                store.persist(at(generation, YAML), bytes)
            };
            let (one, two) = thread::scope(|scope| {
                let one = scope.spawn(|| write(&mut first, b"first"));
                let two = scope.spawn(|| write(&mut second, b"second"));
                (one.join().expect("no panic"), two.join().expect("no panic"))
            });
            let (landed, refused) = match (one, two) {
                (Ok(()), Err(refused)) => (b"first".as_slice(), refused),
                (Err(refused), Ok(())) => (b"second".as_slice(), refused),
                both => panic!("round {generation}: one write lands: {both:?}"),
            };
            assert_eq!(refused.kind(), io::ErrorKind::AlreadyExists);
            let taken = format!("generation {generation} is on file already");
            assert_eq!(refused.to_string(), taken);
            let on_file = fs::read(dir.path().join(name_of(generation, "yaml")));
            assert_eq!(on_file.expect("the winner's file"), landed);
        }
        // A generation's name a round, and no temporary one.
        let all = (1..=ROUNDS).map(|number| name_of(number, "yaml")).collect();
        assert_eq!(names(dir.path()), all);
    }

    /// A write whose directory does not sync after the link is refused, and
    /// its generation's name goes with its temporary one, the directory
    /// synced again after it: nothing of it is on file, and the number
    /// takes a write again.
    #[test]
    fn a_write_the_directory_does_not_sync_leaves_no_name_behind() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut store = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
        let mut syncs = 0;
        let unsynced = |_: &Path| {
            syncs += 1;
            if syncs == 1 {
                return Err(io::Error::other("the directory did not sync"));
            }
            Ok(())
        };
        let refused = store
            .persist_synced(at(1, YAML), b"one", unsynced)
            .expect_err("no sync");
        assert_eq!(refused.to_string(), "the directory did not sync");
        assert_eq!(syncs, 2, "the removal is synced as the link was");
        assert_eq!(names(dir.path()), BTreeSet::new());
        assert_eq!(store.kept(), []);
        store
            .persist(at(1, YAML), b"one")
            .expect("the number is free");
    }

    /// A refused generation whose name does not come off file again is told
    /// beside the write's own failure: a start runs the newest generation
    /// on file. The name here is one no removal of a file takes, a
    /// directory put in its place while the directory's sync fails.
    #[test]
    fn a_refused_generation_that_stays_on_file_is_told() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut store = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
        let name = dir.path().join(name_of(1, "yaml"));
        let stuck = |_: &Path| {
            fs::remove_file(&name).expect("the linked name");
            fs::create_dir(&name).expect("a directory in its place");
            Err(io::Error::other("the directory did not sync"))
        };
        let refused = store
            .persist_synced(at(1, YAML), b"one", stuck)
            .expect_err("no sync");
        let told = refused.to_string();
        let (failed, left) = told
            .split_once("; generation 1 was not taken off file again (")
            .expect("the write's failure, then the removal's");
        assert_eq!(failed, "the directory did not sync");
        assert!(
            left.ends_with(
                "): a start may run it, and beside the other form's name the two refuse a start"
            ),
            "{told}"
        );
        assert_eq!(store.kept(), []);
    }

    /// A writer that finds the other form's name after its own link backs
    /// out, and is told so apart from a number that is on file: its name is
    /// gone again, and once the other name is too, its next write of the
    /// number lands. The other writer's link is a link here that resolves
    /// only once this write's name stands: absent at the look before the
    /// link, there at the look after it.
    #[cfg(unix)]
    #[test]
    fn a_write_that_backs_out_is_told_so_and_its_next_write_lands() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut store = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
        let other = dir.path().join(name_of(1, "json"));
        symlink(name_of(1, "yaml"), &other).expect("a link");
        let refused = store
            .persist(at(1, YAML), b"one")
            .expect_err("the other form's name stands");
        assert_eq!(refused.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(
            refused.to_string(),
            "generation 1 was written in the other form as this write was: this write backs out"
        );
        assert_eq!(names(dir.path()), BTreeSet::from([name_of(1, "json")]));
        assert_eq!(store.kept(), []);
        fs::remove_file(&other).expect("the other writer backed out too");
        store
            .persist(at(1, YAML), b"one")
            .expect("the number is free");
        assert_eq!(store.kept(), [at(1, YAML)]);
    }

    /// A back-out whose name comes off file where the directory does not
    /// sync after it says that, and not that the name stays: the name is
    /// gone as the words say, and a crash may bring it back.
    #[cfg(unix)]
    #[test]
    fn a_back_out_whose_removal_does_not_sync_is_told_so() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut store = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
        symlink(name_of(1, "yaml"), dir.path().join(name_of(1, "json"))).expect("a link");
        let never = |_: &Path| Err(io::Error::other("the directory did not sync"));
        let refused = store
            .persist_synced(at(1, YAML), b"one", never)
            .expect_err("the other form's name stands");
        assert_eq!(
            refused.to_string(),
            "generation 1 was written in the other form as this write was: this write backs out; \
             generation 1 was taken off file again, but the directory did not sync \
             (the directory did not sync): a crash may bring its name back"
        );
        assert_eq!(names(dir.path()), BTreeSet::from([name_of(1, "json")]));
        assert_eq!(store.kept(), []);
    }

    /// Two stores on one directory, as two processes hold one, write one
    /// number at one moment in the two forms: one lands or neither, never
    /// both, and a start after them is not refused by two names.
    #[test]
    fn two_stores_that_write_one_number_in_the_two_forms_do_not_both_land() {
        const ROUNDS: u64 = 100;
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut first = SpecStore::open(dir.path().to_owned()).expect("it opens");
        // A second writer the lock did not keep out.
        let mut second = SpecStore::open_beside(dir.path().to_owned()).expect("it opens");
        // Two processes' temporary names differ, whatever the clock read.
        second.writes = 1 << 32;
        let barrier = std::sync::Barrier::new(2);
        for generation in 1..=ROUNDS {
            let write = |store: &mut SpecStore, format: Format| {
                barrier.wait();
                store.persist(at(generation, format), b"a document")
            };
            let (yaml, json) = thread::scope(|scope| {
                let yaml = scope.spawn(|| write(&mut first, YAML));
                let json = scope.spawn(|| write(&mut second, Format::Json));
                (
                    yaml.join().expect("no panic"),
                    json.join().expect("no panic"),
                )
            });
            match (yaml, json) {
                (Ok(()), Ok(())) => panic!("round {generation}: both forms landed"),
                // Neither landed: each backed out and is told so, the number
                // is free, and the next write of it lands.
                (Err(yaml), Err(json)) => {
                    for told in [yaml, json] {
                        assert!(
                            told.to_string().ends_with("this write backs out"),
                            "round {generation}: {told}"
                        );
                    }
                    first
                        .persist(at(generation, YAML), b"a document")
                        .expect("the number is free");
                }
                _ => {}
            }
        }
        // No number holds two names, and no temporary name is left.
        let left = names(dir.path());
        let beside = SpecStore::open_beside(dir.path().to_owned()).expect("it opens");
        assert_eq!(left.len(), beside.kept().len());
    }

    /// A pipe under a generation's name, with no writer, is refused by its
    /// kind, unread, as the spec file's source refuses one. The read runs
    /// on a thread of its own, waited five seconds, so an open that waits
    /// fails the case.
    #[cfg(unix)]
    #[test]
    fn a_pipe_under_a_generations_name_is_refused_unread() {
        let dir = tempfile::tempdir().expect("a temp dir");
        pipe_at(&dir.path().join(name_of(1, "yaml")));
        let store = SpecStore::open(dir.path().to_owned()).expect("a generation's name");
        assert_eq!(store.kept(), [at(1, YAML)]);

        let (answer, answered) = mpsc::channel();
        thread::spawn(move || {
            let _ = answer.send(store.read(at(1, YAML)));
        });
        let read = answered
            .recv_timeout(Duration::from_secs(5))
            .expect("the read answers: its open does not wait");
        let refused = read.expect_err("no regular file");
        assert_eq!(refused.to_string(), "the path leads to no regular file");
    }

    #[test]
    fn a_refused_generation_goes_off_file_again() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut store = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
        for generation in [1, 2] {
            store
                .persist(at(generation, YAML), b"document")
                .expect("it persists");
        }
        store.forget().expect("the newest is removed");
        assert_eq!(store.kept(), [at(1, YAML)]);
        assert_eq!(names(dir.path()), BTreeSet::from([name_of(1, "yaml")]));
    }

    #[test]
    fn a_write_the_directory_does_not_take_leaves_no_generation_behind() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let inner = dir.path().join("store");
        fs::create_dir(&inner).expect("the directory");
        let mut store = SpecStore::open(inner.clone()).expect("an empty directory");
        store.persist(at(1, YAML), b"one").expect("it persists");
        fs::remove_dir_all(&inner).expect("the directory goes");

        store
            .persist(at(2, YAML), b"two")
            .expect_err("no directory takes the write");
        assert_eq!(store.kept(), [at(1, YAML)]);
    }

    #[test]
    fn a_file_past_the_size_bound_is_read_one_byte_past_it_and_no_further() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
        fs::write(dir.path().join(name_of(1, "yaml")), vec![b'#'; bound + 64]).expect("the file");
        let store = SpecStore::open(dir.path().to_owned()).expect("it opens");
        assert_eq!(store.read(at(1, YAML)).expect("it reads").len(), bound + 1);
    }
}
