//! The policy spec's store (design 4.6): the directory an applied document
//! is persisted in before it runs, where the spec's source is the apply
//! call.
//!
//! A generation is one file, named by its number and its form
//! (`00000000000000000042.yaml`). The bytes are the document as its caller
//! sent it: the number lives in the name, never inside those bytes. A file
//! is written whole under another name, synced, renamed into place and the
//! directory synced, in that order, so a name of that shape holds a whole
//! document or does not exist. A number on file is not written over: an
//! apply whose number is on file already, in either form, is refused, so a
//! second process on the directory replaces no generation of the first.
//! The door ([`Generations`]) persists a
//! document here and only then calls the store's entry, one apply at a
//! time: the directory is written under the door's lock alone.
//!
//! A start reads the directory: the highest number is the generation to
//! run, and a file a write left under its temporary name is removed. Any
//! other name is left alone. Two files under one number, or a path that
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
    process,
    time::SystemTime,
};

use super::generations::{Format, SpecRejected, MAX_DOCUMENT_BYTES};

/// How many generations a store keeps: the standing one and the seven
/// before it.
pub(crate) const RETAINED: usize = 8;

/// The first letters of a name a write is in flight under: no start reads
/// it.
const WRITING: &str = ".writing-";

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
/// whole.
pub(crate) fn read_in(dir: &Path, persisted: Persisted) -> io::Result<Vec<u8>> {
    let mut document = Vec::new();
    fs::File::open(path_in(dir, persisted))?
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

/// The store: the directory and the generations on file in it.
#[derive(Debug)]
pub(crate) struct SpecStore {
    dir: PathBuf,
    /// The generations on file, ascending.
    kept: Vec<Persisted>,
    /// The writes this store began: a temporary name's own number.
    writes: u64,
}

impl SpecStore {
    /// Reads the directory at `dir`: the generations on file. A file left
    /// under a temporary name is removed: its write never ended.
    pub(crate) fn open(dir: PathBuf) -> Result<Self, StoreRefused> {
        let unreadable = |error: io::Error| StoreRefused::Unreadable(error.to_string());
        match fs::metadata(&dir) {
            Ok(found) if found.is_dir() => {}
            Ok(_) => return Err(StoreRefused::NotADirectory),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(StoreRefused::NotADirectory)
            }
            Err(error) => return Err(unreadable(error)),
        }
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
        Ok(Self {
            dir,
            kept,
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
    /// name, synced, renamed into place, then the directory synced. A write
    /// that fails leaves no file under the generation's name, and a number
    /// on file already takes no write.
    pub(crate) fn persist(&mut self, persisted: Persisted, document: &[u8]) -> io::Result<()> {
        let Persisted { generation, .. } = persisted;
        for format in [Format::Yaml, Format::Json] {
            if self.path(Persisted { generation, format }).try_exists()? {
                // Another process wrote the directory, or a refused
                // generation stayed on file.
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("generation {generation} is on file already"),
                ));
            }
        }
        self.writes += 1;
        let writing = self
            .dir
            .join(format!("{WRITING}{}-{}", process::id(), self.writes));
        let written = (|| {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&writing)?;
            file.write_all(document)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&writing, self.path(persisted))?;
            sync_dir(&self.dir)
        })();
        if written.is_err() {
            // Whichever step failed, neither name stays: the rename may
            // have happened where the directory's sync failed.
            let _ = fs::remove_file(&writing);
            let _ = fs::remove_file(self.path(persisted));
            return written;
        }
        self.kept.push(persisted);
        Ok(())
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
    use std::collections::BTreeSet;

    use super::*;

    const YAML: Format = Format::Yaml;

    fn at(generation: u64, format: Format) -> Persisted {
        Persisted { generation, format }
    }

    /// The names in `dir`, sorted.
    pub(crate) fn names(dir: &Path) -> BTreeSet<String> {
        fs::read_dir(dir)
            .expect("the directory reads")
            .map(|entry| entry.expect("an entry").file_name())
            .map(|name| name.into_string().expect("a name in UTF-8"))
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
