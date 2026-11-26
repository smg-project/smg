#[cfg(unix)]
use std::os::unix::fs::symlink;
use std::{collections::BTreeSet, sync::mpsc};

use super::*;
#[cfg(unix)]
use crate::document::operator::test_support::pipe_at;
use crate::document::operator::test_support::Logged;

impl SpecStore {
    /// [`SpecStore::open_within`] as the cases open theirs: a start that
    /// waits a case's room, as the controller's waits its grace period. A
    /// directory whose store a case dropped is free a moment later where
    /// another case's child process holds a copy of the lock file.
    pub(crate) fn open(dir: PathBuf) -> Result<Self, StoreRefused> {
        Self::open_within(dir, Duration::from_secs(30))
    }

    /// A second store on a directory another holds, its lock file open and
    /// not locked, as a file system whose locks do not hold between two
    /// hosts lets one in: the cases of two writers open theirs so.
    pub(in crate::document::operator) fn open_beside(dir: PathBuf) -> Result<Self, StoreRefused> {
        let (dir, root) = Self::rooted(&dir)?;
        let unheld = Self::lock_file(&root)?;
        Self::read_under(dir, root, unheld)
    }
}

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
    let through = file.join("store");
    for path in [file, through, dir.path().join("nowhere")] {
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

/// A start in a thread of its own, with the lines it says.
struct Waiter {
    start: thread::JoinHandle<Result<SpecStore, StoreRefused>>,
    lines: Logged,
    /// A second subscriber registered beside the thread's own for as
    /// long as a case reads the lines, as in the door's audit case.
    _beside: tracing::Dispatch,
}

impl Waiter {
    /// The room a case gives a waiting start, and its line.
    const ROOM: Duration = Duration::from_secs(30);

    /// A start on `path` whose thread's own subscriber keeps its
    /// lines. Returns once the start said that it waits: it holds its
    /// directory, found it locked, and has read no name yet.
    fn held_off(path: PathBuf) -> Self {
        let lines = Logged::default();
        let beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let writer = lines.clone();
        let start = thread::spawn(move || {
            let subscriber = tracing_subscriber::fmt()
                .with_ansi(false)
                .with_writer(move || writer.clone())
                .finish();
            tracing::subscriber::with_default(subscriber, || {
                SpecStore::open_within(path, Self::ROOM)
            })
        });
        let asked = Instant::now();
        while Self::said(&lines) == 0 {
            assert!(
                asked.elapsed() < Self::ROOM,
                "the waiter never said it waits"
            );
            thread::sleep(Duration::from_millis(5));
        }
        Self {
            start,
            lines,
            _beside: beside,
        }
    }

    /// How often a start said in `lines` that it waits.
    fn said(lines: &Logged) -> usize {
        let text = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        text.matches("held by another controller: waiting for it to stop")
            .count()
    }
}

/// A start that waits reads the directory only once it holds it: what
/// the holder wrote while the start waited is on file for it. The wait
/// is said in one line.
#[test]
fn a_waiting_start_reads_the_directory_once_its_holder_let_go() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut first = SpecStore::open(dir.path().to_owned()).expect("it opens");
    // The waiter found the directory held, and has read no name yet.
    let waiter = Waiter::held_off(dir.path().to_owned());
    // Three more tries pass under the lock: a line at every try would
    // be said again.
    thread::sleep(3 * HELD_PACE);
    first.persist(at(1, YAML), b"first").expect("it persists");
    drop(first);
    let second = waiter
        .start
        .join()
        .expect("the waiter's thread")
        .expect("the directory is free");
    assert_eq!(second.kept(), [at(1, YAML)]);
    assert_eq!(Waiter::said(&waiter.lines), 1, "one line for the wait");
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

/// Where the lock file is not there and may not be made, the refusal is
/// the system's for the making: the open to read that follows finds no
/// file, and its words are not the ones told.
#[cfg(unix)]
#[test]
#[expect(
    clippy::print_stdout,
    reason = "a run the bits do not hold in reports it on stdout under --nocapture"
)]
fn a_lock_file_that_may_not_be_made_refuses_the_start_in_the_makings_words() {
    use std::os::unix::fs::PermissionsExt as _;

    let dir = tempfile::tempdir().expect("a temp dir");
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).expect("its bits");
    if fs::File::create(dir.path().join("made")).is_ok() {
        // The bits do not hold for this user (the superuser): the lock
        // file is made, and there is no refusal to read here.
        println!(
            "a lock file that may not be made: this run asserted another lawful world: \
                 the directory's bits do not hold for this user"
        );
        return;
    }
    let refused = SpecStore::open_within(dir.path().to_owned(), Duration::ZERO);
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).expect("its bits");
    let refused = refused.expect_err("no lock");
    assert!(
        matches!(&refused, StoreRefused::Unmade(told) if told.contains("denied")),
        "{refused:?}"
    );
    let said = refused.to_string();
    assert!(
        said.ends_with("a store's first start needs a directory it may write"),
        "{said}"
    );
}

/// A pipe under the lock file's name, under bits its user may not
/// write, holds no start: the open to read does not wait on it, what
/// is no regular file is refused unread, and the refusal is the
/// system's for the making. The open runs on a thread of its own,
/// waited five seconds, so an open that waits fails the case.
#[cfg(unix)]
#[test]
#[expect(
    clippy::print_stdout,
    reason = "a run the bits do not hold in reports it on stdout under --nocapture"
)]
fn a_pipe_its_user_may_not_write_under_the_locks_name_holds_no_start() {
    use std::os::unix::fs::PermissionsExt as _;

    let dir = tempfile::tempdir().expect("a temp dir");
    let lock = dir.path().join(HELD);
    pipe_at(&lock);
    fs::set_permissions(&lock, fs::Permissions::from_mode(0o444)).expect("its bits");
    // An open to read and write of a pipe does not wait, as one to
    // write alone would where the bits do not hold.
    if fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&lock)
        .is_ok()
    {
        // The bits do not hold for this user (the superuser): the first
        // open serves, and there is nothing of the second to read here.
        println!(
            "a pipe its user may not write: this run asserted another lawful world: \
                 the pipe's bits do not hold for this user"
        );
        return;
    }
    let path = dir.path().to_owned();
    let (answer, answered) = mpsc::channel();
    thread::spawn(move || {
        let _ = answer.send(SpecStore::open_within(path, Duration::ZERO));
    });
    let opened = answered
        .recv_timeout(Duration::from_secs(5))
        .expect("the open answers: it does not wait on the pipe");
    let refused = opened.expect_err("no lock on a pipe");
    assert!(
        matches!(&refused, StoreRefused::Unlocked(told) if told.contains("denied")),
        "{refused:?}"
    );
}

/// A write of the lock file the user's bits refuse, and one a file
/// system mounted to read alone refuses, leave it to an open to read;
/// another refusal stands as it is. The case reads the system's
/// refusals by their numbers: it mounts no file system.
#[cfg(unix)]
#[test]
fn a_write_refused_by_bits_or_a_read_only_mount_leaves_an_open_to_read() {
    use rustix::io::Errno;

    let refused = |errno: Errno| io::Error::from(errno);
    assert!(SpecStore::read_serves(&refused(Errno::ACCESS)));
    assert!(SpecStore::read_serves(&refused(Errno::ROFS)));
    assert!(!SpecStore::read_serves(&refused(Errno::ISDIR)));
    assert!(!SpecStore::read_serves(&refused(Errno::NOENT)));
}

/// A link in the store's path that turns while a start waits leads
/// that start nowhere else: the path is resolved once, and the store
/// reads and writes the directory it locked.
#[cfg(unix)]
#[test]
fn a_link_turned_under_a_waiting_start_leaves_it_the_directory_it_locked() {
    let base = tempfile::tempdir().expect("a temp dir");
    let (first, second) = (base.path().join("first"), base.path().join("second"));
    let link = base.path().join("store");
    fs::create_dir(&first).expect("the first directory");
    fs::create_dir(&second).expect("the second directory");
    symlink(&first, &link).expect("the link");
    let mut holder = SpecStore::open(first.clone()).expect("the holder");
    // The waiter holds its directory and found it locked: the link
    // turns under it, and the holder writes once more and lets go.
    let waiter = Waiter::held_off(link.clone());
    fs::remove_file(&link).expect("the link goes");
    symlink(&second, &link).expect("the link turns");
    holder
        .persist(at(1, YAML), b"one")
        .expect("the holder's write");
    drop(holder);
    let mut store = waiter
        .start
        .join()
        .expect("the waiter")
        .expect("the directory is taken");
    assert_eq!(store.dir(), fs::canonicalize(&first).expect("its path"));
    assert_eq!(store.kept(), [at(1, YAML)], "the directory it locked");
    store.persist(at(2, YAML), b"two").expect("a write");
    let both = BTreeSet::from([name_of(1, "yaml"), name_of(2, "yaml")]);
    assert_eq!(names(&first), both);
    assert!(
        names(&second).is_empty(),
        "nothing lands where the link leads"
    );
}

/// A store whose path leads to another directory by the time of a
/// write refuses the write, and nothing lands there: the directory it
/// holds is the one it locked.
#[cfg(unix)]
#[test]
fn a_write_under_a_path_that_leads_elsewhere_is_refused() {
    let base = tempfile::tempdir().expect("a temp dir");
    let (dir, moved) = (base.path().join("store"), base.path().join("moved"));
    fs::create_dir(&dir).expect("the directory");
    let mut store = SpecStore::open(dir.clone()).expect("the store");
    store.persist(at(1, YAML), b"one").expect("a write");
    fs::rename(&dir, &moved).expect("the directory moves");
    fs::create_dir(&dir).expect("another stands at its path");
    let refused = store
        .persist(at(2, YAML), b"two")
        .expect_err("the path leads elsewhere");
    assert_eq!(refused.to_string(), MOVED);
    assert!(names(&dir).is_empty(), "nothing lands in the other");
    assert_eq!(names(&moved), BTreeSet::from([name_of(1, "yaml")]));
}

/// A store whose path leads to no directory by the time of a write
/// refuses the write as a start is refused there, in the start's own
/// words: a file laid at the path, or nothing.
#[cfg(unix)]
#[test]
fn a_write_under_a_path_that_leads_to_no_directory_is_refused_as_a_start_is() {
    let base = tempfile::tempdir().expect("a temp dir");
    let (dir, moved) = (base.path().join("store"), base.path().join("moved"));
    fs::create_dir(&dir).expect("the directory");
    let mut store = SpecStore::open(dir.clone()).expect("the store");
    store.persist(at(1, YAML), b"one").expect("a write");
    fs::rename(&dir, &moved).expect("the directory moves");
    let no_directory = StoreRefused::NotADirectory.to_string();
    fs::write(&dir, "a file").expect("a file stands at its path");
    let refused = store
        .persist(at(2, YAML), b"two")
        .expect_err("the path leads to a file");
    assert_eq!(refused.to_string(), no_directory);
    let start = SpecStore::open(dir.clone()).expect_err("a start there");
    assert_eq!(start, StoreRefused::NotADirectory);
    fs::remove_file(&dir).expect("the file goes");
    let refused = store
        .persist(at(2, YAML), b"two")
        .expect_err("the path leads nowhere");
    assert_eq!(refused.to_string(), no_directory);
    assert_eq!(names(&moved), BTreeSet::from([name_of(1, "yaml")]));
}

/// A store whose path leads through a file by the time of a write
/// refuses the write as a start is refused there, as a path to no
/// directory: the directory above its own moved away, and a file
/// stands where that one stood.
#[cfg(unix)]
#[test]
fn a_write_under_a_path_through_a_file_is_refused_as_a_start_is() {
    let base = tempfile::tempdir().expect("a temp dir");
    let (above, moved) = (base.path().join("above"), base.path().join("moved"));
    let dir = above.join("store");
    fs::create_dir_all(&dir).expect("the directory");
    let mut store = SpecStore::open(dir.clone()).expect("the store");
    store.persist(at(1, YAML), b"one").expect("a write");
    fs::rename(&above, &moved).expect("the directory above moves");
    fs::write(&above, "a file").expect("a file stands in the path's way");
    let refused = store
        .persist(at(2, YAML), b"two")
        .expect_err("the path leads through a file");
    assert_eq!(refused.to_string(), StoreRefused::NotADirectory.to_string());
    let start = SpecStore::open(dir).expect_err("a start there");
    assert_eq!(start, StoreRefused::NotADirectory);
    let held = names(&moved.join("store"));
    assert_eq!(held, BTreeSet::from([name_of(1, "yaml")]));
}

/// A directory replaced at the path of an open store leaves that store
/// the directory it holds, whole: the generations it listed there are
/// the ones it reads, the bytes and the time of the file held, and a
/// write is refused, since the path leads to the other directory by
/// then.
#[cfg(unix)]
#[test]
fn a_directory_replaced_under_an_open_store_is_read_where_it_is_held() {
    let base = tempfile::tempdir().expect("a temp dir");
    let (dir, moved) = (base.path().join("store"), base.path().join("moved"));
    fs::create_dir(&dir).expect("the directory");
    let mut store = SpecStore::open(dir.clone()).expect("the store");
    store.persist(at(1, YAML), b"held").expect("a write");
    fs::rename(&dir, &moved).expect("the directory moves");
    fs::create_dir(&dir).expect("another stands at its path");
    let other = dir.join(name_of(1, "yaml"));
    fs::write(&other, "other").expect("the other's first");
    fs::write(dir.join(name_of(2, "yaml")), "newer").expect("the other's second");
    // The other's file under the same name is of another time.
    let apart = UNIX_EPOCH + Duration::from_secs(86_400);
    fs::File::options()
        .write(true)
        .open(&other)
        .and_then(|file| file.set_modified(apart))
        .expect("the other's time");
    assert_eq!(store.kept(), [at(1, YAML)], "the directory it holds");
    assert_eq!(store.read(at(1, YAML)).expect("it reads"), b"held");
    let held = fs::metadata(moved.join(name_of(1, "yaml")))
        .and_then(|file| file.modified())
        .expect("the held file's time");
    assert_ne!(held, apart);
    assert_eq!(store.written(at(1, YAML)).expect("its time"), held);
    let refused = store
        .persist(at(2, YAML), b"two")
        .expect_err("the path leads elsewhere");
    assert_eq!(refused.to_string(), MOVED);
}

/// A directory renamed while a start waits for its lock, another laid
/// at its path, refuses that start by name once the lock is held: the
/// directory it locked is no longer the one its path names. A new
/// start takes the one that is, whole.
#[cfg(unix)]
#[test]
fn a_directory_replaced_under_a_waiting_start_refuses_that_start() {
    let base = tempfile::tempdir().expect("a temp dir");
    let (dir, moved) = (base.path().join("store"), base.path().join("moved"));
    fs::create_dir(&dir).expect("the directory");
    let mut holder = SpecStore::open(dir.clone()).expect("the holder");
    holder
        .persist(at(1, YAML), b"held")
        .expect("the holder's write");
    let waiter = Waiter::held_off(dir.clone());
    fs::rename(&dir, &moved).expect("the directory moves");
    fs::create_dir(&dir).expect("another stands at its path");
    fs::write(dir.join(name_of(1, "yaml")), "other").expect("the other's first");
    fs::write(dir.join(name_of(2, "yaml")), "newer").expect("the other's second");
    drop(holder);
    let refused = waiter
        .start
        .join()
        .expect("the waiter")
        .expect_err("the directory moved under the wait");
    assert_eq!(refused, StoreRefused::Moved);
    let store = SpecStore::open(dir.clone()).expect("a new start");
    assert_eq!(store.kept(), [at(1, YAML), at(2, YAML)]);
    assert_eq!(store.read(at(2, YAML)).expect("it reads"), b"newer");
    assert_eq!(names(&moved), BTreeSet::from([name_of(1, "yaml")]));
}

/// A directory renamed while a start waits for its lock, with nothing
/// laid at its path, refuses that start as a path to no directory.
#[cfg(unix)]
#[test]
fn a_directory_gone_under_a_waiting_start_refuses_that_start() {
    let base = tempfile::tempdir().expect("a temp dir");
    let (dir, moved) = (base.path().join("store"), base.path().join("moved"));
    fs::create_dir(&dir).expect("the directory");
    let holder = SpecStore::open(dir.clone()).expect("the holder");
    let waiter = Waiter::held_off(dir.clone());
    fs::rename(&dir, &moved).expect("the directory moves");
    drop(holder);
    let refused = waiter
        .start
        .join()
        .expect("the waiter")
        .expect_err("the path leads nowhere");
    assert_eq!(refused, StoreRefused::NotADirectory);
}

/// Where a directory is renamed under a waiting start and a file is
/// laid at its path, that start is refused as a new start is: as a
/// path to no directory.
#[cfg(unix)]
#[test]
fn a_file_laid_under_a_waiting_start_refuses_it_as_a_new_start_is() {
    let base = tempfile::tempdir().expect("a temp dir");
    let (dir, moved) = (base.path().join("store"), base.path().join("moved"));
    fs::create_dir(&dir).expect("the directory");
    let holder = SpecStore::open(dir.clone()).expect("the holder");
    let waiter = Waiter::held_off(dir.clone());
    fs::rename(&dir, &moved).expect("the directory moves");
    fs::write(&dir, "a file").expect("a file stands at its path");
    drop(holder);
    let refused = waiter
        .start
        .join()
        .expect("the waiter")
        .expect_err("no directory stands at the path");
    assert_eq!(refused, StoreRefused::NotADirectory);
    let new = SpecStore::open(dir).expect_err("a file is no store");
    assert_eq!(new, StoreRefused::NotADirectory);
}

/// A removal goes through the directory the store holds. Where the
/// path leads to another one by then, a refused generation still goes
/// off file there and the generations under the retained ones are
/// removed there: nothing of the other directory is touched.
#[cfg(unix)]
#[test]
fn a_removal_under_a_path_that_leads_elsewhere_acts_on_the_directory_held() {
    let base = tempfile::tempdir().expect("a temp dir");
    let (dir, moved) = (base.path().join("store"), base.path().join("moved"));
    fs::create_dir(&dir).expect("the directory");
    let mut store = SpecStore::open(dir.clone()).expect("the store");
    let count = u64::try_from(RETAINED).expect("a count") + 2;
    for generation in 1..=count {
        store
            .persist(at(generation, YAML), b"one")
            .expect("a write");
    }
    fs::rename(&dir, &moved).expect("the directory moves");
    fs::create_dir(&dir).expect("another stands at its path");
    fs::write(dir.join(name_of(1, "yaml")), "other").expect("the other's own");
    store.forget().expect("the refused one goes off file");
    let pruned = store.prune();
    assert!(matches!(pruned.as_slice(), [(1, Ok(()))]), "{pruned:?}");
    assert_eq!(names(&dir), BTreeSet::from([name_of(1, "yaml")]));
    let held = names(&moved);
    assert_eq!(held.len(), RETAINED, "the retained ones stay: {held:?}");
    assert!(!held.contains(&name_of(count, "yaml")), "the refused one");
    assert!(!held.contains(&name_of(1, "yaml")), "the one under them");
}

/// The write behind its check goes through the directory the store
/// holds, every step: a path that turns once the check has passed
/// leads the write nowhere else, and it lands where the lock is held.
/// Its sync is that directory's, and its look finds a number on file
/// there.
#[cfg(unix)]
#[test]
fn a_write_whose_path_turns_behind_its_check_lands_in_the_directory_held() {
    use std::os::unix::fs::MetadataExt as _;

    let base = tempfile::tempdir().expect("a temp dir");
    let (dir, moved) = (base.path().join("store"), base.path().join("moved"));
    fs::create_dir(&dir).expect("the directory");
    let mut store = SpecStore::open(dir.clone()).expect("the store");
    store.persist(at(1, YAML), b"one").expect("a write");
    fs::rename(&dir, &moved).expect("the directory moves");
    fs::create_dir(&dir).expect("another stands at its path");
    let mut synced = Vec::new();
    store
        .write_held(at(2, YAML), b"two", |held| {
            synced.push(held.metadata()?.ino());
            Ok(())
        })
        .expect("the write lands");
    let both = BTreeSet::from([name_of(1, "yaml"), name_of(2, "yaml")]);
    assert_eq!(names(&moved), both, "where the lock is held");
    assert!(names(&dir).is_empty(), "nothing lands where the path leads");
    assert_eq!(store.read(at(2, YAML)).expect("it reads"), b"two");
    let held = fs::metadata(&moved).expect("the directory held").ino();
    assert_eq!(synced, [held], "the sync is the held directory's");
    let taken = store
        .write_held(at(1, Format::Json), b"{}", |_| Ok(()))
        .expect_err("generation 1 is on file where the lock is held");
    assert_eq!(taken.to_string(), "generation 1 is on file already");
}

/// A write that backs out removes its own names, in the directory the
/// store holds. Where the path leads to another directory by then,
/// that one keeps a generation under the same number, and the
/// directory held keeps none of the refused write.
#[cfg(unix)]
#[test]
fn a_back_out_under_a_path_that_turned_removes_the_writes_own_name() {
    let base = tempfile::tempdir().expect("a temp dir");
    let (dir, moved) = (base.path().join("store"), base.path().join("moved"));
    fs::create_dir(&dir).expect("the directory");
    let mut store = SpecStore::open(dir.clone()).expect("the store");
    store.persist(at(1, YAML), b"one").expect("a write");
    let mut syncs = 0;
    let refused = store
        .persist_synced(at(2, YAML), b"two", |_| {
            syncs += 1;
            if syncs > 1 {
                return Ok(());
            }
            // The link is made: the path turns before the sync fails.
            fs::rename(&dir, &moved).expect("the directory moves");
            fs::create_dir(&dir).expect("another stands at its path");
            fs::write(dir.join(name_of(2, "yaml")), "other").expect("the other's own");
            Err(io::Error::other("the directory did not sync"))
        })
        .expect_err("no sync");
    assert_eq!(refused.to_string(), "the directory did not sync");
    assert_eq!(names(&dir), BTreeSet::from([name_of(2, "yaml")]));
    assert_eq!(names(&moved), BTreeSet::from([name_of(1, "yaml")]));
    assert_eq!(store.kept(), [at(1, YAML)]);
}

/// A pipe its user may write under the lock file's name takes no lock:
/// the open to read and write does not wait on it either, and what is
/// no regular file is refused by its kind.
#[cfg(unix)]
#[test]
fn a_pipe_under_the_locks_name_takes_no_lock() {
    let dir = tempfile::tempdir().expect("a temp dir");
    pipe_at(&dir.path().join(HELD));
    let refused = SpecStore::open_within(dir.path().to_owned(), Duration::ZERO)
        .expect_err("no lock on a pipe");
    assert_eq!(refused, StoreRefused::Unlocked(NOT_A_FILE.to_owned()));
}

/// A write that never ended leaves a temporary name, and a start
/// removes it. Where this user may not, by the directory's bits, the
/// name is left in place and the start goes on: no start reads it.
#[cfg(unix)]
#[test]
#[expect(
    clippy::print_stdout,
    reason = "a run the bits do not hold in reports it on stdout under --nocapture"
)]
fn a_temporary_name_a_start_may_not_remove_is_left_in_place() {
    use std::os::unix::fs::PermissionsExt as _;

    let dir = tempfile::tempdir().expect("a temp dir");
    let mut store = SpecStore::open(dir.path().to_owned()).expect("the store");
    store.persist(at(1, YAML), b"one").expect("a write");
    drop(store);
    let left = dir.path().join(format!("{WRITING}left"));
    fs::write(&left, "half").expect("a write that never ended");
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).expect("its bits");
    if fs::File::create(dir.path().join("made")).is_ok() {
        // The bits do not hold for this user (the superuser): the name
        // is removed, and there is nothing left in place to read here.
        println!(
            "a temporary name a start may not remove: this run asserted another lawful \
                 world: the directory's bits do not hold for this user"
        );
        return;
    }
    // This start waits: a child process of another case may hold a copy
    // of the lock's open file until its exec.
    let opened = SpecStore::open(dir.path().to_owned());
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).expect("its bits");
    let store = opened.expect("the start goes on");
    assert_eq!(store.kept(), [at(1, YAML)]);
    assert!(left.is_file(), "the name is left in place");
}

/// A temporary name whose removal is refused in another way than by
/// this user's bits or a file system mounted to read alone refuses the
/// start: here a directory stands under such a name, which no system
/// removes as a file.
#[test]
fn a_temporary_name_whose_removal_fails_otherwise_refuses_the_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    fs::create_dir(dir.path().join(format!("{WRITING}dir"))).expect("a directory");
    let refused = SpecStore::open_within(dir.path().to_owned(), Duration::ZERO)
        .expect_err("the name is not removed");
    assert!(
        matches!(refused, StoreRefused::Unreadable(_)),
        "{refused:?}"
    );
}

/// A temporary name that is gone when a start comes to remove it is
/// taken as removed, which is what the start wanted: a live writer
/// took its own name away after its link, where the file system's
/// locks do not hold two apart.
#[test]
fn a_temporary_name_gone_before_its_removal_is_taken_as_removed() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let (_, root) = SpecStore::rooted(dir.path()).expect("the directory");
    SpecStore::clear(&root, &format!("{WRITING}gone")).expect("taken as removed");
}

/// A directory under a temporary name refuses the start behind the
/// store's bits too. There the system refuses its removal as it
/// refuses a file's, by permission, and the name is told from a write
/// that never ended by its kind.
#[cfg(unix)]
#[test]
fn a_directory_under_a_temporary_name_refuses_the_start_behind_the_stores_bits() {
    use std::os::unix::fs::PermissionsExt as _;

    let dir = tempfile::tempdir().expect("a temp dir");
    drop(SpecStore::open(dir.path().to_owned()).expect("the store"));
    fs::create_dir(dir.path().join(format!("{WRITING}dir"))).expect("a directory");
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).expect("its bits");
    let opened = SpecStore::open(dir.path().to_owned());
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).expect("its bits");
    let refused = opened.expect_err("a directory is no write that never ended");
    assert!(
        matches!(&refused, StoreRefused::Unreadable(told) if told.contains("directory")),
        "{refused:?}"
    );
}

/// A link under a temporary name is judged as the name it is, not by
/// what it leads to: behind the store's bits one that leads to a
/// directory refuses no start, where a directory itself does.
#[cfg(unix)]
#[test]
fn a_link_to_a_directory_under_a_temporary_name_refuses_no_start() {
    use std::os::unix::fs::PermissionsExt as _;

    let dir = tempfile::tempdir().expect("a temp dir");
    drop(SpecStore::open(dir.path().to_owned()).expect("the store"));
    let led = tempfile::tempdir().expect("another directory");
    symlink(led.path(), dir.path().join(format!("{WRITING}link"))).expect("a link");
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).expect("its bits");
    let opened = SpecStore::open(dir.path().to_owned());
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).expect("its bits");
    drop(opened.expect("a link is no directory"));
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
    let unsynced = |_: &fs::File| {
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
    let stuck = |_: &fs::File| {
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

/// The second try of a refused generation's removal is answered by no
/// call, so each failure of it is said in a line: a name that stays,
/// and a directory that does not sync once the name went.
#[test]
fn a_second_try_that_fails_is_said_in_a_line() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut store = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
    store.persist(at(1, YAML), b"one").expect("it persists");
    let name = dir.path().join(name_of(1, "yaml"));
    fs::remove_file(&name).expect("the persisted file");
    fs::create_dir(&name).expect("a directory in its place");
    store
        .forget()
        .expect_err("no removal of a file takes a directory");

    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    // A second subscriber stays registered beside the case's own, as in
    // the door's audit case: with one alone a line is on or off by the
    // thread that reaches it first.
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);
    let said = |words: &str| {
        let text = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        text.lines().filter(|line| line.contains(words)).count()
    };

    store.persist(at(2, YAML), b"two").expect("it persists");
    let stayed = said("stays on file: a start may run it generation=1");

    fs::remove_dir(&name).expect("the directory goes");
    fs::write(&name, b"one").expect("a file again");
    let mut syncs = 0;
    let unsynced_once = |root: &fs::File| {
        syncs += 1;
        if syncs == 1 {
            return Err(io::Error::other("the directory did not sync"));
        }
        root.sync_all()
    };
    store
        .persist_synced(at(3, YAML), b"three", unsynced_once)
        .expect("the write's own syncs hold");
    let unsynced = said("a crash may bring its name back generations=[1]");
    // The name stayed at one try and went at the next.
    assert_eq!((stayed, said("stays on file"), unsynced), (1, 1, 1));
    let on_file = BTreeSet::from([name_of(2, "yaml"), name_of(3, "yaml")]);
    assert_eq!(names(dir.path()), on_file);
}

/// A back-out whose linked name is gone already, another hand's work
/// between the link and the back-out, takes it as removed and says the
/// write's own failure alone.
#[test]
fn a_back_out_whose_linked_name_is_gone_says_the_writes_own_failure() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut store = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
    let name = dir.path().join(name_of(1, "yaml"));
    let mut syncs = 0;
    let taken = |_: &fs::File| {
        syncs += 1;
        if syncs > 1 {
            return Ok(());
        }
        fs::remove_file(&name).expect("another hand takes the linked name");
        Err(io::Error::other("the directory did not sync"))
    };
    let refused = store
        .persist_synced(at(1, YAML), b"one", taken)
        .expect_err("no sync");
    assert_eq!(refused.to_string(), "the directory did not sync");
    assert_eq!(store.kept(), []);
    assert!(!name.exists(), "the linked name is gone");
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
    let never = |_: &fs::File| Err(io::Error::other("the directory did not sync"));
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

/// A refused generation whose file another hand took off first is
/// taken as removed: the store forgets it and says no failure.
#[test]
fn a_refused_generation_gone_from_file_is_taken_as_removed() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut store = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
    for generation in [1, 2] {
        store
            .persist(at(generation, YAML), b"document")
            .expect("it persists");
    }
    fs::remove_file(dir.path().join(name_of(2, "yaml"))).expect("another hand");
    store.forget().expect("the newest is gone already");
    assert_eq!(store.kept(), [at(1, YAML)]);
}

/// A generation under the retained ones whose file is gone already is
/// taken as removed: the removal says so once, and no apply tries it
/// again.
#[test]
fn a_generation_gone_from_file_under_the_retained_ones_is_not_tried_again() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut store = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
    let count = u64::try_from(RETAINED).expect("a small count") + 1;
    for generation in 1..=count {
        store
            .persist(at(generation, YAML), b"document")
            .expect("it persists");
    }
    fs::remove_file(dir.path().join(name_of(1, "yaml"))).expect("another hand");
    let pruned = store.prune();
    assert!(matches!(pruned.as_slice(), [(1, Ok(()))]), "{pruned:?}");
    assert!(store.prune().is_empty(), "nothing is tried again");
    assert_eq!(store.kept().len(), RETAINED);
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
