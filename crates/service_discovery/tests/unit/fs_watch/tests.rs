use super::*;

fn event(kind: EventKind, paths: &[&str]) -> Event {
    paths
        .iter()
        .fold(Event::new(kind), |event, path| event.add_path(path.into()))
}

fn names(names: &[&str]) -> Vec<OsString> {
    names.iter().map(OsString::from).collect()
}

/// A file is watched in its directory; a bare name has none, so it is
/// watched in the working directory.
#[test]
fn a_file_is_watched_in_its_directory_or_the_working_one() {
    assert_eq!(
        directory_of(Path::new("/etc/smg/api_key")),
        Path::new("/etc/smg")
    );
    assert_eq!(directory_of(Path::new("keys/api_key")), Path::new("keys"));
    assert_eq!(directory_of(Path::new("api_key")), Path::new("."));
}

/// Which events in the directory cause a reread.
#[test]
fn only_a_change_to_a_named_file_wakes() {
    use notify::event::{
        AccessKind, CreateKind, DataChange, Flag, ModifyKind, RemoveKind, RenameMode,
    };

    let watched = names(&["workers.json"]);
    let wakes = [
        // A writer renaming its temporary file over the file.
        event(
            EventKind::Modify(ModifyKind::Name(RenameMode::Both)),
            &["/run/smg/workers.json.tmp", "/run/smg/workers.json"],
        ),
        event(
            EventKind::Modify(ModifyKind::Name(RenameMode::To)),
            &["/run/smg/workers.json"],
        ),
        // Written in place, created late, or deleted.
        event(
            EventKind::Modify(ModifyKind::Data(DataChange::Content)),
            &["/run/smg/workers.json"],
        ),
        event(
            EventKind::Create(CreateKind::File),
            &["/run/smg/workers.json"],
        ),
        event(
            EventKind::Remove(RemoveKind::File),
            &["/run/smg/workers.json"],
        ),
        // Kubernetes swapping a ConfigMap's `..data` symlink.
        event(
            EventKind::Modify(ModifyKind::Name(RenameMode::To)),
            &["/etc/smg/..data"],
        ),
        event(EventKind::Create(CreateKind::Any), &["/etc/smg/..data_tmp"]),
        // The watch dropped events; anything may have changed.
        event(EventKind::Other, &[]).set_flag(Flag::Rescan),
    ];
    for event in &wakes {
        assert!(concerns(event, &watched), "{event:?} should wake");
    }

    let ignored = [
        // The reader's own read.
        event(
            EventKind::Access(AccessKind::Read),
            &["/run/smg/workers.json"],
        ),
        event(
            EventKind::Access(AccessKind::Any),
            &["/run/smg/workers.json"],
        ),
        // Another file in the same directory, including the writer's
        // temporary file before its rename.
        event(
            EventKind::Modify(ModifyKind::Data(DataChange::Any)),
            &["/run/smg/workers.json.tmp"],
        ),
        event(
            EventKind::Create(CreateKind::File),
            &["/run/smg/other.json"],
        ),
    ];
    for event in &ignored {
        assert!(!concerns(event, &watched), "{event:?} should not wake");
    }

    // Several files in one directory: any of them wakes.
    let keys = names(&["api-key", "tenant-keys"]);
    assert!(concerns(
        &event(
            EventKind::Modify(ModifyKind::Data(DataChange::Content)),
            &["/etc/smg/tenant-keys"]
        ),
        &keys
    ));
}

async fn woken(wake: &mut mpsc::Receiver<()>, within: Duration) -> bool {
    time::timeout(within, wake.recv()).await.is_ok()
}

/// The real watcher: a rename over the file wakes, a burst of writes
/// settles into one wake, and another file in the directory does not
/// wake.
#[tokio::test]
async fn the_watch_reports_the_named_file_and_only_that_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.json");
    let (tx, mut rx) = mpsc::channel(1);
    let _watch = DirectoryWatch::of_file(&path, tx).unwrap();

    let staged = dir.path().join("workers.json.staged");
    std::fs::write(&staged, "{}").unwrap();
    std::fs::rename(&staged, &path).unwrap();
    assert!(
        woken(&mut rx, Duration::from_secs(10)).await,
        "rename not seen"
    );
    settle(&mut rx).await;

    for _ in 0..20 {
        std::fs::write(&path, "{}").unwrap();
    }
    assert!(
        woken(&mut rx, Duration::from_secs(10)).await,
        "writes not seen"
    );
    settle(&mut rx).await;
    assert!(rx.try_recv().is_err(), "the burst woke twice");

    // Let any late event from the burst arrive and be drained first.
    time::sleep(Duration::from_secs(1)).await;
    while rx.try_recv().is_ok() {}
    std::fs::write(dir.path().join("other.json"), "{}").unwrap();
    assert!(
        !woken(&mut rx, Duration::from_secs(1)).await,
        "another file woke the watch"
    );
}

/// A wake that arrives while a burst settles is drained into that
/// burst: with two named files, a write inside the settle window leaves
/// no wake behind the settle, and the watch still wakes for the next
/// change. Each file's burst is taken and settled first, so the channel
/// is empty when the window opens and the second file is known to wake
/// on its own. One file's burst cannot pin this: its writes all land
/// before the first wake is taken, so the channel's one slot drops all
/// but one wake and the settle finds nothing to drain.
#[tokio::test]
async fn a_wake_during_the_settle_is_drained_into_the_burst() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("workers.json");
    let second = dir.path().join("tenants.json");
    let (tx, mut rx) = mpsc::channel(1);
    let _watch =
        DirectoryWatch::start(dir.path(), names(&["workers.json", "tenants.json"]), tx).unwrap();

    std::fs::write(&first, "{}").unwrap();
    assert!(
        woken(&mut rx, Duration::from_secs(10)).await,
        "the first file's write not seen"
    );
    settle(&mut rx).await;
    // A distinct event, never coalesced with the first file's, and the
    // channel is empty: this wake is the second file's own.
    std::fs::write(&second, "{}").unwrap();
    assert!(
        woken(&mut rx, Duration::from_secs(10)).await,
        "the second file's write not seen"
    );
    settle(&mut rx).await;

    // The write lands early in the settle window, its event well before
    // the window ends: the settle drains the wake it sends.
    let write_in_the_window = async {
        time::sleep(SETTLE / 10).await;
        std::fs::write(&second, "{}").unwrap();
    };
    tokio::join!(settle(&mut rx), write_in_the_window);
    assert!(
        rx.try_recv().is_err(),
        "the wake inside the settle window was not drained into the burst"
    );

    std::fs::write(&first, "{}").unwrap();
    assert!(
        woken(&mut rx, Duration::from_secs(10)).await,
        "the watch did not wake after the settle"
    );
}

/// Only an event about the watched directory itself, removed or moved
/// away, marks the watch lost; events about files inside it do not.
#[test]
fn only_the_directory_going_away_loses_the_watch() {
    use notify::event::{CreateKind, RemoveKind, RenameMode};

    let dir = [PathBuf::from("/run/smg")];
    for event in [
        // inotify's DELETE_SELF and MOVE_SELF.
        event(EventKind::Remove(RemoveKind::Folder), &["/run/smg"]),
        event(
            EventKind::Modify(ModifyKind::Name(RenameMode::From)),
            &["/run/smg"],
        ),
    ] {
        assert!(removes_directory(&event, &dir), "{event:?}");
    }
    for event in [
        event(
            EventKind::Remove(RemoveKind::File),
            &["/run/smg/workers.json"],
        ),
        event(
            EventKind::Modify(ModifyKind::Name(RenameMode::Both)),
            &["/run/smg/workers.json.tmp", "/run/smg/workers.json"],
        ),
        event(EventKind::Create(CreateKind::Folder), &["/run/smg"]),
    ] {
        assert!(!removes_directory(&event, &dir), "{event:?}");
    }
}

/// Removing the watched directory marks the watch lost and wakes the
/// reader, which then watches the path again.
#[tokio::test]
async fn removing_the_directory_loses_the_watch() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("smg");
    std::fs::create_dir(&dir).unwrap();
    let (tx, mut rx) = mpsc::channel(1);
    let watch = DirectoryWatch::of_file(&dir.join("workers.json"), tx).unwrap();
    assert!(!watch.is_lost());

    std::fs::remove_dir(&dir).unwrap();
    assert!(
        woken(&mut rx, Duration::from_secs(10)).await,
        "removal not seen"
    );
    let deadline = time::Instant::now() + Duration::from_secs(10);
    while !watch.is_lost() {
        assert!(time::Instant::now() < deadline, "the watch never lost");
        time::sleep(Duration::from_millis(50)).await;
    }
}

/// The directory must exist to be watched. The `stat` that refuses it
/// names the cause and the path.
#[test]
fn a_missing_directory_cannot_be_watched() {
    let dir = tempfile::tempdir().unwrap();
    let (tx, _rx) = mpsc::channel(1);
    let Err(error) = DirectoryWatch::of_file(&dir.path().join("absent/workers.json"), tx.clone())
    else {
        panic!("watched a missing directory");
    };
    assert!(
        matches!(&error.kind, notify::ErrorKind::Io(io) if io.kind() == std::io::ErrorKind::NotFound),
        "{error}"
    );
    assert!(
        error.paths.iter().any(|path| path.ends_with("absent")),
        "{error}"
    );

    let file = dir.path().join("a-file");
    std::fs::write(&file, "").unwrap();
    assert!(DirectoryWatch::of_file(&file.join("workers.json"), tx).is_err());
}
