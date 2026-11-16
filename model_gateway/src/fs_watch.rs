//! A watch on one directory for the files a running gateway reads again:
//! the worker-discovery manifest and the mounted key files. The directory
//! rather than the file: a writer that renames a new file over the watched
//! one replaces its inode, and a watch on the old inode never reports it.
//! Kubernetes updates a mounted ConfigMap or Secret by swapping a `..data`
//! symlink beside the files, never touching their names, so that swap wakes
//! the watch too. A burst of events settles into one wake.

use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use notify::{event::ModifyKind, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::{sync::mpsc, time};

/// How long a burst of filesystem events is left to settle before the one
/// reread it causes. A writer's rename, or an editor's several writes, arrive
/// as a handful of events within a few milliseconds.
pub(crate) const SETTLE: Duration = Duration::from_millis(200);

/// The directory a file is watched in: its parent, or the working directory
/// for a bare name, which has none.
pub(crate) fn directory_of(path: &Path) -> &Path {
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    }
}

/// A watch on a directory for the named files in it. Dropping it stops the
/// watch.
pub(crate) struct DirectoryWatch {
    _watcher: RecommendedWatcher,
    /// Set once the watched directory itself is removed or moved away. The
    /// watch then reports nothing more, not even for a directory recreated at
    /// the same path, so it has to be started again.
    lost: Arc<AtomicBool>,
}

impl DirectoryWatch {
    /// The watch for one file: its directory, for its name.
    pub(crate) fn of_file(path: &Path, wake: mpsc::Sender<()>) -> notify::Result<Self> {
        let names = path
            .file_name()
            .map(OsStr::to_os_string)
            .into_iter()
            .collect();
        Self::start(directory_of(path), names, wake)
    }

    /// Watch `dir` and send on `wake` whenever one of the `names` in it may
    /// have changed, or the watch stopped working. A send that finds the
    /// channel full is dropped, because a wake is already pending and the
    /// reread it causes reads the latest files.
    pub(crate) fn start(
        dir: &Path,
        names: Vec<OsString>,
        wake: mpsc::Sender<()>,
    ) -> notify::Result<Self> {
        // Watched by its absolute path, which is how inotify names the
        // directory in an event about the directory itself. FSEvents names it
        // resolved.
        let dir = std::path::absolute(dir).map_err(notify::Error::io)?;
        // Checked before building the watcher, which on Linux starts an
        // inotify instance and a thread even when the watch then fails.
        match std::fs::metadata(&dir) {
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => return Err(notify::Error::generic("not a directory").add_path(dir)),
            Err(error) => return Err(notify::Error::io(error).add_path(dir)),
        }
        let dir_names: Vec<PathBuf> = [Some(dir.clone()), dir.canonicalize().ok()]
            .into_iter()
            .flatten()
            .collect();
        let lost = Arc::new(AtomicBool::new(false));
        let lost_by_event = Arc::clone(&lost);
        let mut watcher = notify::recommended_watcher(move |event: notify::Result<Event>| {
            let wakes = match event {
                Ok(event) if removes_directory(&event, &dir_names) => {
                    lost_by_event.store(true, Ordering::Relaxed);
                    true
                }
                Ok(event) => concerns(&event, &names),
                // A failed watch may have dropped a change, so it rereads too.
                Err(_) => true,
            };
            if wakes {
                let _ = wake.try_send(());
            }
        })?;
        watcher.watch(&dir, RecursiveMode::NonRecursive)?;
        Ok(Self {
            _watcher: watcher,
            lost,
        })
    }

    pub(crate) fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Relaxed)
    }
}

/// Whether an event says the watched directory itself was removed or moved
/// away, given the names an event may call it by.
fn removes_directory(event: &Event, dir_names: &[PathBuf]) -> bool {
    matches!(
        event.kind,
        EventKind::Remove(_) | EventKind::Modify(ModifyKind::Name(_))
    ) && event.paths.iter().any(|path| dir_names.contains(path))
}

/// Whether an event in the watched directory may have changed one of the
/// named files.
fn concerns(event: &Event, names: &[OsString]) -> bool {
    if event.need_rescan() {
        return true;
    }
    // Reads change nothing, and the reader's own reread is one of them.
    if matches!(event.kind, EventKind::Access(_)) {
        return false;
    }
    event.paths.iter().any(|path| {
        path.file_name().is_some_and(|file| {
            // Kubernetes updates a mounted ConfigMap or Secret by swapping a
            // `..data` symlink beside the files, never touching their names.
            names.iter().any(|name| name.as_os_str() == file)
                || file.as_encoded_bytes().starts_with(b"..data")
        })
    })
}

/// Let a burst of events settle, then take every wake it sent as one.
pub(crate) async fn settle(wake: &mut mpsc::Receiver<()>) {
    time::sleep(SETTLE).await;
    while wake.try_recv().is_ok() {}
}

#[cfg(test)]
mod tests {
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
            DirectoryWatch::start(dir.path(), names(&["workers.json", "tenants.json"]), tx)
                .unwrap();

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
        let Err(error) =
            DirectoryWatch::of_file(&dir.path().join("absent/workers.json"), tx.clone())
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
}
