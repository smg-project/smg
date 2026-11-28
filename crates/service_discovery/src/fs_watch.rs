//! A watch on one directory for the files a running host reads again when
//! they change: the file provider's manifest of workers, and files of the
//! host's own, such as mounted key files. The directory rather than the
//! file: a writer that renames a new file over the watched one replaces its
//! inode, and a watch on the old inode never reports it. Kubernetes updates
//! a mounted ConfigMap or Secret by swapping a `..data` symlink beside the
//! files, never touching their names, so that swap wakes the watch too. A
//! burst of events settles into one wake.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

/// The watcher package. A start answers with its error, so the package is
/// named here: a host reads that error by this crate's version of it and
/// pins none of its own.
pub use notify;
use notify::{event::ModifyKind, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::{sync::mpsc, time};

/// How long a burst of filesystem events is left to settle before the one
/// reread it causes. A writer's rename, or an editor's several writes, arrive
/// as a handful of events within a few milliseconds.
pub(crate) const SETTLE: Duration = Duration::from_millis(200);

/// The directory a file is watched in: its parent, or the working directory
/// for a bare name, which has none.
pub fn directory_of(path: &Path) -> &Path {
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    }
}

/// A watch on a directory for the named files in it. Dropping it stops the
/// watch.
pub struct DirectoryWatch {
    _watcher: RecommendedWatcher,
    /// Set once the watched directory itself is removed or moved away. The
    /// watch then reports nothing more, not even for a directory recreated at
    /// the same path, so it has to be started again.
    lost: Arc<AtomicBool>,
}

impl DirectoryWatch {
    /// The watch for one file: its directory, for its name. The file provider
    /// is its one caller, so it is built with that provider and for the cases.
    #[cfg(any(test, feature = "file"))]
    pub(crate) fn of_file(path: &Path, wake: mpsc::Sender<()>) -> notify::Result<Self> {
        let names = path.file_name().map(OsString::from).into_iter().collect();
        Self::start(directory_of(path), names, wake)
    }

    /// Watch `dir` and send on `wake` whenever one of the `names` in it may
    /// have changed, or the watch stopped working. A send that finds the
    /// channel full is dropped, because a wake is already pending and the
    /// reread it causes reads the latest files.
    pub fn start(dir: &Path, names: Vec<OsString>, wake: mpsc::Sender<()>) -> notify::Result<Self> {
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

    /// Whether the watched directory was removed or moved away. The watch
    /// reports nothing more then, so its owner starts a new one.
    pub fn is_lost(&self) -> bool {
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
pub async fn settle(wake: &mut mpsc::Receiver<()>) {
    time::sleep(SETTLE).await;
    while wake.try_recv().is_ok() {}
}

#[cfg(test)]
#[path = "../tests/unit/fs_watch/tests.rs"]
mod tests;
