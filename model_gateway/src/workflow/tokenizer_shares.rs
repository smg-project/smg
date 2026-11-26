//! One tokenizer for the names that load the same files
//!
//! The registry keys a tokenizer by name, so two model names whose
//! tokenizers load from one directory each parsed it and each held its
//! tables: tens of megabytes resident a name for a large vocabulary. The
//! registration's loader asks here first: a load handed the same source,
//! whose files read the same as those of a tokenizer still held, takes that
//! tokenizer. Each name keeps its own registry entry and id, and its own
//! cache wrapper around the shared tokenizer; a chat template file a name
//! hands in is part of what was read, so names with different templates
//! share nothing.
//!
//! Only weak references are kept: a tokenizer lives as long as an entry
//! holds it and no longer.

use std::{
    fs,
    future::Future,
    io::Read,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Weak},
    time::UNIX_EPOCH,
};

use dashmap::DashMap;
use llm_tokenizer::traits::Tokenizer;
use tokio::sync::Mutex;

/// The bytes one [`FilesRead`] hashes at most. A file longer than this
/// enters by path, size and change time alone: model weights sit beside the
/// tokenizer files, which are far smaller (such a file rewritten in place
/// under the same size and time reads the same). A tree whose other files
/// sum to more is not read for sharing: the sum is taken from the sizes,
/// before any file is opened.
const HASHED_BYTES: u64 = 64 * 1024 * 1024;

/// A tree with more entries than this, files and directories at any depth,
/// is not read for sharing.
const MAX_ENTRIES: usize = 256;

/// What a load from files reads, as one hash: the source as the load was
/// handed it, every file below the directory that holds the tokenizer (the
/// source itself, or the directory beside a source file) by path, size,
/// change time and bytes, and the chat template file handed in, as one more
/// file. A load reads files beside the one it is pointed at (the
/// tokenizer's config, a template) and in a directory below, so the whole
/// tree stands in; and it reads how its source is spelled, so two spellings
/// of one directory are two sources.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct FilesRead([u8; blake3::OUT_LEN]);

impl FilesRead {
    /// `None` where the load runs alone and reports its own error: a source
    /// that is no path on this host (a hub id), a file or a directory that
    /// cannot be read, a tree beyond [`MAX_ENTRIES`] or [`HASHED_BYTES`].
    pub(super) fn of(source: &str, chat_template: Option<&str>) -> Option<Self> {
        let given = Path::new(source);
        // A load reads beside the path it is pointed at, not beside what a
        // link to a file resolves to: the directory part alone is resolved,
        // once, so one read walks one place.
        let dir = if fs::metadata(given).ok()?.is_dir() {
            given
        } else {
            let dir = given.parent()?;
            if dir.as_os_str().is_empty() {
                Path::new(".")
            } else {
                dir
            }
        };

        let mut dirs = vec![fs::canonicalize(dir).ok()?];
        let mut files = Vec::new();
        let mut entries = 0_usize;
        while let Some(next) = dirs.pop() {
            for entry in fs::read_dir(next).ok()? {
                entries += 1;
                if entries > MAX_ENTRIES {
                    return None;
                }
                let path = entry.ok()?.path();
                // Through a link, as the load reads it.
                let Ok(metadata) = fs::metadata(&path) else {
                    continue;
                };
                if metadata.is_dir() {
                    dirs.push(path);
                } else if metadata.is_file() {
                    files.push((path, metadata));
                }
            }
        }
        files.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        let below = files.len();
        if let Some(template) = chat_template {
            files.push((PathBuf::from(template), fs::metadata(template).ok()?));
        }
        let to_hash: u64 = files
            .iter()
            .map(|(_, metadata)| metadata.len())
            .filter(|len| *len <= HASHED_BYTES)
            .sum();
        if to_hash > HASHED_BYTES {
            return None;
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(&(source.len() as u64).to_le_bytes());
        hasher.update(source.as_bytes());
        hasher.update(&(below as u64).to_le_bytes());
        for (path, metadata) in &files {
            let name = path.as_os_str().as_encoded_bytes();
            hasher.update(&(name.len() as u64).to_le_bytes());
            hasher.update(name);
            let changed = metadata.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
            hasher.update(&changed.as_nanos().to_le_bytes());
            let len = metadata.len();
            hasher.update(&len.to_le_bytes());
            if len <= HASHED_BYTES {
                // No more than the size summed above, and no less: a file
                // that shrank under the read is not what was summed.
                let mut bytes = fs::File::open(path).ok()?.take(len);
                hasher.update_reader(&mut bytes).ok()?;
                if bytes.limit() != 0 {
                    return None;
                }
            }
        }
        Some(Self(*hasher.finalize().as_bytes()))
    }
}

/// The tokenizer last loaded for one [`FilesRead`]; the mutex is held
/// through a load, so two loads of the same files are one.
type Slot = Arc<Mutex<Option<Weak<dyn Tokenizer>>>>;

/// The tokenizers loaded from files and still held, by what was read.
#[derive(Default)]
pub(super) struct TokenizerShares {
    slots: DashMap<FilesRead, Slot>,
}

/// The gateway's shares: one table a process, as a tokenizer is the same
/// whichever registry holds it.
pub(super) fn shares() -> &'static TokenizerShares {
    static SHARES: LazyLock<TokenizerShares> = LazyLock::new(TokenizerShares::default);
    &SHARES
}

impl TokenizerShares {
    /// The tokenizer of the files `read` names: the living tokenizer loaded
    /// for the same files, else the one `load` makes.
    ///
    /// `read` runs before the load and again behind it: the tokenizer is
    /// recorded only where both agree, so one loaded while its files were
    /// rewritten is never handed to a later load. A `read` of `None` (see
    /// [`FilesRead::of`]) loads alone and records nothing.
    pub(super) async fn load<F, Fut, E>(
        &self,
        read: impl Fn() -> Option<FilesRead>,
        load: F,
    ) -> Result<Arc<dyn Tokenizer>, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Arc<dyn Tokenizer>, E>>,
    {
        let Some(files) = read() else {
            return load().await;
        };

        // A key whose tokenizer is gone and that no load holds is forgotten:
        // a rewritten file leaves its old key behind otherwise.
        self.slots.retain(|_, slot| {
            Arc::strong_count(slot) > 1
                || slot.try_lock().map_or(true, |held| {
                    held.as_ref().is_some_and(|held| held.strong_count() > 0)
                })
        });

        let slot = Arc::clone(&self.slots.entry(files.clone()).or_default());
        let mut held = slot.lock().await;
        if let Some(tokenizer) = held.as_ref().and_then(Weak::upgrade) {
            return Ok(tokenizer);
        }

        let tokenizer = load().await?;
        if read().as_ref() == Some(&files) {
            *held = Some(Arc::downgrade(&tokenizer));
        }
        Ok(tokenizer)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        convert::Infallible,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use llm_tokenizer::MockTokenizer;

    use super::*;
    use crate::workflow::tokenizer_registration::test_support::tokenizer_dir;

    fn mock() -> Arc<dyn Tokenizer> {
        Arc::new(MockTokenizer::new())
    }

    fn source(dir: &tempfile::TempDir) -> String {
        dir.path().to_str().unwrap().to_owned()
    }

    #[test]
    fn what_was_read_names_the_directory_whichever_file_of_it_is_the_source() {
        let dir = tokenizer_dir();
        let by_dir = FilesRead::of(&source(&dir), None).unwrap();
        assert!(by_dir == FilesRead::of(&source(&dir), None).unwrap());

        // The file is another source than its directory, and a source that
        // is no path has nothing read.
        let file = dir.path().join("tokenizer.json");
        let by_file = FilesRead::of(file.to_str().unwrap(), None).unwrap();
        assert!(by_dir != by_file);
        assert!(FilesRead::of("owner/model", None).is_none());

        // A file beside the tokenizer is read too, whichever the source.
        fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();
        assert!(by_dir != FilesRead::of(&source(&dir), None).unwrap());
        assert!(by_file != FilesRead::of(file.to_str().unwrap(), None).unwrap());
    }

    #[test]
    fn a_tree_beyond_the_limits_is_not_read() {
        let dir = tokenizer_dir();
        let read = || FilesRead::of(&source(&dir), None);
        let sized = |name: &str, len: u64| {
            let file = fs::File::create(dir.path().join(name)).unwrap();
            file.set_len(len).unwrap();
        };

        // A file above the limit enters by its size, unread ...
        sized("weights", HASHED_BYTES + 1);
        let before = read().unwrap();
        sized("weights", HASHED_BYTES + 2);
        assert!(before != read().unwrap());
        // ... and smaller files that sum above it leave the load alone.
        sized("half", HASHED_BYTES / 2);
        sized("other half", HASHED_BYTES / 2);
        assert!(read().is_none());

        // As many entries as the limit are read, one more is not.
        let dir = tokenizer_dir();
        for entry in 2..MAX_ENTRIES {
            fs::write(dir.path().join(entry.to_string()), "").unwrap();
        }
        fs::create_dir(dir.path().join("below")).unwrap();
        assert!(FilesRead::of(&source(&dir), None).is_some());
        fs::write(dir.path().join("below").join("one more"), "").unwrap();
        assert!(FilesRead::of(&source(&dir), None).is_none());
    }

    #[test]
    fn bytes_rewritten_under_the_same_size_and_change_time_read_differently() {
        let dir = tokenizer_dir();
        let file = dir.path().join("tokenizer.json");
        let before = FilesRead::of(&source(&dir), None).unwrap();
        let metadata = fs::metadata(&file).unwrap();

        let rewritten = fs::read_to_string(&file).unwrap().replace("hello", "hallo");
        fs::write(&file, rewritten).unwrap();
        fs::File::options()
            .write(true)
            .open(&file)
            .unwrap()
            .set_modified(metadata.modified().unwrap())
            .unwrap();

        let after = fs::metadata(&file).unwrap();
        assert_eq!(after.len(), metadata.len());
        assert_eq!(after.modified().unwrap(), metadata.modified().unwrap());
        assert!(before != FilesRead::of(&source(&dir), None).unwrap());
    }

    #[test]
    fn a_chat_template_file_is_part_of_what_was_read() {
        let dir = tokenizer_dir();
        let templates = tempfile::tempdir().unwrap();
        let template = templates.path().join("template.jinja");
        fs::write(&template, "one").unwrap();
        let template = template.to_str().unwrap();

        let plain = FilesRead::of(&source(&dir), None).unwrap();
        let with_template = FilesRead::of(&source(&dir), Some(template)).unwrap();
        assert!(plain != with_template);

        fs::write(template, "two").unwrap();
        assert!(with_template != FilesRead::of(&source(&dir), Some(template)).unwrap());
        // A template that cannot be read leaves the load alone.
        assert!(FilesRead::of(&source(&dir), Some("/no/such/template.jinja")).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_link_to_a_tokenizer_file_reads_the_directory_of_the_link() {
        use std::os::unix::fs::symlink;

        let shared = tokenizer_dir();
        let linked = || {
            let dir = tempfile::tempdir().unwrap();
            symlink(
                shared.path().join("tokenizer.json"),
                dir.path().join("tokenizer.json"),
            )
            .unwrap();
            dir
        };
        let (one, other) = (linked(), linked());
        let read = |dir: &tempfile::TempDir| {
            FilesRead::of(dir.path().join("tokenizer.json").to_str().unwrap(), None).unwrap()
        };
        assert!(read(&one) != read(&other));

        // A file beside the link is read, one beside its target is not.
        let before = read(&one);
        fs::write(shared.path().join("tokenizer_config.json"), "{}").unwrap();
        assert!(before == read(&one));
        fs::write(one.path().join("tokenizer_config.json"), "{}").unwrap();
        assert!(before != read(&one));

        // A link to the directory is another spelling, so another source.
        let link = one.path().join("link");
        symlink(shared.path(), &link).unwrap();
        let by_link = FilesRead::of(link.to_str().unwrap(), None).unwrap();
        assert!(by_link != FilesRead::of(&source(&shared), None).unwrap());
    }

    #[test]
    fn a_file_below_the_directory_is_part_of_what_was_read() {
        let dir = tokenizer_dir();
        let below = dir.path().join("one").join("two");
        fs::create_dir_all(&below).unwrap();
        for file in [dir.path().join("one").join("file"), below.join("file")] {
            fs::write(&file, "old").unwrap();
            let before = FilesRead::of(&source(&dir), None).unwrap();
            fs::write(&file, "new").unwrap();
            assert!(before != FilesRead::of(&source(&dir), None).unwrap());
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn loads_at_once_of_the_same_files_run_one_load() {
        let dir = tokenizer_dir();
        let shares = Arc::new(TokenizerShares::default());
        let loads = Arc::new(AtomicUsize::new(0));

        let tasks: Vec<_> = (0..8)
            .map(|_| {
                let (shares, loads, source) = (shares.clone(), loads.clone(), source(&dir));
                #[expect(
                    clippy::disallowed_methods,
                    reason = "the test awaits the task it spawns"
                )]
                tokio::spawn(async move {
                    shares
                        .load(
                            || FilesRead::of(&source, None),
                            || async {
                                loads.fetch_add(1, Ordering::SeqCst);
                                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                                Ok::<_, Infallible>(mock())
                            },
                        )
                        .await
                        .unwrap()
                })
            })
            .collect();
        let mut held = Vec::new();
        for task in tasks {
            held.push(task.await.unwrap());
        }

        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(held
            .iter()
            .all(|tokenizer| Arc::ptr_eq(tokenizer, &held[0])));
    }

    #[tokio::test]
    async fn a_tokenizer_loaded_while_its_files_were_rewritten_is_not_recorded() {
        let dir = tokenizer_dir();
        let shares = TokenizerShares::default();
        let read = || FilesRead::of(&source(&dir), None);

        let first = shares
            .load(read, || async {
                fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();
                Ok::<_, Infallible>(mock())
            })
            .await
            .unwrap();
        let second = shares
            .load(read, || async { Ok::<_, Infallible>(mock()) })
            .await
            .unwrap();

        assert!(!Arc::ptr_eq(&first, &second));
    }

    #[tokio::test]
    async fn a_failed_load_and_a_dropped_tokenizer_leave_nothing_behind() {
        let dir = tokenizer_dir();
        let shares = TokenizerShares::default();
        let read = || FilesRead::of(&source(&dir), None);

        let failed = shares.load(read, || async { Err("no") }).await;
        assert!(failed.is_err());

        let tokenizer = shares
            .load(read, || async { Ok::<_, &str>(mock()) })
            .await
            .unwrap();
        let weak = Arc::downgrade(&tokenizer);
        drop(tokenizer);
        assert!(weak.upgrade().is_none());

        // The next load of any files forgets the key of the dropped one.
        let other = tokenizer_dir();
        let _held = shares
            .load(
                || FilesRead::of(&source(&other), None),
                || async { Ok::<_, &str>(mock()) },
            )
            .await
            .unwrap();
        assert_eq!(shares.len(), 1);
    }
}
