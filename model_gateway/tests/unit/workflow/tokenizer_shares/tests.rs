use std::{
    convert::Infallible,
    io::Write as _,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use llm_tokenizer::MockTokenizer;
use tokio::sync::Notify;

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
    let read = {
        let source = source(&dir);
        move || FilesRead::of(&source, None)
    };
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
fn bytes_rewritten_under_the_same_size_and_modification_time_read_differently() {
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
                        move || FilesRead::of(&source, None),
                        || async {
                            loads.fetch_add(1, Ordering::SeqCst);
                            tokio::time::sleep(Duration::from_millis(20)).await;
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
    let read = {
        let source = source(&dir);
        move || FilesRead::of(&source, None)
    };

    let first = shares
        .load(read.clone(), || async {
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
    let read = {
        let source = source(&dir);
        move || FilesRead::of(&source, None)
    };

    let failed = shares.load(read.clone(), || async { Err("no") }).await;
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
    let other_source = source(&other);
    let _held = shares
        .load(
            move || FilesRead::of(&other_source, None),
            || async { Ok::<_, &str>(mock()) },
        )
        .await
        .unwrap();
    assert_eq!(shares.len(), 1);
}

/// One runtime thread: a read that ran on it would wait in vain for the
/// task below, which that thread alone can run.
#[tokio::test]
async fn a_read_holds_no_runtime_worker() {
    let shares = TokenizerShares::default();
    let ran = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(AtomicBool::new(false));
    #[expect(
        clippy::disallowed_methods,
        reason = "the test awaits the task it spawns"
    )]
    let other = tokio::spawn({
        let ran = ran.clone();
        async move { ran.store(true, Ordering::SeqCst) }
    });

    let read = {
        let (ran, seen) = (ran.clone(), seen.clone());
        move || {
            let until = Instant::now() + Duration::from_secs(5);
            while !ran.load(Ordering::SeqCst) && Instant::now() < until {
                std::thread::yield_now();
            }
            seen.store(ran.load(Ordering::SeqCst), Ordering::SeqCst);
            None
        }
    };
    shares
        .load(read, || async { Ok::<_, Infallible>(mock()) })
        .await
        .unwrap();
    other.await.unwrap();

    assert!(seen.load(Ordering::SeqCst));
}

#[tokio::test]
async fn a_load_cancelled_under_the_slot_lock_leaves_the_slot_to_the_next() {
    let dir = tokenizer_dir();
    let shares = Arc::new(TokenizerShares::default());
    let read = {
        let source = source(&dir);
        move || FilesRead::of(&source, None)
    };
    let entered = Arc::new(Notify::new());

    #[expect(
        clippy::disallowed_methods,
        reason = "the test awaits the task it spawns"
    )]
    let cancelled = tokio::spawn({
        let (shares, read, entered) = (shares.clone(), read.clone(), entered.clone());
        async move {
            shares
                .load(read, || {
                    entered.notify_one();
                    std::future::pending::<Result<Arc<dyn Tokenizer>, Infallible>>()
                })
                .await
        }
    });
    entered.notified().await;
    cancelled.abort();
    assert!(cancelled.await.is_err_and(|ended| ended.is_cancelled()));

    let next = shares
        .load(read.clone(), || async { Ok::<_, Infallible>(mock()) })
        .await
        .unwrap();
    let shared = shares
        .load(read, || async { Ok::<_, Infallible>(mock()) })
        .await
        .unwrap();
    assert!(Arc::ptr_eq(&next, &shared));
}

/// A file above the bound enters by path, size and time: its bytes
/// rewritten in place under both read the same.
#[test]
fn a_file_above_the_bound_enters_without_its_bytes() {
    let dir = tokenizer_dir();
    let file = dir.path().join("weights.bin");
    // The length alone is set: no block is written.
    fs::File::create(&file)
        .unwrap()
        .set_len(HASHED_BYTES + 1)
        .unwrap();
    let metadata = fs::metadata(&file).unwrap();
    let before = FilesRead::of(&source(&dir), None).unwrap();

    let mut rewritten = fs::File::options().write(true).open(&file).unwrap();
    rewritten.write_all(b"other").unwrap();
    rewritten
        .set_modified(metadata.modified().unwrap())
        .unwrap();
    drop(rewritten);

    let after = fs::metadata(&file).unwrap();
    assert_eq!(after.len(), metadata.len());
    assert_eq!(after.modified().unwrap(), metadata.modified().unwrap());
    assert!(before == FilesRead::of(&source(&dir), None).unwrap());
}

/// A link back up the tree is walked until the entries bound: such a
/// tree is not read.
#[cfg(unix)]
#[test]
fn a_link_loop_below_the_directory_is_not_read() {
    use std::os::unix::fs::symlink;

    let dir = tokenizer_dir();
    for n in 0..9 {
        fs::write(dir.path().join(format!("file-{n}")), "x").unwrap();
    }
    symlink(dir.path(), dir.path().join("loop")).unwrap();

    assert!(FilesRead::of(&source(&dir), None).is_none());
}

/// A source or a chat template that is not there is not read; a link to
/// nothing below the directory is passed by.
#[cfg(unix)]
#[test]
fn what_is_not_there_is_not_read() {
    use std::os::unix::fs::symlink;

    let dir = tokenizer_dir();
    let before = FilesRead::of(&source(&dir), None).unwrap();

    let gone = dir.path().join("gone");
    assert!(FilesRead::of(gone.to_str().unwrap(), None).is_none());
    assert!(FilesRead::of(&source(&dir), gone.to_str()).is_none());

    symlink(&gone, dir.path().join("link")).unwrap();
    assert!(before == FilesRead::of(&source(&dir), None).unwrap());
}
