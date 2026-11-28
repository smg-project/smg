//! A file read whole or not at all: what a file's identity reads, the
//! read's bound, and a file that moves, leaves or is replaced around a
//! read. Also an error of the opened file's own metadata.

use super::*;

/// The read of a file that left its path and came back to it, and that
/// stood as `before` when it left. A rename stamps the file's change
/// time, and the stamp has the file system's grain. Where it shows, the
/// file moved and takes the two looks of [`settle`]. Where it does not,
/// the watch finds the file as it last saw it, and the next look reads
/// it.
fn returned(file: &mut SpecFile, before: &Seen) -> Result<ApplyOutcome, SpecRejected> {
    if look(&file.path) != *before {
        return settle(file);
    }
    match file.poll() {
        Look::Read(outcome) => outcome,
        other => panic!("the file stood as last seen and the look was {other:?}"),
    }
}

/// The identity reads the file's change time: an edit in place that
/// keeps the file's size and puts its modified time back is a new
/// identity. The change time has the file system's grain, so the edit
/// is made again until it shows, a bounded number of times.
#[cfg(unix)]
#[test]
fn an_edit_that_keeps_the_size_and_the_modified_time_is_a_new_identity() {
    use std::io::Write as _;

    let bed = bed(LEAST_LOAD.as_bytes());
    let before = identity(&bed.path).expect("the file");
    let modified = before.modified.expect("a modified time");
    let edited = LEAST_LOAD.replace("best", "tseb");
    let mut after = before.clone();
    for _ in 0..2_000 {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .open(&bed.path)
            .expect("the file");
        file.write_all(edited.as_bytes()).expect("the edit");
        file.set_modified(modified)
            .expect("the modified time put back");
        drop(file);
        after = identity(&bed.path).expect("the file");
        if after != before {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }
    assert_ne!(after, before, "the edit never showed");
    assert_eq!(
        (after.len, after.modified, after.inode.number),
        (before.len, before.modified, before.inode.number),
        "the change time alone moved"
    );
    assert_ne!(after.inode.changed, before.inode.changed);
}

/// A file that grew since its look is not read through a descriptor that
/// no longer has the looked identity.
#[test]
fn a_file_that_grew_since_its_look_is_not_read() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let looked = identity(&bed.path).expect("the file");
    let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    fs::write(&bed.path, vec![b'#'; bound + 5]).expect("the file grows");
    let unexpected_reads = AtomicUsize::new(0);
    assert_eq!(read_counting_unexpected(&looked, &unexpected_reads), None);
    assert_eq!(unexpected_reads.load(Ordering::Relaxed), 0);
}

/// A read stops one byte past the bound, whatever the file holds: the
/// byte over tells the door a document over the bound.
#[test]
fn a_read_stops_one_byte_past_the_bound() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    fs::write(&bed.path, vec![b'#'; bound + 5]).expect("the file grows");
    let file = fs::File::open(&bed.path).expect("the file");
    let mut bytes = Vec::new();
    read_to_bound(&file, &mut bytes).expect("the read");
    assert_eq!(bytes.len(), bound + 1);
}

/// The identity reads where the path leads: a link turned to another
/// name of the same file (one inode, one size, the same times) is a new
/// identity.
#[cfg(unix)]
#[test]
fn a_link_turned_to_another_name_of_the_file_is_a_new_identity() {
    use std::os::unix::fs::symlink;

    let bed = bed(LEAST_LOAD.as_bytes());
    let (other, link) = (
        bed.dir.path().join("other.yaml"),
        bed.dir.path().join("link.yaml"),
    );
    fs::hard_link(&bed.path, &other).expect("another name of the file");
    symlink(&bed.path, &link).expect("the link");
    let before = identity(&link).expect("the file");
    fs::remove_file(&link).expect("the link goes");
    symlink(&other, &link).expect("the link, turned");
    let after = identity(&link).expect("the file");
    assert_ne!(after, before);
    assert_eq!(
        (after.len, after.modified, after.inode),
        (before.len, before.modified, before.inode),
        "where the path leads alone moved"
    );
}

/// What a look made before a mount's rotation saw: the file, in a
/// version's directory the rotation has removed since.
fn before_the_rotation(dir: &Path, path: &Path) -> Seen {
    match look(path) {
        Seen::File(identity) => Seen::File(Identity {
            resolved: dir.join("..version_1/policy.yaml"),
            ..identity
        }),
        unreadable @ Seen::Unreadable(_) => unreadable,
    }
}

/// A rotation takes the version a look resolved from under the open:
/// the read fails and the file moved, so the start reads it again where
/// the path leads next. The same for a look that could not read the
/// path between two versions. An error is the file's own where the file
/// stood still: one read between two looks that agree, and the start is
/// refused by the error's text, not as a file that moved. An open that
/// says the looked file disappeared contradicts that look and is retried.
#[test]
fn a_file_gone_at_its_read_is_read_where_the_path_leads_next() {
    let rotated = bed(LEAST_LOAD.as_bytes());
    let mut looks = 0;
    let file = SpecFile::start_looking(
        rotated.path.clone(),
        Arc::clone(&rotated.generations),
        |path| {
            looks += 1;
            if looks == 1 {
                before_the_rotation(rotated.dir.path(), path)
            } else {
                look(path)
            }
        },
    );
    assert!(file.is_ok(), "the second read stood still");
    assert_eq!(looks, 3, "the missing open and the read that stood");
    assert_eq!(rotated.answered().2, LEAST_LOAD.as_bytes());

    let between = bed(RANDOM.as_bytes());
    let mut looks = 0;
    let file = SpecFile::start_looking(
        between.path.clone(),
        Arc::clone(&between.generations),
        |path| {
            looks += 1;
            if looks == 1 {
                Seen::Unreadable("between two versions".to_owned())
            } else {
                look(path)
            }
        },
    );
    assert!(file.is_ok(), "the read stood still");
    assert_eq!(looks, 4, "the look that failed, and a read between two");
    assert_eq!(between.answered().2, RANDOM.as_bytes());

    let still = bed(LEAST_LOAD.as_bytes());
    let mut looks = 0;
    let refused =
        SpecFile::start_looking(still.path.clone(), Arc::clone(&still.generations), |path| {
            looks += 1;
            before_the_rotation(still.dir.path(), path)
        });
    assert_eq!(
        refused.err(),
        Some(SpecRejected::Unreadable(
            "the file moved under each of 3 reads".to_owned()
        ))
    );
    assert_eq!(looks, 3, "each missing open is retried");
    assert!(!still.generations.bound());
}

/// Under the watch a read that fails while the file moves is no state
/// of the file: nothing is rejected and nothing counted, and the file
/// is read once it stood still. A missing open also contradicts looks
/// that claim the file stands there.
#[test]
fn a_read_that_fails_under_a_moving_file_is_no_rejected_state() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let mut file = bed.start();
    replace(&bed.path, RANDOM.as_bytes());
    // An injected look names a version already gone at the open. The
    // missing open contradicts the look, so no second look is made.
    let dir = bed.dir.path();
    let gone = |path: &Path| before_the_rotation(dir, path);
    assert_eq!(file.poll_looking(gone), Look::Settling);
    let mut looks = 0;
    let failed = file.poll_looking(|path| {
        looks += 1;
        gone(path)
    });
    assert_eq!(looks, 1, "the missing open ends this read");
    assert_eq!(failed, Look::Settling, "the missing open is retried");
    assert_eq!((bed.rejected(), bed.generation()), (0, 1));
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));

    // The same impossible lost version at every look: its missing open
    // still contradicts the injected look.
    assert_eq!(file.poll_looking(gone), Look::Settling);
    assert_eq!(file.poll_looking(gone), Look::Settling);
    assert_eq!((bed.rejected(), bed.generation()), (0, 2));
}

/// A missing open between two looks of the same standing file is no
/// state of that file: the start retries it after the name comes back.
#[test]
fn a_missing_open_between_matching_start_looks_is_retried() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let standing = look(&bed.path);
    let saved = bed.dir.path().join("standing.yaml");
    let mut looks = 0;

    let started = SpecFile::start_looking(bed.path.clone(), Arc::clone(&bed.generations), |path| {
        looks += 1;
        match looks {
            1 => {
                fs::rename(path, &saved).expect("the file leaves before the open");
                standing.clone()
            }
            2 => {
                fs::rename(&saved, path).expect("the file returns before the next look");
                standing.clone()
            }
            _ => look(path),
        }
    });

    assert!(started.is_ok(), "the missing open is retried");
    assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
}

/// A missing open between two looks of the same settled file is no state
/// of that file: the watch retries it without rejecting the standing file.
#[test]
fn a_missing_open_between_matching_watch_looks_is_retried() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let mut file = bed.start();
    replace(&bed.path, RANDOM.as_bytes());
    assert_eq!(file.poll(), Look::Settling);
    let standing = look(&bed.path);
    let saved = bed.dir.path().join("standing.yaml");
    let mut looks = 0;

    let missing = file.poll_looking(|path| {
        looks += 1;
        match looks {
            1 => {
                fs::rename(path, &saved).expect("the file leaves before the open");
                standing.clone()
            }
            2 => {
                fs::rename(&saved, path).expect("the file returns before the next look");
                standing.clone()
            }
            _ => look(path),
        }
    });

    assert_eq!(missing, Look::Settling, "the missing open is retried");
    assert_eq!((bed.rejected(), bed.generation()), (0, 1));
    if saved.exists() {
        fs::rename(&saved, &bed.path).expect("the file returns after the missing open");
    }
    assert_eq!(
        returned(&mut file, &standing),
        Ok(ApplyOutcome::Generation(2))
    );
    assert_eq!(bed.answered().2, RANDOM.as_bytes());
}

/// A file is read whole or not at all. At the start one that moved
/// under its read is read again, and one that moves under every read
/// refuses the start; under the watch the bytes read while it moved are
/// not applied, and the file is read once it stood still.
#[test]
fn a_file_that_moves_under_its_read_is_read_again() {
    let started = bed(LEAST_LOAD.as_bytes());
    let mut looks = 0;
    let file = SpecFile::start_looking(
        started.path.clone(),
        Arc::clone(&started.generations),
        |path| {
            looks += 1;
            if looks == 2 {
                // After the first read: a writer replaces the document.
                fs::write(path, RANDOM).expect("the file");
            }
            look(path)
        },
    );
    assert!(file.is_ok(), "the second read stood still");
    assert_eq!(looks, 4, "two reads, each between two looks");
    assert_eq!(started.answered().2, RANDOM.as_bytes());

    let restless = bed(LEAST_LOAD.as_bytes());
    let mut looks = 0;
    let refused = SpecFile::start_looking(
        restless.path.clone(),
        Arc::clone(&restless.generations),
        |path| {
            looks += 1;
            if looks % 2 == 0 {
                let document = if looks % 4 == 0 { LEAST_LOAD } else { RANDOM };
                fs::write(path, document).expect("the file");
            }
            look(path)
        },
    );
    let moved = SpecRejected::Unreadable("the file moved under each of 3 reads".to_owned());
    assert_eq!(refused.err(), Some(moved));
    assert!(!restless.generations.bound());

    let watched = bed(LEAST_LOAD.as_bytes());
    let mut file = watched.start();
    replace(&watched.path, RANDOM.as_bytes());
    assert_eq!(file.poll(), Look::Settling);
    let mut looks = 0;
    let moving = file.poll_looking(|path| {
        looks += 1;
        if looks == 2 {
            // After the read: the document is replaced again.
            fs::write(path, with_line(LEAST_LOAD, 9)).expect("the file");
        }
        look(path)
    });
    assert_eq!(moving, Look::Settling, "the bytes read while it moved");
    assert_eq!(watched.generation(), 1);
    assert_eq!(file.poll(), Look::Read(Ok(ApplyOutcome::Generation(9))));
}

/// A start does not accept bytes from a replacement that appears only
/// between two looks of the standing file.
#[test]
fn an_aba_replacement_during_start_is_not_accepted() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let standing = look(&bed.path);
    let saved = bed.dir.path().join("standing.yaml");
    fs::rename(&bed.path, &saved).expect("the standing file is saved");
    fs::write(&bed.path, RANDOM).expect("the transient replacement");
    let mut looks = 0;
    let unexpected_reads = AtomicUsize::new(0);

    let started = SpecFile::start_reading(
        bed.path.clone(),
        Arc::clone(&bed.generations),
        |path| {
            looks += 1;
            match looks {
                1 => standing.clone(),
                2 => {
                    fs::rename(&saved, path).expect("the standing file is restored");
                    standing.clone()
                }
                _ => look(path),
            }
        },
        |identity| read_counting_unexpected(identity, &unexpected_reads),
    );

    assert!(started.is_ok(), "the standing file starts after the retry");
    // The look whose read met the replacement, and the two around the
    // read of the restored file. Where the stamp of the renames shows,
    // that read is made once more.
    assert!(looks >= 3, "the transient read is retried");
    assert_eq!(unexpected_reads.load(Ordering::Relaxed), 0);
    assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
}

/// A watch does not accept bytes from a replacement that appears only
/// between two looks of the settled file.
#[test]
fn an_aba_replacement_during_watch_is_not_accepted() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let mut file = bed.start();
    replace(&bed.path, RANDOM.as_bytes());
    assert_eq!(file.poll(), Look::Settling);

    let settled = look(&bed.path);
    let saved = bed.dir.path().join("settled.yaml");
    fs::rename(&bed.path, &saved).expect("the settled file is saved");
    fs::write(&bed.path, with_line(LEAST_LOAD, 3)).expect("the transient replacement");
    let mut looks = 0;
    let unexpected_reads = AtomicUsize::new(0);
    let read = file.poll_reading(
        |path| {
            looks += 1;
            match looks {
                1 => settled.clone(),
                2 => {
                    fs::rename(&saved, path).expect("the settled file is restored");
                    settled.clone()
                }
                _ => look(path),
            }
        },
        |identity| read_counting_unexpected(identity, &unexpected_reads),
    );

    assert_eq!(read, Look::Settling, "the transient read is retried");
    assert_eq!(unexpected_reads.load(Ordering::Relaxed), 0);
    assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
    if saved.exists() {
        fs::rename(&saved, &bed.path).expect("the settled file is restored");
    }
    assert_eq!(
        returned(&mut file, &settled),
        Ok(ApplyOutcome::Generation(2))
    );
    assert_eq!(bed.answered().2, RANDOM.as_bytes());
}

/// An error reading the opened descriptor's metadata is the standing
/// file's error at the start, not evidence that the file moved.
#[test]
fn a_descriptor_metadata_error_during_start_is_reported() {
    const ERROR: &str = "descriptor metadata failed";

    let bed = bed(LEAST_LOAD.as_bytes());
    let started = SpecFile::start_reading(
        bed.path.clone(),
        Arc::clone(&bed.generations),
        look,
        |identity| read_with(identity, |_| Err(io::Error::other(ERROR)), read_to_bound),
    );

    assert_eq!(
        started.err(),
        Some(SpecRejected::Unreadable(ERROR.to_owned()))
    );
    assert!(!bed.generations.bound());
}

/// An error reading descriptor metadata after its bytes is the standing
/// file's error under the watch, not evidence that the file moved.
#[test]
fn a_descriptor_metadata_error_after_a_watch_read_is_reported() {
    const ERROR: &str = "descriptor metadata failed after the read";

    let bed = bed(LEAST_LOAD.as_bytes());
    let mut file = bed.start();
    replace(&bed.path, RANDOM.as_bytes());
    assert_eq!(file.poll(), Look::Settling);
    let metadata_calls = Cell::new(0);

    let failed = file.poll_reading(look, |identity| {
        read_with(
            identity,
            |file| {
                metadata_calls.set(metadata_calls.get() + 1);
                if metadata_calls.get() == 2 {
                    Err(io::Error::other(ERROR))
                } else {
                    file.metadata()
                }
            },
            read_to_bound,
        )
    });

    assert_eq!(
        failed,
        Look::Read(Err(SpecRejected::Unreadable(ERROR.to_owned())))
    );
    assert_eq!((bed.rejected(), bed.generation()), (1, 1));
}
