//! The watch's poll, look by look: when a changed file is read, a state
//! that left and a swapped link. Also the rejected states and how often
//! each is told, and a file that does not rest.

use super::*;

/// A changed file is read once it stood still for one poll, never at the
/// look that first sees it moved; a file that moves again before it
/// stood still waits one poll more, so an edit in place is not read half
/// written.
#[test]
fn a_changed_file_is_read_once_it_stood_still_for_one_poll() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let mut file = bed.start();
    assert_eq!(file.poll(), Look::Same);

    replace(&bed.path, RANDOM.as_bytes());
    assert_eq!(file.poll(), Look::Settling);
    assert_eq!(bed.generation(), 1, "not read at the look that saw it move");
    assert_eq!(file.poll(), Look::Read(Ok(ApplyOutcome::Generation(2))));
    assert_eq!(bed.answered().2, RANDOM.as_bytes());
    assert_eq!(file.poll(), Look::Same);

    // An edit in place, in two writes with a look between them.
    fs::write(&bed.path, with_line(LEAST_LOAD, 9)).expect("the first write");
    assert_eq!(file.poll(), Look::Settling);
    fs::write(&bed.path, with_line(RANDOM, 11)).expect("the second write");
    assert_eq!(file.poll(), Look::Settling, "it moved since the last look");
    assert_eq!(bed.generation(), 2);
    assert_eq!(file.poll(), Look::Read(Ok(ApplyOutcome::Generation(11))));
    assert_eq!(file.poll(), Look::Same);
}

/// A state that left is forgotten: a file seen moved at one look and as
/// it was last read at the next has not stood still, and when the same
/// new state comes back it waits its poll again.
#[cfg(unix)]
#[test]
fn a_state_that_left_is_forgotten() {
    use std::os::unix::fs::symlink;

    let bed = bed(b"");
    let mount = bed.dir.path().join("mount");
    for (version, document) in [("..v1", LEAST_LOAD), ("..v2", RANDOM)] {
        fs::create_dir_all(mount.join(version)).expect("the version's directory");
        fs::write(mount.join(version).join("policy.yaml"), document).expect("the file");
    }
    let turn = |version: &str| {
        let staged = mount.join("..data_tmp");
        symlink(version, &staged).expect("the staged link");
        fs::rename(&staged, mount.join("..data")).expect("the swap");
    };
    turn("..v1");
    symlink("..data/policy.yaml", mount.join("policy.yaml")).expect("the file's link");
    let mut file = SpecFile::start(mount.join("policy.yaml"), Arc::clone(&bed.generations))
        .expect("a valid document");

    turn("..v2");
    assert_eq!(file.poll(), Look::Settling);
    turn("..v1");
    assert_eq!(file.poll(), Look::Same, "as it was last read");
    turn("..v2");
    assert_eq!(file.poll(), Look::Settling, "it waits its poll again");
    assert_eq!(bed.generation(), 1);
    assert_eq!(file.poll(), Look::Read(Ok(ApplyOutcome::Generation(2))));
}

/// A mounted ConfigMap's update: the file is a link through `..data`,
/// and an update swaps `..data` to a new directory in one rename. The
/// path then leads elsewhere, which is a new identity whatever the
/// file's size and times are.
#[cfg(unix)]
#[test]
fn a_swapped_link_is_the_next_generation() {
    use std::os::unix::fs::symlink;

    let bed = bed(b"");
    let mount = bed.dir.path().join("mount");
    fs::create_dir(&mount).expect("the mount");
    let publish = |version: &str, document: &str| {
        fs::create_dir(mount.join(version)).expect("the version's directory");
        fs::write(mount.join(version).join("policy.yaml"), document).expect("the file");
        symlink(version, mount.join("..data_tmp")).expect("the staged link");
        fs::rename(mount.join("..data_tmp"), mount.join("..data")).expect("the swap");
    };
    publish("..version_1", LEAST_LOAD);
    symlink("..data/policy.yaml", mount.join("policy.yaml")).expect("the file's link");

    let mut file = SpecFile::start(mount.join("policy.yaml"), Arc::clone(&bed.generations))
        .expect("a valid document");
    assert_eq!(bed.generation(), 1);
    assert_eq!(file.poll(), Look::Same);

    publish("..version_2", RANDOM);
    fs::remove_dir_all(mount.join("..version_1")).expect("the old version goes");
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
    assert_eq!(bed.answered().2, RANDOM.as_bytes());
    assert_eq!(file.poll(), Look::Same);
}

/// The events are bounded by the file's states: a document that does
/// not run is rejected once, at the look that reads it, and every look
/// after it finds the file as it was last read. The running generation
/// and its document stay, and the next valid file is the next
/// generation.
#[test]
fn a_rejected_state_is_one_event_however_long_it_stands() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let mut file = bed.start();

    replace(
        &bed.path,
        LEAST_LOAD.replace("[healthy]", "[nowhere]").as_bytes(),
    );
    assert!(matches!(
        settle(&mut file),
        Err(SpecRejected::Invalid(SpecError::UnknownFilter { .. }))
    ));
    for _ in 0..10 {
        assert_eq!(file.poll(), Look::Same);
    }
    assert_eq!((bed.rejected(), bed.generation()), (1, 1));
    assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());

    replace(&bed.path, RANDOM.as_bytes());
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
    assert_eq!(bed.rejected(), 1);
}

/// A file that is gone, and a file over the bound, are rejected states
/// like a document that does not run: one event each, the running
/// generation kept. The file over the bound is refused by its size,
/// unread.
#[test]
fn a_file_that_is_gone_or_over_the_bound_is_a_rejected_state() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let mut file = bed.start();

    fs::remove_file(&bed.path).expect("the file goes");
    assert!(matches!(
        settle(&mut file),
        Err(SpecRejected::Unreadable(_))
    ));
    assert_eq!(file.poll(), Look::Same);
    assert_eq!((bed.rejected(), bed.generation()), (1, 1));

    let over = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits") + 5;
    fs::write(&bed.path, vec![b'#'; over]).expect("the file");
    assert_eq!(
        settle(&mut file),
        Err(SpecRejected::TooLarge(MAX_DOCUMENT_BYTES + 5))
    );
    assert_eq!(file.poll(), Look::Same);
    assert_eq!((bed.rejected(), bed.generation()), (2, 1));

    replace(&bed.path, RANDOM.as_bytes());
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
    assert_eq!(bed.rejected(), 2);
}

/// What the door refuses of a start it refuses of a running controller,
/// as a rejected state: the standing document with a `kind` line added
/// (its digest is the standing one's, and the validator refuses it), and
/// a file its aliases read past the bound. One event and one count
/// each, and the running generation stays.
#[test]
fn a_file_the_door_refuses_is_a_rejected_state_though_its_digest_stands() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let mut file = bed.start();

    replace(
        &bed.path,
        format!("kind: PolicySpec\n{LEAST_LOAD}").as_bytes(),
    );
    assert_eq!(
        settle(&mut file),
        Err(SpecRejected::Invalid(SpecError::Kind(
            "PolicySpec".to_owned()
        )))
    );
    assert_eq!((bed.rejected(), bed.generation()), (1, 1));

    replace(&bed.path, long_text_under_aliases(8_192, 2_048).as_bytes());
    match settle(&mut file) {
        Err(SpecRejected::Unparsed { format: "yaml", .. }) => {}
        other => panic!("the look read it as {other:?}"),
    }
    assert_eq!(file.poll(), Look::Same);
    assert_eq!((bed.rejected(), bed.generation()), (2, 1));
    assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
}

/// A file whose document faults its own pass is a rejected state like
/// another: one event and one count, the running generation kept, and
/// the watch looks on.
#[test]
fn a_file_whose_document_faults_is_a_rejected_state() {
    let bed = bed_validating(LEAST_LOAD.as_bytes(), faulting);
    let mut file = bed.start();

    replace(&bed.path, RANDOM.as_bytes());
    assert_eq!(
        settle(&mut file),
        Err(SpecRejected::Faulted("a fault of the validator".to_owned()))
    );
    assert_eq!(file.poll(), Look::Same);
    assert_eq!((bed.rejected(), bed.generation()), (1, 1));

    replace(&bed.path, &with_line(LEAST_LOAD, 3));
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(3)));
    assert_eq!(bed.rejected(), 1);
}

/// A file that does not stand still is never read, and it is told once:
/// at the fifth look in a row that finds it moved, as a rejected state
/// is, and at no look after it. Once it rests it is read, and a file
/// that goes restless again is told again.
#[test]
fn a_file_that_never_rests_is_told_once() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let mut file = bed.start();
    // A writer that grows the file before every look.
    let mut writes = 0;
    let mut restless = |file: &mut SpecFile| {
        file.poll_looking(|path| {
            writes += 1;
            let grown = format!("{RANDOM}#{}\n", "x".repeat(writes));
            fs::write(path, grown).expect("the file");
            look(path)
        })
    };

    for _ in 1..RESTLESS_LOOKS {
        assert_eq!(restless(&mut file), Look::Settling);
    }
    assert_eq!(restless(&mut file), Look::Restless);
    for _ in 0..100 {
        assert_eq!(restless(&mut file), Look::Settling);
    }
    assert_eq!((bed.rejected(), bed.generation()), (1, 1));

    // It rests: the next look finds it standing and reads it.
    assert_eq!(file.poll(), Look::Read(Ok(ApplyOutcome::Generation(2))));
    assert_eq!(file.poll(), Look::Same);
    for _ in 1..RESTLESS_LOOKS {
        assert_eq!(restless(&mut file), Look::Settling);
    }
    assert_eq!(restless(&mut file), Look::Restless);
    assert_eq!((bed.rejected(), bed.generation()), (2, 2));
}

/// A look that finds the file as it was last read ends a run of moving
/// looks: the count starts again from the next look that finds it
/// moved, and a file is told at the fifth such look in a row, by the
/// number.
#[test]
fn a_look_that_finds_the_file_as_last_read_ends_the_restless_count() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let mut file = bed.start();
    // A look that finds another state each time, no file behind it.
    let mut moves = 0;
    let mut moved = |file: &mut SpecFile| {
        file.poll_looking(|_| {
            moves += 1;
            Seen::Unreadable(format!("moved {moves}"))
        })
    };

    for _ in 0..4 {
        assert_eq!(moved(&mut file), Look::Settling);
    }
    assert_eq!(file.poll(), Look::Same);
    for _ in 0..4 {
        assert_eq!(moved(&mut file), Look::Settling, "the run began again");
    }
    assert_eq!(moved(&mut file), Look::Restless, "the fifth look in a row");
    assert_eq!((bed.rejected(), bed.generation()), (1, 1));
}

/// A read ends a run of moving looks as a look at the file last read
/// does: the look after the read that finds the file moved is the first
/// of a new run, and the fifth of that run tells it.
#[test]
fn a_read_ends_the_restless_count() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let mut file = bed.start();
    // A writer that grows the file before every look.
    let mut writes = 0;
    let mut restless = |file: &mut SpecFile| {
        file.poll_looking(|path| {
            writes += 1;
            let grown = format!("{RANDOM}#{}\n", "x".repeat(writes));
            fs::write(path, grown).expect("the file");
            look(path)
        })
    };

    for _ in 1..RESTLESS_LOOKS {
        assert_eq!(restless(&mut file), Look::Settling);
    }
    // It rests one look short of being told, and is read.
    assert_eq!(file.poll(), Look::Read(Ok(ApplyOutcome::Generation(2))));
    for _ in 1..RESTLESS_LOOKS {
        assert_eq!(restless(&mut file), Look::Settling, "the run began again");
    }
    assert_eq!(
        restless(&mut file),
        Look::Restless,
        "the fifth look in a row"
    );
    assert_eq!((bed.rejected(), bed.generation()), (1, 2));
}
