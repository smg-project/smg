//! The cases of a door with a spec store bound over a directory: an
//! enforce and its guard, a start, a bind, a generation persisted or
//! refused, and a rollback.

use std::{collections::BTreeSet, sync::mpsc};

use super::*;
use crate::document::operator::spec_store::{
    tests::{name_of, names},
    RETAINED,
};

const UNKNOWN_FILTER: &str =
    "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [nowhere]\n    pick: random\n";

/// What a rollback to generation `wanted`, one on file, ran.
fn roll_back(generations: &DocumentManager, wanted: u64) -> Result<(u64, Digest), SpecRejected> {
    generations.roll_back(wanted).expect("on file").ran
}

/// [`bind`] for a second door on a directory another holds, as a file
/// system whose locks do not hold between two hosts lets one in.
fn bind_beside(generations: &DocumentManager, dir: &Path) {
    let opened = SpecStore::open_beside(dir.to_owned()).expect("the directory opens");
    generations
        .bind_store(opened, || false)
        .expect("the store binds");
}

/// The guard: an enforce names the digest its caller read as standing,
/// and is refused where another apply landed first.
#[test]
fn an_enforce_is_refused_where_another_apply_landed_first() {
    let (dir, store, generations) = stored();
    let (_, first) = enforce(&generations, LEAST_LOAD).expect("it runs");
    let read = first.to_string();
    let second = generations
        .enforce(&read, RANDOM.into(), Format::Yaml)
        .expect("the digest stands");
    let (number, digest) = second.ran.expect("it runs");
    assert_eq!(number, 2);

    // The same guard again: the digest it names stands no longer.
    let late = generations.enforce(&read, bound_to(1).into(), Format::Yaml);
    assert!(
        matches!(late, Err(EnforceRefused::Guarded(2, Some(stands))) if stands == digest),
        "{late:?}"
    );
    assert_eq!(standing(&store), (2, Some(digest)));
    assert_eq!(names(dir.path()).len(), 2);

    // Before any generation no digest stands, so a guard names none.
    let (_dir, _store, empty) = stored();
    let guarded = empty.enforce(&read, LEAST_LOAD.into(), Format::Yaml);
    assert!(
        matches!(guarded, Err(EnforceRefused::Guarded(0, None))),
        "{guarded:?}"
    );
}

/// An apply call writes the next number: a document's own line is a
/// claim of that number, and any other is refused by naming the
/// standing one. A dry run answers the same.
#[test]
fn a_line_is_a_claim_of_the_next_number_under_a_store() {
    let (dir, store, generations) = stored();
    let ran = |document| enforce(&generations, document).map(|(number, _)| number);
    assert_eq!(ran(with_line(LEAST_LOAD, 1)), Ok(1));
    assert_eq!(ran(with_line(RANDOM, 2)), Ok(2));
    let stands = standing(&store).1.expect("a digest");

    for line in [1, 4, 9] {
        let refused = enforce(&generations, with_line(LEAST_LOAD, line)).expect_err("a line");
        let another = SpecRejected::NotTheNext {
            offered: line,
            standing: 2,
        };
        assert_eq!(refused, another);
        assert_eq!(refused.field_path(), "generation");
        assert_eq!(
            refused.to_string(),
            format!("generation {line}: an apply writes generation 3, the next after the standing generation 2; write that number, or none")
        );
    }
    // A changed document under the standing line, as under a file.
    let changed = enforce(&generations, with_line(LEAST_LOAD, 2)).expect_err("its line");
    assert!(
        matches!(changed, SpecRejected::ChangedUnderItsLine { generation: 2, standing, .. } if standing == stands),
        "{changed:?}"
    );
    // Nothing of the four is on file, and the running generation stays.
    assert_eq!(names(dir.path()).len(), 2);
    assert_eq!(standing(&store), (2, Some(stands)));
    assert_eq!(generations.apply_counts(), applied(2, 4));

    let dry = |line| {
        generations
            .dry_run("", &with_line(LEAST_LOAD, line), Format::Yaml)
            .expect("no guard")
            .would_run
            .map(|(number, _)| number)
    };
    let another = SpecRejected::NotTheNext {
        offered: 9,
        standing: 2,
    };
    assert_eq!((dry(9), dry(3)), (Err(another), Ok(3)));
}

#[test]
fn a_document_that_does_not_run_is_refused_with_nothing_on_file() {
    let (dir, store, generations) = stored();
    let (_, stands) = enforce(&generations, LEAST_LOAD).expect("it runs");

    let refused = enforce(&generations, UNKNOWN_FILTER).expect_err("no such filter");
    assert!(
        matches!(
            refused,
            SpecRejected::Invalid(SpecError::UnknownFilter { .. })
        ),
        "{refused:?}"
    );
    let unread = enforce(&generations, "apiVersion: [").expect_err("no document");
    assert!(
        matches!(unread, SpecRejected::Unparsed { .. }),
        "{unread:?}"
    );

    assert_eq!(names(dir.path()), BTreeSet::from([name_of(1, "yaml")]));
    assert_eq!(standing(&store), (1, Some(stands)));
    assert_eq!(generations.apply_counts(), applied(1, 2));
}

/// A start: the newest generation on file runs under the number its
/// name states, not under a first number, and the next enforce writes
/// the one after it.
#[test]
fn a_start_runs_the_newest_generation_on_file_under_its_number() {
    let (dir, _store, before) = stored();
    for document in [LEAST_LOAD, RANDOM, LEAST_LOAD] {
        enforce(&before, document).expect("it runs");
    }
    let (_, stood) = before.standing();
    drop(before);

    let (store, after) = generations();
    bind(&after, dir.path());
    assert_eq!(standing(&store), (3, stood));
    let read = after.document(0).expect("the standing document");
    assert_eq!(
        (read.generation, read.document.as_slice()),
        (3, LEAST_LOAD.as_bytes())
    );
    assert_eq!(after.apply_counts(), applied(1, 0));
    assert_eq!(enforce(&after, RANDOM).map(|(number, _)| number), Ok(4));
}

/// A store that runs the newest file's document already, by another hand
/// than this door's and under a lower number, takes the file's number at
/// the bind: the next enforce writes the one after it, which no file
/// holds. Under the number the store runs, the file is the standing
/// document again and nothing is applied.
#[test]
fn a_start_takes_the_files_number_for_a_document_the_store_runs_under_a_lower_one() {
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::write(dir.path().join(name_of(2, "yaml")), LEAST_LOAD).expect("the file");
    let (store, first) = generations();
    assert_eq!(yaml(&first, LEAST_LOAD), Ok(ApplyOutcome::Generation(1)));
    let (_, runs) = standing(&store);

    let bound = DocumentManager::new(Arc::clone(&store), stock(), Vec::new());
    bind(&bound, dir.path());
    assert_eq!(standing(&store), (2, runs));
    assert_eq!(enforce(&bound, RANDOM).map(|(number, _)| number), Ok(3));

    let same = tempfile::tempdir().expect("a temp dir");
    std::fs::write(same.path().join(name_of(3, "yaml")), RANDOM).expect("the file");
    let again = DocumentManager::new(Arc::clone(&store), stock(), Vec::new());
    bind(&again, same.path());
    assert_eq!(standing(&store).0, 3);
    assert_eq!(again.apply_counts(), applied(0, 0));
}

/// A dry run reads the standing generation and its source's rule on one
/// side of a bind. While a bind's side is held, as a bind in flight holds
/// it, a dry run reads nothing and a bind applies nothing; once it is let
/// go, the line the dry run is asked is the next number on either side.
#[test]
fn a_dry_run_reads_on_one_side_of_a_bind() {
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::write(dir.path().join(name_of(4, "yaml")), LEAST_LOAD).expect("the file");
    let (store, generations) = generations();
    let opened = SpecStore::open(dir.path().to_owned()).expect("the directory opens");
    let next = with_line(RANDOM, 5);
    let (answer, answered) = mpsc::channel();
    thread::scope(|scope| {
        let side = generations.binding.lock();
        let (generations, next) = (&generations, &next);
        scope.spawn(move || {
            generations
                .bind_store(opened, || false)
                .expect("the store binds");
        });
        scope.spawn(move || {
            let _ = answer.send(generations.dry_run("", next, Format::Yaml));
        });
        let waited = answered.recv_timeout(Duration::from_millis(100));
        assert!(waited.is_err(), "the dry run waits");
        assert_eq!(standing(&store).0, 0, "the bind waits");
        drop(side);
    });
    let dry = answered.recv().expect("an answer").expect("no guard");
    assert_eq!(dry.would_run.map(|(number, _)| number), Ok(5));
}

/// A document on file runs under its file's number whatever line its
/// bytes write.
#[test]
fn a_generation_on_file_runs_under_its_names_number_whatever_its_line() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let lined = with_line(LEAST_LOAD, 2);
    std::fs::write(dir.path().join(name_of(7, "yaml")), lined).expect("the file");
    let (store, generations) = generations();
    bind(&generations, dir.path());
    assert_eq!(standing(&store).0, 7);
}

/// A process that ended between the rename and the store's entry: the
/// generation is on file, and the next start runs it.
#[test]
fn a_generation_persisted_and_never_entered_runs_at_the_next_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut on_file = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
    let first = Persisted {
        generation: 1,
        format: Format::Yaml,
    };
    on_file
        .persist(first, LEAST_LOAD.as_bytes())
        .expect("it persists");
    drop(on_file);

    let (store, generations) = generations();
    bind(&generations, dir.path());
    assert_eq!(standing(&store).0, 1);
}

#[test]
fn a_newest_generation_that_does_not_run_refuses_the_bind() {
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::write(dir.path().join(name_of(4, "yaml")), LEAST_LOAD).expect("the file");
    std::fs::write(dir.path().join(name_of(5, "yaml")), UNKNOWN_FILTER).expect("the file");
    let (store, generations) = generations();
    let opened = SpecStore::open(dir.path().to_owned()).expect("the directory opens");

    let refused = generations
        .bind_store(opened, || false)
        .expect_err("it does not run");
    assert!(
        matches!(
            &refused,
            StoreRefused::Rejected {
                generation: 5,
                source: SpecRejected::Invalid(_)
            }
        ),
        "{refused:?}"
    );
    assert!(
        refused.to_string().starts_with("generation 5: "),
        "{refused}"
    );
    // Nothing runs and no source is bound: the generation under the
    // newest is not taken in its place.
    assert_eq!(standing(&store), (0, None));
    assert!(!generations.writable() && !generations.bound());
}

/// A spec that writes a local tag in front of a key.
const TAGGED_KEY: &str = "apiVersion: smg.policy/v1\nprofiles:\n  default: {!mine pick: random}\n";
/// The same spec with the tag taken out.
const NO_TAG: &str = "apiVersion: smg.policy/v1\nprofiles:\n  default: {pick: random}\n";

/// What a start over the directory at `dir` runs: the generation of a new
/// door bound there, or the bind's refusal as it prints.
fn start(dir: &Path) -> Result<u64, String> {
    let (store, generations) = generations();
    let opened = SpecStore::open(dir.to_owned()).expect("the directory opens");
    match generations.bind_store(opened, || false) {
        Ok(()) => Ok(standing(&store).0),
        Err(refused) => Err(refused.to_string()),
    }
}

/// A store whose newest file writes a local tag in front of a key does
/// not start. A start reads that file through the form's read, which
/// refuses such a key by the tag's name, and the bind names the
/// generation. A build that read the key as if its tag were not written
/// applied such a spec and kept its bytes, tag and all: a store keeps a
/// document as it was applied. The file here is written by the store's
/// own writer, as that build's apply wrote it.
#[test]
fn a_newest_generation_with_a_local_tag_on_a_key_refuses_the_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut on_file = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
    let first = Persisted {
        generation: 1,
        format: Format::Yaml,
    };
    on_file
        .persist(first, TAGGED_KEY.as_bytes())
        .expect("it persists");
    drop(on_file);

    let refused = "generation 1: not a policy spec in yaml: profiles.default: the tag `!mine` is none JSON writes: take the tag out at line 3 column 12";
    assert_eq!(start(dir.path()), Err(refused.to_owned()));
}

/// The way out of that refusal is taken on the build that reads the tagged
/// file, ahead of the newer build's start: the same spec with the tag
/// taken out, under the next generation's number. The two texts are one
/// spec to that build. So with no number, or under the standing one, its
/// apply changes nothing and writes no file: the newest file keeps its
/// tag, and the newer build does not start. Under the next number its
/// apply writes a file with no tag in it, and the newer build starts on
/// that one.
///
/// That build's door is stood in for by this build's, bound while the
/// first file holds the text with no tag. A door reads a file once, at its
/// bind: behind the bind the file takes the tagged text that build's apply
/// wrote. What an apply writes for a spec its door holds already is the
/// same on both builds.
#[test]
fn a_store_with_a_tagged_key_starts_once_the_next_generation_writes_none() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let first = dir.path().join(name_of(1, "yaml"));
    let before = || {
        std::fs::write(&first, NO_TAG).expect("the file");
        let (_, door) = generations();
        bind(&door, dir.path());
        std::fs::write(&first, TAGGED_KEY).expect("the file as that build wrote it");
        door
    };
    let ran = |door: &DocumentManager, document: Vec<u8>| {
        enforce(door, document).map(|(number, _)| number)
    };

    // With no number and under the standing one: the spec stands already.
    let door = before();
    for again in [NO_TAG.as_bytes().to_vec(), with_line(NO_TAG, 1)] {
        assert_eq!(ran(&door, again), Ok(1));
    }
    drop(door);
    assert_eq!(names(dir.path()), BTreeSet::from([name_of(1, "yaml")]));
    let refused = start(dir.path()).expect_err("the newest file keeps its tag");
    assert!(refused.starts_with("generation 1: "), "{refused}");
    assert!(refused.contains("the tag `!mine`"), "{refused}");

    // Under the next number: a new file, with no tag in it.
    let door = before();
    assert_eq!(ran(&door, with_line(NO_TAG, 2)), Ok(2));
    drop(door);
    let second = dir.path().join(name_of(2, "yaml"));
    assert_eq!(
        std::fs::read(&second).expect("the new file"),
        with_line(NO_TAG, 2)
    );
    // The newer build starts on it. The first file stays as it was, and
    // stops no start.
    assert_eq!(start(dir.path()), Ok(2));
    assert_eq!(
        std::fs::read(&first).expect("the first file"),
        TAGGED_KEY.as_bytes()
    );
}

/// A number on file that this door did not write, as another process on
/// the directory leaves one: the apply is refused by name, the file
/// stays as it was, and the running generation stays.
#[test]
fn a_generation_on_file_already_is_not_written_over() {
    let (dir, store, generations) = stored();
    enforce(&generations, LEAST_LOAD).expect("it runs");
    let theirs = dir.path().join(name_of(2, "json"));
    std::fs::write(&theirs, b"theirs").expect("the file");

    let refused = enforce(&generations, RANDOM).expect_err("the number is taken");
    let taken = "generation 2 is on file already".to_owned();
    assert_eq!(refused, SpecRejected::Unpersisted(taken));
    assert_eq!(standing(&store).0, 1);
    assert_eq!(std::fs::read(&theirs).expect("the file"), b"theirs");
    assert_eq!(generations.apply_counts(), applied(1, 1));
}

/// A directory that does not take the write: nothing is applied, the
/// running generation stays, and the refusal says what the directory
/// said.
#[test]
fn a_document_the_directory_does_not_take_is_not_applied() {
    let outer = tempfile::tempdir().expect("a temp dir");
    let dir = outer.path().join("store");
    std::fs::create_dir(&dir).expect("the directory");
    let (store, generations) = generations();
    bind(&generations, &dir);
    let (_, stands) = enforce(&generations, LEAST_LOAD).expect("it runs");
    std::fs::remove_dir_all(&dir).expect("the directory goes");

    let refused = enforce(&generations, RANDOM).expect_err("nowhere to persist");
    assert!(
        matches!(refused, SpecRejected::Unpersisted(_)),
        "{refused:?}"
    );
    assert!(
        refused.to_string().starts_with("not persisted: "),
        "{refused}"
    );
    assert_eq!(refused.field_path(), "");
    assert_eq!(standing(&store), (1, Some(stands)));
    assert_eq!(generations.apply_counts(), applied(1, 1));
}

/// The entry's refusal takes the persisted generation off file before
/// the door's lock goes: a start never reads a generation that did not
/// run. The case's validator enters the number itself, behind the
/// door's back, so the door's own call of the entry is refused.
#[test]
fn a_generation_the_entry_refuses_goes_off_file_again() {
    static BEHIND: OnceLock<Arc<MemoryStore>> = OnceLock::new();
    fn racing(
        document: &PolicySpec,
        digest: Digest,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<AppliedSpec, SpecError> {
        if let Some(store) = BEHIND.get().filter(|_| document.generation == 2) {
            let taken = AppliedSpec::validate_digested(document, digest, catalog, denied_headers)?;
            store.apply(taken).expect("the number is free");
        }
        AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
    }

    let dir = tempfile::tempdir().expect("a temp dir");
    let store = Arc::new(MemoryStore::new());
    let generations = DocumentManager::validating(Arc::clone(&store), stock(), Vec::new(), racing);
    bind(&generations, dir.path());
    enforce(&generations, LEAST_LOAD).expect("it runs");
    assert!(BEHIND.set(Arc::clone(&store)).is_ok(), "set once");

    let refused = enforce(&generations, RANDOM).expect_err("an equal number");
    let stale = StaleGeneration {
        standing: 2,
        offered: 2,
    };
    assert_eq!(refused, SpecRejected::Stale(stale));
    assert_eq!(names(dir.path()), BTreeSet::from([name_of(1, "yaml")]));
    assert_eq!(generations.apply_counts(), applied(1, 1));
}

/// A generation the entry refuses whose name does not come off file: the
/// write says both, the store finds the number no more, and the next
/// write takes the name off first. Another hand enters the number while
/// the write waits for the documents' lock, its file persisted, and lays
/// a directory under the name, which no removal of a file takes; the name
/// is a file again by the next write.
#[test]
fn a_refused_generation_that_stays_on_file_is_told_and_goes_at_the_next_write() {
    let (dir, store, generations) = stored();
    enforce(&generations, LEAST_LOAD).expect("it runs");
    let name = dir.path().join(name_of(2, "yaml"));
    let refused = thread::scope(|scope| {
        let documents = generations.held.lock();
        let write = scope.spawn(|| enforce(&generations, RANDOM));
        let began = Instant::now();
        while !name.exists() {
            assert!(
                began.elapsed() < Duration::from_secs(30),
                "the write persists"
            );
            thread::yield_now();
        }
        let behind = DocumentManager::new(Arc::clone(&store), stock(), Vec::new());
        let entered = yaml(&behind, with_line(&bound_to(0), 2));
        assert_eq!(entered, Ok(ApplyOutcome::Generation(2)));
        std::fs::remove_file(&name).expect("the persisted file");
        std::fs::create_dir(&name).expect("a directory in its place");
        drop(documents);
        write.join().expect("no panic")
    });
    let Err(SpecRejected::Unpersisted(told)) = refused else {
        panic!("{refused:?}");
    };
    let (stale, left) = told
        .split_once("; generation 2 was not taken off file again (")
        .expect("the entry's refusal, then the removal's");
    assert_eq!(stale, "generation 2 is not above the standing generation 2");
    assert!(
        left.ends_with(
            "): a start may run it, and beside the other form's name the two refuse a start"
        ),
        "{told}"
    );
    assert!(matches!(
        generations.roll_back(2),
        Err(EnforceRefused::NotHeld(_))
    ));

    std::fs::remove_dir(&name).expect("the directory goes");
    std::fs::write(&name, RANDOM).expect("a file again");
    assert_eq!(
        enforce(&generations, RANDOM).map(|(number, _)| number),
        Ok(3)
    );
    let on_file = BTreeSet::from([name_of(1, "yaml"), name_of(3, "yaml")]);
    assert_eq!(names(dir.path()), on_file);
    assert_eq!(generations.apply_counts(), applied(2, 1));
}

/// Two enforces that read one digest as standing: one lands, and the
/// others are told that another apply landed first.
#[test]
fn enforces_under_one_guard_land_once() {
    const WRITERS: u64 = 8;
    let (dir, store, generations) = stored();
    let (_, first) = enforce(&generations, LEAST_LOAD).expect("it runs");
    let read = first.to_string();
    let landed: Vec<bool> = thread::scope(|scope| {
        let writers: Vec<_> = (0..WRITERS)
            .map(|n| {
                let (generations, read) = (&generations, &read);
                scope.spawn(move || {
                    let document = bound_to(n).into_bytes();
                    match generations.enforce(read, document, Format::Yaml) {
                        // The guard admits one, and its document runs.
                        Ok(enforced) => {
                            enforced.ran.expect("the one the guard admits runs");
                            true
                        }
                        Err(EnforceRefused::Guarded(..)) => false,
                        Err(other) => panic!("{other:?}"),
                    }
                })
            })
            .collect();
        writers
            .into_iter()
            .map(|writer| writer.join().expect("no writer panics"))
            .collect()
    });
    assert_eq!(
        landed.iter().filter(|landed| **landed).count(),
        1,
        "{landed:?}"
    );
    assert_eq!(standing(&store).0, 2);
    assert_eq!(names(dir.path()).len(), 2);
}

/// The bound of what a store keeps on file: the last eight
/// generations, at every apply.
#[test]
fn a_store_keeps_its_last_generations_on_file() {
    let (dir, _store, generations) = stored();
    let last = u64::try_from(RETAINED).expect("a small count") + 2;
    for number in 1..=last {
        let ran = enforce(&generations, bound_to(number)).map(|(number, _)| number);
        assert_eq!(ran, Ok(number));
        assert!(names(dir.path()).len() <= RETAINED);
    }
    let on_file = (3..=last).map(|number| name_of(number, "yaml")).collect();
    assert_eq!(names(dir.path()), on_file);
    // The number the flag's help states.
    assert_eq!(names(dir.path()).len(), 8);
}

/// Two doors on one directory, as two processes hold one, each enforce
/// a document of their own at one moment: one runs it as generation 1,
/// and the other is refused by name with nothing applied. The directory
/// holds the document that runs, and a start on it runs that one.
#[test]
fn two_doors_on_one_directory_do_not_both_run_one_number() {
    for round in 0..20 {
        let dir = tempfile::tempdir().expect("a temp dir");
        // The second door is a writer the directory's lock did not keep
        // out.
        let binds = [bind as fn(&DocumentManager, &Path), bind_beside];
        let doors = [(LEAST_LOAD, binds[0]), (RANDOM, binds[1])].map(|(document, bind)| {
            let (store, generations) = generations();
            bind(&generations, dir.path());
            (store, generations, document)
        });
        let barrier = std::sync::Barrier::new(2);
        let [first, second] = thread::scope(|scope| {
            let calls = doors.each_ref().map(|(_, generations, document)| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    enforce(generations, *document)
                })
            });
            calls.map(|call| call.join().expect("no panic"))
        });
        let taken = SpecRejected::Unpersisted("generation 1 is on file already".to_owned());
        let ((won, ran), (lost, refused)) = match (first, second) {
            (Ok(ran), Err(refused)) => ((&doors[0], ran), (&doors[1], refused)),
            (Err(refused), Ok(ran)) => ((&doors[1], ran), (&doors[0], refused)),
            both => panic!("round {round}: one door runs generation 1: {both:?}"),
        };
        assert_eq!(refused, taken);
        let (number, digest) = ran;
        assert_eq!(number, 1);
        assert_eq!(standing(&won.0), (1, Some(digest)));
        assert_eq!(standing(&lost.0), (0, None));
        let on_file = std::fs::read(dir.path().join(name_of(1, "yaml"))).expect("the file");
        assert_eq!(on_file, won.2.as_bytes());

        // The two are gone, as their processes are at the next start.
        drop(doors);
        let (store, again) = generations();
        bind(&again, dir.path());
        assert_eq!(standing(&store), (1, Some(digest)));
    }
}

/// A pipe under a generation's name, with no writer, holds no start:
/// under the newest name it refuses the bind by that generation, and
/// under an older one it is not listed, as a file that is no document.
/// A bind runs on a thread of its own, waited five seconds, so an open
/// that waits fails the case.
#[cfg(unix)]
#[test]
fn a_pipe_under_a_generations_name_holds_no_start() {
    let bound_in_time = |dir: &Path| {
        let (store, generations) = generations();
        let opened = SpecStore::open(dir.to_owned()).expect("the directory opens");
        let (answer, answered) = mpsc::channel();
        thread::spawn(move || {
            let bound = generations.bind_store(opened, || false);
            let listed = generations.status().map(|status| status.retained);
            let _ = answer.send((bound, listed));
        });
        let (bound, listed) = answered
            .recv_timeout(Duration::from_secs(5))
            .expect("the bind answers: its open does not wait");
        (bound, standing(&store).0, listed)
    };
    let dir = tempfile::tempdir().expect("a temp dir");
    pipe_at(&dir.path().join(name_of(1, "yaml")));

    let (bound, runs, listed) = bound_in_time(dir.path());
    let refused = bound.expect_err("no regular file");
    assert!(
        matches!(
            &refused,
            StoreRefused::Rejected {
                generation: 1,
                source: SpecRejected::Unreadable(told)
            } if told == "the path leads to no regular file"
        ),
        "{refused:?}"
    );
    assert_eq!((runs, listed), (0, None));

    std::fs::write(dir.path().join(name_of(2, "yaml")), LEAST_LOAD).expect("the file");
    let (bound, runs, listed) = bound_in_time(dir.path());
    bound.expect("the newest runs");
    assert_eq!((runs, listed), (2, Some(Vec::new())));
}

/// At the last number none is left: an enforce and a rollback are
/// refused in the entry's words, as their dry run is, before a file is
/// written.
#[test]
fn a_write_at_the_last_number_is_refused_as_its_dry_run_is() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let last = u64::MAX;
    std::fs::write(dir.path().join(name_of(last - 1, "yaml")), RANDOM).expect("the file");
    std::fs::write(dir.path().join(name_of(last, "yaml")), LEAST_LOAD).expect("the file");
    let (store, generations) = generations();
    bind(&generations, dir.path());
    assert_eq!(standing(&store).0, last);
    let on_file = names(dir.path());
    let stale = || {
        SpecRejected::Stale(StaleGeneration {
            standing: last,
            offered: last,
        })
    };

    let dry = generations
        .dry_run("", RANDOM.as_bytes(), Format::Yaml)
        .expect("no guard");
    assert_eq!(dry.would_run, Err(stale()));
    assert_eq!(enforce(&generations, RANDOM), Err(stale()));
    assert_eq!(roll_back(&generations, last - 1), Err(stale()));
    assert_eq!(names(dir.path()), on_file);
    assert_eq!(generations.apply_counts(), applied(1, 2));
}

/// A generation that is not on file is not held: one never applied, and
/// one past the generations the store keeps. Nothing is applied and
/// nothing counted.
#[test]
fn a_rollback_to_a_generation_not_on_file_is_not_held() {
    let (_dir, store, generations) = stored();
    let last = u64::try_from(RETAINED).expect("a small count") + 1;
    for number in 1..=last {
        enforce(&generations, bound_to(number)).expect("it runs");
    }
    for gone in [1, last + 1] {
        let Err(EnforceRefused::NotHeld(not_held)) = generations.roll_back(gone) else {
            panic!("generation {gone} is not on file");
        };
        let expected = NotHeld {
            asked: gone,
            held: Some(last),
            standing: last,
            on_file: Some(OnFile {
                count: RETAINED,
                span: (2, last),
            }),
        };
        assert_eq!(not_held, expected);
        assert_eq!(
            not_held.to_string(),
            format!("no document is held for generation {gone}: the spec store has 8 on file, from generation 2 to generation 9")
        );
    }
    assert_eq!(standing(&store).0, last);
    assert_eq!(generations.apply_counts(), applied(last, 0));
}

/// A rollback mints its number whatever line the generation's bytes
/// write. A file that is no document any more, as a build that dropped
/// a stage finds one, is refused by name with the running generation
/// kept, counted, and nothing more on file.
#[test]
fn a_rollback_mints_its_number_and_refuses_a_document_that_does_not_run() {
    let (dir, store, generations) = stored();
    enforce(&generations, with_line(LEAST_LOAD, 1)).expect("it runs");
    enforce(&generations, RANDOM).expect("it runs");
    assert_eq!(roll_back(&generations, 1).map(|(number, _)| number), Ok(3));
    let on_file = std::fs::read(dir.path().join(name_of(3, "yaml"))).expect("the file");
    assert_eq!(on_file, with_line(LEAST_LOAD, 1));

    std::fs::write(dir.path().join(name_of(2, "yaml")), UNKNOWN_FILTER).expect("the file");
    let refused = roll_back(&generations, 2).expect_err("no such filter");
    assert!(matches!(refused, SpecRejected::Invalid(_)), "{refused:?}");
    assert_eq!(standing(&store).0, 3);
    assert_eq!(names(dir.path()).len(), 3);
    assert_eq!(generations.apply_counts(), applied(3, 1));
}

#[test]
fn the_standing_document_again_is_no_change_under_a_store() {
    let (dir, store, generations) = stored();
    let first = enforce(&generations, LEAST_LOAD).expect("it runs");
    // Again as written, again under a comment, again under the
    // standing line: one document, and nothing more on file.
    let commented = format!("# the same\n{LEAST_LOAD}").into_bytes();
    for again in [LEAST_LOAD.into(), commented, with_line(LEAST_LOAD, 1)] {
        assert_eq!(enforce(&generations, again), Ok(first));
    }
    assert_eq!(standing(&store).0, 1);
    assert_eq!(names(dir.path()).len(), 1);
    assert_eq!(generations.apply_counts(), applied(1, 0));
}
