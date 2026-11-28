//! What an operator reads: the document the read call answers, and the
//! status a snapshot tells of the spec.

use std::sync::Arc;

use super::{
    super::{Generations, Unanswered},
    bind, enforce, generations, roll_back, stored, yaml,
};
use crate::{
    document::{
        applied::AppliedSpec,
        operator::{
            test_support::{bound_to, standing, stock, LEAST_LOAD, RANDOM},
            ApplyOutcome, EnforceRefused, NotHeld, OnFile, SpecRejected,
        },
        spec::PolicySpec,
    },
    service::{
        controller::{
            spec_store::{
                tests::{name_of, names},
                SpecStore,
            },
            wire,
            wire::unix_micros,
        },
        proto,
    },
};

/// The read call answers the source's bytes, the comment included, with
/// the number and the digest beside them; a generation that does not
/// stand has no document, and neither has a generation the store was
/// handed behind the source.
#[test]
fn the_read_answer_is_the_sources_bytes_beside_the_number_and_the_digest() {
    let (store, generations) = generations();
    let nothing = NotHeld {
        asked: 0,
        held: None,
        standing: 0,
        on_file: None,
    };
    assert_eq!(generations.document(0), Err(nothing));

    let written = format!("# routed by load\n{LEAST_LOAD}").into_bytes();
    assert_eq!(
        yaml(&generations, written.clone()),
        Ok(ApplyOutcome::Generation(1))
    );
    let (_, Some(digest)) = standing(&store) else {
        panic!("an applied generation has a digest");
    };
    for wanted in [0, 1] {
        let answer =
            wire::spec_document(generations.document(wanted).expect("the standing document"));
        assert_eq!(answer.generation, 1);
        assert_eq!(answer.digest, digest.to_string());
        assert_eq!(
            answer.document, written,
            "the bytes as the source wrote them"
        );
        assert_eq!(answer.format, "yaml");
        assert_eq!(answer.mode, i32::from(proto::ApplyMode::Enforce));
        assert!(answer.applied_unix_micros > 0);
        assert!(answer.acks.is_empty());
    }

    let older = generations.document(2).expect_err("no generation 2");
    assert_eq!(
            older.to_string(),
            "no document is held for generation 2: the spec source's document is generation 1, generation 1 stands, and the standing generation's document alone is retained"
        );

    let mut preset = PolicySpec::preset("random").expect("a preset");
    preset.generation = 7;
    store
        .apply(AppliedSpec::validate(&preset).expect("a preset validates"))
        .expect("above the standing one");
    let behind = NotHeld {
        asked: 7,
        held: Some(1),
        standing: 7,
        on_file: None,
    };
    assert_eq!(generations.document(0), Err(behind));
}

/// The status a snapshot tells of the spec. No source: none. A file:
/// the standing document's number, digest and time, and nothing
/// retained. A store: the generations on file that a rollback can name
/// beside the standing one, and a start on the directory lists the same.
#[test]
fn the_status_names_the_standing_generation_and_what_a_rollback_can_name() {
    let (_store, sourceless) = generations();
    assert_eq!(sourceless.status(), None);

    let (_store, file) = generations();
    yaml(&file, LEAST_LOAD).expect("it runs");
    let status = file.status().map(wire::spec_status).expect("a source");
    assert!(status.applied_unix_micros > 0, "{status:?}");
    assert_eq!(
        status,
        proto::SpecStatus {
            generation: 1,
            digest: file.standing().1.expect("a digest").to_string(),
            applied_unix_micros: status.applied_unix_micros,
            mode: proto::ApplyMode::Enforce.into(),
            retained: Vec::new(),
            source: "file".to_owned(),
        }
    );

    // A store with nothing on file: the store's own number, and no
    // document's digest, time or mode.
    let (dir, _store, stored) = stored();
    let empty = proto::SpecStatus {
        source: "api".to_owned(),
        ..proto::SpecStatus::default()
    };
    assert_eq!(stored.status().map(wire::spec_status), Some(empty));
    for (number, document) in [
        (1, LEAST_LOAD.to_owned()),
        (2, RANDOM.to_owned()),
        (3, bound_to(3)),
    ] {
        let ran = enforce(&stored, document).map(|(number, _)| number);
        assert_eq!(ran, Ok(number));
    }
    let status = stored.status().map(wire::spec_status).expect("a source");
    assert_eq!(
        (
            status.generation,
            status.retained.as_slice(),
            status.source.as_str()
        ),
        (3, &[1, 2][..], "api")
    );
    assert_eq!(
        status.digest,
        stored.standing().1.expect("a digest").to_string()
    );
    assert_eq!(status.mode, i32::from(proto::ApplyMode::Enforce));
    assert!(status.applied_unix_micros > 0, "{status:?}");

    // A rollback runs generation 1 again as generation 4: the three
    // before it are what the next rollback can name.
    assert_eq!(roll_back(&stored, 1).map(|(number, _)| number), Ok(4));
    let rolled = stored.status().map(wire::spec_status).expect("a source");
    assert_eq!(
        (rolled.generation, rolled.retained.as_slice()),
        (4, &[1, 2, 3][..])
    );

    // A start on the same directory lists the same.
    drop(stored);
    let (_store, again) = generations();
    bind(&again, dir.path());
    let listed = again.status().map(wire::spec_status).expect("a source");
    assert_eq!(
        (
            listed.generation,
            listed.retained,
            listed.digest,
            listed.source
        ),
        (4, vec![1, 2, 3], rolled.digest, "api".to_owned())
    );
}

/// A door tells the time and the mode of a document it holds, and of no
/// other: where the store runs a generation another door applied, the
/// status names that generation with its digest and no time.
#[test]
fn the_status_tells_no_time_of_a_document_its_door_does_not_hold() {
    let (store, first) = generations();
    yaml(&first, LEAST_LOAD).expect("it runs");
    let second = Generations::new(Arc::clone(&store), stock());
    yaml(&second, RANDOM).expect("it runs");
    let (generation, digest) = first.standing();
    assert_eq!(generation, 2);
    assert_eq!(
        first.status().map(wire::spec_status),
        Some(proto::SpecStatus {
            generation,
            digest: digest.expect("a digest").to_string(),
            source: "file".to_owned(),
            ..proto::SpecStatus::default()
        })
    );
    // The door that applied it tells its time.
    let told = second.status().map(wire::spec_status).expect("a source");
    assert_eq!(told.mode, i32::from(proto::ApplyMode::Enforce));
    assert!(told.applied_unix_micros > 0, "{told:?}");
}

/// A generation whose file is gone under the list, as an apply's prune
/// takes one between a read's look and its open, is not held: the read
/// answers so, and no internal fault.
#[tokio::test]
async fn a_generation_whose_file_is_gone_under_the_list_is_not_held() {
    let (dir, _store, generations) = stored();
    enforce(&generations, LEAST_LOAD).expect("it runs");
    enforce(&generations, RANDOM).expect("it runs");
    std::fs::remove_file(dir.path().join(name_of(1, "yaml"))).expect("the file goes");

    let gone = generations.read(1).await;
    let Err(Unanswered::Manager(EnforceRefused::NotHeld(not_held))) = gone else {
        panic!("generation 1 has no file: {gone:?}");
    };
    assert_eq!((not_held.asked, not_held.standing), (1, 2));
}

/// The read call answers a generation on file by its number, with the
/// digest it ran under and its file's time, while the door's lock is
/// held: it waits on no apply. A generation that is not on file is not
/// held. A start lists the generations on file again, under the same
/// digests and times.
#[tokio::test]
#[expect(
    clippy::await_holding_lock,
    reason = "the case holds the door's lock across the read: the read takes none"
)]
async fn the_read_call_answers_a_generation_on_file() {
    let (dir, _store, generations) = stored();
    let (_, first) = enforce(&generations, LEAST_LOAD).expect("it runs");
    let (_, second) = enforce(&generations, RANDOM).expect("it runs");

    let applying = generations.applying_lock().lock();
    let old = generations.read(1).await.expect("on file");
    drop(applying);
    assert_eq!(
        (old.generation, old.document.as_slice()),
        (1, LEAST_LOAD.as_bytes())
    );
    assert_eq!(
        (old.digest.clone(), old.format.as_str()),
        (first.to_string(), "yaml")
    );
    let written = std::fs::metadata(dir.path().join(name_of(1, "yaml")))
        .and_then(|file| file.modified())
        .expect("the file's time");
    assert_eq!(old.applied_unix_micros, unix_micros(written));
    let stands = generations.read(0).await.expect("the standing one");
    assert_eq!(
        (stands.generation, stands.digest.clone()),
        (2, second.to_string())
    );

    let Err(Unanswered::Manager(EnforceRefused::NotHeld(not_held))) = generations.read(7).await
    else {
        panic!("generation 7 is not on file");
    };
    let two = OnFile {
        count: 2,
        span: (1, 2),
    };
    assert_eq!(not_held.on_file, Some(two));

    // The door is gone, as its process is at the next start.
    drop(generations);
    let (_store, again) = self::generations();
    bind(&again, dir.path());
    let again = Arc::new(again);
    assert_eq!(again.read(1).await.expect("on file"), old);
    assert_eq!(again.read(2).await.expect("on file"), stands);
}

/// The read call reads a generation on file through the directory the
/// store opened. Where the path leads to another directory before the
/// door takes the store, the answer is still the held file's bytes,
/// not the other's under that name.
#[tokio::test]
async fn the_read_call_reads_the_directory_held_where_the_path_leads_elsewhere() {
    let base = tempfile::tempdir().expect("a temp dir");
    let (dir, moved) = (base.path().join("store"), base.path().join("moved"));
    std::fs::create_dir(&dir).expect("the directory");
    std::fs::write(dir.join(name_of(1, "yaml")), LEAST_LOAD).expect("the first");
    std::fs::write(dir.join(name_of(2, "yaml")), RANDOM).expect("the second");
    let opened = SpecStore::open(dir.clone()).expect("the directory opens");
    std::fs::rename(&dir, &moved).expect("the directory moves");
    std::fs::create_dir(&dir).expect("another stands at its path");
    std::fs::write(dir.join(name_of(1, "yaml")), "other").expect("the other's own");
    let (_store, generations) = generations();
    generations.bind_store(opened).expect("the store binds");
    let old = generations.read(1).await.expect("on file");
    assert_eq!(
        (old.generation, old.document.as_slice()),
        (1, LEAST_LOAD.as_bytes())
    );
}

/// An older file that is no document of this build stays on file and is
/// listed nowhere: the start runs the newest, the read call does not
/// answer the older one, and a rollback to it is refused by name.
#[tokio::test]
async fn an_older_file_that_is_no_document_is_left_on_file_and_not_read() {
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::write(dir.path().join(name_of(1, "yaml")), "profiles: [").expect("the file");
    std::fs::write(dir.path().join(name_of(2, "yaml")), LEAST_LOAD).expect("the file");
    let (store, generations) = generations();
    bind(&generations, dir.path());
    let generations = Arc::new(generations);
    assert_eq!(standing(&store).0, 2);

    let Err(Unanswered::Manager(EnforceRefused::NotHeld(not_held))) = generations.read(1).await
    else {
        panic!("generation 1 is no document");
    };
    let one = OnFile {
        count: 1,
        span: (2, 2),
    };
    assert_eq!(not_held.on_file, Some(one));
    let refused = roll_back(&generations, 1).expect_err("no document");
    assert!(
        matches!(refused, SpecRejected::Unparsed { .. }),
        "{refused:?}"
    );
    assert_eq!(names(dir.path()).len(), 2);
}
