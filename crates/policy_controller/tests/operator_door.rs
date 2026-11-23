//! The operator calls over the same store a host uses in process.

use std::{error::Error, sync::Arc, time::Instant};

use smg_policy_controller::{
    document::operator::{
        snapshot::{worker_rows_page, Rows, WorkerRowsQuery},
        ApplyOutcome, DocumentManager, EnforceRefused, Format, SpecRejected, MAX_DOCUMENT_BYTES,
    },
    stages::StageRegistry,
    store::{MemoryStore, StaleGeneration, WorkerFacts, WorkersStore},
};

const START: &[u8] = br#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"pick":"random"}}}"#;

#[test]
fn a_host_applies_its_start_document_and_serves_a_read_only_source() {
    let store = Arc::new(MemoryStore::new());
    let door = DocumentManager::new(
        Arc::clone(&store),
        Arc::new(StageRegistry::builtin()),
        Vec::new(),
    );
    let changed = br#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"score":["least-load"],"pick":"best"}}}"#;
    assert_eq!(door.status(), None);

    let dry = door.dry_run("", START, Format::Json).expect("no guard");
    let (generation, digest) = dry.would_run.expect("the start would run");
    assert_eq!((dry.standing, generation), ((0, None), 1));
    assert_eq!(door.standing(), (0, None));
    assert_eq!(
        door.apply(START.to_vec(), Format::Json),
        Ok(ApplyOutcome::Generation(1))
    );

    let status = door.status().expect("the start's status");
    assert_eq!((status.generation, status.digest), (1, Some(digest)));
    assert!(status.applied.is_some());
    assert!(status.retained.is_empty() && !status.writable);
    let held = door.document(0).expect("the start's document");
    assert_eq!(held.document, START);
    assert_eq!(held.format, Format::Json);
    assert_eq!(Some(held.applied), status.applied);

    let dry = door.dry_run("", changed, Format::Json).expect("no guard");
    assert_eq!(dry.would_run.expect("the changed document would run").0, 2);
    let refused = door
        .enforce("", changed.to_vec(), Format::Json)
        .expect("no guard");
    assert_eq!(
        refused.ran,
        Err(SpecRejected::Unpersisted(
            "no spec store is bound".to_owned()
        ))
    );
    // A roll back is a write too: with no spec store bound it is refused in
    // an enforce's words, and each refusal is counted.
    assert!(!door.writable());
    let rolled = door.roll_back(1).expect("a refusal of the write");
    assert_eq!(rolled.standing, (1, Some(digest)));
    assert_eq!(rolled.ran, refused.ran);
    assert_eq!(door.document(0), Ok(held));
    assert_eq!(
        (door.apply_counts().applied, door.apply_counts().rejected),
        (1, 2)
    );

    for (key, model) in [
        ("grpc://b:8000", "served"),
        ("grpc://a:8000", "other"),
        ("grpc://operator:secret@c:8000", "served"),
    ] {
        store.upsert(WorkerFacts {
            key: key.to_owned(),
            engine: key.to_owned(),
            incarnation: 1,
            models: vec![model.to_owned()],
            dp_size: 1,
            ..WorkerFacts::default()
        });
    }
    let asked = WorkerRowsQuery {
        models: vec!["served".to_owned()],
        limit: 1,
    };
    let (rows, serving) = store.rows_serving(&asked.models, asked.limit);
    let page = worker_rows_page(&asked, Rows::Told(rows, serving), Instant::now());
    let [row] = page.rows.as_slice() else {
        panic!("one row fits the limit");
    };
    assert_eq!(row.facts.key, "grpc://b:8000");
    assert!(page.truncated);

    let asked = WorkerRowsQuery { limit: 2, ..asked };
    let page = worker_rows_page(
        &asked,
        Rows::Shared(Arc::new(store.checkpoint())),
        Instant::now(),
    );
    assert!(!page.truncated);
    let last = page.rows.last().expect("the second matching row");
    assert_eq!(last.facts.key, "grpc://c:8000");
    assert_eq!(last.facts.engine, "grpc://c:8000");
}

/// A refusal is an error a host can box and print, and a missing document
/// says what was asked, what is held and what stands.
#[test]
fn a_host_reads_a_refusal_as_an_error() {
    let store = Arc::new(MemoryStore::new());
    let door = DocumentManager::new(store, Arc::new(StageRegistry::builtin()), Vec::new());
    let guarded: Box<dyn Error> = Box::new(
        door.dry_run("another digest", START, Format::Json)
            .expect_err("no digest stands"),
    );
    assert_eq!(guarded.to_string(), "expected_digest: no generation stands");

    door.apply(START.to_vec(), Format::Json)
        .expect("the start runs");
    let missing = door.document(2).expect_err("generation 1 alone is held");
    assert_eq!(
        (
            missing.asked,
            missing.held,
            missing.standing,
            missing.on_file
        ),
        (2, Some(1), 1, None)
    );
    let words = missing.to_string();
    assert!(
        words.starts_with("no document is held for generation 2: "),
        "{words}"
    );
    let unanswered: Box<dyn Error> = Box::new(EnforceRefused::NotHeld(missing));
    assert_eq!(unanswered.to_string(), words);
}

/// A host's refused apply counts once: by the form's reader, by the bound
/// a host reads, and by the host's own publication.
#[test]
fn a_host_counts_each_document_it_cannot_apply() {
    let store = Arc::new(MemoryStore::new());
    let door = DocumentManager::new(store, Arc::new(StageRegistry::builtin()), Vec::new());
    let json = Format::named("json").expect("a form's name");
    let unread = door.apply(b"[".to_vec(), json);
    assert!(
        matches!(unread, Err(SpecRejected::Unparsed { .. })),
        "{unread:?}"
    );
    let long = vec![b' '; MAX_DOCUMENT_BYTES as usize + 1];
    assert_eq!(
        door.apply(long, Format::Yaml),
        Err(SpecRejected::TooLarge(MAX_DOCUMENT_BYTES + 1))
    );
    let stale = StaleGeneration {
        offered: 1,
        standing: 1,
    };
    let unpublished = door.apply_where(START.to_vec(), json, |_| None, |_, _| Err(stale));
    assert_eq!(unpublished, Err(SpecRejected::Stale(stale)));
    assert_eq!(
        (door.apply_counts().applied, door.apply_counts().rejected),
        (0, 3)
    );
}

/// A null under a built-in rule's name, the slip of a bare name written
/// with a colon, is told both forms at the rule's own path, in both
/// formats: the door hands the rule's type what the document wrote.
#[test]
fn a_null_under_a_built_in_rule_is_told_both_forms_at_its_path() {
    let store = Arc::new(MemoryStore::new());
    let door = DocumentManager::new(store, Arc::new(StageRegistry::builtin()), Vec::new());
    let json = br#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"random","filter":[{"has-room":null}]}}}"#;
    let yaml = b"apiVersion: smg.policy/v1\nprofiles:\n  p:\n    pick: random\n    filter:\n    - has-room:\n";
    for (format, bytes) in [
        (Format::Json, json.as_slice()),
        (Format::Yaml, yaml.as_slice()),
    ] {
        let dry = door.dry_run("", bytes, format).expect("no guard");
        let refused = dry.would_run.expect_err("a null rule");
        assert!(
            matches!(
                &refused,
                SpecRejected::Unparsed { path, message, .. }
                    if path == "profiles.p.filter[0].has-room"
                        && message.contains(
                            "expected a rule with `used` and `capacity`, or the bare name `has-room`"
                        )
            ),
            "{refused:?}"
        );
    }
}

/// A parameter no built-in rule holds is refused at its own path, in both
/// formats: the rule's keys are read by the door's own reader, so the first
/// key written keeps its place.
#[test]
fn an_unknown_parameter_of_a_built_in_rule_is_refused_at_its_own_path() {
    let store = Arc::new(MemoryStore::new());
    let door = DocumentManager::new(store, Arc::new(StageRegistry::builtin()), Vec::new());
    let json = br#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"random","filter":[{"has-room":{"arp":0,"used":{"load.running":1},"capacity":10}}]}}}"#;
    let yaml = b"apiVersion: smg.policy/v1\nprofiles:\n  p:\n    pick: random\n    filter:\n    - has-room: {arp: 0, used: {load.running: 1}, capacity: 10}\n";
    for (format, bytes) in [
        (Format::Json, json.as_slice()),
        (Format::Yaml, yaml.as_slice()),
    ] {
        let dry = door.dry_run("", bytes, format).expect("no guard");
        let refused = dry.would_run.expect_err("an unknown parameter");
        assert!(
            matches!(
                &refused,
                SpecRejected::Unparsed { path, message, .. }
                    if path == "profiles.p.filter[0].has-room.arp"
                        && message.starts_with("unknown field `arp`")
            ),
            "{refused:?}"
        );
    }
}
