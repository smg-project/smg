//! The operator calls over the same store a host uses in process.

use std::{sync::Arc, time::Instant};

use smg_policy_controller::{
    operator::{
        snapshot::{told, Asked, Rows},
        Applied, Format, SpecDoor, SpecRejected, Unanswered,
    },
    store::{MemoryStore, WorkerFacts, WorkersStore},
    Catalog,
};

#[test]
fn a_host_applies_its_start_document_and_serves_a_read_only_source() {
    let store = Arc::new(MemoryStore::new());
    let door = SpecDoor::new(Arc::clone(&store), Arc::new(Catalog::builtin()), Vec::new());
    let start = br#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"pick":"random"}}}"#;
    let changed = br#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"score":["least-load"],"pick":"best"}}}"#;
    assert_eq!(door.status(), None);

    let dry = door.dry_run("", start, Format::Json).expect("no guard");
    let (generation, digest) = dry.would_run.expect("the start would run");
    assert_eq!((dry.standing, generation), ((0, None), 1));
    assert_eq!(door.standing(), (0, None));
    assert_eq!(
        door.apply(start.to_vec(), Format::Json),
        Ok(Applied::Generation(1))
    );

    let status = door.status().expect("the start's status");
    assert_eq!((status.generation, status.digest), (1, Some(digest)));
    assert!(status.applied.is_some());
    assert!(status.retained.is_empty() && !status.writable);
    let held = door.document(0).expect("the start's document");
    assert_eq!(held.document, start);
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
    let Err(Unanswered::NotHeld(refused)) = door.roll_back(1) else {
        panic!("a rollback needs a generation on file");
    };
    assert_eq!(
        refused.to_string(),
        "no document is held for generation 1: the spec source's document is generation 1, generation 1 stands, and the standing generation's document alone is retained"
    );
    assert_eq!(door.document(0), Ok(held));
    assert_eq!((door.counts().applied, door.counts().rejected), (1, 1));

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
    let asked = Asked {
        models: vec!["served".to_owned()],
        limit: 1,
    };
    let (rows, serving) = store.rows_serving(&asked.models, asked.limit);
    let page = told(&asked, Rows::Told(rows, serving), Instant::now());
    let [row] = page.rows.as_slice() else {
        panic!("one row fits the limit");
    };
    assert_eq!(row.facts.key, "grpc://b:8000");
    assert!(page.truncated);

    let asked = Asked { limit: 2, ..asked };
    let page = told(
        &asked,
        Rows::Shared(Arc::new(
            store.cut(smg_policy_controller::store::Kinds::Watched),
        )),
        Instant::now(),
    );
    assert!(!page.truncated);
    let last = page.rows.last().expect("the second matching row");
    assert_eq!(last.facts.key, "grpc://c:8000");
    assert_eq!(last.facts.engine, "grpc://c:8000");
}
