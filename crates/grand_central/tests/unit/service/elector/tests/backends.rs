//! What every in-tree backend does with a record: it stores the record
//! equal, and it writes only over the record its writer expects.

use super::*;

// The backend contract, over every in-tree backend. The third element
// keeps the backend's place alive: the file's directory, the Kubernetes
// stand-in's runtime and server.

type Backend = (&'static str, Box<dyn LeaseBackend>, Box<dyn std::any::Any>);

fn backends() -> Vec<Backend> {
    let dir = tempfile::tempdir().expect("a temp dir");
    let file = FileLease::new(dir.path().join("lease"));
    let backends: Vec<Backend> = vec![
        ("memory", Box::new(MemoryLease::new()), Box::new(())),
        ("file", Box::new(file), Box::new(dir)),
    ];
    #[cfg(feature = "kubernetes")]
    let backends = {
        let mut backends = backends;
        let cell = kubernetes::tests::Cell::new();
        backends.push(("kubernetes", Box::new(cell.backend()), Box::new(cell)));
        backends
    };
    backends
}

#[test]
fn a_backend_stores_a_record_equal_and_writes_only_over_the_expected_one() {
    for (name, mut backend, _place) in backends() {
        assert_eq!(
            backend.read().expect("read"),
            None,
            "{name}: empty at first"
        );
        let first = LeaseRecord {
            holder: "a@host:1".to_string(),
            term: 1,
            acquired_at_micros: 5,
            renewed_at_micros: 5,
            duration_ms: 3_000,
        };
        assert_eq!(
            backend.write(Some(&first), &first).expect("write"),
            ConditionalWriteOutcome::Moved,
            "{name}: nothing to replace yet"
        );
        assert_eq!(
            backend.read().expect("read"),
            None,
            "{name}: a refused write stores nothing"
        );
        assert_eq!(
            backend.write(None, &first).expect("write"),
            ConditionalWriteOutcome::Done,
            "{name}"
        );
        assert_eq!(
            backend.read().expect("read"),
            Some(first.clone()),
            "{name}: equal, field for field"
        );
        let second = LeaseRecord {
            renewed_at_micros: 6,
            ..first.clone()
        };
        assert_eq!(
            backend.write(None, &second).expect("write"),
            ConditionalWriteOutcome::Moved,
            "{name}: the record exists"
        );
        assert_eq!(
            backend.write(Some(&second), &second).expect("write"),
            ConditionalWriteOutcome::Moved,
            "{name}: not the stored record"
        );
        assert_eq!(backend.read().expect("read"), Some(first.clone()), "{name}");
        assert_eq!(
            backend.write(Some(&first), &second).expect("write"),
            ConditionalWriteOutcome::Done,
            "{name}"
        );
        assert_eq!(backend.read().expect("read"), Some(second), "{name}");
        assert!(!backend.place().is_empty());
    }
}
