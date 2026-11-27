use openai_protocol::worker::WorkerStatus;

use super::*;
use crate::policies::test_support::make_worker;

#[test]
fn test_selects_only_worker() {
    let policy = PassthroughPolicy::new();
    let workers = vec![make_worker("http://w1:8000")];
    for _ in 0..10 {
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }
}

#[test]
fn test_selects_first_healthy_with_two_workers() {
    let policy = PassthroughPolicy::new();
    let workers = vec![make_worker("http://w1:8000"), make_worker("http://w2:8000")];
    // Deterministic: always the first healthy index.
    for _ in 0..10 {
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }
}

#[test]
fn test_skips_unhealthy_first_worker() {
    let policy = PassthroughPolicy::new();
    let workers = vec![make_worker("http://w1:8000"), make_worker("http://w2:8000")];
    workers[0].set_status(WorkerStatus::NotReady);
    for _ in 0..10 {
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }
}

#[test]
fn test_none_when_empty() {
    let policy = PassthroughPolicy::new();
    let workers: Vec<Arc<dyn Worker>> = vec![];
    assert_eq!(
        policy.select_worker(&workers, &SelectWorkerInfo::default()),
        None
    );
}

#[test]
fn test_none_when_all_unhealthy() {
    let policy = PassthroughPolicy::new();
    let workers = vec![make_worker("http://w1:8000"), make_worker("http://w2:8000")];
    workers[0].set_status(WorkerStatus::NotReady);
    workers[1].set_status(WorkerStatus::NotReady);
    assert_eq!(
        policy.select_worker(&workers, &SelectWorkerInfo::default()),
        None
    );
}

#[test]
fn test_name() {
    assert_eq!(PassthroughPolicy::new().name(), "passthrough");
}
