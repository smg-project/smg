//! The data-parallel load manager: cached rank loads per worker, the lowest-loaded rank selected
//! and incremented, nothing for a worker it has not seen.

use super::super::*;
use crate::{identity::basic::WorkerType, test_support::BasicWorkerBuilder};

fn worker_at(url: &str) -> Arc<dyn Worker> {
    Arc::new(BasicWorkerBuilder::new(url).build())
}

#[test]
fn test_new_dp_load_manager_instance() {
    let dp_load_manager = WorkerLoadManager::new();
    let cached = dp_load_manager.dp_cached_loads.read();
    assert!(cached.is_empty());
}

#[test]
fn test_update_dp_load() {
    let manager = WorkerLoadManager::new();
    let worker1 = worker_at("http://worker1:8080");
    let worker2 = worker_at("http://worker2:8080");

    let mut worker1_load = HashMap::new();
    worker1_load.insert(0, 2);
    worker1_load.insert(1, 1);

    let mut worker2_load = HashMap::new();
    worker2_load.insert(0, 3);

    manager.update_dp_loads(vec![(&worker1, worker1_load), (&worker2, worker2_load)]);

    let cached = manager.dp_cached_loads.read();
    assert_eq!(cached.len(), 2);

    let worker2_cache = &cached.get("http://worker2:8080").unwrap().loads;
    assert_eq!(worker2_cache.get(&0), Some(&3));
}

#[test]
fn test_select_and_increment_lowest_dp_load_multiple() {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test_key")
            .build(),
    );

    let manager = WorkerLoadManager::new();
    let mut worker_load = HashMap::new();
    worker_load.insert(0, 10);
    worker_load.insert(1, 3);
    worker_load.insert(2, 7);
    manager.update_dp_loads(vec![(&worker, worker_load)]);

    let selected = manager.select_and_increment_lowest_dp_load(worker.as_ref(), 4);

    assert_eq!(selected, Some(1));
    let cached = manager.dp_cached_loads.read();
    assert_eq!(
        *cached.get(worker.url()).unwrap().loads.get(&1).unwrap(),
        3 + 4
    );
}

#[test]
fn test_select_and_increment_lowest_dp_load_none_worker() {
    let worker = BasicWorkerBuilder::new("http://nonexist:8080")
        .worker_type(WorkerType::Regular)
        .api_key("test")
        .build();

    let manager = WorkerLoadManager::new();
    let result = manager.select_and_increment_lowest_dp_load(&worker, 1);
    assert_eq!(result, None);
}

/// The update line names each worker's address masked. The map is keyed by
/// the address as registered, which may hold a credential in its userinfo or
/// a token in its query; the loads are cached under the addresses as they
/// are.
#[test]
#[tracing_test::traced_test]
fn the_update_line_names_each_address_masked() {
    let credentialed = "grpc://user:s3cret@10.0.0.7:9000?token=t0ken";
    let plain = "grpc://10.0.0.8:9000";
    let manager = WorkerLoadManager::new();
    let (credentialed_worker, plain_worker) = (worker_at(credentialed), worker_at(plain));
    manager.update_dp_loads(vec![
        (&credentialed_worker, HashMap::from([(0, 3)])),
        (&plain_worker, HashMap::from([(1, 5)])),
    ]);

    logs_assert(|lines: &[&str]| {
        let written: Vec<&str> = lines
            .iter()
            .copied()
            .filter(|line| line.contains("update_dp_loads"))
            .collect();
        let [line] = written.as_slice() else {
            return Err(format!(
                "{} update lines, wanted one: {written:?}",
                written.len()
            ));
        };
        for hidden in ["s3cret", "t0ken", "user:"] {
            if line.contains(hidden) {
                return Err(format!("the update line shows `{hidden}`: {line}"));
            }
        }
        for shown in [
            "\"grpc://***@10.0.0.7:9000\": {0: 3}",
            "\"grpc://10.0.0.8:9000\": {1: 5}",
        ] {
            if !line.contains(shown) {
                return Err(format!("the update line does not write `{shown}`: {line}"));
            }
        }
        Ok(())
    });
    let cached = manager.dp_cached_loads.read();
    assert_eq!(
        cached.get(credentialed).map(|laid| &laid.loads),
        Some(&HashMap::from([(0, 3)]))
    );
    assert_eq!(
        cached.get(plain).map(|laid| &laid.loads),
        Some(&HashMap::from([(1, 5)]))
    );
}

/// The loads a poll laid are its worker's. A poll can end behind its
/// worker's eviction and lay them where the next worker registered at the
/// URL takes requests: that worker counts on a runtime of its own and is
/// given no rank by them.
#[test]
fn loads_laid_for_a_worker_give_no_rank_to_one_registered_anew_at_its_url() {
    let manager = WorkerLoadManager::new();
    let left = worker_at("http://worker:8080");
    let came = worker_at("http://worker:8080");
    manager.update_dp_loads(vec![(&left, HashMap::from([(0, 900), (1, 100)]))]);

    assert_eq!(
        manager.select_and_increment_lowest_dp_load(came.as_ref(), 0),
        None,
        "the worker that came is given a rank by the loads of the worker that left"
    );
    assert_eq!(
        manager.select_and_increment_lowest_dp_load(left.as_ref(), 0),
        Some(1)
    );
}

/// A same-URL `replace()` hands the worker that comes the runtime of the
/// handle that leaves: it reads the loads the handle's poll laid, as it
/// did, until a poll of its own lays the next.
#[test]
fn loads_laid_for_a_worker_are_read_by_the_worker_that_replaces_it() {
    let manager = WorkerLoadManager::new();
    let left = worker_at("http://worker:8080");
    let came = worker_at("http://worker:8080");
    assert!(came.inherit_shared_state_from(left.as_ref()));
    manager.update_dp_loads(vec![(&left, HashMap::from([(0, 900), (1, 100)]))]);
    drop(left);

    assert_eq!(
        manager.select_and_increment_lowest_dp_load(came.as_ref(), 0),
        Some(1)
    );
}
