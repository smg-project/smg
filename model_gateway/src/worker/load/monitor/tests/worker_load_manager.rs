//! The data-parallel load manager: cached rank loads per worker, the lowest-loaded rank selected
//! and incremented, nothing for a worker it has not seen.

use super::super::*;
use crate::worker::{BasicWorkerBuilder, WorkerType};

#[test]
fn test_new_dp_load_manager_instance() {
    let dp_load_manager = WorkerLoadManager::new();
    let cached = dp_load_manager.dp_cached_loads.read();
    assert!(cached.is_empty());
}

#[test]
fn test_update_dp_load() {
    let manager = WorkerLoadManager::new();
    let mut loads = HashMap::new();

    let mut worker1_load = HashMap::new();
    worker1_load.insert(0, 2);
    worker1_load.insert(1, 1);
    loads.insert("http://worker1:8080".to_string(), worker1_load);

    let mut worker2_load = HashMap::new();
    worker2_load.insert(0, 3);
    loads.insert("http://worker2:8080".to_string(), worker2_load);

    manager.update_dp_loads(&loads);

    let cached = manager.dp_cached_loads.read();
    assert_eq!(cached.len(), 2);

    let worker2_cache = cached.get("http://worker2:8080").unwrap();
    assert_eq!(worker2_cache.get(&0), Some(&3));
}

#[test]
fn test_select_and_increment_lowest_dp_load_multiple() {
    let worker = BasicWorkerBuilder::new("http://worker:8080")
        .worker_type(WorkerType::Regular)
        .api_key("test_key")
        .build();

    let manager = WorkerLoadManager::new();
    let mut loads = HashMap::new();
    let mut worker_load = HashMap::new();
    worker_load.insert(0, 10);
    worker_load.insert(1, 3);
    worker_load.insert(2, 7);
    loads.insert(worker.url().to_string(), worker_load);
    manager.update_dp_loads(&loads);

    let selected = manager.select_and_increment_lowest_dp_load(&worker, 4);

    assert_eq!(selected, Some(1));
    let cached = manager.dp_cached_loads.read();
    assert_eq!(*cached.get(worker.url()).unwrap().get(&1).unwrap(), 3 + 4);
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
