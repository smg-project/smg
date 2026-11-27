use std::{collections::HashMap, time::Instant};

use smg_auth::RequestId;
use tokio::time::{sleep, Duration};

use super::*;
use crate::routers::grpc::backend_client::BasicWorkerBuilder;

#[tokio::test]
async fn priority_mode_applies_capacity_changes_after_startup() {
    let registry = Arc::new(WorkerRegistry::new());
    let config = RouterConfig {
        max_concurrent_requests: 256,
        priority_scheduler_enabled: true,
        ..RouterConfig::default()
    };
    let AdmissionMode::Priority(state) =
        AdmissionMode::try_build_priority(&config, Arc::clone(&registry), None).unwrap()
    else {
        panic!("priority scheduler should start");
    };

    let mut labels = HashMap::new();
    labels.insert("max_running_requests".to_string(), "1".to_string());
    let worker = Arc::new(
        BasicWorkerBuilder::new("http://capacity-test:8000")
            .labels(labels)
            .status(openai_protocol::worker::WorkerStatus::Ready)
            .build(),
    );
    registry.register(worker).expect("worker should register");

    // The dispatcher must retain the tracker and apply its update. Once
    // capacity drops to one, a second System request cannot acquire a
    // slot while the first is still in flight.
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut attempt = 0;
    loop {
        let first = state.scheduler.acquire_inflight(
            Class::System,
            RequestId(format!("capacity-first-{attempt}")),
        );
        let second = state.scheduler.acquire_inflight(
            Class::System,
            RequestId(format!("capacity-second-{attempt}")),
        );
        let updated = first.is_some() && second.is_none();
        drop(second);
        drop(first);

        if updated {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "scheduler did not apply the worker capacity update"
        );
        attempt += 1;
        sleep(Duration::from_millis(10)).await;
    }
}
