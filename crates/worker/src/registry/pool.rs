//! The pool-size gauge: how many workers the registry holds in each pool
//! group.
//!
//! The registry is the gauge's one writer. A worker enters, is replaced or
//! leaves at one of the registry's three writes, and each moves the count
//! of the groups it touches and writes their gauge. So the gauge reads the
//! same for a worker a workflow registered, for one a membership store's
//! row laid and for one a mesh peer published.
//!
//! A worker counts once, in the group of its primary model, so the groups
//! add up to the registry's workers. The model index lists a worker under
//! every model it serves, and this count does not.

use std::collections::HashMap;

use crate::{
    identity::basic::{ConnectionMode, ConnectionModeExt, Worker, WorkerType, WorkerTypeExt},
    metrics::writers::Metrics,
};

/// The key the pool-size gauge is kept by: a worker's type, connection mode
/// and primary model id.
type PoolGroup = (WorkerType, ConnectionMode, String);

/// The gauge's key for one worker. It reads the worker's spec, which stands
/// for the handle's life: a refresh of the worker's model cards moves no
/// group, so a worker leaves the group it entered.
fn pool_group(worker: &dyn Worker) -> PoolGroup {
    let meta = worker.metadata();
    (
        meta.spec.worker_type,
        meta.spec.connection_mode,
        worker.model_id().to_string(),
    )
}

/// The count of workers in each pool group, moved by a delta at each of the
/// registry's writes: no write walks the registry to count a group.
///
/// A count and its gauge move in one hold. Written after the release, the
/// gauge writes of two workers of one group could land in the reverse
/// order and pin a stale value until the group's next write.
#[derive(Debug, Default)]
pub(super) struct PoolSizes {
    sizes: parking_lot::Mutex<HashMap<PoolGroup, usize>>,
}

impl PoolSizes {
    /// `worker` entered the registry.
    pub(super) fn entered(&self, worker: &dyn Worker) {
        self.moved(None, Some(pool_group(worker)));
    }

    /// `new` took the place of `old`. Both groups move in this one step,
    /// and a worker that stays in its group moves nothing.
    pub(super) fn replaced(&self, old: &dyn Worker, new: &dyn Worker) {
        self.moved(Some(pool_group(old)), Some(pool_group(new)));
    }

    /// `worker` left the registry.
    pub(super) fn left(&self, worker: &dyn Worker) {
        self.moved(Some(pool_group(worker)), None);
    }

    /// Take one worker out of the group `left` and put one into the group
    /// `entered`, and write the gauge of each group that moved.
    fn moved(&self, left: Option<PoolGroup>, entered: Option<PoolGroup>) {
        if left == entered {
            return;
        }
        let mut sizes = self.sizes.lock();
        if let Some(group) = left {
            // A group that empties leaves the map, and its gauge keeps a
            // zero sample.
            let size = match sizes.get(&group) {
                Some(size) => size.saturating_sub(1),
                None => {
                    uncounted(&group);
                    0
                }
            };
            write_gauge(&group, size);
            if size == 0 {
                sizes.remove(&group);
            } else {
                sizes.insert(group, size);
            }
        }
        if let Some(group) = entered {
            let size = sizes.get(&group).map_or(1, |size| size + 1);
            write_gauge(&group, size);
            sizes.insert(group, size);
        }
    }
}

/// A worker left `group` and the group holds no count, so a write of the
/// registry missed its call. Nothing counts a group again, and its gauge
/// would read wrong until a restart: a release build says so in a log line
/// and a debug build stops here, where the registry's cases catch it.
fn uncounted(group: &PoolGroup) {
    let (worker_type, connection_mode, model_id) = group;
    tracing::warn!(
        worker_type = worker_type.as_metric_label(),
        connection_mode = connection_mode.as_metric_label(),
        model = %model_id,
        "a worker left a pool group that holds no count"
    );
    debug_assert!(false, "pool group {group:?} left with no count");
}

/// Write `size` to the pool-size gauge of `group`.
fn write_gauge(group: &PoolGroup, size: usize) {
    let (worker_type, connection_mode, model_id) = group;
    Metrics::set_worker_pool_size(
        worker_type.as_metric_label(),
        connection_mode.as_metric_label(),
        model_id,
        size,
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

    use crate::{
        identity::basic::{ConnectionMode, Worker},
        model_card::ModelCard,
        test_support::{BasicWorkerBuilder, WorkerRegistry},
    };

    const A: &str = "http://a:8000";
    const B: &str = "http://b:8000";

    fn worker(url: &str, model: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model))
                .build(),
        )
    }

    /// The gauge's sample for the group whose labels `labels` names, or
    /// none where the group never had a worker.
    fn size(handle: &PrometheusHandle, labels: &[&str]) -> Option<String> {
        handle
            .render()
            .lines()
            .find(|line| {
                line.starts_with("smg_worker_pool_size{")
                    && labels.iter().all(|label| line.contains(label))
            })
            .and_then(|line| line.rsplit(' ').next().map(str::to_owned))
    }

    /// Each of the registry's writes moves the gauge of the groups it
    /// touches, with no step beside it: a worker that enters, a second of
    /// its group, and the last one's leaving, which keeps a zero sample.
    #[test]
    fn the_gauge_counts_a_group_as_its_workers_enter_and_leave() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let of = |model: &str| size(&handle, &[&format!("model=\"{model}\"")]);
        metrics::with_local_recorder(&recorder, || {
            let registry = WorkerRegistry::new();
            assert_eq!(of("m"), None);
            let first = registry.register(worker(A, "m")).expect("a registers");
            assert_eq!(of("m").as_deref(), Some("1"));
            let second = registry.register_or_replace(worker(B, "m"));
            assert_eq!(of("m").as_deref(), Some("2"));

            assert!(registry.remove(&first).is_some());
            assert_eq!(of("m").as_deref(), Some("1"));
            assert!(registry.remove_by_url(B).is_some());
            assert_eq!(of("m").as_deref(), Some("0"));
            // A removal that finds no worker moves nothing.
            assert!(registry.remove(&second).is_none());
            assert_eq!(of("m").as_deref(), Some("0"));
        });
    }

    /// A replacement by a worker of another group moves both groups in one
    /// step: the old group gives the worker up and the new one counts it.
    /// A replacement inside the group moves nothing.
    #[test]
    fn a_replacement_moves_the_gauge_of_the_group_it_leaves_and_the_one_it_joins() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let of = |model: &str| size(&handle, &[&format!("model=\"{model}\"")]);
        metrics::with_local_recorder(&recorder, || {
            let registry = WorkerRegistry::new();
            let id = registry.register(worker(A, "m")).expect("a registers");
            registry.register(worker(B, "m")).expect("b registers");

            // By id, and by the URL's next registration.
            assert!(registry.replace(&id, worker(A, "n")));
            assert_eq!(of("m").as_deref(), Some("1"));
            assert_eq!(of("n").as_deref(), Some("1"));
            assert_eq!(registry.register_or_replace(worker(B, "n")), {
                registry.get_id_by_url(B).expect("b stands")
            });
            assert_eq!(of("m").as_deref(), Some("0"));
            assert_eq!(of("n").as_deref(), Some("2"));

            assert!(registry.replace(&id, worker(A, "n")));
            assert_eq!(of("n").as_deref(), Some("2"));
        });
    }

    /// The group is the worker's type, its connection mode and its primary
    /// model: two workers of one model on two connection modes are two
    /// groups.
    #[test]
    fn a_group_is_kept_by_type_mode_and_primary_model() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let of = |mode: &str| size(&handle, &["model=\"m\"", mode]);
        metrics::with_local_recorder(&recorder, || {
            let registry = WorkerRegistry::new();
            registry.register(worker(A, "m")).expect("a registers");
            let grpc: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new("grpc://b:9000")
                    .model(ModelCard::new("m"))
                    .connection_mode(ConnectionMode::Grpc)
                    .build(),
            );
            registry.register(grpc).expect("b registers");
            assert_eq!(of("connection_mode=\"http\"").as_deref(), Some("1"));
            assert_eq!(of("connection_mode=\"grpc\"").as_deref(), Some("1"));
        });
    }

    /// A worker counts once, under its primary model: its first card. The
    /// model index lists it under its second card too, and that group's
    /// gauge does not count it. A refresh of its cards moves no group, so
    /// it leaves the group it entered.
    #[test]
    fn a_worker_of_two_cards_counts_once_under_its_primary_model() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let of = |model: &str| size(&handle, &[&format!("model=\"{model}\"")]);
        metrics::with_local_recorder(&recorder, || {
            let registry = WorkerRegistry::new();
            let both: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(A)
                    .models(vec![ModelCard::new("m"), ModelCard::new("n")])
                    .build(),
            );
            let id = registry.register(Arc::clone(&both)).expect("a registers");
            registry.register(worker(B, "n")).expect("b registers");
            assert_eq!(registry.get_by_model("n").len(), 2);
            assert_eq!(of("m").as_deref(), Some("1"));
            assert_eq!(of("n").as_deref(), Some("1"));

            let turned = vec![ModelCard::new("n"), ModelCard::new("m")];
            assert!(registry.refresh_models(&both, turned));
            assert_eq!(of("m").as_deref(), Some("1"));
            assert_eq!(of("n").as_deref(), Some("1"));

            assert!(registry.remove(&id).is_some());
            assert_eq!(of("m").as_deref(), Some("0"));
            assert_eq!(of("n").as_deref(), Some("1"));
        });
    }

    /// A leaving that finds no count is a write of the registry that missed
    /// its call. Nothing counts a group again, so a debug build stops there
    /// and the registry's cases catch it.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "left with no count")]
    fn a_leaving_that_finds_no_count_stops_a_debug_build() {
        super::PoolSizes::default().left(worker(A, "m").as_ref());
    }
}
