//! Dispatch metadata: per-attempt response metadata derived from the stamped
//! plan and the attempt's worker selection.

use std::time::{SystemTime, UNIX_EPOCH};

use smg_rl::RlState;

use crate::routers::grpc::context::{DispatchMetadata, ExecutionPlan, WorkerSelection};

/// Metadata for one dispatch attempt. `dispatch_model` was captured from the
/// request at the build boundary (already canonical); the weight version
/// comes from the attempt's selected worker.
///
/// `rl` is `None` when the control plane is off, which leaves the label-only
/// behavior the gateway had before it.
pub(crate) fn prepare_dispatch_metadata(
    plan: &ExecutionPlan,
    dispatch_model: &str,
    workers: Option<&WorkerSelection>,
    rl: Option<&RlState>,
) -> DispatchMetadata {
    let worker = workers.map(|w| match w {
        WorkerSelection::Single { worker } => worker,
        WorkerSelection::Disaggregated { decode, .. } => decode,
    });
    // The RL table is the live view of an engine's version; the registration
    // label is what it reported when it joined.
    let weight_version = worker
        .and_then(|w| {
            rl.and_then(|rl| rl.table().version_of(w.base_url()))
                .map(|v| v.as_str().to_string())
                .or_else(|| w.metadata().spec.labels.get("weight_version").cloned())
        })
        .unwrap_or_else(|| "default".to_string());

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    DispatchMetadata {
        request_id: plan.request_id().to_string(),
        model: dispatch_model.to_string(),
        created,
        weight_version: Some(weight_version),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
    use smg_rl::{RlConfig, Version, VersionSource};

    use super::*;
    use crate::{
        rl_adapter::RegistryRlView,
        routers::grpc::context::ExecutionPlanKind,
        worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerRegistry},
    };

    const MODEL: &str = "dispatch-metadata-model";
    const URL: &str = "grpc://127.0.0.1:39001";

    /// A plan whose only role here is to carry the request id.
    fn plan() -> ExecutionPlan {
        ExecutionPlan::Batch {
            kind: ExecutionPlanKind::Single,
            shared_request_id: "req-1".to_string(),
            requests: vec![],
        }
    }

    /// One regular gRPC worker, optionally carrying a `weight_version` label.
    fn worker(label: Option<&str>) -> WorkerSelection {
        let mut builder = BasicWorkerBuilder::new(URL)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            });
        if let Some(label) = label {
            builder = builder.label("weight_version", label);
        }
        WorkerSelection::Single {
            worker: Arc::new(builder.build()) as Arc<dyn Worker>,
        }
    }

    fn rl_state() -> Arc<RlState> {
        let registry = Arc::new(WorkerRegistry::new());
        Arc::new(RlState::new(
            Arc::new(RegistryRlView::new(registry)),
            RlConfig::default(),
        ))
    }

    #[test]
    fn no_worker_reports_the_default_sentinel() {
        let metadata = prepare_dispatch_metadata(&plan(), MODEL, None, None);
        assert_eq!(metadata.weight_version.as_deref(), Some("default"));
        assert_eq!(metadata.request_id, "req-1");
        assert_eq!(metadata.model, MODEL);
    }

    #[test]
    fn without_the_control_plane_the_registration_label_is_the_version() {
        let metadata = prepare_dispatch_metadata(&plan(), MODEL, Some(&worker(Some("7"))), None);
        assert_eq!(metadata.weight_version.as_deref(), Some("7"));
    }

    #[test]
    fn an_unlabeled_worker_reports_the_default_sentinel() {
        let metadata = prepare_dispatch_metadata(&plan(), MODEL, Some(&worker(None)), None);
        assert_eq!(metadata.weight_version.as_deref(), Some("default"));
    }

    #[test]
    fn the_table_version_wins_over_the_registration_label() {
        let rl = rl_state();
        rl.table()
            .set_version(URL, MODEL, Version::parse("11"), VersionSource::Api);
        let metadata =
            prepare_dispatch_metadata(&plan(), MODEL, Some(&worker(Some("7"))), Some(&rl));
        assert_eq!(metadata.weight_version.as_deref(), Some("11"));
    }

    #[test]
    fn an_engine_the_table_has_no_version_for_falls_back_to_the_label() {
        let rl = rl_state();
        let metadata =
            prepare_dispatch_metadata(&plan(), MODEL, Some(&worker(Some("7"))), Some(&rl));
        assert_eq!(metadata.weight_version.as_deref(), Some("7"));
    }
}
