pub(crate) mod scoring;

mod preparation;
mod request_building;
mod response_processing;

pub(crate) use preparation::DecisionsPreparationStage;
pub(crate) use request_building::DecisionsRequestBuildingStage;
pub(crate) use response_processing::DecisionsResponseProcessingStage;

use crate::worker::{ConnectionMode, RuntimeType, Worker, WorkerType};

pub(crate) fn supports_worker(worker: &dyn Worker) -> bool {
    *worker.connection_mode() == ConnectionMode::Grpc
        && worker.metadata().spec.runtime_type == RuntimeType::Sglang
        && worker.worker_type() == &WorkerType::Regular
}
