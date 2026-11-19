//! The DP minimum-tokens policy: for an engine that takes an external DP
//! dispatch, the rank with the lowest token load. The pick adds the
//! request's estimated tokens to that rank's cached load, so the picks
//! before the next load report count what was already sent its way.

use std::sync::Arc;

use smg_worker::{identity::basic::Worker, load::monitor::WorkerLoadManager};

use super::DPRankLoadPolicy;

#[derive(Debug)]
pub struct MinimumTokensPolicy {
    worker_load_manager: Option<Arc<WorkerLoadManager>>,
}

impl MinimumTokensPolicy {
    pub fn new(worker_load_manager: Option<Arc<WorkerLoadManager>>) -> Self {
        Self {
            worker_load_manager,
        }
    }
}

impl DPRankLoadPolicy for MinimumTokensPolicy {
    fn select_dp_rank(&self, worker: &dyn Worker, estimated_cost: isize) -> Option<isize> {
        if let Some(worker_load) = self.worker_load_manager.as_ref() {
            return worker_load.select_and_increment_lowest_dp_load(worker, estimated_cost);
        }
        None
    }
}
