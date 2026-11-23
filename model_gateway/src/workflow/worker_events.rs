//! What a registration or removal workflow's steps do, done on the registry's
//! events for a worker with no workflow here: one imported from a mesh peer.

use std::sync::Arc;

use smg_worker::{
    identity::basic::Worker,
    registry::{event::WorkerEvent, WorkerId, WorkerOrigin},
};
use tokio::sync::broadcast::error::RecvError;

use super::steps::local::{submit_tokenizer_job::submit_tokenizer_job, ReleaseTokenizerStep};
use crate::app_context::AppContext;

/// Spawns the task that follows the registry's events. It walks the registry
/// at its start and after a lag, when an event had no reader.
pub(crate) fn spawn_worker_event_reactions(app_context: &Arc<AppContext>) {
    // With autoload off the gateway manages no tokenizer: nothing to follow.
    if app_context.router_config.disable_tokenizer_autoload {
        return;
    }
    let mut events = app_context.worker_registry.subscribe_events();
    let app_context = Arc::downgrade(app_context);
    #[expect(
        clippy::disallowed_methods,
        reason = "the task ends with the registry; nothing waits at shutdown for a job it has not submitted"
    )]
    tokio::spawn(async move {
        let mut event = None;
        loop {
            let Some(context) = app_context.upgrade() else {
                break;
            };
            match event {
                None => {
                    for (worker_id, worker) in context.worker_registry.get_all_with_ids() {
                        load_tokenizer_of_import(&context, &worker_id, &worker).await;
                    }
                    // A removal that had no reader left its model's tokenizer.
                    for entry in context.tokenizer_registry.list() {
                        ReleaseTokenizerStep::release_if_unserved(&context, &entry.name);
                    }
                }
                Some(WorkerEvent::Registered { worker_id, worker }) => {
                    load_tokenizer_of_import(&context, &worker_id, &worker).await;
                }
                Some(WorkerEvent::Replaced {
                    worker_id,
                    old,
                    new,
                }) => {
                    load_tokenizer_of_import(&context, &worker_id, &new).await;
                    ReleaseTokenizerStep::release_if_unserved(&context, old.model_id());
                }
                // A peer's tombstone runs no removal workflow, and the origin
                // went with the worker: every removal asks as that workflow does.
                Some(WorkerEvent::Removed { worker, .. }) => {
                    ReleaseTokenizerStep::release_if_unserved(&context, worker.model_id());
                }
                Some(WorkerEvent::StatusChanged { .. }) => {}
            }
            drop(context);
            event = match events.recv().await {
                Ok(event) => Some(event),
                Err(RecvError::Lagged(_)) => None,
                Err(RecvError::Closed) => break,
            };
        }
    });
}

/// Submits the tokenizer job a local registration's step would. A worker of
/// local origin is left to that step, which runs right after its event.
async fn load_tokenizer_of_import(
    context: &AppContext,
    worker_id: &WorkerId,
    worker: &Arc<dyn Worker>,
) {
    let model_id = worker.model_id();
    if !worker.connection_mode().uses_grpc_pipeline()
        || context.worker_registry.origin_of(worker_id) != Some(WorkerOrigin::Mesh)
        || context.tokenizer_registry.contains(model_id)
    {
        return;
    }
    let Some(job_queue) = context.worker_job_queue.get() else {
        return;
    };
    let spec = &worker.metadata().spec;
    // The card's template is a path on the owner's node: it is the model's own
    // where the file stands here too; where not, this node's template is first.
    let card = spec
        .models
        .primary()
        .and_then(|card| card.chat_template.clone());
    let chat_template = match card {
        Some(path) if tokio::fs::try_exists(&path).await.unwrap_or(false) => Some(path),
        card => context.router_config.chat_template.clone().or(card),
    };
    submit_tokenizer_job(context, job_queue, model_id, &spec.labels, chat_template).await;
}
