//! Step to release the tokenizers of the models whose last worker left.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use llm_tokenizer::{registry::TokenizerEntry, LoadOutcome};
use smg_worker::identity::basic::Worker;
use tracing::{debug, info, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    app_context::AppContext,
    workflow::data::{require_app_context, WorkerRemovalWorkflowData},
};

/// Step to release the tokenizer of a model no worker serves any more.
///
/// A worker's registration loads a tokenizer under its model id
/// (`SubmitTokenizerJobStep`) and nothing unloaded it: every retired model
/// id kept its tokenizer resident, tens of megabytes each, so model-id churn
/// grew the registry and the process without bound. Once the removed workers
/// have left the worker registry, the primary model of every removed worker
/// is asked after, whatever its wire: the registration step loads under the
/// primary alone and for the gRPC pipeline alone, but a model served over
/// both wires keeps its tokenizer until its last worker of either wire
/// leaves (with autoload off the gateway manages none and the step skips).
/// A model no worker serves any more loses the tokenizer registered under
/// its id, with the multimodal config that came with it, when a worker's
/// registration loaded that entry (the registration workflow's loader records
/// the entries it loads afresh for a worker, `AppContext::worker_loaded_tokenizers`,
/// under the registry's per-name lock before the entry is published; a load
/// that published after the model's last worker left releases its own entry,
/// see `release_if_unserved`): the startup entry and an operator's
/// registration under a model's name stay. The model's next registration
/// loads the tokenizer again. A model a removed worker only lists keeps its
/// entry until a worker whose primary it is leaves.
pub struct ReleaseTokenizerStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for ReleaseTokenizerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = require_app_context(context.data.host.as_ref())?;
        // The gateway manages no tokenizer when autoload is off: every entry
        // is the operator's, as the registration step leaves them.
        if app_context.router_config.disable_tokenizer_autoload {
            return Ok(StepResult::Skip);
        }
        // The workers found and the ones the registry step removed, since a
        // same-URL replacement during the drain is the one removed.
        let removed: Vec<&Arc<dyn Worker>> = context
            .data
            .actual_workers_to_remove
            .iter()
            .flatten()
            .chain(&context.data.removed_workers)
            .collect();
        let mut released = 0usize;
        for (model_id, entry_id) in Self::candidates(app_context, &removed) {
            if Self::release(app_context, &model_id, &entry_id) {
                released += 1;
            }
        }
        debug!(
            "Released {} tokenizer(s) of the removed workers' models",
            released
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

impl ReleaseTokenizerStep {
    /// The models whose tokenizer is to leave: the primary model of a removed
    /// worker of either wire (the registration step loads under the primary
    /// alone; a model served over both wires keeps its entry until its last
    /// worker of either leaves), served by no worker any more, whose entry a
    /// worker's registration loaded; with that entry's id, the one the
    /// release takes (an entry that replaces it under the name is not).
    fn candidates(app_context: &AppContext, removed: &[&Arc<dyn Worker>]) -> Vec<(String, String)> {
        let mut asked: HashSet<String> = HashSet::new();
        let mut candidates = Vec::new();
        for worker in removed {
            let model_id = worker.model_id().to_string();
            if !asked.insert(model_id.clone())
                || app_context.worker_registry.contains_model(&model_id)
            {
                continue;
            }
            let Some(entry) = app_context.tokenizer_registry.get_by_name(&model_id) else {
                continue;
            };
            if !app_context.worker_loaded_tokenizers.contains(&entry.id) {
                debug!(
                    model = %model_id,
                    tokenizer_id = %entry.id,
                    "Tokenizer of a model no worker serves kept: no worker's registration loaded it"
                );
                continue;
            }
            candidates.push((model_id, entry.id));
        }
        candidates
    }

    /// Removes the entry inspected under the model's name, by that name and
    /// its id in one step (`TokenizerRegistry::remove_if_id`: an entry that
    /// replaced it under the name since the look, an operator's say, is not
    /// taken, nor one a load has stored under this very id and not yet
    /// published under the name, a put-back's when two removals read the
    /// same entry and the first's release took it), and the multimodal
    /// config keyed by it, unless a worker of the model appeared since the
    /// candidates were read (its registration's tokenizer job read the entry
    /// as already loaded): then the entry goes back as it was, under its own
    /// id, and the config stays (see [`Self::put_back`] for the job that
    /// loaded afresh instead). The put-back runs on its own task: its wait
    /// on the registry's per-name lock, which a registration's load of the
    /// same name may hold for its whole load, is not the step's budget to
    /// cut, and a step cut with the removed entry in hand would lose it.
    /// Whether the entry left.
    fn release(app_context: &Arc<AppContext>, model_id: &str, entry_id: &str) -> bool {
        let Some(entry) = app_context
            .tokenizer_registry
            .remove_if_id(model_id, entry_id)
        else {
            return false;
        };
        if app_context.worker_registry.contains_model(model_id) {
            #[expect(
                clippy::disallowed_methods,
                reason = "a put-back of a tokenizer a racing registration still needs; nothing waits for it at shutdown, and one cut short is redone by the next start's registration"
            )]
            tokio::spawn(Self::put_back(Arc::clone(app_context), entry));
            return false;
        }
        app_context.multimodal_config_registry.remove(&entry.id);
        app_context.worker_loaded_tokenizers.remove(&entry.id);
        info!(
            model = %model_id,
            tokenizer_id = %entry.id,
            source = %entry.source,
            "Released the tokenizer of a model no worker serves"
        );
        true
    }

    /// Gives a removed entry back, under its own id, to a model a worker
    /// serves again. When the registration's tokenizer job ran between the
    /// remove and this (the name was free, so it loaded a fresh entry under
    /// its own id), that entry stays and the removed one's multimodal config,
    /// keyed by an id no entry has any more, goes with the removed entry. Once
    /// the load returns the put-back looks again: the worker that kept the
    /// model served may have left while it waited (see
    /// [`Self::release_if_unserved`]).
    async fn put_back(app_context: Arc<AppContext>, entry: TokenizerEntry) {
        let tokenizer = Arc::clone(&entry.tokenizer);
        match app_context
            .tokenizer_registry
            .load(&entry.id, &entry.name, &entry.source, || async {
                Ok(tokenizer)
            })
            .await
        {
            Ok(LoadOutcome::Loaded { .. }) => {
                Self::release_if_unserved(&app_context, &entry.name);
            }
            Ok(LoadOutcome::AlreadyExists { id }) if id == entry.id => {
                Self::release_if_unserved(&app_context, &entry.name);
            }
            Ok(LoadOutcome::AlreadyExists { .. }) => {
                app_context.multimodal_config_registry.remove(&entry.id);
                app_context.worker_loaded_tokenizers.remove(&entry.id);
                Self::release_if_unserved(&app_context, &entry.name);
            }
            Err(error) => {
                warn!(
                    model = %entry.name,
                    "A model registered again under the release of its tokenizer could not take it back: {error}"
                );
            }
        }
    }

    /// The second look of a put-back, and of a worker's load, once it has
    /// published: the worker that kept the model served may have left while
    /// the put-back waited on the registry's per-name lock, or the model's
    /// last worker while its registration's load ran (that removal's release
    /// found no entry to take), so a model no worker serves any more loses
    /// the entry under its name again, config and record with it, when a
    /// worker's registration loaded it: that entry and no replacement (the
    /// release takes it by name and id). The per-name lock orders the insert
    /// before any later removal's remove, so this look closes the window.
    pub(crate) fn release_if_unserved(app_context: &Arc<AppContext>, model_id: &str) {
        if app_context.worker_registry.contains_model(model_id) {
            return;
        }
        let Some(entry) = app_context.tokenizer_registry.get_by_name(model_id) else {
            return;
        };
        if !app_context.worker_loaded_tokenizers.contains(&entry.id) {
            return;
        }
        if Self::release(app_context, model_id, &entry.id) {
            debug!(
                model = %model_id,
                "Released the tokenizer of a model whose last worker left while it was put back or loaded"
            );
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/workflow/steps/local/release_tokenizer/tests.rs"]
mod tests;
