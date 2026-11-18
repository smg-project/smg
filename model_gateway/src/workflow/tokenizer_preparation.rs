//! Keep a preparing worker's tokenizer alive until its registry commit.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Weak},
};

use llm_tokenizer::TokenizerRegistry;
use parking_lot::{Mutex, MutexGuard};
use tokio::sync::{Semaphore, SemaphorePermit};
use tokio_util::sync::CancellationToken;
use wfaas::{StepId, WorkflowError, WorkflowResult};

use super::{tokenizer_registration::register_tokenizer, TokenizerConfigRequest};
use crate::{
    app_context::AppContext,
    worker::{ModelCard, Worker},
    workflow::steps::local::ReleaseTokenizerStep,
};

/// Tokenizers held by registrations that have not published their workers yet.
#[derive(Debug)]
pub struct TokenizerPreparations {
    scopes: Mutex<HashMap<String, HashSet<String>>>,
    loads: Semaphore,
}

impl Default for TokenizerPreparations {
    fn default() -> Self {
        Self {
            scopes: Mutex::default(),
            loads: Semaphore::new(16),
        }
    }
}

impl TokenizerPreparations {
    pub(crate) async fn load_slot(&self) -> Result<SemaphorePermit<'_>, String> {
        self.loads
            .acquire()
            .await
            .map_err(|error| format!("Tokenizer preparation limiter closed: {error}"))
    }
    pub(crate) fn lock(&self) -> MutexGuard<'_, HashMap<String, HashSet<String>>> {
        self.scopes.lock()
    }
}

/// A cancellable preparation shared by a registration's workflow state.
#[derive(Debug)]
pub struct TokenizerPreparation {
    app_context: Weak<AppContext>,
    id: String,
    cancellation: CancellationToken,
}

impl TokenizerPreparation {
    pub(crate) fn new(app_context: &Arc<AppContext>) -> Arc<Self> {
        let id = TokenizerRegistry::generate_id();
        app_context
            .tokenizer_preparations
            .lock()
            .insert(id.clone(), HashSet::new());
        Arc::new(Self {
            app_context: Arc::downgrade(app_context),
            id,
            cancellation: CancellationToken::new(),
        })
    }

    pub(crate) fn pin(&self, model: &str) -> WorkflowResult<()> {
        let app_context = self.app_context.upgrade().ok_or_else(Self::closed)?;
        let mut scopes = app_context.tokenizer_preparations.lock();
        scopes
            .get_mut(&self.id)
            .ok_or_else(Self::closed)?
            .insert(model.to_owned());
        Ok(())
    }

    pub(crate) async fn cancelled(&self) {
        self.cancellation.cancelled().await;
    }

    pub(crate) fn closed() -> WorkflowError {
        WorkflowError::StepFailed {
            step_id: StepId::new("prepare_tokenizer"),
            message: "Worker preparation was cancelled or has already ended".to_string(),
        }
    }

    /// Serialize publication with cancellation and last-worker tokenizer release.
    /// The callback must be synchronous and must not release a tokenizer.
    pub(crate) fn with_pins<T>(
        &self,
        operation: impl FnOnce(&mut HashSet<String>) -> WorkflowResult<T>,
    ) -> WorkflowResult<T> {
        let app_context = self.app_context.upgrade().ok_or_else(Self::closed)?;
        let mut scopes = app_context.tokenizer_preparations.lock();
        let models = scopes.get_mut(&self.id).ok_or_else(Self::closed)?;
        operation(models)
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    pub(crate) fn commit<T>(
        &self,
        commit: impl FnOnce() -> WorkflowResult<T>,
    ) -> WorkflowResult<T> {
        let app_context = self.app_context.upgrade().ok_or_else(Self::closed)?;
        let result = {
            let scopes = app_context.tokenizer_preparations.lock();
            if !scopes.contains_key(&self.id) {
                return Err(Self::closed());
            }
            if scopes[&self.id]
                .iter()
                .any(|model| app_context.tokenizer_registry.get_by_name(model).is_none())
            {
                return Err(WorkflowError::StepFailed {
                    step_id: StepId::new("prepare_tokenizer"),
                    message: "A tokenizer was removed before worker publication".to_owned(),
                });
            }
            commit()
        };
        self.close();
        result
    }

    pub(crate) fn close(&self) {
        if let Some(app_context) = self.app_context.upgrade() {
            let models = app_context.tokenizer_preparations.lock().remove(&self.id);
            self.cancellation.cancel();
            for model in models.into_iter().flatten() {
                ReleaseTokenizerStep::release_if_unserved(&app_context, &model);
            }
        } else {
            self.cancellation.cancel();
        }
    }

    pub(crate) fn guard(self: &Arc<Self>) -> PreparationGuard {
        PreparationGuard(Arc::clone(self))
    }
}

impl Drop for TokenizerPreparation {
    fn drop(&mut self) {
        self.close();
    }
}

/// End a preparation when its owning job exits, even if workflow state retains it.
pub(crate) struct PreparationGuard(Arc<TokenizerPreparation>);

impl Drop for PreparationGuard {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// Prepare every canonical model of a candidate without publishing its worker.
pub(crate) async fn prepare_worker_tokenizers(
    app_context: &Arc<AppContext>,
    workers: &[Arc<dyn Worker>],
    preparation: &TokenizerPreparation,
) -> WorkflowResult<()> {
    let mut prepared = HashSet::new();
    for worker in workers {
        let cards = if worker.models().is_empty() {
            vec![ModelCard::new(worker.model_id())]
        } else {
            worker.models()
        };
        for card in cards {
            if !prepared.insert(card.id.clone()) {
                continue;
            }
            preparation.pin(&card.id)?;
            if app_context.tokenizer_registry.contains(&card.id) {
                continue;
            }
            let router = &app_context.router_config;
            let source = card
                .tokenizer_path
                .as_deref()
                .filter(|source| !source.is_empty())
                .or_else(|| {
                    if card.id == worker.model_id() {
                        tokenizer_source(
                            &worker.metadata().spec.labels,
                            router.tokenizer_path.as_deref(),
                            router.model_path.as_deref(),
                        )
                    } else {
                        router
                            .tokenizer_path
                            .as_deref()
                            .filter(|source| !source.is_empty())
                            .or(router.model_path.as_deref())
                            .filter(|source| !source.is_empty())
                    }
                })
                .unwrap_or(&card.id);
            let config = TokenizerConfigRequest {
                id: TokenizerRegistry::generate_id(),
                name: card.id.clone(),
                source: source.to_owned(),
                chat_template_path: card
                    .chat_template
                    .clone()
                    .or_else(|| router.chat_template.clone()),
                cache_config: router.tokenizer_cache.to_option(),
                fail_on_duplicate: false,
                for_worker: true,
            };
            tokio::select! {
                biased;
                () = preparation.cancelled() => return Err(TokenizerPreparation::closed()),
                result = register_tokenizer(app_context, &config, workers) => { result?; }
            }
        }
    }
    Ok(())
}

/// The source a model's tokenizer is loaded from, in order (an empty value is
/// no source): the worker's `tokenizer_path` label, its `model_path` label,
/// the router's `--tokenizer-path`, its `--model-path`, and last the worker's
/// `model_id` label. A dialect whose model info names the model by its id
/// only (a path or hub id on the worker's side) still gets a job that way:
/// for a gRPC worker a local load that fails on the id falls back to the
/// worker's own bundle (`fetch_tokenizer_from_worker`); a ZMQ worker's id has
/// to be a path or hub id the gateway itself can load. Coming last, the id
/// never displaces an operator's flag.
pub(crate) fn tokenizer_source<'a>(
    labels: &'a HashMap<String, String>,
    router_tokenizer_path: Option<&'a str>,
    router_model_path: Option<&'a str>,
) -> Option<&'a str> {
    let usable = |path: &&str| !path.is_empty();
    let label = |key: &str| labels.get(key).map(String::as_str).filter(usable);
    label("tokenizer_path")
        .or_else(|| label("model_path"))
        .or_else(|| router_tokenizer_path.filter(usable))
        .or_else(|| router_model_path.filter(usable))
        .or_else(|| label("model_id"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        app_context::testing::app_context_for_tests,
        config::RouterConfig,
        worker::{BasicWorkerBuilder, ConnectionMode, WorkerRegistry},
        workflow::steps::local::test_support::labels_of,
    };

    #[tokio::test]
    async fn secondary_sources_skip_empty_router_paths_and_keep_card_precedence() {
        let directory = tempfile::tempdir().unwrap();
        let tokenizer = directory.path().join("tokenizer.json");
        std::fs::write(
            &tokenizer,
            serde_json::json!({
                "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
                "normalizer":null, "pre_tokenizer":null, "post_processor":null, "decoder":null,
                "model":{"type":"BPE", "vocab":{"hello":0}, "merges":[]}
            })
            .to_string(),
        )
        .unwrap();
        let router_tokenizer = directory.path().join("router-tokenizer.json");
        std::fs::copy(&tokenizer, &router_tokenizer).unwrap();
        for card_path in [None, Some(""), tokenizer.to_str()] {
            let app = app_context_for_tests(
                RouterConfig::builder()
                    .wait_for_tokenizer(true)
                    .tokenizer_path("")
                    .model_path(router_tokenizer.to_str().unwrap())
                    .build_unchecked(),
                Arc::new(WorkerRegistry::new()),
                None,
            );
            // The primary is already prepared; a secondary must use the real
            // router path, or its own nonempty card path, without gRPC fallback.
            let mut secondary = ModelCard::new("secondary");
            secondary.tokenizer_path = card_path.map(str::to_string);
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new("ipc:///fixture")
                    .connection_mode(ConnectionMode::Zmq)
                    .models(vec![ModelCard::new("primary"), secondary])
                    .build(),
            );
            assert_eq!(worker.model_id(), "primary");
            assert_eq!(worker.models().len(), 2);
            app.tokenizer_registry
                .load("primary-tokenizer", "primary", "warm", || async {
                    Ok(Arc::new(llm_tokenizer::mock::MockTokenizer::default())
                        as Arc<dyn llm_tokenizer::traits::Tokenizer>)
                })
                .await
                .unwrap();
            let preparation = TokenizerPreparation::new(&app);
            prepare_worker_tokenizers(&app, &[worker], &preparation)
                .await
                .unwrap();
            assert_eq!(
                app.tokenizer_registry
                    .get_by_name("secondary")
                    .unwrap()
                    .source,
                card_path
                    .filter(|path| !path.is_empty())
                    .unwrap_or(router_tokenizer.to_str().unwrap())
            );
        }
    }

    /// A worker that names its model by id only, with neither a path label
    /// nor a router flag: the id is the source (a local load that fails on
    /// it falls back to the worker's bundle); an empty id is no source.
    #[test]
    fn tokenizer_source_is_the_model_id_when_nothing_else_names_a_path() {
        let id_only = labels_of(&[("model_id", "/engines/served")]);
        assert_eq!(
            tokenizer_source(&id_only, None, None),
            Some("/engines/served")
        );
        assert_eq!(
            tokenizer_source(&id_only, Some(""), Some("")),
            Some("/engines/served")
        );
        assert_eq!(
            tokenizer_source(&labels_of(&[("model_id", "")]), None, None),
            None
        );
        assert_eq!(tokenizer_source(&labels_of(&[]), None, None), None);
    }

    /// The operator's flags and the worker's path labels keep their
    /// precedence over the id: `--tokenizer-path`, then `--model-path`, win
    /// over an id-only worker, and a `tokenizer_path` or `model_path` label
    /// wins over the flags as before.
    #[test]
    fn tokenizer_source_reads_the_model_id_only_after_the_flags_and_path_labels() {
        let id_only = labels_of(&[("model_id", "/engines/served")]);
        assert_eq!(
            tokenizer_source(&id_only, Some("/opt/tok"), None),
            Some("/opt/tok")
        );
        assert_eq!(
            tokenizer_source(&id_only, None, Some("/opt/model")),
            Some("/opt/model")
        );
        assert_eq!(
            tokenizer_source(&id_only, Some("/opt/tok"), Some("/opt/model")),
            Some("/opt/tok")
        );

        let with_paths = labels_of(&[
            ("model_id", "/engines/served"),
            ("model_path", "/engines/weights"),
            ("tokenizer_path", "/engines/tok"),
        ]);
        assert_eq!(
            tokenizer_source(&with_paths, Some("/opt/tok"), None),
            Some("/engines/tok")
        );
        let with_model_path = labels_of(&[
            ("model_id", "/engines/served"),
            ("model_path", "/engines/weights"),
        ]);
        assert_eq!(
            tokenizer_source(&with_model_path, Some("/opt/tok"), None),
            Some("/engines/weights")
        );
    }
}
