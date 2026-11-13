//! Step to release the tokenizers of the models whose last worker left.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use llm_tokenizer::{registry::TokenizerEntry, LoadOutcome};
use tracing::{debug, info, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{app_context::AppContext, worker::Worker, workflow::data::WorkerRemovalWorkflowData};

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
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
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
    /// as already loaded): then
    /// the entry goes back as it was, under its own id, and the config stays
    /// (see [`Self::put_back`] for the job that loaded afresh instead). The
    /// put-back runs on its own task: its wait on the registry's per-name
    /// lock, which a registration's load of the same name may hold for its
    /// whole load, is not the step's budget to cut, and a step cut with the
    /// removed entry in hand would lose it. Whether the entry left.
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
mod tests {
    use std::sync::Arc;

    use llm_multimodal::PreProcessorConfig;
    use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer, LoadOutcome, TokenizerRegistry};
    use tokio::sync::Notify;
    use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

    use super::ReleaseTokenizerStep;
    use crate::{
        app_context::{testing::app_context_for_tests, AppContext},
        config::RouterConfig,
        routers::grpc::multimodal::MultimodalModelConfig,
        worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, Worker, WorkerRegistry},
        workflow::{
            data::WorkerRemovalWorkflowData,
            steps::local::{
                create_worker_removal_workflow_data, RemoveFromPolicyRegistryStep,
                RemoveFromWorkerRegistryStep,
            },
        },
    };

    fn app_context() -> Arc<AppContext> {
        app_context_for_tests(
            RouterConfig::builder()
                .round_robin_policy()
                .build_unchecked(),
            Arc::new(WorkerRegistry::new()),
            None,
        )
    }

    /// A registered worker of `model` on the gRPC pipeline whose
    /// `tokenizer_path` label names `source`, as a worker whose registration
    /// loaded the model's tokenizer.
    fn worker(app_context: &AppContext, url: &str, model: &str, source: &str) -> Arc<dyn Worker> {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model))
                .label("tokenizer_path", source)
                .connection_mode(ConnectionMode::Grpc)
                .build(),
        );
        app_context.worker_registry.register(Arc::clone(&worker));
        worker
    }

    /// A tokenizer registered under `name` from `source` by no worker's
    /// registration: the startup entry or the operator's `POST /v1/tokenizers`.
    async fn tokenizer(app_context: &AppContext, name: &str, source: &str) {
        app_context
            .tokenizer_registry
            .load(&TokenizerRegistry::generate_id(), name, source, || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();
    }

    /// A tokenizer a worker's registration loaded under `name` from `source`,
    /// recorded as the worker's when the load is a fresh one, as the
    /// registration workflow's load leaves it.
    async fn worker_tokenizer(app_context: &AppContext, name: &str, source: &str) {
        let outcome = app_context
            .tokenizer_registry
            .load(&TokenizerRegistry::generate_id(), name, source, || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();
        if let LoadOutcome::Loaded { id } = outcome {
            app_context.worker_loaded_tokenizers.insert(id);
        }
    }

    /// The removal workflow's registry steps after the drain, in the
    /// workflow's order, with the context they leave for the release step.
    async fn remove_from_registries(
        app_context: &Arc<AppContext>,
        worker: Arc<dyn Worker>,
    ) -> WorkflowContext<WorkerRemovalWorkflowData> {
        let url = worker.url().to_string();
        let mut data =
            create_worker_removal_workflow_data(url.clone(), None, Arc::clone(app_context));
        data.worker_urls = vec![url];
        data.actual_workers_to_remove = Some(vec![worker]);
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        RemoveFromWorkerRegistryStep
            .execute(&mut context)
            .await
            .unwrap();
        RemoveFromPolicyRegistryStep
            .execute(&mut context)
            .await
            .unwrap();
        context
    }

    /// The removal workflow's steps after the drain, the release included.
    async fn remove(app_context: &Arc<AppContext>, worker: Arc<dyn Worker>) {
        let mut context = remove_from_registries(app_context, worker).await;
        ReleaseTokenizerStep.execute(&mut context).await.unwrap();
    }

    /// A registered worker of `model` on the HTTP wire with the same
    /// `tokenizer_path` label as its gRPC peers.
    fn http_worker(
        app_context: &AppContext,
        url: &str,
        model: &str,
        source: &str,
    ) -> Arc<dyn Worker> {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model))
                .label("tokenizer_path", source)
                .connection_mode(ConnectionMode::Http)
                .build(),
        );
        app_context.worker_registry.register(Arc::clone(&worker));
        worker
    }

    /// A registered worker of `model` on the HTTP wire with no label naming
    /// a tokenizer source.
    fn bare_http_worker(app_context: &AppContext, url: &str, model: &str) -> Arc<dyn Worker> {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model))
                .connection_mode(ConnectionMode::Http)
                .build(),
        );
        app_context.worker_registry.register(Arc::clone(&worker));
        worker
    }

    /// Lets the tasks the step spawned run to their next wait.
    async fn settle() {
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
    }

    /// A load of `name` that holds the registry's per-name lock until `gate`
    /// is notified, then succeeds or fails as told: a registration's
    /// tokenizer job that takes long, on its own task, recording the entry
    /// as a worker's where the registration's loader does, under the lock
    /// before the publish.
    fn competing_load(
        app_context: &Arc<AppContext>,
        name: &str,
        gate: Arc<Notify>,
        succeeds: bool,
    ) -> tokio::task::JoinHandle<Result<LoadOutcome, llm_tokenizer::LoadError>> {
        let registry = Arc::clone(&app_context.tokenizer_registry);
        let loaded_by_workers = Arc::clone(&app_context.worker_loaded_tokenizers);
        let name = name.to_string();
        let id = TokenizerRegistry::generate_id();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test awaits the task it spawns"
        )]
        tokio::spawn(async move {
            registry
                .load(&id.clone(), &name, "/tok/m", || async move {
                    gate.notified().await;
                    if succeeds {
                        loaded_by_workers.insert(id);
                        Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
                    } else {
                        Err("the source did not load".to_string())
                    }
                })
                .await
        })
    }

    #[tokio::test]
    async fn the_tokenizer_leaves_with_the_models_last_worker() {
        let app_context = app_context();
        let first = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        let second = worker(&app_context, "http://w2:8000", "m", "/tok/m");
        worker_tokenizer(&app_context, "m", "/tok/m").await;

        remove(&app_context, first).await;
        assert!(
            app_context.tokenizer_registry.contains("m"),
            "another worker serves the model"
        );
        remove(&app_context, second).await;
        assert!(
            !app_context.tokenizer_registry.contains("m"),
            "the model's last worker left"
        );
        assert!(app_context.tokenizer_registry.is_empty());
    }

    /// An operator's registration under a worker's model name: no worker's
    /// registration loaded it, so the model's last worker leaving keeps it.
    #[tokio::test]
    async fn an_operators_tokenizer_under_a_workers_model_stays() {
        let app_context = app_context();
        let only = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        tokenizer(&app_context, "m", "/tok/operator").await;

        remove(&app_context, only).await;
        assert!(
            app_context.tokenizer_registry.contains("m"),
            "no worker's registration loaded it"
        );
    }

    /// The registration step loads under a worker's primary model alone, so
    /// the release asks after the primary alone: an endpoint that lists a
    /// second model takes its primary's tokenizer with it and leaves the
    /// listed model's entry.
    #[tokio::test]
    async fn a_model_a_worker_only_lists_keeps_its_entry() {
        let app_context = app_context();
        let endpoint: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://e:8000")
                .models(vec![ModelCard::new("a"), ModelCard::new("b")])
                .label("tokenizer_path", "/tok/shared")
                .connection_mode(ConnectionMode::Grpc)
                .build(),
        );
        app_context.worker_registry.register(Arc::clone(&endpoint));
        worker_tokenizer(&app_context, "a", "/tok/shared").await;
        worker_tokenizer(&app_context, "b", "/tok/shared").await;

        remove(&app_context, endpoint).await;
        assert!(
            !app_context.tokenizer_registry.contains("a"),
            "no worker serves the primary any more"
        );
        assert!(
            app_context.tokenizer_registry.contains("b"),
            "a listed model's entry is not the endpoint's to release"
        );
    }

    /// With autoload off the gateway manages no tokenizer: every entry is the
    /// operator's and the step skips.
    #[tokio::test]
    async fn autoload_disabled_keeps_the_entry() {
        let app_context = app_context_for_tests(
            RouterConfig::builder()
                .round_robin_policy()
                .disable_tokenizer_autoload(true)
                .build_unchecked(),
            Arc::new(WorkerRegistry::new()),
            None,
        );
        let only = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        worker_tokenizer(&app_context, "m", "/tok/m").await;

        remove(&app_context, only).await;
        assert!(
            app_context.tokenizer_registry.contains("m"),
            "the gateway manages no tokenizer with autoload off"
        );
    }

    /// A model served over both wires: the tokenizer its gRPC worker's
    /// registration loaded stays until the last worker of either wire
    /// leaves, in either order.
    #[tokio::test]
    async fn a_mixed_pools_tokenizer_leaves_with_its_last_worker_of_either_wire() {
        for grpc_first in [true, false] {
            let app_context = app_context();
            let grpc = worker(&app_context, "http://w1:8000", "m", "/tok/m");
            let http = http_worker(&app_context, "http://w2:8000", "m", "/tok/m");
            worker_tokenizer(&app_context, "m", "/tok/m").await;
            let (first, last) = if grpc_first {
                (grpc, http)
            } else {
                (http, grpc)
            };

            remove(&app_context, first).await;
            assert!(
                app_context.tokenizer_registry.contains("m"),
                "a worker of the other wire serves the model (gRPC first: {grpc_first})"
            );
            remove(&app_context, last).await;
            assert!(
                !app_context.tokenizer_registry.contains("m"),
                "the model's last worker left (gRPC first: {grpc_first})"
            );
        }
    }

    /// The startup entry (`--model-path`, registered under the path as its
    /// name) serves an HTTP worker whose model id is that path: no worker's
    /// registration loaded it (an HTTP registration loads none), so the last
    /// such worker leaving keeps it, and the worker added again finds it.
    #[tokio::test]
    async fn the_startup_entry_stays_when_an_http_workers_model_is_the_model_path() {
        let app_context = app_context_for_tests(
            RouterConfig::builder()
                .round_robin_policy()
                .model_path("/models/m")
                .build_unchecked(),
            Arc::new(WorkerRegistry::new()),
            None,
        );
        tokenizer(&app_context, "/models/m", "/models/m").await;
        let only = bare_http_worker(&app_context, "http://w1:8000", "/models/m");

        remove(&app_context, only).await;
        assert!(
            app_context.tokenizer_registry.contains("/models/m"),
            "the startup entry after its last HTTP worker left"
        );

        bare_http_worker(&app_context, "http://w1:8000", "/models/m");
        assert!(
            app_context.tokenizer_registry.get("/models/m").is_some(),
            "the worker added again finds the startup entry"
        );
    }

    /// A registration that lands between the step's read of the candidates
    /// and its remove, its tokenizer job reading the entry as already loaded:
    /// the entry goes back as it was, under its own id, and the model's
    /// requests find it.
    #[tokio::test]
    async fn a_registration_under_the_release_keeps_the_entry() {
        let app_context = app_context();
        let first = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        worker_tokenizer(&app_context, "m", "/tok/m").await;
        let id = app_context.tokenizer_registry.get_by_name("m").unwrap().id;
        remove_from_registries(&app_context, Arc::clone(&first)).await;

        let candidates = ReleaseTokenizerStep::candidates(&app_context, &[&first]);
        assert_eq!(candidates, vec![("m".to_string(), id.clone())]);
        // The registration that lands now: its worker, then its tokenizer job.
        worker(&app_context, "http://w2:8000", "m", "/tok/m");
        worker_tokenizer(&app_context, "m", "/tok/m").await;

        assert!(!ReleaseTokenizerStep::release(&app_context, "m", &id));
        settle().await;
        let entry = app_context.tokenizer_registry.get_by_name("m").unwrap();
        assert_eq!(entry.id, id, "the entry went back as it was");
    }

    /// A registration whose tokenizer job takes the name's lock between the
    /// release's remove and its put-back and holds it past the step's
    /// budget: the put-back, on its own task, waits the load out and yields
    /// to it (one entry under the name, the removed entry's config gone).
    #[tokio::test]
    async fn a_put_back_waits_out_a_competing_load_that_succeeds() {
        let app_context = app_context();
        let first = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        worker_tokenizer(&app_context, "m", "/tok/m").await;
        let old = app_context.tokenizer_registry.get_by_name("m").unwrap().id;
        app_context.multimodal_config_registry.insert(
            old.clone(),
            Arc::new(MultimodalModelConfig::new(
                serde_json::json!({}),
                PreProcessorConfig::default(),
                None,
            )),
        );
        remove_from_registries(&app_context, Arc::clone(&first)).await;

        let removed = app_context.tokenizer_registry.remove("m").unwrap();
        worker(&app_context, "http://w2:8000", "m", "/tok/m");
        let gate = Arc::new(Notify::new());
        let competing = competing_load(&app_context, "m", Arc::clone(&gate), true);
        settle().await;
        #[expect(
            clippy::disallowed_methods,
            reason = "the test awaits the task it spawns"
        )]
        let put_back = tokio::spawn(ReleaseTokenizerStep::put_back(
            Arc::clone(&app_context),
            removed,
        ));
        settle().await;
        assert!(
            !put_back.is_finished(),
            "the put-back waits on the lock the competing load holds"
        );

        gate.notify_one();
        let outcome = competing.await.unwrap().unwrap();
        let LoadOutcome::Loaded { id: fresh } = outcome else {
            panic!("the competing load loaded afresh, not {outcome:?}");
        };
        put_back.await.unwrap();
        let entry = app_context.tokenizer_registry.get_by_name("m").unwrap();
        assert_eq!(entry.id, fresh, "the fresh entry stays");
        assert_eq!(app_context.tokenizer_registry.len(), 1, "no orphan entry");
        assert!(
            app_context.multimodal_config_registry.get(&old).is_none(),
            "the removed entry's config went with it"
        );
    }

    /// The same competing load, failing: the put-back then takes the lock
    /// and the model keeps its previous tokenizer, under its own id.
    #[tokio::test]
    async fn a_put_back_waits_out_a_competing_load_that_fails() {
        let app_context = app_context();
        let first = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        worker_tokenizer(&app_context, "m", "/tok/m").await;
        let old = app_context.tokenizer_registry.get_by_name("m").unwrap().id;
        remove_from_registries(&app_context, Arc::clone(&first)).await;

        let removed = app_context.tokenizer_registry.remove("m").unwrap();
        worker(&app_context, "http://w2:8000", "m", "/tok/m");
        let gate = Arc::new(Notify::new());
        let competing = competing_load(&app_context, "m", Arc::clone(&gate), false);
        settle().await;
        #[expect(
            clippy::disallowed_methods,
            reason = "the test awaits the task it spawns"
        )]
        let put_back = tokio::spawn(ReleaseTokenizerStep::put_back(
            Arc::clone(&app_context),
            removed,
        ));
        settle().await;
        assert!(!put_back.is_finished());

        gate.notify_one();
        assert!(
            competing.await.unwrap().is_err(),
            "the competing load failed"
        );
        put_back.await.unwrap();
        let entry = app_context.tokenizer_registry.get_by_name("m").unwrap();
        assert_eq!(entry.id, old, "the previous tokenizer is back under its id");
        assert_eq!(app_context.tokenizer_registry.len(), 1);
    }

    /// A registration whose tokenizer job runs between the release's remove
    /// and its put-back: the name is free, so the job loads a fresh entry
    /// under its own id; the put-back leaves that entry and takes the
    /// removed entry's multimodal config with the removed entry.
    #[tokio::test]
    async fn a_fresh_load_under_the_put_back_drops_the_removed_entry_config() {
        let app_context = app_context();
        let first = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        worker_tokenizer(&app_context, "m", "/tok/m").await;
        let old = app_context.tokenizer_registry.get_by_name("m").unwrap().id;
        app_context.multimodal_config_registry.insert(
            old.clone(),
            Arc::new(MultimodalModelConfig::new(
                serde_json::json!({}),
                PreProcessorConfig::default(),
                None,
            )),
        );
        remove_from_registries(&app_context, Arc::clone(&first)).await;
        assert_eq!(
            ReleaseTokenizerStep::candidates(&app_context, &[&first]),
            vec![("m".to_string(), old.clone())]
        );

        // The release's remove; the registration that lands now (its worker,
        // then its tokenizer job, which finds the name free); the put-back.
        let removed = app_context.tokenizer_registry.remove("m").unwrap();
        worker(&app_context, "http://w2:8000", "m", "/tok/m");
        worker_tokenizer(&app_context, "m", "/tok/m").await;
        let fresh = app_context.tokenizer_registry.get_by_name("m").unwrap().id;
        assert_ne!(fresh, old);
        ReleaseTokenizerStep::put_back(Arc::clone(&app_context), removed).await;

        let entry = app_context.tokenizer_registry.get_by_name("m").unwrap();
        assert_eq!(entry.id, fresh, "the fresh entry stays");
        assert!(
            app_context.multimodal_config_registry.get(&old).is_none(),
            "the removed entry's config went with it"
        );
    }

    /// The worker that kept the model served leaves while the put-back waits
    /// on the per-name lock (that removal's release finds no entry to take):
    /// the put-back looks again once its load returns, and the registry ends
    /// empty, the removed entry's config with it.
    #[tokio::test]
    async fn the_replacement_leaving_under_the_put_back_empties_the_registry() {
        let app_context = app_context();
        let first = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        worker_tokenizer(&app_context, "m", "/tok/m").await;
        let id = app_context.tokenizer_registry.get_by_name("m").unwrap().id;
        app_context.multimodal_config_registry.insert(
            id.clone(),
            Arc::new(MultimodalModelConfig::new(
                serde_json::json!({}),
                PreProcessorConfig::default(),
                None,
            )),
        );
        remove_from_registries(&app_context, Arc::clone(&first)).await;

        // The release's remove; the replacement's registration, its tokenizer
        // job holding the per-name lock; the put-back, blocked on that lock.
        let removed = app_context.tokenizer_registry.remove("m").unwrap();
        let second = worker(&app_context, "http://w2:8000", "m", "/tok/m");
        let gate = Arc::new(Notify::new());
        let competing = competing_load(&app_context, "m", Arc::clone(&gate), false);
        settle().await;
        #[expect(
            clippy::disallowed_methods,
            reason = "the test awaits the task it spawns"
        )]
        let put_back = tokio::spawn(ReleaseTokenizerStep::put_back(
            Arc::clone(&app_context),
            removed,
        ));
        settle().await;
        assert!(!put_back.is_finished());

        // The replacement leaves while the put-back waits: its removal's
        // release finds no entry under the name.
        remove(&app_context, second).await;
        assert!(!app_context.worker_registry.contains_model("m"));
        gate.notify_one();
        assert!(competing.await.unwrap().is_err());
        put_back.await.unwrap();

        assert!(
            app_context.tokenizer_registry.is_empty(),
            "the put-back's second look released the entry of a model no worker serves"
        );
        assert!(
            app_context.multimodal_config_registry.get(&id).is_none(),
            "the removed entry's config went with it"
        );
        assert!(!app_context.worker_loaded_tokenizers.contains(&id));
    }

    /// The same leave while the replacement's own load, which succeeds, holds
    /// the lock: the fresh entry that load made is a worker's too, and the
    /// put-back's second look takes it with the removed entry's config.
    #[tokio::test]
    async fn the_replacement_leaving_under_a_fresh_load_empties_the_registry() {
        let app_context = app_context();
        let first = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        worker_tokenizer(&app_context, "m", "/tok/m").await;
        let old = app_context.tokenizer_registry.get_by_name("m").unwrap().id;
        app_context.multimodal_config_registry.insert(
            old.clone(),
            Arc::new(MultimodalModelConfig::new(
                serde_json::json!({}),
                PreProcessorConfig::default(),
                None,
            )),
        );
        remove_from_registries(&app_context, Arc::clone(&first)).await;

        let removed = app_context.tokenizer_registry.remove("m").unwrap();
        let second = worker(&app_context, "http://w2:8000", "m", "/tok/m");
        let gate = Arc::new(Notify::new());
        let competing = competing_load(&app_context, "m", Arc::clone(&gate), true);
        settle().await;
        #[expect(
            clippy::disallowed_methods,
            reason = "the test awaits the task it spawns"
        )]
        let put_back = tokio::spawn(ReleaseTokenizerStep::put_back(
            Arc::clone(&app_context),
            removed,
        ));
        settle().await;
        remove(&app_context, second).await;
        gate.notify_one();
        let fresh = competing.await.unwrap().unwrap().id().to_string();
        put_back.await.unwrap();

        assert_ne!(fresh, old);
        assert!(
            app_context.tokenizer_registry.is_empty(),
            "the fresh entry of a model no worker serves left with the put-back's second look"
        );
        assert!(app_context.multimodal_config_registry.get(&old).is_none());
        assert!(!app_context.worker_loaded_tokenizers.contains(&old));
        assert!(!app_context.worker_loaded_tokenizers.contains(&fresh));
    }

    /// The model's last worker leaving while its own registration's load is
    /// in flight: that removal's release finds no entry to take, the load
    /// then publishes one, recorded, and the look after the publish (the
    /// registration step's) releases it.
    #[tokio::test]
    async fn the_last_worker_leaving_under_its_own_load_empties_the_registry() {
        let app_context = app_context();
        let only = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        let gate = Arc::new(Notify::new());
        let load = competing_load(&app_context, "m", Arc::clone(&gate), true);
        settle().await;
        remove(&app_context, only).await;
        assert!(
            app_context.tokenizer_registry.get_by_name("m").is_none(),
            "the removal found no entry to take"
        );

        gate.notify_one();
        let LoadOutcome::Loaded { id } = load.await.unwrap().unwrap() else {
            panic!("the load published afresh");
        };
        assert!(app_context.worker_loaded_tokenizers.contains(&id));
        ReleaseTokenizerStep::release_if_unserved(&app_context, "m");
        assert_eq!(app_context.tokenizer_registry.len(), 0, "the entry left");
        assert!(!app_context.worker_loaded_tokenizers.contains(&id));
    }

    /// A replacement registered under the model's name between the release's
    /// look and its remove (a competing removal took the worker's entry and
    /// its record, then an operator registered one of their own): the
    /// release takes the entry it inspected, by name and id, and the
    /// replacement stays.
    #[tokio::test]
    async fn a_replacement_registered_between_the_look_and_the_release_stays() {
        let app_context = app_context();
        let only = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        worker_tokenizer(&app_context, "m", "/tok/m").await;
        let id = app_context.tokenizer_registry.get_by_name("m").unwrap().id;
        remove_from_registries(&app_context, Arc::clone(&only)).await;
        let candidates = ReleaseTokenizerStep::candidates(&app_context, &[&only]);
        assert_eq!(candidates, vec![("m".to_string(), id.clone())]);

        // The competing removal's release takes the worker's entry and its
        // record; an operator registers a replacement under the name.
        app_context
            .tokenizer_registry
            .remove_if_id("m", &id)
            .unwrap();
        app_context.worker_loaded_tokenizers.remove(&id);
        tokenizer(&app_context, "m", "/tok/operator").await;
        let replacement = app_context.tokenizer_registry.get_by_name("m").unwrap().id;

        assert!(!ReleaseTokenizerStep::release(&app_context, "m", &id));
        let entry = app_context.tokenizer_registry.get_by_name("m").unwrap();
        assert_eq!(entry.id, replacement, "the operator's replacement stays");
        assert!(!app_context.worker_loaded_tokenizers.contains(&replacement));
    }

    /// Two removals read the same entry (the model's last two workers leaving
    /// together); the first's release took it and, a worker having appeared,
    /// its put-back loads the entry again under its own id. The second's
    /// release, with the id it read, takes nothing while that load runs: it
    /// removes through the name's publication (`TokenizerRegistry::
    /// remove_if_id`), so an entry stored under the id and not yet published
    /// under the name is not taken either, and the put-back's entry is there,
    /// name and id agreeing, once it publishes.
    #[tokio::test]
    async fn a_release_under_a_put_backs_load_of_the_same_entry_takes_nothing() {
        let app_context = app_context();
        let only = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        worker_tokenizer(&app_context, "m", "/tok/m").await;
        let id = app_context.tokenizer_registry.get_by_name("m").unwrap().id;
        remove_from_registries(&app_context, Arc::clone(&only)).await;
        let candidates = ReleaseTokenizerStep::candidates(&app_context, &[&only]);
        assert_eq!(candidates, vec![("m".to_string(), id.clone())]);

        // The first release took the entry; a worker registered; its put-back
        // loads the entry again, under its own id, and takes long about it.
        let removed = app_context
            .tokenizer_registry
            .remove_if_id("m", &id)
            .unwrap();
        worker(&app_context, "http://w2:8000", "m", "/tok/m");
        let gate = Arc::new(Notify::new());
        let registry = Arc::clone(&app_context.tokenizer_registry);
        let held = Arc::clone(&gate);
        #[expect(
            clippy::disallowed_methods,
            reason = "the test awaits the task it spawns"
        )]
        let put_back = tokio::spawn(async move {
            let tokenizer = Arc::clone(&removed.tokenizer);
            registry
                .load(&removed.id, &removed.name, &removed.source, || async move {
                    held.notified().await;
                    Ok(tokenizer)
                })
                .await
        });
        settle().await;

        assert!(!ReleaseTokenizerStep::release(&app_context, "m", &id));
        gate.notify_one();
        assert!(put_back.await.unwrap().unwrap().is_newly_loaded());
        let entry = app_context.tokenizer_registry.get_by_name("m").unwrap();
        assert_eq!(
            entry.id, id,
            "the put-back's entry is there under its own id"
        );
        assert_eq!(app_context.tokenizer_registry.len(), 1);
    }
}
