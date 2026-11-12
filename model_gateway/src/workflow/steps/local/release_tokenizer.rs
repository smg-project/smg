//! Step to release the tokenizers of the models whose last worker left.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use llm_tokenizer::{registry::TokenizerEntry, LoadOutcome};
use tracing::{debug, info, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::submit_tokenizer_job::tokenizer_source;
use crate::{
    app_context::AppContext,
    worker::{basic::merge_labels, Worker},
    workflow::data::WorkerRemovalWorkflowData,
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
/// its id, with the multimodal config that came with it, when that
/// tokenizer's source is the one a removed worker of the model loads it from
/// (its labels and the router's flags, as at registration), so an operator's
/// registration of another source under the same name stays, and so does the
/// entry of a last worker that names no source of its own. The model's next
/// registration loads the tokenizer again.
/// A model a removed worker only lists keeps its entry: the operator's
/// registration of a listed model stays, and so does a same-source entry a
/// worker whose primary it was loaded earlier, until a worker whose primary
/// it is leaves.
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
        for model_id in Self::candidates(app_context, &removed) {
            if Self::release(app_context, &model_id) {
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
    /// worker of either leaves), served by no worker any more, whose entry's
    /// source is the one a removed worker of the model loads it from.
    fn candidates(app_context: &AppContext, removed: &[&Arc<dyn Worker>]) -> Vec<String> {
        let router = &app_context.router_config;
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
            let loaded_by_a_worker = removed.iter().any(|worker| {
                worker.model_id() == model_id && {
                    let labels = merge_labels(
                        &worker.metadata().discovered_labels,
                        &worker.metadata().spec.labels,
                    );
                    tokenizer_source(
                        &labels,
                        router.tokenizer_path.as_deref(),
                        router.model_path.as_deref(),
                    ) == Some(entry.source.as_str())
                }
            });
            if !loaded_by_a_worker {
                debug!(
                    model = %model_id,
                    source = %entry.source,
                    "Tokenizer of a model no worker serves kept: not the source its workers load from"
                );
                continue;
            }
            candidates.push(model_id);
        }
        candidates
    }

    /// Removes the model's entry and the multimodal config keyed by it, unless
    /// a worker of the model appeared since the candidates were read (its
    /// registration's tokenizer job read the entry as already loaded): then
    /// the entry goes back as it was, under its own id, and the config stays
    /// (see [`Self::put_back`] for the job that loaded afresh instead). The
    /// put-back runs on its own task: its wait on the registry's per-name
    /// lock, which a registration's load of the same name may hold for its
    /// whole load, is not the step's budget to cut, and a step cut with the
    /// removed entry in hand would lose it. Whether the entry left.
    fn release(app_context: &Arc<AppContext>, model_id: &str) -> bool {
        let Some(entry) = app_context.tokenizer_registry.remove(model_id) else {
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
    /// keyed by an id no entry has any more, goes with the removed entry.
    async fn put_back(app_context: Arc<AppContext>, entry: TokenizerEntry) {
        let tokenizer = Arc::clone(&entry.tokenizer);
        match app_context
            .tokenizer_registry
            .load(&entry.id, &entry.name, &entry.source, || async {
                Ok(tokenizer)
            })
            .await
        {
            Ok(LoadOutcome::Loaded { .. }) => {}
            Ok(LoadOutcome::AlreadyExists { id }) if id == entry.id => {}
            Ok(LoadOutcome::AlreadyExists { .. }) => {
                app_context.multimodal_config_registry.remove(&entry.id);
            }
            Err(error) => {
                warn!(
                    model = %entry.name,
                    "A model registered again under the release of its tokenizer could not take it back: {error}"
                );
            }
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

    /// A tokenizer registered under `name` from `source`, as a worker's
    /// registration job or the operator's `POST /v1/tokenizers` leaves it.
    async fn tokenizer(app_context: &AppContext, name: &str, source: &str) {
        app_context
            .tokenizer_registry
            .load(&TokenizerRegistry::generate_id(), name, source, || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();
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

    /// Lets the tasks the step spawned run to their next wait.
    async fn settle() {
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
    }

    /// A load of `name` that holds the registry's per-name lock until `gate`
    /// is notified, then succeeds or fails as told: a registration's
    /// tokenizer job that takes long, on its own task.
    fn competing_load(
        app_context: &Arc<AppContext>,
        name: &str,
        gate: Arc<Notify>,
        succeeds: bool,
    ) -> tokio::task::JoinHandle<Result<LoadOutcome, llm_tokenizer::LoadError>> {
        let app_context = Arc::clone(app_context);
        let name = name.to_string();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test awaits the task it spawns"
        )]
        tokio::spawn(async move {
            app_context
                .tokenizer_registry
                .load(
                    &TokenizerRegistry::generate_id(),
                    &name,
                    "/tok/m",
                    || async move {
                        gate.notified().await;
                        if succeeds {
                            Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
                        } else {
                            Err("the source did not load".to_string())
                        }
                    },
                )
                .await
        })
    }

    #[tokio::test]
    async fn the_tokenizer_leaves_with_the_models_last_worker() {
        let app_context = app_context();
        let first = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        let second = worker(&app_context, "http://w2:8000", "m", "/tok/m");
        tokenizer(&app_context, "m", "/tok/m").await;

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

    #[tokio::test]
    async fn an_operators_tokenizer_of_another_source_stays() {
        let app_context = app_context();
        let only = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        tokenizer(&app_context, "m", "/tok/operator").await;

        remove(&app_context, only).await;
        assert!(
            app_context.tokenizer_registry.contains("m"),
            "not the source the model's workers load from"
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
        tokenizer(&app_context, "a", "/tok/shared").await;
        tokenizer(&app_context, "b", "/tok/shared").await;

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
        tokenizer(&app_context, "m", "/tok/m").await;

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
            tokenizer(&app_context, "m", "/tok/m").await;
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

    /// A last worker that names no source of its own (no path label, no
    /// router flag, no model id label): nothing says the entry was loaded
    /// for it, so it stays.
    #[tokio::test]
    async fn a_last_worker_naming_no_source_keeps_the_entry() {
        let app_context = app_context();
        let only: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .model(ModelCard::new("m"))
                .connection_mode(ConnectionMode::Http)
                .build(),
        );
        app_context.worker_registry.register(Arc::clone(&only));
        tokenizer(&app_context, "m", "/tok/m").await;

        remove(&app_context, only).await;
        assert!(
            app_context.tokenizer_registry.contains("m"),
            "the worker names no source the entry could have come from"
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
        tokenizer(&app_context, "m", "/tok/m").await;
        let id = app_context.tokenizer_registry.get_by_name("m").unwrap().id;
        remove_from_registries(&app_context, Arc::clone(&first)).await;

        let candidates = ReleaseTokenizerStep::candidates(&app_context, &[&first]);
        assert_eq!(candidates, vec!["m".to_string()]);
        // The registration that lands now: its worker, then its tokenizer job.
        worker(&app_context, "http://w2:8000", "m", "/tok/m");
        tokenizer(&app_context, "m", "/tok/m").await;

        assert!(!ReleaseTokenizerStep::release(&app_context, "m"));
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
        tokenizer(&app_context, "m", "/tok/m").await;
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
        tokenizer(&app_context, "m", "/tok/m").await;
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
        tokenizer(&app_context, "m", "/tok/m").await;
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
            vec!["m".to_string()]
        );

        // The release's remove; the registration that lands now (its worker,
        // then its tokenizer job, which finds the name free); the put-back.
        let removed = app_context.tokenizer_registry.remove("m").unwrap();
        worker(&app_context, "http://w2:8000", "m", "/tok/m");
        tokenizer(&app_context, "m", "/tok/m").await;
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
}
