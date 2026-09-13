//! Step to update worker properties.

use std::sync::Arc;

use async_trait::async_trait;
use openai_protocol::worker::WorkerStatus;
use tracing::{debug, info};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    worker::{
        overload::OverloadThresholds, BasicWorker, BasicWorkerBuilder, ConnectionMode, Worker,
    },
    workflow::data::WorkerUpdateWorkflowData,
};

/// Step to update worker properties.
///
/// This step creates new worker instances with updated properties and
/// re-registers them to replace the old workers in the registry.
pub struct UpdateWorkerPropertiesStep;

#[async_trait]
impl StepExecutor<WorkerUpdateWorkflowData> for UpdateWorkerPropertiesStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerUpdateWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let request = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();
        let workers_to_update = context
            .data
            .workers_to_update
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers_to_update".to_string()))?
            .clone();

        debug!(
            "Updating properties for {} worker(s)",
            workers_to_update.len()
        );

        let mut updated_workers: Vec<Arc<dyn Worker>> = Vec::with_capacity(workers_to_update.len());

        for worker in &workers_to_update {
            // Build updated labels - merge new labels into existing ones
            let mut updated_labels = worker.metadata().spec.labels.clone();
            if let Some(ref new_labels) = request.labels {
                for (key, value) in new_labels {
                    updated_labels.insert(key.clone(), value.clone());
                }
            }

            // Resolve priority and cost: use update value if specified, otherwise keep existing
            let updated_priority = request.priority.unwrap_or_else(|| worker.priority());
            let updated_cost = request.cost.unwrap_or_else(|| worker.cost());

            // Build updated health config from resolved runtime config
            let existing_health = &worker.metadata().health_config;
            let updated_health_config = match &request.health {
                Some(update) => update.apply_to(existing_health),
                None => existing_health.clone(),
            };
            let health_endpoint = worker.metadata().health_endpoint.clone();

            // Determine API key: use new one if provided, otherwise keep existing
            let updated_api_key = request
                .api_key
                .clone()
                .or_else(|| worker.metadata().spec.api_key.clone());

            // Keep the full spec so a metadata update preserves PD transport
            // and DP configuration.
            //
            // Status: a metadata-only update is not a re-registration — the
            // worker is the same endpoint. Preserve the old status so a
            // healthy worker stays routable through the update. The one
            // exception is when the update flips `disable_health_check` to
            // `true`: the health checker skips disabled workers, so a stale
            // Pending/NotReady status would never recover. Force Ready in
            // that case.
            let next_status = if updated_health_config.disable_health_check {
                WorkerStatus::Ready
            } else {
                worker.status()
            };
            let mut builder = BasicWorkerBuilder::from_spec((*worker.metadata().spec).clone())
                .http2(worker.http2())
                .labels(updated_labels)
                .health_config(updated_health_config.clone())
                .health_endpoint(&health_endpoint)
                .resilience(worker.resilience().clone())
                .priority(updated_priority)
                .cost(updated_cost)
                // The overload block is not updatable here, but it must survive
                // the rebuild; effective thresholds re-resolve against the same
                // gateway defaults registration used.
                .overload(worker.metadata().spec.overload)
                .overload_defaults(OverloadThresholds::from_gateway_config(
                    &app_context.router_config,
                ))
                .status(next_status);

            // Adopt the old worker's client only if it was materialized: a
            // never-HTTP worker (ZMQ) keeps a deferred slot instead of having
            // a client it will never use built for the replacement.
            if let Some(client) = worker.http_client_handle_if_initialized() {
                builder = builder.http_client(client);
            }

            if let Some(ref api_key) = updated_api_key {
                builder = builder.api_key(api_key.clone());
            }

            // The spec retains the ZMQ address and engine count. The connect
            // signal still has to come from the registry.
            if *worker.connection_mode() == ConnectionMode::Zmq {
                builder =
                    builder.connect_signal_tx(app_context.worker_registry.connect_signal_sender());
            }

            let mut new_worker = builder.build();
            if let Some(previous) = worker.as_any().downcast_ref::<BasicWorker>() {
                // A health probe may finish after this update. Share its state
                // so the result reaches the replacement too.
                new_worker.share_kv_engine_state(previous);
            }
            let new_worker: Arc<dyn Worker> = Arc::new(new_worker);

            // Replace the worker in the registry (overwrite-then-diff)
            let worker_id = app_context
                .worker_registry
                .register_or_replace(new_worker.clone());

            // Same-URL replace preserves the existing shared runtime. If the
            // update disables health checks, force the lifecycle to Ready so
            // the worker does not stay stuck in a non-routable pre-update
            // status while the manager correctly skips future probes.
            if updated_health_config.disable_health_check {
                let _ = app_context
                    .worker_registry
                    .transition_status(&worker_id, WorkerStatus::Ready);
            }

            updated_workers.push(new_worker);
        }

        // Log result
        if updated_workers.len() == 1 {
            info!("Updated worker {}", updated_workers[0].url());
        } else {
            info!("Updated {} workers", updated_workers.len());
        }

        // Store updated workers for subsequent steps
        context.data.updated_workers = Some(updated_workers);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::Duration};

    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response},
        EngineId,
    };
    use openai_protocol::worker::{
        HealthCheckConfig, HealthCheckUpdate, OverloadUpdate, WorkerSpec, WorkerType,
        WorkerUpdateRequest,
    };
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::AppContext,
        middleware::AuthConfig,
        routers::{
            common::kv_transfer::{connector_mode_for_worker, KvConnectorMode},
            grpc::{
                backend_client::BackendClient,
                zmq_client::{EosTokenIds, ZmqEngineClient},
            },
        },
        worker::{BasicWorker, RuntimeType},
    };

    fn make_app_context(workers: &[Arc<dyn Worker>]) -> Arc<AppContext> {
        use crate::{
            config::RouterConfig,
            middleware::TokenBucket,
            observability::inflight_tracker::InFlightRequestTracker,
            routers::{
                common::{openai_bridge, realtime::RealtimeRegistry},
                grpc::multimodal::MultimodalConfigRegistry,
            },
            worker::{WorkerRegistry, WorkerService},
        };

        let router_config = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .build_unchecked();
        let registry = Arc::new(WorkerRegistry::new());
        for w in workers {
            registry.register(Arc::clone(w)).unwrap();
        }
        let job_queue = Arc::new(std::sync::OnceLock::new());

        Arc::new(AppContext {
            gateway_auth: AuthConfig::new(None),
            client: reqwest::Client::new(),
            router_config: router_config.clone(),
            rate_limiter: Some(Arc::new(TokenBucket::new(1000, 1000))),
            rate_limit_manager: None,
            worker_registry: Arc::clone(&registry),
            policy_registry: Arc::new(crate::policies::PolicyRegistry::new(
                router_config.policy.clone(),
            )),
            reasoning_parser_factory: None,
            tool_parser_factory: None,
            gateway: None,
            response_storage: Arc::new(smg_data_connector::MemoryResponseStorage::new()),
            conversation_storage: Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            conversation_item_storage: Arc::new(
                smg_data_connector::MemoryConversationItemStorage::new(),
            ),
            worker_monitor: None,
            configured_reasoning_parser: None,
            configured_tool_parser: None,
            worker_job_queue: Arc::clone(&job_queue),
            workflow_engines: Arc::new(std::sync::OnceLock::new()),
            mcp_orchestrator: Arc::new(std::sync::OnceLock::new()),
            mcp_format_registry: openai_bridge::FormatRegistry::new(),
            tokenizer_registry: Arc::new(llm_tokenizer::registry::TokenizerRegistry::new()),
            multimodal_config_registry: Arc::new(MultimodalConfigRegistry::new()),
            wasm_manager: None,
            worker_client_cache: Arc::new(crate::worker::WorkerHttpClientCache::new(
                &router_config,
            )),
            worker_service: Arc::new(WorkerService::new(registry, job_queue, router_config)),
            inflight_tracker: InFlightRequestTracker::new(),
            kv_event_monitor: None,
            rl: None,
            realtime_registry: Arc::new(RealtimeRegistry::new()),
            webrtc_bind_addr: None,
            webrtc_stun_server: None,
        })
    }

    fn make_context(
        app_context: Arc<AppContext>,
        worker: Arc<dyn Worker>,
        labels: HashMap<String, String>,
    ) -> WorkflowContext<WorkerUpdateWorkflowData> {
        let data = WorkerUpdateWorkflowData {
            config: WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: Some(labels),
                api_key: None,
                health: None,
            },
            worker_url: worker.url().to_string(),
            dp_aware: false,
            app_context: Some(app_context),
            workers_to_update: Some(vec![worker]),
            updated_workers: None,
        };
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    #[tokio::test]
    async fn pd_update_preserves_connector_and_pair_descriptor() {
        for (connector, mode, protocol, transport, pair_key, expected_mode) in [
            (
                "MooncakeConnector",
                ConnectionMode::Http,
                None,
                "mooncake",
                "vllm/mooncake/page=16",
                KvConnectorMode::Mooncake {
                    host: "prefill".to_string(),
                    port: 9123,
                    engine_id: Some("engine-1_dp1".to_string()),
                },
            ),
            (
                "NixlConnector",
                ConnectionMode::Grpc,
                Some("cluster-blue"),
                "nixl",
                "cluster-blue",
                KvConnectorMode::Nixl,
            ),
        ] {
            let mut spec = WorkerSpec::new("http://prefill:8000");
            spec.worker_type = WorkerType::Prefill;
            spec.runtime_type = RuntimeType::Vllm;
            spec.connection_mode = mode;
            spec.kv_connector = Some(connector.to_string());
            spec.kv_role = Some("kv_producer".to_string());
            spec.kv_engine_id = Some("engine-1".to_string());
            spec.bootstrap_port = Some(9123);
            spec.pairing_protocol = protocol.map(str::to_string);
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::from_spec(spec)
                    .dp_config(1, 2)
                    .label("block_size", "16")
                    .label("tier", "silver")
                    .priority(2)
                    .cost(1.5)
                    .health_config(HealthCheckConfig {
                        timeout_secs: 3,
                        check_interval_secs: 11,
                        ..Default::default()
                    })
                    .status(WorkerStatus::Ready)
                    .build(),
            );
            let app_ctx = make_app_context(std::slice::from_ref(&worker));
            let mut ctx = make_context(
                Arc::clone(&app_ctx),
                Arc::clone(&worker),
                HashMap::from([("tier".to_string(), "gold".to_string())]),
            );
            ctx.data.config.priority = Some(9);
            ctx.data.config.health = Some(HealthCheckUpdate {
                timeout_secs: Some(7),
                ..Default::default()
            });

            assert_eq!(
                UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap(),
                StepResult::Success
            );

            let updated = app_ctx
                .worker_registry
                .get_by_url("http://prefill:8000@1")
                .expect("worker stays registered");
            assert!(Arc::ptr_eq(
                &updated,
                &ctx.data.updated_workers.as_ref().unwrap()[0]
            ));
            assert_eq!(updated.base_url(), "http://prefill:8000");
            assert_eq!(updated.url(), "http://prefill:8000@1");
            assert_eq!(updated.dp_rank(), Some(1));
            assert_eq!(updated.dp_size(), Some(2));
            assert_eq!(
                updated.metadata().spec.kv_role.as_deref(),
                Some("kv_producer")
            );
            assert_eq!(connector_mode_for_worker(updated.as_ref()), expected_mode);
            assert_eq!(updated.pd_pairing().transport(), Some(transport));
            assert_eq!(updated.pd_pairing().explicit(), protocol);
            assert_eq!(updated.pd_pairing().key(), pair_key);
            assert_eq!(updated.metadata().spec.labels["tier"], "gold");
            assert_eq!(updated.metadata().spec.labels["block_size"], "16");
            assert_eq!(updated.priority(), 9);
            assert_eq!(updated.cost(), 1.5);
            assert_eq!(updated.metadata().health_config.timeout_secs, 7);
            assert_eq!(updated.metadata().health_config.check_interval_secs, 11);
            assert_eq!(updated.status(), WorkerStatus::Ready);
        }
    }

    #[tokio::test]
    async fn pd_update_retains_live_kv_engine_state() {
        for current_id in [Some("engine-current"), None] {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new("grpc://prefill:8000")
                    .worker_type(WorkerType::Prefill)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::Vllm)
                    .kv_connector("MooncakeConnector")
                    .kv_engine_id("engine-initial")
                    .bootstrap_port(Some(9123))
                    .status(WorkerStatus::Ready)
                    .build(),
            );
            worker.refresh_kv_engine_id(current_id.map(str::to_string));
            worker.set_kv_engine_id_confirmed(false);
            let app_ctx = make_app_context(std::slice::from_ref(&worker));
            let mut ctx = make_context(app_ctx, Arc::clone(&worker), HashMap::new());

            UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap();

            let updated = &ctx.data.updated_workers.as_ref().unwrap()[0];
            assert_eq!(updated.kv_engine_id().as_deref(), current_id);
            assert!(!updated.kv_engine_id_confirmed());

            // A probe that began before the update still holds the old worker.
            worker.refresh_kv_engine_id(Some("engine-late".to_string()));
            worker.set_kv_engine_id_confirmed(true);
            assert_eq!(updated.kv_engine_id().as_deref(), Some("engine-late"));
            assert!(updated.kv_engine_id_confirmed());
            assert_eq!(
                connector_mode_for_worker(updated.as_ref()),
                KvConnectorMode::Mooncake {
                    host: "prefill".to_string(),
                    port: 9123,
                    engine_id: Some("engine-late".to_string()),
                }
            );
        }
    }

    /// A label-only PATCH must not disturb a ZMQ worker's transport: the engine
    /// handshakes once at its own startup, so losing the connected client (or
    /// the addresses and engine count that define the sockets) would black-hole
    /// every request until the engine group is restarted.
    #[tokio::test]
    async fn zmq_update_preserves_transport_state_and_the_live_client() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let _engine = engine.expect("mock engine");
        let client = Arc::new(BackendClient::Zmq(client.expect("adapter connect")));

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(ep("ts0.ipc"))
                .connection_mode(ConnectionMode::Zmq)
                .runtime_type(RuntimeType::Vllm)
                .zmq_handshake_address(handshake.clone())
                .zmq_engine_group(2)
                .health_config(HealthCheckConfig::default())
                .overload(OverloadUpdate {
                    waiting_requests: Some(8),
                    token_usage: None,
                })
                .status(WorkerStatus::Ready)
                .build(),
        );
        let old = worker
            .as_any()
            .downcast_ref::<BasicWorker>()
            .expect("BasicWorker");
        old.backend_client
            .load_full()
            .set(Arc::clone(&client))
            .ok()
            .expect("cell empty");

        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let mut ctx = make_context(
            Arc::clone(&app_ctx),
            Arc::clone(&worker),
            HashMap::from([("tier".to_string(), "gold".to_string())]),
        );

        let result = UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap();
        assert_eq!(result, StepResult::Success);

        let updated = ctx.data.updated_workers.as_ref().expect("updated workers");
        assert_eq!(updated.len(), 1);
        let new = updated[0]
            .as_any()
            .downcast_ref::<BasicWorker>()
            .expect("BasicWorker");

        assert_eq!(
            new.metadata().spec.labels.get("tier"),
            Some(&"gold".to_string())
        );
        assert_eq!(
            new.metadata().spec.zmq_handshake_address.as_deref(),
            Some(handshake.as_str())
        );
        assert_eq!(new.metadata().zmq_engine_count(), 2);
        // The overload block survives a metadata-only update, and so does the
        // effective threshold resolved from it.
        assert_eq!(new.metadata().spec.overload.waiting_requests, Some(8));
        assert_eq!(new.metadata().overload.waiting_requests, Some(8));
        assert!(
            new.connect_signal_tx.is_some(),
            "ZMQ promotion stays event-driven after an update"
        );

        let adopted = new
            .backend_client
            .load_full()
            .get()
            .cloned()
            .expect("replacement must inherit the connected client");
        assert!(Arc::ptr_eq(&adopted, &client));
        assert_eq!(new.status(), WorkerStatus::Ready);

        // Neither side of a ZMQ replacement may materialize an HTTP client
        // it will never use.
        assert!(old.http_client.cell_is_empty());
        assert!(new.http_client.cell_is_empty());
    }

    /// An HTTP worker's replacement must adopt the same shared client — a
    /// fresh client would silently detach it from the worker-client cache
    /// entry the old worker was keeping alive.
    #[tokio::test]
    async fn http_update_adopts_the_materialized_shared_client() {
        let client = Arc::new(reqwest::Client::new());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w:8080")
                .http_client(Arc::clone(&client))
                .status(WorkerStatus::Ready)
                .build(),
        );
        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let mut ctx = make_context(app_ctx, Arc::clone(&worker), HashMap::new());

        let result = UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap();
        assert_eq!(result, StepResult::Success);

        let updated = &ctx.data.updated_workers.as_ref().expect("updated workers")[0];
        let adopted = updated
            .http_client_handle_if_initialized()
            .expect("materialized client is adopted");
        assert!(Arc::ptr_eq(&adopted, &client));
    }

    /// The adopted client speaks one HTTP version; the replacement must keep
    /// describing it correctly.
    #[tokio::test]
    async fn http_update_preserves_http2() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w:8080")
                .http_client(Arc::new(reqwest::Client::new()))
                .http2(true)
                .status(WorkerStatus::Ready)
                .build(),
        );
        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let mut ctx = make_context(app_ctx, Arc::clone(&worker), HashMap::new());

        UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap();

        let updated = &ctx.data.updated_workers.as_ref().expect("updated workers")[0];
        assert!(updated.http2());
    }
}
