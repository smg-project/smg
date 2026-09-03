//! Router-to-Worker discovery service for the Rust mock worker.
//!
//! The mock worker serves this control plane alongside its token-ID scheduler
//! data plane. It provides a GPU-free contract test for explicit Worker SMG
//! identity, health, capabilities, and engine topology.

use std::{collections::HashMap, sync::Arc, time::SystemTime};

use smg_grpc_client::worker_proto::{
    self as proto,
    worker_control_server::{WorkerControl, WorkerControlServer},
};
use tonic::{Request, Response, Status};

use crate::config::Config;

struct State {
    identity: proto::WorkerIdentity,
    capabilities: proto::WorkerCapabilities,
    topology: proto::WorkerTopology,
}

/// In-memory WorkerControl implementation used by GPU-free integration tests.
#[derive(Clone)]
pub struct MockWorkerControl {
    state: Arc<State>,
}

impl MockWorkerControl {
    pub fn new(config: &Config, host: &str, port: u16) -> Self {
        let worker_id = format!("mock-worker-{port}");
        // Every real WorkerControlServer advertises its engine transport, and
        // the Router refuses registrations that do not (it decides string-stop
        // ownership from it). The mock fronts its scheduler over gRPC.
        let engine_attributes: HashMap<String, String> =
            [("engine_transport".to_string(), "grpc".to_string())].into();
        let endpoint = format!("grpc://{host}:{port}");
        Self {
            state: Arc::new(State {
                identity: proto::WorkerIdentity {
                    worker_id: worker_id.clone(),
                    instance_id: format!("{worker_id}-{}", std::process::id()),
                    hostname: host.to_string(),
                    zone: String::new(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    started_at: now(),
                    labels: [("role".to_string(), "smg-worker".to_string())].into(),
                },
                capabilities: proto::WorkerCapabilities {
                    api_major: 1,
                    api_minor: 0,
                    features: vec![
                        "generate".to_string(),
                        "abort".to_string(),
                        "token_id_streaming".to_string(),
                    ],
                    engines: vec![proto::EngineCapability {
                        engine_type: "tokenspeed".to_string(),
                        engine_version: "mock".to_string(),
                        model_ids: vec![config.model_id.clone()],
                        features: vec!["generate".to_string(), "abort".to_string()],
                    }],
                    max_concurrent_requests: u32::try_from(config.engine.max_running)
                        .unwrap_or(u32::MAX),
                    attributes: engine_attributes.clone(),
                },
                topology: proto::WorkerTopology {
                    worker_id,
                    topology_version: 1,
                    engines: vec![proto::EngineEndpoint {
                        engine_id: "mock-engine-0".to_string(),
                        engine_type: "tokenspeed".to_string(),
                        endpoint,
                        model_ids: vec![config.model_id.clone()],
                        replica_group: "mock".to_string(),
                        data_parallel_rank: Some(0),
                        tensor_parallel_rank: None,
                        pipeline_parallel_rank: None,
                        attributes: engine_attributes,
                    }],
                    observed_at: now(),
                },
            }),
        }
    }

    pub fn into_server(self) -> WorkerControlServer<Self> {
        WorkerControlServer::new(self)
    }

    fn health_response(&self, include_components: bool) -> proto::GetHealthResponse {
        let state = proto::WorkerHealthState::Serving;
        proto::GetHealthResponse {
            state: state.into(),
            message: "ready".to_string(),
            checked_at: now(),
            components: include_components
                .then(|| proto::ComponentHealth {
                    component_id: "mock-engine-0".to_string(),
                    state: state.into(),
                    message: "ready".to_string(),
                    checked_at: now(),
                })
                .into_iter()
                .collect(),
        }
    }
}

#[tonic::async_trait]
impl WorkerControl for MockWorkerControl {
    async fn get_identity(
        &self,
        _request: Request<proto::GetIdentityRequest>,
    ) -> Result<Response<proto::GetIdentityResponse>, Status> {
        Ok(Response::new(proto::GetIdentityResponse {
            identity: Some(self.state.identity.clone()),
        }))
    }

    async fn get_capabilities(
        &self,
        _request: Request<proto::GetCapabilitiesRequest>,
    ) -> Result<Response<proto::GetCapabilitiesResponse>, Status> {
        Ok(Response::new(proto::GetCapabilitiesResponse {
            capabilities: Some(self.state.capabilities.clone()),
        }))
    }

    async fn get_health(
        &self,
        request: Request<proto::GetHealthRequest>,
    ) -> Result<Response<proto::GetHealthResponse>, Status> {
        Ok(Response::new(
            self.health_response(request.into_inner().include_components),
        ))
    }

    async fn get_topology(
        &self,
        _request: Request<proto::GetTopologyRequest>,
    ) -> Result<Response<proto::GetTopologyResponse>, Status> {
        Ok(Response::new(proto::GetTopologyResponse {
            topology: Some(self.state.topology.clone()),
        }))
    }
}

fn now() -> Option<prost_types::Timestamp> {
    Some(SystemTime::now().into())
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::worker_proto::worker_control_client::WorkerControlClient;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::transport::Server;

    use super::*;
    use crate::engine::EngineParams;

    fn config() -> Config {
        Config {
            host: "127.0.0.1".to_string(),
            http_base_port: 0,
            http_count: 0,
            grpc_base_port: 19_000,
            grpc_count: 1,
            zmq_handshake: None,
            zmq_count: 0,
            zmq_start_index: 0,
            model_id: "mock-model".to_string(),
            tokenizer_path: "mock-model".to_string(),
            gen_delay: std::time::Duration::ZERO,
            output_tokens: 8,
            realistic: false,
            engine: EngineParams::default(),
        }
    }

    #[tokio::test]
    async fn discovery_contract_works_over_a_real_tonic_channel() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let control = MockWorkerControl::new(&config(), "127.0.0.1", address.port());
        let mut servers = tokio::task::JoinSet::new();
        servers.spawn(async move {
            Server::builder()
                .add_service(control.into_server())
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
        });

        let mut client = WorkerControlClient::connect(format!("http://{address}"))
            .await
            .unwrap();
        let identity = client
            .get_identity(proto::GetIdentityRequest {})
            .await
            .unwrap()
            .into_inner()
            .identity
            .unwrap();
        assert_eq!(
            identity.labels.get("role").map(String::as_str),
            Some("smg-worker")
        );

        let capabilities = client
            .get_capabilities(proto::GetCapabilitiesRequest {})
            .await
            .unwrap()
            .into_inner()
            .capabilities
            .unwrap();
        assert!(capabilities
            .features
            .iter()
            .any(|feature| feature == "generate"));

        let health = client
            .get_health(proto::GetHealthRequest {
                include_components: true,
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(health.state(), proto::WorkerHealthState::Serving);
        assert_eq!(health.components.len(), 1);

        let topology = client
            .get_topology(proto::GetTopologyRequest {})
            .await
            .unwrap()
            .into_inner()
            .topology
            .unwrap();
        assert_eq!(topology.engines.len(), 1);
        assert_eq!(topology.engines[0].model_ids, ["mock-model"]);
        servers.abort_all();
    }
}
