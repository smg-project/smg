use prost::Message;
use smg_grpc_client::worker_proto::{self as proto, WorkerIdentity, WorkerTopology};

#[test]
fn discovery_messages_round_trip() {
    let identity = WorkerIdentity {
        worker_id: "worker-a".into(),
        instance_id: "worker-a-1".into(),
        hostname: "node-a".into(),
        labels: [("role".into(), "smg-worker".into())].into(),
        ..Default::default()
    };
    let decoded = WorkerIdentity::decode(identity.encode_to_vec().as_slice()).unwrap();
    assert_eq!(decoded, identity);

    let topology = WorkerTopology {
        worker_id: "worker-a".into(),
        topology_version: 7,
        engines: vec![proto::EngineEndpoint {
            engine_id: "engine-0".into(),
            engine_type: "sglang".into(),
            endpoint: "grpc://127.0.0.1:30000".into(),
            model_ids: vec!["model-a".into()],
            ..Default::default()
        }],
        ..Default::default()
    };
    let decoded = WorkerTopology::decode(topology.encode_to_vec().as_slice()).unwrap();
    assert_eq!(decoded, topology);
}

#[test]
fn tonic_generates_both_client_and_server_bindings() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<proto::worker_control_client::WorkerControlClient<tonic::transport::Channel>>(
    );

    fn accepts_server<T: proto::worker_control_server::WorkerControl>() {}
    let _ = accepts_server::<TestOnlyWorkerControl>;
}

#[derive(Default)]
struct TestOnlyWorkerControl;

#[tonic::async_trait]
impl proto::worker_control_server::WorkerControl for TestOnlyWorkerControl {
    async fn get_identity(
        &self,
        _request: tonic::Request<proto::GetIdentityRequest>,
    ) -> Result<tonic::Response<proto::GetIdentityResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("compile-only test"))
    }

    async fn get_capabilities(
        &self,
        _request: tonic::Request<proto::GetCapabilitiesRequest>,
    ) -> Result<tonic::Response<proto::GetCapabilitiesResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("compile-only test"))
    }

    async fn get_health(
        &self,
        _request: tonic::Request<proto::GetHealthRequest>,
    ) -> Result<tonic::Response<proto::GetHealthResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("compile-only test"))
    }

    async fn get_topology(
        &self,
        _request: tonic::Request<proto::GetTopologyRequest>,
    ) -> Result<tonic::Response<proto::GetTopologyResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("compile-only test"))
    }
}
