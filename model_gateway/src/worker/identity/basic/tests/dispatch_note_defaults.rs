//! The dispatch notes' defaults: a worker that answers the door with its state and overrides none
//! of them has no request counted in and none counted out.

use std::{any::Any, sync::Arc, time::Duration};

use async_trait::async_trait;
use openai_protocol::worker::{ConnectionMode, WorkerStatus, WorkerType};

use crate::{
    routers::grpc::backend_client::BackendClient,
    worker::{
        health::circuit_breaker::CircuitState,
        identity::basic::{BasicWorker, WorkerMetadata},
        runtime::WorkerRuntime,
        BasicWorkerBuilder, ResolvedResilience, Worker, WorkerResult,
    },
};

/// A worker that answers the door with its state and overrides nothing that
/// has a default: the shape of a worker written outside this crate that
/// keeps a [`WorkerRuntime`] and writes none of the dispatch notes itself.
#[derive(Debug)]
struct DoorOnly(BasicWorker);

#[async_trait]
impl Worker for DoorOnly {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn url(&self) -> &str {
        self.0.url()
    }
    fn api_key(&self) -> Option<&String> {
        self.0.api_key()
    }
    fn worker_type(&self) -> &WorkerType {
        self.0.worker_type()
    }
    fn connection_mode(&self) -> &ConnectionMode {
        self.0.connection_mode()
    }
    fn status(&self) -> WorkerStatus {
        self.0.status()
    }
    fn set_status(&self, status: WorkerStatus) {
        self.0.set_status(status);
    }
    fn load(&self) -> usize {
        self.0.load()
    }
    fn increment_load(&self) {
        self.0.increment_load();
    }
    fn decrement_load(&self) {
        self.0.decrement_load();
    }
    fn routing_key_load(&self) -> usize {
        self.0.routing_key_load()
    }
    fn routing_key_inflight(&self, routing_key: &str) -> usize {
        self.0.routing_key_inflight(routing_key)
    }
    fn increment_routing_key_load(&self, routing_key: &str) {
        self.0.increment_routing_key_load(routing_key);
    }
    fn decrement_routing_key_load(&self, routing_key: &str) {
        self.0.decrement_routing_key_load(routing_key);
    }
    fn processed_requests(&self) -> usize {
        self.0.processed_requests()
    }
    fn increment_processed(&self) {
        self.0.increment_processed();
    }
    fn metadata(&self) -> &WorkerMetadata {
        self.0.metadata()
    }
    fn circuit_breaker_state(&self) -> CircuitState {
        self.0.circuit_breaker_state()
    }
    fn circuit_breaker_can_execute(&self) -> bool {
        self.0.circuit_breaker_can_execute()
    }
    fn record_circuit_breaker_outcome(&self, success: bool) {
        self.0.record_circuit_breaker_outcome(success);
    }
    fn resilience(&self) -> &ResolvedResilience {
        self.0.resilience()
    }
    fn http_client(&self) -> &reqwest::Client {
        self.0.http_client()
    }
    fn http_client_handle_if_initialized(&self) -> Option<Arc<reqwest::Client>> {
        self.0.http_client_handle_if_initialized()
    }
    async fn check_health_async(&self) -> WorkerResult<()> {
        self.0.check_health_async().await
    }
    async fn get_backend_client(&self) -> WorkerResult<Option<Arc<BackendClient>>> {
        self.0.get_backend_client().await
    }
    async fn grpc_health_check(&self) -> WorkerResult<bool> {
        self.0.grpc_health_check().await
    }
    async fn zmq_health_check(&self) -> WorkerResult<bool> {
        self.0.zmq_health_check().await
    }
    async fn http_health_check(&self) -> WorkerResult<bool> {
        self.0.http_health_check().await
    }
    fn runtime_state(&self) -> Option<Arc<WorkerRuntime>> {
        self.0.runtime_state()
    }
}

/// The trait's own dispatch notes are empty, the starts as the ends: a
/// worker that answers the door and overrides none of them has no request
/// counted in and none counted out, so its tracked load and its prefill
/// backlog read zero after a start and after its end. A start that wrote
/// through the door beside an empty end counted every dispatch in and none
/// out.
#[test]
fn the_dispatch_note_defaults_count_nothing_in_and_nothing_out() {
    let worker = DoorOnly(BasicWorkerBuilder::new("http://w:1").build());
    let state = worker.runtime_state().unwrap();

    worker.note_tracked_started_at(1);
    worker.note_prefill_started_at(8_192, true, 1);
    assert_eq!(state.tracked_load(), 0, "a start counts no request in");
    assert_eq!(state.prefill_backlog(), Duration::ZERO, "nor a prompt");

    worker.note_tracked_ended();
    worker.note_prefill_ended(8_192, true, Some(0));
    assert_eq!(state.tracked_load(), 0);
    assert_eq!(state.prefill_backlog(), Duration::ZERO);
}
