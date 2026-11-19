//! A dispatch notes on the state its worker's door answers: a worker that only answers the door
//! has its requests counted on that state, and a note's end lands where its start did.

use std::{any::Any, sync::Arc, time::Duration};

use async_trait::async_trait;
use openai_protocol::worker::{ConnectionMode, WorkerStatus, WorkerType};
use smg_worker::{
    connection::engine::EngineConnection,
    health::{circuit_breaker::CircuitState, resilience::ResolvedResilience},
    identity::{
        basic::{BasicWorker, Worker, WorkerMetadata},
        error::WorkerResult,
    },
    runtime::WorkerRuntime,
};

use crate::routers::grpc::{backend_client::BasicWorkerBuilder, proto_wrapper::DispatchBooking};

/// A worker that answers the door with its state and overrides nothing that
/// has a default: the shape of a worker written outside this crate that
/// keeps a [`WorkerRuntime`].
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
    async fn engine_connection(&self) -> WorkerResult<Option<Arc<dyn EngineConnection>>> {
        self.0.engine_connection().await
    }
    fn runtime_state(&self) -> Option<Arc<WorkerRuntime>> {
        self.0.runtime_state()
    }
}

/// A worker that answers the door and overrides nothing has its dispatch
/// counted on the state it returns: the request is in its pile and its
/// prompt in its backlog from the booking, and both are given back with it.
#[test]
fn a_dispatch_on_a_worker_that_only_answers_the_door_is_counted_on_its_state() {
    let worker: Arc<dyn Worker> = Arc::new(DoorOnly(BasicWorkerBuilder::new("http://w:1").build()));
    let state = worker.runtime_state().unwrap();

    let booking = DispatchBooking::book(&worker, 8_192, true);
    assert_eq!(state.tracked_load(), 1, "in the pile from the booking");
    assert_eq!(
        state.prefill_backlog(),
        Duration::from_millis(819),
        "and the prompt in the backlog"
    );

    drop(booking);
    assert_eq!(state.tracked_load(), 0, "given back with the booking");
    assert_eq!(state.prefill_backlog(), Duration::ZERO);
}

/// A note's end lands on the state its start was written on. The case swaps
/// the worker's state under a live booking, which the door's doc forbids, to
/// pin where the end goes: the state the booking took reads zero again, and
/// the state swapped in never moved.
#[test]
fn a_bookings_end_lands_on_the_state_its_start_was_written_on() {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w:1").build());
    let other: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w:1").build());
    let booked_on = worker.runtime_state().unwrap();

    let booking = DispatchBooking::book(&worker, 8_192, true);
    assert_eq!(booked_on.tracked_load(), 1);

    assert!(worker.inherit_shared_state_from(other.as_ref()));
    let swapped_in = worker.runtime_state().unwrap();
    assert!(
        !Arc::ptr_eq(&booked_on, &swapped_in),
        "the state was swapped"
    );

    drop(booking);
    assert_eq!(
        booked_on.tracked_load(),
        0,
        "the end lands where the start did"
    );
    assert_eq!(booked_on.prefill_backlog(), Duration::ZERO);
    assert_eq!(
        swapped_in.tracked_load(),
        0,
        "the state swapped in never moved"
    );
    assert_eq!(swapped_in.prefill_backlog(), Duration::ZERO);
}
