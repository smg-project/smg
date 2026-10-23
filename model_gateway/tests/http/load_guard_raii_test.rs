//! Tests for the worker load guard's RAII lifecycle on a response body.
//!
//! A streaming handler hands its `WorkerLoadGuard` to the body's own stream
//! (`GuardedStream`). These tests verify that the guard decrements the
//! worker's load when:
//! - the response body is read to its end
//! - the response body is dropped (a client disconnect)
//! - several guards ride together (dual prefill/decode workers)
//! and that the guard lives with the body, not with the response head.

use std::{convert::Infallible, sync::Arc};

use axum::{body::Body, response::Response};
use bytes::Bytes;
use futures_util::{stream, StreamExt};
use http_body_util::BodyExt;
use smg::worker::{BasicWorkerBuilder, GuardedStream, Worker, WorkerLoadGuard};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// A response whose body is `frames`, with `guard` riding in the body's
/// stream.
fn guarded_response<G: Send + Unpin + 'static>(frames: Vec<Bytes>, guard: G) -> Response {
    let stream = stream::iter(frames.into_iter().map(Ok::<_, Infallible>));
    Response::new(Body::from_stream(GuardedStream::new(stream, guard)))
}

/// An SSE streaming response fed from `rx`, with `guard` riding in the
/// body's stream.
fn guarded_sse_response<G: Send + Unpin + 'static>(
    rx: mpsc::UnboundedReceiver<Bytes>,
    guard: G,
) -> Response {
    let stream = UnboundedReceiverStream::new(rx).map(Ok::<_, std::io::Error>);
    Response::new(Body::from_stream(GuardedStream::new(stream, guard)))
}

/// Helper to create a test worker
fn create_test_worker() -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new("http://localhost:8000")
            .health_config(openai_protocol::worker::HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    )
}

#[tokio::test]
async fn test_guard_dropped_when_response_body_consumed() {
    let worker = create_test_worker();
    assert_eq!(worker.load(), 0);

    let guard = WorkerLoadGuard::new(worker.clone(), None);
    assert_eq!(worker.load(), 1);

    let guarded_response = guarded_response(vec![Bytes::from("Hello, World!")], guard);

    // Load should still be 1 (guard is in the body)
    assert_eq!(worker.load(), 1);

    // Consume the response body
    let body = guarded_response.into_body();
    let bytes = body.collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[..], b"Hello, World!");

    // After consuming, guard should be dropped, load should be 0
    assert_eq!(worker.load(), 0);
}

#[tokio::test]
async fn test_guard_dropped_when_response_dropped_without_consumption() {
    let worker = create_test_worker();
    assert_eq!(worker.load(), 0);

    {
        let guard = WorkerLoadGuard::new(worker.clone(), None);
        assert_eq!(worker.load(), 1);

        let _guarded_response = guarded_response(vec![Bytes::from("Hello, World!")], guard);

        // Load is still 1
        assert_eq!(worker.load(), 1);

        // Response goes out of scope here
    }

    // After response is dropped, guard should be dropped, load should be 0
    assert_eq!(worker.load(), 0);
}

#[expect(clippy::disallowed_methods, reason = "test infrastructure")]
#[tokio::test]
async fn test_streaming_guard_dropped_when_stream_ends() {
    let worker = create_test_worker();
    assert_eq!(worker.load(), 0);

    // Create a channel for SSE streaming
    let (tx, rx) = mpsc::unbounded_channel::<Bytes>();

    let guard = WorkerLoadGuard::new(worker.clone(), None);
    assert_eq!(worker.load(), 1);

    let guarded_response = guarded_sse_response(rx, guard);

    // Spawn a task to consume the response
    let worker_clone = worker.clone();
    let consume_task = tokio::spawn(async move {
        {
            let mut body = guarded_response.into_body();
            while let Some(result) = body.frame().await {
                if result.is_err() {
                    break;
                }
            }
            // Body is still in scope here, guard not dropped yet
        }
        // Body dropped here, guard should be dropped
        assert_eq!(worker_clone.load(), 0);
    });

    // Send some data
    tx.send(Bytes::from("data: chunk1\n\n")).unwrap();
    tx.send(Bytes::from("data: chunk2\n\n")).unwrap();

    // Load should still be 1 while streaming
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    assert_eq!(worker.load(), 1);

    // Close the sender to end the stream
    drop(tx);

    // Wait for consumer to finish
    consume_task.await.unwrap();

    // Load should now be 0
    assert_eq!(worker.load(), 0);
}

#[tokio::test]
async fn test_streaming_guard_dropped_on_client_disconnect() {
    let worker = create_test_worker();
    assert_eq!(worker.load(), 0);

    let (tx, rx) = mpsc::unbounded_channel::<Bytes>();

    let guard = WorkerLoadGuard::new(worker.clone(), None);
    assert_eq!(worker.load(), 1);

    let guarded_response = guarded_sse_response(rx, guard);

    // Start consuming but drop early (simulate client disconnect)
    {
        let mut body = guarded_response.into_body();

        // Read one frame
        tx.send(Bytes::from("data: chunk1\n\n")).unwrap();
        let _ = body.frame().await;

        // Load still 1
        assert_eq!(worker.load(), 1);

        // Body dropped here (simulating client disconnect)
    }

    // Guard should be dropped when body is dropped
    assert_eq!(worker.load(), 0);

    // tx is still open but no one is listening
    drop(tx);
}

/// The guard lives with the body, not with the response head: a consumer
/// that keeps the body and drops the head (another handler reading the
/// stream in-process) holds the worker's load until the body goes.
#[tokio::test]
async fn test_guard_rides_in_the_body_not_the_head() {
    let worker = create_test_worker();
    let (tx, rx) = mpsc::unbounded_channel::<Bytes>();

    let guard = WorkerLoadGuard::new(worker.clone(), None);
    let (parts, mut body) = guarded_sse_response(rx, guard).into_parts();
    drop(parts);
    assert_eq!(worker.load(), 1, "the head is gone, the body is not");

    tx.send(Bytes::from("data: chunk1\n\n")).unwrap();
    assert!(body.frame().await.is_some());
    assert_eq!(worker.load(), 1);

    drop(body);
    assert_eq!(worker.load(), 0);
}

#[tokio::test]
async fn test_multiple_guards_all_dropped() {
    let worker1 = create_test_worker();
    let worker2 = create_test_worker();
    assert_eq!(worker1.load(), 0);
    assert_eq!(worker2.load(), 0);

    {
        // Create guards for both workers (simulates dual prefill/decode)
        let guard1 = WorkerLoadGuard::new(worker1.clone(), None);
        let guard2 = WorkerLoadGuard::new(worker2.clone(), None);
        assert_eq!(worker1.load(), 1);
        assert_eq!(worker2.load(), 1);

        let _response = guarded_response(vec![Bytes::from("Hello")], vec![guard1, guard2]);

        // Both loads are 1
        assert_eq!(worker1.load(), 1);
        assert_eq!(worker2.load(), 1);
    }

    // Both guards dropped when response goes out of scope
    assert_eq!(worker1.load(), 0);
    assert_eq!(worker2.load(), 0);
}

#[tokio::test]
async fn test_guard_with_empty_body() {
    let worker = create_test_worker();
    assert_eq!(worker.load(), 0);

    {
        let guard = WorkerLoadGuard::new(worker.clone(), None);
        assert_eq!(worker.load(), 1);

        let guarded_response = guarded_response(Vec::new(), guard);

        // Consume empty body
        let body = guarded_response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        assert!(bytes.is_empty());
    }

    assert_eq!(worker.load(), 0);
}
