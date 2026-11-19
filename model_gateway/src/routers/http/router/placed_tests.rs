//! The regular HTTP router over a store and a registry of loopback workers.

use std::{error::Error, sync::atomic::Ordering};

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use serde_json::{json, Value};
use smg_policy_controller::{
    store::{GatewayId, LeasesStore, WorkersStore},
    AppliedSpec, PolicySpec, PowerOfTwoMetric,
};
use tokio::time::timeout;

use super::{
    tests::{fixed_status_upstream, streamed_request},
    *,
};
use crate::{
    config::PolicyConfig,
    middleware::ReportedRefusal,
    policy_controller::host::testing::{bed_over, MODEL},
    routers::{
        common::{
            request_lease::test_probe::{spawn_release_gated_stub, DropProbeRequest},
            test_support::fast_retry_config,
        },
        grpc::backend_client::BasicWorkerBuilder,
    },
    tenant::TenantKey,
};

type Case<T = ()> = Result<T, Box<dyn Error>>;

const GATEWAY: GatewayId = GatewayId::LOCAL;
const CHAT: &str = "/v1/chat/completions";
const GENERATE: &str = "/generate";
const TRANSCRIBE: &str = "/v1/audio/transcriptions";

fn router(worker_registry: Arc<WorkerRegistry>, host: Option<Host>) -> Router {
    Router {
        worker_registry,
        policy_registry: Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
        host: host.map(Arc::new),
        retry_config: fast_retry_config(1),
        max_payload_size: 1 << 20,
        stream_stall_timeout: None,
        max_buffered_request_bytes: 0,
        realtime_registry: Arc::new(RealtimeRegistry::new()),
        webrtc_bind_addr: None,
        webrtc_stun_server: None,
        rate_limit_manager: None,
    }
}

async fn chat(router: &Router) -> Case<Response> {
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let request = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}]
    }))?;
    Ok(router
        .route_chat(None, &meta, Arc::new(request), MODEL)
        .await)
}

/// Six chats with two registry workers and one row in the store.
async fn six_chats(hosted: bool) -> Case<(usize, usize)> {
    let (a, hits_a) = fixed_status_upstream(&[CHAT], StatusCode::OK).await;
    let (b, hits_b) = fixed_status_upstream(&[CHAT], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, workers) = bed_over(Arc::clone(&registry), &[&a, &b]);
    let row = store.by_key(&a).ok_or("worker A had no row to remove")?;
    store
        .remove(row.id, row.revision)
        .ok_or("worker A's row stood")?;
    // Only the store decides overload for a hosted attempt.
    registry.set_overload_shed(true);
    assert!(registry.set_worker_overloaded(&workers[1], true));
    let router = router(registry, hosted.then_some(host));
    for _ in 0..6 {
        assert_eq!(chat(&router).await?.status(), StatusCode::OK);
    }
    Ok((hits_a.load(Ordering::SeqCst), hits_b.load(Ordering::SeqCst)))
}

#[tokio::test]
async fn an_attempt_under_the_host_goes_to_the_worker_the_controller_placed() -> Case {
    assert_eq!(six_chats(true).await?, (0, 6));
    Ok(())
}

#[tokio::test]
async fn the_same_bed_with_no_host_is_placed_by_the_classic_policy() -> Case {
    let (a, b) = six_chats(false).await?;
    assert!(a > 0);
    assert_eq!(a + b, 6);
    Ok(())
}

#[tokio::test]
async fn an_attempt_is_never_placed_on_a_worker_of_another_wire() -> Case {
    let (a, hits_a) = fixed_status_upstream(&[CHAT], StatusCode::OK).await;
    let (b, hits_b) = fixed_status_upstream(&[CHAT], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, _store, _) = bed_over(Arc::clone(&registry), &[&a, &b]);
    // A's row stands in the store, and the gateway's worker at A speaks
    // another wire.
    registry.register_or_replace(Arc::new(
        BasicWorkerBuilder::new(a.as_str())
            .model(ModelCard::new(MODEL))
            .status(WorkerStatus::Ready)
            .connection_mode(ConnectionMode::Grpc)
            .build(),
    ));
    let router = router(registry, Some(host));
    for _ in 0..16 {
        assert_eq!(chat(&router).await?.status(), StatusCode::OK);
    }
    assert_eq!(
        (hits_a.load(Ordering::SeqCst), hits_b.load(Ordering::SeqCst)),
        (0, 16)
    );
    Ok(())
}

#[tokio::test]
async fn a_placed_attempt_says_its_dispatch_and_its_lease_ends_with_it() -> Case {
    let probe = Arc::new(());
    let gate = Arc::clone(&probe);
    let (url, released) = spawn_release_gated_stub(Arc::downgrade(&probe)).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, workers) = bed_over(Arc::clone(&registry), &[&url]);
    let router = router(registry, Some(host));
    let request = router.route_typed_request(
        None,
        None,
        DropProbeRequest {
            text: "hi".into(),
            _probe: Arc::clone(&probe),
        },
        GENERATE,
        MODEL,
    );
    let (response, inspected) = tokio::join!(request, async {
        let ready = timeout(Duration::from_secs(2), async {
            while workers[0].load() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await;
        // A dispatched lease survives a sweep past its dispatch deadline.
        // The store removes a single lease at either kind of end.
        let live = LeasesStore::live(&*store, GATEWAY);
        let expired = LeasesStore::expire(&*store, Instant::now() + Duration::from_secs(60));
        drop(gate);
        ready.map(|()| (live, expired))
    });
    let (live, expired) = inspected?;
    assert_eq!(live, 1);
    assert!(expired.is_empty());
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(to_bytes(response.into_body(), 1024).await?.as_ref(), b"{}");
    assert!(released.load(Ordering::SeqCst));
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    assert_eq!(workers[0].load(), 0);
    Ok(())
}

#[tokio::test]
async fn a_retry_names_the_row_of_its_failed_attempt() -> Case {
    let (a, hits_a) = fixed_status_upstream(&[CHAT], StatusCode::SERVICE_UNAVAILABLE).await;
    let (b, hits_b) = fixed_status_upstream(&[CHAT], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &[&a, &b]);
    let mut doc = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)?;
    doc.generation += 1;
    store.apply(AppliedSpec::validate(&doc)?)?;
    // Book B so A is preferred until the retry names A's row.
    let a_row = store.by_key(&a).ok_or("worker A has no row")?.id;
    let b_row = store.by_key(&b).ok_or("worker B has no row")?.id;
    let other = Ask {
        id: b"other request",
        model: MODEL,
        streaming: false,
        headers: None,
        rid: None,
        tried: &[a_row],
        hashes: None,
    };
    let mut booking = host
        .place(&other, Instant::now())
        .await
        .map_err(|why| format!("worker B could not be booked: {why:?}"))?;
    assert_eq!(booking.id(), b_row);
    booking.dispatched(Instant::now());

    let mut router = router(registry, Some(host));
    router.retry_config = fast_retry_config(2);
    let response = chat(&router).await?;
    assert_eq!(
        (
            response.status(),
            hits_a.load(Ordering::SeqCst),
            hits_b.load(Ordering::SeqCst),
        ),
        (StatusCode::OK, 1, 1),
    );
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 1);
    drop(booking);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    Ok(())
}

#[test]
fn a_failed_placement_adds_its_row_once() -> Case {
    let url = "http://127.0.0.1:1";
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, workers) = bed_over(Arc::clone(&registry), &[url]);
    let router = router(registry, Some(host));
    let row = store.by_key(url).ok_or("the worker has no row")?.id;
    let lease = RequestLease::new(
        (),
        RoutingDerivatives::default(),
        ReleasePoint::AfterDispatch,
    );
    let tried = parking_lot::Mutex::new(Vec::new());
    let tried_rows = parking_lot::Mutex::new(Vec::new());
    let ledger = AttemptLedger::default();
    let attempt = TypedAttempt {
        headers: None,
        lease: &lease,
        route: CHAT,
        model_id: MODEL,
        canonical_model: None,
        is_stream: false,
        ledger: &ledger,
        tried: &tried,
        tried_rows: &tried_rows,
    };
    router.account_attempt_outcome(
        &attempt,
        &workers[0],
        StatusCode::OK.into_response(),
        false,
        Some(row),
    );
    assert!(tried_rows.lock().is_empty());
    for _ in 0..2 {
        router.account_attempt_outcome(
            &attempt,
            &workers[0],
            StatusCode::SERVICE_UNAVAILABLE.into_response(),
            false,
            Some(row),
        );
    }
    assert_eq!(*tried_rows.lock(), [row]);
    Ok(())
}

#[tokio::test]
async fn an_attempt_the_controller_does_not_place_reads_the_hosts_answer() -> Case {
    let (url, hits) = fixed_status_upstream(&[CHAT], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &[&url]);
    let row = store
        .by_key(&url)
        .ok_or("the worker had no row to remove")?;
    store
        .remove(row.id, row.revision)
        .ok_or("the worker's row stood")?;
    let router = router(registry, Some(host));
    let response = chat(&router).await?;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        extract_error_code_from_response(&response),
        "no_available_workers"
    );
    assert!(response.extensions().get::<ReportedRefusal>().is_some());
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    Ok(())
}

async fn no_model(router: &Router) -> Case<Response> {
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let request = serde_json::from_value(json!({"text": "hi"}))?;
    Ok(router
        .route_generate(None, &meta, request, UNKNOWN_MODEL_ID)
        .await)
}

async fn transcription(router: &Router) -> Response {
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let request = TranscriptionRequest {
        model: MODEL.into(),
        ..TranscriptionRequest::default()
    };
    let audio = AudioFile {
        bytes: Bytes::from_static(b"audio"),
        file_name: "speech.wav".into(),
        content_type: Some("audio/wav".into()),
    };
    router
        .route_audio_transcriptions(None, &meta, &request, audio, MODEL)
        .await
}

async fn streamed_body(router: &Router) -> Case<Response> {
    router
        .route_streaming_request(streamed_request(&[br#"{"text":"hi"}"#]), GENERATE, false)
        .await
        .map_err(|_| "the router buffered the request".into())
}

/// The refusal by name of `shape`. The refusal's own case holds the whole
/// text of the message.
async fn assert_refused(response: Response, shape: &str) -> Case {
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let body = to_bytes(response.into_body(), 1024).await?;
    let value: Value = serde_json::from_slice(&body)?;
    assert_eq!(value["error"]["code"], "not_decided_by_policy_controller");
    let message = value["error"]["message"]
        .as_str()
        .ok_or("the refusal has no message")?;
    assert!(message.contains(shape), "{message}");
    Ok(())
}

#[tokio::test]
async fn a_request_with_no_model_under_the_host_is_refused_by_name() -> Case {
    let (url, hits) = fixed_status_upstream(&[GENERATE], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &[&url]);
    let router = router(registry, Some(host));
    assert_refused(no_model(&router).await?, "a request that names no model").await?;
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    Ok(())
}

#[tokio::test]
async fn a_transcription_under_the_host_is_refused_by_name() -> Case {
    let (url, hits) = fixed_status_upstream(&[TRANSCRIBE], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &[&url]);
    let router = router(registry, Some(host));
    assert_refused(transcription(&router).await, "a transcription request").await?;
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    Ok(())
}

#[tokio::test]
async fn a_streamed_body_under_the_host_is_refused_by_name() -> Case {
    let (url, hits) = fixed_status_upstream(&[GENERATE], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &[&url]);
    let router = router(registry, Some(host));

    let mut invalid = streamed_request(&[br#"{"text":"hi"}"#]);
    invalid
        .headers_mut()
        .insert("anthropic-version", HeaderValue::from_static("invalid"));
    let answer = router
        .route_streaming_request(invalid, "/v1/messages", false)
        .await
        .map_err(|_| "the router buffered the request")?;
    assert_eq!(answer.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        extract_error_code_from_response(&answer),
        "invalid_anthropic_version"
    );

    assert_refused(
        streamed_body(&router).await?,
        "a request whose body is streamed to its worker",
    )
    .await?;
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    Ok(())
}

#[tokio::test]
async fn the_classic_policy_serves_the_three_shapes_without_a_host() -> Case {
    let (url, hits) = fixed_status_upstream(&[GENERATE, TRANSCRIBE], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (_, store, _) = bed_over(Arc::clone(&registry), &[&url]);
    let router = router(registry, None);
    assert_eq!(no_model(&router).await?.status(), StatusCode::OK);
    assert_eq!(transcription(&router).await.status(), StatusCode::OK);
    assert_eq!(streamed_body(&router).await?.status(), StatusCode::OK);
    assert_eq!(hits.load(Ordering::SeqCst), 3);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    Ok(())
}
