//! The regular HTTP router over a store and a registry of loopback workers.

use std::{
    error::Error,
    sync::atomic::{AtomicUsize, Ordering},
};

use axum::{
    body::BodyDataStream,
    routing::{post, MethodRouter},
};
use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use serde_json::{json, Value};
use smg_policy_controller::{
    spec::{KeySourceSpec, RoutingKeySpec},
    store::{
        GatewayId, HealthSink, LeasesStore, MemoryStore, WorkerStatus as RowStatus, WorkersStore,
    },
    AppliedSpec, PolicySpec, PowerOfTwoMetric,
};
use smg_worker::test_support::RefusedAddress;
use tokio::{net::TcpListener, time::timeout};

use super::{
    tests::{fixed_status_upstream, streamed_request},
    *,
};
use crate::{
    config::{PolicyConfig, RouterConfig},
    middleware::ReportedRefusal,
    policy_controller::host::testing::{bed_over, bed_over_models, record_bed_over, MODEL},
    routers::{
        common::{
            request_lease::test_probe::{spawn_release_gated_stub, DropProbeRequest},
            test_support::fast_retry_config,
        },
        grpc::backend_client::BasicWorkerBuilder,
        http::test_support::spawn_sized_sse_worker_cut_after,
    },
    server::ServerConfig,
    tenant::TenantKey,
};

type Case<T = ()> = Result<T, Box<dyn Error>>;

const GATEWAY: GatewayId = GatewayId::LOCAL;
const CHAT: &str = "/v1/chat/completions";
const GENERATE: &str = "/generate";
const GENERATE_BODY: &[u8] = br#"{"text":"hi"}"#;
const TRANSCRIBE: &str = "/v1/audio/transcriptions";
const AUDIO_TYPE: &str = "audio/wav";

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

async fn chat(router: &dyn RouterTrait) -> Case<Response> {
    chat_with_stream(router, false).await
}

async fn chat_with_stream(router: &dyn RouterTrait, streaming: bool) -> Case<Response> {
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let request = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": streaming
    }))?;
    Ok(router
        .route_chat(None, &meta, Arc::new(request), MODEL)
        .await)
}

/// Two registry workers, with only the overloaded worker in the store.
async fn one_store_row(
    registry: &Arc<WorkerRegistry>,
) -> Case<(Host, Arc<MemoryStore>, [Arc<AtomicUsize>; 2])> {
    let (a, hits_a) = fixed_status_upstream(&[CHAT], StatusCode::OK).await;
    let (b, hits_b) = fixed_status_upstream(&[CHAT], StatusCode::OK).await;
    let (host, store, workers) = bed_over(Arc::clone(registry), &[&a, &b]);
    let row = store.by_key(&a).ok_or("worker A had no row to remove")?;
    store
        .remove(row.id, row.revision)
        .ok_or("worker A's row stood")?;
    // Only the store decides overload for a hosted attempt.
    registry.set_overload_shed(true);
    assert!(registry.set_worker_overloaded(&workers[1], true));
    Ok((host, store, [hits_a, hits_b]))
}

/// Six chats with two registry workers and one row in the store.
async fn six_chats(hosted: bool) -> Case<(usize, usize)> {
    let registry = Arc::new(WorkerRegistry::new());
    let (host, _, [hits_a, hits_b]) = one_store_row(&registry).await?;
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
async fn a_chat_request_through_the_gateway_object_reaches_the_placed_worker() -> Case {
    let mut config = RouterConfig {
        policy_controller: true,
        ..RouterConfig::default()
    };
    config.health_check.disable_health_check = true;
    let mut context = AppContext::from_config(config.clone(), 30, None, None).await?;
    let (host, store, [hits_a, hits_b]) = one_store_row(&context.worker_registry).await?;
    context.policy_host = Some(Arc::new(host));
    context.policy_store = Some(store);
    let gateway = Gateway::from_config(&ServerConfig::minimal(config), &Arc::new(context)).await?;

    assert_eq!(chat(gateway.as_ref()).await?.status(), StatusCode::OK);
    assert_eq!(hits_a.load(Ordering::SeqCst), 0);
    assert_eq!(hits_b.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn the_same_bed_with_no_host_is_placed_by_the_classic_policy() -> Case {
    let (a, b) = six_chats(false).await?;
    assert!(a > 0);
    assert_eq!(a + b, 6);
    Ok(())
}

/// A repeated prompt keeps its worker over plain HTTP. The route states the
/// request's text, the host hashes its windows, and the record of sent
/// prefixes places by them: the workers publish no cache events and state no
/// block size.
#[tokio::test]
async fn a_repeated_prompt_keeps_its_worker_over_http() -> Case {
    let mut workers = Vec::new();
    for _ in 0..3 {
        workers.push(fixed_status_upstream(&[CHAT], StatusCode::OK).await);
    }
    let urls: Vec<&str> = workers.iter().map(|(url, _)| url.as_str()).collect();
    let registry = Arc::new(WorkerRegistry::new());
    let (host, _store) = record_bed_over(Arc::clone(&registry), &urls);
    let router = router(registry, Some(host));
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    // Five windows of text at the document's block size.
    let prompt = "the same long prompt, ".repeat(16);
    let request = Arc::new(serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }))?);
    for _ in 0..40 {
        let answer = router
            .route_chat(None, &meta, Arc::clone(&request), MODEL)
            .await;
        assert_eq!(answer.status(), StatusCode::OK);
    }
    let mut hits: Vec<usize> = workers
        .iter()
        .map(|(_, hits)| hits.load(Ordering::SeqCst))
        .collect();
    hits.sort_unstable();
    assert_eq!(hits, [0, 0, 40]);
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
        tried: &[a_row],
        ..Ask::of(b"other request", MODEL)
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

#[tokio::test]
async fn a_whole_answer_completes_its_lease_as_ok() -> Case {
    let (url, hits) = fixed_status_upstream(&[CHAT], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &[&url]);
    let row = store.by_key(&url).ok_or("the worker has no row")?.id;
    let mut router = router(registry, Some(host));
    let start = Instant::now();

    let response = chat(&router).await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    assert!(store.completed_ok_since(GATEWAY, row, start));

    // An answer over the cap is the gateway's own, made behind the worker's
    // head. It says so, for the class of the lease's end.
    router.max_payload_size = 1;
    let response = chat(&router).await?;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(response.extensions().get::<BehindAnswer>().is_some());
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    Ok(())
}

#[tokio::test]
async fn a_workers_failure_completes_its_lease_before_the_retry_asks() -> Case {
    let first = TcpListener::bind("127.0.0.1:0").await?;
    let retry = TcpListener::bind("127.0.0.1:0").await?;
    let a = format!("http://{}", first.local_addr()?);
    let b = format!("http://{}", retry.local_addr()?);
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &[&a, &b]);
    let a_row = store.by_key(&a).ok_or("worker A has no row")?.id;
    let b_row = store.by_key(&b).ok_or("worker B has no row")?;
    store
        .set_status(b_row.id, RowStatus::NotReady, b_row.revision)
        .ok_or("worker B's row stood")?;
    store.publish_now();
    let b_row = store.by_key(&b).ok_or("worker B has no row")?;

    // Only A can take the first attempt. Its answer makes B ready to retry.
    let first_store = Arc::clone(&store);
    serve_worker(
        first,
        CHAT,
        post(move || {
            let ready = first_store
                .set_status(b_row.id, RowStatus::Ready, b_row.revision)
                .is_some();
            first_store.publish_now();
            async move {
                assert!(ready);
                StatusCode::SERVICE_UNAVAILABLE
            }
        }),
    );
    let at_retry = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&at_retry);
    let retry_store = Arc::clone(&store);
    serve_worker(
        retry,
        CHAT,
        post(move || {
            count.store(LeasesStore::live(&*retry_store, GATEWAY), Ordering::SeqCst);
            async { StatusCode::OK }
        }),
    );

    let mut router = router(registry, Some(host));
    router.retry_config = fast_retry_config(2);
    let start = Instant::now();
    assert_eq!(chat(&router).await?.status(), StatusCode::OK);
    assert_eq!(at_retry.load(Ordering::SeqCst), 1);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    assert!(!store.completed_ok_since(GATEWAY, a_row, start));
    assert!(store.completed_ok_since(GATEWAY, b_row.id, start));
    Ok(())
}

#[tokio::test]
async fn a_send_that_reaches_no_worker_completes_as_a_failed_connection() -> Case {
    let refused = RefusedAddress::reserve();
    let url = refused.http_url();
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &[&url]);
    let row = store.by_key(&url).ok_or("the worker has no row")?.id;
    let router = router(registry, Some(host));
    let start = Instant::now();

    let response = chat(&router).await?;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        extract_error_code_from_response(&response),
        "call_upstream_connection_failed"
    );
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    assert!(!store.completed_ok_since(GATEWAY, row, start));
    Ok(())
}

#[expect(
    clippy::disallowed_methods,
    reason = "the test server stops with its runtime"
)]
fn serve_worker(listener: TcpListener, path: &'static str, handler: MethodRouter) {
    tokio::spawn(
        async move { axum::serve(listener, axum::Router::new().route(path, handler)).await },
    );
}

const FIRST_FRAME: &[u8] = b"data: a\n\n";
const FRAME_WAIT: Duration = Duration::from_secs(5);

/// The caller sends each frame after reading the previous one.
async fn gated_stream(path: &'static str) -> Case<(String, mpsc::Sender<Bytes>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}", listener.local_addr()?);
    let (frames, body) = mpsc::channel(1);
    let body = Arc::new(parking_lot::Mutex::new(Some(body)));
    serve_worker(
        listener,
        path,
        post(move || {
            let body = body.lock().take();
            async move {
                let body = body.map_or_else(Body::empty, |body| {
                    Body::from_stream(ReceiverStream::new(body).map(Ok::<_, Infallible>))
                });
                ([(CONTENT_TYPE, "text/event-stream")], body)
            }
        }),
    );
    Ok((url, frames))
}

async fn next_frame(body: &mut BodyDataStream) -> Case<Bytes> {
    Ok(timeout(FRAME_WAIT, body.next())
        .await?
        .ok_or("the body ended before the next frame")??)
}

#[tokio::test]
async fn a_stream_holds_its_lease_to_its_last_frame() -> Case {
    let (url, frames) = gated_stream(CHAT).await?;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &[&url]);
    let row = store.by_key(&url).ok_or("the worker has no row")?.id;
    let router = router(registry, Some(host));

    let response = timeout(FRAME_WAIT, chat_with_stream(&router, true)).await??;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
    let at_head = LeasesStore::live(&*store, GATEWAY);
    let mut body = response.into_body().into_data_stream();
    frames.send(Bytes::from_static(FIRST_FRAME)).await?;
    assert_eq!(next_frame(&mut body).await?.as_ref(), FIRST_FRAME);
    assert_eq!((at_head, LeasesStore::live(&*store, GATEWAY)), (1, 1));
    for frame in [b"data: b\n\n".as_slice(), b"data: [DONE]\n\n"] {
        frames.send(Bytes::from_static(frame)).await?;
        assert_eq!(next_frame(&mut body).await?.as_ref(), frame);
    }
    let last_frame = Instant::now();
    drop(frames);
    assert!(timeout(FRAME_WAIT, body.next()).await?.is_none());
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    assert!(store.completed_ok_since(GATEWAY, row, last_frame));
    Ok(())
}

#[tokio::test]
async fn a_stream_dropped_before_its_end_is_cancelled() -> Case {
    let (url, frames) = gated_stream(CHAT).await?;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, workers) = bed_over(Arc::clone(&registry), &[&url]);
    let row = store.by_key(&url).ok_or("the worker has no row")?.id;
    let router = router(registry, Some(host));
    let start = Instant::now();

    let response = timeout(FRAME_WAIT, chat_with_stream(&router, true)).await??;
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    frames.send(Bytes::from_static(FIRST_FRAME)).await?;
    assert_eq!(next_frame(&mut body).await?.as_ref(), FIRST_FRAME);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 1);
    assert_eq!(workers[0].load(), 1);
    drop(body);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    assert_eq!(workers[0].load(), 0);
    assert!(!store.completed_ok_since(GATEWAY, row, start));
    Ok(())
}

#[tokio::test]
async fn a_stream_the_worker_cuts_ends_its_lease_as_a_failure() -> Case {
    const BODY: &str = "data: a\n\ndata: b\n\ndata: [DONE]\n\n";
    let url = spawn_sized_sse_worker_cut_after(BODY, FIRST_FRAME.len()).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &[&url]);
    let row = store.by_key(&url).ok_or("the worker has no row")?.id;
    let router = router(registry, Some(host));
    let start = Instant::now();

    let response = chat_with_stream(&router, true).await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 1);
    let bytes = to_bytes(response.into_body(), 1024).await?;
    assert!(bytes.starts_with(FIRST_FRAME));
    assert!(bytes.ends_with(b"data: [DONE]\n\n"));
    assert!(std::str::from_utf8(&bytes)?.contains("\"type\":\"stream_error\""));
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    // The relay ends the body cleanly; the lease still hears of the cut.
    assert!(!store.completed_ok_since(GATEWAY, row, start));
    Ok(())
}

#[tokio::test]
async fn a_worker_that_answered_definitively_takes_no_later_attempt() -> Case {
    let (a, hits_a) = fixed_status_upstream(&[CHAT], StatusCode::INTERNAL_SERVER_ERROR).await;
    let (b, hits_b) = fixed_status_upstream(&[CHAT], StatusCode::SERVICE_UNAVAILABLE).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, _store, _) = bed_over(Arc::clone(&registry), &[&a, &b]);
    let mut router = router(registry, Some(host));
    // Eight attempts: with no rule at the commit a later one goes back to
    // A in all but one run of sixty.
    router.retry_config = fast_retry_config(8);
    let response = chat(&router).await?;
    assert!(response.status().is_server_error());
    assert_eq!(hits_a.load(Ordering::SeqCst), 1);
    assert!(hits_b.load(Ordering::SeqCst) >= 1);
    Ok(())
}

#[test]
fn a_failed_placement_adds_its_row_each_time() -> Case {
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
        rid: None,
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
    assert_eq!(*tried_rows.lock(), [row, row]);
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

/// Under the host a request that names its worker is refused by name where
/// the model's classic policy holds it to that worker, on the typed line
/// and on the transcription line, and no worker is asked. Under a policy
/// that reads no target the header changes nothing.
#[tokio::test]
async fn a_request_that_names_its_worker_is_refused_by_name_under_the_host() -> Case {
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let mut headers = HeaderMap::new();
    headers.insert("x-smg-target-worker", HeaderValue::from_static("0"));
    for (policy, refused) in [
        (PolicyConfig::ConsistentHashing, true),
        (PolicyConfig::RoundRobin, false),
    ] {
        let (url, hits) = fixed_status_upstream(&[GENERATE, TRANSCRIBE], StatusCode::OK).await;
        let registry = Arc::new(WorkerRegistry::new());
        let (host, ..) = bed_over(Arc::clone(&registry), &[&url]);
        let mut router = router(registry, Some(host));
        router.policy_registry = Arc::new(PolicyRegistry::new(policy));
        let request = serde_json::from_value(json!({"text": "hi"}))?;
        let generated = router
            .route_generate(Some(&headers), &meta, request, MODEL)
            .await;
        let request = TranscriptionRequest {
            model: MODEL.into(),
            ..TranscriptionRequest::default()
        };
        let audio = AudioFile {
            bytes: Bytes::from_static(b"audio"),
            file_name: "speech.wav".into(),
            content_type: Some(AUDIO_TYPE.into()),
        };
        let transcribed = router
            .route_audio_transcriptions(Some(&headers), &meta, &request, audio, MODEL)
            .await;
        for response in [generated, transcribed] {
            if refused {
                assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
                assert_eq!(
                    extract_error_code_from_response(&response),
                    "not_decided_by_policy_controller"
                );
            } else {
                assert_eq!(response.status(), StatusCode::OK);
            }
        }
        assert_eq!(hits.load(Ordering::SeqCst), if refused { 0 } else { 2 });
    }
    Ok(())
}

async fn no_model(router: &Router) -> Case<Response> {
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let request = serde_json::from_value(json!({"text": "hi"}))?;
    Ok(router
        .route_generate(None, &meta, request, UNKNOWN_MODEL_ID)
        .await)
}

async fn transcription(
    router: &Router,
    model_id: &str,
    streaming: bool,
    content_type: &str,
) -> Response {
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let request = TranscriptionRequest {
        model: model_id.into(),
        stream: streaming.then_some(true),
        ..TranscriptionRequest::default()
    };
    let audio = AudioFile {
        bytes: Bytes::from_static(b"audio"),
        file_name: "speech.wav".into(),
        content_type: Some(content_type.into()),
    };
    router
        .route_audio_transcriptions(None, &meta, &request, audio, model_id)
        .await
}

/// A proxied GET and a realtime session pick their worker by no routing
/// policy. Under the host each reads as it reads without one.
#[tokio::test]
async fn a_proxied_get_and_a_realtime_session_read_as_on_the_classic_line_under_the_host() -> Case {
    let registry = Arc::new(WorkerRegistry::new());
    let (host, _, _) = bed_over(Arc::clone(&registry), &[]);
    let session = serde_json::from_value(json!({"type": "realtime", "model": MODEL}))?;

    for router in [
        router(Arc::clone(&registry), Some(host)),
        router(registry, None),
    ] {
        let hosted = router.host.is_some();
        let get = router.health_generate(Request::new(Body::empty())).await;
        assert_eq!(get.status(), StatusCode::SERVICE_UNAVAILABLE, "{hosted}");
        assert_eq!(extract_error_code_from_response(&get), "no_workers");
        let realtime = router.route_realtime_session(None, &session).await;
        assert_eq!(realtime.status(), StatusCode::NOT_FOUND, "{hosted}");
        assert_eq!(
            extract_error_code_from_response(&realtime),
            "model_not_found"
        );
    }
    Ok(())
}

#[tokio::test]
async fn a_request_that_names_no_model_is_asked_under_the_one_model_served() -> Case {
    assert_unnamed_model([Some(MODEL); 2], Some(MODEL));
    let (url, hits) = fixed_status_upstream(&[GENERATE], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &[&url]);
    let mut doc = PolicySpec::preset("round_robin")?;
    doc.generation += 1;
    doc.bindings[0].matcher.model = MODEL.into();
    store.apply(AppliedSpec::validate(&doc)?)?;
    let router = router(registry, Some(host));
    assert_eq!(no_model(&router).await?.status(), StatusCode::OK);

    let row = store.by_key(&url).ok_or("the worker has no row")?;
    store.remove(row.id, row.revision).ok_or("the row stands")?;
    store.publish_now();
    let response = no_model(&router).await?;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(response.into_body(), 1024).await?;
    let value: Value = serde_json::from_slice(&body)?;
    assert_eq!(
        value["error"]["message"],
        format!("No available workers for model '{MODEL}'")
    );
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    Ok(())
}

fn assert_unnamed_model(models: [Option<&str>; 2], expected: Option<&str>) {
    let registry = Arc::new(WorkerRegistry::new());
    let urls = ["http://127.0.0.1:1", "http://127.0.0.1:2"];
    let (host, _, _) = bed_over_models(Arc::clone(&registry), urls.into_iter().zip(models), None);
    let router = router(registry, Some(host));
    assert_eq!(router.model_of_an_unnamed_request().as_deref(), expected);
}

async fn assert_model_not_found(response: Response, model: &str) -> Case {
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert!(response.extensions().get::<ReportedRefusal>().is_some());
    let body = to_bytes(response.into_body(), 1024).await?;
    let value: Value = serde_json::from_slice(&body)?;
    assert_eq!(value["error"]["code"], "model_not_found");
    assert_eq!(
        value["error"]["message"],
        format!("No worker available for model '{model}'")
    );
    Ok(())
}

#[tokio::test]
async fn a_request_that_names_no_model_is_not_found_where_several_are_served() -> Case {
    let models = [Some(MODEL), Some("model-b")];
    assert_unnamed_model(models, None);
    let (a, hits_a) = fixed_status_upstream(&[GENERATE], StatusCode::OK).await;
    let (b, hits_b) = fixed_status_upstream(&[GENERATE], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over_models(
        Arc::clone(&registry),
        [a.as_str(), b.as_str()].into_iter().zip(models),
        None,
    );
    let router = router(registry, Some(host));
    assert_model_not_found(no_model(&router).await?, UNKNOWN_MODEL_ID).await?;
    assert_eq!(hits_a.load(Ordering::SeqCst), 0);
    assert_eq!(hits_b.load(Ordering::SeqCst), 0);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    Ok(())
}

#[tokio::test]
async fn one_model_beside_a_worker_with_none_takes_the_unnamed_request_alone() -> Case {
    let models = [Some(MODEL), None];
    assert_unnamed_model(models, Some(MODEL));
    let (a, hits_a) = fixed_status_upstream(&[GENERATE], StatusCode::OK).await;
    let (b, hits_b) = fixed_status_upstream(&[GENERATE], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, _, _) = bed_over_models(
        Arc::clone(&registry),
        [a.as_str(), b.as_str()].into_iter().zip(models),
        None,
    );
    let router = router(registry, Some(host));
    for _ in 0..6 {
        assert_eq!(no_model(&router).await?.status(), StatusCode::OK);
    }
    assert_eq!(hits_a.load(Ordering::SeqCst), 6);
    assert_eq!(hits_b.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn workers_with_no_model_are_asked_under_the_no_model_id() -> Case {
    assert_unnamed_model([None; 2], Some(UNKNOWN_MODEL_ID));
    let (url, hits) = fixed_status_upstream(&[GENERATE], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, _, _) = bed_over_models(Arc::clone(&registry), [(url.as_str(), None)], None);
    let router = router(registry, Some(host));
    assert_eq!(no_model(&router).await?.status(), StatusCode::OK);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn a_named_model_no_worker_serves_is_not_found_and_asks_nothing() -> Case {
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &["http://127.0.0.1:1"]);
    let router = router(registry, Some(host));
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let model = "unserved-model";
    let request = serde_json::from_value(json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}]
    }))?;
    let response = router
        .route_chat(None, &meta, Arc::new(request), model)
        .await;
    assert_model_not_found(response, model).await?;
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    Ok(())
}

#[tokio::test]
async fn a_transcription_under_the_host_goes_to_the_placed_worker() -> Case {
    let (url, hits) = fixed_status_upstream(&[TRANSCRIBE], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, workers) = bed_over(Arc::clone(&registry), &[&url]);
    let row = store.by_key(&url).ok_or("the worker has no row")?.id;
    registry.set_overload_shed(true);
    assert!(registry.set_worker_overloaded(&workers[0], true));
    let mut router = router(registry, Some(host));
    let start = Instant::now();

    let response = transcription(&router, MODEL, false, AUDIO_TYPE).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    assert!(store.completed_ok_since(GATEWAY, row, start));

    // A capped read that fails ends the lease by the answer the client reads.
    router.max_payload_size = 1;
    let start = Instant::now();
    let response = transcription(&router, MODEL, false, AUDIO_TYPE).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(hits.load(Ordering::SeqCst), 2);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    assert!(!store.completed_ok_since(GATEWAY, row, start));

    let response = transcription(&router, MODEL, false, "invalid").await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        extract_error_code_from_response(&response),
        "multipart_build_failed"
    );
    assert_eq!(hits.load(Ordering::SeqCst), 2);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    Ok(())
}

fn ranked_transcription_bed(ranked: &str, plain: Option<&str>) -> (Router, Arc<MemoryStore>) {
    let registry = Arc::new(WorkerRegistry::new());
    let urls_and_models = std::iter::once(ranked)
        .chain(plain)
        .map(|url| (url, Some(MODEL)));
    let (host, store, _) = bed_over_models(Arc::clone(&registry), urls_and_models, Some(ranked));
    (router(registry, Some(host)), store)
}

#[tokio::test]
async fn a_transcription_passes_over_a_data_parallel_worker() -> Case {
    let plain = "http://127.0.0.1:2";
    let (router, store) = ranked_transcription_bed("http://127.0.0.1:1", Some(plain));
    let row = store.by_key(plain).ok_or("the plain worker has no row")?;
    store
        .set_status(row.id, RowStatus::NotReady, row.revision)
        .ok_or("the plain worker's row stood")?;
    store.publish_now();

    // Only the ranked row can be picked. If committed, it would reach the
    // bad form and return its 400 before sending to the worker.
    let response = transcription(&router, MODEL, false, "invalid").await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        extract_error_code_from_response(&response),
        "no_available_workers"
    );
    assert!(response.extensions().get::<ReportedRefusal>().is_some());
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);

    let (ranked, ranked_hits) = fixed_status_upstream(&[TRANSCRIBE], StatusCode::OK).await;
    let (plain, plain_hits) = fixed_status_upstream(&[TRANSCRIBE], StatusCode::OK).await;
    let (router, store) = ranked_transcription_bed(&ranked, Some(&plain));
    let mut doc = PolicySpec::preset("round_robin")?;
    doc.generation += 1;
    store.apply(AppliedSpec::validate(&doc)?)?;
    for _ in 0..6 {
        assert_eq!(
            transcription(&router, MODEL, false, AUDIO_TYPE)
                .await
                .status(),
            StatusCode::OK
        );
    }
    assert_eq!(
        (
            ranked_hits.load(Ordering::SeqCst),
            plain_hits.load(Ordering::SeqCst)
        ),
        (0, 6)
    );
    Ok(())
}

#[tokio::test]
async fn a_pool_of_data_parallel_workers_alone_reads_the_routes_400() -> Case {
    let (mut router, store) = ranked_transcription_bed("http://127.0.0.1:1", None);
    let response = transcription(&router, MODEL, false, AUDIO_TYPE).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let hosted = to_bytes(response.into_body(), 1024).await?;
    let error: Value = serde_json::from_slice(&hosted)?;
    assert_eq!(error["error"]["code"], "dp_aware_not_supported");
    assert_eq!(
        error["error"]["message"],
        "/v1/audio/transcriptions does not yet support DP-aware workers"
    );
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);

    router.host = None;
    let response = transcription(&router, MODEL, false, AUDIO_TYPE).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(to_bytes(response.into_body(), 1024).await?, hosted);
    Ok(())
}

#[tokio::test]
async fn a_transcription_for_a_model_no_worker_serves_is_not_found() -> Case {
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &["http://127.0.0.1:1"]);
    let router = router(registry, Some(host));
    let model = "unserved-model";
    let recorder = PrometheusBuilder::new().build_recorder();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let response = transcription(&router, model, false, AUDIO_TYPE).await;
    assert_model_not_found(response, model).await?;
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    let rendered = recorder.handle().render();
    assert!(
        rendered.lines().any(|line| {
            line.starts_with("smg_router_upstream_responses_total{")
                && line.contains("status_code=\"404\"")
                && line.contains("error_code=\"model_not_found\"")
                && line.ends_with(" 1")
        }),
        "{rendered}"
    );
    Ok(())
}

#[tokio::test]
async fn a_streamed_transcription_holds_its_lease_to_its_last_frame() -> Case {
    let (url, frames) = gated_stream(TRANSCRIBE).await?;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &[&url]);
    let row = store.by_key(&url).ok_or("the worker has no row")?.id;
    let router = router(registry, Some(host));

    let response = timeout(FRAME_WAIT, transcription(&router, MODEL, true, AUDIO_TYPE)).await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 1);
    let mut body = response.into_body().into_data_stream();
    frames.send(Bytes::from_static(FIRST_FRAME)).await?;
    assert_eq!(next_frame(&mut body).await?.as_ref(), FIRST_FRAME);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 1);
    let last_frame = Instant::now();
    drop(frames);
    assert!(timeout(FRAME_WAIT, body.next()).await?.is_none());
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    assert!(store.completed_ok_since(GATEWAY, row, last_frame));
    Ok(())
}

#[tokio::test]
async fn a_body_that_would_stream_takes_the_buffered_path_under_the_host() -> Case {
    let registry = Arc::new(WorkerRegistry::new());
    let (host, _, _) = bed_over(Arc::clone(&registry), &["http://127.0.0.1:1"]);
    let hosted = router(registry, Some(host));
    let recorder = PrometheusBuilder::new().build_recorder();
    let request = {
        let _guard = metrics::set_default_local_recorder(&recorder);
        hosted
            .route_streaming_request(streamed_request(&[GENERATE_BODY]), GENERATE, false)
            .await
    }
    .err()
    .ok_or("the router did not hand back the request")?;
    assert_eq!(
        to_bytes(request.into_body(), 1024).await?.as_ref(),
        GENERATE_BODY
    );
    assert!(recorder.handle().render().lines().any(|line| {
        line == r#"smg_router_request_body_path_total{path="buffered",reason="placed_by_controller"} 1"#
    }));
    Ok(())
}

#[tokio::test]
async fn the_classic_policy_serves_the_three_shapes_without_a_host() -> Case {
    let (url, hits) = fixed_status_upstream(&[GENERATE, TRANSCRIBE], StatusCode::OK).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (_, store, _) = bed_over(Arc::clone(&registry), &[&url]);
    let router = router(registry, None);
    assert_eq!(no_model(&router).await?.status(), StatusCode::OK);
    assert_eq!(
        transcription(&router, MODEL, false, AUDIO_TYPE)
            .await
            .status(),
        StatusCode::OK
    );
    let streamed = router
        .route_streaming_request(streamed_request(&[GENERATE_BODY]), GENERATE, false)
        .await
        .map_err(|_| "the router buffered the request")?;
    assert_eq!(streamed.status(), StatusCode::OK);
    assert_eq!(hits.load(Ordering::SeqCst), 3);
    assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    Ok(())
}

#[tokio::test]
async fn a_request_for_a_model_no_binding_covers_reads_declined_with_a_line() -> Case {
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = bed_over(Arc::clone(&registry), &["http://127.0.0.1:1"]);
    let mut doc = PolicySpec::preset("random")?;
    doc.generation += 1;
    doc.bindings[0].matcher.model = "other-model".into();
    store.apply(AppliedSpec::validate(&doc)?)?;

    let answer = chat(&router(registry, Some(host))).await?;
    assert_eq!(answer.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(answer.extensions().get::<ReportedRefusal>().is_some());
    assert_eq!(
        placement::refusal_reason(&answer),
        Some("the policy declined at binding")
    );
    let body = to_bytes(answer.into_body(), 1024).await?;
    let error: Value = serde_json::from_slice(&body)?;
    assert_eq!(error["error"]["code"], "policy_declined");
    assert_eq!(
        error["error"]["message"],
        format!("The policy declined the request for model '{MODEL}'")
    );
    Ok(())
}

#[tokio::test]
async fn a_keyed_request_keeps_its_worker_through_the_router() -> Case {
    const KEY_HEADER: &str = "x-conversation";
    // The key's one source by row: a request header, then the request's own id.
    for by_rid in [false, true] {
        let (a, hits_a) = fixed_status_upstream(&[GENERATE], StatusCode::OK).await;
        let (b, hits_b) = fixed_status_upstream(&[GENERATE], StatusCode::OK).await;
        let registry = Arc::new(WorkerRegistry::new());
        let (host, store, _) = bed_over(Arc::clone(&registry), &[&a, &b]);
        let config = RouterConfig {
            policy_controller: true,
            policy: PolicyConfig::ConsistentHashing,
            ..RouterConfig::default()
        };
        let mut document = config.policy_document()?;
        document.generation += 1;
        document.bindings[0].routing_key = Some(RoutingKeySpec {
            sources: vec![KeySourceSpec {
                header: (!by_rid).then(|| KEY_HEADER.to_owned()),
                rid: by_rid,
                ..KeySourceSpec::default()
            }],
            strip_lineage: false,
        });
        store.apply(config.hosted_policy(&document)?)?;
        let router = router(registry, Some(host));
        let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
        let hits = || [hits_a.load(Ordering::SeqCst), hits_b.load(Ordering::SeqCst)];

        // One key sixteen times, then sixteen keys once each.
        for turn in 0..32 {
            let key = if turn < 16 {
                "conv".to_owned()
            } else {
                format!("conv-{turn}")
            };
            let mut headers = HeaderMap::new();
            let mut body = json!({"text": "hi"});
            if by_rid {
                body["rid"] = json!(key);
            } else {
                headers.insert(KEY_HEADER, HeaderValue::from_str(&key)?);
            }
            let request = serde_json::from_value(body)?;
            let response = router
                .route_generate(Some(&headers), &meta, request, MODEL)
                .await;
            assert_eq!(response.status(), StatusCode::OK);
            if turn == 15 {
                assert!(hits().contains(&16), "by_rid {by_rid}: {:?}", hits());
            }
        }
        assert!(
            hits().iter().all(|&hit| hit > 0),
            "by_rid {by_rid}: {:?}",
            hits()
        );
    }
    Ok(())
}
