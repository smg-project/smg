//! The HTTP pair router over a store and a registry of loopback workers.

use std::{error::Error, time::Duration};

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use serde_json::json;
use smg_policy_controller::{
    store::{
        GatewayId, HealthSink, LeasesStore, MemoryStore, PairingDescriptor, WorkerFacts,
        WorkerStatus as RowStatus, WorkersStore,
    },
    view::{Role, WireModes},
};
use smg_worker::identity::basic::ConnectionMode;

use super::{
    tests::{create_test_pd_router, spawn_cut_body_stub, spawn_delayed_stub, spawn_recording_stub},
    *,
};
use crate::{
    config::{RouterConfig, RoutingMode},
    policy_controller::{host::testing::MODEL, start},
    routers::grpc::backend_client::BasicWorkerBuilder,
    tenant::TenantKey,
};

type Case<T = ()> = Result<T, Box<dyn Error>>;

/// A host, its store, and the registry's workers of each side in its URLs'
/// order: the prefill workers, then the decode workers.
type PairBed = (Host, Arc<MemoryStore>, [Vec<Arc<dyn Worker>>; 2]);

/// A host over the store a prefill and decode gateway starts, with one ready
/// row and one registry worker a URL of each side: the prefill workers, then
/// the decode workers, each in its URLs' order.
fn pair_bed_over(
    registry: Arc<WorkerRegistry>,
    prefill_urls: &[&str],
    decode_urls: &[&str],
) -> PairBed {
    pair_bed_of(|_, worker| worker, registry, prefill_urls, decode_urls)
}

/// [`pair_bed_over`] with every worker of `runtime`: a pair router picks its
/// dispatch by its prefill worker's runtime.
fn pair_bed_running(
    runtime: RuntimeType,
    registry: Arc<WorkerRegistry>,
    prefill_urls: &[&str],
    decode_urls: &[&str],
) -> PairBed {
    let running = |_: &str, worker: BasicWorkerBuilder| worker.runtime_type(runtime);
    pair_bed_of(running, registry, prefill_urls, decode_urls)
}

/// [`pair_bed_over`] with each registry worker as `shaped` leaves it, by
/// its URL. The rows are written the same for every shape.
fn pair_bed_of(
    shaped: impl Fn(&str, BasicWorkerBuilder) -> BasicWorkerBuilder,
    registry: Arc<WorkerRegistry>,
    prefill_urls: &[&str],
    decode_urls: &[&str],
) -> PairBed {
    let config = RouterConfig {
        policy_controller: true,
        mode: RoutingMode::PrefillDecode {
            prefill_urls: Vec::new(),
            decode_urls: Vec::new(),
            prefill_policy: None,
            decode_policy: None,
        },
        ..RouterConfig::default()
    };
    let store = start(&config)
        .expect("the start takes the routing")
        .expect("the controller was picked");
    let pairing = Arc::new(PairingDescriptor {
        protocol: Some("pair-wire".to_owned()),
        ..PairingDescriptor::default()
    });
    let roles = [
        (Role::Prefill, WorkerType::Prefill, prefill_urls),
        (Role::Decode, WorkerType::Decode, decode_urls),
    ];
    let sides = roles.map(|(role, worker_type, urls)| {
        let side = urls.iter().map(|&url| {
            let builder = BasicWorkerBuilder::new(url)
                .model(ModelCard::new(MODEL))
                .worker_type(worker_type)
                .status(WorkerStatus::Ready);
            let worker: Arc<dyn Worker> = Arc::new(shaped(url, builder).build());
            let facts = WorkerFacts {
                key: worker.url().to_string(),
                engine: url.to_string(),
                incarnation: 1,
                role,
                wire_modes: WireModes::HTTP,
                models: vec![MODEL.to_string()],
                dp_size: 1,
                max_running: Some(64),
                pairing: Some(Arc::clone(&pairing)),
                ..WorkerFacts::default()
            };
            let id = store.upsert(facts).id();
            let row = WorkersStore::get(&*store, id).expect("the row");
            if row.status != RowStatus::Ready {
                store
                    .set_status(id, RowStatus::Ready, row.revision)
                    .expect("the row stands");
            }
            registry
                .register(Arc::clone(&worker))
                .expect("the worker registers");
            worker
        });
        side.collect()
    });
    store.publish_now();
    (Host::at_keys(Arc::clone(&store), registry), store, sides)
}

const GATEWAY: GatewayId = GatewayId::LOCAL;

fn router(worker_registry: Arc<WorkerRegistry>, host: Host) -> PDRouter {
    PDRouter {
        worker_registry,
        host: Some(Arc::new(host)),
        ..create_test_pd_router()
    }
}

async fn generate(router: &PDRouter) -> Case<Response> {
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let request = serde_json::from_value(json!({"text": "hi"}))?;
    Ok(router.route_generate(None, &meta, request, MODEL).await)
}

async fn chat(router: &PDRouter) -> Case<Response> {
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
    }))?;
    Ok(router
        .route_chat(None, &meta, Arc::new(request), MODEL)
        .await)
}

/// A pair as the router's attempt asks for it, with no admission.
async fn sent_pair(host: &Host) -> Case<Sent> {
    let id = host::mark();
    let ask = Ask {
        phase: Phase::Prefill,
        ..Ask::of(&id, MODEL)
    };
    let placed = host.place_pair(&ask, None, None).await;
    let (placed, _guard) = placed.map_err(|unplaced| format!("{unplaced:?}"))?;
    Ok(Sent::from(placed))
}

/// The gateway's live leases, read until they number `expected` or the
/// patience ends: a report is applied by whoever holds the lease's file.
async fn live(store: &MemoryStore, expected: usize) -> usize {
    for _ in 0..500 {
        if LeasesStore::live(store, GATEWAY) == expected {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    LeasesStore::live(store, GATEWAY)
}

#[tokio::test]
async fn a_hosted_pair_is_sent_to_the_prefill_and_the_decode_the_controller_placed() -> Case {
    let (prefill, prefill_seen) = spawn_recording_stub("{}").await;
    let (decode, decode_seen) = spawn_recording_stub(r#"{"text":"done"}"#).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = pair_bed_over(Arc::clone(&registry), &[&prefill], &[&decode]);
    let router = router(registry, host);

    let response = generate(&router).await?;

    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert_eq!(prefill_seen.lock().map_err(|e| e.to_string())?.len(), 1);
    assert_eq!(decode_seen.lock().map_err(|e| e.to_string())?.len(), 1);
    // Both sides ended with the answer: no lease is left behind.
    assert_eq!(live(&store, 0).await, 0);
    Ok(())
}

/// The store's rows decide, not the registry's workers: with the decode's
/// row gone the request is refused, though both workers stand registered
/// and the classic selection would pair them.
#[tokio::test]
async fn a_pair_the_store_cannot_make_is_refused_though_the_registry_holds_one() -> Case {
    let (prefill, prefill_seen) = spawn_recording_stub("{}").await;
    let (decode, decode_seen) = spawn_recording_stub(r#"{"text":"done"}"#).await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = pair_bed_over(Arc::clone(&registry), &[&prefill], &[&decode]);
    let row = store
        .by_key(&decode)
        .ok_or("the decode worker had no row")?;
    store
        .remove(row.id, row.revision)
        .ok_or("the decode worker's row stood")?;
    store.publish_now();
    let router = router(registry, host);

    let response = generate(&router).await?;

    assert!(!response.status().is_success(), "{}", response.status());
    assert!(prefill_seen.lock().map_err(|e| e.to_string())?.is_empty());
    assert!(decode_seen.lock().map_err(|e| e.to_string())?.is_empty());
    assert_eq!(live(&store, 0).await, 0);
    Ok(())
}

/// A pair whose answer a retry may replace names both its rows to the next
/// ask where the answer names no side, each once. Its lease is given up
/// with the answer.
#[tokio::test]
async fn a_retryable_answer_that_names_no_side_names_both_rows_as_tried() -> Case {
    let (prefill, _) = spawn_recording_stub("{}").await;
    let (decode, _) = spawn_recording_stub("{}").await;
    let registry = Arc::new(WorkerRegistry::new());
    let (host, store, _) = pair_bed_over(Arc::clone(&registry), &[&prefill], &[&decode]);
    let sent = sent_pair(&host).await?;
    let rows = [Some(sent.id()), sent.decode_id()];
    let tried = parking_lot::Mutex::new(Vec::new());
    let refusal = error::service_unavailable("worker_unavailable", "a worker is down");

    let response = PDRouter::end_placed(Some(sent), &tried, refusal);

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let named: Vec<_> = tried.lock().iter().copied().map(Some).collect();
    assert_eq!(named, rows);
    assert_eq!(live(&store, 0).await, 0);
    Ok(())
}

/// An answer whose failure the ledger gave to one side names that row alone
/// as tried: the partner stays as selectable as it was.
#[tokio::test]
async fn an_answer_that_names_a_failed_side_names_that_row_alone_as_tried() -> Case {
    for failed in [0, 1] {
        let (prefill, _) = spawn_recording_stub("{}").await;
        let (decode, _) = spawn_recording_stub("{}").await;
        let registry = Arc::new(WorkerRegistry::new());
        let (host, store, sides) = pair_bed_over(Arc::clone(&registry), &[&prefill], &[&decode]);
        let sent = sent_pair(&host).await?;
        let rows = [Some(sent.id()), sent.decode_id()];
        let tried = parking_lot::Mutex::new(Vec::new());
        let mut refusal = error::internal_error("worker_failed", "the worker failed the request");
        mark_failed_side(&mut refusal, sides[failed][0].as_ref());

        let _ = PDRouter::end_placed(Some(sent), &tried, refusal);

        let named: Vec<_> = tried.lock().iter().copied().map(Some).collect();
        assert_eq!(named, [rows[failed]], "side {failed}");
        assert_eq!(live(&store, 0).await, 0);
    }
    Ok(())
}

/// The tried row is the one the request's ledger closed, which is not
/// always the one that answered. With two prefills and one decode, the
/// decode's own failure leaves no pair without it: the ledger closes the
/// prefill, and the retry takes the other prefill with the same decode.
#[tokio::test]
async fn the_tried_row_is_the_side_the_ledger_closed() -> Case {
    let (first, _) = spawn_recording_stub("{}").await;
    let (second, _) = spawn_recording_stub("{}").await;
    let (decode, _) = spawn_recording_stub("{}").await;
    let registry = Arc::new(WorkerRegistry::new());
    let bed = pair_bed_over(Arc::clone(&registry), &[&first, &second], &[&decode]);
    let (host, store, [_, decodes]) = bed;
    let sent = sent_pair(&host).await?;
    let prefill_row = sent.id();
    let pair = [Arc::clone(sent.worker()), Arc::clone(&decodes[0])];
    let router = router(registry, host);
    let ledger = AttemptLedger::default();
    let mut answer = error::internal_error("worker_failed", "the decode failed the request");
    mark_upstream_answer(&mut answer);
    answer.extensions_mut().insert(OwnFailure(PdSide::Decode));

    let answer = router.settle_attempt(&ledger, MODEL, [&pair[0], &pair[1]], answer);
    let tried = parking_lot::Mutex::new(Vec::new());
    let _ = PDRouter::end_placed(Some(sent), &tried, answer);

    assert!(!ledger.admits(pair[0].as_ref()), "the prefill stayed open");
    assert!(ledger.admits(pair[1].as_ref()), "the one decode was closed");
    assert_eq!(*tried.lock(), [prefill_row]);
    assert_eq!(live(&store, 0).await, 0);
    Ok(())
}

/// A decode that waits on its prefill is said dispatched with it. The
/// prefill here holds its answer while a sweep runs past the dispatch
/// deadline: the sweep ends no side, and the pair is then served.
#[tokio::test]
async fn a_decode_that_waits_on_its_prefill_outlives_the_dispatch_deadline() -> Case {
    let hold = Duration::from_millis(400);
    let (prefill, prefill_seen) = spawn_delayed_stub(StatusCode::OK, json!({}), hold).await;
    let (decode, decode_seen) = spawn_recording_stub(r#"{"object":"chat.completion"}"#).await;
    let registry = Arc::new(WorkerRegistry::new());
    let bed = pair_bed_running(
        RuntimeType::Vllm,
        Arc::clone(&registry),
        &[&prefill],
        &[&decode],
    );
    let (host, store, _) = bed;
    let router = router(registry, host);

    let sweep = async {
        // The prefill holds its answer: its request is in, the decode's is not.
        for _ in 0..200 {
            if prefill_seen.lock().is_ok_and(|seen| !seen.is_empty()) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let waits = decode_seen.lock().is_ok_and(|seen| seen.is_empty());
        let past = Instant::now() + Duration::from_secs(6);
        (waits, LeasesStore::expire(&*store, past).len())
    };
    let (response, (waits, swept)) = tokio::join!(chat(&router), sweep);

    assert_eq!(response?.status(), StatusCode::OK);
    assert!(waits, "the decode went out ahead of its prefill's answer");
    assert_eq!(swept, 0, "the sweep ended a side of the pair");
    assert_eq!(decode_seen.lock().map_err(|e| e.to_string())?.len(), 1);
    assert_eq!(live(&store, 0).await, 0);
    Ok(())
}

/// A retry is not placed on a worker that answered the request
/// definitively. The pair's commit asks the request's ledger of both
/// workers, so a pick that names the closed decode is passed over for the
/// other one, whichever the pick lists first.
#[tokio::test]
async fn a_retry_is_not_placed_on_a_worker_the_ledger_closed() -> Case {
    for closed in [0, 1] {
        let (prefill, _) = spawn_recording_stub("{}").await;
        let (first, _) = spawn_recording_stub("{}").await;
        let (second, _) = spawn_recording_stub("{}").await;
        let registry = Arc::new(WorkerRegistry::new());
        let bed = pair_bed_over(Arc::clone(&registry), &[&prefill], &[&first, &second]);
        let (host, store, [prefills, decodes]) = bed;
        let ledger = AttemptLedger::default();
        let mut failed = error::internal_error("worker_failed", "the worker failed the request");
        mark_failed_side(&mut failed, decodes[closed].as_ref());
        ledger.settle(
            &mut failed,
            [&prefills[0], &decodes[closed]],
            decodes.iter(),
        );
        let router = router(registry, host);
        let host = router.host.as_deref().ok_or("the router holds the host")?;
        let context = PDRequestContext {
            route: "/generate",
            batch_size: None,
            is_stream: false,
            return_logprob: false,
            model_id: MODEL,
            headers: None,
            rid: None,
        };
        let tried = parking_lot::Mutex::new(Vec::new());
        for _ in 0..4 {
            let placed = router
                .place_pair_attempt(host, &context, 0, None, &tried, &ledger)
                .await;
            let (sent, (_, decode), _guards) =
                placed.map_err(|refusal| format!("refused: {}", refusal.status()))?;
            assert_eq!(decode.url(), decodes[1 - closed].url(), "closed {closed}");
            drop(sent);
        }
        assert_eq!(live(&store, 0).await, 0);
    }
    Ok(())
}

/// A streamed answer its decode worker cuts is told as the cut it was. The
/// relay answers the client with an error frame and a clean end, and writes
/// the cut on the placement's mark: the decode's row hears no success. A
/// whole answer over the same path is the control.
#[tokio::test]
async fn a_streamed_answer_its_decode_cuts_is_told_as_no_success() -> Case {
    let whole = spawn_recording_stub(r#"{"text":"done"}"#).await.0;
    for (decode, served) in [(whole, true), (spawn_cut_body_stub().await, false)] {
        let (prefill, _) = spawn_recording_stub("{}").await;
        let registry = Arc::new(WorkerRegistry::new());
        let (host, store, _) = pair_bed_over(Arc::clone(&registry), &[&prefill], &[&decode]);
        let row = store
            .by_key(&decode)
            .ok_or("the decode worker had no row")?;
        let router = router(registry, host);
        let start = Instant::now();
        let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
        let request = serde_json::from_value(json!({"text": "hi", "stream": true}))?;

        let response = router.route_generate(None, &meta, request, MODEL).await;

        assert_eq!(response.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(response.into_body(), usize::MAX).await;
        assert_eq!(live(&store, 0).await, 0);
        let ok = HealthSink::completed_ok_since(&*store, GATEWAY, row.id, start);
        assert_eq!(ok, served, "served whole: {served}");
    }
    Ok(())
}

/// A streamed pair's prefill is drained beside the answer, and the drain
/// tells the prefill's end. One that breaks off is told as no success,
/// though the answer went out whole. A whole drain is the control.
#[tokio::test]
async fn a_drained_prefill_that_breaks_off_is_told_as_no_success() -> Case {
    let hold = Duration::from_millis(300);
    let whole = spawn_recording_stub("{}").await.0;
    for (prefill, served) in [(whole, true), (spawn_cut_body_stub().await, false)] {
        let reply = json!({"text": "done"});
        let (decode, _) = spawn_delayed_stub(StatusCode::OK, reply, hold).await;
        let registry = Arc::new(WorkerRegistry::new());
        let (host, store, _) = pair_bed_over(Arc::clone(&registry), &[&prefill], &[&decode]);
        let row = store
            .by_key(&prefill)
            .ok_or("the prefill worker had no row")?;
        let router = router(registry, host);
        let start = Instant::now();
        let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
        let request = serde_json::from_value(json!({"text": "hi", "stream": true}))?;

        let response = router.route_generate(None, &meta, request, MODEL).await;

        assert_eq!(response.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(response.into_body(), usize::MAX).await;
        assert_eq!(live(&store, 0).await, 0);
        let ok = HealthSink::completed_ok_since(&*store, GATEWAY, row.id, start);
        assert_eq!(ok, served, "drained whole: {served}");
    }
    Ok(())
}

/// A prefill that is read whole ends where its guard drops, ahead of the
/// answer its decode writes. The decode here holds its answer, and the
/// prefill's row hears its success well inside that hold. Both shapes: the
/// two sides sent together, and the decode sent behind its prefill.
#[tokio::test]
async fn a_prefill_read_whole_ends_ahead_of_its_decodes_answer() -> Case {
    let hold = Duration::from_millis(600);
    for runtime in [RuntimeType::default(), RuntimeType::Vllm] {
        let (prefill, _) = spawn_recording_stub("{}").await;
        let reply = json!({"object": "chat.completion"});
        let (decode, decode_seen) = spawn_delayed_stub(StatusCode::OK, reply, hold).await;
        let registry = Arc::new(WorkerRegistry::new());
        let bed = pair_bed_running(runtime, Arc::clone(&registry), &[&prefill], &[&decode]);
        let (host, store, _) = bed;
        let row = store
            .by_key(&prefill)
            .ok_or("the prefill worker had no row")?;
        let router = router(registry, host);
        let start = Instant::now();

        let early = async {
            // The patience ends inside the decode's hold: a prefill that
            // ends with the answer is read as not ended here.
            for _ in 0..100 {
                let held = decode_seen.lock().is_ok_and(|seen| !seen.is_empty());
                if held && HealthSink::completed_ok_since(&*store, GATEWAY, row.id, start) {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            false
        };
        let (response, ended) = tokio::join!(chat(&router), early);

        assert_eq!(response?.status(), StatusCode::OK, "{runtime:?}");
        assert!(
            ended,
            "{runtime:?}: the prefill stayed booked behind its decode"
        );
        assert_eq!(live(&store, 0).await, 0);
    }
    Ok(())
}

/// A pair of another wire is not placed: the commit asks each side's HTTP
/// pool of its worker, so a pick that names the gRPC decode is passed over
/// for the HTTP one, whichever the pick lists first.
#[tokio::test]
async fn a_pair_of_another_wire_is_passed_over_at_its_commit() -> Case {
    for other in [0, 1] {
        let (prefill, _) = spawn_recording_stub("{}").await;
        let (first, _) = spawn_recording_stub("{}").await;
        let (second, _) = spawn_recording_stub("{}").await;
        let urls = [first.as_str(), second.as_str()];
        let grpc = |url: &str, worker: BasicWorkerBuilder| {
            if url == urls[other] {
                worker.connection_mode(ConnectionMode::Grpc)
            } else {
                worker
            }
        };
        let registry = Arc::new(WorkerRegistry::new());
        let bed = pair_bed_of(grpc, Arc::clone(&registry), &[&prefill], &urls);
        let (host, store, [_, decodes]) = bed;
        let router = router(registry, host);
        let host = router.host.as_deref().ok_or("the router holds the host")?;
        let context = PDRequestContext {
            route: "/generate",
            batch_size: None,
            is_stream: false,
            return_logprob: false,
            model_id: MODEL,
            headers: None,
            rid: None,
        };
        let tried = parking_lot::Mutex::new(Vec::new());
        let ledger = AttemptLedger::default();
        for _ in 0..4 {
            let placed = router
                .place_pair_attempt(host, &context, 0, None, &tried, &ledger)
                .await;
            let (sent, (_, decode), _guards) =
                placed.map_err(|refusal| format!("refused: {}", refusal.status()))?;
            assert_eq!(
                decode.url(),
                decodes[1 - other].url(),
                "the gRPC decode {other}"
            );
            drop(sent);
        }
        assert_eq!(live(&store, 0).await, 0);
    }
    Ok(())
}
