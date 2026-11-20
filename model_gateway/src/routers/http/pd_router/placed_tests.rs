//! The HTTP pair router over a store and a registry of loopback workers.

use std::{error::Error, time::Duration};

use serde_json::json;
use smg_policy_controller::store::{GatewayId, HealthSink, LeasesStore, MemoryStore, WorkersStore};

use super::{
    tests::{create_test_pd_router, spawn_cut_body_stub, spawn_delayed_stub, spawn_recording_stub},
    *,
};
use crate::{
    policy_controller::host::testing::{pair_bed_over, pair_bed_running, MODEL},
    tenant::TenantKey,
};

type Case<T = ()> = Result<T, Box<dyn Error>>;

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

/// A definitive answer names its side, and that row alone is tried: the
/// partner stays as selectable as it was.
#[tokio::test]
async fn a_definitive_answer_names_its_own_side_alone_as_tried() -> Case {
    for (side, row) in [(PdSide::Prefill, 0), (PdSide::Decode, 1)] {
        let (prefill, _) = spawn_recording_stub("{}").await;
        let (decode, _) = spawn_recording_stub("{}").await;
        let registry = Arc::new(WorkerRegistry::new());
        let (host, store, _) = pair_bed_over(Arc::clone(&registry), &[&prefill], &[&decode]);
        let sent = sent_pair(&host).await?;
        let rows = [Some(sent.id()), sent.decode_id()];
        let tried = parking_lot::Mutex::new(Vec::new());
        let mut refusal = error::internal_error("worker_failed", "the worker failed the request");
        refusal.extensions_mut().insert(OwnFailure(side));

        let _ = PDRouter::end_placed(Some(sent), &tried, refusal);

        let named: Vec<_> = tried.lock().iter().copied().map(Some).collect();
        assert_eq!(named, [rows[row]], "{side:?}");
        assert_eq!(live(&store, 0).await, 0);
    }
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
