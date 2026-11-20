//! The HTTP pair router over a store and a registry of loopback workers.

use std::{error::Error, time::Duration};

use serde_json::json;
use smg_policy_controller::store::{GatewayId, LeasesStore, MemoryStore, WorkersStore};

use super::{
    tests::{create_test_pd_router, spawn_recording_stub},
    *,
};
use crate::{
    policy_controller::host::testing::{pair_bed_over, MODEL},
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
