//! A sticky key's PD bootstrap rooms under `--grouped-dp-sticky`, through the
//! gRPC router to a scripted TokenSpeed prefill/decode pair that records every
//! request it gets.
//!
//! The prefill worker stands in for a grouped ZMQ prefill worker of
//! [`ENGINES`] engines, as no CPU mock speaks TokenSpeed's ZMQ wire: it is a
//! gRPC worker given that `dp_size` and no rank (registration gives a
//! `dp_size` without a rank only to a ZMQ worker). A gRPC connection counts
//! no engines, so the pin sizes the ranks by the `dp_size`.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use common::{
    engine_answers::TokenSpeedAnswers,
    grpc_harness::{
        grpc_router, grpc_router_config, grpc_test_context_with_tokenizer, grpc_worker,
        serve_engine,
    },
    scripted_tokenizer::ScriptedTokenizer,
    scripted_worker::ScriptedWorker,
};
use openai_protocol::{completion::CompletionRequest, model_card::ModelCard};
use smg::{
    config::GroupedDpSticky,
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, Worker, WorkerType},
};
use smg_grpc_client::tokenspeed_scheduler::tokenspeed_proto::GenerateRequest;

use crate::common;

const MODEL: &str = "pd-grouped-dp-room-model";
/// The prefill worker's engines: the modulus a pinned room carries its rank by.
const ENGINES: i64 = 4;

/// What a scripted engine records: every request it got.
type Requests = Arc<Mutex<Vec<GenerateRequest>>>;

/// A scripted TokenSpeed engine answering one token, served in-process; its
/// port and its record.
async fn scripted_engine() -> (u16, Requests) {
    let requests = Requests::default();
    let port = serve_engine(
        ScriptedWorker {
            output_tokens: 1,
            finish_reasons: vec!["stop"],
            generation: Default::default(),
            requests: Arc::clone(&requests),
        }
        .into_server(),
    )
    .await;
    (port, requests)
}

/// The bootstrap rooms of the requests `requests` recorded, sorted, and the
/// record emptied.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn take_rooms(requests: &Requests) -> Vec<i64> {
    let mut rooms: Vec<i64> = std::mem::take(&mut *requests.lock().unwrap())
        .into_iter()
        .map(|request| {
            request
                .kv_bootstrap_info
                .expect("a PD request carries a room")
                .bootstrap_room
        })
        .collect();
    rooms.sort_unstable();
    rooms
}

/// A PD router under `sticky` over a scripted prefill worker grouped with
/// [`ENGINES`] engines and a scripted decode worker; their records.
async fn pd_router(sticky: GroupedDpSticky) -> (Box<dyn RouterTrait>, Requests, Requests) {
    let (prefill_port, prefill) = scripted_engine().await;
    let (decode_port, decode) = scripted_engine().await;
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .random_policy()
        .grouped_dp_sticky(sticky)
        .build_unchecked();
    let tokenizer = Arc::new(ScriptedTokenizer::new("ok"));
    let app_context = grpc_test_context_with_tokenizer(config, MODEL, tokenizer).await;
    let worker = |port, worker_type| {
        grpc_worker(
            port,
            worker_type,
            RuntimeType::TokenSpeed,
            ModelCard::new(MODEL),
        )
    };
    let workers: Vec<Arc<dyn Worker>> = vec![
        Arc::new(
            worker(prefill_port, WorkerType::Prefill)
                .zmq_engine_group(ENGINES as usize)
                .build(),
        ),
        Arc::new(worker(decode_port, WorkerType::Decode).build()),
    ];
    (grpc_router(&app_context, workers).await, prefill, decode)
}

/// Complete a prompt in `n` samples under routing key `key`.
#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn complete(router: &dyn RouterTrait, key: &str, n: u32) {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        "x-smg-routing-key",
        http::HeaderValue::from_str(key).unwrap(),
    );
    let request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": "Hello world",
        "n": n,
        "max_tokens": 1,
    }))
    .unwrap();
    let tenant = TenantRequestMeta::new(TenantKey::new("pd-grouped-dp-room-tenant"));
    let response = router
        .route_completion(Some(&headers), &tenant, request, MODEL)
        .await;
    assert_eq!(response.status(), http::StatusCode::OK, "{key}, n={n}");
}

/// Under `hash` every room a sticky key's PD requests carry, a single
/// sample's and each sample's of an n>1 request, is in the residue class of
/// the key's prefill rank `xxh3(key) % 4`, and the decode side gets the same
/// rooms. Under `off` (today's behavior) one key's rooms fall on several
/// ranks.
#[tokio::test]
async fn a_sticky_keys_pd_rooms_carry_its_prefill_rank() {
    let (router, prefill, decode) = pd_router(GroupedDpSticky::Hash).await;
    for i in 0..8 {
        let key = format!("conv-{i}");
        let rank = (xxhash_rust::xxh3::xxh3_64(key.as_bytes()) % ENGINES as u64) as i64;
        for n in [1, 4] {
            complete(router.as_ref(), &key, n).await;
            let rooms = take_rooms(&prefill);
            assert_eq!(rooms.len(), n as usize, "{key}, n={n}: a room per sample");
            assert_eq!(take_rooms(&decode), rooms, "{key}, n={n}: decode's rooms");
            for room in rooms {
                assert_eq!(room % ENGINES, rank, "{key}, n={n}: room {room}");
            }
        }
    }

    let (router, prefill, _decode) = pd_router(GroupedDpSticky::Off).await;
    for _ in 0..32 {
        complete(router.as_ref(), "conv-0", 1).await;
    }
    let ranks: HashSet<i64> = take_rooms(&prefill)
        .into_iter()
        .map(|room| room % ENGINES)
        .collect();
    assert!(ranks.len() > 1, "off: one key's rooms on ranks {ranks:?}");
}
