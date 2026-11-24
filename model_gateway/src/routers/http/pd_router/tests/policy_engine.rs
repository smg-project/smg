//! The pair route where the policy engine places the pair. Each case's
//! gateway starts the engine by its own start (`policy_controller: on`),
//! over a table that holds a scorer of the case's own and under a supplied
//! document that names that scorer on the decode side.

use std::sync::{atomic::AtomicBool, Mutex, PoisonError};

use smg_policy_controller::{
    document::{DecodePick, PolicySpec},
    engine::{Candidate, Decision, RequestFacts, Scorer, Target, TieBand, WorkerId},
    stages::{Lifecycle, StageRegistry},
    tracking::{EventSet, Lease, Outcome, ReleaseReason, Side},
};

use super::*;
use crate::{
    app_context::AppContextBuilder,
    config::{RouterConfig, RoutingMode},
    routers::common::snapshot::WorkerMonitor,
};

/// The case's stage, as its table and its document name it.
const STAGE: &str = "of-the-case";

/// A pair router of a gateway that runs the policy controller, and the load
/// monitor its engine reads. The gateway's own start builds the engine over
/// `stages`, under the stock pair document with [`STAGE`] alone ranking its
/// decode side. `decode` says when that side is picked: with the prefill,
/// or behind it.
fn pair_router_under(stages: &StageRegistry, decode: DecodePick) -> (PDRouter, WorkerMonitor) {
    let document = PolicySpec::pair_presets("round_robin", "round_robin");
    let mut document = document.expect("the pair presets");
    document.generation = 1;
    let side = document.profiles.get_mut("decode");
    let side = side.expect("the decode side's profile");
    side.score = vec![STAGE.into()];
    side.pick = "best".into();
    document.plans.pd_pair.decode = decode;
    let config = RouterConfig {
        mode: RoutingMode::PrefillDecode {
            prefill_urls: Vec::new(),
            decode_urls: Vec::new(),
            prefill_policy: None,
            decode_policy: None,
        },
        policy_controller: true,
        policy_spec: Some(document),
        ..RouterConfig::default()
    };
    let router = create_test_pd_router();
    let monitor = WorkerMonitor::new(
        Arc::clone(&router.worker_registry),
        Arc::clone(&router.policy_registry),
        1,
        false,
        true,
    );
    AppContextBuilder::start_policy_engine_by_hand(
        &config,
        Some(stages),
        &router.policy_registry,
        &router.worker_registry,
        &monitor,
    )
    .expect("the table holds the stage the document names");
    (router, monitor)
}

/// The table of a host: the stock stages and the case's scorer.
fn table_with(scorer: impl Fn() -> Box<dyn Scorer> + Send + Sync + 'static) -> StageRegistry {
    let mut stages = StageRegistry::builtin();
    stages.scorer(STAGE, scorer).expect("a free name");
    stages
}

// ---- the rank on the wire ------------------------------------------------------------------

/// The ranks of the decode engine below.
const RANKS: usize = 4;

/// A scorer that puts first the row whose rank the case wants next.
#[derive(Debug)]
struct WantsRank(Arc<AtomicUsize>);

impl Lifecycle for WantsRank {}

impl Scorer for WantsRank {
    fn name(&self) -> &'static str {
        STAGE
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let wanted = self.0.load(Ordering::Relaxed);
        for (score, &row) in out.iter_mut().zip(kept) {
            let rank = usize::from(candidates[row as usize].dp_rank);
            *score = if rank == wanted { 1.0 } else { 0.0 };
        }
        TieBand::NORMALIZED
    }
}

/// A decode engine of four ranks stands in the registry as four worker
/// objects, one a rank, behind one address. The engine picks a row, and the
/// rank the decode side is sent with is that row's own: each request names
/// the rank its scorer put first, in an order no turn over the ranks gives.
/// No rank choice runs behind the pick: beside an engine the gateway's rank
/// scheduler is refused at the start
/// (`the_rank_option_beside_a_supplied_document_is_refused_by_name`). The
/// prefill worker has no rank, and neither side is sent one for it.
#[tokio::test]
async fn the_decode_sides_rank_on_the_wire_is_the_picked_rows_own() {
    let (prefill_url, prefill_seen) = spawn_recording_stub("{}").await;
    let (decode_url, decode_seen) = spawn_recording_stub("{}").await;
    let wanted = Arc::new(AtomicUsize::new(0));
    let asked = Arc::clone(&wanted);
    let stages = table_with(move || Box::new(WantsRank(Arc::clone(&asked))));
    let (router, _monitor) = pair_router_under(&stages, DecodePick::WithPrefill);
    router
        .worker_registry
        .register_or_replace(Arc::from(create_test_worker(
            prefill_url,
            WorkerType::Prefill,
            true,
        )));
    for rank in 0..RANKS {
        let row = BasicWorkerBuilder::new(decode_url.clone())
            .worker_type(WorkerType::Decode)
            .dp_config(rank, RANKS)
            .build();
        row.set_status(openai_protocol::worker::WorkerStatus::Ready);
        router.worker_registry.register_or_replace(Arc::new(row));
    }
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));

    for rank in [2, 2, 0, 3, 3, 1] {
        wanted.store(rank, Ordering::Relaxed);
        let request: GenerateRequest =
            serde_json::from_value(json!({"text": "hi"})).expect("valid generate request");
        let response = router
            .route_generate(None, &tenant, request, UNKNOWN_MODEL_ID)
            .await;
        assert_eq!(response.status(), StatusCode::OK);

        let (to_prefill, to_decode) = (taken(&prefill_seen), taken(&decode_seen));
        assert_eq!((to_prefill.len(), to_decode.len()), (1, 1));
        let (prefill_body, decode_body) = (&to_prefill[0].1, &to_decode[0].1);
        assert_eq!(decode_body["routed_dp_rank"], json!(rank));
        assert_eq!(prefill_body.get("routed_dp_rank"), None);
        assert_eq!(decode_body.get("disagg_prefill_dp_rank"), None);
    }
}

/// The rank header of each request a server took, in the order it took
/// them.
type Ranks = Arc<Mutex<Vec<Option<String>>>>;

/// A server that writes down the rank header of each request and answers a
/// chat completion. It runs `then` as it answers.
async fn spawn_noting_rank(then: impl Fn() + Clone + Send + Sync + 'static) -> (String, Ranks) {
    let ranks = Ranks::default();
    let log = Arc::clone(&ranks);
    let app = axum::Router::new().fallback(axum::routing::any(move |request: Request| {
        let (log, then) = (Arc::clone(&log), then.clone());
        async move {
            let rank = rank_header(&request);
            log.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(rank);
            then();
            (
                [(CONTENT_TYPE, "application/json")],
                r#"{"object":"chat.completion"}"#,
            )
        }
    }));
    (spawn_stub(app).await, ranks)
}

/// The rank headers `ranks` holds, taken out.
fn noted(ranks: &Ranks) -> Vec<Option<String>> {
    std::mem::take(&mut *ranks.lock().unwrap_or_else(PoisonError::into_inner))
}

/// A worker object of the runtime whose pair goes out prefill first, one
/// rank of `ranks` where a rank is given, ready.
fn sequential_row(
    url: &str,
    worker_type: WorkerType,
    rank: Option<usize>,
    ranks: usize,
) -> Arc<dyn Worker> {
    let row = BasicWorkerBuilder::new(url)
        .worker_type(worker_type)
        .runtime_type(RuntimeType::Vllm);
    let row = match rank {
        Some(rank) => row.dp_config(rank, ranks),
        None => row,
    };
    let row: Arc<dyn Worker> = Arc::new(row.build());
    row.set_status(openai_protocol::worker::WorkerStatus::Ready);
    row
}

/// One chat request through the pair route, answered whole.
async fn one_chat(router: &PDRouter) {
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let chat: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .expect("valid chat request");
    let response = router
        .route_chat(None, &tenant, Arc::new(chat), UNKNOWN_MODEL_ID)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await;
    drop(body.expect("the decode's answer"));
}

/// A pair that goes out prefill first names each side's rank on the wire
/// too. Two decode engines of two ranks stand in the registry as four worker
/// objects, and the prefill engine's two ranks as two. The engine picks a
/// row, and the rank header of the decode's request is that row's own: each
/// request names the rank its scorer put first, whichever of the two
/// engines took it. The prefill's request names a rank of its own engine.
#[tokio::test]
async fn a_pair_sent_prefill_first_names_the_picked_rows_rank() {
    let (prefill_url, to_prefill) = spawn_noting_rank(|| ()).await;
    let (one_url, to_one) = spawn_noting_rank(|| ()).await;
    let (other_url, to_other) = spawn_noting_rank(|| ()).await;
    let wanted = Arc::new(AtomicUsize::new(0));
    let asked = Arc::clone(&wanted);
    let stages = table_with(move || Box::new(WantsRank(Arc::clone(&asked))));
    let (router, _monitor) = pair_router_under(&stages, DecodePick::WithPrefill);
    for rank in 0..2 {
        for (url, worker_type) in [
            (&prefill_url, WorkerType::Prefill),
            (&one_url, WorkerType::Decode),
            (&other_url, WorkerType::Decode),
        ] {
            let row = sequential_row(url, worker_type, Some(rank), 2);
            router.worker_registry.register_or_replace(row);
        }
    }

    for rank in [1, 1, 0, 1, 0, 0] {
        wanted.store(rank, Ordering::Relaxed);
        one_chat(&router).await;

        let mut to_decodes = noted(&to_one);
        to_decodes.append(&mut noted(&to_other));
        assert_eq!(to_decodes, [Some(rank.to_string())]);
        let prefill_ranks = noted(&to_prefill);
        assert!(
            matches!(prefill_ranks.as_slice(), [Some(sent)] if sent == "0" || sent == "1"),
            "{prefill_ranks:?}"
        );
    }
}

/// A decode that moves behind its prefill's answer is sent the rank of the
/// row it moved to. One decode engine of two ranks stands in the registry
/// as two worker objects. The scorer puts rank 0 in front until the prefill
/// has answered and rank 1 from then on, under a document that picks the
/// decode behind its prefill: the request goes out with rank 1, the row the
/// later decision named, and not with the rank of the row named first. A
/// prefill registered once is sent no rank.
#[tokio::test]
async fn a_decode_that_moved_is_sent_the_rank_of_the_row_it_moved_to() {
    let wanted = Arc::new(AtomicUsize::new(0));
    let answered = Arc::clone(&wanted);
    let (prefill_url, to_prefill) =
        spawn_noting_rank(move || answered.store(1, Ordering::Relaxed)).await;
    let (decode_url, to_decode) = spawn_noting_rank(|| ()).await;
    let asked = Arc::clone(&wanted);
    let stages = table_with(move || Box::new(WantsRank(Arc::clone(&asked))));
    let (router, _monitor) = pair_router_under(&stages, DecodePick::AfterPrefill);
    let registry = &router.worker_registry;
    registry.register_or_replace(sequential_row(&prefill_url, WorkerType::Prefill, None, 1));
    for rank in 0..2 {
        registry.register_or_replace(sequential_row(
            &decode_url,
            WorkerType::Decode,
            Some(rank),
            2,
        ));
    }

    one_chat(&router).await;

    assert_eq!(noted(&to_decode), [Some("1".to_string())]);
    assert_eq!(noted(&to_prefill), [None]);
}

// ---- a decode decided behind its prefill ---------------------------------------------------

/// What the scorer and the engines' servers said, in the order they said
/// it.
type Said = Arc<Mutex<Vec<String>>>;

fn say(said: &Said, word: impl Into<String>) {
    let mut said = said.lock().unwrap_or_else(PoisonError::into_inner);
    said.push(word.into());
}

/// A scorer that takes every word of a lease, so the engine leases each
/// pick under it, and writes down each time it is asked and each word it is
/// told. Until the prefill has answered it puts `first` in front, and from
/// then on `later`: the rows of a later instant rank another way.
#[derive(Debug)]
struct Hears {
    said: Said,
    answered: Arc<AtomicBool>,
    first: WorkerId,
    later: WorkerId,
}

impl Lifecycle for Hears {
    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }

    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        say(&self.said, format!("issued {:?}", lease.side));
    }

    fn on_dispatched(&self, _: &Lease, _: &Target, side: Side, _: u32) {
        say(&self.said, format!("dispatched {side:?}"));
    }

    fn on_completed(&self, _: &Lease, outcome: &Outcome) {
        say(&self.said, format!("ended {:?}", outcome.side));
    }

    fn on_released(&self, lease: &Lease, _: ReleaseReason) {
        say(&self.said, format!("released {:?}", lease.side));
    }
}

impl Scorer for Hears {
    fn name(&self) -> &'static str {
        STAGE
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        // How often one decision asks its scorer is the engine's own: asked
        // twice in a row is written once. A word of a lease is always written.
        let asked_last = {
            let said = self.said.lock().unwrap_or_else(PoisonError::into_inner);
            said.last().is_some_and(|word| word == "scored")
        };
        if !asked_last {
            say(&self.said, "scored");
        }
        let front = if self.answered.load(Ordering::Relaxed) {
            self.later
        } else {
            self.first
        };
        for (score, &row) in out.iter_mut().zip(kept) {
            *score = if candidates[row as usize].id == front {
                1.0
            } else {
                0.0
            };
        }
        TieBand::NORMALIZED
    }
}

/// A server that writes `word` down for each request and answers `reply`.
/// With `answered` it is the prefill's: it raises the mark as it answers.
async fn spawn_saying(
    said: &Said,
    word: &'static str,
    reply: &'static str,
    answered: Option<&Arc<AtomicBool>>,
) -> String {
    let (said, answered) = (Arc::clone(said), answered.cloned());
    spawn_stub(axum::Router::new().fallback(axum::routing::any(move || {
        let (said, answered) = (Arc::clone(&said), answered.clone());
        async move {
            say(&said, word);
            if let Some(answered) = answered {
                answered.store(true, Ordering::Relaxed);
            }
            ([(CONTENT_TYPE, "application/json")], reply)
        }
    })))
    .await
}

/// A pair goes out prefill first under a document that picks its decode
/// behind the prefill, and the decode side's scorer hears the request's
/// words in one order with the two engines. The pair's first decision
/// books the prefill alone. When the prefill has answered the route asks
/// the decode again: the scorer is asked once more, the decode it now puts
/// first is issued, and the prefill side's end is told then, where the
/// route learned the prefill answered and not at the response's end. The
/// decode's dispatch follows, the request goes to the decode decided
/// behind the prefill and not to the one named first, and the decode's end
/// is the response's. No side is told twice, and none is released.
#[tokio::test]
async fn a_decode_decided_behind_its_prefill_is_heard_behind_the_prefills_answer() {
    let said = Said::default();
    let answered = Arc::new(AtomicBool::new(false));
    let prefill_url = spawn_saying(
        &said,
        "the prefill answers",
        r#"{"kv_transfer_params":{"remote_engine_id":"eng0","remote_block_ids":[1,2]}}"#,
        Some(&answered),
    )
    .await;
    let chat_completion = r#"{"object":"chat.completion"}"#;
    let first_url = spawn_saying(
        &said,
        "the decode named first is asked",
        chat_completion,
        None,
    )
    .await;
    let later_url = spawn_saying(
        &said,
        "the decode decided later is asked",
        chat_completion,
        None,
    )
    .await;

    let (first, later) = (
        WorkerId::from_address(&first_url),
        WorkerId::from_address(&later_url),
    );
    let (heard, mark) = (Arc::clone(&said), Arc::clone(&answered));
    let stages = table_with(move || {
        Box::new(Hears {
            said: Arc::clone(&heard),
            answered: Arc::clone(&mark),
            first,
            later,
        })
    });
    let (router, _monitor) = pair_router_under(&stages, DecodePick::AfterPrefill);
    let [prefill, named_first] = sequential_pair(prefill_url, first_url);
    let decided_later: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(later_url)
            .worker_type(WorkerType::Decode)
            .runtime_type(RuntimeType::Vllm)
            .build(),
    );
    decided_later.set_status(openai_protocol::worker::WorkerStatus::Ready);
    for worker in [prefill, named_first, decided_later] {
        router.worker_registry.register_or_replace(worker);
    }
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));

    let chat: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .expect("valid chat request");
    let response = router
        .route_chat(None, &tenant, Arc::new(chat), UNKNOWN_MODEL_ID)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    // The decode side has ended once its answer is read whole and let go.
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await;
    drop(body.expect("the decode's answer"));

    let order = said.lock().unwrap_or_else(PoisonError::into_inner).clone();
    assert_eq!(
        order,
        [
            "scored",
            "issued Prefill",
            "dispatched Prefill",
            "the prefill answers",
            "scored",
            "issued Decode",
            "ended Prefill",
            "dispatched Decode",
            "the decode decided later is asked",
            "ended Decode",
        ]
    );
}

// ---- the ask keeps the shape the two sides were built for ----------------------------------

/// A pair goes out prefill first under a document that picks its decode
/// behind the prefill, and the decode its scorer puts in front there is a
/// MoRI-IO engine. The two sides were built for the decode named first,
/// which is no such engine: the prefill was told of no transfer, and a
/// MoRI-IO decode would wait for one that never comes. So the route's rule
/// at the ask refuses that engine, and the request goes to the decode named
/// first. The prefill reports no transport, so the pair index pairs it with
/// either decode. The scorer is asked behind the prefill's answer: the
/// decode is asked again, and it is the rule that keeps it where it was.
#[tokio::test]
async fn a_decode_asked_again_does_not_move_to_a_moriio_engine() {
    let said = Said::default();
    let answered = Arc::new(AtomicBool::new(false));
    let prefill_url = spawn_saying(&said, "the prefill answers", "{}", Some(&answered)).await;
    let chat_completion = r#"{"object":"chat.completion"}"#;
    let first_url = spawn_saying(
        &said,
        "the decode named first is asked",
        chat_completion,
        None,
    )
    .await;
    let moriio_url =
        spawn_saying(&said, "the MoRI-IO decode is asked", chat_completion, None).await;

    let (first, later) = (
        WorkerId::from_address(&first_url),
        WorkerId::from_address(&moriio_url),
    );
    let (heard, mark) = (Arc::clone(&said), Arc::clone(&answered));
    let stages = table_with(move || {
        Box::new(Hears {
            said: Arc::clone(&heard),
            answered: Arc::clone(&mark),
            first,
            later,
        })
    });
    let (router, _monitor) = pair_router_under(&stages, DecodePick::AfterPrefill);
    let worker_at = |url: String, worker_type: WorkerType| {
        BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .runtime_type(RuntimeType::Vllm)
    };
    let workers: [Arc<dyn Worker>; 3] = [
        Arc::new(worker_at(prefill_url, WorkerType::Prefill).build()),
        Arc::new(worker_at(first_url, WorkerType::Decode).build()),
        Arc::new(
            worker_at(moriio_url, WorkerType::Decode)
                .kv_connector("MoRIIOConnector")
                .label("moriio_mode", "read")
                .build(),
        ),
    ];
    for worker in workers {
        worker.set_status(openai_protocol::worker::WorkerStatus::Ready);
        router.worker_registry.register_or_replace(worker);
    }
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));

    let chat: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .expect("valid chat request");
    let response = router
        .route_chat(None, &tenant, Arc::new(chat), UNKNOWN_MODEL_ID)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await;
    drop(body.expect("the decode's answer"));

    let order = said.lock().unwrap_or_else(PoisonError::into_inner).clone();
    let at = |word: &str| order.iter().position(|line| line == word);
    let answered_at = at("the prefill answers").expect("the prefill went first");
    assert!(
        order[answered_at..].iter().any(|word| word == "scored"),
        "the decode is asked again behind the prefill's answer: {order:?}"
    );
    assert_eq!(at("the MoRI-IO decode is asked"), None, "{order:?}");
    assert!(
        at("the decode named first is asked").is_some_and(|sent| sent > answered_at),
        "{order:?}"
    );
}

// ---- the request's output limit ------------------------------------------------------------

/// A scorer that writes down the output limit of each request it is shown.
#[derive(Debug)]
struct NotesLimit(Said);

impl Lifecycle for NotesLimit {}

impl Scorer for NotesLimit {
    fn name(&self) -> &'static str {
        STAGE
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        say(&self.0, format!("limit {}", facts.max_output_tokens));
        out.fill(1.0);
        TieBand::NORMALIZED
    }
}

/// The pair route states the body's output limit to the engine. A chat
/// that asks for at most 50 tokens goes out prefill first under a document
/// that picks its decode behind the prefill. The decode side's scorer is
/// asked at the pair's first decision and again behind the prefill's
/// answer, and the request it is shown states 50 both times: the attempt
/// reads the leased body for the first, and the decode's answer keeps the
/// limit for the second. How often one decision asks its scorer is the
/// engine's own, so each side of the prefill's answer holds one ask or
/// more.
///
/// The same holds under prefill admission, where the selection runs inside
/// the admission's own lock, once a pass: the reading is made there.
#[tokio::test]
async fn the_engine_reads_the_output_limit_a_pair_request_states() {
    for admission in [false, true] {
        let said = Said::default();
        let prefill_url = spawn_saying(
            &said,
            "the prefill answers",
            r#"{"kv_transfer_params":{"remote_engine_id":"eng0","remote_block_ids":[1,2]}}"#,
            None,
        )
        .await;
        let chat_completion = r#"{"object":"chat.completion"}"#;
        let one_url = spawn_saying(&said, "a decode is asked", chat_completion, None).await;
        let other_url = spawn_saying(&said, "a decode is asked", chat_completion, None).await;
        let heard = Arc::clone(&said);
        let stages = table_with(move || Box::new(NotesLimit(Arc::clone(&heard))));
        let (mut router, _monitor) = pair_router_under(&stages, DecodePick::AfterPrefill);
        if admission {
            router.prefill_admission = Some(Arc::new(PrefillAdmission::new(1, 1, PENDING_ANSWER)));
        }
        let [prefill, one] = sequential_pair(prefill_url, one_url);
        let other = sequential_row(&other_url, WorkerType::Decode, None, 1);
        for worker in [prefill, one, other] {
            router.worker_registry.register_or_replace(worker);
        }

        one_chat(&router).await;

        let order = said.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let answered_at = order.iter().position(|word| word == "the prefill answers");
        let (before, behind) = order.split_at(answered_at.expect("the prefill went first"));
        for side in [before, behind] {
            let limits: Vec<&String> = side
                .iter()
                .filter(|word| word.starts_with("limit"))
                .collect();
            assert!(!limits.is_empty(), "admission {admission}: {order:?}");
            assert!(
                limits.iter().all(|word| *word == "limit 50"),
                "admission {admission}: {order:?}"
            );
        }
    }
}
