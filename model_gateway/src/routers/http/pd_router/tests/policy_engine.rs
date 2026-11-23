//! The pair route where the policy engine places the pair. Each case's
//! gateway starts the engine by its own start (`policy_controller: on`),
//! over a table that holds a scorer of the case's own and under a supplied
//! document that names that scorer on the decode side.

use smg_policy_controller::{
    document::{DecodePick, PolicySpec},
    engine::{Candidate, RequestFacts, Scorer, TieBand},
    stages::{Lifecycle, StageRegistry},
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
