//! A pair's decode side over an engine of several ranks, each rank a worker
//! object of its own, where the policy engine places the pair.

use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec},
    engine::{Candidate, RequestFacts, Scorer, TieBand},
    stages::{Lifecycle, StageRegistry},
};

use super::*;
use crate::routers::common::placement::tests::engine_on;

const MODEL: &str = "test-model-ranks";

/// The ranks of the decode engine below.
const RANKS: usize = 4;

/// The case's stage, as its table and its document name it.
const STAGE: &str = "wants-a-rank";

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

/// A gRPC worker of `MODEL` in `role`, at `url`.
fn grpc_worker(url: &str, role: WorkerType) -> BasicWorkerBuilder {
    BasicWorkerBuilder::new(url)
        .model(ModelCard::new(MODEL))
        .worker_type(role)
        .connection_mode(ConnectionMode::Grpc)
        .health_config(no_health_check())
}

/// A decode engine of four ranks stands in the registry as four worker
/// objects, one a rank, behind one address. The engine picks a row, and the
/// decode worker the pair stage hands on is that row: each selection names
/// the rank its scorer put first, in an order no turn over the ranks gives.
/// The request execution sends each side with its worker's own rank
/// (`tokenspeed_parallel_pd_forwards_selected_ranks_without_changing_the_room`),
/// so that rank is the one on the wire.
#[test]
fn the_pair_stage_hands_on_the_rank_row_the_engine_picked() {
    let registry = Arc::new(WorkerRegistry::new());
    let prefill = grpc_worker("grpc://127.0.0.1:8000", WorkerType::Prefill).build();
    registry.register(Arc::new(prefill)).unwrap();
    for rank in 0..RANKS {
        let row = grpc_worker("grpc://127.0.0.1:8100", WorkerType::Decode)
            .dp_config(rank, RANKS)
            .build();
        registry.register(Arc::new(row)).unwrap();
    }

    let wanted = Arc::new(AtomicUsize::new(0));
    let asked = Arc::clone(&wanted);
    let mut stages = StageRegistry::builtin();
    stages
        .scorer(STAGE, move || Box::new(WantsRank(Arc::clone(&asked))))
        .expect("a free name");
    let document = PolicySpec::pair_presets("round_robin", "round_robin");
    let mut document = document.expect("the pair presets");
    document.generation = 1;
    let side = document.profiles.get_mut("decode");
    let side = side.expect("the decode side's profile");
    side.score = vec![STAGE.into()];
    side.pick = "best".into();
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    engine_on(&policies, spec.expect("the table holds the stage"));
    let stage =
        WorkerSelectionStage::new(registry, policies, WorkerSelectionMode::PrefillDecode, None);

    for rank in [2, 2, 0, 3, 3, 1] {
        wanted.store(rank, Ordering::Relaxed);
        let (prefill, decode, _) = stage
            .select_pd_pair(
                MODEL,
                PlacementInputs::default(),
                None,
                None,
                PairDemands::default(),
            )
            .expect("the engine places the pair");
        assert_eq!(decode.dp_rank(), Some(rank));
        assert_eq!(prefill.dp_rank(), None);
    }
}
