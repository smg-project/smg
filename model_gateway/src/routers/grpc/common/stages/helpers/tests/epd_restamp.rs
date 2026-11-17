//! An encode-prefill-decode restamp re-mints the prefill-decode room and keeps the encode rooms.

use std::sync::Arc;

use smg_grpc_client::tokenspeed_proto;

use super::*;
use crate::{
    routers::grpc::{context::ExecutionPlanKind, proto_wrapper::EncodeItemBootstrapInfo},
    worker::{BasicWorkerBuilder, RuntimeType, Worker, WorkerType},
};

fn tokenspeed_pd_pair(prefill_url: &str, decode_url: &str) -> WorkerSelection {
    let prefill: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(prefill_url)
            .worker_type(WorkerType::Prefill)
            .build(),
    );
    let decode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(decode_url)
            .worker_type(WorkerType::Decode)
            .build(),
    );
    WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill,
        decode,
        runtime_type: RuntimeType::TokenSpeed,
    }
}

fn tokenspeed_rendezvous(
    plan: &ExecutionPlan,
) -> (
    &tokenspeed_proto::EncodeBootstrapInfo,
    &tokenspeed_proto::KvBootstrapInfo,
) {
    match plan {
        ExecutionPlan::EncodePrefillDecode {
            request: ProtoGenerateRequest::TokenSpeed(req),
        } => (
            req.encode_bootstrap_info.as_ref().unwrap(),
            req.kv_bootstrap_info.as_ref().unwrap(),
        ),
        _ => panic!("expected a TokenSpeed EPD plan"),
    }
}

/// An EPD retry re-selects only the prefill/decode pair. The PD KV room
/// must be re-minted for the new prefill, while the encode rooms must
/// survive untouched — the first attempt's encode jobs are already
/// parked at those rendezvous points, so re-minting them would strand
/// the embeddings.
#[test]
fn epd_restamp_remints_pd_room_and_keeps_encode_rooms() {
    let mut request = ProtoGenerateRequest::TokenSpeed(Box::default());
    request.set_encode_bootstrap_info(vec![EncodeItemBootstrapInfo {
        item_index: 0,
        bootstrap_host: "encode-worker".to_string(),
        bootstrap_port: 9000,
        bootstrap_room: 41,
    }]);
    request.set_kv_bootstrap_info("attempt1-prefill".to_string(), 8998, 42);
    let mut plan = ExecutionPlan::generate(ExecutionPlanKind::EncodePrefillDecode, request);

    let stamp = AttemptStamp {
        id: IdStamp::Exact,
        sampling_mask: None,
        sampling_baseline: None,
        inject_pd_metadata: false,
        tool_constraint: None,
    };
    let retry_pair = tokenspeed_pd_pair("grpc://retry-prefill:30000", "grpc://decode:30000");
    restamp_plan_for_attempt(&mut plan, &stamp, &retry_pair, &ToolParserFactory::new()).unwrap();

    let ExecutionPlan::EncodePrefillDecode { request } = &plan else {
        panic!("expected an EPD plan")
    };
    assert!(
        request.awaits_encode(),
        "the retried prefill side still waits on the encoder's rooms, so its bound is the encode-aware one"
    );
    let (encode, kv) = tokenspeed_rendezvous(&plan);
    assert_eq!(
        (
            encode.items[0].bootstrap_host.as_str(),
            encode.items[0].bootstrap_room
        ),
        ("encode-worker", 41),
        "encode rendezvous must survive the retry"
    );
    assert_ne!(kv.bootstrap_room, 42, "PD KV room is re-minted per attempt");
    let new_prefill_host = match &retry_pair {
        WorkerSelection::Disaggregated { prefill, .. } => {
            prefill.metadata().bootstrap_host().to_string()
        }
        WorkerSelection::Single { .. } => {
            panic!("tokenspeed_pd_pair builds a disaggregated selection")
        }
    };
    assert_eq!(
        kv.bootstrap_host, new_prefill_host,
        "KV rendezvous names the newly selected prefill"
    );
}
