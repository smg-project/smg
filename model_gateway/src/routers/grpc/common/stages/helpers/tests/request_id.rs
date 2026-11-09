//! The request ids a plan stamps on its attempts: the caller's rid, the middleware's id or a minted
//! one, stable or suffixed per attempt as the mode asks.

use std::sync::Arc;

use openai_protocol::chat::ChatCompletionRequest;

use super::*;
use crate::tenant::TenantKey;

fn chat_request_type(rid: Option<&str>) -> RequestType {
    RequestType::Chat(Arc::new(ChatCompletionRequest {
        rid: rid.map(str::to_string),
        ..Default::default()
    }))
}

fn meta_with_request_id(id: &str) -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("test-tenant")).with_extension(RequestId(id.to_string()))
}

fn single_plan(request_id: &str) -> ExecutionPlan {
    use smg_grpc_client::sglang_proto;

    use crate::routers::grpc::proto_wrapper::ProtoRequest;
    let request = ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
        request_id: request_id.to_string(),
        ..Default::default()
    }));
    ExecutionPlan::Single(ProtoRequest::Generate(request))
}

fn batch_plan(shared: &str, sub_ids: &[String]) -> ExecutionPlan {
    use smg_grpc_client::sglang_proto;

    use crate::routers::grpc::context::ExecutionPlanKind;
    ExecutionPlan::Batch {
        kind: ExecutionPlanKind::Single,
        shared_request_id: shared.to_string(),
        requests: sub_ids
            .iter()
            .map(|id| {
                ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
                    request_id: id.clone(),
                    ..Default::default()
                }))
            })
            .collect(),
    }
}

fn plan_ids(plan: &ExecutionPlan) -> (String, Vec<String>) {
    match plan {
        ExecutionPlan::Batch {
            shared_request_id,
            requests,
            ..
        } => (
            shared_request_id.clone(),
            requests
                .iter()
                .map(|r| r.request_id().to_string())
                .collect(),
        ),
        other => (other.request_id().to_string(), Vec::new()),
    }
}

#[test]
fn rid_is_used_exactly_outside_pd_and_stays_stable_on_restamp() {
    let request_type = chat_request_type(Some("client-rid"));
    let meta = meta_with_request_id("chatcmpl-mw");

    let (id, stamp) = resolve_request_id_stamp(&request_type, Some(&meta), "chatcmpl-", false);
    assert_eq!(id, "client-rid");

    let mut plan = single_plan(&id);
    stamp.restamp(&mut plan).unwrap();
    assert_eq!(
        plan.request_id(),
        "client-rid",
        "rid is the caller's contract"
    );
}

#[test]
fn rid_gets_per_attempt_suffix_in_pd() {
    let request_type = chat_request_type(Some("client-rid"));

    let (id, stamp) = resolve_request_id_stamp(&request_type, None, "chatcmpl-", true);
    assert!(id.starts_with("client-rid-") && id != "client-rid");

    let mut plan = single_plan(&id);
    stamp.restamp(&mut plan).unwrap();
    let restamped = plan.request_id().to_string();
    assert!(restamped.starts_with("client-rid-"));
    assert_ne!(restamped, id, "each attempt gets a fresh engine id");
}

#[test]
fn middleware_id_is_base_with_per_execution_suffix() {
    let request_type = chat_request_type(None);
    let meta = meta_with_request_id("chatcmpl-mw");

    let (first, stamp) = resolve_request_id_stamp(&request_type, Some(&meta), "chatcmpl-", false);
    assert!(first.starts_with("chatcmpl-mw-"));

    let mut plan = single_plan(&first);
    stamp.restamp(&mut plan).unwrap();
    let second = plan.request_id().to_string();
    assert!(second.starts_with("chatcmpl-mw-"));
    assert_ne!(first, second);
}

/// The suffix a request's engine id gets is stamped at the request's own
/// time (its charge id's: the one wall-clock read the edge made for it), not
/// at a read of its own; a request without meta reads the clock.
#[test]
fn the_engine_id_suffix_is_stamped_at_the_requests_time() {
    let request_type = chat_request_type(None);
    let mut meta = meta_with_request_id("chatcmpl-mw");
    meta.request_charge_id = uuid::Uuid::new_v7(uuid::Timestamp::from_unix(
        uuid::NoContext,
        1_700_000_000,
        0,
    ));
    assert_eq!(request_wall_clock(Some(&meta)), (1_700_000_000, 0));

    let (id, _) = resolve_request_id_stamp(&request_type, Some(&meta), "chatcmpl-", false);
    let suffix = id
        .strip_prefix("chatcmpl-mw-")
        .expect("the middleware id and a suffix");
    let stamp = uuid::Uuid::parse_str(suffix)
        .expect("a uuid suffix")
        .get_timestamp()
        .expect("a version-7 uuid carries its time");
    assert_eq!(stamp.to_unix().0, 1_700_000_000);

    let before = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let (seconds, _) = request_wall_clock(None);
    assert!((before..=before + 1).contains(&seconds));
}

#[test]
fn falls_back_to_prefixed_mint_without_rid_or_middleware_id() {
    let request_type = chat_request_type(None);
    let meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

    let (id, stamp) = resolve_request_id_stamp(&request_type, Some(&meta), "chatcmpl-", false);
    assert!(id.starts_with("chatcmpl-"));

    let mut plan = single_plan(&id);
    stamp.restamp(&mut plan).unwrap();
    assert!(plan.request_id().starts_with("chatcmpl-"));
    assert_ne!(plan.request_id(), id, "minted ids are fresh per attempt");
}

#[test]
fn batch_rid_subs_stay_stable_outside_pd() {
    let request_type = chat_request_type(Some("client-rid"));
    let (shared, stamp) = resolve_batch_id_stamp(&request_type, None, "cmpl_", false);
    assert_eq!(shared, "client-rid");
    assert!(!stamp.unique_subs);

    let subs: Vec<String> = (0..2).map(|i| batch_sub_id(&shared, i, false)).collect();
    assert_eq!(subs, ["client-rid-p0", "client-rid-p1"]);

    let mut plan = batch_plan(&shared, &subs);
    IdStamp::Batch(stamp).restamp(&mut plan).unwrap();
    assert_eq!(plan_ids(&plan), (shared, subs), "rid batch ids are stable");
}

#[test]
fn batch_middleware_subs_are_unique_per_attempt_under_stable_shared() {
    let request_type = chat_request_type(None);
    let meta = meta_with_request_id("mw-id");
    let (shared, stamp) = resolve_batch_id_stamp(&request_type, Some(&meta), "cmpl_", false);
    assert_eq!(shared, "mw-id");
    assert!(stamp.unique_subs);

    let subs: Vec<String> = (0..2)
        .map(|i| batch_sub_id(&shared, i, stamp.unique_subs))
        .collect();
    let mut plan = batch_plan(&shared, &subs);
    IdStamp::Batch(stamp).restamp(&mut plan).unwrap();
    let (restamped_shared, restamped_subs) = plan_ids(&plan);
    assert_eq!(restamped_shared, "mw-id");
    for (i, (before, after)) in subs.iter().zip(&restamped_subs).enumerate() {
        assert!(after.starts_with(&format!("mw-id-p{i}-")));
        assert_ne!(before, after, "sub engine ids must be fresh per attempt");
    }
}

#[test]
fn batch_minted_shared_is_fresh_per_attempt() {
    let request_type = chat_request_type(None);
    let (shared, stamp) = resolve_batch_id_stamp(&request_type, None, "cmpl_", false);
    assert!(shared.starts_with("cmpl_"));
    assert!(stamp.stable_shared.is_none());

    let subs: Vec<String> = (0..2).map(|i| batch_sub_id(&shared, i, false)).collect();
    let mut plan = batch_plan(&shared, &subs);
    IdStamp::Batch(stamp).restamp(&mut plan).unwrap();
    let (restamped_shared, restamped_subs) = plan_ids(&plan);
    assert!(restamped_shared.starts_with("cmpl_"));
    assert_ne!(restamped_shared, shared);
    assert_eq!(
        restamped_subs,
        [
            format!("{restamped_shared}-p0"),
            format!("{restamped_shared}-p1")
        ]
    );
}
