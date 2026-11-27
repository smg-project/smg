use openai_protocol::messages::CreateMessageRequest;
use serde_json::json;

use super::*;
use crate::routers::grpc::{
    context::{ClientSelection, ExecutionPlanKind},
    regular::stages::testing::{
        assert_fixed_request_plan, components, mock_backend_client, FORCED_CALL_TAG,
    },
};

const MODEL: &str = "test-model";

/// The builder's sequence on the fixed request of
/// [`assert_fixed_request_plan`], asked as a Messages request with no
/// worker selected and the prompt's thought decided closed: the plan and
/// the stamp read as pinned there, with the id under the Messages prefix.
#[tokio::test]
async fn a_fixed_messages_request_builds_the_pinned_plan_and_stamp() {
    let (client, engine) = mock_backend_client(MODEL).await;
    let messages: Arc<CreateMessageRequest> = Arc::new(
        serde_json::from_value(json!({
            "model": MODEL, "temperature": 0.5, "max_tokens": 7,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .expect("a Messages request"),
    );
    let mut ctx = RequestContext::for_messages_carrying_its_tools(
        Arc::clone(&messages),
        None,
        MODEL.to_string(),
        components(),
    );
    ctx.state.clients = Some(ClientSelection::Single { client });
    let (plan, stamp) = build_messages_backed_plan(
        &mut ctx,
        &messages,
        "<prompt>".to_string(),
        vec![1, 2, 3],
        Some(("structural_tag".to_string(), FORCED_CALL_TAG.to_string())),
        None,
        false,
        false,
        None,
        PlanSettings {
            inject_pd_metadata: false,
            plan_kind: ExecutionPlanKind::Single,
        },
    )
    .map_err(|refusal| refusal.status())
    .expect("built");
    engine.abort();
    assert_fixed_request_plan(&plan, &stamp, "msg_");
}
