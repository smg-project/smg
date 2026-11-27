use super::*;
use crate::routers::grpc::common::responses::utils::namespace_test_request;
#[test]
fn namespace_replay_preserves_structured_identity() {
    let request: ResponsesRequest = namespace_test_request();
    let mut state = ToolLoopState::default();
    state.record_call(
        request.tools.as_deref(),
        "call_test".into(),
        "weather.lookup".into(),
        "{}".into(),
        "sunny".into(),
        ResponseOutputItem::new_function_tool_call(
            "fc_test".into(),
            "call_test".into(),
            "weather.lookup".into(),
            "{}".into(),
            None,
            "completed".into(),
        ),
    );
    let wire = serde_json::to_value(&state.calls_to_carry[0]).unwrap();

    assert_eq!(wire["name"], "lookup");
    assert_eq!(wire["namespace"], "weather");
    assert_eq!(wire["call_id"], "call_test");
}
