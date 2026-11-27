use super::*;
use crate::routers::grpc::common::responses::utils::namespace_test_request;
#[tokio::test]
async fn namespace_stream_events_and_completed_output_agree() {
    let request: ResponsesRequest = namespace_test_request();
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    emitter.set_original_request(request);
    let (index, id) = emitter.allocate_output_index(OutputItemKind::FunctionCall);
    let item = json!({"id":id,"type":"function_call","call_id":"call_test","name":"weather.lookup","arguments":"{}","status":"completed"});
    let (tx, mut rx) = sse_channel();
    emitter
        .send_output_item_added(index, &item, &tx)
        .await
        .unwrap();
    emitter
        .send_output_item_done(index, item, RETAIN_OUTPUT_ITEM, &tx)
        .await
        .unwrap();
    for _ in 0..2 {
        let frame = rx
            .recv()
            .await
            .expect("a frame")
            .expect("a frame, not an error");
        let text = String::from_utf8(frame.to_vec()).unwrap();
        let data = text.lines().find_map(|l| l.strip_prefix("data: ")).unwrap();
        let event: serde_json::Value = serde_json::from_str(data).unwrap();
        assert_eq!(event["item"]["name"], "lookup", "{text}");
        assert_eq!(event["item"]["namespace"], "weather", "{text}");
    }
    emitter.complete_output_item(index);
    let event = emitter.emit_completed(None);
    assert_eq!(event["response"]["output"][0]["name"], "lookup");
    assert_eq!(event["response"]["output"][0]["namespace"], "weather");
}
