use std::{env, fs, sync::Arc};

use openai_protocol::chat::ChatCompletionRequest;
use serde_json::{json, Map, Value};

use super::{HarmonyPreparationStage, HarmonyRequestBuildingStage};
use crate::routers::grpc::{
    common::stages::{BuildStage, PipelineStage},
    context::{ClientSelection, ExecutionPlan, ExecutionPlanKind, PlanSettings, RequestContext},
    proto_wrapper::{ProtoGenerateRequest, ProtoRequest},
    regular::stages::testing::{components, mock_backend_client},
    testing::worker,
};

/// Exercise preparation and worker selection together. The optional export lets
/// the CPU matcher test consume the exact grammar and suppression list on the wire.
#[tokio::test]
async fn vllm_harmony_wire_constraints() {
    let model = "gpt-oss";
    let (client, engine) = mock_backend_client(model).await;
    let stage = HarmonyRequestBuildingStage::new(PlanSettings {
        inject_pd_metadata: false,
        plan_kind: ExecutionPlanKind::Single,
    });
    let schema = json!({
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": false
    });
    let tools =
        json!([{"type": "function", "function": {"name": "weather", "parameters": schema}}]);
    let answer =
        json!({"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}});
    let mut fixtures = Map::new();
    for vocabulary in ["older", "current"] {
        let mut cases = Map::new();
        for (name, options) in [
            ("auto", json!({"tools": tools, "response_format": answer})),
            (
                "required",
                json!({"tools": tools, "tool_choice": "required"}),
            ),
            ("json", json!({"response_format": answer})),
        ] {
            let mut body = options;
            body["model"] = json!(model);
            body["messages"] = json!([{"role": "user", "content": "Check the weather."}]);
            let request: ChatCompletionRequest =
                serde_json::from_value(body).expect("valid request");
            let mut ctx =
                RequestContext::for_chat(Arc::new(request), None, model.into(), components());
            ctx.state.clients = Some(ClientSelection::Single {
                client: client.clone(),
            });
            ctx.state.workers = Some(worker(&[("structural_tag_vocabulary", vocabulary)]));
            HarmonyPreparationStage::new()
                .execute(&mut ctx)
                .await
                .expect("prepare");
            let built = stage.build(&mut ctx).await.expect("build");
            let ExecutionPlan::Single(ProtoRequest::Generate(request)) = built.plan else {
                panic!("expected a single generation request");
            };
            let grammar: Value = serde_json::from_str(request.structural_tag().expect("grammar"))
                .expect("valid structural tag");
            let ProtoGenerateRequest::Vllm(request) = request else {
                panic!("expected vLLM")
            };
            let params = request.sampling_params.expect("sampling parameters");
            assert!(params.grammar_from_first_token);
            assert!(
                params.stop_token_ids.contains(&200002),
                "RETURN ends the turn"
            );
            assert!(
                params.suppressed_token_ids.contains(&200012),
                "CALL is an alternate stop"
            );
            assert!(
                params.suppressed_token_ids.contains(&199999),
                "end-of-text is an alternate stop"
            );
            assert!(!params.suppressed_token_ids.contains(&200002));
            cases.insert(
                name.to_owned(),
                json!({"grammar": grammar, "suppressed": params.suppressed_token_ids}),
            );
        }
        fixtures.insert(vocabulary.to_owned(), Value::Object(cases));
    }
    engine.abort();
    assert_ne!(fixtures["older"], fixtures["current"]);
    if let Some(path) = env::var_os("SMG_HARMONY_TEST_FIXTURE") {
        fs::write(
            path,
            serde_json::to_vec(&fixtures).expect("serialize fixtures"),
        )
        .expect("write fixtures");
    }
}
