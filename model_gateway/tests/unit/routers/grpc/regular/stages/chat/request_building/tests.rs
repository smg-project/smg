use openai_protocol::chat::ChatCompletionRequest;
use serde_json::json;

use super::*;
use crate::routers::grpc::{
    context::{ClientSelection, ExecutionPlanKind},
    multimodal::{single_image_intermediate, MultimodalIntermediate},
    proto_wrapper::ProtoRequest,
    regular::stages::testing::{
        assert_fixed_request_plan, components, mock_backend_client, FORCED_CALL_TAG,
    },
    ProcessedMessages,
};

const MODEL: &str = "test-model";

/// The builder's sequence on the fixed request of
/// [`assert_fixed_request_plan`], asked as a chat request with no worker
/// selected: the plan and the stamp read as pinned there, with the id
/// under the chat prefix, and the prompt leaves no thought open.
#[tokio::test]
async fn a_fixed_chat_request_builds_the_pinned_plan_and_stamp() {
    let (client, engine) = mock_backend_client(MODEL).await;
    let chat: Arc<ChatCompletionRequest> = Arc::new(
        serde_json::from_value(json!({
            "model": MODEL, "temperature": 0.5, "max_tokens": 7,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .expect("a chat request"),
    );
    let mut ctx =
        RequestContext::for_chat(Arc::clone(&chat), None, MODEL.to_string(), components());
    ctx.state.clients = Some(ClientSelection::Single { client });
    let (plan, stamp, reasoning_in_prefill) = build_chat_backed_plan(
        &mut ctx,
        &chat,
        ProviderProfile::for_model(MODEL),
        "<prompt>".to_string(),
        true,
        false,
        vec![1, 2, 3],
        Some(("structural_tag".to_string(), FORCED_CALL_TAG.to_string())),
        FormatSetAside::default(),
        None,
        false,
        "chatcmpl-",
        PlanSettings {
            inject_pd_metadata: false,
            plan_kind: ExecutionPlanKind::Single,
        },
    )
    .await
    .map_err(|refusal| refusal.status())
    .expect("built");
    engine.abort();
    assert!(!reasoning_in_prefill);
    assert_fixed_request_plan(&plan, &stamp, "chatcmpl-");
}

/// The handoff the usage depends on, through the stage's own entry: the
/// image tokens the expansion wrote sit on the state's intermediate,
/// request building reads them before the plan's assembly takes the
/// intermediate, and the chat spec carries them; with no intermediate
/// the spec carries none.
#[tokio::test]
async fn the_spec_carries_the_image_tokens_the_intermediate_holds() {
    for image_tokens in [Some(326_u32), None] {
        let (client, engine) = mock_backend_client(MODEL).await;
        let chat: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(json!({
                "model": MODEL, "max_tokens": 7,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .expect("a chat request"),
        );
        let mut ctx =
            RequestContext::for_chat(Arc::clone(&chat), None, MODEL.to_string(), components());
        ctx.state.clients = Some(ClientSelection::Single { client });
        ctx.state.preparation = Some(PreparationOutput::Chat {
            token_ids: vec![1, 2, 3],
            processed_messages: ProcessedMessages {
                text: "<prompt>".to_string(),
                unbilled_prompt_tokens: 0,
                continued_final_message: false,
                generation_prompt: true,
            },
            tool_constraints: Ok(symphony::Prepared::None),
        });
        ctx.state.multimodal_intermediate = image_tokens.map(|count| {
            MultimodalIntermediate::try_new(vec![single_image_intermediate()])
                .expect("one image batch")
                .with_image_prompt_tokens(Some(count))
        });
        let stage = ChatRequestBuildingStage::new(PlanSettings {
            inject_pd_metadata: false,
            plan_kind: ExecutionPlanKind::Single,
        });
        let built = stage
            .build_plan(&mut ctx)
            .await
            .map_err(|refusal| refusal.status())
            .expect("built");
        engine.abort();
        let ResponseSpec::Chat(spec) = built.spec else {
            panic!("a chat spec");
        };
        assert_eq!(spec.image_tokens, image_tokens);
        assert!(
            ctx.state.multimodal_intermediate.is_none(),
            "the plan's assembly took the intermediate after the read"
        );
    }
}

/// The set-aside reaches the builder through the shared sequence: the
/// mock engine's wire reads a JSON `response_format` as the constraint
/// when no tag is handed, and the same request builds a plan with no
/// constraint at all once the stage sets the field aside.
#[tokio::test]
async fn a_response_format_set_aside_is_not_read_on_the_way_to_the_engine() {
    let chat: Arc<ChatCompletionRequest> = Arc::new(
        serde_json::from_value(json!({
            "model": MODEL, "temperature": 0.5, "max_tokens": 7,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}}
            }
        }))
        .expect("a chat request"),
    );
    let (client, engine) = mock_backend_client(MODEL).await;
    let mut read = Vec::new();
    for set_aside in [
        FormatSetAside::default(),
        FormatSetAside {
            response_format: true,
            grammar: false,
        },
    ] {
        let mut ctx =
            RequestContext::for_chat(Arc::clone(&chat), None, MODEL.to_string(), components());
        ctx.state.clients = Some(ClientSelection::Single {
            client: client.clone(),
        });
        let (plan, _, _) = build_chat_backed_plan(
            &mut ctx,
            &chat,
            ProviderProfile::for_model(MODEL),
            "<prompt>".to_string(),
            true,
            false,
            vec![1, 2, 3],
            None,
            set_aside,
            None,
            false,
            "chatcmpl-",
            PlanSettings {
                inject_pd_metadata: false,
                plan_kind: ExecutionPlanKind::Single,
            },
        )
        .await
        .map_err(|refusal| refusal.status())
        .expect("built");
        let ExecutionPlan::Single(ProtoRequest::Generate(request)) = plan else {
            panic!("a single generate plan")
        };
        read.push((
            request.json_schema().map(str::to_owned),
            request.structural_tag().map(str::to_owned),
        ));
    }
    engine.abort();
    assert_eq!(
        read[0],
        (Some(r#"{"type":"object"}"#.to_string()), None),
        "nothing set aside: the body's schema is the constraint"
    );
    assert_eq!(read[1], (None, None), "set aside: the field is not read");
}

/// What the stage sets aside follows what the constraint answers: the
/// `response_format` when Symphony's tag stands in for a JSON answer, both
/// kinds of field for the whole turn's grammar, and nothing when the
/// legacy parsers read the response, when the request names no format, or
/// when the constraint is not a tag.
#[test]
fn the_set_aside_follows_what_the_constraint_answers() {
    let tag = ("structural_tag".to_string(), FORCED_CALL_TAG.to_string());
    let schema = ("json_schema".to_string(), "{}".to_string());
    let aside = |response_format, grammar| FormatSetAside {
        response_format,
        grammar,
    };
    assert_eq!(
        set_aside_for(true, true, Some(&tag), false),
        aside(true, false)
    );
    assert_eq!(
        set_aside_for(true, true, Some(&tag), true),
        aside(true, true)
    );
    assert_eq!(set_aside_for(false, false, None, true), aside(true, true));
    assert_eq!(
        set_aside_for(false, true, Some(&tag), false),
        aside(false, false)
    );
    assert_eq!(
        set_aside_for(true, false, Some(&tag), false),
        aside(false, false)
    );
    assert_eq!(
        set_aside_for(true, true, Some(&schema), false),
        aside(false, false)
    );
    assert_eq!(set_aside_for(true, true, None, false), aside(false, false));
}
