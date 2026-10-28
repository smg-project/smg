//! Message request building stage: Build proto GenerateRequest for message requests

use std::{borrow::Cow, sync::Arc};

use async_trait::async_trait;
use axum::response::Response;

use crate::routers::{
    error,
    grpc::{
        common::stages::{
            helpers::{self, TagForStart},
            BuildStage,
        },
        context::{
            stage_output, AttemptStamp, BuildOutput, ExecutionPlan, ExecutionPlanKind,
            PreparationOutput, RequestContext,
        },
        proto_wrapper::MultimodalData,
        regular::{
            stages::{backed_plan, reading},
            symphony,
        },
        spec::ResponseSpec,
        utils,
    },
};

/// Message request building stage
///
/// Builds a backend-specific proto GenerateRequest from the PreparationOutput
/// and CreateMessageRequest sampling parameters.
pub(crate) struct MessageRequestBuildingStage {
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
}

impl MessageRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool, plan_kind: ExecutionPlanKind) -> Self {
        Self {
            inject_pd_metadata,
            plan_kind,
        }
    }
}

/// Build the backend `GenerateRequest` from the Messages request through the
/// sequence the chat and Messages stages share ([`backed_plan::build`]), with
/// the Messages request's client builder, its id prefix and the sampling mask
/// of its request type; returns the plan and the attempt stamp.
#[expect(clippy::too_many_arguments)]
fn build_messages_backed_plan(
    ctx: &mut RequestContext,
    messages_request: &openai_protocol::messages::CreateMessageRequest,
    processed_text: String,
    token_ids: Vec<u32>,
    tool_constraints: Option<(String, String)>,
    table_tag: Option<Arc<dyn TagForStart>>,
    first_token_tag: bool,
    require_reasoning: bool,
    multimodal_data: Option<MultimodalData>,
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
) -> Result<(ExecutionPlan, AttemptStamp), Response> {
    let sampling_mask = helpers::SamplingDefaultsMask::from_request_type(&ctx.input.request_type);
    backed_plan::build(
        ctx,
        backed_plan::Inputs {
            function: "MessageRequestBuildingStage::build",
            id_prefix: "msg_",
            model: &messages_request.model,
            processed_text,
            token_ids,
            tool_constraints,
            table_tag,
            first_token_tag,
            require_reasoning,
            multimodal_data,
            sampling_mask,
            inject_pd_metadata,
            plan_kind,
        },
        |client, request_id, processed_text, token_ids, options| {
            client.build_messages_request(
                request_id,
                messages_request,
                processed_text,
                token_ids,
                options,
            )
        },
    )
}

#[async_trait]
impl BuildStage for MessageRequestBuildingStage {
    async fn build(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        let multimodal_data =
            helpers::assemble_prompt_media(ctx, "MessageRequestBuildingStage::build").await?;

        // Take preparation state (last consumer — worker_selection already ran)
        let prep = stage_output(
            ctx.state.preparation.take(),
            "MessageRequestBuildingStage::build",
        )?;

        let messages_request = ctx.messages_request_arc();

        let PreparationOutput::Messages {
            token_ids,
            processed_messages,
            tool_constraints,
            mut spec,
        } = prep
        else {
            debug_assert!(false, "pipeline guarantees Messages variant");
            return Err(error::internal_error(
                "wrong_preparation_type",
                "Expected Messages preparation output",
            ));
        };

        let continued_final_message = processed_messages.continued_final_message;
        let (require_reasoning, thought_closed) = reading::reasoning_decision(
            ctx,
            &messages_request.model,
            &processed_messages.text,
            |tokenizer| {
                utils::messages_reasoning_starts_in_prefill(
                    &messages_request,
                    continued_final_message,
                    tokenizer,
                )
            },
        );

        // What constrains the output on the selected worker, and Symphony's
        // assignment, decided as for a chat request (`reading::decide`), in
        // the order the chat builder has: the assignment takes the constraint
        // as the request carries it, before the worker's shaping in the plan
        // builder below. The `auto` tag derives for the request over the
        // tools the template sees. The state records which parser reads the
        // message, for the response's header.
        let reading::Decided {
            table_tag,
            tool_constraints,
            first_token,
            assignment,
            ..
        } = reading::decide(
            ctx,
            symphony::Asks::Messages {
                request: &messages_request,
                continues: messages_request.continues_final_assistant(),
            },
            &spec.chat_tools,
            &processed_messages.text,
            tool_constraints,
            |_| {
                // The tag shares the spec's tools where `tool_choice` narrows
                // nothing (the common case): no copy of the schemas on the
                // first attempt; a narrowed list is the one the narrowing owns.
                let tools: Arc<[_]> = match utils::template_tools(
                    &spec.chat_tools,
                    messages_request.tool_choice.as_ref(),
                ) {
                    Cow::Owned(narrowed) => Arc::from(narrowed),
                    Cow::Borrowed([]) => Arc::default(),
                    Cow::Borrowed(_) => Arc::clone(&spec.chat_tools),
                };
                reading::Auto {
                    request: symphony::AutoRequest::Messages {
                        request: Arc::clone(&messages_request),
                        tools,
                        continues: continued_final_message,
                    },
                    reasoning_open: require_reasoning,
                    thought_closed,
                }
            },
        )?;

        let (plan, stamp) = build_messages_backed_plan(
            ctx,
            &messages_request,
            processed_messages.text,
            token_ids,
            tool_constraints,
            table_tag,
            first_token,
            require_reasoning,
            multimodal_data,
            self.inject_pd_metadata,
            self.plan_kind,
        )?;
        spec.reasoning_in_prefill = require_reasoning;
        spec.symphony = assignment;
        Ok(BuildOutput {
            plan,
            spec: ResponseSpec::Messages(spec),
            stamp,
        })
    }

    fn name(&self) -> &'static str {
        "MessageRequestBuilding"
    }

    #[cfg(test)]
    fn signature(&self) -> String {
        format!(
            "MessageRequestBuildingStage(inject_pd_metadata={}, {:?})",
            self.inject_pd_metadata, self.plan_kind
        )
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::messages::CreateMessageRequest;
    use serde_json::json;

    use super::*;
    use crate::routers::grpc::{
        context::ClientSelection,
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
        let mut ctx = RequestContext::for_messages(
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
            false,
            ExecutionPlanKind::Single,
        )
        .map_err(|refusal| refusal.status())
        .expect("built");
        engine.abort();
        assert_fixed_request_plan(&plan, &stamp, "msg_");
    }
}
