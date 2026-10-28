//! Chat request building stage: Build proto GenerateRequest for chat requests

use std::sync::Arc;

use axum::response::Response;

use crate::routers::{
    error,
    grpc::{
        common::stages::helpers::{self, TagForStart},
        context::{
            stage_output, AttemptStamp, BuildOutput, ExecutionPlan, ExecutionPlanKind,
            PreparationOutput, RequestContext,
        },
        regular::{
            stages::{backed_plan, reading},
            symphony,
        },
        spec::{ChatResponseSpec, ResponseSpec},
        utils,
    },
};

/// Chat request building stage
///
/// Extracts chat-specific request building logic from the old unified RequestBuildingStage.
pub(crate) struct ChatRequestBuildingStage {
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
}

impl ChatRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool, plan_kind: ExecutionPlanKind) -> Self {
        Self {
            inject_pd_metadata,
            plan_kind,
        }
    }
}

/// [`reading::reasoning_decision`] with a chat request's tokenizer-driven
/// switch (`utils::chat_reasoning_starts_in_rendered_prefill`);
/// `generation_prompt` and `continued_final_message` are the rendered prompt's.
fn reasoning_decision(
    ctx: &RequestContext,
    chat_request: &openai_protocol::chat::ChatCompletionRequest,
    processed_text: &str,
    generation_prompt: bool,
    continued_final_message: bool,
) -> (bool, bool) {
    reading::reasoning_decision(ctx, &chat_request.model, processed_text, |tokenizer| {
        utils::chat_reasoning_starts_in_rendered_prefill(
            chat_request,
            generation_prompt,
            continued_final_message,
            tokenizer,
        )
    })
}

/// Build the backend `GenerateRequest` from a chat-shaped request, shared by
/// the chat endpoint and the transcription endpoint: the request's media
/// assembled and its reasoning decision taken, then the sequence the chat and
/// Messages stages share ([`backed_plan::build`]) with the chat request's
/// client builder and sampling mask. Returns the plan, the attempt stamp and
/// the rendered reasoning state; the caller supplies the `ResponseSpec`.
/// `generation_prompt` and `continued_final_message` are the rendered
/// prompt's (`ProcessedMessages`). `table_tag` derives the
/// table's tag for a worker's grammar start when `tool_constraints` is one
/// the table shapes by the start, the `auto` tag or a JSON answer's
/// (`regular::symphony::constraint::WorkerConstraint::table_tag`).
#[expect(clippy::too_many_arguments)]
pub(crate) async fn build_chat_backed_plan(
    ctx: &mut RequestContext,
    chat_request: &openai_protocol::chat::ChatCompletionRequest,
    processed_text: String,
    generation_prompt: bool,
    continued_final_message: bool,
    token_ids: Vec<u32>,
    tool_constraints: Option<(String, String)>,
    table_tag: Option<Arc<dyn TagForStart>>,
    first_token_tag: bool,
    id_prefix: &'static str,
    inject_pd_metadata: bool,
    plan_kind: ExecutionPlanKind,
) -> Result<(ExecutionPlan, AttemptStamp, bool), Response> {
    let multimodal_data = helpers::assemble_prompt_media(ctx, "build_chat_backed_plan").await?;

    // The plan needs the decision alone; the prompt's closed thought shapes the
    // `auto` tag in `reading::decide`, which reads it with the decision.
    let (require_reasoning, _thought_closed) = reasoning_decision(
        ctx,
        chat_request,
        &processed_text,
        generation_prompt,
        continued_final_message,
    );

    let (plan, stamp) = backed_plan::build(
        ctx,
        backed_plan::Inputs {
            function: "build_chat_backed_plan",
            id_prefix,
            model: &chat_request.model,
            processed_text,
            token_ids,
            tool_constraints,
            table_tag,
            first_token_tag,
            require_reasoning,
            multimodal_data,
            sampling_mask: Some(helpers::SamplingDefaultsMask::from_chat_request(
                chat_request,
            )),
            inject_pd_metadata,
            plan_kind,
        },
        |client, request_id, processed_text, token_ids, options| {
            client.build_chat_request(request_id, chat_request, processed_text, token_ids, options)
        },
    )?;
    Ok((plan, stamp, require_reasoning))
}

impl ChatRequestBuildingStage {
    /// The stage's work. The chat+generate dispatcher calls this directly, so
    /// the stage's future lives inline in the dispatcher's: one box per stage
    /// call, not a second one for a boxed trait method.
    pub(crate) async fn build_plan(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<BuildOutput, Response> {
        // Take preparation state (last consumer — worker_selection already ran)
        let prep = stage_output(
            ctx.state.preparation.take(),
            "ChatRequestBuildingStage::build",
        )?;

        let chat_request = ctx.chat_request_arc();

        let PreparationOutput::Chat {
            token_ids,
            processed_messages,
            tool_constraints,
        } = prep
        else {
            debug_assert!(false, "pipeline guarantees Chat variant");
            return Err(error::internal_error(
                "wrong_preparation_type",
                "Expected Chat preparation output",
            ));
        };

        let unbilled_prompt_tokens = processed_messages.unbilled_prompt_tokens;
        // What constrains the output on the selected worker, and Symphony's
        // assignment, decided as for a Messages request (`reading::decide`):
        // the `auto` tag derives for the request over the tools its
        // `tool_choice` leaves callable, on the reasoning state the plan takes
        // (`reasoning_decision`). The state records which parser reads the
        // response, for the response's header.
        let reading::Decided {
            table_tag,
            tool_constraints,
            whole_turn,
            first_token,
            assignment,
        } = reading::decide(
            ctx,
            symphony::Asks::Chat(&chat_request),
            &chat_request.effective_tool_list(),
            &processed_messages.text,
            tool_constraints,
            |ctx| {
                let (reasoning_open, thought_closed) = reasoning_decision(
                    ctx,
                    &chat_request,
                    &processed_messages.text,
                    processed_messages.generation_prompt,
                    processed_messages.continued_final_message,
                );
                reading::Auto {
                    request: symphony::AutoRequest::Chat(Arc::clone(&chat_request)),
                    reasoning_open,
                    thought_closed,
                }
            },
        )?;
        // The table's structural tag stands in for a JSON response_format (see
        // `regular::symphony::constraint`), and the whole turn's grammar for a
        // regex or an ebnf too: the engine's request carries the tag alone, so
        // no client reads the two as competing constraints.
        let answered = assignment.is_some()
            && chat_request.response_format.is_some()
            && tool_constraints
                .as_ref()
                .is_some_and(|(kind, _)| kind == "structural_tag");
        let engine_request = (answered || whole_turn).then(|| {
            let mut request = chat_request.as_ref().clone();
            request.response_format = None;
            if whole_turn {
                request.regex = None;
                request.ebnf = None;
            }
            request
        });
        let engine_request = match &engine_request {
            Some(request) => request,
            None => chat_request.as_ref(),
        };
        let generation_prompt = processed_messages.generation_prompt;
        let continued_final_message = processed_messages.continued_final_message;
        let (plan, stamp, reasoning_in_prefill) = build_chat_backed_plan(
            ctx,
            engine_request,
            processed_messages.text,
            generation_prompt,
            continued_final_message,
            token_ids,
            tool_constraints,
            table_tag,
            first_token,
            "chatcmpl-",
            self.inject_pd_metadata,
            self.plan_kind,
        )
        .await?;

        // Only the client-facing usage drops them; settlement keeps the engine's count.
        let mut spec = ChatResponseSpec::from(chat_request.as_ref());
        spec.unbilled_prompt_tokens = unbilled_prompt_tokens;
        spec.reasoning_in_prefill = reasoning_in_prefill;
        spec.symphony = assignment;

        Ok(BuildOutput {
            plan,
            spec: ResponseSpec::Chat(Box::new(spec)),
            stamp,
        })
    }

    /// Stable descriptor of the stage plus its mode-bearing args, compared
    /// against golden literals in the pipeline parity test (through the
    /// chat+generate dispatcher's own `signature`).
    #[cfg(test)]
    pub(crate) fn signature(&self) -> String {
        format!(
            "ChatRequestBuildingStage(inject_pd_metadata={}, {:?})",
            self.inject_pd_metadata, self.plan_kind
        )
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::chat::ChatCompletionRequest;
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
            "<prompt>".to_string(),
            true,
            false,
            vec![1, 2, 3],
            Some(("structural_tag".to_string(), FORCED_CALL_TAG.to_string())),
            None,
            false,
            "chatcmpl-",
            false,
            ExecutionPlanKind::Single,
        )
        .await
        .map_err(|refusal| refusal.status())
        .expect("built");
        engine.abort();
        assert!(!reasoning_in_prefill);
        assert_fixed_request_plan(&plan, &stamp, "chatcmpl-");
    }
}
