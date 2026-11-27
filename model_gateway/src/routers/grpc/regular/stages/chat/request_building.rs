//! Chat request building stage: Build proto GenerateRequest for chat requests

use std::sync::Arc;

use axum::response::Response;
use openai_protocol::profile::ProviderProfile;
use smg_grpc_client::FormatSetAside;

use crate::routers::{
    error,
    grpc::{
        common::stages::helpers::{self, TagForStart},
        context::{
            stage_output, AttemptStamp, BuildOutput, ExecutionPlan, PlanSettings,
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
    settings: PlanSettings,
}

impl ChatRequestBuildingStage {
    pub fn new(settings: PlanSettings) -> Self {
        Self { settings }
    }
}

/// [`reading::reasoning_decision`] with a chat request's tokenizer-driven
/// switch (`utils::chat_reasoning_starts_in_rendered_prefill_for`, over
/// `provider`, the model's profile as the request's preparation derived it);
/// `generation_prompt` and `continued_final_message` are the rendered prompt's.
fn reasoning_decision(
    ctx: &RequestContext,
    chat_request: &openai_protocol::chat::ChatCompletionRequest,
    provider: ProviderProfile,
    processed_text: &str,
    generation_prompt: bool,
    continued_final_message: bool,
) -> (bool, bool) {
    reading::reasoning_decision(ctx, &chat_request.model, processed_text, |tokenizer| {
        utils::chat_reasoning_starts_in_rendered_prefill_for(
            chat_request,
            provider,
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
/// `provider` is the model's profile as the request's preparation derived it
/// (`ProcessingState::provider`); `generation_prompt` and
/// `continued_final_message` are the rendered prompt's (`ProcessedMessages`).
/// `format_set_aside` names which of the body's own format fields
/// `tool_constraints` answers, so the builders read them as absent and the
/// engine's request carries the constraint alone. `table_tag` derives the
/// table's tag for a worker's grammar start when `tool_constraints` is one
/// the table shapes by the start, the `auto` tag or a JSON answer's
/// (`regular::symphony::constraint::WorkerConstraint::table_tag`).
#[expect(clippy::too_many_arguments)]
pub(crate) async fn build_chat_backed_plan(
    ctx: &mut RequestContext,
    chat_request: &openai_protocol::chat::ChatCompletionRequest,
    provider: ProviderProfile,
    processed_text: String,
    generation_prompt: bool,
    continued_final_message: bool,
    token_ids: Vec<u32>,
    tool_constraints: Option<(String, String)>,
    format_set_aside: FormatSetAside,
    table_tag: Option<Arc<dyn TagForStart>>,
    first_token_tag: bool,
    id_prefix: &'static str,
    settings: PlanSettings,
) -> Result<(ExecutionPlan, AttemptStamp, bool), Response> {
    let multimodal_data = helpers::assemble_prompt_media(ctx, "build_chat_backed_plan").await?;

    // The plan needs the decision alone; the prompt's closed thought shapes the
    // `auto` tag in `reading::decide`, which reads it with the decision.
    let (require_reasoning, _thought_closed) = reasoning_decision(
        ctx,
        chat_request,
        provider,
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
            return_prompt_text: chat_request.return_prompt_text,
            token_ids,
            tool_constraints,
            format_set_aside,
            table_tag,
            first_token_tag,
            require_reasoning,
            multimodal_data,
            sampling_mask: Some(helpers::SamplingDefaultsMask::from_chat_request(
                chat_request,
            )),
            settings,
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
        // The image tokens the expansion wrote into the prompt, read before
        // the plan's assembly takes the intermediate; the usage reports them.
        let image_tokens = ctx
            .state
            .multimodal_intermediate
            .as_ref()
            .and_then(|intermediate| intermediate.image_prompt_tokens());
        // The model's profile as preparation derived it, read by the
        // reasoning decision, the plan and the response spec below.
        let provider = ctx
            .state
            .provider
            .unwrap_or_else(|| ProviderProfile::for_model(&chat_request.model));
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
            &chat_request.effective_tool_list_for(provider),
            &processed_messages.text,
            tool_constraints,
            |ctx| {
                let (reasoning_open, thought_closed) = reasoning_decision(
                    ctx,
                    &chat_request,
                    provider,
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
        let format_set_aside = set_aside_for(
            assignment.is_some(),
            chat_request.response_format.is_some(),
            tool_constraints.as_ref(),
            whole_turn,
        );
        let generation_prompt = processed_messages.generation_prompt;
        let continued_final_message = processed_messages.continued_final_message;
        let (plan, stamp, reasoning_in_prefill) = build_chat_backed_plan(
            ctx,
            &chat_request,
            provider,
            processed_messages.text,
            generation_prompt,
            continued_final_message,
            token_ids,
            tool_constraints,
            format_set_aside,
            table_tag,
            first_token,
            "chatcmpl-",
            self.settings,
        )
        .await?;

        // Only the client-facing usage drops them; settlement keeps the engine's count.
        let mut spec = ChatResponseSpec::for_request(chat_request.as_ref(), provider);
        spec.unbilled_prompt_tokens = unbilled_prompt_tokens;
        spec.image_tokens = image_tokens;
        spec.reasoning_in_prefill = reasoning_in_prefill;
        spec.symphony = assignment;
        spec.parser_names.clone_from(&ctx.state.parser_names);

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
        format!("ChatRequestBuildingStage({})", self.settings.signature())
    }
}

/// Which of the body's own format fields the constraint handed to the
/// builders answers, so that they read those as absent and the engine's
/// request is built from the client's request as sent, with no two competing
/// constraints in it: the `response_format` when the table's structural tag
/// stands in for the JSON answer (Symphony reads the response, the request
/// names a format and the constraint is a tag; see
/// `regular::symphony::constraint`) or when the whole turn's grammar is the
/// constraint, which stands in for a `regex` or an `ebnf` too.
fn set_aside_for(
    symphony_reads: bool,
    names_a_format: bool,
    tool_constraints: Option<&(String, String)>,
    whole_turn: bool,
) -> FormatSetAside {
    let answered = symphony_reads
        && names_a_format
        && tool_constraints.is_some_and(|(kind, _)| kind == "structural_tag");
    FormatSetAside {
        response_format: answered || whole_turn,
        grammar: whole_turn,
    }
}

#[cfg(test)]
#[path = "../../../../../../tests/unit/routers/grpc/regular/stages/chat/request_building/tests.rs"]
mod tests;
