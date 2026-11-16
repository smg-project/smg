//! The backend `GenerateRequest` the chat and Messages stages build alike
//! once the worker is known, and the stamp a retry re-mints it from. The
//! sequence is one ([`build`]): the request id stamped for the attempt
//! (`helpers::resolve_request_id_stamp`), the request built by the client for
//! the stage's request type with the constraint the table shaped for the
//! worker's start, the sampling defaults, the stops and the PD metadata
//! (`helpers::finish_generate_request`), and the attempt stamp, which carries
//! what derives the table's tag for a start (the `auto` tag, a JSON answer's),
//! so a retry gives the worker it selects the same shaping
//! (`helpers::ToolConstraintStamp`). Each stage keeps what
//! differs ([`Inputs`]): the request its client builds from, the sampling
//! mask, the id prefix, and where its media is assembled and its reasoning
//! decision taken.

use std::sync::Arc;

use axum::response::Response;
use llm_tokenizer::chat_template::ThinkingKeyName;
use openai_protocol::profile::ProviderProfile;
use smg_grpc_client::FormatSetAside;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        backend_client::BackendClient,
        client::GenerateRequestBuildOptions,
        common::stages::helpers::{self, SamplingDefaultsMask, TagForStart},
        context::{
            stage_output, AttemptStamp, ClientSelection, ExecutionPlan, PlanSettings,
            RequestContext, RequestType,
        },
        proto_wrapper::{MultimodalData, ProtoGenerateRequest},
    },
};

/// What a stage hands [`build`] beside the request its client builds from.
pub(crate) struct Inputs<'a> {
    /// The stage's name on the log lines and the errors of the sequence.
    pub function: &'static str,
    /// The prefix of a request id minted for the attempt
    /// (`helpers::IdStamp::Minted`).
    pub id_prefix: &'static str,
    /// The model the tool parser that shapes the constraint is resolved for.
    pub model: &'a str,
    /// The rendered prompt; it rides on the wire only when the request
    /// carries media (`helpers::wire_prompt_text`).
    pub processed_text: String,
    pub token_ids: Vec<u32>,
    /// The constraint as the request carries it, before the worker's shaping.
    pub tool_constraints: Option<(String, String)>,
    /// Which of the body's own format fields the constraint answers (a
    /// table's tag for a JSON `response_format`; the whole turn's grammar for
    /// the `regex` and the `ebnf` too): the client's builders read them as
    /// absent, so the engine's request carries the constraint alone.
    pub format_set_aside: FormatSetAside,
    /// What derives the table's tag for a worker's start, where the table
    /// shapes it by the start (`regular::symphony::constraint::WorkerConstraint::table_tag`).
    pub table_tag: Option<Arc<dyn TagForStart>>,
    /// The constraint was shaped for an engine that starts the grammar at the
    /// first generated token.
    pub first_token_tag: bool,
    /// The rendered prompt leaves the model inside its thought.
    pub require_reasoning: bool,
    /// The request's media, assembled for the engine.
    pub multimodal_data: Option<MultimodalData>,
    /// Which sampling knobs the client left unset, for the worker's defaults.
    pub sampling_mask: Option<SamplingDefaultsMask>,
    /// The mode's plan settings: the plan the request becomes, the PD metadata.
    pub settings: PlanSettings,
}

/// Build the backend `GenerateRequest` from `inputs` through `build_request`,
/// the client's builder for the stage's request type (its arguments: the
/// builder client, the request id, the prompt text for the wire, the token
/// ids and the build options), and return the plan and the attempt stamp.
/// The errors are the stage's: a missing client selection, a constraint the
/// worker's shaping refuses, a request the builder refuses, and what
/// `helpers::finish_generate_request` refuses.
pub(crate) fn build(
    ctx: &mut RequestContext,
    inputs: Inputs<'_>,
    build_request: impl FnOnce(
        &BackendClient,
        String,
        String,
        Vec<u32>,
        GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String>,
) -> Result<(ExecutionPlan, AttemptStamp), Response> {
    let Inputs {
        function,
        id_prefix,
        model,
        processed_text,
        token_ids,
        tool_constraints,
        format_set_aside,
        table_tag,
        first_token_tag,
        require_reasoning,
        multimodal_data,
        sampling_mask,
        settings,
    } = inputs;
    let clients = stage_output(ctx.state.clients.as_ref(), function)?;

    let builder_client = clients.builder();
    let disaggregated = matches!(clients, ClientSelection::Disaggregated { .. });
    let (request_id, id_stamp) = helpers::resolve_request_id_stamp(
        &ctx.input.request_type,
        ctx.input.tenant_request_meta.as_ref(),
        id_prefix,
        disaggregated,
    );

    // The ids are the prompt; the text rides along only when the request
    // carries media (see `helpers::wire_prompt_text`).
    let carries_media = multimodal_data.is_some() || ctx.state.multimodal_refs.is_some();
    let mut proto_request = build_request(
        builder_client,
        request_id,
        helpers::wire_prompt_text(processed_text, carries_media),
        token_ids,
        GenerateRequestBuildOptions {
            multimodal_inputs: multimodal_data,
            tool_constraints,
            format_set_aside,
            require_reasoning,
            first_token_grammar: first_token_tag,
        },
    )
    .map_err(|e| {
        error!(function = function, error = %e, "Failed to build generate request");
        error::bad_request(
            "invalid_request_parameters",
            format!("Invalid request parameters: {e}"),
        )
    })?;

    if let ProtoGenerateRequest::Vllm(request) = &mut proto_request {
        let provider = ctx
            .state
            .provider
            .unwrap_or_else(|| ProviderProfile::for_model(model));
        let key = ctx
            .state
            .tokenizer
            .as_ref()
            .and_then(|t| t.thinking_key_name());
        request.chat_template_kwargs_json =
            thinking_mode_kwargs(&ctx.input.request_type, provider, key);
    }

    let sampling_baseline = helpers::apply_sampling_defaults(
        &mut proto_request,
        sampling_mask,
        ctx.state.workers.as_ref(),
    );

    helpers::finish_generate_request(
        ctx,
        &mut proto_request,
        settings.inject_pd_metadata,
        function,
    )?;

    let tool_constraint =
        table_tag.and_then(|tag| helpers::ToolConstraintStamp::capture_table(tag, &proto_request));
    Ok((
        ExecutionPlan::generate(settings.plan_kind, proto_request),
        AttemptStamp {
            id: id_stamp,
            sampling_mask,
            sampling_baseline,
            inject_pd_metadata: settings.inject_pd_metadata,
            tool_constraint,
        },
    ))
}

/// Match the template's tri-state switch, including explicit kwargs precedence.
/// A missing mode stays absent so the engine retains its adaptive default.
fn thinking_mode_kwargs(
    request: &RequestType,
    provider: ProviderProfile,
    key: Option<ThinkingKeyName>,
) -> Option<String> {
    if key != Some(ThinkingKeyName::ThinkingMode) {
        return None;
    }
    let thinking = match request {
        RequestType::Chat(request) => {
            if let Some(mode) = request
                .chat_template_kwargs
                .as_ref()
                .and_then(|k| k.get("thinking_mode"))
            {
                return Some(serde_json::json!({"thinking_mode": mode}).to_string());
            }
            request.template_thinking_for(provider)
        }
        RequestType::Messages(request) => request.template_thinking_for(provider),
        _ => None,
    }?;
    Some(
        serde_json::json!({"thinking_mode": if thinking { "enabled" } else { "disabled" }})
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Match the rendered tri-state mode (the typed toggle, else the profile's
    /// reading of `reasoning_effort`) and leave other template switches untouched.
    #[test]
    fn minimax_thinking_mode_matches_template_precedence() {
        for (mut body, expected) in [
            (json!({}), None),
            (json!({"thinking": {"type": "adaptive"}}), None),
            (
                json!({"thinking": {"type": "enabled"}}),
                Some(json!("enabled")),
            ),
            (
                json!({"thinking": {"type": "disabled"}}),
                Some(json!("disabled")),
            ),
            (
                json!({"thinking": {"type": "enabled"}, "chat_template_kwargs": {"thinking_mode": "adaptive"}}),
                Some(json!("adaptive")),
            ),
            (
                json!({"thinking": {"type": "disabled"}, "chat_template_kwargs": {"thinking_mode": "enabled"}}),
                Some(json!("enabled")),
            ),
            (
                json!({"chat_template_kwargs": {"thinking_mode": null}}),
                Some(json!(null)),
            ),
            // The profile's reading of the effort: `none` is thinking off under
            // MiniMax; `minimal` is its lowest level, thinking on, so the
            // template's default mode stands.
            (json!({"reasoning_effort": "none"}), Some(json!("disabled"))),
            (json!({"reasoning_effort": "minimal"}), None),
            (json!({"reasoning_effort": "high"}), None),
        ] {
            body["model"] = json!("minimax-m3");
            body["messages"] = json!([]);
            let request = RequestType::Chat(Arc::new(serde_json::from_value(body).unwrap()));
            let actual = thinking_mode_kwargs(
                &request,
                ProviderProfile::Minimax,
                Some(ThinkingKeyName::ThinkingMode),
            )
            .map(|s| serde_json::from_str::<serde_json::Value>(&s).unwrap());
            assert_eq!(actual, expected.map(|mode| json!({"thinking_mode": mode})));
            for key in [
                None,
                Some(ThinkingKeyName::EnableThinking),
                Some(ThinkingKeyName::Thinking),
                Some(ThinkingKeyName::ReasoningEffort),
            ] {
                assert_eq!(
                    thinking_mode_kwargs(&request, ProviderProfile::Minimax, key),
                    None
                );
            }
        }
    }
}
