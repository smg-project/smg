//! The backend `GenerateRequest` the chat and Messages stages build alike
//! once the worker is known, and the stamp a retry re-mints it from. The
//! sequence is one ([`build`]): the request id stamped for the attempt
//! (`helpers::resolve_request_id_stamp`), the forced call's grammar shaped
//! for the worker (the bare calls behind a reasoning parser's gate, the
//! thought and then the calls from the first token: the worker's
//! `constrained_decoding_start` label, `utils::tool_constraints_for_worker`),
//! the request built by the client for the stage's request type, the
//! sampling defaults, the stops and the PD metadata
//! (`helpers::finish_generate_request`), and the attempt stamp, which carries
//! the unshaped constraint, or what derives the table's tag for a start (the
//! `auto` tag, a JSON answer's), so a retry gives the worker it selects the
//! same shaping (`helpers::ToolConstraintStamp`). Each stage keeps what
//! differs ([`Inputs`]): the request its client builds from, the sampling
//! mask, the id prefix, and where its media is assembled and its reasoning
//! decision taken.

use std::sync::Arc;

use axum::response::Response;
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        backend_client::BackendClient,
        client::GenerateRequestBuildOptions,
        common::stages::helpers::{self, SamplingDefaultsMask, TagForStart},
        context::{
            stage_output, AttemptStamp, ClientSelection, ExecutionPlan, ExecutionPlanKind,
            RequestContext,
        },
        proto_wrapper::{MultimodalData, ProtoGenerateRequest},
        utils,
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
    pub inject_pd_metadata: bool,
    pub plan_kind: ExecutionPlanKind,
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
        table_tag,
        first_token_tag,
        require_reasoning,
        multimodal_data,
        sampling_mask,
        inject_pd_metadata,
        plan_kind,
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

    // Now that the worker is known, a forced call's grammar takes the shape
    // its engine needs: the bare calls behind a reasoning parser's gate, the
    // thought and then the calls from the first token (the worker's
    // `constrained_decoding_start` label; see `utils::tool_constraints_for_worker`).
    // The unshaped constraint stays on the attempt stamp, or what derives the
    // table's tag for a start (the `auto` tag, a JSON answer's): a retry gives
    // the worker it selects the same shaping (`helpers::ToolConstraintStamp`).
    let tool_parser = ctx.components.parser_resolver.tool_parser(model);
    let unshaped_tool_constraints = tool_constraints.clone();
    let (tool_constraints, led_for_first_token) = utils::tool_constraints_for_worker(
        &ctx.components.tool_parser_factory,
        tool_parser.as_deref(),
        tool_constraints,
        ctx.state.workers.as_ref(),
    )
    .map_err(|e| {
        error!(function = function, error = %e, "Failed to shape the tool constraint for the worker");
        error::internal_error("invalid_tool_constraint", format!("Invalid tool constraint: {e}"))
    })?;

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
            require_reasoning,
            first_token_grammar: first_token_tag || led_for_first_token,
        },
    )
    .map_err(|e| {
        error!(function = function, error = %e, "Failed to build generate request");
        error::bad_request(
            "invalid_request_parameters",
            format!("Invalid request parameters: {e}"),
        )
    })?;

    let sampling_baseline = helpers::apply_sampling_defaults(
        &mut proto_request,
        sampling_mask,
        ctx.state.workers.as_ref(),
    );

    helpers::finish_generate_request(ctx, &mut proto_request, inject_pd_metadata, function)?;

    let tool_constraint = match table_tag {
        Some(table_tag) => helpers::ToolConstraintStamp::capture_table(table_tag, &proto_request),
        None => helpers::ToolConstraintStamp::capture(
            unshaped_tool_constraints,
            tool_parser.as_deref(),
            first_token_tag,
            &proto_request,
        ),
    };
    Ok((
        ExecutionPlan::generate(plan_kind, proto_request),
        AttemptStamp {
            id: id_stamp,
            sampling_mask,
            sampling_baseline,
            inject_pd_metadata,
            tool_constraint,
        },
    ))
}
