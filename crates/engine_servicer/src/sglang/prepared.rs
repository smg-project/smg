//! Prepared-input handoff on the ordinary Generate RPC and request registry.
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use prost::Message;
use smg_grpc_client::{
    sglang_managed::{proto, validate_binding, Limits, VERSION},
    sglang_proto as sg,
};
use tonic::Status;

use super::{
    native::{ManagedNativeRuntime, PreparedRequest},
    State,
};
use crate::BoxStream;

/// Register before native submission. P exports the layout and D binds that exact layout;
/// neither side falls back to media preprocessing.
pub(super) fn generate(
    state: &Arc<State>,
    request: sg::GenerateRequest,
) -> Result<BoxStream<sg::GenerateResponse>, Status> {
    let (registration, cancel) = state.shared.registry.register(&request.request_id)?;
    let (runtime, input) = admission(state, request)?;
    runtime.submit(input, registration, cancel)
}

/// Validate prepared-input control/layout envelopes against the worker contract and bounds.
fn admission(
    state: &Arc<State>,
    mut request: sg::GenerateRequest,
) -> Result<(&Arc<ManagedNativeRuntime>, PreparedRequest), Status> {
    let config = state
        .managed_config
        .as_ref()
        .ok_or_else(|| Status::failed_precondition("worker does not support prepared inputs"))?;
    let runtime = state
        .prepared
        .get()
        .ok_or_else(|| Status::unavailable("prepared runtime is not ready"))?;
    if !runtime.ready() {
        return Err(Status::unavailable("prepared runtime is unhealthy"));
    }
    let limits = Limits::default();
    if request.prepared_control.len() > limits.manifest_bytes
        || request.prepared_input.len() > limits.manifest_bytes
    {
        return Err(Status::resource_exhausted(
            "prepared metadata exceeds byte limit",
        ));
    }
    let control = proto::PrefillStart::decode(request.prepared_control.as_slice())
        .map_err(|_| Status::invalid_argument("malformed prepared control"))?;
    let binding = control
        .binding
        .ok_or_else(|| Status::invalid_argument("missing prepared binding"))?;
    validate_binding(&binding)?;
    if control.version != VERSION
        || binding.model_revision != config.model_revision
        || binding.processor_contract != config.processor_contract
    {
        return Err(Status::failed_precondition("prepared contract mismatch"));
    }
    if request.mm_inputs.is_some()
        || !request.extra_mm_inputs.is_empty()
        || !request.input_embeds.is_empty()
        || request.media_identity.is_some()
        || request.sampling_params.as_ref().is_none_or(|p| p.n > 1)
        || !request.lora_id.is_empty()
        || request.return_hidden_states
    {
        return Err(Status::invalid_argument(
            "prepared inputs require n=1 without raw media, LoRA or hidden states",
        ));
    }
    if control.timeout_ms == 0 || control.timeout_ms > config.max_duration_ms {
        return Err(Status::invalid_argument("invalid prepared timeout"));
    }
    let deadline = Instant::now() + Duration::from_millis(control.timeout_ms);
    let (external, manifest) = if config.role == "prefill" {
        if request.request_id != binding.prefill_rid || !request.prepared_input.is_empty() {
            return Err(Status::invalid_argument("invalid prepared prefill request"));
        }
        if control.image_count as usize > limits.images
            || (control.image_count > 0 && control.encoder_receive_endpoint.is_empty())
        {
            return Err(Status::invalid_argument("invalid prepared encode plan"));
        }
        (
            Some((control.encoder_receive_endpoint, control.image_count)),
            None,
        )
    } else {
        if request.request_id != binding.decode_rid || request.prepared_input.is_empty() {
            return Err(Status::invalid_argument("invalid prepared decode request"));
        }
        let manifest = proto::PreparedDecodeInput::decode(request.prepared_input.as_slice())
            .map_err(|_| Status::invalid_argument("malformed prepared input"))?;
        runtime.validate_layout(&manifest, &binding)?;
        (None, Some(Box::new(manifest)))
    };
    request.prepared_control.clear();
    request.prepared_input.clear();
    Ok((
        runtime,
        PreparedRequest {
            request,
            binding,
            external,
            manifest,
            deadline,
        },
    ))
}
