//! Prepared-input handoff on the ordinary Generate RPC and request registry.
use std::{
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use futures::StreamExt;
use prost::Message;
use smg_grpc_client::{
    sglang_managed::{
        budget::Deadline, clock, proto, validate_binding, validate_external_plan,
        validate_manifest, Limits, VERSION,
    },
    sglang_proto as sg,
};
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

use super::State;
use crate::{
    requests::{CancelReason, Registration},
    BoxStream,
};

/// Register before native submission. Only P may export layout, and D must bind
/// that exact layout; neither side may fall back to media preprocessing.
pub(super) fn generate(
    state: &Arc<State>,
    request: sg::GenerateRequest,
) -> Result<BoxStream<sg::GenerateResponse>, Status> {
    // Own the unique rid before validation: even a malformed duplicate must
    // not claim that the earlier request has no native work. Failed admission
    // drops this registration immediately, without a tombstone.
    let (registration, cancel) = state
        .shared
        .registry
        .register(&request.request_id)
        .map_err(|status| {
            // register checks live duplicates first. Its only Unavailable
            // outcomes are the warming/draining holds, before insertion.
            // Poisoned state and existing owners remain ambiguous.
            if status.code() == tonic::Code::Unavailable {
                smg_grpc_client::sglang_managed::not_submitted(status)
            } else {
                status
            }
        })?;
    admission(state, request, false)
        .and_then(|input| input.submit(state, registration, cancel, None))
        .map_err(|status| {
            // submit can still find an existing operation under its binding. Only
            // a proven pre-submit rejection may carry the receipt; stream errors
            // after the native owner starts never pass through this function.
            if status.code() == tonic::Code::AlreadyExists {
                status
            } else {
                smg_grpc_client::sglang_managed::not_submitted(status)
            }
        })
}

struct Admission {
    request: sg::GenerateRequest,
    operation: proto::OperationRef,
    binding: proto::InputBinding,
    external: Option<proto::ExternalEncodePlan>,
    manifest: Option<Box<proto::PreparedDecodeInput>>,
    deadline: Deadline,
}

impl Admission {
    fn submit(
        self,
        state: &Arc<State>,
        registration: Registration,
        cancel: oneshot::Receiver<CancelReason>,
        context: Option<OwnedSemaphorePermit>,
    ) -> Result<BoxStream<sg::GenerateResponse>, Status> {
        state
            .prepared
            .get()
            .ok_or_else(|| Status::unavailable("prepared runtime unavailable"))?
            .submit(
                self.operation,
                self.binding,
                self.request,
                self.external,
                self.manifest,
                self.deadline,
                registration,
                cancel,
                context,
            )
    }
}

fn admission(
    state: &Arc<State>,
    mut request: sg::GenerateRequest,
    deferred: bool,
) -> Result<Admission, Status> {
    let config = state
        .managed_config
        .as_ref()
        .ok_or_else(|| Status::failed_precondition("worker does not support prepared inputs"))?;
    clock::check(config.max_clock_error_ms)?;
    let runtime = state
        .prepared
        .get()
        .ok_or_else(|| Status::unavailable("prepared runtime is not ready"))?;
    if !runtime.ready() {
        return Err(Status::unavailable("prepared runtime is unhealthy"));
    }
    let limits = Limits::default();
    if request.prepared_control.len() + request.prepared_input.len() > limits.manifest_bytes {
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
    if control.generation.is_some()
        || control.version != VERSION
        || binding.model_revision != config.contract.model_revision
        || binding.processor_contract != config.contract.processor_contract
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
    if deferred && (config.role != "decode" || !request.prepared_input.is_empty()) {
        return Err(Status::invalid_argument(
            "Start requires a decode worker and no prepared input",
        ));
    }
    let (operation, external, manifest) = if config.role == "prefill" {
        if !request.prepared_input.is_empty() || request.request_id != binding.prefill_rid {
            return Err(Status::invalid_argument("invalid prepared prefill request"));
        }
        let operation = control
            .operation
            .ok_or_else(|| Status::invalid_argument("missing prefill operation"))?;
        let decode = binding
            .decode
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing decode binding"))?;
        if operation.identity != decode.identity
            || operation.request_digest != decode.request_digest
            || operation.plan_digest != decode.plan_digest
            || operation.target != binding.prefill
        {
            return Err(Status::failed_precondition(
                "prefill operation differs from binding",
            ));
        }
        let plan = control
            .encode_plan
            .ok_or_else(|| Status::invalid_argument("missing encode plan"))?;
        let images = plan
            .image_counts
            .iter()
            .try_fold(0usize, |n, count| n.checked_add(*count as usize))
            .ok_or_else(|| Status::invalid_argument("image count overflow"))?;
        validate_external_plan(&plan, images)?;
        (operation, Some(plan), None)
    } else {
        let input = if deferred {
            None
        } else {
            let input = proto::PreparedDecodeInput::decode(request.prepared_input.as_slice())
                .map_err(|_| Status::invalid_argument("missing or malformed prepared input"))?;
            validate_manifest(&input, &binding, limits)?;
            Some(Box::new(input))
        };
        let operation = binding
            .decode
            .clone()
            .ok_or_else(|| Status::invalid_argument("missing decode operation"))?;
        if request.request_id != operation.operation_id {
            return Err(Status::invalid_argument(
                "decode request id differs from binding",
            ));
        }
        (operation, None, input)
    };
    if operation.target.as_ref().is_none_or(|target| {
        target.worker_id != config.worker_id || target.generation != config.generation
    }) {
        return Err(Status::failed_precondition(
            "stale prepared worker instance",
        ));
    }
    let budget = control
        .budget
        .ok_or_else(|| Status::invalid_argument("missing prepared deadline"))?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Status::unavailable("invalid clock"))?
        .as_millis() as u64;
    let deadline = Deadline::admit(&budget, now, Instant::now(), 0, 0, config.max_duration_ms)?;
    request.prepared_control.clear();
    request.prepared_input.clear();
    Ok(Admission {
        request,
        operation,
        binding,
        external,
        manifest,
        deadline,
    })
}

#[tonic::async_trait]
impl proto::managed_decode_server::ManagedDecode for super::service::SglangService {
    type GenerateStream = BoxStream<sg::GenerateResponse>;

    async fn generate(
        &self,
        request: Request<Streaming<proto::DecodeCommand>>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        use proto::decode_command::Command;
        let mut commands = request.into_inner();
        let Some(proto::DecodeCommand {
            command: Some(Command::Start(start)),
        }) = commands.message().await?
        else {
            return Err(Status::invalid_argument(
                "decode stream must begin with Start",
            ));
        };
        let (registration, mut cancel) = self.state.shared.registry.register(&start.request_id)?;
        let start_bytes = start.encode_to_vec();
        let input = admission(&self.state, start, true)?;
        let deadline = tokio::time::Instant::now() + input.deadline.remaining(Instant::now())?;
        drop(input);
        let context = self
            .state
            .prepared
            .get()
            .ok_or_else(|| Status::unavailable("prepared runtime unavailable"))?
            .reserve_start(start_bytes.capacity())?;
        let (output, receiver) = mpsc::channel(8);
        let state = Arc::clone(&self.state);
        #[expect(
            clippy::disallowed_methods,
            reason = "WaitingInput owns the existing request registration until bind or cancellation"
        )]
        tokio::spawn(async move {
            let bind = async {
                let manifest = match commands.message().await? {
                    Some(proto::DecodeCommand {
                        command: Some(Command::BindPreparedInput(manifest)),
                    }) => manifest,
                    Some(proto::DecodeCommand {
                        command: Some(Command::Cancel(reason)),
                    }) => return Err(Status::cancelled(reason)),
                    _ => {
                        return Err(Status::invalid_argument(
                            "Start requires exactly one BindPreparedInput",
                        ))
                    }
                };
                let start = sg::GenerateRequest::decode(start_bytes.as_slice())
                    .map_err(|_| Status::internal("stored decode Start malformed"))?;
                let mut input = admission(&state, start, true)?;
                validate_manifest(&manifest, &input.binding, Limits::default())?;
                input.manifest = Some(Box::new(manifest));
                Ok(input)
            };
            let bound = tokio::select! {
                result = tokio::time::timeout_at(deadline, bind) => result.unwrap_or_else(|_| Err(Status::deadline_exceeded("prepared input deadline expired"))),
                _ = &mut cancel => Err(Status::cancelled("decode Start cancelled")),
                () = output.closed() => return,
            };
            drop(start_bytes);
            let (input, binding, digest) = match bound {
                Ok(input) => {
                    let binding = input.binding.clone();
                    let digest = input
                        .manifest
                        .as_ref()
                        .map(|input| input.integrity_digest.clone());
                    (input, binding, digest)
                }
                Err(error) => {
                    let _ = output.try_send(Err(error));
                    return;
                }
            };
            let result = input.submit(&state, registration, cancel, Some(context));
            match result {
                Ok(mut native) => {
                    let mut commands_open = true;
                    let mut pending = None;
                    loop {
                        tokio::select! {
                            () = output.closed() => break,
                            () = tokio::time::sleep_until(deadline) => {
                                drop(native);
                                let _ = output.try_send(Err(Status::deadline_exceeded("decode deadline expired")));
                                break;
                            },
                            // Keep one slot for cancellation/deadline even when tokens are backpressured.
                            permit = output.reserve_many(2), if pending.is_some() => match permit {
                                Ok(mut permits) => { if let Some(permit) = permits.next() { if let Some(item) = pending.take() { permit.send(item); } } },
                                Err(_) => break,
                            },
                            item = native.next(), if pending.is_none() => match item {
                                Some(item) => pending = Some(item),
                                None => break,
                            },
                            command = commands.message(), if commands_open => {
                                let error = match command {
                                    Ok(None) => { commands_open = false; continue; },
                                    Ok(Some(proto::DecodeCommand { command: Some(Command::BindPreparedInput(input)) })) => {
                                        match validate_manifest(&input, &binding, Limits::default()) {
                                            Ok(()) if Some(&input.integrity_digest) == digest.as_ref() => continue,
                                            Ok(()) => Status::already_exists("conflicting prepared input binding"),
                                            Err(error) => error,
                                        }
                                    },
                                    Ok(Some(proto::DecodeCommand { command: Some(Command::Cancel(reason)) })) => Status::cancelled(reason),
                                    Ok(_) => Status::invalid_argument("decode Start may occur only once"),
                                    Err(error) => error,
                                };
                                // Dropping the original stream closes only its observer;
                                // the existing native owner performs and acknowledges cleanup.
                                drop(native);
                                let _ = tokio::time::timeout_at(deadline, output.send(Err(error))).await;
                                break;
                            }
                        }
                    }
                }
                Err(error) => {
                    let _ = output.send(Err(error)).await;
                }
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(receiver))))
    }
}
