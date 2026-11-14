//! Managed P/D control service. A native adapter owns ordered submit/abort and safe release.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use futures::StreamExt;
use prost::Message;
use smg_grpc_client::{
    sglang_managed::{
        budget::Deadline,
        clock,
        proto::{
            self, control_event::Event, decode_command::Command,
            managed_sglang_server::ManagedSglang,
        },
        validate_binding, validate_external_plan, validate_operation, Limits, VERSION,
    },
    sglang_proto,
};
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use super::{
    decode::{DecodeGate, NativeInput},
    ledger::Ledger,
};
use crate::{BoxFuture, BoxStream};

/// No default implementation: a legacy engine cannot silently consume managed metadata.
pub trait NativeRuntime: Send + Sync + 'static {
    /// A process fence is usable only after its native engine connection is established.
    fn ready(&self) -> bool {
        false
    }
    /// Register P export ownership before submitting its external native encode plan.
    fn prefill(
        &self,
        request: proto::PrefillStart,
        deadline: Deadline,
    ) -> BoxFuture<Result<BoxStream<proto::ControlEvent>, Status>>;
    /// Submit complete native input once, without media processing or any fallback.
    fn decode(
        &self,
        operation: proto::OperationRef,
        binding: proto::InputBinding,
        deadline: Deadline,
        input: NativeInput,
    ) -> BoxFuture<Result<BoxStream<sglang_proto::GenerateResponse>, Status>>;
    /// Abort through the actual rank's ordered channel; true means references safely exited.
    fn terminate(&self, operation: proto::OperationRef) -> BoxFuture<Result<bool, Status>>;
}

struct Registry {
    ledger: Ledger,
    cancellation: HashMap<String, watch::Sender<bool>>,
}

pub struct ManagedService<R> {
    runtime: Arc<R>,
    instance: proto::Instance,
    registry: Arc<Mutex<Registry>>,
    limits: Limits,
    max_duration_ms: u64,
    capabilities: Option<Arc<proto::Capabilities>>,
    max_clock_error_ms: Option<u64>,
}

impl<R> Clone for ManagedService<R> {
    /// RPC handlers share one ledger and cancellation owner, never independent admission state.
    fn clone(&self) -> Self {
        Self {
            runtime: Arc::clone(&self.runtime),
            instance: self.instance.clone(),
            registry: Arc::clone(&self.registry),
            limits: self.limits,
            max_duration_ms: self.max_duration_ms,
            capabilities: self.capabilities.clone(),
            max_clock_error_ms: self.max_clock_error_ms,
        }
    }
}

impl<R: NativeRuntime> ManagedService<R> {
    /// Finite local admission and tombstones must be configured before binding the service.
    pub fn new(
        runtime: Arc<R>,
        instance: proto::Instance,
        ledger: Ledger,
        limits: Limits,
        max_duration_ms: u64,
    ) -> Self {
        Self {
            runtime,
            instance,
            registry: Arc::new(Mutex::new(Registry {
                ledger,
                cancellation: HashMap::new(),
            })),
            limits,
            max_duration_ms,
            capabilities: None,
            max_clock_error_ms: None,
        }
    }

    /// Install launch-validated capability facts without enabling the legacy request surface.
    pub fn with_capabilities(
        mut self,
        capabilities: proto::Capabilities,
        max_clock_error_ms: u64,
    ) -> Self {
        self.capabilities = Some(Arc::new(capabilities));
        self.max_clock_error_ms = Some(max_clock_error_ms);
        self
    }

    /// Drain affects new Starts only. Existing input binding and termination stay enabled.
    pub fn drain(&self) -> Result<(), Status> {
        self.registry
            .lock()
            .map_err(|_| Status::internal("managed registry poisoned"))?
            .ledger
            .drain();
        Ok(())
    }

    /// Reject a foreign role or deployment contract before allocating native or registry state.
    fn validate_contract(&self, role: &str, binding: &proto::InputBinding) -> Result<(), Status> {
        if let Some(capability) = &self.capabilities {
            if capability.role != role
                || capability.model_revision != binding.model_revision
                || capability.processor_contract != binding.processor_contract
                || capability.layout_contract != binding.layout_contract
            {
                return Err(Status::failed_precondition(
                    "managed worker contract mismatch",
                ));
            }
        }
        Ok(())
    }

    /// Atomically register deadline/cancel ownership before returning Accepted.
    fn register(
        &self,
        operation: proto::OperationRef,
        budget: &proto::Budget,
    ) -> Result<(u64, Deadline, watch::Receiver<bool>), Status> {
        if let Some(tolerance) = self.max_clock_error_ms {
            clock::check(tolerance)?;
        }
        let now_ms = wall_time_ms()?;
        let deadline = Deadline::admit(budget, now_ms, Instant::now(), 0, 0, self.max_duration_ms)?;
        let mut registry = self
            .registry
            .lock()
            .map_err(|_| Status::internal("managed registry poisoned"))?;
        let (seq, fresh) = registry.ledger.admit(operation.clone(), budget, now_ms)?;
        if !fresh {
            return Err(Status::already_exists(
                "active managed stream cannot be taken over",
            ));
        }
        let (sender, receiver) = watch::channel(false);
        registry
            .cancellation
            .insert(operation.operation_id.clone(), sender);
        Ok((seq, deadline, receiver))
    }
}

/// Keep this bounded admission owner's completion observer until native cleanup resolves.
async fn cleanup<R: NativeRuntime>(
    runtime: &R,
    registry: &Mutex<Registry>,
    operation: proto::OperationRef,
    native: bool,
) {
    let safe = if native {
        matches!(runtime.terminate(operation.clone()).await, Ok(true))
    } else {
        true
    };
    if let Ok(mut registry) = registry.lock() {
        registry.cancellation.remove(&operation.operation_id);
        if safe {
            let _ = registry.ledger.quiesce(&operation);
        } else {
            tracing::warn!(operation = %operation.operation_id, "managed native cleanup remains unknown");
        }
    }
}

/// Wall-clock admission becomes a monotonic deadline immediately after registration.
fn wall_time_ms() -> Result<u64, Status> {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Status::unavailable("invalid worker clock"))?
            .as_millis(),
    )
    .map_err(|_| Status::unavailable("worker clock overflow"))
}

/// All sends obey the original budget, even when the client stops consuming events.
async fn emit(
    sender: &mpsc::Sender<Result<proto::ControlEvent, Status>>,
    operation: &proto::OperationRef,
    seq: &mut u64,
    admission: u64,
    event: Event,
    deadline: Deadline,
) -> Result<(), Status> {
    *seq = seq
        .checked_add(1)
        .ok_or_else(|| Status::internal("event sequence exhausted"))?;
    let value = proto::ControlEvent {
        operation: Some(operation.clone()),
        event_seq: *seq,
        admission_seq: admission,
        part_idx: None,
        event: Some(event),
    };
    tokio::time::timeout(deadline.remaining(Instant::now())?, sender.send(Ok(value)))
        .await
        .map_err(|_| Status::deadline_exceeded("managed event delivery expired"))?
        .map_err(|_| Status::cancelled("managed response disconnected"))
}

#[tonic::async_trait]
impl<R: NativeRuntime> ManagedSglang for ManagedService<R> {
    type GenerateDecodeStream = BoxStream<proto::ControlEvent>;
    type GeneratePrefillStream = BoxStream<proto::ControlEvent>;

    /// Combine immutable launch contracts with current clock, connection and drain health.
    async fn get_capabilities(
        &self,
        _: Request<proto::CapabilityQuery>,
    ) -> Result<Response<proto::Capabilities>, Status> {
        let mut capabilities = self
            .capabilities
            .as_deref()
            .ok_or_else(|| Status::failed_precondition("managed contract not configured"))?
            .clone();
        capabilities.wall_time_unix_ms = wall_time_ms()?;
        capabilities.ready = self.runtime.ready()
            && !self
                .registry
                .lock()
                .map_err(|_| Status::internal("managed registry poisoned"))?
                .ledger
                .is_draining()
            && self
                .max_clock_error_ms
                .is_some_and(|limit| clock::check(limit).is_ok());
        Ok(Response::new(capabilities))
    }

    /// Register Start immediately, then bind input on this same stream before one native submit.
    async fn generate_decode(
        &self,
        request: Request<tonic::Streaming<proto::DecodeCommand>>,
    ) -> Result<Response<Self::GenerateDecodeStream>, Status> {
        let mut commands = request.into_inner();
        // Waiting for the first frame is bounded; waiting for Bind is inside the returned stream.
        let first = tokio::time::timeout(Duration::from_secs(5), commands.message())
            .await
            .map_err(|_| Status::deadline_exceeded("managed Start missing"))??;
        let Some(proto::DecodeCommand {
            command: Some(Command::Start(start)),
        }) = first
        else {
            return Err(Status::invalid_argument(
                "first managed decode frame must be Start",
            ));
        };
        self.validate_contract(
            "decode",
            start
                .binding
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("missing D binding"))?,
        )?;
        let operation = start
            .binding
            .as_ref()
            .and_then(|b| b.decode.clone())
            .ok_or_else(|| Status::invalid_argument("missing decode operation"))?;
        let budget = start
            .budget
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing managed budget"))?;
        let reserved_input = if start.deferred_from_prefill {
            self.limits.manifest_bytes
        } else {
            0
        };
        if start.encoded_len().saturating_add(reserved_input) > budget.context_bytes as usize {
            return Err(Status::resource_exhausted(
                "D input exceeds reserved context budget",
            ));
        }
        let deadline = Deadline::admit(
            budget,
            wall_time_ms()?,
            Instant::now(),
            0,
            0,
            self.max_duration_ms,
        )?;
        let mut gate = DecodeGate::register(
            start.clone(),
            &self.instance,
            deadline,
            self.limits,
            Instant::now(),
        )?;
        let (admission, deadline, mut cancel) = self.register(operation.clone(), budget)?;
        let (sender, receiver) = mpsc::channel(8);
        let runtime = Arc::clone(&self.runtime);
        let registry = Arc::clone(&self.registry);
        #[expect(
            clippy::disallowed_methods,
            reason = "bounded registry owns native cleanup after RPC disconnect"
        )]
        tokio::spawn(async move {
            let mut native = false;
            let result: Result<(), Status> = async {
                let mut seq = 0;
                emit(&sender, &operation, &mut seq, admission, Event::Stage(proto::Stage::Accepted as i32), deadline).await?;
                let mut closed = false;
                loop {
                    if *cancel.borrow() || sender.is_closed() { gate.cancel(); return Err(Status::cancelled("managed decode cancelled")); }
                    if let Some(input) = gate.take_submission(Instant::now())? {
                        emit(&sender, &operation, &mut seq, admission, Event::Stage(proto::Stage::InputBound as i32), deadline).await?;
                        native = true;
                        // Do not drop an in-flight send on cancellation. The adapter serializes
                        // Generate and cleanup's Abort on the same native rank channel.
                        let binding = start.binding.clone().ok_or_else(|| Status::internal("managed binding disappeared"))?;
                        let mut generation = runtime.decode(operation.clone(), binding, deadline, input).await?;
                        gate.submitted()?;
                        emit(&sender, &operation, &mut seq, admission, Event::Stage(proto::Stage::NativeSubmitted as i32), deadline).await?;
                        let mut complete = false;
                        loop {
                            tokio::select! {
                                biased;
                                _ = cancel.changed() => { gate.cancel(); return Err(Status::cancelled("managed decode terminated")); }
                                () = sender.closed() => { gate.cancel(); return Err(Status::cancelled("managed response disconnected")); }
                                () = tokio::time::sleep(deadline.remaining(Instant::now())?) => { gate.cancel(); return Err(Status::deadline_exceeded("managed decode expired")); }
                                command = commands.message(), if !closed => {
                                    match command? {
                                        Some(proto::DecodeCommand { command: Some(Command::BindPreparedInput(input)) }) => { gate.bind(input, Instant::now())?; }
                                        Some(proto::DecodeCommand { command: Some(Command::Cancel(_)) }) => { gate.cancel(); return Err(Status::cancelled("managed stream cancel")); }
                                        None => { gate.half_close()?; closed = true; }
                                        _ => return Err(Status::invalid_argument("unexpected managed decode command")),
                                    }
                                }
                                response = generation.next() => {
                                    let Some(response) = response else {
                                        return if complete { Ok(()) } else { Err(Status::data_loss("native D ended without completion")) };
                                    };
                                    let response = response?;
                                    complete |= matches!(response.response, Some(sglang_proto::generate_response::Response::Complete(_)));
                                    emit(&sender, &operation, &mut seq, admission, Event::Generation(response), deadline).await?;
                                }
                            }
                        }
                    }
                    tokio::select! {
                        biased;
                        _ = cancel.changed() => { gate.cancel(); return Err(Status::cancelled("managed WaitingInput terminated")); }
                        () = sender.closed() => { gate.cancel(); return Err(Status::cancelled("managed response disconnected")); }
                        () = tokio::time::sleep(deadline.remaining(Instant::now())?) => { gate.cancel(); return Err(Status::deadline_exceeded("managed WaitingInput expired")); }
                        command = commands.message() => {
                            match command? {
                                Some(proto::DecodeCommand { command: Some(Command::BindPreparedInput(input)) }) => { gate.bind(input, Instant::now())?; }
                                Some(proto::DecodeCommand { command: Some(Command::Cancel(_)) }) => { gate.cancel(); return Err(Status::cancelled("managed stream cancel")); }
                                None => { gate.half_close()?; }
                                _ => return Err(Status::invalid_argument("unexpected managed decode command")),
                            }
                        }
                    }
                }
            }.await;
            if let Err(error) = result {
                let _ = sender.try_send(Err(error));
            }
            cleanup(runtime.as_ref(), &registry, operation, native).await;
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(receiver))))
    }

    /// Own the P operation through native completion and relay its authoritative layout.
    async fn generate_prefill(
        &self,
        request: Request<proto::PrefillStart>,
    ) -> Result<Response<Self::GeneratePrefillStream>, Status> {
        let request = request.into_inner();
        if request.version != VERSION {
            return Err(Status::failed_precondition("unsupported managed version"));
        }
        let operation = request
            .operation
            .clone()
            .ok_or_else(|| Status::invalid_argument("missing P operation"))?;
        validate_operation(&operation)?;
        let binding = request
            .binding
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing P binding"))?;
        validate_binding(binding)?;
        self.validate_contract("prefill", binding)?;
        let decode = binding
            .decode
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing D binding"))?;
        if operation.target.as_ref() != Some(&self.instance)
            || binding.prefill.as_ref() != Some(&self.instance)
            || operation.identity != decode.identity
            || operation.request_digest != decode.request_digest
            || operation.plan_digest != decode.plan_digest
        {
            return Err(Status::failed_precondition(
                "managed P operation does not match binding",
            ));
        }
        let generation = request
            .generation
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing P generation"))?;
        let plan = request
            .encode_plan
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing P encode plan"))?;
        let images = plan
            .image_counts
            .iter()
            .try_fold(0usize, |sum, n| sum.checked_add(*n as usize))
            .ok_or_else(|| Status::invalid_argument("image count overflow"))?;
        validate_external_plan(plan, images)?;
        if generation.request_id != binding.prefill_rid
            || plan.receiver_rid != binding.prefill_rid
            || images > self.limits.images
            || generation.mm_inputs.is_some()
            || generation.media_identity.is_some()
            || !generation.extra_mm_inputs.is_empty()
            || !generation.input_embeds.is_empty()
            || generation
                .sampling_params
                .as_ref()
                .is_some_and(|params| params.n > 1)
            || generation.tokenized.as_ref().is_none_or(|input| {
                input.input_ids.is_empty() || input.input_ids.len() > self.limits.tokens
            })
        {
            return Err(Status::invalid_argument(
                "invalid managed P input/assignment",
            ));
        }
        let budget = request
            .budget
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing managed budget"))?;
        if request.encoded_len() > self.limits.manifest_bytes
            || budget.context_bytes < request.encoded_len() as u64
        {
            return Err(Status::resource_exhausted(
                "P context exceeds reserved byte budget",
            ));
        }
        let (admission, deadline, mut cancel) = self.register(operation.clone(), budget)?;
        let (sender, receiver) = mpsc::channel(8);
        let runtime = Arc::clone(&self.runtime);
        let registry = Arc::clone(&self.registry);
        #[expect(
            clippy::disallowed_methods,
            reason = "bounded P registry owns cleanup after RPC disconnect"
        )]
        tokio::spawn(async move {
            let result: Result<(), Status> = async {
                let mut seq = 0;
                emit(&sender, &operation, &mut seq, admission, Event::Stage(proto::Stage::Accepted as i32), deadline).await?;
                if *cancel.borrow() || sender.is_closed() { return Err(Status::cancelled("managed P cancelled before submission")); }
                let mut stream = runtime.prefill(request, deadline).await?;
                let mut complete = false;
                loop {
                    tokio::select! {
                        biased;
                        _ = cancel.changed() => return Err(Status::cancelled("managed P terminated")),
                        () = sender.closed() => return Err(Status::cancelled("managed P disconnected")),
                        () = tokio::time::sleep(deadline.remaining(Instant::now())?) => return Err(Status::deadline_exceeded("managed P expired")),
                        event = stream.next() => {
                            let Some(event) = event else {
                                return if complete { Ok(()) } else { Err(Status::data_loss("native P ended without completion")) };
                            };
                            let event = event?.event.ok_or_else(|| Status::data_loss("empty native P event"))?;
                            complete |= matches!(&event, Event::Generation(response) if matches!(response.response, Some(sglang_proto::generate_response::Response::Complete(_))));
                            emit(&sender, &operation, &mut seq, admission, event, deadline).await?;
                        }
                    }
                }
            }.await;
            if let Err(error) = result {
                let _ = sender.try_send(Err(error));
            }
            cleanup(runtime.as_ref(), &registry, operation, true).await;
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(receiver))))
    }

    /// Preserve pre-Start cancellation and distinguish recorded intent from safe quiescence.
    async fn terminate(
        &self,
        request: Request<proto::Termination>,
    ) -> Result<Response<proto::TerminationReceipt>, Status> {
        let request = request.into_inner();
        let operation = request
            .operation
            .ok_or_else(|| Status::invalid_argument("missing managed termination operation"))?;
        let mut registry = self
            .registry
            .lock()
            .map_err(|_| Status::internal("managed registry poisoned"))?;
        let now = wall_time_ms()?;
        registry
            .ledger
            .cancel(operation.clone(), request.original_expiry_unix_ms, now)?;
        if let Some(sender) = registry.cancellation.get(&operation.operation_id) {
            let _ = sender.send(true);
        }
        let quiesced = match registry.ledger.get(&operation) {
            Ok(record) => record.quiesced,
            Err(error) => {
                error.code() == tonic::Code::NotFound && request.original_expiry_unix_ms <= now
            }
        };
        Ok(Response::new(proto::TerminationReceipt {
            recorded: true,
            quiesced,
        }))
    }
}
