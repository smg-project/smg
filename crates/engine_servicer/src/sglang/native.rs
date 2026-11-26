//! Prepared-input admission and native ownership until the scheduler acknowledges completion.

use std::{
    collections::HashMap,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use engine_zmq_adapter::{ManagedExternalEncode, ZmqEngineClient};
use futures::{Stream, StreamExt};
use prost::Message;
use smg_grpc_client::{
    sglang_managed::{
        normalized_capabilities, proto, validate_binding, validate_capabilities, validate_manifest,
        Limits, VERSION,
    },
    sglang_proto as sg,
};
use tokio::sync::{mpsc, oneshot, watch};
use tonic::Status;
use zeromq::{PullSocket, Socket, SocketRecv};

use super::State;
use crate::{
    requests::{CancelReason, Registration},
    BoxStream,
};

const DEFAULT_ABORT_GRACE: Duration = Duration::from_secs(30);

/// Explicit managed-only launch configuration; legacy launch has no such value.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ManagedConfig {
    #[serde(flatten)]
    pub capabilities: proto::Capabilities,
    pub metadata_endpoint: Option<String>,
    #[serde(default)]
    pub disable_routing_labels: bool,
    pub max_records: usize,
    pub max_duration_ms: u64,
}

impl ManagedConfig {
    /// Validate before binding any listener or launching a scheduler.
    pub fn validate(&self) -> Result<(), Status> {
        if self.max_duration_ms == 0 || self.max_records == 0 {
            return Err(Status::invalid_argument(
                "incomplete managed capability contract",
            ));
        }
        validate_capabilities(&self.capabilities)?;
        let role = self.capabilities.role.as_str();
        if !matches!(role, "prefill" | "decode")
            || (role == "prefill"
                && !self
                    .metadata_endpoint
                    .as_ref()
                    .is_some_and(|s| s.starts_with("ipc://")))
            || (role == "decode" && self.metadata_endpoint.is_some())
        {
            return Err(Status::invalid_argument(
                "invalid managed role/metadata endpoint",
            ));
        }
        Ok(())
    }

    /// Advertise the managed contract; readiness is checked per query.
    pub fn capabilities(&self) -> proto::Capabilities {
        normalized_capabilities(self.capabilities.clone(), false)
    }
}

/// Register before native submission. P exports the layout and D binds that exact layout;
/// neither side falls back to media preprocessing.
pub(super) fn generate(
    state: &Arc<State>,
    mut request: sg::GenerateRequest,
) -> Result<BoxStream<sg::GenerateResponse>, Status> {
    let (registration, cancel) = state.shared.registry.register(&request.request_id)?;
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
    let mut control = proto::PrefillStart::decode(request.prepared_control.as_slice())
        .map_err(|_| Status::invalid_argument("malformed prepared control"))?;
    let binding = control
        .binding
        .take()
        .ok_or_else(|| Status::invalid_argument("missing prepared binding"))?;
    validate_binding(&binding)?;
    if control.version != VERSION
        || binding.model_revision != config.capabilities.model_revision
        || binding.processor_contract != config.capabilities.processor_contract
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
    let (external, manifest) = if config.capabilities.role == "prefill" {
        if request.request_id != binding.prefill_rid || !request.prepared_input.is_empty() {
            return Err(Status::invalid_argument("invalid prepared prefill request"));
        }
        let external = parse_external_encode(&control, limits)?;
        (Some(external), None)
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
    runtime.submit(
        PreparedRequest {
            request,
            binding,
            external,
            manifest,
            deadline,
        },
        registration,
        cancel,
    )
}

/// Validate `PrefillStart` encoder endpoints and per-encoder modality counts.
fn parse_external_encode(
    control: &proto::PrefillStart,
    limits: Limits,
) -> Result<ManagedExternalEncode, Status> {
    let total_items = control.image_count.saturating_add(control.video_count);
    if total_items as usize > limits.images {
        return Err(Status::invalid_argument("invalid prepared encode plan"));
    }
    if control.encoder_receive_endpoints.is_empty() {
        if !control.encoder_image_counts.is_empty()
            || !control.encoder_video_counts.is_empty()
            || (total_items > 0 && control.encoder_receive_endpoint.is_empty())
        {
            return Err(Status::invalid_argument("invalid prepared encode plan"));
        }
        return Ok(ManagedExternalEncode::single(
            control.encoder_receive_endpoint.clone(),
            control.image_count,
            control.video_count,
        ));
    }
    let num_encoders = control.encoder_receive_endpoints.len();
    if total_items == 0
        || num_encoders > total_items as usize
        || control
            .encoder_receive_endpoints
            .iter()
            .any(String::is_empty)
    {
        return Err(Status::invalid_argument("invalid prepared encode plan"));
    }
    let valid_counts = |expected: u32, counts: &[u32]| -> bool {
        if expected == 0 {
            counts.is_empty()
        } else {
            counts.len() == num_encoders
                && counts.iter().try_fold(0u32, |acc, &n| acc.checked_add(n)) == Some(expected)
        }
    };
    if !valid_counts(control.image_count, &control.encoder_image_counts)
        || !valid_counts(control.video_count, &control.encoder_video_counts)
    {
        return Err(Status::invalid_argument("invalid prepared encode plan"));
    }
    for idx in 0..num_encoders {
        let img = control.encoder_image_counts.get(idx).copied().unwrap_or(0);
        let vid = control.encoder_video_counts.get(idx).copied().unwrap_or(0);
        if img == 0 && vid == 0 {
            return Err(Status::invalid_argument("invalid prepared encode plan"));
        }
    }
    Ok(ManagedExternalEncode {
        encoder_urls: control.encoder_receive_endpoints.clone(),
        image_counts: control.encoder_image_counts.clone(),
        video_counts: control.encoder_video_counts.clone(),
    })
}

/// Validated request passed from prepared admission to its independent native owner.
pub(crate) struct PreparedRequest {
    pub(crate) request: sg::GenerateRequest,
    pub(crate) binding: proto::InputBinding,
    pub(crate) external: Option<ManagedExternalEncode>,
    pub(crate) manifest: Option<Box<proto::PreparedDecodeInput>>,
    pub(crate) deadline: Instant,
}

struct Entry {
    cancel: watch::Sender<bool>,
    layout: mpsc::Sender<Vec<u8>>,
    binding: proto::InputBinding,
    #[cfg(test)]
    output_observer: mpsc::WeakSender<Result<sg::GenerateResponse, Status>>,
    _registration: Registration,
}

/// Each native task owns its stream independently of the lifetime of a control RPC.
pub struct ManagedNativeRuntime {
    client: Arc<dyn Fn() -> Option<ZmqEngineClient> + Send + Sync>,
    entries: Mutex<HashMap<String, Entry>>,
    max_entries: usize,
    limits: Limits,
    disable_routing_labels: bool,
    abort_grace_ms: AtomicU64,
    metadata_failed: AtomicBool,
    quarantined: AtomicUsize,
}

impl ManagedNativeRuntime {
    /// Bind before native scheduler handshake, then keep the reader owned by that server.
    pub async fn bind_metadata(endpoint: &str) -> Result<PullSocket, Status> {
        if !endpoint.starts_with("ipc://") {
            return Err(Status::invalid_argument("metadata requires same-host IPC"));
        }
        let mut socket = PullSocket::new();
        socket
            .bind(endpoint)
            .await
            .map_err(|e| Status::unavailable(e.to_string()))?;
        Ok(socket)
    }

    /// One reader per P instance; malformed/oversized events cannot be attributed safely and are
    /// dropped (bounded by each owner's prepared deadline), while binding-matched invalid layouts
    /// cancel their owner.
    pub async fn read_metadata(&self, mut socket: PullSocket) -> Result<(), Status> {
        let result = loop {
            let frames = match socket.recv().await {
                Ok(msg) => msg.into_vec(),
                Err(error) => break Err(Status::unavailable(error.to_string())),
            };
            if frames.len() != 2 {
                tracing::warn!(
                    frame_count = frames.len(),
                    "native managed metadata frame count mismatch; dropping"
                );
                continue;
            }
            let Ok(rid) = std::str::from_utf8(&frames[0]) else {
                tracing::warn!("invalid native managed metadata rid; dropping");
                continue;
            };
            if let Err(error) = self.publish_layout(rid, &frames[1]) {
                tracing::warn!(rid, %error, "native managed metadata refused");
            }
        };
        self.metadata_failed.store(true, Ordering::Release);
        if let Ok(entries) = self.entries.lock() {
            for entry in entries.values() {
                let _ = entry.cancel.send(true);
            }
        }
        result
    }

    /// Construct a bounded managed runtime backed by the given ZMQ client supplier.
    pub fn new(
        client: Arc<dyn Fn() -> Option<ZmqEngineClient> + Send + Sync>,
        config: &ManagedConfig,
        limits: Limits,
    ) -> Result<Self, Status> {
        if config.max_records == 0 {
            return Err(Status::invalid_argument("native registry must be bounded"));
        }
        Ok(Self {
            client,
            entries: Mutex::default(),
            max_entries: config.max_records,
            limits,
            disable_routing_labels: config.disable_routing_labels,
            abort_grace_ms: AtomicU64::new(DEFAULT_ABORT_GRACE.as_millis() as u64),
            metadata_failed: AtomicBool::new(false),
            quarantined: AtomicUsize::new(0),
        })
    }

    #[cfg(test)]
    pub(crate) fn set_abort_grace(&self, grace: Duration) {
        self.abort_grace_ms
            .store(grace.as_millis() as u64, Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn output_queued(&self, rid: &str) -> usize {
        self.entries
            .lock()
            .ok()
            .and_then(|entries| entries.get(rid).and_then(|e| e.output_observer.upgrade()))
            .map_or(0, |tx| tx.max_capacity() - tx.capacity())
    }

    /// Validate a prepared decode manifest against the request binding and capability policy.
    pub(crate) fn validate_layout(
        &self,
        manifest: &proto::PreparedDecodeInput,
        binding: &proto::InputBinding,
    ) -> Result<(), Status> {
        if self.disable_routing_labels
            && manifest
                .images
                .iter()
                .any(|item| item.routing_label.is_some())
        {
            return Err(Status::failed_precondition(
                "native engine does not support media routing labels",
            ));
        }
        validate_manifest(manifest, binding, self.limits)
    }

    /// Registry lookup and binding verification precede cancellation: oversized, malformed, or
    /// mismatched-binding frames never cancel an unrelated attempt.
    pub fn publish_layout(&self, rid: &str, bytes: &[u8]) -> Result<(), Status> {
        if bytes.len() > self.limits.manifest_bytes {
            return Err(Status::resource_exhausted(
                "native layout exceeds byte limit",
            ));
        }
        let entries = self
            .entries
            .lock()
            .map_err(|_| Status::internal("native registry poisoned"))?;
        let entry = entries
            .get(rid)
            .ok_or_else(|| Status::not_found("unregistered native layout"))?;
        if *entry.cancel.borrow() {
            return Err(Status::cancelled(
                "native layout arrived after cancellation",
            ));
        }
        let manifest = proto::PreparedDecodeInput::decode(bytes)
            .map_err(|_| Status::data_loss("malformed native layout"))?;
        if manifest.binding.as_ref() != Some(&entry.binding) {
            return Err(Status::invalid_argument(
                "managed manifest binding mismatch",
            ));
        }
        if let Err(error) = self.validate_layout(&manifest, &entry.binding) {
            let _ = entry.cancel.send(true);
            return Err(error);
        }
        entry.layout.try_send(bytes.to_vec()).map_err(|_| {
            let _ = entry.cancel.send(true);
            Status::resource_exhausted("native layout channel full or closed")
        })
    }

    /// Reserve output and cancellation ownership before native Generate is sent.
    pub(crate) fn submit(
        self: &Arc<Self>,
        input: PreparedRequest,
        registration: Registration,
        mut abort: oneshot::Receiver<CancelReason>,
    ) -> Result<BoxStream<sg::GenerateResponse>, Status> {
        if self.metadata_failed.load(Ordering::Acquire)
            || self.quarantined.load(Ordering::Acquire) > 0
        {
            return Err(Status::unavailable("native runtime quarantined"));
        }
        let client =
            (self.client)().ok_or_else(|| Status::unavailable("native engine not ready"))?;
        let rid = input.request.request_id.clone();
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| Status::internal("native registry poisoned"))?;
        if entries.contains_key(&rid) {
            return Err(Status::already_exists(
                "native operation already registered",
            ));
        }
        if entries.len() >= self.max_entries {
            return Err(Status::resource_exhausted("native registry full"));
        }
        let (cancel, cancelled) = watch::channel(false);
        let (layout, layouts) = mpsc::channel(1);
        let (output, receiver) = mpsc::channel(8);
        let terminal_error = Arc::new(Mutex::new(None));
        entries.insert(
            rid.clone(),
            Entry {
                cancel: cancel.clone(),
                layout,
                binding: input.binding.clone(),
                #[cfg(test)]
                output_observer: output.downgrade(),
                _registration: registration,
            },
        );
        drop(entries);
        let runtime = Arc::clone(self);
        let task_terminal_error = Arc::clone(&terminal_error);
        let abort_grace = Duration::from_millis(runtime.abort_grace_ms.load(Ordering::Acquire));
        #[expect(
            clippy::disallowed_methods,
            reason = "native task retains submission through acknowledged termination"
        )]
        tokio::spawn(async move {
            let work = run_native(
                client,
                input,
                cancelled,
                layouts,
                output,
                task_terminal_error,
                &runtime.quarantined,
                abort_grace,
            );
            tokio::pin!(work);
            let (safe, was_quarantined) = tokio::select! {
                biased;
                _ = &mut abort => { let _ = cancel.send(true); work.await }
                result = &mut work => result,
            };
            if safe {
                let removed = runtime
                    .entries
                    .lock()
                    .ok()
                    .and_then(|mut entries| entries.remove(&rid));
                if was_quarantined {
                    runtime.quarantined.fetch_sub(1, Ordering::AcqRel);
                }
                drop(removed);
            } else {
                runtime.metadata_failed.store(true, Ordering::Release);
            }
        });
        Ok(Box::pin(PreparedOutputStream {
            receiver,
            terminal_error,
        }))
    }

    /// A connected engine cannot serve prepared inputs while quarantined or after permanent failure.
    pub fn ready(&self) -> bool {
        !self.metadata_failed.load(Ordering::Acquire)
            && self.quarantined.load(Ordering::Acquire) == 0
            && (self.client)().is_some_and(|client| client.is_alive())
    }
}

/// Typed admission refusals (and pre-queue `ClientClosed` before `submit_owned` returns a stream)
/// precede enqueue; unknown transport writes retain ownership.
pub(crate) fn definitely_unsubmitted(error: &Status) -> bool {
    matches!(
        error.code(),
        tonic::Code::InvalidArgument | tonic::Code::FailedPrecondition | tonic::Code::Unimplemented
    ) || std::error::Error::source(error)
        .and_then(|source| source.downcast_ref::<engine_zmq_client::Error>())
        .is_some_and(|error| {
            matches!(
                error,
                engine_zmq_client::Error::EngineInputFull { .. }
                    | engine_zmq_client::Error::EngineInputBytesFull { .. }
                    | engine_zmq_client::Error::ClientClosed { .. }
            )
        })
}

/// Bounded chunk stream with an out-of-band terminal error slot so cancellation or
/// failure status is never dropped when the 8-slot chunk queue is full.
struct PreparedOutputStream {
    receiver: mpsc::Receiver<Result<sg::GenerateResponse, Status>>,
    terminal_error: Arc<Mutex<Option<Status>>>,
}

impl Stream for PreparedOutputStream {
    type Item = Result<sg::GenerateResponse, Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Ready(None) => Poll::Ready(
                self.terminal_error
                    .lock()
                    .ok()
                    .and_then(|mut slot| slot.take())
                    .map(Err),
            ),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Deliver a terminal error over `output`, falling back to `terminal_error` when full.
pub(crate) fn push_terminal_error(
    output: &mpsc::Sender<Result<sg::GenerateResponse, Status>>,
    terminal_error: &Mutex<Option<Status>>,
    status: Status,
) {
    let Ok(mut slot) = terminal_error.lock() else {
        let _ = output.try_send(Err(status));
        return;
    };
    if slot.is_some() {
        return;
    }
    if let Err(mpsc::error::TrySendError::Full(Err(status))) = output.try_send(Err(status)) {
        *slot = Some(status);
    }
}

/// Preserve Generate->Abort ordering while retaining the native output registration.
///
/// The prepared `deadline` bounds P (encode/layout/prefill) and D admission; once D is
/// submitted to the scheduler, total generation duration follows `--grpc-request-timeout-secs`.
#[expect(clippy::too_many_arguments)]
async fn run_native(
    client: ZmqEngineClient,
    input: PreparedRequest,
    mut cancel: watch::Receiver<bool>,
    mut layouts: mpsc::Receiver<Vec<u8>>,
    output: mpsc::Sender<Result<sg::GenerateResponse, Status>>,
    terminal_error: Arc<Mutex<Option<Status>>>,
    quarantined: &AtomicUsize,
    abort_grace: Duration,
) -> (bool, bool) {
    let PreparedRequest {
        request,
        binding,
        external,
        manifest,
        deadline,
    } = input;
    if *cancel.borrow() || output.is_closed() {
        return (true, false);
    }
    if Instant::now() >= deadline {
        push_terminal_error(
            &output,
            &terminal_error,
            Status::deadline_exceeded("prepared request deadline expired before submission"),
        );
        return (true, false);
    }
    let mut native = match client
        .generate_sglang_managed(request, &binding, external.as_ref(), manifest.as_deref())
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            let unsubmitted = definitely_unsubmitted(&error);
            push_terminal_error(&output, &terminal_error, error);
            return (unsubmitted, false);
        }
    };
    let rid = native.request_id().to_owned();
    let rank = native.engine_index().unwrap_or(0);
    let mut abort_until: Option<Instant> = None;
    let mut layout_sent = false;
    let mut stalled = output.is_closed();
    let needs_layout = external
        .as_ref()
        .is_some_and(ManagedExternalEncode::has_media);
    let mut native_finished = false;
    let mut pending = Vec::new();
    let active_deadline = external.is_some().then_some(deadline);
    let begin_abort = |abort_until: &mut Option<Instant>| {
        if abort_until.is_none() {
            *abort_until = Some(Instant::now() + abort_grace);
            // SGLang's decode transfer/prealloc queues handle AbortReq with drain-ack accounting,
            // so withholding AbortReq on D deadlocks D behind an already-aborted P.
            if let Err(error) = client.abort_sglang_managed(rank, &rid) {
                tracing::warn!(rid, %error, "managed Abort write unconfirmed; retaining terminal observer");
            }
        }
    };
    loop {
        if abort_until.is_none()
            && (stalled
                || *cancel.borrow()
                || output.is_closed()
                || active_deadline.is_some_and(|dl| Instant::now() >= dl))
        {
            if native_finished {
                if needs_layout && !layout_sent {
                    push_terminal_error(
                        &output,
                        &terminal_error,
                        Status::deadline_exceeded("prepared layout export timed out"),
                    );
                }
                return (true, false);
            }
            begin_abort(&mut abort_until);
        }
        let aborting = abort_until.is_some();
        let sleep_duration = abort_until
            .or(active_deadline)
            .map(|until| until.saturating_duration_since(Instant::now()));
        tokio::select! {
            biased;
            _ = cancel.changed(), if !aborting => {}
            () = output.closed(), if !aborting => {}
            () = tokio::time::sleep(sleep_duration.unwrap_or_default()), if sleep_duration.is_some() => {
                if aborting {
                    // A silent D may still own RDMA receive buffers. Quarantine the worker,
                    // close the RPC, and keep draining until a native terminal receipt arrives.
                    quarantined.fetch_add(1, Ordering::AcqRel);
                    push_terminal_error(
                        &output,
                        &terminal_error,
                        Status::unavailable("native cancellation unconfirmed; worker quarantined"),
                    );
                    drop(output);
                    while native.next().await.is_some() {}
                    return (native.native_terminal_received(), true);
                }
            }
            layout = layouts.recv(), if needs_layout && !layout_sent && !aborting => {
                if let Some(layout_bytes) = layout {
                    layout_sent = true;
                    let event = sg::GenerateResponse {
                        request_id: rid.clone(),
                        prepared_input: layout_bytes,
                        ..Default::default()
                    };
                    if !send_owned_event(&output, event, &mut cancel, Some(deadline)).await {
                        push_terminal_error(
                            &output,
                            &terminal_error,
                            Status::resource_exhausted("managed output stalled"),
                        );
                        stalled = true;
                    }
                    for response in pending.drain(..) {
                        if !send_owned_event(&output, response, &mut cancel, Some(deadline)).await {
                            stalled = true;
                            break;
                        }
                    }
                    if native_finished {
                        return (true, false);
                    }
                }
            }
            item = native.next(), if !native_finished => {
                let safe = native.native_terminal_received();
                match item {
                    Some(Ok(response)) if !aborting => {
                        let event = native.to_sglang_response(&rid, response);
                        if needs_layout && !layout_sent {
                            if pending.len() >= 8 {
                                stalled = true;
                            } else {
                                pending.push(event);
                            }
                        } else if !send_owned_event(&output, event, &mut cancel, active_deadline).await {
                            stalled = true;
                        }
                    }
                    Some(Err(error)) => {
                        push_terminal_error(&output, &terminal_error, error);
                        if safe {
                            return (true, false);
                        }
                        begin_abort(&mut abort_until);
                    }
                    None if safe && needs_layout && !layout_sent && !aborting => {
                        native_finished = true;
                    }
                    None => return (safe, false),
                    _ => {}
                }
            }
        }
    }
}

/// Reserve a slot on `output` and send `event` unless cancelled or past `deadline`.
async fn send_owned_event(
    output: &mpsc::Sender<Result<sg::GenerateResponse, Status>>,
    event: sg::GenerateResponse,
    cancel: &mut watch::Receiver<bool>,
    deadline: Option<Instant>,
) -> bool {
    if *cancel.borrow() || deadline.is_some_and(|dl| Instant::now() >= dl) {
        return false;
    }
    let remaining = deadline.map(|dl| dl.saturating_duration_since(Instant::now()));
    let permit = tokio::select! {
        biased;
        _ = cancel.changed() => None,
        () = tokio::time::sleep(remaining.unwrap_or_default()), if remaining.is_some() => None,
        permit = output.reserve() => permit.ok(),
    };
    if let Some(permit) = permit {
        permit.send(Ok(event));
        true
    } else {
        false
    }
}
