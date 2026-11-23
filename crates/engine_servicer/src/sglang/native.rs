//! Own native submission and output registration until the scheduler acknowledges completion.

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

use engine_zmq_adapter::ZmqEngineClient;
use futures::{Stream, StreamExt};
use prost::Message;
use smg_grpc_client::{
    sglang_managed::{proto, validate_manifest, Limits, LAYOUT, VERSION},
    sglang_proto,
};
use tokio::sync::{mpsc, oneshot, watch};
use tonic::Status;
use zeromq::{PullSocket, Socket, SocketRecv};

use crate::{
    requests::{CancelReason, Registration},
    BoxStream,
};

const DEFAULT_ABORT_GRACE: Duration = Duration::from_secs(30);

/// Explicit managed-only launch configuration; legacy launch has no such value.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagedConfig {
    pub role: String,
    pub metadata_endpoint: Option<String>,
    #[serde(default)]
    pub disable_routing_labels: bool,
    pub max_records: usize,
    pub max_duration_ms: u64,
    pub model_revision: String,
    pub processor_contract: String,
    pub embedding_dtype: String,
    pub kv_backend: String,
    pub embedding_backend: String,
    pub topology_contract: String,
    pub bootstrap_host: String,
    pub bootstrap_port: u32,
    pub tp_size: u32,
}

impl ManagedConfig {
    /// Validate before binding any listener or launching a scheduler.
    pub fn validate(&self) -> Result<(), Status> {
        if self.max_duration_ms == 0
            || self.max_records == 0
            || self.model_revision.is_empty()
            || self.processor_contract.is_empty()
            || self.topology_contract.is_empty()
            || self.tp_size == 0
            || !matches!(
                self.embedding_dtype.as_str(),
                "bfloat16" | "float16" | "float32"
            )
            || !matches!(
                self.embedding_backend.as_str(),
                "mooncake" | "zmq_to_scheduler"
            )
            || self.kv_backend.is_empty()
            || (self.role == "prefill"
                && (self.bootstrap_host.is_empty()
                    || self.bootstrap_port == 0
                    || self.bootstrap_port > u32::from(u16::MAX)))
        {
            return Err(Status::invalid_argument(
                "incomplete managed capability contract",
            ));
        }
        if !matches!(self.role.as_str(), "prefill" | "decode")
            || (self.role == "prefill"
                && !self
                    .metadata_endpoint
                    .as_ref()
                    .is_some_and(|s| s.starts_with("ipc://")))
            || (self.role == "decode" && self.metadata_endpoint.is_some())
        {
            return Err(Status::invalid_argument(
                "invalid managed role/metadata endpoint",
            ));
        }
        Ok(())
    }

    /// Advertise the managed contract; readiness is checked per query.
    pub fn capabilities(&self) -> proto::Capabilities {
        proto::Capabilities {
            version: VERSION,
            role: self.role.clone(),
            model_revision: self.model_revision.clone(),
            processor_contract: self.processor_contract.clone(),
            layout_contract: LAYOUT.into(),
            embedding_dtype: self.embedding_dtype.clone(),
            kv_backend: self.kv_backend.clone(),
            embedding_backend: self.embedding_backend.clone(),
            topology_contract: self.topology_contract.clone(),
            bootstrap_host: self.bootstrap_host.clone(),
            bootstrap_port: self.bootstrap_port,
            tp_size: self.tp_size,
            ..Default::default()
        }
    }
}

/// Validated request passed from prepared admission to its independent native owner.
pub(crate) struct PreparedRequest {
    pub(crate) request: sglang_proto::GenerateRequest,
    pub(crate) binding: proto::InputBinding,
    pub(crate) external: Option<(String, u32)>,
    pub(crate) manifest: Option<Box<proto::PreparedDecodeInput>>,
    pub(crate) deadline: Instant,
}

struct Entry {
    cancel: watch::Sender<bool>,
    layout: mpsc::Sender<Vec<u8>>,
    binding: proto::InputBinding,
    #[cfg(test)]
    output_observer: mpsc::WeakSender<Result<sglang_proto::GenerateResponse, Status>>,
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

    /// One reader per P instance; malformed/oversized events cannot be attributed safely and are dropped
    /// (bounded by each owner's prepared deadline), while binding-matched invalid layouts cancel their owner.
    pub async fn read_metadata(&self, mut socket: PullSocket) -> Result<(), Status> {
        let result = self.read_metadata_inner(&mut socket).await;
        self.metadata_failed.store(true, Ordering::Release);
        if let Ok(entries) = self.entries.lock() {
            for entry in entries.values() {
                let _ = entry.cancel.send(true);
            }
        }
        result
    }

    async fn read_metadata_inner(&self, socket: &mut PullSocket) -> Result<(), Status> {
        loop {
            let frames = socket
                .recv()
                .await
                .map_err(|e| Status::unavailable(e.to_string()))?
                .into_vec();
            if frames.len() != 2 {
                return Err(Status::data_loss("native metadata frame count mismatch"));
            }
            let rid = std::str::from_utf8(&frames[0])
                .map_err(|_| Status::data_loss("invalid native metadata rid"))?;
            if let Err(error) = self.publish_layout(rid, &frames[1]) {
                tracing::warn!(rid, %error, "native managed metadata refused");
            }
        }
    }

    /// Create a bounded native runtime backed by the engine client accessor.
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
    pub(crate) fn with_abort_grace(self, grace: Duration) -> Self {
        self.abort_grace_ms
            .store(grace.as_millis() as u64, Ordering::Release);
        self
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
    /// mismatched-binding frames never cancel an unrelated attempt (its prepared deadline bounds D).
    /// Only validation or delivery failures whose binding matches the registered attempt cancel it.
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
    ) -> Result<BoxStream<sglang_proto::GenerateResponse>, Status> {
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

/// Typed admission refusals precede enqueue; unknown transport writes retain ownership.
fn definitely_unsubmitted(error: &Status) -> bool {
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
            )
        })
}

/// Bounded chunk stream with an out-of-band terminal error slot so cancellation or
/// failure status is never dropped when the 8-slot chunk queue is full.
struct PreparedOutputStream {
    receiver: mpsc::Receiver<Result<sglang_proto::GenerateResponse, Status>>,
    terminal_error: Arc<Mutex<Option<Status>>>,
}

impl Stream for PreparedOutputStream {
    type Item = Result<sglang_proto::GenerateResponse, Status>;

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

pub(crate) fn push_terminal_error(
    output: &mpsc::Sender<Result<sglang_proto::GenerateResponse, Status>>,
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
/// submitted to the scheduler, total generation duration (both streaming and buffered
/// `stream=false` requests) follows the gateway's `--grpc-request-timeout-secs` (via
/// stream close/abort), matching ordinary text PD.
#[expect(clippy::too_many_arguments)]
async fn run_native(
    client: ZmqEngineClient,
    input: PreparedRequest,
    mut cancel: watch::Receiver<bool>,
    mut layouts: mpsc::Receiver<Vec<u8>>,
    output: mpsc::Sender<Result<sglang_proto::GenerateResponse, Status>>,
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
    let ext_ref = external.as_ref().map(|(ep, n)| (ep.as_str(), *n));
    let mut native = match client
        .generate_sglang_managed(request, &binding, ext_ref, manifest.as_deref())
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
    // D must retain its receive buffers until native output proves the PD handoff finished.
    let mut waiting_for_decode = external.is_none();
    let mut aborting = false;
    let mut abort_until = Instant::now();
    let mut layout_sent = false;
    let mut stalled = output.is_closed();
    let needs_layout = external.as_ref().is_some_and(|(_, images)| *images > 0);
    let mut native_finished = false;
    let mut pending = Vec::new();
    let send_abort = |client: &ZmqEngineClient| {
        if let Err(error) = client.abort_sglang_managed(rank, &rid) {
            tracing::warn!(rid, %error, "managed Abort write unconfirmed; retaining terminal observer");
        }
    };
    let active_deadline = external.is_some().then_some(deadline);
    loop {
        if !aborting
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
            aborting = true;
            abort_until = Instant::now() + abort_grace;
            if !waiting_for_decode {
                send_abort(&client);
            }
        }
        let sleep_duration = if aborting {
            Some(abort_until.saturating_duration_since(Instant::now()))
        } else {
            active_deadline.map(|dl| dl.saturating_duration_since(Instant::now()))
        };
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
                    while let Some(item) = native.next().await {
                        if waiting_for_decode && item.is_ok() {
                            waiting_for_decode = false;
                            if !native.native_terminal_received() {
                                send_abort(&client);
                            }
                        }
                    }
                    return (native.native_terminal_received(), true);
                }
            }
            layout = layouts.recv(), if needs_layout && !layout_sent && !aborting => {
                if let Some(layout_bytes) = layout {
                    layout_sent = true;
                    let event = sglang_proto::GenerateResponse {
                        request_id: rid.clone(),
                        prepared_input: layout_bytes,
                        ..Default::default()
                    };
                    if !send_owned_event(&output, &terminal_error, Ok(event), &mut cancel, Some(deadline)).await {
                        push_terminal_error(
                            &output,
                            &terminal_error,
                            Status::resource_exhausted("managed output stalled"),
                        );
                        stalled = true;
                    }
                    for response in pending.drain(..) {
                        if !send_owned_event(&output, &terminal_error, Ok(response), &mut cancel, Some(deadline)).await {
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
                if waiting_for_decode && matches!(&item, Some(Ok(_))) {
                    waiting_for_decode = false;
                    if aborting && !safe {
                        send_abort(&client);
                    }
                }
                match item {
                    Some(Ok(response)) if !aborting => {
                        let event = native.to_sglang_response(&rid, response);
                        if needs_layout && !layout_sent {
                            if pending.len() >= 8 {
                                stalled = true;
                            } else {
                                pending.push(event);
                            }
                        } else if !send_owned_event(&output, &terminal_error, Ok(event), &mut cancel, active_deadline).await {
                            stalled = true;
                        }
                    }
                    Some(Err(error)) => {
                        push_terminal_error(&output, &terminal_error, error);
                        if safe {
                            return (true, false);
                        }
                        if !aborting {
                            aborting = true;
                            abort_until = Instant::now() + abort_grace;
                            if !waiting_for_decode {
                                send_abort(&client);
                            }
                        }
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

async fn send_owned_event(
    output: &mpsc::Sender<Result<sglang_proto::GenerateResponse, Status>>,
    terminal_error: &Mutex<Option<Status>>,
    item: Result<sglang_proto::GenerateResponse, Status>,
    cancel: &mut watch::Receiver<bool>,
    deadline: Option<Instant>,
) -> bool {
    if *cancel.borrow() || deadline.is_some_and(|dl| Instant::now() >= dl) {
        if let Err(status) = item {
            push_terminal_error(output, terminal_error, status);
        }
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
        permit.send(item);
        true
    } else {
        if let Err(status) = item {
            push_terminal_error(output, terminal_error, status);
        }
        false
    }
}
