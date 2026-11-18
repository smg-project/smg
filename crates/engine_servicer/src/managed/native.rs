//! Own native submission and output registration until the scheduler acknowledges completion.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use engine_zmq_adapter::ZmqEngineClient;
use futures::StreamExt;
use prost::Message;
use smg_grpc_client::{
    sglang_managed::{budget::Deadline, proto, validate_manifest, Limits},
    sglang_proto,
};
use tokio::sync::{mpsc, oneshot, watch, OwnedSemaphorePermit, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;
use zeromq::{PullSocket, Socket, SocketRecv};

use super::ManagedConfig;
use crate::{
    requests::{CancelReason, Registration},
    BoxStream,
};

struct Entry {
    rid: String,
    expiry: u64,
    cancel: watch::Sender<bool>,
    layout: mpsc::Sender<proto::PreparedDecodeInput>,
    binding: proto::InputBinding,
    registration: Option<Registration>,
    context: Option<OwnedSemaphorePermit>,
}

/// Each native task owns its stream independently of the lifetime of a control RPC.
pub struct ManagedNativeRuntime {
    client: Arc<dyn Fn() -> Option<ZmqEngineClient> + Send + Sync>,
    entries: Arc<Mutex<HashMap<String, Entry>>>,
    max_entries: usize,
    context: Arc<Semaphore>,
    max_active: usize,
    retention_ms: u64,
    limits: Limits,
    metadata_failed: Arc<AtomicBool>,
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

    /// One reader per P instance; malformed events fail their owner instead of hanging Bind.
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

    /// A fatal IPC error closes this instance's capability, cancelling existing export owners.
    async fn read_metadata_inner(&self, socket: &mut PullSocket) -> Result<(), Status> {
        loop {
            let message = socket
                .recv()
                .await
                .map_err(|e| Status::unavailable(e.to_string()))?;
            let frames = message.into_vec();
            if frames.len() != 2 {
                return Err(Status::data_loss("native metadata frame count mismatch"));
            }
            let rid = std::str::from_utf8(&frames[0])
                .map_err(|_| Status::data_loss("invalid native metadata rid"))?;
            if let Err(error) = self.publish_layout(rid, &frames[1]) {
                tracing::warn!(rid, %error, "native managed metadata refused");
                // A malformed export belongs to a registered attempt and must cancel it.
                if let Ok(entries) = self.entries.lock() {
                    if let Some(entry) = entries.values().find(|entry| entry.rid == rid) {
                        let _ = entry.cancel.send(true);
                    }
                }
            }
        }
    }
    /// The client getter must refer to the same instance's established native connection.
    pub fn new(
        client: Arc<dyn Fn() -> Option<ZmqEngineClient> + Send + Sync>,
        config: &ManagedConfig,
        limits: Limits,
    ) -> Result<Self, Status> {
        let max_entries = config.max_records;
        let max_active = config
            .max_work
            .min(config.max_context_bytes / (2 * limits.manifest_bytes as u64))
            as usize;
        if max_entries == 0 || max_active == 0 {
            return Err(Status::invalid_argument("native registry must be bounded"));
        }
        let context_bytes = usize::try_from(config.max_context_bytes)
            .ok()
            .filter(|bytes| *bytes <= Semaphore::MAX_PERMITS)
            .ok_or_else(|| {
                Status::invalid_argument("native context byte budget exceeds address space")
            })?;
        Ok(Self {
            client,
            entries: Arc::new(Mutex::new(HashMap::new())),
            max_entries,
            context: Arc::new(Semaphore::new(context_bytes)),
            max_active,
            retention_ms: config.retention_ms,
            limits,
            metadata_failed: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The same-host IPC reader calls this only for the native P rid in its message.
    /// Registry lookup precedes protobuf decoding and no early unregistered event is retained.
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
            .values()
            .find(|entry| entry.rid == rid)
            .ok_or_else(|| Status::not_found("unregistered native layout"))?;
        if *entry.cancel.borrow() {
            return Err(Status::cancelled(
                "native layout arrived after cancellation",
            ));
        }
        let manifest = proto::PreparedDecodeInput::decode(bytes)
            .map_err(|_| Status::data_loss("malformed native layout"))?;
        validate_manifest(&manifest, &entry.binding, self.limits)?;
        entry.layout.try_send(manifest).map_err(|_| {
            let _ = entry.cancel.send(true);
            Status::resource_exhausted("native layout channel full or closed")
        })
    }

    /// Charge compact Start bytes plus bounded bookkeeping, not a native execution slot.
    pub(crate) fn reserve_start(&self, bytes: usize) -> Result<OwnedSemaphorePermit, Status> {
        const START_BOOKKEEPING_BYTES: usize = 4096;
        let bytes = bytes
            .checked_add(START_BOOKKEEPING_BYTES)
            .and_then(|bytes| u32::try_from(bytes).ok())
            .ok_or_else(|| Status::resource_exhausted("decode Start byte count overflow"))?;
        Arc::clone(&self.context)
            .try_acquire_many_owned(bytes)
            .map_err(|_| Status::resource_exhausted("prepared context byte budget exhausted"))
    }

    /// Reserve both output and cancellation ownership before native Generate can be sent.
    #[expect(
        clippy::too_many_arguments,
        reason = "one native owner takes the frozen request and its ordinary admission guard"
    )]
    pub(crate) fn submit(
        &self,
        operation: proto::OperationRef,
        binding: proto::InputBinding,
        generation: sglang_proto::GenerateRequest,
        external: Option<proto::ExternalEncodePlan>,
        manifest: Option<Box<proto::PreparedDecodeInput>>,
        deadline: Deadline,
        registration: Registration,
        mut abort: oneshot::Receiver<CancelReason>,
        context: Option<OwnedSemaphorePermit>,
    ) -> Result<BoxStream<sglang_proto::GenerateResponse>, Status> {
        if self.metadata_failed.load(Ordering::Acquire) {
            return Err(Status::unavailable("native metadata channel failed"));
        }
        let client =
            (self.client)().ok_or_else(|| Status::unavailable("native engine not ready"))?;
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| Status::internal("native registry poisoned"))?;
        let now = wall_time_ms()?;
        // Only confirmed terminal requests can leave this registry. Unknown is never free.
        entries.retain(|_, entry| entry.expiry >= now || entry.registration.is_some());
        if entries
            .values()
            .filter(|e| e.registration.is_some())
            .count()
            >= self.max_active
        {
            return Err(Status::resource_exhausted(
                "native context capacity exhausted",
            ));
        }
        if entries.contains_key(&operation.operation_id) {
            return Err(Status::already_exists(
                "native operation already registered",
            ));
        }
        if entries.len() >= self.max_entries {
            return Err(Status::resource_exhausted("native registry full"));
        }
        deadline.remaining(Instant::now())?;
        // Upgrade the waiting charge to the existing native context envelope,
        // retaining the same permit until acknowledged native release.
        let bytes = self
            .limits
            .manifest_bytes
            .checked_mul(2)
            .ok_or_else(|| Status::resource_exhausted("native context byte count overflow"))?;
        let mut context = context.unwrap_or(
            Arc::clone(&self.context)
                .try_acquire_many_owned(0)
                .map_err(|_| Status::unavailable("native context budget closed"))?,
        );
        let extra = u32::try_from(bytes.saturating_sub(context.num_permits()))
            .map_err(|_| Status::resource_exhausted("native context byte count overflow"))?;
        context.merge(
            Arc::clone(&self.context)
                .try_acquire_many_owned(extra)
                .map_err(|_| Status::resource_exhausted("native context byte budget exhausted"))?,
        );
        let (cancel, cancelled) = watch::channel(false);
        let (layout, layouts) = mpsc::channel(1);
        let (output, receiver) = mpsc::channel(8);
        entries.insert(
            operation.operation_id.clone(),
            Entry {
                rid: generation.request_id.clone(),
                expiry: deadline.expiry_unix_ms().saturating_add(self.retention_ms),
                cancel: cancel.clone(),
                layout,
                binding: binding.clone(),
                registration: Some(registration),
                context: Some(context),
            },
        );
        let owners = Arc::clone(&self.entries);
        let failed = Arc::clone(&self.metadata_failed);
        #[expect(
            clippy::disallowed_methods,
            reason = "native task retains submission through acknowledged termination"
        )]
        tokio::spawn(async move {
            let work = run_native(
                client,
                binding,
                generation,
                external,
                manifest,
                deadline,
                cancelled,
                layouts,
                output,
                Arc::clone(&failed),
            );
            tokio::pin!(work);
            let safe = tokio::select! {
                result = &mut work => result,
                _ = &mut abort => { let _ = cancel.send(true); work.await }
            };
            if !safe {
                failed.store(true, Ordering::Release);
            }
            if safe {
                if let Ok(mut entries) = owners.lock() {
                    if let Some(entry) = entries.get_mut(&operation.operation_id) {
                        entry.registration.take();
                        entry.context.take();
                    }
                }
            }
        });
        Ok(Box::pin(ReceiverStream::new(receiver)))
    }
}

impl ManagedNativeRuntime {
    /// A connected engine cannot serve prepared inputs after metadata export fails.
    pub fn ready(&self) -> bool {
        !self.metadata_failed.load(Ordering::Acquire)
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

/// Preserve Generate→Abort ordering while retaining the native output registration.
#[expect(
    clippy::too_many_arguments,
    reason = "single task owns native submission and its bounded channels"
)]
async fn run_native(
    client: ZmqEngineClient,
    binding: proto::InputBinding,
    generation: sglang_proto::GenerateRequest,
    external: Option<proto::ExternalEncodePlan>,
    manifest: Option<Box<proto::PreparedDecodeInput>>,
    deadline: Deadline,
    mut cancel: watch::Receiver<bool>,
    mut layouts: mpsc::Receiver<proto::PreparedDecodeInput>,
    output: mpsc::Sender<Result<sglang_proto::GenerateResponse, Status>>,
    failed: Arc<AtomicBool>,
) -> bool {
    if *cancel.borrow() || deadline.remaining(Instant::now()).is_err() {
        return true;
    }
    // This future is owned here even if the RPC disappears during send.
    let mut native = match client
        .generate_sglang_managed(generation, &binding, external.as_ref(), manifest.as_deref())
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            let unsubmitted = definitely_unsubmitted(&error);
            let _ = output.try_send(Err(error));
            return unsubmitted;
        }
    };
    let rid = native.request_id().to_owned();
    let rank = native.engine_index().unwrap_or(0);
    // D must retain its receive buffers until native output proves the PD handoff
    // finished. P always carries an external plan, including the text-only plan.
    let mut waiting_for_decode = external.is_none();
    let mut aborting = false;
    let mut abort_until = Instant::now();
    let mut layout_sent = false;
    let mut stalled = output.is_closed();
    let needs_layout = external.as_ref().is_some_and(|plan| !plan.parts.is_empty());
    let mut native_finished = false;
    let mut pending = Vec::new();
    loop {
        if !aborting
            && (stalled
                || *cancel.borrow()
                || output.is_closed()
                || deadline.remaining(Instant::now()).is_err())
        {
            if native_finished {
                return true;
            }
            aborting = true;
            abort_until = Instant::now() + Duration::from_secs(30);
            if !waiting_for_decode {
                send_abort(&client, rank, &rid).await;
            }
        }
        let remaining = if aborting {
            abort_until.saturating_duration_since(Instant::now())
        } else {
            deadline.remaining(Instant::now()).unwrap_or_default()
        };
        tokio::select! {
            biased;
            _ = cancel.changed(), if !aborting => {}
            () = output.closed(), if !aborting => {}
            () = tokio::time::sleep(remaining) => {
                if aborting {
                    // A silent D may still own RDMA receive buffers. Quarantine
                    // the instance and close the RPC at the grace deadline, but
                    // retain its native observer and admission until a receipt.
                    failed.store(true, Ordering::Release);
                    let _ = output.try_send(Err(Status::unavailable("native cancellation unconfirmed; worker quarantined")));
                    drop(output);
                    while let Some(item) = native.next().await {
                        if waiting_for_decode && item.is_ok() {
                            waiting_for_decode = false;
                            if !native.native_terminal_received() {
                                send_abort(&client, rank, &rid).await;
                            }
                        }
                    }
                    return native.native_terminal_received();
                }
            }
            layout = layouts.recv(), if needs_layout && !layout_sent && !aborting => {
                if let Some(layout) = layout {
                    layout_sent = true;
                    if !send_owned_event(&output, sglang_proto::GenerateResponse { request_id: rid.clone(), prepared_input: layout.encode_to_vec(), ..Default::default() }, &mut cancel, &deadline).await {
                        // Force the owned task into ordered abort without releasing its stream.
                        let _ = output.try_send(Err(Status::resource_exhausted("managed output stalled")));
                        stalled = true;
                    }
                    for response in pending.drain(..) {
                        if !send_owned_event(&output, response, &mut cancel, &deadline).await { stalled = true; break; }
                    }
                    if native_finished { return true; }
                }
            }
            item = native.next(), if !native_finished => {
                let safe = native.native_terminal_received();
                if waiting_for_decode && matches!(&item, Some(Ok(_))) {
                    waiting_for_decode = false;
                    if aborting && !safe {
                        send_abort(&client, rank, &rid).await;
                    }
                }
                match item {
                    Some(Ok(response)) if !aborting => {
                        let response = native.to_sglang_response(&rid, response);
                        let event = response;
                        if needs_layout && !layout_sent {
                            if pending.len() >= 8 { stalled = true; } else { pending.push(event); }
                        } else if !send_owned_event(&output, event, &mut cancel, &deadline).await { stalled = true; }
                    }
                    Some(Err(error)) => { let _ = output.try_send(Err(error)); return safe; }
                    None if safe && needs_layout && !layout_sent && !aborting => { native_finished = true; }
                    None => return safe,
                    _ => {}
                }
            }
        }
    }
}

/// Send Abort only once the native receive buffers may be cancelled, retaining the receipt owner.
async fn send_abort(client: &ZmqEngineClient, rank: u32, rid: &str) {
    if !matches!(
        tokio::time::timeout(
            Duration::from_secs(5),
            client.abort_sglang_managed(rank, rid)
        )
        .await,
        Ok(Ok(()))
    ) {
        tracing::warn!(
            rid,
            "managed Abort write unconfirmed; retaining native terminal observer"
        );
    }
}

/// Backpressure transient output bursts while preserving cancellation and the original deadline.
async fn send_owned_event(
    output: &mpsc::Sender<Result<sglang_proto::GenerateResponse, Status>>,
    event: sglang_proto::GenerateResponse,
    cancel: &mut watch::Receiver<bool>,
    deadline: &Deadline,
) -> bool {
    if *cancel.borrow() {
        return false;
    }
    let Ok(remaining) = deadline.remaining(Instant::now()) else {
        return false;
    };
    tokio::select! {
        biased;
        _ = cancel.changed() => false,
        () = tokio::time::sleep(remaining) => false,
        result = output.send(Ok(event)) => result.is_ok(),
    }
}

/// Reject clocks outside the wire timestamp range instead of wrapping expiry.
fn wall_time_ms() -> Result<u64, Status> {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Status::unavailable("invalid native worker clock"))?
            .as_millis(),
    )
    .map_err(|_| Status::unavailable("native clock overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A status code or message alone cannot establish that a native write never happened.
    #[test]
    fn admission_refusal_requires_typed_transport_evidence() {
        let mut refused = Status::unavailable("input full");
        assert!(!definitely_unsubmitted(&refused));
        refused.set_source(Arc::new(engine_zmq_client::Error::EngineInputFull {
            frames: 1,
            bytes: 1,
            request_bytes: 1,
            limit_frames: 1,
            limit_bytes: 1,
        }));
        assert!(definitely_unsubmitted(&refused));
        refused.set_source(Arc::new(engine_zmq_client::Error::EngineInputBytesFull {
            retained_bytes: 1,
            request_bytes: 1,
            limit_bytes: 1,
        }));
        assert!(definitely_unsubmitted(&refused));
        refused.set_source(Arc::new(engine_zmq_client::Error::EngineInputBlocked {
            timeout: Duration::from_secs(1),
        }));
        assert!(!definitely_unsubmitted(&refused));
        assert!(definitely_unsubmitted(&Status::invalid_argument(
            "invalid prompt"
        )));
    }

    /// A transiently full channel waits for its consumer; cancellation still interrupts that wait.
    #[tokio::test]
    async fn output_burst_backpressures_without_cancelling_native_work() {
        let (sender, mut receiver) = mpsc::channel(1);
        sender
            .send(Ok(sglang_proto::GenerateResponse::default()))
            .await
            .unwrap();
        let (cancel, mut cancelled) = watch::channel(false);
        let deadline = Deadline::admit(
            &proto::Budget {
                original_expiry_unix_ms: 2000,
                remaining_timeout_ms: 1000,
                ..Default::default()
            },
            1000,
            Instant::now(),
            0,
            0,
            1000,
        )
        .unwrap();
        let pending = send_owned_event(
            &sender,
            sglang_proto::GenerateResponse::default(),
            &mut cancelled,
            &deadline,
        );
        tokio::pin!(pending);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut pending)
                .await
                .is_err()
        );
        assert!(receiver.recv().await.is_some());
        assert!(pending.await);
        let mut cancelled = cancel.subscribe();
        cancel.send(true).unwrap();
        assert!(
            !send_owned_event(
                &sender,
                sglang_proto::GenerateResponse::default(),
                &mut cancelled,
                &deadline
            )
            .await
        );
    }
}
