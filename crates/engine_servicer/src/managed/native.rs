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
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;
use zeromq::{PullSocket, Socket, SocketRecv};

use super::{decode::NativeInput, service::NativeRuntime};
use crate::{BoxFuture, BoxStream};

struct Entry {
    operation: proto::OperationRef,
    rid: String,
    expiry: u64,
    cancel: watch::Sender<bool>,
    completion: watch::Receiver<Option<bool>>,
    layout: mpsc::Sender<proto::PreparedDecodeInput>,
    binding: proto::InputBinding,
}

/// Each native task owns its stream independently of the lifetime of a control RPC.
pub struct ManagedNativeRuntime {
    client: Arc<dyn Fn() -> Option<ZmqEngineClient> + Send + Sync>,
    entries: Mutex<HashMap<String, Entry>>,
    max_entries: usize,
    limits: Limits,
    metadata_failed: AtomicBool,
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
        max_entries: usize,
        limits: Limits,
    ) -> Result<Self, Status> {
        if max_entries == 0 {
            return Err(Status::invalid_argument("native registry must be bounded"));
        }
        Ok(Self {
            client,
            entries: Mutex::new(HashMap::new()),
            max_entries,
            limits,
            metadata_failed: AtomicBool::new(false),
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

    /// Reserve both output and cancellation ownership before native Generate can be sent.
    fn submit(
        &self,
        operation: proto::OperationRef,
        binding: proto::InputBinding,
        generation: sglang_proto::GenerateRequest,
        external: Option<proto::ExternalEncodePlan>,
        manifest: Option<Box<proto::PreparedDecodeInput>>,
        deadline: Deadline,
    ) -> Result<BoxStream<proto::ControlEvent>, Status> {
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
        entries.retain(|_, entry| entry.expiry >= now || *entry.completion.borrow() != Some(true));
        if entries.contains_key(&operation.operation_id) {
            return Err(Status::already_exists(
                "native operation already registered",
            ));
        }
        if entries.len() >= self.max_entries {
            return Err(Status::resource_exhausted("native registry full"));
        }
        deadline.remaining(Instant::now())?;
        let (cancel, cancelled) = watch::channel(false);
        let (completion, completed) = watch::channel(None);
        let (layout, layouts) = mpsc::channel(1);
        let (output, receiver) = mpsc::channel(8);
        entries.insert(
            operation.operation_id.clone(),
            Entry {
                operation: operation.clone(),
                rid: generation.request_id.clone(),
                expiry: deadline.expiry_unix_ms(),
                cancel,
                completion: completed,
                layout,
                binding: binding.clone(),
            },
        );
        #[expect(
            clippy::disallowed_methods,
            reason = "bounded native registry owns submission through acknowledged termination"
        )]
        tokio::spawn(async move {
            let safe = run_native(
                client, binding, generation, external, manifest, deadline, cancelled, layouts,
                &output,
            )
            .await;
            let _ = completion.send(Some(safe));
        });
        Ok(Box::pin(ReceiverStream::new(receiver)))
    }
}

impl NativeRuntime for ManagedNativeRuntime {
    /// A connected engine is insufficient when its required metadata channel has failed.
    fn ready(&self) -> bool {
        !self.metadata_failed.load(Ordering::Acquire) && (self.client)().is_some()
    }
    /// Install export ownership synchronously before returning the dispatch future.
    fn prefill(
        &self,
        request: proto::PrefillStart,
        deadline: Deadline,
    ) -> BoxFuture<Result<BoxStream<proto::ControlEvent>, Status>> {
        // Registration happens synchronously, before the caller can cancel this future.
        let result = (|| {
            self.submit(
                request
                    .operation
                    .ok_or_else(|| Status::invalid_argument("missing P operation"))?,
                request
                    .binding
                    .ok_or_else(|| Status::invalid_argument("missing P binding"))?,
                request
                    .generation
                    .ok_or_else(|| Status::invalid_argument("missing P generation"))?,
                Some(
                    request
                        .encode_plan
                        .ok_or_else(|| Status::invalid_argument("missing P encode plan"))?,
                ),
                None,
                deadline,
            )
        })();
        Box::pin(async move { result })
    }

    /// Only a complete validated input crosses the native Generate boundary.
    fn decode(
        &self,
        operation: proto::OperationRef,
        binding: proto::InputBinding,
        deadline: Deadline,
        input: NativeInput,
    ) -> BoxFuture<Result<BoxStream<sglang_proto::GenerateResponse>, Status>> {
        let (generation, manifest) = match input {
            NativeInput::Text(generation) => (generation, None),
            NativeInput::Prepared {
                generation,
                manifest,
            } => (generation, Some(manifest)),
        };
        let result = self.submit(operation, binding, generation, None, manifest, deadline);
        Box::pin(async move {
            let mut stream = result?;
            // Registration of the owner is not native submission. Wait for the
            // ordered transport write; dropping this wait never drops that owner.
            let submitted =
                tokio::time::timeout(deadline.remaining(Instant::now())?, stream.next())
                    .await
                    .map_err(|_| Status::deadline_exceeded("native D submission expired"))?
                    .ok_or_else(|| Status::cancelled("native D closed before submission"))??;
            if !matches!(submitted.event, Some(proto::control_event::Event::Stage(stage)) if stage == proto::Stage::NativeSubmitted as i32)
            {
                return Err(Status::data_loss(
                    "native D output preceded submission acknowledgement",
                ));
            }
            let stream = stream.filter_map(|event| async move {
                match event {
                    Ok(proto::ControlEvent {
                        event: Some(proto::control_event::Event::Generation(value)),
                        ..
                    }) => Some(Ok(value)),
                    Err(error) => Some(Err(error)),
                    _ => None,
                }
            });
            Ok(Box::pin(stream) as BoxStream<_>)
        })
    }

    /// Native terminal output, rather than an Abort write, establishes safe release.
    fn terminate(&self, operation: proto::OperationRef) -> BoxFuture<Result<bool, Status>> {
        let result = (|| {
            let entries = self
                .entries
                .lock()
                .map_err(|_| Status::internal("native registry poisoned"))?;
            // This owner registers synchronously before sending, and only collects
            // confirmed terminal entries. Absence therefore proves no live native owner.
            let Some(entry) = entries.get(&operation.operation_id) else {
                return Ok(None);
            };
            if entry.operation != operation {
                return Err(Status::failed_precondition(
                    "native operation fence mismatch",
                ));
            }
            let _ = entry.cancel.send(true);
            Ok(Some(entry.completion.clone()))
        })();
        Box::pin(async move {
            let Some(mut completion) = result? else {
                return Ok(true);
            };
            loop {
                if let Some(safe) = *completion.borrow_and_update() {
                    return Ok(safe);
                }
                completion
                    .changed()
                    .await
                    .map_err(|_| Status::unavailable("native owner lost"))?;
            }
        })
    }
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
    output: &mpsc::Sender<Result<proto::ControlEvent, Status>>,
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
            // These refusals originate in adapter validation before client.submit.
            // Transport failures stay unknown because a write may have reached native.
            let unsubmitted = matches!(
                error.code(),
                tonic::Code::InvalidArgument
                    | tonic::Code::FailedPrecondition
                    | tonic::Code::Unimplemented
            );
            let _ = output.try_send(Err(error));
            return unsubmitted;
        }
    };
    let rid = native.request_id().to_owned();
    let rank = native.engine_index().unwrap_or(0);
    let mut aborting = false;
    let mut abort_until = Instant::now();
    let mut layout_sent = false;
    // This event acknowledges the actual Generate write, not task creation or GPU execution.
    let mut stalled = !send_owned_event(
        output,
        proto::ControlEvent {
            event: Some(proto::control_event::Event::Stage(
                proto::Stage::NativeSubmitted as i32,
            )),
            ..Default::default()
        },
        &mut cancel,
        &deadline,
    )
    .await;
    let needs_layout = external.as_ref().is_some_and(|plan| !plan.parts.is_empty());
    let mut native_finished = false;
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
            // Failure to send an Abort still leaves native occupancy unknown.
            if !matches!(
                tokio::time::timeout(
                    Duration::from_secs(5),
                    client.abort_sglang_managed(rank, &rid)
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
                    // Keep the bounded native registration: a late terminal output
                    // must still release admission, without replaying Generate/Abort.
                    tracing::warn!(rid, "managed native abort acknowledgement still pending");
                    abort_until = Instant::now() + Duration::from_secs(30);
                }
            }
            layout = layouts.recv(), if needs_layout && !layout_sent && !aborting => {
                if let Some(layout) = layout {
                    layout_sent = true;
                    if !send_owned_event(output, proto::ControlEvent { event: Some(proto::control_event::Event::MediaPrepared(layout)), ..Default::default() }, &mut cancel, &deadline).await {
                        // Force the owned task into ordered abort without releasing its stream.
                        let _ = output.try_send(Err(Status::resource_exhausted("managed output stalled")));
                        stalled = true;
                    }
                    if native_finished { return true; }
                }
            }
            item = native.next(), if !native_finished => {
                let safe = native.native_terminal_received();
                match item {
                    Some(Ok(response)) if !aborting => {
                        let response = native.to_sglang_response(&rid, response);
                        if !send_owned_event(output, proto::ControlEvent { event: Some(proto::control_event::Event::Generation(response)), ..Default::default() }, &mut cancel, &deadline).await {
                            stalled = true;
                        }
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

/// Backpressure transient output bursts while preserving cancellation and the original deadline.
async fn send_owned_event(
    output: &mpsc::Sender<Result<proto::ControlEvent, Status>>,
    event: proto::ControlEvent,
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

    /// A transiently full channel waits for its consumer; cancellation still interrupts that wait.
    #[tokio::test]
    async fn output_burst_backpressures_without_cancelling_native_work() {
        let (sender, mut receiver) = mpsc::channel(1);
        sender
            .send(Ok(proto::ControlEvent::default()))
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
            proto::ControlEvent::default(),
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
                proto::ControlEvent::default(),
                &mut cancelled,
                &deadline
            )
            .await
        );
    }
}
