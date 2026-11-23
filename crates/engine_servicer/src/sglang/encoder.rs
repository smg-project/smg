//! Same-host native encoder control behind gRPC. A cancellation message is only a wakeup;
//! only an explicit released response permits the control owner to return execution permits.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::sync_channel,
        Arc, Mutex,
    },
    time::Duration,
};

use bytes::Bytes;
use futures::{future::BoxFuture, FutureExt, TryFutureExt};
use prost::Message;
use serde::Deserialize;
use serde_json::Value;
use smg_grpc_client::{
    common_proto as common,
    sglang_encoder_proto::{self as ep, EncodeRequest},
    sglang_managed::{proto::Capabilities, LAYOUT, VERSION},
    sglang_proto as sg,
};
use tokio::{
    sync::{mpsc, oneshot, watch, Semaphore},
    time::Instant,
};
use tonic::{Request, Response, Status};
use zeromq::{PullSocket, PushSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use crate::{BoxStream, ServerThread, ServicerError};

type Reply = Result<Value, Status>;

const CANCEL_RECEIPT_GRACE: Duration = Duration::from_secs(30);

enum NativeCallError {
    /// The command frame was never handed to the native peer (e.g. ZMQ `ReturnToSender` or
    /// pre-send failure), so the operation never started in Python and its slot may be released.
    NotSent(Status),
    /// Delivery or release outcome is ambiguous; the owner must retain the slot and fail closed.
    Ambiguous(Status),
}

impl NativeCallError {
    fn into_status(self) -> Status {
        match self {
            Self::NotSent(status) | Self::Ambiguous(status) => status,
        }
    }
}

type PendingReply = Result<Value, NativeCallError>;

/// Private same-host IPC command/reply channel to Python native encoder runtime.
#[derive(Clone)]
pub struct NativeEncoderLink {
    state: Arc<NativeState>,
    commands: mpsc::Sender<(u64, ZmqMessage)>,
}

struct NativeState {
    sequence: AtomicU64,
    pending: Mutex<HashMap<u64, oneshot::Sender<PendingReply>>>,
    connected: AtomicBool,
    failed: AtomicBool,
}

struct PendingCall {
    state: Arc<NativeState>,
    sequence: u64,
}
impl Drop for PendingCall {
    fn drop(&mut self) {
        if let Ok(mut pending) = self.state.pending.lock() {
            pending.remove(&self.sequence);
        }
    }
}

#[derive(Deserialize)]
struct NativeReply {
    #[serde(default)]
    sequence: u64,
    #[serde(default)]
    ready: bool,
    #[serde(default)]
    result: Value,
    error: Option<String>,
    error_code: Option<i32>,
}

impl NativeEncoderLink {
    /// Bind bounded same-host IPC sockets and spawn dedicated writer and reply-reader tasks.
    pub async fn bind(
        commands_endpoint: &str,
        events_endpoint: &str,
        capacity: usize,
    ) -> Result<Self, Status> {
        if !commands_endpoint.starts_with("ipc://")
            || !events_endpoint.starts_with("ipc://")
            || capacity == 0
        {
            return Err(Status::invalid_argument(
                "encoder wire requires bounded same-host IPC",
            ));
        }
        let mut commands = PushSocket::new();
        let mut events = PullSocket::new();
        commands.bind(commands_endpoint).await.map_err(wire_error)?;
        events.bind(events_endpoint).await.map_err(wire_error)?;
        let (tx, mut rx) = mpsc::channel(capacity);
        let state = Arc::new(NativeState {
            sequence: AtomicU64::new(1),
            pending: Mutex::default(),
            connected: AtomicBool::new(false),
            failed: AtomicBool::new(false),
        });
        let link = Self {
            state: state.clone(),
            commands: tx,
        };
        let (closed, mut closing) = watch::channel(false);
        let writer = state.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "server runtime owns the native IPC writer"
        )]
        tokio::spawn(async move {
            while let Some((sequence, mut message)) = rx.recv().await {
                for attempt in 0..20 {
                    match commands.send(message).await {
                        Ok(()) => break,
                        Err(zeromq::ZmqError::ReturnToSender {
                            message: returned, ..
                        }) if attempt < 19 => {
                            message = returned;
                            tokio::time::sleep(Duration::from_millis(5)).await;
                        }
                        Err(zeromq::ZmqError::ReturnToSender { .. }) => {
                            if let Ok(mut pending) = writer.pending.lock() {
                                if let Some(sender) = pending.remove(&sequence) {
                                    let _ = sender.send(Err(NativeCallError::NotSent(
                                        Status::unavailable(
                                            "native encoder IPC peer not connected",
                                        ),
                                    )));
                                }
                            }
                            break;
                        }
                        Err(error) => {
                            writer.fail(wire_error(error));
                            closed.send_replace(true);
                            return;
                        }
                    }
                }
            }
            writer.fail(Status::unavailable("native encoder command owner closed"));
            closed.send_replace(true);
        });
        #[expect(
            clippy::disallowed_methods,
            reason = "server runtime owns the native IPC reader"
        )]
        tokio::spawn(async move {
            loop {
                let message = tokio::select! {
                    message = events.recv() => message,
                    _ = closing.changed() => return,
                };
                let result = match message {
                    Ok(message) => state.receive(message),
                    Err(error) => Err(wire_error(error)),
                };
                if let Err(error) = result {
                    state.fail(error);
                    return;
                }
            }
        });
        Ok(link)
    }

    /// Enqueue one JSON command (and optional binary artifact frame) on the FIFO command channel
    /// and return its pending reply handle.
    async fn enqueue(
        &self,
        mut command: Value,
        artifact: Option<Bytes>,
    ) -> Result<(PendingCall, oneshot::Receiver<PendingReply>), NativeCallError> {
        if self.state.failed.load(Ordering::Acquire) {
            return Err(NativeCallError::NotSent(Status::unavailable(
                "native encoder connection failed",
            )));
        }
        let sequence = self.state.sequence.fetch_add(1, Ordering::Relaxed);
        let fields = command.as_object_mut().ok_or_else(|| {
            NativeCallError::NotSent(Status::internal("native encoder command must be an object"))
        })?;
        fields.insert("sequence".into(), sequence.into());
        let header = serde_json::to_vec(&command)
            .map_err(wire_error)
            .map_err(NativeCallError::NotSent)?;
        let mut message = ZmqMessage::from(Bytes::from(header));
        if let Some(artifact) = artifact {
            message.push_back(artifact);
        }
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.state.pending.lock().map_err(|_| {
                NativeCallError::NotSent(Status::internal("native reply ledger poisoned"))
            })?;
            if self.state.failed.load(Ordering::Acquire) {
                return Err(NativeCallError::NotSent(Status::unavailable(
                    "native encoder connection failed",
                )));
            }
            pending.insert(sequence, tx);
        }
        let pending = PendingCall {
            state: self.state.clone(),
            sequence,
        };
        if self.commands.send((sequence, message)).await.is_err() {
            self.state
                .fail(Status::unavailable("native encoder command stream closed"));
            return Err(NativeCallError::NotSent(Status::unavailable(
                "native encoder command stream closed",
            )));
        }
        Ok((pending, rx))
    }

    /// Send one JSON command (and optional binary artifact frame) and await its sequenced reply.
    pub async fn call(&self, command: Value, artifact: Option<Bytes>) -> Reply {
        let (_pending, rx) = self
            .enqueue(command, artifact)
            .await
            .map_err(NativeCallError::into_status)?;
        rx.await
            .map_err(|_| Status::unavailable("native encoder reply stream closed"))?
            .map_err(NativeCallError::into_status)
    }

    /// Return whether Python has connected its IPC sockets and the link has no fatal error.
    pub fn ready(&self) -> bool {
        !self.state.failed.load(Ordering::Acquire) && self.state.connected.load(Ordering::Acquire)
    }

    /// Best-effort notification to wake early HTTP facade waiters when an encode fails before `execute`.
    pub fn notify_forget(&self, rid: &str) {
        if rid.is_empty()
            || rid.len() > 256
            || self.state.failed.load(Ordering::Acquire)
            || !self.state.connected.load(Ordering::Acquire)
        {
            return;
        }
        if let Ok(header) = serde_json::to_vec(&serde_json::json!({
            "operation": "forget",
            "rid": rid,
        })) {
            let _ = self
                .commands
                .try_send((0, ZmqMessage::from(Bytes::from(header))));
        }
    }
}

impl NativeState {
    /// Decode one sequenced native reply frame and wake its pending caller.
    fn receive(&self, message: ZmqMessage) -> Result<(), Status> {
        let frames = message.into_vec();
        if frames.len() != 1 || frames[0].len() > 4 * 1024 * 1024 {
            return Err(Status::data_loss("invalid native encoder reply frames"));
        }
        let reply: NativeReply = serde_json::from_slice(&frames[0]).map_err(wire_error)?;
        if reply.ready {
            self.connected.store(true, Ordering::Release);
            return Ok(());
        }
        let sender = self
            .pending
            .lock()
            .map_err(|_| Status::internal("native reply ledger poisoned"))?
            .remove(&reply.sequence);
        if let Some(sender) = sender {
            let result = reply.error.map_or(Ok(reply.result), |error| {
                Err(NativeCallError::Ambiguous(Status::new(
                    tonic::Code::from_i32(reply.error_code.unwrap_or(13)),
                    error,
                )))
            });
            let _ = sender.send(result);
        }
        Ok(())
    }

    /// Mark the IPC link unhealthy and fail every in-flight caller with the transport error.
    fn fail(&self, error: Status) {
        self.failed.store(true, Ordering::Release);
        if let Ok(mut pending) = self.pending.lock() {
            for (_, sender) in pending.drain() {
                let _ = sender.send(Err(NativeCallError::Ambiguous(error.clone())));
            }
        }
    }
}

/// Map low-level IPC or framing failures to gRPC `UNAVAILABLE`.
fn wire_error(error: impl std::fmt::Display) -> Status {
    Status::unavailable(error.to_string())
}

pub struct EncoderArtifact {
    pub bytes: Bytes,
    pub embedding_tokens: u64,
}

/// One image URL plus optional per-item preprocessing overrides forwarded from the gateway.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderMediaItem {
    pub url: String,
    pub detail: Option<String>,
    pub max_long_side_pixel: Option<u32>,
}

pub type EncoderProcessor = Arc<
    dyn Fn(Vec<EncoderMediaItem>) -> BoxFuture<'static, Result<EncoderArtifact, Status>>
        + Send
        + Sync,
>;

#[derive(Clone, Deserialize)]
pub struct EncoderConfig {
    pub max_active: usize,
    pub max_cpu: usize,
    pub execution_capacity: usize,
    pub max_images: usize,
    pub max_control_bytes: usize,
    #[serde(default)]
    pub control_listen: String,
    #[serde(default)]
    pub native_commands: String,
    #[serde(default)]
    pub native_events: String,
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub tokenizer_path: String,
    #[serde(default)]
    pub served_model_name: String,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub model_revision: String,
    #[serde(default)]
    pub processor_contract: String,
    #[serde(default)]
    pub embedding_dtype: String,
    #[serde(default)]
    pub kv_backend: String,
    #[serde(default)]
    pub embedding_backend: String,
    #[serde(default)]
    pub topology_contract: String,
    #[serde(default)]
    pub receive_endpoint: String,
    #[serde(default)]
    pub tp_size: u32,
}

struct EncodeOwner {
    deadline: Instant,
    cancel: watch::Sender<bool>,
    result: watch::Sender<Option<Result<(), Status>>>,
    running: AtomicBool,
}

/// Rust owns E admission, CPU preprocessing, and execution-slot accounting.
/// Execution permits return only on an explicit native release receipt.
pub struct EncoderRuntime {
    config: EncoderConfig,
    native: NativeEncoderLink,
    processor: EncoderProcessor,
    owners: Mutex<HashMap<String, Arc<EncodeOwner>>>,
    execution: Arc<Semaphore>,
    cpu: Arc<Semaphore>,
    draining: AtomicBool,
    fatal: Mutex<Option<String>>,
    cancel_grace: Duration,
}

impl EncoderRuntime {
    /// Validate admission/CPU/execution bounds and create the runtime state.
    pub fn new(
        config: EncoderConfig,
        native: NativeEncoderLink,
        processor: EncoderProcessor,
    ) -> Result<Arc<Self>, Status> {
        Self::with_cancel_grace(config, native, processor, CANCEL_RECEIPT_GRACE)
    }

    /// Create the runtime state with an explicit post-cancel receipt timeout (used by tests).
    pub(crate) fn with_cancel_grace(
        config: EncoderConfig,
        native: NativeEncoderLink,
        processor: EncoderProcessor,
        cancel_grace: Duration,
    ) -> Result<Arc<Self>, Status> {
        if config.max_active == 0
            || config.max_cpu == 0
            || config.execution_capacity == 0
            || config.execution_capacity > config.max_active
            || config.max_cpu > Semaphore::MAX_PERMITS
            || config.execution_capacity > Semaphore::MAX_PERMITS
        {
            return Err(Status::invalid_argument("invalid encoder control limits"));
        }
        Ok(Arc::new(Self {
            execution: Arc::new(Semaphore::new(config.execution_capacity)),
            cpu: Arc::new(Semaphore::new(config.max_cpu)),
            config,
            native,
            processor,
            owners: Mutex::new(HashMap::new()),
            draining: AtomicBool::new(false),
            fatal: Mutex::new(None),
            cancel_grace,
        }))
    }

    /// Return any fatal unconfirmed-release error that requires supervised shutdown.
    pub fn fatal_error(&self) -> Option<String> {
        self.fatal.lock().ok().and_then(|guard| guard.clone())
    }

    /// Return whether the encoder accepts new requests and its native IPC link is healthy.
    pub fn ready(&self) -> bool {
        !self.draining.load(Ordering::Acquire) && self.native.ready()
    }

    /// Count `(running, waiting)` admitted encode operations for load reporting.
    pub fn loads(&self) -> (usize, usize) {
        self.owners.lock().map_or((0, 0), |owners| {
            owners.values().fold((0, 0), |(running, waiting), owner| {
                if owner.running.load(Ordering::Acquire) {
                    (running + 1, waiting)
                } else {
                    (running, waiting + 1)
                }
            })
        })
    }

    /// Admit an encode RPC, spawn its independent owner task, and cancel on observer drop.
    pub async fn encode(
        self: &Arc<Self>,
        request: EncodeRequest,
        remaining: Duration,
    ) -> Result<(), Status> {
        if !self.ready() {
            self.forget_unowned(&request.req_id);
            return Err(Status::unavailable("encoder is not ready"));
        }
        let owner = match self.admit(&request, remaining) {
            Ok(owner) => owner,
            Err(status) => {
                self.forget_unowned(&request.req_id);
                return Err(status);
            }
        };
        let runtime = self.clone();
        let execution_owner = owner.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "registry owns preprocessing/native work independently of RPC observers"
        )]
        tokio::spawn(async move {
            runtime.execute(request, execution_owner).await;
        });
        let _observer = CancelOnDrop(owner.cancel.clone());
        let mut result = owner.result.subscribe();
        loop {
            if let Some(result) = result.borrow().clone() {
                return result;
            }
            result
                .changed()
                .await
                .map_err(|_| Status::unavailable("encoder owner lost"))?;
        }
    }

    /// Validate request bounds/deadline and reserve an active ledger entry.
    fn admit(
        &self,
        request: &EncodeRequest,
        remaining: Duration,
    ) -> Result<Arc<EncodeOwner>, Status> {
        if remaining.is_zero()
            || remaining > Duration::from_secs(122)
            || request.req_id.is_empty()
            || request.req_id.len() > 256
            || request.num_parts != 1
            || request.part_idx != 0
            || request.mm_items.is_empty()
            || request.mm_items.len() > self.config.max_images
            || (!request.item_options.is_empty()
                && (request.item_options.len() != request.mm_items.len()
                    || request
                        .item_options
                        .iter()
                        .any(|opt| !is_valid_detail(&opt.detail))))
            || request.encoded_len() > self.config.max_control_bytes
            || request.prefill_host.is_empty()
            || request.prefill_host.len() > 256
        {
            return Err(Status::invalid_argument("invalid encoder request/deadline"));
        }
        let owner = Arc::new(EncodeOwner {
            deadline: Instant::now() + remaining,
            cancel: watch::channel(false).0,
            result: watch::channel(None).0,
            running: AtomicBool::new(false),
        });
        let mut owners = self
            .owners
            .lock()
            .map_err(|_| Status::internal("encoder ledger poisoned"))?;
        if owners.contains_key(&request.req_id) {
            return Err(Status::already_exists("duplicate native request"));
        }
        if self.draining.load(Ordering::Acquire) || owners.len() >= self.config.max_active {
            return Err(Status::resource_exhausted("encoder admission full"));
        }
        owners.insert(request.req_id.clone(), owner.clone());
        Ok(owner)
    }

    /// Remove a finished or safely released operation from the active ledger.
    fn release_owner(&self, rid: &str) {
        if let Ok(mut owners) = self.owners.lock() {
            owners.remove(rid);
        }
    }

    /// Wake P's early facade waiters for a rid that no live owner holds.
    fn forget_unowned(&self, rid: &str) {
        if self
            .owners
            .lock()
            .is_ok_and(|owners| !owners.contains_key(rid))
        {
            self.native.notify_forget(rid);
        }
    }

    /// Run bounded CPU preprocessing, dispatch native execution, and retain permits until `released=true`.
    async fn execute(self: Arc<Self>, request: EncodeRequest, owner: Arc<EncodeOwner>) {
        let mut cancelled = owner.cancel.subscribe();
        let rid = request.req_id;
        let image_count = request.mm_items.len();
        let permit = tokio::select! {
            permit = self.execution.clone().acquire_owned() => permit.ok(),
            () = wait_cancel(&mut cancelled, owner.deadline) => None,
        };
        let prepared = async {
            if permit.is_none() {
                return Err(Status::cancelled("encoder cancelled before execution"));
            }
            let cpu = tokio::select! {
                permit = self.cpu.clone().acquire_owned() => permit
                    .map_err(|_| Status::unavailable("encoder CPU supply closed"))?,
                () = wait_cancel(&mut cancelled, owner.deadline) => {
                    return Err(Status::cancelled("encoder cancelled while waiting for CPU"));
                }
            };
            owner.running.store(true, Ordering::Release);
            if *owner.cancel.borrow() || Instant::now() >= owner.deadline {
                return Err(Status::cancelled("encoder cancelled before preprocessing"));
            }
            let options = request.item_options;
            let items = request
                .mm_items
                .into_iter()
                .enumerate()
                .map(|(index, url)| {
                    let option = options.get(index);
                    EncoderMediaItem {
                        url,
                        detail: option
                            .and_then(|o| (!o.detail.is_empty()).then(|| o.detail.clone())),
                        max_long_side_pixel: option
                            .map(|o| o.max_long_side_pixel)
                            .filter(|&cap| cap > 0),
                    }
                })
                .collect();
            let artifact = (self.processor)(items).await;
            drop(cpu);
            let artifact = artifact?;
            let remaining_ms = owner
                .deadline
                .saturating_duration_since(Instant::now())
                .as_millis() as u64;
            if *owner.cancel.borrow() || remaining_ms == 0 {
                return Err(Status::cancelled("encoder expired after preprocessing"));
            }
            Ok((artifact, remaining_ms))
        }
        .await;
        let (artifact, remaining_ms) = match prepared {
            Ok(ready) => ready,
            Err(error) => {
                self.release_owner(&rid);
                drop(permit);
                self.native.notify_forget(&rid);
                owner.result.send_replace(Some(Err(error)));
                return;
            }
        };
        // Enqueue `execute` on the FIFO command channel before racing `cancel`, so Python
        // always observes `execute` before any subsequent `cancel` for the same `rid`.
        let receipt: PendingReply = match self
            .native
            .enqueue(
                serde_json::json!({
                    "operation": "execute",
                    "rid": rid,
                    "prefill_host": request.prefill_host,
                    "remaining_ms": remaining_ms,
                    "image_count": image_count,
                    "embedding_tokens": artifact.embedding_tokens,
                }),
                Some(artifact.bytes),
            )
            .await
        {
            Ok((_pending, rx)) => {
                let exec = async move {
                    rx.await.map_err(|_| {
                        NativeCallError::Ambiguous(Status::unavailable(
                            "native encoder reply stream closed",
                        ))
                    })?
                };
                tokio::pin!(exec);
                tokio::select! {
                    receipt = &mut exec => receipt,
                    () = wait_cancel(&mut cancelled, owner.deadline) => {
                        let cancel = self.native.call(
                            serde_json::json!({"operation": "cancel", "rid": rid}),
                            None,
                        );
                        tokio::time::timeout(self.cancel_grace, async {
                            tokio::pin!(cancel);
                            tokio::select! {
                                receipt = &mut exec => receipt,
                                _ = &mut cancel => (&mut exec).await,
                            }
                        })
                        .await
                        .unwrap_or_else(|_| {
                            Err(NativeCallError::Ambiguous(Status::unavailable(
                                "native encoder release receipt timed out",
                            )))
                        })
                    }
                }
            }
            Err(error) => Err(error),
        };
        if let Err(NativeCallError::NotSent(error)) = receipt {
            self.release_owner(&rid);
            drop(permit);
            owner.result.send_replace(Some(Err(error)));
            return;
        }
        let safe = receipt
            .as_ref()
            .is_ok_and(|receipt| receipt.get("released").and_then(Value::as_bool) == Some(true));
        let outcome = match receipt {
            Ok(receipt) if safe => receipt
                .get("error")
                .and_then(Value::as_str)
                .map_or(Ok(()), |error| Err(Status::internal(error.to_owned()))),
            Ok(_) => Err(Status::unavailable(
                "native encoder did not confirm safe release",
            )),
            Err(error) => Err(error.into_status()),
        };
        if safe {
            self.release_owner(&rid);
            drop(permit);
        } else {
            self.draining.store(true, Ordering::Release);
            if let Ok(mut fatal) = self.fatal.lock() {
                fatal.get_or_insert_with(|| {
                    outcome.as_ref().err().map_or_else(
                        || "native encoder did not confirm safe release".into(),
                        |error| error.message().to_owned(),
                    )
                });
            }
            if let Some(permit) = permit {
                permit.forget();
            }
        }
        owner.result.send_replace(Some(outcome));
    }

    /// Stop admitting new requests and signal cancellation to every active owner.
    pub fn drain(&self) {
        self.draining.store(true, Ordering::Release);
        if let Ok(owners) = self.owners.lock() {
            for owner in owners.values() {
                owner.cancel.send_replace(true);
            }
        }
    }
}

struct CancelOnDrop(watch::Sender<bool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.send_replace(true);
    }
}

/// Accept empty or OpenAI-style detail hints case-insensitively.
fn is_valid_detail(detail: &str) -> bool {
    detail.is_empty()
        || detail.eq_ignore_ascii_case("auto")
        || detail.eq_ignore_ascii_case("low")
        || detail.eq_ignore_ascii_case("high")
}

/// Resolve when cancellation is signalled or the request deadline elapses.
async fn wait_cancel(cancel: &mut watch::Receiver<bool>, deadline: Instant) {
    if *cancel.borrow_and_update() {
        return;
    }
    tokio::select! { _ = cancel.changed() => {}, () = tokio::time::sleep_until(deadline) => {} }
}

#[derive(Clone)]
struct EncoderService {
    runtime: Arc<EncoderRuntime>,
}

impl EncoderService {
    /// Snapshot running, waiting, and total active encode counts as a scheduler load record.
    fn load(&self) -> sg::SchedulerLoad {
        let (running, waiting) = self.runtime.loads();
        sg::SchedulerLoad {
            num_running_reqs: running as i32,
            num_waiting_reqs: waiting as i32,
            num_total_reqs: (running + waiting) as i32,
            max_running_requests: self.runtime.config.max_active as i32,
            ..Default::default()
        }
    }

    /// Build the advertised prepared-input capability contract for worker registration.
    fn capability(&self) -> Capabilities {
        let c = &self.runtime.config;
        Capabilities {
            version: VERSION,
            role: "encoder".into(),
            model_revision: c.model_revision.clone(),
            processor_contract: c.processor_contract.clone(),
            layout_contract: LAYOUT.into(),
            embedding_dtype: c.embedding_dtype.clone(),
            kv_backend: c.kv_backend.clone(),
            embedding_backend: c.embedding_backend.clone(),
            topology_contract: c.topology_contract.clone(),
            receive_endpoint: c.receive_endpoint.clone(),
            tp_size: c.tp_size,
            ready: self.runtime.ready(),
            ..Default::default()
        }
    }
}

#[tonic::async_trait]
impl ep::sglang_encoder_server::SglangEncoder for EncoderService {
    async fn encode(
        &self,
        request: Request<EncodeRequest>,
    ) -> Result<Response<ep::EncodeResponse>, Status> {
        let remaining = crate::grpc_timeout(request.metadata())
            .ok_or_else(|| Status::invalid_argument("Encode requires a valid bounded deadline"))?;
        self.runtime.encode(request.into_inner(), remaining).await?;
        Ok(Response::new(ep::EncodeResponse::default()))
    }

    async fn send(
        &self,
        _: Request<ep::SendRequest>,
    ) -> Result<Response<ep::SendResponse>, Status> {
        Err(Status::unimplemented(
            "native receivers use the bound HTTP facade",
        ))
    }

    async fn scheduler_receive_url(
        &self,
        _: Request<ep::SchedulerReceiveUrlRequest>,
    ) -> Result<Response<ep::SchedulerReceiveUrlResponse>, Status> {
        Err(Status::unimplemented(
            "native receivers use the bound HTTP facade",
        ))
    }
}

impl crate::load_stream::RankedLoads for EncoderService {
    fn rank_count(&self) -> usize {
        1
    }

    fn rank_record(&self, rank: i32) -> Option<common::EngineLoad> {
        if rank != 0 {
            return None;
        }
        let load = self.load();
        Some(common::EngineLoad {
            running_requests: load.num_running_reqs as u32,
            waiting_requests: load.num_waiting_reqs as u32,
            max_running_requests: load.max_running_requests as u32,
            ..Default::default()
        })
    }
}

macro_rules! unsupported_encoder_rpcs {
    ($($method:ident($request:ty) -> $response:ty;)+) => {
        $(
            fn $method<'service, 'rpc>(
                &'service self,
                _: Request<$request>,
            ) -> futures::future::BoxFuture<'rpc, Result<Response<$response>, Status>>
            where
                'service: 'rpc,
                Self: 'rpc,
            {
                Box::pin(std::future::ready(Err(Status::unimplemented("not available on the encoder"))))
            }
        )+
    };
}

#[tonic::async_trait]
impl sg::sglang_scheduler_server::SglangScheduler for EncoderService {
    type GenerateStream = BoxStream<sg::GenerateResponse>;
    type GetTokenizerStream = BoxStream<common::GetTokenizerChunk>;
    type SubscribeKvEventsStream = BoxStream<common::KvEventBatch>;

    async fn health_check(
        &self,
        _: Request<sg::HealthCheckRequest>,
    ) -> Result<Response<sg::HealthCheckResponse>, Status> {
        Ok(Response::new(sg::HealthCheckResponse {
            healthy: self.runtime.ready(),
            ..Default::default()
        }))
    }

    async fn get_model_info(
        &self,
        _: Request<sg::GetModelInfoRequest>,
    ) -> Result<Response<sg::GetModelInfoResponse>, Status> {
        let c = &self.runtime.config;
        Ok(Response::new(sg::GetModelInfoResponse {
            model_path: c.model_path.clone(),
            tokenizer_path: c.tokenizer_path.clone(),
            served_model_name: c.served_model_name.clone(),
            model_type: c.model_type.clone(),
            is_generation: true,
            supports_vision: true,
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _: Request<sg::GetServerInfoRequest>,
    ) -> Result<Response<sg::GetServerInfoResponse>, Status> {
        let c = &self.runtime.config;
        let args = serde_json::json!({
            "model_path": c.model_path,
            "served_model_name": c.served_model_name,
            "supports_vision": true,
            "mm_processor": "smg",
            "mm_media_ref_schemes": "http,https,data",
            "sg_computation_role": "encoder",
            "sglang_prepared": serde_json::to_string(&self.capability())
                .map_err(|error| Status::internal(error.to_string()))?,
            "max_running_requests": c.max_active,
        });
        Ok(Response::new(sg::GetServerInfoResponse {
            server_args: Some(crate::proto_json::struct_from_json(&args.to_string())),
            active_requests: self.load().num_total_reqs,
            ..Default::default()
        }))
    }

    async fn get_loads(
        &self,
        _: Request<sg::GetLoadsRequest>,
    ) -> Result<Response<sg::GetLoadsResponse>, Status> {
        let load = self.load();
        Ok(Response::new(sg::GetLoadsResponse {
            dp_rank_count: 1,
            aggregate: Some(sg::AggregateMetrics {
                total_running_reqs: load.num_running_reqs,
                total_waiting_reqs: load.num_waiting_reqs,
                total_reqs: load.num_total_reqs,
                ..Default::default()
            }),
            loads: vec![load],
            ..Default::default()
        }))
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        if !request.into_inner().loads_only {
            return Err(Status::unimplemented("encoder has no KV cache events"));
        }
        Ok(Response::new(crate::load_stream::load_records(
            Arc::new(self.clone()),
            Default::default(),
        )))
    }

    unsupported_encoder_rpcs! {
        generate(sg::GenerateRequest) -> Self::GenerateStream;
        embed(sg::EmbedRequest) -> sg::EmbedResponse;
        abort(sg::AbortRequest) -> sg::AbortResponse;
        flush_cache(common::FlushCacheRequest) -> common::FlushCacheResponse;
        start_profile(common::StartProfileRequest) -> common::ProfileResponse;
        stop_profile(common::StopProfileRequest) -> common::ProfileResponse;
        get_tokenizer(common::GetTokenizerRequest) -> Self::GetTokenizerStream;
        load_lo_ra_adapter(sg::LoadLoRaAdapterRequest) -> sg::LoadLoRaAdapterResponse;
        unload_lo_ra_adapter(sg::UnloadLoRaAdapterRequest) -> sg::UnloadLoRaAdapterResponse;
        list_loaded_lo_ra_adapters(sg::ListLoadedLoRaAdaptersRequest) -> sg::ListLoadedLoRaAdaptersResponse;
    }
}

pub struct EncoderServer {
    thread: ServerThread,
    runtime: Arc<EncoderRuntime>,
}

impl EncoderServer {
    /// Parse encoder configuration and start the gRPC control server on a dedicated runtime thread.
    pub fn start(config_json: &str, processor: EncoderProcessor) -> Result<Self, ServicerError> {
        let config: EncoderConfig = serde_json::from_str(config_json)
            .map_err(|error| ServicerError::InvalidConfig(error.to_string()))?;
        if config.control_listen.is_empty() {
            return Err(ServicerError::InvalidConfig(
                "missing encoder control listener".into(),
            ));
        }
        let address = config.control_listen.clone();
        let (ready, state) = sync_channel(1);
        let thread = ServerThread::start(
            "sglang-encoder",
            &address,
            move |listener, shutdown, _| async move {
                let link = NativeEncoderLink::bind(
                    &config.native_commands,
                    &config.native_events,
                    config.max_active * 16,
                )
                .await
                .map_err(|error| error.to_string())?;
                let max_control_bytes = config.max_control_bytes;
                let runtime = EncoderRuntime::new(config, link, processor)
                    .map_err(|error| error.to_string())?;
                let service = EncoderService {
                    runtime: runtime.clone(),
                };
                ready
                    .send(runtime.clone())
                    .map_err(|error| error.to_string())?;
                let draining = shutdown.clone().map(move |_| runtime.drain()).shared();
                let grpc = tonic::transport::Server::builder()
                    .add_service(
                        ep::sglang_encoder_server::SglangEncoderServer::new(service.clone())
                            .max_decoding_message_size(max_control_bytes),
                    )
                    .add_service(sg::sglang_scheduler_server::SglangSchedulerServer::new(
                        service,
                    ))
                    .serve_with_incoming_shutdown(
                        tokio_stream::wrappers::TcpListenerStream::new(listener),
                        draining,
                    )
                    .map_err(|error| error.to_string());
                crate::servicer::serve_with_grace(grpc, shutdown).await
            },
        )?;
        let runtime = state
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| ServicerError::Startup(error.to_string()))?;
        Ok(Self { thread, runtime })
    }

    /// Mark the encoder draining and cancel active operations before shutdown.
    pub fn drain(&self) {
        self.runtime.drain();
    }

    /// Return the total count of running and waiting encode operations.
    pub fn active(&self) -> usize {
        let (running, waiting) = self.runtime.loads();
        running + waiting
    }

    /// Return whether the gRPC control server thread is still running.
    pub fn running(&self) -> bool {
        self.thread.running()
    }

    /// Return any terminal error captured by the control server thread or fatal release quarantine.
    pub fn last_error(&self) -> Result<Option<String>, ServicerError> {
        if let Some(error) = self.thread.last_error()? {
            return Ok(Some(error));
        }
        Ok(self.runtime.fatal_error())
    }

    /// Gracefully stop the gRPC control server thread within `timeout`.
    pub fn stop(&self, timeout: Duration) -> Result<(), ServicerError> {
        self.drain();
        self.thread.stop(timeout)
    }
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "unit tests spawn concurrent encoder tasks"
)]
mod tests {
    use serde_json::json;

    use super::*;

    fn test_config() -> EncoderConfig {
        EncoderConfig {
            max_active: 4,
            max_cpu: 2,
            execution_capacity: 1,
            max_images: 4,
            max_control_bytes: 1024,
            control_listen: String::new(),
            native_commands: String::new(),
            native_events: String::new(),
            model_path: "model".into(),
            tokenizer_path: "tok".into(),
            served_model_name: "model".into(),
            model_type: "qwen3_vl".into(),
            model_revision: "rev1".into(),
            processor_contract: "smg-image-grid-v1".into(),
            embedding_dtype: "bfloat16".into(),
            kv_backend: "mooncake".into(),
            embedding_backend: "mooncake".into(),
            topology_contract: "single-host-tp1".into(),
            receive_endpoint: "http://127.0.0.1:9000".into(),
            tp_size: 1,
        }
    }

    fn encode_request(id: &str) -> EncodeRequest {
        EncodeRequest {
            req_id: id.into(),
            mm_items: vec!["data:image/png;base64,AA==".into()],
            num_parts: 1,
            part_idx: 0,
            prefill_host: "127.0.0.1".into(),
            embedding_port: vec![],
            item_options: vec![],
        }
    }

    async fn recv_cmd(pull: &mut PullSocket) -> Value {
        let message = tokio::time::timeout(Duration::from_secs(2), pull.recv())
            .await
            .expect("timed out waiting for encoder IPC command")
            .unwrap();
        serde_json::from_slice(&message.into_vec()[0]).unwrap()
    }

    struct EncoderHarness {
        _dir: tempfile::TempDir,
        runtime: Arc<EncoderRuntime>,
        pull: Option<PullSocket>,
        push: PushSocket,
    }

    impl EncoderHarness {
        async fn start(
            connect_pull: bool,
            cancel_grace: Duration,
            processor: Option<EncoderProcessor>,
        ) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let commands = format!("ipc://{}/commands.sock", dir.path().display());
            let events = format!("ipc://{}/events.sock", dir.path().display());
            let link = NativeEncoderLink::bind(&commands, &events, 32)
                .await
                .unwrap();
            let pull = if connect_pull {
                let mut pull = PullSocket::new();
                pull.connect(&commands).await.unwrap();
                Some(pull)
            } else {
                None
            };
            let mut push = PushSocket::new();
            push.connect(&events).await.unwrap();
            let processor = processor.unwrap_or_else(|| {
                Arc::new(|_| {
                    Box::pin(async {
                        Ok(EncoderArtifact {
                            bytes: Bytes::from_static(b"pixels"),
                            embedding_tokens: 4,
                        })
                    })
                })
            });
            let runtime =
                EncoderRuntime::with_cancel_grace(test_config(), link, processor, cancel_grace)
                    .unwrap();
            assert!(!runtime.ready());
            Self {
                _dir: dir,
                runtime,
                pull,
                push,
            }
        }

        async fn mark_ready(&mut self) {
            let ready_msg = serde_json::to_vec(&json!({"ready": true})).unwrap();
            self.push
                .send(ZmqMessage::from(Bytes::from(ready_msg)))
                .await
                .unwrap();
            for _ in 0..50 {
                if self.runtime.ready() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            assert!(self.runtime.ready());
        }
    }

    #[tokio::test]
    async fn cancel_releases_queued_immediately_and_inflight_only_after_receipt() {
        let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, None).await;
        let early = h
            .runtime
            .encode(encode_request("early-req"), Duration::from_secs(5))
            .await
            .unwrap_err();
        assert_eq!(early.code(), tonic::Code::Unavailable);
        assert!(h.runtime.fatal_error().is_none());
        h.mark_ready().await;

        let h1 = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("req-1"), Duration::from_secs(5))
                    .await
            }
        });
        let pull = h.pull.as_mut().unwrap();
        let exec1 = recv_cmd(pull).await;
        assert_eq!(exec1["operation"], "execute");

        // Queued request cancelled before execution releases immediately and notifies Python.
        let h2 = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("req-2"), Duration::from_secs(5))
                    .await
            }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(h.runtime.loads(), (1, 1));
        h2.abort();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(h.runtime.loads(), (1, 0));
        let forget2 = recv_cmd(pull).await;
        assert_eq!(forget2["operation"], "forget");
        assert_eq!(forget2["rid"], "req-2");

        // Cancelling in-flight req-1 sends cancel wakeup but retains slot until released=true.
        h1.abort();
        let cancel1 = recv_cmd(pull).await;
        assert_eq!(cancel1["operation"], "cancel");
        assert_eq!(cancel1["rid"], "req-1");
        let reply_c =
            serde_json::to_vec(&json!({"sequence": cancel1["sequence"], "result": {}})).unwrap();
        h.push
            .send(ZmqMessage::from(Bytes::from(reply_c)))
            .await
            .unwrap();
        assert_eq!(h.runtime.loads().0, 1);

        // Unconfirmed release drains runtime and retains the owner.
        let reply_e = serde_json::to_vec(
            &json!({"sequence": exec1["sequence"], "result": {"released": false}}),
        )
        .unwrap();
        h.push
            .send(ZmqMessage::from(Bytes::from(reply_e)))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!h.runtime.ready());
        assert!(h.runtime.fatal_error().is_some());
        assert_eq!(h.runtime.loads().0, 1);
    }

    #[tokio::test]
    async fn unsent_command_releases_slot_without_quarantine() {
        // Connect ONLY the events socket so PushSocket::send returns ReturnToSender through retry.
        let mut h = EncoderHarness::start(false, CANCEL_RECEIPT_GRACE, None).await;
        h.mark_ready().await;

        let err = h
            .runtime
            .encode(encode_request("unsent-req"), Duration::from_secs(5))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::Unavailable);
        assert!(
            h.runtime.ready(),
            "unsent command must not quarantine encoder"
        );
        assert!(h.runtime.fatal_error().is_none());
        assert_eq!(h.runtime.loads(), (0, 0));
    }

    #[tokio::test]
    async fn cancel_receipt_timeout_marks_fatal_and_retains_slot() {
        let mut h = EncoderHarness::start(true, Duration::from_millis(80), None).await;
        h.mark_ready().await;

        let handle = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("stuck-req"), Duration::from_secs(5))
                    .await
            }
        });
        let pull = h.pull.as_mut().unwrap();
        let exec = recv_cmd(pull).await;
        assert_eq!(exec["operation"], "execute");

        // Leave BOTH cancel and execute unacknowledged to verify the timeout covers wedged cancel.
        handle.abort();
        let cancel = recv_cmd(pull).await;
        assert_eq!(cancel["operation"], "cancel");

        for _ in 0..50 {
            if h.runtime.fatal_error().is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!h.runtime.ready());
        assert!(h
            .runtime
            .fatal_error()
            .as_deref()
            .is_some_and(|msg| msg.contains("timed out")));
        assert_eq!(h.runtime.loads().0, 1);
    }

    #[tokio::test]
    async fn forget_unowned_protects_live_owner_and_notifies_while_draining() {
        let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, None).await;
        h.mark_ready().await;

        let _live = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("req-live"), Duration::from_secs(5))
                    .await
            }
        });
        let pull = h.pull.as_mut().unwrap();
        let exec = recv_cmd(pull).await;
        assert_eq!(exec["operation"], "execute");
        assert_eq!(exec["rid"], "req-live");

        // Malformed request reusing a live rid fails InvalidArgument but must NOT emit forget.
        let mut malformed = encode_request("req-live");
        malformed.mm_items.clear();
        let err = h
            .runtime
            .encode(malformed, Duration::from_secs(5))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(
            tokio::time::timeout(Duration::from_millis(80), pull.recv())
                .await
                .is_err(),
            "malformed duplicate rid must not emit forget while owner is live"
        );

        // Draining rejections on an unowned rid emit forget while IPC is still connected.
        h.runtime.drain();
        let err = h
            .runtime
            .encode(encode_request("req-draining"), Duration::from_secs(5))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::Unavailable);
        let mut saw_forget = false;
        for _ in 0..2 {
            let cmd = recv_cmd(pull).await;
            if cmd["operation"] == "forget" && cmd["rid"] == "req-draining" {
                saw_forget = true;
                break;
            }
        }
        assert!(saw_forget, "expected forget notification while draining");
    }

    #[tokio::test]
    async fn item_options_forward_to_processor_and_reject_mismatches() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_proc = seen.clone();
        let processor: EncoderProcessor = Arc::new(move |items| {
            seen_proc.lock().unwrap().push(items);
            Box::pin(async {
                Ok(EncoderArtifact {
                    bytes: Bytes::from_static(b"pixels"),
                    embedding_tokens: 4,
                })
            })
        });
        let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, Some(processor)).await;
        h.mark_ready().await;

        let mut mismatch = encode_request("bad-len");
        mismatch.item_options.push(ep::ImageOption {
            detail: "high".into(),
            max_long_side_pixel: 504,
        });
        mismatch.item_options.push(ep::ImageOption::default());
        assert_eq!(
            h.runtime
                .encode(mismatch, Duration::from_secs(5))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );

        let mut bad_detail = encode_request("bad-detail");
        bad_detail.item_options.push(ep::ImageOption {
            detail: "bogus".into(),
            max_long_side_pixel: 0,
        });
        assert_eq!(
            h.runtime
                .encode(bad_detail, Duration::from_secs(5))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        assert!(seen.lock().unwrap().is_empty());
        let pull = h.pull.as_mut().unwrap();
        for expected_rid in ["bad-len", "bad-detail"] {
            let forget = recv_cmd(pull).await;
            assert_eq!(forget["operation"], "forget");
            assert_eq!(forget["rid"], expected_rid);
        }

        let req = EncodeRequest {
            req_id: "opts-ok".into(),
            mm_items: vec![
                "https://example.test/a.png".into(),
                "https://example.test/b.png".into(),
            ],
            num_parts: 1,
            part_idx: 0,
            prefill_host: "127.0.0.1".into(),
            embedding_port: vec![],
            item_options: vec![
                ep::ImageOption {
                    detail: "high".into(),
                    max_long_side_pixel: 504,
                },
                ep::ImageOption {
                    detail: "auto".into(),
                    max_long_side_pixel: 0,
                },
            ],
        };
        let task = tokio::spawn({
            let rt = h.runtime.clone();
            async move { rt.encode(req, Duration::from_secs(5)).await }
        });
        let exec = recv_cmd(pull).await;
        assert_eq!(exec["operation"], "execute");
        let reply = serde_json::to_vec(
            &json!({"sequence": exec["sequence"], "result": {"released": true}}),
        )
        .unwrap();
        h.push
            .send(ZmqMessage::from(Bytes::from(reply)))
            .await
            .unwrap();
        task.await.unwrap().unwrap();
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[vec![
                EncoderMediaItem {
                    url: "https://example.test/a.png".into(),
                    detail: Some("high".into()),
                    max_long_side_pixel: Some(504),
                },
                EncoderMediaItem {
                    url: "https://example.test/b.png".into(),
                    detail: Some("auto".into()),
                    max_long_side_pixel: None,
                },
            ]]
        );
    }
}
