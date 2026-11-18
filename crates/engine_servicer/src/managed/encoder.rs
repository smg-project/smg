//! Same-host native encoder commands. A successful cancellation response is only a wakeup;
//! only an explicit released response permits the control owner to return resource charges.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::sync_channel,
        Arc, Mutex,
    },
};

use bytes::Bytes;
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tonic::Status;
use zeromq::{PullSocket, PushSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

type Reply = Result<Value, Status>;

/// Commands and completions use a private IPC directory scoped to one worker generation.
/// Transport loss fails observers but never manufactures a native release receipt.
#[derive(Clone)]
pub struct NativeEncoderLink {
    state: Arc<NativeState>,
    commands: mpsc::Sender<ZmqMessage>,
}

struct NativeState {
    generation: String,
    sequence: AtomicU64,
    pending: Mutex<HashMap<u64, oneshot::Sender<Reply>>>,
    failed: AtomicBool,
}

struct PendingCall {
    state: Arc<NativeState>,
    sequence: u64,
}
impl Drop for PendingCall {
    /// Remove only this IPC reply observer; native execution owns its own completion.
    fn drop(&mut self) {
        if let Ok(mut pending) = self.state.pending.lock() {
            pending.remove(&self.sequence);
        }
    }
}

#[derive(Deserialize)]
struct NativeReply {
    generation: String,
    sequence: u64,
    #[serde(default)]
    result: Value,
    error: Option<String>,
    error_code: Option<i32>,
}

impl NativeEncoderLink {
    /// Bind both sockets before launching the native receiver. Bound the command queue
    /// independently of the data owners; the caller supplies its finite control capacity.
    pub async fn bind(
        commands_endpoint: &str,
        events_endpoint: &str,
        generation: String,
        capacity: usize,
    ) -> Result<Self, Status> {
        if !commands_endpoint.starts_with("ipc://")
            || !events_endpoint.starts_with("ipc://")
            || generation.is_empty()
            || capacity == 0
        {
            return Err(Status::invalid_argument(
                "encoder wire requires fenced, bounded IPC",
            ));
        }
        let mut commands = PushSocket::new();
        let mut events = PullSocket::new();
        commands.bind(commands_endpoint).await.map_err(wire_error)?;
        events.bind(events_endpoint).await.map_err(wire_error)?;
        let (tx, mut rx) = mpsc::channel(capacity);
        let state = Arc::new(NativeState {
            generation,
            sequence: AtomicU64::new(1),
            pending: Mutex::default(),
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
            while let Some(message) = rx.recv().await {
                if let Err(error) = commands.send(message).await {
                    writer.fail(wire_error(error));
                    closed.send_replace(true);
                    return;
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

    /// Keep the reply registration alive even when an observer disconnects. The execution
    /// owner must await this future; dropping it is never evidence that native work stopped.
    pub async fn call(&self, mut command: Value, artifact: Option<Bytes>) -> Reply {
        if self.state.failed.load(Ordering::Acquire) {
            return Err(Status::unavailable("native encoder connection failed"));
        }
        let sequence = self.state.sequence.fetch_add(1, Ordering::Relaxed);
        let fields = command
            .as_object_mut()
            .ok_or_else(|| Status::internal("native encoder command must be an object"))?;
        fields.insert(
            "generation".into(),
            self.state.generation.to_string().into(),
        );
        fields.insert("sequence".into(), sequence.into());
        let header = serde_json::to_vec(&command).map_err(wire_error)?;
        let mut message = ZmqMessage::from(Bytes::from(header));
        if let Some(artifact) = artifact {
            message.push_back(artifact);
        }
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self
                .state
                .pending
                .lock()
                .map_err(|_| Status::internal("native reply ledger poisoned"))?;
            // Pair insertion with fail under the same mutex: no registration may land after
            // the failure sweep and wait forever without a native completion source.
            if self.state.failed.load(Ordering::Acquire) {
                return Err(Status::unavailable("native encoder connection failed"));
            }
            pending.insert(sequence, tx);
        }
        let _pending = PendingCall {
            state: self.state.clone(),
            sequence,
        };
        if self.commands.send(message).await.is_err() {
            self.state
                .fail(Status::unavailable("native encoder command stream closed"));
        }
        rx.await
            .map_err(|_| Status::unavailable("native encoder reply stream closed"))?
    }

    /// Report IPC link failure independently of native resource completion.
    pub fn ready(&self) -> bool {
        !self.state.failed.load(Ordering::Acquire)
    }
}

impl NativeState {
    /// Resolve only matching-generation reply observers; late duplicates create no owners.
    fn receive(&self, message: ZmqMessage) -> Result<(), Status> {
        let frames = message.into_vec();
        if frames.len() != 1 || frames[0].len() > 4 * 1024 * 1024 {
            return Err(Status::data_loss(
                "invalid native encoder completion frames",
            ));
        }
        let reply: NativeReply = serde_json::from_slice(&frames[0]).map_err(wire_error)?;
        if reply.generation != self.generation {
            // A previous generation's delayed release must not release current owners.
            return Err(Status::failed_precondition(
                "native encoder generation mismatch",
            ));
        }
        let sender = self
            .pending
            .lock()
            .map_err(|_| Status::internal("native reply ledger poisoned"))?
            .remove(&reply.sequence);
        if let Some(sender) = sender {
            let result = reply.error.map_or(Ok(reply.result), |error| {
                Err(Status::new(
                    tonic::Code::from_i32(reply.error_code.unwrap_or(13)),
                    error,
                ))
            });
            let _ = sender.send(result);
        }
        // Duplicate replies are harmless, but never create new request state.
        Ok(())
    }

    /// Fail pending IPC observers atomically without declaring native resources released.
    fn fail(&self, error: Status) {
        self.failed.store(true, Ordering::Release);
        if let Ok(mut pending) = self.pending.lock() {
            for (_, sender) in pending.drain() {
                let _ = sender.send(Err(error.clone()));
            }
        }
    }
}

/// Translate transport and serialization failures into an unavailable control result.
fn wire_error(error: impl std::fmt::Display) -> Status {
    Status::unavailable(error.to_string())
}

use std::time::Duration;

use futures::{future::BoxFuture, FutureExt, TryFutureExt};
use prost::Message;
use smg_grpc_client::{
    sglang_encoder_proto::EncodeRequest,
    sglang_managed::{clock, proto::Capabilities},
};
use tokio::{
    sync::{watch, Notify, Semaphore},
    time::Instant,
};

/// The preprocessing owner returns the count while the protobuf is already decoded in Rust.
/// No Python parse of the entire pixel artifact is required to check the output budget.
pub struct EncoderArtifact {
    pub bytes: Bytes,
    pub embedding_tokens: u64,
}

pub type EncoderProcessor =
    Arc<dyn Fn(Vec<String>) -> BoxFuture<'static, Result<EncoderArtifact, Status>> + Send + Sync>;

#[derive(Clone, Deserialize)]
pub struct EncoderConfig {
    pub generation: String,
    pub max_active: usize,
    pub max_records: usize,
    pub max_cpu: usize,
    pub execution_capacity: usize,
    pub max_images: usize,
    pub max_control_bytes: usize,
    pub retention_ms: u64,
    pub max_clock_error_ms: u64,
    pub max_early_waiters: usize,
}

struct EncodeOwner {
    deadline: Instant,
    retain_until: Instant,
    cancel: watch::Sender<bool>,
    registered: watch::Sender<bool>,
    result: watch::Sender<Option<Result<(), Status>>>,
    quiesced: AtomicBool,
    running: AtomicBool,
}

/// The only E admission ledger. Python retains native tensor/send references, not a
/// second admission queue. Execution permits leave this owner only on a release receipt.
pub struct EncoderRuntime {
    config: EncoderConfig,
    native: NativeEncoderLink,
    processor: EncoderProcessor,
    owners: Mutex<HashMap<String, Arc<EncodeOwner>>>,
    execution: Arc<Semaphore>,
    cpu: Arc<Semaphore>,
    early: Semaphore,
    admitted: Notify,
    draining: AtomicBool,
}

impl EncoderRuntime {
    /// Validate finite limits and create separate CPU and native execution permit pools.
    pub fn new(
        config: EncoderConfig,
        native: NativeEncoderLink,
        processor: EncoderProcessor,
    ) -> Result<Arc<Self>, Status> {
        if config.max_active == 0
            || config.max_records < config.max_active
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
            early: Semaphore::new(config.max_early_waiters),
            admitted: Notify::new(),
            config,
            native,
            processor,
            owners: Mutex::new(HashMap::new()),
            draining: AtomicBool::new(false),
        }))
    }

    /// Accept work only while admission, the native link, and the clock check are healthy.
    pub fn ready(&self) -> bool {
        !self.draining.load(Ordering::Acquire)
            && self.native.ready()
            && clock::check(self.config.max_clock_error_ms).is_ok()
    }

    /// Count all owners lacking a release receipt, including cancelled or expired work.
    pub fn loads(&self) -> (usize, usize) {
        self.owners.lock().map_or((0, 0), |owners| {
            owners
                .values()
                .filter(|owner| !owner.quiesced.load(Ordering::Acquire))
                .fold((0, 0), |(running, waiting), owner| {
                    if owner.running.load(Ordering::Acquire) {
                        (running + 1, waiting)
                    } else {
                        (running, waiting + 1)
                    }
                })
        })
    }

    /// Detach execution from its RPC observer; observer loss requests cancellation only.
    pub async fn encode(
        self: &Arc<Self>,
        request: EncodeRequest,
        remaining: Duration,
    ) -> Result<(), Status> {
        if !self.ready() {
            return Err(Status::unavailable("encoder is not ready"));
        }
        let owner = self.admit(&request, remaining)?;
        self.admitted.notify_waiters();
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

    /// Fence request identity and bounds; expire replay records only after confirmed release.
    fn admit(
        &self,
        request: &EncodeRequest,
        remaining: Duration,
    ) -> Result<Arc<EncodeOwner>, Status> {
        if remaining.is_zero()
            || remaining > Duration::from_secs(122)
            || request.req_id.len() > 256
            || !request
                .req_id
                .starts_with(&format!("{}.", self.config.generation))
            || request.num_parts != 1
            || request.part_idx != 0
            || request.mm_items.is_empty()
            || request.mm_items.len() > self.config.max_images
            || request.encoded_len() > self.config.max_control_bytes
            || request.prefill_host.is_empty()
        {
            return Err(Status::invalid_argument(
                "invalid fenced encoder request/deadline",
            ));
        }
        let now = Instant::now();
        let owner = Arc::new(EncodeOwner {
            deadline: now + remaining,
            retain_until: now + remaining + Duration::from_millis(self.config.retention_ms),
            cancel: watch::channel(false).0,
            registered: watch::channel(false).0,
            result: watch::channel(None).0,
            quiesced: AtomicBool::new(false),
            running: AtomicBool::new(false),
        });
        {
            let mut owners = self
                .owners
                .lock()
                .map_err(|_| Status::internal("encoder ledger poisoned"))?;
            owners.retain(|_, owner| {
                !owner.quiesced.load(Ordering::Acquire) || now <= owner.retain_until
            });
            if owners.contains_key(&request.req_id) {
                return Err(Status::already_exists("duplicate native request"));
            }
            if self.draining.load(Ordering::Acquire)
                || owners.len() >= self.config.max_records
                || owners
                    .values()
                    .filter(|owner| !owner.quiesced.load(Ordering::Acquire))
                    .count()
                    >= self.config.max_active
            {
                return Err(Status::resource_exhausted("encoder admission full"));
            }
            owners.insert(request.req_id.clone(), owner.clone());
        }
        Ok(owner)
    }

    /// Bound preprocessing with a CPU permit; retain native charge until an explicit release receipt.
    async fn execute(self: Arc<Self>, request: EncodeRequest, owner: Arc<EncodeOwner>) {
        let mut cancelled = owner.cancel.subscribe();
        let remaining_ms = owner
            .deadline
            .saturating_duration_since(Instant::now())
            .as_millis() as u64;
        if *owner.cancel.borrow() || remaining_ms == 0 {
            owner.quiesced.store(true, Ordering::Release);
            owner.result.send_replace(Some(Err(Status::cancelled(
                "encoder cancelled before native registration",
            ))));
            return;
        }
        // Match Python admission: create only the empty native context before waiting
        // for tensor/CPU capacity, so early P controls can bind while E is queued.
        let rid = request.req_id;
        let register = serde_json::json!({"operation":"register", "rid":rid,
            "remaining_ms": remaining_ms,
            "prefill_host":request.prefill_host});
        // A normal observer cancellation must not turn a healthy in-flight registration
        // into ambiguous transport loss. Keep its receipt, then the cancellation watch
        // closes the registered native owner through the ordinary cleanup path.
        let registered = tokio::time::timeout_at(owner.deadline, self.native.call(register, None))
            .await
            .unwrap_or_else(|_| {
                Err(Status::deadline_exceeded(
                    "native registration acknowledgement lost",
                ))
            });
        if let Err(error) = registered {
            // Submission is ambiguous. Best-effort wakeup is bounded, and is never a
            // release receipt. The native lease also wakes orphaned registered owners.
            tracing::warn!(request_id = %rid, %error, "native encoder registration unconfirmed; retaining admitted owner");
            self.draining.store(true, Ordering::Release);
            let _ = tokio::time::timeout(
                Duration::from_secs(1),
                self.native
                    .call(serde_json::json!({"operation":"cancel", "rid":rid}), None),
            )
            .await;
            owner.result.send_replace(Some(Err(error)));
            return;
        }
        owner.registered.send_replace(true);
        let (finished, mut native_finished) = watch::channel(false);
        let native = self.native.clone();
        let cancel_rid = rid.clone();
        let deadline = owner.deadline;
        #[expect(
            clippy::disallowed_methods,
            reason = "cancellation only wakes native owners; execution still awaits their completion"
        )]
        let watchdog = tokio::spawn(async move {
            tokio::select! {
                () = wait_cancel(&mut cancelled, deadline) => {
                    // A lost cancel acknowledgement must not block an independent
                    // completion receipt. The caller still requires released=true;
                    // ending this observer never proves native quiescence by itself.
                    tokio::select! {
                        result = native.call(serde_json::json!({"operation":"cancel", "rid":cancel_rid}), None) => { result?; }
                        _ = native_finished.changed() => {}
                    }
                }
                _ = native_finished.changed() => {}
            }
            Ok::<_, Status>(())
        });
        let mut queued_cancel = owner.cancel.subscribe();
        let permit = tokio::select! {
            permit = self.execution.clone().acquire_owned() => permit.ok(),
            () = wait_cancel(&mut queued_cancel, owner.deadline) => None,
        };
        let image_count = request.mm_items.len();
        let prepared = async {
            if permit.is_none() {
                return Err(Status::cancelled("encoder cancelled before execution"));
            }
            let cpu = tokio::select! {
                permit = self.cpu.clone().acquire_owned() => permit
                    .map_err(|_| Status::unavailable("encoder CPU supply closed"))?,
                () = wait_cancel(&mut queued_cancel, owner.deadline) => {
                    return Err(Status::cancelled("encoder cancelled while waiting for CPU"));
                }
            };
            owner.running.store(true, Ordering::Release);
            if *owner.cancel.borrow() || Instant::now() >= owner.deadline {
                return Err(Status::cancelled("encoder cancelled before preprocessing"));
            }
            let artifact = (self.processor)(request.mm_items).await;
            drop(cpu);
            let artifact = artifact?;
            if *owner.cancel.borrow() || Instant::now() >= owner.deadline {
                return Err(Status::cancelled("encoder expired after preprocessing"));
            }
            Ok(artifact)
        }
        .await;
        let receipt = match prepared {
            Ok(artifact) => {
                self.native
                    .call(
                        serde_json::json!({"operation":"execute", "rid":rid,
                "image_count":image_count, "embedding_tokens":artifact.embedding_tokens}),
                        Some(artifact.bytes),
                    )
                    .await
            }
            Err(error) => {
                self.native
                    .call(
                        serde_json::json!({
                "operation":"finish", "rid":rid, "error":error.message()}),
                        None,
                    )
                    .await
            }
        };
        // An execute transport error has ambiguous completion: never retry cleanup.
        finished.send_replace(true);
        let watchdog_result = watchdog.await;
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
            Err(error) => Err(error),
        };
        if !matches!(watchdog_result, Ok(Ok(()))) {
            tracing::warn!(request_id = %rid, ?watchdog_result,
                "encoder cancellation notification failed; native release receipt remains authoritative");
        }
        if safe {
            owner.quiesced.store(true, Ordering::Release);
            drop(permit);
        } else {
            self.draining.store(true, Ordering::Release);
            if let Some(permit) = permit {
                permit.forget();
            }
        }
        owner.result.send_replace(Some(outcome));
    }

    /// P may reach E before Encode. Wait with the same bounded observer deadline, then
    /// bind to the registered operation; never create speculative native state.
    pub async fn facade(&self, kind: &str, payload: Value) -> Result<Value, Status> {
        let rid = payload
            .get("req_id")
            .and_then(Value::as_str)
            .ok_or_else(|| Status::invalid_argument("missing receiver request id"))?
            .to_owned();
        if !rid.starts_with(&format!("{}.", self.config.generation)) || rid.len() > 512 {
            return Err(Status::invalid_argument("stale native part"));
        }
        let early_permit = self
            .early
            .try_acquire()
            .map_err(|_| Status::resource_exhausted("early facade capacity full"))?;
        let early_deadline = Instant::now() + Duration::from_secs(5);
        let owner = loop {
            let notified = self.admitted.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let owner = self
                .owners
                .lock()
                .map_err(|_| Status::internal("encoder ledger poisoned"))?
                .get(&rid)
                .cloned();
            if let Some(owner) = owner {
                break owner;
            }
            tokio::select! {
                () = notified => {},
                () = tokio::time::sleep_until(early_deadline) => return Err(Status::deadline_exceeded("early native receiver expired")),
            }
        };
        drop(early_permit);
        let mut registered = owner.registered.subscribe();
        let mut result = owner.result.subscribe();
        while !*registered.borrow_and_update() {
            if result.borrow_and_update().is_some() {
                return Err(Status::cancelled("native part closed"));
            }
            tokio::select! {
                value = registered.changed() => { value.map_err(|_| Status::unavailable("encoder registration lost"))?; }
                _ = result.changed() => return Err(Status::cancelled("native part closed")),
                () = tokio::time::sleep_until(owner.deadline) => return Err(Status::deadline_exceeded("receiver registration expired")),
            }
        }
        if owner.quiesced.load(Ordering::Acquire)
            || *owner.cancel.borrow()
            || Instant::now() >= owner.deadline
        {
            return Err(Status::cancelled("native part closed"));
        }
        // Expiring this observer never retires the native execution owner.
        tokio::time::timeout_at(owner.deadline, self.native.call(
            serde_json::json!({"operation":"facade", "rid":rid, "kind":kind, "payload":payload}), None,
        )).await.map_err(|_| Status::deadline_exceeded("native receiver deadline expired"))?
    }

    /// Close admission and signal existing owners; their cleanup still determines release.
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
    /// Request cancellation when the RPC observer disappears, without releasing native work.
    fn drop(&mut self) {
        self.0.send_replace(true);
    }
}
/// Observe cancellation or deadline without claiming the underlying work has stopped.
async fn wait_cancel(cancel: &mut watch::Receiver<bool>, deadline: Instant) {
    if *cancel.borrow_and_update() {
        return;
    }
    tokio::select! { _ = cancel.changed() => {}, () = tokio::time::sleep_until(deadline) => {} }
}

use smg_grpc_client::{common_proto as common, sglang_encoder_proto as ep, sglang_proto as sg};
use tonic::{Request, Response};

use crate::{BoxStream, ServerThread, ServicerError};

#[derive(Clone)]
struct EncoderService {
    runtime: Arc<EncoderRuntime>,
    facts: Arc<Value>,
    http_slots: Arc<Semaphore>,
}

impl EncoderService {
    /// Read a string launch fact, using an empty value when it was not supplied.
    fn text(&self, key: &str) -> String {
        self.facts
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    }
    /// Expose queued and running encoder owners through the scheduler load schema.
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
    /// Publish the current instance fence and its processor, transport, and readiness facts.
    fn capability(&self) -> Capabilities {
        use smg_grpc_client::sglang_managed::proto::Instance;
        Capabilities {
            version: 1,
            instance: Some(Instance {
                worker_id: self.text("worker_id"),
                generation: self.runtime.config.generation.clone(),
            }),
            role: "encoder".into(),
            model_revision: self.text("model_revision"),
            processor_contract: self.text("processor_contract"),
            layout_contract: "sglang_native_manifest_v1".into(),
            embedding_dtype: self.text("embedding_dtype"),
            kv_backend: self.text("kv_backend"),
            embedding_backend: self.text("embedding_backend"),
            topology_contract: self.text("topology_contract"),
            receive_endpoint: self.text("receive_endpoint"),
            capacity: self.runtime.config.max_active as u64,
            tp_size: self.facts["tp_size"].as_u64().unwrap_or(0) as u32,
            wall_time_unix_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            ready: self.runtime.ready(),
            ..Default::default()
        }
    }
}

#[tonic::async_trait]
impl ep::sglang_encoder_server::SglangEncoder for EncoderService {
    /// Require a bounded gRPC deadline before admitting the single encoder operation.
    async fn encode(
        &self,
        request: Request<EncodeRequest>,
    ) -> Result<Response<ep::EncodeResponse>, Status> {
        let remaining = crate::rpc::grpc_timeout(request.metadata())
            .ok_or_else(|| Status::invalid_argument("Encode requires a valid bounded deadline"))?;
        self.runtime.encode(request.into_inner(), remaining).await?;
        Ok(Response::new(ep::EncodeResponse::default()))
    }
    /// Reject gRPC sends: native embedding receivers use the bound HTTP facade.
    async fn send(
        &self,
        _: Request<ep::SendRequest>,
    ) -> Result<Response<ep::SendResponse>, Status> {
        Err(Status::unimplemented(
            "native receivers use the bound HTTP facade",
        ))
    }
    /// Reject gRPC registration: native receivers register through the HTTP facade.
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
    /// Expose the single encoder execution unit, not one load record per TP rank.
    fn rank_count(&self) -> usize {
        1
    }
    /// Return encoder load only for rank zero; observers never own request release.
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

#[tonic::async_trait]
impl sg::sglang_scheduler_server::SglangScheduler for EncoderService {
    type GenerateStream = BoxStream<sg::GenerateResponse>;
    type GetTokenizerStream = BoxStream<common::GetTokenizerChunk>;
    type SubscribeKvEventsStream = BoxStream<common::KvEventBatch>;
    /// Report admission readiness without creating native work.
    async fn health_check(
        &self,
        _: Request<sg::HealthCheckRequest>,
    ) -> Result<Response<sg::HealthCheckResponse>, Status> {
        Ok(Response::new(sg::HealthCheckResponse {
            healthy: self.runtime.ready(),
            ..Default::default()
        }))
    }
    /// Expose configured model identity and vision support for worker discovery.
    async fn get_model_info(
        &self,
        _: Request<sg::GetModelInfoRequest>,
    ) -> Result<Response<sg::GetModelInfoResponse>, Status> {
        Ok(Response::new(sg::GetModelInfoResponse {
            model_path: self.text("model_path"),
            tokenizer_path: self.text("tokenizer_path"),
            served_model_name: self.text("served_model_name"),
            model_type: self.text("model_type"),
            is_generation: true,
            supports_vision: true,
            ..Default::default()
        }))
    }
    /// Advertise the fenced encoder contract and current retained request count.
    async fn get_server_info(
        &self,
        _: Request<sg::GetServerInfoRequest>,
    ) -> Result<Response<sg::GetServerInfoResponse>, Status> {
        let args = serde_json::json!({"model_path":self.text("model_path"),
            "served_model_name":self.text("served_model_name"), "supports_vision":true,
            "mm_processor":"smg", "mm_media_ref_schemes":"http,https,data",
            "sg_computation_role":"encoder", "sglang_prepared":serde_json::to_string(&self.capability()).map_err(|error| Status::internal(error.to_string()))?,
            "max_running_requests":self.runtime.config.max_active});
        Ok(Response::new(sg::GetServerInfoResponse {
            server_args: Some(crate::proto_json::struct_from_json(&args.to_string())),
            active_requests: self.load().num_total_reqs,
            ..Default::default()
        }))
    }
    /// Return a snapshot of retained encoder owners through the existing load API.
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
    /// Reuse load streaming only; this encoder does not publish KV cache events.
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
    /// Reject token generation on the encoder role.
    async fn generate(
        &self,
        _: Request<sg::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("not available on the encoder"))
    }
    /// Reject the embedding RPC; vision encoding uses the encoder protocol.
    async fn embed(
        &self,
        _: Request<sg::EmbedRequest>,
    ) -> Result<Response<sg::EmbedResponse>, Status> {
        Err(Status::unimplemented("not available on the encoder"))
    }
    /// Reject scheduler aborts; encoder cancellation follows its Encode observer.
    async fn abort(
        &self,
        _: Request<sg::AbortRequest>,
    ) -> Result<Response<sg::AbortResponse>, Status> {
        Err(Status::unimplemented("not available on the encoder"))
    }
    /// Reject cache flushing on the encoder control service.
    async fn flush_cache(
        &self,
        _: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("not available on the encoder"))
    }
    /// Reject scheduler profiling controls on the encoder role.
    async fn start_profile(
        &self,
        _: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("not available on the encoder"))
    }
    /// Reject scheduler profiling controls on the encoder role.
    async fn stop_profile(
        &self,
        _: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("not available on the encoder"))
    }
    /// Reject tokenizer streaming from the encoder control service.
    async fn get_tokenizer(
        &self,
        _: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("not available on the encoder"))
    }
    /// Reject dynamic LoRA loading on this encoder role.
    async fn load_lo_ra_adapter(
        &self,
        _: Request<sg::LoadLoRaAdapterRequest>,
    ) -> Result<Response<sg::LoadLoRaAdapterResponse>, Status> {
        Err(Status::unimplemented("not available on the encoder"))
    }
    /// Reject dynamic LoRA unloading on this encoder role.
    async fn unload_lo_ra_adapter(
        &self,
        _: Request<sg::UnloadLoRaAdapterRequest>,
    ) -> Result<Response<sg::UnloadLoRaAdapterResponse>, Status> {
        Err(Status::unimplemented("not available on the encoder"))
    }
    /// Reject LoRA inventory queries on this encoder role.
    async fn list_loaded_lo_ra_adapters(
        &self,
        _: Request<sg::ListLoadedLoRaAdaptersRequest>,
    ) -> Result<Response<sg::ListLoadedLoRaAdaptersResponse>, Status> {
        Err(Status::unimplemented("not available on the encoder"))
    }
}

/// Bound receiver bodies and concurrency, forwarding native results with HTTP error mapping.
async fn receiver_http(
    axum::extract::State(service): axum::extract::State<EncoderService>,
    axum::extract::Path(route): axum::extract::Path<String>,
    request: axum::extract::Request,
) -> axum::response::Response {
    let response = async {
        let _permit = service
            .http_slots
            .try_acquire()
            .map_err(|_| Status::resource_exhausted("native HTTP capacity full"))?;
        let kind = match route.as_str() {
            "scheduler_receive_meta_data" => "metadata",
            "scheduler_receive_url" => "register",
            "send" => "send",
            _ => return Err(Status::not_found("unsupported native receiver operation")),
        };
        let body = axum::body::to_bytes(
            request.into_body(),
            service.runtime.config.max_control_bytes,
        )
        .await
        .map_err(|_| Status::out_of_range("control request too large"))?;
        let payload: Value = serde_json::from_slice(&body)
            .map_err(|_| Status::invalid_argument("malformed native receiver control"))?;
        service.runtime.facade(kind, payload).await
    }
    .await;
    use axum::{
        http::{header, StatusCode},
        response::IntoResponse,
    };
    let (status, value) = match response {
        Ok(Value::Null) => (StatusCode::OK, serde_json::json!({"status":"ok"})),
        Ok(value) => (StatusCode::OK, value),
        Err(error) => (
            match error.code() {
                tonic::Code::OutOfRange => StatusCode::PAYLOAD_TOO_LARGE,
                tonic::Code::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
                tonic::Code::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
                tonic::Code::NotFound => StatusCode::NOT_FOUND,
                _ => StatusCode::BAD_REQUEST,
            },
            serde_json::json!({"message":error.message()}),
        ),
    };
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        value.to_string(),
    )
        .into_response()
}

/// Existing Python launcher owns this runtime thread and the native TP process group.
/// No new user-facing option or process entry is required.
pub struct EncoderServer {
    thread: ServerThread,
    runtime: Arc<EncoderRuntime>,
}
impl EncoderServer {
    /// Bind both control listeners and the fenced IPC link; the launcher owns native TP processes.
    pub fn start(config_json: &str, processor: EncoderProcessor) -> Result<Self, ServicerError> {
        let facts: Value = serde_json::from_str(config_json)
            .map_err(|error| ServicerError::InvalidConfig(error.to_string()))?;
        let config: EncoderConfig = serde_json::from_value(facts.clone())
            .map_err(|error| ServicerError::InvalidConfig(error.to_string()))?;
        let address = facts["control_listen"]
            .as_str()
            .ok_or_else(|| ServicerError::InvalidConfig("missing control address".into()))?
            .to_owned();
        let (ready, state) = sync_channel(1);
        let thread = ServerThread::start(
            "sglang-encoder",
            &address,
            move |listener, shutdown, _| async move {
                let link = NativeEncoderLink::bind(
                    facts["native_commands"]
                        .as_str()
                        .ok_or("missing native command endpoint")?,
                    facts["native_events"]
                        .as_str()
                        .ok_or("missing native event endpoint")?,
                    config.generation.clone(),
                    config.max_active * 16,
                )
                .await
                .map_err(|error| error.to_string())?;
                let runtime = EncoderRuntime::new(config, link, processor)
                    .map_err(|error| error.to_string())?;
                let max_control_bytes = runtime.config.max_control_bytes;
                let http_address = facts["http_listen"]
                    .as_str()
                    .ok_or("missing HTTP address")?;
                let http = tokio::net::TcpListener::bind(http_address)
                    .await
                    .map_err(|error| error.to_string())?;
                let slots = facts["max_http_connections"]
                    .as_u64()
                    .ok_or("missing HTTP capacity")? as usize;
                let service = EncoderService {
                    runtime: runtime.clone(),
                    facts: Arc::new(facts),
                    http_slots: Arc::new(Semaphore::new(slots)),
                };
                let app = axum::Router::new()
                    .route("/{route}", axum::routing::post(receiver_http))
                    .with_state(service.clone());
                ready
                    .send(runtime.clone())
                    .map_err(|error| error.to_string())?;
                let draining = shutdown.clone().map(move |_| runtime.drain()).shared();
                let http = axum::serve(http, app).with_graceful_shutdown(draining.clone());
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
                    );
                let serving = async {
                    tokio::try_join!(
                        async { http.await.map_err(|error| error.to_string()) },
                        grpc.map_err(|error| error.to_string())
                    )?;
                    Ok::<(), String>(())
                };
                crate::server::serve_with_grace(serving, shutdown).await
            },
        )?;
        let runtime = state
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| ServicerError::Startup(error.to_string()))?;
        Ok(Self { thread, runtime })
    }
    /// Close admission and request cancellation while native completion remains observable.
    pub fn drain(&self) {
        self.runtime.drain();
    }
    /// Count owners still lacking confirmed native release, regardless of observer lifetime.
    pub fn active(&self) -> usize {
        let (running, waiting) = self.runtime.loads();
        running + waiting
    }
    /// Report control-thread liveness, not native request quiescence.
    pub fn running(&self) -> bool {
        self.thread.running()
    }
    /// Expose control-thread failure so the launcher can enter native cleanup.
    pub fn last_error(&self) -> Result<Option<String>, ServicerError> {
        self.thread.last_error()
    }
    /// Stop control listeners within their grace; the caller must drain native owners first.
    pub fn stop(&self, timeout: Duration) -> Result<(), ServicerError> {
        self.drain();
        self.thread.stop(timeout)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Use tiny finite budgets so ownership leaks fail without exhausting host resources.
    fn test_config() -> EncoderConfig {
        EncoderConfig {
            generation: "current".into(),
            max_active: 1,
            max_records: 2,
            max_cpu: 1,
            execution_capacity: 1,
            max_images: 4,
            max_control_bytes: 1024,
            retention_ms: 1000,
            max_clock_error_ms: 1000,
            max_early_waiters: 2,
        }
    }

    /// Build an isolated native control link whose events are driven by each test.
    fn local_link(capacity: usize) -> (NativeEncoderLink, mpsc::Receiver<ZmqMessage>) {
        let (commands, input) = mpsc::channel(capacity);
        let link = NativeEncoderLink {
            commands,
            state: Arc::new(NativeState {
                generation: "current".into(),
                sequence: AtomicU64::new(1),
                pending: Mutex::default(),
                failed: AtomicBool::new(false),
            }),
        };
        (link, input)
    }

    fn control_runtime(link: NativeEncoderLink) -> Arc<EncoderRuntime> {
        EncoderRuntime::new(
            test_config(),
            link,
            Arc::new(|_| Box::pin(async { panic!("unexpected preprocessing") })),
        )
        .unwrap()
    }

    async fn connected_native_peer(
        directory: &tempfile::TempDir,
    ) -> (NativeEncoderLink, PullSocket, PushSocket) {
        let commands = format!("ipc://{}/commands", directory.path().display());
        let events = format!("ipc://{}/events", directory.path().display());
        let link = NativeEncoderLink::bind(&commands, &events, "current".into(), 8)
            .await
            .unwrap();
        let mut input = PullSocket::new();
        input.connect(&commands).await.unwrap();
        let mut output = PushSocket::new();
        output.connect(&events).await.unwrap();
        (link, input, output)
    }

    fn native_reply(command: &Value, result: Value) -> ZmqMessage {
        let reply =
            json!({"generation":"current", "sequence":command["sequence"], "result":result});
        Bytes::from(reply.to_string()).into()
    }

    /// Give each test operation a valid instance-bound native request identity.
    fn admission_request(id: &str) -> EncodeRequest {
        EncodeRequest {
            req_id: id.into(),
            prefill_host: "p".into(),
            num_parts: 1,
            mm_items: vec!["image".into()],
            ..Default::default()
        }
    }

    #[tokio::test(start_paused = true)]
    /// A cancellation request alone cannot make an admitted native owner reusable.
    async fn admission_fences_duplicates_and_retains_unknown_native_work() {
        let runtime = control_runtime(local_link(2).0);
        let request = admission_request("current.first");
        let owner = runtime.admit(&request, Duration::from_secs(1)).unwrap();
        assert_eq!(runtime.loads(), (0, 1));
        for id in ["current.first", "old.first"] {
            let error = runtime.admit(&admission_request(id), Duration::from_secs(1));
            let expected = if id == "current.first" {
                tonic::Code::AlreadyExists
            } else {
                tonic::Code::InvalidArgument
            };
            assert_eq!(error.err().unwrap().code(), expected);
        }
        owner.cancel.send_replace(true);
        tokio::time::advance(Duration::from_secs(3)).await;
        assert_eq!(
            runtime
                .admit(&admission_request("current.second"), Duration::from_secs(1))
                .err()
                .unwrap()
                .code(),
            tonic::Code::ResourceExhausted
        );
        assert!(runtime.owners.lock().unwrap().contains_key("current.first"));
        assert_eq!(runtime.loads(), (0, 1));
        owner.quiesced.store(true, Ordering::Release);
        runtime
            .admit(&admission_request("current.second"), Duration::from_secs(1))
            .unwrap();
        assert!(!runtime.owners.lock().unwrap().contains_key("current.first"));
    }

    #[tokio::test(start_paused = true)]
    /// Completed requests remain fenced through their original deadline and retention horizon.
    async fn released_replay_fence_expires_only_after_original_horizon() {
        let runtime = control_runtime(local_link(2).0);
        let request = admission_request("current.first");
        let owner = runtime.admit(&request, Duration::from_secs(1)).unwrap();
        let horizon = owner.retain_until;
        owner.quiesced.store(true, Ordering::Release);
        tokio::time::advance(Duration::from_secs(2)).await;
        assert_eq!(Instant::now(), horizon);
        assert_eq!(
            runtime
                .admit(&request, Duration::from_secs(1))
                .err()
                .unwrap()
                .code(),
            tonic::Code::AlreadyExists
        );
        assert_eq!(owner.retain_until, horizon);
        tokio::time::advance(Duration::from_nanos(1)).await;
        runtime.admit(&request, Duration::from_secs(1)).unwrap();
    }

    #[tokio::test]
    /// Routing load includes cancelled execution until native completion proves it is gone.
    async fn reported_load_tracks_execution_and_requires_real_quiescence() {
        use crate::load_stream::RankedLoads;
        let runtime = control_runtime(local_link(2).0);
        let owner = runtime
            .admit(&admission_request("current.first"), Duration::from_secs(1))
            .unwrap();
        let service = EncoderService {
            runtime,
            facts: Arc::new(serde_json::json!({})),
            http_slots: Arc::new(Semaphore::new(1)),
        };
        let waiting = service.rank_record(0).unwrap();
        assert_eq!(
            (
                waiting.running_requests,
                waiting.waiting_requests,
                waiting.max_running_requests
            ),
            (0, 1, 1)
        );
        owner.running.store(true, Ordering::Release);
        owner.cancel.send_replace(true);
        let cancelling = service.rank_record(0).unwrap();
        assert_eq!(
            (cancelling.running_requests, cancelling.waiting_requests),
            (1, 0)
        );
        owner.quiesced.store(true, Ordering::Release);
        let released = service.rank_record(0).unwrap();
        assert_eq!(
            (released.running_requests, released.waiting_requests),
            (0, 0)
        );
        assert!(service.rank_record(1).is_none());
    }

    #[tokio::test]
    /// TP receivers share one operation while each waiting HTTP control consumes bounded capacity.
    async fn early_tp_receivers_share_admission_but_charge_each_waiter() {
        let (link, mut input) = local_link(4);
        let native = link.state.clone();
        let runtime = control_runtime(link);
        let mut receivers = Vec::new();
        for _ in 0..2 {
            let observer = runtime.clone();
            #[expect(
                clippy::disallowed_methods,
                reason = "test joins both early TP receivers"
            )]
            receivers.push(tokio::spawn(async move {
                observer
                    .facade("metadata", json!({"req_id":"current.req", "part_idx":0}))
                    .await
            }));
        }
        tokio::time::timeout(Duration::from_secs(1), async {
            while runtime.early.available_permits() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(runtime.owners.lock().unwrap().is_empty());
        assert_eq!(
            runtime
                .facade("metadata", json!({"req_id":"current.req"}))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::ResourceExhausted
        );
        let owner = runtime
            .admit(&admission_request("current.req"), Duration::from_secs(30))
            .unwrap();
        owner.registered.send_replace(true);
        runtime.admitted.notify_waiters();
        for _ in 0..2 {
            let command: Value =
                serde_json::from_slice(&input.recv().await.unwrap().into_vec()[0]).unwrap();
            native.receive(native_reply(&command, Value::Null)).unwrap();
        }
        for receiver in receivers {
            receiver.await.unwrap().unwrap();
        }
        assert_eq!(runtime.owners.lock().unwrap().len(), 1);
        assert_eq!(runtime.early.available_permits(), 2);
        assert_eq!(runtime.loads(), (0, 1));
    }

    #[tokio::test]
    /// Native receiver metadata and failures must survive the Rust HTTP bridge unchanged.
    async fn native_http_routes_preserve_body_and_receiver_errors() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        /// Exercise the actual HTTP parser and response path rather than calling handlers directly.
        async fn post(address: std::net::SocketAddr, route: &str, body: &str) -> String {
            let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
            client.write_all(format!("POST /{route} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            let mut response = Vec::new();
            client.read_to_end(&mut response).await.unwrap();
            String::from_utf8(response).unwrap()
        }

        let (link, mut input) = local_link(4);
        let native = link.state.clone();
        let runtime = control_runtime(link);
        let owner = runtime
            .admit(&admission_request("current.req"), Duration::from_secs(30))
            .unwrap();
        owner.registered.send_replace(true);
        let slots = Arc::new(Semaphore::new(1));
        let service = EncoderService {
            runtime,
            facts: Arc::new(json!({})),
            http_slots: slots.clone(),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = axum::Router::new()
            .route("/{route}", axum::routing::post(receiver_http))
            .with_state(service);
        let (stop, stopped) = oneshot::channel();
        #[expect(
            clippy::disallowed_methods,
            reason = "test owns and joins its HTTP server"
        )]
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        let body = json!({"req_id":"current.req", "part_idx":0, "receive_count":1, "receive_url":"p:1234"}).to_string();
        for (route, kind, error) in [
            ("scheduler_receive_meta_data", "metadata", false),
            ("scheduler_receive_url", "register", false),
            ("send", "send", false),
            ("send", "send", true),
        ] {
            let (response, ()) = tokio::join!(post(address, route, &body), async {
                let command: Value =
                    serde_json::from_slice(&input.recv().await.unwrap().into_vec()[0]).unwrap();
                assert_eq!(command["kind"], kind);
                assert_eq!(
                    command["payload"],
                    serde_json::from_str::<Value>(&body).unwrap()
                );
                let mut reply =
                    json!({"generation":"current", "sequence":command["sequence"], "result":null});
                if error {
                    reply["error"] = json!("receiver expired");
                    reply["error_code"] = json!(4);
                }
                native
                    .receive(Bytes::from(reply.to_string()).into())
                    .unwrap();
            });
            assert!(
                response.starts_with(if error {
                    "HTTP/1.1 504"
                } else {
                    "HTTP/1.1 200"
                }),
                "{response}"
            );
            if error {
                assert!(response.contains("\"message\":\"receiver expired\""));
            }
        }
        assert!(post(address, "send", &"x".repeat(1025))
            .await
            .starts_with("HTTP/1.1 413"));
        assert!(post(address, "send", "[]")
            .await
            .starts_with("HTTP/1.1 400"));
        assert!(
            input.try_recv().is_err(),
            "invalid body must not reach native control"
        );
        assert!(post(address, "send", "{").await.starts_with("HTTP/1.1 400"));
        assert!(post(address, "unknown", &body)
            .await
            .starts_with("HTTP/1.1 404"));
        let held = slots.acquire().await.unwrap();
        assert!(post(address, "send", &body)
            .await
            .starts_with("HTTP/1.1 429"));
        drop(held);
        stop.send(()).unwrap();
        server.await.unwrap();
    }

    /// A missing facade response cannot retain an HTTP observer beyond the owner deadline.
    #[tokio::test(start_paused = true)]
    async fn managed_facade_deadline_retires_only_its_observer() {
        let (link, mut received) = local_link(2);
        let runtime = control_runtime(link);
        let owner = runtime
            .admit(&admission_request("current.facade"), Duration::from_secs(1))
            .unwrap();
        owner.registered.send_replace(true);
        let (response, command) = tokio::join!(
            tokio::time::timeout(
                Duration::from_secs(2),
                runtime.facade("send", json!({"req_id":"current.facade"}))
            ),
            received.recv(),
        );
        assert!(command.is_some());
        assert_eq!(
            response
                .expect("facade must obey the owner's deadline")
                .unwrap_err()
                .code(),
            tonic::Code::DeadlineExceeded
        );
        assert!(runtime.native.state.pending.lock().unwrap().is_empty());
        assert!(!owner.quiesced.load(Ordering::Acquire));
        assert!(runtime
            .owners
            .lock()
            .unwrap()
            .contains_key("current.facade"));
        assert!(runtime
            .admit(&admission_request("current.next"), Duration::from_secs(1))
            .is_err());
    }

    #[tokio::test]
    /// Stopping the encoder must retire a connected load observer without waiting for its client.
    async fn stop_closes_a_gateway_load_stream_within_the_grace() {
        let directory = tempfile::tempdir().unwrap();
        let facts = json!({
            "generation":"current", "max_active":2, "max_records":4,
            "max_cpu":1, "execution_capacity":1, "max_images":4,
            "max_control_bytes":1024, "retention_ms":1000,
            "max_clock_error_ms":1000, "max_early_waiters":2,
            "control_listen":"127.0.0.1:0", "http_listen":"127.0.0.1:0",
            "max_http_connections":2,
            "native_commands":format!("ipc://{}/commands", directory.path().display()),
            "native_events":format!("ipc://{}/events", directory.path().display()),
        });
        let processor: EncoderProcessor = Arc::new(|_| {
            Box::pin(async { panic!("a load subscriber must not trigger preprocessing") })
        });
        let server = EncoderServer::start(&facts.to_string(), processor).unwrap();
        let mut client = sg::sglang_scheduler_client::SglangSchedulerClient::connect(format!(
            "http://{}",
            server.thread.address()
        ))
        .await
        .unwrap();
        let mut stream = client
            .subscribe_kv_events(common::SubscribeKvEventsRequest {
                loads_only: true,
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        assert!(stream.message().await.unwrap().is_some());
        let owner = server
            .runtime
            .admit(
                &admission_request("current.active"),
                Duration::from_secs(30),
            )
            .unwrap();
        drop(stream);
        assert_eq!(server.active(), 1);
        assert!(!owner.quiesced.load(Ordering::Acquire));
        // Reconnect a load observer; neither disconnect nor reconnect owns request release.
        let mut stream = client
            .subscribe_kv_events(common::SubscribeKvEventsRequest {
                loads_only: true,
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        assert!(stream.message().await.unwrap().is_some());
        assert_eq!(server.active(), 1);
        owner.quiesced.store(true, Ordering::Release);
        assert_eq!(server.active(), 0);
        // Keep the gateway and its stream alive while the server closes its listeners.
        server.stop(Duration::from_secs(1)).unwrap();
        assert!(!server.thread.running());
        tokio::time::timeout(Duration::from_secs(2), async {
            while let Ok(Some(_)) = stream.message().await {}
        })
        .await
        .unwrap();
        drop(client);
    }

    #[tokio::test]
    /// A late receipt from an earlier native generation cannot release the current operation.
    async fn stale_generation_never_completes_current_owner() {
        let (link, _) = local_link(2);
        let (tx, mut rx) = oneshot::channel();
        link.state.pending.lock().unwrap().insert(7, tx);
        let reply = json!({"generation":"old", "sequence":7, "result":{"released":true}});
        assert!(link
            .state
            .receive(Bytes::from(reply.to_string()).into())
            .is_err());
        assert!(matches!(
            rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert_eq!(link.state.pending.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    /// A lost native link must retain uncertain ownership instead of inventing quiescence.
    async fn connection_failure_never_claims_native_release() {
        let (link, _) = local_link(2);
        let (tx, rx) = oneshot::channel();
        link.state.pending.lock().unwrap().insert(1, tx);
        link.state.fail(Status::unavailable("lost native stream"));
        assert!(rx.await.unwrap().is_err());
        assert!(!link.ready());
        assert!(link
            .call(json!({"operation":"cancel"}), None)
            .await
            .is_err());
    }

    #[tokio::test]
    /// An unused native link must not leave its socket tasks alive after its owner is dropped.
    async fn dropping_link_retires_idle_native_sockets() {
        let directory = tempfile::tempdir().unwrap();
        let commands = format!("ipc://{}/commands", directory.path().display());
        let events = format!("ipc://{}/events", directory.path().display());
        let link = NativeEncoderLink::bind(&commands, &events, "current".into(), 2)
            .await
            .unwrap();
        let state = Arc::downgrade(&link.state);
        drop(link);
        tokio::time::timeout(Duration::from_secs(1), async {
            while state.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
    #[tokio::test]
    /// Prefill may establish transfer state before execution; queued cancellation still needs a release receipt.
    async fn early_prefill_control_precedes_execution_slot_and_queued_cancel_proves_release() {
        for (prefill_first, released, cpu_wait, expire) in [
            (true, true, false, false),
            (false, true, false, false),
            (true, false, false, false),
            (true, true, true, false),
            (true, false, true, false),
            (true, true, true, true),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let (link, mut input, mut output) = connected_native_peer(&directory).await;
            let runtime = control_runtime(link);
            let held = if cpu_wait {
                runtime.cpu.clone()
            } else {
                runtime.execution.clone()
            }
            .acquire_owned()
            .await
            .unwrap();
            let request = admission_request("current.req");
            let remaining = Duration::from_secs(if expire { 1 } else { 10 });
            let receiver_runtime = runtime.clone();
            let receiver = async move {
                receiver_runtime
                    .facade("metadata", json!({"req_id":"current.req", "part_idx":0}))
                    .await
            };
            let admitted = (!prefill_first).then(|| runtime.admit(&request, remaining).unwrap());
            #[expect(
                clippy::disallowed_methods,
                reason = "test awaits the receiver observer"
            )]
            let receiver = tokio::spawn(receiver);
            let owner = if let Some(owner) = admitted {
                owner
            } else {
                tokio::time::timeout(Duration::from_secs(1), async {
                    while runtime.early.available_permits() == 2 {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .unwrap();
                let owner = runtime.admit(&request, remaining).unwrap();
                runtime.admitted.notify_waiters();
                owner
            };
            #[expect(
                clippy::disallowed_methods,
                reason = "test awaits the queued execution owner"
            )]
            let mut execution = tokio::spawn(runtime.clone().execute(request, owner.clone()));
            for operation in ["register", "facade"] {
                let message = tokio::time::timeout(Duration::from_secs(1), input.recv())
                    .await
                    .unwrap()
                    .unwrap()
                    .into_vec();
                assert_eq!(
                    message.len(),
                    1,
                    "no tensor exists before execution admission"
                );
                let command: Value = serde_json::from_slice(&message[0]).unwrap();
                assert_eq!(command["operation"], operation);
                if !cpu_wait {
                    assert_eq!(runtime.execution.available_permits(), 0);
                }
                assert_eq!(runtime.cpu.available_permits(), usize::from(!cpu_wait));
                assert!(!owner.running.load(Ordering::Acquire));
                output
                    .send(native_reply(&command, Value::Null))
                    .await
                    .unwrap();
            }
            receiver.await.unwrap().unwrap();
            if cpu_wait {
                tokio::time::timeout(Duration::from_secs(1), async {
                    while runtime.execution.available_permits() != 0 {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .unwrap();
            }
            if expire {
                tokio::time::sleep_until(owner.deadline).await;
            } else {
                owner.cancel.send_replace(true);
            }
            tokio::time::timeout(Duration::from_secs(2),async {
                loop {
                    tokio::select! {
                        done=&mut execution=> {done.unwrap();break;}
                        command=input.recv()=> {
                            let frames=command.unwrap().into_vec();
                            assert_eq!(frames.len(),1,"queued cancellation cannot send tensor work");
                            let command:Value=serde_json::from_slice(&frames[0]).unwrap();
                            assert!(!owner.quiesced.load(Ordering::Acquire));
                            let result=match command["operation"].as_str().unwrap() {
                                "cancel"=>Value::Null,
                                "finish"=>json!({"released":released,"error":"cancelled"}),
                                other=>panic!("unexpected native operation {other}"),
                            };
                            output.send(native_reply(&command, result)).await.unwrap();
                        }
                    }
                }
            }).await.unwrap();
            assert_eq!(owner.quiesced.load(Ordering::Acquire), released);
            assert_eq!(runtime.draining.load(Ordering::Acquire), !released);
            assert_eq!(runtime.loads(), (0, usize::from(!released)));
            assert_eq!(
                runtime.execution.available_permits(),
                usize::from(cpu_wait && released)
            );
            drop(held);
            assert_eq!(
                runtime.execution.available_permits(),
                usize::from(!cpu_wait || released)
            );
            assert_eq!(runtime.cpu.available_permits(), 1);
        }
    }

    #[tokio::test]
    /// The execution budget remains charged while an already-started native operation is cancelling.
    async fn native_cancel_does_not_release_execution_until_real_completion() {
        for (acknowledge_cancel, cancel_error, released) in [
            (true, false, true),
            (false, false, true),
            (false, false, false),
            (true, true, true),
            (true, true, false),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let (link, mut input, mut output) = connected_native_peer(&directory).await;
            let config = test_config();
            let process: EncoderProcessor = Arc::new(|_| {
                Box::pin(async {
                    Ok(EncoderArtifact {
                        bytes: Bytes::from_static(b"artifact"),
                        embedding_tokens: 4,
                    })
                })
            });
            let runtime = EncoderRuntime::new(config, link, process).unwrap();
            let mut request = admission_request("current.req");
            request.mm_items = vec!["data:image".into()];
            let owner = runtime.admit(&request, Duration::from_secs(10)).unwrap();
            #[expect(
                clippy::disallowed_methods,
                reason = "test awaits the owned execution handle"
            )]
            let execution = tokio::spawn(runtime.clone().execute(request, owner.clone()));
            let register: Value =
                serde_json::from_slice(&input.recv().await.unwrap().into_vec()[0]).unwrap();
            assert_eq!(register["operation"], "register");
            output
                .send(native_reply(&register, Value::Null))
                .await
                .unwrap();
            let message = input.recv().await.unwrap().into_vec();
            let execute: Value = serde_json::from_slice(&message[0]).unwrap();
            assert_eq!(execute["operation"], "execute");
            assert_eq!(message[1], Bytes::from_static(b"artifact"));
            owner.cancel.send_replace(true);
            let cancel: Value =
                serde_json::from_slice(&input.recv().await.unwrap().into_vec()[0]).unwrap();
            assert_eq!(cancel["operation"], "cancel");
            if acknowledge_cancel {
                output
            .send(
                Bytes::from(
                    json!({"sequence":cancel["sequence"], "generation":"current", "result":null,
                        "error":cancel_error.then_some("cancel delivery failed")})
                        .to_string(),
                )
                .into(),
            )
            .await
            .unwrap();
            }
            if acknowledge_cancel {
                let sequence = cancel["sequence"].as_u64().unwrap();
                while runtime
                    .native
                    .state
                    .pending
                    .lock()
                    .unwrap()
                    .contains_key(&sequence)
                {
                    tokio::task::yield_now().await;
                }
            }
            tokio::task::yield_now().await;
            assert_eq!(runtime.execution.available_permits(), 0);
            assert!(!owner.quiesced.load(Ordering::Acquire));
            output
                .send(native_reply(
                    &execute,
                    json!({"released":released, "error":"cancelled"}),
                ))
                .await
                .unwrap();
            tokio::time::timeout(Duration::from_secs(2), execution)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(runtime.execution.available_permits(), usize::from(released));
            assert_eq!(owner.quiesced.load(Ordering::Acquire), released);
            assert_eq!(runtime.draining.load(Ordering::Acquire), !released);
            assert!(runtime.native.state.pending.lock().unwrap().is_empty());
            // A late or replayed acknowledgement cannot create a new permit or owner.
            for _ in 0..2 {
                runtime
                    .native
                    .state
                    .receive(native_reply(&cancel, Value::Null))
                    .unwrap();
            }
            assert_eq!(runtime.execution.available_permits(), usize::from(released));
            assert!(owner.result.borrow().as_ref().unwrap().is_err());
        }
    }
    #[tokio::test]
    /// Registration timeout ends the caller wait without reusing capacity that native code may own.
    async fn lost_registration_ack_is_bounded_but_retains_uncertain_charge() {
        let (link, _receiver) = local_link(8);
        let runtime = control_runtime(link);
        let request = admission_request("current.req");
        let owner = runtime.admit(&request, Duration::from_millis(20)).unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            runtime.clone().execute(request, owner.clone()),
        )
        .await
        .unwrap();
        assert!(owner.result.borrow().as_ref().unwrap().is_err());
        assert!(!owner.quiesced.load(Ordering::Acquire));
        assert_eq!(runtime.execution.available_permits(), 1);
        assert_eq!(runtime.loads(), (0, 1));
        assert!(runtime.draining.load(Ordering::Acquire));
        assert!(runtime.native.state.pending.lock().unwrap().is_empty());
    }

    #[tokio::test(start_paused = true)]
    /// Request cancellation must not turn an otherwise healthy encoder into a draining instance.
    async fn ordinary_cancel_before_or_during_registration_never_drains_worker() {
        for (before_send, remaining) in [
            (true, Duration::from_secs(10)),
            (false, Duration::from_secs(10)),
            (false, Duration::from_micros(500)),
        ] {
            let skip_registration = before_send || remaining < Duration::from_millis(1);
            let (link, mut receiver) = local_link(8);
            let runtime = control_runtime(link);
            let request = admission_request("current.req");
            let owner = runtime.admit(&request, remaining).unwrap();
            if before_send {
                owner.cancel.send_replace(true);
            }
            #[expect(
                clippy::disallowed_methods,
                reason = "test awaits the owned execution handle"
            )]
            let mut execution = tokio::spawn(runtime.clone().execute(request, owner.clone()));
            if !skip_registration {
                let register: Value =
                    serde_json::from_slice(&receiver.recv().await.unwrap().into_vec()[0]).unwrap();
                assert_eq!(register["operation"], "register");
                owner.cancel.send_replace(true);
                tokio::time::sleep(Duration::from_millis(5)).await;
                assert!(!runtime.draining.load(Ordering::Acquire));
                assert_eq!(runtime.execution.available_permits(), 1);
                runtime
                    .native
                    .state
                    .receive(native_reply(&register, Value::Null))
                    .unwrap();
            }
            tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    tokio::select! {
                        result = &mut execution => { result.unwrap(); break; }
                        command = receiver.recv() => {
                            assert!(!skip_registration, "cancelled owner submitted a native command");
                            let command: Value = serde_json::from_slice(&command.unwrap().into_vec()[0]).unwrap();
                            let result = match command["operation"].as_str().unwrap() {
                                "cancel" => Value::Null,
                                "finish" => json!({"released":true,"error":"cancelled"}),
                                other => panic!("unexpected native operation {other}"),
                            };
                            runtime.native.state.receive(native_reply(&command, result)).unwrap();
                        }
                    }
                }
            }).await.unwrap();
            assert_eq!(runtime.execution.available_permits(), 1);
            assert!(owner.quiesced.load(Ordering::Acquire));
            assert!(!runtime.draining.load(Ordering::Acquire));
            assert!(owner.result.borrow().as_ref().unwrap().is_err());
        }
    }
}
