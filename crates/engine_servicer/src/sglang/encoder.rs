//! Same-host native encoder control behind gRPC. A cancellation message is only a wakeup;
//! only an explicit released response permits the control owner to return execution permits.

use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::sync_channel,
        Arc, Mutex,
    },
    time::Duration,
};

use bytes::Bytes;
use futures::{future::BoxFuture, FutureExt, TryFutureExt};
use prost::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use smg_grpc_client::{
    common_proto as common,
    sglang_encoder_proto::{self as ep, EncodeRequest},
    sglang_managed::{normalized_capabilities, proto::Capabilities},
    sglang_proto as sg,
};
use tokio::{
    sync::{mpsc, oneshot, watch, Semaphore},
    time::Instant,
};
use tonic::{Request, Response, Status};
use zeromq::{PullSocket, PushSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use crate::{BoxStream, ServerThread, ServicerError};

const CANCEL_RECEIPT_GRACE: Duration = Duration::from_secs(30);

/// Distinguishes pre-send failures (slot may be released) from ambiguous post-send outcomes.
enum NativeCallError {
    NotSent(Status),
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
                let result = message.map_err(wire_error).and_then(|m| state.receive(m));
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

    /// Return whether Python has connected its IPC sockets and the link has no fatal error.
    pub fn ready(&self) -> bool {
        !self.state.failed.load(Ordering::Acquire) && self.state.connected.load(Ordering::Acquire)
    }

    fn control_frame(&self, operation: &str, rid: &str) -> Option<ZmqMessage> {
        if rid.is_empty()
            || rid.len() > 256
            || self.state.failed.load(Ordering::Acquire)
            || !self.state.connected.load(Ordering::Acquire)
        {
            return None;
        }
        let header = serde_json::to_vec(&serde_json::json!({
            "operation": operation,
            "rid": rid,
        }))
        .ok()?;
        Some(ZmqMessage::from(Bytes::from(header)))
    }

    /// Best-effort notification to wake early HTTP facade waiters when an encode fails before `execute`.
    pub fn notify_forget(&self, rid: &str) {
        if let Some(frame) = self.control_frame("forget", rid) {
            let _ = self.commands.try_send((0, frame));
        }
    }

    /// Enqueue a FIFO cancellation wakeup behind `execute` without waiting for a separate reply.
    async fn send_cancel(&self, rid: &str) {
        if let Some(frame) = self.control_frame("cancel", rid) {
            let _ = self.commands.send((0, frame)).await;
        }
    }
}

impl NativeState {
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

    fn fail(&self, error: Status) {
        self.failed.store(true, Ordering::Release);
        if let Ok(mut pending) = self.pending.lock() {
            for (_, sender) in pending.drain() {
                let _ = sender.send(Err(NativeCallError::Ambiguous(error.clone())));
            }
        }
    }
}

fn wire_error(error: impl std::fmt::Display) -> Status {
    Status::unavailable(error.to_string())
}

const PRE_DECODE_CACHE_MAX_ENTRIES: usize = 65_536;

/// RAII cleanup guard for a `/dev/shm` pixel payload file owned by an in-flight encoder command.
/// Unlinks the file on drop if Python has not already unlinked it.
#[derive(Debug)]
pub struct ShmGuard {
    path: PathBuf,
}

impl ShmGuard {
    /// Wrap a `/dev/shm` file path so it is unlinked when dropped.
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for ShmGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Compact metadata for one preprocessed tensor batch (no protobuf pixel wrapping).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeBatchMeta {
    pub pixel_dtype: String,
    pub pixel_shape: Vec<u32>,
    pub pixel_offset: usize,
    pub pixel_nbytes: usize,
    pub grid_thw: Vec<[i64; 3]>,
    pub placeholder_lengths: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub second_per_grid_ts: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub second_per_grid_shape: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub item_hashes: Vec<String>,
}

/// Payload carried from the Rust encoder processor to the Python native encoder runtime.
#[derive(Debug)]
pub enum EncoderPayload {
    /// Legacy serialized `sglang_proto::GenerateRequest` protobuf bytes.
    LegacyProto(Bytes),
    /// Compact v2 metadata + zero-copy `/dev/shm` (or inline raw bytes) pixel tensors.
    Compact {
        batches: Vec<NativeBatchMeta>,
        shm_name: Option<String>,
        shm_nbytes: usize,
        shm_guard: Option<ShmGuard>,
        inline_pixels: Option<Bytes>,
    },
}

pub struct EncoderArtifact {
    pub payload: EncoderPayload,
    pub embedding_tokens: u64,
}

impl EncoderArtifact {
    /// Construct an artifact carrying legacy serialized `GenerateRequest` protobuf bytes.
    pub fn legacy(bytes: Bytes, embedding_tokens: u64) -> Self {
        Self {
            payload: EncoderPayload::LegacyProto(bytes),
            embedding_tokens,
        }
    }
}

#[derive(Debug, Clone)]
struct CachedSourceMeta {
    item_key_hex: String,
    grid_thw: [i64; 3],
    token_count: u32,
}

struct PreDecodeHit {
    item_keys: Vec<String>,
    grid_thw: Vec<[i64; 3]>,
    token_counts: Vec<u32>,
    embedding_tokens: u64,
}

/// Bounded Rust-side pre-decode index mapping immutable `data:` image sources + options
/// to resident Python `PerItemEmbeddingCache` keys and layout metadata.
struct PreDecodeCache {
    by_source: HashMap<[u8; 32], CachedSourceMeta>,
    order: VecDeque<[u8; 32]>,
    by_item_key: HashMap<String, Vec<[u8; 32]>>,
    max_entries: usize,
}

impl PreDecodeCache {
    /// Create a bounded FIFO pre-decode index with at most `max_entries` cached sources.
    fn new(max_entries: usize) -> Self {
        Self {
            by_source: HashMap::new(),
            order: VecDeque::new(),
            by_item_key: HashMap::new(),
            max_entries,
        }
    }

    /// Return cached item keys and layout metadata only when every item in the batch hits.
    fn lookup_all(&self, source_keys: &[Option<[u8; 32]>]) -> Option<PreDecodeHit> {
        if source_keys.is_empty() {
            return None;
        }
        let mut item_keys = Vec::with_capacity(source_keys.len());
        let mut grid_thw = Vec::with_capacity(source_keys.len());
        let mut token_counts = Vec::with_capacity(source_keys.len());
        let mut embedding_tokens = 0u64;
        for maybe_key in source_keys {
            let src_key = (*maybe_key)?;
            let meta = self.by_source.get(&src_key)?;
            item_keys.push(meta.item_key_hex.clone());
            grid_thw.push(meta.grid_thw);
            token_counts.push(meta.token_count);
            embedding_tokens = embedding_tokens.saturating_add(u64::from(meta.token_count));
        }
        Some(PreDecodeHit {
            item_keys,
            grid_thw,
            token_counts,
            embedding_tokens,
        })
    }

    /// Record or update a source-to-resident-item mapping, evicting oldest entries at capacity.
    fn insert(&mut self, source_key: [u8; 32], meta: CachedSourceMeta) {
        if self.max_entries == 0 {
            return;
        }
        if let Some(prev) = self.by_source.get(&source_key) {
            if prev.item_key_hex == meta.item_key_hex
                && prev.grid_thw == meta.grid_thw
                && prev.token_count == meta.token_count
            {
                return;
            }
            let old_key = prev.item_key_hex.clone();
            if let Some(list) = self.by_item_key.get_mut(&old_key) {
                list.retain(|k| *k != source_key);
                if list.is_empty() {
                    self.by_item_key.remove(&old_key);
                }
            }
        } else {
            while self.by_source.len() >= self.max_entries {
                let Some(oldest) = self.order.pop_front() else {
                    break;
                };
                if let Some(evicted) = self.by_source.remove(&oldest) {
                    if let Some(list) = self.by_item_key.get_mut(&evicted.item_key_hex) {
                        list.retain(|k| *k != oldest);
                        if list.is_empty() {
                            self.by_item_key.remove(&evicted.item_key_hex);
                        }
                    }
                }
            }
            if self.order.len() >= self.max_entries.saturating_mul(2) {
                self.order.retain(|k| self.by_source.contains_key(k));
            }
            self.order.push_back(source_key);
        }
        self.by_item_key
            .entry(meta.item_key_hex.clone())
            .or_default()
            .push(source_key);
        self.by_source.insert(source_key, meta);
    }

    /// Drop all source mappings associated with item keys evicted by the Python cache.
    fn evict_item_keys<'a>(&mut self, keys: impl IntoIterator<Item = &'a str>) {
        for key in keys {
            if let Some(sources) = self.by_item_key.remove(key) {
                for src in sources {
                    self.by_source.remove(&src);
                }
            }
        }
    }
}

/// Compute a pre-decode cache key for immutable inline (`data:`) images + resize options.
/// Remote URLs (`http:`/`https:`) return `None` because their bytes may change remotely.
fn compute_source_key(modality: EncoderModality, item: &EncoderMediaItem) -> Option<[u8; 32]> {
    if modality != EncoderModality::Image {
        return None;
    }
    if item.url.len() < 5 || !item.url[..5].eq_ignore_ascii_case("data:") {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"smg-epd-src-v1\0");
    hasher.update(modality.as_str().as_bytes());
    let detail_tag = match item.detail.as_deref() {
        None | Some("") => 0u8,
        Some(d) if d.eq_ignore_ascii_case("auto") => 1u8,
        Some(d) if d.eq_ignore_ascii_case("low") => 2u8,
        Some(d) if d.eq_ignore_ascii_case("high") => 3u8,
        Some(_) => return None,
    };
    hasher.update(&[detail_tag]);
    hasher.update(&(item.url.len() as u64).to_le_bytes());
    hasher.update(item.url.as_bytes());
    Some(*hasher.finalize().as_bytes())
}

/// Media modality handled by one managed encoder request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderModality {
    Image,
    Video,
}

impl EncoderModality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
        }
    }

    pub fn from_proto(raw: &str) -> Result<Self, Status> {
        if raw.is_empty() || raw.eq_ignore_ascii_case("image") {
            Ok(Self::Image)
        } else if raw.eq_ignore_ascii_case("video") {
            Ok(Self::Video)
        } else {
            Err(Status::invalid_argument(
                "managed encoder supports image and video modalities only",
            ))
        }
    }
}

/// One media URL plus optional per-item preprocessing overrides forwarded from the gateway.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderMediaItem {
    pub url: String,
    pub detail: Option<String>,
}

pub type EncoderProcessor = Arc<
    dyn Fn(
            EncoderModality,
            Vec<EncoderMediaItem>,
        ) -> BoxFuture<'static, Result<EncoderArtifact, Status>>
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
    pub max_queued_control_bytes: usize,
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
    pub mm_item_limits: String,
    #[serde(flatten)]
    pub capabilities: Capabilities,
}

struct EncodeOwner {
    deadline: Instant,
    cancel: watch::Sender<bool>,
    result: watch::Sender<Option<Result<(), Status>>>,
    running: AtomicBool,
    queued_control_bytes: AtomicUsize,
}

#[derive(Default)]
struct OwnerLedger {
    owners: HashMap<String, Arc<EncodeOwner>>,
    queued_control_bytes: usize,
}

/// Rust owns E admission, CPU preprocessing, and execution-slot accounting.
/// Execution permits return only on an explicit native release receipt.
pub struct EncoderRuntime {
    config: EncoderConfig,
    max_queued_control_bytes: usize,
    native: NativeEncoderLink,
    processor: EncoderProcessor,
    owners: Mutex<OwnerLedger>,
    pre_decode_cache: Mutex<PreDecodeCache>,
    execution: Arc<Semaphore>,
    cpu: Arc<Semaphore>,
    draining: AtomicBool,
    quarantined: AtomicUsize,
    fatal: Mutex<Option<String>>,
    cancel_grace: Duration,
}

impl EncoderRuntime {
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
        let max_queued_control_bytes = if config.max_queued_control_bytes > 0 {
            config.max_queued_control_bytes
        } else {
            config.max_active.saturating_mul(config.max_control_bytes)
        };
        Ok(Arc::new(Self {
            execution: Arc::new(Semaphore::new(config.execution_capacity)),
            cpu: Arc::new(Semaphore::new(config.max_cpu)),
            max_queued_control_bytes,
            config,
            native,
            processor,
            owners: Mutex::new(OwnerLedger::default()),
            pre_decode_cache: Mutex::new(PreDecodeCache::new(PRE_DECODE_CACHE_MAX_ENTRIES)),
            draining: AtomicBool::new(false),
            quarantined: AtomicUsize::new(0),
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
        !self.draining.load(Ordering::Acquire)
            && self.quarantined.load(Ordering::Acquire) == 0
            && self.native.ready()
    }

    pub fn loads(&self) -> (usize, usize) {
        self.owners.lock().map_or((0, 0), |ledger| {
            ledger
                .owners
                .values()
                .fold((0, 0), |(running, waiting), owner| {
                    if owner.running.load(Ordering::Acquire) {
                        (running + 1, waiting)
                    } else {
                        (running, waiting + 1)
                    }
                })
        })
    }

    pub async fn encode(
        self: &Arc<Self>,
        request: EncodeRequest,
        remaining: Duration,
    ) -> Result<(), Status> {
        if !self.ready() {
            self.forget_unowned(&request.req_id);
            return Err(Status::unavailable("encoder is not ready"));
        }
        let (owner, modality) = match self.admit(&request, remaining) {
            Ok(admitted) => admitted,
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
            runtime.execute(request, modality, execution_owner).await;
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

    fn admit(
        &self,
        request: &EncodeRequest,
        remaining: Duration,
    ) -> Result<(Arc<EncodeOwner>, EncoderModality), Status> {
        let modality = EncoderModality::from_proto(&request.modality)?;
        let control_bytes = request.encoded_len();
        if remaining.is_zero()
            || remaining > Duration::from_secs(122)
            || request.req_id.is_empty()
            || request.req_id.len() > 256
            || request.num_parts == 0
            || request.num_parts as usize > self.config.max_images.max(2)
            || request.part_idx >= request.num_parts
            || request.mm_items.is_empty()
            || request.mm_items.len() > self.config.max_images
            || (!request.item_options.is_empty()
                && (modality != EncoderModality::Image
                    || request.item_options.len() != request.mm_items.len()
                    || request
                        .item_options
                        .iter()
                        .any(|opt| !is_valid_detail(&opt.detail))))
            || control_bytes > self.config.max_control_bytes
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
            queued_control_bytes: AtomicUsize::new(control_bytes),
        });
        let mut ledger = self
            .owners
            .lock()
            .map_err(|_| Status::internal("encoder ledger poisoned"))?;
        if ledger.owners.contains_key(&request.req_id) {
            return Err(Status::already_exists("duplicate native request"));
        }
        if self.draining.load(Ordering::Acquire) || ledger.owners.len() >= self.config.max_active {
            return Err(Status::resource_exhausted("encoder admission full"));
        }
        if ledger.queued_control_bytes.saturating_add(control_bytes) > self.max_queued_control_bytes
        {
            return Err(Status::resource_exhausted(
                "encoder queued control memory full",
            ));
        }
        ledger.queued_control_bytes = ledger.queued_control_bytes.saturating_add(control_bytes);
        ledger.owners.insert(request.req_id.clone(), owner.clone());
        Ok((owner, modality))
    }

    /// Release `owner`'s queued-control byte reservation once it enters execution or exits.
    fn release_queued_control(&self, owner: &EncodeOwner) {
        let bytes = owner.queued_control_bytes.swap(0, Ordering::AcqRel);
        if bytes > 0 {
            if let Ok(mut ledger) = self.owners.lock() {
                ledger.queued_control_bytes = ledger.queued_control_bytes.saturating_sub(bytes);
            }
        }
    }

    /// Remove `rid` from the active owner ledger after execution completes or is released.
    fn release_owner(&self, rid: &str) {
        if let Ok(mut ledger) = self.owners.lock() {
            if let Some(owner) = ledger.owners.remove(rid) {
                let bytes = owner.queued_control_bytes.swap(0, Ordering::AcqRel);
                ledger.queued_control_bytes = ledger.queued_control_bytes.saturating_sub(bytes);
            }
        }
    }

    /// Wake P's early facade waiters for a rid that no live owner holds.
    fn forget_unowned(&self, rid: &str) {
        if self
            .owners
            .lock()
            .is_ok_and(|ledger| !ledger.owners.contains_key(rid))
        {
            self.native.notify_forget(rid);
        }
    }

    /// Apply Python per-item cache evictions and resident keys reported on the execute receipt.
    fn apply_cache_feedback(
        &self,
        receipt: &Value,
        source_keys: &[Option<[u8; 32]>],
        item_metas: Option<&[([i64; 3], u32)]>,
    ) {
        let evicted = receipt.get("cache_evicted").and_then(Value::as_array);
        let resident = receipt.get("cache_resident").and_then(Value::as_array);
        let has_error = receipt.get("error").and_then(Value::as_str).is_some();
        if evicted.is_none() && (has_error || resident.is_none()) {
            return;
        }
        let Ok(mut cache) = self.pre_decode_cache.lock() else {
            return;
        };
        if let Some(evicted_keys) = evicted {
            cache.evict_item_keys(evicted_keys.iter().filter_map(Value::as_str));
        }
        if !has_error {
            if let (Some(resident_keys), Some(metas)) = (resident, item_metas) {
                if resident_keys.len() == source_keys.len() && metas.len() == source_keys.len() {
                    for ((src_key, key_val), &(grid_thw, token_count)) in
                        source_keys.iter().zip(resident_keys).zip(metas)
                    {
                        if let (Some(src), Some(key_hex)) = (*src_key, key_val.as_str()) {
                            if !key_hex.is_empty() {
                                cache.insert(
                                    src,
                                    CachedSourceMeta {
                                        item_key_hex: key_hex.to_owned(),
                                        grid_thw,
                                        token_count,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Enqueue an execute command to the native runtime and await its receipt or cancellation.
    async fn send_and_await_execute(
        &self,
        rid: &str,
        command: Value,
        binary_frame: Option<Bytes>,
        cancelled: &mut watch::Receiver<bool>,
        owner: &Arc<EncodeOwner>,
    ) -> (PendingReply, bool) {
        let header_bytes = serde_json::to_vec(&command)
            .map(|bytes| bytes.len().saturating_add(32))
            .unwrap_or(usize::MAX);
        if header_bytes > self.config.max_control_bytes {
            return (
                Err(NativeCallError::NotSent(Status::invalid_argument(
                    "serialized native encoder command exceeds max_control_bytes",
                ))),
                false,
            );
        }
        let mut timed_out_sent = false;
        let receipt: PendingReply = match self.native.enqueue(command, binary_frame).await {
            Ok((_pending, rx)) => {
                let mut exec = std::pin::pin!(async move {
                    rx.await.map_err(|_| {
                        NativeCallError::Ambiguous(Status::unavailable(
                            "native encoder reply stream closed",
                        ))
                    })?
                });
                let timed = tokio::select! {
                    receipt = &mut exec => Ok(receipt),
                    () = wait_cancel(cancelled, owner.deadline) => {
                        self.native.send_cancel(rid).await;
                        tokio::time::timeout(self.cancel_grace, &mut exec).await
                    }
                };
                match timed {
                    Ok(receipt) => receipt,
                    Err(_elapsed) => {
                        // A slow cancel/release may still own RDMA/staging memory: quarantine
                        // health, answer the caller promptly, and keep draining until Python
                        // confirms safe release so the worker can recover without crashing.
                        self.quarantined.fetch_add(1, Ordering::AcqRel);
                        timed_out_sent = true;
                        owner.result.send_replace(Some(Err(Status::unavailable(
                            "native encoder release receipt timed out",
                        ))));
                        let late = (&mut exec).await;
                        let late_safe = late.as_ref().is_ok_and(|r| {
                            r.get("released").and_then(Value::as_bool) == Some(true)
                        });
                        if !late_safe {
                            self.draining.store(true, Ordering::Release);
                        }
                        self.quarantined.fetch_sub(1, Ordering::AcqRel);
                        late
                    }
                }
            }
            Err(error) => Err(error),
        };
        (receipt, timed_out_sent)
    }

    /// Run bounded CPU preprocessing, dispatch native execution, and retain permits until `released=true`.
    async fn execute(
        self: Arc<Self>,
        request: EncodeRequest,
        modality: EncoderModality,
        owner: Arc<EncodeOwner>,
    ) {
        let mut cancelled = owner.cancel.subscribe();
        let EncodeRequest {
            req_id: rid,
            mm_items: raw_items,
            num_parts,
            part_idx,
            prefill_host,
            item_options: options,
            ..
        } = request;
        let item_count = raw_items.len();
        let permit = tokio::select! {
            biased;
            () = wait_cancel(&mut cancelled, owner.deadline) => None,
            permit = self.execution.clone().acquire_owned() => permit.ok(),
        };
        if permit.is_none() {
            self.release_owner(&rid);
            self.native.notify_forget(&rid);
            let status = if *owner.cancel.borrow() {
                Status::cancelled("encoder cancelled before execution")
            } else {
                Status::deadline_exceeded("encoder timed out waiting for execution slot")
            };
            owner.result.send_replace(Some(Err(status)));
            return;
        }
        self.release_queued_control(&owner);
        let items: Vec<EncoderMediaItem> = raw_items
            .into_iter()
            .enumerate()
            .map(|(index, url)| {
                let option = options.get(index);
                EncoderMediaItem {
                    url,
                    detail: option.and_then(|o| (!o.detail.is_empty()).then(|| o.detail.clone())),
                }
            })
            .collect();

        let can_pre_decode_cache =
            modality == EncoderModality::Image && self.config.capabilities.tp_size == 1;
        let source_keys: Vec<Option<[u8; 32]>> = if can_pre_decode_cache {
            items
                .iter()
                .map(|item| compute_source_key(modality, item))
                .collect()
        } else {
            Vec::new()
        };

        // Fast path: if every item is a known warm-cache hit, skip Rust decode/preprocess and
        // pixel IPC altogether. If Python reports a race eviction (`cache_miss=true`), fall
        // through seamlessly to the normal decode path under the same owner and permit.
        let cached_hit = if can_pre_decode_cache {
            self.pre_decode_cache
                .lock()
                .ok()
                .and_then(|cache| cache.lookup_all(&source_keys))
        } else {
            None
        };
        if let Some(PreDecodeHit {
            item_keys: hit_keys,
            grid_thw: hit_grids,
            token_counts: hit_counts,
            embedding_tokens: hit_tokens,
        }) = cached_hit
        {
            owner.running.store(true, Ordering::Release);
            let remaining_ms = owner
                .deadline
                .saturating_duration_since(Instant::now())
                .as_millis() as u64;
            if *owner.cancel.borrow() || remaining_ms == 0 {
                self.release_owner(&rid);
                drop(permit);
                self.native.notify_forget(&rid);
                let status = if *owner.cancel.borrow() {
                    Status::cancelled("encoder cancelled before cache hit dispatch")
                } else {
                    Status::deadline_exceeded("encoder expired before cache hit dispatch")
                };
                owner.result.send_replace(Some(Err(status)));
                return;
            }
            let (receipt, timed_out_sent) = self
                .send_and_await_execute(
                    &rid,
                    serde_json::json!({
                        "operation": "execute",
                        "rid": rid,
                        "prefill_host": prefill_host,
                        "remaining_ms": remaining_ms,
                        "image_count": item_count,
                        "item_count": item_count,
                        "modality": modality.as_str(),
                        "num_parts": num_parts,
                        "part_idx": part_idx,
                        "embedding_tokens": hit_tokens,
                        "artifact_v2": {
                            "kind": "cache_hit",
                            "item_keys": hit_keys,
                            "grid_thw": hit_grids,
                            "token_counts": hit_counts,
                        },
                    }),
                    None,
                    &mut cancelled,
                    &owner,
                )
                .await;
            let is_cache_miss_retry = !timed_out_sent
                && receipt.as_ref().is_ok_and(|val| {
                    val.get("released").and_then(Value::as_bool) == Some(true)
                        && val.get("cache_miss").and_then(Value::as_bool) == Some(true)
                });
            if is_cache_miss_retry {
                if let Ok(ref val) = receipt {
                    if let Ok(mut cache) = self.pre_decode_cache.lock() {
                        if let Some(evicted) = val.get("cache_evicted").and_then(Value::as_array) {
                            cache.evict_item_keys(evicted.iter().filter_map(Value::as_str));
                        } else {
                            cache.evict_item_keys(hit_keys.iter().map(String::as_str));
                        }
                    }
                }
                if *owner.cancel.borrow() || Instant::now() >= owner.deadline {
                    self.release_owner(&rid);
                    drop(permit);
                    self.native.notify_forget(&rid);
                    let status = if *owner.cancel.borrow() {
                        Status::cancelled("encoder cancelled after cache miss")
                    } else {
                        Status::deadline_exceeded("encoder expired after cache miss")
                    };
                    owner.result.send_replace(Some(Err(status)));
                    return;
                }
            } else {
                if let Ok(ref val) = receipt {
                    if val.get("released").and_then(Value::as_bool) == Some(true) {
                        self.apply_cache_feedback(val, &source_keys, None);
                    }
                }
                self.finish_execute(&rid, permit, &owner, receipt, timed_out_sent);
                return;
            }
        }

        let prepared = async {
            let cpu = tokio::select! {
                biased;
                () = wait_cancel(&mut cancelled, owner.deadline) => {
                    return Err(if *owner.cancel.borrow() {
                        Status::cancelled("encoder cancelled while waiting for CPU")
                    } else {
                        Status::deadline_exceeded("encoder timed out waiting for CPU")
                    });
                }
                permit = self.cpu.clone().acquire_owned() => permit
                    .map_err(|_| Status::unavailable("encoder CPU supply closed"))?,
            };
            owner.running.store(true, Ordering::Release);
            if *owner.cancel.borrow() {
                return Err(Status::cancelled("encoder cancelled before preprocessing"));
            }
            if Instant::now() >= owner.deadline {
                return Err(Status::deadline_exceeded(
                    "encoder expired before preprocessing",
                ));
            }
            let artifact = (self.processor)(modality, items).await;
            drop(cpu);
            let artifact = artifact?;
            let remaining_ms = owner
                .deadline
                .saturating_duration_since(Instant::now())
                .as_millis() as u64;
            if *owner.cancel.borrow() {
                return Err(Status::cancelled("encoder cancelled after preprocessing"));
            }
            if remaining_ms == 0 {
                return Err(Status::deadline_exceeded(
                    "encoder expired after preprocessing",
                ));
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

        let mut command = serde_json::json!({
            "operation": "execute",
            "rid": rid,
            "prefill_host": prefill_host,
            "remaining_ms": remaining_ms,
            "image_count": item_count,
            "item_count": item_count,
            "modality": modality.as_str(),
            "num_parts": num_parts,
            "part_idx": part_idx,
            "embedding_tokens": artifact.embedding_tokens,
        });
        let (binary_frame, _shm_guard, compact_item_metas) = match artifact.payload {
            EncoderPayload::LegacyProto(bytes) => (Some(bytes), None, None),
            EncoderPayload::Compact {
                batches,
                shm_name,
                shm_nbytes,
                shm_guard,
                inline_pixels,
            } => {
                let metas = if can_pre_decode_cache {
                    let flat: Vec<([i64; 3], u32)> = batches
                        .iter()
                        .filter(|b| b.grid_thw.len() == b.placeholder_lengths.len())
                        .flat_map(|b| {
                            b.grid_thw
                                .iter()
                                .copied()
                                .zip(b.placeholder_lengths.iter().copied())
                        })
                        .collect();
                    (flat.len() == item_count).then_some(flat)
                } else {
                    None
                };
                if let Some(fields) = command.as_object_mut() {
                    fields.insert(
                        "artifact_v2".into(),
                        serde_json::json!({
                            "kind": "compact",
                            "shm_name": shm_name,
                            "shm_nbytes": shm_nbytes,
                            "batches": batches,
                        }),
                    );
                }
                (inline_pixels, shm_guard, metas)
            }
        };

        // Enqueue `execute` on the FIFO command channel before racing `cancel`, so Python
        // always observes `execute` before any subsequent `cancel` for the same `rid`.
        let (receipt, timed_out_sent) = self
            .send_and_await_execute(&rid, command, binary_frame, &mut cancelled, &owner)
            .await;
        if let Ok(ref val) = receipt {
            if val.get("released").and_then(Value::as_bool) == Some(true) {
                self.apply_cache_feedback(val, &source_keys, compact_item_metas.as_deref());
            }
        }
        self.finish_execute(&rid, permit, &owner, receipt, timed_out_sent);
    }

    /// Finalize an encode execution, releasing admission state and publishing the result.
    fn finish_execute(
        &self,
        rid: &str,
        permit: Option<tokio::sync::OwnedSemaphorePermit>,
        owner: &Arc<EncodeOwner>,
        receipt: PendingReply,
        timed_out_sent: bool,
    ) {
        if let Err(NativeCallError::NotSent(error)) = receipt {
            self.release_owner(rid);
            drop(permit);
            self.native.notify_forget(&rid);
            if !timed_out_sent {
                owner.result.send_replace(Some(Err(error)));
            }
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
            self.release_owner(rid);
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
        if !timed_out_sent {
            owner.result.send_replace(Some(outcome));
        }
    }

    pub fn drain(&self) {
        self.draining.store(true, Ordering::Release);
        if let Ok(ledger) = self.owners.lock() {
            for owner in ledger.owners.values() {
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

fn is_valid_detail(detail: &str) -> bool {
    detail.is_empty()
        || detail.eq_ignore_ascii_case("auto")
        || detail.eq_ignore_ascii_case("low")
        || detail.eq_ignore_ascii_case("high")
}

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

    fn capability(&self) -> Capabilities {
        let mut cap = self.runtime.config.capabilities.clone();
        cap.role = "encoder".into();
        normalized_capabilities(cap, self.runtime.ready())
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
        let mut args = serde_json::json!({
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
        if !c.mm_item_limits.is_empty() {
            args["mm_item_limits"] = Value::String(c.mm_item_limits.clone());
        }
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
                    config.max_active.saturating_mul(16).clamp(64, 65536),
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

    pub fn drain(&self) {
        self.runtime.drain();
    }

    pub fn active(&self) -> usize {
        let (running, waiting) = self.runtime.loads();
        running + waiting
    }

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
            max_queued_control_bytes: 0,
            control_listen: String::new(),
            native_commands: String::new(),
            native_events: String::new(),
            model_path: "model".into(),
            tokenizer_path: "tok".into(),
            served_model_name: "model".into(),
            model_type: "qwen3_vl".into(),
            mm_item_limits: String::new(),
            capabilities: Capabilities {
                role: "encoder".into(),
                model_revision: "rev1".into(),
                processor_contract: "smg-image-grid-v1".into(),
                embedding_dtype: "bfloat16".into(),
                kv_backend: "mooncake".into(),
                embedding_backend: "mooncake".into(),
                topology_contract: "single-host-tp1".into(),
                receive_endpoint: "http://127.0.0.1:9000".into(),
                tp_size: 1,
                ..Default::default()
            },
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
            modality: String::new(),
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
        async fn start_with_config(
            config: EncoderConfig,
            connect_pull: bool,
            cancel_grace: Duration,
            processor: Option<EncoderProcessor>,
        ) -> Self {
            Self::start_with_config(test_config(), connect_pull, cancel_grace, processor).await
        }

        async fn start_with_config(
            config: EncoderConfig,
            connect_pull: bool,
            cancel_grace: Duration,
            processor: Option<EncoderProcessor>,
        ) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let commands = format!("ipc://{}/commands.sock", dir.path().display());
            let events = format!("ipc://{}/events.sock", dir.path().display());
            let channel_cap = config.max_active.saturating_mul(16).clamp(64, 65536);
            let link = NativeEncoderLink::bind(&commands, &events, channel_cap)
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
                Arc::new(|_, _| {
                    Box::pin(async {
                        Ok(EncoderArtifact::legacy(Bytes::from_static(b"pixels"), 4))
                    })
                })
            });
            let runtime =
                EncoderRuntime::with_cancel_grace(config, link, processor, cancel_grace).unwrap();
            assert!(!runtime.ready());
            Self {
                _dir: dir,
                runtime,
                pull,
                push,
            }
        }

        async fn start(
            connect_pull: bool,
            cancel_grace: Duration,
            processor: Option<EncoderProcessor>,
        ) -> Self {
            Self::start_with_config(test_config(), connect_pull, cancel_grace, processor).await
        }

        async fn send_reply(&mut self, value: Value) {
            let bytes = serde_json::to_vec(&value).unwrap();
            self.push
                .send(ZmqMessage::from(Bytes::from(bytes)))
                .await
                .unwrap();
        }

        async fn wait_for(&self, mut pred: impl FnMut() -> bool) {
            for _ in 0..50 {
                if pred() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }

        async fn mark_ready(&mut self) {
            self.send_reply(json!({"ready": true})).await;
            self.wait_for(|| self.runtime.ready()).await;
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
        let exec1 = recv_cmd(h.pull.as_mut().unwrap()).await;
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
        let forget2 = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(forget2["operation"], "forget");
        assert_eq!(forget2["rid"], "req-2");

        // Cancelling in-flight req-1 sends cancel wakeup but retains slot until released=true.
        h1.abort();
        let cancel1 = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(cancel1["operation"], "cancel");
        assert_eq!(cancel1["rid"], "req-1");
        assert_eq!(h.runtime.loads().0, 1);

        // Unconfirmed release drains runtime and retains the owner.
        h.send_reply(json!({"sequence": exec1["sequence"], "result": {"released": false}}))
            .await;
        h.wait_for(|| h.runtime.fatal_error().is_some()).await;
        assert!(!h.runtime.ready());
        assert!(h.runtime.fatal_error().is_some());
        assert_eq!(h.runtime.loads().0, 1);
    }

    #[tokio::test]
    async fn unsent_command_releases_slot_without_quarantine() {
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
    async fn cancel_receipt_timeout_quarantines_and_recovers_on_late_release() {
        let mut h = EncoderHarness::start(true, Duration::from_millis(80), None).await;
        h.mark_ready().await;

        let handle = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("stuck-req"), Duration::from_secs(5))
                    .await
            }
        });
        let exec = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(exec["operation"], "execute");

        handle.abort();
        let cancel = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(cancel["operation"], "cancel");

        h.wait_for(|| !h.runtime.ready()).await;
        assert!(!h.runtime.ready());
        assert!(h.runtime.fatal_error().is_none());
        assert_eq!(h.runtime.loads().0, 1);

        let rejected = h
            .runtime
            .encode(encode_request("while-quarantined"), Duration::from_secs(5))
            .await
            .unwrap_err();
        assert_eq!(rejected.code(), tonic::Code::Unavailable);

        // Late release clears quarantine and returns the slot.
        h.send_reply(
            json!({"sequence": exec["sequence"], "result": {"released": true, "error": "cancelled"}}),
        )
        .await;
        h.wait_for(|| h.runtime.ready() && h.runtime.loads().0 == 0)
            .await;
        assert!(h.runtime.ready());
        assert!(h.runtime.fatal_error().is_none());
        assert_eq!(h.runtime.loads(), (0, 0));
    }

    #[tokio::test]
    async fn unconfirmed_release_receipt_marks_fatal_and_retains_slot() {
        let mut h = EncoderHarness::start(true, Duration::from_millis(80), None).await;
        h.mark_ready().await;

        let handle = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("unconfirmed-req"), Duration::from_secs(5))
                    .await
            }
        });
        let exec = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(exec["operation"], "execute");

        handle.abort();
        let cancel = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(cancel["operation"], "cancel");

        h.wait_for(|| !h.runtime.ready()).await;
        assert!(!h.runtime.ready());
        assert!(h.runtime.fatal_error().is_none());

        h.send_reply(
            json!({"sequence": exec["sequence"], "result": {"released": false, "error": "busy"}}),
        )
        .await;
        h.wait_for(|| h.runtime.fatal_error().is_some()).await;
        assert!(!h.runtime.ready());
        assert!(h.runtime.fatal_error().is_some());
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
        let processor: EncoderProcessor = Arc::new(move |modality, items| {
            seen_proc.lock().unwrap().push((modality, items));
            Box::pin(async { Ok(EncoderArtifact::legacy(Bytes::from_static(b"pixels"), 4)) })
        });
        let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, Some(processor)).await;
        h.mark_ready().await;

        for (rid, mutate) in [
            (
                "bad-len",
                Box::new(|r: &mut EncodeRequest| {
                    r.item_options = vec![ep::ImageOption::default(); 2];
                }) as Box<dyn FnOnce(&mut EncodeRequest)>,
            ),
            (
                "bad-detail",
                Box::new(|r: &mut EncodeRequest| {
                    r.item_options.push(ep::ImageOption {
                        detail: "bogus".into(),
                    });
                }),
            ),
            (
                "bad-modality",
                Box::new(|r: &mut EncodeRequest| r.modality = "audio".into()),
            ),
            (
                "video-with-opts",
                Box::new(|r: &mut EncodeRequest| {
                    r.modality = "video".into();
                    r.item_options.push(ep::ImageOption::default());
                }),
            ),
        ] {
            let mut req = encode_request(rid);
            mutate(&mut req);
            assert_eq!(
                h.runtime
                    .encode(req, Duration::from_secs(5))
                    .await
                    .unwrap_err()
                    .code(),
                tonic::Code::InvalidArgument,
                "{rid}"
            );
            let forget = recv_cmd(h.pull.as_mut().unwrap()).await;
            assert_eq!(forget["operation"], "forget");
            assert_eq!(forget["rid"], rid);
        }
        assert!(seen.lock().unwrap().is_empty());

        let req = EncodeRequest {
            req_id: "opts-ok".into(),
            mm_items: vec![
                "https://example.test/a.png".into(),
                "https://example.test/b.png".into(),
            ],
            num_parts: 2,
            part_idx: 0,
            prefill_host: "127.0.0.1".into(),
            embedding_port: vec![],
            item_options: vec![
                ep::ImageOption {
                    detail: "high".into(),
                },
                ep::ImageOption {
                    detail: "auto".into(),
                },
            ],
            modality: "image".into(),
        };
        let task = tokio::spawn({
            let rt = h.runtime.clone();
            async move { rt.encode(req, Duration::from_secs(5)).await }
        });
        let exec = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(
            (
                &exec["operation"],
                &exec["modality"],
                &exec["num_parts"],
                &exec["part_idx"]
            ),
            (&json!("execute"), &json!("image"), &json!(2), &json!(0))
        );
        h.send_reply(json!({"sequence": exec["sequence"], "result": {"released": true}}))
            .await;
        task.await.unwrap().unwrap();

        let video_req = EncodeRequest {
            req_id: "video-ok".into(),
            mm_items: vec!["https://example.test/clip.mp4".into()],
            num_parts: 2,
            part_idx: 1,
            prefill_host: "127.0.0.1".into(),
            embedding_port: vec![],
            item_options: vec![],
            modality: "video".into(),
        };
        let video_task = tokio::spawn({
            let rt = h.runtime.clone();
            async move { rt.encode(video_req, Duration::from_secs(5)).await }
        });
        let video_exec = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(
            (
                &video_exec["operation"],
                &video_exec["modality"],
                &video_exec["num_parts"],
                &video_exec["part_idx"],
                &video_exec["item_count"]
            ),
            (
                &json!("execute"),
                &json!("video"),
                &json!(2),
                &json!(1),
                &json!(1)
            )
        );
        h.send_reply(json!({"sequence": video_exec["sequence"], "result": {"released": true}}))
            .await;
        video_task.await.unwrap().unwrap();

        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[
                (
                    EncoderModality::Image,
                    vec![
                        EncoderMediaItem {
                            url: "https://example.test/a.png".into(),
                            detail: Some("high".into()),
                        },
                        EncoderMediaItem {
                            url: "https://example.test/b.png".into(),
                            detail: Some("auto".into()),
                        },
                    ],
                ),
                (
                    EncoderModality::Video,
                    vec![EncoderMediaItem {
                        url: "https://example.test/clip.mp4".into(),
                        detail: None,
                    }],
                ),
            ]
        );
    }

    #[tokio::test]
    async fn compact_artifact_and_pre_decode_cache_hit_skips_processor() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_proc = calls.clone();
        let processor: EncoderProcessor = Arc::new(move |_modality, items| {
            calls_proc.fetch_add(1, Ordering::SeqCst);
            let count = items.len();
            Box::pin(async move {
                let hashes = (0..count).map(|i| format!("{i:064x}")).collect();
                Ok(EncoderArtifact {
                    payload: EncoderPayload::Compact {
                        batches: vec![NativeBatchMeta {
                            pixel_dtype: "bfloat16".into(),
                            pixel_shape: vec![4 * count as u32, 1],
                            pixel_offset: 0,
                            pixel_nbytes: 8 * count,
                            grid_thw: vec![[1, 2, 2]; count],
                            placeholder_lengths: vec![4; count],
                            second_per_grid_ts: None,
                            second_per_grid_shape: Vec::new(),
                            item_hashes: hashes,
                        }],
                        shm_name: None,
                        shm_nbytes: 0,
                        shm_guard: None,
                        inline_pixels: Some(Bytes::from(vec![0u8; 8 * count])),
                    },
                    embedding_tokens: 4 * count as u64,
                })
            })
        });
        let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, Some(processor)).await;
        h.mark_ready().await;

        // 1. Cold request populates PreDecodeCache via cache_resident feedback.
        let t1 = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("cold-1"), Duration::from_secs(5))
                    .await
            }
        });
        let cmd1 = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(cmd1["artifact_v2"]["kind"], "compact");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let resident_key = "11".repeat(32);
        h.send_reply(json!({
            "sequence": cmd1["sequence"],
            "result": {"released": true, "cache_resident": [resident_key]}
        }))
        .await;
        t1.await.unwrap().unwrap();

        // 2. Identical data: image hits PreDecodeCache without invoking processor.
        let t2 = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("warm-2"), Duration::from_secs(5))
                    .await
            }
        });
        let cmd2 = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(cmd2["artifact_v2"]["kind"], "cache_hit");
        assert_eq!(cmd2["artifact_v2"]["item_keys"], json!([resident_key]));
        assert_eq!(cmd2["artifact_v2"]["grid_thw"], json!([[1, 2, 2]]));
        assert_eq!(cmd2["artifact_v2"]["token_counts"], json!([4]));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        h.send_reply(json!({"sequence": cmd2["sequence"], "result": {"released": true}}))
            .await;
        t2.await.unwrap().unwrap();

        // 3. Different detail option must NOT collide with cached source key.
        let mut diff_opt = encode_request("diff-opt-3");
        diff_opt.item_options.push(ep::ImageOption {
            detail: "high".into(),
        });
        let t3 = tokio::spawn({
            let rt = h.runtime.clone();
            async move { rt.encode(diff_opt, Duration::from_secs(5)).await }
        });
        let cmd3 = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(cmd3["artifact_v2"]["kind"], "compact");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        h.send_reply(json!({"sequence": cmd3["sequence"], "result": {"released": true}}))
            .await;
        t3.await.unwrap().unwrap();

        // 4. Remote HTTP URLs are never cached in PreDecodeCache.
        let mut http_req = encode_request("http-4");
        http_req.mm_items = vec!["https://example.test/img.png".into()];
        let http_req4 = http_req.clone();
        let t4 = tokio::spawn({
            let rt = h.runtime.clone();
            async move { rt.encode(http_req4, Duration::from_secs(5)).await }
        });
        let cmd4 = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(cmd4["artifact_v2"]["kind"], "compact");
        h.send_reply(json!({
            "sequence": cmd4["sequence"],
            "result": {"released": true, "cache_resident": ["22".repeat(32)]}
        }))
        .await;
        t4.await.unwrap().unwrap();

        http_req.req_id = "http-5".into();
        let t5 = tokio::spawn({
            let rt = h.runtime.clone();
            async move { rt.encode(http_req, Duration::from_secs(5)).await }
        });
        let cmd5 = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(cmd5["artifact_v2"]["kind"], "compact");
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        h.send_reply(json!({"sequence": cmd5["sequence"], "result": {"released": true}}))
            .await;
        t5.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn pre_decode_cache_eviction_and_race_miss_fallback_to_full_decode() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_proc = calls.clone();
        let processor: EncoderProcessor = Arc::new(move |_, _| {
            calls_proc.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Ok(EncoderArtifact {
                    payload: EncoderPayload::Compact {
                        batches: vec![NativeBatchMeta {
                            pixel_dtype: "bfloat16".into(),
                            pixel_shape: vec![4, 1],
                            pixel_offset: 0,
                            pixel_nbytes: 8,
                            grid_thw: vec![[1, 2, 2]],
                            placeholder_lengths: vec![4],
                            second_per_grid_ts: None,
                            second_per_grid_shape: Vec::new(),
                            item_hashes: vec!["aa".repeat(32)],
                        }],
                        shm_name: None,
                        shm_nbytes: 0,
                        shm_guard: None,
                        inline_pixels: Some(Bytes::from_static(&[0u8; 8])),
                    },
                    embedding_tokens: 4,
                })
            })
        });
        let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, Some(processor)).await;
        h.mark_ready().await;

        let resident_key = "ab".repeat(32);
        // Populate PreDecodeCache.
        let t1 = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("seed"), Duration::from_secs(5))
                    .await
            }
        });
        let cmd1 = recv_cmd(h.pull.as_mut().unwrap()).await;
        h.send_reply(json!({
            "sequence": cmd1["sequence"],
            "result": {"released": true, "cache_resident": [resident_key]}
        }))
        .await;
        t1.await.unwrap().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Simulate LRU eviction race on Python: Python returns cache_miss=true; Rust falls back
        // to full decode under the exact same rid and completes cleanly.
        let t2 = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("race-req"), Duration::from_secs(5))
                    .await
            }
        });
        let probe = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(probe["artifact_v2"]["kind"], "cache_hit");
        assert_eq!(probe["rid"], "race-req");
        h.send_reply(json!({
            "sequence": probe["sequence"],
            "result": {"released": true, "cache_miss": true, "cache_evicted": [resident_key]}
        }))
        .await;

        let fallback = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(fallback["artifact_v2"]["kind"], "compact");
        assert_eq!(fallback["rid"], "race-req");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        h.send_reply(json!({
            "sequence": fallback["sequence"],
            "result": {"released": true, "cache_resident": [resident_key]}
        }))
        .await;
        t2.await.unwrap().unwrap();

        // Explicit cache_evicted feedback clears PreDecodeCache before the next request.
        let t3 = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("evict-trig"), Duration::from_secs(5))
                    .await
            }
        });
        let hit3 = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(hit3["artifact_v2"]["kind"], "cache_hit");
        h.send_reply(json!({
            "sequence": hit3["sequence"],
            "result": {"released": true, "cache_evicted": [resident_key]}
        }))
        .await;
        t3.await.unwrap().unwrap();

        let t4 = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("post-evict"), Duration::from_secs(5))
                    .await
            }
        });
        let miss4 = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(miss4["artifact_v2"]["kind"], "compact");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        h.send_reply(json!({"sequence": miss4["sequence"], "result": {"released": true}}))
            .await;
        t4.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn shm_guard_unlinks_after_release_and_on_unsent_error() {
        let dir = tempfile::tempdir().unwrap();
        let path1 = dir.path().join("smg-epd-test-1");
        std::fs::write(&path1, b"pixel-bytes").unwrap();
        let path1_proc = path1.clone();
        let processor: EncoderProcessor = Arc::new(move |_, _| {
            let p = path1_proc.clone();
            Box::pin(async move {
                Ok(EncoderArtifact {
                    payload: EncoderPayload::Compact {
                        batches: vec![NativeBatchMeta {
                            pixel_dtype: "bfloat16".into(),
                            pixel_shape: vec![4, 1],
                            pixel_offset: 0,
                            pixel_nbytes: 8,
                            grid_thw: vec![[1, 2, 2]],
                            placeholder_lengths: vec![4],
                            second_per_grid_ts: None,
                            second_per_grid_shape: Vec::new(),
                            item_hashes: vec!["cc".repeat(32)],
                        }],
                        shm_name: Some("smg-epd-test-1".into()),
                        shm_nbytes: 8,
                        shm_guard: Some(ShmGuard::new(p)),
                        inline_pixels: None,
                    },
                    embedding_tokens: 4,
                })
            })
        });
        let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, Some(processor)).await;
        h.mark_ready().await;

        let task = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("shm-ok"), Duration::from_secs(5))
                    .await
            }
        });
        let cmd = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(cmd["artifact_v2"]["shm_name"], "smg-epd-test-1");
        assert!(path1.exists(), "SHM file must stay alive while in-flight");
        h.send_reply(json!({"sequence": cmd["sequence"], "result": {"released": true}}))
            .await;
        task.await.unwrap().unwrap();
        assert!(
            !path1.exists(),
            "ShmGuard must unlink SHM file after release"
        );
    }

    #[test]
    fn pre_decode_cache_compacts_evicted_order_entries_under_churn() {
        let mut cache = PreDecodeCache::new(4);
        for i in 0u32..50 {
            let mut src = [0u8; 32];
            src[..4].copy_from_slice(&i.to_le_bytes());
            let key = format!("{i:064x}");
            cache.insert(
                src,
                CachedSourceMeta {
                    item_key_hex: key.clone(),
                    grid_thw: [1, 2, 2],
                    token_count: 4,
                },
            );
            cache.evict_item_keys([key.as_str()]);
        }
        assert!(cache.by_source.is_empty());
        assert!(
            cache.order.len() <= cache.max_entries * 2,
            "order queue must stay bounded under LRU churn (len={})",
            cache.order.len()
        );
    }

    #[tokio::test]
    async fn burst_128_queues_behind_execution_capacity_without_resource_exhausted() {
        let mut cfg = test_config();
        cfg.max_active = 256;
        cfg.execution_capacity = 8;
        cfg.max_cpu = 8;
        let mut h = EncoderHarness::start_with_config(cfg, true, CANCEL_RECEIPT_GRACE, None).await;
        h.mark_ready().await;

        let mut tasks = Vec::with_capacity(128);
        for idx in 0..128 {
            let rt = h.runtime.clone();
            tasks.push(tokio::spawn(async move {
                rt.encode(
                    encode_request(&format!("burst-{idx}")),
                    Duration::from_secs(10),
                )
                .await
            }));
        }

        // All 128 requests must be admitted (running <= 8, waiting >= 120), never RESOURCE_EXHAUSTED.
        h.wait_for(|| {
            let (running, waiting) = h.runtime.loads();
            running + waiting == 128
        })
        .await;
        let (running, waiting) = h.runtime.loads();
        assert_eq!(running + waiting, 128);
        assert!(
            running <= 8,
            "running {running} exceeded execution_capacity=8"
        );
        assert!(
            waiting >= 120,
            "expected queued waiters, got waiting={waiting}"
        );

        // Drain all 128 executes in waves of up to execution_capacity, verifying bounded concurrency.
        let mut completed = 0usize;
        let mut in_flight: Vec<u64> = Vec::new();
        let mut peak_inflight = 0usize;
        while completed < 128 {
            while in_flight.len() < 8 && completed + in_flight.len() < 128 {
                let cmd = recv_cmd(h.pull.as_mut().unwrap()).await;
                assert_eq!(cmd["operation"], "execute");
                in_flight.push(cmd["sequence"].as_u64().unwrap());
                peak_inflight = peak_inflight.max(in_flight.len());
            }
            let (r, _) = h.runtime.loads();
            assert!(r <= 8, "running {r} exceeded execution_capacity=8");
            for seq in in_flight.drain(..) {
                h.send_reply(json!({"sequence": seq, "result": {"released": true}}))
                    .await;
                completed += 1;
            }
        }
        assert_eq!(peak_inflight, 8);

        for task in tasks {
            task.await.unwrap().expect("burst request must succeed");
        }
        assert_eq!(h.runtime.loads(), (0, 0));
    }

    #[tokio::test]
    async fn queue_enforces_hard_bound_and_releases_on_cancel_or_deadline() {
        let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, None).await;
        h.mark_ready().await;

        // Fill max_active = 4 (1 running + 3 queued).
        let h1 = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("q-1"), Duration::from_secs(5))
                    .await
            }
        });
        let exec1 = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(exec1["operation"], "execute");

        let h2 = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("q-2"), Duration::from_secs(5))
                    .await
            }
        });
        let h3 = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("q-3-short"), Duration::from_millis(60))
                    .await
            }
        });
        let h4 = tokio::spawn({
            let rt = h.runtime.clone();
            async move {
                rt.encode(encode_request("q-4"), Duration::from_secs(5))
                    .await
            }
        });
        h.wait_for(|| h.runtime.loads() == (1, 3)).await;
        assert_eq!(h.runtime.loads(), (1, 3));

        // 5th request exceeds hard max_active=4 -> ResourceExhausted.
        let overflow = h
            .runtime
            .encode(encode_request("q-5-full"), Duration::from_secs(5))
            .await
            .unwrap_err();
        assert_eq!(overflow.code(), tonic::Code::ResourceExhausted);
        let forget_full = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(forget_full["operation"], "forget");
        assert_eq!(forget_full["rid"], "q-5-full");

        // Cancel q-2 and let q-3-short expire while queued; both release immediately without running.
        h2.abort();
        let err3 = h3.await.unwrap().unwrap_err();
        assert_eq!(err3.code(), tonic::Code::DeadlineExceeded);
        h.wait_for(|| h.runtime.loads() == (1, 1)).await;
        assert_eq!(h.runtime.loads(), (1, 1));

        let mut forgets = Vec::new();
        for _ in 0..2 {
            let cmd = recv_cmd(h.pull.as_mut().unwrap()).await;
            assert_eq!(cmd["operation"], "forget");
            forgets.push(cmd["rid"].as_str().unwrap().to_owned());
        }
        forgets.sort();
        assert_eq!(forgets, vec!["q-2", "q-3-short"]);

        // Releasing q-1 lets live q-4 proceed directly (never stolen by cancelled/expired waiters).
        h.send_reply(json!({"sequence": exec1["sequence"], "result": {"released": true}}))
            .await;
        h1.await.unwrap().unwrap();
        let exec4 = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(exec4["operation"], "execute");
        assert_eq!(exec4["rid"], "q-4");
        h.send_reply(json!({"sequence": exec4["sequence"], "result": {"released": true}}))
            .await;
        h4.await.unwrap().unwrap();
        assert_eq!(h.runtime.loads(), (0, 0));
    }

    #[tokio::test]
    async fn admit_accepts_multi_encoder_part_counts_up_to_max_images() {
        let mut h = EncoderHarness::start(true, CANCEL_RECEIPT_GRACE, None).await;
        h.mark_ready().await;

        let mut req = encode_request("part-3-of-4");
        req.num_parts = 4;
        req.part_idx = 3;
        let task = tokio::spawn({
            let rt = h.runtime.clone();
            async move { rt.encode(req, Duration::from_secs(5)).await }
        });
        let exec = recv_cmd(h.pull.as_mut().unwrap()).await;
        assert_eq!(
            (&exec["num_parts"], &exec["part_idx"]),
            (&json!(4), &json!(3))
        );
        h.send_reply(json!({"sequence": exec["sequence"], "result": {"released": true}}))
            .await;
        task.await.unwrap().unwrap();

        // Exceeding max_images (4 in test_config) or part_idx >= num_parts is still rejected.
        for (rid, num_parts, part_idx) in [("zero", 0, 0), ("oob-idx", 4, 4), ("too-many", 5, 0)] {
            let mut bad = encode_request(rid);
            bad.num_parts = num_parts;
            bad.part_idx = part_idx;
            assert_eq!(
                h.runtime
                    .encode(bad, Duration::from_secs(5))
                    .await
                    .unwrap_err()
                    .code(),
                tonic::Code::InvalidArgument,
                "{rid}"
            );
        }
    }

    #[tokio::test]
    async fn encoder_enforces_shared_queued_control_byte_budget() {
        let mut config = test_config();
        config.max_active = 8;
        config.execution_capacity = 1;
        config.max_control_bytes = 2048;
        let large_req = |id: &str| EncodeRequest {
            req_id: id.into(),
            mm_items: vec![format!("data:image/png;base64,{}", "A".repeat(400))],
            num_parts: 1,
            part_idx: 0,
            prefill_host: "127.0.0.1".into(),
            embedding_port: vec![],
            item_options: vec![],
            modality: "image".into(),
        };
        let req_len = large_req("r1").encoded_len();
        // Allow exactly 1 large request queued while another holds the single execution slot.
        config.max_queued_control_bytes = req_len + 16;

        let mut h =
            EncoderHarness::start_with_config(config, true, Duration::from_secs(1), None).await;
        h.mark_ready().await;
        // r1 acquires the execution permit and releases its queued-control reservation.
        let r1 = {
            let rt = h.runtime.clone();
            tokio::spawn(async move { rt.encode(large_req("r1"), Duration::from_secs(5)).await })
        };
        let r1_cmd = recv_cmd(h.pull.as_mut().unwrap()).await;

        // r2 queues behind r1, consuming almost all of max_queued_control_bytes.
        let r2 = {
            let rt = h.runtime.clone();
            tokio::spawn(async move { rt.encode(large_req("r2"), Duration::from_secs(5)).await })
        };
        tokio::time::sleep(Duration::from_millis(30)).await;

        // r3 exceeds the shared queued-control byte budget even though max_active=8.
        let err = h
            .runtime
            .encode(large_req("r3"), Duration::from_secs(5))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::ResourceExhausted);
        assert!(err.message().contains("queued control"));

        // Releasing r1 promotes r2 into execution (freeing r2's queued reservation), so r3 now admits.
        h.send_reply(json!({"sequence": r1_cmd["sequence"], "result": {"released": true}}))
            .await;
        r1.await.unwrap().unwrap();
        let mut r2_cmd = recv_cmd(h.pull.as_mut().unwrap()).await;
        while r2_cmd["operation"] == "forget" {
            r2_cmd = recv_cmd(h.pull.as_mut().unwrap()).await;
        }
        assert_eq!(r2_cmd["rid"], "r2");

        let r3 = {
            let rt = h.runtime.clone();
            tokio::spawn(async move { rt.encode(large_req("r3"), Duration::from_secs(5)).await })
        };
        h.send_reply(json!({"sequence": r2_cmd["sequence"], "result": {"released": true}}))
            .await;
        r2.await.unwrap().unwrap();
        let r3_cmd = recv_cmd(h.pull.as_mut().unwrap()).await;
        h.send_reply(json!({"sequence": r3_cmd["sequence"], "result": {"released": true}}))
            .await;
        r3.await.unwrap().unwrap();
    }
}
