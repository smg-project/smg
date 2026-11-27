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
    /// Unwrap the underlying gRPC status regardless of send certainty.
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
    /// Bind same-host IPC command/event sockets and spawn the writer and reader tasks.
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

    /// Build a JSON control frame for `rid` when the native IPC link is healthy and connected.
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
    /// Decode one JSON reply frame from Python and resolve the matching pending waiter.
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

    /// Mark the IPC link failed and wake all in-flight waiters with an ambiguous error.
    fn fail(&self, error: Status) {
        self.failed.store(true, Ordering::Release);
        if let Ok(mut pending) = self.pending.lock() {
            for (_, sender) in pending.drain() {
                let _ = sender.send(Err(NativeCallError::Ambiguous(error.clone())));
            }
        }
    }
}

/// Convert a transport or serialization failure into an `UNAVAILABLE` gRPC status.
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
    by_source: HashMap<[u8; 32], (CachedSourceMeta, u64)>,
    order: VecDeque<([u8; 32], u64)>,
    by_item_key: HashMap<String, Vec<[u8; 32]>>,
    max_entries: usize,
    next_gen: u64,
}

impl PreDecodeCache {
    /// Create a bounded FIFO pre-decode index with at most `max_entries` cached sources.
    fn new(max_entries: usize) -> Self {
        Self {
            by_source: HashMap::new(),
            order: VecDeque::new(),
            by_item_key: HashMap::new(),
            max_entries,
            next_gen: 0,
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
            let (meta, _) = self.by_source.get(&src_key)?;
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
        let entry_gen = if let Some((prev, prev_gen)) = self.by_source.get(&source_key) {
            if prev.item_key_hex == meta.item_key_hex
                && prev.grid_thw == meta.grid_thw
                && prev.token_count == meta.token_count
            {
                return;
            }
            let prev_gen = *prev_gen;
            let old_key = prev.item_key_hex.clone();
            if let Some(list) = self.by_item_key.get_mut(&old_key) {
                list.retain(|k| *k != source_key);
                if list.is_empty() {
                    self.by_item_key.remove(&old_key);
                }
            }
            prev_gen
        } else {
            while self.by_source.len() >= self.max_entries {
                let Some((oldest, oldest_gen)) = self.order.pop_front() else {
                    break;
                };
                let matches_gen = self
                    .by_source
                    .get(&oldest)
                    .is_some_and(|(_, current_gen)| *current_gen == oldest_gen);
                if !matches_gen {
                    continue;
                }
                if let Some((evicted, _)) = self.by_source.remove(&oldest) {
                    if let Some(list) = self.by_item_key.get_mut(&evicted.item_key_hex) {
                        list.retain(|k| *k != oldest);
                        if list.is_empty() {
                            self.by_item_key.remove(&evicted.item_key_hex);
                        }
                    }
                }
            }
            if self.order.len() >= self.max_entries.saturating_mul(2) {
                self.order.retain(|(k, g)| {
                    self.by_source
                        .get(k)
                        .is_some_and(|(_, current_gen)| current_gen == g)
                });
            }
            let generation = self.next_gen;
            self.next_gen = self.next_gen.wrapping_add(1);
            self.order.push_back((source_key, generation));
            generation
        };
        self.by_item_key
            .entry(meta.item_key_hex.clone())
            .or_default()
            .push(source_key);
        self.by_source.insert(source_key, (meta, entry_gen));
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
    if !item
        .url
        .as_bytes()
        .get(..5)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"data:"))
    {
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
    /// Return the lowercase protocol label (`"image"` or `"video"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
        }
    }

    /// Parse the optional protobuf modality string, defaulting empty values to image.
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
    /// Create a managed encoder runtime with the default post-cancel receipt grace period.
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

    /// Return `(running, waiting)` request counts across currently admitted encoder owners.
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

    /// Admit an encode RPC, spawn an independent execution owner, and observe its outcome.
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

    /// Validate request limits, modality, and options, then register a bounded execution owner.
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
                        let late_safe = matches!(late, Err(NativeCallError::NotSent(_)))
                            || late.as_ref().is_ok_and(|r| {
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
            self.release_queued_control(&owner);
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
            self.native.notify_forget(rid);
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

    /// Begin draining active requests and reject new admissions.
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

/// Return whether an image detail override is empty or one of `auto`, `low`, or `high`.
fn is_valid_detail(detail: &str) -> bool {
    detail.is_empty()
        || detail.eq_ignore_ascii_case("auto")
        || detail.eq_ignore_ascii_case("low")
        || detail.eq_ignore_ascii_case("high")
}

/// Wait until either the cancellation channel triggers or `deadline` elapses.
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
    /// Snapshot current running and waiting encoder requests as a scheduler load record.
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

    /// Build the advertised managed encoder capability payload with current readiness.
    fn capability(&self) -> Capabilities {
        let mut cap = self.runtime.config.capabilities.clone();
        cap.role = "encoder".into();
        normalized_capabilities(cap, self.runtime.ready())
    }
}

#[tonic::async_trait]
impl ep::sglang_encoder_server::SglangEncoder for EncoderService {
    /// Admit an encode RPC, spawn an independent execution owner, and observe its outcome.
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
    /// Parse `config_json`, bind the IPC and gRPC endpoints, and start the encoder server thread.
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

    /// Begin draining active requests and reject new admissions.
    pub fn drain(&self) {
        self.runtime.drain();
    }

    /// Return the total number of currently running and queued encode requests.
    pub fn active(&self) -> usize {
        let (running, waiting) = self.runtime.loads();
        running + waiting
    }

    /// Return whether the background gRPC server thread is still running.
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

    /// Drain in-flight requests and shut down the control server thread within `timeout`.
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
#[path = "../../tests/unit/sglang/encoder/tests.rs"]
mod tests;
