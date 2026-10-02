//! Worker-side media processing (`media_refs`): the request names its media
//! by reference, and this worker fetches and processes it instead of the
//! Router. The processor is supplied by the lifecycle owner through
//! [`MediaProcessor`]; the Python binding bridges to the processors the
//! Python servicer runs (vLLM's in-process renderer, the Redis sidecar), so
//! the engine's own input processor does the work either way. What it
//! produces, the engine's `mm_features` as vLLM encoded them, is relayed to
//! the engine untouched.

use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use engine_zmq_adapter::ProcessedMedia as EngineMedia;
use prost::Message;
use smg_grpc_client::{common_proto as common, vllm_proto as vllm};
use tokio::sync::{oneshot, OwnedSemaphorePermit, Semaphore};
use tonic::Status;

use super::State;

/// A `Send` boxed future, the shape a processor's calls take.
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

/// One `media_refs` item: the modality name (`image`, `video`) and the
/// reference to fetch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaRefItem {
    pub modality: String,
    pub url: String,
}

/// What a processor gets: the request's prompt as the Router sent it (one
/// placeholder anchor per item, not yet expanded), its text when known, and
/// the references in prompt order.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaRequest {
    pub request_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub prompt_text: Option<String>,
    pub items: Vec<MediaRefItem>,
    pub arrival_time: f64,
    /// Whether to also build the identity a PD prefill leg returns on its
    /// `Complete`, so the decode leg is served without pixels or references.
    pub want_identity: bool,
}

/// What a processor produced: the prompt with its placeholders expanded, the
/// engine's `mm_features` as vLLM's `MsgpackEncoder` wrote them (the primary
/// buffer, plus the aux tensor frames it split off), the cache salt, and,
/// when asked, the serialized `MediaIdentity` proto.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProcessedMedia {
    pub prompt_token_ids: Vec<u32>,
    pub mm_features: Option<Bytes>,
    pub aux_frames: Vec<Bytes>,
    pub cache_salt: Option<String>,
    pub media_identity: Option<Bytes>,
}

/// Why a processor refused or failed a request, mapped to the status the
/// Python servicer answers for the same failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaError {
    /// The caller's: a bad reference, media the model cannot take, a prompt
    /// the placeholders do not fit (`INVALID_ARGUMENT`).
    Invalid(String),
    /// Not this worker's fault right now: a fetch that timed out, a sidecar
    /// saturated or unreachable; the Router re-selects a worker
    /// (`UNAVAILABLE`).
    Unavailable(String),
    /// A processor failure (`INTERNAL`).
    Internal(String),
}

impl MediaError {
    fn into_status(self) -> Status {
        match self {
            Self::Invalid(message) => Status::invalid_argument(message),
            Self::Unavailable(message) => Status::unavailable(message),
            Self::Internal(message) => Status::internal(message),
        }
    }
}

/// A worker-side media processor. Everything advertised to the Router
/// (`GetServerInfo`) comes from here, and so does the per-worker in-flight
/// cap the servicer enforces around [`process`](Self::process).
pub trait MediaProcessor: Send + Sync + 'static {
    /// The backend's advertised name (`mm_processor`), e.g. `inprocess`.
    fn name(&self) -> &str;
    /// The URL schemes it fetches (`mm_media_ref_schemes`), comma-separated;
    /// read after [`probe`](Self::probe), since a sidecar announces its own.
    fn schemes(&self) -> String;
    /// Where the mode came from (`mm_processor_source`): `flag`, `env`,
    /// `default`.
    fn source(&self) -> &str;
    /// How many requests it processes at once; as many again may wait.
    fn max_inflight(&self) -> usize;
    /// Whether it can take requests now (a sidecar's liveness); the backend
    /// is advertised only while this holds.
    fn probe(&self) -> BoxFuture<bool>;
    fn process(&self, request: MediaRequest) -> BoxFuture<Result<ProcessedMedia, MediaError>>;
}

/// The processor behind its in-flight cap, as the Python servicer runs it:
/// `max_inflight` requests process at once, as many again may wait, and the
/// ones behind those are shed with a retryable refusal rather than queued
/// past their callers' patience.
pub(crate) struct MediaGate {
    pub(super) processor: Arc<dyn MediaProcessor>,
    inflight: Arc<Semaphore>,
    waiting: AtomicUsize,
    limit: usize,
}

impl MediaGate {
    pub(super) fn new(processor: Arc<dyn MediaProcessor>) -> Self {
        let limit = processor.max_inflight().max(1);
        Self {
            processor,
            inflight: Arc::new(Semaphore::new(limit)),
            waiting: AtomicUsize::new(0),
            limit,
        }
    }

    /// A slot, or the saturation refusal. The waiter count is checked and
    /// taken in one step, and given back on every exit, a cancelled wait
    /// included: a caller that gives up while queued must not leave a
    /// phantom waiter behind, or enough of them would refuse every request
    /// for the life of the process.
    async fn acquire(&self) -> Result<OwnedSemaphorePermit, Status> {
        let admitted = self
            .waiting
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |waiting| {
                (waiting < self.limit).then_some(waiting + 1)
            });
        if admitted.is_err() {
            return Err(Status::unavailable(format!(
                "worker is saturated: {} multimodal requests in flight and as many waiting",
                self.limit
            )));
        }
        let _waiting = Waiting(&self.waiting);
        let permit = Arc::clone(&self.inflight).acquire_owned().await;
        permit.map_err(|_| Status::unavailable("media processing is shutting down"))
    }
}

/// Counts a waiter down when it leaves the queue, however it leaves.
struct Waiting<'a>(&'a AtomicUsize);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The Python servicer's refusal when references arrive with processing off.
pub(super) const PROCESSOR_OFF: &str =
    "media_refs sent but --mm-processor (SMG_VLLM_MM_PROCESSOR) \
                                        is off on this worker; check the router's --mm-processing \
                                        and this worker's mm_processor label";

/// The request's `media_refs` as processor items, checked as the Python
/// servicer checks them (`parse_media_refs`): a fetchable modality and a
/// non-empty reference each, at least one item.
fn parse_media_refs(refs: &vllm::MediaRefs) -> Result<Vec<MediaRefItem>, Status> {
    let mut items = Vec::with_capacity(refs.items.len());
    for (index, item) in refs.items.iter().enumerate() {
        let modality = match common::Modality::try_from(item.modality) {
            Ok(common::Modality::Image) => "image",
            Ok(common::Modality::Video) => "video",
            other => {
                let name = other.map_or_else(
                    |_| item.modality.to_string(),
                    |m| m.as_str_name().to_string(),
                );
                return Err(Status::invalid_argument(format!(
                    "media_refs[{index}]: unsupported modality {name}"
                )));
            }
        };
        if item.url.is_empty() {
            return Err(Status::invalid_argument(format!(
                "media_refs[{index}]: empty url"
            )));
        }
        items.push(MediaRefItem {
            modality: modality.to_string(),
            url: item.url.clone(),
        });
    }
    if items.is_empty() {
        return Err(Status::invalid_argument(
            "media_refs is set but carries no items",
        ));
    }
    Ok(items)
}

/// Resolve a request's `media_refs` through the configured processor. The
/// request leaves with its references consumed and its prompt expanded; what
/// comes back is the engine-side media to attach and, for a PD prefill leg,
/// the identity its `Complete` carries. `None` when the request named no
/// media.
pub(super) async fn process_media_refs(
    state: &State,
    req: &mut vllm::GenerateRequest,
) -> Result<Option<(EngineMedia, Option<vllm::MediaIdentity>)>, Status> {
    let Some(refs) = req.media_refs.take().filter(|refs| !refs.items.is_empty()) else {
        return Ok(None);
    };
    let has_batches = req.mm_inputs.is_some() || !req.extra_mm_inputs.is_empty();
    let tokenized = match req.input.as_ref() {
        Some(vllm::generate_request::Input::Tokenized(tokenized)) if !has_batches => tokenized,
        _ => {
            return Err(Status::invalid_argument(
                "media_refs requires tokenized input and cannot be combined with preprocessed \
                 multimodal inputs",
            ))
        }
    };
    let Some(gate) = state.media.as_ref() else {
        return Err(Status::invalid_argument(PROCESSOR_OFF));
    };
    let items = parse_media_refs(&refs)?;
    let request = MediaRequest {
        request_id: req.request_id.clone(),
        prompt_token_ids: tokenized.input_ids.clone(),
        prompt_text: (!tokenized.original_text.is_empty()).then(|| tokenized.original_text.clone()),
        items,
        arrival_time: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0),
        want_identity: req.kv_transfer_params_json.is_some() || req.kv_transfer_params.is_some(),
    };
    let permit = gate.acquire().await?;
    // The processor runs in its own task holding the slot: a caller that
    // gives up mid-fetch does not stop the work behind the bridge, so the
    // slot comes back when that work ends, not when the caller stops
    // waiting, and the cap bounds the work actually in flight.
    let processor = Arc::clone(&gate.processor);
    let (tx, rx) = oneshot::channel();
    #[expect(
        clippy::disallowed_methods,
        reason = "the slot must outlive a caller that gives up; the task ends when the processor answers"
    )]
    tokio::spawn(async move {
        let outcome = processor.process(request).await;
        drop(permit);
        let _ = tx.send(outcome);
    });
    let processed = rx
        .await
        .map_err(|_| Status::internal("media processing ended without a result"))?
        .map_err(MediaError::into_status)?;
    // The expanded prompt replaces the anchors the Router sent.
    if let Some(vllm::generate_request::Input::Tokenized(tokenized)) = req.input.as_mut() {
        tokenized.input_ids = processed.prompt_token_ids;
    }
    let identity = processed
        .media_identity
        .map(vllm::MediaIdentity::decode)
        .transpose()
        .map_err(|error| {
            Status::internal(format!("media identity could not be decoded: {error}"))
        })?;
    Ok(Some((
        EngineMedia {
            mm_features: processed.mm_features,
            aux_frames: processed.aux_frames,
            cache_salt: processed.cache_salt,
        },
        identity,
    )))
}
