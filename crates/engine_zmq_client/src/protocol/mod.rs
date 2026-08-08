//! Per-engine wire protocol modules plus the engine-neutral seam the transport
//! and connector are generic over.
//!
//! vLLM's EngineCore protocol and TokenSpeed's native msgpack protocol
//! (issues #2003/#2006) live alongside each other. Both are msgpack-over-ZMQ
//! with a single-byte request-type frame and share the same transport — only
//! the message struct shapes differ, so the transport and connector are
//! parameterized over the [`EngineProtocol`] trait and each engine family
//! provides one implementation.

use bytes::Bytes;

use crate::Result;

pub mod handshake;
pub mod tokenspeed;
pub mod vllm;

/// Engine-neutral per-rank load signal. Each protocol maps its native scheduler
/// stats into this shape (vLLM maps `SchedulerStats`; TokenSpeed carries none
/// yet), so the connector and gateway consume one load type regardless of
/// engine. Extend it as more engines report load (task #11).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EngineLoad {
    /// Requests currently in model-execution batches.
    pub num_running: u64,
    /// Requests waiting in the scheduler queue.
    pub num_waiting: u64,
    /// KV-cache usage ratio (`1.0` == 100%).
    pub kv_cache_usage: f64,
}

/// A *wave* is vLLM's term (not ours) for one round of work by a data-parallel
/// group whose ranks step in lockstep: they all-reduce every step, so they drain
/// the wave together and then all park. A request handed to one parked rank
/// leaves the others asleep until someone starts the next wave. Upstream names:
/// `current_wave`, `wave_complete`, `START_DP_WAVE` in
/// `vllm/v1/engine/core.py`. Only vLLM's MoE data parallelism steps this way;
/// dense DP ranks are independent and report no waves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveEvent {
    /// The group drained this wave and is now parked (vLLM `wave_complete`).
    Complete(u32),
    /// A rank took a request for an already-drained wave and needs the rest of
    /// the group started on this one (vLLM `start_wave`).
    Start(u32),
}

/// A single per-request output decoded from one engine tick. Lets the
/// engine-neutral connector route outputs to their request streams without
/// knowing the concrete protocol payload.
pub trait EngineOutput {
    /// The request id this output belongs to (the registry routing key).
    fn request_id(&self) -> &str;
    /// Whether this output is terminal for the request.
    fn finished(&self) -> bool;
}

/// One decoded engine tick: the per-request outputs plus the piggybacked load
/// signal and any out-of-band finished ids. The connector consumes this shape
/// for every protocol.
pub struct EngineBatch<O> {
    /// The DP rank / ZMQ index the batch came from.
    pub engine_index: u32,
    /// Per-request outputs produced in this tick.
    pub outputs: Vec<O>,
    /// Request ids the engine reported finished out-of-band.
    pub finished_request_ids: Vec<String>,
    /// Per-rank load signal, when the protocol carries it (`None` otherwise).
    pub load: Option<EngineLoad>,
    /// Wave-control notification from a lockstep engine group (`None` on every
    /// ordinary tick).
    pub wave: Option<WaveEvent>,
}

impl<O> Default for EngineBatch<O> {
    fn default() -> Self {
        Self {
            engine_index: 0,
            outputs: Vec::new(),
            finished_request_ids: Vec::new(),
            load: None,
            wave: None,
        }
    }
}

/// A per-engine wire protocol spoken over the shared ZMQ transport.
///
/// The transport and connector are generic over this trait; each engine family
/// (vLLM EngineCore, TokenSpeed) provides one zero-sized implementation. Only
/// the struct shapes and framing differ — the request registry, streaming,
/// auto-abort, and load tracking are shared.
pub trait EngineProtocol: Send + Sync + 'static {
    /// The typed add-request the connector submits.
    type Request: Send + 'static;
    /// The typed per-request output streamed back.
    type Output: EngineOutput + Send + 'static;

    /// The single-byte request-type frame prepended to an add-request.
    fn add_frame() -> Bytes;
    /// The single-byte request-type frame prepended to an abort.
    fn abort_frame() -> Bytes;

    /// The request's id (the registry routing key).
    fn request_id(request: &Self::Request) -> &str;
    /// SMG-pinned DP rank, if any (else the sole engine is used).
    fn data_parallel_rank(request: &Self::Request) -> Option<u32>;
    /// Reject fields this client cannot represent on the wire.
    fn validate(request: &Self::Request) -> Result<()>;
    /// Encode the add payload plus any aux tensor frames (empty on the text path).
    fn encode_add(request: &Self::Request) -> Result<(Vec<u8>, Vec<Bytes>)>;
    /// Encode the abort payload for one request id.
    fn encode_abort(request_id: &str) -> Result<Vec<u8>>;
    /// Encode "start `wave` on every rank but `exclude_engine_index`" (the one
    /// already holding the request) as `(request-type frame, payload)`.
    /// `Ok(None)` when the protocol has no wave protocol — its ranks run
    /// independently, so there is nothing to wake.
    fn encode_start_wave(wave: u32, exclude_engine_index: u32) -> Result<Option<(Bytes, Vec<u8>)>>;
    /// Decode one output message (frame 0 plus ordered aux frames) into a batch.
    fn decode_batch(frames: &[Bytes]) -> Result<EngineBatch<Self::Output>>;
}
