//! Per-engine wire protocol modules plus the engine-neutral seam the transport
//! and connector are generic over.
//!
//! vLLM's EngineCore protocol, TokenSpeed's native msgpack protocol (issues
//! #2003/#2006) and SGLang's scheduler structs (spoken by the SMG plugin inside
//! the scheduler) live alongside each other. All are msgpack-over-ZMQ with a
//! single-byte request-type frame and share the same transport — only the
//! message struct shapes differ, so the transport and connector are
//! parameterized over the [`EngineProtocol`] trait and each engine family
//! provides one implementation.

use bytes::Bytes;

use self::handshake::EngineCoreReadyResponse;
use crate::{codec::OpaqueValue, Result};

pub mod handshake;
pub(crate) mod positional;
pub mod sglang;
pub mod tokenspeed;
pub mod vllm;

/// Engine-neutral per-rank load signal. Each protocol maps its native scheduler
/// stats into this shape (vLLM its `SchedulerStats`, TokenSpeed and SGLang the
/// load tail of their slim batches), so the connector and gateway consume one
/// load type regardless of engine.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EngineLoad {
    /// Requests currently in model-execution batches.
    pub num_running: u64,
    /// Total waiting requests, including deferred requests.
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
    Complete(u64),
    /// A rank took a request for an already-drained wave and needs the rest of
    /// the group started on this one (vLLM `start_wave`).
    Start(u64),
}

/// An engine's answer to a utility RPC (vLLM `call_utility`), matched to the
/// waiting caller by `call_id`. Protocols without utility RPCs never produce
/// one.
#[derive(Debug, Clone, PartialEq)]
pub struct UtilityReply {
    pub call_id: i64,
    /// The returned value, or the engine's failure message.
    pub outcome: std::result::Result<OpaqueValue, String>,
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
    /// Request ids finished in this tick, including ones the connector could
    /// derive from a terminal output: a protocol without an out-of-band finish
    /// channel may list them here too. The connector unions both sources, so
    /// the overlap is harmless.
    pub finished_request_ids: Vec<String>,
    /// Per-rank load signal, when the protocol carries it (`None` otherwise).
    pub load: Option<EngineLoad>,
    /// Wave-control notification from a lockstep engine group (`None` on every
    /// ordinary tick).
    pub wave: Option<WaveEvent>,
    /// A utility RPC reply (`None` on every ordinary tick).
    pub utility: Option<UtilityReply>,
}

impl<O> Default for EngineBatch<O> {
    fn default() -> Self {
        Self {
            engine_index: 0,
            outputs: Vec::new(),
            finished_request_ids: Vec::new(),
            load: None,
            wave: None,
            utility: None,
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
    /// What this protocol reads off an engine's handshake and needs when it
    /// decodes that engine's outputs: where the engine declared the optional
    /// output columns this client knows (`()` for a protocol whose outputs
    /// carry none). Read once per connection ([`EngineProtocol::read_layout`])
    /// and handed to every [`EngineProtocol::decode_batch`].
    type Layout: Clone + Default + PartialEq + std::fmt::Debug + Send + Sync + 'static;

    /// The single-byte request-type frame prepended to an add-request.
    fn add_frame() -> Bytes;
    /// The single-byte request-type frame prepended to an abort.
    fn abort_frame() -> Bytes;

    /// The request's id (the registry routing key).
    fn request_id(request: &Self::Request) -> &str;
    /// The prompt tokens of an add-request: what the engine has to prefill
    /// before the request's first output.
    fn prompt_tokens(request: &Self::Request) -> u64;
    /// Whether the engine reports this request on every scheduler step that
    /// decodes it, so an engine silent with such a request in flight is a
    /// scheduler that is not stepping: the connector counts the silence it
    /// reports over these requests only. A request the engine reports at
    /// intervals (one that does not stream, whose output comes at its end or
    /// every forced interval; one streamed every so many tokens) is quiet
    /// while healthy, and is left out of that count.
    fn reported_every_step(request: &Self::Request) -> bool;
    /// SMG-pinned DP rank, if any (else the sole engine is used).
    fn data_parallel_rank(request: &Self::Request) -> Option<u32>;
    /// Reject fields this client cannot represent on the wire.
    fn validate(request: &Self::Request) -> Result<()>;
    /// Encode the add payload plus any aux tensor frames (empty on the text path).
    fn encode_add(request: &Self::Request) -> Result<(Vec<u8>, Vec<Bytes>)>;
    /// Encode the abort payload for one request id.
    fn encode_abort(request_id: &str) -> Result<Vec<u8>>;
    /// Encode "start `wave` on every rank" as `(request-type frame, payload)`.
    /// No rank is excluded: a rank that skips the update keeps an older wave
    /// number, and an engine ignores any wave below its own. `Ok(None)` when
    /// the protocol has no wave protocol — its ranks run independently, so
    /// there is nothing to wake.
    fn encode_start_wave(wave: u64) -> Result<Option<(Bytes, Vec<u8>)>>;
    /// Encode a utility (control) call, `method(*args)` on one rank answered by
    /// a [`UtilityReply`] under `call_id`, as `(request-type frame, payload)`.
    /// `Ok(None)` when the protocol has no control messages.
    fn encode_utility(
        _call_id: i64,
        _method: &str,
        _args: &[OpaqueValue],
    ) -> Result<Option<(Bytes, Vec<u8>)>> {
        Ok(None)
    }
    /// Check the positional layouts the engine declared at its handshake
    /// ([`EngineCoreReadyResponse::positional_layouts`], the field order of
    /// each array-encoded struct it decodes) against the prefixes this
    /// protocol's encoders fill, once per engine before its first request: a
    /// struct whose field order moved under this client is named here, with
    /// the field that differs, instead of failing every request it is sent
    /// in. `Ok(())` for an engine that declares no layouts, and for a
    /// protocol with no positional request structs to check.
    fn check_layouts(_ready: &EngineCoreReadyResponse) -> Result<()> {
        Ok(())
    }
    /// The connection's [`EngineProtocol::Layout`], read off the engine's
    /// ready response: the positions the engine declares for the optional
    /// output columns this protocol knows, unset (the default) for a column
    /// it does not declare. Never an error: an undeclared column is an older
    /// engine, whose outputs decode as they always did.
    fn read_layout(_ready: &EngineCoreReadyResponse) -> Self::Layout {
        Self::Layout::default()
    }
    /// Decode one output message (frame 0 plus ordered aux frames) into a
    /// batch, reading the optional output columns where `layout` puts them.
    fn decode_batch(frames: &[Bytes], layout: &Self::Layout) -> Result<EngineBatch<Self::Output>>;
}
