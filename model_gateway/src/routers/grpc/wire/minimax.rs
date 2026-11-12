//! MiniMax's wire (MiniMax-Provider-Verifier): its stream-QoS contract
//! bounds the size of each SSE event's delta (almost none may be shorter
//! than five characters or longer than two hundred), so a chat stream is
//! re-sliced on its way out by [`RECHUNK`] (`routers::common::sse_rechunk`):
//! on the gRPC lane, where the gateway encodes the frames itself, and on the
//! regular HTTP relay, where the worker's body passes through. Everything
//! else is the OpenAI wire.

use std::time::Duration;

use crate::routers::common::sse_rechunk::RechunkPolicy;

/// The contract's numbers: slices of at most 160 characters, cut once 80
/// are buffered, never leaving a tail under 5, and buffered text flushed
/// after 250 ms of upstream silence.
pub(super) const RECHUNK: RechunkPolicy = RechunkPolicy {
    slice_chars: 160,
    emit_threshold: 80,
    min_tail_chars: 5,
    idle_flush: Duration::from_millis(250),
};
