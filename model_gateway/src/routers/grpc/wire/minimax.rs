//! MiniMax's wire (MiniMax-Provider-Verifier): its stream-QoS contract
//! bounds the size of each SSE event's delta, so a chat stream is re-sliced
//! on its way out (`routers::common::sse_rechunk`): on the gRPC lane, where
//! the gateway encodes the frames itself, and on the regular HTTP relay,
//! where the worker's body passes through. Everything else is the OpenAI
//! wire.

/// Whether a chat stream's events are re-sliced to the contract's bounds.
pub(super) const RECHUNKS_SSE: bool = true;
