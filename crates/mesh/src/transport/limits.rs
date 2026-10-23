//! Cross-cutting transport limits and timeouts.
//!
//! Tonic encode/decode caps, derived chunk-byte caps, batch-shape
//! defaults, and the idle timeout shared by inbound and outbound
//! sync_stream loops live here so all wire-shaping constants can be
//! audited in one place.

use std::time::Duration;

/// Maximum gRPC message size (encode and decode caps applied on
/// every tonic channel/builder in this crate).
pub const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024;

/// Headroom reserved below [`MAX_MESSAGE_SIZE`] for protobuf envelope
/// overhead (field tags, lengths, repeated wrappers). Without this,
/// a chunk sized exactly at `MAX_MESSAGE_SIZE` pushes the serialised
/// `StreamMessage` past tonic's cap and the send fails.
pub const STREAM_CHUNK_OVERHEAD_MARGIN: usize = 64 * 1024;

/// Maximum payload bytes per stream chunk after reserving envelope
/// headroom. Senders MUST split values larger than this into multiple
/// chunks via `chunk_value` in the transport's chunking module.
pub const MAX_STREAM_CHUNK_BYTES: usize = MAX_MESSAGE_SIZE - STREAM_CHUNK_OVERHEAD_MARGIN;

/// Default cap on how many chunk entries a single `StreamBatch`
/// frame may carry. Prevents pathological packing of tiny entries
/// into near-`MAX_MESSAGE_SIZE` batches that defeat per-frame
/// fairness on a shared mpsc channel.
pub const DEFAULT_MAX_CHUNKS_PER_BATCH: usize = 5;

/// Idle timeout applied to both inbound and outbound `sync_stream`
/// receive loops. If no frame arrives within this window the loop
/// closes, freeing the mpsc channel and per-stream task.
pub const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
