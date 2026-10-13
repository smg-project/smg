//! gRPC client for the TokenSpeed EPD encode service.
//!
//! The gateway calls `encode()` to hand a vision-tower-only worker the
//! preprocessed multimodal tensors for one request; the worker runs the tower
//! and ships the embedding to the paired prefill worker over Mooncake (keyed by
//! `bootstrap_room`). The response only confirms the request was accepted — the
//! embedding transfer happens out of band.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use tonic::transport::Channel;
use tracing::{debug, warn};

use crate::{deadline::deadline_request, BoxedTraceInjector, NoopTraceInjector};

/// Process-global cache of connected channels, keyed by encode-worker endpoint.
///
/// The EPD encode stage dispatches one `Encode` RPC per image per request, so
/// without pooling every call paid a fresh TCP + HTTP/2 (+ TLS) handshake on the
/// request's critical path. A `tonic::Channel` is cheap to clone and multiplexes
/// concurrent RPCs over a single HTTP/2 connection, so we reuse one channel per
/// endpoint — the same connection reuse the prefill/decode legs already get from
/// their cached per-worker client.
/// A single body is large; several concurrent images to the SAME worker would
/// head-of-line-block on one HTTP/2 connection's write buffer + flow-control
/// credit. So hold a small POOL of independent connections per endpoint and
/// round-robin RPCs across them (independent sockets = independent windows).
fn channel_cache() -> &'static Mutex<HashMap<String, Vec<Channel>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Vec<Channel>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// How many independent HTTP/2 connections to keep per encode endpoint.
const ENCODE_CONNS_PER_ENDPOINT: usize = 4;

/// Round-robin cursor over an endpoint's connection pool.
static ENCODE_CONN_CURSOR: AtomicUsize = AtomicUsize::new(0);

#[expect(clippy::allow_attributes)]
pub mod tokenspeed_encoder_proto {
    #![allow(clippy::all, clippy::absolute_paths, unused_qualifications)]
    tonic::include_proto!("tokenspeed.grpc.encoder");
}

/// gRPC client for the TokenSpeed encode worker.
#[derive(Clone)]
pub struct TokenSpeedEncoderClient {
    client: tokenspeed_encoder_proto::token_speed_encoder_client::TokenSpeedEncoderClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

impl TokenSpeedEncoderClient {
    /// Connect reusing a cached `Channel` for `endpoint` (see [`channel_cache`]).
    /// The lock is never held across the connect `await`, so a
    /// rare concurrent first-connect may build two channels for the same endpoint
    /// — harmless: the first cached wins and the extra is dropped.
    pub async fn connect_cached(
        endpoint: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Fast path: pool already built for this endpoint -> pick a connection
        // round-robin so concurrent images spread across the pool's sockets.
        {
            let cache = channel_cache()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(pool) = cache.get(endpoint) {
                if !pool.is_empty() {
                    let i = ENCODE_CONN_CURSOR.fetch_add(1, Ordering::Relaxed);
                    return Ok(Self::from_channel(pool[i % pool.len()].clone()));
                }
            }
        }
        // Slow path: build N independent connections (lock NOT held across the
        // connect awaits). A rare concurrent first-connect may build two pools;
        // or_insert_with keeps the first and the extra is dropped.
        debug!(
            "Connecting to TokenSpeed encoder at {} ({} conns, caching)",
            endpoint, ENCODE_CONNS_PER_ENDPOINT
        );
        let mut built = Vec::with_capacity(ENCODE_CONNS_PER_ENDPOINT);
        for _ in 0..ENCODE_CONNS_PER_ENDPOINT {
            built.push(crate::channel::connect_channel(endpoint).await?);
        }
        {
            let mut cache = channel_cache()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let pool = cache
                .entry(endpoint.to_string())
                .or_insert_with(|| built.clone());
            let i = ENCODE_CONN_CURSOR.fetch_add(1, Ordering::Relaxed);
            Ok(Self::from_channel(pool[i % pool.len()].clone()))
        }
    }

    fn from_channel(channel: Channel) -> Self {
        Self {
            client:
                tokenspeed_encoder_proto::token_speed_encoder_client::TokenSpeedEncoderClient::new(
                    channel,
                ),
            trace_injector: Arc::new(NoopTraceInjector),
        }
    }

    /// Trigger the vision tower on a request's multimodal inputs. Returns once
    /// the worker has accepted (enqueued) the request; the embedding then ships
    /// to the prefill peer asynchronously over Mooncake.
    pub async fn encode(
        &self,
        req: tokenspeed_encoder_proto::EncodeRequest,
    ) -> Result<tokenspeed_encoder_proto::EncodeResponse, tonic::Status> {
        let mut client = self.client.clone();
        // The unary encode call under the request deadline: tonic's client
        // bounds it from the header, and the worker's server cancels it.
        let (_, mut request) = deadline_request(req);

        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {}", e);
        }

        let response = client.encode(request).await?;
        Ok(response.into_inner())
    }
}
