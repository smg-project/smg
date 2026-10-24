//! The encode dispatch of an SGLang language-side worker (`--language-only`
//! under encode/prefill disaggregation): what the engine's own frontend does
//! with a request's media before the request reaches the scheduler, done
//! from the servicer process through a dispatcher the lifecycle owner
//! supplies.
//!
//! The scheduler never fetches media on that side. The frontend hands the
//! request's image references to the encoders (SGLang's own receiver object:
//! it splits the items across the encoder list, posts each encoder its
//! share over SGLang's encoder protocol, and keeps the URL snapshot), and
//! marks the scheduler's request with the three slots the scheduler's
//! receiver reads (`need_wait_for_mm_inputs`, `num_items_assigned`,
//! `encoder_urls`): the scheduler rank then registers its own embedding
//! socket with the encoders, waits for the embeddings, and rebuilds the
//! prompt from `input_text` once they land. A dispatch that fails after the
//! request went out reaches the scheduler as the scheduler's own
//! `EncoderDispatchErrorReq` on the same rank, after the request, so the
//! scheduler aborts it with the encoder's message and code instead of
//! waiting out its receive timeout. This module is the Rust side of that
//! choreography; the dispatcher behind [`SglangEncodeDispatcher`] is
//! SGLang's own receiver (the Python binding bridges to it), so the
//! assignment, the transports and their timeouts are the engine's.

use std::{
    collections::BTreeMap,
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
};

pub use engine_zmq_client::protocol::sglang::request::EncoderDispatchErrorReq;
use engine_zmq_client::{codec::encode_msgpack, connector::SglangClient, EngineId};
use tracing::warn;

/// A `Send` boxed future, the shape a dispatcher's calls take.
pub type EncodeFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

/// Where a dispatch failure goes once the request is on the wire: the
/// scheduler rank that holds the request (see [`EncodeNotice`]). Callable
/// from any thread; a failure reported before the request went out, or for
/// a request that never did, is logged and dropped.
pub type EncodeDispatchErrorSink = Arc<dyn Fn(EncoderDispatchErrorReq) + Send + Sync>;

/// One request's media, handed to the dispatcher: the scheduler-side request
/// id (the fan-out's `{rid}-{i}` for `n > 1`: every choice is its own
/// scheduler request and needs its own embeddings), its image references in
/// prompt order, and where a later failure of the dispatch is to be reported.
pub struct EncodeDispatchRequest {
    pub rid: String,
    pub image_urls: Vec<String>,
    pub on_error: EncodeDispatchErrorSink,
}

/// What the dispatcher did, as the engine's frontend records it on the
/// request: whether the scheduler is to wait for embeddings at all (false
/// when no encoder is registered: the frontend then leaves the request
/// alone), how many items each encoder got per modality (the modality's enum
/// value to the per-encoder counts, in the encoder list's order), the
/// encoder list that assignment indexes, and `sent`: what to run once the
/// request is on the wire (or will never be), after which the dispatcher may
/// report failures through the sink. The dispatcher's encode work itself
/// runs in the background; nothing here waits for the encoders.
pub struct EncodeDispatchOutcome {
    pub need_wait_for_mm_inputs: bool,
    pub num_items_assigned: Option<BTreeMap<u32, Vec<u32>>>,
    pub encoder_urls: Option<Vec<String>>,
    pub sent: Option<Box<dyn FnOnce() -> EncodeFuture<()> + Send>>,
}

/// The dispatcher the lifecycle owner supplies for a language-side worker:
/// one call per scheduler request carrying image references, before the
/// request is submitted.
pub trait SglangEncodeDispatcher: Send + Sync + 'static {
    fn dispatch(
        &self,
        request: EncodeDispatchRequest,
    ) -> EncodeFuture<Result<EncodeDispatchOutcome, String>>;
}

/// The scheduler rank a dispatched request went to, learned at submit time,
/// and the client that reaches it: a dispatch failure reported through the
/// sink becomes the scheduler's own notice on that rank. Shared between the
/// sink the dispatcher holds and the submit path that learns the rank.
pub(crate) struct EncodeNotice {
    rank: Mutex<Option<u32>>,
    client: Arc<SglangClient>,
    runtime: tokio::runtime::Handle,
}

impl EncodeNotice {
    pub(crate) fn new(client: Arc<SglangClient>) -> Arc<Self> {
        Arc::new(Self {
            rank: Mutex::new(None),
            client,
            runtime: tokio::runtime::Handle::current(),
        })
    }

    /// The rank the request was handed to; from here on failures reach it.
    pub(crate) fn sent_to(&self, rank: u32) {
        *self
            .rank
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(rank);
    }

    /// The sink the dispatcher reports through.
    pub(crate) fn sink(self: &Arc<Self>) -> EncodeDispatchErrorSink {
        let notice = Arc::clone(self);
        Arc::new(move |error| notice.deliver(error))
    }

    fn deliver(&self, error: EncoderDispatchErrorReq) {
        let rank = *self
            .rank
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(rank) = rank else {
            warn!(
                rid = %error.rid,
                error = %error.error_msg,
                "encode dispatch failed for a request that never reached the scheduler; dropped"
            );
            return;
        };
        let payload = match encode_msgpack(&error) {
            Ok(payload) => payload,
            Err(encode_error) => {
                warn!(rid = %error.rid, %encode_error, "encode dispatch notice could not be encoded");
                return;
            }
        };
        let client = Arc::clone(&self.client);
        let rid = error.rid.clone();
        // Called from the dispatcher's own thread: the send runs on the
        // servicer's runtime.
        self.runtime.spawn(async move {
            if let Err(send_error) = client
                .notify(&EngineId::from_engine_index(rank), payload)
                .await
            {
                warn!(rid, rank, %send_error, "encode dispatch notice could not reach the scheduler");
            }
        });
    }
}
