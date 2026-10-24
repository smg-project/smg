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
/// scheduler rank that holds the request (see the crate-private
/// `EncodeNotice`). Callable from any thread; a failure reported before the
/// request went out, or for a request that never did, is logged and dropped.
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
/// encoder list that assignment indexes, and `sent`: the release to run once
/// the request is on the wire (or will never be), after which the dispatcher
/// may report failures through the sink. The dispatcher's encode work itself
/// runs in the background; nothing here waits for the encoders.
pub struct EncodeDispatchOutcome {
    pub need_wait_for_mm_inputs: bool,
    pub num_items_assigned: Option<BTreeMap<u32, Vec<u32>>>,
    pub encoder_urls: Option<Vec<String>>,
    pub sent: Option<EncodeRelease>,
}

/// The dispatcher's release, a guard: the engine's receiver waits on it
/// without a timeout before it reports a failed encode, so a handler dropped
/// while its submit or its dispatch was pending (the client gone, its
/// deadline passed) must still release. Run by the handler once the request
/// is on the wire (or never will be); dropped unrun, it runs on the runtime.
pub struct EncodeRelease {
    release: Option<Box<dyn FnOnce() -> EncodeFuture<()> + Send>>,
    runtime: tokio::runtime::Handle,
}

impl EncodeRelease {
    /// A release that runs on `runtime` if it is dropped before [`Self::run`].
    pub fn new(
        release: Box<dyn FnOnce() -> EncodeFuture<()> + Send>,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        Self {
            release: Some(release),
            runtime,
        }
    }

    /// Run the release here and wait for it.
    pub async fn run(mut self) {
        if let Some(release) = self.release.take() {
            release().await;
        }
    }
}

impl Drop for EncodeRelease {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            self.runtime.spawn(release());
        }
    }
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

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, time::Duration};

    use tokio::sync::oneshot;

    use super::*;

    fn release_signalling(tx: oneshot::Sender<()>) -> EncodeRelease {
        EncodeRelease::new(
            Box::new(move || {
                Box::pin(async move {
                    let _ = tx.send(());
                })
            }),
            tokio::runtime::Handle::current(),
        )
    }

    #[tokio::test]
    async fn a_release_run_by_the_handler_runs_there() {
        let (tx, rx) = oneshot::channel();
        release_signalling(tx).run().await;
        assert!(rx.await.is_ok(), "the release ran");
    }

    #[tokio::test]
    async fn a_release_dropped_unrun_runs_on_the_runtime() {
        let (tx, rx) = oneshot::channel();
        drop(release_signalling(tx));
        tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .expect("the dropped release ran within the timeout")
            .expect("the release sent its signal");
    }

    #[tokio::test]
    async fn a_handler_cancelled_during_its_dispatch_still_releases() {
        // The handler is cancelled while its dispatch's blocking work runs;
        // that work ends after the handler is gone: the runtime drops the
        // task's output, outcome and release with it, and the release still
        // runs.
        let (tx, rx) = oneshot::channel();
        let (started_tx, started_rx) = oneshot::channel();
        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        let runtime = tokio::runtime::Handle::current();
        #[expect(
            clippy::disallowed_methods,
            reason = "the handler is cancelled by the test"
        )]
        let handler = tokio::spawn(async move {
            let outcome = tokio::task::spawn_blocking(move || {
                let _ = started_tx.send(());
                gate_rx.recv().ok();
                EncodeDispatchOutcome {
                    need_wait_for_mm_inputs: true,
                    num_items_assigned: None,
                    encoder_urls: None,
                    sent: Some(EncodeRelease::new(
                        Box::new(move || {
                            Box::pin(async move {
                                let _ = tx.send(());
                            })
                        }),
                        runtime,
                    )),
                }
            })
            .await
            .expect("the dispatch ran");
            if let Some(sent) = outcome.sent {
                sent.run().await;
            }
        });
        started_rx
            .await
            .expect("the dispatch started before the handler was cancelled");
        handler.abort();
        let _ = handler.await;
        gate_tx
            .send(())
            .expect("the blocking dispatch is still waiting");
        tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .expect("the release ran after the handler was cancelled")
            .expect("the release sent its signal");
    }
}
