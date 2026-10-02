//! The `Generate` RPC: the one request path. Resolves the frontend's stop and
//! EOS duties, submits per-choice engine streams, matches string stops on the
//! decoded output, and merges the choices into one response stream that the
//! `Abort` RPC can end.

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use engine_zmq_adapter::{
    fold_tokenizer_eos_backstop, stops::take_vllm_string_stops, VllmGenerateStream,
};
use futures::{stream::SelectAll, Stream};
use llm_tokenizer::stop::{StopSequenceDecoder, StopSequenceDecoderBuilder};
use smg_grpc_client::vllm_proto as vllm;
use tokio::sync::oneshot;
use tonic::Status;

use super::{
    requests::{register, Registration},
    State,
};
use crate::BoxStream;

/// The string-stop matcher for a request, or `None` when it carries no
/// string stops. Visible (included-in-output) stops only change the text
/// the Router trims; the servicer only needs the match itself.
pub(super) fn stop_decoder(
    state: &State,
    stops: &[String],
    skip_special_tokens: bool,
) -> Result<Option<StopSequenceDecoder>, Status> {
    if stops.is_empty() {
        return Ok(None);
    }
    let Some(tokenizer) = state.tokenizer() else {
        return Err(Status::failed_precondition(
            "string `stop` sequences need the model tokenizer, which this servicer could not \
             load; start it with a local tokenizer directory",
        ));
    };
    let mut builder = StopSequenceDecoderBuilder::new(Arc::clone(tokenizer))
        .skip_special_tokens(skip_special_tokens);
    for stop in stops {
        builder = builder.stop_sequence(stop.clone());
    }
    Ok(Some(builder.build()))
}

/// Handle one `Generate` request against the connected engine.
pub(super) async fn generate(
    state: &Arc<State>,
    mut req: vllm::GenerateRequest,
) -> Result<BoxStream<vllm::GenerateResponse>, Status> {
    let client = state.engine()?;
    if req.request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    let request_id = req.request_id.clone();
    let streaming = req.stream;
    let skip_special_tokens = req
        .sampling_params
        .as_ref()
        .is_some_and(|sp| sp.skip_special_tokens);
    // The same stop/EOS finalization the Router applies on its direct-ZMQ
    // lane: string stops leave the engine request (single-token ones become
    // stop ids) and come back as this servicer's own obligation; the
    // tokenizer's EOS set backstops a config without ids.
    let tokenizer = state.tokenizer();
    let stops = req
        .sampling_params
        .as_mut()
        .map(|params| take_vllm_string_stops(params, tokenizer))
        .unwrap_or_default();
    client.adopt_tokenizer_eos(tokenizer);
    fold_tokenizer_eos_backstop(&mut req, tokenizer);
    let decoder = stop_decoder(state, &stops, skip_special_tokens)?;

    // Register before submitting: an `Abort` landing in the gap must find
    // the entry, or the engine keeps generating behind an accepted abort.
    let (registration, cancel) = register(state, &request_id)?;
    let subs = client.generate_vllm_streams(req).await?;
    let mut choices = SelectAll::new();
    for sub in subs {
        // Each choice decodes its own text: the matcher is per sequence.
        let decoder = match &decoder {
            Some(_) => stop_decoder(state, &stops, skip_special_tokens)?,
            None => None,
        };
        choices.push(ChoiceStream::new(sub, decoder, streaming));
    }
    Ok(Box::pin(GenerateStream {
        choices,
        cancel,
        cancelled: false,
        _registration: registration,
    }))
}

/// One choice of a generate request: the ZMQ-mapped stream plus the string
/// stop matcher the engine cannot run itself.
struct ChoiceStream {
    /// `None` once this choice ended (terminal response yielded or error).
    /// Dropping it before the engine's own terminal output aborts the
    /// engine-side request.
    inner: Option<VllmGenerateStream>,
    /// Decodes the output incrementally and matches the request's string
    /// stops; `None` when the request carries none.
    decoder: Option<StopSequenceDecoder>,
    /// Whether the client asked for incremental chunks. A non-streaming
    /// request receives only the terminal `Complete`, as from the Python
    /// servicer (`output_kind=FINAL_ONLY`).
    streaming: bool,
    /// A frontend-synthesized terminal `Complete` to yield next.
    pending: Option<vllm::GenerateResponse>,
}

impl ChoiceStream {
    fn new(
        inner: VllmGenerateStream,
        decoder: Option<StopSequenceDecoder>,
        streaming: bool,
    ) -> Self {
        Self {
            inner: Some(inner),
            decoder,
            streaming,
            pending: None,
        }
    }
}

type ChoiceItem = Result<vllm::GenerateResponse, Status>;

impl Stream for ChoiceStream {
    type Item = ChoiceItem;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(complete) = this.pending.take() {
                // The frontend ended this choice: drop the engine stream so the
                // engine-side request is aborted, then deliver the Complete.
                this.inner = None;
                return Poll::Ready(Some(Ok(complete)));
            }
            let Some(inner) = this.inner.as_mut() else {
                return Poll::Ready(None);
            };
            let item = match Pin::new(inner).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    this.inner = None;
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(status))) => {
                    this.inner = None;
                    return Poll::Ready(Some(Err(status)));
                }
                Poll::Ready(Some(Ok(item))) => item,
            };
            let Some(vllm::generate_response::Response::Chunk(chunk)) = item.response.as_ref()
            else {
                // The engine's own terminal `Complete` (or an empty response)
                // passes through as-is; the engine stream ends after it.
                return Poll::Ready(Some(Ok(item)));
            };
            if let Some(decoder) = this.decoder.as_mut() {
                if let Err(decode_error) = decoder.process_tokens(&chunk.token_ids) {
                    this.inner = None;
                    return Poll::Ready(Some(Err(Status::internal(format!(
                        "incremental detokenization failed: {decode_error}"
                    )))));
                }
                if decoder.is_stopped() {
                    // Only string sequences are registered on the decoder, so a
                    // stop always names its matched string.
                    let matched = decoder.matched_stop().unwrap_or_default().to_string();
                    if let Some(inner) = this.inner.as_mut() {
                        // A stop string that is also a single token reaches the
                        // engine as a stop id, so the engine may finish on the
                        // very tick the string matched. Its parked `Complete`
                        // already holds the accumulated ids and logprobs; let
                        // it through instead of synthesizing an empty one.
                        if !inner.has_parked_complete() {
                            this.pending = Some(inner.complete_with_matched_stop(matched));
                        }
                    }
                    if !this.streaming {
                        continue;
                    }
                    // The triggering delta goes out first, as vLLM's own
                    // frontend delivers the step that matched.
                    return Poll::Ready(Some(Ok(item)));
                }
            }
            if !this.streaming {
                continue;
            }
            return Poll::Ready(Some(Ok(item)));
        }
    }
}

/// The merged response stream of one generate request: its choices polled
/// together (interleaved, each tagged with its `index`), ended early by the
/// `Abort` RPC.
struct GenerateStream {
    choices: SelectAll<ChoiceStream>,
    cancel: oneshot::Receiver<()>,
    cancelled: bool,
    _registration: Registration,
}

impl Stream for GenerateStream {
    type Item = ChoiceItem;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.cancelled {
            return Poll::Ready(None);
        }
        // Either an `Abort` fired the sender or shutdown dropped the registry:
        // both end the request. Dropping the choices aborts the engine side.
        if Pin::new(&mut this.cancel).poll(cx).is_ready() {
            this.cancelled = true;
            this.choices.clear();
            return Poll::Ready(Some(Err(Status::cancelled(
                "request aborted on the servicer",
            ))));
        }
        Pin::new(&mut this.choices).poll_next(cx)
    }
}
