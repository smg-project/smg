//! String-stop matching on the decoded output: what an engine that sees
//! token ids only cannot do, so the servicer that owns the tokenizer does.

use std::sync::Arc;

use llm_tokenizer::{
    stop::{StopSequenceDecoder, StopSequenceDecoderBuilder},
    traits::Tokenizer,
};
use tonic::Status;

/// The string-stop matcher for a request, or `None` when it carries no
/// string stops. Visible (included-in-output) stops only change the text
/// the Router trims; the servicer only needs the match itself.
pub(crate) fn stop_decoder(
    tokenizer: Option<&Arc<dyn Tokenizer>>,
    stops: &[String],
    skip_special_tokens: bool,
) -> Result<Option<StopSequenceDecoder>, Status> {
    if stops.is_empty() {
        return Ok(None);
    }
    let Some(tokenizer) = tokenizer else {
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

/// One choice's stop matcher: feeds generated tokens to the decoder and
/// honours `min_tokens` the way vLLM does (string stops are only checked
/// once the output exceeds it, so a match inside that window is dropped;
/// the matcher keeps its decode context and the matched text's tail, so a
/// stop that ends past the window still matches).
pub(crate) struct StopMatcher {
    decoder: Option<StopSequenceDecoder>,
    min_tokens: u32,
    generated: u32,
}

impl StopMatcher {
    pub(crate) fn new(decoder: Option<StopSequenceDecoder>, min_tokens: u32) -> Self {
        Self {
            decoder,
            min_tokens,
            generated: 0,
        }
    }

    /// Feed this tick's tokens; the matched stop string when one matched.
    pub(crate) fn feed(&mut self, token_ids: &[u32]) -> Result<Option<String>, Status> {
        let Some(decoder) = self.decoder.as_mut() else {
            self.generated = self
                .generated
                .saturating_add(u32::try_from(token_ids.len()).unwrap_or(u32::MAX));
            return Ok(None);
        };
        for &token in token_ids {
            self.generated = self.generated.saturating_add(1);
            decoder.process_token(token).map_err(|error| {
                Status::internal(format!("incremental detokenization failed: {error}"))
            })?;
            if !decoder.is_stopped() {
                continue;
            }
            if self.generated <= self.min_tokens {
                decoder.resume();
                continue;
            }
            // Only string sequences are registered on the decoder, so a stop
            // always names its matched string.
            return Ok(Some(decoder.matched_stop().unwrap_or_default().to_string()));
        }
        Ok(None)
    }
}
