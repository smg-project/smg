//! String-stop matching on the decoded output: what an engine that sees
//! token ids only cannot do, so the servicer that owns the tokenizer does.

use std::sync::Arc;

use llm_tokenizer::{
    stop::{StopSequenceConfig, StopSequenceDecoder},
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
    let mut config = StopSequenceConfig::default();
    for stop in stops {
        config = config.with_stop_sequence(stop.clone());
    }
    Ok(Some(StopSequenceDecoder::for_stop_matching(
        Arc::clone(tokenizer),
        config,
        skip_special_tokens,
    )))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_fallback_tokenizer() -> Arc<dyn Tokenizer> {
        let definition = serde_json::json!({
            "version":"1.0", "added_tokens":[],
            "model":{"type":"WordLevel", "unk_token":"[UNK]", "vocab":{
                "[UNK]":0, "<0x74>":1, "<0xC3>":2, "<0xA9>":3,
                "<0xF2>":4, " suffix":5
            }},
            "decoder":{"type":"Sequence", "decoders":[
                {"type":"ByteFallback"}, {"type":"Fuse"}
            ]}
        });
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), serde_json::to_vec(&definition).unwrap()).unwrap();
        Arc::new(
            llm_tokenizer::HuggingFaceTokenizer::from_file(file.path().to_str().unwrap()).unwrap(),
        )
    }

    #[test]
    fn byte_fallback_stop_observation_honors_min_tokens() {
        for cached in [false, true] {
            let native = byte_fallback_tokenizer();
            let tokenizer: Arc<dyn Tokenizer> = if cached {
                Arc::new(llm_tokenizer::CachedTokenizer::new(
                    native,
                    llm_tokenizer::CacheConfig::default(),
                ))
            } else {
                native
            };
            // "t" ends inside min_tokens. Closing the byte run later must
            // not make that old stop eligible.
            let decoder = stop_decoder(Some(&tokenizer), &["t".into()], true).unwrap();
            let mut matcher = StopMatcher::new(decoder, 1);
            assert_eq!(
                matcher.feed(&[1, 2, 3, 5]).unwrap(),
                None,
                "cached={cached}"
            );

            // A stop beginning inside the window but ending past it still
            // matches; the incomplete UTF-8 character crosses the boundary.
            let decoder = stop_decoder(Some(&tokenizer), &["té".into()], true).unwrap();
            let mut matcher = StopMatcher::new(decoder, 1);
            assert_eq!(matcher.feed(&[1, 2]).unwrap(), None);
            assert_eq!(matcher.feed(&[3]).unwrap(), Some("té".into()));
        }
    }
}
