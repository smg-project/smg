use std::{collections::HashSet, sync::Arc};

use aho_corasick::{AhoCorasick, Input};
use anyhow::Result;

use crate::{
    sequence::Sequence,
    traits::{self, TokenIdType},
};

/// Output from the sequence decoder
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceDecoderOutput {
    /// Normal text output
    Text(String),
    /// Text is being held due to partial stop sequence match
    Held,
    /// Stop sequence matched (hidden - not included in output)
    Stopped,
    /// Stop sequence matched with text (visible - included in output)
    StoppedWithText(String),
}

/// Configuration for stop sequences
#[derive(Debug, Clone, Default)]
pub struct StopSequenceConfig {
    /// Token IDs that trigger a stop
    pub stop_tokens: HashSet<TokenIdType>,
    /// String sequences that trigger a stop
    pub stop_sequences: Vec<String>,
    /// Token IDs for visible stops (included in output)
    pub visible_stop_tokens: HashSet<TokenIdType>,
    /// String sequences for visible stops (included in output)
    pub visible_stop_sequences: Vec<String>,
}

impl StopSequenceConfig {
    /// Builder pattern - add a stop token
    pub fn with_stop_token(mut self, token_id: TokenIdType) -> Self {
        self.stop_tokens.insert(token_id);
        self
    }

    /// Builder pattern - add a stop sequence
    pub fn with_stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.stop_sequences.push(sequence.into());
        self
    }

    /// Builder pattern - add a visible stop token
    pub fn with_visible_stop_token(mut self, token_id: TokenIdType) -> Self {
        self.visible_stop_tokens.insert(token_id);
        self
    }

    /// Builder pattern - add a visible stop sequence
    pub fn with_visible_stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.visible_stop_sequences.push(sequence.into());
        self
    }
}

/// Decoder that handles stop sequences
pub struct StopSequenceDecoder {
    /// Sequence for incremental decoding (replaces token_buffer + offsets)
    sequence: Sequence,
    config: StopSequenceConfig,
    /// Aho-Corasick automaton for O(N) stop sequence matching
    aho_corasick: Option<AhoCorasick>,
    /// Index boundary: patterns [0..visible_boundary_idx) are hidden,
    /// patterns [visible_boundary_idx..) are visible
    visible_boundary_idx: usize,
    /// Buffer for partial matches (the "jail")
    jail_buffer: String,
    /// Whether we've stopped
    stopped: bool,
    /// The string stop sequence that triggered the stop, if any. Set only for
    /// string-sequence matches; token-level stops leave this `None`.
    matched_stop: Option<String>,
    /// True when there are no string stop sequences (only token-level stops).
    /// In this mode the jail buffer is bypassed entirely for lower overhead.
    token_only: bool,
}

impl StopSequenceDecoder {
    /// Create a new stop sequence decoder
    pub fn new(
        tokenizer: Arc<dyn traits::Tokenizer>,
        config: StopSequenceConfig,
        skip_special_tokens: bool,
    ) -> Self {
        // Build Aho-Corasick automaton from all stop sequences
        // Hidden sequences come first, then visible sequences
        let mut patterns: Vec<&str> = config
            .stop_sequences
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str())
            .collect();
        let visible_boundary_idx = patterns.len();
        patterns.extend(
            config
                .visible_stop_sequences
                .iter()
                .filter(|s| !s.is_empty())
                .map(|s| s.as_str()),
        );

        let aho_corasick = if patterns.is_empty() {
            None
        } else {
            // AhoCorasick::new is infallible for non-empty, pre-filtered string patterns.
            // Failure would indicate a bug in the Aho-Corasick library itself.
            #[expect(
                clippy::expect_used,
                reason = "AhoCorasick::new with pre-filtered non-empty &str patterns is practically infallible"
            )]
            Some(AhoCorasick::new(patterns).expect("Failed to build Aho-Corasick automaton"))
        };

        let token_only = aho_corasick.is_none();

        StopSequenceDecoder {
            sequence: Sequence::new_with_options(tokenizer, skip_special_tokens),
            config,
            aho_corasick,
            visible_boundary_idx,
            jail_buffer: String::new(),
            stopped: false,
            matched_stop: None,
            token_only,
        }
    }

    /// Byte length of the longest suffix of the jail buffer that is a *proper*
    /// prefix of some stop sequence.
    ///
    /// That suffix is the only part of the buffer that can still grow into a stop
    /// sequence once more tokens arrive; everything before it can never take part
    /// in a match. A full-length match is excluded because the Aho-Corasick scan
    /// has already ruled one out by the time this runs.
    fn pending_match_len(&self) -> usize {
        let buf = self.jail_buffer.as_bytes();
        let Some(&last) = buf.last() else {
            return 0;
        };

        let mut longest = 0;
        for pattern in self
            .config
            .stop_sequences
            .iter()
            .chain(&self.config.visible_stop_sequences)
        {
            let pat = pattern.as_bytes();
            // Only proper prefixes count, and only ones longer than the best so far.
            let max_n = pat.len().saturating_sub(1).min(buf.len());
            for n in (longest + 1..=max_n).rev() {
                // Cheap necessary condition first: the prefix has to end on the
                // byte the buffer ends on, which skips most of the comparisons.
                if pat[n - 1] != last {
                    continue;
                }
                // A prefix ending mid-character would split a multi-byte codepoint
                // in the buffer, so only consider character-aligned prefixes.
                if pattern.is_char_boundary(n) && buf.ends_with(&pat[..n]) {
                    longest = n;
                    break;
                }
            }
        }
        longest
    }

    /// Process a single token
    pub fn process_token(&mut self, token_id: TokenIdType) -> Result<SequenceDecoderOutput> {
        if self.stopped {
            return Ok(SequenceDecoderOutput::Stopped);
        }

        // Check for token-level stops first
        if self.config.stop_tokens.contains(&token_id) {
            self.stopped = true;

            // Flush any jailed text before stopping - use mem::take to avoid clone
            if !self.jail_buffer.is_empty() {
                return Ok(SequenceDecoderOutput::StoppedWithText(std::mem::take(
                    &mut self.jail_buffer,
                )));
            }
            return Ok(SequenceDecoderOutput::Stopped);
        }

        if self.config.visible_stop_tokens.contains(&token_id) {
            self.stopped = true;

            // Include jailed text plus the stop token
            let stop_text = self
                .sequence
                .tokenizer()
                .decode(&[token_id], self.sequence.skip_special_tokens())?;
            let output = format!("{}{}", self.jail_buffer, stop_text);
            self.jail_buffer.clear();
            return Ok(SequenceDecoderOutput::StoppedWithText(output));
        }

        // Use Sequence for incremental decoding
        let new_text = self.sequence.append_token(token_id)?;

        // Token-only fast path: no string stops means no jail is needed.
        if self.token_only {
            if new_text.is_empty() {
                return Ok(SequenceDecoderOutput::Held);
            }
            return Ok(SequenceDecoderOutput::Text(new_text));
        }

        self.jail_buffer.push_str(&new_text);

        // Check for stop sequences using Aho-Corasick. The jail only ever retains a
        // proper prefix of some pattern, so the buffer is bounded by
        // `longest_pattern + one token of text` and scanning it whole is cheap.
        if let Some(ac) = &self.aho_corasick {
            if let Some(mat) = ac.find(Input::new(&self.jail_buffer)) {
                self.stopped = true;
                self.matched_stop = Some(self.jail_buffer[mat.start()..mat.end()].to_string());
                let is_visible = mat.pattern().as_usize() >= self.visible_boundary_idx;

                if is_visible {
                    // Visible stop sequence: include it in output
                    let output = self.jail_buffer[..mat.end()].to_string();
                    self.jail_buffer.clear();
                    return Ok(SequenceDecoderOutput::StoppedWithText(output));
                } else {
                    // Hidden stop sequence: exclude it from output
                    let output = self.jail_buffer[..mat.start()].to_string();
                    self.jail_buffer.clear();
                    return Ok(if output.is_empty() {
                        SequenceDecoderOutput::Stopped
                    } else {
                        SequenceDecoderOutput::StoppedWithText(output)
                    });
                }
            }
        }

        // Withhold only the longest suffix that is still a *partial* stop sequence.
        // Everything before it can never take part in a match, so emit it now
        // rather than trailing the stream by the length of the longest stop word.
        let drain_to = self.jail_buffer.len() - self.pending_match_len();
        if drain_to > 0 {
            let suffix = self.jail_buffer.split_off(drain_to);
            let to_output = std::mem::replace(&mut self.jail_buffer, suffix);
            return Ok(SequenceDecoderOutput::Text(to_output));
        }

        // The whole buffer is a partial stop sequence — hold it.
        Ok(SequenceDecoderOutput::Held)
    }

    /// Process multiple tokens.
    ///
    /// Early-exits after a `Stopped` result — remaining tokens are not processed.
    pub fn process_tokens(
        &mut self,
        token_ids: &[TokenIdType],
    ) -> Result<Vec<SequenceDecoderOutput>> {
        let mut outputs = Vec::with_capacity(token_ids.len());
        for &token_id in token_ids {
            let output = self.process_token(token_id)?;
            let done = matches!(
                output,
                SequenceDecoderOutput::Stopped | SequenceDecoderOutput::StoppedWithText(_)
            );
            outputs.push(output);
            if done {
                break;
            }
        }
        Ok(outputs)
    }

    /// Flush any held text. Returns `Held` if the buffer is empty.
    pub fn flush(&mut self) -> SequenceDecoderOutput {
        if self.jail_buffer.is_empty() {
            SequenceDecoderOutput::Held
        } else {
            // Use mem::take to avoid clone - transfers ownership and leaves empty string
            SequenceDecoderOutput::Text(std::mem::take(&mut self.jail_buffer))
        }
    }

    /// Check if decoding has stopped
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// The string stop sequence that triggered the stop, if a string sequence
    /// matched. `None` for token-level stops or when no stop has fired.
    pub fn matched_stop(&self) -> Option<&str> {
        self.matched_stop.as_deref()
    }

    /// Reset the decoder state
    pub fn reset(&mut self) {
        self.jail_buffer.clear();
        self.sequence.clear();
        self.stopped = false;
        self.matched_stop = None;
    }
}

/// Builder for StopSequenceDecoder
pub struct StopSequenceDecoderBuilder {
    tokenizer: Arc<dyn traits::Tokenizer>,
    config: StopSequenceConfig,
    skip_special_tokens: bool,
}

impl StopSequenceDecoderBuilder {
    pub fn new(tokenizer: Arc<dyn traits::Tokenizer>) -> Self {
        StopSequenceDecoderBuilder {
            tokenizer,
            config: StopSequenceConfig::default(),
            skip_special_tokens: true,
        }
    }

    pub fn stop_token(mut self, token_id: TokenIdType) -> Self {
        self.config.stop_tokens.insert(token_id);
        self
    }

    pub fn stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.config.stop_sequences.push(sequence.into());
        self
    }

    pub fn visible_stop_token(mut self, token_id: TokenIdType) -> Self {
        self.config.visible_stop_tokens.insert(token_id);
        self
    }

    pub fn visible_stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.config.visible_stop_sequences.push(sequence.into());
        self
    }

    pub fn skip_special_tokens(mut self, skip: bool) -> Self {
        self.skip_special_tokens = skip;
        self
    }

    pub fn build(self) -> StopSequenceDecoder {
        StopSequenceDecoder::new(self.tokenizer, self.config, self.skip_special_tokens)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::StopSequenceDecoderBuilder;
    use crate::{
        mock::MockTokenizer, SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder,
    };

    #[test]
    fn test_stop_token_detection() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_token(999); // <eos> token

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process tokens before stop
        let result = decoder.process_token(1).unwrap(); // "Hello"
        assert!(matches!(result, SequenceDecoderOutput::Text(_)));

        // Process stop token
        let result = decoder.process_token(999).unwrap(); // <eos>
        assert_eq!(result, SequenceDecoderOutput::Stopped);

        // Further tokens should also return Stopped
        let result = decoder.process_token(2).unwrap();
        assert_eq!(result, SequenceDecoderOutput::Stopped);
    }

    #[test]
    fn test_visible_stop_token() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_visible_stop_token(999);

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        let result = decoder.process_token(999).unwrap();
        assert!(matches!(result, SequenceDecoderOutput::StoppedWithText(_)));
    }

    #[test]
    fn test_builder_pattern() {
        let tokenizer = Arc::new(MockTokenizer::new());

        let decoder = StopSequenceDecoderBuilder::new(tokenizer)
            .stop_token(999)
            .stop_sequence("STOP")
            .visible_stop_token(1000)
            .skip_special_tokens(true)
            .build();

        assert!(!decoder.is_stopped());
    }

    #[test]
    fn test_incremental_decoding_no_repetition() {
        // This test verifies the critical fix: no repeated output
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default();
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process tokens one by one and collect outputs
        let mut outputs = Vec::new();

        // Token 1: "Hello"
        let result = decoder.process_token(1).unwrap();
        if let SequenceDecoderOutput::Text(text) = result {
            outputs.push(text.clone());
        }

        // Token 2: "world"
        let result = decoder.process_token(2).unwrap();
        if let SequenceDecoderOutput::Text(text) = result {
            outputs.push(text.clone());
        }

        // Token 3: "test"
        let result = decoder.process_token(3).unwrap();
        if let SequenceDecoderOutput::Text(text) = result {
            outputs.push(text.clone());
        }

        // CRITICAL: Each output should be unique (no accumulation)
        // The fix ensures we only output NEW text, not accumulated text
        assert_eq!(outputs.len(), 3);

        for i in 0..outputs.len() {
            for j in i + 1..outputs.len() {
                // No output should contain another (no accumulation)
                assert!(!outputs[j].contains(&outputs[i]));
            }
        }
    }

    #[test]
    fn test_stop_sequence_detection() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence("test");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process "Hello world"
        decoder.process_token(1).unwrap(); // "Hello"
        decoder.process_token(2).unwrap(); // "world"

        // Process "test" which should trigger stop
        let result = decoder.process_token(3).unwrap(); // "test"

        // Should stop when we hit "test"
        assert!(matches!(
            result,
            SequenceDecoderOutput::Stopped | SequenceDecoderOutput::StoppedWithText(_)
        ));
    }

    #[test]
    fn test_matched_stop_reports_matched_string() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence("test");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // No match yet.
        assert_eq!(decoder.matched_stop(), None);

        decoder.process_token(1).unwrap(); // "Hello"
        decoder.process_token(2).unwrap(); // "world"
        assert_eq!(decoder.matched_stop(), None);

        // "test" triggers the string stop; the matched string is captured.
        decoder.process_token(3).unwrap(); // "test"
        assert_eq!(decoder.matched_stop(), Some("test"));

        // Reset clears it.
        decoder.reset();
        assert_eq!(decoder.matched_stop(), None);
    }

    #[test]
    fn test_matched_stop_none_for_token_stop() {
        // A token-id stop is not a string match, so matched_stop stays None.
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_token(999);
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        let result = decoder.process_token(999).unwrap();
        assert_eq!(result, SequenceDecoderOutput::Stopped);
        assert_eq!(decoder.matched_stop(), None);
    }

    #[test]
    fn test_flush_returns_pending_partial_match() {
        let tokenizer = Arc::new(MockTokenizer::new());
        // "Hello" is a proper prefix of the stop sequence, so it must be held
        // back: the next token could complete the match.
        let config = StopSequenceConfig::default().with_stop_sequence("Hello world");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        assert_eq!(
            decoder.process_token(1).unwrap(), // "Hello"
            SequenceDecoderOutput::Held,
            "a proper prefix of the stop sequence must be withheld"
        );

        // The stream ended without completing the match, so flush releases it.
        assert_eq!(
            decoder.flush(),
            SequenceDecoderOutput::Text("Hello".to_string())
        );
        assert_eq!(decoder.flush(), SequenceDecoderOutput::Held);
    }

    #[test]
    fn test_text_that_cannot_match_is_emitted_immediately() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence("NEVER_MATCH");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // No suffix of "Hello" can grow into "NEVER_MATCH", so nothing is jailed
        // and the text goes out with the token that produced it.
        assert_eq!(
            decoder.process_token(1).unwrap(),
            SequenceDecoderOutput::Text("Hello".to_string())
        );

        // Nothing is left behind for the caller to flush unparsed at end of stream.
        assert_eq!(decoder.flush(), SequenceDecoderOutput::Held);
    }

    #[test]
    fn test_control_token_is_not_sliced_by_the_jail() {
        let tokenizer = Arc::new(MockTokenizer::new());
        // A stop sequence long enough to span a control token, and sharing its
        // "<|" opening. This is the shape that leaked: the jail used to retain
        // the last `len(stop)` bytes unconditionally, so a control token landing
        // on that boundary was cut in half and its tail escaped through `flush()`.
        let config = StopSequenceConfig::default().with_stop_sequence("<|im_end|>");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // "<|im_start|>" cannot become "<|im_end|>" — no suffix of it is a prefix
        // of the stop sequence — so it must be emitted whole and at once.
        assert_eq!(
            decoder.process_token(1001).unwrap(),
            SequenceDecoderOutput::Text("<|im_start|>".to_string())
        );
        assert_eq!(
            decoder.flush(),
            SequenceDecoderOutput::Held,
            "no fragment of the control token may be left for an unparsed flush"
        );
        assert!(!decoder.is_stopped());
    }

    #[test]
    fn test_partial_match_is_released_when_it_diverges() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence("Hello world");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // "Hello" looks like the start of the stop sequence — hold it.
        assert_eq!(
            decoder.process_token(1).unwrap(),
            SequenceDecoderOutput::Held
        );

        // "test" makes the match impossible; the held text must come back out
        // in full rather than being dropped or trimmed.
        let out = decoder.process_token(3).unwrap();
        assert_eq!(out, SequenceDecoderOutput::Text("Hello test".to_string()));
        assert_eq!(decoder.flush(), SequenceDecoderOutput::Held);
    }

    #[test]
    fn test_reset_functionality() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_token(999);
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process and stop
        decoder.process_token(1).unwrap();
        decoder.process_token(999).unwrap();
        assert!(decoder.is_stopped());

        // Reset should clear everything
        decoder.reset();
        assert!(!decoder.is_stopped());

        // Should be able to process again
        let result = decoder.process_token(2).unwrap();
        assert!(matches!(result, SequenceDecoderOutput::Text(_)));
    }

    #[test]
    fn test_visible_stop_sequence() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_visible_stop_sequence("world");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process "Hello"
        decoder.process_token(1).unwrap();

        // Process "world" - should include it in output
        let result = decoder.process_token(2).unwrap();

        if let SequenceDecoderOutput::StoppedWithText(text) = result {
            // Should include "world" in the output
            assert!(text.contains("world"));
        } else {
            panic!("Expected StoppedWithText with visible stop sequence");
        }
    }

    #[test]
    fn test_multiple_tokens_processing() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default();
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process multiple tokens at once
        let results = decoder.process_tokens(&[1, 2, 3]).unwrap();

        // Should get results for each token
        assert_eq!(results.len(), 3);

        // Each result should be Text (no stops configured)
        for result in results {
            assert!(matches!(
                result,
                SequenceDecoderOutput::Text(_) | SequenceDecoderOutput::Held
            ));
        }
    }

    /// Test that the jail buffer correctly handles a stop sequence that arrives
    /// across 2+ tokens.  The MockTokenizer decodes token 1 as "Hello" and
    /// token 2's incremental contribution as " world", so the jail buffer
    /// progressively becomes "Hello" then "Hello world".
    ///
    /// With stop sequence "Hello world":
    ///   - Token 1: jail = "Hello" — partial prefix match → Held (or Text of
    ///     the portion before the potential match, which is empty here)
    ///   - Token 2: jail = "Hello world" — full match → Stopped
    #[test]
    fn test_stop_sequence_spanning_multiple_tokens() {
        let tokenizer = Arc::new(MockTokenizer::new());

        // "Hello world" spans token 1 ("Hello") and token 2 (" world")
        let config = StopSequenceConfig::default().with_stop_sequence("Hello world");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Token 1 ("Hello"): The jail buffer now contains "Hello".
        // "Hello" is a prefix of stop sequence "Hello world", so the text
        // must be held — we should NOT see it emitted as Text yet.
        let result1 = decoder.process_token(1).unwrap();
        assert!(
            matches!(result1, SequenceDecoderOutput::Held),
            "Expected Held while jail buffer is a prefix of the stop sequence, got {result1:?}"
        );
        assert!(
            !decoder.is_stopped(),
            "Decoder should not be stopped after a partial match"
        );

        // Token 2 (" world"): The jail buffer now contains "Hello world",
        // which fully matches the stop sequence. The decoder should stop.
        let result2 = decoder.process_token(2).unwrap();
        assert_eq!(
            result2,
            SequenceDecoderOutput::Stopped,
            "Expected Stopped when jail buffer matches the hidden stop sequence"
        );
        assert!(
            decoder.is_stopped(),
            "Decoder should be stopped after the full stop sequence match"
        );

        // Any further tokens should also return Stopped
        let result3 = decoder.process_token(3).unwrap();
        assert_eq!(result3, SequenceDecoderOutput::Stopped);
    }

    /// Same as above but with a *visible* stop sequence.  When the stop
    /// sequence "Hello world" is visible, the matched text should be included
    /// in the output via StoppedWithText.
    #[test]
    fn test_visible_stop_sequence_spanning_multiple_tokens() {
        let tokenizer = Arc::new(MockTokenizer::new());

        let config = StopSequenceConfig::default().with_visible_stop_sequence("Hello world");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Token 1 ("Hello"): partial match, should be held
        let result1 = decoder.process_token(1).unwrap();
        assert!(
            matches!(result1, SequenceDecoderOutput::Held),
            "Expected Held for partial visible stop sequence match, got {result1:?}"
        );

        // Token 2 (" world"): completes "Hello world" — visible stop
        let result2 = decoder.process_token(2).unwrap();
        match &result2 {
            SequenceDecoderOutput::StoppedWithText(text) => {
                assert!(
                    text.contains("Hello world"),
                    "Visible stop output should contain the full stop sequence, got: {text:?}"
                );
            }
            other => panic!("Expected StoppedWithText for visible stop sequence, got {other:?}"),
        }
        assert!(decoder.is_stopped());
    }

    /// Test a stop sequence that spans 3 tokens, with preceding text that
    /// should be emitted before the jailed portion.
    ///
    /// Tokens: 3 ("test"), 1 ("Hello"), 2 ("world")
    /// Stop sequence: "Hello world" (11 bytes)
    ///
    /// Only a suffix that is a proper prefix of the stop sequence is withheld,
    /// so text flows out as soon as it can no longer be part of a match:
    ///   - Token 3: "test" cannot start "Hello world"           → Text("test")
    ///   - Token 1: jail = " Hello", holds "Hello"              → Text(" ")
    ///   - Token 2: jail = "Hello world" — Aho-Corasick matches → Stopped
    ///
    /// The caller sees "test " either way; the difference is that it arrives
    /// with the tokens that produced it instead of trailing the stream.
    #[test]
    fn test_stop_sequence_spanning_tokens_with_preceding_text() {
        let tokenizer = Arc::new(MockTokenizer::new());

        let config = StopSequenceConfig::default().with_stop_sequence("Hello world");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        let mut emitted = String::new();
        let mut stopped = false;
        for token in [3u32, 1, 2] {
            match decoder.process_token(token).unwrap() {
                SequenceDecoderOutput::Text(t) => emitted.push_str(&t),
                SequenceDecoderOutput::StoppedWithText(t) => {
                    emitted.push_str(&t);
                    stopped = true;
                }
                SequenceDecoderOutput::Stopped => stopped = true,
                SequenceDecoderOutput::Held => {}
            }
        }

        assert!(
            stopped,
            "the stop sequence spanning tokens should have fired"
        );
        assert_eq!(
            emitted, "test ",
            "everything before the hidden stop sequence is emitted, and nothing more"
        );
        assert_eq!(
            decoder.flush(),
            SequenceDecoderOutput::Held,
            "a completed stop must leave nothing jailed for an unparsed flush"
        );
        assert!(decoder.is_stopped());
        assert!(
            !emitted.contains("Hello world"),
            "the hidden stop sequence must not appear in the output, got: {emitted:?}"
        );
    }

    #[test]
    fn test_utf8_multibyte_character_boundaries() {
        // This test verifies the fix for the UTF-8 boundary panic
        // The panic occurred when trying to slice jail_buffer at a byte index
        // that was in the middle of a multi-byte UTF-8 character (e.g., '×')
        use crate::mock::MockTokenizer;

        let tokenizer = Arc::new(MockTokenizer::new());

        // Configure stop sequence with a multi-byte character
        let config = StopSequenceConfig::default().with_stop_sequence(" ×");

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Simulate the scenario: jail_buffer will contain " ×" (space + multiplication sign)
        // The '×' character is UTF-8 encoded as bytes [0xC3, 0x97] (2 bytes)
        // When checking for partial matches, we must not slice in the middle of these bytes

        // This should not panic - the fix ensures we only slice at char boundaries
        let result = decoder.process_token(1); // Will add some text to jail_buffer
        assert!(result.is_ok());

        // Even with multi-byte UTF-8 characters in the buffer, processing should work
        let result = decoder.process_token(2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_utf8_multibyte_delta_character() {
        // Test for: byte index 1 is not a char boundary; it is inside 'Δ' (bytes 0..2) of `Δ`
        // 'Δ' (U+0394 GREEK CAPITAL LETTER DELTA) is encoded as [0xCE, 0x94] (2 bytes)
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence("Δ");

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process tokens - should not panic when checking partial matches
        let result = decoder.process_token(1);
        assert!(result.is_ok());
        let result = decoder.process_token(2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_utf8_multibyte_degree_character() {
        // Test for: byte index 1 is not a char boundary; it is inside '°' (bytes 0..2) of `°`
        // '°' (U+00B0 DEGREE SIGN) is encoded as [0xC2, 0xB0] (2 bytes)
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence("°");

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process tokens - should not panic when checking partial matches
        let result = decoder.process_token(1);
        assert!(result.is_ok());
        let result = decoder.process_token(2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_utf8_multibyte_triangle_character() {
        // Test for: byte index 4 is not a char boundary; it is inside '∆' (bytes 2..5) of ` (∆`
        // '∆' (U+2206 INCREMENT) is encoded as [0xE2, 0x88, 0x86] (3 bytes)
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence(" (∆");

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process tokens - should not panic when checking partial matches
        let result = decoder.process_token(1);
        assert!(result.is_ok());
        let result = decoder.process_token(2);
        assert!(result.is_ok());
        let result = decoder.process_token(3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_utf8_multibyte_en_dash_character() {
        // Test for: byte index 3 is not a char boundary; it is inside '–' (bytes 1..4) of ` –`
        // '–' (U+2013 EN DASH) is encoded as [0xE2, 0x80, 0x93] (3 bytes)
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = StopSequenceConfig::default().with_stop_sequence(" –");

        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        // Process tokens - should not panic when checking partial matches
        let result = decoder.process_token(1);
        assert!(result.is_ok());
        let result = decoder.process_token(2);
        assert!(result.is_ok());
        let result = decoder.process_token(3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_utf8_multibyte_various_characters() {
        // Comprehensive test with multiple multi-byte UTF-8 characters
        // Tests 2-byte, 3-byte, and 4-byte UTF-8 sequences
        let test_cases = vec![
            ("×", "multiplication sign - 2 bytes"),
            ("Δ", "Greek Delta - 2 bytes"),
            ("°", "degree sign - 2 bytes"),
            ("∆", "increment - 3 bytes"),
            ("–", "en dash - 3 bytes"),
            ("€", "euro sign - 3 bytes"),
            ("中", "Chinese character - 3 bytes"),
            ("🚀", "rocket emoji - 4 bytes"),
            ("💡", "lightbulb emoji - 4 bytes"),
        ];

        for (stop_char, description) in test_cases {
            let tokenizer = Arc::new(MockTokenizer::new());
            let config = StopSequenceConfig::default().with_stop_sequence(stop_char);

            let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

            // Process multiple tokens - should not panic
            for token_id in 1..=5 {
                let result = decoder.process_token(token_id);
                assert!(
                    result.is_ok(),
                    "Failed on {description} with token {token_id}"
                );
            }
        }
    }
}
