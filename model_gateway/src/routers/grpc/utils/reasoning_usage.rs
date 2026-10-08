//! Verify reasoning attribution using original IDs and dedicated control tokens.

use std::collections::HashSet;

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::common::Usage;
use reasoning_parser::ReasoningParser;

// Bound observation memory independently of generated response length. Exceeding
// this bound makes the count unknown rather than estimating it.
const MAX_OBSERVED_IDS: usize = 1_048_576;

#[derive(Clone)]
pub(crate) struct ReasoningTokenCounter {
    start: u32,
    end: u32,
    selected: Vec<u32>,
    leading: Vec<u32>,
    in_reasoning: bool,
    start_consumed: bool,
    stopped: bool,
    valid: bool,
    stop_ids: HashSet<u32>,
}

impl ReasoningTokenCounter {
    pub(crate) fn new(parser: &dyn ReasoningParser, tokenizer: &dyn Tokenizer) -> Option<Self> {
        let (start, end) = parser.reasoning_token_boundaries()?;
        // A standalone multi-token encoding cannot prove a generated boundary:
        // ordinary BPE tokens can contain both reasoning and answer text.
        let marker_id = |marker: &str| {
            let id = tokenizer.token_to_id(marker)?;
            (tokenizer.decode(&[id], false).ok()? == marker).then_some(id)
        };
        Some(Self {
            start: marker_id(start)?,
            end: marker_id(end)?,
            selected: Vec::new(),
            leading: Vec::new(),
            in_reasoning: parser.is_in_reasoning(),
            start_consumed: false,
            stopped: false,
            valid: true,
            stop_ids: HashSet::new(),
        })
    }

    pub(crate) fn configure(&mut self, stripped_start: bool, stop_ids: Vec<u32>) {
        self.start_consumed = stripped_start;
        self.stop_ids = stop_ids.into_iter().collect();
    }

    pub(crate) fn invalidate(&mut self) {
        self.valid = false;
        self.selected.clear();
        self.leading.clear();
    }

    pub(crate) fn record(&mut self, ids: &[u32]) {
        for &id in ids {
            if self.stopped || !self.valid {
                break;
            }
            if self.stop_ids.contains(&id) {
                self.stopped = true;
                break;
            }
            if id == self.start && !self.start_consumed {
                self.in_reasoning = true;
                self.start_consumed = true;
            } else if id == self.end && self.in_reasoning {
                self.in_reasoning = false;
            } else if self.in_reasoning {
                self.selected.push(id);
            } else if !self.start_consumed {
                self.leading.push(id);
            }
            if self.selected.len() + self.leading.len() > MAX_OBSERVED_IDS {
                self.valid = false;
                self.selected.clear();
                self.leading.clear();
            }
        }
    }

    pub(crate) fn verified_count(&self, tokenizer: &dyn Tokenizer, reasoning: &str) -> Option<u32> {
        if !self.valid {
            return None;
        }
        if tokenizer.decode(&self.selected, false).ok()? == reasoning {
            return Some(self.selected.len() as u32);
        }
        // Base parsers can route a same-chunk/nonstream preamble before the
        // first start marker into reasoning. Accept it only if original IDs
        // reproduce exactly what the actual parser emitted.
        if !self.leading.is_empty() {
            let mut ids = self.leading.clone();
            ids.extend_from_slice(&self.selected);
            if tokenizer.decode(&ids, false).ok()? == reasoning {
                return Some(ids.len() as u32);
            }
        }
        None
    }
}

// Called only while parser accounting is active. Unknown means the aggregate
// must be omitted, even if an engine supplied a count for another choice.
pub(crate) fn with_known_reasoning_tokens(mut usage: Usage, count: Option<u32>) -> Usage {
    if let Some(count) = count {
        usage
            .completion_tokens_details
            .get_or_insert_default()
            .reasoning_tokens = Some(count);
    } else if let Some(details) = &mut usage.completion_tokens_details {
        details.reasoning_tokens = None;
    }
    usage
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::{
        traits::{Decoder, Encoder, Encoding},
        SpecialTokens,
    };
    use reasoning_parser::{PassthroughParser, Qwen3Parser};

    use super::*;

    #[derive(Default)]
    struct TokenMap {
        special: SpecialTokens,
    }
    impl Encoder for TokenMap {
        fn encode(&self, text: &str, _: bool) -> anyhow::Result<Encoding> {
            Ok(Encoding::Plain(match text {
                "<think>" => vec![10],
                "</think>" => vec![12],
                _ => anyhow::bail!("generated content must never be re-encoded"),
            }))
        }
        fn encode_batch(&self, texts: &[&str], special: bool) -> anyhow::Result<Vec<Encoding>> {
            texts
                .iter()
                .map(|text| self.encode(text, special))
                .collect()
        }
    }
    impl Decoder for TokenMap {
        fn decode(&self, ids: &[u32], _: bool) -> anyhow::Result<String> {
            ids.iter()
                .map(|id| match id {
                    10 => Ok("<think>"),
                    11 | 13 => Ok("partial"),
                    12 => Ok("</think>"),
                    100 => Ok("a long reasoning token"),
                    101 => Ok("🦀"),
                    102 => Ok("answer"),
                    103 => Ok("reason</think>answer"),
                    _ => Err(anyhow::anyhow!("unknown token")),
                })
                .collect()
        }
    }
    impl Tokenizer for TokenMap {
        fn vocab_size(&self) -> usize {
            103
        }
        fn get_special_tokens(&self) -> &SpecialTokens {
            &self.special
        }
        fn token_to_id(&self, text: &str) -> Option<u32> {
            match text {
                "<think>" => Some(10),
                "</think>" => Some(12),
                _ => None,
            }
        }
        fn id_to_token(&self, id: u32) -> Option<String> {
            self.decode(&[id], false).ok()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn reasoning_token_ids_are_counted_independently_of_chunk_boundaries_and_text_length() {
        let ids = [10, 100, 101, 12, 102];
        for split in 0..=ids.len() {
            let mut counter =
                ReasoningTokenCounter::new(&Qwen3Parser::new(), &TokenMap::default()).unwrap();
            counter.record(&ids[..split]);
            counter.record(&ids[split..]);
            assert_eq!(
                counter.verified_count(&TokenMap::default(), "a long reasoning token🦀"),
                Some(2),
                "split {split}"
            );
        }
    }

    #[test]
    fn prefilled_truncated_and_no_reasoning_counts_preserve_original_token_identity() {
        let tokenizer = TokenMap::default();
        let mut parser = Qwen3Parser::new();
        parser.mark_reasoning_started();
        let mut counter = ReasoningTokenCounter::new(&parser, &tokenizer).unwrap();
        counter.record(&[100, 101]);
        assert_eq!(
            counter.verified_count(&tokenizer, "a long reasoning token🦀"),
            Some(2)
        );
        let mut counter = ReasoningTokenCounter::new(&Qwen3Parser::new(), &tokenizer).unwrap();
        counter.record(&[102]);
        assert_eq!(counter.verified_count(&tokenizer, ""), Some(0));
        assert!(ReasoningTokenCounter::new(&PassthroughParser::new(), &tokenizer).is_none());
    }

    #[test]
    fn ambiguous_merged_boundary_is_unknown_and_prefilled_start_is_not_stripped_twice() {
        let tokenizer = TokenMap::default();
        let mut counter = ReasoningTokenCounter::new(&Qwen3Parser::new(), &tokenizer).unwrap();
        counter.record(&[10, 103]);
        assert_eq!(counter.verified_count(&tokenizer, "reason"), None);
        let mut parser = Qwen3Parser::new();
        parser.mark_reasoning_started();
        let mut counter = ReasoningTokenCounter::new(&parser, &tokenizer).unwrap();
        counter.configure(true, Vec::new());
        counter.record(&[10, 100, 12]);
        assert_eq!(
            counter.verified_count(&tokenizer, "<think>a long reasoning token"),
            Some(2)
        );
    }

    #[test]
    fn original_id_attribution_agrees_with_actual_parser_for_adversarial_boundaries() {
        let tokenizer = TokenMap::default();
        for streaming in [false, true] {
            for (batches, prefilled, stripped, expected) in [
                (vec![vec![102, 10, 100, 12]], false, false, Some(2)),
                (
                    vec![vec![102], vec![10, 100, 12]],
                    false,
                    false,
                    if streaming { Some(1) } else { Some(2) },
                ),
                (vec![vec![10, 103]], false, false, None),
                (
                    vec![vec![10, 100, 12]],
                    true,
                    true,
                    if streaming { Some(2) } else { Some(1) },
                ),
                (
                    vec![vec![100, 12], vec![10, 101, 12]],
                    true,
                    false,
                    if streaming { Some(2) } else { None },
                ),
            ] {
                let mut parser = Qwen3Parser::new();
                if prefilled {
                    parser.mark_reasoning_started();
                }
                let mut counter = ReasoningTokenCounter::new(&parser, &tokenizer).unwrap();
                if stripped && streaming {
                    parser.mark_think_start_stripped();
                }
                counter.configure(stripped && streaming, Vec::new());
                let mut reasoning = String::new();
                let mut all_ids = Vec::new();
                for ids in &batches {
                    counter.record(ids);
                    all_ids.extend_from_slice(ids);
                    if streaming {
                        reasoning.push_str(
                            &parser
                                .parse_reasoning_streaming_incremental(
                                    &tokenizer.decode(ids, false).unwrap(),
                                )
                                .unwrap()
                                .reasoning_text,
                        );
                    }
                }
                if streaming {
                    reasoning.push_str(&parser.flush().unwrap().reasoning_text);
                } else {
                    reasoning = parser
                        .detect_and_parse_reasoning(&tokenizer.decode(&all_ids, false).unwrap())
                        .unwrap()
                        .reasoning_text;
                }
                assert_eq!(counter.verified_count(&tokenizer, &reasoning), expected, "stream={streaming}, prefill={prefilled}, stripped={stripped}, batches={batches:?}, reasoning={reasoning}");
            }
        }
    }

    #[test]
    fn unknown_aggregate_clears_partial_reasoning_without_other_detail_loss() {
        let usage = Usage::from_counts(10, 4)
            .with_reasoning_tokens(2)
            .with_speculative_tokens(2, 4)
            .with_cached_tokens(3);
        let usage = with_known_reasoning_tokens(usage, None);
        let details = usage.completion_tokens_details.unwrap();
        assert_eq!(details.reasoning_tokens, None);
        assert_eq!(details.accepted_prediction_tokens, Some(2));
        assert_eq!(details.rejected_prediction_tokens, Some(2));
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, 3);
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 4);
    }

    #[test]
    fn supported_zero_reasoning_does_not_clear_speculative_details() {
        let usage = with_known_reasoning_tokens(
            Usage::from_counts(10, 4).with_speculative_tokens(2, 4),
            Some(0),
        );
        let details = usage.completion_tokens_details.unwrap();
        assert_eq!(details.reasoning_tokens, Some(0));
        assert_eq!(details.accepted_prediction_tokens, Some(2));
        assert_eq!(details.rejected_prediction_tokens, Some(2));
    }
}
