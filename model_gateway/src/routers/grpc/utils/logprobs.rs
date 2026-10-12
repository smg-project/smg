//! Proto-to-OpenAI logprob conversion functions.

use std::{collections::HashMap, sync::Arc};

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::common::{ChatLogProbs, ChatLogProbsContent, LogProbs, TopLogProb};

use crate::routers::{
    error,
    grpc::proto_wrapper::{ProtoInputLogProbs, ProtoOutputLogProbs, ProtoTopLogProbs},
};

/// Convert OutputLogProbs to OpenAI ChatLogProbs format
///
/// Generic over the token decoding strategy. The `decode_token` closure maps a
/// single token ID to its text representation.
pub(crate) fn convert_proto_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
    decode_token: impl Fn(u32) -> String,
) -> ChatLogProbs {
    let mut content_items = Vec::with_capacity(proto_logprobs.token_logprobs.len());

    for (i, &logprob) in proto_logprobs.token_logprobs.iter().enumerate() {
        let token_id = proto_logprobs.token_ids.get(i).copied().unwrap_or(0);
        let token_text = decode_token(token_id);
        let bytes = Some(token_text.as_bytes().to_vec());

        // Build top_logprobs for this position
        let top_logprobs = if let Some(top_logprobs_entry) = proto_logprobs.top_logprobs.get(i) {
            top_logprobs_entry
                .values
                .iter()
                .enumerate()
                .filter_map(|(j, &top_logprob)| {
                    top_logprobs_entry.token_ids.get(j).map(|&tid| {
                        let text = decode_token(tid);
                        let bytes = Some(text.as_bytes().to_vec());
                        TopLogProb {
                            token: text,
                            logprob: top_logprob,
                            bytes,
                        }
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        content_items.push(ChatLogProbsContent {
            token: token_text,
            logprob,
            bytes,
            top_logprobs,
        });
    }

    ChatLogProbs::Detailed {
        content: (!content_items.is_empty()).then_some(content_items),
        refusal: None,
    }
}

/// Convert OutputLogProbs to OpenAI ChatLogProbs format using a Tokenizer.
///
/// `top_logprobs` is the number of alternatives the request asked for per
/// token (`None` when it asked for none). The engine may report more, e.g.
/// the chosen token when only its own logprob was requested; the OpenAI API
/// returns exactly the requested alternatives, an empty list when none were.
pub(crate) fn convert_proto_to_openai_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
    tokenizer: &Arc<dyn Tokenizer>,
    top_logprobs: Option<u32>,
) -> ChatLogProbs {
    let mut logprobs = convert_proto_logprobs(proto_logprobs, |token_id| {
        tokenizer
            .decode(&[token_id], false)
            .unwrap_or_else(|_| format!("<token_{token_id}>"))
    });
    truncate_top_logprobs(&mut logprobs, top_logprobs.unwrap_or(0) as usize);
    logprobs
}

/// Keep at most `requested` alternatives per token.
fn truncate_top_logprobs(logprobs: &mut ChatLogProbs, requested: usize) {
    if let ChatLogProbs::Detailed {
        content: Some(items),
        ..
    } = logprobs
    {
        for item in items {
            item.top_logprobs.truncate(requested);
        }
    }
}

/// Legacy `/v1/completions` logprobs of one choice, built token by token.
///
/// Every token is its own decode with special tokens kept, its logprob, the
/// alternatives the engine ranked for its position (its first `logprobs`, in
/// rank order; `null` where none were asked for and at the first prompt
/// position, which has no logprob), and its character offset in the choice's
/// text (the sum of the earlier tokens' lengths). Prompt tokens (`echo`) come
/// before the generated ones. Two alternatives that decode to the same text
/// share one entry, as a Python dict built in rank order holds them: at the
/// first one's place, with the later (less likely) one's value. A client that
/// builds the same map from the ranked ids with a Python dict gets the same
/// entries, so per-text comparisons agree (byte-fallback pieces all decode to
/// U+FFFD).
pub(crate) struct CompletionLogprobs<'t> {
    tokenizer: &'t dyn Tokenizer,
    top_k: usize,
    text_offset: u32,
    texts: HashMap<u32, String>,
    logprobs: LogProbs,
}

impl<'t> CompletionLogprobs<'t> {
    pub fn new(tokenizer: &'t dyn Tokenizer, top_k: u32) -> Self {
        Self {
            tokenizer,
            top_k: top_k as usize,
            text_offset: 0,
            texts: HashMap::new(),
            logprobs: LogProbs::default(),
        }
    }

    /// Append the prompt's tokens, the first one without a logprob.
    pub fn push_prompt(&mut self, input: &ProtoInputLogProbs) {
        for (i, (&id, &logprob)) in input
            .token_ids
            .iter()
            .zip(&input.token_logprobs)
            .enumerate()
        {
            self.push(id, logprob, input.top_logprobs.get(i));
        }
    }

    /// Append generated tokens.
    pub fn push_output(&mut self, output: &ProtoOutputLogProbs) {
        for (i, (&id, &logprob)) in output
            .token_ids
            .iter()
            .zip(&output.token_logprobs)
            .enumerate()
        {
            self.push(id, Some(logprob), output.top_logprobs.get(i));
        }
    }

    /// The tokens appended since the last take, `None` if there are none;
    /// offsets keep counting across takes.
    pub fn take(&mut self) -> Option<LogProbs> {
        (!self.logprobs.tokens.is_empty()).then(|| std::mem::take(&mut self.logprobs))
    }

    /// Every token appended since the last take.
    pub fn finish(self) -> LogProbs {
        self.logprobs
    }

    fn push(&mut self, id: u32, logprob: Option<f32>, alternatives: Option<&ProtoTopLogProbs>) {
        let token = self.text(id).to_owned();
        let top_k = self.top_k;
        let top = (logprob.is_some() && top_k > 0).then(|| {
            let mut ranked: Vec<(String, f32)> = Vec::with_capacity(top_k);
            let candidates = alternatives
                .into_iter()
                .flat_map(|top| top.token_ids.iter().zip(&top.values))
                .take(top_k);
            for (&alternative, &value) in candidates {
                let text = self.text(alternative).to_owned();
                match ranked.iter_mut().find(|(seen, _)| *seen == text) {
                    Some(entry) => entry.1 = value,
                    None => ranked.push((text, value)),
                }
            }
            ranked.into_iter().collect()
        });
        let length = token.chars().count() as u32;
        self.logprobs.tokens.push(token);
        self.logprobs.token_logprobs.push(logprob);
        self.logprobs.top_logprobs.push(top);
        self.logprobs.text_offset.push(self.text_offset);
        self.text_offset = self.text_offset.saturating_add(length);
    }

    fn text(&mut self, id: u32) -> &str {
        let tokenizer = self.tokenizer;
        self.texts.entry(id).or_insert_with(|| {
            tokenizer
                .decode(&[id], false)
                .unwrap_or_else(|_| format!("<token_{id}>"))
        })
    }
}

/// Why an `echo` with `logprobs` cannot list the prompt.
#[derive(Debug, PartialEq)]
pub(crate) enum PromptLogprobsError {
    /// The backend returned no prompt logprobs (it cannot score the prompt).
    Unavailable,
    /// The backend's prompt logprobs do not cover the prompt.
    Incomplete {
        values: usize,
        tokens: usize,
        prompt_tokens: u32,
    },
}

impl std::fmt::Display for PromptLogprobsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable => f.write_str(
                "echo with logprobs needs the prompt's logprobs, and the backend returned none",
            ),
            Self::Incomplete {
                values,
                tokens,
                prompt_tokens,
            } => write!(
                f,
                "the backend returned {values} prompt logprobs for {tokens} tokens of a \
                 {prompt_tokens}-token prompt"
            ),
        }
    }
}

impl PromptLogprobsError {
    /// The error response before any byte of the answer went out: a backend
    /// that cannot score prompts is a request this deployment cannot serve
    /// (400); prompt logprobs that miss positions are the backend's fault.
    pub(crate) fn into_response(self) -> axum::response::Response {
        match self {
            Self::Unavailable => {
                error::bad_request("prompt_logprobs_unavailable", self.to_string())
            }
            Self::Incomplete { .. } => {
                error::internal_error("prompt_logprobs_incomplete", self.to_string())
            }
        }
    }
}

/// The prompt logprobs an `echo` with `logprobs` lists: present, one value
/// per token, and one token per prompt token (`prompt_tokens`, when the
/// backend reports it).
pub(crate) fn echoed_prompt_logprobs(
    input: Option<ProtoInputLogProbs>,
    prompt_tokens: u32,
) -> Result<ProtoInputLogProbs, PromptLogprobsError> {
    let input = input.ok_or(PromptLogprobsError::Unavailable)?;
    let tokens = input.token_ids.len();
    if input.token_logprobs.len() != tokens
        || (prompt_tokens > 0 && tokens != prompt_tokens as usize)
    {
        return Err(PromptLogprobsError::Incomplete {
            values: input.token_logprobs.len(),
            tokens,
            prompt_tokens,
        });
    }
    Ok(input)
}

/// Convert OutputLogProbs to Generate format Vec<Vec<Option<f64>>>
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// Each inner vec contains [logprob (f64), token_id (u32), ...]
pub(crate) fn convert_generate_output_logprobs(
    proto_logprobs: &ProtoOutputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(&logprob, &token_id)| vec![Some(logprob as f64), Some(token_id as f64)])
        .collect()
}

/// Convert InputLogProbs to Generate format Vec<Vec<Option<f64>>>
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// First token has null logprob: [[null, token_id], [logprob, token_id], ...]
pub(crate) fn convert_generate_input_logprobs(
    proto_logprobs: &ProtoInputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(&token_logprob, &token_id)| {
            // token_logprob is Option<f32> in unified type
            let logprob_value = token_logprob.map(|v| v as f64);
            vec![logprob_value, Some(token_id as f64)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routers::grpc::proto_wrapper::ProtoTopLogProbs;

    /// Two tokens; the engine reports the chosen token and one runner-up as
    /// alternatives for each, whether or not alternatives were requested.
    fn proto() -> ProtoOutputLogProbs {
        ProtoOutputLogProbs {
            token_logprobs: vec![-0.1, -0.2],
            token_ids: vec![1, 2],
            top_logprobs: vec![
                ProtoTopLogProbs {
                    values: vec![-0.1, -1.5],
                    token_ids: vec![1, 3],
                },
                ProtoTopLogProbs {
                    values: vec![-0.2, -1.6],
                    token_ids: vec![2, 4],
                },
            ],
        }
    }

    fn alternatives(top_logprobs: Option<u32>) -> Vec<Vec<String>> {
        let mut logprobs = convert_proto_logprobs(&proto(), |id| format!("t{id}"));
        truncate_top_logprobs(&mut logprobs, top_logprobs.unwrap_or(0) as usize);
        let ChatLogProbs::Detailed {
            content: Some(items),
            ..
        } = logprobs
        else {
            panic!("detailed logprobs expected");
        };
        assert_eq!(items[0].token, "t1");
        assert_eq!(items[1].logprob, -0.2);
        items
            .iter()
            .map(|item| item.top_logprobs.iter().map(|t| t.token.clone()).collect())
            .collect()
    }

    #[test]
    fn the_logprobs_object_carries_content_and_refusal() {
        let value = serde_json::to_value(convert_proto_logprobs(&proto(), |id| format!("t{id}")))
            .expect("serialize");
        assert_eq!(value["content"][0]["token"], "t1", "{value}");
        assert_eq!(
            value.get("refusal"),
            Some(&serde_json::Value::Null),
            "{value}"
        );
    }

    #[test]
    fn top_logprobs_are_empty_unless_requested() {
        assert_eq!(alternatives(None), vec![Vec::<String>::new(), Vec::new()]);
        assert_eq!(
            alternatives(Some(0)),
            vec![Vec::<String>::new(), Vec::new()]
        );
    }

    fn top(pairs: &[(u32, f32)]) -> ProtoTopLogProbs {
        ProtoTopLogProbs {
            token_ids: pairs.iter().map(|p| p.0).collect(),
            values: pairs.iter().map(|p| p.1).collect(),
        }
    }

    /// Prompt `Hello world`, then the generated `test`.
    fn echoed(top_k: u32) -> LogProbs {
        let tokenizer = llm_tokenizer::MockTokenizer::new();
        let mut logprobs = CompletionLogprobs::new(&tokenizer, top_k);
        logprobs.push_prompt(&ProtoInputLogProbs {
            token_logprobs: vec![None, Some(-1.0)],
            token_ids: vec![1, 2],
            top_logprobs: vec![
                ProtoTopLogProbs {
                    values: vec![],
                    token_ids: vec![],
                },
                top(&[(2, -1.0), (3, -2.0), (4, -3.0)]),
            ],
        });
        logprobs.push_output(&ProtoOutputLogProbs {
            token_logprobs: vec![-0.5],
            token_ids: vec![3],
            top_logprobs: vec![top(&[(3, -0.5), (1, -1.5), (2, -2.5)])],
        });
        logprobs.finish()
    }

    fn alternatives_of(logprobs: &LogProbs) -> Vec<Option<Vec<(String, f32)>>> {
        logprobs
            .top_logprobs
            .iter()
            .map(|top| {
                top.as_ref()
                    .map(|map| map.iter().map(|(t, v)| (t.clone(), *v)).collect())
            })
            .collect()
    }

    #[test]
    fn completion_logprobs_list_the_prompt_then_the_generated_tokens() {
        let logprobs = echoed(2);
        assert_eq!(logprobs.tokens, vec!["Hello", "world", "test"]);
        assert_eq!(logprobs.token_logprobs, vec![None, Some(-1.0), Some(-0.5)]);
        assert_eq!(
            alternatives_of(&logprobs),
            vec![
                None,
                Some(vec![("world".into(), -1.0), ("test".into(), -2.0)]),
                Some(vec![("test".into(), -0.5), ("Hello".into(), -1.5)]),
            ]
        );
        assert_eq!(logprobs.text_offset, vec![0, 5, 10]);
        // The alternatives serialize most likely first.
        let json = serde_json::to_string(&logprobs.top_logprobs[1]).expect("serialize");
        assert_eq!(json, r#"{"world":-1.0,"test":-2.0}"#);
    }

    #[test]
    fn completion_logprobs_without_alternatives_list_none() {
        let logprobs = echoed(0);
        assert_eq!(logprobs.tokens.len(), 3);
        assert_eq!(alternatives_of(&logprobs), vec![None, None, None]);
    }

    #[test]
    fn alternatives_that_decode_alike_share_the_first_place_with_the_later_value() {
        let tokenizer = llm_tokenizer::MockTokenizer::new();
        let mut logprobs = CompletionLogprobs::new(&tokenizer, 3);
        // 77 and 78 are not in the mock vocabulary: both decode to "". Only
        // the first 3 alternatives count, as `{text: value for ...}` over the
        // engine's top 3 would hold them.
        logprobs.push_output(&ProtoOutputLogProbs {
            token_logprobs: vec![-0.1],
            token_ids: vec![77],
            top_logprobs: vec![top(&[(77, -0.1), (78, -0.2), (3, -0.3), (4, -0.4)])],
        });
        let logprobs = logprobs.finish();
        assert_eq!(
            alternatives_of(&logprobs),
            vec![Some(vec![(String::new(), -0.2), ("test".into(), -0.3)])]
        );
    }

    #[test]
    fn an_echoed_prompt_needs_one_logprob_per_prompt_token() {
        let input = |n: usize| ProtoInputLogProbs {
            token_logprobs: (0..n).map(|i| (i > 0).then_some(-(i as f32))).collect(),
            token_ids: (0..n as u32).collect(),
            top_logprobs: vec![],
        };
        assert!(echoed_prompt_logprobs(Some(input(3)), 3).is_ok());
        // A backend that does not report the prompt size is taken at its word.
        assert!(echoed_prompt_logprobs(Some(input(3)), 0).is_ok());
        assert_eq!(
            echoed_prompt_logprobs(None, 3).err(),
            Some(PromptLogprobsError::Unavailable)
        );
        assert_eq!(
            echoed_prompt_logprobs(Some(input(2)), 3).err(),
            Some(PromptLogprobsError::Incomplete {
                values: 2,
                tokens: 2,
                prompt_tokens: 3
            })
        );
        let response = PromptLogprobsError::Unavailable.into_response();
        assert_eq!(response.status().as_u16(), 400);
    }

    #[test]
    fn taken_completion_logprobs_keep_counting_offsets() {
        let tokenizer = llm_tokenizer::MockTokenizer::new();
        let mut logprobs = CompletionLogprobs::new(&tokenizer, 1);
        assert!(logprobs.take().is_none());
        let output = |id: u32| ProtoOutputLogProbs {
            token_logprobs: vec![-1.0],
            token_ids: vec![id],
            top_logprobs: vec![],
        };
        logprobs.push_output(&output(1));
        let first = logprobs.take().expect("one token");
        assert_eq!(first.text_offset, vec![0]);
        logprobs.push_output(&output(4));
        logprobs.push_output(&output(2));
        let second = logprobs.take().expect("two tokens");
        assert_eq!(second.tokens, vec!["token", "world"]);
        assert_eq!(second.text_offset, vec![5, 10]);
        // An engine that ranked nothing leaves the alternatives empty.
        assert_eq!(alternatives_of(&second), vec![Some(vec![]), Some(vec![])]);
    }

    #[test]
    fn top_logprobs_are_capped_at_the_requested_count() {
        assert_eq!(alternatives(Some(1)), vec![vec!["t1"], vec!["t2"]]);
        assert_eq!(
            alternatives(Some(2)),
            vec![vec!["t1", "t3"], vec!["t2", "t4"]]
        );
        assert_eq!(
            alternatives(Some(5)),
            vec![vec!["t1", "t3"], vec!["t2", "t4"]]
        );
    }
}
