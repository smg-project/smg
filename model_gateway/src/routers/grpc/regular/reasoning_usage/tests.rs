use llm_tokenizer::{
    chat_template::{ThinkingKeyName, ThinkingToggle},
    traits::{Decoder, Encoder, Encoding, SpecialTokens},
};
use openai_protocol::chat::ChatCompletionRequest;
use smg_grpc_client::{sglang_proto as sglang, vllm_proto as vllm};

use super::*;

const START: u32 = 0x110000;
const END: u32 = START + 1;
const EOS: u32 = START + 2;
const M3_MARKERS: [(&str, u32); 3] = [("<mm:think>", START), ("</mm:think>", END), ("<eos>", EOS)];
const MARKERS: [(&str, u32); 3] = [("<think>", START), ("</think>", END), ("<eos>", EOS)];

/// Character tokens plus dedicated control tokens, including a prompt-opened
/// thinking template. Used by the exact-gRPC-frame tests as well.
#[derive(Default)]
pub(crate) struct CountingTokenizer {
    special: SpecialTokens,
    minimax: bool,
}

impl CountingTokenizer {
    pub(crate) fn minimax() -> Self {
        Self {
            minimax: true,
            ..Self::default()
        }
    }

    fn markers(&self) -> &'static [(&'static str, u32)] {
        if self.minimax {
            &M3_MARKERS
        } else {
            &MARKERS
        }
    }
}

impl Encoder for CountingTokenizer {
    fn encode(&self, mut text: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
        let mut ids = Vec::new();
        while !text.is_empty() {
            if let Some(&(marker, id)) = self
                .markers()
                .iter()
                .find(|(marker, _)| text.starts_with(marker))
            {
                ids.push(id);
                text = &text[marker.len()..];
            } else {
                let c = text.chars().next().unwrap();
                ids.push(u32::from(c));
                text = &text[c.len_utf8()..];
            }
        }
        Ok(Encoding::Plain(ids))
    }

    fn encode_batch(&self, texts: &[&str], special: bool) -> anyhow::Result<Vec<Encoding>> {
        texts
            .iter()
            .map(|text| self.encode(text, special))
            .collect()
    }
}

impl Decoder for CountingTokenizer {
    fn decode(&self, ids: &[u32], skip_special: bool) -> anyhow::Result<String> {
        let mut text = String::new();
        for &id in ids {
            if let Some(&(marker, _)) = self
                .markers()
                .iter()
                .find(|(_, marker_id)| *marker_id == id)
            {
                // The reasoning markers must survive detokenization.
                if id != EOS || !skip_special {
                    text.push_str(marker);
                }
            } else {
                text.push(char::from_u32(id).ok_or_else(|| anyhow::anyhow!("invalid token"))?);
            }
        }
        Ok(text)
    }
}

impl Tokenizer for CountingTokenizer {
    fn vocab_size(&self) -> usize {
        (EOS + 1) as usize
    }
    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special
    }
    fn token_to_id(&self, text: &str) -> Option<u32> {
        self.markers()
            .iter()
            .find(|(marker, _)| *marker == text)
            .map(|(_, id)| *id)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.decode(&[id], false).ok()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn eos_token_ids(&self) -> &[u32] {
        &[EOS]
    }
    fn thinking_toggle(&self) -> ThinkingToggle {
        ThinkingToggle::DefaultOn
    }
    fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        Some(ThinkingKeyName::EnableThinking)
    }
    fn think_in_prefill(&self) -> bool {
        true
    }
}

fn request(thinking: bool) -> ChatResponseSpec {
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "qwen3", "messages": [],
        "chat_template_kwargs": {"enable_thinking": thinking}
    }))
    .unwrap();
    ChatResponseSpec::from(&request)
}

fn counter(thinking: bool) -> Counter {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CountingTokenizer::default());
    Counter::new(
        &ParserFactory::new(),
        Some("qwen3"),
        "qwen3",
        &tokenizer,
        &request(thinking),
    )
    .unwrap()
}

fn complete(index: u32, ids: Vec<u32>) -> ProtoGenerateComplete {
    ProtoGenerateComplete::Vllm(Box::new(vllm::GenerateComplete {
        index,
        completion_tokens: ids.len() as u32,
        output_ids: ids,
        finish_reason: "stop".into(),
        ..Default::default()
    }))
}

#[test]
fn counts_generated_reasoning_not_markers_eos_or_answer() {
    for (thinking, ids, expected) in [
        (false, vec![START, 1, 2, END, 3], 2),
        (true, vec![1, 2, END, 3], 2),
        (true, vec![START, 1, END, 3], 1),
        (true, vec![1, 2, EOS], 2),
        (false, vec![1, 2], 0),
        (false, vec![START, END, 3], 0),
        (true, vec![END, 3, START, 4], 0),
        (true, vec![], 0),
    ] {
        assert_eq!(counter(thinking).count(&ids), Some(expected));
    }
}

#[test]
fn configured_parser_and_continuation_control_support() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CountingTokenizer::default());
    let factory = ParserFactory::new();
    let mut req = request(false);
    let r1 = Counter::new(&factory, Some("deepseek_r1"), "alias", &tokenizer, &req).unwrap();
    assert_eq!(r1.count(&[1, 2, END]), Some(2));
    assert!(Counter::new(&factory, Some("passthrough"), "qwen3", &tokenizer, &req).is_none());
    assert!(Counter::new(
        &factory,
        Some("qwen3"),
        "alias",
        &(Arc::new(llm_tokenizer::mock::MockTokenizer::new()) as Arc<dyn Tokenizer>),
        &req
    )
    .is_none());
    assert_eq!(counter(false).count(&[END, START, 1, END]), None);
    req.continues_final_assistant = true;
    assert!(Counter::new(&factory, Some("qwen3"), "qwen3", &tokenizer, &req).is_none());
}

#[test]
fn resolves_added_control_tokens_with_a_huggingface_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    let added: Vec<_> = ["<think>", "</think>"]
        .into_iter()
        .enumerate()
        .map(|(index, text)| {
            serde_json::json!({
                "id": index + 4, "content": text, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false, "special": true
            })
        })
        .collect();
    let data = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": added, "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "foo": 1, "bar": 2, "answer": 3}, "unk_token": "[UNK]"}
    });
    std::fs::write(&path, data.to_string()).unwrap();
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(llm_tokenizer::HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap());
    let counter = Counter::new(
        &ParserFactory::new(),
        Some("qwen3"),
        "alias",
        &tokenizer,
        &request(false),
    )
    .unwrap();
    let ids = tokenizer
        .encode("<think>foo bar</think>answer", false)
        .unwrap();
    assert_eq!(ids.token_ids(), &[4, 1, 2, 5, 3]);
    assert_eq!(counter.count(ids.token_ids()), Some(2));
}
#[test]
fn sglang_never_initializes_counter_or_changes_usage() {
    let mut tracker = VllmReasoningUsage::default();
    tracker.record_chunk(
        &ProtoGenerateStreamChunk::Sglang(sglang::GenerateStreamChunk {
            token_ids: vec![START, 1, END],
            ..Default::default()
        }),
        || panic!("SGLang must not resolve a local counter"),
    );
    for tokens in [0, 3] {
        let complete = ProtoGenerateComplete::Sglang(sglang::GenerateComplete {
            reasoning_tokens: tokens,
            ..Default::default()
        });
        tracker.record(&complete, || {
            panic!("SGLang must not resolve a local counter")
        });
        let mut usage = Usage::from_counts(5, 10).with_reasoning_tokens(tokens);
        let before = serde_json::to_value(&usage).unwrap();
        tracker.apply(&mut usage, 1);
        assert_eq!(serde_json::to_value(&usage).unwrap(), before);
    }
}

#[test]
fn counts_choices_once_and_preserves_known_zero_and_speculative_details() {
    let mut tracker = VllmReasoningUsage::default();
    tracker.record(&complete(0, vec![START, 1, END]), || Some(counter(false)));
    tracker.record(&complete(0, vec![START, 1, END]), || {
        panic!("initialized once")
    });
    tracker.record(&complete(1, vec![2, 3]), || panic!("initialized once"));
    let mut usage = Usage::from_counts(5, 5).with_speculative_tokens(2, 3);
    tracker.apply(&mut usage, 2);
    let details = usage.completion_tokens_details.unwrap();
    assert_eq!(details.reasoning_tokens, Some(1));
    assert_eq!(details.accepted_prediction_tokens, Some(2));
    assert_eq!(usage.total_tokens, 10);

    let mut zero = VllmReasoningUsage::default();
    zero.record(&complete(0, vec![2, 3]), || Some(counter(false)));
    let mut usage = Usage::from_counts(5, 2);
    zero.apply(&mut usage, 1);
    assert_eq!(
        usage.completion_tokens_details.unwrap().reasoning_tokens,
        Some(0)
    );
}

#[test]
fn partial_unknown_and_malformed_completions_do_not_claim_a_total() {
    for bad in [None, Some(complete(1, vec![1])), Some(complete(1, vec![]))] {
        let mut tracker = VllmReasoningUsage::default();
        tracker.record(&complete(0, vec![START, 1, END]), || Some(counter(false)));
        if let Some(mut bad) = bad {
            if let ProtoGenerateComplete::Vllm(ref mut c) = bad {
                c.completion_tokens = 2;
            }
            tracker.record(&bad, || panic!("initialized once"));
        }
        let mut usage = Usage::from_counts(5, 5);
        tracker.apply(&mut usage, 2);
        assert!(usage.completion_tokens_details.is_none());
    }
    let mut unsupported = VllmReasoningUsage::default();
    unsupported.record(&complete(0, vec![1]), || None);
    let mut usage = Usage::from_counts(5, 1);
    unsupported.apply(&mut usage, 1);
    assert!(usage.completion_tokens_details.is_none());

    let mut stopped = VllmReasoningUsage::default();
    stopped.record(&complete(0, vec![START, 1, END, 2]), || {
        Some(counter(false))
    });
    let mut usage = Usage::from_counts(5, 3);
    stopped.apply(&mut usage, 1);
    assert!(usage.completion_tokens_details.is_none());
}

#[test]
fn cached_markers_preserve_request_state_and_tokenizer_identity() {
    let factory = ParserFactory::new();
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CountingTokenizer::default());
    let make = |thinking| {
        Counter::new(
            &factory,
            Some("qwen3"),
            "alias",
            &tokenizer,
            &request(thinking),
        )
        .unwrap()
    };
    let off = make(false);
    let on = make(true);
    assert!(Arc::ptr_eq(&off.markers, &on.markers));
    assert_eq!(off.count(&[1, 2]), Some(0));
    assert_eq!(off.count(&[1, 2, END]), None);
    assert_eq!(on.count(&[1, 2, END]), Some(2));
    // Different parser state must not be cached with the token IDs.
    let r1 = Counter::new(
        &factory,
        Some("deepseek_r1"),
        "alias",
        &tokenizer,
        &request(false),
    )
    .unwrap();
    assert_eq!(r1.count(&[1, 2, END]), Some(2));
    // The same tokenizer with a different marker pair must not reuse a hit.
    for _ in 0..2 {
        assert!(Counter::new(
            &factory,
            Some("minimax_m3"),
            "alias",
            &tokenizer,
            &request(false)
        )
        .is_none());
    }
    let other: Arc<dyn Tokenizer> = Arc::new(llm_tokenizer::mock::MockTokenizer::new());
    assert!(Counter::new(&factory, Some("qwen3"), "alias", &other, &request(false)).is_none());
    assert_eq!(make(true).count(&[1, END]), Some(1));
    let mut continuation = request(true);
    continuation.continues_final_assistant = true;
    assert!(Counter::new(&factory, Some("qwen3"), "alias", &tokenizer, &continuation).is_none());
    let weak = Arc::downgrade(&tokenizer);
    drop(tokenizer);
    assert!(
        weak.upgrade().is_none(),
        "thread cache must not retain tokenizer"
    );
}

#[test]
fn exact_total_requires_every_valid_choice_and_matching_completion() {
    let mut tracker = VllmReasoningUsage::default();
    tracker.record(&complete(0, vec![START, 1, END]), || Some(counter(false)));
    assert_eq!(tracker.exact_total(3, 1), Some(1));
    assert_eq!(tracker.exact_total(2, 1), None);
    assert_eq!(tracker.exact_total(3, 2), None);
    // A later invalid choice must keep the parser fallback available.
    tracker.record(&complete(u32::MAX, vec![START, 2, 3, END]), || {
        panic!("initialized")
    });
    assert_eq!(tracker.exact_total(7, 2), Some(3));
    let mut bad = complete(u32::MAX, vec![]);
    if let ProtoGenerateComplete::Vllm(c) = &mut bad {
        c.finish_reason = "abort".into();
    }
    tracker.record(&bad, || panic!("initialized"));
    assert_eq!(tracker.exact_total(3, 2), None);
}

#[test]
fn unavailable_exact_count_preserves_existing_estimate() {
    for estimate in [0, 7] {
        for supported in [false, true] {
            let mut tracker = VllmReasoningUsage::default();
            tracker.record(&complete(0, vec![START, 1, END]), || {
                supported.then(|| counter(false))
            });
            let mut usage = Usage::from_counts(5, 4);
            usage
                .completion_tokens_details
                .get_or_insert_default()
                .reasoning_tokens = Some(estimate);
            let expected = usage.clone();
            tracker.apply(&mut usage, 1);
            assert_eq!(
                serde_json::to_value(usage).unwrap(),
                serde_json::to_value(expected).unwrap()
            );
        }
    }
}
