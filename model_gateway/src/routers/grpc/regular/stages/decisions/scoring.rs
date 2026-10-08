//! SGLang v0.5.21 SystemOne prompt and scoring semantics for OpenAI Decisions.

use std::{collections::HashSet, sync::Arc};

use llm_tokenizer::{chat_template::ChatTemplateParams, traits::Tokenizer, PromptEncoding};
use openai_protocol::decisions::{DecisionResponse, DecisionsRequest};
use serde_json::{json, Map, Value};

use crate::routers::{common::decisions::SglangDecisionAdapter, grpc::utils::encode_blocking};

pub(crate) struct DecisionPrompt {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub label_ids: Vec<u32>,
}

#[derive(Clone)]
pub(crate) struct DecisionScoring {
    adapter: SglangDecisionAdapter,
    questions: Vec<QuestionKind>,
}

#[derive(Clone, Copy)]
enum QuestionKind {
    Predicate,
    Choice(usize),
    Score(usize),
}

impl QuestionKind {
    fn count(self) -> usize {
        match self {
            Self::Predicate => 2,
            Self::Choice(count) | Self::Score(count) => count,
        }
    }
}

/// Share the HTTP adapter's supported-input validation and native question
/// mapping. After encoding, only the response mapping survives in `scoring`.
pub(crate) async fn prepare_decisions(
    request: &DecisionsRequest,
    tokenizer: Arc<dyn Tokenizer>,
    answers_open_reasoning: bool,
) -> Result<(Vec<DecisionPrompt>, DecisionScoring), String> {
    let mut adapter = SglangDecisionAdapter::new(request)?;
    let native = adapter.request();
    let state = render_text(&native["state"]);
    let questions = native["questions"]
        .as_object()
        .ok_or("Decisions question mapping is missing")?;
    let mut prompts = Vec::with_capacity(questions.len());
    let mut kinds = Vec::with_capacity(questions.len());
    let mut pair_labels = None;
    for (name, question) in questions {
        let kind = match question["type"].as_str() {
            Some("noul") => QuestionKind::Predicate,
            Some("choice") => QuestionKind::Choice(
                question["criteria"]
                    .as_object()
                    .ok_or("Invalid choice mapping")?
                    .len(),
            ),
            Some("score") => QuestionKind::Score(
                question["criteria"]
                    .as_array()
                    .ok_or("Invalid score mapping")?
                    .len(),
            ),
            _ => return Err("Invalid Decisions question mapping".into()),
        };
        let labels: Vec<String> = match kind {
            QuestionKind::Predicate => vec!["yes".into(), "no".into()],
            QuestionKind::Score(count) => (0..count).map(|i| i.to_string()).collect(),
            QuestionKind::Choice(count) if count <= 26 => (b'A'..=b'Z')
                .take(count)
                .map(|letter| char::from(letter).to_string())
                .collect(),
            QuestionKind::Choice(count) => {
                if pair_labels.is_none() {
                    pair_labels = Some(encode_pair_labels(&tokenizer).await?);
                }
                let available = pair_labels
                    .as_ref()
                    .ok_or("Pair labels were not prepared")?;
                if count > available.len() {
                    return Err(format!("question {name} has {count} choices, but the tokenizer supports only {} two-letter labels", available.len()));
                }
                available[..count].to_vec()
            }
        };
        let content = render_question(&state, question, kind, &labels)?;
        let text = render_prompt(&tokenizer, &content)?;
        reject_reasoning(&tokenizer, &content, &text, answers_open_reasoning)?;
        let token_ids = encode_ids(&tokenizer, &text).await?;
        let label_context = label_suffix(&tokenizer, &text, &token_ids).await?;
        let (label_text, label_prefix) = label_context
            .as_ref()
            .map_or((text.as_str(), token_ids.as_slice()), |(text, ids)| {
                (text.as_str(), ids.as_slice())
            });
        let mut label_ids = Vec::with_capacity(labels.len());
        // Check labels at this prompt's answer position, never in isolation.
        // A verified added-token boundary permits the same bounded suffix
        // check as native SGLang, avoiding repeated encoding of the evidence.
        for label in labels {
            let Some(id) = label_token_id(&tokenizer, label_text, label_prefix, &label).await?
            else {
                return Err(format!("question {name}: answer label {label:?} is not one distinct token after the chat prompt"));
            };
            if label_ids.contains(&id) {
                return Err(format!("question {name}: answer label {label:?} is not one distinct token after the chat prompt"));
            }
            label_ids.push(id);
        }
        prompts.push(DecisionPrompt {
            text,
            token_ids,
            label_ids,
        });
        kinds.push(kind);
    }
    adapter.release_request();
    Ok((
        prompts,
        DecisionScoring {
            adapter,
            questions: kinds,
        },
    ))
}

fn render_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn render_question(
    state: &str,
    question: &Value,
    kind: QuestionKind,
    labels: &[String],
) -> Result<String, String> {
    let instructions = question["instructions"]
        .as_str()
        .ok_or("Missing question instructions")?;
    let mut lines = vec![state.to_owned(), String::new()];
    match kind {
        QuestionKind::Predicate => {
            lines.push(format!("Is the following true? {instructions}"));
            lines.push("Answer with yes or no only.".into());
        }
        QuestionKind::Choice(_) => {
            lines.push(format!("Question: {instructions}"));
            let criteria = question["criteria"]
                .as_object()
                .ok_or("Invalid choice criteria")?;
            for (label, (name, detail)) in labels.iter().zip(criteria) {
                lines.push(format!("{label}: {name} - {}", render_text(detail)));
            }
            lines.push("Answer with the letter of one option only.".into());
        }
        QuestionKind::Score(_) => {
            lines.push(format!("Question: {instructions}"));
            let criteria = question["criteria"]
                .as_array()
                .ok_or("Invalid score criteria")?;
            for (label, detail) in labels.iter().zip(criteria) {
                lines.push(format!("{label}: {}", render_text(detail)));
            }
            lines.push("Answer with the number of one level only.".into());
        }
    }
    Ok(lines.join("\n"))
}

fn render_prompt(tokenizer: &Arc<dyn Tokenizer>, content: &str) -> Result<String, String> {
    let rendered = tokenizer
        .apply_chat_template_with_encoding(
            &[json!({"role":"user", "content":content})],
            ChatTemplateParams {
                add_generation_prompt: true,
                thinking: Some(false),
                ..Default::default()
            },
            None,
        )
        .map_err(|error| format!("Decisions chat template failed: {error}"))?;
    if !matches!(rendered.encoding, PromptEncoding::FromText)
        || rendered.unbilled_prompt_tokens != 0
    {
        return Err(
            "Decisions requires a lossless text chat template, not a deferred prompt encoder"
                .into(),
        );
    }
    Ok(rendered.text)
}

fn reject_reasoning(
    tokenizer: &Arc<dyn Tokenizer>,
    content: &str,
    prompt: &str,
    answers_open_reasoning: bool,
) -> Result<(), String> {
    let closing = content
        .rsplit('\n')
        .next()
        .ok_or("Missing Decisions prompt")?;
    let suffix = prompt
        .rfind(closing)
        .map_or(prompt, |at| &prompt[at + closing.len()..]);
    // Probe a completed assistant reply as the pinned native implementation
    // does: some templates start reasoning only when the reply is present.
    let reply = tokenizer.apply_chat_template(
        &[
            json!({"role":"user","content":closing}),
            json!({"role":"assistant","content":"DECISION_ANSWER"}),
        ],
        ChatTemplateParams {
            add_generation_prompt: false,
            thinking: Some(false),
            ..Default::default()
        },
    );
    // HyV4 templates use a model-specific suffix on their think tags. Include
    // either opening or closing tags, since an always-reasoning parser may
    // need only an explicit closing tag at the answer position.
    let suffixed: Vec<_> = [suffix, reply.as_deref().unwrap_or_default()]
        .into_iter()
        .flat_map(suffixed_reasoning_markers)
        .collect();
    let mut closed_reasoning = false;
    for (start, end) in [
        ("<think>", "</think>"),
        ("<mm:think>", "</mm:think>"),
        ("[THINK]", "[/THINK]"),
        ("<|think|>", "<|/think|>"),
        ("◁think▷", "◁/think▷"),
        ("<|START_THINKING|>", "<|END_THINKING|>"),
        ("<|open|>think<|sep|>", "<|close|>think<|sep|>"),
    ]
    .into_iter()
    .chain(
        suffixed
            .iter()
            .map(|(start, end)| (start.as_str(), end.as_str())),
    ) {
        closed_reasoning |= suffix.contains(end);
        if suffix
            .rfind(start)
            .is_some_and(|opened| suffix.rfind(end).is_none_or(|closed| opened > closed))
        {
            return Err(
                "Decisions chat template leaves a reasoning block open at the answer position"
                    .into(),
            );
        }
        // An auxiliary completed-reply render is allowed to fail, matching
        // native SGLang; the generation prompt check above still applies.
        if let Ok(reply) = &reply {
            if let Some(begin) = reply.rfind(closing) {
                if let Some(answer) = reply[begin..].find("DECISION_ANSWER") {
                    if reply[begin..begin + answer].matches(start).count()
                        > suffix.matches(start).count()
                    {
                        return Err(
                            "Decisions chat template starts every answer with reasoning".into()
                        );
                    }
                }
            }
        }
    }
    if answers_open_reasoning && !closed_reasoning {
        return Err("Decisions requires the chat template to close the reasoning block before the answer position".into());
    }
    Ok(())
}

fn suffixed_reasoning_markers(text: &str) -> impl Iterator<Item = (String, String)> + '_ {
    text.split('<').filter_map(|tag| {
        let tag = tag.strip_prefix('/').unwrap_or(tag);
        let (name, _) = tag.split_once('>')?;
        name.starts_with("think:")
            .then(|| (format!("<{name}>"), format!("</{name}>")))
    })
}

async fn encode_ids(tokenizer: &Arc<dyn Tokenizer>, text: &str) -> Result<Vec<u32>, String> {
    encode_blocking(tokenizer.clone(), text.to_owned(), false)
        .await
        .map(|encoding| encoding.token_ids().to_vec())
        .map_err(|error| format!("Decisions tokenization failed: {error}"))
}

async fn label_token_id(
    tokenizer: &Arc<dyn Tokenizer>,
    text: &str,
    ids: &[u32],
    label: &str,
) -> Result<Option<u32>, String> {
    let extended = encode_ids(tokenizer, &format!("{text}{label}")).await?;
    Ok((extended.len() == ids.len() + 1 && extended.starts_with(ids)).then(|| extended[ids.len()]))
}

/// A constant-size suffix behind an added special token makes the pair-label
/// search independent of user input. Verify it in two prompts before using it.
async fn label_suffix(
    tokenizer: &Arc<dyn Tokenizer>,
    prompt: &str,
    ids: &[u32],
) -> Result<Option<(String, Vec<u32>)>, String> {
    let added = &tokenizer.get_special_tokens().additional_special_tokens;
    for (index, id) in ids.iter().enumerate().rev() {
        let Some(token) = added
            .iter()
            .find(|token| tokenizer.token_to_id(token) == Some(*id))
        else {
            continue;
        };
        let Some(at) = prompt.rfind(token) else {
            continue;
        };
        let suffix = &prompt[at + token.len()..];
        let suffix_ids = encode_ids(tokenizer, suffix).await?;
        if suffix_ids == ids[index + 1..] {
            return Ok(Some((suffix.to_owned(), suffix_ids)));
        }
    }
    Ok(None)
}

async fn encode_pair_labels(tokenizer: &Arc<dyn Tokenizer>) -> Result<Vec<String>, String> {
    let first_prompt = render_prompt(tokenizer, "x")?;
    let first_ids = encode_ids(tokenizer, &first_prompt).await?;
    let first = label_suffix(tokenizer, &first_prompt, &first_ids).await?;
    let second_prompt = render_prompt(tokenizer, "y")?;
    let second_ids = encode_ids(tokenizer, &second_prompt).await?;
    let second = label_suffix(tokenizer, &second_prompt, &second_ids).await?;
    let Some((suffix, ids)) = first.filter(|first| second.as_ref() == Some(first)) else {
        return Err(
            "More than 26 Decisions choices requires a stable added-token answer boundary".into(),
        );
    };
    let mut labels = Vec::new();
    let mut seen = HashSet::new();
    for a in b'A'..=b'Z' {
        for b in b'A'..=b'Z' {
            let label = format!("{}{}", char::from(a), char::from(b));
            if let Some(id) = label_token_id(tokenizer, &suffix, &ids, &label).await? {
                if seen.insert(id) {
                    labels.push(label);
                }
            }
        }
    }
    Ok(labels)
}

pub(crate) fn finish_decisions(
    scoring: &DecisionScoring,
    model: &str,
    token_logprobs: &[Vec<f64>],
    input_tokens: u64,
) -> Result<DecisionResponse, String> {
    if token_logprobs.len() != scoring.questions.len() {
        return Err("Decisions returned an unexpected number of scoring rows".into());
    }
    let mut answers = Map::new();
    for (index, (kind, row)) in scoring.questions.iter().zip(token_logprobs).enumerate() {
        if row.len() != kind.count() || row.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "Decisions question {index} returned missing or nonfinite candidate logprobs"
            ));
        }
        let maximum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = row.iter().map(|value| (value - maximum).exp()).collect();
        let denominator: f64 = weights.iter().sum();
        let probabilities: Vec<f64> = weights.iter().map(|value| value / denominator).collect();
        let top = probabilities.iter().enumerate().fold(0, |best, (i, p)| {
            if *p > probabilities[best] {
                i
            } else {
                best
            }
        });
        let answer = match kind {
            QuestionKind::Predicate => json!({"type":"noul","noul":probabilities[0]}),
            QuestionKind::Choice(count) => {
                let confidence = if *count == 1 {
                    1.0
                } else {
                    ((*count as f64 * probabilities[top] - 1.0) / (*count - 1) as f64)
                        .clamp(0.0, 1.0)
                };
                let distribution: Map<String, Value> = probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (format!("o{i}"), json!(p)))
                    .collect();
                json!({"type":"choice","choice":format!("o{top}"),"confidence":confidence,"probabilities":distribution})
            }
            QuestionKind::Score(count) => {
                let score: f64 = probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * p)
                    .sum();
                let confidence = if *count == 1 {
                    1.0
                } else {
                    let spread: f64 = probabilities
                        .iter()
                        .enumerate()
                        .map(|(i, p)| p * i.abs_diff(top) as f64)
                        .sum();
                    let uniform: f64 = (0..*count)
                        .map(|i| (i as f64 - (*count - 1) as f64 / 2.0).abs())
                        .sum::<f64>()
                        / *count as f64;
                    (1.0 - spread / uniform).max(0.0)
                };
                let distribution: Map<String, Value> = probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (i.to_string(), json!(p)))
                    .collect();
                json!({"type":"score","score":score,"confidence":confidence,"probabilities":distribution})
            }
        };
        answers.insert(format!("q{index}"), answer);
    }
    scoring.adapter.map_response(&json!({"model":model,"answers":answers,"usage":{"input_tokens":input_tokens,"output_tokens":0}}))
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::{HuggingFaceTokenizer, MockTokenizer};
    use serde_json::{json, Map, Value};

    use super::*;

    const TEMPLATE: &str = "{% for message in messages %}<{{ message.role }}>{{ message.content }}</{{ message.role }}>{% endfor %}{% if add_generation_prompt %}<assistant>{% if enable_thinking|default(true) %}<think>{% endif %}{% endif %}";

    // A real HuggingFace WordLevel tokenizer and Jinja template, entirely local.
    // Distinct labels plus added role markers exercise boundary tokenization.
    fn tokenizer(template: &str) -> Arc<dyn Tokenizer> {
        let dir = tempfile::tempdir().unwrap();
        let mut vocab = Map::new();
        for label in ["[UNK]", "yes", "no", "0", "1", "2"] {
            vocab.insert(label.into(), json!(vocab.len()));
        }
        for a in b'A'..=b'Z' {
            vocab.insert(char::from(a).to_string(), json!(vocab.len()));
        }
        for a in b'A'..=b'Z' {
            for b in b'A'..=b'Z' {
                vocab.insert(
                    format!("{}{}", char::from(a), char::from(b)),
                    json!(vocab.len()),
                );
            }
        }
        let added: Vec<Value> = [
            "<user>",
            "</user>",
            "<assistant>",
            "</assistant>",
            "<think>",
            "</think>",
            "<think:6124c78e>",
            "</think:6124c78e>",
        ]
        .into_iter()
        .enumerate()
        .map(|(i, text)| {
            json!({"id":vocab.len()+i,"content":text,"single_word":false,
                "lstrip":false,"rstrip":false,"normalized":false,"special":true})
        })
        .collect();
        let data = json!({"version":"1.0","truncation":null,"padding":null,
            "added_tokens":added,"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},
            "post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":vocab,"unk_token":"[UNK]"}});
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, data.to_string()).unwrap();
        let mut tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
        tokenizer.set_chat_template(template.into()).unwrap();
        Arc::new(tokenizer)
    }

    fn request() -> DecisionsRequest {
        serde_json::from_value(json!({"model":"test", "input":"Evidence.", "questions":[
            {"type":"predicate","name":"same","instructions":"Is it true?"},
            {"type":"choice","name":"same","instructions":"Choose.","choices":[
                {"value":true},{"value":"true","description":"Literal text"}]},
            {"type":"score","instructions":"Rate.","levels":[
                {"label":"low"},{"label":"high","description":"Excellent"}]}
        ]}))
        .unwrap()
    }

    #[tokio::test]
    async fn matches_pinned_native_prompts_and_contextual_label_ids() {
        let (prompts, _) = prepare_decisions(&request(), tokenizer(TEMPLATE), false)
            .await
            .unwrap();
        assert_eq!(prompts.len(), 3);
        assert_eq!(prompts[0].text, "<user>Evidence.\n\nIs the following true? Is it true?\nAnswer with yes or no only.</user><assistant>");
        assert_eq!(prompts[1].text, "<user>Evidence.\n\nQuestion: Choose.\nA: o0 - {\"value\":true}\nB: o1 - {\"value\":\"true\",\"description\":\"Literal text\"}\nAnswer with the letter of one option only.</user><assistant>");
        assert_eq!(prompts[2].text, "<user>Evidence.\n\nQuestion: Rate.\n0: {\"label\":\"low\"}\n1: {\"label\":\"high\",\"description\":\"Excellent\"}\nAnswer with the number of one level only.</user><assistant>");
        assert_eq!(prompts[0].label_ids, [1, 2]);
        assert_eq!(prompts[1].label_ids, [6, 7]);
        assert_eq!(prompts[2].label_ids, [3, 4]);
        assert!(prompts.iter().all(|prompt| !prompt.token_ids.is_empty()));
    }

    #[tokio::test]
    async fn preserves_structured_evidence_instead_of_joining_text() {
        let mut raw = serde_json::to_value(request()).unwrap();
        raw["input"] = json!([{"role":"user","content":[{"type":"input_text","text":"A"},{"type":"input_text","text":"B"}]}]);
        let request = serde_json::from_value(raw).unwrap();
        let (prompts, _) = prepare_decisions(&request, tokenizer(TEMPLATE), false)
            .await
            .unwrap();
        assert!(prompts[0].text.starts_with("<user>[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"A\"},{\"type\":\"input_text\",\"text\":\"B\"}]}]\n\n"));
    }

    #[tokio::test]
    async fn uses_native_pair_labels_for_more_than_twenty_six_choices() {
        let mut raw = serde_json::to_value(request()).unwrap();
        raw["questions"] = json!([{"type":"choice","instructions":"Choose.","choices":(0..27).map(|i| json!({"value":format!("value{i}")})).collect::<Vec<_>>()}]);
        let request = serde_json::from_value(raw).unwrap();
        let (prompts, _) = prepare_decisions(&request, tokenizer(TEMPLATE), false)
            .await
            .unwrap();
        assert!(prompts[0].text.contains("\nAA: o0 -"));
        assert!(prompts[0].text.contains("\nBA: o26 -"));
        assert_eq!(prompts[0].label_ids, (32..59).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn rejects_labels_that_change_the_prompt_boundary() {
        let template = "{% for message in messages %}{{ message.content }}{% endfor %}prefix";
        let result = prepare_decisions(&request(), tokenizer(template), false).await;
        assert!(result.err().unwrap().contains("distinct token"));
    }

    #[tokio::test]
    async fn rejects_reasoning_that_cannot_be_disabled() {
        for template in [
            "{% for message in messages %}{{ message.content }}{% endfor %}<assistant><think>",
            "{% for message in messages %}{% if message.role == 'assistant' %}<think>{% endif %}{{ message.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}",
            "{% for message in messages %}{{ message.content }}{% endfor %}<assistant>◁think▷",
            "{% for message in messages %}{{ message.content }}{% endfor %}<assistant><|START_THINKING|>",
        ] {
            let result = prepare_decisions(&request(), tokenizer(template), false).await;
            assert!(result.err().unwrap().contains("reasoning"));
        }
    }

    #[tokio::test]
    async fn rejects_deferred_encoders_before_label_scoring() {
        let tokenizer = Arc::new(MockTokenizer::new().with_deferred_chat_ids(vec![1]));
        let result = prepare_decisions(&request(), tokenizer, false).await;
        assert!(result.err().unwrap().contains("encoder"));
    }

    #[tokio::test]
    async fn always_reasoning_parser_requires_an_explicitly_closed_block() {
        let result = prepare_decisions(&request(), tokenizer(TEMPLATE), true).await;
        assert!(result.err().unwrap().contains("reasoning"));
        let template = "{% for message in messages %}<{{ message.role }}>{{ message.content }}</{{ message.role }}>{% endfor %}{% if add_generation_prompt %}<assistant><think></think>{% endif %}";
        assert!(prepare_decisions(&request(), tokenizer(template), true)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn validates_suffixed_reasoning_markers_at_the_answer_position() {
        for template in [
            "{% for message in messages %}{{ message.content }}{% endfor %}<assistant><think:6124c78e>",
            "{% for message in messages %}{% if message.role == 'assistant' %}<think:6124c78e>{% endif %}{{ message.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}",
        ] {
            let result = prepare_decisions(&request(), tokenizer(template), false).await;
            assert!(result.err().unwrap().contains("reasoning"));
        }
        let closed = "{% for message in messages %}{{ message.content }}{% endfor %}<assistant><think:6124c78e></think:6124c78e>";
        assert!(prepare_decisions(&request(), tokenizer(closed), true)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn normalizes_selected_logprobs_stably_and_maps_native_confidence() {
        let (_, scoring) = prepare_decisions(&request(), tokenizer(TEMPLATE), false)
            .await
            .unwrap();
        let response = finish_decisions(
            &scoring,
            "served",
            &[
                vec![-1001.0, -1002.0],
                vec![0.2_f64.ln(), 0.8_f64.ln()],
                vec![0.25_f64.ln(), 0.75_f64.ln()],
            ],
            42,
        )
        .unwrap();
        let value = serde_json::to_value(response).unwrap();
        assert!(
            (value["answers"][0]["probability"].as_f64().unwrap() - 0.7310585786300049).abs()
                < 1e-12
        );
        assert_eq!(value["answers"][1]["choice"], "true");
        assert!((value["answers"][1]["confidence"].as_f64().unwrap() - 0.6).abs() < 1e-12);
        assert_eq!(value["answers"][1]["probabilities"][0]["value"], true);
        assert!((value["answers"][2]["score"].as_f64().unwrap() - 0.75).abs() < 1e-12);
        assert!((value["answers"][2]["confidence"].as_f64().unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(value["model"], "served");
        assert_eq!(value["usage"]["input_tokens"], 42);
        assert_eq!(value["usage"]["output_tokens"], 0);
    }

    #[tokio::test]
    async fn rejects_missing_extra_and_nonfinite_scoring_results() {
        let (_, scoring) = prepare_decisions(&request(), tokenizer(TEMPLATE), false)
            .await
            .unwrap();
        for rows in [
            vec![],
            vec![vec![0.0, 0.0]; 4],
            vec![vec![0.0]; 3],
            vec![vec![f64::NAN, 0.0]; 3],
            vec![vec![f64::INFINITY, 0.0]; 3],
            vec![vec![f64::NEG_INFINITY, 0.0]; 3],
        ] {
            assert!(finish_decisions(&scoring, "served", &rows, 42).is_err());
        }
    }
}
