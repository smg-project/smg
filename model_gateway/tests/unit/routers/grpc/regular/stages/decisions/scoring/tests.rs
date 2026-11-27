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

/// The label probes of one call charge one in-place budget: over a prompt
/// of a few kibibytes they encode on the calling thread up to the bound
/// and on the pool past it, as every one of them did below the earlier
/// bound.
#[tokio::test]
async fn label_probes_encode_in_place_up_to_the_bound_and_the_rest_on_the_pool() {
    use crate::routers::grpc::testing::thread_recording_tokenizer;

    let (tokenizer, ran_on) = thread_recording_tokenizer();
    let text = "word ".repeat(1024);
    let ids = tokenizer.encode(&text, false).unwrap().into_token_ids();
    ran_on.lock().unwrap().clear();
    let here = std::thread::current().id();

    let mut budget = InPlaceBudget::default();
    for label in ["A", "B", "C", "D", "E", "F"] {
        label_token_id(&tokenizer, &text, &ids, label, &mut budget)
            .await
            .unwrap();
    }

    let threads = std::mem::take(&mut *ran_on.lock().unwrap());
    assert_eq!(threads.len(), 6, "every probe is encoded");
    assert_eq!(
        threads.iter().filter(|t| **t == here).count(),
        3,
        "three probes of five kibibytes fit the bound in place; the rest go to the pool"
    );
}

/// A probe of a few bytes behind a verified boundary never takes the
/// pool, spent share or not: it yields the worker and goes on in place.
#[tokio::test]
async fn short_probes_after_a_spent_share_stay_in_place() {
    use crate::routers::grpc::testing::thread_recording_tokenizer;

    let (tokenizer, ran_on) = thread_recording_tokenizer();
    let mut budget = InPlaceBudget::default();
    let prompt = "word ".repeat(3 * 1024); // 15 KiB: the share all but spent
    encode_ids(&tokenizer, &prompt, &mut budget).await.unwrap();
    let suffix = "assistant\n";
    let suffix_ids = tokenizer.encode(suffix, false).unwrap().into_token_ids();
    ran_on.lock().unwrap().clear();
    let here = std::thread::current().id();

    for probe in 0..2_000 {
        let label = format!("{probe}");
        label_token_id(&tokenizer, suffix, &suffix_ids, &label, &mut budget)
            .await
            .unwrap();
    }

    let threads = std::mem::take(&mut *ran_on.lock().unwrap());
    assert_eq!(threads.len(), 2_000, "every probe is encoded");
    assert!(
        threads.iter().all(|t| *t == here),
        "no probe of a few bytes pays the pool's round trip"
    );
}

#[tokio::test]
async fn rejects_deferred_encoders_before_label_scoring() {
    let tokenizer = Arc::new(MockTokenizer::new().with_deferred_chat_ids(vec![1]));
    let result = prepare_decisions(&request(), tokenizer, false).await;
    assert!(result.err().unwrap().contains("deferred encode"));
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
        (value["answers"][0]["probability"].as_f64().unwrap() - 0.7310585786300049).abs() < 1e-12
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
