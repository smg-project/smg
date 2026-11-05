use std::sync::Arc;

use minijinja::value::{Enumerator, Object};

use super::*;

fn render(template: &str, messages: &[JsonValue], params: ChatTemplateParams) -> String {
    ChatTemplateProcessor::new(template.to_string())
        .unwrap()
        .apply_chat_template(messages, params)
        .unwrap()
}

fn user(text: &str) -> JsonValue {
    serde_json::json!({"role": "user", "content": text})
}

/// The context answers the render's names as the map did: a set token reads,
/// an unset one is undefined (not absent), a name the render does not set is
/// absent.
#[test]
fn the_render_context_answers_the_names_a_template_reads() {
    let tokens = crate::traits::SpecialTokens {
        bos_token: Some("<s>".to_string()),
        ..Default::default()
    };
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        special_tokens: Some(&tokens),
        ..Default::default()
    };
    let template =
        "{{ bos_token }}{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}\
                    {% if add_generation_prompt %}>{% endif %}\
                    {% if eos_token is defined %}EOS{% else %}no-eos{% endif %}\
                    {% if tools is none %}no-tools{% endif %}\
                    {% if nothing is defined %}?{% endif %}";
    assert_eq!(
        render(template, &[user("hi")], params),
        "<s>user:hi;>no-eosno-tools"
    );
}

/// A caller's kwarg reads beside the names the render sets and fills one the
/// render leaves undefined (an unset special token); a name the render sets
/// keeps the render's value (the request's messages and flag, a set token),
/// as the merge of the two context maps read them: the render's map first.
#[test]
fn a_template_kwarg_reads_beside_the_base_and_fills_an_unset_token() {
    let tokens = crate::traits::SpecialTokens {
        bos_token: Some("<s>".to_string()),
        ..Default::default()
    };
    let kwargs: HashMap<String, JsonValue> = [
        ("custom".to_string(), JsonValue::String("yes".to_string())),
        (
            "messages".to_string(),
            serde_json::json!([{"role": "user", "content": "swapped"}]),
        ),
        ("add_generation_prompt".to_string(), JsonValue::Bool(true)),
        (
            "bos_token".to_string(),
            JsonValue::String("<kw>".to_string()),
        ),
        (
            "eos_token".to_string(),
            JsonValue::String("<eos>".to_string()),
        ),
    ]
    .into_iter()
    .collect();
    let params = ChatTemplateParams {
        add_generation_prompt: false,
        special_tokens: Some(&tokens),
        template_kwargs: Some(&kwargs),
        ..Default::default()
    };
    let template = "{{ custom }}|{% for m in messages %}{{ m.content }}{% endfor %}\
                    {% if add_generation_prompt %}>{% endif %}|{{ bos_token }}{{ eos_token }}";
    assert_eq!(
        render(template, &[user("original")], params),
        "yes|original|<s><eos>"
    );
}

/// The context enumerates the render's names, plus a caller's kwargs once
/// each, so `items`/`in` over the root read what a map held.
#[test]
fn the_render_context_enumerates_its_names_once() {
    let base = Arc::new(RenderContext {
        messages: Value::from_object(Vec::<Value>::new()),
        add_generation_prompt: Value::from(false),
        strftime_now: Value::from(()),
        tools: Value::from(()),
        documents: Value::from(()),
        bos_token: Value::UNDEFINED,
        eos_token: Value::UNDEFINED,
        unk_token: Value::UNDEFINED,
        pad_token: Value::UNDEFINED,
        kwargs: None,
    });
    assert!(matches!(base.enumerate(), Enumerator::Str(names) if names == RENDER_CONTEXT_NAMES));

    let kwargs: HashMap<String, JsonValue> = [
        ("custom".to_string(), JsonValue::Bool(true)),
        ("tools".to_string(), JsonValue::Null),
    ]
    .into_iter()
    .collect();
    let with_kwargs = Arc::new(RenderContext {
        kwargs: Some(Value::from_serialize(&kwargs)),
        ..Arc::try_unwrap(base).unwrap()
    });
    let Enumerator::Values(names) = with_kwargs.enumerate() else {
        panic!("a context with kwargs enumerates values");
    };
    let names: Vec<&str> = names.iter().filter_map(Value::as_str).collect();
    assert_eq!(names.len(), RENDER_CONTEXT_NAMES.len() + 1);
    assert_eq!(names.iter().filter(|name| **name == "tools").count(), 1);
    assert!(names.contains(&"custom"));
    assert_eq!(
        with_kwargs.get_value_by_str("custom"),
        Some(Value::from(true))
    );
    assert_eq!(with_kwargs.get_value_by_str("absent"), None);
}

/// The next render starts from the last one's length, capped.
#[test]
fn the_rendered_length_sizes_the_next_render_and_caps() {
    let length = RenderedLength::default();
    assert_eq!(length.hint(), 0);
    length.record(3_500);
    assert_eq!(length.hint(), 3_500);
    length.record(10 * RENDERED_LENGTH_CAP);
    assert_eq!(length.hint(), RENDERED_LENGTH_CAP);

    let processor = ChatTemplateProcessor::new("{{ messages[0].content }}".to_string()).unwrap();
    let long = "x".repeat(2_048);
    assert_eq!(
        processor
            .apply_chat_template(&[user(&long)], ChatTemplateParams::default())
            .unwrap(),
        long
    );
    assert_eq!(processor.output_len.hint(), 2_048);
    assert_eq!(
        processor
            .apply_chat_template(&[user("short")], ChatTemplateParams::default())
            .unwrap(),
        "short"
    );
    assert_eq!(processor.output_len.hint(), 5);
}

/// A short render after a long one through the same template does not carry
/// the long one's buffer into its request; a render about as long as the last
/// keeps the buffer it started from.
#[test]
fn a_short_render_after_a_long_one_gives_the_slack_back() {
    let processor = ChatTemplateProcessor::new("{{ messages[0].content }}".to_string()).unwrap();
    let long = "x".repeat(RENDERED_LENGTH_CAP);
    processor
        .apply_chat_template(&[user(&long)], ChatTemplateParams::default())
        .unwrap();
    assert_eq!(processor.output_len.hint(), RENDERED_LENGTH_CAP);

    let short = processor
        .apply_chat_template(&[user("short")], ChatTemplateParams::default())
        .unwrap();
    assert_eq!(short, "short");
    assert!(
        short.capacity() <= 2 * short.len() + RENDERED_SLACK,
        "a {}-byte text kept a capacity of {}",
        short.len(),
        short.capacity()
    );

    // A render a little shorter than the hint keeps the buffer it started
    // from: the long text sets the hint back to the cap, the near one fills
    // most of it, and the guard leaves the capacity at the cap (a shrink on
    // every render would leave it at the length).
    processor
        .apply_chat_template(&[user(&long)], ChatTemplateParams::default())
        .unwrap();
    let near = "y".repeat(RENDERED_LENGTH_CAP - 1_000);
    let rendered = processor
        .apply_chat_template(&[user(&near)], ChatTemplateParams::default())
        .unwrap();
    assert_eq!(rendered.len(), near.len());
    assert!(
        rendered.capacity() >= RENDERED_LENGTH_CAP,
        "a {}-byte render after a long one kept a capacity of {}, under the cap {}",
        rendered.len(),
        rendered.capacity(),
        RENDERED_LENGTH_CAP
    );
}
