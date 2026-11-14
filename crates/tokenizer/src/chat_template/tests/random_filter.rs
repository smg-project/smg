use std::collections::BTreeSet;

use super::*;

fn render(template: &str) -> String {
    ChatTemplateProcessor::new(template.to_string())
        .unwrap()
        .apply_chat_template(
            &[serde_json::json!({"role": "user", "content": "Hi"})],
            ChatTemplateParams::default(),
        )
        .unwrap()
}

/// A template that draws a nonce after a role marker with Jinja2's `random`
/// filter renders: a draw from a string of two-byte characters is one of its
/// characters (not a byte), a draw from a list one of its items, a draw from
/// `range()` one of its numbers, and the draws are fresh on every render, as
/// the reference renderer makes them.
#[test]
fn jinja_s_random_filter_draws_from_a_string_a_list_and_a_range() {
    let template = "{% set chars = '¡°Äºäé' %}{% for m in messages %}\
                    <|open|>{{ m.role }}<|:|>{{ chars | random }}{{ chars | random }}<|close|>\
                    {{ m.content }}<|open_end|>{% endfor %}{{ [1, 2, 3] | random }}/{{ range(3) | random }}";
    let mut nonces = BTreeSet::new();
    for _ in 0..32 {
        let rendered = render(template);
        let nonce = rendered
            .strip_prefix("<|open|>user<|:|>")
            .and_then(|rest| rest.split_once("<|close|>Hi<|open_end|>"))
            .unwrap_or_else(|| {
                panic!("the render keeps the template's bytes around the draws: {rendered}")
            });
        assert_eq!(nonce.0.chars().count(), 2, "{rendered}");
        assert!(nonce.0.chars().all(|c| "¡°Äºäé".contains(c)), "{rendered}");
        let (item, number) = nonce
            .1
            .split_once('/')
            .unwrap_or_else(|| panic!("{rendered}"));
        assert!(["1", "2", "3"].contains(&item), "{rendered}");
        assert!(["0", "1", "2"].contains(&number), "{rendered}");
        nonces.insert(nonce.0.to_string());
    }
    assert!(
        nonces.len() > 1,
        "the draws are fresh on every render: {nonces:?}"
    );
}
