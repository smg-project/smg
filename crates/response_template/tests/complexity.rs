//! Outputs that a plain port of the parser would take far longer on than
//! transformers does. transformers rescans its buffer from the current
//! position for every delimiter it commits, and backtracking can enter a
//! state many times; the port remembers searches of an unchanged buffer and
//! the states a search has been in. Each output here parses in well under a
//! second; without that, each takes half a minute or more in a debug build.

#![expect(clippy::unwrap_used, reason = "a test: a bad template should panic")]

use std::time::{Duration, Instant};

use serde_json::{json, Value};
use smg_response_template::{load_response_template, parse_response};

fn assert_fast(template: &Value, output: &str) {
    let template = load_response_template(template).unwrap();
    let took = (0..3)
        .map(|_| {
            let start = Instant::now();
            parse_response(output, &template, "", &[]).unwrap();
            start.elapsed()
        })
        .min()
        .unwrap();
    assert!(took < Duration::from_secs(5), "took {took:?}");
}

/// The template of `transformers serve` for Qwen3.5.
fn serve_qwen3_5() -> Value {
    json!({
        "defaults": {"role": "assistant"},
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {
            "thinking": {"open": "<think>", "close": "</think>"},
            "tool_calls": {
                "open_pattern": r"\s*<tool_call>\s*<function=(?P<name>[^>\n]+)>",
                "close_pattern": r"</function>\s*</tool_call>",
                "repeats": true,
                "content": "xml-inline",
                "content_args": {
                    "tag_pattern": r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>"
                },
                "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}}
            },
            "content": {"close_pattern": r"\s*(?:<\|im_end\|>|<\|endoftext\|>)"}
        }
    })
}

#[test]
fn many_regions_are_not_rescanned() {
    let unit = "text <think>plan</think> more text \
                <tool_call><function=f><parameter=a>1</parameter></function></tool_call>\n";
    assert_fast(&serve_qwen3_5(), &unit.repeat(1024));
}

#[test]
fn a_long_whitespace_run_is_not_rescanned() {
    let output = format!(
        "a{}<tool_call><function=f></function></tool_call>",
        " ".repeat(20_000)
    );
    assert_fast(&serve_qwen3_5(), &output);
}

#[test]
fn optional_groups_do_not_backtrack_exponentially() {
    // Each group doubles the ways to fail.
    let template = json!({
        "start_anchor": "<s>",
        "fields": {"x": {"open_pattern": format!("{}!", r"(?:\w|\s)?".repeat(22)), "close": "</x>"}}
    });
    assert_fast(&template, &"a".repeat(24));
}
