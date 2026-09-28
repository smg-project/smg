//! Public-API behavior matrix for complete and incremental parsing.
//!
//! The cases in this file intentionally exercise the parser only through its
//! public API.  Each streaming scenario is replayed as one chunk and at every
//! byte boundary, and every replay must produce the same outcome.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration-test helpers fail immediately when a fixture or assertion is invalid"
)]

use std::collections::BTreeMap;

use response_template_parser::{
    ClosePattern, ContentArgs, DelimiterBound, FieldTemplate, ParserConfig, ResponseTemplate,
    ResponseTemplateError, ResponseTemplateParser, Transform, ValueParser, ValueParserArgs,
};
use serde_json::{json, Value};

const FOUR_MIB: usize = 4_194_304;
const MODEL: &str = "generic-response-template-test";
const PREFIX: &str = "<|start|>assistant";

#[test]
fn default_limits_are_4_mib() {
    let config = ParserConfig::default();
    assert_eq!(config.max_pending_bytes, FOUR_MIB);
    assert_eq!(config.max_structured_field_bytes, FOUR_MIB);
    assert_eq!(config.max_body_bytes, FOUR_MIB);
}

fn template() -> ResponseTemplate {
    let mut fields = BTreeMap::new();
    fields.insert(
        "thinking".to_owned(),
        FieldTemplate {
            open_pattern: r"<\|channel\|>analysis<\|message\|>".to_owned(),
            close: ClosePattern::One("<|end|>".to_owned()),
            content: "text".to_owned(),
            content_args: None,
            repeats: false,
            transform: None,
        },
    );
    fields.insert(
        "content".to_owned(),
        FieldTemplate {
            open_pattern: r"<\|channel\|>final<\|message\|>".to_owned(),
            close: ClosePattern::Many(vec!["<|return|>".to_owned(), "<|end|>".to_owned()]),
            content: "text".to_owned(),
            content_args: None,
            repeats: false,
            transform: None,
        },
    );
    fields.insert(
        "tool_calls".to_owned(),
        FieldTemplate {
            open_pattern: concat!(
                r"(?:<\|start\|>assistant )?to=(?P<name>.+?)",
                r"<\|channel\|>analysis(?: <\|constrain\|>xml)?<\|message\|>"
            )
            .to_owned(),
            close: ClosePattern::Many(vec!["<|end|>".to_owned(), "<|call|>".to_owned()]),
            content: "xml-inline".to_owned(),
            content_args: Some(ContentArgs {
                tag_pattern: concat!(r#"<arg name=\"(?P<key>[^\"]+)\">"#, r"(?P<value>.*?)</arg>")
                    .to_owned(),
                value_parser: ValueParser {
                    name: "text".to_owned(),
                    args: ValueParserArgs { strip: false },
                },
            }),
            repeats: true,
            transform: Some(Transform(json!({
                "type": "function",
                "function": {
                    "name": "{name}",
                    "arguments": "{content}"
                }
            }))),
        },
    );

    ResponseTemplate {
        defaults: BTreeMap::from([
            ("thinking".to_owned(), Value::String(String::new())),
            ("content".to_owned(), Value::String(String::new())),
            ("tool_calls".to_owned(), Value::Array(Vec::new())),
        ]),
        start_anchor_pattern: r"<\|start\|>assistant".to_owned(),
        fields,
    }
}

fn config_with_limits(pending: usize, structured: usize, body: usize) -> ParserConfig {
    ParserConfig {
        max_pending_bytes: pending,
        max_structured_field_bytes: structured,
        max_body_bytes: body,
    }
}

fn parser(config: ParserConfig) -> ResponseTemplateParser {
    ResponseTemplateParser::new(MODEL, template(), config).expect("fixture must load")
}

fn complete_wire() -> String {
    concat!(
        "<|channel|>analysis<|message|>",
        "thinking\u{200b} café",
        "<|end|>",
        "<|start|>assistant to=weather.lookup<|channel|>analysis <|constrain|>xml<|message|>",
        "<arg name=\"city\"> Montréal </arg>",
        "<arg name=\"units\">metric</arg>",
        "<|call|>",
        "<|start|>assistant to=calendar.lookup<|channel|>analysis <|constrain|>xml<|message|>",
        "<arg name=\"day\">tomorrow</arg>",
        "<|end|>",
        "<|channel|>final<|message|>",
        "done\u{301}",
        "<|return|>"
    )
    .to_owned()
}

#[derive(Debug, Clone, PartialEq)]
struct NormalizedError {
    variant: String,
    model_name: String,
    field: String,
    limit: usize,
}

impl From<&ResponseTemplateError> for NormalizedError {
    fn from(error: &ResponseTemplateError) -> Self {
        Self {
            variant: error.variant_name().to_owned(),
            model_name: error.model_name().to_owned(),
            field: error.field().to_owned(),
            limit: error.limit(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
struct Replay {
    output: response_template_parser::ParseOutput,
    error: Option<NormalizedError>,
    overflow_feed_output_empty: Option<bool>,
    later_feed_returns_same_error: Option<bool>,
    finish_returns_same_error: Option<bool>,
}

fn replay(config: ParserConfig, input: &[u8], chunk_lengths: &[usize]) -> Replay {
    assert_eq!(chunk_lengths.iter().sum::<usize>(), input.len());
    let parser = parser(config);
    let mut stream = parser.stream(PREFIX);
    let mut observed = response_template_parser::ParseOutput::default();
    let mut cursor = 0;
    for &length in chunk_lengths {
        let end = cursor + length;
        match stream.feed(&input[cursor..end]) {
            Ok(output) => observed.merge(output),
            Err(error) => {
                let normalized = NormalizedError::from(&error);
                let output_empty = observed.is_empty();
                let later = stream
                    .feed(b"input-cannot-clear-poison")
                    .err()
                    .map(|later| NormalizedError::from(&later) == normalized)
                    .unwrap_or(false);
                let finish = stream
                    .finish()
                    .err()
                    .map(|later| NormalizedError::from(&later) == normalized)
                    .unwrap_or(false);
                return Replay {
                    output: observed,
                    error: Some(normalized),
                    overflow_feed_output_empty: Some(output_empty),
                    later_feed_returns_same_error: Some(later),
                    finish_returns_same_error: Some(finish),
                };
            }
        }
        cursor = end;
    }

    match stream.finish() {
        Ok(output) => {
            observed.merge(output);
            Replay {
                output: observed,
                ..Replay::default()
            }
        }
        Err(error) => Replay {
            output: observed,
            error: Some(NormalizedError::from(&error)),
            ..Replay::default()
        },
    }
}

fn all_chunkings(config: ParserConfig, input: &[u8]) -> Replay {
    let baseline = replay(config, input, &[input.len()]);

    // Two chunks at every byte offset, including offsets inside UTF-8 scalars
    // and inside both regex and literal delimiters.
    for offset in 0..=input.len() {
        let split = replay(config, input, &[offset, input.len() - offset]);
        assert_eq!(split, baseline, "outcome changed at split offset {offset}");
    }

    // Also exercise maximally fragmented byte-at-a-time delivery (feed accepts
    // bytes, so invalid partial UTF-8 is held).
    let byte_lengths = vec![1; input.len()];
    let bytewise = replay(config, input, &byte_lengths);
    assert_eq!(bytewise, baseline, "byte-at-a-time outcome changed");

    baseline
}

fn output_delimiter_leakage(output: &response_template_parser::ParseOutput) -> usize {
    let encoded = serde_json::to_string(output).expect("output must serialize");
    [
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<arg",
        "</arg>",
    ]
    .iter()
    .map(|delimiter| encoded.matches(delimiter).count())
    .sum()
}

fn one_chunk_equals_every_byte_split(config: ParserConfig, input: &str) -> Replay {
    all_chunkings(config, input.as_bytes())
}

fn invalid_template(pattern: &str) -> Result<ResponseTemplateParser, ResponseTemplateError> {
    let mut invalid = template();
    pattern.clone_into(
        &mut invalid
            .fields
            .get_mut("thinking")
            .expect("thinking fixture")
            .open_pattern,
    );
    ResponseTemplateParser::new(MODEL, invalid, ParserConfig::default())
}

fn exactly_cap(config: ParserConfig, input: &[u8]) {
    let parser = parser(config);
    let mut stream = parser.stream(PREFIX);
    let accepted = stream.feed(input).is_ok() && stream.finish().is_ok();
    assert!(accepted, "an input exactly at its byte cap was rejected");
}

/// Feed `input` in one chunk, require the typed overflow, and require the
/// stream to stay poisoned with the same error.
fn cap_plus_one(config: ParserConfig, input: &[u8], variant: &str, field: &str) -> NormalizedError {
    let parser = parser(config);
    let mut stream = parser.stream(PREFIX);
    let error = match stream.feed(input) {
        Err(error) => error,
        Ok(output) => panic!("cap plus one returned output instead of overflow: {output:?}"),
    };
    let normalized = NormalizedError::from(&error);
    assert_eq!(normalized.variant, variant);
    assert_eq!(normalized.model_name, MODEL);
    assert_eq!(normalized.field, field);
    let expected_limit = match field {
        "max_pending_bytes" => config.max_pending_bytes,
        "max_structured_field_bytes" => config.max_structured_field_bytes,
        "max_body_bytes" => config.max_body_bytes,
        other => panic!("unexpected limit field {other}"),
    };
    assert_eq!(normalized.limit, expected_limit);

    let later_error = stream
        .feed(b"later input")
        .expect_err("a poisoned parser must reject later feeds");
    let finish_error = stream
        .finish()
        .expect_err("a poisoned parser must reject finish");
    assert_eq!(NormalizedError::from(&later_error), normalized);
    assert_eq!(NormalizedError::from(&finish_error), normalized);
    normalized
}

fn tool_wire(name: &str, body: &str) -> String {
    format!(" to={name}<|channel|>analysis <|constrain|>xml<|message|>{body}<|end|>")
}

fn repeated_tool_wire(name: &str, body: &str) -> String {
    format!(
        "<|start|>assistant to={name}<|channel|>analysis <|constrain|>xml<|message|>{body}<|end|>"
    )
}

#[test]
fn incremental_feed_emits_and_drains_many_repeated_calls() {
    let pending_cap = 128;
    let parser = parser(config_with_limits(pending_cap, 64, 128));
    let mut stream = parser.stream(PREFIX);

    let thinking = stream
        .feed(b"<|channel|>analysis<|message|>ready<|end|>")
        .expect("a complete thinking field must emit from feed");
    assert_eq!(thinking.thinking, "ready");
    assert!(thinking.content.is_empty());
    assert!(thinking.tool_calls.is_empty());
    assert!(thinking.wire_bytes.is_empty());
    assert_eq!(stream.pending_bytes(), 0);

    let mut emitted_calls = 0usize;
    let mut largest_pending = stream.pending_bytes();
    for index in 0..128 {
        let body = format!("<arg name=\"index\">{index}</arg>");
        let wire = repeated_tool_wire("counter", &body);
        let output = stream
            .feed(wire.as_bytes())
            .expect("each complete repeated tool call must emit");
        assert!(output.thinking.is_empty());
        assert!(output.content.is_empty());
        assert!(output.wire_bytes.is_empty());
        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(output.tool_calls[0].name, "counter");
        assert_eq!(output.tool_calls[0].arguments["index"], index.to_string());
        emitted_calls += output.tool_calls.len();
        largest_pending = largest_pending.max(stream.pending_bytes());
        assert_eq!(stream.pending_bytes(), 0, "complete calls must be drained");
        assert!(stream.pending_bytes() <= pending_cap);
    }
    assert_eq!(emitted_calls, 128);
    assert_eq!(largest_pending, 0);

    let content = stream
        .feed(b"<|channel|>final<|message|>finished<|return|>")
        .expect("a complete content field must emit from feed");
    assert_eq!(content.content, "finished");
    assert!(content.thinking.is_empty());
    assert!(content.tool_calls.is_empty());
    assert!(content.wire_bytes.is_empty());
    assert_eq!(stream.pending_bytes(), 0);
    assert!(stream
        .finish()
        .expect("drained stream must finish")
        .is_empty());
}

#[test]
fn retained_ignorable_whitespace_is_bounded_by_pending_limit() {
    let pending_cap = 8;
    let parser = parser(config_with_limits(pending_cap, 64, 128));

    let mut exact = parser.stream(PREFIX);
    assert!(exact.feed(&vec![b' '; pending_cap]).is_ok());
    assert_eq!(exact.pending_bytes(), pending_cap);

    let mut overflow = parser.stream(PREFIX);
    let oversized = vec![b' '; 1_000_000];
    let error = overflow
        .feed(&oversized)
        .expect_err("retained whitespace beyond the pending cap must fail");
    assert!(matches!(
        error,
        ResponseTemplateError::PendingOverflow {
            ref field,
            limit,
            ..
        } if field == "max_pending_bytes" && limit == pending_cap
    ));
    assert_eq!(overflow.pending_bytes(), 0, "failed feed must not commit");
    assert_eq!(overflow.feed(b"ignored").unwrap_err(), error);
    assert_eq!(overflow.finish().unwrap_err(), error);
}

#[expect(
    clippy::unwrap_used,
    reason = "the synthetic template declares tool_calls"
)]
fn bounded_openers_parser() -> ResponseTemplateParser {
    let mut bounded = template();
    "<tool:(?P<name>foo)>"
        .clone_into(&mut bounded.fields.get_mut("tool_calls").unwrap().open_pattern);
    let parser = ResponseTemplateParser::new(MODEL, bounded, config_with_limits(32, 64, 256))
        .expect("bounded tool-name capture remains a supported template");
    assert!(parser
        .delimiter_metadata()
        .iter()
        .filter(|delimiter| delimiter.role == "open")
        .all(
            |delimiter| matches!(delimiter.bound, DelimiterBound::Bounded(maximum) if maximum < 32)
        ));
    parser
}

#[test]
fn bounded_openers_pending_limit_exact_and_overflow() {
    let parser = bounded_openers_parser();
    let mut exact = parser.stream(PREFIX);
    assert!(exact.feed(&[b' '; 32]).unwrap().is_empty());
    assert_eq!(exact.pending_bytes(), 32);
    assert!(exact.finish().unwrap().is_empty());

    // Unrecognized text, like whitespace, is retained before a complete opener.
    for byte in *b" x" {
        let mut overflow = parser.stream(PREFIX);
        let error = overflow
            .feed(&[byte; 33])
            .expect_err("bounded opener width must not hide retained bytes");
        assert!(matches!(
            error,
            ResponseTemplateError::PendingOverflow { ref field, limit: 32, .. }
                if field == "max_pending_bytes"
        ));
        assert_eq!(overflow.pending_bytes(), 0, "failed feed must not commit");
        assert_eq!(overflow.feed(b"ignored").unwrap_err(), error);
        assert_eq!(overflow.finish().unwrap_err(), error);
    }
}

#[test]
fn bounded_openers_pending_limit_accumulates_whitespace_chunks() {
    let parser = bounded_openers_parser();
    let mut stream = parser.stream(PREFIX);
    for count in 1..=8 {
        assert!(stream.feed(b"    ").unwrap().is_empty());
        assert_eq!(stream.pending_bytes(), count * 4);
    }
    let error = stream
        .feed(b" ")
        .expect_err("pending cap applies across feeds, not to individual chunks");
    assert!(matches!(
        error,
        ResponseTemplateError::PendingOverflow { ref field, limit: 32, .. }
            if field == "max_pending_bytes"
    ));
    assert_eq!(stream.pending_bytes(), 32, "keep the last committed state");
    assert_eq!(
        stream
            .feed(b"<|channel|>final<|message|>ok<|return|>")
            .unwrap_err(),
        error
    );
    assert_eq!(stream.finish().unwrap_err(), error);
}

#[test]
fn bounded_openers_pending_limit_keeps_separate_body_and_close_holdback() {
    let parser = bounded_openers_parser();
    let mut stream = parser.stream(PREFIX);
    let body = "a".repeat(128);
    let partial = format!("<|channel|>final<|message|>{body}<|ret");
    assert!(stream.feed(partial.as_bytes()).unwrap().is_empty());
    // The body has its own 256-byte limit; only the undecided close suffix
    // counts against the pending cap once an opener has been recognized.
    assert!(stream.pending_bytes() > 32);
    let output = stream.feed(b"urn|>").unwrap();
    assert_eq!(output.content, body);
    assert!(output.wire_bytes.is_empty());
    assert_eq!(stream.pending_bytes(), 0);
    assert!(stream.finish().unwrap().is_empty());
}

#[test]
fn bounded_openers_pending_limit_counts_undrained_leading_prefix() {
    let parser = bounded_openers_parser();
    for (opener, body, close) in [
        ("<|channel|>final<|message|>", "x", "<|return|>"),
        ("<tool:foo>", "", "<|end|>"),
    ] {
        let mut exact = parser.stream(PREFIX);
        let partial = format!("{}{opener}{body}", " ".repeat(32));
        assert!(exact.feed(partial.as_bytes()).unwrap().is_empty());
        assert_eq!(exact.pending_bytes(), partial.len());
        assert!(!exact.feed(close.as_bytes()).unwrap().is_empty());
        assert_eq!(exact.pending_bytes(), 0);
        assert!(exact.finish().unwrap().is_empty());

        let mut overflow = parser.stream(PREFIX);
        let partial = format!("{}{opener}{body}", " ".repeat(33));
        let error = overflow
            .feed(partial.as_bytes())
            .expect_err("an unfinished field must not hide its retained prefix");
        assert!(matches!(
            error,
            ResponseTemplateError::PendingOverflow { ref field, limit: 32, .. }
                if field == "max_pending_bytes"
        ));
        assert_eq!(overflow.pending_bytes(), 0, "failed feed must not commit");
        assert_eq!(overflow.feed(close.as_bytes()).unwrap_err(), error);
        assert_eq!(overflow.finish().unwrap_err(), error);
    }
}

#[test]
fn bounded_openers_pending_limit_combines_prefix_and_close_holdback() {
    let parser = bounded_openers_parser();
    let body = "x".repeat(128);
    for prefix_len in [27, 28] {
        let mut stream = parser.stream(PREFIX);
        let partial = format!(
            "{}<|channel|>final<|message|>{body}<|ret",
            " ".repeat(prefix_len)
        );
        let result = stream.feed(partial.as_bytes());
        if prefix_len == 27 {
            // 27 retained prefix bytes + 5 undecided close bytes = exact cap.
            assert!(result.unwrap().is_empty());
            assert_eq!(stream.feed(b"urn|>").unwrap().content, body);
            assert_eq!(stream.pending_bytes(), 0);
        } else {
            assert!(matches!(
                result,
                Err(ResponseTemplateError::PendingOverflow { limit: 32, .. })
            ));
            assert_eq!(stream.pending_bytes(), 0);
        }
    }

    // A complete field and its prefix drain in this feed. Neither its body nor
    // its already-consumed whitespace belongs to the remaining pending suffix.
    let mut complete = parser.stream(PREFIX);
    let wire = format!(
        "{}<|channel|>final<|message|>{body}<|return|>",
        " ".repeat(33)
    );
    assert_eq!(complete.feed(wire.as_bytes()).unwrap().content, body);
    assert_eq!(complete.pending_bytes(), 0);
    assert!(complete.finish().unwrap().is_empty());
}

#[test]
fn unsupported_text_field_is_rejected_at_load() {
    let mut unsupported = template();
    let tool_calls = unsupported
        .fields
        .get_mut("tool_calls")
        .expect("tool fixture");
    tool_calls.content = "text".to_owned();
    tool_calls.content_args = None;
    tool_calls.repeats = false;
    tool_calls.transform = None;
    let error = ResponseTemplateParser::new(MODEL, unsupported, ParserConfig::default())
        .expect_err("unsupported text field capability must fail template loading");
    assert!(matches!(
        error,
        ResponseTemplateError::InvalidTemplate {
            ref field,
            limit: 0,
            ..
        } if field == "tool_calls"
    ));
}

fn generic_tag_template() -> ResponseTemplate {
    let mut generic = template();
    r#"<arg key=\"(?P<key>[^\"]+)\">(?P<value>.*?)</arg>"#.clone_into(
        &mut generic
            .fields
            .get_mut("tool_calls")
            .expect("tool fixture")
            .content_args
            .as_mut()
            .expect("tag fixture")
            .tag_pattern,
    );
    generic
}

fn generic_unfinished_value(value: &str) -> String {
    format!(" to=x<|channel|>analysis <|constrain|>xml<|message|><arg key=\"k\">{value}")
}

#[test]
fn generic_tag_pattern_unfinished_value_cap_plus_one_overflows_during_feed() {
    let cap = "éé".len();
    assert_eq!(cap, 4);
    assert_eq!("éé".chars().count(), 2);
    let config = config_with_limits(256, cap, 256);
    let parser = ResponseTemplateParser::new(MODEL, generic_tag_template(), config)
        .expect("generic tag pattern must load");

    let mut exact = parser.stream(PREFIX);
    assert!(exact
        .feed(generic_unfinished_value("éé").as_bytes())
        .is_ok());

    let mut overflow = parser.stream(PREFIX);
    let error = overflow
        .feed(generic_unfinished_value("ééa").as_bytes())
        .expect_err("generic unfinished value cap plus one must fail during feed");
    assert!(matches!(
        error,
        ResponseTemplateError::StructuredFieldOverflow {
            ref model_name,
            ref field,
            limit,
        } if model_name == MODEL && field == "max_structured_field_bytes" && limit == cap
    ));
    assert_eq!(overflow.feed(b"ignored").unwrap_err(), error);
    assert_eq!(overflow.finish().unwrap_err(), error);
}

#[test]
fn nonempty_thinking_and_content_defaults_apply_when_omitted() {
    let mut with_defaults = template();
    with_defaults.defaults.insert(
        "thinking".to_owned(),
        Value::String("default thinking".to_owned()),
    );
    with_defaults.defaults.insert(
        "content".to_owned(),
        Value::String("default content".to_owned()),
    );
    let parser = ResponseTemplateParser::new(MODEL, with_defaults, ParserConfig::default())
        .expect("fixture with defaults must load");
    let output = parser
        .parse_complete(
            PREFIX,
            &tool_wire("only.tool", "<arg name=\"value\">present</arg>"),
        )
        .expect("omitted text fields must use their configured defaults");
    assert_eq!(output.thinking, "default thinking");
    assert_eq!(output.content, "default content");
    assert_eq!(output.tool_calls.len(), 1);
    assert_eq!(output.tool_calls[0].name, "only.tool");
}

#[test]
fn start_anchor_and_pending_holdback_are_enforced() {
    let parser = parser(config_with_limits(64, 4_096, 4_096));
    let mut mismatched = parser.stream("prompt without the assistant anchor");
    let mismatch = mismatched
        .feed(b"<|channel|>final<|message|>x<|return|>")
        .expect_err("a mismatched rendered prefix must fail closed");
    assert!(matches!(
        mismatch,
        ResponseTemplateError::RuntimeFailure { ref field, .. }
            if field == "start_anchor_pattern"
    ));

    let long_closed = format!("<|channel|>analysis<|message|>{}<|end|>", "x".repeat(128));
    assert!(long_closed.len() > 64);
    let parsed = parser
        .parse_complete(PREFIX, &long_closed)
        .expect("completed fields larger than pending cap do not remain held back");
    assert_eq!(parsed.thinking, "x".repeat(128));
}

#[test]
fn finish_without_feed_validates_start_anchor() {
    let parser = parser(ParserConfig::default());
    let mut stream = parser.stream("unmatched prompt");
    let error = stream
        .finish()
        .expect_err("finish must validate the prompt");
    assert!(matches!(
        &error,
        ResponseTemplateError::RuntimeFailure { field, .. } if field == "start_anchor_pattern"
    ));
    assert_eq!(stream.feed(&[]).unwrap_err(), error);
    assert_eq!(stream.finish().unwrap_err(), error);
    assert!(parser.stream(PREFIX).finish().unwrap().is_empty());
}

#[test]
fn nonempty_tool_call_defaults_are_rejected_at_load() {
    let mut template = template();
    template
        .defaults
        .insert("tool_calls".to_owned(), json!([{"name": "lookup"}]));
    let error = ResponseTemplateParser::new(MODEL, template, ParserConfig::default())
        .expect_err("unsupported defaults must not be silently discarded");
    assert!(matches!(
        error,
        ResponseTemplateError::InvalidTemplate { field, .. } if field == "tool_calls"
    ));
}

#[test]
fn incomplete_xml_value_observes_structured_byte_limit() {
    let parser = parser(config_with_limits(1_024, 4, 1_024));
    let mut stream = parser.stream(PREFIX);
    let error = stream
        .feed(tool_wire("x", "<arg name=\"k\">12345").as_bytes())
        .expect_err("an unfinished over-limit parameter value must overflow before finish");
    assert!(matches!(
        error,
        ResponseTemplateError::StructuredFieldOverflow {
            ref field,
            limit: 4,
            ..
        } if field == "max_structured_field_bytes"
    ));
}

#[test]
fn behavior_matrix_exactly_cap_cap_plus_one_poisoned_after_overflow_every_split_offset_multibyte_split(
) {
    let complete_wire = complete_wire();
    let complete_direct = parser(ParserConfig::default())
        .parse_complete(PREFIX, &complete_wire)
        .expect("complete fields must parse");
    let complete_streamed =
        one_chunk_equals_every_byte_split(ParserConfig::default(), &complete_wire);
    assert_eq!(complete_streamed.error, None);
    assert_eq!(complete_streamed.output, complete_direct);
    assert_eq!(complete_direct.thinking, "thinking\u{200b} café");
    assert_eq!(complete_direct.content, "done\u{301}");
    assert_eq!(complete_direct.tool_calls.len(), 2);
    assert_eq!(complete_direct.tool_calls[0].name, "weather.lookup");
    assert_eq!(
        complete_direct.tool_calls[0].arguments["city"],
        " Montréal "
    );
    assert_eq!(complete_direct.tool_calls[0].arguments["units"], "metric");
    assert_eq!(complete_direct.tool_calls[1].name, "calendar.lookup");
    assert_eq!(complete_direct.tool_calls[1].arguments["day"], "tomorrow");
    assert_eq!(
        complete_direct.tool_calls[0].transformed,
        json!({
            "type": "function",
            "function": {
                "name": "weather.lookup",
                "arguments": {
                    "city": " Montréal ",
                    "units": "metric",
                },
            },
        })
    );
    assert_eq!(
        complete_direct.tool_calls[1].transformed,
        json!({
            "type": "function",
            "function": {
                "name": "calendar.lookup",
                "arguments": {"day": "tomorrow"},
            },
        })
    );
    assert_eq!(output_delimiter_leakage(&complete_direct), 0);

    // The two-byte scalar is split between its UTF-8 bytes by both the every-
    // offset loop and the byte-at-a-time replay.
    let multibyte_wire = concat!(
        "<|channel|>analysis<|message|>é<|end|>",
        "<|channel|>final<|message|>界<|return|>"
    );
    let multibyte = one_chunk_equals_every_byte_split(ParserConfig::default(), multibyte_wire);
    assert_eq!(multibyte.error, None);
    assert_eq!(multibyte.output.thinking, "é");
    assert_eq!(multibyte.output.content, "界");

    let zero_width_wire = concat!(
        "<|channel|>analysis<|message|>a\u{200b}b<|end|>",
        "<|channel|>final<|message|>x\u{200d}y\u{301}<|return|>"
    );
    let zero_width = one_chunk_equals_every_byte_split(ParserConfig::default(), zero_width_wire);
    assert_eq!(zero_width.error, None);
    assert_eq!(zero_width.output.thinking, "a\u{200b}b");
    assert_eq!(zero_width.output.content, "x\u{200d}y\u{301}");

    let unterminated_wire = "<|channel|>analysis<|message|>unterminated";
    let unterminated =
        one_chunk_equals_every_byte_split(ParserConfig::default(), unterminated_wire);
    assert!(unterminated.output.is_empty());
    assert_eq!(
        unterminated
            .error
            .as_ref()
            .map(|error| error.variant.as_str()),
        Some("RuntimeFailure")
    );

    let reserved_wire = tool_wire("x", "<arg name=\"q\">before </arg> after</arg>");
    let reserved = one_chunk_equals_every_byte_split(ParserConfig::default(), &reserved_wire);
    assert!(reserved.output.is_empty());
    assert_eq!(
        reserved.error.as_ref().map(|error| error.variant.as_str()),
        Some("RuntimeFailure")
    );

    let invalid_one = invalid_template("(").expect_err("invalid regex must be rejected");
    let invalid_two = invalid_template("(").expect_err("invalid regex must be deterministic");
    assert!(matches!(
        invalid_one,
        ResponseTemplateError::InvalidTemplate { .. }
    ));
    assert_eq!(invalid_one, invalid_two);

    let empty_one = invalid_template("a*").expect_err("empty-matching regex must be rejected");
    let empty_two = invalid_template("a*").expect_err("empty rejection must be deterministic");
    assert!(matches!(
        empty_one,
        ResponseTemplateError::InvalidTemplate { .. }
    ));
    assert_eq!(empty_one, empty_two);

    // Independent byte limits. The complete unbounded opener is exactly at
    // the pending cap; adding one tool-name byte makes that held delimiter
    // state cap+1 while the surrounding response remains otherwise valid.
    let pending_name = "p".repeat(8);
    let pending_exact = tool_wire(&pending_name, "<arg name=\"k\">v</arg>");
    let pending_plus_one = tool_wire(&format!("{pending_name}p"), "<arg name=\"k\">v</arg>");
    let pending_cap = pending_exact
        .find("<arg")
        .expect("tool body begins after opener")
        - 1;
    assert_eq!(
        pending_plus_one
            .find("<arg")
            .expect("tool body begins after opener")
            - 1,
        pending_cap + 1
    );
    let pending_config = config_with_limits(pending_cap, 4_096, 4_096);

    let name_cap = 8usize;
    let name_exact_wire = tool_wire(&"n".repeat(name_cap), "<arg name=\"k\">v</arg>");
    let name_plus_one_wire = tool_wire(&"n".repeat(name_cap + 1), "<arg name=\"k\">v</arg>");
    let name_config = config_with_limits(4_096, name_cap, 4_096);

    // Limits count UTF-8 bytes: two two-byte scalars reach the cap, and one
    // following ASCII byte overflows it.
    let value_cap = "éé".len();
    assert_eq!(value_cap, 4);
    assert_eq!("éé".chars().count(), 2);
    let value_exact_wire = tool_wire("x", "<arg name=\"k\">éé</arg>");
    let value_plus_one_wire = tool_wire("x", "<arg name=\"k\">ééa</arg>");
    let value_config = config_with_limits(4_096, value_cap, 4_096);

    let body_exact = "<arg name=\"k\">v</arg>";
    let body_plus_one = "<arg name=\"k\">vv</arg>";
    let body_cap = body_exact.len();
    assert_eq!(body_plus_one.len(), body_cap + 1);
    let body_exact_wire = tool_wire("x", body_exact);
    let body_plus_one_wire = tool_wire("x", body_plus_one);
    let body_config = config_with_limits(4_096, 4_096, body_cap);

    for (config, input) in [
        (pending_config, pending_exact.as_bytes()),
        (name_config, name_exact_wire.as_bytes()),
        (value_config, value_exact_wire.as_bytes()),
        (body_config, body_exact_wire.as_bytes()),
    ] {
        exactly_cap(config, input);
        all_chunkings(config, input);
    }

    for (config, input, variant, field, limit) in [
        (
            pending_config,
            pending_plus_one.as_bytes(),
            "PendingOverflow",
            "max_pending_bytes",
            pending_cap,
        ),
        (
            name_config,
            name_plus_one_wire.as_bytes(),
            "StructuredFieldOverflow",
            "max_structured_field_bytes",
            name_cap,
        ),
        (
            value_config,
            value_plus_one_wire.as_bytes(),
            "StructuredFieldOverflow",
            "max_structured_field_bytes",
            value_cap,
        ),
        (
            body_config,
            body_plus_one_wire.as_bytes(),
            "StructuredFieldOverflow",
            "max_body_bytes",
            body_cap,
        ),
    ] {
        let error = cap_plus_one(config, input, variant, field);
        assert_eq!(error.limit, limit);
        let streamed = all_chunkings(config, input);
        assert_eq!(streamed.error, Some(error));
    }
}
