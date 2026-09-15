//! Public-API behavior matrix for complete and incremental parsing.
//!
//! The cases in this file intentionally exercise the parser only through its
//! public API.  Each streaming scenario is replayed as one chunk and at every
//! byte boundary so the evidence records observed, chunk-independent results.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration-test helpers fail immediately when a fixture or assertion is invalid"
)]

use std::{collections::BTreeMap, env, fs::OpenOptions, io::Write, path::Path};

use response_template_parser::{
    ClosePattern, ContentArgs, FieldTemplate, ParserConfig, ResponseTemplate,
    ResponseTemplateError, ResponseTemplateParser, Transform, ValueParser, ValueParserArgs,
};
use serde::Serialize;
use serde_json::{json, Value};

const FOUR_MIB: usize = 4_194_304;
const MODEL: &str = "generic-response-template-test";
const PREFIX: &str = "<|start|>assistant";

fn observed_default_limits() -> BTreeMap<&'static str, usize> {
    let config = ParserConfig::default();
    assert_eq!(config.max_pending_bytes, FOUR_MIB);
    assert_eq!(config.max_structured_field_bytes, FOUR_MIB);
    assert_eq!(config.max_body_bytes, FOUR_MIB);

    BTreeMap::from([
        ("max_pending_bytes", config.max_pending_bytes),
        (
            "max_structured_field_bytes",
            config.max_structured_field_bytes,
        ),
        ("max_body_bytes", config.max_body_bytes),
    ])
}

fn write_new_json(path: impl AsRef<Path>, value: &impl Serialize) {
    let path = path.as_ref();
    let bytes = serde_json::to_vec_pretty(value).expect("summary must serialize");
    assert!(!bytes.is_empty());
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap_or_else(|error| panic!("refusing to replace summary {}: {error}", path.display()));
    output
        .write_all(&bytes)
        .unwrap_or_else(|error| panic!("could not write summary {}: {error}", path.display()));
    output
        .flush()
        .unwrap_or_else(|error| panic!("could not flush summary {}: {error}", path.display()));
}

#[test]
fn default_limits_are_4_mib() {
    let observed = observed_default_limits();
    let invoked = observed.values().all(|value| *value == FOUR_MIB);
    assert!(invoked);

    if let Some(path) = env::var_os("RESPONSE_TEMPLATE_DEFAULT_LIMITS_SUMMARY") {
        // This named run owns only the default-limits summary.  In particular,
        // it never writes RESPONSE_TEMPLATE_TEST_SUMMARY.
        write_new_json(
            path,
            &json!({
                "default_limits_test_name": "default_limits_are_4_mib",
                "default_limits_test_invoked": invoked,
                "observed_default_limits": observed,
            }),
        );
    }
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
                tag_pattern: concat!(
                    r#"<rfl:parameter name=\"(?P<key>[^\"]+)\">"#,
                    r"(?P<value>.*?)</rfl:parameter>"
                )
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
        "<rfl:parameter name=\"city\"> Montréal </rfl:parameter>",
        "<rfl:parameter name=\"units\">metric</rfl:parameter>",
        "<|call|>",
        "<|start|>assistant to=calendar.lookup<|channel|>analysis <|constrain|>xml<|message|>",
        "<rfl:parameter name=\"day\">tomorrow</rfl:parameter>",
        "<|end|>",
        "<|channel|>final<|message|>",
        "done\u{301}",
        "<|return|>"
    )
    .to_owned()
}

#[derive(Debug, Clone, PartialEq, Serialize)]
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

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
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

fn all_chunkings(config: ParserConfig, input: &[u8]) -> (Replay, usize, bool) {
    let baseline = replay(config, input, &[input.len()]);
    let mut checked = 0;
    let mut independent = true;

    // Two chunks at every byte offset, including offsets inside UTF-8 scalars
    // and inside both regex and literal delimiters.
    for offset in 0..=input.len() {
        let split = replay(config, input, &[offset, input.len() - offset]);
        independent &= split == baseline;
        assert_eq!(split, baseline, "outcome changed at split offset {offset}");
        checked += 1;
    }

    // Also exercise the maximally fragmented byte-at-a-time delivery required
    // by the work package (feed accepts bytes, so invalid partial UTF-8 is held).
    let byte_lengths = vec![1; input.len()];
    let bytewise = replay(config, input, &byte_lengths);
    independent &= bytewise == baseline;
    assert_eq!(bytewise, baseline, "byte-at-a-time outcome changed");
    checked += input.len();

    (baseline, checked, independent)
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
        "<rfl:parameter",
        "</rfl:parameter>",
    ]
    .iter()
    .map(|delimiter| encoded.matches(delimiter).count())
    .sum()
}

fn one_chunk_equals_every_byte_split(config: ParserConfig, input: &str) -> (Replay, usize, bool) {
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

#[derive(Debug)]
struct OverflowProbe {
    error: NormalizedError,
    whole_feed_output_discarded: bool,
    wire_bytes_leaked: usize,
    poisoned_after_overflow: bool,
    later_feed_returns_same_error: bool,
    finish_returns_same_error: bool,
}

fn exactly_cap(config: ParserConfig, input: &[u8]) -> bool {
    let parser = parser(config);
    let mut stream = parser.stream(PREFIX);
    let accepted = stream.feed(input).is_ok() && stream.finish().is_ok();
    assert!(accepted, "an input exactly at its byte cap was rejected");
    accepted
}

fn cap_plus_one(config: ParserConfig, input: &[u8], variant: &str, field: &str) -> OverflowProbe {
    let parser = parser(config);
    let mut stream = parser.stream(PREFIX);
    let feed_result = stream.feed(input);
    let (error, feed_output) = match feed_result {
        Err(error) => (error, None),
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
    let later_same = NormalizedError::from(&later_error) == normalized;
    let finish_same = NormalizedError::from(&finish_error) == normalized;
    assert!(later_same);
    assert!(finish_same);

    let whole_feed_output_discarded = feed_output.is_none();
    let wire_bytes_leaked = feed_output
        .as_ref()
        .map(|output: &response_template_parser::ParseOutput| output.wire_bytes.len())
        .unwrap_or(0);
    let poisoned_after_overflow = later_same && finish_same;
    assert!(whole_feed_output_discarded);
    assert_eq!(wire_bytes_leaked, 0);
    assert!(poisoned_after_overflow);
    OverflowProbe {
        error: normalized,
        whole_feed_output_discarded,
        wire_bytes_leaked,
        poisoned_after_overflow,
        later_feed_returns_same_error: later_same,
        finish_returns_same_error: finish_same,
    }
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
        let body = format!("<rfl:parameter name=\"index\">{index}</rfl:parameter>");
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
            &tool_wire(
                "only.tool",
                "<rfl:parameter name=\"value\">present</rfl:parameter>",
            ),
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
        .feed(tool_wire("x", "<rfl:parameter name=\"k\">12345").as_bytes())
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
    let defaults = observed_default_limits();
    let default_limits_test_invoked = defaults.values().all(|value| *value == FOUR_MIB);
    assert!(default_limits_test_invoked);

    let mut per_case = BTreeMap::<String, Value>::new();
    let mut split_offsets_checked = 0usize;
    let mut outcomes_chunk_independent = true;

    let complete_wire = complete_wire();
    let complete_direct = parser(ParserConfig::default())
        .parse_complete(PREFIX, &complete_wire)
        .expect("complete fields must parse");
    let (complete_streamed, checked, independent) =
        one_chunk_equals_every_byte_split(ParserConfig::default(), &complete_wire);
    split_offsets_checked += checked;
    outcomes_chunk_independent &= independent;
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
    let delimiter_leakage = output_delimiter_leakage(&complete_direct);
    assert_eq!(delimiter_leakage, 0);
    per_case.insert(
        "complete_fields".to_owned(),
        json!({
            "one_chunk_equals_every_byte_split": independent,
            "output": complete_direct,
        }),
    );
    let repeated_tool_calls_observed = complete_streamed.output.tool_calls.len() == 2;
    assert!(repeated_tool_calls_observed);
    per_case.insert(
        "repeated_tool_calls".to_owned(),
        json!({
            "one_chunk_equals_every_byte_split": independent,
            "output": complete_streamed.output.tool_calls,
        }),
    );
    per_case.insert(
        "every_split_offset".to_owned(),
        json!({
            "one_chunk_equals_every_byte_split": independent,
            "split_offsets_checked": checked,
            "output": complete_direct,
        }),
    );

    // The two-byte scalar is split between its UTF-8 bytes by both the every-
    // offset loop and the byte-at-a-time replay.
    let multibyte_wire = concat!(
        "<|channel|>analysis<|message|>é<|end|>",
        "<|channel|>final<|message|>界<|return|>"
    );
    let (multibyte, checked, independent) =
        one_chunk_equals_every_byte_split(ParserConfig::default(), multibyte_wire);
    split_offsets_checked += checked;
    outcomes_chunk_independent &= independent;
    assert_eq!(multibyte.error, None);
    assert_eq!(multibyte.output.thinking, "é");
    assert_eq!(multibyte.output.content, "界");
    let multibyte_split_cases = usize::from(
        independent
            && "é".len() == 2
            && "é".chars().count() == 1
            && "界".len() == 3
            && "界".chars().count() == 1,
    );
    assert!(multibyte_split_cases > 0);
    per_case.insert(
        "multibyte_split".to_owned(),
        json!({
            "one_chunk_equals_every_byte_split": independent,
            "output": multibyte.output,
        }),
    );

    let zero_width_wire = concat!(
        "<|channel|>analysis<|message|>a\u{200b}b<|end|>",
        "<|channel|>final<|message|>x\u{200d}y\u{301}<|return|>"
    );
    let (zero_width, checked, independent) =
        one_chunk_equals_every_byte_split(ParserConfig::default(), zero_width_wire);
    split_offsets_checked += checked;
    outcomes_chunk_independent &= independent;
    assert_eq!(zero_width.error, None);
    assert_eq!(zero_width.output.thinking, "a\u{200b}b");
    assert_eq!(zero_width.output.content, "x\u{200d}y\u{301}");
    let zero_width_cases = usize::from(
        zero_width.output.thinking.contains('\u{200b}')
            && zero_width.output.content.contains('\u{200d}')
            && zero_width.output.content.contains('\u{301}'),
    );
    assert!(zero_width_cases > 0);
    per_case.insert(
        "zero_width_content".to_owned(),
        json!({
            "one_chunk_equals_every_byte_split": independent,
            "output": zero_width.output,
        }),
    );

    let unterminated_wire = "<|channel|>analysis<|message|>unterminated";
    let (unterminated, checked, independent) =
        one_chunk_equals_every_byte_split(ParserConfig::default(), unterminated_wire);
    split_offsets_checked += checked;
    outcomes_chunk_independent &= independent;
    let unterminated_finish_error = unterminated.output.is_empty()
        && matches!(
            unterminated
                .error
                .as_ref()
                .map(|error| error.variant.as_str()),
            Some("RuntimeFailure")
        );
    assert!(unterminated_finish_error);
    per_case.insert(
        "unterminated_field".to_owned(),
        json!({
            "one_chunk_equals_every_byte_split": independent,
            "error": unterminated.error,
            "output": unterminated.output,
        }),
    );

    let reserved_wire = tool_wire(
        "x",
        "<rfl:parameter name=\"q\">before </rfl:parameter> after</rfl:parameter>",
    );
    let (reserved, checked, independent) =
        one_chunk_equals_every_byte_split(ParserConfig::default(), &reserved_wire);
    split_offsets_checked += checked;
    outcomes_chunk_independent &= independent;
    let reserved_delimiter_rejected_no_partial_output = reserved.output.is_empty()
        && matches!(
            reserved.error.as_ref().map(|error| error.variant.as_str()),
            Some("RuntimeFailure")
        );
    assert!(reserved_delimiter_rejected_no_partial_output);
    per_case.insert(
        "reserved_parameter_close".to_owned(),
        json!({
            "one_chunk_equals_every_byte_split": independent,
            "error": reserved.error,
            "output": reserved.output,
        }),
    );

    let invalid_one = invalid_template("(").expect_err("invalid regex must be rejected");
    let invalid_two = invalid_template("(").expect_err("invalid regex must be deterministic");
    let invalid_regex_rejected_at_load =
        matches!(invalid_one, ResponseTemplateError::InvalidTemplate { .. })
            && invalid_one == invalid_two;
    assert!(invalid_regex_rejected_at_load);
    per_case.insert(
        "invalid_regex".to_owned(),
        json!({
            "one_chunk_equals_every_byte_split": invalid_regex_rejected_at_load,
            "error": NormalizedError::from(&invalid_one),
        }),
    );

    let empty_one = invalid_template("a*").expect_err("empty-matching regex must be rejected");
    let empty_two = invalid_template("a*").expect_err("empty rejection must be deterministic");
    let empty_pattern_rejected_at_load =
        matches!(empty_one, ResponseTemplateError::InvalidTemplate { .. })
            && empty_one == empty_two;
    assert!(empty_pattern_rejected_at_load);
    per_case.insert(
        "empty_pattern".to_owned(),
        json!({
            "one_chunk_equals_every_byte_split": empty_pattern_rejected_at_load,
            "error": NormalizedError::from(&empty_one),
        }),
    );

    // Independent byte limits. The complete unbounded opener is exactly at
    // the pending cap; adding one tool-name byte makes that held delimiter
    // state cap+1 while the surrounding response remains otherwise valid.
    let pending_name = "p".repeat(8);
    let pending_exact = tool_wire(&pending_name, "<rfl:parameter name=\"k\">v</rfl:parameter>");
    let pending_plus_one = tool_wire(
        &format!("{pending_name}p"),
        "<rfl:parameter name=\"k\">v</rfl:parameter>",
    );
    let pending_cap = pending_exact
        .find("<rfl:parameter")
        .expect("tool body begins after opener")
        - 1;
    assert_eq!(
        pending_plus_one
            .find("<rfl:parameter")
            .expect("tool body begins after opener")
            - 1,
        pending_cap + 1
    );
    let pending_config = config_with_limits(pending_cap, 4_096, 4_096);

    let name_cap = 8usize;
    let name_exact_wire = tool_wire(
        &"n".repeat(name_cap),
        "<rfl:parameter name=\"k\">v</rfl:parameter>",
    );
    let name_plus_one_wire = tool_wire(
        &"n".repeat(name_cap + 1),
        "<rfl:parameter name=\"k\">v</rfl:parameter>",
    );
    let name_config = config_with_limits(4_096, name_cap, 4_096);

    let value_cap = "éé".len();
    let value_exact_wire = tool_wire("x", "<rfl:parameter name=\"k\">éé</rfl:parameter>");
    let value_plus_one_wire = tool_wire("x", "<rfl:parameter name=\"k\">ééa</rfl:parameter>");
    let value_config = config_with_limits(4_096, value_cap, 4_096);

    let body_exact = "<rfl:parameter name=\"k\">v</rfl:parameter>";
    let body_plus_one = "<rfl:parameter name=\"k\">vv</rfl:parameter>";
    let body_cap = body_exact.len();
    assert_eq!(body_plus_one.len(), body_cap + 1);
    let body_exact_wire = tool_wire("x", body_exact);
    let body_plus_one_wire = tool_wire("x", body_plus_one);
    let body_config = config_with_limits(4_096, 4_096, body_cap);

    let exact_inputs = [
        (pending_config, pending_exact.as_bytes()),
        (name_config, name_exact_wire.as_bytes()),
        (value_config, value_exact_wire.as_bytes()),
        (body_config, body_exact_wire.as_bytes()),
    ];
    let mut exactly_cap_all_accepted = true;
    for (config, input) in exact_inputs {
        let accepted = exactly_cap(config, input);
        exactly_cap_all_accepted &= accepted;
        let (_, checked, independent) = all_chunkings(config, input);
        split_offsets_checked += checked;
        outcomes_chunk_independent &= independent;
    }
    let exactly_cap_cases = exact_inputs.len();
    assert!(exactly_cap_cases >= 3);
    assert!(exactly_cap_all_accepted);

    let overflow_inputs = [
        (
            pending_config,
            pending_plus_one.as_bytes(),
            "PendingOverflow",
            "max_pending_bytes",
        ),
        (
            name_config,
            name_plus_one_wire.as_bytes(),
            "StructuredFieldOverflow",
            "max_structured_field_bytes",
        ),
        (
            value_config,
            value_plus_one_wire.as_bytes(),
            "StructuredFieldOverflow",
            "max_structured_field_bytes",
        ),
        (
            body_config,
            body_plus_one_wire.as_bytes(),
            "StructuredFieldOverflow",
            "max_body_bytes",
        ),
    ];
    let mut probes = Vec::new();
    for (config, input, variant, field) in overflow_inputs {
        probes.push(cap_plus_one(config, input, variant, field));
        let (_, checked, independent) = all_chunkings(config, input);
        split_offsets_checked += checked;
        outcomes_chunk_independent &= independent;
    }
    let cap_plus_one_cases = probes.len();
    let cap_plus_one_all_typed_overflow = probes.iter().all(|probe| {
        matches!(
            probe.error.variant.as_str(),
            "PendingOverflow" | "StructuredFieldOverflow"
        )
    });
    assert!(cap_plus_one_cases >= 3);
    assert!(cap_plus_one_all_typed_overflow);

    let overflow_variants_observed = probes
        .iter()
        .map(|probe| probe.error.variant.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(
        overflow_variants_observed,
        vec!["PendingOverflow", "StructuredFieldOverflow"]
    );
    let variants_name_field_and_limit = probes.iter().all(|probe| {
        let expected = match probe.error.field.as_str() {
            "max_pending_bytes" => Some(pending_cap),
            "max_structured_field_bytes" => Some(if probe.error.limit == name_cap {
                name_cap
            } else {
                value_cap
            }),
            "max_body_bytes" => Some(body_cap),
            _ => None,
        };
        !probe.error.field.is_empty() && expected == Some(probe.error.limit)
    }) && overflow_variants_observed.len() == 2;
    assert!(variants_name_field_and_limit);

    let whole_feed_output_discarded = probes.iter().all(|probe| probe.whole_feed_output_discarded);
    let wire_bytes_leaked = probes
        .iter()
        .map(|probe| probe.wire_bytes_leaked)
        .sum::<usize>();
    let poisoned_after_overflow = probes.iter().all(|probe| probe.poisoned_after_overflow);
    let later_feed_returns_same_error = probes
        .iter()
        .all(|probe| probe.later_feed_returns_same_error);
    let finish_returns_same_error = probes.iter().all(|probe| probe.finish_returns_same_error);
    assert!(whole_feed_output_discarded);
    assert_eq!(wire_bytes_leaked, 0);
    assert!(poisoned_after_overflow);
    assert!(later_feed_returns_same_error);
    assert!(finish_returns_same_error);
    assert!(outcomes_chunk_independent);
    assert!(split_offsets_checked > 0);

    // This verdict is produced by the executed 4-byte UTF-8 value case: two
    // scalars reach the cap, and a single following ASCII byte overflows it.
    let value_exact_accepted = exactly_cap(value_config, value_exact_wire.as_bytes());
    let value_overflow = probes.iter().find(|probe| {
        probe.error.field == "max_structured_field_bytes" && probe.error.limit == value_cap
    });
    let counted_in_utf8_bytes = value_exact_accepted
        && "éé".len() == value_cap
        && "éé".chars().count() < value_cap
        && value_overflow
            .map(|probe| probe.error.variant == "StructuredFieldOverflow")
            .unwrap_or(false);
    assert!(counted_in_utf8_bytes);

    let case_ids = per_case.keys().cloned().collect::<Vec<_>>();
    let expected_case_ids = vec![
        "complete_fields",
        "empty_pattern",
        "every_split_offset",
        "invalid_regex",
        "multibyte_split",
        "repeated_tool_calls",
        "reserved_parameter_close",
        "unterminated_field",
        "zero_width_content",
    ];
    assert_eq!(case_ids, expected_case_ids);
    assert!(per_case.values().all(|case| {
        case["one_chunk_equals_every_byte_split"]
            .as_bool()
            .unwrap_or(false)
    }));
    let cases_total = per_case.len();

    let complete_output = json!({
        "thinking": complete_streamed.output.thinking,
        "content": complete_streamed.output.content,
        "tool_calls": complete_streamed.output.tool_calls.iter().map(|call| json!({
            "name": call.name,
            "arguments": call.arguments,
            "transformed": call.transformed,
        })).collect::<Vec<_>>(),
        "tool_call_count": complete_streamed.output.tool_calls.len(),
        "delimiter_leakage": delimiter_leakage,
    });

    let summary = json!({
        "observed_default_limits": defaults,
        "limits": defaults,
        "exactly_cap_cases": exactly_cap_cases,
        "exactly_cap_all_accepted": exactly_cap_all_accepted,
        "cap_plus_one_cases": cap_plus_one_cases,
        "cap_plus_one_all_typed_overflow": cap_plus_one_all_typed_overflow,
        "overflow_variants_observed": overflow_variants_observed,
        "whole_feed_output_discarded": whole_feed_output_discarded,
        "wire_bytes_leaked": wire_bytes_leaked,
        "poisoned_after_overflow": poisoned_after_overflow,
        "later_feed_returns_same_error": later_feed_returns_same_error,
        "finish_returns_same_error": finish_returns_same_error,
        "split_offsets_checked": split_offsets_checked,
        "outcomes_chunk_independent": outcomes_chunk_independent,
        "multibyte_split_cases": multibyte_split_cases,
        "zero_width_cases": zero_width_cases,
        "empty_pattern_rejected_at_load": empty_pattern_rejected_at_load,
        "default_limits_test_invoked": default_limits_test_invoked,
        "counted_in_utf8_bytes": counted_in_utf8_bytes,
        "variants_name_field_and_limit": variants_name_field_and_limit,
        "cases_total": cases_total,
        "case_ids": case_ids,
        "per_case": per_case,
        "complete_output": complete_output,
        "unterminated_finish_error": unterminated_finish_error,
        "reserved_delimiter_rejected_no_partial_output": reserved_delimiter_rejected_no_partial_output,
        "invalid_regex_rejected_at_load": invalid_regex_rejected_at_load,
    });

    if let Some(path) = env::var_os("RESPONSE_TEMPLATE_TEST_SUMMARY") {
        // The aggregate owns only this path; the named default test above owns
        // RESPONSE_TEMPLATE_DEFAULT_LIMITS_SUMMARY in its isolated cargo run.
        write_new_json(path, &summary);
    }
}
