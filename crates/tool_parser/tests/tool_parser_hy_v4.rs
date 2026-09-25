use openai_protocol::common::Tool;
use serde_json::json;
use tool_parser::{parsers::HyV4Parser, traits::ToolParser};

#[expect(clippy::unwrap_used, reason = "literal test fixture must deserialize")]
fn tools() -> Vec<Tool> {
    vec![serde_json::from_value(json!({"type":"function","function":{"name":"run","parameters":{"type":"object","properties":{"text":{"type":"string"},"count":{"type":"integer"},"mixed":{"anyOf":[{"type":"string"},{"type":"boolean"}]}}}}})).unwrap()]
}

#[tokio::test]
async fn hy4_all_split_points_and_typed_arguments() {
    for suffix in ["", ":6124c78e", ":another-checkpoint"] {
        let text=format!("hello<tool_calls{suffix}><tool_call{suffix}>run<arg_key{suffix}>text</arg_key{suffix}><arg_value{suffix}>  中文🙂\"true\" &amp; </tool_call{suffix}>  </arg_value{suffix}><arg_key{suffix}>count</arg_key{suffix}><arg_value{suffix}>42</arg_value{suffix}><arg_key{suffix}>mixed</arg_key{suffix}><arg_value{suffix}>TRUE</arg_value{suffix}></tool_call{suffix}><tool_call{suffix}>run</tool_call{suffix}></tool_calls{suffix}>tail");
        let mut expected = None;
        for boundary in (0..=text.len()).filter(|i| text.is_char_boundary(*i)) {
            let mut p = HyV4Parser::new();
            let mut content = String::new();
            let mut calls = vec![];
            for chunk in [&text[..boundary], &text[boundary..]] {
                let r = p.parse_incremental(chunk, &tools()).await.unwrap();
                content.push_str(&r.normal_text);
                calls.extend(r.calls);
            }
            content.push_str(&p.take_unstreamed_normal_text());
            assert_eq!(content, "hellotail", "split {boundary}");
            let calls = join_calls(calls);
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].tool_index, 0);
            assert_eq!(calls[1].tool_index, 1);
            let args: serde_json::Value = serde_json::from_str(&calls[0].parameters).unwrap();
            assert_eq!(args["count"], 42);
            assert_eq!(args["mixed"], true);
            assert_eq!(
                args["text"],
                format!("  中文🙂\"true\" &amp; </tool_call{suffix}>  ")
            );
            expected = Some(args);
        }
        let (content, calls) = HyV4Parser::new()
            .parse_complete_with_tools(&text, &tools())
            .await
            .unwrap();
        assert_eq!(content, "hellotail");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
            expected.unwrap()
        );
    }
}
#[tokio::test]
async fn hy4_text_truncation_and_reset() {
    let mut p = HyV4Parser::new();
    let r = p.parse_incremental("text <tool_ca", &[]).await.unwrap();
    assert_eq!(r.normal_text, "text ");
    assert_eq!(p.take_unstreamed_normal_text(), "<tool_ca");
    p.reset();
    let (_, calls) = p
        .parse_complete("<tool_calls><tool_call>run</tool_call></tool_calls>")
        .await
        .unwrap();
    assert_eq!(calls[0].function.arguments, "{}");
}

#[tokio::test]
async fn hy4_character_chunks_eof_and_factory() {
    let factory = tool_parser::factory::ParserFactory::new();
    let p = factory.get_parser("tencent/Hy4-preview-FP8").unwrap();
    assert!(p.has_tool_markers("<tool_calls:6124c78e>"));
    let text =
        "<tool_calls:6124c78e><tool_call:6124c78e>run</tool_call:6124c78e></tool_calls:6124c78e>";
    let mut parser = HyV4Parser::new();
    let mut calls = vec![];
    for ch in text.chars() {
        calls.extend(
            parser
                .parse_incremental(&ch.to_string(), &[])
                .await
                .unwrap()
                .calls,
        );
    }
    let calls = join_calls(calls);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].parameters, "{}");
    parser.reset();
    let incomplete = "<tool_calls:6124c78e><tool_call:6124c78e>unfinished";
    assert!(parser
        .parse_incremental(incomplete, &[])
        .await
        .unwrap()
        .calls
        .is_empty());
    assert_eq!(parser.take_unstreamed_normal_text(), incomplete);
    assert!(parser.take_unstreamed_normal_text().is_empty());
}

#[tokio::test]
async fn hy4_malformed_is_not_a_partial_call_and_buffer_is_bounded() {
    let malformed="before<tool_calls><tool_call>run<arg_key>x</arg_key>missing_value</tool_call></tool_calls>after";
    let (content, calls) = HyV4Parser::new().parse_complete(malformed).await.unwrap();
    assert!(calls.is_empty());
    assert_eq!(content, malformed);
    let mut p = HyV4Parser::new();
    assert!(p
        .parse_incremental(&"x".repeat(4 * 1024 * 1024 + 1), &[])
        .await
        .is_err());
}

#[tokio::test]
async fn hy4_unannounced_malformed_calls_fall_back_to_text() {
    for text in [
        "<tool_calls><tool_call>run<parameter>x</parameter></tool_call></tool_calls>tail",
        "<tool_calls><tool_call></tool_call></tool_calls>",
    ] {
        let (expected, calls) = HyV4Parser::new().parse_complete(text).await.unwrap();
        assert!(calls.is_empty());
        assert_eq!(expected, text);

        let mut parser = HyV4Parser::new();
        let mut content = String::new();
        for ch in text.chars() {
            let result = parser
                .parse_incremental(&ch.to_string(), &[])
                .await
                .unwrap();
            assert!(result.calls.is_empty());
            content.push_str(&result.normal_text);
        }
        content.push_str(&parser.take_unstreamed_normal_text());
        assert_eq!(content, expected);
    }
}

#[tokio::test]
async fn hy4_typed_value_end_marker_can_be_split_after_cached_scan() {
    let schemas: Vec<Tool> = vec![serde_json::from_value(json!({
        "type": "function",
        "function": {
            "name": "run",
            "parameters": {
                "properties": {"items": {"type": "array"}}
            }
        }
    }))
    .unwrap()];
    let mut parser = HyV4Parser::new();
    let first = parser
        .parse_incremental("<tool_calls><tool_call>run<arg_key>it", &schemas)
        .await
        .unwrap();
    assert_eq!(first.calls[0].parameters, "{");
    assert!(parser
        .parse_incremental("ems</arg_k", &schemas)
        .await
        .unwrap()
        .calls
        .is_empty());
    let value = parser
        .parse_incremental("ey><arg_value>[1,2]", &schemas)
        .await
        .unwrap();
    assert_eq!(value.calls[0].parameters, "\"items\":");
    assert!(parser
        .parse_incremental("</arg_va", &schemas)
        .await
        .unwrap()
        .calls
        .is_empty());
    let last = parser
        .parse_incremental("lue></tool_call></tool_calls>", &schemas)
        .await
        .unwrap();
    assert_eq!(last.calls[0].parameters, "[1,2]}");
}

fn join_calls(
    deltas: Vec<tool_parser::types::ToolCallItem>,
) -> Vec<tool_parser::types::ToolCallItem> {
    let mut calls: Vec<tool_parser::types::ToolCallItem> = vec![];
    for delta in deltas {
        if delta.tool_index == calls.len() {
            assert!(delta.name.is_some());
            calls.push(delta);
        } else {
            assert!(
                delta.name.is_none(),
                "tool name must be emitted exactly once"
            );
            calls[delta.tool_index]
                .parameters
                .push_str(&delta.parameters);
        }
    }
    calls
}

#[tokio::test]
async fn hy4_emits_name_and_string_before_call_ends() {
    let mut p = HyV4Parser::new();
    let name = p
        .parse_incremental(
            "<tool_calls:6124c78e><tool_call:6124c78e>run<arg_key:6124c78e>",
            &tools(),
        )
        .await
        .unwrap();
    assert_eq!(name.calls.len(), 1);
    assert_eq!(name.calls[0].name.as_deref(), Some("run"));
    assert_eq!(name.calls[0].parameters, "{");
    let value = p
        .parse_incremental("text</arg_key:6124c78e><arg_value:6124c78e>hello", &tools())
        .await
        .unwrap();
    assert_eq!(value.calls[0].name, None);
    assert_eq!(value.calls[0].parameters, "\"text\":\"hello");
    let split = p.parse_incremental("</arg_va", &tools()).await.unwrap();
    assert!(split.calls.is_empty());
    let close = p
        .parse_incremental("lue:6124c78e>", &tools())
        .await
        .unwrap();
    assert_eq!(close.calls[0].parameters, "\"");
    let end = p
        .parse_incremental("</tool_call:6124c78e></tool_calls:6124c78e>", &tools())
        .await
        .unwrap();
    assert_eq!(end.calls[0].parameters, "}");
    assert!(p.take_unstreamed_normal_text().is_empty());
}

#[tokio::test]
async fn hy4_unicode_escaping_unions_and_adjacent_calls_are_append_only() {
    let text="<tool_calls><tool_call>run<arg_key>text</arg_key><arg_value>中文🙂\"\\\n\t</tool_call></arg_value><arg_key>mixed</arg_key><arg_value>TRUE</arg_value></tool_call><tool_call>run</tool_call></tool_calls>";
    let mut p = HyV4Parser::new();
    let mut deltas = vec![];
    for ch in text.chars() {
        let r = p
            .parse_incremental(&ch.to_string(), &tools())
            .await
            .unwrap();
        assert!(r.normal_text.is_empty());
        deltas.extend(r.calls);
    }
    assert!(deltas.len() > 5, "string must arrive incrementally");
    let calls = join_calls(deltas);
    assert_eq!(calls.len(), 2);
    let args: serde_json::Value = serde_json::from_str(&calls[0].parameters).unwrap();
    assert_eq!(args["text"], "中文🙂\"\\\n\t</tool_call>");
    assert_eq!(args["mixed"], true);
    assert_eq!(calls[1].parameters, "{}");
}

#[tokio::test]
async fn hy4_truncated_announced_call_is_not_repaired_or_leaked() {
    let mut p = HyV4Parser::new();
    let r = p
        .parse_incremental(
            "<tool_calls><tool_call>run<arg_key>text</arg_key><arg_value>unfinished</arg_v",
            &tools(),
        )
        .await
        .unwrap();
    assert_eq!(r.calls[0].parameters, "{\"text\":\"unfinished");
    assert!(p.take_unstreamed_normal_text().is_empty());
    assert!(p.get_unstreamed_tool_args().is_none());
    p.reset();
    let r = p
        .parse_incremental(
            "<tool_calls><tool_call>run</tool_call></tool_calls>",
            &tools(),
        )
        .await
        .unwrap();
    assert_eq!(r.calls[0].tool_index, 0);
    assert_eq!(r.calls[0].parameters, "{}");
}

#[tokio::test]
async fn hy4_union_waits_for_value_but_not_whole_call() {
    let mut p = HyV4Parser::new();
    let r = p
        .parse_incremental(
            "<tool_calls><tool_call>run<arg_key>mixed</arg_key><arg_value>T",
            &tools(),
        )
        .await
        .unwrap();
    assert_eq!(r.calls[0].parameters, "{\"mixed\":");
    let r = p
        .parse_incremental("RUE</arg_value>", &tools())
        .await
        .unwrap();
    assert_eq!(r.calls[0].parameters, "true");
}

#[tokio::test]
async fn hy4_malformed_after_announcement_errors_instead_of_rewriting() {
    let mut p = HyV4Parser::new();
    let r = p
        .parse_incremental("<tool_calls><tool_call>run<arg_key>", &tools())
        .await
        .unwrap();
    assert_eq!(r.calls[0].name.as_deref(), Some("run"));
    assert!(p
        .parse_incremental("text</arg_key>bad markup", &tools())
        .await
        .is_err());
    assert!(p.take_unstreamed_normal_text().is_empty());
    assert!(p.parse_incremental("ignored", &tools()).await.is_err());
}

#[tokio::test]
async fn hy4_typed_values_emit_before_later_arguments_and_preserve_types() {
    let schemas:Vec<Tool>=vec![serde_json::from_value(json!({"type":"function","function":{"name":"run","parameters":{"properties":{"items":{"type":"array"},"object":{"type":"object"},"count":{"type":"integer"}}}}})).unwrap()];
    let mut p = HyV4Parser::new();
    let first = p
        .parse_incremental(
            "<tool_calls><tool_call>run<arg_key>items</arg_key><arg_value>[1,",
            &schemas,
        )
        .await
        .unwrap();
    assert_eq!(first.calls[0].parameters, "{\"items\":");
    let array = p
        .parse_incremental("2]</arg_value>", &schemas)
        .await
        .unwrap();
    assert_eq!(array.calls[0].parameters, "[1,2]");
    let rest=p.parse_incremental("<arg_key>object</arg_key><arg_value>{\"x\":true}</arg_value><arg_key>count</arg_key><arg_value>9007199254740993</arg_value></tool_call></tool_calls>",&schemas).await.unwrap();
    let calls = join_calls(
        first
            .calls
            .into_iter()
            .chain(array.calls)
            .chain(rest.calls)
            .collect(),
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&calls[0].parameters).unwrap(),
        json!({"items":[1,2],"object":{"x":true},"count":9007199254740993u64})
    );
}

#[tokio::test]
async fn hy4_does_not_buffer_the_entire_streamed_string() {
    let mut p = HyV4Parser::new();
    p.parse_incremental(
        "<tool_calls><tool_call>run<arg_key>text</arg_key><arg_value>",
        &tools(),
    )
    .await
    .unwrap();
    let text = "x".repeat(1024 * 1024);
    for _ in 0..5 {
        let r = p.parse_incremental(&text, &tools()).await.unwrap();
        assert_eq!(r.calls[0].parameters.len(), text.len());
    }
    let r = p
        .parse_incremental("</arg_value></tool_call></tool_calls>", &tools())
        .await
        .unwrap();
    assert_eq!(r.calls[0].parameters, "\"}");
}

#[tokio::test]
async fn hy4_eof_after_call_does_not_leak_partial_group_closer() {
    let mut p = HyV4Parser::new();
    let r = p
        .parse_incremental("<tool_calls><tool_call>run</tool_call></tool_ca", &[])
        .await
        .unwrap();
    assert_eq!(r.calls[0].parameters, "{}");
    assert!(p.take_unstreamed_normal_text().is_empty());
}
