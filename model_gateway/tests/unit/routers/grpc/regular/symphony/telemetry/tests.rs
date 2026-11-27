use serde_json::Value;
use smg_external_router::metrics::handles::force_on_this_thread;
use symphony::{
    CallEnd, Declared, DropReason, Engine, EngineFinish, Events, FinishReason, Input, Parser, Text,
};

use super::*;
use crate::routers::grpc::regular::symphony::testing::{
    json_lines, line_with, rendered, request_span,
};

fn malformed(text: &str, why: MalformedReason) -> Event<'static> {
    Event::Malformed {
        text: Text::uncounted(text.to_owned()),
        why,
    }
}

fn call_start(index: u32, source: &str) -> Event<'static> {
    call_named(index, "lookup", source)
}

fn call_named(index: u32, name: &str, source: &str) -> Event<'static> {
    Event::ToolCallStart {
        index,
        id: format!("call_{index}"),
        name: name.to_owned(),
        source: Text::uncounted(source.to_owned()),
    }
}

fn finish(tool_calls: u32) -> Event<'static> {
    Event::Finish {
        reason: FinishReason::Stop,
        tool_calls,
        reasoning_tokens: 0,
    }
}

#[test]
fn every_registry_table_labels_itself_and_the_rest_is_named() {
    for table in Table::ALL {
        assert_eq!(table_label(table.name()), table.name());
    }
    assert_eq!(table_label("response_template"), "response_template");
    assert_eq!(table_label("markers"), "other");
    assert_eq!(table_label(""), "other");
}

#[test]
fn a_tally_counts_the_runs_their_bytes_and_where_the_first_one_starts() {
    let mut tally = Tally::default();
    let events = [
        Event::Content(Text::uncounted("Hello ")),
        Event::Dropped {
            text: Text::uncounted("<tool_call>"),
            why: DropReason::Wrapper,
        },
        malformed("\nnot a call\n", MalformedReason::InvalidArguments),
        Event::Dropped {
            text: Text::uncounted("</tool_call>"),
            why: DropReason::Wrapper,
        },
        malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
        malformed("x", MalformedReason::CallWithoutName),
        finish(0),
    ];
    for event in &events {
        tally.see(event, false);
    }
    assert_eq!(tally.malformed(), 3);
    assert_eq!(tally.malformed_bytes, 12 + 12 + 1);
    assert_eq!(
        tally.first,
        Some(FirstRun {
            reason: "invalid_arguments",
            why: "arguments that did not parse under the call's syntax",
            at: "Hello <tool_call>".len(),
            bytes: 12,
        })
    );
    assert_eq!(
        tally.reasons_seen(),
        "unterminated_region,invalid_arguments,call_without_name"
    );
    assert_eq!(tally.suppressed_calls, 0);
}

#[test]
fn a_run_counts_once_however_the_stream_cut_the_output() {
    // One output a reason that arrives in pieces, each through its own table: a run of free
    // text is an event a word when it comes whole, an event a character by character.
    let tally_of = |table: &str, prompt: &str, pieces: &[&str]| {
        let table = Table::named(table).expect("a table of that name");
        let mut engine = Engine::new(table.format(), Declared::default());
        let mut tally = Tally::default();
        let prompt = Input::Prompt {
            token_ids: &[],
            text: prompt,
        };
        let deltas = pieces.iter().copied().map(|text| Input::Delta {
            token_ids: &[],
            text,
            spans: &[],
        });
        let end = Input::End {
            finish: EngineFinish::Stop,
        };
        for input in std::iter::once(prompt).chain(deltas).chain([end]) {
            let mut events = Events::new();
            engine.feed(input, &mut events).expect("the reader");
            for event in events.drain() {
                tally.see(&event, false);
            }
        }
        tally
    };
    for (table, prompt, output, runs, reasons) in [
        (
            "qwen3_tagged",
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "<tool_call>\n{\"name\": \"check\", \"arguments\": {\"n\": 97}}\n</tool_call>",
            2,
            "text_between_tags,call_without_function_tag",
        ),
        (
            "deepseek_v4.1",
            "",
            "<｜DSML｜ calls>\nFirst the number.\n</｜DSML｜ calls>",
            1,
            "text_between_calls",
        ),
        (
            "command",
            "",
            "<|START_ACTION|>check the number 97<|END_ACTION|>",
            1,
            "text_in_place_of_the_list",
        ),
        (
            "function_gemma",
            "",
            "<start_function_call>call:check{n:97} is the call<end_function_call>",
            1,
            "text_after_the_arguments",
        ),
        (
            "mistral",
            "",
            "[TOOL_CALLS] [{\"name\": \"check\", \"arguments\": {\"n\": 97}}] is the list",
            1,
            "text_after_the_list",
        ),
    ] {
        let whole = tally_of(table, prompt, &[output]);
        let characters: Vec<&str> = output.split_inclusive(|_: char| true).collect();
        let cut = tally_of(table, prompt, &characters);
        assert_eq!(
            (whole.malformed(), cut.malformed()),
            (runs, runs),
            "{table}"
        );
        assert_eq!(whole.reasons_seen(), reasons, "{table}");
        assert_eq!(cut.reasons, whole.reasons, "{table}");
        assert_eq!(cut.first, whole.first, "{table}");
    }
}

#[test]
fn refused_bytes_size_the_run_that_announced_them() {
    // The refuse order: the event of no bytes, then the bytes it refuses; a second refused
    // run's bytes count, but are not the first run's.
    let refused = |text: &str| Event::Dropped {
        text: Text::uncounted(text.to_owned()),
        why: DropReason::MalformedCall,
    };
    let mut tally = Tally::default();
    for event in [
        Event::Content(Text::uncounted("Hi")),
        malformed("", MalformedReason::TagCutShort),
        refused("<arg_key>format=2D"),
        malformed("", MalformedReason::TagOutOfPlace),
        refused("</arg_value>"),
        finish(1),
    ] {
        tally.see(&event, false);
    }
    assert_eq!(tally.malformed(), 2);
    assert_eq!(tally.malformed_bytes, 18 + 12);
    assert_eq!(
        tally.first,
        Some(FirstRun {
            reason: "tag_cut_short",
            why: "a tag that another tag or the call's end cut short",
            at: 2,
            bytes: 18,
        })
    );
    // The cut order: the bytes held at the cut, then the word before the call's end: the
    // bytes count, and the run keeps the size its event carried.
    let mut tally = Tally::default();
    for event in [
        refused("<"),
        malformed("", MalformedReason::UnterminatedRegion),
        Event::ToolCallEnd {
            index: 0,
            source: Text::uncounted("</tool_call>"),
            ended: CallEnd::Closed,
        },
        finish(1),
    ] {
        tally.see(&event, false);
    }
    assert_eq!(tally.malformed_bytes, 1);
    assert_eq!(
        tally.first,
        Some(FirstRun {
            reason: "unterminated_region",
            why: "the stream ended inside a region that never closed",
            at: 1,
            bytes: 0,
        })
    );
    // The template's wrapping after a run is not the run's.
    let mut tally = Tally::default();
    for event in [
        malformed("x", MalformedReason::CallWithoutName),
        Event::Dropped {
            text: Text::uncounted("\n"),
            why: DropReason::Wrapper,
        },
        finish(0),
    ] {
        tally.see(&event, false);
    }
    assert_eq!(tally.malformed_bytes, 1);
    assert_eq!(tally.first.as_ref().map(|first| first.bytes), Some(1));
}

#[test]
fn a_botched_keyed_block_is_sized_by_its_refused_bytes() {
    // Through the keyed table itself: a call whose key tag never opened. The assembler refuses
    // the tags and the value behind events of no bytes; the tally counts the runs and sizes
    // them by the refused bytes, the first by the bytes right after it.
    let output = "</think>At once:<tool_call>f</arg_key><arg_value>1975</arg_value></tool_call>";
    let mut engine = Engine::new(Table::Glm.format(), Declared::default());
    let mut out = Events::new();
    engine
        .feed(
            Input::Prompt {
                token_ids: &[],
                text: "<|assistant|><think>",
            },
            &mut out,
        )
        .expect("the prompt");
    engine
        .feed(
            Input::Delta {
                token_ids: &[],
                text: output,
                spans: &[],
            },
            &mut out,
        )
        .expect("the delta");
    engine
        .feed(
            Input::End {
                finish: EngineFinish::Stop,
            },
            &mut out,
        )
        .expect("the end");
    let mut tally = Tally::default();
    for event in out.as_slice() {
        tally.see(event, false);
    }
    assert_eq!(tally.malformed(), 4, "{:?}", out.as_slice());
    assert_eq!(
        tally.malformed_bytes,
        "</arg_key><arg_value>1975</arg_value>".len() as u64
    );
    assert_eq!(
        tally.first,
        Some(FirstRun {
            reason: "tag_out_of_place",
            why: "a tag where the call's syntax has none",
            at: "</think>At once:<tool_call>f".len(),
            bytes: "</arg_key>".len(),
        })
    );
    assert_eq!(tally.tool_calls, 1);
}

#[test]
fn the_first_run_of_a_syntax_reason_carries_its_own_label_and_sentence() {
    let mut tally = Tally::default();
    tally.see(
        &malformed("<|end|>", MalformedReason::NameWithoutACall),
        false,
    );
    let first = tally.first.as_ref().expect("a first run");
    assert_eq!(first.reason, "name_without_a_call");
    assert_eq!(first.why, "a name without a call");
    assert_eq!(first.at, 0);
}

#[test]
fn every_reason_the_crate_names_has_a_label_of_its_own_spelled_as_the_crate_spells_it() {
    // The crate's own list of its reasons (`MalformedReason::ALL`, written with the enum by
    // one macro list) drives this: the label is the crate's serde spelling of the reason, so a
    // line, a trace and the counter agree on the name; no two reasons share a label, none
    // lands on `other`, and a reason added to the crate fails this test until it has a label.
    let mut labels = std::collections::HashSet::new();
    for (index, why) in MalformedReason::ALL.iter().copied().enumerate() {
        let label = reason_label(why);
        let spelled = serde_json::to_value(why).expect("a reason serializes");
        assert_eq!(Some(label), spelled.as_str(), "{why:?}");
        assert_eq!(reason_index(why), index, "{why:?}");
        assert_ne!(label, "other", "{why:?}");
        assert!(labels.insert(label), "{why:?} shares its label");
    }
    assert_eq!(labels.len() + 1, REASONS.len());
    assert_eq!(REASONS[OTHER_REASON], "other");
}

#[test]
fn a_clean_tally_allocates_no_counters_and_a_botched_one_counts_each_reason_by_name() {
    let mut tally = Tally::default();
    tally.see(&Event::Content(Text::uncounted("Hi")), false);
    tally.see(&finish(0), false);
    assert!(tally.reasons.is_none());
    assert_eq!(tally.malformed(), 0);
    assert_eq!(tally.reasons_seen(), "");
    let mut tally = Tally::default();
    for event in [
        malformed("", MalformedReason::TextWhereTheNameShouldBe),
        malformed("", MalformedReason::TagOutOfPlace),
        malformed("", MalformedReason::TextBetweenTags),
        malformed("", MalformedReason::TagOutOfPlace),
        malformed("", MalformedReason::CallWithoutName),
        finish(0),
    ] {
        tally.see(&event, false);
    }
    assert_eq!(tally.malformed(), 5);
    assert_eq!(
        tally.reasons_seen(),
        "text_between_tags,tag_out_of_place,call_without_name,text_where_the_name_should_be"
    );
    let reasons = tally.reasons();
    assert_eq!(reasons[reason_index(MalformedReason::TagOutOfPlace)], 2);
    assert_eq!(reasons[reason_index(MalformedReason::TextBetweenTags)], 1);
}

#[test]
fn a_kind_said_again_inside_an_answer_counts_the_output_as_clean() {
    // Through the table whose messages say their kind after their opener, as a request
    // without tools reads it: a JSON answer whose text marker the model wrote twice, one
    // with a second thought opened inside the answer's message, as an engine that starts the
    // grammar after its own reasoning parser lets the model write them, and one whose text
    // marker the model wrote inside its thought. The markers are framing, the thoughts are
    // reasoning, the JSON is the content, and the tally sees no malformed run: the output
    // counts as clean, under no reason.
    let json = "{\"city\": \"Oslo\"}";
    for output in [
        format!(
            "<|content_thinking|>plan<|end_message|><|message_model|><|content_text|>\
                 <|content_text|>{json}<|end_message|>"
        ),
        format!(
            "<|content_thinking|>plan<|end_message|><|message_model|><|content_text|>\
                 <|content_thinking|>more<|end_message|><|message_model|><|content_text|>{json}\
                 <|end_message|>"
        ),
        format!("<|content_thinking|>plan<|content_text|>{json}<|end_message|>"),
    ] {
        let mut engine = Engine::new(Table::Inkling.format().without_calls(), Declared::default());
        let mut out = Events::new();
        engine
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: "<|message_model|>",
                },
                &mut out,
            )
            .expect("the prompt");
        engine
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: &output,
                    spans: &[],
                },
                &mut out,
            )
            .expect("the delta");
        engine
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("the end");
        let mut tally = Tally::default();
        let mut content = String::new();
        for event in out.as_slice() {
            tally.see(event, false);
            if let Event::Content(text) = event {
                content.push_str(&text.text);
            }
        }
        assert_eq!(content, json, "{:?}", out.as_slice());
        assert_eq!(tally.malformed(), 0, "{:?}", out.as_slice());
        assert_eq!(tally.malformed_bytes, 0);
        assert_eq!(tally.reasons_seen(), "");
        assert_eq!(tally.first, None);
    }
}

#[test]
fn a_marker_out_of_place_has_a_label_of_its_own_and_counts_the_output_as_malformed() {
    // The crate says so with an event of no bytes before the marker's text: the run counts,
    // the first run stands where the marker does, and its bytes are none.
    let mut tally = Tally::default();
    tally.see(&Event::Content(Text::uncounted("<|text|>")), false);
    tally.see(&malformed("", MalformedReason::MarkerOutOfPlace), false);
    tally.see(&Event::Content(Text::uncounted("<|text|>{}")), false);
    assert_eq!(tally.malformed(), 1);
    assert_eq!(tally.malformed_bytes, 0);
    assert_eq!(tally.reasons_seen(), "marker_out_of_place");
    assert_eq!(
        tally.first,
        Some(FirstRun {
            reason: "marker_out_of_place",
            why: "a marker of the table where no row leads, kept as text",
            at: "<|text|>".len(),
            bytes: 0,
        })
    );
    assert_eq!(
        reason_label(MalformedReason::MarkerOutOfPlace),
        "marker_out_of_place"
    );
}

#[tracing_test::traced_test]
#[test]
fn a_marker_in_an_output_run_past_the_models_end_is_no_run_and_writes_a_debug_line() {
    // The request ran the output past the model's own end: the two closes the tail wrote
    // are kept as text like any marker out of place, counted apart from the runs, and the
    // output is clean. One line says so, at debug.
    let scrape = rendered(|| {
        let mut tally = Tally::new(true);
        tally.see(&Event::Content(Text::uncounted("OK")), false);
        tally.see(&malformed("", MalformedReason::MarkerOutOfPlace), false);
        tally.see(&Event::Content(Text::uncounted("</think>OK")), false);
        tally.see(&malformed("", MalformedReason::MarkerOutOfPlace), false);
        tally.see(&Event::Content(Text::uncounted("</think>")), false);
        tally.see(&finished(FinishReason::Length), false);
        assert_eq!(tally.malformed(), 0);
        assert_eq!(tally.reasons_seen(), "");
        assert_eq!(tally.first, None);
        assert_eq!(tally.markers_past_the_end, 2);
        assert!(!tally.is_empty());
        tally.report(
            "qwen3",
            Who {
                request_id: "req-past-the-end",
                model: "m",
                choice: 0,
            },
        );
    });
    let outputs = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"clean\""])
        .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(
        line_with(
            &scrape,
            OUTPUTS,
            &["table=\"qwen3\"", "outcome=\"malformed\""]
        )
        .is_none_or(|line| line.ends_with(" 0")),
        "{scrape}"
    );
    assert!(!scrape.contains(MALFORMED), "{scrape}");
    assert!(!scrape.contains(MALFORMED_BYTES), "{scrape}");
    assert!(!logs_contain("did not read the whole output"));
    logs_assert(|lines: &[&str]| {
        let line = lines
            .iter()
            .find(|line| line.contains("req-past-the-end"))
            .ok_or_else(|| "no line for the output".to_string())?;
        if !line.contains("DEBUG") {
            return Err(format!("the line is not at debug: {line}"));
        }
        for field in [
            "Symphony kept markers of the table as text in an output the request ran past the model's own end",
            "markers=2",
            "finish=\"length\"",
            "table=\"qwen3\"",
            "bytes=20",
            "choice=0",
        ] {
            if !line.contains(field) {
                return Err(format!("{field} missing from the line: {line}"));
            }
        }
        Ok(())
    });
}

#[tracing_test::traced_test]
#[test]
fn a_refused_run_in_an_output_run_past_the_models_end_still_counts_and_warns() {
    // The tail's marker is counted apart; the region the table gave up on is a run as in
    // any output: the output is malformed under its reason alone, and the warning stands.
    let scrape = rendered(|| {
        let mut tally = Tally::new(true);
        tally.see(&Event::Content(Text::uncounted("Hello ")), false);
        tally.see(
            &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
            false,
        );
        tally.see(&malformed("", MalformedReason::MarkerOutOfPlace), false);
        tally.see(&Event::Content(Text::uncounted("</think>")), false);
        tally.see(&finished(FinishReason::Length), false);
        assert_eq!(tally.malformed(), 1);
        assert_eq!(tally.reasons_seen(), "unterminated_region");
        assert_eq!(tally.markers_past_the_end, 1);
        tally.report(
            "qwen3",
            Who {
                request_id: "req-past-the-end-refused",
                model: "m",
                choice: 0,
            },
        );
    });
    let outputs = line_with(
        &scrape,
        OUTPUTS,
        &["table=\"qwen3\"", "outcome=\"malformed\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    let runs = line_with(
        &scrape,
        MALFORMED,
        &["table=\"qwen3\"", "reason=\"unterminated_region\""],
    )
    .expect("the reason counter");
    assert!(runs.ends_with(" 1"), "{runs}");
    assert!(
        line_with(
            &scrape,
            MALFORMED,
            &["table=\"qwen3\"", "reason=\"marker_out_of_place\""]
        )
        .is_none_or(|line| line.ends_with(" 0")),
        "{scrape}"
    );
    assert!(logs_contain(
        "Symphony did not read the whole output as content, reasoning and calls"
    ));
    assert!(logs_contain("reason=\"unterminated_region\""));
    assert!(logs_contain("reasons=unterminated_region"));
    assert!(logs_contain("malformed=1"));
    assert!(logs_contain("markers=1"));
}

#[test]
fn a_tally_not_run_past_the_end_counts_a_marker_as_a_run() {
    let mut tally = Tally::new(false);
    tally.see(&Event::Content(Text::uncounted("a")), false);
    tally.see(&malformed("", MalformedReason::MarkerOutOfPlace), false);
    tally.see(&Event::Content(Text::uncounted("</think>b")), false);
    assert_eq!(tally.malformed(), 1);
    assert_eq!(tally.reasons_seen(), "marker_out_of_place");
    assert_eq!(tally.markers_past_the_end, 0);
}

#[test]
fn calls_after_the_first_count_as_suppressed_only_when_the_reader_drops_them() {
    let events = [
        call_start(0, "<tool_call>"),
        call_start(1, "<tool_call>"),
        call_start(2, "<tool_call>"),
        finish(3),
    ];
    let mut dropping = Tally::default();
    let mut keeping = Tally::default();
    for event in &events {
        dropping.see(event, true);
        keeping.see(event, false);
    }
    assert_eq!(dropping.suppressed_calls, 2);
    assert_eq!(keeping.suppressed_calls, 0);
    // The count the client gets: one call where the reader drops the rest.
    assert_eq!(dropping.tool_calls, 1);
    assert_eq!(keeping.tool_calls, 3);
    assert_eq!(dropping.at, 33);
}

/// The span object of the last line `work` wrote: the access log's last line of the request,
/// written under the request span after the output ended, as the JSON layer renders it.
fn last_span_of(work: impl FnOnce(&Span)) -> Value {
    let mut lines = json_lines(|| {
        let span = request_span();
        let _entered = span.enter();
        work(&span);
        tracing::info!(target: "smg::response", "finished processing request");
    });
    let mut last = lines.pop().expect("the access log line");
    last["span"].take()
}

#[test]
fn a_malformed_output_leaves_its_reading_on_the_request_span() {
    let span = last_span_of(|span| {
        let mut tally = Tally::default();
        tally.see(&Event::Content(Text::uncounted("Hi ")), false);
        tally.see(
            &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
            false,
        );
        tally.see(&finished(FinishReason::Length), false);
        tally.mark(span, &AtomicBool::new(false), "qwen3", 1);
    });
    assert_eq!(span["name"], "http_request");
    assert_eq!(span["symphony.table"], "qwen3");
    assert_eq!(span["symphony.outcome"], "malformed");
    assert_eq!(span["symphony.choice"], 1);
    assert_eq!(span["symphony.reason"], "unterminated_region");
    assert_eq!(span["symphony.bytes"], 15);
    assert_eq!(span["symphony.malformed_bytes"], 12);
    assert_eq!(span["symphony.finish"], "length");
}

#[test]
fn a_clean_output_leaves_nothing_on_the_request_span() {
    let span = last_span_of(|span| {
        let mut tally = Tally::default();
        tally.see(&Event::Content(Text::uncounted("Hi")), false);
        tally.see(&finished(FinishReason::Stop), false);
        tally.mark(span, &AtomicBool::new(false), "qwen3", 0);
    });
    assert_eq!(span["name"], "http_request");
    assert!(span.get("symphony.outcome").is_none(), "{span}");
    assert!(span.get("symphony.table").is_none(), "{span}");
    assert!(span.get("symphony.bytes").is_none(), "{span}");
}

#[test]
fn an_empty_and_an_unknown_tool_output_leave_their_outcome_and_what_names_it() {
    let empty = last_span_of(|span| {
        let mut tally = Tally::default();
        tally.see(&dropped("\n", 1, DropReason::Whitespace), false);
        tally.see(&finished(FinishReason::Stop), false);
        tally.mark(span, &AtomicBool::new(false), "olmo3", 0);
    });
    assert_eq!(empty["symphony.outcome"], "empty");
    assert_eq!(empty["symphony.table"], "olmo3");
    assert_eq!(empty["symphony.bytes"], 1);
    assert_eq!(empty["symphony.finish"], "stop");
    assert!(empty.get("symphony.reason").is_none(), "{empty}");
    assert!(empty.get("symphony.malformed_bytes").is_none(), "{empty}");
    let unknown = last_span_of(|span| {
        let mut tally = Tally::default();
        tally.see(&malformed("", MalformedReason::UnknownTool), false);
        tally.see(&call_named(0, "lookup_v2", "<tool_call>"), false);
        tally.see(&finished(FinishReason::ToolCalls), false);
        tally.mark(span, &AtomicBool::new(false), "qwen3", 0);
    });
    assert_eq!(unknown["symphony.outcome"], "unknown_tool");
    assert_eq!(unknown["symphony.reason"], "unknown_tool");
    assert_eq!(unknown["symphony.finish"], "tool_calls");
    assert!(
        unknown.get("symphony.malformed_bytes").is_none(),
        "{unknown}"
    );
}

#[test]
fn a_failed_and_a_panicked_reading_leave_the_outcome_and_the_table() {
    let failed = last_span_of(|span| mark_failed(span, &AtomicBool::new(false), "hy4", 7));
    assert_eq!(failed["symphony.outcome"], "failed");
    assert_eq!(failed["symphony.table"], "hy4");
    assert_eq!(failed["symphony.bytes"], 7);
    assert!(failed.get("symphony.finish").is_none(), "{failed}");
    assert!(failed.get("symphony.choice").is_none(), "{failed}");
    assert!(failed.get("symphony.reason").is_none(), "{failed}");
    let panicked = last_span_of(|span| mark_panicked(span, None));
    assert_eq!(panicked["symphony.outcome"], "panicked");
    assert_eq!(panicked["symphony.table"], "none");
    assert!(panicked.get("symphony.bytes").is_none(), "{panicked}");
    let read = last_span_of(|span| mark_panicked(span, Some("glm4")));
    assert_eq!(read["symphony.table"], "glm4");
}

#[test]
fn one_reading_per_request_stands_on_the_span_the_first_that_was_not_clean() {
    // Three choices and a failure after them: the clean reading marks nothing, the malformed
    // one marks, and the empty reading and the failure leave the mark, so the span shows one
    // reading's fields and never a mix.
    let span = last_span_of(|span| {
        let marked = AtomicBool::new(false);
        let mut clean = Tally::default();
        clean.see(&Event::Content(Text::uncounted("Hi")), false);
        clean.see(&finished(FinishReason::Stop), false);
        clean.mark(span, &marked, "qwen3", 0);
        assert!(
            !marked.load(Ordering::Acquire),
            "a clean reading marks nothing"
        );
        let mut refused = Tally::default();
        refused.see(&Event::Content(Text::uncounted("Hi ")), false);
        refused.see(
            &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
            false,
        );
        refused.see(&finished(FinishReason::Length), false);
        refused.mark(span, &marked, "qwen3", 1);
        let mut empty = Tally::default();
        empty.see(&dropped("\n", 1, DropReason::Whitespace), false);
        empty.see(&finished(FinishReason::Stop), false);
        empty.mark(span, &marked, "qwen3", 2);
        mark_failed(span, &marked, "qwen3", 3);
    });
    assert_eq!(span["symphony.outcome"], "malformed");
    assert_eq!(span["symphony.choice"], 1);
    assert_eq!(span["symphony.reason"], "unterminated_region");
    assert_eq!(span["symphony.bytes"], 15);
    assert_eq!(span["symphony.malformed_bytes"], 12);
    assert_eq!(span["symphony.finish"], "length");
}

#[test]
fn a_streamed_reading_marks_the_trace_and_its_line_after_the_access_line_went_out() {
    // A streamed response's access-log line is written when the head goes out, before the
    // body drives the reading, so it carries no field of the reading; the reading then marks
    // the span and writes its sampled line inside it, as the reader does, so that line carries
    // the request's span and the fields just marked.
    let lines = json_lines(|| {
        let span = request_span();
        span.in_scope(|| tracing::info!(target: "smg::response", "finished processing request"));
        let marked = AtomicBool::new(false);
        let mut tally = Tally::default();
        tally.see(&Event::Content(Text::uncounted("Hi ")), false);
        tally.see(
            &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
            false,
        );
        tally.see(&finished(FinishReason::Length), false);
        tally.mark(&span, &marked, "glm", 0);
        span.in_scope(|| {
            tally.report(
                "glm",
                Who {
                    request_id: "req-streamed",
                    model: "glm-5",
                    choice: 0,
                },
            );
        });
    });
    let access = &lines[0]["span"];
    assert_eq!(access["name"], "http_request");
    assert!(access.get("symphony.outcome").is_none(), "{access}");
    let line = lines
        .iter()
        .find(|line| line["span"].get("symphony.outcome").is_some())
        .expect("the output's line, written inside the span");
    assert_eq!(line["span"]["name"], "http_request");
    assert_eq!(line["span"]["symphony.outcome"], "malformed");
    assert_eq!(line["span"]["symphony.reason"], "unterminated_region");
    assert_eq!(line["span"]["symphony.malformed_bytes"], 12);
}

#[test]
fn a_span_without_the_fields_and_no_span_take_nothing() {
    // Another span's fields are its own: a mark on it records nothing and nothing panics;
    // the span of no subscriber takes nothing either.
    let lines = json_lines(|| {
        let other = tracing::info_span!("grpc_execute", request_type = "chat");
        let _entered = other.enter();
        mark_failed(&other, &AtomicBool::new(false), "qwen3", 1);
        tracing::info!("under another span");
    });
    let span = &lines[0]["span"];
    assert_eq!(span["name"], "grpc_execute");
    assert!(span.get("symphony.outcome").is_none(), "{span}");
    mark_failed(&Span::none(), &AtomicBool::new(false), "qwen3", 1);
}

#[tracing_test::traced_test]
#[test]
fn malformed_lines_take_from_a_bucket_per_reason() {
    // Forty outputs whose first run is an unterminated region drain that reason's bucket on
    // the table; an output whose first run is of another reason still writes its line, and
    // counts none of the forty's suppressed lines as its own.
    let unterminated: u32 = 40;
    rendered(|| {
        for choice in 0..unterminated {
            let mut tally = Tally::default();
            tally.see(
                &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
                false,
            );
            tally.see(&finished(FinishReason::Stop), false);
            tally.report(
                "ling",
                Who {
                    request_id: "req-unterminated-flow",
                    model: "m",
                    choice,
                },
            );
        }
        let mut tally = Tally::default();
        tally.see(
            &malformed("{\"q\": }", MalformedReason::InvalidArguments),
            false,
        );
        tally.see(&finished(FinishReason::Stop), false);
        tally.report(
            "ling",
            Who {
                request_id: "req-invalid-after-the-flow",
                model: "m",
                choice: 0,
            },
        );
    });
    logs_assert(|lines: &[&str]| {
        let flow = lines
            .iter()
            .filter(|line| line.contains("req-unterminated-flow"))
            .count();
        if flow >= unterminated as usize {
            return Err(format!(
                "no line of the flow was suppressed: {flow} written"
            ));
        }
        match lines
            .iter()
            .find(|line| line.contains("req-invalid-after-the-flow"))
        {
            Some(line) if line.contains("suppressed_lines=0") => Ok(()),
            Some(line) => Err(format!(
                "the line counts the other reason's suppressed lines: {line}"
            )),
            None => Err("the flow of one reason suppressed the line of another".to_string()),
        }
    });
}

#[test]
fn a_bucket_allows_its_burst_then_counts_what_it_refuses() {
    let sampler = Sampler::with(2.0, 0.0);
    assert_eq!(sampler.allow(), Some(0));
    assert_eq!(sampler.allow(), Some(0));
    assert_eq!(sampler.allow(), None);
    assert_eq!(sampler.allow(), None);
    assert_eq!(sampler.allow(), None);
    // Refilled by hand: the next line reports the three refused.
    sampler.state.lock().0 = 1.0;
    assert_eq!(sampler.allow(), Some(3));
    assert_eq!(sampler.allow(), None);
}

#[test]
fn a_bucket_refills_with_time() {
    let sampler = Sampler::with(1.0, 1.0e9);
    assert_eq!(sampler.allow(), Some(0));
    // A nanosecond refills a token at this rate.
    std::thread::sleep(std::time::Duration::from_millis(1));
    assert_eq!(sampler.allow(), Some(0));
}

#[test]
fn every_table_has_series_and_a_bucket_of_its_own() {
    let mut distinct = std::collections::HashSet::new();
    for table in Table::ALL {
        let own = series(table.name());
        assert_eq!(own.outputs[CLEAN].labels().count(), 2);
        assert!(own.outputs[CLEAN]
            .labels()
            .any(|label| label.key() == "table" && label.value() == table.name()));
        distinct.insert(own as *const Series);
    }
    distinct.insert(series("response_template") as *const Series);
    distinct.insert(series("other") as *const Series);
    distinct.insert(series("markers") as *const Series);
    assert_eq!(distinct.len(), Table::ALL.len() + 2);
}

#[test]
fn a_clean_output_records_one_counter_and_no_line() {
    let scrape = rendered(|| {
        let mut tally = Tally::default();
        tally.see(&Event::Content(Text::uncounted("Hello")), false);
        tally.see(&finish(0), false);
        tally.report(
            "qwen3",
            Who {
                request_id: "req-1",
                model: "m",
                choice: 0,
            },
        );
    });
    let line = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"clean\""])
        .expect("the outputs counter");
    assert!(line.ends_with(" 1"), "{line}");
    assert!(!scrape.contains(MALFORMED), "{scrape}");
    assert!(!scrape.contains(MALFORMED_BYTES), "{scrape}");
    assert!(!scrape.contains(CALLS_SUPPRESSED), "{scrape}");
}

#[test]
fn a_malformed_output_records_the_runs_by_reason_and_their_bytes() {
    let scrape = rendered(|| {
        let mut tally = Tally::default();
        tally.see(
            &malformed("\nnot a call\n", MalformedReason::InvalidArguments),
            false,
        );
        tally.see(
            &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
            false,
        );
        tally.see(&malformed("{", MalformedReason::UnterminatedRegion), false);
        tally.see(&finish(0), false);
        tally.report(
            "qwen3",
            Who {
                request_id: "req-2",
                model: "m",
                choice: 0,
            },
        );
    });
    let outputs = line_with(
        &scrape,
        OUTPUTS,
        &["table=\"qwen3\"", "outcome=\"malformed\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    let invalid = line_with(
        &scrape,
        MALFORMED,
        &["table=\"qwen3\"", "reason=\"invalid_arguments\""],
    )
    .expect("the invalid_arguments counter");
    assert!(invalid.ends_with(" 1"), "{invalid}");
    let unterminated = line_with(
        &scrape,
        MALFORMED,
        &["table=\"qwen3\"", "reason=\"unterminated_region\""],
    )
    .expect("the unterminated_region counter");
    assert!(unterminated.ends_with(" 2"), "{unterminated}");
    let bytes_sum = line_with(
        &scrape,
        &format!("{MALFORMED_BYTES}_sum"),
        &["table=\"qwen3\""],
    )
    .expect("the bytes histogram");
    assert!(bytes_sum.ends_with(" 25"), "{bytes_sum}");
    let bytes_count = line_with(
        &scrape,
        &format!("{MALFORMED_BYTES}_count"),
        &["table=\"qwen3\""],
    )
    .expect("the bytes histogram count");
    assert!(bytes_count.ends_with(" 1"), "{bytes_count}");
}

#[test]
fn suppressed_calls_are_counted_on_an_otherwise_clean_output() {
    let scrape = rendered(|| {
        let mut tally = Tally::default();
        tally.see(&call_start(0, "<tool_call>"), true);
        tally.see(&call_start(1, "<tool_call>"), true);
        tally.see(&finish(2), true);
        tally.report(
            "qwen3",
            Who {
                request_id: "req-3",
                model: "m",
                choice: 0,
            },
        );
    });
    let outputs = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"clean\""])
        .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    let suppressed =
        line_with(&scrape, CALLS_SUPPRESSED, &["table=\"qwen3\""]).expect("the suppressed counter");
    assert!(suppressed.ends_with(" 1"), "{suppressed}");
}

/// A forbidden call's bytes are dropped, not shown: the text around it decides the outcome,
/// the call is counted as forbidden and never as suppressed, begun or undeclared, and the
/// finish is the `stop` the client gets.
#[test]
fn forbidden_calls_are_counted_and_their_bytes_are_dropped_not_shown() {
    let mut seen = None;
    let scrape = rendered(|| {
        let mut tally = Tally::default();
        tally.see_forbidden(&Event::Content(Text::uncounted("Hello")));
        tally.see_forbidden(&call_start(0, "<tool_call>"));
        tally.see_forbidden(&Event::ToolCallEnd {
            index: 0,
            source: Text::uncounted("</tool_call>"),
            ended: CallEnd::Closed,
        });
        tally.see_forbidden(&malformed("", MalformedReason::UnknownTool));
        tally.see_forbidden(&call_named(1, "nope", "<tool_call>"));
        tally.see_forbidden(&Event::Finish {
            reason: FinishReason::ToolCalls,
            tool_calls: 2,
            reasoning_tokens: 0,
        });
        tally.report(
            "qwen3",
            Who {
                request_id: "req-5",
                model: "m",
                choice: 0,
            },
        );
        seen = Some((
            tally.shown,
            tally.calls_begun,
            tally.forbidden_calls,
            tally.dropped_wrapping,
            tally.tool_calls,
            tally.finish,
            tally.undeclared_calls(),
            tally.malformed(),
            tally.at,
        ));
    });
    assert_eq!(
        seen,
        Some((5, 0, 2, 34, 0, "stop", 0, 0, 39)),
        "shown, calls begun, forbidden, dropped bytes, tool_calls, finish, undeclared, malformed, at"
    );
    let outputs = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"clean\""])
        .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    let forbidden =
        line_with(&scrape, CALLS_FORBIDDEN, &["table=\"qwen3\""]).expect("the forbidden counter");
    assert!(forbidden.ends_with(" 2"), "{forbidden}");
    assert!(
        !scrape.contains(CALLS_SUPPRESSED),
        "a forbidden call is not a suppressed one: {scrape}"
    );
    assert!(
        !scrape.contains("outcome=\"unknown_tool\""),
        "a forbidden call to an undeclared function names nothing to the client: {scrape}"
    );
}

/// An output of forbidden calls alone is an empty one for the client, counted as such, and
/// the forbidden counter still says why.
#[test]
fn an_output_of_forbidden_calls_alone_is_empty_and_still_counts_them() {
    let scrape = rendered(|| {
        let mut tally = Tally::default();
        tally.see_forbidden(&call_start(0, "<tool_call>"));
        tally.see_forbidden(&call_start(1, "<tool_call>"));
        tally.see_forbidden(&finish(2));
        assert!(tally.is_empty(), "the client was shown nothing");
        tally.report(
            "qwen3",
            Who {
                request_id: "req-6",
                model: "m",
                choice: 0,
            },
        );
    });
    let outputs = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"empty\""])
        .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    let forbidden =
        line_with(&scrape, CALLS_FORBIDDEN, &["table=\"qwen3\""]).expect("the forbidden counter");
    assert!(forbidden.ends_with(" 2"), "{forbidden}");
}

#[test]
fn a_refused_output_is_counted_as_failed() {
    let scrape = rendered(|| record_failed("qwen3"));
    let line = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"failed\""])
        .expect("the outputs counter");
    assert!(line.ends_with(" 1"), "{line}");
}

#[test]
fn a_refused_message_is_counted_by_its_reason_and_not_as_an_output() {
    let scrape = rendered(|| record_refused("qwen3"));
    let line = line_with(
        &scrape,
        REFUSED,
        &["table=\"qwen3\"", "reason=\"tool_use_input\""],
    )
    .expect("the refused counter");
    assert!(line.ends_with(" 1"), "{line}");
    assert!(!scrape.contains(OUTPUTS), "{scrape}");
}

#[tracing_test::traced_test]
#[test]
fn a_malformed_output_writes_one_line_with_the_first_run() {
    let _scrape = rendered(|| {
        let mut tally = Tally::default();
        tally.see(&Event::Content(Text::uncounted("Hello ")), false);
        tally.see(
            &malformed("\nnot a call\n", MalformedReason::InvalidArguments),
            false,
        );
        tally.see(&finish(0), false);
        tally.report(
            "plain",
            Who {
                request_id: "req-line",
                model: "the-model",
                choice: 1,
            },
        );
    });
    assert!(logs_contain(
        "Symphony did not read the whole output as content, reasoning and calls"
    ));
    assert!(logs_contain("request_id=\"req-line\""));
    assert!(logs_contain("table=\"plain\""));
    assert!(logs_contain("reason=\"invalid_arguments\""));
    assert!(logs_contain("at=6"));
    assert!(logs_contain("choice=1"));
    // The run's text is model output: no line at any level carries it.
    assert!(!logs_contain("not a call"));
    assert!(!logs_contain("excerpt"));
}

#[tracing_test::traced_test]
#[test]
fn a_clean_output_writes_no_line() {
    let _scrape = rendered(|| {
        let mut tally = Tally::default();
        tally.see(&Event::Content(Text::uncounted("Hello")), false);
        tally.see(&finish(0), false);
        tally.report(
            "plain",
            Who {
                request_id: "req-quiet",
                model: "the-model",
                choice: 0,
            },
        );
    });
    assert!(!logs_contain("req-quiet"));
}

#[tracing_test::traced_test]
#[test]
fn suppressed_calls_write_no_line() {
    let _scrape = rendered(|| {
        let mut tally = Tally::default();
        tally.see(&call_start(0, "<tool_call>"), true);
        tally.see(&call_start(1, "<tool_call>"), true);
        tally.see(&finish(2), true);
        tally.report(
            "plain",
            Who {
                request_id: "req-dropped",
                model: "the-model",
                choice: 0,
            },
        );
    });
    assert!(!logs_contain("req-dropped"));
}

fn dropped(text: &str, tokens: u32, why: DropReason) -> Event<'static> {
    Event::Dropped {
        text: Text::new(text.to_owned(), tokens),
        why,
    }
}

fn finished(reason: FinishReason) -> Event<'static> {
    Event::Finish {
        reason,
        tool_calls: 0,
        reasoning_tokens: 0,
    }
}

#[tracing_test::traced_test]
#[test]
fn an_output_of_dropped_bytes_alone_is_empty_and_writes_one_line() {
    let scrape = rendered(|| {
        let mut tally = Tally::default();
        tally.see(&dropped("</think>", 1, DropReason::Wrapper), false);
        tally.see(&dropped("\n\n\n", 3, DropReason::Whitespace), false);
        tally.see(&finished(FinishReason::Length), false);
        tally.report(
            "qwen3",
            Who {
                request_id: "req-empty",
                model: "m",
                choice: 0,
            },
        );
    });
    let outputs = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"empty\""])
        .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(
        line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"clean\""])
            .is_none_or(|line| line.ends_with(" 0"))
    );
    assert!(!scrape.contains(MALFORMED), "{scrape}");
    assert!(logs_contain(
        "Symphony read an output with no text, reasoning or call: whitespace and framing alone, or nothing"
    ));
    assert!(logs_contain("req-empty"));
    assert!(logs_contain("released_bytes=0"));
    assert!(logs_contain("finish=\"length\""));
    assert!(logs_contain("bytes=11"));
    assert!(logs_contain("whitespace_bytes=3"));
    assert!(logs_contain("wrapper_bytes=8"));
    assert!(logs_contain("dropped_tokens=4"));
}

#[tracing_test::traced_test]
#[test]
fn an_output_of_no_bytes_is_empty_with_the_engines_finish() {
    let scrape = rendered(|| {
        let mut tally = Tally::default();
        tally.see(&finished(FinishReason::Stop), false);
        tally.report(
            "qwen3",
            Who {
                request_id: "req-nothing",
                model: "m",
                choice: 0,
            },
        );
    });
    let outputs = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"empty\""])
        .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(logs_contain("req-nothing"));
    assert!(logs_contain("finish=\"stop\""));
    assert!(logs_contain("bytes=0"));
}

/// The empty lines have a bucket of their own: a flow of empty outputs past the burst
/// suppresses no malformed line, whose `suppressed_lines` keeps counting malformed lines alone.
#[tracing_test::traced_test]
#[test]
fn empty_lines_take_from_a_bucket_of_their_own() {
    let empties: u32 = 40;
    rendered(|| {
        for choice in 0..empties {
            let mut tally = Tally::default();
            tally.see(&finished(FinishReason::Stop), false);
            tally.report(
                "olmo3",
                Who {
                    request_id: "req-empty-flow",
                    model: "m",
                    choice,
                },
            );
        }
        let mut tally = Tally::default();
        tally.see(
            &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
            false,
        );
        tally.see(&finished(FinishReason::Stop), false);
        tally.report(
            "olmo3",
            Who {
                request_id: "req-malformed-after-empties",
                model: "m",
                choice: 0,
            },
        );
    });
    logs_assert(|lines: &[&str]| {
        let empty_lines = lines
            .iter()
            .filter(|line| line.contains("req-empty-flow"))
            .count();
        if empty_lines >= empties as usize {
            return Err(format!(
                "no empty line was suppressed: {empty_lines} written"
            ));
        }
        match lines
            .iter()
            .find(|line| line.contains("req-malformed-after-empties"))
        {
            Some(line) if line.contains("suppressed_lines=0") => Ok(()),
            Some(line) => Err(format!(
                "the malformed line counts suppressed empty lines: {line}"
            )),
            None => Err("the empty flow suppressed the malformed line".to_string()),
        }
    });
}

#[tracing_test::traced_test]
#[test]
fn whitespace_before_text_or_a_call_is_not_an_empty_output() {
    let scrape = rendered(|| {
        let mut text = Tally::default();
        text.see(&dropped("\n", 1, DropReason::Whitespace), false);
        text.see(&Event::Content(Text::new("Hi", 1)), false);
        text.see(&finished(FinishReason::Stop), false);
        text.report(
            "qwen3",
            Who {
                request_id: "req-text",
                model: "m",
                choice: 0,
            },
        );
        let mut call = Tally::default();
        call.see(&dropped("\n", 1, DropReason::Whitespace), false);
        call.see(&call_start(0, "<tool_call>"), false);
        call.see(&finish(1), false);
        call.report(
            "qwen3",
            Who {
                request_id: "req-call",
                model: "m",
                choice: 0,
            },
        );
    });
    let outputs = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"clean\""])
        .expect("the outputs counter");
    assert!(outputs.ends_with(" 2"), "{outputs}");
    assert!(
        line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"empty\""])
            .is_none_or(|line| line.ends_with(" 0"))
    );
    assert!(!logs_contain("req-text"));
    assert!(!logs_contain("req-call"));
}

#[test]
fn a_request_the_legacy_parsers_read_is_counted_by_reason_under_none() {
    let scrape = rendered(|| {
        record_fallback(Fallback::NoTable);
        record_fallback(Fallback::NoTable);
        record_fallback(Fallback::Passthrough);
    });
    let none = line_with(
        &scrape,
        FALLBACK,
        &["reason=\"no_table\"", "table=\"none\""],
    )
    .expect("the no_table counter");
    assert!(none.ends_with(" 2"), "{none}");
    let passthrough = line_with(
        &scrape,
        FALLBACK,
        &["reason=\"passthrough\"", "table=\"none\""],
    )
    .expect("the passthrough counter");
    assert!(passthrough.ends_with(" 1"), "{passthrough}");
    assert_eq!(
        scrape
            .lines()
            .filter(|line| line.starts_with(FALLBACK))
            .count(),
        2,
        "{scrape}"
    );
    assert!(!scrape.contains(OUTPUTS), "{scrape}");
    // Every reason has a label of its own, in the spelling the dashboards read.
    let every = [
        (Fallback::NoTable, "no_table"),
        (Fallback::ReasoningNameOnly, "reasoning_name_only"),
        (Fallback::LegacyToolName, "legacy_tool_name"),
        (Fallback::Passthrough, "passthrough"),
    ];
    for (why, label) in every {
        assert_eq!(FALLBACK_REASONS[fallback_index(why)], label);
    }
    // `none` has a series of its own, apart from the template's and the hand-built formats'.
    assert!(!std::ptr::eq(series(NO_TABLE), series(OTHER_TABLE)));
    assert!(!std::ptr::eq(series(NO_TABLE), series(TEMPLATE)));
    assert_eq!(series(NO_TABLE).fallback.len(), FALLBACK_REASONS.len());
}

/// The events of a call to `name`, as the engine gives them for a function the request did
/// not declare: its event of no bytes, then the call's start.
fn undeclared_call(index: u32, name: &str) -> [Event<'static>; 2] {
    [
        malformed("", MalformedReason::UnknownTool),
        call_named(index, name, &format!("<tool_call>{name}")),
    ]
}

#[tracing_test::traced_test]
#[test]
fn a_call_to_an_undeclared_function_is_an_outcome_of_its_own_and_the_line_names_it() {
    assert_eq!(REASONS[UNKNOWN_TOOL_REASON], "unknown_tool");
    assert_eq!(
        reason_index(MalformedReason::UnknownTool),
        UNKNOWN_TOOL_REASON
    );
    assert_eq!(OUTCOMES[UNKNOWN_TOOL], "unknown_tool");
    let scrape = rendered(|| {
        let mut tally = Tally::default();
        tally.see(&Event::Content(Text::uncounted("Sure.")), false);
        for event in undeclared_call(0, "math_pow") {
            tally.see(&event, false);
        }
        tally.see(&finish(1), false);
        tally.report(
            "qwen3",
            Who {
                request_id: "req-undeclared",
                model: "m",
                choice: 0,
            },
        );
    });
    let outputs = line_with(
        &scrape,
        OUTPUTS,
        &["table=\"qwen3\"", "outcome=\"unknown_tool\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(
        line_with(
            &scrape,
            OUTPUTS,
            &["table=\"qwen3\"", "outcome=\"malformed\""]
        )
        .is_none(),
        "{scrape}"
    );
    let runs = line_with(
        &scrape,
        MALFORMED,
        &["table=\"qwen3\"", "reason=\"unknown_tool\""],
    )
    .expect("the reason counter");
    assert!(runs.ends_with(" 1"), "{runs}");
    assert!(
        !scrape.contains(MALFORMED_BYTES),
        "no refused bytes to observe: {scrape}"
    );
    assert!(logs_contain(
        "Symphony read a call to a function the request did not declare; the call went out as the model named it"
    ));
    assert!(logs_contain("outcome=\"unknown_tool\""));
    assert!(logs_contain("table=\"qwen3\""));
    assert!(logs_contain("tool=\"math_pow\""));
    assert!(logs_contain("tool_bytes=8"));
    assert!(logs_contain("undeclared_calls=1"));
    assert!(logs_contain("at=5"));
    assert!(logs_contain("tool_calls=1"));
    assert!(logs_contain("request_id=\"req-undeclared\""));
    assert!(!logs_contain("did not read the whole output"));
}

#[tracing_test::traced_test]
#[test]
fn every_undeclared_call_is_a_run_and_the_output_counts_once_under_the_first_name() {
    let scrape = rendered(|| {
        let mut tally = Tally::default();
        for event in undeclared_call(0, "lookup_v2") {
            tally.see(&event, false);
        }
        tally.see(&call_start(1, "<tool_call>lookup"), false);
        for event in undeclared_call(2, "look_up") {
            tally.see(&event, false);
        }
        tally.see(&finish(3), false);
        tally.report(
            "glm",
            Who {
                request_id: "req-undeclared-twice",
                model: "m",
                choice: 0,
            },
        );
    });
    let outputs = line_with(
        &scrape,
        OUTPUTS,
        &["table=\"glm\"", "outcome=\"unknown_tool\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    let runs = line_with(
        &scrape,
        MALFORMED,
        &["table=\"glm\"", "reason=\"unknown_tool\""],
    )
    .expect("the reason counter");
    assert!(runs.ends_with(" 2"), "{runs}");
    assert!(logs_contain("undeclared_calls=2"));
    assert!(logs_contain("tool=\"lookup_v2\""));
    assert!(!logs_contain("tool=\"look_up\""));
    assert!(logs_contain("tool_calls=3"));
    assert!(logs_contain("at=0"));
}

#[tracing_test::traced_test]
#[test]
fn an_undeclared_call_beside_a_refused_run_leaves_the_output_malformed() {
    let scrape = rendered(|| {
        let mut tally = Tally::default();
        for event in undeclared_call(0, "lookup_v2") {
            tally.see(&event, false);
        }
        tally.see(
            &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
            false,
        );
        tally.see(&finish(1), false);
        tally.report(
            "qwen3",
            Who {
                request_id: "req-undeclared-and-refused",
                model: "m",
                choice: 0,
            },
        );
    });
    let outputs = line_with(
        &scrape,
        OUTPUTS,
        &["table=\"qwen3\"", "outcome=\"malformed\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(
        line_with(
            &scrape,
            OUTPUTS,
            &["table=\"qwen3\"", "outcome=\"unknown_tool\""]
        )
        .is_none(),
        "{scrape}"
    );
    for reason in ["unknown_tool", "unterminated_region"] {
        let runs = line_with(
            &scrape,
            MALFORMED,
            &["table=\"qwen3\"", &format!("reason=\"{reason}\"")],
        )
        .expect("the reason counter");
        assert!(runs.ends_with(" 1"), "{runs}");
    }
    let count = line_with(&scrape, &format!("{MALFORMED_BYTES}_count"), &[])
        .expect("the refused bytes observed");
    assert!(count.ends_with(" 1"), "{count}");
    assert!(logs_contain(
        "Symphony did not read the whole output as content, reasoning and calls"
    ));
    // The line's first run is the refused one, after the undeclared call's 20 bytes; the
    // event of no bytes before that call is not a run the line locates.
    assert!(logs_contain("reason=\"unterminated_region\""));
    assert!(logs_contain("at=20 bytes=12"));
    assert!(!logs_contain("reason=\"unknown_tool\""));
    assert!(logs_contain("reasons=unterminated_region,unknown_tool"));
    assert!(!logs_contain(
        "Symphony read a call to a function the request did not declare"
    ));
}

#[tracing_test::traced_test]
#[test]
fn a_long_undeclared_name_is_cut_on_the_line_with_its_whole_length_beside_it() {
    // Forty two-byte characters: the cut falls on a character boundary, below the bound.
    let name = "ü".repeat(40);
    rendered(|| {
        let mut tally = Tally::default();
        for event in undeclared_call(0, &name) {
            tally.see(&event, false);
        }
        tally.see(&finish(1), false);
        tally.report(
            "qwen3",
            Who {
                request_id: "req-undeclared-long",
                model: "m",
                choice: 0,
            },
        );
    });
    assert!(logs_contain(&format!("tool=\"{}\"", "ü".repeat(32))));
    assert!(logs_contain("tool_bytes=80"));
}

#[tracing_test::traced_test]
#[test]
fn unknown_tool_lines_take_from_a_bucket_of_their_own() {
    let flow: u32 = 40;
    rendered(|| {
        for choice in 0..flow {
            let mut tally = Tally::default();
            for event in undeclared_call(0, "lookup_v2") {
                tally.see(&event, false);
            }
            tally.see(&finish(1), false);
            tally.report(
                "hy4",
                Who {
                    request_id: "req-undeclared-flow",
                    model: "m",
                    choice,
                },
            );
        }
        let mut tally = Tally::default();
        tally.see(
            &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
            false,
        );
        tally.see(&finished(FinishReason::Stop), false);
        tally.report(
            "hy4",
            Who {
                request_id: "req-malformed-after-undeclared",
                model: "m",
                choice: 0,
            },
        );
    });
    logs_assert(|lines: &[&str]| {
        let written = lines
            .iter()
            .filter(|line| line.contains("req-undeclared-flow"))
            .count();
        if written >= flow as usize {
            return Err(format!(
                "no undeclared-call line was suppressed: {written} written"
            ));
        }
        match lines
            .iter()
            .find(|line| line.contains("req-malformed-after-undeclared"))
        {
            Some(line) if line.contains("suppressed_lines=0") => Ok(()),
            Some(line) => Err(format!(
                "the malformed line counts suppressed undeclared-call lines: {line}"
            )),
            None => Err("the undeclared-call flow suppressed the malformed line".to_string()),
        }
    });
}

#[tracing_test::traced_test]
#[test]
fn a_scrape_takes_its_lines_from_series_of_its_own() {
    // One scrape drains a table's bucket for a reason; the next scrape's first line of that
    // reason, on the same table, is written and counts no suppressed line: the series are
    // the scrape's own, not the binary's, so a flow in one test starves no other.
    let flow: u32 = 40;
    rendered(|| {
        for choice in 0..flow {
            let mut tally = Tally::default();
            tally.see(
                &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
                false,
            );
            tally.see(&finished(FinishReason::Stop), false);
            tally.report(
                "qwen3",
                Who {
                    request_id: "req-flow-of-another-scrape",
                    model: "m",
                    choice,
                },
            );
        }
    });
    rendered(|| {
        let mut tally = Tally::default();
        tally.see(
            &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
            false,
        );
        tally.see(&finished(FinishReason::Stop), false);
        tally.report(
            "qwen3",
            Who {
                request_id: "req-after-another-scrape",
                model: "m",
                choice: 0,
            },
        );
    });
    logs_assert(|lines: &[&str]| {
        let written = lines
            .iter()
            .filter(|line| line.contains("req-flow-of-another-scrape"))
            .count();
        if written >= flow as usize {
            return Err(format!(
                "no line of the flow was suppressed: {written} written"
            ));
        }
        match lines
            .iter()
            .find(|line| line.contains("req-after-another-scrape"))
        {
            Some(line) if line.contains("suppressed_lines=0") => Ok(()),
            Some(line) => Err(format!(
                "the line counts the other scrape's suppressed lines: {line}"
            )),
            None => Err("the other scrape's flow suppressed the line".to_string()),
        }
    });
}

#[test]
fn the_own_series_fold_an_unregistered_name_as_the_shared_ones_do() {
    // Under the guard a name that is no table's reads the own series of the rest, labelled
    // as the shared path labels it: a test sees the label production writes.
    let _own = own_series();
    let rest = series(OTHER_TABLE);
    assert!(std::ptr::eq(series("markers"), rest));
    assert!(std::ptr::eq(series("made-up"), rest));
    assert!(!std::ptr::eq(series(Table::ALL[0].name()), rest));
    assert!(rest.outputs[CLEAN]
        .labels()
        .any(|label| label.key() == "table" && label.value() == OTHER_TABLE));
}

#[test]
fn json_lines_take_their_lines_from_series_of_their_own() {
    // The isolation `a_scrape_takes_its_lines_from_series_of_its_own` pins, through the
    // log-line helper: a flow in one call's closure drains a table's bucket, the next call's
    // one line is written and counts no suppressed line.
    let flow: u32 = 40;
    let drained = json_lines(|| {
        for choice in 0..flow {
            let mut tally = Tally::default();
            tally.see(
                &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
                false,
            );
            tally.see(&finished(FinishReason::Stop), false);
            tally.report(
                "qwen3",
                Who {
                    request_id: "req-flow-of-another-call",
                    model: "m",
                    choice,
                },
            );
        }
    });
    assert!(
        drained.len() < flow as usize,
        "no line of the flow was suppressed"
    );
    let after = json_lines(|| {
        let mut tally = Tally::default();
        tally.see(
            &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
            false,
        );
        tally.see(&finished(FinishReason::Stop), false);
        tally.report(
            "qwen3",
            Who {
                request_id: "req-after-another-call",
                model: "m",
                choice: 0,
            },
        );
    });
    let line = after
        .iter()
        .find(|line| line.to_string().contains("req-after-another-call"))
        .expect("the line after the other call's flow");
    assert_eq!(line["suppressed_lines"], 0, "{line}");
}

/// A table's failure classes exist at zero from the moment the table is published, so an
/// alert on `failed` or `malformed` reads 0 and not "no data" until the first such output,
/// and the first output counts on the published series.
#[test]
fn a_published_table_has_every_failure_class_at_zero_and_counts_on_it() {
    let scrape = rendered(|| {
        init_table_series("qwen3");
        record_failed("qwen3");
    });
    for outcome in OUTCOMES {
        let line = line_with(
            &scrape,
            OUTPUTS,
            &["table=\"qwen3\"", &format!("outcome=\"{outcome}\"")],
        )
        .expect(outcome);
        let expected = if outcome == "failed" { " 1" } else { " 0" };
        assert!(line.ends_with(expected), "{line}");
    }
    let reasons = scrape
        .lines()
        .filter(|line| line.starts_with(MALFORMED) && line.contains("table=\"qwen3\""))
        .count();
    assert_eq!(reasons, REASONS.len(), "{scrape}");
    for reason in REASONS {
        let line = line_with(
            &scrape,
            MALFORMED,
            &["table=\"qwen3\"", &format!("reason=\"{reason}\"")],
        )
        .expect(reason);
        assert!(line.ends_with(" 0"), "{line}");
    }
    let zero = |name: &str, labels: &[&str]| {
        line_with(&scrape, name, labels).is_some_and(|line| line.ends_with(" 0"))
    };
    assert!(zero(CALLS_SUPPRESSED, &["table=\"qwen3\""]), "{scrape}");
    assert!(zero(CALLS_FORBIDDEN, &["table=\"qwen3\""]), "{scrape}");
    assert!(
        zero(REFUSED, &["table=\"qwen3\"", "reason=\"tool_use_input\""]),
        "{scrape}"
    );
    // The fallback reasons count under `none`, never under a table.
    for reason in &FALLBACK_REASONS {
        assert!(
            line_with(
                &scrape,
                FALLBACK,
                &["table=\"qwen3\"", &format!("reason=\"{reason}\"")]
            )
            .is_none(),
            "{reason}"
        );
    }
    // The histogram family is published with an empty distribution.
    assert!(
        zero(&format!("{MALFORMED_BYTES}_count"), &["table=\"qwen3\""]),
        "{scrape}"
    );
    // Nothing of another table.
    assert!(!scrape.contains("table=\"mistral\""), "{scrape}");
}

/// The requests no table reads are counted under `none` by four reasons known at start-up:
/// their series exist at zero before any worker registers.
#[test]
fn the_startup_series_are_the_four_fallbacks_of_a_request_no_table_reads() {
    let scrape = rendered(init_startup_series);
    assert_eq!(FALLBACK_REASONS.len(), 4);
    for reason in &FALLBACK_REASONS {
        let line = line_with(
            &scrape,
            FALLBACK,
            &["table=\"none\"", &format!("reason=\"{reason}\"")],
        )
        .expect(reason);
        assert!(line.ends_with(" 0"), "{line}");
    }
    assert!(!scrape.contains(OUTPUTS), "{scrape}");
}

/// Records through kept handles reach the recorder's series: two failed outputs and one
/// refused response under one table render with their counts (an unknown table name would
/// fold into `other`, so the test takes a registered one).
#[test]
fn kept_counter_handles_update_the_recorders_series() {
    let table = Table::ALL[0].name();
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        force_on_this_thread(Some(true));
        record_failed(table);
        record_failed(table);
        record_refused(table);
        force_on_this_thread(None);
    });
    assert_eq!(KEPT_COUNTERS.with_borrow(|kept| kept.len()), 2);
    reset_kept();
    let rendered = handle.render();
    assert!(
        rendered.contains(&format!(
            "smg_symphony_outputs_total{{table=\"{table}\",outcome=\"failed\"}} 2"
        )),
        "{rendered}"
    );
    assert!(
        rendered.contains(&format!(
            "smg_symphony_refused_total{{table=\"{table}\",reason=\"tool_use_input\"}} 1"
        )),
        "{rendered}"
    );
}
