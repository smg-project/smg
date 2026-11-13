//! The parse endpoints: a model's output text read whole through the Symphony table the request
//! names (`routers::grpc::regular::symphony`), for its tool calls (`/parse/function_call`) or its
//! reasoning (`/parse/reasoning`). A table answers to its own name and to the older parser names
//! the gateway maps onto it (`routers::grpc::utils::ALIASES`; `passthrough` is `plain`); any other
//! name is a 400 that says why an older name is not read with and lists the names that are. The
//! text is read as one delta after the prompt's tail: empty, so the text reads as a model's output
//! would from its first token, except under a reasoning name whose family's template opens the
//! thought in the prompt (`OPENED_IN_THE_PROMPT`), where the tail is that opener.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::{
    chat::ChatCompletionMessage,
    common::Tool,
    parser::{ParseFunctionCallRequest, SeparateReasoningRequest},
};
use symphony::{adapt, formats::Table, Declared, Engine, EngineFinish, Events, Input, Parser};
use tracing::error;

use crate::routers::grpc::utils::{answering_to, ALIASES, UNANSWERED};

/// Helper to create error responses
fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": message,
            "success": false
        })),
    )
        .into_response()
}

/// The older reasoning parsers that started inside the thought, by name, each with the markers
/// its thought opens and closes on. Their families' templates write the opener at the prompt's
/// end, so the model's output begins inside the thought and closes it without having opened it.
/// Serving replays the real prompt and the table starts where it ends; this endpoint has no
/// prompt, so it feeds the opener as the prompt's tail ([`prompt_tail`]). The older `step3`
/// parser started there too; no table reads Step 3 (`UNANSWERED`), so the endpoint refuses the
/// name with the reason. No older tool-call parser started inside a call: each found its markers
/// in the text, so the call parse feeds no tail.
const OPENED_IN_THE_PROMPT: [(&str, &str, &str); 4] = [
    ("deepseek_r1", "<think>", "</think>"),
    ("kimi_thinking", "<think>", "</think>"),
    ("minimax", "<think>", "</think>"),
    ("qwen3_thinking", "<think>", "</think>"),
];

/// The table that reads `name`'s thought at the reasoning endpoint: the one that answers to the
/// name, and the `qwen3` table for `qwen3_thinking`. No table answers to that name where calls
/// are read, since two tables' families share its thought (`UNANSWERED`); this endpoint reads
/// no calls, and the thought is the same in both.
fn thought_table(name: &str) -> Option<Table> {
    answering_to(name).or_else(|| {
        name.trim()
            .eq_ignore_ascii_case("qwen3_thinking")
            .then_some(Table::Qwen3)
    })
}

/// The prompt's tail the reasoning endpoint feeds ahead of `text` under `name`: the thought's
/// opener for a name in [`OPENED_IN_THE_PROMPT`], so that a text that closes a thought it never
/// opened, or closes none, reads from inside the thought as the older parser read it. Empty for
/// a text that opens a thought itself before it closes one, which reads from content as written,
/// and for every other name.
fn prompt_tail(name: &str, text: &str) -> &'static str {
    let name = name.trim().to_ascii_lowercase();
    let Some((_, open, close)) = OPENED_IN_THE_PROMPT
        .iter()
        .find(|(inside, ..)| *inside == name)
    else {
        return "";
    };
    let opens_itself = match (text.find(open), text.find(close)) {
        (Some(opened), Some(closed)) => opened < closed,
        (Some(_), None) => true,
        (None, _) => false,
    };
    if opens_itself {
        ""
    } else {
        open
    }
}

/// The 400 for a `kind` parser name no table answers to: the reason the older parser of that
/// name is not read with, when it was one (`UNANSWERED`), then the names that are: the tables
/// and the older names they answer to.
fn unknown_parser(kind: &str, name: &str) -> String {
    let spelled = name.trim().to_ascii_lowercase();
    let older = UNANSWERED
        .iter()
        .find(|(older, _)| *older == spelled)
        .map_or_else(String::new, |(_, reason)| format!(" ({reason})"));
    let tables: Vec<&str> = Table::ALL.iter().map(|table| table.name()).collect();
    let aliases: Vec<&str> = ALIASES.iter().map(|(alias, _)| *alias).collect();
    format!(
        "Unknown {kind} parser: {name}{older}; the tables are {}, and the older names they \
         answer to are {}",
        tables.join(", "),
        aliases.join(", ")
    )
}

/// `text` read whole with the table `name` selects, over the declared `tools`, as the message its
/// events fold into: content, reasoning and calls. Without `calls` the table's variant that reads
/// no calls reads, so a call's text stays content (the reasoning endpoint separates the thought
/// alone, as before), from where the family's prompt leaves the model ([`prompt_tail`]). A name
/// no table answers to is the 400 ([`unknown_parser`]); a table that refuses the three lifecycle
/// steps, which are in order, is a 500.
fn read_whole(
    kind: &str,
    name: &str,
    text: &str,
    tools: Vec<Tool>,
    calls: bool,
) -> Result<ChatCompletionMessage, Response> {
    let table = if calls {
        answering_to(name)
    } else {
        thought_table(name)
    };
    let Some(table) = table else {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            &unknown_parser(kind, name),
        ));
    };
    // The call parse reads from an empty prompt; the reasoning read from the prompt's tail.
    let prompt = if calls { "" } else { prompt_tail(name, text) };
    // The call parse reads no thought: the text beside the calls, a thought's markers
    // included, comes back as written, as the older parser left it.
    let mut engine = if calls {
        Engine::new(
            (*table.shared()).clone().without_reasoning(),
            Declared::shared(tools.into()),
        )
    } else {
        Engine::new(
            (*table.shared()).clone().without_calls(),
            Declared::default(),
        )
    };
    let mut events = Events::new();
    for step in [
        Input::Prompt {
            token_ids: &[],
            text: prompt,
        },
        Input::Delta {
            token_ids: &[],
            text,
            spans: &[],
        },
        Input::End {
            finish: EngineFinish::Stop,
        },
    ] {
        engine.feed(step, &mut events).map_err(|e| {
            error!(table = table.name(), error = %e, "the table refused the text");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!(
                    "Failed to read the text with the {} table: {e}",
                    table.name()
                ),
            )
        })?;
    }
    Ok(adapt::chat::message(0, events.as_slice()).message)
}

/// The tool calls in `text` as the table named by `tool_call_parser` reads them, and the content
/// left beside them, a thought's markers included: the call parse reads no thought.
pub fn parse_function_call(req: &ParseFunctionCallRequest) -> Response {
    match read_whole(
        "tool",
        &req.tool_call_parser,
        &req.text,
        req.tools.clone(),
        true,
    ) {
        Ok(message) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "remaining_text": message.content.unwrap_or_default(),
                "tool_calls": message.tool_calls.unwrap_or_default(),
                "success": true
            })),
        )
            .into_response(),
        Err(response) => response,
    }
}

/// The reasoning in `text` as the table named by `reasoning_parser` reads it, and the text left
/// beside it, a call's text included.
pub fn parse_reasoning(req: &SeparateReasoningRequest) -> Response {
    match read_whole(
        "reasoning",
        &req.reasoning_parser,
        &req.text,
        Vec::new(),
        false,
    ) {
        Ok(message) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "normal_text": message.content.unwrap_or_default(),
                "reasoning_text": message.reasoning_content.unwrap_or_default(),
                "success": true
            })),
        )
            .into_response(),
        Err(response) => response,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reasoning and the text beside it, as the reasoning endpoint reads `text` under `name`.
    fn separated(name: &str, text: &str) -> (String, String) {
        let Ok(message) = read_whole("reasoning", name, text, Vec::new(), false) else {
            panic!("no table answers to {name}");
        };
        (
            message.reasoning_content.unwrap_or_default(),
            message.content.unwrap_or_default(),
        )
    }

    /// A name whose older reasoning parser started inside the thought (its template opens the
    /// thought in the prompt) reads a text that closes a thought it never opened as that thought
    /// and the answer after it, as that parser did.
    #[test]
    fn a_name_whose_older_parser_started_inside_the_thought_separates_an_unopened_thought() {
        let wrong: Vec<_> = ["deepseek_r1", "minimax", "kimi_thinking", "qwen3_thinking"]
            .into_iter()
            .map(|name| (name, separated(name, "Let me think.</think>4")))
            .filter(|(_, read)| *read != ("Let me think.".to_string(), "4".to_string()))
            .collect();
        assert!(wrong.is_empty(), "read as (reasoning, text): {wrong:?}");
    }

    /// Under those names a text that opens its thought itself reads from content, as written, and
    /// a text with no marker is a thought the end cut short, as the older parser read it; a name
    /// whose older parser started outside the thought feeds no tail.
    #[test]
    fn a_text_that_opens_its_own_thought_reads_from_content_and_one_with_no_marker_is_a_thought() {
        let read = |text: &str| separated("deepseek_r1", text);
        assert_eq!(
            read("<think>plan</think>4"),
            ("plan".to_string(), "4".to_string())
        );
        assert_eq!(
            read("still thinking"),
            ("still thinking".to_string(), String::new())
        );
        assert_eq!(prompt_tail("DeepSeek_R1 ", "plan</think>4"), "<think>");
        assert_eq!(prompt_tail("deepseek_r1", "a<think>b</think>c"), "");
        assert_eq!(prompt_tail("deepseek_r1", "a</think>b<think>c"), "<think>");
        assert_eq!(prompt_tail("qwen3", "plan</think>4"), "");
        assert_eq!(prompt_tail("deepseek_v4", "plan</think>4"), "");
    }

    /// Every name in the table has a table that reads its thought, which its opener opens.
    /// `qwen3_thinking` has one at this endpoint alone: the call parse refuses the name. The
    /// older parsers no table reads, Step 3's, which started inside the thought, among them,
    /// are refused with the reason, beside the names the endpoint takes.
    #[test]
    fn every_name_opened_in_the_prompt_has_a_table_its_opener_opens_the_thought_of() {
        for (name, open, _) in OPENED_IN_THE_PROMPT {
            let table = thought_table(name).unwrap_or_else(|| panic!("no table reads {name}"));
            assert!(
                table.shared().reasoning_open(open),
                "{name}: {open} leaves no thought open"
            );
        }
        assert_eq!(thought_table(" Qwen3_Thinking"), Some(Table::Qwen3));
        assert!(read_whole("tool", "qwen3_thinking", "", Vec::new(), true).is_err());
        for name in ["step3", "glm45"] {
            assert!(thought_table(name).is_none(), "a table reads {name}");
            let refusal = unknown_parser("reasoning", name);
            let (_, reason) = UNANSWERED
                .iter()
                .find(|(older, _)| *older == name)
                .unwrap_or_else(|| panic!("{name} has no reason"));
            assert!(
                refusal.contains(reason)
                    && refusal.contains("qwen3")
                    && refusal.contains("minimax"),
                "{refusal}"
            );
        }
    }
}
