//! DeepSeek V3: the calls as one block, `<｜tool▁calls▁begin｜>` to `<｜tool▁calls▁end｜>`, holding
//! one call after another, each as `<｜tool▁call▁begin｜>function<｜tool▁sep｜>`, the tool's name,
//! a `json` code fence around the arguments object, and `<｜tool▁call▁end｜>`, the calls after the
//! first on a new line. Content comes before the block. As the checkpoint's template renders it
//! for `deepseek-ai/DeepSeek-V3-0324`:
//!
//! ```text
//! <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
//! ```json
//! {"city": "Paris"}
//! ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
//! ```
//!
//! The spelling is the gateway's older `deepseek` parser's; DeepSeek V3.1 drops the type and the
//! fence ([`deepseek_v3_1`]). V3's template writes no thought; the table reads one between
//! `<think>` and `</think>` all the same, since the checkpoints that share the call spelling and
//! write a thought read with it: the DeepSeek R1 line (R1, R1-Zero, the bf16 mirror, the six
//! distills, R1-0528 and R1-0528-Qwen3-8B, by id, and the gateway's older pairing `deepseek` +
//! `deepseek_r1` under the `deepseek` alias), whose templates render V3's block from the history
//! and either open the thought in the prompt (`<think>` after the assistant header: R1, R1-Zero,
//! the distills) or leave the model to open it (R1-0528, R1-0528-Qwen3-8B, the bf16 mirror); a
//! thought must never land in the content.
//! Recorded as `deepseek-v3-0324`, whose parse sets hold content alone: the template takes a
//! call's arguments as a string, so the recorder, whose corpus holds them as an object, refused
//! every call case; the calls stand on the turns the template renders, held in the tests below.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.
//!
//! [`deepseek_v3_1`]: super::deepseek_v3_1()

use crate::format::{CallSyntax, Emits, Format};

/// The text between a call's name and its arguments: the fence the template opens.
pub const FENCE_OPEN: &str = "\n```json\n";

/// The DeepSeek V3 table. The tool outputs are a state of their own, entered from the calls block
/// or from content on `<｜tool▁outputs▁begin｜>`, or from content or the thought on
/// `<｜tool▁output▁begin｜>` (R1's, R1-Zero's, the bf16 mirror's and the distills' templates write
/// the outer opener, like `<｜Assistant｜>`, for the conversation's first tool turn only, so a later
/// turn's text is walked in content, where a quoted `<think>` opens the thought), and left for
/// content on `<｜tool▁outputs▁end｜>`, with no other row: the
/// template writes the turn opener after a user message only, so after a
/// tool result the prompt's replay walks the call turn, and the R1 distills' template ends a turn
/// with one call with neither `<｜tool▁calls▁end｜>` nor `<｜end▁of▁sentence｜>`, so the opener is
/// what leaves the block there; nothing quoted inside a tool's output moves the walk; and the
/// close starts the answer with nothing begun, so whitespace alone ahead of its thought or call
/// is held, and the model's own `<think>`, or one the client prefilled, opens the thought from
/// content as it does after `<｜Assistant｜>`.
pub fn deepseek_v3() -> Format {
    Format::new("deepseek_v3")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("calls_open", "<｜tool▁calls▁begin｜>")
        .terminal("calls_close", "<｜tool▁calls▁end｜>")
        .terminal("call_open", "<｜tool▁call▁begin｜>function<｜tool▁sep｜>")
        .terminal("call_close", "```<｜tool▁call▁end｜>")
        .terminal("outputs_open", "<｜tool▁outputs▁begin｜>")
        .terminal("output_open", "<｜tool▁output▁begin｜>")
        .terminal("outputs_close", "<｜tool▁outputs▁end｜>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Wrapper)
        .state("call", Emits::Arguments)
        .state("outputs", Emits::Content)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "calls_open", "calls")
        .transition("reasoning", "calls_open", "calls")
        .transition("calls", "call_open", "call")
        .transition("call", "call_close", "calls")
        .transition("call", "call_open", "call")
        .transition("call", "calls_close", "content")
        .transition("calls", "calls_close", "content")
        .transition("calls", "outputs_open", "outputs")
        .transition("content", "outputs_open", "outputs")
        .transition("content", "output_open", "outputs")
        .transition("reasoning", "output_open", "outputs")
        .transition("outputs", "outputs_close", "content")
        .calls(CallSyntax::NamedArguments {
            marker: FENCE_OPEN,
            between_calls: "\n",
        })
        .opens_turn("<｜Assistant｜>")
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, FinishReason},
        grammar::{GrammarStart, Options},
        input::{EngineFinish, Input},
        parser::Parser,
        testing::{bytes, calls, content, finish, reasoning},
        Declared,
    };

    /// The generation prompt's tail: the turn's opener and nothing more.
    const PROMPT: &str = "<｜Assistant｜>";
    /// The tail of a checkpoint whose template opens the thought in the prompt: DeepSeek R1's,
    /// R1-Zero's and the R1 distills' write `<think>` and a newline after the assistant header.
    /// R1-0528's, R1-0528-Qwen3-8B's and the bf16 mirror's end at the header, and the model writes
    /// `<think>` itself.
    const THINKING: &str = "<｜Assistant｜><think>\n";

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        let mut parser = Engine::new(deepseek_v3(), Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                &mut out,
            )
            .expect("prompt");
        for piece in pieces {
            parser
                .feed(
                    Input::Delta {
                        token_ids: &[],
                        text: piece,
                        spans: &[],
                    },
                    &mut out,
                )
                .expect("delta");
        }
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        out.into_owned().drain()
    }

    fn malformed<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// A turn as the checkpoint's chat template renders it (`chat_template` of
    /// `deepseek-ai/DeepSeek-V3-0324`, the arguments given as the strings the template
    /// concatenates): the prompt's tail, the turn, and the message the template rendered.
    struct RenderedTurn {
        tail: &'static str,
        output: &'static str,
        reasoning: &'static str,
        content: &'static str,
        calls: &'static [(&'static str, &'static str)],
    }

    /// The turns the template renders for the shapes a tool-calling conversation takes: one call,
    /// three calls in one block (a newline before each call after the first), a call after
    /// content, a call with no arguments, nested and non-ASCII values, and content alone; then a
    /// call and content after a thought the prompt opens (R1, R1-Zero and the distills, whose
    /// template writes `<think>` and a newline after the header), and after a thought the model
    /// opens itself, the prompt ending at the header (R1-0528, R1-0528-Qwen3-8B and the bf16 mirror).
    const RENDERED_TURNS: &[RenderedTurn] = &[
        RenderedTurn {
            tail: PROMPT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[("get_weather", "{\"city\": \"Paris\"}")],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Rome\"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_stock_price\n```json\n{\"ticker\": \"MSFT\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[
                ("get_weather", "{\"city\": \"Paris\"}"),
                ("get_weather", "{\"city\": \"Rome\"}"),
                ("get_stock_price", "{\"ticker\": \"MSFT\"}"),
            ],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "Let me check.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\", \"unit\": \"celsius\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "Let me check.",
            calls: &[("get_weather", "{\"city\": \"Paris\", \"unit\": \"celsius\"}")],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_stock_price\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[("get_stock_price", "{}")],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>save_note\n```json\n{\"title\": \"Plan\", \"body\": \"Line one\\nLine two with \\\"quotes\\\" and a backslash \\\\\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[(
                "save_note",
                "{\"title\": \"Plan\", \"body\": \"Line one\\nLine two with \\\"quotes\\\" and a backslash \\\\\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}",
            )],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>save_note\n```json\n{\"title\": \"計画 🌍\", \"body\": \"café\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[("save_note", "{\"title\": \"計画 🌍\", \"body\": \"café\"}")],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "It is sunny in Paris.",
            reasoning: "",
            content: "It is sunny in Paris.",
            calls: &[],
        },
        RenderedTurn {
            tail: THINKING,
            output: "The user asks about Paris.</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "The user asks about Paris.",
            content: "",
            calls: &[("get_weather", "{\"city\": \"Paris\"}")],
        },
        RenderedTurn {
            tail: THINKING,
            output: "97 is not divisible by 2, 3, 5 or 7.</think>Yes, 97 is prime.",
            reasoning: "97 is not divisible by 2, 3, 5 or 7.",
            content: "Yes, 97 is prime.",
            calls: &[],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "<think>\nThe user asks about Paris.\n</think>\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "\nThe user asks about Paris.\n",
            content: "",
            calls: &[("get_weather", "{\"city\": \"Paris\"}")],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "<think>\n97 is not divisible by 2, 3, 5 or 7.\n</think>\n\nYes, 97 is prime.",
            reasoning: "\n97 is not divisible by 2, 3, 5 or 7.\n",
            content: "\n\nYes, 97 is prime.",
            calls: &[],
        },
    ];

    fn check(turn: &RenderedTurn, events: &[Event], how: &str) {
        assert_eq!(bytes(events), turn.output, "{how}: {:?}", turn.output);
        assert_eq!(
            reasoning(events),
            turn.reasoning,
            "{how}: {:?}",
            turn.output
        );
        assert_eq!(content(events), turn.content, "{how}: {:?}", turn.output);
        let expected: Vec<(String, String)> = turn
            .calls
            .iter()
            .map(|(name, arguments)| ((*name).to_string(), (*arguments).to_string()))
            .collect();
        assert_eq!(calls(events), expected, "{how}: {:?}", turn.output);
        assert_eq!(
            malformed(events),
            Vec::<&str>::new(),
            "{how}: {:?}",
            turn.output
        );
        let count = u32::try_from(turn.calls.len()).expect("a few calls");
        assert_eq!(
            finish(events),
            (FinishReason::Stop, count),
            "{how}: {:?}",
            turn.output
        );
    }

    #[test]
    fn every_turn_the_template_renders_reads_back_as_the_message_it_rendered() {
        for turn in RENDERED_TURNS {
            let events = run(turn.tail, &[turn.output]);
            check(turn, &events, "whole");
        }
    }

    #[test]
    fn every_turn_reads_the_same_at_every_cut() {
        for turn in RENDERED_TURNS {
            for (at, _) in turn.output.char_indices().skip(1) {
                let (head, rest) = turn.output.split_at(at);
                let events = run(turn.tail, &[head, rest]);
                check(turn, &events, &format!("cut at {at}"));
            }
            let pieces: Vec<&str> = turn
                .output
                .char_indices()
                .map(|(at, c)| &turn.output[at..at + c.len_utf8()])
                .collect();
            let events = run(turn.tail, &pieces);
            check(turn, &events, "character by character");
        }
    }

    #[test]
    fn the_markers_and_the_fence_are_wrapping_and_the_arguments_are_the_models_bytes() {
        let events = run(PROMPT, &[RENDERED_TURNS[1].output]);
        for marker in [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>",
            "```<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ] {
            assert!(
                events.iter().any(|event| matches!(
                    event,
                    Event::Dropped { text, why: DropReason::Wrapper } if text.text == marker
                )),
                "{marker} dropped as wrapping"
            );
        }
        // The newline before the fence's close and the one between two calls are the template's.
        assert!(
            events.iter().all(|event| !matches!(
                event,
                Event::Content(text) if text.text.trim().is_empty() && !text.text.is_empty()
            )),
            "no whitespace reaches the content between the calls"
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallStart { index: 2, name, .. } if name == "get_stock_price"
        )));
    }

    #[test]
    fn content_keeps_the_fenced_block_and_a_thought_the_model_writes_is_read() {
        // The two probes: the fenced `<tool_call>` block is no marker of this table, so it stays
        // content; `<think>` and `</think>` are, though V3's template never writes them, so a
        // thought the model writes is reasoning and a `</think>` quoted inside it ends it (the
        // engine's rule for a marker inside a channel, bellwether #16).
        let fence =
            "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
                     {\"city\": \"Paris\"}}\n</tool_call>\n```\n\nThat is all.";
        let events = run(PROMPT, &[fence]);
        assert_eq!(content(&events), fence);
        assert!(calls(&events).is_empty());
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
        let events = run(
            PROMPT,
            &["<think>I could write </think> here.</think>Paris."],
        );
        assert_eq!(reasoning(&events), "I could write ");
        assert_eq!(content(&events), " here.</think>Paris.");
        // A thought the prompt opened and the model closed at once leaves the output in content,
        // as the older pairing of the `deepseek` and `deepseek_r1` parsers read it.
        let events = run(THINKING, &["</think>Paris is sunny."]);
        assert_eq!(reasoning(&events), "");
        assert_eq!(content(&events), "Paris is sunny.");
    }

    #[test]
    fn a_stream_cut_inside_a_call_ends_it_with_what_arrived() {
        let mut parser = Engine::new(deepseek_v3(), Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: PROMPT,
                },
                &mut out,
            )
            .expect("prompt");
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Par",
                    spans: &[],
                },
                &mut out,
            )
            .expect("delta");
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Length,
                },
                &mut out,
            )
            .expect("end");
        let events = out.drain();
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), "{\"city\": \"Par".to_string())]
        );
        assert_eq!(finish(&events).0, FinishReason::Length);
    }

    #[test]
    fn the_prompt_decides_where_the_output_starts() {
        let format = deepseek_v3();
        assert!(!format.reasoning_open("<｜User｜>Hi<｜Assistant｜>"));
        assert!(format.reasoning_open(THINKING));
        // After a tool result the prompt ends in `<｜tool▁outputs▁end｜>` with no new opener: the
        // replay walks the previous turn's calls block from its `<｜Assistant｜>` and leaves the
        // output in content, where the template puts it.
        let prompt = "<｜User｜>Weather in Paris?<｜Assistant｜><｜tool▁calls▁begin｜>\
                      <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n\
                      {\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\
                      <｜end▁of▁sentence｜><｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>\
                      {\"temperature\": 18}<｜tool▁output▁end｜><｜tool▁outputs▁end｜>";
        assert_eq!(
            format.replay(prompt),
            &prompt[prompt.find("<｜Assistant｜>").expect("the opener")..]
        );
        assert!(!format.reasoning_open(prompt));
        let events = run(prompt, &["It is 18 degrees."]);
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
        // The R1 distills' template writes `<｜tool▁calls▁end｜><｜end▁of▁sentence｜>` after the
        // second call and later only, so a turn with one call ends with neither; the tool
        // outputs' opener leaves the block all the same, and the model's own `<think>` opens the
        // thought from content, as it does after `<｜Assistant｜>`.
        let one_call = "<｜User｜>Weather in Paris?<｜Assistant｜><｜tool▁calls▁begin｜>\
                        <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n\
                        {\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁outputs▁begin｜>\
                        <｜tool▁output▁begin｜>{\"temperature\": 18}<｜tool▁output▁end｜>\
                        <｜tool▁outputs▁end｜>";
        assert!(
            !format.reasoning_open(one_call),
            "the tool outputs left the block"
        );
        let events = run(one_call, &["<think>Warm.</think>It is 18 degrees."]);
        assert_eq!(reasoning(&events), "Warm.");
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
        // The tool outputs are a state of their own, left for content on their close: the
        // answer starts with nothing begun, and whitespace alone ahead of its thought is held,
        // not shown as content.
        let events = run(one_call, &["\n<think>Warm.</think>It is 18 degrees."]);
        assert_eq!(reasoning(&events), "Warm.");
        assert_eq!(
            content(&events),
            "It is 18 degrees.",
            "no whitespace-only content"
        );
        let events = run(prompt, &["\n\n<think>Warm.</think>It is 18 degrees."]);
        assert_eq!(reasoning(&events), "Warm.");
        assert_eq!(
            content(&events),
            "It is 18 degrees.",
            "the lead is held after an ended block too"
        );
        // Nothing quoted inside a tool's output moves the walk: a literal `<think>` or a calls
        // opener in the result's text stays there, and the answer after it is content.
        let quoted = "<｜User｜>Weather in Paris?<｜Assistant｜><｜tool▁calls▁begin｜>\
                      <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n\
                      {\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁outputs▁begin｜>\
                      <｜tool▁output▁begin｜>{\"note\": \"<think> and <｜tool▁calls▁begin｜> \
                      are text here\"}<｜tool▁output▁end｜><｜tool▁outputs▁end｜>";
        assert!(
            !format.reasoning_open(quoted),
            "a quoted marker moves nothing"
        );
        let events = run(quoted, &["It is 18 degrees."]);
        assert_eq!(content(&events), "It is 18 degrees.");
        assert_eq!(reasoning(&events), "");
        assert!(calls(&events).is_empty());
        // R1's, R1-Zero's, the bf16 mirror's and the distills' templates write `<｜Assistant｜>`
        // and `<｜tool▁outputs▁begin｜>` for the conversation's first tool turn only (R1-0528's
        // reset them every tool turn), so a later turn's call and results are walked from that
        // first opener, in content: the inner `<｜tool▁output▁begin｜>` takes the walk into the
        // outputs there, and the close leaves them with nothing begun.
        let later = "<｜User｜>u1<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function\
                     <｜tool▁sep｜>f\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\
                     <｜end▁of▁sentence｜><｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>r1\
                     <｜tool▁output▁end｜><｜tool▁outputs▁end｜>answer1<｜end▁of▁sentence｜>\
                     <｜User｜>u2\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>g\n```json\n{}\n```\
                     <｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>\
                     {\"note\": \"<think>\"}<｜tool▁output▁end｜><｜tool▁outputs▁end｜>";
        assert!(
            !format.reasoning_open(later),
            "a later turn's result moves nothing"
        );
        let events = run(later, &["\n<think>Warm.</think>It is 18 degrees."]);
        assert_eq!(reasoning(&events), "Warm.");
        assert_eq!(
            content(&events),
            "It is 18 degrees.",
            "nothing begun after a later result"
        );
        assert!(calls(&events).is_empty());
        // An unbalanced `<think>` quoted in the text walked in content before the inner opener
        // (a user's question, a call's arguments) opens the thought; the inner opener takes the
        // walk from the thought into the outputs as it does from content.
        let quoted_later = later.replace("<｜User｜>u2", "<｜User｜>What does the <think> tag do?");
        assert!(
            !format.reasoning_open(&quoted_later),
            "the inner opener leaves the thought"
        );
        let events = run(&quoted_later, &["<think>Warm.</think>It is 18 degrees."]);
        assert_eq!(reasoning(&events), "Warm.");
        assert_eq!(content(&events), "It is 18 degrees.");
        // A thought the client prefilled after the tool result (`continue_final_message`) is
        // read: the walk went past the tool outputs, and `<think>` opened the thought.
        let prefilled = format!("{one_call}<think>\nThe tool says");
        assert!(
            format.reasoning_open(&prefilled),
            "the prefilled thought is open"
        );
        let events = run(&prefilled, &[" 18.</think>It is 18 degrees."]);
        assert_eq!(reasoning(&events), " 18.");
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
        // A marker quoted in an earlier turn moves nothing.
        let quoted =
            "<｜User｜>What does <｜tool▁calls▁begin｜> do?<｜Assistant｜>It opens the calls.\
                      <｜end▁of▁sentence｜><｜User｜>Thanks.<｜Assistant｜>";
        let events = run(quoted, &["You are welcome."]);
        assert_eq!(content(&events), "You are welcome.");
    }

    #[test]
    fn the_table_validates_and_names_its_opener() {
        let format = deepseek_v3();
        format.validate().expect("a valid table");
        assert_eq!(format.name(), "deepseek_v3");
        assert_eq!(format.turn_opener(), Some("<｜Assistant｜>"));
    }

    #[test]
    fn a_forced_call_is_the_block_holding_the_fenced_call_and_the_tools_schema() {
        // The tag is written from the same rows as V3.1's: the block's ways in (from the turn's
        // own state, or `</think>` first) around the one call, `<｜tool▁call▁begin｜>function
        // <｜tool▁sep｜>`, the tool's name and the fence the template opens, the arguments the
        // tool's schema accepts, and the fence's close with `<｜tool▁call▁end｜>`; then the block's
        // close, and nothing after it for a named function.
        let tool = Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {"city": {"type": "string"}}}),
                strict: None,
                extra: Default::default(),
            },
        };
        let got = deepseek_v3()
            .grammar_for(
                std::slice::from_ref(&tool),
                Options {
                    at_least_one: true,
                    reasoning_open: false,
                    thought_closed: false,
                    one_call: true,
                    start: GrammarStart::AfterReasoning,
                },
            )
            .expect("a tag")
            .payload();
        let one_call = value!({"type": "or", "elements": [{"type": "sequence", "elements": [
            {"type": "const_string", "value": "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n"},
            {"type": "json_schema", "json_schema": tool.function.parameters},
            {"type": "const_string", "value": "```<｜tool▁call▁end｜>"},
        ]}]});
        let tag = |begin: &str| {
            value!({
                "type": "tag",
                "begin": begin,
                "content": one_call,
                "end": "<｜tool▁calls▁end｜>",
            })
        };
        assert_eq!(
            got,
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<｜tool▁calls▁begin｜>", "</think>"],
                "tags": [
                    tag("<｜tool▁calls▁begin｜>"),
                    tag("</think><｜tool▁calls▁begin｜>"),
                ],
                "at_least_one": true,
                "stop_after_first": true,
            }})
        );
        // `required` with parallel calls allowed: the template's newline before every call after
        // the first, and nothing before the first.
        let more = deepseek_v3()
            .grammar_for(
                std::slice::from_ref(&tool),
                Options {
                    at_least_one: true,
                    reasoning_open: false,
                    thought_closed: false,
                    one_call: false,
                    start: GrammarStart::AfterReasoning,
                },
            )
            .expect("a tag")
            .payload();
        assert_eq!(
            more["format"]["tags"][0]["content"],
            value!({"type": "sequence", "elements": [
                one_call,
                {"type": "star", "content": {"type": "sequence", "elements": [
                    {"type": "const_string", "value": "\n"},
                    one_call,
                ]}},
            ]})
        );
        // With the prompt's thought open and the grammar run from the first token, the thought's
        // text may not write the table's terminals (the tool outputs' opener among them, a
        // marker the template writes and the model does not), and may still hold a fenced draft
        // of the arguments: the fence is ordinary text, and a call opens only at the call's
        // opener.
        let open = deepseek_v3()
            .grammar_for(
                std::slice::from_ref(&tool),
                Options {
                    at_least_one: true,
                    reasoning_open: true,
                    thought_closed: false,
                    one_call: true,
                    start: GrammarStart::FirstToken,
                },
            )
            .expect("a tag")
            .payload();
        assert_eq!(
            open["format"]["elements"][0]["content"]["excludes"],
            value!([
                "<think>",
                "</think>",
                "<｜tool▁calls▁begin｜>",
                "<｜tool▁calls▁end｜>",
                "<｜tool▁call▁begin｜>function<｜tool▁sep｜>",
                "```<｜tool▁call▁end｜>",
                "<｜tool▁outputs▁begin｜>",
                "<｜tool▁output▁begin｜>",
                "<｜tool▁outputs▁end｜>",
            ])
        );
        // The call as the tag writes it, the fence's close right after the object (the template
        // puts a newline between them, which the table drops as wrapping either way), reads as
        // the call whole and at every cut.
        let output = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}```<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        let expected = vec![(
            "get_weather".to_string(),
            "{\"city\": \"Paris\"}".to_string(),
        )];
        let whole = run(PROMPT, &[output]);
        assert_eq!(bytes(&whole), output);
        assert_eq!(calls(&whole), expected);
        assert_eq!(content(&whole), "");
        assert_eq!(malformed(&whole), Vec::<&str>::new());
        assert_eq!(finish(&whole), (FinishReason::Stop, 1));
        for (at, _) in output.char_indices().skip(1) {
            let (head, rest) = output.split_at(at);
            let events = run(PROMPT, &[head, rest]);
            assert_eq!(calls(&events), expected, "cut at {at}");
            assert_eq!(content(&events), "", "cut at {at}");
            assert_eq!(malformed(&events), Vec::<&str>::new(), "cut at {at}");
            assert_eq!(finish(&events), (FinishReason::Stop, 1), "cut at {at}");
        }
    }
}
