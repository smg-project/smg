//! Gemma 4's call syntax: `call:NAME{key:value,…}` between the table's markers, the events of one
//! call as its bytes arrive.
//!
//! Gemma 4 writes a call as
//!
//! ```text
//! <|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}<tool_call|>
//! ```
//!
//! The engine's table owns the markers ([`formats::gemma4`]): it enters this assembler after
//! `<|tool_call>` and leaves it at `<tool_call|>`, which it hands over as the call's end. Between
//! them the template writes `call:`, the tool's name and the arguments object in a spelling of its
//! own: keys bare, before `:`; a string between two `<|"|>` tokens, its text as it is, newlines
//! and quotes included, nothing escaped; a number, `true`, `false` and `null` bare; a list between
//! `[` and `]` and an object between `{` and `}`, their members joined by `,` with no space and
//! keys bare at every depth; the keys in sorted order (the template's `dictsort`), whatever order
//! the request gave them.
//!
//! The assembler writes the arguments as JSON while they arrive: each key quoted, `": "` after it
//! and `", "` between members, as the engines' parsers spell them; a string's text escaped as JSON
//! requires and streamed as it comes; a bare value as it is. The `<|"|>` token may be cut by a
//! piece's end, so a trailing start of it is held until the next bytes say. Every byte lands in one
//! event: the `call:NAME{` header is the start's source, each fragment's bytes its own, the closing
//! marker the end's. Text after the object closed and before the marker, which the template never
//! writes, is reported; a header that is not `call:NAME{` is reported and no call starts.
//!
//! [`formats::gemma4`]: crate::formats::gemma4()

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    tagged::assembler::{push_escaped, push_quoted},
};

/// The token around a string's text.
pub const STRING: &str = "<|\"|>";
/// The call's header up to the tool's name.
const HEADER: &str = "call:";
const NOT_A_CALL: &str = "a call that does not start with `call:NAME{`";
const TEXT_AFTER_THE_OBJECT: &str = "text after the arguments closed";
const EMPTY_KEY: &str = "a key without a name";
const KEY_CUT_SHORT: &str = "a key the call's end cut short";
const VALUE_CUT_SHORT: &str = "a value the call's end cut short";

/// The events of one call, from the bytes after `<|tool_call>`.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    /// Bytes read into no event yet: the header before its `{`, a key before its `:`, a bare value
    /// before its end, or a trailing start of `<|"|>`.
    carried: String,
    stage: Stage,
    /// The open containers, innermost last, each with whether a member has been written in it.
    containers: Vec<Container>,
    started: bool,
    done: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// Inside `call:NAME{`.
    Header,
    /// Where a key or the object's close stands.
    Key,
    /// Where a value or an empty list's close stands.
    Value,
    /// Inside a string: the text up to the next `<|"|>`.
    Text,
    /// Where `,` or the container's close stands.
    AfterValue,
    /// The arguments object closed; only the marker should follow.
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Container {
    list: bool,
    members: u32,
}

impl Assembler {
    /// An assembler for the call at `index` with the id the format minted for it, positioned just
    /// after `<|tool_call>`.
    pub fn new(index: u32, id: impl Into<String>) -> Self {
        Self {
            index,
            id: id.into(),
            carried: String::new(),
            stage: Stage::Header,
            containers: Vec::new(),
            started: false,
            done: false,
        }
    }

    /// Whether a call has started, that is, whether `ToolCallStart` has been pushed.
    pub fn started(&self) -> bool {
        self.started
    }

    /// Append the next bytes of the call and push the events they complete. Every byte is the
    /// call's until the engine says otherwise, so this takes them all.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) {
        if self.done || bytes.is_empty() {
            return;
        }
        let mut text = std::mem::take(&mut self.carried);
        text.push_str(bytes);
        let mut rest = text.as_str();
        while !rest.is_empty() {
            let taken = match self.stage {
                Stage::Header => self.header(rest, out),
                Stage::Key => self.key(rest, out),
                Stage::Value => self.value(rest, out),
                Stage::Text => self.text(rest, out),
                Stage::AfterValue => self.after_value(rest, out),
                Stage::Closed => after_the_object(rest, out),
            };
            if taken == 0 {
                // What is left is held: a header, key or bare value still arriving, or a start of
                // `<|"|>`.
                self.carried.push_str(rest);
                return;
            }
            rest = &rest[taken..];
        }
    }

    /// The call's end at the table's closing marker, whose bytes go into `ToolCallEnd`. An object
    /// still open is closed for the client, with what was cut short reported; a call that never
    /// started has its bytes reported.
    pub fn close(mut self, terminal: &str, out: &mut Events) {
        if self.done {
            return;
        }
        self.end(out);
        if self.started {
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::uncounted(terminal),
            });
        } else if !terminal.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(terminal),
                why: MalformedReason::Other(NOT_A_CALL.to_string()),
            });
        }
        self.done = true;
    }

    /// The stream was cut: what was held comes back as `Malformed { UnterminatedRegion }`, an
    /// object still open is closed for the client, and a call that started still ends, with no
    /// bytes of its own, as every assembler ends one.
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        if !self.carried.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(std::mem::take(&mut self.carried)),
                why: MalformedReason::UnterminatedRegion,
            });
        }
        if self.started {
            if self.stage == Stage::Text {
                self.push_fragment("\"".to_string(), Text::default(), out);
            }
            self.close_open_containers(out);
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::default(),
            });
        }
        self.done = true;
    }

    /// `call:NAME{`: the start, once the `{` arrives. Anything else up to the `{` is reported.
    fn header(&mut self, rest: &str, out: &mut Events) -> usize {
        let Some(brace) = rest.find('{') else {
            return 0;
        };
        let head = &rest[..brace];
        let name = head.strip_prefix(HEADER).map(str::trim);
        match name {
            Some(name) if !name.is_empty() && !name.contains(char::is_whitespace) => {
                self.started = true;
                out.push(Event::ToolCallStart {
                    index: self.index,
                    id: self.id.clone(),
                    name: name.to_string(),
                    source: Text::uncounted(&rest[..=brace]),
                });
                self.push_fragment("{".to_string(), Text::default(), out);
                self.containers.push(Container {
                    list: false,
                    members: 0,
                });
                self.stage = Stage::Key;
            }
            _ => {
                out.push(Event::Malformed {
                    text: Text::uncounted(&rest[..=brace]),
                    why: MalformedReason::Other(NOT_A_CALL.to_string()),
                });
                self.stage = Stage::Closed;
            }
        }
        brace + 1
    }

    /// A key up to its `:`, or the object's close.
    fn key(&mut self, rest: &str, out: &mut Events) -> usize {
        let space = rest.len() - rest.trim_start().len();
        if space == rest.len() {
            return 0;
        }
        if rest[space..].starts_with('}') {
            self.close_container(&rest[..=space], out);
            return space + 1;
        }
        let Some(colon) = rest.find(':') else {
            return 0;
        };
        let key = rest[space..colon].trim_end();
        if key.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(&rest[..=colon]),
                why: MalformedReason::Other(EMPTY_KEY.to_string()),
            });
        } else {
            let mut json = self.separator();
            push_quoted(&mut json, key);
            json.push_str(": ");
            self.push_fragment(json, Text::uncounted(&rest[..=colon]), out);
        }
        self.stage = Stage::Value;
        colon + 1
    }

    /// A value's start: a string's opening token, a container's opening bracket, an empty list's
    /// close, or a bare literal up to the byte that ends it.
    fn value(&mut self, rest: &str, out: &mut Events) -> usize {
        let space = rest.len() - rest.trim_start().len();
        if space == rest.len() {
            return 0;
        }
        let after = &rest[space..];
        let first = after.chars().next().unwrap_or_default();
        let string = after.starts_with(STRING);
        // Nothing is taken until the value's start is whole: a start of `<|"|>` still arriving, or
        // a bare literal whose end has not come, is held.
        if (!string && STRING.starts_with(after))
            || (!string && !matches!(first, '{' | '[' | ']') && !after.contains([',', '}', ']']))
        {
            return 0;
        }
        // In a list each value is a member and takes the separator; in an object the key took it.
        let separator = if self.in_list() && first != ']' {
            self.separator()
        } else {
            String::new()
        };
        if string {
            let json = format!("{separator}\"");
            self.push_fragment(json, Text::uncounted(&rest[..space + STRING.len()]), out);
            self.stage = Stage::Text;
            return space + STRING.len();
        }
        match first {
            '{' | '[' => {
                let json = format!("{separator}{first}");
                self.push_fragment(json, Text::uncounted(&rest[..=space]), out);
                self.containers.push(Container {
                    list: first == '[',
                    members: 0,
                });
                self.stage = if first == '[' {
                    Stage::Value
                } else {
                    Stage::Key
                };
                space + 1
            }
            ']' if self.in_list() => {
                self.close_container(&rest[..=space], out);
                space + 1
            }
            _ => {
                let end = after.find([',', '}', ']']).unwrap_or(after.len());
                let literal = after[..end].trim_end();
                let mut json = separator;
                push_bare(&mut json, literal);
                self.push_fragment(json, Text::uncounted(&rest[..space + end]), out);
                self.stage = Stage::AfterValue;
                space + end
            }
        }
    }

    /// A string's text up to its closing token, streamed as it arrives; a trailing start of the
    /// token is held.
    fn text(&mut self, rest: &str, out: &mut Events) -> usize {
        if let Some(at) = rest.find(STRING) {
            let mut json = String::with_capacity(at + 1);
            push_escaped(&mut json, &rest[..at]);
            json.push('"');
            self.push_fragment(json, Text::uncounted(&rest[..at + STRING.len()]), out);
            self.stage = Stage::AfterValue;
            return at + STRING.len();
        }
        let keep = (1..STRING.len())
            .rev()
            .find(|&n| {
                rest.len() >= n
                    && rest.is_char_boundary(rest.len() - n)
                    && STRING.starts_with(&rest[rest.len() - n..])
            })
            .unwrap_or(0);
        let give = &rest[..rest.len() - keep];
        if give.is_empty() {
            return 0;
        }
        let mut json = String::with_capacity(give.len());
        push_escaped(&mut json, give);
        self.push_fragment(json, Text::uncounted(give), out);
        give.len()
    }

    /// After a value: `,` before the next member, or the container's close.
    fn after_value(&mut self, rest: &str, out: &mut Events) -> usize {
        let space = rest.len() - rest.trim_start().len();
        if space == rest.len() {
            return 0;
        }
        let first = rest[space..].chars().next().unwrap_or_default();
        match first {
            ',' => {
                self.push_fragment(String::new(), Text::uncounted(&rest[..=space]), out);
                self.stage = if self.in_list() {
                    Stage::Value
                } else {
                    Stage::Key
                };
            }
            '}' | ']' => self.close_container(&rest[..=space], out),
            _ => {
                // The template writes a separator or a close here; anything else is reported and
                // read as the next value's start.
                out.push(Event::Malformed {
                    text: Text::uncounted(&rest[..space]),
                    why: MalformedReason::Other(VALUE_CUT_SHORT.to_string()),
                });
                self.stage = if self.in_list() {
                    Stage::Value
                } else {
                    Stage::Key
                };
                return space.max(1);
            }
        }
        space + 1
    }

    /// The innermost container closes: `}` or `]` written, with `source` as the bytes.
    fn close_container(&mut self, source: &str, out: &mut Events) {
        let container = self.containers.pop();
        let json = if container.is_some_and(|c| c.list) {
            "]"
        } else {
            "}"
        };
        self.push_fragment(json.to_string(), Text::uncounted(source), out);
        self.stage = if self.containers.is_empty() {
            Stage::Closed
        } else {
            Stage::AfterValue
        };
    }

    fn in_list(&self) -> bool {
        self.containers.last().is_some_and(|c| c.list)
    }

    /// `", "` before every member of a container but its first, and the member counted.
    fn separator(&mut self) -> String {
        match self.containers.last_mut() {
            Some(container) => {
                container.members += 1;
                if container.members > 1 {
                    ", ".to_string()
                } else {
                    String::new()
                }
            }
            None => String::new(),
        }
    }

    /// The call's end before its marker: a key or a value cut short is reported, a string left
    /// open gets its quote, and every open container is closed, so the arguments are an object.
    fn end(&mut self, out: &mut Events) {
        let held = std::mem::take(&mut self.carried);
        match self.stage {
            Stage::Header if !held.is_empty() => out.push(Event::Malformed {
                text: Text::uncounted(held),
                why: MalformedReason::Other(NOT_A_CALL.to_string()),
            }),
            Stage::Key | Stage::Value if !held.trim().is_empty() => {
                let why = if self.stage == Stage::Key {
                    KEY_CUT_SHORT
                } else {
                    VALUE_CUT_SHORT
                };
                out.push(Event::Malformed {
                    text: Text::uncounted(held),
                    why: MalformedReason::Other(why.to_string()),
                });
            }
            Stage::Text => {
                // The held bytes are the string's text, cut before its closing token.
                let mut json = String::with_capacity(held.len());
                push_escaped(&mut json, &held);
                self.push_fragment(json, Text::uncounted(held), out);
            }
            _ if !held.is_empty() => out.push(Event::Dropped {
                text: Text::uncounted(held),
                why: DropReason::Wrapper,
            }),
            _ => {}
        }
        if self.started {
            if self.stage == Stage::Text {
                self.push_fragment("\"".to_string(), Text::default(), out);
            }
            self.close_open_containers(out);
        }
    }

    fn close_open_containers(&mut self, out: &mut Events) {
        while let Some(container) = self.containers.pop() {
            let json = if container.list { "]" } else { "}" };
            self.push_fragment(json.to_string(), Text::default(), out);
        }
        self.stage = Stage::Closed;
    }

    fn push_fragment(&self, json: String, source: Text, out: &mut Events) {
        out.push(Event::ToolCallArguments {
            index: self.index,
            json,
            source,
        });
    }
}

/// Text after the arguments closed: whitespace is dropped, anything else reported.
fn after_the_object(rest: &str, out: &mut Events) -> usize {
    if rest.trim().is_empty() {
        out.push(Event::Dropped {
            text: Text::uncounted(rest),
            why: DropReason::Wrapper,
        });
    } else {
        out.push(Event::Malformed {
            text: Text::uncounted(rest),
            why: MalformedReason::Other(TEXT_AFTER_THE_OBJECT.to_string()),
        });
    }
    rest.len()
}

/// A bare value as JSON: a number, `true`, `false` or `null` as it is; any other bare text, which
/// the template never writes, as a JSON string, so the arguments stay an object a client can read.
fn push_bare(json: &mut String, literal: &str) {
    let is_number = literal
        .strip_prefix('-')
        .unwrap_or(literal)
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit())
        && serde_json::from_str::<serde_json::Value>(literal).is_ok();
    if is_number || matches!(literal, "true" | "false" | "null") {
        json.push_str(literal);
    } else {
        push_quoted(json, literal);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLOSE: &str = "<tool_call|>";

    fn run(pieces: &[&str], cut: bool) -> Vec<Event> {
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, &mut out);
        }
        if cut {
            assembler.finish(&mut out);
        } else {
            assembler.close(CLOSE, &mut out);
        }
        out.drain()
    }

    fn bytes(events: &[Event]) -> String {
        events
            .iter()
            .map(|e| match e {
                Event::Content(t) | Event::Reasoning(t) => t.text.as_str(),
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => source.text.as_str(),
                Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
            })
            .collect()
    }

    fn arguments(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect()
    }

    fn malformed(events: &[Event]) -> Vec<(String, String)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed {
                    text,
                    why: MalformedReason::Other(why),
                } => Some((text.text.clone(), why.clone())),
                _ => None,
            })
            .collect()
    }

    const CALL: &str = concat!(
        "call:get_weather{city:<|\"|>Paris, \"la\" ville\n<|\"|>,days:3,metric:true,",
        "options:{tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],none:null,ratio:-0.5},empty:[]}"
    );

    #[test]
    fn a_rendered_call_gives_its_name_its_arguments_as_json_and_every_byte() {
        let events = run(&[CALL], false);
        assert_eq!(bytes(&events), format!("{CALL}{CLOSE}"));
        assert!(matches!(
            &events[0],
            Event::ToolCallStart { index: 0, name, source, .. }
                if name == "get_weather" && source.text == "call:get_weather{"
        ));
        assert_eq!(
            arguments(&events),
            concat!(
                r#"{"city": "Paris, \"la\" ville\n", "days": 3, "metric": true, "#,
                r#""options": {"tags": ["a", "b"], "none": null, "ratio": -0.5}, "empty": []}"#
            )
        );
        assert!(matches!(
            events.last(),
            Some(Event::ToolCallEnd { source, .. }) if source.text == CLOSE
        ));
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn the_call_reads_the_same_however_the_bytes_are_cut() {
        let whole = run(&[CALL], false);
        let expected = arguments(&whole);
        let boundaries: Vec<usize> = CALL.char_indices().map(|(at, _)| at).collect();
        for &at in &boundaries[1..] {
            let events = run(&[&CALL[..at], &CALL[at..]], false);
            assert_eq!(bytes(&events), format!("{CALL}{CLOSE}"), "cut at {at}");
            assert_eq!(arguments(&events), expected, "cut at {at}");
            assert!(malformed(&events).is_empty(), "cut at {at}");
        }
        let one_by_one: Vec<&str> = CALL
            .char_indices()
            .map(|(at, c)| &CALL[at..at + c.len_utf8()])
            .collect();
        let events = run(&one_by_one, false);
        assert_eq!(arguments(&events), expected);
        assert_eq!(bytes(&events), format!("{CALL}{CLOSE}"));
    }

    #[test]
    fn a_string_streams_as_its_pieces_arrive() {
        let events = run(
            &[
                "call:f{patch:<|\"|>diff --git",
                " a/x\n+if a < b",
                "<|\"|>}",
            ],
            false,
        );
        let fragments: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } if !json.is_empty() => Some(json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            fragments,
            [
                "{",
                "\"patch\": ",
                "\"",
                "diff --git",
                " a/x\\n+if a < b",
                "\"",
                "}"
            ]
        );
    }

    #[test]
    fn no_arguments_is_an_empty_object_and_a_cut_stream_closes_what_is_open() {
        let none = run(&["call:ping{}"], false);
        assert_eq!(arguments(&none), "{}");
        assert!(malformed(&none).is_empty());
        let cut = run(&["call:f{city:<|\"|>Par"], true);
        assert_eq!(arguments(&cut), r#"{"city": "Par"}"#);
        assert!(cut.iter().any(
            |event| matches!(event, Event::ToolCallEnd { source, .. } if source.text.is_empty())
        ));
        let cut_in_key = run(&["call:f{ci"], true);
        assert_eq!(arguments(&cut_in_key), "{}");
        assert_eq!(bytes(&cut_in_key), "call:f{ci");
    }

    #[test]
    fn what_is_not_a_call_is_reported_and_so_is_text_after_the_object() {
        let not = run(&["hello{x:1}"], false);
        assert!(!not
            .iter()
            .any(|event| matches!(event, Event::ToolCallStart { .. })));
        assert_eq!(bytes(&not), format!("hello{{x:1}}{CLOSE}"));
        assert_eq!(malformed(&not)[0].1, NOT_A_CALL);
        let tail = run(&["call:f{x:1} trailing"], false);
        assert_eq!(arguments(&tail), r#"{"x": 1}"#);
        assert_eq!(
            malformed(&tail),
            vec![(" trailing".to_string(), TEXT_AFTER_THE_OBJECT.to_string())]
        );
        assert_eq!(bytes(&tail), format!("call:f{{x:1}} trailing{CLOSE}"));
    }

    #[test]
    fn a_bare_value_that_is_no_literal_is_kept_as_a_string() {
        let events = run(&["call:f{mode:fast,n:12}"], false);
        assert_eq!(arguments(&events), r#"{"mode": "fast", "n": 12}"#);
    }
}
