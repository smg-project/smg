//! The structural tag of a call given as data ([`CallSyntax::Shaped`]), written as the table's
//! [`Layout`] says the model's chat template writes one.
//!
//! A shape's patterns read many spellings of a call; the tag writes the layout's. A call is the
//! region's terminal and the header as written, the tool's name in its place; then the tool's
//! parameters, one per property of its schema in the schema's order, each optional unless the
//! schema requires it, each written as the layout writes a parameter (its tag with the key, the
//! value, the end), with the layout's text before the first, between two and after the last; then
//! the trailer and the terminal that closes the call. What follows a parameter whether another
//! follows or not (the text the layout writes between two parameters and after the last begin with
//! alike, as the newline a template writes after each) is the parameter's own, so a call that
//! writes no parameter has the header, the text before the first and the rest of the text after the
//! last. The terminal triggers the tag, so free text cannot write a call the tag does not
//! constrain. Where the request asks for a call at once, the layout's lead comes first: the text
//! the template writes before the terminal, such as a message's header the response template reads
//! as text outside every field, written for each tool where it names one.
//!
//! A value follows its property's type. A string is any text that holds none of the texts a
//! value's ends begin with, nor a terminal that takes the call out of its region, so the value ends
//! where the tag ends it; a string `enum` is one of its values. A number is JSON; a boolean or null
//! the word the layout spells it with; a list or an object JSON the property's schema accepts, or
//! any text where the template writes it as text. A property without a type takes any text, and a
//! schema without properties any parameters.
//!
//! A tool whose name the header does not read back as written (a character its class leaves out)
//! has no tag; nor has one whose schema requires a key the parameter's tag does not read back, and
//! such a key the schema does not require is left out.
//!
//! A tag of one call ends with what the layout writes after the last call, and nothing may follow
//! it. Where the stream's stop closes the last call (Harmony's `<|call|>`), a call's region stays
//! open after its parameters until the terminal that starts the next call, or the stop, so text
//! after a tag of the calls would be the call's own: the calls a request may make at once are
//! written one after another, the layout's text between two, and nothing follows the last but the
//! stop. A tag the stop ends holds no repetition, which an engine cannot end on its own: a schema
//! without properties takes at most one parameter there. Where no call is owed and more than one
//! may come, such a call has no tag: the free text after a tag would be the call's own, and a
//! parameter written there would count with nothing to constrain it.
//!
//! With the prompt's thought open, a tag that owes a call follows the free prefix every syntax's
//! tag takes; one that owes none takes no prefix, its own free text writing the thought: an engine
//! reads free text before triggered tags so that the prefix could end inside the call's terminal
//! and the triggered tags' free text write the rest, a call no tag constrains.
//!
//! Arguments written as JSON, or as an object that names the call, have no tag yet: a layout has
//! no places for them.
//!
//! [`CallSyntax::Shaped`]: crate::CallSyntax::Shaped

use openai_protocol::common::Tool;
use serde_json::Value;

use super::{
    schema::{self, shape as typed, Definitions},
    Count, Grammar, Tag,
};
use crate::{
    format::{Emits, Format},
    shape::{Arguments, Atom, CallShape, End, Layout, Nested, Segment, Slot, Spellings, Words},
};

/// One call as a tag writes it: the layout's lead for the tool, what the tag begins with (the
/// terminal, the header and the text before the parameters), and the parameters.
#[derive(Clone)]
struct Call {
    lead: String,
    begin: String,
    arguments: Grammar,
}

/// The structural tag of a shaped call to one of `tools`, as `count` asks, written as `layout`
/// says the template writes one (see the module documentation). `None` when the layout does not
/// fit the table, the arguments are not tagged parameters, a value's end holds no text, no tool
/// can be called through the header, or the stream's stop closes the last call while no call is
/// owed and more than one may come.
pub(super) fn calls(
    format: &Format,
    shape: &CallShape,
    layout: &Layout,
    tools: &[&Tool],
    count: Count<'_>,
) -> Option<Grammar> {
    layout.check(format).ok()?;
    let Arguments::Params { ends, .. } = &shape.arguments else {
        return None;
    };
    let state = (0..format.states()).find(|&state| format.emits(state) == Emits::Arguments)?;
    let terminals = rows(format, |from, to| from != state && to == state);
    let exits = rows(format, |from, to| from == state && to != state);
    let values = Values {
        excludes: excludes(ends, &exits)?,
        spellings: layout.values,
    };
    let call = &layout.call;
    let parameters = Parameters::of(layout);
    let tail = format!(
        "{}{}{}",
        parameters.closing,
        call.trailer,
        call.close.as_deref().unwrap_or("")
    );
    let one = count.one || layout.between_calls.is_none();
    // Where the stop closes the last call, its region stays open after its parameters: with no
    // call owed and more than one allowed, free text after a tag would be the call's own, where a
    // parameter no tag constrains would count, so there is no tag.
    if call.close.is_none() && !one && !count.at_least_one {
        return None;
    }
    // Where the stop closes the last call, the calls are written one after another, nothing after
    // the last; otherwise each call is a tag of its own, free text after it outside its region.
    let in_a_row = call.close.is_none() && !one;
    let end = match (in_a_row, one) {
        (false, true) => format!("{tail}{}", layout.after_calls),
        _ => tail.clone(),
    };
    let limited = end.is_empty();
    let mut written: Vec<Call> = Vec::new();
    for tool in tools {
        let name = &tool.function.name;
        let Some(arguments) = parameters.of_tool(shape, layout, &values, tool, limited) else {
            continue;
        };
        let (lead, _) = call.lead.around_the_value(name, "");
        for terminal in &terminals {
            if let Some(header) = layout.header_for(shape, terminal, name) {
                written.push(Call {
                    lead: lead.clone(),
                    begin: format!("{terminal}{header}{}", call.before),
                    arguments: arguments.clone(),
                });
            }
        }
    }
    if written.is_empty() {
        return None;
    }
    if in_a_row {
        let tags = written
            .iter()
            .map(|call| {
                let begin = format!("{}{}", call.lead, call.begin);
                Tag::new(begin, call.arguments.clone(), end.clone())
            })
            .collect();
        let calls = Grammar::TagsWithSeparator {
            tags,
            separator: layout.between_calls.as_deref().unwrap_or("").to_string(),
            at_least_one: true,
            stop_after_first: false,
        };
        return Some(sequence(vec![calls, text(&layout.after_calls)]));
    }
    let tags = written
        .iter()
        .map(|call| Tag::new(call.begin.clone(), call.arguments.clone(), end.clone()))
        .collect();
    let triggered = Grammar::TriggeredTags {
        triggers: terminals
            .iter()
            .map(|terminal| terminal.to_string())
            .collect(),
        tags,
        at_least_one: count.at_least_one,
        stop_after_first: one,
    };
    match lead(&written) {
        Some(lead) if count.at_least_one => Some(Grammar::Sequence(vec![lead, triggered])),
        _ => Some(triggered),
    }
}

/// The texts of the terminals of the rows `which` takes, `(from, to)`, each once, in the order of
/// the rows.
fn rows(format: &Format, which: impl Fn(usize, usize) -> bool) -> Vec<&str> {
    let mut texts: Vec<&str> = Vec::new();
    for (from, on, to) in format.transitions() {
        let text = format.terminal_text(on);
        if which(from, to) && !texts.contains(&text) {
            texts.push(text);
        }
    }
    texts
}

/// What the request's first call is written after: the layout's lead, for each tool where it
/// names one; `None` when the layout writes nothing before a call.
fn lead(calls: &[Call]) -> Option<Grammar> {
    let mut leads: Vec<&str> = Vec::new();
    for call in calls {
        if !call.lead.is_empty() && !leads.contains(&call.lead.as_str()) {
            leads.push(&call.lead);
        }
    }
    match leads.as_slice() {
        [] => None,
        [only] => Some(text(only)),
        many => Some(Grammar::Or(many.iter().map(|lead| text(lead)).collect())),
    }
}

/// How the layout joins a call's parameters. What follows a parameter whether another follows or
/// not, the text the layout's `between` and `after` begin with alike, ends each parameter (the
/// newline after each of Qwen 3.5's and Muse Glimmer's); the rest of `between` stands between two
/// parameters, and the rest of `after` after the last, or right after `before` in a call that
/// writes none.
struct Parameters<'l> {
    terminator: &'l str,
    separator: &'l str,
    closing: &'l str,
}

impl<'l> Parameters<'l> {
    fn of(layout: &'l Layout) -> Self {
        let (between, after) = (&*layout.call.between, &*layout.call.after);
        let common = between
            .char_indices()
            .zip(after.chars())
            .find(|&((_, a), b)| a != b)
            .map_or_else(|| between.len().min(after.len()), |((at, _), _)| at);
        Self {
            terminator: &between[..common],
            separator: &between[common..],
            closing: &after[common..],
        }
    }

    /// The parameters of a call to `tool` as the layout writes them: one per property, in the
    /// schema's order, each optional unless the schema requires it, each with its terminator and
    /// the separator between two. `None` when the tag does not read back a key the schema
    /// requires, or, for a tag the stop ends (`limited`), a value with no text after it, which an
    /// engine could not end. A schema without properties takes any parameters, at most one in a
    /// tag the stop ends.
    fn of_tool(
        &self,
        shape: &CallShape,
        layout: &Layout,
        values: &Values,
        tool: &Tool,
        limited: bool,
    ) -> Option<Grammar> {
        let parameters = &tool.function.parameters;
        let properties = parameters
            .get("properties")
            .and_then(Value::as_object)
            .filter(|properties| !properties.is_empty());
        let Some(properties) = properties else {
            let any = sequence(vec![any_parameter(layout, values)?, text(self.terminator)]);
            if limited {
                return Some(Grammar::Optional(Box::new(any)));
            }
            let more = Grammar::Star(Box::new(sequence(vec![text(self.separator), any.clone()])));
            return Some(Grammar::Optional(Box::new(sequence(vec![any, more]))));
        };
        let required: Vec<&str> = parameters
            .get("required")
            .and_then(Value::as_array)
            .map(|names| names.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let definitions = Definitions::of(parameters);
        let mut pairs: Vec<(Grammar, bool)> = Vec::new();
        for (key, property) in properties {
            let required = required.contains(&key.as_str());
            let Some((before, after)) = layout.parameter_for(shape, key) else {
                if required {
                    return None;
                }
                continue;
            };
            let after = format!("{after}{}", self.terminator);
            if limited && after.is_empty() {
                return None;
            }
            let value = values.of(property, definitions);
            pairs.push((sequence(vec![text(&before), value, text(&after)]), required));
        }
        Some(self.joined(pairs))
    }

    /// `pairs` one after another, the separator between two present ones.
    fn joined(&self, pairs: Vec<(Grammar, bool)>) -> Grammar {
        if pairs.is_empty() {
            return text("");
        }
        if !self.separator.is_empty() {
            return schema::joined(&pairs, self.separator);
        }
        sequence(
            pairs
                .into_iter()
                .map(|(pair, required)| {
                    if required {
                        pair
                    } else {
                        Grammar::Optional(Box::new(pair))
                    }
                })
                .collect(),
        )
    }
}

/// A parameter with any key and any value, as the layout writes one: the key any text up to the
/// first character the layout writes after it, holding none of the texts a value may not hold
/// either, since a terminal there would take the call out of its region.
fn any_parameter(layout: &Layout, values: &Values) -> Option<Grammar> {
    let segments = &layout.call.parameter.0;
    let mut parts: Vec<Grammar> = Vec::new();
    for (at, segment) in segments.iter().enumerate() {
        parts.push(match segment {
            Segment::Lit(lit) => text(lit),
            Segment::Slot(Slot::Key) => {
                let next = segments.get(at + 1).and_then(|next| match next {
                    Segment::Lit(lit) => lit.chars().next(),
                    Segment::Slot(_) => None,
                })?;
                let mut excludes = vec![next.to_string()];
                excludes.extend(values.excludes.iter().cloned());
                Grammar::AnyText { excludes }
            }
            Segment::Slot(Slot::Value) => values.text(),
            Segment::Slot(Slot::Name) => return None,
        });
    }
    Some(sequence(parts))
}

/// How a call's values are written: the texts a string may not hold, and the layout's spelling of
/// what is not a string.
struct Values {
    excludes: Vec<String>,
    spellings: Spellings,
}

impl Values {
    /// A value written as it is: any text the value's ends and the call's exits are not in.
    fn text(&self) -> Grammar {
        Grammar::AnyText {
            excludes: self.excludes.clone(),
        }
    }

    /// The value of a property, by the type its schema pins and the layout's spelling.
    fn of(&self, property: &Value, definitions: Definitions<'_>) -> Grammar {
        let python = self.spellings.words == Words::Python;
        let words = |json: &[&str], spelled: &[&str]| {
            let words = if python { spelled } else { json };
            match words {
                [one] => text(one),
                many => Grammar::Or(many.iter().map(|word| text(word)).collect()),
            }
        };
        match typed(property, definitions, || self.text()) {
            Some(("string", value)) => value,
            Some(("boolean", _)) => words(&["true", "false"], &["True", "False"]),
            Some(("null", _)) => words(&["null"], &["None"]),
            Some(("object" | "array", value)) => match self.spellings.nested {
                Nested::Json => value,
                Nested::Text => self.text(),
            },
            Some((_, value)) => value,
            None => self.text(),
        }
    }
}

/// The texts a string value may not hold, or the value would end inside it: the text every match
/// of each of the value's ends holds, its literals from the first on, and each terminal that
/// takes the call out of its region (`exits`). `None` when an end holds no literal, so that any
/// text could end the value.
fn excludes(ends: &[End], exits: &[&str]) -> Option<Vec<String>> {
    let mut excludes: Vec<String> = Vec::new();
    for end in ends {
        let core = core(end)?;
        if !excludes.contains(&core) {
            excludes.push(core);
        }
    }
    for exit in exits {
        if !excludes.iter().any(|known| known == exit) {
            excludes.push((*exit).to_string());
        }
    }
    Some(excludes)
}

/// The literals every match of `end` holds one after another: from its first literal, what it
/// takes and then what it looks ahead at, up to the first atom that is not a literal.
fn core(end: &End) -> Option<String> {
    let mut core = String::new();
    for atom in end.consume.0.iter().chain(end.ahead.0.iter()) {
        match atom {
            Atom::Lit(lit) => core.push_str(lit),
            _ if core.is_empty() => {}
            _ => break,
        }
    }
    (!core.is_empty()).then_some(core)
}

fn text(text: &str) -> Grammar {
    Grammar::ConstString(text.to_string())
}

/// `parts` one after another, an empty text left out; one part is itself, and none the empty text.
fn sequence(parts: Vec<Grammar>) -> Grammar {
    let mut parts: Vec<Grammar> = parts
        .into_iter()
        .filter(|part| !matches!(part, Grammar::ConstString(text) if text.is_empty()))
        .collect();
    match parts.len() {
        0 => text(""),
        1 => parts.pop().unwrap_or_else(|| text("")),
        _ => Grammar::Sequence(parts),
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;
    use serde_json::json as value;

    use super::*;
    use crate::{
        adapt,
        engine::Engine,
        event::Events,
        grammar::{GrammarStart, Options},
        input::{EngineFinish, Input},
        parser::Parser as _,
        shape::{intern, CallLayout, Class, Message, NameSource, Pattern, ValueParser, Written},
        tagged::Declared,
        CallSyntax,
    };

    fn tool(name: &str, parameters: Value) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters,
                strict: None,
                extra: Default::default(),
            },
        }
    }

    /// A tool with a required string, an optional integer and a required boolean.
    fn lookup() -> Tool {
        tool(
            "lookup",
            value!({
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "n": {"type": "integer"},
                    "on": {"type": "boolean"},
                },
                "required": ["q", "on"],
            }),
        )
    }

    fn none_of(chars: &str) -> Class {
        Class::new(chars.chars().map(|c| (c, c)), true)
    }

    fn word() -> Class {
        Class::new([('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false)
    }

    fn run(class: Class, min: u16, greedy: bool, capture: Option<Slot>) -> Atom {
        Atom::Run {
            class,
            min,
            max: None,
            greedy,
            capture,
        }
    }

    fn lit(text: &str) -> Atom {
        Atom::Lit(text.into())
    }

    fn written(segments: impl IntoIterator<Item = Segment>) -> Written {
        Written(segments.into_iter().collect())
    }

    fn piece(text: &str) -> Segment {
        Segment::Lit(text.into())
    }

    /// A call on abstract tags whose terminal `<c` is followed by attributes up to `n="NAME">`,
    /// each parameter `<p KEY>` with its value up to `</p>`, closed by `</c>`; the text outside it
    /// is framing, and an answer opens with `<a>`.
    fn named() -> Format {
        let shape = intern(CallShape {
            header: Pattern(
                [
                    Atom::Boundary,
                    run(none_of(">"), 0, false, None),
                    Atom::Boundary,
                    lit("n=\""),
                    run(none_of("\""), 1, true, Some(Slot::Name)),
                    lit("\">"),
                ]
                .into(),
            ),
            name: NameSource::Header,
            arguments: Arguments::Params {
                tag: Pattern(
                    [
                        lit("<p "),
                        run(none_of(">"), 1, true, Some(Slot::Key)),
                        lit(">"),
                        run(Class::any(), 0, false, Some(Slot::Value)),
                    ]
                    .into(),
                ),
                ends: [End {
                    consume: Pattern([lit("</p>")].into()),
                    ahead: Pattern::default(),
                }]
                .into(),
                values: ValueParser::Raw,
            },
            trailer: Pattern::default(),
            word: word(),
            hold_cap: 8192,
        });
        Format::new("named")
            .terminal("open", "<c")
            .terminal("close", "</c>")
            .terminal("answer_open", "<a>")
            .terminal("answer_close", "</a>")
            .state("outside", Emits::Content)
            .state("content", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("outside", "open", "call")
            .transition("call", "close", "outside")
            .transition("outside", "answer_open", "content")
            .transition("content", "answer_close", "outside")
            .calls(CallSyntax::Shaped(shape))
            .framing("outside")
    }

    /// How the abstract template writes a turn of `named`: `@NAME` and a newline before a call, a
    /// space before `n="NAME">`, a parameter a line, and a blank line between two calls.
    fn named_layout() -> Layout {
        Layout {
            call: CallLayout {
                lead: written([piece("@"), Segment::Slot(Slot::Name), piece("\n")]),
                header: written([piece(" n=\""), Segment::Slot(Slot::Name), piece("\">")]),
                before: "\n".into(),
                parameter: written([
                    piece("<p "),
                    Segment::Slot(Slot::Key),
                    piece(">"),
                    Segment::Slot(Slot::Value),
                    piece("</p>"),
                ]),
                between: "\n".into(),
                after: "\n".into(),
                trailer: "".into(),
                close: Some("</c>".into()),
            },
            between_calls: Some("\n\n".into()),
            after_calls: "\n".into(),
            values: Spellings::default(),
            thought: None,
            answer: Some(Message {
                before: " ".into(),
                after: "".into(),
                stops: false,
                close: None,
            }),
        }
    }

    fn options(at_least_one: bool, one_call: bool) -> Options {
        Options {
            at_least_one,
            reasoning_open: false,
            one_call,
            start: GrammarStart::Unknown,
        }
    }

    /// The payload of `format`'s tag for `tools`.
    fn payload(format: &Format, tools: &[Tool], options: Options) -> Option<Value> {
        format
            .grammar_for(tools, options)
            .map(|grammar| grammar.payload())
    }

    /// The parameters of `lookup` as the abstract layout writes them, each string a text that
    /// holds neither the value's end nor the call's close.
    fn lookup_parameters(boolean: Value) -> Value {
        // The newline the layout writes after each parameter, whether another follows or not.
        let parameter = |key: &str, value: Value| {
            value!({"type": "sequence", "elements": [
                {"type": "const_string", "value": format!("<p {key}>")},
                value,
                {"type": "const_string", "value": "</p>\n"},
            ]})
        };
        let q = parameter(
            "q",
            value!({"type": "any_text", "excludes": ["</p>", "</c>"]}),
        );
        let n = parameter(
            "n",
            value!({"type": "json_schema", "json_schema": {"type": "integer"}}),
        );
        let on = parameter("on", boolean);
        value!({"type": "sequence", "elements": [
            q,
            {"type": "optional", "content": n},
            on,
        ]})
    }

    fn json_words() -> Value {
        value!({"type": "or", "elements": [
            {"type": "const_string", "value": "true"},
            {"type": "const_string", "value": "false"},
        ]})
    }

    #[test]
    fn a_call_is_its_terminal_its_header_and_each_parameter_as_the_layout_writes_them() {
        let format = named().with_layout(named_layout());
        let tag = value!({
            "type": "tag",
            "begin": "<c n=\"lookup\">\n",
            "content": lookup_parameters(json_words()),
            "end": "</c>",
        });
        // A call at once: the lead for the tool first, then the calls.
        assert_eq!(
            payload(&format, &[lookup()], options(true, false)),
            Some(value!({"format": {"type": "sequence", "elements": [
                {"type": "const_string", "value": "@lookup\n"},
                {
                    "type": "triggered_tags",
                    "triggers": ["<c"],
                    "tags": [tag],
                    "at_least_one": true,
                },
            ]}}))
        );
        // No call owed: the lead is free text before the terminal.
        assert_eq!(
            payload(&format, &[lookup()], options(false, false)),
            Some(value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<c"],
                "tags": [tag],
                "at_least_one": false,
            }}))
        );
    }

    #[test]
    fn one_call_ends_with_what_the_layout_writes_after_the_last() {
        let format = named().with_layout(named_layout());
        let got = payload(&format, &[lookup()], options(true, true)).expect("a tag");
        let triggered = &got["format"]["elements"][1];
        assert_eq!(triggered["stop_after_first"], true);
        assert_eq!(triggered["tags"][0]["end"], "</c>\n");
        // A template that writes one call in a turn holds one in every tag.
        let mut single = named_layout();
        single.between_calls = None;
        let format = named().with_layout(single);
        let got = payload(&format, &[lookup()], options(true, false)).expect("a tag");
        assert_eq!(got["format"]["elements"][1]["stop_after_first"], true);
    }

    #[test]
    fn values_are_spelled_as_the_layout_says_and_a_lead_names_each_tool() {
        let mut python = named_layout();
        python.values = Spellings {
            words: Words::Python,
            nested: Nested::Text,
        };
        let format = named().with_layout(python);
        let tools = [
            lookup(),
            tool(
                "other",
                value!({
                    "type": "object",
                    "properties": {"o": {"type": "object"}, "z": {"type": "null"}},
                }),
            ),
        ];
        let got = payload(&format, &tools, options(true, false)).expect("a tag");
        assert_eq!(
            got["format"]["elements"][0],
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": "@lookup\n"},
                {"type": "const_string", "value": "@other\n"},
            ]})
        );
        let tags = &got["format"]["elements"][1]["tags"];
        assert_eq!(
            tags[0]["content"],
            lookup_parameters(value!({"type": "or", "elements": [
                {"type": "const_string", "value": "True"},
                {"type": "const_string", "value": "False"},
            ]}))
        );
        let other = tags[1]["content"].to_string();
        assert!(
            other.contains("{\"type\":\"const_string\",\"value\":\"None\"}"),
            "{other}"
        );
        assert!(
            !other.contains("json_schema"),
            "an object written as text: {other}"
        );
    }

    #[test]
    fn a_name_or_a_key_the_patterns_do_not_read_back_has_no_tag() {
        let format = named().with_layout(named_layout());
        let quoted = tool("we\"ird", value!({"type": "object"}));
        let keyed = |required: bool| {
            tool(
                "keyed",
                value!({
                    "type": "object",
                    "properties": {"a>b": {"type": "string"}, "q": {"type": "string"}},
                    "required": if required { value!(["a>b"]) } else { value!([]) },
                }),
            )
        };
        assert_eq!(
            payload(&format, std::slice::from_ref(&quoted), options(true, false)),
            None
        );
        assert_eq!(payload(&format, &[keyed(true)], options(true, false)), None);
        let got = payload(&format, &[quoted, keyed(false)], options(true, false))
            .expect("the readable tool");
        let tags = got["format"]["elements"][1]["tags"]
            .as_array()
            .expect("tags");
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0]["begin"], "<c n=\"keyed\">\n");
        assert!(!tags[0]["content"].to_string().contains("a>b"));
    }

    #[test]
    fn a_table_without_a_layout_that_fits_has_no_tag() {
        assert_eq!(payload(&named(), &[lookup()], options(true, false)), None);
        let mut unread = named_layout();
        unread.call.header = written([piece(" m=\""), Segment::Slot(Slot::Name), piece("\">")]);
        let format = named().with_layout(unread);
        assert_eq!(payload(&format, &[lookup()], options(true, false)), None);
    }

    #[test]
    fn a_key_no_property_names_holds_no_terminal_that_leaves_the_call() {
        // A schema without properties takes any parameters. A key is any text up to the `>` the
        // layout writes after it, and holds no value's end, nor the call's close, which would end
        // the call inside the key.
        let format = named().with_layout(named_layout());
        let bare = tool("bare", value!({"type": "object"}));
        let got = payload(&format, &[bare], options(true, true))
            .expect("a tag")
            .to_string();
        let key = value!({"type": "any_text", "excludes": [">", "</p>", "</c>"]}).to_string();
        assert!(got.contains(&key), "{got}");
    }

    /// A call on abstract tags the stream's stop closes: after `[c]`, the tool's name is the
    /// template's own (`f`); each parameter `<k KEY>`, its value up to `</k>` or up to the next
    /// parameter's tag; `[e]` closes a call another follows, and the stop token `[s]` the last.
    fn stopped() -> Format {
        let shape = intern(CallShape {
            header: Pattern::default(),
            name: NameSource::Literal("f".into()),
            arguments: Arguments::Params {
                tag: Pattern(
                    [
                        lit("<k "),
                        run(word(), 1, true, Some(Slot::Key)),
                        lit(">"),
                        run(Class::any(), 0, false, Some(Slot::Value)),
                    ]
                    .into(),
                ),
                ends: [
                    End {
                        consume: Pattern([lit("</k>")].into()),
                        ahead: Pattern::default(),
                    },
                    End {
                        consume: Pattern::default(),
                        ahead: Pattern([lit("<k ")].into()),
                    },
                ]
                .into(),
                values: ValueParser::Raw,
            },
            trailer: Pattern::default(),
            word: word(),
            hold_cap: 8192,
        });
        Format::new("stopped")
            .terminal("open", "[c]")
            .terminal("next", "[e]")
            .terminal("stop", "[s]")
            .state("outside", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("outside", "open", "call")
            .transition("call", "next", "outside")
            .transition("call", "stop", "outside")
            .calls(CallSyntax::Shaped(shape))
            .framing("outside")
            .closes_at_stop("call")
            .stop_tokens(["[s]"])
    }

    fn stopped_layout() -> Layout {
        Layout {
            call: CallLayout {
                lead: Written::default(),
                header: Written::default(),
                before: "".into(),
                parameter: written([
                    piece("<k "),
                    Segment::Slot(Slot::Key),
                    piece(">"),
                    Segment::Slot(Slot::Value),
                    piece("</k>"),
                ]),
                between: " ".into(),
                after: "".into(),
                trailer: "".into(),
                close: None,
            },
            between_calls: Some("[e]".into()),
            after_calls: "".into(),
            values: Spellings::default(),
            thought: None,
            answer: None,
        }
    }

    fn f() -> Tool {
        tool(
            "f",
            value!({
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            }),
        )
    }

    #[test]
    fn where_the_stop_closes_the_last_call_the_calls_are_written_in_a_row() {
        let format = stopped().with_layout(stopped_layout());
        let text = value!({"type": "any_text", "excludes": ["</k>", "<k ", "[e]", "[s]"]});
        let parameters = value!({"type": "sequence", "elements": [
            {"type": "sequence", "elements": [
                {"type": "const_string", "value": "<k a>"},
                text,
                {"type": "const_string", "value": "</k>"},
            ]},
            {"type": "sequence", "elements": [
                {"type": "const_string", "value": " "},
                {"type": "sequence", "elements": [
                    {"type": "const_string", "value": "<k b>"},
                    {"type": "json_schema", "json_schema": {"type": "integer"}},
                    {"type": "const_string", "value": "</k>"},
                ]},
            ]},
        ]});
        let tag = value!({"type": "tag", "begin": "[c]", "content": parameters, "end": ""});
        assert_eq!(
            payload(&format, &[f()], options(true, false)),
            Some(value!({"format": {
                "type": "tags_with_separator",
                "tags": [tag],
                "separator": "[e]",
                "at_least_one": true,
            }}))
        );
        // One call, owed or not: the tag ends at the stop, and nothing follows it.
        let one = payload(&format, &[f()], options(true, true)).expect("a tag");
        assert_eq!(one["format"]["type"], "triggered_tags");
        assert_eq!(one["format"]["stop_after_first"], true);
        assert_eq!(one["format"]["tags"][0]["end"], "");
        let free = payload(&format, &[f()], options(false, true)).expect("a tag");
        assert_eq!(free["format"]["type"], "triggered_tags");
        assert_eq!(free["format"]["at_least_one"], false);
        assert_eq!(free["format"]["stop_after_first"], true);
        // The template writes its own name: another tool has no tag.
        assert_eq!(
            payload(
                &format,
                &[tool("g", value!({"type": "object"}))],
                options(true, false)
            ),
            None
        );
        // A schema without properties takes at most one parameter where the stop ends the tag.
        let bare = payload(
            &format,
            &[tool("f", value!({"type": "object"}))],
            options(true, true),
        )
        .expect("a tag");
        assert_eq!(bare["format"]["tags"][0]["content"]["type"], "optional");
    }

    #[test]
    fn an_optional_call_the_stop_closes_has_no_tag_where_more_than_one_may_come() {
        let format = stopped().with_layout(stopped_layout());
        // After the tag `[c]<k a>x</k> <k b>1</k>`, the text up to the stop holds no `[c]`: free
        // text a tag that owes no call lets through, and the call's region reads it as its own,
        // `b` and an undeclared `z` with nothing to constrain them.
        assert_eq!(
            read(
                &format,
                &[f()],
                "",
                "[c]<k a>x</k> <k b>1</k> <k b>oops</k> <k z>extra</k>[s]",
            ),
            value!({"content": null, "calls": [
                {"name": "f", "arguments": {"a": "x", "b": "oops", "z": "extra"}},
            ]})
        );
        assert_eq!(payload(&format, &[f()], options(false, false)), None);
    }

    #[test]
    fn a_shaped_calls_tags_are_its_parameters_tag_and_its_values_ends() {
        let Some(CallSyntax::Shaped(shape)) = stopped().call_syntax().copied() else {
            panic!("a shaped call");
        };
        assert_eq!(shape.tag_texts(), vec!["<k ", "</k>"]);
        let Some(CallSyntax::Shaped(shape)) = named().call_syntax().copied() else {
            panic!("a shaped call");
        };
        assert_eq!(
            CallSyntax::Shaped(shape).tag_texts(),
            vec!["<p ", "</p>"],
            "the header's `n=\"` is the template's text"
        );
    }

    /// The message `output` reads as under `format` after `prompt`, typed by `tools`.
    fn read(format: &Format, tools: &[Tool], prompt: &str, output: &str) -> Value {
        let mut engine = Engine::new(format.clone(), Declared::of(tools));
        let mut out = Events::new();
        let inputs = [
            Input::Prompt {
                token_ids: &[],
                text: prompt,
            },
            Input::Delta {
                token_ids: &[],
                text: output,
                spans: &[],
            },
            Input::End {
                finish: EngineFinish::Stop,
            },
        ];
        for input in inputs {
            engine.feed(input, &mut out).expect("a parse");
        }
        let message = adapt::chat::message(0, &out.drain()).message;
        let calls: Vec<Value> = message
            .tool_calls
            .unwrap_or_default()
            .iter()
            .map(|call| {
                let arguments = call.function.arguments.as_deref().unwrap_or("{}");
                value!({
                    "name": call.function.name,
                    "arguments": serde_json::from_str::<Value>(arguments).expect("JSON"),
                })
            })
            .collect();
        value!({"content": message.content, "calls": calls})
    }

    #[test]
    fn the_samples_of_the_tags_read_back_as_their_calls() {
        let tools = [lookup()];
        let call = value!({"name": "lookup", "arguments": {"q": "Paris", "n": 3, "on": true}});
        let named = named().with_layout(named_layout());
        let sample =
            "@lookup\n<c n=\"lookup\">\n<p q>Paris</p>\n<p n>3</p>\n<p on>true</p>\n</c>\n";
        assert_eq!(
            read(&named, &tools, "", sample),
            value!({"content": null, "calls": [call]})
        );
        let two = format!("{sample}\n{sample}");
        assert_eq!(
            read(&named, &tools, "", &two),
            value!({"content": null, "calls": [call, call]})
        );
        let answer = named
            .answer_grammar(&value!({"type": "object"}), false)
            .expect("an answer");
        assert_eq!(
            answer.payload(),
            value!({"format": {
                "type": "tag",
                "begin": " <a>",
                "content": {"type": "json_schema", "json_schema": {"type": "object"}},
                "end": "</a>",
            }})
        );
        assert_eq!(
            read(&named, &[], "", " <a>{\"x\": 1}</a>"),
            value!({"content": "{\"x\": 1}", "calls": []})
        );
        let stopped = stopped().with_layout(stopped_layout());
        let f = value!({"name": "f", "arguments": {"a": "x y", "b": 2}});
        assert_eq!(
            read(
                &stopped,
                &[self::f()],
                "",
                "[c]<k a>x y</k> <k b>2</k>[e][c]<k a>x y</k> <k b>2</k>"
            ),
            value!({"content": null, "calls": [f, f]})
        );
    }
}
