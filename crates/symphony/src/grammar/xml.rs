//! MiniMax M3's structural tag: the block of calls as the template writes it, from the table's
//! markers, its separator and the request's tools.
//!
//! The template writes its separator token before every tag. The block is `<tool_call>` and a
//! newline, one invoke or more, then the separator and `</tool_call>`; an invoke is the separator,
//! `<invoke name="NAME">`, its arguments, the separator, `</invoke>` and a newline. Each argument
//! is an element, the separator and `<key>`, its value, the separator and `</key>`, one per
//! property in the schema's order, present when the schema requires the property and otherwise
//! optional. `<tool_call>` triggers the tag, as the terminal that opens the block is the trigger
//! under every syntax, and the separator the template writes before it is admitted, optional,
//! right before the trigger: a forced call's tag is written at once, so the separator the model
//! writes first has to be the tag's own, while the block without it, which the grammar forced
//! until now and which the table reads as well since it drops the separator wherever it stands,
//! stays valid. What precedes that (the model's thought, or the bare close of an empty one, when
//! the prompt leaves the thought to the model; the free text that closes a thought the prompt
//! left open) is the table's ([`Format::grammar`](crate::Format::grammar)), the same under every
//! syntax.
//!
//! A value follows the property's type as the template writes it and the XML assembler reads it
//! ([`tagged::xml`](crate::tagged::xml)): a string as it is, any text that holds neither the
//! separator, nor the element's own closing tag, nor a marker that ends the call; a string `enum`
//! as one of its values; a number as the JSON its schema accepts; a boolean as one of the words the
//! assembler reads, `true`, `false`, `True` or `False`; an object with properties as nested
//! elements, one per property, as deep as the schema goes; an array with items as `<item>`
//! elements, each a value of the items' type; `null` as nothing at all, since the template leaves
//! out an argument whose value is None and writes an item that is None as an empty element. A
//! property that pins no type, an object without properties or an array without items takes any
//! text that holds neither its closing tag nor a marker that ends the call, separators and nested
//! tags included, which the assembler reads as it infers; a schema without properties takes the
//! same for the whole invoke.
//!
//! One invoke writes at most [`NODES`] elements; past that, the values still to write (the rest of
//! an object, a list) are the same any text, so an acyclic schema with a wide fan-out, which the
//! depth cap alone does not bound, stays bounded. The budget is spent depth first, in the schema's
//! order, but an object's later properties each keep one element while an earlier one is written,
//! so what runs short loses its structure from the inside out: the nested objects first, the
//! top-level arguments last. A property typed `null`, written as nothing, costs nothing. The
//! leaves' spellings count against [`LEAF_BYTES`] besides: an enum's words, and a number's schema
//! with the definitions its pointers reach; once they are spent, an enum is text as it is and a
//! number any text, so a value that is large or reused through references cannot multiply the tag.

use std::collections::HashMap;

use openai_protocol::common::Tool;
use serde_json::Value;

use super::{
    schema::{json_bytes, keyword, kind, shape, string_enum, Definitions},
    CallMarkers, Count, Grammar, Tag,
};

/// The words the assembler reads as a boolean.
const BOOLEAN_WORDS: [&str; 4] = ["true", "false", "True", "False"];
/// A list's element tag.
const ITEM: &str = "item";
/// How deep the elements follow the schema; below that, a value is any text.
const DEPTH: usize = 16;
/// How many elements one invoke's grammar may hold; past that, the values still to write are any
/// text. An acyclic schema unfolds as k^d elements for k properties over d levels, which the depth
/// cap alone does not bound; ordinary tool schemas stay far below this. The budget is each invoke's
/// own: a tag for many tools holds up to this many elements per tool. It counts elements, not
/// bytes: a value's own spelling (an enum's words, a number's schema) is counted by
/// [`LEAF_BYTES`]. A value past the budget is any text up to its own closing tag, as one past the
/// depth cap is, so it cannot hold a descendant whose tag spells the same close (the separator
/// stands before inner tags too); the budget makes that fallback reachable at any depth, where the
/// depth cap kept it past the sixteenth.
const NODES: usize = 1024;
/// How many bytes the leaves of one invoke may spell: an enum's words, a number's schema with the
/// definitions its pointers reach. Past that, a string with an enum is text as it is and a number
/// any text, so a value that is large, or reused through references, cannot multiply the tag: a
/// 5,000-word enum reached by a thousand number leaves is written while it fits, not a thousand
/// times. Each invoke's own, as [`NODES`] is, and spent in the schema's order.
const LEAF_BYTES: usize = 256 * 1024;
/// What an enum's word costs beyond its length: the alternative that spells it.
const WORD_FRAMING: usize = 36;

/// The elements one invoke may still write, how many of them are held for the properties after
/// the one being written, at every level of the descent together, and the bytes its leaves may
/// still spell.
struct Budget {
    nodes: usize,
    reserved: usize,
    leaf_bytes: usize,
    /// The serialized size of each root definition a fragment reached, by the entry's address,
    /// worked out once per invoke.
    entry_bytes: HashMap<usize, usize>,
    /// What each number leaf's fragment costs, by the leaf schema's address: `None` for a leaf
    /// the budget refused, which it refuses again, the budget only shrinking.
    fragment_bytes: HashMap<usize, Option<usize>>,
    /// What each enum costs, by its word list's address.
    enum_bytes: HashMap<usize, usize>,
}

impl Budget {
    fn new() -> Self {
        Self {
            nodes: NODES,
            reserved: 0,
            leaf_bytes: LEAF_BYTES,
            entry_bytes: HashMap::new(),
            fragment_bytes: HashMap::new(),
            enum_bytes: HashMap::new(),
        }
    }

    /// What an enum's words cost, worked out once per word list per invoke.
    fn enum_cost(&mut self, words: &[Value]) -> usize {
        let key = words.as_ptr() as usize;
        *self.enum_bytes.entry(key).or_insert_with(|| {
            words
                .iter()
                .filter_map(Value::as_str)
                .map(|word| word.len() + WORD_FRAMING)
                .sum()
        })
    }

    /// What a number leaf's fragment would cost, with the definitions its pointers reach, when
    /// that fits what the leaves may still spell ([`Definitions::reached_within`]); `None` when
    /// it does not. Worked out once per leaf schema per invoke: a leaf refused once is refused
    /// again, the budget only shrinking.
    fn fragment_cost(&mut self, schema: &Value, definitions: Definitions<'_>) -> Option<usize> {
        let key = std::ptr::from_ref(schema) as usize;
        if let Some(known) = self.fragment_bytes.get(&key) {
            return known.filter(|cost| *cost <= self.leaf_bytes);
        }
        let limit = self.leaf_bytes;
        let mut entry_bytes = std::mem::take(&mut self.entry_bytes);
        let found = definitions
            .reached_within(schema, limit, |entry| {
                let key = std::ptr::from_ref(entry) as usize;
                *entry_bytes.entry(key).or_insert_with(|| json_bytes(entry))
            })
            .map(|(cost, _)| cost);
        self.entry_bytes = entry_bytes;
        self.fragment_bytes.insert(key, found);
        found
    }

    /// `cost` bytes off the leaves' budget; `false` when they do not fit, when the caller writes
    /// the value's fallback and the budget stays for a smaller leaf.
    fn takes_bytes(&mut self, cost: usize) -> bool {
        if cost > self.leaf_bytes {
            return false;
        }
        self.leaf_bytes -= cost;
        true
    }

    /// Whether an element is left beyond the ones held for later siblings.
    fn has_an_element(&self) -> bool {
        self.nodes > self.reserved
    }

    /// One element off the budget, which [`Self::has_an_element`] said was there.
    fn take_an_element(&mut self) {
        self.nodes = self.nodes.saturating_sub(1);
    }
}

/// How a value's schema is written: as any text, at the depth cap or where the schema pins no type
/// the syntax has; or by the type it pins, [`kind`]'s word for it, at the depth below. A schema
/// typed `null` is written as nothing, and is neither ([`Writer::how`]).
enum Writes {
    AnyText,
    Typed { kind: &'static str, depth: usize },
}

/// The block of calls for `tools`, written with `separator` before every tag, as many as `count`
/// lets the model write, or `None` for a table whose calls sit in no block.
pub(super) fn calls(
    markers: &CallMarkers<'_>,
    separator: &str,
    tools: &[&Tool],
    count: Count<'_>,
) -> Option<Grammar> {
    let block = markers.block.as_ref()?;
    let writer = Writer {
        separator,
        exits: &markers.call_exits,
    };
    let invokes = tools
        .iter()
        .map(|tool| writer.invoke(markers, &tool.function.name, &tool.function.parameters))
        .collect();
    // The block holds at least one invoke whether or not the request asks for a call: an empty
    // block is no answer.
    let invokes = Count {
        at_least_one: true,
        ..count
    }
    .separated(invokes);
    let block = count.triggered(
        vec![block.open.to_string()],
        vec![Tag::new(
            format!("{}\n", block.open),
            invokes,
            format!("{separator}{}", block.close),
        )],
    );
    if separator.is_empty() {
        return Some(block);
    }
    Some(Grammar::Sequence(vec![
        Grammar::Optional(Box::new(Grammar::ConstString(separator.to_string()))),
        block,
    ]))
}

/// The spelling the table gives: the separator before every tag, and the markers that end a call
/// wherever they stand, which no value may hold.
struct Writer<'a> {
    separator: &'a str,
    exits: &'a [&'a str],
}

impl Writer<'_> {
    /// One invoke: the separator and the opener with the tool's name, the arguments, the
    /// separator, the call's close and a newline.
    fn invoke(&self, markers: &CallMarkers<'_>, name: &str, parameters: &Value) -> Tag {
        let definitions = Definitions::of(parameters);
        let mut budget = Budget::new();
        Tag::new(
            format!("{}{}{name}\">", self.separator, markers.call_open),
            self.members(
                parameters,
                definitions,
                markers.call_close,
                DEPTH,
                &mut budget,
            ),
            format!("{}{}\n", self.separator, markers.call_close),
        )
    }

    /// An object's elements: one per property in the schema's order, each optional unless the
    /// schema requires it, a property typed `null` left out at no cost to the budget; any text up
    /// to `close` for a schema without properties, and for the properties left once the invoke's
    /// element budget is spent. While a property's value is written, one element is held for each
    /// property after it that writes one, so each still gets its own tag.
    fn members(
        &self,
        schema: &Value,
        definitions: Definitions<'_>,
        close: &str,
        depth: usize,
        budget: &mut Budget,
    ) -> Grammar {
        let target = definitions.target_of(schema);
        let properties = keyword(schema, target, "properties")
            .and_then(Value::as_object)
            .filter(|properties| !properties.is_empty());
        let Some(properties) = properties else {
            return self.any_text(close);
        };
        if !budget.has_an_element() {
            // What the loop below would write for the whole object, before any property of it
            // is classified: an object reached once the budget is spent costs nothing to skip.
            return self.any_text(close);
        }
        let required: Vec<&str> = keyword(schema, target, "required")
            .and_then(Value::as_array)
            .map(|names| names.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        // Each property classified once, for the count of those that write (one element is held
        // for each of them after the one being written) and for the writing itself.
        let classified: Vec<(&String, &Value, String, Option<Writes>)> = properties
            .iter()
            .map(|(key, schema)| {
                let writes = Self::how(schema, definitions, depth);
                (key, schema, format!("</{key}>"), writes)
            })
            .collect();
        let mut later = classified
            .iter()
            .filter(|(_, _, _, writes)| writes.is_some())
            .count();
        let mut elements: Vec<Grammar> = Vec::new();
        for (key, schema, own, writes) in classified {
            let Some(writes) = writes else {
                continue;
            };
            if !budget.has_an_element() {
                elements.push(self.any_text(close));
                break;
            }
            budget.take_an_element();
            later = later.saturating_sub(1);
            budget.reserved += later;
            let value = self.value(schema, definitions, &own, writes, budget);
            budget.reserved -= later;
            let element = Grammar::Sequence(vec![
                Grammar::ConstString(format!("{}<{key}>", self.separator)),
                value,
                Grammar::ConstString(format!("{}{own}", self.separator)),
            ]);
            elements.push(if required.contains(&key.as_str()) {
                element
            } else {
                Grammar::Optional(Box::new(element))
            });
        }
        Grammar::Sequence(elements)
    }

    /// How a value's schema is written at `depth`, or `None` for a schema typed `null`, written
    /// as nothing: the one place that decides, for the count of an object's properties that write
    /// and for the writing itself. The kind alone is asked ([`kind`]), so classifying an object or
    /// an array builds no grammar for it.
    fn how(schema: &Value, definitions: Definitions<'_>, depth: usize) -> Option<Writes> {
        let Some(depth) = depth.checked_sub(1) else {
            return Some(Writes::AnyText);
        };
        match kind(schema, definitions) {
            Some("null") => None,
            Some(kind) => Some(Writes::Typed { kind, depth }),
            None => Some(Writes::AnyText),
        }
    }

    /// The value between an element's tags, as [`Self::how`] says it is written: any text, or by
    /// the type the property's schema pins, a string's or a number's grammar [`shape`]'s where
    /// the leaves' byte budget takes it, text as it is or any text where it does not.
    fn value(
        &self,
        schema: &Value,
        definitions: Definitions<'_>,
        close: &str,
        writes: Writes,
        budget: &mut Budget,
    ) -> Grammar {
        let Writes::Typed { kind, depth } = writes else {
            return self.any_text(close);
        };
        match kind {
            "boolean" => words(&BOOLEAN_WORDS),
            "object" => self.members(schema, definitions, close, depth, budget),
            "array" => match keyword(schema, definitions.target_of(schema), "items") {
                Some(items) if budget.has_an_element() => {
                    budget.take_an_element();
                    self.items(items, definitions, depth, budget)
                }
                _ => self.any_text(close),
            },
            "string" => {
                if let Some(words) = string_enum(schema, definitions.target_of(schema)) {
                    let cost = budget.enum_cost(words);
                    if !budget.takes_bytes(cost) {
                        return self.text(close);
                    }
                }
                shape(schema, definitions, || self.text(close))
                    .map_or_else(|| self.text(close), |(_, shaped)| shaped)
            }
            "number" => {
                let Some(cost) = budget.fragment_cost(schema, definitions) else {
                    return self.any_text(close);
                };
                if !budget.takes_bytes(cost) {
                    return self.any_text(close);
                }
                shape(schema, definitions, || self.text(close))
                    .map_or_else(|| self.any_text(close), |(_, shaped)| shaped)
            }
            _ => shape(schema, definitions, || self.text(close))
                .map_or_else(|| self.any_text(close), |(_, shaped)| shaped),
        }
    }

    /// A list's items: any number of `<item>` elements, each a value of the items' type, an item
    /// that is `null` an empty element.
    fn items(
        &self,
        items: &Value,
        definitions: Definitions<'_>,
        depth: usize,
        budget: &mut Budget,
    ) -> Grammar {
        let open = format!("{}<{ITEM}>", self.separator);
        let close = format!("{}</{ITEM}>", self.separator);
        let own = format!("</{ITEM}>");
        let item = match Self::how(items, definitions, depth) {
            Some(writes) => Grammar::Sequence(vec![
                Grammar::ConstString(open),
                self.value(items, definitions, &own, writes, budget),
                Grammar::ConstString(close),
            ]),
            None => Grammar::ConstString(format!("{open}{close}")),
        };
        Grammar::Star(Box::new(item))
    }

    /// A string written as it is: any text that holds neither the separator, nor `close`, nor a
    /// marker that ends the call.
    fn text(&self, close: &str) -> Grammar {
        self.excluding(&[self.separator, close])
    }

    /// A value the schema does not shape: any text that holds neither `close` nor a marker that
    /// ends the call, separators and nested tags included.
    fn any_text(&self, close: &str) -> Grammar {
        self.excluding(&[close])
    }

    /// Any text that holds none of `first` and none of the markers that end the call, each once.
    fn excluding(&self, first: &[&str]) -> Grammar {
        let mut excludes: Vec<String> = Vec::new();
        for text in first.iter().chain(self.exits.iter()) {
            if !excludes.iter().any(|known| known == text) {
                excludes.push(text.to_string());
            }
        }
        Grammar::AnyText { excludes }
    }
}

/// One of the words.
fn words(words: &[&str]) -> Grammar {
    Grammar::Or(
        words
            .iter()
            .map(|word| Grammar::ConstString(word.to_string()))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;
    use serde_json::json as value;

    use super::*;
    use crate::formats;

    const SEP: &str = "]<]minimax[>[";

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

    fn weather_tools() -> Vec<Tool> {
        vec![
            tool(
                "get_weather",
                value!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer", "minimum": 1},
                        "units": {"type": "string", "enum": ["c", "f"]},
                    },
                    "required": ["city", "days"],
                }),
            ),
            tool("ping", value!({"type": "object"})),
        ]
    }

    /// The markers that end a call wherever they stand, in the order of the table's rows.
    fn exits() -> Vec<&'static str> {
        vec!["</invoke>", "<invoke name=\"", "</tool_call>"]
    }

    /// The table's terminals in order, the separator included: what the thought's text may not
    /// write.
    fn terminals() -> Vec<&'static str> {
        vec![
            "<mm:think>",
            "</mm:think>",
            "<tool_call>",
            "</tool_call>",
            "<invoke name=\"",
            "</invoke>",
            SEP,
        ]
    }

    /// The block as the tag writes it: the separator, optional, then the triggered tag.
    fn block_of(triggered: Value) -> Value {
        value!({"type": "sequence", "elements": [
            {"type": "optional", "content": {"type": "const_string", "value": SEP}},
            triggered,
        ]})
    }

    /// What the tag admits before the block when the prompt leaves the thought to the model: the
    /// thought, or the bare close of an empty one, or nothing.
    fn thought_first() -> Value {
        value!({"type": "optional", "content": {"type": "or", "elements": [
            {
                "type": "tag",
                "begin": "<mm:think>",
                "content": {"type": "any_text", "excludes": terminals()},
                "end": "</mm:think>",
            },
            {"type": "const_string", "value": "</mm:think>"},
        ]}})
    }

    /// An element: the separator and the key's tag, the value, the separator and the closing tag.
    fn element(key: &str, value: Value) -> Value {
        value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": format!("{SEP}<{key}>")},
            value,
            {"type": "const_string", "value": format!("{SEP}</{key}>")},
        ]})
    }

    /// A string's text inside the element `key`.
    fn text(key: &str) -> Value {
        let mut excludes = vec![SEP, ""];
        let close = format!("</{key}>");
        excludes[1] = &close;
        excludes.extend(exits());
        value!({"type": "any_text", "excludes": excludes})
    }

    /// The first invoke's elements of the block for `tools`.
    fn first_invoke(tools: &[Tool]) -> Value {
        let payload = formats::minimax_m3()
            .grammar(tools, true, false)
            .expect("a grammar")
            .payload();
        payload["format"]["elements"][1]["elements"][1]["tags"][0]["content"]["tags"][0].clone()
    }

    #[test]
    fn minimax_m3_writes_the_block_with_its_separator_before_every_tag() {
        // The recorded shape: `]<]minimax[>[<tool_call>\n`, each invoke
        // `]<]minimax[>[<invoke name="NAME">`, its elements each `]<]minimax[>[<key>VALUE` and
        // `]<]minimax[>[</key>`, then `]<]minimax[>[</invoke>\n`, and `]<]minimax[>[</tool_call>`.
        let payload = formats::minimax_m3()
            .grammar(&weather_tools(), true, false)
            .expect("a grammar")
            .payload();
        let weather = value!({
            "type": "tag",
            "begin": format!("{SEP}<invoke name=\"get_weather\">"),
            "content": {"type": "sequence", "elements": [
                element("city", text("city")),
                element(
                    "days",
                    value!({
                        "type": "json_schema",
                        "json_schema": {"type": "integer", "minimum": 1},
                    }),
                ),
                {"type": "optional", "content": element(
                    "units",
                    value!({"type": "or", "elements": [
                        {"type": "const_string", "value": "c"},
                        {"type": "const_string", "value": "f"},
                    ]}),
                )},
            ]},
            "end": format!("{SEP}</invoke>\n"),
        });
        let ping = value!({
            "type": "tag",
            "begin": format!("{SEP}<invoke name=\"ping\">"),
            "content": {"type": "any_text", "excludes": exits()},
            "end": format!("{SEP}</invoke>\n"),
        });
        assert_eq!(
            payload,
            value!({"format": {"type": "sequence", "elements": [thought_first(), block_of(value!({
                "type": "triggered_tags",
                "triggers": ["<tool_call>"],
                "tags": [{
                    "type": "tag",
                    "begin": "<tool_call>\n",
                    "content": {
                        "type": "tags_with_separator",
                        "tags": [weather, ping],
                        "separator": "",
                        "at_least_one": true,
                    },
                    "end": format!("{SEP}</tool_call>"),
                }],
                "at_least_one": true,
            }))]}})
        );
    }

    #[test]
    fn nested_objects_are_elements_lists_are_items_and_a_null_is_left_out() {
        let invoke = first_invoke(&[tool(
            "t",
            value!({
                "type": "object",
                "properties": {
                    "where": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}, "label": {"type": "string"}},
                        "required": ["x"],
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "flag": {"type": "boolean"},
                    "nothing": {"type": "null"},
                    "free": {"type": "object"},
                    "any": {},
                },
                "required": ["where", "tags", "flag", "nothing", "free", "any"],
            }),
        )]);
        let elements = invoke["content"]["elements"]
            .as_array()
            .expect("five elements");
        assert_eq!(elements.len(), 5, "the null is left out");
        assert_eq!(
            elements[0],
            element(
                "where",
                value!({"type": "sequence", "elements": [
                    element(
                        "x",
                        value!({"type": "json_schema", "json_schema": {"type": "number"}}),
                    ),
                    {"type": "optional", "content": element("label", text("label"))},
                ]}),
            )
        );
        assert_eq!(
            elements[1],
            element(
                "tags",
                value!({"type": "star", "content": element("item", text("item"))}),
            )
        );
        assert_eq!(
            elements[2],
            element(
                "flag",
                value!({"type": "or", "elements": [
                    {"type": "const_string", "value": "true"},
                    {"type": "const_string", "value": "false"},
                    {"type": "const_string", "value": "True"},
                    {"type": "const_string", "value": "False"},
                ]}),
            )
        );
        for (slot, key) in [(&elements[3], "free"), (&elements[4], "any")] {
            let mut excludes = vec![format!("</{key}>")];
            excludes.extend(exits().into_iter().map(str::to_string));
            assert_eq!(
                *slot,
                element(key, value!({"type": "any_text", "excludes": excludes})),
                "{key} takes any text"
            );
        }
    }

    #[test]
    fn a_reference_lends_its_targets_properties_and_a_list_of_nulls_holds_empty_items() {
        let invoke = first_invoke(&[tool(
            "t",
            value!({
                "type": "object",
                "$defs": {
                    "Point": {"type": "object", "properties": {"x": {"type": "integer"}}},
                    "Count": {"type": "integer"},
                },
                "properties": {
                    "at": {"$ref": "#/$defs/Point"},
                    "count": {"$ref": "#/$defs/Count"},
                    "holes": {"type": "array", "items": {"type": "null"}},
                },
                "required": ["at", "count", "holes"],
            }),
        )]);
        let elements = &invoke["content"]["elements"];
        assert_eq!(
            elements[0],
            element(
                "at",
                value!({"type": "sequence", "elements": [
                    {"type": "optional", "content": element(
                        "x",
                        value!({"type": "json_schema", "json_schema": {"type": "integer"}}),
                    )},
                ]}),
            )
        );
        assert_eq!(
            elements[1]["elements"][1],
            value!({"type": "json_schema", "json_schema": {
                "$ref": "#/$defs/Count",
                "$defs": {"Count": {"type": "integer"}},
            }})
        );
        assert_eq!(
            elements[2],
            element(
                "holes",
                value!({"type": "star", "content": {
                    "type": "const_string",
                    "value": format!("{SEP}<item>{SEP}</item>"),
                }}),
            )
        );
    }

    #[test]
    fn the_thought_open_takes_the_prefix_and_closed_admits_the_models_thought_first() {
        // The prompt that leaves the thought to the model (the template's default): the thought,
        // or the bare close of an empty one, or nothing, then the block, which a forced call
        // writes at once. The prompt that opened the thought: free text that writes neither a
        // call marker nor the separator but may close the thought, nothing owed at its end, then
        // the block as it is.
        let tools = weather_tools();
        let closed = formats::minimax_m3()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let block = closed["format"]["elements"][1].clone();
        assert_eq!(
            closed["format"],
            value!({"type": "sequence", "elements": [thought_first(), block]})
        );
        assert_eq!(block, block_of(block["elements"][1].clone()));
        let triggered = &block["elements"][1];
        assert_eq!(triggered["type"], "triggered_tags");
        assert_eq!(triggered["triggers"], value!(["<tool_call>"]));
        assert_eq!(triggered["at_least_one"], true);
        let open = formats::minimax_m3()
            .grammar(&tools, true, true)
            .expect("a grammar")
            .payload();
        assert_eq!(
            open["format"],
            value!({"type": "sequence", "elements": [
                {"type": "any_text", "excludes": [
                    "<tool_call>", "</tool_call>", "<invoke name=\"", "</invoke>", SEP,
                ]},
                block,
            ]})
        );
        let either = formats::minimax_m3()
            .grammar(&tools, false, false)
            .expect("a grammar")
            .payload();
        assert_eq!(either["format"]["elements"][0], thought_first());
        let either_block = &either["format"]["elements"][1]["elements"][1];
        assert_eq!(either_block["at_least_one"], false);
        assert_eq!(
            either_block["tags"][0]["content"]["at_least_one"], true,
            "a block holds at least one invoke"
        );
        let mut same = block;
        same["elements"][1]["at_least_one"] = Value::Bool(false);
        assert_eq!(either["format"]["elements"][1], same);
    }

    /// A schema of `k` properties per object over `d` levels, the leaves strings: k^d leaves.
    fn fan_out(k: usize, d: usize) -> Value {
        if d == 0 {
            return value!({"type": "string"});
        }
        let properties: serde_json::Map<String, Value> = (0..k)
            .map(|i| (format!("p{i}"), fan_out(k, d - 1)))
            .collect();
        let required: Vec<String> = (0..k).map(|i| format!("p{i}")).collect();
        value!({"type": "object", "properties": properties, "required": required})
    }

    /// The elements a payload writes: one per opening tag the writer spells.
    fn elements_written(node: &Value) -> usize {
        match node {
            Value::Object(object) => {
                let opener = object.get("type").and_then(Value::as_str) == Some("const_string")
                    && object
                        .get("value")
                        .and_then(Value::as_str)
                        .is_some_and(|text| {
                            text.starts_with(&format!("{SEP}<"))
                                && !text.starts_with(&format!("{SEP}</"))
                        });
                usize::from(opener) + object.values().map(elements_written).sum::<usize>()
            }
            Value::Array(items) => items.iter().map(elements_written).sum(),
            _ => 0,
        }
    }

    /// The any-text runs a payload holds whose excludes do not name the separator: an untyped
    /// value's, and the rest of an object past the budget (a typed string's text excludes it).
    fn any_text_runs(node: &Value) -> usize {
        match node {
            Value::Object(object) => {
                let free = object.get("type").and_then(Value::as_str) == Some("any_text")
                    && object
                        .get("excludes")
                        .and_then(Value::as_array)
                        .is_some_and(|excludes| !excludes.iter().any(|e| e == SEP));
                usize::from(free) + object.values().map(any_text_runs).sum::<usize>()
            }
            Value::Array(items) => items.iter().map(any_text_runs).sum(),
            _ => 0,
        }
    }

    /// The keys of an invoke's elements, in order, an optional element unwrapped.
    fn element_keys(invoke: &Value) -> Vec<String> {
        invoke["content"]["elements"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|element| {
                let element = if element["type"] == "optional" {
                    &element["content"]
                } else {
                    element
                };
                let opener = element["elements"][0]["value"].as_str()?;
                Some(
                    opener
                        .trim_start_matches(SEP)
                        .trim_start_matches('<')
                        .trim_end_matches('>')
                        .to_string(),
                )
            })
            .collect()
    }

    #[test]
    fn a_wide_fan_out_stays_within_the_element_budget() {
        // 8 + 64 + 512 + 4096 = 4680 elements unbounded: the budget, the rest of an object past
        // it any text, and the payload small; a schema just under the budget is written in full.
        let wide = first_invoke(&[tool("t", fan_out(8, 4))]);
        assert_eq!(elements_written(&wide), 1024, "the budget, not 4680");
        assert!(any_text_runs(&wide) >= 1);
        assert!(
            wide.to_string().len() < 320_000,
            "{} bytes",
            wide.to_string().len()
        );
        let deep = first_invoke(&[tool("t", fan_out(4, 5))]); // 1364 unbounded
        assert_eq!(elements_written(&deep), 1024);
        let under = first_invoke(&[tool("t", fan_out(31, 2))]); // 31 + 961 = 992
        assert_eq!(elements_written(&under), 992, "written in full");
        assert_eq!(any_text_runs(&under), 0, "nothing fell back");
    }

    #[test]
    fn a_wide_property_leaves_each_later_sibling_its_element_and_a_null_costs_none() {
        // The budget is spent depth first, but a property written first cannot take the elements
        // its later siblings need: each keeps one, so the top-level arguments stay typed, a `null`
        // among them holds none, and a `null` after the last writer adds no any-text tail.
        let mut properties = serde_json::Map::new();
        properties.insert("config".to_string(), fan_out(8, 4));
        properties.insert("none".to_string(), value!({"type": "null"}));
        properties.insert("path".to_string(), value!({"type": "string"}));
        properties.insert("count".to_string(), value!({"type": "integer"}));
        properties.insert("zz".to_string(), value!({"type": "null"}));
        let invoke = first_invoke(&[tool(
            "t",
            value!({
                "type": "object",
                "properties": properties,
                "required": ["config", "none", "path", "count", "zz"],
            }),
        )]);
        assert_eq!(elements_written(&invoke), 1024, "the budget, not 4682");
        assert_eq!(element_keys(&invoke), ["config", "path", "count"]);
        let elements = &invoke["content"]["elements"];
        assert_eq!(elements[1], element("path", text("path")));
        assert_eq!(
            elements[2],
            element(
                "count",
                value!({"type": "json_schema", "json_schema": {"type": "integer"}})
            )
        );
        assert!(
            any_text_runs(&elements[0]) >= 1,
            "the wide object ran short"
        );
        assert_eq!(
            any_text_runs(&elements[1]) + any_text_runs(&elements[2]),
            0,
            "no tail after the typed arguments"
        );

        // 1,023 strings, a null, a required boolean: 1,024 elements, the boolean typed; 1,024
        // strings then a null: no any-text run.
        let mut properties = serde_json::Map::new();
        for i in 0..1023 {
            properties.insert(format!("s{i:04}"), value!({"type": "string"}));
        }
        properties.insert("unull".to_string(), value!({"type": "null"}));
        properties.insert("zflag".to_string(), value!({"type": "boolean"}));
        let required: Vec<String> = properties.keys().cloned().collect();
        let invoke = first_invoke(&[tool(
            "t",
            value!({"type": "object", "properties": properties, "required": required}),
        )]);
        assert_eq!(elements_written(&invoke), 1024);
        assert_eq!(any_text_runs(&invoke), 0);
        assert_eq!(
            invoke["content"]["elements"][1023],
            element(
                "zflag",
                value!({"type": "or", "elements": [
                    {"type": "const_string", "value": "true"},
                    {"type": "const_string", "value": "false"},
                    {"type": "const_string", "value": "True"},
                    {"type": "const_string", "value": "False"},
                ]})
            )
        );
        let mut properties = serde_json::Map::new();
        for i in 0..1024 {
            properties.insert(format!("s{i:04}"), value!({"type": "string"}));
        }
        properties.insert("zz".to_string(), value!({"type": "null"}));
        let required: Vec<String> = properties.keys().cloned().collect();
        let invoke = first_invoke(&[tool(
            "t",
            value!({"type": "object", "properties": properties, "required": required}),
        )]);
        assert_eq!(elements_written(&invoke), 1024);
        assert_eq!(any_text_runs(&invoke), 0, "no tail after the last string");
    }

    #[test]
    fn lists_count_against_the_budget_and_every_property_but_a_null_one_has_its_element() {
        let lists = first_invoke(&[tool(
            "t",
            value!({
                "type": "object",
                "properties": {"a": {"type": "array", "items": {
                    "type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"],
                }}},
                "required": ["a"],
            }),
        )]);
        assert_eq!(
            elements_written(&lists),
            3,
            "the property, its item, the item's property"
        );
        // 1,023 strings, then a list: the list is the 1,024th element, its item past the budget.
        let mut properties = serde_json::Map::new();
        for i in 0..1023 {
            properties.insert(format!("s{i:04}"), value!({"type": "string"}));
        }
        properties.insert(
            "zlist".to_string(),
            value!({"type": "array", "items": {"type": "string"}}),
        );
        let required: Vec<String> = properties.keys().cloned().collect();
        let edge = first_invoke(&[tool(
            "t",
            value!({"type": "object", "properties": properties, "required": required}),
        )]);
        assert_eq!(elements_written(&edge), 1024);
        assert_eq!(any_text_runs(&edge), 1, "the list's content is any text");

        // Exactly the properties that are not `null` have elements, whatever the shape: a
        // reference chain, a pointer at nothing, a reference that names itself, `null` written
        // out or behind a reference; and a nest past the depth cap keeps an element at every
        // level, the innermost any text.
        let shapes = first_invoke(&[tool(
            "t",
            value!({
                "type": "object",
                "$defs": {
                    "S": {"type": "string"},
                    "N": {"type": "null"},
                    "Chain1": {"$ref": "#/$defs/Chain2"},
                    "Chain2": {"$ref": "#/$defs/S"},
                    "Loop": {"$ref": "#/$defs/Loop"},
                },
                "properties": {
                    "a_string": {"type": "string"},
                    "b_chain": {"$ref": "#/$defs/Chain1"},
                    "c_missing": {"$ref": "#/$defs/Missing"},
                    "d_loop": {"$ref": "#/$defs/Loop"},
                    "e_null": {"type": "null"},
                    "f_null_ref": {"$ref": "#/$defs/N"},
                    "g_last": {"type": "boolean"},
                },
                "required": [
                    "a_string", "b_chain", "c_missing", "d_loop", "e_null", "f_null_ref", "g_last",
                ],
            }),
        )]);
        assert_eq!(
            element_keys(&shapes),
            ["a_string", "b_chain", "c_missing", "d_loop", "g_last"]
        );
        assert_eq!(elements_written(&shapes), 5);
        let nest = first_invoke(&[tool("t", fan_out(1, 17))]);
        assert_eq!(elements_written(&nest), 17);
        assert_eq!(any_text_runs(&nest), 1, "the innermost, past the depth cap");
    }

    /// The `json_schema` leaves a payload holds, in order.
    fn fragments(node: &Value) -> Vec<Value> {
        match node {
            Value::Object(object) => {
                if object.get("type").and_then(Value::as_str) == Some("json_schema") {
                    return vec![object["json_schema"].clone()];
                }
                object.values().flat_map(fragments).collect()
            }
            Value::Array(items) => items.iter().flat_map(fragments).collect(),
            _ => Vec::new(),
        }
    }

    /// How many `or` nodes of `width` alternatives a payload holds: the enums spelled out.
    fn enums_spelled(node: &Value, width: usize) -> usize {
        match node {
            Value::Object(object) => {
                let spelled = object.get("type").and_then(Value::as_str) == Some("or")
                    && object
                        .get("elements")
                        .and_then(Value::as_array)
                        .is_some_and(|elements| elements.len() == width);
                usize::from(spelled)
                    + object
                        .values()
                        .map(|value| enums_spelled(value, width))
                        .sum::<usize>()
            }
            Value::Array(items) => items.iter().map(|item| enums_spelled(item, width)).sum(),
            _ => 0,
        }
    }

    #[test]
    fn number_fragments_past_the_byte_budget_are_any_text() {
        // A 5,000-word enum under `$defs`, reached by 32 x 32 number leaves through `not`: it
        // travelled with every fragment (about 110 MB of tag from a 113 KB request); it travels
        // while the budget lasts, the leaves after that are any text, and the tag stays well
        // under a megabyte.
        let words: Vec<String> = (0..5000).map(|i| format!("option_number_{i:05}")).collect();
        let row: serde_json::Map<String, Value> = (0..32)
            .map(|i| {
                (
                    format!("n{i:02}"),
                    value!({"type": "number", "not": {"$ref": "#/$defs/Big"}}),
                )
            })
            .collect();
        let row_required: Vec<String> = row.keys().cloned().collect();
        let root: serde_json::Map<String, Value> = (0..32)
            .map(|i| (format!("r{i:02}"), value!({"$ref": "#/$defs/Row"})))
            .collect();
        let required: Vec<String> = root.keys().cloned().collect();
        let invoke = first_invoke(&[tool(
            "t",
            value!({
                "type": "object",
                "$defs": {
                    "Big": {"type": "string", "enum": words},
                    "Row": {"type": "object", "properties": row, "required": row_required},
                },
                "properties": root,
                "required": required,
            }),
        )]);
        let payload = invoke.to_string();
        assert!(payload.len() < 700_000, "{} bytes", payload.len());
        let leaves = fragments(&invoke);
        assert!(
            (1..=3).contains(&leaves.len()),
            "the definition travels while the budget lasts: {} fragments",
            leaves.len()
        );
        assert!(leaves.iter().all(|leaf| leaf["$defs"]["Big"]["enum"]
            .as_array()
            .is_some_and(|w| w.len() == 5000)));
        assert_eq!(elements_written(&invoke), 1024, "the element budget stands");
        assert!(
            any_text_runs(&invoke) >= 900,
            "the number leaves past the budget are any text: {}",
            any_text_runs(&invoke)
        );
    }

    #[test]
    fn a_hub_of_thousands_of_definitions_is_walked_within_the_budget() {
        // A hub that reaches four thousand small definitions, from a thousand number leaves: the
        // walk behind each leaf stops at the budget and is remembered per leaf, so the tag stays
        // small and the build quick however many definitions a refused leaf would have reached.
        let mut defs = serde_json::Map::new();
        let branches: Vec<Value> = (0..4000)
            .map(|i| {
                defs.insert(format!("d{i:05}"), value!({}));
                value!({"$ref": format!("#/$defs/d{i:05}")})
            })
            .collect();
        defs.insert("Hub".to_string(), value!({"anyOf": branches}));
        let row: serde_json::Map<String, Value> = (0..32)
            .map(|i| {
                (
                    format!("n{i:02}"),
                    value!({"type": "number", "not": {"$ref": "#/$defs/Hub"}}),
                )
            })
            .collect();
        let row_required: Vec<String> = row.keys().cloned().collect();
        defs.insert(
            "Row".to_string(),
            value!({"type": "object", "properties": row, "required": row_required}),
        );
        let root: serde_json::Map<String, Value> = (0..32)
            .map(|i| (format!("r{i:02}"), value!({"$ref": "#/$defs/Row"})))
            .collect();
        let required: Vec<String> = root.keys().cloned().collect();
        let started = std::time::Instant::now();
        let invoke = first_invoke(&[tool(
            "t",
            value!({"type": "object", "$defs": defs, "properties": root, "required": required}),
        )]);
        let took = started.elapsed();
        let payload = invoke.to_string();
        assert!(payload.len() < 700_000, "{} bytes", payload.len());
        assert_eq!(elements_written(&invoke), 1024);
        let leaves = fragments(&invoke);
        assert!(
            (1..=3).contains(&leaves.len()),
            "{} fragments",
            leaves.len()
        );
        assert!(leaves.iter().all(|leaf| leaf["$defs"]["Hub"].is_object()));
        assert!(
            took.as_secs() < 5,
            "bounded and remembered per leaf: {took:?}"
        );
    }

    #[test]
    fn an_enum_reused_through_references_stops_at_the_byte_budget() {
        // One copy is 2,000 words of 11 bytes and their framing, about 94 KB: two fit the budget,
        // the third does not, and the properties after that are text as it is, strings still.
        let words: Vec<String> = (0..2000).map(|i| format!("choice_{i:04}")).collect();
        let properties: serde_json::Map<String, Value> = (0..30)
            .map(|i| (format!("e{i:02}"), value!({"$ref": "#/$defs/E"})))
            .collect();
        let required: Vec<String> = properties.keys().cloned().collect();
        let invoke = first_invoke(&[tool(
            "t",
            value!({
                "type": "object",
                "$defs": {"E": {"type": "string", "enum": words}},
                "properties": properties,
                "required": required,
            }),
        )]);
        let payload = invoke.to_string();
        assert_eq!(enums_spelled(&invoke, 2000), 2, "{} bytes", payload.len());
        assert!(payload.len() < 400_000, "{} bytes", payload.len());
        let elements = &invoke["content"]["elements"];
        assert_eq!(
            elements[2],
            element("e02", text("e02")),
            "typed text, no enum"
        );
        assert_eq!(elements[29], element("e29", text("e29")));
        assert_eq!(any_text_runs(&invoke), 0, "nothing untyped");
    }

    #[test]
    fn a_recursive_definition_of_many_properties_keeps_every_top_level_element() {
        // `D` has k properties, each a reference to `D`, and so has the root: k^16 unbounded.
        // Under the budget, 1,024 elements, every top-level property with its own; an object
        // reached once the budget is spent is any text before its properties are classified, so
        // the walk is a few k-squared, not the budget times that.
        let k = 300;
        let mut properties = serde_json::Map::new();
        for i in 0..k {
            properties.insert(format!("p{i:03}"), value!({"$ref": "#/$defs/D"}));
        }
        let required: Vec<String> = properties.keys().cloned().collect();
        let definition = value!({"type": "object", "properties": properties, "required": required});
        let invoke = first_invoke(&[tool(
            "t",
            value!({
                "type": "object",
                "$defs": {"D": definition},
                "properties": properties,
                "required": required,
            }),
        )]);
        assert_eq!(elements_written(&invoke), 1024);
        assert_eq!(
            element_keys(&invoke).len(),
            k,
            "every top-level property has its element"
        );
    }
}
