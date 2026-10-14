//! Gemma's calls, written from the table's markers as Gemma 4's `format_argument` macro writes
//! them: `<|tool_call>call:NAME{`, the arguments, `}<tool_call|>`, in the table's spelling of the
//! string token and the null word (FunctionGemma: `<escape>`, `None`, between its own markers).
//! The arguments are
//! `key:value` pairs joined by `,` with no space, one per property of the tool's schema, in the
//! order of the keys ignoring case (the template's `dictsort`), each optional unless the schema
//! requires it; the comma stands between two present pairs only, so the pairs take the [`joined`]
//! rule. A value is spelled as the template spells it: a string between two `<|"|>` tokens, its
//! text as it is; a string `enum` as one of its values between the tokens; `true`, `false` and
//! `null` bare; a number as the JSON the property's schema accepts; an object as `{`, its own
//! properties in this same spelling, `}`; a list as `[`, its items in this spelling joined by `,`,
//! `]`. A property whose `type` is a list, or an `anyOf` or `oneOf`, takes one of its branches'
//! spellings. A property that pins no type takes a string or a bare literal, and a schema without
//! properties any pairs of a bare key and such a value: nothing nested, since the tag format has
//! no rule that names itself. For the same reason the tag is bounded, each bound falling back to
//! a string or a bare literal: a definition that names itself is written once (the `$ref`
//! pointers on the way down are kept); a value nests at most [`DEPTH`] levels, which bounds the
//! walk's recursion; a tag spells at most [`VALUES`] values, which bounds its work however wide
//! the schema; and an object's members take the full spelling where it fits [`NODES`] once
//! [`joined`] has copied each pair into the starts before it, estimated before the copies are
//! made with a JSON schema leaf counting by its size, else any pairs. An object's members hold
//! every object below it, so the cap is the whole tag's, and a request's schema gives a finite
//! tag of bounded size in bounded time.

use openai_protocol::common::Tool;
use serde_json::Value;

use super::{
    schema::{joined, joined_nodes, keyword, shape, Definitions},
    CallMarkers, Grammar, Tag,
};
use crate::gemma::Spelling;

/// A bare literal as the template writes one, but for the null word: a number as Python prints
/// it, `true` or `false`; [`Ends::literal`] adds the spelling's null word, which is letters.
const LITERALS: &str = r"-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?|true|false";
/// A key written as it is: anything but the bytes that end or open a member.
const KEY: &str = r"[^:,{}\[\]<]+";
/// How deep a value may nest before its spelling gives way to a string or a bare literal: the walk
/// recurses once per level on the request's thread, and a chain of distinct definitions could
/// otherwise nest as deep as the value budget, which overflows a two-megabyte stack in a debug
/// build; no recorded tool nests further.
const DEPTH: usize = 8;
/// How many nodes an object's members may take, the objects below it included, so the whole tag
/// stays within it: the full spelling copies each pair into the starts before it ([`joined`]), and
/// an object of objects would multiply by that at every level without the bound. Twenty-nine
/// optional strings still take the full spelling; thirty take any pairs.
const NODES: usize = 4_096;
/// The JSON schema types, once each: the branches a `type` list can have.
const TYPES: [&str; 7] = [
    "string", "integer", "number", "boolean", "null", "object", "array",
];
/// How many values a tag spells before the rest give way to a string or a bare literal: the walk
/// down a schema is depth first and eager, so this bounds its work whatever the schema's shape.
const VALUES: usize = 256;

/// What the walk down a schema carries: the `$ref` pointers on the way down, so a definition that
/// names itself is written once, the levels left of [`DEPTH`] and the values left to spell.
struct Walk {
    expanding: Vec<String>,
    depth: usize,
    values: usize,
}

impl Walk {
    fn new() -> Self {
        Self {
            expanding: Vec::new(),
            depth: DEPTH,
            values: VALUES,
        }
    }
}

/// The calls for `tools`: one tag per tool, triggered by the call's opener, `call:NAME{` after it,
/// the arguments in `spelling`, then `}` and the call's close.
pub(super) fn calls(
    markers: &CallMarkers<'_>,
    spelling: Spelling,
    tools: &[&Tool],
    at_least_one: bool,
) -> Grammar {
    let ends = Ends {
        call_exits: &markers.call_exits,
        spelling,
    };
    let tags = tools
        .iter()
        .map(|tool| {
            let parameters = &tool.function.parameters;
            Tag::new(
                format!("{}call:{}{{", markers.call_open, tool.function.name),
                ends.members(parameters, Definitions::of(parameters), &mut Walk::new()),
                format!("}}{}", markers.call_close),
            )
        })
        .collect();
    Grammar::TriggeredTags {
        triggers: vec![markers.call_open.to_string()],
        tags,
        at_least_one,
    }
}

/// What ends a string's text besides its own token: the terminals the table has a row for out of
/// the arguments state, which the engine takes wherever they stand inside a call; and the table's
/// spelling of the token and the null word.
struct Ends<'a> {
    call_exits: &'a [&'a str],
    spelling: Spelling,
}

impl Ends<'_> {
    /// The members of the object `schema` describes: one `key:value` pair per property, the keys
    /// in sorted order ignoring case, joined by `,`, where the spelling fits [`NODES`], estimated
    /// from the pairs' weights before [`joined`] copies them; any pairs where it does not, and
    /// for a schema without properties. The properties are read through the schema's `$ref`, as
    /// every keyword is; `walk` carries the `$ref` pointers on the way down and the values left
    /// to spell.
    fn members(&self, schema: &Value, definitions: Definitions<'_>, walk: &mut Walk) -> Grammar {
        let target = definitions.target_of(schema);
        let properties = keyword(schema, target, "properties")
            .and_then(Value::as_object)
            .filter(|properties| !properties.is_empty());
        let Some(properties) = properties else {
            return self.any_members();
        };
        let required: Vec<&str> = keyword(schema, target, "required")
            .and_then(Value::as_array)
            .map(|names| names.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let mut sorted: Vec<(&String, &Value)> = properties.iter().collect();
        sorted.sort_by_cached_key(|(key, _)| key.to_lowercase());
        let pairs: Vec<(Grammar, bool)> = sorted
            .into_iter()
            .map(|(key, property)| {
                let pair = Grammar::Sequence(vec![
                    Grammar::ConstString(format!("{key}:")),
                    self.value(property, definitions, walk),
                ]);
                (pair, required.contains(&key.as_str()))
            })
            .collect();
        let weights: Vec<(usize, bool)> = pairs
            .iter()
            .map(|(pair, required)| (pair.nodes(), *required))
            .collect();
        if joined_nodes(&weights) <= NODES {
            joined(&pairs, ",")
        } else {
            self.any_members()
        }
    }

    /// A value as the template spells one, or a string or a bare literal where the value nests
    /// deeper than [`DEPTH`], the tag has spelled [`VALUES`] values already, or its `$ref` is one
    /// on the way down: a definition that names itself is written once.
    fn value(&self, schema: &Value, definitions: Definitions<'_>, walk: &mut Walk) -> Grammar {
        let (Some(depth), Some(values)) = (walk.depth.checked_sub(1), walk.values.checked_sub(1))
        else {
            return self.any_value();
        };
        walk.values = values;
        let pointer = schema.get("$ref").and_then(Value::as_str);
        if let Some(pointer) = pointer {
            if walk.expanding.iter().any(|seen| seen == pointer) {
                return self.any_value();
            }
            walk.expanding.push(pointer.to_string());
        }
        walk.depth = depth;
        let grammar = self.typed(schema, definitions, walk);
        walk.depth += 1;
        if pointer.is_some() {
            walk.expanding.pop();
        }
        grammar
    }

    /// A value by what its schema says: one of its branches where the `type` is a list (each known
    /// type once, the schema with that one type and the `enum` members of that type, none where
    /// the `enum` has no member of the type) or an `anyOf` or `oneOf` stands; a string `enum` as
    /// one of its values between the tokens; a string between the tokens; `true`, `false` and
    /// `null` as the words; a number as the JSON the property's schema accepts; an object and a
    /// list in this same spelling; a string or a bare literal where the schema pins no type, or a
    /// pointer in it resolves to nothing. Every keyword is read through the property's `$ref`, as
    /// [`shape`] reads them.
    fn typed(&self, schema: &Value, definitions: Definitions<'_>, walk: &mut Walk) -> Grammar {
        let target = definitions.target_of(schema);
        if let Some(types) = keyword(schema, target, "type").and_then(Value::as_array) {
            // Each branch is the schema with that one type, the property's own keyword winning,
            // and the `enum` narrowed to the members of that type (OpenAI's strict mode writes a
            // nullable enum as `"type": ["string", "null"], "enum": ["F", "C", null]`); a type
            // no member has admits nothing, since `enum` constrains every value, and has no
            // branch. The known types once each, so a list cannot cost more than seven clones.
            let options = keyword(schema, target, "enum").and_then(Value::as_array);
            let mut names: Vec<&str> = Vec::new();
            for one in types.iter().filter_map(Value::as_str) {
                if TYPES.contains(&one) && !names.contains(&one) {
                    names.push(one);
                }
            }
            let branches: Vec<Grammar> = names
                .into_iter()
                .filter_map(|one| {
                    let mut single = schema.clone();
                    if let Value::Object(map) = &mut single {
                        map.insert("type".to_string(), Value::String(one.to_string()));
                        if let Some(options) = options {
                            let narrowed: Vec<Value> = options
                                .iter()
                                .filter(|option| is_of_type(option, one))
                                .cloned()
                                .collect();
                            if narrowed.is_empty() {
                                return None;
                            }
                            map.insert("enum".to_string(), Value::Array(narrowed));
                        }
                    }
                    Some(self.typed(&single, definitions, walk))
                })
                .collect();
            return one_of(branches, || self.any_value());
        }
        for union in ["anyOf", "oneOf"] {
            if let Some(members) = keyword(schema, target, union).and_then(Value::as_array) {
                let branches: Vec<Grammar> = members
                    .iter()
                    .map(|member| self.value(member, definitions, walk))
                    .collect();
                return one_of(branches, || self.any_value());
            }
        }
        if let Some(options) = keyword(schema, target, "enum").and_then(Value::as_array) {
            if !options.is_empty() && options.iter().all(Value::is_string) {
                return Grammar::Or(
                    options
                        .iter()
                        .filter_map(Value::as_str)
                        .map(|option| {
                            let token = self.spelling.string;
                            Grammar::ConstString(format!("{token}{option}{token}"))
                        })
                        .collect(),
                );
            }
        }
        match shape(schema, definitions, || self.string()) {
            Some(("boolean", _)) => Grammar::Or(vec![
                Grammar::ConstString("true".to_string()),
                Grammar::ConstString("false".to_string()),
            ]),
            Some(("object", _)) => Grammar::Sequence(vec![
                Grammar::ConstString("{".to_string()),
                self.members(schema, definitions, walk),
                Grammar::ConstString("}".to_string()),
            ]),
            Some(("array", _)) => self.list(schema, definitions, walk),
            Some(("null", _)) => Grammar::ConstString(self.spelling.null.to_string()),
            Some((_, value)) => value,
            None => self.any_value(),
        }
    }

    /// A list as the template spells one: `[`, its items in this spelling joined by `,`, `]`; an
    /// item is what the schema's `items` pins, or a string or a bare literal when it pins nothing.
    fn list(&self, schema: &Value, definitions: Definitions<'_>, walk: &mut Walk) -> Grammar {
        let target = definitions.target_of(schema);
        let item = match keyword(schema, target, "items") {
            Some(items) if items.is_object() => self.value(items, definitions, walk),
            _ => self.any_value(),
        };
        Grammar::Sequence(vec![
            Grammar::ConstString("[".to_string()),
            Grammar::Optional(Box::new(separated(item, ","))),
            Grammar::ConstString("]".to_string()),
        ])
    }

    /// A string: the token, any text without the token or a marker the engine takes inside a
    /// call, the token.
    fn string(&self) -> Grammar {
        let token = self.spelling.string;
        Grammar::Sequence(vec![
            Grammar::ConstString(token.to_string()),
            Grammar::AnyText {
                excludes: self.excluding(&[token]),
            },
            Grammar::ConstString(token.to_string()),
        ])
    }

    /// A bare literal in the table's spelling: a number, `true`, `false`, or the null word.
    fn literal(&self) -> String {
        format!("{LITERALS}|{}", self.spelling.null)
    }

    /// A value the schema does not type: a string, or a bare literal.
    fn any_value(&self) -> Grammar {
        Grammar::Or(vec![self.string(), Grammar::Regex(self.literal())])
    }

    /// The members of an object the schema gives no properties: pairs of a bare key, `:` and a
    /// string or a bare literal, joined by `,`; or none.
    fn any_members(&self) -> Grammar {
        let pair = Grammar::Sequence(vec![
            Grammar::Regex(KEY.to_string()),
            Grammar::ConstString(":".to_string()),
            self.any_value(),
        ]);
        Grammar::Optional(Box::new(separated(pair, ",")))
    }

    /// `texts` and the call's exits, as a text's excludes.
    fn excluding(&self, texts: &[&str]) -> Vec<String> {
        texts
            .iter()
            .chain(self.call_exits.iter())
            .map(|text| text.to_string())
            .collect()
    }
}

/// Whether `option`, a member of an `enum`, is a value of the JSON schema type `one`.
fn is_of_type(option: &Value, one: &str) -> bool {
    match one {
        "string" => option.is_string(),
        "integer" => option.is_i64() || option.is_u64(),
        "number" => option.is_number(),
        "boolean" => option.is_boolean(),
        "null" => option.is_null(),
        "object" => option.is_object(),
        "array" => option.is_array(),
        _ => false,
    }
}

/// One of `branches`: the one alone when there is one, any text when there is none.
fn one_of(mut branches: Vec<Grammar>, none: impl FnOnce() -> Grammar) -> Grammar {
    match branches.len() {
        0 => none(),
        1 => branches.swap_remove(0),
        _ => Grammar::Or(branches),
    }
}

/// `item` once or more, `separator` between them.
fn separated(item: Grammar, separator: &str) -> Grammar {
    Grammar::Sequence(vec![
        item.clone(),
        Grammar::Star(Box::new(Grammar::Sequence(vec![
            Grammar::ConstString(separator.to_string()),
            item,
        ]))),
    ])
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;
    use serde_json::json as value;

    use super::*;
    use crate::formats;

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

    /// A string as the grammar writes one: the text may hold neither the token nor the call's
    /// close, which the engine takes wherever it stands.
    fn text() -> Value {
        value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": "<|\"|>"},
            {"type": "any_text", "excludes": ["<|\"|>", "<tool_call|>"]},
            {"type": "const_string", "value": "<|\"|>"},
        ]})
    }

    /// A bare literal in Gemma 4's spelling.
    fn literal() -> String {
        format!("{LITERALS}|null")
    }

    fn untyped() -> Value {
        value!({"type": "or", "elements": [text(), {"type": "regex", "pattern": literal()}]})
    }

    fn pair(key: &str, value: Value) -> Value {
        value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": format!("{key}:")},
            value,
        ]})
    }

    /// An object's members as the grammar writes them past the cap, and for a schema without
    /// properties: pairs of a bare key and a string or a bare literal, in any order and number,
    /// or none.
    fn any_pairs() -> Value {
        let pair = value!({"type": "sequence", "elements": [
            {"type": "regex", "pattern": KEY},
            {"type": "const_string", "value": ":"},
            untyped(),
        ]});
        value!({"type": "optional", "content": {"type": "sequence", "elements": [
            pair,
            {"type": "star", "content": then(pair)},
        ]}})
    }

    fn then(pair: Value) -> Value {
        value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": ","},
            pair,
        ]})
    }

    #[test]
    fn gemma4_writes_each_call_in_the_templates_spelling_with_the_keys_sorted() {
        // The template: `<|tool_call>call:NAME{`, the pairs joined by `,` in `dictsort`'s order,
        // which ignores case (`artist`, `n`, `Shuffle`), `}<tool_call|>`; a string between the
        // tokens, a number as it is, a boolean as the word; a tool whose schema has no properties
        // takes any pairs, or none.
        let tools = [
            tool(
                "play",
                value!({"type": "object", "properties": {
                    "n": {"type": "integer", "minimum": 1},
                    "artist": {"type": "string"},
                    "Shuffle": {"type": "boolean"},
                }, "required": ["artist"]}),
            ),
            tool("ping", value!({"type": "object"})),
        ];
        let payload = formats::gemma4()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let n = value!({"type": "json_schema", "json_schema": {"type": "integer", "minimum": 1}});
        let shuffle = value!({"type": "or", "elements": [
            {"type": "const_string", "value": "true"},
            {"type": "const_string", "value": "false"},
        ]});
        let any_pair = value!({"type": "sequence", "elements": [
            {"type": "regex", "pattern": KEY},
            {"type": "const_string", "value": ":"},
            untyped(),
        ]});
        assert_eq!(
            payload,
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<|tool_call>"],
                "tags": [
                    {
                        "type": "tag",
                        "begin": "<|tool_call>call:play{",
                        "content": {"type": "sequence", "elements": [
                            pair("artist", text()),
                            {"type": "optional", "content": then(pair("n", n))},
                            {"type": "optional", "content": then(pair("Shuffle", shuffle))},
                        ]},
                        "end": "}<tool_call|>",
                    },
                    {
                        "type": "tag",
                        "begin": "<|tool_call>call:ping{",
                        "content": {"type": "optional", "content": {
                            "type": "sequence",
                            "elements": [
                                any_pair,
                                {"type": "star", "content": then(any_pair)},
                            ],
                        }},
                        "end": "}<tool_call|>",
                    },
                ],
                "at_least_one": true,
            }})
        );
        let optional = formats::gemma4()
            .grammar(&tools, false, false)
            .expect("a grammar");
        assert_eq!(optional.payload()["format"]["at_least_one"], false);
    }

    #[test]
    fn function_gemma_spells_its_string_token_and_null_word() {
        // functiongemma-270m-it: the tag between its own markers, a string between `<escape>`
        // tokens, a null `None`, `true` and `false` as under Gemma 4; the untyped value's literal
        // takes the same word.
        let tools = [tool(
            "note",
            value!({"type": "object", "properties": {
                "title": {"type": "string"},
                "due": {"type": ["string", "null"]},
                "done": {"type": "boolean"},
                "extra": {},
            }, "required": ["title", "due", "done", "extra"]}),
        )];
        let payload = formats::function_gemma()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let string = value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": "<escape>"},
            {"type": "any_text", "excludes": ["<escape>", "<end_function_call>"]},
            {"type": "const_string", "value": "<escape>"},
        ]});
        let none = value!({"type": "const_string", "value": "None"});
        let done = value!({"type": "or", "elements": [
            {"type": "const_string", "value": "true"},
            {"type": "const_string", "value": "false"},
        ]});
        let extra = value!({"type": "or", "elements": [
            string,
            {"type": "regex", "pattern": format!("{LITERALS}|None")},
        ]});
        assert_eq!(
            payload,
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<start_function_call>"],
                "tags": [{
                    "type": "tag",
                    "begin": "<start_function_call>call:note{",
                    "content": {"type": "sequence", "elements": [
                        pair("done", done),
                        then(pair("due", value!({"type": "or", "elements": [string, none]}))),
                        then(pair("extra", extra)),
                        then(pair("title", string)),
                    ]},
                    "end": "}<end_function_call>",
                }],
                "at_least_one": true,
            }})
        );
    }

    #[test]
    fn an_object_and_a_list_take_the_same_spelling_and_an_enum_stands_between_the_tokens() {
        // Recorded: `new_preferences:{size:<|"|>large<|"|>,temperature:<|"|>hot<|"|>}` and
        // `recipients:[<|"|>admin@example.com<|"|>,<|"|>support@example.com<|"|>]`, `t0:null`.
        let tools = [tool(
            "order",
            value!({"type": "object", "properties": {
                "prefs": {"type": "object", "properties": {
                    "temperature": {"type": "string", "enum": ["hot", "cold"]},
                    "size": {"type": "string"},
                }, "required": ["size", "temperature"]},
                "recipients": {"type": "array", "items": {"type": "string"}},
                "note": {},
                "when": {"type": "null"},
            }, "required": ["prefs", "recipients", "note", "when"]}),
        )];
        let payload = formats::gemma4()
            .grammar(&tools, true, false)
            .expect("a grammar")
            .payload();
        let members = payload["format"]["tags"][0]["content"]["elements"]
            .as_array()
            .expect("four pairs, sorted");
        assert_eq!(members[0], pair("note", untyped()));
        let temperature = value!({"type": "or", "elements": [
            {"type": "const_string", "value": "<|\"|>hot<|\"|>"},
            {"type": "const_string", "value": "<|\"|>cold<|\"|>"},
        ]});
        assert_eq!(
            members[1],
            then(pair(
                "prefs",
                value!({"type": "sequence", "elements": [
                    {"type": "const_string", "value": "{"},
                    {"type": "sequence", "elements": [
                        pair("size", text()),
                        then(pair("temperature", temperature)),
                    ]},
                    {"type": "const_string", "value": "}"},
                ]})
            ))
        );
        assert_eq!(
            members[2],
            then(pair(
                "recipients",
                value!({"type": "sequence", "elements": [
                    {"type": "const_string", "value": "["},
                    {"type": "optional", "content": {"type": "sequence", "elements": [
                        text(),
                        {"type": "star", "content": {"type": "sequence", "elements": [
                            {"type": "const_string", "value": ","},
                            text(),
                        ]}},
                    ]}},
                    {"type": "const_string", "value": "]"},
                ]})
            ))
        );
        assert_eq!(
            members[3],
            then(pair(
                "when",
                value!({"type": "const_string", "value": "null"})
            ))
        );
    }

    #[test]
    fn a_string_enum_behind_a_reference_takes_the_tokens_and_a_number_keeps_its_definitions() {
        // Pydantic writes an `Enum` field as a `$ref` to a definition holding the `enum`; a
        // number's schema goes to the engine as a document of its own, the definition it reaches
        // attached and no other.
        let root = value!({
            "type": "object",
            "properties": {
                "unit": {"$ref": "#/$defs/Unit"},
                "count": {"$ref": "#/$defs/Count"},
            },
            "$defs": {
                "Unit": {"type": "string", "enum": ["c", "f"]},
                "Count": {"type": "integer"},
            },
        });
        let ends = Ends {
            call_exits: &["<tool_call|>"],
            spelling: Spelling::GEMMA4,
        };
        let definitions = Definitions::of(&root);
        assert_eq!(
            ends.value(&root["properties"]["unit"], definitions, &mut Walk::new())
                .to_json(),
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": "<|\"|>c<|\"|>"},
                {"type": "const_string", "value": "<|\"|>f<|\"|>"},
            ]})
        );
        assert_eq!(
            ends.value(&root["properties"]["count"], definitions, &mut Walk::new())
                .to_json(),
            value!({"type": "json_schema", "json_schema": {
                "$ref": "#/$defs/Count",
                "$defs": {"Count": {"type": "integer"}},
            }})
        );
        // A list without an item schema takes strings or bare literals.
        assert_eq!(
            ends.value(&value!({"type": "array"}), definitions, &mut Walk::new())
                .to_json()["elements"][1]["content"]["elements"][0],
            untyped()
        );
    }

    #[test]
    fn a_schema_that_names_itself_is_written_once_and_a_union_as_its_branches() {
        // Pydantic's recursive models and `Optional[...]` fields: a `Node` whose child is a `Node`,
        // a `Tree` whose items are `Tree`s, a nullable list, an `anyOf` of a `Node` and `null`.
        // The second reference on the way down is untyped, so the tag is finite; a union is one
        // of its branches.
        let root = value!({
            "type": "object",
            "properties": {
                "root": {"$ref": "#/$defs/Node"},
                "tree": {"$ref": "#/$defs/Tree"},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}},
                "inner": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]},
            },
            "required": ["root", "tree", "tags", "inner"],
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/Node"}},
                    "required": ["child"],
                },
                "Tree": {"type": "array", "items": {"$ref": "#/$defs/Tree"}},
            },
        });
        let ends = Ends {
            call_exits: &["<tool_call|>"],
            spelling: Spelling::GEMMA4,
        };
        let definitions = Definitions::of(&root);
        let members = ends.members(&root, definitions, &mut Walk::new()).to_json();
        let members = members["elements"].as_array().expect("four pairs");
        let node = value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": "{"},
            {"type": "sequence", "elements": [pair("child", untyped())]},
            {"type": "const_string", "value": "}"},
        ]});
        let list_of = |item: Value| {
            value!({"type": "sequence", "elements": [
                {"type": "const_string", "value": "["},
                {"type": "optional", "content": {"type": "sequence", "elements": [
                    item,
                    {"type": "star", "content": then(item)},
                ]}},
                {"type": "const_string", "value": "]"},
            ]})
        };
        // Sorted: `inner`, `root`, `tags`, `tree`.
        assert_eq!(
            members[0],
            pair(
                "inner",
                value!({"type": "or", "elements": [
                    node.clone(),
                    {"type": "const_string", "value": "null"},
                ]})
            )
        );
        assert_eq!(members[1], then(pair("root", node)));
        assert_eq!(
            members[2],
            then(pair(
                "tags",
                value!({"type": "or", "elements": [
                    list_of(text()),
                    {"type": "const_string", "value": "null"},
                ]})
            ))
        );
        assert_eq!(members[3], then(pair("tree", list_of(untyped()))));
    }

    #[test]
    fn a_chain_of_distinct_unions_and_a_wide_dag_stay_within_the_budget() {
        // Forty definitions, each an `anyOf` of two references to the next: the path check alone
        // lets every level double (D1 is popped after the first branch and expanded again by the
        // second), so union levels count against the depth, and the values a tag spells are
        // counted in all; the tag stays small.
        let mut defs = serde_json::Map::new();
        for i in 0..40 {
            let next = format!("#/$defs/D{}", i + 1);
            defs.insert(
                format!("D{i}"),
                value!({"anyOf": [{"$ref": next}, {"$ref": next}]}),
            );
        }
        defs.insert("D40".to_string(), value!({"type": "string"}));
        let chain = value!({
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/D0"}},
            "required": ["x"],
            "$defs": defs,
        });
        let ends = Ends {
            call_exits: &["<tool_call|>"],
            spelling: Spelling::GEMMA4,
        };
        let written = serde_json::to_string(
            &ends
                .members(&chain, Definitions::of(&chain), &mut Walk::new())
                .to_json(),
        )
        .expect("JSON");
        assert!(written.len() < 200_000, "{} bytes", written.len());

        // Eight definitions, each ten optional properties referring to the next: distinct along
        // the path, within the depth, so the value budget bounds the walk (ten to the eighth
        // values otherwise) and the node cap the tree, the whole of it, since an object's members
        // hold every object below.
        let mut defs = serde_json::Map::new();
        for i in 0..8 {
            let mut properties = serde_json::Map::new();
            for j in 0..10 {
                let next = if i < 7 {
                    value!({"$ref": format!("#/$defs/W{}", i + 1)})
                } else {
                    value!({"type": "string"})
                };
                properties.insert(format!("p{j}"), next);
            }
            defs.insert(
                format!("W{i}"),
                value!({"type": "object", "properties": properties}),
            );
        }
        let wide = value!({
            "type": "object",
            "properties": {"w": {"$ref": "#/$defs/W0"}},
            "required": ["w"],
            "$defs": defs,
        });
        let mut walk = Walk::new();
        let grammar = ends.members(&wide, Definitions::of(&wide), &mut walk);
        assert_eq!(walk.values, 0, "the budget is spent");
        assert!(grammar.nodes() <= NODES, "{} nodes", grammar.nodes());

        // Ten optional properties, each an object of ten optional strings: 110 values, within the
        // budget, but the outer pairs would be copied into the starts before them (each with the
        // inner object's own copies), past the node cap, so the object takes any pairs, the inner
        // spellings with it: the fallback is one, and it is the whole object's.
        let mut inner = serde_json::Map::new();
        for j in 0..10 {
            inner.insert(format!("s{j}"), value!({"type": "string"}));
        }
        let mut outer = serde_json::Map::new();
        for i in 0..10 {
            outer.insert(
                format!("o{i}"),
                value!({"type": "object", "properties": inner}),
            );
        }
        let two_levels = value!({"type": "object", "properties": outer});
        let mut walk = Walk::new();
        let grammar = ends.members(&two_levels, Definitions::of(&two_levels), &mut walk);
        assert_eq!(
            walk.values,
            VALUES - 110,
            "the values are walked before the cap is read"
        );
        assert_eq!(grammar.to_json(), any_pairs());
    }

    #[test]
    fn a_nullable_enum_keeps_its_members_in_the_string_branch() {
        // OpenAI's strict mode: `{"type": ["string", "null"], "enum": ["F", "C", null]}` is one
        // of the two strings between the tokens, or `null`; a list type whose enum lists no
        // `null` admits none, so its null branch is dropped and the enum stands alone; a type
        // listed twice is one branch.
        let ends = Ends {
            call_exits: &["<tool_call|>"],
            spelling: Spelling::GEMMA4,
        };
        let root = value!({});
        let definitions = Definitions::of(&root);
        let strings = value!({"type": "or", "elements": [
            {"type": "const_string", "value": "<|\"|>F<|\"|>"},
            {"type": "const_string", "value": "<|\"|>C<|\"|>"},
        ]});
        let nullable = value!({"type": ["string", "null"], "enum": ["F", "C", null]});
        assert_eq!(
            ends.value(&nullable, definitions, &mut Walk::new())
                .to_json(),
            value!({"type": "or", "elements": [
                strings,
                {"type": "const_string", "value": "null"},
            ]})
        );
        let not_null = value!({"type": ["string", "null"], "enum": ["F", "C"]});
        assert_eq!(
            ends.value(&not_null, definitions, &mut Walk::new())
                .to_json(),
            strings
        );
        let repeated = value!({"type": ["null", "null", "null", "string", "unknown"]});
        assert_eq!(
            ends.value(&repeated, definitions, &mut Walk::new())
                .to_json(),
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": "null"},
                text(),
            ]})
        );
    }

    #[test]
    fn past_the_node_cap_an_object_takes_any_pairs() {
        // Twenty-nine optional strings take 3,859 nodes once joined (each pair copied into the
        // starts before it, with its separator and option), within the cap: the eighth pair
        // stands in eight starts. Thirty would take 4,127, past it: the object writes any pairs
        // instead, each a bare key and a string or a bare literal, in any order and number.
        let strings = |count: usize| {
            let mut properties = serde_json::Map::new();
            for i in 0..count {
                properties.insert(format!("k{i:02}"), value!({"type": "string"}));
            }
            value!({"type": "object", "properties": properties})
        };
        let ends = Ends {
            call_exits: &["<tool_call|>"],
            spelling: Spelling::GEMMA4,
        };
        let fits = strings(29);
        let grammar = ends.members(&fits, Definitions::of(&fits), &mut Walk::new());
        assert_eq!(grammar.nodes(), 3_859);
        let written = serde_json::to_string(&grammar.to_json()).expect("JSON");
        assert_eq!(written.matches("\"k07:\"").count(), 8, "{written}");
        let past = strings(30);
        let grammar = ends.members(&past, Definitions::of(&past), &mut Walk::new());
        assert_eq!(grammar.to_json(), any_pairs());

        // A number behind a reference carries the definition it reaches in its JSON schema, which
        // every copy repeats, and the leaf counts by its size: ten pairs reaching a definition of
        // two thousand enum values pass the cap, and the object takes any pairs.
        let mut properties = serde_json::Map::new();
        for i in 0..10 {
            properties.insert(format!("n{i}"), value!({"$ref": "#/$defs/Big"}));
        }
        let heavy = value!({
            "type": "object",
            "properties": properties,
            "$defs": {"Big": {"type": "integer", "enum": (0..2000).collect::<Vec<i32>>()}},
        });
        let grammar = ends.members(&heavy, Definitions::of(&heavy), &mut Walk::new());
        assert_eq!(grammar.to_json(), any_pairs());
    }

    #[test]
    fn a_value_nested_past_the_depth_is_untyped_so_the_walk_stays_shallow() {
        // Twelve objects, each the only property of the one above: the spelling stops at `DEPTH`,
        // so the walk recurses no deeper whatever a request nests or chains through its
        // definitions (left to the value budget, it went two hundred and fifty-seven levels down
        // and overflowed the test thread's stack).
        let mut schema = value!({"type": "string"});
        for _ in 0..12 {
            schema = value!({"type": "object", "properties": {"n": schema}, "required": ["n"]});
        }
        let ends = Ends {
            call_exits: &["<tool_call|>"],
            spelling: Spelling::GEMMA4,
        };
        let grammar = ends.members(&schema, Definitions::of(&schema), &mut Walk::new());
        let written = serde_json::to_string(&grammar.to_json()).expect("JSON");
        // The first `DEPTH` values are typed, each an object with its `n:` pair; the value below
        // the last of them is untyped, and the string at the bottom is never reached.
        assert_eq!(written.matches("\"n:\"").count(), DEPTH + 1, "{written}");
        let untyped = serde_json::to_string(&pair("n", untyped())).expect("JSON");
        assert!(written.contains(&untyped), "{written}");
        let string = serde_json::to_string(&pair("n", text())).expect("JSON");
        assert!(!written.contains(&string), "{written}");
    }

    #[test]
    fn with_the_thought_open_the_prefix_excludes_the_call_markers_and_the_string_token() {
        // The thought's own markers stay writable; the call's two and the string's token are not.
        let payload = formats::gemma4()
            .grammar(&[tool("ping", value!({"type": "object"}))], true, true)
            .expect("a grammar")
            .payload();
        assert_eq!(payload["format"]["type"], "sequence");
        assert_eq!(
            payload["format"]["elements"][0],
            value!({
                "type": "any_text",
                "excludes": ["<|tool_call>", "<tool_call|>", "<|\"|>"],
            })
        );
        assert_eq!(payload["format"]["elements"][1]["type"], "triggered_tags");
    }
}
