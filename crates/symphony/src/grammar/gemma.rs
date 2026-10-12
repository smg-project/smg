//! Gemma 4's calls, written from the table's markers as the template's `format_argument` macro
//! writes them: `<|tool_call>call:NAME{`, the arguments, `}<tool_call|>`. The arguments are
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
//! no rule that names itself. For the same reason the tag is bounded three ways, each falling back
//! to a string or a bare literal: a definition that names itself is written once (the `$ref`
//! pointers on the way down are kept), a value nests at most [`DEPTH`] levels, unions included,
//! a tag spells at most [`VALUES`] values in all, which bounds the work, and an object's members
//! take at most [`NODES`] nodes of the tree once [`joined`] has copied each pair into the starts
//! before it, or the object takes any pairs instead, which bounds the size (estimated from the
//! pairs before the copies are made). The last two bound a schema whose definitions are distinct
//! at every level or reused across one (a chain of unions, a wide DAG), where the first two do
//! not, so a request's schema gives a finite tag of bounded size in bounded time.

use openai_protocol::common::Tool;
use serde_json::Value;

use super::{
    schema::{joined, joined_nodes, keyword, shape, Definitions},
    CallMarkers, Grammar, Tag,
};
use crate::gemma::STRING;

/// A bare literal as the template writes one: a number as Python prints it, `true`, `false` or
/// `null`.
const LITERAL: &str = r"-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?|true|false|null";
/// A key written as it is: anything but the bytes that end or open a member.
const KEY: &str = r"[^:,{}\[\]<]+";
/// How deep a value may nest before its spelling gives way to a string or a bare literal; no
/// recorded tool nests further. The cap is small because [`joined`] clones each pair into every
/// start before it, so each level of optional pairs multiplies the tree.
const DEPTH: usize = 8;
/// How many nodes an object's members may take once [`joined`] has copied each pair into the
/// starts before it; past that the object takes any pairs. With ten optional properties a pair is
/// copied up to ten times, so an object of objects multiplies by that at every level, and this
/// bounds the whole tree whatever the schema's shape.
const NODES: usize = 2_048;
/// How many values a tag spells before the rest give way to a string or a bare literal: the walk
/// down a schema is depth first and eager, so this bounds its work whatever the schema's shape.
const VALUES: usize = 256;

/// What the walk down a schema carries: the `$ref` pointers on the way down, so a definition that
/// names itself is written once, and the values left to spell.
struct Walk {
    expanding: Vec<String>,
    values: usize,
}

impl Walk {
    fn new() -> Self {
        Self {
            expanding: Vec::new(),
            values: VALUES,
        }
    }
}

/// The calls for `tools`: one tag per tool, triggered by the call's opener, `call:NAME{` after it,
/// the arguments, then `}` and the call's close.
pub(super) fn calls(markers: &CallMarkers<'_>, tools: &[&Tool], at_least_one: bool) -> Grammar {
    let ends = Ends {
        call_exits: &markers.call_exits,
    };
    let tags = tools
        .iter()
        .map(|tool| {
            let parameters = &tool.function.parameters;
            Tag::new(
                format!("{}call:{}{{", markers.call_open, tool.function.name),
                ends.members(
                    parameters,
                    Definitions::of(parameters),
                    DEPTH,
                    &mut Walk::new(),
                ),
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
/// the arguments state, which the engine takes wherever they stand inside a call.
struct Ends<'a> {
    call_exits: &'a [&'a str],
}

impl Ends<'_> {
    /// The members of the object `schema` describes: one `key:value` pair per property, the keys
    /// in sorted order ignoring case, joined by `,`; any pairs for a schema without properties, or
    /// whose pairs would take more than [`NODES`] nodes once joined. The properties are read
    /// through the schema's `$ref`, as every keyword is. `depth` is what
    /// is left of [`DEPTH`]; `walk` carries the `$ref` pointers on the way down and the values
    /// left to spell.
    fn members(
        &self,
        schema: &Value,
        definitions: Definitions<'_>,
        depth: usize,
        walk: &mut Walk,
    ) -> Grammar {
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
                    self.value(property, definitions, depth, walk),
                ]);
                (pair, required.contains(&key.as_str()))
            })
            .collect();
        if joined_nodes(&pairs) > NODES {
            return self.any_members();
        }
        joined(&pairs, ",")
    }

    /// A value as the template spells one, or a string or a bare literal where the value nests
    /// deeper than [`DEPTH`], the tag has spelled [`VALUES`] values already, or its `$ref` is one
    /// on the way down: a definition that names itself is written once.
    fn value(
        &self,
        schema: &Value,
        definitions: Definitions<'_>,
        depth: usize,
        walk: &mut Walk,
    ) -> Grammar {
        let Some(depth) = depth.checked_sub(1) else {
            return self.any_value();
        };
        let Some(values) = walk.values.checked_sub(1) else {
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
        let grammar = self.typed(schema, definitions, depth, walk);
        if pointer.is_some() {
            walk.expanding.pop();
        }
        grammar
    }

    /// A value by what its schema says: one of its branches where the `type` is a list (each the
    /// schema with that one type and the `enum` members of that type) or an `anyOf` or `oneOf`
    /// stands, a union level counting as one of nesting; a string `enum` as one of its values
    /// between the tokens; a
    /// string between the tokens; `true`, `false` and `null` as the words; a number as the JSON
    /// the property's schema accepts; an object and a list in this same spelling; a string or a
    /// bare literal where the schema pins no type, or a pointer in it resolves to nothing. Every
    /// keyword is read through the property's `$ref`, as [`shape`] reads them.
    fn typed(
        &self,
        schema: &Value,
        definitions: Definitions<'_>,
        depth: usize,
        walk: &mut Walk,
    ) -> Grammar {
        let target = definitions.target_of(schema);
        if let Some(types) = keyword(schema, target, "type").and_then(Value::as_array) {
            // Each branch is the schema with that one type, the property's own keyword winning,
            // and the `enum` narrowed to the members of that type (OpenAI's strict mode writes a
            // nullable enum as `"type": ["string", "null"], "enum": ["F", "C", null]`).
            let options = keyword(schema, target, "enum").and_then(Value::as_array);
            let branches: Vec<Grammar> = types
                .iter()
                .filter_map(Value::as_str)
                .map(|one| {
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
                                map.remove("enum");
                            } else {
                                map.insert("enum".to_string(), Value::Array(narrowed));
                            }
                        }
                    }
                    self.typed(&single, definitions, depth, walk)
                })
                .collect();
            return one_of(branches, || self.any_value());
        }
        for union in ["anyOf", "oneOf"] {
            if let Some(members) = keyword(schema, target, union).and_then(Value::as_array) {
                let branches: Vec<Grammar> = members
                    .iter()
                    .map(|member| self.value(member, definitions, depth, walk))
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
                        .map(|option| Grammar::ConstString(format!("{STRING}{option}{STRING}")))
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
                self.members(schema, definitions, depth, walk),
                Grammar::ConstString("}".to_string()),
            ]),
            Some(("array", _)) => self.list(schema, definitions, depth, walk),
            Some((_, value)) => value,
            None => self.any_value(),
        }
    }

    /// A list as the template spells one: `[`, its items in this spelling joined by `,`, `]`; an
    /// item is what the schema's `items` pins, or a string or a bare literal when it pins nothing.
    fn list(
        &self,
        schema: &Value,
        definitions: Definitions<'_>,
        depth: usize,
        walk: &mut Walk,
    ) -> Grammar {
        let target = definitions.target_of(schema);
        let item = match keyword(schema, target, "items") {
            Some(items) if items.is_object() => self.value(items, definitions, depth, walk),
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
        Grammar::Sequence(vec![
            Grammar::ConstString(STRING.to_string()),
            Grammar::AnyText {
                excludes: self.excluding(&[STRING]),
            },
            Grammar::ConstString(STRING.to_string()),
        ])
    }

    /// A value the schema does not type: a string, or a bare literal.
    fn any_value(&self) -> Grammar {
        Grammar::Or(vec![self.string(), Grammar::Regex(LITERAL.to_string())])
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

    fn untyped() -> Value {
        value!({"type": "or", "elements": [text(), {"type": "regex", "pattern": LITERAL}]})
    }

    fn pair(key: &str, value: Value) -> Value {
        value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": format!("{key}:")},
            value,
        ]})
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
        // number's schema goes to the engine as a document of its own, its definitions attached.
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
        };
        let definitions = Definitions::of(&root);
        assert_eq!(
            ends.value(
                &root["properties"]["unit"],
                definitions,
                DEPTH,
                &mut Walk::new()
            )
            .to_json(),
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": "<|\"|>c<|\"|>"},
                {"type": "const_string", "value": "<|\"|>f<|\"|>"},
            ]})
        );
        assert_eq!(
            ends.value(
                &root["properties"]["count"],
                definitions,
                DEPTH,
                &mut Walk::new()
            )
            .to_json(),
            value!({"type": "json_schema", "json_schema": {
                "$ref": "#/$defs/Count",
                "$defs": root["$defs"],
            }})
        );
        // A list without an item schema takes strings or bare literals.
        assert_eq!(
            ends.value(
                &value!({"type": "array"}),
                definitions,
                DEPTH,
                &mut Walk::new()
            )
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
        };
        let definitions = Definitions::of(&root);
        let members = ends
            .members(&root, definitions, DEPTH, &mut Walk::new())
            .to_json();
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
        };
        let written = serde_json::to_string(
            &ends
                .members(&chain, Definitions::of(&chain), DEPTH, &mut Walk::new())
                .to_json(),
        )
        .expect("JSON");
        assert!(written.len() < 200_000, "{} bytes", written.len());

        // Eight definitions, each ten optional properties referring to the next: distinct along
        // the path, within the depth, so the value budget bounds the walk (ten to the eighth
        // values otherwise) and the node cap the tree (the pairs are copied into the starts before
        // them, so an object of objects takes any pairs instead of its spelling).
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
        let grammar = ends.members(&wide, Definitions::of(&wide), DEPTH, &mut walk);
        let written = serde_json::to_string(&grammar.to_json()).expect("JSON");
        assert_eq!(walk.values, 0, "the budget is spent");
        assert!(written.len() < 300_000, "{} bytes", written.len());

        // Ten optional properties, each an object of ten optional strings: 110 values, within the
        // budget, but the outer pairs would be copied into the starts before them (each with the
        // inner object's own copies), past the node cap, so the outer object takes any pairs and
        // the inner ones keep their spelling.
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
        let grammar = ends.members(&two_levels, Definitions::of(&two_levels), DEPTH, &mut walk);
        let written = serde_json::to_string(&grammar.to_json()).expect("JSON");
        assert_eq!(walk.values, VALUES - 110);
        let key = serde_json::to_string(KEY).expect("JSON");
        assert!(
            written.contains(&key),
            "the outer object takes any pairs: {written}"
        );
        assert!(written.len() < 10_000, "{} bytes", written.len());
    }

    #[test]
    fn a_nullable_enum_keeps_its_members_in_the_string_branch() {
        // OpenAI's strict mode: `{"type": ["string", "null"], "enum": ["F", "C", null]}` is one
        // of the two strings between the tokens, or `null`; a list type whose enum lists no
        // `null` takes its enum in the string branch alone.
        let ends = Ends {
            call_exits: &["<tool_call|>"],
        };
        let root = value!({});
        let definitions = Definitions::of(&root);
        for unit in [
            value!({"type": ["string", "null"], "enum": ["F", "C", null]}),
            value!({"type": ["string", "null"], "enum": ["F", "C"]}),
        ] {
            assert_eq!(
                ends.value(&unit, definitions, DEPTH, &mut Walk::new())
                    .to_json(),
                value!({"type": "or", "elements": [
                    {"type": "or", "elements": [
                        {"type": "const_string", "value": "<|\"|>F<|\"|>"},
                        {"type": "const_string", "value": "<|\"|>C<|\"|>"},
                    ]},
                    {"type": "const_string", "value": "null"},
                ]}),
                "{unit}"
            );
        }
    }

    #[test]
    fn a_value_nested_past_the_depth_is_untyped_so_the_tag_stays_bounded() {
        // Twelve objects, each the only property of the one above: the spelling stops at `DEPTH`.
        let mut schema = value!({"type": "string"});
        for _ in 0..12 {
            schema = value!({"type": "object", "properties": {"n": schema}, "required": ["n"]});
        }
        let ends = Ends {
            call_exits: &["<tool_call|>"],
        };
        let grammar = ends.members(&schema, Definitions::of(&schema), DEPTH, &mut Walk::new());
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
