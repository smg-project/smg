//! What a tool's JSON schema says about each argument, for the syntaxes that write arguments one
//! by one with a type of their own (Kimi K3's XTML, the keyed tags): the type a property's schema
//! pins and the grammar of a value of that type, and the definitions a property's schema carries
//! along so a `$ref` inside it still resolves when the engine compiles it on its own; and the walk
//! over a tool's properties the syntaxes share: one argument per property, in the schema's order,
//! optional unless the schema requires it; and the rule for pairs a separator joins, which stands
//! between two present pairs only (Olmo 3's `, `, Gemma 4's `,`).
//!
//! A JSON schema leaf goes to the engine as a document of its own, and the grammars copy it (a pair
//! into every start before it), so the document is kept small: only the definitions its pointers
//! reach travel with it, and the annotations JSON Schema defines as such (`title`, `description`,
//! `examples`, `$comment`), which constrain nothing, are dropped at every schema position.

use std::collections::HashSet;

use serde_json::{json, Value};

use super::Grammar;

/// The arguments a tool's schema asks for, as a syntax writes them: `each` property in the
/// schema's order, each optional unless the schema requires it; `none` for a schema without
/// properties, which takes whatever arguments the syntax allows.
pub(super) fn arguments(
    parameters: &Value,
    each: impl Fn(&str, &Value, Definitions<'_>) -> Grammar,
    none: impl FnOnce() -> Grammar,
) -> Grammar {
    let properties = parameters
        .get("properties")
        .and_then(Value::as_object)
        .filter(|properties| !properties.is_empty());
    let Some(properties) = properties else {
        return none();
    };
    let required: Vec<&str> = parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let definitions = Definitions::of(parameters);
    Grammar::Sequence(
        properties
            .iter()
            .map(|(key, schema)| {
                let argument = each(key, schema, definitions);
                if required.contains(&key.as_str()) {
                    argument
                } else {
                    Grammar::Optional(Box::new(argument))
                }
            })
            .collect(),
    )
}

/// `pairs` joined by `separator`, each pair present when its flag says the schema requires it and
/// otherwise optional. The separator stands between two present pairs only, so the sequence is
/// one choice per pair that may stand first (every pair up to the first required one), each
/// followed by the pairs after it, a required one with its separator, an optional one with its
/// separator inside the option; the whole is optional when no pair is required, and no pairs are
/// the empty text.
pub(super) fn joined(pairs: &[(Grammar, bool)], separator: &str) -> Grammar {
    let (first_required, may_start) = starts_of(pairs.iter().map(|(_, required)| *required));
    let mut starts: Vec<Grammar> = (0..may_start)
        .map(|first| {
            let mut elements = vec![pairs[first].0.clone()];
            for (pair, required) in &pairs[first + 1..] {
                let with_separator = Grammar::Sequence(vec![
                    Grammar::ConstString(separator.to_string()),
                    pair.clone(),
                ]);
                elements.push(if *required {
                    with_separator
                } else {
                    Grammar::Optional(Box::new(with_separator))
                });
            }
            Grammar::Sequence(elements)
        })
        .collect();
    let some = match starts.len() {
        0 => return Grammar::Sequence(Vec::new()),
        1 => starts.swap_remove(0),
        _ => Grammar::Or(starts),
    };
    if first_required.is_some() {
        some
    } else {
        Grammar::Optional(Box::new(some))
    }
}

/// The position of the first required pair, and how many pairs may stand first: every pair up to
/// and including it, or all of them when none is required.
fn starts_of(required: impl Iterator<Item = bool>) -> (Option<usize>, usize) {
    let mut count = 0;
    let mut first_required = None;
    for (at, required) in required.enumerate() {
        count += 1;
        if required && first_required.is_none() {
            first_required = Some(at);
        }
    }
    (first_required, first_required.map_or(count, |at| at + 1))
}

/// The nodes [`joined`] makes of pairs of the given weights (each a pair's nodes), as
/// [`Grammar::nodes`] counts them: each pair is copied into every start before it, so the pair at
/// `i` goes into `i + 1` starts, or into every start when the required pairs leave fewer, each
/// copy after a start's first with its separator (and its option, when the pair is optional).
/// Each weight is read once, from the last pair back. A derivation that bounds its tree asks
/// before joining.
pub(super) fn joined_nodes(pairs: &[(usize, bool)]) -> usize {
    let (first_required, may_start) = starts_of(pairs.iter().map(|(_, required)| *required));
    // `after` is what the pairs after `at` add to a start that opens with it.
    let mut after = 0;
    let mut starts = 0;
    for (at, (weight, required)) in pairs.iter().enumerate().rev() {
        if at < may_start {
            starts += 1 + weight + after;
        }
        after += weight + if *required { 2 } else { 3 };
    }
    let or = usize::from(may_start > 1);
    let optional = usize::from(first_required.is_none());
    starts + or + optional
}

/// The schema's own keyword, or its `$ref` target's: a `$ref` lends its target's keywords, and
/// the property's own win, as in JSON Schema 2020-12.
pub(super) fn keyword<'a>(
    schema: &'a Value,
    target: Option<&'a Value>,
    name: &str,
) -> Option<&'a Value> {
    schema
        .get(name)
        .or_else(|| target.and_then(|target| target.get(name)))
}

/// The type a property's schema pins, by the name the templates write (`integer` as `number`), a
/// string `enum` a string; `None` when the schema pins no type the syntaxes have, or a `$ref` in
/// it points at nothing (a grammar with such a pointer would not compile, and the whole call with
/// it). What [`shape`] decides before it writes the value's grammar, on its own for a writer that
/// spells objects and arrays itself and needs no grammar for them. A `$ref` lends its target's
/// keywords, and the property's own win, as in JSON Schema 2020-12.
pub(super) fn kind(schema: &Value, definitions: Definitions<'_>) -> Option<&'static str> {
    if !definitions.pointers_resolve(schema) {
        return None;
    }
    let target = definitions.target_of(schema);
    if string_enum(schema, target).is_some() {
        return Some("string");
    }
    match keyword(schema, target, "type").and_then(Value::as_str)? {
        "string" => Some("string"),
        "integer" | "number" => Some("number"),
        "boolean" => Some("boolean"),
        "object" => Some("object"),
        "array" => Some("array"),
        "null" => Some("null"),
        _ => None,
    }
}

/// The words of a string `enum`, where the schema (or its `$ref` target) has one with a word in
/// it and nothing else.
pub(super) fn string_enum<'a>(schema: &'a Value, target: Option<&'a Value>) -> Option<&'a [Value]> {
    keyword(schema, target, "enum")
        .and_then(Value::as_array)
        .filter(|values| !values.is_empty() && values.iter().all(Value::is_string))
        .map(Vec::as_slice)
}

/// The type a property's schema pins ([`kind`]) and the grammar of a value of that type: a string
/// as `text`, the syntax's own spelling of a value written as it is; a string `enum` as one of its
/// values; a number, an object or an array as JSON the property's schema accepts, the tool's
/// definitions attached; a boolean as JSON; `null` as the word.
pub(super) fn shape(
    schema: &Value,
    definitions: Definitions<'_>,
    text: impl FnOnce() -> Grammar,
) -> Option<(&'static str, Grammar)> {
    let kind = kind(schema, definitions)?;
    let target = definitions.target_of(schema);
    let json = |schema: Value| Grammar::JsonSchema {
        schema,
        style: None,
    };
    let grammar = match kind {
        "string" => match string_enum(schema, target) {
            Some(values) => Grammar::Or(
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|value| Grammar::ConstString(value.to_string()))
                    .collect(),
            ),
            None => text(),
        },
        "number" | "object" | "array" => json(definitions.attached_to(schema)),
        "boolean" => json(json!({"type": "boolean"})),
        // The one kind left: `null`.
        _ => Grammar::ConstString("null".to_string()),
    };
    Some((kind, grammar))
}

/// The definitions at the root of a tool's schema, `$defs` and `definitions`. Each property's
/// schema goes to the engine as a document of its own, and a `$ref` such as `#/$defs/Node` points
/// from the root of the document it stands in, so the definitions travel with the property's
/// schema. A pointer to the root itself, `#`, names the tool's schema in the tool and the property
/// in the property's document, so it resolves to nothing here, and the property takes any value.
#[derive(Clone, Copy)]
pub(super) struct Definitions<'a> {
    defs: Option<&'a Value>,
    definitions: Option<&'a Value>,
    /// Whether every local pointer anywhere in the root names something, worked out once, so
    /// the check each property's [`kind`] makes costs nothing when the tool's schema is whole.
    whole: bool,
}

impl<'a> Definitions<'a> {
    pub(super) fn of(root: &'a Value) -> Self {
        let definitions = Self {
            defs: root.get("$defs"),
            definitions: root.get("definitions"),
            whole: false,
        };
        let whole = local_pointers(root)
            .into_iter()
            .all(|pointer| definitions.resolve(pointer).is_some());
        Self {
            whole,
            ..definitions
        }
    }

    /// The schema a local pointer names: `#/$defs/Name` or `#/definitions/Name` an entry, with the
    /// pointer's escapes (`~1` for `/`, `~0` for `~`) undone; `#` itself nothing, since it would
    /// name the property's own document once the engine compiles it alone.
    fn resolve(self, pointer: &str) -> Option<&'a Value> {
        let mut segments = pointer.strip_prefix("#/")?.split('/');
        let mut node = match segments.next()? {
            "$defs" => self.defs?,
            "definitions" => self.definitions?,
            _ => return None,
        };
        for segment in segments {
            node = node.get(segment.replace("~1", "/").replace("~0", "~"))?;
        }
        Some(node)
    }

    /// The schema the property's own `$ref` names, one hop: its keywords stand beside the
    /// property's, and a cycle of definitions is the engine's to expand, not this function's.
    pub(super) fn target_of(self, schema: &'a Value) -> Option<&'a Value> {
        self.resolve(schema.get("$ref")?.as_str()?)
    }

    /// Whether every local pointer in `schema` names something, and every pointer in what it
    /// names, as far as the pointers reach: the definitions travel with the property, so a pointer
    /// at nothing inside one of them reaches the engine too. A definition that names itself is
    /// followed once.
    fn pointers_resolve(self, schema: &Value) -> bool {
        self.whole || self.pointers_resolve_from(schema, &mut HashSet::new())
    }

    fn pointers_resolve_from<'s>(self, schema: &'s Value, followed: &mut HashSet<&'s str>) -> bool
    where
        'a: 's,
    {
        match schema {
            Value::Object(map) => map.iter().all(|(key, value)| {
                let local = value
                    .as_str()
                    .filter(|pointer| is_reference(key) && pointer.starts_with('#'));
                let named_resolves = local.is_none_or(|pointer| match self.resolve(pointer) {
                    None => false,
                    Some(target) => {
                        !followed.insert(pointer) || self.pointers_resolve_from(target, followed)
                    }
                });
                named_resolves && self.pointers_resolve_from(value, followed)
            }),
            Value::Array(items) => items
                .iter()
                .all(|item| self.pointers_resolve_from(item, followed)),
            _ => true,
        }
    }

    /// The property's schema as the engine gets it: its annotations dropped, and the root's
    /// definitions its pointers reach attached, so a local pointer in it still resolves when the
    /// engine compiles it on its own. A definition the schema declares itself stays; the root's
    /// fill in the names it lacks, those reached and no other, since the grammars copy the leaf.
    fn attached_to(self, schema: &Value) -> Value {
        let mut attached = without_annotations(schema);
        if self.defs.is_none() && self.definitions.is_none() || !has_local_reference(schema) {
            return attached;
        }
        let reached = self.reached_from(schema);
        let Some(object) = attached.as_object_mut() else {
            return attached;
        };
        for (name, block) in [("$defs", self.defs), ("definitions", self.definitions)] {
            let Some(Value::Object(entries)) = block else {
                continue;
            };
            let entries: Vec<(&String, Value)> = entries
                .iter()
                .filter(|(key, _)| reached.contains(&(name, key.as_str())))
                .map(|(key, value)| (key, without_annotations(value)))
                .collect();
            if entries.is_empty() {
                continue;
            }
            match object.get_mut(name) {
                Some(Value::Object(own)) => {
                    for (key, value) in entries {
                        own.entry(key.clone()).or_insert(value);
                    }
                }
                Some(_) => {}
                None => {
                    object.insert(
                        name.to_string(),
                        Value::Object(entries.into_iter().map(|(k, v)| (k.clone(), v)).collect()),
                    );
                }
            }
        }
        attached
    }

    /// The root's definitions the pointers in `schema` reach, directly or through one another:
    /// each as its block (`$defs` or `definitions`) and its name, the first segment after the
    /// block, since a pointer deeper into an entry needs the whole entry.
    fn reached_from(self, schema: &Value) -> HashSet<(&'static str, &'a str)> {
        let mut reached: HashSet<(&'static str, &'a str)> = HashSet::new();
        let mut pending: Vec<&Value> = vec![schema];
        while let Some(node) = pending.pop() {
            for pointer in local_pointers(node) {
                let Some((block, name, entry)) = self.entry_of(pointer) else {
                    continue;
                };
                if reached.insert((block, name)) {
                    pending.push(entry);
                }
            }
        }
        reached
    }

    /// What a fragment of `schema` would cost, the leaf's schema and the entries
    /// [`Self::reached_from`] finds each sized by `size`, within `limit`: `None` as soon as the
    /// running count passes it, so a leaf a budget refuses costs the walk up to the budget and no
    /// more.
    pub(super) fn reached_within(
        self,
        schema: &Value,
        limit: usize,
        mut size: impl FnMut(&Value) -> usize,
    ) -> Option<usize> {
        let mut cost = size(schema);
        if cost > limit {
            return None;
        }
        let mut seen: HashSet<(&'static str, &'a str)> = HashSet::new();
        let mut pending: Vec<&Value> = vec![schema];
        while let Some(node) = pending.pop() {
            for pointer in local_pointers(node) {
                let Some((block, name, entry)) = self.entry_of(pointer) else {
                    continue;
                };
                if !seen.insert((block, name)) {
                    continue;
                }
                cost += size(entry);
                if cost > limit {
                    return None;
                }
                pending.push(entry);
            }
        }
        Some(cost)
    }

    /// The block, the entry name and the entry a local pointer names, with the pointer's escapes
    /// undone (`~1` for `/`, `~0` for `~`), so an entry named `a/b` is followed by its name.
    fn entry_of(self, pointer: &str) -> Option<(&'static str, &'a str, &'a Value)> {
        let mut segments = pointer.strip_prefix("#/")?.split('/');
        let (block, entries) = match segments.next()? {
            "$defs" => ("$defs", self.defs?),
            "definitions" => ("definitions", self.definitions?),
            _ => return None,
        };
        let name = segments.next()?.replace("~1", "/").replace("~0", "~");
        let (key, entry) = entries.as_object()?.get_key_value(name.as_str())?;
        Some((block, key.as_str(), entry))
    }
}

/// The keywords that annotate a schema and constrain nothing.
const ANNOTATIONS: [&str; 4] = ["title", "description", "examples", "$comment"];
/// The keywords whose value is one schema, or a list of schemas (`items` in its tuple form).
const ONE_SCHEMA: [&str; 11] = [
    "items",
    "additionalItems",
    "additionalProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
];
/// The keywords whose value maps names to schemas (`dependencies` to a schema or a list of
/// names, which passes as it is).
const NAMED_SCHEMAS: [&str; 6] = [
    "properties",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "$defs",
    "definitions",
];
/// The keywords whose value lists schemas.
const LISTED_SCHEMAS: [&str; 4] = ["anyOf", "oneOf", "allOf", "prefixItems"];

/// `schema` with its annotations dropped at every schema position (a property named
/// `description` is a name, not an annotation, and stays), as the engine gets it.
fn without_annotations(schema: &Value) -> Value {
    let Value::Object(map) = schema else {
        return schema.clone();
    };
    let mut stripped = serde_json::Map::with_capacity(map.len());
    for (key, value) in map {
        if ANNOTATIONS.contains(&key.as_str()) {
            continue;
        }
        let value = if ONE_SCHEMA.contains(&key.as_str()) {
            match value {
                Value::Array(items) => {
                    Value::Array(items.iter().map(without_annotations).collect())
                }
                one => without_annotations(one),
            }
        } else if NAMED_SCHEMAS.contains(&key.as_str()) {
            match value {
                Value::Object(named) => Value::Object(
                    named
                        .iter()
                        .map(|(name, schema)| (name.clone(), without_annotations(schema)))
                        .collect(),
                ),
                other => other.clone(),
            }
        } else if LISTED_SCHEMAS.contains(&key.as_str()) {
            match value {
                Value::Array(items) => {
                    Value::Array(items.iter().map(without_annotations).collect())
                }
                other => other.clone(),
            }
        } else {
            value.clone()
        };
        stripped.insert(key.clone(), value);
    }
    Value::Object(stripped)
}

/// The local pointers in `schema`, at any depth, in document order.
fn local_pointers(schema: &Value) -> Vec<&str> {
    let mut pointers = Vec::new();
    collect_local_pointers(schema, &mut pointers);
    pointers
}

fn collect_local_pointers<'v>(schema: &'v Value, pointers: &mut Vec<&'v str>) {
    match schema {
        Value::Object(map) => {
            for (key, value) in map {
                if is_reference(key) {
                    if let Some(pointer) = value.as_str().filter(|pointer| pointer.starts_with('#'))
                    {
                        pointers.push(pointer);
                    }
                }
                collect_local_pointers(value, pointers);
            }
        }
        Value::Array(items) => items
            .iter()
            .for_each(|item| collect_local_pointers(item, pointers)),
        _ => {}
    }
}

/// About how many bytes `schema` takes as the engine gets it, its annotations dropped
/// ([`json_bytes`] of [`without_annotations`]): what a number leaf, or a definition it reaches,
/// costs an invoke's byte budget before it is written.
pub(super) fn sent_bytes(schema: &Value) -> usize {
    json_bytes(&without_annotations(schema))
}

/// About how many bytes `value` takes serialized, the escapes counted.
fn json_bytes(value: &Value) -> usize {
    match value {
        Value::Null => 4,
        Value::Bool(flag) => {
            if *flag {
                4
            } else {
                5
            }
        }
        Value::Number(number) => number.to_string().len(),
        Value::String(text) => {
            text.len()
                + 2
                + text
                    .bytes()
                    .filter(|byte| matches!(byte, b'"' | b'\\') || *byte < 0x20)
                    .count()
        }
        Value::Array(items) => {
            2 + items.iter().map(json_bytes).sum::<usize>() + items.len().saturating_sub(1)
        }
        Value::Object(object) => {
            2 + object
                .iter()
                .map(|(key, value)| key.len() + 3 + json_bytes(value))
                .sum::<usize>()
                + object.len().saturating_sub(1)
        }
    }
}

/// Whether `key` is one of the keywords that point at another schema.
fn is_reference(key: &str) -> bool {
    matches!(key, "$ref" | "$dynamicRef" | "$recursiveRef")
}

/// Whether `schema` points anywhere inside its own document.
pub(super) fn has_local_reference(schema: &Value) -> bool {
    match schema {
        Value::Object(map) => map.iter().any(|(key, value)| {
            is_reference(key)
                && value
                    .as_str()
                    .is_some_and(|pointer| pointer.starts_with('#'))
                || has_local_reference(value)
        }),
        Value::Array(items) => items.iter().any(has_local_reference),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json as value;

    use super::*;

    #[test]
    fn a_leaf_carries_the_definitions_its_pointers_reach_and_no_other() {
        // `count` reaches `Count` alone; `chain` reaches `Chain`, and through it `Deep`; `Unused`
        // travels with neither, and a `definitions` block is read the same way.
        let root = value!({
            "type": "object",
            "properties": {
                "count": {"$ref": "#/$defs/Count"},
                "chain": {"$ref": "#/$defs/Chain"},
                "old": {"$ref": "#/definitions/Old"},
            },
            "$defs": {
                "Count": {"type": "integer"},
                "Chain": {"type": "object", "properties": {"deep": {"$ref": "#/$defs/Deep"}}},
                "Deep": {"type": "number"},
                "Unused": {"type": "string"},
            },
            "definitions": {"Old": {"type": "integer"}, "Older": {"type": "integer"}},
        });
        let definitions = Definitions::of(&root);
        assert_eq!(
            definitions.attached_to(&root["properties"]["count"]),
            value!({"$ref": "#/$defs/Count", "$defs": {"Count": {"type": "integer"}}})
        );
        assert_eq!(
            definitions.attached_to(&root["properties"]["chain"]),
            value!({"$ref": "#/$defs/Chain", "$defs": {
                "Chain": {"type": "object", "properties": {"deep": {"$ref": "#/$defs/Deep"}}},
                "Deep": {"type": "number"},
            }})
        );
        assert_eq!(
            definitions.attached_to(&root["properties"]["old"]),
            value!({"$ref": "#/definitions/Old", "definitions": {"Old": {"type": "integer"}}})
        );
    }

    #[test]
    fn annotations_are_dropped_at_schema_positions_and_kept_as_property_names() {
        let root = value!({"$defs": {"Count": {
            "type": "integer", "title": "Count", "description": "how many", "examples": [1, 2],
        }}});
        let definitions = Definitions::of(&root);
        let schema = value!({
            "type": "object",
            "description": "a long one",
            "$comment": "noted",
            "properties": {
                "description": {"type": "string", "description": "the field named so"},
                "n": {"$ref": "#/$defs/Count"},
            },
            "anyOf": [{"title": "a", "required": ["n"]}],
            "items": {"description": "x", "type": "string"},
        });
        assert_eq!(
            definitions.attached_to(&schema),
            value!({
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "n": {"$ref": "#/$defs/Count"},
                },
                "anyOf": [{"required": ["n"]}],
                "items": {"type": "string"},
                "$defs": {"Count": {"type": "integer"}},
            })
        );
    }

    #[test]
    fn an_entry_whose_name_holds_a_slash_is_followed_to_what_it_points_at() {
        // A pointer spells `a/b` as `a~1b`; the entry is followed by its name, so what it points
        // at travels too.
        let root = value!({"$defs": {
            "a/b": {"type": "object", "properties": {"x": {"$ref": "#/$defs/Inner"}}},
            "Inner": {"type": "integer"},
            "Unused": {"type": "string"},
        }});
        let definitions = Definitions::of(&root);
        assert_eq!(
            definitions.attached_to(&value!({"$ref": "#/$defs/a~1b"})),
            value!({"$ref": "#/$defs/a~1b", "$defs": {
                "a/b": {"type": "object", "properties": {"x": {"$ref": "#/$defs/Inner"}}},
                "Inner": {"type": "integer"},
            }})
        );
    }

    #[test]
    fn annotations_drop_from_tuple_items_additional_items_and_dependencies() {
        let root = value!({});
        let definitions = Definitions::of(&root);
        let schema = value!({
            "type": "array",
            "items": [
                {"type": "string", "title": "first"},
                {"type": "integer", "description": "d"},
            ],
            "additionalItems": {"type": "null", "examples": [null]},
            "dependencies": {
                "a": {"required": ["b"], "description": "needs b"},
                "b": ["c"],
            },
        });
        assert_eq!(
            definitions.attached_to(&schema),
            value!({
                "type": "array",
                "items": [{"type": "string"}, {"type": "integer"}],
                "additionalItems": {"type": "null"},
                "dependencies": {"a": {"required": ["b"]}, "b": ["c"]},
            })
        );
    }

    #[test]
    fn a_schema_without_pointers_still_loses_its_annotations_and_gains_no_definitions() {
        let root = value!({"$defs": {"Unused": {"type": "string"}}});
        let definitions = Definitions::of(&root);
        assert_eq!(
            definitions.attached_to(&value!({"type": "integer", "description": "d"})),
            value!({"type": "integer"})
        );
    }
}
