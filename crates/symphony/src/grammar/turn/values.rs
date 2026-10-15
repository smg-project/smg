//! What a call's parameter value may be, read from the tool's parameters schema: the JSON types it
//! takes, each under the part of the schema that types it, or a constant.
//!
//! A value takes one option per alternative the schema spells: each `const` and `enum` value, each
//! branch of an `anyOf` or `oneOf`, each name of a type array. `$ref`s resolve inside the tool's
//! schema, each target expanded once per value (an alternative repeated adds nothing, and a
//! definition many branches share would otherwise expand once per path to it); an `allOf` of one
//! schema is that schema, and of several any value. An object or array schema without `type` is
//! typed by `properties`, `additionalProperties` or `items`, and `"nullable": true`, OpenAPI's
//! optional value, adds `null`. Type names take the aliases tool schemas use (`int`, `dict`,
//! `list`, ...). A schema that constrains nothing, a type name none of those is, or references
//! that only lead back into the value, take any value.

use std::{borrow::Cow, collections::HashSet};

use serde_json::{json, Map, Value};

/// How deep `$ref`s and combinators are followed within one value.
const MAX_DEPTH: usize = 32;

/// The schema that takes any value.
pub(super) static ANY: Value = Value::Bool(true);

/// The type of a value option.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    String,
    Integer,
    Number,
    Boolean,
    Null,
    Object,
    Array,
}

impl Kind {
    /// The types of JSON values; an integer is a number.
    const ALL: [Self; 6] = [
        Self::String,
        Self::Number,
        Self::Boolean,
        Self::Null,
        Self::Object,
        Self::Array,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Integer => "integer",
            Self::Number => "number",
            Self::Boolean => "boolean",
            Self::Null => "null",
            Self::Object => "object",
            Self::Array => "array",
        }
    }

    /// The schema of every value of this type.
    pub(super) fn any_schema(self) -> Value {
        match self {
            Self::Object => json!({"type": "object", "additionalProperties": true}),
            Self::Array => json!({"type": "array", "items": true}),
            kind => json!({"type": kind.name()}),
        }
    }

    /// A type name as tool schemas write it, aliases included.
    fn parse(name: &str) -> Option<Self> {
        let name = name.trim().to_ascii_lowercase();
        let starts = |prefixes: &[&str]| prefixes.iter().any(|prefix| name.starts_with(prefix));
        Some(match name.as_str() {
            "string" | "str" | "text" | "varchar" | "char" | "enum" => Self::String,
            "integer" | "int" => Self::Integer,
            "number" | "float" | "double" => Self::Number,
            "boolean" | "bool" | "binary" => Self::Boolean,
            "object" | "dict" | "map" => Self::Object,
            "array" | "arr" | "list" | "sequence" => Self::Array,
            "null" => Self::Null,
            _ if starts(&["int", "uint", "long", "short", "unsigned"]) => Self::Integer,
            _ if starts(&["num", "float"]) => Self::Number,
            _ if starts(&["dict"]) => Self::Object,
            _ if starts(&["list"]) => Self::Array,
            _ => return None,
        })
    }

    fn of(value: &Value) -> Self {
        match value {
            Value::String(_) => Self::String,
            Value::Number(number) if number.is_f64() => Self::Number,
            Value::Number(_) => Self::Integer,
            Value::Bool(_) => Self::Boolean,
            Value::Null => Self::Null,
            Value::Object(_) => Self::Object,
            Value::Array(_) => Self::Array,
        }
    }
}

/// One option of a value.
#[derive(Clone, Debug)]
pub(super) struct ValueOption<'a> {
    pub(super) kind: Kind,
    pub(super) source: Source<'a>,
}

/// Where a [`ValueOption`] comes from.
#[derive(Clone, Debug)]
pub(super) enum Source<'a> {
    /// The values of the option's type under this schema ([`narrowed`] sets the type on it).
    Schema(&'a Map<String, Value>),
    /// A `const` or `enum` value.
    Literal(&'a Value),
    /// Every value of the option's type.
    Any,
}

impl<'a> ValueOption<'a> {
    fn typed(schema: &'a Map<String, Value>, kind: Kind) -> Self {
        Self {
            kind,
            source: Source::Schema(schema),
        }
    }

    fn literal(value: &'a Value) -> Self {
        Self {
            kind: Kind::of(value),
            source: Source::Literal(value),
        }
    }

    fn any() -> Vec<Self> {
        Kind::ALL
            .into_iter()
            .map(|kind| Self {
                kind,
                source: Source::Any,
            })
            .collect()
    }
}

/// `schema` with its `type` the option's, copied only when that changes it.
pub(super) fn narrowed(schema: &Map<String, Value>, kind: Kind) -> Cow<'_, Map<String, Value>> {
    if schema.get("type").and_then(Value::as_str) == Some(kind.name()) {
        Cow::Borrowed(schema)
    } else {
        let mut schema = schema.clone();
        schema.insert("type".to_string(), Value::String(kind.name().to_string()));
        Cow::Owned(schema)
    }
}

/// A tool's parameters schema, which local `$ref`s point into.
#[derive(Clone, Copy, Debug)]
pub(super) struct Root<'a>(pub(super) &'a Value);

/// The reference targets one value's options expanded, and whether one was skipped as expanded.
#[derive(Default)]
struct Expanded {
    targets: HashSet<*const Value>,
    skipped: bool,
}

impl<'a> Root<'a> {
    /// The parameters schema past its `$ref`s and one-schema `allOf`s; one that does not resolve,
    /// or resolves too deep, takes any value.
    pub(super) fn resolved(self) -> &'a Value {
        let mut schema = self.0;
        for _ in 0..=MAX_DEPTH {
            if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
                match self.target(reference) {
                    Some(target) => schema = target,
                    None => return &ANY,
                }
                continue;
            }
            match schema
                .get("allOf")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
            {
                Some([only]) => schema = only,
                _ => return schema,
            }
        }
        &ANY
    }

    fn target(self, reference: &str) -> Option<&'a Value> {
        self.0.pointer(reference.strip_prefix('#')?)
    }

    /// Every option of a value under `schema`.
    pub(super) fn options(self, schema: &'a Value) -> Vec<ValueOption<'a>> {
        let mut expanded = Expanded::default();
        let mut options = self.options_at(schema, 0, &mut expanded);
        if options.is_empty() && expanded.skipped {
            options = ValueOption::any();
        }
        let nullable = schema.get("nullable").and_then(Value::as_bool) == Some(true);
        if nullable && !options.iter().any(|option| option.kind == Kind::Null) {
            options.push(ValueOption {
                kind: Kind::Null,
                source: Source::Any,
            });
        }
        options
    }

    fn options_at(
        self,
        schema: &'a Value,
        depth: usize,
        expanded: &mut Expanded,
    ) -> Vec<ValueOption<'a>> {
        let schema = match schema {
            _ if depth > MAX_DEPTH => return ValueOption::any(),
            Value::Bool(false) => return Vec::new(),
            Value::Object(schema) => schema,
            _ => return ValueOption::any(),
        };
        if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
            let Some(target) = self.target(reference) else {
                return ValueOption::any();
            };
            if !expanded.targets.insert(std::ptr::from_ref(target)) {
                expanded.skipped = true;
                return Vec::new();
            }
            return self.options_at(target, depth + 1, expanded);
        }
        if let Some(value) = schema.get("const") {
            return vec![ValueOption::literal(value)];
        }
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            return values.iter().map(ValueOption::literal).collect();
        }
        if let Some(branches) = schema
            .get("anyOf")
            .or_else(|| schema.get("oneOf"))
            .and_then(Value::as_array)
        {
            return branches
                .iter()
                .flat_map(|branch| self.options_at(branch, depth + 1, expanded))
                .collect();
        }
        if let Some(all) = schema.get("allOf").and_then(Value::as_array) {
            return match all.as_slice() {
                [only] => self.options_at(only, depth + 1, expanded),
                _ => ValueOption::any(),
            };
        }
        // A type name that names no JSON type takes any value: the call stays writable, a
        // required parameter of it included.
        match schema.get("type") {
            Some(Value::String(name)) => match Kind::parse(name) {
                Some(kind) => vec![ValueOption::typed(schema, kind)],
                None => ValueOption::any(),
            },
            Some(Value::Array(names)) => {
                let kinds: Option<Vec<Kind>> = names
                    .iter()
                    .map(|name| name.as_str().and_then(Kind::parse))
                    .collect();
                match kinds {
                    Some(kinds) => kinds
                        .into_iter()
                        .map(|kind| ValueOption::typed(schema, kind))
                        .collect(),
                    None => ValueOption::any(),
                }
            }
            Some(_) => ValueOption::any(),
            None if schema.contains_key("properties")
                || schema.contains_key("additionalProperties") =>
            {
                vec![ValueOption::typed(schema, Kind::Object)]
            }
            None if schema.contains_key("items") => vec![ValueOption::typed(schema, Kind::Array)],
            None => ValueOption::any(),
        }
    }
}

/// `schema` with the `properties` of every object it describes listed in the order of their
/// names, nested ones, branches, items and definitions included (constants, enums, defaults and
/// examples are values, not schemas, and keep their keys); whether that changed it.
pub(super) fn sort_properties(schema: &mut Value) -> bool {
    const VALUES: [&str; 4] = ["const", "enum", "default", "examples"];
    let mut changed = false;
    match schema {
        Value::Object(map) => {
            if let Some(Value::Object(properties)) = map.get_mut("properties") {
                let mut names: Vec<&String> = properties.keys().collect();
                names.sort_unstable();
                if !names.iter().copied().eq(properties.keys()) {
                    let mut entries: Vec<(String, Value)> =
                        std::mem::take(properties).into_iter().collect();
                    entries.sort_by(|(a, _), (b, _)| a.cmp(b));
                    properties.extend(entries);
                    changed = true;
                }
            }
            for (key, member) in map.iter_mut() {
                if !VALUES.contains(&key.as_str()) {
                    changed |= sort_properties(member);
                }
            }
        }
        Value::Array(members) => {
            for member in members {
                changed |= sort_properties(member);
            }
        }
        _ => {}
    }
    changed
}

/// Whether xgrammar's regex converter compiles `pattern`, as far as its syntax tells: no
/// lookaround, no group modifier other than `(?:`, no backreference, no Unicode property and no
/// word boundary. xgrammar refuses those (lookahead and `\p` fail the whole structural tag on
/// 0.2.3 and 0.2.8), so a pattern holding one is left out and the value it types is free text.
pub(super) fn compiles(pattern: &str) -> bool {
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some('p' | 'P' | 'b' | 'B' | 'k' | '1'..='9') | None => return false,
                Some(_) => {}
            },
            '(' if chars.peek() == Some(&'?') => {
                chars.next();
                if chars.next() != Some(':') {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
}

/// `schema` without the `pattern`s xgrammar does not compile ([`compiles`]), at every depth;
/// whether it held one.
pub(super) fn drop_uncompiled_patterns(schema: &mut Value) -> bool {
    const VALUES: [&str; 4] = ["const", "enum", "default", "examples"];
    let mut changed = false;
    match schema {
        Value::Object(map) => {
            if map
                .get("pattern")
                .and_then(Value::as_str)
                .is_some_and(|pattern| !compiles(pattern))
            {
                map.remove("pattern");
                changed = true;
            }
            for (key, member) in map.iter_mut() {
                if !VALUES.contains(&key.as_str()) {
                    changed |= drop_uncompiled_patterns(member);
                }
            }
        }
        Value::Array(members) => {
            for member in members {
                changed |= drop_uncompiled_patterns(member);
            }
        }
        _ => {}
    }
    changed
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn kinds(schema: &Value) -> Vec<Kind> {
        Root(schema)
            .options(schema)
            .iter()
            .map(|option| option.kind)
            .collect()
    }

    #[test]
    fn a_type_name_no_json_type_is_takes_any_value() {
        let all = Kind::ALL.to_vec();
        assert_eq!(kinds(&json!({"type": "tuple"})), all);
        assert_eq!(kinds(&json!({"type": "any"})), all);
        assert_eq!(kinds(&json!({"type": ["string", "tuple"]})), all);
        assert_eq!(
            kinds(&json!({"type": ["string", "null"]})),
            [Kind::String, Kind::Null]
        );
        assert_eq!(kinds(&json!({"type": "int64"})), [Kind::Integer]);
        assert_eq!(kinds(&json!({"type": "Dict[str, int]"})), [Kind::Object]);
    }

    #[test]
    fn references_expand_once_and_unions_split() {
        let schema = json!({
            "$defs": {"Unit": {"enum": ["c", "f"]}},
            "properties": {"u": {"anyOf": [{"$ref": "#/$defs/Unit"}, {"$ref": "#/$defs/Unit"}]}}
        });
        let value = &schema["properties"]["u"];
        let options = Root(&schema).options(value);
        assert_eq!(
            options.len(),
            2,
            "the second reference to the same target adds nothing"
        );
        assert!(matches!(options[0].source, Source::Literal(_)));
        let cyclic = json!({"$defs": {"A": {"$ref": "#/$defs/A"}}, "$ref": "#/$defs/A"});
        assert_eq!(kinds(&cyclic), Kind::ALL.to_vec());
        assert_eq!(
            kinds(&json!({"type": "integer", "nullable": true})),
            [Kind::Integer, Kind::Null]
        );
    }

    #[test]
    fn properties_sort_by_name_at_every_depth_but_inside_values() {
        let mut schema = json!({
            "type": "object",
            "properties": {"end": {"type": "string"}, "start": {"type": "string"},
                           "inner": {"type": "object", "properties": {"b": {}, "a": {}}}},
            "default": {"z": 1, "a": 2}
        });
        assert!(sort_properties(&mut schema));
        let keys = |value: &Value| {
            value
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<Vec<_>>()
        };
        assert_eq!(keys(&schema["properties"]), ["end", "inner", "start"]);
        assert_eq!(
            keys(&schema["properties"]["inner"]["properties"]),
            ["a", "b"]
        );
        assert_eq!(keys(&schema["default"]), ["z", "a"]);
        assert!(!sort_properties(&mut schema), "sorted already");
    }

    #[test]
    fn patterns_xgrammar_refuses_are_told_apart() {
        for pattern in [
            r"^[A-Z]{3}$",
            r"(?:ab)+",
            r"[\w.+-]+@[\w-]+\.[\w.]+",
            r"\d{2}\s",
        ] {
            assert!(compiles(pattern), "{pattern}");
        }
        for pattern in [
            r"^(?!\s*$).+$",
            r"^\p{L}+$",
            r"(?<=a)b",
            r"(a)\1",
            r"\bword\b",
            r"(?i)x",
        ] {
            assert!(!compiles(pattern), "{pattern}");
        }
        let mut schema = json!({"type": "object", "properties": {
            "a": {"type": "string", "pattern": r"^(?!x)"},
            "b": {"type": "string", "pattern": "^b$"}
        }});
        assert!(drop_uncompiled_patterns(&mut schema));
        assert!(schema["properties"]["a"].get("pattern").is_none());
        assert_eq!(schema["properties"]["b"]["pattern"], "^b$");
    }
}
