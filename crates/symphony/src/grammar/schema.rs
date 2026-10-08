//! What a tool's JSON schema says about each argument, for the syntaxes that write arguments one
//! by one with a type of their own (Kimi K3's XTML, the keyed tags): the type a property's schema
//! pins and the grammar of a value of that type, and the definitions a property's schema carries
//! along so a `$ref` inside it still resolves when the engine compiles it on its own; and the walk
//! over a tool's properties the syntaxes share: one argument per property, in the schema's order,
//! optional unless the schema requires it.

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

/// The type a property's schema pins, by the name the templates write (`integer` as `number`),
/// and the grammar of a value of that type: a string as `text`, the syntax's own spelling of a
/// value written as it is; a string `enum` as one of its values; a number, an object or an array
/// as JSON the property's schema accepts, the tool's definitions attached; a boolean as JSON;
/// `null` as the word. `None` when the schema pins no type the syntaxes have, or a `$ref` in it
/// points at nothing (a grammar with such a pointer would not compile, and the whole call with
/// it). A `$ref` lends its target's keywords, and the property's own win, as in JSON Schema
/// 2020-12.
pub(super) fn shape<'a>(
    schema: &'a Value,
    definitions: Definitions<'a>,
    text: impl FnOnce() -> Grammar,
) -> Option<(&'static str, Grammar)> {
    let definitions = definitions.within(schema);
    if !definitions.pointers_resolve(schema) {
        return None;
    }
    let target = definitions.target_of(schema);
    let keyword = |name: &str| {
        schema
            .get(name)
            .or_else(|| target.and_then(|target| target.get(name)))
    };
    if let Some(values) = keyword("enum").and_then(Value::as_array) {
        if !values.is_empty() && values.iter().all(Value::is_string) {
            let options = values
                .iter()
                .filter_map(Value::as_str)
                .map(|value| Grammar::ConstString(value.to_string()))
                .collect();
            return Some(("string", Grammar::Or(options)));
        }
    }
    let json = |schema: Value| Grammar::JsonSchema {
        schema,
        style: None,
    };
    match keyword("type").and_then(Value::as_str)? {
        "string" => Some(("string", text())),
        "integer" | "number" => Some(("number", json(definitions.attached_to(schema)))),
        "boolean" => Some(("boolean", json(json!({"type": "boolean"})))),
        "object" => Some(("object", json(definitions.attached_to(schema)))),
        "array" => Some(("array", json(definitions.attached_to(schema)))),
        "null" => Some(("null", Grammar::ConstString("null".to_string()))),
        _ => None,
    }
}

/// The definitions a property's schema can point into: its own `$defs` and `definitions` first,
/// then the tool's root ones, which is the document the engine gets ([`Self::attached_to`]). Each
/// property's schema goes to the engine as a document of its own, and a `$ref` such as
/// `#/$defs/Node` points from the root of the document it stands in, so the definitions travel
/// with the property's schema. A pointer to the root itself, `#`, names the tool's schema in the
/// tool and the property in the property's document, so it resolves to nothing here, and the
/// property takes any value.
#[derive(Clone, Copy)]
pub(super) struct Definitions<'a> {
    own: Blocks<'a>,
    root: Blocks<'a>,
}

/// A schema's two definition blocks.
#[derive(Clone, Copy)]
struct Blocks<'a> {
    defs: Option<&'a Value>,
    definitions: Option<&'a Value>,
}

impl<'a> Blocks<'a> {
    fn of(schema: &'a Value) -> Self {
        Self {
            defs: schema.get("$defs"),
            definitions: schema.get("definitions"),
        }
    }

    fn named(self, block: &str) -> Option<&'a Value> {
        match block {
            "$defs" => self.defs,
            "definitions" => self.definitions,
            _ => None,
        }
    }
}

/// How deep a chain of pointers the check follows before it gives up on a schema: a tool whose
/// definitions chain deeper pins nothing, and the walk's depth stays bounded whatever a request
/// sends. Breadth is not counted: a schema may point at any number of definitions, each walked
/// once.
pub(super) const POINTER_DEPTH_AT_MOST: usize = 64;

impl<'a> Definitions<'a> {
    pub(super) fn of(root: &'a Value) -> Self {
        Self {
            own: Blocks::of(root),
            root: Blocks::of(root),
        }
    }

    /// The definitions as the property `schema` sees them: its own blocks first, then the root's.
    fn within(self, schema: &'a Value) -> Self {
        Self {
            own: Blocks::of(schema),
            root: self.root,
        }
    }

    /// The schema a local pointer names: `#/$defs/Name` or `#/definitions/Name` an entry, the
    /// property's own block before the root's, with the pointer's escapes (`~1` for `/`, `~0` for
    /// `~`) undone; `#` itself nothing, since it would name the property's own document once the
    /// engine compiles it alone.
    fn resolve(self, pointer: &str) -> Option<&'a Value> {
        let mut segments = pointer.strip_prefix("#/")?.split('/');
        let block = segments.next()?;
        let name = segments.next()?.replace("~1", "/").replace("~0", "~");
        // A block the property has but that is no object (`"$defs": null`) stays as it is in the
        // document the engine gets, with no entry to find, so nothing resolves through it.
        let mut node = match self.own.named(block) {
            Some(Value::Object(entries)) => match entries.get(&name) {
                Some(entry) => entry,
                None => self.root.named(block)?.get(&name)?,
            },
            Some(_) => return None,
            None => self.root.named(block)?.get(&name)?,
        };
        for segment in segments {
            node = node.get(segment.replace("~1", "/").replace("~0", "~"))?;
        }
        Some(node)
    }

    /// The schema the property's own `$ref` names, one hop: its keywords stand beside the
    /// property's, and a cycle of definitions is the engine's to expand, not this function's.
    fn target_of(self, schema: &'a Value) -> Option<&'a Value> {
        self.resolve(schema.get("$ref")?.as_str()?)
    }

    /// Whether every local pointer in `schema` names something, and every pointer in what it
    /// names, as far as the pointers reach: the definitions travel with the property, so a pointer
    /// at nothing inside one of them reaches the engine too. Each distinct pointer is walked once
    /// (`followed` remembers them, so a definition that names itself, or a wide schema, costs one
    /// walk per definition), and a chain is followed no deeper than [`POINTER_DEPTH_AT_MOST`].
    fn pointers_resolve(self, schema: &Value) -> bool {
        self.pointers_resolve_from(schema, &mut Vec::new(), 0)
    }

    fn pointers_resolve_from(
        self,
        schema: &Value,
        followed: &mut Vec<String>,
        depth: usize,
    ) -> bool {
        match schema {
            Value::Object(map) => map.iter().all(|(key, value)| {
                let local = value
                    .as_str()
                    .filter(|pointer| is_reference(key) && pointer.starts_with('#'));
                let named_resolves = local.is_none_or(|pointer| match self.resolve(pointer) {
                    None => false,
                    Some(_) if followed.iter().any(|seen| seen == pointer) => true,
                    Some(_) if depth >= POINTER_DEPTH_AT_MOST => false,
                    Some(target) => {
                        followed.push(pointer.to_string());
                        self.pointers_resolve_from(target, followed, depth + 1)
                    }
                });
                named_resolves && self.pointers_resolve_from(value, followed, depth)
            }),
            Value::Array(items) => items
                .iter()
                .all(|item| self.pointers_resolve_from(item, followed, depth)),
            _ => true,
        }
    }

    /// The property's schema with the root's definitions attached, so a local pointer in it still
    /// resolves when the engine compiles it on its own. A definition the schema declares itself
    /// stays; the root's fill in the names it lacks. A schema with no local pointer goes as it is.
    fn attached_to(self, schema: &Value) -> Value {
        let mut attached = schema.clone();
        let root = self.root;
        if root.defs.is_none() && root.definitions.is_none() || !has_local_reference(schema) {
            return attached;
        }
        let Some(object) = attached.as_object_mut() else {
            return attached;
        };
        for (name, block) in [("$defs", root.defs), ("definitions", root.definitions)] {
            let Some(Value::Object(entries)) = block else {
                continue;
            };
            match object.get_mut(name) {
                Some(Value::Object(own)) => {
                    for (key, value) in entries {
                        own.entry(key.clone()).or_insert_with(|| value.clone());
                    }
                }
                Some(_) => {}
                None => {
                    object.insert(name.to_string(), Value::Object(entries.clone()));
                }
            }
        }
        attached
    }
}

/// Whether `key` is one of the keywords that point at another schema.
fn is_reference(key: &str) -> bool {
    matches!(key, "$ref" | "$dynamicRef" | "$recursiveRef")
}

/// Whether `schema` points anywhere inside its own document.
fn has_local_reference(schema: &Value) -> bool {
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
