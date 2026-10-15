//! What an output grammar is built into: the grammar itself ([`Tree`]), or the length of the text
//! [`Grammar::payload_text`] writes for it ([`Length`]), counted without building it. One builder
//! makes both, so a turn over a size bound is refused before anything of it is built: the
//! lengths of the pieces a grammar repeats (a value in each list that holds it, a call in each
//! place one is written) are added up, not copied.

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use serde_json::{json, Map, Value};

use super::values::{self, Kind, Root};
use crate::grammar::{json_len, schema::without_annotations, Bound, Grammar, Tag};

/// The two groups of definitions a tool's schema may carry, which each JSON value of it takes
/// along so its `$ref`s resolve.
const DEFINITIONS: [&str; 2] = ["$defs", "definitions"];

/// One JSON value of a parameter, before it is written: the values of a type under a part of the
/// tool's schema, a constant, or every value of a type.
#[derive(Clone, Copy, Debug)]
pub(super) enum Json<'a> {
    Schema(&'a Map<String, Value>, Kind),
    Literal(&'a Value),
    Any(Kind),
}

impl Json<'_> {
    /// The schema of the value, without the tool's definitions.
    fn own(self) -> Value {
        match self {
            Self::Schema(schema, kind) => {
                Value::Object(values::narrowed(schema, kind).into_owned())
            }
            Self::Literal(value) => json!({"const": value}),
            Self::Any(kind) => kind.any_schema(),
        }
    }
}

/// What an output grammar is built into, node by node.
pub(super) trait Build {
    type Node: Clone;

    /// A leaf, as it is.
    fn leaf(&self, grammar: Grammar) -> Self::Node;
    fn sequence(&self, elements: Vec<Self::Node>) -> Self::Node;
    fn or(&self, elements: Vec<Self::Node>) -> Self::Node;
    fn optional(&self, content: Self::Node) -> Self::Node;
    fn star(&self, content: Self::Node) -> Self::Node;
    fn tag(&self, begin: String, content: Self::Node, end: Bound) -> Self::Node;
    /// JSON of `schema`, the request's answer format.
    fn schema(&self, schema: &Value) -> Self::Node;
    /// JSON of one value of a call, which carries the definitions of `root`, the tool's schema,
    /// so its `$ref`s resolve, without the patterns xgrammar does not compile, its strings
    /// holding none of `excludes`.
    fn value(&self, json: Json<'_>, root: Root<'_>, excludes: &[String]) -> Self::Node;
}

/// The parts of a turn's JSON values, each read once: a value's own schema, and each definition
/// of a tool's schema.
#[derive(Default)]
struct Parts {
    own: RefCell<HashMap<OwnKey, Rc<Part>>>,
    definitions: RefCell<HashMap<*const Value, Rc<Definitions>>>,
}

/// A value's own schema: the schema it is typed under, and its type.
type OwnKey = (*const Map<String, Value>, Kind);

impl Parts {
    /// The value's own schema, without the tool's definitions.
    fn own(&self, json: Json<'_>) -> Rc<Part> {
        let key = match json {
            Json::Schema(schema, kind) => Some((std::ptr::from_ref(schema), kind)),
            Json::Literal(_) | Json::Any(_) => None,
        };
        if let Some(part) = key.and_then(|key| self.own.borrow().get(&key).cloned()) {
            return part;
        }
        let part = Rc::new(Part::of(json.own()));
        if let Some(key) = key {
            self.own.borrow_mut().insert(key, part.clone());
        }
        part
    }

    /// The definitions of `root`, a tool's schema.
    fn definitions(&self, root: Root<'_>) -> Rc<Definitions> {
        let key = std::ptr::from_ref(root.0);
        if let Some(definitions) = self.definitions.borrow().get(&key) {
            return definitions.clone();
        }
        let definitions = Rc::new(Definitions::of(root.0));
        self.definitions
            .borrow_mut()
            .insert(key, definitions.clone());
        definitions
    }
}

/// A part of a value's schema as it is written: without its annotations and the patterns
/// xgrammar does not compile; its length; and the definitions it refers to.
struct Part {
    schema: Value,
    len: usize,
    refs: Refs,
}

impl Part {
    fn of(schema: Value) -> Self {
        let mut schema = without_annotations(&schema);
        values::drop_uncompiled_patterns(&mut schema);
        let refs = Refs::of(&schema);
        Self {
            len: json_len(&schema),
            schema,
            refs,
        }
    }
}

/// The definitions a part refers to: each `$ref` to `#/$defs/NAME` or `#/definitions/NAME`, or
/// below one, names one; any other reference may reach any of them.
#[derive(Default)]
struct Refs {
    to: Vec<(usize, String)>,
    any: bool,
}

impl Refs {
    fn of(schema: &Value) -> Self {
        let mut refs = Self::default();
        refs.add(schema);
        refs
    }

    fn add(&mut self, value: &Value) {
        match value {
            Value::Object(map) => {
                for (key, member) in map {
                    match (key.as_str(), member) {
                        ("$ref", Value::String(pointer)) => match definition(pointer) {
                            Some(named) => self.to.push(named),
                            None => self.any = true,
                        },
                        ("$dynamicRef" | "$recursiveRef", _) => self.any = true,
                        _ => self.add(member),
                    }
                }
            }
            Value::Array(items) => items.iter().for_each(|item| self.add(item)),
            _ => {}
        }
    }
}

/// The group and name of the definition `pointer` points at or below, or `None` for any other
/// pointer and for a name with an escape.
fn definition(pointer: &str) -> Option<(usize, String)> {
    let mut segments = pointer.strip_prefix("#/")?.split('/');
    let group = segments.next()?;
    let at = DEFINITIONS.iter().position(|known| *known == group)?;
    let name = segments
        .next()
        .filter(|name| !name.is_empty() && !name.contains(['~', '%']))?;
    Some((at, name.to_string()))
}

/// A tool's groups of definitions, in [`DEFINITIONS`]' order: each definition of a group that
/// maps names to schemas is a part; any other group is written whole.
struct Definitions {
    groups: Vec<(usize, Group)>,
}

enum Group {
    Named(Vec<(String, Part)>),
    Whole(Part),
}

impl Definitions {
    fn of(root: &Value) -> Self {
        let groups = DEFINITIONS
            .iter()
            .enumerate()
            .filter_map(|(at, key)| {
                let group = match root.as_object()?.get(*key)? {
                    Value::Object(named) => Group::Named(
                        named
                            .iter()
                            .map(|(name, schema)| (name.clone(), Part::of(schema.clone())))
                            .collect(),
                    ),
                    whole => Group::Whole(Part::of(whole.clone())),
                };
                Some((at, group))
            })
            .collect();
        Self { groups }
    }

    /// The definitions `own` reaches, directly or through others, per group: the indices of
    /// the named ones in the group's order (all of them where a reference may reach any).
    fn reached(&self, own: &Part) -> Vec<Vec<usize>> {
        let mut seen: Vec<Vec<bool>> = self
            .groups
            .iter()
            .map(|(_, group)| match group {
                Group::Named(named) => vec![false; named.len()],
                Group::Whole(_) => Vec::new(),
            })
            .collect();
        let mut any = own.refs.any;
        let mut todo: Vec<&(usize, String)> = own.refs.to.iter().collect();
        while let Some((at, name)) = todo.pop() {
            if any {
                break;
            }
            for ((group_at, group), seen) in self.groups.iter().zip(&mut seen) {
                let Group::Named(named) = group else { continue };
                if group_at != at {
                    continue;
                }
                if let Some(index) = named.iter().position(|(known, _)| known == name) {
                    if !seen[index] {
                        seen[index] = true;
                        any |= named[index].1.refs.any;
                        todo.extend(named[index].1.refs.to.iter());
                    }
                }
            }
        }
        seen.into_iter()
            .map(|seen| {
                (0..seen.len())
                    .filter(|&index| any || seen[index])
                    .collect()
            })
            .collect()
    }
}

/// One JSON value's schema: its own part, and each group of definitions it carries (one its own
/// schema holds itself is not carried), whole or with the named definitions it reaches.
struct Document<'a> {
    own: &'a Part,
    carried: Vec<(&'static str, Carried<'a>)>,
}

enum Carried<'a> {
    Named(Vec<&'a (String, Part)>),
    Whole(&'a Part),
}

impl<'a> Document<'a> {
    fn of(own: &'a Part, definitions: &'a Definitions) -> Self {
        let held = |key: &str| {
            own.schema
                .as_object()
                .is_some_and(|map| map.contains_key(key))
        };
        let carried = definitions
            .groups
            .iter()
            .zip(definitions.reached(own))
            .filter(|((at, _), _)| !held(DEFINITIONS[*at]))
            .filter_map(|((at, group), reached)| {
                let carried = match group {
                    Group::Whole(part) => Carried::Whole(part),
                    Group::Named(_) if reached.is_empty() => return None,
                    Group::Named(named) => {
                        Carried::Named(reached.iter().map(|&index| &named[index]).collect())
                    }
                };
                Some((DEFINITIONS[*at], carried))
            })
            .collect();
        Self { own, carried }
    }

    fn schema(&self) -> Value {
        let mut schema = self.own.schema.clone();
        if let Some(map) = schema.as_object_mut() {
            for (key, carried) in &self.carried {
                let group = match carried {
                    Carried::Whole(part) => part.schema.clone(),
                    Carried::Named(named) => Value::Object(
                        named
                            .iter()
                            .map(|(name, part)| (name.clone(), part.schema.clone()))
                            .collect(),
                    ),
                };
                map.insert((*key).to_string(), group);
            }
        }
        schema
    }

    fn len(&self) -> usize {
        let members = self.own.schema.as_object().map_or(0, Map::len);
        let mut len = self.own.len;
        for (added, (key, carried)) in self.carried.iter().enumerate() {
            let group = match carried {
                Carried::Whole(part) => part.len,
                Carried::Named(named) => {
                    let total = named.iter().fold(0, |sum: usize, (name, part)| {
                        sum.saturating_add(json_len(name) + 1)
                            .saturating_add(part.len)
                    });
                    Length::listed(json_len(&Map::new()), named.len(), total)
                }
            };
            // A comma before every member but the object's first.
            len = len
                .saturating_add(usize::from(members + added > 0))
                .saturating_add(json_len(*key) + 1)
                .saturating_add(group);
        }
        len
    }
}

/// The grammar itself.
#[derive(Default)]
pub(super) struct Tree {
    parts: Parts,
}

impl Build for Tree {
    type Node = Grammar;

    fn leaf(&self, grammar: Grammar) -> Grammar {
        grammar
    }

    fn sequence(&self, elements: Vec<Grammar>) -> Grammar {
        Grammar::Sequence(elements)
    }

    fn or(&self, elements: Vec<Grammar>) -> Grammar {
        Grammar::Or(elements)
    }

    fn optional(&self, content: Grammar) -> Grammar {
        Grammar::Optional(Box::new(content))
    }

    fn star(&self, content: Grammar) -> Grammar {
        Grammar::Star(Box::new(content))
    }

    fn tag(&self, begin: String, content: Grammar, end: Bound) -> Grammar {
        Grammar::Tag(Tag::bounded(begin, content, end))
    }

    fn schema(&self, schema: &Value) -> Grammar {
        Grammar::JsonSchema {
            schema: schema.clone(),
            style: None,
            excludes: Vec::new(),
        }
    }

    fn value(&self, json: Json<'_>, root: Root<'_>, excludes: &[String]) -> Grammar {
        let own = self.parts.own(json);
        let definitions = self.parts.definitions(root);
        Grammar::JsonSchema {
            schema: Document::of(&own, &definitions).schema(),
            style: None,
            excludes: excludes.to_vec(),
        }
    }
}

/// The length of the text each node is written as. The fixed pieces are worked out once, each
/// the length of a node around placeholders less the placeholders'; a value's schema and each
/// tool's definitions are measured once, and every copy of them is a sum.
pub(super) struct Length {
    pieces: Pieces,
    parts: Parts,
}

#[derive(Clone, Copy, Debug)]
struct Pieces {
    sequence: usize,
    or: usize,
    optional: usize,
    star: usize,
    tag: usize,
    json_schema: usize,
    excludes: usize,
}

impl Length {
    pub(super) fn new() -> Self {
        let empty = Grammar::ConstString(String::new());
        let empty_len = json_len(&empty);
        let wrapped = |grammar: Grammar| json_len(&grammar) - empty_len;
        let json_schema = |excludes: Vec<String>| Grammar::JsonSchema {
            schema: Value::Null,
            style: None,
            excludes,
        };
        let word = vec![String::new()];
        let pieces = Pieces {
            sequence: json_len(&Grammar::Sequence(Vec::new())),
            or: json_len(&Grammar::Or(Vec::new())),
            optional: wrapped(Grammar::Optional(Box::new(empty.clone()))),
            star: wrapped(Grammar::Star(Box::new(empty.clone()))),
            tag: wrapped(Grammar::Tag(Tag::new("", empty, ""))) - 2 * json_len(""),
            json_schema: json_len(&json_schema(Vec::new())) - json_len(&Value::Null),
            excludes: json_len(&json_schema(word.clone()))
                - json_len(&json_schema(Vec::new()))
                - json_len(&word),
        };
        Self {
            pieces,
            parts: Parts::default(),
        }
    }

    /// `count` elements of `total` length in a list `empty` long without them.
    fn listed(empty: usize, count: usize, total: usize) -> usize {
        empty
            .saturating_add(total)
            .saturating_add(count.saturating_sub(1))
    }

    fn json_schema(&self, document: usize, excludes: &[String]) -> usize {
        let excludes = if excludes.is_empty() {
            0
        } else {
            self.pieces.excludes.saturating_add(json_len(excludes))
        };
        self.pieces
            .json_schema
            .saturating_add(document)
            .saturating_add(excludes)
    }
}

impl Build for Length {
    type Node = usize;

    fn leaf(&self, grammar: Grammar) -> usize {
        json_len(&grammar)
    }

    fn sequence(&self, elements: Vec<usize>) -> usize {
        let total = elements
            .iter()
            .fold(0, |sum: usize, &len| sum.saturating_add(len));
        Self::listed(self.pieces.sequence, elements.len(), total)
    }

    fn or(&self, elements: Vec<usize>) -> usize {
        let total = elements
            .iter()
            .fold(0, |sum: usize, &len| sum.saturating_add(len));
        Self::listed(self.pieces.or, elements.len(), total)
    }

    fn optional(&self, content: usize) -> usize {
        self.pieces.optional.saturating_add(content)
    }

    fn star(&self, content: usize) -> usize {
        self.pieces.star.saturating_add(content)
    }

    fn tag(&self, begin: String, content: usize, end: Bound) -> usize {
        self.pieces
            .tag
            .saturating_add(json_len(&begin))
            .saturating_add(content)
            .saturating_add(json_len(&end))
    }

    fn schema(&self, schema: &Value) -> usize {
        self.json_schema(json_len(schema), &[])
    }

    fn value(&self, json: Json<'_>, root: Root<'_>, excludes: &[String]) -> usize {
        let own = self.parts.own(json);
        let definitions = self.parts.definitions(root);
        self.json_schema(Document::of(&own, &definitions).len(), excludes)
    }
}
