//! What a request declares about its calls.
//!
//! A tagged format writes each argument as text that does not say its type, and so does the
//! pythonic dialect that quotes every value; the request's tools do, and one [`Declared`] per
//! request holds what they say: the kind each parameter declares (`Declared::kind`, and
//! `Declared::kind_at` for a value inside another), the parameter's schema itself, whether a tool
//! declares a function ([`Declared::declares`]), and the function the request's `tool_choice`
//! forces ([`Declared::forced`]). It is shared by every reader of the request's choices and copies
//! no schema.

use std::sync::Arc;

use openai_protocol::common::Tool;
use serde_json::Value;

use crate::tagged::value::{below, Kind};

/// What a request declares about its calls: the parameter types its tools declare, by function
/// name and then parameter name, and the function its `tool_choice` forces, when it names one.
/// The tools are read in place, shared with whoever else holds them ([`Declared::shared`]), and
/// read when a value asks: a kind is computed from the parameter's schema at the lookup, once per
/// argument a call writes, not once per parameter the request declares. So neither building one,
/// nor a reader for one more choice, nor the clone the engine takes for a shaped call copies or
/// indexes a schema. Equal when they hold equal tools (every field of every tool, in order) and
/// force the same function, so two that differ in a description, or in the order of their tools,
/// are not equal.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Declared {
    tools: Arc<[Tool]>,
    /// The function the request forces, whose arguments alone the output is under the gateway's
    /// JSON-schema constraint ([`formats::forced`](crate::formats::forced())).
    forced: Option<String>,
}

impl Declared {
    /// Over a copy of `tools`. When two tools share a name, the first one's parameters count.
    pub fn of(tools: &[Tool]) -> Self {
        Self::shared(Arc::from(tools))
    }

    /// The request's tools, as it declared them, in its order.
    pub(crate) fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// Whether a tool of the request declares `function`.
    pub fn declares(&self, function: &str) -> bool {
        self.tools.iter().any(|tool| tool.function.name == function)
    }

    /// [`Declared::of`] over tools the caller keeps: they are read in place, and nothing of them
    /// is copied or indexed.
    pub fn shared(tools: Arc<[Tool]>) -> Self {
        Self {
            tools,
            forced: None,
        }
    }

    /// With the function `name` the request forces (`tool_choice` naming it), for the format that
    /// reads its arguments alone.
    pub fn forcing(mut self, name: impl Into<String>) -> Self {
        self.forced = Some(name.into());
        self
    }

    /// The function the request forces, if it names one.
    pub fn forced(&self) -> Option<&str> {
        self.forced.as_deref()
    }

    /// `function`'s whole `parameters` schema, as the first tool of that name declares it.
    fn schema(&self, function: &str) -> Option<&Value> {
        self.tools
            .iter()
            .find(|tool| tool.function.name == function)
            .map(|tool| &tool.function.parameters)
    }

    /// The kind `function` declares for `parameter`, if it declares one.
    pub(crate) fn kind(&self, function: &str, parameter: &str) -> Option<Kind> {
        let root = self.schema(function)?;
        Kind::of(root, root.get("properties")?.get(parameter)?)
    }

    /// The schema `function` declares for `parameter`: its whole `parameters`, where a `$ref`
    /// points, and the parameter's own schema, if it declares one.
    pub fn parameter(&self, function: &str, parameter: &str) -> Option<(&Value, &Value)> {
        let root = self.schema(function)?;
        Some((root, root.get("properties")?.get(parameter)?))
    }

    /// The kind `function` declares at `path` below its parameters, for a value inside another:
    /// below an object a segment selects that property's schema, below an array any segment
    /// selects the items' schema (MiniMax writes a list's elements as `<item>` elements, and a
    /// property may be named `item` too, so the schema's shape decides, not the segment). Nothing
    /// when the path leads nowhere in the schema.
    pub(crate) fn kind_at(&self, function: &str, path: &[&str]) -> Option<Kind> {
        let root = self.schema(function)?;
        let mut schema = root;
        for segment in path {
            schema = below(root, schema, segment)?;
        }
        Kind::of(root, schema)
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;
    use serde_json::json as value;

    use super::*;
    use crate::tagged::value::json;

    fn tool(name: &str, properties: Value) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": properties}),
                strict: None,
                extra: Default::default(),
            },
        }
    }

    #[test]
    fn a_tools_declared_types_are_found_by_function_and_parameter() {
        let declared = Declared::of(&[
            tool(
                "search",
                value!({"query": {"type": "string"}, "limit": {"type": "integer"}}),
            ),
            tool("count", value!({"query": {"type": "integer"}})),
        ]);
        assert_eq!(declared.kind("search", "query"), Some(Kind::String));
        assert_eq!(declared.kind("search", "limit"), Some(Kind::Integer));
        assert_eq!(declared.kind("count", "query"), Some(Kind::Integer));
        assert_eq!(declared.kind("count", "limit"), None);
        assert_eq!(declared.kind("missing", "query"), None);
    }

    #[test]
    fn shared_tools_are_read_in_place_and_equal_a_copy_of_them() {
        let tools: Arc<[Tool]> = Arc::from(vec![tool(
            "search",
            value!({"query": {"type": "string"}, "limit": {"type": "integer"}}),
        )]);
        let declared = Declared::shared(Arc::clone(&tools));
        assert_eq!(
            Arc::strong_count(&tools),
            2,
            "the tools are shared, not copied"
        );
        let clone = declared.clone();
        assert_eq!(Arc::strong_count(&tools), 3, "a clone shares them too");
        assert_eq!(clone.kind("search", "limit"), Some(Kind::Integer));
        assert_eq!(clone.kind_at("search", &["query"]), Some(Kind::String));
        assert_eq!(declared, Declared::of(&tools));
        assert_ne!(
            declared,
            Declared::of(&[tool("search", value!({"query": {"type": "integer"}}))])
        );
        assert_ne!(declared, declared.clone().forcing("search"));
    }

    #[test]
    fn a_schema_that_admits_a_string_declares_one_nullable_when_it_admits_null_too() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                "plain": {"type": "string"},
                "listed": {"type": ["string", "null"]},
                // What Pydantic writes for `Optional[str]`.
                "optional": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "either": {"oneOf": [{"type": "null"}, {"type": "string"}]},
                "choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "named": {"enum": ["celsius", "fahrenheit"]},
                "named_or_null": {"enum": ["celsius", "fahrenheit"], "type": ["string", "null"]},
                "named_or_none": {"enum": ["celsius", "fahrenheit", null]},
                "nested": {"anyOf": [{"anyOf": [{"type": "string"}]}, {"type": "null"}]},
            }),
        )]);
        for (parameter, kind) in [
            ("plain", Kind::String),
            ("listed", Kind::NullableString),
            ("optional", Kind::NullableString),
            ("either", Kind::NullableString),
            ("choice", Kind::String),
            ("named", Kind::String),
            ("named_or_null", Kind::NullableString),
            ("named_or_none", Kind::NullableString),
            ("nested", Kind::NullableString),
        ] {
            assert_eq!(declared.kind("f", parameter), Some(kind), "{parameter}");
        }
    }

    #[test]
    fn a_type_decides_alone_and_an_enum_beside_it_adds_nothing() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                // BFCL's shape for a list whose members come from the enum.
                "members": {"type": "array", "items": {"type": "string"}, "enum": ["a", "b"]},
                "counted": {"type": "integer", "enum": ["one", "two"]},
                "named": {"type": "string", "enum": [1, 2]},
                "listed": {"type": ["string", "null"], "enum": ["a", "b"]},
            }),
        )]);
        assert_eq!(declared.kind("f", "members"), Some(Kind::Array));
        assert_eq!(declared.kind("f", "counted"), Some(Kind::Integer));
        assert_eq!(declared.kind("f", "named"), Some(Kind::String));
        assert_eq!(declared.kind("f", "listed"), Some(Kind::NullableString));
    }

    #[test]
    fn an_integer_an_array_or_an_object_alone_is_declared_and_every_other_schema_nothing() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "on": {"type": "boolean"},
                "points": {"type": "array"},
                "style": {"type": "object"},
                "nothing": {"type": "null"},
                "optional_count": {"type": ["integer", "null"]},
                "untyped": {"description": "anything"},
                "mixed": {"enum": [1, "two"]},
                "empty": {"enum": []},
                "only_null": {"enum": [null]},
                // Aliases and other spellings of a type name are not read today; the engine's parser reads
                // these as string and integer. Whether Symphony follows is the maintainer's
                // decision.
                "alias": {"type": "str"},
                "cased": {"type": "String"},
                "padded": {"type": " integer "},
                "width": {"type": "int32"},
            }),
        )]);
        assert_eq!(declared.kind("f", "count"), Some(Kind::Integer));
        assert_eq!(declared.kind("f", "points"), Some(Kind::Array));
        assert_eq!(declared.kind("f", "style"), Some(Kind::Object));
        for parameter in [
            "ratio",
            "on",
            "nothing",
            "optional_count",
            "untyped",
            "mixed",
            "empty",
            "only_null",
            "alias",
            "cased",
            "padded",
            "width",
        ] {
            assert_eq!(declared.kind("f", parameter), None, "{parameter}");
        }
    }

    #[test]
    fn a_kind_below_the_top_level_follows_properties_and_items() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                "recipients": {"type": "array", "items": {"type": "string"}},
                "points": {"type": "array", "items": {"type": "object", "properties": {
                    "x": {"type": "integer"}, "tags": {"type": "array"}}}},
                "options": {"type": "object", "properties": {"id": {"type": "string"}}},
                "loose": {"type": "object"},
                "cart": {"type": "object", "properties": {"item": {"type": "string"}}},
            }),
        )]);
        assert_eq!(declared.kind_at("f", &["recipients"]), Some(Kind::Array));
        assert_eq!(
            declared.kind_at("f", &["recipients", "item"]),
            Some(Kind::String)
        );
        assert_eq!(
            declared.kind_at("f", &["points", "item", "x"]),
            Some(Kind::Integer)
        );
        assert_eq!(
            declared.kind_at("f", &["points", "item", "tags"]),
            Some(Kind::Array)
        );
        assert_eq!(
            declared.kind_at("f", &["options", "id"]),
            Some(Kind::String)
        );
        assert_eq!(declared.kind_at("f", &["loose", "anything"]), None);
        // Below an array any segment is an item; below an object `item` is a property like any.
        assert_eq!(
            declared.kind_at("f", &["recipients", "x"]),
            Some(Kind::String)
        );
        assert_eq!(declared.kind_at("f", &["cart", "item"]), Some(Kind::String));
        assert_eq!(declared.kind_at("f", &["cart", "item", "x"]), None);
        assert_eq!(declared.kind_at("g", &["recipients"]), None);
        assert_eq!(declared.kind_at("f", &[]), Some(Kind::Object));
    }

    #[test]
    fn a_tool_without_properties_declares_nothing_and_the_first_of_two_names_counts() {
        let mut bare = tool("bare", value!({}));
        bare.function.parameters = value!({"type": "object"});
        let declared = Declared::of(&[
            bare,
            tool("twice", value!({"x": {"type": "integer"}})),
            tool("twice", value!({"x": {"type": "string"}})),
        ]);
        assert_eq!(declared.kind("bare", "x"), None);
        assert_eq!(declared.kind("twice", "x"), Some(Kind::Integer));
    }

    #[test]
    fn a_kind_below_a_wrapped_array_or_object_comes_from_the_member_that_has_the_shape() {
        // Pydantic's `Optional[List[str]]`, an object inline under `anyOf` (Pydantic refers to a
        // model through `$defs`, which the `$ref` test covers), and a wrapper inside a wrapper.
        let declared = Declared::of(&[tool(
            "f",
            value!({
                "tags": {"anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"}
                ]},
                "meta": {"anyOf": [
                    {"type": "object", "properties": {"id": {"type": "integer"}}},
                    {"type": "null"}
                ]},
                "deep": {"anyOf": [
                    {"anyOf": [{"type": "array", "items": {"type": "integer"}}]},
                    {"type": "null"}
                ]},
                "either": {"oneOf": [
                    {"type": "object", "properties": {"n": {"type": "string"}}},
                    {"type": "null"}
                ]},
            }),
        )]);
        assert_eq!(declared.kind_at("f", &["tags", "item"]), Some(Kind::String));
        assert_eq!(declared.kind_at("f", &["meta", "id"]), Some(Kind::Integer));
        assert_eq!(
            declared.kind_at("f", &["deep", "item"]),
            Some(Kind::Integer)
        );
        assert_eq!(declared.kind_at("f", &["either", "n"]), Some(Kind::String));
        assert_eq!(declared.kind_at("f", &["meta", "missing"]), None);
    }

    #[test]
    fn a_union_of_an_array_and_an_object_leaves_what_is_below_it_to_inference() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                "x": {"anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "object", "properties": {"id": {"type": "integer"}}}
                ]},
            }),
        )]);
        assert_eq!(declared.kind_at("f", &["x", "id"]), None);
        assert_eq!(declared.kind_at("f", &["x", "item"]), None);
    }

    #[test]
    fn a_kind_through_a_ref_into_the_definitions_is_the_targets() {
        // Pydantic writes `Optional[Model]` as a `$ref` under `anyOf`, a `Model` field as a
        // bare `$ref`, an enum as a `$ref` to an `enum` definition, and a recursive model as a
        // `$ref` to itself.
        let parameters = value!({
            "type": "object",
            "properties": {
                "meta": {"anyOf": [{"$ref": "#/$defs/Meta"}, {"type": "null"}]},
                "owner": {"$ref": "#/$defs/Meta"},
                "color": {"$ref": "#/$defs/Color"},
                "tree": {"$ref": "#/$defs/Node"},
                "old": {"$ref": "#/definitions/Old"},
                "nowhere": {"$ref": "#/$defs/Missing"},
            },
            "$defs": {
                "Meta": {"type": "object", "properties": {
                    "id": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}}}},
                "Color": {"enum": ["red", "blue"]},
                "Node": {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}}},
                "Loop": {"$ref": "#/$defs/Loop"},
            },
            "definitions": {"Old": {"type": "integer"}},
        });
        let declared = Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters,
                strict: None,
                extra: Default::default(),
            },
        }]);
        assert_eq!(declared.kind("f", "owner"), Some(Kind::Object));
        assert_eq!(declared.kind("f", "color"), Some(Kind::String));
        assert_eq!(declared.kind("f", "old"), Some(Kind::Integer));
        assert_eq!(declared.kind("f", "nowhere"), None);
        assert_eq!(declared.kind_at("f", &["meta", "id"]), Some(Kind::Integer));
        assert_eq!(
            declared.kind_at("f", &["meta", "tags", "item"]),
            Some(Kind::String)
        );
        assert_eq!(declared.kind_at("f", &["owner", "id"]), Some(Kind::Integer));
        assert_eq!(
            declared.kind_at(
                "f",
                &["tree", "children", "item", "children", "item", "name"]
            ),
            Some(Kind::String)
        );
        assert_eq!(declared.kind_at("f", &["nowhere", "x"]), None);
        // A definition that refers to itself ends at the hop limit and admits nothing.
        let looped = value!({"$ref": "#/$defs/Loop"});
        let (root, _) = declared
            .parameter("f", "meta")
            .expect("a declared parameter");
        assert_eq!(Kind::of(root, &looped), None);
    }

    #[test]
    fn a_definition_that_refers_to_itself_many_times_over_is_read_once_per_lookup() {
        // Four references to itself at every level: read once, this is a handful of steps; read
        // again at every reference, the walk would branch four ways sixteen times.
        let parameters = value!({
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/A"}},
            "$defs": {"A": {"anyOf": [
                {"$ref": "#/$defs/A"}, {"$ref": "#/$defs/A"},
                {"$ref": "#/$defs/A"}, {"$ref": "#/$defs/A"},
                {"type": "string"}
            ]}},
        });
        let started = std::time::Instant::now();
        let declared = Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters,
                strict: None,
                extra: Default::default(),
            },
        }]);
        assert_eq!(declared.kind("f", "a"), Some(Kind::String));
        assert_eq!(declared.kind_at("f", &["a", "x"]), None);
        assert!(
            started.elapsed().as_secs() < 2,
            "{:?} for a self-referring definition",
            started.elapsed()
        );
    }

    #[test]
    fn a_one_member_all_of_around_a_reference_is_the_reference() {
        // Pydantic v1 writes a field with a description so, under `definitions`.
        let parameters = value!({
            "type": "object",
            "properties": {
                "meta": {"allOf": [{"$ref": "#/definitions/Meta"}], "description": "the meta"},
                "both": {"allOf": [{"type": "string"}, {"minLength": 1}]},
            },
            "definitions": {"Meta": {"type": "object", "properties": {"id": {"type": "integer"}}}},
        });
        let declared = Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters,
                strict: None,
                extra: Default::default(),
            },
        }]);
        assert_eq!(declared.kind("f", "meta"), Some(Kind::Object));
        assert_eq!(declared.kind_at("f", &["meta", "id"]), Some(Kind::Integer));
        // An `allOf` with more than one member admits what its typed members all admit.
        assert_eq!(declared.kind("f", "both"), Some(Kind::String));
    }

    #[test]
    fn an_all_of_admits_what_every_member_that_names_a_type_admits() {
        let declared = Declared::of(&[tool_with(value!({
            "type": "object",
            "properties": {
                "bounded": {"allOf": [{"type": "string"}, {"maxLength": 8}]},
                "referred": {"allOf": [{"$ref": "#/$defs/Name"}, {"description": "a name"}]},
                "narrowed": {"allOf": [{"type": ["string", "null"]}, {"type": "string"}]},
                "optional": {"allOf": [{"type": ["string", "null"]}, {"minLength": 1}]},
                "counted": {"allOf": [{"type": "integer"}, {"minimum": 0}]},
                "merged": {"allOf": [{"$ref": "#/$defs/Base"}, {"properties": {"x": {}}}]},
                "contradicted": {"allOf": [{"type": "string"}, {"type": "integer"}]},
                "untyped": {"allOf": [{"minLength": 1}, {"description": "anything"}]},
            },
            "$defs": {
                "Name": {"type": "string"},
                "Base": {"type": "object", "properties": {"id": {"type": "integer"}}},
            },
        }))]);
        for (parameter, kind) in [
            ("bounded", Some(Kind::String)),
            ("referred", Some(Kind::String)),
            ("narrowed", Some(Kind::String)),
            ("optional", Some(Kind::NullableString)),
            ("counted", Some(Kind::Integer)),
            ("merged", Some(Kind::Object)),
            ("contradicted", None),
            ("untyped", None),
        ] {
            assert_eq!(declared.kind("f", parameter), kind, "{parameter}");
        }
        // A digit for a string bounded in a second member stays the string.
        assert_eq!(json("3", declared.kind("f", "bounded")), r#""3""#);
    }

    #[test]
    fn a_number_and_an_integer_in_an_all_of_admit_the_integers() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                "counted": {"allOf": [{"type": "integer"}, {"type": "number"}]},
                "reversed": {"allOf": [{"type": "number"}, {"type": "integer"}]},
                "optional": {"allOf": [{"type": ["integer", "null"]}, {"type": "number"}]},
                "beside": {"type": "number", "allOf": [{"type": "integer"}, {"minimum": 0}]},
                "measured": {"allOf": [{"type": "number"}, {"minimum": 0}]},
                "either": {"allOf": [{"type": ["integer", "number"]}, {"type": "number"}]},
            }),
        )]);
        for parameter in ["counted", "reversed", "optional", "beside"] {
            assert_eq!(
                declared.kind("f", parameter),
                Some(Kind::Integer),
                "{parameter}"
            );
        }
        assert_eq!(json("007", declared.kind("f", "counted")), "7");
        // A number that need not be whole is inferred, as a `number` alone is.
        assert_eq!(declared.kind("f", "measured"), None);
        assert_eq!(declared.kind("f", "either"), None);
    }

    #[test]
    fn an_all_of_narrows_the_types_named_beside_it() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                // A string or null whose `allOf` admits a string only: a string, never null.
                "bounded": {
                    "type": ["string", "null"],
                    "allOf": [{"type": "string"}, {"maxLength": 8}]
                },
                "optional": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "allOf": [{"type": "string"}, {"maxLength": 8}]
                },
                "named": {
                    "enum": ["a", "b", null],
                    "allOf": [{"type": "string"}, {"maxLength": 1}]
                },
                // Members that name no type narrow nothing.
                "loose": {
                    "type": ["string", "null"],
                    "allOf": [{"maxLength": 8}, {"description": "a note"}]
                },
                // Types that no member admits leave nothing to read the value as.
                "contradicted": {
                    "type": "integer",
                    "allOf": [{"type": "string"}, {"maxLength": 8}]
                },
                // OpenAPI's nullable admits null beside what the rest admits, `allOf` and all.
                "nullable": {
                    "type": "string",
                    "nullable": true,
                    "allOf": [{"type": "string"}, {"maxLength": 8}]
                },
                // Below the top level too: an array, so its items are read.
                "listed": {
                    "type": ["array", "object"],
                    "allOf": [{"type": "array"}, {"minItems": 1}],
                    "items": {"type": "integer"}
                },
            }),
        )]);
        for (parameter, kind) in [
            ("bounded", Some(Kind::String)),
            ("optional", Some(Kind::String)),
            ("named", Some(Kind::String)),
            ("loose", Some(Kind::NullableString)),
            ("contradicted", None),
            ("nullable", Some(Kind::NullableString)),
            ("listed", Some(Kind::Array)),
        ] {
            assert_eq!(declared.kind("f", parameter), kind, "{parameter}");
        }
        assert_eq!(json("None", declared.kind("f", "bounded")), r#""None""#);
        assert_eq!(json("null", declared.kind("f", "bounded")), r#""null""#);
        assert_eq!(json("None", declared.kind("f", "nullable")), "null");
        assert_eq!(
            declared.kind_at("f", &["listed", "item"]),
            Some(Kind::Integer)
        );
    }

    #[test]
    fn each_member_of_an_all_of_brings_all_it_admits_whatever_another_member_read() {
        let declared = Declared::of(&[tool_with(value!({
            "type": "object",
            "properties": {
                // A string or an integer, and a string or null: a string, through references as
                // when the same schema is written inline.
                "referred": {"allOf": [
                    {"anyOf": [{"$ref": "#/$defs/S"}, {"$ref": "#/$defs/I"}]},
                    {"anyOf": [{"$ref": "#/$defs/S"}, {"type": "null"}]}
                ]},
                "inline": {"allOf": [
                    {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                    {"anyOf": [{"type": "string"}, {"type": "null"}]}
                ]},
                // A string or null, and a string: a string, whichever member reads it first.
                "narrowed": {"allOf": [
                    {"anyOf": [{"$ref": "#/$defs/S"}, {"type": "null"}]},
                    {"$ref": "#/$defs/S"}
                ]},
                // What a node admits inside an `allOf` reaches the union around it too.
                "beside": {"anyOf": [
                    {"allOf": [{"$ref": "#/$defs/S"}, {"type": "integer"}]},
                    {"$ref": "#/$defs/S"}
                ]},
                // A definition met again while it is read admits nothing there.
                "looped": {"$ref": "#/$defs/L"},
            },
            "$defs": {
                "S": {"type": "string"},
                "I": {"type": "integer"},
                "L": {"allOf": [{"$ref": "#/$defs/L"}, {"type": "string"}]},
            },
        }))]);
        for parameter in ["referred", "inline", "narrowed", "beside", "looped"] {
            assert_eq!(
                declared.kind("f", parameter),
                Some(Kind::String),
                "{parameter}"
            );
        }
        assert_eq!(json("3", declared.kind("f", "referred")), r#""3""#);
        assert_eq!(json("null", declared.kind("f", "narrowed")), r#""null""#);
    }

    #[test]
    fn a_definition_every_member_of_an_all_of_refers_to_is_read_once_per_lookup() {
        // Twenty-four levels, each an `allOf` of two references to the next: read once per node,
        // this is a few dozen steps; read again for every member, it would be two to the power of
        // twenty-four.
        let depth = 24;
        let mut definitions = serde_json::Map::new();
        for level in 0..depth {
            let next = format!("#/$defs/D{}", level + 1);
            definitions.insert(
                format!("D{level}"),
                value!({"allOf": [{"$ref": next}, {"$ref": next}]}),
            );
        }
        definitions.insert(format!("D{depth}"), value!({"type": "string"}));
        let started = std::time::Instant::now();
        let declared = Declared::of(&[tool_with(value!({
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/D0"}},
            "$defs": definitions,
        }))]);
        assert_eq!(declared.kind("f", "x"), Some(Kind::String));
        assert!(
            started.elapsed().as_secs() < 2,
            "{:?} for an `allOf` of shared references",
            started.elapsed()
        );
    }

    #[test]
    fn a_const_is_an_enum_of_one_and_a_type_beside_it_decides() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                "word": {"const": "7"},
                "number": {"const": 7},
                "nothing": {"const": null},
                "typed": {"const": "7", "type": "integer"},
                "listed": {"enum": ["a", "b"], "const": 1},
            }),
        )]);
        assert_eq!(declared.kind("f", "word"), Some(Kind::String));
        assert_eq!(json("7", declared.kind("f", "word")), r#""7""#);
        assert_eq!(declared.kind("f", "number"), None);
        assert_eq!(json("7", declared.kind("f", "number")), "7");
        assert_eq!(declared.kind("f", "nothing"), None);
        assert_eq!(declared.kind("f", "typed"), Some(Kind::Integer));
        // The enum, when there is one, is what the value is read from.
        assert_eq!(declared.kind("f", "listed"), Some(Kind::String));
    }

    #[test]
    fn openapis_nullable_admits_null_beside_the_schemas_types() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                "note": {"type": "string", "nullable": true},
                "named": {"enum": ["a", "b"], "nullable": true},
                "count": {"type": "integer", "nullable": true},
                "style": {"type": "object", "nullable": true},
                "points": {"type": "array", "nullable": true},
                "plain": {"type": "string", "nullable": false},
            }),
        )]);
        assert_eq!(declared.kind("f", "note"), Some(Kind::NullableString));
        assert_eq!(json("null", declared.kind("f", "note")), "null");
        assert_eq!(json("None", declared.kind("f", "note")), "null");
        assert_eq!(json("Paris", declared.kind("f", "note")), r#""Paris""#);
        assert_eq!(declared.kind("f", "named"), Some(Kind::NullableString));
        // An integer that may be null admits two types, as `["integer", "null"]` does: inferred.
        assert_eq!(declared.kind("f", "count"), None);
        assert_eq!(json("5", declared.kind("f", "count")), "5");
        assert_eq!(json("null", declared.kind("f", "count")), "null");
        // So is an object or an array that may be null: an XML element with no child is not `{}`
        // or `[]` under it.
        assert_eq!(declared.kind("f", "style"), None);
        assert_eq!(declared.kind("f", "points"), None);
        assert_eq!(declared.kind("f", "plain"), Some(Kind::String));
        assert_eq!(json("null", declared.kind("f", "plain")), r#""null""#);
    }

    #[test]
    fn a_wide_union_costs_its_size() {
        // Twenty thousand members: read once each, a lookup is a few milliseconds; searched
        // again for every member, it would be the square of that.
        let member = value!({"type": "object", "properties": {"id": {"type": "integer"}}});
        let members: Vec<Value> = std::iter::repeat_n(member, 20_000).collect();
        let parameters = value!({
            "type": "object",
            "properties": {"x": {"anyOf": members}},
        });
        let started = std::time::Instant::now();
        let declared = Declared::of(&[tool_with(parameters)]);
        assert_eq!(declared.kind("f", "x"), Some(Kind::Object));
        assert_eq!(declared.kind_at("f", &["x", "id"]), Some(Kind::Integer));
        assert!(
            started.elapsed().as_secs() < 2,
            "{:?} for a wide union",
            started.elapsed()
        );
    }

    #[test]
    fn a_type_named_twice_is_one_type_and_an_unresolved_reference_is_nothing() {
        let declared = Declared::of(&[tool_with(value!({
            "type": "object",
            "properties": {
                "twice": {"type": "integer", "allOf": [{"$ref": "#/$defs/Int"}]},
                "nowhere": {"$ref": "#/$defs/Missing", "type": "string"},
                "mixed": {"anyOf": [
                    {"type": "array"},
                    {"type": "integer", "items": {"type": "string"}}
                ]},
            },
            "$defs": {"Int": {"type": "integer"}},
        }))]);
        assert_eq!(declared.kind("f", "twice"), Some(Kind::Integer));
        // A reference that leads nowhere admits nothing, whatever stands beside it.
        assert_eq!(declared.kind("f", "nowhere"), None);
        assert_eq!(declared.kind_at("f", &["nowhere", "x"]), None);
        // The items of a member typed as something else are not the array's.
        assert_eq!(declared.kind_at("f", &["mixed", "item"]), None);
    }

    /// The function `f` with `parameters`.
    fn tool_with(parameters: Value) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters,
                strict: None,
                extra: Default::default(),
            },
        }
    }
}
