use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

use serde::{
    de::{
        self,
        value::{MapDeserializer, SeqAccessDeserializer},
        SeqAccess, Visitor,
    },
    Deserialize, Deserializer, Serialize,
};
use serde_json::{Map, Value};
use validator;

// ============================================================================
// Default value helpers
// ============================================================================

/// A typed request as a `Value` whose `f32` fields print as the shortest
/// decimal that round-trips the parsed value. `serde_json::to_value` stores
/// an `f32` widened to `f64`, so a client's `"top_p": 0.95` would go
/// upstream as `0.949999988079071`; the writer prints the `f32` itself, so
/// encode with it and parse that back. This restores what the parsed `f32`
/// holds, not the client's original token: precision the parse discarded
/// (`0.950000001` is the same `f32` as `0.95`) is gone, and numbers kept
/// as `Value` (an `other` map, nested values) were normalized at parse time
/// (`1e2` is `100.0`, `-0` is `-0.0`).
pub fn to_value_exact<T: Serialize>(value: &T) -> serde_json::Result<Value> {
    serde_json::from_slice(&serde_json::to_vec(value)?)
}

/// Default model for endpoints where model is optional (e.g., /generate).
/// Uses UNKNOWN_MODEL_ID so routers treat it as "any available worker."
pub fn default_unknown_model() -> String {
    super::UNKNOWN_MODEL_ID.to_string()
}

/// Serde skip predicate: the wildcard model id is a gateway-internal
/// placeholder for "the request named no model", never a value to forward.
/// A model-less SGLang-native `/generate` (slime, verl) must reach the
/// engine without a `model` key, exactly as the client sent it; a
/// model-aware upstream (a sidecar in front of another gateway) would
/// otherwise look up a model called "unknown" and answer 404.
pub fn is_unknown_model(model: &str) -> bool {
    model == super::UNKNOWN_MODEL_ID
}

/// Helper function for serde default value (returns true)
pub fn default_true() -> bool {
    true
}

/// Helper for `#[serde(skip_serializing_if = "is_false")]` on default-`false` flags.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes &T"
)]
pub fn is_false(v: &bool) -> bool {
    !*v
}

/// Helper for `#[serde(skip_serializing_if = "is_true")]` on default-`true` flags.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes &T"
)]
pub fn is_true(v: &bool) -> bool {
    *v
}

/// Deserialize a bool that also accepts JSON `null` (mapped to `false`).
///
/// Use with `#[serde(default, deserialize_with = "deserialize_null_as_false")]`
/// on fields that the OpenAI spec defines as `Optional[bool]` defaulting to `false`.
pub fn deserialize_null_as_false<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<bool>::deserialize(deserializer).map(|opt| opt.unwrap_or(false))
}

/// Deserialize a list that also accepts JSON `null` (mapped to the empty list).
///
/// Use with `#[serde(default, deserialize_with = "deserialize_null_as_empty_vec")]`
/// on always-present lists an upstream may write as `null`.
pub fn deserialize_null_as_empty_vec<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<Vec<T>>::deserialize(deserializer).map(Option::unwrap_or_default)
}

/// Deserialize an `Option<bool>` where JSON `null` means the default `true`.
///
/// Use with `#[serde(default = "default_true", deserialize_with = "deserialize_null_as_true")]`
/// on extension fields that default to `true`, so a client that serializes an
/// unset option as `null` gets the default instead of a rejected request.
pub fn deserialize_null_as_true<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<bool>::deserialize(deserializer).map(|opt| opt.unwrap_or(true))
}

// ============================================================================
// GenerationRequest Trait
// ============================================================================

/// The request fields that partition a backend's prefix cache.
///
/// Engines namespace their prefix (KV) caches by client-supplied values: a
/// cache salt, an extra classification key, and the LoRA adapter. Two
/// requests that share a prompt but differ in any of these can never share
/// cached blocks on the engine, so routing derives its cache namespace from
/// this projection. All fields absent means the request is unpartitioned.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CachePartition<'a> {
    pub cache_salt: Option<&'a str>,
    pub extra_key: Option<&'a str>,
    pub lora_path: Option<&'a str>,
}

impl CachePartition<'_> {
    /// True when no partitioning field is set. An empty string is "unset":
    /// engines treat an empty salt or adapter as absent, so it must not
    /// split the request off from the shared unpartitioned cache.
    pub fn is_empty(&self) -> bool {
        [self.cache_salt, self.extra_key, self.lora_path]
            .iter()
            .all(|field| field.is_none_or(str::is_empty))
    }
}

/// Trait for unified access to generation request properties
/// Implemented by ChatCompletionRequest, CompletionRequest, GenerateRequest,
/// EmbeddingRequest, RerankRequest, and ResponsesRequest
pub trait GenerationRequest: Send + Sync {
    /// Check if the request is for streaming
    fn is_stream(&self) -> bool;

    /// Get the model name if specified
    fn get_model(&self) -> Option<&str>;

    /// Extract text content for routing decisions
    fn extract_text_for_routing(&self) -> String;

    /// Token IDs for routing when the request is already tokenized.
    /// Some(_) routes on the token radix tree instead of the decimal-string
    /// rendering of the same IDs.
    fn routing_tokens(&self) -> Option<&[i32]> {
        None
    }

    /// Client-provided request id, when the protocol carries one. Routing may
    /// derive a session-affinity key from it; a batch reports its first id.
    fn rid(&self) -> Option<&str> {
        None
    }

    /// The fields that partition the backend's prefix cache for this request.
    /// Protocols without such fields are unpartitioned.
    fn cache_partition(&self) -> CachePartition<'_> {
        CachePartition::default()
    }
}

// ============================================================================
// String/Array Utilities
// ============================================================================

/// A type that can be either a single string or an array of strings
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum StringOrArray {
    String(String),
    Array(Vec<String>),
}

impl StringOrArray {
    /// Get the number of items in the StringOrArray
    pub fn len(&self) -> usize {
        match self {
            StringOrArray::String(_) => 1,
            StringOrArray::Array(arr) => arr.len(),
        }
    }

    /// Check if the StringOrArray is empty
    pub fn is_empty(&self) -> bool {
        match self {
            StringOrArray::String(s) => s.is_empty(),
            StringOrArray::Array(arr) => arr.is_empty(),
        }
    }

    /// Convert to a vector of strings (clones the data)
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            StringOrArray::String(s) => vec![s.clone()],
            StringOrArray::Array(arr) => arr.clone(),
        }
    }

    /// Returns an iterator over string references without cloning.
    /// Use this instead of `to_vec()` when you only need to iterate.
    pub fn iter(&self) -> StringOrArrayIter<'_> {
        StringOrArrayIter {
            inner: self,
            index: 0,
        }
    }

    /// Returns the first string, or None if empty
    pub fn first(&self) -> Option<&str> {
        match self {
            StringOrArray::String(s) => {
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            }
            StringOrArray::Array(arr) => arr.first().map(|s| s.as_str()),
        }
    }
}

/// Iterator over StringOrArray that yields string references without cloning
pub struct StringOrArrayIter<'a> {
    inner: &'a StringOrArray,
    index: usize,
}

impl<'a> Iterator for StringOrArrayIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner {
            StringOrArray::String(s) => {
                if self.index == 0 {
                    self.index = 1;
                    Some(s.as_str())
                } else {
                    None
                }
            }
            StringOrArray::Array(arr) => {
                if self.index < arr.len() {
                    let item = &arr[self.index];
                    self.index += 1;
                    Some(item.as_str())
                } else {
                    None
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = match self.inner {
            StringOrArray::String(_) => 1 - self.index,
            StringOrArray::Array(arr) => arr.len() - self.index,
        };
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for StringOrArrayIter<'a> {}

/// A validation error with the public API's `code` and message.
pub(crate) fn invalid(
    code: &'static str,
    message: impl Into<Cow<'static, str>>,
) -> validator::ValidationError {
    let mut e = validator::ValidationError::new(code);
    e.message = Some(message.into());
    e
}

/// Validates stop sequences (max 4, non-empty strings)
/// Used by both ChatCompletionRequest and ResponsesRequest
pub fn validate_stop(stop: &StringOrArray) -> Result<(), validator::ValidationError> {
    match stop {
        StringOrArray::String(s) => {
            if s.is_empty() {
                return Err(validator::ValidationError::new(
                    "stop sequences cannot be empty",
                ));
            }
        }
        StringOrArray::Array(arr) => {
            if arr.len() > 4 {
                return Err(invalid(
                    "array_above_max_length",
                    format!(
                        "Invalid 'stop': array too long. Expected an array with maximum length 4, but got an array with length {} instead.",
                        arr.len()
                    ),
                ));
            }
            for s in arr {
                if s.is_empty() {
                    return Err(validator::ValidationError::new(
                        "stop sequences cannot be empty",
                    ));
                }
            }
        }
    }
    Ok(())
}

// ============================================================================
// Content Parts (for multimodal messages)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    #[serde(rename = "audio_url")]
    AudioUrl { audio_url: AudioUrl },
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: InputAudio },
    #[serde(rename = "video_url")]
    VideoUrl { video_url: VideoUrl },
    /// Pass through unknown content objects to HTTP backends.
    /// Also matches malformed known types; gRPC backends reject this variant.
    #[serde(untagged)]
    Unknown(Map<String, Value>),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>, // "auto", "low", or "high"
    /// MiniMax-M3 extension: cap the image's long side before preprocessing.
    ///
    /// For M3 router preprocessing, any positive cap is accepted (default 2016).
    /// Branch on the source size: shrink if its long side exceeds the cap;
    /// otherwise raise a short side below 112. Then round both sides to the
    /// nearest multiple of 28, so the aligned long side can exceed the cap.
    /// Outputs above 12,845,056 pixels are rejected; the legacy `max_pixels`
    /// area shrink is not applied a second time. Other families retain their
    /// connector's positive-multiple-of-28 restriction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_long_side_pixel: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
pub struct AudioUrl {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
pub struct InputAudio {
    /// Base64-encoded audio bytes.
    pub data: String,
    /// Encoded audio format. The OpenAI Chat API supports `wav` and `mp3`.
    pub format: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, schemars::JsonSchema)]
pub struct VideoUrl {
    pub url: String,
    /// MiniMax-M3 extension: frames per second to sample the clip at.
    ///
    /// Valid range is 0.2 to 5.0 inclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
    /// MiniMax-M3 extension: cap each sampled frame's long side.
    ///
    /// Must be a multiple of the vision patch factor (28) within 150..=3584.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_long_side_pixel: Option<u32>,
}

// ============================================================================
// Response Format (for structured outputs)
// ============================================================================

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchemaFormat },
}

/// The kind a `response_format` object names under `type`, read with the
/// words serde uses for a tag: an unknown name lists the three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseFormatKind {
    Text,
    JsonObject,
    JsonSchema,
}

impl ResponseFormatKind {
    const ALL: [ResponseFormatKind; 3] = [
        ResponseFormatKind::Text,
        ResponseFormatKind::JsonObject,
        ResponseFormatKind::JsonSchema,
    ];

    /// The name under `type`: the variant's rename on [`ResponseFormat`].
    const fn name(self) -> &'static str {
        match self {
            ResponseFormatKind::Text => "text",
            ResponseFormatKind::JsonObject => "json_object",
            ResponseFormatKind::JsonSchema => "json_schema",
        }
    }
}

/// The names an unknown kind's error lists, in the variants' order.
const RESPONSE_FORMAT_KINDS: &[&str] = &[
    ResponseFormatKind::Text.name(),
    ResponseFormatKind::JsonObject.name(),
    ResponseFormatKind::JsonSchema.name(),
];

impl<'de> Deserialize<'de> for ResponseFormatKind {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KindVisitor;

        impl Visitor<'_> for KindVisitor {
            type Value = ResponseFormatKind;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("variant identifier")
            }

            fn visit_str<E: de::Error>(self, name: &str) -> Result<Self::Value, E> {
                ResponseFormatKind::ALL
                    .into_iter()
                    .find(|kind| kind.name() == name)
                    .ok_or_else(|| E::unknown_variant(name, RESPONSE_FORMAT_KINDS))
            }
        }

        deserializer.deserialize_identifier(KindVisitor)
    }
}

/// The object is read field by field. serde's derive for an internally
/// tagged enum buffers the whole object before it picks the variant, so a
/// JSON schema would be built twice (the buffer, then the `Value`); this
/// reads `type` and `json_schema` in whatever order they arrive and builds
/// the schema once: straight into its struct once the kind is known to want
/// it, skipped once the kind is known not to, and kept as it arrived only
/// when it arrives before the kind (a `KeptSchema`: an object as its entries
/// in order, any other shape as a `Value`, for the schema's own derive to
/// read at the end, so a repeated or malformed field is the derive's error
/// in either order and ignored by the two kinds that carry no schema). The
/// accepted shapes and the errors are the derive's: a missing `type`, an
/// unknown kind, a missing or malformed `json_schema` for the kind that
/// wants it, a duplicate key; a kind that carries no schema ignores a
/// `json_schema` field of any shape, and every other key is ignored. One
/// corner differs in text alone: with `type` first, a malformed schema
/// followed by a repeated `type` reads the schema's error here, because the
/// schema is read the moment the kind is known (the point of this reader),
/// where the derive, still buffering, reads a duplicate `type`; both refuse
/// the body.
impl<'de> Deserialize<'de> for ResponseFormat {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        enum Field {
            Kind,
            JsonSchema,
            Other,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct FieldVisitor;

                impl Visitor<'_> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("field identifier")
                    }

                    fn visit_str<E: de::Error>(self, name: &str) -> Result<Self::Value, E> {
                        Ok(match name {
                            "type" => Field::Kind,
                            "json_schema" => Field::JsonSchema,
                            _ => Field::Other,
                        })
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        /// What was read under `json_schema` before the end of the object.
        enum Schema {
            /// Read straight into its struct: the kind was known to want it.
            Read(JsonSchemaFormat),
            /// Kept as read: the kind was not known yet.
            Kept(KeptSchema),
            /// Skipped: the kind was known not to want it.
            Skipped,
        }

        struct FormatVisitor;

        impl<'de> Visitor<'de> for FormatVisitor {
            type Value = ResponseFormat;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("internally tagged enum ResponseFormat")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                // serde's derive also reads the kind and then the content
                // from a sequence; the shape and its errors are kept
                let kind: ResponseFormatKind = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("type"))?;
                let (format, read) = match kind {
                    ResponseFormatKind::Text => (ResponseFormat::Text, 0),
                    ResponseFormatKind::JsonObject => (ResponseFormat::JsonObject, 0),
                    ResponseFormatKind::JsonSchema => {
                        let json_schema = seq.next_element()?.ok_or_else(|| {
                            de::Error::invalid_length(
                                0,
                                &"struct variant ResponseFormat::JsonSchema with 1 element",
                            )
                        })?;
                        (ResponseFormat::JsonSchema { json_schema }, 1)
                    }
                };
                let mut extra = 0;
                while seq.next_element::<de::IgnoredAny>()?.is_some() {
                    extra += 1;
                }
                if extra > 0 {
                    let expected: &dyn de::Expected = if read == 1 {
                        &"1 element in sequence"
                    } else {
                        &"0 elements in sequence"
                    };
                    return Err(de::Error::invalid_length(read + extra, expected));
                }
                Ok(format)
            }

            fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut kind: Option<ResponseFormatKind> = None;
                let mut schema: Option<Schema> = None;
                let mut schema_twice = false;
                while let Some(field) = map.next_key::<Field>()? {
                    match field {
                        Field::Kind => {
                            if kind.is_some() {
                                return Err(de::Error::duplicate_field("type"));
                            }
                            kind = Some(map.next_value()?);
                        }
                        Field::JsonSchema => {
                            if schema.is_some() {
                                // the derive reports the duplicate once the
                                // object is read, whatever the second value
                                schema_twice = true;
                                map.next_value::<de::IgnoredAny>()?;
                                continue;
                            }
                            schema = Some(match kind {
                                Some(ResponseFormatKind::JsonSchema) => {
                                    Schema::Read(map.next_value()?)
                                }
                                Some(_) => {
                                    map.next_value::<de::IgnoredAny>()?;
                                    Schema::Skipped
                                }
                                None => Schema::Kept(map.next_value()?),
                            });
                        }
                        Field::Other => {
                            map.next_value::<de::IgnoredAny>()?;
                        }
                    }
                }
                let kind = kind.ok_or_else(|| de::Error::missing_field("type"))?;
                Ok(match kind {
                    ResponseFormatKind::Text => ResponseFormat::Text,
                    ResponseFormatKind::JsonObject => ResponseFormat::JsonObject,
                    ResponseFormatKind::JsonSchema => {
                        let json_schema = match schema {
                            Some(Schema::Read(json_schema)) => json_schema,
                            Some(Schema::Kept(KeptSchema::Object(entries))) => {
                                // the derive's own visitor reads the entries as they
                                // arrived, duplicates included, and raises its error first
                                JsonSchemaFormat::deserialize(
                                    MapDeserializer::<_, serde_json::Error>::new(
                                        entries.into_iter(),
                                    ),
                                )
                                .map_err(de::Error::custom)?
                            }
                            Some(Schema::Kept(KeptSchema::Other(value))) => {
                                serde_json::from_value(value).map_err(de::Error::custom)?
                            }
                            Some(Schema::Skipped) | None => {
                                return Err(de::Error::missing_field("json_schema"))
                            }
                        };
                        if schema_twice {
                            // the derive reads the first value before it meets the
                            // second key: the value's own error comes first
                            return Err(de::Error::duplicate_field("json_schema"));
                        }
                        ResponseFormat::JsonSchema { json_schema }
                    }
                })
            }
        }

        deserializer.deserialize_any(FormatVisitor)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct JsonSchemaFormat {
    pub name: String,
    pub schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A `json_schema` object that arrived before the kind, kept for the end of
/// the enclosing object: an object's entries as they arrived (duplicates
/// included), so that the schema's own derive reads them at the end and raises
/// its own error first, whatever it is, exactly as it does when `type` comes
/// first; any other shape is kept as a `Value`, for the kind to accept or
/// refuse. The two kinds that carry no schema drop it unread.
enum KeptSchema {
    Object(Vec<(String, Value)>),
    Other(Value),
}

impl<'de> Deserialize<'de> for KeptSchema {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KeptVisitor;

        impl<'de> Visitor<'de> for KeptVisitor {
            type Value = KeptSchema;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("any valid JSON value")
            }

            fn visit_unit<E: de::Error>(self) -> Result<KeptSchema, E> {
                Ok(KeptSchema::Other(Value::Null))
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<KeptSchema, E> {
                Ok(KeptSchema::Other(Value::Bool(value)))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<KeptSchema, E> {
                Ok(KeptSchema::Other(Value::from(value)))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<KeptSchema, E> {
                Ok(KeptSchema::Other(Value::from(value)))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<KeptSchema, E> {
                Ok(KeptSchema::Other(Value::from(value)))
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<KeptSchema, E> {
                Ok(KeptSchema::Other(Value::String(value.to_owned())))
            }

            fn visit_string<E: de::Error>(self, value: String) -> Result<KeptSchema, E> {
                Ok(KeptSchema::Other(Value::String(value)))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<KeptSchema, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(KeptSchema::Other(Value::Array(items)))
            }

            fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<KeptSchema, A::Error> {
                let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some(key) = map.next_key::<String>()? {
                    let value = map.next_value()?;
                    entries.push((key, value));
                }
                Ok(KeptSchema::Object(entries))
            }
        }

        deserializer.deserialize_any(KeptVisitor)
    }
}

/// Strict structured outputs (`strict: true`) require every object node of the
/// schema to pin `additionalProperties: false`, as the public API enforces.
/// Walks `properties`, `items`, `prefixItems`, `anyOf`/`oneOf`/`allOf`,
/// `$defs`/`definitions` and `not`; returns the JSON-pointer-like path of the
/// first object node that lacks the pin.
pub fn strict_schema_missing_additional_properties(schema: &Value) -> Option<String> {
    fn walk(node: &Value, path: &str) -> Option<String> {
        let obj = node.as_object()?;
        let is_object_node = obj.get("type").and_then(Value::as_str) == Some("object")
            || obj
                .get("type")
                .and_then(Value::as_array)
                .is_some_and(|types| types.iter().any(|t| t.as_str() == Some("object")))
            || (obj.get("type").is_none() && obj.contains_key("properties"));
        if is_object_node && obj.get("additionalProperties") != Some(&Value::Bool(false)) {
            return Some(if path.is_empty() {
                "()".to_string()
            } else {
                path.to_string()
            });
        }
        for key in ["properties", "$defs", "definitions"] {
            if let Some(children) = obj.get(key).and_then(Value::as_object) {
                for (name, child) in children {
                    if let Some(found) = walk(child, &format!("{path}/{key}/{name}")) {
                        return Some(found);
                    }
                }
            }
        }
        for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
            if let Some(children) = obj.get(key).and_then(Value::as_array) {
                for (index, child) in children.iter().enumerate() {
                    if let Some(found) = walk(child, &format!("{path}/{key}/{index}")) {
                        return Some(found);
                    }
                }
            }
        }
        for key in ["items", "not"] {
            if let Some(child) = obj.get(key) {
                if let Some(found) = walk(child, &format!("{path}/{key}")) {
                    return Some(found);
                }
            }
        }
        None
    }
    walk(schema, "")
}

/// Shared shape rules for a json_schema format: name non-empty, schema a JSON object.
pub fn validate_json_schema_shape(
    name: &str,
    schema: &Value,
) -> Result<(), validator::ValidationError> {
    let (code, message) = if name.is_empty() {
        ("json_schema_name_empty", "JSON schema name cannot be empty")
    } else if !schema.is_object() {
        (
            "json_schema_schema_not_object",
            "JSON schema 'schema' must be a JSON object",
        )
    } else {
        return Ok(());
    };
    Err(invalid(code, message))
}

// ============================================================================
// Streaming
// ============================================================================

#[derive(Debug, Clone, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub struct StreamOptions {
    /// Chat Completions / Completions: include usage block at end of stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,

    /// Chat Completions / Completions: emit a usage chunk with every streamed
    /// delta instead of only in the final chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continuous_usage_stats: Option<bool>,

    /// Responses API: add random chars on `obfuscation` field of delta events
    /// to normalize payload sizes. Defaults to `true` upstream when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,

    /// Additional fields not explicitly defined above (e.g. engine-specific
    /// streaming options). Without this, the gateway silently drops them while
    /// re-serializing the request for the backend.
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    pub function: Option<FunctionCallDelta>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct FunctionCallDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

// ============================================================================
// Tools and Function Calling
// ============================================================================

/// Tool choice value for simple string options
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceValue {
    Auto,
    Required,
    None,
}

/// Tool choice for both Chat Completion and Responses APIs
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ToolChoice {
    Value(ToolChoiceValue),
    Function {
        #[serde(rename = "type")]
        tool_type: String, // "function"
        function: FunctionChoice,
    },
    AllowedTools {
        #[serde(rename = "type")]
        tool_type: String, // "allowed_tools"
        mode: String, // "auto" | "required" TODO: need validation
        tools: Vec<ToolReference>,
    },
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Value(ToolChoiceValue::Auto)
    }
}

impl ToolChoice {
    /// Whether the choice leaves the model no choice but a call: a named
    /// function, `required`, or allowed tools in `required` mode.
    pub fn forces_call(&self) -> bool {
        match self {
            ToolChoice::Function { .. } => true,
            ToolChoice::Value(ToolChoiceValue::Required) => true,
            ToolChoice::AllowedTools { mode, .. } => mode == "required",
            ToolChoice::Value(_) => false,
        }
    }

    /// Serialize tool_choice to string for ResponsesResponse
    ///
    /// Returns the JSON-serialized tool_choice or "auto" as default
    pub fn serialize_to_string(tool_choice: Option<&ToolChoice>) -> String {
        tool_choice
            .map(|tc| serde_json::to_string(tc).unwrap_or_else(|_| "auto".to_string()))
            .unwrap_or_else(|| "auto".to_string())
    }

    /// The subset of `tools` this choice lets the model call, in the original
    /// order: the named function for the function form, the listed functions
    /// for `allowed_tools`. `None` when the choice does not narrow the list
    /// (`auto`, `none`, `required`).
    pub fn narrow_tools(&self, tools: &[Tool]) -> Option<Vec<Tool>> {
        match self {
            ToolChoice::AllowedTools { tools: allowed, .. } => {
                let allowed: HashSet<&str> = allowed
                    .iter()
                    .filter_map(ToolReference::function_name)
                    .collect();
                Some(
                    tools
                        .iter()
                        .filter(|tool| allowed.contains(tool.function.name.as_str()))
                        .cloned()
                        .collect(),
                )
            }
            ToolChoice::Function { function, .. } => Some(
                tools
                    .iter()
                    .filter(|tool| tool.function.name == function.name)
                    .cloned()
                    .collect(),
            ),
            ToolChoice::Value(_) => None,
        }
    }
}

/// Function choice specification for ToolChoice::Function
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct FunctionChoice {
    pub name: String,
}

/// Tool reference for ToolChoice::AllowedTools
///
/// Represents a reference to a specific tool in the allowed_tools array.
/// Different tool types have different required fields.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ToolReference {
    /// Reference to a function tool
    #[serde(rename = "function")]
    Function { name: String },

    /// Reference to an MCP tool
    #[serde(rename = "mcp")]
    Mcp {
        server_label: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// File search hosted tool
    #[serde(rename = "file_search")]
    FileSearch,

    /// Web search preview hosted tool
    #[serde(rename = "web_search_preview")]
    WebSearchPreview,

    /// Computer use preview hosted tool
    #[serde(rename = "computer_use_preview")]
    ComputerUsePreview,

    /// Code interpreter hosted tool
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,

    /// Image generation hosted tool
    #[serde(rename = "image_generation")]
    ImageGeneration,
}

impl ToolReference {
    /// Get a unique identifier for this tool reference
    pub fn identifier(&self) -> String {
        match self {
            ToolReference::Function { name } => format!("function:{name}"),
            ToolReference::Mcp { server_label, name } => {
                if let Some(n) = name {
                    format!("mcp:{server_label}:{n}")
                } else {
                    format!("mcp:{server_label}")
                }
            }
            ToolReference::FileSearch => "file_search".to_string(),
            ToolReference::WebSearchPreview => "web_search_preview".to_string(),
            ToolReference::ComputerUsePreview => "computer_use_preview".to_string(),
            ToolReference::CodeInterpreter => "code_interpreter".to_string(),
            ToolReference::ImageGeneration => "image_generation".to_string(),
        }
    }

    /// Get the tool name if this is a function tool
    pub fn function_name(&self) -> Option<&str> {
        match self {
            ToolReference::Function { name } => Some(name.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: Function,
}

/// The schema a function declared without `parameters` is read with wherever
/// a schema is needed: per the OpenAI spec, omitting the field defines a
/// function with an empty parameter list.
static EMPTY_PARAMETERS_SCHEMA: LazyLock<Value> = LazyLock::new(|| Value::Object(Map::new()));

/// `parameters` as the client gave it: `None` for an explicit `null` as for
/// an absent field. The serving engine reads both as "no schema" and its own
/// template sees `null` for both, so the two must stay apart from a client's
/// own `{}`, which the template renders as `{}`.
fn deserialize_parameters<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Value>, D::Error> {
    let value = Value::deserialize(deserializer)?;
    Ok((!value.is_null()).then_some(value))
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct Function {
    pub name: String,
    pub description: Option<String>,
    /// The parameter schema as the client gave it; `None` when it gave none
    /// (the field absent or `null`). Left out of a serialized function then,
    /// as any optional field the client left out (a Responses function tool,
    /// whose reference lists the field as required and nullable, writes `null`
    /// for it: [`crate::responses::FunctionTool`]); the chat template is handed
    /// `null` for it (what the engine's own template reads for such a
    /// function). Read through [`Function::parameters_schema`] where a schema
    /// object is needed.
    #[serde(default, deserialize_with = "deserialize_parameters")]
    pub parameters: Option<Value>, // JSON Schema
    /// Whether to enable strict schema adherence (OpenAI structured outputs)
    pub strict: Option<bool>,
    /// The fields of the function object beyond the ones above, as the request
    /// carried them and in their order: a vendor's extension, such as the
    /// `response` schema some tool corpora describe a tool's result with. The
    /// chat template renders the tool object as given, so they must reach it
    /// as they reach the engine's own template, and a forwarded request keeps
    /// them.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Function {
    /// The schema the function is read with: the client's, or the empty
    /// schema when it gave none. The constraint builders, the structural tags
    /// and the parsers read an object; only the template sees the difference.
    pub fn parameters_schema(&self) -> &Value {
        self.parameters.as_ref().unwrap_or(&EMPTY_PARAMETERS_SCHEMA)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: FunctionCallResponse,
}

/// Deprecated `function_call` field from the OpenAI API.
/// Can be `"none"`, `"auto"`, or `{"name": "function_name"}`.
#[derive(Debug, Clone)]
pub enum FunctionCall {
    None,
    Auto,
    Function { name: String },
}

impl Serialize for FunctionCall {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            FunctionCall::None => serializer.serialize_str("none"),
            FunctionCall::Auto => serializer.serialize_str("auto"),
            FunctionCall::Function { name } => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("name", name)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for FunctionCall {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match &value {
            Value::String(s) => match s.as_str() {
                "none" => Ok(FunctionCall::None),
                "auto" => Ok(FunctionCall::Auto),
                other => Err(de::Error::custom(format!(
                    "unknown function_call value: \"{other}\""
                ))),
            },
            Value::Object(map) => {
                if let Some(Value::String(name)) = map.get("name") {
                    Ok(FunctionCall::Function { name: name.clone() })
                } else {
                    Err(de::Error::custom(
                        "function_call object must have a \"name\" string field",
                    ))
                }
            }
            _ => Err(de::Error::custom(
                "function_call must be a string or object",
            )),
        }
    }
}

impl schemars::JsonSchema for FunctionCall {
    fn schema_name() -> Cow<'static, str> {
        "FunctionCall".into()
    }
    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        // FunctionCall is either "none", "auto", or {"name": "..."}
        let name_schema = generator.subschema_for::<String>();
        schemars::json_schema!({
            "anyOf": [
                {
                    "type": "string",
                    "enum": ["none", "auto"]
                },
                {
                    "type": "object",
                    "properties": { "name": name_schema },
                    "required": ["name"]
                }
            ]
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct FunctionCallResponse {
    pub name: String,
    /// The call's arguments as JSON text, the shape the API documents and a
    /// response carries. A request's history may carry a call's arguments as
    /// the JSON object itself (a client that replays the parsed call rather
    /// than the text it received); the engines accept both shapes and hand
    /// their template an object either way, so both are read here, an
    /// object kept as its JSON text.
    #[serde(default, deserialize_with = "deserialize_arguments_text")]
    pub arguments: Option<String>, // JSON string
}

/// Tool-call `arguments` as text: a JSON string as written, a JSON object as
/// its compact JSON text (keys in their order), null or absent as none. Any
/// other shape is refused.
fn deserialize_arguments_text<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    match Option::<Value>::deserialize(deserializer)? {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text)),
        Some(object @ Value::Object(_)) => Ok(Some(object.to_string())),
        Some(Value::Array(_)) => Err(de::Error::invalid_type(
            de::Unexpected::Seq,
            &"a JSON string or an object",
        )),
        Some(Value::Bool(flag)) => Err(de::Error::invalid_type(
            de::Unexpected::Bool(flag),
            &"a JSON string or an object",
        )),
        Some(Value::Number(_)) => Err(de::Error::invalid_type(
            de::Unexpected::Other("a number"),
            &"a JSON string or an object",
        )),
    }
}

// ============================================================================
// Usage and Logging
// ============================================================================
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_tokens_details: Option<PromptTokenUsageInfo>,
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

impl Usage {
    /// Create a Usage from prompt and completion token counts
    pub fn from_counts(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }

    /// Add cached token details to this Usage
    pub fn with_cached_tokens(mut self, cached_tokens: u32) -> Self {
        // Calling this builder means the backend supplied cache accounting.
        // Zero is therefore evidence of a cold miss, not absence of support,
        // and must remain distinguishable from `prompt_tokens_details: None`.
        self.prompt_tokens_details = Some(PromptTokenUsageInfo {
            cached_tokens,
            audio_tokens: None,
        });
        self
    }

    /// Add reasoning token details to this Usage
    pub fn with_reasoning_tokens(mut self, reasoning_tokens: u32) -> Self {
        if reasoning_tokens > 0 {
            self.completion_tokens_details
                .get_or_insert_default()
                .reasoning_tokens = Some(reasoning_tokens);
        }
        self
    }

    /// Add speculative decoding details to this Usage.
    pub fn with_speculative_tokens(mut self, accepted: u32, drafted: u32) -> Self {
        if drafted > 0 {
            let details = self.completion_tokens_details.get_or_insert_default();
            details.accepted_prediction_tokens = Some(accepted);
            details.rejected_prediction_tokens = Some(drafted.saturating_sub(accepted));
        }
        self
    }

    /// Drop prompt tokens the provider does not bill (the rendered generation stub).
    pub fn with_unbilled_prompt_tokens(mut self, unbilled: u32) -> Self {
        self.prompt_tokens = self.prompt_tokens.saturating_sub(unbilled);
        self.total_tokens = self.prompt_tokens + self.completion_tokens;
        if let Some(details) = &mut self.prompt_tokens_details {
            details.cached_tokens = details.cached_tokens.min(self.prompt_tokens);
        }
        self
    }

    /// Every detail counter present, zero where nothing was counted: the
    /// shape the OpenAI API sends on every chat completion
    /// (`prompt_tokens_details.{cached_tokens,audio_tokens}` and
    /// `completion_tokens_details.{reasoning_tokens,audio_tokens,
    /// accepted_prediction_tokens,rejected_prediction_tokens}`), which typed
    /// clients read without a presence check. Counters already set are kept.
    pub fn with_complete_details(mut self) -> Self {
        let prompt = self
            .prompt_tokens_details
            .get_or_insert(PromptTokenUsageInfo {
                cached_tokens: 0,
                audio_tokens: None,
            });
        prompt.audio_tokens.get_or_insert(0);
        let completion = self.completion_tokens_details.get_or_insert_default();
        completion.reasoning_tokens.get_or_insert(0);
        completion.audio_tokens.get_or_insert(0);
        completion.accepted_prediction_tokens.get_or_insert(0);
        completion.rejected_prediction_tokens.get_or_insert(0);
        self
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: Option<u32>,
    pub audio_tokens: Option<u32>,
    pub accepted_prediction_tokens: Option<u32>,
    pub rejected_prediction_tokens: Option<u32>,
}

/// Usage information (used by rerank and other endpoints)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub reasoning_tokens: Option<u32>,
    pub prompt_tokens_details: Option<PromptTokenUsageInfo>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct PromptTokenUsageInfo {
    pub cached_tokens: u32,
    #[serde(default)]
    pub audio_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct LogProbs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<Option<HashMap<String, f32>>>,
    pub text_offset: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ChatLogProbs {
    /// The public API's object: `content` and `refusal` are both always
    /// present (`null` when there is none); the gateway produces no refusals,
    /// so `refusal` is `null`.
    Detailed {
        #[serde(default)]
        content: Option<Vec<ChatLogProbsContent>>,
        #[serde(default)]
        refusal: Option<Vec<ChatLogProbsContent>>,
    },
    Raw(Value),
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ChatLogProbsContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogProb>,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

// ============================================================================
// Input Types
// ============================================================================

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum InputIds {
    Single(Vec<i32>),
    Batch(Vec<Vec<i32>>),
}

/// Shape probe for the first `input_ids` element: it alone picks the variant,
/// so the rest parses in place instead of through untagged-enum buffering.
enum FirstInputId {
    Id(i32),
    Ids(Vec<i32>),
}

impl<'de> Deserialize<'de> for FirstInputId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FirstInputIdVisitor;

        impl<'de> Visitor<'de> for FirstInputIdVisitor {
            type Value = FirstInputId;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a token id or an array of token ids")
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                i32::try_from(v)
                    .map(FirstInputId::Id)
                    .map_err(|_| E::invalid_value(de::Unexpected::Signed(v), &self))
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                i32::try_from(v)
                    .map(FirstInputId::Id)
                    .map_err(|_| E::invalid_value(de::Unexpected::Unsigned(v), &self))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
                Deserialize::deserialize(SeqAccessDeserializer::new(seq)).map(FirstInputId::Ids)
            }
        }

        deserializer.deserialize_any(FirstInputIdVisitor)
    }
}

impl<'de> Deserialize<'de> for InputIds {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct InputIdsVisitor;

        impl<'de> Visitor<'de> for InputIdsVisitor {
            type Value = InputIds;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an array of token ids or an array of token id arrays")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                // An empty array is Single, matching untagged first-match order.
                let Some(first) = seq.next_element::<FirstInputId>()? else {
                    return Ok(InputIds::Single(Vec::new()));
                };
                let remaining = seq.size_hint().unwrap_or(0);
                match first {
                    FirstInputId::Id(id) => {
                        let mut ids = Vec::with_capacity(remaining.saturating_add(1));
                        ids.push(id);
                        while let Some(id) = seq.next_element()? {
                            ids.push(id);
                        }
                        Ok(InputIds::Single(ids))
                    }
                    FirstInputId::Ids(head) => {
                        let mut seqs = Vec::with_capacity(remaining.saturating_add(1));
                        seqs.push(head);
                        while let Some(ids) = seq.next_element()? {
                            seqs.push(ids);
                        }
                        Ok(InputIds::Batch(seqs))
                    }
                }
            }
        }

        deserializer.deserialize_seq(InputIdsVisitor)
    }
}

/// Token ids as their decimal rendering joined by spaces: the routing text of
/// a prompt given as ids, on `/generate` and `/v1/completions` alike.
pub(crate) fn join_ids<T: std::fmt::Display>(ids: impl IntoIterator<Item = T>) -> String {
    use std::fmt::Write as _;
    let ids = ids.into_iter();
    let mut text = String::with_capacity(ids.size_hint().0 * 6);
    for (i, id) in ids.enumerate() {
        if i > 0 {
            text.push(' ');
        }
        let _ = write!(text, "{id}");
    }
    text
}

/// LoRA adapter path - can be single path or batch of paths (an engine extension)
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum LoRAPath {
    Single(Option<String>),
    Batch(Vec<Option<String>>),
}

// ============================================================================
// Redacted Types
// ============================================================================
#[derive(Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Redacted(pub String);

impl std::fmt::Debug for Redacted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[REDACTED]")
    }
}

// ============================================================================
// Response Prompt
// ============================================================================

/// Reference to a prompt template and its variables.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ResponsePrompt {
    pub id: String,
    pub variables: Option<HashMap<String, PromptVariable>>,
    pub version: Option<String>,
}

/// A prompt variable value: plain string or a typed input (text, image, file).
///
/// Variant order matters for `#[serde(untagged)]`: a bare JSON string succeeds
/// as `String`; a JSON object falls through to `Typed`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum PromptVariable {
    String(String),
    Typed(PromptVariableTyped),
}

/// Typed prompt variable input.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type")]
#[expect(
    clippy::enum_variant_names,
    reason = "variant names match OpenAI API spec"
)]
pub enum PromptVariableTyped {
    #[serde(rename = "input_text")]
    ResponseInputText { text: String },
    #[serde(rename = "input_image")]
    ResponseInputImage {
        detail: Option<Detail>,
        file_id: Option<String>,
        image_url: Option<String>,
    },
    #[serde(rename = "input_file")]
    ResponseInputFile {
        file_data: Option<String>,
        file_id: Option<String>,
        file_url: Option<String>,
        filename: Option<String>,
    },
}

/// Image detail level for [`PromptVariableTyped::ResponseInputImage`] and
/// [`crate::responses::ResponseContentPart::InputImage`]. Spec allows
/// `"low" | "high" | "auto" | "original"`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Detail {
    Low,
    High,
    #[default]
    Auto,
    Original,
}

// ============================================================================
// Responses API: prompt-cache retention & context management
// ============================================================================

/// Retention policy for prompt-cache entries on the Responses API.
///
/// Spec: `prompt_cache_retention: "in-memory" | "24h"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub enum PromptCacheRetention {
    #[serde(rename = "in-memory")]
    InMemory,
    #[serde(rename = "24h")]
    Duration24h,
}

/// A single entry in the Responses API `context_management` array.
///
/// Spec: each entry has `type` (currently only `"compaction"`) and an optional
/// `compact_threshold` token count.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ContextManagementEntry {
    #[serde(rename = "type")]
    pub r#type: ContextManagementType,
    pub compact_threshold: Option<u32>,
}

/// Type tag for [`ContextManagementEntry`]. Currently only `compaction` is
/// defined by the spec; the enum is kept small so unknown values serde-fail
/// (consistent with P5's fail-fast direction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ContextManagementType {
    Compaction,
}

// ============================================================================
// Responses API: conversation reference
// ============================================================================

/// Reference to a conversation the response belongs to.
///
/// Spec: `conversation: string | ResponseConversationParam { id: string }`.
/// Variant order matters for `#[serde(untagged)]`: a bare JSON string succeeds
/// as `Id`; an object falls through to `Object`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ConversationRef {
    Id(String),
    Object { id: String },
}

impl ConversationRef {
    /// Return the underlying conversation id regardless of the wire shape.
    pub fn as_id(&self) -> &str {
        match self {
            Self::Id(id) | Self::Object { id } => id.as_str(),
        }
    }

    /// `true` when the underlying conversation id is the empty string.
    /// Mirrors `String::is_empty` for callers that previously treated
    /// `Option<String>` empty values as "unset".
    pub fn is_empty(&self) -> bool {
        self.as_id().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use serde_json::json;

    use super::*;

    /// A history tool call's `arguments`: the JSON text the API documents,
    /// or the JSON object itself, kept as its text so every reader and a
    /// forwarded request see the documented shape; null and absent are
    /// none; any other shape is refused.
    #[test]
    fn tool_call_arguments_are_read_as_text_or_as_an_object() {
        let call = |arguments: Value| -> Result<ToolCall, serde_json::Error> {
            serde_json::from_value(json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": arguments}
            }))
        };
        let text = call(json!("{\"city\": \"Paris\"}")).expect("text");
        assert_eq!(
            text.function.arguments.as_deref(),
            Some("{\"city\": \"Paris\"}")
        );

        let object = call(json!({"unit": "celsius", "city": "Paris"})).expect("object");
        assert_eq!(
            object.function.arguments.as_deref(),
            Some("{\"unit\":\"celsius\",\"city\":\"Paris\"}")
        );
        let forwarded = serde_json::to_value(&object).expect("serialize");
        assert_eq!(
            forwarded["function"]["arguments"],
            json!("{\"unit\":\"celsius\",\"city\":\"Paris\"}")
        );

        assert!(call(Value::Null)
            .expect("null")
            .function
            .arguments
            .is_none());
        let absent: ToolCall = serde_json::from_value(json!({
            "id": "call_1", "type": "function", "function": {"name": "get_weather"}
        }))
        .expect("absent");
        assert!(absent.function.arguments.is_none());

        for other in [json!(42), json!(["x"]), json!(true)] {
            let error = call(other.clone()).expect_err("refused").to_string();
            assert!(
                error.contains("expected a JSON string or an object"),
                "{other}: {error}"
            );
        }
    }

    #[test]
    fn the_logprobs_object_carries_content_and_refusal() {
        // Parsed as an engine reports it (content only), the object leaves
        // with both of the public API's properties, `refusal` null.
        let logprobs: ChatLogProbs = serde_json::from_value(json!({
            "content": [{"token": "Hi", "logprob": -0.1, "bytes": [72, 105], "top_logprobs": []}]
        }))
        .expect("parse");
        assert!(matches!(logprobs, ChatLogProbs::Detailed { .. }));
        let value = serde_json::to_value(&logprobs).expect("serialize");
        assert_eq!(value["content"][0]["token"], "Hi", "{value}");
        assert_eq!(value.get("refusal"), Some(&Value::Null), "{value}");
    }

    #[test]
    fn to_value_exact_keeps_f32_fields_as_the_client_wrote_them() {
        let request: crate::chat::ChatCompletionRequest = serde_json::from_value(json!({
            "model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.95,
            "temperature": 0.7
        }))
        .unwrap();
        // The widening this guards against: 0.95f32 is 0.949999988079071 as f64.
        assert_ne!(
            serde_json::to_value(&request).unwrap()["top_p"],
            json!(0.95)
        );
        let exact = to_value_exact(&request).unwrap();
        assert_eq!(exact["top_p"], json!(0.95));
        assert_eq!(exact["temperature"], json!(0.7));
        assert_eq!(exact["model"], json!("glm-5.3-flash"));
    }

    /// A tool definition may carry fields the OpenAI schema does not name
    /// (the `response` schema of a BFCL tool); the typed view keeps them, in
    /// their order, so the chat template and a forwarded request see the tool
    /// as the client wrote it.
    #[test]
    fn a_function_keeps_the_fields_beyond_the_schema_in_their_order() {
        let written = json!({
            "type": "function",
            "function": {
                "name": "authenticate",
                "description": "Authenticate a user.",
                "parameters": {"type": "object", "properties": {"user": {"type": "string"}}},
                "response": {"type": "dict", "properties": {"ok": {"type": "boolean"}}},
                "x_vendor": ["kept", 1]
            }
        });
        let tool: Tool = serde_json::from_value(written.clone()).unwrap();
        assert_eq!(tool.function.name, "authenticate");
        assert_eq!(tool.function.extra.len(), 2);
        assert_eq!(
            serde_json::to_string(&tool).unwrap(),
            serde_json::to_string(&written).unwrap()
        );
    }

    #[derive(Deserialize)]
    struct NullableBoolTest {
        #[serde(default, deserialize_with = "deserialize_null_as_false")]
        field: bool,
    }

    #[test]
    fn test_deserialize_null_as_false() {
        let cases = [
            (json!({"field": true}), true),
            (json!({"field": false}), false),
            (json!({"field": null}), false),
            (json!({}), false),
        ];
        for (input, expected) in cases {
            let t: NullableBoolTest = serde_json::from_value(input).unwrap();
            assert_eq!(t.field, expected);
        }
    }

    #[test]
    fn test_deserialize_null_as_false_rejects_non_bool() {
        let result = serde_json::from_value::<NullableBoolTest>(json!({"field": "yes"}));
        assert!(result.is_err());
    }

    #[test]
    fn stream_options_preserve_unknown_fields() {
        let raw = json!({
            "include_usage": true,
            "continuous_usage_stats": true,
            "step_usage_chunks": "all",
            "engine_specific": {"nested": 1},
        });

        let opts: StreamOptions = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(opts.include_usage, Some(true));
        assert_eq!(opts.continuous_usage_stats, Some(true));
        assert_eq!(opts.other.get("step_usage_chunks"), Some(&json!("all")));

        // Re-serializing must round-trip the engine-specific keys, otherwise the
        // gateway would strip them on the way to the backend.
        assert_eq!(serde_json::to_value(&opts).unwrap(), raw);
    }

    #[test]
    fn stream_options_without_unknown_fields_stay_compact() {
        let opts: StreamOptions = serde_json::from_value(json!({"include_usage": true})).unwrap();
        assert!(opts.other.is_empty());
        assert_eq!(
            serde_json::to_value(&opts).unwrap(),
            json!({"include_usage": true})
        );
    }

    #[test]
    fn cached_token_builder_preserves_explicit_zero() {
        let usage = Usage::from_counts(16, 1).with_cached_tokens(0);
        assert!(matches!(
            usage.prompt_tokens_details,
            Some(PromptTokenUsageInfo {
                cached_tokens: 0,
                audio_tokens: None,
            })
        ));
    }

    #[test]
    fn unbilled_prompt_tokens_come_off_the_prompt_and_total() {
        let usage = Usage::from_counts(10, 4)
            .with_cached_tokens(10)
            .with_unbilled_prompt_tokens(3);
        assert_eq!(
            (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens
            ),
            (7, 4, 11)
        );
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens),
            Some(7),
            "cached tokens are clamped to the billed prompt"
        );

        let saturated = Usage::from_counts(2, 4).with_unbilled_prompt_tokens(3);
        assert_eq!((saturated.prompt_tokens, saturated.total_tokens), (0, 4));
        assert!(saturated.prompt_tokens_details.is_none());

        let unchanged = Usage::from_counts(10, 4).with_unbilled_prompt_tokens(0);
        assert_eq!((unchanged.prompt_tokens, unchanged.total_tokens), (10, 14));
    }

    #[test]
    fn content_part_deserializes_audio_url() {
        let value = json!({
            "type": "audio_url",
            "audio_url": {
                "url": "https://example.com/audio.wav"
            }
        });
        let part: ContentPart = serde_json::from_value(value).expect("audio_url content part");
        assert_eq!(
            part,
            ContentPart::AudioUrl {
                audio_url: AudioUrl {
                    url: "https://example.com/audio.wav".to_string(),
                },
            }
        );
    }

    #[test]
    fn content_part_round_trips_unknown_type() {
        let value = json!({
            "type": "vendor_special",
            "payload": {"items": [1, "two", null], "enabled": true},
            "vendor_option": "keep"
        });
        let part: ContentPart = serde_json::from_value(value.clone()).unwrap();

        assert!(
            matches!(&part, ContentPart::Unknown(fields) if fields["type"] == "vendor_special")
        );
        assert_eq!(serde_json::to_value(&part).unwrap(), value);
    }

    #[test]
    fn content_part_rejects_non_objects() {
        for value in [
            json!(null),
            json!(true),
            json!(42),
            json!("text"),
            json!([]),
        ] {
            assert!(serde_json::from_value::<ContentPart>(value).is_err());
        }
    }

    #[test]
    fn content_part_known_types_take_precedence_over_fallback() {
        for value in [
            json!({"type": "text", "text": "hello"}),
            json!({"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}),
            json!({"type": "audio_url", "audio_url": {"url": "https://example.com/audio.wav"}}),
            json!({"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}),
            json!({"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}),
        ] {
            let part: ContentPart = serde_json::from_value(value.clone()).unwrap();
            assert!(!matches!(&part, ContentPart::Unknown(_)));
            assert_eq!(serde_json::to_value(&part).unwrap(), value);
        }
    }

    #[test]
    fn content_part_round_trips_input_audio() {
        let value = json!({
            "type": "input_audio",
            "input_audio": {
                "data": "UklGRg==",
                "format": "wav"
            }
        });
        let part: ContentPart =
            serde_json::from_value(value.clone()).expect("input_audio content part");
        assert_eq!(
            part,
            ContentPart::InputAudio {
                input_audio: InputAudio {
                    data: "UklGRg==".to_string(),
                    format: "wav".to_string(),
                },
            }
        );
        assert_eq!(serde_json::to_value(part).unwrap(), value);
    }

    #[test]
    fn conversation_ref_deserializes_bare_string() {
        let v = json!("conv_abc");
        let r: ConversationRef = serde_json::from_value(v).expect("string form");
        assert!(matches!(r, ConversationRef::Id(ref s) if s == "conv_abc"));
        assert_eq!(r.as_id(), "conv_abc");
        // Bare string round-trips back to a JSON string.
        assert_eq!(serde_json::to_value(&r).unwrap(), json!("conv_abc"));
    }

    #[test]
    fn conversation_ref_deserializes_object() {
        let v = json!({"id": "conv_xyz"});
        let r: ConversationRef = serde_json::from_value(v).expect("object form");
        assert!(matches!(r, ConversationRef::Object { ref id } if id == "conv_xyz"));
        assert_eq!(r.as_id(), "conv_xyz");
        // Object round-trips back to an object.
        assert_eq!(serde_json::to_value(&r).unwrap(), json!({"id": "conv_xyz"}));
    }

    #[test]
    fn conversation_ref_is_empty() {
        assert!(ConversationRef::Id(String::new()).is_empty());
        assert!(!ConversationRef::Id("conv_1".to_string()).is_empty());
        assert!(ConversationRef::Object { id: String::new() }.is_empty());
    }

    #[test]
    fn input_ids_deserializes_single() {
        let ids: InputIds = serde_json::from_str("[1, -2, 3]").unwrap();
        assert!(matches!(ids, InputIds::Single(ref v) if v == &[1, -2, 3]));
    }

    #[test]
    fn input_ids_deserializes_batch() {
        let ids: InputIds = serde_json::from_str("[[1, 2], [3], []]").unwrap();
        assert!(matches!(ids, InputIds::Batch(ref v) if v == &[vec![1, 2], vec![3], vec![]]));
    }

    #[test]
    fn input_ids_empty_array_is_single() {
        let ids: InputIds = serde_json::from_str("[]").unwrap();
        assert!(matches!(ids, InputIds::Single(ref v) if v.is_empty()));
    }

    #[test]
    fn input_ids_rejects_invalid_input() {
        for input in [
            "[1, [2]]",
            "[[1], 2]",
            "[\"a\"]",
            "[1.5]",
            "[5000000000]",
            "\"nope\"",
            "null",
            "7",
        ] {
            assert!(
                serde_json::from_str::<InputIds>(input).is_err(),
                "accepted {input}"
            );
        }
    }

    #[test]
    fn input_ids_round_trip() {
        for input in [json!([1, 2, 3]), json!([[1, 2], [3]]), json!([])] {
            let ids: InputIds = serde_json::from_value(input.clone()).unwrap();
            assert_eq!(serde_json::to_value(&ids).unwrap(), input);
        }
    }

    #[test]
    fn function_deserializes_without_parameters() {
        // Per the OpenAI spec, omitting `parameters` defines a function with
        // an empty parameter list: no schema was given, and a consumer that
        // needs one reads the empty schema.
        let value = json!({"name": "web_search", "description": ""});
        let function: Function = serde_json::from_value(value).expect("parameterless function");
        assert_eq!(function.parameters, None);
        assert_eq!(function.parameters_schema(), &json!({}));

        // An explicit null is the same thing: `Value::Null` is not a JSON
        // Schema and must not reach the constraint builders as one.
        let value = json!({"name": "web_search", "parameters": null});
        let function: Function = serde_json::from_value(value).expect("null parameters");
        assert_eq!(function.parameters, None);
        assert_eq!(function.parameters_schema(), &json!({}));

        // Serialized, such a function leaves the field out, as the client
        // did; a client's own `{}` stays apart from it.
        assert_eq!(
            serde_json::to_value(&function).unwrap(),
            json!({"name": "web_search"})
        );
        let empty: Function =
            serde_json::from_value(json!({"name": "web_search", "parameters": {}})).unwrap();
        assert_eq!(empty.parameters, Some(json!({})));

        // A real schema passes through untouched.
        let schema = json!({"type": "object", "properties": {"q": {"type": "string"}}});
        let value = json!({"name": "web_search", "parameters": schema});
        let function: Function = serde_json::from_value(value).expect("schema");
        assert_eq!(function.parameters.as_ref(), Some(&schema));
        assert_eq!(function.parameters_schema(), &schema);
    }

    #[test]
    fn reasoning_tokens_do_not_clear_speculative_tokens() {
        let details = Usage::from_counts(10, 20)
            .with_speculative_tokens(12, 16)
            .with_reasoning_tokens(5)
            .completion_tokens_details
            .expect("details");
        assert_eq!(details.reasoning_tokens, Some(5));
        assert_eq!(details.accepted_prediction_tokens, Some(12));
        assert_eq!(details.rejected_prediction_tokens, Some(4));
    }

    #[test]
    fn speculative_tokens_do_not_clear_reasoning_tokens() {
        let details = Usage::from_counts(10, 20)
            .with_reasoning_tokens(5)
            .with_speculative_tokens(12, 16)
            .completion_tokens_details
            .expect("details");
        assert_eq!(details.reasoning_tokens, Some(5));
        assert_eq!(details.accepted_prediction_tokens, Some(12));
        assert_eq!(details.rejected_prediction_tokens, Some(4));
    }

    #[test]
    fn zero_speculative_counts_leave_usage_untouched() {
        let usage = Usage::from_counts(10, 20).with_speculative_tokens(0, 0);
        assert!(usage.completion_tokens_details.is_none());
    }

    #[test]
    fn json_schema_shape_requires_name_and_object_schema() {
        let cases = [
            ("", json!({}), Some("json_schema_name_empty")),
            ("", json!("x"), Some("json_schema_name_empty")),
            ("w", json!("x"), Some("json_schema_schema_not_object")),
            ("w", json!(1), Some("json_schema_schema_not_object")),
            ("w", json!([]), Some("json_schema_schema_not_object")),
            ("w", json!(null), Some("json_schema_schema_not_object")),
            ("w", json!(true), Some("json_schema_schema_not_object")),
            ("w", json!({}), None),
            (
                "w",
                json!({"type": "object", "properties": {"city": {"type": "string"}}}),
                None,
            ),
        ];
        for (name, schema, expected) in cases {
            let result = validate_json_schema_shape(name, &schema);
            match expected {
                Some(code) => {
                    let err = result.expect_err(&format!("{name:?}/{schema} should fail"));
                    assert_eq!(err.code, code, "{name:?}/{schema}");
                }
                None => assert!(result.is_ok(), "{name:?}/{schema} should pass"),
            }
        }
    }

    /// A named function, `required`, and allowed tools in `required` mode
    /// force a call; `auto`, `none`, and allowed tools in `auto` mode do not.
    #[test]
    fn a_tool_choice_says_whether_it_forces_a_call() {
        let choice =
            |body: Value| -> ToolChoice { serde_json::from_value(body).expect("a tool choice") };
        for forcing in [
            json!("required"),
            json!({"type": "function", "function": {"name": "lookup"}}),
            json!({"type": "allowed_tools", "mode": "required", "tools": []}),
        ] {
            assert!(choice(forcing.clone()).forces_call(), "{forcing}");
        }
        for free in [
            json!("auto"),
            json!("none"),
            json!({"type": "allowed_tools", "mode": "auto", "tools": []}),
        ] {
            assert!(!choice(free.clone()).forces_call(), "{free}");
        }
    }

    /// `response_format` on the wire: `{"type": ...}` with the schema object
    /// under `json_schema`. The three kinds read in either key order, an
    /// unknown key is ignored, and a kind that carries no schema ignores a
    /// `json_schema` field of any shape.
    #[test]
    fn a_response_format_reads_its_three_kinds_in_either_key_order() {
        let read = |text: &str| serde_json::from_str::<ResponseFormat>(text);
        assert!(matches!(
            read(r#"{"type":"text"}"#).expect("text"),
            ResponseFormat::Text
        ));
        assert!(matches!(
            read(r#"{"type":"json_object"}"#).expect("json_object"),
            ResponseFormat::JsonObject
        ));
        let schema = json!({
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
            "additionalProperties": false
        });
        let type_first = format!(
            r#"{{"type":"json_schema","json_schema":{{"name":"answer","schema":{schema},"strict":true}}}}"#
        );
        let type_last = format!(
            r#"{{"json_schema":{{"strict":true,"schema":{schema},"name":"answer"}},"type":"json_schema"}}"#
        );
        for text in [&type_first, &type_last] {
            match read(text).expect("json_schema") {
                ResponseFormat::JsonSchema { json_schema } => {
                    assert_eq!(json_schema.name, "answer");
                    assert_eq!(json_schema.schema, schema);
                    assert_eq!(json_schema.strict, Some(true));
                }
                other => panic!("{other:?}"),
            }
        }
        let format = read(&type_first).expect("json_schema");
        assert_eq!(
            serde_json::to_value(&format).expect("serializes"),
            json!({"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": true}})
        );
        assert!(matches!(
            read(r#"{"type":"text","verbosity":"low"}"#).expect("an unknown key is ignored"),
            ResponseFormat::Text
        ));
        assert!(matches!(
            read(r#"{"type":"text","json_schema":"not an object"}"#).expect("ignored for text"),
            ResponseFormat::Text
        ));
        assert!(matches!(
            read(r#"{"json_schema":{"name":1},"type":"json_object"}"#)
                .expect("ignored for json_object"),
            ResponseFormat::JsonObject
        ));
        assert!(matches!(
            read(r#"{"json_schema":{"name":"a","name":"b"},"type":"text"}"#)
                .expect("a repeated field is ignored with the object, for text"),
            ResponseFormat::Text
        ));
        assert!(matches!(
            read(r#"{"json_schema":{"strict":true,"strict":false},"type":"json_object"}"#)
                .expect("a repeated field is ignored with the object, for json_object"),
            ResponseFormat::JsonObject
        ));
        assert!(matches!(
            read(r#"{"type":"json_schema","json_schema":{"name":"a","schema":{}},"strict":true}"#)
                .expect("a stray key beside the schema is ignored"),
            ResponseFormat::JsonSchema { .. }
        ));
        assert!(serde_json::from_str::<Option<ResponseFormat>>("null")
            .expect("null")
            .is_none());
    }

    /// The errors are the derive's, word for word: the missing tag, an
    /// unknown kind, the missing or malformed schema object, a duplicate key,
    /// the sequence form's lengths. Read on their own they carry a position
    /// where the derive gave none (it read the variant from its buffer after
    /// the object); inside a request both point at the field's end, except a
    /// malformed schema object under `json_schema`, whose column is the
    /// failing token's instead of the closing brace's (pinned last).
    #[test]
    fn a_response_format_error_names_what_is_wrong() {
        let err = |text: &str| {
            serde_json::from_str::<ResponseFormat>(text)
                .expect_err(text)
                .to_string()
        };
        for (text, message) in [
            (
                r#"{"json_schema":{"name":"a","schema":{}}}"#,
                "missing field `type` at line 1 column 40",
            ),
            ("{}", "missing field `type` at line 1 column 2"),
            (
                r#"{"type":"xml"}"#,
                "unknown variant `xml`, expected one of `text`, `json_object`, `json_schema` at line 1 column 13",
            ),
            (
                r#"{"type":1}"#,
                "invalid type: integer `1`, expected variant identifier at line 1 column 9",
            ),
            (
                r#"{"type":"json_schema"}"#,
                "missing field `json_schema` at line 1 column 22",
            ),
            (
                r#"{"type":"json_schema","json_schema":null}"#,
                "invalid type: null, expected struct JsonSchemaFormat at line 1 column 40",
            ),
            (
                r#"{"type":"json_schema","json_schema":"text"}"#,
                "invalid type: string \"text\", expected struct JsonSchemaFormat at line 1 column 42",
            ),
            (
                r#"{"json_schema":"text","type":"json_schema"}"#,
                "invalid type: string \"text\", expected struct JsonSchemaFormat at line 1 column 43",
            ),
            (
                r#"{"type":"json_schema","json_schema":{"schema":{}}}"#,
                "missing field `name` at line 1 column 49",
            ),
            (
                r#"{"type":"json_schema","json_schema":{"name":"a"}}"#,
                "missing field `schema` at line 1 column 48",
            ),
            (
                r#"{"type":"text","type":"text"}"#,
                "duplicate field `type` at line 1 column 21",
            ),
            (
                r#"{"type":"json_schema","json_schema":{"name":"a","schema":{}},"json_schema":{"name":"b","schema":{}}}"#,
                "duplicate field `json_schema` at line 1 column 100",
            ),
            (
                r#""text""#,
                "invalid type: string \"text\", expected internally tagged enum ResponseFormat at line 1 column 6",
            ),
            ("[]", "missing field `type` at line 1 column 2"),
            (
                r#"["xml"]"#,
                "unknown variant `xml`, expected one of `text`, `json_object`, `json_schema` at line 1 column 6",
            ),
            (
                "[1]",
                "invalid type: integer `1`, expected variant identifier at line 1 column 2",
            ),
            (
                r#"["json_schema"]"#,
                "invalid length 0, expected struct variant ResponseFormat::JsonSchema with 1 element at line 1 column 15",
            ),
            (
                r#"["json_schema","text"]"#,
                "invalid type: string \"text\", expected struct JsonSchemaFormat at line 1 column 21",
            ),
            (
                r#"["json_schema",{"name":"a","schema":{}},"extra"]"#,
                "invalid length 2, expected 1 element in sequence at line 1 column 48",
            ),
            (
                r#"["text",{"json_schema":"x"}]"#,
                "invalid length 1, expected 0 elements in sequence at line 1 column 28",
            ),
            (
                r#"["json_object","x","y"]"#,
                "invalid length 2, expected 0 elements in sequence at line 1 column 23",
            ),
        ] {
            assert_eq!(err(text), message, "{text}");
        }
        // the sequence form the derive read is still read
        assert!(matches!(
            serde_json::from_str::<ResponseFormat>(r#"["text"]"#).expect("text"),
            ResponseFormat::Text
        ));
        match serde_json::from_str::<ResponseFormat>(
            r#"["json_schema",["a",{"type":"object"},true]]"#,
        )
        .expect("json_schema")
        {
            ResponseFormat::JsonSchema { json_schema } => {
                assert_eq!(json_schema.name, "a");
                assert_eq!(json_schema.strict, Some(true));
            }
            other => panic!("{other:?}"),
        }
        // inside a request: the field's end, and the one moved column
        let request_err = |format: &str| {
            let text = format!(
                r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"response_format":{format},"max_tokens":8}}"#
            );
            serde_json::from_str::<crate::chat::ChatCompletionRequest>(&text)
                .expect_err(format)
                .to_string()
        };
        assert_eq!(
            request_err(r#"{"json_schema":{"name":"a","schema":{}},"type":"xml"}"#),
            "unknown variant `xml`, expected one of `text`, `json_object`, `json_schema` at line 1 column 127"
        );
        assert_eq!(
            request_err(r#"{"type":"json_schema","json_schema":"text"}"#),
            "invalid type: string \"text\", expected struct JsonSchemaFormat at line 1 column 117"
        );
    }

    /// One value of each kind writes its name under `type` and reads back as
    /// the same kind, from a `Value` and from text; `kind_of` matches every
    /// variant of `ResponseFormat`, so a new variant fails this build until
    /// the reader learns it, and the unknown-kind list is the names in order.
    #[test]
    fn a_response_format_round_trips_each_kind_under_its_name() {
        fn kind_of(format: &ResponseFormat) -> ResponseFormatKind {
            match format {
                ResponseFormat::Text => ResponseFormatKind::Text,
                ResponseFormat::JsonObject => ResponseFormatKind::JsonObject,
                ResponseFormat::JsonSchema { .. } => ResponseFormatKind::JsonSchema,
            }
        }
        let formats = [
            ResponseFormat::Text,
            ResponseFormat::JsonObject,
            ResponseFormat::JsonSchema {
                json_schema: JsonSchemaFormat {
                    name: "answer".to_owned(),
                    schema: json!({"type": "object"}),
                    strict: None,
                },
            },
        ];
        assert_eq!(formats.len(), ResponseFormatKind::ALL.len());
        for (format, kind) in formats.iter().zip(ResponseFormatKind::ALL) {
            assert_eq!(kind_of(format), kind);
            let written = serde_json::to_value(format).expect("writes");
            assert_eq!(written["type"], kind.name(), "{written}");
            let read: ResponseFormat = serde_json::from_value(written.clone()).expect("reads back");
            assert_eq!(kind_of(&read), kind, "{written}");
            let read: ResponseFormat =
                serde_json::from_str(&written.to_string()).expect("reads back from text");
            assert_eq!(kind_of(&read), kind, "{written}");
        }
        assert_eq!(
            RESPONSE_FORMAT_KINDS,
            ResponseFormatKind::ALL.map(ResponseFormatKind::name)
        );
    }

    /// A schema object that arrives before `type` is kept as it arrived and
    /// read by the schema's own derive once the kind wants it: a repeated
    /// field is the duplicate-field error, a field malformed before it repeats
    /// the invalid-type error, in that order as in the other (the position
    /// then being the enclosing object's end, the derive's); a duplicate
    /// deeper in the schema keeps the last value in both orders, as a `Value`
    /// does.
    #[test]
    fn a_kept_schema_object_refuses_a_repeated_field_in_either_order() {
        let err = |text: &str| {
            serde_json::from_str::<ResponseFormat>(text)
                .expect_err(text)
                .to_string()
        };
        for (text, message) in [
            (
                r#"{"type":"json_schema","json_schema":{"name":"a","name":"b","schema":{}}}"#,
                "duplicate field `name` at line 1 column 54",
            ),
            (
                r#"{"json_schema":{"name":"a","name":"b","schema":{}},"type":"json_schema"}"#,
                "duplicate field `name` at line 1 column 72",
            ),
            (
                r#"{"type":"json_schema","json_schema":{"name":"a","schema":{},"schema":{}}}"#,
                "duplicate field `schema` at line 1 column 68",
            ),
            (
                r#"{"json_schema":{"name":"a","schema":{},"schema":{}},"type":"json_schema"}"#,
                "duplicate field `schema` at line 1 column 73",
            ),
            (
                r#"{"type":"json_schema","json_schema":{"name":"a","schema":{},"strict":true,"strict":false}}"#,
                "duplicate field `strict` at line 1 column 82",
            ),
            (
                r#"{"json_schema":{"name":"a","schema":{},"strict":true,"strict":false},"type":"json_schema"}"#,
                "duplicate field `strict` at line 1 column 90",
            ),
            (
                r#"{"type":"json_schema","json_schema":{"name":1,"name":"b","schema":{}}}"#,
                "invalid type: integer `1`, expected a string at line 1 column 45",
            ),
            (
                r#"{"json_schema":{"name":1,"name":"b","schema":{}},"type":"json_schema"}"#,
                "invalid type: integer `1`, expected a string at line 1 column 70",
            ),
            // the field repeated at the top level: the first value is read
            // before the repeat is raised, as the derive's visitor does
            (
                r#"{"json_schema":{"name":1},"json_schema":{"name":"a","schema":{}},"type":"json_schema"}"#,
                "invalid type: integer `1`, expected a string at line 1 column 86",
            ),
            (
                r#"{"json_schema":5,"json_schema":{},"type":"json_schema"}"#,
                "invalid type: integer `5`, expected struct JsonSchemaFormat at line 1 column 55",
            ),
            (
                r#"{"json_schema":{"name":"a","schema":{}},"json_schema":{},"type":"json_schema"}"#,
                "duplicate field `json_schema` at line 1 column 78",
            ),
            // the eager read's one difference in text: a repeated `type` after
            // a malformed schema reads the schema's error here and
            // `duplicate field `type`` under the derive; both refuse the body
            (
                r#"{"type":"json_schema","json_schema":{"name":1},"type":"text"}"#,
                "invalid type: integer `1`, expected a string at line 1 column 45",
            ),
        ] {
            assert_eq!(err(text), message, "{text}");
        }
        for text in [
            r#"{"type":"json_schema","json_schema":{"name":"a","schema":{"x":1,"x":2}}}"#,
            r#"{"json_schema":{"name":"a","schema":{"x":1,"x":2}},"type":"json_schema"}"#,
        ] {
            match serde_json::from_str::<ResponseFormat>(text).expect(text) {
                ResponseFormat::JsonSchema { json_schema } => {
                    assert_eq!(json_schema.schema, json!({"x": 2}), "{text}");
                }
                other => panic!("{other:?}"),
            }
        }
        // a kept object with a repeated unknown key is as the derive read it:
        // the key ignored, no error
        assert!(matches!(
            serde_json::from_str::<ResponseFormat>(
                r#"{"json_schema":{"name":"a","schema":{},"note":1,"note":2},"type":"json_schema"}"#
            )
            .expect("an unknown key twice is ignored"),
            ResponseFormat::JsonSchema { .. }
        ));
    }
}
