use std::{collections::HashMap, fmt};

use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::Value;
use validator::Validate;

use super::common::{default_true, GenerationRequest, StringOrArray, UsageInfo};

fn default_rerank_object() -> String {
    "rerank".to_string()
}

fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs() as i64
}

// ============================================================================
// Rerank API
// ============================================================================

/// A request to the rerank route: the query, the documents to rank against it,
/// the model, how many of the best to return and whether to return the
/// documents' text.
#[derive(Debug, Clone, Deserialize, Serialize, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_rerank_request"))]
pub struct RerankRequest {
    /// The query text to rank documents against
    #[validate(custom(function = "validate_query"))]
    pub query: String,

    /// List of documents to be ranked
    #[validate(custom(function = "validate_documents"))]
    pub documents: Vec<String>,

    /// Model to use for reranking. The gateway's wildcard (`unknown`, a
    /// `/v1/rerank` body that named none) selects any worker and is not
    /// written to the body forwarded to it, so a single-model server applies
    /// its own model instead of refusing a name it does not serve.
    #[serde(skip_serializing_if = "super::common::is_unknown_model")]
    pub model: String,

    /// Maximum number of documents to return (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub top_k: Option<usize>,

    /// Whether to return documents in addition to scores
    #[serde(default = "default_true")]
    pub return_documents: bool,

    // Engine-specific extensions
    /// Request ID for tracking; absent from the body when unset, so the
    /// forwarded bytes carry no `rid` key to strip.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rid: Option<StringOrArray>,

    /// User identifier; written to the forwarded body only when given, as
    /// `rid` is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl GenerationRequest for RerankRequest {
    fn rid(&self) -> Option<&str> {
        self.rid.as_ref().and_then(StringOrArray::first)
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn is_stream(&self) -> bool {
        false // Reranking doesn't support streaming
    }

    fn extract_text_for_routing(&self) -> String {
        self.query.clone()
    }
}

impl super::validated::Normalizable for RerankRequest {
    // Use default no-op normalization
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validates that the query is not empty
fn validate_query(query: &str) -> Result<(), validator::ValidationError> {
    if query.trim().is_empty() {
        return Err(validator::ValidationError::new("query cannot be empty"));
    }
    Ok(())
}

/// Validates that the documents list is not empty
fn validate_documents(documents: &[String]) -> Result<(), validator::ValidationError> {
    if documents.is_empty() {
        return Err(validator::ValidationError::new(
            "documents list cannot be empty",
        ));
    }
    Ok(())
}

/// Schema-level validation for cross-field dependencies
#[expect(
    clippy::unnecessary_wraps,
    reason = "validator crate requires Result return type"
)]
fn validate_rerank_request(req: &RerankRequest) -> Result<(), validator::ValidationError> {
    // Validate top_k if specified
    if let Some(k) = req.top_k {
        if k > req.documents.len() {
            tracing::warn!(
                "top_k ({}) is greater than number of documents ({})",
                k,
                req.documents.len()
            );
        }
    }
    Ok(())
}

impl RerankRequest {
    /// Get the effective top_k value
    pub fn effective_top_k(&self) -> usize {
        self.top_k.unwrap_or(self.documents.len())
    }
}

/// Individual rerank result
///
/// Written in the gateway's own spelling; read from a worker's answer in
/// every spelling the rerank contracts write: `relevance_score` for the
/// score and `text` for the document (the Cohere-style and the
/// embeddings-server contracts), and a document given as the `{text}`
/// object.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RerankResult {
    /// Relevance score for the document
    #[serde(alias = "relevance_score")]
    pub score: f32,

    /// The document text (if return_documents was true)
    #[serde(default, alias = "text", deserialize_with = "deserialize_document")]
    pub document: Option<String>,

    /// Original index of the document in the request
    pub index: usize,

    /// Additional metadata about the ranking
    pub meta_info: Option<HashMap<String, Value>>,
}

/// A result's document as the contracts write it: the text itself, the
/// `{text}` object of the Cohere-style contract, or nothing. Read by a
/// visitor rather than an untagged enum, so a wrong shape is refused in
/// serde's own words and nothing is buffered first.
fn deserialize_document<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct DocumentVisitor;

    impl<'de> de::Visitor<'de> for DocumentVisitor {
        type Value = Option<String>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a document as text, as an object with `text`, or null")
        }

        fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
            Ok(Some(text.to_owned()))
        }

        fn visit_string<E: de::Error>(self, text: String) -> Result<Self::Value, E> {
            Ok(Some(text))
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_some<D2: Deserializer<'de>>(
            self,
            deserializer: D2,
        ) -> Result<Self::Value, D2::Error> {
            deserializer.deserialize_any(DocumentVisitor)
        }

        fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut text = None;
            while let Some(key) = map.next_key::<String>()? {
                if key == "text" {
                    text = map.next_value::<Option<String>>()?;
                } else {
                    map.next_value::<de::IgnoredAny>()?;
                }
            }
            Ok(text)
        }
    }

    deserializer.deserialize_any(DocumentVisitor)
}

/// A rerank worker's answer: the ranked results as the bare list the gateway
/// itself answers with (the embeddings-server contract writes the same list,
/// `text` for the document), or under `results` as the Cohere-style object
/// of the OpenAI-compatible rerank servers (`relevance_score`, the document
/// as text or as `{text}`), whose `usage` and `id` are read with them (its
/// `model` is not: the gateway names the model that ran). One `from_slice`
/// reads every shape; the results are in the worker's order. Read by a
/// visitor on the first token rather than an untagged enum, so a body in no
/// rerank shape is refused in serde's own words (the field a new server's
/// spelling lacks, not "no variant matched") and the body streams through
/// without being buffered first.
#[derive(Debug)]
pub struct RerankWorkerResponse {
    /// The ranked results, in the worker's order.
    pub results: Vec<RerankResult>,
    /// The usage the object form reports, in the rerank servers' spelling
    /// (`prompt_tokens` and `total_tokens`; no `completion_tokens`, which
    /// reads as none generated); a usage in another spelling is left out,
    /// the results read as before; the bare list has none.
    pub usage: Option<UsageInfo>,
    /// The id the object form reports, a string or a list of strings; an id
    /// in another shape is left out, the results read as before; the bare
    /// list has none.
    pub id: Option<StringOrArray>,
}

impl<'de> Deserialize<'de> for RerankWorkerResponse {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ResponseVisitor;

        impl<'de> de::Visitor<'de> for ResponseVisitor {
            type Value = RerankWorkerResponse;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a list of rerank results or an object with `results`")
            }

            fn visit_seq<A: de::SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
                Vec::deserialize(de::value::SeqAccessDeserializer::new(seq)).map(|results| {
                    RerankWorkerResponse {
                        results,
                        usage: None,
                        id: None,
                    }
                })
            }

            fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut results = None;
                let mut usage = None;
                let mut id = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "results" => results = Some(map.next_value()?),
                        "usage" => {
                            usage = map
                                .next_value::<Option<Value>>()?
                                .and_then(|usage| serde_json::from_value::<UsageInfo>(usage).ok());
                        }
                        "id" => {
                            id = map
                                .next_value::<Option<Value>>()?
                                .and_then(|id| serde_json::from_value::<StringOrArray>(id).ok());
                        }
                        _ => {
                            map.next_value::<de::IgnoredAny>()?;
                        }
                    }
                }
                results
                    .map(|results| RerankWorkerResponse { results, usage, id })
                    .ok_or_else(|| de::Error::missing_field("results"))
            }
        }

        deserializer.deserialize_any(ResponseVisitor)
    }
}

/// Rerank response containing sorted results
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RerankResponse {
    /// Ranked results sorted by score (highest first)
    pub results: Vec<RerankResult>,

    /// Model used for reranking
    pub model: String,

    /// Usage information
    pub usage: Option<UsageInfo>,

    /// Response object type
    #[serde(default = "default_rerank_object")]
    pub object: String,

    /// Response ID
    pub id: Option<StringOrArray>,

    /// Creation timestamp
    pub created: i64,
}

impl RerankResponse {
    /// A response of `results` sorted highest score first, whatever order
    /// they came in (every rerank contract promises that order; a stable
    /// sort keeps the worker's order among equal scores), with the model
    /// that ran, the usage the worker reported and the answer's `id` (the
    /// client's `rid`, or the worker's id).
    pub fn new(
        mut results: Vec<RerankResult>,
        model: String,
        usage: Option<UsageInfo>,
        id: Option<StringOrArray>,
    ) -> Self {
        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        RerankResponse {
            results,
            model,
            usage,
            object: default_rerank_object(),
            id,
            created: current_timestamp(),
        }
    }

    /// Keep the `k` best results: the list is sorted highest score first.
    pub fn apply_top_k(&mut self, k: usize) {
        self.results.truncate(k);
    }

    /// Drop documents from results (when return_documents is false)
    pub fn drop_documents(&mut self) {
        for result in &mut self.results {
            result.document = None;
        }
    }
}

/// The body of `/v1/rerank`: the request of the OpenAI-compatible rerank
/// servers, `query` and `documents` with an optional `model`, `top_n` and
/// `return_documents`. A body without `model` is routed as the unknown
/// model, to any worker, the shape a single-model server's client sends;
/// the native `/rerank` route takes [`RerankRequest`], whose `model` is
/// required. `query` and `documents` are refused empty, by the rules of the
/// native route; `top_n` has no bound, because 0 means all of them here.
#[derive(Debug, Clone, Serialize, Deserialize, Validate, schemars::JsonSchema)]
pub struct V1RerankReqInput {
    #[validate(custom(function = "validate_query"))]
    pub query: String,
    #[validate(custom(function = "validate_documents"))]
    pub documents: Vec<String>,

    /// The model to rerank with, which selects the worker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Maximum number of documents to return; absent and 0 mean all of them,
    /// as the OpenAI-compatible servers read it. `top_k`, the native
    /// spelling, is read as well.
    #[serde(default, alias = "top_k", skip_serializing_if = "Option::is_none")]
    pub top_n: Option<usize>,

    /// Whether the documents are returned with their scores (the default).
    #[serde(default = "default_true")]
    pub return_documents: bool,

    /// Request ID for tracking, as on the native route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rid: Option<StringOrArray>,

    /// User identifier, as on the native route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl super::validated::Normalizable for V1RerankReqInput {}

/// Convert V1RerankReqInput to RerankRequest
impl From<V1RerankReqInput> for RerankRequest {
    fn from(v1: V1RerankReqInput) -> Self {
        RerankRequest {
            query: v1.query,
            documents: v1.documents,
            model: v1
                .model
                .unwrap_or_else(super::common::default_unknown_model),
            top_k: v1.top_n.filter(|n| *n > 0),
            return_documents: v1.return_documents,
            rid: v1.rid,
            user: v1.user,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A request without a `rid` writes no `rid` key, so the proxy's strip
    /// of the engine defaults finds nothing to drop and forwards the
    /// serialized bytes as they are; a set `rid` is written.
    #[test]
    fn a_rerank_request_without_rid_writes_no_rid_key() {
        let request: RerankRequest =
            serde_json::from_value(json!({"model": "m", "query": "q", "documents": ["d"]}))
                .expect("a well-formed request");
        let body = serde_json::to_value(&request).expect("a body");
        assert!(body.get("rid").is_none(), "{body}");
        let with_rid: RerankRequest = serde_json::from_value(
            json!({"model": "m", "query": "q", "documents": ["d"], "rid": "r-1"}),
        )
        .expect("a well-formed request");
        assert_eq!(
            serde_json::to_value(&with_rid).expect("a body")["rid"],
            "r-1"
        );
    }

    /// A request without a `user` writes no `user` key, as one without a
    /// `rid` writes no `rid` key; a set `user` is written.
    #[test]
    fn a_rerank_request_without_user_writes_no_user_key() {
        let request: RerankRequest =
            serde_json::from_value(json!({"model": "m", "query": "q", "documents": ["d"]}))
                .expect("a well-formed request");
        let body = serde_json::to_value(&request).expect("a body");
        assert!(body.get("user").is_none(), "{body}");
        let with_user: RerankRequest = serde_json::from_value(
            json!({"model": "m", "query": "q", "documents": ["d"], "user": "u-1"}),
        )
        .expect("a well-formed request");
        assert_eq!(
            serde_json::to_value(&with_user).expect("a body")["user"],
            "u-1"
        );
    }

    fn result(index: usize, score: f32) -> RerankResult {
        RerankResult {
            score,
            document: None,
            index,
            meta_info: None,
        }
    }

    /// The object form's `usage` (in the rerank servers' spelling, without
    /// `completion_tokens`) and `id` are read with the results; the bare list
    /// has none.
    #[test]
    fn the_worker_s_usage_and_id_are_read_from_the_object_form() {
        let object: RerankWorkerResponse = serde_json::from_str(
            r#"{"id":"rerank-up-1","model":"m","results":[{"index":0,"relevance_score":0.5}],"usage":{"prompt_tokens":11,"total_tokens":11},"meta":{"billed_units":{"search_units":1}}}"#,
        )
        .expect("the object form reads");
        assert_eq!(object.results.len(), 1);
        let usage = object.usage.expect("the object's usage is read");
        assert_eq!(
            (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens
            ),
            (11, 0, 11)
        );
        assert_eq!(
            object.id,
            Some(StringOrArray::String("rerank-up-1".to_string()))
        );
        let list: RerankWorkerResponse =
            serde_json::from_str(r#"[{"index":0,"score":0.5}]"#).expect("the list form reads");
        assert!(list.usage.is_none() && list.id.is_none());
    }

    /// A usage in another spelling (no `prompt_tokens`) is left out and the
    /// results still read, as they did before the usage was read at all.
    #[test]
    fn a_usage_in_another_spelling_is_left_out_and_the_results_still_read() {
        let object: RerankWorkerResponse = serde_json::from_str(
            r#"{"results":[{"index":0,"relevance_score":0.5}],"usage":{"total_tokens":9}}"#,
        )
        .expect("the results read");
        assert_eq!(object.results.len(), 1);
        assert!(
            object.usage.is_none(),
            "a usage in another spelling is left out"
        );
    }

    /// An `id` in another shape (a number, an object) is left out and the
    /// results still read, as they did before the id was read at all.
    #[test]
    fn an_id_in_another_shape_is_left_out_and_the_results_still_read() {
        for body in [
            r#"{"id":42,"results":[{"index":0,"relevance_score":0.5}]}"#,
            r#"{"id":{"req":"x"},"results":[{"index":0,"relevance_score":0.5}]}"#,
        ] {
            let object: RerankWorkerResponse =
                serde_json::from_str(body).expect("the results read");
            assert_eq!(object.results.len(), 1, "{body}");
            assert!(
                object.id.is_none(),
                "an id in another shape is left out: {body}"
            );
        }
    }

    /// The response's results are sorted highest score first whatever order
    /// they came in, the worker's order kept among equal scores, and `top_k`
    /// keeps the best of them.
    #[test]
    fn a_response_sorts_by_score_before_the_cut() {
        let mut response = RerankResponse::new(
            vec![result(0, 0.1), result(1, 0.9), result(2, 0.9)],
            "m".to_string(),
            None,
            None,
        );
        let order: Vec<usize> = response.results.iter().map(|r| r.index).collect();
        assert_eq!(order, [1, 2, 0]);
        response.apply_top_k(1);
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].index, 1);
    }
}
