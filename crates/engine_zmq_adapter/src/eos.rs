//! EOS stop ids for the tokenizer-less EngineCore: resolved from the model
//! directory at connect time, backstopped from the tokenizer per request.

use std::{path::Path, sync::Arc};

use llm_tokenizer::traits::Tokenizer;
use smg_grpc_client::vllm_proto as vllm;

/// The model's EOS stop set, resolved from its local directory. EngineCore
/// has no tokenizer or model config — stopping at EOS is the frontend's job
/// (the ids ride each request), and without them generation only ends at
/// `max_tokens`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EosTokenIds {
    /// Primary EOS id, carried as the request's `_eos_token_id`.
    pub(crate) primary: Option<u32>,
    /// Extra EOS ids (multi-EOS models), merged into `stop_token_ids`.
    pub(crate) extra: Vec<u32>,
}

impl EosTokenIds {
    pub fn new(primary: Option<u32>, extra: Vec<u32>) -> Self {
        Self { primary, extra }
    }

    /// Build from the tokenizer's merged EOS set (same ordering as
    /// [`Self::from_model_dir`]: config first, generation config after).
    pub fn from_ids(ids: &[u32]) -> Self {
        let mut ids = ids.iter().copied();
        Self {
            primary: ids.next(),
            extra: ids.collect(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.primary.is_none() && self.extra.is_empty()
    }

    /// Resolve from `config.json` + `generation_config.json` in a local model
    /// directory: primary = the model config's first id, extras = every other
    /// listed id. Missing files or fields degrade to fewer ids.
    pub async fn from_model_dir(dir: &Path) -> Self {
        let model_ids = eos_ids_from_file(&dir.join("config.json")).await;
        let gen_ids = eos_ids_from_file(&dir.join("generation_config.json")).await;
        let primary = (model_ids.first().or_else(|| gen_ids.first())).copied();
        let mut extra = Vec::new();
        for id in model_ids.into_iter().chain(gen_ids) {
            if Some(id) != primary && !extra.contains(&id) {
                extra.push(id);
            }
        }
        Self { primary, extra }
    }
}

/// Read a config file's `eos_token_id`, which is a single id or a list.
///
/// A missing file is expected (a model ships `config.json`,
/// `generation_config.json`, or both), so read errors stay silent. A file
/// that exists but holds corrupt JSON is worth a `warn!`: it runs once at
/// connect time, and losing the EOS ids here silently manifests later as
/// generation running to `max_tokens`.
pub(crate) async fn eos_ids_from_file(path: &Path) -> Vec<u32> {
    let Ok(text) = tokio::fs::read_to_string(path).await else {
        return Vec::new();
    };
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(config) => eos_ids_from_value(config.get("eos_token_id")),
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "failed to parse model config for EOS ids");
            Vec::new()
        }
    }
}

pub(crate) fn eos_ids_from_value(value: Option<&serde_json::Value>) -> Vec<u32> {
    let as_id = |v: &serde_json::Value| v.as_u64().and_then(|id| u32::try_from(id).ok());
    match value {
        Some(serde_json::Value::Array(ids)) => ids.iter().filter_map(as_id).collect(),
        Some(id) => as_id(id).into_iter().collect(),
        None => Vec::new(),
    }
}

/// Request-time EOS backstop for the tokenizer-less EngineCore.
///
/// EOS injection has exactly one owner — this file. The connect-time
/// [`EosTokenIds`] model-dir resolution has nothing to read when the worker's
/// model id is a repo id rather than a local path, so the tokenizer's merged
/// EOS set is folded into `stop_token_ids` here as the always-available
/// backstop; without it an uncapped request generates to the full context
/// window. Not needed for TokenSpeed (its scheduler stops at EOS itself),
/// which is why this takes the vLLM request: a consumer that holds a
/// per-engine request wrapper dispatches on the variant before calling.
pub fn fold_tokenizer_eos_backstop(
    req: &mut vllm::GenerateRequest,
    tokenizer: Option<&Arc<dyn Tokenizer>>,
) {
    let Some(params) = req.sampling_params.as_mut() else {
        return;
    };
    if params.ignore_eos {
        return;
    }
    if let Some(tokenizer) = tokenizer {
        for &id in tokenizer.eos_token_ids() {
            if !params.stop_token_ids.contains(&id) {
                params.stop_token_ids.push(id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer};

    use super::*;

    fn eos_request(stop_token_ids: Vec<u32>, ignore_eos: bool) -> vllm::GenerateRequest {
        vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams {
                stop_token_ids,
                ignore_eos,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn eos_stop_ids(req: &vllm::GenerateRequest) -> &[u32] {
        &req.sampling_params.as_ref().unwrap().stop_token_ids
    }

    #[test]
    fn eos_backstop_appends_tokenizer_ids_without_duplicates() {
        // MockTokenizer's EOS set is {999}.
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());

        let mut req = eos_request(vec![7], false);
        fold_tokenizer_eos_backstop(&mut req, Some(&tokenizer));
        assert_eq!(eos_stop_ids(&req), &[7, 999]);

        // Already-present EOS ids are not duplicated.
        let mut req = eos_request(vec![999], false);
        fold_tokenizer_eos_backstop(&mut req, Some(&tokenizer));
        assert_eq!(eos_stop_ids(&req), &[999]);
    }

    #[test]
    fn eos_backstop_respects_ignore_eos() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let mut req = eos_request(vec![7], true);
        fold_tokenizer_eos_backstop(&mut req, Some(&tokenizer));
        assert_eq!(eos_stop_ids(&req), &[7]);
    }

    #[tokio::test]
    async fn eos_token_ids_resolve_from_model_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.json"), r#"{"eos_token_id": 5}"#).unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"eos_token_id": [5, 7, 9]}"#,
        )
        .unwrap();
        assert_eq!(
            EosTokenIds::from_model_dir(dir.path()).await,
            EosTokenIds::new(Some(5), vec![7, 9]),
        );

        // Missing files degrade to no ids, not an error.
        let empty = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            EosTokenIds::from_model_dir(empty.path()).await,
            EosTokenIds::default(),
        );
    }
}
