use std::{collections::BTreeSet, path::Path};

use serde_json::Value;

use crate::traits::TokenIdType;

fn collect_eos_ids(cfg: &Value, ids: &mut BTreeSet<TokenIdType>) {
    match cfg.get("eos_token_id") {
        Some(Value::Number(n)) => {
            if let Some(id) = n.as_u64() {
                ids.insert(id as TokenIdType);
            }
        }
        Some(Value::Array(arr)) => {
            for v in arr {
                if let Some(id) = v.as_u64() {
                    ids.insert(id as TokenIdType);
                }
            }
        }
        _ => {}
    }
}

/// Load and merge EOS token IDs from `config.json` and `generation_config.json`.
///
/// Models may define different EOS tokens in each file (e.g. Kimi-K2.5 uses
/// `[EOS]` (163585) in config.json and `<|im_end|>` (163586) in
/// generation_config.json). We merge both into a deduplicated, sorted list so
/// the StopDecoder can strip any of them before decoding.
///
/// This matches how vllm and sglang resolve EOS:
/// - vllm: `hf_config.eos_token_id` (from config.json via AutoConfig) +
///   `generation_config.eos_token_id` (merged in `update_from_generation_config`)
/// - sglang: `model_info["eos_token_ids"]` (from model config, includes both sources)
pub fn load_eos_token_ids(dir: &Path) -> Vec<TokenIdType> {
    let mut ids = BTreeSet::new();

    for filename in ["config.json", "generation_config.json"] {
        if let Ok(content) = std::fs::read_to_string(dir.join(filename)) {
            if let Ok(cfg) = serde_json::from_str::<Value>(&content) {
                collect_eos_ids(&cfg, &mut ids);
            }
        }
    }

    ids.into_iter().collect()
}

/// Add the tokenizer's own `eos_token` to the EOS set.
///
/// `config.json` and `generation_config.json` name the id generation stops on,
/// while `tokenizer_config.json` names the tokenizer's `eos_token`, and the
/// two can differ (Kimi K3: `<|im_end|>` 163586 vs `[EOS]` 163585). Engine
/// grammars for structured outputs terminate on the tokenizer's eos, so the
/// stop decoder must know it too, or it is decoded as literal text after the
/// JSON. Keeps the list sorted and deduplicated.
pub fn with_tokenizer_eos(
    mut ids: Vec<TokenIdType>,
    eos_id: Option<TokenIdType>,
) -> Vec<TokenIdType> {
    if let Some(id) = eos_id {
        if !ids.contains(&id) {
            ids.push(id);
            ids.sort_unstable();
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_single_int() {
        let cfg: Value = serde_json::from_str(r#"{"eos_token_id": 42}"#).unwrap();
        let mut ids = BTreeSet::new();
        collect_eos_ids(&cfg, &mut ids);
        assert_eq!(ids.into_iter().collect::<Vec<_>>(), vec![42]);
    }

    #[test]
    fn test_collect_array() {
        let cfg: Value = serde_json::from_str(r#"{"eos_token_id": [10, 20, 30]}"#).unwrap();
        let mut ids = BTreeSet::new();
        collect_eos_ids(&cfg, &mut ids);
        assert_eq!(ids.into_iter().collect::<Vec<_>>(), vec![10, 20, 30]);
    }

    #[test]
    fn test_collect_missing_field() {
        let cfg: Value = serde_json::from_str(r#"{"model_type": "llama"}"#).unwrap();
        let mut ids = BTreeSet::new();
        collect_eos_ids(&cfg, &mut ids);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_merge_deduplicates() {
        let mut ids = BTreeSet::new();
        let cfg1: Value = serde_json::from_str(r#"{"eos_token_id": 100}"#).unwrap();
        let cfg2: Value = serde_json::from_str(r#"{"eos_token_id": [100, 200]}"#).unwrap();
        collect_eos_ids(&cfg1, &mut ids);
        collect_eos_ids(&cfg2, &mut ids);
        assert_eq!(ids.into_iter().collect::<Vec<_>>(), vec![100, 200]);
    }

    #[test]
    fn test_load_from_nonexistent_dir() {
        let ids = load_eos_token_ids(Path::new("/nonexistent/path"));
        assert!(ids.is_empty());
    }

    #[test]
    fn test_with_tokenizer_eos_adds_missing_id_sorted() {
        // Kimi K3: config files say <|im_end|>, the tokenizer's eos is [EOS].
        assert_eq!(
            with_tokenizer_eos(vec![163586], Some(163585)),
            vec![163585, 163586]
        );
    }

    #[test]
    fn test_with_tokenizer_eos_is_idempotent_and_optional() {
        assert_eq!(with_tokenizer_eos(vec![2, 7], Some(7)), vec![2, 7]);
        assert_eq!(with_tokenizer_eos(vec![2, 7], None), vec![2, 7]);
        assert_eq!(with_tokenizer_eos(Vec::new(), Some(1)), vec![1]);
    }
}
