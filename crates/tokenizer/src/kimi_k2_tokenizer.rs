//! Kimi-K2 / K2.5 / K2.6 detection and special-token helpers.
//!
//! Kimi models use the standard tiktoken BPE engine but with a Han-aware regex
//! and a 256-slot reserved-special-token range starting at `len(mergeable_ranks)`.
//! These helpers let the generic `TiktokenTokenizer` loader specialize itself
//! when it sees a Kimi directory, without exposing a separate public type.
//!
//! Upstream reference (identical across all three Kimi variants):
//!   - moonshotai/Kimi-K2-Thinking/tokenization_kimi.py
//!   - moonshotai/Kimi-K2.5/tokenization_kimi.py
//!   - moonshotai/Kimi-K2.6/tokenization_kimi.py

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use serde_json::Value;

use crate::traits::TokenIdType;

const NUM_RESERVED_SPECIAL_TOKENS: usize = 256;

/// Han-aware tokenization regex used by Kimi K2/K2.5/K2.6. Byte-identical to
/// the `pat_str` in upstream `tokenization_kimi.py`.
pub(crate) const KIMI_K2_PATTERN: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Returns true if `dir` looks like a Kimi K2/K2.5/K2.6 model directory.
///
/// Primary signal: the already-parsed `tokenizer_config.json` references
/// `tokenization_kimi` (via `auto_map`, `tokenizer_class`, etc.). Callers pass
/// the parsed JSON so we don't re-read the file the tiktoken loader already
/// parsed. Fallback: read sibling `config.json` and check `model_type` ∈
/// `{kimi_k2, kimi_k25, kimi_k3}`.
pub(crate) fn matches(tokenizer_config: Option<&Value>, dir: &Path) -> bool {
    if tokenizer_config.is_some_and(value_mentions_kimi_tokenizer) {
        return true;
    }
    read_json(&dir.join("config.json")).is_some_and(|config| model_config_is_kimi(&config))
}

/// Fill the 256-slot reserved special-token range starting at `base_vocab_size`
/// with synthetic `<|reserved_token_{id}|>` entries, preserving any explicit
/// `added_tokens_decoder` entries that already occupy slots in that range.
///
/// Mirrors upstream `tokenization_kimi.py`:
/// ```python
/// {special_tokens_mapping.get(i, f"<|reserved_token_{i}|>"): i
///  for i in range(num_base_tokens, num_base_tokens + 256)}
/// ```
/// where `num_base_tokens = len(mergeable_ranks)` — i.e., `encoder.len()`.
pub(crate) fn apply_reserved_special_tokens(
    added_tokens: &mut HashMap<String, TokenIdType>,
    base_vocab_size: usize,
) {
    let Ok(start) = TokenIdType::try_from(base_vocab_size) else {
        return;
    };

    let occupied_ids: HashSet<TokenIdType> = added_tokens.values().copied().collect();
    for offset in 0..NUM_RESERVED_SPECIAL_TOKENS {
        let id = start + offset as TokenIdType;
        if occupied_ids.contains(&id) {
            continue;
        }

        added_tokens
            .entry(format!("<|reserved_token_{id}|>"))
            .or_insert(id);
    }
}

fn read_json(path: &Path) -> Option<Value> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn model_config_is_kimi(config: &Value) -> bool {
    let model_type = config.get("model_type").and_then(Value::as_str);
    matches!(
        model_type,
        Some("kimi_k2") | Some("kimi_k25") | Some("kimi_k3") | Some("kimi_k31")
    )
}

fn value_mentions_kimi_tokenizer(value: &Value) -> bool {
    match value {
        Value::String(s) => mentions_kimi_tokenizer_module(s),
        Value::Array(values) => values.iter().any(value_mentions_kimi_tokenizer),
        Value::Object(map) => map.values().any(value_mentions_kimi_tokenizer),
        _ => false,
    }
}

/// Match the dotted-path identifier `tokenization_kimi` as a whole segment,
/// not a substring — so `tokenization_kimi.TikTokenTokenizer` matches but
/// `tokenization_kimi_v2` or `my_tokenization_kimi_helper` do not.
fn mentions_kimi_tokenizer_module(s: &str) -> bool {
    s.split('.').any(|seg| seg == "tokenization_kimi")
}

#[cfg(test)]
mod tests {
    use base64::{engine::general_purpose::STANDARD, Engine as _};

    use super::*;
    use crate::{
        tiktoken::TiktokenTokenizer,
        traits::{Decoder, Encoder, Tokenizer},
    };

    // Minimal BPE: bytes 'a' (rank 0), 'b' (rank 1). Used for tests that only
    // exercise decode of synthetic special tokens (no BPE encode).
    const MINIMAL_TIKTOKEN_MODEL: &str = "YQ== 0\nYg== 1\n";
    // Minimal BPE: "hello's" (rank 0), "hello" (rank 1), "'s" (rank 2).
    const CONTRACTION_TIKTOKEN_MODEL: &str = "aGVsbG8ncw== 0\naGVsbG8= 1\nJ3M= 2\n";

    /// Build a tiktoken model file with all 256 single-byte tokens (ranks 0..256)
    /// plus the given multi-byte tokens at successive ranks starting at 256.
    /// tiktoken's BPE requires every input byte to have a rank, so a real-world
    /// encode test needs the full byte layer.
    fn full_byte_bpe(extra: &[&[u8]]) -> String {
        let mut out = String::new();
        for b in 0u32..256 {
            out.push_str(&format!("{} {}\n", STANDARD.encode([b as u8]), b));
        }
        for (offset, bytes) in extra.iter().enumerate() {
            out.push_str(&format!("{} {}\n", STANDARD.encode(bytes), 256 + offset));
        }
        out
    }

    fn write_kimi_dir(model: &str, tokenizer_config: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiktoken.model"), model).unwrap();
        std::fs::write(dir.path().join("tokenizer_config.json"), tokenizer_config).unwrap();
        dir
    }

    const KIMI_AUTO_MAP_CONFIG: &str = r#"{
        "tokenizer_class": "TikTokenTokenizer",
        "auto_map": {
            "AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", null]
        }
    }"#;

    /// Read tokenizer_config.json from `dir`, mirroring what the real loader
    /// hands to `matches`. Used so each test exercises the same call shape.
    fn tokenizer_config(dir: &Path) -> Option<Value> {
        read_json(&dir.join("tokenizer_config.json"))
    }

    #[test]
    fn reserved_special_tokens_are_synthesized() {
        let dir = write_kimi_dir(
            MINIMAL_TIKTOKEN_MODEL,
            r#"{
                "tokenizer_class": "TikTokenTokenizer",
                "auto_map": {
                    "AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", null]
                },
                "added_tokens_decoder": {
                    "2": { "content": "[BOS]", "special": true },
                    "5": { "content": "<|im_assistant|>", "special": true }
                }
            }"#,
        );
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();

        assert_eq!(tokenizer.vocab_size(), 258);
        assert_eq!(
            tokenizer.decode(&[4], false).unwrap(),
            "<|reserved_token_4|>"
        );
        assert_eq!(tokenizer.decode(&[5], false).unwrap(), "<|im_assistant|>");
        assert_eq!(tokenizer.token_to_id("<|reserved_token_4|>"), Some(4));
        assert_eq!(
            tokenizer.id_to_token(4).as_deref(),
            Some("<|reserved_token_4|>")
        );
    }

    #[test]
    fn matches_via_model_type_kimi_k2() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiktoken.model"), MINIMAL_TIKTOKEN_MODEL).unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{ "added_tokens_decoder": {} }"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{ "model_type": "kimi_k2" }"#,
        )
        .unwrap();

        assert!(matches(tokenizer_config(dir.path()).as_ref(), dir.path()));
        // Round-trip a synthetic reserved token to confirm Kimi load path was taken.
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        assert_eq!(
            tokenizer.decode(&[42], false).unwrap(),
            "<|reserved_token_42|>"
        );
    }

    #[test]
    fn matches_via_model_type_kimi_k25() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiktoken.model"), MINIMAL_TIKTOKEN_MODEL).unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{ "added_tokens_decoder": {} }"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{ "model_type": "kimi_k25" }"#,
        )
        .unwrap();

        assert!(matches(tokenizer_config(dir.path()).as_ref(), dir.path()));
    }

    #[test]
    fn matches_via_model_type_kimi_k3() {
        for model_type in ["kimi_k3", "kimi_k31"] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("tiktoken.model"), MINIMAL_TIKTOKEN_MODEL).unwrap();
            std::fs::write(
                dir.path().join("tokenizer_config.json"),
                r#"{ "added_tokens_decoder": {} }"#,
            )
            .unwrap();
            std::fs::write(
                dir.path().join("config.json"),
                format!(r#"{{ "model_type": "{model_type}" }}"#),
            )
            .unwrap();

            assert!(
                matches(tokenizer_config(dir.path()).as_ref(), dir.path()),
                "{model_type}"
            );
        }
    }

    #[test]
    fn substring_does_not_falsely_match_kimi() {
        // Names that *contain* "tokenization_kimi" as a substring but aren't
        // the module identifier itself must not trigger Kimi detection.
        assert!(!mentions_kimi_tokenizer_module("tokenization_kimi_v2"));
        assert!(!mentions_kimi_tokenizer_module(
            "my_tokenization_kimi_helper"
        ));
        // Real upstream forms should still match.
        assert!(mentions_kimi_tokenizer_module("tokenization_kimi"));
        assert!(mentions_kimi_tokenizer_module(
            "tokenization_kimi.TikTokenTokenizer"
        ));
        assert!(mentions_kimi_tokenizer_module(
            "pkg.tokenization_kimi.TikTokenTokenizer"
        ));
    }

    #[test]
    fn uses_kimi_pattern_for_contractions() {
        let dir = write_kimi_dir(CONTRACTION_TIKTOKEN_MODEL, KIMI_AUTO_MAP_CONFIG);
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();
        // The Kimi regex keeps "hello's" as a single match (contraction handling
        // in the third alternation), so the BPE returns rank 0.
        assert_eq!(
            tokenizer.encode("hello's", false).unwrap().token_ids(),
            &[0]
        );
    }

    #[test]
    fn han_input_round_trips_through_kimi_pattern() {
        // The Kimi regex's leading alternation is `[\p{Han}]+`. The main
        // regressions this guards against are (a) the character-class
        // intersection `[X&&[^\p{Han}]]` failing to compile under tiktoken-rs's
        // fancy-regex backend, and (b) Han input being rejected at the
        // pre-tokenizer. A minimal synthetic BPE can't reproduce a real Kimi
        // vocab, so we assert byte-level round-trip rather than exact token
        // IDs: encode must not panic, and decode must reconstruct the input.
        let model = full_byte_bpe(&[]);
        let dir = write_kimi_dir(&model, KIMI_AUTO_MAP_CONFIG);
        let tokenizer = TiktokenTokenizer::from_dir(dir.path()).unwrap();

        let text = "你好世界 hello!";
        let encoding = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(encoding.token_ids(), false).unwrap();
        assert_eq!(decoded, text);
    }
}
