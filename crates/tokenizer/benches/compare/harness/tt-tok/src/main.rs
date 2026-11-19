//! tt-tok: the adapter of the tiktoken Rust crate (`tiktoken-rs` 0.12.1), read directly: a `tiktoken.model` ranks file with the checkpoint
//! class's regex pattern and special tokens (Kimi K3), or the crate's own `o200k_harmony` ranks (gpt-oss).
//! The crate has no incremental decoder: the streamer below is the form a user writes
//! over `decode_bytes` (bytes accumulated, the longest valid UTF-8 prefix emitted), named as such in the backend string.
#![allow(clippy::expect_used)]

#[path = "../../cells.rs"]
mod cells;

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use base64::Engine as _;
use cells::{Streamer, Tok};
use serde_json::{json, Value};
use tiktoken_rs::CoreBPE;

struct Tt {
    inner: CoreBPE,
    kind: String,
    specials: Vec<(String, u32)>,
    vocab: usize,
    /// id of each single-byte token where the ranks hold one (the failure cells' invalid-UTF-8 inputs).
    byte_ids: HashMap<u8, u32>,
}

struct TtStream<'a> {
    bpe: &'a CoreBPE,
    held: Vec<u8>,
}

impl Streamer for TtStream<'_> {
    fn step(&mut self, id: u32, out: &mut String) -> Result<bool, String> {
        let bytes = self.bpe.decode_bytes(&[id]).map_err(|e| e.to_string())?;
        self.held.extend_from_slice(&bytes);
        // emit the longest valid prefix; keep an incomplete trailing sequence
        match std::str::from_utf8(&self.held) {
            Ok(s) => {
                let any = !s.is_empty();
                out.push_str(s);
                self.held.clear();
                Ok(any)
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if e.error_len().is_some() {
                    // an invalid sequence (not a truncation): replace it, as a lossy decode does
                    let s = String::from_utf8_lossy(&self.held).into_owned();
                    out.push_str(&s);
                    self.held.clear();
                    return Ok(!s.is_empty());
                }
                let s = std::str::from_utf8(&self.held[..valid])
                    .unwrap_or("")
                    .to_string();
                out.push_str(&s);
                self.held.drain(..valid);
                Ok(!s.is_empty())
            }
        }
    }
    fn finish(&mut self, out: &mut String) -> Result<(), String> {
        if !self.held.is_empty() {
            out.push_str(&String::from_utf8_lossy(&self.held));
            self.held.clear();
        }
        Ok(())
    }
}

impl Tok for Tt {
    fn backend(&self) -> String {
        format!(
            "tiktoken-rs 0.12.1 {} (streamer: bytes held to a valid prefix, harness form)",
            self.kind
        )
    }
    fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>, String> {
        if special {
            Ok(self.inner.encode_with_special_tokens(text))
        } else {
            Ok(self.inner.encode_ordinary(text))
        }
    }
    fn decode(&self, ids: &[u32], skip: bool) -> Result<String, String> {
        if skip {
            let special: HashSet<u32> = self.specials.iter().map(|(_, id)| *id).collect();
            let kept: Vec<u32> = ids
                .iter()
                .copied()
                .filter(|id| !special.contains(id))
                .collect();
            self.inner.decode(&kept).map_err(|e| e.to_string())
        } else {
            self.inner.decode(ids).map_err(|e| e.to_string())
        }
    }
    fn stream<'a>(&'a self, _skip: bool) -> Box<dyn Streamer + 'a> {
        Box::new(TtStream {
            bpe: &self.inner,
            held: Vec::new(),
        })
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner
            .decode_bytes(&[id])
            .ok()
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        let ids = self.inner.encode_with_special_tokens(token);
        if ids.len() == 1 {
            Some(ids[0])
        } else {
            None
        }
    }
    fn is_special(&self, id: u32) -> Option<bool> {
        Some(self.specials.iter().any(|(_, i)| *i == id))
    }
    fn encode_batch(
        &self,
        _texts: &[&str],
        _special: bool,
    ) -> Option<Result<Vec<Vec<u32>>, String>> {
        None
    }
    fn eos_ids(&self) -> Vec<u32> {
        Vec::new()
    }
    fn specials(&self) -> Value {
        json!({"special_tokens": self.specials.len(), "first": self.specials.iter().take(4).map(|(s, i)| json!({"token": s, "id": i})).collect::<Vec<_>>()})
    }
    fn added_tokens(&self) -> Vec<String> {
        self.specials.iter().map(|(s, _)| s.clone()).collect()
    }
    fn byte_token_id(&self, byte: u8) -> Option<u32> {
        self.byte_ids.get(&byte).copied()
    }
}

/// Kimi's pattern: `tokenization_kimi.py`'s `pat_str` list joined by `|` (the checkpoint class's own regex).
const KIMI_PATTERN: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn read_specials(dir: &Path) -> Vec<(String, u32)> {
    let cfg: Value = std::fs::read_to_string(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null);
    let mut v = Vec::new();
    if let Some(map) = cfg.get("added_tokens_decoder").and_then(|m| m.as_object()) {
        for (id, entry) in map {
            if let (Ok(id), Some(content)) = (
                id.parse::<u32>(),
                entry.get("content").and_then(|c| c.as_str()),
            ) {
                v.push((content.to_string(), id));
            }
        }
    }
    v.sort_by_key(|(_, id)| *id);
    v
}

fn factory(dir: &Path, flags: &HashMap<String, String>) -> Result<Box<dyn Tok>, String> {
    let model = dir.join("tiktoken.model");
    let (inner, kind, specials, vocab, byte_ids) = if model.exists() {
        let text = std::fs::read_to_string(&model).map_err(|e| e.to_string())?;
        let mut encoder: rustc_hash::FxHashMap<Vec<u8>, u32> = rustc_hash::FxHashMap::default();
        let mut byte_ids = HashMap::new();
        for line in text.lines() {
            let mut it = line.split_whitespace();
            let (Some(b64), Some(rank)) = (it.next(), it.next()) else {
                continue;
            };
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| e.to_string())?;
            let rank: u32 = rank
                .parse()
                .map_err(|e: std::num::ParseIntError| e.to_string())?;
            if bytes.len() == 1 {
                byte_ids.insert(bytes[0], rank);
            }
            encoder.insert(bytes, rank);
        }
        let specials = read_specials(dir);
        let mut special_map: rustc_hash::FxHashMap<String, u32> = rustc_hash::FxHashMap::default();
        for (s, id) in &specials {
            special_map.insert(s.clone(), *id);
        }
        let pattern = flags
            .get("pattern")
            .map(String::as_str)
            .unwrap_or(KIMI_PATTERN);
        let n = encoder.len() + specials.len();
        let bpe = CoreBPE::new(encoder, special_map, pattern).map_err(|e| e.to_string())?;
        (
            bpe,
            "tiktoken.model + the class pattern".to_string(),
            specials,
            n,
            byte_ids,
        )
    } else {
        // the crate's own ranks for gpt-oss (o200k_harmony); the single-byte ids found by decoding every id once
        let bpe = tiktoken_rs::o200k_harmony().map_err(|e| e.to_string())?;
        let mut byte_ids = HashMap::new();
        let mut vocab = 0usize;
        for id in 0..210_000u32 {
            if let Ok(b) = bpe.decode_bytes(&[id]) {
                vocab = id as usize + 1;
                if b.len() == 1 {
                    byte_ids.insert(b[0], id);
                }
            }
        }
        let specials: Vec<(String, u32)> = bpe
            .special_tokens()
            .into_iter()
            .map(|s| (s.to_string(), bpe.encode_with_special_tokens(s)[0]))
            .collect();
        (
            bpe,
            "o200k_harmony (crate ranks)".to_string(),
            specials,
            vocab,
            byte_ids,
        )
    };
    Ok(Box::new(Tt {
        inner,
        kind,
        specials,
        vocab,
        byte_ids,
    }))
}

fn main() {
    cells::run(factory, "tt-tok");
}
