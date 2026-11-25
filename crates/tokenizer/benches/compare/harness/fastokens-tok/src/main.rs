//! fastokens-tok: the adapter of the fastokens stack, the open-source Rust tokenizer crate that puts `fastokens`,
//! `riptoken`, `tekken` and `tokenizers` behind one trait (a path dependency on a checkout at the pinned commit).
#![allow(clippy::expect_used)]

#[path = "../../cells.rs"]
mod cells;

use std::{collections::HashMap, path::Path};

use cells::{Streamer, Tok};
use fastokens_stack::{
    HuggingFaceTokenizer, IncrementalDecoder, TekkenTokenizer, TiktokenTokenizer, Tokenizer,
};
use serde_json::{json, Value};

struct V {
    inner: Box<dyn Tokenizer>,
    kind: String,
}

struct VStream<'a> {
    dec: Box<dyn IncrementalDecoder + 'a>,
}

impl Streamer for VStream<'_> {
    fn step(&mut self, id: u32, out: &mut String) -> Result<bool, String> {
        self.dec.push_token(id).map_err(|e| e.to_string())?;
        match self.dec.next_chunk() {
            Some(chunk) => {
                let any = !chunk.text.is_empty();
                out.push_str(&chunk.text);
                Ok(any)
            }
            None => Ok(false),
        }
    }
    fn finish(&mut self, out: &mut String) -> Result<(), String> {
        let (last, _full) = self.dec.flush(None).map_err(|e| e.to_string())?;
        if let Some(chunk) = last {
            out.push_str(&chunk.text);
        }
        Ok(())
    }
}

impl Tok for V {
    fn backend(&self) -> String {
        format!("fastokens-stack {}", self.kind)
    }
    fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>, String> {
        self.inner.encode(text, special).map_err(|e| e.to_string())
    }
    fn decode(&self, ids: &[u32], skip: bool) -> Result<String, String> {
        self.inner.decode(ids, skip).map_err(|e| e.to_string())
    }
    fn stream<'a>(&'a self, skip: bool) -> Box<dyn Streamer + 'a> {
        Box::new(VStream {
            dec: self.inner.create_decode_stream(&[], skip, 0),
        })
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
    fn is_special(&self, id: u32) -> Option<bool> {
        Some(self.inner.is_special_id(id))
    }
    fn encode_batch(
        &self,
        _texts: &[&str],
        _special: bool,
    ) -> Option<Result<Vec<Vec<u32>>, String>> {
        None
    }
    fn eos_ids(&self) -> Vec<u32> {
        // the crate has no eos API; its callers resolve eos from the config files above it
        Vec::new()
    }
    fn specials(&self) -> Value {
        let added = self.inner.added_vocab();
        let specials: Vec<&(String, u32)> = added
            .iter()
            .filter(|(_, id)| self.inner.is_special_id(*id))
            .collect();
        json!({"added_vocab": added.len(), "special_ids": specials.len(), "first_specials": specials.iter().take(8).map(|(s, id)| json!({"token": s, "id": id})).collect::<Vec<_>>()})
    }
    fn added_tokens(&self) -> Vec<String> {
        self.inner
            .added_vocab()
            .iter()
            .map(|(s, _)| s.clone())
            .collect()
    }
}

fn find(dir: &Path, pred: impl Fn(&str) -> bool) -> Option<std::path::PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(&pred)
                .unwrap_or(false)
        })
        .collect();
    v.sort();
    v.into_iter().next()
}

/// The selection order of the stack's own loader: tekken.json, then a tiktoken file,
/// then a `tokenizer_class` naming Tiktoken, then tokenizer.json. `--file` and `--backend` override.
fn factory(dir: &Path, flags: &HashMap<String, String>) -> Result<Box<dyn Tok>, String> {
    let backend = flags.get("backend").map(String::as_str).unwrap_or("auto");
    let file = match flags.get("file") {
        Some(f) => dir.join(f),
        None => {
            if dir.join("tekken.json").exists() {
                dir.join("tekken.json")
            } else if let Some(p) = find(dir, |n| n == "tiktoken.model" || n.ends_with(".tiktoken"))
            {
                p
            } else {
                let class_tiktoken = std::fs::read_to_string(dir.join("tokenizer_config.json"))
                    .ok()
                    .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                    .and_then(|v| {
                        v.get("tokenizer_class")
                            .and_then(|c| c.as_str())
                            .map(|c| c.to_ascii_lowercase().contains("tiktoken"))
                    })
                    .unwrap_or(false);
                if class_tiktoken {
                    find(dir, |n| n == "tiktoken.model" || n.ends_with(".tiktoken"))
                        .ok_or("tokenizer_class names Tiktoken but no BPE file")?
                } else {
                    dir.join("tokenizer.json")
                }
            }
        }
    };
    let name = file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    let (inner, kind): (Box<dyn Tokenizer>, String) = if name == "tekken.json" {
        (
            Box::new(TekkenTokenizer::new(&file).map_err(|e| e.to_string())?),
            "tekken".into(),
        )
    } else if name == "tiktoken.model" || name.ends_with(".tiktoken") {
        match backend {
            "tiktoken-rs" => (
                Box::new(TiktokenTokenizer::new_tiktoken_rs(&file).map_err(|e| e.to_string())?),
                "tiktoken-rs".into(),
            ),
            "riptoken" => (
                Box::new(TiktokenTokenizer::new_riptoken(&file).map_err(|e| e.to_string())?),
                "riptoken".into(),
            ),
            _ => (
                Box::new(TiktokenTokenizer::new(&file).map_err(|e| e.to_string())?),
                "tiktoken(auto: riptoken first)".into(),
            ),
        }
    } else {
        match backend {
            "hf" => (
                Box::new(HuggingFaceTokenizer::new_hf(&file).map_err(|e| e.to_string())?),
                "hf-tokenizers".into(),
            ),
            "fastokens" => (
                Box::new(HuggingFaceTokenizer::new_fastokens(&file).map_err(|e| e.to_string())?),
                "fastokens".into(),
            ),
            _ => {
                // report which backend `new` actually took: try fastokens first exactly as `new` does
                match HuggingFaceTokenizer::new_fastokens(&file) {
                    Ok(t) => (Box::new(t), "fastokens".into()),
                    Err(e) => {
                        let fallback = HuggingFaceTokenizer::new_hf(&file)
                            .map_err(|e2| format!("fastokens: {e}; tokenizers: {e2}"))?;
                        (
                            Box::new(fallback),
                            format!("hf-tokenizers(fastokens declined: {e})"),
                        )
                    }
                }
            }
        }
    };
    Ok(Box::new(V { inner, kind }))
}

fn main() {
    cells::run(factory, "fastokens-tok");
}
