//! hf-tok: the adapter of the `tokenizers` crate, read directly through its default pipeline.
#![allow(clippy::expect_used)]

#[path = "../../cells.rs"]
mod cells;

use std::{collections::HashMap, path::Path};

use cells::{Streamer, Tok};
use serde_json::{json, Value};
use tokenizers::{DecodeStream, Tokenizer};

struct Hf {
    inner: Tokenizer,
}

struct HfStream<'a>(
    DecodeStream<
        'a,
        tokenizers::ModelWrapper,
        tokenizers::NormalizerWrapper,
        tokenizers::PreTokenizerWrapper,
        tokenizers::PostProcessorWrapper,
        tokenizers::DecoderWrapper,
    >,
);

impl Streamer for HfStream<'_> {
    fn step(&mut self, id: u32, out: &mut String) -> Result<bool, String> {
        match self.0.step(id).map_err(|e| e.to_string())? {
            Some(s) => {
                let any = !s.is_empty();
                out.push_str(&s);
                Ok(any)
            }
            None => Ok(false),
        }
    }
    fn finish(&mut self, _out: &mut String) -> Result<(), String> {
        // the crate's stream has no end-of-stream flush: whatever is held stays held
        Ok(())
    }
}

impl Tok for Hf {
    fn backend(&self) -> String {
        "tokenizers-crate 0.23.2 default-pipeline".to_string()
    }
    fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>, String> {
        self.inner
            .encode_fast(text, special)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| e.to_string())
    }
    fn decode(&self, ids: &[u32], skip: bool) -> Result<String, String> {
        self.inner.decode(ids, skip).map_err(|e| e.to_string())
    }
    fn stream<'a>(&'a self, skip: bool) -> Box<dyn Streamer + 'a> {
        Box::new(HfStream(self.inner.decode_stream(skip)))
    }
    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
    fn is_special(&self, id: u32) -> Option<bool> {
        Some(
            self.inner
                .get_added_vocabulary()
                .is_special_token(&self.inner.id_to_token(id).unwrap_or_default()),
        )
    }
    fn encode_batch(&self, texts: &[&str], special: bool) -> Option<Result<Vec<Vec<u32>>, String>> {
        Some(
            self.inner
                .encode_batch_fast(texts.to_vec(), special)
                .map(|v| v.into_iter().map(|e| e.get_ids().to_vec()).collect())
                .map_err(|e| e.to_string()),
        )
    }
    fn eos_ids(&self) -> Vec<u32> {
        Vec::new()
    }
    fn specials(&self) -> Value {
        let added = self.inner.get_added_vocabulary();
        json!({"added_tokens": added.len()})
    }
    fn added_tokens(&self) -> Vec<String> {
        self.inner
            .get_added_vocabulary()
            .get_vocab()
            .keys()
            .cloned()
            .collect()
    }
}

/// `tokenizer.json` of the directory (or `--file <name>`); the crate reads no `tokenizer_config.json`.
fn factory(dir: &Path, flags: &HashMap<String, String>) -> Result<Box<dyn Tok>, String> {
    let file = match flags.get("file") {
        Some(f) => dir.join(f),
        None => dir.join("tokenizer.json"),
    };
    let inner = Tokenizer::from_file(&file).map_err(|e| e.to_string())?;
    Ok(Box::new(Hf { inner }))
}

fn main() {
    cells::run(factory, "hf-tok");
}
