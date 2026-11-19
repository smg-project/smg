//! smg-tok: the adapter of this crate (`llm-tokenizer`, a path dependency on the checkout around it).
#![allow(clippy::expect_used)]

#[path = "../../cells.rs"]
mod cells;

use std::{collections::HashMap, path::Path, sync::Arc};

use cells::{Streamer, Tok};
use llm_tokenizer::{
    create_tokenizer, encode_path_loads, eos, traits::Tokenizer as TokenizerTrait, Decoder as _,
    HuggingFaceTokenizer, Sequence,
};
use serde_json::{json, Value};

struct Smg {
    inner: Arc<dyn TokenizerTrait>,
    dir: std::path::PathBuf,
    /// The encode path the bare tokenizer took ("hf:native", "hf:native+bytefallback", "tiktoken"), read in the factory
    /// before any wrapper hides the concrete type; "+l0" appended when `--l0 on` wrapped it.
    path: String,
}

struct SmgStream(Sequence);

impl Streamer for SmgStream {
    fn step(&mut self, id: u32, out: &mut String) -> Result<bool, String> {
        self.0.append_token_into(id, out).map_err(|e| e.to_string())
    }
    fn finish(&mut self, out: &mut String) -> Result<(), String> {
        let tail = self.0.finish().map_err(|e| e.to_string())?;
        out.push_str(&tail);
        Ok(())
    }
}

impl Tok for Smg {
    fn backend(&self) -> String {
        let loads: Vec<String> = encode_path_loads()
            .iter()
            .filter(|l| l.loads > 0)
            .map(|l| format!("{}x{}", l.path, l.loads))
            .collect();
        format!("smg {} [{}]", self.path, loads.join(","))
    }
    fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>, String> {
        self.inner
            .encode(text, special)
            .map(|e| e.into_token_ids())
            .map_err(|e| e.to_string())
    }
    fn decode(&self, ids: &[u32], skip: bool) -> Result<String, String> {
        self.inner.decode(ids, skip).map_err(|e| e.to_string())
    }
    fn stream<'a>(&'a self, skip: bool) -> Box<dyn Streamer + 'a> {
        Box::new(SmgStream(Sequence::new_with_options(
            self.inner.clone(),
            skip,
        )))
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
    fn is_special(&self, _id: u32) -> Option<bool> {
        None
    }
    fn encode_batch(&self, texts: &[&str], special: bool) -> Option<Result<Vec<Vec<u32>>, String>> {
        Some(
            self.inner
                .encode_batch(texts, special)
                .map(|v| v.into_iter().map(|e| e.into_token_ids()).collect())
                .map_err(|e| e.to_string()),
        )
    }
    fn eos_ids(&self) -> Vec<u32> {
        let mut ids = self.inner.eos_token_ids().to_vec();
        if ids.is_empty() {
            ids = eos::load_eos_token_ids(&self.dir);
        }
        ids
    }
    fn specials(&self) -> Value {
        let s = self.inner.get_special_tokens();
        let id = |t: &Option<String>| {
            t.as_ref()
                .map(|t| json!({"token": t, "id": self.inner.token_to_id(t)}))
        };
        json!({"bos": id(&s.bos_token), "eos": id(&s.eos_token), "pad": id(&s.pad_token), "unk": id(&s.unk_token), "additional": s.additional_special_tokens.len()})
    }
    fn added_tokens(&self) -> Vec<String> {
        let s = self.inner.get_special_tokens();
        let mut v: Vec<String> = [
            &s.bos_token,
            &s.eos_token,
            &s.pad_token,
            &s.unk_token,
            &s.sep_token,
            &s.cls_token,
            &s.mask_token,
        ]
        .into_iter()
        .flatten()
        .cloned()
        .collect();
        v.extend(s.additional_special_tokens.iter().cloned());
        v.extend(self.inner.added_token_strings());
        v.extend(self.inner.added_tokens_after_normalization());
        v.sort();
        v.dedup();
        v
    }
}

fn factory(dir: &Path, flags: &HashMap<String, String>) -> Result<Box<dyn Tok>, String> {
    let target = match flags.get("file") {
        Some(f) => dir.join(f),
        None => dir.to_path_buf(),
    };
    let inner =
        create_tokenizer(target.to_str().expect("utf-8 path")).map_err(|e| format!("{e:#}"))?;
    let mut path = inner
        .as_any()
        .downcast_ref::<HuggingFaceTokenizer>()
        .map(|h| {
            format!(
                "hf:{}{}",
                h.encode_path(),
                if h.uses_byte_fallback() {
                    "+bytefallback"
                } else {
                    ""
                }
            )
        })
        .unwrap_or_else(|| "tiktoken".to_string());
    // `--l0 on` wraps the tokenizer in the crate's whole-text
    // exact-match cache, `CachedTokenizer::new(inner, CacheConfig::default())` (L0 on: 10,000 entries, 256 MiB; L1 off),
    // The crate is untouched; the wrap lives in this harness, and the published tables run without it.
    let inner: Arc<dyn TokenizerTrait> = if flags.get("l0").map(String::as_str) == Some("on") {
        path.push_str("+l0");
        Arc::new(llm_tokenizer::CachedTokenizer::new(
            inner,
            llm_tokenizer::CacheConfig::default(),
        ))
    } else {
        inner
    };
    Ok(Box::new(Smg {
        inner,
        dir: dir.to_path_buf(),
        path,
    }))
}

fn main() {
    // SMG_TOK_TRACE=1 with RUST_LOG=info prints the crate's load log (the encode path chosen and the declines).
    if std::env::var_os("SMG_TOK_TRACE").is_some() {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .init();
    }
    // `dump-json <dir>`: the configured pipeline as the crate serializes it (what the v1 and fastokens paths read).
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("dump-json") {
        let inner = create_tokenizer(&args[2]).expect("load");
        let hf = inner
            .as_any()
            .downcast_ref::<HuggingFaceTokenizer>()
            .expect("a Hugging Face tokenizer");
        println!(
            "{}",
            hf.tokenizers_pipeline()
                .to_string(false)
                .expect("serialize")
        );
        return;
    }
    cells::run(factory, "smg-tok");
}
