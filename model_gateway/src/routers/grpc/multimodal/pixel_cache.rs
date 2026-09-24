//! Host-DRAM LRU cache of preprocessed per-image encoder inputs for the gateway
//! multimodal path. Disabled by default (`--mm-pixel-cache-mb` /
//! `SMG_MM_PIXEL_CACHE_MB` unset or 0).

use std::{
    mem::size_of,
    sync::{Arc, OnceLock},
};

use llm_multimodal::{ModelSpecificValue, PreprocessedEncoderInputs};
use lru::LruCache;
use parking_lot::Mutex;

/// Identifies a preprocessed image output. Hashing raw image bytes alone is not
/// enough because the same bytes preprocess differently under another model config.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct PixelCacheKey {
    /// blake3 hex digest of the raw encoded image bytes (`ImageFrame.hash`).
    pub image_hash: String,
    /// Stable hash of model identity/config for this deployment.
    pub config_fingerprint: u64,
}

/// Cached per-image vision processor output before backend-specific serialization.
#[derive(Debug)]
pub(crate) struct CachedPreprocessedItem {
    pub preprocessed: PreprocessedEncoderInputs,
}

impl CachedPreprocessedItem {
    fn heap_bytes(&self) -> usize {
        preprocessed_heap_bytes(&self.preprocessed)
    }
}

fn preprocessed_heap_bytes(preprocessed: &PreprocessedEncoderInputs) -> usize {
    let model_specific: usize = preprocessed
        .model_specific
        .iter()
        .map(|(key, value)| key.len() + model_specific_value_heap_bytes(value))
        .sum();
    preprocessed.encoder_input.len() * size_of::<f32>()
        + preprocessed.encoder_input.ndim() * size_of::<usize>()
        + preprocessed.feature_token_counts.len() * size_of::<usize>()
        + preprocessed.item_sizes.len() * size_of::<(u32, u32)>()
        + model_specific
}

fn model_specific_value_heap_bytes(value: &ModelSpecificValue) -> usize {
    match value {
        ModelSpecificValue::Tensor { data, shape } => {
            data.len() * size_of::<f32>() + shape.len() * size_of::<usize>()
        }
        ModelSpecificValue::IntTensor { data, shape } => {
            data.len() * size_of::<i64>() + shape.len() * size_of::<usize>()
        }
        ModelSpecificValue::UintTensor { data, shape } => {
            data.len() * size_of::<u32>() + shape.len() * size_of::<usize>()
        }
        ModelSpecificValue::Int(_) => size_of::<i64>(),
        ModelSpecificValue::Float(_) => size_of::<f64>(),
        ModelSpecificValue::IntVec(values) => values.len() * size_of::<i64>(),
        ModelSpecificValue::UintVec(values) => values.len() * size_of::<u32>(),
        ModelSpecificValue::FloatVec(values) => values.len() * size_of::<f32>(),
        ModelSpecificValue::TupleVec(values) => values.len() * size_of::<(u32, u32)>(),
        ModelSpecificValue::Bool(_) => size_of::<bool>(),
    }
}

const PIXEL_CACHE_KEY_OVERHEAD: usize = 128;

struct PixelCacheInner {
    map: LruCache<PixelCacheKey, Arc<CachedPreprocessedItem>>,
    cur_bytes: usize,
    max_bytes: usize,
}

/// Thread-safe, byte-budgeted LRU of per-image vision processor outputs.
pub(crate) struct PixelCache {
    inner: Mutex<PixelCacheInner>,
}

impl PixelCache {
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(PixelCacheInner {
                map: LruCache::unbounded(),
                cur_bytes: 0,
                max_bytes,
            }),
        }
    }

    pub(crate) fn get(&self, key: &PixelCacheKey) -> Option<Arc<CachedPreprocessedItem>> {
        let mut inner = self.inner.lock();
        inner.map.get(key).cloned()
    }

    pub(crate) fn insert(&self, key: PixelCacheKey, value: Arc<CachedPreprocessedItem>) {
        let entry_bytes = value.heap_bytes() + PIXEL_CACHE_KEY_OVERHEAD;
        let mut inner = self.inner.lock();
        if entry_bytes > inner.max_bytes {
            return;
        }
        let PixelCacheInner {
            map,
            cur_bytes,
            max_bytes,
        } = &mut *inner;
        if let Some(previous) = map.put(key, value) {
            *cur_bytes = cur_bytes.saturating_sub(previous.heap_bytes() + PIXEL_CACHE_KEY_OVERHEAD);
        }
        *cur_bytes += entry_bytes;
        while *cur_bytes > *max_bytes {
            match map.pop_lru() {
                Some((_, evicted)) => {
                    *cur_bytes =
                        cur_bytes.saturating_sub(evicted.heap_bytes() + PIXEL_CACHE_KEY_OVERHEAD);
                }
                None => break,
            }
        }
    }

    #[cfg(test)]
    fn current_bytes(&self) -> usize {
        self.inner.lock().cur_bytes
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.inner.lock().map.len()
    }
}

/// The process-wide pixel cache with a budget of `mb` MiB; 0 keeps it off.
/// Built once: the first budget wins, later routers share it.
pub(crate) fn pixel_cache_with_budget(mb: usize) -> Option<Arc<PixelCache>> {
    static CACHE: OnceLock<Option<Arc<PixelCache>>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            if mb == 0 {
                return None;
            }
            let max_bytes = mb.saturating_mul(1024 * 1024);
            tracing::info!(
                target: "smg::request",
                cache_mb = mb,
                "multimodal pixel_values cache enabled (host DRAM, preprocessed encoder inputs)"
            );
            Some(Arc::new(PixelCache::new(max_bytes)))
        })
        .clone()
}

pub(crate) fn config_fingerprint(tokenizer_id: &str, config: &serde_json::Value) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(tokenizer_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(config.to_string().as_bytes());
    let digest = hasher.finalize();
    let mut head = [0u8; 8];
    head.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_le_bytes(head)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ndarray::{ArrayD, IxDyn};

    use super::*;

    fn pixel_cache_item(token_count: usize, payload: usize) -> Arc<CachedPreprocessedItem> {
        Arc::new(CachedPreprocessedItem {
            preprocessed: PreprocessedEncoderInputs {
                encoder_input: ArrayD::from_elem(IxDyn(&[1, payload]), token_count as f32),
                feature_token_counts: vec![token_count],
                item_sizes: vec![(payload as u32, 1)],
                model_specific: HashMap::new(),
            },
        })
    }

    fn pixel_cache_key(hash: &str) -> PixelCacheKey {
        PixelCacheKey {
            image_hash: hash.to_string(),
            config_fingerprint: 7,
        }
    }

    #[test]
    fn pixel_cache_hit_returns_shared_arc() {
        let cache = PixelCache::new(1024 * 1024);
        let value = pixel_cache_item(10, 64);
        cache.insert(pixel_cache_key("a"), value.clone());
        let got = cache.get(&pixel_cache_key("a")).expect("hit");
        assert_eq!(got.preprocessed.feature_token_counts, vec![10]);
        assert!(Arc::ptr_eq(&got, &value));
        assert!(cache.get(&pixel_cache_key("missing")).is_none());
    }

    #[test]
    fn pixel_cache_key_distinguishes_fingerprint() {
        let cache = PixelCache::new(1024 * 1024);
        let mut k_other_model = pixel_cache_key("a");
        k_other_model.config_fingerprint = 99;
        cache.insert(pixel_cache_key("a"), pixel_cache_item(1, 8));
        cache.insert(k_other_model.clone(), pixel_cache_item(3, 8));
        assert_eq!(
            cache
                .get(&pixel_cache_key("a"))
                .unwrap()
                .preprocessed
                .feature_token_counts,
            vec![1]
        );
        assert_eq!(
            cache
                .get(&k_other_model)
                .unwrap()
                .preprocessed
                .feature_token_counts,
            vec![3]
        );
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn pixel_cache_evicts_lru_when_over_budget() {
        let entry = pixel_cache_item(1, 4096).heap_bytes() + PIXEL_CACHE_KEY_OVERHEAD;
        let budget = entry * 2 + entry / 2;
        let cache = PixelCache::new(budget);
        cache.insert(pixel_cache_key("a"), pixel_cache_item(1, 4096));
        cache.insert(pixel_cache_key("b"), pixel_cache_item(2, 4096));
        assert!(cache.get(&pixel_cache_key("a")).is_some());
        assert!(cache.get(&pixel_cache_key("b")).is_some());
        assert!(cache.get(&pixel_cache_key("a")).is_some());
        cache.insert(pixel_cache_key("c"), pixel_cache_item(3, 4096));
        assert!(cache.get(&pixel_cache_key("a")).is_some());
        assert!(cache.get(&pixel_cache_key("b")).is_none());
        assert!(cache.get(&pixel_cache_key("c")).is_some());
        assert!(cache.current_bytes() <= budget);
    }

    #[test]
    fn pixel_cache_oversized_entry_is_bypassed() {
        let cache = PixelCache::new(32);
        cache.insert(pixel_cache_key("huge"), pixel_cache_item(1, 1024));
        assert!(cache.get(&pixel_cache_key("huge")).is_none());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.current_bytes(), 0);
    }
}
