//! Host-DRAM LRU cache of preprocessed per-image encoder inputs for the gateway
//! multimodal path. Disabled by default (`--mm-pixel-cache-mb` /
//! `SMG_MM_PIXEL_CACHE_MB` unset or 0).

use std::{
    collections::{hash_map::Entry, HashMap},
    io::{self, BufWriter, Write},
    mem::size_of,
    num::NonZeroUsize,
    sync::{Arc, OnceLock},
};

use llm_multimodal::{ModelSpecificValue, PreprocessedEncoderInputs};
use lru::LruCache;
use parking_lot::{Mutex, MutexGuard};
use serde_json::Value;

use crate::observability::metrics::Metrics;

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
    preprocessed
        .encoder_input
        .shared_allocation()
        .map_or_else(|| preprocessed.encoder_input.nbytes(), |_| 0)
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
    recent_large_misses: LruCache<PixelCacheKey, u64>,
    large_fill_bytes: u64,
    allocations: HashMap<usize, usize>,
    cur_bytes: usize,
}

impl PixelCacheInner {
    /// Charge a shared wire allocation when its first slice enters the cache.
    fn charge(&mut self, value: &CachedPreprocessedItem) {
        self.cur_bytes += value.heap_bytes() + PIXEL_CACHE_KEY_OVERHEAD;
        if let Some((id, capacity)) = value.preprocessed.encoder_input.shared_allocation() {
            let count = self.allocations.entry(id).or_default();
            if *count == 0 {
                self.cur_bytes += capacity;
            }
            *count += 1;
        }
    }

    /// Release a shared allocation's charge when its last slice leaves.
    fn uncharge(&mut self, value: &CachedPreprocessedItem) {
        self.cur_bytes -= value.heap_bytes() + PIXEL_CACHE_KEY_OVERHEAD;
        if let Some((id, capacity)) = value.preprocessed.encoder_input.shared_allocation() {
            if let Entry::Occupied(mut count) = self.allocations.entry(id) {
                *count.get_mut() -= 1;
                if *count.get() == 0 {
                    count.remove();
                    self.cur_bytes -= capacity;
                }
            }
        }
    }
}

/// Thread-safe, byte-budgeted LRU of per-image vision processor outputs.
pub(crate) struct PixelCache {
    max_bytes: usize,
    inner: Mutex<PixelCacheInner>,
}

impl PixelCache {
    pub(crate) fn new(max_bytes: usize) -> Self {
        // The cache's occupancy and evictions are on the first scrape, at
        // zero, rather than appearing with the first image.
        Metrics::init_mm_pixel_cache_series();
        Self {
            max_bytes,
            inner: Mutex::new(PixelCacheInner {
                map: LruCache::unbounded(),
                recent_large_misses: LruCache::sparse(
                    NonZeroUsize::new(1024).unwrap_or(NonZeroUsize::MIN),
                ),
                large_fill_bytes: 0,
                allocations: HashMap::new(),
                cur_bytes: 0,
            }),
        }
    }

    /// Upper bound for the total size of resident entries.
    pub(super) fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub(crate) fn get(&self, key: &PixelCacheKey) -> Option<Arc<CachedPreprocessedItem>> {
        let mut inner = self.inner.lock();
        inner.map.get(key).cloned()
    }

    /// Remember large cold fills without pinning their pixels. Admit when at
    /// least half the images recurred within one cache budget of intervening fills.
    pub(super) fn admit_repeated<'a>(
        &self,
        keys: impl ExactSizeIterator<Item = &'a PixelCacheKey>,
        capacity: usize,
    ) -> bool {
        let required = keys.len().div_ceil(2);
        let mut inner = self.inner.lock();
        let start = inner.large_fill_bytes;
        let end = start.wrapping_add(capacity as u64);
        inner.large_fill_bytes = end;
        keys.filter_map(|key| inner.recent_large_misses.put(key.clone(), end))
            .filter(|previous| start.wrapping_sub(*previous) < self.max_bytes as u64)
            .count()
            >= required
    }

    pub(crate) fn insert(&self, key: PixelCacheKey, value: Arc<CachedPreprocessedItem>) {
        self.insert_locked(self.inner.lock(), [(key, value)], true);
    }

    /// Insert the request's misses that are still absent under the cache lock.
    pub(super) fn insert_batch(
        &self,
        entries: impl IntoIterator<Item = (PixelCacheKey, Arc<CachedPreprocessedItem>)>,
    ) {
        self.insert_locked(self.inner.lock(), entries, false);
    }

    /// Check capacity plus the separately allocated metadata of every slice.
    pub(super) fn shared_batch_fits<'a>(
        &self,
        capacity: usize,
        entries: impl Iterator<Item = &'a PreprocessedEncoderInputs>,
    ) -> bool {
        self.max_bytes
            .checked_sub(capacity)
            .is_some_and(|remaining| {
                entries
                    .map(|entry| preprocessed_heap_bytes(entry) + PIXEL_CACHE_KEY_OVERHEAD)
                    .try_fold(remaining, usize::checked_sub)
                    .is_some()
            })
    }

    /// Install a whole shared allocation only if it fits and replacing its
    /// images releases every older allocation touched by the replacement.
    pub(super) fn insert_shared_batch(
        &self,
        entries: Vec<(PixelCacheKey, Arc<CachedPreprocessedItem>)>,
        capacity: usize,
    ) -> bool {
        if !self.shared_batch_fits(
            capacity,
            entries.iter().map(|(_, entry)| &entry.preprocessed),
        ) {
            return false;
        }
        let mut replaced = HashMap::new();
        let inner = self.inner.lock();
        for (key, _) in &entries {
            if let Some((owner, _)) = inner
                .map
                .peek(key)
                .and_then(|value| value.preprocessed.encoder_input.shared_allocation())
            {
                *replaced.entry(owner).or_insert(0) += 1;
            }
        }
        if replaced
            .iter()
            .any(|(owner, count)| inner.allocations.get(owner) != Some(count))
        {
            return false;
        }
        self.insert_locked(inner, entries, true);
        true
    }

    /// Keep replacement checks and insertion within the same cache lock.
    fn insert_locked(
        &self,
        mut inner: MutexGuard<'_, PixelCacheInner>,
        entries: impl IntoIterator<Item = (PixelCacheKey, Arc<CachedPreprocessedItem>)>,
        replace: bool,
    ) {
        let mut released = Vec::new();
        let mut evicted_entries = 0;
        let mut inserted = false;
        for (key, value) in entries {
            if !replace && inner.map.contains(&key) {
                continue;
            }
            let capacity = value
                .preprocessed
                .encoder_input
                .shared_allocation()
                .map_or(0, |(_, bytes)| bytes);
            if value.heap_bytes() + PIXEL_CACHE_KEY_OVERHEAD + capacity > self.max_bytes {
                continue;
            }
            inserted = true;
            inner.charge(&value);
            if let Some(previous) = inner.map.put(key, value) {
                inner.uncharge(&previous);
                released.push(previous);
            }
        }
        while inner.cur_bytes > self.max_bytes {
            if let Some((_, evicted)) = inner.map.pop_lru() {
                inner.uncharge(&evicted);
                released.push(evicted);
                evicted_entries += 1;
            } else {
                break;
            }
        }
        let (bytes, entries) = (inner.cur_bytes, inner.map.len());
        drop(inner);
        if inserted {
            Metrics::record_mm_pixel_cache_insert(evicted_entries, bytes, entries);
        }
        drop(released);
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

/// A process-local cache identity without cloning the model or building its JSON string.
/// Object order is ignored; types, array order and numeric spelling are retained.
/// The model is borrowed separately from the request's remaining config fields.
pub(crate) fn config_fingerprint(
    tokenizer_id: &str,
    config: &Value,
    model_config: &Value,
) -> io::Result<u64> {
    let mut writer = BufWriter::with_capacity(4096, blake3::Hasher::new());
    writer.write_all(b"smg-pixel-config-v1")?;
    write_fingerprint_bytes(&mut writer, tokenizer_id.as_bytes())?;
    write_fingerprint_value(&mut writer, config)?;
    write_fingerprint_value(&mut writer, model_config)?;
    writer.flush()?;
    let digest = writer.get_ref().finalize();
    let mut head = [0u8; 8];
    head.copy_from_slice(&digest.as_bytes()[..8]);
    Ok(u64::from_le_bytes(head))
}

/// Frame arbitrary bytes so embedded separators cannot alias adjacent fields.
fn write_fingerprint_bytes(writer: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
    writer.write_all(bytes)
}

/// Encode typed, framed values, sorting borrowed object entries only.
/// Number spelling keeps integer/float and signed-zero identities distinct.
fn write_fingerprint_value(writer: &mut impl Write, value: &Value) -> io::Result<()> {
    match value {
        Value::Null => writer.write_all(&[0]),
        Value::Bool(value) => writer.write_all(&[1, u8::from(*value)]),
        Value::Number(value) => {
            writer.write_all(&[2])?;
            write!(writer, "{value}")?;
            // 0xff cannot occur in the UTF-8 spelling, including arbitrary-precision numbers.
            writer.write_all(&[0xff])
        }
        Value::String(value) => {
            writer.write_all(&[3])?;
            write_fingerprint_bytes(writer, value.as_bytes())
        }
        Value::Array(values) => {
            writer.write_all(&[4])?;
            writer.write_all(&(values.len() as u64).to_le_bytes())?;
            for value in values {
                write_fingerprint_value(writer, value)?;
            }
            Ok(())
        }
        Value::Object(values) => {
            writer.write_all(&[5])?;
            writer.write_all(&(values.len() as u64).to_le_bytes())?;
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
            for (key, value) in entries {
                write_fingerprint_bytes(writer, key.as_bytes())?;
                write_fingerprint_value(writer, value)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/multimodal/pixel_cache/tests.rs"]
mod tests;
