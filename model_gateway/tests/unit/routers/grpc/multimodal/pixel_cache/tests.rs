use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use serde_json::json;

use super::*;

/// Canonicalization ignores nested object order without mutating either input.
#[test]
fn fingerprint_is_canonical_and_borrowed() {
    let model = json!({"z": [{"b": 1, "a": 2}], "a": [3, 4]});
    let config = json!({"z": 1, "a": {"y": true, "x": null}});
    let before = (model.to_string(), config.to_string());
    let mut sorted_model = model.clone();
    let mut sorted_config = config.clone();
    sorted_model.sort_all_objects();
    sorted_config.sort_all_objects();
    assert_eq!(
        config_fingerprint("tokenizer", &config, &model).unwrap(),
        config_fingerprint("tokenizer", &sorted_config, &sorted_model).unwrap()
    );
    assert_eq!(before, (model.to_string(), config.to_string()));
}

/// Typed framing preserves distinctions that JSON serialization previously made.
#[test]
fn fingerprint_distinguishes_types_and_boundaries() {
    let values = [
        json!(null),
        json!(false),
        json!(true),
        json!(0),
        json!(0.0),
        json!(-0.0),
        json!(1),
        json!(1.0),
        json!(f64::from_bits(1)),
        json!(-1),
        json!(u64::MAX),
        json!(""),
        json!("0"),
        json!([]),
        json!({}),
        json!(["a", "bc"]),
        json!(["ab", "c"]),
        json!({"a": "bc"}),
        json!({"ab": "c"}),
        json!([1, 2]),
        json!([2, 1]),
        json!([1, 23]),
        json!([12, 3]),
        json!("\n"),
        json!("\\n"),
        json!("α\0β"),
    ];
    let mut encodings = std::collections::HashSet::new();
    for value in values {
        let mut encoded = Vec::new();
        write_fingerprint_value(&mut encoded, &value).unwrap();
        assert!(encodings.insert(encoded), "aliased value: {value}");
    }
    assert_ne!(
        config_fingerprint("a\0b", &json!("c"), &Value::Null).unwrap(),
        config_fingerprint("a", &json!("b\0c"), &Value::Null).unwrap()
    );
}

/// Each supplied metadata value, tokenizer, and borrowed model participates in cache identity.
#[test]
fn fingerprint_keeps_all_cache_identity_fields() {
    let config = json!({
        "model_id": "model", "model_type": "minimax_m3_vl",
        "preprocessor_config": {"patch_size": 14},
        "image_wire_dtype": "bfloat16", "transparency": "keep_alpha",
    });
    let mut model = json!({"vision_config": {"hidden_size": 1024}});
    let original = config_fingerprint("tokenizer", &config, &model).unwrap();
    let cache = PixelCache::new(1024 * 1024);
    let mut key = pixel_cache_key("same-image");
    key.config_fingerprint = original;
    cache.insert(key.clone(), pixel_cache_item(1, 8));
    assert!(cache.get(&key).is_some());
    for field in config.as_object().unwrap().keys() {
        let mut changed = config.clone();
        changed[field] = json!("different");
        key.config_fingerprint = config_fingerprint("tokenizer", &changed, &model).unwrap();
        assert!(cache.get(&key).is_none(), "omitted field: {field}");
    }
    assert_ne!(
        original,
        config_fingerprint("other", &config, &model).unwrap()
    );
    model["vision_config"]["hidden_size"] = json!(2048);
    assert_ne!(
        original,
        config_fingerprint("tokenizer", &config, &model).unwrap()
    );
}

/// The last buffered bytes participate even around buffer boundaries and large writes.
#[test]
fn fingerprint_includes_long_string_tails() {
    for length in [4095, 4096, 4097, 512 * 1024] {
        let mut text = "\nα\0".repeat(length / 4);
        text.extend(std::iter::repeat_n('x', length - text.len()));
        let before = config_fingerprint("t", &json!({}), &json!({"text": text})).unwrap();
        text.pop().unwrap();
        text.push('y');
        assert_ne!(
            before,
            config_fingerprint("t", &json!({}), &json!({"text": text})).unwrap()
        );
    }
}

fn pixel_cache_item(token_count: usize, payload: usize) -> Arc<CachedPreprocessedItem> {
    Arc::new(CachedPreprocessedItem {
        preprocessed: PreprocessedEncoderInputs {
            encoder_input: ArrayD::from_elem(IxDyn(&[1, payload]), token_count as f32).into(),
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
fn cold_large_fills_require_reuse() {
    let cache = PixelCache::new(400);
    let a = ["a", "b", "c", "d"].map(pixel_cache_key);
    let b = ["d", "e", "f", "g"].map(pixel_cache_key);
    assert!(!cache.admit_repeated(a.iter(), 100));
    assert!(!cache.admit_repeated(b.iter(), 100));
    assert!(cache.admit_repeated(a.iter(), 100));
    for _ in 0..4 {
        cache.admit_repeated(b.iter(), 100);
    }
    assert!(!cache.admit_repeated(a.iter(), 100));
    assert_eq!(cache.current_bytes(), 0);
}

#[test]
fn immediate_reuse_excludes_the_previous_fill_capacity() {
    let keys = ["a", "b"].map(pixel_cache_key);
    for capacity in [400, 800] {
        let cache = PixelCache::new(400);
        assert!(!cache.admit_repeated(keys.iter(), capacity));
        assert!(cache.admit_repeated(keys.iter(), capacity));
    }
}

fn shared_pixel_entries(keys: [&str; 2]) -> Vec<(PixelCacheKey, Arc<CachedPreprocessedItem>)> {
    use llm_multimodal::{EncoderDtype, EncoderInput, PixelNorm};

    let mut bytes = Vec::with_capacity(1024);
    bytes.resize(128, 0);
    let mut wire = EncoderInput::wire(
        bytes,
        EncoderDtype::BFloat16,
        vec![2, 32],
        PixelNorm {
            scale: 1.0,
            offset: [0.0; 3],
            divisor: [1.0; 3],
            channel_block: 1,
        },
    )
    .unwrap();
    assert!(wire.share_wire(1024));
    let parts = wire.split_axis0(&[1, 1]).unwrap();
    keys.into_iter()
        .zip(parts)
        .map(|(key, encoder_input)| {
            (
                pixel_cache_key(key),
                Arc::new(CachedPreprocessedItem {
                    preprocessed: PreprocessedEncoderInputs::new(
                        encoder_input,
                        vec![1],
                        vec![(1, 1)],
                    ),
                }),
            )
        })
        .collect()
}

/// Partial eviction retains the whole charge until the final slice leaves.
#[test]
fn pixel_cache_accounts_for_shared_wire_capacity_once() {
    let entries = shared_pixel_entries(["a", "b"]);
    let (_, capacity) = entries[0]
        .1
        .preprocessed
        .encoder_input
        .shared_allocation()
        .unwrap();
    let meta = entries[0].1.heap_bytes() + PIXEL_CACHE_KEY_OVERHEAD;
    let independent = pixel_cache_item(1, 8);
    let other = independent.heap_bytes() + PIXEL_CACHE_KEY_OVERHEAD;
    let cache = PixelCache::new(capacity + meta + other);
    cache.insert_batch(entries);
    assert_eq!(cache.current_bytes(), capacity + 2 * meta);
    cache.insert(pixel_cache_key("c"), independent.clone());
    assert!(cache.get(&pixel_cache_key("a")).is_none());
    assert_eq!(cache.current_bytes(), capacity + meta + other);
    cache.insert(pixel_cache_key("d"), independent);
    assert!(cache.get(&pixel_cache_key("b")).is_none());
    assert_eq!(cache.current_bytes(), 2 * other);
}

#[test]
fn shared_batch_replacement_keeps_complete_backings_and_metadata_budget() {
    let entries = shared_pixel_entries(["a", "b"]);
    let (owner, capacity) = entries[0]
        .1
        .preprocessed
        .encoder_input
        .shared_allocation()
        .unwrap();
    let entry_bytes = entries[0].1.heap_bytes() + PIXEL_CACHE_KEY_OVERHEAD;

    let too_small = PixelCache::new(capacity + entry_bytes);
    assert!(!too_small.insert_shared_batch(entries.clone(), capacity));
    assert_eq!(too_small.current_bytes(), 0);

    let cache = PixelCache::new(4 * capacity);
    assert!(cache.insert_shared_batch(entries, capacity));
    assert!(!cache.insert_shared_batch(shared_pixel_entries(["a", "c"]), capacity));
    assert_eq!(
        cache
            .get(&pixel_cache_key("a"))
            .unwrap()
            .preprocessed
            .encoder_input
            .shared_allocation(),
        Some((owner, capacity))
    );
    assert!(cache.get(&pixel_cache_key("c")).is_none());
    assert!(cache.insert_shared_batch(shared_pixel_entries(["a", "b"]), capacity));
    assert_eq!(cache.current_bytes(), capacity + 2 * entry_bytes);
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

/// The series a sample is on, as `(bytes, entries, evictions)`; `None`
/// until every one of the three families is rendered.
fn exported(handle: &metrics_exporter_prometheus::PrometheusHandle) -> Option<(f64, f64, f64)> {
    let body = handle.render();
    let sample = |family: &str| -> Option<f64> {
        body.lines()
            .find_map(|line| line.strip_prefix(&format!("{family} ")))
            .and_then(|value| value.trim().parse().ok())
    };
    Some((
        sample("smg_mm_pixel_cache_bytes")?,
        sample("smg_mm_pixel_cache_entries")?,
        sample("smg_mm_pixel_cache_evictions_total")?,
    ))
}

/// What the cache holds and what it has dropped are on `/metrics`: at
/// zero from the moment the cache exists, then after every insert.
#[test]
fn occupancy_and_evictions_are_exported() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let entry = (pixel_cache_item(1, 4096).heap_bytes() + PIXEL_CACHE_KEY_OVERHEAD) as f64;
        let cache = PixelCache::new(entry as usize * 2 + entry as usize / 2);
        assert_eq!(exported(&handle), Some((0.0, 0.0, 0.0)));

        cache.insert(pixel_cache_key("a"), pixel_cache_item(1, 4096));
        cache.insert(pixel_cache_key("b"), pixel_cache_item(2, 4096));
        assert_eq!(exported(&handle), Some((entry * 2.0, 2.0, 0.0)));

        // The third entry pushes the first one out.
        cache.insert(pixel_cache_key("c"), pixel_cache_item(3, 4096));
        assert_eq!(exported(&handle), Some((entry * 2.0, 2.0, 1.0)));
        assert!(cache.get(&pixel_cache_key("a")).is_none());

        // An entry the whole budget cannot hold is bypassed, not counted
        // as an eviction.
        cache.insert(pixel_cache_key("huge"), pixel_cache_item(4, 1 << 20));
        assert_eq!(exported(&handle), Some((entry * 2.0, 2.0, 1.0)));
    });
}

#[test]
fn pixel_cache_oversized_entry_is_bypassed() {
    let cache = PixelCache::new(32);
    cache.insert(pixel_cache_key("huge"), pixel_cache_item(1, 1024));
    assert!(cache.get(&pixel_cache_key("huge")).is_none());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.current_bytes(), 0);
}
