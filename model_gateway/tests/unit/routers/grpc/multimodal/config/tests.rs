use std::{
    fs,
    sync::atomic::{AtomicU16, Ordering},
};

use tempfile::TempDir;

use super::*;

/// The fingerprint is hashed once per tokenizer, model, wire dtype and
/// transparency policy: the same combination reads the first value back
/// without hashing again, a change in any of the four hashes afresh, and
/// the earlier entries keep answering.
#[test]
fn image_fingerprint_is_hashed_once_per_request_shape() {
    let config = MultimodalModelConfig::new(
        serde_json::json!({"model_type": "qwen2_vl"}),
        PreProcessorConfig::default(),
        None,
    );
    let hashes = std::cell::Cell::new(0u64);
    let fingerprint = |tokenizer: &str, model: &str, dtype, transparency| {
        config
            .image_fingerprint(tokenizer, model, dtype, transparency, || {
                hashes.set(hashes.get() + 1);
                Ok(hashes.get())
            })
            .unwrap()
    };
    let keep = TransparencyPolicy::KeepAlpha;
    assert_eq!(fingerprint("tok", "model", None, keep), 1);
    assert_eq!(fingerprint("tok", "model", None, keep), 1);
    assert_eq!(hashes.get(), 1, "the second request read the memo");
    assert_eq!(
        fingerprint("tok", "model", Some(EncoderDtype::BFloat16), keep),
        2
    );
    assert_eq!(
        fingerprint("tok", "model", None, TransparencyPolicy::DropAlpha),
        3
    );
    assert_eq!(fingerprint("other-tokenizer", "model", None, keep), 4);
    assert_eq!(fingerprint("tok", "other-model", None, keep), 5);
    assert_eq!(fingerprint("tok", "model", None, keep), 1);
    assert_eq!(
        fingerprint("tok", "model", Some(EncoderDtype::BFloat16), keep),
        2
    );
    assert_eq!(hashes.get(), 5);
}

#[test]
fn a_config_builds_its_audio_processor_once_and_keeps_its_refusal() {
    use bytes::Bytes;
    use llm_multimodal::{audio::DecodedAudio, AudioClip, AudioSource, ModelMetadata};
    use llm_tokenizer::mock::MockTokenizer;

    let mock = MockTokenizer::new();
    let tokenizer = super::super::RegistryTokenizer(&mock);
    let registry = ModelRegistry::default();
    let lookup = |config: &serde_json::Value| {
        registry
            .lookup(&ModelMetadata {
                model_id: "test",
                tokenizer: &tokenizer,
                config,
            })
            .unwrap()
    };

    let omni = serde_json::json!({"model_type": "qwen3_omni_moe"});
    let spec = lookup(&omni);
    let config_160 = MultimodalModelConfig::new(
        omni.clone(),
        PreProcessorConfig::from_json(r#"{"hop_length": 160}"#).unwrap(),
        None,
    );
    let gate = std::sync::Barrier::new(4);
    let concurrent: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| {
                    gate.wait();
                    config_160.audio_processor(spec).unwrap()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let first = &concurrent[0];
    assert!(concurrent[1..].iter().all(|p| Arc::ptr_eq(first, p)));
    assert!(Arc::ptr_eq(
        first,
        &config_160.audio_processor(spec).unwrap()
    ));

    let config_320 = MultimodalModelConfig::new(
        omni,
        PreProcessorConfig::from_json(r#"{"hop_length": 320}"#).unwrap(),
        None,
    );
    let reloaded = config_320.audio_processor(spec).unwrap();
    assert!(!Arc::ptr_eq(first, &reloaded));
    let clip = [Arc::new(AudioClip::new(
        Bytes::new(),
        DecodedAudio {
            samples: vec![0.1; 16_000],
            sample_rate: 16_000,
        },
        AudioSource::InlineBytes,
        String::new(),
    ))];
    assert_ne!(
        first.preprocess(&clip).unwrap().feature_token_counts,
        reloaded.preprocess(&clip).unwrap().feature_token_counts,
    );

    for (raw, expected) in [
        (
            serde_json::json!({"model_type": "llava"}),
            "No audio processor registered for model spec: llava",
        ),
        (
            serde_json::json!({"model_type": "muse_spark"}),
            "No audio processor registered for model spec: muse_spark",
        ),
        (
            serde_json::json!({
                "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
                "sound_config": {"subsampling_factor": 4}
            }),
            "The audio processor of model spec nemotron_omni cannot be built: ",
        ),
    ] {
        let spec = lookup(&raw);
        let cfg = MultimodalModelConfig::new(raw, PreProcessorConfig::default(), None);
        let cold = cfg.audio_processor(spec).err().unwrap().to_string();
        let warm = cfg.audio_processor(spec).err().unwrap().to_string();
        assert_eq!(cold, warm);
        assert!(cold.starts_with(expected), "{cold}");
        assert!(format!("{cfg:?}").contains("AudioProcessor"));
    }
}

#[tokio::test]
async fn registry_get_or_load_reads_from_local_dir_and_caches() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("config.json"),
        r#"{"model_type":"phi3_v","image_token_index":32044}"#,
    )
    .unwrap();
    fs::write(
        tmp.path().join("preprocessor_config.json"),
        r#"{"image_processor_type":"Phi3VImageProcessor"}"#,
    )
    .unwrap();
    let source = tmp.path().to_string_lossy().into_owned();

    let reg = MultimodalConfigRegistry::new();
    let first = reg.get_or_load("tok-uuid-2", &source).await.unwrap();
    assert_eq!(first.config["model_type"].as_str(), Some("phi3_v"));

    let second = reg.get_or_load("tok-uuid-2", &source).await.unwrap();
    assert!(
        Arc::ptr_eq(&first, &second),
        "second call must hit cache and return same Arc"
    );
}

#[tokio::test]
async fn registry_get_or_load_falls_back_when_preprocessor_config_missing() {
    // Mirrors the bundle-preload behavior in try_load_multimodal_config:
    // a local dir without preprocessor_config.json must still load and
    // cache an entry using PreProcessorConfig::default().
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
    let source = tmp.path().to_string_lossy().into_owned();

    let reg = MultimodalConfigRegistry::new();
    let loaded = reg
        .get_or_load("tok-uuid-nopp", &source)
        .await
        .expect("must fall back to default preprocessor_config");
    assert_eq!(loaded.config["model_type"].as_str(), Some("llama"));
    assert!(reg.get("tok-uuid-nopp").is_some());
}

/// A Hub that answers `status` to every file request, so a checkpoint's
/// processor-config fetches fail or are refused without a network: the
/// Hub client asks for a file its cache lacks with one GET and reads the
/// status before anything else. Shared with the pipeline's tests.
pub(crate) async fn hub_answering(status: Arc<AtomicU16>) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    #[expect(
        clippy::disallowed_methods,
        reason = "the test's Hub lives as long as the test's runtime, which ends it"
    )]
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let status = status.load(Ordering::SeqCst);
            #[expect(
                clippy::disallowed_methods,
                reason = "one answer per connection, ended with the test's runtime"
            )]
            tokio::spawn(async move {
                let mut request = [0u8; 4096];
                let _ = socket.read(&mut request).await;
                let reason = if status == 404 {
                    "Not Found"
                } else {
                    "Service Unavailable"
                };
                let answer = format!(
                    "HTTP/1.1 {status} {reason}\r\ncontent-length: 0\r\n\
                         connection: close\r\n\r\n"
                );
                let _ = socket.write_all(answer.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    endpoint
}

/// A local Hub cache under `home` holding `files` of the checkpoint `repo`
/// at one commit, laid out as the Hub client keeps it (a cached file is
/// served with no request). Returns the snapshot directory the files sit in.
fn hub_cache(home: &Path, repo: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
    let commit = "0".repeat(40);
    let repo_dir = home
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")));
    let snapshot = repo_dir.join("snapshots").join(&commit);
    fs::create_dir_all(&snapshot).unwrap();
    fs::create_dir_all(repo_dir.join("refs")).unwrap();
    fs::write(repo_dir.join("refs").join("main"), commit).unwrap();
    for (name, text) in files {
        fs::write(snapshot.join(name), text).unwrap();
    }
    snapshot
}

/// `HF_HOME` and `HF_ENDPOINT` pointed at a test's cache and Hub, and put
/// back when the test ends, so a failing assertion leaves no other test on
/// this Hub. The variables are process-wide, so the tests that set them
/// take turns: the guard holds the turn until the variables are restored.
pub(crate) struct HubEnv {
    previous: Vec<(&'static str, Option<std::ffi::OsString>)>,
    _turn: tokio::sync::MutexGuard<'static, ()>,
}

static HUB_ENV_TURN: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

impl HubEnv {
    pub(crate) async fn set(home: &Path, endpoint: &str) -> Self {
        let turn = HUB_ENV_TURN.lock().await;
        let previous = ["HF_HOME", "HF_ENDPOINT"]
            .into_iter()
            .map(|name| (name, std::env::var_os(name)))
            .collect();
        std::env::set_var("HF_HOME", home);
        std::env::set_var("HF_ENDPOINT", endpoint);
        Self {
            previous,
            _turn: turn,
        }
    }
}

impl Drop for HubEnv {
    fn drop(&mut self) {
        // The variables go back before the turn is released (the fields
        // drop after this body).
        for (name, value) in self.previous.drain(..) {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

/// A processor config the Hub fails to serve is not guessed: the request
/// fails and nothing is cached, so the next request reads the file once
/// it is there, while the Hub's 404 (a checkpoint without the file) keeps
/// the default, cached as before. No network: the Hub is a loopback
/// listener and each checkpoint's `config.json` sits in a local Hub cache.
#[tokio::test]
async fn registry_get_or_load_does_not_cache_a_default_after_a_failed_fetch() {
    let home = TempDir::new().unwrap();
    let status = Arc::new(AtomicU16::new(503));
    let endpoint = hub_answering(status.clone()).await;
    let _env = HubEnv::set(home.path(), &endpoint).await;
    let config_json = r#"{"model_type":"llama"}"#;
    let reg = MultimodalConfigRegistry::new();

    let snapshot = hub_cache(home.path(), "cell/failing", &[("config.json", config_json)]);
    let failed = reg
        .get_or_load("tok-uuid-failed", "cell/failing")
        .await
        .expect_err("a 503 on preprocessor_config.json fails the request");
    assert!(
        llm_multimodal::hub::HubFetchError::is(&failed),
        "the failure is the typed one the gateway answers as a 502: {failed:#}"
    );
    let failed = format!("{failed:#}");
    assert!(
        failed.contains("preprocessor_config.json") && failed.contains("503"),
        "the error names the file and the Hub's answer: {failed}"
    );
    assert!(
        reg.get("tok-uuid-failed").is_none(),
        "nothing is cached after a failed fetch"
    );

    // The Hub is back and the checkpoint turns out to ship the file (it is
    // in the cache now, the other two answer 404): the next request reads
    // it instead of a guessed default, and keeps it.
    fs::write(
        snapshot.join("preprocessor_config.json"),
        r#"{"size":{"shortest_edge":448}}"#,
    )
    .unwrap();
    status.store(404, Ordering::SeqCst);
    let loaded = reg
        .get_or_load("tok-uuid-failed", "cell/failing")
        .await
        .expect("the file is read on the next request");
    assert_eq!(
        loaded.preprocessor_config.get_shortest_edge(),
        Some(448),
        "the checkpoint's values, not the default"
    );
    let again = reg
        .get_or_load("tok-uuid-failed", "cell/failing")
        .await
        .unwrap();
    assert!(Arc::ptr_eq(&loaded, &again), "cached once it was read");

    // A checkpoint without the file: the Hub's 404 keeps the default, cached.
    hub_cache(home.path(), "cell/absent", &[("config.json", config_json)]);
    let absent = reg
        .get_or_load("tok-uuid-absent", "cell/absent")
        .await
        .expect("a 404 is a checkpoint without the file");
    assert_eq!(absent.preprocessor_config.get_shortest_edge(), None);
    assert!(
        reg.get("tok-uuid-absent").is_some(),
        "an absent file is cached as the default, as before"
    );

    // config.json itself, on a cold checkpoint: a 503 is the same typed
    // failure, nothing cached; a 404 is a checkpoint without the file, the
    // request's error rather than the Hub's.
    status.store(503, Ordering::SeqCst);
    let failed = reg
        .get_or_load("tok-uuid-cold", "cell/cold")
        .await
        .expect_err("a 503 on config.json fails the request");
    assert!(
        llm_multimodal::hub::HubFetchError::is(&failed),
        "config.json's 503 is the typed failure: {failed:#}"
    );
    let failed = format!("{failed:#}");
    assert!(
        failed.contains("download config.json for") && failed.contains("503"),
        "the error names config.json and the Hub's answer: {failed}"
    );
    assert!(reg.get("tok-uuid-cold").is_none(), "nothing is cached");
    status.store(404, Ordering::SeqCst);
    let missing = reg
        .get_or_load("tok-uuid-missing", "cell/missing")
        .await
        .expect_err("a checkpoint without config.json fails the request");
    assert!(
        !llm_multimodal::hub::HubFetchError::is(&missing),
        "config.json's 404 is not a failure of the Hub: {missing:#}"
    );
    assert!(
        format!("{missing:#}").contains("config.json"),
        "the error names the file: {missing:#}"
    );
    assert!(reg.get("tok-uuid-missing").is_none(), "nothing is cached");

    // Without a token the Hub answers a repository that does not exist
    // with a 401: the same plain error, not a failure of the Hub.
    status.store(401, Ordering::SeqCst);
    let refused = reg
        .get_or_load("tok-uuid-refused", "cell/refused")
        .await
        .expect_err("a 401 on config.json fails the request");
    assert!(
        !llm_multimodal::hub::HubFetchError::is(&refused),
        "config.json's 401 is not a failure of the Hub: {refused:#}"
    );
    assert!(reg.get("tok-uuid-refused").is_none(), "nothing is cached");

    // A processor config of a gated checkpoint whose config.json is a
    // cache hit, without access: the same plain error, nothing cached,
    // and no default in place of a file that may exist.
    hub_cache(home.path(), "cell/gated", &[("config.json", config_json)]);
    let gated = reg
        .get_or_load("tok-uuid-gated", "cell/gated")
        .await
        .expect_err("a 401 on a processor config fails the request");
    assert!(
        !llm_multimodal::hub::HubFetchError::is(&gated),
        "a processor config's 401 is not a failure of the Hub: {gated:#}"
    );
    let gated = format!("{gated:#}");
    assert!(
        gated.contains("preprocessor_config.json") && gated.contains("401"),
        "the error names the file and the Hub's answer: {gated}"
    );
    assert!(reg.get("tok-uuid-gated").is_none(), "nothing is cached");
}

/// A source that names a path and has no `config.json` on this host fails
/// where it stands, a path that is not here (a worker's host has it) as a
/// directory that is: the client's sentence leaves the path to the chain
/// the log writes, and the Hub is not asked. The Hub's address is a
/// loopback port with no listener, so a lookup would end as a failed fetch.
#[tokio::test]
async fn registry_get_or_load_does_not_ask_the_hub_for_a_local_source() {
    let home = TempDir::new().unwrap();
    let _env = HubEnv::set(home.path(), "http://127.0.0.1:9").await;
    let model_dir = TempDir::new().unwrap();

    for (source, said) in [
        (model_dir.path().join("worker-only"), "is not on this host"),
        (model_dir.path().to_path_buf(), "no config.json in"),
    ] {
        let source = source.to_string_lossy().into_owned();
        let failed = MultimodalConfigRegistry::new()
            .get_or_load("tok-uuid-local", &source)
            .await
            .expect_err("a local source without config.json fails the request");
        assert!(
            !failed.to_string().contains(&source),
            "the client's sentence names the path: {failed:#}"
        );
        assert!(
            !llm_multimodal::hub::HubFetchError::is(&failed),
            "the Hub was asked: {failed:#}"
        );
        assert!(
            format!("{failed:#}").contains(&source),
            "the log's chain names the path: {failed:#}"
        );
        assert!(
            format!("{failed:#}").contains(said),
            "the log's sentence: {failed:#}"
        );
    }
}

/// A Hub that takes the connection and never answers holds a fetch only
/// for its bound, a processor config's on a warm checkpoint and
/// `config.json`'s on a cold one: the request fails within it as the typed
/// error naming the file and the elapsed bound, and nothing is cached. The
/// clock is paused, so the bound elapses as soon as the runtime idles on
/// the answer that never comes.
#[tokio::test(start_paused = true)]
async fn registry_get_or_load_gives_up_on_a_hub_that_never_answers() {
    let home = TempDir::new().unwrap();
    // Bound and never accepted: the connection completes into the backlog
    // and the request waits for an answer that never comes.
    let silent = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", silent.local_addr().unwrap());
    let _env = HubEnv::set(home.path(), &endpoint).await;
    hub_cache(
        home.path(),
        "cell/silent",
        &[("config.json", r#"{"model_type":"llama"}"#)],
    );
    let reg = MultimodalConfigRegistry::new();

    let failed = tokio::time::timeout(
        std::time::Duration::from_secs(300),
        reg.get_or_load("tok-uuid-silent", "cell/silent"),
    )
    .await
    .expect("the fetch's own bound ends the wait, long before this one")
    .expect_err("a fetch with no answer fails within its bound");
    assert!(
        llm_multimodal::hub::HubFetchError::is(&failed),
        "{failed:#}"
    );
    let failed = format!("{failed:#}");
    assert!(
        failed.contains("preprocessor_config.json") && failed.contains("no answer within"),
        "the error names the file and the elapsed bound: {failed}"
    );
    assert!(
        reg.get("tok-uuid-silent").is_none(),
        "nothing is cached after a fetch that ran out of its bound"
    );

    // A cold checkpoint: `config.json` itself is the first fetch, and the
    // bound covers it the same way.
    let failed = tokio::time::timeout(
        std::time::Duration::from_secs(300),
        reg.get_or_load("tok-uuid-silent-cold", "cell/silent-cold"),
    )
    .await
    .expect("the fetch's own bound ends the wait, long before this one")
    .expect_err("a fetch with no answer fails within its bound");
    assert!(
        llm_multimodal::hub::HubFetchError::is(&failed),
        "{failed:#}"
    );
    let failed = format!("{failed:#}");
    assert!(
        failed.contains("download config.json for") && failed.contains("no answer within"),
        "the error names config.json and the elapsed bound: {failed}"
    );
    assert!(
        reg.get("tok-uuid-silent-cold").is_none(),
        "nothing is cached after a fetch that ran out of its bound"
    );
    drop(silent);
}

#[test]
fn load_video_preprocessor_config_ignores_missing_video_processor_key() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("processor_config.json"),
        r#"{"image_processor":{"image_processor_type":"Qwen3VLImageProcessor"}}"#,
    )
    .unwrap();

    assert!(load_video_preprocessor_config(tmp.path()).is_none());
}

#[test]
fn load_video_preprocessor_config_reads_video_processor_key() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("processor_config.json"),
        r#"{"video_processor":{"image_processor_type":"Qwen3VLVideoProcessor","do_resize":true}}"#,
    )
    .unwrap();

    let config = load_video_preprocessor_config(tmp.path()).expect("video_processor should parse");
    assert_eq!(
        config.image_processor_type.as_deref(),
        Some("Qwen3VLVideoProcessor")
    );
    assert_eq!(config.do_resize, Some(true));
}

#[test]
fn load_image_preprocessor_config_merges_the_feature_extractor_section() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("processor_config.json"),
        r#"{"feature_extractor":{"feature_extractor_type":"AudioFeatureExtractor","sampling_rate":24000,"samples_per_token":1920,"padding_value":0.0},
                "image_processor":{"image_processor_type":"NestedImageProcessor","do_resize":true}}"#,
    )
    .unwrap();

    let nested = load_image_preprocessor_config(tmp.path())
        .expect("the image section with the extractor section merged in");
    assert_eq!(
        nested.image_processor_type.as_deref(),
        Some("NestedImageProcessor")
    );
    assert_eq!(nested.do_resize, Some(true));
    assert_eq!(nested.get_extra::<usize>("sampling_rate"), Some(24_000));
    assert_eq!(nested.get_extra::<usize>("samples_per_token"), Some(1_920));
    assert_eq!(
        nested
            .get_extra::<String>("feature_extractor_type")
            .as_deref(),
        Some("AudioFeatureExtractor")
    );

    // The image config's own keys win over the extractor section's.
    fs::write(
        tmp.path().join("preprocessor_config.json"),
        r#"{"image_processor_type":"LegacyImageProcessor","sampling_rate":16000}"#,
    )
    .unwrap();
    let legacy = load_image_preprocessor_config(tmp.path()).expect("legacy image config");
    assert_eq!(
        legacy.image_processor_type.as_deref(),
        Some("LegacyImageProcessor")
    );
    assert_eq!(legacy.get_extra::<usize>("sampling_rate"), Some(16_000));
    assert_eq!(legacy.get_extra::<usize>("samples_per_token"), Some(1_920));

    // An extractor section without any image config: defaults plus the section.
    fs::remove_file(tmp.path().join("preprocessor_config.json")).unwrap();
    fs::write(
        tmp.path().join("processor_config.json"),
        r#"{"feature_extractor":{"sampling_rate":24000}}"#,
    )
    .unwrap();
    let audio_only = load_image_preprocessor_config(tmp.path())
        .expect("the extractor section alone is a config");
    assert_eq!(audio_only.image_processor_type, None);
    assert_eq!(audio_only.get_extra::<usize>("sampling_rate"), Some(24_000));

    // No section: nothing is merged and a missing image config stays missing.
    fs::write(
        tmp.path().join("processor_config.json"),
        r#"{"video_processor":{"image_processor_type":"VideoProcessor"}}"#,
    )
    .unwrap();
    assert!(load_image_preprocessor_config(tmp.path()).is_none());
}

#[test]
fn load_image_preprocessor_config_preserves_legacy_precedence_and_falls_back() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("processor_config.json"),
        r#"{"image_processor":{"image_processor_type":"NestedImageProcessor"}}"#,
    )
    .unwrap();
    let legacy_path = tmp.path().join("preprocessor_config.json");
    fs::write(
        &legacy_path,
        r#"{"image_processor_type":"LegacyImageProcessor"}"#,
    )
    .unwrap();

    let legacy =
        load_image_preprocessor_config(tmp.path()).expect("legacy image config should parse");
    assert_eq!(
        legacy.image_processor_type.as_deref(),
        Some("LegacyImageProcessor")
    );

    fs::write(&legacy_path, "{malformed-json").unwrap();
    let nested = load_image_preprocessor_config(tmp.path())
        .expect("malformed legacy config should fall back to nested image_processor");
    assert_eq!(
        nested.image_processor_type.as_deref(),
        Some("NestedImageProcessor")
    );

    fs::remove_file(legacy_path).unwrap();
    let nested_without_legacy = load_image_preprocessor_config(tmp.path())
        .expect("missing legacy config should fall back to nested image_processor");
    assert_eq!(
        nested_without_legacy.image_processor_type.as_deref(),
        Some("NestedImageProcessor")
    );
}

fn native_policy() -> serde_json::Value {
    serde_json::json!({
        "patch_size":40,
        "temporal_patch_size":2,
        "include_extra_column":false,
        "rescale_image_frac":null,
        "rescale_image_max_upscaled_long_edge":null,
        "stretch_to_padded_grid":false,
        "overlay_rgba_images_on_checkerboard":true,
        "requires_image_grid_thw":true
    })
}

#[tokio::test]
async fn registry_loads_native_image_preprocessing_policy() {
    use llm_multimodal::vision::processors::Inkling2ImageProcessor;

    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("config.json"),
        r#"{"model_type":"inkling2"}"#,
    )
    .unwrap();
    fs::write(
        tmp.path().join("image_preprocessing.json"),
        native_policy().to_string(),
    )
    .unwrap();
    // The native file must win over a generic processor policy.
    fs::write(
        tmp.path().join("preprocessor_config.json"),
        r#"{"do_resize":true}"#,
    )
    .unwrap();
    let registry = MultimodalConfigRegistry::new();
    let loaded = registry
        .get_or_load("native-policy", tmp.path().to_str().unwrap())
        .await
        .unwrap();
    Inkling2ImageProcessor::validate_config(&loaded.preprocessor_config).unwrap();
}

#[tokio::test]
async fn registry_loads_released_processor_config_section() {
    use llm_multimodal::vision::processors::Inkling2ImageProcessor;

    // The released checkpoint ships only processor_config.json.
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("config.json"),
        r#"{"model_type":"inkling2"}"#,
    )
    .unwrap();
    fs::write(
        tmp.path().join("processor_config.json"),
        serde_json::json!({
            "image_processor": {
                "do_convert_rgb": true,
                "do_normalize": true,
                "do_rescale": true,
                "image_mean": [0.5, 0.5, 0.5],
                "image_processor_type": "Inkling2ImageProcessor",
                "image_std": [0.5, 0.5, 0.5],
                "rescale_factor": 0.00392156862745098,
                "size": {"height": 40, "width": 40}
            },
            "image_token": "<|unused_200054|>",
            "processor_class": "Inkling2Processor"
        })
        .to_string(),
    )
    .unwrap();
    let registry = MultimodalConfigRegistry::new();
    let loaded = registry
        .get_or_load("released-policy", tmp.path().to_str().unwrap())
        .await
        .unwrap();
    Inkling2ImageProcessor::validate_config(&loaded.preprocessor_config).unwrap();
}

#[test]
fn missing_or_malformed_native_policy_cannot_use_generic_defaults() {
    use llm_multimodal::vision::processors::Inkling2ImageProcessor;

    let tmp = TempDir::new().unwrap();
    let generic = tmp.path().join("preprocessor_config.json");
    fs::write(&generic, r#"{"do_resize":true}"#).unwrap();
    let missing = load_image_preprocessor_config(tmp.path()).unwrap();
    assert!(Inkling2ImageProcessor::validate_config(&missing).is_err());
    let native = tmp.path().join("image_preprocessing.json");
    fs::write(&native, "{malformed").unwrap();
    assert!(load_image_preprocessor_config(tmp.path()).is_none());
    assert!(Inkling2ImageProcessor::validate_config(&PreProcessorConfig::default()).is_err());
    let mut policy = native_policy();
    policy["patch_size"] = serde_json::json!(39);
    fs::write(&native, policy.to_string()).unwrap();
    let wrong = load_image_preprocessor_config(tmp.path()).unwrap();
    assert!(Inkling2ImageProcessor::validate_config(&wrong).is_err());
}

#[tokio::test]
async fn registry_remove_drops_cached_entry() {
    let reg = MultimodalConfigRegistry::new();
    let cfg = Arc::new(MultimodalModelConfig::new(
        serde_json::json!({"model_type":"phi3_v"}),
        PreProcessorConfig::from_json(r#"{"image_processor_type":"Phi3VImageProcessor"}"#).unwrap(),
        None,
    ));
    reg.insert("tok-uuid-rm".to_string(), cfg.clone());
    assert!(reg.get("tok-uuid-rm").is_some());

    let removed = reg.remove("tok-uuid-rm").expect("remove returns the entry");
    assert!(Arc::ptr_eq(&removed, &cfg));
    assert!(reg.get("tok-uuid-rm").is_none());
    assert!(reg.remove("tok-uuid-rm").is_none());
}

#[tokio::test]
async fn registry_get_or_load_hits_preloaded_entry_without_touching_source() {
    // Regression test for the IGW bug: preload populates the registry
    // under the tokenizer UUID; `get_or_load` must return it without
    // consulting `tokenizer_source` (which in IGW points to an
    // unreachable worker-only path).
    let reg = MultimodalConfigRegistry::new();
    let cfg = Arc::new(MultimodalModelConfig::new(
        serde_json::json!({"model_type":"phi3_v"}),
        PreProcessorConfig::from_json(r#"{"image_processor_type":"Phi3VImageProcessor"}"#).unwrap(),
        None,
    ));
    reg.insert("tok-uuid-3".to_string(), cfg.clone());

    let bad_source = "/nonexistent/worker-only/path-that-would-fail";
    let got = reg
        .get_or_load("tok-uuid-3", bad_source)
        .await
        .expect("preloaded entry must be returned without touching source");
    assert!(Arc::ptr_eq(&got, &cfg));
}

/// The resolved settings, not the environment, decide the placement mode
/// and the pixel cache the components come up with.
#[test]
fn components_take_placement_and_pixel_cache_from_the_settings() {
    use super::super::settings::{Setting, SettingSource};

    let settings = MultimodalSettings {
        processing: Setting {
            value: MmProcessingMode::Worker,
            source: SettingSource::Flag,
        },
        ..MultimodalSettings::default()
    };
    let components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &settings,
    )
    .unwrap();
    assert_eq!(components.processing, MmProcessingMode::Worker);
    assert!(
        components.pixel_cache.is_none(),
        "0 MiB keeps the cache off"
    );
}

/// The shared router connector applies configured budgets to real stalled
/// image, audio and video downloads, without needing any engine or decoder.
#[tokio::test]
async fn components_enforce_media_fetch_timeouts() {
    use std::time::Duration;

    use llm_multimodal::{ImageFetchConfig, MediaFault, MediaSource, VideoFetchConfig};

    use crate::config::RouterConfig;

    // Leave connections queued without an HTTP response. The listener lives
    // until all fetches finish, so connection refusal cannot pass this test.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/media", listener.local_addr().unwrap());
    let settings = MultimodalSettings::resolve_with(
        &RouterConfig {
            mm_fetch_timeout_ms: Some(50),
            mm_video_fetch_timeout_ms: Some(100),
            mm_allow_private_media_hosts: true,
            ..RouterConfig::default()
        },
        |_| Ok(None),
    )
    .unwrap();
    let components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &settings,
    )
    .unwrap();
    let connector = &components.media_connector;
    for modality in ["image", "audio", "video"] {
        let source = MediaSource::Url(url.to_string().into());
        let fetch = async {
            match modality {
                "image" => connector
                    .fetch_image(source, ImageFetchConfig::default())
                    .await
                    .map(|_| ()),
                "audio" => connector.fetch_audio(source).await.map(|_| ()),
                _ => connector
                    .fetch_video(source, VideoFetchConfig::default())
                    .await
                    .map(|_| ()),
            }
        };
        let error = tokio::time::timeout(Duration::from_secs(2), fetch)
            .await
            .expect("configured budget must expire before the default")
            .unwrap_err();
        assert_eq!(error.fault(), MediaFault::Transient, "{modality}: {error}");
        assert!(
            error.to_string().contains("timed out"),
            "{modality}: {error}"
        );
    }
}

/// Under `require` the components come up exactly when the library did:
/// a host without it is refused with the library and the gauge named, a
/// host with it starts as before.
#[test]
fn a_required_turbojpeg_is_refused_exactly_when_the_library_is_missing() {
    use llm_multimodal::jpeg_turbo::{turbojpeg_available, TurboJpegPolicy};

    use super::super::settings::{Setting, SettingSource};

    let settings = MultimodalSettings {
        turbojpeg: Setting {
            value: TurboJpegPolicy::Require,
            source: SettingSource::Flag,
        },
        ..MultimodalSettings::default()
    };
    let result = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &settings,
    );
    if turbojpeg_available() {
        assert!(result.is_ok(), "{:#}", result.err().unwrap());
    } else {
        let error = format!("{:#}", result.err().expect("refused without the library"));
        assert!(error.contains("libturbojpeg.so.0"), "{error}");
        assert!(error.contains("smg_mm_turbojpeg_available"), "{error}");
    }
}

fn components(max_inflight_bytes: Option<usize>) -> Result<MultimodalComponents> {
    MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        max_inflight_bytes,
        &MultimodalSettings::default(),
    )
}

/// A budget too small to admit anything would turn every request carrying
/// media away. Read as "no limit" it would do the opposite instead, so it
/// is refused and the operator is told which one to ask for.
#[test]
fn a_budget_that_admits_nothing_stops_startup() {
    assert!(components(None).unwrap().inflight.is_none());
    assert!(components(Some(0)).is_err());
    assert!(components(Some(1)).is_err());

    let sized = components(Some(8192)).unwrap();
    let inflight = sized.inflight.expect("a usable budget is kept");
    assert_eq!(inflight.budget_bytes(), 8192);
}
