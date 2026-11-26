use super::*;

#[test]
fn a_hub_cache_snapshot_names_its_model_id() {
    let snapshot = Path::new(
        "/home/u/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/0123abc",
    );
    let deepseek = |inside: &str| {
        Some((
            "deepseek-ai/DeepSeek-V4-Flash".to_string(),
            inside.to_string(),
        ))
    };
    assert_eq!(hub_snapshot(snapshot), deepseek(""));
    assert_eq!(
        hub_snapshot(&snapshot.join("tokenizer.json")),
        deepseek("tokenizer.json")
    );
    assert_eq!(
        hub_snapshot(&snapshot.join("original/tokenizer.model")),
        deepseek("original/tokenizer.model")
    );
    assert_eq!(
        hub_snapshot(Path::new("/cache/models--gpt2/snapshots/abc")),
        Some(("gpt2".to_string(), String::new()))
    );
    assert_eq!(hub_snapshot(Path::new("/models/Qwen3-8B")), None);
    assert_eq!(hub_snapshot(Path::new("/models/snapshots/abc")), None);
    // `..` counts by name: out of the snapshot, no Hub id; into another
    // revision, that one's.
    assert_eq!(
        hub_snapshot(&snapshot.join("../../../../models/my-ckpt")),
        None
    );
    assert_eq!(hub_snapshot(&snapshot.join("../4567def")), deepseek(""));
}

#[test]
fn a_missing_chat_template_is_refused() {
    let Err(Failure::Input(error)) = resolve_chat_template("/no/such/template.jinja") else {
        panic!("refused as an input error");
    };
    assert_eq!(
        error,
        "chat template file not found: /no/such/template.jinja"
    );
}

#[test]
fn a_checkpoint_card_is_read_from_its_config() {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = dir.path().join("config.json");
    fs::write(
        &config,
        r#"{"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss"}"#,
    )
    .expect("config");
    let card = (
        vec!["GptOssForCausalLM".to_string()],
        Some("gpt_oss".to_string()),
    );
    assert_eq!(read_card(dir.path()), card);
    let file = dir.path().join("tokenizer.json");
    fs::write(&file, "{}").expect("tokenizer file");
    assert_eq!(read_card(&file), card);
    fs::write(&config, "not json").expect("config");
    assert_eq!(read_card(dir.path()), (Vec::new(), None));
    // A name that is no path reads no config.json, not even the one in
    // the directory it would sit in.
    fs::write(&config, r#"{"architectures": ["GptOssForCausalLM"]}"#).expect("config");
    assert_eq!(read_card(&dir.path().join("gpt-4o")), (Vec::new(), None));
}

#[test]
fn the_fingerprint_follows_the_tokenizer_files_and_nothing_else() {
    let dir = tempfile::tempdir().expect("temp dir");
    let source = dir.path().to_str().expect("utf-8");
    fs::write(dir.path().join("tokenizer.json"), "{}").expect("tokenizer");
    fs::write(dir.path().join("config.json"), "{}").expect("config");
    let digest = |model: Option<&str>| fingerprint(source, None, model).ok().expect("digest");
    let first = digest(Some("m"));
    assert_eq!(digest(Some("m")), first);
    assert_ne!(digest(Some("n")), first);
    // Weights and other files are not the tokenizer's.
    fs::write(dir.path().join("model.safetensors"), "weights").expect("weights");
    fs::write(dir.path().join("README.md"), "readme").expect("readme");
    assert_eq!(digest(Some("m")), first);
    // A tokenizer file edited in place, or one added, is.
    fs::write(dir.path().join("config.json"), r#"{"eos_token_id": 2}"#).expect("config");
    let edited = digest(Some("m"));
    assert_ne!(edited, first);
    fs::create_dir(dir.path().join("encoding")).expect("encoding dir");
    fs::write(dir.path().join("encoding").join("encoding.py"), "x").expect("encoding");
    assert_ne!(digest(Some("m")), edited);
    // A file source is fingerprinted by its directory's files and by
    // itself, whatever its name: an in-place edit of `tok_v2.json` changes
    // it, and a file source is not its directory.
    let file = |name: &str| {
        let path = dir.path().join(name);
        fingerprint(path.to_str().expect("utf-8"), None, Some("m"))
            .ok()
            .expect("digest")
    };
    let tokenizer_json = file("tokenizer.json");
    assert_ne!(tokenizer_json, digest(Some("m")));
    fs::write(dir.path().join("tok_v2.json"), "{}").expect("tokenizer");
    let tok_v2 = file("tok_v2.json");
    assert_ne!(tok_v2, tokenizer_json);
    fs::write(dir.path().join("tok_v2.json"), r#"{"edited": 1}"#).expect("tokenizer");
    assert_ne!(file("tok_v2.json"), tok_v2);
    assert_eq!(file("tokenizer.json"), tokenizer_json);
}

#[test]
fn media_parts_are_refused_and_text_parts_are_not() {
    let messages: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
    ]))
    .expect("messages");
    assert!(refuse_media(&messages[..1]).is_ok());
    let Err(Failure::Input(error)) = refuse_media(&messages) else {
        panic!("refused as an input error");
    };
    assert!(error.starts_with("messages[1] has image_url content;"));
}

#[test]
fn a_request_renders_under_the_tokenizer_model() {
    let request = r#"{"model": "alias", "messages": [{"role": "user", "content": "hi"}]}"#;
    let parsed = parse_request(request, Some("served")).ok().expect("parsed");
    assert_eq!(parsed.model, "served");
    let parsed = parse_request(request, None).ok().expect("parsed");
    assert_eq!(parsed.model, "alias");
    let unnamed = r#" { "messages": [{"role": "user", "content": "hi"}]}"#;
    let parsed = parse_request(unnamed, Some("served")).ok().expect("parsed");
    assert_eq!(parsed.model, "served");
    let Err(Failure::Input(error)) = parse_request(unnamed, None) else {
        panic!("refused as an input error");
    };
    assert!(error.starts_with("the request names no model"));
    assert_eq!(with_model("{}", "m"), r#"{"model":"m"}"#);
    assert_eq!(with_model(" { } ", "m"), r#"{"model":"m" } "#);
}

#[test]
fn a_duplicate_member_is_refused_with_or_without_a_model() {
    for request in [
        r#"{"model": "m", "messages": [], "messages": []}"#,
        r#"{"messages": [], "messages": []}"#,
    ] {
        let Err(Failure::Input(error)) = parse_request(request, Some("served")) else {
            panic!("refused as an input error: {request}");
        };
        assert!(error.contains("duplicate field `messages`"), "{error}");
    }
}

#[test]
fn a_panic_is_an_internal_failure() {
    let Err(Failure::Internal(error)) = caught::<()>(|| panic!("boom")) else {
        panic!("caught as an internal failure");
    };
    assert_eq!(error, "SMG's tokenizer panicked: boom");
}

#[test]
fn warnings_are_captured_for_the_work_that_logs_them() {
    let ((), warnings) = captured(|| {
        tracing::warn!(tool = "search", bytes = 3, "arguments rendered as {{}}");
        tracing::info!("not a warning");
    });
    assert_eq!(
        warnings,
        ["arguments rendered as {} (tool=search, bytes=3)"]
    );
    let ((), outside) = captured(|| ());
    assert!(outside.is_empty());
}

#[test]
fn a_count_left_by_another_process_counts_for_nothing() {
    let pid = std::process::id();
    assert_eq!(count_for(u64::from(pid) << 32 | 2, pid), 2);
    assert_eq!(count_for(u64::from(pid + 1) << 32 | 2, pid), 0);
    // A count taken over from another process starts from none, and
    // never goes below it.
    let counter = AtomicU64::new(u64::from(pid + 1) << 32 | 2);
    count_up(&counter, pid);
    assert_eq!(count_for(counter.load(Ordering::SeqCst), pid), 1);
    count_down(&counter, pid);
    count_down(&counter, pid);
    assert_eq!(counter.load(Ordering::SeqCst), u64::from(pid) << 32);
}
