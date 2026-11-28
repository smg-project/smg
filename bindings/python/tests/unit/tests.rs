use config::RoutingMode;

use super::*;

#[test]
fn a_provider_backend_is_its_routing_mode_and_an_engine_backend_is_none() {
    let urls = vec!["https://provider.example".to_string()];
    let provider = |backend: BackendType| backend.provider_mode(&urls);

    assert!(matches!(
        provider(BackendType::Openai),
        Some(RoutingMode::OpenAI { worker_urls }) if worker_urls == urls
    ));
    assert!(matches!(
        provider(BackendType::Anthropic),
        Some(RoutingMode::Anthropic { worker_urls }) if worker_urls == urls
    ));
    assert!(matches!(
        provider(BackendType::Gemini),
        Some(RoutingMode::Gemini { worker_urls }) if worker_urls == urls
    ));
    for engine in [
        BackendType::Sglang,
        BackendType::Vllm,
        BackendType::Tokenspeed,
        BackendType::Trtllm,
    ] {
        assert!(
            engine.provider_mode(&urls).is_none(),
            "{engine:?} is an engine backend: the disaggregation flags select its mode"
        );
    }
}

/// An engine backend pins the wire its startup workers speak over ZMQ; no
/// backend, a provider and TensorRT-LLM leave the choice to the detection
/// step, as the CLI does without `--backend`.
#[test]
fn an_engine_backend_pins_its_wire_and_no_backend_pins_none() {
    let pin = |backend: Option<BackendType>| {
        backend
            .as_ref()
            .and_then(BackendType::startup_worker_runtime)
    };
    assert_eq!(pin(None), None);
    assert_eq!(pin(Some(BackendType::Vllm)), Some(RuntimeType::Vllm));
    assert_eq!(
        pin(Some(BackendType::Tokenspeed)),
        Some(RuntimeType::TokenSpeed)
    );
    assert_eq!(pin(Some(BackendType::Sglang)), Some(RuntimeType::Sglang));
    for unpinned in [
        BackendType::Openai,
        BackendType::Anthropic,
        BackendType::Trtllm,
        BackendType::Gemini,
    ] {
        assert_eq!(unpinned.startup_worker_runtime(), None, "{unpinned:?}");
    }
}

/// Each key file the binding read reaches the gateway under its own
/// name, and a key given as a value has no file.
#[test]
fn secret_files_keep_each_file_under_its_own_name() {
    let files = Router::secret_files(Some("/run/keys/api"), None, Some("/run/keys/control"));
    assert_eq!(
        files.api_key.as_deref(),
        Some(std::path::Path::new("/run/keys/api"))
    );
    assert!(files.tenant_api_keys.is_none(), "given as a value: no file");
    assert_eq!(
        files.control_plane_api_keys.as_deref(),
        Some(std::path::Path::new("/run/keys/control"))
    );
    let none = Router::secret_files(None, None, None);
    assert_eq!(none, secrets::SecretFiles::default());
}

/// The sources record names a file for each key a file gave, with the
/// entries read from it; a count without a file or a value names nothing.
#[test]
fn secret_sources_name_the_files_with_their_own_entry_counts() {
    use smg::server::posture::SecretSource;
    let files = Router::secret_files(None, Some("/run/keys/tenants"), Some("/run/keys/control"));
    let sources = Router::secret_sources(&files, false, 3, 2, false, false);
    assert!(
        sources.api_key.is_none(),
        "neither given nor filed: unnamed"
    );
    assert_eq!(sources.tenant_api_keys, Some((SecretSource::File, 3)));
    assert_eq!(
        sources.control_plane_api_keys,
        Some((SecretSource::File, 2))
    );
    assert!(sources.postgres_db_url.is_none() && sources.redis_url.is_none());

    let serving_only = Router::secret_files(Some("/run/keys/api"), None, None);
    let sources = Router::secret_sources(&serving_only, true, 0, 0, false, false);
    assert_eq!(sources.api_key, Some(SecretSource::File));
    assert!(
        sources.tenant_api_keys.is_none(),
        "neither given nor filed names nothing"
    );
    assert!(sources.control_plane_api_keys.is_none());
}

/// A key given as a value and no file is the launcher's in the record,
/// with its entry count, and so is the selected backend's store URL; the
/// binding does not tell flag from environment.
#[test]
fn secret_sources_name_a_value_given_key_as_the_launchers() {
    use smg::server::posture::SecretSource;
    let none = Router::secret_files(None, None, None);
    let sources = Router::secret_sources(&none, true, 3, 2, true, false);
    assert_eq!(sources.api_key, Some(SecretSource::Launcher));
    assert_eq!(sources.postgres_db_url, Some(SecretSource::Launcher));
    assert!(
        sources.redis_url.is_none(),
        "the other backend's URL is not given"
    );
    assert_eq!(sources.tenant_api_keys, Some((SecretSource::Launcher, 3)));
    assert_eq!(
        sources.control_plane_api_keys,
        Some((SecretSource::Launcher, 2))
    );
    assert!(Router::secret_sources(&none, false, 0, 0, false, false)
        .api_key
        .is_none());
}

/// A path written under a temporary directory, for the key-file tests.
fn key_file(dir: &tempfile::TempDir, name: &str, content: &str) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, content).expect("the file written");
    path.display().to_string()
}

/// A key file keyword alone is read: the shared key as its file's one
/// line, the tenant and control-plane keys as entries with blank and
/// comment lines skipped; the control-plane entries join the launcher's
/// auth config, or one with its defaults. Without a file the values
/// pass as given.
#[test]
fn key_files_are_read_by_the_binding() {
    let dir = tempfile::tempdir().expect("a directory");
    let api_key = key_file(&dir, "api-key", "file-shared-key\n");
    let tenants = key_file(
        &dir,
        "tenants",
        "# the tenants\n\nred:file-red\n  blue : file-blue  \n",
    );
    let operators = key_file(
        &dir,
        "control-plane-keys",
        "ops:Ops:admin:file-ops-key\nro:Read:user:file-ro-key\n",
    );
    let read = Router::read_key_files(
        None,
        Some(api_key.as_str()),
        vec![],
        Some(tenants.as_str()),
        None,
        Some(operators.as_str()),
    )
    .expect("the files read");
    assert_eq!(read.api_key.as_deref(), Some("file-shared-key"));
    assert_eq!(
        read.tenant_api_keys,
        vec![
            ("red".to_string(), "file-red".to_string()),
            ("blue".to_string(), "file-blue".to_string()),
        ]
    );
    let auth = read
        .control_plane_auth
        .expect("an auth config from the file alone");
    assert_eq!(auth.api_keys.len(), 2);
    assert_eq!(
        (
            auth.api_keys[0].id.as_str(),
            auth.api_keys[0].name.as_str(),
            auth.api_keys[0].key.as_str(),
            auth.api_keys[0].role.clone(),
        ),
        ("ops", "Ops", "file-ops-key", PyRole::Admin)
    );
    assert_eq!(auth.api_keys[1].role, PyRole::User);
    assert!(
        !auth.audit_enabled,
        "the defaults when the launcher built none"
    );

    let launcher = PyControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![],
        audit_enabled: true,
    };
    let read = Router::read_key_files(
        None,
        None,
        vec![],
        None,
        Some(launcher),
        Some(operators.as_str()),
    )
    .expect("the file read into the launcher's config");
    let auth = read.control_plane_auth.expect("kept");
    assert!(auth.audit_enabled, "the launcher's setting stands");
    assert_eq!(auth.api_keys.len(), 2);

    let read = Router::read_key_files(
        Some("k".to_string()),
        None,
        vec![("t".to_string(), "k2".to_string())],
        None,
        None,
        None,
    )
    .expect("no file");
    assert_eq!(read.api_key.as_deref(), Some("k"));
    assert_eq!(read.tenant_api_keys.len(), 1);
    assert!(read.control_plane_auth.is_none());
}

/// A value and a file for one key are refused as given twice; a file
/// the reader refuses, or a bad entry, is refused under the file's
/// setting and path; no refusal carries a key.
#[test]
fn a_key_given_twice_or_a_bad_key_file_is_refused() {
    let dir = tempfile::tempdir().expect("a directory");
    let api_key = key_file(&dir, "api-key", "file-shared-key\n");
    let err = Router::read_key_files(
        Some("k".to_string()),
        Some(api_key.as_str()),
        vec![],
        None,
        None,
        None,
    )
    .expect_err("given twice");
    assert_eq!(
        err,
        "api_key and api_key_file are both given; give one source"
    );
    let err = Router::read_key_files(
        None,
        None,
        vec![("t".to_string(), "k".to_string())],
        Some(api_key.as_str()),
        None,
        None,
    )
    .expect_err("given twice");
    assert!(
        err.starts_with("tenant_api_keys and tenant_api_keys_file"),
        "{err}"
    );
    let launcher = PyControlPlaneAuthConfig {
        jwt: None,
        api_keys: vec![PyApiKeyEntry {
            id: "a".to_string(),
            name: "A".to_string(),
            key: "k".to_string(),
            role: PyRole::Admin,
        }],
        audit_enabled: false,
    };
    let err = Router::read_key_files(
        None,
        None,
        vec![],
        None,
        Some(launcher),
        Some(api_key.as_str()),
    )
    .expect_err("given twice");
    assert!(
        err.starts_with("control_plane_auth.api_keys and control_plane_api_keys_file"),
        "{err}"
    );

    let empty = key_file(&dir, "empty", "");
    let err = Router::read_key_files(None, Some(empty.as_str()), vec![], None, None, None)
        .expect_err("an empty file");
    assert!(
        err.contains("api-key-file") && err.contains(&empty) && err.contains("the file is empty"),
        "{err}"
    );
    let bad = key_file(&dir, "operators", "ops:Ops:owner:file-ops-key\n");
    let err = Router::read_key_files(None, None, vec![], None, None, Some(bad.as_str()))
        .expect_err("a bad role");
    assert!(
        err.contains("control-plane-api-keys-file")
            && err.contains(&bad)
            && err.contains("the role must be 'admin' or 'user'"),
        "{err}"
    );
    assert!(!err.contains("file-ops-key"), "{err}");
}

/// The join peers parse the same from a comma list and from separate
/// values, and an empty entry is refused rather than dropped.
#[test]
fn mesh_peer_urls_split_on_commas_and_refuse_an_empty_entry() {
    let separate =
        Router::parse_mesh_peer_urls(&["10.0.0.1:1".to_string(), "10.0.0.2:2".to_string()])
            .expect("two entries");
    let comma =
        Router::parse_mesh_peer_urls(&["10.0.0.1:1,10.0.0.2:2".to_string()]).expect("a comma list");
    assert_eq!(separate, comma);
    assert_eq!(separate.len(), 2);
    for bad in [
        vec![String::new()],
        vec!["10.0.0.1:1,".to_string()],
        vec!["10.0.0.1:1".to_string(), " ".to_string()],
    ] {
        let err = Router::parse_mesh_peer_urls(&bad).expect_err("an empty entry is refused");
        assert!(err.contains("is empty"), "{bad:?}: {err}");
    }
    let err = Router::parse_mesh_peer_urls(&["not-an-address".to_string()])
        .expect_err("a bad entry is refused");
    assert!(
        err.contains("Invalid mesh peer URL 'not-an-address'"),
        "{err}"
    );
}
