use std::collections::HashMap;

use smg_grand_central::document::PolicySpec;
use tempfile::tempdir;

use super::*;

#[test]
fn a_document_that_cannot_be_read_stops_the_build_by_its_path() {
    let dir = tempdir().expect("a temporary directory");
    let path = dir.path().join("missing.yaml");
    let path = path.to_str().expect("a UTF-8 path");
    let io_error = std::fs::read_to_string(path).expect_err("the file is absent");
    let refusal = RouterConfig::builder()
        .grand_central(true)
        .maybe_policy_spec_path(Some(path))
        .build()
        .expect_err("an unreadable file is refused");
    assert!(matches!(
        refusal,
        ConfigError::ValidationFailed { reason } if reason.contains(path)
            && reason.contains(&io_error.to_string())
    ));
}

#[test]
fn a_document_the_reader_refuses_stops_the_build_in_its_words() {
    let dir = tempdir().expect("a temporary directory");
    let path = dir.path().join("invalid.yaml");
    let path = path.to_str().expect("a UTF-8 path");
    let text = "apiVersion: smg.policy/v1\nprofiles: {}\nunknown: true\n";
    std::fs::write(path, text).expect("the document is written");
    let read_error = Format::Yaml
        .read(text.as_bytes())
        .expect_err("unknown field");
    let refusal = RouterConfig::builder()
        .grand_central(true)
        .maybe_policy_spec_path(Some(path))
        .build()
        .expect_err("an invalid document is refused");
    let ConfigError::ValidationFailed { reason } = refusal else {
        panic!("a refused document fails the validation: {refusal:?}");
    };
    assert_eq!(reason, format!("Grand Central: {path}: {read_error}"));
    // The refusal names the field by its path.
    assert!(
        reason.contains(": not a policy spec in yaml: unknown: unknown field `unknown`"),
        "{reason}"
    );
}

/// The refusal of `text`, supplied as the file `name`, in the words the
/// start prints behind the file's path.
fn refusal_of(name: &str, text: &str) -> String {
    let dir = tempdir().expect("a temporary directory");
    let path = dir.path().join(name);
    let path = path.to_str().expect("a UTF-8 path");
    std::fs::write(path, text).expect("the document is written");
    let refusal = RouterConfig::builder()
        .grand_central(true)
        .maybe_policy_spec_path(Some(path))
        .build_with_validation(false)
        .expect_err("the document is refused");
    let ConfigError::ValidationFailed { reason } = refusal else {
        panic!("a refused document fails the validation: {refusal:?}");
    };
    reason
        .strip_prefix(&format!("Grand Central: {path}: "))
        .unwrap_or_else(|| panic!("the refusal names the file first: {reason}"))
        .to_owned()
}

// Five kinds of document a typed YAML read took and the Grand Central crate's
// read refuses. Each refusal names its place and what a writer does.

#[test]
fn a_bare_number_where_a_field_takes_text_is_refused_with_its_cure() {
    let text = "apiVersion: smg.policy/v1\nprofiles:\n  default: {pick: random}\n\
                    bindings:\n- match: {model: 123}\n  profile: default\n";
    assert_eq!(
        refusal_of("routing.yaml", text),
        "not a policy spec in yaml: bindings[0].match.model: invalid type: integer `123`, \
             expected a string: text that reads as a number, a boolean or a null is written in quotes"
    );
}

#[test]
fn a_key_with_nothing_under_it_is_refused_with_its_cure() {
    let text = "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    score:\n    pick: random\n";
    assert_eq!(
        refusal_of("routing.yaml", text),
        "not a policy spec in yaml: profiles.default.score: invalid type: null, \
             expected a sequence: a key with nothing under it is a null: take the key out, \
             or write [] or {}"
    );
}

#[test]
fn a_tag_is_refused_by_its_name_and_its_place() {
    let text = "apiVersion: smg.policy/v1\nprofiles:\n  default: {pick: random}\n\
                    bindings:\n- match: {model: !mine batch}\n  profile: default\n";
    assert_eq!(
        refusal_of("routing.yaml", text),
        "not a policy spec in yaml: bindings[0].match.model: the tag `!mine` is none JSON writes: \
             take the tag out at line 5 column 18"
    );
}

#[test]
fn aliases_past_the_readers_bounds_are_refused_by_the_bound() {
    // One list of a thousand scorers, shared by a thousand profiles, each
    // named by a binding.
    let shared = vec!["least-load"; 1000].join(", ");
    let sharing: Vec<String> = (0..1000)
        .map(|profile| format!("  p{profile}:\n    score: *s\n    pick: best\n"))
        .collect();
    let naming: Vec<String> = (0..1000)
        .map(|profile| format!("- match: {{model: m{profile}}}\n  profile: p{profile}\n"))
        .collect();
    let text = format!(
        "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    score: &s [{shared}]\n    pick: best\n{}bindings:\n{}",
        sharing.concat(),
        naming.concat()
    );
    let refusal = refusal_of("routing.yaml", &text);
    assert!(
        refusal.contains(
            "the document reads as more than 2 bytes of text for each byte of the size bound \
                 (131072): an alias reads as its anchor's value at each use at line "
        ),
        "{refusal}"
    );
}

#[test]
fn a_yaml_document_in_a_file_named_json_is_refused_by_the_name() {
    let text = "apiVersion: smg.policy/v1\nprofiles:\n  default: {pick: random}\n";
    assert_eq!(
        refusal_of("routing.json", text),
        "not a policy spec in json: expected value at line 1 column 1: \
             a file named .json is read as JSON, any other name as YAML"
    );
}

/// A whole number past 64 bits where a fraction stands reads as a
/// fraction, in a YAML file as in a JSON file.
#[test]
fn a_whole_number_past_64_bits_reads_as_a_fraction_in_both_forms() {
    let dir = tempdir().expect("a temporary directory");
    let forms = [
        (
            "routing.yaml",
            "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    \
                 score: [{least-load: {weight: 18446744073709551616}}]\n    pick: best\n",
        ),
        (
            "routing.json",
            r#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"score":[{"least-load":{"weight":18446744073709551616}}],"pick":"best"}}}"#,
        ),
    ];
    let [yaml, json] = forms.map(|(file, text)| {
        let path = dir.path().join(file);
        let path = path.to_str().expect("a UTF-8 path");
        std::fs::write(path, text).expect("the document is written");
        RouterConfig::builder()
            .grand_central(true)
            .maybe_policy_spec_path(Some(path))
            .build_with_validation(false)
            .expect("the document reads")
            .policy_spec
    });
    assert!(yaml.is_some(), "the YAML file reads");
    assert_eq!(yaml, json);
}

/// A file's form is read off its name, and what the print of a
/// document writes reads back as that document.
#[test]
fn a_json_file_and_a_printed_document_read_as_the_document() {
    let dir = tempdir().expect("a temporary directory");
    for name in ["random", "cache_aware", "prefix_hash"] {
        let document = PolicySpec::preset(name).expect("a preset");
        let forms = [
            (
                "printed.yaml",
                serde_yaml::to_string(&document).expect("prints"),
            ),
            (
                "written.json",
                serde_json::to_string(&document).expect("writes"),
            ),
        ];
        for (file, text) in forms {
            let path = dir.path().join(file);
            let path = path.to_str().expect("a UTF-8 path");
            std::fs::write(path, text).expect("the document is written");
            let config = RouterConfig::builder()
                .grand_central(true)
                .maybe_policy_spec_path(Some(path))
                .build_with_validation(false)
                .expect("the document reads");
            assert_eq!(
                config.policy_spec.as_ref(),
                Some(&document),
                "{name} as {file}"
            );
        }
    }
}

#[test]
fn a_document_with_the_switch_off_is_refused() {
    let dir = tempdir().expect("a temporary directory");
    let path = dir.path().join("routing.yaml");
    let path = path.to_str().expect("a UTF-8 path");
    std::fs::write(
        path,
        "apiVersion: smg.policy/v1\nprofiles: {default: {pick: random}}\n",
    )
    .expect("the document is written");
    let refusal = RouterConfig::builder()
        .igw(true)
        .maybe_policy_spec_path(Some(path))
        .build()
        .expect_err("the document needs the controller");
    assert!(matches!(
        refusal,
        ConfigError::InvalidValue { field, value, reason }
            if field == "policy_spec" && value == path
                && reason == "a policy document is run by Grand Central: set --grand-central on"
    ));
}

#[test]
fn a_document_given_without_its_path_needs_the_switch_too() {
    // A host hands the document itself, and no launcher stands before it.
    let document = PolicySpec::preset("random").expect("the preset");
    let config = RouterConfig {
        enable_igw: true,
        policy_spec: Some(document),
        ..RouterConfig::default()
    };
    assert!(matches!(
        config.validate().expect_err("the document needs the controller"),
        ConfigError::InvalidValue { field, value, reason }
            if field == "policy_spec" && value == "the given document"
                && reason == "a policy document is run by Grand Central: set --grand-central on"
    ));
}

#[test]
fn a_path_with_no_loaded_document_is_refused() {
    let config = RouterConfig {
        enable_igw: true,
        grand_central: true,
        policy_spec_path: Some("routing.yaml".to_owned()),
        ..RouterConfig::default()
    };
    assert!(matches!(
        config.validate().expect_err("the document was never read"),
        ConfigError::InvalidValue { field, value, reason }
            if field == "policy_spec" && value == "routing.yaml"
                && reason.starts_with("the policy document of this path is not loaded")
    ));
}

/// A policy document of `len` bytes: a small one, then one YAML comment
/// that fills the file to the length.
fn document_of(len: usize) -> String {
    let document = "apiVersion: smg.policy/v1\nprofiles: {default: {pick: random}}\n";
    let filler = "x".repeat(len - document.len() - 2);
    format!("{document}#{filler}\n")
}

#[test]
fn a_document_at_the_size_bound_is_read() {
    let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    let dir = tempdir().expect("a temporary directory");
    let path = dir.path().join("bound.yaml");
    let path = path.to_str().expect("a UTF-8 path");
    std::fs::write(path, document_of(bound)).expect("the document is written");
    let config = RouterConfig::builder()
        .grand_central(true)
        .maybe_policy_spec_path(Some(path))
        .build_with_validation(false)
        .expect("a document at the bound is read");
    assert!(config.policy_spec.is_some());
}

#[test]
fn a_document_over_the_size_bound_stops_the_build_by_its_size() {
    let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    let dir = tempdir().expect("a temporary directory");
    let path = dir.path().join("over.yaml");
    let path = path.to_str().expect("a UTF-8 path");
    // A document the reader would take, five bytes past the bound: the
    // refusal states the file's whole size, so the file was not read.
    let over = bound + 5;
    std::fs::write(path, document_of(over)).expect("the document is written");
    let refusal = RouterConfig::builder()
        .grand_central(true)
        .maybe_policy_spec_path(Some(path))
        .build_with_validation(false)
        .expect_err("a document over the bound is refused");
    assert!(matches!(
        refusal,
        ConfigError::ValidationFailed { reason } if reason == format!(
            "Grand Central: {path}: could not read the policy document (the document is {over} bytes; at most {bound})"
        )
    ));
}

#[test]
fn a_source_that_holds_more_than_it_states_is_read_one_byte_past_the_bound() {
    let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
    // A source that states no size and holds ten bytes past the bound.
    let source = std::io::repeat(b'#').take(MAX_DOCUMENT_BYTES + 10);
    let read = bounded_document(0, source).expect("the read stops by itself");
    assert_eq!(read.len(), bound + 1);
    // The Grand Central crate's read refuses what reached past the bound.
    let refusal = Format::Yaml
        .read(&read)
        .expect_err("the source is past the bound");
    assert_eq!(
        refusal.to_string(),
        format!("the document is {} bytes; at most {bound}", bound + 1)
    );
}

/// Test complex routing mode helper method
#[test]
fn test_builder_prefill_decode_mode() {
    let config = RouterConfigBuilder::new()
        .prefill_decode_mode(
            vec![("http://prefill:8000".to_string(), Some(8001))],
            vec!["http://decode:8000".to_string()],
        )
        .power_of_two_policy()
        .build()
        .unwrap();

    assert!(config.mode.is_pd_mode());
    assert_eq!(config.mode.worker_count(), 2);
}

/// Test complex policy helper method with multiple parameters
#[test]
fn test_builder_cache_aware_policy() {
    let config = RouterConfigBuilder::new()
        .regular_mode(vec!["http://worker1:8000".to_string()])
        .cache_aware_policy(0.8, 10, 1.5, 300, 1000)
        .build()
        .unwrap();

    match config.policy {
        PolicyConfig::CacheAware {
            cache_threshold, ..
        } => {
            assert!((cache_threshold - 0.8).abs() < 0.0001);
        }
        _ => panic!("Expected CacheAware policy"),
    }
}

#[test]
fn test_builder_storage_context_headers_round_trip() {
    let headers = HashMap::from([("x-tenant-id".to_string(), "tenant_id".to_string())]);

    let config = RouterConfigBuilder::new()
        .regular_mode(vec!["http://worker1:8000".to_string()])
        .storage_context_headers(headers.clone())
        .build()
        .unwrap();

    assert_eq!(config.storage_context_headers, headers);
}
