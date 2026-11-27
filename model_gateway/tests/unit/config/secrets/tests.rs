use std::io::Write;

use super::*;

const KEY: SecretFile = SecretFile {
    what: "the shared API key",
    field: "api-key-file",
    entry: "",
};
const KEYS: SecretFile = SecretFile {
    what: "the tenant API keys",
    field: "tenant-api-keys-file",
    entry: "tenant_id:key",
};

fn file(content: &str) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
    file.write_all(content.as_bytes()).expect("written");
    file
}

fn refusal(err: ConfigError) -> (String, String, String) {
    match err {
        ConfigError::InvalidValue {
            field,
            value,
            reason,
        } => (field, value, reason),
        other => panic!("not a value refusal: {other}"),
    }
}

#[test]
fn a_value_file_gives_its_one_line_without_the_newline() {
    let f = file("s3cret-value\n");
    assert_eq!(KEY.value(f.path()).expect("the value"), "s3cret-value");
}

#[test]
fn a_value_file_with_two_lines_is_refused_by_field_and_path_alone() {
    let f = file("s3cret-one\ns3cret-two\n");
    let (field, value, reason) = refusal(KEY.value(f.path()).expect_err("refused"));
    assert_eq!(field, "api-key-file");
    assert_eq!(value, f.path().display().to_string());
    assert!(reason.contains("more than one line"), "{reason}");
    assert!(!reason.contains("s3cret"), "{reason}");
}

#[test]
fn an_empty_value_file_and_a_missing_one_are_refused_with_the_path() {
    let f = file("  \n");
    let (field, value, reason) = refusal(KEY.value(f.path()).expect_err("refused"));
    assert_eq!(
        (field.as_str(), value),
        ("api-key-file", f.path().display().to_string())
    );
    assert_eq!(reason, "the file is empty");

    let missing = f.path().with_extension("missing");
    let (_, value, reason) = refusal(KEY.value(&missing).expect_err("refused"));
    assert_eq!(value, missing.display().to_string());
    assert!(reason.starts_with("the file cannot be read"), "{reason}");
}

#[test]
fn an_entries_file_skips_blank_and_comment_lines_and_trims() {
    let f = file("# tenants\n\n  acme:k1  \nbeta:k2\n   \n");
    assert_eq!(
        KEYS.entries(f.path()).expect("entries"),
        ["acme:k1", "beta:k2"]
    );
}

#[test]
fn an_entries_file_without_an_entry_is_refused_naming_the_shape() {
    let f = file("# nothing here\n\n");
    let (field, _, reason) = refusal(KEYS.entries(f.path()).expect_err("refused"));
    assert_eq!(field, "tenant-api-keys-file");
    assert!(reason.contains("one tenant_id:key per line"), "{reason}");
}

#[test]
fn entries_of_splits_at_every_separator_given() {
    assert_eq!(
        entries_of("a:1, b:2\n#c\n\nd:4", &[',', '\n']),
        ["a:1", "b:2", "d:4"]
    );
    assert_eq!(entries_of("a:1, b:2", &['\n']), ["a:1, b:2"]);
}

#[test]
fn a_control_plane_entry_parses_and_a_bad_one_is_refused_unechoed() {
    let entry = parse_control_plane_api_key("ops:Operators:Admin:cp-s3cret").expect("parsed");
    assert_eq!(
        (entry.id.as_str(), entry.name.as_str()),
        ("ops", "Operators")
    );
    assert_eq!(entry.role, Role::Admin);
    assert!(entry.verify("cp-s3cret"));
    for bad in ["ops:Operators:cp-s3cret", "ops:Operators:owner:cp-s3cret"] {
        let (field, value, reason) =
            refusal(parse_control_plane_api_key(bad).expect_err("refused"));
        assert_eq!(
            (field.as_str(), value.as_str()),
            ("control-plane-api-keys", "<redacted>")
        );
        assert!(!reason.contains("s3cret"), "{reason}");
    }
}

/// The parts' debug form names the entry's id, name and role and writes
/// the key redacted, so a logged or expected value never shows it.
#[test]
fn the_control_plane_parts_debug_form_redacts_the_key() {
    let parts = control_plane_api_key_parts("ops:Operators:Admin:cp-s3cret").expect("parsed");
    let shown = format!("{parts:?}");
    assert!(shown.contains("key: \"<redacted>\""), "{shown}");
    assert!(!shown.contains("s3cret"), "{shown}");
    assert!(
        shown.contains("id: \"ops\"")
            && shown.contains("name: \"Operators\"")
            && shown.contains("Admin"),
        "{shown}"
    );
}

#[test]
fn a_tenant_entry_parses_trimmed_and_a_missing_separator_is_refused_unechoed() {
    let entry = parse_tenant_api_key(" acme : t-s3cret ").expect("parsed");
    assert_eq!(
        (entry.tenant_id.as_str(), entry.key.as_str()),
        ("acme", "t-s3cret")
    );
    let (field, value, reason) =
        refusal(parse_tenant_api_key("acme-t-s3cret").expect_err("refused"));
    assert_eq!(
        (field.as_str(), value.as_str()),
        ("tenant-api-key", "<redacted>")
    );
    assert!(!reason.contains("s3cret"), "{reason}");
}
