//! The secret files: how the content of a file that holds the serving key,
//! the tenant keys, the control-plane keys or a store URL becomes a value or
//! entries, and the parsers of a key entry. The command line reads a file
//! through these rules at start-up; a reload of the keys reads it again
//! through the same ones, so the two cannot drift apart.

use std::path::Path;

use smg_auth::{ApiKeyEntry, Role};

use super::{ConfigError, ConfigResult, TenantApiKeyEntry};

/// A secret a file can hold: the names its refusals use. A refusal names
/// the secret and the path; none carries a value.
#[derive(Debug, Clone, Copy)]
pub struct SecretFile {
    /// What the secret is, for the refusals.
    pub what: &'static str,
    /// The `ConfigError` field a bad file is reported under.
    pub field: &'static str,
    /// The shape of one entry of a list-valued secret, named in its
    /// refusals; a single-valued secret has none.
    pub entry: &'static str,
}

/// The shared serving key's file.
pub const API_KEY_FILE: SecretFile = SecretFile {
    what: "the shared API key",
    field: "api-key-file",
    entry: "",
};

/// The tenant keys' file.
pub const TENANT_API_KEYS_FILE: SecretFile = SecretFile {
    what: "the tenant API keys",
    field: "tenant-api-keys-file",
    entry: "tenant_id:key",
};

/// The control-plane keys' file.
pub const CONTROL_PLANE_API_KEYS_FILE: SecretFile = SecretFile {
    what: "the control-plane API keys",
    field: "control-plane-api-keys-file",
    entry: "id:name:role:key",
};

impl SecretFile {
    /// The refusal for a file that cannot be used, naming the path, never
    /// the content.
    pub fn bad_file(&self, path: &Path, reason: &str) -> ConfigError {
        ConfigError::InvalidValue {
            field: self.field.to_string(),
            value: path.display().to_string(),
            reason: reason.to_string(),
        }
    }

    fn read(&self, path: &Path) -> ConfigResult<String> {
        std::fs::read_to_string(path)
            .map_err(|e| self.bad_file(path, &format!("the file cannot be read: {e}")))
    }

    /// The value a single-valued secret's file holds, its surrounding
    /// whitespace (a trailing newline) dropped, since a Secret made from a
    /// file usually carries the editor's newline. A file that is empty or
    /// holds more than one line is refused: a tenant key file or a
    /// concatenated pair in the place of one value is a mistake, not a
    /// value.
    pub fn value(&self, path: &Path) -> ConfigResult<String> {
        let content = self.read(path)?;
        let value = content.trim();
        if value.is_empty() {
            return Err(self.bad_file(path, "the file is empty"));
        }
        if value.lines().count() > 1 {
            return Err(self.bad_file(
                path,
                "the file holds more than one line; it must hold the value alone",
            ));
        }
        Ok(value.to_string())
    }

    /// The entries a list-valued secret's file holds: its lines, blank lines
    /// and `#` comment lines skipped, the whitespace around an entry
    /// dropped. A file that holds no entry is refused: the file was meant to
    /// close the routes its keys guard, and an empty one would leave them
    /// open.
    pub fn entries(&self, path: &Path) -> ConfigResult<Vec<String>> {
        let content = self.read(path)?;
        let entries = entries_of(&content, &['\n']);
        if entries.is_empty() {
            return Err(self.bad_file(
                path,
                &format!(
                    "the file holds no entry; one {} per line, blank and # lines skipped",
                    self.entry
                ),
            ));
        }
        Ok(entries)
    }
}

/// The entries of a text: split at the separators, the whitespace around
/// each dropped, blank entries and `#` comments skipped.
pub fn entries_of(text: &str, separators: &[char]) -> Vec<String> {
    text.split(separators)
        .map(str::trim)
        .filter(|entry| !entry.is_empty() && !entry.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// Parse a control-plane API key entry, `id:name:role:key`. A malformed
/// entry is a configuration error, never a dropped key: a dropped key would
/// leave the control plane open (or lock an operator out) with nothing but
/// a stderr line to show for it. The entry is not echoed, since it carries
/// the credential.
pub fn parse_control_plane_api_key(key_str: &str) -> ConfigResult<ApiKeyEntry> {
    let parts: Vec<&str> = key_str.splitn(4, ':').collect();
    if parts.len() != 4 {
        return Err(ConfigError::InvalidValue {
            field: "control-plane-api-keys".to_string(),
            value: "<redacted>".to_string(),
            reason: "expected 'id:name:role:key'".to_string(),
        });
    }
    let id = parts[0];
    let name = parts[1];
    let role_str = parts[2];
    let key = parts[3];

    let role = match role_str.to_lowercase().as_str() {
        "admin" => Role::Admin,
        "user" => Role::User,
        // The field is not named either: with the fields out of order it is the credential.
        _ => {
            return Err(ConfigError::InvalidValue {
                field: "control-plane-api-keys".to_string(),
                value: "<redacted>".to_string(),
                reason: "the role must be 'admin' or 'user'".to_string(),
            });
        }
    };

    Ok(ApiKeyEntry::new(id, name, key, role))
}

/// Parse a tenant-scoped data-plane API key entry, `tenant_id:key`. Only
/// the ':' separator is checked here; the non-empty and duplicate checks
/// live in `ConfigValidator::tenant_credentials`, which the start and a
/// reload run over the whole set, so they also cover `TenantApiKeyEntry`
/// values from a config file or language binding. Fails
/// hard (an empty `AuthConfig` disables auth entirely, not just narrows it)
/// and never echoes `key_str`, which may be the plaintext credential.
pub fn parse_tenant_api_key(key_str: &str) -> ConfigResult<TenantApiKeyEntry> {
    let Some((tenant_id, key)) = key_str.split_once(':') else {
        return Err(ConfigError::InvalidValue {
            field: "tenant-api-key".to_string(),
            value: "<redacted>".to_string(),
            reason: "expected 'tenant_id:key' (missing ':' separator)".to_string(),
        });
    };

    Ok(TenantApiKeyEntry {
        tenant_id: tenant_id.trim().to_string(),
        key: key.trim().to_string(),
    })
}

#[cfg(test)]
mod tests {
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
}
