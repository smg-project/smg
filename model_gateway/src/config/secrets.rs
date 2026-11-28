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

/// The parts of a control-plane API key entry, `id:name:role:key`, as
/// [`control_plane_api_key_parts`] reads them: the gateway hashes the key
/// into an [`ApiKeyEntry`], a language binding builds its own entry with
/// the key as given.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ControlPlaneApiKeyParts<'a> {
    /// The key's identifier.
    pub id: &'a str,
    /// The key's name.
    pub name: &'a str,
    /// The key itself, as the entry gives it.
    pub key: &'a str,
    /// The role the entry names.
    pub role: Role,
}

/// The key is written `<redacted>`, as the other key-carrying types write
/// it: a logged or expected value never shows the credential.
impl std::fmt::Debug for ControlPlaneApiKeyParts<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlPlaneApiKeyParts")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("key", &"<redacted>")
            .field("role", &self.role)
            .finish()
    }
}

/// Read a control-plane API key entry, `id:name:role:key`. A malformed
/// entry is a configuration error, never a dropped key: a dropped key would
/// leave the control plane open (or lock an operator out) with nothing but
/// a stderr line to show for it. The entry is not echoed, since it carries
/// the credential.
pub fn control_plane_api_key_parts(key_str: &str) -> ConfigResult<ControlPlaneApiKeyParts<'_>> {
    let parts: Vec<&str> = key_str.splitn(4, ':').collect();
    if parts.len() != 4 {
        return Err(ConfigError::InvalidValue {
            field: "control-plane-api-keys".to_string(),
            value: "<redacted>".to_string(),
            reason: "expected 'id:name:role:key'".to_string(),
        });
    }
    let role = match parts[2].to_lowercase().as_str() {
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
    Ok(ControlPlaneApiKeyParts {
        id: parts[0],
        name: parts[1],
        key: parts[3],
        role,
    })
}

/// Parse a control-plane API key entry into the gateway's entry, which
/// keeps the key's hash alone ([`control_plane_api_key_parts`] reads and
/// checks the entry).
pub fn parse_control_plane_api_key(key_str: &str) -> ConfigResult<ApiKeyEntry> {
    let parts = control_plane_api_key_parts(key_str)?;
    Ok(ApiKeyEntry::new(
        parts.id, parts.name, parts.key, parts.role,
    ))
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
#[path = "../../tests/unit/config/secrets/tests.rs"]
mod tests;
