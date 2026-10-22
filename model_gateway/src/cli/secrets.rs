//! The secrets the command line, the environment or a file can give (the
//! serving key, the tenant keys, the store URLs): the names of their sources
//! and the one resolution rule, with the environment lookup as a parameter.

use std::path::{Path, PathBuf};

use smg::{
    config::{ConfigError, ConfigResult},
    server::posture::SecretSource,
};

/// A secret the command line, the environment or a file can give: the
/// names of its sources, for the resolution and for the refusals. A
/// refusal names sources and paths; none carries a value.
pub(crate) struct SecretSetting {
    /// What the secret is, for the refusals.
    what: &'static str,
    /// The `ConfigError` field a bad file is reported under.
    file_field: &'static str,
    /// The flag taking the value.
    flag: &'static str,
    /// The environment variable taking the value.
    env: &'static str,
    /// The flag naming a file that holds the value.
    file_flag: &'static str,
    /// The environment variable naming that file.
    file_env: &'static str,
}

pub(crate) const API_KEY_SETTING: SecretSetting = SecretSetting {
    what: "the shared API key",
    file_field: "api-key-file",
    flag: "--api-key",
    env: "SMG_API_KEY",
    file_flag: "--api-key-file",
    file_env: "SMG_API_KEY_FILE",
};

pub(crate) const TENANT_API_KEYS_SETTING: SecretSetting = SecretSetting {
    what: "the tenant API keys",
    file_field: "tenant-api-keys-file",
    flag: "--tenant-api-key",
    env: "SMG_TENANT_API_KEYS",
    file_flag: "--tenant-api-keys-file",
    file_env: "SMG_TENANT_API_KEYS_FILE",
};

pub(crate) const POSTGRES_DB_URL_SETTING: SecretSetting = SecretSetting {
    what: "the PostgreSQL connection URL",
    file_field: "postgres-db-url-file",
    flag: "--postgres-db-url",
    env: "SMG_POSTGRES_DB_URL",
    file_flag: "--postgres-db-url-file",
    file_env: "SMG_POSTGRES_DB_URL_FILE",
};

pub(crate) const REDIS_URL_SETTING: SecretSetting = SecretSetting {
    what: "the Redis connection URL",
    file_field: "redis-url-file",
    flag: "--redis-url",
    env: "SMG_REDIS_URL",
    file_flag: "--redis-url-file",
    file_env: "SMG_REDIS_URL_FILE",
};

/// An environment lookup: the value of a variable, `None` when it is not
/// set, and a refusal when it is set to bytes that are not UTF-8 (a source
/// that is set but unusable is refused, not passed over as unset). A
/// parameter rather than `std::env::var` so a test need not touch the
/// process environment.
pub(crate) type EnvLookup<'a> = &'a dyn Fn(&str) -> ConfigResult<Option<String>>;

/// The process environment as an [`EnvLookup`].
pub(crate) fn process_env(name: &str) -> ConfigResult<Option<String>> {
    env_value(name, std::env::var_os(name))
}

/// A variable's raw value as a string: `None` when the variable is unset,
/// a refusal naming the variable when its bytes are not UTF-8.
pub(crate) fn env_value(
    name: &str,
    raw: Option<std::ffi::OsString>,
) -> ConfigResult<Option<String>> {
    raw.map(|value| {
        value.into_string().map_err(|_| ConfigError::InvalidValue {
            field: name.to_string(),
            value: "<not UTF-8>".to_string(),
            reason: "set, but its value is not UTF-8".to_string(),
        })
    })
    .transpose()
}

impl SecretSetting {
    /// The file the flag or, failing that, the environment names.
    fn file(&self, flag_value: Option<&Path>, env: EnvLookup<'_>) -> ConfigResult<Option<PathBuf>> {
        Ok(match flag_value {
            Some(path) => Some(path.to_path_buf()),
            None => env(self.file_env)?.map(PathBuf::from),
        })
    }

    /// The refusal for a value and a file given together: ranking them
    /// would honour a stale secret in silence.
    fn given_twice(&self) -> ConfigError {
        ConfigError::IncompatibleConfig {
            reason: format!(
                "{} is given twice, by {} (or {}) and by {} (or {}); give one source",
                self.what, self.flag, self.env, self.file_flag, self.file_env
            ),
        }
    }

    /// The refusal for a file that cannot be used, naming the path, never
    /// the content.
    fn bad_file(&self, path: &Path, reason: &str) -> ConfigError {
        ConfigError::InvalidValue {
            field: self.file_field.to_string(),
            value: path.display().to_string(),
            reason: reason.to_string(),
        }
    }

    fn read_file(&self, path: &Path) -> ConfigResult<String> {
        std::fs::read_to_string(path)
            .map_err(|e| self.bad_file(path, &format!("the file cannot be read: {e}")))
    }

    /// The value of a single-valued secret and its source: the flag's,
    /// else the environment's, else the content of the file the flag or the
    /// environment names; the environment's value and the file's content
    /// have their surrounding whitespace (a trailing newline) dropped, since
    /// a Secret made from a file usually carries the editor's newline into
    /// both. A value and a file together are refused, as is a file that is
    /// empty or holds more than one line (a tenant key file or a
    /// concatenated pair in the place of one value is a mistake, not a
    /// value). An empty value from the flag or the environment passes here
    /// and is refused where the value is validated.
    pub(crate) fn resolve_value(
        &self,
        flag_value: Option<&str>,
        file_flag_value: Option<&Path>,
        env: EnvLookup<'_>,
    ) -> ConfigResult<Option<(String, SecretSource)>> {
        let value = match flag_value {
            Some(value) => Some((value.to_string(), SecretSource::Flag)),
            None => {
                env(self.env)?.map(|value| (value.trim().to_string(), SecretSource::Environment))
            }
        };
        match (value, self.file(file_flag_value, env)?) {
            (Some(_), Some(_)) => Err(self.given_twice()),
            (Some(value), None) => Ok(Some(value)),
            (None, None) => Ok(None),
            (None, Some(path)) => {
                let content = self.read_file(&path)?;
                let value = content.trim();
                if value.is_empty() {
                    return Err(self.bad_file(&path, "the file is empty"));
                }
                if value.lines().count() > 1 {
                    return Err(self.bad_file(
                        &path,
                        "the file holds more than one line; it must hold the value alone",
                    ));
                }
                Ok(Some((value.to_string(), SecretSource::File)))
            }
        }
    }

    /// The entries of a list-valued secret and their source: the flag's
    /// occurrences, else the environment's value split at ',' and at line
    /// ends, else the lines of the file the flag or the environment names,
    /// blank lines and `#` comment lines skipped; the whitespace around an
    /// entry is dropped. Entries and a file together are refused, and so is
    /// a source that is set but holds no entry: the source was meant to
    /// close the serving routes, and an empty one would leave them open.
    pub(crate) fn resolve_entries(
        &self,
        flag_values: &[String],
        file_flag_value: Option<&Path>,
        env: EnvLookup<'_>,
    ) -> ConfigResult<Option<(Vec<String>, SecretSource)>> {
        let entries_of = |text: &str, separators: &[char]| -> Vec<String> {
            text.split(separators)
                .map(str::trim)
                .filter(|entry| !entry.is_empty() && !entry.starts_with('#'))
                .map(str::to_string)
                .collect()
        };
        let value = if flag_values.is_empty() {
            match env(self.env)? {
                Some(text) => {
                    let entries = entries_of(&text, &[',', '\n']);
                    if entries.is_empty() {
                        return Err(ConfigError::InvalidValue {
                            field: self.env.to_string(),
                            value: "<empty>".to_string(),
                            reason: format!(
                                "set but holds no entry; {} are tenant_id:key entries separated by ',' or a newline",
                                self.what
                            ),
                        });
                    }
                    Some((entries, SecretSource::Environment))
                }
                None => None,
            }
        } else {
            Some((flag_values.to_vec(), SecretSource::Flag))
        };
        match (value, self.file(file_flag_value, env)?) {
            (Some(_), Some(_)) => Err(self.given_twice()),
            (Some(value), None) => Ok(Some(value)),
            (None, None) => Ok(None),
            (None, Some(path)) => {
                let content = self.read_file(&path)?;
                let entries = entries_of(&content, &['\n']);
                if entries.is_empty() {
                    return Err(self.bad_file(
                        &path,
                        "the file holds no entry; one tenant_id:key per line, blank and # lines skipped",
                    ));
                }
                Ok(Some((entries, SecretSource::File)))
            }
        }
    }
}
