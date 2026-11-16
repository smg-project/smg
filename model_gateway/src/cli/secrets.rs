//! The secrets the command line, the environment or a file can give (the
//! serving key, the tenant keys, the control-plane keys, the store URLs):
//! the names of their sources and the one resolution rule, with the
//! environment lookup as a parameter. A file is read by the library's rules
//! (`smg::config::secrets`), the ones a reload of the keys reads it by.

use std::path::{Path, PathBuf};

use smg::{
    config::{
        secrets::{
            entries_of, SecretFile, API_KEY_FILE, CONTROL_PLANE_API_KEYS_FILE, TENANT_API_KEYS_FILE,
        },
        ConfigError, ConfigResult,
    },
    server::posture::SecretSource,
};

/// A secret the command line, the environment or a file can give: the
/// names of its sources, for the resolution and for the refusals. A
/// refusal names sources and paths; none carries a value.
pub(crate) struct SecretSetting {
    /// The file's names: what the secret is, the field a bad file is
    /// reported under, the shape of an entry.
    rules: SecretFile,
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
    rules: API_KEY_FILE,
    flag: "--api-key",
    env: "SMG_API_KEY",
    file_flag: "--api-key-file",
    file_env: "SMG_API_KEY_FILE",
};

pub(crate) const TENANT_API_KEYS_SETTING: SecretSetting = SecretSetting {
    rules: TENANT_API_KEYS_FILE,
    flag: "--tenant-api-key",
    env: "SMG_TENANT_API_KEYS",
    file_flag: "--tenant-api-keys-file",
    file_env: "SMG_TENANT_API_KEYS_FILE",
};

pub(crate) const CONTROL_PLANE_API_KEYS_SETTING: SecretSetting = SecretSetting {
    rules: CONTROL_PLANE_API_KEYS_FILE,
    flag: "--control-plane-api-keys",
    env: "CONTROL_PLANE_API_KEYS",
    file_flag: "--control-plane-api-keys-file",
    file_env: "CONTROL_PLANE_API_KEYS_FILE",
};

pub(crate) const POSTGRES_DB_URL_SETTING: SecretSetting = SecretSetting {
    rules: SecretFile {
        what: "the PostgreSQL connection URL",
        field: "postgres-db-url-file",
        entry: "",
    },
    flag: "--postgres-db-url",
    env: "SMG_POSTGRES_DB_URL",
    file_flag: "--postgres-db-url-file",
    file_env: "SMG_POSTGRES_DB_URL_FILE",
};

pub(crate) const REDIS_URL_SETTING: SecretSetting = SecretSetting {
    rules: SecretFile {
        what: "the Redis connection URL",
        field: "redis-url-file",
        entry: "",
    },
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
    pub(crate) fn file(
        &self,
        flag_value: Option<&Path>,
        env: EnvLookup<'_>,
    ) -> ConfigResult<Option<PathBuf>> {
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
                self.rules.what, self.flag, self.env, self.file_flag, self.file_env
            ),
        }
    }

    /// The value of a single-valued secret and its source: the flag's,
    /// else the environment's, else the content of the file the flag or the
    /// environment names ([`SecretFile::value`]: one line, trimmed, an
    /// empty or longer file refused); the environment's value has its
    /// surrounding whitespace dropped like the file's. A value and a file
    /// together are refused. An empty value from the flag or the
    /// environment passes here and is refused where the value is validated.
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
            (None, Some(path)) => Ok(Some((self.rules.value(&path)?, SecretSource::File))),
        }
    }

    /// The entries of a list-valued secret and their source: the flag's
    /// occurrences, else the environment's value split at ',' and at line
    /// ends, else the lines of the file the flag or the environment names
    /// ([`SecretFile::entries`]); blank entries and `#` comment lines are
    /// skipped and the whitespace around an entry dropped. Entries and a
    /// file together are refused, and so is a source that is set but holds
    /// no entry: the source was meant to close the routes its keys guard,
    /// and an empty one would leave them open.
    pub(crate) fn resolve_entries(
        &self,
        flag_values: &[String],
        file_flag_value: Option<&Path>,
        env: EnvLookup<'_>,
    ) -> ConfigResult<Option<(Vec<String>, SecretSource)>> {
        let value = if flag_values.is_empty() {
            match env(self.env)? {
                Some(text) => {
                    let entries = entries_of(&text, &[',', '\n']);
                    if entries.is_empty() {
                        return Err(ConfigError::InvalidValue {
                            field: self.env.to_string(),
                            value: "<empty>".to_string(),
                            reason: format!(
                                "set but holds no entry; {} are {} entries separated by ',' or a newline",
                                self.rules.what, self.rules.entry
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
            (None, Some(path)) => Ok(Some((self.rules.entries(&path)?, SecretSource::File))),
        }
    }
}
