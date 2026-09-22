//! Router-wide multimodal settings that used to be env-only knobs.
//!
//! Each resolves once at startup as flag (`RouterConfig`) > env > built-in
//! default, remembering which of the three supplied the value. An env-sourced
//! value logs one deprecation line: env support ends after this release.

use std::sync::OnceLock;

use anyhow::{Context, Result};
use openai_protocol::worker::MmProcessingMode;
use tracing::warn;

use crate::config::RouterConfig;

/// Which of the three layers supplied a setting's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SettingSource {
    Flag,
    Env,
    Default,
}

impl SettingSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Flag => "flag",
            Self::Env => "env",
            Self::Default => "default",
        }
    }
}

/// A resolved value and where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Setting<T> {
    pub value: T,
    pub source: SettingSource,
}

impl<T> Setting<T> {
    fn from_default(value: T) -> Self {
        Self {
            value,
            source: SettingSource::Default,
        }
    }
}

/// The env-only names each flag replaces.
const ENV_PROCESSING: &str = "SMG_MM_PROCESSING";
const ENV_PIXEL_CACHE_MB: &str = "SMG_MM_PIXEL_CACHE_MB";
const ENV_PIXEL_RDMA: &str = "SMG_MM_PIXEL_RDMA";
const ENV_RDMA_LISTEN_IP: &str = "SMG_RDMA_LISTEN_IP";
const ENV_RDMA_SLOT_TTL_S: &str = "SMG_RDMA_SLOT_TTL_S";
const ENV_LOG_MM_TIMING: &str = "SMG_LOG_MM_TIMING";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MultimodalSettings {
    /// Where media for vLLM gRPC workers is fetched and preprocessed.
    pub processing: Setting<MmProcessingMode>,
    /// Host-DRAM budget for router-side preprocessed media, in MiB; 0 is off.
    pub pixel_cache_mb: Setting<usize>,
    /// Legacy switch for the RDMA pixel lane; `--multimodal-tensor-transport
    /// rdma` is the first-class one.
    pub pixel_rdma: Setting<bool>,
    /// Listener IP for the RDMA metadata exchange; unset keeps the inline path.
    pub rdma_listen_ip: Setting<Option<String>>,
    /// Full-TTL override for leased RDMA slots, in seconds.
    pub rdma_slot_ttl_s: Setting<Option<u64>>,
    /// Per-request multimodal timing at INFO.
    pub log_mm_timing: Setting<bool>,
}

impl Default for MultimodalSettings {
    fn default() -> Self {
        Self {
            processing: Setting::from_default(MmProcessingMode::Auto),
            pixel_cache_mb: Setting::from_default(0),
            pixel_rdma: Setting::from_default(false),
            rdma_listen_ip: Setting::from_default(None),
            rdma_slot_ttl_s: Setting::from_default(None),
            log_mm_timing: Setting::from_default(false),
        }
    }
}

impl MultimodalSettings {
    /// Resolve from the router config and the process environment.
    pub(crate) fn resolve(config: &RouterConfig) -> Result<Self> {
        Self::resolve_with(config, |name| std::env::var(name).ok())
    }

    /// Resolve with an explicit env reader, so tests never touch the process.
    ///
    /// A `SMG_MM_PROCESSING` that cannot be read stops startup: carrying on
    /// with the default would leave the router doing the opposite of what the
    /// operator asked for, and a warning in the startup log is easy to miss.
    pub(crate) fn resolve_with(
        config: &RouterConfig,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let defaults = Self::default();
        let env_nonempty = |name: &str| {
            env(name)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        };
        let deprecated = |env_name: &str, flag: &str| {
            warn!(
                "{env_name} is deprecated in favour of {flag}; env support ends in the next \
                 minor release"
            );
        };

        let processing = match (config.mm_processing, env_nonempty(ENV_PROCESSING)) {
            (Some(mode), _) => flag(mode),
            (None, Some(raw)) => {
                let mode = raw
                    .parse::<MmProcessingMode>()
                    .map_err(|message| anyhow::anyhow!("{message}"))
                    .context(ENV_PROCESSING)?;
                deprecated(ENV_PROCESSING, "--mm-processing");
                from_env(mode)
            }
            (None, None) => defaults.processing,
        };

        let pixel_cache_mb = match (config.mm_pixel_cache_mb, env_nonempty(ENV_PIXEL_CACHE_MB)) {
            (Some(mb), _) => flag(mb),
            (None, Some(raw)) => match raw.parse::<usize>() {
                Ok(mb) => {
                    deprecated(ENV_PIXEL_CACHE_MB, "--mm-pixel-cache-mb");
                    from_env(mb)
                }
                Err(_) => {
                    warn!(value = %raw, "{ENV_PIXEL_CACHE_MB} is not a number; pixel cache stays off");
                    defaults.pixel_cache_mb
                }
            },
            (None, None) => defaults.pixel_cache_mb,
        };

        let pixel_rdma = if config.mm_pixel_rdma {
            flag(true)
        } else {
            match env_nonempty(ENV_PIXEL_RDMA).as_deref() {
                Some("1" | "true") => {
                    deprecated(ENV_PIXEL_RDMA, "--mm-pixel-rdma");
                    from_env(true)
                }
                _ => defaults.pixel_rdma,
            }
        };

        let rdma_listen_ip = match (
            config
                .rdma_listen_ip
                .clone()
                .filter(|ip| !ip.trim().is_empty()),
            env_nonempty(ENV_RDMA_LISTEN_IP),
        ) {
            (Some(ip), _) => flag(Some(ip)),
            (None, Some(ip)) => {
                deprecated(ENV_RDMA_LISTEN_IP, "--rdma-listen-ip");
                from_env(Some(ip))
            }
            (None, None) => defaults.rdma_listen_ip,
        };

        let rdma_slot_ttl_s = match (config.rdma_slot_ttl_s, env_nonempty(ENV_RDMA_SLOT_TTL_S)) {
            (Some(secs), _) => flag(Some(secs)),
            (None, Some(raw)) => match raw.parse::<u64>() {
                Ok(secs) => {
                    deprecated(ENV_RDMA_SLOT_TTL_S, "--rdma-slot-ttl-s");
                    from_env(Some(secs))
                }
                Err(_) => {
                    warn!(value = %raw, "{ENV_RDMA_SLOT_TTL_S} is not a number; using the derived TTL");
                    defaults.rdma_slot_ttl_s
                }
            },
            (None, None) => defaults.rdma_slot_ttl_s,
        };

        let log_mm_timing = if config.log_mm_timing {
            flag(true)
        } else {
            match env_nonempty(ENV_LOG_MM_TIMING).map(|value| value.to_ascii_lowercase()) {
                Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on") => {
                    deprecated(ENV_LOG_MM_TIMING, "--log-mm-timing");
                    from_env(true)
                }
                _ => defaults.log_mm_timing,
            }
        };

        Ok(Self {
            processing,
            pixel_cache_mb,
            pixel_rdma,
            rdma_listen_ip,
            rdma_slot_ttl_s,
            log_mm_timing,
        })
    }
}

fn flag<T>(value: T) -> Setting<T> {
    Setting {
        value,
        source: SettingSource::Flag,
    }
}

fn from_env<T>(value: T) -> Setting<T> {
    Setting {
        value,
        source: SettingSource::Env,
    }
}

static SETTINGS: OnceLock<MultimodalSettings> = OnceLock::new();

/// Seed the process-wide settings from the resolved router config. Call once
/// at startup before serving; idempotent (first call wins).
pub(crate) fn init_mm_settings(settings: MultimodalSettings) {
    let _ = SETTINGS.set(settings);
}

/// The process-wide settings. Never seeded (tests, embedded use): resolved
/// lazily from env and the built-in defaults, and an unreadable
/// `SMG_MM_PROCESSING` falls back to its default while the other settings
/// keep their env values.
pub(crate) fn mm_settings() -> &'static MultimodalSettings {
    SETTINGS.get_or_init(|| {
        let config = RouterConfig::default();
        MultimodalSettings::resolve(&config).unwrap_or_else(|error| {
            warn!(error = %error, "unreadable {ENV_PROCESSING}; using the default placement");
            MultimodalSettings::resolve_with(&config, |name| {
                (name != ENV_PROCESSING)
                    .then(|| std::env::var(name).ok())
                    .flatten()
            })
            .unwrap_or_default()
        })
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name| map.get(name).cloned()
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn defaults_when_neither_flag_nor_env_is_set() {
        let settings = MultimodalSettings::resolve_with(&RouterConfig::default(), no_env).unwrap();
        assert_eq!(settings, MultimodalSettings::default());
        assert_eq!(settings.processing.value, MmProcessingMode::Auto);
        assert_eq!(settings.processing.source, SettingSource::Default);
    }

    #[test]
    fn env_supplies_every_setting_when_the_flags_are_unset() {
        let settings = MultimodalSettings::resolve_with(
            &RouterConfig::default(),
            env(&[
                ("SMG_MM_PROCESSING", " Worker "),
                ("SMG_MM_PIXEL_CACHE_MB", "256"),
                ("SMG_MM_PIXEL_RDMA", "true"),
                ("SMG_RDMA_LISTEN_IP", "10.0.0.7"),
                ("SMG_RDMA_SLOT_TTL_S", "600"),
                ("SMG_LOG_MM_TIMING", "YES"),
            ]),
        )
        .unwrap();
        assert_eq!(settings.processing, from_env(MmProcessingMode::Worker));
        assert_eq!(settings.pixel_cache_mb, from_env(256));
        assert_eq!(settings.pixel_rdma, from_env(true));
        assert_eq!(
            settings.rdma_listen_ip,
            from_env(Some("10.0.0.7".to_string()))
        );
        assert_eq!(settings.rdma_slot_ttl_s, from_env(Some(600)));
        assert_eq!(settings.log_mm_timing, from_env(true));
    }

    #[test]
    fn flags_win_over_env() {
        let config = RouterConfig {
            mm_processing: Some(MmProcessingMode::Router),
            mm_pixel_cache_mb: Some(64),
            mm_pixel_rdma: true,
            rdma_listen_ip: Some("192.168.1.2".to_string()),
            rdma_slot_ttl_s: Some(900),
            log_mm_timing: true,
            ..RouterConfig::default()
        };
        let settings = MultimodalSettings::resolve_with(
            &config,
            env(&[
                ("SMG_MM_PROCESSING", "worker"),
                ("SMG_MM_PIXEL_CACHE_MB", "256"),
                ("SMG_MM_PIXEL_RDMA", "0"),
                ("SMG_RDMA_LISTEN_IP", "10.0.0.7"),
                ("SMG_RDMA_SLOT_TTL_S", "600"),
                ("SMG_LOG_MM_TIMING", "false"),
            ]),
        )
        .unwrap();
        assert_eq!(settings.processing, flag(MmProcessingMode::Router));
        assert_eq!(settings.pixel_cache_mb, flag(64));
        assert_eq!(settings.pixel_rdma, flag(true));
        assert_eq!(
            settings.rdma_listen_ip,
            flag(Some("192.168.1.2".to_string()))
        );
        assert_eq!(settings.rdma_slot_ttl_s, flag(Some(900)));
        assert_eq!(settings.log_mm_timing, flag(true));
    }

    /// A mode that is nearly right would otherwise resolve to the default and
    /// run the opposite of what the operator asked for.
    #[test]
    fn an_unreadable_processing_env_stops_startup_but_blank_is_the_default() {
        let config = RouterConfig::default();
        let error =
            MultimodalSettings::resolve_with(&config, env(&[("SMG_MM_PROCESSING", "routers")]))
                .unwrap_err();
        assert!(error.to_string().contains("SMG_MM_PROCESSING"), "{error:#}");

        let blank =
            MultimodalSettings::resolve_with(&config, env(&[("SMG_MM_PROCESSING", "  ")])).unwrap();
        assert_eq!(
            blank.processing,
            Setting::from_default(MmProcessingMode::Auto)
        );
    }

    /// Without `SMG_MM_PROCESSING` in the reader, the other settings resolve
    /// on their own: what the never-seeded path falls back to when the
    /// placement env is unreadable.
    #[test]
    fn the_other_settings_survive_an_unreadable_placement_env() {
        let full = env(&[
            ("SMG_MM_PROCESSING", "routers"),
            ("SMG_MM_PIXEL_CACHE_MB", "256"),
            ("SMG_LOG_MM_TIMING", "1"),
        ]);
        let config = RouterConfig::default();
        assert!(MultimodalSettings::resolve_with(&config, &full).is_err());
        let rest = MultimodalSettings::resolve_with(&config, |name| {
            (name != ENV_PROCESSING).then(|| full(name)).flatten()
        })
        .unwrap();
        assert_eq!(
            rest.processing,
            Setting::from_default(MmProcessingMode::Auto)
        );
        assert_eq!(rest.pixel_cache_mb, from_env(256));
        assert_eq!(rest.log_mm_timing, from_env(true));
    }

    /// The numeric and boolean env knobs keep their lenient reading: an
    /// unreadable value is ignored, and only the exact legacy spellings switch
    /// the RDMA lane on.
    #[test]
    fn unreadable_numeric_env_and_unknown_switch_values_fall_back() {
        let settings = MultimodalSettings::resolve_with(
            &RouterConfig::default(),
            env(&[
                ("SMG_MM_PIXEL_CACHE_MB", "lots"),
                ("SMG_MM_PIXEL_RDMA", "yes"),
                ("SMG_RDMA_SLOT_TTL_S", "soon"),
                ("SMG_LOG_MM_TIMING", "0"),
                ("SMG_RDMA_LISTEN_IP", "   "),
            ]),
        )
        .unwrap();
        assert_eq!(settings, MultimodalSettings::default());
    }
}
