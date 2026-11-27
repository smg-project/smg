//! Router-wide multimodal settings that used to be env-only knobs.
//!
//! Each resolves once at startup as flag (`RouterConfig`) > env > built-in
//! default, remembering which of the three supplied the value. An env-sourced
//! value logs one deprecation line: env support ends after this release.

use std::{str::FromStr, sync::OnceLock};

use anyhow::{Context, Result};
use llm_multimodal::jpeg_turbo::TurboJpegPolicy;
use openai_protocol::worker::MmProcessingMode;
use tracing::warn;

use crate::config::{parse_listen_ip, RouterConfig};

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
/// The opt-in for media origins on private, loopback or link-local addresses;
/// not a legacy name: it is also how a worker's own media pipeline, which has
/// no flags, is told (the worker reads it through [`mm_settings`]).
const ENV_ALLOW_PRIVATE_MEDIA_HOSTS: &str = "SMG_MM_ALLOW_PRIVATE_MEDIA_HOSTS";
/// The switch of the one-pass wire write, on by default; not a legacy name
/// either: it is the way back to the two-pass write without a rebuild.
const ENV_ONE_PASS_WIRE: &str = "SMG_MM_ONE_PASS_WIRE";
/// The TurboJPEG policy for a process without the flag (a worker's own media
/// pipeline on a launcher that does not carry the setting); not a legacy
/// name either.
const ENV_TURBOJPEG: &str = "SMG_MM_TURBOJPEG";
const ENV_FETCH_TIMEOUT_MS: &str = "SMG_MM_FETCH_TIMEOUT_MS";
const ENV_VIDEO_FETCH_TIMEOUT_MS: &str = "SMG_MM_VIDEO_FETCH_TIMEOUT_MS";
/// The env settings an unreadable value of which stops start-up ([`strict`]).
const STRICT_ENV: [&str; 4] = [
    ENV_PROCESSING,
    ENV_TURBOJPEG,
    ENV_FETCH_TIMEOUT_MS,
    ENV_VIDEO_FETCH_TIMEOUT_MS,
];

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
    /// Hosts the router's own media pipeline may fetch client-supplied media
    /// URLs from, by name, every redirect hop included; `None` is any host
    /// the address policy passes.
    pub allowed_media_domains: Setting<Option<Vec<String>>>,
    /// Whether a client-supplied media URL may name or resolve to a loopback,
    /// link-local, private or otherwise non-public address; off, such URLs
    /// are refused before anything is connected.
    pub allow_private_media_hosts: Setting<bool>,
    /// Whether the images of a request whose workers all read one wire dtype
    /// are preprocessed straight into it, in one pass; off, every request
    /// takes the two passes (the patch layout, then the write in the
    /// worker's dtype at assembly). The bytes are the same either way. With
    /// a pixel cache, the one pass caches the wire bytes, twice the neutral
    /// patches at the half widths and four times at float32, against the
    /// same budget: the same `--mm-pixel-cache-mb` holds fewer images.
    pub one_pass_wire: Setting<bool>,
    /// What the router's own media pipeline does when libjpeg-turbo cannot
    /// be loaded: `Prefer` serves with the pure-Rust decoder behind one
    /// warning, `Require` stops start-up.
    pub turbojpeg: Setting<TurboJpegPolicy>,
    /// Video URL fetch budget in milliseconds.
    pub video_fetch_timeout_ms: Setting<u64>,
    /// Image/audio URL fetch budget in milliseconds.
    pub fetch_timeout_ms: Setting<u64>,
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
            allowed_media_domains: Setting::from_default(None),
            allow_private_media_hosts: Setting::from_default(false),
            one_pass_wire: Setting::from_default(true),
            turbojpeg: Setting::from_default(TurboJpegPolicy::Prefer),
            video_fetch_timeout_ms: Setting::from_default(
                llm_multimodal::media::DEFAULT_VIDEO_FETCH_TIMEOUT.as_millis() as u64,
            ),
            fetch_timeout_ms: Setting::from_default(
                llm_multimodal::media::DEFAULT_FETCH_TIMEOUT.as_millis() as u64,
            ),
        }
    }
}

impl MultimodalSettings {
    /// Resolve from the router config and the process environment, read by
    /// [`process_env`]: unset is unset, and a strict name set to bytes that
    /// are not text is refused rather than read as unset.
    pub(crate) fn resolve(config: &RouterConfig) -> Result<Self> {
        Self::resolve_with(config, process_env)
    }

    /// Resolve with an explicit env reader, so tests never touch the process.
    ///
    /// Every setting is flag > env > default through [`resolve`]; the env
    /// switches share one spelling ([`switch`]; the one that is on by
    /// default reads the off spellings too, [`two_way_switch`]); an
    /// unreadable processing mode, TurboJPEG policy or download budget stops
    /// startup ([`strict`]).
    /// The reader answers `None` for a variable that is unset and an error
    /// for one it refuses to read, which stops startup the same way.
    pub(crate) fn resolve_with(
        config: &RouterConfig,
        env: impl Fn(&str) -> Result<Option<String>>,
    ) -> Result<Self> {
        let defaults = Self::default();
        let env_nonempty = |name: &str| -> Result<Option<String>> {
            Ok(env(name)?
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()))
        };

        let switch_setting =
            |flag_value: bool, env_name: &str, flag_name: Option<&str>| -> Result<Setting<bool>> {
                Ok(resolve(
                    flag_value.then_some(true),
                    env_nonempty(env_name)?,
                    env_name,
                    flag_name,
                    Setting::from_default(false),
                    switch,
                ))
            };

        Ok(Self {
            processing: strict(
                config.mm_processing,
                env_nonempty(ENV_PROCESSING)?,
                ENV_PROCESSING,
                Some("--mm-processing"),
                defaults.processing,
            )?,
            pixel_cache_mb: resolve(
                config.mm_pixel_cache_mb,
                env_nonempty(ENV_PIXEL_CACHE_MB)?,
                ENV_PIXEL_CACHE_MB,
                Some("--mm-pixel-cache-mb"),
                defaults.pixel_cache_mb,
                |raw| number(raw, ENV_PIXEL_CACHE_MB, "pixel cache stays off"),
            ),
            pixel_rdma: switch_setting(
                config.mm_pixel_rdma,
                ENV_PIXEL_RDMA,
                Some("--mm-pixel-rdma"),
            )?,
            rdma_listen_ip: resolve(
                config
                    .rdma_listen_ip
                    .as_deref()
                    .map(str::trim)
                    .filter(|ip| !ip.is_empty())
                    .map(|ip| Some(ip.to_string())),
                env_nonempty(ENV_RDMA_LISTEN_IP)?,
                ENV_RDMA_LISTEN_IP,
                Some("--rdma-listen-ip"),
                defaults.rdma_listen_ip,
                |ip| listen_ip(ip, ENV_RDMA_LISTEN_IP, "the lane stays on the inline path"),
            ),
            rdma_slot_ttl_s: resolve(
                config.rdma_slot_ttl_s.map(Some),
                env_nonempty(ENV_RDMA_SLOT_TTL_S)?,
                ENV_RDMA_SLOT_TTL_S,
                Some("--rdma-slot-ttl-s"),
                defaults.rdma_slot_ttl_s,
                |raw| number(raw, ENV_RDMA_SLOT_TTL_S, "using the derived TTL").map(Some),
            ),
            log_mm_timing: switch_setting(
                config.log_mm_timing,
                ENV_LOG_MM_TIMING,
                Some("--log-mm-timing"),
            )?,
            // A flag only: nothing in the environment lists hosts.
            allowed_media_domains: config
                .mm_allowed_media_domains
                .clone()
                .filter(|domains| !domains.is_empty())
                .map_or(defaults.allowed_media_domains, |domains| {
                    flag(Some(domains))
                }),
            // Not a legacy name (the worker's own pipeline reads it too), so
            // no deprecation line.
            allow_private_media_hosts: switch_setting(
                config.mm_allow_private_media_hosts,
                ENV_ALLOW_PRIVATE_MEDIA_HOSTS,
                None,
            )?,
            // No flag and no deprecation line: a switch that is on by
            // default, read both ways, and a spelling it does not know is
            // said (here a typo would be an ignored rollback).
            one_pass_wire: resolve(
                None,
                env_nonempty(ENV_ONE_PASS_WIRE)?,
                ENV_ONE_PASS_WIRE,
                None,
                defaults.one_pass_wire,
                |raw| {
                    two_way_switch_or_warn(
                        raw,
                        ENV_ONE_PASS_WIRE,
                        "the one-pass wire write stays on",
                    )
                },
            ),
            fetch_timeout_ms: timeout_setting(
                config.mm_fetch_timeout_ms,
                "mm_fetch_timeout_ms",
                env_nonempty(ENV_FETCH_TIMEOUT_MS)?,
                ENV_FETCH_TIMEOUT_MS,
                defaults.fetch_timeout_ms,
            )?,
            video_fetch_timeout_ms: timeout_setting(
                config.mm_video_fetch_timeout_ms,
                "mm_video_fetch_timeout_ms",
                env_nonempty(ENV_VIDEO_FETCH_TIMEOUT_MS)?,
                ENV_VIDEO_FETCH_TIMEOUT_MS,
                defaults.video_fetch_timeout_ms,
            )?,
            // Not a legacy name either (a worker's pipeline without the flag
            // reads it), so no deprecation line; unreadable, it stops
            // start-up: `require` misread as the default would serve the
            // pixels the operator asked to refuse.
            turbojpeg: strict(
                config.mm_turbojpeg,
                env_nonempty(ENV_TURBOJPEG)?,
                ENV_TURBOJPEG,
                None,
                defaults.turbojpeg,
            )?,
        })
    }
}

/// Resolve a positive millisecond budget without deprecating its env spelling.
fn timeout_setting(
    flag: Option<u64>,
    config_name: &str,
    env: Option<String>,
    name: &str,
    default: Setting<u64>,
) -> Result<Setting<u64>> {
    let setting = strict(flag, env, name, None, default)?;
    let name = if setting.source == SettingSource::Flag {
        config_name
    } else {
        name
    };
    anyhow::ensure!(setting.value > 0, "{name} must be positive");
    Ok(setting)
}

/// The TurboJPEG policy for a process that was given no setting of its own
/// (a worker pipeline whose launcher sent none): `SMG_MM_TURBOJPEG` read as
/// the router reads it, strictly, so a misspelt `require` refuses the
/// process instead of serving the fallback decoder's pixels under the
/// default, and so does a value that is not text ([`process_env`]); unset
/// or blank is the default, `prefer`.
pub(crate) fn turbojpeg_policy_from_env() -> Result<TurboJpegPolicy> {
    turbojpeg_policy(process_env(ENV_TURBOJPEG)?)
}

/// [`turbojpeg_policy_from_env`] over an explicit value of the variable.
fn turbojpeg_policy(env_value: Option<String>) -> Result<TurboJpegPolicy> {
    let env_value = env_value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let defaults = MultimodalSettings::default();
    Ok(strict(None, env_value, ENV_TURBOJPEG, None, defaults.turbojpeg)?.value)
}

/// A variable of the process environment as the settings read it:
/// [`env_text`] over what `std::env::var` answers for it.
fn process_env(name: &str) -> Result<Option<String>> {
    env_text(name, std::env::var(name))
}

/// What `std::env::var` answered for `name`, as a setting reads it: unset is
/// `None` and text is itself. A value set to bytes that are not text is not
/// "unset" (`require` read as unset would serve the fallback decoder's
/// pixels under the default): a strict name is refused with the variable
/// named, and any other leaves its default with a line, as a value it
/// cannot parse does.
fn env_text(name: &str, read: Result<String, std::env::VarError>) -> Result<Option<String>> {
    match read {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(raw)) if STRICT_ENV.contains(&name) => Err(
            anyhow::anyhow!("{name} is set to a value that is not valid UTF-8 ({raw:?})"),
        ),
        Err(std::env::VarError::NotUnicode(raw)) => {
            warn!(value = ?raw, "{name} is not valid UTF-8; leaving its default");
            Ok(None)
        }
    }
}

/// A setting flag > env > default like the rest, except that an env value
/// that cannot be read stops startup: carrying on with the default would
/// leave the process doing the opposite of what the operator asked for
/// (preprocessing where it should forward, serving the fallback decoder
/// where it should refuse), and a warning in the startup log is easy to
/// miss. An env-sourced value logs the deprecation line when `flag_name`
/// names the flag that replaces `env_name`.
fn strict<T>(
    flag_value: Option<T>,
    env_value: Option<String>,
    env_name: &str,
    flag_name: Option<&str>,
    default: Setting<T>,
) -> Result<Setting<T>>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    Ok(match (flag_value, env_value) {
        (Some(value), _) => flag(value),
        (None, Some(raw)) => {
            let value = raw
                .parse::<T>()
                .map_err(|message| anyhow::anyhow!("{message}"))
                .context(env_name.to_string())?;
            if let Some(flag_name) = flag_name {
                deprecated(env_name, flag_name);
            }
            from_env(value)
        }
        (None, None) => default,
    })
}

/// One setting, flag > env > default: `flag_value` when the flag is set;
/// else the env value `parse` can read (an unreadable one leaves the
/// default, and `parse` says so itself where that deserves a line); else
/// `default`. An env-sourced value logs the deprecation line when
/// `flag_name` names the flag that replaces `env_name`.
fn resolve<T>(
    flag_value: Option<T>,
    env_value: Option<String>,
    env_name: &str,
    flag_name: Option<&str>,
    default: Setting<T>,
    parse: impl FnOnce(&str) -> Option<T>,
) -> Setting<T> {
    if let Some(value) = flag_value {
        return flag(value);
    }
    match env_value.as_deref().and_then(parse) {
        Some(value) => {
            if let Some(flag_name) = flag_name {
                deprecated(env_name, flag_name);
            }
            from_env(value)
        }
        None => default,
    }
}

/// An env switch is on with `1`, `true`, `yes` or `on`, in any case; any
/// other spelling leaves it at its default without a word, as before (an
/// operator's typo in a legacy switch is not an outage).
fn switch(raw: &str) -> Option<bool> {
    two_way_switch(raw).filter(|&on| on)
}

/// A switch that is on by default reads the off spellings as well: `0`,
/// `false`, `no` or `off` turn it off, the on spellings of [`switch`] turn
/// it on, anything else leaves the default without a word, as [`switch`]
/// does.
fn two_way_switch(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// [`two_way_switch`] for a switch that is on by default, where a spelling
/// it does not know is not a typo to pass over in silence but a rollback
/// that did not happen: it is logged with its consequence, and the default
/// stands as it does for the other switches.
fn two_way_switch_or_warn(raw: &str, env_name: &str, consequence: &str) -> Option<bool> {
    let parsed = two_way_switch(raw);
    if parsed.is_none() {
        warn!(value = %raw, "{env_name} is not an on/off spelling; {consequence}");
    }
    parsed
}

/// A numeric env value; one that is not a number is logged with its
/// consequence and left unused.
fn number<N: FromStr>(raw: &str, env_name: &str, consequence: &str) -> Option<N> {
    let parsed = raw.parse::<N>().ok();
    if parsed.is_none() {
        warn!(value = %raw, "{env_name} is not a number; {consequence}");
    }
    parsed
}

/// An env listener address, read by the rule the validator applies to the
/// flag ([`parse_listen_ip`]); one that is not an IP address is logged with
/// its consequence and left unused, like a numeric value that is not a
/// number.
fn listen_ip(raw: &str, env_name: &str, consequence: &str) -> Option<Option<String>> {
    match parse_listen_ip(raw) {
        Ok(_) => Some(Some(raw.to_string())),
        Err(_) => {
            warn!(value = %raw, "{env_name} is not an IP address; {consequence}");
            None
        }
    }
}

fn deprecated(env_name: &str, flag_name: &str) {
    warn!(
        "{env_name} is deprecated in favour of {flag_name}; env support ends in the next minor \
         release"
    );
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
/// setting in [`STRICT_ENV`] falls back to its default while the other
/// settings keep their env values.
pub(crate) fn mm_settings() -> &'static MultimodalSettings {
    SETTINGS.get_or_init(|| resolve_unseeded_with(process_env))
}

/// Embedded use has no startup error boundary: default only the invalid strict
/// settings, retaining valid policies and budgets. The normal startup resolver
/// remains strict. An explicit env reader keeps tests away from global state.
fn resolve_unseeded_with(env: impl Fn(&str) -> Result<Option<String>>) -> MultimodalSettings {
    let config = RouterConfig::default();
    MultimodalSettings::resolve_with(&config, &env).unwrap_or_else(|error| {
        warn!(error = %error, "unreadable multimodal env setting; using its default");
        let invalid: Vec<_> = STRICT_ENV
            .into_iter()
            .filter(|key| {
                MultimodalSettings::resolve_with(&config, |name| {
                    if name == *key {
                        env(name)
                    } else {
                        Ok(None)
                    }
                })
                .is_err()
            })
            .collect();
        MultimodalSettings::resolve_with(&config, |name| {
            if invalid.contains(&name) {
                Ok(None)
            } else {
                env(name)
            }
        })
        .unwrap_or_default()
    })
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/multimodal/settings/tests.rs"]
mod tests;
