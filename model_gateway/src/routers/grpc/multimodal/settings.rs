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
/// The env settings an unreadable value of which stops start-up ([`strict`]).
const STRICT_ENV: [&str; 2] = [ENV_PROCESSING, ENV_TURBOJPEG];

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
    /// Every setting is flag > env > default through [`resolve`]; the env
    /// switches share one spelling ([`switch`]; the one that is on by
    /// default reads the off spellings too, [`two_way_switch`]); an
    /// unreadable `SMG_MM_PROCESSING` or `SMG_MM_TURBOJPEG` stops startup
    /// ([`strict`]).
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

        let switch_setting = |flag_value: bool, env_name: &str, flag_name: Option<&str>| {
            resolve(
                flag_value.then_some(true),
                env_nonempty(env_name),
                env_name,
                flag_name,
                Setting::from_default(false),
                switch,
            )
        };

        Ok(Self {
            processing: strict(
                config.mm_processing,
                env_nonempty(ENV_PROCESSING),
                ENV_PROCESSING,
                Some("--mm-processing"),
                defaults.processing,
            )?,
            pixel_cache_mb: resolve(
                config.mm_pixel_cache_mb,
                env_nonempty(ENV_PIXEL_CACHE_MB),
                ENV_PIXEL_CACHE_MB,
                Some("--mm-pixel-cache-mb"),
                defaults.pixel_cache_mb,
                |raw| number(raw, ENV_PIXEL_CACHE_MB, "pixel cache stays off"),
            ),
            pixel_rdma: switch_setting(
                config.mm_pixel_rdma,
                ENV_PIXEL_RDMA,
                Some("--mm-pixel-rdma"),
            ),
            rdma_listen_ip: resolve(
                config
                    .rdma_listen_ip
                    .as_deref()
                    .map(str::trim)
                    .filter(|ip| !ip.is_empty())
                    .map(|ip| Some(ip.to_string())),
                env_nonempty(ENV_RDMA_LISTEN_IP),
                ENV_RDMA_LISTEN_IP,
                Some("--rdma-listen-ip"),
                defaults.rdma_listen_ip,
                |ip| listen_ip(ip, ENV_RDMA_LISTEN_IP, "the lane stays on the inline path"),
            ),
            rdma_slot_ttl_s: resolve(
                config.rdma_slot_ttl_s.map(Some),
                env_nonempty(ENV_RDMA_SLOT_TTL_S),
                ENV_RDMA_SLOT_TTL_S,
                Some("--rdma-slot-ttl-s"),
                defaults.rdma_slot_ttl_s,
                |raw| number(raw, ENV_RDMA_SLOT_TTL_S, "using the derived TTL").map(Some),
            ),
            log_mm_timing: switch_setting(
                config.log_mm_timing,
                ENV_LOG_MM_TIMING,
                Some("--log-mm-timing"),
            ),
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
            ),
            // No flag and no deprecation line: a switch that is on by
            // default, read both ways, and a spelling it does not know is
            // said (here a typo would be an ignored rollback).
            one_pass_wire: resolve(
                None,
                env_nonempty(ENV_ONE_PASS_WIRE),
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
            // Not a legacy name either (a worker's pipeline without the flag
            // reads it), so no deprecation line; unreadable, it stops
            // start-up: `require` misread as the default would serve the
            // pixels the operator asked to refuse.
            turbojpeg: strict(
                config.mm_turbojpeg,
                env_nonempty(ENV_TURBOJPEG),
                ENV_TURBOJPEG,
                None,
                defaults.turbojpeg,
            )?,
        })
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
/// `SMG_MM_PROCESSING` or `SMG_MM_TURBOJPEG` falls back to its default while
/// the other settings keep their env values.
pub(crate) fn mm_settings() -> &'static MultimodalSettings {
    SETTINGS.get_or_init(|| {
        let config = RouterConfig::default();
        MultimodalSettings::resolve(&config).unwrap_or_else(|error| {
            warn!(error = %error, "unreadable multimodal env setting; using its default");
            MultimodalSettings::resolve_with(&config, |name| {
                (!STRICT_ENV.contains(&name))
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
        assert_eq!(settings.one_pass_wire, Setting::from_default(true));
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
                ("SMG_MM_ALLOW_PRIVATE_MEDIA_HOSTS", "True"),
                ("SMG_MM_ONE_PASS_WIRE", "off"),
                ("SMG_MM_TURBOJPEG", " Require "),
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
        assert_eq!(settings.allow_private_media_hosts, from_env(true));
        assert_eq!(settings.one_pass_wire, from_env(false));
        assert_eq!(settings.turbojpeg, from_env(TurboJpegPolicy::Require));
        // The allow-list is a flag only: nothing in the environment lists hosts.
        assert_eq!(settings.allowed_media_domains, Setting::from_default(None));
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
            mm_allowed_media_domains: Some(vec!["media.example".to_string()]),
            mm_allow_private_media_hosts: true,
            mm_turbojpeg: Some(TurboJpegPolicy::Require),
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
                ("SMG_MM_ALLOW_PRIVATE_MEDIA_HOSTS", "0"),
                ("SMG_MM_TURBOJPEG", "prefer"),
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
        assert_eq!(
            settings.allowed_media_domains,
            flag(Some(vec!["media.example".to_string()]))
        );
        assert_eq!(settings.allow_private_media_hosts, flag(true));
        assert_eq!(settings.turbojpeg, flag(TurboJpegPolicy::Require));
    }

    /// The media fetch policy's defaults: any public host, no private ones;
    /// an empty allow-list is no allow-list, and only a true-ish opt-in
    /// opens private hosts.
    #[test]
    fn media_fetch_policy_defaults_and_opt_in_spellings() {
        let config = RouterConfig {
            mm_allowed_media_domains: Some(Vec::new()),
            ..RouterConfig::default()
        };
        let settings = MultimodalSettings::resolve_with(&config, no_env).unwrap();
        assert_eq!(settings.allowed_media_domains, Setting::from_default(None));
        assert_eq!(
            settings.allow_private_media_hosts,
            Setting::from_default(false)
        );
        for (value, expected) in [
            ("1", from_env(true)),
            ("yes", from_env(true)),
            ("ON", from_env(true)),
            ("0", Setting::from_default(false)),
            ("false", Setting::from_default(false)),
            ("sometimes", Setting::from_default(false)),
        ] {
            let settings = MultimodalSettings::resolve_with(
                &RouterConfig::default(),
                env(&[("SMG_MM_ALLOW_PRIVATE_MEDIA_HOSTS", value)]),
            )
            .unwrap();
            assert_eq!(settings.allow_private_media_hosts, expected, "{value}");
        }
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

    /// `require` misread as the default would serve the very pixels the
    /// operator asked to refuse, so an unreadable policy stops start-up; a
    /// blank value is unset, and either value reads in any case. The name
    /// is not a legacy one: no deprecation line is involved.
    #[test]
    fn an_unreadable_turbojpeg_env_stops_startup_but_blank_is_the_default() {
        let config = RouterConfig::default();
        let error =
            MultimodalSettings::resolve_with(&config, env(&[("SMG_MM_TURBOJPEG", "required")]))
                .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("SMG_MM_TURBOJPEG"), "{message}");
        assert!(message.contains("expected prefer|require"), "{message}");

        let blank =
            MultimodalSettings::resolve_with(&config, env(&[("SMG_MM_TURBOJPEG", "  ")])).unwrap();
        assert_eq!(
            blank.turbojpeg,
            Setting::from_default(TurboJpegPolicy::Prefer)
        );
        let required =
            MultimodalSettings::resolve_with(&config, env(&[("SMG_MM_TURBOJPEG", "REQUIRE")]))
                .unwrap();
        assert_eq!(required.turbojpeg, from_env(TurboJpegPolicy::Require));
    }

    /// Without the strict names in the reader, the other settings resolve
    /// on their own: what the never-seeded path falls back to when the
    /// placement or the TurboJPEG env is unreadable.
    #[test]
    fn the_other_settings_survive_an_unreadable_placement_env() {
        let full = env(&[
            ("SMG_MM_PROCESSING", "routers"),
            ("SMG_MM_TURBOJPEG", "mandatory"),
            ("SMG_MM_PIXEL_CACHE_MB", "256"),
            ("SMG_LOG_MM_TIMING", "1"),
        ]);
        let config = RouterConfig::default();
        assert!(MultimodalSettings::resolve_with(&config, &full).is_err());
        let rest = MultimodalSettings::resolve_with(&config, |name| {
            (!STRICT_ENV.contains(&name)).then(|| full(name)).flatten()
        })
        .unwrap();
        assert_eq!(
            rest.processing,
            Setting::from_default(MmProcessingMode::Auto)
        );
        assert_eq!(
            rest.turbojpeg,
            Setting::from_default(TurboJpegPolicy::Prefer)
        );
        assert_eq!(rest.pixel_cache_mb, from_env(256));
        assert_eq!(rest.log_mm_timing, from_env(true));
    }

    /// The numeric and boolean env knobs keep their lenient reading: an
    /// unreadable value is ignored, and a switch spelling outside `1`, `true`,
    /// `yes`, `on` leaves the switch off.
    #[test]
    fn unreadable_numeric_env_and_unknown_switch_values_fall_back() {
        let settings = MultimodalSettings::resolve_with(
            &RouterConfig::default(),
            env(&[
                ("SMG_MM_PIXEL_CACHE_MB", "lots"),
                ("SMG_MM_PIXEL_RDMA", "maybe"),
                ("SMG_RDMA_SLOT_TTL_S", "soon"),
                ("SMG_LOG_MM_TIMING", "0"),
                ("SMG_RDMA_LISTEN_IP", "   "),
                ("SMG_MM_ONE_PASS_WIRE", "sometimes"),
            ]),
        )
        .unwrap();
        assert_eq!(settings, MultimodalSettings::default());
    }

    /// The one-pass wire write is on unless its switch says off: the off
    /// spellings turn it off, the on spellings leave it on (an opt-in from
    /// before it became the default still reads), anything else is the
    /// default; and the switch reads its own name only.
    #[test]
    fn the_one_pass_wire_switch_is_on_by_default_and_reads_off_as_well() {
        let config = RouterConfig::default();
        let unset = MultimodalSettings::resolve_with(&config, no_env).unwrap();
        assert_eq!(unset.one_pass_wire, Setting::from_default(true));
        for (value, expected) in [
            ("0", from_env(false)),
            ("false", from_env(false)),
            ("no", from_env(false)),
            ("OFF", from_env(false)),
            (" Off ", from_env(false)),
            ("1", from_env(true)),
            ("true", from_env(true)),
            ("yes", from_env(true)),
            ("ON", from_env(true)),
            ("", Setting::from_default(true)),
            ("sometimes", Setting::from_default(true)),
        ] {
            let settings =
                MultimodalSettings::resolve_with(&config, env(&[("SMG_MM_ONE_PASS_WIRE", value)]))
                    .unwrap();
            assert_eq!(settings.one_pass_wire, expected, "{value:?}");
            assert_eq!(
                settings.pixel_rdma,
                Setting::from_default(false),
                "{value:?} reaches no other switch"
            );
        }
    }

    /// The env listener address keeps the lenient reading of the other env
    /// knobs: one that is not an IP address is left unused (the lane stays
    /// inline) where the flag's value is refused at validation; a readable
    /// one is kept trimmed, as a flag value is.
    #[test]
    fn an_env_listen_ip_that_is_not_an_address_falls_back() {
        let settings = MultimodalSettings::resolve_with(
            &RouterConfig::default(),
            env(&[("SMG_RDMA_LISTEN_IP", "999.1.1.1")]),
        )
        .unwrap();
        assert_eq!(settings.rdma_listen_ip, Setting::from_default(None));

        let settings = MultimodalSettings::resolve_with(
            &RouterConfig::default(),
            env(&[("SMG_RDMA_LISTEN_IP", " fd00::1 ")]),
        )
        .unwrap();
        assert_eq!(
            settings.rdma_listen_ip,
            from_env(Some("fd00::1".to_string()))
        );

        let config = RouterConfig {
            rdma_listen_ip: Some(" 10.0.0.7 ".to_string()),
            ..RouterConfig::default()
        };
        let settings = MultimodalSettings::resolve_with(&config, no_env).unwrap();
        assert_eq!(settings.rdma_listen_ip, flag(Some("10.0.0.7".to_string())));
    }

    /// The three off-by-default env switches read one set of spellings: `1`,
    /// `true`, `yes` and `on` in any case turn them on, anything else leaves
    /// the default; the one-pass wire switch keeps its own default under
    /// every one of their names. (`SMG_MM_PIXEL_RDMA` took `1` and `true`
    /// only before.)
    #[test]
    fn every_env_switch_reads_the_same_spellings() {
        for name in [
            "SMG_MM_PIXEL_RDMA",
            "SMG_LOG_MM_TIMING",
            "SMG_MM_ALLOW_PRIVATE_MEDIA_HOSTS",
        ] {
            for (value, on) in [
                ("1", true),
                ("true", true),
                ("yes", true),
                ("ON", true),
                (" True ", true),
                ("0", false),
                ("false", false),
                ("off", false),
                ("sometimes", false),
                ("", false),
            ] {
                let settings = MultimodalSettings::resolve_with(
                    &RouterConfig::default(),
                    env(&[(name, value)]),
                )
                .unwrap();
                let [pixel_rdma, log_mm_timing, allow_private] = [
                    settings.pixel_rdma,
                    settings.log_mm_timing,
                    settings.allow_private_media_hosts,
                ];
                let expected = if on {
                    from_env(true)
                } else {
                    Setting::from_default(false)
                };
                let off = Setting::from_default(false);
                let (actual, others) = match name {
                    "SMG_MM_PIXEL_RDMA" => (pixel_rdma, [log_mm_timing, allow_private]),
                    "SMG_LOG_MM_TIMING" => (log_mm_timing, [pixel_rdma, allow_private]),
                    _ => (allow_private, [pixel_rdma, log_mm_timing]),
                };
                assert_eq!(actual, expected, "{name}={value:?}");
                // Each switch reads its own name only.
                assert_eq!(others, [off, off], "{name}={value:?}");
                assert_eq!(
                    settings.one_pass_wire,
                    Setting::from_default(true),
                    "{name}={value:?}"
                );
            }
        }
    }
}
