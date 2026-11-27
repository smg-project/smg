use std::collections::HashMap;

use super::*;

fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Result<Option<String>> {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    move |name| Ok(map.get(name).cloned())
}

/// Both budgets resolve independently and reject malformed/nonpositive input.
#[test]
fn media_fetch_timeout_precedence_and_validation() {
    let config = RouterConfig::default();
    let defaults = MultimodalSettings::resolve_with(&config, no_env).unwrap();
    assert_eq!(defaults.fetch_timeout_ms.value, 10_000);
    assert_eq!(defaults.video_fetch_timeout_ms.value, 30_000);
    let resolved = MultimodalSettings::resolve_with(
        &config,
        env(&[
            ("SMG_MM_FETCH_TIMEOUT_MS", "60000"),
            ("SMG_MM_VIDEO_FETCH_TIMEOUT_MS", "120000"),
        ]),
    )
    .unwrap();
    assert_eq!(resolved.fetch_timeout_ms.value, 60_000);
    assert_eq!(resolved.video_fetch_timeout_ms.value, 120_000);
    assert_eq!(resolved.fetch_timeout_ms.source, SettingSource::Env);
    let explicit = RouterConfig {
        mm_fetch_timeout_ms: Some(1),
        mm_video_fetch_timeout_ms: Some(2),
        ..config.clone()
    };
    let resolved = MultimodalSettings::resolve_with(
        &explicit,
        env(&[
            ("SMG_MM_FETCH_TIMEOUT_MS", "bad"),
            ("SMG_MM_VIDEO_FETCH_TIMEOUT_MS", "bad"),
        ]),
    )
    .unwrap();
    assert_eq!(resolved.fetch_timeout_ms.value, 1);
    assert_eq!(resolved.video_fetch_timeout_ms.value, 2);
    for name in ["SMG_MM_FETCH_TIMEOUT_MS", "SMG_MM_VIDEO_FETCH_TIMEOUT_MS"] {
        for raw in ["0", "-1", "bad", "1.5"] {
            let error = MultimodalSettings::resolve_with(&config, env(&[(name, raw)])).unwrap_err();
            assert!(error.to_string().contains(name), "{error}");
        }
    }
    let zero = RouterConfig {
        mm_fetch_timeout_ms: Some(0),
        ..config
    };
    let error = MultimodalSettings::resolve_with(&zero, no_env).unwrap_err();
    assert_eq!(error.to_string(), "mm_fetch_timeout_ms must be positive");
    let zero = RouterConfig {
        mm_fetch_timeout_ms: None,
        mm_video_fetch_timeout_ms: Some(0),
        ..zero
    };
    let error = MultimodalSettings::resolve_with(&zero, no_env).unwrap_err();
    assert_eq!(
        error.to_string(),
        "mm_video_fetch_timeout_ms must be positive"
    );
}

/// A bad timeout in embedded use must not relax other valid strict settings.
#[test]
fn unseeded_media_fetch_timeout_preserves_valid_settings() {
    for bad in [ENV_FETCH_TIMEOUT_MS, ENV_VIDEO_FETCH_TIMEOUT_MS] {
        for invalid in [Some("bad"), Some("0"), Some("-1"), None] {
            let valid = env(&[
                (ENV_PROCESSING, "worker"),
                (ENV_TURBOJPEG, "require"),
                (ENV_FETCH_TIMEOUT_MS, "60000"),
                (ENV_VIDEO_FETCH_TIMEOUT_MS, "120000"),
                (ENV_PIXEL_CACHE_MB, "32"),
            ]);
            let resolved = resolve_unseeded_with(|name| {
                if name == bad {
                    match invalid {
                        Some(value) => Ok(Some(value.to_string())),
                        None => anyhow::bail!("not Unicode"),
                    }
                } else {
                    valid(name)
                }
            });
            assert_eq!(resolved.processing.value, MmProcessingMode::Worker);
            assert_eq!(resolved.processing.source, SettingSource::Env);
            assert_eq!(resolved.turbojpeg.value, TurboJpegPolicy::Require);
            assert_eq!(resolved.turbojpeg.source, SettingSource::Env);
            assert_eq!(resolved.pixel_cache_mb.value, 32);
            assert_eq!(
                resolved.fetch_timeout_ms.value,
                if bad == ENV_FETCH_TIMEOUT_MS {
                    10000
                } else {
                    60000
                }
            );
            assert_eq!(
                resolved.video_fetch_timeout_ms.value,
                if bad == ENV_VIDEO_FETCH_TIMEOUT_MS {
                    30000
                } else {
                    120000
                }
            );
        }
    }
}

fn no_env(name: &str) -> Result<Option<String>> {
    env(&[])(name)
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
    let error = MultimodalSettings::resolve_with(&config, env(&[("SMG_MM_PROCESSING", "routers")]))
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
    let error = MultimodalSettings::resolve_with(&config, env(&[("SMG_MM_TURBOJPEG", "required")]))
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
        MultimodalSettings::resolve_with(&config, env(&[("SMG_MM_TURBOJPEG", "REQUIRE")])).unwrap();
    assert_eq!(required.turbojpeg, from_env(TurboJpegPolicy::Require));
}

/// A process with no setting of its own (a worker pipeline whose
/// launcher sent none) reads `SMG_MM_TURBOJPEG` under the same strictness:
/// an unreadable value is refused rather than read as the default, a
/// readable one applies in any case, unset or blank is `prefer`.
#[test]
fn a_process_without_the_setting_reads_the_env_strictly() {
    assert_eq!(
        turbojpeg_policy(Some(" Require ".to_string())).unwrap(),
        TurboJpegPolicy::Require
    );
    assert_eq!(turbojpeg_policy(None).unwrap(), TurboJpegPolicy::Prefer);
    assert_eq!(
        turbojpeg_policy(Some("  ".to_string())).unwrap(),
        TurboJpegPolicy::Prefer
    );
    let error = turbojpeg_policy(Some("required".to_string())).unwrap_err();
    assert!(
        format!("{error:#}").contains("SMG_MM_TURBOJPEG"),
        "{error:#}"
    );
}

/// The real unseeded path defaults invalid policies without discarding valid
/// strict timeouts, decoder policies or the lenient settings.
#[test]
fn the_other_settings_survive_an_unreadable_placement_env() {
    for policy in ["mandatory", "require"] {
        let full = env(&[
            (ENV_PROCESSING, "routers"),
            (ENV_TURBOJPEG, policy),
            (ENV_FETCH_TIMEOUT_MS, "60000"),
            (ENV_PIXEL_CACHE_MB, "256"),
            (ENV_LOG_MM_TIMING, "1"),
        ]);
        assert!(MultimodalSettings::resolve_with(&RouterConfig::default(), &full).is_err());
        let rest = resolve_unseeded_with(&full);
        assert_eq!(
            rest.processing,
            Setting::from_default(MmProcessingMode::Auto)
        );
        assert_eq!(
            rest.turbojpeg,
            if policy == "require" {
                from_env(TurboJpegPolicy::Require)
            } else {
                Setting::from_default(TurboJpegPolicy::Prefer)
            }
        );
        assert_eq!(rest.fetch_timeout_ms, from_env(60000));
        assert_eq!(rest.pixel_cache_mb, from_env(256));
        assert_eq!(rest.log_mm_timing, from_env(true));
    }
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
            let settings =
                MultimodalSettings::resolve_with(&RouterConfig::default(), env(&[(name, value)]))
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

/// The process reader: unset is `None`, text is itself, and a value set
/// to bytes that are not text is refused with the variable named when
/// the name is strict (read as unset, `require` would hand the policy
/// its default) and leaves its default, like an unparsable value, when
/// it is not.
#[cfg(unix)]
#[test]
fn an_env_value_that_is_not_text_is_refused_by_name_where_strict_and_unset_is_none() {
    use std::{env::VarError, ffi::OsString, os::unix::ffi::OsStringExt};

    let not_text = || VarError::NotUnicode(OsString::from_vec(vec![b'r', 0xff]));
    assert_eq!(
        env_text(ENV_TURBOJPEG, Err(VarError::NotPresent)).unwrap(),
        None
    );
    assert_eq!(
        env_text(ENV_TURBOJPEG, Ok("require".to_string())).unwrap(),
        Some("require".to_string())
    );
    for name in STRICT_ENV {
        let error = env_text(name, Err(not_text())).unwrap_err();
        assert!(format!("{error:#}").contains(name), "{error:#}");
    }
    assert_eq!(env_text(ENV_PIXEL_CACHE_MB, Err(not_text())).unwrap(), None);
}

/// The settings carry the reader's refusal: a `SMG_MM_TURBOJPEG` the
/// reader refuses stops the resolution, named, instead of reading as the
/// default.
#[test]
fn a_turbojpeg_env_the_reader_refuses_stops_the_settings() {
    let refused = |name: &str| -> Result<Option<String>> {
        if name == ENV_TURBOJPEG {
            Err(anyhow::anyhow!(
                "{name} is set to a value that is not valid UTF-8"
            ))
        } else {
            Ok(None)
        }
    };
    let error = MultimodalSettings::resolve_with(&RouterConfig::default(), refused).unwrap_err();
    assert!(format!("{error:#}").contains(ENV_TURBOJPEG), "{error:#}");
}
