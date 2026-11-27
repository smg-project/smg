use tracing::Level;

use super::*;

fn targets(directives: &str) -> Targets {
    targets_for(directives).unwrap_or_else(|| panic!("{directives:?} must use Targets"))
}

fn hint(directives: &str) -> Option<LevelFilter> {
    build_filter_layer(directives).and_then(|layer| layer.max_level_hint())
}

#[test]
fn plain_target_levels_use_targets() {
    let targets = targets("warn,smg=info");
    assert!(targets.would_enable("smg", &Level::INFO));
    assert!(!targets.would_enable("smg", &Level::DEBUG));
    assert!(targets.would_enable("hyper", &Level::WARN));
    assert!(!targets.would_enable("hyper", &Level::INFO));
    assert_eq!(targets.default_level(), Some(LevelFilter::WARN));
}

#[test]
fn env_filter_decides_validity() {
    // Whatever `EnvFilter` rejected before this change is still rejected,
    // whether or not `Targets` would have taken it (`*smg=debug`, ` smg=debug`).
    for directives in [
        "warn,smg=info",
        "warn,",
        "warn, smg=debug",
        "*smg=debug",
        " , ",
        "info",
        "smg[request]=debug",
        "warn,smg=",
    ] {
        assert_eq!(
            build_filter_layer(directives).is_some(),
            EnvFilter::try_new(directives).is_ok(),
            "{directives:?}"
        );
    }
    assert!(targets_for("*smg=debug").is_none() || build_filter_layer("*smg=debug").is_none());
}

#[test]
fn forms_the_parsers_read_differently_stay_with_env_filter() {
    for directives in ["smg[request]=debug", "warn,smg[{model}]=trace", "warn,smg="] {
        assert!(
            targets_for(directives).is_none(),
            "{directives:?} must not be parsed as Targets"
        );
        assert!(
            build_filter_layer(directives).is_some(),
            "{directives:?} must still produce an EnvFilter"
        );
    }
    // `EnvFilter` reads an empty level as TRACE; `Targets` would have read ERROR.
    assert_eq!(hint("warn,smg="), Some(LevelFilter::TRACE));
}

#[test]
fn a_trailing_comma_is_not_an_empty_target() {
    let trailing = targets("warn,");
    assert_eq!(trailing.default_level(), Some(LevelFilter::WARN));
    assert_eq!(
        trailing.iter().count(),
        0,
        "no per-target directive, let alone an empty one"
    );
    assert!(!trailing.would_enable("hyper", &Level::INFO));
}

/// The default filter drops the audio decoder's probe lines, ERROR
/// included; the decoder's other targets keep the dependencies' WARN.
#[test]
fn the_default_filter_turns_the_audio_probes_target_off() {
    let targets = targets(&build_workspace_filter("info"));
    assert!(!targets.would_enable("symphonia_core::formats::probe", &Level::ERROR));
    assert!(targets.would_enable("symphonia_core::formats", &Level::WARN));
    assert!(targets.would_enable("smg", &Level::INFO));
}

/// The worker crate's lines take the configured level, as the gateway's
/// own do: its debug lines show at a debug level, and its warnings fall
/// silent at an error level.
#[test]
fn the_default_filter_gives_the_worker_crate_the_configured_level() {
    let debug = targets(&build_workspace_filter("debug"));
    assert!(debug.would_enable("smg_worker::periodic_task", &Level::DEBUG));
    assert!(debug.would_enable("smg_worker::jobs::queue", &Level::DEBUG));
    let error = targets(&build_workspace_filter("error"));
    assert!(!error.would_enable("smg_worker::load::metrics_aggregator", &Level::WARN));
    assert!(!error.would_enable("smg_worker::jobs::queue", &Level::WARN));
}

/// The discovery crate's lines take the configured level by the crate's
/// own entry. They do not rest on the gateway's entry, whose name is a
/// prefix of the crate's: with that entry out they still show.
#[test]
fn the_default_filter_names_the_discovery_crate_itself() {
    let filter = build_workspace_filter("info");
    let without_the_gateways_entry = filter.replace(",smg=info", "");
    assert_ne!(without_the_gateways_entry, filter);
    let shown = targets(&without_the_gateways_entry);
    assert!(shown.would_enable("smg_service_discovery::reconciler", &Level::INFO));
    assert!(!shown.would_enable("smg_service_discovery::reconciler", &Level::DEBUG));
    assert!(!shown.would_enable("smg::server", &Level::INFO));
}

#[test]
fn log_cap_follows_the_most_verbose_directive() {
    assert_eq!(hint("warn,smg=info"), Some(LevelFilter::INFO));
    assert_eq!(hint("warn,smg=debug,"), Some(LevelFilter::DEBUG));
    assert_eq!(hint("error"), Some(LevelFilter::ERROR));
}
