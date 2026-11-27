use std::sync::OnceLock;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use smg_grand_central::document::{AppliedSpec, PolicySpec, ShadowSpec};

/// The lib test binary's one process-wide recorder, for records made on
/// threads no local recorder covers (the blocking pool). Installed once;
/// every other test keeps its own local recorder, which takes precedence
/// on its thread.
pub(crate) fn global_recorder() -> &'static PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE.get_or_init(|| {
        PrometheusBuilder::new()
            .install_recorder()
            .expect("the lib test binary's one global recorder")
    })
}

/// The scrape a test renders under its own recorder, with the two
/// readers of it: the worker's test support owns them.
pub(crate) use smg_worker::test_support::assert_metric;
pub(crate) use smg_worker::test_support::{render_with_recorder, sorted_series};

/// A policy document of two bindings. The first is of another model
/// and names no shadow. The second is the `round_robin` preset's, of
/// every model, with the `random` preset's profile as its shadow.
pub(crate) fn shadowing_document() -> PolicySpec {
    let preset = |name| PolicySpec::preset(name).expect("a stock preset");
    let mut document = preset("round_robin");
    document.profiles.extend(preset("random").profiles);
    let mut ahead = document.bindings[0].clone();
    ahead.matcher.model = "another-model".to_owned();
    document.bindings[0].shadow = Some(ShadowSpec::Profile("random".to_owned()));
    document.bindings.insert(0, ahead);
    document
}

/// [`shadowing_document`], as an engine is built from it.
pub(crate) fn shadowing_spec() -> AppliedSpec {
    AppliedSpec::validate(&shadowing_document()).expect("a valid policy document")
}

/// The shadow series of the second binding's single requests in a
/// scrape ([`side_kinds`]).
pub(crate) fn shadow_kinds(rendered: &str) -> [Option<u64>; 4] {
    side_kinds(rendered, 1, "single")
}

/// The shadow series of the binding at `binding`, asked beside `side`,
/// in a scrape, by kind: agreed, other target, other outcome, skipped.
/// `None` where the scrape holds no such series.
pub(crate) fn side_kinds(rendered: &str, binding: u32, side: &str) -> [Option<u64>; 4] {
    ["agreed", "other-target", "other-outcome", "skipped"].map(|kind| {
        let labels = [
            format!(r#"binding="{binding}""#),
            format!(r#"side="{side}""#),
            format!(r#"kind="{kind}""#),
        ];
        let of_kind = |line: &&str| {
            line.starts_with("smg_policy_shadow_decisions_total{")
                && labels.iter().all(|label| line.contains(label.as_str()))
        };
        let line = rendered.lines().find(of_kind)?;
        line.rsplit(' ').next()?.parse().ok()
    })
}

/// The series of the panics Grand Central held and counted in a
/// scrape, by kind: decision, event, upkeep. `None` where the scrape
/// holds no such series.
pub(crate) fn fault_kinds(rendered: &str) -> [Option<u64>; 3] {
    ["decision", "event", "upkeep"].map(|kind| {
        let label = format!(r#"kind="{kind}""#);
        let of_kind =
            |line: &&str| line.starts_with("smg_policy_faults_total{") && line.contains(&label);
        let line = rendered.lines().find(of_kind)?;
        line.rsplit(' ').next()?.parse().ok()
    })
}
