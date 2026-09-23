use openmetrics_parser::{MetricFamily, MetricsExposition, PrometheusType, PrometheusValue};
use tracing::warn;

#[derive(Debug)]
pub struct MetricPack {
    pub labels: Vec<(String, String)>,
    pub metrics_text: String,
}

type PrometheusExposition = MetricsExposition<PrometheusType, PrometheusValue>;
type PrometheusFamily = MetricFamily<PrometheusType, PrometheusValue>;

/// Aggregate Prometheus metrics scraped from multiple sources into a unified one.
///
/// `openmetrics_parser`'s Prometheus grammar requires metric names to start with
/// a lowercase letter followed by `[a-z0-9_]`, while the Prometheus text format
/// allows colons, and engines prefix their metrics with one (`sglang:`, `vllm:`).
/// Colons are swapped for an escape sequence before parsing and restored in the
/// rendered output, so `/engine_metrics` exposes the exact metric names, HELP
/// text, and label values the engines exported. A label name is the one place
/// a colon is illegal, so there the escape becomes `_` instead: restoring it
/// would make the whole page unscrapeable.
pub fn aggregate_metrics(metric_packs: Vec<MetricPack>) -> anyhow::Result<String> {
    let colon_escape = unused_colon_escape(&metric_packs);
    let mut expositions = vec![];
    for metric_pack in metric_packs {
        let metrics_text = metric_pack.metrics_text.replace(':', &colon_escape);

        let exposition = match openmetrics_parser::prometheus::parse_prometheus(&metrics_text) {
            Ok(x) => x,
            Err(err) => {
                warn!(
                    "aggregate_metrics error when parsing text: pack={:?} err={:?}",
                    metric_pack, err
                );
                continue;
            }
        };
        let exposition = sanitize_label_names(exposition, &colon_escape);
        let exposition = transform_metrics(exposition, &metric_pack.labels);
        expositions.push(exposition);
    }

    let text = try_reduce(expositions, merge_exposition)?
        .map(|x| format!("{x}").replace(&colon_escape, ":"))
        .unwrap_or_default();
    Ok(text)
}

/// The escape that stands in for `:` while parsing: `xsmgcolon{n}z` for the
/// first `n` that occurs nowhere in the input. A literal escape can itself
/// occur in a valid metric name, HELP text, or label, and picking an unused
/// one keeps restoring colons from rewriting it. Only the first character is
/// `x`, so adjacent escapes cannot combine into a false match.
fn unused_colon_escape(metric_packs: &[MetricPack]) -> String {
    let mut n = 0u64;
    loop {
        let escape = format!("xsmgcolon{n}z");
        let used = metric_packs.iter().any(|pack| {
            pack.metrics_text.contains(&escape)
                || pack
                    .labels
                    .iter()
                    .any(|(key, value)| key.contains(&escape) || value.contains(&escape))
        });
        if !used {
            return escape;
        }
        n += 1;
    }
}

/// Rebuild every family whose label names carry the colon escape with `_`
/// in its place: label values keep their positions, so the samples move over
/// unchanged.
fn sanitize_label_names(
    mut exposition: PrometheusExposition,
    colon_escape: &str,
) -> PrometheusExposition {
    let families = std::mem::take(&mut exposition.families);
    for (name, family) in families {
        let family = if family
            .get_label_names()
            .iter()
            .any(|n| n.contains(colon_escape))
        {
            let label_names: Vec<String> = family
                .get_label_names()
                .iter()
                .map(|n| n.replace(colon_escape, "_"))
                .collect();
            // Two names that sanitize to one would render a sample with
            // duplicate label names, which invalidates the whole scrape.
            let mut unique = label_names.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != label_names.len() {
                warn!(
                    "aggregate_metrics dropped family {name}: label names collide once sanitized"
                );
                continue;
            }
            let rebuilt = PrometheusFamily::new(
                family.family_name.clone(),
                label_names,
                family.family_type.clone(),
                family.help.clone(),
                family.unit.clone(),
            );
            match rebuilt.with_samples(family.into_iter_samples()) {
                Ok(rebuilt) => rebuilt,
                Err(err) => {
                    warn!(
                        "aggregate_metrics could not sanitize label names: family={name} err={err:?}"
                    );
                    continue;
                }
            }
        } else {
            family
        };
        exposition.families.insert(name, family);
    }
    exposition
}

fn transform_metrics(
    mut exposition: PrometheusExposition,
    extra_labels: &[(String, String)],
) -> PrometheusExposition {
    for family in exposition.families.values_mut() {
        *family = family.with_labels(extra_labels.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    exposition
}

fn merge_exposition(
    a: PrometheusExposition,
    b: PrometheusExposition,
) -> anyhow::Result<PrometheusExposition> {
    let mut ans = a;
    for (name, family_b) in b.families {
        let family_merged = if let Some(family_a) = ans.families.remove(&name) {
            merge_family(family_a, family_b)?
        } else {
            family_b
        };
        ans.families.insert(name, family_merged);
    }
    Ok(ans)
}

fn merge_family(a: PrometheusFamily, b: PrometheusFamily) -> anyhow::Result<PrometheusFamily> {
    // When label schemas differ (e.g., prefill vs decode workers with different DP
    // configs), pad missing labels with empty strings so both families share the
    // same label set before merging.
    let (a, b) = align_labels(a, b);
    a.with_samples(b.into_iter_samples())
        .map_err(|e| anyhow::anyhow!("failed to merge samples: {e:?}"))
}

/// Ensure two families have identical label sets by padding missing labels with `""`.
/// Returns both families unchanged if labels already match.
fn align_labels(a: PrometheusFamily, b: PrometheusFamily) -> (PrometheusFamily, PrometheusFamily) {
    let a_names = a.get_label_names();
    let b_names = b.get_label_names();
    if a_names == b_names {
        return (a, b);
    }

    let pad = |family: PrometheusFamily, other_names: &[String]| -> PrometheusFamily {
        let own_names = family.get_label_names();
        let missing: Vec<(&str, &str)> = other_names
            .iter()
            .filter(|n| !own_names.contains(n))
            .map(|n| (n.as_str(), ""))
            .collect();
        if missing.is_empty() {
            family
        } else {
            family.with_labels(missing)
        }
    };

    // Clone names before moving families into pad()
    let a_names = a_names.to_vec();
    let b_names = b_names.to_vec();
    (pad(a, &b_names), pad(b, &a_names))
}

fn try_reduce<I, T, E, F>(iterable: I, f: F) -> Result<Option<T>, E>
where
    I: IntoIterator<Item = T>,
    F: FnMut(T, T) -> Result<T, E>,
{
    let mut it = iterable.into_iter();
    let first = match it.next() {
        None => return Ok(None),
        Some(x) => x,
    };

    Ok(Some(it.try_fold(first, f)?))
}
