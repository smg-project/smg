//! The `/metrics` read of an engine without a servicer: the Prometheus text
//! parser and the gauge tables of the engine families that expose their
//! scheduler as gauges, read by the load monitor's HTTP fallback.
//!
//! A table answers `None` for a body without the KV-usage gauge: that gauge
//! is the signal load-aware routing acts on, and without it `token_usage`
//! would default to `0.0` and make an engine of unknown pressure look idle,
//! so the read says unavailable instead.
//!
//! The policy controller crate holds the same parser and tables for its own
//! scrape driver (`ingest::scrape`); a gauge rename is an edit in both until
//! the two share one home.

use std::collections::HashMap;

use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};

/// Minimal Prometheus text-format scraper: metric name -> sample values
/// (one per label-set). Only the flat `name{labels} value` / `name value`
/// gauge lines that the engine load readers look up are collected; `#`
/// comment lines and unparsable values are skipped, and histogram or
/// summary buckets are simply never queried by name.
pub(super) struct PromScrape {
    samples: HashMap<String, Vec<f64>>,
}

impl PromScrape {
    pub(super) fn parse(text: &str) -> Self {
        let mut samples: HashMap<String, Vec<f64>> = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Line shape: `name{labels} value [timestamp]`. Split the head
            // (`name{labels}`) from the tail at the end of the label block so
            // spaces inside quoted label values don't confuse the split; fall
            // back to the first whitespace for unlabeled metrics. Then take
            // the FIRST tail token as the value, ignoring any optional
            // trailing timestamp.
            let (head, tail) = match line.rfind('}') {
                Some(close) => (&line[..=close], &line[close + 1..]),
                None => match line.split_once(char::is_whitespace) {
                    Some(split) => split,
                    None => continue,
                },
            };
            let Some(value) = tail.split_whitespace().next() else {
                continue;
            };
            let Ok(value) = value.parse::<f64>() else {
                continue;
            };
            // A value that is not a finite number is no sample. Kept, it
            // would pass for a present gauge and poison a sum or a mean, and
            // a KV usage of NaN converts to zero: an engine of unknown
            // pressure would look idle.
            if !value.is_finite() {
                continue;
            }
            let name = head.split('{').next().unwrap_or(head).trim();
            samples.entry(name.to_string()).or_default().push(value);
        }
        Self { samples }
    }

    /// Sum across all label-set samples for `name` (0.0 if absent). Right for
    /// additive counts like running/waiting requests across DP ranks.
    pub(super) fn sum(&self, name: &str) -> f64 {
        self.samples
            .get(name)
            .map(|v| v.iter().sum())
            .unwrap_or(0.0)
    }

    /// Mean across all label-set samples for `name` (0.0 if absent). Right for
    /// ratios like KV-cache usage that must not be double-counted.
    pub(super) fn mean(&self, name: &str) -> f64 {
        match self.samples.get(name) {
            Some(v) if !v.is_empty() => v.iter().sum::<f64>() / v.len() as f64,
            _ => 0.0,
        }
    }

    /// True when at least one sample exists for `name`.
    pub(super) fn has(&self, name: &str) -> bool {
        self.samples.get(name).is_some_and(|v| !v.is_empty())
    }
}

/// The first family's table: the KV-cache usage ratio (0.0 to 1.0) maps
/// onto `token_usage`; the family renamed the gauge between its v0 and v1
/// engines (`gpu_cache_usage_perc`, then `kv_cache_usage_perc`), so accept
/// either.
pub(super) fn vllm(text: &str) -> Option<SchedulerLoadSnapshot> {
    let m = PromScrape::parse(text);
    let kv_usage = ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"]
        .into_iter()
        .find(|name| m.has(name))?;
    Some(SchedulerLoadSnapshot {
        num_running_reqs: m.sum("vllm:num_requests_running") as i32,
        num_waiting_reqs: m.sum("vllm:num_requests_waiting") as i32,
        token_usage: m.mean(kv_usage),
        cache_hit_rate: m.mean("vllm:gpu_prefix_cache_hit_rate"),
        ..Default::default()
    })
}

/// The second family's table: the KV-usage ratio (0.0 to 1.0) is
/// `<prefix>token_usage`, where the family moved from a colon to an
/// underscore in its metric prefix between releases, so detect whichever
/// is present and use it throughout.
pub(super) fn sglang(text: &str) -> Option<SchedulerLoadSnapshot> {
    let m = PromScrape::parse(text);
    let prefix = ["sglang:", "sglang_"]
        .into_iter()
        .find(|p| m.has(&format!("{p}token_usage")))?;
    Some(SchedulerLoadSnapshot {
        num_running_reqs: m.sum(&format!("{prefix}num_running_reqs")) as i32,
        num_waiting_reqs: m.sum(&format!("{prefix}num_queue_reqs")) as i32,
        token_usage: m.mean(&format!("{prefix}token_usage")),
        gen_throughput: m.mean(&format!("{prefix}gen_throughput")),
        cache_hit_rate: m.mean(&format!("{prefix}cache_hit_rate")),
        utilization: m.mean(&format!("{prefix}utilization")),
        ..Default::default()
    })
}

/// A single scheduler snapshot as a one-rank report: `/metrics` already
/// aggregates across the data-parallel ranks, so one synthetic rank is the
/// right shape.
///
/// Such a snapshot carries the KV-usage ratio (`token_usage`) but no
/// absolute token counts (`max_total_num_tokens` and `num_used_tokens` stay
/// `0`), so `WorkerLoadResponse::has_absolute_token_data` reports `false`
/// for it, which keeps it out of a rank cache.
pub(super) fn single_rank(snapshot: SchedulerLoadSnapshot) -> WorkerLoadResponse {
    WorkerLoadResponse {
        dp_rank_count: 1,
        loads: vec![snapshot],
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Trimmed sample mirroring the fields of a live engine's `/metrics`
    // endpoint of the first family.
    const KV_USAGE_GAUGES: &str = r#"
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="llama"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="llama"} 5.0
# HELP vllm:gpu_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="llama"} 0.75
vllm:generation_tokens_total{model_name="llama"} 123456.0
"#;

    const TOKEN_USAGE_GAUGES: &str = r#"
sglang:num_running_reqs{model="llama"} 2.0
sglang:num_queue_reqs{model="llama"} 4.0
sglang:token_usage{model="llama"} 0.42
sglang:utilization{model="llama"} 0.9
"#;

    #[test]
    fn parses_gauges_ignoring_comments_and_labels() {
        let m = PromScrape::parse(KV_USAGE_GAUGES);
        assert!(m.has("vllm:num_requests_running"));
        assert_eq!(m.sum("vllm:num_requests_running"), 3.0);
        assert_eq!(m.sum("vllm:num_requests_waiting"), 5.0);
        assert_eq!(m.mean("vllm:gpu_cache_usage_perc"), 0.75);
        assert!(!m.has("vllm:does_not_exist"));
        assert_eq!(m.sum("vllm:does_not_exist"), 0.0);
        assert_eq!(m.mean("vllm:does_not_exist"), 0.0);
    }

    #[test]
    fn sum_adds_ranks_mean_averages_ratios() {
        // Two DP-rank series for the same gauge name.
        let text = "g{dp=\"0\"} 10\ng{dp=\"1\"} 30\nr{dp=\"0\"} 0.2\nr{dp=\"1\"} 0.8\n";
        let m = PromScrape::parse(text);
        assert_eq!(m.sum("g"), 40.0); // counts add
        assert_eq!(m.mean("r"), 0.5); // ratios average
    }

    #[test]
    fn parse_ignores_trailing_timestamp_and_label_spaces() {
        // Optional trailing timestamp must not be read as the value, and a
        // space inside a quoted label value must not split the head early.
        let text = "vllm:gpu_cache_usage_perc{model_name=\"my model\"} 0.75 1719849600000\n";
        let m = PromScrape::parse(text);
        assert_eq!(m.mean("vllm:gpu_cache_usage_perc"), 0.75);
    }

    #[test]
    fn the_kv_usage_table_maps_the_gauges_onto_a_one_rank_report() {
        let snap = vllm(KV_USAGE_GAUGES).expect("the KV-usage gauge is present");
        let resp = single_rank(snap);
        assert_eq!(resp.dp_rank_count, 1);
        assert_eq!(resp.effective_token_usage(), 0.75);
        assert_eq!(resp.loads[0].num_running_reqs, 3);
        assert_eq!(resp.loads[0].num_waiting_reqs, 5);
    }

    #[test]
    fn the_kv_usage_gauge_is_accepted_under_its_newer_name() {
        // The family renamed the gauge from `gpu_cache_usage_perc` to
        // `kv_cache_usage_perc`; the reader accepts either name.
        let v1 = "vllm:kv_cache_usage_perc{model_name=\"llama\"} 0.6\n";
        let snap = vllm(v1).expect("the v1 gauge name is accepted");
        assert_eq!(snap.token_usage, 0.6);
    }

    #[test]
    fn the_token_usage_table_maps_the_gauges_onto_the_snapshot() {
        let snap = sglang(TOKEN_USAGE_GAUGES).expect("the KV-usage gauge is present");
        assert_eq!(snap.token_usage, 0.42);
        assert_eq!(snap.num_running_reqs, 2);
        assert_eq!(snap.num_waiting_reqs, 4);
        assert_eq!(snap.utilization, 0.9);
    }

    #[test]
    fn the_token_usage_prefix_is_accepted_in_its_newer_spelling() {
        // The family moved its metric prefix from a colon to an underscore.
        let v054 = "sglang_token_usage{model=\"llama\"} 0.5\n\
                    sglang_num_running_reqs{model=\"llama\"} 7\n\
                    sglang_num_queue_reqs{model=\"llama\"} 1\n";
        let snap = sglang(v054).expect("the underscore prefix is accepted");
        assert_eq!(snap.token_usage, 0.5);
        assert_eq!(snap.num_running_reqs, 7);
        assert_eq!(snap.num_waiting_reqs, 1);
    }

    #[test]
    fn a_body_without_the_kv_usage_gauge_reads_unavailable() {
        let running_only = "vllm:num_requests_running{model_name=\"llama\"} 3.0\n";
        assert!(vllm(running_only).is_none());
        assert!(sglang("sglang:num_running_reqs 2\n").is_none());
        assert!(vllm("").is_none());
    }

    #[test]
    fn a_value_that_is_not_a_finite_number_is_no_sample() {
        let m = PromScrape::parse("g 1\ng NaN\ng +Inf\ng -Inf\nr NaN\n");
        assert_eq!(m.sum("g"), 1.0);
        assert_eq!(m.mean("g"), 1.0);
        assert!(!m.has("r"));
    }

    #[test]
    fn a_kv_usage_gauge_that_is_not_a_number_reads_unavailable() {
        // Unknown pressure is not idle: a body with no finite KV-usage
        // sample reads as one without the gauge.
        let first = "vllm:num_requests_running{model_name=\"llama\"} 3.0\n\
                     vllm:kv_cache_usage_perc{model_name=\"llama\"} NaN\n";
        assert!(vllm(first).is_none());
        let second = "sglang:num_running_reqs 2\nsglang:token_usage NaN\n";
        assert!(sglang(second).is_none());
    }
}
