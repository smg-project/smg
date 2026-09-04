//! Static per-engine capability table, overridable with `rl.*` worker labels.

use std::collections::HashMap;

use openai_protocol::worker::RuntimeType;
use serde::Serialize;

/// What an engine can do for RL control, as reported by discovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Capabilities {
    /// `"static"` from the built-in table, `"label"` when any `rl.*` label overrode it.
    pub source: &'static str,
    pub pause_modes: Vec<String>,
    pub update_from: Vec<String>,
    pub abort: bool,
    pub flush_cache: bool,
    pub sleep_wake: bool,
    pub reports_weight_version: bool,
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn static_for(runtime: RuntimeType) -> Capabilities {
    match runtime {
        RuntimeType::Sglang => Capabilities {
            source: "static",
            pause_modes: strings(&["abort", "retract", "in_place"]),
            update_from: strings(&["disk", "tensor", "distributed"]),
            abort: true,
            flush_cache: true,
            sleep_wake: true,
            reports_weight_version: true,
        },
        RuntimeType::Vllm => Capabilities {
            source: "static",
            pause_modes: strings(&["abort", "wait", "keep"]),
            update_from: strings(&["disk", "distributed"]),
            abort: false,
            flush_cache: false,
            sleep_wake: true,
            reports_weight_version: false,
        },
        _ => Capabilities {
            source: "static",
            pause_modes: Vec::new(),
            update_from: Vec::new(),
            abort: false,
            flush_cache: false,
            sleep_wake: false,
            reports_weight_version: false,
        },
    }
}

fn list_label(labels: &HashMap<String, String>, key: &str) -> Option<Vec<String>> {
    labels.get(key).map(|v| {
        v.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    })
}

fn bool_label(labels: &HashMap<String, String>, key: &str) -> Option<bool> {
    labels
        .get(key)
        .map(|v| v.trim().eq_ignore_ascii_case("true"))
}

/// Static table row for `runtime`, with any `rl.*` label overrides applied.
pub fn capabilities_for(runtime: RuntimeType, labels: &HashMap<String, String>) -> Capabilities {
    let mut caps = static_for(runtime);
    let mut overridden = false;
    if let Some(v) = list_label(labels, "rl.pause_modes") {
        caps.pause_modes = v;
        overridden = true;
    }
    if let Some(v) = list_label(labels, "rl.update_from") {
        caps.update_from = v;
        overridden = true;
    }
    for (key, slot) in [
        ("rl.abort", &mut caps.abort),
        ("rl.flush_cache", &mut caps.flush_cache),
        ("rl.sleep_wake", &mut caps.sleep_wake),
        (
            "rl.reports_weight_version",
            &mut caps.reports_weight_version,
        ),
    ] {
        if let Some(v) = bool_label(labels, key) {
            *slot = v;
            overridden = true;
        }
    }
    if overridden {
        caps.source = "label";
    }
    caps
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use openai_protocol::worker::RuntimeType;

    use super::*;

    fn labels(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn sglang_and_vllm_static_rows() {
        let s = capabilities_for(RuntimeType::Sglang, &HashMap::new());
        assert_eq!(s.source, "static");
        assert_eq!(s.pause_modes, ["abort", "retract", "in_place"]);
        assert_eq!(s.update_from, ["disk", "tensor", "distributed"]);
        assert!(s.abort && s.flush_cache && s.sleep_wake && s.reports_weight_version);

        let v = capabilities_for(RuntimeType::Vllm, &HashMap::new());
        assert_eq!(v.pause_modes, ["abort", "wait", "keep"]);
        assert_eq!(v.update_from, ["disk", "distributed"]);
        assert!(!v.abort && !v.flush_cache && v.sleep_wake && !v.reports_weight_version);
    }

    #[test]
    fn other_runtimes_have_no_capabilities() {
        for rt in [
            RuntimeType::Trtllm,
            RuntimeType::TokenSpeed,
            RuntimeType::Mlx,
            RuntimeType::Generic,
            RuntimeType::External,
            RuntimeType::Unspecified,
        ] {
            let c = capabilities_for(rt, &HashMap::new());
            assert!(
                c.pause_modes.is_empty() && c.update_from.is_empty(),
                "{rt:?}"
            );
            assert!(!c.abort && !c.flush_cache && !c.sleep_wake && !c.reports_weight_version);
        }
    }

    #[test]
    fn labels_override_and_mark_source() {
        let c = capabilities_for(
            RuntimeType::Trtllm,
            &labels(&[
                ("rl.pause_modes", "abort, keep"),
                ("rl.update_from", "disk"),
                ("rl.abort", "true"),
                ("rl.flush_cache", "TRUE"),
                ("rl.sleep_wake", "false"),
                ("rl.reports_weight_version", "yes"),
            ]),
        );
        assert_eq!(c.source, "label");
        assert_eq!(c.pause_modes, ["abort", "keep"]);
        assert_eq!(c.update_from, ["disk"]);
        assert!(c.abort && c.flush_cache && !c.sleep_wake);
        assert!(!c.reports_weight_version, "`yes` is not `true`");
    }

    #[test]
    fn partial_override_keeps_static_rest() {
        let c = capabilities_for(RuntimeType::Sglang, &labels(&[("rl.abort", "false")]));
        assert_eq!(c.source, "label");
        assert!(!c.abort);
        assert!(c.flush_cache);
        assert_eq!(c.pause_modes, ["abort", "retract", "in_place"]);
    }
}
