//! Read-only observation of engine control calls that pass through the
//! proxy: what a 2xx answer to a route tells SMG about the engine. The
//! forwarded bytes are never modified.

use serde::Deserialize;
use serde_json::Value;
use tracing::warn;

use crate::{table::ControlState, version::Version};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Observation {
    Version(Version),
    Control(ControlState),
}

/// Only the version fields are read; everything else in a refit body is
/// skipped without allocation.
#[derive(Deserialize)]
struct VersionProbe {
    #[serde(default)]
    weight_version: Option<Value>,
    #[serde(default)]
    new_version: Option<Value>,
}

/// A passthrough version gets exactly the validation the explicit API write
/// gets (see [`Version::validated`]), so a refit body cannot put a value into
/// the table that `POST /v1/rl/workers/<id>/version` would have refused. A
/// rejected value observes nothing and says why, named by the route that
/// carried it. An absent field stays silent: it says nothing about the engine.
fn version_field(route: &str, value: Option<Value>) -> Option<Version> {
    let raw = match value? {
        Value::String(s) => s,
        Value::Number(n) => n.to_string(),
        other => {
            warn!(
                target: "smg_rl",
                route, reason = "not_a_string_or_number", kind = kind_of(&other),
                "ignoring a weight version an engine reported in an unusable JSON type"
            );
            return None;
        }
    };
    match Version::validated(&raw) {
        Ok(version) => Some(version),
        Err(e) => {
            warn!(
                target: "smg_rl",
                route, reason = e.reason(),
                "ignoring an invalid weight version an engine reported"
            );
            None
        }
    }
}

/// The JSON type name for the rejection log. The value itself is never
/// logged: a refit body is caller-supplied and may be large.
fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
        Value::String(_) | Value::Number(_) => "scalar",
    }
}

/// What a successful call to `path` (the proxied engine route, no leading
/// slash) with request `body` says about the engine, if anything.
pub fn observe(path: &str, body: &[u8]) -> Option<Observation> {
    match path {
        "update_weights_from_disk"
        | "update_weights_from_tensor"
        | "update_weights_from_distributed" => {
            let probe: VersionProbe = serde_json::from_slice(body).ok()?;
            version_field(path, probe.weight_version).map(Observation::Version)
        }
        "update_weight_version" => {
            let probe: VersionProbe = serde_json::from_slice(body).ok()?;
            version_field(path, probe.new_version).map(Observation::Version)
        }
        "pause_generation" | "pause" => Some(Observation::Control(ControlState::Paused)),
        "continue_generation" | "resume" | "resume_memory_occupation" | "wake_up" => {
            Some(Observation::Control(ControlState::Active))
        }
        "release_memory_occupation" | "sleep" => Some(Observation::Control(ControlState::Asleep)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refit_routes_report_the_version_field_when_present() {
        let v = observe(
            "update_weights_from_disk",
            br#"{"model_path": "/ckpt", "weight_version": "42"}"#,
        );
        assert_eq!(v, Some(Observation::Version(Version::parse("42"))));
        let v = observe(
            "update_weights_from_tensor",
            br#"{"serialized_named_tensors": ["..."], "weight_version": 7}"#,
        );
        assert_eq!(v, Some(Observation::Version(Version::parse("7"))));
        let v = observe(
            "update_weights_from_distributed",
            br#"{"names": [], "weight_version": "s3"}"#,
        );
        assert_eq!(v, Some(Observation::Version(Version::parse("s3"))));
        assert_eq!(
            observe(
                "update_weight_version",
                br#"{"new_version": "43", "abort_all_requests": false}"#
            ),
            Some(Observation::Version(Version::parse("43")))
        );
    }

    #[test]
    fn refit_routes_without_a_version_observe_nothing() {
        assert_eq!(
            observe("update_weights_from_disk", br#"{"model_path": "/ckpt"}"#),
            None
        );
        assert_eq!(
            observe("update_weights_from_disk", br#"{"weight_version": ""}"#),
            None
        );
        assert_eq!(
            observe("update_weights_from_disk", br#"{"weight_version": null}"#),
            None
        );
        assert_eq!(observe("update_weights_from_disk", b"not json"), None);
        assert_eq!(
            observe("update_weight_version", br#"{"weight_version": "1"}"#),
            None,
            "wrong field for this route"
        );
    }

    /// Passthrough versions clear the same bar as the API write: a value the
    /// control plane would answer `invalid_version` for observes nothing
    /// rather than landing in the table through the proxy.
    #[test]
    fn a_rejected_passthrough_version_observes_nothing() {
        let long = "x".repeat(129);
        for body in [
            r#"{"weight_version": "   "}"#.to_string(),
            r#"{"weight_version": true}"#.to_string(),
            r#"{"weight_version": ["42"]}"#.to_string(),
            r#"{"weight_version": {"v": "42"}}"#.to_string(),
            format!(r#"{{"weight_version": "{long}"}}"#),
            r#"{"weight_version": "v1\u0007"}"#.to_string(),
            r#"{"weight_version": "step 7"}"#.to_string(),
        ] {
            assert_eq!(
                observe("update_weights_from_disk", body.as_bytes()),
                None,
                "{body}"
            );
        }
        // The same bar on the other field shape.
        for body in [
            r#"{"new_version": "  "}"#,
            r#"{"new_version": false}"#,
            r#"{"new_version": []}"#,
            r#"{"new_version": "v1\u0007"}"#,
        ] {
            assert_eq!(
                observe("update_weight_version", body.as_bytes()),
                None,
                "{body}"
            );
        }
    }

    #[test]
    fn control_routes_map_to_states_regardless_of_body() {
        for (path, state) in [
            ("pause_generation", ControlState::Paused),
            ("pause", ControlState::Paused),
            ("continue_generation", ControlState::Active),
            ("resume", ControlState::Active),
            ("release_memory_occupation", ControlState::Asleep),
            ("sleep", ControlState::Asleep),
            ("resume_memory_occupation", ControlState::Active),
            ("wake_up", ControlState::Active),
        ] {
            assert_eq!(
                observe(path, b"{}"),
                Some(Observation::Control(state)),
                "{path}"
            );
            assert_eq!(
                observe(path, b""),
                Some(Observation::Control(state)),
                "{path}"
            );
        }
        assert_eq!(observe("flush_cache", b"{}"), None);
        assert_eq!(observe("server_info", b""), None);
    }
}
