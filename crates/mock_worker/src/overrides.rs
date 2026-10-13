//! Runtime overrides for what a gRPC worker reports about itself, keyed by
//! its port and set through the admin API (`/admin/serving/{port}`,
//! `/admin/info/{port}`): whether it answers its health check as serving,
//! and the info it reports (`server_args` entries the gateway flattens into
//! worker labels, and the model's `supports_vision`). A bed uses them to
//! change an engine's reported settings between two health transitions, as
//! a restarted or late-loading engine does.

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock, PoisonError,
    },
};

use serde_json::Value;

/// What one worker reports, as far as the overrides reach.
pub struct WorkerOverrides {
    serving: AtomicBool,
    info: Mutex<BTreeMap<String, Value>>,
}

impl WorkerOverrides {
    /// Whether the health check answers `healthy` (the default).
    pub fn serving(&self) -> bool {
        self.serving.load(Ordering::Relaxed)
    }

    pub fn set_serving(&self, serving: bool) {
        self.serving.store(serving, Ordering::Relaxed);
    }

    /// Merge `patch` into the reported info; a `null` value removes the key.
    pub fn patch_info(&self, patch: &serde_json::Map<String, Value>) {
        let mut info = self.info.lock().unwrap_or_else(PoisonError::into_inner);
        for (key, value) in patch {
            if value.is_null() {
                info.remove(key);
            } else {
                info.insert(key.clone(), value.clone());
            }
        }
    }

    /// The reported info, as set.
    pub fn info(&self) -> BTreeMap<String, Value> {
        self.info
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// A boolean entry of the reported info, if set to one.
    pub fn info_bool(&self, key: &str) -> Option<bool> {
        self.info
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .and_then(Value::as_bool)
    }
}

static OVERRIDES: OnceLock<Mutex<BTreeMap<u16, Arc<WorkerOverrides>>>> = OnceLock::new();

/// The overrides of the worker on `port` (created serving, with no info, on
/// first use).
pub fn for_port(port: u16) -> Arc<WorkerOverrides> {
    let map = OVERRIDES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut map = map.lock().unwrap_or_else(PoisonError::into_inner);
    Arc::clone(map.entry(port).or_insert_with(|| {
        Arc::new(WorkerOverrides {
            serving: AtomicBool::new(true),
            info: Mutex::new(BTreeMap::new()),
        })
    }))
}

/// The reported info as the `server_args` struct of a server-info answer.
pub fn server_args(info: &BTreeMap<String, Value>) -> prost_types::Struct {
    let fields = info
        .iter()
        .map(|(key, value)| {
            let kind = match value {
                Value::Bool(b) => prost_types::value::Kind::BoolValue(*b),
                Value::Number(n) => {
                    prost_types::value::Kind::NumberValue(n.as_f64().unwrap_or(0.0))
                }
                Value::String(s) => prost_types::value::Kind::StringValue(s.clone()),
                other => prost_types::value::Kind::StringValue(other.to_string()),
            };
            (key.clone(), prost_types::Value { kind: Some(kind) })
        })
        .collect();
    prost_types::Struct { fields }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn overrides_start_serving_and_patch_in_place() {
        let overrides = for_port(65_001);
        assert!(overrides.serving());
        overrides.set_serving(false);
        assert!(!for_port(65_001).serving(), "one set of overrides per port");

        let patch =
            json!({"max_running_requests": 8, "supports_vision": true, "weight_version": "v2"});
        overrides.patch_info(patch.as_object().unwrap());
        assert_eq!(overrides.info_bool("supports_vision"), Some(true));
        let args = server_args(&overrides.info());
        assert!(matches!(
            args.fields.get("max_running_requests").and_then(|v| v.kind.as_ref()),
            Some(prost_types::value::Kind::NumberValue(n)) if *n == 8.0
        ));

        let removal = json!({"weight_version": null});
        overrides.patch_info(removal.as_object().unwrap());
        assert!(
            !overrides.info().contains_key("weight_version"),
            "a null removes the key"
        );
    }
}
