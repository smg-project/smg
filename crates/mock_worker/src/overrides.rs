//! Runtime overrides for what a gRPC worker reports about itself, keyed by
//! its port: whether it answers its health check as serving and the info it
//! reports (`server_args` entries the gateway flattens into worker labels,
//! and the model's `supports_vision`), set through the admin API
//! (`/admin/serving/{port}`, `/admin/info/{port}`); and whether its model
//! and server info RPCs answer, refuse as not implemented or fail, set in
//! process only (no admin route reaches `set_model_info` and
//! `set_server_info`). A bed uses the admin routes to change an engine's
//! reported settings between two health transitions, as a restarted or
//! late-loading engine does.

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
        Arc, Mutex, OnceLock, PoisonError,
    },
};

use serde_json::Value;

/// What an info RPC of the worker answers: its fields (the default), an
/// `Unimplemented` status (an engine without the call), or a failure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InfoAnswer {
    #[default]
    Served,
    Unimplemented,
    Failing,
}

impl InfoAnswer {
    /// The status this answer refuses `rpc` with, when it is a refusal.
    pub fn refusal(self, rpc: &str) -> Result<(), tonic::Status> {
        match self {
            Self::Served => Ok(()),
            Self::Unimplemented => Err(tonic::Status::unimplemented(format!(
                "the mock engine has no {rpc}"
            ))),
            Self::Failing => Err(tonic::Status::internal(format!(
                "the mock engine's {rpc} is down"
            ))),
        }
    }

    fn from_u8(raw: u8) -> Self {
        match raw {
            1 => Self::Unimplemented,
            2 => Self::Failing,
            _ => Self::Served,
        }
    }
}

/// What one worker reports, as far as the overrides reach.
pub struct WorkerOverrides {
    serving: AtomicBool,
    info: Mutex<BTreeMap<String, Value>>,
    loads_polls: AtomicUsize,
    model_info: AtomicU8,
    server_info: AtomicU8,
}

impl WorkerOverrides {
    /// Serving, with no info, both info RPCs served and no load poll yet.
    fn fresh() -> Self {
        Self {
            serving: AtomicBool::new(true),
            info: Mutex::new(BTreeMap::new()),
            loads_polls: AtomicUsize::new(0),
            model_info: AtomicU8::new(InfoAnswer::Served as u8),
            server_info: AtomicU8::new(InfoAnswer::Served as u8),
        }
    }

    /// Whether the health check answers `healthy` (the default).
    pub fn serving(&self) -> bool {
        self.serving.load(Ordering::Relaxed)
    }

    /// How many `GetLoads` the worker has answered: what a gateway's load
    /// polls cost the engine.
    pub fn loads_polls(&self) -> usize {
        self.loads_polls.load(Ordering::Relaxed)
    }

    pub(crate) fn note_loads_poll(&self) {
        self.loads_polls.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_serving(&self, serving: bool) {
        self.serving.store(serving, Ordering::Relaxed);
    }

    /// What `GetModelInfo` answers (served by default).
    pub fn model_info(&self) -> InfoAnswer {
        InfoAnswer::from_u8(self.model_info.load(Ordering::Relaxed))
    }

    pub fn set_model_info(&self, answer: InfoAnswer) {
        self.model_info.store(answer as u8, Ordering::Relaxed);
    }

    /// What `GetServerInfo` answers (served by default).
    pub fn server_info(&self) -> InfoAnswer {
        InfoAnswer::from_u8(self.server_info.load(Ordering::Relaxed))
    }

    pub fn set_server_info(&self, answer: InfoAnswer) {
        self.server_info.store(answer as u8, Ordering::Relaxed);
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
    Arc::clone(
        map.entry(port)
            .or_insert_with(|| Arc::new(WorkerOverrides::fresh())),
    )
}

/// Fresh overrides for a worker about to serve on `port`, in place of any
/// that a finished worker on the same port left. A test that binds port 0
/// can be handed a port an earlier test's worker served on, and would
/// otherwise inherit that worker's answers (a failing model info, a patched
/// info, a health check answering not serving). Take them after binding and
/// before the worker serves, so the worker and its caller hold the same
/// overrides.
pub fn fresh_for_port(port: u16) -> Arc<WorkerOverrides> {
    let map = OVERRIDES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut map = map.lock().unwrap_or_else(PoisonError::into_inner);
    let fresh = Arc::new(WorkerOverrides::fresh());
    map.insert(port, Arc::clone(&fresh));
    fresh
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

    /// A worker on a port that an earlier worker served on starts from fresh
    /// overrides, not from the answers that worker was left with, and shares
    /// them with whoever asks for the port's overrides after it.
    #[test]
    fn a_reused_port_starts_from_fresh_overrides() {
        let stale = for_port(65_002);
        stale.set_serving(false);
        stale.set_model_info(InfoAnswer::Failing);
        stale.set_server_info(InfoAnswer::Unimplemented);
        stale.patch_info(
            json!({"kv_connector": "MooncakeConnector"})
                .as_object()
                .unwrap(),
        );
        stale.note_loads_poll();

        let fresh = fresh_for_port(65_002);
        assert!(fresh.serving(), "a fresh worker answers serving");
        assert_eq!(
            fresh.model_info(),
            InfoAnswer::Served,
            "a fresh worker serves its model info"
        );
        assert_eq!(
            fresh.server_info(),
            InfoAnswer::Served,
            "a fresh worker serves its server info"
        );
        assert!(fresh.info().is_empty(), "a fresh worker reports no info");
        assert_eq!(
            fresh.loads_polls(),
            0,
            "a fresh worker has answered no load poll"
        );
        assert!(
            Arc::ptr_eq(&fresh, &for_port(65_002)),
            "the worker and its caller hold the same overrides"
        );
    }
}
