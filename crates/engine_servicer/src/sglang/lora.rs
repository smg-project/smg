//! Immutable adapters preloaded at startup. Public names and internal IDs
//! resolve to the exact ID the scheduler loaded; unknown selections fail
//! before dispatch instead of reaching its adapter lookup.

use std::collections::HashMap;

use serde::Deserialize;
use smg_grpc_client::sglang_proto::LoadedLoRaAdapter;
use tonic::Status;

use crate::ServicerError;

#[derive(Default)]
pub(super) struct PreloadedLoras {
    enabled: bool,
    ids: HashMap<String, String>,
    pub(super) adapters: Vec<LoadedLoRaAdapter>,
}

#[derive(Deserialize)]
struct Adapter {
    lora_id: String,
    lora_name: String,
    lora_path: String,
    #[serde(default)]
    pinned: bool,
}

impl PreloadedLoras {
    pub(super) fn from_server_args(json: &str) -> Result<Self, ServicerError> {
        let invalid = |message| ServicerError::InvalidConfig(format!("preloaded LoRA: {message}"));
        if json.trim().is_empty() {
            return Ok(Self::default());
        }
        let args: serde_json::Value =
            serde_json::from_str(json).map_err(|err| invalid(format!("{err}")))?;
        let mut loras = Self {
            enabled: args.get("enable_lora").and_then(serde_json::Value::as_bool) == Some(true),
            ..Self::default()
        };
        let paths = args.get("lora_paths").filter(|value| !value.is_null());
        let adapters: Vec<Adapter> = paths
            .map(|value| serde_json::from_value(value.clone()))
            .transpose()
            .map_err(|err| invalid(format!("{err}")))?
            .unwrap_or_default();
        for adapter in adapters {
            if [&adapter.lora_id, &adapter.lora_name, &adapter.lora_path]
                .iter()
                .any(|value| value.trim().is_empty())
            {
                return Err(invalid("ID, name and path must be non-empty".to_string()));
            }
            for alias in [&adapter.lora_name, &adapter.lora_id] {
                if let Some(previous) = loras.ids.insert(alias.clone(), adapter.lora_id.clone()) {
                    if previous != adapter.lora_id {
                        return Err(invalid(format!("ambiguous name or ID: {alias}")));
                    }
                }
            }
            loras.adapters.push(LoadedLoRaAdapter {
                lora_id: adapter.lora_id,
                lora_name: adapter.lora_name,
                lora_path: adapter.lora_path,
                pinned: adapter.pinned,
            });
        }
        if !loras.enabled && !loras.adapters.is_empty() {
            return Err(invalid("adapters require --enable-lora".to_string()));
        }
        Ok(loras)
    }

    pub(super) fn resolve(&self, selector: &str) -> Result<String, Status> {
        if selector.is_empty() {
            return Ok(String::new());
        }
        if !self.enabled {
            return Err(Status::invalid_argument(
                "LoRA is not enabled; launch the worker with --enable-lora",
            ));
        }
        self.ids.get(selector).cloned().ok_or_else(|| {
            Status::invalid_argument(format!("Unknown preloaded LoRA adapter: {selector}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_or_ambiguous_preloaded_adapters_fail_startup() {
        for paths in [
            serde_json::json!(["unresolved-cli-argument"]),
            serde_json::json!([{"lora_id": "id", "lora_name": "name"}]),
            serde_json::json!([{"lora_id": "", "lora_name": "name", "lora_path": "path"}]),
            serde_json::json!([
                {"lora_id": "id-a", "lora_name": "name", "lora_path": "path"},
                {"lora_id": "id-b", "lora_name": "id-a", "lora_path": "other"},
            ]),
        ] {
            let args = serde_json::json!({"enable_lora": true, "lora_paths": paths});
            assert!(matches!(
                PreloadedLoras::from_server_args(&args.to_string()),
                Err(ServicerError::InvalidConfig(_))
            ));
        }
    }
}
