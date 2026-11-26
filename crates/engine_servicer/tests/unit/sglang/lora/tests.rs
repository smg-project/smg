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
