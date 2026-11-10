//! What a gRPC worker reports about its model: the sampling defaults of the
//! model directory's `generation_config.json`, as an engine servicer reports
//! them, for the gateway to complete a request's omitted sampling parameters.

use mock_worker::config::Config;

mod common;

#[tokio::test]
async fn grpc_model_info_carries_the_models_sampling_defaults() {
    let client = common::start(Config {
        model_id: "a-model".to_string(),
        default_sampling_params_json: r#"{"temperature":0.7,"top_k":20}"#.to_string(),
        ..Config::default()
    })
    .await;
    let info = client.get_model_info().await.expect("model info");
    assert_eq!(info.model_path, "a-model");
    assert_eq!(
        info.default_sampling_params_json,
        r#"{"temperature":0.7,"top_k":20}"#
    );
}

#[tokio::test]
async fn grpc_model_info_reports_no_sampling_defaults_when_the_model_declares_none() {
    let client = common::start(Config::default()).await;
    let info = client.get_model_info().await.expect("model info");
    assert_eq!(info.default_sampling_params_json, "");
}
