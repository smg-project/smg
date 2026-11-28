use super::*;

fn lora_model() -> SglangModelInfo {
    SglangModelInfo {
        server_args_json: serde_json::json!({
            "enable_lora": true,
            "lora_paths": [{
                "lora_name": "adapter", "lora_id": "engine-id",
                "lora_path": "/weights/adapter", "pinned": true,
            }],
        })
        .to_string(),
        ..model_info()
    }
}

#[tokio::test]
async fn preloaded_lora_names_and_ids_reach_every_choice() {
    let mut h = harness(lora_model()).await;
    let loaded = h
        .client
        .list_loaded_lo_ra_adapters(sg::ListLoadedLoRaAdaptersRequest {})
        .await
        .unwrap()
        .into_inner()
        .loaded_adapters;
    assert_eq!(
        loaded,
        vec![sg::LoadedLoRaAdapter {
            lora_name: "adapter".into(),
            lora_id: "engine-id".into(),
            lora_path: "/weights/adapter".into(),
            pinned: true,
        }]
    );
    for streaming in [false, true] {
        for selector in ["adapter", "engine-id", ""] {
            let mut req = generate_request("lora", streaming, Vec::new());
            req.lora_id = selector.into();
            req.sampling_params.as_mut().unwrap().n = 2;
            let mut stream = h.client.generate(req).await.unwrap().into_inner();
            for _ in 0..2 {
                let forwarded = recv_add(&mut h.engine_in).await;
                assert_eq!(
                    forwarded.lora_id.as_deref(),
                    (!selector.is_empty()).then_some("engine-id")
                );
                send(
                    &mut h.engine_out,
                    &batch(&forwarded.rid, vec![10], 1, Some("length"), None),
                )
                .await;
            }
            let mut completes = 0;
            while let Some(response) = stream.message().bounded().await.unwrap() {
                if let Some(sg::generate_response::Response::Complete(done)) = response.response {
                    assert_eq!(done.output_ids, vec![10]);
                    completes += 1;
                }
            }
            assert_eq!(completes, 2);
        }
    }
    let mut req = generate_request("unknown", false, Vec::new());
    req.lora_id = "unknown".into();
    let error = h.client.generate(req).await.unwrap_err();
    assert_eq!(error.code(), Code::InvalidArgument);
    assert_scheduler_idle(&mut h.engine_in).await;
    assert_eq!(h.server.active_requests(), 0);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

#[tokio::test]
async fn a_worker_without_lora_refuses_adapter_requests_before_dispatch() {
    let mut h = harness(model_info()).await;
    let mut req = generate_request("disabled", false, Vec::new());
    req.lora_id = "adapter".into();
    assert_eq!(
        h.client.generate(req).await.unwrap_err().code(),
        Code::InvalidArgument
    );
    assert_scheduler_idle(&mut h.engine_in).await;
    assert_eq!(h.server.active_requests(), 0);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}
