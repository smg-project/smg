use super::*;

const SALT: &str = "salt";
const KEY: &str = "key";

fn partitioned(mut req: sg::GenerateRequest) -> sg::GenerateRequest {
    req.cache_partition = Some(Box::new(sg::CachePartition {
        cache_salt: Some(SALT.into()),
        extra_key: Some(KEY.into()),
    }));
    req
}

/// The next request the scheduler receives has both values.
async fn recv_partitioned(h: &mut Harness) -> TokenizedGenerateReqInput {
    let forwarded = recv_add(&mut h.engine_in).await;
    assert_eq!(
        (
            forwarded.cache_salt.as_deref(),
            forwarded.extra_key.as_deref()
        ),
        (Some(SALT), Some(KEY))
    );
    forwarded
}

/// The prefill side, the decode side and a worker of neither advertise both
/// fields, and give the same two values to every choice.
#[tokio::test]
async fn every_role_forwards_and_advertises_the_cache_partition() {
    for role in ["null", "prefill", "decode"] {
        let mut h = harness(SglangModelInfo {
            server_args_json: serde_json::json!({"disaggregation_mode": role}).to_string(),
            ..model_info()
        })
        .await;
        let args = h
            .client
            .get_server_info(sg::GetServerInfoRequest {})
            .await
            .unwrap()
            .into_inner()
            .server_args
            .expect("server_args");
        assert_eq!(
            args.fields["cache_partition_fields"].kind,
            Some(Kind::StringValue("cache_salt,extra_key".to_string())),
            "{role}"
        );
        let mut req = partitioned(generate_request("partition", false, Vec::new()));
        req.sampling_params.as_mut().unwrap().n = 2;
        req.disaggregated_params = Some(sg::DisaggregatedParams {
            bootstrap_host: "127.0.0.1".into(),
            bootstrap_port: 8998,
            bootstrap_room: 1,
        });
        let stream = h.client.generate(req).await.unwrap();
        for _ in 0..2 {
            recv_partitioned(&mut h).await;
        }
        drop(stream);
        h.server.stop(Duration::from_secs(5)).expect("clean stop");
    }
}

/// A request whose media the encoders serve keeps its cache partition beside
/// the dispatch's slots.
#[tokio::test]
async fn an_encode_dispatched_request_keeps_its_cache_partition() {
    let dispatcher = ScriptedDispatcher::new(DispatchAnswer::Encoders);
    let mut h = harness_with(vision_model(1), dispatcher.configure(1)).await;
    let req = partitioned(media_reference_request("partition", 1));
    let stream = h.client.generate(req).await.unwrap();
    let forwarded = recv_partitioned(&mut h).await;
    assert_eq!(forwarded.need_wait_for_mm_inputs, Some(true));
    assert!(forwarded.encoder_urls.is_some());
    drop(stream);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}
