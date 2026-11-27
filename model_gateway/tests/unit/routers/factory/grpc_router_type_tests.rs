use super::*;
use crate::routers::grpc::testing::grpc_ctx;

#[tokio::test]
async fn grpc_router_type_reflects_disaggregation_mode() {
    let cases = [
        (
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            "grpc",
        ),
        (
            RoutingMode::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: None,
                decode_policy: None,
            },
            "grpc_pd",
        ),
        (
            RoutingMode::EncodePrefillDecode {
                encode_urls: vec![],
                prefill_urls: vec![],
                decode_urls: vec![],
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            },
            "grpc_epd",
        ),
    ];

    for (mode, expected) in cases {
        let ctx = grpc_ctx(mode).await;
        let router = RouterFactory::create_router(&ctx)
            .await
            .expect("router creation");
        assert_eq!(router.router_type(), expected);
    }
}

#[tokio::test]
async fn igw_routers_preserve_disaggregated_role_policies() {
    for mode in [
        RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: Some(PolicyConfig::RoundRobin),
            decode_policy: Some(PolicyConfig::Passthrough),
        },
        RoutingMode::EncodePrefillDecode {
            encode_urls: vec![],
            prefill_urls: vec![],
            decode_urls: vec![],
            encode_policy: Some(PolicyConfig::PowerOfTwo {
                load_metric: Default::default(),
            }),
            prefill_policy: Some(PolicyConfig::RoundRobin),
            decode_policy: Some(PolicyConfig::Passthrough),
        },
    ] {
        let expected_encode_policy = mode
            .get_encode_policy(&PolicyConfig::ConsistentHashing)
            .name();
        let ctx = grpc_ctx(mode).await;
        RouterFactory::create_igw_routers(&ctx.router_config.policy, &ctx).await;

        let policies = &ctx.policy_registry;
        assert_eq!(policies.get_encode_policy().name(), expected_encode_policy);
        assert_eq!(policies.get_prefill_policy().name(), "round_robin");
        assert_eq!(policies.get_decode_policy().name(), "passthrough");
    }
}
