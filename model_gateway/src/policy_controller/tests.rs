//! Cases of the store's start and of the router mounted with it.

use std::{error::Error, sync::Arc};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::StatusCode,
};
use serde_json::Value;

use super::{start, undecided::UNDECIDED, StartError};
use crate::{
    app_context::testing::app_context_for_tests,
    config::{ManualAssignmentMode, PolicyConfig, RouterConfig},
    routers::{common::snapshot::WorkerRegistry, gateway::Gateway, RouterTrait},
    server::ServerConfig,
};

type Case = Result<(), Box<dyn Error>>;

/// `config` with the policy controller picked.
fn picked(mut config: RouterConfig) -> RouterConfig {
    config.policy_controller = true;
    config
}

#[test]
fn the_classic_policies_start_no_store() -> Case {
    assert!(start(&RouterConfig::default())?.is_none());
    Ok(())
}

#[test]
fn the_store_starts_with_the_document_the_routing_configuration_means() -> Case {
    let config = picked(RouterConfig::default());
    let store = start(&config)?.ok_or("the controller was picked and no store started")?;
    let world = store.world();
    assert_eq!(world.spec().generation(), 1);
    assert_eq!(
        world.spec().digest(),
        Some(config.policy_document()?.digest()?)
    );
    Ok(())
}

#[test]
fn a_setting_no_document_says_stops_the_start_by_its_name() -> Case {
    let mut config = picked(RouterConfig::default());
    config.routing_key_override.enabled = true;
    let refusal = start(&config)
        .err()
        .ok_or("the start took a setting no document says")?;
    assert!(matches!(refusal, StartError::Document(_)), "{refusal:?}");
    assert_eq!(
        refusal.to_string(),
        "routing_key_override.enabled = true: the controller holds no recorded form of the sticky sessions' rule yet"
    );
    Ok(())
}

#[test]
fn a_document_this_build_does_not_run_stops_the_start_by_its_name() -> Case {
    let mut policy = PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 1 << 20);
    if let PolicyConfig::CacheAware { overlap_decay, .. } = &mut policy {
        *overlap_decay = 0.5;
    }
    let mut config = picked(RouterConfig::default());
    config.policy = policy;
    let refusal = start(&config)
        .err()
        .ok_or("the start took a document this build does not run")?;
    assert!(matches!(refusal, StartError::Spec(_)), "{refusal:?}");
    assert_eq!(
        refusal.to_string(),
        "profiles.cache_aware.score[1].prefix-affinity: the prefix-affinity decay is not in this build"
    );
    Ok(())
}

#[test]
fn a_view_that_reads_the_gateways_own_header_stops_the_start_by_its_name() -> Case {
    // The tenant header is the gateway's own: the view of a document this
    // gateway hosts may not read it. The manual policy's view reads the
    // routing key headers.
    let mut config = picked(RouterConfig::default());
    config.policy = PolicyConfig::Manual {
        eviction_interval_secs: 60,
        max_idle_secs: 14_400,
        assignment_mode: ManualAssignmentMode::Random,
    };
    let tenant_header = config.tenant_resolution.tenant_header_name.clone();
    config.routing_key_override.headers = vec![tenant_header.clone()];
    let refusal = start(&config)
        .err()
        .ok_or("the start took a view that reads the tenant header")?;
    assert!(matches!(refusal, StartError::Spec(_)), "{refusal:?}");
    assert_eq!(
        refusal.to_string(),
        format!("view.headers[0]: header {tenant_header:?} is on the deny-list")
    );
    Ok(())
}

#[tokio::test]
async fn the_gateway_mounts_one_router_that_refuses_each_request_by_name() -> Case {
    let config = picked(RouterConfig::default());
    let context = app_context_for_tests(config.clone(), Arc::new(WorkerRegistry::new()), None);
    let gateway = Gateway::from_config(&ServerConfig::minimal(config), &context).await?;
    assert_eq!(gateway.router_count(), 1);
    let answer = gateway.health_generate(Request::new(Body::empty())).await;
    assert_eq!(answer.status(), StatusCode::NOT_IMPLEMENTED);
    let body = to_bytes(answer.into_body(), usize::MAX).await?;
    let error: Value = serde_json::from_slice(&body)?;
    assert_eq!(error["error"]["code"], UNDECIDED);
    Ok(())
}

#[tokio::test]
async fn the_classic_policies_mount_the_classic_router() -> Case {
    let config = RouterConfig::default();
    let context = app_context_for_tests(config.clone(), Arc::new(WorkerRegistry::new()), None);
    let gateway = Gateway::from_config(&ServerConfig::minimal(config), &context).await?;
    let router = gateway
        .select_router_for_request(None)
        .ok_or("no router is mounted under the classic policies")?;
    assert_ne!(router.router_type(), "policy_controller_undecided");
    Ok(())
}
