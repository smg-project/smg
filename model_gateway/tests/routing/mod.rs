//! Routing integration tests

pub mod cache_aware_backward_compat_test;
pub mod grpc_completion_batch_test;
pub mod header_forwarding_test;
pub mod header_routing_hints_test;
pub mod load_balancing_test;
pub mod manual_routing_test;
pub mod model_alias_test;
pub mod payload_size_test;
pub mod pd_routing_test;
pub mod policy_registry_integration;
pub mod power_of_two_test;
pub mod prefix_hash_test;
pub mod service_discovery_test;
pub mod stream_relay_disconnect_test;
pub mod stream_request_body_test;
pub mod test_openai_routing;
pub mod test_pd_routing;
pub mod upstream_http2_test;
pub mod worker_management_test;
