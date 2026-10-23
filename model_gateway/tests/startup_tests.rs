//! Start-up integration tests: the gateway binary, started with a configuration, accepts or refuses it

pub use smg_test_harness as common;
#[path = "common/gateway_launch.rs"]
pub mod gateway_launch;

mod startup;
