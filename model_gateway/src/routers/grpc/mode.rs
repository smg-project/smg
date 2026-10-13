//! The gRPC disaggregation mode (Regular/PD/EPD).

use crate::{
    config::types::{RouterConfig, RoutingMode},
    routers::grpc::{common::stages::WorkerSelectionMode, context::ExecutionPlanKind},
    worker::{ConnectionMode, RoutingPool},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Regular,
    PrefillDecode,
    EncodePrefillDecode,
}

impl Mode {
    /// The pools whose workers a request of this mode can be served by, the
    /// legs its pixels can reach: the regular pool; the prefill and decode
    /// legs; the encode, prefill and decode legs. The multimodal wire dtype
    /// is resolved over them before any worker is selected.
    pub(crate) fn multimodal_wire_pools(self) -> &'static [RoutingPool] {
        match self {
            Mode::Regular => &[RoutingPool::GrpcPipelineRegular],
            Mode::PrefillDecode => &[RoutingPool::GrpcPrefill, RoutingPool::GrpcDecode],
            Mode::EncodePrefillDecode => &[
                RoutingPool::GrpcEncode,
                RoutingPool::GrpcPrefill,
                RoutingPool::GrpcDecode,
            ],
        }
    }

    pub(crate) fn worker_selection(self) -> WorkerSelectionMode {
        match self {
            Mode::Regular => WorkerSelectionMode::Regular,
            Mode::PrefillDecode => WorkerSelectionMode::PrefillDecode,
            Mode::EncodePrefillDecode => WorkerSelectionMode::EncodePrefillDecode,
        }
    }
    pub(crate) fn plan_kind(self) -> ExecutionPlanKind {
        match self {
            Mode::Regular => ExecutionPlanKind::Single,
            Mode::PrefillDecode => ExecutionPlanKind::PrefillDecode,
            Mode::EncodePrefillDecode => ExecutionPlanKind::EncodePrefillDecode,
        }
    }
    /// PD only: EPD (TokenSpeed) injects encode bootstrap info instead, so this
    /// must stay false there to avoid double injection.
    pub(crate) fn inject_pd_metadata(self) -> bool {
        matches!(self, Mode::PrefillDecode)
    }
    /// Metrics/introspection label.
    pub(crate) fn router_type(self) -> &'static str {
        match self {
            Mode::Regular => "grpc",
            Mode::PrefillDecode => "grpc_pd",
            Mode::EncodePrefillDecode => "grpc_epd",
        }
    }
}

/// Derive the gRPC mode from config. `None` for non-gRPC backends.
pub(crate) fn grpc_mode(cfg: &RouterConfig) -> Option<Mode> {
    // ZMQ reuses the gRPC router pipeline (same ProtoStream/execution stages).
    if !matches!(
        cfg.connection_mode,
        ConnectionMode::Grpc | ConnectionMode::Zmq
    ) {
        return None;
    }
    match cfg.mode {
        RoutingMode::Regular { .. } => Some(Mode::Regular),
        RoutingMode::PrefillDecode { .. } => Some(Mode::PrefillDecode),
        RoutingMode::EncodePrefillDecode { .. } => Some(Mode::EncodePrefillDecode),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_maps_to_stage_params() {
        let cases = [
            (
                Mode::Regular,
                WorkerSelectionMode::Regular,
                ExecutionPlanKind::Single,
                false,
                "grpc",
            ),
            (
                Mode::PrefillDecode,
                WorkerSelectionMode::PrefillDecode,
                ExecutionPlanKind::PrefillDecode,
                true,
                "grpc_pd",
            ),
            (
                Mode::EncodePrefillDecode,
                WorkerSelectionMode::EncodePrefillDecode,
                ExecutionPlanKind::EncodePrefillDecode,
                false,
                "grpc_epd",
            ),
        ];
        for (m, ws, pk, inject, rt) in cases {
            assert_eq!(m.worker_selection(), ws, "worker_selection {m:?}");
            assert_eq!(m.plan_kind(), pk, "plan_kind {m:?}");
            assert_eq!(m.inject_pd_metadata(), inject, "inject_pd_metadata {m:?}");
            assert_eq!(m.router_type(), rt, "router_type {m:?}");
        }
    }
}
