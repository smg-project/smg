//! The gRPC disaggregation mode (Regular/PD/EPD).

use smg_worker::identity::basic::{ConnectionMode, WorkerType};

use crate::{
    config::types::{RouterConfig, RoutingMode},
    routers::{
        common::snapshot::RoutingPool,
        grpc::{
            common::stages::WorkerSelectionMode,
            context::{ExecutionPlanKind, PlanSettings},
        },
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Regular,
    PrefillDecode,
    EncodePrefillDecode,
}

impl Mode {
    /// The pools whose workers a request of this mode can be served by, the
    /// pools its pixels can reach: the regular pool; the prefill and decode
    /// pools; the encode, prefill and decode pools. The multimodal wire dtype
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

    /// The role whose workers take a request's pixels: the regular workers;
    /// a pair's prefill workers, whose side takes the request's media
    /// whole; the encode workers of an encode/prefill/decode fleet. The
    /// rule that refuses images for a model without a vision encoder judges
    /// these workers alone.
    pub(crate) fn pixels_go_to(self) -> WorkerType {
        match self {
            Mode::Regular => WorkerType::Regular,
            Mode::PrefillDecode => WorkerType::Prefill,
            Mode::EncodePrefillDecode => WorkerType::Encode,
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
    /// Both settings as the one value every request-building stage holds.
    pub(crate) fn plan_settings(self) -> PlanSettings {
        PlanSettings {
            inject_pd_metadata: self.inject_pd_metadata(),
            plan_kind: self.plan_kind(),
        }
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
#[path = "../../../tests/unit/routers/grpc/mode/tests.rs"]
mod tests;
