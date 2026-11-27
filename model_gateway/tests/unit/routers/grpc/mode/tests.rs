use super::*;

/// One row per mode: its worker selection, plan kind, PD-metadata
/// injection, router label and the pools its multimodal wire dtype is
/// resolved over.
type Case = (
    Mode,
    WorkerSelectionMode,
    ExecutionPlanKind,
    bool,
    &'static str,
    &'static [RoutingPool],
    WorkerType,
);

#[test]
fn mode_maps_to_stage_params() {
    let cases: [Case; 3] = [
        (
            Mode::Regular,
            WorkerSelectionMode::Regular,
            ExecutionPlanKind::Single,
            false,
            "grpc",
            &[RoutingPool::GrpcPipelineRegular],
            WorkerType::Regular,
        ),
        (
            Mode::PrefillDecode,
            WorkerSelectionMode::PrefillDecode,
            ExecutionPlanKind::PrefillDecode,
            true,
            "grpc_pd",
            &[RoutingPool::GrpcPrefill, RoutingPool::GrpcDecode],
            WorkerType::Prefill,
        ),
        (
            Mode::EncodePrefillDecode,
            WorkerSelectionMode::EncodePrefillDecode,
            ExecutionPlanKind::EncodePrefillDecode,
            false,
            "grpc_epd",
            &[
                RoutingPool::GrpcEncode,
                RoutingPool::GrpcPrefill,
                RoutingPool::GrpcDecode,
            ],
            WorkerType::Encode,
        ),
    ];
    for (m, ws, pk, inject, rt, pools, pixels) in cases {
        assert_eq!(m.worker_selection(), ws, "worker_selection {m:?}");
        assert_eq!(m.plan_kind(), pk, "plan_kind {m:?}");
        assert_eq!(m.inject_pd_metadata(), inject, "inject_pd_metadata {m:?}");
        assert_eq!(
            m.plan_settings(),
            PlanSettings {
                inject_pd_metadata: inject,
                plan_kind: pk,
            },
            "plan_settings {m:?}"
        );
        assert_eq!(m.router_type(), rt, "router_type {m:?}");
        // The pools a request of the mode can reach: the wire dtype is
        // resolved over exactly these before any worker is selected.
        assert_eq!(
            m.multimodal_wire_pools(),
            pools,
            "multimodal_wire_pools {m:?}"
        );
        // The role the image rule judges: the workers the mode sends a
        // request's pixels to.
        assert_eq!(m.pixels_go_to(), pixels, "pixels_go_to {m:?}");
    }
}
