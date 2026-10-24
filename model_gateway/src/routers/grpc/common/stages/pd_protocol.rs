//! Per-runtime PD (prefill/decode) disaggregation protocol descriptors.
//!
//! Every way one runtime's PD implementation differs from another's is a row
//! in [`PdProtocol::for_runtime`]: dispatch shape, rendezvous carrier, and DP
//! placement carrier. Stage code branches on these semantic axes, never on
//! `RuntimeType` identity, so teaching the router a new runtime's PD protocol
//! means deciding the three axes in one place — the exhaustive match below
//! turns a skipped decision into a compile error instead of an audit of
//! scattered `== RuntimeType::X` checks.

use crate::worker::RuntimeType;

/// Shape of the disaggregated dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PdDispatch {
    /// Prefill runs to completion first, then decode: the router relays the
    /// KV handoff state between the sides (vLLM `kv_transfer_params`).
    Sequential,
    /// Prefill and decode dispatch together and rendezvous on bootstrap info
    /// carried in the request itself. The decode side's first output follows
    /// the prefill side's KV handoff, and may come an engine step before the
    /// prefill side's own response: the prefill silence rule takes it as the
    /// prefill worker's sign of life (`liveness::prefill_side_reprieve`). A
    /// runtime whose decode could answer without the handoff does not fit
    /// this shape as it stands.
    Parallel,
}

/// How the prefill/decode rendezvous travels to the engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PdRendezvous {
    /// SGLang-style `DisaggregatedParams` bootstrap metadata (host, port,
    /// room), injected by `helpers::maybe_inject_pd_metadata`.
    SglangBootstrap,
    /// KV bootstrap host/port/room fields on the generate request
    /// (TokenSpeed), injected by `helpers::maybe_inject_pd_rendezvous`.
    KvBootstrapRoom,
    /// No in-request rendezvous: the sequential dispatch path relays the
    /// handoff state between the sides instead.
    None,
}

/// How DP placement constrains rendezvous construction; requests also carry rank pins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DpPlacement {
    /// The engine honors the per-request `data_parallel_rank` pin field, set
    /// on each side from its selected dp-aware virtual worker.
    PinField,
    /// The engine's prefill side places a disaggregated request on the rank
    /// of its room's residue (`bootstrap_room % dp_size`) and its decode side
    /// locates the KV source there, so the room is minted congruent to the
    /// selected prefill rank and the residue is the placement carrier on
    /// that side. Each request also carries its selected rank pin, which the
    /// decode side honors where its placement is free.
    RoomResidue,
}

/// PD disaggregation protocol for one runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PdProtocol {
    pub dispatch: PdDispatch,
    pub rendezvous: PdRendezvous,
    pub dp_placement: DpPlacement,
}

impl PdProtocol {
    /// The PD protocol table. `None` means the runtime does not support PD
    /// disaggregated mode.
    pub(crate) fn for_runtime(runtime: RuntimeType) -> Option<Self> {
        match runtime {
            // SGLang's prefill side follows the room by default
            // (`follow_bootstrap_room`: its KV sender fails a request placed
            // on another rank than the room's residue) and its decode side
            // looks for the KV on that rank; the decode side's own placement
            // is free and honors the pin.
            RuntimeType::Sglang => Some(Self {
                dispatch: PdDispatch::Parallel,
                rendezvous: PdRendezvous::SglangBootstrap,
                dp_placement: DpPlacement::RoomResidue,
            }),
            RuntimeType::Vllm => Some(Self {
                dispatch: PdDispatch::Sequential,
                rendezvous: PdRendezvous::None,
                dp_placement: DpPlacement::PinField,
            }),
            RuntimeType::TokenSpeed => Some(Self {
                dispatch: PdDispatch::Parallel,
                rendezvous: PdRendezvous::KvBootstrapRoom,
                dp_placement: DpPlacement::RoomResidue,
            }),
            RuntimeType::Trtllm
            | RuntimeType::Mlx
            | RuntimeType::Generic
            | RuntimeType::External
            | RuntimeType::Unspecified => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pd_protocol_table_rows() {
        let sglang = PdProtocol::for_runtime(RuntimeType::Sglang).unwrap();
        assert_eq!(sglang.dispatch, PdDispatch::Parallel);
        assert_eq!(sglang.rendezvous, PdRendezvous::SglangBootstrap);
        assert_eq!(sglang.dp_placement, DpPlacement::RoomResidue);

        let vllm = PdProtocol::for_runtime(RuntimeType::Vllm).unwrap();
        assert_eq!(vllm.dispatch, PdDispatch::Sequential);
        assert_eq!(vllm.rendezvous, PdRendezvous::None);
        assert_eq!(vllm.dp_placement, DpPlacement::PinField);

        let tokenspeed = PdProtocol::for_runtime(RuntimeType::TokenSpeed).unwrap();
        assert_eq!(tokenspeed.dispatch, PdDispatch::Parallel);
        assert_eq!(tokenspeed.rendezvous, PdRendezvous::KvBootstrapRoom);
        assert_eq!(tokenspeed.dp_placement, DpPlacement::RoomResidue);

        for runtime in [
            RuntimeType::Trtllm,
            RuntimeType::Mlx,
            RuntimeType::Generic,
            RuntimeType::External,
            RuntimeType::Unspecified,
        ] {
            assert!(
                PdProtocol::for_runtime(runtime).is_none(),
                "{runtime} must not claim PD support"
            );
        }
    }
}
