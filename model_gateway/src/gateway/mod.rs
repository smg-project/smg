//! The gateway: the parts of request routing that every family shares.
//!
//! Today this holds worker placement. Family dispatch, which router serves a
//! model, still lives in `routers::router_manager` and moves here next.

pub(crate) mod placement;
