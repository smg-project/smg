//! The decision engine: the stages run over one request's candidate workers,
//! what a decision reads (the view) and what it returns, and the engine that
//! decides over the policy store.

pub mod decider;
pub mod decision;
pub mod demands;
pub mod door;
pub mod framework;
pub(crate) mod terms;
pub mod view;
