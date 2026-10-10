//! Token dump: an operator-controlled record of every engine call the gRPC
//! router makes, for debugging, RL data collection and CI replay.

pub mod line;
mod session;

pub use session::{CallRecorder, Session};
