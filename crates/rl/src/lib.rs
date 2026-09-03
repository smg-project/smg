//! RL control plane for SMG: worker discovery, verbatim passthrough of
//! engine-native RL routes, and label-selected fan-out. Compiled into the
//! gateway but inert unless `--enable-rl`.

pub mod capability;
pub mod config;
pub mod error;
pub mod path;
pub mod selector;
pub mod view;

pub use config::RlConfig;
pub use error::RlError;
pub use view::{RlWorkerInfo, RlWorkerView};
