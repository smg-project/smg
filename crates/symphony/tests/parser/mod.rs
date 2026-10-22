//! Parser-level integration tests: the contract every registry table keeps over its outputs at
//! every chunking, every table's robustness under its own scrambled markers, the structural tag of
//! a forced or required call, and the crate's public surface.

mod contract;
mod forced_tag;
mod robustness;
mod surface;
