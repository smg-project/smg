//! The standalone policy controller (design 2: the `remote` mode of the
//! policy engine): the crate's command line over the stock build's stages.
//! A host with stages of its own writes this `main` over its own catalog.

use std::process::ExitCode;

use smg_policy_controller::{controller, Catalog};

fn main() -> ExitCode {
    controller::run(Catalog::builtin())
}
