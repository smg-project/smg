//! Labels of a worker's spec that the worker's own code reads.

/// Which discovery provider owns the worker: `kubernetes`, `file`, `slurm` or
/// `consul`. A provider's registration carries it, and the health checker
/// reads it to tell a worker that discovery registers again from one that
/// nothing re-adds.
pub const DISCOVERY_PROVIDER_LABEL: &str = "smg.ai/discovery-provider";
