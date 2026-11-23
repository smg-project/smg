//! Documents that read sent prefixes, for the store's own cases.

use crate::{
    document::{
        spec,
        spec::{AppliedSpec, PolicySpec},
    },
    stages::registry::StageRegistry,
};

/// The `cache_aware` preset as it stands: its affinity stage names no
/// source, so it reads `merged`.
pub(crate) fn merged(generation: u64) -> Result<PolicySpec, spec::SpecError> {
    let mut document = PolicySpec::preset("cache_aware")?;
    document.generation = generation;
    Ok(document)
}

pub(crate) fn applied(document: &PolicySpec) -> Result<AppliedSpec, spec::SpecError> {
    AppliedSpec::validate_digested(document, document.digest()?, &StageRegistry::builtin(), &[])
}
