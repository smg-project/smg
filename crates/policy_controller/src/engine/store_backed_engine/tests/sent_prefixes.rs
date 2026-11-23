//! Documents that read sent prefixes, for the store's own cases.

use crate::{
    document::{
        spec,
        spec::{AffinitySource, AppliedSpec, PolicySpec, ScoreSpec, ScoreStage},
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

pub(crate) fn use_sent_prefixes(document: &mut PolicySpec, source: AffinitySource) {
    for stage in document
        .profiles
        .values_mut()
        .flat_map(|profile| &mut profile.score)
    {
        if let ScoreSpec::Params(ScoreStage::PrefixAffinity(params)) = stage {
            params.source = Some(source);
        }
    }
}

pub(crate) fn applied(document: &PolicySpec) -> Result<AppliedSpec, spec::SpecError> {
    AppliedSpec::validate_digested(document, document.digest()?, &StageRegistry::builtin(), &[])
}
