//! A worker's `policy` label under Grand Central.
//!
//! A classic registry builds a policy object of its own for a model whose
//! first worker names one by its `policy` label. Under Grand Central no
//! policy object is built. The label names the policy that places the
//! model's single requests in the policy spec in force, and this module
//! puts that spec in force.
//!
//! The registration that finds a model undecided decides it, whatever the
//! worker's role, as a classic registry's first registration does. Its
//! word stands until no worker of this node serves the model: a later
//! registration's label changes nothing. A registration that carries no
//! label decides too, and so does one whose label names no policy.
//!
//! A decided label that names a policy is handed to the builder of the
//! settings' spec, with every label that stands
//! ([`RouterConfig::policy_document_with_labels`]). Its model then runs a
//! profile of its own, as a classic registry builds it an object of its
//! own. A label that names the settings' own policy is handed too: its
//! model's profile holds the settings' options.
//!
//! The spec the builder writes is checked over the table of stages the
//! start was handed and put in force on the running engine
//! ([`InProcess::apply`]). A label that
//! is refused fails alone: the standing spec stays, the registration goes
//! on, and one warning names the reason. It is not built again while the
//! model is served: the model's next first registration decides anew.
//! Beside a supplied spec no label is read at all.
//!
//! Behind each spec put in force the worker lists are published again
//! (`WorkerRegistry::publish_again`). The engine keeps its value of a list
//! in the cell beside the list, one value for each spec the list was asked
//! under, and a read walks them all. A list published again stands beside
//! an empty cell, so no list keeps a value for each label that came and
//! went.
//!
//! Ahead of each spec put in force the trees the engine is lent are put
//! under its document ([`LentTrees::under`]). A label whose spec is the
//! first to read the prefixes sent to each worker gets the trees made
//! there. A classic registry builds that model's trees at its
//! registration. A model's trees are sized by its `index` entry of the
//! spec in force.
//!
//! [`RouterConfig::policy_document_with_labels`]: crate::config::RouterConfig::policy_document_with_labels

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use parking_lot::Mutex;
use smg_grand_central::{
    document::{AppliedSpec, PolicySpec, PRESETS},
    engine::{InProcess, Phase},
};
use tracing::{debug, info, warn};

use super::lent_trees::LentTrees;
use crate::routers::common::snapshot::WorkerRegistry;

/// Makes the checked spec this gateway runs with a set of labels. The
/// labels map a model's exact id to the name of a policy's preset. The
/// spec is the one the routing settings mean with a model's own policy
/// for each label. It is numbered as the second argument says and checked
/// over the table of stages the start was handed.
///
/// The spec's document is answered with its checked form: the lent trees
/// are sized by it. A refusal is answered in its own words, and no spec.
///
/// The start makes it, since the start holds the settings and that table,
/// and hands it to the registry with the engine.
pub(crate) type Respec = Box<
    dyn Fn(&BTreeMap<String, String>, u64) -> Result<(PolicySpec, AppliedSpec), String>
        + Send
        + Sync,
>;

/// What a spec is made and put in force with: the maker, the registry of
/// the lent worker lists, and the cell of the lent trees.
struct Maker {
    respec: Respec,
    workers: Arc<WorkerRegistry>,
    trees: Arc<OnceLock<Arc<LentTrees>>>,
}

/// What the deciding registration of a model said.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Decided {
    /// The preset its label named, the settings' own policy's too. `None`
    /// for no label and for a name that is no policy's.
    preset: Option<&'static str>,
    /// Whether the builder took the label: it is handed again with every
    /// later build. False for a label the builder or the check refused.
    stands: bool,
}

/// The models a registration decided, with the maker of the specs their
/// labels put in force.
///
/// One lock holds the decisions, and it is held across a build and an
/// apply: two registrations of one model decide in turn, and the second
/// finds the model decided. The lock's order is this lock first, then the
/// engine's own lock of an apply. Nothing takes them the other way round:
/// the engine calls no code of the gateway under its lock but the worker
/// registry's reads. The worker lists are published again under this lock
/// too, behind the apply: that takes the worker registry's own guards, and
/// no caller of this module holds one.
#[derive(Default)]
pub(crate) struct Labels {
    decided: Mutex<BTreeMap<String, Decided>>,
    /// The maker of the specs, with the worker lists and the trees the
    /// engine is lent.
    maker: OnceLock<Maker>,
}

impl Labels {
    /// Hands the maker of every later spec, once: the start hands it with
    /// the engine. `workers` is the registry of the worker lists the engine
    /// is lent, and `trees` the cell of the trees it is lent. A second one
    /// is not taken.
    pub(crate) fn set_respec(
        &self,
        respec: Respec,
        workers: Arc<WorkerRegistry>,
        trees: Arc<OnceLock<Arc<LentTrees>>>,
    ) {
        let maker = Maker {
            respec,
            workers,
            trees,
        };
        if self.maker.set(maker).is_err() {
            warn!("a maker of policy specs is already set: the second one is not taken");
        }
    }

    /// A worker of `model` registered, with `hint` as its `policy` label.
    /// Where the model is undecided this registration decides it, and a
    /// label that names a policy is built into the spec and put in force
    /// on `engine`. The settings' own policy is one: its label gives the
    /// model a profile of its own too.
    ///
    /// Answers whether a spec was put in force: the caller then reads again
    /// what it keeps of the spec.
    pub(crate) fn registered(&self, engine: &InProcess, model: &str, hint: Option<&str>) -> bool {
        let mut decided = self.decided.lock();
        if decided.contains_key(model) {
            return false;
        }
        let preset = hint.and_then(policy_named);
        let (stands, applied) = match preset {
            Some(preset) => self.put_in_force(engine, &decided, Some((model, preset))),
            None => (false, false),
        };
        decided.insert(model.to_owned(), Decided { preset, stands });
        applied
    }

    /// No worker of this node serves `model` any more: its decision goes
    /// with its last worker, and where its label stood in the spec, the
    /// spec is built without it and put in force on `engine`.
    ///
    /// Answers whether a spec was put in force.
    pub(crate) fn unserved(&self, engine: &InProcess, model: &str) -> bool {
        let mut decided = self.decided.lock();
        match decided.remove(model) {
            Some(was) if was.stands => self.put_in_force(engine, &decided, None).1,
            _ => false,
        }
    }

    /// Whether `hint`, a worker's `policy` label, names what places
    /// `model`'s single requests under the spec in force on `engine`.
    ///
    /// Under the spec the settings mean, a model whose label stands is
    /// placed by a profile of its own. The crate names that profile for
    /// the label's policy and the model ([`label_profile`]). Every other
    /// model is placed by the settings' own policy, `own`.
    ///
    /// One label stands with no profile of its own: `own` for a model
    /// whose id no binding names alone. The main profile places that
    /// model, under the policy its label names.
    ///
    /// So a label names what places its model where it names the policy
    /// that stands for the model, or `own` for a model none stands for.
    /// Beside a `supplied` spec the profile's name alone is read: the
    /// label names what places the model where the profile that places it
    /// bears the policy's name. False for a name that is no policy's, and
    /// where the spec binds no single request of the model.
    ///
    /// [`label_profile`]: smg_grand_central::document::RoutingSettings::label_profile
    pub(crate) fn places(
        &self,
        engine: &InProcess,
        model: &str,
        hint: &str,
        own: &str,
        supplied: bool,
    ) -> bool {
        let Some(named) = policy_named(hint) else {
            return false;
        };
        // Held across the read of the spec: no label's spec is put in force
        // between the two reads.
        let decided = self.decided.lock();
        let Some(placing) = placing_profile(engine, model) else {
            return false;
        };
        if supplied {
            return placing == named;
        }
        match decided.get(model) {
            Some(Decided {
                preset: Some(preset),
                stands: true,
            }) => {
                let own_profile =
                    smg_grand_central::document::RoutingSettings::label_profile(preset, model);
                named == *preset && (*preset == own || placing == own_profile)
            }
            _ => named == own,
        }
    }

    /// Builds the spec of the labels that stand in `decided`, with `added`
    /// (a model and the preset its label named) where a registration brings
    /// one, and puts it in force on `engine`. The spec takes the number
    /// behind the standing one.
    ///
    /// Answers two words: whether the builder and the check took the labels,
    /// and whether a spec was put in force. A spec that says what the
    /// standing one says is not put in force: a label that binds nothing
    /// there costs no stage its state. Behind a spec put in force the worker
    /// lists are published again.
    ///
    /// Ahead of the apply the lent trees are put under the spec's document.
    /// The table a request builds behind the apply then asks trees that
    /// stand, sized by that spec. A refused apply puts them back under the
    /// document they followed.
    ///
    /// A table built between the two asks trees sized by that document
    /// ahead of its force, and behind a refused apply it keeps them until
    /// its next build. Only a model whose `index` entry differs between
    /// the two documents sees it.
    fn put_in_force(
        &self,
        engine: &InProcess,
        decided: &BTreeMap<String, Decided>,
        added: Option<(&str, &'static str)>,
    ) -> (bool, bool) {
        let Some(Maker {
            respec,
            workers,
            trees,
        }) = self.maker.get()
        else {
            debug!("No maker of policy specs is set: a worker's policy label is not applied");
            return (false, false);
        };
        let mut labels: BTreeMap<String, String> = decided
            .iter()
            .filter(|(_, decided)| decided.stands)
            .filter_map(|(model, decided)| Some((model.clone(), decided.preset?.to_owned())))
            .collect();
        if let Some((model, preset)) = added {
            labels.insert(model.to_owned(), preset.to_owned());
        }
        let standing = engine.spec();
        let next = standing.generation().saturating_add(1);
        let (document, spec) = match respec(&labels, next) {
            Ok(made) => made,
            Err(refused) => {
                match added {
                    Some((model, preset)) => {
                        let cut = cut_mark(model);
                        warn!(
                            "Worker label policy = {preset} of model {model:.64}{cut} is not \
                             applied, and the policy spec in force stays: {refused}"
                        );
                    }
                    None => warn!(
                        "The policy spec is not built again behind a model's last worker, and \
                         the spec in force stays: {refused}"
                    ),
                }
                return (false, false);
            }
        };
        if spec.digest() == standing.digest() {
            return (true, false);
        }
        let reads = spec.needs().sent_prefixes;
        let followed = LentTrees::under(trees, document, reads, spec.block_size(), workers);
        match engine.apply(spec) {
            Ok(()) => {
                // Each list that stands holds a value of the spec before in
                // its cell: published again, it stands beside an empty one.
                workers.publish_again();
                info!(
                    generation = next,
                    labels = labels.len(),
                    "A policy spec is in force with the policies the workers' labels name"
                );
                (true, true)
            }
            Err(stale) => {
                if let (Some(document), Some(owner)) = (followed, trees.get()) {
                    owner.follow(document);
                }
                warn!("A policy spec built for a worker's policy label is not applied: {stale}");
                (false, false)
            }
        }
    }
}

/// The policy `hint` names, as the name of its preset. The name is read
/// in lower case, equal to a preset's name or to it without its
/// underscores, as a classic registry reads a label's spelling. `None` for
/// a name that is no policy's.
fn policy_named(hint: &str) -> Option<&'static str> {
    let hint = hint.to_lowercase();
    PRESETS
        .into_iter()
        .find(|name| hint == *name || hint == name.replace('_', ""))
}

/// The name of the profile that places `model`'s single requests under the
/// spec in force on `engine`. `None` where that spec binds no single
/// request of the model.
pub(crate) fn placing_profile(engine: &InProcess, model: &str) -> Option<String> {
    let spec = engine.spec();
    let (_, profile) = spec.engine().profile_for(model, Phase::Single)?;
    Some(profile.name.clone())
}

/// The mark of a cut behind a model's id in a log line that writes its
/// first 64 characters (`{model:.64}`): a worker names its model, so the id
/// is free text of unknown length.
pub(crate) fn cut_mark(model: &str) -> &'static str {
    if model.chars().nth(64).is_some() {
        "..."
    } else {
        ""
    }
}

#[cfg(test)]
mod tests;
