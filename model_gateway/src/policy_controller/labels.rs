//! A worker's `policy` label under the policy controller.
//!
//! On the classic line a model's first worker names the model's policy by
//! its `policy` label, and the classic registry keeps that policy until the
//! model's last worker leaves. Under the controller that registry stays the
//! one holder of which policy a model has. A registration tells it of a
//! worker's model and label ([`registered`]), and tells it again behind the
//! write of the worker's row ([`confirmed`]). A removal, a replacement and
//! a registration that did not finish tell it of the models no worker
//! serves any more ([`left`]).
//!
//! The document in force is the settings' document with one binding more
//! for each model whose policy is not the main one
//! ([`RouterConfig::policy_document_labelled`]). Where the registry's word
//! on a model moved, it is written again and applied at the operators'
//! door: at a model's first labelled worker, ahead of the write of that
//! worker's row, and at the model's last leave. The document is written
//! with no generation of its own, so the door numbers it, and the standing
//! document again is no change. Each telling and the apply behind it share
//! one turn of the door, so a registration that finds its model told
//! writes its row behind that telling's document.
//!
//! The first telling stands ahead of the row's write, since a row states
//! what the generation in force reads of it. So a model's last leave can
//! cross a registration of the same model, and take the policy out before
//! the new worker stands in the registry. The second telling stands where
//! the classic line's policy step stands, behind the registration, and
//! tells such a model anew. The row written in between states what the
//! generation without the binding reads.
//!
//! Each apply is a new generation, and a generation's stages start empty:
//! every model's stages hear the members again, and a profile's count of
//! sends alone carries over under its name.
//!
//! A later worker of a model moves nothing, whatever its label. A supplied
//! document is its author's whole word: no label binds a model beside one.
//! The registry is told there too, so the refusal of an explicit target
//! reads the model's own policy. A gateway that follows a controller
//! registers no worker of its own, so nothing here runs there.
//!
//! A request that names no model is asked under the one model its pool
//! serves, so that model's binding places it. The refusal of its explicit
//! target reads the policy of the no-model id, as the classic selection
//! does.
//!
//! [`RouterConfig::policy_document_labelled`]: crate::config::RouterConfig::policy_document_labelled

use smg_policy_controller::operator::Applied;
use tracing::{info, warn};

use crate::{app_context::AppContext, policies::PolicyRegistry};

/// Tells the classic registry of a worker of each of `models` and its
/// `policy` label, as the classic line's registration does. Where the label
/// gives a model a policy of its own, the document in force is applied
/// again with the model's binding, once for the call. A caller runs this
/// ahead of the write of the worker's row: a row states what the
/// generation in force reads of it.
///
/// An apply validates a document, so an async caller runs this on the
/// blocking pool.
pub(crate) fn registered(context: &AppContext, models: &[String], label: Option<&str>) {
    told(context, || added(context, models, label, true));
}

/// The second telling of a registration, behind the write of its workers'
/// rows, where the classic line's policy step stands. A model of `models`
/// whose policy a last leave took out since the first telling is told
/// anew. Each of `replaced`, the models the workers this registration
/// replaced served, is told as at a removal where no worker serves it any
/// more. Where the registry's word stands nothing is applied.
pub(crate) fn confirmed(
    context: &AppContext,
    models: &[String],
    label: Option<&str>,
    replaced: &[String],
) {
    told(context, || {
        let anew = added(context, models, label, false);
        // Both tellings run: each moves the registry's word.
        let left = gone(context, replaced);
        anew.or(left)
    });
}

/// Tells the classic registry of each of `models` that no worker of this
/// node serves any more, as the classic line does at a removal. Where a
/// model had a policy of its own, the document in force is applied again
/// without the model's binding, once for the call.
pub(crate) fn left(context: &AppContext, models: &[String]) {
    told(context, || gone(context, models));
}

/// One telling of a registration's `models`: a model the registry holds no
/// policy of takes the one `label` names. Answers a model whose policy is
/// then its own. A `first` telling says in the log that a later worker's
/// label is not read.
fn added<'a>(
    context: &AppContext,
    models: &'a [String],
    label: Option<&str>,
    first: bool,
) -> Option<&'a String> {
    let registry = &context.policy_registry;
    let main = registry.get_default_policy().name();
    let mut moved = None;
    for model in models {
        if let Some(standing) = registry.get_policy(model) {
            // A later worker moves nothing, as on the classic line.
            let other = label.filter(|label| {
                first && !PolicyRegistry::hint_matches_config(label, standing.name())
            });
            if let Some(label) = other {
                info!(
                    "The policy label '{label}' of a later worker of model {model} is not read: the model runs {} since its first worker",
                    standing.name()
                );
            }
        } else if registry.on_worker_added(model, label).name() != main {
            moved = moved.or(Some(model));
        }
    }
    moved
}

/// One telling of the models workers left: each of `models` that no worker
/// of this node serves loses its policy. The worker registry is asked
/// inside the turn, as the classic line asks behind its own write: a model
/// another worker serves keeps its policy. Answers a model whose policy
/// was its own.
fn gone<'a>(context: &AppContext, models: &'a [String]) -> Option<&'a String> {
    let registry = &context.policy_registry;
    let main = registry.get_default_policy().name();
    let mut moved = None;
    for model in models {
        if context.worker_registry.serves_model_locally(model) {
            continue;
        }
        if registry
            .get_policy(model)
            .is_some_and(|policy| policy.name() != main)
        {
            moved = moved.or(Some(model));
        }
        registry.on_model_unserved(model);
    }
    moved
}

/// Runs `tell`, which tells the registry and answers a model whose word
/// moved, and applies the document written again where one did. Both run
/// inside one turn of the operators' door: a telling that finds its model
/// told already ends behind the apply of the telling before it, and a
/// leave's question of the worker registry is never answered across a
/// registration's second telling. Under the classic policies the registry
/// is told and nothing is applied.
fn told<'a>(context: &AppContext, tell: impl FnOnce() -> Option<&'a String>) {
    let Some(operators) = context.policy_operators.as_ref() else {
        tell();
        return;
    };
    let mut moved = None;
    let applied = operators.apply_made(|| {
        let model = tell()?;
        moved = Some(model);
        written(context, model)
    });
    let Some(model) = moved else {
        return;
    };
    match applied {
        Some(Ok(Applied::Generation(generation))) => info!(
            "The policy document is applied again for model {model}, as generation {generation}"
        ),
        Some(Ok(Applied::Unchanged(_) | Applied::Waits(_))) | None => {}
        Some(Err(refusal)) => warn!(
            "The policy document written for model {model} was refused, and the standing one stays: {refusal}"
        ),
    }
}

/// The document in force as YAML, written again from the settings and the
/// registry's word on each model. `model` is one whose word moved, for the
/// log. `None` beside a supplied document, and where the settings'
/// document does not write.
fn written(context: &AppContext, model: &str) -> Option<String> {
    let config = &context.router_config;
    if config.policy_spec.is_some() {
        info!(
            "The policy label of model {model} binds nothing: the policy document in force was supplied"
        );
        return None;
    }
    let (mut document, left) = config
        .policy_document_labelled(&context.policy_registry.named_models())
        .inspect_err(|error| {
            warn!("The policy document was not written again for model {model}: {error}");
        })
        .ok()?;
    for (model, why) in left {
        warn!(
            "The policy label of model {model} binds nothing, and the model is decided as every other: {why}"
        );
    }
    // The door numbers the document: a number of its own is a claim.
    document.generation = 0;
    serde_yaml::to_string(&document)
        .inspect_err(|error| {
            warn!("The policy document did not write as YAML for model {model}: {error}");
        })
        .ok()
}
