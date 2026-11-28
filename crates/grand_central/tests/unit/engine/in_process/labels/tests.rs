use std::{cell::Cell, ptr};

use super::{super::InProcess, *};
use crate::{
    document::spec::{BindingSpec, Match, PolicySpec},
    engine::{decision::Trace, Phase},
    names::tests::ASKED,
};

/// The `round_robin` preset's document with a copy of its profile under
/// each name of `bound`, bound to the model beside it. The last of them
/// stands first, and the preset's own binding of every model last.
fn document(bound: &[(&str, &str)]) -> PolicySpec {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    for (name, model) in bound {
        let profile = document.profiles["round_robin"].clone();
        document.profiles.insert((*name).to_owned(), profile);
        let binding = BindingSpec {
            matcher: Match {
                model: (*model).to_owned(),
                ..Match::default()
            },
            profile: Some((*name).to_owned()),
            ..BindingSpec::default()
        };
        document.bindings.insert(0, binding);
    }
    document
}

/// The label of the profile `name` under `spec`.
fn label_of(spec: &AppliedSpec, name: &str) -> &'static str {
    let profiles = spec.engine().profiles();
    let at = profiles.iter().position(|profile| profile.name == name);
    of(spec)[at.expect("the document holds the profile")]
}

/// A profile's label is read by its position in the document, the
/// number an answer names its profile by: each position here reads its
/// profile's name as the document writes it, one string a name in every
/// engine of the process. A position the document holds no profile at
/// reads the empty label, and so does the number of a decision made
/// ahead of every profile.
#[test]
fn a_profiles_label_is_read_by_its_position_in_the_document() {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    let spare = document.profiles["round_robin"].clone();
    document.profiles.insert("a_spare".to_owned(), spare);
    let engine_of = |document: &PolicySpec| {
        InProcess::new(AppliedSpec::validate(document).expect("the document validates"))
    };
    let (engine, another) = (engine_of(&document), engine_of(&document));
    let labels = (engine.loaded().label_at(0), engine.loaded().label_at(1));
    assert_eq!(labels, ("a_spare", "round_robin"));
    let same = ptr::eq(another.loaded().label_at(1), engine.loaded().label_at(1));
    assert!(same, "one string a name");
    let none = (
        engine.loaded().label_at(2),
        engine.loaded().label_at(Trace::NO_PROFILE),
    );
    assert_eq!(none, ("", ""));
}

/// A labelled model's profile bears its policy's name. The profile is
/// named for a preset and the model's id, and the document places the
/// model's single requests by it. Two labelled models of one policy bear
/// one label. It is the preset's own string and no kept name: of the
/// three profiles the keeper is asked for the one that bears its name.
#[test]
fn a_labelled_models_profile_bears_its_policy() {
    let own = |model: &str| RoutingSettings::label_profile("least_load", model);
    let (first, second) = (own("m-a"), own("m-b"));
    let document = document(&[(&first, "m-a"), (&second, "m-b")]);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let asked = ASKED.with(Cell::get);
    let labels = of(&spec);
    assert_eq!(ASKED.with(Cell::get) - asked, 1, "one name is asked for");
    assert_eq!(*labels, ["least_load", "least_load", "round_robin"]);
    assert!(ptr::eq(labels[0], labels[1]), "one label a policy");
    assert!(!ptr::eq(labels[0], intern("least_load")), "no kept name");
}

/// A profile named as a labelled model's bears its name where the
/// document binds it another way. Here it is bound to another model's
/// id, and the preset's own profile places the model its name bears.
#[test]
fn a_profile_named_for_a_model_it_does_not_place_bears_its_name() {
    let named = RoutingSettings::label_profile("least_load", "m-a");
    let document = document(&[(&named, "another")]);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let placing = spec.engine().profile_for("m-a", Phase::Single);
    let placing = placing.map(|(_, profile)| profile.name.as_str());
    assert_eq!(placing, Some("round_robin"));
    assert_eq!(label_of(&spec, &named), named);
}

/// A name whose text ahead of the colon is no preset's is its profile's
/// label, though the document places the model behind the colon by it.
#[test]
fn a_name_that_begins_with_no_preset_bears_its_name() {
    let document = document(&[("mine:m-a", "m-a")]);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let placing = spec.engine().profile_for("m-a", Phase::Single);
    let placing = placing.map(|(_, profile)| profile.name.as_str());
    assert_eq!(placing, Some("mine:m-a"));
    assert_eq!(label_of(&spec, "mine:m-a"), "mine:m-a");
}
