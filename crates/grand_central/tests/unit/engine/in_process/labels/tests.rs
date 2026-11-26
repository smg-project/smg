use std::ptr;

use super::{super::InProcess, *};
use crate::{document::spec::PolicySpec, engine::decision::Trace};

/// A profile's label is read by its position in the document, the
/// number an answer names its profile by: each position reads its
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
