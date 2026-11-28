//! The model listing's entry against the calls an outside caller writes.

use openai_protocol::{model_card::ModelCard, models::ModelObject};

struct Holder {
    entry: ModelObject,
}

fn a_card() -> ModelCard {
    ModelCard::new("a-model")
}

fn id_len(entry: ModelObject) -> usize {
    entry.id.len()
}

fn the_answer_of_a_function(card: ModelCard) -> ModelObject {
    card.into_model_object()
}

fn called_by_path(card: ModelCard) -> ModelObject {
    ModelCard::into_model_object(card)
}

fn pushed(list: &mut Vec<ModelObject>, card: ModelCard) {
    list.push(card.into_model_object());
}

/// `ModelCard::into_model_object` answers a `ModelObject` and nothing else,
/// so a call compiles wherever its answer goes. Sixteen shapes a caller
/// outside this crate writes; from the tenth on nothing after the call names
/// the type, so the method's own answer has to.
#[test]
fn every_call_shape_of_into_model_object_compiles_from_outside() {
    // 1: a let that names the type
    let named: ModelObject = a_card().into_model_object();
    assert_eq!(named.id, "a-model");
    // 2: a function's own answer
    assert_eq!(the_answer_of_a_function(a_card()).object, "model");
    // 3 and 4: by path and by closure in a map, collected into a typed list
    let by_path: Vec<ModelObject> = [a_card()]
        .into_iter()
        .map(ModelCard::into_model_object)
        .collect();
    let by_closure: Vec<ModelObject> = [a_card()]
        .into_iter()
        .map(|card| card.into_model_object())
        .collect();
    assert_eq!((by_path.len(), by_closure.len()), (1, 1));
    // 5: the argument of a function that names the type
    assert_eq!(id_len(a_card().into_model_object()), "a-model".len());
    // 6: pushed into a typed list
    let mut list = Vec::new();
    pushed(&mut list, a_card());
    assert_eq!(list.len(), 1);
    // 7: a plain let, then handed to a function that names the type
    let handed = a_card().into_model_object();
    assert_eq!(id_len(handed), "a-model".len());
    // 8: a field of a struct
    let holder = Holder {
        entry: a_card().into_model_object(),
    };
    assert_eq!(holder.entry.created, 0);
    // 9: called by path
    assert_eq!(called_by_path(a_card()).owned_by, "self_hosted");
    // 10: a field read off the answer
    assert_eq!(a_card().into_model_object().id, "a-model");
    // 11: a plain let, then a field read
    let entry = a_card().into_model_object();
    assert_eq!(entry.owned_by, "self_hosted");
    // 12 and 13: handed to a generic function, by value and by reference
    let value = serde_json::to_value(a_card().into_model_object()).unwrap();
    assert_eq!(value["id"], "a-model");
    let text = serde_json::to_string(&a_card().into_model_object()).unwrap();
    assert!(text.contains("\"object\":\"model\""), "{text}");
    // 14: printed
    assert!(format!("{:?}", a_card().into_model_object()).contains("a-model"));
    // 15: wrapped, then read in a closure
    let read = Some(a_card().into_model_object()).map(|entry| entry.id);
    assert_eq!(read.as_deref(), Some("a-model"));
    // 16: a method called on the answer
    assert_eq!(a_card().into_model_object().clone().id, "a-model");
}
