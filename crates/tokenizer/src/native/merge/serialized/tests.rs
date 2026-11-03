use serde::{ser::SerializeStruct, Serialize, Serializer};

use super::merges_of;

/// A field value that refuses to be serialized: the reader must never reach
/// it.
struct Untouched;

impl Serialize for Untouched {
    fn serialize<S: Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom(
            "a field other than merges was serialized",
        ))
    }
}

/// A model-shaped struct: a vocabulary and other fields around the merges,
/// in the model's own field order.
struct ModelLike;

impl Serialize for ModelLike {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut model = serializer.serialize_struct("BPE", 4)?;
        model.serialize_field("type", &Untouched)?;
        model.serialize_field("vocab", &Untouched)?;
        model.serialize_field("merges", &[("a", "b"), ("ab", "c")])?;
        model.serialize_field("ignore_merges", &Untouched)?;
        model.end()
    }
}

#[test]
fn a_field_that_is_not_merges_is_never_serialized() {
    // Every other field refuses to serialize, so this passes only when the
    // reader takes the merges field without touching the others' values.
    assert_eq!(
        merges_of(&ModelLike).expect("the merges alone"),
        r#"[["a","b"],["ab","c"]]"#
    );
}

#[test]
fn a_value_that_is_not_a_struct_is_refused() {
    assert_eq!(
        merges_of(&vec![("a", "b")]).expect_err("not a model"),
        "the model is not a struct (serialize_seq)"
    );
    assert_eq!(
        merges_of(&"merges").expect_err("not a model"),
        "the model is not a struct (serialize_str)"
    );
}

#[test]
fn a_struct_without_a_merges_field_is_refused() {
    struct Bare;
    impl Serialize for Bare {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_struct("Bare", 0)?.end()
        }
    }
    assert_eq!(
        merges_of(&Bare).expect_err("no merges"),
        "the model's serialized form has no merges field"
    );
}
