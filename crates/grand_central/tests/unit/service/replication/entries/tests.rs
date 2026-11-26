use super::*;

/// The store's ingress traits, each with its file's text.
const TRAITS: [(&str, &str); 3] = [
    (
        "WorkersStore",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/store/workers.rs")),
    ),
    (
        "HealthSink",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/store/health.rs")),
    ),
    (
        "LeasesStore",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/store/leases.rs")),
    ),
];

/// The entries of `name` as its file declares them, in order.
fn declared(name: &str, text: &str) -> Vec<String> {
    let head = format!("pub trait {name}");
    text.lines()
        .skip_while(|line| !line.starts_with(&head))
        .skip(1)
        .take_while(|line| !line.starts_with('}'))
        .filter_map(|line| line.strip_prefix("    fn "))
        .filter_map(|rest| rest.split(['(', '<']).next())
        .map(str::to_owned)
        .collect()
}

/// The store type, with its own file's text.
const OWN: (&str, &str) = (
    "MemoryStore",
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/store/memory.rs")),
);

/// The public entries of the type's own block in `text`, in order.
fn own(name: &str, text: &str) -> Vec<String> {
    let head = format!("impl {name} {{");
    text.lines()
        .skip_while(|line| *line != head)
        .skip(1)
        .take_while(|line| !line.starts_with('}'))
        .filter_map(|line| line.strip_prefix("    pub fn "))
        .filter_map(|rest| rest.split(['(', '<']).next())
        .map(str::to_owned)
        .collect()
}

/// The entries the list holds for `name`, in the list's order.
fn listed(name: &str) -> Vec<&'static str> {
    ENTRIES
        .iter()
        .filter(|(of, _, _)| *of == name)
        .map(|(_, entry, _)| *entry)
        .collect()
}

#[test]
fn every_entry_of_the_store_stands_on_the_list_once_in_its_order() {
    for (name, text) in TRAITS {
        let declared = declared(name, text);
        assert!(!declared.is_empty(), "{name}: no entry read from its file");
        assert_eq!(
            listed(name),
            declared,
            "{name}: the list and the trait differ"
        );
    }
    let (store, text) = OWN;
    let declared = own(store, text);
    assert!(declared.len() > 1, "{store}: no entry read from its file");
    assert_eq!(
        listed(store),
        declared,
        "{store}: the list and the type differ"
    );
    assert!(
        ENTRIES
            .iter()
            .all(|(of, _, _)| *of == store || TRAITS.iter().any(|(name, _)| name == of)),
        "the list names a trait or a type the store does not have"
    );
}

#[test]
fn a_fate_that_names_a_source_names_one() {
    for (of, entry, fate) in ENTRIES {
        if let EntryReplication::Rebuilt(source) | EntryReplication::NotYet(source) = fate {
            assert!(!source.is_empty(), "{of}::{entry}");
        }
    }
}
