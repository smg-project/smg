use super::*;

#[test]
fn every_alias_spells_its_table_and_is_no_tables_own_name() {
    let mut seen = std::collections::HashSet::new();
    for (alias, table) in ALIASES {
        assert!(seen.insert(alias), "{alias} is listed twice");
        assert!(
            Table::ALL.iter().all(|own| own.name() != alias),
            "{alias} is a table's own name"
        );
        assert_eq!(Table::named(alias), None, "{alias} is a table's own name");
        // Every table a name can select stands in `ALL`: `Table::shared` keeps its tables
        // at their positions there.
        assert!(Table::ALL.contains(&table), "{alias}'s table is not in ALL");
        assert_eq!(answering_to(alias), Some(table), "{alias}");
        assert_eq!(
            answering_to(&alias.to_ascii_uppercase()),
            Some(table),
            "{alias}"
        );
    }
    assert_eq!(answering_to("llama"), Some(Table::Json));
    assert_eq!(answering_to("pythonic"), None);
    assert_eq!(answering_to("passthrough"), None);
    // A launcher's name for a family whose table has no older parser answers for the table.
    assert_eq!(answering_to("muse"), Some(Table::MuseSpark));
    assert_eq!(Table::named("muse"), None);
}

#[test]
fn every_unanswered_name_answers_to_no_table_and_says_why() {
    let mut seen = std::collections::HashSet::new();
    for (name, reason) in UNANSWERED {
        assert!(seen.insert(name), "{name} is listed twice");
        assert_eq!(
            name,
            name.trim().to_ascii_lowercase(),
            "{name} is not spelled as the legacy registries spell their names"
        );
        assert_eq!(Table::named(name), None, "{name} is a table's own name");
        assert_eq!(
            answering_to(name),
            None,
            "{name} answers to a table: remove its row"
        );
        assert!(!reason.trim().is_empty(), "{name} has no reason");
    }
}
