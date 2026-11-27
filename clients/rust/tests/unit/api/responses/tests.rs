use super::*;

#[test]
fn an_empty_query_sends_no_pair() {
    assert!(InputItemsQuery::default().pairs().is_empty());
}

#[test]
fn the_pairs_name_only_the_set_fields_in_the_public_order() {
    let query = InputItemsQuery {
        limit: Some(2),
        order: Some(InputItemsOrder::Asc),
        after: Some("msg_3".to_string()),
        include: vec!["a".to_string(), "b".to_string()],
    };
    assert_eq!(
        query.pairs(),
        vec![
            ("limit", "2".to_string()),
            ("order", "asc".to_string()),
            ("after", "msg_3".to_string()),
            ("include", "a".to_string()),
            ("include", "b".to_string()),
        ]
    );
    let after_only = InputItemsQuery {
        after: Some("msg_9".to_string()),
        ..InputItemsQuery::default()
    };
    assert_eq!(after_only.pairs(), vec![("after", "msg_9".to_string())]);
}
