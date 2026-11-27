use super::*;

#[test]
fn the_capacity_figures_join_the_launchers_entries_under_the_python_servicers_keys() {
    fn number_of(scheduler_info: &Struct, key: &str) -> f64 {
        match scheduler_info.fields[key].kind {
            Some(Kind::NumberValue(value)) => value,
            ref other => panic!("{key}: {other:?}"),
        }
    }
    let mut scheduler_info = struct_from_json(r#"{"is_generation":true}"#);
    insert_capacity(&mut scheduler_info, 4096, 16, 64);
    assert_eq!(number_of(&scheduler_info, "max_total_num_tokens"), 4096.0);
    assert_eq!(number_of(&scheduler_info, "page_size"), 16.0);
    assert_eq!(number_of(&scheduler_info, "max_running_requests"), 64.0);
    assert!(matches!(
        scheduler_info.fields["is_generation"].kind,
        Some(Kind::BoolValue(true))
    ));
    // A page size past i32 saturates, as every figure on this wire does.
    insert_capacity(&mut scheduler_info, 0, u64::MAX, 0);
    assert_eq!(number_of(&scheduler_info, "page_size"), f64::from(i32::MAX));
}
