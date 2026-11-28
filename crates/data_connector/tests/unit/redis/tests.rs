use serde_json::json;

use super::*;

#[test]
fn output_item_ids_reads_the_output_array_only() {
    let raw = json!({
        "id": "resp_1",
        "output": [
            {"type": "reasoning", "id": "rs_1"},
            {"type": "message", "id": "msg_1"},
            {"type": "message"}
        ]
    });
    assert_eq!(output_item_ids(&raw), vec!["rs_1", "msg_1"]);
    assert!(output_item_ids(&json!({"id": "resp_2"})).is_empty());
    assert!(output_item_ids(&Value::Null).is_empty());
}

#[test]
fn link_scores_a_microsecond_apart_keep_their_order() {
    let base = DateTime::from_timestamp_micros(1_700_000_000_123_456).unwrap();
    let later = base + chrono::TimeDelta::microseconds(1);
    assert!(link_score(base) < link_score(later));
    assert!(link_score(later) < link_score(later + chrono::TimeDelta::microseconds(1)));
}

#[test]
fn link_score_of_a_whole_millisecond_is_its_millisecond_count() {
    // The score links were stored with before, so stored and new links sort together.
    let whole = DateTime::from_timestamp_millis(1_700_000_000_123).unwrap();
    assert_eq!(
        link_score(whole).to_bits(),
        1_700_000_000_123.0_f64.to_bits()
    );
}

#[test]
fn create_item_script_answers_an_existing_hash_and_writes_only_a_missing_one() {
    let script = CREATE_ITEM_SCRIPT;
    let answered = script
        .find(
            "if redis.call('EXISTS', KEYS[1]) == 1 then return redis.call('HGETALL', KEYS[1]) end",
        )
        .expect("an existing hash is answered as it is");
    for write in [
        "for i = 2, #ARGV - 1, 2 do redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1]) end",
        "if ARGV[1] ~= '' then redis.call('EXPIRE', KEYS[1], ARGV[1]) end",
    ] {
        let at = script
            .find(write)
            .unwrap_or_else(|| panic!("the script writes: {write}"));
        assert!(
            at > answered,
            "written only past the existence check: {write}"
        );
    }
    assert!(script.trim_end().ends_with("return {}"));
}

#[test]
fn store_if_open_script_writes_an_existing_hash_without_a_status_and_refuses_a_missing_one() {
    let script = store_if_open_script();
    // The object field is read only when the schema has one.
    assert!(script.contains("if ARGV[1] ~= '' then raw = redis.call('HGET', KEYS[1], ARGV[1]) end"));
    // Without it, or without the field in the hash, the hash itself
    // decides between missing and open.
    assert!(script
        .contains("if not raw and redis.call('EXISTS', KEYS[1]) == 0 then return 'missing' end"));
    assert!(script.contains(
        "if status == 'completed' or status == 'incomplete' or status == 'failed' or \
             status == 'cancelled' then return 'closed' end"
    ));
    assert!(script.trim_end().ends_with("return 'stored'"));
}

#[test]
fn store_if_open_script_writes_the_output_item_keys_and_the_expiry_with_the_fields() {
    let script = store_if_open_script();
    let check = script.find("return 'closed'").expect("the status check");
    // Everything a stored record carries is written in the one run, after
    // the check: the fields, one key per output item under the retention
    // (a plain SET without one), the hash's expiry.
    for write in [
        "for i = 4, #ARGV - 1, 2 do redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1]) end",
        "for k = 2, #KEYS do",
        "if ARGV[3] == '' then redis.call('SET', KEYS[k], ARGV[2]) \
             else redis.call('SETEX', KEYS[k], ARGV[3], ARGV[2]) end",
        "if ARGV[3] ~= '' then redis.call('EXPIRE', KEYS[1], ARGV[3]) end",
    ] {
        let at = script
            .find(write)
            .unwrap_or_else(|| panic!("the script writes: {write}"));
        assert!(at > check, "written after the status check: {write}");
    }
    assert!(script.trim_end().ends_with("return 'stored'"));
}

#[test]
fn the_object_field_argument_is_empty_when_the_schema_skips_the_column() {
    let storage = |schema: Option<SchemaConfig>| {
        RedisResponseStorage::new(
            RedisStore::new(RedisConfig {
                url: "redis://127.0.0.1:1/0".to_string(),
                pool_max: 1,
                op_timeout_secs: 1,
                schema,
                retention_days: None,
            })
            .expect("a pool is built without a connection"),
        )
    };
    assert_eq!(storage(None).object_field_arg(), "raw_response");
    let mut mapped = SchemaConfig::default();
    mapped
        .responses
        .columns
        .insert("raw_response".to_string(), "payload".to_string());
    assert_eq!(storage(Some(mapped)).object_field_arg(), "payload");
    let mut skipped = SchemaConfig::default();
    skipped
        .responses
        .skip_columns
        .insert("raw_response".to_string());
    assert_eq!(storage(Some(skipped)).object_field_arg(), "");
}

#[test]
fn the_retention_in_seconds_is_the_configured_days() {
    let store = |retention_days: Option<u64>| {
        RedisStore::new(RedisConfig {
            url: "redis://127.0.0.1:1/0".to_string(),
            pool_max: 1,
            op_timeout_secs: 1,
            schema: None,
            retention_days,
        })
        .expect("a pool is built without a connection")
    };
    assert_eq!(store(Some(2)).retention_seconds(), Some(172_800));
    assert_eq!(store(None).retention_seconds(), None);
}
