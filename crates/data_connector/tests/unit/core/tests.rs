use std::collections::HashSet;

use super::*;

// ========================================================================
// ConversationId tests
// ========================================================================

#[test]
fn conversation_id_new_has_conv_prefix() {
    let id = ConversationId::new();
    assert!(
        id.0.starts_with("conv_"),
        "ConversationId should start with 'conv_', got: {id}"
    );
}

#[test]
fn conversation_id_new_generates_unique_ids() {
    let ids: HashSet<String> = (0..100).map(|_| ConversationId::new().0).collect();
    assert_eq!(ids.len(), 100, "all 100 ConversationIds should be unique");
}

#[test]
fn conversation_id_new_has_consistent_length() {
    // "conv_" (5 chars) + 50 hex chars = 55 total
    for _ in 0..10 {
        let id = ConversationId::new();
        assert_eq!(
            id.0.len(),
            55,
            "ConversationId should be 55 chars (conv_ + 50 hex), got {} chars: {id}",
            id.0.len()
        );
    }
}

#[test]
fn conversation_id_default_works_same_as_new() {
    let id = ConversationId::default();
    assert!(
        id.0.starts_with("conv_"),
        "Default ConversationId should start with 'conv_', got: {id}"
    );
    assert_eq!(id.0.len(), 55, "Default ConversationId should be 55 chars");
}

#[test]
fn conversation_id_from_string() {
    let id = ConversationId::from("my_custom_id".to_string());
    assert_eq!(id.0, "my_custom_id");
}

#[test]
fn conversation_id_from_str() {
    let id = ConversationId::from("my_custom_id");
    assert_eq!(id.0, "my_custom_id");
}

#[test]
fn conversation_id_display() {
    let id = ConversationId::from("conv_abc123");
    assert_eq!(format!("{id}"), "conv_abc123");
}

// ========================================================================
// ConversationItemId tests
// ========================================================================

#[test]
fn conversation_item_id_from_string() {
    let id = ConversationItemId::from("item_123".to_string());
    assert_eq!(id.0, "item_123");
}

#[test]
fn conversation_item_id_from_str() {
    let id = ConversationItemId::from("item_456");
    assert_eq!(id.0, "item_456");
}

#[test]
fn conversation_item_id_display() {
    let id = ConversationItemId::from("msg_abc");
    assert_eq!(format!("{id}"), "msg_abc");
}

// ========================================================================
// ResponseId tests
// ========================================================================

#[test]
fn response_id_new_generates_valid_ulid() {
    let id = ResponseId::new();
    // ULID strings are 26 characters, uppercase alphanumeric (Crockford Base32)
    assert_eq!(
        id.0.len(),
        26,
        "ULID string should be 26 chars, got {} chars: {}",
        id.0.len(),
        id.0
    );
    assert!(
        id.0.chars().all(|c| c.is_ascii_alphanumeric()),
        "ULID should contain only alphanumeric characters, got: {}",
        id.0
    );
}

#[test]
fn response_id_new_generates_unique_ids() {
    let ids: HashSet<String> = (0..100).map(|_| ResponseId::new().0).collect();
    assert_eq!(ids.len(), 100, "all 100 ResponseIds should be unique");
}

#[test]
fn response_id_default_works_same_as_new() {
    let id = ResponseId::default();
    assert_eq!(id.0.len(), 26, "Default ResponseId should be 26-char ULID");
}

#[test]
fn response_id_from_string() {
    let id = ResponseId::from("resp_custom".to_string());
    assert_eq!(id.0, "resp_custom");
}

#[test]
fn response_id_from_str() {
    let id = ResponseId::from("resp_custom");
    assert_eq!(id.0, "resp_custom");
}

// ========================================================================
// make_item_id() tests
// ========================================================================

#[test]
fn make_item_id_message_prefix() {
    let id = make_item_id("message");
    assert!(
        id.0.starts_with("msg_"),
        "message type should produce 'msg_' prefix, got: {id}"
    );
}

#[test]
fn make_item_id_reasoning_prefix() {
    let id = make_item_id("reasoning");
    assert!(
        id.0.starts_with("rs_"),
        "reasoning type should produce 'rs_' prefix, got: {id}"
    );
}

#[test]
fn make_item_id_mcp_call_prefix() {
    let id = make_item_id("mcp_call");
    assert!(
        id.0.starts_with("mcp_"),
        "mcp_call type should produce 'mcp_' prefix, got: {id}"
    );
}

#[test]
fn make_item_id_mcp_list_tools_prefix() {
    let id = make_item_id("mcp_list_tools");
    assert!(
        id.0.starts_with("mcpl_"),
        "mcp_list_tools type should produce 'mcpl_' prefix, got: {id}"
    );
}

#[test]
fn make_item_id_function_call_prefix() {
    let id = make_item_id("function_call");
    assert!(
        id.0.starts_with("fc_"),
        "function_call type should produce 'fc_' prefix, got: {id}"
    );
}

#[test]
fn make_item_id_unknown_type_uses_first_3_chars() {
    let id = make_item_id("custom_type");
    assert!(
        id.0.starts_with("cus_"),
        "unknown type 'custom_type' should produce 'cus_' prefix, got: {id}"
    );
}

#[test]
fn make_item_id_empty_type_uses_itm() {
    let id = make_item_id("");
    assert!(
        id.0.starts_with("itm_"),
        "empty type string should produce 'itm_' prefix, got: {id}"
    );
}

#[test]
fn make_item_id_correct_length() {
    // Each known prefix: prefix + "_" + 50 hex chars
    let test_cases = vec![
        ("message", "msg_"),
        ("reasoning", "rs_"),
        ("mcp_call", "mcp_"),
        ("mcp_list_tools", "mcpl_"),
        ("function_call", "fc_"),
    ];

    for (item_type, prefix) in test_cases {
        let id = make_item_id(item_type);
        let expected_len = prefix.len() + 50;
        assert_eq!(
            id.0.len(),
            expected_len,
            "make_item_id(\"{item_type}\") should be {expected_len} chars ('{prefix}' + 50 hex), got {} chars: {id}",
            id.0.len()
        );
    }

    // Unknown type: first 3 chars + "_" + 50 hex = 54 chars
    let id = make_item_id("custom_type");
    assert_eq!(
        id.0.len(),
        54,
        "unknown type should be 54 chars (3 char prefix + '_' + 50 hex), got {} chars: {id}",
        id.0.len()
    );

    // Empty type: "itm_" + 50 hex = 54 chars
    let id = make_item_id("");
    assert_eq!(
        id.0.len(),
        54,
        "empty type should be 54 chars ('itm_' + 50 hex), got {} chars: {id}",
        id.0.len()
    );
}

// ========================================================================
// Conversation tests
// ========================================================================

#[test]
fn conversation_new_generates_id_if_none_provided() {
    let conv = Conversation::new(NewConversation {
        id: None,
        metadata: None,
        tenant: None,
    });
    assert!(
        conv.id.0.starts_with("conv_"),
        "should generate a ConversationId when none provided, got: {}",
        conv.id
    );
}

#[test]
fn conversation_new_uses_provided_id() {
    let custom_id = ConversationId::from("my_conv_id");
    let conv = Conversation::new(NewConversation {
        id: Some(custom_id.clone()),
        metadata: None,
        tenant: None,
    });
    assert_eq!(conv.id, custom_id, "should use the provided ConversationId");
}

#[test]
fn conversation_with_parts_preserves_all_fields() {
    let id = ConversationId::from("test_id");
    let created_at = Utc::now();
    let mut metadata = ConversationMetadata::new();
    metadata.insert("key".to_string(), Value::String("value".to_string()));

    let conv = Conversation::with_parts(
        id.clone(),
        created_at,
        Some(metadata.clone()),
        Some("header:team-red".to_string()),
    );

    assert_eq!(conv.id, id);
    assert_eq!(conv.created_at, created_at);
    assert_eq!(conv.metadata, Some(metadata));
    assert_eq!(conv.tenant.as_deref(), Some("header:team-red"));
}

/// A chain is served down to the first record of another tenant; a head
/// of another tenant hides the whole chain.
#[test]
fn a_chain_is_cut_at_the_first_record_of_another_tenant() {
    let stored = |id: &str, tenant: Option<&str>| {
        let mut stored = StoredResponse::new(None);
        stored.id = ResponseId::from(id);
        stored.tenant = tenant.map(str::to_string);
        Arc::new(stored)
    };
    let ids = |chain: Vec<Arc<StoredResponse>>| {
        chain
            .iter()
            .map(|stored| stored.id.0.clone())
            .collect::<Vec<_>>()
    };
    let a = Some("header:a");
    let chain = || {
        vec![
            stored("resp_1", None),
            stored("resp_2", Some("header:b")),
            stored("resp_3", a),
            stored("resp_4", a),
        ]
    };
    assert_eq!(ids(visible_chain(chain(), a)), ["resp_3", "resp_4"]);
    assert!(
        visible_chain(chain(), Some("header:b")).is_empty(),
        "the head is a's"
    );
    assert!(visible_chain(chain(), None).is_empty());
    let shared = vec![stored("resp_1", None), stored("resp_2", None)];
    assert_eq!(
        ids(visible_chain(shared, Some("header:b"))),
        ["resp_1", "resp_2"],
        "records without a tenant are served as before"
    );
    assert!(visible_chain(Vec::new(), a).is_empty());
}

/// A record stored under a tenant is served to that tenant only; a record
/// stored under none is served to every caller, as before.
#[test]
fn tenant_scope_serves_a_tenanted_record_to_its_tenant_only() {
    assert!(tenant_scope_allows(None, None));
    assert!(tenant_scope_allows(None, Some("header:b")));
    assert!(tenant_scope_allows(Some("header:a"), Some("header:a")));
    assert!(!tenant_scope_allows(Some("header:a"), Some("header:b")));
    assert!(!tenant_scope_allows(Some("header:a"), None));

    let mut response = StoredResponse::new(None);
    assert!(response.is_visible_to(Some("auth:anyone")));
    response.tenant = Some("auth:team-red".to_string());
    assert!(response.is_visible_to(Some("auth:team-red")));
    assert!(!response.is_visible_to(Some("auth:team-blue")));
    assert!(!response.is_visible_to(None));

    let conversation = Conversation::new(NewConversation {
        id: None,
        metadata: None,
        tenant: Some("header:team-red".to_string()),
    });
    assert_eq!(conversation.tenant.as_deref(), Some("header:team-red"));
    assert!(conversation.is_visible_to(Some("header:team-red")));
    assert!(!conversation.is_visible_to(Some("header:team-blue")));
    assert!(!conversation.is_visible_to(None));
}

// ========================================================================
// StoredResponse tests
// ========================================================================

#[test]
fn stored_response_new_none_has_no_previous() {
    let resp = StoredResponse::new(None);
    assert!(
        resp.previous_response_id.is_none(),
        "new(None) should have no previous_response_id"
    );
}

#[test]
fn stored_response_new_some_has_correct_previous() {
    let prev_id = ResponseId::from("prev_123");
    let resp = StoredResponse::new(Some(prev_id.clone()));
    assert_eq!(
        resp.previous_response_id,
        Some(prev_id),
        "new(Some(id)) should set previous_response_id"
    );
}

#[test]
fn stored_response_default_works() {
    let resp = StoredResponse::default();
    assert!(
        resp.previous_response_id.is_none(),
        "default() should have no previous_response_id"
    );
    assert_eq!(
        resp.input,
        Value::Array(vec![]),
        "default input should be empty array"
    );
    assert_eq!(
        resp.raw_response,
        Value::Null,
        "default raw_response should be Null"
    );
}

// ========================================================================
// ResponseChain tests
// ========================================================================

#[test]
fn response_chain_new_creates_empty_chain() {
    let chain = ResponseChain::new();
    assert!(
        chain.responses.is_empty(),
        "new chain should have no responses"
    );
    assert!(
        chain.metadata.is_empty(),
        "new chain should have no metadata"
    );
}

#[test]
fn response_chain_add_response_appends() {
    let mut chain = ResponseChain::new();
    let r1 = StoredResponse::new(None);
    let r2 = StoredResponse::new(None);
    let r1_id = r1.id.clone();
    let r2_id = r2.id.clone();

    chain.add_response(r1);
    assert_eq!(chain.responses.len(), 1, "chain should have 1 response");

    chain.add_response(r2);
    assert_eq!(chain.responses.len(), 2, "chain should have 2 responses");
    assert_eq!(chain.responses[0].id, r1_id, "first response should be r1");
    assert_eq!(chain.responses[1].id, r2_id, "second response should be r2");
}

// ========================================================================
// AddedAtSequence tests
// ========================================================================

#[test]
fn added_at_sequence_stamps_strictly_increase_at_microsecond_resolution() {
    let mut stamps = AddedAtSequence::new();
    let started = Utc::now();
    let mut previous = stamps.next_added_at();
    assert!(previous >= started.trunc_subsecs(6));
    // A tight loop asks faster than the clock moves: every stamp still gets its own
    // microsecond...
    for _ in 0..1000 {
        let next = stamps.next_added_at();
        assert!(next.timestamp_micros() > previous.timestamp_micros());
        assert_eq!(
            next.timestamp_subsec_nanos() % 1000,
            0,
            "whole microseconds"
        );
        previous = next;
    }
    // ...and the sequence runs at most one microsecond per stamp ahead of the clock.
    assert!(previous <= Utc::now() + chrono::TimeDelta::microseconds(1001));
}

#[test]
fn added_at_sequence_follows_the_clock_once_it_has_moved() {
    let mut stamps = AddedAtSequence::new();
    let first = stamps.next_added_at();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let second = stamps.next_added_at();
    assert!(second - first >= chrono::TimeDelta::milliseconds(1));
}

/// A store that overrides neither conditional write runs the trait's
/// read-then-write defaults: a record that cannot be read is an error to
/// the caller, never an open record, and nothing is written; a closed
/// record is handed back and left as it is; a missing one takes nothing;
/// an open one takes the write.
#[tokio::test]
async fn the_default_conditional_writes_write_nothing_over_a_record_they_cannot_read() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use parking_lot::Mutex;

    enum Read {
        Fails,
        Answers(Option<Box<StoredResponse>>),
    }

    struct PlainStore {
        read: Mutex<Read>,
        stores: AtomicUsize,
    }

    #[async_trait]
    impl ResponseStorage for PlainStore {
        async fn store_response(&self, response: StoredResponse) -> ResponseResult<ResponseId> {
            self.stores.fetch_add(1, Ordering::SeqCst);
            Ok(response.id)
        }

        async fn get_response(
            &self,
            _response_id: &ResponseId,
        ) -> ResponseResult<Option<StoredResponse>> {
            match &*self.read.lock() {
                Read::Fails => Err(ResponseStorageError::Timeout("read timed out".to_string())),
                Read::Answers(record) => Ok(record.as_deref().cloned()),
            }
        }

        async fn delete_response(&self, _response_id: &ResponseId) -> ResponseResult<()> {
            Ok(())
        }

        async fn list_identifier_responses(
            &self,
            _identifier: &str,
            _tenant: Option<&str>,
            _limit: Option<usize>,
        ) -> ResponseResult<Vec<StoredResponse>> {
            Ok(Vec::new())
        }

        async fn delete_identifier_responses(
            &self,
            _identifier: &str,
            _tenant: Option<&str>,
        ) -> ResponseResult<usize> {
            Ok(0)
        }
    }

    let record = |status: &str| {
        let mut response = StoredResponse::new(None);
        response.id = ResponseId::from("resp_1");
        response.raw_response = serde_json::json!({"id": "resp_1", "status": status});
        response
    };
    let moved = serde_json::json!({"id": "resp_1", "status": "in_progress"});
    let store = PlainStore {
        read: Mutex::new(Read::Fails),
        stores: AtomicUsize::new(0),
    };
    let stores = || store.stores.load(Ordering::SeqCst);

    assert!(
        store
            .store_response_if_open(record("completed"))
            .await
            .is_err(),
        "a failed read is the caller's error, not an open record"
    );
    assert!(store
        .update_response_object_if_open(&ResponseId::from("resp_1"), moved.clone())
        .await
        .is_err());
    assert_eq!(
        stores(),
        0,
        "nothing is written while the record's state is unknown"
    );

    *store.read.lock() = Read::Answers(Some(Box::new(record("cancelled"))));
    let ConditionalStore::Closed(current) = store
        .store_response_if_open(record("completed"))
        .await
        .unwrap()
    else {
        panic!("a cancelled record is closed");
    };
    assert_eq!(current.raw_response["status"], "cancelled");
    assert!(matches!(
        store
            .update_response_object_if_open(&ResponseId::from("resp_1"), moved.clone())
            .await
            .unwrap(),
        ConditionalStore::Closed(_)
    ));
    assert_eq!(stores(), 0, "a closed record is left as it is");

    *store.read.lock() = Read::Answers(None);
    assert!(matches!(
        store
            .store_response_if_open(record("completed"))
            .await
            .unwrap(),
        ConditionalStore::Missing
    ));
    assert!(matches!(
        store
            .update_response_object_if_open(&ResponseId::from("resp_1"), moved.clone())
            .await
            .unwrap(),
        ConditionalStore::Missing
    ));
    assert_eq!(stores(), 0, "nothing is written under a missing id");

    *store.read.lock() = Read::Answers(Some(Box::new(record("queued"))));
    assert!(matches!(
        store
            .store_response_if_open(record("completed"))
            .await
            .unwrap(),
        ConditionalStore::Stored
    ));
    assert!(matches!(
        store
            .update_response_object_if_open(&ResponseId::from("resp_1"), moved)
            .await
            .unwrap(),
        ConditionalStore::Stored
    ));
    assert_eq!(stores(), 2, "an open record takes both writes");
}
