use super::*;
use crate::{
    core::config::{Prompt, Resource, Tool},
    inventory::AliasTarget,
};

// Helper to create a test tool
fn create_test_tool(name: &str) -> Tool {
    use std::{borrow::Cow, sync::Arc};

    let schema_obj = serde_json::json!({
        "type": "object",
        "properties": {}
    });

    let schema_map = if let serde_json::Value::Object(m) = schema_obj {
        m
    } else {
        serde_json::Map::new()
    };

    Tool::new(
        Cow::Owned(name.to_string()),
        Cow::Owned(format!("Test tool: {name}")),
        Arc::new(schema_map),
    )
}

/// A fresh inventory holding `tool1` of `server1`.
fn inventory_with_tool1() -> ToolInventory {
    let inventory = ToolInventory::new();
    inventory.insert_tool(
        "tool1".to_string(),
        "server1".to_string(),
        create_test_tool("tool1"),
    );
    inventory
}

// Helper to create a test prompt
fn create_test_prompt(name: &str) -> Prompt {
    Prompt::new(name, Some(format!("Test prompt: {name}")), None)
}

// Helper to create a test resource
fn create_test_resource(uri: &str) -> Resource {
    Resource::new(uri, uri)
        .with_description(format!("Test resource: {uri}"))
        .with_mime_type("text/plain")
}

#[test]
fn test_tool_insert_and_get() {
    let inventory = ToolInventory::new();
    let tool = create_test_tool("test_tool");

    inventory.insert_tool("test_tool".to_string(), "server1".to_string(), tool.clone());

    let result = inventory.get_tool("test_tool");
    assert!(result.is_some());

    let (server_name, retrieved_tool) = result.unwrap();
    assert_eq!(server_name, "server1");
    assert_eq!(retrieved_tool.name, "test_tool");
}

#[test]
fn test_has_tool() {
    let inventory = ToolInventory::new();
    let tool = create_test_tool("check_tool");

    assert!(!inventory.has_tool("check_tool"));

    inventory.insert_tool("check_tool".to_string(), "server1".to_string(), tool);

    assert!(inventory.has_tool("check_tool"));
}

#[test]
fn test_list_tools() {
    let inventory = inventory_with_tool1();
    inventory.insert_tool(
        "tool2".to_string(),
        "server1".to_string(),
        create_test_tool("tool2"),
    );
    inventory.insert_tool(
        "tool3".to_string(),
        "server2".to_string(),
        create_test_tool("tool3"),
    );

    let tools = inventory.list_tools();
    assert_eq!(tools.len(), 3);
}

#[test]
fn test_clear_server_tools() {
    let inventory = inventory_with_tool1();
    inventory.insert_tool(
        "tool2".to_string(),
        "server2".to_string(),
        create_test_tool("tool2"),
    );

    assert_eq!(inventory.list_tools().len(), 2);

    inventory.clear_server_tools("server1");

    let tools = inventory.list_tools();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].0, "tool2");
}

#[test]
fn test_prompt_operations() {
    let inventory = ToolInventory::new();
    let prompt = create_test_prompt("test_prompt");

    inventory.insert_prompt(
        "test_prompt".to_string(),
        "server1".to_string(),
        prompt.clone(),
    );

    assert!(inventory.has_prompt("test_prompt"));

    let result = inventory.get_prompt("test_prompt");
    assert!(result.is_some());

    let (server_name, retrieved_prompt) = result.unwrap();
    assert_eq!(server_name, "server1");
    assert_eq!(retrieved_prompt.name, "test_prompt");
}

#[test]
fn test_resource_operations() {
    let inventory = ToolInventory::new();
    let resource = create_test_resource("file:///test.txt");

    inventory.insert_resource(
        "file:///test.txt".to_string(),
        "server1".to_string(),
        resource.clone(),
    );

    assert!(inventory.has_resource("file:///test.txt"));

    let result = inventory.get_resource("file:///test.txt");
    assert!(result.is_some());

    let (server_name, retrieved_resource) = result.unwrap();
    assert_eq!(server_name, "server1");
    assert_eq!(retrieved_resource.uri, "file:///test.txt");
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "test concurrency: handles are stored and awaited"
)]
async fn test_concurrent_access() {
    use std::sync::Arc;

    let inventory = Arc::new(ToolInventory::new());

    // Spawn multiple tasks that insert tools concurrently
    let mut handles = vec![];
    for i in 0..10 {
        let inv = Arc::clone(&inventory);
        let handle = tokio::spawn(async move {
            let tool = create_test_tool(&format!("tool_{i}"));
            inv.insert_tool(format!("tool_{i}"), format!("server_{}", i % 3), tool);
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Should have 10 tools
    let (tools, _, _) = inventory.counts();
    assert_eq!(tools, 10);
}

#[test]
fn test_clear_all() {
    let inventory = inventory_with_tool1();
    inventory.insert_prompt(
        "prompt1".to_string(),
        "server1".to_string(),
        create_test_prompt("prompt1"),
    );
    inventory.insert_resource(
        "res1".to_string(),
        "server1".to_string(),
        create_test_resource("res1"),
    );

    let (tools, prompts, resources) = inventory.counts();
    assert_eq!(tools, 1);
    assert_eq!(prompts, 1);
    assert_eq!(resources, 1);

    inventory.clear_all();

    let (tools, prompts, resources) = inventory.counts();
    assert_eq!(tools, 0);
    assert_eq!(prompts, 0);
    assert_eq!(resources, 0);
}

#[test]
fn test_qualified_tool_name_new() {
    let qualified = QualifiedToolName::new("server-a", "read_file");
    assert_eq!(qualified.server_key(), "server-a");
    assert_eq!(qualified.tool_name(), "read_file");
}

#[test]
fn test_qualified_tool_name_display() {
    let qualified = QualifiedToolName::new("server-b", "write_file");
    assert_eq!(format!("{qualified}"), "server-b:write_file");
}

#[test]
fn test_qualified_tool_name_hash_eq() {
    use std::collections::HashSet;

    let q1 = QualifiedToolName::new("server", "tool");
    let q2 = QualifiedToolName::new("server", "tool");
    let q3 = QualifiedToolName::new("server", "other");

    assert_eq!(q1, q2);
    assert_ne!(q1, q3);

    let mut set = HashSet::new();
    set.insert(q1.clone());
    assert!(set.contains(&q2));
    assert!(!set.contains(&q3));
}

#[test]
fn test_collision_same_tool_name_different_servers() {
    let inventory = ToolInventory::new();
    let tool_a = create_test_tool("read_file");
    let tool_b = create_test_tool("read_file");

    inventory.insert_tool("read_file".to_string(), "server-a".to_string(), tool_a);
    inventory.insert_tool("read_file".to_string(), "server-b".to_string(), tool_b);

    // Both stored (counts total tools including collisions)
    assert_eq!(inventory.counts().0, 2);

    // Simple lookup returns first registered
    let (server, _) = inventory.get_tool("read_file").unwrap();
    assert_eq!(server, "server-a");

    // Qualified lookup can access both
    assert!(inventory
        .get_tool_qualified("server-a", "read_file")
        .is_some());
    assert!(inventory
        .get_tool_qualified("server-b", "read_file")
        .is_some());

    // Get servers list
    let servers = inventory.get_tool_servers("read_file");
    assert_eq!(servers.len(), 2);
    assert!(servers.contains(&"server-a".to_string()));
    assert!(servers.contains(&"server-b".to_string()));
}

#[test]
fn test_clear_server_updates_all_indices() {
    let inventory = ToolInventory::new();

    // Register same tool name from two servers
    inventory.insert_tool(
        "read_file".to_string(),
        "server-a".to_string(),
        create_test_tool("read_file"),
    );
    inventory.insert_tool(
        "read_file".to_string(),
        "server-b".to_string(),
        create_test_tool("read_file"),
    );

    // Initial state: 2 tools, simple lookup returns server-a
    assert_eq!(inventory.counts().0, 2);
    let (server, _) = inventory.get_tool("read_file").unwrap();
    assert_eq!(server, "server-a");

    // Clear server-a
    inventory.clear_server_tools("server-a");

    // After clear: 1 tool, simple lookup should return server-b
    assert_eq!(inventory.counts().0, 1);
    let (server, _) = inventory.get_tool("read_file").unwrap();
    assert_eq!(server, "server-b");

    // Qualified lookup confirms server-a is gone
    assert!(inventory
        .get_tool_qualified("server-a", "read_file")
        .is_none());
    assert!(inventory
        .get_tool_qualified("server-b", "read_file")
        .is_some());
}

#[test]
fn test_clear_server_removes_from_simple_name_when_last() {
    let inventory = ToolInventory::new();

    inventory.insert_tool(
        "unique_tool".to_string(),
        "server-x".to_string(),
        create_test_tool("unique_tool"),
    );

    assert!(inventory.has_tool("unique_tool"));

    inventory.clear_server_tools("server-x");

    // Tool should no longer exist via simple lookup
    assert!(!inventory.has_tool("unique_tool"));
    assert!(inventory.get_tool("unique_tool").is_none());
}

#[test]
fn test_has_tool_qualified() {
    let inventory = ToolInventory::new();

    inventory.insert_tool(
        "my_tool".to_string(),
        "my_server".to_string(),
        create_test_tool("my_tool"),
    );

    assert!(inventory.has_tool_qualified("my_server", "my_tool"));
    assert!(!inventory.has_tool_qualified("other_server", "my_tool"));
    assert!(!inventory.has_tool_qualified("my_server", "other_tool"));
}

#[test]
fn test_list_tools_qualified() {
    let inventory = inventory_with_tool1();
    inventory.insert_tool(
        "tool2".to_string(),
        "server2".to_string(),
        create_test_tool("tool2"),
    );
    // Collision
    inventory.insert_tool(
        "tool1".to_string(),
        "server3".to_string(),
        create_test_tool("tool1"),
    );

    let qualified_tools = inventory.list_tools_qualified();
    assert_eq!(qualified_tools.len(), 3);

    // Check we can find both tool1 entries
    let tool1_entries: Vec<_> = qualified_tools
        .iter()
        .filter(|(q, _)| q.tool_name() == "tool1")
        .collect();
    assert_eq!(tool1_entries.len(), 2);
}

#[test]
fn test_unique_tool_name_count() {
    let inventory = inventory_with_tool1();
    inventory.insert_tool(
        "tool1".to_string(),
        "server2".to_string(),
        create_test_tool("tool1"),
    );
    inventory.insert_tool(
        "tool2".to_string(),
        "server1".to_string(),
        create_test_tool("tool2"),
    );

    // counts().0 returns total tools (including collisions)
    assert_eq!(inventory.counts().0, 3);
    // unique_tool_name_count returns unique names
    assert_eq!(inventory.unique_tool_name_count(), 2);
}

#[test]
fn test_reinsert_same_server_no_duplicate() {
    let inventory = ToolInventory::new();

    // Insert same tool from same server twice
    inventory.insert_tool(
        "my_tool".to_string(),
        "my_server".to_string(),
        create_test_tool("my_tool"),
    );
    inventory.insert_tool(
        "my_tool".to_string(),
        "my_server".to_string(),
        create_test_tool("my_tool"),
    );

    // Should only have 1 tool (updated, not duplicated)
    assert_eq!(inventory.counts().0, 1);
    assert_eq!(inventory.unique_tool_name_count(), 1);

    // Simple name list shouldn't have duplicates
    let servers = inventory.get_tool_servers("my_tool");
    assert_eq!(servers.len(), 1);
}

#[test]
fn test_insert_entry_with_category() {
    use crate::inventory::ToolCategory;

    let inventory = ToolInventory::new();
    let tool = create_test_tool("dynamic_tool");
    let entry =
        ToolEntry::from_server_tool("dynamic_server", tool).with_category(ToolCategory::Dynamic);

    inventory.insert_entry(entry);

    let dynamic_tools = inventory.list_by_category(ToolCategory::Dynamic);
    assert_eq!(dynamic_tools.len(), 1);
    assert_eq!(dynamic_tools[0].tool_name(), "dynamic_tool");

    let static_tools = inventory.list_by_category(ToolCategory::Static);
    assert_eq!(static_tools.len(), 0);
}

#[test]
fn test_alias_registration_and_resolution() {
    let inventory = ToolInventory::new();

    // Register a tool
    inventory.insert_tool(
        "brave_web_search".to_string(),
        "brave".to_string(),
        create_test_tool("brave_web_search"),
    );

    // Register an alias
    let target = QualifiedToolName::new("brave", "brave_web_search");
    inventory.register_alias("web_search".to_string(), target.clone());

    // Resolve alias
    let resolved = inventory.resolve_alias("web_search");
    assert!(resolved.is_some());
    assert_eq!(resolved.unwrap(), target);

    // Non-existent alias returns None
    assert!(inventory.resolve_alias("unknown").is_none());
}

#[test]
fn test_get_tool_or_alias() {
    let inventory = ToolInventory::new();

    // Register a tool
    inventory.insert_tool(
        "brave_web_search".to_string(),
        "brave".to_string(),
        create_test_tool("brave_web_search"),
    );

    // Register an alias
    let target = QualifiedToolName::new("brave", "brave_web_search");
    inventory.register_alias("search".to_string(), target);

    // Direct lookup works
    let (server, _) = inventory.get_tool_or_alias("brave_web_search").unwrap();
    assert_eq!(server, "brave");

    // Alias lookup works
    let (server, _) = inventory.get_tool_or_alias("search").unwrap();
    assert_eq!(server, "brave");

    // Unknown returns None
    assert!(inventory.get_tool_or_alias("unknown").is_none());
}

#[test]
fn test_list_aliases() {
    let inventory = ToolInventory::new();

    inventory.register_alias(
        "alias1".to_string(),
        QualifiedToolName::new("server1", "tool1"),
    );
    inventory.register_alias(
        "alias2".to_string(),
        QualifiedToolName::new("server2", "tool2"),
    );

    let aliases = inventory.list_aliases();
    assert_eq!(aliases.len(), 2);
}

#[test]
fn test_index_counts() {
    let inventory = inventory_with_tool1();
    inventory.register_alias(
        "alias1".to_string(),
        QualifiedToolName::new("server1", "tool1"),
    );

    let counts = inventory.index_counts();
    assert_eq!(counts.tools, 1);
    assert_eq!(counts.unique_names, 1);
    assert_eq!(counts.servers, 1);
    assert_eq!(counts.aliases, 1);
}

#[test]
fn test_clear_server_removes_from_category_index() {
    use crate::inventory::ToolCategory;

    let inventory = ToolInventory::new();

    // Insert a dynamic tool
    let tool = create_test_tool("dyn_tool");
    let entry =
        ToolEntry::from_server_tool("dyn_server", tool).with_category(ToolCategory::Dynamic);
    inventory.insert_entry(entry);

    assert_eq!(inventory.list_by_category(ToolCategory::Dynamic).len(), 1);

    // Clear server
    inventory.clear_server_tools("dyn_server");

    assert_eq!(inventory.list_by_category(ToolCategory::Dynamic).len(), 0);
}

#[test]
fn test_clear_server_removes_aliases_to_server() {
    let inventory = inventory_with_tool1();

    inventory.register_alias(
        "alias1".to_string(),
        QualifiedToolName::new("server1", "tool1"),
    );

    assert_eq!(inventory.list_aliases().len(), 1);

    inventory.clear_server_tools("server1");

    // Alias should be removed since it points to cleared server
    assert_eq!(inventory.list_aliases().len(), 0);
}

#[test]
fn test_get_entry() {
    let inventory = ToolInventory::new();

    inventory.insert_tool(
        "my_tool".to_string(),
        "my_server".to_string(),
        create_test_tool("my_tool"),
    );

    let entry = inventory.get_entry("my_server", "my_tool");
    assert!(entry.is_some());
    let entry = entry.unwrap();
    assert_eq!(entry.server_key(), "my_server");
    assert_eq!(entry.tool_name(), "my_tool");

    assert!(inventory.get_entry("other", "my_tool").is_none());
}

#[test]
fn clearing_a_server_drops_the_alias_entries_that_targeted_it() {
    let inventory = ToolInventory::new();
    inventory.insert_entry(ToolEntry::new(
        QualifiedToolName::new("s", "search"),
        create_test_tool("search"),
    ));
    inventory.insert_entry(ToolEntry::new(
        QualifiedToolName::new("other", "lookup"),
        create_test_tool("lookup"),
    ));
    inventory.insert_entry(
        ToolEntry::new(
            QualifiedToolName::new(ALIAS_SERVER_KEY, "web_search"),
            create_test_tool("search"),
        )
        .with_alias(AliasTarget::new(QualifiedToolName::new("s", "search"))),
    );
    inventory.register_alias(
        "web_search".to_string(),
        QualifiedToolName::new("s", "search"),
    );
    inventory.insert_prompt("p".to_string(), "s".to_string(), create_test_prompt("p"));
    inventory.insert_resource("r".to_string(), "s".to_string(), create_test_resource("r"));

    inventory.clear_server_tool_entries("s");

    assert!(inventory.get_entry("s", "search").is_none());
    assert!(inventory.resolve_alias("web_search").is_none());
    assert!(inventory
        .get_entry(ALIAS_SERVER_KEY, "web_search")
        .is_none());
    assert!(inventory.get_tool_or_alias("web_search").is_none());
    assert!(inventory.get_entry("other", "lookup").is_some());
    // prompts and resources are not the tool clear's business
    assert_eq!(inventory.counts().1, 1);
    assert_eq!(inventory.counts().2, 1);
}

#[test]
fn retaining_a_listing_drops_the_servers_other_tools_and_their_aliases() {
    let inventory = ToolInventory::new();
    for name in ["kept", "gone"] {
        inventory.insert_entry(ToolEntry::new(
            QualifiedToolName::new("s", name),
            create_test_tool(name),
        ));
        let alias = format!("{name}_alias");
        inventory.insert_entry(
            ToolEntry::new(
                QualifiedToolName::new(ALIAS_SERVER_KEY, &alias),
                create_test_tool(name),
            )
            .with_alias(AliasTarget::new(QualifiedToolName::new("s", name))),
        );
        inventory.register_alias(alias, QualifiedToolName::new("s", name));
    }
    inventory.insert_entry(ToolEntry::new(
        QualifiedToolName::new("other", "gone"),
        create_test_tool("gone"),
    ));

    inventory.retain_server_tool_entries("s", &HashSet::from(["kept".to_string()]));

    assert!(inventory.get_entry("s", "kept").is_some());
    assert!(inventory.get_tool_or_alias("kept_alias").is_some());
    assert!(inventory.get_entry("s", "gone").is_none());
    assert!(inventory.resolve_alias("gone_alias").is_none());
    assert!(inventory
        .get_entry(ALIAS_SERVER_KEY, "gone_alias")
        .is_none());
    // another server's tool of the same name is not this server's to drop
    assert_eq!(inventory.get_tool_servers("gone"), ["other"]);
}

#[test]
fn a_kept_alias_takes_its_targets_refreshed_definition() {
    let inventory = ToolInventory::new();
    let target = QualifiedToolName::new("s", "search");
    inventory.insert_entry(ToolEntry::new(target.clone(), create_test_tool("search")));
    inventory.insert_entry(
        ToolEntry::new(
            QualifiedToolName::new(ALIAS_SERVER_KEY, "web_search"),
            create_test_tool("search"),
        )
        .with_alias(AliasTarget::new(target.clone())),
    );
    inventory.register_alias("web_search".to_string(), target.clone());

    // A refresh: the listing's entry goes in, then the retain.
    let mut relisted = create_test_tool("search");
    relisted.description = Some("searches the newer index".into());
    inventory.insert_entry(ToolEntry::new(target.clone(), relisted));
    inventory.retain_server_tool_entries("s", &HashSet::from(["search".to_string()]));

    let alias = inventory
        .get_entry(ALIAS_SERVER_KEY, "web_search")
        .expect("an alias whose target is still listed stays");
    assert_eq!(
        alias.tool.description.as_deref(),
        Some("searches the newer index"),
        "the definition under the alias"
    );
    // still an alias of the same target, under its own name
    assert_eq!(alias.tool_name(), "web_search");
    assert_eq!(inventory.resolve_alias("web_search"), Some(target));
}

#[test]
fn a_kept_alias_another_server_took_over_keeps_the_takers_tool() {
    let inventory = ToolInventory::new();
    let mine = QualifiedToolName::new("s", "search");
    let theirs = QualifiedToolName::new("other", "lookup");
    inventory.insert_entry(ToolEntry::new(mine.clone(), create_test_tool("search")));
    inventory.insert_entry(ToolEntry::new(theirs.clone(), create_test_tool("lookup")));
    // The alias index still names this server's tool, as the retain read
    // it; the entry under the alias key is already the other server's.
    inventory.register_alias("find".to_string(), mine);
    inventory.insert_entry(
        ToolEntry::new(
            QualifiedToolName::new(ALIAS_SERVER_KEY, "find"),
            create_test_tool("lookup"),
        )
        .with_alias(AliasTarget::new(theirs.clone())),
    );

    inventory.retain_server_tool_entries("s", &HashSet::from(["search".to_string()]));

    let alias = inventory
        .get_entry(ALIAS_SERVER_KEY, "find")
        .expect("the alias stays");
    assert_eq!(
        alias.tool.description.as_deref(),
        Some("Test tool: lookup"),
        "the definition under an alias that targets another server's tool"
    );
    assert_eq!(alias.alias_target.map(|held| held.target), Some(theirs));
}

#[test]
fn a_dropped_alias_another_server_took_over_stays_the_takers() {
    let inventory = ToolInventory::new();
    let mine = QualifiedToolName::new("s", "search");
    let theirs = QualifiedToolName::new("other", "lookup");
    inventory.insert_entry(ToolEntry::new(theirs.clone(), create_test_tool("lookup")));
    inventory.insert_entry(
        ToolEntry::new(
            QualifiedToolName::new(ALIAS_SERVER_KEY, "find"),
            create_test_tool("search"),
        )
        .with_alias(AliasTarget::new(mine.clone())),
    );
    inventory.register_alias("find".to_string(), mine.clone());
    // What a retain reads while the alias is still this server's.
    let read = vec![("find".to_string(), mine)];
    // The other server registers the name before the removal: its
    // entry, then the index.
    inventory.insert_entry(
        ToolEntry::new(
            QualifiedToolName::new(ALIAS_SERVER_KEY, "find"),
            create_test_tool("lookup"),
        )
        .with_alias(AliasTarget::new(theirs.clone())),
    );
    inventory.register_alias("find".to_string(), theirs.clone());

    inventory.drop_aliases(read);

    assert_eq!(inventory.resolve_alias("find"), Some(theirs.clone()));
    let alias = inventory
        .get_entry(ALIAS_SERVER_KEY, "find")
        .expect("the taker's entry under the alias stays");
    assert_eq!(alias.tool.description.as_deref(), Some("Test tool: lookup"));
    assert_eq!(alias.alias_target.map(|held| held.target), Some(theirs));
}

#[test]
fn the_per_kind_clears_leave_the_other_kinds() {
    let inventory = ToolInventory::new();
    inventory.insert_entry(ToolEntry::new(
        QualifiedToolName::new("s", "search"),
        create_test_tool("search"),
    ));
    inventory.insert_prompt("p".to_string(), "s".to_string(), create_test_prompt("p"));
    inventory.insert_resource("r".to_string(), "s".to_string(), create_test_resource("r"));

    inventory.clear_server_prompts("s");
    assert_eq!(inventory.counts(), (1, 0, 1));
    inventory.clear_server_resources("s");
    assert_eq!(inventory.counts(), (1, 0, 0));
    inventory.clear_server_tools("s");
    assert_eq!(inventory.counts(), (0, 0, 0));
}

/// The server index lists each named server's entries under the server's
/// first place in the keys: a key named twice once, an unknown key none,
/// the other servers' entries left out.
#[test]
fn the_entries_of_servers_come_through_the_server_index() {
    let inventory = ToolInventory::new();
    for (server, tool) in [("a", "one"), ("a", "two"), ("b", "three"), ("c", "four")] {
        inventory.insert_tool(tool.to_string(), server.to_string(), create_test_tool(tool));
    }

    let keys = ["b", "missing", "a", "b"].map(str::to_string);
    let mut listed: Vec<(usize, String)> = inventory
        .entries_of_servers(&keys)
        .into_iter()
        .map(|(rank, entry)| (rank, entry.tool_name().to_string()))
        .collect();
    listed.sort();

    let expected =
        [(0, "three"), (2, "one"), (2, "two")].map(|(rank, tool)| (rank, tool.to_string()));
    assert_eq!(listed, expected);
}
