//! Qualified tool index with multi-key lookup.
//!
//! Thread-safe cache for MCP capabilities across all connected servers.
//! Provides multiple indices for efficient lookup:
//! - By qualified name (server + tool)
//! - By simple name (with collision handling)
//! - By server (for bulk operations)
//! - By category (for filtering)

use std::collections::HashSet;

use dashmap::DashMap;
use tracing::warn;

use super::types::{QualifiedToolName, ToolCategory, ToolEntry, ALIAS_SERVER_KEY};
use crate::core::config::{Prompt, RawResource, Tool};

/// Cached prompt with metadata
#[derive(Clone)]
pub(crate) struct CachedPrompt {
    pub server_name: String,
    pub prompt: Prompt,
}

/// Cached resource with metadata
#[derive(Clone)]
pub(crate) struct CachedResource {
    pub server_name: String,
    pub resource: RawResource,
}

pub struct ToolInventory {
    tools_by_qualified: DashMap<QualifiedToolName, ToolEntry>,
    tools_by_simple_name: DashMap<String, Vec<QualifiedToolName>>,
    tools_by_server: DashMap<String, HashSet<String>>,
    tools_by_category: DashMap<ToolCategory, HashSet<QualifiedToolName>>,
    aliases: DashMap<String, QualifiedToolName>,
    prompts: DashMap<String, CachedPrompt>,
    resources: DashMap<String, CachedResource>,
}

impl ToolInventory {
    pub fn new() -> Self {
        Self {
            tools_by_qualified: DashMap::new(),
            tools_by_simple_name: DashMap::new(),
            tools_by_server: DashMap::new(),
            tools_by_category: DashMap::new(),
            aliases: DashMap::new(),
            prompts: DashMap::new(),
            resources: DashMap::new(),
        }
    }
}

impl Default for ToolInventory {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolInventory {
    /// Returns first registered tool on collision. Use `get_tool_qualified()` for specific server.
    pub fn get_tool(&self, tool_name: &str) -> Option<(String, Tool)> {
        let qualified_names = self.tools_by_simple_name.get(tool_name)?;
        let qualified = qualified_names.first()?;
        self.tools_by_qualified
            .get(qualified)
            .map(|entry| (entry.server_key().to_string(), entry.tool.clone()))
    }

    /// Check if a tool with the given simple name is registered.
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tools_by_simple_name.contains_key(tool_name)
    }

    /// Insert a tool. On collision, both are stored; first registered is "primary".
    ///
    /// Note: The `tool_name` parameter is kept for API compatibility but the tool's
    /// internal name field is used. They should match for consistency.
    pub fn insert_tool(&self, _tool_name: String, server_key: String, tool: Tool) {
        let entry = ToolEntry::from_server_tool(&server_key, tool);
        self.insert_entry(entry);
    }

    /// Insert a full tool entry with all metadata.
    pub fn insert_entry(&self, entry: ToolEntry) {
        let qualified = entry.qualified_name.clone();
        let tool_name = qualified.tool_name().to_string();
        let server_key = qualified.server_key().to_string();

        // Log collision warning (single lookup)
        if let Some(existing) = self.tools_by_simple_name.get(&tool_name) {
            if !self.tools_by_qualified.contains_key(&qualified) {
                let existing_servers: Vec<&str> = existing.iter().map(|q| q.server_key()).collect();
                warn!(
                    "Tool name collision: '{}' registered by {:?}, adding from '{}'",
                    tool_name, existing_servers, server_key
                );
            }
        }

        // Update category index
        self.tools_by_category
            .entry(entry.category)
            .or_default()
            .insert(qualified.clone());

        // Insert into primary index
        self.tools_by_qualified.insert(qualified.clone(), entry);

        // Update simple name index
        self.tools_by_simple_name
            .entry(tool_name.clone())
            .and_modify(|v| {
                if !v.contains(&qualified) {
                    v.push(qualified.clone());
                }
            })
            .or_insert_with(|| vec![qualified]);

        // Update server index
        self.tools_by_server
            .entry(server_key)
            .or_default()
            .insert(tool_name);
    }

    /// Get the full tool entry by qualified name.
    pub fn get_entry(&self, server_key: &str, tool_name: &str) -> Option<ToolEntry> {
        let qualified = QualifiedToolName::new(server_key, tool_name);
        self.tools_by_qualified.get(&qualified).map(|e| e.clone())
    }

    /// Atomically update a tool entry if it exists.
    ///
    /// Returns true if the entry was found and updated, false otherwise.
    pub fn update_entry<F>(&self, server_key: &str, tool_name: &str, f: F) -> bool
    where
        F: FnOnce(&mut ToolEntry),
    {
        let qualified = QualifiedToolName::new(server_key, tool_name);
        if let Some(mut entry) = self.tools_by_qualified.get_mut(&qualified) {
            f(&mut entry);
            true
        } else {
            false
        }
    }

    /// Register an alias for a tool.
    pub fn register_alias(&self, alias_name: String, target: QualifiedToolName) {
        self.aliases.insert(alias_name, target);
    }

    /// Resolve an alias to its target qualified name.
    pub fn resolve_alias(&self, alias_name: &str) -> Option<QualifiedToolName> {
        self.aliases.get(alias_name).map(|e| e.clone())
    }

    /// Get a tool by name, resolving aliases if necessary.
    pub fn get_tool_or_alias(&self, name: &str) -> Option<(String, Tool)> {
        // First try direct lookup
        if let Some(result) = self.get_tool(name) {
            return Some(result);
        }
        // Then try alias resolution
        if let Some(target) = self.resolve_alias(name) {
            return self
                .get_tool_qualified(target.server_key(), target.tool_name())
                .map(|tool| (target.server_key().to_string(), tool));
        }
        None
    }

    /// List all tools in a specific category.
    pub fn list_by_category(&self, category: ToolCategory) -> Vec<ToolEntry> {
        self.tools_by_category
            .get(&category)
            .map(|qualified_names| {
                qualified_names
                    .iter()
                    .filter_map(|q| self.tools_by_qualified.get(q).map(|e| e.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// List all registered aliases.
    pub fn list_aliases(&self) -> Vec<(String, QualifiedToolName)> {
        self.aliases
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect()
    }

    /// Returns all tools. If collisions exist, all versions are included.
    pub fn list_tools(&self) -> Vec<(String, String, Tool)> {
        self.tools_by_qualified
            .iter()
            .map(|entry| {
                let (qualified, tool_entry) = entry.pair();
                (
                    qualified.tool_name().to_string(),
                    tool_entry.server_key().to_string(),
                    tool_entry.tool.clone(),
                )
            })
            .collect()
    }

    /// Get a specific server's tool by qualified name.
    pub fn get_tool_qualified(&self, server_key: &str, tool_name: &str) -> Option<Tool> {
        let qualified = QualifiedToolName::new(server_key, tool_name);
        self.tools_by_qualified
            .get(&qualified)
            .map(|entry| entry.tool.clone())
    }

    /// Check if a tool exists by qualified name.
    pub fn has_tool_qualified(&self, server_key: &str, tool_name: &str) -> bool {
        let qualified = QualifiedToolName::new(server_key, tool_name);
        self.tools_by_qualified.contains_key(&qualified)
    }

    /// List all tools with their qualified names.
    pub fn list_tools_qualified(&self) -> Vec<(QualifiedToolName, Tool)> {
        self.tools_by_qualified
            .iter()
            .map(|entry| {
                let (qualified, tool_entry) = entry.pair();
                (qualified.clone(), tool_entry.tool.clone())
            })
            .collect()
    }

    /// Get all servers that have registered a tool with the given name.
    pub fn get_tool_servers(&self, tool_name: &str) -> Vec<String> {
        self.tools_by_simple_name
            .get(tool_name)
            .map(|qualified_names| {
                qualified_names
                    .iter()
                    .map(|q| q.server_key().to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get a prompt by name, returning the server and prompt info.
    pub fn get_prompt(&self, prompt_name: &str) -> Option<(String, Prompt)> {
        self.prompts
            .get(prompt_name)
            .map(|entry| (entry.server_name.clone(), entry.prompt.clone()))
    }

    /// Check if a prompt with the given name is registered.
    pub fn has_prompt(&self, prompt_name: &str) -> bool {
        self.prompts.contains_key(prompt_name)
    }

    /// Insert or update a prompt.
    pub fn insert_prompt(&self, prompt_name: String, server_name: String, prompt: Prompt) {
        self.prompts.insert(
            prompt_name,
            CachedPrompt {
                server_name,
                prompt,
            },
        );
    }

    /// List all prompts as (name, server, prompt) tuples.
    pub fn list_prompts(&self) -> Vec<(String, String, Prompt)> {
        self.prompts
            .iter()
            .map(|entry| {
                let (name, cached) = entry.pair();
                (
                    name.clone(),
                    cached.server_name.clone(),
                    cached.prompt.clone(),
                )
            })
            .collect()
    }

    /// Get a resource by URI, returning the server and resource info.
    pub fn get_resource(&self, resource_uri: &str) -> Option<(String, RawResource)> {
        self.resources
            .get(resource_uri)
            .map(|entry| (entry.server_name.clone(), entry.resource.clone()))
    }

    /// Check if a resource with the given URI is registered.
    pub fn has_resource(&self, resource_uri: &str) -> bool {
        self.resources.contains_key(resource_uri)
    }

    /// Insert or update a resource.
    pub fn insert_resource(
        &self,
        resource_uri: String,
        server_name: String,
        resource: RawResource,
    ) {
        self.resources.insert(
            resource_uri,
            CachedResource {
                server_name,
                resource,
            },
        );
    }

    /// List all resources as (uri, server, resource) tuples.
    pub fn list_resources(&self) -> Vec<(String, String, RawResource)> {
        self.resources
            .iter()
            .map(|entry| {
                let (uri, cached) = entry.pair();
                (
                    uri.clone(),
                    cached.server_name.clone(),
                    cached.resource.clone(),
                )
            })
            .collect()
    }

    /// Clear all cached items for a server: its tools (with the aliases that
    /// target them), its prompts and its resources. Uses the server index for
    /// O(tools_per_server) removal.
    pub fn clear_server_tools(&self, server_key: &str) {
        self.clear_server_tool_entries(server_key);
        self.clear_server_prompts(server_key);
        self.clear_server_resources(server_key);
    }

    /// Clear a server's tools and the aliases that target them, each alias's
    /// own entry under the alias server key included, so a tool the server
    /// dropped is not still offered under its alias. Prompts and resources
    /// stay.
    pub fn clear_server_tool_entries(&self, server_key: &str) {
        if let Some((_, tool_names)) = self.tools_by_server.remove(server_key) {
            for tool_name in tool_names {
                self.remove_entry(&QualifiedToolName::new(server_key, &tool_name));
            }
        }

        // Remove aliases that point to this server, index and entry alike
        let aliases: Vec<String> = self
            .aliases
            .iter()
            .filter(|entry| entry.value().server_key() == server_key)
            .map(|entry| entry.key().clone())
            .collect();
        for alias in aliases {
            self.aliases.remove(&alias);
            let qualified = QualifiedToolName::new(ALIAS_SERVER_KEY, &alias);
            self.remove_entry(&qualified);
            if let Some(mut names) = self.tools_by_server.get_mut(ALIAS_SERVER_KEY) {
                names.remove(&alias);
            }
            self.tools_by_server
                .remove_if(ALIAS_SERVER_KEY, |_, names| names.is_empty());
        }
    }

    /// Clear a server's prompts; its tools and resources stay.
    pub fn clear_server_prompts(&self, server_key: &str) {
        self.prompts
            .retain(|_, cached| cached.server_name != server_key);
    }

    /// Clear a server's resources; its tools and prompts stay.
    pub fn clear_server_resources(&self, server_key: &str) {
        self.resources
            .retain(|_, cached| cached.server_name != server_key);
    }

    /// Remove one entry from the primary, category and simple-name indexes
    /// (the server index is the caller's).
    fn remove_entry(&self, qualified: &QualifiedToolName) {
        let tool_name = qualified.tool_name().to_string();
        // Get entry before removing to access metadata
        if let Some((_, entry)) = self.tools_by_qualified.remove(qualified) {
            // Remove from category index
            if let Some(mut cat_set) = self.tools_by_category.get_mut(&entry.category) {
                cat_set.remove(qualified);
            }
        }

        // Remove from simple name index
        if let Some(mut entry) = self.tools_by_simple_name.get_mut(&tool_name) {
            entry.retain(|q| q != qualified);
        }
        self.tools_by_simple_name
            .remove_if(&tool_name, |_, v| v.is_empty());
    }

    /// Returns (total_tools, prompts, resources). Total includes collision duplicates.
    pub fn counts(&self) -> (usize, usize, usize) {
        (
            self.tools_by_qualified.len(),
            self.prompts.len(),
            self.resources.len(),
        )
    }

    /// Returns unique tool name count (excludes collision duplicates).
    pub fn unique_tool_name_count(&self) -> usize {
        self.tools_by_simple_name.len()
    }

    pub fn clear_all(&self) {
        self.tools_by_qualified.clear();
        self.tools_by_simple_name.clear();
        self.tools_by_server.clear();
        self.tools_by_category.clear();
        self.aliases.clear();
        self.prompts.clear();
        self.resources.clear();
    }

    pub fn index_counts(&self) -> IndexCounts {
        IndexCounts {
            tools: self.tools_by_qualified.len(),
            unique_names: self.tools_by_simple_name.len(),
            servers: self.tools_by_server.len(),
            categories: self.tools_by_category.len(),
            aliases: self.aliases.len(),
            prompts: self.prompts.len(),
            resources: self.resources.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndexCounts {
    pub tools: usize,
    pub unique_names: usize,
    pub servers: usize,
    pub categories: usize,
    pub aliases: usize,
    pub prompts: usize,
    pub resources: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::config::{Prompt, RawResource, Tool},
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

    // Helper to create a test prompt
    fn create_test_prompt(name: &str) -> Prompt {
        Prompt::new(name, Some(format!("Test prompt: {name}")), None)
    }

    // Helper to create a test resource
    fn create_test_resource(uri: &str) -> RawResource {
        RawResource::new(uri, uri)
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
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            create_test_tool("tool1"),
        );
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
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            create_test_tool("tool1"),
        );
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
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            create_test_tool("tool1"),
        );
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
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            create_test_tool("tool1"),
        );
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
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            create_test_tool("tool1"),
        );
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
        let entry = ToolEntry::from_server_tool("dynamic_server", tool)
            .with_category(ToolCategory::Dynamic);

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
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            create_test_tool("tool1"),
        );
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
        let inventory = ToolInventory::new();

        inventory.insert_tool(
            "tool1".to_string(),
            "server1".to_string(),
            create_test_tool("tool1"),
        );

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
}
