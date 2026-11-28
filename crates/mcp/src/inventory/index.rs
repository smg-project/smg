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
use crate::core::config::{Prompt, Resource, Tool};

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
    pub resource: Resource,
}

pub struct ToolInventory {
    tools_by_qualified: DashMap<QualifiedToolName, ToolEntry>,
    tools_by_simple_name: DashMap<String, Vec<QualifiedToolName>>,
    tools_by_server: DashMap<String, HashSet<QualifiedToolName>>,
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
            .entry(tool_name)
            .and_modify(|v| {
                if !v.contains(&qualified) {
                    v.push(qualified.clone());
                }
            })
            .or_insert_with(|| vec![qualified.clone()]);

        // Update server index
        self.tools_by_server
            .entry(server_key)
            .or_default()
            .insert(qualified);
    }

    /// Get the full tool entry by qualified name.
    pub fn get_entry(&self, server_key: &str, tool_name: &str) -> Option<ToolEntry> {
        let qualified = QualifiedToolName::new(server_key, tool_name);
        self.tools_by_qualified.get(&qualified).map(|e| e.clone())
    }

    /// The entries of the servers named, each with its server's place in
    /// `server_keys` (the first, for a key named twice) and cloned once.
    /// Read through the server index: the cost follows these servers' tools,
    /// not the size of the inventory or the shard count of its maps.
    pub(crate) fn entries_of_servers(&self, server_keys: &[String]) -> Vec<(usize, ToolEntry)> {
        let mut entries = Vec::new();
        for (rank, server_key) in server_keys.iter().enumerate() {
            if server_keys
                .iter()
                .take(rank)
                .any(|earlier| earlier == server_key)
            {
                continue;
            }
            let Some(names) = self.tools_by_server.get(server_key) else {
                continue;
            };
            entries.reserve(names.len());
            entries.extend(names.iter().filter_map(|qualified| {
                self.tools_by_qualified
                    .get(qualified)
                    .map(|entry| (rank, entry.clone()))
            }));
        }
        entries
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
    pub fn get_resource(&self, resource_uri: &str) -> Option<(String, Resource)> {
        self.resources
            .get(resource_uri)
            .map(|entry| (entry.server_name.clone(), entry.resource.clone()))
    }

    /// Check if a resource with the given URI is registered.
    pub fn has_resource(&self, resource_uri: &str) -> bool {
        self.resources.contains_key(resource_uri)
    }

    /// Insert or update a resource.
    pub fn insert_resource(&self, resource_uri: String, server_name: String, resource: Resource) {
        self.resources.insert(
            resource_uri,
            CachedResource {
                server_name,
                resource,
            },
        );
    }

    /// List all resources as (uri, server, resource) tuples.
    pub fn list_resources(&self) -> Vec<(String, String, Resource)> {
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
        self.retain_server_tool_entries(server_key, &HashSet::new());
    }

    /// Keep the server's tools named in `listed` and drop its others, with
    /// the aliases that target a dropped tool, index and entry alike. An
    /// alias that targets a kept tool takes that tool's definition as it is
    /// now: its entry is a copy, and the listing may have changed the tool.
    /// A refresh or a reconnect installs its listing first and calls this
    /// after, so a reader never finds the server without the tools both
    /// listings hold.
    pub fn retain_server_tool_entries(&self, server_key: &str, listed: &HashSet<String>) {
        let dropped: Vec<QualifiedToolName> = self
            .tools_by_server
            .get_mut(server_key)
            .map(|mut names| {
                names
                    .extract_if(|name| !listed.contains(name.tool_name()))
                    .collect()
            })
            .unwrap_or_default();
        self.tools_by_server
            .remove_if(server_key, |_, names| names.is_empty());
        for qualified in dropped {
            self.remove_entry(&qualified);
        }

        let (kept, gone): (Vec<_>, Vec<_>) = self
            .aliases
            .iter()
            .filter(|entry| entry.value().server_key() == server_key)
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .partition(|(_, target)| listed.contains(target.tool_name()));
        self.drop_aliases(gone);
        // The entry under the alias key is a copy of the target's tool, taken
        // when the alias was registered: a kept alias takes the tool again,
        // unless another server took the alias's name over since the read
        // above.
        for (alias, target) in kept {
            let tool = self
                .tools_by_qualified
                .get(&target)
                .map(|entry| entry.tool.clone());
            if let Some(tool) = tool {
                self.update_entry(ALIAS_SERVER_KEY, &alias, |entry| {
                    if entry
                        .alias_target
                        .as_ref()
                        .is_some_and(|held| held.target == target)
                    {
                        entry.tool = tool;
                    }
                });
            }
        }
    }

    /// Drop the aliases a retain read as targeting a dropped tool, index and
    /// entry alike. An alias goes only while the index still holds the
    /// target that was read: a name another server took over since the read
    /// is the taker's, and stays.
    fn drop_aliases(&self, gone: Vec<(String, QualifiedToolName)>) {
        for (alias, target) in gone {
            let removed = self.aliases.remove_if(&alias, |_, held| *held == target);
            if removed.is_none() {
                continue;
            }
            let qualified = QualifiedToolName::new(ALIAS_SERVER_KEY, &alias);
            self.remove_entry(&qualified);
            if let Some(mut names) = self.tools_by_server.get_mut(ALIAS_SERVER_KEY) {
                names.remove(&qualified);
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
#[path = "../../tests/unit/inventory/index/tests.rs"]
mod tests;
