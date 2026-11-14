//! Tokenizer Registry for dynamic tokenizer loading
//!
//! Provides thread-safe, deduplicated tokenizer loading for IGW mode where
//! multiple routers (HTTP and gRPC) need to share tokenizers across workers.
//!
//! ## ID vs Name Lookup
//!
//! Tokenizers are stored with two keys:
//! - **ID (UUID)**: Unique identifier generated at registration, immutable
//! - **Name**: User-provided identifier, must be unique
//!
//! Lookup behavior:
//! - `get(key)`: Tries name first, then ID (backward compatible)
//! - `get_by_id(id)`: Exact ID match only
//! - `get_by_name(name)`: Exact name match only
//! - `remove(name)`: Removes by name
//! - `remove_by_id(id)`: Removes by ID
//! - `remove_if_id(name, id)`: Removes by name while the name maps to that ID

use std::sync::Arc;

use dashmap::DashMap;
use rustc_hash::FxBuildHasher;
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{debug, info};
use uuid::Uuid;

use crate::traits::Tokenizer;

/// Outcome of a tokenizer load operation
#[derive(Debug, Clone)]
pub enum LoadOutcome {
    /// Tokenizer was newly loaded and registered
    Loaded { id: String },
    /// Tokenizer already existed, returning existing ID
    AlreadyExists { id: String },
}

impl LoadOutcome {
    /// Get the ID regardless of outcome
    pub fn id(&self) -> &str {
        match self {
            LoadOutcome::Loaded { id } => id,
            LoadOutcome::AlreadyExists { id } => id,
        }
    }

    /// Returns true if the tokenizer was newly loaded
    pub fn is_newly_loaded(&self) -> bool {
        matches!(self, LoadOutcome::Loaded { .. })
    }
}

/// Error type for tokenizer loading operations
#[derive(Debug, Error)]
pub enum LoadError {
    /// Name cannot be empty
    #[error("tokenizer name cannot be empty")]
    EmptyName,
    /// Source cannot be empty
    #[error("tokenizer source cannot be empty")]
    EmptySource,
    /// Loading failed
    #[error("{0}")]
    LoadFailed(String),
}

/// Metadata and tokenizer instance for a registered tokenizer
#[derive(Clone)]
pub struct TokenizerEntry {
    /// Unique identifier (UUID)
    pub id: String,
    /// User-provided name
    pub name: String,
    /// Source path or HuggingFace model ID
    pub source: String,
    /// The tokenizer instance
    pub tokenizer: Arc<dyn Tokenizer>,
}

/// Registry for managing tokenizers keyed by UUID
///
/// Features:
/// - Thread-safe concurrent access using DashMap; the two lookup maps hash
///   with Fx, their keys being the operator's ids and names (a request's
///   string looks one up and never inserts one), read on every request
/// - Per-key locking to prevent duplicate loading
/// - Lookup by UUID (primary) or name (secondary index)
pub struct TokenizerRegistry {
    /// Storage for loaded tokenizers, keyed by UUID
    tokenizers: DashMap<String, TokenizerEntry, FxBuildHasher>,
    /// Secondary index: name -> UUID for lookup
    name_to_id: DashMap<String, String, FxBuildHasher>,
    /// Per-key locks to prevent duplicate loading
    loading_locks: DashMap<String, Arc<Mutex<()>>>,
}

/// RAII guard that removes the loading lock entry on drop.
/// Ensures cleanup happens on normal completion, early return, cancellation
/// or panic.
///
/// The entry stays while another load holds a reference to the lock (a
/// waiter, or a holder whose own reference is not yet gone): the waiter
/// takes the mutex next, a load that arrives meanwhile (a failed load's
/// retry) joins the same lock instead of making a second one, so two loads
/// of one name never run together and publish two entries under it, and
/// the last load out removes the entry. Declared before the mutex guard in
/// [`TokenizerRegistry::load`] and so dropped after it (and after the
/// pending lock future of a load cancelled while it waits), the guard
/// counts once this load's own reference is gone and removes the entry when
/// the map's reference is the last; `remove_if` and `entry` take the same
/// shard lock, so no clone slips in between the count and the removal.
struct LoadingLockGuard<'a> {
    locks: &'a DashMap<String, Arc<Mutex<()>>>,
    key: String,
}

impl Drop for LoadingLockGuard<'_> {
    fn drop(&mut self) {
        self.locks
            .remove_if(&self.key, |_, lock| Arc::strong_count(lock) == 1);
    }
}

impl TokenizerRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tokenizers: DashMap::with_hasher(FxBuildHasher),
            name_to_id: DashMap::with_hasher(FxBuildHasher),
            loading_locks: DashMap::new(),
        }
    }

    /// Generate a new UUID for a tokenizer
    pub fn generate_id() -> String {
        Uuid::now_v7().to_string()
    }

    /// Load and register a tokenizer
    ///
    /// Validates inputs, handles deduplication, and loads the tokenizer if needed.
    /// Per-key locking ensures only one load happens per name, preventing race conditions.
    ///
    /// # Arguments
    /// * `id` - Pre-generated UUID (use `generate_id()` to create one)
    /// * `name` - User-provided name (used for deduplication, must not be empty)
    /// * `source` - Source path or HuggingFace model ID (must not be empty)
    /// * `loader` - Async function that loads the tokenizer
    ///
    /// # Returns
    /// * `Ok(LoadOutcome::Loaded { id })` - Tokenizer was newly loaded
    /// * `Ok(LoadOutcome::AlreadyExists { id })` - Tokenizer already existed
    /// * `Err(LoadError)` - Validation failed or loading failed
    pub async fn load<F, Fut>(
        &self,
        id: &str,
        name: &str,
        source: &str,
        loader: F,
    ) -> Result<LoadOutcome, LoadError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<Arc<dyn Tokenizer>, String>>,
    {
        // Validate inputs
        if name.is_empty() {
            return Err(LoadError::EmptyName);
        }
        if source.is_empty() {
            return Err(LoadError::EmptySource);
        }

        // Fast path: already loaded by name
        if let Some(existing_id) = self.name_to_id.get(name) {
            debug!("Tokenizer already registered for name: {}", name);
            return Ok(LoadOutcome::AlreadyExists {
                id: existing_id.clone(),
            });
        }

        debug!("Tokenizer cache miss for name: {}", name);

        // Acquire per-name lock to prevent duplicate loading
        let lock = self
            .loading_locks
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();

        // Declared before the mutex guard and so dropped after it (and after
        // the pending lock future of a load cancelled while it waits): each
        // load counts once its own reference is gone, and the last one out
        // removes the entry.
        let _lock_cleanup = LoadingLockGuard {
            locks: &self.loading_locks,
            key: name.to_string(),
        };
        // The mutex guard owns this load's reference to the lock.
        let _mutex_guard = lock.lock_owned().await;

        // Double-check after acquiring lock (another thread may have loaded it)
        if let Some(existing_id) = self.name_to_id.get(name) {
            debug!("Tokenizer loaded by another thread for name: {}", name);
            return Ok(LoadOutcome::AlreadyExists {
                id: existing_id.clone(),
            });
        }

        // Load tokenizer
        info!("Loading tokenizer '{}' from source: {}", name, source);
        let result = loader().await;

        let tokenizer = result.map_err(LoadError::LoadFailed)?;

        // Create entry with provided ID
        let entry = TokenizerEntry {
            id: id.to_string(),
            name: name.to_string(),
            source: source.to_string(),
            tokenizer,
        };

        // Store in registry
        self.tokenizers.insert(id.to_string(), entry);
        self.name_to_id.insert(name.to_string(), id.to_string());

        info!(
            "Successfully registered tokenizer '{}' with id: {}",
            name, id
        );

        Ok(LoadOutcome::Loaded { id: id.to_string() })
    }

    /// Register a preloaded tokenizer with a pre-generated ID
    ///
    /// Atomically inserts a tokenizer into the registry only if no tokenizer
    /// with the same name exists. Returns the ID if successful.
    ///
    /// This is primarily used for testing. Production code should use `load()`.
    ///
    /// # Returns
    /// * `Some(id)` - If the tokenizer was successfully registered
    /// * `None` - If a tokenizer with this name already existed
    #[cfg(test)]
    pub fn register(
        &self,
        id: &str,
        name: &str,
        source: &str,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Option<String> {
        use dashmap::mapref::entry::Entry;

        // Check if name already exists
        match self.name_to_id.entry(name.to_string()) {
            Entry::Occupied(_) => {
                debug!(
                    "Tokenizer already exists for name: {}, skipping registration",
                    name
                );
                None
            }
            Entry::Vacant(name_entry) => {
                let entry = TokenizerEntry {
                    id: id.to_string(),
                    name: name.to_string(),
                    source: source.to_string(),
                    tokenizer,
                };

                info!("Registering tokenizer '{}' with id: {}", name, id);
                self.tokenizers.insert(id.to_string(), entry);
                name_entry.insert(id.to_string());
                Some(id.to_string())
            }
        }
    }

    /// Get a tokenizer by UUID
    pub fn get_by_id(&self, id: &str) -> Option<TokenizerEntry> {
        self.tokenizers.get(id).map(|e| e.clone())
    }

    /// Get a tokenizer by name
    pub fn get_by_name(&self, name: &str) -> Option<TokenizerEntry> {
        self.name_to_id
            .get(name)
            .and_then(|id| self.tokenizers.get(id.as_str()).map(|e| e.clone()))
    }

    /// Get a tokenizer (for backward compatibility, tries name first then ID)
    pub fn get(&self, name_or_id: &str) -> Option<Arc<dyn Tokenizer>> {
        let by_name = self.name_to_id.get(name_or_id).and_then(|id| {
            self.tokenizers
                .get(id.as_str())
                .map(|e| Arc::clone(&e.tokenizer))
        });
        by_name.or_else(|| {
            self.tokenizers
                .get(name_or_id)
                .map(|e| Arc::clone(&e.tokenizer))
        })
    }

    /// Check if a tokenizer is registered by name
    pub fn contains(&self, name: &str) -> bool {
        self.name_to_id.contains_key(name)
    }

    /// Check if a tokenizer is registered by ID
    pub fn contains_id(&self, id: &str) -> bool {
        self.tokenizers.contains_key(id)
    }

    /// Get the number of loaded tokenizers
    pub fn len(&self) -> usize {
        self.tokenizers.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tokenizers.is_empty()
    }

    /// List all registered tokenizers
    pub fn list(&self) -> Vec<TokenizerEntry> {
        let mut entries: Vec<TokenizerEntry> =
            self.tokenizers.iter().map(|e| e.value().clone()).collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }

    /// Remove a tokenizer by ID
    ///
    /// Through the name's publication, as [`Self::remove_if_id`], while the
    /// name maps to this ID: the name goes first, then the entry. An entry
    /// whose name is published under another ID is no name's (a removal by
    /// name between its two steps, or two loads of the name that once ran
    /// together) and goes by its ID alone, the name staying that other
    /// entry's: that publication is held read-locked through the removal,
    /// so a removal of the other entry by name or a put-back of this one
    /// under its ID waits on it and no name is left mapping to a removed
    /// entry. An entry `load` has stored and not yet published is not
    /// taken, so a name never outlives its entry. Returns the entry if it
    /// was taken.
    pub fn remove_by_id(&self, id: &str) -> Option<TokenizerEntry> {
        let name = self.tokenizers.get(id)?.name.clone();
        if let Some(entry) = self.remove_if_id(&name, id) {
            return Some(entry);
        }
        // The publication stays held (the name's shard read-locked) until
        // the removal below is done: the name first, then the entry, the
        // order `get` and `register` nest in too.
        let published = self.name_to_id.get(&name)?;
        if published.as_str() == id {
            return None;
        }
        self.tokenizers.remove(id).map(|(_, entry)| entry)
    }

    /// Remove the tokenizer under `name` when it is the entry with `id`
    ///
    /// For a removal that read the entry under its name and acts later: the
    /// name is taken first, and only while it maps to `id`, then the entry
    /// under `id`. An entry `load` has stored and not yet published (no name
    /// yet) is left alone, so is one that replaced the inspected entry under
    /// the name (another ID), and a name never outlives the entry it maps to.
    /// Returns the entry if it was taken.
    pub fn remove_if_id(&self, name: &str, id: &str) -> Option<TokenizerEntry> {
        self.name_to_id
            .remove_if(name, |_, published| published == id)?;
        self.tokenizers.remove(id).map(|(_, entry)| entry)
    }

    /// Remove a tokenizer by name
    ///
    /// Returns the entry if it was present.
    pub fn remove(&self, name: &str) -> Option<TokenizerEntry> {
        if let Some((_, id)) = self.name_to_id.remove(name) {
            self.tokenizers.remove(&id).map(|(_, e)| e)
        } else {
            None
        }
    }

    /// Clear all tokenizers from the registry
    pub fn clear(&self) {
        self.tokenizers.clear();
        self.name_to_id.clear();
        self.loading_locks.clear();
    }
}

impl Default for TokenizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tokio::spawn is fine in unit tests that await all handles"
)]
mod tests {
    use std::{
        future::Future,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        task::{Context, Waker},
        time::Duration,
    };

    use dashmap::try_result::TryResult;
    use tokio::{sync::oneshot, time::sleep};

    use crate::{mock::MockTokenizer, traits::Tokenizer, LoadError, TokenizerRegistry};

    #[tokio::test]
    async fn test_basic_operations() {
        let registry = TokenizerRegistry::new();

        // Registry starts empty
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(!registry.contains("model1"));

        // Load and register a tokenizer
        let id = TokenizerRegistry::generate_id();
        let outcome = registry
            .load(&id, "model1", "path/to/model", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();

        // Verify LoadOutcome::Loaded
        assert!(outcome.is_newly_loaded());
        assert_eq!(outcome.id(), id);

        // Verify it's loaded
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("model1"));
        assert!(registry.contains_id(&id));

        // Get returns the tokenizer
        let entry = registry.get_by_name("model1").unwrap();
        assert_eq!(entry.id, id);
        assert_eq!(entry.name, "model1");
        assert_eq!(entry.source, "path/to/model");

        // Remove works
        let removed = registry.remove_by_id(&id);
        assert!(removed.is_some());
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_load_returns_already_exists() {
        let registry = TokenizerRegistry::new();
        let id1 = TokenizerRegistry::generate_id();
        let id2 = TokenizerRegistry::generate_id();

        // First load should return Loaded
        let outcome1 = registry
            .load(&id1, "model1", "source1", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();
        assert!(outcome1.is_newly_loaded());
        assert_eq!(outcome1.id(), id1);

        // Second load with same name should return AlreadyExists with ORIGINAL id
        let outcome2 = registry
            .load(&id2, "model1", "source2", || async {
                panic!("Loader should not be called for duplicate name");
            })
            .await
            .unwrap();
        assert!(!outcome2.is_newly_loaded());
        assert_eq!(outcome2.id(), id1); // Returns the original ID, not id2

        // Registry still has only one entry
        assert_eq!(registry.len(), 1);

        // Original source is preserved
        let entry = registry.get_by_name("model1").unwrap();
        assert_eq!(entry.source, "source1");
    }

    #[tokio::test]
    async fn test_load_validation() {
        let registry = TokenizerRegistry::new();
        let id = TokenizerRegistry::generate_id();

        // Empty name should fail
        let result = registry
            .load(&id, "", "source", || async {
                panic!("Loader should not be called for invalid input");
            })
            .await;
        assert!(matches!(result, Err(LoadError::EmptyName)));

        // Empty source should fail
        let result = registry
            .load(&id, "model", "", || async {
                panic!("Loader should not be called for invalid input");
            })
            .await;
        assert!(matches!(result, Err(LoadError::EmptySource)));

        // Registry should be empty (nothing was loaded)
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_load_prevents_duplicate_loading() {
        let registry = Arc::new(TokenizerRegistry::new());
        let load_count = Arc::new(AtomicUsize::new(0));

        // Spawn multiple tasks trying to load the same tokenizer
        let mut handles = vec![];
        for i in 0..10 {
            let registry = registry.clone();
            let load_count = load_count.clone();
            let id = format!("id-{i}");
            let handle = tokio::spawn(async move {
                registry
                    .load(&id, "model1", "source", || async {
                        // Simulate slow loading
                        sleep(Duration::from_millis(10)).await;
                        load_count.fetch_add(1, Ordering::SeqCst);
                        Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
                    })
                    .await
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // Verify tokenizer was loaded only once
        assert_eq!(
            load_count.load(Ordering::SeqCst),
            1,
            "Tokenizer should be loaded exactly once despite concurrent requests"
        );
        assert_eq!(registry.len(), 1);
    }

    #[tokio::test]
    async fn test_multiple_models() {
        let registry = TokenizerRegistry::new();

        // Load multiple tokenizers
        for i in 1..=5 {
            let model_name = format!("model{i}");
            let id = TokenizerRegistry::generate_id();
            registry
                .load(&id, &model_name, "source", || async {
                    Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
                })
                .await
                .unwrap();
        }

        assert_eq!(registry.len(), 5);
        assert!(registry.contains("model1"));
        assert!(registry.contains("model5"));
        assert!(!registry.contains("model6"));

        // List returns all with metadata
        let entries = registry.list();
        assert_eq!(entries.len(), 5);
        assert!(entries.iter().any(|e| e.name == "model1"));

        // Clear all
        registry.clear();
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_load_failure() {
        let registry = TokenizerRegistry::new();
        let id = TokenizerRegistry::generate_id();

        // Try to load with a failing loader
        let result = registry
            .load(&id, "failing_model", "source", || async {
                Err("Load failed".to_string())
            })
            .await;

        assert!(result.is_err());
        assert!(!registry.contains("failing_model"));
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_get_by_name_and_id() {
        let registry = TokenizerRegistry::new();
        let id = TokenizerRegistry::generate_id();

        registry
            .load(&id, "my-model", "hf/model", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();

        // Get by name
        let by_name = registry.get_by_name("my-model");
        assert!(by_name.is_some());
        assert_eq!(by_name.as_ref().unwrap().id, id);

        // Get by ID
        let by_id = registry.get_by_id(&id);
        assert!(by_id.is_some());
        assert_eq!(by_id.as_ref().unwrap().name, "my-model");

        // Generic get works with both
        assert!(registry.get("my-model").is_some());
        assert!(registry.get(&id).is_some());
    }

    #[tokio::test]
    async fn test_register_only_if_absent() {
        let registry = TokenizerRegistry::new();
        let id1 = TokenizerRegistry::generate_id();
        let id2 = TokenizerRegistry::generate_id();
        let tokenizer1 = Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>;
        let tokenizer2 = Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>;

        // First registration should succeed
        let result1 = registry.register(&id1, "model1", "source1", tokenizer1.clone());
        assert!(result1.is_some());
        assert_eq!(registry.len(), 1);

        // Second registration with same name should fail
        let result2 = registry.register(&id2, "model1", "source2", tokenizer2.clone());
        assert!(result2.is_none());
        assert_eq!(registry.len(), 1);

        // Original tokenizer should still be there
        let entry = registry.get_by_name("model1").unwrap();
        assert_eq!(entry.id, id1);
        assert_eq!(entry.source, "source1");

        // Registration with different name should succeed
        let id3 = TokenizerRegistry::generate_id();
        let result3 = registry.register(&id3, "model2", "source2", tokenizer2);
        assert!(result3.is_some());
        assert_eq!(registry.len(), 2);
    }

    #[tokio::test]
    async fn test_loading_lock_cleanup_on_panic() {
        let registry = Arc::new(TokenizerRegistry::new());

        // Spawn a task that will panic during loading
        let registry_clone = registry.clone();
        let handle = tokio::spawn(async move {
            registry_clone
                .load(
                    &TokenizerRegistry::generate_id(),
                    "panic-model",
                    "source",
                    || async {
                        panic!("Simulated panic during tokenizer loading");
                    },
                )
                .await
        });

        // Wait for the task - it should panic
        let result = handle.await;
        assert!(result.is_err(), "Task should have panicked");

        // The RAII guard should have cleaned up the loading lock.
        // Verify by attempting another load with the same name - it should work,
        // not deadlock or fail due to stale lock.
        let id = TokenizerRegistry::generate_id();
        let outcome = registry
            .load(&id, "panic-model", "source", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await;

        // Should succeed - the lock was properly cleaned up
        assert!(outcome.is_ok(), "Load should succeed after panic cleanup");
        assert!(outcome.unwrap().is_newly_loaded());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("panic-model"));
    }

    #[tokio::test]
    async fn test_loading_lock_cleanup_on_early_return() {
        let registry = Arc::new(TokenizerRegistry::new());

        // Load a tokenizer
        let id1 = TokenizerRegistry::generate_id();
        registry
            .load(&id1, "model1", "source1", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();

        // Now simulate concurrent load attempts where one thread wins
        // and another thread sees "already exists" in the double-check.
        // The RAII guard should clean up the lock on early return.

        // First, verify loading_locks is empty after successful load
        // by checking that we can load a different model without issues
        let id2 = TokenizerRegistry::generate_id();
        let outcome = registry
            .load(&id2, "model2", "source2", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();

        assert!(outcome.is_newly_loaded());
        assert_eq!(registry.len(), 2);

        // Try to load model1 again - should return AlreadyExists
        // and the lock should be cleaned up (not leak)
        let id3 = TokenizerRegistry::generate_id();
        let outcome = registry
            .load(&id3, "model1", "source1", || async {
                panic!("Loader should not be called for existing model");
            })
            .await
            .unwrap();

        assert!(!outcome.is_newly_loaded());
        assert_eq!(outcome.id(), id1); // Returns original ID
    }

    /// The removal by name and ID together, for a removal that read the
    /// entry under its name and acts later: another ID under the name takes
    /// nothing; the ID read goes with its name, and nothing of it stays.
    #[tokio::test]
    async fn test_remove_if_id() {
        let registry = TokenizerRegistry::new();
        let id = TokenizerRegistry::generate_id();
        registry
            .load(&id, "model1", "source1", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();

        let other = TokenizerRegistry::generate_id();
        assert!(registry.remove_if_id("model1", &other).is_none());
        assert_eq!(registry.get_by_name("model1").unwrap().id, id);

        let removed = registry.remove_if_id("model1", &id).unwrap();
        assert_eq!(removed.id, id);
        assert!(!registry.contains("model1"));
        assert!(!registry.contains_id(&id));
        assert!(registry.is_empty());
        assert!(registry.remove_if_id("model1", &id).is_none());
    }

    /// `load` stores the entry, then publishes its name: a removal by ID
    /// between the two writes would take the stored entry and leave the
    /// name, once published, mapping to nothing. The removal by name and ID
    /// reads the publication and takes nothing until it is there.
    #[tokio::test]
    async fn test_remove_if_id_leaves_an_entry_not_yet_published() {
        let registry = TokenizerRegistry::new();
        let id = TokenizerRegistry::generate_id();
        registry
            .load(&id, "model1", "source1", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();
        // The registry as `load` leaves it between its two writes.
        let (_, published) = registry.name_to_id.remove("model1").unwrap();

        assert!(registry.remove_if_id("model1", &id).is_none());
        assert!(registry.contains_id(&id), "the stored entry is not taken");

        // The publish lands whole.
        registry.name_to_id.insert("model1".to_string(), published);
        assert_eq!(registry.get_by_name("model1").unwrap().id, id);
        assert_eq!(registry.remove_if_id("model1", &id).unwrap().id, id);
        assert!(registry.is_empty());
    }

    /// A removal by ID (the operator's DELETE, the removal job) of an entry
    /// whose name is published under another ID (a removal by name between
    /// its two steps, or two loads of the name that once ran together): the
    /// entry goes by its ID alone and the name keeps mapping to the
    /// replacement.
    #[tokio::test]
    async fn test_remove_by_id_leaves_a_replacements_name() {
        let registry = TokenizerRegistry::new();
        let old = TokenizerRegistry::generate_id();
        registry
            .load(&old, "model1", "source1", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();
        // The registry as a release leaves it between its two steps (the
        // name taken, the entry still stored) once a load has published a
        // replacement under the name.
        registry.name_to_id.remove("model1").unwrap();
        let replacement = TokenizerRegistry::generate_id();
        registry
            .load(&replacement, "model1", "source1", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();
        assert!(registry.contains_id(&old));

        // The removal by the old id takes the old entry alone: the name is
        // the replacement's and stays so.
        assert_eq!(registry.remove_by_id(&old).unwrap().id, old);
        assert!(!registry.contains_id(&old));
        assert_eq!(
            registry.get_by_name("model1").unwrap().id,
            replacement,
            "the replacement keeps its name"
        );
        assert_eq!(registry.remove_by_id(&replacement).unwrap().id, replacement);
        assert!(registry.is_empty());
    }

    /// `load` stores the entry, then publishes its name: a removal by ID
    /// between the two writes takes nothing (the publication is not there),
    /// so the publish lands whole and the next removal takes the entry.
    #[tokio::test]
    async fn test_remove_by_id_leaves_an_entry_not_yet_published() {
        let registry = TokenizerRegistry::new();
        let id = TokenizerRegistry::generate_id();
        registry
            .load(&id, "model1", "source1", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();
        // The registry as `load` leaves it between its two writes.
        let (_, published) = registry.name_to_id.remove("model1").unwrap();

        assert!(registry.remove_by_id(&id).is_none());
        assert!(registry.contains_id(&id), "the stored entry is not taken");

        // The publish lands whole, and the removal takes the entry then.
        registry.name_to_id.insert("model1".to_string(), published);
        assert_eq!(registry.remove_by_id(&id).unwrap().id, id);
        assert!(registry.is_empty());
    }

    /// Yields until the per-name lock under `name` has `clones` strong
    /// references (the map's counted): the count the guard reads on drop.
    async fn wait_for_lock_clones(registry: &TokenizerRegistry, name: &str, clones: usize) {
        for _ in 0..10_000 {
            let count = registry
                .loading_locks
                .get(name)
                .map(|lock| Arc::strong_count(lock.value()));
            if count == Some(clones) {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("the lock under {name} never reached {clones} references");
    }

    /// A load that fails while another waits on its per-name lock: the lock
    /// stays for the waiter, the failed load's retry joins it behind the
    /// waiter, and the name ends with one entry (the waiter's), the retry
    /// answering with it, and no lock. A guard that removed the lock with a
    /// waiter on it let the retry make a second lock and load beside the
    /// waiter: both published, and the loser's entry stayed stored under an
    /// ID its name no longer mapped to.
    #[tokio::test]
    async fn test_a_failed_loads_lock_stays_for_its_waiter() {
        let registry = Arc::new(TokenizerRegistry::new());

        // A holds the lock inside its loader until told to fail.
        let (a_started_tx, a_started_rx) = oneshot::channel();
        let (a_fail_tx, a_fail_rx) = oneshot::channel();
        let a = {
            let registry = registry.clone();
            tokio::spawn(async move {
                let id = TokenizerRegistry::generate_id();
                registry
                    .load(&id, "model1", "source1", || async move {
                        a_started_tx.send(()).unwrap();
                        a_fail_rx.await.unwrap();
                        Err::<Arc<dyn Tokenizer>, String>("the source is down".to_string())
                    })
                    .await
            })
        };
        a_started_rx.await.unwrap();

        // B waits on A's lock.
        let (b_started_tx, b_started_rx) = oneshot::channel();
        let (b_finish_tx, b_finish_rx) = oneshot::channel();
        let id_b = TokenizerRegistry::generate_id();
        let b = {
            let registry = registry.clone();
            let id_b = id_b.clone();
            tokio::spawn(async move {
                registry
                    .load(&id_b, "model1", "source1", || async move {
                        b_started_tx.send(()).unwrap();
                        b_finish_rx.await.unwrap();
                        Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
                    })
                    .await
            })
        };
        wait_for_lock_clones(&registry, "model1", 3).await;

        // A fails: the lock stays for B, which loads next.
        a_fail_tx.send(()).unwrap();
        assert!(matches!(a.await.unwrap(), Err(LoadError::LoadFailed(_))));
        b_started_rx.await.unwrap();
        assert!(
            registry.loading_locks.contains_key("model1"),
            "the lock stays for the waiter"
        );

        // C, A's retry, arrives while B loads: it joins B's lock and answers
        // with B's entry once B has published.
        let id_c = TokenizerRegistry::generate_id();
        let c = {
            let registry = registry.clone();
            let id_c = id_c.clone();
            tokio::spawn(async move {
                registry
                    .load(&id_c, "model1", "source1", || async {
                        panic!("the retry loads nothing: the waiter's load publishes first");
                    })
                    .await
            })
        };
        wait_for_lock_clones(&registry, "model1", 3).await;
        b_finish_tx.send(()).unwrap();
        assert!(b.await.unwrap().unwrap().is_newly_loaded());
        let outcome = c.await.unwrap().unwrap();
        assert!(!outcome.is_newly_loaded());
        assert_eq!(outcome.id(), id_b);

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get_by_name("model1").unwrap().id, id_b);
        assert!(!registry.contains_id(&id_c));
        assert!(
            registry.loading_locks.is_empty(),
            "the last load out removes the lock"
        );
    }

    /// A removal by ID of an entry whose name is published under another ID
    /// holds that publication read-locked through the removal: a removal of
    /// the other entry by name and a put-back of this one under its ID wait
    /// on it, so the entry removed is the one the publication was read
    /// beside and no name is left mapping to a removed entry. Pinned by the
    /// lock itself: with the old entry's shard held read, the removal parks
    /// at its write, and while it is parked a write to the name's
    /// publication is refused.
    #[tokio::test]
    async fn test_remove_by_id_holds_the_names_publication_through_the_removal() {
        let registry = Arc::new(TokenizerRegistry::new());
        let old = TokenizerRegistry::generate_id();
        registry
            .load(&old, "model1", "source1", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();
        // A release between its two steps, then a replacement published.
        registry.name_to_id.remove("model1").unwrap();
        let replacement = TokenizerRegistry::generate_id();
        registry
            .load(&replacement, "model1", "source1", || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();

        // The old entry's shard held read: the removal parks at its write.
        let parked = registry.tokenizers.get(&old).unwrap();
        let removal = {
            let registry = registry.clone();
            let old = old.clone();
            tokio::task::spawn_blocking(move || registry.remove_by_id(&old))
        };
        // While it is parked, the name's publication is held: a write to it
        // is refused, and stays refused (two reads a millisecond apart, so a
        // passing write lock of the name-first step is not mistaken for it).
        let mut refused = 0;
        for _ in 0..5_000 {
            if matches!(registry.name_to_id.try_get_mut("model1"), TryResult::Locked) {
                refused += 1;
                if refused == 2 {
                    break;
                }
            } else {
                refused = 0;
            }
            sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(
            refused, 2,
            "the name's publication is not held through the removal"
        );
        drop(parked);
        assert_eq!(removal.await.unwrap().unwrap().id, old);
        assert!(!registry.contains_id(&old));
        assert_eq!(registry.get_by_name("model1").unwrap().id, replacement);
    }

    /// A waiter cancelled before it takes the mutex (a registration step
    /// that times out) leaves no lock behind: its reference goes with its
    /// pending lock future, its guard counts after that, and the last load
    /// out, whichever it is, removes the entry.
    #[tokio::test]
    async fn test_a_waiter_cancelled_before_the_mutex_leaves_no_lock() {
        let registry = Arc::new(TokenizerRegistry::new());
        let (a_started_tx, a_started_rx) = oneshot::channel();
        let (a_finish_tx, a_finish_rx) = oneshot::channel();
        let a = {
            let registry = registry.clone();
            tokio::spawn(async move {
                let id = TokenizerRegistry::generate_id();
                registry
                    .load(&id, "model1", "source1", || async move {
                        a_started_tx.send(()).unwrap();
                        a_finish_rx.await.unwrap();
                        Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
                    })
                    .await
            })
        };
        a_started_rx.await.unwrap();

        // B, polled once by hand: it clones the lock and waits on the mutex.
        let id_b = TokenizerRegistry::generate_id();
        let mut b = Box::pin(registry.load(&id_b, "model1", "source1", || async {
            panic!("the cancelled waiter loads nothing");
        }));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(b.as_mut().poll(&mut cx).is_pending());
        wait_for_lock_clones(&registry, "model1", 3).await;

        // A publishes and leaves: the entry stays for the waiter.
        a_finish_tx.send(()).unwrap();
        assert!(a.await.unwrap().unwrap().is_newly_loaded());
        assert!(
            registry.loading_locks.contains_key("model1"),
            "the lock stays for the waiter"
        );

        // B is dropped before it is polled again: the last load out.
        drop(b);
        assert!(
            registry.loading_locks.is_empty(),
            "the cancelled waiter removes the entry"
        );
        assert_eq!(registry.len(), 1);
    }
}
