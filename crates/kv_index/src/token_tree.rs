//! Token-based radix tree for gRPC router with pre-tokenized input.
//!
//! This implementation uses token IDs (`u32`) instead of characters,
//! matching SGLang's Python scheduler which operates on token arrays.
//!
//! **Page-aligned design**: Following SGLang's radix cache, tokens are grouped
//! into pages (default 16 tokens). Only page-aligned prefixes are cached.
//! Sequences shorter than PAGE_SIZE get no cache benefit (matching engine behavior).
//!
//! Benefits:
//! - O(1) page-key comparisons vs O(n) single-token lookups
//! - Aligned with SGLang's internal KV cache page structure
//! - Reduced hash table overhead (1 lookup per PAGE_SIZE tokens)

use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    sync::{
        atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering},
        Arc, Weak,
    },
};

use dashmap::{mapref::entry::Entry, DashMap};
use once_cell::sync::Lazy;
use parking_lot::RwLock as ParkingLotRwLock;
use tracing::debug;

use super::{
    common::{MatchResult, TenantId, MATCHED_TENANTS_CAP},
    RadixTree,
};

/// Token ID type (matches SGLang's token representation)
pub type TokenId = u32;

/// Default page size for token grouping (matches SGLang's default radix cache page size).
/// Configure per tree via `TokenTree::with_config` to match the backend's
/// KV page size; affinity below one backend page is unusable by the engine.
pub const PAGE_SIZE: usize = 16;

/// Children map key: a 64-bit digest of a node edge's first page.
/// A digest (not the tokens) keeps the key size independent of page size;
/// collisions are caught by the token comparison every walk already does,
/// and the insert path treats them as a non-match.
pub type TokenPageKey = u64;

type NodeRef = Arc<Node>;

/// Shard counts for DashMaps to balance concurrency vs allocation overhead.
/// Root node has more shards due to higher contention.
const ROOT_SHARD_COUNT: usize = 32;
/// Child nodes typically have few entries, minimize shard overhead.
const NODE_SHARD_COUNT: usize = 4;

/// Eviction policy matching SGLang's scheduler options.
/// The gateway should use the same policy as the backend worker to stay in sync.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used - evict oldest by last access time (default)
    #[default]
    Lru,
    /// Least Frequently Used - evict by (hit_count, last_access_time)
    Lfu,
    /// First In First Out - evict oldest by creation time
    Fifo,
    /// Most Recently Used - evict newest by last access time
    Mru,
    /// First In Last Out (stack) - evict newest by creation time
    Filo,
    /// Priority-based - evict by (priority, last_access_time), lower priority first
    Priority,
}

/// Align token count to page boundary (truncate to nearest page).
/// Matches SGLang's: `page_aligned_len = len(key) // page_size * page_size`
#[inline]
fn align_to_page(len: usize, page_size: usize) -> usize {
    (len / page_size) * page_size
}

/// Digest of the first `page_size` tokens (FxHash-style multiplication mixing).
#[inline]
fn page_key_of(tokens: &[TokenId], page_size: usize) -> TokenPageKey {
    debug_assert!(tokens.len() >= page_size);
    let mut acc = 0u64;
    for &tok in &tokens[..page_size] {
        acc = acc
            .wrapping_add(tok as u64)
            .wrapping_mul(0x517cc1b727220a95);
    }
    acc
}

/// A fast hasher for token page keys.
/// Uses FxHash-style multiplication mixing for excellent distribution.
#[derive(Default)]
struct TokenPageHasher(u64);

impl Hasher for TokenPageHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    // Keys are u64 digests and hash via `write_u64`; fold any other input
    // through the same mix so the hasher stays total.
    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self
                .0
                .wrapping_add(b as u64)
                .wrapping_mul(0x517cc1b727220a95);
        }
    }

    // The key is already a mixed digest (`page_key_of`); pass it through.
    #[inline(always)]
    fn write_u64(&mut self, key: u64) {
        self.0 = key;
    }
}

type TokenPageHasherBuilder = BuildHasherDefault<TokenPageHasher>;

/// Create a children DashMap with page-key lookup
#[inline]
fn new_children_map() -> DashMap<TokenPageKey, NodeRef, TokenPageHasherBuilder> {
    DashMap::with_hasher_and_shard_amount(TokenPageHasherBuilder::default(), NODE_SHARD_COUNT)
}

/// Create a tenant access time DashMap
#[inline]
fn new_tenant_map() -> DashMap<TenantId, u64> {
    DashMap::with_shard_amount(NODE_SHARD_COUNT)
}

/// Result of a prefix match operation with token counts.
#[derive(Debug, Clone)]
pub struct PrefixMatchResult {
    /// The tenant that owns the matched prefix
    pub tenant: TenantId,
    /// Number of tokens matched
    pub matched_token_count: usize,
    /// Total number of tokens in the input
    pub input_token_count: usize,
    /// Tenants holding the deepest matched node (capped at
    /// [`MATCHED_TENANTS_CAP`]): every one of them has the matched prefix,
    /// so a router can pressure-select among them instead of being bound to
    /// the single cached `tenant`. Empty when nothing matched.
    pub matched_tenants: Vec<TenantId>,
}

impl MatchResult for PrefixMatchResult {
    fn tenant(&self) -> &TenantId {
        &self.tenant
    }

    fn matched_count(&self) -> usize {
        self.matched_token_count
    }

    fn input_count(&self) -> usize {
        self.input_token_count
    }
}

/// Global tenant string intern pool to avoid repeated allocations.
/// Uses DashMap for concurrent access with minimal contention.
static TENANT_INTERN_POOL: Lazy<DashMap<Arc<str>, ()>> = Lazy::new(DashMap::new);

/// Intern tenant ID to avoid repeated allocations.
/// Returns cached Arc<str> if tenant was seen before.
fn intern_tenant(tenant: &str) -> TenantId {
    // Fast path: check if already interned
    if let Some(entry) = TENANT_INTERN_POOL.get(tenant) {
        return Arc::clone(entry.key());
    }

    // Slow path: intern new tenant
    let interned: Arc<str> = Arc::from(tenant);
    TENANT_INTERN_POOL.insert(Arc::clone(&interned), ());
    interned
}

/// Global timestamp counter for LRU ordering
static GLOBAL_TIMESTAMP: AtomicU64 = AtomicU64::new(0);

fn next_timestamp() -> u64 {
    GLOBAL_TIMESTAMP.fetch_add(1, Ordering::Relaxed)
}

/// Node in the token-based radix tree.
/// Uses parking_lot RwLock for better performance (no poisoning, smaller size).
/// Parent pointers enable O(d) ancestor cleanup during eviction.
struct Node {
    /// Token sequence stored at this node (always page-aligned length, multiple of PAGE_SIZE)
    tokens: ParkingLotRwLock<Vec<TokenId>>,
    /// Children nodes keyed by first PAGE_SIZE tokens (page key)
    children: DashMap<TokenPageKey, NodeRef, TokenPageHasherBuilder>,
    /// Tenants that own this node with last access timestamps
    tenant_last_access_time: DashMap<TenantId, u64>,
    /// Cached last tenant for fast access (probabilistic update)
    last_tenant: ParkingLotRwLock<Option<TenantId>>,
    /// Parent node (Weak to avoid reference cycles). None for root.
    parent: ParkingLotRwLock<Weak<Node>>,
    /// This node's key in parent's children map (for O(1) removal)
    page_key: ParkingLotRwLock<Option<TokenPageKey>>,
    /// Hit count for LFU eviction policy (incremented on each access)
    hit_count: AtomicU64,
    /// Creation timestamp for FIFO/FILO eviction policies
    creation_time: u64,
    /// Priority for priority-based eviction (higher = less likely to evict)
    priority: AtomicI32,
}

impl Node {
    fn new(tokens: Vec<TokenId>) -> Self {
        Self {
            tokens: ParkingLotRwLock::new(tokens),
            children: new_children_map(),
            tenant_last_access_time: new_tenant_map(),
            last_tenant: ParkingLotRwLock::new(None),
            parent: ParkingLotRwLock::new(Weak::new()),
            page_key: ParkingLotRwLock::new(None),
            hit_count: AtomicU64::new(0),
            creation_time: next_timestamp(),
            priority: AtomicI32::new(0),
        }
    }

    fn new_root() -> Self {
        Self {
            tokens: ParkingLotRwLock::new(Vec::new()),
            children: DashMap::with_hasher_and_shard_amount(
                TokenPageHasherBuilder::default(),
                ROOT_SHARD_COUNT,
            ),
            tenant_last_access_time: DashMap::with_shard_amount(ROOT_SHARD_COUNT),
            last_tenant: ParkingLotRwLock::new(None),
            parent: ParkingLotRwLock::new(Weak::new()),
            page_key: ParkingLotRwLock::new(None),
            hit_count: AtomicU64::new(0),
            creation_time: 0,                   // Root is always oldest
            priority: AtomicI32::new(i32::MIN), // Root has minimum priority (matches SGLang)
        }
    }

    /// Set parent pointer and page key for this node
    fn set_parent(&self, parent: &NodeRef, key: TokenPageKey) {
        *self.parent.write() = Arc::downgrade(parent);
        *self.page_key.write() = Some(key);
    }

    /// Check if this node is empty (no tenants and no children)
    fn is_empty(&self) -> bool {
        self.tenant_last_access_time.is_empty() && self.children.is_empty()
    }

    /// Get any tenant that owns this node (for match results)
    fn get_any_tenant(&self) -> Option<TenantId> {
        // Fast path: check cached tenant (parking_lot has no poisoning)
        let guard = self.last_tenant.read();
        if let Some(ref tenant) = *guard {
            // Use borrowed lookup to avoid Arc clone for validation
            if self.tenant_last_access_time.contains_key(tenant.as_ref()) {
                return Some(Arc::clone(tenant));
            }
        }
        drop(guard);

        // Slow path: iterate to find any tenant
        self.tenant_last_access_time
            .iter()
            .next()
            .map(|entry| Arc::clone(entry.key()))
    }

    /// Up to [`MATCHED_TENANTS_CAP`] tenants of this node, in map order.
    fn matched_tenants(&self) -> Vec<TenantId> {
        self.tenant_last_access_time
            .iter()
            .take(MATCHED_TENANTS_CAP)
            .map(|entry| Arc::clone(entry.key()))
            .collect()
    }

    /// Update tenant access and cache (with probabilistic update to reduce contention).
    ///
    /// # Arguments
    /// * `tenant` - The tenant to touch
    /// * `track_lfu` - If true, increment hit_count for LFU policy (only needed when eviction_policy == LFU)
    /// Returns true when this call newly attached the tenant to the node —
    /// the atomic winner under concurrency, so token accounting tied to it is
    /// exact.
    fn touch_tenant(&self, tenant: &TenantId, track_lfu: bool) -> bool {
        let ts = next_timestamp();

        // Conditionally increment hit count (only for LFU policy to reduce contention)
        if track_lfu {
            self.hit_count.fetch_add(1, Ordering::Relaxed);
        }

        // Fast path: try to update existing entry without Arc clone
        // DashMap supports Borrow<str> lookups, avoiding allocation
        let newly_attached =
            if let Some(mut entry) = self.tenant_last_access_time.get_mut(tenant.as_ref()) {
                *entry = ts;
                false
            } else {
                // Slow path: insert new entry (requires Arc clone)
                self.tenant_last_access_time
                    .insert(Arc::clone(tenant), ts)
                    .is_none()
            };

        // Probabilistic cache update (1/16 chance) to reduce write contention
        if ts & 0xF == 0 {
            if let Some(mut guard) = self.last_tenant.try_write() {
                *guard = Some(Arc::clone(tenant));
            }
        }

        newly_attached
    }
}

/// Token-based radix tree for cache-aware routing.
pub struct TokenTree {
    root: NodeRef,
    /// Track total tokens per tenant for eviction decisions
    tenant_token_count: DashMap<TenantId, usize>,
    /// Tree-wide token total (sum of `tenant_token_count`); the budget
    /// checked by `evict_tenant_by_size`
    total_token_count: AtomicUsize,
    /// Eviction policy (should match the backend worker's policy)
    eviction_policy: EvictionPolicy,
    /// Page size for token grouping (should match the backend worker's page size)
    page_size: usize,
}

impl std::fmt::Debug for TokenTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenTree")
            .field("tenant_count", &self.tenant_token_count.len())
            .field(
                "total_tokens",
                &self.total_token_count.load(Ordering::Relaxed),
            )
            .field("eviction_policy", &self.eviction_policy)
            .field("page_size", &self.page_size)
            .finish_non_exhaustive()
    }
}

impl Default for TokenTree {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenTree {
    /// Create a new TokenTree with default settings (page_size=16, eviction_policy=LRU).
    pub fn new() -> Self {
        Self::with_config(PAGE_SIZE, EvictionPolicy::default())
    }

    /// Create a new TokenTree with a specific eviction policy (uses default page_size=16).
    /// The policy should match the backend worker's policy to stay in sync.
    pub fn with_policy(policy: EvictionPolicy) -> Self {
        Self::with_config(PAGE_SIZE, policy)
    }

    /// Create a new TokenTree with specific page size and eviction policy.
    ///
    /// # Arguments
    /// * `page_size` - Token page size for grouping (must match backend worker's page size).
    ///   SGLang supports: 1, 16, 32, 64, 128 depending on attention backend.
    ///   **Note**: Currently only page_size=16 is supported at compile time.
    /// * `policy` - Eviction policy (should match the backend worker's policy).
    ///
    /// # Panics
    /// Panics if `page_size` != 16 (compile-time limitation, future versions may support other sizes).
    pub fn with_config(page_size: usize, policy: EvictionPolicy) -> Self {
        assert!(page_size >= 1, "page_size must be at least 1");
        Self {
            root: Arc::new(Node::new_root()),
            tenant_token_count: DashMap::with_shard_amount(ROOT_SHARD_COUNT),
            total_token_count: AtomicUsize::new(0),
            eviction_policy: policy,
            page_size,
        }
    }

    /// Get the current eviction policy.
    pub fn eviction_policy(&self) -> EvictionPolicy {
        self.eviction_policy
    }

    /// Get the page size used for token grouping.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Insert a token sequence with associated tenant.
    ///
    /// **Page-aligned**: Input is aligned to PAGE_SIZE boundary.
    /// Sequences shorter than PAGE_SIZE are skipped (no cache benefit).
    pub fn insert_tokens(&self, tokens: &[TokenId], tenant: &str) {
        let page_size = self.page_size;
        // Align to page boundary (truncate to nearest page)
        let aligned_len = align_to_page(tokens.len(), page_size);
        if aligned_len == 0 {
            // Sequence too short for cache benefit (matches SGLang behavior)
            return;
        }
        let tokens = &tokens[..aligned_len];

        let tenant_id = intern_tenant(tenant);

        // Ensure tenant exists at root
        self.root
            .tenant_last_access_time
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

        self.tenant_token_count
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

        // Compute once: only track hit counts for LFU policy
        let track_lfu = self.eviction_policy == EvictionPolicy::Lfu;

        // Descend from the root inserting the whole sequence.
        let tokens_added = Self::insert_from(
            Arc::clone(&self.root),
            tokens,
            Arc::clone(&tenant_id),
            track_lfu,
            page_size,
        );

        // Update tenant token count and tree-wide total
        self.add_tenant_tokens(tenant_id, tokens_added);
    }

    /// Insert `remaining` for `tenant_id` starting the descent at `current`
    /// (treated as the parent of the first edge), returning the number of new
    /// tokens to add to the tenant's count. Holds the exact node-split /
    /// tenant-attach / counting logic of [`Self::insert_tokens`]; the caller
    /// owns root bookkeeping and folding the returned count into
    /// `tenant_token_count`.
    ///
    /// [`Self::match_and_insert_with`] reuses this to splice only the *unmatched
    /// suffix* at the fall-off node (after re-attaching the tenant to the matched
    /// ancestor nodes), so the already-matched prefix is never re-walked.
    fn insert_from(
        mut current: NodeRef,
        mut remaining: &[TokenId],
        tenant_id: TenantId,
        track_lfu: bool,
        page_size: usize,
    ) -> usize {
        let mut tokens_added = 0usize;

        // Result type to carry state out of the match block
        // This allows the entry guard to be dropped before we update current
        enum InsertStep {
            Done(usize),
            Continue {
                next: NodeRef,
                advance: usize,
                counted: usize,
            },
        }

        while remaining.len() >= page_size {
            // Use first PAGE_SIZE tokens as key for children lookup
            let page_key = page_key_of(remaining, page_size);

            let step = match current.children.entry(page_key) {
                Entry::Vacant(entry) => {
                    // No child with this page key - create new node
                    let new_node = Arc::new(Node::new(remaining.to_vec()));
                    new_node.set_parent(&current, page_key);
                    new_node.touch_tenant(&tenant_id, track_lfu);
                    entry.insert(new_node);
                    InsertStep::Done(remaining.len())
                }
                Entry::Occupied(mut entry) => {
                    let child = Arc::clone(entry.get());
                    let child_tokens = child.tokens.read();
                    let child_len = child_tokens.len();

                    // Find common prefix length (page-aligned)
                    let common_len = remaining
                        .iter()
                        .zip(child_tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    // Align common length to page boundary
                    let common_len = align_to_page(common_len, page_size);

                    if common_len == 0 {
                        // No page-aligned match despite same page key (shouldn't happen)
                        drop(child_tokens);
                        InsertStep::Done(0)
                    } else if common_len == child_len {
                        // Full match with child - continue traversal. Count
                        // the tokens only on first attach (atomic via
                        // touch_tenant), or repeat traffic inflates
                        // `tenant_token_count`.
                        drop(child_tokens);
                        let newly_attached = child.touch_tenant(&tenant_id, track_lfu);
                        InsertStep::Continue {
                            next: child,
                            advance: common_len,
                            counted: if newly_attached { common_len } else { 0 },
                        }
                    } else if common_len >= remaining.len() {
                        // Input is prefix of child - split child at page boundary
                        // Strategy: Create NEW intermediate node with prefix tokens,
                        // keep original child as suffix (preserving its children/tenants)
                        let common_len = align_to_page(remaining.len(), page_size);
                        let prefix_tokens: Vec<TokenId> = child_tokens[..common_len].to_vec();
                        let suffix_page_key = page_key_of(&child_tokens[common_len..], page_size);
                        drop(child_tokens);

                        // Truncate the child to its suffix, clone its tenant
                        // map, and re-parent it under the intermediate in one
                        // `tokens` write section: lock-free walkers read the
                        // length and validate the parent pointer under a
                        // `tokens` read guard, so they observe the pre-split
                        // node or the post-split one — never a truncated text
                        // still reachable through the stale edge (which would
                        // silently shift their depth on self-similar content).
                        // Replayers likewise credit the live length under the
                        // read guard, so the clone inherits exactly the
                        // tenants credited for the pre-split span.
                        let mut child_tokens_write = child.tokens.write();
                        let tenant_map = child.tenant_last_access_time.clone();
                        let last_tenant = child.last_tenant.read().clone();
                        let suffix_tokens: Vec<TokenId> = child_tokens_write[common_len..].to_vec();
                        *child_tokens_write = suffix_tokens;

                        // Create intermediate node with prefix
                        // Intermediate inherits metadata from child (represents same prefix)
                        let intermediate_node = Arc::new(Node {
                            tokens: ParkingLotRwLock::new(prefix_tokens),
                            children: new_children_map(),
                            tenant_last_access_time: tenant_map,
                            last_tenant: ParkingLotRwLock::new(last_tenant),
                            parent: ParkingLotRwLock::new(Arc::downgrade(&current)),
                            page_key: ParkingLotRwLock::new(Some(page_key)),
                            hit_count: AtomicU64::new(child.hit_count.load(Ordering::Relaxed)),
                            creation_time: child.creation_time,
                            priority: AtomicI32::new(child.priority.load(Ordering::Relaxed)),
                        });

                        // Update child's parent to point to intermediate
                        child.set_parent(&intermediate_node, suffix_page_key);
                        drop(child_tokens_write);

                        // Add original child (now suffix) as child of intermediate
                        intermediate_node
                            .children
                            .insert(suffix_page_key, Arc::clone(&child));

                        // Attach before publication: the node is unreachable, so
                        // the return is authoritative and the prefix is credited
                        // exactly once even against a concurrent same-tenant
                        // replay that raced the clone above.
                        let newly_attached = intermediate_node.touch_tenant(&tenant_id, track_lfu);

                        // Replace entry with intermediate node
                        entry.insert(intermediate_node);

                        let new_tokens = if newly_attached { common_len } else { 0 };
                        InsertStep::Done(new_tokens)
                    } else {
                        // Partial match - need to split and add new branch at page boundary
                        // Strategy: Create NEW intermediate node with common prefix,
                        // keep original child as one suffix, create new node for other suffix
                        let prefix_tokens: Vec<TokenId> = child_tokens[..common_len].to_vec();
                        let child_suffix_page_key =
                            page_key_of(&child_tokens[common_len..], page_size);
                        drop(child_tokens);

                        // Truncate + tenant-map clone + re-parent in one
                        // `tokens` write section (see the prefix-split arm
                        // above).
                        let mut child_tokens_write = child.tokens.write();
                        let tenant_map = child.tenant_last_access_time.clone();
                        let last_tenant = child.last_tenant.read().clone();
                        let child_suffix: Vec<TokenId> = child_tokens_write[common_len..].to_vec();
                        *child_tokens_write = child_suffix;

                        // Create intermediate node with common prefix
                        // Intermediate inherits metadata from child (represents same prefix)
                        let intermediate_node = Arc::new(Node {
                            tokens: ParkingLotRwLock::new(prefix_tokens),
                            children: new_children_map(),
                            tenant_last_access_time: tenant_map,
                            last_tenant: ParkingLotRwLock::new(last_tenant),
                            parent: ParkingLotRwLock::new(Arc::downgrade(&current)),
                            page_key: ParkingLotRwLock::new(Some(page_key)),
                            hit_count: AtomicU64::new(child.hit_count.load(Ordering::Relaxed)),
                            creation_time: child.creation_time,
                            priority: AtomicI32::new(child.priority.load(Ordering::Relaxed)),
                        });

                        // Update child's parent to point to intermediate
                        child.set_parent(&intermediate_node, child_suffix_page_key);
                        drop(child_tokens_write);

                        // Add original child (now suffix) as child of intermediate
                        intermediate_node
                            .children
                            .insert(child_suffix_page_key, Arc::clone(&child));

                        // Create new node for the remaining input suffix
                        let new_remaining = &remaining[common_len..];
                        let new_branch_tokens = if new_remaining.len() >= page_size {
                            let new_node = Arc::new(Node::new(new_remaining.to_vec()));
                            let new_page_key = page_key_of(new_remaining, page_size);
                            new_node.set_parent(&intermediate_node, new_page_key);
                            new_node.touch_tenant(&tenant_id, track_lfu);
                            intermediate_node.children.insert(new_page_key, new_node);
                            new_remaining.len()
                        } else {
                            0
                        };

                        // Attach before publication (see the prefix-split arm).
                        let newly_attached = intermediate_node.touch_tenant(&tenant_id, track_lfu);

                        // Replace entry with intermediate node
                        entry.insert(intermediate_node);

                        let common_tokens = if newly_attached { common_len } else { 0 };
                        InsertStep::Done(new_branch_tokens + common_tokens)
                    }
                }
            };

            match step {
                InsertStep::Done(added) => {
                    tokens_added += added;
                    break;
                }
                InsertStep::Continue {
                    next,
                    advance,
                    counted,
                } => {
                    tokens_added += counted;
                    remaining = &remaining[advance..];
                    current = next;
                }
            }
        }

        // The caller folds this into `tenant_token_count` (once, after any
        // matched-prefix re-attach in match_and_insert_with).
        tokens_added
    }

    /// Find longest matching prefix with detailed counts.
    ///
    /// **Page-aligned**: Input is aligned to PAGE_SIZE boundary before lookup.
    /// Sequences shorter than PAGE_SIZE return 0 matched tokens.
    pub fn match_prefix_with_counts(&self, tokens: &[TokenId]) -> PrefixMatchResult {
        let page_size = self.page_size;
        let input_token_count = tokens.len();

        // Align to page boundary (truncate to nearest page)
        let aligned_len = align_to_page(tokens.len(), page_size);
        if aligned_len == 0 {
            // Sequence too short for cache lookup (matches SGLang behavior)
            return PrefixMatchResult {
                tenant: self
                    .root
                    .get_any_tenant()
                    .unwrap_or_else(|| intern_tenant("empty")),
                matched_token_count: 0,
                input_token_count,
                matched_tenants: Vec::new(),
            };
        }
        let tokens = &tokens[..aligned_len];

        let mut matched_tokens = 0;
        let mut last_tenant: Option<TenantId> = None;
        let mut last_match_node: Option<NodeRef> = None;
        let mut remaining = tokens;
        let mut current = Arc::clone(&self.root);

        // Compute once: only track hit counts for LFU policy
        let track_lfu = self.eviction_policy == EvictionPolicy::Lfu;

        enum MatchStep {
            Done,
            Retry,
            Continue {
                next: NodeRef,
                advance: usize,
                tenant: Option<TenantId>,
            },
            PartialMatch {
                matched: usize,
                tenant: Option<TenantId>,
                node: NodeRef,
            },
        }

        while remaining.len() >= page_size {
            // Use first PAGE_SIZE tokens as key for children lookup
            let page_key = page_key_of(remaining, page_size);

            let step = match current.children.get(&page_key) {
                None => MatchStep::Done,
                Some(child_ref) => 'probe: {
                    let child = Arc::clone(child_ref.value());
                    drop(child_ref);

                    let child_tokens = child.tokens.read();
                    // Stale-edge check: a split re-parents the child inside
                    // the same `tokens` write section that truncates it, so a
                    // parent no longer equal to `current` means this text may
                    // start deeper than this edge (on self-similar content the
                    // suffix would still compare equal and silently shift the
                    // walk). Re-probe the edge until the split publishes.
                    if !std::ptr::eq(child.parent.read().as_ptr(), Arc::as_ptr(&current)) {
                        drop(child_tokens);
                        std::hint::spin_loop();
                        break 'probe MatchStep::Retry;
                    }

                    // Count matching tokens
                    let match_len = remaining
                        .iter()
                        .zip(child_tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    // Align match length to page boundary
                    let match_len = align_to_page(match_len, page_size);

                    if match_len == 0 {
                        MatchStep::Done
                    } else {
                        let tenant = child.get_any_tenant();

                        // If node has no tenants (all evicted), stop matching here
                        // This ensures we only route to workers that still have the prefix cached
                        if tenant.is_none() {
                            MatchStep::Done
                        } else {
                            // Update timestamp on match to keep LRU in sync with backend
                            // SGLang does: child.last_access_time = access_time
                            if let Some(ref t) = tenant {
                                child.touch_tenant(t, track_lfu);
                            }

                            if match_len < child_tokens.len() {
                                // Partial match within node (at page boundary)
                                drop(child_tokens);
                                MatchStep::PartialMatch {
                                    matched: match_len,
                                    tenant,
                                    node: child,
                                }
                            } else {
                                // Full match - continue
                                drop(child_tokens);
                                MatchStep::Continue {
                                    next: child,
                                    advance: match_len,
                                    tenant,
                                }
                            }
                        }
                    }
                }
            };

            match step {
                MatchStep::Done => break,
                MatchStep::Retry => continue,
                MatchStep::PartialMatch {
                    matched,
                    tenant,
                    node,
                } => {
                    matched_tokens += matched;
                    if let Some(t) = tenant {
                        last_tenant = Some(t);
                        last_match_node = Some(node);
                    }
                    break;
                }
                MatchStep::Continue {
                    next,
                    advance,
                    tenant,
                } => {
                    matched_tokens += advance;
                    if let Some(t) = tenant {
                        last_tenant = Some(t);
                        last_match_node = Some(Arc::clone(&next));
                    }
                    remaining = &remaining[advance..];
                    current = next;
                }
            }
        }

        PrefixMatchResult {
            tenant: last_tenant.unwrap_or_else(|| intern_tenant("empty")),
            matched_token_count: matched_tokens,
            input_token_count,
            matched_tenants: last_match_node
                .map(|node| node.matched_tenants())
                .unwrap_or_default(),
        }
    }

    /// Combined match + insert in a single tree descent.
    ///
    /// Equivalent to [`Self::match_prefix_with_counts`] followed by
    /// [`Self::insert_tokens`] for the same `tokens` (single-threaded), but it
    /// traverses the prefix only once. The match result is frozen at match's stop
    /// condition before insert mutates deeper nodes, so the routed tenant and
    /// matched count are resolved against the pre-insert tree.
    pub fn match_and_insert(&self, tokens: &[TokenId], tenant: &str) -> PrefixMatchResult {
        let page_size = self.page_size;
        let input_token_count = tokens.len();

        // Align to page boundary (truncate to nearest page). Mirrors both
        // `match_prefix_with_counts` and `insert_tokens`.
        let aligned_len = align_to_page(tokens.len(), page_size);
        if aligned_len == 0 {
            // Too short to cache: `insert_tokens` is a no-op and
            // `match_prefix_with_counts` returns 0 matched tokens with any
            // root tenant. Reproduce the match result, skip the insert.
            return PrefixMatchResult {
                tenant: self
                    .root
                    .get_any_tenant()
                    .unwrap_or_else(|| intern_tenant("empty")),
                matched_token_count: 0,
                input_token_count,
                matched_tenants: Vec::new(),
            };
        }
        let tokens = &tokens[..aligned_len];

        let tenant_id = intern_tenant(tenant);

        // Ensure tenant exists at root (insert-side bookkeeping).
        self.root
            .tenant_last_access_time
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);
        self.tenant_token_count
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

        let mut remaining = tokens;
        let mut current = Arc::clone(&self.root);
        let mut tokens_added = 0usize;

        // Match-result accumulators. The tenant set is snapshotted at each
        // match site (pre-insert node state): the insert side touches the
        // node right after, and a post-walk read would see the inserting
        // tenant that match-then-insert could not have.
        let mut matched_tokens = 0usize;
        let mut last_tenant: Option<TenantId> = None;
        let mut matched_tenants: Vec<TenantId> = Vec::new();
        // Once the match descent would have stopped (empty node or partial
        // match), stop updating the match result; insert keeps descending.
        let mut match_frozen = false;

        let track_lfu = self.eviction_policy == EvictionPolicy::Lfu;

        // Carries state out of the entry match so the entry guard can be
        // dropped before we advance `current` (same pattern as `insert_tokens`).
        enum Step {
            Done(usize),
            Continue {
                next: NodeRef,
                advance: usize,
                counted: usize,
            },
        }

        while remaining.len() >= page_size {
            let page_key = page_key_of(remaining, page_size);

            let step = match current.children.entry(page_key) {
                Entry::Vacant(entry) => {
                    // Match: child not found -> match stops (records nothing).
                    // Insert: create a new leaf node holding the remainder.
                    let new_node = Arc::new(Node::new(remaining.to_vec()));
                    new_node.set_parent(&current, page_key);
                    new_node.touch_tenant(&tenant_id, track_lfu);
                    entry.insert(new_node);
                    Step::Done(remaining.len())
                }
                Entry::Occupied(mut entry) => {
                    let child = Arc::clone(entry.get());
                    let child_tokens = child.tokens.read();
                    let child_len = child_tokens.len();

                    let common_len = remaining
                        .iter()
                        .zip(child_tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    let common_len = align_to_page(common_len, page_size);

                    if common_len == 0 {
                        // Same page key but no aligned match (shouldn't happen).
                        // Match stops; insert adds nothing.
                        drop(child_tokens);
                        Step::Done(0)
                    } else if common_len == child_len {
                        // Full match with child -> continue traversal.
                        drop(child_tokens);

                        // --- Match side (pre-insert node state) ---
                        // Read the routed tenant BEFORE insert touches the node:
                        // insert's touch would add `tenant_id` to the map and
                        // corrupt both the all-evicted check and the routed
                        // tenant. This reproduces `match_prefix_with_counts`'s
                        // Continue arm exactly (it touches `get_any_tenant()`).
                        if !match_frozen {
                            match child.get_any_tenant() {
                                None => {
                                    // All tenants evicted: match stops here.
                                    // Insert still continues (re-populates node).
                                    match_frozen = true;
                                }
                                Some(t_match) => {
                                    matched_tokens += common_len;
                                    matched_tenants = child.matched_tenants();
                                    child.touch_tenant(&t_match, track_lfu);
                                    last_tenant = Some(t_match);
                                }
                            }
                        }

                        // --- Insert side ---
                        // `insert_tokens` always touches the inserting tenant on
                        // a full-match continuation. When the match side above
                        // already touched this same tenant, the original code
                        // ALSO touched twice (match then insert), so we keep both
                        // touches to preserve LFU hit_count / timestamp behavior
                        // byte-for-byte.
                        let newly_attached = child.touch_tenant(&tenant_id, track_lfu);
                        Step::Continue {
                            next: child,
                            advance: common_len,
                            counted: if newly_attached { common_len } else { 0 },
                        }
                    } else if common_len >= remaining.len() {
                        // Input is a prefix of the child -> split child at the
                        // page boundary. (Verbatim from `insert_tokens`, with the
                        // match-side touch interleaved before the split so the
                        // intermediate's tenant-map clone inherits it — exactly
                        // as match-then-insert ordered the writes.)

                        // Match side first (touches the pre-split child).
                        if !match_frozen {
                            match child.get_any_tenant() {
                                None => match_frozen = true,
                                Some(t_match) => {
                                    matched_tokens += common_len;
                                    matched_tenants = child.matched_tenants();
                                    child.touch_tenant(&t_match, track_lfu);
                                    last_tenant = Some(t_match);
                                }
                            }
                        }

                        let common_len = align_to_page(remaining.len(), page_size);
                        let prefix_tokens: Vec<TokenId> = child_tokens[..common_len].to_vec();
                        let suffix_page_key = page_key_of(&child_tokens[common_len..], page_size);
                        drop(child_tokens);

                        // Truncate + tenant-map clone + re-parent in one
                        // `tokens` write section (see `insert_from`'s
                        // prefix-split arm).
                        let mut child_tokens_write = child.tokens.write();
                        let tenant_map = child.tenant_last_access_time.clone();
                        let last_tenant = child.last_tenant.read().clone();
                        let suffix_tokens: Vec<TokenId> = child_tokens_write[common_len..].to_vec();
                        *child_tokens_write = suffix_tokens;

                        let intermediate_node = Arc::new(Node {
                            tokens: ParkingLotRwLock::new(prefix_tokens),
                            children: new_children_map(),
                            tenant_last_access_time: tenant_map,
                            last_tenant: ParkingLotRwLock::new(last_tenant),
                            parent: ParkingLotRwLock::new(Arc::downgrade(&current)),
                            page_key: ParkingLotRwLock::new(Some(page_key)),
                            hit_count: AtomicU64::new(child.hit_count.load(Ordering::Relaxed)),
                            creation_time: child.creation_time,
                            priority: AtomicI32::new(child.priority.load(Ordering::Relaxed)),
                        });

                        child.set_parent(&intermediate_node, suffix_page_key);
                        drop(child_tokens_write);

                        intermediate_node
                            .children
                            .insert(suffix_page_key, Arc::clone(&child));

                        // Attach before publication (see `insert_from`).
                        let newly_attached = intermediate_node.touch_tenant(&tenant_id, track_lfu);

                        entry.insert(intermediate_node);

                        let new_tokens = if newly_attached { common_len } else { 0 };
                        Step::Done(new_tokens)
                    } else {
                        // Partial match -> split and add a new branch at the page
                        // boundary. (Verbatim from `insert_tokens`, with the
                        // match-side touch interleaved before the split.)

                        // Match side first (touches the pre-split child).
                        if !match_frozen {
                            match child.get_any_tenant() {
                                None => match_frozen = true,
                                Some(t_match) => {
                                    matched_tokens += common_len;
                                    matched_tenants = child.matched_tenants();
                                    child.touch_tenant(&t_match, track_lfu);
                                    last_tenant = Some(t_match);
                                }
                            }
                        }

                        let prefix_tokens: Vec<TokenId> = child_tokens[..common_len].to_vec();
                        let child_suffix_page_key =
                            page_key_of(&child_tokens[common_len..], page_size);
                        drop(child_tokens);

                        // Truncate + tenant-map clone + re-parent in one
                        // `tokens` write section (see `insert_from`'s
                        // prefix-split arm).
                        let mut child_tokens_write = child.tokens.write();
                        let tenant_map = child.tenant_last_access_time.clone();
                        let last_tenant = child.last_tenant.read().clone();
                        let child_suffix: Vec<TokenId> = child_tokens_write[common_len..].to_vec();
                        *child_tokens_write = child_suffix;

                        let intermediate_node = Arc::new(Node {
                            tokens: ParkingLotRwLock::new(prefix_tokens),
                            children: new_children_map(),
                            tenant_last_access_time: tenant_map,
                            last_tenant: ParkingLotRwLock::new(last_tenant),
                            parent: ParkingLotRwLock::new(Arc::downgrade(&current)),
                            page_key: ParkingLotRwLock::new(Some(page_key)),
                            hit_count: AtomicU64::new(child.hit_count.load(Ordering::Relaxed)),
                            creation_time: child.creation_time,
                            priority: AtomicI32::new(child.priority.load(Ordering::Relaxed)),
                        });

                        child.set_parent(&intermediate_node, child_suffix_page_key);
                        drop(child_tokens_write);

                        intermediate_node
                            .children
                            .insert(child_suffix_page_key, Arc::clone(&child));

                        let new_remaining = &remaining[common_len..];
                        let new_branch_tokens = if new_remaining.len() >= page_size {
                            let new_node = Arc::new(Node::new(new_remaining.to_vec()));
                            let new_page_key = page_key_of(new_remaining, page_size);
                            new_node.set_parent(&intermediate_node, new_page_key);
                            new_node.touch_tenant(&tenant_id, track_lfu);
                            intermediate_node.children.insert(new_page_key, new_node);
                            new_remaining.len()
                        } else {
                            0
                        };

                        // Attach before publication (see `insert_from`).
                        let newly_attached = intermediate_node.touch_tenant(&tenant_id, track_lfu);

                        entry.insert(intermediate_node);

                        let common_tokens = if newly_attached { common_len } else { 0 };
                        Step::Done(new_branch_tokens + common_tokens)
                    }
                }
            };

            match step {
                Step::Done(added) => {
                    tokens_added += added;
                    break;
                }
                Step::Continue {
                    next,
                    advance,
                    counted,
                } => {
                    tokens_added += counted;
                    remaining = &remaining[advance..];
                    current = next;
                }
            }
        }

        // Fold insert's token count in once (insert-side bookkeeping).
        self.add_tenant_tokens(tenant_id, tokens_added);

        PrefixMatchResult {
            tenant: last_tenant.unwrap_or_else(|| intern_tenant("empty")),
            matched_token_count: matched_tokens,
            input_token_count,
            matched_tenants,
        }
    }

    /// Match in a single descent, then insert for a tenant chosen from the match
    /// result — still one top-to-bottom traversal of the prefix.
    ///
    /// The inserting tenant is derived from the match outcome (route to the matched
    /// worker on a hit, else the least-loaded worker), so it is not known until the
    /// match completes. `select` is invoked once with the [`PrefixMatchResult`];
    /// `Some(tenant)` inserts `tokens` for it, `None` skips the insert.
    ///
    /// Equivalent to match-then-[`Self::insert_tokens`] single-threaded. Under
    /// concurrent node splits, `tenant_token_count` and routing stay exact (per-node
    /// advances are frozen at match time); only per-node access timestamps are
    /// approximate — a concurrently-split intermediate may miss a bump and evict
    /// slightly early.
    pub fn match_and_insert_with<'t, F>(&self, tokens: &[TokenId], select: F) -> PrefixMatchResult
    where
        F: FnOnce(&PrefixMatchResult) -> Option<&'t str>,
    {
        let page_size = self.page_size;
        let input_token_count = tokens.len();

        let aligned_len = align_to_page(tokens.len(), page_size);
        if aligned_len == 0 {
            // Too short to cache: `insert_tokens` is a no-op regardless of the
            // selected tenant, so just resolve + return the match result. We
            // still invoke `select` so the caller's routing side effects (if
            // any) run, matching a separate match-then-(skipped)-insert.
            let result = PrefixMatchResult {
                tenant: self
                    .root
                    .get_any_tenant()
                    .unwrap_or_else(|| intern_tenant("empty")),
                matched_token_count: 0,
                input_token_count,
                matched_tenants: Vec::new(),
            };
            let _ = select(&result);
            return result;
        }
        let tokens = &tokens[..aligned_len];

        let track_lfu = self.eviction_policy == EvictionPolicy::Lfu;

        // ---- Phase 1: MATCH descent (mirrors match_prefix_with_counts) ----
        // Additionally record every traversed edge so insert can replay its
        // per-node work without re-walking, and capture the fall-off node +
        // remaining slice for the splice.
        let mut matched_tokens = 0usize;
        let mut last_tenant: Option<TenantId> = None;
        let mut last_match_node: Option<NodeRef> = None;
        let mut remaining = tokens;
        let mut current = Arc::clone(&self.root);
        // Every full-match node we descended through, in order.
        // Pre-allocated; most matched paths are well under this depth.
        let mut path: Vec<NodeRef> = Vec::with_capacity(16);
        // Once match would stop (all-evicted node / partial), freeze the match
        // result but keep descending for insert (insert's reach is a superset).
        let mut match_frozen = false;

        enum MatchStep {
            Stop,
            Retry,
            Continue { next: NodeRef, advance: usize },
        }

        while remaining.len() >= page_size {
            let page_key = page_key_of(remaining, page_size);

            let step = match current.children.get(&page_key) {
                None => MatchStep::Stop,
                Some(child_ref) => 'probe: {
                    let child = Arc::clone(child_ref.value());
                    drop(child_ref);

                    let child_tokens = child.tokens.read();
                    // Stale-edge check (see `match_prefix_with_counts`).
                    if !std::ptr::eq(child.parent.read().as_ptr(), Arc::as_ptr(&current)) {
                        drop(child_tokens);
                        std::hint::spin_loop();
                        break 'probe MatchStep::Retry;
                    }
                    let match_len = remaining
                        .iter()
                        .zip(child_tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    let match_len = align_to_page(match_len, page_size);

                    if match_len == 0 {
                        drop(child_tokens);
                        MatchStep::Stop
                    } else if match_len < child_tokens.len() {
                        // Partial match within the node: match stops here.
                        if !match_frozen {
                            if let Some(t) = child.get_any_tenant() {
                                child.touch_tenant(&t, track_lfu);
                                matched_tokens += match_len;
                                last_tenant = Some(t);
                                last_match_node = Some(Arc::clone(&child));
                            }
                            // (If the node is all-evicted, match records nothing
                            // and simply stops — same as match_prefix_with_counts.)
                            match_frozen = true;
                        }
                        // Insert also stops descending here (it will split this
                        // node). Do NOT push to `path`; the splice handles it.
                        drop(child_tokens);
                        MatchStep::Stop
                    } else {
                        // Full match: match continues (if not frozen) and insert
                        // continues regardless.
                        drop(child_tokens);
                        if !match_frozen {
                            match child.get_any_tenant() {
                                None => {
                                    // All-evicted: match stops, insert continues.
                                    match_frozen = true;
                                }
                                Some(t) => {
                                    child.touch_tenant(&t, track_lfu);
                                    matched_tokens += match_len;
                                    last_tenant = Some(t);
                                    last_match_node = Some(Arc::clone(&child));
                                }
                            }
                        }
                        MatchStep::Continue {
                            next: child,
                            advance: match_len,
                        }
                    }
                }
            };

            match step {
                MatchStep::Stop => break,
                MatchStep::Retry => continue,
                MatchStep::Continue { next, advance } => {
                    path.push(Arc::clone(&next));
                    remaining = &remaining[advance..];
                    current = next;
                }
            }
        }

        // ---- Decide the insert tenant from the match result ----
        let result = PrefixMatchResult {
            tenant: last_tenant.unwrap_or_else(|| intern_tenant("empty")),
            matched_token_count: matched_tokens,
            input_token_count,
            matched_tenants: last_match_node
                .map(|node| node.matched_tenants())
                .unwrap_or_default(),
        };
        let Some(tenant) = select(&result) else {
            // No insert (router selected no worker).
            return result;
        };
        // Intern through the shared pool so the stored Arc dedups with every
        // other call (matches insert_tokens' interning).
        let tenant_id = intern_tenant(tenant);

        // ---- Phase 2: INSERT replay for `tenant_id` (no second walk) ----
        // Mirrors insert_tokens' root bookkeeping.
        self.root
            .tenant_last_access_time
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);
        self.tenant_token_count
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

        let mut tokens_added = 0usize;

        // Replay insert's per-node work on every edge the match descended:
        // `insert_tokens` touches the inserting tenant on each full-match node
        // and counts its tokens.
        for node in &path {
            // Credit the live length under the `tokens` read guard: a
            // concurrent split truncates the node and clones its tenant map in
            // one `tokens` write section, so this ordering decides both what
            // the tenant now owns and whether the clone inherits it. The
            // match-time length would over-credit a node truncated since.
            // Count only newly-attached edges (atomic via touch_tenant).
            let node_tokens = node.tokens.read();
            if node.touch_tenant(&tenant_id, track_lfu) {
                tokens_added += node_tokens.len();
            }
        }

        // Splice only the unmatched suffix at the fall-off node (`current`),
        // reusing insert_tokens' exact descent loop. It re-probes `current`'s
        // children for `remaining` (a single child-map op, not a re-walk of the
        // matched prefix) and handles the vacant / split / — and, under a
        // concurrent split race, full-match-continue — cases identically to a
        // standalone `insert_tokens`. The matched prefix above `current` was
        // already re-attached by the loop above and is never re-walked.
        if remaining.len() >= page_size {
            tokens_added += Self::insert_from(
                current,
                remaining,
                Arc::clone(&tenant_id),
                track_lfu,
                page_size,
            );
        }

        self.add_tenant_tokens(tenant_id, tokens_added);

        result
    }

    /// Legacy prefix_match API returning (matched_tokens, tenant_string).
    pub fn prefix_match_legacy(&self, tokens: &[TokenId]) -> (Vec<TokenId>, String) {
        let result = self.match_prefix_with_counts(tokens);
        let matched: Vec<TokenId> = tokens[..result.matched_token_count].to_vec();
        (matched, result.tenant.to_string())
    }

    /// Get token counts per tenant.
    pub fn get_tenant_token_counts(&self) -> HashMap<String, usize> {
        self.tenant_token_count
            .iter()
            .map(|entry| (entry.key().to_string(), *entry.value()))
            .collect()
    }

    /// Compute eviction priority for a node based on the configured policy.
    /// Returns (primary_key, tiebreaker) where lower values are evicted first.
    ///
    /// Uses wrapping arithmetic for u64→i64 conversion. This preserves relative ordering
    /// for values within i64::MAX of each other, which is sufficient since our timestamps
    /// are monotonic counters starting from 0.
    fn compute_eviction_priority(&self, node: &Node, last_access_time: u64) -> (i64, u64) {
        match self.eviction_policy {
            EvictionPolicy::Lru => {
                // Lower last_access_time = older = evict first
                // Wrapping cast preserves ordering for nearby values
                (last_access_time as i64, 0)
            }
            EvictionPolicy::Lfu => {
                // Lower hit_count = less used = evict first, tiebreak by last_access_time
                let hit_count = node.hit_count.load(Ordering::Relaxed);
                (hit_count as i64, last_access_time)
            }
            EvictionPolicy::Fifo => {
                // Lower creation_time = older = evict first
                (node.creation_time as i64, 0)
            }
            EvictionPolicy::Mru => {
                // Higher last_access_time = newer = evict first (negate for min-heap)
                // wrapping_neg handles i64::MIN correctly (wraps to itself)
                ((last_access_time as i64).wrapping_neg(), 0)
            }
            EvictionPolicy::Filo => {
                // Higher creation_time = newer = evict first (negate for min-heap)
                ((node.creation_time as i64).wrapping_neg(), 0)
            }
            EvictionPolicy::Priority => {
                // Lower priority = less important = evict first, tiebreak by last_access_time
                let priority = node.priority.load(Ordering::Relaxed);
                (priority as i64, last_access_time)
            }
        }
    }

    /// Evict entries for a tenant to reduce to max_tokens.
    /// Uses leaf-first eviction with the configured policy (matching SGLang's behavior).
    pub fn evict_tenant(&self, tenant: &TenantId, max_tokens: usize) {
        use std::{cmp::Reverse, collections::BinaryHeap};

        let current_count = self
            .tenant_token_count
            .get(tenant.as_ref())
            .map(|v| *v)
            .unwrap_or(0);

        if current_count <= max_tokens {
            return;
        }

        let to_evict = current_count - max_tokens;
        let mut evicted = 0;

        let mut leaves: Vec<(NodeRef, u64)> = Vec::new();
        self.collect_tenant_leaves(&self.root, tenant, &mut leaves);

        // Min-heap by eviction priority (policy-dependent)
        let mut heap: BinaryHeap<Reverse<((i64, u64), usize)>> = BinaryHeap::new();

        // Pre-allocate for initial leaves. Additional capacity for leaf promotions
        // (when a parent becomes a leaf after child eviction) will be handled by
        // Vec's amortized growth strategy.
        let mut leaf_data: Vec<NodeRef> = Vec::with_capacity(leaves.len());

        for (node, ts) in leaves.drain(..) {
            let priority = self.compute_eviction_priority(&node, ts);
            let idx = leaf_data.len();
            leaf_data.push(node);
            heap.push(Reverse((priority, idx)));
        }

        while evicted < to_evict {
            let Some(Reverse((_, idx))) = heap.pop() else {
                break;
            };

            let node = &leaf_data[idx];
            let (node_tokens, parent_became_leaf) = self.remove_tenant_and_cleanup(node, tenant);
            if node_tokens > 0 {
                evicted += node_tokens;

                // Incremental leaf promotion (matching SGLang's approach)
                if let Some((parent_node, parent_ts)) = parent_became_leaf {
                    let priority = self.compute_eviction_priority(&parent_node, parent_ts);
                    let new_idx = leaf_data.len();
                    leaf_data.push(parent_node);
                    heap.push(Reverse((priority, new_idx)));
                }
            }
        }

        self.sub_tenant_tokens(tenant, evicted);

        debug!(
            tenant = %tenant.as_ref(),
            evicted = evicted,
            remaining = current_count.saturating_sub(evicted),
            policy = ?self.eviction_policy,
            "Evicted tokens from tenant (leaf-first)"
        );
    }

    /// Collect tenant-specific leaves (nodes where tenant exists but no children have tenant).
    fn collect_tenant_leaves(
        &self,
        node: &NodeRef,
        tenant_id: &TenantId,
        result: &mut Vec<(NodeRef, u64)>,
    ) {
        let is_root = Arc::ptr_eq(node, &self.root);
        let node_has_tenant = !is_root
            && node
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref());

        let mut any_child_has_tenant = false;
        for child_entry in &node.children {
            let child = child_entry.value();
            if child
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref())
            {
                any_child_has_tenant = true;
            }
            self.collect_tenant_leaves(child, tenant_id, result);
        }

        if node_has_tenant && !any_child_has_tenant {
            if let Some(ts) = node.tenant_last_access_time.get(tenant_id.as_ref()) {
                result.push((Arc::clone(node), *ts));
            }
        }
    }

    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the tree instance; keeps API consistent with collect_tenant_leaves"
    )]
    fn is_tenant_leaf(&self, node: &NodeRef, tenant_id: &TenantId) -> bool {
        if !node
            .tenant_last_access_time
            .contains_key(tenant_id.as_ref())
        {
            return false;
        }
        for child_entry in &node.children {
            if child_entry
                .value()
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref())
            {
                return false;
            }
        }
        true
    }

    /// Remove tenant from node and clean up empty ancestors.
    /// Returns (tokens_removed, Option<(parent_node, ts)> if parent became a leaf).
    fn remove_tenant_and_cleanup(
        &self,
        node: &NodeRef,
        tenant_id: &TenantId,
    ) -> (usize, Option<(NodeRef, u64)>) {
        if node
            .tenant_last_access_time
            .remove(tenant_id.as_ref())
            .is_none()
        {
            return (0, None);
        }

        let node_tokens = node.tokens.read().len();
        let parent_leaf_info = self.cleanup_empty_ancestors_with_parent(node, tenant_id);

        (node_tokens, parent_leaf_info)
    }

    /// Remove empty nodes walking up via parent pointers.
    /// Returns Some((parent_node, ts)) if a parent became a tenant-leaf.
    fn cleanup_empty_ancestors_with_parent(
        &self,
        node: &NodeRef,
        tenant_id: &TenantId,
    ) -> Option<(NodeRef, u64)> {
        let mut current = Arc::clone(node);
        let mut parent_leaf_info: Option<(NodeRef, u64)> = None;

        loop {
            if !current.is_empty() {
                if parent_leaf_info.is_none() && self.is_tenant_leaf(&current, tenant_id) {
                    if let Some(ts) = current.tenant_last_access_time.get(tenant_id.as_ref()) {
                        parent_leaf_info = Some((Arc::clone(&current), *ts));
                    }
                }
                break;
            }

            let parent_weak = current.parent.read();
            let Some(parent) = parent_weak.upgrade() else {
                break;
            };
            drop(parent_weak);

            let page_key_guard = current.page_key.read();
            let Some(page_key) = *page_key_guard else {
                break;
            };
            drop(page_key_guard);

            parent.children.remove(&page_key);

            if parent_leaf_info.is_none() && self.is_tenant_leaf(&parent, tenant_id) {
                if let Some(ts) = parent.tenant_last_access_time.get(tenant_id.as_ref()) {
                    parent_leaf_info = Some((Arc::clone(&parent), *ts));
                }
            }

            current = parent;
        }

        parent_leaf_info
    }

    /// Get the token count for a specific tenant.
    pub fn tenant_token_size(&self, tenant: &TenantId) -> usize {
        // Use borrowed lookup to avoid Arc hash overhead
        self.tenant_token_count
            .get(tenant.as_ref())
            .map(|v| *v)
            .unwrap_or(0)
    }

    /// Tree-wide token total (sum of per-tenant counts).
    pub fn total_token_size(&self) -> usize {
        self.total_token_count.load(Ordering::Relaxed)
    }

    /// Fold `tokens` into a tenant's count and the tree-wide total.
    fn add_tenant_tokens(&self, tenant_id: TenantId, tokens: usize) {
        if tokens == 0 {
            return;
        }
        self.tenant_token_count
            .entry(tenant_id)
            .and_modify(|c| *c += tokens)
            .or_insert(tokens);
        self.total_token_count.fetch_add(tokens, Ordering::Relaxed);
    }

    /// Subtract evicted `tokens` from a tenant's count and the tree-wide
    /// total, clamped to the tenant's current count so the two stay in sync.
    fn sub_tenant_tokens(&self, tenant: &TenantId, tokens: usize) {
        if tokens == 0 {
            return;
        }
        if let Some(mut count) = self.tenant_token_count.get_mut(tenant.as_ref()) {
            let removed = tokens.min(*count);
            *count -= removed;
            self.total_token_count.fetch_sub(removed, Ordering::Relaxed);
        }
    }

    /// Clear the tree to empty state. Not synchronized with concurrent
    /// mutation: callers must quiesce writers first.
    pub fn clear(&self) {
        self.root.children.clear();
        self.root.tenant_last_access_time.clear();
        self.tenant_token_count.clear();
        self.total_token_count.store(0, Ordering::Relaxed);
    }

    /// Remove a tenant from every node (including the root), detach nodes
    /// the removal emptied, and drop the tenant's token-count entry.
    /// Size-based eviction never fires for a tenant whose count no longer
    /// grows, so removed workers must be purged eagerly.
    pub fn remove_tenant_all(&self, tenant_id: &TenantId) {
        let expected_tokens = self.tenant_token_size(tenant_id);
        self.root.tenant_last_access_time.remove(tenant_id.as_ref());
        if expected_tokens == 0 {
            self.tenant_token_count.remove(tenant_id.as_ref());
            return;
        }

        let mut nodes: Vec<NodeRef> = Vec::new();
        self.collect_tenant_nodes_pruned(&self.root, tenant_id, &mut nodes);

        // Inserts and leaf-first eviction keep tenant ownership prefix-closed,
        // so the pruned walk normally sees exactly the maintained token count.
        // Fall back to a complete walk if imported or concurrently-mutated
        // state violates that invariant.
        let collected_tokens: usize = nodes.iter().map(|node| node.tokens.read().len()).sum();
        if collected_tokens != expected_tokens {
            nodes.clear();
            self.collect_tenant_nodes(&self.root, tenant_id, &mut nodes);
        }
        for node in &nodes {
            self.remove_tenant_and_cleanup(node, tenant_id);
        }
        if let Some((_, count)) = self.tenant_token_count.remove(tenant_id.as_ref()) {
            self.total_token_count.fetch_sub(count, Ordering::Relaxed);
        }
    }

    /// Fast tenant walk for the common prefix-closed ownership shape. If a
    /// child does not contain the tenant, none of its descendants can contain
    /// it either, so the entire unrelated subtree can be skipped.
    fn collect_tenant_nodes_pruned(
        &self,
        node: &NodeRef,
        tenant_id: &TenantId,
        result: &mut Vec<NodeRef>,
    ) {
        for child in &node.children {
            let child = child.value();
            if !child
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref())
            {
                continue;
            }
            result.push(Arc::clone(child));
            self.collect_tenant_nodes_pruned(child, tenant_id, result);
        }
    }

    /// Collect every node owning `tenant_id`. The root is excluded (never
    /// detached); the caller drops its entry directly.
    fn collect_tenant_nodes(
        &self,
        node: &NodeRef,
        tenant_id: &TenantId,
        result: &mut Vec<NodeRef>,
    ) {
        if !Arc::ptr_eq(node, &self.root)
            && node
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref())
        {
            result.push(Arc::clone(node));
        }
        for child in &node.children {
            self.collect_tenant_nodes(child.value(), tenant_id, result);
        }
    }

    /// Evict cache entries until the tree-wide token total is at or under
    /// `max_size`. Matches the StringTree API.
    ///
    /// The budget is shared across all tenants of this tree: eviction triggers
    /// on the tree-wide total (no single tenant has to exceed anything) and
    /// removes leaves in policy order across tenants — least recently used
    /// first under LRU — using the same leaf-first machinery as
    /// [`Self::evict_tenant`].
    pub fn evict_tenant_by_size(&self, max_size: usize) {
        use std::{cmp::Reverse, collections::BinaryHeap};

        let initial_total = self.total_token_size();
        if initial_total <= max_size {
            return;
        }

        let mut leaves: Vec<(NodeRef, TenantId, u64)> = Vec::new();
        self.collect_all_tenant_leaves(&self.root, &mut leaves);

        // Min-heap by eviction priority (policy-dependent), across all tenants
        let mut heap: BinaryHeap<Reverse<((i64, u64), usize)>> = BinaryHeap::new();
        let mut leaf_data: Vec<(NodeRef, TenantId)> = Vec::with_capacity(leaves.len());

        for (node, tenant, ts) in leaves.drain(..) {
            let priority = self.compute_eviction_priority(&node, ts);
            let idx = leaf_data.len();
            leaf_data.push((node, tenant));
            heap.push(Reverse((priority, idx)));
        }

        while self.total_token_size() > max_size {
            let Some(Reverse((_, idx))) = heap.pop() else {
                break;
            };

            let (node, tenant) = {
                let (node, tenant) = &leaf_data[idx];
                (Arc::clone(node), Arc::clone(tenant))
            };
            // Verify this node is still a leaf for this tenant: a concurrent
            // insert may have attached the tenant to a descendant.
            if !self.is_tenant_leaf(&node, &tenant) {
                continue;
            }
            let (node_tokens, parent_became_leaf) = self.remove_tenant_and_cleanup(&node, &tenant);
            if node_tokens > 0 {
                self.sub_tenant_tokens(&tenant, node_tokens);

                // Incremental leaf promotion (matching SGLang's approach)
                if let Some((parent_node, parent_ts)) = parent_became_leaf {
                    let priority = self.compute_eviction_priority(&parent_node, parent_ts);
                    let new_idx = leaf_data.len();
                    leaf_data.push((parent_node, tenant));
                    heap.push(Reverse((priority, new_idx)));
                }
            }
        }

        debug!(
            evicted = initial_total.saturating_sub(self.total_token_size()),
            remaining = self.total_token_size(),
            max_size = max_size,
            policy = ?self.eviction_policy,
            "Evicted tokens to tree-wide budget (leaf-first)"
        );
    }

    /// Collect `(node, tenant, ts)` for every tenant-leaf of every tenant.
    fn collect_all_tenant_leaves(
        &self,
        node: &NodeRef,
        result: &mut Vec<(NodeRef, TenantId, u64)>,
    ) {
        for child_entry in &node.children {
            self.collect_all_tenant_leaves(child_entry.value(), result);
        }

        // Root is never eligible for eviction
        if Arc::ptr_eq(node, &self.root) {
            return;
        }
        for entry in &node.tenant_last_access_time {
            let tenant = entry.key();
            let any_child_has_tenant = node.children.iter().any(|child| {
                child
                    .value()
                    .tenant_last_access_time
                    .contains_key(tenant.as_ref())
            });
            if !any_child_has_tenant {
                result.push((Arc::clone(node), Arc::clone(tenant), *entry.value()));
            }
        }
    }

    /// Lazily walk the tree in pre-order, yielding each node's
    /// `(prefix_tokens, [(tenant, epoch)])` pair without
    /// materializing the full tree as a single buffer.
    /// Empty-tenant nodes are skipped (they're routing
    /// intermediates, not real entries). Children are visited in
    /// lexicographic page-key order so callers paging the
    /// iterator can resume by re-running and skipping the first
    /// N items if the tree is unchanged.
    ///
    /// The walk is not atomic — concurrent `insert_tokens` may
    /// split or replace nodes mid-walk; the iterator may then
    /// reflect a mix of pre- and post-split state. Acceptable
    /// for mesh sync (eventual consistency).
    pub fn iter_entries(&self) -> EntriesIter {
        EntriesIter::new(self)
    }
}

/// Iterator yielded by [`TokenTree::iter_entries`]. Owns
/// `Arc<Node>` clones for the path it's currently inside so it
/// survives across awaits or temporary releases of `&TokenTree`.
pub struct EntriesIter {
    /// DFS frame stack. The top frame's `remaining_children` is
    /// the queue of children we still have to visit at the
    /// current depth. Frames are pushed when descending, popped
    /// when their children exhaust.
    stack: Vec<EntryFrame>,
    /// Mutable accumulated path-from-root tokens. Pushed when
    /// descending into a child, truncated when popping a frame.
    path: Vec<TokenId>,
    /// Pre-staged emission. Set when a node with tenants is
    /// entered; consumed by the next `next()` call before
    /// continuing the walk.
    pending: Option<EntryItem>,
}

type EntryItem = (Vec<TokenId>, Vec<(TenantId, u64)>);

struct EntryFrame {
    remaining_children: std::vec::IntoIter<NodeRef>,
    /// Number of tokens contributed by this frame's edge — used
    /// to truncate the path buffer correctly when popping.
    edge_token_count: usize,
}

impl EntriesIter {
    fn new(tree: &TokenTree) -> Self {
        let root_tenants = snapshot_tenants(&tree.root);
        let pending = (!root_tenants.is_empty()).then(|| (Vec::<TokenId>::new(), root_tenants));
        Self {
            stack: vec![EntryFrame {
                remaining_children: collect_sorted_children(&tree.root).into_iter(),
                edge_token_count: 0,
            }],
            path: Vec::new(),
            pending,
        }
    }
}

impl Iterator for EntriesIter {
    type Item = EntryItem;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(entry) = self.pending.take() {
            return Some(entry);
        }
        loop {
            let frame = self.stack.last_mut()?;
            if let Some(child) = frame.remaining_children.next() {
                // Extend the path directly through the lock guard
                // — no intermediate Vec<TokenId> clone.
                let edge_token_count = {
                    let guard = child.tokens.read();
                    self.path.extend_from_slice(&guard);
                    guard.len()
                };

                let tenants = snapshot_tenants(&child);
                self.stack.push(EntryFrame {
                    remaining_children: collect_sorted_children(&child).into_iter(),
                    edge_token_count,
                });
                if !tenants.is_empty() {
                    return Some((self.path.clone(), tenants));
                }
                // Empty tenants → loop continues, descending into
                // this child's children on the next iteration.
            } else {
                let edge_token_count = frame.edge_token_count;
                self.stack.pop();
                let new_len = self.path.len().saturating_sub(edge_token_count);
                self.path.truncate(new_len);
            }
        }
    }
}

fn snapshot_tenants(node: &NodeRef) -> Vec<(TenantId, u64)> {
    let mut tenants: Vec<(TenantId, u64)> = node
        .tenant_last_access_time
        .iter()
        .map(|entry| (Arc::clone(entry.key()), *entry.value()))
        .collect();
    tenants.sort_by(|a, b| a.0.cmp(&b.0));
    tenants
}

fn collect_sorted_children(node: &NodeRef) -> Vec<NodeRef> {
    let mut children: Vec<(TokenPageKey, NodeRef)> = node
        .children
        .iter()
        .map(|entry| (*entry.key(), entry.value().clone()))
        .collect();
    // Lexicographic order over the page key — deterministic
    // traversal across hash-shard layouts.
    children.sort_by_key(|(k, _)| *k);
    children.into_iter().map(|(_, n)| n).collect()
}

impl RadixTree for TokenTree {
    type Key = [TokenId];
    type MatchResult = PrefixMatchResult;

    fn insert(&self, key: &Self::Key, tenant: &str) {
        self.insert_tokens(key, tenant);
    }

    fn prefix_match(&self, key: &Self::Key) -> Option<TenantId> {
        let result = self.match_prefix_with_counts(key);
        if result.matched_token_count > 0 {
            Some(result.tenant)
        } else {
            None
        }
    }

    fn prefix_match_with_counts(&self, key: &Self::Key) -> Self::MatchResult {
        self.match_prefix_with_counts(key)
    }

    fn evict(&self, tenant: &TenantId, max_units: usize) {
        self.evict_tenant(tenant, max_units);
    }

    fn tenant_size(&self, tenant: &TenantId) -> usize {
        self.tenant_token_size(tenant)
    }

    fn reset(&self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread};

    use super::*;

    /// Helper to create a page-aligned token sequence starting from `base`.
    /// Creates `pages` full pages of tokens.
    fn make_tokens(base: u32, pages: usize) -> Vec<TokenId> {
        (0..(pages * PAGE_SIZE)).map(|i| base + i as u32).collect()
    }

    #[test]
    fn test_basic_insert_match() {
        let tree = TokenTree::new();

        // Insert 2 pages (32 tokens)
        let tokens = make_tokens(1, 2);
        tree.insert_tokens(&tokens, "tenant1");

        // Exact match
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 32);
        assert_eq!(result.tenant.as_ref(), "tenant1");

        // Match first page only
        let first_page = make_tokens(1, 1);
        let result = tree.match_prefix_with_counts(&first_page);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");

        // Match with extra tokens (truncated to page boundary)
        let mut extended = tokens.clone();
        extended.extend([100, 101, 102, 103, 104]);
        let result = tree.match_prefix_with_counts(&extended);
        assert_eq!(result.matched_token_count, 32);
    }

    #[test]
    fn test_short_sequences_skipped() {
        let tree = TokenTree::new();

        // Sequences shorter than PAGE_SIZE are skipped
        tree.insert_tokens(&[1, 2, 3, 4, 5], "tenant1");

        // Should have no entries (too short)
        let counts = tree.get_tenant_token_counts();
        assert!(counts.is_empty() || counts.get("tenant1").copied().unwrap_or(0) == 0);

        // Lookup also returns 0 for short sequences
        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5]);
        assert_eq!(result.matched_token_count, 0);
        assert_eq!(result.input_token_count, 5);
    }

    #[test]
    fn test_multiple_tenants() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 1);
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant2");

        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        // Either tenant is valid
        assert!(result.tenant.as_ref() == "tenant1" || result.tenant.as_ref() == "tenant2");
    }

    #[test]
    fn test_prefix_split() {
        let tree = TokenTree::new();

        // Insert 3 pages first
        let long_tokens = make_tokens(1, 3);
        tree.insert_tokens(&long_tokens, "tenant1");

        // Insert 1 page (causes split)
        let short_tokens = make_tokens(1, 1);
        tree.insert_tokens(&short_tokens, "tenant2");

        // Short match
        let result = tree.match_prefix_with_counts(&short_tokens);
        assert_eq!(result.matched_token_count, PAGE_SIZE);

        // Long match
        let result = tree.match_prefix_with_counts(&long_tokens);
        assert_eq!(result.matched_token_count, 3 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");
    }

    #[test]
    fn test_empty_input() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 1);
        tree.insert_tokens(&tokens, "tenant1");

        let result = tree.match_prefix_with_counts(&[]);
        assert_eq!(result.matched_token_count, 0);
        assert_eq!(result.input_token_count, 0);
    }

    #[test]
    fn test_matched_tenants_all_holders_of_deepest_node() {
        let tree = TokenTree::new();
        let tokens = make_tokens(1, 2);
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant2");
        // tenant3 holds only the first page — not the deepest matched node.
        tree.insert_tokens(&make_tokens(1, 1), "tenant3");

        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 32);
        let mut tenants: Vec<&str> = result.matched_tenants.iter().map(AsRef::as_ref).collect();
        tenants.sort_unstable();
        assert_eq!(tenants, ["tenant1", "tenant2"]);
    }

    #[test]
    fn test_matched_tenants_empty_on_no_match() {
        let tree = TokenTree::new();
        tree.insert_tokens(&make_tokens(1, 1), "tenant1");

        let result = tree.match_prefix_with_counts(&make_tokens(1000, 1));
        assert_eq!(result.matched_token_count, 0);
        assert!(result.matched_tenants.is_empty());
    }

    #[test]
    fn test_matched_tenants_capped() {
        let tree = TokenTree::new();
        let tokens = make_tokens(1, 1);
        for i in 0..(MATCHED_TENANTS_CAP + 4) {
            tree.insert_tokens(&tokens, &format!("tenant{i}"));
        }

        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.matched_tenants.len(), MATCHED_TENANTS_CAP);
    }

    #[test]
    fn test_match_and_insert_with_populates_matched_tenants() {
        let tree = TokenTree::new();
        let tokens = make_tokens(1, 2);
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant2");

        let result = tree.match_and_insert_with(&tokens, |result| {
            let mut tenants: Vec<&str> = result.matched_tenants.iter().map(AsRef::as_ref).collect();
            tenants.sort_unstable();
            assert_eq!(tenants, ["tenant1", "tenant2"]);
            Some("tenant3")
        });
        assert_eq!(result.matched_token_count, 32);

        // The insert made tenant3 a holder of the same path.
        let result = tree.match_prefix_with_counts(&tokens);
        assert!(result
            .matched_tenants
            .iter()
            .any(|tenant| tenant.as_ref() == "tenant3"));
    }

    #[test]
    fn test_no_match() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 1);
        tree.insert_tokens(&tokens, "tenant1");

        // Different page key
        let other = make_tokens(1000, 1);
        let result = tree.match_prefix_with_counts(&other);
        assert_eq!(result.matched_token_count, 0);
    }

    #[test]
    fn test_eviction() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 2);
        let tokens2 = make_tokens(1, 3);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");

        let counts = tree.get_tenant_token_counts();
        assert!(counts.get("tenant1").unwrap() > &0);

        tree.evict(&TenantId::from("tenant1"), 0);

        // After eviction, counts should be reduced
        let new_counts = tree.get_tenant_token_counts();
        let new_count = new_counts.get("tenant1").copied().unwrap_or(0);
        assert!(new_count < *counts.get("tenant1").unwrap());
    }

    #[test]
    fn test_concurrent_insert_match() {
        let tree = Arc::new(TokenTree::new());
        let mut handles = vec![];

        // Spawn inserters - use page-aligned sequences
        for i in 0..4 {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let base = (i * 1000000 + j * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    tree.insert_tokens(&tokens, &format!("tenant{i}"));
                }
            }));
        }

        // Spawn matchers
        for i in 0..4 {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let base = (i * 1000000 + j * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    let _ = tree.match_prefix_with_counts(&tokens);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_prefix_match_with_counts() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 2);
        tree.insert_tokens(&tokens, "tenant1");

        // Exact match
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.input_token_count, 2 * PAGE_SIZE);

        // Match first page
        let first_page = make_tokens(1, 1);
        let result = tree.match_prefix_with_counts(&first_page);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.input_token_count, PAGE_SIZE);

        // Extended input (aligned)
        let extended = make_tokens(1, 3);
        let result = tree.match_prefix_with_counts(&extended);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.input_token_count, 3 * PAGE_SIZE);
    }

    #[test]
    fn test_disjoint_paths() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(1000, 1);
        let tokens3 = make_tokens(2000, 1);

        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");
        tree.insert_tokens(&tokens3, "tenant3");

        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");

        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");

        let result = tree.match_prefix_with_counts(&tokens3);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant3");
    }

    #[test]
    fn test_branching_paths() {
        let tree = TokenTree::new();

        // Common first page, different second page
        let mut tokens1 = make_tokens(1, 1);
        tokens1.extend(make_tokens(100, 1));

        let mut tokens2 = make_tokens(1, 1);
        tokens2.extend(make_tokens(200, 1));

        let mut tokens3 = make_tokens(1, 1);
        tokens3.extend(make_tokens(300, 1));

        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");
        tree.insert_tokens(&tokens3, "tenant3");

        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");

        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");

        // Partial match at branch point
        let first_page = make_tokens(1, 1);
        let result = tree.match_prefix_with_counts(&first_page);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
    }

    #[test]
    fn test_radix_tree_trait() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 2);
        RadixTree::insert(&tree, &tokens, "tenant1");

        let tenant = RadixTree::prefix_match(&tree, &tokens);
        assert!(tenant.is_some());
        assert_eq!(tenant.unwrap().as_ref(), "tenant1");

        // Extended input - should match 2 pages (short sequences get 0)
        let extended = make_tokens(1, 3);
        let result = RadixTree::prefix_match_with_counts(&tree, &extended);
        assert_eq!(result.matched_count(), 2 * PAGE_SIZE);
        assert_eq!(result.input_count(), 3 * PAGE_SIZE);

        assert!(RadixTree::tenant_size(&tree, &TenantId::from("tenant1")) > 0);
    }

    #[test]
    fn test_clear() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(1000, 1);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");

        assert!(!tree.get_tenant_token_counts().is_empty());

        tree.clear();

        assert!(tree.get_tenant_token_counts().is_empty());
        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(result.matched_token_count, 0);
    }

    #[test]
    fn test_tenant_token_count() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 2);
        let tokens2 = make_tokens(1, 3);
        let tokens3 = make_tokens(1000, 1);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant2");

        let tenant1_id: TenantId = Arc::from("tenant1");
        let tenant2_id: TenantId = Arc::from("tenant2");

        assert!(tree.tenant_token_size(&tenant1_id) >= PAGE_SIZE);
        assert!(tree.tenant_token_size(&tenant2_id) >= PAGE_SIZE);

        let counts = tree.get_tenant_token_counts();
        assert!(counts.contains_key("tenant1"));
        assert!(counts.contains_key("tenant2"));
    }

    #[test]
    fn test_cold_start() {
        let tree = TokenTree::new();
        // Short sequences return 0 (no cache benefit)
        let result = tree.match_prefix_with_counts(&[1, 2, 3, 4, 5]);
        assert_eq!(result.matched_token_count, 0);
        assert_eq!(result.input_token_count, 5);

        // Page-sized sequences also return 0 on empty tree
        let tokens = make_tokens(1, 1);
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 0);
        assert_eq!(result.input_token_count, PAGE_SIZE);
    }

    #[test]
    fn test_exact_match_seq() {
        let tree = TokenTree::new();

        for i in 0..100 {
            let base = (i * 1000) as u32;
            let tokens = make_tokens(base, 2);
            tree.insert_tokens(&tokens, &format!("tenant{i}"));
        }

        for i in 0..100 {
            let base = (i * 1000) as u32;
            let tokens = make_tokens(base, 2);
            let result = tree.match_prefix_with_counts(&tokens);
            assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
            assert_eq!(result.tenant.as_ref(), &format!("tenant{i}"));
        }
    }

    #[test]
    fn test_exact_match_concurrent() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 8;
        let entries_per_thread = 100;

        // Insert phase
        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..entries_per_thread {
                    let base = (t * 1000000 + i * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    tree.insert_tokens(&tokens, &format!("tenant{t}"));
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        // Match phase
        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..entries_per_thread {
                    let base = (t * 1000000 + i * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    let result = tree.match_prefix_with_counts(&tokens);
                    assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
                    assert_eq!(result.tenant.as_ref(), &format!("tenant{t}"));
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_partial_match_concurrent() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 8;
        let entries_per_thread = 100;

        // Insert full sequences (3 pages)
        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..entries_per_thread {
                    let base = (t * 1000000 + i * 1000) as u32;
                    let tokens = make_tokens(base, 3);
                    tree.insert_tokens(&tokens, &format!("tenant{t}"));
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        // Match with partial (1 page)
        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..entries_per_thread {
                    let base = (t * 1000000 + i * 1000) as u32;
                    let partial = make_tokens(base, 1);
                    let result = tree.match_prefix_with_counts(&partial);
                    assert_eq!(result.matched_token_count, PAGE_SIZE);
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_group_prefix_insert_match_concurrent() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 8;

        // All threads share the same prefix (1 page)
        let common_prefix = make_tokens(100, 1);

        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            let prefix = common_prefix.clone();
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let mut tokens = prefix.clone();
                    let suffix = make_tokens((t * 10000 + i * 100) as u32, 1);
                    tokens.extend(suffix);
                    tree.insert_tokens(&tokens, &format!("tenant{t}"));
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify prefix matching works
        let result = tree.match_prefix_with_counts(&common_prefix);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
    }

    #[test]
    fn test_mixed_concurrent_insert_match() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 4;

        // Pre-populate some data
        for i in 0..100 {
            let base = (i * 1000) as u32;
            let tokens = make_tokens(base, 2);
            tree.insert_tokens(&tokens, &format!("initial{i}"));
        }

        let mut handles = vec![];

        // Concurrent inserters
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let base = (10000000 + t * 100000 + i * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    tree.insert_tokens(&tokens, &format!("new_tenant{t}"));
                }
            }));
        }

        // Concurrent matchers (matching existing data)
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let base = ((t * 10) * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    let result = tree.match_prefix_with_counts(&tokens);
                    assert!(result.matched_token_count > 0);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_simple_eviction() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 2);
        let tokens2 = make_tokens(1000, 2);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");

        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // tenant2 should still work
        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");
    }

    #[test]
    fn test_advanced_eviction() {
        let tree = TokenTree::new();

        // Insert multiple paths for tenant1
        let mut tokens1 = make_tokens(1, 1);
        tokens1.extend(make_tokens(100, 1));
        let mut tokens2 = make_tokens(1, 1);
        tokens2.extend(make_tokens(200, 1));
        let mut tokens3 = make_tokens(1, 1);
        tokens3.extend(make_tokens(300, 1));

        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant1");

        let tenant1_id: TenantId = Arc::from("tenant1");

        // Partial eviction
        let initial_size = tree.tenant_token_size(&tenant1_id);
        tree.evict_tenant(&tenant1_id, initial_size / 2);
        let after_size = tree.tenant_token_size(&tenant1_id);

        assert!(after_size <= initial_size);
    }

    #[test]
    fn test_concurrent_operations_with_eviction() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 4;

        // Pre-populate
        for i in 0..100 {
            let base = (i * 1000) as u32;
            let tokens = make_tokens(base, 2);
            tree.insert_tokens(&tokens, &format!("tenant{}", i % 4));
        }

        let mut handles = vec![];

        // Inserters
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let base = (10000000 + t * 100000 + i * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    tree.insert_tokens(&tokens, &format!("tenant{t}"));
                }
            }));
        }

        // Evictors
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                let tenant_id: TenantId = Arc::from(format!("tenant{t}"));
                for _ in 0..10 {
                    tree.evict_tenant(&tenant_id, 50);
                    thread::sleep(std::time::Duration::from_millis(1));
                }
            }));
        }

        // Matchers
        for _ in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let base = (i * 1000) as u32;
                    let tokens = make_tokens(base, 1);
                    let _ = tree.match_prefix_with_counts(&tokens);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Tree-wide total stays in sync with per-tenant counts.
        let sum: usize = tree.get_tenant_token_counts().values().sum();
        assert_eq!(tree.total_token_size(), sum);
    }

    #[test]
    fn test_get_used_size_per_tenant() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 2);
        let tokens2 = make_tokens(1, 3);
        let tokens3 = make_tokens(1000, 1);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant2");

        let counts = tree.get_tenant_token_counts();

        // tokens2 shares tokens1's full 2-page prefix: tenant1 owns exactly
        // 3 pages, not 2 + 3.
        assert_eq!(*counts.get("tenant1").unwrap(), 3 * PAGE_SIZE);
        assert_eq!(*counts.get("tenant2").unwrap(), PAGE_SIZE);
    }

    #[test]
    fn repeat_insert_does_not_inflate_tenant_count() {
        let tree = TokenTree::new();
        let tokens = make_tokens(1, 3);

        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant1");

        let counts = tree.get_tenant_token_counts();
        assert_eq!(*counts.get("tenant1").unwrap(), 3 * PAGE_SIZE);
    }

    #[test]
    fn match_and_insert_with_repeat_keeps_count_stable() {
        let tree = TokenTree::new();
        let tokens = make_tokens(1, 3);

        for _ in 0..3 {
            tree.match_and_insert_with(&tokens, |_| Some("tenant1"));
        }

        let counts = tree.get_tenant_token_counts();
        assert_eq!(*counts.get("tenant1").unwrap(), 3 * PAGE_SIZE);
    }

    #[test]
    fn match_and_insert_repeat_keeps_count_stable() {
        let tree = TokenTree::new();
        let tokens = make_tokens(1, 3);

        tree.match_and_insert(&tokens, "tenant1");
        tree.match_and_insert(&tokens, "tenant1");
        tree.match_and_insert(&tokens, "tenant1");

        let counts = tree.get_tenant_token_counts();
        assert_eq!(*counts.get("tenant1").unwrap(), 3 * PAGE_SIZE);
    }

    #[test]
    fn test_prefix_match_tenant() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 2);
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant2");

        // Both tenants should have access time updated
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        // tenant should be either tenant1 or tenant2 (last_tenant cache)
        assert!(result.tenant.as_ref() == "tenant1" || result.tenant.as_ref() == "tenant2");
    }

    #[test]
    fn test_simple_tenant_eviction() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 2);
        let tokens2 = make_tokens(1000, 2);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");

        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // tenant2 should be unaffected
        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");
    }

    #[test]
    fn test_complex_tenant_eviction() {
        let tree = TokenTree::new();

        // Create overlapping paths (common first page, different second pages)
        let mut tokens1 = make_tokens(1, 1);
        tokens1.extend(make_tokens(100, 1));
        let mut tokens2 = make_tokens(1, 1);
        tokens2.extend(make_tokens(200, 1));
        let mut tokens3 = make_tokens(1, 1);
        tokens3.extend(make_tokens(300, 1));

        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");
        tree.insert_tokens(&tokens3, "tenant1");

        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // tenant2's path should still work
        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");
    }

    #[test]
    fn test_empty_node_cleanup_after_eviction() {
        // Test that empty nodes are removed from the tree after tenant eviction
        // This prevents memory bloat from orphaned nodes
        let tree = TokenTree::new();

        // Insert a path only owned by tenant1
        let tokens1 = make_tokens(500, 2); // Unique path
        tree.insert_tokens(&tokens1, "tenant1");

        // Verify tree has children before eviction
        assert!(
            !tree.root.children.is_empty(),
            "Tree should have children after insert"
        );

        // Evict tenant1 completely
        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // Verify empty nodes were cleaned up
        assert!(
            tree.root.children.is_empty(),
            "Empty nodes should be removed after tenant eviction"
        );

        // Verify tenant count is zero
        assert_eq!(tree.tenant_token_size(&tenant1_id), 0);
    }

    #[test]
    fn test_partial_cleanup_shared_nodes() {
        // Test that shared nodes are NOT cleaned up when still owned by other tenants
        let tree = TokenTree::new();

        // Both tenants share the same path
        let tokens = make_tokens(100, 2);
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant2");

        // Evict tenant1
        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // Tree should still have children (owned by tenant2)
        assert!(
            !tree.root.children.is_empty(),
            "Shared nodes should NOT be removed when other tenants exist"
        );

        // tenant2 should still work
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");
    }

    #[test]
    fn test_leaf_first_eviction() {
        // Test that eviction follows leaf-first LRU like SGLang's radix cache
        //
        // Tree structure for tenant1:
        //   root -> [A: page1] -> [B: page2] -> [C: page3]
        //
        // When we evict, C (leaf) should be evicted first, then B, then A.
        // This ensures shared prefixes (like system prompts near root) survive longer.

        let tree = TokenTree::new();

        // Insert a 3-page path: A -> B -> C
        let path_abc = make_tokens(1, 3); // 3 pages
        tree.insert_tokens(&path_abc, "tenant1");

        // Also insert just the first page (shared prefix)
        let path_a = make_tokens(1, 1); // 1 page
        tree.insert_tokens(&path_a, "tenant1");

        // Initial count should be 3 pages (A, B, C share with the single A)
        let initial_count = tree.tenant_token_size(&Arc::from("tenant1"));
        assert_eq!(initial_count, 3 * PAGE_SIZE);

        // Evict to keep only 2 pages - should evict C (the leaf)
        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 2 * PAGE_SIZE);

        // After evicting C, we should still match A and B
        let result = tree.match_prefix_with_counts(&path_abc);
        // Should match 2 pages (A and B), not 3
        assert!(
            result.matched_token_count <= 2 * PAGE_SIZE,
            "Expected at most 2 pages matched after evicting leaf, got {}",
            result.matched_token_count / PAGE_SIZE
        );

        // The prefix path should still work
        let result = tree.match_prefix_with_counts(&path_a);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "Shared prefix A should still be cached"
        );

        // Evict more - should evict B (now the new leaf)
        tree.evict_tenant(&tenant1_id, PAGE_SIZE);

        // Now only A should remain
        let result = tree.match_prefix_with_counts(&path_abc);
        assert!(
            result.matched_token_count <= PAGE_SIZE,
            "Expected at most 1 page matched after evicting B"
        );

        // A should still work
        let result = tree.match_prefix_with_counts(&path_a);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "Root prefix A should survive longest"
        );
    }

    #[test]
    fn test_single_page_operations() {
        let tree = TokenTree::new();

        // Single page operations
        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(1000, 1);
        let tokens3 = make_tokens(2000, 1);

        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");
        tree.insert_tokens(&tokens3, "tenant3");

        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");

        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");

        let result = tree.match_prefix_with_counts(&tokens3);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant3");
    }

    #[test]
    fn test_prefix_is_subset_of_existing() {
        let tree = TokenTree::new();

        // Insert longer sequence first (3 pages)
        let long_tokens = make_tokens(1, 3);
        tree.insert_tokens(&long_tokens, "tenant1");

        // Insert prefix (1 page)
        let short_tokens = make_tokens(1, 1);
        tree.insert_tokens(&short_tokens, "tenant2");

        // Short match
        let result = tree.match_prefix_with_counts(&short_tokens);
        assert_eq!(result.matched_token_count, PAGE_SIZE);

        // Long match
        let result = tree.match_prefix_with_counts(&long_tokens);
        assert_eq!(result.matched_token_count, 3 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");
    }

    #[test]
    fn test_existing_is_prefix_of_new() {
        let tree = TokenTree::new();

        // Insert shorter first (1 page)
        let short_tokens = make_tokens(1, 1);
        tree.insert_tokens(&short_tokens, "tenant1");

        // Insert longer (3 pages)
        let long_tokens = make_tokens(1, 3);
        tree.insert_tokens(&long_tokens, "tenant2");

        // Short match
        let result = tree.match_prefix_with_counts(&short_tokens);
        assert_eq!(result.matched_token_count, PAGE_SIZE);

        // Long match
        let result = tree.match_prefix_with_counts(&long_tokens);
        assert_eq!(result.matched_token_count, 3 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant2");
    }

    #[test]
    fn test_prefix_match_with_counts_accuracy() {
        let tree = TokenTree::new();

        // Insert 4 pages
        let tokens = make_tokens(1, 4);
        tree.insert_tokens(&tokens, "tenant1");

        // Exact match
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 4 * PAGE_SIZE);
        assert_eq!(result.input_token_count, 4 * PAGE_SIZE);

        // Partial match (2 pages)
        let partial = make_tokens(1, 2);
        let result = tree.match_prefix_with_counts(&partial);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(result.input_token_count, 2 * PAGE_SIZE);

        // Extended match (input longer than inserted)
        let extended = make_tokens(1, 6);
        let result = tree.match_prefix_with_counts(&extended);
        assert_eq!(result.matched_token_count, 4 * PAGE_SIZE);
        assert_eq!(result.input_token_count, 6 * PAGE_SIZE);
    }

    #[test]
    fn test_split_at_page_boundary() {
        let tree = TokenTree::new();

        // Insert 3 pages
        let long_tokens = make_tokens(1, 3);
        tree.insert_tokens(&long_tokens, "tenant1");

        // Insert 1 page (causes split at page boundary)
        let short_tokens = make_tokens(1, 1);
        tree.insert_tokens(&short_tokens, "tenant2");

        // 1 page match
        let result = tree.match_prefix_with_counts(&short_tokens);
        assert_eq!(result.matched_token_count, PAGE_SIZE);

        // 3 pages match
        let result = tree.match_prefix_with_counts(&long_tokens);
        assert_eq!(result.matched_token_count, 3 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");
    }

    #[test]
    fn test_multiple_splits_same_path() {
        let tree = TokenTree::new();

        // Insert 4 pages first
        let tokens4 = make_tokens(1, 4);
        tree.insert_tokens(&tokens4, "tenant1");

        // Insert 3 pages
        let tokens3 = make_tokens(1, 3);
        tree.insert_tokens(&tokens3, "tenant2");

        // Insert 2 pages
        let tokens2 = make_tokens(1, 2);
        tree.insert_tokens(&tokens2, "tenant3");

        // Insert 1 page
        let tokens1 = make_tokens(1, 1);
        tree.insert_tokens(&tokens1, "tenant4");

        // All should match correctly
        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(result.matched_token_count, PAGE_SIZE);

        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);

        let result = tree.match_prefix_with_counts(&tokens3);
        assert_eq!(result.matched_token_count, 3 * PAGE_SIZE);

        let result = tree.match_prefix_with_counts(&tokens4);
        assert_eq!(result.matched_token_count, 4 * PAGE_SIZE);
    }

    #[test]
    fn test_high_contention_same_prefix() {
        let tree = Arc::new(TokenTree::new());
        let prefix = make_tokens(100, 1);
        let num_threads = 16;

        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            let p = prefix.clone();
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let mut tokens = p.clone();
                    let suffix = make_tokens((t * 10000 + i * 100) as u32, 1);
                    tokens.extend(suffix);
                    tree.insert_tokens(&tokens, &format!("tenant{t}"));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify prefix matching
        let result = tree.match_prefix_with_counts(&prefix);
        assert_eq!(result.matched_token_count, PAGE_SIZE);
    }

    #[test]
    fn test_rapid_insert_remove_cycles() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 4;

        let mut handles = vec![];
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                let tenant_id: TenantId = Arc::from(format!("tenant{t}"));
                for cycle in 0..10 {
                    // Insert
                    for i in 0..20 {
                        let base = (t * 10000000 + cycle * 100000 + i * 1000) as u32;
                        let tokens = make_tokens(base, 2);
                        tree.insert_tokens(&tokens, &format!("tenant{t}"));
                    }
                    // Evict
                    tree.evict_tenant(&tenant_id, 10);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_eviction_empty_tree() {
        let tree = TokenTree::new();
        let tenant_id: TenantId = Arc::from("nonexistent");

        // Should not panic
        tree.evict_tenant(&tenant_id, 0);
        tree.evict_tenant(&tenant_id, 100);
    }

    #[test]
    fn test_eviction_zero_max_size() {
        let tree = TokenTree::new();

        let tokens1 = make_tokens(1, 2);
        let tokens2 = make_tokens(1000, 2);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");

        let tenant_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant_id, 0);

        // Eviction with max_size=0 should remove entries
        let size = tree.tenant_token_size(&tenant_id);
        assert!(size < 4 * PAGE_SIZE);
    }

    #[test]
    fn test_eviction_single_tenant_all_entries() {
        let tree = TokenTree::new();

        // Insert multiple entries for one tenant
        for i in 0..10 {
            let base = (i * 1000) as u32;
            let tokens = make_tokens(base, 2);
            tree.insert_tokens(&tokens, "tenant1");
        }

        let tenant_id: TenantId = Arc::from("tenant1");
        let initial_size = tree.tenant_token_size(&tenant_id);
        assert!(initial_size > 0);

        tree.evict_tenant(&tenant_id, 0);

        let final_size = tree.tenant_token_size(&tenant_id);
        assert!(final_size < initial_size);
    }

    #[test]
    fn test_last_tenant_cache_update() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 1);
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant2");

        // First match
        let result1 = tree.match_prefix_with_counts(&tokens);
        let first_tenant = result1.tenant.clone();

        // Match again - should get cached tenant
        let result2 = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result2.tenant, first_tenant);
    }

    #[test]
    fn test_stale_cache_after_tenant_removal() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 2);
        tree.insert_tokens(&tokens, "tenant1");
        tree.insert_tokens(&tokens, "tenant2");

        // Match to populate cache
        let _ = tree.match_prefix_with_counts(&tokens);

        // Evict one tenant
        let tenant1_id: TenantId = Arc::from("tenant1");
        tree.evict_tenant(&tenant1_id, 0);

        // Match should still work (tenant2 or cache still valid)
        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
    }

    #[test]
    fn test_token_count_consistency_after_operations() {
        let tree = TokenTree::new();

        // Insert
        let tokens1 = make_tokens(1, 3);
        let tokens2 = make_tokens(1, 5);
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");

        let tenant1_id: TenantId = Arc::from("tenant1");
        let count1 = tree.tenant_token_size(&tenant1_id);
        assert!(count1 >= PAGE_SIZE);

        // Partial eviction
        tree.evict_tenant(&tenant1_id, count1 / 2);
        let count2 = tree.tenant_token_size(&tenant1_id);
        assert!(count2 <= count1);

        // Insert more
        let tokens3 = make_tokens(2000, 2);
        tree.insert_tokens(&tokens3, "tenant1");
        let count3 = tree.tenant_token_size(&tenant1_id);
        assert!(count3 >= count2);
    }

    #[test]
    fn test_tree_structure_integrity_after_stress() {
        let tree = Arc::new(TokenTree::new());
        let num_threads = 8;

        let mut handles = vec![];

        // Stress insert
        for t in 0..num_threads {
            let tree = Arc::clone(&tree);
            handles.push(thread::spawn(move || {
                for i in 0..200 {
                    let base = (t * 10000000 + i * 1000) as u32;
                    let tokens = make_tokens(base, 2);
                    tree.insert_tokens(&tokens, &format!("tenant{t}"));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify structure by matching
        for t in 0..num_threads {
            for i in 0..10 {
                let base = (t * 10000000 + i * 1000) as u32;
                let tokens = make_tokens(base, 2);
                let result = tree.match_prefix_with_counts(&tokens);
                assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);
            }
        }
    }

    #[test]
    fn test_very_long_sequences() {
        let tree = TokenTree::new();

        // Insert a very long sequence (64 pages = 1024 tokens)
        let long_seq = make_tokens(1, 64);
        tree.insert_tokens(&long_seq, "tenant1");

        // Match the full sequence
        let result = tree.match_prefix_with_counts(&long_seq);
        assert_eq!(result.matched_token_count, 64 * PAGE_SIZE);
        assert_eq!(result.tenant.as_ref(), "tenant1");

        // Match a prefix (32 pages)
        let prefix = make_tokens(1, 32);
        let result = tree.match_prefix_with_counts(&prefix);
        assert_eq!(result.matched_token_count, 32 * PAGE_SIZE);
    }

    #[test]
    fn test_many_tenants_same_path() {
        let tree = TokenTree::new();

        let tokens = make_tokens(1, 2);

        for i in 0..100 {
            tree.insert_tokens(&tokens, &format!("tenant{i}"));
        }

        let result = tree.match_prefix_with_counts(&tokens);
        assert_eq!(result.matched_token_count, 2 * PAGE_SIZE);

        // Some tenants should have registered
        let counts = tree.get_tenant_token_counts();
        assert!(!counts.is_empty()); // At least some tenants tracked
    }

    #[test]
    fn test_token_id_edge_values() {
        let tree = TokenTree::new();

        // Test with edge case token IDs in page-aligned sequences
        // Create page starting with 0
        let mut zeros_page: Vec<TokenId> = (0..PAGE_SIZE as u32).collect();
        zeros_page[0] = 0;
        tree.insert_tokens(&zeros_page, "tenant1");

        // Create page starting with u32::MAX
        let mut max_page: Vec<TokenId> = (0..PAGE_SIZE as u32).collect();
        max_page[0] = u32::MAX;
        tree.insert_tokens(&max_page, "tenant2");

        // Create page with mixed edge values
        let mut mixed_page: Vec<TokenId> = (0..PAGE_SIZE as u32).collect();
        mixed_page[0] = 0;
        mixed_page[1] = u32::MAX;
        tree.insert_tokens(&mixed_page, "tenant3");

        let (matched, tenant) = tree.prefix_match_legacy(&zeros_page);
        assert_eq!(matched.len(), PAGE_SIZE);
        assert!(tenant == "tenant1" || tenant == "tenant3");

        let (matched, tenant) = tree.prefix_match_legacy(&max_page);
        assert_eq!(matched.len(), PAGE_SIZE);
        assert_eq!(tenant, "tenant2");
    }

    #[test]
    fn test_hit_ratio_calculation() {
        use crate::MatchResult;

        let tree = TokenTree::new();
        let one_page = make_tokens(1, 1); // 1 page = PAGE_SIZE tokens
        tree.insert_tokens(&one_page, "tenant1");

        // 100% hit ratio - query exactly the inserted sequence
        let result = tree.match_prefix_with_counts(&one_page);
        assert!((result.hit_ratio() - 1.0).abs() < 0.001);

        // 50% hit ratio - query 2 pages, only 1 page cached
        let two_pages = make_tokens(1, 2);
        let result = tree.match_prefix_with_counts(&two_pages);
        assert!((result.hit_ratio() - 0.5).abs() < 0.001);

        // 0% hit ratio - query non-existent tokens (but page-aligned)
        let no_match = make_tokens(1000, 1);
        let result = tree.match_prefix_with_counts(&no_match);
        assert_eq!(result.hit_ratio(), 0.0);
    }

    #[test]
    fn test_reset_via_trait() {
        use crate::RadixTree;

        let tree = TokenTree::new();
        tree.insert_tokens(&make_tokens(1, 1), "tenant1");
        tree.insert_tokens(&make_tokens(100, 1), "tenant2");

        assert!(!tree.get_tenant_token_counts().is_empty());

        RadixTree::reset(&tree);

        assert!(tree.get_tenant_token_counts().is_empty());
    }

    #[test]
    fn test_eviction_policy_lru() {
        // LRU: Evict oldest by last access time
        let tree = TokenTree::with_policy(EvictionPolicy::Lru);

        // Insert 3 paths, access them in order (oldest first)
        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(100, 1);
        let tokens3 = make_tokens(200, 1);

        tree.insert_tokens(&tokens1, "tenant1"); // Oldest
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant1"); // Newest

        // Access tokens2 to make it newer
        let _ = tree.match_prefix_with_counts(&tokens2);

        // Evict 1 page - should evict tokens1 (oldest by last_access_time)
        tree.evict_tenant(&Arc::from("tenant1"), 2 * PAGE_SIZE);

        // tokens1 should be evicted, tokens2 and tokens3 should remain
        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(
            result.matched_token_count, 0,
            "tokens1 should be evicted (oldest)"
        );

        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "tokens2 should remain"
        );

        let result = tree.match_prefix_with_counts(&tokens3);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "tokens3 should remain"
        );
    }

    #[test]
    fn test_eviction_policy_mru() {
        // MRU: Evict newest by last access time (opposite of LRU)
        let tree = TokenTree::with_policy(EvictionPolicy::Mru);

        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(100, 1);
        let tokens3 = make_tokens(200, 1);

        tree.insert_tokens(&tokens1, "tenant1"); // Oldest
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant1"); // Newest

        // Evict 1 page - should evict tokens3 (newest by last_access_time)
        tree.evict_tenant(&Arc::from("tenant1"), 2 * PAGE_SIZE);

        // tokens3 should be evicted (newest), tokens1 and tokens2 should remain
        let result = tree.match_prefix_with_counts(&tokens3);
        assert_eq!(
            result.matched_token_count, 0,
            "tokens3 should be evicted (newest)"
        );

        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "tokens1 should remain"
        );
    }

    #[test]
    fn test_eviction_policy_fifo() {
        // FIFO: Evict oldest by creation time (not affected by access)
        let tree = TokenTree::with_policy(EvictionPolicy::Fifo);

        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(100, 1);
        let tokens3 = make_tokens(200, 1);

        tree.insert_tokens(&tokens1, "tenant1"); // First created
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant1"); // Last created

        // Access tokens1 multiple times (shouldn't affect FIFO)
        for _ in 0..10 {
            let _ = tree.match_prefix_with_counts(&tokens1);
        }

        // Evict 1 page - should evict tokens1 (oldest by creation_time)
        tree.evict_tenant(&Arc::from("tenant1"), 2 * PAGE_SIZE);

        // tokens1 should be evicted despite being accessed most recently
        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(
            result.matched_token_count, 0,
            "tokens1 should be evicted (oldest creation)"
        );
    }

    #[test]
    fn test_eviction_policy_filo() {
        // FILO: Evict newest by creation time (stack-like)
        let tree = TokenTree::with_policy(EvictionPolicy::Filo);

        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(100, 1);
        let tokens3 = make_tokens(200, 1);

        tree.insert_tokens(&tokens1, "tenant1"); // First created
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant1"); // Last created

        // Evict 1 page - should evict tokens3 (newest by creation_time)
        tree.evict_tenant(&Arc::from("tenant1"), 2 * PAGE_SIZE);

        // tokens3 should be evicted (newest creation)
        let result = tree.match_prefix_with_counts(&tokens3);
        assert_eq!(
            result.matched_token_count, 0,
            "tokens3 should be evicted (newest creation)"
        );

        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "tokens1 should remain"
        );
    }

    #[test]
    fn test_eviction_policy_lfu() {
        // LFU: Evict least frequently used (by hit_count)
        let tree = TokenTree::with_policy(EvictionPolicy::Lfu);

        let tokens1 = make_tokens(1, 1);
        let tokens2 = make_tokens(100, 1);
        let tokens3 = make_tokens(200, 1);

        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant1");
        tree.insert_tokens(&tokens3, "tenant1");

        // Access tokens1 many times to increase its hit_count
        for _ in 0..20 {
            let _ = tree.match_prefix_with_counts(&tokens1);
        }

        // Access tokens3 a few times
        for _ in 0..5 {
            let _ = tree.match_prefix_with_counts(&tokens3);
        }

        // tokens2 has the lowest hit_count (only 1 from insert)

        // Evict 1 page - should evict tokens2 (lowest hit_count)
        tree.evict_tenant(&Arc::from("tenant1"), 2 * PAGE_SIZE);

        // tokens2 should be evicted (least frequently used)
        let result = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(
            result.matched_token_count, 0,
            "tokens2 should be evicted (lowest hit count)"
        );

        // tokens1 should remain (highest hit count)
        let result = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(
            result.matched_token_count, PAGE_SIZE,
            "tokens1 should remain (high hit count)"
        );
    }

    #[test]
    fn test_eviction_policy_enum_default() {
        // Default should be LRU
        assert_eq!(EvictionPolicy::default(), EvictionPolicy::Lru);
    }

    #[test]
    fn test_tree_with_policy_getter() {
        let tree_lru = TokenTree::with_policy(EvictionPolicy::Lru);
        assert_eq!(tree_lru.eviction_policy(), EvictionPolicy::Lru);

        let tree_lfu = TokenTree::with_policy(EvictionPolicy::Lfu);
        assert_eq!(tree_lfu.eviction_policy(), EvictionPolicy::Lfu);

        let tree_fifo = TokenTree::with_policy(EvictionPolicy::Fifo);
        assert_eq!(tree_fifo.eviction_policy(), EvictionPolicy::Fifo);
    }

    #[test]
    fn test_tree_with_config() {
        // Test with_config constructor with valid page_size
        let tree = TokenTree::with_config(PAGE_SIZE, EvictionPolicy::Lfu);
        assert_eq!(tree.page_size(), PAGE_SIZE);
        assert_eq!(tree.eviction_policy(), EvictionPolicy::Lfu);

        // Verify defaults: new() should use PAGE_SIZE=16 and LRU
        let tree_default = TokenTree::new();
        assert_eq!(tree_default.page_size(), 16);
        assert_eq!(tree_default.eviction_policy(), EvictionPolicy::Lru);

        // with_policy should use default page_size
        let tree_policy = TokenTree::with_policy(EvictionPolicy::Fifo);
        assert_eq!(tree_policy.page_size(), PAGE_SIZE);
        assert_eq!(tree_policy.eviction_policy(), EvictionPolicy::Fifo);
    }

    #[test]
    #[should_panic(expected = "page_size must be at least 1")]
    fn test_tree_with_config_invalid_page_size() {
        let _tree = TokenTree::with_config(0, EvictionPolicy::Lru);
    }

    #[test]
    fn test_iter_entries_empty_tree() {
        let tree = TokenTree::new();
        let entries: Vec<_> = tree.iter_entries().collect();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_iter_entries_single_insert() {
        let tree = TokenTree::new();
        let tokens: Vec<TokenId> = (0..16).collect();
        tree.insert_tokens(&tokens, "worker-1");

        let entries: Vec<(Vec<TokenId>, Vec<String>)> = tree
            .iter_entries()
            .map(|(p, ts)| (p, ts.into_iter().map(|(t, _)| t.to_string()).collect()))
            .collect();
        let leaf = entries
            .iter()
            .find(|(p, _)| p == &tokens)
            .expect("leaf with the inserted tokens");
        assert_eq!(leaf.1, vec!["worker-1".to_string()]);
    }

    #[test]
    fn test_iter_entries_pre_order_lex_sorted() {
        // Two inserts with a shared prefix produce a split node.
        // Children sort by `TokenPageKey` lexicographically, so
        // the smaller-page-key child emits first.
        let tree = TokenTree::new();
        let prefix: Vec<TokenId> = (0..16).collect();
        let mut a: Vec<TokenId> = prefix.clone();
        a.extend(0..16); // child page = [0..16]
        let mut b: Vec<TokenId> = prefix.clone();
        b.extend(100..116); // child page = [100..116]
        tree.insert_tokens(&a, "wa");
        tree.insert_tokens(&b, "wb");

        let paths: Vec<Vec<TokenId>> = tree.iter_entries().map(|(p, _)| p).collect();
        // Order: shared "0..16" parent appears first via wa being
        // present at index 0 of children; the actual leaves come
        // through in lex order.
        assert!(paths.contains(&a));
        assert!(paths.contains(&b));
        let pos_a = paths.iter().position(|p| p == &a).unwrap();
        let pos_b = paths.iter().position(|p| p == &b).unwrap();
        assert!(pos_a < pos_b, "page-key 0..16 < 100..116");
    }

    /// Build the same sequence of operations two ways — `match_prefix_with_counts`
    /// + `insert_tokens` (the legacy pair) vs the fused `match_and_insert` — and
    /// assert the returned match results and the resulting per-tenant token
    /// counts match step for step. Timestamps are intentionally not compared
    /// (the fused path may consume a different number of monotonic ticks).
    fn assert_fused_matches_pair(ops: &[(Vec<TokenId>, &str)]) {
        let pair = TokenTree::new();
        let fused = TokenTree::new();
        for (tokens, tenant) in ops {
            let r_pair = pair.match_prefix_with_counts(tokens);
            pair.insert_tokens(tokens, tenant);

            let r_fused = fused.match_and_insert(tokens, tenant);

            assert_eq!(
                r_pair.matched_token_count, r_fused.matched_token_count,
                "matched_token_count mismatch for tenant {tenant}"
            );
            assert_eq!(
                r_pair.input_token_count, r_fused.input_token_count,
                "input_token_count mismatch"
            );
            // Tenant is approximate (get_any_tenant), but for these
            // single-tenant-per-path scenarios it is deterministic.
            assert_eq!(
                r_pair.tenant.as_ref(),
                r_fused.tenant.as_ref(),
                "matched tenant mismatch"
            );
            // Compare as sets: tenant-map iteration order is unstable.
            let sorted = |result: &PrefixMatchResult| {
                let mut tenants: Vec<&str> =
                    result.matched_tenants.iter().map(AsRef::as_ref).collect();
                tenants.sort_unstable();
                tenants.into_iter().map(String::from).collect::<Vec<_>>()
            };
            assert_eq!(
                sorted(&r_pair),
                sorted(&r_fused),
                "matched_tenants mismatch for tenant {tenant}"
            );
        }
        assert_eq!(
            pair.get_tenant_token_counts(),
            fused.get_tenant_token_counts(),
            "tenant token counts diverged between pair and fused"
        );
    }

    #[test]
    fn test_match_and_insert_equiv_fresh_and_full_match() {
        let a = make_tokens(1, 3);
        assert_fused_matches_pair(&[
            (a.clone(), "w1"), // fresh insert (single node)
            (a.clone(), "w1"), // exact re-insert (full match continue)
            (a, "w2"),         // same path, different tenant (full match, adds w2)
        ]);
    }

    #[test]
    fn test_match_and_insert_equiv_prefix_of_child_split() {
        let long = make_tokens(1, 3);
        let short = make_tokens(1, 1); // prefix of `long` -> splits the node
        assert_fused_matches_pair(&[(long, "w1"), (short, "w2")]);
    }

    #[test]
    fn test_match_and_insert_equiv_diverge_split() {
        // Shared first page, diverging second page -> partial/diverge split.
        let mut a = make_tokens(1, 1);
        a.extend(make_tokens(100, 1));
        let mut b = make_tokens(1, 1);
        b.extend(make_tokens(200, 1));
        assert_fused_matches_pair(&[(a, "w1"), (b, "w2")]);
    }

    #[test]
    fn test_match_and_insert_equiv_disjoint() {
        let a = make_tokens(1, 2);
        let b = make_tokens(1000, 2);
        assert_fused_matches_pair(&[(a, "w1"), (b, "w2")]);
    }

    #[test]
    fn test_match_and_insert_equiv_short_sequence() {
        // Below PAGE_SIZE: insert is a no-op, match returns 0.
        assert_fused_matches_pair(&[(vec![1, 2, 3], "w1")]);
    }

    /// The `_with` closure form must behave like `match_and_insert` when the
    /// closure always returns the same tenant, and must skip the insert (leaving
    /// the tree unchanged) when it returns `None`.
    #[test]
    fn test_match_and_insert_with_select_and_skip() {
        let tokens = make_tokens(1, 3);

        // Always-insert closure == match_and_insert(tokens, "w1").
        let via_with = TokenTree::new();
        let r = via_with.match_and_insert_with(&tokens, |_| Some("w1"));
        let via_plain = TokenTree::new();
        let r2 = via_plain.match_and_insert(&tokens, "w1");
        assert_eq!(r.matched_token_count, r2.matched_token_count);
        assert_eq!(
            via_with.get_tenant_token_counts(),
            via_plain.get_tenant_token_counts()
        );

        // Seed a tree, then a None-returning closure must NOT mutate it.
        let tree = TokenTree::new();
        tree.insert_tokens(&tokens, "w1");
        let before = tree.get_tenant_token_counts();
        let r = tree.match_and_insert_with(&tokens, |_| None);
        assert_eq!(r.matched_token_count, tokens.len()); // full match observed
        assert_eq!(
            tree.get_tenant_token_counts(),
            before,
            "None selection must not insert"
        );
    }

    /// Concurrency stress test — settles whether the fused
    /// `match_and_insert` / `match_and_insert_with` path corrupts
    /// `tenant_token_count` or loses routes under concurrent node splits
    /// (the review concern).
    ///
    /// Invariant under test (exact, concurrency-independent): a tenant's
    /// count equals the tokens it owns — `align_to_page(input_len)` counted
    /// once on first attach, never again on repeats — no matter how nodes are
    /// split mid-flight or how calls interleave (first-attach is atomic in
    /// `touch_tenant`, per-node `advance`s are frozen at match time, and the
    /// suffix splice re-walks from the fall-off node). Inflation from repeat
    /// traffic or concurrent splits fails the exact-equality assert below.
    /// There is no background eviction in a bare `TokenTree`.
    ///
    /// Both variants run concurrently (inline + replay) on overlapping, nested,
    /// diverging prefixes chosen to force splits.
    #[test]
    fn test_concurrent_match_and_insert_count_and_route_integrity() {
        const N_PREFIXES: usize = 8;
        const N_THREADS: usize = 8;
        const ITERS: usize = 3000;

        // Nested shared head [1,2,3,…] (a shorter prefix matching inside a
        // longer node forces a split) + a disjoint divergent tail per tenant.
        let prefixes: Vec<Vec<TokenId>> = (0..N_PREFIXES)
            .map(|j| {
                let mut v = make_tokens(1, j + 1);
                v.extend(make_tokens(10_000 + (j as u32) * 100, 1));
                v
            })
            .collect();
        let tenants: Vec<String> = (0..N_PREFIXES).map(|j| format!("t{j}")).collect();
        let lens: Vec<usize> = prefixes
            .iter()
            .map(|p| align_to_page(p.len(), PAGE_SIZE))
            .collect();

        let tree = Arc::new(TokenTree::new());
        let prefixes = Arc::new(prefixes);
        let tenants = Arc::new(tenants);

        let handles: Vec<_> = (0..N_THREADS)
            .map(|t| {
                let tree = Arc::clone(&tree);
                let prefixes = Arc::clone(&prefixes);
                let tenants = Arc::clone(&tenants);
                thread::spawn(move || {
                    let mut local = [0usize; N_PREFIXES];
                    for i in 0..ITERS {
                        let j = (t + i) % N_PREFIXES;
                        if t % 2 == 0 {
                            // replay variant (#1: count-drift concern)
                            let tn = tenants[j].as_str();
                            tree.match_and_insert_with(&prefixes[j], |_| Some(tn));
                        } else {
                            // inline variant (#2: touch-order concern)
                            tree.match_and_insert(&prefixes[j], tenants[j].as_str());
                        }
                        local[j] += 1;
                    }
                    local
                })
            })
            .collect();

        let mut total = [0usize; N_PREFIXES];
        for h in handles {
            let local = h.join().expect("worker panicked (deadlock/corruption)");
            for j in 0..N_PREFIXES {
                total[j] += local[j];
            }
        }

        // DEFINITIVE: exact, concurrency-independent count. Repeat traffic
        // must not inflate ownership: each tenant owns its full path exactly
        // once regardless of iteration count or interleaving (first-attach is
        // atomic in touch_tenant).
        let counts = tree.get_tenant_token_counts();
        for j in 0..N_PREFIXES {
            assert!(total[j] > 0);
            let expected = lens[j];
            let actual = counts.get(&tenants[j]).copied().unwrap_or(0);
            assert_eq!(
                actual, expected,
                "tenant {} token count diverged under concurrency: expected {expected}, got {actual}",
                tenants[j]
            );
        }

        // Route integrity: every prefix is still fully cached at the end.
        for j in 0..N_PREFIXES {
            let r = tree.match_prefix_with_counts(&prefixes[j]);
            assert_eq!(
                r.matched_token_count, lens[j],
                "prefix {j} not fully cached after concurrent stress (route loss)"
            );
        }
    }

    #[test]
    fn test_evict_by_size_shared_budget_lru_tenants() {
        let tree = TokenTree::new();

        // 8 tenants, 2 pages each: every tenant is far below the budget on
        // its own, but the tree-wide total is double the budget.
        for t in 0..8u32 {
            let tokens = make_tokens(t * 100_000 + 1, 2);
            tree.insert_tokens(&tokens, &format!("tenant{t}"));
        }
        assert_eq!(tree.total_token_size(), 8 * 2 * PAGE_SIZE);

        let max_size = 4 * 2 * PAGE_SIZE;
        tree.evict_tenant_by_size(max_size);

        assert!(tree.total_token_size() <= max_size);
        // Least-recently-used tenants are evicted first.
        for t in 0..4u32 {
            let tokens = make_tokens(t * 100_000 + 1, 2);
            let result = tree.match_prefix_with_counts(&tokens);
            assert_eq!(
                result.matched_token_count, 0,
                "tenant{t} (LRU) should be evicted"
            );
            let tenant: TenantId = Arc::from(format!("tenant{t}"));
            assert_eq!(tree.tenant_token_size(&tenant), 0);
        }
        for t in 4..8u32 {
            let tokens = make_tokens(t * 100_000 + 1, 2);
            let result = tree.match_prefix_with_counts(&tokens);
            assert_eq!(
                result.matched_token_count,
                2 * PAGE_SIZE,
                "tenant{t} should survive"
            );
        }
    }

    #[test]
    fn test_evict_by_size_under_budget_is_noop() {
        let tree = TokenTree::new();
        for t in 0..4u32 {
            let tokens = make_tokens(t * 100_000 + 1, 2);
            tree.insert_tokens(&tokens, &format!("tenant{t}"));
        }
        let counts_before = tree.get_tenant_token_counts();
        assert_eq!(tree.total_token_size(), 4 * 2 * PAGE_SIZE);

        // At and above the budget: nothing may change.
        tree.evict_tenant_by_size(4 * 2 * PAGE_SIZE);
        tree.evict_tenant_by_size(usize::MAX);

        assert_eq!(tree.get_tenant_token_counts(), counts_before);
        assert_eq!(tree.total_token_size(), 4 * 2 * PAGE_SIZE);
        for t in 0..4u32 {
            let tokens = make_tokens(t * 100_000 + 1, 2);
            assert_eq!(
                tree.match_prefix_with_counts(&tokens).matched_token_count,
                2 * PAGE_SIZE
            );
        }
    }

    #[test]
    fn test_total_count_consistent_with_tenant_sum() {
        fn assert_consistent(tree: &TokenTree) {
            let sum: usize = tree.get_tenant_token_counts().values().sum();
            assert_eq!(
                tree.total_token_size(),
                sum,
                "total must equal per-tenant sum"
            );
        }

        let tree = TokenTree::new();
        assert_consistent(&tree);

        // All insert paths, including a re-insert and a skipped insert.
        for t in 0..6u32 {
            let tokens = make_tokens(t * 100_000 + 1, 3);
            tree.insert_tokens(&tokens, &format!("tenant{t}"));
        }
        tree.insert_tokens(&make_tokens(1, 3), "tenant0");
        tree.match_and_insert(&make_tokens(600_001, 2), "tenant1");
        tree.match_and_insert_with(&make_tokens(700_001, 2), |_| Some("tenant2"));
        tree.match_and_insert_with(&make_tokens(800_001, 2), |_| None);
        assert_consistent(&tree);

        // Per-tenant eviction.
        tree.evict_tenant(&Arc::from("tenant0"), PAGE_SIZE);
        assert_consistent(&tree);

        // Global eviction.
        let total = tree.total_token_size();
        tree.evict_tenant_by_size(total / 2);
        assert_consistent(&tree);
        assert!(tree.total_token_size() <= total / 2);

        // Tenant purge.
        tree.remove_tenant_all(&Arc::from("tenant2"));
        assert_consistent(&tree);

        // No-op eviction and reset.
        tree.evict_tenant_by_size(usize::MAX);
        assert_consistent(&tree);
        tree.clear();
        assert_consistent(&tree);
        assert_eq!(tree.total_token_size(), 0);
    }

    #[test]
    fn test_remove_tenant_all_purges_tenant_and_keeps_others() {
        let tree = TokenTree::new();

        // Shared first page; a distinct second page per tenant.
        let mut tokens1 = make_tokens(1, 1);
        tokens1.extend(make_tokens(100, 1));
        let mut tokens2 = make_tokens(1, 1);
        tokens2.extend(make_tokens(200, 1));
        tree.insert_tokens(&tokens1, "tenant1");
        tree.insert_tokens(&tokens2, "tenant2");

        tree.remove_tenant_all(&Arc::from("tenant1"));

        // Count entry is gone (not merely zero), root entry included.
        assert!(!tree.get_tenant_token_counts().contains_key("tenant1"));
        assert!(!tree.root.tenant_last_access_time.contains_key("tenant1"));

        // tenant1's path only matches the shared page, now owned by tenant2.
        let r = tree.match_prefix_with_counts(&tokens1);
        assert_eq!(r.matched_token_count, PAGE_SIZE);
        assert_eq!(r.tenant.as_ref(), "tenant2");

        // tenant1's sole-owner branch is detached from the shared node.
        assert_eq!(tree.root.children.len(), 1);
        let shared = Arc::clone(tree.root.children.iter().next().unwrap().value());
        assert_eq!(shared.children.len(), 1);

        // tenant2 still matches its full prefix.
        let r = tree.match_prefix_with_counts(&tokens2);
        assert_eq!(r.matched_token_count, 2 * PAGE_SIZE);
        assert_eq!(r.tenant.as_ref(), "tenant2");
    }

    #[test]
    fn test_remove_tenant_all_prunes_unrelated_subtrees() {
        let tree = TokenTree::new();
        let target = make_tokens(1, 2);
        tree.insert_tokens(&target, "target");

        let mut unrelated = make_tokens(10_000, 1);
        for page in 1..=64 {
            unrelated.extend(make_tokens(10_000 + page * 100, 1));
            tree.insert_tokens(&unrelated, "other");
        }

        let tenant = Arc::from("target");
        let mut nodes = Vec::new();
        tree.collect_tenant_nodes_pruned(&tree.root, &tenant, &mut nodes);

        assert_eq!(nodes.len(), 1);

        tree.remove_tenant_all(&tenant);
        assert!(!tree.get_tenant_token_counts().contains_key("target"));
        assert_eq!(
            tree.match_prefix_with_counts(&unrelated)
                .matched_token_count,
            unrelated.len()
        );
    }

    #[test]
    fn test_remove_tenant_all_sole_owner_detaches_subtree() {
        let tree = TokenTree::new();
        // Chain root -> [page] -> [2 pages], all owned by tenant1 only.
        tree.insert_tokens(&make_tokens(500, 1), "tenant1");
        tree.insert_tokens(&make_tokens(500, 3), "tenant1");
        assert!(!tree.root.children.is_empty());

        tree.remove_tenant_all(&Arc::from("tenant1"));

        assert!(tree.root.children.is_empty());
        assert!(!tree.get_tenant_token_counts().contains_key("tenant1"));
        let r = tree.match_prefix_with_counts(&make_tokens(500, 3));
        assert_eq!(r.matched_token_count, 0);
    }

    /// One full round at a non-default page size: page alignment, sub-page
    /// skip, split at a page boundary, and prefix subsumption across
    /// different-length inserts.
    fn roundtrip_at_page_size(page: usize) {
        let tree = TokenTree::with_config(page, EvictionPolicy::default());
        let seq: Vec<TokenId> = (0..(3 * page) as u32).collect();

        // Sub-page: uncacheable.
        tree.insert_tokens(&seq[..page - 1], "w1");
        assert_eq!(tree.match_prefix_with_counts(&seq).matched_token_count, 0);

        // Three pages under w1; a two-page prefix must subsume.
        tree.insert_tokens(&seq, "w1");
        let full = tree.match_prefix_with_counts(&seq);
        assert_eq!(full.matched_token_count, 3 * page);
        assert_eq!(full.tenant.as_ref(), "w1");
        assert_eq!(
            tree.match_prefix_with_counts(&seq[..2 * page])
                .matched_token_count,
            2 * page
        );

        // Diverge after page 1 under w2: forces a split at a page boundary.
        let mut fork: Vec<TokenId> = seq[..page].to_vec();
        fork.extend((0..(2 * page) as u32).map(|i| 900_000 + i));
        tree.insert_tokens(&fork, "w2");
        let forked = tree.match_prefix_with_counts(&fork);
        assert_eq!(forked.matched_token_count, 3 * page);
        assert_eq!(forked.tenant.as_ref(), "w2");
        // Shared first page still matches; unaligned tails truncate.
        let mut partial: Vec<TokenId> = seq[..page].to_vec();
        partial.extend([1_000_001, 1_000_002]);
        assert_eq!(
            tree.match_prefix_with_counts(&partial).matched_token_count,
            page
        );
    }

    #[test]
    fn configurable_page_size_roundtrip() {
        for page in [1, 3, 16, 512, 4096] {
            roundtrip_at_page_size(page);
        }
    }

    #[test]
    fn per_tree_page_size_is_independent() {
        let coarse = TokenTree::with_config(512, EvictionPolicy::default());
        let seq: Vec<TokenId> = (0..600).collect();
        coarse.insert_tokens(&seq, "w1");
        // 600 tokens = one 512 page; the 88-token tail is uncacheable.
        assert_eq!(
            coarse.match_prefix_with_counts(&seq).matched_token_count,
            512
        );

        let fine = TokenTree::new();
        fine.insert_tokens(&seq, "w1");
        // Default 16: aligned to 592.
        assert_eq!(fine.match_prefix_with_counts(&seq).matched_token_count, 592);
    }
}
