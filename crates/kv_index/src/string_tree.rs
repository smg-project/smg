use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    hash::{BuildHasherDefault, Hasher},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Weak,
    },
};

use dashmap::{mapref::entry::Entry, DashMap};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use tracing::debug;

use super::{
    common::{MatchResult, TenantId, MATCHED_TENANTS_CAP},
    RadixTree,
};

type NodeRef = Arc<Node>;

/// Shard counts for DashMaps to balance concurrency vs allocation overhead.
/// Default DashMap uses num_cpus * 4 shards (e.g., 256 on 64-core machines).
///
/// Root node uses higher shard count since ALL requests pass through it.
/// Other nodes use lower count as traffic diverges through the tree.
///
/// This reduces memory by ~90% vs default while maintaining good concurrency.
const ROOT_SHARD_COUNT: usize = 32;
const NODE_SHARD_COUNT: usize = 8;

/// Create a children DashMap for non-root nodes
#[inline]
fn new_children_map() -> DashMap<char, NodeRef, CharHasherBuilder> {
    DashMap::with_hasher_and_shard_amount(CharHasherBuilder::default(), NODE_SHARD_COUNT)
}

/// Create a tenant access time DashMap for non-root nodes
#[inline]
fn new_tenant_map() -> DashMap<TenantId, u64> {
    DashMap::with_shard_amount(NODE_SHARD_COUNT)
}

/// Result of a prefix match operation, including char counts to avoid recomputation.
#[derive(Debug, Clone)]
pub struct PrefixMatchResult {
    /// The tenant that owns the matched prefix (zero-copy)
    pub tenant: TenantId,
    /// Number of characters matched
    pub matched_char_count: usize,
    /// Total number of characters in the input text
    pub input_char_count: usize,
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
        self.matched_char_count
    }

    fn input_count(&self) -> usize {
        self.input_char_count
    }
}

/// A fast identity hasher for single-character keys (used in children DashMap).
/// Since chars have good distribution already, we use identity hashing with mixing.
#[derive(Default)]
struct CharHasher(u64);

impl Hasher for CharHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        // Fast path for 4-byte (char) writes - avoid loop
        if bytes.len() == 4 {
            let val = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            // Mix with golden ratio for better distribution
            self.0 = (val as u64).wrapping_mul(0x9E3779B97F4A7C15);
            return;
        }
        // Fallback for other sizes (shouldn't happen for char keys)
        for &byte in bytes {
            self.0 = self.0.wrapping_mul(0x100000001b3).wrapping_add(byte as u64);
        }
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        // Chars are u32 - use golden ratio multiplication for distribution
        self.0 = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
    }
}

type CharHasherBuilder = BuildHasherDefault<CharHasher>;

/// Advance a string slice by N characters, returning the remaining slice.
/// Returns empty string if n >= char count.
/// Optimized: uses direct byte slicing for ASCII, falls back to char_indices for UTF-8.
#[inline]
fn advance_by_chars(s: &str, n: usize) -> &str {
    if n == 0 {
        return s;
    }
    if n >= s.len() {
        return "";
    }
    // Fast path: if first N bytes are all ASCII, we can slice directly
    let bytes = s.as_bytes();
    if bytes[..n].is_ascii() {
        // Safe: we verified all bytes in [0..n] are ASCII (valid UTF-8 boundary)
        return &s[n..];
    }
    // Slow path: UTF-8 requires char-by-char traversal
    s.char_indices()
        .nth(n)
        .map(|(idx, _)| &s[idx..])
        .unwrap_or("")
}

/// Get the first N characters of a string as a new String.
/// More efficient than chars().take(n).collect() for known bounds.
#[inline]
fn take_chars(s: &str, n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    s.char_indices()
        .nth(n)
        .map(|(idx, _)| s[..idx].to_string())
        .unwrap_or_else(|| s.to_string())
}

/// Node text with cached character count to avoid repeated O(n) chars().count() calls.
#[derive(Debug)]
struct NodeText {
    /// The actual text stored in this node
    text: String,
    /// Cached character count (UTF-8 chars, not bytes)
    char_count: usize,
}

impl NodeText {
    #[inline]
    fn new(text: String) -> Self {
        let char_count = text.chars().count();
        Self { text, char_count }
    }

    #[inline]
    fn empty() -> Self {
        Self {
            text: String::new(),
            char_count: 0,
        }
    }

    #[inline]
    fn char_count(&self) -> usize {
        self.char_count
    }

    #[inline]
    fn as_str(&self) -> &str {
        &self.text
    }

    #[inline]
    fn first_char(&self) -> Option<char> {
        self.text.chars().next()
    }

    /// Split the text at a character boundary, returning the prefix and suffix.
    /// This is more efficient than slice_by_chars as it computes both at once.
    #[inline]
    fn split_at_char(&self, char_idx: usize) -> (NodeText, NodeText) {
        if char_idx == 0 {
            return (NodeText::empty(), self.clone_text());
        }
        if char_idx >= self.char_count {
            return (self.clone_text(), NodeText::empty());
        }

        // Find byte index for the character boundary
        let byte_idx = self
            .text
            .char_indices()
            .nth(char_idx)
            .map(|(i, _)| i)
            .unwrap_or(self.text.len());

        let prefix = NodeText {
            text: self.text[..byte_idx].to_string(),
            char_count: char_idx,
        };
        let suffix = NodeText {
            text: self.text[byte_idx..].to_string(),
            char_count: self.char_count - char_idx,
        };
        (prefix, suffix)
    }

    #[inline]
    fn clone_text(&self) -> NodeText {
        NodeText {
            text: self.text.clone(),
            char_count: self.char_count,
        }
    }
}

impl Clone for NodeText {
    fn clone(&self) -> Self {
        self.clone_text()
    }
}

/// Global tenant string intern pool to avoid repeated allocations.
/// Uses DashMap for concurrent access with minimal contention.
static TENANT_INTERN_POOL: Lazy<DashMap<Arc<str>, ()>> = Lazy::new(DashMap::new);

/// Global epoch counter for LRU ordering.
/// Uses a simple incrementing counter instead of wall clock time.
///
/// Benefits:
/// - No syscall overhead (vs SystemTime::now())
/// - Smaller memory footprint (u64 vs u128)
/// - Perfectly monotonic (no clock skew issues)
///
/// For LRU eviction, relative ordering is all that matters.
static EPOCH_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Get the next epoch value for LRU timestamp ordering.
/// Uses fetch_add for lock-free, monotonically increasing values.
/// Relaxed ordering is sufficient since we only need eventual consistency
/// for approximate LRU behavior.
#[inline]
fn get_epoch() -> u64 {
    EPOCH_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
struct Node {
    /// Children nodes indexed by first character.
    /// Using custom hasher optimized for char keys.
    children: DashMap<char, NodeRef, CharHasherBuilder>,
    /// Node text with cached character count
    text: RwLock<NodeText>,
    /// Per-tenant last access epoch for LRU ordering. Using TenantId (Arc<str>) for cheap cloning.
    tenant_last_access_time: DashMap<TenantId, u64>,
    /// Parent pointer for upward traversal during timestamp updates.
    /// Uses Weak to avoid Arc reference cycles (parent -> child -> parent).
    parent: RwLock<Option<Weak<Node>>>,
    /// Cached last-accessed tenant for O(1) lookup during prefix match.
    /// Avoids O(shards) DashMap iteration in the common case.
    last_tenant: RwLock<Option<TenantId>>,
}

impl Node {
    /// Up to [`MATCHED_TENANTS_CAP`] tenants of this node, in map order.
    fn matched_tenants(&self) -> Vec<TenantId> {
        self.tenant_last_access_time
            .iter()
            .take(MATCHED_TENANTS_CAP)
            .map(|entry| Arc::clone(entry.key()))
            .collect()
    }

    /// Attach `tenant` with `timestamp` unless already present. Returns true
    /// when this call newly attached the tenant — the atomic winner under
    /// concurrency, so char accounting tied to it is exact.
    fn attach_tenant_if_absent(&self, tenant: &TenantId, timestamp: u64) -> bool {
        match self.tenant_last_access_time.entry(Arc::clone(tenant)) {
            Entry::Occupied(_) => false,
            Entry::Vacant(entry) => {
                entry.insert(timestamp);
                true
            }
        }
    }
}

/// True when `node`'s parent pointer currently designates `parent`. A split
/// re-parents a node inside the same `text` write section that truncates it,
/// so walkers check this under their `text` read guard to reject a child
/// re-probed through a stale edge mid-split.
fn node_parent_is(node: &NodeRef, parent: &NodeRef) -> bool {
    node.parent
        .read()
        .as_ref()
        .is_some_and(|w| std::ptr::eq(w.as_ptr(), Arc::as_ptr(parent)))
}

#[derive(Debug)]
pub struct Tree {
    root: NodeRef,
    /// Per-tenant character count for size tracking. Using TenantId for consistency.
    tenant_char_count: DashMap<TenantId, usize>,
    /// Tree-wide char total (sum of `tenant_char_count`); the budget
    /// checked by `evict_tenant_by_size`
    total_char_count: AtomicUsize,
}

// For the heap

struct EvictionEntry {
    timestamp: u64,
    tenant: TenantId,
    node: NodeRef,
}

impl Eq for EvictionEntry {}

#[expect(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for EvictionEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.timestamp.cmp(&other.timestamp))
    }
}

impl Ord for EvictionEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.timestamp.cmp(&other.timestamp)
    }
}

impl PartialEq for EvictionEntry {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp
    }
}

// For char operations
// Note that in rust, `.len()` or slice is operated on the "byte" level. It causes issues for UTF-8 characters because one character might use multiple bytes.
// https://en.wikipedia.org/wiki/UTF-8

/// Count matching prefix characters between two strings.
/// Returns the number of characters that match from the start.
/// Optimized: uses fast byte comparison for ASCII, falls back to char iteration for UTF-8.
#[inline]
fn shared_prefix_count(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();

    // Find common byte prefix length using iterator (potentially SIMD-optimized)
    let common_byte_len = a_bytes
        .iter()
        .zip(b_bytes)
        .position(|(&a_byte, &b_byte)| a_byte != b_byte)
        .unwrap_or_else(|| a_bytes.len().min(b_bytes.len()));

    // If the common byte prefix is all ASCII, byte count == char count
    // Otherwise, fall back to char-by-char comparison for UTF-8 safety
    if a_bytes[..common_byte_len].is_ascii() {
        common_byte_len
    } else {
        shared_prefix_count_chars(a, b)
    }
}

/// Fallback char-by-char comparison for strings with non-ASCII characters.
#[inline]
fn shared_prefix_count_chars(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(a_char, b_char)| a_char == b_char)
        .count()
}

/// Intern tenant ID to avoid repeated allocations.
/// Returns cached Arc<str> if tenant was seen before.
#[inline]
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

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    /*
    Thread-safe multi tenant radix tree

    1. Storing data for multiple tenants (the overlap of multiple radix tree)
    2. Node-level lock to enable concurrent access on nodes
    3. Leaf LRU eviction based on tenant access time

    Optimizations:
    - Cached character counts in NodeText to avoid O(n) chars().count() calls
    - Interned tenant IDs (Arc<str>) for cheap cloning and comparison
    - Batched timestamp updates to reduce syscalls
    - Custom hasher for char keys in children DashMap
    */

    pub fn new() -> Self {
        Tree {
            // Root uses higher shard count since ALL requests pass through it
            root: Arc::new(Node {
                children: DashMap::with_hasher_and_shard_amount(
                    CharHasherBuilder::default(),
                    ROOT_SHARD_COUNT,
                ),
                text: RwLock::new(NodeText::empty()),
                tenant_last_access_time: DashMap::with_shard_amount(ROOT_SHARD_COUNT),
                parent: RwLock::new(None),
                last_tenant: RwLock::new(None),
            }),
            tenant_char_count: DashMap::with_shard_amount(ROOT_SHARD_COUNT),
            total_char_count: AtomicUsize::new(0),
        }
    }

    pub fn insert_text(&self, text: &str, tenant: &str) {
        // Insert text into tree with given tenant
        // Use slice-based traversal to avoid Vec<char> allocation

        // Intern the tenant ID once for reuse
        let tenant_id = intern_tenant(tenant);

        // Ensure tenant exists at root (don't update timestamp - root is never evicted)
        self.root
            .tenant_last_access_time
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

        self.tenant_char_count
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

        // Descend from the root inserting the whole text.
        self.insert_from(Arc::clone(&self.root), text, tenant_id);
    }

    /// Insert `remaining` for `tenant_id` starting the descent at `start` (treated
    /// as the parent of the first edge), reusing [`Self::insert_text`]'s node-split
    /// / tenant-attach / leaf-timestamp logic. The caller owns the root bookkeeping
    /// (`tenant_last_access_time` / `tenant_char_count` entries) and, when resuming
    /// below the root, attaching `tenant_id` to the ancestor nodes on the matched
    /// path (which this method does not touch).
    fn insert_from(&self, start: NodeRef, mut remaining: &str, tenant_id: TenantId) {
        let mut prev = start;

        // Result type to carry state out of the match block
        // This allows the entry guard to be dropped before we update prev
        enum InsertStep {
            Done,
            Continue {
                next_prev: NodeRef,
                advance_chars: usize,
            },
        }

        while let Some(first_char) = remaining.chars().next() {
            // Use entry API for atomic check-and-insert semantics (required for thread safety)
            let step = match prev.children.entry(first_char) {
                Entry::Vacant(entry) => {
                    // No match - create new node with remaining text (this is the leaf)
                    // Compute remaining char count lazily - only here when creating leaf
                    let remaining_char_count = remaining.chars().count();
                    let epoch = get_epoch();

                    let new_node = Arc::new(Node {
                        children: new_children_map(),
                        text: RwLock::new(NodeText::new(remaining.to_string())),
                        tenant_last_access_time: new_tenant_map(),
                        parent: RwLock::new(Some(Arc::downgrade(&prev))),
                        last_tenant: RwLock::new(Some(Arc::clone(&tenant_id))),
                    });

                    // Attach tenant to the new leaf node with timestamp
                    self.add_tenant_chars(&tenant_id, remaining_char_count);
                    new_node
                        .tenant_last_access_time
                        .insert(Arc::clone(&tenant_id), epoch);

                    entry.insert(new_node);
                    InsertStep::Done
                }

                Entry::Occupied(mut entry) => {
                    let matched_node = entry.get().clone();

                    let matched_node_text = matched_node.text.read();
                    let matched_node_text_count = matched_node_text.char_count();
                    let matched_node_text_str = matched_node_text.as_str();

                    // Use slice-based comparison - no allocation
                    let shared_count = shared_prefix_count(remaining, matched_node_text_str);

                    if shared_count < matched_node_text_count {
                        // Split the matched node
                        let (matched_text, contracted_text) =
                            matched_node_text.split_at_char(shared_count);
                        let matched_text_count = shared_count;

                        // Drop read lock before creating new node
                        drop(matched_node_text);

                        let Some(first_new_char) = contracted_text.first_char() else {
                            // split_at_char with shared_count < char_count guarantees non-empty suffix
                            return;
                        };

                        // Truncate to the suffix, clone the tenant map, and
                        // re-parent under the new prefix node in one `text`
                        // write section: lock-free walkers read the text and
                        // validate the parent pointer under a `text` read
                        // guard, so they observe the pre-split node or the
                        // post-split one — never a truncated text still
                        // reachable through the stale edge (which would
                        // silently shift their depth on self-similar content).
                        // Replayers likewise credit the live char count under
                        // the read guard, so the clone inherits exactly the
                        // tenants credited for the pre-split span.
                        let mut matched_node_text_write = matched_node.text.write();
                        let tenant_map = matched_node.tenant_last_access_time.clone();
                        let last_tenant = matched_node.last_tenant.read().clone();
                        let new_node = Arc::new(Node {
                            text: RwLock::new(matched_text),
                            children: new_children_map(),
                            parent: RwLock::new(Some(Arc::downgrade(&prev))),
                            tenant_last_access_time: tenant_map,
                            last_tenant: RwLock::new(last_tenant),
                        });
                        *matched_node.parent.write() = Some(Arc::downgrade(&new_node));
                        *matched_node_text_write = contracted_text;
                        drop(matched_node_text_write);

                        new_node
                            .children
                            .insert(first_new_char, Arc::clone(&matched_node));

                        // Attach before publication (intermediate - no timestamp
                        // update): the node is unreachable, so the return is
                        // authoritative and the prefix is credited exactly once
                        // even against a concurrent same-tenant replay that
                        // raced the clone above.
                        if new_node.attach_tenant_if_absent(&tenant_id, 0) {
                            self.add_tenant_chars(&tenant_id, matched_text_count);
                        }

                        entry.insert(Arc::clone(&new_node));

                        InsertStep::Continue {
                            next_prev: new_node,
                            advance_chars: shared_count,
                        }
                    } else {
                        // Full match - move to next node (intermediate - no timestamp update)
                        drop(matched_node_text);

                        // Ensure tenant exists at this intermediate node,
                        // counting only on first attach (atomic) so a
                        // concurrent same-tenant attach can't double-credit.
                        if matched_node.attach_tenant_if_absent(&tenant_id, 0) {
                            self.add_tenant_chars(&tenant_id, matched_node_text_count);
                        }

                        InsertStep::Continue {
                            next_prev: matched_node,
                            advance_chars: shared_count,
                        }
                    }
                }
            };

            // Entry guard is now dropped - safe to update prev
            match step {
                InsertStep::Done => return, // New leaf created with timestamp, we're done
                InsertStep::Continue {
                    next_prev,
                    advance_chars,
                } => {
                    prev = next_prev;
                    remaining = advance_by_chars(remaining, advance_chars);
                }
            }
        }

        // Loop exited normally (remaining empty) - prev is the leaf node
        // Update its timestamp for LRU ordering
        let epoch = get_epoch();
        prev.tenant_last_access_time
            .insert(Arc::clone(&tenant_id), epoch);
    }

    /// Performs prefix matching and returns detailed result with char counts.
    /// Optimized: no string allocations, deferred char counting.
    pub fn match_prefix_with_counts(&self, text: &str) -> PrefixMatchResult {
        let mut remaining = text;
        let mut matched_chars = 0;
        let mut prev = Arc::clone(&self.root);

        while let Some(first_char) = remaining.chars().next() {
            let child_node = prev.children.get(&first_char).map(|e| e.value().clone());

            if let Some(matched_node) = child_node {
                let matched_text_guard = matched_node.text.read();
                // Stale-edge check: a split re-parents the child inside the
                // same `text` write section that truncates it, so a parent no
                // longer equal to `prev` means this text may start deeper than
                // this edge (on self-similar content the suffix would still
                // compare equal and silently shift the walk). Re-probe the
                // edge until the split publishes.
                if !node_parent_is(&matched_node, &prev) {
                    drop(matched_text_guard);
                    std::hint::spin_loop();
                    continue;
                }
                let matched_node_text_count = matched_text_guard.char_count();

                // Use slice-based comparison - no allocation
                let shared_count = shared_prefix_count(remaining, matched_text_guard.as_str());
                drop(matched_text_guard);

                if shared_count == matched_node_text_count {
                    // Full match with current node's text, continue to next node
                    matched_chars += shared_count;
                    remaining = advance_by_chars(remaining, shared_count);
                    prev = matched_node;
                } else {
                    // Partial match - still use this node for tenant selection
                    matched_chars += shared_count;
                    prev = matched_node;
                    break;
                }
            } else {
                // No match found, stop here
                break;
            }
        }

        let curr = prev;

        // Try cached tenant first (O(1)) before falling back to O(shards) DashMap iteration.
        // The cache is valid if the tenant still exists in tenant_last_access_time.
        let tenant: TenantId = {
            let cached = curr.last_tenant.read();
            if let Some(ref t) = *cached {
                if curr.tenant_last_access_time.contains_key(t.as_ref()) {
                    Arc::clone(t)
                } else {
                    drop(cached);
                    // Cache stale, fall back to iteration and update cache
                    let t = curr
                        .tenant_last_access_time
                        .iter()
                        .next()
                        .map(|kv| Arc::clone(kv.key()))
                        .unwrap_or_else(|| intern_tenant("empty"));
                    *curr.last_tenant.write() = Some(Arc::clone(&t));
                    t
                }
            } else {
                drop(cached);
                // No cache, iterate and populate cache
                let t = curr
                    .tenant_last_access_time
                    .iter()
                    .next()
                    .map(|kv| Arc::clone(kv.key()))
                    .unwrap_or_else(|| intern_tenant("empty"));
                *curr.last_tenant.write() = Some(Arc::clone(&t));
                t
            }
        };

        // Update timestamp probabilistically (1 in 8 matches) to reduce DashMap contention.
        // LRU eviction doesn't need perfect accuracy - approximate timestamps suffice.
        // Skip the update for the synthetic "empty" tenant to avoid polluting the tree
        // with a tenant that was never inserted via insert_text.
        let epoch = get_epoch();
        if epoch & 0x7 == 0 && tenant.as_ref() != "empty" {
            curr.tenant_last_access_time
                .insert(Arc::clone(&tenant), epoch);
        }

        // Compute input char count directly from input text.
        // This is equivalent to matched_chars + remaining.chars().count() but avoids
        // needing to track remaining precisely through the traversal.
        let input_char_count = text.chars().count();

        PrefixMatchResult {
            tenant,
            matched_char_count: matched_chars,
            input_char_count,
            matched_tenants: if matched_chars > 0 {
                curr.matched_tenants()
            } else {
                Vec::new()
            },
        }
    }

    /// Read-only resolution of a node's "owning" tenant, mirroring the tenant
    /// pick in [`Self::match_prefix_with_counts`] **without** the cache-populate
    /// or probabilistic timestamp side effects. Returns the tenant plus whether
    /// the slow (iteration) path was taken — the caller replays the original's
    /// single deferred side effect on the final node.
    ///
    /// Used by [`Self::match_and_insert`] so the match tenant is resolved from a
    /// node's state **before** the fused insert adds the inserting tenant to it,
    /// reproducing the original "match runs fully before insert" ordering for the
    /// (otherwise non-deterministic) slow-path tenant pick.
    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the tree; mirrors match_prefix_with_counts resolution"
    )]
    fn resolve_tenant_readonly(&self, node: &NodeRef) -> (TenantId, bool) {
        let cached = node.last_tenant.read();
        if let Some(ref t) = *cached {
            if node.tenant_last_access_time.contains_key(t.as_ref()) {
                return (Arc::clone(t), false);
            }
        }
        drop(cached);
        let t = node
            .tenant_last_access_time
            .iter()
            .next()
            .map(|kv| Arc::clone(kv.key()))
            .unwrap_or_else(|| intern_tenant("empty"));
        (t, true)
    }

    /// Combined match + insert for a known `tenant` in a single descent.
    ///
    /// Thin wrapper over [`Self::match_and_insert_with`] for callers that already
    /// know the tenant to insert for before matching (e.g. the imbalanced
    /// min-load path, which routes by worker load, not cache affinity). Returns
    /// the match result for any cache bookkeeping the caller needs.
    pub fn match_and_insert(&self, text: &str, tenant: &str) -> PrefixMatchResult {
        self.match_and_insert_with(text, move |_| Some(tenant))
    }

    /// String analogue of `TokenTree::match_and_insert_with`: match in a single
    /// descent, then insert for the tenant `select` returns from the
    /// [`PrefixMatchResult`] (`None` skips the insert). The matched prefix is
    /// compared once. The match runs fully before any insert mutation, so the
    /// tenant pick observes the un-polluted tree; only insert's timestamps shift
    /// by a few ticks, immaterial to LRU.
    pub fn match_and_insert_with<'t, F>(&self, text: &str, select: F) -> PrefixMatchResult
    where
        F: FnOnce(&PrefixMatchResult) -> Option<&'t str>,
    {
        // ---- Phase 1: MATCH descent (mirrors match_prefix_with_counts) ----
        // Record the full-match nodes (for ancestor re-attach) and the fall-off
        // point (node + remaining slice) for the suffix splice. No mutation here
        // beyond match's own deferred touch below, so the tenant pick sees the
        // un-polluted tree exactly like the standalone match.
        let mut remaining = text;
        let mut matched_chars = 0usize;
        let mut current = Arc::clone(&self.root);
        // The node match resolves its tenant on (its final `curr`): the deepest
        // full-match node, the partial child, or the root if nothing matched.
        let mut match_curr = Arc::clone(&self.root);
        // Every full-match node we descended through, in order.
        // Pre-allocated; most matched paths are well under this depth.
        let mut path: Vec<NodeRef> = Vec::with_capacity(16);

        while let Some(first_char) = remaining.chars().next() {
            let child_node = current.children.get(&first_char).map(|e| e.value().clone());

            let Some(matched_node) = child_node else {
                // No child for this char: match stops at `current`.
                break;
            };

            let matched_text_guard = matched_node.text.read();
            // Stale-edge check (see `match_prefix_with_counts`).
            if !node_parent_is(&matched_node, &current) {
                drop(matched_text_guard);
                std::hint::spin_loop();
                continue;
            }
            let matched_node_text_count = matched_text_guard.char_count();
            let shared_count = shared_prefix_count(remaining, matched_text_guard.as_str());
            drop(matched_text_guard);

            if shared_count == matched_node_text_count {
                // Full match -> continue. Record for ancestor re-attach.
                matched_chars += shared_count;
                path.push(Arc::clone(&matched_node));
                remaining = advance_by_chars(remaining, shared_count);
                current = Arc::clone(&matched_node);
                match_curr = matched_node;
            } else {
                // Partial match: match stops, resolving on this child node.
                matched_chars += shared_count;
                match_curr = matched_node;
                // `current` stays the parent — the splice re-probes it and
                // splits the partial child exactly like `insert_text`.
                break;
            }
        }

        // ---- Match side effect + result (verbatim match_prefix_with_counts) ----
        let (match_tenant, match_slow) = self.resolve_tenant_readonly(&match_curr);
        let result = self.finish_match_and_insert(
            &match_curr,
            &match_tenant,
            match_slow,
            matched_chars,
            text,
        );

        // ---- Decide the insert tenant from the match result ----
        let Some(tenant) = select(&result) else {
            return result;
        };
        // Intern through the shared pool so the stored Arc dedups (matches
        // insert_text's interning).
        let tenant_id = intern_tenant(tenant);

        // ---- Phase 2: INSERT for `tenant_id` without re-walking the prefix ----
        // Root bookkeeping (mirrors insert_text).
        self.root
            .tenant_last_access_time
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);
        self.tenant_char_count
            .entry(Arc::clone(&tenant_id))
            .or_insert(0);

        // Re-attach the inserting tenant to every full-match ancestor node, the
        // epoch-0 intermediate attach `insert_text` performs while descending.
        for node in &path {
            // Credit the live char count under the `text` read guard: a
            // concurrent split truncates the node and clones its tenant map in
            // one `text` write section, so this ordering decides both what the
            // tenant now owns and whether the clone inherits it. The
            // match-time count would over-credit a node truncated since.
            // Count only on first attach (atomic).
            let text_guard = node.text.read();
            if node.attach_tenant_if_absent(&tenant_id, 0) {
                self.add_tenant_chars(&tenant_id, text_guard.char_count());
            }
        }

        if remaining.is_empty() {
            // Loop-end: `current` is the final leaf; give it the real timestamp
            // (insert_text's tail). It already received the epoch-0 attach above
            // if it is on the path.
            let epoch = get_epoch();
            current
                .tenant_last_access_time
                .insert(Arc::clone(&tenant_id), epoch);
        } else {
            // Fall-off: splice only the unmatched suffix below `current`,
            // reusing insert_text's exact loop (handles split-continue + the
            // final-leaf real timestamp). The matched prefix is not re-walked.
            self.insert_from(current, remaining, tenant_id);
        }

        result
    }

    /// Replay `match_prefix_with_counts`'s single deferred side effect (populate
    /// the `last_tenant` cache on the slow path, then a probabilistic 1/8
    /// timestamp touch) on the resolved final node, and build the match result.
    /// Factored out so [`Self::match_and_insert_with`] applies it identically to
    /// the standalone `match_prefix_with_counts` tail.
    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the tree; mirrors match_prefix_with_counts tail"
    )]
    fn finish_match_and_insert(
        &self,
        match_node: &NodeRef,
        tenant: &TenantId,
        took_slow_path: bool,
        matched_chars: usize,
        text: &str,
    ) -> PrefixMatchResult {
        // On the slow path the original resolution populates the cache.
        if took_slow_path {
            *match_node.last_tenant.write() = Some(Arc::clone(tenant));
        }

        // Probabilistic (1 in 8) timestamp touch on the resolved tenant, skipping
        // the synthetic "empty" tenant — identical to match_prefix_with_counts.
        let epoch = get_epoch();
        if epoch & 0x7 == 0 && tenant.as_ref() != "empty" {
            match_node
                .tenant_last_access_time
                .insert(Arc::clone(tenant), epoch);
        }

        let input_char_count = text.chars().count();

        PrefixMatchResult {
            tenant: Arc::clone(tenant),
            matched_char_count: matched_chars,
            input_char_count,
            matched_tenants: if matched_chars > 0 {
                match_node.matched_tenants()
            } else {
                Vec::new()
            },
        }
    }

    /// Legacy prefix_match API for backward compatibility.
    /// Note: This computes matched_text which has allocation overhead.
    pub fn prefix_match_legacy(&self, text: &str) -> (String, String) {
        let result = self.match_prefix_with_counts(text);
        let matched_text = take_chars(text, result.matched_char_count);
        (matched_text, result.tenant.to_string())
    }

    pub fn prefix_match_tenant(&self, text: &str, tenant: &str) -> String {
        // Use slice-based traversal - no Vec<char> allocation

        // Intern tenant ID once for efficient lookups
        let tenant_id = intern_tenant(tenant);

        let mut remaining = text;
        let mut matched_chars = 0;
        let mut prev = Arc::clone(&self.root);

        while let Some(first_char) = remaining.chars().next() {
            let child_node = prev.children.get(&first_char).map(|e| e.value().clone());

            if let Some(matched_node) = child_node {
                // Only continue matching if this node belongs to the specified tenant
                if !matched_node
                    .tenant_last_access_time
                    .contains_key(tenant_id.as_ref())
                {
                    break;
                }

                let matched_text_guard = matched_node.text.read();
                // Stale-edge check (see `match_prefix_with_counts`).
                if !node_parent_is(&matched_node, &prev) {
                    drop(matched_text_guard);
                    std::hint::spin_loop();
                    continue;
                }
                let matched_node_text_count = matched_text_guard.char_count();

                // Use slice-based comparison - no allocation
                let shared_count = shared_prefix_count(remaining, matched_text_guard.as_str());
                drop(matched_text_guard);

                if shared_count == matched_node_text_count {
                    // Full match with current node's text, continue to next node
                    matched_chars += shared_count;
                    remaining = advance_by_chars(remaining, shared_count);
                    prev = matched_node;
                } else {
                    // Partial match - still use this node for timestamp update
                    matched_chars += shared_count;
                    prev = matched_node;
                    break;
                }
            } else {
                // No match found, stop here
                break;
            }
        }

        let curr = prev;

        // Only update timestamp if we found a match for the specified tenant.
        // Update matched node only - ancestor propagation is unnecessary.
        if curr
            .tenant_last_access_time
            .contains_key(tenant_id.as_ref())
        {
            let epoch = get_epoch();
            curr.tenant_last_access_time
                .insert(Arc::clone(&tenant_id), epoch);
        }

        // Build result from original input using char count
        take_chars(text, matched_chars)
    }

    /// Return the list of tenants for which this node is a leaf.
    /// A tenant is a leaf at this node if no children have that tenant.
    fn leaf_of(node: &NodeRef) -> Vec<TenantId> {
        let mut candidates: HashMap<TenantId, bool> = node
            .tenant_last_access_time
            .iter()
            .map(|entry| (Arc::clone(entry.key()), true))
            .collect();

        for child in &node.children {
            for tenant in &child.value().tenant_last_access_time {
                // Mark as non-leaf if any child has this tenant
                candidates.insert(Arc::clone(tenant.key()), false);
            }
        }

        candidates
            .into_iter()
            .filter(|(_, is_leaf)| *is_leaf)
            .map(|(tenant, _)| tenant)
            .collect()
    }

    /// Evict cache entries until the tree-wide char total is at or under
    /// `max_size`.
    ///
    /// The budget is shared across all tenants of this tree: eviction triggers
    /// on the tree-wide total (no single tenant has to exceed anything) and
    /// removes leaves in LRU order across tenants, least recently used first.
    pub fn evict_tenant_by_size(&self, max_size: usize) {
        if self.total_char_size() <= max_size {
            return;
        }

        // Collect leaves across all tenants
        let mut stack = vec![Arc::clone(&self.root)];
        let mut pq = BinaryHeap::new();

        while let Some(curr) = stack.pop() {
            for child in &curr.children {
                stack.push(Arc::clone(child.value()));
            }

            // Root is never eligible for eviction
            if Arc::ptr_eq(&curr, &self.root) {
                continue;
            }

            // Add leaves to priority queue
            for tenant in Tree::leaf_of(&curr) {
                if let Some(timestamp) = curr.tenant_last_access_time.get(tenant.as_ref()) {
                    pq.push(Reverse(EvictionEntry {
                        timestamp: *timestamp,
                        tenant: Arc::clone(&tenant),
                        node: Arc::clone(&curr),
                    }));
                }
            }
        }

        debug!("Before eviction - Used size per tenant:");
        for entry in &self.tenant_char_count {
            debug!("Tenant: {}, Size: {}", entry.key(), entry.value());
        }

        // Process eviction until the tree-wide total obeys the budget
        while self.total_char_size() > max_size {
            let Some(Reverse(entry)) = pq.pop() else {
                break;
            };
            let EvictionEntry { tenant, node, .. } = entry;

            // Root is never eligible for eviction (re-enters via promotion)
            if Arc::ptr_eq(&node, &self.root) {
                continue;
            }

            // Verify this node is still a leaf for this tenant (may have changed)
            // A node is a leaf for a tenant if no children have that tenant
            let is_still_leaf = node.tenant_last_access_time.contains_key(tenant.as_ref())
                && !node.children.iter().any(|child| {
                    child
                        .value()
                        .tenant_last_access_time
                        .contains_key(tenant.as_ref())
                });
            if !is_still_leaf {
                continue;
            }

            // Decrement when removing tenant from node
            let node_len = node.text.read().char_count();
            self.sub_tenant_chars(&tenant, node_len);

            // Remove tenant from node
            node.tenant_last_access_time.remove(tenant.as_ref());

            // Get parent reference outside of the borrow scope
            // Use Weak::upgrade() to get a strong reference if the parent still exists
            let parent_opt = node.parent.read().as_ref().and_then(Weak::upgrade);

            // Remove empty nodes
            if node.children.is_empty() && node.tenant_last_access_time.is_empty() {
                if let Some(ref parent) = parent_opt {
                    if let Some(fc) = node.text.read().first_char() {
                        parent.children.remove(&fc);
                    }
                }
            }

            // If parent has this tenant and no other children have it,
            // parent becomes a new leaf - add to priority queue
            if let Some(ref parent) = parent_opt {
                if parent.tenant_last_access_time.contains_key(tenant.as_ref()) {
                    let has_child_with_tenant = parent.children.iter().any(|child| {
                        child
                            .value()
                            .tenant_last_access_time
                            .contains_key(tenant.as_ref())
                    });

                    if !has_child_with_tenant {
                        // Add parent to priority queue as new leaf
                        if let Some(timestamp) = parent.tenant_last_access_time.get(tenant.as_ref())
                        {
                            pq.push(Reverse(EvictionEntry {
                                timestamp: *timestamp,
                                tenant: Arc::clone(&tenant),
                                node: Arc::clone(parent),
                            }));
                        }
                    }
                }
            }
        }

        debug!("After eviction - Used size per tenant:");
        for entry in &self.tenant_char_count {
            debug!("Tenant: {}, Size: {}", entry.key(), entry.value());
        }
    }

    pub fn get_tenant_char_count(&self) -> HashMap<String, usize> {
        self.tenant_char_count
            .iter()
            .map(|entry| (entry.key().to_string(), *entry.value()))
            .collect()
    }

    pub fn get_used_size_per_tenant(&self) -> HashMap<String, usize> {
        // perform a DFS to traverse all nodes and calculate the total size used by each tenant

        let mut used_size_per_tenant: HashMap<String, usize> = HashMap::new();
        let mut stack = vec![Arc::clone(&self.root)];

        while let Some(curr) = stack.pop() {
            // Use cached char count instead of chars().count()
            let text_count = curr.text.read().char_count();

            for tenant in &curr.tenant_last_access_time {
                let size = used_size_per_tenant
                    .entry(tenant.key().to_string())
                    .or_insert(0);
                *size += text_count;
            }

            for child in &curr.children {
                stack.push(Arc::clone(child.value()));
            }
        }

        used_size_per_tenant
    }

    /// Evict entries for a specific tenant to reduce to max_chars.
    pub fn evict_by_tenant(&self, tenant: &TenantId, max_chars: usize) {
        let current_count = self.tenant_char_count.get(tenant).map(|v| *v).unwrap_or(0);

        if current_count <= max_chars {
            return;
        }

        // Use existing eviction logic by temporarily setting a target
        // We need to evict (current_count - max_chars) chars
        self.evict_tenant_entries(tenant, current_count - max_chars);
    }

    /// Remove a tenant from all nodes in the tree (including root), detach
    /// nodes the removal emptied, and drop the tenant's char-count entry.
    /// Used for mesh eviction propagation and worker removal — size-based
    /// eviction never fires for a tenant whose count no longer grows.
    pub fn remove_tenant_all(&self, tenant_id: &TenantId) {
        let expected_chars = self.tenant_char_size(tenant_id);

        // collect_tenant_nodes skips root (root is never LRU-evicted),
        // but global removal must include it.
        self.remove_tenant_from_node(&self.root, tenant_id);
        if expected_chars == 0 {
            self.tenant_char_count.remove(tenant_id);
            return;
        }

        let mut nodes: Vec<(NodeRef, u64)> = Vec::new();
        self.collect_tenant_nodes_pruned(&self.root, tenant_id, &mut nodes);

        // Normal inserts attach a tenant to every ancestor, allowing the
        // pruned walk above to skip unrelated subtrees. Older snapshots and
        // targeted eviction may not preserve that prefix-closed invariant;
        // the maintained byte count detects that shape and falls back to the
        // complete walk for correctness.
        let collected_chars: usize = nodes
            .iter()
            .map(|(node, _)| node.text.read().char_count())
            .sum();
        if collected_chars != expected_chars {
            nodes.clear();
            self.collect_tenant_nodes(&self.root, tenant_id, &mut nodes);
        }
        for (node, _) in &nodes {
            self.remove_tenant_from_node(node, tenant_id);
            Self::detach_empty_ancestors(node);
        }
        if let Some((_, count)) = self.tenant_char_count.remove(tenant_id) {
            self.total_char_count.fetch_sub(count, Ordering::Relaxed);
        }
    }

    /// Fast tenant walk for the common prefix-closed ownership shape. If a
    /// child does not contain the tenant, none of its descendants can contain
    /// it either, so the entire unrelated subtree can be skipped.
    fn collect_tenant_nodes_pruned(
        &self,
        node: &NodeRef,
        tenant_id: &TenantId,
        result: &mut Vec<(NodeRef, u64)>,
    ) {
        for child in &node.children {
            let child = child.value();
            let Some(ts) = child.tenant_last_access_time.get(tenant_id) else {
                continue;
            };
            result.push((Arc::clone(child), *ts));
            drop(ts);
            self.collect_tenant_nodes_pruned(child, tenant_id, result);
        }
    }

    /// Detach `node` from its parent when it has no tenants and no children
    /// (the `evict_tenant_by_size` empty-node rule), walking up so ancestors
    /// emptied by the detach are removed too. The root (no parent) is never
    /// detached.
    fn detach_empty_ancestors(node: &NodeRef) {
        let mut current = Arc::clone(node);
        while current.children.is_empty() && current.tenant_last_access_time.is_empty() {
            let Some(parent) = current.parent.read().as_ref().and_then(Weak::upgrade) else {
                break;
            };
            if let Some(fc) = current.text.read().first_char() {
                parent.children.remove(&fc);
            }
            current = parent;
        }
    }

    /// Evict a specific number of chars for a tenant using LRU ordering.
    fn evict_tenant_entries(&self, tenant_id: &TenantId, chars_to_evict: usize) {
        if chars_to_evict == 0 {
            return;
        }

        let mut evicted = 0;

        // Collect nodes with timestamps for LRU eviction
        let mut nodes_with_time: Vec<(NodeRef, u64)> = Vec::new();
        self.collect_tenant_nodes(&self.root, tenant_id, &mut nodes_with_time);

        // Sort by timestamp (oldest first)
        nodes_with_time.sort_by_key(|(_, ts)| *ts);

        for (node, _) in nodes_with_time {
            if evicted >= chars_to_evict {
                break;
            }

            let node_chars = node.text.read().char_count();
            if self.remove_tenant_from_node(&node, tenant_id) {
                evicted += node_chars;
            }
        }

        // Update tenant char count and tree-wide total
        self.sub_tenant_chars(tenant_id, evicted);
    }

    fn collect_tenant_nodes(
        &self,
        node: &NodeRef,
        tenant_id: &TenantId,
        result: &mut Vec<(NodeRef, u64)>,
    ) {
        // Skip root node as it should never be evicted
        if !Arc::ptr_eq(node, &self.root) {
            if let Some(ts) = node.tenant_last_access_time.get(tenant_id) {
                result.push((Arc::clone(node), *ts));
            }
        }

        for child in &node.children {
            self.collect_tenant_nodes(child.value(), tenant_id, result);
        }
    }

    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the tree instance; keeps API consistent with collect_tenant_nodes"
    )]
    fn remove_tenant_from_node(&self, node: &NodeRef, tenant_id: &TenantId) -> bool {
        node.tenant_last_access_time.remove(tenant_id).is_some()
    }

    /// Get the char count for a specific tenant.
    pub fn tenant_char_size(&self, tenant: &TenantId) -> usize {
        self.tenant_char_count.get(tenant).map(|v| *v).unwrap_or(0)
    }

    /// Tree-wide char total (sum of per-tenant counts).
    pub fn total_char_size(&self) -> usize {
        self.total_char_count.load(Ordering::Relaxed)
    }

    /// Fold `chars` into a tenant's count and the tree-wide total.
    fn add_tenant_chars(&self, tenant_id: &TenantId, chars: usize) {
        if chars == 0 {
            return;
        }
        self.tenant_char_count
            .entry(Arc::clone(tenant_id))
            .and_modify(|count| *count += chars)
            .or_insert(chars);
        self.total_char_count.fetch_add(chars, Ordering::Relaxed);
    }

    /// Subtract evicted `chars` from a tenant's count and the tree-wide
    /// total, clamped to the tenant's current count so the two stay in sync.
    fn sub_tenant_chars(&self, tenant_id: &TenantId, chars: usize) {
        if chars == 0 {
            return;
        }
        if let Some(mut count) = self.tenant_char_count.get_mut(tenant_id.as_ref()) {
            let removed = chars.min(*count);
            *count -= removed;
            self.total_char_count.fetch_sub(removed, Ordering::Relaxed);
        }
    }

    /// Clear the tree to empty state. Not synchronized with concurrent
    /// mutation: callers must quiesce writers first.
    pub fn clear(&self) {
        // Clear root's children
        self.root.children.clear();
        // Clear root's tenant timestamps (except keep structure)
        self.root.tenant_last_access_time.clear();
        // Clear tenant char counts
        self.tenant_char_count.clear();
        self.total_char_count.store(0, Ordering::Relaxed);
        // Reset root text
        *self.root.text.write() = NodeText::new(String::new());
    }

    fn node_to_string(node: &NodeRef, prefix: &str, is_last: bool) -> String {
        let mut result = String::new();

        // Add prefix and branch character
        result.push_str(prefix);
        result.push_str(if is_last { "└── " } else { "├── " });

        // Add node text
        let node_text = node.text.read();
        result.push_str(&format!("'{}' [", node_text.as_str()));

        // Add tenant information with epoch values
        let mut tenant_info = Vec::new();
        for entry in &node.tenant_last_access_time {
            let tenant_id = entry.key();
            let epoch = entry.value();
            tenant_info.push(format!("{tenant_id} | epoch:{epoch}"));
        }

        result.push_str(&tenant_info.join(", "));
        result.push_str("]\n");

        // Process children
        let children: Vec<_> = node.children.iter().collect();
        let child_count = children.len();

        for (i, entry) in children.iter().enumerate() {
            let is_last_child = i == child_count - 1;
            let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

            result.push_str(&Tree::node_to_string(
                entry.value(),
                &new_prefix,
                is_last_child,
            ));
        }

        result
    }

    #[expect(
        clippy::print_stdout,
        reason = "diagnostic method intended for debugging and test output"
    )]
    pub fn pretty_print(&self) {
        if self.root.children.is_empty() {
            return;
        }

        let mut result = String::new();
        let children: Vec<_> = self.root.children.iter().collect();
        let child_count = children.len();

        for (i, entry) in children.iter().enumerate() {
            let is_last = i == child_count - 1;
            result.push_str(&Tree::node_to_string(entry.value(), "", is_last));
        }

        println!("{result}");
    }

    /// Create a compact snapshot of the tree for mesh synchronization.
    ///
    /// Walks the tree depth-first (pre-order) and emits each node's edge
    /// label + tenant list.  Shared prefixes are stored once — a tree
    /// with 2048 entries sharing 80% prefixes serializes to ~2-4 MB
    /// instead of ~40 MB as flat insert operations.
    ///
    /// # Concurrency
    ///
    /// This traversal is NOT atomic — concurrent `insert_text` calls may
    /// split or replace nodes during the walk.  The per-node DashMap and
    /// RwLock guards ensure individual reads are safe, but the snapshot
    /// may reflect a mix of pre-split and post-split state.  This is
    /// acceptable for mesh sync (eventual consistency).
    pub fn snapshot(&self) -> crate::snapshot::TreeSnapshot {
        let mut nodes = Vec::new();
        Self::snapshot_node(&self.root, &mut nodes);
        crate::snapshot::TreeSnapshot { nodes }
    }

    fn snapshot_node(node: &Node, out: &mut Vec<crate::snapshot::SnapshotNode>) {
        let text = node.text.read();
        let edge = text.as_str().to_string();
        drop(text);

        // Collect tenants with their epochs
        let tenants: Vec<(String, u64)> = node
            .tenant_last_access_time
            .iter()
            .map(|entry| (entry.key().to_string(), *entry.value()))
            .collect();

        // Capture children list BEFORE writing child_count to ensure
        // the count matches the actual children we visit.
        let mut children: Vec<(char, NodeRef)> = node
            .children
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect();
        children.sort_by_key(|(c, _)| *c);

        out.push(crate::snapshot::SnapshotNode {
            edge,
            tenants,
            child_count: children.len() as u32,
        });

        for (_, child) in children {
            Self::snapshot_node(&child, out);
        }
    }

    /// Lazily walk the tree in pre-order, yielding each node's
    /// `(prefix, [(tenant, epoch)])` pair without materializing
    /// the full tree as a single buffer. Empty-tenant nodes are
    /// skipped (they're routing intermediates, not real entries).
    /// Children are visited in deterministic char-sorted order so
    /// callers paging the iterator can resume by re-running and
    /// skipping the first N items if the tree is unchanged.
    ///
    /// Like [`Tree::snapshot`], the walk is not atomic — concurrent
    /// `insert_text` may split or replace nodes mid-walk; the
    /// iterator may then reflect a mix of pre- and post-split
    /// state. Acceptable for mesh sync (eventual consistency).
    pub fn iter_entries(&self) -> EntriesIter {
        EntriesIter::new(self)
    }
}

/// Iterator yielded by [`Tree::iter_entries`]. Owns `Arc<Node>`
/// clones for the path it's currently inside so it survives
/// across awaits or any temporary release of `&Tree`.
pub struct EntriesIter {
    /// DFS frame stack. The top frame's `remaining_children` is
    /// the queue of children we still have to visit at the
    /// current depth. Frames are pushed when descending, popped
    /// when their children exhaust.
    stack: Vec<EntryFrame>,
    /// Mutable accumulated path-from-root. Pushed when descending
    /// into a child, truncated when popping a frame.
    path: String,
    /// Pre-staged emission. Set when a node with tenants is
    /// entered; consumed by the next `next()` call before
    /// continuing the walk.
    pending: Option<(String, Vec<(TenantId, u64)>)>,
}

struct EntryFrame {
    remaining_children: std::vec::IntoIter<NodeRef>,
    /// Byte length of this frame's edge text — used to truncate
    /// the path buffer in O(1) when popping. Each descent pushes
    /// a complete `&str` of this length, so popping the same
    /// byte count always lands on a UTF-8 char boundary.
    edge_byte_len: usize,
}

impl EntriesIter {
    fn new(tree: &Tree) -> Self {
        let root_tenants = snapshot_tenants(&tree.root);
        let pending = (!root_tenants.is_empty()).then(|| (String::new(), root_tenants));
        Self {
            stack: vec![EntryFrame {
                remaining_children: collect_sorted_children(&tree.root).into_iter(),
                edge_byte_len: 0,
            }],
            path: String::new(),
            pending,
        }
    }
}

impl Iterator for EntriesIter {
    type Item = (String, Vec<(TenantId, u64)>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(entry) = self.pending.take() {
            return Some(entry);
        }
        loop {
            let frame = self.stack.last_mut()?;
            if let Some(child) = frame.remaining_children.next() {
                // Push the edge directly through the lock guard —
                // no intermediate clone — and capture its byte
                // length so we can truncate atomically on pop.
                let edge_byte_len = {
                    let guard = child.text.read();
                    let s = guard.as_str();
                    self.path.push_str(s);
                    s.len()
                };

                let tenants = snapshot_tenants(&child);
                self.stack.push(EntryFrame {
                    remaining_children: collect_sorted_children(&child).into_iter(),
                    edge_byte_len,
                });
                if !tenants.is_empty() {
                    return Some((self.path.clone(), tenants));
                }
                // Empty tenants → loop continues, descending into
                // this child's children on the next iteration.
            } else {
                let edge_byte_len = frame.edge_byte_len;
                self.stack.pop();
                let new_len = self.path.len().saturating_sub(edge_byte_len);
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
    let mut children: Vec<(char, NodeRef)> = node
        .children
        .iter()
        .map(|entry| (*entry.key(), entry.value().clone()))
        .collect();
    children.sort_by_key(|(c, _)| *c);
    children.into_iter().map(|(_, n)| n).collect()
}

impl Tree {
    /// Reconstruct a tree from a snapshot.
    ///
    /// The snapshot must be in pre-order (parent before children) with
    /// `child_count` fields indicating tree structure.
    pub fn from_snapshot(snapshot: &crate::snapshot::TreeSnapshot) -> Self {
        let tree = Tree::new();
        if snapshot.nodes.is_empty() {
            return tree;
        }

        let mut idx = 0;
        tree.restore_node(&tree.root, &snapshot.nodes, &mut idx);
        tree
    }

    fn restore_node(
        &self,
        target: &NodeRef,
        nodes: &[crate::snapshot::SnapshotNode],
        idx: &mut usize,
    ) {
        if *idx >= nodes.len() {
            return;
        }

        let snap_node = &nodes[*idx];
        *idx += 1;

        // Set edge text
        *target.text.write() = NodeText::new(snap_node.edge.clone());

        // Set tenants
        let edge_chars = snap_node.edge.chars().count();
        for (tenant_str, epoch) in &snap_node.tenants {
            let tenant_id = intern_tenant(tenant_str);
            target
                .tenant_last_access_time
                .insert(Arc::clone(&tenant_id), *epoch);

            // Track char counts
            self.add_tenant_chars(&tenant_id, edge_chars);
        }

        // Restore children
        for _ in 0..snap_node.child_count {
            if *idx >= nodes.len() {
                break;
            }
            let child_snap = &nodes[*idx];
            let Some(first_char) = child_snap.edge.chars().next() else {
                // Empty edge for a child node — skip it. Advance idx
                // past this node and all its descendants.
                fn skip(nodes: &[crate::snapshot::SnapshotNode], idx: &mut usize) {
                    if *idx >= nodes.len() {
                        return;
                    }
                    let cc = nodes[*idx].child_count;
                    *idx += 1;
                    for _ in 0..cc {
                        skip(nodes, idx);
                    }
                }
                skip(nodes, idx);
                continue;
            };

            let child_node = Arc::new(Node {
                children: new_children_map(),
                text: RwLock::new(NodeText::empty()),
                tenant_last_access_time: new_tenant_map(),
                parent: RwLock::new(Some(Arc::downgrade(target))),
                last_tenant: RwLock::new(None),
            });

            self.restore_node(&child_node, nodes, idx);
            target.children.insert(first_char, child_node);
        }
    }

    /// Merge a remote snapshot into this tree.
    ///
    /// Reconstructs the remote tree from the snapshot, then walks both
    /// trees recursively, handling three cases for each child:
    /// 1. Exact edge match → recurse into both subtrees
    /// 2. Local edge is prefix of remote → descend into local, continue with remainder
    /// 3. Shared prefix shorter than both → split local node, attach remote remainder
    pub fn merge_snapshot(&self, snapshot: &crate::snapshot::TreeSnapshot) {
        if snapshot.nodes.is_empty() {
            return;
        }
        // Reconstruct the remote tree, then merge tree-to-tree
        let remote_tree = Tree::from_snapshot(snapshot);
        self.merge_tree(&remote_tree);
    }

    /// Merge another tree into this tree.
    ///
    /// Walks both trees recursively. At each node, merges tenant maps
    /// (remote wins on newer epoch) and reconciles children using the
    /// three-case edge comparison.
    pub fn merge_tree(&self, remote: &Tree) {
        self.merge_nodes(&self.root, &remote.root);
    }

    fn merge_nodes(&self, local: &NodeRef, remote: &NodeRef) {
        // Merge tenants at this node
        for entry in &remote.tenant_last_access_time {
            let tenant_id = Arc::clone(entry.key());
            let remote_epoch = *entry.value();

            let is_new = !local
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref());

            let should_update = local
                .tenant_last_access_time
                .get(tenant_id.as_ref())
                .map(|local_epoch| remote_epoch > *local_epoch)
                .unwrap_or(true);

            if should_update {
                local
                    .tenant_last_access_time
                    .insert(Arc::clone(&tenant_id), remote_epoch);

                if is_new {
                    let edge_chars = local.text.read().char_count();
                    self.add_tenant_chars(&tenant_id, edge_chars);
                }
            }
        }

        // Merge children
        for remote_entry in &remote.children {
            let rc = *remote_entry.key();
            let remote_child = remote_entry.value().clone();
            let remote_edge = remote_child.text.read().as_str().to_string();

            if let Some(local_entry) = local.children.get(&rc) {
                let local_child = local_entry.value().clone();
                drop(local_entry); // release DashMap guard before mutating

                let local_edge = local_child.text.read().as_str().to_string();
                let shared = shared_prefix_count(&local_edge, &remote_edge);

                if shared == local_edge.chars().count() && shared == remote_edge.chars().count() {
                    // Case 1: exact match — recurse
                    self.merge_nodes(&local_child, &remote_child);
                } else if shared == local_edge.chars().count() {
                    // Case 2: local edge is a prefix of remote edge.
                    // Descend into local child. The remote child's edge
                    // has a remainder that maps to a deeper level.
                    //
                    // First, merge remote tenants into local_child (the prefix node).
                    // The remote tenant owns the full path including this prefix.
                    for entry in &remote_child.tenant_last_access_time {
                        let tid = Arc::clone(entry.key());
                        let epoch = *entry.value();
                        let is_new = !local_child
                            .tenant_last_access_time
                            .contains_key(tid.as_ref());
                        let should_update = local_child
                            .tenant_last_access_time
                            .get(tid.as_ref())
                            .map(|e| epoch > *e)
                            .unwrap_or(true);
                        if should_update {
                            local_child
                                .tenant_last_access_time
                                .insert(Arc::clone(&tid), epoch);
                            if is_new {
                                let edge_chars = local_edge.chars().count();
                                self.add_tenant_chars(&tid, edge_chars);
                            }
                        }
                    }

                    let remote_remainder = advance_by_chars(&remote_edge, shared);
                    let Some(rem_first) = remote_remainder.chars().next() else {
                        continue;
                    };

                    // Create a virtual remote node with the trimmed edge.
                    // Use clone_subtree for children to rewrite parent pointers.
                    let trimmed_remote = Arc::new(Node {
                        children: new_children_map(),
                        text: RwLock::new(NodeText::new(remote_remainder.to_string())),
                        tenant_last_access_time: remote_child.tenant_last_access_time.clone(),
                        parent: RwLock::new(None),
                        last_tenant: RwLock::new(remote_child.last_tenant.read().clone()),
                    });
                    for child_entry in &remote_child.children {
                        let child_clone =
                            Self::clone_subtree(child_entry.value(), Some(&trimmed_remote));
                        trimmed_remote
                            .children
                            .insert(*child_entry.key(), child_clone);
                    }

                    if let Some(deeper_local) = local_child.children.get(&rem_first) {
                        let deeper_local = deeper_local.value().clone();
                        // Recurse: merge trimmed remote into the deeper local child
                        self.merge_nodes(&deeper_local, &trimmed_remote);
                    } else {
                        // No local child at this position — graft remote subtree
                        *trimmed_remote.parent.write() = Some(Arc::downgrade(&local_child));
                        self.accumulate_tenant_counts(&trimmed_remote);
                        local_child.children.insert(rem_first, trimmed_remote);
                    }
                } else {
                    // Case 3: shared prefix shorter than at least one edge.
                    // Covers both "shorter than both" and "remote is prefix of local".
                    // Split local node at the shared prefix point,
                    // then attach remote remainder as a sibling.

                    // Split local edge: "Hello world" at shared=6 →
                    //   split_node edge: "Hello "
                    //   local_child edge becomes: "world"
                    let (split_text, local_remainder_text) = {
                        let text = local_child.text.read();
                        text.split_at_char(shared)
                    };

                    // Case 3 guarantees shared < local_edge, so remainder is non-empty.
                    let Some(local_remainder_first) = local_remainder_text.first_char() else {
                        continue;
                    };

                    // Create the split node (intermediate).
                    // Tenants: union of local and remote at the shared prefix.
                    let split_node = Arc::new(Node {
                        children: new_children_map(),
                        text: RwLock::new(split_text),
                        tenant_last_access_time: local_child.tenant_last_access_time.clone(),
                        parent: RwLock::new(Some(Arc::downgrade(local))),
                        last_tenant: RwLock::new(local_child.last_tenant.read().clone()),
                    });
                    // Union remote tenants into the split node, updating
                    // tenant_char_count for newly added tenants.
                    let split_chars = shared;
                    for entry in &remote_child.tenant_last_access_time {
                        let tid = Arc::clone(entry.key());
                        let epoch = *entry.value();
                        let is_new = !split_node
                            .tenant_last_access_time
                            .contains_key(tid.as_ref());
                        let should_update = split_node
                            .tenant_last_access_time
                            .get(tid.as_ref())
                            .map(|e| epoch > *e)
                            .unwrap_or(true);
                        if should_update {
                            split_node
                                .tenant_last_access_time
                                .insert(Arc::clone(&tid), epoch);
                            if is_new {
                                self.add_tenant_chars(&tid, split_chars);
                            }
                        }
                    }

                    // Push local child down as child of split node. Re-parent
                    // inside the same `text` write section as the truncation
                    // so walkers validating the parent under a `text` read
                    // guard never see the truncated edge as top-level.
                    let mut local_child_text_write = local_child.text.write();
                    *local_child.parent.write() = Some(Arc::downgrade(&split_node));
                    *local_child_text_write = local_remainder_text;
                    drop(local_child_text_write);
                    split_node
                        .children
                        .insert(local_remainder_first, Arc::clone(&local_child));

                    // Create remote remainder as another child of split node
                    // Use clone_subtree to rewrite parent pointers correctly
                    let remote_remainder = advance_by_chars(&remote_edge, shared);
                    if let Some(rem_first) = remote_remainder.chars().next() {
                        let remote_subtree = Arc::new(Node {
                            children: new_children_map(),
                            text: RwLock::new(NodeText::new(remote_remainder.to_string())),
                            tenant_last_access_time: remote_child.tenant_last_access_time.clone(),
                            parent: RwLock::new(Some(Arc::downgrade(&split_node))),
                            last_tenant: RwLock::new(remote_child.last_tenant.read().clone()),
                        });
                        // Clone remote children with correct parent pointers
                        for child_entry in &remote_child.children {
                            let child_clone =
                                Self::clone_subtree(child_entry.value(), Some(&remote_subtree));
                            remote_subtree
                                .children
                                .insert(*child_entry.key(), child_clone);
                        }
                        self.accumulate_tenant_counts(&remote_subtree);
                        split_node.children.insert(rem_first, remote_subtree);
                    }

                    // Replace local's child with the split node
                    local.children.insert(rc, split_node);
                }
            } else {
                // No local child at this char — copy entire remote subtree
                let cloned = Self::clone_subtree(&remote_child, Some(local));
                self.accumulate_tenant_counts(&cloned);
                local.children.insert(rc, cloned);
            }
        }
    }

    /// Walk a subtree and accumulate tenant char counts into the
    /// tree-level `tenant_char_count` map.  Called after grafting a
    /// remote subtree into the local tree so that size tracking and
    /// eviction remain correct.
    fn accumulate_tenant_counts(&self, node: &NodeRef) {
        let edge_chars = node.text.read().char_count();
        for entry in &node.tenant_last_access_time {
            self.add_tenant_chars(entry.key(), edge_chars);
        }
        for child_entry in &node.children {
            self.accumulate_tenant_counts(child_entry.value());
        }
    }

    /// Deep clone a subtree, setting parent pointers correctly.
    fn clone_subtree(node: &NodeRef, parent: Option<&NodeRef>) -> NodeRef {
        let new_node = Arc::new(Node {
            children: new_children_map(),
            text: RwLock::new(node.text.read().clone_text()),
            tenant_last_access_time: node.tenant_last_access_time.clone(),
            parent: RwLock::new(parent.map(Arc::downgrade)),
            last_tenant: RwLock::new(node.last_tenant.read().clone()),
        });

        for entry in &node.children {
            let child_clone = Self::clone_subtree(entry.value(), Some(&new_node));
            new_node.children.insert(*entry.key(), child_clone);
        }

        new_node
    }
}

impl RadixTree for Tree {
    type Key = str;
    type MatchResult = PrefixMatchResult;

    fn insert(&self, key: &Self::Key, tenant: &str) {
        self.insert_text(key, tenant);
    }

    fn prefix_match(&self, key: &Self::Key) -> Option<TenantId> {
        let result = self.match_prefix_with_counts(key);
        if result.matched_char_count > 0 {
            Some(result.tenant)
        } else {
            None
        }
    }

    fn prefix_match_with_counts(&self, key: &Self::Key) -> Self::MatchResult {
        self.match_prefix_with_counts(key)
    }

    fn evict(&self, tenant: &TenantId, max_units: usize) {
        self.evict_by_tenant(tenant, max_units);
    }

    fn tenant_size(&self, tenant: &TenantId) -> usize {
        self.tenant_char_size(tenant)
    }

    fn reset(&self) {
        self.clear();
    }
}

//  Unit tests
#[cfg(test)]
#[expect(clippy::print_stdout, reason = "test diagnostics")]
mod tests {
    use std::{
        thread,
        time::{Duration, Instant},
    };

    use rand::{
        distr::{Alphanumeric, SampleString},
        rng as thread_rng, RngExt,
    };

    use super::*;

    /// Helper to convert tenant_char_count to HashMap<String, usize> for comparison
    fn get_maintained_counts(tree: &Tree) -> HashMap<String, usize> {
        tree.tenant_char_count
            .iter()
            .map(|entry| (entry.key().to_string(), *entry.value()))
            .collect()
    }

    #[test]
    fn test_matched_tenants_all_holders_of_deepest_node() {
        let tree = Tree::new();
        tree.insert_text("hello world", "tenant1");
        tree.insert_text("hello world", "tenant2");
        // tenant3 holds only a shallower prefix.
        tree.insert_text("hello", "tenant3");

        let result = tree.match_prefix_with_counts("hello world");
        assert_eq!(result.matched_char_count, 11);
        let mut tenants: Vec<&str> = result.matched_tenants.iter().map(AsRef::as_ref).collect();
        tenants.sort_unstable();
        assert_eq!(tenants, ["tenant1", "tenant2"]);
    }

    #[test]
    fn test_matched_tenants_empty_on_no_match() {
        let tree = Tree::new();
        tree.insert_text("hello", "tenant1");

        let result = tree.match_prefix_with_counts("zzz");
        assert_eq!(result.matched_char_count, 0);
        assert!(result.matched_tenants.is_empty());
    }

    #[test]
    fn test_matched_tenants_capped() {
        let tree = Tree::new();
        for i in 0..(MATCHED_TENANTS_CAP + 4) {
            tree.insert_text("hello world", &format!("tenant{i}"));
        }

        let result = tree.match_prefix_with_counts("hello world");
        assert_eq!(result.matched_char_count, 11);
        assert_eq!(result.matched_tenants.len(), MATCHED_TENANTS_CAP);
    }

    #[test]
    fn test_match_and_insert_with_populates_matched_tenants() {
        let tree = Tree::new();
        tree.insert_text("hello world", "tenant1");
        tree.insert_text("hello world", "tenant2");

        let result = tree.match_and_insert_with("hello world", |result| {
            let mut tenants: Vec<&str> = result.matched_tenants.iter().map(AsRef::as_ref).collect();
            tenants.sort_unstable();
            assert_eq!(tenants, ["tenant1", "tenant2"]);
            Some("tenant3")
        });
        assert_eq!(result.matched_char_count, 11);

        let result = tree.match_prefix_with_counts("hello world");
        assert!(result
            .matched_tenants
            .iter()
            .any(|tenant| tenant.as_ref() == "tenant3"));
    }

    #[test]
    fn test_tenant_char_count() {
        let tree = Tree::new();

        tree.insert_text("apple", "tenant1");
        tree.insert_text("apricot", "tenant1");
        tree.insert_text("banana", "tenant1");
        tree.insert_text("amplify", "tenant2");
        tree.insert_text("application", "tenant2");

        let computed_sizes = tree.get_used_size_per_tenant();
        let maintained_counts = get_maintained_counts(&tree);

        println!("Phase 1 - Maintained vs Computed counts:");
        println!("Maintained: {maintained_counts:?}\nComputed: {computed_sizes:?}");
        assert_eq!(
            maintained_counts, computed_sizes,
            "Phase 1: Initial insertions"
        );

        tree.insert_text("apartment", "tenant1");
        tree.insert_text("appetite", "tenant2");
        tree.insert_text("ball", "tenant1");
        tree.insert_text("box", "tenant2");

        let computed_sizes = tree.get_used_size_per_tenant();
        let maintained_counts = get_maintained_counts(&tree);

        println!("Phase 2 - Maintained vs Computed counts:");
        println!("Maintained: {maintained_counts:?}\nComputed: {computed_sizes:?}");
        assert_eq!(
            maintained_counts, computed_sizes,
            "Phase 2: Additional insertions"
        );

        tree.insert_text("zebra", "tenant1");
        tree.insert_text("zebra", "tenant2");
        tree.insert_text("zero", "tenant1");
        tree.insert_text("zero", "tenant2");

        let computed_sizes = tree.get_used_size_per_tenant();
        let maintained_counts = get_maintained_counts(&tree);

        println!("Phase 3 - Maintained vs Computed counts:");
        println!("Maintained: {maintained_counts:?}\nComputed: {computed_sizes:?}");
        assert_eq!(
            maintained_counts, computed_sizes,
            "Phase 3: Overlapping insertions"
        );

        tree.evict_tenant_by_size(10);

        let computed_sizes = tree.get_used_size_per_tenant();
        let maintained_counts = get_maintained_counts(&tree);

        println!("Phase 4 - Maintained vs Computed counts:");
        println!("Maintained: {maintained_counts:?}\nComputed: {computed_sizes:?}");
        assert_eq!(maintained_counts, computed_sizes, "Phase 4: After eviction");
    }

    fn random_string(len: usize) -> String {
        Alphanumeric.sample_string(&mut thread_rng(), len)
    }

    #[test]
    fn test_cold_start() {
        let tree = Tree::new();

        let (matched_text, tenant) = tree.prefix_match_legacy("hello");

        assert_eq!(matched_text, "");
        assert_eq!(tenant, "empty");
    }

    #[test]
    fn test_exact_match_seq() {
        let tree = Tree::new();
        tree.insert_text("hello", "tenant1");
        tree.pretty_print();
        tree.insert_text("apple", "tenant2");
        tree.pretty_print();
        tree.insert_text("banana", "tenant3");
        tree.pretty_print();

        let (matched_text, tenant) = tree.prefix_match_legacy("hello");
        assert_eq!(matched_text, "hello");
        assert_eq!(tenant, "tenant1");

        let (matched_text, tenant) = tree.prefix_match_legacy("apple");
        assert_eq!(matched_text, "apple");
        assert_eq!(tenant, "tenant2");

        let (matched_text, tenant) = tree.prefix_match_legacy("banana");
        assert_eq!(matched_text, "banana");
        assert_eq!(tenant, "tenant3");
    }

    #[test]
    fn test_exact_match_concurrent() {
        let tree = Arc::new(Tree::new());

        // spawn 3 threads for insert
        let tree_clone = Arc::clone(&tree);

        let texts = ["hello", "apple", "banana"];
        let tenants = ["tenant1", "tenant2", "tenant3"];

        let mut handles = vec![];

        for i in 0..3 {
            let tree_clone = Arc::clone(&tree_clone);
            let text = texts[i];
            let tenant = tenants[i];

            let handle = thread::spawn(move || {
                tree_clone.insert_text(text, tenant);
            });

            handles.push(handle);
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }

        // spawn 3 threads for match
        let mut handles = vec![];

        let tree_clone = Arc::clone(&tree);

        for i in 0..3 {
            let tree_clone = Arc::clone(&tree_clone);
            let text = texts[i];
            let tenant = tenants[i];

            let handle = thread::spawn(move || {
                let (matched_text, matched_tenant) = tree_clone.prefix_match_legacy(text);
                assert_eq!(matched_text, text);
                assert_eq!(matched_tenant, tenant);
            });

            handles.push(handle);
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_partial_match_concurrent() {
        let tree = Arc::new(Tree::new());

        // spawn 3 threads for insert
        let tree_clone = Arc::clone(&tree);

        static TEXTS: [&str; 3] = ["apple", "apabc", "acbdeds"];

        let mut handles = vec![];

        for text in &TEXTS {
            let tree_clone = Arc::clone(&tree_clone);
            let tenant = "tenant0";

            let handle = thread::spawn(move || {
                tree_clone.insert_text(text, tenant);
            });

            handles.push(handle);
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }

        // spawn 3 threads for match
        let mut handles = vec![];

        let tree_clone = Arc::clone(&tree);

        for text in &TEXTS {
            let tree_clone = Arc::clone(&tree_clone);
            let tenant = "tenant0";

            let handle = thread::spawn(move || {
                let (matched_text, matched_tenant) = tree_clone.prefix_match_legacy(text);
                assert_eq!(matched_text, *text);
                assert_eq!(matched_tenant, tenant);
            });

            handles.push(handle);
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_group_prefix_insert_match_concurrent() {
        static PREFIXES: [&str; 4] = [
            "Clock strikes midnight, I'm still wide awake",
            "Got dreams bigger than these city lights",
            "Time waits for no one, gotta make my move",
            "Started from the bottom, that's no metaphor",
        ];
        let suffixes = [
            "Got too much to prove, ain't got time to lose",
            "History in the making, yeah, you can't erase this",
        ];
        let tree = Arc::new(Tree::new());

        let mut handles = vec![];

        for (i, prefix) in PREFIXES.iter().enumerate() {
            for suffix in &suffixes {
                let tree_clone = Arc::clone(&tree);
                let text = format!("{prefix} {suffix}");
                let tenant = format!("tenant{i}");

                let handle = thread::spawn(move || {
                    tree_clone.insert_text(&text, &tenant);
                });

                handles.push(handle);
            }
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }

        tree.pretty_print();

        // check matching using multi threads
        let mut handles = vec![];

        for (i, prefix) in PREFIXES.iter().enumerate() {
            let tree_clone = Arc::clone(&tree);

            let handle = thread::spawn(move || {
                let (matched_text, matched_tenant) = tree_clone.prefix_match_legacy(prefix);
                let tenant = format!("tenant{i}");
                assert_eq!(matched_text, *prefix);
                assert_eq!(matched_tenant, tenant);
            });

            handles.push(handle);
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_mixed_concurrent_insert_match() {
        // ensure it does not deadlock instead of doing correctness check

        static PREFIXES: [&str; 4] = [
            "Clock strikes midnight, I'm still wide awake",
            "Got dreams bigger than these city lights",
            "Time waits for no one, gotta make my move",
            "Started from the bottom, that's no metaphor",
        ];
        let suffixes = [
            "Got too much to prove, ain't got time to lose",
            "History in the making, yeah, you can't erase this",
        ];
        let tree = Arc::new(Tree::new());

        let mut handles = vec![];

        for (i, prefix) in PREFIXES.iter().enumerate() {
            for suffix in &suffixes {
                let tree_clone = Arc::clone(&tree);
                let text = format!("{prefix} {suffix}");
                let tenant = format!("tenant{i}");

                let handle = thread::spawn(move || {
                    tree_clone.insert_text(&text, &tenant);
                });

                handles.push(handle);
            }
        }

        // check matching using multi threads
        for prefix in &PREFIXES {
            let tree_clone = Arc::clone(&tree);

            let handle = thread::spawn(move || {
                let (_matched_text, _matched_tenant) = tree_clone.prefix_match_legacy(prefix);
            });

            handles.push(handle);
        }

        // wait
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_utf8_split_seq() {
        // The string should be indexed and split by a utf-8 value basis instead of byte basis
        // use .chars() to get the iterator of the utf-8 value
        let tree = Arc::new(Tree::new());

        static TEST_PAIRS: [(&str, &str); 3] = [
            ("你好嗎", "tenant1"),
            ("你好喔", "tenant2"),
            ("你心情好嗎", "tenant3"),
        ];

        // Insert sequentially
        for (text, tenant) in &TEST_PAIRS {
            tree.insert_text(text, tenant);
        }

        tree.pretty_print();

        for (text, tenant) in &TEST_PAIRS {
            let (matched_text, matched_tenant) = tree.prefix_match_legacy(text);
            assert_eq!(matched_text, *text);
            assert_eq!(matched_tenant, *tenant);
        }
    }

    #[test]
    fn test_utf8_split_concurrent() {
        let tree = Arc::new(Tree::new());

        static TEST_PAIRS: [(&str, &str); 3] = [
            ("你好嗎", "tenant1"),
            ("你好喔", "tenant2"),
            ("你心情好嗎", "tenant3"),
        ];

        // Create multiple threads for insertion
        let mut handles = vec![];

        for (text, tenant) in &TEST_PAIRS {
            let tree_clone = Arc::clone(&tree);

            let handle = thread::spawn(move || {
                tree_clone.insert_text(text, tenant);
            });

            handles.push(handle);
        }

        // Wait for all insertions to complete
        for handle in handles {
            handle.join().unwrap();
        }

        tree.pretty_print();

        // Create multiple threads for matching
        let mut handles = vec![];

        for (text, tenant) in &TEST_PAIRS {
            let tree_clone = Arc::clone(&tree);

            let handle = thread::spawn(move || {
                let (matched_text, matched_tenant) = tree_clone.prefix_match_legacy(text);
                assert_eq!(matched_text, *text);
                assert_eq!(matched_tenant, *tenant);
            });

            handles.push(handle);
        }

        // Wait for all matches to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_simple_eviction() {
        let tree = Tree::new();
        let max_size = 5;

        // Insert strings for both tenants
        tree.insert_text("hello", "tenant1"); // size 5

        tree.insert_text("hello", "tenant2"); // size 5
        thread::sleep(Duration::from_millis(10));
        tree.insert_text("world", "tenant2"); // size 5, total for tenant2 = 10

        tree.pretty_print();

        let sizes_before = tree.get_used_size_per_tenant();
        assert_eq!(sizes_before.get("tenant1").unwrap(), &5); // "hello" = 5
        assert_eq!(sizes_before.get("tenant2").unwrap(), &10); // "hello" + "world" = 10

        // Evict: the tree-wide total (15) exceeds max_size, so LRU leaves go
        // first - tenant1's "hello" (oldest), then tenant2's "hello"
        tree.evict_tenant_by_size(max_size);

        tree.pretty_print();

        let sizes_after = tree.get_used_size_per_tenant();
        assert_eq!(*sizes_after.get("tenant1").unwrap_or(&0), 0); // Oldest, evicted
        assert_eq!(sizes_after.get("tenant2").unwrap(), &5); // Only "world" remains

        let (matched, tenant) = tree.prefix_match_legacy("world");
        assert_eq!(matched, "world");
        assert_eq!(tenant, "tenant2");
    }

    #[test]
    fn test_advanced_eviction() {
        let tree = Tree::new();

        // Tree-wide budget
        let max_size: usize = 100;

        // Define prefixes
        let prefixes = ["aqwefcisdf", "iajsdfkmade", "kjnzxcvewqe", "iejksduqasd"];

        // Insert strings with shared prefixes
        for _i in 0..100 {
            for (j, prefix) in prefixes.iter().enumerate() {
                let random_suffix = random_string(10);
                let text = format!("{prefix}{random_suffix}");
                let tenant = format!("tenant{}", j + 1);
                tree.insert_text(&text, &tenant);
            }
        }

        // Perform eviction
        tree.evict_tenant_by_size(max_size);

        // Check sizes after eviction: the tree-wide total obeys the budget
        let sizes_after = tree.get_used_size_per_tenant();
        let total_after: usize = sizes_after.values().sum();
        assert!(
            total_after <= max_size,
            "Tree total exceeds budget. Total: {total_after}, Limit: {max_size}"
        );
    }

    #[test]
    fn test_concurrent_operations_with_eviction() {
        // Ensure eviction works fine with concurrent insert and match operations for a given period

        let tree = Arc::new(Tree::new());
        let mut handles = vec![];
        let test_duration = Duration::from_secs(10);
        let start_time = Instant::now();
        let max_size = 100; // Tree-wide budget

        // Spawn eviction thread
        {
            let tree = Arc::clone(&tree);
            let handle = thread::spawn(move || {
                while start_time.elapsed() < test_duration {
                    // Run eviction
                    tree.evict_tenant_by_size(max_size);

                    // Sleep for 5 seconds
                    thread::sleep(Duration::from_secs(5));
                }
            });
            handles.push(handle);
        }

        // Spawn 4 worker threads
        for thread_id in 0..4 {
            let tree = Arc::clone(&tree);
            let handle = thread::spawn(move || {
                let mut rng = rand::rng();
                let tenant = format!("tenant{}", thread_id + 1);
                let prefix = format!("prefix{thread_id}");

                while start_time.elapsed() < test_duration {
                    // Random decision: match or insert (70% match, 30% insert)
                    if rng.random_bool(0.7) {
                        // Perform match operation
                        let random_len = rng.random_range(3..10);
                        let search_str = format!("{prefix}{}", random_string(random_len));
                        let (_matched, _) = tree.prefix_match_legacy(&search_str);
                    } else {
                        // Perform insert operation
                        let random_len = rng.random_range(5..15);
                        let insert_str = format!("{prefix}{}", random_string(random_len));
                        tree.insert_text(&insert_str, &tenant);
                        // println!("Thread {} inserted: {}", thread_id, insert_str);
                    }

                    // Small random sleep to vary timing
                    thread::sleep(Duration::from_millis(rng.random_range(10..100)));
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // final eviction
        tree.evict_tenant_by_size(max_size);

        // Final size check: the tree-wide total obeys the budget
        let final_sizes = tree.get_used_size_per_tenant();
        println!("Final sizes after test completion: {final_sizes:?}");

        let final_total: usize = final_sizes.values().sum();
        assert!(
            final_total <= max_size,
            "Tree total exceeds budget. Final total: {final_total}, Limit: {max_size}"
        );
    }

    #[test]
    fn test_leaf_of() {
        let tree = Tree::new();

        // Helper to convert leaves to strings for easier assertion
        let leaves_as_strings =
            |leaves: &[TenantId]| -> Vec<String> { leaves.iter().map(|t| t.to_string()).collect() };

        // Single node
        tree.insert_text("hello", "tenant1");
        let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
        assert_eq!(leaves_as_strings(&leaves), vec!["tenant1"]);

        // Node with multiple tenants
        tree.insert_text("hello", "tenant2");
        let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
        let leaves_str = leaves_as_strings(&leaves);
        assert_eq!(leaves_str.len(), 2);
        assert!(leaves_str.contains(&"tenant1".to_string()));
        assert!(leaves_str.contains(&"tenant2".to_string()));

        // Non-leaf node
        tree.insert_text("hi", "tenant1");
        let leaves = Tree::leaf_of(&tree.root.children.get(&'h').unwrap());
        assert!(leaves.is_empty());
    }

    #[test]
    fn test_get_used_size_per_tenant() {
        let tree = Tree::new();

        // Single tenant
        tree.insert_text("hello", "tenant1");
        tree.insert_text("world", "tenant1");
        let sizes = tree.get_used_size_per_tenant();

        tree.pretty_print();
        println!("{sizes:?}");
        assert_eq!(sizes.get("tenant1").unwrap(), &10); // "hello" + "world"

        // Multiple tenants sharing nodes
        tree.insert_text("hello", "tenant2");
        tree.insert_text("help", "tenant2");
        let sizes = tree.get_used_size_per_tenant();

        tree.pretty_print();
        println!("{sizes:?}");
        assert_eq!(sizes.get("tenant1").unwrap(), &10);
        assert_eq!(sizes.get("tenant2").unwrap(), &6); // "hello" + "p"

        // UTF-8 characters
        tree.insert_text("你好", "tenant3");
        let sizes = tree.get_used_size_per_tenant();
        tree.pretty_print();
        println!("{sizes:?}");
        assert_eq!(sizes.get("tenant3").unwrap(), &2); // 2 Chinese characters

        tree.pretty_print();
    }

    #[test]
    fn test_prefix_match_tenant() {
        let tree = Tree::new();

        // Insert overlapping prefixes for different tenants
        tree.insert_text("hello", "tenant1"); // tenant1: hello
        tree.insert_text("hello", "tenant2"); // tenant2: hello
        tree.insert_text("hello world", "tenant2"); // tenant2: hello -> world
        tree.insert_text("help", "tenant1"); // tenant1: hel -> p
        tree.insert_text("helicopter", "tenant2"); // tenant2: hel -> icopter

        assert_eq!(tree.prefix_match_tenant("hello", "tenant1"), "hello"); // Full match for tenant1
        assert_eq!(tree.prefix_match_tenant("help", "tenant1"), "help"); // Exclusive to tenant1
        assert_eq!(tree.prefix_match_tenant("hel", "tenant1"), "hel"); // Shared prefix
        assert_eq!(tree.prefix_match_tenant("hello world", "tenant1"), "hello"); // Should stop at tenant1's boundary
        assert_eq!(tree.prefix_match_tenant("helicopter", "tenant1"), "hel"); // Should stop at tenant1's boundary

        assert_eq!(tree.prefix_match_tenant("hello", "tenant2"), "hello"); // Full match for tenant2
        assert_eq!(
            tree.prefix_match_tenant("hello world", "tenant2"),
            "hello world"
        ); // Exclusive to tenant2
        assert_eq!(
            tree.prefix_match_tenant("helicopter", "tenant2"),
            "helicopter"
        ); // Exclusive to tenant2
        assert_eq!(tree.prefix_match_tenant("hel", "tenant2"), "hel"); // Shared prefix
        assert_eq!(tree.prefix_match_tenant("help", "tenant2"), "hel"); // Should stop at tenant2's boundary

        assert_eq!(tree.prefix_match_tenant("hello", "tenant3"), ""); // Non-existent tenant
        assert_eq!(tree.prefix_match_tenant("help", "tenant3"), ""); // Non-existent tenant
    }

    // NOTE: test_simple_tenant_eviction and test_complex_tenant_eviction removed
    // because remove_tenant was removed (inefficient O(n) implementation).
    // TODO: Re-add these tests when efficient remove_tenant is implemented.

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_empty_string_input() {
        let tree = Tree::new();

        // Insert empty string
        tree.insert_text("", "tenant1");

        // Match empty string
        let (matched, tenant) = tree.prefix_match_legacy("");
        assert_eq!(matched, "");
        assert_eq!(tenant, "tenant1");

        // Insert non-empty, then match empty
        tree.insert_text("hello", "tenant2");
        let (matched, tenant) = tree.prefix_match_legacy("");
        assert_eq!(matched, "");
        assert_eq!(tenant, "tenant1");
    }

    #[test]
    fn test_single_character_operations() {
        let tree = Tree::new();

        // Insert single characters
        tree.insert_text("a", "tenant1");
        tree.insert_text("b", "tenant2");
        tree.insert_text("c", "tenant1");

        let (matched, tenant) = tree.prefix_match_legacy("a");
        assert_eq!(matched, "a");
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match_legacy("b");
        assert_eq!(matched, "b");
        assert_eq!(tenant, "tenant2");

        // Match with longer string starting with single char
        let (matched, tenant) = tree.prefix_match_legacy("abc");
        assert_eq!(matched, "a");
        assert_eq!(tenant, "tenant1");
    }

    #[test]
    fn test_prefix_is_subset_of_existing() {
        let tree = Tree::new();

        // Insert longer string first
        tree.insert_text("application", "tenant1");

        // Now insert prefix of existing
        tree.insert_text("app", "tenant2");

        // Match the prefix - both tenants own "app" node
        let (matched, tenant) = tree.prefix_match_legacy("app");
        assert_eq!(matched, "app");
        assert!(tenant == "tenant1" || tenant == "tenant2");

        // Match longer string
        let (matched, tenant) = tree.prefix_match_legacy("application");
        assert_eq!(matched, "application");
        assert_eq!(tenant, "tenant1");

        // Match "apple" - matches "app" + "l" from the child node = "appl"
        // Then 'e' doesn't match 'i' in the remaining suffix, so stops at 4 chars
        let (matched, _tenant) = tree.prefix_match_legacy("apple");
        assert_eq!(matched, "appl");
    }

    #[test]
    fn test_existing_is_prefix_of_new() {
        let tree = Tree::new();

        // Insert shorter string first
        tree.insert_text("app", "tenant1");

        // Now insert longer string with same prefix
        tree.insert_text("application", "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy("app");
        assert_eq!(matched, "app");
        assert!(tenant == "tenant1" || tenant == "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy("application");
        assert_eq!(matched, "application");
        assert_eq!(tenant, "tenant2");

        // "applesauce" matches "app" + "l" from the child node = "appl"
        // Then 'e' in "esauce" doesn't match 'i' in the suffix, so matching stops
        let (matched, _tenant) = tree.prefix_match_legacy("applesauce");
        assert_eq!(matched, "appl");
    }

    // ==================== prefix_match_with_counts Tests ====================

    #[test]
    fn test_prefix_match_with_counts_accuracy() {
        let tree = Tree::new();

        tree.insert_text("hello world", "tenant1");

        // Exact match
        let result = tree.match_prefix_with_counts("hello world");
        assert_eq!(result.matched_char_count, 11);
        assert_eq!(result.input_char_count, 11);
        assert_eq!(&*result.tenant, "tenant1");

        // Partial match
        let result = tree.match_prefix_with_counts("hello");
        assert_eq!(result.matched_char_count, 5);
        assert_eq!(result.input_char_count, 5);

        // Extended match
        let result = tree.match_prefix_with_counts("hello world and more");
        assert_eq!(result.matched_char_count, 11);
        assert_eq!(result.input_char_count, 20);

        // No match
        let result = tree.match_prefix_with_counts("goodbye");
        assert_eq!(result.matched_char_count, 0);
        assert_eq!(result.input_char_count, 7);
    }

    #[test]
    fn test_prefix_match_with_counts_utf8() {
        let tree = Tree::new();

        // UTF-8 string: 5 characters, more bytes
        tree.insert_text("你好世界呀", "tenant1");

        let result = tree.match_prefix_with_counts("你好世界呀");
        assert_eq!(result.matched_char_count, 5);
        assert_eq!(result.input_char_count, 5);

        let result = tree.match_prefix_with_counts("你好");
        assert_eq!(result.matched_char_count, 2);
        assert_eq!(result.input_char_count, 2);

        // Mixed ASCII and UTF-8
        tree.insert_text("hello你好", "tenant2");
        let result = tree.match_prefix_with_counts("hello你好世界");
        assert_eq!(result.matched_char_count, 7); // "hello你好" = 7 chars
        assert_eq!(result.input_char_count, 9); // "hello你好世界" = 9 chars
    }

    // ==================== Node Splitting Edge Cases ====================

    #[test]
    fn test_split_at_first_character() {
        let tree = Tree::new();

        // Insert "abc"
        tree.insert_text("abc", "tenant1");

        // Insert "aXX" - should split at first char
        tree.insert_text("aXX", "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy("abc");
        assert_eq!(matched, "abc");
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match_legacy("aXX");
        assert_eq!(matched, "aXX");
        assert_eq!(tenant, "tenant2");

        let (matched, _) = tree.prefix_match_legacy("a");
        assert_eq!(matched, "a");
    }

    #[test]
    fn test_split_at_last_character() {
        let tree = Tree::new();

        // Insert "abcd"
        tree.insert_text("abcd", "tenant1");

        // Insert "abcX" - should split at last char of shared prefix
        tree.insert_text("abcX", "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy("abcd");
        assert_eq!(matched, "abcd");
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match_legacy("abcX");
        assert_eq!(matched, "abcX");
        assert_eq!(tenant, "tenant2");

        let (matched, _) = tree.prefix_match_legacy("abc");
        assert_eq!(matched, "abc");
    }

    #[test]
    fn test_multiple_splits_same_path() {
        let tree = Tree::new();

        // Create a chain of splits
        tree.insert_text("abcdefgh", "tenant1");
        tree.insert_text("abcdef", "tenant2");
        tree.insert_text("abcd", "tenant3");
        tree.insert_text("ab", "tenant4");

        // Verify all paths work
        assert_eq!(tree.prefix_match_legacy("abcdefgh").0, "abcdefgh");
        assert_eq!(tree.prefix_match_legacy("abcdef").0, "abcdef");
        assert_eq!(tree.prefix_match_legacy("abcd").0, "abcd");
        assert_eq!(tree.prefix_match_legacy("ab").0, "ab");
        assert_eq!(tree.prefix_match_legacy("a").0, "a");
    }

    // ==================== High Contention Stress Tests ====================

    #[test]
    fn test_high_contention_same_prefix() {
        let tree = Arc::new(Tree::new());
        let num_threads = 16;
        let ops_per_thread = 100;
        let mut handles = vec![];

        // All threads operate on strings with same prefix
        for thread_id in 0..num_threads {
            let tree = Arc::clone(&tree);
            let handle = thread::spawn(move || {
                let tenant = format!("tenant{thread_id}");
                for i in 0..ops_per_thread {
                    let text = format!("shared_prefix_{i}");
                    tree.insert_text(&text, &tenant);

                    // Immediately try to match
                    let (matched, _) = tree.prefix_match_legacy(&text);
                    assert!(
                        matched.starts_with("shared_prefix_"),
                        "Match should start with shared_prefix_"
                    );
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify tree is still consistent
        let sizes = tree.get_used_size_per_tenant();
        assert!(!sizes.is_empty(), "Tree should have entries");
    }

    // NOTE: test_rapid_insert_remove_cycles removed because remove_tenant was removed.
    // TODO: Re-add this test when efficient remove_tenant is implemented.

    // ==================== ASCII/UTF-8 Consistency Tests ====================

    #[test]
    fn test_ascii_utf8_consistency() {
        let tree = Tree::new();

        // Insert ASCII
        tree.insert_text("hello", "tenant1");

        // Insert UTF-8 with same logical prefix (none)
        tree.insert_text("你好", "tenant2");

        // Insert mixed
        tree.insert_text("hello你好", "tenant3");

        // All should be retrievable
        assert_eq!(tree.prefix_match_legacy("hello").0, "hello");
        assert_eq!(tree.prefix_match_legacy("你好").0, "你好");
        assert_eq!(tree.prefix_match_legacy("hello你好").0, "hello你好");

        // Counts should be correct
        let result = tree.match_prefix_with_counts("hello");
        assert_eq!(result.matched_char_count, 5);
        assert_eq!(result.input_char_count, 5);

        let result = tree.match_prefix_with_counts("你好");
        assert_eq!(result.matched_char_count, 2);
        assert_eq!(result.input_char_count, 2);

        let result = tree.match_prefix_with_counts("hello你好");
        assert_eq!(result.matched_char_count, 7);
        assert_eq!(result.input_char_count, 7);
    }

    #[test]
    fn test_emoji_handling() {
        let tree = Tree::new();

        // Emoji are multi-byte UTF-8
        tree.insert_text("hello 👋", "tenant1");
        tree.insert_text("hello 👋🌍", "tenant2");

        let (matched, tenant) = tree.prefix_match_legacy("hello 👋");
        assert_eq!(matched, "hello 👋");
        assert_eq!(tenant, "tenant1");

        let (matched, tenant) = tree.prefix_match_legacy("hello 👋🌍");
        assert_eq!(matched, "hello 👋🌍");
        assert_eq!(tenant, "tenant2");

        // Verify char count (not byte count)
        let result = tree.match_prefix_with_counts("hello 👋");
        assert_eq!(result.matched_char_count, 7);
        assert_eq!(result.input_char_count, 7); // h-e-l-l-o-space-emoji
    }

    // ==================== Eviction Edge Cases ====================

    #[test]
    fn test_eviction_empty_tree() {
        let tree = Tree::new();

        // Should not panic on empty tree
        tree.evict_tenant_by_size(100);

        let sizes = tree.get_used_size_per_tenant();
        assert!(sizes.is_empty());
    }

    #[test]
    fn test_eviction_zero_max_size() {
        let tree = Tree::new();

        tree.insert_text("hello", "tenant1");
        tree.insert_text("world", "tenant1");

        // Evict with max_size = 0 should remove everything
        tree.evict_tenant_by_size(0);

        let sizes = tree.get_used_size_per_tenant();
        assert!(
            sizes.is_empty() || sizes.values().all(|&v| v == 0),
            "All tenants should be evicted or have zero size"
        );
    }

    #[test]
    fn test_eviction_single_tenant_all_entries() {
        let tree = Tree::new();

        // Insert many entries for single tenant
        for i in 0..100 {
            let text = format!("entry{i:03}");
            tree.insert_text(&text, "tenant1");
        }

        let initial_size = *tree.get_used_size_per_tenant().get("tenant1").unwrap();
        assert!(initial_size > 50, "Should have significant size");

        // Evict to small size
        tree.evict_tenant_by_size(50);

        let final_size = *tree.get_used_size_per_tenant().get("tenant1").unwrap_or(&0);
        assert!(
            final_size <= 50,
            "Size {final_size} should be <= 50 after eviction"
        );
    }

    // ==================== Last Tenant Cache Tests ====================

    #[test]
    fn test_last_tenant_cache_update() {
        let tree = Tree::new();

        // Insert for tenant1
        tree.insert_text("hello", "tenant1");

        // First match should return tenant1
        let (_, tenant) = tree.prefix_match_legacy("hello");
        assert_eq!(tenant, "tenant1");

        // Insert for tenant2 on same path
        tree.insert_text("hello", "tenant2");

        // Match again - should still work (cache or iteration)
        let (matched, _) = tree.prefix_match_legacy("hello");
        assert_eq!(matched, "hello");
    }

    // NOTE: test_stale_cache_after_tenant_removal removed because remove_tenant was removed.
    // TODO: Re-add this test when efficient remove_tenant is implemented.

    // ==================== Consistency Verification Tests ====================

    #[test]
    fn test_char_count_consistency_after_operations() {
        let tree = Tree::new();

        // Helper to verify consistency
        let verify_consistency = |tree: &Tree| {
            let maintained = get_maintained_counts(tree);
            let computed = tree.get_used_size_per_tenant();
            assert_eq!(
                maintained, computed,
                "Maintained counts should match computed counts"
            );
        };

        // Insert phase
        for i in 0..50 {
            tree.insert_text(&format!("prefix{i}"), "tenant1");
            tree.insert_text(&format!("other{i}"), "tenant2");
        }
        verify_consistency(&tree);

        // Overlapping inserts
        for i in 0..25 {
            tree.insert_text(&format!("prefix{i}"), "tenant2");
        }
        verify_consistency(&tree);

        // Eviction
        tree.evict_tenant_by_size(100);
        verify_consistency(&tree);

        // NOTE: Tenant removal test removed because remove_tenant was removed.
        // TODO: Re-add when efficient remove_tenant is implemented.
    }

    #[test]
    fn test_tree_structure_integrity_after_stress() {
        let tree = Arc::new(Tree::new());
        let num_threads = 8;
        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let tree = Arc::clone(&tree);
            let handle = thread::spawn(move || {
                let mut rng = rand::rng();
                let tenant = format!("tenant{thread_id}");

                for _ in 0..200 {
                    let op: u8 = rng.random_range(0..10);
                    let key = format!("key{}", rng.random_range(0..50));

                    match op {
                        0..=6 => {
                            // Insert (70%)
                            tree.insert_text(&key, &tenant);
                        }
                        7..=8 => {
                            // Match (20%)
                            let _ = tree.prefix_match_legacy(&key);
                        }
                        _ => {
                            // Match with counts (10%)
                            let _ = tree.match_prefix_with_counts(&key);
                        }
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked during stress test");
        }

        // Verify tree is still functional
        let sizes = tree.get_used_size_per_tenant();
        for (tenant, size) in &sizes {
            assert!(*size > 0, "Tenant {tenant} should have positive size");
        }

        // Verify char count consistency
        let maintained = get_maintained_counts(&tree);
        let computed = tree.get_used_size_per_tenant();
        assert_eq!(
            maintained, computed,
            "Counts should be consistent after stress test"
        );
    }

    // ==================== Boundary Condition Tests ====================

    #[test]
    fn test_very_long_strings() {
        let tree = Tree::new();

        // Create a very long string (10KB)
        let long_string: String = (0..10000)
            .map(|i| ((i % 26) as u8 + b'a') as char)
            .collect();

        tree.insert_text(&long_string, "tenant1");

        let (matched, tenant) = tree.prefix_match_legacy(&long_string);
        assert_eq!(matched.len(), long_string.len());
        assert_eq!(tenant, "tenant1");

        // Partial match of long string
        let partial = &long_string[..5000];
        let (matched, _) = tree.prefix_match_legacy(partial);
        assert_eq!(matched, partial);
    }

    #[test]
    fn test_many_tenants_same_path() {
        let tree = Tree::new();

        // 100 tenants all insert same string
        for i in 0..100 {
            tree.insert_text("shared_path", &format!("tenant{i}"));
        }

        // Match should return one of them
        let (matched, _) = tree.prefix_match_legacy("shared_path");
        assert_eq!(matched, "shared_path");

        // Verify all tenants are tracked
        let sizes = tree.get_used_size_per_tenant();
        assert_eq!(sizes.len(), 100, "Should have 100 tenants");
    }

    #[test]
    fn test_special_characters() {
        let tree = Tree::new();

        // Various special characters
        let test_cases = vec![
            ("hello\nworld", "tenant1"),      // newline
            ("hello\tworld", "tenant2"),      // tab
            ("hello\0world", "tenant3"),      // null byte
            ("hello\u{A0}world", "tenant4"),  // non-breaking space
            ("path/to/file", "tenant5"),      // slashes
            ("query?param=value", "tenant6"), // URL-like
        ];

        for (text, tenant) in &test_cases {
            tree.insert_text(text, tenant);
        }

        for (text, tenant) in &test_cases {
            let (matched, matched_tenant) = tree.prefix_match_legacy(text);
            assert_eq!(matched, *text, "Failed for: {text:?}");
            assert_eq!(matched_tenant, *tenant);
        }
    }

    // ── Snapshot tests ──────────────────────────────────────────────

    #[test]
    fn test_snapshot_empty_tree() {
        let tree = Tree::new();
        let snap = tree.snapshot();
        // Root node always present
        assert_eq!(snap.node_count(), 1);
    }

    #[test]
    fn test_snapshot_round_trip_single_entry() {
        let tree = Tree::new();
        tree.insert_text("Hello world", "worker-1");

        let snap = tree.snapshot();
        assert!(snap.node_count() > 0);

        let restored = Tree::from_snapshot(&snap);
        let result = restored.match_prefix_with_counts("Hello world");
        assert_eq!(result.matched_char_count, 11);
        assert_eq!(result.tenant.as_ref(), "worker-1");
    }

    #[test]
    fn test_snapshot_round_trip_shared_prefixes() {
        let tree = Tree::new();
        tree.insert_text("Hello world", "worker-1");
        tree.insert_text("Hello there", "worker-2");
        tree.insert_text("Goodbye", "worker-3");

        let snap = tree.snapshot();

        let restored = Tree::from_snapshot(&snap);

        // "Hello " is shared prefix — both "world" and "there" branch from it
        let r1 = restored.match_prefix_with_counts("Hello world");
        assert_eq!(r1.matched_char_count, 11);
        assert_eq!(r1.tenant.as_ref(), "worker-1");

        let r2 = restored.match_prefix_with_counts("Hello there");
        assert_eq!(r2.matched_char_count, 11);
        assert_eq!(r2.tenant.as_ref(), "worker-2");

        let r3 = restored.match_prefix_with_counts("Goodbye");
        assert_eq!(r3.matched_char_count, 7);
        assert_eq!(r3.tenant.as_ref(), "worker-3");
    }

    #[test]
    fn test_snapshot_size_vs_flat_ops() {
        let tree = Tree::new();
        // Insert 100 entries sharing a long prefix
        let prefix = "A".repeat(10000);
        for i in 0..100 {
            let text = format!("{prefix}_{i}");
            tree.insert_text(&text, &format!("worker-{i}"));
        }

        let snap = tree.snapshot();
        let snap_bytes = snap.to_bytes().unwrap();

        // Flat ops would be 100 × 10000 chars = ~1 MB
        // Snapshot should be much smaller (shared prefix stored once)
        let flat_size: usize = (0..100).map(|i| format!("{prefix}_{i}").len()).sum();

        assert!(
            snap_bytes.len() < flat_size / 2,
            "Snapshot ({} bytes) should be at least 2x smaller than flat ops ({} bytes)",
            snap_bytes.len(),
            flat_size
        );
    }

    #[test]
    fn test_snapshot_bincode_round_trip() {
        let tree = Tree::new();
        tree.insert_text("Hello world", "worker-1");
        tree.insert_text("Hello there", "worker-2");

        let snap = tree.snapshot();
        let bytes = snap.to_bytes().unwrap();
        let restored_snap = crate::snapshot::TreeSnapshot::from_bytes(&bytes).unwrap();

        let restored = Tree::from_snapshot(&restored_snap);
        let r = restored.match_prefix_with_counts("Hello world");
        assert_eq!(r.matched_char_count, 11);
    }

    #[test]
    fn test_merge_disjoint_trees() {
        let tree1 = Tree::new();
        tree1.insert_text("Hello", "worker-1");

        let tree2 = Tree::new();
        tree2.insert_text("Goodbye", "worker-2");

        let snap2 = tree2.snapshot();
        tree1.merge_snapshot(&snap2);

        // tree1 should now have both entries
        let r1 = tree1.match_prefix_with_counts("Hello");
        assert_eq!(r1.matched_char_count, 5);

        let r2 = tree1.match_prefix_with_counts("Goodbye");
        assert_eq!(r2.matched_char_count, 7);
        assert_eq!(r2.tenant.as_ref(), "worker-2");
    }

    #[test]
    fn test_merge_overlapping_trees() {
        let tree1 = Tree::new();
        tree1.insert_text("Hello world", "worker-1");

        let tree2 = Tree::new();
        tree2.insert_text("Hello there", "worker-2");

        let snap2 = tree2.snapshot();
        tree1.merge_snapshot(&snap2);

        // tree1 should have both branches under "Hello "
        let r1 = tree1.match_prefix_with_counts("Hello world");
        assert_eq!(r1.matched_char_count, 11);

        let r2 = tree1.match_prefix_with_counts("Hello there");
        assert_eq!(r2.matched_char_count, 11);
        assert_eq!(r2.tenant.as_ref(), "worker-2");
    }

    #[test]
    fn test_iter_entries_empty_tree() {
        let tree = Tree::new();
        let entries: Vec<_> = tree.iter_entries().collect();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_iter_entries_single_insert() {
        let tree = Tree::new();
        tree.insert_text("hello", "worker-1");
        let entries: Vec<_> = tree
            .iter_entries()
            .map(|(p, ts)| (p, ts.into_iter().map(|(t, _)| t).collect::<Vec<_>>()))
            .collect();
        // Worker registration would also tag root with the tenant,
        // but we only call insert_text. Root has no tenants here;
        // the leaf "hello" carries `worker-1`.
        let leaf = entries
            .iter()
            .find(|(p, _)| p == "hello")
            .expect("leaf with tenants");
        assert_eq!(leaf.1.len(), 1);
        assert_eq!(leaf.1[0].as_ref(), "worker-1");
    }

    #[test]
    fn test_iter_entries_pre_order_and_split_nodes() {
        // Two inserts that share a prefix produce a split node.
        // The radix tree registers tenants at root and copies the
        // tenant map to intermediate split nodes, so every level
        // along the shared path emits an entry. Pre-order means
        // "" before "Hello " before its leaves; deterministic
        // char order gives "Hello there" before "Hello world"
        // (t < w).
        let tree = Tree::new();
        tree.insert_text("Hello world", "w1");
        tree.insert_text("Hello there", "w2");

        let paths: Vec<String> = tree.iter_entries().map(|(p, _)| p).collect();
        assert_eq!(paths, vec!["", "Hello ", "Hello there", "Hello world"],);
    }

    #[test]
    fn test_iter_entries_round_trips_via_snapshot() {
        // The walker yields the same `(path, tenants)` set that
        // building a tree from a snapshot would round-trip.
        let tree = Tree::new();
        tree.insert_text("alpha", "w1");
        tree.insert_text("alpha-beta", "w2");
        tree.insert_text("zulu", "w3");

        let mut from_iter: Vec<(String, Vec<String>)> = tree
            .iter_entries()
            .map(|(p, ts)| (p, ts.into_iter().map(|(t, _)| t.to_string()).collect()))
            .collect();
        from_iter.sort();

        let restored = Tree::from_snapshot(&tree.snapshot());
        let mut from_restored: Vec<(String, Vec<String>)> = restored
            .iter_entries()
            .map(|(p, ts)| (p, ts.into_iter().map(|(t, _)| t.to_string()).collect()))
            .collect();
        from_restored.sort();

        assert_eq!(from_iter, from_restored);
    }

    #[test]
    fn test_iter_entries_preserves_utf8_paths() {
        // Multi-byte chars must not produce torn paths when the
        // walker pops a frame. Path is truncated by byte length
        // — safe because each descent pushes a complete `&str`,
        // so popping the same byte count always lands on a UTF-8
        // char boundary (`String::truncate` would panic if not).
        // Also exercises the split-node case where the shared
        // "你好" prefix node emits with a clean multi-byte path.
        let tree = Tree::new();
        tree.insert_text("你好世界", "w1");
        tree.insert_text("你好朋友", "w2");

        let mut paths: Vec<String> = tree.iter_entries().map(|(p, _)| p).collect();
        paths.sort();
        assert_eq!(paths, vec!["", "你好", "你好世界", "你好朋友"]);
    }

    /// Run the same op sequence two ways — `match_prefix_with_counts` +
    /// `insert_text` (legacy pair) vs the fused `match_and_insert` — and assert
    /// the match counts and resulting per-tenant char counts agree. Timestamps
    /// and the probabilistic tenant cache are not compared.
    fn assert_fused_matches_pair(ops: &[(&str, &str)]) {
        let pair = Tree::new();
        let fused = Tree::new();
        for (text, tenant) in ops {
            let r_pair = pair.match_prefix_with_counts(text);
            pair.insert_text(text, tenant);

            let r_fused = fused.match_and_insert(text, tenant);

            assert_eq!(
                r_pair.matched_char_count, r_fused.matched_char_count,
                "matched_char_count mismatch for tenant {tenant}"
            );
            assert_eq!(
                r_pair.input_char_count, r_fused.input_char_count,
                "input_char_count mismatch"
            );
            assert_eq!(
                r_pair.tenant.as_ref(),
                r_fused.tenant.as_ref(),
                "matched tenant mismatch for input {text:?}"
            );
        }
        assert_eq!(
            pair.get_tenant_char_count(),
            fused.get_tenant_char_count(),
            "tenant char counts diverged between pair and fused"
        );
    }

    #[test]
    fn test_string_match_and_insert_equiv_basic() {
        // Note: every query here either misses on an empty tree (resolves to the
        // synthetic "empty") or falls off on a node whose `last_tenant` is
        // single-valued, so the (otherwise approximate) tenant pick is
        // deterministic and comparable between the two construction paths.
        assert_fused_matches_pair(&[
            ("hello world", "w1"), // fresh leaf
            ("hello world", "w1"), // exact re-insert (full match, loop-end)
            ("hello there", "w2"), // shared "hello " prefix -> split
            ("hello", "w3"),       // prefix of "hello " split node
        ]);
    }

    #[test]
    fn test_string_match_and_insert_equiv_deep_chain() {
        // Build a multi-node chain ("abc" -> {"def","xyz"}), then a query that
        // FULL-matches several nodes before falling off — exercising the fused
        // path's ancestor re-attach (`path`) plus a deep `insert_from` splice.
        assert_fused_matches_pair(&[
            ("abcdef", "w1"),    // leaf "abcdef"
            ("abcxyz", "w2"),    // split -> "abc" + "def"/"xyz"
            ("abcdefghi", "w1"), // full-match "abc"+"def", then append "ghi"
            ("abcdefghi", "w3"), // full-match whole chain for a new tenant
        ]);
    }

    #[test]
    fn test_string_match_and_insert_equiv_empty_and_unicode() {
        assert_fused_matches_pair(&[
            ("", "w1"),         // empty text -> tenant attached at root
            ("你好世界", "w1"), // multi-byte fresh
            ("你好朋友", "w2"), // multi-byte shared-prefix split
        ]);
    }

    #[test]
    fn test_string_match_and_insert_with_select_and_skip() {
        let text = "the quick brown fox";

        let via_with = Tree::new();
        via_with.match_and_insert_with(text, |_| Some("w1"));
        let via_plain = Tree::new();
        via_plain.match_and_insert(text, "w1");
        assert_eq!(
            via_with.get_tenant_char_count(),
            via_plain.get_tenant_char_count()
        );

        // None selection must leave the tree untouched.
        let tree = Tree::new();
        tree.insert_text(text, "w1");
        let before = tree.get_tenant_char_count();
        let r = tree.match_and_insert_with(text, |_| None);
        assert_eq!(r.matched_char_count, text.chars().count());
        assert_eq!(
            tree.get_tenant_char_count(),
            before,
            "None selection must not insert"
        );
    }

    #[test]
    fn test_evict_by_size_shared_budget_lru_tenants() {
        let tree = Tree::new();

        // 8 tenants, 10 chars each: every tenant is under the budget on its
        // own, but the tree-wide total is double the budget.
        for t in 0..8 {
            tree.insert_text(&format!("{t}aaaaaaaaa"), &format!("tenant{t}"));
        }
        assert_eq!(tree.total_char_size(), 80);

        tree.evict_tenant_by_size(40);

        assert!(tree.total_char_size() <= 40);
        // Least-recently-used tenants are evicted first.
        let sizes = tree.get_used_size_per_tenant();
        for t in 0..4 {
            assert_eq!(
                *sizes.get(&format!("tenant{t}")).unwrap_or(&0),
                0,
                "tenant{t} (LRU) should be evicted"
            );
        }
        for t in 4..8 {
            assert_eq!(
                *sizes.get(&format!("tenant{t}")).unwrap(),
                10,
                "tenant{t} should survive"
            );
        }
    }

    #[test]
    fn test_evict_by_size_under_budget_is_noop() {
        let tree = Tree::new();
        for t in 0..4 {
            tree.insert_text(&format!("{t}aaaaaaaaa"), &format!("tenant{t}"));
        }
        let counts_before = tree.get_tenant_char_count();
        assert_eq!(tree.total_char_size(), 40);

        // At and above the budget: nothing may change.
        tree.evict_tenant_by_size(40);
        tree.evict_tenant_by_size(usize::MAX);

        assert_eq!(tree.get_tenant_char_count(), counts_before);
        assert_eq!(tree.total_char_size(), 40);
        assert_eq!(tree.get_used_size_per_tenant(), counts_before);
        for t in 0..4 {
            let text = format!("{t}aaaaaaaaa");
            let (matched, tenant) = tree.prefix_match_legacy(&text);
            assert_eq!(matched, text);
            assert_eq!(tenant, format!("tenant{t}"));
        }
    }

    #[test]
    fn test_total_count_consistent_with_tenant_sum() {
        fn assert_consistent(tree: &Tree) {
            let sum: usize = tree.get_tenant_char_count().values().sum();
            assert_eq!(
                tree.total_char_size(),
                sum,
                "total must equal per-tenant sum"
            );
        }

        let tree = Tree::new();
        assert_consistent(&tree);

        // All insert paths, including a re-insert and a skipped insert.
        tree.insert_text("apple", "tenant1");
        tree.insert_text("application", "tenant2");
        tree.insert_text("apple", "tenant1");
        tree.match_and_insert("banana", "tenant3");
        tree.match_and_insert_with("bandana", |_| Some("tenant1"));
        tree.match_and_insert_with("cucumber", |_| None);
        assert_consistent(&tree);

        // Per-tenant eviction.
        tree.evict_by_tenant(&Arc::from("tenant1"), 3);
        assert_consistent(&tree);

        // Global eviction.
        tree.evict_tenant_by_size(8);
        assert_consistent(&tree);
        assert!(tree.total_char_size() <= 8);

        // Tenant purge.
        tree.remove_tenant_all(&Arc::from("tenant2"));
        assert_consistent(&tree);

        // Snapshot restore and merge maintain the invariant too.
        let restored = Tree::from_snapshot(&tree.snapshot());
        assert_consistent(&restored);

        let other = Tree::new();
        other.insert_text("delta", "tenant9");
        tree.merge_tree(&other);
        assert_consistent(&tree);

        // No-op eviction and reset.
        tree.evict_tenant_by_size(usize::MAX);
        assert_consistent(&tree);
        tree.clear();
        assert_consistent(&tree);
        assert_eq!(tree.total_char_size(), 0);
    }

    #[test]
    fn test_concurrent_match_and_insert_count_and_route_integrity() {
        const N_PREFIXES: usize = 8;
        const N_THREADS: usize = 8;
        const ITERS: usize = 3000;
        const HEAD_STEP: usize = 8;
        const TAIL_LEN: usize = 5;

        // Nested shared head "aaa…" (a shorter prefix matching inside a longer
        // node forces a split) + a disjoint divergent tail per tenant.
        let prefixes: Vec<String> = (0..N_PREFIXES)
            .map(|j| {
                let mut s = "a".repeat(HEAD_STEP * (j + 1));
                let tail = char::from(b'b' + j as u8);
                s.extend(std::iter::repeat_n(tail, TAIL_LEN));
                s
            })
            .collect();
        let tenants: Vec<String> = (0..N_PREFIXES).map(|j| format!("t{j}")).collect();
        let lens: Vec<usize> = prefixes.iter().map(|p| p.chars().count()).collect();

        let tree = Arc::new(Tree::new());
        let prefixes = Arc::new(prefixes);
        let tenants = Arc::new(tenants);

        let handles: Vec<_> = (0..N_THREADS)
            .map(|t| {
                let tree = Arc::clone(&tree);
                let prefixes = Arc::clone(&prefixes);
                let tenants = Arc::clone(&tenants);
                thread::spawn(move || {
                    for i in 0..ITERS {
                        let j = (t + i) % N_PREFIXES;
                        if t % 2 == 0 {
                            let tn = tenants[j].as_str();
                            tree.match_and_insert_with(&prefixes[j], |_| Some(tn));
                        } else {
                            tree.match_and_insert(&prefixes[j], tenants[j].as_str());
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("worker panicked (deadlock/corruption)");
        }

        // Exact, concurrency-independent count: each tenant owns its full path
        // exactly once regardless of iteration count or interleaving
        // (first-attach is atomic, split-vs-replay ordered by the text lock).
        let counts = tree.get_tenant_char_count();
        for j in 0..N_PREFIXES {
            let actual = counts.get(&tenants[j]).copied().unwrap_or(0);
            assert_eq!(
                actual, lens[j],
                "tenant {} char count diverged under concurrency: expected {}, got {actual}",
                tenants[j], lens[j]
            );
        }

        // Route integrity: every prefix is still fully cached at the end.
        for j in 0..N_PREFIXES {
            let r = tree.match_prefix_with_counts(&prefixes[j]);
            assert_eq!(
                r.matched_char_count, lens[j],
                "prefix {j} not fully cached after concurrent stress (route loss)"
            );
        }
    }

    #[test]
    fn test_remove_tenant_all_purges_tenant_and_keeps_others() {
        let tree = Tree::new();
        tree.insert_text("hello world", "tenant1");
        tree.insert_text("hello there", "tenant2");

        tree.remove_tenant_all(&Arc::from("tenant1"));

        // Count entry is gone (not merely zero), root entry included.
        assert!(!tree.get_tenant_char_count().contains_key("tenant1"));
        assert!(!tree.root.tenant_last_access_time.contains_key("tenant1"));

        // tenant1's sole-owner branch ("world") is detached from the split node.
        assert_eq!(tree.root.children.len(), 1);
        let split = Arc::clone(tree.root.children.iter().next().unwrap().value());
        assert_eq!(split.text.read().as_str(), "hello ");
        assert_eq!(split.children.len(), 1);

        // tenant2 still matches its full prefix.
        let r = tree.match_prefix_with_counts("hello there");
        assert_eq!(r.matched_char_count, "hello there".chars().count());
        assert_eq!(r.tenant.as_ref(), "tenant2");
    }

    #[test]
    fn test_remove_tenant_all_prunes_unrelated_subtrees() {
        let tree = Tree::new();
        tree.insert_text("a-target", "target");
        for depth in 1..=64 {
            tree.insert_text(&format!("z{}", "x".repeat(depth)), "other");
        }

        let tenant = Arc::from("target");
        let mut nodes = Vec::new();
        tree.collect_tenant_nodes_pruned(&tree.root, &tenant, &mut nodes);

        assert_eq!(nodes.len(), 1);

        tree.remove_tenant_all(&tenant);
        assert!(!tree.get_tenant_char_count().contains_key("target"));
        assert_eq!(
            tree.match_prefix_with_counts(&format!("z{}", "x".repeat(64)))
                .matched_char_count,
            65
        );
    }

    #[test]
    fn test_remove_tenant_all_falls_back_after_non_prefix_closed_eviction() {
        let tree = Tree::new();
        tree.insert_text("apple", "tenant1");
        tree.insert_text("application", "tenant2");

        // Targeted string eviction can remove an older ancestor before a
        // newer descendant. The optimized purge must detect that shape and
        // fall back to a complete walk.
        tree.evict_by_tenant(&Arc::from("tenant1"), 3);
        tree.remove_tenant_all(&Arc::from("tenant1"));

        assert!(!tree.get_tenant_char_count().contains_key("tenant1"));
        assert!(!tree.get_used_size_per_tenant().contains_key("tenant1"));
        assert_eq!(
            tree.match_prefix_with_counts("application")
                .matched_char_count,
            "application".chars().count()
        );
    }

    #[test]
    fn test_remove_tenant_all_sole_owner_detaches_subtree() {
        let tree = Tree::new();
        // Chain root -> "solo" -> " prefix chain", owned by tenant1 only.
        tree.insert_text("solo", "tenant1");
        tree.insert_text("solo prefix chain", "tenant1");
        assert!(!tree.root.children.is_empty());

        tree.remove_tenant_all(&Arc::from("tenant1"));

        assert!(tree.root.children.is_empty());
        assert!(!tree.get_tenant_char_count().contains_key("tenant1"));
        let r = tree.match_prefix_with_counts("solo prefix chain");
        assert_eq!(r.matched_char_count, 0);
    }
}
