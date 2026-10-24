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
use rustc_hash::FxBuildHasher;
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

/// Per-tenant maps keyed by interned worker ids. Tenant ids are operator
/// controlled worker URLs, hashed on every request-path lookup, so a fast
/// non-keyed hash is the right trade; map order was already arbitrary (the
/// default hasher is randomly seeded per map).
type TenantMap<V> = DashMap<TenantId, V, FxBuildHasher>;

/// Create a tenant access time DashMap for non-root nodes
#[inline]
fn new_tenant_map() -> TenantMap<u64> {
    DashMap::with_hasher_and_shard_amount(FxBuildHasher, NODE_SHARD_COUNT)
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
    /// `MATCHED_TENANTS_CAP`): every one of them has the matched prefix,
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
static TENANT_INTERN_POOL: Lazy<DashMap<Arc<str>, (), FxBuildHasher>> =
    Lazy::new(|| DashMap::with_hasher(FxBuildHasher));

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
    tenant_last_access_time: TenantMap<u64>,
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

    /// No tenant owns this node and nothing hangs below it: the shape an
    /// eviction detaches from its parent.
    fn is_empty(&self) -> bool {
        self.children.is_empty() && self.tenant_last_access_time.is_empty()
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
    tenant_char_count: TenantMap<usize>,
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

/// Common prefix of `a` and `b` as (chars, bytes), so the caller can slice
/// past it without a second walk over the matched text.
#[inline]
fn shared_prefix_len(a: &str, b: &str) -> (usize, usize) {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let common_byte_len = a_bytes
        .iter()
        .zip(b_bytes)
        .position(|(&a_byte, &b_byte)| a_byte != b_byte)
        .unwrap_or_else(|| a_bytes.len().min(b_bytes.len()));
    if a_bytes[..common_byte_len].is_ascii() {
        return (common_byte_len, common_byte_len);
    }
    // The byte-level prefix may end inside a character: count whole chars.
    let mut chars = 0;
    let mut bytes = 0;
    for ((offset, a_char), b_char) in a.char_indices().zip(b.chars()) {
        if a_char != b_char {
            break;
        }
        chars += 1;
        bytes = offset + a_char.len_utf8();
    }
    (chars, bytes)
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
                tenant_last_access_time: DashMap::with_hasher_and_shard_amount(
                    FxBuildHasher,
                    ROOT_SHARD_COUNT,
                ),
                parent: RwLock::new(None),
                last_tenant: RwLock::new(None),
            }),
            tenant_char_count: DashMap::with_hasher_and_shard_amount(
                FxBuildHasher,
                ROOT_SHARD_COUNT,
            ),
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
    ///
    /// [`Self::match_and_insert_with`] resumes below the root only while its
    /// replay found every recorded edge in place; after a lost edge (an eviction
    /// detached a path node while the caller chose the tenant) it descends again
    /// from the root over the whole text, and this method credits only the edges
    /// it newly attaches, so nothing is counted twice.
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

        // Compute input char count directly from input text.
        // This is equivalent to matched_chars + remaining.chars().count() but avoids
        // needing to track remaining precisely through the traversal.
        let input_char_count = text.chars().count();

        // The tenant pick (the `last_tenant` cache first, the tenant map on a
        // miss), its cache populate on that slow path, the probabilistic
        // timestamp touch and the result are the tail the fused walk shares
        // with this match.
        let (tenant, took_slow_path) = self.resolve_tenant_readonly(&curr);
        self.finish_match(
            &curr,
            tenant,
            took_slow_path,
            matched_chars,
            input_char_count,
        )
    }

    /// Read-only resolution of a node's "owning" tenant: the `last_tenant`
    /// cache when it still names a tenant of the node, else the first tenant
    /// of the node's map (the slow path), **without** the cache-populate or
    /// probabilistic timestamp side effects. Returns the tenant plus whether
    /// the slow path was taken; [`Self::finish_match`] replays the single
    /// deferred side effect on the final node.
    ///
    /// Both [`Self::match_prefix_with_counts`] and the fused walk behind
    /// [`Self::match_and_insert_with`] resolve through it; the fused walk does
    /// so **before** its insert adds the inserting tenant to the node, which
    /// keeps the "match runs fully before insert" ordering for the (otherwise
    /// non-deterministic) slow-path tenant pick.
    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the tree; the match's tenant resolution"
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
    #[inline]
    pub fn match_and_insert_with<'t, F>(&self, text: &str, select: F) -> PrefixMatchResult
    where
        F: FnOnce(&PrefixMatchResult) -> Option<&'t str>,
    {
        // `select` runs exactly once; the Option carries the FnOnce through
        // the FnMut interface of the non-generic walk below.
        let mut select = Some(select);
        self.match_and_insert_dyn(text, &mut |result| {
            select.take().and_then(|select| select(result))
        })
    }

    /// The whole fused descent, behind one indirect call. It is deliberately
    /// not generic over the closure: a generic walk is monomorphized into
    /// every calling crate and compiled at that crate's optimization level,
    /// whereas this body is compiled once, here, at this crate's.
    fn match_and_insert_dyn<'t>(
        &self,
        text: &str,
        select: &mut dyn FnMut(&PrefixMatchResult) -> Option<&'t str>,
    ) -> PrefixMatchResult {
        // ---- Phase 1: MATCH descent (mirrors match_prefix_with_counts) ----
        // Record the full-match nodes (for ancestor re-attach) and the fall-off
        // point (node + remaining slice) for the suffix splice. No mutation here
        // beyond match's own deferred touch below, so the tenant pick sees the
        // un-polluted tree exactly like the standalone match.
        let mut remaining = text;
        let mut matched_chars = 0usize;
        // Chars consumed by full-match nodes; with the unmatched tail this gives
        // the input length without another pass over `text`.
        let mut consumed_chars = 0usize;
        // Every full-match node we descended through, in order. Each node is
        // cloned once, into `path`; `current` indexes it (`None` = root).
        let mut path: Vec<NodeRef> = Vec::with_capacity(16);
        // The char each edge of `path` was descended by, so the replay can
        // confirm an edge still exists before it attaches a tenant under it.
        let mut keys: Vec<char> = Vec::with_capacity(16);
        let mut current: Option<usize> = None;
        // The child the match ended inside, when it ended on a partial match.
        let mut partial: Option<NodeRef> = None;

        while let Some(first_char) = remaining.chars().next() {
            let current_node = match current {
                Some(i) => &path[i],
                None => &self.root,
            };
            let child_node = current_node
                .children
                .get(&first_char)
                .map(|e| e.value().clone());

            let Some(matched_node) = child_node else {
                // No child for this char: match stops at `current`.
                break;
            };

            let matched_text_guard = matched_node.text.read();
            // Stale-edge check (see `match_prefix_with_counts`).
            if !node_parent_is(&matched_node, current_node) {
                drop(matched_text_guard);
                std::hint::spin_loop();
                continue;
            }
            let matched_node_text_count = matched_text_guard.char_count();
            let (shared_count, shared_bytes) =
                shared_prefix_len(remaining, matched_text_guard.as_str());
            drop(matched_text_guard);

            if shared_count == matched_node_text_count {
                // Full match -> continue. Record for ancestor re-attach.
                matched_chars += shared_count;
                consumed_chars += shared_count;
                remaining = &remaining[shared_bytes..];
                path.push(matched_node);
                keys.push(first_char);
                current = Some(path.len() - 1);
            } else {
                // Partial match: match stops, resolving on this child node.
                matched_chars += shared_count;
                partial = Some(matched_node);
                // `current` stays the parent — the splice re-probes it and
                // splits the partial child exactly like `insert_text`.
                break;
            }
        }
        let input_char_count = consumed_chars + remaining.chars().count();

        // ---- Match side effect + result (verbatim match_prefix_with_counts) ----
        // The node match resolves its tenant on: the partial child, the deepest
        // full-match node, or the root if nothing matched.
        let match_curr: &NodeRef = match (&partial, current) {
            (Some(node), _) => node,
            (None, Some(i)) => &path[i],
            (None, None) => &self.root,
        };
        let (match_tenant, match_slow) = self.resolve_tenant_readonly(match_curr);
        let result = self.finish_match(
            match_curr,
            match_tenant,
            match_slow,
            matched_chars,
            input_char_count,
        );

        // ---- Decide the insert tenant from the match result ----
        let Some(tenant) = select(&result) else {
            return result;
        };
        // Intern through the shared pool so the stored Arc dedups (matches
        // insert_text's interning).
        let tenant_id = intern_tenant(tenant);

        // ---- Phase 2: INSERT for `tenant_id` without re-walking the prefix ----
        // Root bookkeeping (mirrors insert_text): ensure-present, so read first
        // and take the write-locked entry only when it is missing.
        if !self
            .root
            .tenant_last_access_time
            .contains_key(tenant_id.as_ref())
        {
            self.root
                .tenant_last_access_time
                .entry(Arc::clone(&tenant_id))
                .or_insert(0);
        }
        if !self.tenant_char_count.contains_key(tenant_id.as_ref()) {
            self.tenant_char_count
                .entry(Arc::clone(&tenant_id))
                .or_insert(0);
        }

        // Re-attach the inserting tenant to every full-match ancestor node, the
        // epoch-0 intermediate attach `insert_text` performs while descending.
        //
        // Nothing held the path while `select` ran. An eviction that emptied a
        // path node meanwhile has detached it from its parent, and a tenant
        // attached to a detached node is credited chars no eviction can find
        // again. So each attach below is made under the parent's child entry
        // (its shard read guard), after confirming the entry still names the
        // node: the detach takes the same shard's write lock and removes the
        // node only while it is still empty (see `detach_if_empty`), so the
        // two serialize. A detach that came first shows here as a lost edge;
        // an attach that came first leaves a node the detach keeps.
        let mut lost_edge = false;
        for (i, node) in path.iter().enumerate() {
            // Already attached (the common hit: the request routes to the
            // worker it matched): nothing to credit, no write lock.
            if node
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref())
            {
                continue;
            }
            let parent = if i == 0 { &self.root } else { &path[i - 1] };
            let edge = parent.children.get(&keys[i]);
            if !edge
                .as_ref()
                .is_some_and(|edge| Arc::ptr_eq(edge.value(), node))
            {
                lost_edge = true;
                break;
            }
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
            drop(text_guard);
            drop(edge);
        }

        if lost_edge {
            // A path node left the tree between the match and the replay. The
            // nodes credited above are attached and live; descend again from
            // the root for the whole text. `insert_from` credits only the
            // edges it newly attaches, so nothing is counted twice, and it
            // splices the suffix under live nodes, so the prompt is cached.
            // One extra descent, in the rare case.
            self.insert_from(Arc::clone(&self.root), text, tenant_id);
        } else if remaining.is_empty() {
            // Loop-end: `current` is the final leaf; give it the real timestamp
            // (insert_text's tail). It already received the epoch-0 attach above
            // if it is on the path.
            let epoch = get_epoch();
            let leaf = match current {
                Some(i) => &path[i],
                None => &self.root,
            };
            leaf.tenant_last_access_time
                .insert(Arc::clone(&tenant_id), epoch);
        } else {
            // Fall-off: splice only the unmatched suffix below `current`,
            // reusing insert_text's exact loop (handles split-continue + the
            // final-leaf real timestamp). The matched prefix is not re-walked.
            let fall_off = match current {
                Some(i) => Arc::clone(&path[i]),
                None => Arc::clone(&self.root),
            };
            self.insert_from(fall_off, remaining, tenant_id);
        }

        result
    }

    /// The match's tail, shared by [`Self::match_prefix_with_counts`] and the
    /// fused walk behind [`Self::match_and_insert_with`]: replay the single
    /// deferred side effect of the tenant resolution (populate the
    /// `last_tenant` cache on the slow path), then the probabilistic 1-in-8
    /// timestamp touch on the resolved final node, and build the match result,
    /// which takes `tenant` as it is (the callers hand over the id they
    /// resolved, so the tail costs what the inline one did).
    #[expect(
        clippy::unused_self,
        reason = "method logically belongs to the tree; the match's tail"
    )]
    fn finish_match(
        &self,
        match_node: &NodeRef,
        tenant: TenantId,
        took_slow_path: bool,
        matched_chars: usize,
        input_char_count: usize,
    ) -> PrefixMatchResult {
        // On the slow path the original resolution populates the cache.
        if took_slow_path {
            *match_node.last_tenant.write() = Some(Arc::clone(&tenant));
        }

        // Probabilistic (1 in 8) timestamp touch on the resolved tenant, to keep
        // the contention on the tenant map low (LRU eviction needs no exact
        // timestamps), skipping the synthetic "empty" tenant so a tenant no
        // insert ever named does not enter the tree.
        let epoch = get_epoch();
        if epoch & 0x7 == 0 && tenant.as_ref() != "empty" {
            match_node
                .tenant_last_access_time
                .insert(Arc::clone(&tenant), epoch);
        }

        PrefixMatchResult {
            tenant,
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

            // Detach the node once it is empty (`detach_if_empty` repeats the
            // checks under the parent's shard lock).
            if node.is_empty() {
                if let Some(ref parent) = parent_opt {
                    Self::detach_if_empty(parent, &node);
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
        while current.is_empty() {
            let Some(parent) = current.parent.read().as_ref().and_then(Weak::upgrade) else {
                break;
            };
            Self::detach_if_empty(&parent, &current);
            current = parent;
        }
    }

    /// Remove `node` from `parent`'s children, only while the parent still holds
    /// this exact node under its first char and the node is still empty. The
    /// children map is concurrent: an insert can install a split's prefix node
    /// under the same char after the caller's emptiness check, and the fused
    /// walk's insert replay attaches a tenant to this very node under this
    /// entry's shard read guard once it has confirmed the edge (see
    /// `match_and_insert_dyn`). Removing the key unconditionally would drop that
    /// live replacement, or a node whose tenant was just credited for it, with
    /// every descendant it owns: the index silently shrinks or the credit leaks.
    /// `remove_if` evaluates the predicate under the shard lock, so the checks
    /// and the removal cannot be split. The first char is read before the entry
    /// is taken: a split writes this node's text while it holds the parent's
    /// entry, the reverse order.
    fn detach_if_empty(parent: &NodeRef, node: &NodeRef) {
        let first_char = node.text.read().first_char();
        if let Some(first_char) = first_char {
            parent.children.remove_if(&first_char, |_, child| {
                Arc::ptr_eq(child, node) && child.is_empty()
            });
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
        self.merge_remote_tenants(local, &remote.tenant_last_access_time, || {
            local.text.read().char_count()
        });

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
                    self.merge_remote_tenants(
                        &local_child,
                        &remote_child.tenant_last_access_time,
                        || local_edge.chars().count(),
                    );

                    let remote_remainder = advance_by_chars(&remote_edge, shared);
                    let Some(rem_first) = remote_remainder.chars().next() else {
                        continue;
                    };

                    // The remote continues below the local child's edge. Merge
                    // the remainder as the one child of a tenant-less virtual
                    // entry standing in for the local child: the children loop
                    // then compares the remainder's edge against the local
                    // child's own child under the same first char (exact,
                    // prefix or diverging) and grafts it when there is none.
                    // The two entries share that first char only, so they are
                    // never merged as one path.
                    let trimmed_remote = Self::clone_subtree_as(
                        &remote_child,
                        NodeText::new(remote_remainder.to_string()),
                        None,
                    );
                    let below_local_child = Arc::new(Node {
                        children: new_children_map(),
                        text: RwLock::new(NodeText::empty()),
                        tenant_last_access_time: new_tenant_map(),
                        parent: RwLock::new(None),
                        last_tenant: RwLock::new(None),
                    });
                    below_local_child.children.insert(rem_first, trimmed_remote);
                    self.merge_nodes(&local_child, &below_local_child);
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
                    self.merge_remote_tenants(
                        &split_node,
                        &remote_child.tenant_last_access_time,
                        || split_chars,
                    );

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

                    // Publish the split entry now: until it replaces the local
                    // child under `local`, a walker through this edge finds the
                    // truncated child re-parented elsewhere and spins on the
                    // stale edge; the remote side below is grafted or merged
                    // into the live split entry after this.
                    local.children.insert(rc, Arc::clone(&split_node));

                    // Create remote remainder as another child of split node,
                    // its children cloned under it with their parent pointers
                    // rewritten.
                    let remote_remainder = advance_by_chars(&remote_edge, shared);
                    if let Some(rem_first) = remote_remainder.chars().next() {
                        let remote_subtree = Self::clone_subtree_as(
                            &remote_child,
                            NodeText::new(remote_remainder.to_string()),
                            Some(&split_node),
                        );
                        self.accumulate_tenant_counts(&remote_subtree);
                        split_node.children.insert(rem_first, remote_subtree);
                    } else {
                        // The remote edge ended at the split point (it is a
                        // prefix of the local edge): its children belong under
                        // the split entry, each compared against the local
                        // remainder in the children loop; the tenants re-merge
                        // as a no-op (same tenants, same epochs).
                        self.merge_nodes(&split_node, &remote_child);
                    }
                }
            } else {
                // No local child at this char — copy entire remote subtree
                let cloned = Self::clone_subtree(&remote_child, Some(local));
                self.accumulate_tenant_counts(&cloned);
                local.children.insert(rc, cloned);
            }
        }
    }

    /// Merge a remote node's tenants into `node`: a tenant the node does not
    /// carry is attached with the remote epoch and credited the node's edge
    /// (`edge_chars`, read only then); a tenant both carry keeps the newer
    /// epoch.
    fn merge_remote_tenants(
        &self,
        node: &NodeRef,
        remote_tenants: &TenantMap<u64>,
        edge_chars: impl Fn() -> usize,
    ) {
        for entry in remote_tenants {
            let tenant_id = Arc::clone(entry.key());
            let remote_epoch = *entry.value();

            let is_new = !node
                .tenant_last_access_time
                .contains_key(tenant_id.as_ref());

            let should_update = node
                .tenant_last_access_time
                .get(tenant_id.as_ref())
                .map(|local_epoch| remote_epoch > *local_epoch)
                .unwrap_or(true);

            if should_update {
                node.tenant_last_access_time
                    .insert(Arc::clone(&tenant_id), remote_epoch);

                if is_new {
                    self.add_tenant_chars(&tenant_id, edge_chars());
                }
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
        let text = node.text.read().clone_text();
        Self::clone_subtree_as(node, text, parent)
    }

    /// Deep clone a subtree whose root carries `text` instead of the node's own
    /// edge (a remote edge trimmed to the part below a local prefix), setting
    /// parent pointers correctly.
    fn clone_subtree_as(node: &NodeRef, text: NodeText, parent: Option<&NodeRef>) -> NodeRef {
        let new_node = Arc::new(Node {
            children: new_children_map(),
            text: RwLock::new(text),
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
mod tests;
