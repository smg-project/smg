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

/// The `_with` closure form must behave like match-then-insert when the
/// closure always returns the same tenant (the fixed-tenant entry is this
/// closure form), and must skip the insert when it returns `None`.
#[test]
fn test_string_match_and_insert_with_select_and_skip() {
    let text = "the quick brown fox";

    // Always-insert closure == match_prefix_with_counts + insert_text.
    let via_with = Tree::new();
    let r = via_with.match_and_insert_with(text, |_| Some("w1"));
    let pair = Tree::new();
    let r2 = pair.match_prefix_with_counts(text);
    pair.insert_text(text, "w1");
    assert_eq!(r.matched_char_count, r2.matched_char_count);
    assert_eq!(
        via_with.get_tenant_char_count(),
        pair.get_tenant_char_count()
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

/// The chars `tenant` holds on nodes the root still reaches: what an
/// eviction can find and subtract, and what a match can return.
fn reachable_chars(tree: &Tree, tenant: &str) -> usize {
    fn walk(node: &NodeRef, tenant: &str) -> usize {
        node.children
            .iter()
            .map(|child| {
                let child = child.value();
                let owned = if child.tenant_last_access_time.contains_key(tenant) {
                    child.text.read().char_count()
                } else {
                    0
                };
                owned + walk(child, tenant)
            })
            .sum()
    }
    walk(&tree.root, tenant)
}

/// The nodes `text` descends through on node boundaries, root excluded.
fn path_nodes(tree: &Tree, text: &str) -> Vec<NodeRef> {
    let mut nodes = Vec::new();
    let mut current = Arc::clone(&tree.root);
    let mut remaining = text;
    while let Some(first_char) = remaining.chars().next() {
        let child = Arc::clone(current.children.get(&first_char).expect("edge").value());
        let node_text = child.text.read().as_str().to_string();
        assert!(
            remaining.starts_with(&node_text),
            "node {node_text:?} must cover the start of {remaining:?}"
        );
        remaining = &remaining[node_text.len()..];
        nodes.push(Arc::clone(&child));
        current = child;
    }
    nodes
}

/// An eviction between the match and the insert replay that empties a
/// path node also detaches it from its parent. The replay must not credit
/// the inserting tenant for that node or splice the suffix under it:
/// afterwards the tenant's char count equals the chars it holds on nodes
/// the root still reaches (what an eviction can subtract again), and the
/// prompt is cached for it.
#[test]
fn test_match_and_insert_with_detached_path_node_keeps_count_reachable() {
    let tree = Tree::new();
    let prefix = "shared system prompt";
    let w1 = intern_tenant("w1");
    tree.insert_text(prefix, "w1"); // one node; w1 is its only tenant
    assert_eq!(path_nodes(&tree, prefix).len(), 1);

    // The request matches the node in full and falls off it with more
    // text, so the node is both a replayed path node and the fall-off
    // node the suffix is spliced under.
    let text = format!("{prefix}, then a user turn");

    let result = tree.match_and_insert_with(&text, |matched| {
        assert_eq!(matched.matched_char_count, prefix.chars().count());
        assert_eq!(matched.tenant.as_ref(), "w1");
        // Between the match and the replay: evict the whole tree down to
        // nothing. No other tenant owns the node, so the eviction
        // detaches it from the root.
        tree.evict_tenant_by_size(0);
        assert!(
            tree.root.children.is_empty(),
            "the eviction detaches the emptied node"
        );
        assert_eq!(tree.tenant_char_size(&w1), 0);
        Some("w1")
    });
    assert_eq!(result.matched_char_count, prefix.chars().count());

    let reachable = reachable_chars(&tree, "w1");
    assert_eq!(
        tree.tenant_char_size(&w1),
        reachable,
        "w1's count must equal the chars it holds on reachable nodes"
    );
    assert_eq!(tree.total_char_size(), reachable);
    let cached = tree.match_prefix_with_counts(&text);
    assert_eq!(
        cached.matched_char_count,
        text.chars().count(),
        "the prompt must be cached for w1"
    );
    assert_eq!(cached.tenant.as_ref(), "w1");
}

/// Seeded traffic with an eviction injected between the match and the
/// insert replay of every request (the `select` seam): per-tenant
/// eviction, tree-wide eviction or a tenant purge, on prompts drawn from a
/// small word alphabet so paths share, split and fall empty often. After
/// every request each tenant's count equals the chars it holds on nodes
/// the root still reaches, the tree-wide total equals the sum of the
/// counts, and the prompt is cached for the tenant it was routed to.
#[test]
fn test_match_and_insert_with_counts_stay_reachable_under_injected_evictions() {
    // splitmix64: the same sequence on every platform and toolchain.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    const TENANTS: usize = 4;
    const WORDS: [&str; 6] = ["aa", "ab", "ac", "ba", "bb", "ca"];
    const WORD_CHARS: usize = 2;
    const MAX_WORDS: u64 = 4;
    const REQUESTS: usize = 2_000;

    let names: Vec<String> = (0..TENANTS).map(|i| format!("w{i}")).collect();
    let tree = Tree::new();
    let mut rng = 0x5EED_u64;
    for request in 0..REQUESTS {
        let words = 1 + (next(&mut rng) % MAX_WORDS) as usize;
        let mut prompt = String::with_capacity(words * WORD_CHARS);
        for _ in 0..words {
            prompt.push_str(WORDS[(next(&mut rng) % WORDS.len() as u64) as usize]);
        }
        let route = &names[(next(&mut rng) % TENANTS as u64) as usize];
        let victim = intern_tenant(&names[(next(&mut rng) % TENANTS as u64) as usize]);
        let budget = (next(&mut rng) % 5) as usize * WORD_CHARS;
        let eviction = next(&mut rng) % 4;

        tree.match_and_insert_with(&prompt, |_| {
            match eviction {
                0 => tree.evict_by_tenant(&victim, budget),
                1 => tree.evict_tenant_by_size(budget * TENANTS),
                2 => tree.remove_tenant_all(&victim),
                _ => {}
            }
            Some(route.as_str())
        });

        for name in &names {
            assert_eq!(
                tree.tenant_char_size(&intern_tenant(name)),
                reachable_chars(&tree, name),
                "request {request}: {name}'s count vs the chars it holds on reachable nodes"
            );
        }
        assert_eq!(
            tree.total_char_size(),
            tree.get_tenant_char_count().values().sum::<usize>(),
            "request {request}: total vs the per-tenant sum"
        );
        assert_eq!(
            tree.match_prefix_with_counts(&prompt).matched_char_count,
            prompt.chars().count(),
            "request {request}: the prompt is cached"
        );
        for node in path_nodes(&tree, &prompt) {
            assert!(
                node.tenant_last_access_time.contains_key(route.as_str()),
                "request {request}: {route} owns the whole prompt"
            );
        }
    }
}
