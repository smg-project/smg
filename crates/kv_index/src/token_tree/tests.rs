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
            let mut tenants: Vec<&str> = result.matched_tenants.iter().map(AsRef::as_ref).collect();
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

/// The `_with` closure form must behave like match-then-insert when the
/// closure always returns the same tenant (the fixed-tenant entry is this
/// closure form), and must skip the insert (leaving the tree unchanged) when
/// it returns `None`.
#[test]
fn test_match_and_insert_with_select_and_skip() {
    let tokens = make_tokens(1, 3);

    // Always-insert closure == match_prefix_with_counts + insert_tokens.
    let via_with = TokenTree::new();
    let r = via_with.match_and_insert_with(&tokens, |_| Some("w1"));
    let pair = TokenTree::new();
    let r2 = pair.match_prefix_with_counts(&tokens);
    pair.insert_tokens(&tokens, "w1");
    assert_eq!(r.matched_token_count, r2.matched_token_count);
    assert_eq!(
        via_with.get_tenant_token_counts(),
        pair.get_tenant_token_counts()
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

/// A node the eviction walk still holds can be replaced under the same page
/// key by a concurrent store. Detaching by key alone would delete the
/// replacement and every token beneath it, so the walk must check identity.
#[test]
fn stale_cleanup_keeps_a_replacement_at_the_same_page_key() {
    let tree = TokenTree::new();
    let tokens = make_tokens(700, 2);
    let page_key = page_key_of(&tokens, PAGE_SIZE);
    let victim: TenantId = Arc::from("victim");

    tree.insert_tokens(&tokens, "victim");
    let stale = tree
        .root
        .children
        .get(&page_key)
        .map(|entry| Arc::clone(entry.value()))
        .expect("inserted child");

    // The walk observed this node empty, and the tree moved on without it.
    stale.tenant_last_access_time.clear();
    tree.root.children.remove(&page_key);

    // A concurrent store installs a different node under the same page key.
    tree.insert_tokens(&tokens, "survivor");
    let replacement = tree
        .root
        .children
        .get(&page_key)
        .map(|entry| Arc::clone(entry.value()))
        .expect("replacement child");
    assert!(!Arc::ptr_eq(&stale, &replacement));

    // The walk runs late, still holding the node it collected earlier.
    tree.cleanup_empty_ancestors_with_parent(&stale, &victim);

    let survivor = tree
        .root
        .children
        .get(&page_key)
        .map(|entry| Arc::clone(entry.value()));
    assert!(
        survivor.is_some_and(|node| Arc::ptr_eq(&node, &replacement)),
        "stale cleanup detached the replacement node"
    );
    assert_eq!(
        tree.match_prefix_with_counts(&tokens).matched_token_count,
        tokens.len(),
        "replacement subtree is no longer reachable"
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

/// The nodes a page-aligned sequence descends through, root excluded.
fn path_nodes(tree: &TokenTree, tokens: &[TokenId]) -> Vec<NodeRef> {
    let mut nodes = Vec::new();
    let mut current = Arc::clone(&tree.root);
    let mut remaining = &tokens[..align_to_page(tokens.len(), tree.page_size)];
    while remaining.len() >= tree.page_size {
        let key = page_key_of(remaining, tree.page_size);
        let child = Arc::clone(current.children.get(&key).expect("edge").value());
        let len = child.tokens.read().len();
        remaining = &remaining[len..];
        nodes.push(Arc::clone(&child));
        current = child;
    }
    nodes
}

/// The tokens `tenant` holds on nodes the root still reaches: what an
/// eviction can find and subtract, and what a match can return.
fn reachable_tokens(tree: &TokenTree, tenant: &str) -> usize {
    fn walk(node: &NodeRef, tenant: &str) -> usize {
        node.children
            .iter()
            .map(|child| {
                let child = child.value();
                let owned = if child.tenant_last_access_time.contains_key(tenant) {
                    child.tokens.read().len()
                } else {
                    0
                };
                owned + walk(child, tenant)
            })
            .sum()
    }
    walk(&tree.root, tenant)
}

fn hit_counts(nodes: &[NodeRef]) -> Vec<u64> {
    nodes
        .iter()
        .map(|node| node.hit_count.load(Ordering::Relaxed))
        .collect()
}

/// Under LFU the fused call must count hits exactly like match-then-insert:
/// every full-match node gets one hit from the match and one from the
/// insert replay, whether the request routes to the tenant it matched or
/// to another one.
#[test]
fn test_match_and_insert_with_lfu_hit_counts_match_two_pass() {
    let short = make_tokens(1, 2);
    let long = make_tokens(1, 3); // splits `short` into a 2-page node + 1-page child

    let fused = TokenTree::with_config(PAGE_SIZE, EvictionPolicy::Lfu);
    let two_pass = TokenTree::with_config(PAGE_SIZE, EvictionPolicy::Lfu);
    for tree in [&fused, &two_pass] {
        tree.insert_tokens(&short, "w1");
        tree.insert_tokens(&long, "w1");
    }
    assert_eq!(path_nodes(&fused, &long).len(), 2);

    // Routed back to the matched tenant, then routed elsewhere.
    for route in ["w1", "w2"] {
        fused.match_and_insert_with(&long, |_| Some(route));
        two_pass.match_prefix_with_counts(&long);
        two_pass.insert_tokens(&long, route);
        assert_eq!(
            hit_counts(&path_nodes(&fused, &long)),
            hit_counts(&path_nodes(&two_pass, &long)),
            "route to {route}"
        );
        assert_eq!(
            fused.get_tenant_token_counts(),
            two_pass.get_tenant_token_counts()
        );
    }
    // The 2-page node: 2 inserts + 2 per request; its child: 1 insert + 2
    // per request. One hit from the match and one from the replay each time.
    assert_eq!(hit_counts(&path_nodes(&fused, &long)), vec![6, 5]);
}

/// A tenant removed from the path between the match and the insert replay
/// (an eviction racing the request) is re-attached and credited by the
/// replay, as it was when the insert re-walked the tree. The request routes
/// to whichever tenant the match stamped, which is what makes the replay
/// take its skip branch; routing to the other tenant would exercise the
/// ordinary `touch_tenant` path whatever the guard does.
#[test]
fn test_match_and_insert_with_reattaches_after_eviction_during_select() {
    let tokens = make_tokens(1, 3);
    let tree = TokenTree::new();
    tree.insert_tokens(&tokens, "w1");
    tree.insert_tokens(&tokens, "w2"); // the node outlives either removal
                                       // One node holds the whole sequence, so the match stamps exactly one
                                       // tenant and reports it as `matched.tenant`.
    assert_eq!(path_nodes(&tree, &tokens).len(), 1);

    let mut evicted: Option<&'static str> = None;
    let result = tree.match_and_insert_with(&tokens, |matched| {
        assert_eq!(matched.matched_token_count, tokens.len());
        // Which tenant the match stamped depends on the node's cached
        // `last_tenant` and map order; take whichever it was.
        let stamped: &'static str = match matched.tenant.as_ref() {
            "w1" => "w1",
            "w2" => "w2",
            other => panic!("unexpected tenant {other}"),
        };
        evicted = Some(stamped);
        // Between the match (which stamped this tenant on the path) and
        // the replay: strip it from the whole tree, then route to it.
        tree.remove_tenant_all(&intern_tenant(stamped));
        Some(stamped)
    });
    assert_eq!(result.matched_token_count, tokens.len());
    let evicted = evicted.expect("select ran");
    let other = if evicted == "w1" { "w2" } else { "w1" };

    for node in path_nodes(&tree, &tokens) {
        assert!(
            node.tenant_last_access_time.contains_key(evicted),
            "replay must re-attach the evicted tenant {evicted}"
        );
        assert!(node.tenant_last_access_time.contains_key(other));
    }
    let counts = tree.get_tenant_token_counts();
    assert_eq!(
        counts.get(evicted),
        Some(&tokens.len()),
        "replay must re-credit the path to {evicted}"
    );
    assert_eq!(counts.get(other), Some(&tokens.len()));
}

/// An eviction between the match and the insert replay that empties a
/// path node also detaches it from its parent. The replay must not credit
/// the inserting tenant for that node or splice the suffix under it:
/// afterwards the tenant's token count equals the tokens it holds on
/// nodes the root still reaches (what an eviction can subtract again),
/// and the prompt is cached for it.
#[test]
fn test_match_and_insert_with_detached_path_node_keeps_count_reachable() {
    let tree = TokenTree::new();
    let prefix = make_tokens(1, 2);
    let w1 = intern_tenant("w1");
    tree.insert_tokens(&prefix, "w1"); // one 2-page node; w1 is its only tenant
    assert_eq!(path_nodes(&tree, &prefix).len(), 1);

    // The request matches the node in full and falls off it with one more
    // page, so the node is both a replayed path node and the fall-off
    // node the suffix is spliced under.
    let mut tokens = prefix.clone();
    tokens.extend(make_tokens(9_001, 1));

    let result = tree.match_and_insert_with(&tokens, |matched| {
        assert_eq!(matched.matched_token_count, prefix.len());
        assert_eq!(matched.tenant.as_ref(), "w1");
        // Between the match (which stamped w1 on the node) and the
        // replay: evict everything w1 holds. No other tenant owns the
        // node, so the eviction detaches it from the root.
        tree.evict_tenant(&w1, 0);
        assert!(
            tree.root.children.is_empty(),
            "the eviction detaches the emptied node"
        );
        assert_eq!(tree.tenant_token_size(&w1), 0);
        Some("w1")
    });
    assert_eq!(result.matched_token_count, prefix.len());

    let reachable = reachable_tokens(&tree, "w1");
    assert_eq!(
        tree.tenant_token_size(&w1),
        reachable,
        "w1's count must equal the tokens it holds on reachable nodes"
    );
    assert_eq!(tree.total_token_size(), reachable);
    let cached = tree.match_prefix_with_counts(&tokens);
    assert_eq!(
        cached.matched_token_count,
        tokens.len(),
        "the prompt must be cached for w1"
    );
    assert_eq!(cached.tenant.as_ref(), "w1");
}

/// Seeded traffic with an eviction injected between the match and the
/// insert replay of every request (the `select` seam): per-tenant
/// eviction, tree-wide eviction or a tenant purge, on prompts drawn from a
/// small page alphabet so paths share, split and fall empty often. After
/// every request each tenant's count equals the tokens it holds on nodes
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
    const PAGE_ALPHABET: u32 = 6;
    const MAX_PAGES: u64 = 4;
    const REQUESTS: usize = 2_000;

    let names: Vec<String> = (0..TENANTS).map(|i| format!("w{i}")).collect();
    let tree = TokenTree::new();
    let mut rng = 0x5EED_u64;
    for request in 0..REQUESTS {
        let pages = 1 + (next(&mut rng) % MAX_PAGES) as usize;
        let mut prompt = Vec::with_capacity(pages * PAGE_SIZE);
        for _ in 0..pages {
            let page = (next(&mut rng) % u64::from(PAGE_ALPHABET)) as u32;
            prompt.extend(make_tokens(1 + page * 1_000, 1));
        }
        let route = &names[(next(&mut rng) % TENANTS as u64) as usize];
        let victim = intern_tenant(&names[(next(&mut rng) % TENANTS as u64) as usize]);
        let budget = (next(&mut rng) % 5) as usize * PAGE_SIZE;
        let eviction = next(&mut rng) % 4;

        tree.match_and_insert_with(&prompt, |_| {
            match eviction {
                0 => tree.evict_tenant(&victim, budget),
                1 => tree.evict_tenant_by_size(budget * TENANTS),
                2 => tree.remove_tenant_all(&victim),
                _ => {}
            }
            Some(route.as_str())
        });

        for name in &names {
            assert_eq!(
                tree.tenant_token_size(&intern_tenant(name)),
                reachable_tokens(&tree, name),
                "request {request}: {name}'s count vs the tokens it holds on reachable nodes"
            );
        }
        assert_eq!(
            tree.total_token_size(),
            tree.get_tenant_token_counts().values().sum::<usize>(),
            "request {request}: total vs the per-tenant sum"
        );
        assert_eq!(
            tree.match_prefix_with_counts(&prompt).matched_token_count,
            prompt.len(),
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

/// `touch_any_tenant` reports only a tenant it stamped in this call. The
/// pick and the stamp share one guard, so an eviction between them cannot
/// make the match name a tenant that no longer owns the node, and a match
/// never re-attaches an evicted tenant.
#[test]
fn test_touch_any_tenant_reports_only_a_stamped_tenant() {
    let node = Node::new(make_tokens(1, 1));
    let (w1, w2) = (intern_tenant("w1"), intern_tenant("w2"));
    node.touch_tenant(&w1, 1, false);
    node.touch_tenant(&w2, 2, false);

    // Cached fast path: the cached tenant still owns the node.
    *node.last_tenant.write() = Some(Arc::clone(&w1));
    let picked = node.touch_any_tenant(10, false).expect("owned node");
    assert!(Arc::ptr_eq(&picked, &w1));
    assert_eq!(*node.tenant_last_access_time.get("w1").expect("w1"), 10);
    assert_eq!(*node.tenant_last_access_time.get("w2").expect("w2"), 2);

    // The cached tenant was evicted: the fallback picks a tenant that
    // owns the node, stamps it, and does not bring the evicted one back.
    node.tenant_last_access_time.remove("w1");
    let picked = node
        .touch_any_tenant(32, false)
        .expect("w2 still owns the node");
    assert!(Arc::ptr_eq(&picked, &w2), "picked {picked}");
    assert_eq!(*node.tenant_last_access_time.get("w2").expect("w2"), 32);
    assert!(
        !node.tenant_last_access_time.contains_key("w1"),
        "a match must not re-attach an evicted tenant"
    );
    // 32 & 0xF == 0: the cache refresh names the tenant actually stamped.
    assert!(node
        .last_tenant
        .read()
        .as_ref()
        .is_some_and(|t| Arc::ptr_eq(t, &w2)));

    // No cache, two owners: evict whichever iterates first; the pick is
    // the other one, stamped.
    node.touch_tenant(&w1, 40, false);
    *node.last_tenant.write() = None;
    let first_in_order = node
        .tenant_last_access_time
        .iter()
        .next()
        .map(|entry| Arc::clone(entry.key()))
        .expect("two owners");
    let other = if first_in_order.as_ref() == "w1" {
        &w2
    } else {
        &w1
    };
    node.tenant_last_access_time.remove(first_in_order.as_ref());
    let picked = node.touch_any_tenant(41, false).expect("one owner left");
    assert!(Arc::ptr_eq(&picked, other), "picked {picked}");
    assert_eq!(
        *node
            .tenant_last_access_time
            .get(other.as_ref())
            .expect("other"),
        41
    );
    assert!(!node
        .tenant_last_access_time
        .contains_key(first_in_order.as_ref()));

    // All evicted: nothing to pick, nothing stamped or attached.
    node.tenant_last_access_time.remove(other.as_ref());
    assert!(node.touch_any_tenant(42, false).is_none());
    assert!(node.tenant_last_access_time.is_empty());
}
