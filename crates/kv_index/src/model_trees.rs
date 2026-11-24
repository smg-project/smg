//! One string tree and one token tree per model.

use std::sync::Arc;

use dashmap::DashMap;
use rustc_hash::FxBuildHasher;

use crate::{TenantId, TokenTree, Tree};

/// The trees of one router, by model: a string tree for text prompts and a
/// token tree for tokenized ones. A model's tree is made when the model is
/// first seen and is never replaced, so a handle on it stays the model's
/// tree.
///
/// A tenant is a worker, by the name its router sends to. The owner bounds
/// the trees ([`Self::evict_string_trees`], [`Self::evict_token_trees`]) and
/// takes a worker that left out of them ([`Self::remove_tenant`],
/// [`Self::remove_tenants_from_model`]).
#[derive(Debug)]
pub struct ModelTrees {
    string_trees: DashMap<String, Arc<Tree>, FxBuildHasher>,
    token_trees: DashMap<String, Arc<TokenTree>, FxBuildHasher>,
    /// The page of every token tree made here, in tokens.
    page_size: usize,
}

/// The per-model tree, created on first sight.
///
/// Reads first: `DashMap::entry` takes the shard's *exclusive* lock (and a
/// `String` key allocation) on every call, and every request for a model lands
/// on the same shard, so the selection hot path would serialize on it. A shared
/// read plus an `Arc` clone is uncontended; only a model's first request pays
/// for the insert.
fn tree_for_model<V>(
    trees: &DashMap<String, Arc<V>, FxBuildHasher>,
    model_id: &str,
    new_tree: impl FnOnce() -> V,
) -> Arc<V> {
    if let Some(tree) = trees.get(model_id) {
        return Arc::clone(tree.value());
    }
    trees
        .entry(model_id.to_string())
        .or_insert_with(|| Arc::new(new_tree()))
        .value()
        .clone()
}

impl ModelTrees {
    /// No tree yet. A token tree made here holds pages of `page_size`
    /// tokens, one at least.
    pub fn new(page_size: usize) -> Self {
        // Keyed by the operator's model ids and read on every request: the Fx
        // hasher.
        Self {
            string_trees: DashMap::with_hasher(FxBuildHasher),
            token_trees: DashMap::with_hasher(FxBuildHasher),
            page_size: page_size.max(1),
        }
    }

    /// `model_id`'s string tree, where one was made.
    pub fn string_tree(&self, model_id: &str) -> Option<Arc<Tree>> {
        self.string_trees
            .get(model_id)
            .map(|tree| Arc::clone(tree.value()))
    }

    /// `model_id`'s token tree, where one was made.
    pub fn token_tree(&self, model_id: &str) -> Option<Arc<TokenTree>> {
        self.token_trees
            .get(model_id)
            .map(|tree| Arc::clone(tree.value()))
    }

    /// `model_id`'s string tree, made on first sight.
    pub fn string_tree_or_new(&self, model_id: &str) -> Arc<Tree> {
        tree_for_model(&self.string_trees, model_id, Tree::new)
    }

    /// `model_id`'s token tree, made on first sight.
    pub fn token_tree_or_new(&self, model_id: &str) -> Arc<TokenTree> {
        tree_for_model(&self.token_trees, model_id, || {
            TokenTree::with_config(self.page_size, Default::default())
        })
    }

    /// How many models have a string tree.
    pub fn string_tree_count(&self) -> usize {
        self.string_trees.len()
    }

    /// How many models have a token tree.
    pub fn token_tree_count(&self) -> usize {
        self.token_trees.len()
    }

    /// Every model's token tree, copied out: a pass of the owner's over the
    /// trees holds no lock of this store, so a model's first tree is not
    /// kept waiting behind it.
    pub fn token_trees(&self) -> Vec<Arc<TokenTree>> {
        self.token_trees
            .iter()
            .map(|tree| Arc::clone(tree.value()))
            .collect()
    }

    /// Inserts the empty prompt for `tenant` in the two trees of `model_id`,
    /// making them where the model is new. The string tree then holds the
    /// tenant at its root and routes to it before it has served a request.
    /// The token tree stores no prompt under one page, so it holds nothing
    /// for the tenant yet.
    pub fn add_tenant(&self, model_id: &str, tenant: &str) {
        self.string_tree_or_new(model_id).insert_text("", tenant);
        self.token_tree_or_new(model_id).insert_tokens(&[], tenant);
    }

    /// Purges `tenant` from every model's string and token tree. A removed
    /// worker's tenant count never grows again, so size-based eviction alone
    /// would retain its subtree forever.
    pub fn remove_tenant(&self, tenant: &TenantId) {
        for tree_ref in &self.string_trees {
            tree_ref.value().remove_tenant_all(tenant);
        }
        for tree_ref in &self.token_trees {
            tree_ref.value().remove_tenant_all(tenant);
        }
    }

    /// Purges `tenants` from the two trees of `model_id`, and from no other
    /// model's.
    pub fn remove_tenants_from_model(&self, model_id: &str, tenants: &[TenantId]) {
        if let Some(tree) = self.string_tree(model_id) {
            for tenant in tenants {
                tree.remove_tenant_all(tenant);
            }
        }
        if let Some(tree) = self.token_tree(model_id) {
            for tenant in tenants {
                tree.remove_tenant_all(tenant);
            }
        }
    }

    /// Evicts every model's string tree down to `max_size` characters
    /// ([`Tree::evict_tenant_by_size`]) and hands each tree to `each` once
    /// its own eviction is done. `each` runs under the read guard of the
    /// model's map entry: it must not make a tree of this store.
    pub fn evict_string_trees(&self, max_size: usize, mut each: impl FnMut(&str, &Tree)) {
        for tree_ref in &self.string_trees {
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            each(tree_ref.key(), tree);
        }
    }

    /// [`Self::evict_string_trees`] for the token trees, `max_size` in
    /// tokens ([`TokenTree::evict_tenant_by_size`]).
    pub fn evict_token_trees(&self, max_size: usize, mut each: impl FnMut(&str, &TokenTree)) {
        for tree_ref in &self.token_trees {
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            each(tree_ref.key(), tree);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tenant(name: &str) -> TenantId {
        TenantId::from(name)
    }

    #[test]
    fn a_models_tree_is_made_once_and_a_lookup_makes_none() {
        let trees = ModelTrees::new(4);
        assert!(trees.string_tree("a").is_none());
        assert!(trees.token_tree("a").is_none());
        assert_eq!(
            (trees.string_tree_count(), trees.token_tree_count()),
            (0, 0)
        );

        let string = trees.string_tree_or_new("a");
        assert!(Arc::ptr_eq(&string, &trees.string_tree_or_new("a")));
        assert!(Arc::ptr_eq(&string, &trees.string_tree("a").unwrap()));
        assert!(!Arc::ptr_eq(&string, &trees.string_tree_or_new("b")));
        // A model's string tree brings no token tree with it.
        assert_eq!(
            (trees.string_tree_count(), trees.token_tree_count()),
            (2, 0)
        );

        let token = trees.token_tree_or_new("a");
        assert!(Arc::ptr_eq(&token, &trees.token_tree_or_new("a")));
        assert!(Arc::ptr_eq(&token, &trees.token_tree("a").unwrap()));
        assert_eq!(token.page_size(), 4);
        // A page of no tokens is a page of one.
        assert_eq!(ModelTrees::new(0).token_tree_or_new("a").page_size(), 1);
    }

    #[test]
    fn every_models_token_tree_is_handed_out_and_no_string_tree() {
        let trees = ModelTrees::new(4);
        assert!(trees.token_trees().is_empty());
        trees.string_tree_or_new("c");
        let made = [trees.token_tree_or_new("a"), trees.token_tree_or_new("b")];

        let handed = trees.token_trees();
        assert_eq!(handed.len(), 2);
        for tree in &made {
            assert!(handed.iter().any(|other| Arc::ptr_eq(other, tree)));
        }
    }

    #[test]
    fn a_new_tenant_stands_at_the_root_of_its_models_string_tree() {
        let trees = ModelTrees::new(4);
        trees.add_tenant("a", "w1");

        let string = trees.string_tree("a").unwrap();
        assert_eq!(string.match_prefix_with_counts("any").tenant.as_ref(), "w1");
        assert_eq!(string.get_tenant_char_count().get("w1"), Some(&0));
        // The token tree is made, and the empty prompt is no page of it.
        let token = trees.token_tree("a").unwrap();
        assert!(token.get_tenant_token_counts().is_empty());
        assert!(trees.string_tree("b").is_none());
    }

    #[test]
    fn a_removal_takes_the_named_tenants_from_the_named_models_alone() {
        let trees = ModelTrees::new(2);
        for model in ["a", "b"] {
            for worker in ["w1", "w2", "w3", "w4"] {
                trees
                    .string_tree_or_new(model)
                    .insert_text("shared", worker);
                trees
                    .token_tree_or_new(model)
                    .insert_tokens(&[1, 2], worker);
            }
        }

        trees.remove_tenants_from_model("a", &[tenant("w1"), tenant("w2")]);
        trees.remove_tenant(&tenant("w3"));
        // A model with no tree is left without one.
        trees.remove_tenants_from_model("c", &[tenant("w4")]);
        assert!(trees.string_tree("c").is_none() && trees.token_tree("c").is_none());

        for (model, left) in [("a", vec!["w4"]), ("b", vec!["w1", "w2", "w4"])] {
            let mut chars: Vec<_> = trees
                .string_tree(model)
                .unwrap()
                .get_tenant_char_count()
                .into_iter()
                .collect();
            let mut tokens: Vec<_> = trees
                .token_tree(model)
                .unwrap()
                .get_tenant_token_counts()
                .into_iter()
                .collect();
            chars.sort();
            tokens.sort();
            let sized = |size: usize| -> Vec<(String, usize)> {
                left.iter()
                    .map(|worker| (worker.to_string(), size))
                    .collect()
            };
            assert_eq!(chars, sized(6), "model {model}");
            assert_eq!(tokens, sized(2), "model {model}");
        }
    }

    #[test]
    fn an_eviction_bounds_every_tree_and_hands_each_over_once_bounded() {
        let trees = ModelTrees::new(2);
        for model in ["a", "b"] {
            let string = trees.string_tree_or_new(model);
            string.insert_text("first prompt", "w1");
            string.insert_text("second prompt", "w2");
            let token = trees.token_tree_or_new(model);
            token.insert_tokens(&[1, 2, 3, 4], "w1");
            token.insert_tokens(&[5, 6, 7, 8], "w2");
        }
        trees.string_tree_or_new("c").insert_text("short", "w1");

        let mut strings = Vec::new();
        trees.evict_string_trees(13, |model, tree| {
            strings.push((model.to_string(), tree.total_char_size()));
        });
        strings.sort();
        // 25 characters a model before: the older prompt leaves, and a tree
        // under the bound keeps what it holds.
        let named =
            |sizes: [(&str, usize); 3]| sizes.map(|(model, size)| (model.to_string(), size));
        assert_eq!(strings, named([("a", 13), ("b", 13), ("c", 5)]));

        let mut tokens = Vec::new();
        trees.evict_token_trees(4, |model, tree| {
            tokens.push((model.to_string(), tree.total_token_size()));
        });
        tokens.sort();
        assert_eq!(tokens, [("a".to_string(), 4), ("b".to_string(), 4)]);
        // The token trees' pass left the string trees alone.
        assert_eq!(trees.string_tree("a").unwrap().total_char_size(), 13);
    }
}
