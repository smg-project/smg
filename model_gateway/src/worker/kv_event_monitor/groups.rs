//! The rank-zero group view shares the subscription's admission and recovery.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use dashmap::DashMap;
use kv_index::group_cache::{GroupCache, GroupEvent, GroupRequest};
use parking_lot::RwLock;
use smg_grpc_client::common_proto::{kv_group_event::Operation, KvEventBatch};

pub(crate) type GroupViewHandle = Arc<RwLock<GroupView>>;

#[derive(Default)]
pub(crate) struct ModelGroups {
    pub(super) workers: DashMap<String, GroupViewHandle>,
    pub(super) enabled: Arc<AtomicBool>,
}

impl ModelGroups {
    pub(crate) fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    pub(crate) fn reusable_tokens(
        &self,
        url: &str,
        request: &GroupRequest,
    ) -> Option<Option<usize>> {
        let view = self.workers.get(url)?;
        let view = view.read();
        view.enabled.then(|| {
            if view.valid {
                view.cache.reusable_tokens(request)
            } else {
                None
            }
        })
    }

    pub(super) fn remove(&self, url: &str) {
        self.workers.remove(url);
        self.enabled.store(
            self.workers.iter().any(|view| view.read().enabled),
            Ordering::Release,
        );
    }
}

pub(crate) struct GroupView {
    pub(super) cache: GroupCache,
    pub(super) enabled: bool,
    pub(super) valid: bool,
    pub(super) malformed: bool,
    pub(super) model_enabled: Option<Arc<AtomicBool>>,
}

impl Default for GroupView {
    fn default() -> Self {
        Self {
            cache: GroupCache::default(),
            enabled: false,
            valid: true,
            malformed: false,
            model_enabled: None,
        }
    }
}

impl GroupView {
    pub(super) fn clear_evidence(&mut self) {
        self.cache.clear_evidence();
        self.valid = false;
    }

    pub(super) fn restart(&mut self) {
        self.cache.invalidate();
        self.valid = true;
        self.malformed = false;
    }

    pub(super) fn apply(&mut self, batch: &KvEventBatch) {
        if batch.dp_rank.unwrap_or(0) != 0 {
            return;
        }
        if batch.group_events_enabled {
            self.enabled = true;
            if let Some(enabled) = &self.model_enabled {
                enabled.store(true, Ordering::Release);
            }
        }
        if !self.enabled {
            return;
        }
        let events: Vec<_> = batch
            .group_events
            .iter()
            .map(|event| match Operation::try_from(event.operation) {
                Ok(Operation::Store) => GroupEvent::Store {
                    group_id: event.group_id,
                    kind: event.kind.clone(),
                    window: event.sliding_window as usize,
                    keys: event.block_hashes.clone(),
                    parent: (!event.parent_block_hash.is_empty())
                        .then(|| event.parent_block_hash.clone()),
                    tokens: event.token_ids.clone(),
                    block_size: event.block_size as usize,
                    tokens_matchable: event.tokens_matchable,
                    skipped_tokens: event
                        .ancestry_provided
                        .then(|| event.skipped_token_ids.clone()),
                    skipped_token_parent_hash: (!event.skipped_token_parent_hash.is_empty())
                        .then(|| event.skipped_token_parent_hash.clone()),
                    copy_counts: event.cache_copy_counts.clone(),
                },
                Ok(Operation::Remove) => GroupEvent::Remove {
                    group_id: event.group_id,
                    keys: event.block_hashes.clone(),
                    copy_counts: event.cache_copy_counts.clone(),
                },
                Ok(Operation::Clear) => GroupEvent::Clear,
                _ => GroupEvent::Invalid,
            })
            .collect();
        let clears = events
            .iter()
            .any(|event| matches!(event, GroupEvent::Clear));
        if self.cache.apply_events(&events).is_err() {
            self.malformed = true;
            self.clear_evidence();
        } else if clears {
            self.malformed = false;
            self.valid = true;
        }
    }
}
