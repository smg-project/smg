//! One placement map per model: the workers each head of a prompt was sent
//! to.

use std::{
    collections::HashSet,
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use rustc_hash::FxBuildHasher;

/// One model's placement map: (boundary position, xxh3 of the token head up
/// to that boundary) → workers recently routed that exact head.
pub type PlacementMap = DashMap<(usize, u64), Vec<PlacementHolder>>;

/// One worker's most recent dispatch of a head; live while `last_touch` is
/// within the lifetime its reader states.
#[derive(Debug, Clone)]
pub struct PlacementHolder {
    /// The worker, by the name its router sends to.
    pub worker_url: String,
    /// When the head was last sent to the worker.
    pub last_touch: Instant,
}

/// Max workers remembered per (boundary, head) key; recording a fourth
/// evicts the stalest.
pub const PLACEMENT_HOLDER_CAP: usize = 3;

/// The hash of a token head, the second half of a [`PlacementMap`] key.
/// `marker` is the request's partition, where it has one, as its caller
/// holds it: whatever gives the two token ids the caller states for the
/// partition. They are read from it here, for a partitioned head alone,
/// and hashed ahead of the head: one head keys apart under two partitions.
pub fn hash_token_head(marker: Option<impl Into<[u32; 2]>>, head: &[u32]) -> u64 {
    match marker {
        // An unpartitioned head hashes exactly as before.
        None => xxhash_rust::xxh3::xxh3_64(bytemuck::cast_slice(head)),
        // A partitioned head hashes as marker ‖ head (the placement map has
        // no pages, so the bare marker suffices): a differing partition is a
        // different key, and a same-partition head keys as before.
        Some(marker) => {
            let marker: [u32; 2] = marker.into();
            let mut hasher = xxhash_rust::xxh3::Xxh3::new();
            hasher.update(bytemuck::cast_slice(&marker));
            hasher.update(bytemuck::cast_slice(head));
            hasher.digest()
        }
    }
}

/// The placement maps of one router, by model. A model's map is made at the
/// model's first write ([`Self::record`]) and is never removed, so a handle
/// on it stays the model's map, and a walk of it holds no guard of this
/// store.
///
/// The owner states a holder's lifetime at each call. It bounds the maps by
/// its sweep ([`Self::sweep`]) and takes a worker that left out of them
/// ([`Self::remove_worker`], [`Self::remove_workers_from_model`]).
#[derive(Debug, Default)]
pub struct ModelPlacements {
    /// Keyed by the operator's model ids. Every call that names a model
    /// looks it up here, a request's read of a head's holders and its
    /// write among them: the Fx hasher.
    maps: DashMap<String, Arc<PlacementMap>, FxBuildHasher>,
}

impl ModelPlacements {
    /// No map yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// `model_id`'s map, where a write named the model.
    pub fn get(&self, model_id: &str) -> Option<Arc<PlacementMap>> {
        self.maps
            .get(model_id)
            .map(|entry| Arc::clone(entry.value()))
    }

    /// Whether no model has a map.
    pub fn is_empty(&self) -> bool {
        self.maps.is_empty()
    }

    /// Hands `with` the live holders of `key` in `model_id`'s map: those
    /// touched within `ttl` of `now`. Expired holders are pruned in place
    /// (lazy expiry on read). `with` runs under the key's guard, over the
    /// holders the store's writes left, the cap at most, and the store's
    /// own lines of this read allocate nothing. `None` where the model has
    /// no map, or its map no such key.
    ///
    /// `with` asks nothing of any model's map: a read or a write from
    /// inside `with` can wait for good on a guard a caller holds, its own
    /// or another's.
    pub fn live_holders<R>(
        &self,
        model_id: &str,
        key: (usize, u64),
        now: Instant,
        ttl: Duration,
        with: impl FnOnce(&[PlacementHolder]) -> R,
    ) -> Option<R> {
        let model = self.get(model_id)?;
        let mut holders = model.get_mut(&key)?;
        holders.retain(|h| now.duration_since(h.last_touch) <= ttl);
        Some(with(&holders))
    }

    /// Record/touch `worker_url` at every boundary of `boundaries` for the
    /// request of `tokens`; above the holder cap the stalest holder is
    /// evicted. `marker` is the request's partition ([`hash_token_head`]).
    /// The model's map is made where it has none, ahead of the first
    /// boundary: a write of no boundary makes it too.
    ///
    /// # Panics
    ///
    /// Where a boundary lies past the request's end: the caller hands the
    /// boundaries the request reaches. Those ahead of it in `boundaries`
    /// are written by then.
    #[expect(
        clippy::too_many_arguments,
        reason = "what the caller holds at the call, no state of the store"
    )]
    pub fn record(
        &self,
        model_id: &str,
        marker: Option<impl Copy + Into<[u32; 2]>>,
        tokens: &[u32],
        boundaries: &[usize],
        worker_url: &str,
        now: Instant,
        ttl: Duration,
    ) {
        let model = if let Some(entry) = self.maps.get(model_id) {
            Arc::clone(entry.value())
        } else {
            Arc::clone(self.maps.entry(model_id.to_string()).or_default().value())
        };
        for &boundary in boundaries {
            let key = (boundary, hash_token_head(marker, &tokens[..boundary]));
            let mut holders = model.entry(key).or_default();
            holders.retain(|h| now.duration_since(h.last_touch) <= ttl);
            if let Some(holder) = holders.iter_mut().find(|h| h.worker_url == worker_url) {
                holder.last_touch = now;
                continue;
            }
            if holders.len() >= PLACEMENT_HOLDER_CAP {
                if let Some(stalest) = holders
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, h)| h.last_touch)
                    .map(|(i, _)| i)
                {
                    holders.swap_remove(stalest);
                }
            }
            holders.push(PlacementHolder {
                worker_url: worker_url.to_string(),
                last_touch: now,
            });
        }
    }

    /// Drops expired holders and empty keys, and hands `each` every model
    /// with the keys its map keeps. Returns the keys kept across models.
    pub fn sweep(&self, ttl: Duration, now: Instant, mut each: impl FnMut(&str, usize)) -> usize {
        let models: Vec<(String, Arc<PlacementMap>)> = self
            .maps
            .iter()
            .map(|model| (model.key().clone(), Arc::clone(model.value())))
            .collect();
        let mut total = 0usize;
        for (model_id, placements) in models {
            placements.retain(|_, holders| {
                holders.retain(|h| now.duration_since(h.last_touch) <= ttl);
                !holders.is_empty()
            });
            let entries = placements.len();
            total += entries;
            each(&model_id, entries);
        }
        total
    }

    /// Takes `worker_url` out of every model's map.
    pub fn remove_worker(&self, worker_url: &str) {
        let placement_maps: Vec<Arc<PlacementMap>> = self
            .maps
            .iter()
            .map(|model| Arc::clone(model.value()))
            .collect();
        for placements in placement_maps {
            placements.retain(|_, holders| {
                holders.retain(|h| h.worker_url != worker_url);
                !holders.is_empty()
            });
        }
    }

    /// Takes `worker_urls` out of `model_id`'s map in one scan of it, and
    /// out of no other model's.
    pub fn remove_workers_from_model(&self, model_id: &str, worker_urls: &HashSet<String>) {
        if let Some(placements) = self.get(model_id) {
            placements.retain(|_, holders| {
                holders.retain(|holder| !worker_urls.contains(&holder.worker_url));
                !holders.is_empty()
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TTL: Duration = Duration::from_secs(180);
    /// A request with no partition.
    const NONE: Option<[u32; 2]> = None;

    /// The live holders of the 16-token head of `tokens` under "m", by name.
    fn live(store: &ModelPlacements, tokens: &[u32], now: Instant) -> Option<Vec<String>> {
        let key = (16, hash_token_head(NONE, &tokens[..16]));
        store.live_holders("m", key, now, TTL, |holders| {
            holders.iter().map(|h| h.worker_url.clone()).collect()
        })
    }

    #[test]
    fn a_models_map_is_made_at_its_first_record_and_a_read_makes_none() {
        let store = ModelPlacements::new();
        let tokens: Vec<u32> = (0..32).collect();
        let now = Instant::now();
        assert_eq!(live(&store, &tokens, now), None);
        assert!(store.is_empty() && store.get("m").is_none());

        store.record("m", NONE, &tokens, &[16, 32], "w1", now, TTL);
        let map = store.get("m").unwrap();
        // One key a boundary, and no key of a head that was not recorded.
        assert_eq!(map.len(), 2);
        assert_eq!(live(&store, &tokens, now), Some(vec!["w1".to_string()]));
        let other: Vec<u32> = (100..132).collect();
        assert_eq!(live(&store, &other, now), None);
        assert!(store.get("n").is_none());
    }

    #[test]
    fn a_head_keeps_three_holders_and_the_stalest_leaves_for_a_fourth() {
        let store = ModelPlacements::new();
        let tokens: Vec<u32> = (0..16).collect();
        let start = Instant::now();
        let at = |secs: u64| start + Duration::from_secs(secs);
        for (secs, worker) in [(0, "w1"), (1, "w2"), (2, "w3")] {
            store.record("m", NONE, &tokens, &[16], worker, at(secs), TTL);
        }
        // A worker recorded again is touched, and held once.
        store.record("m", NONE, &tokens, &[16], "w1", at(3), TTL);
        let mut held = live(&store, &tokens, at(3)).unwrap();
        held.sort();
        assert_eq!(held, ["w1", "w2", "w3"]);

        store.record("m", NONE, &tokens, &[16], "w4", at(4), TTL);
        let mut held = live(&store, &tokens, at(4)).unwrap();
        held.sort();
        assert_eq!(held.len(), PLACEMENT_HOLDER_CAP);
        assert_eq!(held, ["w1", "w3", "w4"]);
    }

    #[test]
    fn a_holder_past_its_lifetime_leaves_at_a_read_and_its_key_at_a_sweep() {
        let store = ModelPlacements::new();
        let tokens: Vec<u32> = (0..16).collect();
        let start = Instant::now();
        store.record("m", NONE, &tokens, &[16], "w1", start, TTL);
        assert_eq!(live(&store, &tokens, start + TTL).unwrap(), ["w1"]);

        let late = start + TTL + Duration::from_secs(1);
        // The read prunes the holder and keeps the key.
        assert_eq!(live(&store, &tokens, late), Some(Vec::new()));
        assert_eq!(store.get("m").unwrap().len(), 1);

        let mut counts = Vec::new();
        let kept = store.sweep(TTL, late, |model, keys| {
            counts.push((model.to_string(), keys));
        });
        assert_eq!((kept, counts), (0, vec![("m".to_string(), 0)]));
        assert!(store.get("m").unwrap().is_empty());
    }

    #[test]
    fn a_removal_takes_a_worker_from_every_model_or_from_the_named_one() {
        let store = ModelPlacements::new();
        let tokens: Vec<u32> = (0..16).collect();
        let now = Instant::now();
        let key = (16, hash_token_head(NONE, &tokens));
        for model in ["m", "n"] {
            for worker in ["w1", "w2", "w3"] {
                store.record(model, NONE, &tokens, &[16], worker, now, TTL);
            }
        }
        let held = |model: &str| {
            let read = |holders: &[PlacementHolder]| {
                let mut names: Vec<String> = holders.iter().map(|h| h.worker_url.clone()).collect();
                names.sort();
                names
            };
            store.live_holders(model, key, now, TTL, read)
        };

        store.remove_worker("w1");
        store.remove_workers_from_model("m", &HashSet::from(["w2".to_string()]));
        // A model with no map is left without one.
        store.remove_workers_from_model("o", &HashSet::from(["w3".to_string()]));
        assert!(store.get("o").is_none());
        assert_eq!(held("m").unwrap(), ["w3"]);
        assert_eq!(held("n").unwrap(), ["w2", "w3"]);

        // A key that loses its last holder leaves with it.
        store.remove_worker("w3");
        assert_eq!(held("m"), None);
        assert!(store.get("m").unwrap().is_empty());
    }

    #[test]
    fn a_head_hashes_under_its_partitions_marker() {
        let head: Vec<u32> = (0..16).collect();
        let plain = hash_token_head(NONE, &head);
        assert_eq!(
            plain,
            xxhash_rust::xxh3::xxh3_64(bytemuck::cast_slice(&head))
        );
        let (one, other) = ([0x8000_0001, 0x8000_0002], [0x8000_0001, 0x8000_0003]);
        assert_eq!(
            hash_token_head(Some(one), &head),
            hash_token_head(Some(one), &head)
        );
        assert_ne!(hash_token_head(Some(one), &head), plain);
        assert_ne!(
            hash_token_head(Some(one), &head),
            hash_token_head(Some(other), &head)
        );
    }
}
