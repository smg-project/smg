//! A list's table: a phase that reads no affinity, whose list keeps none, how
//! long one stands, a number of the index that changes hands, and a model's
//! first block and its last.

use super::*;

/// A list's table is read once, then again by a decision a second later:
/// the decisions between read what stands and ask the host nothing.
#[test]
fn a_lists_table_stands_for_a_second() {
    let (affinity, host, _) = hosted(true);
    // The index holds a block, another worker's of another prompt: a host
    // hands no index that holds none, and is then asked for its tree too.
    let index = host.index.as_ref().expect("the index");
    store(index, 3, &hashes_of(&prompt(9_000, 1)));
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    let reads = || host.asked.load(Ordering::Relaxed);
    assert_eq!(reads(), 0);
    bed.holders(&asked, NOW);
    assert_eq!(reads(), 1);
    for step in 1..100 {
        bed.holders(&asked, NOW + step * (SECOND / 100) - 1);
    }
    assert_eq!(reads(), 1);
    bed.holders(&asked, NOW + SECOND);
    assert_eq!(reads(), 2);
    // Blocks stored behind a read are named by the next one.
    store(index, 2, &hashes_of(&asked));
    assert!(bed.holders(&asked, NOW + SECOND + 1).is_empty());
    assert_eq!(bed.holders(&asked, NOW + 2 * SECOND), [(2, 4, 0)]);
}

/// A number of the index that one worker handed back and the next worker
/// took names the new worker to the next decision: the list's table, which
/// kept the number for the first, is read again before its second is up.
#[test]
fn a_number_that_changed_hands_names_its_new_worker() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("an index");
    let bed = Bed::new(affinity, preset("cache_aware"), 4);
    let asked = prompt(1_000, 4);
    // Worker 0 holds another prompt under its number when the table is read.
    let (number, map) = stored(&index, 0, &hashes_of(&prompt(5_000, 2)));
    assert!(bed.holders(&asked, NOW).is_empty());
    // Its stream ends and worker 1's opens: worker 1 takes the number and
    // alone holds the asked prompt.
    index.remove_worker(number, map);
    store(&index, 1, &hashes_of(&asked));
    assert_eq!(index.worker_id(&address(1)), Some(number));
    assert_eq!(bed.holders(&asked, NOW), [(1, 4, 0)]);
}

/// A number that changes hands under a decision's own lookup names nobody
/// to it: the table read the number for its former worker, and the lookup
/// finds the blocks of its new one. The next decision reads the table again
/// and names the new worker. A side's column holds nothing of the request
/// under such a lookup, and the next column holds the new worker's blocks.
#[test]
fn a_number_handed_on_under_a_lookup_names_nobody() {
    /// A chain index whose first lookup is met by a handoff: the stream of
    /// worker 0 ends and worker 1 stores the asked prompt under its number.
    struct HandedOn {
        index: ChainIndex,
        handoff: Mutex<Option<(u32, ChainBlockMap, Vec<u64>)>>,
    }

    impl CacheIndex for HandedOn {
        fn worker_number(&self, address: &str) -> Option<u32> {
            self.index.worker_id(address)
        }

        fn numbers_reused(&self) -> u64 {
            self.index.ids_reused()
        }

        fn worker_blocks(&self, number: u32) -> usize {
            self.index.worker_block_count(number)
        }

        fn deepest_into(
            &self,
            block_hashes: &[u64],
            block_size: u32,
            within: &[u64],
            cap: usize,
            below: &mut dyn FnMut(u32) -> u32,
            kept: &mut Vec<(u32, u32)>,
        ) {
            if let Some((number, map, hashes)) = self.handoff.lock().take() {
                self.index.remove_worker(number, map);
                store(&self.index, 1, &hashes);
            }
            CacheIndex::deepest_into(
                &self.index,
                block_hashes,
                block_size,
                within,
                cap,
                below,
                kept,
            );
        }
    }

    struct Lends(Arc<HandedOn>);

    impl CacheIndexes for Lends {
        fn index(&self, _: &str) -> Option<Arc<dyn CacheIndex>> {
            Some(Arc::clone(&self.0) as Arc<dyn CacheIndex>)
        }

        fn block_size(&self, _: &str) -> Option<u32> {
            Some(BLOCK)
        }
    }

    let asked = prompt(1_000, 4);
    let handed_on = |spec: AppliedSpec| {
        let index = ChainIndex::new();
        // Worker 0 holds another prompt under its number when the table is
        // read.
        let (number, map) = stored(&index, 0, &hashes_of(&prompt(5_000, 2)));
        let handed = Arc::new(HandedOn {
            index,
            handoff: Mutex::new(Some((number, map, hashes_of(&asked)))),
        });
        let bed = Bed::new(Affinity::default(), spec, 4);
        bed.affinity
            .set_indexes(Arc::new(Lends(Arc::clone(&handed))) as Arc<dyn CacheIndexes>);
        (bed, handed, number)
    };

    let (bed, handed, number) = handed_on(preset("cache_aware"));
    assert!(bed.holders(&asked, NOW).is_empty());
    assert_eq!(handed.index.worker_id(&address(1)), Some(number));
    assert_eq!(bed.holders(&asked, NOW), [(1, 4, 0)]);

    // A side's column is filled by a lookup of its own.
    let (bed, handed, number) = handed_on(paired());
    let facts = request(&asked, 7, NOW);
    assert_eq!(held(&bed.column(&facts, Phase::Prefill).0), [(0, 0); 4]);
    assert_eq!(handed.index.worker_id(&address(1)), Some(number));
    assert_eq!(
        held(&bed.column(&facts, Phase::Prefill).0),
        [(0, 0), (4, 0), (0, 0), (0, 0)]
    );
}

/// A model's first block names a single request its young worker. The
/// host lends its index while it holds no block, as a gateway lends its
/// own from a model's first enrolment, and a worker admitted beside older
/// ones is named no warming row: the warm-up catches a worker up to what
/// the index of cache events holds. Once another worker's events store a
/// block, the next decision finds the index's word moved, reads the table
/// again inside its second and names the young worker by its age. The
/// classic policy calls its warm-up slice on its road of cache events
/// alone, which it takes once the index holds a block
/// (`model_gateway/src/policies/cache_aware.rs:1802-1808`).
#[test]
fn a_models_first_block_names_a_single_request_its_young_worker() {
    let (affinity, index, _tree) = lending_whenever();
    let mut bed = Bed::new(affinity, warming_by(1), 7);
    // Seven workers are older than the window of one second, and the
    // eighth is admitted now.
    thread::sleep(Duration::from_millis(1_100));
    bed.workers.push(ready_worker(&address(7), MODEL));
    bed.ids.push(WorkerId(8));
    let asked = prompt(50_000, 2);
    let warming = || {
        let named = bed.decide(&request(&asked, 7, NOW), None);
        named.map_or_else(Vec::new, |rows| rows.warming)
    };
    assert!(warming().is_empty(), "under an index that holds no block");
    // The events of a worker outside the list store a prompt: the model's
    // index holds a block and no worker of the list holds one. The next
    // decision, at the same instant, names the young worker.
    store(&index, 9, &hashes_of(&prompt(1_000, 8)));
    assert_eq!(warming(), [7], "from the model's first block");
}

/// A host that hands no index while its index holds no block, and whose
/// index keeps the word's default, is decided as before: the engine learns
/// of a model's first block, and of its last block's leaving, with the
/// list's next table, up to a second later.
#[test]
fn a_host_that_hands_no_empty_index_learns_of_a_block_with_the_next_table() {
    /// A chain index that keeps the default word of holding a block.
    struct Quiet(Arc<ChainIndex>);

    impl CacheIndex for Quiet {
        fn worker_number(&self, address: &str) -> Option<u32> {
            self.0.worker_id(address)
        }

        fn numbers_reused(&self) -> u64 {
            self.0.ids_reused()
        }

        fn worker_blocks(&self, number: u32) -> usize {
            self.0.worker_block_count(number)
        }

        fn deepest_into(
            &self,
            block_hashes: &[u64],
            block_size: u32,
            within: &[u64],
            cap: usize,
            below: &mut dyn FnMut(u32) -> u32,
            kept: &mut Vec<(u32, u32)>,
        ) {
            CacheIndex::deepest_into(&*self.0, block_hashes, block_size, within, cap, below, kept);
        }
    }

    /// Hands the index while it holds a block, and its token tree.
    struct Lends {
        index: Arc<ChainIndex>,
        tree: Arc<TokenTree>,
    }

    impl CacheIndexes for Lends {
        fn index(&self, _: &str) -> Option<Arc<dyn CacheIndex>> {
            let held = !self.index.is_empty();
            held.then(|| Arc::new(Quiet(Arc::clone(&self.index))) as Arc<dyn CacheIndex>)
        }

        fn block_size(&self, _: &str) -> Option<u32> {
            Some(BLOCK)
        }

        fn token_tree(&self, _: &str, _: Phase) -> Option<Arc<TokenTree>> {
            Some(Arc::clone(&self.tree))
        }
    }

    let index = Arc::new(ChainIndex::new());
    let tree = Arc::new(TokenTree::with_config(BLOCK as usize, Default::default()));
    let bed = Bed::new(Affinity::default(), preset("cache_aware"), 4);
    let lends = Lends {
        index: Arc::clone(&index),
        tree,
    };
    bed.affinity
        .set_indexes(Arc::new(lends) as Arc<dyn CacheIndexes>);
    let asked = tokens(1_000, 4);
    let hashes = hashed(&asked);
    // What the request is named at `now`, decided by a shed.
    let named_at = |now: u64| {
        let key = TokenKey {
            tokens: Cow::Borrowed(&asked[..]),
            marker: 0,
        };
        let named = bed.decide_keyed(&request(&hashes, 7, now), Some(key), None);
        named.expect("rows named").holders
    };
    // No index is handed: the tree decides, and is written for worker 1.
    assert!(bed.keyed(&asked, 0, Some(1)).is_empty());
    assert_eq!(named_at(NOW), [(1, 0, 4)]);
    // The model's first block. The table that stands was handed no index
    // and asks nobody: the tree decides until its second is up.
    let (number, mut map) = stored(&index, 2, &hashes_of(&hashes));
    assert_eq!(named_at(NOW + SECOND - 1), [(1, 0, 4)]);
    assert_eq!(named_at(NOW + SECOND), [(2, 4, 0)]);
    // The last block leaves. The index's word is the default, so the table
    // stands for its second and its lookup names nobody.
    index.apply_cleared(number, &mut map);
    assert_eq!(named_at(NOW + 2 * SECOND - 1), []);
    assert_eq!(named_at(NOW + 2 * SECOND), [(1, 0, 4)]);
}

/// Under a document where one model's profile reads affinity in the single
/// phase and every other place reads none, each request is decided as under
/// a document of its own. A request of a phase that reads none is named no
/// row, and its list keeps no table. Another model's request of the
/// reader's phase is named none behind its list's table. The reader is
/// named its rows.
#[test]
fn a_profile_that_reads_affinity_is_named_its_rows_beside_one_that_reads_none() {
    let mut document = PolicySpec::preset("least_load").expect("the preset");
    let reader = PolicySpec::preset("cache_aware").expect("the preset");
    let mut binding = reader.bindings[0].clone();
    binding.matcher.model = MODEL.into();
    binding.matcher.phase = vec![Phase::Single];
    document.bindings.insert(0, binding);
    document.profiles.extend(reader.profiles);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = spec.engine();
    assert!(engine.needs_for(MODEL, Phase::Single).affinity);
    assert!(!engine.needs_for("model-b", Phase::Single).affinity);
    assert!(!engine.needs_for_phase(Phase::Decode).affinity);
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 4);
    store(&index, 2, &hashes_of(&asked));
    let bed = Bed::new(affinity, spec, 4);
    // The phase asked is the request's own: no binding of this one reads
    // affinity, so no table is read.
    let decode = RequestFacts {
        phase: Phase::Decode,
        ..request(&asked, 7, NOW)
    };
    assert_eq!(bed.decide(&decode, Some(1)), None);
    assert!(bed.pool.table.load().is_none(), "no table is read");
    // In the reader's phase the other model's profile reads none: its
    // list's table is what says so.
    let other = RequestFacts {
        model: "model-b",
        ..request(&asked, 7, NOW)
    };
    assert_eq!(bed.decide(&other, Some(1)), None);
    assert!(bed.pool.table.load().is_some(), "the list's table says so");
    assert_eq!(bed.holders(&asked, NOW), [(2, 4, 0)]);
}
