//! A request decided inside the placement map its host lends under a hash
//! index: the requests the map is asked for, a head whose holders the pass
//! refuses and one that ends inside a block, a pick that states no share, a
//! holder's lifetime, the classic policy's key, the target a placement is
//! written for, a request that names no entry, and a model its workers share.

use super::*;

/// A side of a pair is asked for no placement map, though its host would
/// lend one at every phase: a side fills its column and decides inside no
/// map, so its table keeps the record of sent prefixes. A pick written
/// behind a side's column is what the side's next column holds, on the
/// prefill side and on the decode side.
#[test]
fn a_side_of_a_pair_is_asked_for_no_placement_map() {
    use crate::document::spec::{IndexKind, IndexSpec};
    for phase in [Phase::Prefill, Phase::Decode] {
        let (affinity, host, _) = lending_everywhere();
        let mut document =
            PolicySpec::pair_presets("cache_aware", "cache_aware").expect("the pair presets");
        document.generation = 1;
        document.view.boundaries = vec![2 * BLOCK, 4 * BLOCK];
        let hash = IndexSpec {
            kind: IndexKind::Hash,
            ..IndexSpec::default()
        };
        document.index.insert("*".into(), hash);
        let spec = AppliedSpec::validate(&document).expect("the pair document validates");
        let side = Bed::new(affinity, spec, 4);
        let asked = prompt(1_000, 5);
        let facts = request(&asked, 7, NOW);
        let (column_of, read) = side.column(&facts, phase);
        let read = read.expect("the record speaks for the side");
        record_by_column(&side.pool, phase, &facts, read, &column_of, side.ids[1]);
        let (column_of, _) = side.column(&facts, phase);
        let held = held(&column_of);
        assert_eq!(held, [(0, 0), (0, 4), (0, 0), (0, 0)], "{phase:?}");
        let asks = host.maps.load(Ordering::Relaxed);
        assert_eq!(asks, 0, "{phase:?}: no map is asked for");
    }
}

/// A request asked alone at a side's phase is asked for no placement map
/// either, though its host would lend one at every phase. At the prefill
/// and the decode phase the list's table is the one a pair's side fills
/// its column from, whichever binding the request asked alone resolves
/// to, so the map is asked for at the single phase alone: at the encode
/// phase no more than at those two. Under a plan of one profile the
/// request keeps the record of sent prefixes at each of the three: its
/// next turn is named the worker it went to, the lent map stays empty,
/// and the host is asked for no map.
#[test]
fn a_request_alone_at_a_sides_phase_is_asked_for_no_placement_map() {
    for phase in [Phase::Prefill, Phase::Decode, Phase::Encode] {
        let (affinity, host, placed) = lending_everywhere();
        let bed = Bed::new(affinity, hashed_at(&[BLOCK], 60), 4);
        let (asked, now) = (tokens(1_000, 2), Instant::now());
        let named = bed.placed_in(phase, &asked, Some(2), now);
        assert!(named.holders.is_empty(), "{phase:?}: nobody held the head");
        let named = bed.placed_in(phase, &asked, Some(2), now);
        assert_eq!(named.holders, [(2, 0, 1)], "{phase:?}: by the record");
        assert!(placed.is_empty(), "{phase:?}: the lent map holds nothing");
        let asks = host.maps.load(Ordering::Relaxed);
        assert_eq!(asks, 0, "{phase:?}: no map is asked for");
    }
}

/// Under a hash index a request whose host names its entry is decided
/// inside the placement map the host lends, as the classic policy decides
/// under that index (`select_worker_hash`,
/// model_gateway/src/policies/cache_aware.rs). A prompt nobody was sent is
/// named nobody, and the worker picked is written at every boundary the
/// prompt reaches (`record_placement`). The next request of the prompt is
/// named that worker at its deepest head, holding the head's blocks, and
/// the rows carry the head's size and the prompt's in tokens. A prompt
/// that shares the first head alone passes the deeper one, which has no
/// holder (`live_holder_candidates`). A pick of another worker is written
/// beside the holder. A prompt under the first boundary reaches none: it
/// is named nobody and its pick writes nothing. A prompt of a boundary's
/// own length reaches it (`select_worker_hash` takes the boundaries at
/// most the prompt's length). The host's index of cache events holds a
/// block, and is not asked for. The map is asked for once a table.
#[test]
fn a_request_is_decided_inside_the_placement_map_its_host_lends() {
    let (affinity, host, placed) = lending_placed(true);
    let bed = Bed::new(affinity, hashed_at(&[BLOCK, 3 * BLOCK], 60), 4);
    let (asked, now) = (tokens(1_000, 4), Instant::now());
    let nobody: &[(u32, u32, u32)] = &[];
    let first = bed.placed_at(&asked, 0, Some(1), now);
    assert_eq!(
        (first.holders.as_slice(), first.tree),
        (nobody, Some(TreeMatch::head(0, 64)))
    );
    assert_eq!(placed.len(), 2);
    let again = bed.placed_at(&asked, 0, None, now);
    assert_eq!(again.holders, [(1, 0, 3)]);
    assert_eq!(again.tree, Some(TreeMatch::head(48, 64)));
    // The same first head, and another second one.
    let mut other = asked.clone();
    other[20] += 9_000;
    let shared = bed.placed_at(&other, 0, None, now);
    assert_eq!(shared.holders, [(1, 0, 1)]);
    assert_eq!(shared.tree, Some(TreeMatch::head(16, 64)));
    assert_eq!(placed.len(), 2, "a shed writes nothing");
    bed.placed_at(&asked, 0, Some(2), now);
    let both = bed.placed_at(&asked, 0, None, now);
    assert_eq!(both.holders, [(1, 0, 3), (2, 0, 3)]);
    let short = bed.placed_at(&asked[..8], 0, Some(3), now);
    assert_eq!((short.holders.as_slice(), short.tree), (nobody, None));
    assert_eq!(placed.len(), 2);
    // A prompt of a boundary's own length reaches that boundary.
    let whole = bed.placed_at(&asked[..16], 0, None, now);
    assert_eq!(whole.holders, [(1, 0, 1), (2, 0, 1)]);
    assert_eq!(whole.tree, Some(TreeMatch::head(16, 16)));
    assert_eq!(host.asked.load(Ordering::Relaxed), 0);
    // Every decision above stands at one instant, so on one table.
    assert_eq!(host.maps.load(Ordering::Relaxed), 1, "one ask a table");
}

/// A head's holders are held to the test the decision's own pass asks of a
/// row, the deepest head first, as the classic policy keeps the holders
/// that are among its available workers (`select_worker_hash`,
/// model_gateway/src/policies/cache_aware.rs). A deepest head whose one
/// holder the pass refuses is passed for the next. That head's holders are
/// named whole where the pass keeps one of them. Where the pass keeps no
/// holder of any head the request is named nobody, and its pick is written
/// at every boundary it reaches all the same.
#[test]
fn a_head_whose_holders_the_pass_refuses_is_passed_for_the_next() {
    let (affinity, _, placed) = lending_placed(false);
    let bed = Bed::new(affinity, hashed_at(&[BLOCK, 3 * BLOCK], 60), 4);
    let (asked, now) = (tokens(1_000, 4), Instant::now());
    // The same first head, and another second one.
    let mut other = asked.clone();
    other[20] += 9_000;
    bed.placed_at(&asked, 0, Some(1), now);
    bed.placed_at(&other, 0, Some(2), now);
    // The pass keeps every row but the first worker's. The head of three
    // blocks names that row alone, and the head of one block both rows.
    let but_one = |holders: &[AffinityHolder]| Some(holders.iter().any(|holder| holder.row != 1));
    let named = bed.placed_kept(&asked, 0, None, now, but_one);
    assert_eq!(named.holders, [(1, 0, 1), (2, 0, 1)]);
    assert_eq!(named.tree, Some(TreeMatch::head(16, 64)));

    let nobody: &[(u32, u32, u32)] = &[];
    let named = bed.placed_kept(&asked, 0, Some(3), now, |_| Some(false));
    assert_eq!(
        (named.holders.as_slice(), named.tree),
        (nobody, Some(TreeMatch::head(0, 64)))
    );
    let written = bed.placed_at(&asked, 0, None, now);
    assert_eq!(written.holders, [(1, 0, 3), (3, 0, 3)]);
    assert_eq!(placed.len(), 3);
}

/// A head at a boundary that is no whole number of blocks is stated by its
/// tokens and priced by them. The rows of a head of 20 tokens, at a block
/// of 16, carry 20 tokens of the request's 64, marked as priced by its
/// whole count ([`TreeMatch::counted`]), and each holder is named one
/// whole block. The classic policy prices a holder by its boundary's
/// tokens (`select_worker_hash` hands `select_final_from_affinity` the
/// boundary, model_gateway/src/policies/cache_aware.rs).
#[test]
fn a_head_that_is_no_whole_number_of_blocks_is_stated_by_its_tokens() {
    let (affinity, _, _) = lending_placed(false);
    let bed = Bed::new(affinity, hashed_at(&[20], 60), 4);
    let (asked, now) = (tokens(1_000, 4), Instant::now());
    bed.placed_at(&asked, 0, Some(1), now);
    let named = bed.placed_at(&asked, 0, None, now);
    assert_eq!(named.holders, [(1, 0, 1)]);
    assert_eq!(named.tree, Some(TreeMatch::head(20, 64)));
}

/// A single request decided inside a lent placement map states no share,
/// whoever its pick names: a pick where nobody was named, a pick of the
/// head's holder and a pick of another worker state nothing alike
/// ([`Overlap::NONE`]). So a host's
/// match-ratio series takes nothing from such a request. The classic
/// policy's hash decision observes none: `select_worker_hash` calls
/// `Metrics::record_cache_aware_match_ratio` nowhere
/// (model_gateway/src/policies/cache_aware.rs).
#[test]
fn a_pick_made_inside_a_lent_placement_map_states_no_share() {
    let (affinity, _, placed) = lending_placed(false);
    let bed = Bed::new(affinity, hashed_at(&[BLOCK], 60), 4);
    let (asked, now) = (tokens(1_000, 2), Instant::now());
    assert_eq!(bed.stated_in_map(&asked, 1, now), Overlap::NONE, "a miss");
    assert_eq!(placed.len(), 1, "the miss is written");
    assert_eq!(bed.stated_in_map(&asked, 1, now), Overlap::NONE, "a hit");
    assert_eq!(bed.stated_in_map(&asked, 2, now), Overlap::NONE, "a spill");
}

/// A holder of the placement map lives by the decision's own instant. A
/// request decided at the holder's whole lifetime is named it, and one a
/// second later is named nobody. The engine's road hands the lookup that
/// instant and reads no clock for it. The classic policy reads the clock
/// once for its map's read and write (`select_worker_hash`,
/// model_gateway/src/policies/cache_aware.rs), and the map keeps a holder
/// no older than the lifetime (crates/kv_index/src/placements.rs,
/// `live_holders_in`).
#[test]
fn a_placement_holder_lives_by_the_decisions_instant() {
    let (affinity, _, _) = lending_placed(false);
    let bed = Bed::new(affinity, hashed_at(&[BLOCK], 60), 4);
    let (asked, sent) = (tokens(1_000, 2), Instant::now());
    bed.placed_at(&asked, 0, Some(1), sent);
    let at = |seconds: u64| bed.placed_at(&asked, 0, None, sent + Duration::from_secs(seconds));
    assert_eq!(at(60).holders, [(1, 0, 1)]);
    assert!(at(61).holders.is_empty());
}

/// The key the engine's road reads and writes is the key the classic
/// policy's road writes, boundary by boundary: the boundary and the hash of
/// the prompt's head, with a partition's two marker ids hashed ahead of the
/// head (`hash_token_head`, model_gateway/src/policies/cache_aware.rs, by
/// `kv_index::hash_token_head`). So one prompt keys apart under a partition
/// and under none, and neither is named the other's holder. The marker's
/// ids are counted in neither size a head states. A marker that is
/// neither two ids nor none is no entry of the map: such a request is
/// named nobody and writes nothing.
#[test]
fn a_placement_is_keyed_as_the_classic_policy_keys_it() {
    let (affinity, _, placed) = lending_placed(false);
    let bed = Bed::new(affinity, hashed_at(&[BLOCK, 3 * BLOCK], 60), 4);
    let (ids, now) = (tokens(1_000, 4), Instant::now());
    let marker = [7_000_001, 7_000_002];
    let keyed: Vec<u32> = marker.iter().chain(&ids).copied().collect();
    bed.placed_at(&ids, 0, Some(1), now);
    bed.placed_at(&keyed, 2, Some(2), now);
    let holders_at = |marker: Option<[u32; 2]>, end: usize| {
        let key = (end, hash_token_head(marker, &ids[..end]));
        let holders = placed.get(&key).expect("a placement at the boundary");
        let addresses = holders.iter().map(|holder| holder.worker_url.clone());
        addresses.collect::<Vec<_>>()
    };
    let address = |row: usize| vec![bed.workers[row].url().to_string()];
    for end in [16, 48] {
        assert_eq!(holders_at(None, end), address(1));
        assert_eq!(holders_at(Some(marker), end), address(2));
    }
    assert_eq!(placed.len(), 4);
    assert_eq!(bed.placed_at(&ids, 0, None, now).holders, [(1, 0, 3)]);
    // The marker's two ids are counted in neither of the head's sizes.
    let named = bed.placed_at(&keyed, 2, None, now);
    assert_eq!(named.holders, [(2, 0, 3)]);
    assert_eq!(named.tree, Some(TreeMatch::head(48, 64)));
    let odd: Vec<u32> = [1, 2, 3].iter().chain(&ids).copied().collect();
    let named = bed.placed_at(&odd, 3, Some(3), now);
    let nobody: &[(u32, u32, u32)] = &[];
    assert_eq!((named.holders.as_slice(), named.tree), (nobody, None));
    assert_eq!(placed.len(), 4);
}

/// A lent placement map is written for the first target of the pick its
/// caller sends to, asked best first, and for no worker where the caller
/// sends to none, as a lent tree stores a prompt. An entry whose marker
/// lies past its ids names nobody: nothing is written, and the caller is
/// asked of no target.
///
/// The classic policy writes the worker it selected, among the candidates
/// its router can send to (`select_worker_hash`, `record_placement`,
/// model_gateway/src/policies/cache_aware.rs).
#[test]
fn a_placement_map_is_written_for_the_first_target_its_caller_sends_to() {
    let (affinity, _, placed) = lending_placed(false);
    let bed = Bed::new(affinity, hashed_at(&[BLOCK], 60), 4);
    // The workers at 1, 2 and 3, best first.
    let three = || {
        let mut targets = Targets::new();
        for position in 1..4 {
            targets.push(Target {
                position,
                id: bed.ids[position as usize],
                ..Target::default()
            });
        }
        Decision::Pick(Pick {
            targets,
            trace: Trace::default(),
            session_verdict: SessionVerdict::None,
        })
    };
    // One decision of `key`, whose first `marker` ids are a partition's,
    // that picks the three, under a caller that sends to the places `sent`
    // names. Says the places the caller was asked of.
    let place = |key: &[u32], marker: usize, sent: &[u32]| {
        let hashes = hashed(key.get(marker..).unwrap_or_default());
        let asked = RefCell::new(Vec::new());
        let entry = TokenKey {
            tokens: Cow::Borrowed(key),
            marker,
        };
        decide(
            &bed.affinity,
            &bed.pool,
            &bed.spec,
            &bed.workers,
            &bed.ids,
            &mut request(&hashes, 7, NOW),
            None,
            |_| Some(entry),
            || None,
            |target| {
                asked.borrow_mut().push(target.position);
                sent.contains(&target.position)
            },
            |_| three(),
        );
        asked.into_inner()
    };
    // Who the map holds for the first head of `key`.
    let holders = |key: &[u32]| {
        let end = BLOCK as usize;
        let held = placed.get(&(end, hash_token_head(None::<[u32; 2]>, &key[..end])));
        let holders = held.iter().flat_map(|held| held.iter());
        holders
            .map(|holder| holder.worker_url.clone())
            .collect::<Vec<_>>()
    };
    // A caller that sends to the first target is asked of it alone.
    let (first, second, none) = (tokens(1_000, 2), tokens(2_000, 2), tokens(3_000, 2));
    assert_eq!(place(&first, 0, &[1, 2, 3]), [1]);
    assert_eq!(holders(&first), [address(1)]);
    // One that passes the first over is written for the second alone.
    assert_eq!(place(&second, 0, &[2, 3]), [1, 2]);
    assert_eq!(holders(&second), [address(2)]);
    // One that sends to none is asked of each, and nothing is written.
    assert_eq!(place(&none, 0, &[]), [1, 2, 3]);
    assert!(holders(&none).is_empty());
    assert_eq!(placed.len(), 2);
    // A marker past the entry's ids: no entry of the map.
    assert!(place(&[7], 2, &[1, 2, 3]).is_empty());
    assert_eq!(placed.len(), 2);
}

/// Where a host lends its placement map, a request that names no entry in
/// it is named nobody and writes nothing, and the engine's record of sent
/// prefixes is not read in the map's place. The classic policy balances a
/// request with no token ids by load under a hash index
/// (`select_worker_hash`, model_gateway/src/policies/cache_aware.rs).
#[test]
fn a_request_that_names_no_entry_in_a_lent_placement_map_is_named_nobody() {
    let asked = tokens(1_000, 4);
    let nobody: &[(u32, u32, u32)] = &[];
    let (affinity, _, placed) = lending_placed(false);
    let bed = Bed::new(affinity, hashed_at(&[BLOCK], 60), 4);
    for _ in 0..2 {
        let named = bed.decide(&request(&hashed(&asked), 7, NOW), Some(1));
        let named = named.expect("rows named");
        assert_eq!((named.holders.as_slice(), named.tree), (nobody, None));
    }
    assert!(placed.is_empty());
}

/// Under a hash index the table asks its host for the map only where the
/// list's profile reads who a prompt was sent to. A profile whose prefix
/// stage reads the index of cache events alone (`source: index`) is lent
/// none: a request that names its entry is named nobody, and its pick
/// writes nothing in the host's map.
#[test]
fn a_profile_that_reads_no_sent_prefix_is_lent_no_map() {
    use crate::document::spec::{AffinitySource, IndexSpec, ScoreSpec};

    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    document.view.boundaries = vec![BLOCK];
    let index = IndexSpec {
        kind: IndexKind::Hash,
        ttl_s: 60,
        ..IndexSpec::default()
    };
    document.index.insert("*".into(), index);
    let profile = document.profiles.get_mut("cache_aware");
    for stage in &mut profile.expect("the preset's profile").score {
        if let ScoreSpec::Params(ScoreStage::PrefixAffinity(prefix)) = stage {
            prefix.source = Some(AffinitySource::Index);
        }
    }
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    assert_eq!(spec.engine().pool_record_for(MODEL, Phase::Single), None);
    let (affinity, _, placed) = lending_placed(false);
    let bed = Bed::new(affinity, spec, 4);
    let (asked, now) = (tokens(1_000, 2), Instant::now());
    let nobody: &[(u32, u32, u32)] = &[];
    for _ in 0..2 {
        let named = bed.placed_at(&asked, 0, Some(1), now);
        assert_eq!((named.holders.as_slice(), named.tree), (nobody, None));
    }
    assert!(placed.is_empty());
}

/// A model its workers list behind their first is decided inside the
/// placement map its host lends for it, under its own index entry. The
/// document sets a hash index for the shared model alone, and the host
/// lends its map under that model's id alone: the first worker of the list
/// states another model first, whose index is a tree and which is lent no
/// map.
#[test]
fn a_shared_model_of_its_workers_is_decided_inside_the_map_lent_for_it() {
    use smg_worker::{
        identity::builder::BasicWorkerBuilder, model_card::ModelCard, test_support::no_health_check,
    };

    use crate::document::spec::IndexSpec;

    const SHARED: &str = "model-s";
    /// A host that lends its placement map for the shared model alone.
    struct Lender(Arc<PlacementMap>);
    impl CacheIndexes for Lender {
        fn index(&self, _: &str) -> Option<Arc<dyn CacheIndex>> {
            None
        }

        fn block_size(&self, _: &str) -> Option<u32> {
            Some(BLOCK)
        }

        fn placement_map(&self, model: &str, phase: Phase) -> Option<Arc<PlacementMap>> {
            (model == SHARED && phase == Phase::Single).then(|| Arc::clone(&self.0))
        }
    }
    let placed = Arc::new(PlacementMap::default());
    let host = Lender(Arc::clone(&placed));
    let affinity = Affinity::default();
    affinity.set_indexes(Arc::new(host) as Arc<dyn CacheIndexes>);
    let mut document = PolicySpec::preset("cache_aware").expect("the preset");
    document.generation = 1;
    document.view.boundaries = vec![BLOCK];
    let index = IndexSpec {
        kind: IndexKind::Hash,
        ..IndexSpec::default()
    };
    document.index.insert(SHARED.into(), index);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let mut bed = Bed::new(affinity, spec, 2);
    for (at, first) in [MODEL, "model-b"].into_iter().enumerate() {
        let cards = vec![ModelCard::new(first), ModelCard::new(SHARED)];
        let built = <BasicWorkerBuilder>::new(address(at)).models(cards);
        bed.workers[at] = Arc::new(built.health_config(no_health_check()).build());
    }
    // A request for the shared model, sent to the worker at 1: the map lent
    // for that model holds its head for that worker.
    let (asked, now) = (tokens(1_000, 4), Instant::now());
    bed.placed_for(SHARED, &asked, Some(1), now);
    assert_eq!(placed.len(), 1);
    let named = bed.placed_for(SHARED, &asked, None, now);
    assert_eq!(named.holders, [(1, 0, 1)]);
}
