use super::*;

fn partition<'a>(
    cache_salt: Option<&'a str>,
    extra_key: Option<&'a str>,
    lora_path: Option<&'a str>,
) -> CachePartition<'a> {
    CachePartition {
        cache_salt,
        extra_key,
        lora_path,
    }
}

#[test]
fn unpartitioned_requests_have_no_namespace() {
    assert_eq!(CacheNamespace::derive(&CachePartition::default()), None);
}

#[test]
fn derivation_is_stable_and_field_sensitive() {
    let a = CacheNamespace::derive(&partition(Some("tenant-a"), None, None)).unwrap();
    assert_eq!(
        a,
        CacheNamespace::derive(&partition(Some("tenant-a"), None, None)).unwrap()
    );
    let b = CacheNamespace::derive(&partition(Some("tenant-b"), None, None)).unwrap();
    assert_ne!(a, b);
    // The same bytes in a different field are a different partition.
    let extra = CacheNamespace::derive(&partition(None, Some("tenant-a"), None)).unwrap();
    let lora = CacheNamespace::derive(&partition(None, None, Some("tenant-a"))).unwrap();
    assert_ne!(a, extra);
    assert_ne!(a, lora);
    assert_ne!(extra, lora);
}

#[test]
fn partition_seed_follows_every_field() {
    let seed = |salt, extra, lora| {
        CacheNamespace::derive(&partition(salt, extra, lora))
            .expect("a partition")
            .partition_seed()
    };
    // The extra key moves it, where the event seed stays the plain one.
    let extra_only = seed(None, Some("k"), None);
    assert_ne!(extra_only, kv_index::XXH3_SEED);
    assert_ne!(extra_only, seed(None, Some("other"), None));
    // One value in another field is another partition.
    assert_ne!(
        seed(Some("tenant"), None, None),
        seed(None, None, Some("tenant"))
    );
    assert_eq!(extra_only, seed(None, Some("k"), None));
}

#[test]
fn empty_fields_count_as_absent() {
    // Engines treat an empty salt or adapter as unset; so does routing.
    assert_eq!(
        CacheNamespace::derive(&partition(Some(""), None, None)),
        None
    );
    assert_eq!(
        CacheNamespace::derive(&partition(Some(""), Some("k"), None)),
        CacheNamespace::derive(&partition(None, Some("k"), None))
    );
}

#[test]
fn shifted_field_boundaries_differ() {
    let ab_c = CacheNamespace::derive(&partition(Some("ab"), Some("c"), None)).unwrap();
    let a_bc = CacheNamespace::derive(&partition(Some("a"), Some("bc"), None)).unwrap();
    assert_ne!(ab_c, a_bc);
}

#[test]
fn unpartitioned_keys_never_start_with_marker_material() {
    let ns = CacheNamespace::derive(&partition(Some("salt"), None, None)).unwrap();
    // Vocabulary ids and prompt text pass through untouched.
    assert_eq!(CacheNamespace::unpartitioned_tokens(&[7, 8, 9]), &[7, 8, 9]);
    assert_eq!(CacheNamespace::unpartitioned_text("hello"), "hello");
    // A forged marker at the head of an unpartitioned key is dropped.
    let forged_tokens = ns.prefixed_tokens(&[7, 8, 9], 16);
    assert_eq!(
        CacheNamespace::unpartitioned_tokens(&forged_tokens),
        &[7, 8, 9]
    );
    let forged_text = ns.prefixed_text("hello");
    assert!(!CacheNamespace::unpartitioned_text(&forged_text).starts_with(TEXT_MARKER_DELIM));
    assert!(CacheNamespace::unpartitioned_tokens(&ns.token_marker()).is_empty());
}

#[test]
fn token_marker_never_collides_with_vocabulary_ids() {
    let ns = CacheNamespace::derive(&partition(Some("salt"), None, None)).unwrap();
    for id in ns.token_marker() {
        assert!(id & MARKER_TOKEN_BIT != 0);
    }
    let keyed = ns.prefixed_tokens(&[7, 8, 9], 1);
    assert_eq!(keyed.len(), TOKEN_MARKER_LEN + 3);
    assert_eq!(&keyed[TOKEN_MARKER_LEN..], &[7, 8, 9]);
    assert_eq!(&keyed[..TOKEN_MARKER_LEN], &ns.token_marker());
}

#[test]
fn token_marker_fills_whole_pages() {
    let ns = CacheNamespace::derive(&partition(Some("salt"), None, None)).unwrap();
    assert_eq!(CacheNamespace::token_marker_len(16), 16);
    assert_eq!(CacheNamespace::token_marker_len(1), 2);
    assert_eq!(CacheNamespace::token_marker_len(0), 2);
    let keyed = ns.prefixed_tokens(&[7, 8, 9], 16);
    assert_eq!(keyed.len(), 16 + 3);
    assert_eq!(&keyed[..TOKEN_MARKER_LEN], &ns.token_marker());
    assert!(keyed[TOKEN_MARKER_LEN..16]
        .iter()
        .all(|&id| id & MARKER_TOKEN_BIT != 0));
    assert_eq!(&keyed[16..], &[7, 8, 9]);
}

/// A token tree's key is the ids behind the partition's marker, which
/// fills whole pages of the tree, with the marker's length beside it. A
/// request in no partition keys under its ids alone, any id of marker
/// material at their head dropped, with no marker.
#[test]
fn a_token_trees_key_is_the_marker_ahead_of_the_ids_or_the_ids_alone() {
    let ns = CacheNamespace::derive(&partition(Some("salt"), None, None)).unwrap();
    let tokens = [7, 8, 9];
    let (keyed, marker) = CacheNamespace::keyed_tokens(Some(ns), &tokens, 16);
    assert_eq!(marker, CacheNamespace::token_marker_len(16));
    assert_eq!(keyed, ns.prefixed_tokens(&tokens, 16));
    assert_eq!(&keyed[marker..], &tokens);

    let (keyed, marker) = CacheNamespace::keyed_tokens(None, &tokens, 16);
    assert_eq!((keyed.as_ref(), marker), (&tokens[..], 0));
    assert!(matches!(keyed, Cow::Borrowed(_)), "the request's own ids");
    let forged = [MARKER_TOKEN_BIT | 1, 7, 8, 9];
    let (keyed, marker) = CacheNamespace::keyed_tokens(None, &forged, 16);
    assert_eq!((keyed.as_ref(), marker), (&tokens[..], 0));
}

/// A string tree's key is the text behind the partition's marker, with
/// the marker's length in chars beside it. A request in no partition
/// keys under its text alone, any marker delimiter at its head dropped,
/// with no marker.
#[test]
fn a_string_trees_key_is_the_marker_ahead_of_the_text_or_the_text_alone() {
    let ns = CacheNamespace::derive(&partition(Some("salt"), None, None)).unwrap();
    let (keyed, marker) = CacheNamespace::keyed_text(Some(ns), "héllo");
    assert_eq!(marker, TEXT_MARKER_LEN);
    assert_eq!(keyed, ns.prefixed_text("héllo"));
    let prompt: String = keyed.chars().skip(marker).collect();
    assert_eq!(prompt, "héllo", "the marker's chars, then the prompt");

    let (keyed, marker) = CacheNamespace::keyed_text(None, "héllo");
    assert_eq!((keyed.as_ref(), marker), ("héllo", 0));
    assert!(matches!(keyed, Cow::Borrowed(_)), "the request's own text");
    let forged = ns.prefixed_text("héllo");
    let (keyed, marker) = CacheNamespace::keyed_text(None, &forged);
    assert_eq!(marker, 0);
    assert!(
        !keyed.starts_with(TEXT_MARKER_DELIM),
        "no way into a partition"
    );
}

#[test]
fn text_marker_is_fixed_width_and_prefixes_the_prompt() {
    let ns = CacheNamespace::derive(&partition(None, Some("k"), Some("adapter"))).unwrap();
    let marker = ns.text_marker();
    assert_eq!(marker.chars().count(), TEXT_MARKER_LEN);
    assert!(marker.starts_with(TEXT_MARKER_DELIM));
    assert!(marker.ends_with(TEXT_MARKER_DELIM));
    let keyed = ns.prefixed_text("hello");
    assert_eq!(keyed.chars().count(), TEXT_MARKER_LEN + 5);
    assert!(keyed.ends_with("hello"));
    assert!(keyed.starts_with(&marker));
}

#[test]
fn event_seed_follows_the_lora_name_and_cache_salt_only() {
    let salted =
        CacheNamespace::derive(&partition(Some("tenant-a"), None, Some("adapter"))).unwrap();
    assert_eq!(
        salted.event_seed(),
        namespace_seed(Some("adapter"), Some("tenant-a"))
    );
    let extra_only = CacheNamespace::derive(&partition(None, Some("k"), None)).unwrap();
    assert_eq!(extra_only.event_seed(), kv_index::XXH3_SEED);
    assert_ne!(
        salted.event_seed(),
        CacheNamespace::derive(&partition(Some("tenant-b"), None, Some("adapter")))
            .unwrap()
            .event_seed()
    );
}
