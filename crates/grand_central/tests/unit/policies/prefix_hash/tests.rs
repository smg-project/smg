use super::*;

/// The heads `rule` states of a prompt, in the order it states them.
fn stated<'a>(
    rule: HeadRule<'_>,
    tokens: Option<&'a [u32]>,
    text: Option<&'a str>,
) -> Vec<Head<'a>> {
    let mut heads = Vec::new();
    rule.heads(tokens, text, |head| heads.push(head));
    heads
}

/// A prompt is cut where the gateway's own policy cuts it
/// (`model_gateway/src/policies/prefix_hash.rs`, lines 143 to 205 at
/// 882be6f4f): a head at each boundary it reaches, the shallowest
/// first, where the last token of a boundary reaches it. A prompt that
/// reaches none has the one head of the rule's depth, and is its own
/// head where it is shorter. A text is measured at four characters a
/// token and cut on a character's edge, whatever bytes a character
/// takes. Token ids lead a text, an empty list of ids leaves the text,
/// and a prompt with nothing to read has no head.
#[test]
fn a_prompt_has_the_heads_the_gateways_own_policy_cuts() {
    let rule = HeadRule {
        depth: 128,
        boundaries: &[64, 256],
    };
    let ids: Vec<u32> = (0..300).collect();
    let tokens = |len: usize| Head::Tokens(&ids[..len]);
    assert_eq!(stated(rule, Some(&ids), None), [tokens(64), tokens(256)]);
    let at_the_last = stated(rule, Some(&ids[..256]), None);
    assert_eq!(at_the_last, [tokens(64), tokens(256)]);
    assert_eq!(stated(rule, Some(&ids[..255]), None), [tokens(64)]);
    assert_eq!(stated(rule, Some(&ids[..63]), None), [tokens(63)]);
    let deep = HeadRule {
        depth: 128,
        boundaries: &[256],
    };
    assert_eq!(stated(deep, Some(&ids[..200]), None), [tokens(128)]);

    // A text: boundaries at eight and sixteen characters, a depth of
    // twelve.
    let rule = HeadRule {
        depth: 3,
        boundaries: &[2, 4],
    };
    let text = "abcdefghijklmnopq";
    let cut = |len: usize| Head::Text(&text[..len]);
    assert_eq!(stated(rule, None, Some(text)), [cut(8), cut(16)]);
    assert_eq!(stated(rule, None, Some(&text[..16])), [cut(8), cut(16)]);
    assert_eq!(stated(rule, None, Some(&text[..15])), [cut(8)]);
    assert_eq!(stated(rule, None, Some(&text[..7])), [cut(7)]);
    let unbounded = HeadRule {
        depth: 3,
        boundaries: &[],
    };
    assert_eq!(stated(unbounded, None, Some(text)), [cut(12)]);
    // The rule's depth counts characters too: twelve of two bytes each.
    let wide = "\u{e9}".repeat(14);
    let twelve = [Head::Text(&wide[..24])];
    assert_eq!(stated(unbounded, None, Some(&wide)), twelve);
    // Boundaries a validated document never holds, out of order: the
    // gateway's own cut reads on from where the text was read to
    // (the same file, lines 185 to 205).
    let unordered = HeadRule {
        depth: 1,
        boundaries: &[2, 1, 3],
    };
    let cuts = [cut(8), cut(8), cut(12)];
    assert_eq!(stated(unordered, None, Some(&text[..16])), cuts);
    // Nine characters of two bytes each: the head of eight holds
    // sixteen bytes.
    let wide = "\u{e9}".repeat(9);
    assert_eq!(stated(rule, None, Some(&wide)), [Head::Text(&wide[..16])]);

    assert_eq!(
        stated(rule, Some(&ids[..5]), Some(text)),
        [tokens(2), tokens(4)]
    );
    assert_eq!(stated(rule, Some(&[]), Some(&text[..7])), [cut(7)]);
    assert!(stated(rule, Some(&[]), Some("")).is_empty());
    assert!(stated(rule, None, None).is_empty());
}

/// A head's debug form says its size and nothing it holds: a client's
/// token ids and text reach no log line through it.
#[test]
fn a_heads_debug_form_says_its_size_alone() {
    assert_eq!(format!("{:?}", Head::Text("a client's words")), "Text(16)");
    assert_eq!(format!("{:?}", Head::Tokens(&[7, 8, 9])), "Tokens(3)");
}
