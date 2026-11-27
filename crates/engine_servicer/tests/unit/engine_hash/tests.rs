use super::*;
use crate::kv_events::golden::bytes as hex;

fn digest(hex_text: &str) -> Digest32 {
    hex(hex_text).try_into().expect("32 bytes")
}

// SGLang vectors published with the router's hash tests
// (experimental/sgl-router/src/state/kv_events/hash.rs).
#[test]
fn sglang_reproduces_the_published_vectors() {
    let ints = |tokens: &[u32], page| -> Vec<i64> {
        sglang_chain(tokens, page, None)
            .into_iter()
            .map(|(_, int)| int)
            .collect()
    };
    assert_eq!(ints(&[1, 2, 3, 4], 4), vec![-3488128144981237669]);
    assert_eq!(
        ints(&[10, 20, 30, 40, 50, 60, 70, 80], 2),
        vec![
            978178666101069530,
            -895308556211281782,
            -8033692805846017938,
            835415944263129316
        ]
    );
    assert_eq!(
        sglang_event_int(&Sha256::digest(b"").into()),
        -2039914840885289964
    );
    assert!(ints(&[1, 2, 3], 4).is_empty(), "only full pages are hashed");
    assert!(sglang_chain(&[1, 2, 3, 4], 0, None).is_empty());
}

// Derived with hashlib from the algorithm above (vectors file in the
// branch notes): salted and bigram chains.
#[test]
fn sglang_salted_and_bigram_chains() {
    let seed = sglang_salt_seed("tenant-a");
    assert_eq!(
        seed,
        digest("f5d0f785efe6042a4e4b0297a4d712917e1763c850835e93df58b373fd47fd2c")
    );
    let salted: Vec<i64> = sglang_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4, Some(&seed))
        .into_iter()
        .map(|(_, int)| int)
        .collect();
    assert_eq!(salted, vec![3718046898735569995, 3308615664605479373]);
    let plain = sglang_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None);
    assert_eq!(
        plain[0].0,
        digest("cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72")
    );
    assert_eq!(
        plain[1].0,
        digest("4ebfa8a1f3c341517621838c6e1b9aa350307e3f00b3cbd1a07ef740f54396d6")
    );
    // Bigram page (1,2) (2,3) (3,4) (4,5): both words of every pair.
    let bigram = sglang_page(None, &[1, 2, 2, 3, 3, 4, 4, 5]);
    assert_eq!(sglang_event_int(&bigram), -638950109823820341);
}

// vLLM vectors: `hash_block_tokens` with `sha256_cbor` from the checkout
// at 0c16eee3f1, run under cbor2 with PYTHONHASHSEED unset.
#[test]
fn vllm_reproduces_the_reference_run() {
    assert_eq!(
        vllm_none_hash(),
        digest("9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0")
    );
    let a = vllm_block(None, &[1, 2, 3, 4], None);
    assert_eq!(
        a,
        digest("58d0879dff3800f65f8c5fd449d73048c0f7699dd238a03b84b146b151d55111")
    );
    assert_eq!(vllm_event_int(&a), -8885242862429187823);
    assert_eq!(vllm_event_int(&a) as u64, 9561501211280363793);
    let b = vllm_block(Some(&a), &[5, 6, 7, 8], None);
    assert_eq!(
        b,
        digest("e1bc29393a2da9fd798f35ff04d215f846aee6df4b803ce3d43b57df56b58fb1")
    );
    assert_eq!(vllm_event_int(&b), -3153830497298837583);
    let chain = vllm_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None);
    assert_eq!(
        chain,
        vec![(a, vllm_event_int(&a)), (b, vllm_event_int(&b))]
    );

    let lora = VllmExtraKey::Lora {
        name: "adapter".into(),
        path: "/adapters/adapter".into(),
    };
    let c = vllm_block(
        None,
        &[1, 2, 3, 4],
        Some(&[
            lora.clone(),
            VllmExtraKey::Mm {
                identifier: "mm-abc".into(),
                offset: 0,
            },
            VllmExtraKey::CacheSalt("salt-1".into()),
            VllmExtraKey::PromptEmbeds((0u8..32).collect()),
        ]),
    );
    assert_eq!(
        c,
        digest("280bce663478b44320e68be593605214c94f7648ac34b4b94274fdcd0f2a8a04")
    );
    assert_eq!(vllm_event_int(&c), 4788731360966248964);
    let d = vllm_block(
        Some(&c),
        &[5, 6, 7, 8],
        Some(&[
            lora,
            VllmExtraKey::Mm {
                identifier: "mm-abc".into(),
                offset: -4,
            },
        ]),
    );
    assert_eq!(
        d,
        digest("33e7394e25b7be46c7bf0c090da02e903b4e9d9536b1c18eed4d6e57e7a3658a")
    );
    assert_eq!(vllm_event_int(&d), -1347299389686454902);

    let e = vllm_block(
        None,
        &[1, 2, 3, 4],
        Some(&[
            VllmExtraKey::Mm {
                identifier: "mm-abc".into(),
                offset: 0,
            },
            VllmExtraKey::CacheSalt("salt-1".into()),
            VllmExtraKey::PromptEmbeds((0u8..32).collect()),
        ]),
    );
    assert_eq!(e, digest(VECTOR_E));
    let f = vllm_block(
        Some(&e),
        &[5, 6, 7, 8],
        Some(&[VllmExtraKey::Mm {
            identifier: "mm-abc".into(),
            offset: -4,
        }]),
    );
    assert_eq!(f, digest(VECTOR_F));
    let g = vllm_block(Some(&f), &[9, 10, 11, 12], None);
    assert_eq!(g, digest(VECTOR_G));
}

// The CBOR bytes `cbor2.dumps(value, canonical=True)` produced for the
// same inputs.
#[test]
fn cbor_matches_cbor2_canonical_bytes() {
    let mut seed = Vec::new();
    cbor::text(&mut seed, VLLM_NONE_HASH_SEED);
    assert_eq!(seed, hex("6e766c6c6d2d6e6f6e652d68617368"));

    let mut ints = Vec::new();
    cbor::array(&mut ints, 14);
    for value in [
        0i64,
        23,
        24,
        255,
        256,
        65535,
        65536,
        4_294_967_295,
        4_294_967_296,
    ] {
        cbor::int(&mut ints, value);
    }
    for value in [-1i64, -24, -25, -256, -257] {
        cbor::int(&mut ints, value);
    }
    assert_eq!(ints, hex(CBOR_INTS));

    // `[NONE, [1, 2, 3, 4], null]` and block D's input.
    let mut a_input = Vec::new();
    cbor::array(&mut a_input, 3);
    cbor::bytes(&mut a_input, &vllm_none_hash());
    cbor::array(&mut a_input, 4);
    for token in 1..=4u64 {
        cbor::uint(&mut a_input, token);
    }
    cbor::null(&mut a_input);
    assert_eq!(
        a_input,
        hex("8358209bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef08401020304f6")
    );
}

#[test]
fn setting_names_parse() {
    assert_eq!(EngineHash::parse("sglang"), Some(EngineHash::Sglang));
    assert_eq!(EngineHash::parse(" SGLang "), Some(EngineHash::Sglang));
    assert_eq!(
        EngineHash::parse("vllm_sha256_cbor"),
        Some(EngineHash::VllmSha256Cbor)
    );
    assert_eq!(
        EngineHash::parse("vllm-sha256-cbor"),
        Some(EngineHash::VllmSha256Cbor)
    );
    assert_eq!(EngineHash::parse("vllm"), None);
    assert_eq!(EngineHash::parse(""), None);
    assert_eq!(EngineHash::VllmSha256Cbor.as_str(), "vllm-sha256-cbor");
}

const VECTOR_E: &str = "0d7b0d8344b79ad5e183b117cacc04aeb415bfdfa968fd28f611bc6971f4abba";
const VECTOR_F: &str = "b81a4631bec909b72bcd0081cc2d7c87bcce5652c48318f9615e4aa3370c1d69";
const VECTOR_G: &str = "024cf74ecdb93c6049fe59a66321192aab04701a75ac3e86619395104df937c4";
const CBOR_INTS: &str =
    "8e0017181818ff19010019ffff1a000100001affffffff1b00000001000000002037381838ff390100";
