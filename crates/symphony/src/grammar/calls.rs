//! What the tag-based call syntaxes share: the calls in the table's block or as tags of their
//! own, and the arguments a tool's schema asks for as pairs of the syntax's own tags.

use serde_json::Value;

use super::{
    block_of,
    schema::{self, shape},
    CallMarkers, Count, Grammar, Tag, WayIn,
};

/// The calls, each its begin text, its content and its end text, as the engine takes them: in a
/// block, the block's ways in around `inside` of one call (any of them); without one, one tag per
/// call, triggered by the call's opener.
pub(super) fn in_block(
    markers: &CallMarkers<'_>,
    ways_in: &[WayIn<'_>],
    calls: Vec<(String, Grammar, String)>,
    count: Count<'_>,
    inside: impl FnOnce(Grammar) -> Grammar,
) -> Grammar {
    match markers.block.as_ref() {
        Some(block) => {
            let one_call = Grammar::Or(
                calls
                    .into_iter()
                    .map(|(begin, content, end)| {
                        Grammar::Sequence(vec![
                            Grammar::ConstString(begin),
                            content,
                            Grammar::ConstString(end),
                        ])
                    })
                    .collect(),
            );
            block_of(ways_in, inside(one_call), block.close, count)
        }
        None => count.triggered(
            vec![markers.call_open.to_string()],
            calls
                .into_iter()
                .map(|(begin, content, end)| Tag::new(begin, content, end))
                .collect(),
        ),
    }
}

/// The arguments the tool's schema asks for: one pair per property, in the schema's order, each
/// optional unless required, its value by the property's shape or the syntax's `text` where the
/// shape gives none; `any` pairs at all for a schema without properties.
pub(super) fn arguments(
    parameters: &Value,
    text: impl Fn() -> Grammar,
    argument: impl Fn(Grammar, Grammar) -> Grammar,
    any: impl FnOnce() -> Grammar,
) -> Grammar {
    schema::arguments(
        parameters,
        |key, property, definitions| {
            let value = shape(property, definitions, &text).map_or_else(&text, |(_, value)| value);
            argument(Grammar::ConstString(key.to_string()), value)
        },
        || Grammar::Star(Box::new(any())),
    )
}
