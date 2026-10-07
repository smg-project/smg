# smg-symphony

One parser for everything a model says. A model's output stream interleaves content, reasoning and
tool calls in a vendor-specific format; this crate turns it into typed events through one object
with one method, `Parser::feed`, whose lifecycle is expressed as input: the prompt the engine was
given, each decoded delta, the end of the stream. Every byte of output lands in exactly one event,
including bytes that were dropped or could not be parsed.

Status: the public types are the contract, the adapters render them for the Chat Completions,
Responses and Messages APIs, and the engine runs a format table; the tables under `formats` are
the families recorded so far, and more follow.

Why the crate has a name: the community has spent years on many parser implementations, and none
of them is right the way we want it, SMG's own two crates included. Not right as working code;
right as a made thing. This crate is the attempt to write the one that is. Code here is a craft,
and "this can be better" is always a good enough reason for a change.

Rules this crate works under:

- It changes nothing outside `crates/symphony/`. It depends on other crates but never edits them;
  wiring it into the gateway is a separate step taken once recorded fixtures show parity.
- Formats are data. A format definition yields both the parser and the guided-decoding grammar;
  hand-written parsers are the exception and say why.
- Every format ships with fixtures recorded from the vendor's reference and the engines, replayed
  at many chunkings, and five property tests: conservation of bytes, prefix-stable arguments,
  chunking invariance, committed output stays committed, and token identity for markers.
- Every commit carries `Co-authored-by: Chang Su`, whose design this is.
