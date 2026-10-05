# smg-symphony

One parser for everything a model says. A model's output stream interleaves content, reasoning and
tool calls in a vendor-specific format; this crate turns it into typed events through one object
with one method, `Parser::feed`, whose lifecycle is expressed as input: the prompt the engine was
given, each decoded delta, the end of the stream. Every byte of output lands in exactly one event,
including bytes that were dropped or could not be parsed.

Status: skeleton. The public types are the contract; the engine, the format definitions and the
protocol adapters follow.

Rules this crate works under:

- It changes nothing outside `crates/symphony/`. It depends on other crates but never edits them;
  wiring it into the gateway is a separate step taken once recorded fixtures show parity.
- Formats are data. A format definition yields both the parser and the guided-decoding grammar;
  hand-written parsers are the exception and say why.
- Every format ships with fixtures recorded from the vendor's reference and the engines, replayed
  at many chunkings, and five property tests: conservation of bytes, prefix-stable arguments,
  chunking invariance, committed output stays committed, and token identity for markers.
- Every commit carries `Co-authored-by: Chang Su`, whose design this is.
