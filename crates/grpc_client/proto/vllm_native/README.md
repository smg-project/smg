# Vendored vLLM native gRPC protocol

- Source repository: `vllm-project/vllm` (Apache-2.0)
- Upstream path: `rust/proto/{inference,control}.proto`
- Vendored at commit: `466855a2bfc57e58f7e2d03ee2deba79ceab2617`
- SHA-256:
  - `inference.proto`: `6152c306583166ecd691c9c715cab950523e8d1ed2db3dc2bcb538f6ca90e56f`
  - `control.proto`: `390c88e94f1b68421c54c6d9440f2088d2709a432549c7a0fe94d35ce7b37476`

These are vLLM's first-party gRPC services (`vllm.Inference` for generation,
`vllm.Control` for server/model info, abort, and KV event source discovery),
served by the engine itself — no injected servicer required. Files are vendored
byte-for-byte with upstream license headers retained. Update the commit and
checksums together when re-vendoring; the client in `src/vllm_native.rs` is
written against this revision.
