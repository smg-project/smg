# vLLM gRPC launcher fixtures

Byte-identical copies of vLLM's gRPC launcher files (Apache-2.0, SPDX headers
kept), for `test_vllm_launcher_switch_main.py`'s CPU smoke, which executes
each launcher through to the servicer construction against a stub engine and
checks that the servicer switch engaged (`scripts/launcher_switch_smoke.py`).
Kept unformatted and excluded from ruff so they stay diffable against upstream
(the files with a shebang are marked executable for the pre-commit hook; the
content is upstream's).

| directory | upstream path | tag | notes |
|---|---|---|---|
| `v0_28/vllm/entrypoints/grpc_server.py` | `vllm/entrypoints/grpc_server.py` | v0.28.0 | the same file from v0.24.0 through v0.29.0: the launcher module itself, `python -m vllm.entrypoints.grpc_server` runs it as `__main__` |
| `v0_30/vllm/entrypoints/launchers/grpc_server.py` | `vllm/entrypoints/launchers/grpc_server.py` | v0.30.0 | the launcher moved; `vllm serve --grpc` imports it from here |
| `v0_30/vllm/entrypoints/grpc_server.py` | `vllm/entrypoints/grpc_server.py` | v0.30.0 | the deprecated shim: `from vllm.entrypoints.launchers.grpc_server import main, serve_grpc` (unchanged through v0.31.0) |
| `v0_31/vllm/entrypoints/launchers/grpc_server.py` | `vllm/entrypoints/launchers/grpc_server.py` | v0.31.0 | gRPC keepalive options and `configure_logging_from_args`; the hook points are unchanged |
| `v0_31/vllm/entrypoints/grpc_server.py` | `vllm/entrypoints/grpc_server.py` | v0.31.0 | the shim |

The hook points are the same in all of them: the servicer classes imported
at module level (inside a `try`), then `async def serve_grpc(args)`, then a
`main()` that builds the parser with `AsyncEngineArgs.add_cli_args` (which
loads this package's plugin), parses, and runs `uvloop.run(serve_grpc(args))`
under `if __name__ == "__main__"`.
