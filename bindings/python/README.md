# SMG Python Bindings

This directory contains the Python bindings for SMG (Shepherd Model Gateway), built using [maturin](https://github.com/PyO3/maturin) and [PyO3](https://github.com/PyO3/pyo3).

## Quick Start

### Installation

```bash
pip install maturin
cd smg/bindings/python
maturin develop --features vendored-openssl
```

### Usage

The `smg serve` command launches backend workers and the SMG router in a single command:

```bash
# sglang with gRPC (default)
smg serve --backend sglang --model-path /path/to/model --port 8080

# sglang with HTTP
smg serve --backend sglang --model-path /path/to/model --port 8080 --connection-mode http

# vLLM (gRPC only)
smg serve --backend vllm --model /path/to/model --port 8080

# TensorRT-LLM (gRPC only)
smg serve --backend trtllm --model /path/to/model --port 8080

# Multiple workers (data parallel)
smg serve --backend sglang --model-path /path/to/model --port 8080 --dp-size 4
```

### Serve Options

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `sglang` | Backend to use: `sglang`, `vllm`, or `trtllm` |
| `--connection-mode` | `grpc` | Connection mode: `grpc`, `http`, or `zmq`. trtllm only supports grpc; tokenspeed only zmq; `zmq` (a same-host direct engine connection) is available for vllm, tokenspeed, and sglang |
| `--host` | `127.0.0.1` | Host for the router |
| `--port` | `8080` | Port for the router |
| `--dp-size` | `1` | Data parallel size (number of worker replicas) |
| `--worker-host` | `127.0.0.1` | Host for worker processes |
| `--worker-base-port` | `31000` | Base port for workers |
| `--worker-startup-timeout` | `300` | Seconds to wait for workers to become healthy |

Backend-specific options (e.g., `--tensor-parallel-size`, `--quantization`) are passed through to the backend.

### Secrets from the environment or files

The serving keys and the store URLs need not be on the command line, where other processes can read them (`/proc/<pid>/cmdline`, process listings, a Deployment's args). Each is also read from an environment variable, or from a file such as a mounted Secret, by the Python launcher (`smg launch`, `smg serve` with the `--router-` prefix) and by the Rust binary alike:

| Setting | Flag | Environment | File flag | File environment |
|---------|------|-------------|-----------|------------------|
| Shared API key | `--api-key` | `SMG_API_KEY` | `--api-key-file` | `SMG_API_KEY_FILE` |
| Per-tenant API keys | `--tenant-api-key` (repeatable) | `SMG_TENANT_API_KEYS` | `--tenant-api-keys-file` | `SMG_TENANT_API_KEYS_FILE` |
| PostgreSQL URL | `--postgres-db-url` | `SMG_POSTGRES_DB_URL` | `--postgres-db-url-file` | `SMG_POSTGRES_DB_URL_FILE` |
| Redis URL | `--redis-url` | `SMG_REDIS_URL` | `--redis-url-file` | `SMG_REDIS_URL_FILE` |

- The flag beats the environment. A value (from the flag or the environment) together with a file for the same secret is refused at start-up rather than ranked: the second source of one secret is a stale one about to be honoured.
- A single-value file holds the value alone; the surrounding whitespace (a trailing newline) is dropped, and so is the whitespace around a single value from the environment (a Secret made from a file carries the editor's newline into both). An empty file, or one with more than one line, is refused.
- `SMG_TENANT_API_KEYS` holds `tenant_id:key` entries separated by `,` or a newline; the file holds one entry per line, blank lines and lines starting with `#` skipped. A source that is set but holds no entry is refused.
- A store URL file is read for the selected `--history-backend` alone. The launcher still reads the older `POSTGRES_DB_URL` and `REDIS_URL` names when the `SMG_` names are unset.
- No value is echoed: a refusal names the sources and the path, and `--help` shows no value of a variable. The Rust binary's start-up record (`CREDENTIAL SOURCES`) names each secret's source as `flag`, `env` or `file`; the Python launcher emits no such record.

## Directory Structure

```
bindings/python/
├── src/                    # Source code (src layout)
│   ├── lib.rs              # Rust/PyO3 bindings implementation
│   └── smg/                # Python source code
│       ├── __init__.py
│       ├── cli.py          # CLI entry point
│       ├── serve.py        # smg serve implementation
│       ├── launch_router.py
│       ├── router.py
│       └── router_args.py
├── tests/                  # Python unit tests
├── Cargo.toml              # Rust package configuration
├── pyproject.toml          # Python package configuration
└── README.md               # This file
```

## Building

### Development Build

```bash
pip install maturin
cd smg/bindings/python
maturin develop --features vendored-openssl
```

### Production Build

```bash
cd smg/bindings/python
maturin build --release --out dist --features vendored-openssl
pip install dist/smg-*.whl
```

## Testing

```bash
cd smg/bindings/python
pytest tests/
```

## Adding a router flag

A flag lives in a few places, and the tests pin them against each other:

1. `model_gateway/src/main.rs`: the clap flag, in its group of `CliArgs`. The Rust CLI is the reference: `tests/test_cli_parity.py` fails on a flag that exists on one side only.
2. `src/smg/router_args.py`: a `RouterArgs` field, and its option in `add_cli_args` under the matching help group. The field goes at its alphabetical place among the keyword-only fields (every field after `worker_urls`, `host`, `port`), never at the tail: `tests/test_arg_parser.py` pins the order and says where a misplaced field belongs.
3. `src/lib.rs`: the `Router` struct field, the `#[pyo3(signature)]` entry with the flag's default, the parameter of `new` and the field of the literal `new` builds, each at the same alphabetical place among the keyword-only parameters (every parameter after `worker_urls`, `policy`, `host`, `port`; pyo3 wants the signature and the parameters in one order), plus the line that hands the value to the router configuration in `to_router_config`. `tests/test_router_config.py` pins the order; `tests/test_router_args_parity.py` fails on a dataclass field the constructor does not take, and on a parameter the launcher never sets.
4. The flag's own tests go next to the related ones, or in a file of their own, not at the end of `tests/test_arg_parser.py`, where every flag's tests would meet.

The keyword-only parameters have no position, so two pull requests that each add a flag change different lines of each file and merge without a conflict.
