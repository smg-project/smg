# SMG Python Bindings

This directory contains the Python bindings for SMG (Shepherd Model Gateway), built using [maturin](https://github.com/PyO3/maturin) and [PyO3](https://github.com/PyO3/pyo3).

## Quick Start

### Installation

```bash
pip install maturin
cd smg/bindings/python
maturin develop --features extension-module,vendored-openssl
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
| `--connection-mode` | `grpc` | Connection mode: `grpc` or `http`. vllm/trtllm only support grpc |
| `--host` | `127.0.0.1` | Host for the router |
| `--port` | `8080` | Port for the router |
| `--dp-size` | `1` | Data parallel size (number of worker replicas) |
| `--worker-host` | `127.0.0.1` | Host for worker processes |
| `--worker-base-port` | `31000` | Base port for workers |
| `--worker-startup-timeout` | `300` | Seconds to wait for workers to become healthy |

Backend-specific options (e.g., `--tensor-parallel-size`, `--quantization`) are passed through to the backend.

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
maturin develop --features extension-module,vendored-openssl
```

### Production Build

```bash
cd smg/bindings/python
maturin build --release --out dist --features extension-module,vendored-openssl
pip install dist/smg-*.whl
```

## Testing

```bash
cd smg/bindings/python
pytest tests/
```
