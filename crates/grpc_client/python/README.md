# smg-grpc-proto

[![PyPI](https://img.shields.io/pypi/v/smg-grpc-proto)](https://pypi.org/project/smg-grpc-proto/)
[![Python](https://img.shields.io/pypi/pyversions/smg-grpc-proto)](https://pypi.org/project/smg-grpc-proto/)

Protocol Buffer definitions for [SMG](https://github.com/smg-project/smg) (Shepherd Model Gateway) gRPC services.

This package provides pre-compiled Python gRPC stubs for:
- **SMG Worker** control plane (`worker_control.proto`) and data plane (`worker_inference.proto`)
- **SGLang** scheduler service (`sglang_scheduler.proto`)
- **SGLang** encoder service (`sglang_encoder.proto`)
- **TokenSpeed** scheduler and encoder services (`tokenspeed_scheduler.proto`, `tokenspeed_encoder.proto`)
- **vLLM** engine service (`vllm_engine.proto`)
- **TensorRT-LLM** service (`trtllm_service.proto`)
- **MLX** engine service (`mlx_engine.proto`)

## Installation

```bash
pip install smg-grpc-proto
```

Requires `grpcio>=1.81.1` and `protobuf>=5.26.0`.

## Usage

```python
from smg_grpc_proto import worker_control_pb2, worker_control_pb2_grpc
from smg_grpc_proto import worker_inference_pb2, worker_inference_pb2_grpc
from smg_grpc_proto import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc
from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc
from smg_grpc_proto import tokenspeed_scheduler_pb2, tokenspeed_scheduler_pb2_grpc
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
from smg_grpc_proto import trtllm_service_pb2, trtllm_service_pb2_grpc
from smg_grpc_proto import mlx_engine_pb2, mlx_engine_pb2_grpc
```

## Proto Source

The proto source files live in [`grpc_client/proto/`](https://github.com/smg-project/smg/tree/main/grpc_client/proto) in the SMG repository. Python stubs are generated at build time using `grpcio-tools` and shipped in the wheel.

## Development

To install in editable mode from the repo root:

```bash
pip install -e grpc_client/python/
```

For CI or environments where symlinks don't work:

```bash
mkdir -p grpc_client/python/smg_grpc_proto/proto
cp grpc_client/proto/*.proto grpc_client/python/smg_grpc_proto/proto/
pip install -e grpc_client/python/
```

### Testing proto changes on a remote GPU machine

After editing `.proto` files locally, build a wheel and install it in the remote environment (e.g. vLLM):

```bash
# 1. Build wheel (regenerates Python stubs from latest .proto files)
cd grpc_client/python
# Copy proto files into the package tree (the repo uses a symlink which
# won't survive wheel packaging)
mkdir -p smg_grpc_proto/proto
cp ../proto/*.proto smg_grpc_proto/proto/
pip wheel . --no-deps -w dist/

# 2. Copy to remote
scp dist/smg_grpc_proto-*.whl remote-gpu:/tmp/

# 3. Install on remote (into vLLM's env or whichever env needs it)
pip install --force-reinstall /tmp/smg_grpc_proto-*.whl
```

No import changes are needed on the remote side — vLLM already imports from `smg_grpc_proto`.

## License

Apache-2.0
