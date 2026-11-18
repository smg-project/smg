# Router images

The default [`Dockerfile`](Dockerfile) builds the Ubuntu router image.
[`ol9.Dockerfile`](ol9.Dockerfile) builds the router on Oracle Linux 9, including
FFmpeg/OpenCV video support and the TurboJPEG/WebP image decoders.

## Oracle Linux 9

Build from the repository root:

```bash
docker build -f docker/ol9.Dockerfile --target router-image -t smg:ol9 .
docker run --rm smg:ol9 --help
```

The build uses the digest-pinned public [`oraclelinux:9`](https://hub.docker.com/_/oraclelinux)
image, public Oracle Linux yum repositories (including CodeReady Builder),
PyPI, crates.io, Rust's public toolchain distribution, and the same checksummed
FFmpeg/OpenCV source releases as the Ubuntu image. No private registry,
corporate proxy, or repository credentials are required.

Python defaults to OL9's Python 3.12; override it with
`--build-arg PYTHON_VERSION=3.11` to use another version available in the OL9
repositories. Rust follows the repository's `rust-toolchain.toml`. The wheel
links against OL9's OpenSSL and the media libraries copied from the build stages;
it is intended for this image rather than redistribution as a manylinux wheel.
The final image runs as UID 65532 and contains no compiler or Rust toolchain.

This Dockerfile builds the standalone router. Run the inference engine in a
separate container. It does not include the Ubuntu image's optional Slurm clients.

### Test against a gRPC worker

For example, with a vLLM gRPC worker named `worker` listening on port 8080 on a
Docker network named `smg-test`, and a local model checkpoint at
`/path/to/models/google/gemma-4-E4B-it`:

```bash
docker run -d --name smg-ol9 --network smg-test \
  -p 127.0.0.1:30000:8080 \
  -v /path/to/models:/models:ro \
  smg:ol9 \
  --host 0.0.0.0 --port 8080 --backend vllm \
  --model-path /models/google/gemma-4-E4B-it \
  --worker-urls grpc://worker:8080

curl --fail http://127.0.0.1:30000/readiness
curl --fail http://127.0.0.1:30000/v1/models
curl --fail http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"vllm-model","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

The request's model name must match the worker's `--served-model-name`.
The router must be able to read the model's tokenizer and configuration;
mounting the same checkpoint path into both containers also supports offline
startup. For a complete worker setup, see the
[gRPC servicer documentation](../grpc_servicer/README.md).
