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
| `--connection-mode` | `grpc` | Connection mode: `grpc`, `http`, or `zmq`. trtllm only supports grpc; tokenspeed only zmq; `zmq` (a same-host direct engine connection) is available for vllm, tokenspeed, and sglang. Over `zmq` the router waits for the engine's startup handshake, whose READY comes only once the engine has loaded its model: the router's `--zmq-startup-timeout-secs` (`--router-zmq-startup-timeout-secs` here; default 3600) bounds how long any one stage of it may stay silent, and must cover the engine's whole init |
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
| Oracle credentials | `--oracle-connect-descriptor`, `--oracle-username`, `--oracle-password` | `ATP_DSN`, `ATP_USER`, `ATP_PASSWORD` | – | – |

- The flag beats the environment. A value (from the flag or the environment) together with a file for the same secret is refused at start-up rather than ranked: the second source of one secret is a stale one about to be honoured.
- A single-value file holds the value alone; the surrounding whitespace (a trailing newline) is dropped, and so is the whitespace around a single value from the environment (a Secret made from a file carries the editor's newline into both); the `ATP_*` values are the exception, see below. An empty file, or one with more than one line, is refused.
- `SMG_TENANT_API_KEYS` holds `tenant_id:key` entries separated by `,` or a newline; the file holds one entry per line, blank lines and lines starting with `#` skipped. A source that is set but holds no entry is refused.
- A store URL file is read for the selected `--history-backend` alone. The launcher still reads the older `POSTGRES_DB_URL` and `REDIS_URL` names when the `SMG_` names are unset.
- The Oracle credentials have no file source; the launcher and the binary (`--oracle-dsn`, `--oracle-user`, `--oracle-password`) read the `ATP_*` variables when the flags are absent, as they always did, and use them exactly as set, with no trimming: a Secret for them is made with `--from-literal`, or from a file that ends without a newline (a password that ends in one is a wrong password to the database).
- No value is echoed: a refusal names the sources and the path, and `--help` shows no value of a variable. The start-up record (`CREDENTIAL SOURCES`) names each secret's source as `flag`, `env` or `file`; a launcher-started gateway writes the same record, with `launcher` for a key, or the selected history backend's store URL, that the launcher gave as a value (the binding does not see whether a flag or the environment gave it) and `file` for a key read from a file.

## Tokenizer

`smg.tokenizer.Tokenizer` is the tokenizer SMG serves with, for Python: the same Rust code (`llm-tokenizer`) loads the checkpoint, encodes and decodes, and chats render through the gateway's own request path, so a training or evaluation pipeline gets the token ids SMG feeds the model without running a gateway.

```python
from smg.tokenizer import Tokenizer

# What the gateway loads, and the id it serves it as.
tokenizer = Tokenizer("/models/Qwen3-8B", model="Qwen/Qwen3-8B")
ids = tokenizer.encode("hello world")  # list[int]
batch = tokenizer.encode_batch(["hello world", "how are you?"])  # parallel, on a Rust thread pool
text = tokenizer.decode(ids)

messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello!"}]
# The ids SMG would feed the model for this conversation.
ids = tokenizer.apply_chat_template(messages)
prompt = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, tokenize=False)

weather = {"type": "object", "properties": {"city": {"type": "string"}}}
tools = [{"type": "function", "function": {"name": "get_weather", "parameters": weather}}]
request = {"messages": messages[:1], "tools": tools, "reasoning_effort": "low"}
rendered = tokenizer.render_chat(request)
rendered["token_ids"], rendered["text"], rendered["from_text"], rendered["warnings"]
```

- `apply_chat_template` takes `transformers`' shape (a conversation or a list of them, `tools`, `add_generation_prompt=False`, `continue_final_message`, `tokenize=True`, extra keyword arguments for the template) and builds the `/v1/chat/completions` request a client would send; `render_chat` takes that request body itself (a mapping or its JSON text, so logged bodies replay as they are; `render_chat_batch` renders a list of them in parallel) and returns the prompt text, its ids, how they were rendered and the rewrites made on the way. Both go through serving's steps: the request deserialized, normalized and validated as the HTTP layer does and refused where the gRPC router refuses it (a refused request raises `ValueError` with the gateway's reason; a tool call needs its `id` and `type`), message content shaped for the template, tool-call `arguments` strings parsed into objects for templates that serialize them, `tool_choice`, `reasoning_effort` and `chat_template_kwargs` handed to the template as the gateway hands them, then the prompt encoded as the gateway does: the text without special tokens, or the renderer's own encode for renderers that encode segment by segment (Kimi K3; DeepSeek V3.2 / V4 / V4.1 when message text spells a control token), where `from_text` is false and no encode of the text gives the ids.
- A request renders under `Tokenizer.model`, as the router renders it under the served model's id (a request naming another model, a client's alias, is rewritten to it): the id selects the provider profile (`provider_profile`: DeepSeek, Kimi, MiniMax, GLM) whose request rewrites apply. It defaults to the Hub id of a Hub source or of a Hugging Face cache snapshot (`.../models--{org}--{name}/snapshots/...`); for any other path, pass `model=` the id SMG serves the checkpoint as, or a vendor-named alias of it (the gateway's `--model-alias` registry is not read). Without one, a request renders under the model it names itself, which selects a profile by that name alone (a logged body carries the client's id, perhaps an alias: pass `model=` when replaying them), and a request that names none, as every `apply_chat_template` call, raises.
- Parity holds against a gateway built from the same smg commit (`Tokenizer.smg_build`; compare the gateway's `--version-verbose`) that loads the same files: pass what the gateway loads (its tokenizer path, else its model path; `config.json` and `generation_config.json` beside the tokenizer files select the renderer and the stop ids, `eos_token_ids`), and `chat_template=` the file the gateway's `--chat-template` gets. `Tokenizer.fingerprint` digests the build, the model id, the chat template file, a file source itself and the checkpoint's tokenizer files (those the gateway's Hub download fetches, with `config.json` and `generation_config.json`, and DeepSeek V4.1's `encoding/encoding.py`, which that download does not fetch): tokenizers with one fingerprint render and encode alike.
- `apply_chat_template` raises where `transformers` would do something SMG does not: a keyword argument naming a variable the render fills itself (`strftime_now` or a special token the checkpoint defines, which `transformers` lets the argument replace, or `messages`) is `TypeError`, and so are a `transformers` argument SMG has no equivalent for (`chat_template`, `documents`, `padding`, `truncation`, `max_length`, `return_tensors`, `return_dict`, `return_assistant_tokens_mask`, `tokenizer_kwargs`, `sanitize_control_tokens`) not given its no-op value and a `continue_final_message` naming a field other than `content`; a rendering that does not honour `add_generation_prompt` or `continue_final_message` (a renderer whose own encoder writes the generation header after a final tool or user message; a continued message that is not the assistant's) is `ValueError`, while `render_chat` returns such a rendering as SMG makes it. A final assistant message the renderer cannot continue itself is continued as SMG continues it, its text appended after the generation header (`generation_prompt` true, `continued_final_message` false). The rewrites serving only logs (tool-call `arguments` that are not a JSON object rendered as `{}`, another provider's message extensions dropped) are `RenderedChat["warnings"]`, and `apply_chat_template` issues them as `smg.tokenizer.RenderWarning`.
- `encode` adds the post-processor's special tokens by default (`add_special_tokens=True`), as `transformers` does; SMG encodes `/v1/completions`, `/generate` and `/v1/tokenize` text with `add_special_tokens=False` and embedding input with `True`. Special-token text in the input encodes to the special token. `decode` takes a list, a numpy array or a torch tensor of ids, and does not apply `transformers`' `clean_up_tokenization_spaces`. A legacy-format Llama-class `tokenizer.json` (a `Prepend` normalizer) is encoded as `transformers`' `LlamaTokenizer` class builds it, the pipeline the engines load, which can differ from `transformers` 4.x in leading-space pieces.
- Short calls (an encode of less than 16 KiB, a decode of fewer than 4096 ids, a chat whose request and prompt are shorter than 16 KiB) run on the calling thread with the GIL held, as Hugging Face's `tokenizers` runs a single encode: releasing it would cost more than the work, since a thread that releases the GIL takes it back only when the thread running Python meanwhile gives it up (up to the 5 ms switch interval later). Encode many texts with `encode_batch`. The batch methods and longer calls release the GIL. A decode of 4096 ids or more, the parse, checks and render of a chat request of 16 KiB or more, a Hub download, and reading and hashing the loaded files for `Tokenizer.fingerprint` stay on the calling thread; what can reach rayon (the batch methods, `apply_chat_template` given several conversations, building the tokenizer, and any encode of 16 KiB or more, a chat's prompt among them) runs on the binding's own Rust thread pool, one per process, of `SMG_TOKENIZER_THREADS` threads (one when `TOKENIZERS_PARALLELISM` is off, rayon's default otherwise: `RAYON_NUM_THREADS`, else the CPU count); each worker of a multi-process pipeline builds its own on first use, so size it for the workers it runs beside. A process forked after the parent used the tokenizer (torch `DataLoader` workers, `datasets.map(num_proc=...)`) builds a pool of its own, where rayon's global pool would leave it hanging, and `os.fork()` waits for the tokenizer calls running in other threads (not for a Hub download, which holds none of their locks; a fork made from C waits for none). A tokenizer pickles as the files it loaded (a Hub id's snapshot), the smg build and the fingerprint, and reloads from those files in the receiving process, from the cache on the same host; where the path it loaded is missing (another host), from a download of the Hub snapshot it was in. Unpickling refuses another smg build, and files whose fingerprint changed (edited in place, only partly downloaded into this host's cache, or the Hub's `main` moved on since; a DeepSeek V4.1 tokenizer loaded with `encoding/encoding.py` reloads only where that file is too, since the Hub download does not fetch it); and because the fingerprint is in the pickle, `datasets.map` recomputes after a template or tokenizer file changes.
- Hub model ids download as the gateway downloads them (`HF_TOKEN`, `HF_HOME` and `HF_ENDPOINT` are honoured, the revision is always `main`); for a pinned revision or an offline host, download with `huggingface_hub.snapshot_download` and pass the directory.
- Not reproduced: media content (refused: SMG expands each image, video or audio item into the model's own tokens), gpt-oss models (refused: SMG renders them with Harmony), and the date a template writes with `strftime_now`, which is the local time at the render, as in serving (the same `llm-tokenizer` code). Where `SOURCE_DATE_EPOCH` (whole seconds since the Unix epoch) is set when the process loads its first chat template, the template writes that instant instead, also in local time (set `TZ=UTC0` too for the same text on every Linux or macOS host; on Windows the system time zone applies): `llm-tokenizer` reads the variable once, then, so set it in the environment the process starts with (a worker forked after its parent loaded one keeps the parent's reading).
- Importing the tokenizer loads the whole `smg` extension: its jemalloc settings apply to the Rust side's allocations (`_RJEM_MALLOC_CONF` overrides them), and the wheel's dependencies (`grpcio`, `grpcio-health-checking`, `setproctitle`, `PyYAML`) come with it.

## Directory Structure

```
bindings/python/
├── src/                    # Source code (src layout)
│   ├── lib.rs              # Rust/PyO3 bindings implementation
│   ├── tokenizer.rs        # The tokenizer binding (Rust half)
│   └── smg/                # Python source code
│       ├── __init__.py
│       ├── cli.py          # CLI entry point
│       ├── serve.py        # smg serve implementation
│       ├── launch_router.py
│       ├── router.py
│       ├── router_args.py
│       └── tokenizer.py    # smg.tokenizer.Tokenizer
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
