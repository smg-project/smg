"""Install release artifacts matching TokenSpeed's installed FlashInfer."""

import platform
import re
import subprocess
import sys
from importlib import metadata


def install(version: str, cuda_version: str, architecture: str) -> None:
    cuda_index = "".join(cuda_version.split(".")[:2])
    nightly = re.fullmatch(r"(\d+\.\d+\.\d+)\.dev(\d{8})", version)
    tag = f"nightly-v{nightly.group(1)}-{nightly.group(2)}" if nightly else f"v{version}"
    release = f"https://github.com/flashinfer-ai/flashinfer/releases/download/{tag}"
    index = f"https://flashinfer.ai/whl/{'nightly/' if nightly else ''}cu{cuda_index}"
    jit_version = f"{version}+cu{cuda_index}"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            f"flashinfer-python=={version}",
            f"{release}/flashinfer_cubin-{version}-py3-none-any.whl",
            f"{release}/flashinfer_jit_cache-{jit_version}"
            f"-cp39-abi3-manylinux_2_28_{architecture}.whl",
            "--extra-index-url",
            index,
        ]
    )
    for package, expected in (
        ("flashinfer-python", version),
        ("flashinfer-cubin", version),
        ("flashinfer-jit-cache", jit_version),
    ):
        installed = metadata.version(package)
        if installed != expected:
            raise RuntimeError(f"{package}: expected {expected}, installed {installed}")
        print(f"{package}=={installed}", flush=True)


if __name__ == "__main__":
    import torch

    if torch.version.cuda is None:
        raise RuntimeError("TokenSpeed CI requires a CUDA-enabled Torch build")
    install(metadata.version("flashinfer-python"), torch.version.cuda, platform.machine())
