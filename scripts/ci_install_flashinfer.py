"""Install release artifacts matching TokenSpeed's installed FlashInfer."""

import platform
import re
import subprocess
import sys
from importlib import metadata

from pip._vendor.packaging.requirements import Requirement


def artifacts_match(expected: tuple[tuple[str, str], ...]) -> bool:
    try:
        if any(metadata.version(package) != version for package, version in expected):
            return False
        for requirement in metadata.requires("flashinfer-jit-cache") or []:
            dependency = Requirement(requirement)
            if dependency.marker is not None and not dependency.marker.evaluate():
                continue
            if metadata.version(dependency.name) not in dependency.specifier:
                return False
    except metadata.PackageNotFoundError:
        return False
    return True


def install(version: str, cuda_version: str, architecture: str) -> None:
    cuda_index = "".join(cuda_version.split(".")[:2])
    nightly = re.fullmatch(r"(\d+\.\d+\.\d+)\.dev(\d{8})", version)
    tag = f"nightly-v{nightly.group(1)}-{nightly.group(2)}" if nightly else f"v{version}"
    release = f"https://github.com/flashinfer-ai/flashinfer/releases/download/{tag}"
    index = f"https://flashinfer.ai/whl/{'nightly/' if nightly else ''}cu{cuda_index}"
    jit_version = f"{version}+cu{cuda_index}"
    expected = (
        ("flashinfer-python", version),
        ("flashinfer-cubin", version),
        ("flashinfer-jit-cache", jit_version),
    )
    if artifacts_match(expected):
        print("FlashInfer artifacts and providers already match; skipping install", flush=True)
        return
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
    if not artifacts_match(expected):
        raise RuntimeError("FlashInfer artifacts or provider dependencies do not match")
    for package, version in expected:
        print(f"{package}=={version}", flush=True)


if __name__ == "__main__":
    import torch

    if torch.version.cuda is None:
        raise RuntimeError("TokenSpeed CI requires a CUDA-enabled Torch build")
    install(metadata.version("flashinfer-python"), torch.version.cuda, platform.machine())
