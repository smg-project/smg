import unittest
from unittest.mock import patch

from scripts.ci_install_flashinfer import install


class FlashInferInstallTest(unittest.TestCase):
    def test_stable_release_matches_installed_frontend_and_cuda(self):
        versions = {
            "flashinfer-python": "0.7.0",
            "flashinfer-cubin": "0.7.0",
            "flashinfer-jit-cache": "0.7.0+cu130",
        }
        with (
            patch("scripts.ci_install_flashinfer.metadata.version", versions.__getitem__),
            patch("scripts.ci_install_flashinfer.subprocess.check_call") as call,
        ):
            install("0.7.0", "13.0", "x86_64")
        command = call.call_args.args[0]
        self.assertIn("flashinfer-python==0.7.0", command)
        self.assertIn(
            "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.0/"
            "flashinfer_cubin-0.7.0-py3-none-any.whl",
            command,
        )
        self.assertIn(
            "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.0/"
            "flashinfer_jit_cache-0.7.0+cu130-cp39-abi3-manylinux_2_28_x86_64.whl",
            command,
        )
        self.assertEqual(command[-2:], ["--extra-index-url", "https://flashinfer.ai/whl/cu130"])

    def test_nightly_release_uses_matching_provider_index(self):
        version = "0.7.0.dev20260920"
        with (
            patch(
                "scripts.ci_install_flashinfer.metadata.version",
                side_effect=[version, version, f"{version}+cu129"],
            ),
            patch("scripts.ci_install_flashinfer.subprocess.check_call") as call,
        ):
            install(version, "12.9", "aarch64")
        command = call.call_args.args[0]
        self.assertIn(
            "https://github.com/flashinfer-ai/flashinfer/releases/download/"
            f"nightly-v0.7.0-20260920/flashinfer_jit_cache-{version}+cu129"
            "-cp39-abi3-manylinux_2_28_aarch64.whl",
            command,
        )
        self.assertEqual(command[-1], "https://flashinfer.ai/whl/nightly/cu129")


if __name__ == "__main__":
    unittest.main()
