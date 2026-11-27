"""The actual stdlib-only worker profile, independent of engine imports."""

from __future__ import annotations

import copy
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_SOURCE = Path(__file__).resolve().parents[2] / "e2e_test/infra/sglang_gptoss.py"
_SPEC = importlib.util.spec_from_file_location("sglang_gptoss_worker_profile", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
profile = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profile)

MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"


class WorkerProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.hf_home = Path(self.temporary.name) / "hf"
        self.snapshot = self.make_snapshot(self.hf_home / "hub")
        self.environment = patch.dict(
            os.environ,
            {
                "E2E_GPU_VENDOR": "amd",
                "E2E_SGLANG_GPTOSS_MXFP4_BF16": "1",
                "HF_HOME": str(self.hf_home),
            },
            clear=True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.spec = {"model": MODEL, "sglang_args": ["--context-length", "4096"]}

    def make_snapshot(self, cache):
        snapshot = cache / "models--openai--gpt-oss-20b" / "snapshots" / REVISION
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        return str(snapshot)

    def launch(
        self, *, engine="sglang", model=MODEL, path=MODEL, spec=None, extra=None, extra_env=None
    ):
        return profile.prepare_launch(
            engine, model, path, self.spec if spec is None else spec, extra, extra_env
        )

    def test_original_packed_checkpoint_and_served_id_are_retained(self):
        path, spec = self.launch()
        self.assertEqual(path, self.snapshot)
        self.assertEqual(
            spec["sglang_args"],
            [
                "--context-length",
                "4096",
                "--served-model-name",
                MODEL,
                "--moe-runner-backend",
                "triton",
                "--moe-a2a-backend",
                "none",
            ],
        )

    def test_cache_environment_precedence(self):
        alternate_home = Path(self.temporary.name) / "alternate-home"
        alternate_hub = Path(self.temporary.name) / "alternate-hub"
        home_snapshot = self.make_snapshot(alternate_home / "hub")
        hub_snapshot = self.make_snapshot(alternate_hub)
        os.environ["HF_HOME"] = str(alternate_home)
        self.assertEqual(self.launch()[0], home_snapshot)
        os.environ["HF_HUB_CACHE"] = str(alternate_hub)
        self.assertEqual(self.launch()[0], hub_snapshot)

    def test_worker_cache_overrides_are_used_without_mutating_environment(self):
        worker_home = Path(self.temporary.name) / "worker-home"
        worker_hub = Path(self.temporary.name) / "worker-hub"
        home_snapshot = self.make_snapshot(worker_home / "hub")
        hub_snapshot = self.make_snapshot(worker_hub)
        # Only the child cache has the pinned revision.
        (Path(self.snapshot) / "config.json").unlink()
        inherited_before = dict(os.environ)
        extra = {"HF_HOME": str(worker_home)}
        self.assertEqual(self.launch(extra_env=extra)[0], home_snapshot)
        self.assertEqual(extra, {"HF_HOME": str(worker_home)})
        os.environ["HF_HUB_CACHE"] = str(worker_hub)
        self.assertEqual(self.launch(extra_env=extra)[0], hub_snapshot)
        self.assertEqual(
            self.launch(extra_env={"HF_HUB_CACHE": "", "HF_HOME": str(worker_home)})[0],
            home_snapshot,
        )
        os.environ.pop("HF_HUB_CACHE")
        self.assertEqual(self.launch(extra_env={"HF_HUB_CACHE": str(worker_hub)})[0], hub_snapshot)
        self.assertEqual(dict(os.environ), inherited_before)

    def test_missing_fixed_snapshot_is_refused_before_launch_preparation_returns(self):
        config = Path(self.snapshot) / "config.json"
        config.unlink()
        before = copy.deepcopy(self.spec)
        for kind in ("missing", "directory"):
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(
                    FileNotFoundError, rf"{MODEL}@{REVISION}.*config.json.*effective HF cache"
                ),
            ):
                if kind == "directory":
                    config.mkdir()
                self.launch()
            self.assertEqual(self.spec, before)

    def test_explicit_vendor_is_normalized_like_the_harness(self):
        for vendor in ("amd", "AMD", " amd "):
            with self.subTest(vendor=vendor):
                os.environ["E2E_GPU_VENDOR"] = vendor
                self.assertEqual(self.launch()[0], self.snapshot)
                env = {}
                profile.apply_environment("sglang", MODEL, env, None)
                self.assertEqual(env["SGLANG_GPTOSS_MXFP4_BF16_FALLBACK"], "1")

    def test_unset_vendor_matches_harness_smi_availability_without_running_smi(self):
        for vendor in (None, "", " "):
            for nvidia, amd, selected in (
                (None, "/mock/amd-smi", True),
                ("/mock/nvidia-smi", "/mock/amd-smi", False),
                ("/mock/nvidia-smi", None, False),
                (None, None, False),
            ):
                with self.subTest(vendor=vendor, nvidia=nvidia, amd=amd):
                    if vendor is None:
                        os.environ.pop("E2E_GPU_VENDOR", None)
                    else:
                        os.environ["E2E_GPU_VENDOR"] = vendor
                    with patch.object(
                        profile.shutil,
                        "which",
                        side_effect=lambda name: nvidia if name == "nvidia-smi" else amd,
                    ):
                        path, spec = self.launch()
                        self.assertEqual(path, self.snapshot if selected else MODEL)
                        if not selected:
                            self.assertIs(spec, self.spec)
                        env = {}
                        profile.apply_environment("sglang", MODEL, env, None)
                        self.assertEqual(bool(env), selected)

    def test_source_model_spec_and_extra_arguments_are_not_mutated(self):
        extra = ["--trust-remote-code"]
        before, extra_before = copy.deepcopy(self.spec), list(extra)
        _, configured = self.launch(extra=extra)
        self.assertEqual(self.spec, before)
        self.assertEqual(extra, extra_before)
        self.assertIsNot(configured, self.spec)
        self.assertIsNot(configured["sglang_args"], self.spec["sglang_args"])

    def test_explicit_original_snapshot_keeps_its_path_and_original_name(self):
        original = self.snapshot
        path, spec = self.launch(path=original)
        self.assertEqual(path, original)
        args = spec["sglang_args"]
        self.assertEqual(args[args.index("--served-model-name") + 1], MODEL)

    def test_unqualified_local_alias_and_different_revision_are_refused(self):
        for path in (
            "/models/original-20b-copy",
            "/models/hub/models--openai--gpt-oss-20b/snapshots/another-revision",
        ):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "checkpoint revision"):
                self.launch(path=path)

    def test_matching_served_name_is_not_duplicated(self):
        for supplied in (["--served-model-name", MODEL], [f"--served-model-name={MODEL}"]):
            with self.subTest(supplied=supplied):
                spec = {"sglang_args": supplied}
                _, configured = self.launch(spec=spec)
                self.assertEqual(
                    sum(
                        arg.split("=", 1)[0] == "--served-model-name"
                        for arg in configured["sglang_args"]
                    ),
                    1,
                )

    def test_conflicting_and_missing_served_names_are_refused(self):
        for supplied in (
            ["--served-model-name", "another-model"],
            ["--served-model-name=another-model"],
            ["--served-model-name"],
        ):
            with (
                self.subTest(supplied=supplied),
                self.assertRaisesRegex(ValueError, "served model name"),
            ):
                self.launch(extra=supplied)

    def test_model_revision_and_backend_overrides_are_refused_in_both_sources(self):
        for flag in (
            "--model-path",
            "--model",
            "--revision",
            "--moe-runner-backend",
            "--moe-a2a-backend",
        ):
            for supplied in ([flag, "override"], [f"{flag}=override"]):
                with (
                    self.subTest(supplied=supplied),
                    self.assertRaisesRegex(ValueError, "owns its checkpoint"),
                ):
                    self.launch(extra=supplied)
                with (
                    self.subTest(spec=supplied),
                    self.assertRaisesRegex(ValueError, "owns its checkpoint"),
                ):
                    self.launch(spec={"sglang_args": supplied})

    def test_reserved_flag_abbreviations_cannot_override_owned_inputs(self):
        for flag in ("--rev", "--moe-runner-back", "--model-p"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "owns its checkpoint"):
                self.launch(extra=[flag, "override"])
        with self.assertRaisesRegex(ValueError, "full served model name flag"):
            self.launch(extra=["--served-model-n", "another-model"])

    def test_nvidia_default_launch_and_environment_are_unchanged(self):
        os.environ["E2E_GPU_VENDOR"] = "nvidia"
        path, spec = self.launch()
        self.assertEqual(path, MODEL)
        self.assertIs(spec, self.spec)
        env = {"SGLANG_USE_AITER": "1", "CUSTOM": "unchanged"}
        before = dict(env)
        profile.apply_environment("sglang", MODEL, env, None)
        self.assertEqual(env, before)

    def test_unset_or_disabled_profile_keeps_existing_inputs(self):
        for value in (None, "0"):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop("E2E_SGLANG_GPTOSS_MXFP4_BF16", None)
                else:
                    os.environ["E2E_SGLANG_GPTOSS_MXFP4_BF16"] = value
                self.assertEqual(self.launch()[0], MODEL)
                self.assertIs(self.launch()[1], self.spec)
                env = {"SGLANG_USE_AITER": "1"}
                profile.apply_environment("sglang", MODEL, env, None)
                self.assertEqual(env, {"SGLANG_USE_AITER": "1"})

    def test_unknown_vendor_other_engines_and_original_120b_do_not_opt_in(self):
        for vendor, engine, model in (
            ("unknown-vendor", "sglang", MODEL),
            ("amd", "vllm", MODEL),
            ("amd", "sglang", "openai/gpt-oss-120b"),
            ("amd", "sglang", "Qwen/Qwen3-VL-8B-Instruct"),
        ):
            with self.subTest(vendor=vendor, engine=engine, model=model):
                os.environ["E2E_GPU_VENDOR"] = vendor
                path, spec = self.launch(engine=engine, model=model, path=model)
                self.assertEqual(path, model)
                self.assertIs(spec, self.spec)
                env = {"CUSTOM": "unchanged"}
                profile.apply_environment(engine, model, env, None)
                self.assertEqual(env, {"CUSTOM": "unchanged"})

    def test_unknown_selected_profile_is_refused(self):
        os.environ["E2E_SGLANG_GPTOSS_MXFP4_BF16"] = "true"
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            self.launch()
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            profile.apply_environment("sglang", MODEL, {}, None)

    def test_selected_environment_sets_only_the_original_required_pair(self):
        env = {"CUSTOM": "unchanged", "SGLANG_USE_AITER": "1"}
        profile.apply_environment("sglang", MODEL, env, None)
        self.assertEqual(
            env,
            {
                "CUSTOM": "unchanged",
                "SGLANG_USE_AITER": "0",
                "SGLANG_GPTOSS_MXFP4_BF16_FALLBACK": "1",
            },
        )

    def test_conflicting_per_worker_environment_is_refused_without_partial_update(self):
        for extra in (
            {"SGLANG_USE_AITER": "1"},
            {"SGLANG_GPTOSS_MXFP4_BF16_FALLBACK": "0"},
        ):
            env = {"CUSTOM": "unchanged"}
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "requires AITER"):
                profile.apply_environment("sglang", MODEL, env, extra)
            self.assertEqual(env, {"CUSTOM": "unchanged"})

    def test_matching_per_worker_environment_is_supported(self):
        env = {}
        extra = {"SGLANG_USE_AITER": "0", "SGLANG_GPTOSS_MXFP4_BF16_FALLBACK": "1"}
        profile.apply_environment("sglang", MODEL, env, extra)
        self.assertEqual(env, extra)


if __name__ == "__main__":
    unittest.main()
