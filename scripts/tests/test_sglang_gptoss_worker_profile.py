"""The actual stdlib-only worker profile, independent of engine imports."""

from __future__ import annotations

import copy
import importlib.util
import os
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
        self.environment = patch.dict(
            os.environ,
            {"E2E_GPU_VENDOR": "amd", "E2E_SGLANG_GPTOSS_MXFP4_BF16": "1"},
            clear=True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.spec = {"model": MODEL, "sglang_args": ["--context-length", "4096"]}

    def launch(self, *, engine="sglang", model=MODEL, path=MODEL, spec=None, extra=None):
        return profile.prepare_launch(
            engine, model, path, self.spec if spec is None else spec, extra
        )

    def test_original_packed_checkpoint_and_served_id_are_retained(self):
        path, spec = self.launch()
        self.assertEqual(path, f"/models/hub/models--openai--gpt-oss-20b/snapshots/{REVISION}")
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
        os.environ["HF_HOME"] = "/owned/hf"
        self.assertEqual(
            self.launch()[0], f"/owned/hf/hub/models--openai--gpt-oss-20b/snapshots/{REVISION}"
        )
        os.environ["HF_HUB_CACHE"] = "/owned/hub"
        self.assertEqual(
            self.launch()[0], f"/owned/hub/models--openai--gpt-oss-20b/snapshots/{REVISION}"
        )

    def test_source_model_spec_and_extra_arguments_are_not_mutated(self):
        extra = ["--trust-remote-code"]
        before, extra_before = copy.deepcopy(self.spec), list(extra)
        _, configured = self.launch(extra=extra)
        self.assertEqual(self.spec, before)
        self.assertEqual(extra, extra_before)
        self.assertIsNot(configured, self.spec)
        self.assertIsNot(configured["sglang_args"], self.spec["sglang_args"])

    def test_explicit_original_snapshot_keeps_its_path_and_original_name(self):
        original = f"/models/hub/models--openai--gpt-oss-20b/snapshots/{REVISION}"
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
            ("", "sglang", MODEL),
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
