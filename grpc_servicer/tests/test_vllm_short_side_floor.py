"""The short-side floor on the worker-side media path (engine-free): which
models have one, the staged size, and the raise itself. The expected sizes
are the ones the router's own pipeline stages for the same images, and the
token grids are the ones its tests pin."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.vllm import short_side_floor as floor_mod
from smg_grpc_servicer.vllm.short_side_floor import (
    SHORT_SIDE_FLOORS,
    FloorRule,
    floor_rule,
    raise_to_floor,
    raise_to_floor_off_loop,
    raised_dimensions,
    short_side_floor,
    smart_resize,
)

FAMILIES = [pytest.param(entry, id=entry.family) for entry in SHORT_SIDE_FLOORS]
M3 = next(entry for entry in SHORT_SIDE_FLOORS if entry.family == "minimax_m3")


def rule_of(entry) -> FloorRule:
    return FloorRule(
        floor=entry.floor,
        factor=entry.factor,
        min_pixels=entry.min_pixels,
        max_pixels=entry.max_pixels,
    )


M3_RULE = rule_of(M3)


class _Image:
    """What the fetcher returns, as far as the raise reads it: a size and a
    resize method that answers with a new image of the asked size."""

    def __init__(self, width: int, height: int, mode: str = "RGB"):
        self.size = (width, height)
        self.mode = mode
        self.resized_with: list[tuple[tuple[int, int], int]] = []

    def resize(self, size, resample):
        self.resized_with.append((tuple(size), resample))
        return _Image(*size, mode=self.mode)


class TestRaisedDimensions:
    @pytest.mark.parametrize("entry", FAMILIES)
    def test_tiny_images_are_raised_to_a_square_at_the_floor(self, entry):
        floor = entry.floor
        assert raised_dimensions(16, 16, floor) == (floor, floor)
        assert raised_dimensions(1, 1, floor) == (floor, floor)
        assert raised_dimensions(floor - 1, floor - 1, floor) == (floor, floor)

    @pytest.mark.parametrize("entry", FAMILIES)
    def test_a_short_side_at_the_floor_or_zero_is_left_alone(self, entry):
        floor = entry.floor
        assert raised_dimensions(floor, 400, floor) is None
        assert raised_dimensions(400, floor, floor) is None
        assert raised_dimensions(floor + 1, floor + 1, floor) is None
        assert raised_dimensions(0, 40, floor) is None
        assert raised_dimensions(40, 0, floor) is None

    def test_the_router_pipelines_sizes_for_minimax_m3(self):
        # The sizes the router's own pipeline raises to (its processor's tests).
        floor = M3.floor
        assert raised_dimensions(400, 40, floor) == (1120, 112)
        assert raised_dimensions(40, 400, floor) == (112, 1120)
        assert raised_dimensions(50, 50, floor) == (112, 112)
        assert raised_dimensions(112, 20, floor) == (627, 112)
        assert raised_dimensions(112, 400, floor) is None
        assert raised_dimensions(0, 40, floor) is None
        # The verifier's band: a fixed long side, short sides below the floor.
        assert raised_dimensions(400, 90, floor) == (498, 112)
        assert raised_dimensions(400, 20, floor) == (2240, 112)
        assert raised_dimensions(400, 10, floor) == (4480, 112)
        assert raised_dimensions(400, 5, floor) == (8960, 112)

    def test_rounding_is_half_up_as_the_router_pipeline_rounds(self):
        # 35 * 112 / 32 = 122.5: half away from zero, not to even.
        assert raised_dimensions(32, 35, 112) == (112, 123)
        assert raised_dimensions(35, 32, 112) == (123, 112)

    def test_an_image_thinner_than_the_resize_takes_is_left_to_its_refusal(self):
        # Above 200:1 the family's own resize refuses the image; raising it
        # first would only stage a huge one before the same refusal.
        assert raised_dimensions(1, 300, 112) is None
        assert raised_dimensions(300, 1, 112) is None
        assert raised_dimensions(1, 200, 112) == (112, 22400)
        assert raised_dimensions(2, 401, 112) is None
        assert raised_dimensions(2, 400, 112) == (112, 22400)


GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "crates/multimodal/tests/fixtures/golden/qwen_smart_resize.json"
)


class TestSmartResize:
    def test_matches_the_pipelines_golden_fixture(self):
        # The family's resize as the reference processor computes it, case by
        # case: the fixture was generated with the reference's own function.
        if not GOLDEN.exists():
            pytest.skip("the pipeline's golden fixture is not in this checkout")
        family = json.loads(GOLDEN.read_text())["families"]["minimax_m3"]
        assert (family["min_pixels"], family["max_pixels"]) == (M3.min_pixels, M3.max_pixels)
        bounds = {
            "factor": family["patch_size"] * family["merge_size"],
            "min_pixels": family["min_pixels"],
            "max_pixels": family["max_pixels"],
        }
        assert bounds["factor"] == M3.factor
        for case in family["cases"]:
            if "error" in case:
                with pytest.raises(ValueError, match="aspect ratio"):
                    smart_resize(case["height"], case["width"], **bounds)
            else:
                assert (
                    list(smart_resize(case["height"], case["width"], **bounds)) == (case["resized"])
                ), case

    def test_zero_sides_and_a_zero_budget_are_handled_as_the_pipeline_handles_them(self):
        with pytest.raises(ValueError, match="positive"):
            smart_resize(0, 10, factor=28, min_pixels=3136, max_pixels=451584)
        # A pixel budget of zero still plans one grid cell, not an empty grid.
        assert smart_resize(10, 10, factor=28, min_pixels=0, max_pixels=0) == (28, 28)


class TestStagedDimensions:
    def test_within_the_budget_the_raised_image_is_staged(self):
        for (width, height), expected in [
            ((16, 16), (112, 112)),
            ((1, 1), (112, 112)),
            ((400, 90), (498, 112)),
            ((400, 40), (1120, 112)),
            ((400, 20), (2240, 112)),
            ((20, 400), (112, 2240)),
        ]:
            assert M3_RULE.staged_dimensions(width, height) == expected

    def test_over_the_budget_the_familys_own_target_is_staged_when_it_holds(self):
        # 4480x112 and 8960x112 overshoot the budget; the resize's target for
        # each fits it and comes back unchanged from the resize, so that is
        # what the engine's processor is handed: one resample, and its own
        # resize leaves it (the pipeline's tests count 453 and 428 tokens).
        assert M3_RULE.staged_dimensions(400, 10) == (4228, 84)
        assert M3_RULE.staged_dimensions(400, 5) == (5992, 56)
        assert M3_RULE.staged_dimensions(10, 400) == (84, 4228)

    def test_a_target_the_resize_refuses_stages_a_shrunk_image_just_over_the_budget(self):
        # 2x400 raised is 112x22400 (200:1); the resize's target for it is
        # thinner than the resize takes, so the raised image is shrunk to about
        # 1.25x the budget with the ratio kept inside the guard.
        staged = M3_RULE.staged_dimensions(2, 400)
        assert staged == (54, 10625)
        w, h = staged
        assert M3_RULE.max_pixels < w * h <= M3_RULE.max_pixels * 1.3
        assert max(w, h) / min(w, h) <= 200
        assert M3_RULE.staged_dimensions(400, 2) == (10625, 54)

    def test_a_lowered_budget_stages_the_target_for_it(self):
        lowered = FloorRule(floor=112, factor=28, min_pixels=3136, max_pixels=100_000)
        # 400x40 raised is 1120x112 = 125,440 px, over this budget: staged at
        # the resize's own target for the raised image, 980x84.
        assert lowered.staged_dimensions(400, 40) == (980, 84)
        assert smart_resize(84, 980, factor=28, min_pixels=3136, max_pixels=100_000) == (84, 980)

    def test_nothing_to_raise_is_none(self):
        assert M3_RULE.staged_dimensions(112, 400) is None
        assert M3_RULE.staged_dimensions(640, 480) is None
        assert M3_RULE.staged_dimensions(1, 300) is None
        assert M3_RULE.staged_dimensions(0, 10) is None

    def test_the_staged_size_is_bounded(self):
        # Whatever the input below the floor, the staged image is at most the
        # raised square of the floor or about 1.25x the budget (the shrunk
        # short side rounds up, so a little over that for the thinnest images).
        bound = max(M3.max_pixels * 1.3, M3.floor * M3.floor)
        for width in range(1, 112):
            for height in (1, 2, 3, 5, 16, 50, 111, 200, 400, 1000, 20000):
                for size in ((width, height), (height, width)):
                    staged = M3_RULE.staged_dimensions(*size)
                    if staged is not None:
                        assert staged[0] * staged[1] <= bound, (size, staged)


class TestFloorRule:
    @staticmethod
    def _config(model="org/model", model_type=None, mm_processor_kwargs=None):
        hf_config = (
            SimpleNamespace() if model_type is None else SimpleNamespace(model_type=model_type)
        )
        # The engine keeps the processor kwargs on its multimodal config.
        multimodal_config = SimpleNamespace(mm_processor_kwargs=mm_processor_kwargs)
        return SimpleNamespace(
            model=model, hf_config=hf_config, multimodal_config=multimodal_config
        )

    @pytest.mark.parametrize("entry", FAMILIES)
    def test_the_family_is_recognized_by_model_type_and_by_id(self, entry):
        for model_type in entry.model_types:
            assert short_side_floor(self._config(model_type=model_type)) == entry.floor
            assert short_side_floor(self._config(model_type=model_type.upper())) == entry.floor
        for words in entry.id_words:
            name = "Org/" + "-".join(word.upper() for word in words) + "-Instruct"
            assert short_side_floor(self._config(model=name)) == entry.floor
            assert short_side_floor(self._config(model="/models/" + "_".join(words))) == entry.floor
        assert floor_rule(self._config(model_type=entry.model_types[0])) == rule_of(entry)

    def test_minimax_m3_as_the_router_registry_recognizes_it(self):
        assert short_side_floor(self._config(model="MiniMaxAI/MiniMax-M3")) == 112
        assert short_side_floor(
            self._config(model="/srv/models/minimax-m3", model_type="other")
        ) == (112)
        assert short_side_floor(
            self._config(model="/srv/models/ckpt", model_type="minimax_m3_vl")
        ) == (112)

    def test_families_without_a_floor_get_none(self):
        assert (
            short_side_floor(self._config(model="Qwen/Qwen3-VL-8B-Instruct", model_type="qwen3_vl"))
            is None
        )
        assert (
            short_side_floor(self._config(model="MiniMaxAI/MiniMax-Text-01", model_type="minimax"))
            is None
        )
        assert short_side_floor(self._config(model="org/m3-mini")) is None
        assert floor_rule(SimpleNamespace()) is None
        assert (
            floor_rule(SimpleNamespace(model=None, hf_config=None, multimodal_config=None)) is None
        )

    def test_the_engines_processor_kwargs_are_read_from_its_multimodal_config(self):
        m3 = "MiniMaxAI/MiniMax-M3"
        assert floor_rule(self._config(model=m3, mm_processor_kwargs={"do_resize": False})) is None
        assert (
            short_side_floor(self._config(model=m3, mm_processor_kwargs={"do_resize": True})) == 112
        )
        assert short_side_floor(self._config(model=m3, mm_processor_kwargs={})) == 112
        assert short_side_floor(self._config(model=m3, mm_processor_kwargs=None)) == 112
        # The model config's own field is init-only on the engine and reads
        # as None there: a value set on it alone is not what the engine runs.
        config = self._config(model=m3)
        config.mm_processor_kwargs = {"do_resize": False}
        assert short_side_floor(config) == 112
        # The bounds the engine's processor runs with come from the same place.
        rule = floor_rule(self._config(model=m3, mm_processor_kwargs={"max_pixels": 100_000}))
        assert (rule.floor, rule.factor, rule.min_pixels, rule.max_pixels) == (
            112,
            28,
            3136,
            100_000,
        )
        rule = floor_rule(self._config(model=m3, mm_processor_kwargs={"min_pixels": 784}))
        assert (rule.min_pixels, rule.max_pixels) == (784, 451584)
        # An integral float is the whole number it spells (the engine takes it as such).
        rule = floor_rule(self._config(model=m3, mm_processor_kwargs={"max_pixels": 1003520.0}))
        assert rule.max_pixels == 1003520 and isinstance(rule.max_pixels, int)

    def test_an_unusable_pixel_bound_keeps_the_familys_and_says_so(self, caplog):
        m3 = "MiniMaxAI/MiniMax-M3"
        caplog.set_level(logging.WARNING)
        rule = floor_rule(
            self._config(model=m3, mm_processor_kwargs={"max_pixels": 0, "min_pixels": "x"})
        )
        assert (rule.min_pixels, rule.max_pixels) == (3136, 451584)
        assert "max_pixels=0 is not a positive whole number" in caplog.text
        assert "min_pixels='x' is not a positive whole number" in caplog.text
        caplog.clear()
        rule = floor_rule(self._config(model=m3, mm_processor_kwargs={"max_pixels": 2.5}))
        assert rule.max_pixels == 451584 and "max_pixels=2.5" in caplog.text
        caplog.clear()
        rule = floor_rule(self._config(model=m3, mm_processor_kwargs={"max_pixels": True}))
        assert rule.max_pixels == 451584 and "max_pixels=True" in caplog.text
        caplog.clear()
        # Absent: the family's own, and nothing to say.
        assert floor_rule(self._config(model=m3, mm_processor_kwargs={})).max_pixels == 451584
        assert caplog.text == ""


class TestRaiseToFloor:
    def test_no_rule_passes_the_object_through(self):
        image = _Image(16, 16)
        assert raise_to_floor(image, None) is image
        assert image.resized_with == []
        assert asyncio.run(raise_to_floor_off_loop(image, None)) is image

    @pytest.mark.parametrize("entry", FAMILIES)
    def test_tiny_images_are_resized_bicubic_to_the_floor(self, entry):
        rule = rule_of(entry)
        floor = entry.floor
        for width, height in [(16, 16), (1, 1)]:
            image = _Image(width, height)
            raised = raise_to_floor(image, rule)
            assert raised is not image and raised.size == (floor, floor)
            assert image.resized_with == [((floor, floor), floor_mod.BICUBIC)]
            assert raised.mode == "RGB"
            off_loop = asyncio.run(raise_to_floor_off_loop(_Image(width, height), rule))
            assert off_loop.size == (floor, floor)

    def test_images_at_the_floor_are_the_same_object(self):
        for width, height in [(112, 400), (400, 112), (640, 480)]:
            image = _Image(width, height)
            assert raise_to_floor(image, M3_RULE) is image
            assert asyncio.run(raise_to_floor_off_loop(image, M3_RULE)) is image
            assert image.resized_with == []

    def test_what_is_not_an_image_is_passed_through(self):
        waveform = ([0.25, -0.25], 16000)
        assert raise_to_floor(waveform, M3_RULE) is waveform
        assert asyncio.run(raise_to_floor_off_loop(waveform, M3_RULE)) is waveform
        frames = SimpleNamespace(size=None)
        assert raise_to_floor(frames, M3_RULE) is frames
        assert raise_to_floor(None, M3_RULE) is None

    def test_the_landscape_case_keeps_the_aspect(self):
        assert raise_to_floor(_Image(400, 20), M3_RULE).size == (2240, 112)
        assert raise_to_floor(_Image(20, 400), M3_RULE).size == (112, 2240)

    def test_over_the_budget_the_staged_size_is_what_is_resized_to(self):
        image = _Image(400, 10)
        assert raise_to_floor(image, M3_RULE).size == (4228, 84)
        assert image.resized_with == [((4228, 84), floor_mod.BICUBIC)]


class TestWithPillow:
    """The raise on real images, through Pillow's own resize."""

    @pytest.fixture(autouse=True)
    def _pil(self):
        self.Image = pytest.importorskip("PIL.Image")

    def _image(self, width, height):
        return self.Image.new("RGB", (width, height), (200, 30, 90))

    @pytest.mark.parametrize("entry", FAMILIES)
    def test_tiny_images_come_out_at_the_floor(self, entry):
        rule = rule_of(entry)
        floor = entry.floor
        for width, height in [(16, 16), (1, 1)]:
            raised = raise_to_floor(self._image(width, height), rule)
            assert raised.size == (floor, floor) and raised.mode == "RGB"
            # A flat image stays flat: the raise invents no pixels of its own.
            assert raised.getpixel((0, 0)) == (200, 30, 90)
            assert raised.getpixel((floor - 1, floor - 1)) == (200, 30, 90)

    def test_the_verifier_band_for_minimax_m3(self):
        for (width, height), expected in [
            ((400, 90), (498, 112)),
            ((400, 40), (1120, 112)),
            ((400, 20), (2240, 112)),
            ((20, 400), (112, 2240)),
            ((400, 10), (4228, 84)),
            ((2, 400), (54, 10625)),
        ]:
            assert raise_to_floor(self._image(width, height), M3_RULE).size == expected

    def test_images_at_the_floor_are_untouched(self):
        image = self._image(400, 112)
        assert raise_to_floor(image, M3_RULE) is image

    def test_off_the_loop_the_same_image_comes_back(self):
        raised = asyncio.run(raise_to_floor_off_loop(self._image(16, 16), M3_RULE))
        assert raised.size == (112, 112)


# --- Token-grid parity with the router's pipeline -------------------------


def cells(width, height, rule=None):
    """Visual tokens the engine's processor produces for an image the worker
    hands it, staged by `rule` first when given (one token per grid cell)."""
    staged = rule.staged_dimensions(width, height) if rule is not None else None
    if staged is not None:
        width, height = staged
    bounds = M3_RULE if rule is None else rule
    h_bar, w_bar = smart_resize(
        height,
        width,
        factor=bounds.factor,
        min_pixels=bounds.min_pixels,
        max_pixels=bounds.max_pixels,
    )
    return (h_bar // bounds.factor) * (w_bar // bounds.factor)


class TestTokenGridParity:
    def test_the_verifier_band_counts_the_pipelines_tokens_once_raised(self):
        # Without the raise the count falls and flattens as the short side
        # shrinks (the symptom); raised, it is the pipeline's, and grows.
        shorts = [90, 40, 20]
        assert [cells(400, short) for short in shorts] == [42, 14, 14]
        assert [cells(400, short, M3_RULE) for short in shorts] == [72, 160, 320]
        assert [cells(short, 400, M3_RULE) for short in shorts] == [72, 160, 320]

    def test_past_the_pixel_budget_the_engines_own_resize_lands_on_the_pipelines_grid(self):
        # The pipeline stages a raised image over its pixel budget at its own
        # resize's target: the engine's processor, given that, counts the same
        # as the pipeline (its tests pin 453 and 428).
        assert [cells(400, short, M3_RULE) for short in [20, 10, 5]] == [320, 453, 428]

    @pytest.mark.parametrize("entry", FAMILIES)
    def test_tiny_images_cost_the_pipelines_tokens(self, entry):
        rule = rule_of(entry)
        square = (entry.floor // entry.factor) ** 2
        assert cells(16, 16, rule) == square
        assert cells(1, 1, rule) == square
        assert cells(16, 16) < square
