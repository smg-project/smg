"""The short-side floor some model families apply to an image before their
own resize, for the worker-side media path.

The router's own media pipeline raises an image whose short side is below
the family's floor to that floor (same aspect ratio, bicubic) before the
family's processor sizes it, as the family's reference processor does. A
worker that fetches and processes media itself hands the engine's processor
the image as fetched, and that processor does not raise it: the same tiny
image then costs fewer visual tokens through the worker than through the
router, and a request's visual token count depends on which component
preprocessed it. This module keeps the rule for the worker-side path: the
families with a floor, the size the pipeline stages for an image below it
(the raised image within the family's pixel budget; over the budget, what
the pipeline stages so that the family's own resize lands where it would
for the raised image, at a bounded size) and the raise itself, applied to a
fetched image before the engine's processor sees it. Video frames are not
raised, as in the pipeline.

Engine-free: the image is whatever the fetcher returned (a PIL image),
resized through its own method, off the event loop.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

# PIL's bicubic filter (`Image.Resampling.BICUBIC`): the reference processors'
# resample, and the filter the pipeline raises with.
BICUBIC = 3
# Above this long-over-short ratio the families' own resize refuses the image;
# the raise leaves such an image to that refusal rather than staging a huge
# one first.
MAX_ASPECT_RATIO = 200.0
# How far over the pixel budget a shrunk staging image may land, so the
# family's own resize still scales it down (the pipeline's margin).
STAGING_OVERSHOOT = 1.25


@dataclasses.dataclass(frozen=True)
class ShortSideFloor:
    """One family's floor, the grid and pixel bounds its resize works with,
    and how its checkpoints are recognized: the config's `model_type`, or a
    model id (name or path) carrying all the words of one of the `id_words`
    tuples, case-insensitively."""

    family: str
    floor: int
    # The grid both resizes round to (patch size times merge size) and the
    # pixel bounds of the family's image resize, its defaults.
    factor: int
    min_pixels: int
    max_pixels: int
    model_types: tuple[str, ...]
    id_words: tuple[tuple[str, ...], ...]

    def matches(self, model_id: str, model_type: str | None) -> bool:
        if model_type is not None and model_type.lower() in self.model_types:
            return True
        lowered = model_id.lower()
        return any(all(word in lowered for word in words) for words in self.id_words)


# The families whose reference image processor raises a short side below a
# floor before its own resize: the rule the pipeline keeps per family
# (`crates/multimodal`, the family's vision processor), mirrored here for the
# worker-side path. MiniMax-M3: 112 px, four patch factors, recognized the
# way the pipeline's registry recognizes it.
SHORT_SIDE_FLOORS: tuple[ShortSideFloor, ...] = (
    ShortSideFloor(
        family="minimax_m3",
        floor=112,
        factor=28,
        min_pixels=4 * 28 * 28,
        max_pixels=576 * 28 * 28,
        model_types=("minimax_m3_vl",),
        id_words=(("minimax", "m3"),),
    ),
)


def smart_resize(
    height: int, width: int, *, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """The families' own resize (the reference processors' `smart_resize`),
    as the engine's processor runs it on what the worker hands it and as the
    pipeline runs it on what it staged: each side rounded to the grid, at
    least one grid cell, then scaled into the pixel bounds. Raises
    `ValueError` for the aspect ratio it does not take."""
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        # A budget of zero scales both sides to nothing, lifted to one cell below.
        beta = math.sqrt((height * width) / max_pixels) if max_pixels > 0 else math.inf
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return max(factor, h_bar), max(factor, w_bar)


def raised_dimensions(width: int, height: int, floor: int) -> tuple[int, int] | None:
    """The `(width, height)` with the short side raised to `floor` and the
    long side scaled with it, rounded half up as the pipeline rounds; None
    when the short side is at the floor already or zero, or the image is
    thinner than the families' resize takes (left to its refusal)."""
    short = min(width, height)
    if short <= 0 or short >= floor:
        return None
    long_side = max(width, height)
    if long_side / short > MAX_ASPECT_RATIO:
        return None
    raised_long = math.floor(long_side * floor / short + 0.5)
    return (floor, raised_long) if width <= height else (raised_long, floor)


def _aspect(width: int, height: int) -> float:
    return max(width, height) / max(1, min(width, height))


@dataclasses.dataclass(frozen=True)
class FloorRule:
    """The floor as it applies to one served model: the family's floor and
    grid, and the pixel bounds the engine's processor runs with."""

    floor: int
    factor: int
    min_pixels: int
    max_pixels: int

    def _resize(self, height: int, width: int) -> tuple[int, int]:
        return smart_resize(
            height,
            width,
            factor=self.factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

    def raised_dimensions(self, width: int, height: int) -> tuple[int, int] | None:
        return raised_dimensions(width, height, self.floor)

    def staged_dimensions(self, width: int, height: int) -> tuple[int, int] | None:
        """What to hand the engine's processor for an image below the floor,
        as the pipeline stages it; None when there is nothing to raise.

        Within the pixel budget that is the raised image itself. Over it, the
        family's own target for the raised image when that target holds (it
        fits the budget, its ratio is one the resize takes, and the resize
        hands it back unchanged): one resample, at most the budget, and the
        engine's resize leaves it. Otherwise, for a very thin image or a
        lowered budget, the raised image shrunk to just over the budget, which
        keeps the bound at about a percent of grid fidelity.
        """
        raised = self.raised_dimensions(width, height)
        if raised is None:
            return None
        raised_w, raised_h = raised
        budget = self.max_pixels
        if raised_w * raised_h <= budget:
            return raised
        target_h, target_w = self._resize(raised_h, raised_w)
        target_fits = (
            target_w * target_h <= budget
            and _aspect(target_w, target_h) <= MAX_ASPECT_RATIO
            and self._resize(target_h, target_w) == (target_h, target_w)
        )
        if target_fits:
            return target_w, target_h
        shrink = math.sqrt(raised_w * raised_h / (budget * STAGING_OVERSHOOT))

        # The short side rounds up and the long side down, so the ratio stays
        # inside the guard.
        def shrunk(side: int, short: bool) -> int:
            scaled = side / shrink
            return max(1, int(math.ceil(scaled) if short else math.floor(scaled)))

        if raised_w <= raised_h:
            w, h = shrunk(raised_w, True), shrunk(raised_h, False)
        else:
            w, h = shrunk(raised_w, False), shrunk(raised_h, True)
        if w * h > budget and _aspect(w, h) <= MAX_ASPECT_RATIO:
            return w, h
        return raised

    @staticmethod
    def _size(image: Any) -> tuple[int, int] | None:
        size = getattr(image, "size", None)
        if not isinstance(size, tuple) or len(size) != 2:
            return None
        return int(size[0]), int(size[1])

    def needs_raise(self, image: Any) -> bool:
        """Whether `apply` would resize `image`."""
        size = self._size(image)
        return size is not None and self.staged_dimensions(*size) is not None

    def apply(self, image: Any) -> Any:
        """`image` (what the fetcher returned: `.size` and `.resize`, a PIL
        image) at the staged size, bicubic; the image itself when there is
        nothing to raise or the object is not an image."""
        size = self._size(image)
        staged = self.staged_dimensions(*size) if size is not None else None
        if staged is None:
            return image
        return image.resize(staged, BICUBIC)


def _family(model_config: Any) -> ShortSideFloor | None:
    model_type = getattr(getattr(model_config, "hf_config", None), "model_type", None)
    model_id = str(getattr(model_config, "model", "") or "")
    for entry in SHORT_SIDE_FLOORS:
        if entry.matches(model_id, model_type if isinstance(model_type, str) else None):
            return entry
    return None


def _processor_kwargs(model_config: Any) -> Mapping[str, Any]:
    """The engine's `mm_processor_kwargs`, read where the engine keeps them:
    its multimodal config (the model config's own field is an init-only
    argument that is moved there at construction)."""
    kwargs = getattr(getattr(model_config, "multimodal_config", None), "mm_processor_kwargs", None)
    return kwargs if isinstance(kwargs, Mapping) else {}


def _pixel_bound(kwargs: Mapping[str, Any], name: str, default: int) -> int:
    """`kwargs[name]` as a positive whole number (an integral float counts),
    else the family's `default`, with a warning when a value is present and
    unusable: the engine's processor runs with the value as given, so the
    staging would then work against a budget the engine is not using."""
    value = kwargs.get(name)
    if value is None:
        return default
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    logger.warning(
        "mm_processor_kwargs %s=%r is not a positive whole number: the short-side raise is "
        "staged against the model family's default of %d, while the engine's processor "
        "runs with the value as given",
        name,
        value,
        default,
    )
    return default


def floor_rule(model_config: Any) -> FloorRule | None:
    """The rule for the served model, or None: no family with a floor, or the
    engine's processor kwargs turn resizing off (`do_resize: false`), when the
    pipeline raises nothing either. `min_pixels` / `max_pixels` in those
    kwargs set the bounds, as the pipeline layers them over the family's."""
    entry = _family(model_config)
    if entry is None:
        return None
    kwargs = _processor_kwargs(model_config)
    if kwargs.get("do_resize") is False:
        return None
    return FloorRule(
        floor=entry.floor,
        factor=entry.factor,
        min_pixels=_pixel_bound(kwargs, "min_pixels", entry.min_pixels),
        max_pixels=_pixel_bound(kwargs, "max_pixels", entry.max_pixels),
    )


def short_side_floor(model_config: Any) -> int | None:
    """The floor for the served model, or None (see `floor_rule`)."""
    rule = floor_rule(model_config)
    return None if rule is None else rule.floor


def raise_to_floor(image: Any, rule: FloorRule | None) -> Any:
    """`image` at the size the rule stages, or the image itself (no rule,
    nothing to raise, not an image)."""
    return image if rule is None else rule.apply(image)


async def raise_to_floor_off_loop(image: Any, rule: FloorRule | None) -> Any:
    """`raise_to_floor` with the resample in a worker thread, off the event
    loop that streams every request's tokens (the fetcher decodes there too);
    an image with nothing to raise comes straight back."""
    if rule is None or not rule.needs_raise(image):
        return image
    return await asyncio.to_thread(rule.apply, image)
