"""Engine-free helpers for `GenerateRequest.media_refs` (worker-side media processing)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

from smg_grpc_proto.generated import common_pb2

# Modalities the worker can process from a reference. Audio is excluded until a
# router spec opts in and an e2e model exists.
_MODALITY_NAMES: dict[int, str] = {
    common_pb2.IMAGE: "image",
    common_pb2.VIDEO: "video",
}

BASE_SCHEMES = ("http", "https", "data")


@dataclass(frozen=True)
class MediaRefItem:
    modality: str
    url: str


def parse_media_refs(refs) -> list[MediaRefItem]:
    """Validate a `MediaRefs` proto and return its items in prompt order."""
    items: list[MediaRefItem] = []
    for index, ref in enumerate(refs.items):
        modality = _MODALITY_NAMES.get(ref.modality)
        if modality is None:
            name = common_pb2.Modality.Name(ref.modality)
            raise ValueError(f"media_refs[{index}]: unsupported modality {name}")
        if not ref.url:
            raise ValueError(f"media_refs[{index}]: empty url")
        items.append(MediaRefItem(modality=modality, url=ref.url))
    if not items:
        raise ValueError("media_refs is set but carries no items")
    return items


def url_scheme(url: str) -> str:
    """Lower-cased URL scheme; `data` for data URLs, empty for bare paths."""
    return urlsplit(url).scheme.lower()


def advertised_schemes(allowed_local_media_path: str | None) -> str:
    """Comma list of schemes this worker will fetch (`file` needs a local media path)."""
    schemes = list(BASE_SCHEMES)
    if allowed_local_media_path:
        schemes.append("file")
    return ",".join(schemes)


def parse_scheme_list(value: str) -> set[str]:
    return {scheme.strip().lower() for scheme in value.split(",") if scheme.strip()}


def validate_schemes(items: Iterable[MediaRefItem], accepted: set[str]) -> None:
    for index, item in enumerate(items):
        scheme = url_scheme(item.url)
        if scheme not in accepted:
            raise ValueError(
                f"media_refs[{index}]: url scheme '{scheme or 'none'}' is not accepted by this "
                f"worker (accepted: {','.join(sorted(accepted))}); file:// requires "
                "--allowed-local-media-path"
            )


def group_urls_by_modality(items: Iterable[MediaRefItem]) -> dict[str, list[str]]:
    """Group URLs per modality, preserving prompt order within each modality."""
    grouped: dict[str, list[str]] = {}
    for item in items:
        grouped.setdefault(item.modality, []).append(item.url)
    return grouped
