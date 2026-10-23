"""Frozen end-to-end ANPR integration surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anpr_engine.integration.anpr_v1 import AnprV1Pipeline, FrozenBundle
    from anpr_engine.integration.review_store import BatchReviewStore


def __getattr__(name: str) -> Any:
    if name in {"AnprV1Pipeline", "FrozenBundle"}:
        from anpr_engine.integration import anpr_v1

        return getattr(anpr_v1, name)
    if name == "BatchReviewStore":
        from anpr_engine.integration.review_store import BatchReviewStore

        return BatchReviewStore
    raise AttributeError(name)


__all__ = ["AnprV1Pipeline", "BatchReviewStore", "FrozenBundle"]
