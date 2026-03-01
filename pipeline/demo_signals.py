from __future__ import annotations

from typing import Sequence


def demo_value_for_item(*, tags: Sequence[str], day_index: int) -> int:
    """
    Deterministic demo signal generator used by replay and policy data collection.

    The pattern intentionally creates leading mood/sleep/stress drift followed by later GI drift,
    so unlock/retest/drift behavior can be exercised in a short synthetic run.
    """
    value = 1
    tag_set = set(str(t) for t in tags)

    if "mood" in tag_set:
        if 15 <= day_index <= 21:
            value = 3
        elif 22 <= day_index <= 25:
            value = 2

    if "sleep" in tag_set and 8 <= day_index <= 12:
        value = max(value, 3)

    if "stress" in tag_set and 15 <= day_index <= 21:
        value = max(value, 3)

    if "gi" in tag_set and 22 <= day_index <= 29:
        value = max(value, 3)

    if "resilience" in tag_set and 15 <= day_index <= 21:
        value = 1

    return max(0, min(4, int(value)))
