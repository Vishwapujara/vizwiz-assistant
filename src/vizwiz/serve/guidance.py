"""Quality issue -> fixed retake guidance (PROJECT_PLAN 6.5).

Fixed text, not model-generated: the advice a blind user hears when a photo cannot
be used must be predictable and reviewed. Every quality label has an entry; a test
checks this against ``configs/data.yaml``.
"""

from __future__ import annotations

from collections.abc import Iterable

# Most important first: when a photo has several issues, the first ones are spoken.
TIPS: dict[str, str] = {
    "unrecognizable": "I can't make out what the photo shows.",
    "obstruction": "Something, maybe a finger, is covering the lens or the object. Move it away.",
    "dark": "The photo is too dark. Turn on a light or move near a window.",
    "bright": "There is glare or too much light. Tilt the object or move out of direct light.",
    "blur": "The photo is blurry. Hold the phone steady, elbows on a table if you can.",
    "framing": "The object may be cut off. Move the phone back a little and centre the object.",
    "rotation": "The photo is tilted or sideways. Hold the phone upright, facing the object.",
}
GENERIC_RETAKE = "Try again with the object centred, about 30 cm away, in good light."
UNUSABLE = "I can't see enough in this photo to help."
CANNOT_ANSWER = "I can't answer that from this photo."
NOT_VISIBLE = (
    "The answer may not be in view. Try showing the part with the information, "
    "such as the label, screen or barcode."
)


def tips_for(issues: Iterable[str], limit: int, skip: Iterable[str] = ()) -> list[str]:
    """Guidance lines for the detected issues, most important first."""
    found, skipped = set(issues), set(skip)
    return [text for label, text in TIPS.items() if label in found and label not in skipped][:limit]


def retake_message(issues: Iterable[str], limit: int) -> tuple[str, list[str]]:
    """Message when the photo is unusable whatever the question."""
    tips = tips_for(issues, limit, skip=("unrecognizable",)) or [GENERIC_RETAKE]
    return " ".join([UNUSABLE, *tips]), tips


def decline_message(issues: Iterable[str], limit: int) -> tuple[str, list[str]]:
    """Message when the photo is usable but the question cannot be answered from it."""
    tips = tips_for(issues, limit) or [NOT_VISIBLE]
    if tips == [TIPS["unrecognizable"]]:  # says what is wrong, not yet how to fix it
        tips.append(GENERIC_RETAKE)
    return " ".join([CANNOT_ANSWER, *tips]), tips
