from typing import Dict, List
from .data import ATTRIBUTE_NAMES


def _level(value: float) -> str:
    if value >= 0.75:
        return "strong"
    if value >= 0.55:
        return "acceptable"
    if value >= 0.35:
        return "weak"
    return "poor"


def generate_feedback(score: float, attributes: Dict[str, float]) -> Dict[str, str]:
    sorted_attrs = sorted(attributes.items(), key=lambda x: x[1], reverse=True)
    best = sorted_attrs[:2]
    worst = sorted_attrs[-2:]
    strengths = [f"The {name} dimension is {_level(value)} ({value:.2f}), indicating useful cinematographic control." for name, value in best]
    hints = {
        "composition": "reconsider framing, subject placement, leading lines, and figure-ground separation",
        "lighting": "use clearer key light direction, contrast, and shadow structure",
        "color": "refine palette consistency, color temperature, and harmony",
        "motion": "clarify subject movement, camera direction, and temporal pacing",
        "narrative": "make the shot intention, emotional tone, and continuity more explicit",
    }
    improvements = [f"The {name} dimension is {_level(value)} ({value:.2f}); {hints[name]}." for name, value in worst]
    if score < 0.4:
        suggestion = "Start with a simple single-subject storyboard exercise, apply rule-of-thirds framing, and use one clear lighting direction before adding complex narrative transitions."
    elif score > 0.75:
        suggestion = "The storyboard is already strong; refine subtle transitions between panels and strengthen the relation between visual rhythm and narrative intent."
    else:
        suggestion = "Revise the storyboard by improving the two weakest dimensions first, then check whether the visual evidence supports the intended narrative function."
    return {"strengths": " ".join(strengths), "areas_for_improvement": " ".join(improvements), "pedagogical_suggestion": suggestion}


def attrs_to_dict(values: List[float]) -> Dict[str, float]:
    return {name: float(v) for name, v in zip(ATTRIBUTE_NAMES, values)}
