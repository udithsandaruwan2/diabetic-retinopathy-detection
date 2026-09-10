"""Diabetic Retinopathy stage detection package."""

__version__ = "0.1.0"

CLASS_NAMES = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative_DR",
}

CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
# Common folder aliases in Kaggle packs
LABEL_ALIASES = {
    "no_dr": 0,
    "nodr": 0,
    "healthy": 0,
    "normal": 0,
    "mild": 1,
    "mild_dr": 1,
    "moderate": 2,
    "moderate_dr": 2,
    "severe": 3,
    "severe_dr": 3,
    "proliferate_dr": 4,
    "proliferative_dr": 4,
    "proliferative": 4,
    "proliferate": 4,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
}
