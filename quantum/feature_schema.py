"""Shared feature contract for training and live prediction."""

FEATURE_NAMES = (
    "mean",
    "variance",
    "entropy",
    "lsb_ratio",
    "horizontal_difference",
    "vertical_difference",
)


def feature_vector(features):
    """Return live feature values in exactly the order used for training."""
    try:
        return [float(features[name]) for name in FEATURE_NAMES]
    except KeyError as exc:
        raise ValueError(f"Missing required model feature: {exc.args[0]}") from exc
