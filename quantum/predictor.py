"""Runtime adapter between the Flask feature dictionary and the trained VQC."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PIPELINE_PATH = ROOT / "preprocessing_pipeline.pkl"
MODEL_PATH = ROOT / "vqc_steganalysis_model.model"

# Training CSVs / saved pipeline use Title Case names; live extraction uses snake_case.
PIPELINE_TO_LIVE = {
    "Mean": "mean",
    "Variance": "variance",
    "Entropy": "entropy",
    "LSB Ratio": "lsb_ratio",
    "Horizontal Difference": "horizontal_difference",
    "Vertical Difference": "vertical_difference",
}

_pipeline = None
_model = None


def _load_assets():
    global _pipeline, _model
    if _pipeline is not None and _model is not None:
        return _pipeline, _model

    if not PIPELINE_PATH.exists():
        raise RuntimeError(
            f"Missing preprocessing pipeline at {PIPELINE_PATH}. "
            "Train with `python feature.py` first."
        )
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Missing VQC model at {MODEL_PATH}. "
            "Train with `python feature.py` (saves .model via VQC.save). "
            "Do not use the empty vqc_steganalysis_model.pkl file."
        )

    try:
        import joblib
        from qiskit.primitives import StatevectorSampler
        from qiskit_machine_learning.algorithms import VQC
    except ImportError as exc:
        raise RuntimeError("Quantum dependencies are missing. Install requirements.txt.") from exc

    try:
        _pipeline = joblib.load(PIPELINE_PATH)
        _model = VQC.load(str(MODEL_PATH))
        _model.sampler = StatevectorSampler()
    except Exception as exc:
        raise RuntimeError("Could not load quantum model assets.") from exc

    return _pipeline, _model


def _ordered_feature_vector(features, pipeline):
    """Build a 1xN matrix using the exact column order the scaler was fitted on."""
    import pandas as pd

    columns = [
        name
        for name in pipeline.get("feature_columns", [])
        if name in PIPELINE_TO_LIVE
    ]
    if not columns:
        columns = list(PIPELINE_TO_LIVE.keys())

    try:
        values = {name: float(features[PIPELINE_TO_LIVE[name]]) for name in columns}
    except KeyError as exc:
        raise ValueError(f"Missing required model feature: {exc.args[0]}") from exc

    return pd.DataFrame([values], columns=columns), columns


def predict(features):
    """Classify extracted image features with the persisted quantum model."""
    pipeline, model = _load_assets()

    try:
        matrix, _ = _ordered_feature_vector(features, pipeline)
        scaled = pipeline["scaler"].transform(matrix)
        reduced = pipeline["pca"].transform(scaled)
        encoded = pipeline["angle_scaler"].transform(reduced)

        label = int(np.asarray(model.predict(encoded)).reshape(-1)[0])
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(encoded)).reshape(-1)
            confidence = float(np.max(probabilities) * 100.0)
        else:
            confidence = 100.0
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError("Could not run the trained quantum model.") from exc

    return {
        "prediction": "stego" if label == 1 else "clean",
        "confidence": round(confidence, 2),
    }
