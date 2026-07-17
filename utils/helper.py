"""Flask-facing wrappers around feature extraction and quantum prediction."""

from PIL import Image

from feature_extraction import extract_features
from quantum import predict as quantum_predict


def extract_features_safe(image_path):
    try:
        return extract_features(image_path)
    except Exception as exc:
        raise ValueError(f"Feature extraction failed: {exc}") from exc


def predict_safe(features):
    try:
        result = quantum_predict(features)
    except Exception as exc:
        raise RuntimeError(f"Quantum model prediction failed: {exc}") from exc

    prediction = str(result.get("prediction", "clean")).strip().lower()
    if prediction not in {"clean", "stego"}:
        prediction = "stego" if "stego" in prediction else "clean"

    confidence = float(result.get("confidence", 0.0))
    # UI expects a percentage in [0, 100].
    if confidence <= 1.0:
        confidence *= 100.0

    return {
        "prediction": prediction,
        "confidence": round(max(0.0, min(confidence, 100.0)), 2),
    }


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def validate_image_file(image_path):
    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("The uploaded image file is corrupted or could not be opened.") from exc


def readable_feature_summary(features):
    return {key: f"{value:.4f}" for key, value in features.items()}
