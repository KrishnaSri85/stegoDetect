import importlib
import math
import os
from statistics import mean, pvariance


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def validate_image_file(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise ValueError("The uploaded file is empty or could not be saved.")

    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
    except ImportError:
        # Pillow is optional; extension and non-empty checks still protect the main flow.
        return
    except Exception as exc:
        raise ValueError("The image appears to be corrupted or unreadable.") from exc


def extract_features_safe(image_path):
    try:
        module = importlib.import_module("feature_extraction")
        extractor = getattr(module, "extract_features")
        return normalize_features(extractor(image_path))
    except ModuleNotFoundError:
        return fallback_extract_features(image_path)
    except AttributeError as exc:
        raise RuntimeError("Feature extraction module does not expose extract_features(image_path).") from exc
    except Exception as exc:
        raise RuntimeError("Feature extraction failed. Please check the image and try again.") from exc


def predict_safe(features):
    try:
        module = importlib.import_module("quantum")
        predictor = getattr(module, "predict")
        return normalize_prediction(predictor(features))
    except ModuleNotFoundError:
        return fallback_predict(features)
    except AttributeError as exc:
        raise RuntimeError("Quantum module does not expose predict(features).") from exc
    except Exception as exc:
        raise RuntimeError("Quantum model prediction failed. Please try again.") from exc


def normalize_features(features):
    if not isinstance(features, dict):
        raise RuntimeError("Feature extraction returned an invalid response.")
    return {str(key): round(float(value), 4) for key, value in features.items() if is_number(value)}


def normalize_prediction(prediction):
    if not isinstance(prediction, dict):
        raise RuntimeError("Quantum prediction returned an invalid response.")

    label = str(prediction.get("prediction", "Unknown")).title()
    confidence = max(0.0, min(100.0, float(prediction.get("confidence", 0.0))))
    return {"prediction": label, "confidence": confidence}


def readable_feature_summary(features):
    labels = {
        "entropy": "Entropy",
        "variance": "Variance",
        "mean": "Mean pixel value",
        "lsb_ratio": "LSB ratio",
    }
    return [
        {"key": key, "label": labels.get(key, key.replace("_", " ").title()), "value": value}
        for key, value in features.items()
    ]


def fallback_extract_features(image_path):
    try:
        from PIL import Image

        with Image.open(image_path).convert("L") as image:
            pixels = list(image.getdata())
    except ImportError:
        with open(image_path, "rb") as handle:
            pixels = list(handle.read())
    except Exception as exc:
        raise RuntimeError("Fallback feature extraction could not read the image.") from exc

    if not pixels:
        raise RuntimeError("Fallback feature extraction found no data.")

    frequencies = {}
    for value in pixels:
        frequencies[value] = frequencies.get(value, 0) + 1

    total = len(pixels)
    entropy = -sum((count / total) * math.log2(count / total) for count in frequencies.values())
    lsb_ratio = sum(value & 1 for value in pixels) / total

    return {
        "entropy": round(entropy, 4),
        "variance": round(pvariance(pixels), 4),
        "mean": round(mean(pixels), 4),
        "lsb_ratio": round(lsb_ratio, 4),
    }


def fallback_predict(features):
    entropy = float(features.get("entropy", 0.0))
    lsb_ratio = float(features.get("lsb_ratio", 0.5))
    variance = float(features.get("variance", 0.0))

    anomaly = min(1.0, abs(lsb_ratio - 0.5) * 3.2 + max(0.0, entropy - 7.4) * 0.18 + min(variance / 12000, 0.25))
    prediction = "Stego" if anomaly >= 0.46 else "Clean"
    confidence = 58 + anomaly * 38 if prediction == "Stego" else 92 - anomaly * 42

    return {"prediction": prediction, "confidence": round(confidence, 2)}


def is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
