"""Train and persist the VQC used by the Flask application.

Run from the repository root: ``python quantum/trainin_model.py``.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from quantum.feature_schema import FEATURE_NAMES


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "quantum" / "artifacts"


def load_data():
    clean = pd.read_csv(DATA_DIR / "original_features.csv")
    stego = pd.read_csv(DATA_DIR / "stego_features.csv")
    column_map = {
        "mean": "Mean", "variance": "Variance", "entropy": "Entropy",
        "lsb_ratio": "LSB Ratio", "horizontal_difference": "Horizontal Difference",
        "vertical_difference": "Vertical Difference",
    }
    ordered_columns = [column_map[name] for name in FEATURE_NAMES]
    features = pd.concat([clean[ordered_columns], stego[ordered_columns]], ignore_index=True)
    labels = np.concatenate([np.zeros(len(clean), dtype=int), np.ones(len(stego), dtype=int)])
    return features.to_numpy(dtype=float), labels


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # The same fitted transformation is saved and reused by predictor.py.
    preprocessor = Pipeline([
        ("standardize", StandardScaler()),
        ("pca", PCA(n_components=min(6, X_train.shape[1]), random_state=42)),
        ("angles", MinMaxScaler(feature_range=(0, 2 * np.pi))),
    ])
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    num_features = X_train_encoded.shape[1]
    model = VQC(
        feature_map=zz_feature_map(feature_dimension=num_features, reps=2, entanglement="linear"),
        ansatz=real_amplitudes(num_qubits=num_features, reps=3, entanglement="full"),
        optimizer=COBYLA(maxiter=250),
        sampler=StatevectorSampler(),
    )
    print(f"Training VQC with {num_features} encoded features...")
    model.fit(X_train_encoded, y_train)

    predictions = model.predict(X_test_encoded)
    print(f"Test accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(classification_report(y_test, predictions, target_names=["Clean", "Stego"]))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.joblib")
    model.save(str(ARTIFACT_DIR / "vqc.model"))
    print(f"Saved model artifacts to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
