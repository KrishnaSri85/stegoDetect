import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

# --- QUANTUM PIPELINE IMPORTS ---
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

# =====================================================================
# 1. DATA LOADING & PREPROCESSING
# =====================================================================

# Read both CSV files
clean_df = pd.read_csv('original_features.csv')
stego_df = pd.read_csv('stego_features.csv')

# Use only the live steganalysis features — never include Label (target leakage).
FEATURE_COLUMNS = [
    "Mean",
    "Variance",
    "Entropy",
    "LSB Ratio",
    "Horizontal Difference",
    "Vertical Difference",
]
X_clean = clean_df[FEATURE_COLUMNS]
X_stego = stego_df[FEATURE_COLUMNS]

# Create explicit target labels (0 = Clean, 1 = Stego)
y_clean = np.zeros(X_clean.shape[0])
y_stego = np.ones(X_stego.shape[0])

# Combine datasets vertically
X_combined = pd.concat([X_clean, X_stego], axis=0, ignore_index=True)
y_combined = np.concatenate([y_clean, y_stego])

# Standardize data distributions
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_combined)

# Reduce dimensionality before mapping to quantum circuits
N_QUBITS = 6  # Number of features / qubits mapped to the engine
pca = PCA(n_components=N_QUBITS, random_state=42)
X_reduced = pca.fit_transform(X_standardized)
print(f"[PCA] Retained variance with {N_QUBITS} components: "
      f"{pca.explained_variance_ratio_.sum() * 100:.2f}%")

# Rescale features into a bounded range [0, 2*pi] for quantum angle encoding
angle_scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
X_normalized = angle_scaler.fit_transform(X_reduced)

# Train/Test Split with stratified labels
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

# =====================================================================
# 2. CLASSICAL BASELINE SANITY CHECK
# =====================================================================
baseline = SVC(kernel='rbf', random_state=42)
baseline.fit(X_train, y_train)
baseline_acc = baseline.score(X_test, y_test)
print(f"\n[Classical Baseline] SVM accuracy on same features: {baseline_acc * 100:.2f}%")

# =====================================================================
# 3. QUANTUM STAGE & CIRCUIT DEFINITION
# =====================================================================
num_features = X_train.shape[1]
print(f"\n[Quantum Engine] Using {num_features} qubits to process data.")

if num_features > 12:
    print(f"[Warning] {num_features} qubits may be slow/impractical on a simulator. "
          f"Consider PCA/feature selection to reduce dimensionality before this step.")

# Quantum Encoding (Feature Map)
feature_map = zz_feature_map(feature_dimension=num_features, reps=2, entanglement='linear')

# Quantum Circuit Ansatz (Trainable Layers)
ansatz = real_amplitudes(num_qubits=num_features, reps=3, entanglement='full')

# Define Sampler Primitive
sampler = StatevectorSampler()

# Initialize Variational Quantum Classifier (VQC)
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=250),
    sampler=sampler
)

# =====================================================================
# 4. CIRCUIT VISUALIZATION EXPORT
# =====================================================================
print("\n--- Generating Quantum Circuit Layout ---")
full_circuit = feature_map.compose(ansatz)
decomposed_circuit = full_circuit.decompose()
print(decomposed_circuit.draw(output='text'))

# Save text structure
decomposed_circuit.draw(output='text', filename='circuit_diagram.txt')
print("[Saved] Circuit text diagram written to circuit_diagram.txt")

# Save graphical picture
try:
    fig = decomposed_circuit.draw(output='mpl')
    fig.savefig('circuit_diagram.png', dpi=300, bbox_inches='tight')
    print("[Saved] Circuit image written to circuit_diagram.png")
except Exception as e:
    print(f"[Warning] Could not save circuit image (is matplotlib/pylatexenc installed?): {e}")

# =====================================================================
# 5. TRAINING AND EVALUATION
# =====================================================================
print("\nTraining the Variational Quantum Classifier (VQC)...")
vqc.fit(X_train, y_train)

print("Predicting stego signatures on test set...")
y_pred = vqc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Quantum Steganalysis Results ---")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Clean', 'Stego']))

# =====================================================================
# 6. SAVE QUANTUM MODEL AND PREPROCESSING PIPELINE FOR EXPORT
# =====================================================================
print("\n--- Saving Model Components for Deployment ---")

# 1. Save the trained VQC model
# NOTE: Do NOT use raw pickle.dump() on a VQC object. It wraps Qiskit
# primitives/samplers internally, which can contain objects that don't
# reliably survive generic pickling (e.g. C-extension bindings). Use the
# model's own .save() method, which qiskit_machine_learning provides
# specifically to serialize these objects correctly. Load it back with
# VQC.load('vqc_steganalysis_model.model').
model_filename = 'vqc_steganalysis_model.model'
vqc.save(model_filename)
print(f"[Saved] Trained VQC model saved to '{model_filename}'")

# 2. Save the scaling and PCA pipeline
# joblib is the sklearn-recommended format for objects holding numpy arrays
# (more efficient than raw pickle for this kind of object).
pipeline_filename = 'preprocessing_pipeline.pkl'
preprocessing_pipeline = {
    'scaler': scaler,
    'pca': pca,
    'angle_scaler': angle_scaler,
    'feature_columns': list(FEATURE_COLUMNS),  # order must match live Flask feature mapping
}
joblib.dump(preprocessing_pipeline, pipeline_filename)
print(f"[Saved] Preprocessing pipeline saved to '{pipeline_filename}'")