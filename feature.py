import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Read both CSV files
clean_df = pd.read_csv('original_features.csv')
stego_df = pd.read_csv('stego_features.csv')

# 2. Filter out non-numeric columns
X_clean = clean_df.select_dtypes(include=['number'])
X_stego = stego_df.select_dtypes(include=['number'])

# 3. Create explicit target labels (0 = Clean, 1 = Stego)
y_clean = np.zeros(X_clean.shape[0])
y_stego = np.ones(X_stego.shape[0])

# 4. Combine datasets vertically
X_combined = pd.concat([X_clean, X_stego], axis=0, ignore_index=True)
y_combined = np.concatenate([y_clean, y_stego])

# 5. Normalize Features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_combined)

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

# --- QUANTUM PIPELINE (UPDATED FOR MODERN QISKIT) ---
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler  # Handles backend routing

num_features = X_train.shape[1]
print(f"\n[Quantum Engine] Using {num_features} qubits to process data.")

# Optional but recommended: warn if the qubit count is too large to simulate comfortably
if num_features > 12:
    print(f"[Warning] {num_features} qubits may be slow/impractical on a simulator. "
          f"Consider PCA/feature selection to reduce dimensionality before this step.")

# Step 7: Quantum Encoding
# NOTE: the functional zz_feature_map takes `feature_dimension`, NOT `num_qubits`.
feature_map = zz_feature_map(feature_dimension=num_features, reps=2, entanglement='linear')

# Step 8: Quantum Circuit Ansatz
# NOTE: real_amplitudes DOES use `num_qubits` as its parameter name.
ansatz = real_amplitudes(num_qubits=num_features, reps=2)

# Step 9: Define Sampler Primitive
sampler = StatevectorSampler()

# Step 10: Initialize VQC
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=40),
    sampler=sampler
)

# --- DRAW THE CIRCUIT ---
# NOTE: VQC has no `.circuit` attribute. Build the combined circuit directly
# from feature_map + ansatz to inspect the layout before training.
print("\n--- Generating Quantum Circuit Layout ---")
full_circuit = feature_map.compose(ansatz)
decomposed_circuit = full_circuit.decompose()
print(decomposed_circuit.draw(output='text'))

# Save the text version of the circuit to a file
decomposed_circuit.draw(output='text', filename='circuit_diagram.txt')
print("[Saved] Circuit text diagram written to circuit_diagram.txt")

# Save an image version of the circuit (requires matplotlib and pylatexenc)
try:
    fig = decomposed_circuit.draw(output='mpl')
    fig.savefig('circuit_diagram.png', dpi=300, bbox_inches='tight')
    print("[Saved] Circuit image written to circuit_diagram.png")
except Exception as e:
    print(f"[Warning] Could not save circuit image (is matplotlib/pylatexenc installed?): {e}")

# Step 11: Train the Quantum Classifier
print("\nTraining the Variational Quantum Classifier (VQC)...")
vqc.fit(X_train, y_train)

# Step 12: Predict and Evaluate
print("Predicting stego signatures on test set...")
y_pred = vqc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Quantum Steganalysis Results ---")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Clean', 'Stego']))