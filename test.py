from qiskit import QuantumCircuit
import numpy as np
normalized=[0.0213, 1.0, 0.0013, 0.0, 0.0021, 0.0021]
qc = QuantumCircuit(len(normalized))

for i, value in enumerate(normalized):
    qc.ry(value * np.pi, i)

print(qc)