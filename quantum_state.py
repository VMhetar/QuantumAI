import numpy as np

class QuantumState:
    def __init__(self, state: np.ndarray):
        state = np.asarray(state, dtype=np.complex128)

        # Normalization check
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0):
            raise ValueError("Quantum state must be normalized")

        self.state = state
        self.dim = len(state)
        self.n_qubits = int(np.log2(self.dim))

        if 2**self.n_qubits != self.dim:
            raise ValueError("State vector length must be power of 2")

    @classmethod
    def zero(cls, n_qubits):
        state = np.zeros(2**n_qubits, dtype=np.complex128)
        state[0] = 1.0
        return cls(state)
