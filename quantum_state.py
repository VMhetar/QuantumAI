import numpy as np

class QuantumState:
    def __init__(self):
        self.state = None
        self.n_qubits = None
        self.dim = None

    def state_vector(self, n, amplitudes):
        """
        n : number of qubits
        amplitudes : iterable of complex amplitudes (length = 2^n)
        """
        self.n_qubits = n
        self.dim = 2 ** n

        psi = np.asarray(amplitudes, dtype=np.complex128)

        if psi.shape[0] != self.dim:
            raise ValueError("State vector size must be 2^n")

        # Normalize
        norm = np.linalg.norm(psi)
        if not np.isclose(norm, 1.0):
            psi = psi / norm

        self.state = psi
        return psi
