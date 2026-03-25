import numpy as np


def get_pauli_matricies() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    return pauli_x, pauli_y, pauli_z


def pauli_x() -> np.ndarray:
    return get_pauli_matricies()[0]


def pauli_y() -> np.ndarray:
    return get_pauli_matricies()[1]


def pauli_z() -> np.ndarray:
    return get_pauli_matricies()[2]
