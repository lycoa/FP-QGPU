import numpy as np

from fp_qgpu.gatter_operationen import cx, u_gate
from fp_qgpu.gatter_operationen_numba import (
    cx_numba_compatible,
    u_gate_numba_compatible,
)


def _random_state_tensor(num_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = rng.normal(size=2**num_qubits) + 1j * rng.normal(size=2**num_qubits)
    flat = flat / np.linalg.norm(flat)
    return flat.reshape([2] * num_qubits)


def _random_unitary_2x2(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, _ = np.linalg.qr(a)
    return q


def test_u_gate_numba_compatible_matches_einsum() -> None:
    num_qubits = 4
    state = _random_state_tensor(num_qubits, seed=1)

    for acting_on in range(num_qubits):
        u = _random_unitary_2x2(seed=100 + acting_on)

        expected = u_gate(num_qubits, acting_on, u, state)
        actual = u_gate_numba_compatible(num_qubits, acting_on, u, state)

        assert np.allclose(actual, expected, atol=1e-12)


def test_cx_numba_compatible_matches_einsum() -> None:
    num_qubits = 5
    state = _random_state_tensor(num_qubits, seed=2)

    for control in range(num_qubits):
        for target in range(num_qubits):
            if control == target:
                continue

            expected = cx(num_qubits, control, target, state)
            actual = cx_numba_compatible(num_qubits, control, target, state)

            assert np.allclose(actual, expected, atol=1e-12)
