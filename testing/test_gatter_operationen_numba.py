import numpy as np
from qiskit import QuantumCircuit, transpile

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


def test_cx_numba_compatible_matches_original_on_transpiled_random_cx_only_circuit() -> None:
    rng = np.random.default_rng(123)
    num_qubits = 5
    depth = 18

    # Build a random CX-only circuit in Qiskit and transpile it.
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        control = int(rng.integers(0, num_qubits))
        target = int(rng.integers(0, num_qubits - 1))
        if target >= control:
            target += 1
        qc.cx(control, target)

    transpiled_qc = transpile(qc, basis_gates=["cx"], optimization_level=0)
    gate_names = [instruction.operation.name for instruction in transpiled_qc.data]
    assert all(name == "cx" for name in gate_names)

    psi_flat = np.zeros(2**num_qubits, dtype=complex)
    psi_flat[0] = 1.0 + 0.0j

    psi_original = psi_flat.reshape([2] * num_qubits)
    psi_numba = psi_flat.reshape([2] * num_qubits)

    for instruction in transpiled_qc.data:
        control_q = transpiled_qc.find_bit(instruction.qubits[0]).index
        target_q = transpiled_qc.find_bit(instruction.qubits[1]).index
        control_axis = num_qubits - 1 - control_q
        target_axis = num_qubits - 1 - target_q

        psi_original = cx(num_qubits, control_axis, target_axis, psi_original)
        psi_numba = cx_numba_compatible(
            num_qubits, control_axis, target_axis, psi_numba
        )

    original_result = psi_original.reshape(2**num_qubits)
    numba_result = psi_numba.reshape(2**num_qubits)

    print("Original CX implementation result:")
    print(original_result)
    print("Numba-compatible CX implementation result:")
    print(numba_result)

    assert np.allclose(numba_result, original_result, atol=1e-12)
