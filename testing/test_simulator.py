from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
from fp_qgpu.simulator_mock import simulator_mock
from qiskit.circuit.random import random_circuit


def create_test_circuit(n=4):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n):
        if i < n - 1:
            qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def test_simulator():
    qc = create_test_circuit()

    # Mock simulator
    test_values = {"qc": qc, "shots": 1024, "seed": 20}
    result_mock = simulator_mock(**test_values)

    # Aer simulator
    simulator = AerSimulator(seed_simulator=test_values["seed"])
    circ = transpile(qc, simulator)

    result = simulator.run(circ, shots=test_values["shots"]).result()
    result_Aer = result.get_counts(circ)

    assert result_mock == result_Aer


def test_random_circuit():
    n = 4
    depth = 10
    qc = random_circuit(n, depth, measure=True)

    # Mock simulator
    test_values = {"qc": qc, "shots": 1024, "seed": 20}
    result_mock = simulator_mock(**test_values)

    # Aer simulator
    simulator = AerSimulator(seed_simulator=test_values["seed"])
    circ = transpile(qc, simulator)

    result = simulator.run(circ, shots=test_values["shots"]).result()
    result_Aer = result.get_counts(circ)

    assert result_mock == result_Aer
