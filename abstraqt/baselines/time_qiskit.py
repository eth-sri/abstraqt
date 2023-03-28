import time

from qiskit.circuit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit import transpile, QiskitError

from abstraqt.experiments.generate_helper import random_operation_on_qubits


def run(n_shots: int):
    start = time.time()
    circuit = QuantumCircuit(20)
    random_operation_on_qubits(circuit, ['h', 's', 'cx'], list(range(20)))

    extended_stabilizer_simulator = AerSimulator(method='extended_stabilizer')
    transpiled = transpile(circuit, extended_stabilizer_simulator)

    job = extended_stabilizer_simulator.run(transpiled, shots=n_shots)
    result = job.result()

    duration = time.time() - start
    print('Duration for', n_shots, 'shots:', duration)

    return result

run(1)
run(1000)

run(1)
run(1000)
