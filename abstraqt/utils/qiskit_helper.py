from typing import Union, Optional

from qiskit.quantum_info import Operator
from qiskit.circuit import QuantumCircuit


class MemoryError(Exception):
    ...


def circuits_equivalent(circuit1: Union[QuantumCircuit, str], circuit2: Union[QuantumCircuit, str], max_memory_bytes: Optional[float] = None):
    if isinstance(circuit1, str):
        circuit1 = QuantumCircuit.from_qasm_file(circuit1)
    if isinstance(circuit2, str):
        circuit2 = QuantumCircuit.from_qasm_file(circuit2)
    
    if max_memory_bytes is not None:
        size1 = (2**circuit1.num_qubits)**2 * 128 / 8
        size2 = (2**circuit2.num_qubits)**2 * 128 / 8
        size = max(size1, size2)
        if size > max_memory_bytes:
            raise MemoryError(f'Would need {size} bytes for simulation')

    op1 = Operator(circuit1)
    op2 = Operator(circuit2)

    ok = op1.equiv(op2)
    return ok
