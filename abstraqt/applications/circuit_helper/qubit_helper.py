from typing import Union, Dict

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit, DAGNode


def get_qubit_indices_from_circuit(circuit: Union[QuantumCircuit, DAGCircuit]):
    d = {gate: i for i, gate in enumerate(circuit.qubits)}
    return d


def get_qubit_indices(qubit_indices: Dict[Qubit, int], node: DAGNode):
    qargs = node.qargs
    ret = [qubit_indices[qubit] for qubit in qargs]
    return ret
