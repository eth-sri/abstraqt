from typing import Union

from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGNode
from qiskit.quantum_info import Operator


parameterized_gates = ['cu1', 'rx', 'ry', 'rz', 'u1', 'u3']


def qiskit_to_abstraqt_name(name: str):
    name = name.upper()
    if name == 'CX':
        name = 'CNOT'
    if name == 'CCX':
        name = 'CCNOT'
    elif name == 'ID':
        name = 'I'
    return name


def abstraqt_to_qiskit_name(name: str):
    name = name.lower()
    if name == 'CNOT':
        name = 'cx'
    elif name == 'i':
        name = 'id'
    return name


def get_matrix_from_gate(gate: Union[DAGNode, Gate]):
    if isinstance(gate, DAGNode):
        gate = gate.op

    op = Operator(gate)
    matrix = op.data
    return matrix
