import os
from typing import Union

from qiskit import QuantumCircuit
from qiskit.circuit import Measure, Reset
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.qasm import QasmError

from abstraqt.utils import logging

logger = logging.getLogger(__name__)


def save_circuit_to_qasm_file(circuit: Union[QuantumCircuit, DAGCircuit], target_file: str):
    if not isinstance(circuit, QuantumCircuit):
        circuit = dag_to_circuit(circuit)

    # convert to string
    s = circuit.qasm()

    # write string to file
    with open(target_file, 'w') as f:
        f.write(s)


def load_from_qasm_file(circuit_file: str, relative_to_directory: str = None, log_error=True):
    try:
        circuit = QuantumCircuit.from_qasm_file(circuit_file)
    except QasmError as e:
        if log_error:
            logger.error('Could not parse %s due to %s', circuit_file, e)
        raise
    if relative_to_directory is None:
        circuit.name = circuit_file
    else:
        circuit.name = os.path.relpath(circuit_file, relative_to_directory)
    logger.verbose('Parsed %s', circuit_file)

    return circuit


def topological_op_nodes(circuit: DAGCircuit, include_directive=False):
    for node in circuit.topological_op_nodes():
        if not include_directive and node.op._directive:
            continue
        else:
            assert isinstance(node, DAGNode)
            yield node


def has_measurement(circuit: DAGCircuit):
    has_measurements = False
    for node in topological_op_nodes(circuit):
        op = node.op
        if isinstance(op, (Measure, Reset)):
            has_measurements = True
            break

    return has_measurements


def number_of_operations(circuit: DAGCircuit):
    n = sum(circuit.count_ops().values())
    return n
