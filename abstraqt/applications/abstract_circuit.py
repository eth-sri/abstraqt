from qiskit.dagcircuit import DAGNode, DAGCircuit
from qiskit.circuit import Gate, Measure, Reset, Barrier
from tqdm import tqdm

from abstraqt.applications.circuit_helper.gate_helper import qiskit_to_abstraqt_name, get_matrix_from_gate
from abstraqt.applications.circuit_helper.qiskit_helper import topological_op_nodes, number_of_operations
from abstraqt.applications.circuit_helper.qubit_helper import get_qubit_indices, get_qubit_indices_from_circuit
from abstraqt.linalg.common_matrices import matrices_all_dict
from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from abstraqt.utils import logging
from abstraqt.utils.logging.log_data import report_runtime

logger = logging.getLogger(__name__)


class AbstractCircuit:

    def __init__(self, circuit: DAGCircuit, max_n_summands: int, ):
        self.circuit = circuit
        self.max_n_summands = max_n_summands
        self.all_qubits = get_qubit_indices_from_circuit(circuit)

    def apply_node(self, node: DAGNode, state: AbstractStabilizerPlus):
        qubits = self.get_qubits(node)
        op = node.op

        if isinstance(op, Gate):
            op = self.get_abstraqt_operation(node)
            state = state.conjugate(op, *qubits)
            state = state.compress(self.max_n_summands)
        elif isinstance(op, Measure):
            state = state.measure('Z', *qubits, both=True)
        elif isinstance(op, Reset):
            state = state.measure('Z', *qubits, reset=True)
        else:
            raise ValueError(f'Unexpected operation type {type(op)}')

        return state

    @staticmethod
    def get_abstraqt_operation(node: Gate):
        name = node.name
        abstraqt_name = qiskit_to_abstraqt_name(name)

        # translate operation
        if abstraqt_name in matrices_all_dict.keys():
            op = abstraqt_name
        else:
            op = get_matrix_from_gate(node)

        return op

    def get_qubits(self, node: DAGNode):
        qubits = get_qubit_indices(self.all_qubits, node)
        return qubits

    def apply_all_nodes(self, state: AbstractStabilizerPlus):
        ops = topological_op_nodes(self.circuit)
        n_ops = number_of_operations(self.circuit)
        t = tqdm(ops, total=n_ops, disable=None, desc=self.circuit.name)
        for node in t:
            logger.slow('Current progress: %s', t)

            state = self.apply_node(node, state)

            yield node, state
