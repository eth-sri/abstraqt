from typing import Union

from qiskit import QuantumCircuit
from qiskit.circuit import Measure, Reset, Gate
from qiskit.circuit.library import SGate, TGate, IGate, SdgGate, TdgGate, HGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGNode

from abstraqt.linalg.common_matrices import matrices_all_dict
from abstraqt.applications.circuit_helper.gate_helper import qiskit_to_abstraqt_name
from abstraqt.applications.circuit_helper.gate_set import GateSet, matrices_all_dict_hashed
from abstraqt.utils import logging
from .gate_helper import parameterized_gates as default_parameterized_gates
from .qiskit_helper import topological_op_nodes
from ...utils.logging.log_data import report_runtime_decorator

logger = logging.getLogger(__name__)


gate_lookup = {
    'S': SGate(),
    'T': TGate(),
    'I': IGate(),
    'SDG': SdgGate(),
    'TDG': TdgGate(),
    'H': HGate(),
}


class DAGDecomposer:

    def __init__(
            self,
            parameterized_gates=default_parameterized_gates,
            supported_gates=matrices_all_dict.keys(),
            support_parameterized=True,
            decompose_hints=None
    ):
        self.parameterized_gates = parameterized_gates
        self.supported_gates = supported_gates
        self.support_parameterized = support_parameterized
        if decompose_hints is None:
            self.decompose_hints = {}
        else:
            self.decompose_hints = decompose_hints

        self.decomposed_gates = GateSet()
        self.incorrect_matrix_names = set()

    @report_runtime_decorator(report_overhead=False)
    def decompose_dag(self, dag: Union[QuantumCircuit, DAGCircuit]):
        if isinstance(dag, QuantumCircuit):
            dag = circuit_to_dag(dag)

        for node in topological_op_nodes(dag):
            name = node.name
            op = node.op

            if isinstance(op, (Measure, Reset)):
                continue
            elif isinstance(op, Gate):
                abstraqt_name = qiskit_to_abstraqt_name(name)

                if name in self.parameterized_gates:
                    # try to lift
                    lifted = self.try_lift_variable(dag, node)
                    if not self.support_parameterized:
                        if lifted is None:
                            raise ValueError(f'Cannot lift variable gate {node.name} to known gate')
                        elif lifted not in self.supported_gates:
                            raise ValueError(f'Lifted variable gate to unsupported gate {lifted}')
                elif abstraqt_name in self.decompose_hints:
                    dag.substitute_node_with_dag(node, self.decompose_hints[abstraqt_name])
                elif abstraqt_name == 'I':
                    # always remove identity
                    dag.remove_op_node(node)
                elif abstraqt_name in self.supported_gates:
                    pass  # no action necessary
                elif op.definition is not None:
                    definition = op.definition

                    # check if same decomposition happened before
                    if op not in self.decomposed_gates:
                        logger.verbose('Decomposing %s to \n%s', name, definition)
                    self.decomposed_gates.add(op)

                    # decompose
                    decomposition = circuit_to_dag(definition)
                    decomposition = self.decompose_dag(decomposition)
                    dag.substitute_node_with_dag(node, decomposition)

                else:
                    raise ValueError(f'Cannot decompose unknown gate {name}')
            else:
                raise ValueError(f'Unexpected operation type {type(op)}')

        return dag

    @staticmethod
    def try_lift_variable(dag: DAGCircuit, node: DAGNode):
        op = node.op
        if op in matrices_all_dict_hashed:
            actual = matrices_all_dict_hashed[op]

            if actual == 'I':
                dag.remove_op_node(node)
            elif actual in gate_lookup:
                gate = gate_lookup[actual]
                dag.substitute_node(node, gate)
                logger.verbose('Lifted unnecessarily decomposed gate to %s', gate.name)

                abstraqt_name = qiskit_to_abstraqt_name(gate.name)
                return abstraqt_name
            else:
                logger.warning('Did not implement lifting the unnecessarily decomposed gate %s to %s', node.name,
                               actual)
        return None
