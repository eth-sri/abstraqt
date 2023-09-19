import os

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.converters import circuit_to_dag

from abstraqt.applications.circuit_helper.dag_helper import DAGDecomposer
from abstraqt.applications.circuit_helper.extract_assertions import bases_assertion_marker, extract_assertions_from_line
from abstraqt.applications.circuit_helper.gate_helper import qiskit_to_abstraqt_name
from abstraqt.applications.circuit_helper.qiskit_helper import topological_op_nodes
from abstraqt.applications.circuit_helper.qubit_helper import get_qubit_indices_from_circuit, get_qubit_indices
from abstraqt.utils import logging


logger = logging.getLogger(__name__)


def yp21_circuit_to_qasm(source_path: str, target_path: str):
    with open(source_path, 'r') as source_file:
        with open(target_path, 'w') as target_file:
            target_file.write('OPENQASM 2.0;\n')
            target_file.write('include "qelib1.inc";\n')
            for line in source_file:
                line, *_ = line.split('//')  # remove comments
                line = line.strip()
                if line == '':
                    continue
                elif line.startswith('circuit:'):
                    n = line.replace('circuit: ', '').replace(' qubits', '')
                    n = int(n)
                    target_file.write(f'qreg q[{n}];\n')
                elif line.startswith('assert'):
                    bases = []
                    while '}' not in line:
                        bases += extract_assertions_from_line(line)
                        line = next(source_file)
                    bases += extract_assertions_from_line(line)
                    # invert, to ensure first entry corresponds to first qubit
                    bases = [b[::-1] for b in bases]
                    bases = [f'|{b}>' for b in bases]
                    target_file.write(bases_assertion_marker + ' '.join(bases) + '\n')
                    continue
                elif line.startswith('measure'):
                    # ensure measurement is the last operation
                    while True:
                        try:
                            line = next(source_file)
                            line = line.strip()
                            assert line == ''
                        except StopIteration:
                            break
                else:
                    op, qubits = line.split('(')

                    op = op.lower()
                    op = op.replace('not', 'x')

                    qubits = qubits.replace(')', '')
                    qubits = qubits.split(',')
                    qubits = [int(q) for q in qubits]

                    # handle inverted controls
                    if op in ['ncx', 'ncncx', 'nccx']:
                        target_file.write(f'x q[{qubits[0]}];\n')
                    if op in ['ncncx']:
                        target_file.write(f'x q[{qubits[1]}];\n')

                    # handle inverted controls operation
                    qubits_str = ','.join([f'q[{q}]' for q in qubits])
                    actual_op = op
                    if actual_op in ['ncx', 'ncncx', 'nccx']:
                        actual_op = actual_op.replace('n', '')
                    if actual_op == 'd':
                        actual_op = 'tdg'

                    # write operation
                    target_file.write(f'{actual_op} {qubits_str};\n')

                    # handle inverted controls
                    if op in ['ncx', 'ncncx', 'nccx']:
                        target_file.write(f'x q[{qubits[0]}];\n')
                    if op in ['ncncx']:
                        target_file.write(f'x q[{qubits[1]}];\n')


def qasm_circuit_to_yp21(circuit: QuantumCircuit, target_path: str):
    circuit = circuit_to_dag(circuit)

    decomposer = DAGDecomposer(
        supported_gates=['H', 'X', 'T', 'Z', 'TDG', 'S', 'SDG', 'CZ', 'CNOT', 'CCNOT'],
        support_parameterized=False,
    )

    circuit = decomposer.decompose_dag(circuit)

    qubit_indices = get_qubit_indices_from_circuit(circuit)

    try:
        with open(target_path, 'w') as target_file:
            target_file.write(f'circuit: {circuit.num_qubits()} qubits\n')

            for node in topological_op_nodes(circuit):
                op = node.op
                if not isinstance(op, Gate):
                    raise ValueError(f'Unexpected operation {op} of type {type(op)}')
                name = qiskit_to_abstraqt_name(op.name)
                if name == 'TDG':
                    name = 'D'

                qubits = get_qubit_indices(qubit_indices, node)

                swapped = False
                if name == 'CNOT':
                    if qubits[0] > qubits[1]:
                        swapped = True
                        qubits[0], qubits[1] = qubits[1], qubits[0]
                        target_file.write(f'H({qubits[0]})\nH({qubits[1]})\n')

                qubits_str = ','.join([str(q) for q in qubits])
                if name == 'SDG':
                    target_file.write(f'S({qubits_str})\n')
                    target_file.write(f'Z({qubits_str})\n')
                else:
                    target_file.write(f'{name}({qubits_str})\n')
                
                if swapped:
                    target_file.write(f'H({qubits[0]})\nH({qubits[1]})\n')

            n_qubits = circuit.num_qubits()
            zeros = ''.join(['0' for _ in range(n_qubits)])
            target_file.write('assert state in span { |' + str(zeros) + '>, |' + str(zeros) + '> }\n')
            target_file.write(f'measure 0..{n_qubits}\n')
    except ValueError:
        os.remove(target_path)
        raise
