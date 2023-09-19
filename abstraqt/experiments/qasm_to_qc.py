from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.converters import circuit_to_dag

from abstraqt.applications.circuit_helper.qiskit_helper import topological_op_nodes
from abstraqt.applications.circuit_helper.qubit_helper import get_qubit_indices_from_circuit, get_qubit_indices

def qasm_circuit_to_qc(circuit: QuantumCircuit, target_path: str):
    circuit = circuit_to_dag(circuit)

    qubit_indices = get_qubit_indices_from_circuit(circuit)

    with open(target_path, 'w') as target_file:
        variables = [f'x{i}' for i in range(circuit.num_qubits())]
        variables = ' '.join(variables)

        target_file.write(f'.v {variables}\n\n')
        target_file.write('BEGIN\n')

        for node in topological_op_nodes(circuit):
            op = node.op
            if not isinstance(op, Gate):
                raise ValueError(f'Unexpected operation {op} of type {type(op)}')
            name = op.name
            if name in ['h', 'x', 'y', 'z', 's', 't']:
                name = name.upper()
            elif name == 'tdg':
                name = 'T*'
            elif name == 'sdg':
                name = 'S*'
            elif name == 'cx':
                name = 'X'
            elif name == 'ccx':
                name = 'X'
            elif name == 'cz':
                name = 'Z'
            elif name == 'rz':
                params = op.params
                assert len(params) == 1
                name = f'Rz {params[0]}'
            else:
                raise ValueError('Unknown op name ' + name)
            
            qubits = get_qubit_indices(qubit_indices, node)
            qubits_str = ' '.join([f'x{q}' for q in qubits])

            target_file.write(f'{name} {qubits_str}\n')
        

        target_file.write('END\n')
