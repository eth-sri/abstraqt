from qiskit.circuit import QuantumCircuit
import pyzx as zx
import numpy as np

from abstraqt.experiments.generate_helper import empty_like, name_to_gate


def optimize_pyzx(circuit: QuantumCircuit):
    qasm = circuit.qasm()
    circuit_zx = zx.Circuit.from_qasm(qasm)
    circuit_zx = zx.optimize.basic_optimization(circuit_zx.to_basic_gates())
    qasm = circuit_zx.to_qasm()
    
    ret = QuantumCircuit.from_qasm_str(qasm)
    ret = lift_operations(ret)

    return ret


def lift_operations(circuit: QuantumCircuit):
    gate_qubit_tuples = circuit.data

    ret = empty_like(circuit)
    for gate_qubit_tuple in gate_qubit_tuples:
        gate_name = gate_qubit_tuple[0].name
        qubits = [qubit.index for qubit in gate_qubit_tuple[1]]
        params = gate_qubit_tuple[0].params

        if gate_name == 'rz':
            if np.isclose(params[0], 0.25*np.pi):
                ret.t(*qubits)
            elif np.isclose(params[0], 0.5*np.pi):
                ret.s(*qubits)
            elif np.isclose(params[0], 0.75*np.pi):
                ret.t(*qubits)
                ret.s(*qubits)
            elif np.isclose(params[0], np.pi):
                ret.z(*qubits)
            elif np.isclose(params[0], 1.25*np.pi):
                ret.t(*qubits)
                ret.z(*qubits)
            elif np.isclose(params[0], 1.5*np.pi):
                ret.sdg(*qubits)
            elif np.isclose(params[0], 1.75*np.pi):
                ret.tdg(*qubits)
            else:
                print('Warning: could not lift', gate_qubit_tuple)
                ret.rz(params[0], *qubits)
        else:
            gate = name_to_gate[gate_name]()
            ret.append(gate, qubits, params)

    return ret
