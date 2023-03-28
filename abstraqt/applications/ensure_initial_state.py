from qiskit.dagcircuit import DAGCircuit

from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus


def ensure_initial_state(circuit: DAGCircuit, s: AbstractStabilizerPlus):
    n_qubits = circuit.num_qubits()
    if s == '0':
        s = AbstractStabilizerPlus.zero_state(n_qubits)
    elif s == '+':
        s = AbstractStabilizerPlus.zero_state(n_qubits)
        for i in range(n_qubits):
            s = s.conjugate('H', i)
    elif s is None:
        s = AbstractStabilizerPlus.zero_state(n_qubits)
    else:
        assert isinstance(s, AbstractStabilizerPlus)
    return s
