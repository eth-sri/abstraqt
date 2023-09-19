import random
from typing import Sequence, List, Tuple, Union, Callable, Any

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import HGate, SGate, SdgGate, CXGate, CZGate, XGate, TGate, CCXGate, ZGate, TdgGate


OP_FUNC = Callable[[QuantumCircuit, int], Any]
OP = Union[str, OP_FUNC]


#########
# GATES #
#########


def get_rx0_8_gate():
    circuit = QuantumCircuit(1, name='rx0.8')
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    return circuit.to_instruction()


def get_rz2_gate():
    circuit = QuantumCircuit(1, name='rz2')
    circuit.rz(2, 0)
    return circuit.to_instruction()


name_to_gate = {
    'h': HGate,
    's': SGate,
    'z': ZGate,
    'sdg': SdgGate,
    'cx': CXGate,
    'cz': CZGate,
    't': TGate,
    'tdg': TdgGate,
    'x': XGate,
    'ccx': CCXGate,
    'rx0.8': get_rx0_8_gate,
    'rz2': get_rz2_gate,
}


def empty_like(circuit: QuantumCircuit):
    if circuit.num_clbits == 0:
        return QuantumCircuit(circuit.num_qubits)
    else:
        return QuantumCircuit(circuit.num_qubits, circuit.num_clbits)


##########
# RANDOM #
##########


def shuffle_gates(circuit: QuantumCircuit):
    circuit = circuit.copy()
    gate_qubit_tuples = circuit.data
    random.shuffle(gate_qubit_tuples)

    shuffled = empty_like(circuit)
    
    for gate_qubit_tuple in gate_qubit_tuples:
        gate_name = gate_qubit_tuple[0].name
        gate = name_to_gate[gate_name]()

        qubits = [qubit.index for qubit in gate_qubit_tuple[1]]
        params = gate_qubit_tuple[0].params
        shuffled.append(gate, qubits, params)

    return shuffled


def random_from_sequence(options: Sequence[int], exclude: List[int]=[]):
    options = list(options)

    for e in exclude:
        if e in options:
            options.remove(e)
    
    if len(options) == 0:
        raise ValueError("No more options left to chose from.")

    return random.choice(options)


def random_operation(circuit: QuantumCircuit, *operations: Tuple[OP, Sequence[List[int]]], n_gates: int=1) -> None:
    randoms = QuantumCircuit(circuit.num_qubits)

    for _ in range(n_gates):
        operation, options_for_qubits = random.choice(operations)

        if isinstance(operation, str):
            operation = name_to_gate[operation]()

        chosen_qubits: List[int] = []
        for options_for_qubit in options_for_qubits:
            qubit = random_from_sequence(options_for_qubit, chosen_qubits)
            chosen_qubits.append(qubit)
        
        randoms.append(operation, chosen_qubits)

    circuit.compose(randoms, inplace=True)

    return randoms


def random_operation_on_qubits(circuit: QuantumCircuit, operation_onlys: List[OP], qubits: Sequence[int], n_gates: int=1):
    qubits = list(qubits)

    operations: List[Tuple[OP, Sequence[List[int]]]] = []
    for operation_only in operation_onlys:
        if isinstance(operation_only, str):
            operation_only = name_to_gate[operation_only]()

        n_arguments = operation_only.num_qubits
        options_for_qubits = n_arguments * (qubits,)

        operations.append((operation_only, options_for_qubits))
    
    return random_operation(circuit, *operations, n_gates=n_gates)
