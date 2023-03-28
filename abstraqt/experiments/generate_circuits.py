import argparse
import os

import pandas as pd
from qiskit.circuit import QuantumCircuit

from abstraqt.experiments.generate_helper import random_operation, random_operation_on_qubits
from abstraqt.baselines.transpile_yp21 import qasm_circuit_to_yp21
from abstraqt.experiments.optimizer import optimize_pyzx


default_n_qubits, default_repeat = 62, 10000


def clifford__clifford(n_qubits: int=default_n_qubits, repeat: int=default_repeat):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['h', 's', 'cx'], upper, repeat=repeat)

    # lower
    random = random_operation_on_qubits(qc, ['h', 's', 'cx'], lower, repeat=repeat)

    # invert lower
    postprocessing: QuantumCircuit = optimize_pyzx(random.inverse())

    return qc, postprocessing


def clifford_t__clifford(n_qubits: int=default_n_qubits, repeat: int=default_repeat):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['h', 's', 'cx', 't'], upper, repeat=repeat)

    # lower
    random = random_operation_on_qubits(qc, ['h', 's', 'cx'], lower, repeat=repeat)

    # invert lower
    postprocessing: QuantumCircuit = optimize_pyzx(random.inverse())

    return qc, postprocessing


def clifford_t__clifford_large(n_qubits: int=default_n_qubits, repeat:int=default_repeat):
    return clifford_t__clifford(n_qubits*4, repeat)


def clifford_t__cx_t(n_qubits: int=default_n_qubits, repeat: int=default_repeat):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['h', 's', 'cx', 't'], upper, repeat=repeat)

    # lower
    random = random_operation(
        qc,
        ('cx', (upper, lower)),
        ('t', (lower,)),
        repeat=repeat
    )

    # invert lower
    postprocessing: QuantumCircuit =  optimize_pyzx(random)

    return qc, postprocessing


def clifford_t__h_cz_rx(n_qubits: int=default_n_qubits, repeat: int=default_repeat):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['h', 's', 'cx', 't'], upper, repeat=repeat)

    # lower
    for i in lower:
        qc.h(i)

    random = random_operation(
        qc,
        ('cz', (upper, lower)),
        ('rx0.8', (lower,)),
        repeat=repeat
    )

    # invert lower
    postprocessing = optimize_pyzx(random)
    for i in lower:
        postprocessing.h(i)

    return qc, postprocessing


def ccx_h__clifford(n_qubits: int=default_n_qubits, repeat: int=default_repeat):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['ccx', 'h'], upper, repeat=repeat)

    # lower
    random = random_operation(
        qc,
        ('h', (lower,)),
        ('s', (lower,)),
        ('cx', (lower, lower)),
        ('cx', (lower, upper)),
        repeat=repeat
    )

    # invert lower
    postprocessing: QuantumCircuit = optimize_pyzx(random.inverse())

    return qc, postprocessing


def ccx_h__cx_t(n_qubits: int=default_n_qubits, repeat: int=default_repeat):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['ccx', 'h'], upper, repeat=repeat)

    # lower
    random = random_operation(
        qc,
        ('cx', (upper, lower)),
        ('t', (lower,)),
        repeat=repeat
    )

    # invert lower
    postprocessing: QuantumCircuit = optimize_pyzx(random)

    return qc, postprocessing


def rz_h__cx(n_qubits: int=default_n_qubits, repeat: int=default_repeat):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['rz2', 'h'], upper, repeat=repeat)

    # lower
    random = random_operation(qc, ('cx', (upper, lower)), repeat=repeat)

    # invert lower
    postprocessing: QuantumCircuit = optimize_pyzx(random.inverse())

    return qc, postprocessing


def measure_ghz(n_qubits: int=default_n_qubits, repeat: int=default_repeat//100):
    qc = QuantumCircuit(n_qubits, 1)

    for _ in range(repeat):
        qc.h(0)
        for j in range(1, n_qubits):
            qc.cx(0, j)
        qc.measure(0, 0)
        for j in range(1, n_qubits):
            qc.cx(0, j)

    postprocessing = QuantumCircuit(n_qubits)

    return qc, postprocessing


###########
# HELPERS #
###########


def save_qasm(circuit: QuantumCircuit, target_file: str):
    with open(target_file, 'w') as file:
        s = circuit.qasm()
        file.write(s)


def save_q(circuit: QuantumCircuit, target_file: str):
    try:
        qasm_circuit_to_yp21(circuit, None, target_file)
        return True
    except ValueError as e:
        print('WARNING: Could not generate .q file ' + target_file)
        print(e)
        return False


def save_qasm_and_q(circuit: QuantumCircuit, target_file: str, do_q: bool=True):
    """
    target_file: without file extension
    """
    circuit = circuit.decompose(gates_to_decompose=['rx0.8', 'rz2'])
    save_qasm(circuit, target_file + '.qasm')
    if do_q:
        return save_q(circuit, target_file + '.q')


###############
# ENTRY POINT #
###############


def generate_all(target_directory: str):
    rows = []

    for f in [
        clifford__clifford,
        # clifford_t__clifford_large,
        clifford_t__clifford,
        clifford_t__cx_t,
        clifford_t__h_cz_rx,
        ccx_h__clifford,
        ccx_h__cx_t,
        rz_h__cx,
        measure_ghz
    ]:
        # generate circuit
        circuit, postprocessing = f()
        composed = circuit.compose(postprocessing)

        # store circuit
        q_success_circuit = save_qasm_and_q(
            circuit,
            os.path.join(target_directory, f.__name__ + '-circuit')
        )
        q_success_postprocessing = save_qasm_and_q(
            postprocessing,
            os.path.join(target_directory, f.__name__ + '-postprocessing'),
            do_q=q_success_circuit
        )
        q_success_full = save_qasm_and_q(
            composed,
            os.path.join(target_directory, f.__name__ + '-full'),
            do_q=q_success_circuit
        )
    
        if q_success_circuit:
            assert q_success_postprocessing
            assert q_success_full

        rows.append({
            'label': f.__name__,
            'q_file_success': q_success_circuit,
            'qubits': circuit.num_qubits,
            **{
                instruction: count for
                instruction, count in composed.count_ops().items()
            }
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(target_directory, 'circuits.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()
    directory_of_script = os.path.dirname(os.path.abspath(__file__))
    target_directory = os.path.join(directory_of_script, 'circuits')

    parser.add_argument('--target-directory', default=target_directory)
    args = parser.parse_args()
    generate_all(args.target_directory)


if __name__ == '__main__':
    main()
