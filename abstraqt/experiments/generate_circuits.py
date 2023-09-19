import argparse
import os
from typing import Optional

import pandas as pd
from qiskit.circuit import QuantumCircuit

from abstraqt.experiments.generate_helper import random_operation, random_operation_on_qubits
from abstraqt.baselines.transpile_yp21 import qasm_circuit_to_yp21
from abstraqt.experiments.optimizer import optimize_pyzx
from abstraqt.experiments.qasm_to_qc import qasm_circuit_to_qc


def clifford__clifford(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['h', 's', 'cx'], upper, n_gates=n_gates)
    simplified = qc.copy()

    # lower
    random = random_operation_on_qubits(qc, ['h', 's', 'cx'], lower, n_gates=n_gates)

    # invert lower
    inverse = random.inverse()
    if optimize_parts:
        inverse = optimize_pyzx(inverse, check_optimized)
    qc.compose(inverse, inplace=True)

    return qc, simplified


def clifford_t__clifford(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['h', 's', 'cx', 't'], upper, n_gates=n_gates)
    simplified = qc.copy()

    # lower
    random = random_operation_on_qubits(qc, ['h', 's', 'cx'], lower, n_gates=n_gates)

    # invert lower
    inverse = random.inverse()
    if optimize_parts:
        inverse = optimize_pyzx(inverse, check_optimized)
    qc.compose(inverse, inplace=True)

    return qc, simplified


def clifford_t__clifford_large(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    return clifford_t__clifford(n_qubits*4, n_gates, optimize_parts, check_optimized)


def clifford_t__cx_t(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['h', 's', 'cx', 't'], upper, n_gates=n_gates)
    simplified = qc.copy()

    # lower
    random = random_operation(
        qc,
        ('cx', (upper, lower)),
        ('t', (lower,)),
        n_gates=n_gates
    )

    # invert lower
    inverse = random.inverse()
    if optimize_parts:
        inverse = optimize_pyzx(inverse, check_optimized)
    qc.compose(inverse, inplace=True)

    return qc, simplified


def clifford_t__cx_t__twice(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['h', 's', 'cx', 't'], upper, n_gates=n_gates)
    simplified = qc.copy()

    # lower
    random = random_operation(
        qc,
        ('cx', (upper, lower)),
        ('t', (lower,)),
        n_gates=n_gates
    )

    # reapply lower
    repeated = random.copy()
    if optimize_parts:
        repeated = optimize_pyzx(repeated, check_optimized)
    qc.compose(repeated, inplace=True)

    return qc, simplified


def clifford_t__h_cz_rx(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['h', 's', 'cx', 't'], upper, n_gates=n_gates)
    simplified = qc.copy()

    # lower
    for i in lower:
        qc.h(i)

    random = random_operation(
        qc,
        ('cz', (upper, lower)),
        ('rx0.8', (lower,)),
        n_gates=n_gates
    )

    # invert lower
    inverse = random.inverse()
    if optimize_parts:
        inverse = optimize_pyzx(inverse, check_optimized)
    qc.compose(inverse, inplace=True)
    for i in lower:
        qc.h(i)

    return qc, simplified


def clifford_t__h_cz_rx__twice(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['h', 's', 'cx', 't'], upper, n_gates=n_gates)
    simplified = qc.copy()

    # lower
    for i in lower:
        qc.h(i)

    random = random_operation(
        qc,
        ('cz', (upper, lower)),
        ('rx0.8', (lower,)),
        n_gates=n_gates
    )

    # repeat lower
    repeated = random.copy()
    if optimize_parts:
        repeated = optimize_pyzx(repeated, check_optimized)
    qc.compose(repeated, inplace=True)
    for i in lower:
        qc.h(i)

    return qc, simplified


def ccx_h__clifford(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['ccx', 'h'], upper, n_gates=n_gates)
    simplified = qc.copy()

    # lower
    random = random_operation(
        qc,
        ('h', (lower,)),
        ('s', (lower,)),
        ('cx', (lower, lower)),
        ('cx', (lower, upper)),
        n_gates=n_gates
    )

    # invert lower
    inverse = random.inverse()
    if optimize_parts:
        inverse = optimize_pyzx(inverse, check_optimized)
    qc.compose(inverse, inplace=True)

    return qc, simplified


def ccx_h__cx_t(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['ccx', 'h'], upper, n_gates=n_gates)
    simplified = qc.copy()

    # lower
    random = random_operation(
        qc,
        ('cx', (upper, lower)),
        ('t', (lower,)),
        n_gates=n_gates
    )

    # invert lower
    inverse = random.inverse()
    if optimize_parts:
        inverse = optimize_pyzx(inverse, check_optimized)
    qc.compose(inverse, inplace=True)

    return qc, simplified


def ccx_h__cx_t__twice(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['ccx', 'h'], upper, n_gates=n_gates)
    simplified = qc.copy()

    # lower
    random = random_operation(
        qc,
        ('cx', (upper, lower)),
        ('t', (lower,)),
        n_gates=n_gates
    )

    # reapply lower
    repeated = random.copy()
    if optimize_parts:
        repeated = optimize_pyzx(repeated, check_optimized)
    qc.compose(repeated, inplace=True)

    return qc, simplified


def rz_h__cx(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['rz2', 'h'], upper, n_gates=n_gates)
    simplified = qc.copy()

    # lower
    random = random_operation(qc, ('cx', (upper, lower)), n_gates=n_gates)

    # invert lower
    inverse = random.inverse()
    if optimize_parts:
        inverse = optimize_pyzx(inverse, check_optimized)
    qc.compose(inverse, inplace=True)

    return qc, simplified


def rz_h__cx__twice(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    qc = QuantumCircuit(n_qubits)

    n_first = n_qubits // 2
    upper = list(range(n_first))
    lower = list(range(n_first, n_qubits))

    # upper
    random_operation_on_qubits(qc, ['rz2', 'h'], upper, n_gates=n_gates)
    simplified = qc.copy()

    # lower
    random = random_operation(qc, ('cx', (upper, lower)), n_gates=n_gates)

    # repeat lower
    repeated = random.copy()
    if optimize_parts:
        repeated = optimize_pyzx(repeated, check_optimized)
    qc.compose(repeated, inplace=True)

    return qc, simplified


def measure_ghz(n_qubits: int, n_gates: int, optimize_parts: bool, check_optimized: bool):
    n_gates //= 100
    qc = QuantumCircuit(n_qubits, 1)

    for _ in range(n_gates):
        qc.h(0)
        for j in range(1, n_qubits):
            qc.cx(0, j)
        qc.measure(0, 0)
        for j in range(1, n_qubits):
            qc.cx(0, j)

    return qc, qc


###########
# HELPERS #
###########


def save_qasm(circuit: QuantumCircuit, target_file: str, comments: Optional[str]=None):
    with open(target_file, 'w') as file:
        if comments is not None:
            comments = '\n'.join(f'// {line}' for line in comments.splitlines())
            file.write(comments)
            file.write('\n')
        s = circuit.qasm()
        file.write(s)


def save_yp21(circuit: QuantumCircuit, target_file: str):
    try:
        qasm_circuit_to_yp21(circuit, target_file)
        return True
    except ValueError as e:
        print('WARNING: Could not generate .q file ' + target_file)
        print(e)
        return False


def save_feynman(circuit: QuantumCircuit, target_file: str):
    try:
        qasm_circuit_to_qc(circuit, target_file)
        return True
    except ValueError as e:
        print('WARNING: Could not generate .qc file ' + target_file)
        print(e)
        return False


def save_all_formats(circuit: QuantumCircuit, target_file: str, do_yp: bool=True, do_feynman: bool=True, comments: Optional[str]=None):
    """
    target_file: without file extension
    """
    circuit = circuit.decompose(gates_to_decompose=['rx0.8', 'rz2'])
    save_qasm(circuit, target_file + '.qasm', comments)

    # YP
    yp = False
    if do_yp:
        yp = save_yp21(circuit, target_file + '.q')

    # Feynman
    feynman = False
    if do_feynman:
        feynman = save_feynman(circuit, target_file + '.qc')
    
    return yp, feynman


###############
# ENTRY POINT #
###############


benchmarks = [
    clifford__clifford,
    # clifford_t__clifford_large,
    clifford_t__clifford,
    clifford_t__cx_t,
    clifford_t__cx_t__twice,
    clifford_t__h_cz_rx,
    clifford_t__h_cz_rx__twice,
    ccx_h__clifford,
    ccx_h__cx_t,
    ccx_h__cx_t__twice,
    rz_h__cx,
    rz_h__cx__twice,
    measure_ghz
]


def generate_all(target_directory: str, n_qubits: int, n_gates: int, repeat: int, optimize_parts: bool, check_optimized: bool):
    rows = []

    file_list = os.listdir(target_directory)
    extensions_to_delete = [".qasm", ".q", ".qc"]
    for filename in file_list:
        for extension in extensions_to_delete:
            if filename.endswith(extension):
                file_path = os.path.join(target_directory, filename)
                os.remove(file_path)

    for i in range(repeat):
        for f in benchmarks:
            label = f'{f.__name__}-{i}'

            # generate circuit
            circuit, simplified = f(n_qubits, n_gates, optimize_parts, check_optimized)

            # store circuit
            comments = f'PARAMETERS: n_qubits: {n_qubits}, n_gates: {n_gates}, repeat: {repeat}, optimize_parts: {optimize_parts}, check_optimized: {check_optimized}'
            yp_circuit, feynman_circuit = save_all_formats(
                circuit,
                os.path.join(target_directory, label + '-full'),
                comments=comments
            )
            yp_simplified, feynman_simplified = save_all_formats(
                simplified,
                os.path.join(target_directory, label + '-simplified'),
                do_yp=yp_circuit,
                do_feynman=feynman_circuit,
                comments=comments
            )
        
            if yp_circuit:
                assert yp_simplified

            if feynman_circuit:
                assert feynman_simplified

            rows.append({
                'label': label,
                'q_file_success': yp_circuit,
                'qc_file_success': feynman_circuit,
                'qubits': circuit.num_qubits,
                **{
                    instruction: count for
                    instruction, count in circuit.count_ops().items()
                }
            })
    
    df = pd.DataFrame(rows)
    for column in ['cx', 'h', 's', 'sdg', 'cz', 'x', 'z', 't', 'tdg', 'rx0.8', 'ccx', 'rz2', 'measure']:
        if column not in df.columns:
            df[column] = 0
        df[column] = df[column].fillna(0)
        df[column] = df[column].astype(int)
    df['clifford'] = \
        df['cx'] + \
        df['h'] + \
        df['s'] + \
        df['sdg'] + \
        df['cz'] + \
        df['x'] + \
        df['z']
    df.to_csv(os.path.join(target_directory, 'circuits.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    directory_of_script = os.path.dirname(os.path.abspath(__file__))
    target_directory = os.path.join(directory_of_script, 'circuits')

    parser.add_argument('--target-directory', default=target_directory, help='Directory to store the generated circuits.')
    parser.add_argument('--qubits', default=62, type=int, help='Number of qubits in each circuit.')
    parser.add_argument('--gates', default=10000, type=int, help='Approximate number of gates in each circuit.')
    parser.add_argument('--repeat', default=1, type=int, help='Number of circuits to generate from each type.')
    parser.add_argument('--no-optimize', action='store_true', help='Do not optimize parts of circuits.')
    parser.add_argument('--check-optimized', action='store_true', help='Check if optimization of parts of circuits is correct (only works for tiny circuits).')

    args = parser.parse_args()
    generate_all(
        args.target_directory,
        args.qubits,
        args.gates,
        args.repeat,
        not args.no_optimize,
        args.check_optimized
    )


if __name__ == '__main__':
    main()
