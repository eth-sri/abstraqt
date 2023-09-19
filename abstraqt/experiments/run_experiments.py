import os
import subprocess
import resource
import re
import argparse
from typing import Callable, Optional, List

import numpy as np
from tqdm import tqdm
import pandas as pd
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.providers.aer import AerSimulator
from qiskit import transpile, QiskitError

from abstraqt.applications.circuit_helper.qiskit_helper import topological_op_nodes
from abstraqt.applications.abstract_circuit import AbstractCircuit
from abstraqt.applications.circuit_helper.qubit_helper import get_qubit_indices, get_qubit_indices_from_circuit
from abstraqt.utils.list_helper import filter_duplicates
from abstraqt.utils.pandas_helper import move_to_front, value_counts_str
from abstraqt.utils.qiskit_helper import circuits_equivalent, MemoryError
from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from abstraqt.abstract_interpretation.interval_array import IntervalArray
from abstraqt.utils.my_multiprocessing import MyProcessMap, MyProcessResult, MyProcessTimeout, MyProcessException
from abstraqt.utils.std_helper import redirect_stderr_to_string


#########
# PATHS #
#########


script_directory = os.path.dirname(os.path.realpath(__file__))
root_directory = os.path.join(script_directory, '..', '..')

yp21_directory = os.path.join(root_directory, 'baseline_yp21', 'v8')

circuits_directory = os.path.join(script_directory, 'circuits')
circuits_csv_file = os.path.join(circuits_directory, 'circuits.csv')
results_directory = os.path.join(script_directory, 'results')

quizx_executable = os.path.join(root_directory, 'QuiZX', 'wrapper', 'target', 'release', 'wrapper')


#########
# OTHER #
#########


TOOL = Callable[[str, float, int], bool]


###################
# RUN EXPERIMENTS #
###################


class ToolException(Exception):

    def __init__(self, short: str, msg: Optional[str]=None) -> None:
        self.short = short
        if msg is None:
            self.msg = short
        else:
            self.msg = short + '; ' + msg
        super().__init__(msg)
    
    def __str__(self):
        return 'ToolException: ' + self.msg


def get_tool_name(tool: TOOL):
    tool_name = tool.__name__.replace('run_', '')
    return tool_name


def run_experiment(tool: TOOL, label: str, timeout: float, max_memory_gb: int):
    return tool(label, timeout, max_memory_gb)


def run_experiments(
        tools: List[TOOL],
        labels: List[str],
        timeout: int,
        max_memory_gb: int,
        max_processes: int,
        file_base_name: str = 'results'
):
    dfs = []

    mapper = MyProcessMap(max_processes, timeout)
    args = [
        (tool, label, timeout, max_memory_gb)
        for label in labels
        for tool in tools
    ]
    outcomes = mapper.map_iterator(
        run_experiment,
        args
    )

    pbar = tqdm(outcomes, total=len(args))
    for outcome in pbar:
        tool = get_tool_name(outcome.args[0])
        label = outcome.args[1]
        error = None
        precise = False

        pbar.set_description(f"{tool.ljust(29)},{label.ljust(21)}")

        # handle internal T/O
        if isinstance(outcome, MyProcessException) and isinstance(outcome.exception, subprocess.TimeoutExpired):
            outcome = MyProcessTimeout(outcome.f, outcome.args, outcome.elapsed_time)
        
        if isinstance(outcome, MyProcessTimeout):
            error = 'T/O'
        elif isinstance(outcome, MyProcessResult):
            precise = outcome.result
            if not isinstance(precise, bool):
                error = f'Got {type(precise).__name__} instead of bool'
        elif isinstance(outcome, MyProcessException):
            if isinstance(outcome.exception, MemoryError):
                error = 'OOM'
            elif isinstance(outcome.exception, ToolException) and outcome.exception.short == 'OOM':
                error = 'OOM'
            elif isinstance(outcome.exception, ToolException):
                error = outcome.exception.short
                # print(f'ToolException running {label} with {tool}: {str(outcome.exception)}')
            else:
                error = 'Unknown error'
                print('Unknown error:', outcome.exception)
        else:
            raise ValueError(f'Unexpected process result {type(outcome).__name__}')

        row = pd.DataFrame({
            'tool': [tool],
            'label': [label],
            'time_s': [outcome.elapsed_time],
            'time_h': [round(outcome.elapsed_time / 3600, 1)],
            'error': [error],
            'precise': [precise]
        })
        dfs.append(row)

        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(os.path.join(results_directory, f'{file_base_name}_raw.csv'))
        df = postprocess_results(df, labels)
        df.to_csv(os.path.join(results_directory, f'{file_base_name}.csv'))


def postprocess_results(df: pd.DataFrame, labels: List[str]):
    # extract runs
    df['run'] = df['label'].str.rsplit('-', n=1).str[1]
    df['label'] = df['label'].str.rsplit('-', n=1).str[0]

    # move runs to columns
    df = df.pivot(index=['tool', 'label'], columns='run')

    # add success-only times
    for i in df.columns.levels[1]:
        error_col = ('error', i)
        time_col = ('time_s', i)
        df[('time_s_suc', i)] = np.where(df[error_col].isnull(), df[time_col], np.nan)

    # compute statistics
    precise = df.loc[:, ('precise', slice(None))]
    df['precise_avg'] = precise.mean(axis=1)

    df['error_summary'] = df.loc[:, ('error', slice(None))].apply(value_counts_str, axis=1)

    time_s_suc = df.loc[:, ('time_s_suc', slice(None))]
    df['time_s_suc_min'] = time_s_suc.min(axis=1)
    df['time_s_suc_max'] = time_s_suc.max(axis=1)

    # rearrange columns
    new_column_names = []
    fronts = []
    for column in list(df.columns):
        if column[1] == '':
            new_column_names.append(column[0])
            fronts.append(column[0])
        else:
            new_column_names.append(column[0] + '_' + column[1])
    df.columns = new_column_names
    df = move_to_front(df, fronts)
    df = df.reset_index(drop=False)

    # sort by label order
    labels = [label.rsplit('-')[0] for label in labels]
    labels = filter_duplicates(labels)
    df['label'] = pd.Categorical(df['label'], categories=labels, ordered=True)
    df = df.sort_values(by=['tool', 'label'])
    df = df.reindex(range(len(df)))

    return df


#########
# TOOLS #
#########


def run_yp21_mode_1(label: str, timeout: int, max_memory_gb: int):
    return run_yp21(label, timeout, max_memory_gb, mode=1)


def run_yp21_mode_2(label: str, timeout: int, max_memory_gb: int):
    return run_yp21(label, timeout, max_memory_gb, mode=2)


def run_yp21(label: str, timeout: int, max_memory_gb: int, mode:int):
    try:
        q_file = os.path.join(circuits_directory, label + '-full.q')

        with open(q_file, 'r') as f:
            circuit_str = f.read()
    except FileNotFoundError:
        raise ToolException("Unsupported operation")

    command = [
        'java',
        '-XX:ActiveProcessorCount=1',
        '-XX:ParallelGCThreads=1',
        f'-Xmx{max_memory_gb}g',
        'Main',
        'static'
    ]
    if mode == 1:
        command += ['2', 'timed']
    else:
        command += ['5', 'timed', 'ex', '3']

    try:
        result = subprocess.run(
            command,
            timeout=timeout*0.99 - 10, # stop subprocess before parent is being stopped
            input=circuit_str.encode(),
            check=True,
            cwd=yp21_directory,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode()
        if 'OutOfMemoryError' in stderr:
            raise ToolException('OOM', stderr)
        elif 'The final abstract state is invalid' in stderr:
            raise ToolException('Internal error (invalid)', stderr)
        else:
            raise ToolException('Unknown error', stderr)

    stdout = result.stdout.decode()
    stdout = stdout.split('\n')
    # Check if YP21 can confirm last qubit is |0> (it checks nothing else)
    stdout = [o for o in stdout if 'Qubit state:' in o]
    stdout = [int(re.search(r'Qubit (\d+)', o).group(1)) for o in stdout]
    n_qubits = len(stdout) > 0

    if mode == 2:
        qasm_file = os.path.join(circuits_directory, label + '-full.qasm')
        circuit = QuantumCircuit.from_qasm_file(qasm_file)
        if circuit.count_ops().get('ccx', 0) == 0:
            raise ToolException('Unsound', 'Running in mode 2 without CCX gates yields unsound results.')
    return n_qubits


def run_quizx(label: str, timeout: int, max_memory_gb: int):
    qasm_file = os.path.join(circuits_directory, label + '-full.qasm')

    command = [quizx_executable, qasm_file]

    try:
        result = subprocess.run(
            command,
            timeout=timeout*0.99 - 10, # stop subprocess before parent is being stopped
            check=True,
            capture_output=True,
            preexec_fn=limit_virtual_memory(max_memory_gb)
        )
        stdout = result.stdout.decode()

        # Check if QuiZX can confirm last qubit is |0> (it checks nothing else)
        if 'is definitively 0' in stdout:
            return True
        elif 'NaN+NaNi' in stdout:
            raise ToolException('Internal error (nan)')
        elif 'graph was not fully reduced' in stdout:
            raise ToolException('Internal error (not fully reduced)')
        else:
            raise ToolException('Wrong result')
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode()
        if 'memory allocation' in stderr:
            raise ToolException('OOM', stderr)
        elif 'linearization error' in stderr:
            raise ToolException('Unsupported operation', stderr)
        else:
            raise ToolException('Unknown error', stderr)


def run_feynman(label: str, timeout: int, max_memory_gb: int):
    qc_full_file = os.path.join(circuits_directory, label + '-full.qc')
    qc_simplified_file = os.path.join(circuits_directory, label + '-simplified.qc')

    if not os.path.exists(qc_simplified_file):
        raise ToolException("Unsupported operation")

    command = ['feynver', qc_full_file, qc_simplified_file]

    try:
        result = subprocess.run(
            command,
            timeout=timeout*0.99 - 10, # stop subprocess before parent is being stopped
            check=True,
            capture_output=True,
            preexec_fn=limit_virtual_memory(max_memory_gb)
        )
        stdout = result.stdout.decode()
        # Check if feynman can confirm simplified ≡ full
        correct = 'Equal' in stdout
        if not correct:
            print(f'Warning: Feynman yielded wrong result for', label)
        return correct

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode()
        if 'not supported' in stderr:
            raise ToolException('Unsupported operation', stderr)
        elif 'out of memory' in stderr:
            raise ToolException('OOM', stderr)
        else:
            raise ToolException('Unknown error', stderr)


def run_qiskit_equivalence(label: str, timeout: int, max_memory_gb: int):
    try:
        full_file = os.path.join(circuits_directory, label + '-full.qasm')
        simplified_file = os.path.join(circuits_directory, label + '-simplified.qasm')

        if not os.path.exists(simplified_file):
            raise ToolException("Unsupported operation")

        try:
            # Check if Qiskit can confirm simplified ≡ full
            return circuits_equivalent(full_file, simplified_file, max_memory_gb*1e9)
        except ValueError as e:
            if 'array is too big' in str(e):
                raise ToolException('OOM', str(e))
            else:
                raise e
    except QiskitError as e:
        if 'Cannot apply operation with classical bits: measure' in str(e):
            raise ToolException('Unsupported operation')
        else:
            raise ToolException('Unknown error', e.message)


def run_qiskit_statevector(label: str, timeout: int, max_memory_gb: int):
    return run_qiskit(label, timeout, max_memory_gb, 'statevector')


def run_qiskit_extended_stabilizer(label: str, timeout: int, max_memory_gb: int):
    return run_qiskit(label, timeout, max_memory_gb, 'extended_stabilizer')


def run_qiskit(label: str, timeout: int, max_memory_gb: int, method: str):
    circuit = load_qasm(label)

    if circuit.count_ops().get('measure', 0) > 0:
        raise ToolException('Unsupported operation', 'Circuit contains internal measurement')

    # add measurement (necessary to get counts)
    with_measurement = append_measurement(circuit)

    # https://qiskit.org/ecosystem/aer/stubs/qiskit_aer.AerSimulator.html
    extended_stabilizer_simulator = AerSimulator(method=method, max_memory_mb=max_memory_gb*1000, max_parallel_threads=1)
    transpiled = transpile(with_measurement, extended_stabilizer_simulator)

    n_shots = 100
    with redirect_stderr_to_string():
        job = extended_stabilizer_simulator.run(transpiled, shots=n_shots)
        result = job.result()

    if not result.success:
        msg = 'Simulation failed with status: ' + result.status
        if 'max_memory_mb' in result.status or 'Insufficient memory' in result.status:
            raise ToolException('OOM', msg)
        raise ToolException('Unknown error', msg)

    # Check if Qiskit can confirm measurement always yields 0
    counts = result.get_counts(with_measurement)
    n_zeros = counts.get('0', 0)
    correct = n_zeros == n_shots
    if not correct:
        print(f'Warning: Qiskit yielded {n_shots - n_zeros}/{n_shots} wrong measurement results for', label)
    return correct


def append_measurement(circuit: QuantumCircuit):
    if circuit.num_clbits == 0:
        with_measurement = QuantumCircuit(circuit.num_qubits, 1)
    else:
        with_measurement = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    with_measurement.compose(circuit, inplace=True)
    with_measurement.measure(circuit.num_qubits-1, 0)
    return with_measurement


def run_abstraqt(label: str, timeout: int, max_memory_gb: int):
    circuit = load_qasm(label)
    dag = circuit_to_dag(circuit)
    all_qubits = get_qubit_indices_from_circuit(circuit)

    state = AbstractStabilizerPlus.zero_state(circuit.num_qubits)

    for node in topological_op_nodes(dag):
        qubits = get_qubit_indices(all_qubits, node)
        if node.name == 'measure':
            state = state.measure('Z', *qubits, both=True)
        else:
            op = AbstractCircuit.get_abstraqt_operation(node)
            state = state.conjugate(op, *qubits)
            state = state.compress(1)

    state = state.measure('-Z', circuit.num_qubits - 1)
    trace = state.get_trace()
    zero = IntervalArray.lift(0)
    is_zero = trace.equal_abstract_object(zero)

    return is_zero


###########
# HELPERS #
###########


def load_qasm(label: str):
    qasm_file = os.path.join(circuits_directory, label + '-full.qasm')
    circuit = QuantumCircuit.from_qasm_file(qasm_file)
    return circuit


def limit_virtual_memory(max_memory_gb: int):
    def limit():
        memory_limit = max_memory_gb * 1_000_000_000
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    return limit


########
# MAIN #
########


all_tools = [
    run_abstraqt,
    run_quizx,
    run_qiskit_extended_stabilizer,
    run_feynman,
    run_yp21_mode_1,
    run_yp21_mode_2,
    run_qiskit_statevector,
    run_qiskit_equivalence,
]
tool_name_to_tool = {get_tool_name(tool): tool for tool in all_tools}


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tool', default=None, type=str, help='Only run this tool.')
    parser.add_argument('--labels', default=None, type=str, nargs='+', help='Only run these labels.')
    parser.add_argument('--timeout', default=6*60*60, type=int, help='Timeout per circuit and per tool [seconds]')
    parser.add_argument('--max-memory-gb', default=12, type=int, help='Maximum memory [GB]')
    parser.add_argument('--max-processes', default=1, type=int, help='Maximum number of processes to run concurrently')
    parser.add_argument('--file-base-name', default='results', type=str, help='Base name for csv results file')

    args = parser.parse_args()

    if args.tool is None:
        tools = all_tools
    else:
        tools = [tool_name_to_tool[args.tool]]
    
    labels: List[str] = pd.read_csv(circuits_csv_file)['label'].to_list()
    if args.labels is not None:
        labels = args.labels

    run_experiments(tools, labels, args.timeout, args.max_memory_gb, args.max_processes, args.file_base_name)


if __name__ == '__main__':
    main()
