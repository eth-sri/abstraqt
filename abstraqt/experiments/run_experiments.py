import os
import time
import subprocess
import resource
import re
import argparse
import multiprocessing
from typing import Callable, Optional
import traceback

from tqdm import tqdm
import pandas as pd
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.providers.aer import AerSimulator
from qiskit import transpile, QiskitError
from qiskit.result import Result

from abstraqt.applications.circuit_helper.qiskit_helper import topological_op_nodes
from abstraqt.applications.abstract_circuit import AbstractCircuit
from abstraqt.applications.circuit_helper.qubit_helper import get_qubit_indices, get_qubit_indices_from_circuit
from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from abstraqt.utils.timeout import TimedOutException


script_directory = os.path.dirname(os.path.realpath(__file__))
root_directory = os.path.join(script_directory, '..', '..')

yp21_directory = os.path.join(root_directory, 'baseline_yp21', 'v8')

circuits_directory = os.path.join(script_directory, 'circuits')
circuits_csv_file = os.path.join(circuits_directory, 'circuits.csv')
results_directory = os.path.join(script_directory, 'results')
tools_file = os.path.join(results_directory, 'tool_names.csv')

quizx_executable = os.path.join(root_directory, 'QuiZX', 'wrapper', 'target', 'release', 'wrapper')

timeout = 6*60*60 # 6h
max_memory_gb = 12


def get_circuits_df():
    circuits = pd.read_csv(circuits_csv_file)['label'].to_frame()
    return circuits


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


def run_tool(f: Callable[[str], bool]):
    f_label = f.__name__

    circuits = get_circuits_df()

    progress_bar = tqdm(circuits.iterrows(), total=len(circuits))
    for index, row in progress_bar:
        label = row['label']
        assert isinstance(label, str)
        progress_bar.set_description(label)

        precise = float('nan')
        start_time = time.time()
        try:
            precise = f(label)
            precise = float(precise)
            error = None
        except (subprocess.TimeoutExpired, TimedOutException):
            error = 'T/O'
        except ToolException as e:
            error = e.short
            print(f'ToolException running {label} with {f_label}: {str(e)}')

        end_time = time.time()
        elapsed_time = end_time - start_time

        circuits.loc[index, 'time'] = elapsed_time
        circuits.loc[index, 'error'] = error
        circuits.loc[index, 'precise'] = precise

    return circuits


def get_tool_name(tool: Callable[[str], bool]):
    tool_name = tool.__name__.replace('run_', '')
    return tool_name


def run_tool_and_record(tool: Callable[[str], bool]):
    tool_name = get_tool_name(tool)
    print('Running', tool_name)
    results = run_tool(tool)

    results_file = os.path.join(results_directory, f'results_{tool_name}.csv')
    results.to_csv(results_file, index=False)


#########
# TOOLS #
#########


def run_yp21_mode_1(label: str):
    return run_yp21(label, mode=1)


def run_yp21_mode_2(label: str):
    return run_yp21(label, mode=2)


def run_yp21(label: str, mode:int):
    try:
        q_file = os.path.join(circuits_directory, label + '-full.q')

        with open(q_file, 'r') as f:
            circuit_str = f.read()
    except FileNotFoundError:
        raise ToolException("Unsupported operation")

    command = ['java', f'-Xmx{max_memory_gb}g', 'Main', 'static']
    if mode == 1:
        command += ['2', 'timed']
    else:
        command += ['5', 'timed', 'ex', '3']

    try:
        result = subprocess.run(
            command,
            timeout=timeout,
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
            raise ToolException('Internal error', stderr)
        else:
            raise ToolException('Unknown error', stderr)

    stdout = result.stdout.decode()
    stdout = stdout.split('\n')
    stdout = [o for o in stdout if 'Qubit state:' in o]
    stdout = [int(re.search(r'Qubit (\d+)', o).group(1)) for o in stdout]
    n_qubits = len(stdout) > 0

    if mode == 2:
        qasm_file = os.path.join(circuits_directory, label + '-full.qasm')
        circuit = QuantumCircuit.from_qasm_file(qasm_file)
        if circuit.count_ops().get('ccx', 0) == 0:
            raise ToolException('Unsound', 'Running in mode 2 without CCX gates yields unsound results.')
    return n_qubits


def run_quizx(label: str):
    qasm_file = os.path.join(circuits_directory, label + '-full.qasm')

    command = [quizx_executable, qasm_file]

    try:
        result = subprocess.run(
            command,
            timeout=timeout,
            check=True,
            capture_output=True,
            preexec_fn=limit_virtual_memory
        )
        stdout = result.stdout.decode()

        if 'is definitively 0' in stdout:
            return True
        elif 'NaN+NaNi' in stdout:
            raise ToolException('Internal error')
        elif 'graph was not fully reduced' in stdout:
            raise ToolException('Internal error')
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


def run_qiskit_statevector(label: str):
    return run_qiskit(label, 'statevector')


def run_qiskit_extended_stabilizer(label: str):
    return run_qiskit(label, 'extended_stabilizer')


def run_qiskit(label: str, method: str):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=_run_qiskit, args=(label,return_dict, method))
    p.start()

    # wait a maximum of timeout seconds
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimedOutException()
    
    if p.exitcode != 0:
        raise ToolException('Unknown error')
    if 'result' not in return_dict:
        raise ToolException('Unknown error', 'No result')
    result = return_dict['result']
    if not result.success:
        msg = 'Simulation failed with status: ' + result.status
        if 'max_memory_mb' in result.status or 'Insufficient memory' in result.status:
            raise ToolException('OOM', msg)
        elif 'internal measurement' in result.status:
            raise ToolException('Unsupported operation', msg)
        else:
            raise ToolException('Unknown error', msg)
    else:
        precise = return_dict['ret']
        return precise


def append_measurement(circuit: QuantumCircuit):
    if circuit.num_clbits == 0:
        with_measurement = QuantumCircuit(circuit.num_qubits, 1)
    else:
        with_measurement = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    with_measurement.compose(circuit, inplace=True)
    with_measurement.measure(circuit.num_qubits-1, 0)
    return with_measurement


def _run_qiskit(label: str, return_dict, method: str):
    circuit = load_qasm(label)

    if circuit.count_ops().get('measure', 0) > 0:
        return_dict['result'] = Result(method, 0, 0, 0, False, None, status='Circuit contains internal measurement')
        return

    # add measurement (necessary to get counts)
    with_measurement = append_measurement(circuit)

    extended_stabilizer_simulator = AerSimulator(method=method, max_memory_mb=max_memory_gb*1000)
    transpiled = transpile(with_measurement, extended_stabilizer_simulator)

    n_shots = 100
    job = extended_stabilizer_simulator.run(transpiled, shots=n_shots)
    result = job.result()
    return_dict['result'] = result

    if not result.success:
        print('Qiskit error:', result.status)
        return

    counts = result.get_counts(with_measurement)
    n_zeros = counts.get('0', 0)
    correct = n_zeros == n_shots
    if not correct:
        print(f'Warning: Qiskit yielded {n_shots - n_zeros}/{n_shots} wrong measurement results for', label)
    return_dict['ret'] = correct


def run_abstraqt(label: str):
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

    n_found = 0
    for i in range(circuit.num_qubits):
        state = state.measure('-Z', i)
        trace = state.get_trace()
        if trace.is_point():
            n_found += 1

    return n_found > 0


###########
# HELPERS #
###########


def load_qasm(label: str):
    qasm_file = q_file = os.path.join(circuits_directory, label + '-full.qasm')
    circuit = QuantumCircuit.from_qasm_file(qasm_file)
    return circuit


def limit_virtual_memory():
    memory_limit = max_memory_gb * 1_000_000_000
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))


########
# MAIN #
########


all_tools = [
    run_abstraqt,
    run_quizx,
    run_qiskit_extended_stabilizer,
    run_yp21_mode_1,
    run_yp21_mode_2,
    run_qiskit_statevector,
]
tool_name_to_tool = {get_tool_name(tool): tool for tool in all_tools}


def run_all_tools():
    tools = {
        'tool_name': []
    }

    for tool in all_tools:
        run_tool_and_record(tool)

        tool_name = get_tool_name(tool)
        tools['tool_name'].append(tool_name)

        df = pd.DataFrame(tools)
        df.to_csv(tools_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tool', default=None, type=str)
    parser.add_argument('--label', default=None, type=str)
    args = parser.parse_args()

    if args.label is None:
        if args.tool is None:
            run_all_tools()
        else:
            tool = tool_name_to_tool[args.tool]
            run_tool_and_record(tool)
    else:
        tools = all_tools
        if args.tool is not None:
            tools = [tool_name_to_tool[args.tool]]
        for tool in tools:
            try:
                precise = tool(args.label)
                print(tool.__name__ + ':', precise)
            except BaseException as e:
                print(tool.__name__ + ' ' + type(e).__name__ + ' exception: ' + str(e))
                traceback.print_exc()
                if isinstance(e, subprocess.CalledProcessError):
                    print(e.stderr.decode())


if __name__ == '__main__':
    main()
