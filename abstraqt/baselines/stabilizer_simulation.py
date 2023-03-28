import argparse
import os

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator, AerJob
from qiskit.result import Result
from qiskit.result.models import ExperimentResult

from abstraqt.applications.circuit_helper.collect_circuits import collect_circuits_helper
from abstraqt.applications.example_circuits_directory import benchmarks_directory
from abstraqt.utils import logging
from abstraqt.utils.logging.log_data import set_data_context, report_runtime, log_data
from abstraqt.utils.profiling import profile


logger = logging.getLogger(__name__)


def simulate(circuit: QuantumCircuit):
    with set_data_context(circuit_file=circuit.name):
        with report_runtime('extended_stabilizer'):
            logger.info('Simulating %s', circuit.name)

            try:
                # Create extended stabilizer method simulator
                extended_stabilizer_simulator = AerSimulator(method='extended_stabilizer')

                # appending measurements is necessary to ensure the simulation actually runs
                circuit.measure_all()

                # Transpile circuit for backend
                transpiled_circuit = transpile(circuit, extended_stabilizer_simulator)

                job = extended_stabilizer_simulator.run(transpiled_circuit, shots=1, validate=True)
                assert isinstance(job, AerJob)
                result = job.result()
                assert isinstance(result, Result)
                success = result.success
                log_data(extended_stabilizer_success=success)

                result = result.results[0]
                assert isinstance(result, ExperimentResult)
                status = result.status
                log_data(result_status=status)
                log_data(memory_error='max_memory' in status)
                log_data(qubits_error='only supports up to' in status)
            except Exception as e:
                log_data(extended_stabilizer_success=False)
                log_data(result_status=str(e))
                log_data(exception=True)
                raise


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('file', nargs='?', default=None)
    args = parser.parse_args()

    for _, circuit in collect_circuits_helper(args.file, benchmarks_directory, args.all):
        try:
            with profile(os.path.basename(circuit.name)):
                simulate(circuit)
        except Exception as e:
            logger.error('Error during verify of %s: %s', circuit.name, e)
            logger.exception(e)


if __name__ == '__main__':
    run()




