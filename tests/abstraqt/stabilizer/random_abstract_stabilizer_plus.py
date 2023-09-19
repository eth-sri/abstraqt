import itertools

import numpy as np
from cachier import cachier
from tqdm import tqdm

from abstraqt.abstract_interpretation import AbstractComplexArray, AbstractBoolArray
from abstraqt.pauli import AbstractPauliBitRepresentation, PauliBitRepresentation
from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from abstraqt.utils.cachier_helper import default_cachier_arguments
from abstraqt.utils.my_numpy.my_numpy import random_number_generator
from tests.abstraqt.abstract_interpretation.interfaces.abstract_test_abstract_bit_pattern_array import \
    random_abstract_bit_pattern_array
from tests.abstraqt.abstract_interpretation.test_abstract_complex_array import random_abstract_complex_array
from tests.abstraqt.pauli.test_abstract_pauli_bit_representation import random_abstract_pauli_bit_representation
from tests.abstraqt.pauli.test_pauli_reference import get_random_pauli_bit_representation
from tests.test_config import default_random_repetitions, skip_expensive


def random_concrete_aaronson_gottesman(n_summands: int, n_bits: int):
    coefficients = np.random.random(n_summands) + np.random.random(n_summands) * 1j
    coefficients = AbstractComplexArray.lift(coefficients)

    paulis = np.random.randint(0, 4, size=(n_summands, n_bits), dtype=np.uint8)
    paulis = AbstractPauliBitRepresentation.lift(paulis)

    inner_products = np.random.randint(0, 2, size=(n_summands, n_bits), dtype=np.uint8)
    inner_products = AbstractBoolArray.lift(inner_products)

    stabilizers = get_random_pauli_bit_representation(n_bits)
    stabilizers = PauliBitRepresentation(stabilizers)

    return AbstractStabilizerPlus(coefficients, paulis, inner_products, stabilizers)


def random_abstract_aaronson_gottesman(n_summands: int, n_bits: int, stabilizers_of_zero=False, non_bottom=False):
    if not non_bottom:
        # even if bottom is allowed, frequently avoid bottom (otherwise, we will see bottom all the time)
        non_bottom = np.random.choice([True, False])

    coefficients = random_abstract_complex_array((n_summands,), non_bottom=non_bottom)
    paulis = random_abstract_pauli_bit_representation((n_summands, n_bits), non_bottom=non_bottom)
    inner_products = random_abstract_bit_pattern_array(AbstractBoolArray, (n_summands, n_bits), non_bottom=non_bottom)
    if stabilizers_of_zero:
        stabilizers = PauliBitRepresentation.stabilizer_of_zero(n_bits)
    else:
        stabilizers = get_random_pauli_bit_representation(n_bits)
        stabilizers = PauliBitRepresentation(stabilizers)

    ret = AbstractStabilizerPlus(coefficients, paulis, inner_products, stabilizers)
    return ret


default_single_qubit_gates = ['H', 'S']


def random_operations(n_bits: int, n_operations: int, single_qubit_gates=default_single_qubit_gates):
    for _ in range(n_operations):
        if random_number_generator.choice([0, 1]) == 0 or n_bits == 1:
            op = random_number_generator.choice(single_qubit_gates)
            yield op, [random_number_generator.choice(n_bits)]
        else:
            # find bits to use
            i = None
            j = None
            while i == j:
                i = random_number_generator.choice(n_bits)
                j = random_number_generator.choice(n_bits)

            yield 'CNOT', [i, j]


@cachier(**default_cachier_arguments)
def generate_random_abstract_aaronson_gottesman(n_bits: int):
    assert n_bits > 1

    a = AbstractStabilizerPlus.zero_state(n_bits)
    n_operations = int(np.power(n_bits, 2))

    for op, qubits in random_operations(n_bits, n_operations):
        # apply operation
        a = a.conjugate(op, *qubits)
    return a


def random_aaronson_gottesmans(point, max_summands=3, max_bits=3, do_repeats=False, non_bottom=False, label=None):
    if skip_expensive:
        max_summands = min(max_summands, 2)
        max_bits = min(max_bits, 2)

    # prepare iterations via tqdm
    t = tqdm(
        itertools.product(
            range(1, max_summands),
            range(1, max_bits),
            range(default_random_repetitions)
        ),
        desc=label,
        total=(max_summands - 1) * (max_bits - 1) * default_random_repetitions
    )
    # iterate
    for n_summands, n_bits, seed in t:
        d = f"{n_summands}/{n_bits}/{seed}"
        if label:
            d = f"{label}({d})"
        t.set_description(d)

        np.random.seed(seed)
        if point:
            g = random_concrete_aaronson_gottesman(n_summands, n_bits)
        else:
            g = random_abstract_aaronson_gottesman(n_summands, n_bits, non_bottom=non_bottom)

        if do_repeats:
            repeats = np.random.randint(0, 2, size=n_summands) + 1
            g.repeats = repeats

        yield seed, g
