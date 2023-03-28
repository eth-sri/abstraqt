import numpy as np

from abstraqt.linalg.common_matrices import get_n_qubits
from abstraqt.pauli import PauliBitRepresentation, AbstractPauliBitRepresentation
from abstraqt.pauli.abstract_pauli_bit_representation import combine_prefactor_and_bare_pauli__representation
from abstraqt.pauli.pauli_bit_representation import conjugate_letters_list
from tests.abstraqt.abstract_interpretation.interfaces.abstract_test_abstract_bit_pattern_array import \
    AbstractTestAbstractBitPatternArrayWrapper
from tests.test_config import default_random_repetitions, default_shapes_list


def random_abstract_pauli_bit_representation(shape, non_bottom=False):
    start = 1 if non_bottom else 0
    bound = 1 << 4

    prefactor = np.random.randint(start, bound, size=shape, dtype=np.uint8)
    bare = np.random.randint(start, bound, size=shape, dtype=np.uint8)
    ret = combine_prefactor_and_bare_pauli__representation(prefactor, bare)

    ret = AbstractPauliBitRepresentation(ret)
    return ret


class TestAbstractPauliBitRepresentation(
    AbstractTestAbstractBitPatternArrayWrapper.AbstractTestAbstractBitPatternArray):

    def __init__(self, *args, **kwargs):
        super().__init__(AbstractPauliBitRepresentation, *args, **kwargs)

    def test_str(self):
        top = AbstractPauliBitRepresentation.top((1,))
        s = str(top)
        self.assertIsInstance(s, str)
        self.assertEqual("['{1,i,-,-i}{I,X,Z,Y}']", s)

    def test_commutes(self):
        self.check_operator(AbstractPauliBitRepresentation.commutes, concrete_op=PauliBitRepresentation.commutes,
                            n_args=2)

    def test_conjugate_h(self):
        top = AbstractPauliBitRepresentation.top((1,))
        result = top.conjugate_with_H(0)
        self.assert_equal_abstract_object(result, top)

        X = AbstractPauliBitRepresentation.lift(PauliBitRepresentation.from_string('X'))
        Z = AbstractPauliBitRepresentation.lift(PauliBitRepresentation.from_string('Z'))
        result = X.conjugate_with_H(0)
        self.assert_equal_abstract_object(result, Z)

    def test_conjugate(self):
        for letter in conjugate_letters_list:
            function_name = 'conjugate_with_' + letter
            abstract = getattr(AbstractPauliBitRepresentation, function_name)
            concrete = getattr(PauliBitRepresentation, function_name)

            n_qubits_for_op = get_n_qubits(letter)

            for shape in default_shapes_list:
                if len(shape) < 1 or shape[-1] < n_qubits_for_op:
                    continue

                for seed in range(default_random_repetitions):
                    self.check_operator_with_args(abstract, concrete, 1, shape, seed, more_args=range(n_qubits_for_op))
