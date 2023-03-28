import pickle
from operator import __mul__
from typing import Sequence, Union

import numpy as np
from abstraqt.union_find.union_find import UnionFind

from abstraqt.abstract_interpretation import AbstractComplexArray, IntervalArray
from abstraqt.abstract_interpretation.abstract_bool_array import \
    dot__abstract_bool_array_representation, solution_space_mod_2_abstract__representation, \
    xor__abstract_bool_array_representation, AbstractBoolArray
from abstraqt.abstract_interpretation.abstract_int2_array import \
    AbstractInt2Array, add__abstract_int2_array_representation, to_int2__abstract_bool_array_representation
from abstraqt.abstract_interpretation.interfaces.abstract_bit_pattern_array import \
    join_abstract_bit_pattern_array__representation
from abstraqt.abstract_interpretation.interfaces.mixins.abstract_lift_join import AbstractLiftJoin
from abstraqt.abstract_interpretation.interfaces.mixins.abstract_superset import AbstractSuperSet
from abstraqt.pauli import PauliBitRepresentation
from abstraqt.pauli.abstract_pauli_bit_representation import get_bare_pauli_concatenated__abstract_representation, \
    get_prefactors_aggregated__abstract_pauli_bit_representation_representation, pauli_mul__abstract_representation, \
    AbstractPauliBitRepresentation, \
    select_abstract__pauli_bit_representation_representation, prefactor_and_bare__abstract_pauli_bit_representation
from abstraqt.pauli.pauli_bit_representation import pauli_mul, get_bare_pauli_concatenated__representation, \
    select__pauli_bit_representation_representation, get_prefactors_aggregated__pauli_bit_representation_representation
from abstraqt.pauli.pauli_reference import decompose_into_pauli_basis
from abstraqt.utils import logging
from abstraqt.utils.array.dimensions.shaping import pairing
from abstraqt.utils.hash_helper import my_numpy_hash
from abstraqt.utils.iterator_helper import get_unique
from abstraqt.utils.numpy import matrix_product
from abstraqt.utils.numpy.my_numpy import map_to_consecutive, ravel_multi_index
from abstraqt.utils.string_helper import dict_to_string

logger = logging.getLogger(__name__)


class AbstractStabilizerPlus(AbstractSuperSet, AbstractLiftJoin):

    ################
    # CONSTRUCTORS #
    ################

    def __init__(
            self,
            coefficients: AbstractComplexArray,
            paulis: AbstractPauliBitRepresentation,
            inner_products: AbstractBoolArray,
            stabilizers: PauliBitRepresentation,
            position_in_sum: np.ndarray = None,
            repeats: np.ndarray = None
    ):
        """
        position_in_sum: indicates the position of every summand in every sum
        """

        self.coefficients = coefficients
        self.paulis = paulis
        self.inner_products = inner_products
        self.stabilizers = stabilizers

        if position_in_sum is None:
            self.position_in_sum = np.expand_dims(np.arange(self.n_summands), axis=1)
        else:
            self.position_in_sum = position_in_sum

        if repeats is None:
            self.repeats = np.zeros(self.n_summands, dtype=np.double)
        else:
            self.repeats = repeats

        self.check_invariants()

    def check_invariants(self):
        assert isinstance(self.coefficients, AbstractComplexArray)
        assert isinstance(self.paulis, AbstractPauliBitRepresentation)
        assert isinstance(self.inner_products, AbstractBoolArray)
        assert isinstance(self.stabilizers, PauliBitRepresentation)
        assert isinstance(self.position_in_sum, np.ndarray)
        assert isinstance(self.repeats, np.ndarray)

        assert self.position_in_sum.dtype == int
        assert self.repeats.dtype == np.double

        assert assert_equal(self.coefficients.shape, (self.n_summands,))
        assert assert_equal(self.paulis.shape, (self.n_summands, self.n_bits))
        assert assert_equal(self.inner_products.shape, (self.n_summands, self.n_bits))
        assert assert_equal(self.stabilizers.shape, (self.n_bits, self.n_bits))
        assert assert_equal(len(self.position_in_sum.shape), 2)
        assert assert_equal(self.position_in_sum.shape[0], self.n_summands)
        assert assert_equal(self.repeats.shape, (self.n_summands,))

    @staticmethod
    def zero_state(n_bits: int):
        coefficients = AbstractComplexArray.ones((1,))
        paulis = AbstractPauliBitRepresentation.identity(1, n_bits)
        inner_products = AbstractBoolArray.zeros((1, n_bits))
        stabilizers = PauliBitRepresentation.stabilizer_of_zero(n_bits)
        return AbstractStabilizerPlus(coefficients, paulis, inner_products, stabilizers)

    @staticmethod
    def bottom(shape: Sequence[int]):
        n_bits, n_summands = shape
        coefficients = AbstractComplexArray.bottom((n_summands,))
        paulis = AbstractPauliBitRepresentation.bottom((n_summands, n_bits))
        inner_products = AbstractBoolArray.bottom((n_summands, n_bits))
        stabilizers = PauliBitRepresentation.stabilizer_of_zero(n_bits)
        return AbstractStabilizerPlus(coefficients, paulis, inner_products, stabilizers)

    def bottom_like(self):
        return self.bottom((self.n_bits, self.n_summands))

    @staticmethod
    def any_computational_basis_state(n_bits: int):
        ret = AbstractStabilizerPlus.zero_state(n_bits)
        ret.inner_products = AbstractBoolArray.top((1, n_bits))
        return ret

    ##############
    # PROPERTIES #
    ##############

    @property
    def n_bits(self):
        return self.stabilizers.shape[0]

    @property
    def n_summands(self):
        return self.coefficients.shape[0]

    @property
    def history_length(self):
        return self.position_in_sum.shape[1]

    ##############
    # OPERATIONS #
    ##############

    def conjugate(self, op: Union[str, np.ndarray], *qubits):
        logger.slow('Applying %s to qubits %s', op, qubits)
        if isinstance(op, str) and op == 'I':
            return self

        ret = None
        if isinstance(op, str):
            ret = self.conjugate_paulis(op, *qubits)

        if ret is None:
            ret = self.conjugate_decomposed(op, *qubits)

        assert isinstance(ret, AbstractStabilizerPlus)

        return ret

    def conjugate_paulis(self, op: str, *qubits: int):
        paulis = self.paulis.conjugate_with(op, *qubits)
        if paulis.is_bottom():
            logger.slow('Paulis could not be conjugated')
            return None

        stabilizers = self.stabilizers.conjugate_with(op, *qubits)
        if stabilizers.encodes_error():
            logger.slow('Stabilizers could not be conjugated')
            return None

        logger.slow('Conjugated %s successfully', op)
        return AbstractStabilizerPlus(
            self.coefficients,
            paulis,
            self.inner_products,
            stabilizers,
            self.position_in_sum,
            self.repeats
        )

    def conjugate_decomposed(self, op: Union[str, np.ndarray], *qubits):
        factors, decomposed_partial = decompose_into_pauli_basis(op)
        d = factors.shape[0]

        factors = AbstractComplexArray.lift(factors)
        decomposed_partial = PauliBitRepresentation(decomposed_partial)
        decomposed = decomposed_partial.pad_by_identities(self.n_bits, *qubits)
        assert assert_equal(decomposed.shape, (d, self.n_bits))

        # multiply coefficients
        coefficients = f_combinations(__mul__, factors, self.coefficients, factors.conjugate())
        assert assert_equal(coefficients.shape, (d, self.n_summands, d))

        # update paulis
        paulis = f_combinations(__mul__, decomposed, self.paulis, decomposed, ignore_last_n_dimensions=1)
        assert assert_equal(paulis.shape, (d, self.n_summands, d, self.n_bits))

        # update inner products
        inner_products = self.inner_products.expand_dims((0, 2))
        inner_products = inner_products.broadcast_to((d, self.n_summands, 1, self.n_bits))

        anti_commutes = ~decomposed.commutes(self.stabilizers, all_pairs=True)
        assert assert_equal(anti_commutes.shape, (d, self.n_bits))
        anti_commutes = np.expand_dims(anti_commutes, (0, 1))
        anti_commutes = AbstractBoolArray.lift(anti_commutes)
        assert assert_equal(anti_commutes.shape, (1, 1, d, self.n_bits))

        inner_products = inner_products ^ anti_commutes
        assert assert_equal(inner_products.shape, (d, self.n_summands, d, self.n_bits))

        # update position in sum
        position_in_sum = np.empty((d, self.n_summands, d, self.history_length + 2), dtype=int)
        position_in_sum[:, :, :, 0] = np.expand_dims(np.arange(d), axis=(1, 2))
        position_in_sum[:, :, :, 1:-1] = np.expand_dims(self.position_in_sum, axis=(0, 2))
        position_in_sum[:, :, :, -1] = np.expand_dims(np.arange(d), axis=(0, 1))

        # update repeats
        repeats = np.tile(self.repeats, reps=(d, 1, d))

        # flatten
        n_summands = np.prod(coefficients.shape)
        coefficients = coefficients.reshape((n_summands,))
        paulis = paulis.reshape((n_summands, self.n_bits))
        inner_products = inner_products.reshape((n_summands, self.n_bits))
        position_in_sum = position_in_sum.reshape((n_summands, self.history_length + 2))
        repeats = repeats.flatten()

        # return result
        ret = AbstractStabilizerPlus(
            coefficients,
            paulis,
            inner_products,
            self.stabilizers,
            position_in_sum,
            repeats
        )
        return ret

    def measure(self, measurement_basis: str, *qubits: int, both=False, reset=False):
        if both or reset:
            measure0 = self._measure(measurement_basis, *qubits)
            measure1 = self._measure('-' + measurement_basis, *qubits)

            if reset:
                assert measurement_basis == 'Z'
                # reset
                measure1 = measure1.conjugate('X', *qubits)

            ret = measure0.join(measure1)
        else:
            ret = self._measure(measurement_basis, *qubits)

        must_discard = ret.coefficients.equal_abstract_object_element_wise(0)
        ret.discard_summands(must_discard)

        return ret

    def _measure(self, measurement_basis: str, *qubits: int):
        if self.is_bottom():
            return self.bottom_like()

        measurement = PauliBitRepresentation.from_string(measurement_basis)
        measurement = measurement.pad_by_identities(self.n_bits, *qubits)

        commute = measurement.commutes(self.stabilizers)

        if np.all(commute):
            measurement_abstract = AbstractPauliBitRepresentation.lift(measurement)
            measurement_commutes = self.paulis.commutes(measurement_abstract)

            x = self.stabilizers.solve(measurement)
            assert x is not None

            prefactor_selected = prefactor_on_select(x, self.inner_products.representation,
                                                     self.stabilizers.representation)
            prefactor_measurement = measurement.get_prefactors(aggregate=True)
            measurement_generated = AbstractInt2Array(prefactor_selected) == AbstractInt2Array.lift(
                prefactor_measurement)
            assert isinstance(measurement_generated, AbstractBoolArray)

            keep = measurement_commutes & measurement_generated
            assert isinstance(keep, AbstractBoolArray)
            discard = ~keep

            zero = AbstractComplexArray.zeros(self.coefficients.shape)
            coefficients = self.coefficients.copy().update_where(discard, zero)
            ret = AbstractStabilizerPlus(
                coefficients,
                self.paulis,
                self.inner_products,
                self.stabilizers,
                self.position_in_sum,
                self.repeats
            )
            return ret
        else:
            # ensure unique anti-commuting
            indices = np.flatnonzero(~commute)
            first = indices[0]
            others = indices[1:]
            # update stabilizers
            stabilizers_rep = self.stabilizers.representation.copy()
            stabilizers_rep[others, :] = pauli_mul(stabilizers_rep[others, :], stabilizers_rep[first, :])
            # update inner products
            inner_products_rep = self.inner_products.representation.copy()
            inner_products_rep[..., others] = xor__abstract_bool_array_representation(
                inner_products_rep[..., others],
                inner_products_rep[..., [first]]
            )

            # update paulis when anti-commuting
            paulis = self.paulis.copy()
            new_paulis = paulis * AbstractPauliBitRepresentation.lift(stabilizers_rep[first, :])
            inner_product_prefactors = self.inner_products[..., first].to_abstract_pauli_bit_representation()
            new_paulis.representation[..., 0] = pauli_mul__abstract_representation(
                new_paulis.representation[..., 0],
                inner_product_prefactors.representation
            )

            paulis_commute = self.paulis.commutes(measurement)
            paulis.update_where(~paulis_commute, new_paulis)

            # replace anti-commuting stabilizer by basis
            stabilizers_rep[first] = measurement.representation
            # update inner product for replaced stabilizer
            inner_products_rep[..., first] = AbstractBoolArray.zero_0D__representation

            # add factor of 1/2
            coefficients = self.coefficients * AbstractComplexArray.lift(1/2)

            ret = AbstractStabilizerPlus(
                coefficients,
                paulis,
                AbstractBoolArray(inner_products_rep),
                PauliBitRepresentation(stabilizers_rep),
                self.position_in_sum,
                self.repeats
            )
            return ret

    def discard_summands(self, to_discard):
        keep = ~to_discard
        coefficients = AbstractComplexArray(self.coefficients.representation[keep])
        paulis = AbstractPauliBitRepresentation(self.paulis.representation[keep, :])
        inner_products = AbstractBoolArray(self.inner_products.representation[keep, :])
        position_in_sum = self.position_in_sum[keep, :]
        repeats = self.repeats[keep]
        return AbstractStabilizerPlus(
            coefficients,
            paulis,
            inner_products,
            self.stabilizers,
            position_in_sum,
            repeats
        )

    def get_trace(self):
        if self.is_bottom():
            return IntervalArray.bottom(())

        # zero unless updated
        ret = AbstractComplexArray.zeros(self.n_summands)

        # check if commute (then should change to non-zero, as pauli can be generated)
        stabilizers = AbstractPauliBitRepresentation.lift(self.stabilizers)
        commutes = self.paulis.commutes(stabilizers, all_pairs=True)
        all_commute = commutes.all(axis=-1)

        # compute values to use for update
        traces = get_trace(self.stabilizers.representation, self.paulis.representation,
                           self.inner_products.representation)

        prefactor = traces.exponent_base_i()
        update_coefficients = self.coefficients * prefactor

        ret.update_where(all_commute, update_coefficients)

        ret = ret * AbstractComplexArray.lift_log_of_real(self.repeats)
        with np.errstate(over='ignore', under='ignore'):  # overflows push us to infinity, which is sound
            ret = ret.real()
            ret = ret.sum()

        return ret

    ############
    # COMPRESS #
    ############

    def move_prefactor_into_coefficient(self):
        pauli_prefactors, paulis_bare = prefactor_and_bare__abstract_pauli_bit_representation(self.paulis.representation)
        paulis_prefactors = AbstractInt2Array(pauli_prefactors).exponent_base_i()
        paulis_bare = AbstractPauliBitRepresentation(paulis_bare)

        coefficients = self.coefficients * paulis_prefactors

        return AbstractStabilizerPlus(
            coefficients,
            paulis_bare,
            self.inner_products,
            self.stabilizers,
            self.position_in_sum,
            self.repeats
        )

    def compress(self, target_n_summands: int):
        if self.n_summands <= target_n_summands:
            return self
        else:
            return self.compress_full(target_n_summands)

    def compress_full(self, target_n_summands: int):
        assert target_n_summands == 1

        # new data structures
        new_coefficients = self.coefficients[[0]].copy()
        new_paulis = self.paulis[[0], :].copy()
        new_inner_products = self.inner_products[[0], :].copy()
        new_repeats = self.repeats[[0]].copy()

        for i in range(1, self.n_summands):
            new_coefficients = new_coefficients.join(self.coefficients[[i]])
            new_paulis = new_paulis.join(self.paulis[[i], :])
            new_inner_products = new_inner_products.join(self.inner_products[[i], :])
            new_repeats = np.logaddexp(new_repeats, self.repeats[[i]])

        return AbstractStabilizerPlus(
            new_coefficients,
            new_paulis,
            new_inner_products,
            self.stabilizers,
            repeats=new_repeats
        )

    def compress_best_pairwise(self, target_n_summands: int):
        if target_n_summands >= self.n_summands:
            return self

        self = self.move_prefactor_into_coefficient()

        # equals[i,j,k] => paulis[i, k] == paulis[j, k]
        equals = np.expand_dims(self.paulis.representation, axis=1) == np.expand_dims(self.paulis.representation, axis=0)
        # n_equals[i,j] => number of k with paulis[i,k] == paulis[j,k]
        n_equals = np.sum(equals, axis=2)

        # union until we have saved enough
        sorted_flattened_indices = np.argsort(n_equals, axis=None)

        # new data structures
        coefficients = self.coefficients.copy()
        paulis = self.paulis.copy()
        inner_products = self.inner_products.copy()
        repeats = self.repeats.copy()

        uf = UnionFind(self.n_summands)

        sorted_flattened_indices_offset = 0
        n_to_flatten = self.n_summands - target_n_summands

        while n_to_flatten > 0:
            flattened_index = sorted_flattened_indices[sorted_flattened_indices_offset]
            sorted_flattened_indices_offset += 1

            i, j = flattened_index // self.n_summands, flattened_index % self.n_summands
            if i == j:
                # no merging is necessary
                continue

            i, j = uf.union(i, j)
            if i == j:
                # no merging was done
                continue

            paulis_i = paulis[i, :]
            paulis_j = paulis[j, :]
            inner_products_i = inner_products[i, :]
            inner_products_j = inner_products[j, :]

            same_paulis = paulis_i == paulis_j
            same_inner_products = inner_products_i == inner_products_j
            same = same_paulis & same_inner_products
            assert isinstance(same, AbstractBoolArray)

            paulis[i, :] = paulis_i.join(paulis_j)
            inner_products[i, :] = inner_products_i.join(inner_products_j)

            if np.all(same.equal_abstract_object_element_wise(1)):
                coefficients[i] = coefficients[i] + coefficients[j]
            else:
                coefficients[i] = coefficients[i].join(coefficients[j])
                with np.errstate(under='ignore'):  # ignore underflows when they happen (may lead to imprecision)
                    repeats[i] = np.logaddexp(repeats[i], repeats[j])

            n_to_flatten -= 1

        # compress move the union-ed data to a new data structure
        active = uf.roots()
        coefficients = coefficients[active].copy()
        paulis = paulis[active, :].copy()
        inner_products = inner_products[active, :].copy()
        position_in_sum = self.position_in_sum[active, :].copy()
        repeats = repeats[active].copy()

        return AbstractStabilizerPlus(coefficients, paulis, inner_products, self.stabilizers, position_in_sum, repeats)

    def compress_best_summand(self, target_n_summands: int):
        if target_n_summands >= self.n_summands:
            return self
        else:
            n_options = self.position_in_sum.shape[1]
            assert n_options > 0

            best = None
            best_size = None

            for i in range(n_options):
                option = self.compress_with_position(i)
                option_size = option.size()

                if best is None or option_size < best_size:
                    best = option
                    best_size = option_size

            logger.verbose("Compressed summands from %s to %s", self.n_summands, best.n_summands)

            # compress further until target is reached
            ret = best.compress(target_n_summands)
            return ret

    def compress_with_position(self, dimension_in_sum: int):
        other_positions = np.hstack((
            self.position_in_sum[:, :dimension_in_sum],
            self.position_in_sum[:, dimension_in_sum + 1:]
        ))
        dims = np.max(other_positions, axis=0) + 1

        # example flattened_positions: [0, 1, 3, 1, 0]
        flattened_positions = ravel_multi_index(other_positions.T, dims)
        assert assert_equal(flattened_positions.shape, (self.n_summands,))

        n_groups, indices = map_to_consecutive(flattened_positions)
        # example indices: [0, 1, 2, 1, 0]
        assert n_groups <= self.n_summands
        assert assert_equal(indices.shape, (self.n_summands,))

        coefficients = AbstractComplexArray.bottom((n_groups,))
        paulis = AbstractPauliBitRepresentation.bottom((n_groups, self.n_bits))
        inner_products = AbstractBoolArray.bottom((n_groups, self.n_bits))
        position_in_sum = np.empty((n_groups, self.history_length - 1), dtype=int)
        repeats = np.full(n_groups, -np.inf, dtype=np.double)

        if not self.is_bottom():
            for i, p in enumerate(indices):
                same_paulis = paulis[p, :] == self.paulis[i]
                same_inner_products = inner_products[p, :] == self.inner_products[i]
                same = same_paulis & same_inner_products
                assert isinstance(same, AbstractBoolArray)

                paulis[p, :] = paulis[p, :].join(self.paulis[i])
                inner_products[p, :] = inner_products[p, :].join(self.inner_products[i])
                position_in_sum[p] = other_positions[i, :]

                if np.all(same.equal_abstract_object_element_wise(1)):
                    coefficients[p] = coefficients[p] + self.coefficients[i]
                else:
                    coefficients[p] = coefficients[p].join(self.coefficients[i])
                    repeats[p] = np.logaddexp(repeats[p], self.repeats[i])

        return AbstractStabilizerPlus(coefficients, paulis, inner_products, self.stabilizers, position_in_sum, repeats)

    ###########
    # CORNERS #
    ###########

    def expand_repeats(self):
        repeats = np.exp(self.repeats).astype(int)
        coefficients = self.coefficients.repeat(repeats, axis=0)
        paulis = self.paulis.repeat(repeats, axis=0)
        inner_products = self.inner_products.repeat(repeats, axis=0)
        position_in_sum = np.repeat(self.position_in_sum, repeats, axis=0)
        return AbstractStabilizerPlus(
            coefficients,
            paulis,
            inner_products,
            self.stabilizers,
            position_in_sum
        )

    def get_corners(self):
        self_ = self.expand_repeats()
        for coefficients in self_.coefficients.get_corners(numpy_array=False):
            for paulis in self_.paulis.get_corners():
                paulis = AbstractPauliBitRepresentation.lift(paulis)
                for inner_products in self_.inner_products.get_corners():
                    inner_products = AbstractBoolArray.lift(inner_products)
                    yield AbstractStabilizerPlus(
                        coefficients,
                        paulis,
                        inner_products,
                        self_.stabilizers,
                        self_.position_in_sum
                    )

    def get_densities(self):
        size = 1 << self.n_bits

        stabilizers = self.stabilizers.to_matrices()

        for corner in self.get_corners():
            paulis = get_unique(corner.paulis.get_corners())
            paulis = paulis.to_matrices()
            assert assert_equal(paulis.shape, (self.n_summands, size, size))

            coefficients = get_unique(corner.coefficients.get_corners())
            inner_products = get_unique(corner.inner_products.get_corners())

            ret = np.zeros((size, size), dtype=complex)

            # prepare data (may fail in case of abstraction)
            eye = np.eye(size)

            # compute sum
            for index in np.ndindex(self.n_summands):
                coefficient = coefficients[index]
                pauli = paulis[index + (slice(None), slice(None))]
                prod = inner_products[index + (slice(None),)]

                factors = [eye + np.power(-1, p.representation) * s for p, s in zip(prod, stabilizers)]
                prod = matrix_product(factors) / (1 << self.n_bits)
                summand = coefficient * pauli @ prod

                assert assert_equal(summand.shape, (size, size))
                ret += summand

            yield ret

    ########
    # JOIN #
    ########

    def move_stabilizer_prefactors(self):
        # separate bare from prefactors
        bare_stabilizers = self.stabilizers.get_bare_pauli()
        prefactor_stabilizers = self.stabilizers.get_prefactors(aggregate=True)
        assert np.all(prefactor_stabilizers % 2 == 0)

        # update inner products
        inner_products = self.inner_products + AbstractBoolArray.lift(prefactor_stabilizers // 2)

        return AbstractStabilizerPlus(
            self.coefficients,
            self.paulis,
            inner_products,
            bare_stabilizers,
            self.position_in_sum,
            self.repeats
        )

    def join(self, other):
        assert isinstance(other, AbstractStabilizerPlus)

        self = self.move_stabilizer_prefactors()
        other = other.move_stabilizer_prefactors()

        coefficients = self.coefficients.join(other.coefficients)
        paulis = self.paulis.join(other.paulis)
        inner_products = self.inner_products.join(other.inner_products)

        if not np.all(self.stabilizers.representation == other.stabilizers.representation):
            raise ValueError(f'Joining conflicting stabilizers with {self.stabilizers} and {other.stabilizers}')
        if not np.all(self.position_in_sum == other.position_in_sum):
            raise ValueError(
                f'Joining conflicting positions in sum with {self.position_in_sum} and {other.position_in_sum}')
        if not np.all(self.repeats == other.repeats):
            raise ValueError(f'Joining conflicting repeats with {self.repeats} and {other.repeats}')

        return AbstractStabilizerPlus(
            coefficients,
            paulis,
            inner_products,
            self.stabilizers,
            self.position_in_sum,
            self.repeats
        )

    ########
    # SIZE #
    ########

    def size(self):
        c = self.coefficients.size()
        p = self.paulis.size()
        i = self.inner_products.size()
        return (c + p + i) / 3

    def is_bottom(self):
        c = self.coefficients.is_bottom()
        p = self.paulis.is_bottom()
        i = self.inner_products.is_bottom()
        return c or p or i

    def is_point(self):
        c = self.coefficients.is_point()
        p = self.paulis.is_point()
        i = self.inner_products.is_point()
        return c and p and i

    ############
    # SUPERSET #
    ############

    def is_super_set_of(self, other):
        assert isinstance(other, AbstractStabilizerPlus)
        if other.is_bottom():
            return True

        self = self.move_stabilizer_prefactors()
        other = other.move_stabilizer_prefactors()

        if not self.coefficients.is_super_set_of(other.coefficients):
            return False
        if not self.paulis.is_super_set_of(other.paulis):
            return False
        if not self.inner_products.is_super_set_of(other.inner_products):
            return False
        return self.equal_concrete_aspects(other)

    ########
    # HASH #
    ########

    def __hash__(self):
        return hash((
            self.coefficients.stable_representation_hash(),
            self.paulis.stable_representation_hash(),
            self.inner_products.stable_representation_hash(),
            self.stabilizers.stable_representation_hash(),
            my_numpy_hash(self.position_in_sum),
            my_numpy_hash(self.repeats)
        ))

    def __eq__(self, other):
        if not isinstance(other, AbstractStabilizerPlus):
            return False

        if not self.coefficients.equal_abstract_object(other.coefficients):
            return False
        if not self.paulis.equal_abstract_object(other.paulis):
            return False
        if not self.inner_products.equal_abstract_object(other.inner_products):
            return False
        return self.equal_concrete_aspects(other)

    def equal_concrete_aspects(self, other):
        assert isinstance(other, AbstractStabilizerPlus)

        if not np.all(self.stabilizers.representation == other.stabilizers.representation):
            return False
        if not np.all(self.position_in_sum == other.position_in_sum):
            return False
        if not np.all(self.repeats == other.repeats):
            return False
        return True

    ##########
    # STRING #
    ##########

    def __str__(self):
        return dict_to_string({
            'coefficients': self.coefficients,
            'paulis': self.paulis,
            'inner_products': self.inner_products,
            'stabilizers': self.stabilizers,
            'position_in_sum': self.position_in_sum,
            'repeats': self.repeats
        })

    def __repr__(self):
        return str(self)

    ########
    # SAVE #
    ########

    def save_to_file(self, file_path: str):
        with open(file_path, 'wb') as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load_from_file(file_path: str):
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def assert_equal(actual, expected):
    assert actual == expected, f'Got {actual} instead of {expected}'
    return True


def f_combinations(f, a, b, c, ignore_last_n_dimensions=0):
    a, b = pairing(a, b, ignore_last_n_dimensions=ignore_last_n_dimensions)
    tmp = f(a, b)

    tmp, c = pairing(tmp, c, ignore_last_n_dimensions=ignore_last_n_dimensions)

    ret = f(tmp, c)

    return ret


#########
# TRACE #
#########


def get_trace(q: np.ndarray, p: np.ndarray, b: np.ndarray):
    """
    return abstract element s such that

    i^s = trace(  P Π_j 1/2 (I + (-1)^(b_j) Q_j)  )

    - q is concrete with shape (n, n)
    - p is abstract with shape (..., n)
    - b is abstract with shape (..., n)
    """
    shape = p.shape[:-1]
    ret = np.zeros(shape, dtype=np.uint8)

    for i in np.ndindex(shape):
        r = __get_trace_single(q, p[i + (slice(None),)], b[i + (slice(None),)])
        ret[i] = r if r is not None else AbstractInt2Array.zero_0D__representation

    return AbstractInt2Array(ret)


def __get_trace_single(q: np.ndarray, p: np.ndarray, b: np.ndarray):
    # Return an abstract element representing x^T @ b + x^T @ prefactor(q) + prefactor(p)
    # where x ranges over vector satisfying bare_pauli( Π_j (q_j)^(x_j) ) = bare_pauli(p)

    bare_q = get_bare_pauli_concatenated__representation(q)
    bare_p = get_bare_pauli_concatenated__abstract_representation(p)

    x, kernel = solution_space_mod_2_abstract__representation(bare_q, bare_p)

    if x is None:
        return None

    x = AbstractBoolArray.lift_concrete_bit_pattern_array(x)
    for k in range(kernel.shape[1]):
        kernel_lifted = AbstractBoolArray.lift_concrete_bit_pattern_array(kernel[:, k])
        x = join_abstract_bit_pattern_array__representation(x, kernel_lifted)

    prefactor = prefactor_on_select__abstract(x, b, q)

    prefactor_p = get_prefactors_aggregated__abstract_pauli_bit_representation_representation(p)

    ret = add__abstract_int2_array_representation(prefactor, prefactor_p)

    return ret


def prefactor_on_select(x: np.ndarray, b: np.ndarray, q: np.ndarray):
    """
    return an abstract element representing:

    b @ x + prefactor(select(q, x)) % 4

    - x: concrete bool array
    - b: abstract bool array
    - q: concrete pauli bit representation
    """
    selected = select__pauli_bit_representation_representation(q, x)
    prefactor = get_prefactors_aggregated__pauli_bit_representation_representation(selected)
    prefactor = AbstractInt2Array.lift_concrete_bit_pattern_array(prefactor)

    xb = dot__abstract_bool_array_representation(b, AbstractBoolArray.lift_concrete_bit_pattern_array(x))
    xb = to_int2__abstract_bool_array_representation(xb, times_two=True)

    prefactor = add__abstract_int2_array_representation(prefactor, xb)
    return prefactor


def prefactor_on_select__abstract(x: np.ndarray, b: np.ndarray, q: np.ndarray):
    """
    Analogous to prefactor_on_select, but:

    - x: abstract bool array
    - b: abstract bool array
    - q: concrete pauli bit representation
    """
    selected = select_abstract__pauli_bit_representation_representation(q, x)
    prefactor = get_prefactors_aggregated__abstract_pauli_bit_representation_representation(selected)

    xb = dot__abstract_bool_array_representation(b, x)
    xb = to_int2__abstract_bool_array_representation(xb, times_two=True)

    prefactor = add__abstract_int2_array_representation(prefactor, xb)
    return prefactor


# def closest_index_pairs(paulis_representation: np.ndarray):
#     n_summands = paulis_representation.shape[0]
#
#     # sort by sums (a primitive hash which maps similar matrices to similar values)
#     sums = np.sum(paulis_representation, axis=1)
#     sums_sorted_indices = np.argsort(sums)
#
#     # count number of equal elements
#     n_to_compare = 20
#     n_equals = np.full((n_summands, n_to_compare), np.iinfo(int).max, dtype=int)
#
#     for i in range(n_summands):
#         ip = sums_sorted_indices[i]
#         for j in range(1, n_to_compare):
#             if i+j < n_summands:
#                 jp = sums_sorted_indices[i+j]
#                 n_equals[i, j] = np.sum(paulis_representation[ip, :] == paulis_representation[jp, :])
#
#     n_equals_sorted_indices = np.argsort(n_equals, axis=None)
#     i_originals = n_equals_sorted_indices // n_to_compare
#     j_originals = n_equals_sorted_indices % n_to_compare
#     ok = i_originals + j_originals < n_summands
#
#     sums_sorted_indices = sums_sorted_indices[ok]
#     i_originals = i_originals[ok]
#     j_originals = j_originals[ok]
#
#     ips = sums_sorted_indices[i_originals]
#     jps = sums_sorted_indices[i_originals + j_originals]
#     ret = np.stack((ips, jps), axis=-1)
#
#     return ret
