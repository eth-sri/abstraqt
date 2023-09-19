import itertools
import os
import time
from tempfile import NamedTemporaryFile
from typing import List, Union

import numpy as np

import abstraqt.utils.logging as logging
from abstraqt.utils.cache_dir import cache_dir
from abstraqt.utils.function_renamer import rename_function
from abstraqt.utils.my_numpy.my_numba import my_njit

logger = logging.getLogger(__name__)

lookup_tables_dir = os.path.join(cache_dir, 'lookup_tables')
os.makedirs(lookup_tables_dir, exist_ok=True)
logger.debug('Storing lookup table caches to %s', lookup_tables_dir)
always_compute_table = False


class LookupTable:

    def __init__(self, reference_implementation, bound_per_input: List[int], label: str):
        self.logger = logging.getLogger(__name__ + '.' + label)

        max_encoded_value = int(np.prod(bound_per_input))
        encoded_dtype = _get_dtype_from_bound(max_encoded_value)

        self.reference_implementation = reference_implementation
        self.bound_per_input = bound_per_input
        # warning: encoding might result in 0 (but this is not an issue because we never use these 0 values)
        self.bound_per_input_encoded = [np.array(b, dtype=encoded_dtype) for b in self.bound_per_input]
        self.label = label
        self.input_dtypes = [_get_dtype_from_bound(b) for b in self.bound_per_input]
        reference_output = self._get_reference_output()
        self.output_dtype = reference_output.dtype
        if reference_output.shape == () or reference_output.shape == (1,):
            self.output_shape = ()
        else:
            self.output_shape = reference_output.shape

        self.logger.verbose('Reference implementation: %s', reference_implementation)
        self.logger.verbose('Reference output: %s', reference_output)
        self.logger.verbose('Output shape: %s', self.output_shape)
        self.logger.verbose('Output dtype: %s', self.output_dtype)

        self.table = self._prepare_table()

        # prepare lookup
        table = self.table
        output_shape = self.output_shape
        output_dtype = self.output_dtype

        @my_njit
        def actual_lookup(encoded_index):
            encoded_index = np.asarray(encoded_index, encoded_dtype)
            flattened = encoded_index.flatten()
            ret = table[flattened]
            ret = np.asarray(ret, dtype=output_dtype)
            ret = np.reshape(ret, encoded_index.shape + output_shape)
            return ret

        if self.n_args == 1:
            @my_njit
            def f(arg):
                return actual_lookup(arg)
        elif self.n_args == 2:
            bound_for_arg_1 = self.bound_per_input_encoded[1]

            @my_njit
            def f(arg0, arg1):
                args_encoded = _encode_2(arg0, arg1, bound_for_arg_1)
                return actual_lookup(args_encoded)
        elif self.n_args == 3:
            bound_for_arg_1 = self.bound_per_input_encoded[1]
            bound_for_arg_2 = self.bound_per_input_encoded[2]

            @my_njit
            def f(arg0, arg1, arg2):
                args_encoded = _encode_3(arg0, arg1, arg2, bound_for_arg_1, bound_for_arg_2)
                return actual_lookup(args_encoded)
        else:
            assert False, "Wrong number of arguments"

        f = rename_function(reference_implementation.__name__ + '_lookup')(f)
        self.lookup = f

    @property
    def n_args(self):
        return len(self.bound_per_input)

    def _call_reference_implementation(self, *args):
        # transforms arguments to the appropriate type
        args = [np.asarray(arg, dtype=dtype) for arg, dtype in zip(args, self.input_dtypes)]
        ret = self.reference_implementation(*args)
        return ret

    def _get_reference_output(self):
        zeros = tuple(0 for _ in range(self.n_args))
        reference_output = self._call_reference_implementation(*zeros)
        reference_output = np.asarray(reference_output)
        self.logger.verbose('Obtained reference output %s', reference_output)
        return reference_output

    def _prepare_table(self):
        f = os.path.join(lookup_tables_dir, self.label + '.npy')
        if always_compute_table or not os.path.isfile(f):
            start = time.perf_counter()
            table = self._compute_table()

            # save .npy
            np_save_atomic(f, table, self.label)
            self.logger.verbose('Saved table to %s', f)

            # save .txt
            f_txt = os.path.join(lookup_tables_dir, self.label + '.txt')
            np_save_str_atomic(f_txt, table, self.label, tuple(self.bound_per_input) + self.output_shape)
            self.logger.verbose('Saved textual representation of table to %s', f_txt)

            self.logger.debug('Computed table in %s seconds', time.perf_counter() - start)
        else:
            self.logger.verbose('Loading table from %s', f)
        table = np.load(f)

        return table

    def _all_arguments(self):
        ranges = [range(n_options) for n_options in self.bound_per_input]
        for args in itertools.product(*ranges):
            args = tuple(args)
            yield args

    def _compute_table(self):
        self.logger.info('Computing table %s...', self.label)
        n_entries = np.prod(self.bound_per_input)
        shape = (n_entries,) + self.output_shape
        dtype = self.output_dtype
        table = np.empty(shape, dtype=dtype)

        for args in self._all_arguments():
            result = self._call_reference_implementation(*args)
            encoded_args = self._encode(*args)
            self.logger.slow('Computed table entry for arguments %s (encoded as %s): %s', args, encoded_args, result)
            table[encoded_args, ...] = result
        return table

    def _encode(self, *args: np.ndarray):
        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return _encode_2(args[0], args[1], self.bound_per_input_encoded[1])
        elif len(args) == 3:
            return _encode_3(args[0], args[1], args[2], self.bound_per_input_encoded[1],
                             self.bound_per_input_encoded[2])
        else:
            raise NotImplementedError()

    def lookup(self, *args: np.ndarray):
        # this function is overwritten by the constructor. This is needed to enable numba optimizations on the lookup
        raise NotImplementedError()

    def __call__(self, *args):
        return self.lookup(*args)


@my_njit
def _encode_2(arg0: np.ndarray, arg1: np.ndarray, bound_for_arg_1: np.ndarray):
    ret = arg0 * bound_for_arg_1 + arg1
    return ret


@my_njit
def _encode_3(arg0: np.ndarray, arg1: np.ndarray, arg2: np.ndarray, bound_for_arg_1: np.ndarray,
              bound_for_arg_2: np.ndarray):
    ret = arg0 * bound_for_arg_1 * bound_for_arg_2 + bound_for_arg_2 * arg1 + arg2
    return ret


def _get_dtype_from_bound(bound: Union[int, np.ndarray]):
    """
    Returns an optimal dtype assuming values are bounded by bound
    """
    bound = int(bound)

    type_per_bits = {
        1: bool,
        8: np.uint8,
        16: np.uint16,
        32: np.uint32,
        64: np.uint64
    }
    best = min([n for n in type_per_bits.keys() if (1 << n) >= bound])
    return type_per_bits[best]


def np_save_atomic(file_path: str, x: np.ndarray, label=None):
    with NamedTemporaryFile(dir=lookup_tables_dir, prefix='tmp-' + label + '-', suffix='.npy', delete=False) as tmp_file:
        tmp_file_name = tmp_file.name

        np.save(tmp_file_name, x)
        # atomically replace (avoid concurrency issues)
        os.replace(tmp_file_name, file_path)


def np_save_str_atomic(file_path: str, x: np.ndarray, label=None, reshape_txt_to=None):
    if reshape_txt_to is not None:
        x = np.reshape(x, newshape=reshape_txt_to)

    with NamedTemporaryFile(dir=lookup_tables_dir, prefix='tmp-' + label + '-', suffix='.txt', delete=False) as tmp_file:
        tmp_file_name = tmp_file.name

        with np.printoptions(threshold=np.inf):
            x = np.array2string(x)

        with open(tmp_file_name, "w") as text_file:
            text_file.write(x)
        # atomically replace (avoid concurrency issues)
        os.replace(tmp_file_name, file_path)
