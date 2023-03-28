from typing import Union, Dict

import numpy as np
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGNode

from abstraqt.linalg.common_matrices import matrices_all_dict
from abstraqt.applications.circuit_helper.gate_helper import get_matrix_from_gate
from abstraqt.utils import logging
from abstraqt.utils.hash_helper import HashableNumpyWrapper

Payload = Union[str, DAGNode, Gate, np.ndarray]

logger = logging.getLogger(__name__)


class GateSet:

    def __init__(self):
        self.d: Dict[HashableNumpyWrapper, Payload] = {}

    def add(self, item: Payload):
        matrix = self._get_matrix(item)
        # logger.slow('Adding %s with hash %s', item, hash(matrix))
        self.d[matrix] = item

    def remove(self, item: Payload):
        matrix = self._get_matrix(item)
        del self.d[matrix]

    def __getitem__(self, item: Payload):
        matrix = self._get_matrix(item)
        return self.d[matrix]

    def __contains__(self, item: Payload):
        matrix = self._get_matrix(item)
        # logger.slow('Checking containment for item %s with hash %s', item, hash(matrix))
        return matrix in self.d

    def get_hash(self, item: Payload):
        matrix = self._get_matrix(item)
        ret = hash(matrix)
        return ret

    @staticmethod
    def _get_matrix(item: Payload):
        if isinstance(item, np.ndarray):
            matrix = item
        elif isinstance(item, (DAGNode, Gate)):
            matrix = get_matrix_from_gate(item)
        else:
            assert isinstance(item, str), f'Found {type(item)}'
            matrix = matrices_all_dict[item]
        matrix = HashableNumpyWrapper(matrix)
        return matrix


matrices_all_dict_hashed = GateSet()
for letter in matrices_all_dict.keys():
    matrices_all_dict_hashed.add(letter)
