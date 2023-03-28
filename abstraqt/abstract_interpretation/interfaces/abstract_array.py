from abc import ABC

import numpy as np

from abstraqt.utils.array.dimensions.representation_wrapper import RepresentationWrapper
from abstraqt.utils.array.dimensions.shaping import Shaping
from abstraqt.utils.array.element_wise_str import ElementWiseStr
from abstraqt.utils.numpy.lift_to_numpy_array import lift_to_numpy_array
from .mixins.abstract_equals import AbstractEquals
from .mixins.abstract_lift_join import AbstractLiftJoin
from .mixins.abstract_size import AbstractSize
from .mixins.abstract_superset import AbstractSuperSet
from .mixins.corners import Corners
from .mixins.update_where import UpdateWhere


class AbstractArray(
    RepresentationWrapper,
    Shaping,
    ElementWiseStr,
    AbstractLiftJoin,
    AbstractSuperSet,
    AbstractSize,
    Corners,
    AbstractEquals,
    UpdateWhere,
    ABC
):

    ################
    # CONSTRUCTORS #
    ################

    def __init__(self, representation: np.ndarray):
        representation = lift_to_numpy_array(representation)
        assert isinstance(representation, np.ndarray)
        self.representation = representation

    #############
    # CANONICAL #
    #############

    def to_canonical(self):
        """
        Switch to a canonical representation for this abstract element
        Returns self
        """
        pass

    def is_bottom(self):
        raise NotImplementedError()

    def is_point(self):
        raise NotImplementedError()

    # upper bound (exclusive) on representation
    bound_on_representation: int = None
