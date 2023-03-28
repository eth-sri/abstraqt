import numpy as np


class AbstractEquals:

    def equal_abstract_object_element_wise(self, other):
        other = self.lift(other)
        return np.isclose(self.representation, other.representation)

    def equal_abstract_object(self, other) -> bool:
        """
        True if the abstract objects self and other are the same
        """
        element_wise = self.equal_abstract_object_element_wise(other)
        return np.all(element_wise)

    #####################
    # TO BE IMPLEMENTED #
    #####################

    @staticmethod
    def lift(x):
        raise NotImplementedError()
