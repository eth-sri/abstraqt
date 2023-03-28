import numpy as np


class EqualAbstractObjectNan:

    def equal_abstract_object(self, other) -> bool:
        return np.all(np.logical_or(
            self.representation == other.representation,
            np.logical_and(
                np.isnan(self.representation),
                np.isnan(other.representation)
            )
        ))
