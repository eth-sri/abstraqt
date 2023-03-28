import numpy as np


class AbstractSize:

    def size_element_wise(self):
        """
        see "size"
        :returns numpy array
        """
        raise NotImplementedError()

    def size(self):
        """
        A value in [0,1] indicating how imprecise this abstract element is.
        0 indicates full precision (either bottom or "a point"), while 1 indicates complete uncertainty ("top")
        """
        sizes = self.size_element_wise()
        size = np.average(sizes)
        return size
