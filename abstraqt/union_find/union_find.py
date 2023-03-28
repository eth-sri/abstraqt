import numpy as np


class UnionFind:

    def __init__(self, n_elements: int):
        self.n_elements = n_elements

        # maps every element to its parent (and roots to themselves)
        self.parents = np.arange(0, n_elements, dtype=int)

        # maps every element to the number of elements below it (valid only for roots)
        self.sizes = np.ones(self.n_elements, dtype=int)

    def find(self, x: int):
        """

        :param x:
        :return: index of the root of x
        """
        assert 0 <= x < self.n_elements

        parent = self.parents[x]
        while x != parent:
            grand_parent = self.parents[parent]
            self.parents[x] = grand_parent  # speed up future calls
            x, parent = parent, grand_parent
        return x

    def union(self, x: int, y: int):
        """

        :param x:
        :param y:
        :return: (i,j) if we merged j into the root i. If i==j, no merging was necessary
        """
        assert 0 <= x < self.n_elements
        assert 0 <= y < self.n_elements

        x = self.find(x)
        y = self.find(y)

        if x == y:
            return x, y

        # ensure x has more children than y
        if self.sizes[x] < self.sizes[y]:
            x, y = y, x

        # make x the new root
        self.parents[y] = x

        # update the number of children of x
        self.sizes[x] += self.sizes[y]

        return x, y

    def roots(self):
        ret = set()
        for i in range(self.n_elements):
            ret.add(self.find(i))
        return np.array(list(ret))
