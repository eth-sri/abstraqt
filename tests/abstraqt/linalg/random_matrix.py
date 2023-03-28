import numpy as np


def random_linear_combinations(a: np.ndarray):
    n = a.shape[0]

    # perform row additions
    for _ in range(4 * n):
        change_rows = np.random.randint(0, 2, size=n).astype(bool)
        reference_row = np.random.randint(0, n)
        change_rows[reference_row] = False

        a[change_rows, :] += a[reference_row, :]

    a %= 2

    return a


def get_random_bool_matrices(sizes, n_tests, full_rank=False, rank=None):
    for n, m in sizes:
        for i in range(n_tests):
            np.random.seed(i)
            a = get_bool_matrix(n, m, full_rank=full_rank, rank=rank)
            yield a


def get_bool_matrix(n, m, full_rank=False, rank=None):
    if not full_rank:
        full_rank = np.random.choice([False, True])

    if full_rank or rank is not None:
        s = min(n, m)
        if rank is not None:
            assert rank >= 1
            s = min(s, rank)

        # random square matrix with full rank
        a = np.eye(s, dtype=np.uint8)

        # extend randomly
        if s < m:
            # add linearly dependent columns
            r = np.random.randint(0, 2, size=(s, m - s))
            r = (a @ r) % 2
            a = np.hstack((a, r))
        assert a.shape == (s, m)

        if s < n:
            # add linearly dependent rows
            r = np.random.randint(0, 2, size=(n - s, s))
            r = (r @ a) % 2
            a = np.vstack((a, r))

        assert a.shape == (n, m)

        # shuffle rows and columns
        np.random.shuffle(a)
        np.random.shuffle(a.T)

        a = random_linear_combinations(a)
    else:
        a = np.random.randint(0, high=2, size=(n, m), dtype=int)

    assert a.shape == (n, m)
    return a.astype(bool)
