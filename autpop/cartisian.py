from itertools import accumulate, repeat, chain
import re
import numpy as np
import itertools


def dstack_product(arrays):
    return np.dstack(
        np.meshgrid(*arrays, indexing='ij')
    ).reshape(-1, len(arrays))


def cartesian_product_transpose_pp(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty((la, *map(len, arrays)), dtype=dtype)
    idx = slice(None), *itertools.repeat(None, la)
    for i, a in enumerate(arrays):
        arr[i, ...] = a[idx[:la-i]]
    return arr.reshape(la, -1).T


def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def cartesian_product_transpose(arrays):
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)

    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T


def cartesian_product_pp(arrays):
    la = len(arrays)
    L = *map(len, arrays), la
    dtype = np.result_type(*arrays)
    arr = np.empty(L, dtype=dtype)
    arrs = *accumulate(chain((arr,), repeat(0, la-1)), np.ndarray.__getitem__),
    idx = slice(None), *itertools.repeat(None, la-1)
    for i in range(la-1, 0, -1):
        arrs[i][..., i] = arrays[i][idx[:la-i]]
        arrs[i-1][1:] = arrs[i]
    arr[..., 0] = arrays[0][idx]
    return arr.reshape(-1, la)


def cartesian_product_itertools(arrays):
    return np.array(list(itertools.product(*arrays)))


# from https://stackoverflow.com/a/1235363/577088
def cartesian_product_recursive(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian_product_recursive(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def generate_all_possible_gens(loci_n):
    r = []
    for i in range(loci_n+1):
        for j in range(loci_n+1-i):
            r.append((i, j, loci_n-i-j))
    S = (loci_n + 1) * (loci_n + 2) / 2
    assert len(r) == S
    return r


if __name__ == "__main__":
    a = generate_all_possible_gens(3)
    b = generate_all_possible_gens(1)
    c = generate_all_possible_gens(2)
    d = generate_all_possible_gens(2)

    P = cartesian_product_transpose_pp((a, b, c, d))
    print(P)
