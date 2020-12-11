from typing import Optional, Literal, Tuple, Union

import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import sklearn.utils as sku
import sklearn.metrics as skm


def my_get_mean_var(X, axis: Union[Literal['gene', 'cell'], int]):
    if not isinstance(axis, int):
        axis = 0 if axis == 'gene' else 1

    if isinstance(X, ss.spmatrix):  # same as sparse.issparse()
        X_copy = X.copy()
        X_copy.data **= 2
        mean = X.mean(axis=axis).A
        var = X_copy.mean(axis=axis).A - mean ** 2
        var *= X.shape[axis] / (X.shape[axis] - 1)
    else:
        mean = np.mean(X, axis=axis)
        var = np.var(X, axis=axis, ddof=1)  # a little overhead (mean counted twice, but it's ok.)
    return mean, var


def pca_sparse(X, n_component=0, solver='arpack', random_state=None):
    if n_component == 0:
        n_component = min(X.shape) // 2 + 1

    random_state = sku.check_random_state(random_state)
    np.random.set_state(random_state.get_state())
    random_init = np.random.rand(np.min(X.shape))

    mu = X.mean(0).A.flatten()[None, :]

#     mdot = mmat = mu.dot
#     mhdot = mhmat = mu.T.dot
#     Xdot = Xmat = X.dot
#     XHdot = XHmat = X.T.conj().dot
#     ones = np.ones(X.shape[0])[None, :].dot
    matvec = matmat = lambda x: X @ x - mu @ x
    rmatvec = rmatmat = lambda x: X.T.conj() @ x - mu.T @ np.ones(X.shape[0])[np.newaxis, :] @ x

#     def matvec(x):
#         return Xdot(x) - mdot(x)

#     def matmat(x):
#         return Xmat(x) - mmat(x)

#     def rmatvec(x):
#         return XHdot(x) - mhdot(ones(x))

#     def rmatmat(x):
#         return XHmat(x) - mhmat(ones(x))

    XL = ssl.LinearOperator(
        matvec=matvec,
        dtype=X.dtype,
        matmat=matmat,
        shape=X.shape,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
    )

    u, s, v = ssl.svds(XL, solver=solver, k=n_component, v0=random_init)
    ux, v = sku.extmath.svd_flip(u, v)
    idx = np.argsort(-s)
    v = v[idx, :]

    X_pca = (u * s)[:, idx]
    ev = s[idx] ** 2 / (X.shape[0] - 1)

    total_var = my_get_mean_var(X, axis='cell')[1].sum()
    ev_ratio = ev / total_var

    output = {
        'X_pca': X_pca,
        'variance': ev,
        'variance_ratio': ev_ratio,
        'components': v,
    }
    return X_pca, ev_ratio

def magic(X, ka=3, embedding=pca_sparse):

    def compute_affinity_matrix(X):
        sortedX = np.argsort(X, axis=1)
        sigma = [X[cell_ind, i] for cell_ind, i in enumerate(sortedX[:, ka])]
        indptr = np.arange(0, (X.shape[0] + 1) * ka, ka)
        indices = sortedX[:, :3 * ka].flatten()
        data = np.array([X[cell_ind, i] for cell_ind, i in enumerate(sortedX[:, :3 * ka])])
        A = ss.csr_matrix((np.exp(-(data.T / sigma).T ** 2).flatten(), indices, indptr), shape=X.shape).tolil()
        A = A + A.T
        A.setdiag(2)
        if A.max() > 2:
            raise ValueError('WRONG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return A
        

    processed_X, variance_ratio = embedding(X)
    dist = skm.pairwise.euclidean_distances(processed_X)  # this thing in sklearn actually can deal with sparse matrix, surprisingly.
    A = compute_affinity_matrix(dist)
    M = (A / A.sum(axis=1)).A
    old_M = M
    t = 1
    while True:
        # https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/
        new_M = old_M * M
        t += 1
        if skm.r2_score(new_M, M) < 0.05:
            return t, new_M @ M
        else:
            old_M = new_M

    return new_M @ X


def gen_test_data(shape=(5, 10)):
    test_X = np.random.normal(0, 1, shape)
    test_X[test_X < 0.5] = 0
    return ss.csr_matrix(test_X)

def add_group_coverages(
    X,
    groupby: str,
    min_reps: int,
    max_reps: int,
    min_cells: int,
    max_cells: int,
    sampling_ratio: float
):

    if X.shape[0] <= min_cells * min_reps:
        for group in X.groupby[groupby]:
            if min_cells / sampling_ratio <= len(group):
                pass
