# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""OptSpace - low-rank matrix completion.

This module implements OptSpace (Keshavan, Oh & Montanari 2010), an algorithm
that completes a low-rank matrix from a small set of observed entries: an SVD
initialization, gradient descent on the Grassmann manifold with an exact line
search, and a least-squares re-solve for the singular values. It underpins
robust Aitchison PCA (RPCA) of sparse compositional data, where the missing
entries are the (masked) zeros of an ``rclr``-transformed table.

This implementation was adapted from the ``gemelli`` package, licensed under the
Modified BSD License:

- https://github.com/biocore/gemelli

The original ``gemelli`` algorithm is itself a Python port of Sewoong Oh's
MATLAB OptSpace code. A copy of the ``gemelli`` license is included in
``licenses/gemelli.txt``.

The default (``"cython"``) engine is a pure NumPy/SciPy implementation that
requires no compiled extension. An optional ``"numba"`` engine JIT-compiles the
fusable element-wise and reduction kernels (masked residual, masked squared-sum,
Grassmann terms, masked rank-one accumulation) while leaving the dense matrix
multiplications, the ``scipy.sparse.linalg.svds`` initialization, and the
``numpy.linalg.lstsq`` re-solve in NumPy/SciPy. The numba engine reproduces the
default engine's results to within ~1e-8.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import svds

from skbio._config import _resolve_engine

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Numba kernels: pure-array, fusable element-wise / reduction parts only.
# These mirror the default-engine NumPy expressions exactly.
# ---------------------------------------------------------------------------

if NUMBA_AVAILABLE:

    @njit(parallel=True)
    def _masked_residual_nb(obs, imputed, mask):
        """Element-wise ``(obs - imputed) * mask`` over the full matrix."""
        n, m = obs.shape
        out = np.empty((n, m), dtype=np.float64)
        for i in prange(n):
            for j in range(m):
                out[i, j] = (obs[i, j] - imputed[i, j]) * mask[i, j]
        return out

    @njit(parallel=True)
    def _masked_sq_sum_nb(recon, obs, mask):
        """Masked squared-sum reduction ``sum(((recon - obs) * mask) ** 2)``."""
        n, m = recon.shape
        total = 0.0
        for i in prange(n):
            acc = 0.0
            for j in range(m):
                r = (recon[i, j] - obs[i, j]) * mask[i, j]
                acc += r * r
            total += acc
        return total

    @njit(parallel=True)
    def _grassmann_one_nb(U, step_size, n_components):
        """First Grassmann penalty (scalar reduction)."""
        n = U.shape[0]
        k = U.shape[1]
        denom = 2.0 * step_size * n_components
        total = 0.0
        for i in prange(n):
            s = 0.0
            for j in range(k):
                s += U[i, j] * U[i, j]
            step = s / denom
            if step < 1.0:
                continue
            val = np.exp((step - 1.0) ** 2) - 1.0
            if np.isinf(val):
                continue
            total += val
        return total

    @njit(parallel=True)
    def _grassmann_two_nb(U, step_size, n_components):
        """Second Grassmann term, an ``(n, r)`` gradient contribution."""
        n = U.shape[0]
        k = U.shape[1]
        denom = 2.0 * step_size * n_components
        scale = step_size * n_components
        out = np.empty((n, k), dtype=np.float64)
        for i in prange(n):
            s = 0.0
            for j in range(k):
                s += U[i, j] * U[i, j]
            step = s / denom
            coef = 2.0 * np.exp((step - 1.0) ** 2) * (step - 1.0)
            if coef < 0.0:
                coef = 0.0
            coef = coef / scale
            for j in range(k):
                out[i, j] = U[i, j] * coef
        return out

    @njit(parallel=True)
    def _masked_rank_one_nb(manifold, x, mask):
        """Masked rank-one outer product ``((manifold @ x).T) * mask``."""
        n = x.shape[0]
        m = manifold.shape[0]
        out = np.empty((n, m), dtype=np.float64)
        for a in prange(n):
            xa = x[a]
            for b in range(m):
                out[a, b] = xa * manifold[b] * mask[a, b]
        return out


# ---------------------------------------------------------------------------
# Default (NumPy) kernels. Used when ``engine == "cython"`` (the no-extra-dep
# path; OptSpace has no Cython extension, so this is plain NumPy/SciPy).
# ---------------------------------------------------------------------------


def _masked_residual_np(obs, imputed, mask):
    """Element-wise ``(obs - imputed) * mask``."""
    return np.multiply((obs - imputed), mask)


def _masked_sq_sum_np(recon, obs, mask):
    """Masked squared-sum ``sum(((recon - obs) * mask) ** 2)``."""
    return np.sum(np.multiply((recon - obs), mask) ** 2)


def _grassmann_one_np(U, step_size, n_components):
    """First Grassmann penalty (scalar)."""
    step = np.sum(U**2, axis=1) / (2 * step_size * n_components)
    manifold = np.exp((step - 1) ** 2) - 1
    manifold[step < 1] = 0
    manifold[manifold == np.inf] = 0
    return manifold.sum()


def _grassmann_two_np(U, step_size, n_components):
    """Second Grassmann term, an ``(n, r)`` gradient contribution."""
    step = np.sum(U**2, axis=1) / (2 * step_size * n_components)
    step = 2 * np.multiply(np.exp((step - 1) ** 2), (step - 1))
    step[step < 0] = 0
    step = step.reshape(len(step), 1)
    step = np.multiply(U, np.tile(step, (1, n_components))) / (step_size * n_components)
    return step


def _masked_rank_one_np(manifold, x, mask):
    """Masked rank-one outer product ``((manifold @ x).T) * mask``."""
    x = x.reshape(1, len(x))
    manifold = manifold.reshape(len(manifold), 1)
    return np.multiply((manifold.dot(x)).T, mask)


# Per-engine kernel dispatch table. Matmuls, ``svds`` and ``lstsq`` are shared
# (NumPy/SciPy/MKL) regardless of engine; only the fusable kernels differ.
_KERNELS = {
    "cython": (
        _masked_residual_np,
        _masked_sq_sum_np,
        _grassmann_one_np,
        _grassmann_two_np,
        _masked_rank_one_np,
    ),
}
if NUMBA_AVAILABLE:
    _KERNELS["numba"] = (
        _masked_residual_nb,
        _masked_sq_sum_nb,
        _grassmann_one_nb,
        _grassmann_two_nb,
        _masked_rank_one_nb,
    )


# ---------------------------------------------------------------------------
# Orchestration: plain Python; matmuls via numpy.dot; kernels from the table.
# ---------------------------------------------------------------------------


def _svd_sort(U, S, V):
    """Sort U, S, V by descending singular value (``svds`` is not sorted)."""
    idx = np.argsort(np.diag(S))[::-1]
    S = S[idx, :][:, idx]
    U = U[:, idx]
    V = V[:, idx]
    return U, S, V


def _cost_function(U, V, S, obs, mask, step_size, rho, kernels):
    """Masked reconstruction distortion + Grassmann penalties (scalar)."""
    _, _masked_sq_sum, _grassmann_one, _, _ = kernels
    n_components = U.shape[1]
    recon = U.dot(S).dot(V.T)
    distortion = _masked_sq_sum(recon, obs, mask) / 2.0
    v_manifold = rho * _grassmann_one(V, step_size, n_components)
    u_manifold = rho * _grassmann_one(U, step_size, n_components)
    return distortion + v_manifold + u_manifold


def _gradient_decent(U, V, S, obs, mask, step_size, rho, kernels):
    """One gradient-descent update of the U and V loadings."""
    _masked_residual, _, _, _grassmann_two, _ = kernels
    n, n_components = U.shape
    m = V.shape[0]
    US = U.dot(S)
    VS = V.dot(S.T)
    imputed = US.dot(V.T)
    resid = _masked_residual(obs, imputed, mask)  # (obs - imputed) * mask
    Qu = U.T.dot(resid).dot(VS) / n
    Qv = V.T.dot(resid.T).dot(US) / m
    # (imputed - obs) * mask == -resid; reuse to avoid a second masked pass.
    neg_resid = -resid
    U_update = (
        neg_resid.dot(VS)
        + U.dot(Qu)
        + rho * _grassmann_two(U, step_size, n_components)
    )
    V_update = (
        neg_resid.T.dot(US)
        + V.dot(Qv)
        + rho * _grassmann_two(V, step_size, n_components)
    )
    return U_update, V_update


def _line_search(
    U, U_update, V, V_update, S, obs, mask, step_size, rho, kernels,
    resolution_limit=20, line=-1e-1,
):
    """Exact line search for the gradient-descent step length."""
    norm_update = norm(U_update, "fro") ** 2 + norm(V_update, "fro") ** 2
    cost = np.zeros(resolution_limit + 1)
    cost[0] = _cost_function(U, V, S, obs, mask, step_size, rho, kernels)
    for i in range(resolution_limit):
        cost[i + 1] = _cost_function(
            U + line * U_update, V + line * V_update, S, obs, mask, step_size,
            rho, kernels,
        )
        if (cost[i + 1] - cost[0]) <= 0.5 * line * norm_update:
            return line
        line = line / 2
    return line


def _singular_values(U, V, obs, mask, kernels):
    """Least-squares re-solve of the (r, r) singular-value matrix given U, V."""
    _, _, _, _, _masked_rank_one = kernels
    n_components = U.shape[1]
    C = np.ravel(U.T.dot(obs).dot(V))
    A = np.zeros((n_components * n_components, n_components * n_components))
    for i in range(n_components):
        for j in range(n_components):
            ind = j * n_components + i
            tmp = _masked_rank_one(V[:, j], U[:, i], mask)
            temp = U.T.dot(tmp).dot(V)
            A[:, ind] = np.ravel(temp)
    S = np.linalg.lstsq(A, C, rcond=1e-12)[0]
    S = S.reshape((n_components, n_components)).T
    return S


def optspace(
    mat,
    n_components=3,
    *,
    max_iterations=5,
    tol=1e-5,
    engine=None,
    validate=True,
):
    r"""Complete a low-rank matrix from its observed entries via OptSpace.

    OptSpace [1]_ recovers a low-rank matrix from a small fraction of observed
    entries. The missing entries of the input are marked as ``NaN``. The
    algorithm initializes with a truncated SVD of the masked matrix, refines the
    sample (``U``) and feature (``V``) loadings via gradient descent on the
    Grassmann manifold with an exact line search, and re-solves the singular
    values (``s``) by least squares each iteration. This is the matrix-completion
    core of robust Aitchison PCA (RPCA), where the missing entries are the
    masked zeros of an ``rclr``-transformed compositional table.

    Parameters
    ----------
    mat : array_like of shape (M, N)
        The masked matrix to complete. Missing entries must be ``NaN`` (they are
        treated as unobserved, not as zeros). Remaining entries are the observed
        values.
    n_components : int, optional
        Target rank of the completion (number of components). Must be a positive
        integer no greater than ``min(M, N) - 1``. Default is 3.
    max_iterations : int, optional
        Maximum number of gradient-descent iterations. Default is 5.
    tol : float, optional
        Early-stopping tolerance on the masked reconstruction error. Iteration
        stops once the per-entry RMSE over observed entries falls below this
        value. Default is 1e-5.
    engine : {"cython", "numba"}, optional
        Compute engine to use. ``"cython"`` (default) uses the pure NumPy/SciPy
        implementation and requires no compiled extension or optional
        dependency. ``"numba"`` JIT-compiles the fusable element-wise and
        reduction kernels and requires the optional Numba dependency; it
        reproduces the default-engine result to within ~1e-8. If not provided,
        the global default is used (see :func:`skbio.set_config`).
    validate : bool, optional
        If ``True`` (default), validate the input shape and ``n_components``.

    Returns
    -------
    U : ndarray of shape (M, n_components)
        Left (sample) loadings.
    s : ndarray of shape (n_components, n_components)
        Diagonal matrix of singular values, in descending order.
    V : ndarray of shape (N, n_components)
        Right (feature) loadings.
    distance : ndarray of shape (M, M)
        Euclidean distances between samples in the completed
        ``U @ s @ V.T`` space (robust Aitchison distances when the input is an
        ``rclr``-transformed table).

    Raises
    ------
    ValueError
        If ``n_components`` is not a positive integer, exceeds
        ``min(M, N) - 1``, if ``mat`` is not 2-D, or if ``engine`` is not a
        supported value.

    Notes
    -----
    This function performs the bare matrix completion only; it does not apply an
    ``rclr`` transform or assemble an ordination. Pass an already-masked matrix
    (``NaN`` = missing).

    The algorithm and its NumPy implementation are adapted from the ``gemelli``
    package [2]_, itself a port of Sewoong Oh's MATLAB OptSpace code.

    References
    ----------
    .. [1] Keshavan, R. H., Montanari, A., & Oh, S. (2010). Matrix completion
       from a few entries. IEEE Transactions on Information Theory, 56(6),
       2980-2998.
    .. [2] Martino, C., Morton, J. T., Marotz, C. A., et al. (2019). A novel
       sparse compositional technique reveals microbial perturbations. mSystems,
       4(1), e00016-19. (gemelli: https://github.com/biocore/gemelli)

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.ordination import optspace
    >>> rng = np.random.default_rng(0)
    >>> truth = rng.standard_normal((20, 2)) @ rng.standard_normal((2, 10))
    >>> obs = truth.copy()
    >>> obs[rng.random(obs.shape) < 0.3] = np.nan  # mask 30% of entries
    >>> U, s, V, dist = optspace(obs, n_components=2, max_iterations=50)
    >>> U.shape, s.shape, V.shape, dist.shape
    ((20, 2), (2, 2), (10, 2), (20, 20))

    """
    engine = _resolve_engine(engine, ("cython", "numba"))
    kernels = _KERNELS[engine]

    obs = np.array(mat, dtype=np.float64, copy=True)

    if validate:
        if obs.ndim != 2:
            raise ValueError("Input matrix must be two-dimensional.")
        if not isinstance(n_components, (int, np.integer)):
            raise ValueError("n_components must be an integer.")
        if n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if n_components > (min(obs.shape) - 1):
            raise ValueError(
                "n_components must be at most 1 minus the minimum shape of the "
                "input matrix."
            )
    n_components = int(n_components)

    # OptSpace treats only zeros as missing; NaN -> 0, mask the observed entries.
    obs[np.isnan(obs)] = 0
    mask = (np.abs(obs) > 0).astype(np.float64)
    n, m = obs.shape
    total_nonzeros = np.count_nonzero(mask)
    eps = total_nonzeros / np.sqrt(m * n)

    # rescale so the observed Frobenius norm matches the target rank scale
    rescal_param = np.count_nonzero(mask) * n_components
    rescal_param = np.sqrt(rescal_param / (norm(obs, "fro") ** 2))
    obs = obs * rescal_param

    # SVD initialization (shared NumPy/SciPy path for both engines)
    U, S, V = svds(obs, n_components, which="LM")
    rho = eps * n
    U = U * np.sqrt(n)
    V = (V * np.sqrt(m)).T
    S = S / eps
    S = _singular_values(U, V, obs, mask, kernels)

    recon = U.dot(S).dot(V.T)
    dist_iter = np.zeros(max_iterations + 2)
    dist_iter[0] = np.sqrt(_masked_sq_sum_np(recon, obs, mask)) / np.sqrt(
        total_nonzeros
    )

    sign = -1
    step_size = 10000
    for i in range(1, max_iterations + 1):
        U_update, V_update = _gradient_decent(
            U, V, S, obs, mask, step_size, rho, kernels
        )
        line = _line_search(
            U, U_update, V, V_update, S, obs, mask, step_size, rho, kernels
        )
        U = U - sign * line * U_update
        V = V - sign * line * V_update
        S = _singular_values(U, V, obs, mask, kernels)
        recon = U.dot(S).dot(V.T)
        dist_iter[i + 1] = np.sqrt(_masked_sq_sum_np(recon, obs, mask)) / np.sqrt(
            total_nonzeros
        )
        if dist_iter[i + 1] < tol:
            break

    S = S / rescal_param
    U, S, V = _svd_sort(U, S, V)

    # Euclidean (robust Aitchison) distances in the completed space.
    completed = U.dot(S).dot(V.T)
    diff = completed[:, np.newaxis, :] - completed[np.newaxis, :, :]
    distance = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    return U, S, V, distance
