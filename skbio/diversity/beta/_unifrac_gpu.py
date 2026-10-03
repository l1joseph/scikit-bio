# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""GPU backend detection for UniFrac (:mod:`skbio.diversity.beta._unifrac_gpu`)."""

import math

import numpy as np

UNWEIGHTED = 0
UNWEIGHTED_UNNORMALIZED = 1
WEIGHTED_NORMALIZED = 2
WEIGHTED_UNNORMALIZED = 3
GENERALIZED = 4

_backend_cache = None

# Compiled kernels, keyed by id() of the cuda-API module they were built
# against. The module itself is stored alongside so the id stays valid.
_KERNEL_CACHE = {}


def detect_gpu_backend():
    """Detect which GPU backend, if any, is usable.

    Returns
    -------
    {'cuda', 'hip', None}
        'cuda' if an NVIDIA GPU and ``numba-cuda`` are both available,
        'hip' if no NVIDIA GPU is found but an AMD GPU and
        ``numba.hip`` are both available, otherwise ``None``.

    """
    global _backend_cache
    if _backend_cache is not None:
        return _backend_cache
    try:
        from numba import cuda

        if cuda.is_available():
            _backend_cache = "cuda"
            return _backend_cache
    except ImportError:
        pass
    try:
        import numba.hip as hip

        if hip.is_available():
            _backend_cache = "hip"
            return _backend_cache
    except ImportError:
        pass
    _backend_cache = None
    return _backend_cache


def get_cuda_module():
    """Return a ``numba.cuda``-API-compatible module for the detected backend.

    Returns
    -------
    module
        ``numba.cuda`` itself on an NVIDIA backend, or ``numba.hip``
        after calling ``pose_as_cuda()`` on an AMD backend, so kernel
        code written against ``cuda.jit``/``cuda.device_array``/etc.
        runs unmodified on either.

    Raises
    ------
    ImportError
        If no usable GPU backend is detected.

    Notes
    -----
    NVIDIA support comes from the ``numba-cuda`` package, installable via
    scikit-bio's ``gpu-nvidia`` extra. AMD support requires ``numba-hip``,
    which is published only on test.pypi and must be pinned to match the
    installed ROCm version (for example
    ``numba-hip[rocm-7-0-0]==0.1.6`` for ROCm 7.0.0), so there is no
    single pip specification that works across ROCm releases and hence no
    ``gpu-amd`` extra. Note that ``hip-python`` alone is not sufficient: it
    provides low-level HIP bindings, not ``numba.hip``/``pose_as_cuda()``.
    See ``docs/superpowers/specs/2026-10-02-ssu-unifrac-numba-phase1-design.md``
    for the full install path.

    """
    backend = detect_gpu_backend()
    if backend == "cuda":
        from numba import cuda

        return cuda
    if backend == "hip":
        import numba.hip as hip

        hip.pose_as_cuda()
        return hip
    raise ImportError(
        "No usable GPU backend found (checked numba-cuda and numba.hip). "
        "Install one of these optional dependencies, or use engine='numba' "
        "for the CPU path."
    )


def _build_pair_index(n_samples):
    """Precompute condensed-index (i, j) pairs for i < j, scipy pdist order.

    ``np.triu_indices`` walks the upper triangle row by row, which is exactly
    ``scipy.spatial.distance.pdist``'s condensed order.
    """
    pair_i, pair_j = np.triu_indices(n_samples, k=1)
    return pair_i.astype(np.int32), pair_j.astype(np.int32)


def _make_unifrac_kernel(cuda):
    """Return the UniFrac pair kernel compiled against a cuda-API module.

    Shared by all 5 UniFrac methods (``UNWEIGHTED``, ``UNWEIGHTED_UNNORMALIZED``,
    ``WEIGHTED_NORMALIZED``, ``WEIGHTED_UNNORMALIZED``, ``GENERALIZED``) so that
    only one kernel is compiled and launched regardless of which method a given
    driver function dispatches.

    ``cuda.jit`` returns a fresh Dispatcher (and so pays full compilation
    cost, seconds) each time this runs, which would otherwise happen on every
    single ``*_unifrac_gpu`` call. The result is therefore memoized per
    backend module in ``_KERNEL_CACHE``; only the compiled kernel object is
    cached, never any per-call state.

    """
    cached = _KERNEL_CACHE.get(id(cuda))
    if cached is not None:
        return cached[1]

    @cuda.jit
    def _unifrac_pair_kernel(
        proportions,
        counts,
        sample_totals,
        branch_lengths,
        method,
        alpha,
        variance_adjust,
        pair_i,
        pair_j,
        out,
    ):
        idx = cuda.grid(1)
        if idx >= out.shape[0]:
            return
        i = pair_i[idx]
        j = pair_j[idx]
        n_nodes = proportions.shape[1]
        numerator = 0.0
        denominator = 0.0
        for k in range(n_nodes):
            p_u = proportions[i, k]
            p_v = proportions[j, k]
            length = branch_lengths[k]
            s = p_u + p_v
            d = abs(p_u - p_v)
            if variance_adjust:
                m = sample_totals[i] + sample_totals[j]
                mi = counts[i, k] + counts[j, k]
                vaw = math.sqrt(mi * (m - mi))
                if vaw <= 0.0:
                    continue
                s = s / vaw
                d = d / vaw
            if method == WEIGHTED_NORMALIZED or method == WEIGHTED_UNNORMALIZED:
                numerator += length * d
                denominator += length * s
            elif method == UNWEIGHTED or method == UNWEIGHTED_UNNORMALIZED:
                if s <= 0.0:
                    continue
                observed = p_u > 0.0 or p_v > 0.0
                differs = (p_u > 0.0) != (p_v > 0.0)
                if observed:
                    if variance_adjust:
                        numerator += length / vaw if differs else 0.0
                        denominator += length / vaw
                    else:
                        numerator += length if differs else 0.0
                        denominator += length
            elif method == GENERALIZED:
                if s == 0.0:
                    continue
                sum_pow = length * s**alpha
                numerator += sum_pow * (d / s)
                denominator += sum_pow
        if method == WEIGHTED_UNNORMALIZED or method == UNWEIGHTED_UNNORMALIZED:
            out[idx] = numerator
        else:
            out[idx] = 0.0 if denominator == 0.0 else numerator / denominator

    # Keep a reference to the module so its id() cannot be reused by another
    # object while this entry lives.
    _KERNEL_CACHE[id(cuda)] = (cuda, _unifrac_pair_kernel)
    return _unifrac_pair_kernel


def weighted_unifrac_gpu(
    counts, taxa, tree, normalized, variance_adjust=False, validate=True
):
    """Compute the condensed weighted UniFrac distance vector on a GPU.

    Host-side driver for the weighted UniFrac methods (``normalized`` selects
    between ``WEIGHTED_NORMALIZED`` and ``WEIGHTED_UNNORMALIZED``), built on
    the shared ``_make_unifrac_kernel`` pair kernel and ``_build_pair_index``
    helper. Requires a GPU backend detected by ``detect_gpu_backend``.

    """
    from skbio.diversity.beta._unifrac import _setup_multiple_unifrac, _get_tip_indices

    cuda = get_cuda_module()
    counts_by_node, tree_index, branch_lengths = _setup_multiple_unifrac(
        counts, taxa, tree, validate
    )
    counts_by_node = np.ascontiguousarray(counts_by_node, dtype=np.float64)
    tip_indices = _get_tip_indices(tree_index)
    sample_totals = counts_by_node[:, tip_indices].sum(axis=1)
    n_samples = counts_by_node.shape[0]
    proportions = np.divide(
        counts_by_node,
        sample_totals[:, None],
        out=np.zeros_like(counts_by_node),
        where=sample_totals[:, None] > 0,
    )
    pair_i, pair_j = _build_pair_index(n_samples)
    n_pairs = pair_i.shape[0]

    d_proportions = cuda.to_device(proportions)
    d_counts = cuda.to_device(counts_by_node)
    d_sample_totals = cuda.to_device(sample_totals)
    d_branch_lengths = cuda.to_device(branch_lengths.astype(np.float64))
    d_pair_i = cuda.to_device(pair_i)
    d_pair_j = cuda.to_device(pair_j)
    d_out = cuda.device_array(n_pairs, dtype=np.float64)

    method = WEIGHTED_NORMALIZED if normalized else WEIGHTED_UNNORMALIZED
    kernel = _make_unifrac_kernel(cuda)
    threads_per_block = 256
    blocks = (n_pairs + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](
        d_proportions,
        d_counts,
        d_sample_totals,
        d_branch_lengths,
        method,
        1.0,
        variance_adjust,
        d_pair_i,
        d_pair_j,
        d_out,
    )
    return d_out.copy_to_host()


def unweighted_unifrac_gpu(
    counts, taxa, tree, normalized, variance_adjust=False, validate=True
):
    """Compute the condensed unweighted UniFrac distance vector on a GPU.

    Host-side driver for the unweighted UniFrac methods (``normalized``
    selects between ``UNWEIGHTED`` and ``UNWEIGHTED_UNNORMALIZED``), built
    on the shared ``_make_unifrac_kernel`` pair kernel and
    ``_build_pair_index`` helper. Requires a GPU backend detected by
    ``detect_gpu_backend``.

    """
    from skbio.diversity.beta._unifrac import _setup_multiple_unifrac, _get_tip_indices

    cuda = get_cuda_module()
    counts_by_node, tree_index, branch_lengths = _setup_multiple_unifrac(
        counts, taxa, tree, validate
    )
    counts_by_node = np.ascontiguousarray(counts_by_node, dtype=np.float64)
    tip_indices = _get_tip_indices(tree_index)
    sample_totals = counts_by_node[:, tip_indices].sum(axis=1)
    n_samples = counts_by_node.shape[0]
    # The unweighted kernel branch only tests "proportions" for > 0 (presence/
    # absence), so raw counts can be passed directly in place of proportions,
    # no division needed.
    pair_i, pair_j = _build_pair_index(n_samples)
    n_pairs = pair_i.shape[0]

    d_proportions = cuda.to_device(counts_by_node)
    d_counts = cuda.to_device(counts_by_node)
    d_sample_totals = cuda.to_device(sample_totals)
    d_branch_lengths = cuda.to_device(branch_lengths.astype(np.float64))
    d_pair_i = cuda.to_device(pair_i)
    d_pair_j = cuda.to_device(pair_j)
    d_out = cuda.device_array(n_pairs, dtype=np.float64)

    method = UNWEIGHTED if normalized else UNWEIGHTED_UNNORMALIZED
    kernel = _make_unifrac_kernel(cuda)
    threads_per_block = 256
    blocks = (n_pairs + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](
        d_proportions,
        d_counts,
        d_sample_totals,
        d_branch_lengths,
        method,
        1.0,
        variance_adjust,
        d_pair_i,
        d_pair_j,
        d_out,
    )
    return d_out.copy_to_host()


def generalized_unifrac_gpu(
    counts, taxa, tree, alpha, variance_adjust=False, validate=True
):
    """Compute the condensed generalized UniFrac distance vector on a GPU.

    Host-side driver for the generalized UniFrac method (``GENERALIZED``),
    built on the shared ``_make_unifrac_kernel`` pair kernel and
    ``_build_pair_index`` helper. Requires a GPU backend detected by
    ``detect_gpu_backend``.

    """
    from skbio.diversity.beta._unifrac import _setup_multiple_unifrac, _get_tip_indices

    cuda = get_cuda_module()
    counts_by_node, tree_index, branch_lengths = _setup_multiple_unifrac(
        counts, taxa, tree, validate
    )
    counts_by_node = np.ascontiguousarray(counts_by_node, dtype=np.float64)
    tip_indices = _get_tip_indices(tree_index)
    sample_totals = counts_by_node[:, tip_indices].sum(axis=1)
    n_samples = counts_by_node.shape[0]
    proportions = np.divide(
        counts_by_node,
        sample_totals[:, None],
        out=np.zeros_like(counts_by_node),
        where=sample_totals[:, None] > 0,
    )
    pair_i, pair_j = _build_pair_index(n_samples)
    n_pairs = pair_i.shape[0]

    d_proportions = cuda.to_device(proportions)
    d_counts = cuda.to_device(counts_by_node)
    d_sample_totals = cuda.to_device(sample_totals)
    d_branch_lengths = cuda.to_device(branch_lengths.astype(np.float64))
    d_pair_i = cuda.to_device(pair_i)
    d_pair_j = cuda.to_device(pair_j)
    d_out = cuda.device_array(n_pairs, dtype=np.float64)

    kernel = _make_unifrac_kernel(cuda)
    threads_per_block = 256
    blocks = (n_pairs + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](
        d_proportions,
        d_counts,
        d_sample_totals,
        d_branch_lengths,
        GENERALIZED,
        alpha,
        variance_adjust,
        d_pair_i,
        d_pair_j,
        d_out,
    )
    return d_out.copy_to_host()
